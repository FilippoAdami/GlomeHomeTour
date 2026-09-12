// GlomeHomeTour Backend: 2DGS HIP Forward Rasterization Kernel
// Target: AMD RDNA4 (Radeon RX 9060 XT, ROCm 7.x, gfx1200)
// Wave32 SIMD, 16x16 tiles, LDS staging of tile-binned primitives.
//
// Direct-radiance formulation (Huang et al., SIGGRAPH 2024): splats carry
// view-independent SH-DC radiance that is alpha-composited straight into the
// image. No deferred BRDF pass.
//
// Each tile walks ONLY the primitives binned to it (tile_ranges), not the
// global splat list.

#include <hip/hip_runtime.h>
#include <torch/types.h>
#include <vector>

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define LDS_CHUNK BLOCK_SIZE
#define ALPHA_MIN (1.0f / 255.0f)
#define ALPHA_MAX 0.99f
#define T_EPS 1e-4f

__global__ void __launch_bounds__(BLOCK_SIZE, 2)
Rasterize2DGSForwardKernel(
    const int H,
    const int W,
    const int grid_w,
    const float* __restrict__ depths,        // (N)
    const float* __restrict__ normals_cam,   // (N, 3)
    const float* __restrict__ inv_covs,      // (N, 3) [ixx, iyy, ixy]
    const float* __restrict__ opacities,     // (N, 1)
    const float* __restrict__ colors,        // (N, 3) SH-DC radiance
    const float* __restrict__ proj_uv,       // (N, 2)
    const int* __restrict__ point_list,      // (T) tile-binned, depth-sorted
    const int* __restrict__ tile_ranges,     // (num_tiles * 2) [start, end)
    float* __restrict__ out_color,           // (3, H, W)
    float* __restrict__ out_normal,          // (3, H, W)
    float* __restrict__ out_depth,           // (1, H, W)
    float* __restrict__ out_alpha,           // (1, H, W)
    int* __restrict__ out_n_contrib          // (H * W)
) {
    const int tile_id = blockIdx.y * grid_w + blockIdx.x;
    const int px = blockIdx.x * BLOCK_X + threadIdx.x;
    const int py = blockIdx.y * BLOCK_Y + threadIdx.y;

    const bool inside = (px < W && py < H);
    const int pix_idx = py * W + px;
    const int tid = threadIdx.y * BLOCK_X + threadIdx.x;

    const int range_start = tile_ranges[tile_id * 2 + 0];
    const int range_end = tile_ranges[tile_id * 2 + 1];
    const int range_len = range_end - range_start;

    __shared__ float lds_proj_uv[LDS_CHUNK * 2];
    __shared__ float lds_inv_cov[LDS_CHUNK * 3];
    __shared__ float lds_opacity[LDS_CHUNK];
    __shared__ float lds_color[LDS_CHUNK * 3];
    __shared__ float lds_normal[LDS_CHUNK * 3];
    __shared__ float lds_depth[LDS_CHUNK];

    float acc_c[3] = {0.0f, 0.0f, 0.0f};
    float acc_n[3] = {0.0f, 0.0f, 0.0f};
    float acc_d = 0.0f;
    float T = 1.0f;

    bool done = !inside;
    int contrib = 0;
    int last_contrib = 0;

    for (int base = 0; base < range_len; base += LDS_CHUNK) {
        const int chunk = min(LDS_CHUNK, range_len - base);

        // Cooperative LDS staging: one primitive per thread per round.
        if (tid < chunk) {
            const int g = point_list[range_start + base + tid];
            lds_proj_uv[tid * 2 + 0] = proj_uv[g * 2 + 0];
            lds_proj_uv[tid * 2 + 1] = proj_uv[g * 2 + 1];
            lds_inv_cov[tid * 3 + 0] = inv_covs[g * 3 + 0];
            lds_inv_cov[tid * 3 + 1] = inv_covs[g * 3 + 1];
            lds_inv_cov[tid * 3 + 2] = inv_covs[g * 3 + 2];
            lds_opacity[tid] = opacities[g];
            lds_color[tid * 3 + 0] = colors[g * 3 + 0];
            lds_color[tid * 3 + 1] = colors[g * 3 + 1];
            lds_color[tid * 3 + 2] = colors[g * 3 + 2];
            lds_normal[tid * 3 + 0] = normals_cam[g * 3 + 0];
            lds_normal[tid * 3 + 1] = normals_cam[g * 3 + 1];
            lds_normal[tid * 3 + 2] = normals_cam[g * 3 + 2];
            lds_depth[tid] = depths[g];
        }
        __syncthreads();

        if (!done) {
            for (int j = 0; j < chunk; ++j) {
                contrib++;

                const float dx = (float)px - lds_proj_uv[j * 2 + 0];
                const float dy = (float)py - lds_proj_uv[j * 2 + 1];
                const float ixx = lds_inv_cov[j * 3 + 0];
                const float iyy = lds_inv_cov[j * 3 + 1];
                const float ixy = lds_inv_cov[j * 3 + 2];

                const float d2 = dx * dx * ixx + 2.0f * dx * dy * ixy + dy * dy * iyy;
                if (d2 > 9.0f || d2 < 0.0f) continue;

                float alpha = lds_opacity[j] * __expf(-0.5f * d2);
                alpha = fminf(alpha, ALPHA_MAX);
                if (alpha < ALPHA_MIN) continue;

                const float w = alpha * T;
                acc_c[0] += w * lds_color[j * 3 + 0];
                acc_c[1] += w * lds_color[j * 3 + 1];
                acc_c[2] += w * lds_color[j * 3 + 2];
                acc_n[0] += w * lds_normal[j * 3 + 0];
                acc_n[1] += w * lds_normal[j * 3 + 1];
                acc_n[2] += w * lds_normal[j * 3 + 2];
                acc_d += w * lds_depth[j];

                T *= (1.0f - alpha);
                last_contrib = contrib;

                if (T < T_EPS) { done = true; break; }
            }
        }
        __syncthreads();
    }

    if (inside) {
        const int hw = H * W;
        out_color[0 * hw + pix_idx] = acc_c[0];
        out_color[1 * hw + pix_idx] = acc_c[1];
        out_color[2 * hw + pix_idx] = acc_c[2];
        // Unnormalized accumulation: normalizing here would make the analytic
        // backward inconsistent. The Python wrapper normalizes for the loss.
        out_normal[0 * hw + pix_idx] = acc_n[0];
        out_normal[1 * hw + pix_idx] = acc_n[1];
        out_normal[2 * hw + pix_idx] = acc_n[2];
        out_depth[pix_idx] = acc_d;
        out_alpha[pix_idx] = 1.0f - T;
        out_n_contrib[pix_idx] = last_contrib;
    }
}

std::vector<torch::Tensor> rasterize_forward_hip(
    const torch::Tensor& depths,
    const torch::Tensor& normals_cam,
    const torch::Tensor& inv_covs,
    const torch::Tensor& opacities,
    const torch::Tensor& colors,
    const torch::Tensor& proj_uv,
    const torch::Tensor& point_list,
    const torch::Tensor& tile_ranges,
    int H,
    int W
) {
    auto opts = depths.options();
    auto int_opts = opts.dtype(torch::kInt32);

    auto out_color = torch::zeros({3, H, W}, opts);
    auto out_normal = torch::zeros({3, H, W}, opts);
    auto out_depth = torch::zeros({1, H, W}, opts);
    auto out_alpha = torch::zeros({1, H, W}, opts);
    auto out_n_contrib = torch::zeros({H, W}, int_opts);

    if (depths.size(0) == 0 || point_list.numel() == 0) {
        return {out_color, out_normal, out_depth, out_alpha, out_n_contrib};
    }

    const int grid_w = (W + BLOCK_X - 1) / BLOCK_X;
    const int grid_h = (H + BLOCK_Y - 1) / BLOCK_Y;

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(grid_w, grid_h);

    // Hoisted: .contiguous() temporaries must outlive the async launch.
    auto c_depths = depths.contiguous();
    auto c_normals = normals_cam.contiguous();
    auto c_invcov = inv_covs.contiguous();
    auto c_opac = opacities.contiguous();
    auto c_colors = colors.contiguous();
    auto c_uv = proj_uv.contiguous();
    auto c_list = point_list.contiguous();
    auto c_ranges = tile_ranges.contiguous();

    hipLaunchKernelGGL(
        Rasterize2DGSForwardKernel,
        grid, block, 0, 0,
        H, W, grid_w,
        c_depths.data_ptr<float>(),
        c_normals.data_ptr<float>(),
        c_invcov.data_ptr<float>(),
        c_opac.data_ptr<float>(),
        c_colors.data_ptr<float>(),
        c_uv.data_ptr<float>(),
        c_list.data_ptr<int>(),
        c_ranges.data_ptr<int>(),
        out_color.data_ptr<float>(),
        out_normal.data_ptr<float>(),
        out_depth.data_ptr<float>(),
        out_alpha.data_ptr<float>(),
        out_n_contrib.data_ptr<int>()
    );

    return {out_color, out_normal, out_depth, out_alpha, out_n_contrib};
}
