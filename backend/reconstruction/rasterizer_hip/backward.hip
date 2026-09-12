// GlomeHomeTour Backend: 2DGS HIP Backward Gradient Kernel
// Target: AMD RDNA4 (Radeon RX 9060 XT, ROCm 7.x, gfx1200)
//
// Canonical alpha-compositing backward pass. Walks each tile's binned range
// BACK-TO-FRONT, reconstructing transmittance by division and carrying the
// running "colour behind this splat" (accum_rec) so that
//
//     dC/d(alpha_j) = T_j * (c_j - C_behind_j)
//
// The previous implementation kept only the T_j * c_j half, which told a splat
// that raising its opacity adds its own colour but never that it occludes what
// is behind it -- biasing every opacity and position gradient in the scene.

#include <hip/hip_runtime.h>
#include <torch/types.h>
#include <vector>

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define LDS_CHUNK BLOCK_SIZE
#define ALPHA_MIN (1.0f / 255.0f)
#define ALPHA_MAX 0.99f

// 3 colour + 3 normal + 1 depth + 1 constant-one channel for accumulated alpha.
// acc_alpha = 1 - prod(1 - a_k) = sum_k a_k T_k, i.e. exactly the compositing
// of a channel whose per-splat value is 1, so it rides the same accum_rec path.
#define N_CH 8

__global__ void __launch_bounds__(BLOCK_SIZE, 2)
Rasterize2DGSBackwardKernel(
    const int H,
    const int W,
    const int grid_w,
    const float* __restrict__ dL_dcolor,     // (3, H, W)
    const float* __restrict__ dL_dnormal,    // (3, H, W)
    const float* __restrict__ dL_ddepth,     // (1, H, W)
    const float* __restrict__ dL_dalpha_img, // (1, H, W)
    const float* __restrict__ final_alpha,   // (1, H, W)
    const int* __restrict__ n_contrib,       // (H * W)
    const float* __restrict__ depths,        // (N)
    const float* __restrict__ normals_cam,   // (N, 3)
    const float* __restrict__ inv_covs,      // (N, 3)
    const float* __restrict__ opacities,     // (N, 1)
    const float* __restrict__ colors,        // (N, 3)
    const float* __restrict__ proj_uv,       // (N, 2)
    const int* __restrict__ point_list,      // (T)
    const int* __restrict__ tile_ranges,     // (num_tiles * 2)
    float* __restrict__ dL_dproj_uv,         // (N, 2)
    float* __restrict__ dL_dinv_cov,         // (N, 3)
    float* __restrict__ dL_dopacity,         // (N, 1)
    float* __restrict__ dL_dcolor_out,       // (N, 3)
    float* __restrict__ dL_dnormal_out,      // (N, 3)
    float* __restrict__ dL_ddepth_out        // (N, 1)
) {
    const int tile_id = blockIdx.y * grid_w + blockIdx.x;
    const int px = blockIdx.x * BLOCK_X + threadIdx.x;
    const int py = blockIdx.y * BLOCK_Y + threadIdx.y;

    const bool inside = (px < W && py < H);
    const int pix_idx = py * W + px;
    const int hw = H * W;
    const int tid = threadIdx.y * BLOCK_X + threadIdx.x;

    const int range_start = tile_ranges[tile_id * 2 + 0];
    const int range_end = tile_ranges[tile_id * 2 + 1];
    const int range_len = range_end - range_start;

    // Per-pixel upstream gradients, laid out to match the N_CH channel order.
    float dL_dpix[N_CH];
    if (inside) {
        dL_dpix[0] = dL_dcolor[0 * hw + pix_idx];
        dL_dpix[1] = dL_dcolor[1 * hw + pix_idx];
        dL_dpix[2] = dL_dcolor[2 * hw + pix_idx];
        dL_dpix[3] = dL_dnormal[0 * hw + pix_idx];
        dL_dpix[4] = dL_dnormal[1 * hw + pix_idx];
        dL_dpix[5] = dL_dnormal[2 * hw + pix_idx];
        dL_dpix[6] = dL_ddepth[pix_idx];
        dL_dpix[7] = dL_dalpha_img[pix_idx];
    } else {
        for (int c = 0; c < N_CH; ++c) dL_dpix[c] = 0.0f;
    }
    // Transmittance left over after the forward pass: T_final = 1 - accumulated alpha.
    float T = inside ? (1.0f - final_alpha[pix_idx]) : 0.0f;
    const int last_contrib = inside ? n_contrib[pix_idx] : 0;

    float accum_rec[N_CH];
    float last_val[N_CH];
    for (int c = 0; c < N_CH; ++c) { accum_rec[c] = 0.0f; last_val[c] = 0.0f; }
    float last_alpha = 0.0f;

    __shared__ float lds_proj_uv[LDS_CHUNK * 2];
    __shared__ float lds_inv_cov[LDS_CHUNK * 3];
    __shared__ float lds_opacity[LDS_CHUNK];
    __shared__ float lds_val[LDS_CHUNK * N_CH];
    __shared__ int lds_gid[LDS_CHUNK];

    // Iterate the tile range back-to-front, one LDS chunk at a time.
    const int num_chunks = (range_len + LDS_CHUNK - 1) / LDS_CHUNK;

    for (int ci = num_chunks - 1; ci >= 0; --ci) {
        const int base = ci * LDS_CHUNK;
        const int chunk = min(LDS_CHUNK, range_len - base);

        if (tid < chunk) {
            const int g = point_list[range_start + base + tid];
            lds_gid[tid] = g;
            lds_proj_uv[tid * 2 + 0] = proj_uv[g * 2 + 0];
            lds_proj_uv[tid * 2 + 1] = proj_uv[g * 2 + 1];
            lds_inv_cov[tid * 3 + 0] = inv_covs[g * 3 + 0];
            lds_inv_cov[tid * 3 + 1] = inv_covs[g * 3 + 1];
            lds_inv_cov[tid * 3 + 2] = inv_covs[g * 3 + 2];
            lds_opacity[tid] = opacities[g];
            lds_val[tid * N_CH + 0] = colors[g * 3 + 0];
            lds_val[tid * N_CH + 1] = colors[g * 3 + 1];
            lds_val[tid * N_CH + 2] = colors[g * 3 + 2];
            lds_val[tid * N_CH + 3] = normals_cam[g * 3 + 0];
            lds_val[tid * N_CH + 4] = normals_cam[g * 3 + 1];
            lds_val[tid * N_CH + 5] = normals_cam[g * 3 + 2];
            lds_val[tid * N_CH + 6] = depths[g];
            lds_val[tid * N_CH + 7] = 1.0f;
        }
        __syncthreads();

        if (inside) {
            for (int j = chunk - 1; j >= 0; --j) {
                // Local index within the tile range; skip anything the forward
                // pass never reached (early-terminated on transmittance).
                const int li = base + j;
                if (li + 1 > last_contrib) continue;

                const float dx = (float)px - lds_proj_uv[j * 2 + 0];
                const float dy = (float)py - lds_proj_uv[j * 2 + 1];
                const float ixx = lds_inv_cov[j * 3 + 0];
                const float iyy = lds_inv_cov[j * 3 + 1];
                const float ixy = lds_inv_cov[j * 3 + 2];

                const float d2 = dx * dx * ixx + 2.0f * dx * dy * ixy + dy * dy * iyy;
                if (d2 > 9.0f || d2 < 0.0f) continue;

                const float gauss = __expf(-0.5f * d2);
                const float op = lds_opacity[j];
                const float raw_alpha = op * gauss;
                const float alpha = fminf(raw_alpha, ALPHA_MAX);
                if (alpha < ALPHA_MIN) continue;

                // Recover the transmittance seen by this splat in the forward pass.
                T /= fmaxf(1.0f - alpha, 1e-6f);

                const float w = alpha * T;
                const int g = lds_gid[j];

                // d(channel)/d(value_j) = alpha_j * T_j
                atomicAdd(&dL_dcolor_out[g * 3 + 0], w * dL_dpix[0]);
                atomicAdd(&dL_dcolor_out[g * 3 + 1], w * dL_dpix[1]);
                atomicAdd(&dL_dcolor_out[g * 3 + 2], w * dL_dpix[2]);
                atomicAdd(&dL_dnormal_out[g * 3 + 0], w * dL_dpix[3]);
                atomicAdd(&dL_dnormal_out[g * 3 + 1], w * dL_dpix[4]);
                atomicAdd(&dL_dnormal_out[g * 3 + 2], w * dL_dpix[5]);
                atomicAdd(&dL_ddepth_out[g], w * dL_dpix[6]);

                // dC/d(alpha_j) = T_j * (value_j - value accumulated behind j)
                float dL_dalpha = 0.0f;
                for (int c = 0; c < N_CH; ++c) {
                    const float val = lds_val[j * N_CH + c];
                    accum_rec[c] = last_alpha * last_val[c] + (1.0f - last_alpha) * accum_rec[c];
                    last_val[c] = val;
                    dL_dalpha += (val - accum_rec[c]) * dL_dpix[c];
                }
                dL_dalpha *= T;
                last_alpha = alpha;

                // alpha was clamped: no gradient flows through a saturated splat.
                if (raw_alpha > ALPHA_MAX) continue;

                atomicAdd(&dL_dopacity[g], dL_dalpha * gauss);

                // alpha = op * exp(-0.5 * d2)  =>  d(alpha)/d(d2) = -0.5 * alpha
                const float dL_dd2 = -0.5f * dL_dalpha * alpha;

                // d(d2)/d(uv) = -2 * (Sigma^-1 [dx, dy])
                atomicAdd(&dL_dproj_uv[g * 2 + 0], -2.0f * dL_dd2 * (dx * ixx + dy * ixy));
                atomicAdd(&dL_dproj_uv[g * 2 + 1], -2.0f * dL_dd2 * (dx * ixy + dy * iyy));

                // d(d2)/d(inv_cov) for [ixx, iyy, ixy]
                atomicAdd(&dL_dinv_cov[g * 3 + 0], dL_dd2 * dx * dx);
                atomicAdd(&dL_dinv_cov[g * 3 + 1], dL_dd2 * dy * dy);
                atomicAdd(&dL_dinv_cov[g * 3 + 2], dL_dd2 * 2.0f * dx * dy);
            }
        }
        __syncthreads();
    }
}

std::vector<torch::Tensor> rasterize_backward_hip(
    const torch::Tensor& dL_dcolor,
    const torch::Tensor& dL_dnormal,
    const torch::Tensor& dL_ddepth,
    const torch::Tensor& dL_dalpha_img,
    const torch::Tensor& final_alpha,
    const torch::Tensor& n_contrib,
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
    auto dL_dproj_uv = torch::zeros_like(proj_uv);
    auto dL_dinv_cov = torch::zeros_like(inv_covs);
    auto dL_dopacity = torch::zeros_like(opacities);
    auto dL_dcolor_out = torch::zeros_like(colors);
    auto dL_dnormal_out = torch::zeros_like(normals_cam);
    auto dL_ddepth_out = torch::zeros_like(depths);

    if (depths.size(0) == 0 || point_list.numel() == 0) {
        return {dL_dproj_uv, dL_dinv_cov, dL_dopacity, dL_dcolor_out, dL_dnormal_out, dL_ddepth_out};
    }

    const int grid_w = (W + BLOCK_X - 1) / BLOCK_X;
    const int grid_h = (H + BLOCK_Y - 1) / BLOCK_Y;

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(grid_w, grid_h);

    // Hoisted: .contiguous() temporaries must outlive the async launch.
    auto c_dc = dL_dcolor.contiguous();
    auto c_dn = dL_dnormal.contiguous();
    auto c_dd = dL_ddepth.contiguous();
    auto c_da = dL_dalpha_img.contiguous();
    auto c_fa = final_alpha.contiguous();
    auto c_nc = n_contrib.contiguous();
    auto c_depths = depths.contiguous();
    auto c_normals = normals_cam.contiguous();
    auto c_invcov = inv_covs.contiguous();
    auto c_opac = opacities.contiguous();
    auto c_colors = colors.contiguous();
    auto c_uv = proj_uv.contiguous();
    auto c_list = point_list.contiguous();
    auto c_ranges = tile_ranges.contiguous();

    hipLaunchKernelGGL(
        Rasterize2DGSBackwardKernel,
        grid, block, 0, 0,
        H, W, grid_w,
        c_dc.data_ptr<float>(),
        c_dn.data_ptr<float>(),
        c_dd.data_ptr<float>(),
        c_da.data_ptr<float>(),
        c_fa.data_ptr<float>(),
        c_nc.data_ptr<int>(),
        c_depths.data_ptr<float>(),
        c_normals.data_ptr<float>(),
        c_invcov.data_ptr<float>(),
        c_opac.data_ptr<float>(),
        c_colors.data_ptr<float>(),
        c_uv.data_ptr<float>(),
        c_list.data_ptr<int>(),
        c_ranges.data_ptr<int>(),
        dL_dproj_uv.data_ptr<float>(),
        dL_dinv_cov.data_ptr<float>(),
        dL_dopacity.data_ptr<float>(),
        dL_dcolor_out.data_ptr<float>(),
        dL_dnormal_out.data_ptr<float>(),
        dL_ddepth_out.data_ptr<float>()
    );

    return {dL_dproj_uv, dL_dinv_cov, dL_dopacity, dL_dcolor_out, dL_dnormal_out, dL_ddepth_out};
}
