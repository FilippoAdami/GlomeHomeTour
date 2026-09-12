// GlomeHomeTour: Wave32 Surfel Frustum Projection HIP Kernel
// Optimized for AMD RDNA4 (gfx1200 / RX 9070 XT)
// Constraints:
// - Native Wave32 execution (dim3 block(32, 1, 1))
// - LDS caching (<= 32 KB per Compute Unit)
// - Boundary checking on image dimensions (W, H)

#include <hip/hip_runtime.h>

#define WAVE_SIZE 32
#define LDS_MAX_ELEMENTS 1024

__global__ void project_surfels_to_camera_wave32_kernel(
    const float* __restrict__ positions,    // (N, 3) world coordinates [x, y, z]
    const float* __restrict__ w2c_matrix,   // (4, 4) world to camera matrix
    float fx, float fy, float cx, float cy,
    int width, int height,
    int num_surfels,
    float* __restrict__ out_uv_depth,       // (N, 3) [u, v, depth]
    unsigned char* __restrict__ out_valid   // (N) 1 if in frustum, 0 otherwise
) {
    // Wave32 lane indexing
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_surfels) return;

    // Cache w2c matrix into shared LDS memory (16 floats = 64 bytes)
    __shared__ float lds_w2c[16];
    if (threadIdx.x < 16) {
        lds_w2c[threadIdx.x] = w2c_matrix[threadIdx.x];
    }
    __syncthreads();

    // Load surfel world position
    float px = positions[tid * 3 + 0];
    float py = positions[tid * 3 + 1];
    float pz = positions[tid * 3 + 2];

    // Camera transform: P_cam = w2c @ [px, py, pz, 1]
    float cam_x = lds_w2c[0] * px + lds_w2c[1] * py + lds_w2c[2]  * pz + lds_w2c[3];
    float cam_y = lds_w2c[4] * px + lds_w2c[5] * py + lds_w2c[6]  * pz + lds_w2c[7];
    float cam_z = lds_w2c[8] * px + lds_w2c[9] * py + lds_w2c[10] * pz + lds_w2c[11];

    // In OpenGL coordinates, camera looks along -Z
    float depth = -cam_z;

    if (depth > 0.1f) {
        float inv_z = 1.0f / depth;
        float u = fx * (cam_x * inv_z) + cx;
        float v = -fy * (cam_y * inv_z) + cy;

        bool in_bounds = (u >= 0.0f) && (u < (float)width) && (v >= 0.0f) && (v < (float)height);

        out_uv_depth[tid * 3 + 0] = u;
        out_uv_depth[tid * 3 + 1] = v;
        out_uv_depth[tid * 3 + 2] = depth;
        out_valid[tid] = in_bounds ? 1 : 0;
    } else {
        out_uv_depth[tid * 3 + 0] = -1.0f;
        out_uv_depth[tid * 3 + 1] = -1.0f;
        out_uv_depth[tid * 3 + 2] = depth;
        out_valid[tid] = 0;
    }
}
