// GlomeHomeTour Backend: C++ / PyTorch bindings for custom 2DGS HIP Rasterizer

#include <torch/extension.h>
#include <vector>

// Forward declaration of HIP launcher functions
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
);

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
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_forward", &rasterize_forward_hip, "2DGS tile-binned forward rasterization (HIP)");
    m.def("rasterize_backward", &rasterize_backward_hip, "2DGS tile-binned backward rasterization (HIP)");
}
