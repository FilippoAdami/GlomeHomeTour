#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include "ssim.h"

namespace {
void check_image(const torch::Tensor& image, const char* name) {
  TORCH_CHECK(image.is_cuda(), name, " must be a CUDA/ROCm tensor");
  TORCH_CHECK(image.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(image.scalar_type() == torch::kFloat32, name, " must be float32");
  TORCH_CHECK(image.dim() == 4 && image.size(0) > 0 && image.size(1) > 0
              && image.size(2) > 0 && image.size(3) > 0, name, " must be nonempty NCHW");
}
void check_pair(const torch::Tensor& img1, const torch::Tensor& img2) {
  check_image(img1, "img1"); check_image(img2, "img2");
  TORCH_CHECK(img1.device() == img2.device() && img1.sizes() == img2.sizes(),
              "images must have the same device and NCHW shape");
  TORCH_CHECK(!img2.requires_grad(), "img2 is a fixed target and must not require gradients");
}
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedssim_checked(float C1, float C2, torch::Tensor img1, torch::Tensor img2, bool train) {
  check_pair(img1, img2);
  TORCH_CHECK(train || !img1.requires_grad(), "train=False cannot be used when img1 requires gradients");
  c10::DeviceGuard guard(img1.device());
  return fusedssim(C1, C2, img1, img2, train);
}

torch::Tensor fusedssim_backward_checked(float C1, float C2, torch::Tensor img1, torch::Tensor img2,
                                         torch::Tensor dmap, torch::Tensor dmu1,
                                         torch::Tensor dsigma1, torch::Tensor dsigma12) {
  check_pair(img1, img2); check_image(dmap, "dL_dmap");
  for (const auto& value : {dmap, dmu1, dsigma1, dsigma12}) {
    check_image(value, "SSIM backward tensor");
    TORCH_CHECK(value.device() == img1.device() && value.sizes() == img1.sizes(),
                "SSIM backward tensors must match image device and NCHW shape");
  }
  c10::DeviceGuard guard(img1.device());
  return fusedssim_backward(C1, C2, img1, img2, dmap, dmu1, dsigma1, dsigma12);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fusedssim", &fusedssim_checked);
  m.def("fusedssim_backward", &fusedssim_backward_checked);
}
