# Native fused SSIM

MIT-licensed derivative of Rahul Goel's `fused-ssim` (2024), copied from the
local `FastGS/submodules/fused-ssim` source on 2026-10-02. Only this independent
MIT component is included; no rasterizer or Inria source is present. It keeps
the original 11×11, sigma 1.5 Gaussian and zero-padded `same` behavior.

The native wrapper accepts contiguous float32 NCHW images on one GPU, uses the
current PyTorch HIP/CUDA stream, and permits gradients only for the first image.
