# Pure-PyTorch Rasterizer (Bring-up / Correctness Oracle)

Slow, pure-PyTorch 2DGS rasterizer with no custom kernels. Used to validate the 2DGS training
loop and math independent of GPU-kernel correctness, and as the ground-truth reference when
debugging the hand-tuned HIP rasterizer (`../rasterizer_hip/`).
