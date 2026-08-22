# Custom 2DGS Rasterizer (HIP)

Hand-authored HIP kernels for the custom 2DGS (surfel-based) differentiable rasterizer, targeting
ROCm on RDNA4 consumer GPUs (developed against RX 9060 XT, 16GB VRAM).

Written from scratch against the 2DGS algorithm as an architectural reference only — **not**
hipify-ported from the CUDA reference implementation (`diff-surfel-rasterization`). Hand-tuning
for RDNA4's 32-wide wavefronts, LDS size, and occupancy characteristics is the point: this is
where the performance ceiling is, at the cost of implementation difficulty (accepted trade-off).

Validate against `../rasterizer_torch_fallback/` (pure PyTorch, no custom kernels) as the
correctness oracle before trusting hand-tuned kernel changes.

ROCm/PyTorch-ROCm version pinning required — RDNA4 consumer-card ROCm support is newer and less
mature than CDNA/Instinct; re-validate the environment after any driver or ROCm update.
