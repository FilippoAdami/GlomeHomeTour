# Stage 03: gsplat training on ROCm

This stage trains 3D Gaussian splats from COLMAP cameras and the depth and normal maps produced by stage 02. `gsplat_train.py` is a local implementation that uses [AMD gsplat](https://github.com/AMD-Ecosystem/gsplat) for rasterization and densification. The AMD gsplat source is Apache-2.0 licensed; its license is included in `GSPLAT_APACHE_LICENSE.txt`. The gfx1200 HIP changes in `gsplat_gfx1200.patch` apply to AMD gsplat tag `1.5.3b2` and target ROCm 7.1; no ROCm downgrade is needed.

Install into the same ROCm-enabled Python environment used by the backend:

```bash
backend/03_FastGS_DNSplatter/install_gsplat_gfx1200.sh backend/.venv/bin/python
```

The script clones the pinned AMD release, applies the local HIP patch, and builds for `PYTORCH_ROCM_ARCH=gfx1200`. It needs a working ROCm compiler, PyTorch ROCm wheel, and internet access. It does not install any Inria rasterizer or `simple-knn` extension.

Run the pipeline step or train directly:

```bash
backend/.venv/bin/python backend/03_FastGS_DNSplatter/step_train.py --workspace backend/current_scene
backend/.venv/bin/python backend/03_FastGS_DNSplatter/gsplat_train.py -s backend/current_scene -m backend/current_scene/03_FastGS_DNSplatter --iterations 22000 --depth-supervision --normal-supervision
```

Output is `checkpoint.pt` plus `point_cloud/iteration_<N>/point_cloud.ply` in the stage 03 model directory. Stage 04 reads this checkpoint to fuse rendered depth into a mesh. For longer quality runs, `--iterations 30000 --save-iterations 22000` also writes a comparable intermediate model under `snapshots/iteration_22000/`. Existing FastGS/Inria checkpoints are incompatible. The directory name remains for pipeline path compatibility.

The trainer independently implements FastGS's ten-view flagged-pixel support, clone/absolute-gradient split gates, weighted stochastic pruning, final pruning, opacity resets, and optimizer cadence using Apache gsplat topology operations. It uses camera-center extent for scale thresholds and position learning rates. Every camera is sampled once per shuffled training cycle before the cycle refills. Refinement runs every 100 steps from 600 through 14,900; final pruning runs at 18,000 and 21,000 for a 22,000-step run. Screen-radius history resets before event pruning, matching the original behavior. Retaining that history degraded coverage in the progressive quality check.

Photometric supervision is `.8 L1 + .2 (1 - SSIM)`. The installer also builds the separately MIT-licensed [fused SSIM component](native_ssim/README.md). DN-Splatter depth/normal losses use valid alpha/depth and neighboring pixels. Default weights are **depth 0.15** and **normal 0.075**, with supervision through the final iteration. `--geometry-until 15000` restores the former supervision window; retaining supervision prevents late appearance updates from drifting geometry. Normal targets use the default stage 02 StableNormal convention; the optional `depth_gradient` producer uses a different convention and needs conversion before training. `torch.compile` fuses the geometry loss; its first call compiles kernels. `--eager-loss` disables fusion for debugging.

FastGS compact footprint defaults to `--footprint-multiplier .5`, matching the original setting. This scales the opacity-cutoff ellipse used to select rectangular tile bounds and intentionally omits peripheral contributions. Set it to `1` for full gsplat support. Tile size is 16. The gfx1200 checkpoint backward applies to RGB and RGB+depth training: waves own Gaussians and accumulate gradients in registers using forward checkpoints. Checkpoint storage is capped at 1 GiB and one third of free VRAM. Large intersection lists first use a native forward prepass to size storage through the last accepted contributions; the standard gsplat kernel remains the fallback. No-grad rendering uses standard gsplat.

Depth/normal targets stay in a shared CPU cache capped by `--target-cache-mb` (default 8192 MiB; 0 disables caching). Source float16 maps widen exactly to float32 on the GPU. Resized maps use float32 interpolation. Cameras are cached on the GPU; RGB stays cached as uint8 on the CPU.

On RX 9060 XT at 1080×1920 and 300k initial splats, the final paired run saved its **22k checkpoint in 439.36 seconds (7m 19s)** and completed **30k in 563.15 seconds (9m 23s)**, including imports, compilation, refinement and exports. The 22k timing excludes process shutdown and subsequent training. On 46 genuinely held-out views, 30k adds only **0.010 dB PSNR** and **0.00011 SSIM** over 22k: use the default **22,000 iterations**. Image PSNR matches the supplied INRIA reference closely, SSIM and normal-prior agreement improve, and depth-prior correlation is slightly lower. These are scene-specific measurements, not a guarantee for every view or scene. See [results, limitations and runnable checks](agents_files/artifacts/gfx1200-performance/REPORT.md).

The repository's root [LICENSE](../../LICENSE) says AGPL-3.0, while `backend/pyproject.toml` declares Apache-2.0. Confirm the intended license for local backend code before distribution. AMD gsplat's Apache-2.0 license and its dependencies must also be observed when distributing builds.
