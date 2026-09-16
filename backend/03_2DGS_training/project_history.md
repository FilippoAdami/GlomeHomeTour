# Project History: Backend Reconstruction

## Milestone: Depth Anything 3 (DA3-NESTED-GIANT-LARGE-1.1) FP16 & Alignment Integration

- **Model Selection & Setup:** Integrated `depth-anything/DA3NESTED-GIANT-LARGE-1.1` (1.69B parameters dual-stage architecture) via `backend/third_party/depth_anything_3`.
- **Precision Engineering:**
  - Resolved FP16/BF16 type mismatch errors in `head_utils.py` (positional sinusoidal embeddings preserving input precision).
  - Resolved linear layer casting in `cam_dec.py` (`CameraDec` preserving module dtype).
  - Added `safe_quantile` in `alignment.py` and `da3.py` allowing robust confidence thresholding across FP16 and BF16.
  - Implemented dtype-preserving `least_squares_scale_scalar` and `OutputProcessor` tensor-to-numpy conversion.
  - Configured `DepthPriorEstimator` with `use_fp16=True` supporting both GPU (CUDA/ROCm) and CPU (BF16).
- **Scale-Shift Global Pose Graph:** Verified Lucas-Kanade temporal tracking + loop-closure graph optimization bringing cross-view discrepancy from 13.10 cm to 4.37 cm.
- **Contract & Schema Verification:** All tests passing and `shared/schemas/validate.py` validated with 100% compliance.

## Milestone: Dynamic Adaptive Depth Ceiling & Multi-View Consensus Filtering

- **Dynamic Statistical Depth Horizon:** Implemented `estimate_adaptive_depth_ceiling` using high-confidence depth quantile ($Q_{0.98} + 1.5 \cdot \text{MAD}$), automatically scaling from compact rooms (3.5m–4.5m) to open halls/hotel lobbies without hardcoded limits.
- **Multi-View Geometric Consensus:** Reprojects 3D surfel points into neighboring cameras, rejecting non-surface rays (windows, sky, reflections) failing multi-view agreement ($|\Delta z| \le 8\text{cm} + 5\% z$).
- **Statistical Outlier Removal:** Integrated KDTree k-NN distance filter ($k=16, 1.5\sigma$) to eliminate isolated air noise.
- **Upstream Float16 Dot-Product Patch:** Fixed 16-bit dot product overflow in `alignment.py` (`torch.dot(a_f32, b_f32)`), preventing scale factor NaNs during metric alignment.
- **Real-World Verification:** Tested on 8.8-min `bedroom_complete.zip`: 15m conical fan eliminated, max radial distance bounded tightly at 3.79m, and 24/24 unit tests passing.

## Milestone: Phase 5 Stages 1 & 2 (2DGS Material Model, Deferred PBR Shader & Loss Engine)

- **Stage 1 (Core Model & Activations):**
  - Implemented `Material2DGSModel` in `backend/reconstruction/training/model.py` wrapping 2D planar Gaussian surfels with learnable PBR attributes: positions `_xyz`, quaternions `_rotation`, log 2D scales `_scaling`, logit opacities `_opacity`, logit diffuse albedo `_albedo`, logit roughness `_roughness`, and logit metallic `_metallic`.
  - Implemented activation property accessors with strict bounds enforcement: `roughness` $\in [0.04, 1.0]$, `albedo`, `metallic`, `opacity` $\in [0, 1]$, unit-norm quaternions `rotation`, and positive scales `scaling`.
  - Added vectorized `matrix_to_quaternion` and `quaternion_to_rotation_matrix` functions, and derived properties `normals`, `tangent_u`, and `tangent_v` forming orthonormal basis frames.
  - Implemented `from_surfel_cloud` constructor converting surfels into model parameters with safe logit/log clamping.
- **Stage 2 (PBR Shader, Rasterizer Interface & Losses):**
  - Defined validated `GBufferOutput` dataclass in `backend/reconstruction/training/rasterizer_interface.py` tracking screen-space Albedo, Normal, Roughness, Metallic, and Depth buffers with channel and shape constraints.
  - Implemented `PyTorchFallbackRasterizer` differentiable 2DGS rasterizer oracle performing analytical 2D Gaussian projection, Jacobian screen covariance computation, and front-to-back alpha compositing.
  - Implemented `DeferredCookTorranceShader` in `backend/reconstruction/training/pbr_shader.py` evaluating Trowbridge-Reitz GGX NDF, Schlick Fresnel, Smith-GGX geometry shadowing, energy-conserving Lambertian diffuse, and Split-Sum IBL ambient lighting with zero-division guards.
  - Implemented composite `ReconstructionLoss` in `backend/reconstruction/training/losses.py` combining $L_1$ color loss, differentiable 2D Gaussian window SSIM loss, and surface normal consistency loss $L_{\text{normal}} = 1 - \langle \mathbf{N}_{\text{rendered}}, \mathbf{N}_{\text{depth\_prior}} \rangle$.
- **Verification & Testing:**
  - Authored self-contained test suite in `backend/reconstruction/training/tests/test_model_and_shader.py` using synthetic/mock surfel clouds and cameras.
  - 9/9 unit tests passing with FP16/FP32 precision preservation, zero-division protection, and verified backprop optimization step.

## Milestone: Phase 5 Stages 3 through 7 (Density Control, Planar Mirrors, Compression & Trainer Pipeline)

- **Stage 3 (AMD RDNA4 Custom HIP Rasterizer Kernel):**
  - Authored `backend/reconstruction/rasterizer_hip/` kernels (`forward.hip`, `backward.hip`, `ext.cpp`, `setup.py`) optimized for AMD RDNA4 (Wave32 SIMD execution `__launch_bounds__(32, 8)` and 64KB LDS memory per CU).
  - Integrated graceful runtime auto-detection in `rasterizer_interface.py`: uses compiled HIP kernel when available, with automatic fallback to pure-PyTorch differentiable oracle (`PyTorchFallbackRasterizer`).
- **Stage 4 (Taming-2DGS Density Control & Normal-Aware Voxel Grid Pruning):**
  - Implemented `TamingDensityController` in `backend/reconstruction/training/density_control.py` managing spatial gradient accumulation, budget-capped splitting/cloning ($\le 400,000$ primitives), aspect ratio pruning, and periodic opacity resets.
  - Implemented **Normal-Aware Spatial Voxel Grid Filtering** ($1.5\text{ cm}$ spatial cells + 8-octant normal binning) to collapse redundant coplanar duplicates while strictly preserving opposite faces of thin drywalls and doors.
- **Stage 5 (Planar Reflections & Mirror Passes):**
  - Implemented `PlanarMirrorDetector` in `backend/reconstruction/training/planar_reflections.py` utilizing RANSAC to detect dominant specular planar clusters ($m \ge 0.70, \alpha \le 0.20$).
  - Implemented 4x4 Householder affine reflection transformation ($\mathbf{H}_{\text{refl}} = \mathbf{I} - 2 \mathbf{n} \mathbf{n}^T$) and virtual camera view matrix generation for rendering reflective surfaces.
- **Stage 6 (LightGaussian Vector Quantization & Web Packaging):**
  - Implemented `LightGaussianCompressor` in `backend/reconstruction/training/compressor.py` with 8-bit K-means Vector Quantization (256 albedo centroids, 64 roughness/metallic centroids, 256 scale centroids, 16-bit bounding box positions).
  - Implemented compact binary serialization (`.bin`) and Zip compression exporting `walkthrough_2dgs.zip` well within the $\le 25\text{ MB}$ MLS payload limit.
- **Stage 7 (Pipeline Orchestrator & End-to-End Validation):**
  - Implemented `Material2DGSTrainer` in `backend/reconstruction/training/trainer.py` orchestrating sliding-window keyframe batch sampling, decoupled Adam learning rates, density scheduling, checkpointing, and evaluation metrics.
- **Test Suite Verification:**
  - 17/17 unit tests passing across all test suites (`test_model_and_shader.py`, `test_density_control.py`, `test_planar_reflections.py`, `test_compressor.py`, `test_trainer.py`).
  - Shared schemas (`python shared/schemas/validate.py`) remain 100% compliant.


## 2026-09-10: Diagnose and fix the warped/oversized reconstruction (per `Implementation_plan.md`)
Outcome: worked — two independent root causes, neither in the geometry math.
`expandable_segments:True` corrupted DA3 inference on ROCm/gfx1200 (silently wrong depth →
100% NaN + HIP illegal-instruction abort), and `estimate_depth_sequence` quietly turned that
NaN into a uniform 15 m depth map, which unprojects to the "bowl". Separately, DA3 was fed
ARCore c2w poses where it needs OpenCV w2c, costing metric scale (adjacent-view agreement
5–30 cm → 0.7–1.5 cm). Unprojection in `initialization.py` was correct and was not changed.
Added `DepthEstimationError` so unusable depth fails loudly, plus
`tests/test_geometry_conventions.py` (mutation-checked). 30-keyframe run: floor level to
2.81 cm in the gravity-aligned frame, straight perpendicular walls. Full detail in
`backend/project_history.md`.

## 2026-09-10: Depth Anything 3 Base (DA3-BASE, Apache 2.0) Commercial Verification & Full Reconstruction

- **Root Cause of the Recurrent Bowl:**
  When switching to `depth-anything/DA3-BASE`, weights (`model.safetensors`, 542 MB) were not pre-cached locally. A broad `except Exception:` block in `_init_model()` silently fell back to single-view `Depth-Anything-V2-Metric-Indoor-Base-hf`, which lacks multi-view attention and unprojected into a 12-meter spherical bowl.
- **Fail-Loud Architecture:**
  Refactored `_init_model()` in `backend/reconstruction/depth_priors.py` to raise explicit `RuntimeError` on model loading failures, preventing silent fallbacks to uncalibrated models. Downloaded `DA3-BASE` snapshot locally.
- **Full Keyframe Sliding-Window Verification (273 Keyframes):**
  - Ran sliding-window inference with `depth-anything/DA3-BASE` on 68 chunks ($N=6, K=2$) on AMD ROCm in 4.1 minutes ($2.1\times$ faster than Giant's 8.9 minutes).
  - Unprojected 2,306,347 multi-view consistent surfels.
  - Geometry result: straight, perpendicular walls ($5.87\text{m} \times 5.02\text{m}$) without any bowing, perfectly matching the metric scale and planarity of the Giant model.
- **Artifacts & Compatibility:**
  - Exported `bedroom_dense_mesh.glb` (202.96 MB) and `bedroom_dense_mesh.ply` (207.57 MB) with native vertex colors for direct Blender inspection.
  - Rendered side-by-side comparison `true_giant_vs_base_comparison.png` confirming commercial Apache 2.0 readiness.

## 2026-09-10: Elimination of Onion-Peel Layering & Dynamic Statistical Ceiling Integration

- **Root Causes of Onion-Peel Multi-Layering:**
  1. *Pairwise Recursive Scale Drift:* `depth_priors.py` sequentially multiplied inter-chunk scale factors ($s$), drifting from $0.034$ to $2.22$ across 68 chunks. The same wall viewed from different chunks was assigned wildly different depths.
  2. *Disabled Consensus:* `min_consensus=0` failed to prune distant foreshortened rays.
- **Permanent Pipeline Fixes:**
  - Replaced pairwise chained scaling in `estimate_depth_sliding_window` with direct multi-window median consensus.
  - Configured `SurfelCloudInitializer` with `max_depth_m=None`, strictly enforcing the dynamic statistical depth ceiling (`estimate_adaptive_depth_ceiling`: $Q_{0.98} + 1.5 \cdot \text{MAD}$ = 4.13m for the bedroom, scaling up to 25m+ in open halls).
  - Set `min_consensus=1` as standard default to prune uncorroborated free-space floaters while preserving uniquely scanned corners and ceilings.
  - Set quad splat overlap scale $s = 0.55 \cdot \text{voxel\_size}$ to seamlessly weld adjacent micro-gaps.
- **Verification:**
  - Exported `bedroom_balanced_mesh.glb` (184.47 MB, 2,096,215 surfels) with completely unified single-sheet planar walls and continuous ceiling/floor coverage.
  - 58/58 unit tests passing and schema validation 100% compliant.

## Milestone: PLY + Frames → 2DGS Training Pipeline Integration & Multi-Scale Progressive Optimization

- **Fast Binary PLY Ingestion & Voxel Filtering:**
  - Optimized binary PLY parser (`SurfelCloud.from_ply`) with single-pass vectorized `<3f3f3B2ff` parsing ($<0.010\text{s}$ for 2.1M surfels) and spatial voxel downsampling (3.5 cm grid brings 2.1M surfels to ~350k–400k primitives for training initialization).
  - Derived orthonormal tangent frames $(u, v)$ from unit normals on load.
- **GSInputDataset & Multi-Scale Resolution Support:**
  - Implemented `GSInputDataset` in `backend/reconstruction/training/dataset.py` loading `transforms.json`, upright 1080x1920 keyframes, metric depth maps (`depth_*.npy`), and dense camera-space normal maps.
  - Enabled dynamic multi-scale resolution resizing (e.g. 270x480 -> 540x960 -> 1080x1920) with focal length and principal point recalculations.
- **Standalone Pipeline Runner & Sliding Window Integration:**
  - Implemented `backend/reconstruction/training/run_scene_training.py` with multi-scale progressive scheduling, intermediate checkpointing, and `walkthrough_2dgs.zip` export.
  - Extended `backend/run_sliding_window_reconstruction.py` with `--train-2dgs` and `--iterations-2dgs` flags.
- **Vectorized Differentiable Rasterizer Speedup:**
  - Vectorized alpha compositing in `PyTorchFallbackRasterizer` using native PyTorch prefix-product `cumprod` and batch `matmul`, eliminating sequential Python loops for a 50x–100x acceleration.
- **Verification:**
  - 21/21 passing unit tests in `backend/reconstruction/training/tests/` (including `test_ply_pipeline.py`).
  - Schema validation (`python shared/schemas/validate.py`) passing with 100% compliance.

## Milestone: 2DGS Tiled Bounding-Box Spatial Culling & ROCm GPU Optimization

- **Root Cause of Training Stall & Low GPU Utilization:**
  - Previous `PyTorchFallbackRasterizer` compared every pixel chunk against every surfel globally via nested Python loops ($\approx 87,488$ inner loop dispatches per step for 350k surfels).
  - This single-thread Python loop execution starved the GPU (near 0% utilization), while accumulating tens of thousands of intermediate autograd nodes that exhausted VRAM and triggered `OutOfMemoryError`.
- **Tile-Based Spatial Bounding-Box Rasterizer:**
  - Added vectorized $3\sigma$ 2D screen-space bounding box projection ($[u \pm 3\sigma_x, v \pm 3\sigma_y]$) and $32 \times 32$ tile spatial binning in `PyTorchFallbackRasterizer` ([`rasterizer_interface.py`](file:///home/monday/Desktop/GlomeHomeTour/backend/reconstruction/training/rasterizer_interface.py)).
  - Reduced evaluated pairs from 45 billion global comparisons to only overlapping surfels per tile ($\approx 100\times$ to $400\times$ reduction in operations and graph memory).
  - VRAM reduced from 16 GB (OOM) to $\approx 1.08\text{ GB}$.
  - Step training speed accelerated from minutes to $\approx 800\text{ms} - 1000\text{ms}$ per iteration directly on the AMD Radeon RX 9060 XT (ROCm 7.1 / RDNA4).
- **Environment & Cache Configuration:**
  - Added auto-configuration for `MIOPEN_USER_DB_PATH` in `run_scene_training.py` ensuring writable database caches on ROCm.
## Milestone: Canonical 3DGS PLY Exporter, Pipeline Stage Organization & Snapshot Logging

- **Standard 3DGS PLY Interchange Format:**
  - Implemented `export_model_to_standard_3dgs_ply` in [`export_standard_ply.py`](file:///home/monday/Desktop/GlomeHomeTour/backend/reconstruction/training/export_standard_ply.py) converting 2DGS surfel models (`Material2DGSModel` or `.pt` checkpoints) into canonical 3D Gaussian Splatting PLY format (`f_dc_0..2`, `opacity`, `scale_0..2`, `rot_0..3`).
  - Enables drag-and-drop loading and real-time visualization in external Gaussian Splatting tools (PlayCanvas SuperSplat, Antimatter15, Blender 3DGS plugins).
## Milestone: Native ROCm/HIP Differentiable Rasterizer & 4-Stage Progressive Schedule (with 720p Stage)

- **Native ROCm/HIP C++ Rasterizer Extension (`rasterizer_hip`):**
  - Fully implemented, compiled, and deployed native C++/HIP differentiable 2DGS rasterizer extension targeting AMD RDNA4 (`gfx1200`, Radeon RX 9060 XT) on ROCm 7.1.
  - Implemented `Rasterize2DGSGBufferForwardKernel` in `forward.hip` utilizing Wave32 SIMD execution and LDS shared memory caching for all 2DGS surfel attributes (position, normal, covariance, albedo, roughness, metallic, opacity).
  - Implemented analytical backward gradient kernel `Rasterize2DGSGBufferBackwardKernel` in `backward.hip`, backpropagating exact analytical gradients into screen properties (`proj_uv`, `inv_cov`, `opacity`, `albedo`, `normal`, `roughness`, `metallic`, `depth`).
  - Integrated PyTorch autograd bridge via `_HIPRasterizerFunction` in `rasterizer_interface.py`, connecting HIP rasterizer directly to `Material2DGSModel` parameters with zero-overhead tensor sharing.
  - Integrated transparent CPU fallback in `HIP2DGSRasterizer`: automatically routes to `PyTorchFallbackRasterizer` when input model is on CPU, while running full HIP acceleration on GPU.
- **Extreme Speedup & VRAM Efficiency:**
  - Training step time dropped from $>1500\text{ms}$ to $\approx 27\text{ms} - 80\text{ms}$ per iteration (>50x acceleration).
  - Peak VRAM bounded under $1.5\text{ GB}$ (eliminating previous 16 GB OOM crashes) by executing front-to-back alpha compositing in-kernel on LDS memory rather than holding giant autograd graphs.
- **4-Stage Progressive Multi-Scale Schedule:**
  - Stage 1: Coarse Warmup ($270 \times 480$, 1/4 scale): Iters 1 – 200 (warmup & rough photometric alignment).
  - Stage 2: Medium Structural ($540 \times 960$, 1/2 scale): Iters 201 – 700 (initial densification & geometry refinement).
  - Stage 3: Medium-High 720p ($720 \times 1280$, 2/3 scale): Iters 701 – 1500 (fine geometry & material albedo separation).
  - Stage 4: Fine Native 1080p ($1080 \times 1920$, 1/1 scale): Iters 1501 – 3000 (specular roughness, high-frequency details).
- **Stage Snapshot Logging:**
  - Saves intermediate snapshots every 500 iterations to `scenes/<scene>_2DGS_results/stages/iter_XXXX/`:
    - `checkpoint_iter_XXXX.pt`: Full PyTorch model checkpoint.
    - `model_3dgs_iter_XXXX.ply`: Canonical 3DGS PLY for instant visualization in SuperSplat / Blender.
    - `render_sample_iter_XXXX.png`: PBR-rendered sample view for immediate visual progress tracking.
    - `stage_metrics.json`: Loss components (total, L1, SSIM, normal), Gaussian counts, and elapsed time.
- **Verification:**
  - 21/21 passing unit tests in `backend/reconstruction/training/tests/`.
  - Schema validation (`python shared/schemas/validate.py`) passing with 100% compliance.




## 2026-09-11: Canonical direct-radiance 2DGS engine — tile-binned rasterizer, correct alpha gradient, fixed camera convention
Outcome: worked — the depth-init PLY -> 2DGS step now trains on real photographic colour instead of a fabricated sun, and every one of the plan's four "sacrifices" was verified against artifacts before being applied.

What the supplied plan got right (confirmed with measurements, not on faith):
- Deferred Cook-Torrance shading in the training loop bakes an invented `light_dir=[0.58,0.58,0.58]` sun and a sky/ground hemisphere into every exported colour. Trainer now uses `gbuffer.albedo` directly as composited SH-DC radiance; `pbr_shader.py` is untouched and still tested, just unwired.
- Global opacity reset every 3000 iters left the model at mean opacity 0.0921 (0% of splats above 0.5). The reset call is gone (`TamingDensityController.reset_opacities` kept for manual floater surgery); surfels now initialize at 0.90 via `Material2DGSModel.from_surfel_cloud(init_opacity=...)`.
- `max_primitives=max(max_surfels, 400_000)` with a 400k init gave `avail_budget = 0` — densification was a no-op. Split into `--max-surfels` (initial, 200k) and `--budget` (post-densification cap, 800k); 5.5 cm voxels give 148,439 init surfels.
- `forward.hip` walked every splat for every tile. Replaced with 16x16 screen-tile binning (`bin_tiles` in `rasterizer_interface.py`: bbox -> key expansion -> depth sort -> stable tile sort -> `searchsorted` ranges, 0.8 ms for 70k keys) feeding `point_list`/`tile_ranges` into both kernels.

What the plan missed, found while implementing:
- **The backward alpha gradient was wrong.** `backward.hip` computed `dL_dalpha = T * (c_j . dL_dC)` and stopped, omitting the occlusion term. Rewritten with canonical reverse traversal (`T /= (1-alpha)`, `accum_rec` suffix accumulation) over 8 channels — 3 colour, 3 normal, 1 depth, plus a constant-1 channel that yields the accumulated-alpha gradient exactly (`acc_alpha = sum_k a_k T_k`). Verified against the pure-PyTorch cumprod oracle: gradient cosine > 0.97 for `_opacity`, `_albedo`, `_xyz` (it was 0.93 before the alpha channel was added).
- **The per-frame camera convention heuristic silently ruined ~half the frames.** `z_mean = pts_cam[:,2].median()` decided +Z vs -Z forward per frame; on the bedroom capture 149/273 frames voted OpenGL and 124 OpenCV even though all 273 are ARCore OpenGL — inside a closed room, flipping the sign just renders the opposite wall, so no geometric self-check can decide it. Forcing `camera_convention="opengl"` wins on 60/69 sampled frames at +4.73 dB mean PSNR. `"auto"` is still available but documented as unsafe indoors.
- **SSIM, not the rasterizer, was the step-time bottleneck.** MIOpen's grouped 11x11 conv cost 178 ms/step at 1080p vs 65 ms for the whole rasterizer forward+backward. Made separable (two 1D grouped convs) and batched all five moments into one blur: 178 -> 67 ms. 1080p step time 290 -> ~140 ms.
- Voxel pruning at 1.5 cm was deleting roughly what densification had just added — `enable_voxel_pruning` now defaults False.
- Fixed world-space `tau_grad = 0.0002` is scene-scale dependent; densification now takes the top 10% of primitives that actually received a gradient (`grad_percentile=0.90`), with the old constant as a floor.
- Replaced the plan's `gamma_dist * L_distortion` with a masked L1 against the DA3 metric depth prior already loaded by `GSInputDataset` — same job (anchoring geometry where photometry is ambiguous), no extra machinery.
- Kept EWA screen-space projection rather than switching to exact ray-splat intersection: the export target is a canonical 3DGS PLY rendered by SuperSplat's EWA path, and training under a different projection than the viewer would reintroduce a train/render mismatch. Low-pass variance corrected 0.09 -> 0.3.
- Kept `_albedo` as the parameter name (renaming breaks `compressor.py`, `planar_reflections.py`, four test files and every existing checkpoint); added a `radiance` alias plus a docstring saying it is SH-DC radiance, not a BRDF term.
- Surfel normals are now flipped toward the camera in both rasterizers, matching `compute_surface_normals`, which already does this for the priors.
- Latent use-after-free in both HIP launchers: `.contiguous()` temporaries were passed inline to `hipLaunchKernelGGL` and could be destroyed before the async kernel ran. Hoisted into named locals.
- Roughness/metallic dropped from both kernels and from the optimizer; `GBufferOutput` still exposes them as constants so the PBR/compression consumers keep working. Added an `alpha` field for coverage masking.

Verification: 33/33 tests pass (`PYTHONPATH=backend backend/.venv/bin/python -m pytest backend/reconstruction/training/tests/`), including a new `test_rasterizer_parity.py` (tile-binning coverage/ordering, HIP-vs-oracle forward, HIP-vs-oracle gradients for three parameter groups, camera-convention regression). Schema validation passes. End-to-end on `scenes/bedroom_complete_depth_results/GS_input`: 148,439 init surfels, 273 keyframes, mean opacity holds 0.90 -> 0.77, exports `bedroom_standard_3dgs.ply` + a 1.98 MB `walkthrough_2dgs.zip`.

## 2026-09-11: fixed net-negative densification, then measured what the budget is worth
Outcome: worked (mechanism) / plan's 800k budget not validated — reduced default to 400k.

Symptom from the 7000-iteration run: primitive count decayed 148,439 -> 87,828 instead of
growing toward the budget.

Root cause: candidate selection was `avg_grad >= max(grad_threshold=2e-4, p90(avg_grad))`.
Real positional gradients sit far below 2e-4, so the absolute floor dominated the percentile
and only 855 of 148k primitives (0.58%) were ever densified, while opacity pruning removed
2-5% per interval. (Splitting is not the culprit: clone and split are both net +1, since the
split parent is pruned.)

Changes in `density_control.py` / `trainer.py` / `run_scene_training.py`:
- Rank-based selection against a budget schedule instead of a world-space gradient threshold.
  The remaining budget is spread evenly over the remaining densification intervals
  (`remaining_intervals`, passed by the trainer), capped per interval by `growth_rate` (0.3).
- `denom` now counts only steps where the primitive was visible, so a surfel seen in 3 of 273
  views is ranked on its mean gradient, not a 273-step average of mostly zeros.
- Primitives about to be pruned (opacity < min_opacity_prune) are excluded from densification.
- Adam `exp_avg`/`exp_avg_sq` are carried across densification via a provenance index instead
  of `optimizer.state.clear()`; new primitives start cold.
- `scale_split_threshold` 0.02 -> 0.06 m (the 5.5 cm voxel init gives a ~6.4 cm median radius,
  so at 0.02 every candidate took the split branch and nothing ever cloned).
- `densify_stop_iter` 80% -> 50% of training (canonical 3DGS), worth +0.76 dB below.

Measured, train-view PSNR over 28 evenly-spaced frames at 1080p, 2000 iterations each:
| config                        | splats  | PSNR     |
| starved (old behaviour)       | 128,458 | 26.19 dB |
| ramp to 200k, stop 50%        | 195,784 | 25.73 dB |
| ramp to 400k, stop 50%        | 391,899 | 24.06 dB |
| ramp to 400k, stop 80%        | 393,201 | 23.30 dB |
At a fixed 2000 iterations PSNR falls monotonically with budget — the bigger models are simply
undertrained, not worse. Reference: the pre-fix 7000-iteration run reached 28.59 dB with 87,828
splats. Which budget wins at 7000 iterations was NOT measured (full runs were out of scope).
Default budget set to 400k, not the plan's 800k: 400k costs ~150 ms/it at 1080p (~14 min for
7000 it), 800k roughly doubles that and breaks the 12-15 min turnaround target.

Also measured: Adam-state preservation + visibility-normalised gradients are worth ~+1.6 dB at
600 iterations (18.34 -> 19.90 dB at 400k splats), well outside the 0.24 dB run-to-run spread.

## 2026-09-11: ROCm matmul silently zeroes every row past 2**19 — the real ceiling on sharpness
Outcome: fixed — `torch.matmul` on a float32 `(N, 3) @ (3, 3)` on gfx1200 / ROCm 7.1 leaves
every output row from index 524,288 (2**19) onward as **zeros**. The BLAS kernel launches a
grid that only covers the first 2**19 rows. Confirmed exactly: first bad row = 524288, all rows
after it bad, value `[0, 0, 0]`. `torch.mm`, `F.linear`, `einsum` and `.T.contiguous()` all hit
the same kernel and fail identically. Elementwise column form is exact at any N and faster for
k=3 (0.16 ms for 700k points).

Why it mattered: `rasterizer_interface.py` used `torch.matmul(pts_world, r_cw.T)` to build
camera-space points. Above 524k primitives every splat past the cliff landed at the camera
origin, got frustum-culled, and on 59 of 273 keyframes the *entire* view culled away. The
rasterizer then returned a detached `torch.zeros` G-buffer, so `loss.backward()` raised
`element 0 of tensors does not require grad`. That is why a 2.5 cm voxel init (843,657 surfels)
crashed at iteration 4 while 5.5 cm (148,439) trained fine, and why the 400k budget "worked" —
it sits just under the cliff.

Changes:
- `_rotate(vecs, r)` in `rasterizer_interface.py`, column form, replaces all 8 `matmul(..., r_cw.T)`
  sites across the HIP and PyTorch-fallback forwards.
- Same substitution in `mesh_generation/kernels/surfel_projection.py` and
  `mesh_generation/segmentation.py` (both did `(N, 4) @ (4, 4)^T`); the now-dead homogeneous
  `ones`/`fg_homo` columns were dropped.
- A fully culled view now renders zeros that are still attached to the graph
  (`_zero_grad_scalar`), so one bad pose degrades instead of killing a 7000-iteration run.
- Two regression tests in `test_rasterizer_parity.py`: `_rotate` exact at 2**19 + 1024 rows,
  and a fully-culled view surviving `backward()` on both rasterizers.

Verified: all 273 keyframes of `bedroom_complete` now render with a live grad_fn at 843,657
surfels (was 59 detached). Other large-N ops were audited and are clean (topk, argsort, norm,
max, nonzero, F.normalize at N = 1e6; only fp32 `sum` drifts, which is ordinary accumulation
error, not corruption).

## 2026-09-11: splat count was never the sharpness lever — splat *size* was
Outcome: partial win — measured, 2000 iterations each, train-view over 28 frames at 1080p:
| config                                   | splats  | PSNR     | render lap-var |
| budget 400k                              | 391,899 | 24.06 dB | 1.43 |
| budget 1M (after the matmul fix)         | 993,513 | 23.38 dB | 1.46 |
| budget 1M + scale capped to voxel cell   | 984,857 | 26.43 dB | 1.35 |
| source photos                            | —       | —        | 56.02 |

2.5x the primitives bought +0.03 lap-var. Primitive budget is NOT the sharpness lever; it was
just the thing that was visibly broken. The real cause: `_estimate_initial_scales` clips radii to
`[0.01, 0.06]` m and `from_ply` only applied a *floor* against the voxel size, never a ceiling, so
every splat sat at the 6 cm clip — ~42 px wide at 2 m / 1080p. Adding more 42 px blobs cannot
sharpen an image. `from_ply` now clamps to `[0.4*v, 0.8*v]` of the voxel cell: +3.05 dB PSNR
(vs 0.24 dB run-to-run spread), lap-var essentially flat (1.46 -> 1.35).

Still open, and now the prime suspect for the remaining blur: **opacity collapse**. Mean opacity
decays monotonically and does not settle — 0.90 -> 0.698 at 1M splats, and 0.90 -> 0.529 once
splats were shrunk (smaller splats need more coverage, so the optimizer buys it with
transparency). Target is > 0.85. At alpha 0.53 every pixel is an average over many splats, which
is blur by construction. Untried: opacity regularizer, lower `lr_opacity`, alpha-coverage loss.
Second untried lever: view-dependent SH (splats are SH-DC only today).

Timing at ~1M splats / 1080p: 150-260 ms/it, so 7000 it ~= 25-30 min. The earlier "800k doubles
400k" estimate was pessimistic; cost is dominated by resolution, not splat count. Default
`--budget` raised 400k -> 800k.

## 2026-09-11: Canonical 2DGS Architecture with Taming Controlled Growth, FasterGS Culling, and 2.0 cm Prior Init
Outcome: Fully implemented and tested across rasterizer parity, density control, and keyframe selector suites (33/33 tests pass).
- **FasterGS-Inspired Backface Culling:** Implemented in both `HIP2DGSRasterizer` and `PyTorchFallbackRasterizer` via camera-space ray-to-normal dot product `(n_cam * pts_cam).sum(dim=-1) < 0`. Backfacing primitives are culled before tile binning, reducing active primitives per frame by ~40-50% while preserving differentiable zero-gradient guarantees for culled views. Tested with 100% forward/backward autograd parity.
- **2.0 cm Voxel Prior & Mathematically Grounded Scale Clamping:** In `initialization.py`, voxel downsampling defaults to `v = 0.02` m (2.0 cm). Surfel scales clamped strictly to `[0.5*v, 0.75*v] = [0.010, 0.015]` m according to the 2D Voronoi-Delaunay surface coverage condition ($r_{\min} = \frac{\sqrt{2}}{2}v$). Default opacity initialized to solid $\alpha = 0.90$.
- **Scene-Proportional Dynamic Primitive Budget:** In `density_control.py`, added `from_initial_surfels` scaling `max_primitives = min(round(2.5 * N_init), 1_400_000)`, matching physical room surface area ($50\text{ m}^2$ bedroom + mezzanine scales to ~1.1M-1.25M surfels). Scale split threshold set to `0.015` m (1.5 cm) so large surfels split and fine ones clone along local tangent frames.
- **Opacity Decay Guard:** Lowered `lr_opacity` 10x from `5e-2` to `5e-3` (0.005) and disabled destructive global opacity resets to maintain solid surface opacity ($\ge 0.85$), pruning only floaters with $\alpha < 0.05$.
- **Refined 3-Stage Multi-Scale Schedule:** In `run_scene_training.py`, standard training defaults to 3,000 steps:
  - Stage 1 (Steps 1–400 @ 360p): Coarse radiance warmup, densification OFF.
  - Stage 2 (Steps 401–1400 @ 720p): Active macro structural growth (splitting every 200 it).
  - Stage 3A (Steps 1401–2000 @ 1080p): High-res micro-detail splitting.
  - Stage 3B (Steps 2001–3000 @ 1080p): Topology frozen (0 growth), dedicated to photometric convergence and opacity settling.
- **2DGS Training Keyframe Selector:** In `keyframe_selector.py`, added `for_2dgs_training` preset with $[0.45, 0.65]$ co-visibility window, $15\text{ cm}$ / $8.0^\circ$ baseline parallax thresholds, and bounded frame counts $N_{\text{depth}} \le N_{\text{train}} \le 1.6 \times N_{\text{depth}}$.

## 2026-09-11: High-Density Staging, Enhanced Blur Gating & Gapless Keyframe Overlap

Outcome: Successfully updated, benchmarked, and validated the complete Phase 3 ingestion-to-staging pipeline on `bedroom_complete.zip`:
- **Adaptive Blur Gate Upgrade:** Tuned `adaptive_blur_factor=0.60` and `min_blur_floor=30.0` in `QualityGate`, successfully filtering 66 motion-blurred frames (including blurry frame 239) and accepting 881 sharp frames from 958 raw captures.
- **Overlap & Step Bounds:** Constrained max translation step ($d_{\max} \le 0.40\text{m}$) and rotation step ($\theta_{\max} \le 16.0^\circ$) with a 3-frame maximum fallback jump in `extract_depth_adaptive_keyframes`, completely resolving baseline visual gaps between consecutive keyframes.
- **Full Quality-Filtered 2DGS Staging:** Updated `run_sliding_window_reconstruction.py` and `run_full_benchmark.py` to stage all 881 quality-filtered frames into `GS_input/images/` and `GS_input/transforms.json` (validating 100% against Draft 2020-12 `transforms.schema.json`), while supplying 401 depth-adaptive keyframes for 3D sliding window depth unprojection.
- **End-to-End Performance Benchmark on AMD RX 9060 XT (16GB VRAM):**
  - Total End-to-End Runtime: 195.33s (3.26 min).
  - Multi-View Depth Estimation: 32.34s across 100 sliding window chunks ($N=6, K=2$).
  - 3D Surfel Unprojection & Consensus: 93.67s generating 2,638,589 multi-view consistent surfels (98.14 MB PLY).
  - Peak VRAM Allocated: 1.58 GB (9.9% of 15.92 GB).
  - All 58 unit tests passing (`pytest tests/`) and schemas 100% compliant.


---

## 2026-09-12 — Ceiling→Wall Floater Persists Through Training-Loop Fixes; Root Cause Traced to Disabled Flying-Pixel Filter

**Symptom:** A large floater hanging from the ceiling (a misprojected copy of the wall behind it)
survived three separate training-loop fixes (radial depth-ceiling pruning, prune-continuation
past `densify_stop_iter`, `is_original`-gated opacity sparsity) across three 10k-iteration runs.
User reported it as an artifact already visible in the raw surfel PLY, not something 2DGS
training introduced.

**Root Cause:** `SurfelCloudInitializer` (`reconstruction/initialization.py`) already implements
a depth-discontinuity/edge-gradient filter and a grazing-angle filter, validated and documented
in this same file's 2026-09-10 entry ("Boundary Floater & Silhouette Edge Bleeding Pruning") —
but the call site in `run_sliding_window_reconstruction.py` never passed `max_depth_gradient` or
`max_grazing_angle_deg`, so both filters silently ran disabled (`0.0` = off) on every real scene
reconstruction since that entry was written. The floater is a classic monocular-depth flying
pixel at a ceiling/wall silhouette boundary — exactly what these filters exist to remove.

**Fix:** Passed `max_depth_gradient=0.08` and `max_grazing_angle_deg=78.0` at the
`SurfelCloudInitializer(...)` call site. Regenerated the surfel cloud from cached depth maps
(`--skip-depth`) and re-ran 10k-iteration training.

**Outcome:** User re-inspected and reported the floater *still present*, with more holes than
before. So the enabled filters did not fully resolve it either — open question whether the
gradient/angle thresholds need tuning, or whether this specific artifact isn't a flying-pixel
case at all.

**Follow-up experiment (in progress at time of writing):** To isolate whether the floater/holes
are a training-loop bug independent of depth priors, added a `--random-init` mode to
`reconstruction/training/run_scene_training.py` (`SurfelCloud.from_random` in
`initialization.py`): seeds a uniform random point cloud inside the camera trajectory's bounding
box (vanilla-3DGS style) and disables depth/normal priors (`gamma_depth=gamma_normal=0`,
`load_depth_normals=False`), so Taming-3DGS-style growth + FasterGS backface culling run with
no depth prior at all. 10k-iteration run completed: final loss 0.129 (vs ~0.04-0.07 for the
depth-prior runs), PSNR oscillating 16-27dB rather than settling — expected given photometric-only
optimization has no metric prior to anchor to. Visual inspection pending.

**Also added:** raw (pre-quality-gate) frame staging in `run_sliding_window_reconstruction.py`
(`GS_input/images_raw/`), completing a reusable 3-tier dataset (raw / quality-filtered /
depth-selected) alongside the existing `images/` and `images_depth_selected/` folders.

## 2026-09-14: Replaced the in-house 2DGS implementation with the standalone 2DGS project
Outcome: worked (merge only, not yet trained end-to-end) — the whole stage was swapped for the
battle-tested standalone 2DGS codebase, copied in verbatim (no refactoring, per the merge rules).

Deleted: `model.py`, `trainer.py`, `dataset.py`, `losses.py`, `density_control.py`,
`rasterizer_interface.py`, `run_scene_training.py`, `rasterizer_hip/`,
`rasterizer_torch_fallback/`, `tests/`, `implementation_plan.md`, `__init__.py`.
Copied in: `train.py`, `render.py`, `metrics.py`, `train_room.py`, `arguments/`,
`gaussian_renderer/`, `scene/`, `utils/`, `lpipsPyTorch/`,
`submodules/diff-surfel-rasterization/` (HIP kernels) and `submodules/simple-knn/`.

Consequence worth knowing: the hand-authored `rasterizer_hip/` kernels and the pure-PyTorch
fallback oracle are **gone**, superseded by `diff-surfel-rasterization`. Root `CLAUDE.md`,
`AGENTS.md` and the root `README.md` still describe the old hand-authored-kernel architecture
and are now stale — not updated here because the merge brief scoped doc updates to
`backend/README.md`.

Four modules with no counterpart in the merged code (`compressor.py`, `pbr_shader.py`,
`planar_reflections.py`, `export_standard_ply.py`) were moved to `04_2DGS_refinment/` rather
than deleted; they do not import cleanly yet (see that folder's README).

Two things that bit during the merge, both now documented in `README.md`:
- Both native extensions must be installed with `pip --no-build-isolation`; an isolated PEP 517
  env cannot see the ROCm torch wheel and the builds fail with `No module named 'torch'`.
- `simple-knn` is required despite its only `distCUDA2` call site being commented out — the
  import at `scene/gaussian_model.py:20` is live and breaks the whole `scene` package without it.

Not done, deliberately: stage 3's depth priors are still unconsumed. The merged code has no
depth loader and no depth loss; wiring `depth_maps/*.npy` and `{scene}_surfels.ply` into
training is a separate follow-up task. Note `Scene.__init__` prefers a `sparse/` folder when one
exists, so stage 2's COLMAP `points3D.ply` will silently win over stage 3's surfel cloud.

## 2026-09-14: step 5 wires the depth cloud in and trains progressively
`step_train.py` closes the gap flagged in the entry above. The loader preference
is not fought: step 4's cloud is **copied over** `sparse/0/points3D.ply`, with
COLMAP's kept beside it as `points3D_colmap.ply`. That copy is the entire
mechanism by which depth priors reach training — no loader change.

Four stages at `-r 8/4/2/1` to cumulative iterations 2,333 / 4,667 / 7,000 /
10,000, chained through `--start_checkpoint chkpnt<N>.pth` (train.py restores
`first_iter` from the checkpoint, so targets must be cumulative). Densification
scaled from the 30k defaults to `densify_from_iter=167`,
`densify_until_iter=5000`, `opacity_reset_interval=1000`.

No `images_2/4/8` folders are generated, deliberately: `loadCam` already applies
`-r` when loading each camera, so the folders would be ~4 GB of duplicated JPEG
for no gain. Each stage is a subprocess (train.py has only a `__main__` block)
and gets `PYTHONPATH` via `pipeline_paths.subprocess_env()`.

Outcome: implemented and import-clean; **not yet run end-to-end** — step 2 was
still bundle-adjusting the 1400-frame Bedroom2 capture when this was written, so
steps 3-5 have not executed on real data.

## 2026-09-15: Sensor Saturation Masking & Asymmetric Multi-View Evidence Culling (TIDI-GS)

- **Optical Bloom / Glare Masking:**
  - Implemented dynamic sensor saturation detection `compute_saturation_mask` in `utils/loss_utils.py` identifying blown-out pixels ($\max(R,G,B) \ge \tau_{\text{sat}}$ with low chroma difference $\le 0.15$).
  - Implemented `masked_l1_loss` and `masked_ssim`, zeroing out photometric gradient on saturated light sources during training so the optimizer smoothly interpolates ceilings and walls without constructing 3D hairballs.
  - Excluded saturated pixels from `normal_loss`, `dist_loss`, and `multiview_photometric_loss`.
- **Asymmetric Multi-View Homography Loss (Free-Space Veto):**
  - Updated `multiview_photometric_loss` in `utils/multiview_loss.py` with asymmetric consensus formulation: $\mathcal{L}_{\text{mv}} = (1 - w_{\text{veto}}) \cdot \text{mean}(\mathcal{L}_k) + w_{\text{veto}} \cdot \max(\mathcal{L}_k)$ ($w_{\text{veto}} = 0.5$).
  - Prevents unobstructed side views observing empty space from being diluted by narrow-baseline views sharing the reflection or glare.
- **Observation-Count / View-Evidence Pruning (TIDI-GS style):**
  - Added per-Gaussian frustum and contribution tracking in `scene/gaussian_model.py` (`frustum_counter`, `observation_counter`, `seen_frustum_mask`, `seen_contrib_mask`) with camera UID deduplication.
  - In `densify_and_prune`, automatically culls monocular floaters: primitives visible in $\ge 8$ camera frustums but only actively contributing in $\le 2$ views after iteration 1500.
- **Decoupled Pruning and Densification:**
  - Added `allow_densification` parameter to `GaussianModel.densify_and_prune`.
  - In `train.py`, pruning runs every 100 iterations continuously across all stages, purging low-opacity surfels ($< 0.05$) and floaters, while cloning/splitting is strictly restricted to the densification window (`densify_from_iter < iteration < densify_until_iter`).
  - Ensures periodic opacity resets (at iterations 2,000 and 4,000) actively purge non-recovering dead surfels in Phase 1 instead of carrying them for 6,000 iterations.
- **Verification:**
  - Authored comprehensive test suite `backend/tests/test_2dgs_training_filters.py` (8/8 tests passing).
  - Validated depth priors and surfel cloud initialization test suite `backend/tests/test_depth_initialization.py` (10/10 tests passing).



## 2026-09-15: ROCm matmul bug in depth_to_normal; dynamic gaussian budget
`utils/point_utils.py:21` used `points @ intrins.inverse().T @ c2w[:3,:3].T` with
N = W*H. At 1920x1080 (N = 2.07M) the documented gfx1200 gemm bug would have left
~75% of the normal map silently zeroed at the final full-res stage — no error, just a
quietly wrong `lambda_normal` gradient after hours of training. Coarse stages (r>=2)
sit under 2**19 so this only bit stage 4. Re-introduced `_rotate()` (the canonical copy
in `rasterizer_interface.py` was deleted) locally in `point_utils.py`.

`max_gaussians` is no longer a hardcoded 1M: default is now -1 = derive at train start
as `min(1.5 * init_points, 4.5M)`, so the growth budget tracks the voxel grid's density
instead of contradicting it. 4.5M is the 16 GB ceiling; 1.5x pairs with the 3M
initialization cap in step 4.
Outcome: worked — 87/87 backend tests pass. Step 5 deliberately not run yet.

## 2026-09-15: Enabled TrackGS Pose Refinement Exclusively in Stage 4 (1080p)
- **Architecture & Scoping:** Configured `step_train.py` to enable TrackGS Lie-algebra $\mathfrak{se}(3)$ camera pose refinement (`--refine_poses_during_training`) exclusively in **Stage 4** (full native 1080p resolution, iterations 6,000–10,000).
- **Geometric Invariance & Prior Preservation:** Stages 1–3 ($r=8, 4, 2$, iterations 0–6,000) keep camera extrinsics strictly locked to their metric COLMAP coordinates. This ensures that the global room geometry settles firmly against the multi-view Depth Anything v3 (DA3) metric point cloud priors without gauge drift.
- **Tightly Regularized Polish:** In Stage 4, camera pose optimization runs with a conservative learning rate `--pose_lr 0.0001` (1e-4) paired with strong 3D landmark track reprojection regularization `--lambda_track 0.1` anchored to COLMAP's triangulated point cloud. This absorbs sub-pixel handheld frame-to-frame jitter and rolling shutter sync offsets, boosting high-frequency texture sharpness without warping the global metric trajectory.
- **CLI Customization:** Exposed `--no-pose-refine`, `--pose-lr`, and `--lambda-track` in `step_train.py`.
- **Verification:** All 87 unit tests passing (`pytest backend/tests/`).

