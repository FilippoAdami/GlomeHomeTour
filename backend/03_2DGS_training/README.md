# 03_2DGS_training — Stage 5: 2D Gaussian Splatting Training & Surface Reconstruction

`03_2DGS_training` reconstructs the scene as a high-fidelity 2D Gaussian Splatting (2DGS) radiance field. Unlike standard 3D volume Gaussians (ellipsoids), 2DGS models the scene as planar, surface-bound oriented surfels with explicit normals, tangent frames, and ray-splat intersection equations, yielding physically plausible, thin surface geometry without volumic fuzz.

---

## What This Stage Does

1. **Depth Prior Seeding (`step_train.py`):** Installs the high-density, filtered surfel cloud from Stage 4 (`02_depth_estimation/depth/points3D_depth.ply`) into `sparse/0/points3D.ply` (backing up COLMAP's sparse cloud as `points3D_colmap.ply`), bootstrapping the optimizer on real metric geometry.
2. **Progressive Multi-Stage Training:** Solves the scene across progressively increasing image resolutions (1/8 -> 1/4 -> 1/2 -> full) via `step_train.py` and `train_room.py`, establishing global low-frequency geometry before refining high-frequency texture.
3. **Decoupled Pruning & Surfel Densification:** Continuously purges dead surfels (< 0.05 opacity) and monocular floaters every 100 iterations across the entire run, while strictly controlling cloning and splitting during fine detail stages.
4. **Sensor Saturation & Glare Masking:** Dynamically masks overexposed light sources to prevent the optimizer from generating 3D floaters and hairballs around light fixtures.
5. **Asymmetric Multi-View Free-Space Veto:** Enforces multi-view photometric consistency with a maximum-error veto, ensuring wide-baseline views penalize geometry floating in empty space.
6. **Hardware-Accelerated Rasterization:** Employs hand-authored HIP kernels (`diff-surfel-rasterization`) compiled natively for AMD ROCm (targeting RDNA4 / `gfx1200`).
7. **View Synthesis & TSDF Extraction:** Renders novel camera trajectories and computes truncated signed distance fields (TSDF) for surface verification via `render.py`.

---

## Modules & Architecture

| Module / Directory | Role |
| --- | --- |
| [`step_train.py`](step_train.py) | **Pipeline Step 5 Entry Point**: Manages init cloud installation, progressive resolution stages (2.5k / 4.5k / 6k / 10k), and training checkpointing. |
| [`train.py`](train.py) | Core 2DGS training loop, masked loss computation, densification, and pruning. |
| [`train_room.py`](train_room.py) | Standalone progressive multi-stage training coordinator ($1/8 \to 1/4 \to 1/2 \to \text{full}$). |
| [`render.py`](render.py) | View synthesis evaluation and Open3D TSDF mesh extraction. |
| [`metrics.py`](metrics.py) | Full-reference quality metrics: PSNR, SSIM, and LPIPS (VGG). |
| [`arguments/`](arguments/) | Parameter definitions (`ModelParams`, `OptimizationParams`, `PipelineParams`). |
| [`gaussian_renderer/`](gaussian_renderer/) | Differentiable surfel rasterization bindings and network GUI connector. |
| [`scene/`](scene/) | `Scene`, `Camera`, `GaussianModel`, dataset readers, and COLMAP loader. |
| [`utils/`](utils/) | Loss functions (`loss_utils.py`), multi-view regularization (`multiview_loss.py`), graphics helpers. |
| [`submodules/diff-surfel-rasterization`](submodules/diff-surfel-rasterization) | Differential 2D surfel rasterizer (HIP source code for ROCm). |
| [`submodules/simple-knn`](submodules/simple-knn) | Spatial nearest-neighbor index for scale initialization. |

---

## Techniques & Mathematical Specifications

### 1. 2D Gaussian Surfel Representation
Each surfel is defined by:
- Center position $p_k \in \mathbb{R}^3$
- Tangent vectors $u_k, v_k \in \mathbb{R}^3$ and unit surface normal $n_k = u_k \times v_k$
- 2D scaling factors $(\sigma_{u}, \sigma_{v}) \in \mathbb{R}^2$
- Opacity $\alpha_k \in [0, 1]$
- Spherical Harmonics (SH) color coefficients $c_k$ up to degree 3

The local 2D covariance matrix in the tangent plane is:
$$\Sigma_{2D} = \begin{bmatrix} \sigma_u^2 & 0 \\ 0 & \sigma_v^2 \end{bmatrix}$$

### 2. Sensor Saturation & Glare Masking
Overexposed light sources (ceiling lamps, pendant lights, direct sunlight through windows) produce flat 2D glare gradients that cause standard 3DGS/2DGS optimizers to spawn dense clusters of floating surfels.
- **Dynamic Masking:** `compute_saturation_mask` dynamically evaluates overexposure:
  $$\tau_{\text{dyn}} = \operatorname{clip}(\operatorname{quantile}_{99.8}(\max(R,G,B)),\, 0.98,\, 0.995)$$
  with a chroma difference check $\max(R,G,B) - \min(R,G,B) \le 0.15$ (distinguishing achromatic optical bloom from vibrant saturated colors).
- **Masked Photometric Loss:** Evaluates $\mathcal{L}_1^{\text{masked}}$ and $\mathcal{L}_{\text{SSIM}}^{\text{masked}}$ strictly on valid, unmasked pixels.
- **Regularization Masking:** Excludes saturated pixels from `normal_loss`, `dist_loss`, and `multiview_photometric_loss`. This allows clean adjacent views to smoothly extrapolate ceilings and walls behind light fixtures without fighting glare gradients.

### 3. Asymmetric Multi-View Homography Loss (Free-Space Veto)
In `multiview_photometric_loss`, rendered depth and normals induce local planar homographies between the active view and neighboring keyframes.
- **Asymmetric Consensus Formulation:**
  $$\mathcal{L}_{\text{mv}} = (1 - w_{\text{veto}}) \cdot \frac{1}{K}\sum_{k=1}^K \mathcal{L}_k + w_{\text{veto}} \cdot \max_{k=1\dots K} \mathcal{L}_k$$
  *(default $w_{\text{veto}} = 0.5$).*
- **Why It Works:** If a floater appears plausible from one narrow-baseline angle (e.g. sharing specular bloom), but an unobstructed wide-baseline view sees empty space, the maximum-error term spikes, vetoing the impossible geometry instead of diluting it in a mean.
- **Saturation Exclusion:** Homography sampling and neighbor reprojected coordinates automatically exclude saturated glare regions.

### 4. Observation-Count & View-Evidence Culling (TIDI-GS style)
Tracks per-Gaussian viewing evidence across training with camera UID deduplication:
- `frustum_counter`: Number of distinct camera frames where the Gaussian was inside the camera frustum (`radii > 0`).
- `observation_counter`: Number of distinct camera frames where the Gaussian actively contributed ($\|\nabla_{\text{viewspace}}\| > 10^{-6}$).
- **Floater Culling Rule:** In `densify_and_prune` (active from iteration 1,500):
  $$\text{Prune if } \text{frustum\_counter} \ge \tau_{\text{frustum}} \quad \text{AND} \quad \text{observation\_counter} \le \tau_{\text{min\_obs}}$$
  *(defaults: $\tau_{\text{frustum}} = 8, \tau_{\text{min\_obs}} = 2$).*
- **Effect:** Monocular floaters and transient haze that are visible in many camera frustums but only used by 1–2 views to overfit transient reflections are automatically purged.

### 5. Decoupled Pruning and Densification
In `train.py` and `GaussianModel.densify_and_prune`:
- **Pruning:** Runs continuously every `densification_interval = 100` iterations across the entire schedule:
  - Opacity threshold cull: $\alpha < 0.05$.
  - View-evidence floater cull ($\ge 8$ frustums, $\le 2$ obs).
  - Spatial extent and screen-size limit culling.
  - Taming3DGS budget cap: sorts by opacity and trims excess over `max_gaussians` (1M).
- **Densification (Clone & Split):** Controlled by `allow_densification` (`densify_from_iter < iteration < densify_until_iter`).
- **Phase 1 Fitting Benefit:** In `step_train.py` Phase 1 (iters 0–6,000), clone/split is disabled to protect the pre-initialized 1.5 cm depth cloud, while pruning remains active. At iterations 2,000 and 4,000, `reset_opacity()` clamps opacities to 0.01; surfels that fail to recover within 100 iterations are immediately eliminated at iterations 2,100 and 4,100, saving ~15–20% compute and VRAM.

### 6. Background Randomization
A fresh random background color is sampled per iteration (`random_background=True`), forcing transparent floaters in front of empty space to generate error spikes that drive their opacities to zero.

---

## Progressive Resolution Schedule (`step_train.py`)

`step_train.py` executes a four-stage cumulative 10,000-iteration schedule:
1. **Stage 1 (1/8 resolution, iter 0–2,500):** Rapid global spatial anchoring on downscaled images (poses locked).
2. **Stage 2 (1/4 resolution, iter 2,500–4,500):** Coarse architectural alignment, normal consistency, view-evidence floater culling (poses locked).
3. **Stage 3 (1/2 resolution, iter 4,500–6,000):** Room-scale geometric convergence; Phase 1 finishes with clean, pruned surfels (poses locked).
4. **Stage 4 (Full 1x resolution 1080p, iter 6,000–10,000):** Phase 2 fine densification on high-gradient edges, texture refinement, color SH optimization, and **TrackGS-style camera pose refinement** (`--refine_poses_during_training`, `--pose_lr 0.0001`, `--lambda_track 0.1`) to absorb sub-pixel handheld frame-to-frame jitter without deforming global metric geometry.

---

## Hardware Target & Native Build

- **Target:** AMD Radeon RX 9060 XT (RDNA4, 16 GB VRAM, `gfx1200` ISA).
- **Environment:** PyTorch with ROCm 7.1 (`diff-surfel-rasterization`).

Build the native submodules via:
```bash
backend/.venv/bin/pip install --no-build-isolation \
    backend/03_2DGS_training/submodules/diff-surfel-rasterization \
    backend/03_2DGS_training/submodules/simple-knn
```

---

## Inputs & Outputs

- **Input:**
  - `<workspace>/images/`: Portrait keyframe images.
  - `<workspace>/sparse/0/`: COLMAP camera poses.
  - `<workspace>/02_depth_estimation/depth/points3D_depth.ply`: Filtered surfel point cloud (seeded as `sparse/0/points3D.ply`).
- **Output:**
  - `<workspace>/03_2DGS_training/2dgs/point_cloud/iteration_10000/point_cloud.ply`: Trained 2DGS model checkpoint.
  - Novel view renders and full-reference PSNR/SSIM/LPIPS evaluation metrics.

## Usage

```bash
# Standard pipeline Step 5 execution:
python 03_2DGS_training/step_train.py --workspace backend/current_scene

# Standalone progressive room training:
python 03_2DGS_training/train_room.py -s backend/current_scene -m backend/current_scene/output
```
