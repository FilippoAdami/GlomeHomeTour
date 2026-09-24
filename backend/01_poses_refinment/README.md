# 01_poses_refinment — Stage 2: COLMAP Pose Refinement & Triangulation

`01_poses_refinment` bridges initial VIO poses from Stage 1 (`transforms.json`) with high-precision multi-view geometry. It constructs a metric, geometrically consistent COLMAP reconstruction (`sparse/0/`) by triangulating visual tracks against fixed or prior-bounded camera poses.

---

## What This Stage Does

1. **Feature Extraction & Sequential Matching:** Extracts SIFT features from upright portrait keyframes and matches them sequentially across temporal windows with vocabulary tree verification.
2. **Fixed-Pose Triangulation (`point_triangulator`):** Holds ARCore VIO poses fixed as ground truth and triangulates 3D points directly. This prevents drift without re-estimating camera trajectories from scratch.
3. **Optional Pose-Prior Bundle Adjustment (`pose_prior_mapper`):** When VIO drift is detected, optimizes camera poses under tight Bayesian priors centered on the VIO trajectory.
4. **Epipolar & Reprojection Diagnostics:** Evaluates Sampson epipolar error and multi-view track statistics to detect registration failures or optical inconsistencies.
5. **Dense Stereo Fallback (`densify_pointcloud.py`):** Provides a PyTorch-native multi-view plane-sweep dense stereo implementation for AMD ROCm environments where CUDA-dependent dense COLMAP binaries cannot run.

---

## Modules & Architecture

| Module | Role |
| --- | --- |
| [`step_colmap.py`](step_colmap.py) | **Pipeline Step 2 Entry Point**: Runs SIFT extraction, matching, and fixed-pose triangulation into `<workspace>/sparse/0/`. |
| [`convert_transforms_to_colmap.py`](convert_transforms_to_colmap.py) | Lower-level bridge script executing COLMAP subcommands (`feature_extractor`, `sequential_matcher`, `point_triangulator`, `pose_prior_mapper`). |
| [`colmap_diagnostics.py`](colmap_diagnostics.py) | Computes reprojection errors, Sampson distance statistics, and verifies 3D point track distributions. |
| [`densify_pointcloud.py`](densify_pointcloud.py) | PyTorch-based plane-sweep multi-view stereo densifier (ROCm/HIP compatible) as fallback when CUDA COLMAP dense stereo is unavailable. |
| [`select_keyframes.py`](select_keyframes.py) | Subsamples keyframes based on baseline and viewing angle spread. |
| [`export_keyframes.py`](export_keyframes.py) | Exports keyframe subsets for specialized inspection or training splits. |
| [`visualize_camera_path.py`](visualize_camera_path.py) | Visualizes 3D camera trajectory paths and spatial distribution of triangulated sparse points. |
| [`convert.py`](convert.py) | Standard 3DGS convenience wrapper for running COLMAP directly. |

---

## Techniques & Mathematical Specifications

### 1. Fixed-Pose Triangulation
Rather than solving unconstrained Bundle Adjustment (which can buckle or scale-drift on repetitive indoor textures), the default pipeline uses `colmap point_triangulator`:
$$\min_{X_i} \sum_{j \in \text{views}(i)} \left\| x_{ij} - \pi(R_j X_i + t_j) \right\|^2$$
where camera extrinsics $(R_j, t_j)$ are locked to the VIO trajectory.

### 2. Sampson Epipolar Error Diagnostics
To evaluate relative pose accuracy between camera pairs without relying on triangulated 3D points:
$$d_{\text{Sampson}}^2(x_1, x_2) = \frac{\left(x_2^T F x_1\right)^2}{\left(F x_1\right)_1^2 + \left(F x_1\right)_2^2 + \left(F^T x_2\right)_1^2 + \left(F^T x_2\right)_2^2}$$
where $F = K_2^{-T} [t]_\times R K_1^{-1}$ is the fundamental matrix. Pairs with median Sampson error $> 1.0\text{ px}$ are flagged as potentially drifted.

### 3. Coordinate Convention Handoff
- **Input (`transforms.json`):** OpenGL / ARCore convention ($+X$ right, $+Y$ up, $-Z$ forward).
- **COLMAP (`images.bin`, `cameras.bin`):** OpenCV convention ($+X$ right, $+Y$ down, $+Z$ forward), stored as world-to-camera ($w2c$):
  $$P_{\text{cam}} = R_{w2c} P_{\text{world}} + t_{w2c}$$
- The conversion inverts the matrix and flips $Y$ and $Z$ axes: $w2c = \left(c2w \cdot \operatorname{diag}(1, -1, -1, 1)\right)^{-1}$.

---

## Inputs & Outputs

- **Input:**
  - `<workspace>/images/`: Filtered portrait keyframes.
  - `<workspace>/transforms.json`: Synchronized poses.
- **Output:**
  - `<workspace>/sparse/0/cameras.bin`: Calibrated camera model.
  - `<workspace>/sparse/0/images.bin`: Extrinsic poses (OpenCV $w2c$).
  - `<workspace>/sparse/0/points3D.bin` / `points3D.txt`: Triangulated sparse 3D landmark points.

## Usage

```bash
# Standard pipeline execution:
python 01_poses_refinment/step_colmap.py --workspace backend/current_scene [--force]

# Diagnostic inspection:
python 01_poses_refinment/colmap_diagnostics.py --colmap-dir backend/current_scene/sparse/0
```

or

```bash
python3 run_pipeline.py --only-step colmap
```

