# 02_depth_estimation — Depth Estimation & Surfel Initialization

`02_depth_estimation` generates multi-view metric depth maps using **Depth Anything 3 (DA3)** and unprojects them into a clean, surface-oriented 2D Gaussian surfel point cloud (`points3D_depth.ply`) to seed 2DGS training.

---

## What This Stage Does

1. **Multi-View Metric Depth Inference:** Executes DA3-Base in a memory-efficient sliding window with cross-chunk overlap blending, COLMAP track scale alignment, and chunk-level resume caching.
2. **Dynamic Saturation & Bloom Masking:** Discards overexposed pixels before 3D unprojection using an adaptive percentile floor and chroma-difference check, preventing light sources from hallucinating floating artifacts.
3. **Cross-View Epipolar / Free-Space Carving:** Reprojects candidate 3D points into unobstructed adjacent camera views and discards points landing in empty space in front of observed surfaces.
4. **Planar Regularization (in place, `regularize_planes.py`):** Aligns storeys that monocular depth left yawed apart, merges the RANSAC fragments of each physical wall/floor onto one global Manhattan frame, and re-samples every snapped surface as an equidistant lattice on its exact plane. Runs last and rewrites `points3D_depth.ply` itself -- disable with `--no-plane-regularization`.
5. **Oriented Surfel Initialization:** Unprojects filtered depth maps, estimates surface normals from depth gradients, constructs right-handed orthonormal tangent frames $(u, v, n)$, and exports degree-0 Spherical Harmonics color representations.

---

## Modules & Architecture

| Module | Role |
| --- | --- |
| [`keyframe_budget.py`](keyframe_budget.py) | Floor area -> keyframe count band (`frame_budget`), and greedy voxel-coverage fitting to it (`coverage_prune` / `coverage_topup`). |
| [`step_depth.py`](step_depth.py) | **Pipeline Step 2 (depth)**: Runs DA3 sliding window depth inference, normal polishing, and builds `depth/points3D_depth.ply`. |
| [`regularize_planes.py`](regularize_planes.py) | Storey alignment, plane merge + Manhattan snap, lattice fill with 3cm existing point protection (`regularize`). Standalone CLI plus `--selftest`. |
| [`label_planes.py`](label_planes.py) | Diagnostic only: colours the big planes and prints the pairwise angle matrix. Writes nothing the pipeline consumes. |
| [`initialization.py`](initialization.py) | Core surfel initialization engine (`SurfelCloudInitializer`), saturation masking (`compute_overexposed_mask`), and free-space filtering (`filter_multiview_consistency`). |
| [`depth_priors.py`](depth_priors.py) | High-level DA3 wrapper (`DepthPriorEstimator`), global scale-shift alignment, and surface normal compute & polishing (`extract_and_polish_normals`). |
| [`colmap_poses_to_da3.py`](colmap_poses_to_da3.py) | Converts COLMAP poses into NumPy arrays for DA3 with strict orthonormality and ARCore agreement checks. |
| [`diagnose_single_frame.py`](diagnose_single_frame.py) | Diagnostic script to run single-frame depth and surface normal inspection. |

---

## Techniques & Mathematical Specifications

### 1. Dynamic Saturated Pixel Masking
To prevent blown-out ceiling lights and window flares from seeding phantom surfels:
- **Dynamic Threshold Floor:**
  $$T_{\text{dyn}} = \operatorname{clip}\left(\operatorname{quantile}_{99.8}\big(\max(R, G, B)\big),\, 250,\, 255\right)$$
- **Achromatic Bloom Test:**
  $$\max(R, G, B) \ge T_{\text{dyn}} \quad \text{and} \quad \big(\max(R, G, B) - \min(R, G, B)\big) \le 35$$
- **Morphological Dilation:** Dilates by radius 1 ($3 \times 3$ ellipse) to eliminate bloom boundary halos and depth tearing.

### 2. Cross-View Epipolar Free-Space Carving & Grazing Angle Filter
- **Grazing Angle Culling:** Rejects extreme oblique ray projections ($> 85.0^\circ$) to avoid degenerate elongated surfels.
- **Perspective Parallax:** Requires baseline $\|t_j - t_i\| \ge 0.08\text{ m}$ or optical angle $\ge 3^\circ$.
- **Empty-Space Violation:**
  $$z_{\text{proj}} < d_{\text{obs}} - \tau_{\text{free}}(z_{\text{proj}}), \quad \tau_{\text{free}}(z) = 0.08 + 0.05 z$$
  If camera $j$ sees an unobstructed surface at $d_{\text{obs}}$ behind the candidate point $z_{\text{proj}}$, the point is located in empty space and culled ($N_{\text{violations}} > \text{max\_freespace\_violations}$).

### 3. Dual Coordinate System Conventions
Two coordinate conventions meet in Step 2 in opposite directions:
- **DA3 consumes OpenCV World-to-Camera ($w2c$):** Provided directly from COLMAP without modification.
- **SurfelCloudInitializer consumes OpenGL Camera-to-World ($c2w$):**
  $$c2w_{\text{GL}} = w2c^{-1} \cdot \operatorname{diag}(1, -1, -1, 1)$$

### 4. Orthonormal Tangent Frame Computation
For each unit normal $n$, computes orthogonal tangents $(u, v)$ such that $(u, v, n)$ forms an orthonormal basis:
$$\text{ref} = [1, 0, 0] \text{ if } |n_z| > 0.9 \text{ else } [0, 0, 1]$$
$$u = \frac{n \times \text{ref}}{\|n \times \text{ref}\|}, \quad v = n \times u$$

---

## Inputs & Outputs

- **Input:**
  - `<workspace>/images/`: Portrait keyframes.
  - `<workspace>/sparse/0/`: COLMAP model (poses + points).
  - `<workspace>/transforms.json`: Camera intrinsics.
- **Output:**
  - `<workspace>/depth/depth_maps/*.npy`: Dense float32 metric depth maps in meters.
  - `<workspace>/depth/normal_maps/*.npy`: Dense float32 polished unit surface normal maps.
  - `<workspace>/depth/depth_images/*.jpg`: Side-by-side RGB and colorized depth diagnostics.
  - `<workspace>/depth/normal_images/*.jpg`: Side-by-side RGB and colorized surface normal diagnostics.
  - `<workspace>/depth/poses_da3.npz`: Aligned camera extrinsics and intrinsics.
  - `<workspace>/depth/points3D_depth.ply`: Fused, filtered, oriented, plane-regularized 2D surfel point cloud (consumed by Step 3 to seed 2DGS training). Planar surfaces in it are exact lattices at `--plane-fill-spacing` (default 5 cm) with a 3 cm avoidance radius around existing inliers; everything else keeps its original samples.

## Usage

```bash
# Metric depth estimation & surfel initialization
python 02_depth_estimation/step_depth.py \
    --workspace backend/current_scene \
    [--no-saturation-mask] \
    [--no-freespace-filter] \
    [--max-freespace-violations 0] \
    [--force]
```
