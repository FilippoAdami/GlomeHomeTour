# Bibliography: Algorithmic & Mathematical Techniques

This document lists the core mathematical, geometric, and engineering techniques implemented or evaluated across Passes 0 to 3 of the Glome Suite backend reconstruction system and mobile capture client.

Items prefixed with `**` represent techniques that were evaluated, benchmarked, or compared against during development but were not integrated into the active production pipeline.

---

## Active & Integrated Techniques (Passes 0 to 3)

### 1. Two-Axis Relative & Intensity-Normalized Laplacian Blur Gating (Pass 0)

- **Description:**
  Evaluates frame clarity on two complementary axes: absolute detail (raw Laplacian variance $\sigma^2(\nabla^2 I)$) and local intensity-normalized detail ($\frac{\sigma^2(\nabla^2 I)}{\sigma^2(I)}$), scored relative to the scan's running median. A frame is rejected as motion-blurred only if it fails *both* tests.
- **Where, How & Why Used in Glome:**
  Implemented in `backend/00_ingestion/quality_gate.py` (`QualityGate.is_sharp`) and executed in `step_filter_quality.py`.
- **Motive & Rationale:**
  Absolute Laplacian variance scales with scene luminance and contrast, causing dimly lit wood ceilings or untextured walls to score identically to real motion blur under single-threshold filters. Normalizing by local intensity variance prevents false discards on low-contrast architectural features while reliably catching true camera-shake smear.
- **Last Verified / Checked:** 2026-09-15.

---

### 2. Clipped-Pixel Fraction Exposure Gating (Pass 0)

- **Description:**
  Assesses over- and under-exposure by computing the fraction of clipped pixels at the extremes ($I \le 5$ and $I \ge 250$) across RGB channels, rejecting frames where clipped fractions exceed calibrated limits.
- **Where, How & Why Used in Glome:**
  Implemented in `backend/00_ingestion/quality_gate.py` (`QualityGate.is_properly_exposed`).
- **Motive & Rationale:**
  Replaces rigid mean luminance gating ($Y_{\text{mean}} < 40$) and deleted `max_illumination_jump` rules. In real-world homes with uneven window lighting, panning from a bright sunlit window to a darker interior wall produces natural 60+ level shifts; judging clipped-pixel fractions preserves legitimate interior transitions without admitting blown-out frames.
- **Last Verified / Checked:** 2026-09-15.

---

### 3. Fast Triangulated Scene Depth via Rolling-Median Optical Flow (Pass 0)

- **Description:**
  Measures per-frame metric scene depth without neural network inference. Tracks Shi-Tomasi corners across expanding temporal baselines (4/8/16/32 frames) using Lucas-Kanade optical flow, triangulates rays against known ARCore poses, and applies a 5-frame rolling median filter to suppress noise.
- **Where, How & Why Used in Glome:**
  Implemented in `backend/00_ingestion/inspect_scan.py` and `keyframe_selector.py` (`estimate_scene_depths`).
- **Motive & Rationale:**
  Keyframe covisibility calculations require knowing subject distance (e.g. measuring overlap at $0.8\text{m}$ vs $2.0\text{m}$). Hardcoded planes cause massive errors on close walls, while running deep neural depth on all raw frames is too slow. Direct optical triangulation computes accurate per-frame depth in ~40 ms/frame on CPU.
- **Last Verified / Checked:** 2026-09-15.

---

### 4. Sub-Millisecond Quaternion SLERP & Spline Trajectory Interpolation (Pass 0)

- **Description:**
  Synchronizes asynchronous 60 Hz ARCore VIO trajectory logs with discrete Camera2 mid-exposure timestamps ($t_{\text{mid}} = t_{\text{start}} + \frac{1}{2} t_{\text{exp}}$). Uses Spherical Linear Interpolation (SLERP) on rotation quaternions and cubic spline interpolation on translation vectors.
- **Where, How & Why Used in Glome:**
  Implemented in `backend/00_ingestion/pose_aligner.py`.
- **Motive & Rationale:**
  Camera frame exposures occur between discrete VIO sensor updates. Sub-millisecond temporal interpolation eliminates synchronization jitter, ensuring accurate epipolar constraints for downstream SfM.
- **Last Verified / Checked:** 2026-09-15.

---

### 5. Fixed-Pose Point Triangulation with ARCore VIO Anchors (Pass 1)

- **Description:**
  Executes COLMAP's `point_triangulator` with ARCore VIO camera poses held strictly fixed as ground truth. Triangulates multi-view SIFT feature matches into 3D landmark points without re-estimating camera extrinsics from scratch.
- **Where, How & Why Used in Glome:**
  Implemented in `backend/01_poses_refinment/convert_transforms_to_colmap.py` and `step_colmap.py`.
- **Motive & Rationale:**
  Unconstrained bundle adjustment on indoor smartphone video frequently suffers from gauge ambiguity, scale drift, and focal length hallucination. Triangulating against metric, gravity-aligned ARCore poses anchors the reconstruction to true physical dimensions.
- **Last Verified / Checked:** 2026-09-15.

---

### 6. Soft Pose-Prior Bundle Adjustment with Locked Intrinsics (Pass 1)

- **Description:**
  When pose refinement is enabled (`--ba_mode pose_prior`), uses COLMAP's `pose_prior_mapper` to optimize camera positions under tight Gaussian priors ($\sigma = 5\text{ cm}$) centered on the ARCore trajectory, while locking camera intrinsics ($f_x, f_y, c_x, c_y$) and distortion parameters.
- **Where, How & Why Used in Glome:**
  Implemented in `backend/01_poses_refinment/convert_transforms_to_colmap.py`.
- **Motive & Rationale:**
  Allows local multi-view refinement to adjust for slight tracking drift while preventing global trajectory warping and preventing the optimizer from fabricating non-existent lens distortion.
- **Last Verified / Checked:** 2026-09-15.

---

### 7. Second-Order Sampson Epipolar Error Validation (Pass 1)

- **Description:**
  Evaluates the second-order geometric Sampson error $d_{\text{Sampson}}^2 = \frac{(x_2^T F x_1)^2}{(F x_1)_1^2 + (F x_1)_2^2 + (F^T x_2)_1^2 + (F^T x_2)_2^2}$ on raw verified inlier matches from `colmap_database.db` using the fundamental matrix derived from refined camera poses. Compares against a control fundamental matrix computed directly from the matcher. Frames with error $> 2.0\text{ px}$ are rejected (with a 20% refusal cap).
- **Where, How & Why Used in Glome:**
  Implemented in `backend/01_poses_refinment/colmap_diagnostics.py` and `step_colmap.py`.
- **Motive & Rationale:**
  Standard bundle adjustment reprojection error can be deceptively low on short 2-view tracks even when poses have drifted severely. Sampson error on raw verified matches detects internally consistent but physically warped solutions.
- **Last Verified / Checked:** 2026-09-15.

---

### 8. True Shared-Track Covisibility Keyframe Selection (Pass 2)

- **Description:**
  Selects optimal keyframes using true 3D geometric covisibility, defined as the shared COLMAP track fraction $\frac{|\text{tracks}(i) \cap \text{tracks}(j)|}{\min(|\text{tracks}(i)|, |\text{tracks}(j)|)}$ paired with median triangulated scene depth. Selects frames within a calibrated $0.25 - 0.60$ covisibility band and $0.15\text{m} / 6^\circ$ motion gate.
- **Where, How & Why Used in Glome:**
  Implemented in `backend/02_depth_estimation/step_filter_depth.py` (`TrackCovisibilitySelector`).
- **Motive & Rationale:**
  Replaces heuristic 2D frustum approximations with ground-truth 3D point track co-visibility. Guarantees gapless baseline coverage across the walk while eliminating redundant stationary views that bias 2DGS training.
- **Last Verified / Checked:** 2026-09-15.

---

### 9. Multi-View Sliding-Window Metric Depth Estimation (Pass 2)

- **Description:**
  Streams keyframes through Depth Anything 3 (`DA3-BASE`) in overlapping sliding-window chunks ($N=6, K=2$) with OpenCV world-to-camera poses ($w2c$). Fuses cross-chunk predictions using median blending across overlapping frames.
- **Where, How & Why Used in Glome:**
  Implemented in `backend/02_depth_estimation/depth_priors.py` and `step_depth.py`.
- **Motive & Rationale:**
  Enables multi-view transformer cross-attention across long sequences without exhausting GPU VRAM, ensuring smooth, metric depth continuity across hundreds of frames.
- **Last Verified / Checked:** 2026-09-15.

---

### 10. Dynamic Sensor Saturation & Bloom Masking (Pass 2 & Pass 3)

- **Description:**
  Detects overexposed light sources using an adaptive high-percentile floor $\text{clip}(Q_{0.998}(\max(R,G,B)), 250, 255)$ paired with a low-chroma difference test ($\max(R,G,B) - \min(R,G,B) \le 35$). Masks saturated pixels from depth unprojection and photometric training losses (`masked_l1_loss`, `masked_ssim`).
- **Where, How & Why Used in Glome:**
  Implemented in `02_depth_estimation/initialization.py` (`compute_overexposed_mask`) and `03_2DGS_training/utils/loss_utils.py` (`compute_saturation_mask`).
- **Motive & Rationale:**
  Blown-out light fixtures cause monocular depth models to hallucinate conical spikes and induce floating 3D hairballs in 2DGS. Masking saturated regions allows the optimizer to interpolate ceiling geometry smoothly without floater artifacts.
- **Last Verified / Checked:** 2026-09-15.

---

### 11. Cross-View Epipolar Free-Space Carving (Pass 2)

- **Description:**
  Reprojects candidate 3D depth points into unobstructed adjacent camera views (baseline $\ge 8\text{ cm}$ or parallax $\ge 3^\circ$). If a point projects into free space in front of an observed surface ($\text{proj}_z < d_{\text{obs}} - (0.08 + 0.05 \cdot \text{proj}_z)$), it is vetoed as a free-space violation and culled before initialization.
- **Where, How & Why Used in Glome:**
  Implemented in `backend/02_depth_estimation/initialization.py` (`filter_multiview_consistency`).
- **Motive & Rationale:**
  Single-view depth estimation frequently bleeds foreground silhouettes into open background space (e.g. doorframes, chandeliers). Cross-view free-space carving eliminates boundary floaters and floating phantom geometry before training begins.
- **Last Verified / Checked:** 2026-09-15.

---

### 12. Dynamic Adaptive Statistical Depth Ceiling (Pass 2)

- **Description:**
  Calculates an adaptive maximum depth horizon using the high-confidence depth quantile and Median Absolute Deviation: $d_{\max} = Q_{0.98} + 1.5 \cdot \text{MAD}$.
- **Where, How & Why Used in Glome:**
  Implemented in `backend/02_depth_estimation/depth_priors.py` and `initialization.py` (`estimate_adaptive_depth_ceiling`).
- **Motive & Rationale:**
  Eliminates hardcoded depth bounds (e.g. 5m or 10m). Automatically scales bounds between compact powder rooms ($3.5\text{m}$) and open high-ceiling great rooms ($15\text{m}+$) while culling distant sky noise through windows.
- **Last Verified / Checked:** 2026-09-15.

---

### 13. Coarsening Voxel Grid Initialization & Delaunay-Voronoi Scale Clamping (Pass 2)

- **Description:**
  Downsamples unprojected depth points using a spatial voxel grid that dynamically coarsens ($1.5 \to 2.0 \to 2.5\text{ cm}$) until the point count fits `max_surfels` (3M ceiling). Clamps initial Gaussian surfel radii strictly to $[0.4v, 0.8v]$ matching the actual voxel size $v$.
- **Where, How & Why Used in Glome:**
  Implemented in `backend/02_depth_estimation/initialization.py` (`initialize_from_keyframes`).
- **Motive & Rationale:**
  Random thinning destroys uniform spatial coverage. Coarsening the grid maintains uniform surface density, while Delaunay-Voronoi scale clamping ($r_{\min} = \frac{\sqrt{2}}{2}v$) guarantees continuous surface coverage without micro-holes or oversized blur blobs.
- **Last Verified / Checked:** 2026-09-15.

---

### 14. Dual Coordinate System Handoff (Pass 2 & Pass 3)

- **Description:**
  Manages exact coordinate conventions across the pipeline:
  - COLMAP & Depth Anything 3 use **OpenCV World-to-Camera ($w2c$)**: $+X$ right, $+Y$ down, $+Z$ forward ($P_{\text{cam}} = R_{w2c} P_{\text{world}} + t_{w2c}$).
  - Surfel unprojection and 2DGS camera loaders use **OpenGL Camera-to-World ($c2w$)**: $+X$ right, $+Y$ up, $-Z$ forward ($c2w_{\text{GL}} = w2c^{-1} \cdot \operatorname{diag}(1, -1, -1, 1)$).
- **Where, How & Why Used in Glome:**
  Implemented in `backend/02_depth_estimation/colmap_poses_to_da3.py` and verified by `tests/test_colmap_poses_to_da3.py`.
- **Motive & Rationale:**
  Inverting or flipping camera axes incorrectly causes subtle geometric inversions ("the bowl" artifact). Strict validation functions enforce determinant $= +1$ and orthonormality across all handoffs.
- **Last Verified / Checked:** 2026-09-15.

---

### 15. 4-Stage Progressive Multi-Scale Training Schedule (Pass 3)

- **Description:**
  Trains 2DGS across progressively increasing image resolutions: Stage 1 ($-r 8$, 1/8 scale, iters 0–2,500), Stage 2 ($-r 4$, 1/4 scale, iters 2,500–4,500), Stage 3 ($-r 2$, 1/2 scale, iters 4,500–6,000), and Stage 4 ($-r 1$, native 1080p, iters 6,000–10,000).
- **Where, How & Why Used in Glome:**
  Implemented in `backend/03_2DGS_training/step_train.py` and `train_room.py`.
- **Motive & Rationale:**
  Coarse-to-fine optimization anchors low-frequency room geometry and surface normals early, preventing the optimizer from overfitting high-frequency textures into misaligned 3D positions. Stages 1–3 lock camera poses to enforce metric consistency with DA3 depth priors.
- **Last Verified / Checked:** 2026-09-15.

---

### 16. PGSR Multi-View Homography Loss with Asymmetric Free-Space Veto (Pass 3)

- **Description:**
  Treats each rendered pixel's depth and normal as a local tangent plane and projects into neighboring views using homography $H = K_n (R - t n^T / d) K_c^{-1}$. Computes multi-view loss with asymmetric consensus: $\mathcal{L}_{\text{mv}} = (1 - w_{\text{veto}}) \cdot \text{mean}(\mathcal{L}_k) + w_{\text{veto}} \cdot \max(\mathcal{L}_k)$ ($w_{\text{veto}} = 0.5$).
- **Where, How & Why Used in Glome:**
  Implemented in `backend/03_2DGS_training/utils/multiview_loss.py`.
- **Motive & Rationale:**
  Prevents unobstructed side views observing empty space from being diluted by narrow-baseline views that share a reflection or specular highlight, enforcing planar wall consistency.
- **Last Verified / Checked:** 2026-09-15.

---

### 17. TrackGS Learnable Lie-Algebra $\mathfrak{se}(3)$ Camera Pose Refinement (Pass 3)

- **Description:**
  Optimizes camera extrinsics during 2DGS training via zero-initialized $\mathfrak{se}(3)$ exponential mapping deltas, regularized by 3D landmark reprojection loss $\mathcal{L}_{\text{track}}$:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{photometric}} + \lambda_{\text{track}} \sum_{i} \|\pi(R(\delta\xi) X_i + t(\delta\xi)) - x_i\|^2$$
- **Where, How & Why Used in Glome:**
  Implemented in `backend/03_2DGS_training/scene/cameras.py` and activated by `step_train.py` **exclusively during Stage 4 (full 1080p, iters 6,000–10,000)** with `--pose_lr 0.0001` and `--lambda_track 0.1`.
- **Motive & Rationale:**
  Absorbs sub-pixel handheld frame-to-frame jitter and sensor clock sync offsets during final high-frequency texture sharpening, while tight landmark tracking anchors prevent the cameras from drifting or warping global room dimensions.
- **Last Verified / Checked:** 2026-09-15 (Enabled in Stage 4 of `step_train.py`).

---

### 18. TIDI-GS Observation-Count / Frustum Ratio Floater Pruning (Pass 3)

- **Description:**
  Tracks camera frustum visibility versus direct alpha rendering contributions per Gaussian surfel. Prunes primitives that appear in $\ge 8$ camera frustums but only actively contribute in $\le 2$ views after iteration 1500.
- **Where, How & Why Used in Glome:**
  Implemented in `backend/03_2DGS_training/scene/gaussian_model.py` (`densify_and_prune`).
- **Motive & Rationale:**
  Monocular floaters receive gradient updates from only one camera angle while occluding valid geometry from other views. Frustum-to-observation ratio gating purges them cleanly.
- **Last Verified / Checked:** 2026-09-15.

---

### 19. FasterGS Backface Surface Normal Culling (Pass 3)

- **Description:**
  Culls backfacing planar surfels before rasterization by evaluating the dot product between the camera-space normal and view ray: $(n_{\text{cam}} \cdot pts_{\text{cam}}) < 0$.
- **Where, How & Why Used in Glome:**
  Implemented in `diff-surfel-rasterization` and `gaussian_renderer/__init__.py`.
- **Motive & Rationale:**
  Reduces active primitives per frame by ~40–50%, accelerating step time and preventing back-wall splats from bleeding through thin partition drywalls.
- **Last Verified / Checked:** 2026-09-15.

---

### 20. Custom Elementwise Rotation Operator (`_rotate()`) for AMD gfx1200 (Pass 3)

- **Description:**
  Performs $(N, 3) @ (3, 3)^T$ rotation using elementwise column arithmetic ($v'_x = v_x R_{00} + v_y R_{01} + v_z R_{02}$, etc.) instead of standard `torch.matmul`.
- **Where, How & Why Used in Glome:**
  Implemented in `backend/03_2DGS_training/utils/point_utils.py` and `rasterizer_interface.py`.
- **Motive & Rationale:**
  Workaround for AMD ROCm 7.1 / `gfx1200` (RDNA4) BLAS GEMM driver bug where `torch.matmul` on $(N, 3) @ (3, 3)$ silently writes zeros to all output rows past index $2^{19} = 524,288$. Elementwise rotation is exact at any $N$ and faster for $K=3$.
- **Last Verified / Checked:** 2026-09-15.

---

## Evaluated & Compared Techniques (Not Integrated)

### **Distance-and-Angle Gated Keyframe Selection (Pass 0)
- **Description:** Gated frame export strictly on fixed translation ($\ge 8\text{ cm}$) or rotation ($\ge 6^\circ$).
- **Evaluation in Glome:** Replaced by streaming decimation and track-covisibility selection because fixed spatial thresholds drop frames during slow panning and cause coverage gaps.
- **Last Checked:** 2026-09-13.

---

### **ARCore Native Monocular Depth & Plane Wireframing (Pass 0 / Mobile)
- **Description:** ARCore's native `Environment Depth` and plane detection.
- **Evaluation in Glome:** Discarded during mobile field trials due to severe phantom floating planes; replaced by closed-form feature parallax landmarking on mobile and DA3 on the backend.
- **Last Checked:** 2026-09-10.

---

### **On-Device Volumetric Marching Cubes Mesh Reconstruction (Mobile)
- **Description:** Real-time marching cubes isosurface extraction over mobile voxel grids.
- **Evaluation in Glome:** Replaced by OpenGL point sprite landmark rendering (`FeatureParallaxTracker.kt`) to eliminate SoC thermal throttling.
- **Last Checked:** 2026-09-10.

---

### **SuperPoint Deep Feature Extraction & Neural Matching (Pass 1)
- **Description:** CNN-based keypoint detection and matching.
- **Evaluation in Glome:** SIFT was retained as the default due to zero PyTorch/GPU execution overhead during initial ingestion.
- **Last Checked:** 2026-09-14.
