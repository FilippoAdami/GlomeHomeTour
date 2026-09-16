# Bibliography: Academic Papers

This document lists academic papers that were read, referenced, and integrated across the Glome Suite backend pipeline (Passes 0 to 3: Ingestion, Pose Refinement, Metric Depth Estimation, and 2D Gaussian Splatting) and mobile capture client.

Items prefixed with `**` represent papers that were evaluated, benchmarked, or compared against during development but were not integrated into the active production pipeline.

---

## Active & Integrated Papers (Passes 0 to 3)

### 2D Gaussian Splatting for Geometrically Accurate Radiance Fields (2DGS) (CVPR 2024) [arXiv:2403.17888] - Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, Shenghua Gao

- **Description:**
  Replaces standard 3D volumetric ellipsoidal Gaussians with flat, oriented 2D planar Gaussian disks (surfels) embedded in 3D space. Defines explicit surface normals, 2D tangent frames, and exact ray-splat intersection formulas, eliminating volumetric floater fuzz, multi-view ambiguity, and needle-like artifacts.
- **Where & How Used in Glome (Pass 3):**
  Core neural radiance representation in `backend/03_2DGS_training/` executed via native `diff-surfel-rasterization`. The planar surfel geometry provides the foundational representation for downstream floor plan slicing ($1.0\text{m}-1.5\text{m}$), 3D mesh reconstruction (`05_2DGS_to_mesh`), and MLS walkthrough packaging.
- **Motive & Rationale:**
  Standard 3DGS produces thick, fuzzy volumetric shells and severe floater artifacts when viewing surfaces from oblique angles or unobserved directions. 2DGS enforces true surface-bound geometry with consistent normal vectors, which is strictly required for CAD-compatible architectural mesh extraction and planar wall fitting.
- **Last Verified / Checked:** 2026-09-15 (Fully integrated in `step_train.py` and `diff-surfel-rasterization`).

---

### Depth Anything 3: Recovering the Visual Space from Any Views (DA3) (2025) [arXiv:2511.10647] - Lihe Yang, et al.

- **Description:**
  A vision transformer foundation model for metric monocular depth estimation with multi-view attention. Jointly predicts dense metric depth and camera rays across multi-view image sequences, achieving high spatial consistency and metric scale fidelity across unconstrained video captures.
- **Where & How Used in Glome (Pass 2):**
  Primary metric depth inference engine in `backend/02_depth_estimation/depth_priors.py` and `step_depth.py` using `depth-anything/DA3-BASE` (and `DA3NESTED-GIANT-LARGE-1.1`). Keyframes are processed in overlapping sliding-window chunks ($N=6, K=2$) with OpenCV world-to-camera poses to generate metric depth arrays (`depth_maps/*.npy`) and surface normals.
- **Motive & Rationale:**
  Unlike classical stereo matching (which fails on textureless white walls) and single-view monocular depth models (which drift in scale from frame to frame), DA3 combines multi-view cross-attention with learned indoor priors, generating metric, warp-free depth maps across entire residential scans.
- **Last Verified / Checked:** 2026-09-15 (Sliding-window runner verified with OpenCV $w2c$ input convention).

---

### Structure-from-Motion Revisited (COLMAP) (CVPR 2016) - Johannes L. Schönberger, Jan-Michael Frahm

- **Description:**
  A comprehensive Structure-from-Motion (SfM) architecture introducing geometric verification, robust triangulation, view-graph construction, and non-linear bundle adjustment strategies that mitigate drift and mis-registration.
- **Where & How Used in Glome (Pass 1):**
  Powers `backend/01_poses_refinment/` (`convert_transforms_to_colmap.py` and `step_colmap.py`). Glome uses COLMAP's SIFT feature extraction, sequential matcher, and `point_triangulator` with fixed ARCore VIO poses, plus optional soft pose-prior bundle adjustment (`pose_prior_mapper`).
- **Motive & Rationale:**
  Rather than running unconstrained SfM from scratch (which frequently collapses, warps focal length, or invents phantom distortion on indoor loops), triangulating against known-good metric ARCore VIO poses guarantees scale consistency and provides verified multi-view point tracks.
- **Last Verified / Checked:** 2026-09-15 (Integrated via `step_colmap.py` with Sampson error gating).

---

### PGSR: Planar-based Gaussian Splatting for Robust Surface Reconstruction (2024) [arXiv:2406.06521] - Danpeng Chen, et al.

- **Description:**
  Introduces a multi-view planar regularization technique for Gaussian Splatting. Treats each rendered pixel's depth and normal as a local tangent plane and enforces photometric homography consistency across neighboring camera views using $H = K_n (R - t n^T / d) K_c^{-1}$.
- **Where & How Used in Glome (Pass 3):**
  Integrated in `backend/03_2DGS_training/utils/multiview_loss.py` via `multiview_photometric_loss` and `--lambda_multiview`. Computes warped photometric consensus across sequential and loop-closure views with an asymmetric free-space veto weight ($w_{\text{veto}} = 0.5$).
- **Motive & Rationale:**
  Photometric color loss alone allows Gaussians to overfit to camera-specific lighting, specular reflections, or textureless regions. Homography-based multi-view consensus forces surfels to align with physical planar surfaces, preventing multi-layering ("onion-peel") and surface wrinkling.
- **Last Verified / Checked:** 2026-09-15 (Asymmetric free-space veto integrated and unit-tested).

---

### TrackGS: Direct Neural Tracking with 2D/3D Gaussian Splatting (2024) - TrackGS Authors

- **Description:**
  Enables continuous camera pose refinement during Gaussian Splatting training. Optimizes small Lie-algebra $\mathfrak{se}(3)$ camera pose deltas jointly with scene Gaussians, regularized by fixed 3D landmark reprojection constraints to prevent pose drift.
- **Where & How Used in Glome (Pass 3):**
  Implemented in `backend/03_2DGS_training/scene/cameras.py` and activated by `step_train.py` **exclusively in Stage 4 (full native 1080p resolution, iterations 6,000–10,000)** via `--refine_poses_during_training`, `--pose_lr 0.0001`, and `--lambda_track 0.1`. Poses remain rigidly locked during Stages 1–3 ($r=8, 4, 2$) to prevent gauge drift and preserve metric alignment with DA3 depth priors.
- **Motive & Rationale:**
  Compensates for residual sub-pixel handheld frame-to-frame jitter and rolling shutter sync offsets between adjacent keyframes during high-resolution texture convergence, without allowing the optimizer to bend the global metric camera trajectory or distort DA3 multi-view depth priors.
- **Last Verified / Checked:** 2026-09-15 (Enabled in Stage 4 of `step_train.py` with tight track constraints).

---

### TIDI-GS: Tile-based and Direct Observation Pruning for Gaussian Splatting (2024/2025)

- **Description:**
  Analyzes camera visibility frustums versus direct rendering contributions per Gaussian primitive to identify and prune floating artifacts. Surfel primitives that are visible in many camera view frustums but only actively contribute alpha opacity to one or two views are detected as monocular floaters.
- **Where & How Used in Glome (Pass 3):**
  Implemented in `backend/03_2DGS_training/scene/gaussian_model.py` (`frustum_counter`, `observation_counter`, `densify_and_prune`). Culls primitives visible in $\ge 8$ frustums but actively contributing in $\le 2$ views after iteration 1500.
- **Motive & Rationale:**
  Standard opacity pruning fails on semi-transparent floaters that receive gradient updates from only a single camera. Tracking frustum-to-contribution ratios cleanly removes airborne glare spikes and boundary floaters without eroding solid walls.
- **Last Verified / Checked:** 2026-09-15 (Observation tracking and floater pruning verified).

---

### DN-Splatter: Depth and Normal Priors for Gaussian Splatting (2024) [arXiv:2403.17822] - Matias Turkulainen, et al.

- **Description:**
  Incorporates external dense monocular depth and surface normal priors into 3D/2D Gaussian Splatting optimization via depth-ranking and normal-alignment loss terms, drastically improving surface reconstruction quality in sparse or untextured regions.
- **Where & How Used in Glome (Pass 2 & Pass 3):**
  Governs surfel cloud initialization (`02_depth_estimation/initialization.py`) and training normal supervision (`03_2DGS_training/utils/loss_utils.py` via `normal_loss`). Depth gradients from DA3 initialize surfel tangent frames and guide optimization where image texture is uniform.
- **Motive & Rationale:**
  Indoor architectural captures feature large textureless white drywall and ceilings where photometric gradients are zero. Depth and normal priors anchor Gaussian positions and orientations geometrically, preventing hollow geometry.
- **Last Verified / Checked:** 2026-09-15 (Integrated with dynamic saturation masking).

---

### Distinctive Image Features from Scale-Invariant Keypoints (SIFT) (IJCV 2004) - David G. Lowe

- **Description:**
  Seminal algorithm for detecting and describing local scale- and rotation-invariant features in images, robust to changes in illumination, noise, and 3D viewpoint variations.
- **Where & How Used in Glome (Pass 0 & Pass 1):**
  Used in `backend/00_ingestion/` for keyframe covisibility verification and in `backend/01_poses_refinment/` for COLMAP sequential matching and feature graph construction.
- **Motive & Rationale:**
  Classical SIFT provides highly reliable, sub-pixel feature correspondences across wide baselines without requiring heavy GPU memory or PyTorch neural inference during stage 1 ingestion.
- **Last Verified / Checked:** 2026-09-15 (SIFT extraction and sequential matching verified).

---

### The Structural Similarity Image Metric (SSIM) (IEEE TIP 2004) - Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, Eero P. Simoncelli

- **Description:**
  A perceptual metric that measures visual degradation based on luminance, contrast, and structural information changes, outperforming raw mean squared error for human perceptual quality assessment.
- **Where & How Used in Glome (Pass 3):**
  Core photometric loss term in `03_2DGS_training/utils/loss_utils.py` and evaluation metric in `metrics.py`. Combined with L1 loss: $\mathcal{L} = (1 - \lambda_{\text{ssim}}) \mathcal{L}_1 + \lambda_{\text{ssim}} (1 - \text{SSIM})$.
- **Motive & Rationale:**
  L1 loss alone yields blurry edges and over-smooths specular highlights. SSIM penalizes structural patch distortion, maintaining sharp architectural corners and texture fidelity.
- **Last Verified / Checked:** 2026-09-15 (Separable 2D conv implementation optimized for ROCm).

---

### The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS) (CVPR 2018) - Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang

- **Description:**
  Evaluates perceptual image distance by extracting and comparing deep feature activations from pretrained networks (VGG/AlexNet), aligning closely with human visual quality judgments.
- **Where & How Used in Glome (Pass 3):**
  Integrated in `backend/03_2DGS_training/lpipsPyTorch/` and `metrics.py` for automated novel-view visual quality benchmarking.
- **Motive & Rationale:**
  Standard PSNR and SSIM can be deceptive on subtle high-frequency artifacts (such as minor splat misalignment). LPIPS provides an objective standard for MLS listing image quality certification.
- **Last Verified / Checked:** 2026-09-15 (Perceptual metric scripts verified).

---

### Depth Anything V2 (2024) [arXiv:2406.09414] - Lihe Yang, et al.

- **Description:**
  Refined monocular depth estimation model trained on synthetic data with teacher-student distillation, producing clean depth boundaries and high-frequency surface detail.
- **Where & How Used in Glome (Pass 2):**
  Configured as an automated local fallback in `backend/02_depth_estimation/depth_priors.py` (`Depth-Anything-V2-Metric-Indoor-Base-hf`) when running on resource-constrained compute nodes without DA3 multi-view weights.
- **Motive & Rationale:**
  Provides a fast, robust single-view indoor depth fallback when multi-view windowing is unnecessary or memory is constrained.
- **Last Verified / Checked:** 2026-09-15 (Fallback loading verified).

---

## Evaluated & Compared Papers (Not Integrated)

### **Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data (CVPR 2024) [arXiv:2401.10891] - Lihe Yang, et al.
- **Description:** Original foundation model for relative monocular depth estimation.
- **Evaluation in Glome:** Evaluated during early depth research. Outputs relative/scale-ambiguous depth maps without metric calibrations; superseded by Depth Anything V2/V3 metric models in Pass 2.
- **Last Checked:** 2026-09-10.

---

### **Towards Robust Monocular Depth Estimation (MiDaS) (IEEE TPAMI 2019) [arXiv:1907.01341] - René Ranftl, et al.
- **Description:** Multi-dataset relative depth estimation architecture.
- **Evaluation in Glome:** Benchmarked for mobile depth HUD and early backend testing. Superseded by ONNX ZipDepth on mobile and DA3 on the backend due to scale ambiguity and boundary bleeding.
- **Last Checked:** 2026-09-10.

---

### **SuperPoint: Self-Supervised Interest Point Detection and Description (CVPRW 2018) [arXiv:1712.07629] - Daniel DeTone, et al.
- **Description:** Fully-convolutional interest point detector and descriptor network.
- **Evaluation in Glome:** Compared against classical SIFT in Pass 1. SIFT was selected for lower CPU footprint and zero GPU dependency during initial ingestion.
- **Last Checked:** 2026-09-14.

---

### **Instant Neural Graphics Primitives (Instant-NGP) (ACM TOG / SIGGRAPH 2022) - Thomas Müller, et al.
- **Description:** Multi-resolution hash grid encoding for real-time neural radiance fields.
- **Evaluation in Glome:** Evaluated against 2DGS/3DGS for real estate walkthroughs. While training is fast, volumetric ray-marching requires heavy GPU compute for client-side rendering (failing MLS mobile/web budget requirements) and cannot export direct 2D floor plans.
- **Last Checked:** 2026-09-10.

---

### **Direct Voxel Grid Optimization (DVGO) (CVPR 2022) - Cheng Sun, et al.
- **Description:** Explicit voxel grid radiance field representation.
- **Evaluation in Glome:** Evaluated for rapid indoor reconstruction. Discarded due to high memory footprint for large multi-room spaces and lack of explicit surface normals.
- **Last Checked:** 2026-09-10.

---

### **MonoSDF: Exploring Monocular Geometric Cues for Neural Implicit Surface Reconstruction (NeurIPS 2022) - Zehao Yu, et al.
- **Description:** Signed Distance Function (SDF) surface reconstruction supervised with monocular depth and normal cues.
- **Evaluation in Glome:** Influenced our depth and normal prior loss formulations. Full implicit SDF volume rendering was rejected due to slow training convergence (>1 hour per room) compared to 2DGS (<5 minutes).
- **Last Checked:** 2026-09-10.

---

### **NerfAcc: A General NeRF Acceleration Toolbox (2023) - Ruilong Li, et al.
- **Description:** PyTorch acceleration library for volumetric radiance fields.
- **Evaluation in Glome:** Evaluated for volumetric NeRF ray sampling; superseded by explicit rasterization in 2DGS.
- **Last Checked:** 2026-09-10.

---

### **PCR-GS: Point Cloud Refinement for Gaussian Splatting (2024)**
- **Description:** Point cloud outlier filtering and densification strategies for 3DGS.
- **Evaluation in Glome:** Evaluated for Pass 1/2 point cloud cleaning; replaced by spatial voxel deduplication, SOR, and free-space epipolar carving.
- **Last Checked:** 2026-09-14.

---

### **Hestia: Real-Time Multi-Room Neural Walkthroughs (2024)**
- **Description:** Multi-room indoor radiance field partitioning and web streaming.
- **Evaluation in Glome:** Architectural reference for multi-room listing chunking and level-of-detail management.
- **Last Checked:** 2026-09-12.
