# Bibliography: Open-Source Projects & Frameworks

This document lists open-source software projects, libraries, frameworks, and developer tools used or evaluated in the development of the Glome Suite across mobile client (`mobile/android/`) and backend infrastructure (Passes 0 to 3: `backend/00_ingestion/`, `backend/01_poses_refinment/`, `backend/02_depth_estimation/`, `backend/03_2DGS_training/`).

Items prefixed with `**` represent projects that were evaluated, benchmarked, or compared against during development but were not integrated into the active production pipeline.

---

## Active & Integrated Projects (Passes 0 to 3)

### COLMAP (v4.3.0.dev0, ROCm/HIP build) [https://colmap.github.io/]

- **Where & How Used in Glome (Pass 1):**
  Core Structure-from-Motion engine in `backend/01_poses_refinment/` (`convert_transforms_to_colmap.py` and `step_colmap.py`). Glome drives COLMAP via subprocess execution with `--keep_database`, invoking `feature_extractor` (SIFT), `sequential_matcher`, and `point_triangulator` with fixed ARCore poses, plus soft pose-prior bundle adjustment (`pose_prior_mapper`).
- **Motive & Rationale:**
  COLMAP is the gold standard for robust geometric triangulation and multi-view bundle adjustment. Driving it with fixed/prior ARCore poses prevents scale drift and focal length wandering while generating verified multi-view point tracks and SQLite match databases for Sampson error diagnostics.
- **Last Verified / Checked:** 2026-09-15.

---

### Depth Anything v3 (DA3) [https://github.com/HKU-Edtech/Depth-Anything-3]

- **Where & How Used in Glome (Pass 2):**
  Integrated in `backend/02_depth_estimation/` via `third_party/depth_anything_3` and consumed by `depth_priors.py` and `step_depth.py`. Evaluates keyframes in sliding-window chunks with OpenCV $w2c$ poses to produce metric depth arrays (`.npy`) and normal vectors.
- **Motive & Rationale:**
  DA3 is the state-of-the-art multi-view monocular metric depth foundation model. Its learned cross-view transformer attention yields metric consistency across large indoor spaces without requiring per-scene neural training.
- **Last Verified / Checked:** 2026-09-15.

---

### `diff-surfel-rasterization` (2DGS Native C++/HIP Extension)

- **Where & How Used in Glome (Pass 3):**
  Integrated in `backend/03_2DGS_training/submodules/diff-surfel-rasterization` and consumed by `gaussian_renderer/__init__.py`. Provides hardware-accelerated forward and backward differentiable rasterization of 2D planar Gaussian surfels.
- **Motive & Rationale:**
  Enforces true 2D planar disk projection, exact normal derivation, and low-latency front-to-back alpha blending directly on GPU/ROCm hardware. Essential for generating thin, clean architectural surfaces.
- **Last Verified / Checked:** 2026-09-15 (Compiled with `--no-build-isolation` on ROCm 7.1).

---

### `simple-knn` (Spatial K-Nearest Neighbors CUDA/HIP Extension)

- **Where & How Used in Glome (Pass 3):**
  Integrated in `backend/03_2DGS_training/submodules/simple-knn` and consumed by `scene/gaussian_model.py`. Computes mean k-NN spatial distances ($k=3$) to initialize scale dimensions for newly spawned or densified Gaussian primitives.
- **Motive & Rationale:**
  Fast $O(N \log N)$ GPU-accelerated nearest-neighbor queries ensure initial Gaussian disk radii adapt smoothly to local point density without creating geometric micro-gaps.
- **Last Verified / Checked:** 2026-09-15.

---

### `lpipsPyTorch` (Learned Perceptual Image Patch Similarity)

- **Where & How Used in Glome (Pass 3):**
  Vendored in `backend/03_2DGS_training/lpipsPyTorch/` and executed via `metrics.py`. Evaluates perceptual similarity between novel 2DGS view renders and ground truth keyframes.
- **Motive & Rationale:**
  Provides a standalone PyTorch implementation of LPIPS with zero external pip networking dependencies, ensuring reproducible quality gating in offline listing pipelines.
- **Last Verified / Checked:** 2026-09-15.

---

### PyTorch & PyTorch ROCm [https://pytorch.org/]

- **Where & How Used in Glome (Passes 0 to 3):**
  Foundational deep learning and tensor processing engine across `backend/`. Includes custom patches in `point_utils.py` and `rasterizer_interface.py` to handle AMD ROCm/RDNA4 (`gfx1200`) GEMM row limits and quantile memory allocations.
- **Motive & Rationale:**
  Standard Python tensor framework with direct ROCm support on AMD Radeon hardware (RX 9060 XT), enabling high-throughput parallel autograd optimization.
- **Last Verified / Checked:** 2026-09-15.

---

### OpenCV (`opencv-python` / `cv2`) [https://opencv.org/]

- **Where & How Used in Glome (Passes 0 to 2):**
  Used across `00_ingestion/` (`quality_gate.py`, `keyframe_selector.py`, `pose_aligner.py`) and `01_poses_refinment/` for two-axis Laplacian blur detection, Shi-Tomasi corner tracking, Lucas-Kanade optical flow, SIFT feature extraction, and image transposition.
- **Motive & Rationale:**
  Industry-standard high-performance computer vision library with optimized C++ image processing primitives.
- **Last Verified / Checked:** 2026-09-15.

---

### SciPy & NumPy [https://scipy.org/]

- **Where & How Used in Glome (Passes 0 to 3):**
  Used throughout `backend/` for quaternion SLERP trajectory interpolation (`scipy.spatial.transform.Rotation`), cubic spline trajectory fitting, KDTree spatial lookups (`scipy.spatial.KDTree`), and dense array calculations.
- **Motive & Rationale:**
  Core scientific computing stack providing numerical routines for geometric interpolation and spatial partitioning.
- **Last Verified / Checked:** 2026-09-15.

---

### Open3D [http://www.open3d.org/]

- **Where & How Used in Glome (Pass 1 & Pass 2):**
  Used in `01_poses_refinment/densify_pointcloud.py` and `02_depth_estimation/initialization.py` for spatial voxel downsampling, statistical outlier removal (SOR), and point cloud I/O.
- **Motive & Rationale:**
  Efficient 3D point cloud algorithms for filtering airborne noise and cleaning sparse SfM models.
- **Last Verified / Checked:** 2026-09-15.

---

### Google ARCore / ARKit [https://developers.google.com/ar]

- **Where & How Used in Glome (Mobile Capture & Pass 0 Ingestion):**
  Used in `mobile/android/` for Visual-Inertial Odometry (VIO) tracking, continuous 60 Hz 6-DoF camera pose estimation, and `SharedCamera` hardware synchronization. Poses are logged in `transforms.json` and ingested in Pass 0.
- **Motive & Rationale:**
  Hardware-accelerated VIO delivers sub-millisecond metric motion tracking on commodity smartphones without external beacons or markers.
- **Last Verified / Checked:** 2026-09-15.

---

### Android Camera2 API [https://developer.android.com/training/camerax/architecture]

- **Where & How Used in Glome (Mobile Capture & Pass 0 Ingestion):**
  Used in `mobile/android/` (`CameraPipeline.kt`) to enforce manual hardware register overrides (60 FPS sensor readout, locked manual WB, locked joint shutter/ISO, disabled software OIS/EIS, disabled auto lens distortion).
- **Motive & Rationale:**
  Direct hardware register control preserves static optical intrinsics $\mathbf{K}$, which is required for multi-view SfM and 2DGS radiance field convergence.
- **Last Verified / Checked:** 2026-09-15.

---

## Evaluated & Compared Projects (Not Integrated)

### **ZipDepth [https://github.com/zipdepth/zipdepth]
- **Where & Why Evaluated:** Evaluated on Android (`mobile/android/app/src/main/assets/zipdepth_base_256x256.onnx`) for on-device depth guidance. Replaced on the backend by DA3 for metric indoor accuracy.
- **Last Checked:** 2026-09-10.

---

### **MiDaS [https://github.com/isl-org/MiDaS]
- **Where & Why Evaluated:** Evaluated for zero-shot monocular depth estimation. Rejected due to scale-shift ambiguity and depth boundary bleeding.
- **Last Checked:** 2026-09-10.

---

### **SuperSplat (PlayCanvas) [https://github.com/playcanvas/supersplat]
- **Where & Why Evaluated:** Used as a web-based 3D Gaussian Splatting manual inspection tool for reviewing canonical PLY exports (`model_3dgs_iter_XXXX.ply`).
- **Last Checked:** 2026-09-11.

---

### **Blackmagic Camera App [https://www.blackmagicdesign.com/products/blackmagiccamera]
- **Where & Why Evaluated:** Benchmarking reference for manual smartphone camera exposure and shutter curves.
- **Last Checked:** 2026-09-10.

---

### **MipMap Software & Sharp Frame**
- **Where & Why Evaluated:** Third-party photogrammetry frame extractors; replaced by Glome's automated two-axis Laplacian quality gating and keyframe selection.
- **Last Checked:** 2026-09-10.
