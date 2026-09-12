# Bibliography: Techniques

This document lists the core mathematical, algorithmic, and engineering techniques implemented or evaluated across the Glome Suite mobile capture app (`mobile/android/`) and backend ingestion/reconstruction system (`backend/`).

Items prefixed with `**` represent techniques that were evaluated, benchmarked, or compared against during development but were not integrated into the active production pipeline.

---

## Active & Integrated Techniques

### Low-Level Hardware Camera Register Overrides & Joint Exposure Metering

**Description:**
Bypasses high-level operating system auto-camera processing by configuring low-level camera register overrides via Android's Camera2 `SharedCamera` interface. Enforces 60 FPS sensor readout, locked manual white balance, disabled software image stabilization (OIS/EIS), disabled auto lens-distortion correction, and joint shutter/ISO exposure metering on a shared tradeoff curve ($1/500\text{s}$ fast shutter ceiling down to $1/50\text{s}$ indoor floor before increasing ISO).

**How, Where & Why Used in Glome:**
Implemented in `mobile/android/app/src/main/java/com/glomehometour/arscan/CameraPipeline.kt`. Chosen over default Android CameraX auto-exposure because automatic computational shifts alter focal length, principal point, and frame-to-frame exposure, corrupting camera intrinsics $\mathbf{K}$ and breaking monocular structure-from-motion assumptions.

---

### Photometric Mean Luma & Transient Illumination Spike Gating

**Description:**
Calculates real-time frame luminance ($Y_{\text{mean}} = 0.2126R + 0.7152G + 0.0722B$) across live camera feeds. Rejects under-exposed frames ($Y_{\text{mean}} < 40$) and transient light-switch illumination spikes ($\vert\Delta Y\vert > 30$ between consecutive frames) while maintaining continuous 60 Hz VIO trajectory tracking. Also includes a 5-second multi-angle pre-flight grey-world calibration sweep to lock baseline RGB gains.

**How, Where & Why Used in Glome:**
Implemented in `mobile/android/app/src/main/java/com/glomehometour/arscan/Photometric.kt` and `backend/ingestion/quality_gate.py`. Chosen to prevent ruined keyframes (e.g. from real estate agents toggling light switches or entering dim hallways) from corrupting downstream feature matching without losing session tracking state.

---

### Closed-Form Two-Ray Geometric Triangulation & Multi-View Parallax Tracking

**Description:**
Computes closed-form geometric line-of-sight ray intersections from pair camera poses across frames with sufficient angular baseline displacement ($\ge 12^\circ$). Candidate features that converge to physical surfaces are promoted to permanent 3D world-space surface landmarks (Emerald Green), while unverified single-view candidates (Amber) cleanly fade out when out of view.

**How, Where & Why Used in Glome:**
Implemented in `mobile/android/app/src/main/java/com/glomehometour/arscan/FeatureParallaxTracker.kt`. Chosen over monocular depth estimation on-device because closed-form triangulation guarantees exact metric multi-view spatial consistency without floater artifacts, providing stable 3D visual feedback for spatial coverage HUDs.

---

### Tri-State Sparse Voxel Occupancy Mapping

**Description:**
Maintains a 10 cm 3D spatial voxel hash map classifying space into three distinct states along camera ray casts: FREE (voxels in front of hit points), OCCUPIED (voxels containing surface hit points), and OCCLUDED (voxels behind hit points from the current viewpoint). Evaluates multi-angle observation parallax ($\ge 25^\circ$) to verify surface completion and calculates centroids of occluded volumes.

**How, Where & Why Used in Glome:**
Implemented in `mobile/android/app/src/main/java/com/glomehometour/arscan/VoxelGrid.kt` and `CoverageWorker.kt`. Chosen over simple 2D bounding boxes or 2-state grids because tri-state tracking differentiates between unobserved open space and occluded space behind obstacles (e.g. kitchen islands), driving targeted occlusion guidance arrows.

---

### Laplacian Variance Motion Blur & Image Quality Filtering

**Description:**
Evaluates image sharpness by computing the variance of the 2D Laplacian operator ($\nabla^2 I$) over keyframe images. Frames with Laplacian variance below a calibrated blur threshold are identified as motion-blurred and filtered out prior to bundle adjustment.

**How, Where & Why Used in Glome:**
Implemented in `backend/ingestion/quality_gate.py` (`QualityGate.compute_laplacian_variance`). Chosen because calculating second-derivative intensity gradients is computationally lightweight ($O(N)$ CPU pass) and highly effective at eliminating handheld walking motion blur before feature extraction.

---

### OpenGL to OpenCV Pose Coordinate Transformation & SLERP Quaternion Interpolation

**Description:**
Transforms camera poses from OpenGL/ARCore conventions (+Y up, -Z forward, camera-to-world) into OpenCV/photogrammetry conventions (+Y down, +Z forward, world-to-camera). Uses Spherical Linear Interpolation (SLERP) on rotation quaternions and cubic spline interpolation on translation vectors to align sub-millisecond keyframe timestamps to continuous VIO trajectory logs.

**How, Where & Why Used in Glome:**
Implemented in `backend/ingestion/pose_aligner.py`. Chosen because ARCore and backend computer vision libraries use conflicting coordinate conventions; precise SLERP alignment reduces inter-view pose synchronization errors down to sub-millisecond precision.

---

### Monocular Scale-Shift Metric Depth Graph Alignment & Surface Normal Calculation

**Description:**
Aligns monocular depth map predictions against sparse 3D VIO/SfM landmarks by solving an optimal least-squares scale ($s$) and shift ($t$) transform ($d_{\text{metric}} = s \cdot d_{\text{mono}} + t$). Connects sequential and loop-closure keyframes in a sparse linear depth graph system, and computes 3D surface unit normal vectors from metric depth spatial gradients.

**How, Where & Why Used in Glome:**
Implemented in `backend/reconstruction/depth_priors.py` (`DepthPriorEstimator`, `GlobalDepthGraphResult`, `compute_surface_normals`). Chosen because monocular neural depth predictions exhibit scale drift across long video walks; solving a global sparse depth graph restores metric scale consistency (reducing cross-view error down to 0.7–1.5 cm).

---

### Multi-View Depth Consistency Filtering

**Description:**
Projects 3D unprojected depth points into neighboring keyframe camera frustums and checks depth agreement against neighboring depth maps. Points that fail cross-view depth corroboration are identified as non-surface artifacts (e.g. sky through windows, mirror reflections, open doorways) and pruned.

**How, Where & Why Used in Glome:**
Implemented in `backend/reconstruction/initialization.py` (`filter_multiview_consistency`). Chosen over single-frame thresholding because multi-view depth consistency filtering cleanly removes sky, window glass, and specular floaters before initializing surfel clouds.

---

## Evaluated & Compared Techniques (Not Integrated)

### **Distance-and-Angle Gated Keyframe Selection

**Description:**
Gated frame export based on spatial displacement thresholds (e.g. exporting a keyframe only when translation $\ge 8\text{cm}$ or rotation $\ge 6^\circ$).

**How, Where & Why Evaluated in Glome:**
Evaluated in early mobile app prototypes (`mobile/android/`). Replaced in favor of streaming decimation (subsampling 1 frame out of every 6–10 frames post-illumination gate) because fixed decimation preserves uniform temporal sampling and prevents frame loss during slow panning motions.

---

### **ARCore Native Monocular Depth & Plane Wireframing

**Description:**
Uses ARCore's native monocular depth API (`Environment Depth`) and plane detection (`HORIZONTAL_AND_VERTICAL`) to construct surface wireframe meshes on mobile devices.

**How, Where & Why Evaluated in Glome:**
Tested in early Android client builds. Discarded during field trials because monocular smartphone plane estimation produced severe phantom floating planes and visual HUD clutter; superseded by closed-form feature parallax landmarking (`FeatureParallaxTracker.kt`).

---

### **On-Device Volumetric Marching Cubes Mesh Reconstruction

**Description:**
Runs a CPU/GPU marching cubes isosurface extraction pass over live mobile voxel grids to generate translucent translucent mesh overlays.

**How, Where & Why Evaluated in Glome:**
Evaluated in early mobile specifications. Replaced by direct OpenGL point sprite rendering (`ArScanRenderer.kt`) to eliminate high mobile SoC thermal overhead and maintain 60 FPS UI performance on standard smartphones.

---

### **SuperPoint Deep Feature Extraction & Neural Matching

**Description:**
Uses a deep convolutional neural network (SuperPoint) to extract dense keypoints and descriptors for multi-view stereo matching.

**How, Where & Why Evaluated in Glome:**
Compared against SIFT in `backend/ingestion/sfm_refinement.py`. SIFT was retained as the production default because it provides robust classical feature matching without introducing PyTorch/GPU execution overhead during backend ingestion.
