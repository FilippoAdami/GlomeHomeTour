# Spatial Capture Architecture & Strategic Product Plan: 3D Gaussian Walkthrough Engine

## 1. Executive Summary & Strategic Rationale

This document defines the technical architecture, sensor configuration, UX guidance mechanics, and product tiering strategy for our spatial indoor capture platform.

The objective is to produce high-fidelity 3D spatial reconstructions (via 2D/3D Gaussian Splatting and monocular depth priors) using standard, commodity smartphones without requiring LiDAR hardware or specialized external 360° cameras.

**Core Architectural Pillars**

* **Continuous Guided Video Walkthrough > Static 360° Panoramas:** Video paths capture the multi-view motion parallax required for 2D/3D Gaussian Splatting.
* **Ultra-Wide Fixed-Focus Sensor > Main Autofocus Lens:** Eliminates focus hunting and keeps intrinsic parameters strictly static throughout capture.


* **Low-Level Hardware Overrides > OS Computational Auto-Features:** Locks shutter, ISO, white balance, and lens distortion to preserve rigid perspective geometry.



---

## 2. Capture Modality: Guided Video Trajectory vs. 360° Panoramas

A foundational technical decision is capturing continuous, trajectory-based video rather than static 360° panoramas from the room center.

**Why Static 360° Panoramas Fail for 3D Reconstruction**

* **Zero Parallax Baseline:** A single 360° photo has only one optical center $(\mathbf{p}_x, \mathbf{p}_y, \mathbf{p}_z)$. Without baseline displacement ($\Delta \mathbf{p} > 0$), depth triangulation and radiance field training fail.
* **Severe Occlusion Shadows:** Anything behind kitchen islands, tables, or partitions remains entirely invisible and creates gaping holes in the final 3D model.
* **Distortion Incompatibility:** Equirectangular projection distorts perspective geometry, degrading monocular depth estimation models like Depth Anything v3.
* **Commercial Limitation:** Limits the output to traditional 3-DoF panoramic bubble viewers (Google Street View style).

**Why Continuous Guided Video Walkthrough Succeeds**

* **Dense Multi-View Parallax:** Moving the camera through space provides continuous baseline displacement, enabling accurate 3D triangulation and sub-centimeter metric precision.
* **Active Occlusion Elimination:** The user is guided around obstacles to capture 100% of room surfaces.
* **Native Perspective Projection:** Standard pinhole perspective video feeds cleanly into depth estimators and Gaussian rasterizers.
* **Rich Commercial Deliverables:** Generates photorealistic 6-DoF interactive walkthroughs, metric CAD/BIM floor plans, and virtual staging.

---

## 3. Optical Selection: Why the Ultra-Wide Lens is Primary

A major failure mode in indoor photogrammetry is autofocus hunting and focus breathing. The **Ultra-Wide Lens ($0.5\times$, $\sim 13\text{mm}-15\text{mm}$ equivalent)** is designated as our primary capture sensor due to its optical behavior:

* **Fixed Hyperfocal Distance:** Ultra-wide smartphone lenses have tiny physical focal lengths ($f \approx 1.5\text{mm}-2.2\text{mm}$), creating a hyperfocal distance of roughly $15\text{ cm}$. Everything from $15\text{ cm}$ to infinity stays optically sharp without moving lens elements.


* **Zero Focus Breathing:** Because the physical lens elements do not translate, the intrinsic camera matrix $\mathbf{K}$ (focal length $f_x, f_y$ and principal point $c_x, c_y$) remains mathematically static throughout the entire session.


* **Hallway & Close-Proximity Stability:** Moving through narrow doorways ($< 1\text{ meter}$) will not trigger accidental background or foreground defocusing.


* **Automatic Distortion Modeling:** While ultra-wide lenses exhibit barrel distortion, Visual-Inertial Odometry (ARKit / ARCore) and Structure-from-Motion (SfM) solvers model radial and tangential distortion parameters ($k_1, k_2, k_3, p_1, p_2$) natively, solving distortion automatically during bundle adjustment.



---

## 4. Hardware-Level Camera Configuration

To maintain photogrammetric rigor, the app bypasses standard OS auto-camera processing and enforces strict manual overrides:

* **Sensor Readout Rate:** 60 FPS. Minimizes rolling shutter skew during camera motion, matching real pixel projections with rigid mathematical models.


* **Recording Decimation:** Subsample and record 1 frame every 6–10 frames. Reduces file sizes (~10x smaller) and prevents mobile SoC overheating during long scanning sessions.


* **Video Codec & Bitrate:** H.265 (HEVC) at maximum bitrate. Preserves fine edge data and high-frequency gradients critical for stereo matching while maintaining high compression efficiency. (H.264 fallback available for legacy tools).


* **Shutter Speed & ISO (jointly metered):** Locked-exposure, but not to a single fixed pair — pre-flight ambient-light metering picks a shutter/ISO point on a shared tradeoff curve and locks it for the session. Shutter ranges $1/500\text{ s}$ (bright/outdoor, minimizes motion blur) down to $1/50\text{ s}$ (dim indoor rooms, admits more light without pushing ISO into visible grain); ISO ranges 50 (bright/outdoor baseline, minimal noise) up to a device-appropriate ceiling for dim rooms. Metering walks shutter down from its fastest end first — since indoor scenes are rarely fast-walking-blur-limited at typical agent pace — and only raises ISO once shutter has bottomed out at $1/50\text{ s}$ and the scene still isn't at target exposure. This is a genuine tradeoff, not a knob with a free answer: slower shutter costs more motion blur risk during handheld walking, higher ISO costs more grain/noise — the app picks the point on that curve closest to target exposure for the *specific* scene's ambient light, re-metered fresh each session, not a fixed indoor/outdoor preset.


* **Digital Image Stabilization:** HARD DISABLED. Software stabilization dynamically crops, shifts principal points, and modifies focal length, corrupting intrinsic matrices. (Use physical gimbals for stabilization if needed).


* **Auto-Lens Distortion Correction:** HARD DISABLED. Preserves raw optical distortion so downstream reconstruction solvers can model true physical optics.


* **White Balance:** Manually tuned and locked pre-flight. Prevents color temperature shifts across views that cause cloudiness or radiance artifacts.



---

## 5. End-to-End System Architecture & Data Flow

**Sensor Input Layer**

* Standard RGB sensor streaming at 60 FPS with fixed exposure, ISO, WB, and hyperfocal focus.


* IMU sensor providing high-frequency acceleration and angular velocity to 6-DoF VIO (ARKit / ARCore).
* Sub-millisecond hardware timestamp synchronization between camera exposure midpoint and VIO poses.
* **Standardized JSON Schema Contracts (`shared/schemas/`):** All mobile capture sessions output `schema_version: "1.0.0"` compliant manifests:
  * `transforms.json`: OpenCV pinhole camera model, intrinsic focal lengths ($f_x, f_y$), principal points ($c_x, c_y$), resolution ($w, h$), field-of-view angle, distortion coefficients, and per-keyframe $4\times 4$ camera-to-world extrinsics with hardware timestamp synchronization.
  * `trajectory.csv`: Continuous 60 Hz VIO 6-DoF trajectory stream logging every tracked frame ($t_x, t_y, t_z, q_x, q_y, q_z, q_w$, tracking state, export flag).
  * `coverage_summary.json`: Session metrics (duration, coverage fraction, voxel stats, frame drop diagnostics by reason, tracking loss counter, depth source).
  * `focus_metadata.json`: Per-frame optical metadata (focus distance in diopters & metres, autofocus state/mode, physical focal length).

**Illumination & Quality Gatekeeper**

* Computes mean frame luma ($Y_{\text{mean}}$) in real time.
* Rejects dark frames ($Y_{\text{mean}} < 40$).
* Rejects transient exposure spikes ($\vert{}\Delta Y\vert{} > 30$) when room lights are toggled mid-scan, dropping unstable transition frames while keeping VIO tracking active.
* Subsamples the 60 FPS stream to export 6–10 verified keyframes per second.



**Real-Time UI & Background Workers**

* **UI Thread (60 Hz):** Renders dynamic AR coverage overlays, motion warnings, and 3D guidance vectors.
* **Background Worker (10–15 Hz):** Runs a 10 cm sparse voxel hash map, performs ray casting, checks parallax angles ($\theta_{\text{parallax}} \ge 25^\circ$), and calculates centroids of occluded volumes.

**Export Package & Reconstruction Backend**

* **Package Contents:** Clean 1080p/4K frames, `transforms.json` with initial focal length priors, and raw `trajectory.csv` logs.


* **Backend Processing:** Monocular metric depth estimation (Depth Anything v3) and 2D/3D Gaussian Splatting radiance field reconstruction.

---

## 6. User Experience (UX) & Guided Interface

**Pre-Flight Setup Flow**

* **Step 1: Lens Selection:** Defaults to Ultra-Wide ($0.5\times$) for room capture. (Optional detail mode uses main lens with tap-to-lock autofocus).


* **Step 2: Interactive White Balance Tuning:** A reticle appears over the live preview with an intuitive slider: *"Point at a neutral wall and adjust until colors match your room, then tap Lock Color."*

* **Step 3: Trajectory Horizon Guide:** An on-screen level indicator prompts the user to hold the camera at a downward tilt ($5^\circ - 20^\circ$) optimized for environment acquisition (prioritizing floor and room geometry while avoiding excessive empty ceiling or sky).



**In-Flight Active Guidance Mechanics**

* **Dynamic Voxel Mesh Overlay:** Real-time translucent mesh highlights unvisited spaces in amber, turning clear as sufficient multi-angle parallax ($\theta_{\text{parallax}} \ge 25^\circ$) is achieved.
* **Occlusion Guidance Arrows:** Computes the 3D center of hidden surface pockets (such as the back of a kitchen island) and projects floating directional arrows guiding the user around the obstacle.
* **Motion Guardrail Warnings:** Immediate alerts display if linear velocity exceeds $0.4\text{ m/s}$ or angular rotation exceeds $30^\circ/\text{s}$.


* **Light Switch Resilience:** Automatically suppresses dark and transiently overexposed frames when room lights are turned on midway, without restarting the session.

---

## 7. Business Model & Feature Tiering

Standardizing on a guided video walkthrough allows clear feature separation based on compute cost:

**Free Tier (Zero Cloud Compute Cost)**

* **Instant 2D Floor Plan Sketch:** Generated on-device from the 2D bounding perimeter of the ARKit/ARCore trajectory.
* **Interactive 2.5D Keyframe Tour:** Browser-based viewer allowing step-through inspection of clear keyframes along the walking path.
* **Standard Walkthrough Video Export:** 1080p stabilized fly-through video rendered directly from captured frames.

**Paid Pro Tier (Full Radiance & CAD Pipeline)**

* **Full 2D/3D Gaussian Splatting (2DGS):** Photorealistic, interactive 6-DoF 3D digital twin.
* **Metric CAD / BIM Floor Plans:** Precision wall dimensions, structural offsets, and DXF/PDF vector exports.
* **Virtual Staging & Measurement Suite:** In-browser metric measurements and automated 3D furniture placement.

---

## 8. Technology Stack & Platform Execution

* **Native Core Execution:** Swift (iOS / ARKit / Metal) and Kotlin (Android / ARCore / OpenGL ES) for real-time sensor loops, sub-millisecond timestamp synchronization, and manual camera register overrides.
* **Shared Logic Layer:** Kotlin Multiplatform (KMP) manages serialization, trajectory formats, state machines, and export manifests (`transforms.json`).
* **Hardware Target:** Standard Android and iOS devices released within the past 4 years; no LiDAR or specialized hardware required.
---

## 9. Current Implementation Status & Real-Time Guidance Evolution (2026-09-02)

### On-Device Mobile Capture Engine (`mobile/android`)
- **Sensor Pipeline:** Enforced Camera2 manual register overrides via `SharedCamera` (60 FPS sensor readout, locked ISO/shutter, manual/auto WB locked, software OIS/EIS hard disabled).
- **5-Second Dynamic Lighting & Auto White-Balance Calibration:** Pre-flight multi-angle 5-second sweep automatically calculates optimal ISO/shutter (ISO 100-800, 1/120s shutter baseline) and locks grey-world RGB gains without requiring manual slider friction.
- **Sparse Multi-View Parallax & Ray Triangulation (`FeatureParallaxTracker`):**
  - Live closed-form two-ray geometric triangulation solves exact line-of-sight intersections ($\ge 12^\circ$ baseline), snapping points to real 3D physical surfaces and removing monocular depth floaters.
  - Verified points are promoted to **Permanent 3D Landmarks (Emerald Green)** in world space and never discarded when out of frame.
  - Unverified single-view candidate points (Amber) fade out when out of view.
- **Gamified Multi-Object 3DGS Capture UX:**
  - Guides users to dynamically orbit furniture and objects of interest to accumulate **350–500 verified landmarks** (200 point baseline minimum).
  - Real-time 2D Bird's-Eye Mini-Map tracks user location, orientation, and room coverage footprint.
- **Enforced Loop Closure:** Requires returning to the starting doorway anchor ($< 1.0\text{m}$) to seal loop closure before export.

### On-Device Quality Verification & 3D Point Cloud Viewer (`mobile_app/`) (2026-11)
- **Standalone 3D Point Cloud Inspection Viewer (`PointCloudViewerActivity` & `PointCloudRenderer`):**
  - Ultra-optimized OpenGL ES 2.0 viewer rendering spatial hash voxel-deduplicated point clouds and VIO trajectory paths directly on an opaque dark canvas (`#0B0E14`).
  - **3 Navigation Modes:** 3D Orbit (touch rotation, pinch-to-zoom, pan), Top-Down (bird's-eye floorplan orthographic view), and Immersive Gyro Walk (1:1 physical look-around via `Sensor.TYPE_GAME_ROTATION_VECTOR` with closed-form $Z$-to-$Y$ axis remap, touch drag walking along look heading, and room boundary clamping).
  - **Cohesive Surface Shading:** Solid-core anti-aliasing discs with dynamic high-DPI scaling (`⚪ Thick`, `⚪ Max`, `⚪ Fine` presets) and toggleable subtle translucent amber trajectory thread (`🛤 Track`).
  - **Height-Based Color Ramp:** Turbo 5-stop gradient mapping vertical elevation from navy floor to coral ceiling.
- **Card-Based UI/UX Overhaul & Scan Integration:**
  - Redesigned `HomeActivity`, `PropertyDetailActivity`, and `ScanGalleryActivity` with clean property and room scan cards, real-time status badges (`PROCESSED`), and direct action buttons (`Inspect 3D`, `Add Geometry`, `Process`, `Delete`).
  - **Scan Integration:** Viewfinder semi-transparent overlay of the previous scan's final keyframe allows users to align seamlessly and integrate additional geometry into existing scans.
  - **Non-Reversible Pre-Processing Warning Modal:** Warning modal prompts users to inspect scan quality in 3D before irreversible processing.
- **Detailed Documentation:** Complete mobile architecture, data schemas, and build instructions documented in [`mobile_app/README.md`](file:///home/monday/Desktop/GlomeHomeTour/mobile_app/README.md).

### Backend Reconstruction & Metric Priors Engine (`backend/`) (2026-09-10/11)
- **Commercial Depth Anything 3 (`DA3-BASE`, Apache 2.0):**
  - Integrated and validated `depth-anything/DA3-BASE` (0.12B parameters, commercially unrestricted under Apache 2.0), replacing non-commercial CC BY-NC models.
  - Accelerated multi-view sliding-window inference by **$2.1\times$** on AMD ROCm (4.1 min for 273 keyframes vs 8.9 min for Giant) while achieving planar metric fidelity matching Giant.
- **Scale-Drift-Free Sliding Window Ensembling:**
  - Eliminated compounding pairwise inter-chunk scale drift ($\prod s_i$) in favor of direct multi-window median consensus on calibrated depth tensors, completely removing multi-layer wall duplications ("onion peeling").
- **Dynamic Statistical Depth Ceiling (`estimate_adaptive_depth_ceiling`):**
  - Room depth horizon is dynamically derived per scan using statistical surface distribution ($Q_{0.98} + 1.5 \cdot \text{MAD}$), auto-scaling from compact rooms ($4.13\text{m}$ for bedroom) to grand hotel lobbies ($25\text{m}+$) without any hardcoded thresholds.
- **Occlusion-Aware Multi-View Consensus (`min_consensus=1`):**
  - Prunes floating artifacts and non-surface rays by reprojecting 3D points into neighboring camera frustums, while safely preserving uniquely viewed ceiling and corner patches to guarantee continuous, hole-free reconstructions.
- **FastGS + DN-Splatter Geometric Supervision Pipeline (`backend/03_FastGS_DNSplatter/`):**
  - High-speed 3D Gaussian Splatting training with multiview-aware tile & sample selection, accelerating radiance field convergence.
  - Joint DAv3 metric depth supervision and StableNormal unit surface normal supervision (`--depth_supervision`, `--normal_supervision`, `--depth_loss pearson`), eliminating floaters and enforcing crisp planar surface geometry.
  - Initialized directly from the refined DA3 dense surfel cloud (`points3D_depth.ply` $\to$ `sparse/0/points3D.ply`).
- **Export & Verification:**
  - Fully verified on real-world capture packages: exports clean, colored 3D Gaussian Splatting models and surfels into standard interchange formats with 100% compliant shared schema contracts.

### Continuous Cinematic Flythrough & Paired Rendering Engine (`backend/08_artifacts_generation/`) (2026-11)
- **F1 Curvature-Aware Velocity Scheduling & Kinematic Trajectory (`generate_cinematic_flythrough.py`):**
  - Continuous 6-DoF flight through all numbered checkpoints with pinned photo poses (exact matches within $10^{-6}$ atol) and zero roll wobble ($0.0^\circ$ open-flight roll).
  - Two-pass DP curvature-constrained velocity schedule: peak cruise speed of **$1.25\text{ m/s}$**, corner apex speed $v_{\text{apex}} = \min(v_{\text{max}}, \sqrt{a_{\text{lat}} / \max(\kappa, 10^{-8})})$, asymmetric braking ($a_{\text{brake}} = 1.10\text{ m/s}^2$) establishing crisp corner entry braking zones, and acceleration ($a_{\text{accel}} = 0.95\text{ m/s}^2$) powering smoothly out of apexes.
  - Doorway passages centered through door openings with fallback safety offsets; snappy door transits ($\le 0.5\text{ s}$) at up to $1.25\text{ m/s}$ with quintic Slerp orientation transitions.
  - Continuous panorama viewpoints receiving $\ge 240^\circ$ sweeps paced down to $\le 0.35\text{ m/s}$ ($0.22\text{ m/s}$ on floor transitions), smoothly continuing into the terminal overhead ceiling ascent ($+60^\circ$).
  - Strict physical bounds verified: inside acceleration $\le 1.21\text{ m/s}^2$ (measured $1.1929\text{ m/s}^2$), turn angular speed $\le 65.1^\circ/\text{s}$ (measured $64.72^\circ/\text{s}$), cruise angular speed $\le 37.6^\circ/\text{s}$ (measured $14.44^\circ/\text{s}$), and non-door clearance $\ge 0.08\text{ m}$ (measured $0.0935\text{ m}$).
- **Native GPU Hardware MSAA & Fast SE(3) Inversion:**
  - Configured Open3D Filament with native $4\times$ multisample antialiasing (MSAA), rendering directly at target preview and final resolutions and completely eliminating CPU-side supersampling and `cv2.resize` downscaling ($3.59\text{ ms/frame}$, a **$3.05\times$ speedup**).
  - Replaced $O(N^3)$ LU matrix inversions with closed-form Euclidean camera pose transpositions ($R^T$ and $-R^T t$), with cached CUDA intrinsic and background tensors.
  - Direct in-place slice composition in `compose_minimap()` avoiding redundant full-frame image allocations.
- **Asynchronous Video Encoding & Dual-Pass In-Memory Caching (`--mode both`):**
  - Decoupled ffmpeg pipe I/O from GPU rendering using background worker threads (`AsyncVideoWriter` with bounded queue), slashing encoding pipe latency to $0.12\text{ s}$ preview / $0.58\text{ s}$ final (down from ~11 s).
  - Shared in-memory caching of preloaded GLB mesh geometry and the PyTorch `GaussianRenderer` across preview and final passes, reducing pass-to-pass asset reload overhead from $12.36\text{ s}$ to $0.62\text{ s}$ (**$19.9\times$ faster**).
- **Benchmark Performance (`current_scene`):**
  - Preview ($270\times 480$): **52.89 s** (Input validation & setup: 0.9s, Planning & retiming: 9.3s, Model loading: 11.9s, Mesh: 19.87s, Gaussian: 9.35s, Minimap: 0.38s, Async Encoding IO: 0.12s).
  - Final ($540\times 960$): **47.49 s** (Verification: 0.7s, In-memory model reuse: 0.62s, Mesh: 34.59s, Gaussian: 9.51s, Minimap: 0.81s, Async Encoding IO: 0.58s).
  - Combined Active Wall Time: **100.38 s** (~1 min 40 s) for 3,822 frames (7,644 total video frames across mesh and Gaussian channels), delivering a **33.82 s (25%) speedup** over baseline.
  - Complete logging streamed to `generate_cinematic_flythrough.log` and structured atomic timers published in `generate_cinematic_flythrough_manifest.json`. All 68 cinematic unit tests pass.

---

## 10. Bibliography & Project References (`bibliography/`)

Academic papers, mathematical techniques, and open-source software projects referenced or evaluated during the architectural design and implementation of the Glome Suite are indexed in the [`bibliography/`](file:///home/monday/Desktop/GlomeHomeTour/bibliography/) folder:

1. **[papers.md](file:///home/monday/Desktop/GlomeHomeTour/bibliography/papers.md):** Academic literature read and referenced across the app and backend. Each entry provides the paper title, publication year, venue/journal, authors, a concise summary, a dedicated paragraph detailing how and where it was applied in Glome, and direct links to ArXiv or publisher repositories.
2. **[Techniques.md](file:///home/monday/Desktop/GlomeHomeTour/bibliography/Techniques.md):** Algorithmic and mathematical techniques (e.g. low-level Camera2 hardware overrides, photometric luma spike gating, closed-form two-ray geometric triangulation, scale-shift metric depth graph alignment, and multi-view depth consistency filtering).
3. **[Projects.md](file:///home/monday/Desktop/GlomeHomeTour/bibliography/Projects.md):** Open-source libraries, SDKs, and developer tools (e.g. ARCore, Depth Anything v3, ONNX Runtime Mobile, PyTorch ROCm, OpenCV, SciPy).

> [!NOTE]
> **Pipeline Integration Notation (`**`):**
> Entries prefixed with double stars (`**`) denote models, algorithms, papers, or tools that were evaluated, benchmarked, or compared against during development (e.g. MiDaS, ZipDepth, Marching Cubes, SuperPoint) but are **not** part of the active production pipeline. Active, integrated components carry standard un-prefixed headers.

### Bibliography Maintenance Rules

When adding new research papers, mathematical techniques, or open-source dependencies to the Glome Suite:
* **Academic Papers:** Add to `bibliography/papers.md` using the standard layout: `Title (Year) [Venue] - Authors`, followed by a 1-paragraph description, a 1-paragraph Glome usage breakdown, and an ArXiv/web link.
* **Techniques:** Add to `bibliography/Techniques.md` under `Active & Integrated` if incorporated into source code, or under `Evaluated & Compared` with a `**` prefix if tested as a benchmark/alternative.
* **Projects & Libraries:** Add to `bibliography/Projects.md` with official repository links and a justification paragraph detailing why the library was selected over alternatives.
* **Contract Integrity:** Keep references synchronized with `project_history.md` logs in active subsystem folders (`mobile/android/`, `backend/00_ingestion/`, `backend/03_FastGS_DNSplatter/`).

