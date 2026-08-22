# Glome Home Tour

## 1. Executive Summary & Product Vision

**Glome Home Tour** is an automated, AI-assisted spatial capture and reconstruction system designed for the residential real estate market. The product transforms standard monocular smartphone video captured by non-technical real estate agents into an MLS-compliant digital listing bundle: an interactive **3D Gaussian Splatting (3DGS) Web Walkthrough**, an **Automated 2D Floor Plan with Room Dimensions**, and a set of **Synthetic 4K 360° Equirectangular Panoramas** mapped directly to the floor plan.

To eliminate the universal failure mode of mobile spatial reconstruction (operator error, motion blur, incomplete scene coverage, and tracking failure), Glome couples a **Native VIO Mobile Guidance Capture App** with a **Headless Cloud/Desktop 3DGS Reconstruction Engine**.

## 2. Core Functional Requirements & Features

### A. Intelligent Mobile Capture & Guidance Engine

- **Kinematic & Motion-Blur Guard:** Real-time IMU monitoring enforcing angular velocity limits (ω<30∘/s) and maximum walking speed (v≈0.5 m/s) with dynamic on-screen HUD alerts.
- **Persistent Spatial Memory Grid:** World-locked 2D/3D ground plane tile occupancy grid (25 cm×25 cm resolution) rendering visual coverage feedback ("painting the floor") to guarantee complete room coverage.
- **VIO Tracking Protection & Blank Wall Safeguards:** Automated pitch angle bounds (θ∈[−70∘,+50∘]) and native feature-tracking state listeners. Automatically pauses video frame buffering during low-texture exposure (blank walls, uniform ceilings) to prevent corrupt data ingestion.
- **Keyframe Relocalization Engine:** Continuous background buffering of sparse visual landmarks and keyframe poses. In the event of tracking degradation, the interface displays an alignment ghost of the last valid spatial frame to restore coordinate alignment seamlessly.
- **Enforced Loop Closure Anchor:** Places a persistent spatial anchor at the entry door, locking capture finalization until the operator returns within 1.5 m of the origin.
- **On-Device Pre-Flight Validation Gate:** 5-second post-capture sanity check evaluating Laplacian blur variance, pose trajectory continuity, and duration-to-area metrics before uploading data to the compute engine.

### B. Spatial Reconstruction & Derivation Engine

- **Monocular Radiance Field Training:** High-throughput 3D Gaussian Splatting pipeline optimizing from monocular RGB video frames and SfM/VIO camera pose priors.
- **Gravity Alignment & Ground Plane Detection:** Automated leveling of the spatial scene to world-space coordinate gravity vectors.
- **Automated 2D Floor Plan Vectorization:** Density-based horizontal cross-section slicing between 1.0 m and 1.5 m above the detected floor plane, followed by planar line extraction (RANSAC) to extract clean wall polygons, structural boundaries, and room square meterage.
- **Semantic Viewpoint & Node Discovery:** Spatial clustering algorithm identifying geometric room centroids at standard human eye height (1.6 m) with dynamic wall-clearance checks to avoid occlusions.
- **360° Equirectangular Panorama Synthesis:** Automated 6-pass orthogonal cubemap rendering from each discovered room centroid, projected into standard 4K equirectangular images.

### C. Distribution & Delivery Interfaces

- **Interactive 3DGS Web Viewer:** Lightweight Three.js/WebGL-based player supporting compressed scene formats (≤25 MB) with smooth touch/first-person navigation.
- **Interactive 2D Minimap Widget:** Synchronized SVG/PNG floor plan with interactive node markers linking 2D locations to their corresponding 360° panoramic views.
- **Listing Portal Asset Pack:** Direct ZIP bundle export containing 4K equirectangular JPEGs, dimensioned PNG/PDF floor plans, and embeddable iframe snippets for direct integration into real estate listing portals (Idealista, Immobiliare.it, Zillow).

## 3. System Architecture & Technical Specifications

| **System Layer** | **Subsystem / Component** | **Technical Specification & Operational Parameters** |
| --- | --- | --- |
| **Mobile Capture (Client)** | Operating System & Hardware | Standard iOS (ARKit) and Android (ARCore) smartphones; 1080p/4K @ 30/60 fps. |
|  | Visual-Inertial Odometry | Native 6DOF pose estimation fusing IMU (100–200 Hz) with optical flow (30–60 Hz). |
|  | Video & Pose Packaging | Synchronized H.264/H.265 video stream paired with timestamped 6DOF camera trajectory JSON metadata. |
|  | Real-Time HUD & Shaders | Lightweight AR ground grid renderer; low-overhead UI overlay for speed, exposure, and coverage metrics. |
| **Backend Compute (Host)** | Compute Hardware Requirement | NVIDIA or AMD GPU (≥16 GB VRAM) supporting CUDA or ROCm architectures. |
|  | Ingestion & Frame Extraction | Automated keyframe downsampling, frame filtering, and timestamp-pose matching. |
|  | 3DGS Optimization Engine | GPU batch solver with density control, spherical harmonics pruning, and model quantization. |
|  | Turnaround Latency | ≤12–15 minutes total processing time for standard 80–120 sqm properties. |
| **Delivery & Storage** | Web Hosting & Storage | Serverless cloud hosting, distributed CDN asset delivery, and encrypted object storage. |
|  | Target Asset Payload | Compressed 3DGS asset: ≤25 MB; 360° panoramas: 4096×2048 JPEG (≈3–5 MB per room). |

## 4. Implementation Guidelines & Architectural Suggestions

### A. Mobile Client Development (The Capture Guard)

- **Decouple Tracking from Frame Recording:** Run the ARKit/ARCore session purely for pose extraction, spatial memory, and HUD feedback. Capture the primary video feed through a dedicated camera capture pipeline locked at a constant exposure, white balance, and focus to prevent optical instability during 3DGS training.
- **Maintain Lightweight Spatial Voxels:** Use a 2D bitset grid rather than a dense 3D mesh to track floor coverage. This keeps memory allocation negligible and prevents mobile thermal throttling during 5-minute recording sessions.
- **Graceful Relocalization Fallback:** If visual tracking degrades completely, do not terminate the session. Pause the video frame buffer, keep the audio/IMU stream alive, and provide clear directional UI arrows pointing toward the closest cached keyframe anchor.

### B. Backend Pipeline Orchestration (The Reconstruction Engine)

- **Hybrid Pose Ingestion:** Use the mobile device's recorded 6DOF VIO poses as an initial initialization prior for the Structure-from-Motion (SfM) solver. This accelerates feature matching and guarantees absolute metric scale without requiring manual reference markers.
- **Floor Plan Geometric Regularization:** Raw point slices contain noise from furniture, open doors, and curtains. Apply Manhattan World Assumptions (enforcing orthogonal 90∘ and 45∘ wall junctions) to regularize extracted 2D lines into architectural-grade floor plans.
- **Viewpoint Collision Avoidance:** Before finalizing a virtual 360° camera node at a room's geometric centroid, project radial rays in 360∘ within the point cloud. If an obstacle (e.g., a chandelier or pillar) is detected within 0.8 m, shift the node toward the largest open convex sub-region.

## 5. Verification Checklist & Acceptance Gates

Before advancing Glome Home Tour from Phase 1 (Alpha) to Phase 2 (Closed Pilot), verify:

- [ ]  Mobile capture app successfully enforces angular velocity limits and alerts the user when panning too fast.
- [ ]  Visual floor-painting HUD accurately maintains persistent state when an operator leaves a room and returns.
- [ ]  Pointing the phone at a featureless surface pauses the recording buffer without causing unrecoverable coordinate loss.
- [ ]  Loop closure is strictly verified at the physical starting point before the capture upload is unlocked.
- [ ]  End-to-end processing pipeline converts an uploaded dataset into an aligned 3DGS model in ≤15 minutes.
- [ ]  2D floor plan extraction correctly identifies enclosed room polygons and computes square meterage within ±5% accuracy of physical laser measurements.
- [ ]  360° equirectangular panoramas render without visible optical seams or perspective distortion.
- [ ]  Generated web viewer and 360° image assets load cleanly on standard mobile and desktop web browsers without specialized plugins.
