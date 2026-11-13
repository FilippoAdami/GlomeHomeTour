# Glome AR-Scan (Mobile Application)

The **Glome AR-Scan** Android application is the high-precision spatial capture client for the Glome 3D Gaussian Walkthrough Engine. It transforms commodity smartphones into metric photogrammetry and radiance field capture devices without requiring LiDAR hardware or external 360° cameras.

---

## 1. System Overview & Core Capabilities

* **Target Device Baseline:** Xiaomi Redmi Note 10S (`rosemary`, MediaTek Helio G95, Android 13 / API 33, ARCore 1.55).
* **Sensor Architecture:** Guided video capture on the Ultra-Wide ($0.5\times$) camera sensor paired with 6-DoF Visual-Inertial Odometry (VIO). Autofocus waits for a deliberate tap on the camera preview; it does not continuously refocus during a scan.
* **Hardware-Level Camera2 Overrides:** Manual exposure metering with locked shutter and ISO, locked white balance, disabled computational stabilization (EIS), and disabled digital lens distortion correction to preserve rigid perspective geometry.
* **Keyframe Decimation:** Motion and photometric gating writes keyframes only when camera moves $\ge 8\text{ cm}$ or rotates $\ge 6^\circ$ (capped at 5 Hz), preserving unbroken 60 Hz VIO pose streams in `trajectory.csv`.
* **Standardized Data Contracts (`shared/schemas/`):**
  * `transforms.json`: OpenCV pinhole camera model, intrinsic matrix $\mathbf{K}$, distortion parameters, and per-keyframe $4\times 4$ camera-to-world extrinsics.
  * `trajectory.csv`: Continuous 60 Hz VIO 6-DoF trajectory stream ($t_x, t_y, t_z, q_x, q_y, q_z, q_w$, tracking state).
  * `coverage_summary.json`: Quality diagnostics, voxel statistics, tracking loss events, and capture metrics.
  * `pointcloud.ply`: Binary little-endian XYZ point cloud; camera poses are in `trajectory.csv`. Finalization clips points to 6 m from the sampled camera path and applies statistical outlier removal (20 neighbors, 2σ threshold); clouds with 20 or fewer points skip SOR. Previews and Inspect 3D use the same filter.

---

## 2. Card-Based UI/UX Overhaul

The mobile application features a modern, clean, card-based interface designed for property and multi-room management:

### Property Management (`HomeActivity`)
* **Card-Based Property Listing:** Each property displays its address/title, room count, creation date, and status badges (e.g. `PROCESSED`).
* **Direct Actions:**
  * `Listing`: Opens property details and room breakdown.
  * `+ Add Room`: Launches capture flow directly into the property dataset.
  * `Delete`: Removes property records with real-time progress feedback.
* **In-session room changes:** `Next room` saves the finished room as its own session under the property, then lets the operator rename the upcoming room before capture starts.

### Scan & Room Gallery (`PropertyDetailActivity` & `ScanGalleryActivity`)
* **Room Scan Cards:** Individual cards for each room scan displaying interactive dual-preview thumbnails, scan date, keyframe counts, coverage metrics, and status badges.
* **Dual-Preview Thumbnails & Horizontal Swipe Gesture (`ScanPreviewLoader` & `PointCloudPreviewGenerator`):**
  * **Default First Frame Photo:** Every scan card defaults to displaying the first keyframe photo of the scan, annotated with a `📷 Photo` indicator pill badge.
  * **Resulting 3D Point Cloud Snapshot:** Automatically renders an isometric 3D projection of the room's binary PLY point cloud, colored with an elevation Turbo colormap and camera trajectory breadcrumbs (with a fallback `☁️ 3D Pending` placeholder if not yet available).
  * **Interactive Flip Gesture:**
    * Swiping horizontally (left or right) on the thumbnail flips smoothly between the First Frame Photo (`📷 Photo`) and the 3D Point Cloud Preview (`☁️ 3D`) with directional sliding animations (`slide_in_right`/`slide_out_left` and `slide_in_left`/`slide_out_right`).
    * Tapping on the thumbnail also flips the view.
    * Touch disambiguation dynamically locks horizontal touch events (`requestDisallowInterceptTouchEvent(true)`) while preserving fluid vertical scrolling throughout the room list.
  * **Two-Tier Performance Caching:** Utilizes in-memory `LruCache` alongside persistent disk caching (`context.cacheDir`) for instantaneous, stutter-free scrolling.
  * **Multi-Source Archive Resolution:** Seamlessly resolves frames and point clouds across MediaStore zip archives, loose flat session files, app-private fallback directories, and unified property scan packages (`<address>_scan.zip`).
* **Per-Card Actions:**
  * **`Inspect 3D`:** Instantly launches the standalone 3D point cloud viewer to inspect room geometry and trajectory quality.
  * **`Add Geometry`:** Overlays the final keyframe from the previous scan as a semi-transparent viewfinder guide, allowing the user to re-align with the room and integrate missing areas before processing.
  * **`Process`:** Initiates backend 3DGS and mesh reconstruction pipeline.
  * **`Delete`:** Removes scan data with progress tracking.

### Non-Reversible Pre-Processing Warning Banner (`dialog_process_confirmation.xml`)
To prevent accidental processing of incomplete or low-quality scans:
* When tapping **`Process`**, a confirmation modal appears warning the user:
  > *"Low-quality scans produce low-quality 3D tours. Processing is permanent and cannot be undone. Please ensure full coverage before proceeding."*
* **Direct Modal Actions:**
  * **`Inspect 3D`:** Jumps straight to the 3D Point Cloud Viewer to verify wall coverage and structural completeness.
  * **`Proceed / Upload`:** Confirms and initiates backend reconstruction.
  * **`Cancel`:** Aborts processing to allow taking additional geometry.

---

## 3. Standalone 3D Point Cloud Inspection Viewer

Located in `PointCloudViewerActivity` and rendered via `PointCloudRenderer`, this component provides a dedicated, high-performance OpenGL ES 2.0 inspection environment:

### Zero-Overhead Mobile Rendering Engine
* **Standalone Dark Canvas:** Opaque dark background (`#0B0E14`), decoupled from the AR scene, allowing users to inspect scans anywhere, anytime.
* **VBO GPU Caching:** Points and trajectories are uploaded directly to GPU Vertex Buffer Objects (`GL_STATIC_DRAW`).
* **Zero-Allocation Loop:** Pre-allocated FloatArrays and vectors eliminate garbage collection pauses during rendering.
* **On-Demand Rendering (`RENDERMODE_WHEN_DIRTY`):** Renders only on user touch in Orbit and Top-Down modes to conserve battery and avoid thermal throttling.

### Three Interactive Navigation Modes
1. **🔄 3D Orbit Mode:**
   * 1-finger drag: rotates camera around the room's geometric center.
   * 2-finger pinch: smooth, continuous scale zooming.
   * 2-finger drag: camera panning along the viewing plane.
   * `⌖ Center` button: instantly recenters camera to room midpoint.
2. **📐 Top-Down (Floorplan) Mode:**
  * Bird's-eye orthographic projection ($pitch = 89.9^\circ$).
   * Allows evaluating room perimeter, wall boundaries, furniture placement, and coverage holes.
   * Supports 2-finger pinch zoom and horizontal rotation.
3. **📱 Immersive Gyro Walk Mode:**
   * **1:1 Physical Look-Around:** Tilting the phone UP looks at the ceiling, tilting DOWN looks at the floor, and turning left/right rotates the room with 1:1 real-world physical correspondence.
   * **Magnetic Distortion Immunity (`Sensor.TYPE_GAME_ROTATION_VECTOR`):** Prioritizes gyroscope + accelerometer fusion over the magnetometer, completely eliminating erratic compass snaps caused by indoor electrical wiring and steel rebar.
   * **Mathematical Sensor Mapping:** Corrects Android's row-major $4\times 4$ sensor matrix layout to ARCore's $(+Y\text{ up}, -Z\text{ forward})$ coordinates:
     $$PC_X = -R[2], \quad PC_Y = -R[10], \quad PC_Z = R[6], \quad \vec{V}_{up, PC} = (R[1], R[9], -R[5])$$
   * **Touch-Drag Walking:**
     * 1-finger vertical swipe ($\Delta y$): walks forward/backward along current horizontal look direction (or top-of-phone direction when looking steep up/down).
     * 1-finger horizontal swipe ($\Delta x$): adjusts yaw heading offset (`gyroYawOffset`), enabling $360^\circ$ navigation while seated.
     * 2-finger drag: adjusts eye height ($walkY$) and horizontal strafe.
   * **Safety Clamping:** Walking translation is clamped within room bounding box margins ($[minX-0.5, maxX+0.5]$, $[minZ-0.5, maxZ+0.5]$), preventing accidental launches into empty space.
   * `⌖ Center` button: resets camera position back to room center at human eye height ($1.55\text{ m}$).

### Visual Display Controls
* **Solid Surface Point Cloud Shading:**
  * Uses a solid core disc fragment shader (`smoothstep(0.25, 0.04, distSq)`) with distance-adaptive sizing (`clamp(u_PointScale / dist, 6.0, 96.0)`).
  * Points merge into continuous, recognizable geometric surfaces rather than appearing as scattered dust.
  * **Point Size Presets (`⚪ Thick` button):** Tap to cycle between `⚪ Thick` ($72.0\text{f}$, default), `⚪ Max` ($105.0\text{f}$), and `⚪ Fine` ($42.0\text{f}$).
* **Subtle Trajectory Track (`🛤 Track` button):**
  * Rendered as an elegant, semi-transparent warm amber thread (`vec4(0.95, 0.78, 0.28, 0.38)`, $2.0\text{ px}$ width).
  * Tap the `🛤 Track` button on the top bar to toggle trajectory visibility ON or OFF dynamically.
* **Height-Based Turbo Color Ramp:**
  * Automatically maps point elevation ($Y$-axis) across a 5-stop color gradient:
    * Floor: Deep Navy (`#1433A6`) $\rightarrow$ Cyan (`#00BFE6`)
    * Eye Level: Emerald Green (`#1FD966`) $\rightarrow$ Warm Yellow (`#F2D933`)
    * Ceiling: Coral Red (`#F25940`)
* **Outlier clipping:** The viewer and card preview discard points beyond the padded camera-path bounds; this keeps distant reconstruction spikes from setting the room scale, colors, and gyro-walk limits.
  * The path bounds extend 4 m horizontally and 3 m vertically; this can omit unusually distant geometry in very large rooms.

---

## 4. Point Cloud Accumulation & Storage Pipeline

* **`ScanPointCloudAccumulator.kt`:**
  * Utilizes a spatial hash voxel grid ($3.5\text{ cm}$) to deduplicate points in $\mathcal{O}(1)$ time without per-frame heap allocations.
  * Caps maximum point memory at 60,000 points to preserve GPU vertex throughput.
  * Records camera positions every $\ge 8\text{ cm}$ and rejects pose jumps over 0.5 m.
* **Flexible Binary PLY Format:**
  * Reads and writes binary little-endian PLY files (`format binary_little_endian 1.0`).
  * Supports both 12-byte float XYZ and 16-byte float XYZ + uchar RGBA layouts.
  * Fully interoperable with backend COLMAP reconstructions via `backend/Utilities/convert_colmap_to_mobile_pointcloud.py`.

---

## 5. Building, Testing & Deployment

### Prerequisites
* Android Studio Iguana / Jellyfish or Gradle command-line tools.
* Android SDK Platform 34, Build Tools 34.0.0.
* JDK 17.
* ADB with USB debugging enabled on a connected physical device.

### Gradle Commands
```bash
# Navigate to the Android project root
cd mobile_app/mobile/android

# Run unit tests (including accumulator and voxel deduplication tests)
./gradlew testDebugUnitTest

# Assemble debug APK
./gradlew assembleDebug

# Build and install directly onto connected device
./gradlew installDebug
```

### Launching via ADB (Testing)
```bash
# Launch main property dashboard
adb shell am start -n com.glomehometour.arscan/.HomeActivity

# Launch standalone 3D point cloud inspection directly
adb shell am start -n com.glomehometour.arscan/.PointCloudViewerActivity \
  --es session_name "scan_20261024_164826" \
  --es scan_folder "scan_0" \
  --es room_name "Bedroom"
```
