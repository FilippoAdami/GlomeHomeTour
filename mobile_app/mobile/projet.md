# Functional & Technical Specification: Guided Spatial Video Acquisition System (AR-Scan)

## 1. Executive Summary & System Overview

This specification defines the requirements, architecture, UX/UI guidelines, and acceptance criteria for a cross-platform mobile application (iOS and Android) designed for guided 3D spatial video acquisition.

The client application serves as the capture frontend for a downstream processing pipeline consisting of:

1. Video frame extraction with accurate 6-DoF camera poses.
2. Monocular metric depth estimation using deep spatial priors (e.g., Depth Anything v3).
3. 2D Gaussian Splatting (2DGS) reconstruction.

The app's primary objective is to guide non-expert users in capturing continuous video with sufficient multi-view baseline (motion parallax) while eliminating geometric occlusions and illumination artifacts in complex indoor spaces.

---

## 2. Core Functional Features

### 2.1 6-DoF VIO Pose Tracking & Keyframe Recording

* Continuous estimation of device position $\mathbf{p} = [x, y, z]^T \in \mathbb{R}^3$ and orientation quaternion $\mathbf{q} = [q_w, q_x, q_y, q_z]^T$ via platform-native Visual-Inertial Odometry (ARKit / ARCore).
* Synchronized logging of:
* Raw video frames (1080p, fixed target exposure/WB).
* Timestamps (nanosecond precision).
* Camera intrinsic matrices $\mathbf{K}$:

$$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$


* Extrinsic camera-to-world transformation matrices $\mathbf{T}_{C\to W} \in \mathrm{SE}(3)$.
* Normalized gravity vector and device IMU acceleration/gyroscope stream.



### 2.2 Illumination Filtering & Transient State Rejection

To prevent photometric inconsistencies in the downstream 2DGS radiance field:

* **Real-Time Luma Calculation:** Compute mean luminance $Y_{\text{mean}}$ on the GPU/camera callback:

$$Y = 0.2126\,R + 0.7152\,G + 0.0722\,B, \quad Y_{\text{mean}} = \frac{1}{N} \sum_{i=1}^{N} Y_i$$



normalized to the range $[0, 255]$.
* **Under-Exposure Cutoff:** Discard frames where $Y_{\text{mean}} < 40$ (underexposed/dark).
* **Transient Auto-Exposure/Light Switch Rejection:**
* Calculate delta between consecutive frames: $\Delta Y = \vert{}Y_{\text{mean}}(t) - Y_{\text{mean}}(t-1)\vert{}$.
* If $\Delta Y > 30$ (indicating a light switch event or rapid exposure ramp), flag a stabilization window.
* Drop all intermediate frames during the transition until variance $\sigma^2(Y_{\text{mean}}) < 5.0$ over a sliding window of 10 frames (~330 ms).


* **Dataset Pruning:** Dropped frames are omitted from both the exported `.mp4`/image sequences and `transforms.json`. VIO tracking continues running in the background to preserve global trajectory continuity.

### 2.3 Lightweight Volumetric Coverage & Occlusion Mapping

* **Sparse Voxel Grid:** Spatial volume represented via a coarse spatial hash map with voxel resolution $s_v = 10\text{ cm}$.
* **Ray Casting & Surface Integration:** For detected feature points and planar estimates, cast viewing rays from camera position $\mathbf{c}$.
* Voxels along the ray before the hit point are updated as *Free Space*.
* Voxels at the hit point are updated as *Occupied Surface*.
* Voxels behind the hit point along the ray are treated as *Occluded / Unknown*.


* **Multi-View Parallax Validation:** A surface voxel $\mathbf{v}$ is marked as **Adequately Captured** only when observed from at least two poses $\mathbf{c}_1, \mathbf{c}_2$ such that the baseline angle $\theta_{\text{parallax}} \ge 25^\circ$:

$$\theta_{\text{parallax}} = \arccos\left( \frac{(\mathbf{v} - \mathbf{c}_1) \cdot (\mathbf{v} - \mathbf{c}_2)}{\Vert{}\mathbf{v} - \mathbf{c}_1\Vert{}_2 \Vert{}\mathbf{v} - \mathbf{c}_2\Vert{}_2} \right) \ge 25^\circ$$



### 2.4 Real-Time Guidance & Feedback Engine

* **Coverage Mesh Shader:** Visual surface representation colored dynamically based on coverage state:
* *Unobserved / Low Parallax:* Translucent Amber / Red.
* *Adequately Captured:* Translucent Green (fading to invisible once fully verified to prevent visual clutter).


* **Occlusion Vector Guide:** Computes the centroid $\mathbf{p}_{\text{target}}$ of unobserved voxels that have adjacent observed voxels (e.g., rear faces of tables/islands) and projects a dynamic 3D directional arrow onto the viewport guiding the user to traverse around the occluder.
* **Motion & Lighting Guard:** Monitors linear velocity ($v_{\text{lin}}$), angular velocity ($\omega_{\text{ang}}$), and ambient light. Triggers real-time alerts if:
* $v_{\text{lin}} > 0.4\text{ m/s}$ (Motion blur prevention).
* $\omega_{\text{ang}} > 30^\circ/\text{s}$ (Rotational blur prevention).
* $Y_{\text{mean}} < 40$ ("Too Dark: Turn on lights").



---

## 3. Hardware, Platform, & Minimum System Requirements

### Hardware Targets

* **No LiDAR Required:** Operates purely on standard RGB monocular cameras + IMU.
* **Legacy Support Baseline:** Devices released up to 4 years prior (e.g., iPhone 11 / A13 Bionic; Android devices with Snapdragon 778G / Exynos 990 or newer).

### Platform & Tooling Matrix

| Component | iOS Specification | Android Specification |
| --- | --- | --- |
| **OS Version** | iOS 16.0+ | Android 11.0+ (API Level 30+) |
| **Tracking Framework** | ARKit 6.0+ | ARCore 1.35+ |
| **Graphics Engine** | Metal / SceneKit / RealityKit | OpenGL ES 3.2 / Filament / Vulkan |
| **Language & Architecture** | Swift 5.9+ (Native Core) | Kotlin 1.9+ (Native Core) |
| **Shared Logic Layer** | Kotlin Multiplatform (KMP) for math, serialization, data contracts |  |

---

## 4. Architecture & Data Flow

```
[Camera Sensor (RGB)] + [IMU (VIO)]
          │
          ├──> [Platform Native Engine (ARKit / ARCore)]
          │          │
          │          ├──> 60 Hz Render Loop: Viewport Overlay & Motion Check
          │          │
          │          └──> 10-15 Hz Voxel Thread (Background Worker):
          │                     ├── Ray Casting & Occlusion Updates
          │                     ├── Parallax Verification Engine
          │                     └── Occlusion Centroid / Vector Calculation
          │
          └──> [Illumination & Quality Filter]
                     │
                     ├── Y_mean < 40 OR Transient State ──> [Drop Frame from Export]
                     │                                      (Maintain VIO State)
                     │
                     └── Valid Frame ──> [Frame Writer Pipeline]
                                              ├── 1080p Video / Frames (.jpg/.png)
                                              └── transforms.json (NeRF/2DGS format)

```

---

## 5. UX Analysis

### User Personas & Edge Cases

* **Persona:** Non-technical home users, realtors, or field technicians.
* **Edge Cases Addressed:**
* *Dark Room Entry:* User enters a dark room and turns on the light midway. The app suppresses dark and transiently over/underexposed frames automatically without requiring a scan restart.
* *Blind Spots:* User circles around free-standing objects prompted by clear 3D vectors rather than guessing blind spots.
* *Fast Movement:* Real-time feedback alerts the user before motion blur degrades reconstruction quality.



---

## 6. UI Specification

```
+-------------------------------------------------------------+
| [X] Exit          Coverage: [||||||||||....] 68%    [Flash] |
|                                                             |
|                                                             |
|                       ^                                     |
|                      / \   (3D Floating Guidance Arrow:     |
|                       |     "Walk around the kitchen island")|
|                                                             |
|               [ Amber Mesh Overlay:                         |
|                 Unscanned island rear ]                     |
|                                                             |
|                                                             |
|      (!) ROOM TOO DARK / STABILIZING LIGHT                  |
|      Please turn on the room light                          |
|                                                             |
|                         [ (O) Finish Scan ]                 |
+-------------------------------------------------------------+

```

### UI Indicators

* **Lighting State Toast:** Displays "Adjusting to lighting change..." during illumination transitions.
* **Global Coverage Progress Bar:** Increments only when valid, properly illuminated, and parallax-verified surfaces are scanned.
* **Finish Button:** Enabled once global coverage meets threshold ($P \ge 85\%$).

---

## 7. Quality Assurance & Pass/Fail Criteria

### 7.1 Automated Benchmarks

* **Tracking Robustness:** $\le 1$ tracking loss event per 3-minute indoor scan ($\ge 150\text{ lux}$).
* **Memory Footprint:** Peak RAM $\le 350\text{ MB}$ over 5 minutes of continuous recording.
* **Frame Drop Rate:** Sustained $\ge 58\text{ fps}$ (60 fps target) or $\ge 29\text{ fps}$ (30 fps target).
* **Photometric Consistency:** Zero frames in final dataset with $Y_{\text{mean}} < 40$ or $\Delta Y > 30$.

### 7.2 Empirical Test Scenarios

```
+---------------------------------------------------------------------------------------------------+
| Scenario ID | Test Case Setup                  | Expected Behavior               | Pass Criteria  |
|:------------|:---------------------------------|:--------------------------------|:---------------|
| **TC-01**   | Free-standing table / island     | App detects rear occlusion;     | Mesh turns     |
|             | scanned only from front.         | guides user to circle object.   | green 360°; no  |
|             |                                  |                                 | rear holes.     |
|-------------+----------------------------------+---------------------------------+----------------|
| **TC-02**   | High-speed panning               | Warning triggers immediately;   | Blurry frames  |
|             | ($\omega > 45^\circ/\text{s}$).  | clears upon stabilization.      | omitted/marked.|
|-------------+----------------------------------+---------------------------------+----------------|
| **TC-03**   | Light switch toggled during scan | Transient frames rejected;      | No black/blown |
|             | (dark to bright room).           | VIO trajectory remains unbroken.| frames in export|
+---------------------------------------------------------------------------------------------------+

```

---

## 8. Export Package Format

```
scan_dataset_<timestamp>/
├── images/
│   ├── frame_00000.jpg
│   ├── frame_00001.jpg
│   └── ...
├── transforms.json          # Filtered frames + valid camera poses
├── trajectory.csv           # Full unbroken VIO pose log
└── coverage_summary.json    # Metric volume coverage stats and dropped frame metrics

```