# 00_ingestion — Stage 1: Capture Ingestion, Quality Gate & Pose Alignment

`00_ingestion` is the entry point of the GlomeHomeTour backend pipeline. It consumes raw mobile captures (Camera2 RGB video frames + 60 Hz ARCore VIO trajectory and camera metadata) and produces a clean, synchronized, motion-filtered visual chain ready for Structure-from-Motion (COLMAP) and depth estimation.

---

## What This Stage Does

1. **Package Loading & Validation:** Unpacks `.zip` capture archives or directory structures into structured `CapturePackage` representations, validating Camera2 metadata, optical intrinsics, and VIO timestamps against `shared/schemas/`.
2. **Quality Gating (Motion Blur Rejection):** Assesses frame clarity in real time and discards images degraded by fast motion, camera shake, or poor focus.
3. **Sub-Millisecond Pose Synchronization:** Aligns the asynchronous 60 Hz ARCore VIO trajectory with Camera2 frame mid-exposure timestamps using SLERP and spline interpolation.
4. **Dynamic Keyframe Selection:** Decimates redundant stationary frames while preserving necessary multi-view baseline and parallax using depth-adaptive strides, Lowe's ratio SIFT matching, and epipolar RANSAC.
5. **Portrait Optics Transpose:** Automatically transposes horizontal sensor frames 90° clockwise into standard vertical portrait format ($1080 \times 1920$) and adjusts optical focal lengths and principal points accordingly.

---

## Modules & Architecture

| Module | Role |
| --- | --- |
| [`step_filter_quality.py`](step_filter_quality.py) | **Pipeline Step 1 Entry Point**: Orchestrates quality filtering, pose alignment, and keyframe selection into the target workspace. |
| [`package_loader.py`](package_loader.py) | Parses and validates capture archives; defines `CapturePackage`, `Keyframe`, `CameraIntrinsics`, and `TrajectorySample`. |
| [`quality_gate.py`](quality_gate.py) | Implements motion blur rejection (`QualityGate`) via variance of Laplacian and luminance edge statistics. |
| [`pose_aligner.py`](pose_aligner.py) | Performs sub-millisecond continuous pose interpolation (`PoseAligner`) via Quaternion SLERP and cubic position splines. |
| [`keyframe_selector.py`](keyframe_selector.py) | Downsamples frames using depth-adaptive spatial baseline and optical angle gating (`KeyframeSelector`). |
| [`inspect_scan.py`](inspect_scan.py) | Diagnostic utility to inspect video properties, track health, and timestamp alignment of raw mobile packages. |

---

## Techniques & Mathematical Specifications

### 1. Motion Blur Gating
To prevent blurry frames from destabilizing feature matching and 2DGS training:
$$\text{Blur Metric} = \operatorname{Var}\big(\nabla^2 I\big)$$
Frames falling below the calibrated sharpness threshold $\tau_{\text{blur}}$ (default: $80.0$) are rejected immediately.

### 2. Quaternion SLERP & Spline Synchronization
Camera exposure timestamps $t_{\text{mid}} = t_{\text{start}} + \frac{1}{2} t_{\text{exposure}}$ rarely coincide exactly with 60 Hz VIO samples $(t_k, t_{k+1})$. The aligner synchronizes them via:
- **Rotations:** Spherical Linear Interpolation (SLERP):
  $$q(t) = \operatorname{SLERP}\left(q_k, q_{k+1}; \frac{t - t_k}{t_{k+1} - t_k}\right)$$
- **Translations:** Cubic spline interpolation across the local temporal window, ensuring smooth velocity and acceleration continuity.

### 3. Depth-Adaptive Dynamic Keyframe Selection
Stationary pauses generate duplicate views that bias 2DGS optimization, while fast sweeps create covisibility gaps. The selector evaluates:
- **Translational Baseline:** Minimum baseline $\Delta x \ge \tau_{\text{dist}}(d_{\text{scene}})$ scaled by the scene depth.
- **Rotational Baseline:** Minimum angular sweep $\Delta \theta \ge \tau_{\text{rot}}$ (typically $5^\circ - 8^\circ$).
- **Visual Covisibility Verification:** SIFT feature extraction with Lowe's ratio test ($\le 0.75$) and epipolar RANSAC to guarantee sufficient 2-view overlap before accepting a frame.

### 4. Coordinate System & Sensor Transposition
Mobile captures are taken in upright vertical portrait orientation, but sensor hardware outputs in landscape:
- **Image Transpose:** Rotated $90^\circ$ clockwise ($W \times H \to H \times W$).
- **Intrinsics Transformation:**
  $$f_x^{\prime} = f_y, \quad f_y^{\prime} = f_x, \quad c_x^{\prime} = H - c_y, \quad c_y^{\prime} = c_x$$
- **Camera Convention:** Output camera poses in `transforms.json` use the **OpenGL / ARCore coordinate convention** ($+X$ right, $+Y$ up, $-Z$ forward).

---

## Inputs & Outputs

- **Input:**
  - Raw capture directory or `.zip` containing `video.mp4`, `trajectory.csv`, and `camera_metadata.json`.
- **Output:**
  - `<workspace>/images/*.jpg`: Sharp, upright vertical portrait keyframes ($1080 \times 1920$).
  - `<workspace>/transforms.json`: Synchronized camera intrinsics and $4 \times 4$ camera-to-world transformation matrices.

## Usage

```bash
python 00_ingestion/step_filter_quality.py \
    --input-package /path/to/capture.zip \
    --workspace backend/current_scene \
    [--force]
```
