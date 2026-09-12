# Bibliography: Projects

This document lists open-source software projects, libraries, frameworks, and developer tools used or evaluated in the development of the Glome Suite across mobile client (`mobile/android/`) and backend infrastructure (`backend/`).

Items prefixed with `**` represent projects that were evaluated, benchmarked, or compared against during development but were not integrated into the active production pipeline.

---

## Active & Integrated Projects

### Google ARCore / ARKit [https://developers.google.com/ar]

**Where, How & Why Used in Glome:**
Used in `mobile/android/` for Visual-Inertial Odometry (VIO) tracking, continuous 6-DoF camera pose estimation, and `SharedCamera` hardware synchronization. ARCore was chosen because native VIO hardware integration provides sub-millisecond motion estimation and hardware exposure timestamps on commodity smartphones without external tracking sensors.

---

### Android Camera2 API / Jetpack Camera Pipeline [https://developer.android.com/training/camerax/architecture]

**Where, How & Why Used in Glome:**
Used in `mobile/android/app/src/main/java/com/glomehometour/arscan/CameraPipeline.kt` to enforce manual camera register overrides (60 FPS sensor readout, locked WB, locked joint shutter/ISO, disabled software OIS/EIS, disabled auto lens correction). Chosen over standard camera wrappers because direct register access is required to preserve static optical intrinsics.

---

### Depth Anything v3 Repository [https://github.com/HKU-Edtech/Depth-Anything-3]

**Where, How & Why Used in Glome:**
Vendored in `backend/third_party/depth_anything_3` and consumed via `backend/reconstruction/depth_priors.py`. Chosen as the primary backend monocular metric depth model provider due to its superior multi-view spatial consistency and native metric depth inference capabilities across unconstrained video frames.

---

### ONNX Runtime Mobile [https://onnxruntime.ai/]

**Where, How & Why Used in Glome:**
Used on Android (`mobile/android/app/src/main/java/com/glomehometour/arscan/MonoDepth.kt`) to execute quantised lightweight monocular depth models (`zipdepth_base_256x256.onnx`) on mobile GPU/NNAPI delegates. Chosen for its optimized cross-platform C++ inference runtime and minimal binary footprint.

---

### PyTorch & PyTorch ROCm [https://pytorch.org/]

**Where, How & Why Used in Glome:**
Used across `backend/` for deep learning model inference, tensor math, and ROCm GPU acceleration on AMD Radeon hardware (RX 9060 XT / RDNA4). Includes custom patches in `depth_priors.py` to prevent ROCm/HIP quantile memory crashes.

---

### OpenCV (Open Source Computer Vision Library) [https://opencv.org/]

**Where, How & Why Used in Glome:**
Used extensively across `backend/ingestion/` (`quality_gate.py`, `sfm_refinement.py`, `pose_aligner.py`) for SIFT feature extraction, image blur evaluation, matrix math, coordinate conversions, and pinhole camera model calculations. Chosen for its industry-standard C++/Python performance and comprehensive vision algorithms.

---

### SciPy & NumPy [https://scipy.org/]

**Where, How & Why Used in Glome:**
Used throughout `backend/` for sparse linear algebra systems (`scipy.sparse.linalg`), KDTree spatial lookups, Umeyama scale-shift metric depth graph fitting, and numerical array processing. Chosen as foundational scientific computing libraries in Python.

---

## Evaluated & Compared Projects (Not Integrated)

### **ZipDepth [https://github.com/zipdepth/zipdepth]

**Where, How & Why Evaluated in Glome:**
Evaluated as a lightweight mobile monocular depth model (`zipdepth_base_256x256.onnx` asset stored in `mobile/android/app/src/main/assets/`). Used for initial mobile depth HUD guidance prototyping; compared against Depth Anything and MiDaS variants.

---

### **MiDaS [https://github.com/isl-org/MiDaS]

**Where, How & Why Evaluated in Glome:**
Evaluated in early mobile (`ml/depth-model/`) and backend tests for zero-shot monocular depth estimation. Benchmarked against Depth Anything models; omitted from backend production in favor of Depth Anything V3 metric depth models.

---

### **Blackmagic Camera App [https://www.blackmagicdesign.com/products/blackmagiccamera]

**Where, How & Why Evaluated in Glome:**
Used as an empirical benchmarking reference tool (`mobile/Gaussian_Splatting_Tips.md`) to analyze low-level smartphone manual camera overrides (fixed shutter $\ge 1/500\text{s}$, minimum ISO, locked white balance, disabled computational stabilization).

---

### **SuperSplat (PlayCanvas) [https://github.com/playcanvas/supersplat]

**Where, How & Why Evaluated in Glome:**
Evaluated as a web-based 3D Gaussian Splatting inspection and manual post-processing cleanup tool (`mobile/Gaussian_Splatting_Tips.md`). Used for inspecting early splat outputs and testing statistical histogram filtering mechanisms.

---

### **MipMap Software & Sharp Frame**

**Where, How & Why Evaluated in Glome:**
Evaluated as third-party photogrammetry frame extraction and focal length seeding utilities (`mobile/Gaussian_Splatting_Tips.md`). Compared against Glome's automated streaming decimation and Camera2 intrinsic logging.
