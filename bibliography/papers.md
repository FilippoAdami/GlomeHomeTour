# Bibliography: Papers

This document lists academic papers that were read, referenced, and used during the research and development of the Glome Suite across both the mobile capture client and backend ingestion/reconstruction systems.

Items prefixed with `**` represent papers that were evaluated, benchmarked, or compared against during development but were not integrated into the active production pipeline.

---

## Active & Integrated Papers

### Depth Anything 3: Recovering the Visual Space from Any Views (2025) [arXiv:2511.10647] - Lihe Yang, et al.

**Description:**
Depth Anything 3 (DA3) introduces a plain transformer-based foundation model for monocular depth estimation that handles multi-view geometry natively. By predicting depth maps and camera rays jointly across multiple input frames, DA3 achieves state-of-the-art spatial consistency and metric depth accuracy across diverse unconstrained video sequences without requiring heavy test-time optimization.

**How and Where Used in Glome:**
DA3 is the primary monocular metric depth estimation backend engine integrated in `backend/reconstruction/depth_priors.py`. It is executed over keyframe sequences using sliding-window chunk streaming (`DA3NESTED-GIANT-LARGE-1.1`). The resulting metric depth predictions serve as the foundational geometric prior for sparse landmark scale-shift graph optimization and multi-view surface normal estimation.

**Link / Source:**
[arXiv:2511.10647](https://arxiv.org/abs/2511.10647)

---

### Depth Anything V2 (2024) [arXiv:2406.09414] - Lihe Yang, et al.

**Description:**
Depth Anything V2 significantly refines monocular depth estimation accuracy and detail by training discriminative vision transformers on large-scale synthetic images paired with teacher-student distillation. It produces smoother depth boundaries and finer surface details while remaining robust against challenging indoor lighting and reflective materials.

**How and Where Used in Glome:**
Integrated in `backend/reconstruction/depth_priors.py` as the automated metric fallback model (`Depth-Anything-V2-Metric-Indoor-Base-hf`). When DA3 is unequipped or hardware constraints dictate lower VRAM usage, Depth Anything V2 generates metric depth predictions for least-squares scale-shift alignment against VIO landmarks.

**Link / Source:**
[arXiv:2406.09414](https://arxiv.org/abs/2406.09414)

---

### Structure-from-Motion Revisited (2016) [IEEE CVPR 2016] - Johannes L. Schönberger, Jan-Michael Frahm

**Description:**
This paper presents COLMAP, a comprehensive review and redesign of Structure-from-Motion (SfM) pipelines. It details robust algorithms for geometric verification, geometric-aware keyframe selection, non-linear bundle adjustment, and scene graph construction that collectively overcome drift and mis-registrations in monocular multi-view datasets.

**How and Where Used in Glome:**
The geometric bundle adjustment principles from this paper govern `backend/ingestion/sfm_refinement.py`. Glome uses SIFT feature detection and FLANN-based matching to seed non-linear pose refinement over VIO trajectory priors, refining camera intrinsics ($f_x, f_y, c_x, c_y$) and solving radial/tangential distortion parameters ($k_1, k_2, p_1, p_2$).

**Link / Source:**
[IEEE Xplore / CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf)

---

### Distinctive Image Features from Scale-Invariant Keypoints (2004) [International Journal of Computer Vision] - David G. Lowe

**Description:**
This seminal paper introduces the Scale-Invariant Feature Transform (SIFT), an algorithm for detecting and describing local scale- and rotation-invariant features in images. SIFT features are highly distinctive and invariant to image scaling, rotation, illumination changes, and 3D viewpoint variations.

**How and Where Used in Glome:**
SIFT is the core feature extraction algorithm used in `backend/ingestion/sfm_refinement.py` (via OpenCV's `cv2.SIFT_create`). It extracts robust keypoints across keyframe pairs to establish geometric correspondences for Essential matrix estimation and VIO trajectory pose refinement.

**Link / Source:**
[IJCV 2004 / Springer](https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94)

---

## Evaluated & Compared Papers (Not Integrated)

### **Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data (2024) [IEEE CVPR 2024 / arXiv:2401.10891] - Lihe Yang, et al.

**Description:**
The original Depth Anything paper presents a foundation model for relative monocular depth estimation trained on 62M unlabeled images using data augmentation and scale-invariant loss functions. It established strong zero-shot depth estimation performance across general unconstrained images.

**How and Where Evaluated in Glome:**
Evaluated during early backend depth prior research. It was compared against DA2 and DA3 metric indoor models; because v1 outputs relative/scale-ambiguous depth maps without metric calibrations, it was superseded by Depth Anything V2/V3 metric models in the backend pipeline.

**Link / Source:**
[arXiv:2401.10891](https://arxiv.org/abs/2401.10891)

---

### **Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer (MiDaS) (2019) [IEEE TPAMI / arXiv:1907.01341] - René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, Vladlen Koltun

**Description:**
MiDaS introduces a scale- and shift-invariant loss function and dataset mixing strategy that enables monocular depth networks to train across disparate, incompatible depth sources. It pioneered generalizable zero-shot monocular depth estimation across diverse environments.

**How and Where Evaluated in Glome:**
Benchmarked on-device for live mobile HUD depth rendering and in early backend tests (`ml/depth-model/`). Compared directly against ZipDepth and Depth Anything; MiDaS-small was evaluated for mobile runtime but discarded in favor of ONNX-optimized ZipDepth on mobile and Depth Anything V3 on the backend.

**Link / Source:**
[arXiv:1907.01341](https://arxiv.org/abs/1907.01341)

---

### **SuperPoint: Self-Supervised Interest Point Detection and Description (2018) [IEEE CVPRW 2018 / arXiv:1712.07629] - Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich

**Description:**
SuperPoint proposes a fully-convolutional neural network architecture that operates on full-size images and computes interest point locations and descriptors in a single forward pass. It uses self-supervised synthetic homography adaptation for training interest point detectors.

**How and Where Evaluated in Glome:**
Compared against SIFT for keyframe feature matching in `backend/ingestion/sfm_refinement.py`. While SuperPoint demonstrated strong matching performance on textureless surfaces, SIFT was retained as the default due to lower CPU dependency footprint and zero GPU model execution requirement during ingestion.

**Link / Source:**
[arXiv:1712.07629](https://arxiv.org/abs/1712.07629)

---

### **Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography (RANSAC) (1981) [Communications of the ACM] - Martin A. Fischler, Robert C. Bolles

**Description:**
RANSAC is an iterative method to estimate parameters of a mathematical model from a set of observed data that contains a high proportion of outliers. It repeatedly selects random subsets of data, fits candidate models, and evaluates inlier consensus.

**How and Where Evaluated in Glome:**
Evaluated for 2D floor plan wall segment extraction from horizontal point slices. The technique is specified for future vectorization stages but is not yet consolidated into the live execution codebase.

**Link / Source:**
[ACM Digital Library](https://dl.acm.org/doi/10.1145/358669.358692)

---

### **Marching Cubes: A High Resolution 3D Surface Construction Algorithm (1987) [ACM SIGGRAPH Computer Graphics] - William E. Lorensen, Harvey E. Cline

**Description:**
Marching Cubes is a 3D surface construction algorithm that extracts a polygonal mesh of an isosurface from a 3D scalar field. It processes voxel cubes sequentially, generating triangle facets based on edge intersection lookup tables.

**How and Where Evaluated in Glome:**
Evaluated in early mobile client variants (`mobile/android/` specifications) for real-time 3D voxel coverage mesh visualization. Replaced on-device by live two-ray closed-form triangulation and OpenGL point sprite landmark rendering (`FeatureParallaxTracker.kt`) to eliminate mobile GPU overhead.

**Link / Source:**
[ACM Digital Library](https://dl.acm.org/doi/10.1145/37401.37422)
