# Bibliography: 2DGS & 3DGS Advanced Optimizations & Material Papers

This document tracks academic papers, mathematical architectures, and optimization techniques for 2D Gaussian Splatting (2DGS), material property learning (PBR/BRDF decomposition), and hardware compute acceleration across Passes 0 to 3 of the Glome Home Tour reconstruction engine.

---

## 1. Core 2DGS & Multi-View Geometry Foundations

### 2D Gaussian Splatting for Geometrically Accurate Radiance Fields (2DGS) (CVPR 2024) [arXiv:2403.17888] - Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, Shenghua Gao

- **Description:**
  Replaces standard 3D volumetric Gaussian ellipsoids with flat, oriented 2D planar Gaussian disks (surfels) embedded in 3D space. Formulates explicit surface normal vectors, 2D tangent frames, and exact ray-splat intersection formulas, eliminating volumetric floater fuzz, multi-view depth ambiguity, and needle-like artifacts.
- **Where & How Used in Glome (Pass 3):**
  Core neural radiance representation in `backend/03_2DGS_training/` executed via native `diff-surfel-rasterization`. The explicit surface normals derived from 2D planar disks directly enable cross-section slicing ($1.0\text{m}-1.5\text{m}$) for RANSAC wall fitting and provide stable surface normals for physically-based material shading.
- **Motive & Rationale:**
  Standard 3DGS produces thick, fuzzy volumetric shells and severe floater artifacts when viewing surfaces from oblique angles or unobserved directions. 2DGS enforces true surface-bound geometry with consistent normal vectors, which is strictly required for CAD-compatible architectural mesh extraction and planar wall fitting.
- **Last Verified / Checked:** 2026-09-15 (Fully integrated in `step_train.py` and `diff-surfel-rasterization`).

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

## 2. Material Learning & Physically-Based Reflective Surfaces

### MaterialRefGS: Endowing 2D Gaussian Splatting with Material Properties for Reflective Surface Rendering (NeurIPS 2025) [arXiv:2510.11387] - MaterialRefGS Authors

- **Description:**
  Built directly on top of 2DGS, MaterialRefGS endows each 2D Gaussian primitive with learnable physical material attributes (metallic, roughness, diffuse albedo) instead of relying solely on spherical harmonics. It rasterizes these properties to screen-space material buffers followed by a BRDF lighting pass.
- **Where & How Used in Glome (Pass 3 / Refinement):**
  Blueprint for material parameterization (`04_2DGS_refinment/pbr_shader.py` and `compressor.py`). Real estate listings feature glossy hardwood floors, marble countertops, and glass windows; separating diffuse albedo from view-dependent specular reflectance prevents baking specular highlights into false 3D geometry.
- **Motive & Rationale:**
  Pure spherical harmonics (SH) bake view-dependent specular highlights into geometric floaters when viewed from unobserved angles. Material decomposition keeps multi-view geometry flat while reproducing specular sheens.
- **Last Verified / Checked:** 2026-09-15 (PBR shader and material structures verified).

---

### GaussianShader: 3D Gaussian Splatting with Shading Functions for Reflective Surfaces (CVPR 2024) [arXiv:2311.17977] - Yingwen Jiang, et al.

- **Description:**
  Incorporates explicit shading functions on top of Gaussians to predict environmental illumination and surface reflectance properties (diffuse/specular/residual decomposition).
- **Where & How Used in Glome:**
  Foundational deferred-shading reference for Glome's PBR pipeline. Demonstrates how residual color decomposition prevents over-fitting reflective specular highlights into spurious Gaussian floaters.
- **Last Verified / Checked:** 2026-09-15.

---

### Ref-Gaussian: Empowering 3D Gaussian Splatting with BRDF and Ray Tracing (ICLR 2025) - Ref-Gaussian Authors

- **Description:**
  Integrates pixel-level BRDF material parameters into Gaussian primitives using a split-sum approximation to avoid expensive Monte Carlo integration, paired with BVH-based ray tracing over extracted surface meshes.
- **Where & How Used in Glome:**
  Provides the split-sum approximation blueprint for real-time deferred shader evaluation in WebGL/WebGPU walkthroughs.
- **Last Verified / Checked:** 2026-09-15.

---

### Relightable 3D Gaussians (R3DG) (2024) [arXiv:2408.12282] - Gao et al.

- **Description:**
  Decomposes scenes into explicit metallic, roughness, base color, and normal components with neural environment lighting fields.
- **Where & How Used in Glome:**
  Reference for synthetic $4096 \times 2048$ 360° panorama generation and virtual staging under variable lighting conditions.
- **Last Verified / Checked:** 2026-09-15.

---

### GS-IR: 3D Gaussian Splatting for Inverse Rendering (CVPR 2024) [arXiv:2311.16473] - Zhihao Liang, et al.

- **Description:**
  Estimates object geometry (normals), unlit albedo, material parameters, and environment maps from multi-view images using physically-based deferred shading.
- **Where & How Used in Glome:**
  Evaluated for normal map regularization and albedo extraction prior to floor plan wall vectorization.
- **Last Verified / Checked:** 2026-09-15.

---

## 3. Compute Efficiency, Hardware Acceleration & Density Control

### Taming 3D Gaussian Splatting for Efficient Rendering (SIGGRAPH Asia 2024)

- **Description:**
  Introduces budget-aware densification control to cap the total number of Gaussians while maintaining rendering fidelity, eliminating unnecessary splat creation.
- **Where & How Used in Glome (Pass 3):**
  Directly guides Glome's dynamic primitive budget in `backend/03_2DGS_training/step_train.py` ($\min(1.5 \times N_{\text{init}}, 4.5\text{M})$) and decoupled densification scheduling (densifying only during early stages, freezing topology during final convergence).
- **Motive & Rationale:**
  Unconstrained densification leads to 5M–10M Gaussians, causing memory exhaust on 16GB GPUs and violating the $\le 25\text{ MB}$ MLS web walkthrough budget.
- **Last Verified / Checked:** 2026-09-15 (Dynamic budget scaling verified).

---

### LiteGS & BalanceGS (2025)

- **Description:**
  Algorithm-system co-design achieving $1.4\times - 3.4\times$ speedups through tiled memory layout, fused sorting, LDS memory caching, and workload-sensitive density control.
- **Where & How Used in Glome (Pass 3):**
  Critical architectural reference for AMD RDNA4 (RX 9060 XT) 32-wide wavefront execution and memory alignment in `diff-surfel-rasterization`.
- **Motive & Rationale:**
  Prevents tile warp divergence and maximizes L2 cache hit rates on AMD RDNA4 architecture.
- **Last Verified / Checked:** 2026-09-15.

---

### LightGaussian: Unbounded 3D Gaussian Compression with Vector Quantization (ECCV 2024) [arXiv:2311.14513] - Fan Ma, et al.

- **Description:**
  Compresses Gaussians via pruning, spherical harmonics degree reduction, and 8-bit vector quantization (VQ), achieving $>15\times$ memory reduction with minimal visual quality degradation.
- **Where & How Used in Glome:**
  Implemented in `backend/04_2DGS_refinment/compressor.py` (`LightGaussianCompressor`) for post-training web walkthrough packaging, compressing models into `walkthrough_2dgs.zip` ($\le 25\text{ MB}$).
- **Motive & Rationale:**
  MLS platforms and mobile web browsers enforce strict file size caps ($\le 25\text{ MB}$). Vector quantization achieves 10x compression over raw PLY files without perceptible rendering artifacts.
- **Last Verified / Checked:** 2026-09-15 (Compression engine verified).

---

## 4. Hardware Optimization & Driver Workarounds (AMD RDNA4 / ROCm 7.x)

1. **Wavefront Alignment (Wave32 SIMD):** AMD RDNA4 (`gfx1200`) executes 32-wide wavefronts natively. Tile binning and radix sorting operators in `diff-surfel-rasterization` leverage 32-thread SIMD execution units to prevent lane masking overhead.
2. **Local Data Share (LDS) Caching:** Front-to-back alpha compositing evaluates surfels in LDS shared memory per compute unit, bounding peak VRAM under $2.0\text{ GB}$.
3. **Elementwise Rotation Operator (`_rotate()`):** Workaround for AMD ROCm 7.1 BLAS GEMM bug on `gfx1200` where `torch.matmul` on $(N, 3) @ (3, 3)$ silently zeroes output rows past index $2^{19} = 524,288$. Replaced across `point_utils.py` with custom elementwise column formulation. (Verified 2026-09-15).
4. **Separable SSIM Grouped Convolutions:** Replaced monolithic 11x11 grouped convolutions in MIOpen with separable 1D horizontal/vertical convolutions, dropping step time by 60% on AMD hardware. (Verified 2026-09-15).
