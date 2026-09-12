# Glome Home Tour — Backend Reconstruction & Ingestion Engine

**Glome Home Tour Backend** is an automated high-precision spatial compute pipeline that ingests smartphone sensor captures (Camera2 RGB video + 60 Hz ARCore VIO trajectory) and reconstructs MLS-compliant digital listing assets:
1. **Interactive 2D/3D Gaussian Splatting (2DGS) Web Walkthrough** ($\le 25\text{ MB}$ compressed).
2. **Automated Vector Floor Plan** with metric dimensions and room square meterage.
3. **Synthetic 4K 360° Equirectangular Panoramas** synthesized from semantic centroids.

---

## 1. Hardware Targets & Execution Environment

The backend is engineered for high-throughput local workstation compute:
- **Primary GPU Target:** AMD Radeon RX 9060 XT (16 GB VRAM, RDNA4 architecture, `gfx1200` ISA).
- **Driver & Framework:** AMD ROCm 7.1, PyTorch `2.13.0+rocm7.1`, Linux kernel 6.x.
- **Display Server Interoperability:** Because the target hardware simultaneously drives the host display server (X11/Wayland compositor), the engine enforces non-blocking compositor yields (`time.sleep(0.18)`) and isolated cache memory namespaces (`MIOPEN_USER_DB_PATH=/tmp/miopen`).

---

## 2. Ingestion & Dynamic Keyframe Filtering Engine

Raw video captures typically contain redundant stationary pauses, fast camera sweeps, and high-frequency motion blur. The ingestion engine filters the incoming frame stream to extract an optimal, continuous visual chain.

```
Raw Mobile Capture (.zip)
    │
    ▼
[ Quality Gate ] ────────► Rejects motion blur: Var(∇² I) < τ_blur
    │
    ▼
[ Pose Aligner ] ────────► Sub-ms Quaternion SLERP + Spline sync to 60 Hz VIO
    │
    ▼
[ Dynamic Selector ] ────► Depth-Adaptive Stride & Rotation Gating
    │                      + SIFT Lowe Ratio (≤ 0.75) + Epipolar RANSAC
    ▼
[ Upright Transform ] ───► Transposes 90° CW to Vertical Portrait (1080 × 1920)
    │
    ▼
[ Depth Anything 3 ] ────► Multi-View Metric Depth Estimation (ViT-Giant, BFloat16 AMP)
    │
    ▼
[ Surfel Cloud Init ] ───► 2D Gaussian Surfel Initialization (Positions, Normals, Scales)
```

---

## 3. Mathematical Specifications

### 3.1 Smartphone Vertical Portrait Optics (Anisotropic Field of View)
When a smartphone is held in portrait orientation, the sensor aspect ratio ($1080 \times 1920$) produces an anisotropic field of view:
$$\text{HFOV} = 2 \arctan\left(\frac{W}{2 f_x}\right) \approx 40.8^\circ$$
$$\text{VFOV} = 2 \arctan\left(\frac{H}{2 f_y}\right) \approx 67.0^\circ$$

Because the horizontal field of view is narrow ($\text{HFOV} = 40.8^\circ$), horizontal camera panning ($\Delta\theta_{\text{yaw}}$) displaces visual features out of the frame **$\sim 1.64\times$ faster** than vertical tilting ($\Delta\theta_{\text{pitch}}$). 

To normalize rotational displacement across both axes, the engine computes the **Anisotropic Effective Rotation** $\Delta\theta_{\text{norm}}$:
$$\Delta\theta_{\text{norm}} = \sqrt{ \left(\frac{\Delta\theta_{\text{yaw}}}{\text{HFOV}}\right)^2 + \left(\frac{\Delta\theta_{\text{pitch}}}{\text{VFOV}}\right)^2 + \left(\frac{\Delta\theta_{\text{roll}}}{\text{HFOV}}\right)^2 } \times \text{HFOV}$$

---

### 3.2 Depth-Adaptive Scaling Laws
Fixed translation and rotation thresholds create severe failure modes in real scenes: small steps across large rooms produce redundant images, while standard steps near close objects cause severe parallax tearing.

#### The Parallax Disparity Principle
The pixel displacement $\Delta p$ of a 3D physical point under camera baseline translation $t_x$ is inversely proportional to metric scene depth $Z$:
$$\Delta p \approx \frac{f \cdot t_x}{Z}$$

Furthermore, the physical field-of-view span at depth $Z$ expands linearly:
$$W_{3D}(Z) = 2 \cdot Z \cdot \tan\left(\frac{\text{HFOV}}{2}\right)$$

#### The Dynamic Scaling Equations
Let $Z$ be the median metric scene depth of the current keyframe, and $Z_0 = 2.0\text{m}$ be the indoor reference depth. The allowable step parameters scale dynamically:

1. **Dynamic Maximum Translation Stride $d_{\max}(Z)$:**
   $$d_{\max}(Z) = \text{clamp}\left( d_0 \cdot \left(\frac{Z}{Z_0}\right)^{0.75}, \; d_{\min}, \; d_{\text{ceiling}} \right)$$
   - *Default parameters:* $d_0 = 1.20\text{m}$, $d_{\min} = 0.65\text{m}$, $d_{\text{ceiling}} = 1.65\text{m}$.
   - *Behavior:* In near-field areas ($Z \le 1.0\text{m}$), stride compresses to $0.65\text{m}$ to protect fine geometry; in open areas ($Z \ge 3.5\text{m}$), stride expands to $1.40\text{m}-1.65\text{m}$, eliminating redundant walking steps.

2. **Dynamic Maximum Rotation Cap $\theta_{\max}(Z)$:**
   $$\theta_{\max}(Z) = \text{clamp}\left( \theta_0 \cdot \left(\frac{Z}{Z_0}\right)^{0.50}, \; \theta_{\min}, \; \theta_{\text{absolute\_max}} \right)$$
   - *Default parameters:* $\theta_0 = 24.0^\circ$, $\theta_{\min} = 16.0^\circ$, $\theta_{\text{absolute\_max}} = 33.0^\circ$ (always strictly $< 35.0^\circ$).
   - *Behavior:* Near walls and desks, rotations are restricted to $16.0^\circ$ to prevent close-up feature shearing. In open spaces, wide panoramic sweeps up to $33.0^\circ$ are accepted.

---

### 3.3 Epipolar Geometric Verification & Lowe's Ratio Test
To prevent false-positive visual overlap on repetitive indoor textures (radiator fins, wood grain, blinds), every candidate keyframe must pass two-tier geometric verification:

1. **Lowe's Second-Nearest-Neighbor Ratio Test:**
   $$\frac{\|\mathbf{d}_i - \mathbf{d}_{j, 1}\|_2}{\|\mathbf{d}_i - \mathbf{d}_{j, 2}\|_2} < 0.75$$
   Eliminates ambiguous descriptor matches in high-dimensional SIFT space.

2. **Fundamental Matrix Epipolar Constraint:**
   $$\mathbf{x}_2^T \mathbf{F} \mathbf{x}_1 = 0$$
   Computed via RANSAC (`cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0)`). Candidate frames are rejected if verified inliers drop below $\tau_{\text{inliers}} = 8$, preventing disconnected scene islands.

---

## 4. Depth Prior Architecture: Depth Anything 3

The reconstruction engine utilizes **Depth Anything 3 Giant (`DA3NESTED-GIANT-LARGE-1.1`)**:
- **Backbone:** Hierarchical Vision Transformer (ViT-Giant) with SwiGLU feed-forward networks.
- **Precision:** Automatic Mixed Precision (BFloat16 AMP) on ROCm with native FP32 accumulator retention.
- **Metric Scale Alignment:** Ingests metric camera intrinsics and extrinsics via `align_to_input_ext_scale=True`, enforcing physical metric consistency across sequential frames.

---

---

## 5. Pipeline Stages & Directory Layout

The reconstruction pipeline is organized into distinct decoupled stages under `backend/scenes/<scene_name>/`:

```
backend/scenes/<scene_name>/
├── GS_input/                           # Stage 0: Keyframes, transforms.json, depth maps & initial surfels
├── 2DGS_results/                       # Stage 1: Trained 2DGS radiance field & standard 3DGS PLY
│   ├── material_2dgs_checkpoint.pt     # - Full FP32 PyTorch checkpoint (positions, quaternions, scales, PBR)
│   ├── walkthrough_2dgs.zip            # - MLS-compliant WebGL bundle (<= 25 MB)
│   ├── <scene>_standard_3dgs.ply       # - Canonical 3DGS/2DGS PLY (SuperSplat, Antimatter15, Blender)
│   ├── training_summary.json           # - Convergence & loss statistics
│   └── stages/                         # - Intermediate checkpoints & PLY snapshots every 500 iters
├── mesh_results/                       # Stage 2: SuGaR-like watertight CAD/BIM mesh & PBR textures
│   ├── scene_architecture.glb          # - Watertight room shell + completed furniture instances
│   ├── scene_cad.dxf / scene.ifc       # - Layered CAD / BIM exchange models
│   ├── textures/*.png                  # - 2048x2048 PBR texture maps (Albedo, Normal, Roughness, Metal)
│   └── mesh_manifest.json              # - Scene graph manifest (validated by shared/schemas/)
└── floorplan_results/                  # Stage 3: Dimensioned 2D floor plan JSON & SVG
```

---

## 6. Execution & Verification

### Running 2DGS Model Training
To train the progressive multi-scale 2DGS radiance field on AMD ROCm:
```bash
PYTHONPATH=backend backend/.venv/bin/python3 backend/reconstruction/training/run_scene_training.py \
    --scene-dir backend/scenes/<scene_name>/GS_input \
    --output-dir backend/scenes/<scene_name>_2DGS_results \
    --iterations 3000 \
    --voxel-size 0.035 \
    --max-surfels 400000 \
    --checkpoint-interval 500 \
    --device cuda \
    --log-interval 50
```

### Running Automated Test Suites
```bash
# Run all reconstruction and 2DGS training tests
PYTHONPATH=backend backend/.venv/bin/pytest backend/reconstruction/training/tests/

# Validate shared interchange schemas
backend/.venv/bin/python3 shared/schemas/validate.py
```

