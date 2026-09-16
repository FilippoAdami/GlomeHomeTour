# Glome Home Tour — Backend Reconstruction & Ingestion Engine

**Glome Home Tour Backend** is an automated high-precision spatial compute pipeline that ingests smartphone sensor captures (Camera2 RGB video + 60 Hz ARCore VIO trajectory) and reconstructs MLS-compliant digital listing assets:
1. **Interactive 2D/3D Gaussian Splatting (2DGS) Web Walkthrough** ($\le 25\text{ MB}$ compressed).
2. **Automated Vector Floor Plan** with metric dimensions and room square meterage.
3. **Synthetic 4K 360° Equirectangular Panoramas** synthesized from semantic centroids.

---

## 1. Repository Structure

Source code under `backend/` is organized into **numbered pipeline-stage folders**, ordered to
match the actual data flow from a raw capture to a refined mesh. Each folder is a flat
collection of modules (no nested subpackages) importing each other by bare module name — see
"Cross-stage imports" below for why.

```
backend/
├── 00_ingestion/           # Step 1: package loading, quality gate, parallax dedup, VIO pose sync
│                            # -> filtered images + transforms.json per scene
├── 01_poses_refinment/      # Step 2: COLMAP triangulation against fixed ARCore poses,
│                            #   drift/Sampson diagnostics -> sparse/ reconstruction
├── 02_depth_estimation/      # Step 3: Depth Anything 3 inference + surfel-cloud initialization
│                            #   from the refined poses/images
├── 03_2DGS_training/           # Step 4: 2DGS training/rendering/metrics, merged in from the
│   ├── scene/ utils/ arguments/   #   standalone 2DGS project; reads COLMAP sparse/ (stage 2)
│   ├── gaussian_renderer/         #   or transforms.json (stage 3)
│   └── submodules/                #   diff-surfel-rasterization (HIP kernels, ROCm/RDNA4)
├── 04_2DGS_refinment/         # Step 5: material learning & 2DGS refinement (unfilled; also
│                            #   holds compressor/pbr_shader/planar_reflections/export_standard_ply
│                            #   parked by the 2DGS merge — see its README)
├── 05_2DGS_to_mesh/             # Step 6: mesh extraction (SuGaR-like) from the trained 2DGS scene
│   ├── exporters/                  #   CAD (.dxf/.ifc) and glTF exporters
│   └── kernels/                     #   surfel-projection compute kernels
├── 06_mesh_refinment/             # Step 7: mesh refinement (unfilled)
├── Utilities/                       # cross-stage tooling, not itself a pipeline step
│   ├── pipeline_paths.py               # sys.path bootstrap (see below)
│   ├── run_full_benchmark.py            # end-to-end benchmark spanning multiple stages
│   ├── worker/                           # RQ queue consumer / GPU job runner
│   └── third_party/                       # vendored deps (e.g. depth_anything_3)
├── tests/                            # top-level test suite (pytest `testpaths`)
├── scenes/                            # per-scene capture archives & pipeline output (see §6)
├── conftest.py                         # bootstraps sys.path for pytest collection
└── pipeline_paths.py -> Utilities/pipeline_paths.py  (imported as `Utilities.pipeline_paths`)
```

Folders described as "unfilled" above exist as placeholders for work not yet migrated in; treat
them as reserved names, not indicators that the stage is unimplemented product-wide.

**Cross-stage imports:** stage folder names start with a digit (`00_ingestion`, ...), which can't
be a dotted Python package name (`from 00_ingestion import x` is a syntax error). Instead, every
entry-point script and `conftest.py` calls `Utilities.pipeline_paths.bootstrap()`, which adds
every stage directory to `sys.path`. Modules then import each other by flat name regardless of
which stage's script is the current entry point, e.g. `02_depth_estimation/dataset.py`-adjacent
code can do `from package_loader import CameraIntrinsics` even though `package_loader.py` lives in
`00_ingestion/`. `Utilities/` itself is a real dotted package (`Utilities.worker`,
`Utilities.pipeline_paths`) since it isn't numbered.

---

## 2. Hardware Targets & Execution Environment

The backend is engineered for high-throughput local workstation compute:
- **Primary GPU Target:** AMD Radeon RX 9060 XT (16 GB VRAM, RDNA4 architecture, `gfx1200` ISA).
- **Driver & Framework:** AMD ROCm 7.1, PyTorch `2.13.0+rocm7.1`, Linux kernel 6.x.
- **Display Server Interoperability:** Because the target hardware simultaneously drives the host display server (X11/Wayland compositor), the engine enforces non-blocking compositor yields (`time.sleep(0.18)`) and isolated cache memory namespaces (`MIOPEN_USER_DB_PATH=/tmp/miopen`).

---

## 3. Ingestion & Dynamic Keyframe Filtering Engine

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
[ Depth Anything 3 ] ────► Multi-View Metric Depth Estimation (DA3-Base, BFloat16 AMP)
    │
    ▼
[ Surfel Cloud Init ] ───► 2D Gaussian Surfel Initialization (Positions, Normals, Scales)
```

---

## 4. Mathematical Specifications

### 4.1 Smartphone Vertical Portrait Optics (Anisotropic Field of View)
When a smartphone is held in portrait orientation, the sensor aspect ratio ($1080 \times 1920$) produces an anisotropic field of view:
$$\text{HFOV} = 2 \arctan\left(\frac{W}{2 f_x}\right) \approx 40.8^\circ$$
$$\text{VFOV} = 2 \arctan\left(\frac{H}{2 f_y}\right) \approx 67.0^\circ$$

Because the horizontal field of view is narrow ($\text{HFOV} = 40.8^\circ$), horizontal camera panning ($\Delta\theta_{\text{yaw}}$) displaces visual features out of the frame **$\sim 1.64\times$ faster** than vertical tilting ($\Delta\theta_{\text{pitch}}$). 

To normalize rotational displacement across both axes, the engine computes the **Anisotropic Effective Rotation** $\Delta\theta_{\text{norm}}$:
$$\Delta\theta_{\text{norm}} = \sqrt{ \left(\frac{\Delta\theta_{\text{yaw}}}{\text{HFOV}}\right)^2 + \left(\frac{\Delta\theta_{\text{pitch}}}{\text{VFOV}}\right)^2 + \left(\frac{\Delta\theta_{\text{roll}}}{\text{HFOV}}\right)^2 } \times \text{HFOV}$$

---

### 4.2 Depth-Adaptive Scaling Laws
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

### 4.3 Epipolar Geometric Verification & Lowe's Ratio Test
To prevent false-positive visual overlap on repetitive indoor textures (radiator fins, wood grain, blinds), every candidate keyframe must pass two-tier geometric verification:

1. **Lowe's Second-Nearest-Neighbor Ratio Test:**
   $$\frac{\|\mathbf{d}_i - \mathbf{d}_{j, 1}\|_2}{\|\mathbf{d}_i - \mathbf{d}_{j, 2}\|_2} < 0.75$$
   Eliminates ambiguous descriptor matches in high-dimensional SIFT space.

2. **Fundamental Matrix Epipolar Constraint:**
   $$\mathbf{x}_2^T \mathbf{F} \mathbf{x}_1 = 0$$
   Computed via RANSAC (`cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0)`). Candidate frames are rejected if verified inliers drop below $\tau_{\text{inliers}} = 8$, preventing disconnected scene islands.

---

## 5. Depth Prior Architecture: Depth Anything 3

The reconstruction engine utilizes **Depth Anything 3 Base (`depth-anything/DA3-BASE`, 0.12B
parameters)** — **not** the Giant/Large checkpoints. This is a licensing constraint, not a
quality tradeoff: DA3-Base ships under Apache 2.0 (commercially unrestricted), while DA3's
Giant/Large checkpoints are CC BY-NC (non-commercial). Using the larger model would make the
whole project commercially unusable. Never swap in a Giant/Large checkpoint without re-checking
its license first.

- **Backbone:** Vision Transformer (ViT-Base) with SwiGLU feed-forward networks.
- **Precision:** Automatic Mixed Precision (BFloat16 AMP) on ROCm with native FP32 accumulator retention.
- **Metric Scale Alignment:** Ingests metric camera intrinsics and extrinsics via `align_to_input_ext_scale=True`, enforcing physical metric consistency across sequential frames.

---

---

## 6. Scene Output Layout & Pipeline Stages

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

## 7. One-Command Pipeline (`run_pipeline.py`)

A compressed capture in, a trained 2DGS scene out:

```bash
.venv/bin/python run_pipeline.py scenes/Bedroom2.zip
```

Work happens in a temporary `current_scene/` workspace. Six steps, each also a
standalone script that can be run, inspected and resumed on its own:

| # | Step | Script | Writes |
| --- | --- | --- | --- |
| 0 | extract | `Utilities/step_extract.py` | `images/`, `transforms.json` |
| 1 | filter_quality | `00_ingestion/step_filter_quality.py` | `discarded/` |
| 2 | colmap | `01_poses_refinment/step_colmap.py` | `sparse/0/`, `colmap_diagnostics/` |
| 3 | filter_depth | `02_depth_estimation/step_filter_depth.py` | `depth_discarded_images/`, `depth_discarded_sparse/` |
| 4 | depth | `02_depth_estimation/step_depth.py` | `depth/depth_maps/`, `depth/points3D_depth.ply` |
| 5 | train | `03_2DGS_training/step_train.py` | `2dgs/point_cloud/iteration_10000/` |

**Nothing is deleted mid-pipeline.** A frame a step rejects is *moved* into that
step's own discard folder together with its camera entry, so every discard folder
is itself a loadable scene and any step can be re-run in isolation. Frame
basenames are never renumbered — depth maps, COLMAP image names and stats keys
are all keyed by basename.

One exception, and it is not cosmetic: **step 3 prunes `sparse/0/` in place**, and
`merge_back()` restores images and `transforms.json` but cannot un-prune a COLMAP
model. Re-running step 3 with `--force` therefore selects against a model already
missing the frames it just restored, and emits a scene whose model covers fewer
cameras than its own `transforms.json` — silently, since step 5 simply trains on
the smaller set. Step 2 owns `sparse/0/`, so re-running step 3 against a different
threshold means re-running step 2 first. `filter_depth()` enforces this: it
refuses to start when `sparse/0/` does not cover every frame in `transforms.json`.

Each step writes `<name>_log.txt`, `<name>_stats.json` and an entry in
`pipeline_state.json`; a completed step skips on re-run, and the run ends with
`pipeline_summary.txt`.

```bash
--from-step colmap      # start there, skip everything earlier
--only-step depth       # run exactly one step
--force                 # re-run; restores previously discarded frames first
                        # (frames and transforms.json only -- not a pruned sparse/0/)
--keep-workspace        # keep current_scene/ on success
```

Out of scope here: `04_2DGS_refinment/`, `05_2DGS_to_mesh/`, floorplan, panorama,
and API/queue wiring.

## 8. Execution & Verification

### Running Stage 2 (COLMAP pose refinement)
Requires a system COLMAP binary on `PATH`, and `03_2DGS_training/` on `PYTHONPATH` (these
scripts import `scene.colmap_loader` from there):

```bash
PYTHONPATH=backend/03_2DGS_training backend/.venv/bin/python3 \
    backend/01_poses_refinment/convert_transforms_to_colmap.py \
    -s backend/scenes/<scene_name>/GS_input --refine_poses --diagnostics
```

### Running 2DGS Model Training
To train the progressive multi-scale 2DGS radiance field on AMD ROCm:
```bash
cd backend/03_2DGS_training && ../.venv/bin/python3 train.py \
    -s ../scenes/<scene_name>/GS_input \
    -m ../scenes/<scene_name>/2DGS_results \
    --iterations 30000
```

`train_room.py` wraps this in a 4-stage progressive-resolution schedule (1/8 -> 1/4 -> 1/2 ->
full). `render.py` renders held-out views and extracts a TSDF mesh; `metrics.py` reports
PSNR/SSIM/LPIPS. Full flag reference: `03_2DGS_training/PIPELINE_NOTES.md`.

The two native extensions must be built once into the venv first:
```bash
backend/.venv/bin/pip install --no-build-isolation \
    backend/03_2DGS_training/submodules/diff-surfel-rasterization \
    backend/03_2DGS_training/submodules/simple-knn
```
`--no-build-isolation` is required: an isolated PEP 517 env cannot see the ROCm torch build,
and these extensions import `torch` at setup time.

### Running Automated Test Suites
```bash
# Run the backend suite (pytest testpaths = backend/tests/)
backend/.venv/bin/pytest backend/tests/

# Validate shared interchange schemas
backend/.venv/bin/python3 shared/schemas/validate.py
```

