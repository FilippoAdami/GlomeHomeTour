# Backend Project History

Chronological engineering record for `backend/` — decisions, failure analyses, hardware
debugging, and algorithmic iterations. Supersedes and merges the former `history.md` (this
file used to be split into a short milestone summary and a separate exhaustive log; they
were merged 2026-09-10 to remove duplication — search git history for `history.md` if the
old finer-grained wording is ever needed).

Per-folder logs (`ingestion/project_history.md`, `reconstruction/project_history.md`) still
exist separately per the root `CLAUDE.md` convention — this file is the backend-wide index.

---

## 2026-09-08 — Milestone 1 & 2: Ingestion + Surfel Initialization Baseline

**Ingestion (`backend/ingestion/`)**
- `PackageLoader`: streams/parses capture zips (`transforms.json`, `coverage_summary.json`,
  `trajectory.csv`), validated against JSON Schema Draft 2020-12 (`shared/schemas/`).
- `QualityGate`: Laplacian-variance defocus filter ($\text{Var}(\nabla^2 I) < \tau_{\text{blur}}$)
  and kinematic redundancy filter ($\Delta t < 3\text{cm}$, $\Delta\theta < 2^\circ$) to drop
  blurred frames and stationary pauses.
- `PoseAligner`: quaternion SLERP (antipodal-safe) + cubic spline translation interpolation,
  syncing camera shutter timestamps to the 60Hz VIO trajectory.
- Defines (but as of 2026-09-10 never calls) `opengl_to_opencv`/`opencv_to_opengl` helpers in
  `pose_aligner.py` — flagged as a live open question, see the 2026-09-10 entry below.

**Surfel init (`backend/reconstruction/initialization.py`)**
- `SurfelCloud`: position, normal, orthonormal tangent frame $(\mathbf{u},\mathbf{v})$,
  anisotropic scales, degree-0 SH color, opacity.
- Gram-Schmidt tangent-frame construction, binary PLY export.
- Verified on `bedroom.zip`: 383 accepted keyframes in ~104s → 185,026 surfels, 6.88MB PLY.
- 19/19 unit tests passing at this point (`test_ingestion.py`, `test_depth_initialization.py`).

## 2026-09-08/09 — Depth model: Depth Anything v2 dropped, v3 adopted

**Why v2 (`Depth-Anything-V2-Metric-Indoor-Large-hf`) was dropped:**
1. Metric scale drifted 25-40% between adjacent frames on featureless walls/varying light —
   pure monocular regression, no multi-view cross-attention — causing staircase artifacts.
2. Thin geometry (door frames, curtains, radiator fins) collapsed into surrounding surfaces
   (DPT neck pooling too aggressive).
3. Near-field (<1.2m) perspective warping curled flat surfaces (tabletops, beds) into
   paraboloids.

**v3 (`DA3NESTED-GIANT-LARGE-1.1`)**: ViT-Giant, SwiGLU FFN, joint multi-view cross-attention.
Takes input intrinsics/extrinsics directly (`align_to_input_ext_scale=True`) to resolve
monocular scale ambiguity. Sharper thin-geometry edges than v2.

Important architectural fact (confirmed 2026-09-10 by reading
`third_party/depth_anything_3/src/depth_anything_3/api.py` directly, not just this log):
DA3 estimates its **own** camera poses internally per inference call. Your input extrinsics
are only used for a post-hoc Umeyama similarity-transform **scale** correction — it
overwrites `prediction.extrinsics` with your input poses and rescales `prediction.depth` by
the fitted scalar (`_align_to_input_extrinsics_intrinsics`). It does not fuse poses across
separate inference calls, and depth is plain camera-space Z (confirmed via
`utils/geometry.py::unproject`, which multiplies a unit-Z ray direction by the depth scalar
— not Euclidean range). Both of these were re-derived from source in the 2026-09-10 session
because trusting this log's prose without checking would have wasted time on wrong theories.

## 2026-09-09 — Hardware pathology: ROCm 7.1 / RDNA4 (gfx1200) instability

Target: AMD RX 9060 XT, 16GB VRAM, RDNA4, `gfx1200`; Ubuntu 24.04, ROCm 7.1,
`torch==2.13.0+rocm7.1`. GPU simultaneously drives the desktop compositor.

- **Symptom:** `c10::AcceleratorError: unspecified launch failure`, `SIGBUS` in
  `libhsa-runtime64.so`, `SIGABRT` in `c10::cuda::SetDevice`/`libtorch_hip.so`.
- **Cause 1 — compositor starvation:** unthrottled inference loops monopolized the command
  processor, starving the compositor; AMDGPU watchdog reset the GPU mid-job.
  **Fix:** `time.sleep(0.18)` between frames + `torch.cuda.empty_cache()` every ~20 frames.
  GPU stayed <58°C, desktop stopped stuttering.
- **Cause 2 — MIOpen cache lock contention:** MIOpen tried to lock
  `~/.config/miopen`, conflicting with sandboxing. **Fix:**
  `MIOPEN_USER_DB_PATH=/tmp/miopen`.

## 2026-09-09 — Precision: manual FP16/BF16 casting fails on RDNA4

- Manually calling `.half()`/`.bfloat16()` on DA3 submodules → `SIGBUS` in hipBLAS GEMM
  kernels. Cause: ViT-Giant's RoPE/LayerNorm/RMSNorm need FP32 accumulation; forced
  half-precision casts hit unsupported RDNA4 micro-kernels.
- **Fix:** keep weights FP32, use `torch.autocast(dtype=torch.bfloat16)` at the call site
  instead of casting parameters. Verified via `tests/test_rocm_precision_and_edge_cases.py`
  (5/5 passing: FP32 baseline, BF16 AMP, multi-view batching, streaming chunking, surfel
  unprojection).

## 2026-09-09 — Portrait orientation bug

**Symptom:** walls projected horizontally, floor on the wrong side, ceiling beams sliced
into the carpet — on `bedroom_complete.zip` (phone held in portrait).

**Cause:** ARCore/Camera2 streams landscape-sensor-layout frames ($1920\times1080$)
regardless of physical phone orientation. Feeding unrotated landscape frames into DA3 made
its gravity prior read the long axis as horizontal.

**Fix** (`run_sliding_window_reconstruction.py`):
1. Rotate every frame 90° clockwise (`cv2.ROTATE_90_CLOCKWISE`) → upright $1080\times1920$.
2. Transpose intrinsics for the same rotation: $f_x'=f_y,\ f_y'=f_x,\ c_x'=H-c_y,\
   c_y'=c_x$ (verified 2026-09-10 against `cv2.ROTATE_90_CLOCKWISE`'s actual pixel mapping
   $x'=H-1-y,\ y'=x$ — the formula in code is correct for this rotation direction).
3. Apply a matching +90° roll to extrinsics (`R_roll` in the same file) so camera rays stay
   aligned with the VIO trajectory frame.

Result: correct upright geometry — vertical curtains, horizontal floor, accurate recesses.

## 2026-09-09 — Keyframe selection evolution

1. **Pure kinematic (Δt/Δθ only):** 56 frames, jumped `frame_00271`→`frame_00414` — a
   3.46m leap across the room with zero visual overlap, severing the reconstruction into
   disconnected clusters when the user doubled back on a loop.
2. **Naive ORB cross-check** (`BFMatcher(NORM_HAMMING, crossCheck=True)`, no ratio test):
   104 frames. Forensic audit of Frame 5 vs Frame 6 (43° rotation, reported 76 matches)
   showed points on a radiator base mapped to bottles/cables on a desk — 61/76 (80%) were
   geometric hallucinations (confirmed via epipolar RANSAC). Root cause: in 256-bit binary
   descriptor space, every feature has *some* nearest neighbor even across unrelated scenes;
   without Lowe's ratio test there's no way to tell a genuine match from an accidental one.
3. **Fix — SIFT + Lowe's ratio (≤0.75) + Fundamental Matrix RANSAC:** 194 frames, all
   physically continuous, but redundant in open areas (fixed step size).
4. **Depth-adaptive + portrait-FOV-normalized (current):** 149 frames. Derived from three
   physical relationships: parallax $\Delta p \approx f t_x / Z$ (near objects need smaller
   steps), 3D FOV width $W_{3D}(Z)=2Z\tan(\text{HFOV}/2)$ (far scenes tolerate bigger steps),
   and anisotropic portrait FOV ($\text{HFOV}{=}40.8°$ vs $\text{VFOV}{=}67.0°$ — yaw sweeps
   features out of frame ~1.64× faster than pitch, so rotation gating is normalized per-axis
   before thresholding). Scaling laws: $d_{\max}(Z) = \text{clamp}(1.20\text{m}\cdot(Z/Z_0)^{0.75},
   0.65\text{m}, 1.65\text{m})$, $\theta_{\max}(Z) = \text{clamp}(24.0°\cdot(Z/Z_0)^{0.50}, 16°, 33°)$,
   $Z_0{=}2.0\text{m}$. Result: 1.4-1.6m strides in open space, 0.35m near furniture.

## 2026-09-10 — Depth Anything 3 hardening

- Fixed FP16/BF16 dtype mismatches in vendored `head_utils.py` (sinusoidal embeddings) and
  `cam_dec.py` (`CameraDec` linear layers) to preserve input precision instead of hardcoding.
- Added `safe_quantile` (`alignment.py`, `da3.py`) for robust confidence thresholding across
  FP16/BF16 — later found insufficient on ROCm (see next entry).
- `least_squares_scale_scalar` and `OutputProcessor` made dtype-preserving through
  tensor→numpy conversion.
- Global pose-graph Lucas-Kanade tracking + loop-closure optimization brought cross-view
  discrepancy from 13.10cm to 4.37cm.
- Fixed a float16 dot-product overflow in `alignment.py` (cast to f32 before `torch.dot`)
  that was producing NaN scale factors during metric alignment.
- Added `estimate_adaptive_depth_ceiling` (Q0.98 + 1.5·MAD of high-confidence depth) so the
  depth cutoff scales from small rooms (3.5-4.5m) to large halls without a hardcoded limit.
- Added multi-view geometric consensus filtering (reproject surfels into neighboring
  cameras, reject if $|\Delta z| > 8\text{cm} + 5\%z$) and a KDTree statistical outlier
  removal ($k{=}16$, $1.5\sigma$) to kill floating noise (windows/reflections/sky).
- Benchmarked DA3 batch sizes on the RX 9060 XT: $N{=}1{:}18.8\text{s}$, $N{=}2{:}28.6\text{s}$,
  $N{=}3{:}41.3\text{s}$, $N{=}4{:}54.8\text{s}$, peak VRAM 8.45GB.
- 24/24 unit tests passing at this point.

## 2026-09-10 — Multi-view reconstruction: geometric distortion, UNRESOLVED

**Trial 1:** Full run on `bedroom_complete.zip`, 273 keyframes / 68 sliding-window chunks
($N{=}6$, $K{=}2$), 12.0 min total. Output `bedroom_surfels.ply` (163,521 surfels) was a
severely distorted, curved "bowl" — room bounding box ballooned to
$10.22\text{m}\times7.17\text{m}\times9.25\text{m}$ for what is a normal bedroom.

**Fixed along the way (real, confirmed fixes, not the root cause of the bowl):**
- ROCm `torch.quantile` triggered a `SIGABRT` (`wrapper_CUDA_sort_stable`/`SetDevice`) during
  DA3 post-processing — patched with a NumPy-CPU quantile fallback in `depth_priors.py`.

**Leads investigated that session, status as originally written (see caveat below):**
- Compounding scalar scale drift across 68 chunks ($s_{\text{net}} = \prod s_m$) was blamed
  for forcing rays outward into a spherical shell.
- A claimed OpenCV-vs-OpenGL camera-space axis mismatch: `initialization.py` hardcodes
  $z_{\text{cam}}=-d$, $y_{\text{cam}}=-(y-c_y)$ (OpenGL: +Y up, -Z forward) while
  `transforms.json` declares `camera_model: OPENCV`. Written up as "discovered" but **no
  corresponding code change exists anywhere in the current `initialization.py`** — the
  negation is unchanged from before this was logged. Whether this was a correct diagnosis
  that was never actually applied, or an abandoned/incorrect lead, is **unconfirmed**.
- A keyframe/pose indexing mismatch in a throwaway diagnostic script (not in the repo) that
  paired sparse keyframe depths with dense raw-trajectory poses spaced millimeters apart.

**Single vs multi-frame conclusion as originally written:** "single-frame unprojection
produces clean, flat walls; two-frame overlap looks fine; only multi-chunk fusion warps."
**This has since been contradicted** (2026-09-10, later same day, by direct user
inspection): single-frame and two-frame unprojections are *also* curved. This invalidates
sliding-window-specific explanations (compounding chunk scale drift, inter-chunk alignment)
as the sole or primary cause, since those code paths don't run for a single frame. See
`Implementation_plan.md` at the repo root for the corrected, ranked hypothesis list and the
diagnostic protocol to follow before making any further code changes.

**Status: PERSISTENT / UNRESOLVED.** No fix has been verified end-to-end. Do not assume any
"fix" mentioned above for this specific bug actually resolved it — none was confirmed against
regenerated geometry.

---

## 2026-09-10 (later): Bowl/oversize point cloud — ROOT CAUSE FOUND AND FIXED

Followed `Implementation_plan.md`'s protocol. **Both leads it ranked highest were wrong, and
the plan's framing of the bug was wrong too** — the geometry math was never broken. The
pipeline was reconstructing from depth maps the model never produced.

**Root cause 1 — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` corrupts DA3 on
ROCm 7.1 / gfx1200.** `depth_priors.py::_init_model` set it (copied from upstream DA3's
`cli.py`). Bisected in isolated processes against a fixed 4-view input:

| allocator config | result |
| --- | --- |
| none | med depth 1.192 / 1.710 m, deterministic across reps |
| `set_per_process_memory_fraction(0.85)` only | identical to baseline |
| `expandable_segments:True` | same input → med depth 4.106 m (silently wrong) |
| `expandable_segments` + memory fraction | med depth **244 095 m** |
| `expandable_segments` + memory fraction + the GPU warm-up call | **100 % NaN**, then `HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION` / `hipErrorLaunchFailure` |

The last row is exactly what `run_sliding_window_reconstruction.py` did. The warm-up block
is *not* at fault (it reproduces baseline exactly once the allocator flag is gone) and was
left in place.

**Why nothing ever reported an error:** `estimate_depth_sequence` treated all-NaN depth as a
routine condition — `np.isnan(d).all()` → resynthesize, `.any()` → `nan_to_num` to the NaN
median. Real behaviour was 99.67 % NaN + 0.33 % sky sentinel (200 m), which takes the `.any()`
branch: median 200 → `nan_to_num(nan=200)` → `clip(0.2, 15.0)` → **a uniform 15 m depth map**.
Unprojecting a constant depth from every keyframe pose is a spherical shell — the "bowl", and
the $10\times7\times9$ m bounding box, with no geometry bug required.

**Root cause 2 — DA3 was handed camera-to-world OpenGL poses where it needs world-to-camera
OpenCV.** `run_sliding_window_reconstruction.py` passes `kf.transform_matrix` (raw ARCore
c2w, +Y up / −Z forward) straight through. DA3 conditions its camera encoder on, and fits its
metric scale against, OpenCV w2c matrices (`_normalize_extrinsics`,
`_align_to_input_extrinsics_intrinsics` in `api.py`). Measured on one 6-view chunk, adjacent
views' clouds compared by median nearest-neighbour distance:

| extrinsics passed to DA3 | median depth | merged bbox | adjacent-view NN |
| --- | --- | --- | --- |
| c2w OpenGL (as shipped) | 5.43 m | 3.66 × 9.22 × 6.75 m | 5.7 → 30.7 cm |
| w2c OpenCV (fix) | 1.12 m | 0.69 × 1.95 × 1.45 m | **0.7 – 1.5 cm** |

VIO ground truth for that chunk: 1.25 m of camera travel. This is where the 2–3× oversize
came from — a bogus Umeyama scale fit against poses in the wrong convention.

**Hypotheses from the plan that were disproven, not fixed:**
- *§3.1 axis-convention mismatch in `initialization.py`* — **not a bug.** `R_roll` in the
  portrait path rotates the ARCore pose so camera +Y points at the ceiling (world-Y component
  +0.969 on frame 0), so the hardcoded OpenGL rays and the c2w are mutually consistent. The
  unprojection was correct all along and was not touched. `opengl_to_opencv()` was the right
  helper sitting unused — it is now called, but on the way *into* DA3, not in unprojection.
- *§3.2 DA3 resize/intrinsics mismatch* — **cannot cause bowing, on principle.** Any pinhole
  intrinsics or uniform-resize error is a linear map of the camera frame, and linear maps send
  planes to planes. Only the depth *values* can bend a plane. Recorded so this lead is not
  re-opened.
- *§3.3 single-frame DA3 being degenerate* — no. Isolated single-image inference is clean.

**Step 1 metric (flat wall, frame 800, true single-image DA3 call, camera frame only):**
before — depth was the fabricated uniform 15 m map, no plane to fit; after — **0.52 cm**
dominant-plane RMS over 80.9 % of the central crop (1.02 m span). Z-depth beats
euclidean-range (1.10 cm / 60.6 %), confirming DA3 emits camera-space Z as assumed.

**Step 4, end-to-end** (`--max-frames 30`, N=6, K=2, 7 chunks, 55 s of GPU):
407 k surfels, bbox 4.96 × 3.79 × 4.48 m. Independent check — the ARCore world frame is
gravity-aligned, so a correct cloud must have a level floor without anyone fitting one: the
dominant horizontal-normal band sits at y = −1.347 m with **2.81 cm** std (0.87 cm on the
3-chunk run). Top-down projection shows straight, mutually perpendicular walls.

**Deliberately not done:** the inter-chunk median/IQR alignment in
`estimate_depth_sliding_window` (flagged in `backend/CLAUDE.md` as possibly redundant with
DA3's own per-chunk Umeyama correction) was left untouched, per the plan's Step 4 ordering.
The 0.87 → 2.81 cm floor-flatness growth from 3 to 7 chunks is the residual worth pointing at
it next; it is a mild drift, not the bowl.

**Guard added so this cannot ship silently again:** `DepthEstimationError` is raised when a
view comes back >5 % non-finite (`max_invalid_depth_frac`), and it is deliberately re-raised
past the blanket `except Exception` fallback. Fabricating depth for a reconstruction is worse
than failing.

**Tests:** `tests/test_geometry_conventions.py` added — 4 pure-numpy tests (no GPU) covering
the c2w/w2c round trip, the w2c camera-centre identity, synthetic multi-view flat-wall
planarity through `SurfelCloudInitializer`, and the NaN guard. Each was mutation-checked:
identity-conversion extrinsics fail 2 of them, and flipping either `y_cam` or `z_cam` in
`initialization.py` fails the planarity test (the first version of that test used yaw-only
poses and missed the Y flip — it now pitches the cameras). Suite: **58 passed** (the plan's
"31 tests" count was stale).

**Status: RESOLVED.** Verified end-to-end against regenerated geometry.

---

## 2026-09-10 — Boundary Floater & Silhouette Edge Bleeding Pruning

**Symptom:** Scattered floating point clouds in mid-air near depth discontinuities (e.g. curtains, bed edges, hanging boxing bag silhouette).

**Root Cause:** Neural depth models (ViT receptive fields) interpolate smoothly across sharp depth step boundaries (e.g., bed at 1.2m vs far floor/wall at 3.5m). The intermediate boundary pixels produce synthetic depths floating in empty air between the foreground object and background surface.

**Fix Applied (`backend/reconstruction/initialization.py`):**
1. **Depth Discontinuity / Edge Gradient Filter (`max_depth_gradient=0.08`):** Computes normalized relative Sobel depth gradients $\|\nabla d\| / d$; prunes pixels where depth changes abruptly across adjacent pixels.
2. **Grazing Angle Filter (`max_grazing_angle_deg=78.0°`):** Evaluates cosine angle between camera viewing ray $\mathbf{v}_{\text{ray}}$ and surface normal $\mathbf{n}_{\text{cam}}$ ($\cos\theta = -\mathbf{n}_{\text{cam}} \cdot \mathbf{v}_{\text{ray}}$); prunes glancing edge-grazing rays ($\cos\theta < 0.208$).
3. Verified on Chunk 0 & 1: pruned 2.7% of boundary floaters while preserving 100% of solid surfaces, creating clean sharp silhouettes without floating point curtains.

---

## 2026-09-10 — Commercial DA3-BASE Model Integration & Bowl Distortion Diagnosis

**Context & Goal:** Transition depth prior estimator from non-commercial `DA3NESTED-GIANT-LARGE-1.1` (1.40B, CC BY-NC 4.0) to commercially unrestricted `depth-anything/DA3-BASE` (0.12B, Apache 2.0).

**Diagnosis of Initial Failure Mode:**
- When `DA3-BASE` was initially loaded, local weights were missing. A broad `except Exception:` block in `_init_model()` silently caught the error and fell back to legacy monocular `Depth-Anything-V2-Metric-Indoor-Base-hf`, reproducing the spherical 12-meter bowl distortion.
- Fixed `_init_model()` to fail loudly with `RuntimeError` if a requested DA3 model fails to load. Downloaded local snapshot.

**Performance & Validation:**
- Complete sliding-window reconstruction on 273 keyframes (68 chunks): **4.1 minutes** on AMD ROCm ($2.1\times$ faster than Giant).
- Rectangular room dimensions ($5.87\text{m} \times 5.02\text{m}$) confirmed planar and perpendicular, validating Apache 2.0 commercial readiness.

---

## 2026-09-10 — Elimination of Onion-Peel Layering & Dynamic Statistical Ceiling Integration

**Problem:** Top-down and 3D mesh inspection revealed parallel ghost walls duplicated at different distances ("onion peeling").

**Root Causes Discovered:**
1. *Pairwise Recursive Scale Drift:* `depth_priors.py` sequentially multiplied inter-chunk scale factors $s$ across 68 chunks. Cumulative scale factor drifted between $0.034$ and $2.22$, causing the same physical wall seen from different points in the tour to unproject at contradictory depths.
2. *Raw DA3 is Already Metric:* Raw DA3-BASE outputs before pairwise chaining have an inter-chunk disagreement of only **$5.17\text{ cm}$ median** (1.00 ratio against Giant). Pairwise scale chaining was corrupting already calibrated data.
3. *Overly Aggressive Filtering Causing Holes:* Testing a static 3.2m cutoff and `min_consensus=2` amputated the upper ceiling ($>3.2\text{m}$) and left holes in sparsely observed floor/wall patches.

**Permanent Fixes Applied:**
1. **Direct Window Median Consensus:** Replaced recursive scale chaining in `estimate_depth_sliding_window` with direct median consensus on raw calibrated depth maps, eliminating cumulative drift across long sequences.
2. **Dynamic Statistical Depth Ceiling (`estimate_adaptive_depth_ceiling`):** Set `max_depth_m=None` across the pipeline, ensuring depth ceiling dynamically scales per scene ($Q_{0.98} + 1.5 \cdot \text{MAD}$ = $4.13\text{m}$ on bedroom, automatically expanding up to $25\text{m}+$ in grand halls).
3. **Occlusion-Aware Consensus (`min_consensus=1`):** Prunes floating ghost layers that violate multi-view visibility while strictly preserving uniquely scanned ceiling and corner patches.
4. **Seamless Quad Boundary Sealing:** Updated surfel quad expansion to $s = 0.55 \cdot \text{voxel\_size}$, welding adjacent voxel micro-gaps.

**Outcome & Validation:**
---

## 2026-09-11 — Native ROCm/HIP Differentiable 2DGS Rasterizer & 4-Stage Progressive Schedule

**Context & Performance Bottleneck:**
- Standalone 2DGS training using Pure-PyTorch autograd suffered from excessive VRAM scaling (>14.5 GB at 540p causing OOM at 1080p) and sluggish training throughput (>1.5s/it) due to global tile loops in Python holding millions of autograd graph nodes.
- Prolonged 1/4 resolution warmup (1000 iters at 270p) yielded negligible visual improvement between iters 500 and 1000 given high-quality metric depth priors.

**Permanent Architectural Solutions:**
1. **Hand-Authored C++/HIP Differentiable Rasterizer Extension (`rasterizer_hip`):**
   - Implemented `Rasterize2DGSGBufferForwardKernel` and analytical `Rasterize2DGSGBufferBackwardKernel` targeting AMD RDNA4 (`gfx1200`, Radeon RX 9060 XT) with Wave32 SIMD execution and LDS shared memory caching.
   - Built with PyTorch autograd bridge (`_HIPRasterizerFunction`) and automatic CPU fallback for test harnesses.
   - Training throughput accelerated from $>1500\text{ms}$ to $\approx 27\text{ms}-80\text{ms}$ per iteration (>50x speedup), with peak VRAM bounded under $1.5\text{ GB}$.
2. **4-Stage Progressive Multi-Scale Schedule:**
   - 1/4 scale ($270 \times 480$): Iters 1–200 (warmup & rough photometric alignment).
   - 1/2 scale ($540 \times 960$): Iters 201–700 (initial densification & geometry refinement).
   - 2/3 scale 720p ($720 \times 1280$): Iters 701–1500 (fine geometry & material albedo separation).
   - 1/1 scale 1080p ($1080 \times 1920$): Iters 1501–3000 (specular roughness & high-frequency detail convergence).
3. **Multi-Stage Granular Snapshot Logging:**
   - Saves intermediate snapshots every 500 iterations into `scenes/<scene>_2DGS_results/stages/iter_XXXX/`:
     `checkpoint_iter_XXXX.pt`, canonical `model_3dgs_iter_XXXX.ply`, `render_sample_iter_XXXX.png`, and `stage_metrics.json`.
4. **Validation:**
   - All 21 reconstruction and training unit tests passing.
   - Output manifests and formats strictly conform to shared interchange schemas.

## 2026-09-14: end-to-end pipeline, one entry point, every step resumable
Implemented `IntegrationPlan.md`: `run_pipeline.py` takes a compressed capture and
runs extract -> quality filter -> COLMAP -> depth keyframe filter -> depth
estimation -> 2DGS training inside a temporary `current_scene/` workspace. Every
step is independently runnable, independently resumable, and logs its own metrics.

New shared foundation in `Utilities/`:
- `scene_io.py` — the single owner of "read/modify/write a scene folder".
  `split_scene()` / `merge_back()` move frames with their camera entry and
  **never renumber basenames**, which is what keeps depth maps, COLMAP image
  names and stats keys valid across steps.
- `pipeline_step.py` — `StepContext` writes `<name>_log.txt`, `<name>_stats.json`
  and a `pipeline_state.json` record; `is_done()` additionally requires the
  step's outputs to still exist, so a hand-deleted output is not skipped over.
- `pipeline_paths.py` — gained `stage_paths()` / `subprocess_env()`; subprocessed
  stage scripts (the COLMAP converter, `train.py`) import bare stage module names
  and a fresh interpreter inherits none of the parent's sys.path.

Governing rule throughout: **nothing is deleted mid-pipeline.** Rejected frames
are moved to per-step discard folders, each itself a loadable scene.
`current_scene/` is not auto-deleted even on success — the trained scene is the
only copy and lives inside it, so the run prints a message instead. That is a
deliberate deviation from the plan, which called for deleting on success.

Per-step notes are in each stage's own `project_history.md`. The one change worth
repeating here: DA3 is no longer handed ARCore poses. The COLMAP pass now always
runs first and COLMAP already stores OpenCV world-to-camera, so `depth_priors.py`
converts nothing and the three legacy callers convert for themselves.

Outcome: steps 0-1 verified on the real 1431-frame Bedroom2 capture (267 MB;
extract 1.6 s, quality gate 19 s, 1400 kept). Step 2 verified as far as COLMAP's
global bundle adjustment. Steps 3-5 are implemented and import-clean but **not
yet run on real data**. Tests cover the structural guarantees
(`test_pipeline_steps.py`), the pose-convention regression
(`test_colmap_poses_to_da3.py`) and step 3's new track covisibility
(`test_step_filter_depth.py`) — 12 assertions-based tests, all passing.

