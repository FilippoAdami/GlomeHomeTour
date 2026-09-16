# Integration Plan — end-to-end reconstruction pipeline

Status: **plan only, nothing implemented.** Written 2026-09-14, after the 2DGS merge
(see `03_2DGS_training/project_history.md`).

Goal: one entry point that takes a path to a compressed capture and runs
extract -> quality filter -> COLMAP -> depth keyframe filter -> depth estimation -> 2DGS
training, in a temporary `current_scene/` workspace, where **every step is independently
runnable, independently resumable, and logs its own metrics**.

---

## 1. Scope

**In scope:** steps 0-5 below, the workspace contract, the step protocol, the orchestrator,
and the edits to existing files that integration forces.

**Out of scope** (explicitly deferred, per the brief): 2DGS refinement (`04_2DGS_refinment/`),
mesh extraction (`05_2DGS_to_mesh/`), floorplan, panorama, API/queue wiring.

**Design rule driving everything:** nothing is ever deleted mid-pipeline. Every frame removed by
a step is *moved* to a discard folder together with its camera information, so any step can be
inspected, re-run, or resumed in isolation. `current_scene/` is deleted only on a fully
successful run.

---

## 2. Workspace layout

`current_scene/` (default `backend/current_scene/`, overridable with `--workspace`):

```
current_scene/
├── images/                       # the live set — shrinks at steps 1, 2, 3
├── transforms.json               # always describes exactly what is in images/
├── sparse/0/                     # COLMAP model (step 2), pruned at step 3
│   ├── cameras.{bin,txt}
│   ├── images.{bin,txt}
│   ├── points3D.{bin,txt}
│   └── points3D.ply              # cleaned cloud (step 2)
├── images_2/  images_4/  images_8/   # built at step 5, after the image set is final
│
├── discarded/                    # step 1 rejects
│   ├── images/
│   └── transforms.json
├── colmap_discarded_images/      # step 2 rejects (same internal shape)
│   ├── images/
│   └── transforms.json
├── depth_discarded_images/       # step 3 rejects
│   ├── images/
│   └── transforms.json
├── depth_discarded_sparse/0/     # step 3: the COLMAP entries for those frames
│
├── depth/                        # step 4 outputs
│   ├── depth_maps/*.npy
│   └── points3D_depth.ply        # voxel-downsampled fused cloud (1 cm)
├── 2DGS_results/                 # step 5 outputs
│
├── pipeline_state.json           # step -> {status, started, finished, output_hash}
├── extract_log.txt      extract_stats.json
├── filter_quality_log.txt        filter_quality_stats.json
├── colmap_log.txt                colmap_stats.json
├── filter_depth_log.txt          filter_depth_stats.json
├── depth_log.txt                 depth_stats.json
└── train_2dgs_log.txt            train_2dgs_stats.json
```

### Why each discard folder is a self-contained mini-scene

`shared/schemas/transforms.schema.json` is **frozen at 1.0.0**, with
`"additionalProperties": false` and `file_path` constrained to `"^images/"`. So a discard
manifest cannot legally carry a `reason` field, and cannot legally point at
`discarded/frame_0007.jpg`.

Therefore each discard folder gets its own `images/` + `transforms.json` pair with
`file_path: "images/..."` relative to that folder. The result stays schema-valid and is directly
loadable by any existing tool. Per-frame *reasons and measurements* go in the step's
`*_stats.json`, keyed by filename — never in the transforms file.

---

## 3. Step protocol (shared contract)

Two new helpers, both in `Utilities/` (not a numbered stage, so it is importable as a real
package):

### `Utilities/scene_io.py`
The single owner of "read/modify/write a scene folder". Every step uses it; no step hand-rolls
JSON surgery.

- `load_scene(dir) -> Scene` — `transforms.json` + image paths, validated against the frozen schema.
- `write_scene(dir, header, frames, *, validate=True)` — schema-checked write.
- `split_scene(src_dir, dst_dir, reject_names) -> (kept, rejected)` — the workhorse: moves the
  rejected image files into `dst_dir/images/`, writes `dst_dir/transforms.json` for them, rewrites
  `src_dir/transforms.json` to the survivors. Atomic: builds both JSONs in memory, moves files,
  then writes, so an interrupted step leaves a recoverable state.
- `merge_back(dst_dir, src_dir)` — inverse, for debugging a rejection.

This is a generalization of `00_ingestion/run_staged_filtering.py::write_stage`, which already
writes a schema-validated `images/` + `transforms.json` pair. Lift its body rather than rewrite.

### `Utilities/pipeline_step.py`
- `StepContext(name, workspace)` context manager: timestamps, writes `<name>_log.txt` (human) and
  `<name>_stats.json` (machine), records `pipeline_state.json` on success/failure.
- `ctx.metric(key, value)`, `ctx.timer("sub_phase")`, `ctx.note(text)`.
- `ctx.skip_if_done(outputs=[...])` — resume logic: a step whose declared outputs exist and whose
  state entry says `ok` is skipped unless `--force`.
- On failure, state is written as `failed` with the traceback in the log; the workspace is left
  untouched for inspection.

### Every step script
- Has its own `argparse` `main()` and is runnable standalone:
  `python 01_poses_refinment/step_colmap.py --workspace current_scene/`
- Defaults `--workspace` to `backend/current_scene/`, so bare invocation works.
- Reads only from the workspace; writes only into the workspace.
- Never deletes an image — only `split_scene`.
- Is idempotent: re-running a completed step with `--force` reproduces the same result from the
  same inputs (discard folders are merged back first).

---

## 4. Steps

### Step 0 — `Utilities/step_extract.py` (new)

**In:** path to `.zip`. **Out:** `current_scene/images/` + `transforms.json`.

Extract the archive into the workspace. Handle the three real layouts (flat; single wrapper dir
as in `scenes/Bedroom2/scan_20260913_133535/`; already-extracted directory) by locating the
`transforms.json` and treating its parent as the scene root. Validate against the frozen schema
and assert every `frames[].file_path` resolves to a real file.

**Log:** archive path/size, resolved scene root, frame count, image resolution, intrinsics,
declared distortion (`k1,k2,p1,p2`), extraction time, schema validation result.

**Risk:** capture zips may be the *mobile package* format (video + trajectory CSV) rather than
images + transforms. `00_ingestion/package_loader.py` handles that shape. Decide at
implementation time whether step 0 also runs `PackageLoader` for that layout; if so it becomes
"extract or ingest". `scenes/Bedroom2.zip` should be checked first as the reference input.

---

### Step 1 — `00_ingestion/step_filter_quality.py` (new thin wrapper)

**In:** `images/` + `transforms.json`. **Out:** survivors in place; rejects in `discarded/`.

Blur / overexposure / underexposure only — this is stage 1 of the existing three-stage filter.
Reuse `quality_gate.py::QualityGate` **unchanged**; it already implements exactly this and its
thresholds are tuned (relative blur at 0.45 of scene-median sharpness combined with an absolute
floor, `dark_threshold=12.0`, `blown_threshold=250.0`, plus `_apply_reject_cap` which re-accepts
the sharpest rejects when the cull is implausibly large).

The wrapper's only job: run the gate over the workspace, then `split_scene(..., 'discarded')`.

> Note: the existing parallax/keyframe stages 2 and 3 of `run_staged_filtering.py` are **not**
> used here. They move to step 3, where real COLMAP geometry replaces their estimated depths.

**Log:** total in / kept / discarded + percentage; per-reason counts (blurred, too dark, blown,
low texture); for each discarded frame its metric value and the threshold it missed, and the mean
and worst margin below threshold; whether the reject cap fired; elapsed time.

---

### Step 2 — `01_poses_refinment/step_colmap.py` (new wrapper)

**In:** cleaned `images/` + `transforms.json`. **Out:** `sparse/0/` with a cleaned
`points3D.ply`; COLMAP-unusable frames in `colmap_discarded_images/`.

1. Run `convert_transforms_to_colmap.py` with `-s current_scene/` and **`--no_keyframes`**.
   Multi-resolution folders are deliberately *not* generated here (see step 5 — the image set is
   not final until step 3, and stale `images_4/` entries would silently poison training).
   Recommended flags: `--refine_poses --diagnostics <ws>/colmap_diagnostics --max_sampson_px 2.0`.
2. Determine the unusable frames — the two mechanisms already exist in
   `01_poses_refinment/export_keyframes.py` and should be reused, not reimplemented:
   - `registered_names(sparse_dir)` — frames COLMAP actually registered; unregistered frames have
     no tracks and carry no geometry.
   - `sampson_rejects(db_path, sparse_dir, max_px)` + `filter_by_sampson(...)` — frames whose pose
     disagrees with the raw verified matches. **Keep its `MAX_SAMPSON_REJECT_FRAC = 0.2` refusal**:
     if the filter wants to drop >20% it reports instead of applying, because that means the model
     is wrong, not the frames. Same philosophy as the rotation-drift refusal.
3. `split_scene(..., 'colmap_discarded_images')` for the union of both sets, and
   `prune_model(sparse_dir, keep)` (also already in `export_keyframes.py`) so `sparse/0/` matches
   `images/` exactly.
4. Clean the triangulated cloud with the **existing Open3D path** in
   `densify_pointcloud.py`: `voxel_dedup(...)` then `remove_outliers(...)`
   (`pcd.remove_statistical_outlier`, defaults `nb_neighbors=20`, `std_ratio=1.5`). Write the
   result to `sparse/0/points3D.ply`.

**Log:** frames in / registered / Sampson-rejected / final kept; whether the Sampson refusal
triggered; wall time per sub-phase (feature_extractor, matcher, point_triangulator, BA,
diagnostics); points in the raw triangulated cloud, points after voxel dedup, points after SOR,
and both as absolute + percentage removed; mean/median reprojection error before and after;
rotation-drift outlier count.

**Risks:**
- `convert_transforms_to_colmap.py` **errors out by design on non-zero distortion**, and step 0
  passes the capture's `k1,k2,p1,p2` through unchanged. Not a design blocker: converting a
  `transforms.json` into a `sparse/` folder with COLMAP has already been done by hand several
  times on real captures without trouble. Confirm on `scenes/Bedroom2.zip` when this step is
  first exercised; if a capture does declare distortion, undistort in step 0 and zero the
  coefficients before handoff.
- Requires the system COLMAP binary (4.3.0.dev0, HIP build) on `PATH`; it is not a pip dependency.
- `points3D.ply` is what the 2DGS `sparse/` loader picks up by default. Step 4 overwrites the
  init cloud choice — see step 5.

---

### Step 3 — `02_depth_estimation/step_filter_depth.py` (new)

**In:** `images/` + `transforms.json` + `sparse/0/`.
**Out:** a reduced set; rejects in `depth_discarded_images/` and `depth_discarded_sparse/0/`.

This is the "merged second and third level filter" — the parallax and keyframe stages of
`run_staged_filtering.py`, but driven by **real COLMAP geometry instead of estimated depth**.

Reuse `00_ingestion/keyframe_selector.py::DynamicKeyframeSelector`, which already implements
depth-adaptive translation/rotation gating plus a covisibility floor and ceiling, and already has
a `for_2dgs_training(...)` factory (`min_translation_m=0.08`, `min_rotation_deg=4.0`,
`min_covisibility=0.50`, `max_covisibility=0.80`).

The one substantive change: **replace its inputs, not its logic.**
- Instead of `estimate_scene_depths()` (optical-flow triangulation, ~40 ms/frame, approximate),
  compute each frame's depth from the cleaned cloud: median z of that frame's triangulated points
  in camera coordinates, parsed from `images.txt`/`points3D.txt` via
  `colmap_diagnostics.py::parse_images_txt`. Feed it in through the existing
  `scene_depths=` parameter of `select_keyframes()` — the selector already accepts it.
- Instead of the frustum-overlap approximation at a reference plane, compute **true covisibility**
  from shared tracks: `|tracks(i) ∩ tracks(j)| / min(|tracks(i)|, |tracks(j)|)`. This is exactly
  the "features matching" signal the brief asks for, and it is exact where the frustum model is a
  proxy. Implement as an alternative covisibility backend so the selector's accept/reject logic is
  untouched.

Target: enough frames for full coverage and solid consecutive overlap, meaningfully fewer than the
full set. Enforce with `min_keyframes`/`max_keyframes` bounds and a coverage guard — never drop a
frame whose removal would push consecutive overlap below `min_covisibility`.

Then `split_scene(..., 'depth_discarded_images')`, and prune the discarded frames out of
`sparse/0/` into `depth_discarded_sparse/0/` (reusing `prune_model`, plus its inverse to write the
removed entries).

**Log:** frames in / selected / discarded + percentage; the covisibility distribution of the kept
consecutive pairs (min/mean/median — the number that proves coverage held); per-discarded-frame
the rule that dropped it (redundant overlap / insufficient baseline / insufficient rotation) and by
how much it missed; median scene depth per frame and the resulting adaptive thresholds; number of
sparse points orphaned by the prune; elapsed time.

---

### Step 4 — `02_depth_estimation/step_depth.py` (new wrapper)

**In:** filtered `images/` + `sparse/0/`. **Out:** `depth/depth_maps/*.npy`,
`depth/poses_da3.npz` (substep 4a) and `depth/points3D_depth.ply`, voxel-downsampled on a
**1 cm** grid.

Run DA3 sliding-window inference and surfel-cloud initialization, then downsample at 1 cm
(`--voxel-size 0.01`; today the runner defaults to `0.015` and `SurfelCloudInitializer` to `0.02`).

**This step requires the one unavoidable change to stage 3's entry point.**
`run_sliding_window_reconstruction.py` currently takes `--scene <zip>` and performs *its own*
package loading, quality gating and frame selection internally. In the new pipeline all of that
has already happened in steps 0-3, and the refined COLMAP poses — not the raw ARCore ones — are
what must be used. So the runner needs a workspace-input mode.

Scope the change tightly:
- **Change:** the CLI/orchestration layer only — accept `--workspace` with prepared
  `images/` + `sparse/`, bypassing the internal load/gate/select path.
- **Do not touch:** DA3 inference, the **body** of `arcore_c2w_to_da3_w2c()`, the autocast
  handling, the cooperative `time.sleep(0.18)` yields and `empty_cache()` calls, or the
  unprojection in `initialization.py`. Those are documented-correct and hardware-fragile
  (`backend/CLAUDE.md`). The only permitted change in this area is the additive
  `poses_are_w2c` flag that lets substep 4a *skip* that call — the conversion itself is left
  exactly as written, and every existing caller keeps today's behavior by default.
- Prefer a **new thin wrapper module that imports `DepthPriorEstimator` and
  `SurfelCloudInitializer` directly** over editing the existing script's `main()`. That leaves the
  existing script working as-is and keeps the blast radius near zero.
- This also retires `run_sliding_window_reconstruction.py:587`'s dead
  `from run_scene_training import ...` (flagged in the merge) — the new orchestrator owns
  step sequencing, so the `--train-2dgs` shortcut becomes redundant.

#### Substep 4a — `02_depth_estimation/colmap_poses_to_da3.py` (new, small)

DA3 **must** consume the COLMAP-refined poses from `sparse/0/`, never the raw ARCore poses in
`transforms.json` — the refined ones are the corrected ones, and feeding ARCore poses here
reintroduces exactly the drift step 2 just removed.

The good news: **no conversion math is actually required.** Tracing both sides:

- `arcore_c2w_to_da3_w2c()` (`depth_priors.py:61`) computes
  `inv(opengl_to_opencv(c2w_gl))` — i.e. flip the OpenGL axes to OpenCV, then invert to
  world-to-camera. DA3 wants **OpenCV-convention world-to-camera** 4x4 matrices.
- `convert_transforms_to_colmap.py:174-177` writes COLMAP poses as
  `c2w = transform_matrix @ OPENGL_TO_OPENCV; w2c = inv(c2w)`, storing `qvec = rotmat2qvec(w2c[:3,:3])`
  and `tvec = w2c[:3,3]`.

These are the *same composition*. COLMAP's stored `(qvec, tvec)` is already the OpenCV
world-to-camera pose DA3 expects. So the substep is a direct assembly:

```python
R = qvec2rotmat(img.qvec)          # scene.colmap_loader
w2c = np.eye(4, dtype=np.float32)
w2c[:3, :3] = R
w2c[:3,  3] = img.tvec
```

**The hazard is doing more than this.** Passing these matrices through
`arcore_c2w_to_da3_w2c()` would apply a second inversion and a second axis flip, producing
plausible-but-wrong depth — the precise failure mode `backend/CLAUDE.md` documents (5-30 cm
adjacent-view disagreement, ~5x oversized scene). The wrapper in step 4 must call
`estimate_depth_sequence(...)` on a path that does **not** re-apply that conversion. Since
`DepthPriorEstimator.estimate_depth_sequence` currently calls `arcore_c2w_to_da3_w2c()`
internally (`depth_priors.py:298`), add a `poses_are_w2c: bool = False` parameter that skips the
conversion when the caller already supplies world-to-camera. That is a strictly additive change
with a default preserving today's behavior for every existing caller.

Two details the substep owns:
- **Ordering.** COLMAP's `images.txt` is keyed by image id, not capture order. Sort by image
  `name` and build the extrinsics array parallel to the image list handed to DA3; a mismatch
  silently pairs each frame with another frame's pose.
- **Intrinsics.** Build per-frame `K` (3,3) from `cameras.txt`. If images are downscaled for
  inference, `fx, fy, cx, cy` must be scaled by the same factor; the COLMAP model keeps
  full-resolution intrinsics.

Output a `depth/poses_da3.npz` (`w2c` (N,4,4) float32, `K` (N,3,3) float32, `names` (N,)) so the
conversion is inspectable and step 4 is resumable without re-reading the COLMAP model.

**Validation gate:** after conversion, assert `det(R) ≈ +1` and orthonormality for every pose,
and check the recovered camera centers `-R.T @ t` against the `transforms.json` positions — they
should agree to within the refinement magnitude (centimetres), *not* be mirrored or inverted. A
failure here is the double-application bug.

**Log:** frames processed; chunk count/size/overlap; per-chunk inference time and total; DA3 model
name; points before and after 1 cm voxel downsampling (absolute + % reduction); final point count;
adjacent-view agreement (cm) as the geometry sanity check; floor-flatness residual in the
gravity-aligned frame; peak VRAM; any chunk that fell back or produced NaNs.

---

### Step 5 — `03_2DGS_training/step_train.py` (new wrapper)

**In:** final `images/`, `sparse/0/`, `depth/points3D_depth.ply`. **Out:** `2DGS_results/`.

1. **Build `images_2/`, `images_4/`, `images_8/` here** — the first point at which the image set is
   final. Reuse `export_keyframes.py::_resize_one` (threaded PIL resize, quality 95) and
   `_link_or_copy` (hardlinks, so full-res costs nothing). Do not re-derive intrinsics: as
   `export_keyframes.py` documents, `cameras.txt` keeps full-resolution intrinsics and the 2DGS
   loader derives resolution-independent FovX/FovY, sizing each camera from the file it loads —
   which is why `--images images_4` needs no model change.
2. **Init cloud.** `Scene.__init__` prefers the `sparse/` branch and reads
   `sparse/0/points3D.ply`. The dense 1 cm depth cloud is the better initialization, so step 5
   installs `depth/points3D_depth.ply` as `sparse/0/points3D.ply` (backing up the COLMAP-triangulated
   cloud as `points3D_colmap.ply`). This resolves the ordering collision flagged in the merge, with
   no change to merged 2DGS code.
3. **Progressive training**, reusing `train_room.py`'s structure: four stages 1/8 -> 1/4 -> 1/2 ->
   full, each resuming from the previous checkpoint. Note `train_room.py` currently expects
   `images_keyframes` / `transforms_keyframes.json` (`IMAGES_BASE`/`TRANSFORMS_BASE`) and drives
   COLMAP itself — the wrapper should call `train.py` per stage directly rather than reuse
   `train_room.py`'s `main()`, keeping only its staging concept.

**Iteration budget — capped at 10k** (from 30k), scaled from `train_room.py`'s cumulative
`STAGES = [(_8, 8, 7_000), (_4, 4, 14_000), (_2, 2, 21_000), ("", 1, 30_000)]`. These are
cumulative targets, not per-stage durations. Same proportions at 10k:

| stage | images | cumulative target |
| --- | --- | --- |
| 1/8 | `images_8` | 2,333 |
| 1/4 | `images_4` | 4,667 |
| 1/2 | `images_2` | 7,000 |
| full | `images` | 10,000 |

**Iteration-coupled parameters — approved 2026-09-14.** Three `OptimizationParams` defaults are
schedules expressed in iterations, not tuning knobs, and are rescaled with the run length:

| parameter | 30k default | at 10k | why |
| --- | --- | --- | --- |
| `densify_until_iter` | 15,000 | **5,000** | larger than the whole run at 10k; densification would never terminate and `max_gaussians` would become the only limit |
| `position_lr_max_steps` | 30,000 | **10,000** | LR would otherwise anneal through only a third of its schedule, ending ~3x too high |
| `densify_from_iter` | 500 | **167** | keeps the warm-up the same fraction of the run |

`opacity_reset_interval = 3_000` is left as-is (3 resets, last at 9k).

**Everything else is untouched**, which is the actual intent of "keep the current
hyperparameters": all learning rates (`feature_lr`, `opacity_lr`, `scaling_lr`, `rotation_lr`,
`position_lr_init/final`), all loss weights (`lambda_dssim`, `lambda_dist`, `lambda_normal`,
`lambda_multiview`, `lambda_track`), `percent_dense`, `opacity_cull`, `densify_grad_threshold`,
`densification_interval`, `max_gaussians`, `sh_degree`, and the multi-view / pose-refinement
settings.

**Log:** per stage — resolution, image count, iteration range, wall time, final loss, PSNR, and
Gaussian count at stage end; densification/prune events; peak VRAM; total time; output checkpoint
paths and sizes.

---

## 5. Orchestrator — `run_pipeline.py` (backend root)

```bash
python run_pipeline.py scenes/Bedroom2.zip            # full run
python run_pipeline.py --resume                       # continue where it stopped
python run_pipeline.py --from-step colmap             # re-run from a step onward
python run_pipeline.py --only-step filter_depth --force
python run_pipeline.py --keep-workspace                # don't clean up on success
```

- Calls each step **in-process** (import + `main(args)`), not by subprocess, so a failure gives a
  real traceback. Each step remains independently executable as a script.
- Reads/writes `pipeline_state.json`; `--resume` skips steps marked `ok` whose declared outputs
  still exist.
- Deletes `current_scene/` **only** after step 5 succeeds, and only without `--keep-workspace`.
  On any failure the workspace is left intact, and the failing step is named with the path to its
  log.
- Writes a final `pipeline_summary.txt` aggregating each step's headline metrics.
- Calls `Utilities.pipeline_paths.bootstrap()` once at startup. Note `bootstrap()` does **not**
  currently include `04_2DGS_refinment` or `06_mesh_refinment`; unchanged, since neither is in scope.

---

## 6. Changes to existing files

**New files** (no existing behavior touched):
`run_pipeline.py`, `Utilities/scene_io.py`, `Utilities/pipeline_step.py`,
`Utilities/step_extract.py`, `00_ingestion/step_filter_quality.py`,
`01_poses_refinment/step_colmap.py`, `02_depth_estimation/step_filter_depth.py`,
`02_depth_estimation/colmap_poses_to_da3.py` (substep 4a),
`02_depth_estimation/step_depth.py`, `03_2DGS_training/step_train.py`.

**Modified:**
| File | Change | Why |
| --- | --- | --- |
| `02_depth_estimation/` depth runner | new workspace-input path (wrapper preferred over editing `main()`) | steps 0-3 already did the loading/filtering; must use refined poses |
| `02_depth_estimation/depth_priors.py` | **additive only**: `poses_are_w2c: bool = False` on `estimate_depth_sequence`/`estimate_depth_sliding_window`, skipping `arcore_c2w_to_da3_w2c()` when set | lets substep 4a pass COLMAP world-to-camera poses without double-converting; default preserves existing behavior |
| `01_poses_refinment/export_keyframes.py` | expose `registered_names`, `sampson_rejects`, `filter_by_sampson`, `prune_model`, `_resize_one` as importable helpers; add the inverse of `prune_model` | steps 2/3/5 reuse them instead of duplicating |
| `00_ingestion/run_staged_filtering.py` | lift `write_stage` into `Utilities/scene_io.py`, import it back | single owner of scene writing |
| `backend/README.md`, stage READMEs | document the pipeline and the workspace contract | |

**Explicitly unchanged:** all merged 2DGS code under `03_2DGS_training/` (kept verbatim),
`quality_gate.py`'s thresholds and logic, DA3 inference internals,
`initialization.py`'s unprojection, `shared/schemas/transforms.schema.json` (frozen).

---

## 7. Decisions

**Resolved 2026-09-14:**
- ~~Iteration-coupled hyperparameters~~ — **approved**, scale the LR and densification schedules
  (§4 step 5). The hyperparameters meant by "keep current" are the rates, weights and thresholds,
  all of which stay untouched.
- ~~COLMAP-vs-ARCore poses for DA3~~ — **resolved**: use the COLMAP-refined poses, via the
  explicit conversion substep 4a.
- ~~Distortion refusal~~ — **deferred to testing**, not a design blocker. Fixing
  `transforms.json` into a `sparse/` folder with COLMAP has already been done successfully
  several times by hand, so the path is known to work; step 2 just needs to be exercised on a
  real capture to confirm the automated version behaves the same.

**Still open (non-blocking):**
1. **Input format** — is the pipeline input always images + `transforms.json`, or must step 0 also
   accept the mobile package (video + trajectory CSV) that `package_loader.py` handles?
2. **`current_scene/` concurrency** — a single fixed folder means one scene at a time. Fine for
   now; say so if parallel scenes are wanted (then it becomes `current_scene/<scene_id>/`).
3. **Workspace disk cost** — keeping every discard set plus `images_2/4/8` plus depth maps roughly
   doubles the raw capture footprint. Acceptable for debuggability?

## 8. Suggested sequencing

1. `scene_io.py` + `pipeline_step.py` + step 0, tested on `scenes/Bedroom2.zip`.
2. Step 1 (wrapper over the existing gate) — smallest real step, validates the discard contract.
3. Step 2 — riskiest external dependency (COLMAP binary, distortion refusal); surface failures early.
4. Step 3 — the genuinely new logic (track-based covisibility).
5. Substep 4a then step 4 — build and unit-test the pose conversion **first** (orthonormality +
   camera-centre agreement), then run depth and validate against the documented adjacent-view
   agreement (0.7-1.5 cm) before trusting the output.
6. Step 5 + orchestrator + resume testing.

Each step gets one small `test_*.py` in `backend/tests/` exercising it on a 3-5 frame fixture, plus
one resume test asserting that re-running a completed step is a no-op and that `--force` reproduces
the same output.
