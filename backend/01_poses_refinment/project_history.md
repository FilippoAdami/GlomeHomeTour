# 01_poses_refinment — project history

## 2026-09-14: Replaced in-house SfM refiner with the 2DGS project's COLMAP pose pipeline
Outcome: worked (merge only, not yet run end-to-end) — `sfm_refinement.py`
(`HybridSfMRefiner`, scipy `least_squares` bundle adjustment over SIFT/ORB matches) was deleted
and superseded by the COLMAP-driven pipeline copied in from the standalone 2DGS project:
`convert_transforms_to_colmap.py` (feature_extractor -> sequential_matcher -> point_triangulator
with ARCore poses held fixed, optional pose-prior BA), plus `colmap_diagnostics.py`,
`export_keyframes.py`, `select_keyframes.py`, `densify_pointcloud.py`,
`visualize_camera_path.py`, `convert.py` and two self-check scripts.

Rationale: the merged version triangulates against known-good ARCore poses rather than
re-estimating them, and ships drift / multi-view reprojection / Sampson-epipolar diagnostics
that the in-house refiner had no equivalent for.

Fallout handled: `HybridSfMRefiner` was imported by `00_ingestion/__init__.py`,
`00_ingestion/inspect_scan.py` and `tests/test_ingestion.py`; those imports and their call
sites were removed. Pre-merge copy of the deleted module kept in the merge backup (see
`03_2DGS_training/project_history.md`).

Known gap, not addressed: this stage requires a system COLMAP binary (4.3.0.dev0, HIP build) —
it is not a pip dependency, so a fresh machine needs it installed separately.
`convert_transforms_to_colmap.py` errors out by design on a `transforms.json` declaring
non-zero distortion; stage 1 currently propagates the capture's distortion coefficients
unchanged, so that handoff is untested on a real distorted capture.

## 2026-09-14: wrapped as pipeline step 2 (`step_colmap.py`)
`convert_transforms_to_colmap.py` is driven as a **subprocess**, not imported: its
`main()` calls `parse_args()` with no argv and would eat the caller's arguments.
Run with `--no_keyframes` (keyframe selection is step 3's job now, with better
geometry) plus `--keep_database`, which is then mandatory — the converter deletes
the database unless keyframes were exported, and the Sampson check needs it.

Outcome: worked, after one real bug. The subprocess died instantly with
`ModuleNotFoundError: No module named 'scene'` — a fresh interpreter inherits
none of the parent's `pipeline_paths.bootstrap()` sys.path. Fixed at the root by
adding `stage_paths()` / `subprocess_env()` to `Utilities/pipeline_paths.py` and
passing `PYTHONPATH` through; step 5 subprocesses `train.py`, which imports
`scene` the same way, so it needed the shared fix rather than a local one.

Frames leave for two independent reasons, both as moves to
`colmap_discarded_images/`: unregistered, or Sampson-rejected above 2.0 px. The
`MAX_SAMPSON_REJECT_FRAC = 0.2` refusal is honoured — past that the model is
suspect, not the frames, so it logs loudly and drops nothing. `prune_model()`
takes the **delete** set (confirmed at `export_keyframes.py:169`), and the model
must be pruned to match `images/` or the 2DGS loader dies opening a missing file.
Also writes a cleaned `sparse/0/points3D.ply` (voxel dedup + statistical outlier
removal), which is what `readColmapSceneInfo` initialises from.

## 2026-09-17: Pre-COLMAP ARCore geometry keyframing, spatial matching & rotation clamping
1. **Pre-COLMAP keyframe selection (`select_keyframes_arcore.py`):**
   - Estimates room horizontal span and dilated convex hull with a 0.8m standoff buffer (each horizontal axis extends ~1.6m beyond trajectory span).
   - Derives optimal keyframe budget $N = 50 + (8..11) \cdot A_{\text{floor}}$ (e.g. ~220-250 frames for Bedroom2).
   - Non-keyframe frames are staged into `colmap_non_keyframes/` before COLMAP runs. Reduces runtime from ~14 minutes on 1,413 frames to ~1.5 minutes on ~230 frames.
2. **Loop closure with `spatial_matcher`:**
   - Pose priors are inserted into the COLMAP database before matching.
   - `spatial_matcher` (max distance 2.5m, ignore Z) runs alongside `sequential_matcher` (overlap 5), matching views across room passes without needing an external vocabulary tree.
3. **Rigid rotation clamping (`convert_transforms_to_colmap.py`):**
   - Removed the 10% outlier cap on `--fix_rotation_drift`: all frames drifting $> 2.0^\circ$ are held at ARCore IMU/gravity orientation.
   - Re-projection of $t_{w2c} = -R_{w2c} C$ strictly preserves the refined camera center $C$ while enforcing ARCore orientation during re-triangulation.
4. **SIFT feature extraction:**
   - Capped features at 8,192 and image dimension at 1600.

## 2026-09-17: Pose Drift Filtering Against ARCore VIO Prior
1. **Pose Drift Gating in `step_colmap.py`:**
   - Evaluated rotation and translation delta between COLMAP refined poses and ARCore VIO initialization.
   - Identified catastrophic bundle adjustment divergence on 14 keyframes (rotation error $14.4^\circ$ to $56.2^\circ$, translation drift $30\text{ cm}$ to $100\text{ cm}$) caused by local minima on textureless walls and repetitive wooden rafters.
   - Added `MAX_ROTATION_DRIFT_DEG = 10.0` and `MAX_TRANSLATION_DRIFT_M = 0.25` gating in `colmap_step`: frames exceeding tolerance are pruned from `sparse/0/` and moved to `colmap_discarded_images/`.
   - Result: eliminates unprojected stray planes and false multi-layer geometry downstream.
