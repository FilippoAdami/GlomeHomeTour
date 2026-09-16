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
