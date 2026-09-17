#!/usr/bin/env python3
"""Step 2 -- COLMAP SfM: refine the ARCore poses and triangulate the scene.

In:  ``<workspace>/images/`` + ``transforms.json``.
Out: ``<workspace>/sparse/0/`` (TXT + ``points3D.ply``), rejects in
     ``colmap_discarded_images/``, diagnostics in ``colmap_diagnostics/``.

``convert_transforms_to_colmap.py`` is run as a subprocess, not imported: its
``main()`` calls ``parse_args()`` with no argv, so importing and calling it would
consume this script's arguments. It runs with ``--no_keyframes`` because
selecting keyframes is step 3's job and it now has better geometry to do it
with; ``--keep_database`` is then required, since the converter deletes the
database unless keyframes were exported and the Sampson check below needs it.

Two independent reasons a frame leaves here, and both are moves, never deletes:

* **unregistered** -- COLMAP could not place it at all.
* **badly posed** -- it registered, but the raw feature matches do not support
  the pose (``sampson_rejects``). Refused wholesale above
  ``MAX_SAMPSON_REJECT_FRAC``: wanting to drop that many frames means the model
  is wrong, not the frames.

    python 01_poses_refinment/step_colmap.py [--workspace DIR]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
from Utilities.pipeline_paths import bootstrap, subprocess_env

bootstrap()

from Utilities.pipeline_step import StepContext, is_done
from Utilities.scene_io import load_scene, merge_back, split_scene
from colmap_diagnostics import parse_points3D_txt
from densify_pointcloud import remove_outliers, voxel_dedup
from export_keyframes import (MAX_SAMPSON_REJECT_FRAC, filter_by_sampson,
                              prune_model, registered_names, sampson_rejects)
from scene.dataset_readers import storePly
from scene_extent import scene_extent, write_scene_size_txt
from select_keyframes_arcore import select_keyframes_arcore

DEFAULT_WORKSPACE = _backend_dir / "current_scene"
MANIFEST_DIRNAME = "00_ingestion"    # transforms.json lives here; images/ stays at workspace root
STAGE_DIRNAME = "01_poses_refinment"
DISCARD_DIRNAME = "colmap_discarded_images"
DIAGNOSTICS_DIRNAME = "colmap_diagnostics"
CONVERTER = _backend_dir / "01_poses_refinment" / "convert_transforms_to_colmap.py"

MAX_SAMPSON_PX = 2.0
# A run that registers less of the capture than this has failed, whatever the
# per-frame numbers say -- carrying on would train on a fragment of the room.
MIN_REGISTERED_FRAC = 0.7
MAX_ROTATION_DRIFT_DEG = 10.0
MAX_TRANSLATION_DRIFT_M = 0.25
VOXEL_SIZE_M = 0.01
OUTLIER_NEIGHBOURS, OUTLIER_STD_RATIO = 20, 2.0


def run_colmap(workspace: Path, diagnostics_dir: Path, ctx: StepContext,
               matcher: str = "sequential") -> None:
    cmd = [sys.executable, str(CONVERTER),
           "-s", str(workspace),
           "--transforms", f"{MANIFEST_DIRNAME}/transforms.json",
           "--database", f"{STAGE_DIRNAME}/colmap_database.db",
           "--matcher", matcher,
           "--refine_poses",
           "--no_keyframes",
           "--keep_database",
           "--diagnostics", str(diagnostics_dir)]
    ctx.note(f"$ {' '.join(cmd)}")
    with ctx.timer("colmap"):
        # Streamed, not captured: COLMAP is the long pole in the pipeline and a
        # silent half hour is indistinguishable from a hang.
        proc = subprocess.run(cmd, cwd=str(_backend_dir), env=subprocess_env())
    if proc.returncode != 0:
        raise RuntimeError(
            f"convert_transforms_to_colmap.py failed (exit {proc.returncode}). "
            f"See the output above and {diagnostics_dir / 'colmap_log.txt'}.")


def log_diagnostics(diagnostics_dir: Path, ctx: StepContext) -> None:
    """Fold the converter's own diagnostics into this step's stats."""
    report_path = diagnostics_dir / "diagnostics_report.json"
    if not report_path.exists():
        ctx.note("No diagnostics_report.json written; skipping diagnostic metrics")
        return
    report = json.loads(report_path.read_text())

    drift = report.get("drift", {})
    if drift:
        ctx.metric("drift", drift)
        rot, trans = drift.get("rotation_drift", {}), drift.get("translation_drift", {})
        cloud = drift.get("point_cloud", {})
        ctx.note(f"Drift vs ARCore over {drift.get('num_frames', 0)} frames: "
                 f"translation mean {trans.get('mean_m', 0):.3f} m / max {trans.get('max_m', 0):.3f} m, "
                 f"rotation mean {rot.get('mean_deg', 0):.2f} deg / max {rot.get('max_deg', 0):.2f} deg")
        ctx.note(f"Sparse cloud: {cloud.get('total_points', 0)} points, "
                 f"mean reproj err {cloud.get('mean_reprojection_error_px', 0):.3f} px, "
                 f"mean track length {cloud.get('mean_track_length', 0):.2f}")

        # COLMAP has no rotation prior, so this is the one failure mode bundle
        # adjustment cannot catch itself. Reported, not acted on: --fix_rotation_drift
        # is a judgement call for an operator looking at the plots.
        outliers = drift.get("rotation_outliers", [])
        ctx.metric("rotation_outliers", outliers)
        ctx.metric("rotation_outlier_count", len(outliers))
        if outliers:
            ctx.note(f"Rotation outliers: {len(outliers)} frame(s) beyond "
                     f"{rot.get('threshold_deg', 2.0)} deg, e.g. {outliers[:5]}")

    reproj = report.get("reprojection", {})
    if reproj:
        ctx.metric("reprojection", reproj)
    sampson = report.get("sampson", {})
    if sampson:
        ctx.metric("sampson", sampson)


def clean_point_cloud(sparse_dir: Path, ctx: StepContext) -> None:
    """Write ``sparse/0/points3D.ply`` -- the 2DGS loader's initialisation cloud.

    ``readColmapSceneInfo`` reads this file if it exists, so cleaning here is how
    the training init improves without touching the loader. The COLMAP text model
    is left as the source of truth; this is a derived product.
    """
    points = parse_points3D_txt(sparse_dir / "points3D.txt")
    if not points:
        raise RuntimeError(f"{sparse_dir / 'points3D.txt'} has no points; COLMAP triangulated nothing")

    xyz = np.array([p["xyz"] for p in points.values()], dtype=np.float64)
    rgb = np.array([p["rgb"] for p in points.values()], dtype=np.float64)
    nrm = np.zeros_like(xyz)
    before = len(xyz)

    xyz, rgb, nrm = voxel_dedup(xyz, rgb, nrm, VOXEL_SIZE_M)
    after_dedup = len(xyz)
    xyz, rgb, nrm = remove_outliers(xyz, rgb, nrm, OUTLIER_NEIGHBOURS, OUTLIER_STD_RATIO)

    storePly(str(sparse_dir / "points3D.ply"), xyz, rgb)
    ctx.metric("cloud_points", {"triangulated": before, "after_dedup": after_dedup,
                                "after_outlier_removal": len(xyz)})
    ctx.note(f"Point cloud: {before} -> {after_dedup} (voxel {VOXEL_SIZE_M} m) "
             f"-> {len(xyz)} (outlier removal), written to points3D.ply")


def colmap_step(workspace: Path, manifest_dir: Path, stage_dir: Path, diagnostics_dir: Path,
                 discard_dir: Path, ctx: StepContext, matcher: str = "sequential") -> None:
    scene = load_scene(manifest_dir, images_root=workspace)
    raw_names = set(scene.names)
    raw_total = len(raw_names)
    ctx.note(f"Input scene has {raw_total} frames")

    # Keyframe selection based on room extent & ARCore trajectory geometry
    selected_frames, kf_meta = select_keyframes_arcore(scene.frames)
    ctx.metric("arcore_keyframe_selection", kf_meta)
    extent = kf_meta.get("extent", {})
    area = extent.get("floor_area_m2", "?")
    ctx.note(f"ARCore room extent: {extent.get('room_dim_x', '?')}m x {extent.get('room_dim_z', '?')}m, "
             f"estimated floor area {area} m² (standoff {extent.get('standoff_m', 0.8)}m)")

    selected_names = {Path(f["file_path"]).name for f in selected_frames}
    non_kf_names = raw_names - selected_names
    if non_kf_names:
        ctx.note(f"Pre-COLMAP keyframe selection: keeping {len(selected_names)}/{raw_total} frames "
                 f"(budget [{kf_meta.get('budget', [0, 0])[0]}..{kf_meta.get('budget', [0, 0])[1]}])")
        non_kf_dir = stage_dir / "colmap_non_keyframes"
        split_scene(manifest_dir, non_kf_dir, non_kf_names, images_root=workspace)

    # Re-read scene containing only selected keyframes for COLMAP
    scene = load_scene(manifest_dir, images_root=workspace)
    all_names = set(scene.names)
    total = len(all_names)
    ctx.note(f"Running COLMAP on {total} frames")

    run_colmap(workspace, diagnostics_dir, ctx, matcher)

    sparse_dir = workspace / "sparse" / "0"
    db_path = workspace / STAGE_DIRNAME / "colmap_database.db"
    registered = registered_names(str(sparse_dir))
    unregistered = all_names - registered
    ctx.metric("total_in", total)
    ctx.metric("registered", len(registered))
    ctx.metric("unregistered", len(unregistered))
    ctx.note(f"Registered {len(registered)}/{total} "
             f"({100.0 * len(registered) / max(1, total):.1f}%), {len(unregistered)} unregistered")

    if len(registered) < MIN_REGISTERED_FRAC * total:
        raise RuntimeError(
            f"COLMAP registered only {len(registered)}/{total} frames "
            f"({100.0 * len(registered) / max(1, total):.1f}%), under the "
            f"{MIN_REGISTERED_FRAC:.0%} floor. The capture or the matcher is at fault; "
            f"inspect {diagnostics_dir}/ before continuing.")

    with ctx.timer("sampson"):
        rejects = sampson_rejects(str(db_path), str(sparse_dir), MAX_SAMPSON_PX)
    keep, refused = filter_by_sampson(registered, rejects)

    ctx.metric("sampson_rejects", len(rejects))
    ctx.metric("sampson_refused", refused)
    ctx.metric("sampson_threshold_px", MAX_SAMPSON_PX)
    if rejects:
        worst = sorted(rejects.items(), key=lambda kv: -kv[1])
        ctx.metric("sampson_worst", {n: round(e, 2) for n, e in worst[:10]})
    if refused:
        # Deliberately not fatal: the model may still be usable, and the operator
        # is better placed than this script to judge from the diagnostics.
        ctx.note(f"REFUSED to apply the Sampson filter: {len(rejects)}/{len(registered)} frames "
                 f"exceed {MAX_SAMPSON_PX} px, over the {MAX_SAMPSON_REJECT_FRAC:.0%} cap. "
                 "That many bad poses means the model is suspect, not the frames. "
                 f"Keeping every registered frame -- inspect {diagnostics_dir}/.")
    elif rejects:
        ctx.note(f"Sampson filter: dropping {len(rejects)} badly posed frame(s) above {MAX_SAMPSON_PX} px")
        for name, err in sorted(rejects.items(), key=lambda kv: -kv[1])[:8]:
            ctx.note(f"    {name}  {err:8.2f} px")

    log_diagnostics(diagnostics_dir, ctx)

    # Filter catastrophic bundle adjustment drift against ARCore VIO prior
    drift_outliers = set()
    per_frame_deltas_path = diagnostics_dir / "per_frame_deltas.csv"
    if per_frame_deltas_path.exists():
        import csv
        with open(per_frame_deltas_path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    rot_err = float(r["rotation_delta_deg"])
                    trans_err = float(r["translation_delta_m"])
                    if rot_err > MAX_ROTATION_DRIFT_DEG or trans_err > MAX_TRANSLATION_DRIFT_M:
                        drift_outliers.add(r["image"])
                except (ValueError, KeyError):
                    continue
    if drift_outliers:
        ctx.metric("pose_drift_outliers", len(drift_outliers))
        ctx.note(f"Pose drift filter: dropping {len(drift_outliers)} frame(s) with severe ARCore drift "
                 f"(>{MAX_ROTATION_DRIFT_DEG} deg or >{MAX_TRANSLATION_DRIFT_M*100:.0f} cm): {sorted(drift_outliers)}")
        keep = keep - drift_outliers

    drop = sorted(all_names - keep)
    kept, rejected = split_scene(manifest_dir, discard_dir, drop, images_root=workspace)
    if drop:
        with ctx.timer("prune_model"):
            # The model must name exactly what images/ holds: the 2DGS loader
            # opens every image listed and dies on the first one missing.
            prune_model(str(sparse_dir), set(drop))

    ctx.metric("kept", kept)
    ctx.metric("discarded", rejected)
    ctx.metric("kept_pct", round(100.0 * kept / max(1, total), 1))
    ctx.metric("discarded_unregistered", sorted(unregistered)[:50])
    ctx.note(f"Kept {kept}/{total} ({100.0 * kept / max(1, total):.1f}%), "
             f"moved {rejected} to {discard_dir}/")

    clean_point_cloud(sparse_dir, ctx)

    extent = scene_extent(sparse_dir)
    (stage_dir / "scene_extent.json").write_text(json.dumps(extent, indent=2))
    write_scene_size_txt(stage_dir, extent)
    ctx.metric("scene_extent", extent)
    aligned = extent["aligned"]
    x, y, z = aligned["point_cloud"]["size_m"]
    ctx.note(f"Wall-aligned scene size ({aligned['rotation_deg_about_up_axis']:.1f} deg "
             f"about up axis {aligned['up_axis']}): x={x:.2f} y={y:.2f} z={z:.2f} m, "
             f"written to scene_size.txt")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--matcher", choices=["sequential", "exhaustive"], default="sequential")
    parser.add_argument("--force", action="store_true", help="Re-run even if already complete")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    manifest_dir = workspace / MANIFEST_DIRNAME
    stage_dir = workspace / STAGE_DIRNAME
    diagnostics_dir = stage_dir / DIAGNOSTICS_DIRNAME
    discard_dir = stage_dir / DISCARD_DIRNAME
    outputs = [workspace / "sparse" / "0" / "images.txt", manifest_dir / "transforms.json"]
    if not args.force and is_done(workspace, "colmap", outputs):
        print("[colmap] already done, skipping (use --force to re-run)")
        return 0

    non_kf_dir = stage_dir / "colmap_non_keyframes"
    if args.force:
        restored_disc = merge_back(discard_dir, manifest_dir, images_root=workspace)
        restored_non_kf = merge_back(non_kf_dir, manifest_dir, images_root=workspace)
        restored = restored_disc + restored_non_kf
        if restored:
            print(f"[colmap] --force: restored {restored} previously moved frame(s)")

    with StepContext("colmap", workspace, artifacts_dir=diagnostics_dir) as ctx:
        colmap_step(workspace, manifest_dir, stage_dir, diagnostics_dir, discard_dir, ctx, args.matcher)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
