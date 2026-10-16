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

import time
from Utilities.pipeline_step import StepContext, is_done, update_pipeline_stats
from Utilities.scene_io import load_scene, merge_back, split_scene
from colmap_diagnostics import parse_points3D_txt
from densify_pointcloud import remove_outliers, voxel_dedup
from export_keyframes import (MAX_SAMPSON_REJECT_FRAC, filter_by_sampson,
                              prune_model, registered_names, sampson_rejects)
from scene.dataset_readers import storePly
from scene_extent import scene_extent, write_scene_size_txt
from select_keyframes_arcore import select_keyframes_arcore
import floor_count
import topdown_view

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
MIN_FEATURE_PRUNE = 30          # Catastrophic: <30 registered 3D points are pruned
MIN_FEATURE_LOCK = 100          # Weak: 30..99 registered 3D points kept, but poses locked to ARCore VIO
MAX_FEATURE_REJECT_FRAC = 0.10  # Refuse wholesale deletion if >10% of frames have <30 pts


def count_registered_features(sparse_dir: Path) -> dict[str, int]:
    """Parse sparse/0/images.txt and return image_name -> count of triangulated 3D points."""
    images_txt = sparse_dir / "images.txt"
    if not images_txt.exists():
        return {}
    counts = {}
    lines = [l for l in images_txt.read_text().splitlines() if not l.startswith("#")]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        i += 1
        if len(parts) < 10:
            continue
        name = parts[9]
        obs_parts = lines[i].split() if i < len(lines) else []
        i += 1
        reg_count = 0
        if obs_parts:
            for k in range(2, len(obs_parts), 3):
                if int(obs_parts[k]) != -1:
                    reg_count += 1
        counts[Path(name).name] = reg_count
    return counts


def lock_poses_to_arcore(sparse_dir: Path, manifest_dir: Path, lock_names: set[str], ctx: StepContext) -> int:
    """Reset poses of lock_names in sparse/0/ to ARCore VIO poses, preventing BA distortion."""
    if not lock_names:
        return 0
    transforms_path = manifest_dir / "transforms.json"
    if not transforms_path.exists():
        return 0
    transforms = json.loads(transforms_path.read_text())

    from convert_transforms_to_colmap import OPENGL_TO_OPENCV
    from scene.colmap_loader import rotmat2qvec

    name_to_pose = {}
    for f in transforms.get("frames", []):
        base = Path(f["file_path"]).name
        c2w = np.array(f["transform_matrix"]) @ OPENGL_TO_OPENCV
        w2c = np.linalg.inv(c2w)
        qvec = rotmat2qvec(w2c[:3, :3])
        tvec = w2c[:3, 3]
        name_to_pose[base] = (qvec, tvec)

    images_txt = sparse_dir / "images.txt"
    lines = images_txt.read_text().splitlines()
    new_lines = []
    i = 0
    locked_count = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#") or not line.strip():
            new_lines.append(line)
            i += 1
            continue
        parts = line.split()
        if len(parts) >= 10:
            name = parts[9]
            base = Path(name).name
            obs_line = lines[i + 1] if i + 1 < len(lines) else ""
            if (name in lock_names or base in lock_names) and base in name_to_pose:
                qvec, tvec = name_to_pose[base]
                image_id = parts[0]
                camera_id = parts[8]
                new_line = (f"{image_id} {qvec[0]:.8f} {qvec[1]:.8f} {qvec[2]:.8f} {qvec[3]:.8f} "
                            f"{tvec[0]:.8f} {tvec[1]:.8f} {tvec[2]:.8f} {camera_id} {name}")
                new_lines.append(new_line)
                new_lines.append(obs_line)
                locked_count += 1
            else:
                new_lines.append(line)
                new_lines.append(obs_line)
            i += 2
        else:
            new_lines.append(line)
            i += 1

    images_txt.write_text("\n".join(new_lines) + "\n")

    # Remove existing .bin files before model_converter so it converts the updated .txt -> .bin
    for bin_file in sparse_dir.glob("*.bin"):
        bin_file.unlink()

    cmd = ["colmap", "model_converter",
           "--input_path", str(sparse_dir),
           "--output_path", str(sparse_dir),
           "--output_type", "BIN"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, env=subprocess_env({"QT_QPA_PLATFORM": "xcb"}))
    ctx.note(f"Locked {locked_count} weak frame(s) (30..99 pts) to original ARCore VIO poses in sparse/0/")
    return locked_count


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
    all_names = set(scene.names)
    total = len(all_names)
    ctx.note(f"Running COLMAP on all {total} frames from capture")

    # 1. COLMAP processing (feature matching, manual model, triangulation, pose prior mapper)
    t0_colmap = time.time()
    run_colmap(workspace, diagnostics_dir, ctx, matcher)
    t_colmap = time.time() - t0_colmap
    update_pipeline_stats(workspace, "colmap_processing", t_colmap, total)

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

    # 3. Sampson error gating & model pruning
    t0_sampson = time.time()
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
        ctx.note(f"REFUSED to apply the Sampson filter: {len(rejects)}/{len(registered)} frames "
                 f"exceed {MAX_SAMPSON_PX} px, over the {MAX_SAMPSON_REJECT_FRAC:.0%} cap. "
                 "That many bad poses means the model is suspect, not the frames. "
                 f"Keeping every registered frame -- inspect {diagnostics_dir}/.")
    elif rejects:
        ctx.note(f"Sampson filter: dropping {len(rejects)} badly posed frame(s) above {MAX_SAMPSON_PX} px")
        for name, err in sorted(rejects.items(), key=lambda kv: -kv[1])[:8]:
            ctx.note(f"    {name}  {err:8.2f} px")

    log_diagnostics(diagnostics_dir, ctx)

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
                    if rot_err > 30.0 or trans_err > 0.80:
                        drift_outliers.add(r["image"])
                except (ValueError, KeyError):
                    continue
    if drift_outliers:
        ctx.metric("pose_drift_outliers", len(drift_outliers))
        ctx.note(f"Pose drift filter: dropping {len(drift_outliers)} frame(s) with catastrophic tracking glitch "
                 f"(>30 deg or >80 cm): {sorted(drift_outliers)}")
        keep = keep - drift_outliers

    # Feature count gating (prune catastrophic < 30 pts; lock weak 30..99 pts to ARCore VIO)
    feature_counts = count_registered_features(sparse_dir)
    catastrophic_frames = set()
    if feature_counts:
        catastrophic_frames = {n for n, c in feature_counts.items() if c < MIN_FEATURE_PRUNE and n in keep}
        if len(catastrophic_frames) / max(1, len(registered)) > MAX_FEATURE_REJECT_FRAC:
            ctx.note(f"REFUSED to drop frames with <{MIN_FEATURE_PRUNE} features: "
                     f"{len(catastrophic_frames)}/{len(registered)} frames exceed {MAX_FEATURE_REJECT_FRAC:.0%} cap.")
            catastrophic_frames = set()
        elif catastrophic_frames:
            ctx.note(f"Feature count filter: dropping {len(catastrophic_frames)} frame(s) with <{MIN_FEATURE_PRUNE} "
                     f"triangulated points: {sorted(catastrophic_frames)}")
            keep = keep - catastrophic_frames

    drop = sorted(all_names - keep)
    kept, rejected = split_scene(manifest_dir, discard_dir, drop, images_root=workspace)
    if drop:
        with ctx.timer("prune_model"):
            prune_model(str(sparse_dir), set(drop))

    # Lock surviving weak frames (MIN_FEATURE_PRUNE <= count < MIN_FEATURE_LOCK) back to ARCore VIO
    surviving_weak = {n for n, c in feature_counts.items() if MIN_FEATURE_PRUNE <= c < MIN_FEATURE_LOCK and n in keep}
    locked_count = 0
    if surviving_weak:
        locked_count = lock_poses_to_arcore(sparse_dir, manifest_dir, surviving_weak, ctx)

    t_sampson = time.time() - t0_sampson
    update_pipeline_stats(workspace, "sampson_error_gating_and_model_pruning", t_sampson, total)

    ctx.metric("kept", kept)
    ctx.metric("discarded", rejected)
    ctx.metric("kept_pct", round(100.0 * kept / max(1, total), 1))
    ctx.metric("discarded_unregistered", sorted(unregistered)[:50])
    if feature_counts:
        ctx.metric("feature_counts_median", float(np.median(list(feature_counts.values()))))
    ctx.metric("catastrophic_feature_frames", len(catastrophic_frames))
    ctx.metric("weak_feature_frames_locked", locked_count)
    ctx.note(f"Kept {kept}/{total} ({100.0 * kept / max(1, total):.1f}%), "
             f"moved {rejected} to {discard_dir}/")

    # 4. Point cloud refinement
    t0_cloud = time.time()
    clean_point_cloud(sparse_dir, ctx)
    t_cloud = time.time() - t0_cloud
    update_pipeline_stats(workspace, "pointcloud_refinement", t_cloud, kept)

    # 4. Scene alignment & diagnostics
    t0_align = time.time()
    extent = scene_extent(sparse_dir, transforms_path=manifest_dir / "transforms.json")
    (stage_dir / "scene_extent.json").write_text(json.dumps(extent, indent=2))
    write_scene_size_txt(workspace=stage_dir, extent=extent)
    ctx.metric("scene_extent", extent)
    aligned = extent["aligned"]
    x, y, z = aligned["point_cloud"]["size_m"]
    mode = aligned.get("alignment_mode", "wall")
    ctx.note(f"{mode.capitalize()}-aligned scene size ({aligned['rotation_deg_about_up_axis']:.1f} deg "
             f"about up axis {aligned['up_axis']}): x={x:.2f} y={y:.2f} z={z:.2f} m, "
             f"written to scene_size.txt")

    with ctx.timer("room_diagnostics"):
        try:
            floor_count.main(["--workspace", str(workspace)])
            topdown_view.main(["--workspace", str(workspace), "--transforms", str(manifest_dir / "transforms.json")])
            moved = workspace / "scene_extent_topdown.png"
            if moved.exists():
                moved.rename(stage_dir / "scene_extent_topdown.png")
        except Exception as e:
            ctx.note(f"room diagnostics failed (non-fatal): {e}")
    t_align = time.time() - t0_align
    update_pipeline_stats(workspace, "scene_alignment_and_diagnostics", t_align, kept)
    ctx.metric("frames_processed", total)


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
