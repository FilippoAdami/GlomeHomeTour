#!/usr/bin/env python3
"""Step 3 -- keyframe selection driven by real COLMAP geometry.

In:  ``<workspace>/images/`` + ``transforms.json`` + ``sparse/0/``.
Out: a reduced set in place; rejects in ``depth_discarded_images/`` and the
     matching model entries in ``depth_discarded_sparse/0/``.

This is the merged second and third level filter -- the parallax and keyframe
stages of the old ``run_staged_filtering.py`` -- but with COLMAP's triangulated
geometry replacing their estimates. The selector's accept/reject *logic* is
reused untouched (:class:`DynamicKeyframeSelector`); only its two inputs change:

* **scene depth** per frame is the median depth of that frame's own
  triangulated points, not optical-flow triangulation.
* **covisibility** is the true shared-track fraction
  ``|tracks(i) & tracks(j)| / min(|tracks(i)|, |tracks(j)|)``, not the
  frustum-overlap approximation against a reference plane. The frustum model is
  a proxy for exactly this number, and here the real one is available.

    python 02_depth_estimation/step_filter_depth.py [--workspace DIR]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
from Utilities.pipeline_paths import bootstrap

bootstrap()

from Utilities.pipeline_step import StepContext, is_done
from Utilities.scene_io import load_scene, merge_back, split_scene
from colmap_diagnostics import parse_images_txt, parse_points3D_txt
from export_keyframes import prune_model
from keyframe_selector import DynamicKeyframeSelector

DEFAULT_WORKSPACE = _backend_dir / "current_scene"
DISCARD_IMAGES = "depth_discarded_images"
DISCARD_SPARSE = "depth_discarded_sparse"

# Depth estimation is the most expensive step in the pipeline (~seconds/frame on
# GPU), so this stage exists to make it affordable: aim to hand step 4 a third
# to a half of what COLMAP registered, not nearly all of it.
RETENTION_MIN = 0.30
RETENTION_MAX = 0.50

# `for_2dgs_training` calibrated its overlap band against *frustum* covisibility.
# This stage scores |a & b| / min(|a|, |b|) over shared COLMAP tracks, which runs
# materially lower on the same geometry, so the inherited 0.50/0.80 band put the
# floor above the typical consecutive score. That matters because the redundancy
# rules in `_select_chain` are only enforceable while overlap is *above* the
# floor -- below it every frame is force-accepted to avoid re-opening a gap. The
# effect was a stage that kept 90.5% and fired `redundant_covisibility` 14 times.
MIN_COVIS = 0.25
MAX_COVIS = 0.60

# At a 1.47 m median scene depth an 8 cm baseline is a near-duplicate view; the
# inherited defaults were set for a stage that could not rely on triangulated
# geometry to tell it otherwise.
MIN_TRANSLATION_M = 0.15
MIN_ROTATION_DEG = 6.0


def colmap_tracks_and_depths(
    sparse_dir: Path, names: Sequence[str]
) -> tuple[list[set[int]], np.ndarray]:
    """Per-frame ``(track id set, median triangulated depth in metres)``.

    Frames COLMAP registered but gave no points fall back to the scene median
    depth, so one starved frame cannot drag the adaptive thresholds.
    """
    images = parse_images_txt(sparse_dir / "images.txt")
    points = parse_points3D_txt(sparse_dir / "points3D.txt")
    by_name = {Path(name).name: img for name, img in images.items()}

    tracks: list[set[int]] = []
    depths = np.full(len(names), np.nan)
    for i, name in enumerate(names):
        img = by_name.get(name)
        if img is None:
            tracks.append(set())
            continue
        ids = {int(pid) for pid in img["p3d_ids"] if pid > 0 and int(pid) in points}
        tracks.append(ids)
        if ids:
            xyz = np.array([points[pid]["xyz"] for pid in ids])
            # z in camera coordinates == depth along the optical axis (OpenCV).
            z = (xyz @ img["R_w2c"].T + img["t_w2c"])[:, 2]
            z = z[z > 0]
            if len(z):
                depths[i] = float(np.median(z))

    if np.all(np.isnan(depths)):
        raise RuntimeError("No frame has triangulated points; sparse/0/ is empty or unmatched")
    depths[np.isnan(depths)] = float(np.nanmedian(depths))
    return tracks, depths


class TrackCovisibilitySelector(DynamicKeyframeSelector):
    """:class:`DynamicKeyframeSelector` with shared-track covisibility.

    Subclassed rather than edited: the walk, the thresholds, the coverage-gap
    backtracking and the min/max re-walk all stay exactly as tuned, and only the
    one measurement they consult is swapped for the exact version.
    """

    def __init__(self, tracks: Sequence[set[int]], **kwargs):
        super().__init__(**kwargs)
        self._tracks = list(tracks)

    def _mutual_covisibility(self, i, j, keyframes, intrinsics, scene_depths=None) -> float:
        a, b = self._tracks[i], self._tracks[j]
        if not a or not b:
            return 0.0
        return len(a & b) / min(len(a), len(b))


def filter_depth(
    workspace: Path,
    ctx: StepContext,
    min_keyframes: Optional[int] = None,
    max_keyframes: Optional[int] = None,
) -> None:
    sparse_dir = workspace / "sparse" / "0"
    scene = load_scene(workspace)
    keyframes = scene.keyframes()
    names = scene.names
    total = len(names)

    # This stage prunes sparse/0 in place, and `merge_back` only restores images
    # and transforms.json -- it cannot un-prune the model. So a second --force
    # run would select against a model already missing the frames it restored,
    # and quietly emit a scene whose model covers fewer cameras than its own
    # transforms.json (step 5 then trains on the smaller set without erroring).
    # Step 2 owns sparse/0; if it no longer covers every frame, re-run it.
    modelled = {Path(img["name"]).name
                for img in parse_images_txt(sparse_dir / "images.txt").values()}
    missing = [n for n in names if Path(n).name not in modelled]
    if missing:
        raise SystemExit(
            f"[filter_depth] sparse/0 covers {len(modelled)} frames but transforms.json has "
            f"{total}; {len(missing)} have no model entry (e.g. {missing[0]}). "
            "sparse/0 was pruned by an earlier run and cannot be un-pruned here -- "
            "re-run step 2 (`01_poses_refinment/step_colmap.py --force`) first.")

    # Best-effort band, met by re-walking at a tighter/looser overlap threshold
    # (never by subsampling a chain built for overlap). Coverage still outranks
    # compactness: the walk refuses to go below `min_covisibility`, so a fast pan
    # can legitimately overshoot RETENTION_MAX.
    if min_keyframes is None:
        min_keyframes = int(total * RETENTION_MIN)
    if max_keyframes is None:
        max_keyframes = int(total * RETENTION_MAX)

    ctx.note(f"Selecting from {total} frames against {sparse_dir} "
             f"(target {min_keyframes}-{max_keyframes})")

    with ctx.timer("parse_model"):
        tracks, depths = colmap_tracks_and_depths(sparse_dir, names)

    ctx.metric("scene_depth_m", {
        "median": round(float(np.median(depths)), 3),
        "p10": round(float(np.percentile(depths, 10)), 3),
        "p90": round(float(np.percentile(depths, 90)), 3),
    })
    ctx.note(f"Scene depth: median {np.median(depths):.2f} m "
             f"(p10 {np.percentile(depths, 10):.2f}, p90 {np.percentile(depths, 90):.2f})")
    ctx.note(f"Tracks per frame: median {int(np.median([len(t) for t in tracks]))}")

    selector = TrackCovisibilitySelector(
        tracks,
        **vars(DynamicKeyframeSelector.for_2dgs_training(
            min_translation_m=MIN_TRANSLATION_M,
            min_rotation_deg=MIN_ROTATION_DEG,
            min_covisibility=MIN_COVIS,
            max_covisibility=MAX_COVIS,
        )),
    )
    ctx.metric("thresholds", {
        "min_translation_m": selector.min_translation_m,
        "min_rotation_deg": selector.min_rotation_deg,
        "min_covisibility": selector.min_covisibility,
        "max_covisibility": selector.max_covisibility,
    })

    with ctx.timer("select"):
        result = selector.select_keyframes(
            keyframes, scene.intrinsics,
            min_keyframes=min_keyframes, max_keyframes=max_keyframes,
            scene_depths=depths,
        )

    selected = set(result.selected_indices)
    reject_names = [names[i] for i in range(total) if i not in selected]

    # Covisibility actually held between consecutive kept frames -- the number
    # that proves coverage survived the cull.
    chain = result.selected_indices
    pair_covis = [selector._mutual_covisibility(chain[k], chain[k + 1], keyframes, scene.intrinsics)
                  for k in range(len(chain) - 1)]
    if pair_covis:
        ctx.metric("consecutive_covisibility", {
            "min": round(float(np.min(pair_covis)), 3),
            "mean": round(float(np.mean(pair_covis)), 3),
            "median": round(float(np.median(pair_covis)), 3),
            "below_floor": int(np.sum(np.array(pair_covis) < selector.min_covisibility)),
        })
        ctx.note(f"Consecutive covisibility: min {np.min(pair_covis):.2f}, "
                 f"mean {np.mean(pair_covis):.2f}, median {np.median(pair_covis):.2f}, "
                 f"{int(np.sum(np.array(pair_covis) < selector.min_covisibility))} below floor")

    # Per-frame: the covisibility each dropped frame had against the anchor that
    # was live when it was judged. The selector reports its rules as aggregate
    # counts only, so the rule name is in `reasons`, the margin is here.
    anchor_for = {}
    last = chain[0]
    for i in range(total):
        if i in selected:
            last = i
        elif i > chain[0]:
            anchor_for[names[i]] = round(
                selector._mutual_covisibility(i, last, keyframes, scene.intrinsics), 3)
    ctx.metric("discarded_covisibility_vs_anchor", anchor_for)
    ctx.metric("reasons", result.reasons)
    ctx.note(f"Selector reasons: {result.reasons}")

    points_before = len(parse_points3D_txt(sparse_dir / "points3D.txt"))

    kept, rejected = split_scene(workspace, workspace / DISCARD_IMAGES, reject_names)
    if reject_names:
        with ctx.timer("prune_model"):
            # The model split mirrors the image split: copy, then delete the
            # complement from each half, so the discard folder is a loadable
            # mini-model rather than a list of names.
            discard_sparse = workspace / DISCARD_SPARSE / "0"
            shutil.rmtree(discard_sparse.parent, ignore_errors=True)
            shutil.copytree(sparse_dir, discard_sparse)
            prune_model(str(discard_sparse), set(names) - set(reject_names))
            prune_model(str(sparse_dir), set(reject_names))

    points_after = len(parse_points3D_txt(sparse_dir / "points3D.txt"))
    ctx.metric("total_in", total)
    ctx.metric("kept", kept)
    ctx.metric("discarded", rejected)
    ctx.metric("kept_pct", round(100.0 * kept / max(1, total), 1))
    ctx.metric("sparse_points_before", points_before)
    ctx.metric("sparse_points_after", points_after)
    ctx.metric("sparse_points_orphaned", points_before - points_after)
    ctx.note(f"Kept {kept}/{total} ({100.0 * kept / max(1, total):.1f}%), discarded {rejected}")
    ctx.note(f"Sparse points {points_before} -> {points_after} "
             f"({points_before - points_after} orphaned by the prune)")

    # Write human-inspectable markdown summary report
    summary_md = [
        "# Track Covisibility Selection Summary (Step 3)",
        "",
        f"**Workspace:** `{workspace}`  ",
        f"**Selected Keyframes:** {kept} / {total} ({100.0 * kept / max(1, total):.1f}%)  ",
        f"**Discarded Frames:** {rejected}  ",
        "",
        "## Scene Depth & Track Statistics",
        "",
        f"- **Scene Depth (Triangulated Points):** Median = {np.median(depths):.2f}m (p10 = {np.percentile(depths, 10):.2f}m, p90 = {np.percentile(depths, 90):.2f}m)",
        f"- **Shared Tracks Per Frame:** Median = {int(np.median([len(t) for t in tracks]))}",
        f"- **Sparse 3D Landmarks:** {points_before} -> {points_after} points ({points_before - points_after} orphaned)",
        "",
        "## Selection Parameters & Thresholds",
        "",
        f"- **Min Translation Baseline:** `{selector.min_translation_m:.2f} m`",
        f"- **Min Rotation:** `{selector.min_rotation_deg:.1f}°`",
        f"- **Target Covisibility Band:** `[{selector.min_covisibility:.2f}, {selector.max_covisibility:.2f}]`",
        "",
        "## Consecutive Coverage & Quality",
        "",
    ]
    if pair_covis:
        summary_md.extend([
            f"- **Consecutive Covisibility:** Min = {np.min(pair_covis):.2f}, Mean = {np.mean(pair_covis):.2f}, Median = {np.median(pair_covis):.2f}",
            f"- **Pairs Below Minimum Floor:** {int(np.sum(np.array(pair_covis) < selector.min_covisibility))}",
            "",
        ])
    summary_md.extend([
        "## Selector Decision Breakdown",
        "",
        "| Decision Reason | Count | Description |",
        "| :--- | :--- | :--- |",
    ])
    reason_descriptions = {
        "initial_frame": "Starting reference frame for trajectory chain",
        "sufficient_motion": "Exceeded translation/rotation threshold with good covisibility",
        "coverage_gap_prevention": "Forced keep to prevent covisibility gap dropping below floor",
        "redundant_motion": "Too close in translation/rotation to previous anchor",
        "redundant_covisibility": "Shared track fraction above max covisibility threshold (redundant view)",
        "unavoidable_gap": "Covisibility dropped below floor due to rapid camera movement / turn",
    }
    for r_key, r_count in result.reasons.items():
        desc = reason_descriptions.get(r_key, "")
        summary_md.append(f"| `{r_key}` | **{r_count}** | {desc} |")
    summary_md.append("")

    summary_file = workspace / "filter_depth_summary.md"
    summary_file.write_text("\n".join(summary_md), encoding="utf-8")
    ctx.note(f"Human-inspectable summary written to {summary_file}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--min-keyframes", type=int, default=None)
    parser.add_argument("--max-keyframes", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Re-run even if already complete")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    if not args.force and is_done(workspace, "filter_depth", [workspace / "transforms.json"]):
        print("[filter_depth] already done, skipping (use --force to re-run)")
        return 0

    if args.force:
        restored = merge_back(workspace / DISCARD_IMAGES, workspace)
        if restored:
            # The pruned model cannot be un-pruned in place; step 2 owns sparse/0/.
            print(f"[filter_depth] --force: restored {restored} frame(s). "
                  "Re-run step 2 as well if sparse/0/ was already pruned.")

    with StepContext("filter_depth", workspace) as ctx:
        filter_depth(workspace, ctx, args.min_keyframes, args.max_keyframes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
