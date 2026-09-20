#!/usr/bin/env python3
"""Render a side (elevation) view of the wall-aligned COLMAP point cloud to
eyeball how many floors the capture spans, plus a height histogram to make
the floor bands numeric instead of just visual.

Collapses one of the *horizontal* aligned axes (X, picked arbitrarily -- the
alignment only fixes orientation about the up axis, not which horizontal
axis is "wide") and keeps the up axis (Y) on the vertical plot axis. Floor
count itself comes from collapsing further, to a 1D histogram of camera
(acquisition point) heights only -- point-cloud height is dominated by
wall/ceiling clutter at every floor, but camera height directly reflects
where the person walked. A single floor gives one mode in that histogram;
each extra floor is a second mode offset by roughly a floor-to-floor rise
(see MIN_FLOOR_GAP_M).

Also appends a ``floors: N`` row to scene_extent.py's ``<workspace>/scene_size.txt``.

Usage:
    python 01_poses_refinment/floor_count.py [--workspace DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
from Utilities.pipeline_paths import bootstrap

bootstrap()

from scene.colmap_loader import read_extrinsics_text
from scene.dataset_readers import fetchPly

from scene_extent import (
    DEFAULT_WORKSPACE,
    STAGE_DIRNAME,
    PCT_LO,
    PCT_HI,
    horizontal_align_deg,
    rotate_horizontal,
)
from floor_bands import count_floor_bands as _count_floor_bands

# Camera-center height histogram is what actually separates floors -- point
# cloud density is dominated by walls/floor/ceiling clutter at every level,
# but each floor's walk has one band of camera heights. Bin width tuned for
# typical ~2.4-3m room height: coarser and a two-floor scan with a short
# stairwell run merges into one bin.
HEIGHT_BIN_M = 0.15


def _plot_side(ax, points: np.ndarray, centers: np.ndarray, title: str, up_axis: int = 1) -> None:
    horiz_axis = [i for i in range(3) if i != up_axis][0]
    ax.scatter(points[:, horiz_axis], points[:, up_axis], s=0.5, alpha=0.15, label="points")
    ax.scatter(centers[:, horiz_axis], centers[:, up_axis], s=12, c="red", label="camera path")
    ax.set_title(title)
    ax.set_xlabel(f"axis {horiz_axis}")
    ax.set_ylabel(f"up axis {up_axis} (height, m)")
    ax.set_ylim(bottom=0)
    ax.set_aspect("equal")
    ax.legend()


def _plot_height_hist(ax, centers: np.ndarray, up_axis: int = 1) -> None:
    heights = centers[:, up_axis]
    lo, hi = heights.min(), heights.max()
    bins = max(1, int(np.ceil((hi - lo) / HEIGHT_BIN_M))) + 1
    ax.hist(heights, bins=bins, orientation="horizontal")
    ax.set_title("camera height histogram")
    ax.set_xlabel("count")
    ax.set_ylabel(f"up axis {up_axis} (height, m)")


# Trim window for the camera-height stats below: same idea as
# scene_extent.PCT_LO/PCT_HI but tighter -- a handful of cameras at
# doorways/stairwells can sit well off the main walking-height band and
# would otherwise dominate the "3 lowest/3 highest" spread.
HEIGHT_STAT_PCT_LO, HEIGHT_STAT_PCT_HI = 5.0, 95.0


def camera_height_stats(centers: np.ndarray, up_axis: int = 1) -> dict:
    heights = centers[:, up_axis]
    lo, hi = np.percentile(heights, [HEIGHT_STAT_PCT_LO, HEIGHT_STAT_PCT_HI])
    trimmed = np.sort(heights[(heights >= lo) & (heights <= hi)])
    lowest3, highest3 = trimmed[:3], trimmed[-3:]
    return {
        "mean": float(trimmed.mean()),
        "median": float(np.median(trimmed)),
        "lowest3_mean": float(lowest3.mean()),
        "highest3_mean": float(highest3.mean()),
        "abs_diff_3lowest_3highest": float(abs(highest3.mean() - lowest3.mean())),
        "n_kept": int(trimmed.size),
        "n_total": int(heights.size),
    }


def count_floor_bands(centers: np.ndarray, up_axis: int = 1) -> int:
    """See floor_bands.count_floor_bands -- the shared threshold-crossing rule,
    applied to this stage's (COLMAP-triangulated) camera centers."""
    return _count_floor_bands(centers[:, up_axis])


def write_floors_txt(workspace: Path, n_floors: int) -> Path:
    """Append/update the ``floors:`` row in scene_extent.py's scene_size.txt."""
    out_path = workspace / STAGE_DIRNAME / "scene_size.txt"
    lines = [l for l in out_path.read_text().splitlines() if l.strip()] if out_path.exists() else []
    lines = [l for l in lines if not l.startswith("floors:")]
    lines.append(f"floors: {n_floors}")
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--up-axis", type=int, default=1)
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    sparse_dir = workspace / "sparse" / "0"
    cloud = fetchPly(str(sparse_dir / "points3D.ply"))
    cameras = read_extrinsics_text(str(sparse_dir / "images.txt"))
    centers = np.array([-e.qvec2rotmat().T @ e.tvec for e in cameras.values()])

    lo, hi = np.percentile(cloud.points, PCT_LO, axis=0), np.percentile(cloud.points, PCT_HI, axis=0)
    points = cloud.points[np.all((cloud.points >= lo) & (cloud.points <= hi), axis=1)]

    align_deg = horizontal_align_deg(points, args.up_axis)
    points_aligned = rotate_horizontal(points, align_deg, args.up_axis)
    centers_aligned = rotate_horizontal(centers, align_deg, args.up_axis)

    # Floor (min aligned point height) at 0 for the debug view -- ylim bottom
    # is pinned to 0 in _plot_side, so without this shift a scan whose floor
    # sits below the origin gets silently clipped off-plot. Stats below are
    # reported in this same shifted frame so they read directly off the plot.
    floor_shift = points_aligned[:, args.up_axis].min()
    points_aligned = points_aligned.copy()
    centers_aligned = centers_aligned.copy()
    points_aligned[:, args.up_axis] -= floor_shift
    centers_aligned[:, args.up_axis] -= floor_shift

    n_floors = count_floor_bands(centers_aligned, args.up_axis)
    stats = camera_height_stats(centers_aligned, args.up_axis)
    print(f"[floor_count] camera height stats ({stats['n_kept']}/{stats['n_total']} kept, "
          f"{HEIGHT_STAT_PCT_LO}-{HEIGHT_STAT_PCT_HI} pct trim):")
    print(f"[floor_count]   mean={stats['mean']:.3f}m  median={stats['median']:.3f}m")
    print(f"[floor_count]   3-lowest mean={stats['lowest3_mean']:.3f}m  "
          f"3-highest mean={stats['highest3_mean']:.3f}m  "
          f"abs diff={stats['abs_diff_3lowest_3highest']:.3f}m")

    fig, (ax_side, ax_hist) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={"width_ratios": [3, 1]})
    _plot_side(ax_side, points_aligned, centers_aligned,
               f"side view, wall-aligned ({align_deg:.1f} deg about axis {args.up_axis})", args.up_axis)
    _plot_height_hist(ax_hist, centers_aligned, args.up_axis)
    fig.suptitle(f"estimated floor bands (camera path, {HEIGHT_BIN_M}m bins): {n_floors}")
    fig.tight_layout()

    out_path = workspace / STAGE_DIRNAME / "scene_extent_side.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[floor_count] estimated {n_floors} floor band(s) from camera height histogram")
    print(f"[floor_count] wrote {out_path}")

    size_path = write_floors_txt(workspace, n_floors)
    print(f"[floor_count] wrote {size_path}")
    return 0


def _demo() -> None:
    """count_floor_bands(centers, up_axis) on synthetic camera paths -- the
    up_axis-selecting wrapper around floor_bands.count_floor_bands, which has
    its own threshold-formula self-check (floor_bands.py --demo)."""
    two_floors = np.zeros((60, 3))
    two_floors[:30, 1] = 0.0
    two_floors[30:, 1] = 3.5   # 30 points clear the 3.0m floor-2 threshold
    assert count_floor_bands(two_floors) == 2, "expected 2 floor bands"

    three_floors = np.zeros((90, 3))
    three_floors[:30, 1] = 0.0
    three_floors[30:60, 1] = 3.5   # clears floor 2 (3.0m)
    three_floors[60:, 1] = 6.0     # clears floor 3 (5.8m)
    assert count_floor_bands(three_floors) == 3, "expected 3 floor bands"

    not_enough_points = np.zeros((30, 3))
    not_enough_points[:20, 1] = 0.0
    not_enough_points[20:, 1] = 3.5   # only 10 points above 3.0m, <= FLOOR_MIN_POINTS
    assert count_floor_bands(not_enough_points) == 1, "expected 1 floor band (not enough points)"

    one_floor = np.zeros((200, 3))
    one_floor[:, 1] = np.random.default_rng(2).normal(1.5, 0.3, 200)
    assert count_floor_bands(one_floor) == 1, "expected 1 floor band"
    print("[floor_count._demo] OK")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        _demo()
        raise SystemExit(0)
    raise SystemExit(main())
