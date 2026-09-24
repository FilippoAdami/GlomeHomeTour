#!/usr/bin/env python3
"""Render the corrected top-down (floor-plan) view of the COLMAP point cloud:
raw vs. wall-aligned, side by side.

Reuses scene_extent's outlier trim (drops specular/reflection stray points --
see PCT_LO/PCT_HI there) and wall alignment. The only thing scene_extent
doesn't do is plot the result, and that has one non-obvious trap: ARCore's
world frame is right-handed Y-up, so plotting raw X right / raw Z up (as a
naive `plt.scatter(x, z)` does) draws the room as seen from *below* (+Y
looking up) -- a left-right mirror of what someone standing in the room
looking down would see. Negating Z before it's the plot's vertical axis
fixes it (look down -Y, i.e. from above).

    python 01_poses_refinment/topdown_view.py [--workspace DIR]
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
    PCT_LO,
    PCT_HI,
    estimate_north_heading_deg,
    horizontal_align_deg,
    rotate_horizontal,
)


def draw_compass(ax, heading_deg: float, is_magnetic: bool = True) -> None:
    """Draw an orientation / compass rose badge in a dedicated square inset on the axis."""
    ax_compass = ax.inset_axes([0.80, 0.77, 0.18, 0.18])
    ax_compass.set_aspect("equal")
    ax_compass.axis("off")
    ax_compass.set_xlim(-1.65, 1.65)
    ax_compass.set_ylim(-1.65, 1.65)

    # Background disk
    disk = plt.Circle((0, 0), 1.25, color="white", alpha=0.92, ec="#78909C", lw=1.5, zorder=1)
    ax_compass.add_patch(disk)

    # Cardinal ticks
    for ang, lbl in [(0, "-Z"), (90, "+X"), (180, "+Z"), (270, "-X")]:
        rad = np.radians(ang)
        ax_compass.plot([0.95 * np.sin(rad), 1.15 * np.sin(rad)],
                        [0.95 * np.cos(rad), 1.15 * np.cos(rad)],
                        color="#B0BEC5", lw=1.0, zorder=2)
        ax_compass.text(1.40 * np.sin(rad), 1.40 * np.cos(rad), lbl,
                        fontsize=7, color="#78909C", ha="center", va="center", zorder=2)

    # Needle pointer (heading_deg is clockwise from top / +Y_plot)
    theta = np.radians(heading_deg)
    dx = np.sin(theta)
    dy = np.cos(theta)
    px = 0.30 * np.cos(theta)
    py = -0.30 * np.sin(theta)

    if is_magnetic:
        # Magnetic North half (Red)
        north_r = plt.Polygon([[dx, dy], [px, py], [0, 0]], color="#E53935", zorder=3)
        north_l = plt.Polygon([[dx, dy], [-px, -py], [0, 0]], color="#C62828", zorder=3)
        ax_compass.add_patch(north_r)
        ax_compass.add_patch(north_l)

        # South half (Blue-Gray)
        south_r = plt.Polygon([[-dx, -dy], [px, py], [0, 0]], color="#90A4AE", zorder=3)
        south_l = plt.Polygon([[-dx, -dy], [-px, -py], [0, 0]], color="#607D8B", zorder=3)
        ax_compass.add_patch(south_r)
        ax_compass.add_patch(south_l)

        # 'N' and 'S' text
        ax_compass.text(1.55 * dx, 1.55 * dy, "N", fontsize=11, fontweight="bold",
                        color="#D32F2F", ha="center", va="center", zorder=5)
        ax_compass.text(-1.55 * dx, -1.55 * dy, "S", fontsize=9, fontweight="bold",
                        color="#546E7A", ha="center", va="center", zorder=5)
    else:
        # Wall Alignment axis needle (Indigo / Purple)
        needle_r = plt.Polygon([[dx, dy], [px, py], [0, 0]], color="#3F51B5", zorder=3)
        needle_l = plt.Polygon([[dx, dy], [-px, -py], [0, 0]], color="#283593", zorder=3)
        ax_compass.add_patch(needle_r)
        ax_compass.add_patch(needle_l)

        opp_r = plt.Polygon([[-dx, -dy], [px, py], [0, 0]], color="#9FA8DA", zorder=3)
        opp_l = plt.Polygon([[-dx, -dy], [-px, -py], [0, 0]], color="#7986CB", zorder=3)
        ax_compass.add_patch(opp_r)
        ax_compass.add_patch(opp_l)

        ax_compass.text(1.55 * dx, 1.55 * dy, "Wall", fontsize=8, fontweight="bold",
                        color="#283593", ha="center", va="center", zorder=5)

    # Pivot dot
    pivot = plt.Circle((0, 0), 0.10, color="#37474F", zorder=4)
    ax_compass.add_patch(pivot)


def _plot_topdown(ax, points: np.ndarray, centers: np.ndarray, title: str,
                  up_axis: int = 1, heading_deg: float | None = None, is_magnetic: bool = True) -> None:
    h_idx = [i for i in range(3) if i != up_axis]  # -> [X, Z]
    pts_2d = points[:, h_idx].copy()
    pts_2d[:, 1] *= -1.0  # look down -Y (from above), not up +Y -- see module docstring
    cam_2d = centers[:, h_idx].copy()
    cam_2d[:, 1] *= -1.0

    ax.scatter(pts_2d[:, 0], pts_2d[:, 1], s=0.5, alpha=0.15, c="k")
    ax.scatter(cam_2d[:, 0], cam_2d[:, 1], s=12, c="red", label="camera path")
    lo, hi = pts_2d.min(axis=0), pts_2d.max(axis=0)
    ax.add_patch(plt.Rectangle(lo, *(hi - lo), fill=False, ec="black", lw=1))
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("-Z (m), i.e. forward")
    ax.set_aspect("equal")
    ax.legend(loc="lower left")

    if heading_deg is not None:
        draw_compass(ax, heading_deg, is_magnetic=is_magnetic)


def render_topdown(sparse_dir: Path, up_axis: int = 1, transforms_path: Path | None = None):
    cloud = fetchPly(str(sparse_dir / "points3D.ply"))
    cameras = read_extrinsics_text(str(sparse_dir / "images.txt"))
    centers = np.array([-e.qvec2rotmat().T @ e.tvec for e in cameras.values()])

    # Drop specular/reflection stray points (see scene_extent.PCT_LO/PCT_HI)
    # before they blow up the plot bounds or fake an edge for alignment.
    lo, hi = np.percentile(cloud.points, PCT_LO, axis=0), np.percentile(cloud.points, PCT_HI, axis=0)
    points = cloud.points[np.all((cloud.points >= lo) & (cloud.points <= hi), axis=1)]

    north_deg = estimate_north_heading_deg(sparse_dir, transforms_path)
    if north_deg is not None:
        align_deg = float((90.0 - north_deg) % 360.0)
        raw_title = f"raw (world X, -Z; North {north_deg:.1f}°)"
        aligned_title = f"compass-aligned (+X=North, {align_deg:.1f}° about Y)"
        raw_heading = north_deg
        aligned_heading = 90.0  # +X is 90 deg clockwise from top
        is_magnetic = True
    else:
        align_deg = horizontal_align_deg(points, up_axis)
        raw_title = "raw (world X, -Z)"
        aligned_title = f"wall-aligned ({align_deg:.1f} deg about Y)"
        raw_heading = float((90.0 - align_deg) % 360.0)
        aligned_heading = 90.0  # primary wall aligned with +X
        is_magnetic = False

    points_aligned = rotate_horizontal(points, align_deg, up_axis)
    centers_aligned = rotate_horizontal(centers, align_deg, up_axis)

    fig, (ax_raw, ax_aligned) = plt.subplots(1, 2, figsize=(20, 10))
    _plot_topdown(ax_raw, points, centers, raw_title, up_axis, heading_deg=raw_heading, is_magnetic=is_magnetic)
    _plot_topdown(ax_aligned, points_aligned, centers_aligned, aligned_title, up_axis, heading_deg=aligned_heading, is_magnetic=is_magnetic)
    fig.tight_layout()
    return fig, {"rotation_deg_about_up_axis": align_deg, "north_heading_deg": north_deg}


def _demo() -> None:
    """Two points offset in +X and +Z only -- after the -Y-looking-down
    projection, the +Z point must land *below* the origin on the plot (not
    above), i.e. Z isn't silently plotted unnegated (the mirroring bug this
    script exists to fix)."""
    pts = np.array([[0.0, 1.0, 0.0], [2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])  # origin, +X, +Z
    h_idx = [0, 2]
    pts_2d = pts[:, h_idx].copy()
    pts_2d[:, 1] *= -1.0
    assert pts_2d[2, 1] < pts_2d[0, 1], "expected +Z point to plot below the origin after -Y projection"
    assert pts_2d[1, 0] > pts_2d[0, 0], "expected +X point to plot to the right of the origin"
    print("[topdown_view._demo] OK: +Z plots down, +X plots right -- matches looking down from above")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--transforms", default=None, help="Optional explicit path to transforms.json")
    parser.add_argument("--up-axis", type=int, default=1)
    parser.add_argument("--demo", action="store_true", help="run the self-check instead of rendering")
    args = parser.parse_args(argv)

    if args.demo:
        _demo()
        return 0

    workspace = Path(args.workspace)
    stage_dir = workspace / "01_poses_refinment"
    transforms_path = Path(args.transforms) if args.transforms else None
    sparse_dir = workspace / "sparse" / "0"
    if not (sparse_dir / "points3D.ply").exists():
        for alt in ["sparse_full", "sparse_large", "sparse_medium", "sparse_lite"]:
            if (workspace / alt / "0" / "points3D.ply").exists():
                sparse_dir = workspace / alt / "0"
                break
    fig, meta = render_topdown(sparse_dir, args.up_axis, transforms_path=transforms_path)
    
    out_path = workspace / "scene_extent_topdown.png"
    fig.savefig(out_path, dpi=150)
    if stage_dir.exists():
        fig.savefig(stage_dir / "scene_extent_topdown.png", dpi=150)
    plt.close(fig)
    print(f"[topdown_view] wrote {out_path} (aligned {meta['rotation_deg_about_up_axis']:.1f} deg)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
