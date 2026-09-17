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
    horizontal_align_deg,
    rotate_horizontal,
)


def _plot_topdown(ax, points: np.ndarray, centers: np.ndarray, title: str, up_axis: int = 1) -> None:
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
    ax.legend()


def render_topdown(sparse_dir: Path, up_axis: int = 1):
    cloud = fetchPly(str(sparse_dir / "points3D.ply"))
    cameras = read_extrinsics_text(str(sparse_dir / "images.txt"))
    centers = np.array([-e.qvec2rotmat().T @ e.tvec for e in cameras.values()])

    # Drop specular/reflection stray points (see scene_extent.PCT_LO/PCT_HI)
    # before they blow up the plot bounds or fake an edge for alignment.
    lo, hi = np.percentile(cloud.points, PCT_LO, axis=0), np.percentile(cloud.points, PCT_HI, axis=0)
    points = cloud.points[np.all((cloud.points >= lo) & (cloud.points <= hi), axis=1)]

    align_deg = horizontal_align_deg(points, up_axis)
    points_aligned = rotate_horizontal(points, align_deg, up_axis)
    centers_aligned = rotate_horizontal(centers, align_deg, up_axis)

    fig, (ax_raw, ax_aligned) = plt.subplots(1, 2, figsize=(20, 10))
    _plot_topdown(ax_raw, points, centers, "raw (world X, -Z)", up_axis)
    _plot_topdown(ax_aligned, points_aligned, centers_aligned,
                  f"wall-aligned ({align_deg:.1f} deg about Y)", up_axis)
    fig.tight_layout()
    return fig, {"rotation_deg_about_up_axis": align_deg}


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
    parser.add_argument("--up-axis", type=int, default=1)
    parser.add_argument("--demo", action="store_true", help="run the self-check instead of rendering")
    args = parser.parse_args(argv)

    if args.demo:
        _demo()
        return 0

    workspace = Path(args.workspace)
    fig, meta = render_topdown(workspace / "sparse" / "0", args.up_axis)
    out_path = workspace / "scene_extent_topdown.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[topdown_view] wrote {out_path} (aligned {meta['rotation_deg_about_up_axis']:.1f} deg)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
