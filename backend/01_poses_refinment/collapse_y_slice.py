#!/usr/bin/env python3
"""Collapse point cloud Y-axis between arbitrary bounds and save topdown image.

Usage:
    python collapse_y_slice.py --min-y 0.0 --max-y 2.0 --out slice_0_2m.png [--ply PATH] [--workspace DIR]
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


def find_floor_plane_y(points_aligned: np.ndarray, normals: np.ndarray, up_axis: int = 1) -> float:
    """Detect the lowest large horizontal plane (floor) using normal vector density."""
    ny_vals = normals[:, up_axis]
    # Filter horizontal surface candidates (vertical normals |ny| > 0.7)
    floor_candidate_mask = np.abs(ny_vals) > 0.7
    y_floors = points_aligned[floor_candidate_mask, up_axis]

    if len(y_floors) == 0:
        return float(np.percentile(points_aligned[:, up_axis], 5.0))

    hist, bin_edges = np.histogram(y_floors, bins=100)
    max_bin_idx = np.argmax(hist)
    floor_y = float((bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2.0)
    return floor_y


def get_camera_height_clusters(centers_aligned: np.ndarray, up_axis: int = 1, bin_width: float = 0.15) -> dict:
    """Analyze camera height histogram to find primary acquisition height clusters (floors/passes)."""
    cam_y = centers_aligned[:, up_axis]
    lo, hi = cam_y.min(), cam_y.max()
    nbins = max(1, int(np.ceil((hi - lo) / bin_width)))
    hist, bin_edges = np.histogram(cam_y, bins=nbins)

    # Find local maxima in camera histogram (> 5 cameras per bin)
    peaks = []
    for i in range(len(hist)):
        if hist[i] >= 5:
            is_peak = (i == 0 or hist[i] >= hist[i-1]) and (i == len(hist)-1 or hist[i] >= hist[i+1])
            if is_peak:
                peak_y = (bin_edges[i] + bin_edges[i+1]) / 2.0
                peaks.append((peak_y, hist[i]))

    return {"histogram": hist, "bin_edges": bin_edges, "peaks": peaks, "cam_y_min": lo, "cam_y_max": hi}


def render_y_slice(
    ply_path: Path,
    sparse_dir: Path,
    min_y: float,
    max_y: float,
    xz_margin: float = 0.8,
    up_axis: int = 1,
    align_mode: str = "wall",
    camera_size: float = 18.0,
    point_size: float = 0.5,
    dpi: int = 150,
):
    cloud = fetchPly(str(ply_path))
    cameras = read_extrinsics_text(str(sparse_dir / "images.txt"))
    centers = np.array([-e.qvec2rotmat().T @ e.tvec for e in cameras.values()])

    # Outlier filter on global percentiles (matching scene_extent / topdown_view)
    lo, hi = np.percentile(cloud.points, PCT_LO, axis=0), np.percentile(cloud.points, PCT_HI, axis=0)
    valid_mask = np.all((cloud.points >= lo) & (cloud.points <= hi), axis=1)
    points = cloud.points[valid_mask]
    normals = cloud.normals[valid_mask]

    # Use wall-alignment Sobel edge orientation by default so walls line up straight with grid axes
    if align_mode == "wall":
        align_deg = horizontal_align_deg(points, up_axis)
    else:
        align_deg = 0.0

    # Rotate ALL points and ALL camera centers to aligned space first
    points_aligned = rotate_horizontal(points, align_deg, up_axis)
    centers_aligned = rotate_horizontal(centers, align_deg, up_axis)

    # Detect the floor plane height (lowest large horizontal plane)
    floor_y = find_floor_plane_y(points_aligned, normals, up_axis)

    # Shift Y axis so the main floor plane sits exactly at Y = 0
    points_aligned[:, up_axis] -= floor_y
    centers_aligned[:, up_axis] -= floor_y

    # Camera height stats & clustering (ingestion floor_count schema)
    cam_clusters = get_camera_height_clusters(centers_aligned, up_axis)

    # 1. Filter camera positions by shifted Y bounds [min_y, max_y]
    y_mask_cams = (centers_aligned[:, up_axis] >= min_y) & (centers_aligned[:, up_axis] <= max_y)
    slice_centers = centers_aligned[y_mask_cams]

    # 2. Trim extreme camera height outliers (using 5th-95th percentiles of retained cameras if > 10 cameras)
    if len(slice_centers) >= 10:
        cam_y_retained = slice_centers[:, up_axis]
        p5_y, p95_y = np.percentile(cam_y_retained, [5.0, 95.0])
        # Tighter mask trimming stray top/bottom cameras in transition
        tight_cam_mask = (slice_centers[:, up_axis] >= p5_y - 0.1) & (slice_centers[:, up_axis] <= p95_y + 0.1)
        cluster_cams = slice_centers[tight_cam_mask]
    else:
        cluster_cams = slice_centers

    # 3. Compute X and Z bounds from retained camera cluster (with xz_margin)
    if len(cluster_cams) > 0:
        cam_x_min, cam_x_max = cluster_cams[:, 0].min(), cluster_cams[:, 0].max()
        cam_z_min, cam_z_max = cluster_cams[:, 2].min(), cluster_cams[:, 2].max()

        x_min, x_max = cam_x_min - xz_margin, cam_x_max + xz_margin
        z_min, z_max = cam_z_min - xz_margin, cam_z_max + xz_margin
    else:
        x_min, x_max = -np.inf, np.inf
        z_min, z_max = -np.inf, np.inf

    # 4. Filter points by shifted Y bounds AND X/Z bounds of camera cluster
    y_mask_pts = (points_aligned[:, up_axis] >= min_y) & (points_aligned[:, up_axis] <= max_y)
    xz_mask_pts = (
        (points_aligned[:, 0] >= x_min) & (points_aligned[:, 0] <= x_max) &
        (points_aligned[:, 2] >= z_min) & (points_aligned[:, 2] <= z_max)
    )
    slice_points = points_aligned[y_mask_pts & xz_mask_pts]

    print(f"[collapse_y_slice] PLY: {ply_path.name}")
    print(f"[collapse_y_slice] Floor plane detected at Y_raw = {floor_y:.3f} m -> shifted to Y = 0.0 m")
    print(f"[collapse_y_slice] Camera height peaks detected at Y: {[round(p[0], 2) for p in cam_clusters['peaks']]} m")
    print(f"[collapse_y_slice] Y slice range (floor-relative): [{min_y:.2f}, {max_y:.2f}] m")
    print(f"[collapse_y_slice] Retained cameras in range: {len(slice_centers)} / {len(centers)} (cluster core: {len(cluster_cams)})")
    if len(cluster_cams) > 0:
        print(f"[collapse_y_slice] Camera X range: [{cam_x_min:.2f}, {cam_x_max:.2f}] m -> point filter [{x_min:.2f}, {x_max:.2f}] m (margin={xz_margin}m)")
        print(f"[collapse_y_slice] Camera Z range: [{cam_z_min:.2f}, {cam_z_max:.2f}] m -> point filter [{z_min:.2f}, {z_max:.2f}] m (margin={xz_margin}m)")
    print(f"[collapse_y_slice] Points retained: {len(slice_points)} / {len(cloud.points)}")

    # Prepare topdown 2D coordinates (X, -Z)
    h_idx = [i for i in range(3) if i != up_axis]  # -> [X, Z]
    pts_2d = slice_points[:, h_idx].copy()
    pts_2d[:, 1] *= -1.0  # look down -Y (from above)

    cam_2d = slice_centers[:, h_idx].copy()
    cam_2d[:, 1] *= -1.0

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("white")

    if len(pts_2d) > 0:
        ax.scatter(pts_2d[:, 0], pts_2d[:, 1], s=point_size, c="black", alpha=0.8, linewidths=0)

    if len(cam_2d) > 0:
        ax.scatter(cam_2d[:, 0], cam_2d[:, 1], s=camera_size, c="red", label=f"camera path ({len(cam_2d)})", alpha=0.9, edgecolors="black", linewidths=0.5)

    ax.set_title(f"Floor-Zeroed Y-Slice [{min_y:.2f}m to {max_y:.2f}m] (Camera Schema Bounded)\n({len(pts_2d)} pts, {len(cam_2d)} cams, XZ margin ±{xz_margin}m, aligned {align_deg:.1f}°)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("-Z (m), i.e. forward")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    if len(cam_2d) > 0:
        ax.legend(loc="lower left")

    fig.tight_layout()
    return fig


def read_floors_from_scene_size(workspace: Path) -> int:
    """Read floor count from scene_size.txt or default to 1."""
    scene_size_path = workspace / "01_poses_refinment" / "scene_size.txt"
    if scene_size_path.exists():
        for line in scene_size_path.read_text().splitlines():
            if line.strip().startswith("floors:"):
                try:
                    return int(line.split(":")[1].strip())
                except ValueError:
                    pass
    return 1


def render_floor_slices(
    ply_path: Path,
    sparse_dir: Path,
    workspace: Path,
    override_min_y: float | None = None,
    override_max_y: float | None = None,
    xz_margin: float = 0.8,
    up_axis: int = 1,
    align_mode: str = "wall",
    camera_size: float = 18.0,
    point_size: float = 0.5,
    out_prefix: str = "slice_floor",
):
    cloud = fetchPly(str(ply_path))
    cameras = read_extrinsics_text(str(sparse_dir / "images.txt"))
    centers = np.array([-e.qvec2rotmat().T @ e.tvec for e in cameras.values()])

    # Outlier filter on global percentiles (matching scene_extent / topdown_view)
    lo, hi = np.percentile(cloud.points, PCT_LO, axis=0), np.percentile(cloud.points, PCT_HI, axis=0)
    valid_mask = np.all((cloud.points >= lo) & (cloud.points <= hi), axis=1)
    points = cloud.points[valid_mask]
    normals = cloud.normals[valid_mask]

    # Use wall-alignment Sobel edge orientation by default so walls line up straight with grid axes
    if align_mode == "wall":
        align_deg = horizontal_align_deg(points, up_axis)
    else:
        align_deg = 0.0

    # Rotate ALL points and ALL camera centers to aligned space first
    points_aligned = rotate_horizontal(points, align_deg, up_axis)
    centers_aligned = rotate_horizontal(centers, align_deg, up_axis)

    # Detect the floor plane height (lowest large horizontal plane)
    floor_y = find_floor_plane_y(points_aligned, normals, up_axis)

    # If the floor plane is already aligned at Y=0 (within 5cm), do not double shift
    if abs(floor_y) < 0.05:
        floor_shift = 0.0
    else:
        floor_shift = floor_y

    # Shift Y axis so the main floor plane sits exactly at Y = 0
    points_aligned[:, up_axis] -= floor_shift
    centers_aligned[:, up_axis] -= floor_shift

    num_floors = read_floors_from_scene_size(workspace)
    print(f"[collapse_y_slice] PLY: {ply_path.name}")
    print(f"[collapse_y_slice] Floor plane detected at Y_raw = {floor_y:.3f} m -> applied shift = {floor_shift:.3f} m")
    print(f"[collapse_y_slice] Total floors detected in scene_size.txt: {num_floors}")

    # Build per-floor specs or single custom range spec
    floor_specs = []
    if override_min_y is not None and override_max_y is not None:
        floor_specs.append({
            "floor_idx": 0,
            "cam_min_y": override_min_y,
            "cam_max_y": override_max_y,
            "pts_min_y": override_min_y,
            "pts_max_y": override_max_y,
            "name": f"{out_prefix}_custom"
        })
    else:
        for f in range(num_floors):
            # Camera interval: 3m intervals with 0.2m overlap
            # floor 0: y=0 to y<3; floor 1: y=2.8 to y<6; floor 2: y=5.6 to y<8.8 ...
            cam_min = f * 2.8
            cam_max = cam_min + 3.0

            # Points interval: 3m intervals
            # floor 0: y=-1 to y<2; floor 1: y=1.8 to y<4.8; floor 2: y=4.6 to y<7.6 ...
            pts_min = f * 2.8 - 1.0
            pts_max = pts_min + 3.0

            floor_specs.append({
                "floor_idx": f,
                "cam_min_y": cam_min,
                "cam_max_y": cam_max,
                "pts_min_y": pts_min,
                "pts_max_y": pts_max,
                "name": f"{out_prefix}_{f}"
            })

    output_files = []
    for spec in floor_specs:
        cam_min_y, cam_max_y = spec["cam_min_y"], spec["cam_max_y"]
        pts_min_y, pts_max_y = spec["pts_min_y"], spec["pts_max_y"]
        f_idx = spec["floor_idx"]

        # Filter cameras for this floor band
        y_mask_cams = (centers_aligned[:, up_axis] >= cam_min_y) & (centers_aligned[:, up_axis] < cam_max_y)
        slice_centers = centers_aligned[y_mask_cams]

        # X/Z bounding based on retained cameras for this floor + margin
        if len(slice_centers) > 0:
            cam_x_min, cam_x_max = slice_centers[:, 0].min(), slice_centers[:, 0].max()
            cam_z_min, cam_z_max = slice_centers[:, 2].min(), slice_centers[:, 2].max()

            x_min, x_max = cam_x_min - xz_margin, cam_x_max + xz_margin
            z_min, z_max = cam_z_min - xz_margin, cam_z_max + xz_margin
        else:
            x_min, x_max = -np.inf, np.inf
            z_min, z_max = -np.inf, np.inf

        # Filter points by floor's point Y bounds AND retained camera X/Z bounds
        y_mask_pts = (points_aligned[:, up_axis] >= pts_min_y) & (points_aligned[:, up_axis] < pts_max_y)
        xz_mask_pts = (
            (points_aligned[:, 0] >= x_min) & (points_aligned[:, 0] <= x_max) &
            (points_aligned[:, 2] >= z_min) & (points_aligned[:, 2] <= z_max)
        )
        slice_points = points_aligned[y_mask_pts & xz_mask_pts]

        print(f"\n--- Floor {f_idx} ---")
        print(f"Camera Y range: [{cam_min_y:.2f}, {cam_max_y:.2f}) m -> {len(slice_centers)} cameras retained")
        if len(slice_centers) > 0:
            print(f"XZ camera box: X=[{cam_x_min:.2f}, {cam_x_max:.2f}], Z=[{cam_z_min:.2f}, {cam_z_max:.2f}]")
            print(f"XZ point box (+-{xz_margin}m): X=[{x_min:.2f}, {x_max:.2f}], Z=[{z_min:.2f}, {z_max:.2f}]")
        print(f"Points Y range: [{pts_min_y:.2f}, {pts_max_y:.2f}) m -> {len(slice_points)} points retained")

        # Prepare topdown 2D coordinates (X, -Z)
        h_idx = [i for i in range(3) if i != up_axis]  # -> [X, Z]
        pts_2d = slice_points[:, h_idx].copy()
        pts_2d[:, 1] *= -1.0  # look down -Y (from above)

        cam_2d = slice_centers[:, h_idx].copy()
        cam_2d[:, 1] *= -1.0

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_facecolor("white")

        if len(pts_2d) > 0:
            ax.scatter(pts_2d[:, 0], pts_2d[:, 1], s=point_size, c="black", alpha=0.8, linewidths=0)

        if len(cam_2d) > 0:
            ax.scatter(cam_2d[:, 0], cam_2d[:, 1], s=camera_size, c="red", label=f"cameras ({len(cam_2d)})", alpha=0.9, edgecolors="black", linewidths=0.5)

        ax.set_title(f"Floor {f_idx} (Cam Y: [{cam_min_y:.1f},{cam_max_y:.1f})m, Pts Y: [{pts_min_y:.1f},{pts_max_y:.1f})m)\n({len(pts_2d)} pts, {len(cam_2d)} cams, XZ margin ±{xz_margin}m, aligned {align_deg:.1f}°)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("-Z (m), i.e. forward")
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.3)
        if len(cam_2d) > 0:
            ax.legend(loc="lower left")

        fig.tight_layout()

        out_path = workspace / f"{spec['name']}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[collapse_y_slice] Floor {f_idx} slice output saved to {out_path}")
        output_files.append(out_path)

    return output_files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--ply", default=None, help="Explicit path to PLY file. Defaults to 02_depth_estimation/depth/points3D_depth.ply")
    parser.add_argument("--min-y", type=float, default=None, help="Optional custom lower Y bound (overrides automatic per-floor schema)")
    parser.add_argument("--max-y", type=float, default=None, help="Optional custom upper Y bound (overrides automatic per-floor schema)")
    parser.add_argument("--xz-margin", type=float, default=0.8, help="Margin around camera cluster min/max X and Z (default 0.8m)")
    parser.add_argument("--out-prefix", type=str, default="slice_floor", help="Prefix for output PNG file(s)")
    parser.add_argument("--align-mode", choices=["wall", "none"], default="wall", help="Alignment mode: 'wall' (default) or 'none'")
    parser.add_argument("--up-axis", type=int, default=1)
    parser.add_argument("--camera-size", type=float, default=18.0, help="Camera marker size (default 18.0)")
    parser.add_argument("--point-size", type=float, default=0.5, help="Point display size (default 0.5)")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    sparse_dir = workspace / "sparse" / "0"
    if not (sparse_dir / "images.txt").exists():
        for alt in ["sparse_full", "sparse_large", "sparse_medium", "sparse_lite"]:
            if (workspace / alt / "0" / "images.txt").exists():
                sparse_dir = workspace / alt / "0"
                break

    if args.ply:
        ply_path = Path(args.ply)
    else:
        ply_path = workspace / "02_depth_estimation" / "depth" / "points3D_depth.ply"

    if not ply_path.exists():
        raise FileNotFoundError(f"Target PLY file not found: {ply_path}")

    render_floor_slices(
        ply_path=ply_path,
        sparse_dir=sparse_dir,
        workspace=workspace,
        override_min_y=args.min_y,
        override_max_y=args.max_y,
        xz_margin=args.xz_margin,
        up_axis=args.up_axis,
        align_mode=args.align_mode,
        camera_size=args.camera_size,
        point_size=args.point_size,
        out_prefix=args.out_prefix,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

