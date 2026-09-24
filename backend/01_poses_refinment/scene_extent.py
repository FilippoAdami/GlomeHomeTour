#!/usr/bin/env python3
"""Rough scene dimensions from the refined COLMAP output, before depth estimation.

COLMAP here is seeded with ARCore poses and only pose-prior bundle-adjusted /
triangulated (see ``convert_transforms_to_colmap.py``) -- it never rescales,
so ``points3D.ply`` and the camera centers are already in the same metric
ARCore frame. No Sim(3) alignment is needed; this just bounds what's there.

    python 01_poses_refinment/scene_extent.py [--workspace DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from shapely import concave_hull, MultiPoint

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
from Utilities.pipeline_paths import bootstrap

bootstrap()

from scene.colmap_loader import read_extrinsics_text
from scene.dataset_readers import fetchPly

DEFAULT_WORKSPACE = _backend_dir / "current_scene"
STAGE_DIRNAME = "01_poses_refinment"

# Trim the outer tail of each axis before taking the extent: a handful of
# stray triangulated points (reflections, far-through-a-window matches) can
# blow up a plain min/max by meters.
PCT_LO, PCT_HI = 1.0, 99.0


def extent_from_points(xyz: np.ndarray) -> dict:
    lo = np.percentile(xyz, PCT_LO, axis=0)
    hi = np.percentile(xyz, PCT_HI, axis=0)
    size = hi - lo
    return {"min": lo.tolist(), "max": hi.tolist(), "size_m": size.tolist()}


# ARCore's world Y axis is gravity-locked at session start, but heading
# (the horizontal X/Z orientation) is arbitrary -- whichever way the phone
# faced when the session began. This is a Manhattan-world proxy, not real
# wall detection -- real wall RANSAC belongs in floorplan/ once that stage
# exists; this just makes scene_extent's X/Z numbers mean something before
# then.
#
# Collapses the horizontal plane to a density image (splat + blur), takes
# Sobel gradients, and takes the weighted circular mean of gradient angle
# folded mod 90 deg -- wall/furniture edges all vote for their orientation,
# folding mod-90 lets doors/windows/clutter vote too instead of only unbroken
# wall runs. Tried a convex-hull min-area rect first: it locks onto whatever
# clutter blob sticks out furthest rather than the walls. Tried raw
# cv2.HoughLines on a binary occupancy grid next: dominated by the grid's own
# raster diagonal, not the room. This is what actually lines walls up straight
# (checked visually -- see current_scene/scene_extent_topdown.png).
GRID_RES_M = 0.03
EDGE_PERCENTILE = 90.0


def horizontal_align_deg(xyz: np.ndarray, up_axis: int = 1) -> float:
    horiz = xyz[:, [i for i in range(3) if i != up_axis]]
    h_lo = horiz.min(axis=0)
    grid_shape = np.ceil((horiz.max(axis=0) - h_lo) / GRID_RES_M).astype(int) + 1
    idx = np.floor((horiz - h_lo) / GRID_RES_M).astype(int)

    density = np.zeros(grid_shape[::-1], dtype=np.float32)
    np.add.at(density, (idx[:, 1], idx[:, 0]), 1.0)
    density = cv2.GaussianBlur(density, (0, 0), sigmaX=1.5)

    gx = cv2.Sobel(density, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(density, cv2.CV_32F, 0, 1, ksize=5)
    mag, ang = np.sqrt(gx ** 2 + gy ** 2), np.arctan2(gy, gx)

    strong = mag > np.percentile(mag, EDGE_PERCENTILE)
    mean_vec = np.sum(mag[strong] * np.exp(1j * 4 * ang[strong]))
    # Negated: this is the wall orientation *in* the raw frame, so undoing it
    # with rotate_horizontal(pts, result) requires the opposite sign.
    return -float(np.degrees(np.angle(mean_vec)) / 4.0)


def rotate_horizontal(xyz: np.ndarray, angle_deg: float, up_axis: int = 1) -> np.ndarray:
    h_idx = [i for i in range(3) if i != up_axis]
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot2d = np.array([[c, -s], [s, c]])
    out = xyz.copy()
    out[:, h_idx] = xyz[:, h_idx] @ rot2d.T
    return out


# x*z bounding-box area overshoots on L-shaped/non-rectangular rooms -- a
# concave hull of the top-down projection traces the actual wall footprint
# instead. ratio=0.0 is the tightest hull (follows noise/gaps in the wall
# points); 1.0 degenerates to the convex hull (back to overshooting concave
# corners). 0.3 was picked by eye against current_scene/scene_extent_topdown.png:
# tight enough to catch an L-corner, loose enough not to zigzag through gaps
# between wall-point clusters.
CONCAVE_HULL_RATIO = 0.3


def footprint_area_m2(points_aligned: np.ndarray, up_axis: int = 1,
                       ratio: float = CONCAVE_HULL_RATIO) -> float:
    horiz_idx = [i for i in range(3) if i != up_axis]
    hull = concave_hull(MultiPoint(points_aligned[:, horiz_idx]), ratio=ratio)
    return float(hull.area)


def estimate_north_heading_deg(sparse_dir: Path, transforms_path: Path | None = None) -> float | None:
    """Estimate the angle of Magnetic North in degrees [0, 360) clockwise from +Y_plot (-Z_world).

    Returns None if no valid compass_heading_deg entries are found.
    """
    if transforms_path is None or not transforms_path.exists():
        workspace = sparse_dir.parents[1] if len(sparse_dir.parents) >= 2 else sparse_dir.parent
        candidates = [
            workspace / "00_ingestion" / "transforms.json",
            workspace / "transforms.json",
            sparse_dir.parent / "transforms.json",
        ]
        for c in candidates:
            if c.exists():
                transforms_path = c
                break

    if transforms_path is None or not transforms_path.exists():
        return None

    try:
        data = json.loads(transforms_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    diffs = []
    for f in data.get("frames", []):
        compass_deg = f.get("compass_heading_deg")
        if compass_deg is None:
            continue
        try:
            mat = np.array(f["transform_matrix"], dtype=np.float64)
            # Look direction is -Z in camera coordinates: R @ [0, 0, -1].T = [-R02, -R12, -R22]
            # Horizontal forward vector in (X, Z) is (-R02, -R22).
            # In (X, -Z) top-down plot coordinates, (v_x, v_y_plot) is (-R02, R22).
            vx = -mat[0, 2]
            vy_plot = mat[2, 2]
            cam_yaw_deg = float(np.degrees(np.arctan2(vx, vy_plot))) % 360.0

            # Delta is the angle of Magnetic North in (X, -Z) top-down coordinates
            delta = (cam_yaw_deg - float(compass_deg)) % 360.0
            diffs.append(delta)
        except Exception:
            continue

    if not diffs:
        return None

    # Circular mean across frames
    rads = np.radians(diffs)
    c = float(np.mean(np.cos(rads)))
    s = float(np.mean(np.sin(rads)))
    north_deg = float(np.degrees(np.arctan2(s, c))) % 360.0
    return north_deg


def scene_extent(sparse_dir: Path, up_axis: int = 1, transforms_path: Path | None = None) -> dict:
    cloud = fetchPly(str(sparse_dir / "points3D.ply"))
    cameras = read_extrinsics_text(str(sparse_dir / "images.txt"))
    centers = np.array([-e.qvec2rotmat().T @ e.tvec for e in cameras.values()])

    lo, hi = np.percentile(cloud.points, PCT_LO, axis=0), np.percentile(cloud.points, PCT_HI, axis=0)
    trimmed = cloud.points[np.all((cloud.points >= lo) & (cloud.points <= hi), axis=1)]

    north_deg = estimate_north_heading_deg(sparse_dir, transforms_path)
    if north_deg is not None:
        # Align scene so +X is North (which is 90 deg clockwise from +Y_plot)
        align_deg = float((90.0 - north_deg) % 360.0)
        alignment_mode = "compass"
    else:
        align_deg = horizontal_align_deg(trimmed, up_axis)
        alignment_mode = "wall"

    points_aligned = rotate_horizontal(cloud.points, align_deg, up_axis)
    trimmed_aligned = rotate_horizontal(trimmed, align_deg, up_axis)
    centers_aligned = rotate_horizontal(centers, align_deg, up_axis)

    aligned_point_cloud_extent = extent_from_points(points_aligned)

    # Clip to the exact same bbox extent_from_points just derived (it applies
    # its own PCT_LO/PCT_HI trim on top of `trimmed_aligned`'s) -- otherwise
    # points inside trimmed_aligned but outside that tighter bbox can bow the
    # hull out past the reported x/z size, giving a footprint area bigger
    # than the bounding box it's supposed to be smaller than.
    bbox_lo, bbox_hi = np.array(aligned_point_cloud_extent["min"]), np.array(aligned_point_cloud_extent["max"])
    footprint_points = trimmed_aligned[np.all(
        (trimmed_aligned >= bbox_lo) & (trimmed_aligned <= bbox_hi), axis=1)]

    return {
        "point_cloud": extent_from_points(cloud.points),
        "camera_path": extent_from_points(centers),
        "aligned": {
            "alignment_mode": alignment_mode,
            "north_heading_deg": round(north_deg, 2) if north_deg is not None else None,
            "rotation_deg_about_up_axis": round(align_deg, 2),
            "up_axis": up_axis,
            "point_cloud": aligned_point_cloud_extent,
            "camera_path": extent_from_points(centers_aligned),
            # concave-hull footprint, clipped to the same bbox as point_cloud
            # above -- see footprint_area_m2 / CONCAVE_HULL_RATIO.
            "footprint_area_m2": footprint_area_m2(footprint_points, up_axis),
        },
        "num_points": int(cloud.points.shape[0]),
        "num_cameras": len(centers),
        "percentile_trim": [PCT_LO, PCT_HI],
    }


def write_scene_size_txt(workspace: Path, extent: dict[str, Any], floors: int = 1) -> Path:
    """Write scene spatial dimensions, per-floor breakdowns, and total floor area to scene_size.txt.

    ``x``/``z`` are horizontal axes, ``y`` is gravity-locked vertical height.
    Writes each floor's x_span, avg_y, z_span, and floor_area, followed by total_floor_area.
    """
    out_path = workspace / "scene_size.txt"

    # Extract global dimensions if present in the extent dict
    if "aligned" in extent and "point_cloud" in extent["aligned"]:
        x, y, z = extent["aligned"]["point_cloud"]["size_m"]
    else:
        x = extent.get("room_dim_x", extent.get("camera_span_x", 0.0))
        y = extent.get("height_m", 0.0)
        z = extent.get("room_dim_z", extent.get("camera_span_z", 0.0))

    # Retrieve per-floor breakdown and total area
    floors_details: list[dict[str, float]] = extent.get("floors_details", [])
    total_floor_area: float = extent.get(
        "total_floor_area",
        extent.get("floor_area_m2", extent.get("aligned", {}).get("footprint_area_m2", 0.0)),
    )

    # Determine floor count: prefer explicit floors_details length, then extent dict, then fallback
    floor_count = len(floors_details) if floors_details else extent.get("floors", floors)

    # Preserve existing floor count from disk if not resolved above
    if not floors_details and out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.startswith("floors:"):
                try:
                    floor_count = int(float(line.split(":", 1)[1].strip()))
                except ValueError:
                    pass

    # Build structured text output
    lines = [
        f"x: {x:.3f}",
        f"y: {y:.3f}",
        f"z: {z:.3f}",
        f"floors: {floor_count}",
    ]

    # Write itemized per-floor metrics
    for idx, f_info in enumerate(floors_details):
        f_idx = f_info.get("floor_index", idx)
        lines.append(f"floor_{f_idx}_x_span: {f_info['x_span']:.3f}")
        lines.append(f"floor_{f_idx}_avg_y: {f_info['avg_y']:.3f}")
        lines.append(f"floor_{f_idx}_z_span: {f_info['z_span']:.3f}")
        lines.append(f"floor_{f_idx}_area_m2: {f_info['floor_area']:.3f}")

    # Terminate with the cumulative total area
    lines.append(f"total_floor_area_m2: {total_floor_area:.3f}")
    if "aligned" in extent and extent["aligned"].get("north_heading_deg") is not None:
        lines.append(f"north_heading_deg: {extent['aligned']['north_heading_deg']:.2f}")

    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def _demo() -> None:
    """Synthetic rectangular room, points on wall/floor/ceiling planes, rotated
    a known amount about the up axis -- checks horizontal_align_deg() recovers
    it mod 90 deg."""
    rng = np.random.default_rng(0)
    w, h, d = 5.0, 2.5, 3.0
    n_per_wall = 5000
    pts = []
    for lo_val, hi_val, fixed_axis, fixed_val in [
        (0, w, 2, 0.0), (0, w, 2, d),      # front/back walls (vary x,y)
        (0, d, 0, 0.0), (0, d, 0, w),      # side walls (vary z,y)
    ]:
        p = rng.uniform([0, 0], [w if fixed_axis != 0 else d, h], size=(n_per_wall, 2))
        full = np.zeros((n_per_wall, 3))
        free_axes = [i for i in range(3) if i != fixed_axis and i != 1]
        full[:, free_axes[0]] = p[:, 0]
        full[:, 1] = p[:, 1]
        full[:, fixed_axis] = fixed_val
        pts.append(full)
    pts = np.concatenate(pts) - [w / 2, h / 2, d / 2]

    true_angle = 37.0
    rotated = rotate_horizontal(pts, true_angle, up_axis=1)
    found = horizontal_align_deg(rotated, up_axis=1)
    # The real check: applying `found` via rotate_horizontal must undo the
    # rotation and shrink the bbox back down to the true w x d footprint --
    # not that `found` numerically equals true_angle (sign/mod-90 convention
    # is an implementation detail rotate_horizontal and this must agree on,
    # not something callers should have to reason about separately).
    corrected = rotate_horizontal(rotated, found, up_axis=1)
    horiz = corrected[:, [0, 2]]
    recovered_wh = sorted((horiz.max(axis=0) - horiz.min(axis=0)).tolist())
    expected_wh = sorted([w, d])
    assert all(abs(a - b) < 0.15 for a, b in zip(recovered_wh, expected_wh)), (
        f"expected footprint ~{expected_wh}, got {recovered_wh}")
    print(f"[scene_extent._demo] OK: found {found:.2f} deg, corrected footprint {recovered_wh}")


def _demo_l_shape() -> None:
    """Synthetic L-shaped room (big rectangle minus a corner notch), wall
    points only -- checks footprint_area_m2 tracks the true concave area,
    not the bounding-box area (which is what x*z gives and is the whole
    point of using a concave hull over the bbox)."""
    W, D, h = 6.0, 4.0, 2.5          # outer bbox
    notch_w, notch_d = 2.5, 2.0      # cut from the top-right corner
    true_area = W * D - notch_w * notch_d
    bbox_area = W * D

    # L-shaped perimeter, walked corner to corner (outer rectangle with one
    # corner cut off), each edge densely sampled at random heights to stand
    # in for a wall's vertical spread of triangulated points.
    corners = [
        (0.0, 0.0), (W, 0.0), (W, D - notch_d), (W - notch_w, D - notch_d),
        (W - notch_w, D), (0.0, D),
    ]
    rng = np.random.default_rng(0)
    n_per_edge = 2000
    pts = []
    for (x0, z0), (x1, z1) in zip(corners, corners[1:] + corners[:1]):
        t = rng.uniform(0.0, 1.0, n_per_edge)
        x = x0 + t * (x1 - x0)
        z = z0 + t * (z1 - z0)
        y = rng.uniform(0.0, h, n_per_edge)
        pts.append(np.stack([x, y, z], axis=1))
    pts = np.concatenate(pts)

    area = footprint_area_m2(pts, up_axis=1)
    assert area < bbox_area * 0.95, (
        f"hull area {area:.2f} too close to bbox area {bbox_area:.2f} -- not tracking the notch")
    assert abs(area - true_area) / true_area < 0.1, (
        f"expected footprint area ~{true_area:.2f}, got {area:.2f}")
    print(f"[scene_extent._demo_l_shape] OK: hull area {area:.2f} m^2 "
          f"(true {true_area:.2f}, bbox {bbox_area:.2f})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--transforms", default=None, help="Optional explicit path to transforms.json")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    stage_dir = workspace / STAGE_DIRNAME
    stage_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir = workspace / "sparse" / "0"
    if not (sparse_dir / "points3D.ply").exists():
        for alt in ["sparse_full", "sparse_large", "sparse_medium", "sparse_lite"]:
            if (workspace / alt / "0" / "points3D.ply").exists():
                sparse_dir = workspace / alt / "0"
                break
    transforms_path = Path(args.transforms) if args.transforms else None
    result = scene_extent(sparse_dir, transforms_path=transforms_path)
    out_path = stage_dir / "scene_extent.json"
    out_path.write_text(json.dumps(result, indent=2))
    size_path = write_scene_size_txt(stage_dir, result)

    size = result["point_cloud"]["size_m"]
    print(f"[scene_extent] raw room footprint (points, {PCT_LO}-{PCT_HI} pct): "
         f"{size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m")
    aligned_size = result["aligned"]["point_cloud"]["size_m"]
    mode = result["aligned"].get("alignment_mode", "wall")
    print(f"[scene_extent] {mode}-aligned footprint ({result['aligned']['rotation_deg_about_up_axis']:.1f} deg "
         f"about up axis {result['aligned']['up_axis']}): "
         f"{aligned_size[0]:.2f} x {aligned_size[1]:.2f} x {aligned_size[2]:.2f} m")
    print(f"[scene_extent] concave-hull footprint area: {result['aligned']['footprint_area_m2']:.2f} m^2 "
          f"(vs. bbox {aligned_size[0] * aligned_size[2]:.2f} m^2)")
    print(f"[scene_extent] written to {out_path} and {size_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
