#!/usr/bin/env python3
"""Estimate scene bounds from the depth stage's surfel cloud and camera poses.

    python 02_depth_estimation/scene_extent.py --workspace current_scene/scan_0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from shapely import concave_hull, MultiPoint

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
from Utilities.pipeline_paths import bootstrap

bootstrap()

from scene.dataset_readers import fetchPly
from Utilities.scene_alignment import horizontal_align_deg, rotate_horizontal

DEFAULT_WORKSPACE = _backend_dir / "current_scene"
STAGE_DIRNAME = "depth"

# Trim the outer tail of each axis before taking the extent: a handful of
# stray points (reflections, far-through-a-window matches) can blow up a plain
# min/max by several scene units.
PCT_LO, PCT_HI = 1.0, 99.0

def extent_from_points(xyz: np.ndarray) -> dict:
    lo = np.percentile(xyz, PCT_LO, axis=0)
    hi = np.percentile(xyz, PCT_HI, axis=0)
    size = hi - lo
    return {"min": lo.tolist(), "max": hi.tolist(), "size": size.tolist()}


# x*z bounding-box area overshoots on L-shaped/non-rectangular rooms -- a
# concave hull of the top-down projection traces the actual wall footprint
# instead. ratio=0.0 is the tightest hull (follows noise/gaps in the wall
# points); 1.0 degenerates to the convex hull (back to overshooting concave
# corners). 0.3 was picked by eye against the scene's top-down diagnostic:
# tight enough to catch an L-corner, loose enough not to zigzag through gaps
# between wall-point clusters.
CONCAVE_HULL_RATIO = 0.3


def footprint_area(points_aligned: np.ndarray, up_axis: int = 1,
                   ratio: float = CONCAVE_HULL_RATIO) -> float:
    horiz_idx = [i for i in range(3) if i != up_axis]
    hull = concave_hull(MultiPoint(points_aligned[:, horiz_idx]), ratio=ratio)
    return float(hull.area)


def estimate_north_heading_deg(workspace: Path, transforms_path: Path | None = None) -> float | None:
    """Estimate the angle of Magnetic North in degrees [0, 360) clockwise from +Y_plot (-Z_world).

    Returns None if no valid compass_heading_deg entries are found.
    """
    if transforms_path is None or not transforms_path.exists():
        candidates = [
            workspace / "00_ingestion" / "transforms.json",
            workspace / "transforms.json",
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


def load_depth_scene(workspace: Path, ply_path: Path | None = None):
    """Load the depth surfel cloud and matching COLMAP-derived camera centers."""
    depth_dir = workspace / STAGE_DIRNAME
    cloud = fetchPly(str(ply_path or depth_dir / "points3D_depth.ply"))
    with np.load(depth_dir / "poses_da3.npz") as poses:
        w2c = poses["w2c"].astype(np.float64)
    centers = -np.einsum("nij,nj->ni", w2c[:, :3, :3].transpose(0, 2, 1), w2c[:, :3, 3])
    return cloud, centers


def load_metric_depth_scene(workspace: Path, ply_path: Path | None = None,
                            transforms_path: Path | None = None):
    """Return metric geometry, converting legacy unscaled COLMAP outputs in memory."""
    cloud, centers = load_depth_scene(workspace, ply_path)
    calibration_path = Path(workspace) / "metric_scale.json"
    try:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if not isinstance(calibration, dict) or not isinstance(calibration.get("available"), bool):
            raise ValueError("invalid metric scale report")
    except FileNotFoundError:
        calibration = {"available": False, "reason": "missing COLMAP-stage metric_scale.json"}
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        calibration = {"available": False, "reason": f"cannot read COLMAP-stage metric scale report: {exc}"}
    scale = calibration.get("meters_per_colmap_unit", 1.0) if calibration["available"] else 1.0
    if (calibration["available"]
            and not calibration.get("scale_applied_to_colmap_geometry", False)):
        cloud = cloud._replace(points=cloud.points * scale)
        centers = centers * scale
    return cloud, centers, calibration


def scene_extent(workspace: Path, up_axis: int = 1,
                 transforms_path: Path | None = None) -> dict:
    cloud, centers, calibration = load_metric_depth_scene(workspace, transforms_path=transforms_path)

    lo, hi = np.percentile(cloud.points, PCT_LO, axis=0), np.percentile(cloud.points, PCT_HI, axis=0)
    trimmed = cloud.points[np.all((cloud.points >= lo) & (cloud.points <= hi), axis=1)]

    north_deg = estimate_north_heading_deg(workspace, transforms_path)
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
        "coordinate_units": "m" if calibration["available"] else "unscaled COLMAP model units",
        "metric_scale_available": bool(calibration["available"]),
        "metric_calibration": calibration,
        "point_cloud": extent_from_points(cloud.points),
        "camera_path": extent_from_points(centers),
        "aligned": {
            "alignment_mode": alignment_mode,
            "north_heading_deg": round(north_deg, 2) if north_deg is not None else None,
            "rotation_deg_about_up_axis": round(align_deg, 2),
            "up_axis": up_axis,
            "point_cloud": aligned_point_cloud_extent,
            "camera_path": extent_from_points(centers_aligned),
            # Concave-hull footprint, clipped to the same bbox as point_cloud.
            "footprint_area": footprint_area(footprint_points, up_axis),
        },
        "num_points": int(cloud.points.shape[0]),
        "num_cameras": len(centers),
        "percentile_trim": [PCT_LO, PCT_HI],
    }


def write_scene_size_txt(workspace: Path, extent: dict[str, Any], floors: int | None = None) -> Path:
    """Write scene spatial dimensions, per-floor breakdowns, and total floor area to scene_size.txt.

    ``x``/``z`` are horizontal axes, ``y`` is gravity-locked vertical height.
    Writes each floor's x_span, avg_y, z_span, and floor_area, followed by total_floor_area.
    """
    out_path = workspace / "scene_size.txt"

    # Extract global dimensions if present in the extent dict
    if "aligned" in extent and "point_cloud" in extent["aligned"]:
        x, y, z = extent["aligned"]["point_cloud"]["size"]
    else:
        x = extent.get("room_dim_x", extent.get("camera_span_x", 0.0))
        y = extent.get("height_m", 0.0)
        z = extent.get("room_dim_z", extent.get("camera_span_z", 0.0))

    # Retrieve per-floor breakdown and total area
    floors_details: list[dict[str, float]] = extent.get("floors_details", [])
    footprint_area_value = extent.get("aligned", {}).get("footprint_area", 0.0)
    floor_count = len(floors_details) if floors_details else extent.get("floors", floors)
    total_floor_area = extent.get("total_floor_area")
    if total_floor_area is None and floor_count is not None:
        total_floor_area = footprint_area_value * floor_count

    # Build structured text output
    units = extent.get("coordinate_units", "unknown")
    area_name = "area_m2" if units == "m" else "area_colmap_units2"
    total_area_name = "total_floor_area_m2" if units == "m" else "total_floor_area_colmap_units2"
    lines = [
        f"coordinate_units: {units}",
        f"x: {x:.3f}",
        f"y: {y:.3f}",
        f"z: {z:.3f}",
    ]
    if floor_count is not None:
        lines.append(f"floors: {floor_count}")
    lines.append(f"{area_name}: {footprint_area_value:.3f}")

    # Write itemized per-floor metrics
    for idx, f_info in enumerate(floors_details):
        f_idx = f_info.get("floor_index", idx)
        lines.append(f"floor_{f_idx}_x_span: {f_info['x_span']:.3f}")
        lines.append(f"floor_{f_idx}_avg_y: {f_info['avg_y']:.3f}")
        lines.append(f"floor_{f_idx}_z_span: {f_info['z_span']:.3f}")
        lines.append(f"floor_{f_idx}_area_m2: {f_info['floor_area']:.3f}")

    # Terminate with the cumulative total area
    if total_floor_area is not None:
        lines.append(f"{total_area_name}: {total_floor_area:.3f}")
    if "aligned" in extent and extent["aligned"].get("north_heading_deg") is not None:
        lines.append(f"north_heading_deg: {extent['aligned']['north_heading_deg']:.2f}")

    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def write_scene_extent_outputs(workspace: Path, transforms_path: Path | None = None) -> dict:
    """Write metric-aware extent and floor metadata without changing source geometry."""
    workspace = Path(workspace)
    depth_dir = workspace / STAGE_DIRNAME
    depth_dir.mkdir(parents=True, exist_ok=True)
    result = scene_extent(workspace, transforms_path=transforms_path)
    if result["metric_scale_available"]:
        from floor_count import estimate_floor_count

        result["floors"], _ = estimate_floor_count(workspace)
    json_path = depth_dir / "scene_extent.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_scene_size_txt(depth_dir, result)
    return result


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

    area = footprint_area(pts, up_axis=1)
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
    transforms_path = Path(args.transforms) if args.transforms else None
    result = write_scene_extent_outputs(workspace, transforms_path)
    out_path = workspace / STAGE_DIRNAME / "scene_extent.json"
    size_path = workspace / STAGE_DIRNAME / "scene_size.txt"

    units = "m" if result["metric_scale_available"] else "COLMAP units"
    if result["metric_scale_available"]:
        scale = result["metric_calibration"]["meters_per_colmap_unit"]
        print(f"[scene_extent] metric scale: {scale:.5f} m/COLMAP unit")
        calibration = result["metric_calibration"]
        print(f"[scene_extent] early motion: {calibration['steps_used']} consistent step(s), "
              f"{len(calibration['steps_rejected'])} transition(s) split the trajectory")
        for step in calibration["steps_rejected"]:
            print(f"[scene_extent] step disagreement {step['from_frame']} -> {step['to_frame']}: "
                  f"{step['meters_per_colmap_unit']:.5f} m/COLMAP unit "
                  f"({step['deviation_from_step_consensus_pct']:+.1f}% from consensus)")
    else:
        print(f"[scene_extent] WARNING: {result['metric_calibration']['reason']}; dimensions are unscaled")
    size = result["point_cloud"]["size"]
    print(f"[scene_extent] raw room extent (points, {PCT_LO}-{PCT_HI} pct): "
         f"{size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} {units}")
    aligned_size = result["aligned"]["point_cloud"]["size"]
    mode = result["aligned"].get("alignment_mode", "wall")
    print(f"[scene_extent] {mode}-aligned footprint ({result['aligned']['rotation_deg_about_up_axis']:.1f} deg "
         f"about up axis {result['aligned']['up_axis']}): "
         f"{aligned_size[0]:.2f} x {aligned_size[1]:.2f} x {aligned_size[2]:.2f} {units}")
    area_units = "m²" if result["metric_scale_available"] else "COLMAP units²"
    print(f"[scene_extent] concave-hull footprint area: {result['aligned']['footprint_area']:.2f} {area_units} "
          f"(vs. bbox {aligned_size[0] * aligned_size[2]:.2f} {area_units})")
    print(f"[scene_extent] written to {out_path} and {size_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
