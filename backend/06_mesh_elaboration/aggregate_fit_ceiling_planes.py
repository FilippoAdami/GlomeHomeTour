#!/usr/bin/env python3
"""Aggregate co-planar ceiling instances and append fitted plane previews."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
import open3d as o3d

STAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE_DIR))
sys.path.insert(0, str(STAGE_DIR.parent / "04_3DGS_to_mesh"))
from fill_mesh_holes import export_ply_with_face_attributes, read_ply_face_flags
from fit_floor_level_planes import _plane_quad
from fit_floors_ceilings_planes import fit_robust_plane
from planar_outlier_filter import FIT_DISTANCE_M, KEEP_DISTANCE_M, filter_planar_outliers

PLANE_FIT_DISTANCE_M = FIT_DISTANCE_M
GROUP_PLANE_DISTANCE_M = KEEP_DISTANCE_M
MIN_GROUP_SUPPORT_FRACTION = 0.80
MIN_MEMBER_SUPPORT_FRACTION = 0.80
SNAP_TILT_DEG = 5.0
PREVIEW_COLOR = np.array([0, 255, 255], dtype=np.uint8)


def _fit_ceiling_plane(vertices: np.ndarray, triangles: np.ndarray) -> dict:
    corners = vertices[triangles]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1e-12
    if not valid.any():
        raise ValueError("Ceiling aggregate has no non-degenerate triangles")
    points = corners[valid].mean(axis=1)
    weights = double_area[valid] * 0.5
    normals = cross[valid] / double_area[valid, None]
    normal, offset, support = fit_robust_plane(
        points, weights, PLANE_FIT_DISTANCE_M, prefer_normal_up=True, face_normals=normals,
    )
    if normal[1] < 0:
        normal, offset = -normal, -offset
    supported = ((np.abs(points @ normal + offset) <= PLANE_FIT_DISTANCE_M) &
                 (np.abs(normals @ normal) >= np.cos(np.deg2rad(30.0))))
    if not supported.any():
        raise ValueError("Could not find supported ceiling plane")
    tilt = float(np.rad2deg(np.arccos(np.clip(normal[1], -1.0, 1.0))))
    snapped = tilt < SNAP_TILT_DEG
    if snapped:
        normal = np.array([0.0, 1.0, 0.0])
        offset = -float(np.average(points[supported, 1], weights=weights[supported]))
        tilt = 0.0
    return {"normal": normal, "offset": float(offset), "tilt_deg": tilt,
            "snapped_horizontal": snapped, "support_fraction": float(support),
            "area_m2": float(weights.sum()), "triangle_count": int(valid.sum())}


def _read_mesh(path: Path, filename: str) -> dict:
    if Path(filename).name != filename or not re.fullmatch(r"ceiling_inst\d+\.ply", filename):
        raise ValueError(f"Invalid ceiling instance filename: {filename!r}")
    mesh_path = path / filename
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Ceiling instance listed in the manifest is missing: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if not len(vertices) or not len(triangles) or not np.isfinite(vertices).all():
        raise ValueError(f"Ceiling instance has empty or non-finite geometry: {mesh_path}")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError(f"Ceiling instance has invalid triangle indices: {mesh_path}")
    colors = np.asarray(mesh.vertex_colors)
    if len(colors) == len(vertices):
        colors = np.rint(np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
    elif not len(colors):
        colors = None
    else:
        raise ValueError(f"Ceiling instance has incomplete vertex colors: {mesh_path}")
    flags = read_ply_face_flags(mesh_path, "is_hole_fill")
    if flags is None:
        flags = np.zeros(len(triangles), dtype=np.uint8)
    elif len(flags) != len(triangles):
        raise ValueError(f"Ceiling face flags do not match its triangles: {mesh_path}")
    keep, cleanup = filter_planar_outliers(vertices, triangles)
    triangles, flags = triangles[keep], flags[keep]
    if not len(triangles):
        raise ValueError(f"Ceiling instance has no triangles after plane filtering: {mesh_path}")
    used, inverse = np.unique(triangles, return_inverse=True)
    vertices, triangles = vertices[used], inverse.reshape(-1, 3)
    if colors is not None:
        colors = colors[used]
    return {
        "file_name": filename,
        "vertices": vertices,
        "triangles": triangles,
        "colors": colors,
        "flags": flags,
        "cleanup": cleanup,
    }


def _member_plane_support(item: dict, normal: np.ndarray, offset: float) -> float:
    corners = item["vertices"][item["triangles"]]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1e-12
    if not valid.any():
        return 0.0
    points = corners[valid].mean(axis=1)
    areas = double_area[valid] * 0.5
    supported = np.abs(points @ normal + offset) <= GROUP_PLANE_DISTANCE_M
    return float(areas[supported].sum() / areas.sum())


def _joint_coplanar(group: list[dict], candidate: list[dict]) -> bool:
    if any(item["floor_level"] != group[0]["floor_level"] for item in candidate):
        return False
    merged = _merge(group + candidate)
    try:
        plane = _fit_ceiling_plane(merged["vertices"], merged["triangles"])
    except ValueError:
        return False
    if plane["support_fraction"] < MIN_GROUP_SUPPORT_FRACTION:
        return False
    return all(_member_plane_support(item, plane["normal"], plane["offset"])
               >= MIN_MEMBER_SUPPORT_FRACTION for item in group + candidate)


def _ceiling_groups(items: list[dict]) -> list[list[dict]]:
    groups = [[item] for item in items]
    while True:
        merged_any = False
        for first in range(len(groups)):
            for second in range(first + 1, len(groups)):
                if _joint_coplanar(groups[first], groups[second]):
                    groups[first].extend(groups.pop(second))
                    merged_any = True
                    break
            if merged_any:
                break
        if not merged_any:
            break
    return groups


def _merge(group: list[dict]) -> dict:
    vertices, triangles, flags, colors = [], [], [], []
    have_colors = any(item["colors"] is not None for item in group)
    offset = 0
    for item in group:
        vertices.append(item["vertices"])
        triangles.append(item["triangles"] + offset)
        flags.append(item["flags"])
        if have_colors:
            colors.append(item["colors"] if item["colors"] is not None else
                          np.full((len(item["vertices"]), 3), 128, dtype=np.uint8))
        offset += len(item["vertices"])
    return {
        "vertices": np.concatenate(vertices),
        "triangles": np.concatenate(triangles),
        "flags": np.concatenate(flags),
        "colors": np.concatenate(colors) if have_colors else None,
    }


def _floor_preview_quad(instances_dir: Path, level: dict) -> np.ndarray:
    filename = level.get("fitted_output_file")
    if not isinstance(filename, str) or not re.fullmatch(r"floor_level_\d{3}_fitted\.ply", filename):
        raise ValueError(f"Floor level {level.get('level')} has no valid fitted plane output")
    path = instances_dir / "intermediate_floor_levels" / filename
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    preview = read_ply_face_flags(path, "is_fitted_plane")
    area_ids = read_ply_face_flags(path, "floor_area_id")
    if preview is None or area_ids is None or len(preview) != len(triangles) or len(area_ids) != len(triangles):
        raise ValueError(f"Fitted floor mesh is missing plane preview attributes: {path}")
    area_id = int(level.get("floor_areas", [{}])[0].get("area_id", 0))
    faces = np.flatnonzero((preview == 1) & (area_ids == area_id))
    point_ids = np.unique(triangles[faces]) if len(faces) else np.empty(0, dtype=np.int64)
    if len(point_ids) != 4:
        raise ValueError(f"Fitted floor plane {area_id} does not have a four-corner preview: {path}")
    return vertices[point_ids]


def _same_dimensions_quad(floor_quad: np.ndarray, normal: np.ndarray, offset: float) -> tuple[np.ndarray, list[float]]:
    floor_u, floor_v = floor_quad[1] - floor_quad[0], floor_quad[3] - floor_quad[0]
    lengths = [float(np.linalg.norm(floor_u)), float(np.linalg.norm(floor_v))]
    if min(lengths) <= 1e-9:
        raise ValueError("Fitted floor plane has a zero-length side")
    tangent_u = floor_u - normal * float(floor_u @ normal)
    if np.linalg.norm(tangent_u) <= 1e-9:
        tangent_u = floor_v - normal * float(floor_v @ normal)
    if np.linalg.norm(tangent_u) <= 1e-9:
        raise ValueError("Cannot transfer floor dimensions to a parallel vertical ceiling plane")
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.cross(normal, tangent_u)
    if tangent_v @ floor_v < 0:
        tangent_v = -tangent_v
    center = floor_quad.mean(axis=0)
    center -= (float(center @ normal) + offset) * normal
    half_u, half_v = lengths[0] * 0.5, lengths[1] * 0.5
    quad = np.array([
        center - half_u * tangent_u - half_v * tangent_v,
        center + half_u * tangent_u - half_v * tangent_v,
        center + half_u * tangent_u + half_v * tangent_v,
        center - half_u * tangent_u + half_v * tangent_v,
    ])
    return quad, lengths


def _write_group(output_path: Path, merged: dict, quad: np.ndarray, group_id: int) -> None:
    vertices, triangles = merged["vertices"], merged["triangles"]
    offset = len(vertices)
    triangles_out = np.concatenate((triangles, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32) + offset))
    flags = np.concatenate((merged["flags"], np.zeros(2, dtype=np.uint8)))
    roles = np.full(len(triangles_out), 2, dtype=np.uint8)
    fitted = np.r_[np.zeros(len(triangles), dtype=np.uint8), np.ones(2, dtype=np.uint8)]
    colors = merged["colors"]
    if colors is not None:
        colors = np.concatenate((colors, np.tile(PREVIEW_COLOR, (4, 1))))
    if not export_ply_with_face_attributes(
        output_path, np.concatenate((vertices, quad)), triangles_out, colors,
        face_attributes={
            "is_hole_fill": ("uchar", flags),
            "surface_role": ("uchar", roles),
            "is_fitted_plane": ("uchar", fitted),
            "ceiling_group_id": ("int", np.full(len(triangles_out), group_id, dtype=np.int32)),
        },
    ):
        raise OSError(f"Could not write fitted ceiling group: {output_path}")


def aggregate_fit_ceiling_planes(instances_dir: Path) -> Path:
    structural_path = instances_dir / "structural_instances_manifest.json"
    level_dir = instances_dir / "intermediate_floor_levels"
    floor_manifest_path = level_dir / "floor_level_intermediate_manifest.json"
    source = json.loads(structural_path.read_text(encoding="utf-8"))
    floor_manifest = json.loads(floor_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or not isinstance(source.get("instances"), list):
        raise ValueError(f"Invalid structural instance manifest: {structural_path}")
    if not isinstance(floor_manifest, dict) or not isinstance(floor_manifest.get("levels"), list):
        raise ValueError(f"Invalid floor-level manifest: {floor_manifest_path}")

    levels_by_id, assignments = {}, {}
    assigned_filenames = set()
    for level in floor_manifest["levels"]:
        if not isinstance(level, dict) or not isinstance(level.get("ceilings", []), list):
            raise ValueError("Invalid ceiling association in floor-level manifest")
        level_id = level.get("level")
        if not isinstance(level_id, int) or level_id < 0:
            raise ValueError(f"Invalid floor level id: {level_id!r}")
        if level_id in levels_by_id:
            raise ValueError(f"Duplicate floor level id: {level_id}")
        levels_by_id[level_id] = level
        for ceiling in level.get("ceilings", []):
            if not isinstance(ceiling, dict):
                raise ValueError(f"Invalid ceiling association for level {level_id}")
            filename = ceiling.get("file_name")
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError(f"Invalid ceiling filename for level {level_id}: {filename!r}")
            assigned_filenames.add(filename)
            assignments.setdefault(filename, set()).add(level_id)

    if any(not isinstance(entry, dict) for entry in source["instances"]):
        raise ValueError(f"Invalid instance entry in {structural_path}")
    ceiling_entries = [entry for entry in source["instances"]
                       if entry.get("role") == "ceiling"]
    filenames = [entry.get("file_name") for entry in ceiling_entries]
    if any(not isinstance(filename, str) for filename in filenames):
        raise ValueError("Ceiling instance has no valid filename")
    if len(set(filenames)) != len(filenames):
        raise ValueError("Duplicate ceiling PLY in structural instance manifest")
    if not assigned_filenames.issubset(set(filenames)):
        raise ValueError("Floor-level manifest references a ceiling missing from the structural manifest")
    items = []
    for entry in sorted(ceiling_entries, key=lambda value: str(value.get("file_name", ""))):
        filename = entry.get("file_name")
        if not isinstance(filename, str):
            raise ValueError("Ceiling instance has no filename")
        mesh_data = _read_mesh(instances_dir, filename)
        floor_levels = sorted(assignments.get(filename, ())) or [None]
        for floor_level in floor_levels:
            items.append({**mesh_data, "floor_level": floor_level})

    items.sort(key=lambda item: (item["floor_level"] is None,
                                 -1 if item["floor_level"] is None else item["floor_level"],
                                 item["file_name"]))
    groups = _ceiling_groups(items)
    output_manifest_path = level_dir / "ceiling_plane_groups_manifest.json"
    output_groups, output_names = [], []
    cleanup_by_instance = {item["file_name"]: item["cleanup"] for item in items}
    with tempfile.TemporaryDirectory(prefix=".ceiling_planes_", dir=level_dir) as temp_name:
        temp_dir = Path(temp_name)
        for group_id, group in enumerate(groups):
            merged = _merge(group)
            plane = _fit_ceiling_plane(merged["vertices"], merged["triangles"])
            if group[0]["floor_level"] is not None:
                level = levels_by_id[group[0]["floor_level"]]
                floor_quad = _floor_preview_quad(instances_dir, level)
                quad, dimensions = _same_dimensions_quad(floor_quad, plane["normal"], plane["offset"])
                dimension_source = level["fitted_output_file"]
            else:
                bounds = (merged["vertices"].min(axis=0), merged["vertices"].max(axis=0))
                quad = _plane_quad(bounds, plane["normal"], plane["offset"])
                dimensions = [float(np.linalg.norm(quad[1] - quad[0])),
                              float(np.linalg.norm(quad[3] - quad[0]))]
                dimension_source = "ceiling_group_bounds"
            filename = f"ceiling_plane_{group_id:03d}_fitted.ply"
            _write_group(temp_dir / filename, merged, quad, group_id)
            output_groups.append({
                "group_id": group_id,
                "source_ceiling_instances": sorted({item["file_name"] for item in group}),
                "member_plane_support_fraction": {
                    item["file_name"]: _member_plane_support(item, plane["normal"], plane["offset"])
                    for item in group
                },
                "associated_floor_level": group[0]["floor_level"],
                "fitted_output_file": filename,
                "plane_equation": [*plane["normal"].tolist(), float(plane["offset"])],
                "normal_unit": plane["normal"].tolist(),
                "tilt_deg": plane["tilt_deg"],
                "snapped_horizontal": plane["snapped_horizontal"],
                "support_fraction": plane["support_fraction"],
                "triangle_count": int(len(merged["triangles"])),
                "fitted_plane_triangle_count": 2,
                "dimensions_m": dimensions,
                "dimensions_source": dimension_source,
            })
            output_names.append(filename)

        manifest = {
            "algorithm": "iterative_joint_plane_ceiling_groups",
            "coplanar_tolerance": {
                "joint_fit_distance_m": PLANE_FIT_DISTANCE_M,
                "joint_fit_minimum_support_fraction": MIN_GROUP_SUPPORT_FRACTION,
                "each_instance_minimum_area_within_distance_fraction": MIN_MEMBER_SUPPORT_FRACTION,
                "member_plane_distance_m": GROUP_PLANE_DISTANCE_M,
            },
            "source_ceiling_cleanup": {
                "method": "shared dominant-plane outlier filter",
                "plane_fit_distance_m": FIT_DISTANCE_M,
                "maximum_kept_plane_distance_m": KEEP_DISTANCE_M,
                "instances": cleanup_by_instance,
            },
            "horizontal_snap_below_deg": SNAP_TILT_DEG,
            "preview_face_property": "is_fitted_plane",
            "surface_role_value": 2,
            "groups": output_groups,
        }
        (temp_dir / output_manifest_path.name).write_text(json.dumps(manifest, indent=2) + "\n",
                                                           encoding="utf-8")
        for filename in output_names:
            os.replace(temp_dir / filename, level_dir / filename)
        os.replace(temp_dir / output_manifest_path.name, output_manifest_path)

    keep = set(output_names)
    for path in level_dir.glob("ceiling_plane_*_fitted.ply"):
        if path.name not in keep:
            path.unlink()
    print(f"Aggregated {len(filenames)} ceiling instances into {len(groups)} co-planar fitted groups: {level_dir}")
    return output_manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path, required=True,
                        help="Stage 06 output directory containing ceiling instances and fitted floor levels")
    args = parser.parse_args(argv)
    try:
        aggregate_fit_ceiling_planes(args.instances_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: Could not aggregate and fit ceiling planes: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
