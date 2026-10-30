#!/usr/bin/env python3
"""Fit one floor plane per ramp-separated area and append finite previews."""
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_3DGS_to_mesh"))
from fill_mesh_holes import export_ply_with_face_attributes, read_ply_face_flags
from fit_floors_ceilings_planes import fit_robust_plane

FLOOR_ROLE, RAMP_ROLE, CEILING_ROLE = 0, 1, 2
DEFAULT_DISTANCE_M = 0.10
SNAP_TILT_DEG = 5.0


def _floor_areas(vertices: np.ndarray, triangles: np.ndarray, roles: np.ndarray,
                 has_ramps: bool) -> list[np.ndarray]:
    floor_faces = np.flatnonzero(roles == FLOOR_ROLE)
    if not len(floor_faces):
        raise ValueError("Intermediate floor has no triangles labeled as floor")
    if not has_ramps:
        return [floor_faces]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles[floor_faces])
    labels, _, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    components = [floor_faces[labels == label] for label in np.unique(labels)]
    return sorted(components, key=lambda faces: int(faces.min()))


def _fit_area(vertices: np.ndarray, triangles: np.ndarray, face_ids: np.ndarray,
              distance_m: float) -> dict:
    corners = vertices[triangles[face_ids]]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    double_areas = np.linalg.norm(cross, axis=1)
    valid = double_areas > 1e-12
    if not valid.any():
        raise ValueError("Floor area has no non-degenerate triangles")
    points = corners[valid].mean(axis=1)
    weights = double_areas[valid] * 0.5
    normals = cross[valid] / double_areas[valid, None]
    normal, offset, support = fit_robust_plane(
        points, weights, distance_m, prefer_normal_up=True, face_normals=normals,
    )
    if normal[1] < 0:
        normal, offset = -normal, -offset
    tilt = float(np.rad2deg(np.arccos(np.clip(normal[1], -1.0, 1.0))))
    snapped = tilt < SNAP_TILT_DEG
    if snapped:
        normal = np.array([0.0, 1.0, 0.0])
        offset = -float(np.average(points[:, 1], weights=weights))
        tilt = 0.0
    return {
        "normal": normal,
        "offset": float(offset),
        "tilt_deg": tilt,
        "snapped_horizontal": snapped,
        "support_fraction": float(support),
        "area_m2": float(weights.sum()),
        "triangle_count": int(len(points)),
    }


def _plane_quad(bounds: tuple[np.ndarray, np.ndarray], normal: np.ndarray, offset: float) -> np.ndarray:
    low, high = bounds
    box = np.array([[x, y, z] for x in (low[0], high[0])
                    for y in (low[1], high[1]) for z in (low[2], high[2])])
    reference = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    axis_u = np.cross(normal, reference)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    origin = -offset * normal
    projected_u, projected_v = (box - origin) @ axis_u, (box - origin) @ axis_v
    u0, u1 = float(projected_u.min()), float(projected_u.max())
    v0, v1 = float(projected_v.min()), float(projected_v.max())
    return np.array([origin + u0 * axis_u + v0 * axis_v,
                     origin + u1 * axis_u + v0 * axis_v,
                     origin + u1 * axis_u + v1 * axis_v,
                     origin + u0 * axis_u + v1 * axis_v])


def _fit_level(input_path: Path, output_path: Path, distance_m: float,
               has_ramps: bool) -> list[dict]:
    mesh = o3d.io.read_triangle_mesh(str(input_path))
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if not len(vertices) or not len(triangles) or not np.isfinite(vertices).all():
        raise ValueError(f"Intermediate floor mesh has empty or non-finite geometry: {input_path}")
    roles = read_ply_face_flags(input_path, "surface_role")
    if roles is None:
        roles = np.zeros(len(triangles), dtype=np.uint8)
    flags = read_ply_face_flags(input_path, "is_hole_fill")
    if flags is None:
        flags = np.zeros(len(triangles), dtype=np.uint8)
    if len(roles) != len(triangles) or len(flags) != len(triangles):
        raise ValueError(f"Intermediate floor face attributes do not match its triangles: {input_path}")
    roles = np.asarray(roles, dtype=np.uint8)
    flags = np.asarray(flags, dtype=np.uint8)
    if not np.isin(roles, [FLOOR_ROLE, RAMP_ROLE, CEILING_ROLE]).all():
        raise ValueError(f"Intermediate floor contains an unknown surface role: {input_path}")

    areas = _floor_areas(vertices, triangles, roles, has_ramps=has_ramps and np.any(roles == RAMP_ROLE))
    fitted, area_ids = [], np.full(len(triangles), -1, dtype=np.int32)
    for area_id, faces in enumerate(areas):
        plane = _fit_area(vertices, triangles, faces, distance_m)
        area_ids[faces] = area_id
        plane.update({"area_id": area_id, "source_face_ids": faces})
        fitted.append(plane)

    colors = np.asarray(mesh.vertex_colors)
    has_colors = len(colors) == len(vertices)
    if has_colors:
        colors = np.rint(np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
    vertices_out, triangles_out = [vertices], [triangles]
    flags_out, roles_out = [flags], [roles]
    area_ids_out, preview_flags = [area_ids], [np.zeros(len(triangles), dtype=np.uint8)]
    colors_out = [colors] if has_colors else []
    offset = len(vertices)
    footprint_faces = np.flatnonzero(np.isin(roles, (FLOOR_ROLE, RAMP_ROLE)))
    footprint_vertices = vertices[np.unique(triangles[footprint_faces])]
    bounds = (footprint_vertices.min(axis=0), footprint_vertices.max(axis=0))
    for plane in fitted:
        quad = _plane_quad(bounds, plane["normal"], plane["offset"])
        vertices_out.append(quad)
        triangles_out.append(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32) + offset)
        flags_out.append(np.zeros(2, dtype=np.uint8))
        roles_out.append(np.full(2, FLOOR_ROLE, dtype=np.uint8))
        area_ids_out.append(np.full(2, plane["area_id"], dtype=np.int32))
        preview_flags.append(np.ones(2, dtype=np.uint8))
        if has_colors:
            colors_out.append(np.full((4, 3), [255, 0, 255], dtype=np.uint8))
        offset += 4

    if not export_ply_with_face_attributes(
        output_path, np.concatenate(vertices_out), np.concatenate(triangles_out),
        np.concatenate(colors_out) if has_colors else None,
        face_attributes={
            "is_hole_fill": ("uchar", np.concatenate(flags_out)),
            "surface_role": ("uchar", np.concatenate(roles_out)),
            "is_fitted_plane": ("uchar", np.concatenate(preview_flags)),
            "floor_area_id": ("int", np.concatenate(area_ids_out)),
        },
    ):
        raise OSError(f"Could not write fitted floor-level PLY: {output_path}")
    return [{
        "area_id": plane["area_id"],
        "normal_unit": plane["normal"].tolist(),
        "plane_equation": [*plane["normal"].tolist(), plane["offset"]],
        "tilt_deg": plane["tilt_deg"],
        "snapped_horizontal": plane["snapped_horizontal"],
        "support_fraction": plane["support_fraction"],
        "area_m2": plane["area_m2"],
        "triangle_count": plane["triangle_count"],
    } for plane in fitted]


def fit_floor_level_planes(instances_dir: Path, distance_m: float = DEFAULT_DISTANCE_M) -> Path:
    if not np.isfinite(distance_m) or distance_m <= 0:
        raise ValueError("Plane-fit distance must be finite and positive")
    output_dir = instances_dir / "intermediate_floor_levels"
    manifest_path = output_dir / "floor_level_intermediate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("levels"), list):
        raise ValueError(f"Invalid floor-level manifest: {manifest_path}")

    updated_levels, output_names = [], []
    with tempfile.TemporaryDirectory(prefix=".fit_floor_levels_", dir=output_dir) as temp_name:
        temp_dir = Path(temp_name)
        for source_level in manifest["levels"]:
            if not isinstance(source_level, dict):
                raise ValueError("Invalid floor-level manifest entry")
            source_name = source_level.get("output_file")
            if not isinstance(source_name, str) or not re.fullmatch(r"floor_level_\d{3}_intermediate\.ply", source_name):
                raise ValueError(f"Invalid intermediate floor filename: {source_name!r}")
            level = int(source_level.get("level", -1))
            output_name = f"floor_level_{level:03d}_fitted.ply"
            planes = _fit_level(output_dir / source_name, temp_dir / output_name, distance_m,
                                has_ramps=bool(source_level.get("ramps")))
            updated = dict(source_level)
            updated["fitted_output_file"] = output_name
            updated["floor_areas"] = planes
            updated["num_fitted_planes"] = len(planes)
            updated["num_fitted_plane_triangles"] = 2 * len(planes)
            updated["num_fitted_output_triangles"] = int(updated.get("num_triangles", 0)) + 2 * len(planes)
            updated_levels.append(updated)
            output_names.append(output_name)

        manifest["levels"] = updated_levels
        manifest["floor_plane_fit"] = {
            "algorithm": "area_weighted_robust_plane_per_ramp_separated_floor_area",
            "distance_threshold_m": distance_m,
            "horizontal_snap_below_deg": SNAP_TILT_DEG,
            "plane_preview_color_rgb": [255, 0, 255],
            "plane_preview_face_property": "is_fitted_plane",
            "floor_area_face_property": "floor_area_id",
        }
        (temp_dir / manifest_path.name).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for filename in output_names:
            os.replace(temp_dir / filename, output_dir / filename)
        os.replace(temp_dir / manifest_path.name, manifest_path)

    keep = set(output_names)
    for path in output_dir.glob("floor_level_*_fitted.ply"):
        if path.name not in keep:
            path.unlink()
    for level in updated_levels:
        print(f"Level {level['level']:03d}: fitted {len(level['floor_areas'])} floor area planes -> "
              f"{level['fitted_output_file']}")
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path, required=True,
                        help="Stage 06 output directory containing cleaned intermediate floor levels")
    parser.add_argument("--distance-m", type=float, default=DEFAULT_DISTANCE_M,
                        help="Robust plane support distance in metres")
    args = parser.parse_args(argv)
    try:
        fit_floor_level_planes(args.instances_dir, args.distance_m)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: Could not fit floor-level planes: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
