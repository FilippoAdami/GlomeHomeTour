#!/usr/bin/env python3
"""Write one plane-bounded slice of the sealed scene mesh per fitted floor."""
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

FLOOR_ROLE, RAMP_ROLE, CEILING_ROLE, OTHER_ROLE = 0, 1, 2, 255
PLANE_CLEARANCE_M = 0.20
CHUNK_SIZE = 100_000


def _unit_plane(equation: list[float], description: str) -> np.ndarray:
    plane = np.asarray(equation, dtype=np.float64)
    if plane.shape != (4,) or not np.isfinite(plane).all():
        raise ValueError(f"Invalid {description} plane equation")
    length = float(np.linalg.norm(plane[:3]))
    if length <= 1e-12:
        raise ValueError(f"{description.capitalize()} plane has a zero-length normal")
    plane /= length
    if abs(plane[1]) <= 1e-8:
        raise ValueError(f"{description.capitalize()} plane is vertical and cannot bound a floor slice")
    if plane[1] < 0:
        plane = -plane
    return plane


def _read_preview(path: Path, id_property: str) -> dict:
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    fitted = read_ply_face_flags(path, "is_fitted_plane")
    roles = read_ply_face_flags(path, "surface_role")
    plane_ids = read_ply_face_flags(path, id_property)
    hole_fill = read_ply_face_flags(path, "is_hole_fill")
    if (not len(vertices) or not len(triangles) or fitted is None or roles is None
            or plane_ids is None or hole_fill is None
            or any(len(values) != len(triangles) for values in (fitted, roles, plane_ids, hole_fill))):
        raise ValueError(f"Fitted plane PLY is missing preview attributes: {path}")
    face_ids = np.flatnonzero(fitted == 1)
    if not len(face_ids):
        raise ValueError(f"Fitted plane PLY has no preview triangles: {path}")
    used, inverse = np.unique(triangles[face_ids], return_inverse=True)
    colors = np.asarray(mesh.vertex_colors)
    preview_colors = None
    if len(colors) == len(vertices):
        preview_colors = np.rint(np.clip(colors[used], 0.0, 1.0) * 255).astype(np.uint8)
    return {
        "vertices": vertices[used],
        "triangles": inverse.reshape(-1, 3),
        "colors": preview_colors,
        "roles": roles[face_ids].astype(np.uint8),
        "hole_fill": hole_fill[face_ids].astype(np.uint8),
        "floor_area_ids": plane_ids[face_ids].astype(np.int32) if id_property == "floor_area_id"
                          else np.full(len(face_ids), -1, dtype=np.int32),
        "ceiling_group_ids": plane_ids[face_ids].astype(np.int32) if id_property == "ceiling_group_id"
                             else np.full(len(face_ids), -1, dtype=np.int32),
        "center_y": float(vertices[used, 1].mean()),
    }


def _class_roles(path: Path) -> tuple[dict[int, int], list[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    classes = data.get("classes", data.get("classifications")) if isinstance(data, dict) else None
    if not isinstance(classes, dict):
        raise ValueError(f"Class metadata has no classes mapping: {path}")
    role_ids: dict[int, int] = {}
    for class_id, info in classes.items():
        try:
            class_id = int(class_id)
        except (TypeError, ValueError):
            continue
        if not isinstance(info, dict):
            continue
        role = str(info.get("structural_role", "")).lower()
        role_ids[class_id] = (CEILING_ROLE if "ceiling" in role else
                              FLOOR_ROLE if "floor" in role else
                              RAMP_ROLE if "ramp" in role else OTHER_ROLE)
    ceiling_ids = [class_id for class_id, role in role_ids.items() if role == CEILING_ROLE]
    return role_ids, ceiling_ids


def _triangle_centers(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    centers = np.empty((len(triangles), 3), dtype=np.float64)
    for start in range(0, len(triangles), CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, len(triangles))
        centers[start:stop] = vertices[triangles[start:stop]].mean(axis=1)
    return centers


def _inside_floor_cut(vertices: np.ndarray, triangles: np.ndarray, centers: np.ndarray,
                      floor_planes: list[np.ndarray]) -> np.ndarray:
    # With ramp-separated areas, use the fitted area plane nearest each face centroid.
    nearest_distance = np.full(len(triangles), np.inf)
    nearest_plane = np.zeros(len(triangles), dtype=np.int32)
    for plane_id, plane in enumerate(floor_planes):
        signed = centers @ plane[:3] + plane[3]
        nearer = np.abs(signed) < nearest_distance
        nearest_distance[nearer] = np.abs(signed[nearer])
        nearest_plane[nearer] = plane_id

    shifted_offsets = np.asarray([plane[3] + PLANE_CLEARANCE_M for plane in floor_planes])
    keep = np.zeros(len(triangles), dtype=bool)
    for start in range(0, len(triangles), CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, len(triangles))
        points = vertices[triangles[start:stop]]
        normals = np.asarray([floor_planes[index][:3] for index in nearest_plane[start:stop]])
        signed = np.einsum("fvc,fc->fv", points, normals)
        signed += shifted_offsets[nearest_plane[start:stop], None]
        keep[start:stop] = np.all(signed >= -1e-9, axis=1)
    return keep


def _inside_upper_floor_cut(vertices: np.ndarray, triangles: np.ndarray,
                            centers: np.ndarray, floor_planes: list[np.ndarray]) -> np.ndarray:
    """Keep faces below the next fitted floor, independent of ceiling geometry."""
    if not floor_planes:
        return np.ones(len(triangles), dtype=bool)
    distances = np.column_stack([np.abs(centers @ plane[:3] + plane[3])
                                 for plane in floor_planes])
    nearest = np.argmin(distances, axis=1)
    normals = np.asarray([plane[:3] for plane in floor_planes])[nearest]
    offsets = np.asarray([plane[3] + PLANE_CLEARANCE_M for plane in floor_planes])[nearest]
    points = vertices[triangles]
    signed = np.einsum("fvc,fc->fv", points, normals) + offsets[:, None]
    return np.all(signed <= 1e-9, axis=1)


def _append_preview(parts: dict[str, list], preview: dict) -> None:
    if not len(preview["triangles"]):
        return
    offset = sum(len(vertices) for vertices in parts["vertices"])
    parts["vertices"].append(preview["vertices"])
    parts["triangles"].append(preview["triangles"] + offset)
    parts["hole_fill"].append(preview["hole_fill"])
    parts["roles"].append(preview["roles"])
    parts["fitted"].append(np.ones(len(preview["triangles"]), dtype=np.uint8))
    parts["source_class_ids"].append(np.full(len(preview["triangles"]), -1, dtype=np.int32))
    parts["floor_area_ids"].append(preview["floor_area_ids"])
    parts["ceiling_group_ids"].append(preview["ceiling_group_ids"])
    if preview["colors"] is None:
        parts["colors"].append(np.full((len(preview["vertices"]), 3), 128, dtype=np.uint8))
    else:
        parts["colors"].append(preview["colors"])


def slice_scene_by_floor(mesh_path: Path, labels_path: Path, class_metadata_path: Path,
                         instances_dir: Path) -> Path:
    level_dir = instances_dir / "intermediate_floor_levels"
    floor_manifest_path = level_dir / "floor_level_intermediate_manifest.json"
    floors = json.loads(floor_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(floors, dict) or not isinstance(floors.get("levels"), list) or not floors["levels"]:
        raise ValueError(f"Invalid or empty floor-level manifest: {floor_manifest_path}")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    labels = np.load(labels_path)
    if (not len(vertices) or not len(triangles) or not np.isfinite(vertices).all()
            or triangles.min() < 0 or triangles.max() >= len(vertices)):
        raise ValueError(f"Original sealed mesh is empty, non-finite, or has invalid faces: {mesh_path}")
    if labels.shape != (len(triangles),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Triangle labels must be one integer class ID per source face: {labels_path}")
    if labels.min() < np.iinfo(np.int32).min or labels.max() > np.iinfo(np.int32).max:
        raise ValueError("Source class IDs do not fit the output PLY integer property")
    role_ids, ceiling_class_ids = _class_roles(class_metadata_path)
    ceiling_faces = np.isin(labels, ceiling_class_ids)
    source_roles = np.full(len(triangles), OTHER_ROLE, dtype=np.uint8)
    for class_id, role in role_ids.items():
        source_roles[labels == class_id] = role
    hole_fill = read_ply_face_flags(mesh_path, "is_hole_fill")
    if hole_fill is None:
        hole_fill = np.zeros(len(triangles), dtype=np.uint8)
    elif len(hole_fill) != len(triangles):
        raise ValueError(f"Source hole-fill flags do not match mesh triangles: {mesh_path}")
    centers = _triangle_centers(vertices, triangles)
    source_colors = np.asarray(mesh.vertex_colors)
    if len(source_colors) == len(vertices):
        source_colors = np.rint(np.clip(source_colors, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        source_colors = None

    floor_records = []
    for level in floors["levels"]:
        if not isinstance(level, dict) or not isinstance(level.get("floor_areas"), list) or not level["floor_areas"]:
            raise ValueError("Every floor level must have fitted floor areas")
        level_id = level.get("level")
        if not isinstance(level_id, int) or level_id < 0:
            raise ValueError(f"Invalid floor level ID: {level_id!r}")
        filename = level.get("fitted_output_file")
        if not isinstance(filename, str) or not re.fullmatch(r"floor_level_\d{3}_fitted\.ply", filename):
            raise ValueError(f"Floor level {level_id} has no valid fitted PLY")
        preview = _read_preview(level_dir / filename, "floor_area_id")
        equations = [_unit_plane(area.get("plane_equation", []), f"floor level {level_id}")
                     for area in level["floor_areas"]]
        floor_y = float(level.get("floor_triangle_centroid_y_median",
                                  level.get("elevation_y_median", preview["center_y"])))
        floor_records.append({"level": level_id, "floor_y": floor_y, "planes": equations,
                              "preview": preview})

    floor_records.sort(key=lambda item: (item["floor_y"], item["level"]))
    wall_manifest_path = instances_dir / "wall_planes_manifest.json"
    wall_manifest = json.loads(wall_manifest_path.read_text(encoding="utf-8")) if wall_manifest_path.is_file() else {}
    wall_bounds = wall_manifest.get("wall_vertical_bounds_m")
    wall_top = float(wall_bounds[1]) if isinstance(wall_bounds, list) and len(wall_bounds) == 2 else None
    output_manifest_path = level_dir / "floor_level_scene_slices_manifest.json"
    output_names, slice_records = [], []
    with tempfile.TemporaryDirectory(prefix=".floor_scene_slices_", dir=level_dir) as temp_name:
        temp_dir = Path(temp_name)
        for floor_index, floor in enumerate(floor_records):
            level_id = floor["level"]
            next_floor = floor_records[floor_index + 1] if floor_index + 1 < len(floor_records) else None
            floor_inside = _inside_floor_cut(vertices, triangles, centers, floor["planes"])
            upper_planes = next_floor["planes"] if next_floor is not None else (
                [np.array([0.0, 1.0, 0.0, -wall_top], dtype=np.float64)]
                if wall_top is not None else []
            )
            upper_inside = _inside_upper_floor_cut(vertices, triangles, centers, upper_planes)
            keep = floor_inside & upper_inside & ~ceiling_faces
            source_face_ids = np.flatnonzero(keep)
            if not len(source_face_ids):
                raise ValueError(f"No source scene triangles remain for floor level {level_id}")

            used, inverse = np.unique(triangles[source_face_ids], return_inverse=True)
            parts = {
                "vertices": [vertices[used]],
                "triangles": [inverse.reshape(-1, 3)],
                "hole_fill": [hole_fill[source_face_ids].astype(np.uint8)],
                "roles": [source_roles[source_face_ids]],
                "fitted": [np.zeros(len(source_face_ids), dtype=np.uint8)],
                "source_class_ids": [labels[source_face_ids].astype(np.int32)],
                "floor_area_ids": [np.full(len(source_face_ids), -1, dtype=np.int32)],
                "ceiling_group_ids": [np.full(len(source_face_ids), -1, dtype=np.int32)],
                "colors": [source_colors[used] if source_colors is not None else
                           np.full((len(used), 3), 128, dtype=np.uint8)],
            }
            _append_preview(parts, floor["preview"])

            output_path = temp_dir / f"floor_level_{level_id:03d}_scene_slice.ply"
            vertices_out = np.concatenate(parts["vertices"])
            triangles_out = np.concatenate(parts["triangles"])
            colors_out = np.concatenate(parts["colors"]) if parts["colors"] else None
            if not export_ply_with_face_attributes(
                output_path, vertices_out, triangles_out, colors_out,
                face_attributes={
                    "is_hole_fill": ("uchar", np.concatenate(parts["hole_fill"])),
                    "surface_role": ("uchar", np.concatenate(parts["roles"])),
                    "is_fitted_plane": ("uchar", np.concatenate(parts["fitted"])),
                    "source_class_id": ("int", np.concatenate(parts["source_class_ids"])),
                    "floor_area_id": ("int", np.concatenate(parts["floor_area_ids"])),
                    "ceiling_group_id": ("int", np.concatenate(parts["ceiling_group_ids"])),
                },
            ):
                raise OSError(f"Could not write floor scene slice: {output_path}")
            output_names.append(output_path.name)
            slice_records.append({
                "level": level_id,
                "output_file": output_path.name,
                "source_mesh": str(mesh_path),
                "source_triangles": int(len(triangles)),
                "retained_source_triangles": int(len(source_face_ids)),
                "removed_ceiling_class_triangles": int(np.count_nonzero(ceiling_faces & ~keep)),
                "retained_ceiling_class_triangles": int(np.count_nonzero(ceiling_faces & keep)),
                "removed_below_floor_cut_triangles": int(np.count_nonzero(~floor_inside & ~keep)),
                "removed_above_upper_boundary_triangles": int(np.count_nonzero(~upper_inside & ~keep)),
                "fitted_plane_triangles": int(len(triangles_out) - len(source_face_ids)),
                "floor_plane_equations": [plane.tolist() for plane in floor["planes"]],
                "upper_boundary_source": "next_floor_level" if next_floor is not None else
                                         "wall_enclosure_top" if wall_top is not None else None,
                "upper_boundary_plane_equations": [plane.tolist() for plane in upper_planes],
            })

        manifest = {
            "source_mesh": str(mesh_path),
            "source_triangle_labels": str(labels_path),
            "plane_clearance_m": PLANE_CLEARANCE_M,
            "triangle_selection": "Keep non-ceiling source triangles above each fitted floor shifted down 0.20 m and below the next fitted floor (or the wall enclosure top) shifted down 0.20 m. Append only fitted floor plane previews.",
            "ceiling_class_ids": ceiling_class_ids,
            "levels": slice_records,
        }
        (temp_dir / output_manifest_path.name).write_text(json.dumps(manifest, indent=2) + "\n",
                                                           encoding="utf-8")
        for filename in output_names:
            os.replace(temp_dir / filename, level_dir / filename)
        os.replace(temp_dir / output_manifest_path.name, output_manifest_path)

    keep_names = set(output_names)
    for path in level_dir.glob("floor_level_*_scene_slice.ply"):
        if path.name not in keep_names:
            path.unlink()
    for record in slice_records:
        print(f"Level {record['level']:03d}: retained {record['retained_source_triangles']:,} source triangles, "
              f"added {record['fitted_plane_triangles']} fitted plane triangles -> {record['output_file']}")
    return output_manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path, required=True,
                        help="Stage 06 output directory containing fitted floor and ceiling planes")
    parser.add_argument("--mesh-path", type=Path, required=True,
                        help="Original sealed scene mesh used by Stage 05")
    parser.add_argument("--labels-path", type=Path, required=True,
                        help="Face-order aligned triangle_class_labels.npy")
    parser.add_argument("--class-metadata", type=Path, required=True,
                        help="Stage 05 class_metadata.json")
    args = parser.parse_args(argv)
    try:
        slice_scene_by_floor(args.mesh_path, args.labels_path, args.class_metadata, args.instances_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: Could not slice the sealed scene by floor: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
