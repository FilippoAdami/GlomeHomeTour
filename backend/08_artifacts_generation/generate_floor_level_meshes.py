#!/usr/bin/env python3
"""Cut aligned meshes at each floor's next ceiling and grey lower geometry."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree, Delaunay, QhullError


CLEARANCE_M = 0.20
OWNERSHIP_PROTECTION_DEPTH_M = 0.70
FLOOR_GRAY_OFFSET_M = 0.15
MAX_TRIANGLE_ASPECT_RATIO = 20.0
NEAR_CEILING_TRIANGLE_DEPTH_M = 1.0
MAX_NEAR_CEILING_TRIANGLE_ASPECT_RATIO = 5.0
PLANE_MATCH_TOLERANCE_M = 2e-5
SLAB_PLANE_TOLERANCE_M = 0.05
SEALED_SUPPORT_PADDING_M = 0.20
DEFAULT_ALIGNMENT_DIR = Path(__file__).resolve().parents[1] / "current_scene" / "multi_room_alignment"


def _read_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path), mmap="r", known_list_len={"face": {"vertex_indices": 3}})
    vertex_data = ply["vertex"].data
    vertices = np.column_stack([vertex_data[name] for name in ("x", "y", "z")]).astype(np.float32)
    face_data = np.asarray(ply["face"].data["vertex_indices"])
    if face_data.ndim != 2 or face_data.shape[1] != 3:
        raise ValueError(f"Expected triangle faces in {path}")
    faces = face_data.astype(np.int32, copy=True)
    if not len(faces) or not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"Mesh has no valid triangles: {path}")
    color_names = ("red", "green", "blue")
    if not all(name in vertex_data.dtype.names for name in color_names):
        colors = np.full((len(vertices), 4), 200, dtype=np.uint8)
        colors[:, 3] = 255
    else:
        channels = [vertex_data[name] for name in color_names]
        alpha = vertex_data["alpha"] if "alpha" in vertex_data.dtype.names else np.full(len(vertices), 255)
        colors = np.column_stack((*channels, alpha)).astype(np.uint8)
    return vertices, faces, colors


def _plane(record: dict) -> tuple[np.ndarray, float]:
    equation = np.asarray(record["plane_equation"], dtype=np.float64)
    length = np.linalg.norm(equation[:3])
    if equation.shape != (4,) or not np.isfinite(equation).all() or length < 1e-8:
        raise ValueError(f"Invalid plane equation for {record.get('plane_id')}")
    return equation[:3] / length, float(equation[3] / length)


def _plane_group(planes: list[dict]) -> tuple[np.ndarray, float, float]:
    weights = np.asarray([max(float(item.get("support_area_m2", 0.0)), 1e-6) for item in planes])
    equations = np.asarray([item["plane_equation"] for item in planes], dtype=np.float64)
    normal_lengths = np.linalg.norm(equations[:, :3], axis=1)
    normals = equations[:, :3] / normal_lengths[:, None]
    average_normal = np.average(normals, axis=0, weights=weights)
    normal_length = np.linalg.norm(average_normal)
    normal = average_normal / normal_length
    offset = float(np.average(equations[:, 3] / normal_lengths, weights=weights))
    offset /= normal_length
    elevation = float(np.average([item["mean_elevation_m"] for item in planes], weights=weights))
    return normal, offset, elevation


def _face_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers = np.empty((len(faces), 3), dtype=np.float32)
    normals = np.zeros_like(centers)
    for start in range(0, len(faces), 250_000):
        stop = min(start + 250_000, len(faces))
        triangles = vertices[faces[start:stop]]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(cross, axis=1)
        centers[start:stop] = triangles.mean(axis=1)
        valid = lengths > 1e-12
        normals[start:stop][valid] = cross[valid] / lengths[valid, None]
    return centers, normals


def _selected_ceiling_faces(
    path: Path, vertices: np.ndarray, faces: np.ndarray, record: dict,
) -> np.ndarray:
    if path.name.startswith("ceiling_group_"):
        return np.ones(len(faces), dtype=bool)

    normal, offset = _plane(record)
    centers, face_normals = _face_geometry(vertices, faces)
    horizontal = np.abs(face_normals[:, 1]) >= np.cos(np.deg2rad(5.0))
    distance = np.abs(centers @ normal + offset)
    return (horizontal & (face_normals[:, 1] < 0)
            & (distance <= SLAB_PLANE_TOLERANCE_M))


def _same_triangle(first: np.ndarray, second: np.ndarray) -> bool:
    first = first[np.lexsort((first[:, 2], first[:, 1], first[:, 0]))]
    second = second[np.lexsort((second[:, 2], second[:, 1], second[:, 0]))]
    return bool(np.max(np.abs(first - second)) <= PLANE_MATCH_TOLERANCE_M)


def _canonical_triangles(triangles: np.ndarray) -> np.ndarray:
    order = np.lexsort((triangles[:, :, 2], triangles[:, :, 1], triangles[:, :, 0]), axis=1)
    return np.take_along_axis(triangles, np.repeat(order[:, :, None], 3, axis=2), axis=1)


def _instance_paths(segmentation_dir: Path, role: str) -> list[Path]:
    masks_dir = segmentation_dir / "3d_masks"
    if masks_dir.is_dir():
        paths = sorted(masks_dir.glob(f"class_*_{role}/instance_*.ply"))
        if paths:
            return paths

    legacy_dir = segmentation_dir / "Instances PLYs"
    legacy_role_dir = legacy_dir / ("Non-Structural" if role == "object" else "Structural/Wall")
    if role == "wall":
        return sorted(legacy_role_dir.glob("*.ply"))
    return sorted(legacy_role_dir.glob("*/*.ply"))


def _matched_source_faces(
    faces: np.ndarray,
    source_paths: list[Path],
    transform: np.ndarray,
    target_tree: cKDTree,
    target_triangles: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    result = np.zeros(len(faces), dtype=bool)
    source_triangle_count = 0
    matched_source_count = 0
    for source_path in source_paths:
        source_vertices, source_faces, _ = _read_mesh(source_path)
        source_vertices = source_vertices @ transform[:3, :3].T + transform[:3, 3]
        source_triangles = source_vertices[source_faces]
        source_triangle_count += len(source_faces)
        source_centers = source_triangles.mean(axis=1)
        for start in range(0, len(source_faces), 250_000):
            stop = min(start + 250_000, len(source_faces))
            distances, nearest = target_tree.query(source_centers[start:stop], workers=-1)
            possible = distances <= PLANE_MATCH_TOLERANCE_M
            if not np.any(possible):
                continue
            local_ids = np.flatnonzero(possible)
            source_ids = start + local_ids
            target_ids = nearest[local_ids]
            exact = np.max(
                np.abs(_canonical_triangles(source_triangles[source_ids])
                       - _canonical_triangles(target_triangles[target_ids])),
                axis=(1, 2),
            ) <= PLANE_MATCH_TOLERANCE_M
            if np.any(exact):
                result[target_ids[exact]] = True
                matched_source_count += int(exact.sum())

            for source_id, target_id in zip(source_ids[~exact], target_ids[~exact]):
                candidates = target_tree.query_ball_point(source_centers[source_id], PLANE_MATCH_TOLERANCE_M)
                for candidate in candidates:
                    if _same_triangle(source_triangles[source_id], target_triangles[candidate]):
                        result[candidate] = True
                        matched_source_count += 1
                        break
    return result, source_triangle_count, matched_source_count


def _ceiling_ownership_masks(
    scene_dir: Path,
    scan_id: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    segmentation_dir = scene_dir / scan_id / "segmentation"
    object_paths = _instance_paths(segmentation_dir, "object")
    wall_paths = _instance_paths(segmentation_dir, "wall")
    if not object_paths or not wall_paths:
        raise FileNotFoundError(
            f"Ceiling ownership needs Phase 05 object and wall instance PLYs for {scan_id} in {segmentation_dir}"
        )

    centers, _ = _face_geometry(vertices, faces)
    tree = cKDTree(centers)
    triangles = vertices[faces]
    object_mask, object_source_count, object_matched_count = _matched_source_faces(
        faces, object_paths, transform, tree, triangles,
    )
    wall_mask, wall_source_count, wall_matched_count = _matched_source_faces(
        faces, wall_paths, transform, tree, triangles,
    )
    return object_mask, wall_mask, {
        "object_source_triangles": object_source_count,
        "object_source_triangles_matched": object_matched_count,
        "object_mesh_faces_matched": int(object_mask.sum()),
        "wall_source_triangles": wall_source_count,
        "wall_source_triangles_matched": wall_matched_count,
        "wall_mesh_faces_matched": int(wall_mask.sum()),
    }


def _regularized_ceiling_mask(
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    source_path: Path,
    record: dict,
) -> tuple[np.ndarray, dict]:
    source_vertices, source_faces, _ = _read_mesh(source_path)
    selected = _selected_ceiling_faces(source_path, source_vertices, source_faces, record)
    source_face_ids = np.flatnonzero(selected)
    result = np.zeros(len(mesh_faces), dtype=bool)
    if not len(source_face_ids):
        return result, {"source_triangles": 0, "matched_triangles": 0, "unmatched_triangles": 0}

    mesh_centers, _ = _face_geometry(mesh_vertices, mesh_faces)
    tree = cKDTree(mesh_centers)
    source_triangles = source_vertices[source_faces[source_face_ids]]
    source_centers = source_triangles.mean(axis=1)
    distances, nearest = tree.query(source_centers, workers=-1)
    matched = np.zeros(len(source_face_ids), dtype=bool)
    for index, main_face_id in enumerate(nearest):
        if distances[index] <= PLANE_MATCH_TOLERANCE_M and _same_triangle(
            source_triangles[index], mesh_vertices[mesh_faces[main_face_id]],
        ):
            matched[index] = True
            result[main_face_id] = True
            continue
        candidates = tree.query_ball_point(source_centers[index], PLANE_MATCH_TOLERANCE_M)
        for candidate in candidates:
            if _same_triangle(source_triangles[index], mesh_vertices[mesh_faces[candidate]]):
                matched[index] = True
                result[candidate] = True
                break
    return result, {
        "source_triangles": int(len(source_face_ids)),
        "matched_triangles": int(matched.sum()),
        "unmatched_triangles": int((~matched).sum()),
    }


def _sealed_ceiling_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    source_path: Path,
    record: dict,
) -> tuple[np.ndarray, dict]:
    source_vertices, source_faces, _ = _read_mesh(source_path)
    selected = _selected_ceiling_faces(source_path, source_vertices, source_faces, record)
    source_face_ids = np.flatnonzero(selected)
    result = np.zeros(len(faces), dtype=bool)
    if not len(source_face_ids):
        return result, {"source_triangles": 0, "matched_triangles": 0}

    normal, offset = _plane(record)
    reference = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    axis_u = np.cross(normal, reference)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)

    source_triangles = source_vertices[source_faces[source_face_ids]]
    source_centers = source_triangles.mean(axis=1)
    source_residual = source_centers @ normal + offset
    source_projection = source_triangles.reshape(-1, 3) @ np.column_stack((axis_u, axis_v))
    lower = source_projection.min(axis=0) - 0.05
    upper = source_projection.max(axis=0) + 0.05
    # Phase 04 sealed surfaces can sit below the corresponding Phase 06 fitted ceiling.
    residual_pad = max(SEALED_SUPPORT_PADDING_M, min(0.50, 3.0 * float(record.get("fit_rms_m", 0.0))))

    centers, face_normals = _face_geometry(vertices, faces)
    signed = centers @ normal + offset
    # A sealed room mesh's interior ceiling faces point toward the room below.
    aligned = face_normals @ normal <= -np.cos(np.deg2rad(45.0))
    candidates = np.flatnonzero(
        aligned
        & (signed >= float(source_residual.min()) - residual_pad)
        & (signed <= float(source_residual.max()) + 0.05)
    )
    if not len(candidates):
        return result, {"source_triangles": int(len(source_face_ids)), "matched_triangles": 0}

    projected = centers[candidates] @ np.column_stack((axis_u, axis_v))
    in_bounds = np.all((projected >= lower) & (projected <= upper), axis=1)
    candidates, projected = candidates[in_bounds], projected[in_bounds]
    if not len(candidates):
        return result, {"source_triangles": int(len(source_face_ids)), "matched_triangles": 0}

    hull_points = np.unique(source_projection, axis=0)
    if len(hull_points) >= 3:
        try:
            inside = Delaunay(hull_points).find_simplex(projected) >= 0
            candidates = candidates[inside]
        except QhullError:
            pass
    result[candidates] = True
    return result, {
        "source_triangles": int(len(source_face_ids)),
        "matched_triangles": int(len(candidates)),
    }


def _ownership_cut_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    ceiling_plane: tuple[np.ndarray, float] | None,
    object_mask: np.ndarray,
    wall_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    remove = np.zeros(len(faces), dtype=bool)
    preserve = np.zeros(len(faces), dtype=bool)
    if ceiling_plane is None or not len(faces):
        return remove, preserve, {
            "ownership_band_triangles": 0,
            "unassigned_non_border_triangles_removed": 0,
            "object_triangles_preserved": 0,
            "border_wall_triangles_preserved": 0,
        }

    normal, offset = ceiling_plane
    vertex_distance = vertices @ normal + offset
    face_distance = vertex_distance[faces]
    band = np.any(
        (face_distance >= -OWNERSHIP_PROTECTION_DEPTH_M - 1e-6)
        & (face_distance <= 1e-6),
        axis=1,
    )
    preserve = band & (object_mask | wall_mask)
    remove = band & ~preserve
    return remove, preserve, {
        "ownership_band_triangles": int(band.sum()),
        "unassigned_non_border_triangles_removed": int(remove.sum()),
        "object_triangles_preserved": int((band & object_mask).sum()),
        "border_wall_triangles_preserved": int((band & wall_mask & ~object_mask).sum()),
    }


def _gray(colors: np.ndarray) -> np.ndarray:
    result = colors.copy()
    luminance = np.rint(
        0.2126 * colors[:, 0].astype(np.float32)
        + 0.7152 * colors[:, 1].astype(np.float32)
        + 0.0722 * colors[:, 2].astype(np.float32)
    ).astype(np.uint8)
    result[:, :3] = luminance[:, None]
    return result


def _elongated_triangle_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    ceiling_plane: tuple[np.ndarray, float] | None,
) -> np.ndarray:
    elongated = np.zeros(len(faces), dtype=bool)
    ceiling_distance = (
        vertices @ ceiling_plane[0] + ceiling_plane[1]
        if ceiling_plane is not None else None
    )
    for start in range(0, len(faces), 250_000):
        stop = min(start + 250_000, len(faces))
        face_vertices = faces[start:stop]
        triangles = vertices[face_vertices]
        edge_squared = np.stack((
            np.sum((triangles[:, 1] - triangles[:, 0]) ** 2, axis=1),
            np.sum((triangles[:, 2] - triangles[:, 1]) ** 2, axis=1),
            np.sum((triangles[:, 0] - triangles[:, 2]) ** 2, axis=1),
        ), axis=1)
        longest_squared = edge_squared.max(axis=1)
        twice_area = np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=1,
        )
        aspect_ratio = np.divide(
            longest_squared, twice_area,
            out=np.full(len(twice_area), np.inf, dtype=np.float32),
            where=twice_area > 1e-12,
        )
        chunk_mask = aspect_ratio > MAX_TRIANGLE_ASPECT_RATIO
        if ceiling_distance is not None:
            distance = ceiling_distance[face_vertices]
            near_ceiling = np.any(
                (distance >= -NEAR_CEILING_TRIANGLE_DEPTH_M - 1e-6)
                & (distance <= 1e-6),
                axis=1,
            )
            chunk_mask |= near_ceiling & (aspect_ratio > MAX_NEAR_CEILING_TRIANGLE_ASPECT_RATIO)
        elongated[start:stop] = chunk_mask
    return elongated


def _clip_and_color(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    remove_faces: np.ndarray,
    ceiling_plane: tuple[np.ndarray, float] | None,
    floor_plane: tuple[np.ndarray, float],
    preserve_faces: np.ndarray | None = None,
    filter_elongated: bool = True,
    gray_all: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    floor_normal, floor_offset = floor_plane
    floor_distance = vertices @ floor_normal + floor_offset
    if ceiling_plane is None:
        distance = None
        face_distance = None
        whole = ~remove_faces
        crossing_ids = np.empty(0, dtype=np.int64)
        discarded_above = 0
    else:
        ceiling_normal, ceiling_offset = ceiling_plane
        distance = vertices @ ceiling_normal + ceiling_offset
        face_distance = distance[faces]
        protected = np.zeros(len(faces), dtype=bool)
        if preserve_faces is not None:
            in_ownership_band = np.any(
                (face_distance >= -OWNERSHIP_PROTECTION_DEPTH_M - 1e-6)
                & (face_distance <= 1e-6),
                axis=1,
            )
            protected = preserve_faces & in_ownership_band
        below_cut = face_distance <= -CLEARANCE_M + 1e-6
        whole = (np.all(below_cut, axis=1) & ~remove_faces) | protected
        crossing = (
            np.any(below_cut, axis=1) & ~np.all(below_cut, axis=1)
            & ~remove_faces & ~protected
        )
        crossing_ids = np.flatnonzero(crossing)
        discarded_above = int((~np.any(below_cut, axis=1) & ~remove_faces & ~protected).sum())

    face_floor_distance = floor_distance[faces].mean(axis=1)
    gray_faces = np.ones(len(faces), dtype=bool) if gray_all else face_floor_distance < -FLOOR_GRAY_OFFSET_M
    kept_ids = np.flatnonzero(whole)
    kept_gray = gray_faces[kept_ids]
    normal_ids = np.unique(faces[kept_ids[~kept_gray]].reshape(-1)) if np.any(~kept_gray) else np.empty(0, np.int32)
    gray_ids = np.unique(faces[kept_ids[kept_gray]].reshape(-1)) if np.any(kept_gray) else np.empty(0, np.int32)

    normal_map = np.full(len(vertices), -1, dtype=np.int32)
    gray_map = np.full(len(vertices), -1, dtype=np.int32)
    normal_map[normal_ids] = np.arange(len(normal_ids), dtype=np.int32)
    gray_map[gray_ids] = np.arange(len(gray_ids), dtype=np.int32) + len(normal_ids)
    out_vertices = np.concatenate((vertices[normal_ids], vertices[gray_ids]), axis=0)
    out_colors = np.concatenate((colors[normal_ids], _gray(colors[gray_ids])), axis=0)
    out_faces = np.empty((len(kept_ids), 3), dtype=np.int32)
    if np.any(~kept_gray):
        out_faces[~kept_gray] = normal_map[faces[kept_ids[~kept_gray]]]
    if np.any(kept_gray):
        out_faces[kept_gray] = gray_map[faces[kept_ids[kept_gray]]]

    crossing_vertices = []
    crossing_colors = []
    crossing_faces = []
    crossing_gray_flags = []
    cut_value = -CLEARANCE_M
    for face_id in crossing_ids:
        face = faces[face_id]
        if distance is None:
            continue
        polygon = []
        for edge in range(3):
            first_id, second_id = int(face[edge]), int(face[(edge + 1) % 3])
            first_inside = distance[first_id] <= cut_value
            second_inside = distance[second_id] <= cut_value
            first = (vertices[first_id], colors[first_id].astype(np.float32), float(distance[first_id]))
            second = (vertices[second_id], colors[second_id].astype(np.float32), float(distance[second_id]))
            if first_inside and second_inside:
                polygon.append(second)
            elif first_inside and not second_inside:
                fraction = (cut_value - first[2]) / (second[2] - first[2])
                polygon.append((first[0] + fraction * (second[0] - first[0]),
                                first[1] + fraction * (second[1] - first[1]), cut_value))
            elif not first_inside and second_inside:
                fraction = (cut_value - first[2]) / (second[2] - first[2])
                polygon.append((first[0] + fraction * (second[0] - first[0]),
                                first[1] + fraction * (second[1] - first[1]), cut_value))
                polygon.append(second)
        if len(polygon) < 3:
            continue
        base = len(out_vertices) + len(crossing_vertices)
        polygon_vertices = np.asarray([item[0] for item in polygon], dtype=np.float32)
        polygon_colors = np.clip(np.rint([item[1] for item in polygon]), 0, 255).astype(np.uint8)
        if gray_faces[face_id]:
            polygon_colors = _gray(polygon_colors)
        crossing_vertices.extend(polygon_vertices)
        crossing_colors.extend(polygon_colors)
        for index in range(1, len(polygon) - 1):
            triangle = polygon_vertices[[0, index, index + 1]]
            if np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])) <= 1e-12:
                continue
            crossing_faces.append([base, base + index, base + index + 1])
            crossing_gray_flags.append(gray_faces[face_id])

    if crossing_vertices:
        out_vertices = np.concatenate((out_vertices, np.asarray(crossing_vertices, dtype=np.float32)), axis=0)
        out_colors = np.concatenate((out_colors, np.asarray(crossing_colors, dtype=np.uint8)), axis=0)
        out_faces = np.concatenate((out_faces, np.asarray(crossing_faces, dtype=np.int32)), axis=0)

    output_gray_flags = np.concatenate((kept_gray, np.asarray(crossing_gray_flags, dtype=bool)))
    elongated = (
        _elongated_triangle_mask(out_vertices, out_faces, ceiling_plane)
        if filter_elongated else np.zeros(len(out_faces), dtype=bool)
    )
    removed_elongated = int(elongated.sum())
    output_gray_triangles = int(output_gray_flags[~elongated].sum())
    out_faces = out_faces[~elongated]
    used_vertices = np.zeros(len(out_vertices), dtype=bool)
    used_vertices[out_faces.reshape(-1)] = True
    used_ids = np.flatnonzero(used_vertices)
    remap = np.full(len(out_vertices), -1, dtype=np.int32)
    remap[used_ids] = np.arange(len(used_ids), dtype=np.int32)
    out_vertices = out_vertices[used_ids]
    out_colors = out_colors[used_ids]
    out_faces = remap[out_faces]

    max_cut_distance = None
    if ceiling_plane is not None and len(out_vertices):
        max_cut_distance = float(np.max(out_vertices @ ceiling_plane[0] + ceiling_plane[1]))
        allowed_max = -CLEARANCE_M + 2e-5
        if preserve_faces is not None and np.any(protected):
            protected_ids = np.unique(faces[protected].reshape(-1))
            allowed_max = max(allowed_max, float(np.max(distance[protected_ids])) + 2e-5)
        if max_cut_distance > allowed_max:
            raise ValueError(f"Clipped mesh exceeds ceiling clearance: {max_cut_distance:.6f} m")

    return out_vertices, out_faces, out_colors, {
        "input_triangles": int(len(faces)),
        "removed_ceiling_triangles": int(remove_faces.sum()),
        "discarded_above_cut_triangles": discarded_above,
        "clipped_triangles": int(len(crossing_ids)),
        "output_triangles": int(len(out_faces)),
        "removed_elongated_triangles": removed_elongated,
        "greyscale_triangles": output_gray_triangles,
        "max_signed_distance_to_ceiling_m": max_cut_distance,
        "protected_near_ceiling_triangles": int(protected.sum()) if ceiling_plane is not None else 0,
    }


def _merge_parts(parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertex_count = sum(len(item[0]) for item in parts)
    face_count = sum(len(item[1]) for item in parts)
    vertices = np.empty((vertex_count, 3), dtype=np.float32)
    colors = np.empty((vertex_count, 4), dtype=np.uint8)
    faces = np.empty((face_count, 3), dtype=np.int32)
    vertex_offset = 0
    face_offset = 0
    for part_vertices, part_faces, part_colors in parts:
        vertices[vertex_offset:vertex_offset + len(part_vertices)] = part_vertices
        colors[vertex_offset:vertex_offset + len(part_colors)] = part_colors
        faces[face_offset:face_offset + len(part_faces)] = part_faces + vertex_offset
        vertex_offset += len(part_vertices)
        face_offset += len(part_faces)
    return vertices, faces, colors


def _write_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex_dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("alpha", "u1"),
    ])
    face_dtype = np.dtype([("count", "u1"), ("vertex_indices", "<i4", (3,))])
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    vertex_data = np.empty(len(vertices), dtype=vertex_dtype)
    vertex_data["x"], vertex_data["y"], vertex_data["z"] = vertices.T
    vertex_data["red"], vertex_data["green"], vertex_data["blue"], vertex_data["alpha"] = colors.T
    face_data = np.empty(len(faces), dtype=face_dtype)
    face_data["count"] = 3
    face_data["vertex_indices"] = faces
    with temporary.open("wb") as stream:
        stream.write(header)
        vertex_data.tofile(stream)
        face_data.tofile(stream)
    temporary.replace(path)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9+-]+", "_", name).strip("_")


def _path_label(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def generate_floor_level_meshes(
    alignment_dir: Path,
    metadata_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    alignment_dir = Path(alignment_dir).resolve()
    scene_artifacts = alignment_dir.parent / "artifacts"
    metadata_path = Path(metadata_path or scene_artifacts / "floor_level_metadata.json").resolve()
    output_dir = Path(output_dir or scene_artifacts / "floor_level_meshes").resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Floor metadata is missing: {metadata_path}; run assign_floor_level_metadata.py first")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    transforms_path = alignment_dir / "stats" / "aligned_scans_transforms.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(f"Missing Phase 07 scan transforms: {transforms_path}")
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))["scans"]
    scene_dir = alignment_dir.parent

    grouped_floors: dict[str, dict[str, list[dict]]] = {}
    for floor in metadata["floor_planes"]:
        grouped_floors.setdefault(floor["floor_level_name"], {}).setdefault(floor["scan_id"], []).append(floor)
    grouped_ceilings: dict[tuple[str, str], list[dict]] = {}
    for ceiling in metadata["ceiling_planes"]:
        if ceiling.get("floor_level_name"):
            grouped_ceilings.setdefault((ceiling["scan_id"], ceiling["floor_level_name"]), []).append(ceiling)
    level_order = sorted(
        grouped_floors,
        key=lambda name: (min(item["mean_elevation_m"] for group in grouped_floors[name].values()
                              for item in group), name),
    )
    floor_scan_ids = {scan_id for scans in grouped_floors.values() for scan_id in scans}
    aligned_scan_ids = {path.parent.name for path in (alignment_dir / "scans").glob("scan_*/mesh.ply")}
    missing_floor_scans = aligned_scan_ids - floor_scan_ids
    if missing_floor_scans:
        raise ValueError(f"Aligned sealed scans have no classified floor plane: {sorted(missing_floor_scans)}")
    scan_order = sorted(
        aligned_scan_ids,
        key=lambda scan_id: (
            int(scan_id.split("_")[-1]) if scan_id.split("_")[-1].isdigit() else 10**9,
            scan_id,
        ),
    )

    manifest = {
        "schema_version": 1,
        "source_metadata": _path_label(metadata_path, alignment_dir),
        "cutoff_below_ceiling_m": CLEARANCE_M,
        "sealed_ownership_protection_depth_m": OWNERSHIP_PROTECTION_DEPTH_M,
        "greyscale_below_floor_m": FLOOR_GRAY_OFFSET_M,
        "max_triangle_aspect_ratio": MAX_TRIANGLE_ASPECT_RATIO,
        "near_ceiling_triangle_filter_depth_m": NEAR_CEILING_TRIANGLE_DEPTH_M,
        "max_near_ceiling_triangle_aspect_ratio": MAX_NEAR_CEILING_TRIANGLE_ASPECT_RATIO,
        "levels": [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    sealed_parts_by_scan = {}
    regularized_parts_by_scan = {}
    ownership_cache = {}
    latest_scan_records = {}
    previous_level_name = None
    for level_name in level_order:
        level_floor_records = [
            floor for floors in grouped_floors[level_name].values() for floor in floors
        ]
        level_floor_normal, level_floor_offset, _ = _plane_group(level_floor_records)
        level_floor_plane = (level_floor_normal, level_floor_offset)
        current_scan_records = {}
        for scan_id, floor_records in sorted(
            grouped_floors[level_name].items(),
            key=lambda item: (int(item[0].split("_")[-1]) if item[0].split("_")[-1].isdigit() else 10**9, item[0]),
        ):
            floor_normal, floor_offset, floor_elevation = _plane_group(floor_records)
            ceilings = [item for item in grouped_ceilings.get((scan_id, level_name), [])
                        if item["mean_elevation_m"] > floor_elevation]
            ceiling = min(ceilings, key=lambda item: item["mean_elevation_m"]) if ceilings else None
            ceiling_plane = _plane(ceiling) if ceiling else None
            source_path = alignment_dir / ceiling["source_mesh"] if ceiling else None

            regularized_path = alignment_dir / "scans" / scan_id / "mesh_elaboration" / "complete.ply"
            sealed_path = alignment_dir / "scans" / scan_id / "mesh.ply"
            if not regularized_path.is_file() or not sealed_path.is_file():
                raise FileNotFoundError(f"Missing aligned mesh for {scan_id}: {regularized_path} or {sealed_path}")

            reg_vertices, reg_faces, reg_colors = _read_mesh(regularized_path)
            sealed_vertices, sealed_faces, sealed_colors = _read_mesh(sealed_path)
            reg_ceiling_mask = np.zeros(len(reg_faces), dtype=bool)
            sealed_ceiling_mask = np.zeros(len(sealed_faces), dtype=bool)
            reg_ceiling_info = {"source_triangles": 0, "matched_triangles": 0, "unmatched_triangles": 0}
            sealed_ceiling_info = {"source_triangles": 0, "matched_triangles": 0}
            if ceiling and source_path and source_path.is_file():
                reg_ceiling_mask, reg_ceiling_info = _regularized_ceiling_mask(
                    reg_vertices, reg_faces, source_path, ceiling,
                )
                sealed_ceiling_mask, sealed_ceiling_info = _sealed_ceiling_mask(
                    sealed_vertices, sealed_faces, source_path, ceiling,
                )

            if ceiling_plane is not None:
                if scan_id not in ownership_cache:
                    transform = np.asarray(transforms[scan_id]["transform_matrix"], dtype=np.float64)
                    if transform.shape != (4, 4) or not np.isfinite(transform).all():
                        raise ValueError(f"Invalid Phase 07 transform for {scan_id}")
                    ownership_cache[scan_id] = _ceiling_ownership_masks(
                        scene_dir, scan_id, sealed_vertices, sealed_faces, transform,
                    )
                object_mask, wall_mask, ownership_source_info = ownership_cache[scan_id]
                ownership_remove, preserve_faces, ownership_info = _ownership_cut_mask(
                    sealed_vertices, sealed_faces, ceiling_plane, object_mask, wall_mask,
                )
                ownership_info.update(ownership_source_info)
                sealed_remove_mask = sealed_ceiling_mask | ownership_remove
                sealed_remove_mask &= ~preserve_faces
            else:
                preserve_faces = None
                ownership_info = {
                    "ownership_band_triangles": 0,
                    "unassigned_non_border_triangles_removed": 0,
                    "object_triangles_preserved": 0,
                    "border_wall_triangles_preserved": 0,
                }
                sealed_remove_mask = sealed_ceiling_mask

            reg_result = _clip_and_color(
                reg_vertices, reg_faces, reg_colors, reg_ceiling_mask,
                ceiling_plane, level_floor_plane, filter_elongated=False,
            )
            sealed_result = _clip_and_color(
                sealed_vertices, sealed_faces, sealed_colors, sealed_remove_mask,
                ceiling_plane, level_floor_plane, preserve_faces,
            )
            sealed_result = sealed_result[:3] + (
                sealed_result[3] | {"ownership_cut": ownership_info},
            )
            regularized_parts_by_scan[scan_id] = reg_result[:3]
            sealed_parts_by_scan[scan_id] = sealed_result[:3]
            current_scan_records[scan_id] = {
                "scan_id": scan_id,
                "output_status": "generated_for_floor_level",
                "floor_plane_ids": [item["plane_id"] for item in floor_records],
                "floor_elevation_m": floor_elevation,
                "ceiling_plane_id": ceiling["plane_id"] if ceiling else None,
                "ceiling_mean_elevation_m": ceiling["mean_elevation_m"] if ceiling else None,
                "ceiling_cut_plane_equation": (
                    [float(value) for value in np.r_[ceiling_plane[0], ceiling_plane[1] + CLEARANCE_M]]
                    if ceiling_plane else None
                ),
                "regularized": reg_result[3] | {
                    "ceiling_instance_match": reg_ceiling_info,
                    "greyscale_reference_floor_level": level_name,
                },
                "sealed": sealed_result[3] | {
                    "ceiling_instance_match": sealed_ceiling_info,
                    "greyscale_reference_floor_level": level_name,
                },
            }

        for scan_id in scan_order:
            if scan_id not in current_scan_records:
                if scan_id not in regularized_parts_by_scan or scan_id not in sealed_parts_by_scan:
                    raise ValueError(f"No earlier floor output is available to carry forward {scan_id}")
                carried = latest_scan_records[scan_id].copy()
                recolored_regularized = _clip_and_color(
                    *regularized_parts_by_scan[scan_id],
                    np.zeros(len(regularized_parts_by_scan[scan_id][1]), dtype=bool),
                    None,
                    level_floor_plane,
                    filter_elongated=False,
                    gray_all=True,
                )
                regularized_parts_by_scan[scan_id] = recolored_regularized[:3]
                carried_regularized = carried["regularized"].copy()
                carried_regularized["greyscale_triangles"] = recolored_regularized[3]["greyscale_triangles"]
                carried_regularized["greyscale_reference_floor_level"] = level_name
                carried["regularized"] = carried_regularized
                carried_mesh = sealed_parts_by_scan[scan_id]
                recolored = _clip_and_color(
                    *carried_mesh,
                    np.zeros(len(carried_mesh[1]), dtype=bool),
                    None,
                    level_floor_plane,
                    gray_all=True,
                )
                sealed_parts_by_scan[scan_id] = recolored[:3]
                carried_sealed = carried["sealed"].copy()
                carried_sealed["greyscale_triangles"] = recolored[3]["greyscale_triangles"]
                carried_sealed["greyscale_reference_floor_level"] = level_name
                carried["sealed"] = carried_sealed
                carried["output_status"] = "carried_forward"
                carried["carried_forward_from_floor_level"] = previous_level_name
                current_scan_records[scan_id] = carried
        scan_records = [current_scan_records[scan_id] for scan_id in scan_order]
        sealed_parts = [sealed_parts_by_scan[scan_id] for scan_id in scan_order]

        regularized_mesh = _merge_parts([regularized_parts_by_scan[scan_id] for scan_id in scan_order])
        sealed_mesh = _merge_parts(sealed_parts)
        name = _safe_name(level_name)
        regularized_output = output_dir / f"floor_{name}_regularized.ply"
        sealed_output = output_dir / f"floor_{name}_sealed.ply"
        _write_mesh(regularized_output, *regularized_mesh)
        _write_mesh(sealed_output, *sealed_mesh)
        manifest["levels"].append({
            "floor_level_name": level_name,
            "greyscale_reference_plane_equation": [
                float(value) for value in np.r_[level_floor_normal, level_floor_offset]
            ],
            "scan_ids": scan_order,
            "regularized_scan_ids": sorted(
                regularized_parts_by_scan,
                key=lambda scan_id: (
                    int(scan_id.split("_")[-1]) if scan_id.split("_")[-1].isdigit() else 10**9,
                    scan_id,
                ),
            ),
            "generated_regularized_scan_ids": [
                scan_id for scan_id in scan_order if scan_id in grouped_floors[level_name]
            ],
            "regularized_mesh": _path_label(regularized_output, alignment_dir),
            "regularized_vertex_count": int(len(regularized_mesh[0])),
            "regularized_triangle_count": int(len(regularized_mesh[1])),
            "sealed_mesh": _path_label(sealed_output, alignment_dir),
            "sealed_vertex_count": int(len(sealed_mesh[0])),
            "sealed_triangle_count": int(len(sealed_mesh[1])),
            "scans": scan_records,
        })
        latest_scan_records.update(current_scan_records)
        previous_level_name = level_name

    manifest_path = output_dir / "floor_level_meshes_manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-dir", type=Path, default=DEFAULT_ALIGNMENT_DIR,
                        help="Phase 07 output directory containing aligned per-scan meshes.")
    parser.add_argument("--metadata", type=Path,
                        help="Floor metadata JSON; defaults to current_scene/artifacts/floor_level_metadata.json.")
    parser.add_argument("--output-dir", type=Path,
                        help="Output directory; defaults to current_scene/artifacts/floor_level_meshes.")
    args = parser.parse_args()
    generate_floor_level_meshes(args.alignment_dir, args.metadata, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
