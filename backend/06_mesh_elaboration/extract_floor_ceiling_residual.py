#!/usr/bin/env python3
"""Fit residual floor/ceiling planes and export full-scene per-floor projections."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_3DGS_to_mesh"))
from extract_structural_meshes import (  # noqa: E402
    component_face_indices,
    load_associated_object_instances,
    load_door_window_associations,
)
from fit_floors_ceilings_planes import fit_robust_plane  # noqa: E402
from fill_mesh_holes import read_ply_face_flags  # noqa: E402

SOURCE_FACE_MATCH_TOLERANCE_M = 2e-5
PLANE_DISTANCE_M = 0.10
MAIN_COMPONENT_AREA_M2 = 1.0
MIN_PLANE_AREA_M2 = 0.5
MIN_PLANE_NORMAL_Y = 0.2
PLANE_SNAP_TILT_DEG = 5.0
RAMP_MIN_RISE_M = 0.10
RAMP_MIN_RUN_M = 0.5
RAMP_MIN_WIDTH_M = 0.2
RAMP_MAX_LATERAL_GAP_M = 0.20
RAMP_MIN_CONTACT_AREA_M2 = 0.05
RAMP_MIN_PLANARITY_M = 0.02
RAMP_MAX_PLANARITY_M = 0.05
RAMP_MAX_FLOOR_FRACTION = 0.50
FLOOR_LEVEL_TOLERANCE_M = 0.15
PROJECTION_CEILING_CLEARANCE_M = 0.15
PROJECTION_LOWER_CEILING_MAX_GAP_M = 0.40
PROJECTION_FALLBACK_BELOW_FLOOR_M = 0.10
PROJECTION_MIN_FLOATING_COMPONENT_TRIANGLES = 10_000
FLOOR_COLOR = np.array([1.0, 0.22, 0.22])
CEILING_COLOR = np.array([0.28, 0.48, 1.0])
FLOOR_PLANE_COLOR = np.array([0.78, 0.0, 0.0])
CEILING_PLANE_COLOR = np.array([0.0, 0.05, 0.78])
PROJECTION_COLORS = np.array([
    [0.90, 0.04, 0.04],  # floor
    [0.04, 0.16, 0.90],  # ceiling
    [0.72, 0.72, 0.72],  # other
])
HOLE_FILL_PLANE_ANGLE_DEG = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-mesh", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--structural-classes", type=Path, required=True)
    parser.add_argument("--wall-filtered-mesh", type=Path, required=True,
                        help="Wall mesh after removing near-horizontal triangles.")
    parser.add_argument("--fitted-wall-planes", type=Path, required=True,
                        help="Finite fitted wall planes from the current geometry-only wall fit.")
    parser.add_argument("--include-wall-instances", action="store_true",
                        help="Add filtered wall-instance triangles back after projection floater cleanup.")
    parser.add_argument("--wall-instance-floor-clearance-m", type=float, default=0.0,
                        help="Start restored wall triangles this far above the target floor plane.")
    parser.add_argument("--segmentation-output", type=Path, required=True)
    parser.add_argument("--associations", type=Path)
    parser.add_argument("--output", type=Path, required=True,
                        help="Path for the single combined evaluation PLY; must not already exist.")
    return parser.parse_args()


def _axis_alignment_from_wall_manifest(fitted_wall_planes: Path) -> tuple[np.ndarray, dict]:
    manifest_path = fitted_wall_planes.parent / "wall_planes_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Wall orientation manifest not found beside {fitted_wall_planes}")
    data = json.loads(manifest_path.read_text())
    angle = float(data.get("dominant_orientation_deg", 0.0))
    theta = np.deg2rad(angle)
    rotation = np.array([
        [np.cos(theta), 0.0, np.sin(theta)],
        [0.0, 1.0, 0.0],
        [-np.sin(theta), 0.0, np.cos(theta)],
    ])
    return rotation, {
        "source_manifest": str(manifest_path),
        "source_orientation_deg": angle,
        "rotation_deg_y": -angle,
        "rotation_matrix": rotation.tolist(),
        "translation_m": [0.0, 0.0, 0.0],
    }


def _centres(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return vertices[faces].mean(axis=1)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    index = np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2.0, side="left")
    return float(values[order[min(index, len(order) - 1)]])


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="stable")
    index = np.searchsorted(np.cumsum(weights[order]), quantile * weights.sum(), side="left")
    return float(values[order[min(index, len(order) - 1)]])


def _tilt_degrees(normal: np.ndarray) -> float:
    return float(np.rad2deg(np.arctan2(np.linalg.norm(normal[[0, 2]]), abs(normal[1]))))


def _nearest_five_degree_tilt(tilt: float) -> float:
    return float(np.floor(tilt / PLANE_SNAP_TILT_DEG + 0.5) * PLANE_SNAP_TILT_DEG)


def _snap_normal_to_five_degrees(normal: np.ndarray) -> np.ndarray:
    tilt = _nearest_five_degree_tilt(_tilt_degrees(normal))
    sign_y = np.sign(normal[1])
    if tilt == 0.0:
        return np.array([0.0, sign_y, 0.0])
    horizontal = normal[[0, 2]]
    horizontal /= np.linalg.norm(horizontal)
    snapped = np.zeros(3)
    snapped[[0, 2]] = horizontal * np.sin(np.deg2rad(tilt))
    snapped[1] = sign_y * np.cos(np.deg2rad(tilt))
    return snapped


def _canonical_triangles(points: np.ndarray) -> np.ndarray:
    order = np.lexsort((points[:, :, 2], points[:, :, 1], points[:, :, 0]), axis=1)
    return np.take_along_axis(points, order[:, :, None], axis=1)


def map_mesh_faces_to_scene(
    scene_vertices: np.ndarray,
    scene_faces: np.ndarray,
    scene_centres: np.ndarray,
    scene_tree: cKDTree,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    source_name: str,
) -> np.ndarray:
    """Map exact source-mesh triangles by centroid, verifying all three corners."""
    mapped = np.full(len(mesh_faces), -1, dtype=np.int64)
    for start in range(0, len(mesh_faces), 100_000):
        stop = min(start + 100_000, len(mesh_faces))
        faces = mesh_faces[start:stop]
        points = mesh_vertices[faces]
        centres = points.mean(axis=1)
        distances, nearest = scene_tree.query(centres, workers=-1)
        source_points = scene_vertices[scene_faces[nearest]]
        same = np.max(np.abs(_canonical_triangles(points) - _canonical_triangles(source_points)), axis=(1, 2))
        exact = (distances <= SOURCE_FACE_MATCH_TOLERANCE_M) & (same <= SOURCE_FACE_MATCH_TOLERANCE_M)
        mapped[start:stop][exact] = nearest[exact]

        # Resolve the rare case where distinct triangles share a centroid.
        for local in np.flatnonzero((distances <= SOURCE_FACE_MATCH_TOLERANCE_M) & ~exact):
            candidates = scene_tree.query_ball_point(centres[local], SOURCE_FACE_MATCH_TOLERANCE_M)
            candidate_points = scene_vertices[scene_faces[candidates]]
            candidate_error = np.max(
                np.abs(_canonical_triangles(candidate_points) - _canonical_triangles(points[local:local + 1])),
                axis=(1, 2),
            )
            matches = np.flatnonzero(candidate_error <= SOURCE_FACE_MATCH_TOLERANCE_M)
            if len(matches):
                mapped[start + local] = candidates[int(matches[0])]

    unmatched = int(np.count_nonzero(mapped < 0))
    if unmatched:
        raise ValueError(f"{source_name}: could not map {unmatched}/{len(mesh_faces)} triangles to the original mesh")
    return mapped


def _triangle_edge_keys(triangles: np.ndarray, vertex_count: int) -> np.ndarray:
    if not len(triangles):
        return np.empty(0, dtype=np.int64)
    edges = np.sort(np.concatenate((
        triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]],
    )), axis=1)
    return np.unique(edges[:, 0].astype(np.int64) * vertex_count + edges[:, 1])


def _edge_connected_face_components(face_ids: np.ndarray, faces: np.ndarray,
                                    vertex_count: int) -> list[np.ndarray]:
    if not len(face_ids):
        return []
    triangles = faces[face_ids]
    edges = np.sort(np.stack((
        triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]],
    ), axis=1).reshape(-1, 2), axis=1)
    edge_keys = edges[:, 0].astype(np.int64) * vertex_count + edges[:, 1]
    edge_faces = np.repeat(np.arange(len(face_ids)), 3)
    order = np.argsort(edge_keys, kind="stable")
    ordered_keys, ordered_faces = edge_keys[order], edge_faces[order]
    starts = np.r_[0, np.flatnonzero(ordered_keys[1:] != ordered_keys[:-1]) + 1]
    group_starts = np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    positions = np.arange(len(order))
    shared = positions > group_starts
    first, second = ordered_faces[group_starts[shared]], ordered_faces[positions[shared]]
    if len(first):
        adjacency = coo_matrix((np.ones(2 * len(first)),
                                (np.r_[first, second], np.r_[second, first])),
                               shape=(len(face_ids), len(face_ids))).tocsr()
        _, labels = connected_components(adjacency, directed=False)
    else:
        labels = np.arange(len(face_ids))
    components = [face_ids[labels == label] for label in np.unique(labels)]
    return sorted(components, key=lambda component: int(component.min()))


def _propagate_hole_fill_classes(vertices: np.ndarray, faces: np.ndarray,
                                 labels: np.ndarray, hole_fill: np.ndarray) -> tuple[np.ndarray, dict]:
    """Assign each filled mesh patch from its edge-connected surrounding class."""
    hole_faces = np.flatnonzero(hole_fill)
    updated = np.asarray(labels).copy()
    if not len(hole_faces):
        return updated, {"hole_fill_triangles": 0, "patch_count": 0, "patches": []}

    components = _edge_connected_face_components(hole_faces, faces, len(vertices))
    component_for_face = np.full(len(faces), -1, dtype=np.int32)
    for component_index, component in enumerate(components):
        component_for_face[component] = component_index

    non_fill_faces = np.flatnonzero(~hole_fill)
    non_fill_triangles = faces[non_fill_faces]
    non_fill_edges = np.sort(np.stack((
        non_fill_triangles[:, [0, 1]],
        non_fill_triangles[:, [1, 2]],
        non_fill_triangles[:, [2, 0]],
    ), axis=1).reshape(-1, 2), axis=1)
    non_fill_edge_keys = non_fill_edges[:, 0].astype(np.int64) * len(vertices) + non_fill_edges[:, 1]
    non_fill_edge_faces = np.repeat(non_fill_faces, 3)
    non_fill_edge_order = np.argsort(non_fill_edge_keys, kind="stable")

    fill_triangles = faces[hole_faces]
    fill_edges = np.stack((
        fill_triangles[:, [0, 1]],
        fill_triangles[:, [1, 2]],
        fill_triangles[:, [2, 0]],
    ), axis=1).reshape(-1, 2)
    fill_edge_keys = np.sort(fill_edges, axis=1)
    fill_edge_keys = fill_edge_keys[:, 0].astype(np.int64) * len(vertices) + fill_edge_keys[:, 1]
    fill_edge_faces = np.repeat(hole_faces, 3)
    fill_edge_lengths = np.linalg.norm(
        vertices[fill_edges[:, 0]] - vertices[fill_edges[:, 1]], axis=1
    )
    left = np.searchsorted(non_fill_edge_keys, fill_edge_keys, sorter=non_fill_edge_order)
    right = np.searchsorted(non_fill_edge_keys, fill_edge_keys, sorter=non_fill_edge_order, side="right")
    neighbor_counts = right - left

    boundary_records: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for edge_index in np.flatnonzero(neighbor_counts):
        fill_face = int(fill_edge_faces[edge_index])
        patch_index = int(component_for_face[fill_face])
        neighbor_ids = non_fill_edge_faces[
            non_fill_edge_order[left[edge_index]:right[edge_index]]
        ]
        for neighbor_face in neighbor_ids:
            class_id = int(labels[neighbor_face])
            if class_id < 0:
                continue
            boundary_records.setdefault((patch_index, class_id), []).append(
                (int(neighbor_face), float(fill_edge_lengths[edge_index]))
            )

    by_patch: list[dict[int, list[tuple[int, float]]]] = [{} for _ in components]
    for (patch_index, class_id), records in boundary_records.items():
        by_patch[patch_index][class_id] = records
    cosine_limit = float(np.cos(np.deg2rad(HOLE_FILL_PLANE_ANGLE_DEG)))
    patch_records = []
    for patch_index, patch_faces in enumerate(components):
        by_class = by_patch[patch_index]
        candidates = sorted(by_class)
        chosen_class = None
        method = "no_class_neighbor"
        candidate_scores = []
        if len(candidates) == 1:
            chosen_class = candidates[0]
            method = "single_neighbor_class"
        elif len(candidates) > 1:
            method = "no_coplanar_neighbor"
            patch_triangles = vertices[faces[patch_faces]]
            patch_cross = np.cross(
                patch_triangles[:, 1] - patch_triangles[:, 0],
                patch_triangles[:, 2] - patch_triangles[:, 0],
            )
            patch_areas = np.linalg.norm(patch_cross, axis=1) * 0.5
            valid_patch = patch_areas > 1e-12
            if valid_patch.any():
                patch_normals = patch_cross[valid_patch] / (2.0 * patch_areas[valid_patch, None])
                patch_centres = patch_triangles[valid_patch].mean(axis=1)
                reference_normal = patch_normals[np.argmax(patch_areas[valid_patch])]
                patch_normals *= np.where(patch_normals @ reference_normal < 0, -1.0, 1.0)[:, None]
                patch_normal = np.average(patch_normals, axis=0, weights=patch_areas[valid_patch])
                patch_normal /= max(np.linalg.norm(patch_normal), 1e-12)
                for class_id, records in by_class.items():
                    neighbor_ids = np.asarray([item[0] for item in records], dtype=np.int64)
                    boundary_lengths = np.asarray([item[1] for item in records], dtype=float)
                    neighbor_triangles = vertices[faces[neighbor_ids]]
                    neighbor_cross = np.cross(
                        neighbor_triangles[:, 1] - neighbor_triangles[:, 0],
                        neighbor_triangles[:, 2] - neighbor_triangles[:, 0],
                    )
                    neighbor_areas = np.linalg.norm(neighbor_cross, axis=1) * 0.5
                    valid_neighbors = neighbor_areas > 1e-12
                    neighbor_normals = np.zeros_like(neighbor_cross)
                    neighbor_normals[valid_neighbors] = (
                        neighbor_cross[valid_neighbors] / (2.0 * neighbor_areas[valid_neighbors, None])
                    )
                    similar_normals = valid_neighbors & (np.abs(neighbor_normals @ patch_normal) >= cosine_limit)
                    if not similar_normals.any():
                        candidate_scores.append({"class_id": class_id, "coplanar": False})
                        continue
                    oriented = neighbor_normals[similar_normals].copy()
                    oriented *= np.where(oriented @ patch_normal < 0, -1.0, 1.0)[:, None]
                    weights = boundary_lengths[similar_normals] * neighbor_areas[similar_normals]
                    candidate_normal = np.average(oriented, axis=0, weights=weights)
                    candidate_normal /= max(np.linalg.norm(candidate_normal), 1e-12)
                    angle_q90 = float(np.percentile(np.rad2deg(np.arccos(np.clip(
                        np.abs(patch_normals @ candidate_normal), 0.0, 1.0
                    ))), 90))
                    neighbor_centres = neighbor_triangles[similar_normals].mean(axis=1)
                    candidate_origin = np.average(neighbor_centres, axis=0, weights=weights)
                    residual_q90 = float(np.percentile(
                        np.abs((patch_centres - candidate_origin) @ candidate_normal), 90
                    ))
                    support_length = float(boundary_lengths[similar_normals].sum())
                    coplanar = angle_q90 <= HOLE_FILL_PLANE_ANGLE_DEG and residual_q90 <= PLANE_DISTANCE_M
                    candidate_scores.append({
                        "class_id": class_id,
                        "coplanar": coplanar,
                        "boundary_support_m": support_length,
                        "normal_difference_q90_deg": angle_q90,
                        "plane_residual_q90_m": residual_q90,
                    })
                    if coplanar:
                        score = (-support_length, angle_q90, residual_q90, class_id)
                        if chosen_class is None or score < chosen_score:
                            chosen_class, chosen_score = class_id, score
                if chosen_class is not None:
                    method = "most_confining_coplanar_neighbor"

        if chosen_class is not None:
            updated[patch_faces] = chosen_class
        patch_records.append({
            "patch_index": patch_index,
            "triangle_count": int(len(patch_faces)),
            "neighbor_class_ids": candidates,
            "assigned_class_id": chosen_class,
            "assignment_method": method,
            "candidate_scores": candidate_scores,
        })

    return updated, {
        "hole_fill_triangles": int(len(hole_faces)),
        "patch_count": int(len(components)),
        "assigned_triangles": int(np.count_nonzero(updated[hole_faces] >= 0)),
        "unassigned_triangles": int(np.count_nonzero(updated[hole_faces] < 0)),
        "newly_assigned_triangles": int(np.count_nonzero(
            (np.asarray(labels)[hole_faces] < 0) & (updated[hole_faces] >= 0)
        )),
        "multi_class_normal_tolerance_deg": HOLE_FILL_PLANE_ANGLE_DEG,
        "multi_class_plane_residual_tolerance_m": PLANE_DISTANCE_M,
        "patches": patch_records,
    }


def _load_object_instances(segmentation_dir: Path, vertices: np.ndarray,
                           faces: np.ndarray, centres: np.ndarray, scene_tree: cKDTree,
                           labels: np.ndarray, roles: dict[str, list[int]],
                           already_removed_faces: np.ndarray) -> list[dict]:
    metadata_path = segmentation_dir / "class_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Segmentation class metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text()).get("classes")
    if not isinstance(metadata, dict):
        raise ValueError(f"Segmentation metadata has no classes object: {metadata_path}")

    masks_dir = segmentation_dir / "3d_masks"
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Segmented object masks not found: {masks_dir}")
    protected_labels = np.unique(np.r_[roles["wall"], roles["floor"], roles["ceiling"]])
    already_removed_mask = np.zeros(len(faces), dtype=bool)
    already_removed_mask[already_removed_faces] = True
    result = []
    for class_dir in sorted(masks_dir.glob("class_*_object")):
        class_token = class_dir.name.removeprefix("class_").removesuffix("_object")
        if not class_token.isdecimal():
            raise ValueError(f"Unexpected segmented object class directory: {class_dir}")
        class_id = int(class_token)
        if metadata.get(str(class_id), {}).get("structural_role") != "object":
            continue
        for path in sorted(class_dir.glob("instance_*.ply")):
            instance_token = path.stem.removeprefix("instance_")
            if not instance_token.isdecimal():
                raise ValueError(f"Unexpected segmented instance path: {path}")
            mesh = o3d.io.read_triangle_mesh(str(path))
            mesh_vertices, mesh_faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
            if not len(mesh_faces):
                raise ValueError(f"Segmented object instance is empty: {path}")
            source_faces = np.unique(map_mesh_faces_to_scene(
                vertices, faces, centres, scene_tree, mesh_vertices, mesh_faces, str(path),
            ))
            can_remove = ~np.isin(labels[source_faces], protected_labels)
            can_remove &= ~already_removed_mask[source_faces]
            removable_faces = source_faces[can_remove]
            for component_index, component_faces in enumerate(
                    _edge_connected_face_components(removable_faces, faces, len(vertices))):
                result.append({
                    "source_class_id": class_id,
                    "instance_id": int(instance_token),
                    "component_index": component_index,
                    "source_path": str(path),
                    "source_triangle_count": int(len(component_faces)),
                    "edge_keys": _triangle_edge_keys(faces[component_faces], len(vertices)),
                    "removable_face_ids": component_faces,
                })
    return result


def _floor_component_neighbors(vertices: np.ndarray, faces: np.ndarray,
                               source_face_ids: np.ndarray, level: dict,
                               object_instances: list[dict]) -> tuple[set[int], set[int]]:
    floor_faces = faces[np.isin(source_face_ids, level["component_face_ids"])]
    if not len(floor_faces):
        raise ValueError("No projection-source triangles found for target floor component")
    floor_edges = _triangle_edge_keys(floor_faces, len(vertices))
    direct = {
        index for index, instance in enumerate(object_instances)
        if np.intersect1d(instance["edge_keys"], floor_edges, assume_unique=True).size
    }
    if not object_instances:
        return direct, set()

    edge_keys = np.concatenate([item["edge_keys"] for item in object_instances])
    edge_owners = np.repeat(np.arange(len(object_instances)), [len(item["edge_keys"]) for item in object_instances])
    order = np.argsort(edge_keys, kind="stable")
    ordered_keys, ordered_owners = edge_keys[order], edge_owners[order]
    starts = np.r_[0, np.flatnonzero(ordered_keys[1:] != ordered_keys[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(ordered_keys)])
    group_starts = np.repeat(starts, counts)
    positions = np.arange(len(ordered_keys))
    shared = positions > group_starts
    first_owners = ordered_owners[group_starts[shared]]
    other_owners = ordered_owners[positions[shared]]
    pairs = np.unique(np.sort(np.column_stack((first_owners, other_owners)), axis=1), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]

    neighbors = [set() for _ in object_instances]
    for first, second in pairs:
        neighbors[first].add(int(second))
        neighbors[second].add(int(first))
    second_hop = {neighbor for index in direct for neighbor in neighbors[index]} - direct
    return direct, second_hop


def _floating_component_exclusions(object_instances: list[dict], direct: set[int],
                                   second_hop: set[int]) -> tuple[np.ndarray, list[dict], list[dict]]:
    reachable = direct | second_hop
    removed, retained_large = [], []
    excluded_faces = []
    for index, instance in enumerate(object_instances):
        if index in reachable:
            continue
        record = {
            "source_class_id": instance["source_class_id"],
            "instance_id": instance["instance_id"],
            "component_index": instance["component_index"],
            "source_path": instance["source_path"],
            "triangle_count": instance["source_triangle_count"],
        }
        if instance["source_triangle_count"] < PROJECTION_MIN_FLOATING_COMPONENT_TRIANGLES:
            excluded_faces.append(instance["removable_face_ids"])
            removed.append(record)
        else:
            retained_large.append(record)
    excluded_face_ids = (np.unique(np.concatenate(excluded_faces))
                         if excluded_faces and any(len(ids) for ids in excluded_faces)
                         else np.empty(0, dtype=np.int64))
    return excluded_face_ids, removed, retained_large


def _unconnected_projection_component_exclusions(
        vertices: np.ndarray, faces: np.ndarray, selected: np.ndarray,
        floor_faces: np.ndarray, supported_object_faces: np.ndarray,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    removed, retained_large, excluded = [], [], []
    for component in _edge_connected_face_components(
            np.flatnonzero(selected), faces, len(vertices)):
        if floor_faces[component].any() or supported_object_faces[component].any():
            continue
        record = {"triangle_count": int(len(component))}
        if len(component) < PROJECTION_MIN_FLOATING_COMPONENT_TRIANGLES:
            excluded.append(component)
            removed.append(record)
        else:
            retained_large.append(record)
    excluded_face_ids = (np.concatenate(excluded) if excluded
                         else np.empty(0, dtype=np.int64))
    return excluded_face_ids, removed, retained_large


def _role_ids(structural_classes_path: Path) -> dict[str, list[int]]:
    data = json.loads(structural_classes_path.read_text())
    classes = data.get("classifications")
    if not isinstance(classes, dict):
        raise ValueError(f"No structural classifications in {structural_classes_path}")
    result = {"wall": [], "floor": [], "ceiling": []}
    for class_id, item in classes.items():
        role = str(item.get("structural_role", "")).lower()
        for target in result:
            if target in role:
                result[target].append(int(class_id))
    if not result["wall"] or not result["floor"] or not result["ceiling"]:
        raise ValueError("Structural metadata must identify wall, floor, and ceiling classes")
    return result


def _main_planes(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    class_ids: list[int],
    role: str,
    included: np.ndarray,
) -> list[dict]:
    role_face_ids = np.flatnonzero(included & np.isin(labels, class_ids))
    if not len(role_face_ids):
        return []
    role_faces = faces[role_face_ids]
    used_vertices, remapped = np.unique(role_faces, return_inverse=True)
    submesh = o3d.geometry.TriangleMesh()
    submesh.vertices = o3d.utility.Vector3dVector(vertices[used_vertices])
    submesh.triangles = o3d.utility.Vector3iVector(remapped.reshape(-1, 3))
    cluster_ids, _, areas = submesh.cluster_connected_triangles()
    clusters = component_face_indices(np.asarray(cluster_ids), len(areas))
    major = [i for i, area in enumerate(areas) if area >= MAIN_COMPONENT_AREA_M2]
    if not major and len(areas):
        major = [int(np.argmax(areas))]

    normal_sign = 1.0 if role == "floor" else -1.0
    planes = []
    for component_id in major:
        source_face_ids = role_face_ids[clusters[component_id]]
        component_faces = faces[source_face_ids]
        corners = vertices[component_faces]
        cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        double_areas = np.linalg.norm(cross, axis=1)
        valid = double_areas > 1e-12
        normals = np.zeros_like(cross)
        normals[valid] = cross[valid] / double_areas[valid, None]
        areas = double_areas / 2.0
        oriented_normals = normals.copy()
        oriented_normals[normals[:, 1] * normal_sign < 0] *= -1.0
        surface = valid & (oriented_normals[:, 1] * normal_sign >= MIN_PLANE_NORMAL_Y)
        if not surface.any():
            continue
        centres = corners.mean(axis=1)
        candidates = []
        available = surface.copy()
        while areas[available].sum() >= MIN_PLANE_AREA_M2:
            normal, offset, confidence = fit_robust_plane(
                centres[available], areas[available], PLANE_DISTANCE_M,
                role == "floor", oriented_normals[available],
            )
            if normal[1] * normal_sign < MIN_PLANE_NORMAL_Y:
                break
            compatible = oriented_normals @ normal >= np.cos(np.deg2rad(30.0))
            plane_faces = available & compatible & (np.abs(centres @ normal + offset) <= PLANE_DISTANCE_M)
            plane_area = float(areas[plane_faces].sum())
            if plane_area < MIN_PLANE_AREA_M2:
                break

            fit_points, fit_weights = centres[plane_faces], areas[plane_faces]
            # Orient signed distance so points above the plane are positive for floors and ceilings.
            mean_signed_distance = np.average(
                (fit_points @ normal + offset) * np.sign(normal[1]), weights=fit_weights
            )
            offset -= np.sign(normal[1]) * mean_signed_distance

            residual = np.abs(centres[plane_faces] @ normal + offset)
            candidates.append({
                "normal": normal,
                "offset": float(offset),
                "faces": plane_faces.copy(),
                "area": plane_area,
                "component_surface_area": float(areas[surface].sum()),
                "confidence": float(confidence),
                "tilt": _tilt_degrees(normal),
                "noise": _weighted_median(residual, fit_weights),
                "component": int(component_id),
                "component_area": float(areas.sum()),
            })
            available[plane_faces] = False
            if role == "ceiling":
                break

        if role == "ceiling":
            candidates = sorted(candidates, key=lambda item: item["area"], reverse=True)[:1]
            for candidate in candidates:
                planes.append(_plane_record(
                    candidate, role, centres, corners, areas, "surface",
                    component_face_ids=source_face_ids,
                    snap_horizontal=candidate["tilt"] < PLANE_SNAP_TILT_DEG,
                    snap_to_five=candidate["tilt"] >= PLANE_SNAP_TILT_DEG,
                ))
            continue

        ramps = [candidate for candidate in candidates if _is_supported_ramp(candidate, candidates, centres, areas)]
        ramp_ids = {id(candidate) for candidate in ramps}
        ramp_area = sum(candidate["area"] for candidate in ramps)
        level_area = sum(candidate["area"] for candidate in candidates if id(candidate) not in ramp_ids)
        component_surface_area = candidates[0]["component_surface_area"] if candidates else 0.0
        if (ramp_area >= level_area
                or ramp_area > RAMP_MAX_FLOOR_FRACTION * component_surface_area):
            ramps = []
        ramp_faces = {id(candidate) for candidate in ramps}
        for candidate in candidates:
            if candidate["area"] < MIN_PLANE_AREA_M2:
                continue
            is_ramp = id(candidate) in ramp_faces and _nearest_five_degree_tilt(candidate["tilt"]) > 0.0
            planes.append(_plane_record(
                candidate, role, centres, corners, areas,
                "ramp" if is_ramp else "surface",
                component_face_ids=source_face_ids,
                snap_horizontal=not is_ramp,
                snap_to_five=is_ramp,
            ))
    return planes


def _plane_record(candidate: dict, role: str, centres: np.ndarray, corners: np.ndarray,
                  areas: np.ndarray, kind: str, component_face_ids: np.ndarray,
                  snap_horizontal: bool = False,
                  snap_to_five: bool = False) -> dict:
    face_ids = np.flatnonzero(candidate["faces"])
    normal, offset = candidate["normal"].copy(), candidate["offset"]
    if snap_horizontal:
        normal = np.array([0.0, 1.0 if role == "floor" else -1.0, 0.0])
        offset = -normal[1] * np.average(centres[face_ids, 1], weights=areas[face_ids])
    elif snap_to_five:
        normal = _snap_normal_to_five_degrees(normal)
        offset = -np.average(centres[face_ids] @ normal, weights=areas[face_ids])
    plane_y = -(centres[face_ids, 0] * normal[0] + centres[face_ids, 2] * normal[2] + offset) / normal[1]
    mean_y = np.average(plane_y, weights=areas[face_ids])
    return {
        "role": role,
        "kind": kind,
        "component": candidate["component"],
        "component_face_ids": component_face_ids,
        "plane_face_ids": component_face_ids[face_ids],
        "component_area_m2": candidate["component_area"],
        "plane_area_m2": candidate["area"],
        "mean_y_m": float(mean_y),
        "tilt_deg": _tilt_degrees(normal),
        "plane_equation": [float(v) for v in (*normal, offset)],
        "fit_confidence": candidate["confidence"],
        "support_points": corners[face_ids].reshape(-1, 3),
    }


def _is_supported_ramp(candidate: dict, candidates: list[dict], centres: np.ndarray,
                       areas: np.ndarray) -> bool:
    """Accept a planar floor patch only when it bridges two distinct level areas."""
    normal, offset = candidate["normal"], candidate["offset"]
    gradient = -normal[[0, 2]] / normal[1]
    gradient_size = float(np.linalg.norm(gradient))
    if gradient_size < 1e-6 or candidate["area"] < MIN_PLANE_AREA_M2:
        return False
    axis = gradient / gradient_size
    face_ids = np.flatnonzero(candidate["faces"])
    ramp_points = centres[face_ids]
    ramp_xz = ramp_points[:, [0, 2]]
    ramp_t = ramp_xz @ axis
    run = _weighted_quantile(ramp_t, areas[face_ids], 0.98) - _weighted_quantile(ramp_t, areas[face_ids], 0.02)
    across = np.array([-axis[1], axis[0]])
    width = (_weighted_quantile(ramp_xz @ across, areas[face_ids], 0.98)
             - _weighted_quantile(ramp_xz @ across, areas[face_ids], 0.02))
    if run < RAMP_MIN_RUN_M or width < RAMP_MIN_WIDTH_M:
        return False

    residual = np.abs(ramp_points @ normal + offset)
    plateau_noise = max((item["noise"] for item in candidates if item is not candidate
                         and item["tilt"] < PLANE_SNAP_TILT_DEG), default=0.0)
    planarity_limit = max(RAMP_MIN_PLANARITY_M, min(RAMP_MAX_PLANARITY_M, 2.0 * plateau_noise))
    if _weighted_quantile(residual, areas[face_ids], 0.90) > planarity_limit:
        return False

    ramp_across = ramp_xz @ across
    ramp_across_min = _weighted_quantile(ramp_across, areas[face_ids], 0.02)
    ramp_across_max = _weighted_quantile(ramp_across, areas[face_ids], 0.98)
    contacts = []
    for level in candidates:
        if level is candidate or level["tilt"] >= PLANE_SNAP_TILT_DEG or level["area"] < MIN_PLANE_AREA_M2:
            continue
        level_ids = np.flatnonzero(level["faces"])
        level_points = centres[level_ids]
        level_xz = level_points[:, [0, 2]]
        lateral = level_xz @ across
        level_height = _weighted_median(level_points[:, 1], areas[level_ids])
        height_error = np.abs(gradient[0] * level_xz[:, 0] + gradient[1] * level_xz[:, 1]
                              - offset / normal[1] - level_height)
        contact_limit = max(0.03, min(0.10, 2.0 * level["noise"]))
        touching = (lateral >= ramp_across_min - RAMP_MAX_LATERAL_GAP_M)
        touching &= (lateral <= ramp_across_max + RAMP_MAX_LATERAL_GAP_M)
        touching &= height_error <= contact_limit
        if areas[level_ids[touching]].sum() < RAMP_MIN_CONTACT_AREA_M2:
            continue
        contact_t = _weighted_median(level_xz[touching] @ axis, areas[level_ids[touching]])
        contacts.append((contact_t, level_height, level["noise"]))

    ramp_min_t = _weighted_quantile(ramp_t, areas[face_ids], 0.02)
    ramp_max_t = _weighted_quantile(ramp_t, areas[face_ids], 0.98)
    for i, (first_t, first_y, first_noise) in enumerate(contacts):
        for second_t, second_y, second_noise in contacts[i + 1:]:
            delta_t, delta_y = second_t - first_t, second_y - first_y
            level_change = max(RAMP_MIN_RISE_M, 3.0 * max(first_noise, second_noise))
            if delta_t * delta_y <= 0 or abs(delta_y) <= level_change:
                continue
            if (abs(delta_t) < RAMP_MIN_RUN_M or min(first_t, second_t) > ramp_min_t + 0.35
                    or max(first_t, second_t) < ramp_max_t - 0.35):
                continue
            expected_rise = gradient_size * abs(delta_t)
            if abs(expected_rise - abs(delta_y)) <= max(0.10, 2.0 * max(first_noise, second_noise)):
                return True
    return False


def _append_chunk(vertices_out: list, faces_out: list, colors_out: list,
                  vertices: np.ndarray, faces: np.ndarray, color: np.ndarray | None = None,
                  vertex_colors: np.ndarray | None = None) -> None:
    if not len(faces):
        return
    used, inverse = np.unique(faces, return_inverse=True)
    offset = sum(len(chunk) for chunk in vertices_out)
    vertices_out.append(vertices[used])
    faces_out.append(inverse.reshape(-1, 3) + offset)
    colors_out.append(np.tile(color, (len(used), 1)) if color is not None else vertex_colors[used])


def _floor_levels(floor_planes: list[dict]) -> list[dict]:
    groups: list[list[dict]] = []
    for plane in sorted((item for item in floor_planes if item["kind"] != "ramp"),
                        key=lambda item: item["mean_y_m"]):
        if (not groups or plane["mean_y_m"]
                - np.average([item["mean_y_m"] for item in groups[-1]],
                             weights=[item["plane_area_m2"] for item in groups[-1]])
                > FLOOR_LEVEL_TOLERANCE_M):
            groups.append([plane])
        else:
            groups[-1].append(plane)

    levels = []
    for index, group in enumerate(groups):
        weights = np.asarray([item["plane_area_m2"] for item in group])
        elevation = float(np.average([item["mean_y_m"] for item in group], weights=weights))
        components = {}
        for item in group:
            components[item["component"]] = {
                "component": item["component"],
                "component_area_m2": item["component_area_m2"],
                "component_face_ids": item["component_face_ids"],
            }
        levels.append({
            "index": index,
            "mean_y_m": elevation,
            "equation": np.array([0.0, 1.0, 0.0, -elevation]),
            "plane_components": [item["component"] for item in group],
            "components": list(components.values()),
            "component_face_ids": np.unique(np.concatenate(
                [item["component_face_ids"] for item in group]
            )),
        })
    return levels


def _plane_y(equation: np.ndarray, points: np.ndarray) -> np.ndarray:
    return -(points[:, 0] * equation[0] + points[:, 2] * equation[2] + equation[3]) / equation[1]


def _triangles_outside_plane(vertices: np.ndarray, faces: np.ndarray,
                             equation: np.ndarray, above: bool) -> np.ndarray:
    outside = np.zeros(len(faces), dtype=bool)
    for start in range(0, len(faces), 100_000):
        stop = min(start + 100_000, len(faces))
        triangles = vertices[faces[start:stop]]
        plane_y = _plane_y(equation, triangles.reshape(-1, 3)).reshape(-1, 3)
        difference = triangles[:, :, 1] - plane_y
        outside[start:stop] = np.any(difference > 1e-9 if above else difference < -1e-9, axis=1)
    return outside


def _wall_faces_intersecting_level(vertices: np.ndarray, faces: np.ndarray,
                                   floor_equation: np.ndarray,
                                   ceiling_equation: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    points = triangles.reshape(-1, 3)
    floor_y = _plane_y(floor_equation, points).reshape(-1, 3)
    ceiling_y = _plane_y(ceiling_equation, points).reshape(-1, 3)
    reaches_above_floor = np.any(triangles[:, :, 1] >= floor_y - 1e-6, axis=1)
    reaches_below_ceiling = np.any(triangles[:, :, 1] <= ceiling_y + 1e-6, axis=1)
    return reaches_above_floor & reaches_below_ceiling


def _offset_plane(equation: np.ndarray, distance_along_normal: float) -> np.ndarray:
    """Shift a plane by a signed perpendicular distance along its normal."""
    shifted = np.asarray(equation, dtype=float).copy()
    normal_length = np.linalg.norm(shifted[:3])
    if normal_length <= 1e-12:
        raise ValueError("Plane equation has a zero-length normal")
    shifted /= normal_length
    shifted[3] -= distance_along_normal
    return shifted


def _nearest_lower_ceiling(level: dict, ceiling_planes: list[dict]) -> dict | None:
    lower = [item for item in ceiling_planes
             if 0.0 < level["mean_y_m"] - item["mean_y_m"] <= PROJECTION_LOWER_CEILING_MAX_GAP_M]
    return max(lower, key=lambda item: item["mean_y_m"], default=None)


def _projection_source_mask(retained_faces: np.ndarray, labels: np.ndarray,
                             wall_class_ids: list[int]) -> np.ndarray:
    return retained_faces & ~np.isin(labels, wall_class_ids)


def _select_projection_faces(below_lower_limit: np.ndarray, above_upper_limit: np.ndarray,
                             upper_floor_faces: np.ndarray, floor_instance_faces: np.ndarray,
                             lower_ceiling_instance_faces: np.ndarray,
                             upper_ceiling_instance_faces: np.ndarray) -> np.ndarray:
    selected = ~(below_lower_limit | above_upper_limit | upper_floor_faces)
    selected |= floor_instance_faces | lower_ceiling_instance_faces
    return selected & ~upper_ceiling_instance_faces


def _write_pre_collapse_mesh(path: Path, vertices: np.ndarray, triangles: np.ndarray,
                             face_color_ids: np.ndarray | None = None) -> None:
    if face_color_ids is None:
        face_color_ids = np.full(len(triangles), 2, dtype=np.int8)
    vertex_color_pairs = np.column_stack((
        triangles.reshape(-1), np.repeat(np.asarray(face_color_ids, dtype=np.int8), 3)
    ))
    unique_pairs, remapped = np.unique(vertex_color_pairs, axis=0, return_inverse=True)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices[unique_pairs[:, 0]])
    mesh.triangles = o3d.utility.Vector3iVector(remapped.reshape(-1, 3))
    mesh.vertex_colors = o3d.utility.Vector3dVector(PROJECTION_COLORS[unique_pairs[:, 1]])
    if not o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False,
                                      compressed=False, write_vertex_colors=True,
                                      write_vertex_normals=False):
        raise OSError(f"Failed to write pre-collapse floor mesh: {path}")


def _render_floor_projection(output_path: Path, level: dict, ceiling: dict,
                             lower_ceiling: dict | None,
                             vertices: np.ndarray, faces: np.ndarray, centres: np.ndarray,
                             labels: np.ndarray, roles: dict[str, list[int]],
                             floor_levels: list[dict], source_face_ids: np.ndarray,
                             wall_instance_vertices: np.ndarray,
                             wall_instance_faces: np.ndarray,
                             wall_vertices: np.ndarray, wall_faces: np.ndarray,
                             object_instances: list[dict],
                             include_wall_instances: bool,
                             wall_floor_clearance_m: float) -> dict:
    floor_y = _plane_y(level["equation"], centres)
    upper_cut = _offset_plane(ceiling["plane_equation"], PROJECTION_CEILING_CLEARANCE_M)
    lower_cut = (
        _offset_plane(level["equation"], -PROJECTION_FALLBACK_BELOW_FLOOR_M)
        if lower_ceiling is None
        else _offset_plane(lower_ceiling["plane_equation"], PROJECTION_CEILING_CLEARANCE_M)
    )
    below_lower_limit = _triangles_outside_plane(vertices, faces, lower_cut, above=False)
    above_upper_limit = _triangles_outside_plane(vertices, faces, upper_cut, above=True)
    ceiling_role_faces = np.isin(labels, roles["ceiling"])
    floor_faces = np.isin(labels, roles["floor"])
    upper_floor_faces = np.zeros(len(faces), dtype=bool)
    for other in floor_levels:
        if other["mean_y_m"] <= level["mean_y_m"] + FLOOR_LEVEL_TOLERANCE_M:
            continue
        other_floor_y = _plane_y(other["equation"], centres)
        upper_floor_faces |= floor_faces & (np.abs(centres[:, 1] - other_floor_y) <= PLANE_DISTANCE_M)

    floor_instance_faces = np.isin(source_face_ids, level["component_face_ids"])
    lower_ceiling_instance_faces = (
        np.isin(source_face_ids, lower_ceiling["component_face_ids"])
        if lower_ceiling else np.zeros(len(faces), dtype=bool)
    )
    upper_ceiling_instance_faces = np.isin(source_face_ids, ceiling["component_face_ids"])
    selected_before_component_filter = _select_projection_faces(
        below_lower_limit, above_upper_limit, upper_floor_faces, floor_instance_faces,
        lower_ceiling_instance_faces, upper_ceiling_instance_faces,
    )
    direct, second_hop = _floor_component_neighbors(
        vertices, faces, source_face_ids, level, object_instances
    )
    excluded_face_ids, removed_floaters, retained_large_floaters = _floating_component_exclusions(
        object_instances, direct, second_hop
    )
    excluded_instance_faces = np.isin(source_face_ids, excluded_face_ids)
    removed_floating_segmented_component_triangles = int(
        (selected_before_component_filter & excluded_instance_faces).sum()
    )
    selected = selected_before_component_filter & ~excluded_instance_faces
    supported_component_ids = direct | second_hop
    supported_object_face_ids = [object_instances[index]["removable_face_ids"]
                                 for index in supported_component_ids]
    supported_object_faces = (
        np.isin(source_face_ids, np.concatenate(supported_object_face_ids))
        if supported_object_face_ids else np.zeros(len(faces), dtype=bool)
    )
    excluded_mesh_component_ids, removed_mesh_floaters, retained_large_mesh_floaters = (
        _unconnected_projection_component_exclusions(
            vertices, faces, selected, floor_instance_faces, supported_object_faces,
        )
    )
    selected[excluded_mesh_component_ids] = False
    kept_face_ids = np.flatnonzero(selected)
    if not len(kept_face_ids) and not include_wall_instances:
        raise ValueError(f"No scene triangles remain in floor level {level['index']} height band")

    mesh_path = output_path.with_name(output_path.stem.replace("_topdown", "_pre_collapse") + ".ply")
    kept_color_ids = np.full(len(kept_face_ids), 2, dtype=np.int8)
    kept_color_ids[np.isin(labels[kept_face_ids], roles["floor"])] = 0
    kept_color_ids[np.isin(labels[kept_face_ids], roles["ceiling"])] = 1
    wall_instance_face_ids = np.empty(0, dtype=np.int64)
    if include_wall_instances:
        wall_band_floor = _offset_plane(level["equation"], wall_floor_clearance_m)
        wall_in_level = _wall_faces_intersecting_level(
            wall_instance_vertices, wall_instance_faces,
            wall_band_floor, ceiling["plane_equation"],
        )
        wall_instance_face_ids = np.flatnonzero(wall_in_level)
        # shortcut: keep whole intersecting triangles; clip only if their size causes visible band spill.
    wall_instance_triangles = wall_instance_faces[wall_instance_face_ids]
    projection_vertices = vertices
    projection_triangles = faces[kept_face_ids]
    if len(wall_instance_face_ids):
        projection_vertices = np.vstack((vertices, wall_instance_vertices))
        projection_triangles = np.vstack((
            projection_triangles,
            wall_instance_triangles + len(vertices),
        ))
        kept_color_ids = np.r_[kept_color_ids, np.full(len(wall_instance_face_ids), 2, dtype=np.int8)]
    _write_pre_collapse_mesh(mesh_path, projection_vertices, projection_triangles, kept_color_ids)
    projected_triangle_count = len(projection_triangles)
    saved_mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    projection_vertices = np.asarray(saved_mesh.vertices)
    projection_faces = np.asarray(saved_mesh.triangles)
    if len(projection_faces) != projected_triangle_count:
        raise OSError(f"Pre-collapse PLY triangle count mismatch: {mesh_path}")
    projection_centres = _centres(projection_vertices, projection_faces)

    # Highest surfaces win the top-down projection where collapsed faces overlap.
    draw_order = np.argsort(projection_centres[:, 1], kind="stable")
    projected_triangles = projection_vertices[projection_faces[draw_order]][:, :, (0, 2)]
    projected_labels = np.r_[labels[kept_face_ids], np.full(len(wall_instance_face_ids), -1)][draw_order]
    face_colors = np.tile([0.58, 0.58, 0.58, 0.35], (len(draw_order), 1))
    face_colors[np.isin(projected_labels, roles["floor"])] = [0.92, 0.04, 0.04, 0.88]
    face_colors[np.isin(projected_labels, roles["ceiling"])] = [0.04, 0.15, 0.92, 0.82]
    min_x, min_z = projected_triangles.reshape(-1, 2).min(axis=0)
    max_x, max_z = projected_triangles.reshape(-1, 2).max(axis=0)
    pad = max(max_x - min_x, max_z - min_z, 1.0) * 0.02

    fig, ax = plt.subplots(figsize=(13, 11), dpi=180)
    batch_size = 100_000
    for start in range(0, len(draw_order), batch_size):
        stop = min(start + batch_size, len(draw_order))
        collection = PolyCollection(
            projected_triangles[start:stop], facecolors=face_colors[start:stop],
            edgecolors="none", linewidths=0, antialiased=False, rasterized=True,
        )
        collection.set_zorder(1 + start // batch_size)
        ax.add_collection(collection)

    def whole_meter_label(value: float, _position: int) -> str:
        nearest = round(value)
        return f"{nearest:d}" if abs(value - nearest) < 1e-8 else ""

    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_z - pad, max_z + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(FuncFormatter(whole_meter_label))
    ax.yaxis.set_major_formatter(FuncFormatter(whole_meter_label))
    ax.grid(which="major", color="#858585", linewidth=0.65, alpha=0.55, zorder=100)
    ax.grid(which="minor", color="#b8b8b8", linewidth=0.35, alpha=0.32, zorder=99)
    ax.tick_params(which="major", length=7, width=1.0)
    ax.tick_params(which="minor", length=3, width=0.6)
    ax.set_title(
        f"Floor {level['index']} — Y collapsed to {level['mean_y_m']:.3f} m; "
        f"upper ceiling {ceiling['mean_y_m']:.3f} m ({PROJECTION_CEILING_CLEARANCE_M * 100:.0f} cm offset)"
    )
    projection_legend = [
        Patch(facecolor="#e60a0a", alpha=0.88, label="Floor triangles"),
        Patch(facecolor="#0a26e6", alpha=0.82, label="Ceiling triangles"),
        Patch(facecolor="#949494", alpha=0.35, label="Other triangles"),
    ]
    ax.legend(handles=projection_legend, loc="upper right", framealpha=0.9, fontsize=8)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")

    wall_in_level = _wall_faces_intersecting_level(
        wall_vertices, wall_faces, level["equation"], ceiling["plane_equation"]
    )
    wall_face_ids = np.flatnonzero(wall_in_level)
    wall_triangles = wall_vertices[wall_faces[wall_face_ids]][:, :, (0, 2)]
    if len(wall_triangles):
        first = wall_triangles[:, 1] - wall_triangles[:, 0]
        second = wall_triangles[:, 2] - wall_triangles[:, 0]
        projected_area = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) * 0.5
        polygon_mask = projected_area > 1e-6
        if polygon_mask.any():
            ax.add_collection(PolyCollection(
                wall_triangles[polygon_mask], facecolors="#20a640", edgecolors="none",
                alpha=0.3, rasterized=True, zorder=5,
            ))
        line_triangles = wall_triangles[~polygon_mask]
        if len(line_triangles):
            pairs = np.array([[0, 1], [0, 2], [1, 2]])
            endpoints = line_triangles[:, pairs]
            lengths = np.sum(np.diff(endpoints, axis=2) ** 2, axis=(2, 3))
            longest = np.argmax(lengths, axis=1)
            segments = endpoints[np.arange(len(endpoints)), longest]
            has_length = lengths.max(axis=1) > 1e-12
            segments = segments[has_length]
            if len(segments):
                reverse = ((segments[:, 1, 0] < segments[:, 0, 0])
                           | ((segments[:, 1, 0] == segments[:, 0, 0])
                              & (segments[:, 1, 1] < segments[:, 0, 1])))
                segments[reverse] = segments[reverse, ::-1]
                segments = np.unique(np.round(segments, 4).reshape(-1, 4), axis=0).reshape(-1, 2, 2)
                ax.add_collection(LineCollection(
                    segments, colors=[(0.125, 0.65, 0.25, 0.3)], linewidths=7.5, zorder=5,
                ))
        ax.legend(handles=projection_legend + [
            Patch(facecolor="#20a640", alpha=0.3, label="Fitted wall planes")
        ], loc="upper right", framealpha=0.9, fontsize=8)
    overlay_path = output_path.with_name(output_path.stem + "_walls_overlay.png")
    fig.savefig(overlay_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    records = lambda indices: [
        {"source_class_id": object_instances[index]["source_class_id"],
         "instance_id": object_instances[index]["instance_id"],
         "component_index": object_instances[index]["component_index"],
         "source_path": object_instances[index]["source_path"],
         "triangle_count": object_instances[index]["source_triangle_count"]}
        for index in sorted(indices)
    ]
    return {
        "floor_index": level["index"],
        "floor_y_m": level["mean_y_m"],
        "floor_plane_components": level["plane_components"],
        "ceiling_component": ceiling["component"],
        "ceiling_mean_y_m": ceiling["mean_y_m"],
        "removed_upper_ceiling_instance_triangles": int(upper_ceiling_instance_faces.sum()),
        "upper_cut_plane_equation": upper_cut.tolist(),
        "upper_ceiling_clearance_m": PROJECTION_CEILING_CLEARANCE_M,
        "component_filter_rule": f"Split each segmented object instance into source-mesh edge-connected pieces; keep pieces touching the target floor or first-hop pieces. Then remove selected-mesh components under {PROJECTION_MIN_FLOATING_COMPONENT_TRIANGLES:,} triangles unless they contain floor or retained two-hop object support; larger disconnected components remain.",
        "component_connection": "shared source-mesh triangle edge",
        "component_max_hops_from_floor": 2,
        "floating_component_triangle_threshold": PROJECTION_MIN_FLOATING_COMPONENT_TRIANGLES,
        "direct_floor_connected_components": records(direct),
        "second_hop_floor_connected_components": records(second_hop),
        "removed_floating_components_below_threshold": removed_floaters,
        "retained_disconnected_components_at_or_above_threshold": retained_large_floaters,
        "removed_floating_segmented_component_triangles": removed_floating_segmented_component_triangles,
        "removed_unconnected_projection_components_below_threshold": removed_mesh_floaters,
        "retained_unconnected_projection_components_at_or_above_threshold": retained_large_mesh_floaters,
        "removed_unconnected_projection_triangles": int(len(excluded_mesh_component_ids)),
        "wall_instance_triangles_added": int(len(wall_instance_face_ids)),
        "bounds_fitted_wall_faces": int(len(wall_face_ids)),
        "lower_limit_source": "ceiling_beneath_floor" if lower_ceiling else "floor_fallback",
        "lower_ceiling_component": lower_ceiling["component"] if lower_ceiling else None,
        "lower_ceiling_mean_y_m": lower_ceiling["mean_y_m"] if lower_ceiling else None,
        "lower_cut_plane_equation": lower_cut.tolist(),
        "lower_clearance_m": PROJECTION_CEILING_CLEARANCE_M if lower_ceiling else PROJECTION_FALLBACK_BELOW_FLOOR_M,
        "included_floor_instance_triangles": int((floor_instance_faces & selected).sum()),
        "included_lower_ceiling_instance_triangles": int((lower_ceiling_instance_faces & selected).sum()),
        "projected_triangle_count": int(projected_triangle_count),
        "pre_collapse_mesh": str(mesh_path),
        "projected_below_floor_ceiling_triangles": int((ceiling_role_faces & (centres[:, 1] < floor_y) & selected).sum()),
        "removed_upper_floor_triangles": int(upper_floor_faces.sum()),
        "removed_below_or_above_band_triangles": int((below_lower_limit | above_upper_limit).sum()),
        "image": str(output_path),
        "walls_overlay_image": str(overlay_path),
    }

def project_floor_levels(vertices: np.ndarray, faces: np.ndarray, centres: np.ndarray,
                         labels: np.ndarray, roles: dict[str, list[int]],
                         floor_planes: list[dict], ceiling_planes: list[dict],
                         output_dir: Path, source_face_ids: np.ndarray,
                         wall_instance_vertices: np.ndarray,
                         wall_instance_faces: np.ndarray,
                         wall_vertices: np.ndarray, wall_faces: np.ndarray,
                         fitted_wall_planes_path: Path,
                         axis_alignment: dict,
                         object_instances: list[dict],
                         include_wall_instances: bool,
                         wall_floor_clearance_m: float) -> list[dict]:
    floor_levels = _floor_levels(floor_planes)
    if not floor_levels:
        raise ValueError("No horizontal floor levels are available for projection")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for level in floor_levels:
        higher_ceilings = [item for item in ceiling_planes if item["mean_y_m"] > level["mean_y_m"]]
        if not higher_ceilings:
            raise ValueError(f"No ceiling plane is above floor level {level['index']}")
        ceiling = min(higher_ceilings, key=lambda item: item["mean_y_m"])
        lower_ceiling = _nearest_lower_ceiling(level, ceiling_planes)
        output_path = output_dir / f"floor_{level['index']:02d}_topdown.png"
        manifest.append(_render_floor_projection(
            output_path, level, ceiling, lower_ceiling, vertices, faces, centres, labels, roles,
            floor_levels, source_face_ids, wall_instance_vertices, wall_instance_faces,
            wall_vertices, wall_faces, object_instances, include_wall_instances,
            wall_floor_clearance_m,
        ))
    ceiling_manifest = []
    for ceiling_index, ceiling in enumerate(ceiling_planes):
        global_face_ids = np.asarray(
            ceiling.get("plane_face_ids", ceiling["component_face_ids"]), dtype=np.int64
        )
        local_face_ids = np.searchsorted(source_face_ids, global_face_ids)
        valid = local_face_ids < len(source_face_ids)
        valid[valid] &= source_face_ids[local_face_ids[valid]] == global_face_ids[valid]
        local_face_ids = local_face_ids[valid]
        if not len(local_face_ids):
            raise ValueError(f"Ceiling component {ceiling['component']} has no projection-source triangles")
        projection_path = output_dir / f"ceiling_{ceiling_index:02d}_pre_collapse.ply"
        _write_pre_collapse_mesh(
            projection_path, vertices, faces[local_face_ids],
            np.ones(len(local_face_ids), dtype=np.int8),
        )
        ceiling_manifest.append({
            "ceiling_index": ceiling_index,
            "component": ceiling["component"],
            "mean_y_m": ceiling["mean_y_m"],
            "tilt_deg": ceiling["tilt_deg"],
            "plane_equation": ceiling["plane_equation"],
            "projection_mesh": str(projection_path),
            "support_triangle_count": int(len(local_face_ids)),
            "hole_fill_support_triangle_count": ceiling.get("hole_fill_support_triangles", 0),
            "support_area_m2": ceiling["plane_area_m2"],
        })
    (output_dir / "floor_projection_manifest.json").write_text(json.dumps({
        "source_mesh": "original_scene_mesh",
        "fitted_wall_planes_mesh": str(fitted_wall_planes_path),
        "axis_alignment": axis_alignment,
        "projection_geometry_source": "saved_per_floor_pre_collapse_ply",
        "wall_instances_added_after_floater_cleanup": include_wall_instances,
        "wall_instance_floor_clearance_m": wall_floor_clearance_m if include_wall_instances else None,
        "wall_instance_band_selection": "Triangles intersecting from the target floor plane plus the configured clearance through the next higher ceiling plane are appended after floater cleanup; intersecting triangles are retained whole.",
        "walls_overlay_color": "semi-transparent green (30% opacity)",
        "projection_ply_colors": "floor red, ceiling blue, other elements gray; RGB PLY colors are opaque",
        "projection_source_triangle_count": int(len(faces)),
        "hole_fill_class_assignment": str(output_dir.parent / "hole_fill_class_assignment.json")
            if (output_dir.parent / "hole_fill_class_assignment.json").is_file() else None,
        "ceiling_projection_rule": "Each fitted ceiling support patch is saved separately with its original Y coordinates for X/Z shape fitting.",
        "ceiling_planes": ceiling_manifest,
        "projection_axes": ["X", "Z"],
        "height_selection": "15 cm parallel offsets toward the floor from the next higher ceiling and beneath a lower ceiling within 40 cm; triangles crossing either cut are removed; full target floor and qualifying lower ceiling components included; otherwise the lower bound is 10 cm below the floor; full next-higher ceiling component excluded",
        "component_filter_rule": f"Split segmented object instances into source-mesh edge-connected pieces; retain pieces touching the target floor or touching a first-hop piece. Also inspect every selected projection-mesh component, removing disconnected components under {PROJECTION_MIN_FLOATING_COMPONENT_TRIANGLES:,} triangles unless supported by the floor or a retained two-hop object piece; larger unconnected components remain.",
        "component_connection": "shared source-mesh triangle edge",
        "component_max_hops_from_floor": 2,
        "floating_component_triangle_threshold": PROJECTION_MIN_FLOATING_COMPONENT_TRIANGLES,
        "grid_interval_m": 0.1,
        "tick_interval_m": 0.5,
        "label_interval_m": 1.0,
        "levels": manifest,
    }, indent=2) + "\n")
    return manifest


def build_evaluation_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    labels: np.ndarray,
    removed_faces: np.ndarray,
    roles: dict[str, list[int]],
    planes: list[dict],
) -> o3d.geometry.TriangleMesh:
    keep = np.ones(len(faces), dtype=bool)
    keep[np.unique(removed_faces)] = False
    remaining_ids = np.flatnonzero(keep)
    remaining_faces = faces[remaining_ids]
    remaining_labels = labels[remaining_ids]
    floor = np.isin(remaining_labels, roles["floor"])
    ceiling = np.isin(remaining_labels, roles["ceiling"])
    other = ~(floor | ceiling)

    vertices_out, faces_out, colors_out = [], [], []
    _append_chunk(vertices_out, faces_out, colors_out, vertices, remaining_faces[other], vertex_colors=colors)
    _append_chunk(vertices_out, faces_out, colors_out, vertices, remaining_faces[floor], color=FLOOR_COLOR)
    _append_chunk(vertices_out, faces_out, colors_out, vertices, remaining_faces[ceiling], color=CEILING_COLOR)
    for plane in planes:
        normal = np.asarray(plane["plane_equation"][:3])
        offset = plane["plane_equation"][3]
        support = plane["support_points"]
        x_min, z_min = support[:, [0, 2]].min(axis=0)
        x_max, z_max = support[:, [0, 2]].max(axis=0)
        corners_xz = np.array([[x_min, z_min], [x_max, z_min], [x_max, z_max], [x_min, z_max]])
        plane_vertices = np.column_stack((
            corners_xz[:, 0],
            -(normal[0] * corners_xz[:, 0] + normal[2] * corners_xz[:, 1] + offset) / normal[1],
            corners_xz[:, 1],
        ))
        plane_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        if np.dot(np.cross(plane_vertices[1] - plane_vertices[0],
                           plane_vertices[2] - plane_vertices[0]), normal) < 0:
            plane_faces[:, [1, 2]] = plane_faces[:, [2, 1]]
        color = FLOOR_PLANE_COLOR if plane["role"] == "floor" else CEILING_PLANE_COLOR
        _append_chunk(vertices_out, faces_out, colors_out, plane_vertices,
                      plane_faces, color=color)

    output = o3d.geometry.TriangleMesh()
    output.vertices = o3d.utility.Vector3dVector(np.vstack(vertices_out))
    output.triangles = o3d.utility.Vector3iVector(np.vstack(faces_out))
    output.vertex_colors = o3d.utility.Vector3dVector(np.vstack(colors_out))
    return output


def main() -> int:
    args = parse_args()
    if not np.isfinite(args.wall_instance_floor_clearance_m) or args.wall_instance_floor_clearance_m < 0:
        raise ValueError("Wall-instance floor clearance must be finite and non-negative")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation mesh: {args.output}")
    scene = o3d.io.read_triangle_mesh(str(args.scene_mesh))
    vertices, faces = np.asarray(scene.vertices), np.asarray(scene.triangles)
    if not len(faces):
        raise ValueError(f"Scene mesh has no triangles: {args.scene_mesh}")
    labels = np.load(args.labels)
    if len(labels) != len(faces):
        raise ValueError(f"Label count {len(labels)} does not match scene triangles {len(faces)}")
    hole_fill_flags = read_ply_face_flags(args.scene_mesh, "is_hole_fill")
    hole_fill_stats = {"source_mesh": str(args.scene_mesh), "flag_property_present": hole_fill_flags is not None}
    if hole_fill_flags is not None:
        if len(hole_fill_flags) != len(faces):
            raise ValueError(
                f"Hole-fill flag count {len(hole_fill_flags)} does not match scene triangles {len(faces)}"
            )
        original_labels = labels
        labels, propagated = _propagate_hole_fill_classes(
            vertices, faces, labels, hole_fill_flags == 1
        )
        hole_fill_stats.update(propagated)
        hole_fill_stats.update({
            "unknown_fill_triangles_before": int(np.count_nonzero((original_labels < 0) & (hole_fill_flags == 1))),
            "unknown_fill_triangles_after": int(np.count_nonzero((labels < 0) & (hole_fill_flags == 1))),
            "assigned_class_triangle_counts": {
                str(class_id): int(np.count_nonzero(labels[hole_fill_flags == 1] == class_id))
                for class_id in np.unique(labels[hole_fill_flags == 1]) if class_id >= 0
            },
        })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        labels_path = args.output.parent / "triangle_class_labels_holefill_propagated.npy"
        np.save(labels_path, labels)
        hole_fill_stats["derived_labels"] = str(labels_path)
        assignment_path = args.output.parent / "hole_fill_class_assignment.json"
        hole_fill_stats["assignment_manifest"] = str(assignment_path)
        assignment_path.write_text(json.dumps(hole_fill_stats, indent=2) + "\n")
        print(
            f"Propagated labels to {hole_fill_stats['assigned_triangles']:,}/"
            f"{hole_fill_stats['hole_fill_triangles']:,} hole-fill triangles across "
            f"{hole_fill_stats['patch_count']} patches; "
            f"{hole_fill_stats['unassigned_triangles']:,} remain unassigned"
        )
    if not args.wall_filtered_mesh.is_file():
        raise FileNotFoundError(f"Horizontal-filtered wall mesh not found: {args.wall_filtered_mesh}")
    colors = np.asarray(scene.vertex_colors)
    if len(colors) != len(vertices):
        colors = np.full((len(vertices), 3), 0.65)
    roles = _role_ids(args.structural_classes)

    scene_centres = _centres(vertices, faces)
    scene_tree = cKDTree(scene_centres)
    wall_mesh = o3d.io.read_triangle_mesh(str(args.wall_filtered_mesh))
    wall_vertices, wall_source_faces = np.asarray(wall_mesh.vertices), np.asarray(wall_mesh.triangles)
    if not len(wall_source_faces):
        raise ValueError(f"Horizontal-filtered wall mesh has no triangles: {args.wall_filtered_mesh}")
    if not args.fitted_wall_planes.is_file():
        raise FileNotFoundError(f"Fitted wall planes not found: {args.fitted_wall_planes}")
    fitted_wall_planes = o3d.io.read_triangle_mesh(str(args.fitted_wall_planes))
    fitted_wall_vertices = np.asarray(fitted_wall_planes.vertices)
    fitted_wall_faces = np.asarray(fitted_wall_planes.triangles)
    if not len(fitted_wall_faces):
        raise ValueError(f"Fitted wall planes have no triangles: {args.fitted_wall_planes}")
    wall_faces = map_mesh_faces_to_scene(
        vertices, faces, scene_centres, scene_tree,
        wall_vertices, wall_source_faces, str(args.wall_filtered_mesh),
    )
    wall_faces = wall_faces[np.isin(labels[wall_faces], roles["wall"])]
    if hole_fill_flags is not None:
        wall_fill_faces = np.flatnonzero((hole_fill_flags == 1) & np.isin(labels, roles["wall"]))
        wall_faces = np.unique(np.r_[wall_faces, wall_fill_faces])

    associations_path = args.associations or args.segmentation_output / "non_structural_similarity" / "valid_class_associations.json"
    if not associations_path.is_file() and args.associations is None:
        associations_path = args.scene_mesh.parent / "non_structural_similarity" / "valid_class_associations.json"
    associations = load_door_window_associations(associations_path)
    windows = load_associated_object_instances(
        {class_id: 2 for class_id, text_id in associations.items() if text_id == 2},
        args.segmentation_output,
    )
    window_faces = []
    for item in windows:
        mapped = map_mesh_faces_to_scene(
            vertices, faces, scene_centres, scene_tree,
            item["vertices"], item["triangles"], item["source_path"],
        )
        window_faces.append(mapped)
    window_faces = np.concatenate(window_faces) if window_faces else np.empty(0, dtype=np.int64)
    removed_faces = np.unique(np.r_[wall_faces, window_faces])
    object_instances = _load_object_instances(
        args.segmentation_output, vertices, faces, scene_centres, scene_tree,
        labels, roles, removed_faces,
    )
    print(f"Mapped {len(object_instances)} edge-connected segmented object pieces for floor connectivity filtering")

    rotation, axis_alignment = _axis_alignment_from_wall_manifest(args.fitted_wall_planes)
    vertices = (rotation @ vertices.T).T
    wall_vertices = (rotation @ wall_vertices.T).T
    fitted_wall_vertices = (rotation @ fitted_wall_vertices.T).T
    scene_centres = _centres(vertices, faces)
    print(f"Axis-aligned scene and projection meshes by {-axis_alignment['source_orientation_deg']:.3f}° around Y")

    remaining = np.ones(len(faces), dtype=bool)
    remaining[removed_faces] = False
    floor_planes = _main_planes(vertices, faces, labels, roles["floor"], "floor", remaining)
    ceiling_planes = _main_planes(vertices, faces, labels, roles["ceiling"], "ceiling", remaining)
    if hole_fill_flags is not None:
        for plane in floor_planes + ceiling_planes:
            plane["hole_fill_support_triangles"] = int(
                np.count_nonzero(hole_fill_flags[plane["plane_face_ids"]] == 1)
            )
    planes = floor_planes + ceiling_planes
    if not floor_planes or not ceiling_planes:
        raise ValueError(f"Expected fitted floor and ceiling planes; found {len(floor_planes)} and {len(ceiling_planes)}")

    result = build_evaluation_mesh(vertices, faces, colors, labels, removed_faces, roles, planes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(args.output), result, write_ascii=False,
                                     compressed=False, write_vertex_colors=True,
                                     write_vertex_normals=False):
        raise OSError(f"Failed to write evaluation mesh: {args.output}")
    projection_faces = _projection_source_mask(remaining, labels, roles["wall"])
    assert not np.isin(labels[projection_faces], roles["wall"]).any()
    print(f"Projection source triangles after removing walls/windows: {int(projection_faces.sum()):,}")
    projections = project_floor_levels(
        vertices, faces[projection_faces], scene_centres[projection_faces], labels[projection_faces], roles,
        floor_planes, ceiling_planes, args.output.parent / "floor_projections",
        np.flatnonzero(projection_faces), wall_vertices, wall_source_faces,
        fitted_wall_vertices, fitted_wall_faces,
        args.fitted_wall_planes, axis_alignment, object_instances,
        args.include_wall_instances, args.wall_instance_floor_clearance_m,
    )
    print(f"Original triangles: {len(faces):,}")
    print(f"Removed cleaned wall triangles: {len(np.unique(wall_faces)):,}")
    print(f"Removed window instance triangles: {len(np.unique(window_faces)):,}")
    print(f"Residual triangles: {int(remaining.sum()):,}; fitted planes: {len(planes)}")
    for plane in planes:
        print(f"  {plane['role']} {plane['kind']} #{plane['component']}: "
              f"mean Y={plane['mean_y_m']:.3f} m, tilt={plane['tilt_deg']:.2f} deg, "
              f"support={plane['plane_area_m2']:.2f} m2, normal={np.round(plane['plane_equation'][:3], 4).tolist()}")
    print(f"Saved combined evaluation mesh: {args.output}")
    for projection in projections:
        print(f"Saved floor {projection['floor_index']} projection: {projection['projected_triangle_count']:,} triangles -> "
              f"{projection['image']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
