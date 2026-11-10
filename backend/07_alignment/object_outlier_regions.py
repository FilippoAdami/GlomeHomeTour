"""Detect and mark final object regions extending beyond their regularized room shell."""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import plyfile
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import Polygon
from shapely.ops import unary_union

try:
    from shapely import contains_xy
except ImportError:
    contains_xy = None


AREA_LIMIT_M2 = 0.5
DEPTH_LIMIT_M = 0.05
OUTSIDE_EPSILON_M = 0.001
PATCH_GAP_M = 0.001
SEAM_LIMIT = 3.0
RECTANGLE_LIMIT = 0.7
ELONGATED_TRIANGLE_ASPECT_RATIO = 20.0
ELONGATED_BOUNDARY_WEIGHT = 0.5
FACE_MARKER = "is_phase07_outlier"
CHUNK_SIZE = 100_000


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"Expected a non-empty triangular mesh: {path}")
    return mesh


def _inside_footprint(footprint, points: np.ndarray) -> np.ndarray:
    if contains_xy is not None:
        return np.asarray(contains_xy(footprint, points[:, 0], points[:, 1]), dtype=bool)
    from shapely.geometry import Point
    return np.asarray([footprint.covers(Point(float(x), float(z))) for x, z in points], dtype=bool)


def _make_ray_scene(triangles: np.ndarray):
    vertices = np.asarray(triangles, dtype=np.float32).reshape(-1, 3)
    faces = np.arange(len(vertices), dtype=np.uint32).reshape(-1, 3)
    mesh = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(vertices), o3d.core.Tensor(faces))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    return scene


def _build_exterior_surface(scan, transform: np.ndarray) -> dict[str, Any]:
    elements = scan.info.regularized_elements_dir
    if not elements or not elements.is_dir() or not scan.info.regularized_walls_path:
        raise FileNotFoundError(f"{scan.info.scan_id} has no Phase 06 regularized elements")
    floors = [_load_mesh(path) for path in scan.info.regularized_floor_paths]
    ceilings = [_load_mesh(path) for path in scan.info.regularized_ceiling_paths]
    if not floors or not ceilings:
        raise FileNotFoundError(f"{scan.info.scan_id} requires regularized floor and ceiling elements")
    for mesh in floors + ceilings:
        mesh.apply_transform(transform)

    floor_polygons = []
    for mesh in floors:
        for triangle in mesh.vertices[mesh.faces]:
            polygon = Polygon(triangle[:, [0, 2]])
            if polygon.area > 1e-9:
                floor_polygons.append(polygon)
    footprint = unary_union(floor_polygons)
    if footprint.is_empty:
        raise ValueError(f"{scan.info.scan_id} regularized floors have no XZ footprint")

    walls = _load_mesh(scan.info.regularized_walls_path)
    walls.apply_transform(transform)
    normals = walls.face_normals
    axes = np.abs(normals).argmax(axis=1)
    axis_normal = np.abs(normals[np.arange(len(normals)), axes])
    candidates = np.flatnonzero((axis_normal >= 0.98) & ((axes == 0) | (axes == 2)))
    centers = walls.triangles_center[candidates][:, [0, 2]]
    coordinate_axis = np.where(axes[candidates] == 0, 0, 1)
    plus, minus = centers.copy(), centers.copy()
    rows = np.arange(len(candidates))
    # shortcut: floor-footprint side probes orient perimeter walls; refine if complex floor outlines misclassify a side.
    plus[rows, coordinate_axis] += 0.02
    minus[rows, coordinate_axis] -= 0.02
    plus_inside, minus_inside = _inside_footprint(footprint, plus), _inside_footprint(footprint, minus)
    perimeter = candidates[plus_inside != minus_inside]

    triangle_parts = [walls.vertices[walls.faces[perimeter]]]
    axis_parts = [axes[perimeter].astype(np.int8)]
    sign_parts = [np.where(plus_inside[plus_inside != minus_inside], -1, 1).astype(np.int8)]
    for meshes, sign in ((floors, -1), (ceilings, 1)):
        for mesh in meshes:
            face_normals = mesh.face_normals
            face_axes = np.abs(face_normals).argmax(axis=1)
            horizontal = (face_axes == 1) & (np.abs(face_normals[np.arange(len(face_normals)), face_axes]) >= 0.98)
            if np.any(horizontal):
                triangle_parts.append(mesh.vertices[mesh.faces[horizontal]])
                axis_parts.append(np.ones(int(horizontal.sum()), dtype=np.int8))
                sign_parts.append(np.full(int(horizontal.sum()), sign, dtype=np.int8))

    triangles = np.concatenate(triangle_parts).astype(np.float32)
    if not len(triangles):
        raise ValueError(f"{scan.info.scan_id} has no external regularized faces")
    surface = {
        "triangles": triangles,
        "axes": np.concatenate(axis_parts),
        "signs": np.concatenate(sign_parts),
        "scene": _make_ray_scene(triangles),
    }
    probes = []
    for floor in floors:
        polygons = [Polygon(triangle[:, [0, 2]]) for triangle in floor.vertices[floor.faces]]
        polygons = [polygon for polygon in polygons if polygon.area > 1e-9]
        if polygons:
            point = unary_union(polygons).representative_point()
            y = float(np.median(floor.vertices[:, 1]))
            probes.extend(([point.x, y + 1.0, point.y], [point.x, y - 0.1, point.y]))
    probe_distances = _signed_outside_distance(np.asarray(probes, dtype=np.float32), surface)
    if np.any(probe_distances[0::2] > OUTSIDE_EPSILON_M) or np.any(probe_distances[1::2] <= DEPTH_LIMIT_M):
        raise AssertionError(f"{scan.info.scan_id} regularized exterior face orientation check failed: {probe_distances.tolist()}")
    return surface


def _closest(scene, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = np.empty(len(points), dtype=np.float32)
    primitive_ids = np.empty(len(points), dtype=np.int64)
    for start in range(0, len(points), CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, len(points))
        result = scene.compute_closest_points(o3d.core.Tensor(np.asarray(points[start:stop], dtype=np.float32)))
        closest = result["points"].numpy()
        ids = result["primitive_ids"].numpy().astype(np.int64)
        distances[start:stop] = np.linalg.norm(points[start:stop] - closest, axis=1)
        primitive_ids[start:stop] = ids
    return distances, primitive_ids


def _signed_outside_distance(points: np.ndarray, surface: dict[str, Any]) -> np.ndarray:
    result = np.empty(len(points), dtype=np.float32)
    for start in range(0, len(points), CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, len(points))
        chunk = np.asarray(points[start:stop], dtype=np.float32)
        closest = surface["scene"].compute_closest_points(o3d.core.Tensor(chunk))
        nearest = closest["primitive_ids"].numpy().astype(np.int64)
        axis = surface["axes"][nearest]
        sign = surface["signs"][nearest]
        result[start:stop] = (chunk[np.arange(len(chunk)), axis] - closest["points"].numpy()[np.arange(len(chunk)), axis]) * sign
    return result


def _connected_regions(faces: np.ndarray, selected: np.ndarray, adjacency: np.ndarray) -> list[np.ndarray]:
    ids = np.flatnonzero(selected)
    if not len(ids):
        return []
    linked = adjacency[selected[adjacency[:, 0]] & selected[adjacency[:, 1]]] if len(adjacency) else np.empty((0, 2), dtype=np.int64)
    local = np.full(len(faces), -1, dtype=np.int64)
    local[ids] = np.arange(len(ids))
    if len(linked):
        edges = local[linked]
        graph = coo_matrix(
            (np.ones(2 * len(edges), dtype=np.uint8),
             (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
            shape=(len(ids), len(ids)),
        ).tocsr()
    else:
        graph = coo_matrix((len(ids), len(ids))).tocsr()
    _, labels = connected_components(graph, directed=False)
    return [ids[labels == label] for label in range(labels.max() + 1)]


def _attachment_lengths(
    vertices: np.ndarray,
    faces: np.ndarray,
    selected: np.ndarray,
    adjacency: np.ndarray,
    edges: np.ndarray,
) -> tuple[float, float]:
    if not len(adjacency):
        return 0.0, 0.0
    boundary = selected[adjacency[:, 0]] != selected[adjacency[:, 1]]
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)[boundary]
    raw_length = float(lengths.sum())
    if not np.any(boundary):
        return raw_length, raw_length

    selected_faces = np.where(selected[adjacency[:, 0]], adjacency[:, 0], adjacency[:, 1])[boundary]
    triangles = vertices[faces[selected_faces]]
    edge_squared = np.stack((
        np.sum((triangles[:, 1] - triangles[:, 0]) ** 2, axis=1),
        np.sum((triangles[:, 2] - triangles[:, 1]) ** 2, axis=1),
        np.sum((triangles[:, 0] - triangles[:, 2]) ** 2, axis=1),
    ), axis=1)
    twice_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    aspect_ratio = np.divide(
        edge_squared.max(axis=1), twice_area,
        out=np.full(len(twice_area), np.inf, dtype=np.float64), where=twice_area > 1e-12,
    )
    weights = np.where(aspect_ratio > ELONGATED_TRIANGLE_ASPECT_RATIO,
                       ELONGATED_BOUNDARY_WEIGHT, 1.0)
    return raw_length, float(np.dot(lengths, weights))


def _region_surface(triangles: np.ndarray) -> dict[str, Any]:
    triangles = np.asarray(triangles, dtype=np.float32)
    vertices = triangles.reshape(-1, 3)
    unique = np.unique(vertices, axis=0)
    samples = np.concatenate((unique, triangles.mean(axis=1),
                              (triangles[:, 0] + triangles[:, 1]) * 0.5,
                              (triangles[:, 1] + triangles[:, 2]) * 0.5,
                              (triangles[:, 2] + triangles[:, 0]) * 0.5))
    return {
        "scene": _make_ray_scene(triangles),
        "samples": samples.astype(np.float32),
        "minimum": vertices.min(axis=0),
        "maximum": vertices.max(axis=0),
    }


def _visible_groups(regions: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    parent = list(range(len(regions)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i, first in enumerate(regions):
        for j in range(i + 1, len(regions)):
            second = regions[j]
            if first["scan_id"] != second["scan_id"]:
                continue
            gap_vector = np.maximum(0, np.maximum(first["surface"]["minimum"] - second["surface"]["maximum"],
                                                   second["surface"]["minimum"] - first["surface"]["maximum"]))
            if np.linalg.norm(gap_vector) > PATCH_GAP_M:
                continue
            b_to_a = _closest(first["surface"]["scene"], second["surface"]["samples"])[0]
            a_to_b = _closest(second["surface"]["scene"], first["surface"]["samples"])[0]
            if min(float(b_to_a.min()), float(a_to_b.min())) <= PATCH_GAP_M:
                first_root, second_root = root(i), root(j)
                parent[second_root] = first_root

    groups: dict[int, list[dict[str, Any]]] = {}
    for index, region in enumerate(regions):
        groups.setdefault(root(index), []).append(region)
    return list(groups.values())


def _plane_key(triangles: np.ndarray) -> list[tuple[tuple[float, ...], float]]:
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-15)
    axes = np.abs(normals).argmax(axis=1)
    flip = normals[np.arange(len(normals)), axes] < 0
    normals[flip] *= -1
    offsets = np.einsum("ij,ij->i", normals, triangles.mean(axis=1))
    return [(tuple(np.round(normal, 2).tolist()), round(float(offset), 2)) for normal, offset in zip(normals, offsets)]


def _rectangle_similarity(group: list[dict[str, Any]], surface: dict[str, Any]) -> float:
    triangles = np.concatenate([region["triangles"] for region in group])
    areas = np.concatenate([region["triangle_areas"] for region in group])
    _, nearest = _closest(surface["scene"], triangles.mean(axis=1))
    if "plane_keys" not in surface:
        surface["plane_keys"] = _plane_key(surface["triangles"])
    keys = surface["plane_keys"]
    votes: dict[tuple[tuple[float, ...], float], float] = {}
    for face_id, area in zip(nearest, areas):
        key = keys[face_id]
        votes[key] = votes.get(key, 0.0) + float(area)
    dominant = sorted(votes, key=lambda key: (-votes[key], key))[0]
    selected = np.fromiter((key == dominant for key in keys), dtype=bool, count=len(keys))
    plane_triangles = surface["triangles"][selected]
    normals = np.cross(plane_triangles[:, 1] - plane_triangles[:, 0], plane_triangles[:, 2] - plane_triangles[:, 0])
    plane_areas = np.linalg.norm(normals, axis=1) * 0.5
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-15)
    normal_axes = np.abs(normals).argmax(axis=1)
    flip = normals[np.arange(len(normals)), normal_axes] < 0
    normals[flip] *= -1
    normal = np.average(normals, axis=0, weights=np.maximum(plane_areas, 1e-12))
    normal /= np.linalg.norm(normal)
    axis = int(np.argmax(np.abs(normal)))
    if normal[axis] < 0:
        normal *= -1
    offset = float(np.average(plane_triangles.mean(axis=1) @ normal, weights=np.maximum(plane_areas, 1e-12)))
    reference = np.array([0.0, 1.0, 0.0]) if abs(normal[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(reference, normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    origin = normal * offset
    polygons = []
    for triangle in triangles:
        coordinates = (triangle - origin) @ np.column_stack((u, v))
        polygon = Polygon(coordinates)
        if polygon.is_valid and polygon.area > 1e-12:
            polygons.append(polygon)
    if not polygons:
        return 0.0
    projected = unary_union(polygons)
    if not projected.is_valid:
        projected = projected.buffer(0)
    rectangle_area = float(projected.minimum_rotated_rectangle.area)
    return min(1.0, float(projected.area) / rectangle_area) if rectangle_area > 0 else 0.0


def _split_red_vertices(vertices: np.ndarray, faces: np.ndarray, red_faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = ("red", "green", "blue")
    if not all(name in vertices.dtype.names for name in rgb):
        raise ValueError("Aligned object PLY must have red, green and blue vertex properties")
    red_ids = np.unique(faces[red_faces].reshape(-1))
    other_ids = np.unique(faces[~red_faces].reshape(-1))
    red_used = np.zeros(len(vertices), dtype=bool)
    other_used = np.zeros(len(vertices), dtype=bool)
    red_used[red_ids] = True
    other_used[other_ids] = True
    mixed = red_used & other_used
    output_vertices = vertices.copy()
    for channel, value in zip(rgb, (255, 0, 0)):
        output_vertices[channel][red_used & ~other_used] = value
    duplicates = vertices[mixed].copy()
    for channel, value in zip(rgb, (255, 0, 0)):
        duplicates[channel] = value
    duplicate_map = np.full(len(vertices), -1, dtype=np.int64)
    duplicate_map[mixed] = len(vertices) + np.arange(int(mixed.sum()), dtype=np.int64)
    output_faces = faces.copy()
    selected = output_faces[red_faces]
    use_duplicate = mixed[selected]
    selected[use_duplicate] = duplicate_map[selected[use_duplicate]]
    output_faces[red_faces] = selected
    return np.concatenate((output_vertices, duplicates)), output_faces


def _write_marked_ply(source: plyfile.PlyData, vertices: np.ndarray, faces: np.ndarray,
                      red_faces: np.ndarray, output: Path) -> Path:
    source_face_element = source["face"]
    properties = source_face_element.properties
    if FACE_MARKER in source_face_element.data.dtype.names:
        raise ValueError(f"Input PLY already contains the reserved face property {FACE_MARKER}")
    dtype = list(source_face_element.data.dtype.descr) + [(FACE_MARKER, "u1")]
    face_data = np.empty(len(faces), dtype=dtype)
    list_lengths, list_values = {}, {}
    for prop in properties:
        name = prop.name
        if isinstance(prop, plyfile.PlyListProperty):
            list_lengths[name] = prop.len_dtype
            list_values[name] = prop.val_dtype
            if name == "vertex_indices":
                face_data[name] = [face.astype(np.int32, copy=False) for face in faces]
            else:
                face_data[name] = source_face_element.data[name]
        else:
            face_data[name] = source_face_element.data[name]
    if "vertex_indices" not in source_face_element.data.dtype.names:
        raise ValueError("Aligned object PLY faces have no vertex_indices property")
    face_data[FACE_MARKER] = red_faces.astype(np.uint8)
    list_lengths.setdefault("vertex_indices", "u1")
    list_values.setdefault("vertex_indices", "i4")
    face_element = plyfile.PlyElement.describe(
        face_data, "face", len_types=list_lengths, val_types=list_values,
        comments=list(source_face_element.comments) + [f"{FACE_MARKER}=1 marks retained Phase 07 red outlier triangles"],
    )
    vertex_element = plyfile.PlyElement.describe(vertices, "vertex", comments=source["vertex"].comments)
    result = plyfile.PlyData([vertex_element, face_element], text=source.text, byte_order=source.byte_order,
                             comments=source.comments, obj_info=source.obj_info)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".ply", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        result.write(str(temp_path))
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def _signed_outside_distances_by_object(
    point_sets: list[np.ndarray], surface: dict[str, Any]
) -> list[np.ndarray]:
    lengths = [len(points) for points in point_sets]
    distances = _signed_outside_distance(np.concatenate(point_sets, axis=0), surface)
    return list(np.split(distances, np.cumsum(lengths)[:-1]))


def create_final_red_objects(scans: dict[str, Any], transforms: dict[str, np.ndarray],
                             output_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Apply the locked two-cutoff rule and write the single annotated object PLY."""
    timings: dict[str, float] = {}
    source_path = output_dir / "aligned_objects.ply"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing final aligned object mesh: {source_path}")
    stage_started = time.perf_counter()
    source = plyfile.PlyData.read(str(source_path))
    source_vertices = source["vertex"].data
    source_faces = np.stack([np.asarray(face, dtype=np.int64) for face in source["face"].data["vertex_indices"]])
    source_xyz = np.column_stack([source_vertices[name] for name in ("x", "y", "z")])
    selected_faces = np.zeros(len(source_faces), dtype=bool)
    offset = 0
    vertex_offset = 0
    selected_regions = []
    visible_group_count = 0
    surfaces = {}
    timings["load_combined_objects"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    for scan_id, scan in sorted(scans.items()):
        object_paths = sorted((output_dir / "objects" / scan_id).rglob("*.ply"))
        if not object_paths:
            continue
        transform = np.asarray(transforms[scan_id], dtype=np.float64)
        surface = _build_exterior_surface(scan, transform)
        surfaces[scan_id] = surface
        prepared_objects = []
        for path in object_paths:
            obj = plyfile.PlyData.read(str(path))
            vertices = obj["vertex"].data
            xyz = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(np.float32)
            faces = np.stack([np.asarray(face, dtype=np.int64) for face in obj["face"].data["vertex_indices"]])
            stop = offset + len(faces)
            # The synchronized combined PLY is exported in this exact scan/path order.
            if not np.array_equal(source_xyz[vertex_offset:vertex_offset + len(vertices)], xyz):
                raise ValueError(f"Object PLY order/vertices differ from {source_path}: {path}")
            expected_faces = faces + vertex_offset
            if not np.array_equal(source_faces[offset:stop], expected_faces):
                raise ValueError(f"Object PLY face order differs from {source_path}: {path}")

            prepared_objects.append((path, xyz, faces, offset))
            offset = stop
            vertex_offset += len(vertices)

        distances_by_object = _signed_outside_distances_by_object(
            [item[1] for item in prepared_objects], surface
        )
        for (path, xyz, faces, face_offset), distances in zip(prepared_objects, distances_by_object):
            outside_vertices = distances > OUTSIDE_EPSILON_M
            triangles = xyz[faces]
            triangle_outside = outside_vertices[faces].any(axis=1)
            triangle_areas = 0.5 * np.linalg.norm(
                np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
            )
            mesh = trimesh.Trimesh(vertices=xyz, faces=faces, process=False)
            adjacency = mesh.face_adjacency
            adjacency_edges = mesh.face_adjacency_edges
            for component in _connected_regions(faces, triangle_outside, adjacency):
                area = float(triangle_areas[component].sum())
                depth = float(distances[faces[component]].max())
                if area <= AREA_LIMIT_M2 or depth <= DEPTH_LIMIT_M:
                    continue
                component_mask = np.zeros(len(faces), dtype=bool)
                component_mask[component] = True
                cut, effective_cut = _attachment_lengths(
                    xyz, faces, component_mask, adjacency, adjacency_edges
                )
                region_triangles = triangles[component]
                selected_regions.append({
                    "scan_id": scan_id,
                    "source": path.relative_to(output_dir / "objects" / scan_id).as_posix(),
                    "face_ids": face_offset + component,
                    "area": area,
                    "cut_length": cut,
                    "effective_cut_length": effective_cut,
                    "triangles": region_triangles,
                    "triangle_areas": triangle_areas[component],
                    "surface": _region_surface(region_triangles),
                })
    timings["scan_surfaces_distances_and_patch_selection"] = time.perf_counter() - stage_started

    if offset != len(source_faces) or vertex_offset != len(source_vertices):
        raise ValueError(f"Processed {offset}/{len(source_faces)} faces and {vertex_offset}/{len(source_vertices)} vertices")

    stage_started = time.perf_counter()
    groups = _visible_groups(selected_regions)
    visible_group_count = len(groups)
    group_scores = []
    for group_index, group in enumerate(groups, 1):
        area = float(sum(region["area"] for region in group))
        raw_cut = float(sum(region["cut_length"] for region in group))
        effective_cut = float(sum(region["effective_cut_length"] for region in group))
        raw_seam = float(raw_cut / np.sqrt(area)) if area > 0 else float("inf")
        seam = float(effective_cut / np.sqrt(area)) if area > 0 else float("inf")
        similarity = float(_rectangle_similarity(group, surfaces[group[0]["scan_id"]]))
        seam_pass = bool(seam < SEAM_LIMIT)
        similarity_pass = bool(similarity > RECTANGLE_LIMIT)
        keep = seam_pass or similarity_pass
        group_scores.append({
            "group_id": f"G{group_index:02d}",
            "scan_id": group[0]["scan_id"],
            "members": [region["source"] for region in group],
            "member_count": len(group),
            "area_m2": area,
            "raw_attachment_boundary_m": raw_cut,
            "effective_attachment_boundary_m": effective_cut,
            "raw_seam_score": raw_seam,
            "effective_seam_score": seam,
            "seam_pass": seam_pass,
            "rectangle_similarity": similarity,
            "similarity_pass": similarity_pass,
            "outcome": (
                "PASS (both)" if seam_pass and similarity_pass else
                "PASS (seam)" if seam_pass else
                "PASS (similarity)" if similarity_pass else "DISCARDED"
            ),
            "member_details": [
                {
                    "source": region["source"],
                    "area_m2": region["area"],
                    "raw_attachment_boundary_m": region["cut_length"],
                    "effective_attachment_boundary_m": region["effective_cut_length"],
                }
                for region in group
            ],
        })
        if keep:
            for region in group:
                selected_faces[region["face_ids"]] = True
    score_path = output_dir / "stats" / "phase07_outlier_group_scores.json"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    staged_score_path = score_path.with_suffix(".json.tmp")
    staged_score_path.write_text(json.dumps({
        "schema_version": 1,
        "criteria": {
            "area_minimum_m2_exclusive": AREA_LIMIT_M2,
            "depth_minimum_m_exclusive": DEPTH_LIMIT_M,
            "patch_join_gap_m_maximum": PATCH_GAP_M,
            "seam_score_passes_below": SEAM_LIMIT,
            "elongated_triangle_aspect_ratio_above": ELONGATED_TRIANGLE_ASPECT_RATIO,
            "elongated_boundary_weight": ELONGATED_BOUNDARY_WEIGHT,
            "rectangle_similarity_passes_above": RECTANGLE_LIMIT,
        },
        "group_count": len(group_scores),
        "passed_count": sum(item["outcome"].startswith("PASS") for item in group_scores),
        "discarded_count": sum(item["outcome"] == "DISCARDED" for item in group_scores),
        "groups": group_scores,
    }, indent=2) + "\n")
    staged_score_path.replace(score_path)
    timings["group_and_filter_regions"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    colored_vertices, output_faces = _split_red_vertices(source_vertices, source_faces, selected_faces)
    output_path = output_dir / "aligned_objects_final_red.ply"
    staged_path = _write_marked_ply(source, colored_vertices, output_faces, selected_faces, output_path)

    # Read-back checks the face marker, exact geometry and per-triangle colors before publication.
    try:
        check = plyfile.PlyData.read(str(staged_path))
        check_faces = np.stack([np.asarray(face, dtype=np.int64) for face in check["face"].data["vertex_indices"]])
        check_vertices = check["vertex"].data
        if len(check_faces) != len(source_faces) or not np.array_equal(check["face"].data[FACE_MARKER], selected_faces.astype(np.uint8)):
            raise AssertionError("Final red PLY face marker does not match the selected triangles")
        check_xyz = np.column_stack([check_vertices[name] for name in ("x", "y", "z")])
        if not np.array_equal(source_xyz[source_faces], check_xyz[check_faces]):
            raise AssertionError("Final red PLY changed object triangle geometry")
        check_rgb = np.column_stack([check_vertices[name] for name in ("red", "green", "blue")])
        if not np.all(np.all(check_rgb[check_faces[selected_faces]] == [255, 0, 0], axis=2)):
            raise AssertionError("A marked outlier triangle is not solid red")
        source_rgb = np.column_stack([source_vertices[name] for name in ("red", "green", "blue")])
        if not np.array_equal(source_rgb[source_faces[~selected_faces]], check_rgb[check_faces[~selected_faces]]):
            raise AssertionError("A non-outlier object triangle lost its original color")
        staged_path.replace(output_path)
    finally:
        staged_path.unlink(missing_ok=True)
    timings["write_and_validate_red_mesh"] = time.perf_counter() - stage_started
    summary = {
        "first_cutoff_regions": len(selected_regions),
        "contiguous_groups": visible_group_count,
        "red_regions": sum(1 for group in groups if selected_faces[np.concatenate([r["face_ids"] for r in group])].any()),
        "red_triangles": int(selected_faces.sum()),
        "timings_seconds": timings,
    }
    return output_path, summary
