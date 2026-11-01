#!/usr/bin/env python3
"""Assemble Phase 06's finalized walls, floors, ceilings, and floor-ceiling slabs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import shapely
import trimesh
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon

STAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE_DIR))
sys.path.insert(0, str(STAGE_DIR.parent / "04_3DGS_to_mesh"))
from fill_mesh_holes import boundary_loops, export_ply_with_face_attributes, read_ply_face_flags
from mesh_elaboration_inputs import resolve_elaboration_output_dir
from structural_geometry import close_slab, weld_float32_boundary_edges

WALL_NAME = "wall_enclosing_planes_fitted_to_regularized_contours.ply"
CEILING_NAME = "ceiling_planes_fitted_to_support_contours_regularized.ply"
CEILING_MANIFEST_NAME = "ceiling_planes_fitted_to_support_contours_regularized_manifest.json"
OUTPUT_NAME = "all_single_elements_finalized.ply"
MANIFEST_NAME = "all_single_elements_finalized_manifest.json"
BASE_FLOOR_ALIGNMENT_TOLERANCE_M = 0.15


def _mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"Mesh is empty or invalid: {path}")
    if not np.isfinite(mesh.vertices).all() or mesh.faces.min() < 0 or mesh.faces.max() >= len(mesh.vertices):
        raise ValueError(f"Mesh has invalid geometry: {path}")
    colors = np.asarray(mesh.visual.vertex_colors)
    if len(colors) != len(mesh.vertices):
        mesh.visual.vertex_colors = np.tile(np.array([166, 166, 166, 255], dtype=np.uint8), (len(mesh.vertices), 1))
    return mesh


def _face_property(path: Path, name: str, count: int, default: int | None = None) -> np.ndarray:
    values = read_ply_face_flags(path, name)
    if values is None:
        if default is None:
            raise ValueError(f"{path} is missing required face property {name!r}")
        return np.full(count, default, dtype=np.int32)
    values = np.asarray(values)
    if len(values) != count:
        raise ValueError(f"{path} has a mismatched {name!r} face property")
    return values


def _ceiling_patch_max_y(ceiling: trimesh.Trimesh, groups: np.ndarray,
                         associations: np.ndarray, areas: np.ndarray,
                         junctions: np.ndarray, patch: dict) -> float:
    area_id = -1 if patch.get("support_area_id") is None else int(patch["support_area_id"])
    association = (-1 if patch.get("associated_floor_level") is None
                   else int(patch["associated_floor_level"]))
    patch_mask = (
        (groups == int(patch["group_id"]))
        & (associations == association)
        & (areas == area_id)
        & (junctions == 0)
    )
    patch_faces = ceiling.faces[patch_mask]
    if not len(patch_faces):
        raise ValueError(f"No finalized ceiling faces found for upper ceiling patch {patch}")
    return float(ceiling.vertices[patch_faces, 1].max())


def _floor_area_supports(levels_dir: Path, level: dict, floor_path: Path,
                         floor: trimesh.Trimesh) -> list[dict]:
    """Return the lowest level's fitted floor planes and their projected supports."""
    level_id = int(level["level"])
    face_areas = _face_property(floor_path, "floor_area_id", len(floor.faces))
    polygons_path = levels_dir / f"floor_level_{level_id:03d}_polygons.json"
    polygon_areas = {}
    if polygons_path.is_file():
        polygon_areas = {
            int(area["area_id"]): area
            for area in json.loads(polygons_path.read_text(encoding="utf-8")).get("areas", [])
        }

    supports = []
    for area in level.get("floor_areas", []):
        area_id = int(area["area_id"])
        if not np.any(face_areas == area_id):
            continue
        equation = np.asarray(area.get("plane_equation"), dtype=np.float64)
        if equation.shape != (4,) or not np.isfinite(equation).all():
            raise ValueError(f"Invalid fitted floor plane for level {level_id}, area {area_id}")
        length = float(np.linalg.norm(equation[:3]))
        if length <= 1e-10:
            raise ValueError(f"Invalid fitted floor plane for level {level_id}, area {area_id}")
        equation /= length
        if equation[1] < 0:
            equation = -equation
        if equation[1] < 0.7:
            raise ValueError(f"Fitted floor plane is too vertical for level {level_id}, area {area_id}")

        record = polygon_areas.get(area_id, {})
        rings = record.get("floor_polygon_rings_world_xz_m", [])
        support = None
        for ring in rings:
            polygon = Polygon(np.asarray(ring, dtype=np.float64))
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if not polygon.is_empty and shapely.area(polygon) > 1e-8:
                support = polygon if support is None else support.symmetric_difference(polygon)
        if support is None:
            contour = np.asarray(record.get("regularized_wall_contour_world_xz_m", []), dtype=np.float64)
            if contour.ndim == 2 and contour.shape[0] >= 3 and contour.shape[1] == 2:
                support = Polygon(contour)
        if support is None or support.is_empty:
            continue
        if not support.is_valid:
            support = support.buffer(0)
        if support.is_empty:
            continue
        supports.append({
            "area_id": area_id,
            "equation": equation,
            "polygon": support.buffer(BASE_FLOOR_ALIGNMENT_TOLERANCE_M),
        })
    return supports


def _wall_face_floor_planes(wall: trimesh.Trimesh, supports: list[dict],
                            fallback_y: float) -> tuple[np.ndarray, dict]:
    """Assign a floor plane only when it supports the wall face's full X/Z projection."""
    if not np.isfinite(fallback_y):
        raise ValueError("The fallback floor elevation must be finite")
    fallback = np.array([0.0, 1.0, 0.0, -fallback_y], dtype=np.float64)
    triangles = np.asarray(wall.vertices, dtype=np.float64)[wall.faces]
    centers = triangles.mean(axis=1)
    equations = np.tile(fallback, (len(triangles), 1))
    selected = np.full(len(triangles), -1, dtype=np.int32)
    for face_id, center in enumerate(centers):
        projected_face = Polygon(triangles[face_id][:, [0, 2]]).convex_hull
        candidates = [support for support in supports
                      if support["polygon"].covers(projected_face)]
        if not candidates:
            continue
        candidates.sort(key=lambda support: (
            -float(support["equation"][0] * center[0]
                   + support["equation"][2] * center[2]
                   + support["equation"][3]) / float(support["equation"][1]),
            int(support["area_id"]),
        ))
        equations[face_id] = candidates[0]["equation"]
        selected[face_id] = int(candidates[0]["area_id"])
    return equations, {
        "area_ids_used": sorted(int(value) for value in np.unique(selected) if value >= 0),
        "wall_face_count": int(len(triangles)),
        "wall_face_count_using_fitted_floor_area": int(np.count_nonzero(selected >= 0)),
        "wall_face_count_using_lowest_y_fallback": int(np.count_nonzero(selected < 0)),
    }


def _floor_height_at_xz(x: float, z: float, supports: list[dict], fallback_y: float) -> tuple[float, bool]:
    point = Point(float(x), float(z))
    candidates = [support for support in supports if support["polygon"].covers(point)]
    if not candidates:
        return float(fallback_y), False
    heights = [-(support["equation"][0] * x + support["equation"][2] * z
                 + support["equation"][3]) / support["equation"][1]
               for support in candidates]
    return float(min(heights)), True


def _subset(mesh: trimesh.Trimesh, mask: np.ndarray) -> trimesh.Trimesh:
    if len(mask) != len(mesh.faces) or not np.any(mask):
        raise ValueError("Cannot build a mesh subset from an empty or mismatched face selection")
    result = mesh.copy()
    result.update_faces(mask)
    result.remove_unreferenced_vertices()
    result.merge_vertices(digits_vertex=7)
    return result


def _rgb(mesh: trimesh.Trimesh) -> np.ndarray:
    colors = np.asarray(mesh.visual.vertex_colors)
    if colors.ndim != 2 or colors.shape[1] < 3 or len(colors) != len(mesh.vertices):
        return np.tile(np.array([166, 166, 166], dtype=np.uint8), (len(mesh.vertices), 1))
    return colors[:, :3].astype(np.uint8, copy=False)


def _clip_wall_to_floor_and_ceiling(wall: trimesh.Trimesh, lower_y: float | np.ndarray,
                                    ceiling_plane: np.ndarray) -> tuple[trimesh.Trimesh, dict]:
    """Trim walls below their assigned floor plane and above the fitted ceiling plane."""
    ceiling_plane = np.asarray(ceiling_plane, dtype=np.float64).reshape(4)
    normal_length = float(np.linalg.norm(ceiling_plane[:3]))
    if (not np.isfinite(ceiling_plane).all()
            or normal_length <= 1e-10 or abs(ceiling_plane[1]) <= 1e-8):
        raise ValueError("Invalid wall floor/ceiling clipping planes")
    ceiling_plane = ceiling_plane / normal_length
    if ceiling_plane[1] < 0:
        ceiling_plane = -ceiling_plane

    lower = np.asarray(lower_y, dtype=np.float64)
    if lower.ndim == 0:
        if not np.isfinite(lower):
            raise ValueError("Invalid wall floor/ceiling clipping planes")
        lower_planes = np.tile([0.0, 1.0, 0.0, -float(lower)], (len(wall.faces), 1))
    elif lower.shape == (4,):
        lower_planes = np.tile(lower, (len(wall.faces), 1))
    elif lower.shape == (len(wall.faces), 4):
        lower_planes = lower.copy()
    else:
        raise ValueError("Wall floor planes must be one equation or one equation per face")
    if not np.isfinite(lower_planes).all():
        raise ValueError("Invalid wall floor/ceiling clipping planes")
    lower_lengths = np.linalg.norm(lower_planes[:, :3], axis=1)
    if np.any(lower_lengths <= 1e-10):
        raise ValueError("Invalid wall floor/ceiling clipping planes")
    lower_planes /= lower_lengths[:, None]
    lower_planes[lower_planes[:, 1] < 0] *= -1
    if np.any(lower_planes[:, 1] < 0.7):
        raise ValueError("Wall lower boundary must use upward-facing floor planes")

    vertices = np.asarray(wall.vertices, dtype=np.float64)
    faces = np.asarray(wall.faces, dtype=np.int64)
    triangles = vertices[faces]
    lower_values = (np.einsum("fvc,fc->fv", triangles, lower_planes[:, :3])
                    + lower_planes[:, None, 3])
    ceiling_values = triangles @ ceiling_plane[:3] + ceiling_plane[3]
    inside = (lower_values.min(axis=1) >= -1e-9) & (ceiling_values.max(axis=1) <= 1e-9)
    outside = (lower_values.max(axis=1) < -1e-9) | (ceiling_values.min(axis=1) > 1e-9)
    crossing_ids = np.flatnonzero(~inside & ~outside)
    source_colors = np.asarray(wall.visual.vertex_colors, dtype=np.uint8)
    vertex_chunks = [vertices]
    color_chunks = [source_colors]
    face_chunks = [faces[inside]]
    next_vertex = len(vertices)
    generated_faces = 0

    for face_id in crossing_ids:
        polygon = [(point.copy(), color.astype(np.float64)) for point, color in
                   zip(triangles[face_id], source_colors[faces[face_id]])]
        for equation, keep_below in ((lower_planes[face_id], False), (ceiling_plane, True)):
            clipped = []
            for index, (point, color) in enumerate(polygon):
                next_point, next_color = polygon[(index + 1) % len(polygon)]
                point_value = float(point @ equation[:3] + equation[3])
                next_value = float(next_point @ equation[:3] + equation[3])
                point_inside = point_value <= 1e-9 if keep_below else point_value >= -1e-9
                next_inside = next_value <= 1e-9 if keep_below else next_value >= -1e-9
                if point_inside:
                    clipped.append((point, color))
                if point_inside != next_inside:
                    fraction = -point_value / (next_value - point_value)
                    clipped.append((point + fraction * (next_point - point),
                                    color + fraction * (next_color - color)))
            deduplicated = []
            for point, color in clipped:
                if not deduplicated or np.linalg.norm(point - deduplicated[-1][0]) > 1e-10:
                    deduplicated.append((point, color))
            if (len(deduplicated) > 1 and
                    np.linalg.norm(deduplicated[0][0] - deduplicated[-1][0]) <= 1e-10):
                deduplicated.pop()
            polygon = deduplicated
            if not polygon:
                break
        if len(polygon) < 3:
            continue
        points = np.asarray([point for point, _ in polygon], dtype=np.float64)
        colors = np.rint([color for _, color in polygon]).clip(0, 255).astype(np.uint8)
        for index in range(1, len(points) - 1):
            triangle = points[[0, index, index + 1]]
            if np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])) <= 1e-12:
                continue
            vertex_chunks.append(triangle)
            color_chunks.append(colors[[0, index, index + 1]])
            face_chunks.append(np.array([[next_vertex, next_vertex + 1, next_vertex + 2]], dtype=np.int64))
            next_vertex += 3
            generated_faces += 1

    output_faces = np.concatenate(face_chunks)
    if not len(output_faces):
        raise ValueError("Vertical floor/ceiling clipping removed every wall triangle")
    output_vertices = np.concatenate(vertex_chunks).astype(np.float32)
    retained = trimesh.Trimesh(vertices=output_vertices, faces=output_faces, process=False)
    retained.visual.vertex_colors = np.concatenate(color_chunks)
    retained.remove_unreferenced_vertices()
    xz_min, xz_max = vertices[:, [0, 2]].min(axis=0), vertices[:, [0, 2]].max(axis=0)
    corners = np.array([[xz_min[0], xz_min[1]], [xz_min[0], xz_max[1]],
                        [xz_max[0], xz_min[1]], [xz_max[0], xz_max[1]]])
    ceiling_heights = -(ceiling_plane[0] * corners[:, 0]
                        + ceiling_plane[2] * corners[:, 1] + ceiling_plane[3]) / ceiling_plane[1]
    floor_heights = -(lower_planes[:, None, 0] * triangles[:, :, 0]
                      + lower_planes[:, None, 2] * triangles[:, :, 2]
                      + lower_planes[:, None, 3]) / lower_planes[:, None, 1]
    return retained, {
        "source_y_range_m": [float(vertices[:, 1].min()), float(vertices[:, 1].max())],
        "retained_y_range_m": [float(retained.vertices[:, 1].min()), float(retained.vertices[:, 1].max())],
        "lowest_floor_y_m": float(floor_heights.min()),
        "floor_boundary_plane_count": int(len(np.unique(np.round(lower_planes, decimals=8), axis=0))),
        "ceiling_plane_equation": ceiling_plane.tolist(),
        "ceiling_height_range_over_wall_bounds_m": [float(ceiling_heights.min()), float(ceiling_heights.max())],
        "highest_ceiling_y_m": float(ceiling_heights.max()),
        "applied": bool(np.any(outside) or len(crossing_ids)),
        "removed_source_triangle_count": int(np.count_nonzero(outside)),
        "clipped_source_triangle_count": int(len(crossing_ids)),
        "generated_triangle_count": int(generated_faces),
    }


def _extend_wall_top_to_ceiling(wall: trimesh.Trimesh, lower_y: float,
                                ceiling_plane: np.ndarray,
                                floor_supports: list[dict] | None = None) -> tuple[trimesh.Trimesh, dict]:
    """Extend the upper wall perimeter vertically wherever it falls below the roof."""
    ceiling_plane = np.asarray(ceiling_plane, dtype=np.float64).reshape(4)
    normal_length = float(np.linalg.norm(ceiling_plane[:3]))
    if (not np.isfinite(lower_y) or not np.isfinite(ceiling_plane).all()
            or normal_length <= 1e-10 or abs(ceiling_plane[1]) <= 1e-8):
        raise ValueError("Invalid wall ceiling extension plane")
    ceiling_plane /= normal_length
    if ceiling_plane[1] < 0:
        ceiling_plane = -ceiling_plane
    if abs(ceiling_plane[1]) <= 1e-8:
        raise ValueError("The fitted ceiling plane must have a vertical component to extend walls")

    source_vertices = np.asarray(wall.vertices, dtype=np.float64)
    source_faces = np.asarray(wall.faces, dtype=np.int64)
    source_colors = _rgb(wall)
    topology = _subset(wall, np.ones(len(wall.faces), dtype=bool))
    vertices = np.asarray(topology.vertices, dtype=np.float64)
    faces = np.asarray(topology.faces, dtype=np.int64)
    loops = boundary_loops(faces, vertices)
    perimeter_loops = []
    for loop in loops:
        points = vertices[loop]
        xz = points[:, [0, 2]]
        area = abs(float(np.sum(xz[:, 0] * np.roll(xz[:, 1], -1)
                                - np.roll(xz[:, 0], -1) * xz[:, 1]) * 0.5))
        perimeter_loops.append((loop, area, float(points[:, 1].mean())))

    # Window/door loops collapse in XZ; tiny projected loops are disconnected scraps.
    max_area = max((area for _, area, _ in perimeter_loops), default=0.0)
    footprint_loops = [(loop, area, mean_y) for loop, area, mean_y in perimeter_loops
                       if area > max(0.02, max_area * 0.001)]
    if not footprint_loops:
        return wall, {"extended_boundary_loop_count": 0,
                      "extended_boundary_edge_count": 0, "added_triangle_count": 0,
                      "maximum_extension_m": 0.0}

    floor_supports = [] if floor_supports is None else floor_supports

    def lies_on_floor(loop: np.ndarray) -> bool:
        points = vertices[loop]
        samples = [_floor_height_at_xz(point[0], point[2], floor_supports, lower_y)
                   for point in points]
        if all(covered for _, covered in samples):
            return max(abs(point[1] - height)
                       for point, (height, _) in zip(points, samples)) <= BASE_FLOOR_ALIGNMENT_TOLERANCE_M
        return abs(float(points[:, 1].mean()) - lower_y) <= 1e-4

    top_loops = [loop for loop, _, mean_y in footprint_loops
                 if mean_y > lower_y + 1e-4 and not lies_on_floor(loop)]
    if not top_loops:
        return wall, {"extended_boundary_loop_count": 0,
                      "extended_boundary_edge_count": 0, "added_triangle_count": 0,
                      "maximum_extension_m": 0.0}

    boundary_faces = {}
    for face_id, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = (min(int(a), int(b)), max(int(a), int(b)))
            boundary_faces.setdefault(key, []).append((int(a), int(b), face_id))

    added_vertices, added_colors, added_faces = [], [], []
    projected_vertices = {}
    maximum_extension = 0.0
    extended_edges = 0
    extended_loops = 0

    def project_to_ceiling(vertex_id: int) -> int:
        nonlocal maximum_extension
        if vertex_id in projected_vertices:
            return projected_vertices[vertex_id]
        point = source_vertices[vertex_id].copy()
        roof_y = -(ceiling_plane[0] * point[0] + ceiling_plane[2] * point[2]
                   + ceiling_plane[3]) / ceiling_plane[1]
        extension = roof_y - point[1]
        point[1] = roof_y
        if (extension <= 1e-7 or
                np.array_equal(point.astype(np.float32), source_vertices[vertex_id].astype(np.float32))):
            projected_vertices[vertex_id] = vertex_id
            return vertex_id
        maximum_extension = max(maximum_extension, float(extension))
        projected_vertices[vertex_id] = len(source_vertices) + len(added_vertices)
        added_vertices.append(point)
        added_colors.append(source_colors[vertex_id])
        return projected_vertices[vertex_id]

    for loop in top_loops:
        loop_extended = False
        for index, a_value in enumerate(loop):
            a, b = int(a_value), int(loop[(index + 1) % len(loop)])
            edge = boundary_faces.get((min(a, b), max(a, b)), [])
            if len(edge) != 1:
                continue
            topology_a, topology_b, face_id = edge[0]
            source_face = source_faces[face_id]
            source_face_points = source_vertices[source_face]
            face_a = int(source_face[np.argmin(np.linalg.norm(
                source_face_points - vertices[topology_a], axis=1))])
            face_b = int(source_face[np.argmin(np.linalg.norm(
                source_face_points - vertices[topology_b], axis=1))])
            upper_a, upper_b = project_to_ceiling(face_a), project_to_ceiling(face_b)
            if upper_a == face_a and upper_b == face_b:
                continue
            triangles = [(face_a, face_b, upper_b), (face_a, upper_b, upper_a)]
            reference_normal = topology.face_normals[face_id]
            faces_before_edge = len(added_faces)
            for triangle in triangles:
                if len(set(triangle)) < 3:
                    continue
                points = [source_vertices[v] if v < len(source_vertices)
                          else added_vertices[v - len(source_vertices)]
                          for v in triangle]
                points = np.asarray(points, dtype=np.float32).astype(np.float64)
                normal = np.cross(points[1] - points[0], points[2] - points[0])
                if np.linalg.norm(normal) <= 1e-12:
                    continue
                if np.dot(normal, reference_normal) < 0:
                    triangle = (triangle[0], triangle[2], triangle[1])
                added_faces.append(triangle)
            if len(added_faces) > faces_before_edge:
                extended_edges += 1
                loop_extended = True
        extended_loops += int(loop_extended)

    if not added_faces:
        return wall, {"extended_boundary_loop_count": 0,
                      "extended_boundary_edge_count": 0, "added_triangle_count": 0,
                      "maximum_extension_m": 0.0}
    result = trimesh.Trimesh(
        vertices=np.vstack((source_vertices, np.asarray(added_vertices))),
        faces=np.vstack((source_faces, np.asarray(added_faces, dtype=np.int64))),
        process=False,
    )
    result.visual.vertex_colors = np.vstack((source_colors, np.asarray(added_colors, dtype=np.uint8)))
    return result, {
        "extended_boundary_loop_count": extended_loops,
        "extended_boundary_edge_count": extended_edges,
        "added_triangle_count": len(added_faces),
        "maximum_extension_m": maximum_extension,
    }


def _wall_base_footprint(wall: trimesh.Trimesh) -> tuple[Polygon, float]:
    wall = _subset(wall, np.ones(len(wall.faces), dtype=bool))
    base_y = float(wall.vertices[:, 1].min())
    rings = []
    for loop in boundary_loops(wall.faces, wall.vertices):
        points = wall.vertices[loop]
        if np.max(np.abs(points[:, 1] - base_y)) <= 1e-5:
            ring = Polygon(points[:, [0, 2]])
            if not ring.is_valid:
                ring = ring.buffer(0)
            if not ring.is_empty and ring.area > 1e-8:
                rings.append(ring)
    if not rings:
        raise ValueError("The fitted walls have no horizontal boundary at their lowest height")
    footprint = rings[0]
    for ring in rings[1:]:
        footprint = footprint.symmetric_difference(ring)
    if footprint.is_empty or not footprint.is_valid:
        raise ValueError("Could not form a valid floor support from the wall base contour")
    return footprint, base_y


def _wall_footprint_at_y(wall: trimesh.Trimesh, y: float,
                         fallback: Polygon | None = None) -> Polygon:
    """Return the largest closed wall contour at a horizontal height."""
    base_y = float(wall.vertices[:, 1].min())
    if abs(float(y) - base_y) <= 1e-5:
        base_footprint, _ = _wall_base_footprint(wall)
        return base_footprint if fallback is None else base_footprint.intersection(fallback)

    triangles = wall.vertices[wall.faces]
    lines = []
    for triangle in triangles:
        distances = triangle[:, 1] - y
        if distances.min() > 1e-8 or distances.max() < -1e-8 or np.all(np.abs(distances) <= 1e-8):
            continue
        hits = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if abs(distances[a]) <= 1e-8:
                hits.append(triangle[a, [0, 2]])
            if distances[a] * distances[b] < -1e-16:
                fraction = -distances[a] / (distances[b] - distances[a])
                hits.append((triangle[a] + fraction * (triangle[b] - triangle[a]))[[0, 2]])
        points = np.unique(np.round(hits, decimals=6), axis=0) if hits else np.empty((0, 2))
        if len(points) >= 2:
            distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
            a, b = np.unravel_index(np.argmax(distances), distances.shape)
            if distances[a, b] > 1e-7:
                lines.append(LineString(points[[a, b]]))

    if lines:
        network = shapely.union_all(lines)
        contours = [part for part in shapely.get_parts(
            shapely.polygonize(list(shapely.get_parts(network))))
                    if part.geom_type == "Polygon" and part.area > 1e-8]
    else:
        contours = []
    if contours:
        contour = max(contours, key=lambda part: (
            part.area if fallback is None else shapely.area(part.intersection(fallback))))
        footprint = contour if fallback is None else contour.intersection(fallback)
    elif fallback is not None:
        footprint = fallback
    else:
        raise ValueError(f"The fitted walls have no closed horizontal contour at Y={y:.6f} m")
    if not footprint.is_valid:
        footprint = footprint.buffer(0)
    if footprint.is_empty or shapely.area(footprint) <= 1e-8:
        raise ValueError(f"The wall contour at Y={y:.6f} m has no overlap with its floor support")
    return footprint


def _wall_support_over_y_range(wall: trimesh.Trimesh, y_min: float, y_max: float,
                               fallback: Polygon | None = None) -> Polygon:
    """Constrain a surface to wall contours at both ends and every horizontal transition."""
    triangles = wall.vertices[wall.faces]
    horizontal = np.ptp(triangles[:, :, 1], axis=1) <= 1e-6
    transitions = np.unique(np.round(triangles[horizontal, 0, 1], decimals=6))
    heights = sorted({float(y_min), float(y_max), *(float(y) for y in transitions
                                                       if y_min < y < y_max)})
    supports = [_wall_footprint_at_y(wall, y, fallback) for y in heights]
    footprint = supports[0]
    for support in supports[1:]:
        footprint = footprint.intersection(support)
    if not footprint.is_valid:
        footprint = footprint.buffer(0)
    if footprint.is_empty or shapely.area(footprint) <= 1e-8:
        raise ValueError(f"The wall contours do not share a floor support between Y={y_min:.6f} and Y={y_max:.6f} m")
    return footprint


def _regularized_wall_contour(levels_dir: Path, level_id: int,
                              area_id: int) -> Polygon | None:
    path = levels_dir / f"floor_level_{level_id:03d}_polygons.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    area = next((item for item in data.get("areas", [])
                 if int(item.get("area_id", -1)) == area_id), None)
    if area is None:
        return None
    points = np.asarray(area.get("regularized_wall_contour_world_xz_m", []), dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        return None
    footprint = Polygon(points)
    if not footprint.is_valid:
        footprint = footprint.buffer(0)
    return footprint if not footprint.is_empty and shapely.area(footprint) > 1e-8 else None


def _wall_contour_footprint(levels_dir: Path, level_id: int) -> Polygon | shapely.GeometryCollection | None:
    path = levels_dir / f"floor_level_{level_id:03d}_polygons.json"
    if not path.is_file():
        return None
    polygons = []
    for area in json.loads(path.read_text(encoding="utf-8")).get("areas", []):
        points = np.asarray(area.get("regularized_wall_contour_world_xz_m", []), dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            continue
        polygon = Polygon(points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty and shapely.area(polygon) > 1e-8:
            polygons.append(polygon)
    if not polygons:
        return None
    footprint = shapely.union_all(polygons)
    return footprint if not footprint.is_empty else None


def _clip_surface_to_footprint(surface: trimesh.Trimesh, wall_footprint: Polygon,
                               target_y: float | None = None,
                               retain_degenerate_inside: bool = False,
                               allow_empty: bool = False) -> tuple[trimesh.Trimesh, dict]:
    """Translate optionally, then clip every triangle to its wall support footprint."""
    source_vertices = np.asarray(surface.vertices, dtype=np.float64).copy()
    shift = 0.0 if target_y is None else float(target_y - np.median(source_vertices[:, 1]))
    source_vertices[:, 1] += shift
    source_faces = np.asarray(surface.faces, dtype=np.int64)
    triangles = source_vertices[source_faces]
    triangle_xz = triangles[:, :, [0, 2]]
    rings = np.concatenate((triangle_xz, triangle_xz[:, :1]), axis=1)
    triangle_polygons = shapely.polygons(rings)
    fully_inside = shapely.covers(wall_footprint, triangle_polygons)
    if retain_degenerate_inside:
        projected_areas = shapely.area(triangle_polygons)
        for face_id in np.flatnonzero(projected_areas <= 1e-12):
            fully_inside[face_id] = all(
                wall_footprint.covers(Point(float(point[0]), float(point[1])))
                for point in triangle_xz[face_id]
            )
    partial_ids = np.flatnonzero(~fully_inside)
    clipped = shapely.intersection(wall_footprint, triangle_polygons[partial_ids])
    clipped_areas = shapely.area(clipped)
    keep_clipped = clipped_areas > 1e-12
    partial_ids = partial_ids[keep_clipped]
    clipped = clipped[keep_clipped]
    clipped_areas = clipped_areas[keep_clipped]
    triangulations = shapely.constrained_delaunay_triangles(clipped)

    source_colors = np.asarray(surface.visual.vertex_colors, dtype=np.uint8)
    vertex_chunks = [source_vertices]
    color_chunks = [source_colors]
    face_chunks = [source_faces[fully_inside]]
    next_vertex = len(source_vertices)
    clipped_face_count = discarded_clip_slivers = 0
    for face_id, pieces in zip(partial_ids, triangulations):
        source_triangle = triangles[face_id]
        source_xz = triangle_xz[face_id]
        source_rgba = source_colors[source_faces[face_id]].astype(np.float64)
        first_edge = source_xz[1] - source_xz[0]
        second_edge = source_xz[2] - source_xz[0]
        denominator = first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]
        if abs(denominator) <= 1e-15:
            continue
        for piece in pieces.geoms:
            points_xz = np.asarray(piece.exterior.coords[:3], dtype=np.float64)
            quantized = np.round(points_xz, decimals=6)
            first = quantized[1] - quantized[0]
            second = quantized[2] - quantized[0]
            area = abs(first[0] * second[1] - first[1] * second[0]) * 0.5
            if area <= 1e-12 or len(np.unique(quantized, axis=0)) != 3:
                discarded_clip_slivers += 1
                continue
            relative = points_xz - source_xz[0]
            first_weight = (relative[:, 0] * second_edge[1] - relative[:, 1] * second_edge[0]) / denominator
            second_weight = (first_edge[0] * relative[:, 1] - first_edge[1] * relative[:, 0]) / denominator
            weights = np.column_stack((1.0 - first_weight - second_weight,
                                       first_weight, second_weight))
            vertices = weights @ source_triangle
            colors = np.rint(weights @ source_rgba).clip(0, 255).astype(np.uint8)
            vertex_chunks.append(vertices)
            color_chunks.append(colors)
            face_chunks.append(np.arange(next_vertex, next_vertex + 3, dtype=np.int64)[None, :])
            next_vertex += 3
            clipped_face_count += 1

    faces = np.concatenate(face_chunks)
    if not len(faces) and not allow_empty:
        raise ValueError("Clipping the surface to the wall footprint removed every triangle")
    vertices = np.concatenate(vertex_chunks).astype(np.float32)
    if not len(faces):
        result = trimesh.Trimesh(vertices=np.empty((0, 3), dtype=np.float32),
                                 faces=np.empty((0, 3), dtype=np.int64), process=False)
        result.visual.vertex_colors = np.empty((0, 4), dtype=np.uint8)
        input_area = float(shapely.area(triangle_polygons).sum())
        return result, {
            "vertical_translation_m": shift,
            "support_elevation_y_m": (float(target_y) if target_y is not None
                                       else float(np.median(source_vertices[:, 1]))),
            "wall_footprint_area_m2": float(shapely.area(wall_footprint)),
            "input_projected_area_m2": input_area,
            "retained_projected_area_m2": 0.0,
            "removed_projected_area_m2": input_area,
            "input_triangle_count": int(len(source_faces)),
            "output_triangle_count": 0,
            "clipped_source_triangle_count": int(len(partial_ids)),
            "removed_source_triangle_count": int(len(source_faces) - int(fully_inside.sum()) - len(partial_ids)),
            "generated_clip_triangle_count": 0,
            "discarded_degenerate_clip_triangle_count": int(discarded_clip_slivers),
            "discarded_float32_degenerate_triangle_count": 0,
        }
    triangle_vertices = vertices[faces].astype(np.float64)
    areas = np.linalg.norm(np.cross(triangle_vertices[:, 1] - triangle_vertices[:, 0],
                                    triangle_vertices[:, 2] - triangle_vertices[:, 0]), axis=1)
    nondegenerate = areas > 0.0
    discarded_float32_degenerate_count = int(np.count_nonzero(~nondegenerate))
    faces = faces[nondegenerate]
    if not len(faces):
        raise ValueError("Clipping and PLY precision removed every surface triangle")
    result = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    result.visual.vertex_colors = np.concatenate(color_chunks)
    result.remove_unreferenced_vertices()
    input_area = float(shapely.area(triangle_polygons).sum())
    retained_area = float(shapely.area(triangle_polygons[fully_inside]).sum() + clipped_areas.sum())
    return result, {
        "vertical_translation_m": shift,
        "support_elevation_y_m": (float(target_y) if target_y is not None
                                   else float(np.median(source_vertices[:, 1]))),
        "wall_footprint_area_m2": float(shapely.area(wall_footprint)),
        "input_projected_area_m2": input_area,
        "retained_projected_area_m2": retained_area,
        "removed_projected_area_m2": max(0.0, input_area - retained_area),
        "input_triangle_count": int(len(source_faces)),
        "output_triangle_count": int(len(faces)),
        "clipped_source_triangle_count": int(len(partial_ids)),
        "removed_source_triangle_count": int(len(source_faces) - int(fully_inside.sum()) - len(partial_ids)),
        "generated_clip_triangle_count": clipped_face_count,
        "discarded_degenerate_clip_triangle_count": discarded_clip_slivers,
        "discarded_float32_degenerate_triangle_count": discarded_float32_degenerate_count,
    }


def _append_ply_mesh(vertices: list, faces: list, colors: list, roles: list, levels: list,
                     mesh: trimesh.Trimesh, role: int, level: int = -1,
                     corner_colors: np.ndarray | None = None,
                     face_colors: np.ndarray | None = None) -> int:
    source_faces = np.asarray(mesh.faces, dtype=np.int32)
    corners = np.asarray(mesh.vertices, dtype=np.float32)[source_faces]
    if corner_colors is None:
        vertex_colors = _rgb(mesh)
        corner_colors = vertex_colors[source_faces]
    if face_colors is not None:
        corner_colors = np.broadcast_to(face_colors[:, None, :], (len(source_faces), 3, 3))
    flat_vertices = corners.reshape(-1, 3)
    flat_colors = np.asarray(corner_colors, dtype=np.uint8).reshape(-1, 3)
    start = sum(len(item) for item in vertices)
    vertices.append(flat_vertices)
    faces.append(np.arange(start, start + len(flat_vertices), dtype=np.int32).reshape(-1, 3))
    colors.append(flat_colors)
    roles.append(np.full(len(source_faces), role, dtype=np.uint8))
    levels.append(np.full(len(source_faces), level, dtype=np.int32))
    return len(source_faces)


def _write_element_ply(path: Path, pieces: list[tuple[trimesh.Trimesh, np.ndarray | None, np.ndarray | None]]) -> None:
    vertices, faces, colors = [], [], []
    start = 0
    for mesh, corner_colors, face_colors in pieces:
        source_faces = np.asarray(mesh.faces, dtype=np.int32)
        corners = np.asarray(mesh.vertices, dtype=np.float32)[source_faces]
        if face_colors is not None:
            corner_colors = np.broadcast_to(face_colors[:, None, :], (len(source_faces), 3, 3))
        elif corner_colors is None:
            corner_colors = _rgb(mesh)[source_faces]
        flat_vertices = corners.reshape(-1, 3)
        vertices.append(flat_vertices)
        faces.append(np.arange(start, start + len(flat_vertices), dtype=np.int32).reshape(-1, 3))
        colors.append(np.asarray(corner_colors, dtype=np.uint8).reshape(-1, 3))
        start += len(flat_vertices)
    if not faces:
        raise ValueError(f"Cannot write an empty element PLY: {path}")
    export_ply_with_face_attributes(path, np.concatenate(vertices), np.concatenate(faces),
                                    np.concatenate(colors))


def _unmatched_wall_boundaries(wall: trimesh.Trimesh, stitches: list[dict]) -> list[dict]:
    matched = {int(item["wall_boundary_loop"]) for item in stitches}
    result = []
    for index, loop in enumerate(boundary_loops(wall.faces, wall.vertices)):
        if index in matched:
            continue
        points = wall.vertices[loop]
        result.append({
            "loop": index,
            "vertex_count": int(len(loop)),
            "bounds_xyz_m": np.stack((points.min(axis=0), points.max(axis=0))).tolist(),
            "mean_y_m": float(points[:, 1].mean()),
            "perimeter_m": float(np.linalg.norm(np.diff(np.vstack((points, points[:1])), axis=0), axis=1).sum()),
        })
    return result


def _orphan_ceilings(ceiling: trimesh.Trimesh, association: np.ndarray,
                     group_ids: np.ndarray, manifest: dict,
                     wall_max_y: float) -> tuple[list[tuple[int, trimesh.Trimesh]], list[dict]]:
    patches = {int(item["group_id"]): item for item in manifest.get("output_planes", [])
               if item.get("associated_floor_level") is None}
    groups = sorted(int(value) for value in np.unique(group_ids[association < 0]))
    elements, adjustments = [], []
    for group_id in groups:
        mask = (group_ids == group_id) & (association < 0)
        if not np.any(mask):
            continue
        patch = patches.get(group_id)
        if patch is None:
            raise ValueError(f"No fitted support plane is recorded for orphan ceiling group {group_id}")
        mesh = _subset(ceiling, mask)
        equation = np.asarray(patch.get("plane_equation"), dtype=np.float64)
        if equation.shape != (4,) or not np.isfinite(equation).all() or equation[1] < 0.7 * np.linalg.norm(equation[:3]):
            raise ValueError(f"Invalid upward-facing support plane for orphan ceiling group {group_id}")
        elements.append((group_id, mesh))
        adjustments.append({
            "ceiling_group_id": group_id,
            "original_max_y_m": float(mesh.vertices[:, 1].max()),
            "wall_max_y_m": float(wall_max_y),
            "downward_translation_m": 0.0,
            "translated_plane_equation": equation.tolist(),
        })
    return elements, adjustments


def assemble_final_regularized_mesh(workspace: Path, output: Path | None = None,
                                   elements_dir: Path | None = None,
                                   diagnostics_dir: Path | None = None) -> dict:
    workspace = workspace.resolve()
    root = resolve_elaboration_output_dir(workspace)
    levels_dir = root / "intermediate_floor_levels"
    wall_path = root / WALL_NAME
    ceiling_path = root / CEILING_NAME
    ceiling_manifest_path = root / CEILING_MANIFEST_NAME
    levels_path = levels_dir / "floor_level_intermediate_manifest.json"
    for path in (wall_path, ceiling_path, ceiling_manifest_path, levels_path):
        if not path.is_file():
            raise FileNotFoundError(f"Run Phase 06 Steps 1–4 first; missing {path}")

    wall_raw = _mesh(wall_path)
    ceiling_raw = _mesh(ceiling_path)
    ceiling_association = _face_property(ceiling_path, "associated_floor_level", len(ceiling_raw.faces))
    ceiling_groups = _face_property(ceiling_path, "ceiling_group_id", len(ceiling_raw.faces))
    ceiling_areas = _face_property(ceiling_path, "support_area_id", len(ceiling_raw.faces))
    ceiling_junctions = _face_property(
        ceiling_path, "is_vertical_junction", len(ceiling_raw.faces), default=0
    )
    ceiling_manifest = json.loads(ceiling_manifest_path.read_text(encoding="utf-8"))
    floor_manifest = json.loads(levels_path.read_text(encoding="utf-8"))
    floor_levels = {int(item["level"]): item for item in floor_manifest.get("levels", [])}
    if not floor_levels:
        raise ValueError(f"No fitted floor levels found in {levels_path}")
    floor_sources = {}
    for level_id in floor_levels:
        floor_path = levels_dir / f"floor_level_{level_id:03d}_patterns_regularized.ply"
        if not floor_path.is_file():
            raise FileNotFoundError(f"Final regularized floor is missing: {floor_path}")
        floor_sources[level_id] = _mesh(floor_path)
    floor_min_y_by_level = {
        level_id: float(floor.vertices[floor.faces, 1].min())
        for level_id, floor in floor_sources.items()
    }
    lowest_floor_candidate_level = min(
        floor_min_y_by_level, key=lambda level_id: (floor_min_y_by_level[level_id], level_id)
    )

    non_junction_faces = ceiling_raw.faces[ceiling_junctions == 0]
    if not len(non_junction_faces):
        raise ValueError("No non-junction ceiling surface is available to define the upper boundary")
    highest_ceiling_y = float(ceiling_raw.vertices[non_junction_faces, 1].max())
    upper_ceiling_patches = []
    for patch in ceiling_manifest.get("output_planes", []):
        equation = np.asarray(patch.get("plane_equation"), dtype=np.float64)
        if equation.shape != (4,) or not np.isfinite(equation).all() or abs(equation[1]) <= 1e-8:
            raise ValueError(f"Invalid fitted ceiling plane in manifest: {patch}")
        if equation[1] < 0:
            equation = -equation
        maximum_y = _ceiling_patch_max_y(
            ceiling_raw, ceiling_groups, ceiling_association,
            ceiling_areas, ceiling_junctions, patch,
        )
        upper_ceiling_patches.append((maximum_y, patch, equation / np.linalg.norm(equation[:3])))
    if upper_ceiling_patches:
        highest_surface_y, upper_ceiling_patch, wall_ceiling_plane = max(
            upper_ceiling_patches, key=lambda item: item[0]
        )
        wall_ceiling_source = {
            "ceiling_group_id": int(upper_ceiling_patch["group_id"]),
            "associated_floor_level": upper_ceiling_patch.get("associated_floor_level"),
            "highest_surface_y_m": float(highest_surface_y),
            "plane_equation": wall_ceiling_plane.tolist(),
        }
    else:
        wall_ceiling_plane = np.array([0.0, 1.0, 0.0, -highest_ceiling_y])
        wall_ceiling_source = {"ceiling_group_id": None, "plane_equation": wall_ceiling_plane.tolist(),
                               "fallback": "highest finalized ceiling vertex"}

    pairs: dict[tuple[int, int], set[int]] = {}
    for patch in ceiling_manifest.get("output_planes", []):
        level = patch.get("associated_floor_level")
        area = patch.get("support_area_id")
        if level is not None:
            if area is None or int(level) not in floor_levels:
                raise ValueError(f"Invalid associated floor/area in ceiling manifest: {patch}")
            pairs.setdefault((int(level), int(area)), set()).add(int(patch["group_id"]))

    orphan_elements, ceiling_adjustments = _orphan_ceilings(
        ceiling_raw, ceiling_association, ceiling_groups, ceiling_manifest,
        float(wall_raw.vertices[:, 1].max()),
    )
    ply_vertices, ply_faces, ply_colors, ply_roles, ply_levels = [], [], [], [], []
    slab_records = []
    element_exports: list[tuple[str, list[tuple[trimesh.Trimesh, np.ndarray | None, np.ndarray | None]]]] = []
    counts = {"walls": 0, "detached_floors": 0, "detached_ceilings": 0, "slabs": 0, "boundary_fill_faces": 0}
    wall_topology = _subset(wall_raw, np.ones(len(wall_raw.faces), dtype=bool))
    try:
        wall_footprint, wall_base_y = _wall_base_footprint(wall_topology)
        wall_footprint_source = "fitted_wall_base_contour"
    except ValueError:
        wall_base_y = float(wall_topology.vertices[:, 1].min())
        wall_footprint = _wall_contour_footprint(levels_dir, lowest_floor_candidate_level)
        if wall_footprint is None or wall_footprint.is_empty:
            raise
        wall_footprint_source = "lowest_floor_regularized_wall_contours"
    base_floor_level = min(
        floor_levels,
        key=lambda level_id: abs(
            float(floor_levels[level_id].get("elevation_y_median", np.inf)) - wall_base_y
        ),
    )
    base_floor_y = float(floor_levels[base_floor_level].get("elevation_y_median", np.inf))
    if abs(base_floor_y - wall_base_y) > BASE_FLOOR_ALIGNMENT_TOLERANCE_M:
        base_floor_level = None

    paired_areas: dict[int, set[int]] = {}
    assembled_floor_min_y = np.inf
    for (level_id, area_id), group_ids in sorted(pairs.items()):
        level = floor_levels[level_id]
        floor_path = levels_dir / f"floor_level_{level_id:03d}_patterns_regularized.ply"
        if not floor_path.is_file():
            raise FileNotFoundError(f"Final regularized floor is missing: {floor_path}")
        floor_raw = floor_sources[level_id]
        floor_area_ids = _face_property(floor_path, "floor_area_id", len(floor_raw.faces))
        area_meta = next((item for item in level.get("floor_areas", [])
                          if int(item.get("area_id", -1)) == area_id), None)
        if area_meta is None:
            raise ValueError(f"Floor level {level_id} has no fitted plane for area {area_id}")
        floor_area_mesh = _subset(floor_raw, floor_area_ids == area_id)
        floor_equation = np.asarray(area_meta.get("plane_equation"), dtype=np.float64)
        if floor_equation.shape != (4,) or not np.isfinite(floor_equation).all() or abs(floor_equation[1]) < 0.7:
            raise ValueError(f"Invalid fitted floor plane for level {level_id}, area {area_id}")

        junction_group_mask = np.isin(ceiling_groups, list(group_ids)) & (ceiling_areas < 0)
        ceiling_mask = (ceiling_association == level_id) & (
            (ceiling_areas == area_id) | junction_group_mask
        )
        if not np.any(ceiling_mask):
            raise ValueError(f"No finalized ceiling faces found for floor {level_id}, area {area_id}")
        ceiling_bottom = _subset(ceiling_raw, ceiling_mask)
        bottom_y = ceiling_bottom.vertices[:, 1]
        floor_y = -(floor_equation[0] * ceiling_bottom.vertices[:, 0]
                    + floor_equation[2] * ceiling_bottom.vertices[:, 2]
                    + floor_equation[3]) / floor_equation[1]
        floor_y += max(0.0, float(bottom_y.max() + 1e-5 - floor_y.min()))
        support_y_min = float(min(bottom_y.min(), floor_y.min()))
        support_y_max = float(max(bottom_y.max(), floor_y.max()))
        floor_contour = _regularized_wall_contour(levels_dir, level_id, area_id)
        wall_support = _wall_support_over_y_range(
            wall_topology, support_y_min, support_y_max, floor_contour,
        )
        ceiling_bottom, slab_clip = _clip_surface_to_footprint(ceiling_bottom, wall_support)
        precision_edge_count, precision_face_count = weld_float32_boundary_edges(ceiling_bottom)
        slab_clip["float32_ulp_boundary_edge_candidate_count"] = precision_edge_count
        slab_clip["float32_ulp_collapsed_face_count"] = precision_face_count
        slab_clip["support_y_range_m"] = [support_y_min, support_y_max]
        bottom_source_colors = _rgb(ceiling_bottom)
        floor_source_colors = _rgb(floor_area_mesh)
        solid, _, lift = close_slab(
            ceiling_bottom.vertices, ceiling_bottom.faces, floor_equation,
            diagnostics_dir=(None if diagnostics_dir is None else
                             diagnostics_dir / f"floor_{level_id:03d}_area_{area_id:03d}"),
            diagnostic_context={
                "floor_level": level_id,
                "floor_area_id": area_id,
                "ceiling_group_ids": sorted(group_ids),
                "support_y_range_m": [support_y_min, support_y_max],
                "float32_ulp_boundary_edge_candidate_count": precision_edge_count,
                "float32_ulp_collapsed_face_count": precision_face_count,
                "clip": slab_clip,
            },
        )
        if not solid.is_watertight:
            raise ValueError(f"Paired floor {level_id}/{area_id} did not produce a watertight slab")

        bottom_count = len(ceiling_bottom.faces)
        if len(solid.faces) < 2 * bottom_count:
            raise ValueError("Slab closure did not retain both paired surface caps")
        floor_cap_min_y = float(solid.vertices[solid.faces[bottom_count:2 * bottom_count], 1].min())
        assembled_floor_min_y = min(assembled_floor_min_y, floor_cap_min_y)
        side_color = np.rint((np.median(floor_source_colors, axis=0).astype(float)
                              + np.median(bottom_source_colors, axis=0).astype(float)) * 0.5).astype(np.uint8)
        face_corners = np.empty((len(solid.faces), 3, 3), dtype=np.uint8)
        face_corners[:bottom_count] = bottom_source_colors[
            cKDTree(ceiling_bottom.vertices).query(solid.vertices[solid.faces[:bottom_count]])[1]
        ]
        floor_tree = cKDTree(floor_area_mesh.vertices[:, [0, 2]])
        top_faces = solid.faces[bottom_count:2 * bottom_count]
        face_corners[bottom_count:2 * bottom_count] = floor_source_colors[
            floor_tree.query(solid.vertices[top_faces][:, :, [0, 2]])[1]
        ]
        face_corners[2 * bottom_count:] = side_color
        slab_count = _append_ply_mesh(
            ply_vertices, ply_faces, ply_colors, ply_roles, ply_levels,
            solid, 3, level_id, corner_colors=face_corners,
        )
        element_exports.append((
            f"slab_level_{level_id:03d}_area_{area_id:03d}.ply",
            [(solid, face_corners, None)],
        ))
        counts["slabs"] += slab_count
        paired_areas.setdefault(level_id, set()).add(area_id)
        slab_records.append({
            "floor_level": level_id,
            "floor_area_id": area_id,
            "triangle_count": slab_count,
            "is_watertight": bool(solid.is_watertight),
            "volume_m3": float(solid.volume),
            "additional_clearance_lift_m": float(lift),
            "wall_contour_clip": slab_clip,
        })

    floor_alignment = []
    for level_id, level in sorted(floor_levels.items()):
        floor_path = levels_dir / f"floor_level_{level_id:03d}_patterns_regularized.ply"
        floor_raw = floor_sources[level_id]
        area_ids = _face_property(floor_path, "floor_area_id", len(floor_raw.faces))
        unpaired = ~np.isin(area_ids, list(paired_areas.get(level_id, set())))
        if np.any(unpaired):
            for area_id in sorted(int(value) for value in np.unique(area_ids[unpaired])):
                area_mask = area_ids == area_id
                surface = _subset(floor_raw, area_mask)
                floor_contour = _regularized_wall_contour(levels_dir, level_id, area_id)
                target_y = wall_base_y if level_id == base_floor_level else None
                surface_y = np.asarray(surface.vertices)[:, 1]
                support_y_min = float(surface_y.min() + (0.0 if target_y is None else target_y - np.median(surface_y)))
                support_y_max = float(surface_y.max() + (0.0 if target_y is None else target_y - np.median(surface_y)))
                wall_support = _wall_support_over_y_range(
                    wall_topology, support_y_min, support_y_max, floor_contour,
                )
                surface, alignment = _clip_surface_to_footprint(surface, wall_support, target_y)
                area_min_y = float(surface.vertices[:, 1].min())
                assembled_floor_min_y = min(assembled_floor_min_y, area_min_y)
                alignment.update({
                    "floor_level": level_id,
                    "floor_area_id": area_id,
                    "support_y_range_m": [support_y_min, support_y_max],
                })
                floor_alignment.append(alignment)
                _append_ply_mesh(ply_vertices, ply_faces, ply_colors, ply_roles, ply_levels,
                                 surface, 1, level_id)
                name = f"floor_level_{level_id:03d}_area_{area_id:03d}"
                element_exports.append((f"{name}.ply", [(surface, None, None)]))
                counts["detached_floors"] += int(len(surface.faces))

    if not np.isfinite(assembled_floor_min_y):
        raise ValueError("No finalized floor surface is available to bound the wall shell")
    floor_alignment_by_area = {
        (int(item["floor_level"]), int(item["floor_area_id"])): float(item["vertical_translation_m"])
        for item in floor_alignment
    }
    floor_surface_min_by_level = {}
    for level_id, floor in floor_sources.items():
        floor_path = levels_dir / f"floor_level_{level_id:03d}_patterns_regularized.ply"
        area_ids = _face_property(floor_path, "floor_area_id", len(floor.faces))
        minimum = np.inf
        for area_id in np.unique(area_ids):
            selected_faces = floor.faces[area_ids == area_id]
            if not len(selected_faces):
                continue
            shift = floor_alignment_by_area.get((level_id, int(area_id)), 0.0)
            minimum = min(minimum, float(floor.vertices[selected_faces, 1].min()) + shift)
        if np.isfinite(minimum):
            floor_surface_min_by_level[level_id] = minimum
    if not floor_surface_min_by_level:
        raise ValueError("No fitted floor surface is available to define the lower boundary")
    lowest_floor_level = min(
        floor_surface_min_by_level,
        key=lambda level_id: (floor_surface_min_by_level[level_id], level_id),
    )
    lowest_floor_y = floor_surface_min_by_level[lowest_floor_level]
    lowest_floor_path = levels_dir / f"floor_level_{lowest_floor_level:03d}_patterns_regularized.ply"
    floor_boundary_supports = _floor_area_supports(
        levels_dir, floor_levels[lowest_floor_level], lowest_floor_path,
        floor_sources[lowest_floor_level],
    )
    for support in floor_boundary_supports:
        area_id = int(support["area_id"])
        shift = (0.0 if area_id in paired_areas.get(lowest_floor_level, set())
                 else floor_alignment_by_area.get((lowest_floor_level, area_id), 0.0))
        support["equation"][3] -= float(support["equation"][1]) * shift
        support["vertical_translation_m"] = float(shift)

    clipped_orphans = []
    for group_id, mesh in orphan_elements:
        mesh, clip = _clip_surface_to_footprint(
            mesh, wall_footprint, retain_degenerate_inside=True, allow_empty=True,
        )
        next(item for item in ceiling_adjustments if item["ceiling_group_id"] == group_id)[
            "wall_footprint_clip"
        ] = clip
        if len(mesh.faces):
            clipped_orphans.append((group_id, mesh))
    for group_id, mesh in clipped_orphans:
        name = f"ceiling_group_{group_id:03d}"
        _append_ply_mesh(ply_vertices, ply_faces, ply_colors, ply_roles, ply_levels, mesh, 2)
        pieces = [(mesh, None, None)]
        element_exports.append((f"{name}.ply", pieces))
        counts["detached_ceilings"] += int(len(mesh.faces))

    wall_floor_planes, wall_floor_selection = _wall_face_floor_planes(
        wall_raw, floor_boundary_supports, float(lowest_floor_y),
    )
    wall_final, wall_clip = _clip_wall_to_floor_and_ceiling(
        wall_raw, wall_floor_planes, wall_ceiling_plane,
    )
    wall_final, wall_gap_fill = _extend_wall_top_to_ceiling(
        wall_final, float(lowest_floor_y), wall_ceiling_plane,
        floor_supports=floor_boundary_supports,
    )
    wall_clip["lowest_floor_level"] = int(lowest_floor_level)
    wall_clip["lowest_floor_surface_y_m"] = float(lowest_floor_y)
    wall_clip["floor_boundary_selection"] = wall_floor_selection
    wall_clip["wall_footprint_source"] = wall_footprint_source
    wall_element_pieces = [(wall_final, None, None)]
    wall_vertices, wall_faces, wall_colors, wall_roles, wall_levels = [], [], [], [], []
    _append_ply_mesh(wall_vertices, wall_faces, wall_colors, wall_roles, wall_levels, wall_final, 0)
    wall_vertex_count = len(wall_vertices[0])
    ply_vertices[:0] = wall_vertices
    ply_faces = wall_faces + [faces + wall_vertex_count for faces in ply_faces]
    ply_colors[:0] = wall_colors
    ply_roles[:0] = wall_roles
    ply_levels[:0] = wall_levels
    counts["walls"] = len(wall_final.faces)

    stitches = []
    wall_boundary_mesh = _subset(wall_final, np.ones(len(wall_final.faces), dtype=bool))
    unmatched_wall_boundaries = _unmatched_wall_boundaries(wall_boundary_mesh, stitches)

    default_output = (root / OUTPUT_NAME).resolve()
    output_path = (output or default_output).resolve()
    manifest_path = output_path.with_name(MANIFEST_NAME)
    inputs = [wall_path, ceiling_path, ceiling_manifest_path, levels_path,
              *[levels_dir / f"floor_level_{level_id:03d}_patterns_regularized.ply" for level_id in floor_levels]]
    if output_path in {path.resolve() for path in inputs} or manifest_path in {path.resolve() for path in inputs}:
        raise ValueError("Final output must not overwrite a Phase 06 input")
    if output_path.exists() and output_path != default_output:
        raise ValueError("Refusing to overwrite an existing file outside the default final PLY path")
    if manifest_path.exists() and manifest_path.resolve() != (root / MANIFEST_NAME).resolve():
        raise ValueError("Refusing to overwrite an existing manifest outside the default final path")

    all_vertices = np.concatenate(ply_vertices)
    all_faces = np.concatenate(ply_faces)
    all_colors = np.concatenate(ply_colors)
    all_roles = np.concatenate(ply_roles)
    all_levels = np.concatenate(ply_levels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{output_path.name}.", suffix=".tmp",
                                     dir=output_path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        if not export_ply_with_face_attributes(
            temporary_path, all_vertices, all_faces, all_colors,
            face_attributes={"element_role": ("uchar", all_roles),
                             "source_floor_level": ("int", all_levels)},
        ):
            raise OSError(f"Could not write final regularized mesh: {temporary_path}")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    combined = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=False)
    combined.merge_vertices()
    edge_counts = np.bincount(combined.edges_unique_inverse)
    boundary_edges = combined.edges_unique[edge_counts == 1]
    boundary_y = combined.vertices[boundary_edges, 1] if len(boundary_edges) else np.empty(0)
    topology = {
        "is_watertight": bool(combined.is_watertight),
        "boundary_edge_count": int(np.count_nonzero(edge_counts == 1)),
        "nonmanifold_edge_count": int(np.count_nonzero(edge_counts > 2)),
        "surface_components": int(len(combined.split(only_watertight=False))),
        "boundary_edge_y_range_m": ([float(boundary_y.min()), float(boundary_y.max())]
                                    if len(boundary_y) else None),
        "floor_wall_base_y_m": wall_base_y,
    }
    result = {
        "algorithm": "phase06_final_walls_floors_ceilings_and_associated_slabs",
        "output_ply": str(output_path),
        "source_wall_ply": str(wall_path),
        "source_ceiling_ply": str(ceiling_path),
        "paired_slab_count": len(pairs),
        "slabs": slab_records,
        "counts": counts,
        "wall_ceiling_alignment": {
            "wall_max_y_m": float(wall_raw.vertices[:, 1].max()),
            "orphan_ceiling_adjustments": ceiling_adjustments,
            "wall_ceiling_plane": wall_ceiling_source,
            "wall_clipping": wall_clip,
            "wall_gap_fill": wall_gap_fill,
        },
        "floor_wall_alignment": floor_alignment,
        "boundary_stitches": stitches,
        "unmatched_wall_boundaries": unmatched_wall_boundaries,
        "topology": topology,
        "topology_note": (
            "Walls are clipped against the lowest finalized floor level's fitted area planes where a plane supports the full projected wall face, with the actual lowest floor Y as fallback, and against the fitted plane of the ceiling patch reaching the highest actual Y. Upper wall-perimeter edges below that ceiling plane are extended to close gaps, while loops on the local floor boundary remain untouched. Standalone floor footprints use their own wall contour; paired slabs are clipped to the intersection of fitted wall contours across their vertical span. Only the floor level nearest the source wall base is vertically aligned; no floor-wall connector is synthesized. "
            "Orphan ceilings retain their Step 4 height and relief inside the wall-base footprint; their projected overhang is clipped without changing paired slab closure. No separate roof cap is synthesized. Topology is audited from the exact float32 coordinates written to the PLY; separate element boundaries may remain open, while each paired slab must be watertight."
        ),
    }
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                     prefix=f".{manifest_path.name}.", suffix=".tmp",
                                     dir=manifest_path.parent, delete=False) as temporary:
        temporary.write(json.dumps(result, indent=2) + "\n")
        temporary_manifest = Path(temporary.name)
    try:
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    if elements_dir is not None:
        elements_dir = elements_dir.resolve()
        if elements_dir.exists() and any(elements_dir.iterdir()):
            raise ValueError(f"Element PLY directory must be empty: {elements_dir}")
        elements_dir.mkdir(parents=True, exist_ok=True)
        _write_element_ply(elements_dir / "walls.ply", wall_element_pieces)
        for name, pieces in element_exports:
            _write_element_ply(elements_dir / name, pieces)
    result["manifest"] = str(manifest_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="Phase 06 scan workspace after Step 4")
    parser.add_argument("--output", type=Path,
                        help=f"Final PLY path (default: mesh_elaboration_steps/{OUTPUT_NAME})")
    parser.add_argument("--elements-dir", type=Path,
                        help="Optional empty directory for separate wall, slab, detached floor, and detached ceiling PLYs")
    args = parser.parse_args(argv)
    try:
        result = assemble_final_regularized_mesh(args.workspace, args.output, args.elements_dir)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(f"Final regularized PLY -> {result['output_ply']}")
    print(f"Assembly manifest -> {result['manifest']}")
    print(f"Paired slabs: {result['paired_slab_count']}")
    print("Topology: " + json.dumps(result["topology"], sort_keys=True))
    print(f"Wall-shell boundary loops: {len(result['unmatched_wall_boundaries'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
