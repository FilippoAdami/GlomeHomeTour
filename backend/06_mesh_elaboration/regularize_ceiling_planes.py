#!/usr/bin/env python3
"""Fit ceiling groups to floor or wall support contours and regularize their appearance."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import open3d as o3d
import shapely
import trimesh
from scipy.optimize import linprog, minimize
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Polygon
from shapely.ops import polygonize, unary_union

STAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE_DIR))
sys.path.insert(0, str(STAGE_DIR.parent / "04_3DGS_to_mesh"))
from fill_mesh_holes import boundary_loops, export_ply_with_face_attributes, read_ply_face_flags
from planar_pattern_regularization import regularize_planar_patches
from regularize_floor_level_patterns import _polygon_with_holes
from structural_geometry import polygon_triangles
from surface_footprints import regularize_orthogonal_footprint
from mesh_elaboration_inputs import resolve_elaboration_output_dir

OUTPUT_NAME = "ceiling_planes_fitted_to_support_contours_regularized.ply"
MANIFEST_NAME = "ceiling_planes_fitted_to_support_contours_regularized_manifest.json"
MIN_CEILING_CLEARANCE_M = 0.10
WALL_SLICE_BELOW_CEILING_M = 0.30
MAX_WALL_TOP_CONTACT_GAP_M = 0.30
SUPPORT_TOLERANCE_M = 1e-8
MAX_SOURCE_SUPPORT_DISTANCE_M = 0.20
MIN_SOURCE_NORMAL_DOT = float(np.cos(np.deg2rad(30.0)))
PATTERN_MAX_EDGE_M = 0.15
MIN_WALL_TOP_LOOP_SPAN_M = 0.5
WALL_TOP_GRID_SIZE_M = 0.20
WALL_TOP_FIT_QUANTILE = 0.95
WALL_TOP_CLEARANCE_M = 0.02


def _read_flag(path: Path, name: str, count: int, default: int) -> np.ndarray:
    flags = read_ply_face_flags(path, name)
    if flags is None:
        return np.full(count, default, dtype=np.int32)
    if len(flags) != count:
        raise ValueError(f"{path} has a mismatched {name!r} face property")
    return np.asarray(flags)


def _plane_values(equation: np.ndarray, xz: np.ndarray) -> np.ndarray:
    equation = np.asarray(equation, dtype=np.float64).reshape(4)
    if abs(equation[1]) <= 1e-8:
        raise ValueError("Ceiling support plane is too vertical to project onto X/Z")
    return -(equation[0] * xz[:, 0] + equation[2] * xz[:, 1] + equation[3]) / equation[1]


def _plane_mesh(polygon, equation: np.ndarray) -> trimesh.Trimesh:
    xz, faces = polygon_triangles(polygon)
    if not len(faces):
        raise ValueError("Support contour could not be triangulated")
    vertices = np.column_stack((xz[:, 0], _plane_values(equation, xz), xz[:, 1]))
    normal = equation[:3] / np.linalg.norm(equation[:3])
    crosses = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                       vertices[faces[:, 2]] - vertices[faces[:, 0]])
    if float(np.mean(crosses @ normal)) < 0:
        faces = faces[:, ::-1]
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _polygon_boundary_xz(polygon) -> np.ndarray:
    components = list(polygon.geoms) if hasattr(polygon, "geoms") else [polygon]
    rings = []
    for component in components:
        if not isinstance(component, Polygon):
            continue
        rings.append(np.asarray(component.exterior.coords[:-1], dtype=np.float64))
        rings.extend(np.asarray(ring.coords[:-1], dtype=np.float64) for ring in component.interiors)
    if not rings:
        raise ValueError("Support polygon has no boundary rings")
    return np.concatenate(rings)


def _projected_footprint(vertices: np.ndarray, faces: np.ndarray):
    projected_triangles = vertices[faces][:, :, [0, 2]]
    twice_area = np.abs(
        (projected_triangles[:, 1, 0] - projected_triangles[:, 0, 0])
        * (projected_triangles[:, 2, 1] - projected_triangles[:, 0, 1])
        - (projected_triangles[:, 1, 1] - projected_triangles[:, 0, 1])
        * (projected_triangles[:, 2, 0] - projected_triangles[:, 0, 0])
    )
    projected_triangles = projected_triangles[twice_area > 1e-12]
    if not len(projected_triangles):
        return MultiPoint(vertices[np.unique(faces), :][:, [0, 2]]).convex_hull
    return shapely.union_all(shapely.polygons(projected_triangles), grid_size=1e-4)


def _source_support(path: Path, equation: np.ndarray, *, surface_role: int | None = None,
                    footprint=None) -> dict:
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    if not len(vertices) or not len(faces) or len(colors) != len(vertices):
        raise ValueError(f"Fitted ceiling support PLY is empty or has no vertex colors: {path}")
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"Fitted ceiling support PLY has invalid geometry: {path}")

    previews = _read_flag(path, "is_fitted_plane", len(faces), 0) == 0
    hole_fill = _read_flag(path, "is_hole_fill", len(faces), 0) == 0
    corners = vertices[faces]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = previews & hole_fill & (double_area > 1e-12)
    centers = corners.mean(axis=1)
    if surface_role is not None:
        roles = _read_flag(path, "surface_role", len(faces), -1)
        valid &= roles == surface_role
    if footprint is not None:
        valid &= shapely.contains_xy(footprint, centers[:, 0], centers[:, 2])
    plane = equation / np.linalg.norm(equation[:3])
    normal = plane[:3]
    face_normals = cross / np.maximum(double_area[:, None], 1e-12)
    valid &= (np.abs(centers @ normal + plane[3]) <= MAX_SOURCE_SUPPORT_DISTANCE_M)
    valid &= (np.abs(face_normals @ normal) >= MIN_SOURCE_NORMAL_DOT)
    if not np.any(valid):
        raise ValueError(f"No original ceiling triangles support the fitted plane in {path}")
    support_faces = faces[valid]
    support_corners = vertices[support_faces]
    support_cross = np.cross(support_corners[:, 1] - support_corners[:, 0],
                             support_corners[:, 2] - support_corners[:, 0])
    areas = np.linalg.norm(support_cross, axis=1) * 0.5
    support_centers = support_corners.mean(axis=1)
    return {
        "mesh": mesh,
        "projected_footprint": _projected_footprint(vertices, support_faces),
        "support_centers": support_centers,
        "support_areas": areas,
        "support_colors": colors[support_faces].mean(axis=1),
        "source_triangle_count": int(len(support_faces)),
    }


def _floor_supports(elaboration_dir: Path) -> dict[int, dict]:
    level_dir = elaboration_dir / "intermediate_floor_levels"
    levels = json.loads((level_dir / "floor_level_intermediate_manifest.json").read_text())
    floor_polygons = {}
    for level in levels.get("levels", []):
        level_id = int(level["level"])
        polygon_path = level_dir / f"floor_level_{level_id:03d}_polygons.json"
        if not polygon_path.is_file():
            raise FileNotFoundError(f"Final floor polygons are missing: {polygon_path}")
        data = json.loads(polygon_path.read_text())
        areas = []
        fitted_by_id = {int(area["area_id"]): area for area in level.get("floor_areas", [])}
        for area in data.get("areas", []):
            area_id = int(area["area_id"])
            fitted = fitted_by_id.get(area_id)
            if fitted is None:
                raise ValueError(f"Floor polygon area {area_id} has no fitted plane at {polygon_path}")
            polygon = _polygon_with_holes(area.get("floor_polygon_rings_world_xz_m", []))
            equation = np.asarray(fitted.get("plane_equation"), dtype=np.float64)
            if equation.shape != (4,) or not np.isfinite(equation).all():
                raise ValueError(f"Invalid fitted floor plane for level {level_id}, area {area_id}")
            equation /= np.linalg.norm(equation[:3])
            areas.append({"area_id": area_id, "polygon": polygon, "equation": equation})
        if not areas:
            raise ValueError(f"Floor level {level_id} has no final polygon areas")
        fitted_name = level.get("fitted_output_file")
        if not isinstance(fitted_name, str):
            raise ValueError(f"Floor level {level_id} has no cleaned fitted PLY entry")
        floor_polygons[level_id] = {
            "path": polygon_path, "fitted_path": level_dir / fitted_name, "areas": areas,
        }
    return floor_polygons


def _slice_triangle(triangle: np.ndarray, height: float) -> list[np.ndarray]:
    points = []
    for first_index, second_index in ((0, 1), (1, 2), (2, 0)):
        first, second = triangle[first_index], triangle[second_index]
        dy_first, dy_second = first[1] - height, second[1] - height
        if abs(dy_first) <= SUPPORT_TOLERANCE_M:
            points.append(first[[0, 2]])
        if dy_first * dy_second < -SUPPORT_TOLERANCE_M ** 2:
            ratio = (height - first[1]) / (second[1] - first[1])
            point = first + ratio * (second - first)
            points.append(point[[0, 2]])
    unique = []
    for point in points:
        if not any(np.linalg.norm(point - previous) <= SUPPORT_TOLERANCE_M for previous in unique):
            unique.append(point)
    if len(unique) < 2:
        return []
    pairs = [(np.linalg.norm(unique[i] - unique[j]), i, j)
             for i in range(len(unique)) for j in range(i + 1, len(unique))]
    _, first, second = max(pairs)
    if np.linalg.norm(unique[first] - unique[second]) <= SUPPORT_TOLERANCE_M:
        return []
    return [unique[first], unique[second]]


def _wall_contours_at(wall_mesh: o3d.geometry.TriangleMesh, height: float):
    vertices, faces = np.asarray(wall_mesh.vertices), np.asarray(wall_mesh.triangles)
    segments = []
    for triangle in vertices[faces]:
        points = _slice_triangle(triangle, height)
        if points:
            segments.append(LineString(points))
    if not segments:
        raise ValueError(f"Fitted wall shell does not cross Y={height:.4f} m")
    linework = shapely.set_precision(unary_union(segments), grid_size=1e-4)
    polygons = list(polygonize(linework))
    if not polygons:
        raise ValueError(f"Could not form a closed wall contour at Y={height:.4f} m")
    # Symmetric difference preserves nested loops as holes while joining adjacent cells.
    result = GeometryCollection()
    for polygon in polygons:
        result = result.symmetric_difference(polygon)
    if result.is_empty or result.area <= 1e-8 or not result.is_valid:
        raise ValueError(f"Wall contour at Y={height:.4f} m is empty or invalid")
    return regularize_orthogonal_footprint(result, min_edge=0.10, tolerance=0.02)


def _wall_top_plane(wall_meshes: o3d.geometry.TriangleMesh | list[o3d.geometry.TriangleMesh]) -> tuple[np.ndarray, dict]:
    """Fit an upper-quantile plane to spatial samples of significant wall boundaries."""
    if isinstance(wall_meshes, o3d.geometry.TriangleMesh):
        wall_meshes = [wall_meshes]
    boundary_points, loop_records = [], []
    for mesh_index, wall_mesh in enumerate(wall_meshes):
        topology = trimesh.Trimesh(
            vertices=np.asarray(wall_mesh.vertices, dtype=np.float64),
            faces=np.asarray(wall_mesh.triangles, dtype=np.int64), process=False,
        )
        topology.merge_vertices(digits_vertex=6)
        for loop_index, loop in enumerate(boundary_loops(topology.faces, topology.vertices)):
            loop_points = topology.vertices[loop]
            if np.linalg.norm(np.ptp(loop_points[:, [0, 2]], axis=0)) < MIN_WALL_TOP_LOOP_SPAN_M:
                continue
            boundary_points.append(loop_points)
            loop_records.append({"wall_mesh_index": int(mesh_index), "loop_index": int(loop_index)})
    if not boundary_points:
        raise ValueError("Extracted wall instances have no significant boundary for a top-contour plane")
    points = np.unique(np.concatenate(boundary_points), axis=0)
    grid = np.floor(points[:, [0, 2]] / WALL_TOP_GRID_SIZE_M).astype(np.int64)
    _, cell_ids = np.unique(grid, axis=0, return_inverse=True)
    cell_max_y = np.full(int(cell_ids.max()) + 1, -np.inf)
    np.maximum.at(cell_max_y, cell_ids, points[:, 1])
    top = points[:, 1] >= cell_max_y[cell_ids] - 0.02
    sums = np.zeros((len(cell_max_y), 3), dtype=np.float64)
    counts = np.zeros(len(cell_max_y), dtype=np.int64)
    np.add.at(sums, cell_ids[top], points[top])
    np.add.at(counts, cell_ids[top], 1)
    samples = sums[counts > 0] / counts[counts > 0, None]
    design = np.column_stack((samples[:, 0], samples[:, 2], np.ones(len(samples))))
    if len(samples) < 3 or np.linalg.matrix_rank(design) < 3:
        raise ValueError("Upper extracted-wall contour does not span a plane in X/Z")

    # Quantile fitting keeps lower openings and missing wall pieces from pulling the cap down.
    count = len(samples)
    equality = np.column_stack((design, np.eye(count), -np.eye(count)))
    objective = np.concatenate((
        np.zeros(3),
        np.full(count, WALL_TOP_FIT_QUANTILE),
        np.full(count, 1.0 - WALL_TOP_FIT_QUANTILE),
    ))
    fitted = linprog(
        objective, A_eq=equality, b_eq=samples[:, 1],
        bounds=[(None, None)] * 3 + [(0.0, None)] * (2 * count), method="highs",
    )
    if not fitted.success:
        raise ValueError(f"Could not fit the extracted-wall top contour: {fitted.message}")
    coefficients = fitted.x[:3]
    residual = samples[:, 1] - design @ coefficients
    clearance = max(0.0, float(residual.max())) + WALL_TOP_CLEARANCE_M
    coefficients[2] += clearance
    residual -= clearance
    upper_residual = np.sort(np.abs(residual))[:max(3, int(np.ceil(0.10 * count)))]
    a, b, c = coefficients
    plane = np.array([-a, 1.0, -b, -c], dtype=np.float64)
    plane /= np.linalg.norm(plane[:3])
    return plane, {
        "plane_equation": plane.tolist(),
        "boundary_loops": loop_records,
        "boundary_vertex_count": int(sum(len(points) for points in boundary_points)),
        "fit_sample_count": int(count),
        "fit_quantile": WALL_TOP_FIT_QUANTILE,
        "grid_size_m": WALL_TOP_GRID_SIZE_M,
        "vertical_clearance_m": float(clearance),
        "boundary_y_range_m": [float(points[:, 1].min()), float(points[:, 1].max())],
        "fit_rmse_m": float(np.sqrt(np.mean(upper_residual ** 2))),
    }


def _extracted_wall_meshes(elaboration_dir: Path) -> tuple[list[o3d.geometry.TriangleMesh], list[Path]]:
    manifest_path = elaboration_dir / "structural_instances_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Extracted wall support manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [item for item in manifest.get("instances", []) if item.get("role") == "wall"]
    if not records:
        raise ValueError(f"No extracted wall instances are listed in {manifest_path}")
    paths = [elaboration_dir / Path(item["file_name"]).name for item in records]
    meshes = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Extracted wall instance PLY is missing: {path}")
        mesh = o3d.io.read_triangle_mesh(str(path))
        if not len(mesh.vertices) or not len(mesh.triangles):
            raise ValueError(f"Extracted wall instance PLY is empty: {path}")
        meshes.append(mesh)
    return meshes, paths


def _orphan_levels(records: list[dict]) -> list[list[dict]]:
    remaining = set(range(len(records)))
    levels = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        level, queue = [], [first]
        while queue:
            index = queue.pop()
            current = records[index]
            level.append(current)
            neighbors = []
            for other_index in sorted(remaining):
                other = records[other_index]
                first = current["source"]["projected_footprint"]
                second = other["source"]["projected_footprint"]
                overlap_ratio = first.intersection(second).area / max(min(first.area, second.area), 1e-9)
                if first.distance(second) <= 0.05 and overlap_ratio <= 0.10:
                    neighbors.append(other_index)
            for other_index in neighbors:
                remaining.remove(other_index)
                queue.append(other_index)
        levels.append(level)
    return levels


def _wall_support_for_level(records: list[dict], wall_mesh: o3d.geometry.TriangleMesh):
    hulls = [record["source"]["projected_footprint"] for record in records]
    lowest_ceiling_y = min(
        float(_plane_values(record["equation"], _polygon_boundary_xz(hull)).min())
        for record, hull in zip(records, hulls)
    )
    slice_y = lowest_ceiling_y - WALL_SLICE_BELOW_CEILING_M
    contours = _wall_contours_at(wall_mesh, slice_y)
    candidates = list(contours.geoms) if hasattr(contours, "geoms") else [contours]
    selected = [polygon for polygon in candidates
                if isinstance(polygon, Polygon)
                and any(polygon.intersection(hull).area > 1e-4 for hull in hulls)]
    if not selected:
        raise ValueError(
            f"No wall contour at Y={slice_y:.3f} m overlaps ceiling groups "
            f"{[record['group']['group_id'] for record in records]}"
        )
    return [{"polygon": unary_union(selected), "area_id": None, "floor_equation": None,
             "support_source": "wall_slice", "support_height_y_m": slice_y}]


def _partition_support(polygon, records: list[dict]) -> list[tuple[dict, object]]:
    if len(records) == 1:
        return [(records[0], polygon)]
    point = polygon.representative_point()
    point_xz = np.asarray([[point.x, point.y]], dtype=np.float64)
    ordered = sorted(records, key=lambda record:
                       float(_plane_values(record["equation"], point_xz)[0]), reverse=True)
    remaining = polygon
    assigned = []
    for record in ordered:
        region = polygon.intersection(record["source"]["projected_footprint"]).intersection(remaining)
        if not region.is_empty and region.area > 1e-8:
            assigned.append((record, region))
            remaining = remaining.difference(region)
    if not remaining.is_empty and remaining.area > 1e-8:
        nearest = min(records, key=lambda record:
                      record["source"]["projected_footprint"].distance(remaining.representative_point()))
        assigned.append((nearest, remaining))
    return assigned


def _supports_for_level(records: list[dict], floor_polygons: dict,
                         wall_mesh: o3d.geometry.TriangleMesh | None,
                         wall_top_equation: np.ndarray | None = None) -> list[tuple[dict, dict]]:
    floor_level = records[0]["group"].get("associated_floor_level")
    if floor_level is None:
        supports = _wall_support_for_level(records, wall_mesh)
    else:
        floor_level = int(floor_level)
        if floor_level not in floor_polygons:
            raise ValueError(f"Ceiling group references missing floor {floor_level}")
        supports = [{"polygon": area["polygon"], "area_id": area["area_id"],
                     "floor_equation": area["equation"], "support_source": "associated_floor",
                     "support_height_y_m": None,
                     "wall_top_plane_equation": wall_top_equation}
                    for area in floor_polygons[floor_level]["areas"]]
    return [(record, {**support, "polygon": region,
                      "wall_top_plane_equation": wall_top_equation})
            for support in supports
            for record, region in _partition_support(support["polygon"], records)]


def _clearance_adjustment(polygon, equation: np.ndarray, floor_equation: np.ndarray):
    boundary = _polygon_boundary_xz(polygon)
    ceiling_points = np.column_stack((boundary[:, 0], _plane_values(equation, boundary), boundary[:, 1]))
    signed = ceiling_points @ floor_equation[:3] + floor_equation[3]
    if float(signed.mean()) < 0.0:
        gap = max(0.0, -float(signed.max()))
        shift = max(0.0, MIN_CEILING_CLEARANCE_M + float(signed.max()))
        side = -1.0
    else:
        gap = max(0.0, float(signed.min()))
        shift = max(0.0, MIN_CEILING_CLEARANCE_M - float(signed.min()))
        side = 1.0
    if shift:
        direction = side * floor_equation[:3]
        translation = direction * shift
        moved = equation.copy()
        moved[3] -= float(moved[:3] @ translation)
        signed += side * shift
    final_gap = max(0.0, -float(signed.max())) if side < 0 else max(0.0, float(signed.min()))
    return moved if shift else equation.copy(), translation if shift else np.zeros(3), gap, final_gap


def _fit_plane_above_wall_top(equation: np.ndarray, polygon,
                              wall_top_equation: np.ndarray) -> tuple[np.ndarray, dict]:
    """Keep the fitted ceiling plane closest to its source while staying above the wall cap."""
    equation = np.asarray(equation, dtype=np.float64)
    wall_top_equation = np.asarray(wall_top_equation, dtype=np.float64)
    boundary = _polygon_boundary_xz(polygon)
    initial_clearance = _plane_values(equation, boundary) - _plane_values(wall_top_equation, boundary)
    initial_coefficients = np.array([
        -equation[0] / equation[1], -equation[2] / equation[1], -equation[3] / equation[1],
    ])
    if float(initial_clearance.min()) >= -1e-9:
        return equation.copy(), {
            "applied": False,
            "initial_min_clearance_m": float(initial_clearance.min()),
            "final_min_clearance_m": float(initial_clearance.min()),
            "maximum_boundary_adjustment_m": 0.0,
        }

    xz_vertices, faces = polygon_triangles(polygon)
    samples = np.vstack((boundary, xz_vertices[faces].mean(axis=1)))
    design = np.column_stack((samples[:, 0], samples[:, 1], np.ones(len(samples))))
    source_y = design @ initial_coefficients
    constraint_design = np.column_stack((boundary[:, 0], boundary[:, 1], np.ones(len(boundary))))
    wall_y = _plane_values(wall_top_equation, boundary)

    def objective(coefficients):
        residual = design @ coefficients - source_y
        return 0.5 * float(residual @ residual)

    def gradient(coefficients):
        return design.T @ (design @ coefficients - source_y)

    result = minimize(
        objective, np.array([-wall_top_equation[0] / wall_top_equation[1],
                             -wall_top_equation[2] / wall_top_equation[1],
                             -wall_top_equation[3] / wall_top_equation[1]]),
        jac=gradient, method="SLSQP",
        constraints=[{
            "type": "ineq",
            "fun": lambda coefficients: constraint_design @ coefficients - wall_y,
            "jac": lambda coefficients: constraint_design,
        }],
        options={"ftol": 1e-12, "maxiter": 100},
    )
    wall_coefficients = np.array([
        -wall_top_equation[0] / wall_top_equation[1],
        -wall_top_equation[2] / wall_top_equation[1],
        -wall_top_equation[3] / wall_top_equation[1],
    ])
    optimizer_feasible = (
        np.asarray(result.x).shape == (3,) and np.isfinite(result.x).all()
        and float(np.min(constraint_design @ result.x - wall_y)) >= -1e-7
    )
    coefficients = result.x if optimizer_feasible else wall_coefficients
    candidate = np.array([-coefficients[0], 1.0, -coefficients[1], -coefficients[2]])
    candidate /= np.linalg.norm(candidate[:3])
    final_clearance = _plane_values(candidate, boundary) - wall_y
    if float(final_clearance.min()) < -1e-7:
        raise ValueError("Constrained ceiling fit falls below the upper wall contour")
    adjustment = _plane_values(candidate, boundary) - _plane_values(equation, boundary)
    return candidate, {
        "applied": True,
        "optimizer_success": bool(result.success),
        "optimizer_feasible": bool(optimizer_feasible),
        "fallback_to_wall_top_plane": not bool(optimizer_feasible),
        "initial_min_clearance_m": float(initial_clearance.min()),
        "final_min_clearance_m": float(final_clearance.min()),
        "maximum_boundary_adjustment_m": float(np.max(np.abs(adjustment))),
    }


def _vertical_junctions(patches: list[dict]) -> list[dict]:
    junctions = []
    for first_index, first in enumerate(patches):
        for second in patches[first_index + 1:]:
            if first["level_key"] != second["level_key"]:
                continue
            shared = first["polygon"].boundary.intersection(second["polygon"].boundary)
            lines = [shared] if shared.geom_type == "LineString" else [
                item for item in getattr(shared, "geoms", []) if item.geom_type == "LineString"
            ]
            for line in lines:
                coords = np.asarray(line.coords, dtype=np.float64)
                for start, end in zip(coords[:-1], coords[1:]):
                    if np.linalg.norm(end - start) <= 1e-5:
                        continue
                    xz = np.asarray([start, end])
                    y_first = _plane_values(first["plane_equation"], xz)
                    y_second = _plane_values(second["plane_equation"], xz)
                    if np.max(np.abs(y_first - y_second)) <= 1e-4:
                        continue
                    vertices = np.asarray([
                        [start[0], y_first[0], start[1]], [end[0], y_first[1], end[1]],
                        [end[0], y_second[1], end[1]], [start[0], y_second[0], start[1]],
                    ])
                    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                    if mesh.area <= 1e-8:
                        continue
                    first_color = np.median(
                        np.asarray(first["mesh"].visual.face_colors, dtype=np.float64)[:, :3], axis=0
                    )
                    second_color = np.median(
                        np.asarray(second["mesh"].visual.face_colors, dtype=np.float64)[:, :3], axis=0
                    )
                    rgba = np.r_[np.rint((first_color + second_color) * 0.5).astype(np.uint8), 255]
                    mesh.visual = trimesh.visual.ColorVisuals(
                        mesh=mesh, face_colors=np.tile(rgba, (len(mesh.faces), 1))
                    )
                    cross = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
                    normal = cross / np.linalg.norm(cross)
                    equation = np.r_[normal, -float(normal @ vertices[0])]
                    junctions.append({
                        "name": f"ceiling_junction_{first['group_id']:03d}_{second['group_id']:03d}",
                        "group_id": first["group_id"], "associated_floor_level": first["floor_level"],
                        "level_key": first["level_key"],
                        "polygon": LineString(xz), "plane_equation": equation.tolist(), "mesh": mesh,
                        "support_centers": np.empty((0, 3)), "support_areas": np.empty(0),
                        "support_colors": np.empty((0, 3)), "is_junction": True,
                    })
    return junctions


def _fit_group_patches(record: dict, support: dict) -> dict:
    group = record["group"]
    group_mesh = record["source"]
    equation = record["equation"]
    polygon = regularize_orthogonal_footprint(
        support["polygon"], min_edge=0.10, tolerance=0.02
    )
    if polygon.is_empty or polygon.area <= 1e-8:
        raise ValueError(f"Ceiling group {group['group_id']} has no usable support contour")
    plane_equation, translation = equation.copy(), np.zeros(3)
    original_gap = adjusted_gap = None
    floor_equation = support.get("floor_equation")
    if floor_equation is not None:
        plane_equation, translation, original_gap, adjusted_gap = _clearance_adjustment(
            polygon, plane_equation, floor_equation
        )
    wall_top_equation = support.get("wall_top_plane_equation")
    wall_top_clearance = None
    wall_top_max_clearance = None
    if wall_top_equation is not None:
        boundary = _polygon_boundary_xz(polygon)
        clearances = (
            _plane_values(plane_equation, boundary)
            - _plane_values(np.asarray(wall_top_equation), boundary)
        )
        wall_top_clearance = float(clearances.min())
        wall_top_max_clearance = float(clearances.max())
    wall_top_fit = {"applied": False}
    if wall_top_equation is not None and wall_top_max_clearance >= -MAX_WALL_TOP_CONTACT_GAP_M:
        plane_equation, wall_top_fit = _fit_plane_above_wall_top(
            plane_equation, polygon, np.asarray(wall_top_equation),
        )
        wall_top_clearance = float(np.min(
            _plane_values(plane_equation, boundary)
            - _plane_values(np.asarray(wall_top_equation), boundary)
        ))
    centers = np.asarray(group_mesh["support_centers"], dtype=np.float64)
    inside = shapely.contains_xy(polygon, centers[:, 0], centers[:, 2])
    return {
        "name": f"ceiling_group_{int(group['group_id']):03d}_area_{support['area_id']}",
        "group_id": int(group["group_id"]),
        "floor_level": group.get("associated_floor_level"),
        "level_key": record["level_key"],
        "area_id": support.get("area_id"),
        "support_source": support["support_source"],
        "support_height_y_m": support.get("support_height_y_m"),
        "polygon": polygon,
        "mesh": _plane_mesh(polygon, plane_equation),
        "plane_equation": plane_equation.tolist(),
        "relief_direction": [0.0, -1.0, 0.0],
        "translation": translation,
        "original_perpendicular_gap_m": original_gap,
        "final_perpendicular_gap_m": adjusted_gap,
        "support_centers": centers[inside] + translation,
        "support_areas": group_mesh["support_areas"][inside],
        "support_colors": group_mesh["support_colors"][inside],
        "floor_equation": floor_equation,
        "wall_top_plane_equation": (None if wall_top_equation is None
                                     else np.asarray(wall_top_equation).tolist()),
        "wall_top_clearance_before_m": wall_top_clearance,
        "wall_top_max_clearance_before_m": wall_top_max_clearance,
        "wall_top_clearance_after_m": wall_top_clearance,
        "wall_top_plane_fit": wall_top_fit,
        "source_triangle_count": int(np.count_nonzero(inside)),
    }


def _enforce_wall_top_constraint(patch: dict, wall_top_equation: np.ndarray) -> None:
    """Raise a fitted ceiling only as far as needed to clear the wall-top plane."""
    wall_top_equation = np.asarray(wall_top_equation, dtype=np.float64)
    plane_equation = np.asarray(patch["plane_equation"], dtype=np.float64)
    boundary = _polygon_boundary_xz(patch["polygon"])
    plane_clearance = _plane_values(plane_equation, boundary) - _plane_values(wall_top_equation, boundary)
    mesh_vertices = np.asarray(patch["mesh"].vertices, dtype=np.float64)
    mesh_clearance = mesh_vertices[:, 1] - _plane_values(wall_top_equation, mesh_vertices[:, [0, 2]])
    shift = max(0.0, -float(plane_clearance.min()), -float(mesh_clearance.min()))
    if shift:
        patch["mesh"].vertices = mesh_vertices + np.array([0.0, shift, 0.0])
        plane_equation[3] -= plane_equation[1] * shift
        patch["plane_equation"] = plane_equation.tolist()
        patch["translation"] += np.array([0.0, shift, 0.0])
    final_plane = np.asarray(patch["plane_equation"], dtype=np.float64)
    patch["wall_top_plane_equation"] = wall_top_equation.tolist()
    patch["wall_top_clearance_after_m"] = float(np.min(
        _plane_values(final_plane, boundary) - _plane_values(wall_top_equation, boundary)
    ))
    patch["wall_top_vertical_translation_m"] = float(shift)
    patch["wall_top_constraint_applied"] = True


def _is_upper_wall_ceiling(patch: dict) -> bool:
    """Only constrain roof patches that reach the upper wall contour, not floor slabs."""
    maximum_gap = patch.get("wall_top_max_clearance_before_m")
    return maximum_gap is not None and maximum_gap >= -MAX_WALL_TOP_CONTACT_GAP_M


def _enforce_output_clearance(patch: dict) -> None:
    floor_equation = patch.get("floor_equation")
    if floor_equation is None:
        return
    floor_equation = np.asarray(floor_equation, dtype=np.float64)
    vertices = np.asarray(patch["mesh"].vertices, dtype=np.float64)
    signed = vertices @ floor_equation[:3] + floor_equation[3]
    if signed.mean() < 0.0:
        gap = max(0.0, -float(signed.max()))
        shift, side = max(0.0, MIN_CEILING_CLEARANCE_M + float(signed.max())), -1.0
    else:
        gap = max(0.0, float(signed.min()))
        shift, side = max(0.0, MIN_CEILING_CLEARANCE_M - float(signed.min())), 1.0
    if shift:
        translation = side * floor_equation[:3] * shift
        patch["mesh"].vertices = vertices + translation
        equation = np.asarray(patch["plane_equation"], dtype=np.float64)
        equation[3] -= float(equation[:3] @ translation)
        patch["plane_equation"] = equation.tolist()
        patch["translation"] += translation
        signed += side * shift
    patch["final_perpendicular_gap_m"] = (
        max(0.0, -float(signed.max())) if side < 0 else max(0.0, float(signed.min()))
    )


def _serialized(value):
    if isinstance(value, dict):
        return {str(key): _serialized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialized(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def regularize_ceiling_planes(workspace: Path, output: Path | None = None,
                              wall_support: Path | None = None) -> dict:
    elaboration_dir = resolve_elaboration_output_dir(workspace)
    level_dir = elaboration_dir / "intermediate_floor_levels"
    groups_path = level_dir / "ceiling_plane_groups_manifest.json"
    if not groups_path.is_file():
        raise FileNotFoundError(f"Run ceiling plane fitting first; expected {groups_path}")
    wall_path = wall_support or elaboration_dir / "wall_enclosing_planes_fitted_to_regularized_contours.ply"
    groups_data = json.loads(groups_path.read_text(encoding="utf-8"))
    groups = groups_data.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"No fitted ceiling groups found in {groups_path}")
    floor_polygons = _floor_supports(elaboration_dir)
    if not wall_path.is_file():
        raise FileNotFoundError(
            f"Ceiling fitting needs the contour-fitted wall PLY: {wall_path}; "
            "run fit_enclosing_walls_to_regularized_contours.py first"
        )
    wall_mesh = o3d.io.read_triangle_mesh(str(wall_path))
    if not len(wall_mesh.vertices) or not len(wall_mesh.triangles):
        raise ValueError(f"Fitted wall support PLY is empty: {wall_path}")
    extracted_wall_meshes, extracted_wall_paths = _extracted_wall_meshes(elaboration_dir)
    wall_top_equation, wall_top_info = _wall_top_plane(extracted_wall_meshes)

    patches, source_info = [], []
    records, groups_by_floor = [], {}
    for group in groups:
        plane = np.asarray(group.get("plane_equation"), dtype=np.float64)
        if plane.shape != (4,) or not np.isfinite(plane).all() or np.linalg.norm(plane[:3]) <= 1e-10:
            raise ValueError(f"Invalid fitted ceiling equation in group {group.get('group_id')}")
        plane /= np.linalg.norm(plane[:3])
        fitted_name = group.get("fitted_output_file")
        source_path = level_dir / fitted_name if isinstance(fitted_name, str) else None
        if source_path is None or not source_path.is_file():
            raise FileNotFoundError(f"Fitted ceiling group PLY is missing: {source_path}")
        group_source = _source_support(source_path, plane)
        support_path = source_path
        source = group_source
        floor_level = group.get("associated_floor_level")
        if floor_level is not None:
            floor_level = int(floor_level)
            if floor_level not in floor_polygons:
                raise ValueError(f"Ceiling group references missing floor {floor_level}")
            support_path = floor_polygons[floor_level]["fitted_path"]
            if not support_path.is_file():
                raise FileNotFoundError(f"Cleaned ceiling support PLY is missing: {support_path}")
            source = _source_support(
                support_path, plane, surface_role=2,
                footprint=group_source["projected_footprint"],
            )
            source["projected_footprint"] = group_source["projected_footprint"]
        record = {
            "group": group, "equation": plane, "source": source,
            "source_path": source_path, "support_path": support_path,
        }
        records.append(record)
        if floor_level is not None:
            groups_by_floor.setdefault(int(floor_level), []).append(record)
        source_info.append({
            "group_id": int(group["group_id"]),
            "associated_floor_level": group.get("associated_floor_level"),
            "source_ceiling_instances": group.get("source_ceiling_instances", []),
            "fitted_source_ply": str(source_path),
            "pattern_support_ply": str(support_path),
            "ceiling_group_support_triangles": group_source["source_triangle_count"],
            "observed_support_triangles": source["source_triangle_count"],
            "downprojected_source_union_area_m2": float(source["projected_footprint"].area),
        })

    levels = []
    for floor_level, members in sorted(groups_by_floor.items()):
        for record in members:
            record["level_key"] = f"floor:{floor_level}"
        levels.append(members)
    orphan_records = [record for record in records
                      if record["group"].get("associated_floor_level") is None]
    for level_index, members in enumerate(_orphan_levels(orphan_records)):
        for record in members:
            record["level_key"] = f"orphan:{level_index}"
        levels.append(members)
    for members in levels:
        for record, support in _supports_for_level(
            members, floor_polygons, wall_mesh, wall_top_equation,
        ):
            patches.append(_fit_group_patches(record, support))

    regularize_planar_patches(patches, max_edge=PATTERN_MAX_EDGE_M)
    for patch in patches:
        _enforce_output_clearance(patch)
        if _is_upper_wall_ceiling(patch):
            _enforce_wall_top_constraint(patch, wall_top_equation)
        else:
            patch["wall_top_vertical_translation_m"] = 0.0
            patch["wall_top_constraint_applied"] = False
    junctions = _vertical_junctions(patches)
    patches.extend(junctions)

    out_vertices, out_faces, out_colors = [], [], []
    face_groups, face_floors, face_areas, face_roles, face_junctions = [], [], [], [], []
    vertex_offset = 0
    output_records = []
    for patch in patches:
        mesh = patch["mesh"]
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        face_colors = np.asarray(mesh.visual.face_colors, dtype=np.uint8)[:, :3]
        if len(face_colors) != len(faces):
            raise ValueError(f"Regularized ceiling patch has no face colors: {patch['name']}")
        # Duplicate triangle vertices so exported per-vertex PLY colors retain hard material boundaries.
        flat_vertices = vertices[faces].reshape(-1, 3)
        flat_faces = np.arange(len(flat_vertices), dtype=np.int32).reshape(-1, 3)
        flat_colors = np.repeat(face_colors, 3, axis=0)
        out_vertices.append(flat_vertices)
        out_faces.append(flat_faces + vertex_offset)
        out_colors.append(flat_colors)
        count = len(flat_faces)
        face_groups.extend([int(patch["group_id"])] * count)
        face_floors.extend([-1 if patch["floor_level"] is None else int(patch["floor_level"])] * count)
        face_areas.extend([-1 if patch["area_id"] is None else int(patch["area_id"])] * count)
        face_roles.extend([2] * count)
        face_junctions.extend([int(patch.get("is_junction", False))] * count)
        vertex_offset += len(flat_vertices)
        if not patch.get("is_junction"):
            output_records.append({
                "group_id": patch["group_id"],
                "associated_floor_level": patch["floor_level"],
                "support_area_id": patch["area_id"],
                "support_source": patch["support_source"],
                "support_height_y_m": patch["support_height_y_m"],
                "support_polygon_area_m2": float(patch["polygon"].area),
                "support_polygon_bounds_xz_m": list(patch["polygon"].bounds),
                "plane_equation": _serialized(patch["plane_equation"]),
                "wall_top_plane_equation": _serialized(patch["wall_top_plane_equation"]),
                "wall_top_clearance_before_m": patch["wall_top_clearance_before_m"],
                "wall_top_max_clearance_before_m": patch["wall_top_max_clearance_before_m"],
                "wall_top_clearance_after_m": patch["wall_top_clearance_after_m"],
                "wall_top_plane_fit": patch["wall_top_plane_fit"],
                "wall_top_vertical_translation_m": patch["wall_top_vertical_translation_m"],
                "wall_top_constraint_applied": patch["wall_top_constraint_applied"],
                "input_perpendicular_gap_m": patch["original_perpendicular_gap_m"],
                "final_perpendicular_gap_m": patch["final_perpendicular_gap_m"],
                "ceiling_translation_m": patch["translation"].tolist(),
                "source_support_triangle_count": patch["source_triangle_count"],
                "output_triangle_count": int(count),
                "pattern_classification": _serialized(patch["pattern_classification"]),
                "color_split": _serialized(patch["color_split"]),
                "regularization": _serialized(patch["regularization"]),
            })

    output_path = output or elaboration_dir / OUTPUT_NAME
    input_paths = {path.resolve() for path in [
        wall_path, groups_path, elaboration_dir / "structural_instances_manifest.json",
        *extracted_wall_paths,
    ] if path.exists()}
    input_paths.update((level_dir / group["fitted_output_file"]).resolve() for group in groups)
    input_paths.update(record["support_path"].resolve() for record in records)
    if output_path.resolve() in input_paths:
        raise ValueError("Ceiling output must not overwrite any input PLY or manifest")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{output_path.name}.", suffix=".tmp",
                                     dir=output_path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        written = export_ply_with_face_attributes(
            temporary_path, np.concatenate(out_vertices), np.concatenate(out_faces),
            np.concatenate(out_colors),
            face_attributes={
                "surface_role": ("uchar", np.asarray(face_roles, dtype=np.uint8)),
                "ceiling_group_id": ("int", np.asarray(face_groups, dtype=np.int32)),
                "associated_floor_level": ("int", np.asarray(face_floors, dtype=np.int32)),
                "support_area_id": ("int", np.asarray(face_areas, dtype=np.int32)),
                "is_vertical_junction": ("uchar", np.asarray(face_junctions, dtype=np.uint8)),
            },
        )
        if not written:
            raise OSError(f"Could not write regularized ceiling PLY: {temporary_path}")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    manifest_path = output or elaboration_dir / MANIFEST_NAME
    if output is not None:
        manifest_path = output.with_name(MANIFEST_NAME)
    manifest = {
        "algorithm": "support_contour_ceiling_fit_above_extracted_wall_top_with_plane_clearance_and_pattern_regularization",
        "source_ceiling_groups_manifest": str(groups_path),
        "wall_support_ply": str(wall_path),
        "wall_top_source_instance_plys": [str(path) for path in extracted_wall_paths],
        "wall_top_contour_plane": wall_top_info,
        "wall_top_constraint_scope": (
            "Applied only to ceiling patches whose maximum fitted height reaches within "
            f"{MAX_WALL_TOP_CONTACT_GAP_M:.2f} m of the upper wall contour; lower paired ceilings remain unchanged."
        ),
        "minimum_associated_floor_clearance_m": MIN_CEILING_CLEARANCE_M,
        "wall_slice_below_lowest_ceiling_m": WALL_SLICE_BELOW_CEILING_M,
        "pattern_frame": "longest boundary edge and its perpendicular tangent on each fitted plane",
        "source_groups": source_info,
        "vertical_junction_count": len(junctions),
        "output_planes": output_records,
        "output_ply": str(output_path),
    }
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                     prefix=f".{manifest_path.name}.", suffix=".tmp",
                                     dir=manifest_path.parent, delete=False) as temporary:
        temporary.write(json.dumps(manifest, indent=2) + "\n")
        temporary_manifest = Path(temporary.name)
    try:
        os.replace(temporary_manifest, manifest_path)
    finally:
        if temporary_manifest.exists():
            temporary_manifest.unlink()
    return {"output_ply": output_path, "manifest": manifest_path, **manifest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="Phase 06 scan workspace")
    parser.add_argument("--output", type=Path,
                        help=f"Output PLY (defaults to mesh_elaboration_steps/{OUTPUT_NAME})")
    parser.add_argument("--wall-support", type=Path,
                        help="Contour-fitted wall PLY for ceiling support slices (defaults to mesh_elaboration_steps/wall_enclosing_planes_fitted_to_regularized_contours.ply); the upper-wall ceiling constraint comes from extracted wall instance PLYs")
    args = parser.parse_args(argv)
    try:
        result = regularize_ceiling_planes(args.workspace, args.output, args.wall_support)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(f"Wrote regularized ceiling planes: {result['output_ply']}")
    print(f"Wrote ceiling regularization manifest: {result['manifest']}")
    print(f"Processed {len(result['output_planes'])} ceiling support patches; "
          f"added {result['vertical_junction_count']} vertical junctions")
    for patch in result["output_planes"]:
        gap = patch["final_perpendicular_gap_m"]
        gap_text = "unassociated" if gap is None else f"{gap:.3f} m"
        print(f"  Group {patch['group_id']}: support={patch['support_source']}, gap={gap_text}, "
              f"pattern={patch['pattern_classification']['pattern_type']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
