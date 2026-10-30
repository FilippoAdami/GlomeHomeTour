#!/usr/bin/env python3
"""Fit enclosing walls to Step 3 contours, retaining colors and transition ledges."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from planar_pattern_regularization import regularize_planar_patches
from structural_geometry import polygon_triangles
from surface_footprints import regularize_orthogonal_footprint
from mesh_elaboration_inputs import resolve_elaboration_output_dir

OUTPUT_NAME = "wall_enclosing_planes_fitted_to_regularized_contours.ply"
HORIZONTAL_NORMAL_DOT = 0.99
PLANE_TOLERANCE_M = 1e-5
HEIGHT_TOLERANCE_M = 1e-4
FLOOR_TO_WALL_BOUNDARY_TOLERANCE_M = 0.10


def _horizontal_bands(mesh: o3d.geometry.TriangleMesh) -> tuple[list[tuple[float, float]], list[float]]:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if not len(vertices) or not len(triangles):
        raise ValueError("The enclosing wall PLY is empty")

    corners = vertices[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    flat = np.ptp(corners[:, :, 1], axis=1) <= PLANE_TOLERANCE_M
    horizontal = flat & (lengths > 1e-12) & (np.abs(normals[:, 1]) >= HORIZONTAL_NORMAL_DOT * lengths)
    transition_heights = sorted(float(y) for y in corners[horizontal, :, 1].mean(axis=1))
    unique_heights = []
    for y in transition_heights:
        if not unique_heights or y - unique_heights[-1] > HEIGHT_TOLERANCE_M:
            unique_heights.append(y)

    min_y, max_y = map(float, (vertices[:, 1].min(), vertices[:, 1].max()))
    interior = [y for y in unique_heights if min_y + HEIGHT_TOLERANCE_M < y < max_y - HEIGHT_TOLERANCE_M]
    boundaries = [min_y, *interior, max_y]
    return list(zip(boundaries, boundaries[1:])), interior


def _load_floors(elaboration_dir: Path) -> list[dict]:
    levels_path = elaboration_dir / "intermediate_floor_levels" / "floor_level_intermediate_manifest.json"
    levels_data = json.loads(levels_path.read_text())
    elevations = {
        int(level["level"]): float(level["elevation_y_median"])
        for level in levels_data["levels"]
    }
    floors = []
    for path in sorted((elaboration_dir / "intermediate_floor_levels").glob("floor_level_*_polygons.json")):
        match = re.fullmatch(r"floor_level_(\d+)_polygons\.json", path.name)
        if match is None:
            continue
        level = int(match.group(1))
        if level not in elevations:
            raise ValueError(f"Floor polygon file has no fitted elevation: {path}")
        data = json.loads(path.read_text())
        areas = []
        for area in data.get("areas", []):
            contour = np.asarray(area.get("regularized_wall_contour_world_xz_m", []), dtype=float)
            if contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 3 or not np.isfinite(contour).all():
                raise ValueError(f"Invalid regularized wall contour in {path}, area {area.get('area_id')}")
            footprint = regularize_orthogonal_footprint(
                Polygon(contour), min_edge=0.10, tolerance=0.02
            )
            if footprint.geom_type != "Polygon" or footprint.interiors:
                raise ValueError(f"Wall contour in {path}, area {area.get('area_id')} is not one simple polygon")
            contour = np.asarray(footprint.exterior.coords[:-1], dtype=float)
            areas.append({"area_id": area.get("area_id"), "contour": contour})
        if not areas:
            raise ValueError(f"Floor has no regularized wall contours: {path}")
        floors.append({"level": level, "floor_y_m": elevations[level], "areas": areas, "path": path})
    if not floors:
        raise ValueError(f"No Step 3 floor polygon files found in {elaboration_dir / 'intermediate_floor_levels'}")
    return sorted(floors, key=lambda floor: floor["floor_y_m"])


def _horizontal_transition_patches(
    lower: Polygon,
    upper: Polygon,
    y: float,
    source_centers: np.ndarray,
    source_colors: np.ndarray,
    source_horizontal: np.ndarray,
) -> list[dict]:
    source_centers = source_centers[source_horizontal]
    source_colors = source_colors[source_horizontal]
    color_tree = cKDTree(source_centers[:, [0, 2]]) if len(source_centers) else None
    patches = []
    for footprint, normal_y in ((lower.difference(upper), 1.0),
                                 (upper.difference(lower), -1.0)):
        polygons = ([footprint] if footprint.geom_type == "Polygon"
                    else [part for part in getattr(footprint, "geoms", [])
                          if part.geom_type == "Polygon"])
        for polygon in polygons:
            if polygon.is_empty or polygon.area <= 1e-8:
                continue
            vertices_xz, faces = polygon_triangles(polygon)
            vertices = np.column_stack((vertices_xz[:, 0], np.full(len(vertices_xz), y), vertices_xz[:, 1]))
            corners = vertices[faces]
            signed_xz = ((corners[:, 1, 0] - corners[:, 0, 0])
                         * (corners[:, 2, 2] - corners[:, 0, 2])
                         - (corners[:, 1, 2] - corners[:, 0, 2])
                         * (corners[:, 2, 0] - corners[:, 0, 0]))
            valid_faces = np.abs(signed_xz) > 1e-10
            faces, signed_xz = faces[valid_faces], signed_xz[valid_faces]
            reverse = signed_xz * normal_y > 0
            faces[reverse] = faces[reverse][:, [0, 2, 1]]
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            centers = np.asarray(mesh.triangles_center)
            face_colors = (source_colors[color_tree.query(centers[:, [0, 2]], k=1)[1]]
                           if color_tree is not None else np.full((len(faces), 3), 0.65))
            patches.append({"mesh": mesh, "face_colors": face_colors})
    return patches


def _conform_transition_edges(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    heights: list[float],
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split transition edges at existing mesh vertices to remove T-junctions."""
    transition_points = [
        np.unique(vertices[np.abs(vertices[:, 1] - height) <= 1e-5], axis=0)
        for height in heights
    ]
    out_vertices, out_faces, out_colors = [], [], []
    for face, face_colors in zip(faces, colors[faces]):
        pieces = [(vertices[face].copy(), face_colors.copy())]
        for height, points in zip(heights, transition_points):
            for first, second in ((0, 1), (1, 2), (2, 0)):
                start, end = vertices[face[[first, second]]]
                if abs(start[1] - height) > 1e-5 or abs(end[1] - height) > 1e-5:
                    continue
                delta = end[[0, 2]] - start[[0, 2]]
                length2 = float(delta @ delta)
                if length2 <= tolerance * tolerance:
                    continue
                relative = points[:, [0, 2]] - start[[0, 2]]
                fractions = relative @ delta / length2
                projected = start[[0, 2]] + fractions[:, None] * delta
                distance = np.linalg.norm(points[:, [0, 2]] - projected, axis=1)
                mask = ((fractions > tolerance) & (fractions < 1.0 - tolerance)
                        & (distance <= tolerance))
                candidates = points[mask]
                order = np.argsort(fractions[mask])
                for point in candidates[order]:
                    split = []
                    for piece_vertices, piece_colors in pieces:
                        did_split = False
                        for i, j, opposite in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                            edge = piece_vertices[j] - piece_vertices[i]
                            edge_length2 = float(edge @ edge)
                            if edge_length2 <= tolerance * tolerance:
                                continue
                            fraction = float((point - piece_vertices[i]) @ edge / edge_length2)
                            projection = piece_vertices[i] + fraction * edge
                            if (tolerance < fraction < 1.0 - tolerance
                                    and np.linalg.norm(point - projection) <= tolerance):
                                point_color = piece_colors[i] + fraction * (piece_colors[j] - piece_colors[i])
                                split.extend((
                                    (np.array([piece_vertices[i], point, piece_vertices[opposite]]),
                                     np.array([piece_colors[i], point_color, piece_colors[opposite]])),
                                    (np.array([point, piece_vertices[j], piece_vertices[opposite]]),
                                     np.array([point_color, piece_colors[j], piece_colors[opposite]])),
                                ))
                                did_split = True
                                break
                        if not did_split:
                            split.append((piece_vertices, piece_colors))
                    pieces = split
        for piece_vertices, piece_colors in pieces:
            start_index = len(out_vertices)
            out_vertices.extend(piece_vertices)
            out_colors.extend(piece_colors)
            out_faces.append([start_index, start_index + 1, start_index + 2])
    return (np.asarray(out_vertices, dtype=float), np.asarray(out_faces, dtype=np.int32),
            np.asarray(out_colors, dtype=float))


def _fit_panels(
    source_mesh: o3d.geometry.TriangleMesh,
    floors: list[dict],
    bands: list[tuple[float, float]],
) -> tuple[o3d.geometry.TriangleMesh, list[dict]]:
    if len(floors) != len(bands):
        raise ValueError(
            f"Found {len(bands)} enclosing-wall height bands for {len(floors)} floors; "
            "each band must contain one floor contour"
        )
    source_vertices = np.asarray(source_mesh.vertices)
    source_colors = np.asarray(source_mesh.vertex_colors)
    use_colors = len(source_colors) == len(source_vertices)
    source_triangles = np.asarray(source_mesh.triangles)
    source_corners = source_vertices[source_triangles]
    source_normals = np.cross(source_corners[:, 1] - source_corners[:, 0],
                              source_corners[:, 2] - source_corners[:, 0])
    source_areas = np.linalg.norm(source_normals, axis=1) * 0.5
    valid_source = source_areas > 1e-12
    source_normals[valid_source] /= (source_areas[valid_source] * 2.0)[:, None]
    source_centers = source_corners.mean(axis=1)
    source_support_colors = (source_colors[source_triangles].mean(axis=1) if use_colors
                             else np.full((len(source_triangles), 3), 0.65))
    patches, assignments = [], []
    footprints = []
    assigned_levels = set()
    for band_index, (y0, y1) in enumerate(bands):
        if y1 - y0 <= HEIGHT_TOLERANCE_M:
            raise ValueError(f"Invalid enclosing-wall height band: [{y0}, {y1}]")
        if band_index + 1 < len(bands):
            candidates = [floor for floor in floors
                          if y0 - FLOOR_TO_WALL_BOUNDARY_TOLERANCE_M <= floor["floor_y_m"]
                          < y1 - HEIGHT_TOLERANCE_M / 2]
        else:
            candidates = [floor for floor in floors
                          if y0 - FLOOR_TO_WALL_BOUNDARY_TOLERANCE_M <= floor["floor_y_m"]
                          <= y1 + HEIGHT_TOLERANCE_M]
        if len(candidates) != 1:
            raise ValueError(f"Height band [{y0:.4f}, {y1:.4f}] contains {len(candidates)} fitted floors")
        floor = candidates[0]
        assigned_levels.add(floor["level"])
        footprints.append(unary_union([Polygon(area["contour"]) for area in floor["areas"]]))
        edge_count = 0
        for area_index, area in enumerate(floor["areas"]):
            contour = area["contour"]
            signed_area = float(np.sum(contour[:, 0] * np.roll(contour[:, 1], -1)
                                       - np.roll(contour[:, 0], -1) * contour[:, 1]))
            for edge_index, (first, second) in enumerate(zip(contour, np.roll(contour, -1, axis=0))):
                panel = np.asarray([
                    [first[0], y0, first[1]], [second[0], y0, second[1]],
                    [second[0], y1, second[1]], [first[0], y1, first[1]],
                ])
                if signed_area > 0:
                    cell_faces = [[0, 2, 1], [0, 3, 2]]
                else:
                    cell_faces = [[0, 1, 2], [0, 2, 3]]
                mesh = trimesh.Trimesh(vertices=panel, faces=cell_faces, process=False)
                edge = np.array([second[0] - first[0], 0.0, second[1] - first[1]])
                edge_length = float(np.linalg.norm(edge))
                tangent = edge / max(edge_length, 1e-12)
                normal = np.array(mesh.face_normals[0], dtype=float)
                normal /= max(float(np.linalg.norm(normal)), 1e-12)
                inside = np.array([-edge[2], 0.0, edge[0]]) * (1.0 if signed_area > 0 else -1.0)
                distances = (source_centers - panel[0]) @ normal
                along = (source_centers - panel[0]) @ tangent
                supported = (
                    valid_source
                    & (np.abs(source_normals @ normal) >= 0.5)
                    & (np.abs(distances) <= 0.30)
                    & (along >= -0.05) & (along <= edge_length + 0.05)
                    & (source_centers[:, 1] >= y0 - 0.05)
                    & (source_centers[:, 1] <= y1 + 0.05)
                )
                support_centers = (
                    source_centers[supported] - np.median(distances[supported]) * normal
                    if np.any(supported) else np.empty((0, 3))
                )
                patches.append({
                    "name": f"wall_level_{floor['level']}_area_{area_index}_edge_{edge_index}",
                    "mesh": mesh,
                    "plane_equation": [*normal, -float(normal @ panel[0])],
                    "relief_direction": inside,
                    "support_centers": support_centers,
                    "support_areas": source_areas[supported],
                    "support_colors": source_support_colors[supported],
                })
                edge_count += 1
        assignments.append({
            "level": floor["level"], "floor_y_m": floor["floor_y_m"],
            "height_band_y_m": [y0, y1], "contour_area_count": len(floor["areas"]),
            "wall_plane_count": edge_count,
        })
    if assigned_levels != {floor["level"] for floor in floors}:
        raise ValueError("Not every fitted floor was assigned to an enclosing-wall height band")

    horizontal_source = (
        (np.ptp(source_corners[:, :, 1], axis=1) <= PLANE_TOLERANCE_M)
        & (np.abs(source_normals[:, 1]) >= HORIZONTAL_NORMAL_DOT)
        & valid_source
    )
    transitions = []
    for index, y in enumerate(bands[:-1]):
        transition_y = y[1]
        source_at_height = horizontal_source & (np.abs(source_centers[:, 1] - transition_y) <= HEIGHT_TOLERANCE_M)
        transitions.extend(_horizontal_transition_patches(
            footprints[index], footprints[index + 1], transition_y,
            source_centers, source_support_colors, source_at_height,
        ))

    regularize_planar_patches(patches)
    vertices, triangles, colors = [], [], []
    colored_meshes = [
        (patch["mesh"], np.asarray(patch["mesh"].visual.face_colors)[:, :3] / 255.0)
        for patch in patches
    ]
    colored_meshes.extend((patch["mesh"], patch["face_colors"]) for patch in transitions)
    for mesh, face_colors in colored_meshes:
        faces = np.asarray(mesh.faces)
        start = len(vertices)
        vertices.extend(np.asarray(mesh.vertices)[faces].reshape(-1, 3))
        triangles.extend((np.arange(start, start + len(faces) * 3).reshape(-1, 3)).tolist())
        colors.extend(np.repeat(face_colors, 3, axis=0))
    vertices, triangles, colors = _conform_transition_edges(
        np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=np.int32),
        np.asarray(colors, dtype=float), [band[1] for band in bands[:-1]],
    )
    result = o3d.geometry.TriangleMesh()
    result.vertices = o3d.utility.Vector3dVector(vertices)
    result.triangles = o3d.utility.Vector3iVector(triangles)
    result.vertex_colors = o3d.utility.Vector3dVector(colors)
    return result, assignments


def fit_enclosing_walls_to_regularized_contours(workspace: Path, output: Path | None = None) -> dict:
    elaboration_dir = resolve_elaboration_output_dir(workspace)
    source_path = elaboration_dir / "wall_enclosing_shell.ply"
    manifest_path = elaboration_dir / "wall_planes_manifest.json"
    if not source_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Run Phase 06 Step 2 first; expected {source_path} and {manifest_path}")
    source_manifest = json.loads(manifest_path.read_text())
    if not source_manifest.get("wall_planes"):
        raise ValueError(f"No fitted enclosing wall planes in {manifest_path}")
    source_mesh = o3d.io.read_triangle_mesh(str(source_path))
    bands, transition_heights = _horizontal_bands(source_mesh)
    floors = _load_floors(elaboration_dir)
    fitted, assignments = _fit_panels(source_mesh, floors, bands)

    output_path = output or elaboration_dir / OUTPUT_NAME
    if output_path.resolve() == source_path.resolve():
        raise ValueError("Fitted wall planes must not overwrite the source enclosing wall PLY")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output_path), fitted, write_vertex_colors=True):
        raise OSError(f"Could not write fitted wall planes: {output_path}")
    return {
        "source_ply": str(source_path), "output_ply": str(output_path),
        "transition_heights_y_m": transition_heights, "floors": assignments,
        "plane_count": sum(floor["wall_plane_count"] for floor in assignments),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="Scan workspace produced by Phase 06 Steps 1–3")
    parser.add_argument("--output", type=Path,
                        help=f"Output PLY (defaults to mesh_elaboration_steps/{OUTPUT_NAME})")
    args = parser.parse_args(argv)
    try:
        result = fit_enclosing_walls_to_regularized_contours(args.workspace, args.output)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    for floor in result["floors"]:
        print(
            f"Floor {floor['level']}: {floor['wall_plane_count']} planes over "
            f"Y=[{floor['height_band_y_m'][0]:.4f}, {floor['height_band_y_m'][1]:.4f}] m"
        )
    print(f"Transition heights from source wall PLY: {result['transition_heights_y_m']}")
    print(f"Wrote fitted planes: {result['output_ply']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
