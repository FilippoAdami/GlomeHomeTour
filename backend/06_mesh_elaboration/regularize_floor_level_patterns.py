#!/usr/bin/env python3
"""Apply observed patterns and matched textures to the final planar floor polygons."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from collections.abc import Iterator

import numpy as np
import open3d as o3d
import trimesh
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Polygon
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from planar_pattern_regularization import regularize_planar_patches
from structural_geometry import polygon_triangles
from extract_floor_texture_tiles import prepare_floor_texture_tiles
from regularize_floor_texture_tiles import regularize_texture_tiles

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_3DGS_to_mesh"))
from fill_mesh_holes import export_ply_with_face_attributes, read_ply_face_flags

FLOOR_ROLE = 0
TEXTURE_OUTPUT_SAMPLE_M = 0.01
TEXTURE_SUPPORT_NORMAL_DOT_MIN = float(np.cos(np.deg2rad(30.0)))


def _face_property(path: Path, name: str, count: int) -> np.ndarray:
    values = read_ply_face_flags(path, name)
    if values is None or len(values) != count:
        raise ValueError(f"{path} is missing a valid face property {name!r}")
    return np.asarray(values)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _polygon_with_holes(rings: list) -> Polygon:
    ring_polygons = []
    for ring in rings:
        points = np.asarray(ring, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError("Final floor polygon contains an invalid X/Z ring")
        polygon = Polygon(points)
        if not polygon.is_valid or polygon.area <= 1e-8:
            raise ValueError("Final floor polygon contains an invalid or degenerate ring")
        ring_polygons.append(polygon)
    if not ring_polygons:
        raise ValueError("Final floor polygon has no rings")

    parents = []
    for index, polygon in enumerate(ring_polygons):
        containers = [
            other for other, candidate in enumerate(ring_polygons)
            if other != index and candidate.area > polygon.area
            and candidate.contains(polygon.representative_point())
        ]
        parents.append(min(containers, key=lambda other: ring_polygons[other].area)
                       if containers else None)

    depths = []
    for index in range(len(ring_polygons)):
        depth, parent, seen = 0, parents[index], {index}
        while parent is not None:
            if parent in seen:
                raise ValueError("Final floor polygon rings contain a nesting cycle")
            seen.add(parent)
            depth += 1
            parent = parents[parent]
        depths.append(depth)

    components = []
    for index, polygon in enumerate(ring_polygons):
        if depths[index] % 2:
            continue
        holes = [ring_polygons[child].exterior.coords for child, parent in enumerate(parents)
                 if parent == index and depths[child] == depths[index] + 1]
        component = Polygon(polygon.exterior.coords, holes)
        if not component.is_valid or component.area <= 1e-8:
            raise ValueError("Could not assemble final floor polygon rings")
        components.append(component)
    result = unary_union(components)
    if result.is_empty or result.area <= 1e-8:
        raise ValueError("Final floor polygon is empty")
    return result


def _target_mesh(rings: list, equation: np.ndarray) -> trimesh.Trimesh:
    polygon = _polygon_with_holes(rings)
    xz, faces = polygon_triangles(polygon)
    if not len(faces):
        raise ValueError("Could not triangulate final floor polygon")
    normal = equation[:3] / np.linalg.norm(equation[:3])
    offset = float(equation[3] / np.linalg.norm(equation[:3]))
    if abs(normal[1]) < 0.7:
        raise ValueError("Fitted floor plane is too vertical to place a floor polygon")
    vertices = np.column_stack((xz[:, 0],
                                -(normal[0] * xz[:, 0] + normal[2] * xz[:, 1] + offset) / normal[1],
                                xz[:, 1]))
    crosses = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                       vertices[faces[:, 2]] - vertices[faces[:, 0]])
    if np.mean(crosses @ normal) < 0:
        faces = faces[:, ::-1]
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _vertex_colors_from_faces(mesh: trimesh.Trimesh) -> np.ndarray:
    face_colors = np.asarray(mesh.visual.face_colors, dtype=np.float64)[:, :3]
    faces = np.asarray(mesh.faces, dtype=np.int64)
    weights = np.maximum(np.asarray(mesh.area_faces, dtype=np.float64), 1e-12)
    totals = np.zeros((len(mesh.vertices), 3), dtype=np.float64)
    totals_weight = np.zeros(len(mesh.vertices), dtype=np.float64)
    for corner in range(3):
        np.add.at(totals, faces[:, corner], face_colors * weights[:, None])
        np.add.at(totals_weight, faces[:, corner], weights)
    totals /= np.maximum(totals_weight[:, None], 1e-12)
    return np.rint(np.clip(totals, 0, 255)).astype(np.uint8)


def _texture_context(candidates: list[dict], duplicates: list[dict]) -> dict:
    contexts = {}

    def add(candidate: dict, tile_name: str) -> None:
        key = (candidate["level"], candidate["area"])
        area_info = candidate["area_info"]
        context = contexts.setdefault(key, {
            "area_info": area_info,
            "texture_regions": {},
        })
        context["texture_regions"][candidate["region"]] = {
            "tile_name": tile_name,
            "tile_origin_uv": candidate["tile_origin_uv"],
        }

    for candidate in candidates:
        add(candidate, candidate["name"])
    for duplicate in duplicates:
        add(duplicate["removed"], duplicate["kept"]["name"])

    for context in contexts.values():
        labels = np.asarray(context["area_info"]["labels"], dtype=np.int32)
        if not np.any(labels):
            raise ValueError("Texture region map contains no labeled floor support")
        nearest = distance_transform_edt(labels == 0, return_distances=False,
                                         return_indices=True)
        filled = labels.copy()
        missing = filled == 0
        filled[missing] = labels[tuple(nearest[:, missing])]
        context["filled_labels"] = filled
    return contexts


def _labels_for_uv(area_info: dict, filled_labels: np.ndarray,
                   uv: np.ndarray) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    lower = np.asarray(area_info["raster_lower_uv"], dtype=np.float64)
    cell = float(area_info["raster_cell_size_m"])
    x = np.floor((uv[:, 0] - lower[0]) / cell).astype(np.int64)
    y_up = np.floor((uv[:, 1] - lower[1]) / cell).astype(np.int64)
    x = np.clip(x, 0, filled_labels.shape[1] - 1)
    y = np.clip(filled_labels.shape[0] - 1 - y_up, 0, filled_labels.shape[0] - 1)
    return filled_labels[y, x]


def _sample_periodic_tile(tile: np.ndarray, uv: np.ndarray, origin: np.ndarray,
                          cell_size: float) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    tile = np.asarray(tile, dtype=np.float32)
    x = np.mod((uv[:, 0] - origin[0]) / cell_size, tile.shape[1])
    y = np.mod((origin[1] - uv[:, 1]) / cell_size, tile.shape[0])
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    dx, dy = (x - x0)[:, None], (y - y0)[:, None]
    c00 = tile[y0 % tile.shape[0], x0 % tile.shape[1]]
    c10 = tile[y0 % tile.shape[0], (x0 + 1) % tile.shape[1]]
    c01 = tile[(y0 + 1) % tile.shape[0], x0 % tile.shape[1]]
    c11 = tile[(y0 + 1) % tile.shape[0], (x0 + 1) % tile.shape[1]]
    return ((1 - dx) * (1 - dy) * c00 + dx * (1 - dy) * c10
            + (1 - dx) * dy * c01 + dx * dy * c11)


def _subdivide_and_apply_textures(mesh: trimesh.Trimesh, context: dict,
                                 tiles: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    dense_vertices, dense_faces, parent_faces = trimesh.remesh.subdivide_to_size(
        vertices, faces, max_edge=TEXTURE_OUTPUT_SAMPLE_M, max_iter=16, return_index=True
    )
    triangles = dense_vertices[dense_faces]
    centers = triangles.mean(axis=1)
    area_info = context["area_info"]
    axis_u = np.asarray(area_info["axis_u"], dtype=np.float64)
    axis_v = np.asarray(area_info["axis_v"], dtype=np.float64)
    center_uv = np.column_stack((centers @ axis_u, centers @ axis_v))
    region_ids = _labels_for_uv(area_info, context["filled_labels"], center_uv)

    parent_colors = np.asarray(mesh.visual.face_colors, dtype=np.uint8)[parent_faces, :3]
    corner_colors = np.repeat(parent_colors[:, None, :], 3, axis=1).astype(np.float32) / 255.0
    cell_size = float(area_info["raster_cell_size_m"])
    texture_triangle_counts = {}
    for region_id, spec in context["texture_regions"].items():
        tile_name = spec["tile_name"]
        if tile_name not in tiles:
            raise FileNotFoundError(f"Seam-regularized texture tile is missing: {tile_name}")
        selected = np.flatnonzero(region_ids == int(region_id))
        if not len(selected):
            continue
        coordinates = triangles[selected]
        vertex_uv = np.stack((coordinates @ axis_u, coordinates @ axis_v), axis=-1).reshape(-1, 2)
        samples = _sample_periodic_tile(
            tiles[tile_name], vertex_uv,
            np.asarray(spec["tile_origin_uv"], dtype=np.float64), cell_size,
        )
        corner_colors[selected] = samples.reshape(-1, 3, 3)
        texture_triangle_counts[tile_name] = texture_triangle_counts.get(tile_name, 0) + int(len(selected))

    flat_vertices = triangles.reshape(-1, 3)
    flat_faces = np.arange(len(flat_vertices), dtype=np.int32).reshape(-1, 3)
    flat_colors = np.rint(np.clip(corner_colors.reshape(-1, 3), 0.0, 1.0) * 255).astype(np.uint8)
    return flat_vertices, flat_faces, flat_colors, texture_triangle_counts


def _regularize_level(level_dir: Path, level: dict, texture_contexts: dict,
                      texture_tiles: dict[str, np.ndarray]) -> dict:
    level_id = level.get("level")
    source_name = level.get("fitted_output_file")
    if (not isinstance(level_id, int) or level_id < 0 or not isinstance(source_name, str)
            or source_name != f"floor_level_{level_id:03d}_fitted.ply"):
        raise ValueError(f"Invalid fitted floor entry in level manifest: {level!r}")
    source_path = level_dir / source_name
    polygons_path = level_dir / f"floor_level_{level_id:03d}_polygons.json"
    if not polygons_path.is_file():
        raise FileNotFoundError(f"Final floor polygons are missing: {polygons_path}")
    final_polygons = json.loads(polygons_path.read_text(encoding="utf-8"))
    if final_polygons.get("level") != level_id or not isinstance(final_polygons.get("areas"), list):
        raise ValueError(f"Invalid final polygon manifest: {polygons_path}")

    source_mesh = o3d.io.read_triangle_mesh(str(source_path))
    source_vertices = np.asarray(source_mesh.vertices, dtype=np.float64)
    source_faces = np.asarray(source_mesh.triangles, dtype=np.int64)
    if not len(source_vertices) or not len(source_faces) or not np.isfinite(source_vertices).all():
        raise ValueError(f"Fitted floor support PLY is empty or invalid: {source_path}")
    if source_faces.min() < 0 or source_faces.max() >= len(source_vertices):
        raise ValueError(f"Fitted floor support PLY has invalid triangle indices: {source_path}")
    roles = _face_property(source_path, "surface_role", len(source_faces))
    previews = _face_property(source_path, "is_fitted_plane", len(source_faces))
    filled = _face_property(source_path, "is_hole_fill", len(source_faces))
    area_ids = _face_property(source_path, "floor_area_id", len(source_faces))
    observed_floor_faces = np.flatnonzero(
        (roles == FLOOR_ROLE) & (previews == 0) & (filled == 0)
    )
    if not len(observed_floor_faces):
        raise ValueError(f"No observed floor-role triangles available as pattern/texture support: {source_path}")

    source_colors = np.asarray(source_mesh.vertex_colors, dtype=np.float64)
    colors_available = len(source_colors) == len(source_vertices)
    if not colors_available:
        raise ValueError(f"Observed floor support has no vertex colors: {source_path}")
    source_colors = np.clip(source_colors, 0.0, 1.0)

    floor_areas = level.get("floor_areas")
    if not isinstance(floor_areas, list) or not floor_areas:
        raise ValueError(f"Level {level_id} has no fitted floor area metadata")
    fitted_by_id = {}
    for area in floor_areas:
        if not isinstance(area, dict) or not isinstance(area.get("area_id"), int):
            raise ValueError(f"Invalid fitted floor area metadata for level {level_id}")
        fitted_by_id[area["area_id"]] = area
    if not set(area_ids[observed_floor_faces].tolist()).issubset(fitted_by_id):
        raise ValueError(f"Some observed floor triangles in {source_path} have no fitted plane")

    patches, source_count_by_area = [], {}
    output_area_ids = []
    for final_area in final_polygons["areas"]:
        area_id = final_area.get("area_id")
        if not isinstance(area_id, int) or area_id not in fitted_by_id:
            raise ValueError(f"Final polygon references an unknown floor area: {final_area!r}")
        fitted_area = fitted_by_id[area_id]
        equation = np.asarray(fitted_area.get("plane_equation"), dtype=np.float64)
        if equation.shape != (4,) or not np.isfinite(equation).all() or np.linalg.norm(equation[:3]) <= 1e-10:
            raise ValueError(f"Invalid fitted plane for level {level_id}, area {area_id}")
        support_face_ids = observed_floor_faces[area_ids[observed_floor_faces] == area_id]
        if not len(support_face_ids):
            raise ValueError(f"No observed floor support for level {level_id}, area {area_id}")
        support_triangles = source_faces[support_face_ids]
        support_vertices = source_vertices[support_triangles]
        crosses = np.cross(support_vertices[:, 1] - support_vertices[:, 0],
                           support_vertices[:, 2] - support_vertices[:, 0])
        double_areas = np.linalg.norm(crosses, axis=1)
        plane_normal = equation[:3] / np.linalg.norm(equation[:3])
        valid = (double_areas > 1e-12) & (
            np.abs(crosses @ plane_normal) >= TEXTURE_SUPPORT_NORMAL_DOT_MIN * double_areas
        )
        support_triangles = support_triangles[valid]
        support_vertices = support_vertices[valid]
        double_areas = double_areas[valid]
        if not len(support_triangles):
            raise ValueError(f"Floor support is degenerate for level {level_id}, area {area_id}")

        target = _target_mesh(final_area.get("floor_polygon_rings_world_xz_m", []), equation)
        patches.append({
            "name": f"floor_level_{level_id:03d}_area_{area_id:03d}",
            "area_id": area_id,
            "mesh": target,
            "plane_equation": equation.tolist(),
            "relief_direction": [0.0, 1.0, 0.0],
            "support_centers": support_vertices.mean(axis=1),
            "support_areas": double_areas * 0.5,
            "support_colors": source_colors[support_triangles].mean(axis=1),
        })
        source_count_by_area[str(area_id)] = int(len(support_triangles))

    if not patches:
        raise ValueError(f"No final floor polygons found for level {level_id}")
    regularize_planar_patches(patches)

    output_vertices, output_faces, output_colors, output_roles = [], [], [], []
    offset = 0
    for patch in patches:
        target = patch["mesh"]
        equation = np.asarray(patch["plane_equation"], dtype=np.float64)
        normal = equation[:3] / np.linalg.norm(equation[:3])
        offset_plane = float(equation[3] / np.linalg.norm(equation[:3]))
        vertices = np.asarray(target.vertices, dtype=np.float64).copy()
        faces = np.asarray(target.faces, dtype=np.int32)
        texture_context = texture_contexts.get((level_id, int(patch["area_id"])))
        if texture_context is not None:
            textured_vertices, textured_faces, textured_colors, texture_counts = _subdivide_and_apply_textures(
                target, texture_context, texture_tiles
            )
            patch["applied_texture_triangle_counts"] = texture_counts
            if texture_counts:
                vertices, faces, colors = textured_vertices, textured_faces, textured_colors
            else:
                colors = _vertex_colors_from_faces(target)
        else:
            colors = _vertex_colors_from_faces(target)
            patch["applied_texture_triangle_counts"] = {}
        output_vertices.append(vertices)
        output_faces.append(faces + offset)
        output_colors.append(colors)
        output_roles.append(np.full(len(faces), FLOOR_ROLE, dtype=np.uint8))
        output_area_ids.append(np.full(len(faces), patch["area_id"], dtype=np.int32))
        offset += len(vertices)
        patch["output_planarity_max_error_m"] = float(np.max(np.abs(vertices @ normal + offset_plane)))

    return {
        "level": level_id,
        "support_source_file": source_name,
        "final_polygon_source_file": polygons_path.name,
        "output_file": f"floor_level_{level_id:03d}_patterns_regularized.ply",
        "observed_floor_support_triangle_count": int(sum(source_count_by_area.values())),
        "observed_floor_support_triangle_count_by_area": source_count_by_area,
        "source_vertex_colors_available": colors_available,
        "output_geometry": "final floor polygon with detected relief raised on its upward side",
        "output_color_encoding": "per-vertex RGB",
        "output_vertex_count": int(sum(len(vertices) for vertices in output_vertices)),
        "output_triangle_count": int(sum(len(faces) for faces in output_faces)),
        "_vertices": np.concatenate(output_vertices),
        "_faces": np.concatenate(output_faces),
        "_colors": np.concatenate(output_colors),
        "_surface_roles": np.concatenate(output_roles),
        "_floor_area_ids": np.concatenate(output_area_ids),
        "areas": [{
            "area_id": int(patch["area_id"]),
            "pattern_classification": _json_safe(patch["pattern_classification"]),
            "color_split": _json_safe(patch["color_split"]),
            "regularization": _json_safe(patch["regularization"]),
            "applied_texture_triangle_counts": patch["applied_texture_triangle_counts"],
            "output_planarity_max_error_m": patch["output_planarity_max_error_m"],
        } for patch in patches],
    }


def regularize_floor_level_patterns(instances_dir: Path) -> Iterator[dict]:
    level_dir = instances_dir / "intermediate_floor_levels"
    manifest_path = level_dir / "floor_level_intermediate_manifest.json"
    candidates, duplicates = prepare_floor_texture_tiles(instances_dir)
    texture_contexts = _texture_context(candidates, duplicates)
    texture_tiles = regularize_texture_tiles(candidates)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    levels = manifest.get("levels") if isinstance(manifest, dict) else None
    if not isinstance(levels, list) or not levels:
        raise ValueError(f"No fitted floor levels found in {manifest_path}")

    for level in levels:
        yield _regularize_level(level_dir, level, texture_contexts, texture_tiles)


def _write_regularized_floor_ply(level_dir: Path, result: dict) -> Path:
    output_path = level_dir / result["output_file"]
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    written = export_ply_with_face_attributes(
        temporary_path, result["_vertices"], result["_faces"], result["_colors"],
        face_attributes={
            "surface_role": ("uchar", result["_surface_roles"]),
            "floor_area_id": ("int", result["_floor_area_ids"]),
        },
    )
    if not written:
        raise OSError(f"Could not write regularized floor PLY: {temporary_path}")
    os.replace(temporary_path, output_path)
    return output_path


def _clear_texture_debug_outputs(level_dir: Path) -> None:
    for stale in (level_dir / "floor_texture_tiles",
                  level_dir / "floor_texture_tiles_3x3",
                  level_dir / "floor_level_patterns_regularization_manifest.json"):
        if stale.is_dir():
            shutil.rmtree(stale)
        elif stale.exists():
            stale.unlink()


def _print_results(results: list[dict]) -> None:
    for result in results:
        print(f"Level {result['level']:03d}: applied patterns to {result['output_triangle_count']:,} final polygon triangles "
              f"using {result['observed_floor_support_triangle_count']:,} observed floor support triangles; "
              f"repeated texture on {sum(sum(area['applied_texture_triangle_counts'].values())
                                           for area in result['areas']):,} triangles -> "
              f"{result['output_file']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path,
                        default=Path("backend/current_scene/scan_0/mesh_elaboration"))
    args = parser.parse_args()
    level_dir = args.instances_dir / "intermediate_floor_levels"
    _clear_texture_debug_outputs(level_dir)

    output_names = set()
    for result in regularize_floor_level_patterns(args.instances_dir):
        _write_regularized_floor_ply(level_dir, result)
        output_names.add(result["output_file"])
        _print_results([result])
    for path in level_dir.glob("floor_level_[0-9][0-9][0-9]_patterns_regularized.ply"):
        if path.name not in output_names:
            path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
