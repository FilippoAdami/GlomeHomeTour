"""Remove retained outliers and insert silver panels into aligned object and sealed meshes."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import plyfile
import shapely
import trimesh
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.ops import triangulate, unary_union

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import step_classify_red_regions as classification
import step_red_region_contours as contours
from scan_loader import ScanDataset

LOGGER = logging.getLogger("phase07_region_replacement")
SEALED_PLANE_DISTANCE_LIMIT_M = contours.PLANE_DISTANCE_LIMIT_M
SEALED_NORMAL_ALIGNMENT = 0.75
MIRROR_INSET_M = 0.01
AREA_EPSILON_M2 = 1e-10
SILVER_RGBA = np.array([192, 192, 192, 255], dtype=np.uint8)


def _mesh(path: Path) -> trimesh.Trimesh:
    result = trimesh.load(path, process=False)
    if not isinstance(result, trimesh.Trimesh) or not len(result.faces):
        raise ValueError(f"Expected a non-empty triangular mesh: {path}")
    return result


def _vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    colors = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)
    if colors.shape != (len(mesh.vertices), 4):
        raise ValueError("Expected RGBA vertex colors on every aligned mesh")
    return colors


def _plane_corners(bbox: tuple[float, float, float, float], plane: dict[str, Any],
                   displacement: np.ndarray | None = None) -> np.ndarray:
    x0, x1, y0, y1 = bbox
    origin, u, v = contours._projector(plane["normal"], plane["offset"])
    uv = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    corners = origin + uv[:, 0, None] * u + uv[:, 1, None] * v
    if displacement is not None:
        corners += displacement
    return corners


def _outward_direction(plane: dict[str, Any], surface: dict[str, Any]) -> np.ndarray:
    axis = int(np.argmax(np.abs(plane["normal"])))
    keys = contours._plane_keys(surface["triangles"])
    matches = np.fromiter((key == plane["key"] for key in keys), dtype=bool, count=len(keys))
    matches &= surface["axes"] == axis
    signs = surface["signs"][matches]
    if not len(signs) or np.median(signs) == 0:
        raise ValueError(f"Could not determine the interior side of plane {plane['key']}")
    direction = np.zeros(3, dtype=np.float64)
    direction[axis] = int(np.sign(np.median(signs)))
    return direction


def _polygon_parts(geometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, (MultiPolygon, GeometryCollection)):
        return [part for part in geometry.geoms if isinstance(part, Polygon) and not part.is_empty]
    return []


def _triangle_fragments(outside, triangle_2d: np.ndarray, triangle_3d: np.ndarray,
                        triangle_colors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_2d = []
    points_3d = []
    colors = []
    faces = []
    matrix = np.column_stack((triangle_2d[1] - triangle_2d[0], triangle_2d[2] - triangle_2d[0]))
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-14:
        raise ValueError("Cannot split a degenerate projected mesh triangle")
    for part in _polygon_parts(outside):
        generated = []
        generated_area = 0.0
        for candidate in triangulate(part):
            if candidate.area <= AREA_EPSILON_M2 or not part.covers(candidate):
                continue
            vertices_2d = np.asarray(candidate.exterior.coords[:3], dtype=np.float64)
            weights_12 = np.linalg.solve(matrix, (vertices_2d - triangle_2d[0]).T).T
            weights = np.column_stack((1.0 - weights_12.sum(axis=1), weights_12))
            vertices_3d = weights @ triangle_3d
            vertex_colors = np.clip(weights @ triangle_colors.astype(np.float64), 0, 255).round().astype(np.uint8)
            start = len(points_3d) * 3
            points_3d.append(vertices_3d)
            colors.append(vertex_colors)
            generated.append(np.arange(start, start + 3, dtype=np.int64))
            generated_area += float(candidate.area)
        if generated:
            faces.extend(generated)
        if not np.isclose(generated_area, part.area, rtol=1e-7, atol=AREA_EPSILON_M2):
            raise AssertionError("Triangulated clipped mesh fragment changed its projected area")
    if not points_3d:
        return np.empty((0, 3)), np.empty((0, 4), dtype=np.uint8), np.empty((0, 3), dtype=np.int64)
    return np.concatenate(points_3d), np.concatenate(colors), np.asarray(faces, dtype=np.int64)


def _subtract_bbox(mesh: trimesh.Trimesh, plane: dict[str, Any],
                   bbox: tuple[float, float, float, float], distance_limit: float,
                   normal_alignment: float) -> tuple[trimesh.Trimesh, dict[str, float | int]]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    colors = _vertex_colors(mesh)
    triangles = vertices[faces]
    centers = triangles.mean(axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-15)
    origin = plane["normal"] * plane["offset"]
    candidates = (np.abs((centers - origin) @ plane["normal"]) <= distance_limit) & \
                 (np.abs(normals @ plane["normal"]) >= normal_alignment)
    candidate_ids = np.flatnonzero(candidates)
    if not len(candidate_ids):
        return mesh.copy(), {"candidate_faces": 0, "split_faces": 0, "removed_faces": 0,
                             "box_coverage": 0.0, "removed_projected_area_m2": 0.0}

    projected = contours._project_triangles(triangles[candidate_ids], plane["normal"], plane["offset"])
    x0, x1, y0, y1 = bbox
    rect = box(x0, y0, x1, y1)
    bounds_min, bounds_max = projected.min(axis=1), projected.max(axis=1)
    near = ((bounds_max[:, 0] > x0) & (bounds_min[:, 0] < x1) &
            (bounds_max[:, 1] > y0) & (bounds_min[:, 1] < y1))
    candidate_ids, projected = candidate_ids[near], projected[near]
    if not len(candidate_ids):
        return mesh.copy(), {"candidate_faces": len(candidate_ids), "split_faces": 0, "removed_faces": 0,
                             "box_coverage": 0.0, "removed_projected_area_m2": 0.0}

    polygons = shapely.polygons(projected)
    valid = shapely.is_valid(polygons) & (shapely.area(polygons) > AREA_EPSILON_M2)
    candidate_ids, projected, polygons = candidate_ids[valid], projected[valid], polygons[valid]
    if not len(candidate_ids):
        raise ValueError("Candidate mesh triangles are degenerate in the target plane")

    intersections = shapely.intersection(polygons, rect)
    overlap = shapely.area(intersections)
    affected = overlap > AREA_EPSILON_M2
    affected_ids = candidate_ids[affected]
    affected_projected = projected[affected]
    affected_polygons = polygons[affected]
    affected_intersections = intersections[affected]
    coverage = float(unary_union(polygons).intersection(rect).area / rect.area)

    fragment_vertices: list[np.ndarray] = []
    fragment_colors: list[np.ndarray] = []
    fragment_faces: list[np.ndarray] = []
    fragment_vertex_count = 0
    removed_projected_area = 0.0
    removed_ids = []
    for face_id, triangle_2d, triangle_3d, polygon, inside in zip(
            affected_ids, affected_projected, triangles[affected_ids], affected_polygons, affected_intersections):
        outside = polygon.difference(rect)
        if not np.isclose(float(inside.area + outside.area), float(polygon.area), rtol=1e-7,
                          atol=AREA_EPSILON_M2):
            raise AssertionError("Bounding-box subtraction changed projected triangle area")
        removed_projected_area += float(inside.area)
        removed_ids.append(int(face_id))
        if outside.is_empty or outside.area <= AREA_EPSILON_M2:
            continue
        new_vertices, new_colors, new_faces = _triangle_fragments(
            outside, triangle_2d, triangle_3d, colors[faces[face_id]])
        fragment_vertices.append(new_vertices)
        fragment_colors.append(new_colors)
        fragment_faces.append(new_faces + fragment_vertex_count)
        fragment_vertex_count += len(new_vertices)

    if not removed_ids:
        raise ValueError("No target mesh geometry intersects the requested bounding box")
    removed_mask = np.zeros(len(faces), dtype=bool)
    removed_mask[np.asarray(removed_ids, dtype=np.int64)] = True
    new_vertices = np.concatenate(fragment_vertices) if fragment_vertices else np.empty((0, 3), dtype=np.float64)
    new_colors = np.concatenate(fragment_colors) if fragment_colors else np.empty((0, 4), dtype=np.uint8)
    new_faces = (np.concatenate(fragment_faces) + len(vertices) if fragment_faces
                 else np.empty((0, 3), dtype=np.int64))
    result = trimesh.Trimesh(
        vertices=np.vstack((vertices, new_vertices)),
        faces=np.vstack((faces[~removed_mask], new_faces)),
        process=False,
    )
    result.visual.vertex_colors = np.vstack((colors, new_colors))
    return result, {
        "candidate_faces": int(len(candidate_ids)),
        "split_faces": int(sum(len(face_set) for face_set in fragment_faces)),
        "removed_faces": int(len(removed_ids)),
        "box_coverage": coverage,
        "removed_projected_area_m2": removed_projected_area,
    }


def _silver_panel(corners: np.ndarray) -> trimesh.Trimesh:
    plane = trimesh.Trimesh(vertices=corners, faces=np.asarray([[0, 1, 2], [0, 2, 3]]), process=False)
    plane.visual.vertex_colors = np.tile(SILVER_RGBA, (4, 1))
    return plane


def _export_glb(scene: trimesh.Scene, output_path: Path) -> dict[str, Any]:
    staging_path = output_path.with_name(output_path.stem + ".staging.glb")
    try:
        staging_path.write_bytes(scene.export(file_type="glb"))
        data = staging_path.read_bytes()
        offset = 12
        document = None
        while offset < len(data):
            length, chunk_type = struct.unpack_from("<II", data, offset)
            start = offset + 8
            if chunk_type == 0x4E4F534A:
                document = json.loads(data[start:start + length].decode("utf-8").rstrip(" \t\r\n\0"))
                break
            offset = start + length
        if document is None or not document.get("meshes"):
            raise AssertionError("Exported GLB has no mesh geometry")
        loaded = trimesh.load(staging_path, force="scene", process=False)
        if not isinstance(loaded, trimesh.Scene) or not loaded.geometry:
            raise AssertionError("Exported GLB did not round-trip as a mesh scene")
        expected_faces = sum(len(mesh.faces) for mesh in scene.geometry.values())
        actual_faces = sum(len(mesh.faces) for mesh in loaded.geometry.values())
        if actual_faces != expected_faces:
            raise AssertionError(f"GLB round-trip changed face count: {expected_faces} -> {actual_faces}")
        staging_path.replace(output_path)
        return {"mesh_nodes": len(document.get("meshes", [])), "materials": len(document.get("materials", []))}
    finally:
        staging_path.unlink(missing_ok=True)


def _export_ply(mesh: trimesh.Trimesh, output_path: Path) -> None:
    staging_path = output_path.with_name(output_path.stem + ".staging.ply")
    try:
        mesh.export(staging_path)
        check = _mesh(staging_path)
        if len(check.faces) != len(mesh.faces) or not np.array_equal(np.asarray(check.faces), np.asarray(mesh.faces)):
            raise AssertionError("PLY geometry changed during export")
        check_colors = _vertex_colors(check)
        if not np.array_equal(check_colors, np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)):
            raise AssertionError("PLY vertex colors changed during export")
        staging_path.replace(output_path)
    finally:
        staging_path.unlink(missing_ok=True)


def _validate_combined_meshes(alignment_dir: Path, meshes: dict[str, trimesh.Trimesh], filename: str) -> None:
    combined = _mesh(alignment_dir / filename)
    combined_triangles = np.asarray(combined.triangles, dtype=np.float32)
    offset = 0
    for scan_id, mesh in meshes.items():
        stop = offset + len(mesh.faces)
        triangles = np.asarray(mesh.triangles, dtype=np.float32)
        if stop > len(combined_triangles) or not np.allclose(
                combined_triangles[offset:stop], triangles, rtol=0, atol=1e-6):
            raise ValueError(f"Combined {filename} does not match current aligned geometry for {scan_id}")
        offset = stop
    if offset != len(combined_triangles):
        raise ValueError(f"Per-scan meshes cover {offset} of {len(combined_triangles)} triangles in {filename}")


def _new_scene() -> trimesh.Scene:
    return trimesh.Scene(base_frame="world")


def _stage_ply(mesh: trimesh.Trimesh, target: Path, stage_dir: Path, index: int) -> Path:
    staged = stage_dir / f"{index:05d}_{target.name}"
    _export_ply(mesh, staged)
    return staged


def run(workspace: Path, alignment_dir: Path | None = None) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    alignment_dir = Path(alignment_dir or workspace / "multi_room_alignment").resolve()
    red_path = alignment_dir / "aligned_objects_final_red.ply"
    transform_path = alignment_dir / "stats" / "aligned_scans_transforms.json"
    for path in (red_path, transform_path, alignment_dir / "aligned_sealed_meshes.ply"):
        if not path.is_file():
            raise FileNotFoundError(path)

    scans = ScanDataset(workspace).scans
    transforms = json.loads(transform_path.read_text())["scans"]
    red_ply = plyfile.PlyData.read(str(red_path), mmap="r", known_list_len={"face": {"vertex_indices": 3}})
    groups = contours._red_regions(alignment_dir, red_ply, scans)
    surfaces = {scan_id: contours._build_exterior_surface(
                    scans[scan_id], np.asarray(transforms[scan_id]["transform_matrix"], dtype=np.float64))
                for scan_id in {group[0]["scan_id"] for group in groups}}
    window_classes, associations_available = classification._window_classes(scans)
    floor_cache: dict[str, list[dict[str, Any]]] = {}
    region_data = []
    face_removals: dict[str, set[int]] = {}
    for index, group in enumerate(groups, 1):
        scan_id = group[0]["scan_id"]
        plane = contours._dominant_plane(group, surfaces[scan_id])
        triangles = np.concatenate([region["triangles"] for region in group])
        footprint = contours._triangle_union(contours._project_triangles(
            triangles, plane["normal"], plane["offset"]))
        bbox, area_coverage, contour_coverage, empty_area = contours._minimum_axis_aligned_box(footprint)
        bottom = classification._bbox_low_corner(bbox, plane)
        if scan_id not in floor_cache:
            floor_cache[scan_id] = classification._floor_planes(alignment_dir, scan_id)
        floor_height, floor_path = classification._closest_floor_height(floor_cache[scan_id], bottom)
        height_above_floor = float(bottom[1] - floor_height)
        class_ids = classification._instance_classes(group)
        gemma_window_classes = sorted(class_ids & window_classes.get(scan_id, set()))
        height_window = height_above_floor > classification.WINDOW_HEIGHT_ABOVE_FLOOR_M
        is_window = height_window or bool(gemma_window_classes)
        outward = _outward_direction(plane, surfaces[scan_id])
        inset = np.zeros(3, dtype=np.float64) if is_window else -outward * MIRROR_INSET_M
        region_id = f"red_region_{index:03d}_{scan_id}"
        sources = sorted({region["source"] for region in group})
        for region in group:
            source_key = f"{scan_id}/{region['source']}"
            face_removals.setdefault(source_key, set()).update(map(int, region["face_ids"]))
        region_data.append({
            "index": index,
            "id": region_id,
            "scan_id": scan_id,
            "class": "window" if is_window else "mirror",
            "height_above_own_floor_m": height_above_floor,
            "height_window": height_window,
            "gemma_window_classes": gemma_window_classes,
            "classification_available": scan_id in associations_available,
            "instance_sources": sources,
            "plane_normal": plane["normal"].tolist(),
            "plane_offset": float(plane["offset"]),
            "outward_direction": outward.tolist(),
            "plane_inset_m": 0.0 if is_window else MIRROR_INSET_M,
            "bbox_uv_m": [float(value) for value in bbox],
            "bbox_area_m2": float((bbox[1] - bbox[0]) * (bbox[3] - bbox[2])),
            "projected_region_area_m2": float(footprint.area),
            "bbox_area_retained_fraction": float(area_coverage),
            "bbox_contour_retained_fraction": float(contour_coverage),
            "bbox_empty_area_m2": float(empty_area),
            "closest_floor": str(floor_path),
            "panel_ply": str(alignment_dir / "objects" / scan_id / "panel_instances" /
                              f"{region_id}_{'window' if is_window else 'mirror'}_panel.ply"),
            "plane_corners": _plane_corners(bbox, plane, inset).tolist(),
            "sealed_cut": None,
            "removed_object_faces": int(sum(len(region["face_ids"]) for region in group)),
        })

    marked_face_count = int(np.asarray(red_ply["face"].data["is_phase07_outlier"], dtype=bool).sum())
    selected_face_count = sum(len(face_ids) for face_ids in face_removals.values())
    if selected_face_count != marked_face_count:
        raise AssertionError(f"Replacement selected {selected_face_count} of {marked_face_count} marked red faces")

    object_root = alignment_dir / "objects"
    output_paths = {
        "objects_ply": alignment_dir / "aligned_objects.ply",
        "sealed_ply": alignment_dir / "aligned_sealed_meshes.ply",
        "sealed_glb": alignment_dir / "aligned_sealed_meshes.glb",
        "manifest": alignment_dir / "stats" / "phase07_region_replacements.json",
    }
    staging = tempfile.TemporaryDirectory(prefix=".phase07_panel_replacement_", dir=alignment_dir)
    try:
        stage_dir = Path(staging.name)
        staged_outputs: list[tuple[Path, Path]] = []
        stage_index = 0

        def stage_ply(mesh: trimesh.Trimesh, target: Path) -> None:
            nonlocal stage_index
            stage_index += 1
            staged_outputs.append((_stage_ply(mesh, target, stage_dir, stage_index), target))

        object_meshes = []
        original_object_face_count = 0
        for scan_id in sorted(scans):
            for path in sorted((object_root / scan_id).rglob("instance_*.ply")):
                source = path.relative_to(object_root / scan_id).as_posix()
                mesh = _mesh(path)
                colors = _vertex_colors(mesh)
                faces = np.asarray(mesh.faces, dtype=np.int64)
                original_object_face_count += len(faces)
                removed = np.asarray(sorted(face_removals.get(f"{scan_id}/{source}", set())), dtype=np.int64)
                if len(removed) and (removed.min() < 0 or removed.max() >= len(faces)):
                    raise ValueError(f"Red face IDs do not match object instance {scan_id}/{source}")
                keep = np.ones(len(faces), dtype=bool)
                keep[removed] = False
                if not np.any(keep):
                    raise ValueError(f"Red-region removal would empty object instance {scan_id}/{source}")
                result = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=faces[keep], process=False)
                result.visual.vertex_colors = colors
                object_meshes.append(result)
                stage_ply(result, path)

        panel_meshes: list[trimesh.Trimesh] = []
        panels_by_scan: dict[str, list[trimesh.Trimesh]] = {scan_id: [] for scan_id in scans}
        for region in region_data:
            panel = _silver_panel(np.asarray(region["plane_corners"], dtype=np.float64))
            panel_meshes.append(panel)
            panels_by_scan[region["scan_id"]].append(panel)
            stage_ply(panel, Path(region["panel_ply"]))

        if object_meshes or panel_meshes:
            combined_objects = trimesh.util.concatenate([*object_meshes, *panel_meshes])
            expected_faces = original_object_face_count - marked_face_count + 2 * len(region_data)
            if len(combined_objects.faces) != expected_faces:
                raise AssertionError(f"Final object PLY has {len(combined_objects.faces)} faces; expected {expected_faces}")
            stage_ply(combined_objects, output_paths["objects_ply"])

        sealed_meshes = {scan_id: _mesh(alignment_dir / "scans" / scan_id / "mesh.ply")
                         for scan_id in sorted(scans)}
        _validate_combined_meshes(alignment_dir, sealed_meshes, "aligned_sealed_meshes.ply")
        sealed_final: dict[str, trimesh.Trimesh] = {}
        sealed_scene = _new_scene()
        for scan_id, source_mesh in sealed_meshes.items():
            modified = source_mesh
            for region in region_data:
                if region["scan_id"] != scan_id:
                    continue
                plane = contours._dominant_plane(groups[region["index"] - 1], surfaces[scan_id])
                modified, cut_stats = _subtract_bbox(
                    modified, plane, tuple(region["bbox_uv_m"]), SEALED_PLANE_DISTANCE_LIMIT_M,
                    SEALED_NORMAL_ALIGNMENT)
                region["sealed_cut"] = cut_stats
            additions = panels_by_scan[scan_id]
            if additions:
                modified = trimesh.util.concatenate([modified, *additions])
            sealed_final[scan_id] = modified
            stage_ply(modified, alignment_dir / "scans" / scan_id / "mesh.ply")
            node_name = f"sealed_{scan_id}"
            sealed_scene.add_geometry(modified, geom_name=node_name, node_name=node_name)

        aligned_sealed = trimesh.util.concatenate([sealed_final[scan_id] for scan_id in sorted(sealed_final)])
        stage_ply(aligned_sealed, output_paths["sealed_ply"])
        stage_glb = stage_dir / output_paths["sealed_glb"].name
        glb_stats = _export_glb(sealed_scene, stage_glb)
        staged_outputs.append((stage_glb, output_paths["sealed_glb"]))

        manifest = {
            "criteria": {
                "first_cutoff_area_m2": 0.5,
                "first_cutoff_maximum_depth_m": 0.05,
                "second_cutoff": "seam score < 3.0 OR rectangle similarity > 0.7",
                "adjacent_patches": "grouped as one region",
                "bbox_retained_fraction": 0.95,
                "window": "Gemma 2 window association OR bbox minimum > 1.5 m above closest own-scan floor",
                "mirror_panel_inset_m": MIRROR_INSET_M,
                "panel_color_rgba": SILVER_RGBA.tolist(),
                "regularized_meshes_modified": False,
            },
            "source_red_ply": str(red_path),
            "outputs": {name: str(path) for name, path in output_paths.items()},
            "aligned_sealed_glb": glb_stats,
            "removed_object_face_count": marked_face_count,
            "regions": region_data,
        }
        staged_manifest = stage_dir / output_paths["manifest"].name
        staged_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        staged_outputs.append((staged_manifest, output_paths["manifest"]))

        for staged, target in staged_outputs:
            target.parent.mkdir(parents=True, exist_ok=True)
            staged.replace(target)
    finally:
        staging.cleanup()

    # These are generated only for the earlier atomic experiments; final outputs supersede them.
    for name in ("red_region_contours", "red_region_classification", "red_region_pbr_replacements"):
        shutil.rmtree(alignment_dir / name, ignore_errors=True)
    red_path.unlink(missing_ok=True)

    for region in region_data:
        LOGGER.info("%s %s: removed %d object faces; sealed removed %d faces (%.1f%% bbox coverage)",
                    region["id"], region["class"], region["removed_object_faces"],
                    region["sealed_cut"]["removed_faces"], 100.0 * region["sealed_cut"]["box_coverage"])
    LOGGER.info("Wrote final object/sealed PLYs and %s", output_paths["sealed_glb"])
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("backend/current_scene"))
    parser.add_argument("--alignment-dir", type=Path, default=None)
    args = parser.parse_args()
    run(args.workspace, args.alignment_dir)


if __name__ == "__main__":
    main()
