"""Remove retained outliers by source face identity and add silver panels."""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import plyfile
import trimesh

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import step_classify_red_regions as classification
import step_red_region_contours as contours
from scan_loader import ScanDataset

LOGGER = logging.getLogger("phase07_region_replacement")
MIRROR_INSET_M = 0.01
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


def _source_face_ids(scan, segmented_mesh: plyfile.PlyData,
                     source_instance_path: Path, class_id: int,
                     expected_faces: int) -> np.ndarray:
    match = re.fullmatch(r"instance_(\d+)", source_instance_path.stem)
    if not match:
        raise ValueError(f"Cannot read instance index from source mesh: {source_instance_path}")
    face_data = segmented_mesh["face"].data
    required = {"class_id", "instance_id", "vertex_indices"}
    if not required.issubset(face_data.dtype.names or ()):
        raise ValueError(f"Stage 05 segmented mesh has no source instance face labels: {scan.info.scan_id}")
    source_ids = np.flatnonzero(
        (face_data["class_id"] == class_id) & (face_data["instance_id"] == int(match.group(1)))
    )
    if len(source_ids) != expected_faces:
        raise ValueError(
            f"Stage 05 source-face count differs for {source_instance_path}: "
            f"{len(source_ids)} != {expected_faces}"
        )

    source_ply = plyfile.PlyData.read(
        str(source_instance_path), mmap="r", known_list_len={"face": {"vertex_indices": 3}}
    )
    source_vertex_data = source_ply["vertex"].data
    source_faces = contours._face_matrix(source_ply)
    source_triangles = np.stack(
        [source_vertex_data[name][source_faces] for name in ("x", "y", "z")], axis=-1
    )
    segmented_vertex_data = segmented_mesh["vertex"].data
    segmented_faces = np.asarray(face_data["vertex_indices"][source_ids], dtype=np.int64)
    segmented_triangles = np.stack(
        [segmented_vertex_data[name][segmented_faces] for name in ("x", "y", "z")], axis=-1
    )
    if not np.array_equal(source_triangles, segmented_triangles):
        raise ValueError(
            f"Stage 05 instance face order/geometry differs from its source rows: {source_instance_path}"
        )
    return source_ids


def _fit_panel_to_mesh_and_floor(corners: np.ndarray, bounds: np.ndarray,
                                 floor: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    panel_min, panel_max = corners.min(axis=0), corners.max(axis=0)
    bounds_min, bounds_max = bounds
    lower_translation = bounds_min - panel_min
    upper_translation = bounds_max - panel_max
    if np.any(lower_translation > upper_translation):
        raise ValueError("Replacement panel is too large to fit inside its sealed mesh bounds by translation")

    translation = np.maximum(lower_translation, np.minimum(0.0, upper_translation))
    shifted_xz = corners + np.array([translation[0], 0.0, translation[2]])
    normal, offset = floor["normal"], floor["offset"]
    floor_translation = max(
        (offset - normal[0] * point[0] - normal[2] * point[2]) / normal[1] - point[1]
        for point in shifted_xz
    )
    lower_y = max(lower_translation[1], floor_translation)
    if lower_y > upper_translation[1]:
        raise ValueError("Replacement panel cannot fit inside sealed mesh bounds while staying above its floor")
    translation[1] = max(lower_y, min(0.0, upper_translation[1]))
    moved = corners + translation
    if np.any(moved.min(axis=0) < bounds_min) or np.any(moved.max(axis=0) > bounds_max):
        raise AssertionError("Replacement panel could not be moved inside its sealed mesh bounds")
    if np.any(moved[:, 1] < np.asarray([
        (offset - normal[0] * point[0] - normal[2] * point[2]) / normal[1]
        for point in moved
    ]) - 1e-6):
        raise AssertionError("Replacement panel remains below its selected floor plane")
    return moved, translation


def _placement_groups(group: list[dict[str, Any]],
                      source_records: dict[tuple[str, str], dict[str, Any]],
                      scan_id: str) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    for region in group:
        key = (scan_id, region["source"])
        record = source_records.get(key)
        if record is None:
            raise ValueError(f"No aligned object manifest record for {scan_id}/{region['source']}")
        translation = np.asarray(record.get("post_scan_translation_m"), dtype=np.float64)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError(f"Invalid final placement translation for {scan_id}/{region['source']}")
        existing = next((item for item in grouped
                         if np.allclose(item["translation"], translation, rtol=0, atol=1e-9)), None)
        if existing is None:
            existing = {"translation": translation, "regions": []}
            grouped.append(existing)
        existing["regions"].append(region)
    return grouped


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


def _silver_panel(corners: np.ndarray) -> trimesh.Trimesh:
    plane = trimesh.Trimesh(vertices=corners, faces=np.asarray([[0, 1, 2], [0, 2, 3]]), process=False)
    plane.visual.vertex_colors = np.tile(SILVER_RGBA, (4, 1))
    return plane


def _faces_behind_panel(vertices: np.ndarray, faces: np.ndarray, corners: np.ndarray,
                        outward: np.ndarray) -> np.ndarray:
    """Find faces intersecting the panel's outward, axis-aligned projection."""
    axis = int(np.argmax(np.abs(outward)))
    projected_axes = [i for i in range(3) if i != axis]
    lower = corners[:, projected_axes].min(axis=0)
    upper = corners[:, projected_axes].max(axis=0)
    normal = np.cross(corners[1] - corners[0], corners[3] - corners[0])
    normal_length = np.linalg.norm(normal)
    if normal_length == 0 or abs(normal @ outward) < 1e-12:
        raise ValueError("Cannot determine the outward side of a sealed panel")
    normal = normal / normal_length * np.sign(normal @ outward)
    # Saved sealed-mesh vertices use float32; allow their rounding at this coordinate scale.
    tolerance = 2 * np.finfo(np.float32).eps * max(1.0, float(np.abs(corners).max()))
    plane_offset = float(normal @ corners[0]) + tolerance
    planes = []
    for i, dimension in enumerate(projected_axes):
        direction = np.eye(3)[dimension]
        planes.extend(((direction, lower[i]), (-direction, -upper[i])))
    planes.append((normal, plane_offset))

    result = np.zeros(len(faces), dtype=bool)
    for start in range(0, len(faces), 100_000):
        triangles = vertices[faces[start:start + 100_000]]
        projected = triangles[:, :, projected_axes]
        depth = triangles @ normal - plane_offset
        inside = ((projected >= lower) & (projected <= upper)).all(axis=2) & (depth > 0)
        hits = inside.any(axis=1)
        candidates = (~hits & (projected.max(axis=1) >= lower).all(axis=1)
                      & (projected.min(axis=1) <= upper).all(axis=1) & (depth.max(axis=1) > 0))
        # A triangle can cross the projection even if none of its vertices are inside it.
        for index in np.flatnonzero(candidates):
            polygon = triangles[index]
            for direction, offset in planes:
                distances = polygon @ direction - offset
                if np.all(distances >= 0):
                    continue
                if np.all(distances < 0):
                    break
                clipped = []
                previous, previous_distance = polygon[-1], distances[-1]
                for current, current_distance in zip(polygon, distances):
                    if (previous_distance >= 0) != (current_distance >= 0):
                        clipped.append(previous + (current - previous) *
                                       (previous_distance / (previous_distance - current_distance)))
                    if current_distance >= 0:
                        clipped.append(current)
                    previous, previous_distance = current, current_distance
                polygon = np.asarray(clipped)
            else:
                hits[index] = True
        result[start:start + len(triangles)] = hits
    return result


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
    replacement_timings: dict[str, float] = {}
    stage_started = time.perf_counter()
    workspace = Path(workspace).resolve()
    alignment_dir = Path(alignment_dir or workspace / "multi_room_alignment").resolve()
    red_path = alignment_dir / "aligned_objects_final_red.ply"
    transform_path = alignment_dir / "stats" / "aligned_scans_transforms.json"
    for path in (red_path, transform_path, alignment_dir / "aligned_sealed_meshes.ply"):
        if not path.is_file():
            raise FileNotFoundError(path)

    scans = ScanDataset(workspace).scans
    replacement_timings["load_scan_metadata"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    sealed_meshes = {
        scan_id: _mesh(alignment_dir / "scans" / scan_id / "mesh.ply")
        for scan_id in sorted(scans)
    }
    replacement_timings["load_sealed_meshes"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    transforms = json.loads(transform_path.read_text())["scans"]
    object_manifest_path = alignment_dir / "stats" / "aligned_objects_manifest.json"
    if not object_manifest_path.is_file():
        raise FileNotFoundError(object_manifest_path)
    object_manifest = json.loads(object_manifest_path.read_text())
    source_records: dict[tuple[str, str], dict[str, Any]] = {}
    for record in object_manifest.get("objects", []):
        scan_id = record["scan_id"]
        object_root = (alignment_dir / "objects" / scan_id).resolve()
        global_path = Path(record["global_instance_path"]).resolve()
        try:
            relative_path = global_path.relative_to(object_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Aligned object path is outside its scan directory: {global_path}") from exc
        key = (scan_id, relative_path)
        if key in source_records:
            raise ValueError(f"Duplicate aligned object manifest record: {scan_id}/{relative_path}")
        source_records[key] = record

    red_ply = plyfile.PlyData.read(str(red_path), mmap="r", known_list_len={"face": {"vertex_indices": 3}})
    groups = contours._red_regions(alignment_dir, red_ply, scans)
    surfaces = {scan_id: contours._build_exterior_surface(
                    scans[scan_id], np.asarray(transforms[scan_id]["transform_matrix"], dtype=np.float64))
                for scan_id in {group[0]["scan_id"] for group in groups}}
    window_classes, associations_available = classification._window_classes(scans)
    replacement_timings["load_manifests_and_prepare_regions"] = time.perf_counter() - stage_started
    floor_cache: dict[str, list[dict[str, Any]]] = {}
    region_data = []
    face_removals: dict[str, set[int]] = {}
    sealed_face_removals: dict[str, set[int]] = {scan_id: set() for scan_id in scans}
    segmented_meshes: dict[str, plyfile.PlyData] = {}
    source_face_ids_by_instance: dict[tuple[str, str], np.ndarray] = {}
    source_face_mapping_seconds = 0.0
    region_bbox_seconds = 0.0
    placement_bbox_seconds = 0.0
    stage_started = time.perf_counter()
    for index, group in enumerate(groups, 1):
        scan_id = group[0]["scan_id"]
        plane = contours._dominant_plane(group, surfaces[scan_id])
        triangles = np.concatenate([region["triangles"] for region in group])
        footprint = contours._triangle_union(contours._project_triangles(
            triangles, plane["normal"], plane["offset"]))
        bbox_started = time.perf_counter()
        bbox, area_coverage, contour_coverage, empty_area = contours._minimum_axis_aligned_box(footprint)
        region_bbox_seconds += time.perf_counter() - bbox_started
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
        removed_sealed_faces = 0
        for region in group:
            source = region["source"]
            source_key = (scan_id, source)
            object_record = source_records.get(source_key)
            if object_record is None:
                raise ValueError(f"No aligned object manifest record for {scan_id}/{source}")
            if scan_id not in segmented_meshes:
                segmented_path = scans[scan_id].info.scan_dir / "segmentation" / "mesh_segmented.ply"
                if not segmented_path.is_file():
                    raise FileNotFoundError(segmented_path)
                segmented_meshes[scan_id] = plyfile.PlyData.read(
                    str(segmented_path), mmap="r", known_list_len={"face": {"vertex_indices": 3}}
                )
            if source_key not in source_face_ids_by_instance:
                mapping_started = time.perf_counter()
                source_face_ids_by_instance[source_key] = _source_face_ids(
                    scans[scan_id], segmented_meshes[scan_id],
                    Path(object_record["source_instance_path"]), int(object_record["class_id"]),
                    int(object_record["triangles"]),
                )
                source_face_mapping_seconds += time.perf_counter() - mapping_started
            source_face_ids = source_face_ids_by_instance[source_key]
            local_face_ids = np.asarray(region["face_ids"], dtype=np.int64)
            if len(local_face_ids) and (local_face_ids.min() < 0 or local_face_ids.max() >= len(source_face_ids)):
                raise ValueError(f"Red face IDs do not match source instance {scan_id}/{source}")
            object_face_key = f"{scan_id}/{source}"
            object_removed = face_removals.setdefault(object_face_key, set())
            if object_removed.intersection(map(int, local_face_ids)):
                raise ValueError(f"Red object faces were selected more than once: {scan_id}/{source}")
            object_removed.update(map(int, local_face_ids))
            sealed_ids = set(map(int, source_face_ids[local_face_ids]))
            sealed_removed = sealed_face_removals[scan_id]
            if sealed_removed.intersection(sealed_ids):
                raise ValueError(f"Sealed source faces were selected more than once for {scan_id}")
            sealed_removed.update(sealed_ids)
            removed_sealed_faces += len(sealed_ids)

        placement_groups = _placement_groups(group, source_records, scan_id)
        sealed_panels = []
        for placement_index, placement in enumerate(placement_groups, 1):
            placement_regions = placement["regions"]
            if len(placement_groups) == 1:
                # A single placement group has the exact footprint already optimized above.
                placement_bbox = bbox
                placement_area_coverage = area_coverage
                placement_contour_coverage = contour_coverage
                placement_empty_area = empty_area
            else:
                placement_triangles = np.concatenate([item["triangles"] for item in placement_regions])
                placement_footprint = contours._triangle_union(contours._project_triangles(
                    placement_triangles, plane["normal"], plane["offset"]))
                bbox_started = time.perf_counter()
                placement_bbox, placement_area_coverage, placement_contour_coverage, placement_empty_area = \
                    contours._minimum_axis_aligned_box(placement_footprint)
                placement_bbox_seconds += time.perf_counter() - bbox_started
            corners = _plane_corners(placement_bbox, plane, inset) - placement["translation"]
            floor_plane = next(item for item in floor_cache[scan_id] if item["path"] == floor_path)
            corners, placement_correction = _fit_panel_to_mesh_and_floor(
                corners, np.asarray(sealed_meshes[scan_id].bounds, dtype=np.float64), floor_plane
            )
            suffix = f"_{placement_index:03d}" if len(placement_groups) > 1 else ""
            panel_path = (alignment_dir / "scans" / scan_id / "panel_instances" /
                          f"{region_id}{suffix}_{'window' if is_window else 'mirror'}_panel.ply")
            sealed_panels.append({
                "panel_ply": str(panel_path),
                "plane_corners": corners.tolist(),
                "placement_translation_m": placement["translation"].tolist(),
                "sealed_panel_correction_m": placement_correction.tolist(),
                "source_instances": sorted({item["source"] for item in placement_regions}),
                "bbox_uv_m": [float(value) for value in placement_bbox],
                "bbox_area_retained_fraction": float(placement_area_coverage),
                "bbox_contour_retained_fraction": float(placement_contour_coverage),
                "bbox_empty_area_m2": float(placement_empty_area),
            })

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
            "sealed_panels": sealed_panels,
            "sealed_cut": {
                "method": "exact_source_face_ids",
                "removed_faces": removed_sealed_faces,
            },
            "removed_object_faces": int(sum(len(region["face_ids"]) for region in group)),
        })
    replacement_timings["classify_regions_map_faces_and_place_panels"] = time.perf_counter() - stage_started
    replacement_timings["exact_source_face_mapping_and_geometry_validation"] = source_face_mapping_seconds
    replacement_timings["region_bbox_optimization"] = region_bbox_seconds
    replacement_timings["placement_bbox_optimization"] = placement_bbox_seconds

    marked_face_count = int(np.asarray(red_ply["face"].data["is_phase07_outlier"], dtype=bool).sum())
    selected_face_count = sum(len(face_ids) for face_ids in face_removals.values())
    if selected_face_count != marked_face_count:
        raise AssertionError(f"Replacement selected {selected_face_count} of {marked_face_count} marked red faces")
    selected_sealed_face_count = sum(len(face_ids) for face_ids in sealed_face_removals.values())
    if selected_sealed_face_count != marked_face_count:
        raise AssertionError(
            f"Sealed source mapping selected {selected_sealed_face_count} of {marked_face_count} marked red faces"
        )
    LOGGER.info(
        "Exact source-face mapping and geometry validation took %.2f s across %d instances",
        source_face_mapping_seconds, len(source_face_ids_by_instance),
    )

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
        stage_started = time.perf_counter()
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
        replacement_timings["rewrite_object_instances"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        object_panel_meshes: list[trimesh.Trimesh] = []
        sealed_panels_by_scan: dict[str, list[trimesh.Trimesh]] = {scan_id: [] for scan_id in scans}
        panel_projections_by_scan: dict[str, list[tuple[dict[str, Any], np.ndarray, np.ndarray]]] = {
            scan_id: [] for scan_id in scans
        }
        for region in region_data:
            object_panel = _silver_panel(np.asarray(region["plane_corners"], dtype=np.float64))
            object_panel_meshes.append(object_panel)
            stage_ply(object_panel, Path(region["panel_ply"]))
            for panel_spec in region["sealed_panels"]:
                sealed_panel = _silver_panel(np.asarray(panel_spec["plane_corners"], dtype=np.float64))
                sealed_panels_by_scan[region["scan_id"]].append(sealed_panel)
                panel_projections_by_scan[region["scan_id"]].append(
                    (panel_spec, np.asarray(panel_spec["plane_corners"], dtype=np.float64),
                     np.asarray(region["outward_direction"], dtype=np.float64))
                )
                stage_ply(sealed_panel, Path(panel_spec["panel_ply"]))
        replacement_timings["write_replacement_panels"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        if object_meshes or object_panel_meshes:
            combined_objects = trimesh.util.concatenate([*object_meshes, *object_panel_meshes])
            expected_faces = original_object_face_count - marked_face_count + 2 * len(region_data)
            if len(combined_objects.faces) != expected_faces:
                raise AssertionError(f"Final object PLY has {len(combined_objects.faces)} faces; expected {expected_faces}")
            stage_ply(combined_objects, output_paths["objects_ply"])
        replacement_timings["write_combined_objects_mesh"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        _validate_combined_meshes(alignment_dir, sealed_meshes, "aligned_sealed_meshes.ply")
        sealed_final: dict[str, trimesh.Trimesh] = {}
        behind_panel_removals: dict[str, int] = {}
        sealed_scene = _new_scene()
        for scan_id, source_mesh in sealed_meshes.items():
            faces = np.asarray(source_mesh.faces, dtype=np.int64)
            removed = np.asarray(sorted(sealed_face_removals[scan_id]), dtype=np.int64)
            if len(removed) and (removed.min() < 0 or removed.max() >= len(faces)):
                raise ValueError(f"Source face IDs do not match aligned sealed mesh for {scan_id}")
            if len(removed):
                segmented_mesh = segmented_meshes[scan_id]
                face_data = segmented_mesh["face"].data
                segmented_faces = np.asarray(face_data["vertex_indices"][removed], dtype=np.int64)
                vertex_data = segmented_mesh["vertex"].data
                source_triangles = np.stack(
                    [vertex_data[name][segmented_faces] for name in ("x", "y", "z")], axis=-1
                ).astype(np.float64)
                transform = np.asarray(transforms[scan_id]["transform_matrix"], dtype=np.float64)
                expected_triangles = source_triangles @ transform[:3, :3].T + transform[:3, 3]
                if not np.allclose(
                    np.asarray(source_mesh.triangles)[removed], expected_triangles, rtol=0, atol=1e-5
                ):
                    raise ValueError(f"Mapped source face geometry differs from aligned sealed mesh for {scan_id}")
            keep = np.ones(len(faces), dtype=bool)
            keep[removed] = False
            if not np.any(keep):
                raise ValueError(f"Phase 07 face removal would empty sealed mesh for {scan_id}")
            modified = trimesh.Trimesh(
                vertices=np.asarray(source_mesh.vertices), faces=faces[keep], process=False
            ) if len(removed) else source_mesh
            if len(removed):
                modified.visual.vertex_colors = _vertex_colors(source_mesh)
            additions = sealed_panels_by_scan[scan_id]
            if additions:
                modified = trimesh.util.concatenate([modified, *additions])
            vertices = np.asarray(modified.vertices)
            final_faces = np.asarray(modified.faces, dtype=np.int64)
            behind = np.zeros(len(final_faces), dtype=bool)
            panel_projections = panel_projections_by_scan[scan_id]
            first_panel_face = len(final_faces) - 2 * len(panel_projections)
            for index, (_, corners, outward) in enumerate(panel_projections):
                panel_behind = _faces_behind_panel(vertices, final_faces, corners, outward)
                panel_behind[first_panel_face + 2 * index:first_panel_face + 2 * index + 2] = False
                behind |= panel_behind
            for index, (panel_spec, _, _) in enumerate(panel_projections):
                panel_spec["retained_face_indices"] = [
                    face_index for face_index in (0, 1)
                    if not behind[first_panel_face + 2 * index + face_index]
                ]
            behind_panel_removals[scan_id] = int(behind.sum())
            if np.any(behind):
                if np.all(behind):
                    raise ValueError(f"Panel projection removal would empty sealed mesh for {scan_id}")
                colors = _vertex_colors(modified)
                modified = trimesh.Trimesh(vertices=vertices, faces=final_faces[~behind], process=False)
                modified.visual.vertex_colors = colors
            sealed_final[scan_id] = modified
            stage_ply(modified, alignment_dir / "scans" / scan_id / "mesh.ply")
            node_name = f"sealed_{scan_id}"
            sealed_scene.add_geometry(modified, geom_name=node_name, node_name=node_name)
        replacement_timings["remove_sealed_faces_and_write_scan_meshes"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        aligned_sealed = trimesh.util.concatenate([sealed_final[scan_id] for scan_id in sorted(sealed_final)])
        stage_ply(aligned_sealed, output_paths["sealed_ply"])
        stage_glb = stage_dir / output_paths["sealed_glb"].name
        glb_stats = _export_glb(sealed_scene, stage_glb)
        staged_outputs.append((stage_glb, output_paths["sealed_glb"]))
        replacement_timings["write_combined_sealed_mesh_and_glb"] = time.perf_counter() - stage_started

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
                "sealed_mesh_removal": "exact Phase 05 source faces corresponding to selected object faces, then all faces intersecting the outward axis-aligned projection of each same-scan sealed panel",
                "sealed_panel_placement": "reverse each contributing instance placement translation, then fit inside its sealed mesh bounds and above its selected floor plane",
                "regularized_meshes_modified": False,
            },
            "source_red_ply": str(red_path),
            "outputs": {name: str(path) for name, path in output_paths.items()},
            "aligned_sealed_glb": glb_stats,
            "removed_object_face_count": marked_face_count,
            "removed_sealed_face_count": selected_sealed_face_count,
            "removed_sealed_behind_panel_face_count": sum(behind_panel_removals.values()),
            "removed_sealed_behind_panel_face_count_by_scan": behind_panel_removals,
            "timings_seconds": replacement_timings,
            "regions": region_data,
        }
        staged_manifest = stage_dir / output_paths["manifest"].name
        staged_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        staged_outputs.append((staged_manifest, output_paths["manifest"]))

        stage_started = time.perf_counter()
        for staged, target in staged_outputs:
            target.parent.mkdir(parents=True, exist_ok=True)
            staged.replace(target)
        replacement_timings["publish_staged_outputs"] = time.perf_counter() - stage_started
    finally:
        staging.cleanup()

    # These are generated only for the earlier atomic experiments; final outputs supersede them.
    for name in ("red_region_contours", "red_region_classification", "red_region_pbr_replacements"):
        shutil.rmtree(alignment_dir / name, ignore_errors=True)
    red_path.unlink(missing_ok=True)

    for region in region_data:
        LOGGER.info("%s %s: removed %d object faces and %d matching sealed faces; %d sealed-frame panel(s)",
                    region["id"], region["class"], region["removed_object_faces"],
                    region["sealed_cut"]["removed_faces"], len(region["sealed_panels"]))
    LOGGER.info("Wrote final object/sealed PLYs and %s", output_paths["sealed_glb"])
    for name, elapsed in replacement_timings.items():
        LOGGER.info("Timing step5.%-43s %.2f s", name, elapsed)
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
