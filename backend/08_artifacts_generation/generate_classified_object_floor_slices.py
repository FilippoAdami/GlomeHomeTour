#!/usr/bin/env python3
"""Create experimental copies of regularized floor slices with aligned classified objects."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from generate_floor_level_meshes import _read_mesh, _write_mesh
from generate_floor_level_projections import _project_topdown


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MESH_DIR = BACKEND_DIR / "current_scene" / "artifacts" / "floor_level_meshes"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "current_scene" / "artifacts" / "classified_object_slice_experiment"
BOX_EDGE_WIDTH_M = 0.01
BOX_COLOR = np.array([255, 0, 255, 255], dtype=np.uint8)
LABEL_PLATE_COLOR = np.array([255, 255, 255, 255], dtype=np.uint8)
LABEL_TEXT_COLOR = np.array([0, 0, 0, 255], dtype=np.uint8)
CUBOID_FACES = np.array([
    [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
    [0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7],
    [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
], dtype=np.int32)


def _unit_plane(equation: list[float]) -> tuple[np.ndarray, float]:
    plane = np.asarray(equation, dtype=np.float64)
    if plane.shape != (4,) or not np.isfinite(plane).all():
        raise ValueError(f"Invalid slice plane equation: {equation}")
    length = float(np.linalg.norm(plane[:3]))
    if length < 1e-8:
        raise ValueError(f"Invalid slice plane equation: {equation}")
    normal, offset = plane[:3] / length, float(plane[3] / length)
    if normal[1] < 0:
        normal, offset = -normal, -offset
    return normal, offset


def _slice_enclosure(
    bounds: np.ndarray, floor_planes: list[dict], ceiling_equation: list[float] | None,
) -> dict:
    """Measure the object's vertical overlap with a slice at the bbox's X/Z center."""
    x, z = (bounds[0, [0, 2]] + bounds[1, [0, 2]]) * 0.5
    object_low, object_high = float(bounds[0, 1]), float(bounds[1, 1])
    object_height = max(object_high - object_low, 1e-8)
    ceiling_y = None
    if ceiling_equation is not None:
        normal, offset = _unit_plane(ceiling_equation)
        ceiling_y = float(-(normal[0] * x + normal[2] * z + offset) / normal[1])

    candidates = []
    for plane in floor_planes:
        normal, offset = _unit_plane(plane["plane_equation"])
        floor_y = float(-(normal[0] * x + normal[2] * z + offset) / normal[1])
        slice_high = object_high if ceiling_y is None else ceiling_y
        if slice_high >= floor_y:
            overlap = max(0.0, min(object_high, slice_high) - max(object_low, floor_y))
            gap = max(floor_y - object_high, object_low - slice_high, 0.0)
        else:
            overlap = 0.0
            object_center = (object_low + object_high) * 0.5
            gap = min(abs(object_center - floor_y), abs(object_center - slice_high))
        candidates.append({
            "vertical_overlap_m": overlap,
            "enclosure_fraction": overlap / object_height,
            "vertical_gap_m": gap,
            "floor_height_at_bbox_center_m": floor_y,
            "ceiling_cut_height_at_bbox_center_m": ceiling_y,
        })
    return max(candidates, key=lambda item: (item["vertical_overlap_m"], -item["vertical_gap_m"]))


def _assign_slice(bounds: np.ndarray, slices: list[dict]) -> tuple[dict, list[dict]]:
    scores = []
    for index, slice_info in enumerate(slices):
        score = _slice_enclosure(
            bounds, slice_info["floor_planes"], slice_info.get("ceiling_cut_plane_equation"),
        )
        scores.append({"floor_level_name": slice_info["floor_level_name"], **score})
    selected_index = max(
        range(len(scores)),
        key=lambda index: (scores[index]["vertical_overlap_m"], -scores[index]["vertical_gap_m"], -index),
    )
    return scores[selected_index], scores


def _cuboid(bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low, high = bounds
    vertices = np.array([
        [low[0], low[1], low[2]], [high[0], low[1], low[2]],
        [high[0], high[1], low[2]], [low[0], high[1], low[2]],
        [low[0], low[1], high[2]], [high[0], low[1], high[2]],
        [high[0], high[1], high[2]], [low[0], high[1], high[2]],
    ], dtype=np.float32)
    return vertices, CUBOID_FACES.copy()


def _box_edges(
    bounds: np.ndarray, color: np.ndarray, edge_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parts_v, parts_f = [], []
    half_width = edge_width * 0.5
    for axis in range(3):
        other_axes = [dimension for dimension in range(3) if dimension != axis]
        for first_side in (0, 1):
            for second_side in (0, 1):
                low, high = bounds[0].copy(), bounds[1].copy()
                low[other_axes[0]] = high[other_axes[0]] = bounds[first_side, other_axes[0]]
                low[other_axes[1]] = high[other_axes[1]] = bounds[second_side, other_axes[1]]
                low[other_axes] -= half_width
                high[other_axes] += half_width
                vertices, faces = _cuboid(np.stack((low, high)))
                parts_f.append(faces + sum(len(part) for part in parts_v))
                parts_v.append(vertices)
    vertices, faces = np.concatenate(parts_v), np.concatenate(parts_f)
    colors = np.tile(color, (len(vertices), 1))
    return vertices, faces, colors


def _quad_parts(
    vertices: list[list[float]], faces: list[list[int]], colors: list[list[int]],
    x0: float, x1: float, z0: float, z1: float, y: float, color: np.ndarray,
) -> None:
    offset = len(vertices)
    vertices.extend([[x0, y, z0], [x0, y, z1], [x1, y, z1], [x1, y, z0]])
    faces.extend([[offset, offset + 1, offset + 2], [offset, offset + 2, offset + 3]])
    colors.extend([color.tolist()] * 4)


def _label_on_top(bounds: np.ndarray, label: str, y_offset: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, baseline = cv2.getTextSize(label, font, 0.8, 1)
    padding = 4
    image_width = text_size[0] + padding * 2
    image_height = text_size[1] + baseline + padding * 2
    bitmap = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.putText(bitmap, label, (padding, padding + text_size[1]), font, 0.8, 255, 1, cv2.LINE_8)

    low, high = bounds
    world_per_pixel = min(
        (high[0] - low[0]) * 0.45 / image_width,
        (high[2] - low[2]) * 0.3 / image_height,
    )
    if world_per_pixel <= 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.int32),
            np.empty((0, 4), dtype=np.uint8),
        )

    width, depth = image_width * world_per_pixel, image_height * world_per_pixel
    x0 = float((low[0] + high[0] - width) * 0.5)
    z0 = float((low[2] + high[2] - depth) * 0.5)
    y = float(high[1] + y_offset)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    colors: list[list[int]] = []
    _quad_parts(vertices, faces, colors, x0, x0 + width, z0, z0 + depth, y, LABEL_PLATE_COLOR)

    for row, values in enumerate(bitmap >= 128):
        padded = np.r_[False, values, False].astype(np.int8)
        starts = np.flatnonzero(np.diff(padded) == 1)
        stops = np.flatnonzero(np.diff(padded) == -1)
        z_top = z0 + depth - row * world_per_pixel
        z_bottom = z_top - world_per_pixel
        for start, stop in zip(starts, stops):
            _quad_parts(
                vertices, faces, colors,
                x0 + start * world_per_pixel, x0 + stop * world_per_pixel,
                z_bottom, z_top, y + 0.0005, LABEL_TEXT_COLOR,
            )
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32), np.asarray(colors, dtype=np.uint8)


def _merge(parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices, faces, colors = [], [], []
    offset = 0
    for part_vertices, part_faces, part_colors in parts:
        if not len(part_faces):
            continue
        vertices.append(part_vertices)
        faces.append(part_faces + offset)
        colors.append(part_colors)
        offset += len(part_vertices)
    return np.concatenate(vertices), np.concatenate(faces), np.concatenate(colors)


def generate_classified_object_slices(
    mesh_dir: Path = DEFAULT_MESH_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    resolution: int = 1024,
) -> dict:
    mesh_dir, output_dir = Path(mesh_dir).resolve(), Path(output_dir).resolve()
    source_manifest = mesh_dir / "floor_level_meshes_manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Floor mesh manifest is missing: {source_manifest}")
    floor_manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    metadata_path = Path(floor_manifest["source_metadata"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    alignment_dir = Path(metadata["source_alignment_dir"])
    planes = {plane["plane_id"]: plane for plane in metadata["floor_planes"]}
    output_dir.mkdir(parents=True, exist_ok=True)

    slices_by_scan = defaultdict(list)
    for level in floor_manifest["levels"]:
        for scan in level["scans"]:
            if scan["output_status"] != "generated_for_floor_level":
                continue
            floor_planes = [
                planes[plane_id]
                for plane_id in scan.get("floor_plane_ids", [])
                if plane_id in planes and planes[plane_id].get("floor_level_name") == level["floor_level_name"]
            ]
            if floor_planes:
                slices_by_scan[scan["scan_id"]].append({
                    "floor_level_name": level["floor_level_name"],
                    "floor_planes": floor_planes,
                    "ceiling_cut_plane_equation": scan.get("ceiling_cut_plane_equation"),
                })

    assignments = defaultdict(list)
    objects_by_scan = defaultdict(list)
    skipped_classes_by_scan = defaultdict(list)
    scan_skip_reason = {}
    for scan_id, scan_slices in slices_by_scan.items():
        scan_dir = alignment_dir.parent / scan_id
        object_root = alignment_dir / "objects" / scan_id
        association_path = scan_dir / "segmentation" / "non_structural_similarity" / "valid_class_associations.json"
        categories_path = association_path.parent / "categories.json"
        if not association_path.is_file() or not categories_path.is_file():
            scan_skip_reason[scan_id] = "classification_files_missing"
            continue

        associations_data = json.loads(association_path.read_text(encoding="utf-8"))
        categories_data = json.loads(categories_path.read_text(encoding="utf-8"))
        class_metadata_path = scan_dir / "segmentation" / "class_metadata.json"
        class_metadata = (
            json.loads(class_metadata_path.read_text(encoding="utf-8")).get("classes", {})
            if class_metadata_path.is_file() else {}
        )
        category_names = {int(item["id"]): str(item["name"]) for item in categories_data}
        associations = {
            int(source_id): int(category_id)
            for source_id, category_id in associations_data["associations"]
        }

        for source_class_id, category_id in sorted(associations.items()):
            label = category_names[category_id]
            class_info = class_metadata.get(str(source_class_id), {})
            role = class_info.get("structural_role")
            if role not in (None, "object", "wall"):
                skipped_classes_by_scan[scan_id].append({
                    "source_class_id": source_class_id,
                    "label": label,
                    "structural_role": role,
                    "skip_reason": "structural_class_not_exported_as_an_object",
                })
                continue

            object_dir = object_root / f"class_{source_class_id:03d}_object"
            instance_paths = sorted(object_dir.glob("instance_*.ply"))
            if not instance_paths:
                skipped_classes_by_scan[scan_id].append({
                    "source_class_id": source_class_id,
                    "label": label,
                    "structural_role": role,
                    "skip_reason": "aligned_object_mesh_missing",
                })
                continue

            for instance_path in instance_paths:
                object_vertices, object_faces, object_colors = _read_mesh(instance_path)
                bounds = np.stack((object_vertices.min(axis=0), object_vertices.max(axis=0)))
                selected, scores = _assign_slice(bounds, scan_slices)
                record = {
                    "scan_id": scan_id,
                    "source_class_id": source_class_id,
                    "category_id": category_id,
                    "label": label,
                    "source_structural_role": role,
                    "instance_id": instance_path.stem,
                    "aligned_mesh": str(instance_path),
                    "axis_aligned_bbox_min_xyz": bounds[0].tolist(),
                    "axis_aligned_bbox_max_xyz": bounds[1].tolist(),
                    "assigned_floor_level": selected["floor_level_name"],
                    "vertical_overlap_m": selected["vertical_overlap_m"],
                    "enclosure_fraction": selected["enclosure_fraction"],
                    "vertical_gap_m": selected["vertical_gap_m"],
                    "slice_scores": scores,
                }
                assigned_object = {
                    "vertices": object_vertices,
                    "faces": object_faces,
                    "colors": object_colors,
                    "bounds": bounds,
                    "record": record,
                }
                assignments[(scan_id, selected["floor_level_name"])].append(assigned_object)
                objects_by_scan[scan_id].append(assigned_object)

    level_results = []

    for level in floor_manifest["levels"]:
        source_mesh = Path(level["regularized_mesh"])
        base_vertices, base_faces, base_colors = _read_mesh(source_mesh)
        parts = [(base_vertices, base_faces, base_colors)]
        object_records = []
        included_count = 0

        for scan in level["scans"]:
            scan_id = scan["scan_id"]
            scan_result = {
                "scan_id": scan_id,
                "output_status": scan["output_status"],
                "included_objects": 0,
            }
            if scan["output_status"] != "generated_for_floor_level":
                scan_result["skip_reason"] = "scan_has_no_floor_at_this_level"
                object_records.append(scan_result)
                continue

            floor_planes = [
                planes[plane_id]
                for plane_id in scan.get("floor_plane_ids", [])
                if plane_id in planes and planes[plane_id].get("floor_level_name") == level["floor_level_name"]
            ]
            if not floor_planes:
                scan_result["skip_reason"] = "no_floor_plane_for_level"
                object_records.append(scan_result)
                continue
            assigned_objects = assignments[(scan_id, level["floor_level_name"])]
            scan_result["included_objects"] = len(assigned_objects)
            scan_result["objects"] = []
            for assigned_object in assigned_objects:
                bounds = assigned_object["bounds"]
                label = assigned_object["record"]["label"]
                parts.append((
                    assigned_object["vertices"], assigned_object["faces"], assigned_object["colors"],
                ))
                box_width = max(0.004, min(BOX_EDGE_WIDTH_M, float(np.min(bounds[1] - bounds[0])) * 0.04))
                box_vertices, box_faces, box_colors = _box_edges(bounds, BOX_COLOR, box_width)
                parts.append((box_vertices, box_faces, box_colors))
                label_vertices, label_faces, label_colors = _label_on_top(
                    bounds, label, box_width * 0.5 + 0.0001,
                )
                parts.append((label_vertices, label_faces, label_colors))
                scan_result["objects"].append(assigned_object["record"])
                included_count += 1
            scan_result["objects_assigned_to_other_levels"] = (
                len(objects_by_scan[scan_id]) - len(assigned_objects)
            )
            if skipped_classes_by_scan[scan_id]:
                scan_result["skipped_classes"] = skipped_classes_by_scan[scan_id]
            if scan_skip_reason.get(scan_id):
                scan_result["skip_reason"] = scan_skip_reason[scan_id]
            elif not objects_by_scan[scan_id] and not skipped_classes_by_scan[scan_id]:
                scan_result["skip_reason"] = "no_fixed_class_associations_for_objects"
            object_records.append(scan_result)

        vertices, faces, colors = _merge(parts)
        if (not np.array_equal(vertices[:len(base_vertices)], base_vertices)
                or not np.array_equal(faces[:len(base_faces)], base_faces)
                or not np.array_equal(colors[:len(base_colors)], base_colors)):
            raise RuntimeError(f"Source regularized mesh was not preserved in {level['floor_level_name']}")
        output_mesh = output_dir / f"{source_mesh.stem}_with_classified_objects.ply"
        _write_mesh(output_mesh, vertices, faces, colors)

        preview, _, _ = _project_topdown(vertices, faces, colors, "color", resolution)
        preview_path = output_dir / f"{source_mesh.stem}_classified_objects_topdown.png"
        if not cv2.imwrite(str(preview_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Could not write classified-object preview: {preview_path}")

        level_results.append({
            "floor_level_name": level["floor_level_name"],
            "source_regularized_mesh": str(source_mesh),
            "output_mesh_copy": str(output_mesh),
            "output_topdown_preview": str(preview_path),
            "included_object_count": included_count,
            "scans": object_records,
        })

    result = {
        "schema_version": 1,
        "source_floor_mesh_manifest": str(source_manifest),
        "assignment_method": "maximum_vertical_overlap_at_object_bbox_center_xz",
        "each_object_assigned_to_one_floor_level": True,
        "box_color_rgba": BOX_COLOR.tolist(),
        "levels": level_results,
    }
    manifest_path = output_dir / "classified_object_floor_slices_manifest.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(manifest_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR,
                        help="Directory containing floor_level_meshes_manifest.json and source PLYs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Separate directory for experimental copies and previews.")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Square top-down preview resolution (default: 1024).")
    args = parser.parse_args()
    generate_classified_object_slices(args.mesh_dir, args.output_dir, args.resolution)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
