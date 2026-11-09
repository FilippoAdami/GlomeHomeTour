#!/usr/bin/env python3
"""Check Phase 05 per-instance category predictions against Phase 08 floor assignments."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from generate_floor_level_meshes import _read_mesh


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_DIR = BACKEND_DIR / "current_scene"
DEFAULT_OUTPUT_DIR = DEFAULT_SCENE_DIR / "artifacts" / "instance_floor_distance_check"
OUTLIER_FRACTION = 0.10


def _clip_polygon(
    polygon: list[np.ndarray], axis: int, boundary: float, keep_above: bool,
) -> list[np.ndarray]:
    if not polygon:
        return []
    result = []
    previous = polygon[-1]
    previous_inside = previous[axis] >= boundary if keep_above else previous[axis] <= boundary
    for current in polygon:
        current_inside = current[axis] >= boundary if keep_above else current[axis] <= boundary
        if current_inside != previous_inside:
            ratio = (boundary - previous[axis]) / (current[axis] - previous[axis])
            result.append(previous + ratio * (current - previous))
        if current_inside:
            result.append(current)
        previous, previous_inside = current, current_inside
    return result


def _surface_coverage(vertices: np.ndarray, faces: np.ndarray, bounds: np.ndarray) -> float:
    triangles = vertices[faces].astype(np.float64, copy=False)
    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1,
    ) * 0.5
    total_area = float(areas.sum())
    if total_area <= 1e-12:
        inside_vertices = np.all((vertices >= bounds[0]) & (vertices <= bounds[1]), axis=1)
        return float(inside_vertices.mean())

    fully_inside = np.all((triangles >= bounds[0]) & (triangles <= bounds[1]), axis=(1, 2))
    fully_outside = (
        np.any(triangles.max(axis=1) < bounds[0], axis=1)
        | np.any(triangles.min(axis=1) > bounds[1], axis=1)
    )
    retained_area = float(areas[fully_inside].sum())
    for triangle, area in zip(triangles[~fully_inside & ~fully_outside], areas[~fully_inside & ~fully_outside]):
        polygon = list(triangle)
        for axis in range(3):
            polygon = _clip_polygon(polygon, axis, bounds[0, axis], True)
            polygon = _clip_polygon(polygon, axis, bounds[1, axis], False)
            if len(polygon) < 3:
                break
        if len(polygon) >= 3:
            origin = polygon[0]
            retained_area += sum(
                float(np.linalg.norm(np.cross(polygon[index] - origin, polygon[index + 1] - origin))) * 0.5
                for index in range(1, len(polygon) - 1)
            )
    return min(1.0, retained_area / total_area)


def _robust_bounds(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Fit per-axis quantiles with a 10% total budget weighted by incident triangle area."""
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("Expected a non-empty N x 3 vertex array")
    triangles = vertices[faces]
    triangle_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1,
    ) * 0.5
    weights = np.bincount(faces.ravel(), weights=np.repeat(triangle_areas / 3.0, 3), minlength=len(vertices))
    if not np.isfinite(weights).all() or weights.sum() <= 1e-12:
        weights = np.ones(len(vertices), dtype=np.float64)
    tail_weight = OUTLIER_FRACTION * float(weights.sum()) / 6.0
    bounds = np.empty((2, 3), dtype=np.float64)
    for axis in range(3):
        order = np.argsort(vertices[:, axis])
        cumulative = np.cumsum(weights[order])
        low_index = min(int(np.searchsorted(cumulative, tail_weight, side="left")), len(order) - 1)
        high_index = min(int(np.searchsorted(cumulative, weights.sum() - tail_weight, side="left")), len(order) - 1)
        bounds[0, axis] = vertices[order[low_index], axis]
        bounds[1, axis] = vertices[order[high_index], axis]
    full_bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
    coverage = _surface_coverage(vertices, faces, bounds)
    if coverage < 1.0 - OUTLIER_FRACTION:
        # Expand only as much as needed to retain 90% of triangle surface area.
        low_factor, high_factor = 0.0, 1.0
        for _ in range(12):
            factor = (low_factor + high_factor) * 0.5
            candidate = bounds + factor * (full_bounds - bounds)
            if _surface_coverage(vertices, faces, candidate) >= 1.0 - OUTLIER_FRACTION:
                high_factor = factor
            else:
                low_factor = factor
        bounds += high_factor * (full_bounds - bounds)
        coverage = _surface_coverage(vertices, faces, bounds)
    inside = np.all((vertices >= bounds[0]) & (vertices <= bounds[1]), axis=1)
    vertex_support = float(weights[inside].sum() / weights.sum())
    if coverage < 1.0 - OUTLIER_FRACTION - 1e-9:
        raise AssertionError(f"Robust box retained only {coverage:.3%} of triangle surface area")
    return bounds, coverage, vertex_support


def _plane(plane_record: dict) -> tuple[np.ndarray, float]:
    equation = np.asarray(plane_record["plane_equation"], dtype=np.float64)
    length = float(np.linalg.norm(equation[:3]))
    if equation.shape != (4,) or not np.isfinite(equation).all() or length < 1e-8:
        raise ValueError(f"Invalid floor plane: {plane_record.get('plane_id')}")
    normal, offset = equation[:3] / length, float(equation[3] / length)
    if normal[1] < 0:
        normal, offset = -normal, -offset
    if abs(normal[1]) < 1e-6:
        raise ValueError(f"Floor plane is vertical: {plane_record.get('plane_id')}")
    return normal, offset


def _floor_height(normal: np.ndarray, offset: float, x: float, z: float) -> float:
    return float(-(normal[0] * x + normal[2] * z + offset) / normal[1])


def _instance_number(instance_id: str) -> int:
    match = re.fullmatch(r"instance_(\d+)", instance_id)
    if not match:
        raise ValueError(f"Unexpected aligned instance ID: {instance_id}")
    # Phase 08 mesh names are one-based; Phase 05 association IDs are zero-based.
    return int(match.group(1)) - 1


def _object_key(scan_id: str, source_class_id: int, instance_id: int) -> tuple[str, int, int]:
    return scan_id, int(source_class_id), int(instance_id)


def _write_report(
    path: Path,
    phase05_report: Path,
    scores: dict,
    category: dict,
    categories: dict[int, dict],
    association_status: str,
    check_details: str,
    floor_details: str,
    scan_id: str,
    source_class_id: int,
    instance_id: int,
    check_label: str = "Floor check",
) -> None:
    with Image.open(phase05_report) as source:
        # Phase 05 uses this exact 1.25:1 left panel for its ten source views.
        collage_width = round(source.width * 1.25 / 2.25)

    score_by_id = {int(category_id): float(score) for category_id, score in scores.items()}
    rows = sorted(score_by_id.items(), key=lambda item: item[1], reverse=True)[:10]
    if not any(category_id == int(category["id"]) for category_id, _ in rows):
        selected_id = int(category["id"])
        rows = [(selected_id, score_by_id.get(selected_id, 0.0)), *rows[:9]]
    status_text = {
        "retained": "RETAINED",
        "rejected_floor_distance": "REJECTED",
        "retained_room_reassigned": "REASSIGNED",
        "rejected_room_conflict": "UNASSIGNED",
    }.get(association_status, "NOT EVALUATED")
    row_text, row_colors = [], []
    for category_id, score in rows:
        row_category = categories.get(category_id)
        is_associated = category_id == int(category["id"])
        row_text.append([
            row_category["name"] if row_category else f"category {category_id}",
            f"{score:+.3f}" if category_id in score_by_id else "—",
            status_text if is_associated else "candidate",
        ])
        row_color = ({
            "retained": "#d9ead3",
            "retained_room_reassigned": "#d9ead3",
            "rejected_floor_distance": "#f4cccc",
            "rejected_room_conflict": "#f4cccc",
        }.get(association_status, "#e8e8e8")) if is_associated else "#f2f2f2"
        row_colors.append([row_color] * 3)

    figure = plt.figure(figsize=(19, 10))
    left_width = 1.25 / 2.25
    table_axis = figure.add_axes([left_width + 0.01, 0.04, 0.99 - left_width, 0.70])
    table_axis.axis("off")
    report_text = (
        f"{scan_id} · class {source_class_id} / instance {instance_id}\n"
        f"Instance prediction: {category['name']} — {status_text}\n{floor_details}{check_details}"
    )
    report_text = "\n".join(
        textwrap.fill(line, width=92, break_long_words=False, break_on_hyphens=False) if line else ""
        for line in report_text.splitlines()
    )
    figure.text(
        left_width + 0.02, 0.97, report_text,
        fontsize=10, ha="left", va="top",
    )
    if row_text:
        table = table_axis.table(
            cellText=row_text,
            cellColours=row_colors,
            colLabels=["Category", "Instance score", check_label],
            loc="center",
            cellLoc="center",
            colWidths=[0.44, 0.25, 0.31],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 2.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    with Image.open(path) as output, Image.open(phase05_report) as source:
        canvas = output.convert("RGB")
        canvas.paste(source.crop((0, 0, collage_width, source.height)).convert("RGB"), (0, 0))
        canvas.save(path)


def check_instance_floor_distances(
    scene_dir: Path = DEFAULT_SCENE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    artifacts_dir: Path | None = None,
) -> dict:
    scene_dir, output_dir = Path(scene_dir).resolve(), Path(output_dir).resolve()
    artifacts_dir = Path(artifacts_dir or scene_dir / "artifacts").resolve()
    slice_path = artifacts_dir / "classified_object_slice_experiment" / "classified_object_floor_slices_manifest.json"
    floor_manifest_path = artifacts_dir / "floor_level_meshes" / "floor_level_meshes_manifest.json"
    for path in (slice_path, floor_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Phase 08 manifest is missing: {path}")

    slice_manifest = json.loads(slice_path.read_text(encoding="utf-8"))
    floor_manifest = json.loads(floor_manifest_path.read_text(encoding="utf-8"))
    metadata_path = Path(floor_manifest["source_metadata"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    planes_by_id = {item["plane_id"]: item for item in metadata["floor_planes"]}

    floors_by_scan_level = {}
    for level in floor_manifest["levels"]:
        level_name = level["floor_level_name"]
        for scan in level["scans"]:
            if scan["output_status"] != "generated_for_floor_level":
                continue
            planes = [planes_by_id[plane_id] for plane_id in scan.get("floor_plane_ids", [])
                      if plane_id in planes_by_id]
            if planes:
                floors_by_scan_level[(scan["scan_id"], level_name)] = planes

    slice_objects = {}
    for level in slice_manifest["levels"]:
        for scan in level["scans"]:
            scan_id = scan["scan_id"]
            for obj in scan.get("objects", []):
                key = _object_key(scan_id, obj["source_class_id"], _instance_number(obj["instance_id"]))
                if key in slice_objects:
                    raise ValueError(f"Phase 08 assigned one instance to multiple floor slices: {key}")
                slice_objects[key] = obj

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / "instance_reports"
    if report_dir.exists():
        shutil.rmtree(report_dir)
    results, skipped, source_associations = [], [], []
    skipped_uncategorized_count = skipped_door_window_count = 0

    for scan_dir in sorted(scene_dir.glob("scan_*/segmentation/non_structural_similarity")):
        scan_id = scan_dir.parents[1].name
        association_path = scan_dir / "valid_instance_associations.json"
        categories_path = scan_dir / "categories.json"
        if not association_path.is_file() or not categories_path.is_file():
            skipped.append({"scan_id": scan_id, "reason": "phase05_association_files_missing"})
            continue

        associations_data = json.loads(association_path.read_text(encoding="utf-8"))
        categories = {int(item["id"]): item for item in json.loads(categories_path.read_text(encoding="utf-8"))}
        for instance in associations_data.get("instances", []):
            source_class_id, instance_number = int(instance["source_class_id"]), int(instance["instance_id"])
            category_id = instance.get("instance_text_class_id")
            prediction_source = "instance_text_class_id"
            if category_id is None and instance.get("association_source") == "instance":
                category_id = instance.get("text_class_id")
                prediction_source = "text_class_id_from_instance"
            if instance.get("status") != "accepted" or category_id is None:
                skipped_uncategorized_count += 1
                continue
            category_id = int(category_id)
            category = categories.get(category_id)
            if not category:
                skipped.append({
                    "scan_id": scan_id, "source_class_id": source_class_id,
                    "instance_id": instance_number, "category_id": category_id,
                    "reason": "unknown_instance_prediction_category",
                })
                continue
            if category["name"].strip().lower() in {"door", "window"}:
                skipped_door_window_count += 1
                continue

            prediction = {
                "scan_id": scan_id,
                "source_class_id": source_class_id,
                "instance_id": instance_number,
                "category_id": category_id,
                "category": category["name"],
                "prediction_source": prediction_source,
            }
            source_associations.append(prediction)
            object_key = _object_key(scan_id, source_class_id, instance_number)
            assigned_object = slice_objects.get(object_key)
            limits_data = category.get("geometry_constraints", {}).get("base_z_m") if category else None
            record = {**prediction, "association_status": None}
            level_name = assigned_object.get("assigned_floor_level") if assigned_object else None
            record["assigned_floor_level"] = level_name
            reason = None
            floor_details = "Floor check unavailable"
            check_details = ""

            if not limits_data or len(limits_data) != 2:
                reason = "missing_category_floor_distance_limits"
            elif assigned_object is None:
                reason = "no_existing_phase08_floor_assignment"
            else:
                floor_planes = floors_by_scan_level.get((scan_id, level_name), [])
                if not floor_planes:
                    reason = "assigned_floor_plane_missing"
                else:
                    mesh_path = Path(assigned_object["aligned_mesh"])
                    if not mesh_path.is_absolute():
                        mesh_path = scene_dir.parent / mesh_path
                    record["aligned_mesh"] = str(mesh_path)
                    if not mesh_path.is_file():
                        reason = "aligned_instance_mesh_missing"
                    else:
                        vertices, faces, _ = _read_mesh(mesh_path)
                        bounds, surface_coverage, vertex_support = _robust_bounds(vertices, faces)
                        center_x = float((bounds[0, 0] + bounds[1, 0]) * 0.5)
                        center_z = float((bounds[0, 2] + bounds[1, 2]) * 0.5)
                        candidates = []
                        for floor_plane in floor_planes:
                            normal, offset = _plane(floor_plane)
                            floor_y = _floor_height(normal, offset, center_x, center_z)
                            candidates.append((abs(float(bounds[0, 1]) - floor_y), floor_plane, floor_y))
                        _, selected_plane, floor_y = min(candidates, key=lambda item: item[0])
                        distance = float(bounds[0, 1] - floor_y)
                        limits = (float(limits_data[0]), float(limits_data[1]))
                        retained = limits[0] <= distance <= limits[1]
                        bbox_size = bounds[1] - bounds[0]
                        dimension_limits = category.get("geometry_constraints", {})
                        record.update({
                            "association_status": "retained" if retained else "rejected_floor_distance",
                            "floor_plane_id": selected_plane["plane_id"],
                            "base_height_above_floor_m": distance,
                            "allowed_base_height_m": list(limits),
                            "robust_bbox_min_xyz": bounds[0].tolist(),
                            "robust_bbox_max_xyz": bounds[1].tolist(),
                            "bbox_surface_area_coverage_fraction": surface_coverage,
                            "bbox_area_weighted_vertex_support_fraction": vertex_support,
                        })
                        floor_details = (
                            f"Floor {level_name} ({selected_plane['plane_id']}) · "
                            f"base height {distance:.3f} m (valid {limits[0]:.3f}–{limits[1]:.3f} m)"
                        )
                        width_range = dimension_limits.get("width_m", [])
                        depth_range = dimension_limits.get("depth_m", [])
                        height_range = dimension_limits.get("height_m", [])
                        if len(width_range) == len(depth_range) == len(height_range) == 2:
                            check_details = (
                                f"\nBBox X {bbox_size[0]:.2f} m / Z {bbox_size[2]:.2f} m "
                                f"(valid W {width_range[0]:g}–{width_range[1]:g} m, "
                                f"D {depth_range[0]:g}–{depth_range[1]:g} m; either axis may swap)"
                                f"\nBBox Y {bbox_size[1]:.2f} m (valid H "
                                f"{height_range[0]:g}–{height_range[1]:g} m)"
                                f"\nBBox surface coverage: {surface_coverage:.1%}"
                            )
                        else:
                            check_details = (
                                f"\nBBox X/Y/Z: {bbox_size[0]:.2f} × {bbox_size[1]:.2f} × "
                                f"{bbox_size[2]:.2f} m\nBBox surface coverage: {surface_coverage:.1%}"
                            )

            if reason:
                record["association_status"] = f"not_evaluated_{reason}"
                record["not_evaluated_reason"] = reason
                floor_details = f"Floor check unavailable: {reason.replace('_', ' ')}"
                skipped.append({**prediction, "reason": reason})

            phase05_report = scan_dir / (instance.get("report_image") or "")
            if not instance.get("report_image") or not phase05_report.is_file():
                skipped.append({**prediction, "reason": "phase05_instance_report_missing"})
                continue
            report_name = (
                f"class_{source_class_id:03d}_inst_{instance_number:03d}_"
                f"category_{category_id:03d}_{category['name']}.png"
            )
            report_path = report_dir / scan_id / report_name
            _write_report(
                report_path, phase05_report, instance.get("category_mean_specificity") or {},
                category, categories, record["association_status"], check_details, floor_details,
                scan_id, source_class_id, instance_number,
            )
            record["report_image"] = str(report_path)
            results.append(record)

    retained = sum(item["association_status"] == "retained" for item in results)
    rejected = sum(item["association_status"] == "rejected_floor_distance" for item in results)
    evaluated = retained + rejected
    result_by_key = {
        (item["scan_id"], item["source_class_id"], item["instance_id"], item["category_id"]): item
        for item in results
    }
    removed_keys = {
        key for key, item in result_by_key.items()
        if item["association_status"] == "rejected_floor_distance"
    }
    filtered_associations, unevaluated_associations = [], []
    for association in source_associations:
        key = (
            association["scan_id"], association["source_class_id"],
            association["instance_id"], association["category_id"],
        )
        if key in removed_keys:
            continue
        result = result_by_key.get(key)
        outcome = result["association_status"] if result else "unevaluated_preserved"
        filtered_associations.append({**association, "floor_distance_status": outcome})
        if outcome.startswith("not_evaluated") or outcome == "unevaluated_preserved":
            unevaluated_associations.append({**association, "floor_distance_status": outcome})
    manifest = {
        "schema_version": 1,
        "input_instance_associations": str(scene_dir / "scan_*/segmentation/non_structural_similarity/valid_instance_associations.json"),
        "instance_prediction_source": "instance_text_class_id, or text_class_id when association_source is instance; class-propagated labels are not used",
        "phase08_floor_assignment_manifest": str(slice_path),
        "floor_mesh_manifest": str(floor_manifest_path),
        "floor_metadata": str(metadata_path),
        "floor_distance_method": "robust_bbox_min_y_minus_assigned_floor_plane_height_at_robust_bbox_xz_center",
        "robust_bbox_max_outside_surface_area_fraction": OUTLIER_FRACTION,
        "uncategorized_instances_skipped": skipped_uncategorized_count,
        "door_window_instances_skipped": skipped_door_window_count,
        "source_association_count": len(source_associations),
        "reported_instance_prediction_count": len(results),
        "evaluated_association_count": evaluated,
        "retained_association_count": retained,
        "rejected_association_count": rejected,
        "skipped_association_count": len(skipped),
        "association_count_after_filter": len(filtered_associations),
        "removed_associations": [item for item in results if item["association_status"] == "rejected_floor_distance"],
        "unevaluated_associations_preserved": unevaluated_associations,
        "associations_after_floor_check": filtered_associations,
        "associations": results,
        "skipped": skipped,
    }
    manifest_path = output_dir / "instance_floor_distance_manifest.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR,
                        help="Scene root containing scan folders and Phase 08 artifacts.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="New directory for the per-instance report manifest and images.")
    args = parser.parse_args()
    result = check_instance_floor_distances(args.scene_dir, args.output_dir)
    print(
        f"Evaluated {result['evaluated_association_count']} per-instance predictions: "
        f"{result['retained_association_count']} retained, "
        f"{result['rejected_association_count']} rejected, "
        f"{result['reported_instance_prediction_count']} reports written; "
        f"{result['door_window_instances_skipped']} door/window and "
        f"{result['uncategorized_instances_skipped']} uncategorized instances skipped."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
