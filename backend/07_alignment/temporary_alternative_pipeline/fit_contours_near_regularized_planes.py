#!/usr/bin/env python3
"""Move filtered contours toward shell planes under vertex and floor constraints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


ALIGNMENT_DIR = Path(__file__).resolve().parents[1]
CONTACT_DIR = ALIGNMENT_DIR / "temporary_object_contact_contours"
sys.path.insert(0, str(ALIGNMENT_DIR))
sys.path.insert(0, str(CONTACT_DIR))

from detect_large_outside_surface_patches import closest_surface_scene, room_center  # noqa: E402
from align_contours_to_regularized_planes import CONTACT_EPSILON_M  # noqa: E402
from discover_contact_contours import write_combined_instances, write_instance  # noqa: E402
from fit_contours_inside_regularized_mesh import (  # noqa: E402
    load_colored_mesh,
    red_face_mask,
    surface_outward_signs,
    classify_shell_points,
)


DEFAULT_ALIGNMENT_DIR = (
    ALIGNMENT_DIR.parent / "current_scene" / "multi_room_alignment"
    / "temporary_alternative_pipeline"
)
MAX_TRANSLATION_M = 0.15
OUTSIDE_FRACTION_LIMIT = 0.05
MAX_MEDIAN_ADJUSTMENT_M = 0.15
MAX_BELOW_FLOOR_FRACTION = 0.0
MAX_PLANE_TRANSLATIONS = 2
FLOOR_NUMERICAL_TOLERANCE_M = 1e-6
FLOOR_CLEARANCE_M = 1e-4
FLOOR_AXIS = 1
COARSE_STEP_M = 0.001
REFINE_STEP_M = 0.0001
REFINE_RADIUS_M = 0.001
MAX_QUERY_POINTS = 100_000
IMPROVEMENT_EPSILON_M = 1e-8


def load_shell(path: Path):
    mesh = trimesh.load(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"Expected a non-empty regularized mesh: {path}")
    scene, axes, centers, normals = closest_surface_scene(mesh)
    signs, paired_count = surface_outward_signs(
        room_center(mesh), axes, centers, normals
    )
    return scene, axes, centers, signs, paired_count


def lowest_floor_height(regularized_path: Path) -> float:
    floor_paths = sorted((regularized_path.parent / "elements").glob(
        "floor_level_000_area_*.ply"
    ))
    if not floor_paths:
        raise FileNotFoundError(
            f"Missing lowest regularized floor elements beside {regularized_path}"
        )
    heights = []
    for path in floor_paths:
        floor = trimesh.load(path, process=False)
        if not isinstance(floor, trimesh.Trimesh) or not len(floor.faces):
            raise ValueError(f"Invalid regularized floor element: {path}")
        upward = floor.face_normals[:, FLOOR_AXIS] > 0.9
        if not upward.any():
            raise ValueError(f"Floor element has no upward-facing surface: {path}")
        heights.extend(floor.triangles_center[upward, FLOOR_AXIS])
    heights = np.asarray(heights, dtype=np.float64)
    floor_y = float(np.median(heights))
    if np.percentile(np.abs(heights - floor_y), 99) > 0.03:
        raise ValueError(
            f"Lowest floor is not a flat Y-up plane within 3 cm: {regularized_path}"
        )
    return floor_y


def unique_contour_vertices(vertices: np.ndarray, faces: np.ndarray,
                            contour: np.ndarray) -> np.ndarray:
    if not contour.any():
        return np.empty((0, 3), dtype=np.float64)
    return np.unique(vertices[faces[contour]].reshape(-1, 3), axis=0)


def candidate_metrics(samples: np.ndarray, contour_vertices: np.ndarray,
                      candidates: list[tuple[int, int, float]],
                      face_count: int, scene, axes: np.ndarray,
                      outward_signs: np.ndarray,
                      sorted_object_y: np.ndarray,
                      floor_y: float) -> list[dict]:
    """Score centroid distances and constrain unique vertices of red contour faces."""
    flat_samples = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
    flat_contour_vertices = np.asarray(contour_vertices, dtype=np.float32).reshape(-1, 3)
    if not face_count:
        return [{
            "axis": axis,
            "direction": direction,
            "distance": amount,
            "average_distance": 0.0,
            "outside_triangle_count": 0,
            "outside_contour_vertex_count": 0,
            "constraint_outside_count": 0,
            "constraint_sample_count": int(len(flat_contour_vertices)),
            "floor_below_count": int(np.searchsorted(
                sorted_object_y,
                floor_y - FLOOR_NUMERICAL_TOLERANCE_M
                - (direction * amount if axis == FLOOR_AXIS else 0.0),
                side="left",
            )),
            "floor_point_count": int(len(sorted_object_y)),
        } for axis, direction, amount in candidates]
    queries_per_candidate = len(flat_samples) + len(flat_contour_vertices)
    candidates_per_batch = max(1, MAX_QUERY_POINTS // queries_per_candidate)
    results = []
    for start in range(0, len(candidates), candidates_per_batch):
        batch = candidates[start:start + candidates_per_batch]
        offsets = np.zeros((len(batch), 3), dtype=np.float32)
        rows = np.arange(len(batch))
        offsets[rows, [item[0] for item in batch]] = [
            direction * distance for _, direction, distance in batch
        ]
        moved_samples = flat_samples[None, :, :] + offsets[:, None, :]
        moved_contour_vertices = flat_contour_vertices[None, :, :] + offsets[:, None, :]
        moved = np.concatenate((moved_samples, moved_contour_vertices), axis=1).reshape(-1, 3)
        outside, signed_distance, _, _ = classify_shell_points(moved, scene, axes, outward_signs)
        outside = outside.reshape(len(batch), queries_per_candidate)
        outside_samples = outside[:, :len(flat_samples)].reshape(len(batch), face_count, 4)
        outside_faces = outside_samples[:, :, :3].any(axis=2).sum(axis=1)
        outside_vertices = outside[:, len(flat_samples):].sum(axis=1)
        signed_samples = signed_distance.reshape(len(batch), queries_per_candidate)
        distances = np.abs(
            signed_samples[:, :len(flat_samples)].reshape(len(batch), face_count, 4)[:, :, 3]
        )
        for index, (axis, direction, amount) in enumerate(batch):
            constraint_count = outside_vertices[index]
            constraint_sample_count = len(flat_contour_vertices)
            vertical_delta = direction * amount if axis == FLOOR_AXIS else 0.0
            floor_below_count = int(np.searchsorted(
                sorted_object_y,
                floor_y - FLOOR_NUMERICAL_TOLERANCE_M - vertical_delta,
                side="left",
            ))
            results.append({
                "axis": axis,
                "direction": direction,
                "distance": amount,
                "average_distance": float(distances[index].mean()),
                "outside_triangle_count": int(outside_faces[index]),
                "outside_contour_vertex_count": int(outside_vertices[index]),
                "constraint_outside_count": int(constraint_count),
                "constraint_sample_count": constraint_sample_count,
                "floor_below_count": floor_below_count,
                "floor_point_count": int(len(sorted_object_y)),
            })
    return results


def feasible(result: dict, outside_fraction_limit: float,
             max_below_floor_fraction: float = MAX_BELOW_FLOOR_FRACTION) -> bool:
    if "_member_metrics" in result:
        return all(feasible(member, outside_fraction_limit, max_below_floor_fraction)
                   for member in result["_member_metrics"])
    maximum_outside = int(np.floor(
        result["constraint_sample_count"] * outside_fraction_limit + 1e-9
    ))
    maximum_below_floor = int(np.floor(
        result["floor_point_count"] * max_below_floor_fraction + 1e-9
    ))
    return (result["constraint_outside_count"] <= maximum_outside
            and result["floor_below_count"] <= maximum_below_floor)


def best_by_direction(results: list[dict], outside_fraction_limit: float) -> dict[int, dict]:
    best = {}
    for result in results:
        if not feasible(result, outside_fraction_limit):
            continue
        key = result["axis"] * 2 + (result["direction"] > 0)
        previous = best.get(key)
        sort_key = (result["average_distance"], result["distance"])
        if previous is None or sort_key < (
            previous["average_distance"], previous["distance"]
        ):
            best[key] = result
    return best


def find_group_translation(items: list[dict], outside_fraction_limit: float
                           ) -> tuple[np.ndarray, dict, list[dict], list[dict], list[dict]]:
    """Apply up to two improving axis-only translations to a contact group."""
    if not items:
        raise ValueError("Cannot fit an empty contact group")
    def evaluate(candidate_list: list[tuple[int, int, float]],
                 current: np.ndarray) -> list[dict]:
        member_rows = [candidate_metrics(
            item["samples"] + current, item["contour_vertices"] + current,
            candidate_list, int(item["contour"].sum()),
            item["scene"], item["axes"], item["signs"],
            item["sorted_object_y"] + current[FLOOR_AXIS], item["floor_y"],
        ) for item in items]
        combined = []
        for candidate_index, candidate in enumerate(candidate_list):
            rows = [values[candidate_index] for values in member_rows]
            weighted_distance = sum(
                row["average_distance"] * int(item["contour"].sum())
                for item, row in zip(items, rows)
            )
            face_count = sum(int(item["contour"].sum()) for item in items)
            aggregate = {
                "axis": candidate[0],
                "direction": candidate[1],
                "distance": candidate[2],
                "average_distance": weighted_distance / face_count if face_count else 0.0,
                "outside_triangle_count": sum(row["outside_triangle_count"] for row in rows),
                "outside_contour_vertex_count": sum(
                    row["outside_contour_vertex_count"] for row in rows
                ),
                "constraint_outside_count": sum(row["constraint_outside_count"] for row in rows),
                "constraint_sample_count": sum(row["constraint_sample_count"] for row in rows),
                "floor_below_count": sum(row["floor_below_count"] for row in rows),
                "floor_point_count": sum(row["floor_point_count"] for row in rows),
                "_member_metrics": rows,
            }
            combined.append(aggregate)
        return combined

    translation = np.zeros(3, dtype=np.float64)
    baseline = evaluate([(0, 1, 0.0)], translation)[0]
    selected = baseline
    steps = []
    feasible_candidate_count = 0
    used_axes = set()
    for _ in range(MAX_PLANE_TRANSLATIONS):
        candidates = [
            (axis, direction, round(step * COARSE_STEP_M, 9))
            for axis in range(3) if axis not in used_axes
            for direction in (-1, 1)
            for step in range(1, round(MAX_TRANSLATION_M / COARSE_STEP_M) + 1)
        ]
        coarse = evaluate(candidates, translation)
        coarse_best = best_by_direction(coarse, outside_fraction_limit)
        refine_candidates = []
        for result in coarse_best.values():
            low = max(0.0, result["distance"] - REFINE_RADIUS_M)
            high = min(MAX_TRANSLATION_M, result["distance"] + REFINE_RADIUS_M)
            start = int(np.ceil(low / REFINE_STEP_M - 1e-9))
            stop = int(np.floor(high / REFINE_STEP_M + 1e-9))
            for step in range(start, stop + 1):
                candidate = (
                    result["axis"], result["direction"], round(step * REFINE_STEP_M, 9)
                )
                refine_candidates.append(candidate)
        refined = evaluate(refine_candidates, translation) if refine_candidates else []
        feasible_candidates = [row for row in coarse + refined
                               if feasible(row, outside_fraction_limit)]
        feasible_candidate_count += len(feasible_candidates)
        eligible = [row for row in [selected] + feasible_candidates
                    if feasible(row, outside_fraction_limit)]
        if not eligible:
            break
        next_selected = min(eligible, key=lambda row: (
            row["average_distance"], row["distance"], row["axis"], row["direction"]
        ))
        if (feasible(selected, outside_fraction_limit)
                and next_selected["average_distance"] >= selected["average_distance"]
                - IMPROVEMENT_EPSILON_M):
            break
        axis = next_selected["axis"]
        delta = next_selected["direction"] * next_selected["distance"]
        translation[axis] += delta
        used_axes.add(axis)
        steps.append({
            "axis": axis,
            "direction": next_selected["direction"],
            "distance_m": next_selected["distance"],
        })
        selected = next_selected

    member_metrics = selected["_member_metrics"]
    metadata = {
        "baseline_average_distance_m": baseline["average_distance"],
        "selected_average_distance_m": selected["average_distance"],
        "outside_triangle_count_before": baseline["outside_triangle_count"],
        "outside_triangle_count_after": selected["outside_triangle_count"],
        "outside_contour_vertex_count_before": baseline["outside_contour_vertex_count"],
        "outside_contour_vertex_count_after": selected["outside_contour_vertex_count"],
        "constraint_outside_count_before": baseline["constraint_outside_count"],
        "constraint_outside_count_after": selected["constraint_outside_count"],
        "constraint_sample_count": selected["constraint_sample_count"],
        "floor_below_count_before": baseline["floor_below_count"],
        "floor_below_count_after": selected["floor_below_count"],
        "floor_point_count": selected["floor_point_count"],
        "feasible_translation_found": feasible(selected, outside_fraction_limit),
        "feasible_candidate_count": feasible_candidate_count,
        "coarse_step_m": COARSE_STEP_M,
        "refine_step_m": REFINE_STEP_M,
    }
    return translation, metadata, baseline["_member_metrics"], member_metrics, steps


def target_wall_plane(samples: np.ndarray, translation: np.ndarray, scene,
                      face_axes: np.ndarray
                      ) -> tuple[tuple[int, int, int] | None, float | None]:
    moved_axes = np.flatnonzero(np.abs(translation) > 1e-9)
    if len(moved_axes) != 1 or moved_axes[0] == FLOOR_AXIS:
        return None, None
    axis = int(moved_axes[0])
    direction = int(np.sign(translation[axis]))
    centroids = samples[:, 3, :] + translation
    rays = np.zeros((len(centroids), 6), dtype=np.float32)
    rays[:, :3] = centroids
    rays[:, 3 + axis] = direction
    hits = scene.cast_rays(o3d.core.Tensor(rays))
    distances = hits["t_hit"].numpy()
    nearest = hits["primitive_ids"].numpy().astype(np.int64)
    eligible = np.isfinite(distances) & (distances > CONTACT_EPSILON_M)
    hit_samples = np.flatnonzero(eligible)
    eligible[hit_samples] &= face_axes[nearest[hit_samples]] == axis
    if not eligible.any():
        return None, None
    values = centroids[eligible, axis] + direction * distances[eligible]
    bins, counts = np.unique(np.rint(values / 0.10).astype(np.int64), return_counts=True)
    dominant_bin = int(bins[np.argmax(counts)])
    plane_coordinate = float(np.median(values[np.rint(values / 0.10) == dominant_bin]))
    return (axis, direction, dominant_bin), plane_coordinate


def wall_group_medians(items: list[dict]) -> dict[tuple[str, int, int, int], float]:
    grouped: dict[tuple[str, int, int, int], list[float]] = {}
    for item in items:
        key = item["wall_group_key"]
        if key is not None and item["translation_magnitude_m"] > 1e-9:
            grouped.setdefault(key, []).append(item["translation_magnitude_m"])
    return {key: float(np.median(values)) for key, values in grouped.items()}


def check_wall_group_medians() -> None:
    items = [
        {"wall_group_key": ("scan_0", 0, 1, 12), "translation_magnitude_m": 0.02},
        {"wall_group_key": ("scan_0", 0, 1, 12), "translation_magnitude_m": 0.08},
        {"wall_group_key": ("scan_1", 0, 1, 12), "translation_magnitude_m": 0.10},
    ]
    medians = wall_group_medians(items)
    assert medians[("scan_0", 0, 1, 12)] == 0.05
    assert medians[("scan_1", 0, 1, 12)] == 0.10


def run(alignment_dir: Path, *,
        outside_fraction_limit: float = OUTSIDE_FRACTION_LIMIT,
        source_report_path: Path | None = None,
        output_dir_name: str = "nearest_plane_fit") -> dict:
    alignment_dir = alignment_dir.resolve()
    if not 0.0 <= outside_fraction_limit <= 1.0:
        raise ValueError("outside_fraction_limit must be between 0 and 1")
    output_name = Path(output_dir_name)
    if output_name.is_absolute() or ".." in output_name.parts:
        raise ValueError(f"Invalid output directory name: {output_dir_name!r}")
    if source_report_path is None:
        source_report_path = alignment_dir / "contour_inside_fit" / "contour_inside_fit_report.json"
    source_report_path = source_report_path.resolve()
    source_dir = source_report_path.parent
    if not source_report_path.is_file():
        raise FileNotFoundError(f"Run contour shell fitting first: {source_report_path}")
    source_report = json.loads(source_report_path.read_text())
    output_dir = alignment_dir / output_dir_name
    shell_by_scan = {}
    pending = []
    check_wall_group_medians()

    for source_record in source_report["objects"]:
        scan_name = source_record["scan_id"]
        relative_path = Path(source_record["output_path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Invalid source object path: {source_record['output_path']!r}")
        source_path = source_dir / relative_path
        mesh = load_colored_mesh(source_path)
        contour = red_face_mask(mesh)
        if int(contour.sum()) != source_record["contour_triangle_count"]:
            raise ValueError(f"Contour colors do not match source report: {source_path}")

        regularized_path = Path(source_record["regularized_mesh_path"])
        if scan_name not in shell_by_scan:
            if not regularized_path.is_file():
                raise FileNotFoundError(f"Missing aligned regularized mesh: {regularized_path}")
            scene, axes, centers, signs, paired_count = load_shell(regularized_path)
            floor_y = lowest_floor_height(regularized_path)
            shell_by_scan[scan_name] = (
                scene, axes, centers, signs, regularized_path, paired_count, floor_y
            )
        scene, axes, centers, signs, regularized_path, _, floor_y = shell_by_scan[scan_name]

        vertices = np.asarray(mesh.vertices).copy()
        faces = np.asarray(mesh.triangles)
        sorted_object_y = np.sort(np.unique(vertices, axis=0)[:, FLOOR_AXIS])
        floor_below_before = int(np.searchsorted(
            sorted_object_y, floor_y - FLOOR_NUMERICAL_TOLERANCE_M, side="left"
        ))
        samples = np.empty((0, 4, 3), dtype=np.float64)
        contour_vertices = np.empty((0, 3), dtype=np.float64)
        if contour.any():
            triangles = vertices[faces[np.flatnonzero(contour)]]
            samples = np.concatenate(
                (triangles, triangles.mean(axis=1, keepdims=True)), axis=1
            )
            contour_vertices = unique_contour_vertices(vertices, faces, contour)
        pending.append({
            "scan_id": scan_name,
            "contact_group_id": source_record.get(
                "contact_group_id", f"{scan_name}:{source_record['source_instance_path']}"
            ),
            "source_instance_path": source_record["source_instance_path"],
            "source_inside_fit_path": str(source_path),
            "regularized_mesh_path": str(regularized_path),
            "source_path": source_path,
            "mesh": mesh,
            "vertices": vertices,
            "faces": faces,
            "contour": contour,
            "samples": samples,
            "contour_vertices": contour_vertices,
            "sorted_object_y": sorted_object_y,
            "floor_below_count_before": floor_below_before,
            "floor_y": floor_y,
            "scene": scene,
            "axes": axes,
            "signs": signs,
            "translation": np.zeros(3, dtype=np.float64),
            "initial_translation": np.zeros(3, dtype=np.float64),
            "metrics": {},
            "wall_target": None,
            "wall_coordinate": None,
            "wall_group_key": None,
            "translation_magnitude_m": 0.0,
            "median_adjustment": "not_grouped",
            "median_adjustment_reason": None,
            "median_wall_translation_m": None,
        })

    contact_groups: dict[str, list[dict]] = {}
    for item in pending:
        contact_groups.setdefault(item["contact_group_id"], []).append(item)

    group_representatives = []
    for group_id, members in contact_groups.items():
        floor_raise = max(
            max(0.0, item["floor_y"] + FLOOR_CLEARANCE_M
                - float(item["sorted_object_y"][0]))
            for item in members
        )
        floor_correction = np.array([0.0, floor_raise, 0.0], dtype=np.float64)
        for item in members:
            item["floor_correction_translation"] = floor_correction.copy()
            item["vertices"] += floor_correction
            item["samples"] += floor_correction
            item["contour_vertices"] += floor_correction
            item["sorted_object_y"] += floor_raise
        (translation, group_metrics, baseline_rows, selected_rows,
         translation_steps) = find_group_translation(members, outside_fraction_limit)
        for item, baseline, selected in zip(members, baseline_rows, selected_rows):
            item["translation"] = translation.copy()
            item["initial_translation"] = translation.copy()
            item["median_adjustment_translation"] = np.zeros(3, dtype=np.float64)
            item["translation_steps"] = [dict(step) for step in translation_steps]
            item["metrics"] = {
                "baseline_average_distance_m": baseline["average_distance"],
                "selected_average_distance_m": selected["average_distance"],
                "outside_triangle_count_before": baseline["outside_triangle_count"],
                "outside_triangle_count_after": selected["outside_triangle_count"],
                "outside_contour_vertex_count_before": baseline["outside_contour_vertex_count"],
                "outside_contour_vertex_count_after": selected["outside_contour_vertex_count"],
                "constraint_outside_count_before": baseline["constraint_outside_count"],
                "constraint_outside_count_after": selected["constraint_outside_count"],
                "constraint_sample_count": selected["constraint_sample_count"],
                "floor_below_count_before": baseline["floor_below_count"],
                "floor_below_count_after": selected["floor_below_count"],
                "floor_point_count": selected["floor_point_count"],
                "feasible_translation_found": feasible(selected, outside_fraction_limit),
                "feasible_candidate_count": group_metrics["feasible_candidate_count"],
                "coarse_step_m": COARSE_STEP_M,
                "refine_step_m": REFINE_STEP_M,
            }
            item["metrics"]["floor_below_count_before"] = item["floor_below_count_before"]
            item["translation_magnitude_m"] = float(np.linalg.norm(translation))
        samples = np.concatenate(
            [item["samples"] for item in members if len(item["samples"])], axis=0
        ) if any(len(item["samples"]) for item in members) else np.empty((0, 4, 3))
        first = members[0]
        group_scans = {item["scan_id"] for item in members}
        wall_steps = []
        cumulative = np.zeros(3, dtype=np.float64)
        if len(group_scans) == 1 and len(samples):
            for step_index, step in enumerate(translation_steps):
                increment = np.zeros(3, dtype=np.float64)
                increment[step["axis"]] = step["direction"] * step["distance_m"]
                wall_target, wall_coordinate = target_wall_plane(
                    samples + cumulative, increment, first["scene"], first["axes"],
                )
                if wall_target is not None:
                    wall_steps.append({
                        "step_index": step_index,
                        "wall_group_key": (first["scan_id"], *wall_target),
                        "translation_magnitude_m": step["distance_m"],
                        "wall_target": wall_target,
                        "wall_coordinate_m": wall_coordinate,
                    })
                cumulative += increment
        for wall_step in wall_steps:
            for item in members:
                item["translation_steps"][wall_step["step_index"]].update({
                    "wall_target": wall_step["wall_target"],
                    "wall_coordinate_m": wall_step["wall_coordinate_m"],
                    "wall_group_key": wall_step["wall_group_key"],
                })
        wall_target = wall_steps[0]["wall_target"] if wall_steps else None
        wall_coordinate = wall_steps[0]["wall_coordinate_m"] if wall_steps else None
        group_key = wall_steps[0]["wall_group_key"] if wall_steps else None
        for item in members:
            item["wall_target"] = wall_target
            item["wall_coordinate"] = wall_coordinate
            item["wall_group_key"] = group_key
        group_representatives.append({
            "contact_group_id": group_id,
            "wall_group_key": group_key,
            "translation_magnitude_m": members[0]["translation_magnitude_m"],
            "wall_steps": wall_steps,
            "floor_correction_magnitude_m": floor_raise,
        })

    wall_step_representatives = [
        step for representative in group_representatives
        for step in representative["wall_steps"]
    ]
    group_medians = wall_group_medians(wall_step_representatives)
    for representative in group_representatives:
        group_id = representative["contact_group_id"]
        members = contact_groups[group_id]
        for wall_step in representative["wall_steps"]:
            median = group_medians[wall_step["wall_group_key"]]
            current = wall_step["translation_magnitude_m"]
            axis, direction, _ = wall_step["wall_target"]
            for item in members:
                item["median_wall_translation_m"] = median
            if median <= current + 1e-9:
                for item in members:
                    item["median_adjustment"] = "already_at_or_above_median"
                continue
            delta = median - current
            candidate = (axis, direction, delta)
            trials = [candidate_metrics(
                item["samples"] + item["translation"],
                item["contour_vertices"] + item["translation"], [candidate],
                int(item["contour"].sum()), item["scene"], item["axes"],
                item["signs"],
                item["sorted_object_y"] + item["translation"][FLOOR_AXIS],
                item["floor_y"],
            )[0] for item in members]
            reasons = []
            if delta > MAX_MEDIAN_ADJUSTMENT_M + 1e-9:
                reasons.append("15_cm_median_adjustment_limit")
            if any(not feasible(trial, outside_fraction_limit) for trial in trials):
                reasons.append(
                    f"{int(round(outside_fraction_limit * 100))}_percent_contour_vertex_limit"
                )
            if any(trial["floor_below_count"] > 0 for trial in trials):
                reasons.append("zero_percent_floor_limit")
            if reasons:
                for item in members:
                    item["median_adjustment"] = "blocked_by_translation_or_geometry_limit"
                    item["median_adjustment_reason"] = reasons
                continue
            adjustment = np.zeros(3, dtype=np.float64)
            adjustment[axis] = direction * delta
            wall_step["translation_magnitude_m"] = median
            for item, trial in zip(members, trials):
                item["translation"] += adjustment
                item["median_adjustment_translation"] += adjustment
                item["translation_magnitude_m"] = float(np.linalg.norm(item["translation"]))
                item["median_adjustment"] = "applied"
                metrics = item["metrics"]
                metrics["selected_average_distance_m"] = trial["average_distance"]
                metrics["outside_triangle_count_after"] = trial["outside_triangle_count"]
                metrics["outside_contour_vertex_count_after"] = trial[
                    "outside_contour_vertex_count"
                ]
                metrics["constraint_outside_count_after"] = trial["constraint_outside_count"]
                metrics["constraint_sample_count"] = trial["constraint_sample_count"]
                metrics["floor_below_count_after"] = trial["floor_below_count"]

    output_meshes, records = [], []
    for item in pending:
        nearest_translation = item["translation"]
        plane_translation = item["initial_translation"]
        median_translation = item["median_adjustment_translation"]
        floor_correction = item["floor_correction_translation"]
        translation = floor_correction + nearest_translation
        nearest_magnitude = float(np.linalg.norm(plane_translation))
        magnitude = float(np.linalg.norm(translation))
        samples = item["samples"]
        metrics = item["metrics"]
        if len(samples):
            flat_samples = samples.reshape(-1, 3) + nearest_translation
            contour_vertices = item["contour_vertices"] + nearest_translation
            all_samples = np.concatenate((flat_samples, contour_vertices), axis=0)
            outside, signed_distance, _, _ = classify_shell_points(
                all_samples, item["scene"], item["axes"], item["signs"]
            )
            outside_samples = outside[:len(flat_samples)].reshape(-1, 4)
            outside_triangle_count = int(outside_samples.any(axis=1).sum())
            outside_vertex_count = int(outside[len(flat_samples):].sum())
            average_distance = float(np.abs(
                signed_distance[:len(flat_samples)].reshape(-1, 4)[:, 3]
            ).mean())
            constraint_outside_count = outside_vertex_count
            constraint_sample_count = len(item["contour_vertices"])
            maximum_outside = int(np.floor(
                constraint_sample_count * outside_fraction_limit + 1e-9
            ))
            floor_below_after = int(np.searchsorted(
                item["sorted_object_y"],
                item["floor_y"] - FLOOR_NUMERICAL_TOLERANCE_M
                - nearest_translation[FLOOR_AXIS],
                side="left",
            ))
            maximum_below_floor = int(np.floor(
                len(item["sorted_object_y"]) * MAX_BELOW_FLOOR_FRACTION + 1e-9
            ))
            if metrics["feasible_translation_found"] and (
                constraint_outside_count > maximum_outside
                or floor_below_after > maximum_below_floor
            ):
                raise RuntimeError(f"Final object translation violates a hard constraint: {item['source_path']}")
            if abs(average_distance - metrics["selected_average_distance_m"]) > 1e-6:
                raise RuntimeError(f"Selected pose distance changed on final check: {item['source_path']}")
            metrics["outside_triangle_count_after"] = outside_triangle_count
            metrics["outside_contour_vertex_count_after"] = outside_vertex_count
            metrics["constraint_outside_count_after"] = constraint_outside_count
            metrics["constraint_sample_count"] = constraint_sample_count
            metrics["floor_below_count_after"] = floor_below_after
            metrics["feasible_translation_found"] = (
                constraint_outside_count <= maximum_outside
                and floor_below_after <= maximum_below_floor
            )
        else:
            floor_below_after = int(np.searchsorted(
                item["sorted_object_y"],
                item["floor_y"] - FLOOR_NUMERICAL_TOLERANCE_M
                - nearest_translation[FLOOR_AXIS],
                side="left",
            ))
            metrics["floor_below_count_after"] = floor_below_after

        step_axes = [step["axis"] for step in item["translation_steps"]]
        if (len(item["translation_steps"]) > MAX_PLANE_TRANSLATIONS
                or len(step_axes) != len(set(step_axes))
                or any(step["distance_m"] > MAX_TRANSLATION_M + 1e-9
                       for step in item["translation_steps"])):
            raise RuntimeError(
                f"Invalid nearest-plane translation steps for {item['source_path']}: "
                f"{item['translation_steps']}"
            )
        mesh = item["mesh"]
        mesh.vertices = o3d.utility.Vector3dVector(item["vertices"] + nearest_translation)
        target = (output_dir / "objects" / item["scan_id"]
                  / item["source_path"].parent.name
                  / f"{item['source_path'].stem}_nearest_plane_fit.ply")
        write_instance(target, mesh, item["contour"])
        output_meshes.append(target)
        record = {
            "scan_id": item["scan_id"],
            "contact_group_id": item["contact_group_id"],
            "contact_group_size": len(contact_groups[item["contact_group_id"]]),
            "source_instance_path": item["source_instance_path"],
            "source_inside_fit_path": item["source_inside_fit_path"],
            "regularized_mesh_path": item["regularized_mesh_path"],
            "output_path": str(target.relative_to(output_dir)),
            "contour_triangle_count": int(item["contour"].sum()),
            "contour_vertex_count": int(len(item["contour_vertices"])),
            **metrics,
            "floor_height_m": item["floor_y"],
            "floor_below_fraction_before": (
                metrics["floor_below_count_before"] / metrics["floor_point_count"]
                if metrics["floor_point_count"] else 0.0
            ),
            "floor_below_fraction_after": (
                metrics["floor_below_count_after"] / metrics["floor_point_count"]
                if metrics["floor_point_count"] else 0.0
            ),
            "floor_guard_satisfied": metrics["floor_below_count_after"] <= int(np.floor(
                metrics["floor_point_count"] * MAX_BELOW_FLOOR_FRACTION + 1e-9
            )),
            "outside_fraction_before": (
                metrics["constraint_outside_count_before"] / metrics["constraint_sample_count"]
                if metrics["constraint_sample_count"] else 0.0
            ),
            "outside_fraction_after": (
                metrics["constraint_outside_count_after"] / metrics["constraint_sample_count"]
                if metrics["constraint_sample_count"] else 0.0
            ),
            "translation_m": translation.tolist(),
            "floor_correction_translation_m": floor_correction.tolist(),
            "nearest_plane_translation_m": plane_translation.tolist(),
            "translation_magnitude_m": magnitude,
            "nearest_plane_translation_magnitude_m": nearest_magnitude,
            "median_adjustment_translation_m": median_translation.tolist(),
            "median_adjustment_magnitude_m": float(np.linalg.norm(median_translation)),
            "translation_axis": (
                "".join(dict.fromkeys("XYZ"[axis] for axis in step_axes))
                if step_axes else ("Y" if np.any(np.abs(floor_correction) > 1e-9) else None)
            ),
            "nearest_plane_translation_steps": [
                {"axis": "XYZ"[step["axis"]],
                 "direction": step["direction"],
                 "distance_m": step["distance_m"],
                 "wall_plane_coordinate_m": step.get("wall_coordinate_m")}
                for step in item["translation_steps"]
            ],
            "wall_plane_axis": (
                "XYZ"[item["wall_target"][0]] if item["wall_target"] is not None else None
            ),
            "wall_plane_direction": (
                item["wall_target"][1] if item["wall_target"] is not None else None
            ),
            "wall_plane_coordinate_m": item["wall_coordinate"],
            "median_wall_translation_m": item["median_wall_translation_m"],
            "median_adjustment": item["median_adjustment"],
            "median_adjustment_reason": item["median_adjustment_reason"],
        }
        records.append(record)

    if not output_meshes:
        raise ValueError("No shell-fitted objects were found")
    for group_id, members in contact_groups.items():
        first_total = members[0]["floor_correction_translation"] + members[0]["translation"]
        if any(not np.allclose(
            first_total,
            item["floor_correction_translation"] + item["translation"],
            atol=1e-12,
        ) for item in members[1:]):
            raise RuntimeError(f"Touching instances received different translations: {group_id}")
    combined_path = output_dir / "aligned_objects_contours_near_regularized_planes.ply"
    combined_triangles = write_combined_instances(output_meshes, combined_path)
    largest = max(records, key=lambda item: item["translation_magnitude_m"])
    largest_nearest_plane = max(
        records, key=lambda item: item["nearest_plane_translation_magnitude_m"]
    )
    distances_before = [item["baseline_average_distance_m"] for item in records
                        if item["baseline_average_distance_m"] is not None]
    distances_after = [item["selected_average_distance_m"] for item in records
                       if item["selected_average_distance_m"] is not None]
    floor_guard_violations = sum(not item["floor_guard_satisfied"] for item in records)
    wall_group_sizes = {
        f"{scan}:axis{'XYZ'[axis]}:{direction:+d}:bin{plane_bin}": sum(
            item["wall_group_key"] == key for item in wall_step_representatives
        )
        for key in group_medians
        for scan, axis, direction, plane_bin in [key]
    }
    report = {
        "status": "contours_moved_toward_nearest_regularized_plane",
        "maximum_outside_fraction": outside_fraction_limit,
        "outside_measure": "unique vertex positions belonging to red contour triangles",
        "maximum_plane_translation_per_axis_m": MAX_TRANSLATION_M,
        "maximum_additional_wall_median_translation_m": MAX_MEDIAN_ADJUSTMENT_M,
        "maximum_below_floor_fraction": MAX_BELOW_FLOOR_FRACTION,
        "floor_axis": "Y",
        "floor_guard_point_definition": (
            "Unique object mesh vertex positions, including non-contour geometry; any point more "
            "than 1 micrometer below the lowest floor plane counts as below-floor."
        ),
        "floor_correction_rule": (
            "Before plane fitting, move every contact group rigidly upward by the smallest shared "
            "Y translation that places all member vertices at least 0.1 mm above their respective "
            "lowest floor planes. This floor recovery is a separate axis-only translation."
        ),
        "touching_instance_rule": (
            "Instances connected by surface contact within 1 mm in the globally aligned scene share one "
            "translation through both fitting stages; each member's contour and floor limits "
            "are checked independently."
        ),
        "wall_median_adjustment": (
            "Per scan, group only instances translated toward the same detected X/Z wall plane and "
            f"direction. Extend shorter pulls to that group's median only when the {outside_fraction_limit:.0%} "
            "red-contour unique-vertex limit, zero-percent lowest-floor limit, and separate additional "
            "15 cm per-plane translation limit remain satisfied. Cross-scan contact groups are not median-adjusted."
        ),
        "wall_median_group_count": len(group_medians),
        "wall_median_group_sizes": wall_group_sizes,
        "objects_raised_to_wall_median": sum(item["median_adjustment"] == "applied" for item in records),
        "objects_blocked_from_wall_median": sum(
            item["median_adjustment"] == "blocked_by_translation_or_geometry_limit"
            for item in records
        ),
        "wall_median_block_reasons": {
            reason: sum(reason in (item["median_adjustment_reason"] or []) for item in records)
            for reason in (
                "15_cm_median_adjustment_limit",
                f"{int(round(outside_fraction_limit * 100))}_percent_contour_vertex_limit",
                "zero_percent_floor_limit",
            )
        },
        "floor_guard_violations_after": floor_guard_violations,
        "translation_rule": (
            "First translate each contact group upward as needed so none of its unique object vertices "
            "falls below the lowest floor. Then apply up to two sequential signed principal-axis "
            "translations on separate axes, each axis independently limited to 15 cm, "
            "re-evaluating after the first move. Select only moves "
            "that reduce the group's mean absolute nearest-shell-plane distance of red triangle "
            f"centroids, with at most {outside_fraction_limit:.0%} of unique red-contour vertex "
            "positions outside the shell. The subsequent per-plane median pull has a separate "
            "15 cm additional translation limit. "
            "Distances and outside status are recomputed against the nearest shell surface "
            "for every candidate."
        ),
        "rotation_applied": False,
        "regularized_mesh_modified": False,
        "source_report": str(source_report_path),
        "combined_objects_ply": combined_path.name,
        "combined_triangle_count": combined_triangles,
        "object_count": len(records),
        "contact_group_count": len(contact_groups),
        "touching_contact_group_count": sum(len(items) > 1 for items in contact_groups.values()),
        "contact_group_translation_consistent": True,
        "floor_correction_group_count": sum(
            item["floor_correction_magnitude_m"] > 1e-9 for item in group_representatives
        ),
        "maximum_floor_correction_m": max(
            (item["floor_correction_magnitude_m"] for item in group_representatives),
            default=0.0,
        ),
        "objects_moved": sum(item["translation_magnitude_m"] > 1e-9 for item in records),
        "maximum_translation_m": largest["translation_magnitude_m"],
        "maximum_nearest_plane_translation_m": (
            largest_nearest_plane["nearest_plane_translation_magnitude_m"]
        ),
        "maximum_axis_translation_step_m": max(
            (step["distance_m"] for item in records
             for step in item["nearest_plane_translation_steps"]),
            default=0.0,
        ),
        "average_contour_distance_before_m": float(np.mean(distances_before))
            if distances_before else None,
        "average_contour_distance_after_m": float(np.mean(distances_after))
            if distances_after else None,
        "outside_triangle_count_before": sum(item["outside_triangle_count_before"] for item in records),
        "outside_triangle_count_after": sum(item["outside_triangle_count_after"] for item in records),
        "outside_contour_vertex_count_before": sum(
            item["outside_contour_vertex_count_before"] for item in records
        ),
        "outside_contour_vertex_count_after": sum(
            item["outside_contour_vertex_count_after"] for item in records
        ),
        "constraint_outside_count_before": sum(
            item["constraint_outside_count_before"] for item in records
        ),
        "constraint_outside_count_after": sum(
            item["constraint_outside_count_after"] for item in records
        ),
        "objects_over_outside_limit_before": sum(
            item["outside_fraction_before"] > outside_fraction_limit for item in records
        ),
        "objects_over_outside_limit_after": sum(
            item["outside_fraction_after"] > outside_fraction_limit for item in records
        ),
        "objects_over_floor_limit_before": sum(
            item["floor_below_fraction_before"] > MAX_BELOW_FLOOR_FRACTION
            for item in records
        ),
        "objects_with_unresolved_hard_constraints": sum(
            not item["feasible_translation_found"] for item in records
        ),
        "paired_internal_surface_face_count_by_scan": {
            scan: values[5] for scan, values in shell_by_scan.items()
        },
        "largest_translation_instance": {
            "scan_id": largest["scan_id"],
            "source_instance_path": largest["source_instance_path"],
            "translation_m": largest["translation_m"],
        },
        "largest_nearest_plane_translation_instance": {
            "scan_id": largest_nearest_plane["scan_id"],
            "source_instance_path": largest_nearest_plane["source_instance_path"],
            "translation_m": largest_nearest_plane["nearest_plane_translation_m"],
        },
        "objects": records,
    }
    (output_dir / "nearest_plane_fit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-dir", type=Path, default=DEFAULT_ALIGNMENT_DIR)
    args = parser.parse_args(argv)
    report = run(args.alignment_dir)
    print(json.dumps({
        "status": report["status"],
        "objects": report["object_count"],
        "objects_moved": report["objects_moved"],
        "maximum_translation_m": report["maximum_translation_m"],
        "maximum_axis_translation_step_m": report["maximum_axis_translation_step_m"],
        "average_contour_distance_before_m": report["average_contour_distance_before_m"],
        "average_contour_distance_after_m": report["average_contour_distance_after_m"],
        "outside_triangle_count_after": report["outside_triangle_count_after"],
        "outside_contour_vertex_count_after": report["outside_contour_vertex_count_after"],
        "constraint_outside_count_after": report["constraint_outside_count_after"],
        "objects_over_outside_limit_after": report["objects_over_outside_limit_after"],
        "combined_objects_ply": str(args.alignment_dir / "nearest_plane_fit"
                                     / report["combined_objects_ply"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
