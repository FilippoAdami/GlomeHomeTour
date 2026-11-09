"""Detached Phase 07 pipeline for regularized panels from accepted window instances."""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

WINDOW_TEXT_CLASS_ID = 2
WINDOW_PANEL_DEPTH_M = 0.10
WINDOW_DIMENSIONS_M = {
    "width": (0.30, 4.50),
    "depth": (0.03, 0.45),
    "height": (0.30, 3.00),
}
OVERLAP_FRACTION = 0.60
SILL_HEIGHT_STEP_M = 0.15
SILL_HEIGHT_BOUNDS_M = (0.0, 1.8)
DOOR_OVERLAP_EPSILON_M = 1e-6


def _load_floors(alignment_dir: Path, scan_id: str) -> list[tuple[Path, trimesh.Trimesh]]:
    floor_dir = alignment_dir / "scans" / scan_id / "mesh_elaboration" / "elements"
    floors = []
    for path in sorted(floor_dir.glob("floor_level_*.ply")):
        mesh = trimesh.load(path, force="mesh", process=False)
        if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces):
            floors.append((path, mesh))
    return floors


def _sample_floor(floors: list[tuple[Path, trimesh.Trimesh]], x: float, z: float, reference_y: float) -> dict[str, Any] | None:
    """Sample the closest same-scan Phase 06 floor under XZ, matching the door-panel method."""
    point = np.array([x, z], dtype=np.float64)
    samples = []
    for path, mesh in floors:
        triangles = np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(mesh.faces)]
        projected = triangles[:, :, [0, 2]]
        a, b, c = projected[:, 0], projected[:, 1], projected[:, 2]
        ab, ac, ap = b - a, c - a, point - a
        denominator = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
        valid = np.abs(denominator) > 1e-12
        wb, wc = np.zeros(len(triangles)), np.zeros(len(triangles))
        wb[valid] = (ap[valid, 0] * ac[valid, 1] - ap[valid, 1] * ac[valid, 0]) / denominator[valid]
        wc[valid] = (ab[valid, 0] * ap[valid, 1] - ab[valid, 1] * ap[valid, 0]) / denominator[valid]
        wa = 1.0 - wb - wc
        inside = valid & (wa >= -1e-7) & (wb >= -1e-7) & (wc >= -1e-7)
        if np.any(inside):
            weights = np.stack((wa, wb, wc), axis=1)[inside]
            elevation = float(np.median(np.sum(weights * triangles[inside, :, 1], axis=1)))
            samples.append({"elevation_m": elevation, "source_path": str(path), "method": "phase06_floor_triangle", "horizontal_distance_m": 0.0})
        else:
            centers = projected.mean(axis=1)
            nearest = int(np.argmin(np.linalg.norm(centers - point, axis=1)))
            samples.append({
                "elevation_m": float(np.median(triangles[nearest, :, 1])),
                "source_path": str(path), "method": "nearest_phase06_floor_triangle",
                "horizontal_distance_m": float(np.linalg.norm(centers[nearest] - point)),
            })
    if not samples:
        return None
    return min(samples, key=lambda item: (
        item["method"] != "phase06_floor_triangle", item["horizontal_distance_m"],
        abs(item["elevation_m"] - reference_y), item["source_path"],
    ))


def _aabb_overlap(first: np.ndarray, second: np.ndarray) -> dict[str, Any] | None:
    extent = np.minimum(first[1], second[1]) - np.maximum(first[0], second[0])
    if np.any(extent <= DOOR_OVERLAP_EPSILON_M):
        return None
    return {
        "intersection_bounds_xyz_m": [np.maximum(first[0], second[0]).tolist(), np.minimum(first[1], second[1]).tolist()],
        "intersection_extent_xyz_m": extent.tolist(),
        "intersection_volume_m3": float(np.prod(extent)),
    }


def _read_associations(scene_dir: Path) -> list[dict[str, Any]]:
    associations = []
    for scan_dir in sorted(scene_dir.glob("scan_*")):
        path = scan_dir / "segmentation" / "non_structural_similarity" / "valid_instance_associations.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        details = {
            (int(row["source_class_id"]), int(row["instance_id"])): row
            for row in data.get("instances", [])
        }
        for row in data.get("associations", []):
            if isinstance(row, dict):
                class_id = int(row.get("source_class_id", row.get("image_class_id")))
                instance_id = int(row.get("instance_id", 0))
                text_class_id = int(row.get("text_class_id", 0))
            else:
                class_id, instance_id, text_class_id = map(int, row[:3])
            if text_class_id == WINDOW_TEXT_CLASS_ID:
                associations.append({
                    "scan_id": scan_dir.name,
                    "class_id": class_id,
                    "instance_id": instance_id,
                    "instance_index": instance_id + 1,
                    "text_class_id": text_class_id,
                    "classification": details.get((class_id, instance_id), {}),
                })
    return associations


def _nearest_step(value: float, step: float) -> float:
    return round(float(np.floor(value / step + 0.5) * step), 2)


def _panel_box(panel: dict[str, Any], color: tuple[int, int, int, int]) -> trimesh.Trimesh:
    extents = [panel["depth"], panel["height"], panel["width"]]
    center_y = panel["bottom_y"] + panel["height"] / 2.0
    center = [panel["plane_offset"], center_y, panel["center_tangent"]]
    if panel["normal_axis"] == 2:
        extents = [panel["width"], panel["height"], panel["depth"]]
        center = [panel["center_tangent"], center_y, panel["plane_offset"]]
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
    return mesh


def _write_snapshot(path: Path, panels: list[dict[str, Any]], color: tuple[int, int, int, int]) -> None:
    mesh = trimesh.util.concatenate([_panel_box(panel, color) for panel in panels]) if panels else trimesh.Trimesh(process=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))


def _panel_state(panel: dict[str, Any]) -> dict[str, Any]:
    tangent_axis = 2 if panel["normal_axis"] == 0 else 0
    center = [0.0, panel["bottom_y"] + panel["height"] / 2.0, 0.0]
    center[panel["normal_axis"]] = panel["plane_offset"]
    center[tangent_axis] = panel["center_tangent"]
    return {
        "panel_index": panel["panel_index"],
        "source_instances": panel["source_instances"],
        "orientation": panel["orientation"],
        "center_xyz_m": center,
        "center_plane_offset_m": panel["plane_offset"],
        "floor_elevation_m": panel.get("floor_elevation_m"),
        "sill_height_above_floor_m": panel.get("sill_height_m"),
        "panel_bottom_elevation_m": panel["bottom_y"],
        "dimensions_m": {"width": panel["width"], "height": panel["height"], "depth": panel["depth"]},
        "merge_group_id": panel.get("merge_group_id"),
        "plane_group_id": panel.get("plane_group_id"),
        "overlap_group_id": panel.get("overlap_group_id"),
    }


def _overlap_ratio(first: dict[str, Any], second: dict[str, Any]) -> float:
    if first["normal_axis"] != second["normal_axis"]:
        return 0.0
    tangent_overlap = max(0.0, min(
        first["center_tangent"] + first["width"] / 2,
        second["center_tangent"] + second["width"] / 2,
    ) - max(
        first["center_tangent"] - first["width"] / 2,
        second["center_tangent"] - second["width"] / 2,
    ))
    vertical_overlap = max(0.0, min(
        first["bottom_y"] + first["height"],
        second["bottom_y"] + second["height"],
    ) - max(
        first["bottom_y"], second["bottom_y"],
    ))
    smaller_area = min(first["width"] * first["height"], second["width"] * second["height"])
    return tangent_overlap * vertical_overlap / smaller_area if smaller_area > 0 else 0.0


def _panel_from_instance(
    record: dict[str, Any], mesh: trimesh.Trimesh, panel_index: int,
    floor: dict[str, Any] | None, door_overlaps: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    span = bounds[1] - bounds[0]
    horizontal_spans = np.array([span[0], span[2]])
    normal_axis = int(np.argmin(horizontal_spans)) * 2
    tangent_axis = 2 if normal_axis == 0 else 0
    depth, width, height = float(span[normal_axis]), float(span[tangent_axis]), float(span[1])
    reasons = []
    for name, value in (("width", width), ("depth", depth), ("height", height)):
        minimum, maximum = WINDOW_DIMENSIONS_M[name]
        if not minimum <= value <= maximum:
            reasons.append(f"{name} {value:.3f}m outside [{minimum:.2f}, {maximum:.2f}]m")
    dimensions_valid = not reasons
    if door_overlaps:
        reasons.extend(f"AABB overlaps door {item['door_instance']['description']}" for item in door_overlaps)
    floor = floor or {
        "elevation_m": float(bounds[0, 1]), "source_path": None,
        "method": "source_bottom_fallback_missing_phase06_floor", "horizontal_distance_m": None,
    }
    details = {
        "source_instance": record,
        "bounds_xyz_m": bounds.tolist(),
        "inferred_normal_axis": "x" if normal_axis == 0 else "z",
        "raw_dimensions_m": {"width": width, "depth": depth, "height": height},
        "dimension_valid": dimensions_valid,
        "dimensions_within_bounds": dimensions_valid,
        "door_overlap_rejected": bool(door_overlaps),
        "eligible_for_panel_pipeline": dimensions_valid and not door_overlaps,
        "door_overlaps": door_overlaps,
        "floor_sample": floor,
        "source_sill_height_above_floor_m": float(bounds[0, 1] - floor["elevation_m"]),
        "exclusion_reasons": reasons,
    }
    if reasons:
        return None, details
    center_tangent = float((bounds[0, tangent_axis] + bounds[1, tangent_axis]) / 2.0)
    plane_offset = float((bounds[0, normal_axis] + bounds[1, normal_axis]) / 2.0)
    panel = {
        "panel_index": panel_index,
        "source_instances": [{
            "scan_id": record["scan_id"], "class_id": record["class_id"],
            "instance_id": record["instance_id"], "instance_index": record["instance_index"],
            "text_class_id": WINDOW_TEXT_CLASS_ID,
            "classification_status": record["classification"].get("status"),
            "association_source": record["classification"].get("association_source"),
            "source_instance_path": record["source_instance_path"],
            "aligned_instance_path": record["aligned_instance_path"],
            "raw_dimensions_m": details["raw_dimensions_m"],
            "original_bounds_xyz_m": details["bounds_xyz_m"],
            "source_floor_elevation_m": floor["elevation_m"],
            "source_floor_sample": floor,
            "source_sill_height_above_floor_m": details["source_sill_height_above_floor_m"],
        }],
        "normal_axis": normal_axis,
        "orientation": "normal_x" if normal_axis == 0 else "normal_z",
        "plane_offset": plane_offset,
        "center_tangent": center_tangent,
        "bottom_y": float(bounds[0, 1]),
        "floor_elevation_m": float(floor["elevation_m"]),
        "sill_height_m": details["source_sill_height_above_floor_m"],
        "width": width,
        "height": height,
        "depth": WINDOW_PANEL_DEPTH_M,
    }
    return panel, details


def generate_aligned_window_panel_artifacts(scene_dir: Path, alignment_dir: Path) -> tuple[Path, Path, int]:
    """Generate isolated, auditable panels from Phase 05 accepted window instances."""
    scene_dir, alignment_dir = Path(scene_dir), Path(alignment_dir)
    work_dir = alignment_dir / "window_panel_pipeline"
    stats_dir = alignment_dir / "stats"
    manifest_path = stats_dir / "aligned_objects_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_by_key = {}
    door_instances, door_mesh_errors = [], []
    for row in manifest.get("objects", []):
        if row.get("category") == "door" and int(row.get("text_class_id", 0)) == 1:
            door_path = Path(row.get("global_instance_path", ""))
            if not door_path.is_file():
                door_mesh_errors.append({"source_instance_path": row.get("source_instance_path"), "reason": f"aligned door mesh missing: {door_path}"})
            else:
                door_mesh = trimesh.load(door_path, force="mesh", process=False)
                if isinstance(door_mesh, trimesh.Trimesh) and len(door_mesh.faces) and np.isfinite(door_mesh.vertices).all():
                    door_instances.append({
                        "scan_id": row.get("scan_id"), "class_id": row.get("class_id"),
                        "text_class_id": row.get("text_class_id"),
                        "source_instance_path": row.get("source_instance_path"),
                        "aligned_instance_path": str(door_path),
                        "description": f"{row.get('scan_id')} class {row.get('class_id')} {Path(row.get('source_instance_path', door_path)).name}",
                        "bounds_xyz_m": np.asarray(door_mesh.bounds, dtype=np.float64),
                    })
                else:
                    door_mesh_errors.append({"source_instance_path": row.get("source_instance_path"), "reason": f"aligned door mesh empty or invalid: {door_path}"})
        try:
            instance_index = int(Path(row["source_instance_path"]).stem.removeprefix("instance_"))
        except (KeyError, ValueError):
            continue
        manifest_by_key[(str(row["scan_id"]), int(row["class_id"]), instance_index - 1)] = row

    candidates, missing = [], []
    floors_by_scan: dict[str, list[tuple[Path, trimesh.Trimesh]]] = {}
    for association in _read_associations(scene_dir):
        key = (association["scan_id"], association["class_id"], association["instance_id"])
        row = manifest_by_key.get(key)
        if row is None:
            missing.append({**association, "reason": "no aligned object manifest entry"})
            continue
        mesh_path = Path(row["global_instance_path"])
        if not mesh_path.is_file():
            missing.append({**association, "reason": f"aligned mesh missing: {mesh_path}"})
            continue
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            missing.append({**association, "reason": f"aligned mesh is empty or invalid: {mesh_path}"})
            continue
        enriched = {**association, "source_instance_path": row["source_instance_path"], "aligned_instance_path": str(mesh_path)}
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        if association["scan_id"] not in floors_by_scan:
            floors_by_scan[association["scan_id"]] = _load_floors(alignment_dir, association["scan_id"])
        floor = _sample_floor(
            floors_by_scan[association["scan_id"]],
            float((bounds[0, 0] + bounds[1, 0]) / 2), float((bounds[0, 2] + bounds[1, 2]) / 2), float(bounds[0, 1]),
        )
        door_overlaps = []
        for door in door_instances:
            overlap = _aabb_overlap(bounds, door["bounds_xyz_m"])
            if overlap:
                door_overlaps.append({"door_instance": {key: value for key, value in door.items() if key != "bounds_xyz_m"}, **overlap})
        panel, details = _panel_from_instance(enriched, mesh, len(candidates), floor, door_overlaps)
        candidates.append({"panel": panel, "details": details, "mesh": mesh})

    candidate_meshes = []
    for item in candidates:
        mesh = item["mesh"].copy()
        color = [75, 185, 115, 255] if item["panel"] else [215, 75, 75, 255]
        mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
        candidate_meshes.append(mesh)
    raw_mesh_path = work_dir / "00_semantic_window_instances.ply"
    raw_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    (trimesh.util.concatenate(candidate_meshes) if candidate_meshes else trimesh.Trimesh(process=False)).export(str(raw_mesh_path))
    candidate_report_path = work_dir / "00_candidate_report.json"
    candidate_report = {
        "accepted_text_class_id": WINDOW_TEXT_CLASS_ID,
        "source": "Phase 05 valid_instance_associations.json plus Phase 07 aligned_objects_manifest.json",
        "dimension_bounds_m": WINDOW_DIMENSIONS_M,
        "door_overlap_filter": "reject any window whose aligned axis-aligned bounding box has positive 3D intersection with an aligned door instance AABB",
        "door_mesh_errors": door_mesh_errors,
        "candidate_count": len(candidates),
        "dimension_valid_count": sum(item["details"]["dimension_valid"] for item in candidates),
        "door_overlap_rejected_count": sum(item["details"]["door_overlap_rejected"] for item in candidates),
        "eligible_panel_count": sum(item["panel"] is not None for item in candidates),
        "floor_meshes_loaded_by_scan": {scan_id: [str(path) for path, _ in floors] for scan_id, floors in floors_by_scan.items()},
        "missing_instances": missing,
        "candidates": [item["details"] for item in candidates],
    }
    candidate_report_path.write_text(json.dumps(candidate_report, indent=2) + "\n", encoding="utf-8")

    panels = [item["panel"] for item in candidates if item["panel"] is not None]
    valid_path = work_dir / "01_eligible_window_panels.ply"
    _write_snapshot(valid_path, panels, (80, 170, 205, 255))
    stages: dict[str, Any] = {"candidate_filter": candidate_report}

    # Duplicate merge: complete-link groups require <=25 cm face gap and >=15% overlap.
    groups: list[list[dict[str, Any]]] = []
    merge_checks = []
    for i, panel in enumerate(panels):
        for earlier in panels[:i]:
            face_gap = max(0.0, abs(panel["plane_offset"] - earlier["plane_offset"]) - WINDOW_PANEL_DEPTH_M)
            overlap = _overlap_ratio(panel, earlier)
            merge_checks.append({
                "panel_a": earlier["panel_index"], "panel_b": panel["panel_index"],
                "face_to_face_gap_m": face_gap, "smaller_panel_overlap_fraction": overlap,
                "eligible": panel["normal_axis"] == earlier["normal_axis"] and face_gap <= 0.25 + 1e-8 and overlap >= 0.15 - 1e-8,
            })
        group = next((group for group in groups if all(
            member["normal_axis"] == panel["normal_axis"]
            and max(0.0, abs(member["plane_offset"] - panel["plane_offset"]) - WINDOW_PANEL_DEPTH_M) <= 0.25 + 1e-8
            and _overlap_ratio(member, panel) >= 0.15 - 1e-8 for member in group
        )), None)
        if group is None:
            groups.append([panel])
        else:
            group.append(panel)

    merged = []
    merge_groups = []
    for group_index, group in enumerate(groups):
        winner = max(group, key=lambda item: (item["width"] * item["height"], item["panel_index"]))
        panel = dict(winner)
        panel["source_instances"] = [src for item in group for src in item["source_instances"]]
        panel["plane_offset"] = float(np.mean([item["plane_offset"] for item in group]))
        panel["center_tangent"] = float(np.mean([item["center_tangent"] for item in group]))
        panel["bottom_y"] = float(np.median([item["bottom_y"] for item in group]))
        panel["floor_elevation_m"] = float(np.median([src["source_floor_elevation_m"] for src in panel["source_instances"]]))
        panel["sill_height_m"] = float(np.median([src["source_sill_height_above_floor_m"] for src in panel["source_instances"]]))
        panel["merge_group_id"] = f"merge_{group_index:03d}"
        merged.append(panel)
        merge_groups.append({"merge_group_id": panel["merge_group_id"], "panel_indices": [item["panel_index"] for item in group], "source_instances": panel["source_instances"], "dimensions_m": {"width": panel["width"], "height": panel["height"]}})
    merge_path = work_dir / "02_duplicate_merged_panels.ply"
    _write_snapshot(merge_path, merged, (220, 160, 65, 255))
    stages["duplicate_merge"] = {
        "maximum_face_to_face_gap_m": 0.25, "minimum_smaller_panel_overlap_fraction": 0.15,
        "grouping_method": "complete-link", "pair_checks": merge_checks, "groups": merge_groups,
        "panels_after": [_panel_state(panel) for panel in merged],
    }

    # Plane snapping groups parallel center offsets within 15 cm; keep the largest whole panel.
    plane_groups: list[list[dict[str, Any]]] = []
    for normal_axis in (0, 2):
        ordered = sorted((panel for panel in merged if panel["normal_axis"] == normal_axis), key=lambda p: p["plane_offset"])
        for panel in ordered:
            group = next((group for group in reversed(plane_groups) if group[0]["normal_axis"] == normal_axis and panel["plane_offset"] - group[0]["plane_offset"] <= 0.15 + 1e-8), None)
            if group is None:
                plane_groups.append([panel])
            else:
                group.append(panel)
    plane_report = []
    for group_index, group in enumerate(plane_groups):
        target = float(np.mean([panel["plane_offset"] for panel in group]))
        winner = max(group, key=lambda p: (p["width"] * p["height"], p["panel_index"]))
        group_id = f"plane_{group_index:03d}"
        for panel in group:
            panel["plane_group_id"] = group_id
            panel["plane_offset_before_plane_snap_m"] = panel["plane_offset"]
            panel["plane_offset"] = target
            panel["width_before_snap_m"], panel["height_before_snap_m"] = winner["width"], winner["height"]
            panel["size_source_panel_index"] = winner["panel_index"]
            panel["width"] = winner["width"]
            panel["height"] = winner["height"]
        plane_report.append({
            "plane_group_id": group_id, "panel_indices": [panel["panel_index"] for panel in group],
            "source_instances": [src for panel in group for src in panel["source_instances"]],
            "plane_offsets_before_m": [panel["plane_offset_before_plane_snap_m"] for panel in group],
            "plane_offset_after_m": target, "size_source_panel_index": winner["panel_index"],
            "shared_dimensions_m": {"width": winner["width"], "height": winner["height"]},
        })
    plane_path = work_dir / "03_planes_snapped.ply"
    _write_snapshot(plane_path, merged, (165, 105, 195, 255))
    stages["plane_snap"] = {"maximum_center_plane_offset_range_m": 0.15, "groups": plane_report, "panels_after": [_panel_state(panel) for panel in merged]}

    # Keep each panel's local floor, while giving coplanar panels one 15 cm sill height.
    sill_groups = []
    for group in plane_groups:
        source_sills = [src["source_sill_height_above_floor_m"] for panel in group for src in panel["source_instances"]]
        raw_sill = float(np.median(source_sills))
        sill = float(np.clip(_nearest_step(raw_sill, SILL_HEIGHT_STEP_M), *SILL_HEIGHT_BOUNDS_M))
        changes = []
        for panel in group:
            panel["bottom_y_before_floor_alignment_m"] = panel["bottom_y"]
            panel_floors = [src["source_floor_elevation_m"] for src in panel["source_instances"]]
            panel["floor_elevation_m"] = float(np.median(panel_floors))
            panel["sill_height_m"] = sill
            panel["bottom_y"] = panel["floor_elevation_m"] + sill
            changes.append({
                "panel_index": panel["panel_index"], "floor_elevation_m": panel["floor_elevation_m"],
                "bottom_before_m": panel["bottom_y_before_floor_alignment_m"],
                "bottom_after_m": panel["bottom_y"], "sill_height_m": sill,
                "source_floor_samples": [src["source_floor_sample"] for src in panel["source_instances"]],
            })
        sill_groups.append({
            "plane_group_id": group[0]["plane_group_id"],
            "panel_indices": [panel["panel_index"] for panel in group],
            "source_sill_heights_m": source_sills, "median_source_sill_height_m": raw_sill,
            "shared_sill_height_m": sill, "sill_step_m": SILL_HEIGHT_STEP_M, "panels": changes,
        })
    sill_path = work_dir / "04_floor_sill_aligned_panels.ply"
    _write_snapshot(sill_path, merged, (85, 185, 170, 255))
    stages["floor_sill_alignment"] = {
        "floor_source": "same-scan Phase 06 mesh_elaboration/elements/floor_level_*.ply; sampled at each instance center in XZ",
        "coplanar_grouping_source": "plane_snap groups",
        "shared_sill_step_m": SILL_HEIGHT_STEP_M,
        "sill_height_bounds_m": SILL_HEIGHT_BOUNDS_M,
        "grouping_method": "median source sill within each snapped plane group, rounded to nearest 15 cm; each panel keeps its own sampled floor elevation",
        "groups": sill_groups, "panels_after": [_panel_state(panel) for panel in merged],
    }

    # Size snap matches door behavior (5 cm, then 20 cm); floor-relative sill stays fixed.
    for panel in merged:
        panel["width_before_global_size_m"] = panel["width"]
        panel["height_before_global_size_m"] = panel["height"]
        panel["width_snap_5cm_m"] = _nearest_step(panel["width"], 0.05)
        panel["height_snap_5cm_m"] = _nearest_step(panel["height"], 0.05)
        panel["width"] = min(4.5, max(0.3, _nearest_step(panel["width_snap_5cm_m"], 0.20)))
        panel["height"] = min(3.0, max(0.3, _nearest_step(panel["height_snap_5cm_m"], 0.20)))
    dimension_path = work_dir / "05_dimensions_regularized.ply"
    _write_snapshot(dimension_path, merged, (85, 150, 220, 255))
    stages["dimension_snap"] = {
        "order": [0.05, 0.20], "panel_depth_m": WINDOW_PANEL_DEPTH_M,
        "panels_after": [_panel_state(panel) for panel in merged],
    }

    # Final pass uses modes from all dimension-valid merged panels, then aligns qualifying groups in-plane.
    width_counts, height_counts = Counter(round(panel["width"], 2) for panel in merged), Counter(round(panel["height"], 2) for panel in merged)
    common_width = min((value for value, count in width_counts.items() if count == max(width_counts.values())), default=None)
    common_height = min((value for value, count in height_counts.items() if count == max(height_counts.values())), default=None)
    overlap_adjacency = {i: set() for i in range(len(merged))}
    overlap_pairs = []
    for i, first in enumerate(merged):
        for j in range(i + 1, len(merged)):
            second = merged[j]
            ratio = _overlap_ratio(first, second)
            eligible = first["normal_axis"] == second["normal_axis"] and ratio >= OVERLAP_FRACTION - 1e-8
            overlap_pairs.append({
                "panel_indices": [first["panel_index"], second["panel_index"]],
                "overlap_fraction_before": ratio,
                "eligible": eligible,
                "center_plane_separation_m": abs(first["plane_offset"] - second["plane_offset"]),
                "center_plane_separation_preserved": True,
            })
            if eligible:
                overlap_adjacency[i].add(j)
                overlap_adjacency[j].add(i)
    overlap_groups = []
    unseen = {index for index, neighbors in overlap_adjacency.items() if neighbors}
    # shortcut: overlap chains share one target; use complete-link grouping if endpoints over-align.
    while unseen:
        seed = min(unseen)
        component, stack = set(), [seed]
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(overlap_adjacency[current] - component)
        unseen -= component
        members = sorted(component)
        target_t = float(np.mean([merged[index]["center_tangent"] for index in members]))
        group_id = f"overlap_{len(overlap_groups):03d}"
        for index in members:
            panel = merged[index]
            panel["overlap_group_id"] = group_id
            panel["center_tangent_before_overlap_m"] = panel["center_tangent"]
            panel["center_tangent"] = target_t
            panel["width"] = float(common_width)
            panel["height"] = float(common_height)
        overlap_groups.append({
            "overlap_group_id": group_id, "panel_indices": [merged[index]["panel_index"] for index in members],
            "source_instances": [src for index in members for src in merged[index]["source_instances"]],
            "target_center_tangent_m": target_t,
            "mode_dimensions_m": {"width": common_width, "height": common_height},
            "center_plane_offsets_preserved_m": [merged[index]["plane_offset"] for index in members],
            "floor_relative_sill_heights_preserved_m": [merged[index]["sill_height_m"] for index in members],
            "bottom_elevations_preserved_m": [merged[index]["bottom_y"] for index in members],
        })
    final_path = work_dir / "06_overlap_regularized_panels.ply"
    _write_snapshot(final_path, merged, (190, 205, 220, 255))
    shutil.copy2(final_path, work_dir / "05_overlap_regularized_panels.ply")  # Preserve the prior final-snapshot path.
    stages["final_overlap"] = {
        "minimum_projected_overlap_fraction": OVERLAP_FRACTION,
        "center_plane_distance_limit_m": None,
        "dimension_mode_source": "all dimension-valid, non-door-overlapping panels after duplicate, plane and size regularization",
        "global_dimension_modes_m": {"width": {"mode_m": common_width, "distribution": dict(width_counts)}, "height": {"mode_m": common_height, "distribution": dict(height_counts)}},
        "grouping_method": "connected components of qualifying parallel-panel overlap pairs",
        "pair_checks": overlap_pairs, "groups": overlap_groups,
        "panels_after": [_panel_state(panel) for panel in merged],
    }

    # Publish aliases and full per-instance association metadata; only this detached pipeline writes them.
    panel_path = alignment_dir / "aligned_window_panels.ply"
    shutil.copy2(final_path, panel_path)
    metadata_path = stats_dir / "aligned_window_panels.json"
    report_path = stats_dir / "window_panel_regularization_report.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    final_panels = []
    for panel in merged:
        tangent_axis = 2 if panel["normal_axis"] == 0 else 0
        center_xyz = [0.0, panel["bottom_y"] + panel["height"] / 2.0, 0.0]
        center_xyz[panel["normal_axis"]] = panel["plane_offset"]
        center_xyz[tangent_axis] = panel["center_tangent"]
        final_panels.append({
            "panel_index": panel["panel_index"], "source_instances": panel["source_instances"],
            "orientation": panel["orientation"], "panel_center_xyz_m": center_xyz,
            "panel_dimensions_m": {"width": panel["width"], "height": panel["height"], "depth": panel["depth"]},
            "merge_group_id": panel.get("merge_group_id"), "plane_group_id": panel.get("plane_group_id"),
            "overlap_group_id": panel.get("overlap_group_id"),
            "center_plane_offset_before_m": panel.get("plane_offset_before_plane_snap_m", panel["plane_offset"]),
            "center_plane_offset_after_m": panel["plane_offset"],
            "floor_elevation_m": panel["floor_elevation_m"],
            "sill_height_above_floor_m": panel["sill_height_m"],
            "panel_bottom_elevation_m": panel["bottom_y"],
        })
    report = {
        "schema_version": 1,
        "pipeline_scope": "independent Phase 07 window panel artifact pipeline; rejects window AABB intersections with door AABBs, aligns coplanar sill heights to 15 cm increments using Phase 06 floors, and does not modify door, scene, sealed-mesh, or Gaussian outputs",
        "panel_depth_m": WINDOW_PANEL_DEPTH_M,
        "window_dimension_bounds_m": WINDOW_DIMENSIONS_M,
        "stages": stages,
        "final_panels": final_panels,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps({
        "schema_version": 1,
        "verification_ply": str(panel_path),
        "pipeline_report": str(report_path),
        "intermediate_artifact_directory": str(work_dir),
        "panels": final_panels,
    }, indent=2) + "\n", encoding="utf-8")
    # Also publish one JSON beside each intermediate PLY so each stage is independently inspectable.
    stage_files = {
        "00_candidate_report.json": candidate_report,
        "02_duplicate_merge_report.json": stages["duplicate_merge"],
        "03_plane_snap_report.json": stages["plane_snap"],
        "04_floor_sill_alignment_report.json": stages["floor_sill_alignment"],
        "05_dimension_snap_report.json": stages["dimension_snap"],
        "06_final_overlap_report.json": stages["final_overlap"],
    }
    for name, data in stage_files.items():
        (work_dir / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    (work_dir / "01_eligible_window_panels.json").write_text(json.dumps({"panels": [_panel_state(panel) for panel in panels]}, indent=2) + "\n", encoding="utf-8")
    return panel_path, metadata_path, len(final_panels)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--alignment-dir", type=Path, default=None)
    args = parser.parse_args()
    alignment_dir = args.alignment_dir or args.workspace / "multi_room_alignment"
    output, metadata, count = generate_aligned_window_panel_artifacts(args.workspace, alignment_dir)
    print(f"Generated {count} aligned window panels: {output}")
    print(f"Metadata: {metadata}")
