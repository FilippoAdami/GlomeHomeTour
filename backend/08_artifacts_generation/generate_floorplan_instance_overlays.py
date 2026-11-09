#!/usr/bin/env python3
"""Add final categorized-instance SVGs to copies of regularized floor views."""
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from generate_floor_level_meshes import _read_mesh
from generate_floor_level_projections import (
    SMALL_FOOTPRINT_HOLE_MAX_AREA_M2,
    _fill_small_footprint_holes,
    _projection_grid,
)
from generate_floorplan_symbols import CLASS_NAMES, SVG_NS, sized_svg


DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "current_scene" / "artifacts"
LOWER_FLOOR_OPACITY = 0.35
TRIANGLE_BATCH_SIZE = 50_000
MAX_CONTOUR_TRANSLATION_M = 0.25
MAX_OVERLAP_TRANSLATION_M = 0.40
FLOOR_COVERAGE_HIDE_FRACTION = 0.10
WALL_SEGMENT_SNAP_MAX_DISTANCE_M = 0.25
WALL_SEGMENT_PARALLEL_DOT_MIN = 0.9
WALL_ADHERENCE_MAX_GAP_M = 0.25
SAME_CATEGORY_MERGE_GAP_M = 0.05
COMPLETE_OVERLAP_COVERAGE = 0.85
VERTICAL_OVERLAP_HEIGHT_TOLERANCE_M = 0.05
# shortcut: passages stop 1 m from the fitted door plane; raise if valid scan gaps exceed this.
MAX_DOOR_PASSAGE_REACH_M = 1.0
VECTOR_GRAY_STEP = 8
WALL_ORIENTED_CATEGORIES = {18, 23, 24, 25, 26}
FIXED_FOOTPRINT_CATEGORIES = {7, 12, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27}


def _rasterize_triangles(
    vertices: np.ndarray,
    faces: np.ndarray,
    resolution: int,
    grid: dict,
) -> np.ndarray:
    covered = np.zeros((resolution, resolution), dtype=np.uint8)
    for start in range(0, len(faces), TRIANGLE_BATCH_SIZE):
        triangles = vertices[faces[start:start + TRIANGLE_BATCH_SIZE]]
        x_pixels = np.rint(
            (triangles[:, :, 0] - grid["x_min"]) / grid["pixel_size"] - 0.5
        ).astype(np.int32)
        z_pixels = np.rint(
            (grid["z_max"] - triangles[:, :, 2]) / grid["pixel_size"] - 0.5
        ).astype(np.int32)
        polygons = np.stack((x_pixels, z_pixels), axis=2)
        cv2.fillPoly(
            covered,
            [polygon.reshape(-1, 1, 2) for polygon in polygons],
            255,
        )
    return covered > 0


def _scan_footprint_masks(
    vertices: np.ndarray,
    faces: np.ndarray,
    scans: list[dict],
    resolution: int,
    grid: dict,
) -> dict[str, np.ndarray]:
    """Rasterize each scan's own projected mesh so symbols stay within its outline."""
    masks = {}
    face_offset = 0
    for scan in scans:
        face_count = int(scan["regularized"]["output_triangles"])
        face_stop = face_offset + face_count
        masks[str(scan["scan_id"])] = _rasterize_triangles(
            vertices, faces[face_offset:face_stop], resolution, grid,
        )
        face_offset = face_stop
    if face_offset != len(faces):
        raise ValueError("Per-scan regularized triangle counts do not match the floor mesh")
    return masks


def _floor_coverage_mask(
    alignment_dir: Path,
    scans: list[dict],
    planes: dict[str, dict],
    resolution: int,
    grid: dict,
) -> np.ndarray:
    """Project only real floor-plane meshes; furniture and walls do not occlude lower symbols."""
    covered = np.zeros((resolution, resolution), dtype=np.uint8)
    for scan in scans:
        if scan["output_status"] != "generated_for_floor_level":
            continue
        for plane_id in scan.get("floor_plane_ids", []):
            plane = planes.get(plane_id)
            if not plane or plane.get("role") != "floor":
                continue
            source = Path(plane["source_mesh"])
            if not source.is_absolute():
                source = alignment_dir / source
            if not source.is_file():
                raise FileNotFoundError(f"Floor-plane mesh is missing: {source}")
            vertices, faces, _ = _read_mesh(source)
            plane_mask = _rasterize_triangles(vertices, faces, resolution, grid)
            covered[plane_mask] = 255
    # Close only sub-pixel seams between adjacent floor triangles.
    covered = cv2.morphologyEx(covered, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    return covered > 0


def _pixel_rect(bounds_min: list[float], bounds_max: list[float], grid: dict, size: int):
    pixel_size = float(grid["pixel_size"])
    raw_x0 = int(math.floor((bounds_min[0] - grid["x_min"]) / pixel_size))
    raw_x1 = int(math.ceil((bounds_max[0] - grid["x_min"]) / pixel_size))
    raw_y0 = int(math.floor((grid["z_max"] - bounds_max[2]) / pixel_size))
    raw_y1 = int(math.ceil((grid["z_max"] - bounds_min[2]) / pixel_size))
    if raw_x1 <= 0 or raw_x0 >= size or raw_y1 <= 0 or raw_y0 >= size:
        return None
    x0, x1 = max(0, raw_x0), min(size, max(raw_x0 + 1, raw_x1))
    y0, y1 = max(0, raw_y0), min(size, max(raw_y0 + 1, raw_y1))
    return (x0, y0, x1, y1) if x0 < x1 and y0 < y1 else None


def _floor_coverage_fraction(candidate: dict, floor_mask: np.ndarray, grid: dict, size: int) -> float:
    bounds_min = candidate.get("collision_bbox_min_xyz", candidate["bbox_min_xyz"])
    bounds_max = candidate.get("collision_bbox_max_xyz", candidate["bbox_max_xyz"])
    rect = _pixel_rect(bounds_min, bounds_max, grid, size)
    if rect is None:
        return 1.0
    x0, y0, x1, y1 = rect
    return float(floor_mask[y0:y1, x0:x1].mean())


def _overlaps_xz(a: dict, b: dict) -> bool:
    amin = a.get("collision_bbox_min_xyz", a["bbox_min_xyz"])
    amax = a.get("collision_bbox_max_xyz", a["bbox_max_xyz"])
    bmin = b.get("collision_bbox_min_xyz", b["bbox_min_xyz"])
    bmax = b.get("collision_bbox_max_xyz", b["bbox_max_xyz"])
    return (
        min(amax[0], bmax[0]) > max(amin[0], bmin[0])
        and min(amax[2], bmax[2]) > max(amin[2], bmin[2])
    )


def _rect_is_inside(candidate: dict, mask: np.ndarray, grid: dict) -> bool:
    size = mask.shape[0]
    rect = _pixel_rect(candidate["bbox_min_xyz"], candidate["bbox_max_xyz"], grid, size)
    if rect is None:
        return False
    x0, y0, x1, y1 = rect
    if x0 == 0 or y0 == 0 or x1 == size or y1 == size:
        bounds = candidate["bbox_min_xyz"], candidate["bbox_max_xyz"]
        if (bounds[0][0] < grid["x_min"] or bounds[1][0] > grid["x_min"] + size * grid["pixel_size"]
                or bounds[0][2] < grid["z_max"] - size * grid["pixel_size"]
                or bounds[1][2] > grid["z_max"]):
            return False
    return bool(mask[y0:y1, x0:x1].all())


def _shift_candidate(candidate: dict, dx: float, dz: float) -> dict:
    moved = {**candidate}
    moved["bbox_min_xyz"] = list(candidate["bbox_min_xyz"])
    moved["bbox_max_xyz"] = list(candidate["bbox_max_xyz"])
    moved["bbox_min_xyz"][0] += dx
    moved["bbox_max_xyz"][0] += dx
    moved["bbox_min_xyz"][2] += dz
    moved["bbox_max_xyz"][2] += dz
    prior = candidate.get("placement_offset_xz_m", [0.0, 0.0])
    moved["placement_offset_xz_m"] = [prior[0] + dx, prior[1] + dz]
    return moved


def _merge_attached_same_category(candidates: list[dict]) -> list[dict]:
    """Combine projected footprints of same-category instances touching in one scan/floor."""
    groups: dict[tuple[str, str, int], list[dict]] = {}
    for candidate in candidates:
        key = (candidate["scan_id"], candidate["floor_level_name"], candidate["category_id"])
        groups.setdefault(key, []).append(candidate)

    merged = []
    for group in groups.values():
        parent = list(range(len(group)))

        def root(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for i, a in enumerate(group):
            for j in range(i + 1, len(group)):
                b = group[j]
                dx = max(0.0, a["bbox_min_xyz"][0] - b["bbox_max_xyz"][0],
                         b["bbox_min_xyz"][0] - a["bbox_max_xyz"][0])
                dz = max(0.0, a["bbox_min_xyz"][2] - b["bbox_max_xyz"][2],
                         b["bbox_min_xyz"][2] - a["bbox_max_xyz"][2])
                if math.hypot(dx, dz) <= SAME_CATEGORY_MERGE_GAP_M:
                    ri, rj = root(i), root(j)
                    parent[rj] = ri

        components: dict[int, list[dict]] = {}
        for index, candidate in enumerate(group):
            components.setdefault(root(index), []).append(candidate)
        for component in components.values():
            representative = max(component, key=lambda item: (
                (item["bbox_max_xyz"][0] - item["bbox_min_xyz"][0])
                * (item["bbox_max_xyz"][2] - item["bbox_min_xyz"][2])
            ))
            item = {**representative}
            item["bbox_min_xyz"] = [
                min(candidate["bbox_min_xyz"][axis] for candidate in component)
                for axis in range(3)
            ]
            item["bbox_max_xyz"] = [
                max(candidate["bbox_max_xyz"][axis] for candidate in component)
                for axis in range(3)
            ]
            if len(component) > 1:
                item["merged_source_instances"] = [
                    {"scan_id": candidate["scan_id"],
                     "source_class_id": candidate["source_class_id"],
                     "instance_id": candidate["instance_id"]}
                    for candidate in sorted(component, key=_candidate_key)
                ]
            merged.append(item)

    assert sum(len(item.get("merged_source_instances", [])) or 1 for item in merged) == len(candidates)
    return merged


def _suppress_vertical_full_overlaps(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    """Let a lower object win when it covers nearly all of an upper object's footprint."""
    suppressed: dict[int, tuple[dict, dict]] = {}
    vertical_order = sorted(range(len(candidates)), key=lambda i: (
        candidates[i]["bbox_min_xyz"][1], candidates[i]["bbox_max_xyz"][1], _candidate_key(candidates[i]),
    ))
    for position, i in enumerate(vertical_order):
        a = candidates[i]
        if i in suppressed or a["category_id"] == 14:
            continue
        for j in vertical_order[position + 1:]:
            if j in suppressed:
                continue
            b = candidates[j]
            if b["category_id"] == 14 or not _overlaps_xz(a, b):
                continue
            amin, amax = a["bbox_min_xyz"], a["bbox_max_xyz"]
            bmin, bmax = b["bbox_min_xyz"], b["bbox_max_xyz"]
            area_a = (amax[0] - amin[0]) * (amax[2] - amin[2])
            area_b = (bmax[0] - bmin[0]) * (bmax[2] - bmin[2])
            overlap = (min(amax[0], bmax[0]) - max(amin[0], bmin[0])) * (
                min(amax[2], bmax[2]) - max(amin[2], bmin[2])
            )
            if min(area_a, area_b) <= 0 or overlap / min(area_a, area_b) < COMPLETE_OVERLAP_COVERAGE:
                continue
            if amax[1] <= bmin[1] + VERTICAL_OVERLAP_HEIGHT_TOLERANCE_M:
                suppressed[j] = (b, a)
            elif bmax[1] <= amin[1] + VERTICAL_OVERLAP_HEIGHT_TOLERANCE_M:
                suppressed[i] = (a, b)

    kept = [candidate for index, candidate in enumerate(candidates) if index not in suppressed]
    hidden = [{
        "scan_id": item["scan_id"],
        "source_class_id": item["source_class_id"],
        "instance_id": item["instance_id"],
        "category": item["category"],
        "hidden_by": {
            "scan_id": lower["scan_id"],
            "source_class_id": lower["source_class_id"],
            "instance_id": lower["instance_id"],
        },
        "reason": "near_complete_footprint_overlap_with_lower_instance",
    } for item, lower in suppressed.values()]
    return kept, hidden


def _translation_offsets(max_distance_m: float, pixel_size: float) -> list[tuple[float, float]]:
    steps = int(math.floor(max_distance_m / pixel_size))
    offsets = [
        (x * pixel_size, z * pixel_size)
        for x in range(-steps, steps + 1)
        for z in range(-steps, steps + 1)
        if 0 < math.hypot(x * pixel_size, z * pixel_size) <= max_distance_m
    ]
    return sorted(offsets, key=lambda item: (item[0] ** 2 + item[1] ** 2, item[0], item[1]))


def _fit_inside_scan(candidate: dict, mask: np.ndarray | None, grid: dict) -> dict | None:
    if mask is None:
        return None
    if _rect_is_inside(candidate, mask, grid):
        return {
            **candidate,
            "placement_offset_xz_m": [0.0, 0.0],
            "contour_fit_offset_xz_m": [0.0, 0.0],
            "overlap_nudge_xz_m": [0.0, 0.0],
        }
    for dx, dz in _translation_offsets(MAX_CONTOUR_TRANSLATION_M, float(grid["pixel_size"])):
        moved = _shift_candidate(candidate, dx, dz)
        if _rect_is_inside(moved, mask, grid):
            moved["contour_fit_offset_xz_m"] = moved["placement_offset_xz_m"]
            moved["overlap_nudge_xz_m"] = [0.0, 0.0]
            return moved
    return None


def _try_overlap_nudge(
    candidate: dict,
    displayed: list[dict],
    mask: np.ndarray | None,
    grid: dict,
) -> dict | None:
    if mask is None:
        return None
    for dx, dz in _translation_offsets(MAX_OVERLAP_TRANSLATION_M, float(grid["pixel_size"])):
        moved = _shift_candidate(candidate, dx, dz)
        if not _rect_is_inside(moved, mask, grid) or any(_overlaps_xz(moved, item) for item in displayed):
            continue
        prior = candidate.get("overlap_nudge_xz_m", [0.0, 0.0])
        moved["overlap_nudge_xz_m"] = [prior[0] + dx, prior[1] + dz]
        return moved
    return None


def _resolve_overlaps(
    candidates: list[dict], footprints: dict[str, np.ndarray] | None = None,
    grid: dict | None = None, locked: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    ordered = sorted(candidates, key=lambda item: (
        0 if item.get("category_id") in FIXED_FOOTPRINT_CATEGORIES else 1,
        item["bbox_min_xyz"][1], _candidate_key(item),
    ))
    rugs = [item for item in ordered if item.get("category_id") == 14]
    displayed, hidden = [], []
    locked = locked or []
    for candidate in (item for item in ordered if item.get("category_id") != 14):
        blockers = [item for item in (*locked, *displayed) if _overlaps_xz(candidate, item)]
        if not blockers:
            displayed.append(candidate)
            continue
        mask = (footprints or {}).get(candidate["scan_id"])
        moved = _try_overlap_nudge(candidate, [*locked, *displayed], mask, grid) if grid else None
        if moved is not None:
            displayed.append(moved)
            continue
        blocker = blockers[0]
        hidden.append({
            "scan_id": candidate["scan_id"],
            "source_class_id": candidate["source_class_id"],
            "instance_id": candidate["instance_id"],
            "category": candidate["category"],
            "hidden_by": {
                "scan_id": blocker["scan_id"],
                "source_class_id": blocker["source_class_id"],
                "instance_id": blocker["instance_id"],
            },
            "reason": "could_not_place_without_overlap_inside_contour",
        })
    # Rugs are backgrounds; their SVG artwork is transparent outside the rug itself.
    return rugs + displayed, hidden


def _candidate_key(candidate: dict) -> tuple[str, int, int]:
    return candidate["scan_id"], candidate["source_class_id"], candidate["instance_id"]


def _load_candidates(associations: list[dict], levels: list[dict], planes: dict[str, dict]) -> list[dict]:
    level_order = {level["floor_level_name"]: index for index, level in enumerate(levels)}
    candidates = []
    for row in associations:
        category_id = row.get("category_id")
        metadata = row.get("floor_measurement_metadata") or {}
        floor_level = metadata.get("assigned_floor_level")
        floor_plane = planes.get(metadata.get("floor_plane_id"))
        bounds_min = metadata.get("robust_bbox_min_xyz")
        bounds_max = metadata.get("robust_bbox_max_xyz")
        if (category_id is None or floor_level not in level_order or floor_plane is None
                or bounds_min is None or bounds_max is None):
            continue
        category_id = int(category_id)
        if category_id not in CLASS_NAMES:
            continue
        if len(bounds_min) != 3 or len(bounds_max) != 3:
            continue
        bounds_min = [float(value) for value in bounds_min]
        bounds_max = [float(value) for value in bounds_max]
        if (not np.isfinite(bounds_min + bounds_max).all()
                or bounds_max[0] <= bounds_min[0]
                or bounds_max[2] <= bounds_min[2]):
            continue
        candidates.append({
            "scan_id": str(row["scan_id"]),
            "source_class_id": int(row["source_class_id"]),
            "instance_id": int(row["instance_id"]),
            "category_id": category_id,
            "category": str(row.get("category") or CLASS_NAMES[category_id]),
            "floor_level_name": floor_level,
            "floor_level_index": level_order[floor_level],
            "scan_level_name": floor_plane.get("scan_level_name"),
            "bbox_min_xyz": bounds_min,
            "bbox_max_xyz": bounds_max,
            "aligned_mesh": metadata.get("aligned_mesh"),
        })
    return candidates


def _nearest_wall_rotation(candidate: dict, mask: np.ndarray | None, grid: dict) -> int:
    if mask is None:
        return 0
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0
    points = np.concatenate(contours, axis=0).reshape(-1, 2)
    center = np.asarray([
        (candidate["bbox_min_xyz"][0] + candidate["bbox_max_xyz"][0]) / 2,
        (candidate["bbox_min_xyz"][2] + candidate["bbox_max_xyz"][2]) / 2,
    ])
    center_px = np.asarray([
        (center[0] - grid["x_min"]) / grid["pixel_size"],
        (grid["z_max"] - center[1]) / grid["pixel_size"],
    ])
    nearest = points[np.argmin(np.sum((points - center_px) ** 2, axis=1))]
    delta = nearest - center_px
    if abs(delta[0]) > abs(delta[1]):
        return 90 if delta[0] > 0 else 270
    return 180 if delta[1] > 0 else 0


def _wc_back_rotation(candidate: dict, mask: np.ndarray | None, grid: dict) -> int:
    """Keep the WC tank/back edge on its shorter footprint side, toward a wall."""
    span_x = candidate["bbox_max_xyz"][0] - candidate["bbox_min_xyz"][0]
    span_z = candidate["bbox_max_xyz"][2] - candidate["bbox_min_xyz"][2]
    short_axis_parity = 0 if span_x <= span_z else 1
    if mask is None:
        return 0 if short_axis_parity == 0 else 90
    nearest = _nearest_wall_rotation(candidate, mask, grid)
    if (nearest // 90) % 2 == short_axis_parity:
        return nearest
    # The shorter back edge must stay parallel to the nearest compatible wall.
    center_x = (candidate["bbox_min_xyz"][0] + candidate["bbox_max_xyz"][0]) / 2
    center_z = (candidate["bbox_min_xyz"][2] + candidate["bbox_max_xyz"][2]) / 2
    center_px = np.asarray([
        (center_x - grid["x_min"]) / grid["pixel_size"],
        (grid["z_max"] - center_z) / grid["pixel_size"],
    ])
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0 if short_axis_parity == 0 else 90
    points = np.concatenate(contours, axis=0).reshape(-1, 2)
    delta = points - center_px
    if short_axis_parity == 0:
        parallel_extent = (span_x / grid["pixel_size"]) / 2 + 2
        near_back = np.abs(delta[:, 0]) <= parallel_extent
        down = float(np.min(delta[near_back & (delta[:, 1] > 0), 1])) if np.any(near_back & (delta[:, 1] > 0)) else float("inf")
        up = float(np.min(-delta[near_back & (delta[:, 1] < 0), 1])) if np.any(near_back & (delta[:, 1] < 0)) else float("inf")
        return 180 if down <= up else 0
    parallel_extent = (span_z / grid["pixel_size"]) / 2 + 2
    near_back = np.abs(delta[:, 1]) <= parallel_extent
    right = float(np.min(delta[near_back & (delta[:, 0] > 0), 0])) if np.any(near_back & (delta[:, 0] > 0)) else float("inf")
    left = float(np.min(-delta[near_back & (delta[:, 0] < 0), 0])) if np.any(near_back & (delta[:, 0] < 0)) else float("inf")
    return 90 if right <= left else 270


def _stairs_up_rotation(aligned_mesh: str | None) -> int:
    if not aligned_mesh or not Path(aligned_mesh).is_file():
        return 0
    vertices, faces, _ = _read_mesh(Path(aligned_mesh))
    return _triangle_density_rotation(vertices[faces].mean(axis=1))


def _triangle_density_rotation(centroids: np.ndarray) -> int:
    if len(centroids) < 3:
        return 0
    best = None
    for axis in (0, 2):
        span = float(np.ptp(centroids[:, axis]))
        if span < 1e-6:
            continue
        edges = np.linspace(float(centroids[:, axis].min()), float(centroids[:, axis].max()), 9)
        bins = np.clip(np.digitize(centroids[:, axis], edges[1:-1]), 0, 7)
        counts = np.bincount(bins, minlength=8)
        heights = np.bincount(bins, weights=centroids[:, 1], minlength=8)
        present = counts > 0
        if present.sum() < 3:
            continue
        means = heights[present] / counts[present]
        centers = (edges[:-1] + edges[1:])[present] / 2
        slope = float(np.polyfit(centers, means, 1, w=np.sqrt(counts[present]))[0])
        rise = slope * span
        score = abs(rise)
        if best is None or score > best[0]:
            best = (score, axis, slope)
    if best is None or best[0] < 0.15:
        return 0
    _, axis, slope = best
    if axis == 0:
        return 90 if slope > 0 else 270
    return 0 if slope > 0 else 180


def _symbol_rotation(candidate: dict, footprint: np.ndarray | None, grid: dict) -> int:
    if candidate["category_id"] == 3:
        return _stairs_up_rotation(candidate.get("aligned_mesh"))
    if candidate["category_id"] == 13:
        return 0 if (candidate["bbox_max_xyz"][0] - candidate["bbox_min_xyz"][0]
                     >= candidate["bbox_max_xyz"][2] - candidate["bbox_min_xyz"][2]) else 90
    if candidate["category_id"] == 24:
        return _wc_back_rotation(candidate, footprint, grid)
    if candidate["category_id"] in WALL_ORIENTED_CATEGORIES:
        return _nearest_wall_rotation(candidate, footprint, grid)
    return 0


def _append_symbol(
    root: ET.Element, candidate: dict, grid: dict, opacity: float,
    orientation_footprint: np.ndarray | None,
    orientation_grid: dict,
    rotation_override: int | None = None,
    phase07_panel: bool = False,
) -> None:
    bounds_min, bounds_max = candidate["bbox_min_xyz"], candidate["bbox_max_xyz"]
    span_x = bounds_max[0] - bounds_min[0]
    span_z = bounds_max[2] - bounds_min[2]
    rotation = (rotation_override if rotation_override is not None
                else _symbol_rotation(candidate, orientation_footprint, orientation_grid))
    local_length_x, local_depth_z = ((span_z, span_x) if rotation % 180 else (span_x, span_z))
    symbol_root = ET.fromstring(sized_svg(
        candidate["category_id"], local_depth_z, local_length_x, validate_dimensions=False,
    ))
    pixels_per_mm = 1.0 / (float(grid["pixel_size"]) * 1000.0)
    center_x_px = ((bounds_min[0] + bounds_max[0]) / 2 - float(grid["x_min"])) / float(grid["pixel_size"])
    center_y_px = (float(grid["z_max"]) - (bounds_min[2] + bounds_max[2]) / 2) / float(grid["pixel_size"])
    attrs = {
        "transform": (
            f"translate({center_x_px:.8f} {center_y_px:.8f}) scale({pixels_per_mm:.10f}) "
            f"rotate({rotation}) translate({-local_length_x * 500:.6f} {-local_depth_z * 500:.6f})"
        ),
        "opacity": f"{opacity:g}",
        "data-scan-id": candidate["scan_id"],
        "data-source-class-id": str(candidate["source_class_id"]),
        "data-instance-id": str(candidate["instance_id"]),
        "data-category-id": str(candidate["category_id"]),
        "data-floor-level": candidate["floor_level_name"],
        "data-orientation-deg": str(rotation),
        "data-placement-offset-xz-m": ",".join(f"{v:.4f}" for v in candidate.get("placement_offset_xz_m", [0, 0])),
        "data-wall-adherence-offset-xz-m": ",".join(
            f"{v:.4f}" for v in candidate.get("wall_adherence_offset_xz_m", [0, 0])
        ),
    }
    merged_sources = candidate.get("merged_source_instances", [])
    if merged_sources:
        attrs["data-merged-instance-count"] = str(len(merged_sources))
    if phase07_panel:
        attrs["data-phase07-panel-index"] = str(candidate["panel_index"])
    group = ET.SubElement(root, f"{{{SVG_NS}}}g", attrs)
    title = ET.SubElement(group, f"{{{SVG_NS}}}title")
    title.text = (
        f"{candidate['category']} · {candidate['scan_id']} · "
        f"class {candidate['source_class_id']} / instance {candidate['instance_id']}"
    )
    if merged_sources:
        title.text += f" · merged {len(merged_sources)} instances"
    # Keep the generated symbol's white object geometry, but not its standalone canvas backdrop.
    for child in list(symbol_root):
        if child.tag in (f"{{{SVG_NS}}}title", f"{{{SVG_NS}}}rect") and child.get("x") is None:
            continue
        group.append(child)


def _load_phase07_panels(alignment_dir: Path, levels: list[dict], metadata: dict) -> list[dict]:
    floor_planes = metadata["floor_planes"]
    level_elevations = {}
    for level in levels:
        elevations = [
            float(plane["mean_elevation_m"])
            for plane in floor_planes
            if plane.get("floor_level_name") == level["floor_level_name"]
        ]
        level_elevations[level["floor_level_name"]] = float(np.median(elevations)) if elevations else None
    scan_levels = {}
    for plane in floor_planes:
        scan_levels.setdefault(plane["scan_id"], plane.get("scan_level_name"))

    panels = []
    for category_id, filename in ((1, "aligned_door_panels.json"), (2, "aligned_window_panels.json")):
        path = alignment_dir / "stats" / filename
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("panels", []):
            floor_y = item.get("floor_elevation_m")
            if floor_y is None:
                floor_y = item.get("panel_bottom_elevation_m")
            if floor_y is None:
                continue
            nearest_level = min(
                (name for name, elevation in level_elevations.items() if elevation is not None),
                key=lambda name: abs(level_elevations[name] - float(floor_y)),
                default=None,
            )
            if nearest_level is None or abs(level_elevations[nearest_level] - float(floor_y)) > 0.5:
                continue
            sources = item.get("source_instances", [])
            scan_ids = sorted({str(source["scan_id"]) for source in sources if source.get("scan_id")})
            if item.get("scan_id"):
                scan_ids = sorted(set(scan_ids) | {str(item["scan_id"])})
            scan_id = scan_ids[0] if scan_ids else None
            if category_id == 1 and item.get("panel_bounds_xyz_m"):
                bounds_min, bounds_max = item["panel_bounds_xyz_m"]
                min_x, max_x = float(bounds_min[0]), float(bounds_max[0])
                min_z, max_z = float(bounds_min[2]), float(bounds_max[2])
            else:
                center = item["panel_center_xyz_m"]
                dims = item["panel_dimensions_m"]
                orientation = str(item.get("orientation", "normal_z")).lower()
                half_width, half_depth = float(dims["width"]) / 2, float(dims["depth"]) / 2
                if orientation in {"normal_x", "vertical"}:
                    half_x, half_z = half_depth, half_width
                else:
                    half_x, half_z = half_width, half_depth
                min_x, max_x = float(center[0]) - half_x, float(center[0]) + half_x
                min_z, max_z = float(center[2]) - half_z, float(center[2]) + half_z
            if max_x <= min_x or max_z <= min_z:
                continue
            panel = {
                "scan_id": scan_id,
                "scan_level_name": scan_levels.get(scan_id),
                "source_class_id": -1,
                "instance_id": int(item["panel_index"]),
                "category_id": category_id,
                "category": CLASS_NAMES[category_id],
                "floor_level_name": nearest_level,
                "floor_level_index": next(
                    i for i, level in enumerate(levels) if level["floor_level_name"] == nearest_level
                ),
                "bbox_min_xyz": [min_x, float(floor_y), min_z],
                "bbox_max_xyz": [max_x, float(floor_y) + 0.01, max_z],
                "panel_center_xyz_m": [float(value) for value in item.get(
                    "panel_center_xyz_m", [(min_x + max_x) / 2, float(floor_y), (min_z + max_z) / 2],
                )],
                "panel_index": int(item["panel_index"]),
                "panel_kind": "phase07_regularized_panel",
                "source_scans": scan_ids,
                "orientation": str(item.get("orientation", "normal_z")).lower(),
                "panel_dimensions_m": item.get("panel_dimensions_m", {}),
            }
            start = item.get("segment_start_x_z_forward_m")
            end = item.get("segment_end_x_z_forward_m")
            if start is not None and end is not None:
                panel["segment_start_x_z_forward_m"] = [float(value) for value in start]
                panel["segment_end_x_z_forward_m"] = [float(value) for value in end]
            panels.append(panel)
    return panels


def _door_inward_normal(
    panel: dict,
    start_px: np.ndarray,
    tangent: np.ndarray,
    length_m: float,
    footprints: dict[str, np.ndarray],
    pixel_size: float,
    central_room_id: str | None,
) -> np.ndarray:
    normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
    center_px = start_px + tangent * length_m / (2 * pixel_size)
    side_scores: list[dict[str, int]] = []
    for direction in (normal, -normal):
        scores = {scan_id: 0 for scan_id in footprints}
        for distance in (0.15, 0.30, 0.50, 0.75, 1.0):
            for along in (0.2, 0.5, 0.8):
                point = (start_px + tangent * (length_m * along / pixel_size)
                         + direction * distance / pixel_size)
                x, y = np.rint(point).astype(int)
                for scan_id, mask in footprints.items():
                    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                        scores[scan_id] += int(mask[y, x])
        side_scores.append(scores)

    central_scores = [scores.get(central_room_id, 0) for scores in side_scores]
    if central_scores[0] != central_scores[1]:
        return normal if central_scores[0] < central_scores[1] else -normal

    central_mask = footprints.get(central_room_id or "")
    if central_mask is not None and central_mask.any():
        occupied_y, occupied_x = np.nonzero(central_mask)
        centroid = np.asarray([occupied_x.mean(), occupied_y.mean()])
        # Doors at the most connected room swing away from that room's center.
        return normal if np.dot(center_px - centroid, normal) >= 0 else -normal

    fallback_mask = footprints.get(panel.get("scan_id") or "")
    if fallback_mask is None or not fallback_mask.any():
        return normal
    occupied_y, occupied_x = np.nonzero(fallback_mask)
    centroid = np.asarray([occupied_x.mean(), occupied_y.mean()])
    return normal if np.dot(centroid - center_px, normal) >= 0 else -normal


def _door_geometry(panel: dict, grid: dict):
    start = panel.get("segment_start_x_z_forward_m")
    end = panel.get("segment_end_x_z_forward_m")
    width_m = float(panel.get("panel_dimensions_m", {}).get("width", 0.0))
    if start is None or end is None:
        center = panel["panel_center_xyz_m"]
        length = width_m
        if panel["orientation"] in {"vertical", "normal_x"}:
            start = [center[0], -center[2] - length / 2]
            end = [center[0], -center[2] + length / 2]
        else:
            start = [center[0] - length / 2, -center[2]]
            end = [center[0] + length / 2, -center[2]]
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    segment = end - start
    segment_length = float(np.linalg.norm(segment))
    if segment_length <= 0:
        return None
    tangent = segment / segment_length
    start_px = np.asarray([
        (start[0] - float(grid["x_min"])) / float(grid["pixel_size"]),
        (float(grid["z_max"]) + start[1]) / float(grid["pixel_size"]),
    ])
    inward = panel.get("door_opening_normal_xz")
    inward = np.asarray(inward, dtype=float) if inward is not None else np.asarray([-tangent[1], tangent[0]])
    return start, tangent, inward, segment_length, start_px


def _panel_segment(panel: dict) -> tuple[np.ndarray, np.ndarray, float]:
    start = panel.get("segment_start_x_z_forward_m")
    end = panel.get("segment_end_x_z_forward_m")
    if start is not None and end is not None:
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        return start, end, float(np.linalg.norm(end - start))
    center = panel["panel_center_xyz_m"]
    width = float(panel.get("panel_dimensions_m", {}).get("width", 0.0))
    if panel["orientation"] in {"normal_x", "vertical"}:
        return np.asarray([center[0], -center[2] - width / 2]), np.asarray([center[0], -center[2] + width / 2]), width
    return np.asarray([center[0] - width / 2, -center[2]]), np.asarray([center[0] + width / 2, -center[2]]), width


def _scan_contour_segments(mask: np.ndarray, grid: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    pixel_size = float(grid["pixel_size"])
    for contour in contours:
        points = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2)
        if len(points) < 2:
            continue
        world = np.column_stack((
            float(grid["x_min"]) + (points[:, 0] + 0.5) * pixel_size,
            -float(grid["z_max"]) + (points[:, 1] + 0.5) * pixel_size,
        ))
        segments.extend((world[i], world[(i + 1) % len(world)]) for i in range(len(world)))
    return segments


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray):
    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared <= 1e-12:
        return float(np.linalg.norm(point - start)), start.copy()
    fraction = float(np.clip(np.dot(point - start, segment) / length_squared, 0.0, 1.0))
    nearest = start + fraction * segment
    return float(np.linalg.norm(point - nearest)), nearest


def _segment_intersects_rect(start: np.ndarray, end: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> bool:
    delta = end - start
    enter, leave = 0.0, 1.0
    for p, q in ((-delta[0], start[0] - lower[0]), (delta[0], upper[0] - start[0]),
                 (-delta[1], start[1] - lower[1]), (delta[1], upper[1] - start[1])):
        if abs(p) <= 1e-12:
            if q < 0:
                return False
            continue
        fraction = q / p
        if p < 0:
            enter = max(enter, fraction)
        else:
            leave = min(leave, fraction)
        if enter > leave:
            return False
    return True


def _closest_segment_rect_points(
    start: np.ndarray, end: np.ndarray, lower: np.ndarray, upper: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    if _segment_intersects_rect(start, end, lower, upper):
        point = np.clip((start + end) * 0.5, lower, upper)
        return 0.0, point, point
    corners = np.asarray([
        [lower[0], lower[1]], [lower[0], upper[1]],
        [upper[0], lower[1]], [upper[0], upper[1]],
    ])
    best = (float("inf"), corners[0], start)
    for corner in corners:
        distance, wall_point = _point_segment_distance(corner, start, end)
        if distance < best[0]:
            best = (distance, corner, wall_point)
    for wall_point in (start, end):
        box_point = np.clip(wall_point, lower, upper)
        distance = float(np.linalg.norm(wall_point - box_point))
        if distance < best[0]:
            best = (distance, box_point, wall_point)
    return best


def _snap_candidate_to_nearby_wall(candidate: dict, mask: np.ndarray | None, grid: dict) -> dict:
    """Close a small gap to this scan's nearest contour without moving outside its footprint."""
    if mask is None or candidate["category_id"] in {1, 2}:
        return candidate
    bounds_min, bounds_max = candidate["bbox_min_xyz"], candidate["bbox_max_xyz"]
    lower = np.asarray([bounds_min[0], -bounds_max[2]], dtype=float)
    upper = np.asarray([bounds_max[0], -bounds_min[2]], dtype=float)
    best = None
    for start, end in _scan_contour_segments(mask, grid):
        distance, box_point, wall_point = _closest_segment_rect_points(start, end, lower, upper)
        if distance <= WALL_ADHERENCE_MAX_GAP_M and (best is None or distance < best[0]):
            best = (distance, box_point, wall_point)
    if best is None:
        return candidate
    if best[0] <= float(grid["pixel_size"]):
        return {**candidate, "wall_adherence_offset_xz_m": [0.0, 0.0]}

    _, box_point, wall_point = best
    direction = wall_point - box_point
    delta = direction * (max(0.0, best[0] - float(grid["pixel_size"])) / best[0])
    dx, dz = float(delta[0]), float(-delta[1])
    low, high, fitted = 0.0, 1.0, None
    for _ in range(14):
        fraction = (low + high) * 0.5
        moved = _shift_candidate(candidate, dx * fraction, dz * fraction)
        if _rect_is_inside(moved, mask, grid):
            low, fitted = fraction, moved
        else:
            high = fraction
    if fitted is None or low <= 0.01:
        return {**candidate, "wall_adherence_offset_xz_m": [0.0, 0.0]}
    prior = candidate.get("wall_adherence_offset_xz_m", [0.0, 0.0])
    fitted["wall_adherence_offset_xz_m"] = [prior[0] + dx * low, prior[1] + dz * low]
    assert _rect_is_inside(fitted, mask, grid)
    return fitted


def _restore_wall_adherence(
    displayed: list[dict], locked: list[dict], footprints: dict[str, np.ndarray], grid: dict,
) -> list[dict]:
    result = list(displayed)
    for index, candidate in enumerate(result):
        if candidate["category_id"] in {1, 2, 14}:
            continue
        nudge = candidate.get("overlap_nudge_xz_m", [0.0, 0.0])
        if math.hypot(float(nudge[0]), float(nudge[1])) <= 1e-9:
            continue
        moved = _snap_candidate_to_nearby_wall(candidate, footprints.get(candidate["scan_id"]), grid)
        prior = candidate.get("placement_offset_xz_m", [0.0, 0.0])
        updated = moved.get("placement_offset_xz_m", prior)
        if math.hypot(updated[0] - prior[0], updated[1] - prior[1]) <= 1e-9:
            continue
        blockers = [*locked, *(item for i, item in enumerate(result) if i != index)]
        if not any(_overlaps_xz(moved, item) for item in blockers):
            result[index] = moved
    return result


def _snap_panel_to_wall_segment(panel: dict, footprints: dict[str, np.ndarray], grid: dict) -> dict:
    start, end, length = _panel_segment(panel)
    if length <= 0:
        return panel
    center = (start + end) * 0.5
    tangent = (end - start) / length
    sources = panel.get("source_scans") or ([panel["scan_id"]] if panel.get("scan_id") else [])
    best = None
    for scan_id in sources:
        mask = footprints.get(scan_id)
        if mask is None:
            continue
        for wall_start, wall_end in _scan_contour_segments(mask, grid):
            wall_vector = wall_end - wall_start
            wall_length = float(np.linalg.norm(wall_vector))
            if wall_length < 0.1:
                continue
            parallel = abs(float(np.dot(tangent, wall_vector / wall_length)))
            if parallel < WALL_SEGMENT_PARALLEL_DOT_MIN:
                continue
            distance, projected_center = _point_segment_distance(center, wall_start, wall_end)
            if distance > WALL_SEGMENT_SNAP_MAX_DISTANCE_M:
                continue
            endpoint_error = sum(
                _point_segment_distance(point, wall_start, wall_end)[0]
                for point in (start, end)
            )
            score = (endpoint_error + abs(wall_length - length), distance, -parallel, scan_id)
            if best is None or score < best[0]:
                best = (score, wall_start, wall_end, wall_vector / wall_length, wall_length, projected_center, scan_id)
    if best is None:
        panel["wall_segment_snap"] = {"applied": False, "reason": "no_nearby_parallel_contour_segment"}
        return panel

    score, wall_start, wall_end, tangent, wall_length, projected_center, wall_scan_id = best
    half_length = min(length * 0.5, wall_length * 0.5)
    along = float(np.clip(np.dot(projected_center - wall_start, tangent), half_length, wall_length - half_length))
    snapped_center = wall_start + tangent * along
    snapped_start = snapped_center - tangent * half_length
    snapped_end = snapped_center + tangent * half_length
    panel["segment_start_x_z_forward_m"] = snapped_start.tolist()
    panel["segment_end_x_z_forward_m"] = snapped_end.tolist()
    panel.setdefault("panel_dimensions_m", {})["width"] = half_length * 2
    panel["wall_segment_snap"] = {
        "applied": True,
        "scan_id": wall_scan_id,
        "distance_m": float(score[1]),
        "fitted_length_m": half_length * 2,
        "matched_segment_length_m": wall_length,
        "offset_x_z_forward_m": (snapped_center - center).tolist(),
    }
    panel["orientation"] = "normal_z" if abs(tangent[0]) >= abs(tangent[1]) else "normal_x"

    center_x, center_z_forward = snapped_center
    depth = float(panel.get("panel_dimensions_m", {}).get("depth", 0.1))
    if panel["orientation"] == "normal_z":
        min_x, max_x = sorted((float(snapped_start[0]), float(snapped_end[0])))
        min_z, max_z = -center_z_forward - depth / 2, -center_z_forward + depth / 2
    else:
        min_x, max_x = center_x - depth / 2, center_x + depth / 2
        min_z, max_z = sorted((-float(snapped_start[1]), -float(snapped_end[1])))
    panel["bbox_min_xyz"] = [min_x, panel["bbox_min_xyz"][1], min_z]
    panel["bbox_max_xyz"] = [max_x, panel["bbox_max_xyz"][1], max_z]
    panel["panel_center_xyz_m"] = [center_x, panel["panel_center_xyz_m"][1], -center_z_forward]
    return panel


def _prepare_door_direction(panel: dict, footprints: dict[str, np.ndarray], grid: dict,
                            central_room_id: str | None) -> None:
    if panel["category_id"] != 1:
        return
    start, end, length = _panel_segment(panel)
    if length <= 0:
        return
    start_px = np.asarray([
        (start[0] - float(grid["x_min"])) / float(grid["pixel_size"]),
        (float(grid["z_max"]) + start[1]) / float(grid["pixel_size"]),
    ])
    tangent = (end - start) / length
    normal = _door_inward_normal(
        panel, start_px, tangent, length, footprints, float(grid["pixel_size"]), central_room_id,
    )
    panel["door_opening_normal_xz"] = normal.tolist()
    panel["central_room_id"] = central_room_id


def _build_door_passages(
    panels: list[dict], footprints: dict[str, np.ndarray], grid: dict,
) -> tuple[np.ndarray, list[np.ndarray], list[dict]]:
    """Connect disconnected source footprints across fitted, assigned-floor doors."""
    size = next(iter(footprints.values())).shape[0]
    passages = np.zeros((size, size), dtype=np.uint8)
    door_end_cutlines = []
    details = []
    pixel_size = float(grid["pixel_size"])
    max_steps = int(math.ceil(MAX_DOOR_PASSAGE_REACH_M / pixel_size))
    touch_kernel = np.ones((3, 3), dtype=np.uint8)
    connect_kernel = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    def pixel(point: np.ndarray) -> tuple[int, int] | None:
        x = int(np.rint((point[0] - float(grid["x_min"])) / pixel_size - 0.5))
        y = int(np.rint((float(grid["z_max"]) + point[1]) / pixel_size - 0.5))
        return (x, y) if 0 <= x < size and 0 <= y < size else None

    for panel in panels:
        if panel.get("display_kind") != "assigned_floor" or panel.get("category_id") != 1:
            continue
        sources = sorted({scan_id for scan_id in panel.get("source_scans", [])
                          if scan_id in footprints})
        if len(sources) < 2 or not panel.get("wall_segment_snap", {}).get("applied"):
            continue
        segment_start, segment_end, width = _panel_segment(panel)
        if width <= 0:
            continue
        tangent = (segment_end - segment_start) / width
        normal = np.asarray(
            panel.get("door_opening_normal_xz", [-tangent[1], tangent[0]]), dtype=float,
        )
        sample_count = max(2, int(math.ceil(width / pixel_size)))
        along_values = np.linspace(0.0, width, sample_count + 1)
        panel_half_depth = float(panel.get("panel_dimensions_m", {}).get("depth", 0.0)) * 0.5
        first_search_step = int(math.ceil(
            max(0.04, panel_half_depth + 0.02, 2.0 * pixel_size) / pixel_size
        ))

        for scan_a, scan_b in combinations(sources, 2):
            mask_a, mask_b = footprints[scan_a], footprints[scan_b]
            if ((cv2.dilate(mask_a.astype(np.uint8), touch_kernel) > 0) & mask_b).any():
                continue

            side_hits = []
            for side in (-1.0, 1.0):
                hits = []
                counts = {scan_a: 0, scan_b: 0}
                for along in along_values:
                    origin = segment_start + tangent * along
                    sample_hits = {scan_a: [], scan_b: []}
                    for step in range(first_search_step, max_steps + 1):
                        point = pixel(origin + normal * (side * step * pixel_size))
                        if point is None:
                            break
                        x, y = point
                        if mask_a[y, x]:
                            sample_hits[scan_a].append(step * pixel_size)
                        if mask_b[y, x]:
                            sample_hits[scan_b].append(step * pixel_size)
                    for scan_id in (scan_a, scan_b):
                        counts[scan_id] += bool(sample_hits[scan_id])
                    hits.append(sample_hits)
                owner = max(counts, key=counts.get)
                if counts[owner] < math.ceil(len(hits) * 0.8):
                    side_hits = []
                    break
                owner_mask = footprints[owner]
                distances = []
                for along in along_values:
                    origin = segment_start + tangent * along
                    first_distance = np.nan
                    for step in range(max_steps + 1):
                        point = pixel(origin + normal * (side * step * pixel_size))
                        if point is None:
                            break
                        x, y = point
                        if owner_mask[y, x]:
                            first_distance = step * pixel_size
                            break
                    distances.append(first_distance)
                distances = np.asarray(distances, dtype=float)
                valid = np.flatnonzero(np.isfinite(distances))
                if len(valid) < math.ceil(len(hits) * 0.8):
                    side_hits = []
                    break
                distances = np.interp(np.arange(len(hits)), valid, distances[valid])
                side_hits.append((owner, distances))

            if len(side_hits) != 2 or side_hits[0][0] == side_hits[1][0]:
                continue
            minus_owner, minus_distance = side_hits[0]
            plus_owner, plus_distance = side_hits[1]
            passage_depth = minus_distance + plus_distance
            if float(np.median(passage_depth)) <= 1.5 * pixel_size:
                continue

            minus_edge, plus_edge = [], []
            for along, distance in zip(along_values, minus_distance):
                minus_edge.append(pixel(segment_start + tangent * along - normal * (distance + pixel_size)))
            for along, distance in zip(along_values, plus_distance):
                plus_edge.append(pixel(segment_start + tangent * along + normal * (distance + pixel_size)))
            if any(point is None for point in (*minus_edge, *plus_edge)):
                continue
            polygon = np.asarray(minus_edge + plus_edge[::-1], dtype=np.int32).reshape(-1, 1, 2)
            candidate_mask = np.zeros_like(passages)
            cv2.fillPoly(candidate_mask, [polygon], 1)
            connected = cv2.dilate(candidate_mask, connect_kernel).astype(bool)
            if not connected[mask_a].any() or not connected[mask_b].any():
                continue
            passages |= candidate_mask
            door_end_cutlines.extend((
                np.asarray(minus_edge, dtype=np.int32),
                np.asarray(plus_edge, dtype=np.int32),
            ))
            details.append({
                "panel_index": int(panel["panel_index"]),
                "source_scans": [scan_a, scan_b],
                "width_m": float(width),
                "median_passage_depth_m": float(np.median(passage_depth)),
                "max_passage_depth_m": float(np.max(passage_depth)),
                "scan_on_negative_normal": minus_owner,
                "scan_on_positive_normal": plus_owner,
                "passage_edge_outlines_removed": [minus_owner, plus_owner],
            })
            break

    return passages, door_end_cutlines, details


def _central_room_for_doors(panels: list[dict], footprints: dict[str, np.ndarray]) -> tuple[str | None, dict[str, int]]:
    door_counts: dict[str, int] = {}
    for panel in panels:
        if panel["category_id"] != 1:
            continue
        for scan_id in set(panel.get("source_scans", [])):
            door_counts[scan_id] = door_counts.get(scan_id, 0) + 1
    if not door_counts:
        return None, door_counts
    central = max(
        door_counts,
        key=lambda scan_id: (door_counts[scan_id], int(footprints.get(scan_id, np.zeros((0, 0))).sum()), scan_id),
    )
    return central, door_counts


def _contour_path(contours: list[np.ndarray]) -> str:
    rings = []
    for contour in contours:
        points = cv2.approxPolyDP(contour, 0.5, True).reshape(-1, 2)
        if len(points) < 3:
            continue
        rings.append("M " + " L ".join(f"{int(x)} {int(y)}" for x, y in points) + " Z")
    return " ".join(rings)


def _append_vector_floor_base(
    root: ET.Element,
    projection_image: Path,
    footprints: dict[str, np.ndarray],
    grid: dict,
    passage_mask: np.ndarray | None = None,
    door_end_cutlines: list[np.ndarray] | None = None,
) -> None:
    image = cv2.imread(str(projection_image), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"Could not read regularized floor projection: {projection_image}")
    size = image.shape[0]
    if image.shape[1] != size:
        raise ValueError("The regularized floor projection must be square")
    ET.SubElement(root, f"{{{SVG_NS}}}rect", {
        "x": "0", "y": "0", "width": str(size), "height": str(size), "fill": "white",
    })

    occupied = np.zeros_like(image, dtype=bool)
    scan_masks = []
    for mask in footprints.values():
        filled_mask, _ = _fill_small_footprint_holes(
            mask.astype(np.uint8) * 255, float(grid["pixel_size"]),
        )
        scan_masks.append(filled_mask > 0)
        occupied |= filled_mask > 0
    owners = np.zeros_like(image, dtype=np.uint16)
    for index, mask in enumerate(scan_masks, start=1):
        owners[(owners == 0) & mask] = index
    if passage_mask is not None and passage_mask.any():
        passage_fill = passage_mask.astype(bool) & ~occupied
        if passage_fill.any():
            ys, xs = np.nonzero(passage_fill)
            pad = max(8, int(math.ceil(0.25 / float(grid["pixel_size"]))))
            x0, x1 = max(0, int(xs.min()) - pad), min(size, int(xs.max()) + pad + 1)
            y0, y1 = max(0, int(ys.min()) - pad), min(size, int(ys.max()) + pad + 1)
            local_image = image[y0:y1, x0:x1]
            local_occupied = occupied[y0:y1, x0:x1]
            valid_gray = local_occupied & (local_image > 16) & (local_image < 250)
            if valid_gray.any():
                invalid = (~valid_gray).astype(np.uint8) * 255
                interpolated = cv2.inpaint(local_image, invalid, 5, cv2.INPAINT_TELEA)
                local_passage = passage_fill[y0:y1, x0:x1]
                local_image[local_passage] = interpolated[local_passage]
        occupied |= passage_mask.astype(bool)
    scan_interfaces = np.zeros_like(image, dtype=np.uint8)
    horizontal = (owners[:, :-1] > 0) & (owners[:, 1:] > 0) & (owners[:, :-1] != owners[:, 1:])
    vertical = (owners[:-1, :] > 0) & (owners[1:, :] > 0) & (owners[:-1, :] != owners[1:, :])
    scan_interfaces[:, :-1] |= horizontal
    scan_interfaces[:, 1:] |= horizontal
    scan_interfaces[:-1, :] |= vertical
    scan_interfaces[1:, :] |= vertical
    union_contours, hierarchy = cv2.findContours(
        occupied.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
    )
    outer_contours, inner_contours = [], []
    minimum_hole_area = SMALL_FOOTPRINT_HOLE_MAX_AREA_M2 / max(float(grid["pixel_size"]) ** 2, 1e-12)
    for index, contour in enumerate(union_contours):
        depth, parent = 0, int(hierarchy[0, index, 3])
        while parent >= 0:
            depth += 1
            parent = int(hierarchy[0, parent, 3])
        if depth % 2 == 0:
            outer_contours.append(contour)
        elif cv2.contourArea(contour) > minimum_hole_area:
            inner_contours.append(contour)

    mesh_outline = np.zeros_like(image, dtype=np.uint8)
    mesh_outline[scan_interfaces > 0] = 255
    if outer_contours:
        cv2.drawContours(mesh_outline, outer_contours, -1, 255, 5, cv2.LINE_8)
    if inner_contours:
        cv2.drawContours(mesh_outline, inner_contours, -1, 255, 2, cv2.LINE_8)

    for lower in range(0, 255, VECTOR_GRAY_STEP):
        upper = min(254, lower + VECTOR_GRAY_STEP - 1)
        mask = occupied & (image >= lower) & (image <= upper)
        if lower == 0:
            mask &= ~(mesh_outline > 0)
        if not mask.any():
            continue
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        d = _contour_path(contours)
        if not d:
            continue
        shade = int(np.rint(image[mask].mean()))
        ET.SubElement(root, f"{{{SVG_NS}}}path", {
            "d": d,
            "fill": f"rgb({shade},{shade},{shade})",
            "fill-rule": "evenodd",
            "shape-rendering": "crispEdges",
        })

    if outer_contours:
        ET.SubElement(root, f"{{{SVG_NS}}}path", {
            "d": _contour_path(outer_contours), "fill": "none", "stroke": "black",
            "stroke-width": "5", "stroke-linejoin": "round", "stroke-linecap": "round",
        })
    if inner_contours:
        ET.SubElement(root, f"{{{SVG_NS}}}path", {
            "d": _contour_path(inner_contours), "fill": "none", "stroke": "black",
            "stroke-width": "2", "stroke-linejoin": "round", "stroke-linecap": "round",
        })
    interface_contours, _ = cv2.findContours(
        scan_interfaces, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if interface_contours:
        ET.SubElement(root, f"{{{SVG_NS}}}path", {
            "d": _contour_path(interface_contours), "fill": "black", "fill-rule": "evenodd",
        })
    if door_end_cutlines:
        sample_kernel = np.ones((11, 11), dtype=np.uint8)
        passage_gray = image[(passage_mask > 0) & (image > 16) & (image < 250)]
        for cutline in door_end_cutlines:
            line = cutline.reshape(-1, 1, 2)
            line_mask = np.zeros_like(image, dtype=np.uint8)
            cv2.polylines(line_mask, [line], False, 255, 1, cv2.LINE_8)
            nearby = (cv2.dilate(line_mask, sample_kernel) > 0) & occupied & (image > 16) & (image < 250)
            samples = image[nearby]
            if not len(samples):
                samples = passage_gray
            shade = int(np.rint(np.median(samples))) if len(samples) else 240
            points = " L ".join(f"{int(x)} {int(y)}" for x, y in cutline)
            if points:
                ET.SubElement(root, f"{{{SVG_NS}}}path", {
                    "d": "M " + points, "fill": "none", "stroke": f"rgb({shade},{shade},{shade})",
                    "stroke-width": "9", "stroke-linecap": "round", "stroke-linejoin": "round",
                })


def _prepare_panel_collision_bounds(panel: dict, grid: dict) -> None:
    if panel["category_id"] != 1:
        return
    geometry = _door_geometry(panel, grid)
    if geometry is None:
        return
    start, tangent, inward, width, _ = geometry
    corners = np.asarray([start, start + tangent * width, start + inward * width,
                          start + (tangent + inward) * width])
    min_x, max_x = float(corners[:, 0].min()), float(corners[:, 0].max())
    min_z, max_z = -float(corners[:, 1].max()), -float(corners[:, 1].min())
    panel["collision_bbox_min_xyz"] = [min_x, panel["bbox_min_xyz"][1], min_z]
    panel["collision_bbox_max_xyz"] = [max_x, panel["bbox_max_xyz"][1], max_z]


def _append_panel(
    root: ET.Element, panel: dict, grid: dict, opacity: float,
    orientation_footprint: np.ndarray | None,
) -> None:
    if panel["category_id"] == 2:
        _append_symbol(
            root, panel, grid, opacity, orientation_footprint, grid,
            rotation_override=90 if panel["orientation"] == "normal_x" else 0,
            phase07_panel=True,
        )
        return

    geometry = _door_geometry(panel, grid)
    if geometry is None:
        return
    start, tangent, inward, width_m, start_px = geometry
    canvas_m = width_m / 0.9 + 0.04
    if canvas_m < 0.5:
        canvas_m = width_m / 0.828
    margin_mm = min(20.0, canvas_m * 40.0)
    symbol_root = ET.fromstring(sized_svg(1, canvas_m, canvas_m, validate_dimensions=False))
    symbol_group = next(child for child in list(symbol_root) if child.tag == f"{{{SVG_NS}}}g")
    scale = 1.0 / (float(grid["pixel_size"]) * 1000.0)
    a, b = scale * inward[0], scale * inward[1]
    c, d = scale * tangent[0], scale * tangent[1]
    e = start_px[0] - (a + c) * margin_mm
    f = start_px[1] - (b + d) * margin_mm
    group = ET.SubElement(root, f"{{{SVG_NS}}}g", {
        "transform": f"matrix({a:.10f} {b:.10f} {c:.10f} {d:.10f} {e:.8f} {f:.8f})",
        "opacity": f"{opacity:g}",
        "data-scan-id": panel.get("scan_id") or "merged",
        "data-category": panel["category"],
        "data-phase07-panel-index": str(panel["panel_index"]),
        "data-floor-level": panel["floor_level_name"],
    })
    ET.SubElement(group, f"{{{SVG_NS}}}title").text = f"Phase 07 regularized door panel {panel['panel_index']}"
    group.append(symbol_group)


def generate_floorplan_instance_overlays(
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    output_dir: Path | None = None,
) -> dict:
    artifacts_dir = Path(artifacts_dir).resolve()
    output_dir = Path(output_dir or artifacts_dir / "floorplan_instance_overlays").resolve()
    mesh_manifest_path = artifacts_dir / "floor_level_meshes" / "floor_level_meshes_manifest.json"
    projection_manifest_path = artifacts_dir / "floor_level_projections" / "floor_level_projections_manifest.json"
    metadata_path = artifacts_dir / "floor_level_metadata.json"
    associations_path = artifacts_dir / "room_category_consistency" / "associations_after_room_check.json"
    for path in (mesh_manifest_path, projection_manifest_path, metadata_path, associations_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required floorplan input is missing: {path}")

    mesh_manifest = json.loads(mesh_manifest_path.read_text(encoding="utf-8"))
    projection_manifest = json.loads(projection_manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    association_data = json.loads(associations_path.read_text(encoding="utf-8"))
    levels = mesh_manifest["levels"]
    planes = {plane["plane_id"]: plane for plane in metadata["floor_planes"]}
    candidates = _merge_attached_same_category(
        _load_candidates(association_data.get("associations", []), levels, planes)
    )
    alignment_dir = Path(metadata["source_alignment_dir"]).resolve()
    phase07_panels = _load_phase07_panels(alignment_dir, levels, metadata)
    projections = {
        (item["floor_level_name"], item["mesh_type"]): item
        for item in projection_manifest["projections"]
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    orientation_footprints: dict[str, dict[str, np.ndarray]] = {}
    orientation_grids: dict[str, dict] = {}
    prepared_phase07_panels: list[dict] = []
    room_summaries: dict[str, dict] = {}
    canonical_placements: dict[tuple[str, int, int], dict | None] = {}

    for level_index, level in enumerate(levels):
        level_name = level["floor_level_name"]
        projection = projections[(level_name, "regularized")]
        image_path = Path(projection["projection_image"])
        resolution_x, resolution_y = map(int, projection["resolution_px"])
        if resolution_x != resolution_y:
            raise ValueError("The regularized floor projection must be square")
        resolution = resolution_x
        vertices, faces, _ = _read_mesh(Path(level["regularized_mesh"]))
        grid = _projection_grid(vertices, resolution)
        footprints = _scan_footprint_masks(vertices, faces, level["scans"], resolution, grid)
        orientation_footprints[level_name] = footprints
        orientation_grids[level_name] = grid
        source_panels = [
            {**panel} for panel in phase07_panels
            if panel["floor_level_index"] == level_index
        ]
        central_room_id, door_counts = _central_room_for_doors(source_panels, footprints)
        room_summaries[level_name] = {
            "central_room_scan_id": central_room_id,
            "door_count_by_scan": door_counts,
        }
        for panel in source_panels:
            _snap_panel_to_wall_segment(panel, footprints, grid)
            _prepare_door_direction(panel, footprints, grid, central_room_id)
        prepared_phase07_panels.extend(source_panels)
        floor_covered = _floor_coverage_mask(alignment_dir, level["scans"], planes, resolution, grid)
        target_scan_levels = {
            planes[plane_id].get("scan_level_name")
            for scan in level["scans"]
            if scan["output_status"] == "generated_for_floor_level"
            for plane_id in scan.get("floor_plane_ids", [])
            if plane_id in planes and planes[plane_id].get("floor_level_name") == level_name
        }
        level_candidates = []
        occluded_by_floor = []
        outside_floor_contour = []
        for original_candidate in candidates:
            if original_candidate["floor_level_index"] == level_index:
                display_kind = "assigned_floor"
            elif (original_candidate["floor_level_index"] < level_index
                    and original_candidate["scan_level_name"] in target_scan_levels):
                display_kind = "visible_lower_floor"
            else:
                continue
            candidate_key = _candidate_key(original_candidate)
            if display_kind == "assigned_floor":
                candidate = _fit_inside_scan(
                    original_candidate, footprints.get(original_candidate["scan_id"]), grid,
                )
                if candidate is None:
                    canonical_placements[candidate_key] = None
                    outside_floor_contour.append({
                        "scan_id": original_candidate["scan_id"],
                        "source_class_id": original_candidate["source_class_id"],
                        "instance_id": original_candidate["instance_id"],
                        "category": original_candidate["category"],
                        "reason": "outside_floor_contour_translation_exceeded",
                    })
                    continue
            else:
                candidate = canonical_placements.get(candidate_key)
                if candidate is None:
                    continue
                candidate = {**candidate}
            if display_kind == "visible_lower_floor":
                if _floor_coverage_fraction(candidate, floor_covered, grid, resolution) > FLOOR_COVERAGE_HIDE_FRACTION:
                    occluded_by_floor.append({
                        "scan_id": candidate["scan_id"],
                        "source_class_id": candidate["source_class_id"],
                        "instance_id": candidate["instance_id"],
                        "category": candidate["category"],
                        "reason": "fully_covered_by_current_floor_footprint",
                    })
                    continue
            candidate = {**candidate, "display_kind": display_kind}
            if display_kind == "assigned_floor":
                canonical_placements[candidate_key] = candidate
            level_candidates.append(candidate)

        level_panels = []
        for panel in prepared_phase07_panels:
            if panel["floor_level_index"] == level_index:
                panel = {**panel, "display_kind": "assigned_floor"}
            elif (panel["floor_level_index"] < level_index
                    and panel["scan_level_name"] in target_scan_levels):
                panel = {**panel, "display_kind": "visible_lower_floor"}
            else:
                continue
            _prepare_panel_collision_bounds(panel, grid)
            if (panel["display_kind"] == "visible_lower_floor"
                    and _floor_coverage_fraction(panel, floor_covered, grid, resolution)
                    > FLOOR_COVERAGE_HIDE_FRACTION):
                occluded_by_floor.append({
                    "scan_id": panel.get("scan_id"),
                    "source_class_id": -1,
                    "instance_id": panel["panel_index"],
                    "category": panel["category"],
                    "reason": "fully_covered_by_current_floor_footprint",
                })
                continue
            level_panels.append(panel)
        assigned_floor_panels = [
            panel for panel in level_panels if panel["display_kind"] == "assigned_floor"
        ]
        candidate_count = len(level_candidates)
        level_candidates, vertical_overlap_hidden = _suppress_vertical_full_overlaps(level_candidates)
        lower_floor_candidates = [
            candidate for candidate in level_candidates
            if candidate["display_kind"] == "visible_lower_floor"
        ]
        lower_floor_blockers = [
            candidate for candidate in lower_floor_candidates
            if candidate["category_id"] != 14
        ]
        assigned_floor_candidates = [
            candidate for candidate in level_candidates
            if candidate["display_kind"] == "assigned_floor"
        ]
        assigned_floor_candidates = [
            _snap_candidate_to_nearby_wall(
                candidate, footprints.get(candidate["scan_id"]), grid,
            )
            for candidate in assigned_floor_candidates
        ]
        for candidate in assigned_floor_candidates:
            canonical_placements[_candidate_key(candidate)] = candidate
        assigned_displayed, overlap_hidden = _resolve_overlaps(
            assigned_floor_candidates, footprints, grid,
            locked=[*assigned_floor_panels, *lower_floor_blockers],
        )
        assigned_displayed = _restore_wall_adherence(
            assigned_displayed, [*assigned_floor_panels, *lower_floor_blockers],
            footprints, grid,
        )
        for candidate in assigned_displayed:
            canonical_placements[_candidate_key(candidate)] = candidate
        displayed = [
            *(item for item in lower_floor_candidates if item["category_id"] == 14),
            *(item for item in assigned_displayed if item["category_id"] == 14),
            *(item for item in lower_floor_candidates if item["category_id"] != 14),
            *(item for item in assigned_displayed if item["category_id"] != 14),
        ]
        hidden = vertical_overlap_hidden + overlap_hidden
        upper_floor_symbols = [
            item for item in displayed if item["display_kind"] == "assigned_floor"
        ] + assigned_floor_panels
        visible_panels = []
        for panel in level_panels:
            blockers = (
                [item for item in upper_floor_symbols if _overlaps_xz(panel, item)]
                if panel["display_kind"] == "visible_lower_floor" and panel["category_id"] == 1
                else []
            )
            if blockers:
                blocker = blockers[0]
                occluded_by_floor.append({
                    "scan_id": panel.get("scan_id"),
                    "source_class_id": -1,
                    "instance_id": panel["panel_index"],
                    "category": panel["category"],
                    "reason": "lower_floor_door_overlaps_current_floor_symbol",
                    "hidden_by": {
                        "scan_id": blocker.get("scan_id"),
                        "category": blocker.get("category"),
                        "instance_id": blocker.get("instance_id", blocker.get("panel_index")),
                    },
                })
                continue
            visible_panels.append(panel)
        level_panels = visible_panels
        passage_mask, door_end_cutlines, door_passages = _build_door_passages(
            level_panels, footprints, grid,
        )

        ET.register_namespace("", SVG_NS)
        root = ET.Element(f"{{{SVG_NS}}}svg", {
            "width": f"{resolution}px",
            "height": f"{resolution}px",
            "viewBox": f"0 0 {resolution} {resolution}",
            "data-floor-level": level_name,
        })
        _append_vector_floor_base(
            root, image_path, footprints, grid, passage_mask, door_end_cutlines,
        )
        for candidate in displayed:
            is_lower = candidate["display_kind"] == "visible_lower_floor"
            _append_symbol(
                root, candidate, grid,
                LOWER_FLOOR_OPACITY if is_lower else 1.0,
                orientation_footprints.get(candidate["floor_level_name"], {}).get(candidate["scan_id"]),
                orientation_grids.get(candidate["floor_level_name"], grid),
            )
        for panel in level_panels:
            is_lower = panel["display_kind"] == "visible_lower_floor"
            _append_panel(
                root, panel, grid,
                LOWER_FLOOR_OPACITY if is_lower else 1.0,
                orientation_footprints.get(panel["floor_level_name"], {}).get(panel.get("scan_id")),
            )

        output_path = output_dir / f"floor_{level_name}_regularized_instances.svg"
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary_path.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
        temporary_path.replace(output_path)
        results.append({
            "floor_level_name": level_name,
            "source_projection": str(image_path),
            "output_svg": str(output_path),
            "projection_resolution_px": [resolution, resolution],
            "lower_floor_opacity": LOWER_FLOOR_OPACITY,
            "candidate_instance_count": candidate_count,
            "merged_instance_group_count": sum(
                len(item.get("merged_source_instances", [])) > 1 for item in displayed
            ),
            "wall_adherence_count": int(sum(
                abs(item.get("wall_adherence_offset_xz_m", [0, 0])[0]) > 1e-9
                or abs(item.get("wall_adherence_offset_xz_m", [0, 0])[1]) > 1e-9
                for item in displayed
            )),
            "displayed_assigned_floor_count": sum(
                item["display_kind"] == "assigned_floor" for item in displayed
            ),
            "displayed_visible_lower_floor_count": sum(
                item["display_kind"] == "visible_lower_floor" for item in displayed
            ),
            "occluded_by_floor_count": len(occluded_by_floor),
            "occluded_by_floor_instances": occluded_by_floor,
            "outside_floor_contour_count": len(outside_floor_contour),
            "outside_floor_contour_instances": outside_floor_contour,
            "suppressed_overlap_count": len(hidden),
            "overlap_reposition_count": int(sum(
                abs(item.get("overlap_nudge_xz_m", [0, 0])[0]) > 1e-9
                or abs(item.get("overlap_nudge_xz_m", [0, 0])[1]) > 1e-9
                for item in displayed
            )),
            "phase07_panel_count": len(level_panels),
            "door_passage_count": len(door_passages),
            "door_passages": door_passages,
            "phase07_panels": [
                {"panel_index": panel["panel_index"], "category": panel["category"],
                 "display_kind": panel["display_kind"], "source_scans": panel["source_scans"],
                 "wall_segment_snap": panel.get("wall_segment_snap"),
                 "central_room_id": panel.get("central_room_id"),
                 "door_opening_normal_xz": panel.get("door_opening_normal_xz")}
                for panel in level_panels
            ],
            "displayed_instances": [
                {
                    "scan_id": item["scan_id"],
                    "source_class_id": item["source_class_id"],
                    "instance_id": item["instance_id"],
                    "category_id": item["category_id"],
                    "category": item["category"],
                    "assigned_floor_level": item["floor_level_name"],
                    "display_kind": item["display_kind"],
                    "final_center_xz_m": [
                        float((item["bbox_min_xyz"][0] + item["bbox_max_xyz"][0]) * 0.5),
                        float((item["bbox_min_xyz"][2] + item["bbox_max_xyz"][2]) * 0.5),
                    ],
                    "contour_fit_offset_xz_m": item.get("contour_fit_offset_xz_m", [0.0, 0.0]),
                    "overlap_nudge_xz_m": [float(v) for v in item.get("overlap_nudge_xz_m", [0.0, 0.0])],
                    "wall_adherence_offset_xz_m": [
                        float(v) for v in item.get("wall_adherence_offset_xz_m", [0.0, 0.0])
                    ],
                    "merged_source_instances": item.get("merged_source_instances", []),
                }
                for item in displayed
            ],
            "suppressed_instances": hidden,
        })

    result = {
        "schema_version": 1,
        "source_floor_mesh_manifest": str(mesh_manifest_path),
        "source_projection_manifest": str(projection_manifest_path),
        "source_floor_metadata": str(metadata_path),
        "source_final_associations": str(associations_path),
        "output_dir": str(output_dir),
        "overlap_priority": (
            "near-complete footprint overlaps keep the lower object; remaining fixed footprint "
            "categories are placed before movable furniture, with non-rug overlaps nudged up to "
            "0.40 m inside their scan contour or suppressed; each instance's assigned-floor "
            "placement is reused unchanged in higher-floor views; rugs are transparent underlays; "
            "Phase 07 panel positions stay fixed"
        ),
        "lower_floor_visibility": (
            "same scan level; hide the whole lower symbol when current-level aligned floor-plane "
            "coverage exceeds 10% of its footprint, otherwise draw the whole symbol at reduced opacity"
        ),
        "lower_floor_opacity": LOWER_FLOOR_OPACITY,
        "contour_fit_max_translation_m": MAX_CONTOUR_TRANSLATION_M,
        "overlap_translation_max_m": MAX_OVERLAP_TRANSLATION_M,
        "wall_adherence_max_gap_m": WALL_ADHERENCE_MAX_GAP_M,
        "same_category_merge_gap_m": SAME_CATEGORY_MERGE_GAP_M,
        "complete_overlap_coverage_threshold": COMPLETE_OVERLAP_COVERAGE,
        "vertical_overlap_height_tolerance_m": VERTICAL_OVERLAP_HEIGHT_TOLERANCE_M,
        "floor_coverage_hide_fraction": FLOOR_COVERAGE_HIDE_FRACTION,
        "phase07_panel_sources": [
            str(alignment_dir / "stats" / "aligned_door_panels.json"),
            str(alignment_dir / "stats" / "aligned_window_panels.json"),
        ],
        "room_door_counts": room_summaries,
        "base_rendering": (
            "regularized PLI projection is rebuilt from vectorized grayscale regions and direct "
            "2D PLI contour segments; empty projection pixels remain white"
        ),
        "panel_placement": (
            "Phase 07 door/window lines snap to nearby parallel regularized PLI contour segments; "
            "the room with the most associated doors is central and its doors swing away from it"
        ),
        "wall_segment_snap_max_distance_m": WALL_SEGMENT_SNAP_MAX_DISTANCE_M,
        "vector_grayscale_step": VECTOR_GRAY_STEP,
        "floor_levels": results,
    }
    # Catch view-specific placement drift before publishing the floorplan artifacts.
    displayed_centers: dict[tuple[str, int, int], tuple[float, float]] = {}
    for floor in results:
        for item in floor["displayed_instances"]:
            key = (item["scan_id"], item["source_class_id"], item["instance_id"])
            center = tuple(item["final_center_xz_m"])
            previous = displayed_centers.setdefault(key, center)
            assert previous == center, f"Instance {key} moved between floorplan views"
    manifest_path = output_dir / "floorplan_instance_overlays_manifest.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(manifest_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = generate_floorplan_instance_overlays(args.artifacts_dir, args.output_dir)
    print(f"Generated {len(result['floor_levels'])} floorplan instance overlays in {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
