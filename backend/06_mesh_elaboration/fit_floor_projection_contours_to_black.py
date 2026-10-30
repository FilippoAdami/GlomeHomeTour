#!/usr/bin/env python3
"""Trim fitted floor-view wall contours toward the projected black floor mask."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Point, Polygon
from surface_footprints import polygons, regularize_orthogonal_footprint

from project_floor_slices_onto_planes import (
    BLACK_MASK_MAX_CHANNEL,
    WALL_CONTOUR_COLOR,
    WALL_CONTOUR_WIDTH_PX,
    _wall_contour_masks,
)

AXIS_BACKGROUND_RGB = np.array([242, 240, 237], dtype=np.uint8)
# shortcut: merge wall-shell fragments within 2 px; lower this if smaller corners matter.
CONTOUR_SIMPLIFY_TOLERANCE_PX = 2.0
# shortcut: allow one pixel of raster mismatch; tighten if subpixel black-boundary precision is required.
BLACK_COVERAGE_GUARD_PX = 1
TOUCH_DISTANCE_PX = 1.5
MIN_SIGNIFICANT_RECTANGLE_REDUCTION_FRACTION = 0.01
BLACK_BOUNDARY_SIMPLIFY_TOLERANCE_PX = 2.0
MIN_FLOOR_POLYGON_SEGMENT_M = 0.10
DIMENSION_LABEL_FONT_SCALE = 0.42


def _axis_bbox(pixels: np.ndarray) -> tuple[int, int, int, int]:
    background = np.all(np.abs(pixels.astype(np.int16) - AXIS_BACKGROUND_RGB) <= 1, axis=2)
    ys, xs = np.where(background)
    if len(xs) < 1000:
        raise ValueError("Could not locate the plotted image area in the floor projection")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _polygon_mask(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return mask.astype(bool)


def _line_points(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = max(1, int(np.ceil(np.linalg.norm(b - a))))
    t = np.linspace(0.0, 1.0, length + 1)
    points = np.rint(a[None, :] + t[:, None] * (b - a)[None, :]).astype(np.int32)
    points = np.unique(points, axis=0)
    return points[:, 0], points[:, 1]


def _shift_edge(points: np.ndarray, edge: int, offset: float, normal: np.ndarray) -> np.ndarray:
    moved = points.copy()
    moved[edge] += offset * normal
    moved[(edge + 1) % len(points)] += offset * normal
    return moved


def _signed_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _rectangle_mask(rectangle: dict, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if rectangle.get("corners_px"):
        points = np.rint(rectangle["corners_px"]).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
    else:
        left, top, right, bottom = rectangle["bbox_px"]
        mask[max(0, top):min(shape[0], bottom + 1),
             max(0, left):min(shape[1], right + 1)] = 1
    return mask.astype(bool)


def _snap_rectangles_to_contour(rectangles: list[dict], contour: np.ndarray,
                               pixels_per_meter_x: float,
                               pixels_per_meter_y: float) -> list[dict]:
    """Extend near-wall hole edges to the contour so no narrow floor strip remains."""
    scale = np.array([pixels_per_meter_x, pixels_per_meter_y], dtype=np.float64)
    boundary = np.asarray(contour, dtype=np.float64) / scale
    outer = Polygon(boundary)
    if not outer.is_valid or outer.area <= 1e-8:
        return []

    snapped = []
    for rectangle_index, rectangle in enumerate(rectangles):
        if not rectangle.get("corners_px"):
            continue
        corners = np.asarray(rectangle["corners_px"], dtype=np.float64) / scale
        constraints: dict[int, list[tuple[np.ndarray, float]]] = {}
        edge_matches = []
        for edge_index, (start, end) in enumerate(zip(corners, np.roll(corners, -1, axis=0))):
            direction = end - start
            length = float(np.linalg.norm(direction))
            if length <= 1e-8:
                continue
            axis = direction / length
            normal = np.array([-axis[1], axis[0]])
            best = None
            for contour_index, (wall_start, wall_end) in enumerate(
                zip(boundary, np.roll(boundary, -1, axis=0))
            ):
                wall_direction = wall_end - wall_start
                wall_length = float(np.linalg.norm(wall_direction))
                wall_axis = wall_direction / max(wall_length, 1e-8)
                cross = axis[0] * wall_axis[1] - axis[1] * wall_axis[0]
                if wall_length <= 1e-8 or abs(float(cross)) > 0.035:
                    continue
                projections = np.array([(wall_start - start) @ axis, (wall_end - start) @ axis])
                overlap = min(length, float(projections.max())) - max(0.0, float(projections.min()))
                if overlap < 0.8 * length:
                    continue
                gap = float((wall_start - start) @ normal)
                midpoint = (start + end) * 0.5 + normal * gap * 0.5
                if not 1e-6 < abs(gap) < 0.10 or not outer.buffer(1e-6).covers(Point(midpoint)):
                    continue
                candidate = (abs(gap), contour_index, wall_start, wall_axis)
                if best is None or candidate[0] < best[0]:
                    best = candidate
            if best is not None:
                gap, contour_index, wall_start, wall_axis = best
                wall_normal = np.array([-wall_axis[1], wall_axis[0]])
                line_offset = float(wall_normal @ wall_start)
                constraints.setdefault(edge_index, []).append((wall_normal, line_offset))
                next_corner = (edge_index + 1) % len(corners)
                constraints.setdefault(next_corner, []).append((wall_normal, line_offset))
                edge_matches.append({
                    "rectangle_edge_index": edge_index,
                    "contour_edge_index": contour_index,
                    "gap_m": gap,
                })
        if edge_matches:
            moved = corners.copy()
            for corner_index, lines in constraints.items():
                normals = np.stack([normal for normal, _ in lines])
                offsets = np.asarray([offset for _, offset in lines])
                moved[corner_index] += np.linalg.lstsq(
                    normals, offsets - normals @ moved[corner_index], rcond=None
                )[0]
            rectangle["corners_px"] = (moved * scale).tolist()
            rectangle["wall_contour_snaps"] = edge_matches
            snapped.append({"rectangle_index": rectangle_index, "edges": edge_matches})
    return snapped


def _rectangle_areas(mask: np.ndarray, rectangles: list[dict]) -> list[int]:
    return [int(np.count_nonzero(mask & _rectangle_mask(rectangle, mask.shape)))
            for rectangle in rectangles]


def _measure_segments(points: np.ndarray, prefix: str, pixels_per_meter_x: float,
                      pixels_per_meter_y: float, component_id: int | None = None,
                      component_area_m2: float | None = None) -> list[dict]:
    segments = []
    for index, (start, end) in enumerate(zip(points, np.roll(points, -1, axis=0)), start=1):
        dx_m = (float(end[0]) - float(start[0])) / pixels_per_meter_x
        dy_m = (float(end[1]) - float(start[1])) / pixels_per_meter_y
        length_m = float(np.hypot(dx_m, dy_m))
        if length_m <= 1e-8:
            continue
        segment = {
            "id": f"{prefix}{index:02d}",
            "start_px": [float(start[0]), float(start[1])],
            "end_px": [float(end[0]), float(end[1])],
            "length_m": length_m,
        }
        if component_id is not None:
            segment["component_id"] = component_id
            segment["component_area_m2"] = component_area_m2
        segments.append(segment)
    return segments


def _remove_short_polygon_edges(points: np.ndarray, pixels_per_meter_x: float,
                                pixels_per_meter_y: float) -> list[np.ndarray] | None:
    """Fit right-angle rings and remove short edges in world metres."""
    points = np.asarray(points, dtype=np.float64)
    scale = np.array([1.0 / pixels_per_meter_x, 1.0 / pixels_per_meter_y])
    source = Polygon(points * scale)
    if not source.is_valid or source.area <= 1e-8:
        return None
    try:
        fitted = regularize_orthogonal_footprint(
            source, MIN_FLOOR_POLYGON_SEGMENT_M, tolerance=0.02
        )
    except ValueError:
        return None
    if fitted.geom_type not in ("Polygon", "MultiPolygon"):
        return None
    return [
        np.asarray(ring.coords[:-1], dtype=np.float64) / scale
        for polygon in polygons(fitted)
        for ring in [polygon.exterior, *polygon.interiors]
    ]


def _black_part_dimensions(mask: np.ndarray, pixels_per_meter_x: float,
                           pixels_per_meter_y: float) -> tuple[list[np.ndarray], list[dict]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    simplified, segments, component_id = [], [], 0
    for contour in contours:
        points = cv2.approxPolyDP(
            contour, BLACK_BOUNDARY_SIMPLIFY_TOLERANCE_PX, True
        ).reshape(-1, 2).astype(float)
        if len(points) < 3:
            continue
        rings = _remove_short_polygon_edges(points, pixels_per_meter_x, pixels_per_meter_y)
        if not rings:
            continue
        area_m2 = float(cv2.contourArea(contour) / (pixels_per_meter_x * pixels_per_meter_y))
        for points in rings:
            component_id += 1
            simplified.append(points)
            segments.extend(_measure_segments(
                points, f"B{component_id}.", pixels_per_meter_x, pixels_per_meter_y,
                component_id=component_id, component_area_m2=area_m2,
            ))
    return simplified, segments


def _draw_dimension_labels(pixels: np.ndarray, segments: list[dict],
                           avoid_mask: np.ndarray, allowed_mask: np.ndarray,
                           occupied: list[tuple[int, int, int, int]],
                           color: tuple[int, int, int]) -> None:
    height, width = allowed_mask.shape
    for segment in segments:
        start = np.asarray(segment["start_px"], dtype=float)
        end = np.asarray(segment["end_px"], dtype=float)
        direction = end - start
        length = float(np.linalg.norm(direction))
        normal = np.array([-direction[1], direction[0]]) / max(length, 1e-8)
        text = f"{segment['id']} {segment['length_m']:.2f}m"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, DIMENSION_LABEL_FONT_SCALE, 1
        )
        midpoint = (start + end) / 2.0
        candidates = []
        for side in (-1.0, 1.0):
            for offset in (8.0, 18.0, 30.0, 44.0, 60.0, 80.0, 110.0, 145.0):
                center = midpoint + side * offset * normal
                x = int(round(center[0] - text_width / 2))
                y = int(round(center[1] + text_height / 2))
                left, top = x - 3, y - text_height - 3
                right, bottom = x + text_width + 3, y + baseline + 2
                clipped_left, clipped_top = max(0, left), max(0, top)
                clipped_right, clipped_bottom = min(width, right), min(height, bottom)
                if clipped_left >= clipped_right or clipped_top >= clipped_bottom:
                    continue
                overlap = sum(
                    max(0, min(right, ox2) - max(left, ox1))
                    * max(0, min(bottom, oy2) - max(top, oy1))
                    for ox1, oy1, ox2, oy2 in occupied
                )
                region = avoid_mask[clipped_top:clipped_bottom, clipped_left:clipped_right]
                allowed = allowed_mask[clipped_top:clipped_bottom, clipped_left:clipped_right]
                occupied_fraction = float(region.mean()) if region.size else 1.0
                allowed_fraction = float(allowed.mean()) if allowed.size else 0.0
                off_canvas = (right - left) * (bottom - top) - max(0, clipped_right - clipped_left) * max(0, clipped_bottom - clipped_top)
                box_area = max(1, (right - left) * (bottom - top))
                score = overlap / box_area * 10000 + (1 - allowed_fraction) * 1000 + occupied_fraction * 10 + off_canvas
                candidates.append((score, left, top, right, bottom))
        if not candidates:
            continue
        _, left, top, right, bottom = min(candidates)
        left, top = max(0, left), max(0, top)
        right, bottom = min(width - 1, right), min(height - 1, bottom)
        occupied.append((left, top, right, bottom))
        anchor = (
            int(np.clip(midpoint[0], left, right)),
            int(np.clip(midpoint[1], top, bottom)),
        )
        cv2.line(pixels, tuple(np.rint(midpoint).astype(int)), anchor, color, 1, cv2.LINE_AA)
        cv2.rectangle(pixels, (left, top), (right, bottom), (255, 255, 255), thickness=-1)
        cv2.rectangle(pixels, (left, top), (right, bottom), color, thickness=1)
        cv2.putText(
            pixels, text, (left + 3, bottom - baseline - 1), cv2.FONT_HERSHEY_SIMPLEX,
            DIMENSION_LABEL_FONT_SCALE, (25, 25, 25), 1, cv2.LINE_AA,
        )


def _first_valid_contact(points: np.ndarray, edge: int, inward: np.ndarray,
                         distance_map: np.ndarray, current_mask: np.ndarray,
                         black: np.ndarray, rectangles: list[dict],
                         baseline_rectangles: list[int]) -> tuple[np.ndarray, np.ndarray, dict] | None:
    a, b = points[edge], points[(edge + 1) % len(points)]
    xs, ys = _line_points(a, b)
    start_distance = float(distance_map[ys, xs].min())
    if start_distance <= TOUCH_DISTANCE_PX:
        return None
    max_shift = min(max(current_mask.shape), int(np.ceil(start_distance)) + 2)
    guard = np.ones((2 * BLACK_COVERAGE_GUARD_PX + 1,) * 2, dtype=np.uint8)

    for amount in range(1, max_shift + 1):
        candidate_points = _shift_edge(points, edge, amount, inward)
        line_x, line_y = _line_points(
            candidate_points[edge], candidate_points[(edge + 1) % len(points)]
        )
        if float(distance_map[line_y, line_x].min()) > TOUCH_DISTANCE_PX:
            continue

        candidate_mask = _polygon_mask(candidate_points, current_mask.shape)
        if candidate_mask.sum() >= current_mask.sum():
            return None
        protected = cv2.dilate(candidate_mask.astype(np.uint8), guard).astype(bool)
        if np.any(black & ~protected):
            return None

        rectangle_areas = _rectangle_areas(candidate_mask, rectangles)
        reductions = [before - after for before, after in zip(baseline_rectangles, rectangle_areas)]
        decreased_rectangles = sum(reduction > 0 for reduction in reductions)
        significant_reductions = sum(
            reduction >= max(1, int(np.ceil(before * MIN_SIGNIFICANT_RECTANGLE_REDUCTION_FRACTION)))
            for before, reduction in zip(baseline_rectangles, reductions)
        )
        if decreased_rectangles and significant_reductions < 2:
            return None

        return candidate_points, candidate_mask, {
            "offset_px": amount,
            "rectangle_areas_before_px": baseline_rectangles,
            "rectangle_areas_after_px": rectangle_areas,
            "rectangles_reduced": decreased_rectangles,
            "significant_rectangle_reductions": significant_reductions,
            "free_white_pixels_removed": int(current_mask.sum() - candidate_mask.sum()),
        }
    return None


def _fit_polygon(image: Image.Image, rectangles: list[dict], pixels_per_meter: float,
                 pixels_per_meter_y: float,
                 contour_points_px: np.ndarray | None = None) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray, np.ndarray]:
    pixels = np.asarray(image.convert("RGB"))
    if contour_points_px is None:
        _, interior = _wall_contour_masks(image, pixels_per_meter, pixels_per_meter_y)
        contours, _ = cv2.findContours(interior.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("The fitted wall contour does not enclose a usable area")
        contour = max(contours, key=cv2.contourArea)
        points = cv2.approxPolyDP(contour, CONTOUR_SIMPLIFY_TOLERANCE_PX, True).reshape(-1, 2).astype(float)
    else:
        points = np.asarray(contour_points_px, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("Regularized wall contour coordinates are invalid")
    if len(points) < 3:
        raise ValueError("The fitted wall contour has fewer than three corners")
    hole_snaps = _snap_rectangles_to_contour(
        rectangles, points, pixels_per_meter, pixels_per_meter_y
    )
    initial_points = points.copy()
    current_mask = _polygon_mask(points, pixels.shape[:2])
    if contour_points_px is not None:
        interior = current_mask

    x0, y0, x1, y1 = _axis_bbox(pixels)
    in_plot = np.zeros(interior.shape, dtype=bool)
    in_plot[y0:y1, x0:x1] = True
    source_black = (np.max(pixels, axis=2) <= BLACK_MASK_MAX_CHANNEL) & in_plot
    hole_mask = np.zeros(interior.shape, dtype=bool)
    for rectangle in rectangles:
        hole_mask |= _rectangle_mask(rectangle, interior.shape)
    black = source_black & interior & ~hole_mask

    if not current_mask.any() or not black.any():
        raise ValueError("The wall contour or retained black projection is empty")
    # shortcut: one-pixel coverage tolerance absorbs mask differences from contour simplification.
    guard = np.ones((2 * BLACK_COVERAGE_GUARD_PX + 1,) * 2, dtype=np.uint8)
    if np.any(black & ~cv2.dilate(current_mask.astype(np.uint8), guard).astype(bool)):
        raise ValueError("Simplified wall contour cannot cover all retained black pixels")

    moves = []
    if contour_points_px is None:
        distance_map = distance_transform_edt(~black)
        for _ in range(len(points) + 1):
            moved_in_pass = False
            for edge in range(len(points)):
                a, b = points[edge], points[(edge + 1) % len(points)]
                direction = b - a
                length = float(np.linalg.norm(direction))
                if length <= 1e-6:
                    continue
                normal = np.array([-direction[1], direction[0]]) / length
                area = _signed_area(points)
                inward = None
                for sign in (-1.0, 1.0):
                    trial = _shift_edge(points, edge, sign, normal)
                    if abs(_signed_area(trial)) < abs(area):
                        inward = sign * normal
                        break
                if inward is None:
                    continue
                rectangle_areas = _rectangle_areas(current_mask, rectangles)
                result = _first_valid_contact(
                    points, edge, inward, distance_map, current_mask, black,
                    rectangles, rectangle_areas,
                )
                if result is None:
                    continue
                points, current_mask, move = result
                move.update({
                    "segment_index": edge,
                    "offset_m": move["offset_px"] / pixels_per_meter,
                })
                moves.append(move)
                moved_in_pass = True
            if not moved_in_pass:
                break

    retained = black & in_plot
    guard_mask = cv2.dilate(current_mask.astype(np.uint8), guard).astype(bool)
    assert not np.any(retained & ~guard_mask), "Contour movement cut off retained black pixels"
    report = {
        "source_black_pixels_in_plot": int(source_black.sum()),
        "black_pixels_outside_wall_contour_removed": int((source_black & ~interior).sum()),
        "black_pixels_inside_hole_rectangles_removed": int((source_black & interior & hole_mask).sum()),
        "hole_rectangle_wall_snaps": hole_snaps,
        "retained_black_pixels": int(retained.sum()),
        "initial_free_white_pixels": int(_polygon_mask(
            initial_points, interior.shape).sum() - retained.sum()),
        "final_free_white_pixels": int(current_mask.sum() - retained.sum()),
        "segment_moves": moves,
        "segment_count": len(points),
    }
    return points, report, retained, current_mask, hole_mask


def _render_result(source_path: Path, output_path: Path, rectangles: list[dict],
                   pixels_per_meter: float, pixels_per_meter_y: float,
                   contour_points_px: np.ndarray | None = None) -> dict:
    image = Image.open(source_path).convert("RGB")
    pixels = np.asarray(image).copy()
    axis_bbox = _axis_bbox(pixels)
    x0, y0, x1, y1 = axis_bbox
    points, report, retained_black, fitted_mask, hole_mask = _fit_polygon(
        image, rectangles, pixels_per_meter, pixels_per_meter_y, contour_points_px
    )

    red = (pixels[:, :, 0] >= 180) & (pixels[:, :, 1] <= 120) & (pixels[:, :, 2] <= 120)
    pixels[red] = AXIS_BACKGROUND_RGB
    yy, xx = np.indices(pixels.shape[:2])
    in_plot = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
    black = (np.max(np.asarray(image), axis=2) <= BLACK_MASK_MAX_CHANNEL) & in_plot
    pixels[black] = AXIS_BACKGROUND_RGB
    fill_mask = fitted_mask & in_plot & ~hole_mask
    pixels[fill_mask] = 0
    assert not np.any(fill_mask & hole_mask), "Final black fill overlaps a hole rectangle"
    assert np.all(pixels[fill_mask] == 0), "Final fitted interior was not filled black"
    report["newly_filled_interior_pixels"] = int(np.count_nonzero(fill_mask & ~retained_black))

    output = Image.fromarray(pixels)
    draw = ImageDraw.Draw(output)
    vertices = [tuple(np.rint(point).astype(int)) for point in points]
    draw.line(vertices + [vertices[0]], fill=WALL_CONTOUR_COLOR, width=WALL_CONTOUR_WIDTH_PX)
    # Measure the filled polygon itself, not the jagged source projection boundary.
    black_contours, black_segments = _black_part_dimensions(fill_mask, pixels_per_meter, pixels_per_meter_y)
    contour_segments = _measure_segments(points, "C", pixels_per_meter, pixels_per_meter_y)
    pixels = np.asarray(output).copy()
    black_part_color = (255, 165, 0)
    for contour in black_contours:
        cv2.polylines(pixels, [np.rint(contour).astype(np.int32)], True, black_part_color, 2, cv2.LINE_AA)
    cv2.polylines(
        pixels, [np.rint(points).astype(np.int32)], True, WALL_CONTOUR_COLOR,
        WALL_CONTOUR_WIDTH_PX, cv2.LINE_AA,
    )
    occupied: list[tuple[int, int, int, int]] = []
    _draw_dimension_labels(pixels, contour_segments, fill_mask, in_plot, occupied, WALL_CONTOUR_COLOR)
    _draw_dimension_labels(pixels, black_segments, retained_black, in_plot, occupied, black_part_color)
    output = Image.fromarray(pixels)
    output.save(output_path)
    report["image"] = output_path.name
    report["segment_dimensions"] = {
        "units": "m",
        "minimum_floor_polygon_segment_length_m": MIN_FLOOR_POLYGON_SEGMENT_M,
        "black_boundary_simplification_tolerance_px": BLACK_BOUNDARY_SIMPLIFY_TOLERANCE_PX,
        "black_part_boundary_source": "final filled polygon mask: fitted outer contour minus hole rectangles",
        "fitted_contour_segments": contour_segments,
        "black_part_segments": black_segments,
        "black_part_contour_count": len(black_contours),
    }
    return report


def run(instances_dir: Path, level_id: int | None = None) -> Path:
    level_dir = instances_dir / "intermediate_floor_levels"
    source_manifest = level_dir / "floor_level_collapsed_first_refinement_holes_manifest.json"
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    levels = manifest.get("levels") if isinstance(manifest, dict) else None
    if not isinstance(levels, list):
        raise ValueError("Hole manifest must contain a levels list")
    selected = [level for level in levels if level_id is None or level.get("level") == level_id]
    if level_id is not None and len(selected) != 1:
        raise ValueError(f"Floor level {level_id} is missing or duplicated")
    output_manifest = level_dir / "floor_level_contour_fitted_manifest.json"
    existing = {}
    if level_id is not None and output_manifest.is_file():
        old = json.loads(output_manifest.read_text(encoding="utf-8"))
        existing = {item["level"]: item for item in old.get("levels", []) if isinstance(item, dict)}

    with tempfile.TemporaryDirectory(prefix=".floor_contour_fit_", dir=level_dir) as temp_name:
        temp_dir = Path(temp_name)
        staged = []
        for level in selected:
            areas = level.get("areas")
            if not isinstance(areas, list):
                raise ValueError(f"Floor level {level.get('level')} has no area records")
            area_results = []
            for area in areas:
                source_name = area.get("regularized_first_refinement_image", area.get("first_refinement_image"))
                scale = area.get("first_refinement", {}).get("pixels_per_meter")
                hole_detection = area.get("regularized_hole_detection", area.get("hole_detection", {}))
                rectangles = hole_detection.get("rectangles", [])
                contour_data = area.get("contour_regularization", {})
                contour_points_px = contour_data.get("points_px")
                if contour_points_px is not None:
                    contour_points_px = np.asarray(contour_points_px, dtype=float)
                if not isinstance(source_name, str) or not isinstance(scale, (int, float)):
                    raise ValueError(f"Floor level {level.get('level')} has incomplete refinement metadata")
                if not isinstance(rectangles, list):
                    raise ValueError("Hole rectangles must be a list")
                source_path = level_dir / source_name
                if not source_path.is_file():
                    raise ValueError(f"First-refinement image is missing: {source_path}")
                output_name = source_path.stem + "_contour_fitted.png"
                temp_output = temp_dir / output_name
                report = _render_result(
                    source_path, temp_output, rectangles, float(scale), float(scale), contour_points_px
                )
                report.update({"area_id": area.get("area_id"), "source_image": source_name})
                area_results.append(report)
                staged.append((temp_output, level_dir / output_name))
            existing[level["level"]] = {"level": level["level"], "areas": area_results}

        updated = {
            "method": "preserve supplied regularized wall contours (otherwise fit to first black contact), then fill the interior except for hole rectangles and annotate segment lengths",
            "contour_simplify_tolerance_px": CONTOUR_SIMPLIFY_TOLERANCE_PX,
            "black_coverage_guard_px": BLACK_COVERAGE_GUARD_PX,
            "minimum_significant_rectangle_reduction_fraction": MIN_SIGNIFICANT_RECTANGLE_REDUCTION_FRACTION,
            "black_boundary_simplify_tolerance_px": BLACK_BOUNDARY_SIMPLIFY_TOLERANCE_PX,
            "levels": [existing[key] for key in sorted(existing)],
        }
        temp_manifest = temp_dir / output_manifest.name
        temp_manifest.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
        for temp_path, output_path in staged:
            temp_path.replace(output_path)
        temp_manifest.replace(output_manifest)
    for level in selected:
        for area in existing[level["level"]]["areas"]:
            print(f"Level {level['level']:03d}, area {area['area_id']:03d}: "
                  f"moved {len(area['segment_moves'])} contour segments -> {area['image']}")
    return output_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path, required=True,
                        help="Stage 06 output directory containing floor projection images")
    parser.add_argument("--level", type=int, help="Process one floor level and preserve other output records")
    args = parser.parse_args(argv)
    try:
        run(args.instances_dir, args.level)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: Could not fit floor projection contours: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
