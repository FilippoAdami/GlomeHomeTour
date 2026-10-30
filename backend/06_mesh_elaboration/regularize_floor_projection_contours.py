#!/usr/bin/env python3
"""Align corresponding wall-contour edges across floor levels before infill."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from fit_floor_projection_contours_to_black import (
    AXIS_BACKGROUND_RGB,
    CONTOUR_SIMPLIFY_TOLERANCE_PX,
    _axis_bbox,
    _rectangle_mask,
)
from project_floor_slices_onto_planes import (
    BLACK_MASK_MAX_CHANNEL,
    WALL_CONTOUR_COLOR,
    WALL_CONTOUR_WIDTH_PX,
    _draw_hole_rectangles,
    _plane_frame,
    _regularize_black_mask,
    _wall_contour_masks,
)
from surface_footprints import polygons, regularize_orthogonal_footprint
from shapely.geometry import Polygon

MIN_CONTOUR_SEGMENT_M = 0.10


def _contour_points(image: Image.Image, pixels_per_meter: float) -> np.ndarray:
    _, inside = _wall_contour_masks(image, pixels_per_meter, pixels_per_meter)
    contours, _ = cv2.findContours(inside.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Wall contour does not enclose an area")
    points = cv2.approxPolyDP(
        max(contours, key=cv2.contourArea), CONTOUR_SIMPLIFY_TOLERANCE_PX, True
    ).reshape(-1, 2).astype(float)
    if len(points) < 3:
        raise ValueError("Wall contour has fewer than three simplified segments")
    return points


def _signed_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _image_to_world_xz(points_px: np.ndarray, area: dict, image: Image.Image) -> np.ndarray:
    x0, y0, x1, y1 = _axis_bbox(np.asarray(image.convert("RGB")))
    low, high = np.asarray(area["expanded_plane_bounds_uv_m"], dtype=float)
    span = max(float(np.max(high - low)), 1.0)
    padding = max(span * 0.02, 0.05)
    uv = np.empty_like(points_px, dtype=float)
    uv[:, 0] = low[0] - padding + (points_px[:, 0] - x0) / (x1 - x0) * (high[0] - low[0] + 2 * padding)
    uv[:, 1] = high[1] + padding - (points_px[:, 1] - y0) / (y1 - y0) * (high[1] - low[1] + 2 * padding)
    plane, axis_u, axis_v = _plane_frame(area["plane_equation"], "floor-area")
    points_3d = uv[:, 0, None] * axis_u + uv[:, 1, None] * axis_v - plane[3] * plane[:3]
    world = points_3d[:, [0, 2]]
    return world if _signed_area(world) > 0 else world[::-1].copy()


def _world_xz_to_image(points_xz: np.ndarray, area: dict, image: Image.Image) -> np.ndarray:
    x0, y0, x1, y1 = _axis_bbox(np.asarray(image.convert("RGB")))
    low, high = np.asarray(area["expanded_plane_bounds_uv_m"], dtype=float)
    span = max(float(np.max(high - low)), 1.0)
    padding = max(span * 0.02, 0.05)
    plane, axis_u, axis_v = _plane_frame(area["plane_equation"], "floor-area")
    points_3d = np.column_stack((points_xz[:, 0], np.zeros(len(points_xz)), points_xz[:, 1]))
    points_3d[:, 1] = -(plane[0] * points_3d[:, 0] + plane[2] * points_3d[:, 2] + plane[3]) / plane[1]
    uv = np.column_stack((points_3d @ axis_u, points_3d @ axis_v))
    points_px = np.empty_like(uv)
    points_px[:, 0] = x0 + (uv[:, 0] - (low[0] - padding)) / (high[0] - low[0] + 2 * padding) * (x1 - x0)
    points_px[:, 1] = y0 + ((high[1] + padding) - uv[:, 1]) / (high[1] - low[1] + 2 * padding) * (y1 - y0)
    return points_px


def _edges(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ends = np.roll(points, -1, axis=0)
    vectors = ends - points
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 1e-8):
        raise ValueError("Contour contains a zero-length segment")
    directions = vectors / lengths[:, None]
    normals = np.column_stack((directions[:, 1], -directions[:, 0]))
    return ends, directions, normals, lengths


def _match_cost(reference_points: np.ndarray, source_points: np.ndarray) -> np.ndarray:
    _, ref_directions, ref_normals, ref_lengths = _edges(reference_points)
    _, src_directions, src_normals, src_lengths = _edges(source_points)
    ref_mid = (reference_points + np.roll(reference_points, -1, axis=0)) / 2
    src_mid = (source_points + np.roll(source_points, -1, axis=0)) / 2
    costs = np.empty((len(source_points), len(reference_points)), dtype=float)
    for source_index, (mid, direction, normal, length) in enumerate(
        zip(src_mid, src_directions, src_normals, src_lengths)
    ):
        for reference_index, (ref_midpoint, ref_direction, ref_normal, ref_length) in enumerate(
            zip(ref_mid, ref_directions, ref_normals, ref_lengths)
        ):
            facing = float(np.dot(normal, ref_normal))
            angle = float(np.arccos(np.clip(facing, -1.0, 1.0)))
            distance = abs(float(np.dot(mid - ref_midpoint, ref_normal)))
            source_interval = sorted(float(np.dot(point, ref_direction)) for point in (
                source_points[source_index], np.roll(source_points, -1, axis=0)[source_index]
            ))
            reference_interval = sorted(float(np.dot(point, ref_direction)) for point in (
                reference_points[reference_index], np.roll(reference_points, -1, axis=0)[reference_index]
            ))
            gap = max(0.0, reference_interval[0] - source_interval[1], source_interval[0] - reference_interval[1])
            costs[source_index, reference_index] = (
                angle * 0.5 + distance + gap * 0.5 + abs(float(length - ref_length)) * 0.05
                + (10.0 if facing <= 0 else 0.0)
                + source_index * 1e-9 + reference_index * 1e-12
            )
    return costs


def _intersect_lines(first_normal: np.ndarray, first_offset: float,
                     second_normal: np.ndarray, second_offset: float) -> np.ndarray:
    matrix = np.stack((first_normal, second_normal))
    if abs(float(np.linalg.det(matrix))) <= 1e-7:
        raise ValueError("Regularized neighboring contour edges are parallel")
    return np.linalg.solve(matrix, np.array([first_offset, second_offset]))


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p, q, r):
        first, second = q - p, r - p
        return float(first[0] * second[1] - first[1] * second[0])

    def on_segment(p, q, r):
        return (min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9
                and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9)

    eps = 1e-9
    ab_c, ab_d = orient(a, b, c), orient(a, b, d)
    cd_a, cd_b = orient(c, d, a), orient(c, d, b)
    return (
        (ab_c * ab_d < -eps and cd_a * cd_b < -eps)
        or (abs(ab_c) <= eps and on_segment(a, c, b))
        or (abs(ab_d) <= eps and on_segment(a, d, b))
        or (abs(cd_a) <= eps and on_segment(c, a, d))
        or (abs(cd_b) <= eps and on_segment(c, b, d))
    )


def _validate_polygon(points: np.ndarray, expected_count: int, original_area: float) -> None:
    assert len(points) == expected_count, "Contour segment count changed during regularization"
    if not np.isfinite(points).all() or abs(_signed_area(points)) <= 1e-6:
        raise ValueError("Regularized contour is degenerate")
    if np.sign(_signed_area(points)) != np.sign(original_area):
        raise ValueError("Regularization reversed the contour winding")
    ends = np.roll(points, -1, axis=0)
    edges = ends - points
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths < MIN_CONTOUR_SEGMENT_M - 1e-8):
        raise ValueError("Regularization produced a contour segment shorter than 10 cm")
    if np.any((np.abs(edges[:, 0]) > 1e-8) & (np.abs(edges[:, 1]) > 1e-8)):
        raise ValueError("Regularization produced a diagonal contour segment")
    if np.any(np.abs(np.sum(np.roll(edges, 1, axis=0) * edges, axis=1)) > 1e-8):
        raise ValueError("Regularization produced neighboring contour edges that are not perpendicular")
    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            if second in (first, (first + 1) % len(points)) or first == (second + 1) % len(points):
                continue
            if _segments_intersect(points[first], ends[first], points[second], ends[second]):
                raise ValueError("Regularized contour self-intersects")


def _parallel_segment_clearance_errors(points: np.ndarray) -> list[dict]:
    ends = np.roll(points, -1, axis=0)
    edges = ends - points
    errors = []
    for first in range(len(points)):
        first_axis = int(np.argmax(np.abs(edges[first])))
        first_interval = sorted(points[[first, (first + 1) % len(points)], first_axis])
        for second in range(first + 1, len(points)):
            second_axis = int(np.argmax(np.abs(edges[second])))
            if first_axis != second_axis:
                continue
            second_interval = sorted(points[[second, (second + 1) % len(points)], second_axis])
            overlap = min(first_interval[1], second_interval[1]) - max(first_interval[0], second_interval[0])
            if overlap <= 1e-8:
                continue
            gap = abs(float(points[first, 1 - first_axis] - points[second, 1 - second_axis]))
            if gap < MIN_CONTOUR_SEGMENT_M - 1e-8:
                errors.append({"segment_indices": [first, second], "parallel_gap_m": gap,
                               "overlap_m": float(overlap)})
    return errors


def _clean_aligned_contour(points: np.ndarray) -> tuple[np.ndarray, dict]:
    source = Polygon(points)
    flagged_errors = _parallel_segment_clearance_errors(points)
    original_area = float(source.area)
    removed_components = 0
    if flagged_errors:
        opened = source.buffer(-MIN_CONTOUR_SEGMENT_M / 2, join_style=2).buffer(
            MIN_CONTOUR_SEGMENT_M / 2, join_style=2
        )
        parts = polygons(opened)
        if not parts:
            raise ValueError("10 cm parallel-segment cleanup removed the entire floor contour")
        removed_components += len(parts) - 1
        source = max(parts, key=lambda part: part.area)
    footprint = regularize_orthogonal_footprint(source, MIN_CONTOUR_SEGMENT_M, tolerance=0.02)
    if footprint.geom_type != "Polygon" or footprint.interiors:
        raise ValueError("Aligned wall contour cleanup must produce one hole-free polygon")
    errors = _parallel_segment_clearance_errors(np.asarray(footprint.exterior.coords[:-1]))
    while errors:
        opened = footprint.buffer(-MIN_CONTOUR_SEGMENT_M / 2, join_style=2).buffer(
            MIN_CONTOUR_SEGMENT_M / 2, join_style=2
        )
        parts = polygons(opened)
        if not parts:
            raise ValueError("10 cm parallel-segment cleanup removed the entire floor contour")
        removed_components += len(parts) - 1
        footprint = max(parts, key=lambda part: part.area)
        footprint = regularize_orthogonal_footprint(
            footprint, MIN_CONTOUR_SEGMENT_M, tolerance=0.02
        )
        if footprint.geom_type != "Polygon" or footprint.interiors:
            raise ValueError("10 cm parallel-segment cleanup must leave one hole-free polygon")
        errors = _parallel_segment_clearance_errors(np.asarray(footprint.exterior.coords[:-1]))
    report = {
        "minimum_parallel_segment_gap_m": MIN_CONTOUR_SEGMENT_M,
        "flagged_parallel_segment_pairs": flagged_errors,
        "remaining_parallel_segment_pairs": errors,
        "narrow_components_removed": int(removed_components),
        "removed_area_m2": max(0.0, original_area - float(footprint.area)),
    }
    cleaned = np.asarray(footprint.exterior.coords[:-1], dtype=float)
    cleaned = cleaned if _signed_area(cleaned) * _signed_area(points) > 0 else cleaned[::-1].copy()
    return cleaned, report


def _regularize_group(records: list[dict]) -> None:
    reference = max(records, key=lambda record: (len(record["world_points"]), -record["level"]))
    reference_points = reference["world_points"]
    reference_count = len(reference_points)
    mappings = {id(reference): {index: index for index in range(reference_count)}}
    for record in records:
        if record is reference:
            continue
        points = record["world_points"]
        if len(points) > reference_count:
            raise ValueError("Reference contour must have the most segments")
        rows, columns = linear_sum_assignment(_match_cost(reference_points, points))
        if len(rows) != len(points):
            raise ValueError("Could not match every segment in the smaller contour")
        mappings[id(record)] = {int(row): int(column) for row, column in zip(rows, columns)}

    _, _, reference_normals, _ = _edges(reference_points)
    member_edges: dict[int, list[tuple[dict, int]]] = {index: [(reference, index)] for index in range(reference_count)}
    for record in records:
        if record is reference:
            continue
        for source_edge, reference_edge in mappings[id(record)].items():
            member_edges[reference_edge].append((record, source_edge))

    target_lines: dict[int, tuple[np.ndarray, float]] = {}
    for reference_edge, members in member_edges.items():
        normal = reference_normals[reference_edge]
        supports = []
        for record, edge_index in members:
            points = record["world_points"]
            end = np.roll(points, -1, axis=0)[edge_index]
            midpoint = (points[edge_index] + end) / 2
            edge_normal = _edges(points)[2][edge_index]
            if float(np.dot(normal, edge_normal)) <= 0:
                raise ValueError("Matched contour segments face opposite directions")
            supports.append(float(np.dot(normal, midpoint)))
        target_lines[reference_edge] = (normal, max(supports))

    for record in records:
        points = record["world_points"]
        original_area = _signed_area(points)
        _, _, normals, _ = _edges(points)
        offsets = np.sum(normals * points, axis=1)
        if record is reference:
            edge_to_reference = {index: index for index in range(reference_count)}
        else:
            edge_to_reference = {source: target for source, target in mappings[id(record)].items()}
        for edge_index, reference_edge in edge_to_reference.items():
            normals[edge_index], offsets[edge_index] = target_lines[reference_edge]
        result = np.array([
            _intersect_lines(normals[(index - 1) % len(points)], offsets[(index - 1) % len(points)],
                             normals[index], offsets[index])
            for index in range(len(points))
        ])
        record["aligned_world_points"] = result
        result, clearance = _clean_aligned_contour(result)
        _validate_polygon(result, len(result), original_area)
        record["regularized_world_points"] = result
        record["segments_removed_after_alignment"] = len(record["aligned_world_points"]) - len(result)
        record["parallel_segment_clearance"] = clearance
        record["reference"] = reference
        record["edge_to_reference"] = edge_to_reference
        record["unmatched_segment_indices"] = (
            [index for index in range(reference_count) if len(member_edges[index]) == 1]
            if record is reference else []
        )


def _draw_contour(image: Image.Image, points: np.ndarray) -> Image.Image:
    pixels = np.asarray(image.convert("RGB")).copy()
    cv2.polylines(
        pixels, [np.rint(points).astype(np.int32)], True, WALL_CONTOUR_COLOR,
        WALL_CONTOUR_WIDTH_PX, cv2.LINE_AA,
    )
    return Image.fromarray(pixels)


def _clear_rectangle_black_pixels(image: Image.Image, rectangles: list[dict],
                                  axis_bbox: tuple[int, int, int, int]) -> tuple[Image.Image, int]:
    pixels = np.asarray(image.convert("RGB")).copy()
    black = np.max(pixels, axis=2) <= BLACK_MASK_MAX_CHANNEL
    holes = np.zeros(black.shape, dtype=bool)
    for rectangle in rectangles:
        holes |= _rectangle_mask(rectangle, black.shape)
    x0, y0, x1, y1 = axis_bbox
    in_plot = np.zeros(black.shape, dtype=bool)
    in_plot[y0:y1, x0:x1] = True
    removed = black & holes & in_plot
    pixels[removed] = AXIS_BACKGROUND_RGB
    return Image.fromarray(pixels), int(removed.sum())


def run(instances_dir: Path) -> Path:
    level_dir = instances_dir / "intermediate_floor_levels"
    manifest_path = level_dir / "floor_level_collapsed_first_refinement_holes_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    levels = manifest.get("levels") if isinstance(manifest, dict) else None
    if not isinstance(levels, list) or not levels:
        raise ValueError("Hole manifest must contain floor levels")

    records = []
    for level in levels:
        for area in level.get("areas", []):
            source_name = area.get("first_refinement_image")
            raw_name = area.get("original_projection_image")
            pixels_per_meter = area.get("first_refinement", {}).get("pixels_per_meter")
            if (not isinstance(source_name, str) or not isinstance(raw_name, str)
                    or not isinstance(pixels_per_meter, (int, float))):
                raise ValueError(f"Floor {level.get('level')} area {area.get('area_id')} has incomplete image metadata")
            source_path, raw_path = level_dir / source_name, level_dir / raw_name
            if not source_path.is_file() or not raw_path.is_file():
                raise ValueError(f"Projection image is missing for floor {level.get('level')} area {area.get('area_id')}")
            with Image.open(source_path) as source:
                contour_image = source.convert("RGB")
                image_points = _contour_points(contour_image, float(pixels_per_meter))
                world_points = _image_to_world_xz(image_points, area, contour_image)
                footprint = regularize_orthogonal_footprint(
                    Polygon(world_points), MIN_CONTOUR_SEGMENT_M, tolerance=0.02
                )
                if footprint.geom_type != "Polygon" or footprint.interiors:
                    raise ValueError("Wall contour regularization must produce one hole-free polygon")
                world_points = np.asarray(footprint.exterior.coords[:-1], dtype=float)
            if _signed_area(world_points) < 0:
                image_points = image_points[::-1].copy()
                world_points = world_points[::-1].copy()
            record = {
                "level": int(level["level"]), "area": area, "image_points": image_points,
                "world_points": world_points, "pixels_per_meter": float(pixels_per_meter),
                "raw_path": raw_path,
            }
            records.append(record)

    grouped: dict[int, list[dict]] = {}
    for record in records:
        area_id = record["area"].get("area_id")
        grouped.setdefault(int(area_id), []).append(record)
    for group in grouped.values():
        _regularize_group(group)

    with tempfile.TemporaryDirectory(prefix=".floor_contour_regularization_", dir=level_dir) as temp_name:
        temp_dir = Path(temp_name)
        staged_images = []
        for record in records:
            area = record["area"]
            level_id, area_id = record["level"], int(area["area_id"])
            original_name = area["first_refinement_image"]
            regularized_name = Path(original_name).stem + "_contour_regularized.png"
            holes_name = Path(original_name).stem + "_contour_regularized_with_holes.png"
            hole_detection = area.get("hole_detection", {})
            rectangles = hole_detection.get("rectangles", [])
            if not isinstance(rectangles, list):
                raise ValueError("Fitted hole rectangles must be a list")
            with Image.open(record["raw_path"]) as raw_image:
                raw_image = raw_image.convert("RGB")
                axis_bbox = _axis_bbox(np.asarray(raw_image))
                base, _ = _regularize_black_mask(raw_image.copy(), axis_bbox, record["pixels_per_meter"])
            base, cleared_black_pixels = _clear_rectangle_black_pixels(base, rectangles, axis_bbox)
            contour_px = _world_xz_to_image(record["regularized_world_points"], area, base)
            base = _draw_contour(base, contour_px)
            hole_metrics = {
                **hole_detection,
                "rectangles": rectangles,
                "rectangle_count": len(rectangles),
                "black_pixels_cleared_inside_rectangles": cleared_black_pixels,
                "rectangles_preserved_after_contour_regularization": True,
                "post_regularization_constraints_applied": False,
            }
            regularized_path, holes_path = temp_dir / regularized_name, temp_dir / holes_name
            base.save(regularized_path)
            _draw_hole_rectangles(base.copy(), rectangles, axis_bbox).save(holes_path)
            staged_images.extend(((regularized_path, level_dir / regularized_name), (holes_path, level_dir / holes_name)))

            matches = [
                {"segment_index": int(edge), "reference_segment_index": int(reference_edge)}
                for edge, reference_edge in sorted(record["edge_to_reference"].items())
            ]
            regularized_world = record["regularized_world_points"]
            original_world = record["world_points"]
            area["regularized_first_refinement_image"] = regularized_name
            area["regularized_holes_overlay_image"] = holes_name
            area["regularized_hole_detection"] = hole_metrics
            area["contour_regularization"] = {
                "method": "one-to-one segment matching by facing, orientation, and finite-segment distance; matched edges use the outermost shared line, then short aligned jogs are removed",
                "reference_level": record["reference"]["level"],
                "reference_area_id": int(record["reference"]["area"]["area_id"]),
                "segment_count_before": int(len(original_world)),
                "segment_count_after_alignment": int(len(record["aligned_world_points"])),
                "segment_count_after": int(len(regularized_world)),
                "segments_removed_after_alignment": int(record["segments_removed_after_alignment"]),
                "parallel_segment_clearance": record["parallel_segment_clearance"],
                "matched_segments_before_minimum_edge_cleanup": matches,
                "unmatched_segment_indices_before_minimum_edge_cleanup": record["unmatched_segment_indices"],
                "max_boundary_deviation_m": float(
                    Polygon(regularized_world).boundary.hausdorff_distance(
                        Polygon(original_world).boundary
                    )
                ),
                "points_px": contour_px.tolist(),
                "points_world_xz_m": regularized_world.tolist(),
            }

        manifest["contour_regularization"] = {
            "stage": "after floor-plane projection and hole detection, before black contour fitting/fill",
            "unmatched_edge_lines_fixed": True,
            "unmatched_edge_endpoints_may_slide_along_their_lines": True,
            "segment_count_preserved_during_alignment": True,
            "minimum_edge_length_enforced_after_alignment": MIN_CONTOUR_SEGMENT_M,
            "hole_rectangles_revalidated_after_contour_regularization": False,
        }
        temp_manifest = temp_dir / manifest_path.name
        temp_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for temp_path, output_path in staged_images:
            os.replace(temp_path, output_path)
        os.replace(temp_manifest, manifest_path)

    for record in records:
        info = record["area"]["contour_regularization"]
        print(f"Level {record['level']:03d}, area {record['area']['area_id']:03d}: "
              f"regularized {info['segment_count_after']} segments, max boundary "
              f"deviation {info['max_boundary_deviation_m']:.3f} m")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path, required=True,
                        help="Stage 06 output directory containing floor projections")
    args = parser.parse_args(argv)
    try:
        run(args.instances_dir)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Error: Could not regularize floor contours: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
