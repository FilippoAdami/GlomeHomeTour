#!/usr/bin/env python3
"""Add approximate wall-segment measurements to regularized and furnished floorplans."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
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
from generate_floorplan_instance_overlays import (
    DEFAULT_ARTIFACTS_DIR,
    SVG_NS,
    _scan_footprint_masks,
)


ORANGE = "#ff7100"
MIN_WALL_SEGMENT_LENGTH_M = 0.10
CONTOUR_APPROXIMATION_PX = 1.5
DIMENSION_OFFSET_PX = 16.0
DIMENSION_EXTENSION_PX = 5.0
TEXT_OFFSET_PX = 18.0
MEASURE_FONT_SIZE_PX = 13
MEASURE_HEADER_HEIGHT_PX = 38
WARNING_TEXT = "Measures are approximations and precision is not guaranteed."
INTER_SCAN_ADJACENCY_DISTANCE_M = 0.70


def _wall_contours(
    vertices: np.ndarray,
    faces: np.ndarray,
    scans: list[dict],
    resolution: int,
    grid: dict,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, np.ndarray]]:
    occupied = np.zeros((resolution, resolution), dtype=np.uint8)
    scan_masks = {}
    for scan_id, mask in _scan_footprint_masks(vertices, faces, scans, resolution, grid).items():
        filled, _ = _fill_small_footprint_holes(
            mask.astype(np.uint8) * 255, float(grid["pixel_size"]),
        )
        occupied |= filled
        scan_masks[scan_id] = filled > 0

    contours, hierarchy = cv2.findContours(
        occupied, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
    )
    if hierarchy is None:
        return occupied > 0, [], scan_masks

    minimum_hole_area = SMALL_FOOTPRINT_HOLE_MAX_AREA_M2 / max(
        float(grid["pixel_size"]) ** 2, 1e-12,
    )
    wall_contours = []
    for index, contour in enumerate(contours):
        depth, parent = 0, int(hierarchy[0, index, 3])
        while parent >= 0:
            depth += 1
            parent = int(hierarchy[0, parent, 3])
        if depth % 2 == 0 or cv2.contourArea(contour) > minimum_hole_area:
            wall_contours.append(contour)
    return occupied > 0, wall_contours, scan_masks


def _inside(mask: np.ndarray, point: np.ndarray) -> bool:
    x, y = np.rint(point).astype(int)
    if x < 0 or y < 0 or x >= mask.shape[1] or y >= mask.shape[0]:
        return False
    return bool(mask[y, x])


def _outside_normal(start: np.ndarray, end: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    delta = end - start
    length = float(np.linalg.norm(delta))
    tangent = delta / length
    normal = np.asarray([-tangent[1], tangent[0]])
    midpoint = (start + end) * 0.5
    for probe_distance in (3.0, 7.0, 12.0):
        left_inside = _inside(occupied, midpoint + normal * probe_distance)
        right_inside = _inside(occupied, midpoint - normal * probe_distance)
        if left_inside != right_inside:
            return -normal if left_inside else normal
    return normal


def _contour_scan_owner(
    start: np.ndarray,
    end: np.ndarray,
    inside_normal: np.ndarray,
    scan_masks: dict[str, np.ndarray],
) -> str | None:
    scores = {scan_id: 0 for scan_id in scan_masks}
    for fraction in np.linspace(0.15, 0.85, 5):
        point = start + (end - start) * fraction
        for scan_id, mask in scan_masks.items():
            if any(_inside(mask, point + inside_normal * distance) for distance in (3, 6, 9)):
                scores[scan_id] += 1
    owner = max(scores, key=scores.get, default=None)
    return owner if owner is not None and scores[owner] else None


def _faces_another_scan(
    start: np.ndarray,
    end: np.ndarray,
    outside_normal: np.ndarray,
    owner_scan_id: str | None,
    scan_masks: dict[str, np.ndarray],
    max_distance_px: int,
) -> bool:
    if owner_scan_id is None:
        return False
    neighbors = [mask for scan_id, mask in scan_masks.items() if scan_id != owner_scan_id]
    if not neighbors:
        return False

    sample_fractions = np.linspace(0.15, 0.85, 7)
    outward_distances = range(3, max_distance_px + 1, 3)
    facing_samples = 0
    for fraction in sample_fractions:
        point = start + (end - start) * fraction
        if any(
            _inside(mask, point + outside_normal * distance)
            for distance in outward_distances
            for mask in neighbors
        ):
            facing_samples += 1
    return facing_samples >= max(1, int(np.ceil(len(sample_fractions) * 0.2)))


def _label_box(center: np.ndarray, width: float) -> tuple[float, float, float, float]:
    return (
        float(center[0] - width * 0.5 - 3), float(center[1] - MEASURE_FONT_SIZE_PX * 0.5 - 3),
        float(center[0] + width * 0.5 + 3), float(center[1] + MEASURE_FONT_SIZE_PX * 0.5 + 3),
    )


def _choose_label_center(
    base: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
    length_px: float,
    label: str,
    resolution: int,
    placed_boxes: list[tuple[float, float, float, float]],
    clutter_mask: np.ndarray,
) -> np.ndarray:
    text_width = len(label) * MEASURE_FONT_SIZE_PX * 0.66
    max_tangent_shift = max(36.0, min(96.0, length_px * 0.4))
    offsets = [0.0]
    for step in np.arange(12.0, max_tangent_shift + 1.0, 12.0):
        offsets.extend((float(step), float(-step)))
    candidates = []
    for normal_shift in (0.0, 12.0, 24.0, 36.0):
        for tangent_shift in offsets:
            center = base + tangent * tangent_shift + normal * normal_shift
            box = _label_box(center, text_width)
            if (box[0] < 2 or box[2] > resolution - 2
                    or box[1] < 2 or box[3] > resolution - 2):
                continue
            if any(
                box[0] < other[2] and box[2] > other[0]
                and box[1] < other[3] and box[3] > other[1]
                for other in placed_boxes
            ):
                continue
            x0, y0 = np.maximum(np.floor(box[:2]).astype(int), 0)
            x1, y1 = np.minimum(np.ceil(box[2:]).astype(int), resolution)
            clutter = cv2.countNonZero(clutter_mask[y0:y1, x0:x1])
            displacement = normal_shift + abs(tangent_shift) * 0.1
            candidates.append((clutter, displacement, center, box))

    if candidates:
        _, _, center, box = min(candidates, key=lambda item: (item[0], item[1]))
        placed_boxes.append(box)
        return center

    center = np.clip(base, [text_width * 0.5 + 5, 12], [resolution - text_width * 0.5 - 5, resolution - 12])
    placed_boxes.append(_label_box(center, text_width))
    return center


def _svg_png_renderer() -> str:
    renderer = next((shutil.which(name) for name in (
        "google-chrome", "chromium", "chromium-browser", "chrome",
    ) if shutil.which(name)), None)
    if renderer is None:
        raise RuntimeError("Rendering furnished measured PNGs requires Chrome or Chromium on PATH")
    return renderer


def _render_svg_png(svg_path: Path, resolution: int) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="floorplan-svg-render-") as temp_dir:
        png_path = Path(temp_dir) / "floorplan.png"
        user_data_dir = Path(temp_dir) / "chrome-profile"
        subprocess.run([
            _svg_png_renderer(),
            "--headless", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
            f"--window-size={resolution},{resolution}",
            f"--user-data-dir={user_data_dir}",
            f"--screenshot={png_path}",
            svg_path.resolve().as_uri(),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
        image = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (resolution, resolution):
        raise ValueError(f"Could not render furnished floorplan at {resolution}px: {svg_path}")
    return image


def _measurement_clutter_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    color_range = image.max(axis=2).astype(np.int16) - image.min(axis=2).astype(np.int16)
    clutter = ((gray < 90) | ((color_range > 45) & (gray < 248))).astype(np.uint8) * 255
    return cv2.dilate(clutter, np.ones((3, 3), dtype=np.uint8))


def _append_measurements(
    root: ET.Element,
    occupied: np.ndarray,
    contours: list[np.ndarray],
    resolution: int,
    pixel_size: float,
    clutter_mask: np.ndarray,
    scan_masks: dict[str, np.ndarray],
) -> list[dict]:
    group = ET.SubElement(root, f"{{{SVG_NS}}}g", {
        "id": "wall-segment-measurements",
        "fill": "none",
        "stroke": ORANGE,
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
    })
    measurements = []
    placed_label_boxes = []
    placed_lines = np.zeros((resolution, resolution), dtype=np.uint8)
    placed_label_mask = np.zeros_like(placed_lines)
    for contour in contours:
        points = cv2.approxPolyDP(contour, CONTOUR_APPROXIMATION_PX, True).reshape(-1, 2)
        if len(points) < 3:
            continue
        for index, point in enumerate(points):
            start = point.astype(float)
            end = points[(index + 1) % len(points)].astype(float)
            length_px = float(np.linalg.norm(end - start))
            length_m = length_px * pixel_size
            if length_m < MIN_WALL_SEGMENT_LENGTH_M:
                continue

            normal = _outside_normal(start, end, occupied)
            inside_normal = -normal
            owner_scan_id = _contour_scan_owner(start, end, inside_normal, scan_masks)
            between_scans = _faces_another_scan(
                start, end, normal, owner_scan_id, scan_masks,
                max(3, int(np.ceil(INTER_SCAN_ADJACENCY_DISTANCE_M / pixel_size))),
            )
            label = f"{length_m:.2f} m"
            label_width = len(label) * MEASURE_FONT_SIZE_PX * 0.66
            label_normal_extent = (
                label_width * 0.5 * abs(normal[0])
                + (MEASURE_FONT_SIZE_PX * 0.5 + 3) * abs(normal[1])
            )
            label_distance = max(
                DIMENSION_OFFSET_PX + TEXT_OFFSET_PX,
                DIMENSION_OFFSET_PX + label_normal_extent + 4,
            )
            tangent = (end - start) / length_px
            side_candidates = []
            sides = (("internal", inside_normal),) if between_scans else (
                ("external", normal), ("internal", inside_normal),
            )
            for side, side_normal in sides:
                dimension_start = start + side_normal * DIMENSION_OFFSET_PX
                dimension_end = end + side_normal * DIMENSION_OFFSET_PX
                extension_start = start + side_normal * (DIMENSION_OFFSET_PX + DIMENSION_EXTENSION_PX)
                extension_end = end + side_normal * (DIMENSION_OFFSET_PX + DIMENSION_EXTENSION_PX)
                candidate_boxes = list(placed_label_boxes)
                label_center = _choose_label_center(
                    (start + end) * 0.5 + side_normal * label_distance,
                    tangent, side_normal, length_px, label, resolution,
                    candidate_boxes, clutter_mask,
                )
                label_box = _label_box(
                    label_center, len(label) * MEASURE_FONT_SIZE_PX * 0.66,
                )
                lines = (
                    (start, extension_start),
                    (end, extension_end),
                    (dimension_start, dimension_end),
                )
                line_mask = np.zeros_like(placed_lines)
                for line_start, line_end in lines:
                    cv2.line(
                        line_mask,
                        tuple(np.rint(line_start).astype(int)),
                        tuple(np.rint(line_end).astype(int)),
                        255, 5, cv2.LINE_AA,
                    )
                label_mask = np.zeros_like(placed_label_mask)
                x0, y0 = np.maximum(np.floor(label_box[:2]).astype(int), 0)
                x1, y1 = np.minimum(np.ceil(label_box[2:]).astype(int), resolution)
                cv2.rectangle(label_mask, (x0, y0), (x1, y1), 255, -1)
                collision_pixels = sum(cv2.countNonZero(cv2.bitwise_and(a, b)) for a, b in (
                    (line_mask, placed_lines),
                    (line_mask, placed_label_mask),
                    (label_mask, placed_lines),
                    (label_mask, placed_label_mask),
                    (label_mask, line_mask),
                ))
                clutter_pixels = (
                    cv2.countNonZero(cv2.bitwise_and(line_mask, clutter_mask))
                    + 2 * cv2.countNonZero(cv2.bitwise_and(label_mask, clutter_mask))
                )
                outside_canvas = sum(
                    int(point[0] < 1 or point[0] >= resolution - 1
                        or point[1] < 1 or point[1] >= resolution - 1)
                    for point in (dimension_start, dimension_end, label_center)
                )
                side_candidates.append((
                    collision_pixels * 10000 + clutter_pixels + outside_canvas * 100000,
                    side, side_normal, dimension_start, dimension_end,
                    extension_start, extension_end, label_center, label_box,
                    lines, line_mask, label_mask,
                ))

            (
                _, side, normal, dimension_start, dimension_end,
                extension_start, extension_end, label_center, label_box,
                lines, line_mask, label_mask,
            ) = min(side_candidates, key=lambda item: item[0])
            placed_label_boxes.append(label_box)
            placed_lines = cv2.bitwise_or(placed_lines, line_mask)
            placed_label_mask = cv2.bitwise_or(placed_label_mask, label_mask)

            annotation = ET.SubElement(group, f"{{{SVG_NS}}}g", {
                "data-wall-segment-index": str(len(measurements)),
                "data-length-m": f"{length_m:.6f}",
                "data-placement-side": side,
                "data-between-scans": str(between_scans).lower(),
            })
            ET.SubElement(annotation, f"{{{SVG_NS}}}title").text = f"Wall segment: {label}"
            for a, b, width in (
                (start, extension_start, "1.5"),
                (end, extension_end, "1.5"),
                (dimension_start, dimension_end, "2.5"),
            ):
                ET.SubElement(annotation, f"{{{SVG_NS}}}line", {
                    "x1": f"{a[0]:.3f}", "y1": f"{a[1]:.3f}",
                    "x2": f"{b[0]:.3f}", "y2": f"{b[1]:.3f}",
                    "stroke": ORANGE, "stroke-width": width,
                })
            text = ET.SubElement(annotation, f"{{{SVG_NS}}}text", {
                "x": f"{label_center[0]:.3f}",
                "y": f"{label_center[1]:.3f}",
                "fill": ORANGE,
                "stroke": "white",
                "stroke-width": "3.5",
                "paint-order": "stroke fill",
                "font-family": "sans-serif",
                "font-size": str(MEASURE_FONT_SIZE_PX),
                "font-weight": "bold",
                "text-anchor": "middle",
                "dominant-baseline": "central",
            })
            text.text = label
            measurements.append({
                "index": len(measurements),
                "length_m": length_m,
                "label": label,
                "annotation_side": side,
                "between_scans": between_scans,
                "owner_scan_id": owner_scan_id,
                "span_px": [start.tolist(), end.tolist()],
                "dimension_span_px": [dimension_start.tolist(), dimension_end.tolist()],
                "extension_spans_px": [
                    [start.tolist(), extension_start.tolist()],
                    [end.tolist(), extension_end.tolist()],
                ],
                "label_center_px": label_center.tolist(),
            })

    existing_children = list(root)
    for child in existing_children:
        root.remove(child)
    root.set("height", f"{resolution + MEASURE_HEADER_HEIGHT_PX}px")
    root.set("viewBox", f"0 0 {resolution} {resolution + MEASURE_HEADER_HEIGHT_PX}")
    ET.SubElement(root, f"{{{SVG_NS}}}rect", {
        "x": "0", "y": "0", "width": str(resolution),
        "height": str(MEASURE_HEADER_HEIGHT_PX), "fill": "white",
    })
    content = ET.SubElement(root, f"{{{SVG_NS}}}g", {
        "transform": f"translate(0 {MEASURE_HEADER_HEIGHT_PX})",
    })
    for child in existing_children:
        content.append(child)
    warning = ET.SubElement(root, f"{{{SVG_NS}}}text", {
        "x": str(resolution * 0.5),
        "y": str(MEASURE_HEADER_HEIGHT_PX * 0.5),
        "fill": ORANGE,
        "stroke": "white",
        "stroke-width": "4",
        "paint-order": "stroke fill",
        "font-family": "sans-serif",
        "font-size": "14",
        "font-weight": "bold",
        "text-anchor": "middle",
        "dominant-baseline": "central",
        "data-measurement-warning": "true",
    })
    warning.text = WARNING_TEXT
    assert all(item["label"] == f"{item['length_m']:.2f} m" for item in measurements)
    return measurements


def _write_measured_projection_png(
    source_path: Path | np.ndarray,
    output_path: Path,
    measurements: list[dict],
    resolution: int,
) -> None:
    image = source_path if isinstance(source_path, np.ndarray) else cv2.imread(
        str(source_path), cv2.IMREAD_COLOR,
    )
    if image is None or image.shape[:2] != (resolution, resolution):
        raise ValueError(f"Expected a {resolution}x{resolution} floorplan image: {source_path}")
    image = cv2.copyMakeBorder(
        image, MEASURE_HEADER_HEIGHT_PX, 0, 0, 0,
        cv2.BORDER_CONSTANT, value=(255, 255, 255),
    )
    orange = (0, 113, 255)  # BGR equivalent of the SVG's bright orange.

    def shifted(point: list[float]) -> tuple[int, int]:
        x, y = np.rint(point).astype(int)
        return int(x), int(y + MEASURE_HEADER_HEIGHT_PX)

    for measurement in measurements:
        for start, end in measurement["extension_spans_px"]:
            cv2.line(image, shifted(start), shifted(end), orange, 2, cv2.LINE_AA)
        cv2.line(
            image,
            shifted(measurement["dimension_span_px"][0]),
            shifted(measurement["dimension_span_px"][1]),
            orange,
            3,
            cv2.LINE_AA,
        )
        label = measurement["label"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.42
        (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, 1)
        center_x, center_y = shifted(measurement["label_center_px"])
        origin = (center_x - text_width // 2, center_y + (text_height - baseline) // 2)
        cv2.putText(image, label, origin, font, scale, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(image, label, origin, font, scale, orange, 1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    warning_scale = 0.40
    (warning_width, warning_height), warning_baseline = cv2.getTextSize(
        WARNING_TEXT, font, warning_scale, 1,
    )
    warning_origin = (
        max(4, (resolution - warning_width) // 2),
        (MEASURE_HEADER_HEIGHT_PX + warning_height - warning_baseline) // 2,
    )
    cv2.putText(image, WARNING_TEXT, warning_origin, font, warning_scale,
                (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(image, WARNING_TEXT, warning_origin, font, warning_scale,
                orange, 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.stem + ".tmp.png")
    if not cv2.imwrite(str(temporary_path), image):
        raise OSError(f"Could not write measured floorplan image: {output_path}")
    temporary_path.replace(output_path)


def generate_floorplan_measurements(
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    output_dir: Path | None = None,
    overlay_manifest: dict | None = None,
) -> dict:
    artifacts_dir = Path(artifacts_dir).resolve()
    output_dir = Path(output_dir or artifacts_dir / "floorplan_measurements").resolve()
    mesh_manifest_path = artifacts_dir / "floor_level_meshes" / "floor_level_meshes_manifest.json"
    overlay_manifest_path = artifacts_dir / "floorplan_instance_overlays" / "floorplan_instance_overlays_manifest.json"
    if not mesh_manifest_path.is_file():
        raise FileNotFoundError(f"Required floorplan measurement input is missing: {mesh_manifest_path}")

    mesh_manifest = json.loads(mesh_manifest_path.read_text(encoding="utf-8"))
    if overlay_manifest is None:
        if not overlay_manifest_path.is_file():
            raise FileNotFoundError(f"Required floorplan measurement input is missing: {overlay_manifest_path}")
        overlay_manifest = json.loads(overlay_manifest_path.read_text(encoding="utf-8"))
    if overlay_manifest.get("furnished_views_measured"):
        existing_manifest = output_dir / "floorplan_measurements_manifest.json"
        if existing_manifest.is_file():
            result = json.loads(existing_manifest.read_text(encoding="utf-8"))
            expected_outputs = [
                Path(item[key])
                for item in result.get("floor_levels", [])
                for key in (
                    "measured_regularized_floorplan_png",
                    "measured_furnished_floorplan_svg",
                    "measured_furnished_floorplan_png",
                )
            ]
            if expected_outputs and all(path.is_file() for path in expected_outputs):
                return result
        raise RuntimeError(
            "Furnished floorplans are already finalized; rerun generate_floor_plans.py to rebuild them"
        )

    levels = {item["floor_level_name"]: item for item in mesh_manifest["levels"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    floor_results = []
    for floor in overlay_manifest["floor_levels"]:
        level = levels[floor["floor_level_name"]]
        source_svg = Path(floor["output_svg"])
        root = ET.parse(source_svg).getroot()
        resolution = int(floor["projection_resolution_px"][0])
        furnished_projection = _render_svg_png(source_svg, resolution)
        vertices, faces, _ = _read_mesh(Path(level["regularized_mesh"]))
        grid = _projection_grid(vertices, resolution)
        occupied, contours, scan_masks = _wall_contours(
            vertices, faces, level["scans"], resolution, grid,
        )
        measurements = _append_measurements(
            root, occupied, contours, resolution, float(grid["pixel_size"]),
            _measurement_clutter_mask(furnished_projection), scan_masks,
        )
        measured_furnished_svg = output_dir / f"floor_{floor['floor_level_name']}_regularized_instances_with_measures.svg"
        temporary_path = measured_furnished_svg.with_suffix(measured_furnished_svg.suffix + ".tmp")
        temporary_path.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
        temporary_path.replace(measured_furnished_svg)
        source_projection = Path(floor["source_projection"])
        measured_regularized_png = output_dir / f"floor_{floor['floor_level_name']}_regularized_topdown_with_measures.png"
        _write_measured_projection_png(
            source_projection, measured_regularized_png, measurements, resolution,
        )
        measured_furnished_png = output_dir / f"floor_{floor['floor_level_name']}_regularized_instances_with_measures.png"
        _write_measured_projection_png(
            furnished_projection, measured_furnished_png, measurements, resolution,
        )
        side_counts = {
            side: sum(item["annotation_side"] == side for item in measurements)
            for side in ("external", "internal")
        }
        floor_results.append({
            "floor_level_name": floor["floor_level_name"],
            "source_regularized_projection_png": str(source_projection),
            "measured_regularized_floorplan_png": str(measured_regularized_png),
            "source_furnished_floorplan_svg": str(source_svg),
            "measured_furnished_floorplan_svg": str(measured_furnished_svg),
            "measured_furnished_floorplan_png": str(measured_furnished_png),
            "measured_floorplan": str(measured_furnished_svg),
            "measured_view_count": 2,
            "measured_output_file_count": 3,
            "wall_segment_count": len(measurements),
            "between_scan_segment_count": sum(item["between_scans"] for item in measurements),
            "annotation_side_counts": side_counts,
            "measurements": measurements,
        })

    result = {
        "schema_version": 1,
        "source_floor_mesh_manifest": str(mesh_manifest_path),
        "source_floorplan_manifest": str(overlay_manifest_path),
        "output_dir": str(output_dir),
        "measurements_are_approximate": True,
        "minimum_wall_segment_length_m": MIN_WALL_SEGMENT_LENGTH_M,
        "measure_precision_decimal_places": 2,
        "warning_text": WARNING_TEXT,
        "measure_color": ORANGE,
        "furnished_png_renderer": "headless Chrome or Chromium",
        "measured_view_count": sum(item["measured_view_count"] for item in floor_results),
        "measured_output_file_count": sum(item["measured_output_file_count"] for item in floor_results),
        "between_scan_segment_count": sum(
            item["between_scan_segment_count"] for item in floor_results
        ),
        "wall_segment_annotation_count": sum(
            item["wall_segment_count"] * item["measured_view_count"] for item in floor_results
        ),
        "floor_levels": floor_results,
    }
    manifest_path = output_dir / "floorplan_measurements_manifest.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(manifest_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = generate_floorplan_measurements(args.artifacts_dir, args.output_dir)
    print(
        f"Available {result['measured_view_count']} measured views across "
        f"{len(result['floor_levels'])} floors with "
        f"{result['wall_segment_annotation_count']} wall-segment annotations."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
