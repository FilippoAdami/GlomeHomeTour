#!/usr/bin/env python3
"""Generate transparent approximate wall-segment measurements for each floor."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from generate_floor_level_meshes import _read_mesh
from generate_floor_level_projections import (
    SMALL_FOOTPRINT_HOLE_MAX_AREA_M2,
    _fill_small_footprint_holes,
    _projection_grid,
)
from generate_floorplan_instance_overlays import (
    DEFAULT_ARTIFACTS_DIR,
    _scan_footprint_masks,
)


ORANGE = "#ff7100"
MIN_WALL_SEGMENT_LENGTH_M = 0.10
CONTOUR_APPROXIMATION_PX = 1.5
DIMENSION_OFFSET_PX = 16.0
DIMENSION_EXTENSION_PX = 5.0
TEXT_OFFSET_PX = 18.0
MEASURE_FONT_SIZE_PX = 22
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


def _label_box(center: np.ndarray, width: float, font_size: float) -> tuple[float, float, float, float]:
    return (
        float(center[0] - width * 0.5 - 3), float(center[1] - font_size * 0.5 - 3),
        float(center[0] + width * 0.5 + 3), float(center[1] + font_size * 0.5 + 3),
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
    font_size = min(MEASURE_FONT_SIZE_PX, resolution / 5)
    text_width = len(label) * font_size * 0.66
    max_tangent_shift = max(36.0, min(96.0, length_px * 0.4))
    offsets = [0.0]
    for step in np.arange(12.0, max_tangent_shift + 1.0, 12.0):
        offsets.extend((float(step), float(-step)))
    candidates = []
    for normal_shift in (0.0, 12.0, 24.0, 36.0):
        for tangent_shift in offsets:
            center = base + tangent * tangent_shift + normal * normal_shift
            box = _label_box(center, text_width, font_size)
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
    placed_boxes.append(_label_box(center, text_width, font_size))
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


def _wall_segment_measurements(
    occupied: np.ndarray,
    contours: list[np.ndarray],
    resolution: int,
    pixel_size: float,
    clutter_mask: np.ndarray,
    scan_masks: dict[str, np.ndarray],
) -> list[dict]:
    measurements = []
    font_size = min(MEASURE_FONT_SIZE_PX, resolution / 5)
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
            label_width = len(label) * font_size * 0.66
            label_normal_extent = (
                label_width * 0.5 * abs(normal[0])
                + (font_size * 0.5 + 3) * abs(normal[1])
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
                    label_center, len(label) * font_size * 0.66, font_size,
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

    assert all(item["label"] == f"{item['length_m']:.2f} m" for item in measurements)
    return measurements


def _write_measurement_overlay_png(
    output_path: Path,
    measurements: list[dict],
    resolution: int,
) -> None:
    lines = np.zeros((resolution, resolution), dtype=np.uint8)
    labels = np.zeros_like(lines)

    def pixel(point: list[float]) -> tuple[int, int]:
        x, y = np.rint(point).astype(int)
        return int(x), int(y)

    for measurement in measurements:
        for start, end in measurement["extension_spans_px"]:
            cv2.line(lines, pixel(start), pixel(end), 255, 2, cv2.LINE_AA)
        cv2.line(
            lines,
            pixel(measurement["dimension_span_px"][0]),
            pixel(measurement["dimension_span_px"][1]),
            255,
            3,
            cv2.LINE_AA,
        )
        label = measurement["label"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.75 * min(1.0, resolution / 110)
        thickness = 2 if resolution >= 110 else 1
        (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
        center_x, center_y = pixel(measurement["label_center_px"])
        origin = (center_x - text_width // 2, center_y + (text_height - baseline) // 2)
        cv2.putText(labels, label, origin, font, scale, 255, thickness, cv2.LINE_AA)

    image = Image.new("RGBA", (resolution, resolution))
    outlines = cv2.dilate(labels, np.ones((5, 5), dtype=np.uint8))
    for mask, color in ((lines, (255, 113, 0)), (outlines, (255, 255, 255)),
                        (labels, (255, 113, 0))):
        layer = Image.new("RGBA", image.size, color)
        layer.putalpha(Image.fromarray(mask))
        image = Image.alpha_composite(image, layer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.stem + ".tmp.png")
    image.save(temporary_path)
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
        raise RuntimeError("Furnished floorplans are measured; rerun generate_floor_plans.py to rebuild them")

    levels = {item["floor_level_name"]: item for item in mesh_manifest["levels"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    floor_results = []
    for floor in overlay_manifest["floor_levels"]:
        level = levels[floor["floor_level_name"]]
        source_svg = Path(floor["output_svg"])
        resolution = int(floor["projection_resolution_px"][0])
        if floor["projection_resolution_px"] != [resolution, resolution]:
            raise ValueError("Floorplan measurement projection must be square")
        furnished_projection = _render_svg_png(source_svg, resolution)
        vertices, faces, _ = _read_mesh(Path(level["regularized_mesh"]))
        grid = _projection_grid(vertices, resolution)
        occupied, contours, scan_masks = _wall_contours(
            vertices, faces, level["scans"], resolution, grid,
        )
        measurements = _wall_segment_measurements(
            occupied, contours, resolution, float(grid["pixel_size"]),
            _measurement_clutter_mask(furnished_projection), scan_masks,
        )
        source_projection = Path(floor["source_projection"])
        overlay_png = output_dir / f"floor_{floor['floor_level_name']}_measurements.png"
        _write_measurement_overlay_png(overlay_png, measurements, resolution)
        side_counts = {
            side: sum(item["annotation_side"] == side for item in measurements)
            for side in ("external", "internal")
        }
        floor_results.append({
            "floor_level_name": floor["floor_level_name"],
            "source_regularized_projection_png": str(source_projection),
            "source_furnished_floorplan_svg": str(source_svg),
            "overlay_png": str(overlay_png),
            "projection_grid": {
                "x_min": float(grid["x_min"]), "z_max": float(grid["z_max"]),
                "size": float(grid["pixel_size"]), "width": resolution, "height": resolution,
            },
            "wall_segment_count": len(measurements),
            "between_scan_segment_count": sum(item["between_scans"] for item in measurements),
            "annotation_side_counts": side_counts,
            "measurements": measurements,
        })

    result = {
        "schema_version": 2,
        "source_floor_mesh_manifest": str(mesh_manifest_path),
        "source_floorplan_manifest": str(overlay_manifest_path),
        "output_dir": str(output_dir),
        "measurements_are_approximate": True,
        "minimum_wall_segment_length_m": MIN_WALL_SEGMENT_LENGTH_M,
        "measure_precision_decimal_places": 2,
        "warning_text": WARNING_TEXT,
        "measure_color": ORANGE,
        "furnished_png_renderer": "headless Chrome or Chromium",
        "measurement_overlay_count": len(floor_results),
        "between_scan_segment_count": sum(
            item["between_scan_segment_count"] for item in floor_results
        ),
        "wall_segment_annotation_count": sum(item["wall_segment_count"] for item in floor_results),
        "floor_levels": floor_results,
    }
    manifest_path = output_dir / "floorplan_measurements_manifest.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(manifest_path)
    for pattern in ("floor_*_regularized_topdown_with_measures.*",
                    "floor_*_regularized_instances_with_measures.*"):
        for old in output_dir.glob(pattern):
            if old.suffix in {".png", ".pdf", ".svg"}:
                old.unlink()
    current_overlays = {Path(item["overlay_png"]) for item in floor_results}
    for old in output_dir.glob("floor_*_measurements.png"):
        if old not in current_overlays:
            old.unlink()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = generate_floorplan_measurements(args.artifacts_dir, args.output_dir)
    print(
        f"Available {result['measurement_overlay_count']} measurement overlays with "
        f"{result['wall_segment_annotation_count']} wall-segment annotations."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
