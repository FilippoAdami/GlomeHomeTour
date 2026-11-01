#!/usr/bin/env python3
"""Export final floor polygons and dimensioned regularized contours."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np
from PIL import Image

from fit_floor_projection_contours_to_black import (
    MIN_FLOOR_POLYGON_SEGMENT_M,
    _axis_bbox,
    _remove_short_polygon_edges,
)
from project_floor_slices_onto_planes import _plane_frame
from floor_opening_constraints import CACHE_NAME, apply_saved_openings, update_opening_cache

SHARED_EDGE_TOLERANCE_M = 0.03
PARALLEL_TOLERANCE_DEG = 2.0
# shortcut: omit raster-sized edge fragments under 5 cm from dimensions; lower if tiny edges are meaningful.
MIN_MEASUREMENT_LENGTH_M = 0.05


def _image_points_to_world(points_px: np.ndarray, area: dict,
                           source_image_path: Path) -> np.ndarray:
    with Image.open(source_image_path) as source:
        pixels = np.asarray(source.convert("RGB"))
    x0, y0, x1, y1 = _axis_bbox(pixels)
    low, high = np.asarray(area["expanded_plane_bounds_uv_m"], dtype=float)
    if low.shape != (2,) or high.shape != (2,) or not np.isfinite([*low, *high]).all():
        raise ValueError("Projection manifest has invalid expanded plane bounds")
    padding = max(float(np.max(high - low)), 1.0) * 0.02
    padding = max(padding, 0.05)
    u_limits = (low[0] - padding, high[0] + padding)
    v_limits = (low[1] - padding, high[1] + padding)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Could not determine the projection axes bounds")

    uv = np.empty_like(points_px, dtype=float)
    uv[:, 0] = u_limits[0] + (points_px[:, 0] - x0) / (x1 - x0) * (u_limits[1] - u_limits[0])
    uv[:, 1] = v_limits[1] - (points_px[:, 1] - y0) / (y1 - y0) * (v_limits[1] - v_limits[0])
    plane, axis_u, axis_v = _plane_frame(area["plane_equation"], "floor-area")
    points_xyz = (uv[:, 0, None] * axis_u[None, :] + uv[:, 1, None] * axis_v[None, :]
                  - plane[3] * plane[:3][None, :])
    if not np.allclose(points_xyz @ plane[:3] + plane[3], 0.0, atol=1e-6):
        raise ValueError("Could not reconstruct floor polygon vertices on the fitted plane")
    return points_xyz


def _rings_from_segments(segments: list[dict], area: dict,
                         source_image: Path) -> list[np.ndarray]:
    groups: dict[int, list[dict]] = {}
    for segment in segments:
        groups.setdefault(int(segment.get("component_id", 1)), []).append(segment)
    rings = []
    for component_id, items in sorted(groups.items()):
        starts = np.asarray([item["start_px"] for item in items], dtype=float)
        ends = np.asarray([item["end_px"] for item in items], dtype=float)
        if (len(starts) < 3 or not np.isfinite(starts).all() or not np.isfinite(ends).all()
                or not np.allclose(ends, np.roll(starts, -1, axis=0), atol=1e-3)):
            raise ValueError(f"Final floor polygon ring {component_id} is not a closed contour")
        ring_xyz = _image_points_to_world(starts, area, source_image)
        ring_xz = ring_xyz[:, [0, 2]]
        simplified_rings_xz = _remove_short_polygon_edges(ring_xz, 1.0, 1.0)
        if not simplified_rings_xz:
            raise ValueError(
                f"Final floor polygon ring {component_id} cannot satisfy the "
                f"{MIN_FLOOR_POLYGON_SEGMENT_M:.2f} m minimum edge length"
            )
        plane = np.asarray(area.get("plane_equation"), dtype=float)
        if plane.shape != (4,) or not np.isfinite(plane).all() or abs(plane[1]) <= 1e-8:
            raise ValueError(f"Floor polygon ring {component_id} has no usable fitted plane")
        for simplified_xz in simplified_rings_xz:
            y = -(simplified_xz[:, 0] * plane[0] + simplified_xz[:, 1] * plane[2] + plane[3]) / plane[1]
            rings.append(np.column_stack((simplified_xz[:, 0], y, simplified_xz[:, 1])))
    return rings


def _dimension(start: np.ndarray, end: np.ndarray, identifier: str, source: str) -> dict:
    return {
        "id": identifier,
        "source": source,
        "length_m": float(np.linalg.norm(end - start)),
        "start_world_xyz_m": start.tolist(),
        "end_world_xyz_m": end.tolist(),
        "start_world_xz_m": start[[0, 2]].tolist(),
        "end_world_xz_m": end[[0, 2]].tolist(),
    }


def _unshared_floor_fragments(start: np.ndarray, end: np.ndarray,
                              walls: list[tuple[np.ndarray, np.ndarray]]) -> tuple[list[tuple[np.ndarray, np.ndarray]], bool]:
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length <= 1e-9:
        return [], False
    direction = vector / length
    parallel_cos = float(np.cos(np.deg2rad(PARALLEL_TOLERANCE_DEG)))
    shared_intervals = []
    for wall_start, wall_end in walls:
        wall_vector = wall_end - wall_start
        wall_length = float(np.linalg.norm(wall_vector))
        if wall_length <= 1e-9 or abs(float(np.dot(direction, wall_vector / wall_length))) < parallel_cos:
            continue
        offsets = np.asarray([wall_start - start, wall_end - start])
        if np.max(np.linalg.norm(offsets - np.outer(offsets @ direction, direction), axis=1)) > SHARED_EDGE_TOLERANCE_M:
            continue
        interval = sorted((float(offsets[0] @ direction), float(offsets[1] @ direction)))
        left, right = max(0.0, interval[0]), min(length, interval[1])
        if right - left > 1e-6:
            shared_intervals.append((left, right))
    if not shared_intervals:
        return [(start, end)], False

    shared_intervals.sort()
    merged = []
    for left, right in shared_intervals:
        if merged and left <= merged[-1][1] + 1e-6:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    fragments, cursor = [], 0.0
    for left, right in merged:
        if left - cursor > 1e-4:
            fragments.append((start + cursor * direction, start + left * direction))
        cursor = max(cursor, right)
    if length - cursor > 1e-4:
        fragments.append((start + cursor * direction, end))
    return fragments, True


def _area_record(level_id: int, area: dict, fitted: dict, level_dir: Path) -> dict:
    source_name = fitted.get("source_image")
    regularization = area.get("contour_regularization", {})
    wall_px = np.asarray(regularization.get("points_px", []), dtype=float)
    if wall_px.ndim != 2 or wall_px.shape[1] != 2 or len(wall_px) < 3:
        raise ValueError(f"Floor {level_id} area {area.get('area_id')} has no regularized wall contour")
    if not isinstance(source_name, str) or not (level_dir / source_name).is_file():
        raise ValueError(f"Floor {level_id} area {area.get('area_id')} has no source projection")
    source_image = level_dir / source_name
    wall_xyz = _image_points_to_world(wall_px, area, source_image)
    wall_edges = [(wall_xyz[i], wall_xyz[(i + 1) % len(wall_xyz)]) for i in range(len(wall_xyz))]
    polygon_segments = fitted.get("segment_dimensions", {}).get("black_part_segments")
    if not isinstance(polygon_segments, list) or not polygon_segments:
        raise ValueError(f"Floor {level_id} area {area.get('area_id')} has no final floor polygon")
    rings_xyz = _rings_from_segments(polygon_segments, area, source_image)

    measurements = [
        _dimension(start, end, f"W{index + 1:02d}", "regularized_wall_contour")
        for index, (start, end) in enumerate(wall_edges)
    ]
    omitted_shared = 0
    for ring_index, ring in enumerate(rings_xyz, start=1):
        for edge_index, (start, end) in enumerate(zip(ring, np.roll(ring, -1, axis=0)), start=1):
            fragments, shared = _unshared_floor_fragments(start, end, wall_edges)
            omitted_shared += int(shared)
            for fragment_index, (a, b) in enumerate(fragments, start=1):
                if np.linalg.norm(b - a) < MIN_MEASUREMENT_LENGTH_M:
                    continue
                measurements.append(_dimension(
                    a, b, f"P{ring_index}.{edge_index:02d}{chr(96 + fragment_index) if len(fragments) > 1 else ''}",
                    "floor_polygon_boundary",
                ))

    return {
        "area_id": int(area["area_id"]),
        "_floor_plane_equation_world": area["plane_equation"],
        "minimum_floor_polygon_segment_length_m": MIN_FLOOR_POLYGON_SEGMENT_M,
        "regularized_wall_contour_world_xz_m": wall_xyz[:, [0, 2]].tolist(),
        "floor_polygon_rings_world_xz_m": [ring[:, [0, 2]].tolist() for ring in rings_xyz],
        "shared_floor_segments_measured_once": omitted_shared,
        "dimensions": measurements,
    }


def _refresh_floor_boundary_dimensions(area: dict) -> None:
    equation = np.asarray(area.pop("_floor_plane_equation_world"), dtype=float)
    if equation.shape != (4,) or not np.isfinite(equation).all():
        raise ValueError("Could not refresh dimensions for a floor polygon without a valid plane")
    norm = float(np.linalg.norm(equation[:3]))
    if norm <= 1e-10:
        raise ValueError("Could not refresh dimensions for a floor polygon without a valid plane")
    normal, offset = equation[:3] / norm, float(equation[3] / norm)
    if abs(normal[1]) <= 1e-8:
        raise ValueError("Could not refresh floor dimensions on a vertical plane")

    def to_world(xz: np.ndarray) -> np.ndarray:
        y = -(xz[:, 0] * normal[0] + xz[:, 1] * normal[2] + offset) / normal[1]
        return np.column_stack((xz[:, 0], y, xz[:, 1]))

    wall_edges = to_world(np.asarray(area["regularized_wall_contour_world_xz_m"], dtype=float))
    walls = [(wall_edges[index], wall_edges[(index + 1) % len(wall_edges)])
             for index in range(len(wall_edges))]
    dimensions = [item for item in area["dimensions"] if item["source"] != "floor_polygon_boundary"]
    shared_count = 0
    for ring_index, ring_xz in enumerate(area["floor_polygon_rings_world_xz_m"], start=1):
        ring = to_world(np.asarray(ring_xz, dtype=float))
        for edge_index, (start, end) in enumerate(zip(ring, np.roll(ring, -1, axis=0)), start=1):
            fragments, shared = _unshared_floor_fragments(start, end, walls)
            shared_count += int(shared)
            for fragment_index, (a, b) in enumerate(fragments, start=1):
                if np.linalg.norm(b - a) < MIN_MEASUREMENT_LENGTH_M:
                    continue
                suffix = chr(96 + fragment_index) if len(fragments) > 1 else ""
                dimensions.append(_dimension(a, b, f"P{ring_index:02d}.{edge_index:02d}{suffix}",
                                             "floor_polygon_boundary"))
    area["dimensions"] = dimensions
    area["shared_floor_segments_measured_once"] = shared_count


def _ring_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return abs(float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))))


def _inside(point: np.ndarray, ring: np.ndarray) -> bool:
    x, y = point
    inside = False
    for a, b in zip(ring, np.roll(ring, -1, axis=0)):
        if (a[1] > y) != (b[1] > y):
            crossing = (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]
            if x < crossing:
                inside = not inside
    return inside


def _collect_floor_records(level_dir: Path) -> list[dict]:
    projection = json.loads((level_dir / "floor_level_collapsed_first_refinement_holes_manifest.json").read_text())
    fitted = json.loads((level_dir / "floor_level_contour_fitted_manifest.json").read_text())
    fitted_by_id = {
        (level["level"], area["area_id"]): area
        for level in fitted.get("levels", []) for area in level.get("areas", [])
    }
    records = []
    for level in sorted(projection.get("levels", []), key=lambda item: item["level"]):
        areas = []
        for area in sorted(level.get("areas", []), key=lambda item: item["area_id"]):
            fitted_area = fitted_by_id.get((level["level"], area["area_id"]))
            if fitted_area is None:
                raise ValueError(f"No filled floor polygon for floor {level['level']}, area {area['area_id']}")
            areas.append(_area_record(level["level"], area, fitted_area, level_dir))
        records.append({"level": int(level["level"]), "areas": areas})
    if not records:
        raise ValueError("No final floor polygons were found")
    return records


def _draw_dimension(ax, record: dict, color: tuple[float, float, float],
                   offset: tuple[float, float]):
    start = np.asarray(record["start_world_xz_m"], dtype=float)
    end = np.asarray(record["end_world_xz_m"], dtype=float)
    middle = (start + end) / 2
    return ax.annotate(
        f"{record['_label']} {record['length_m']:.2f}m", xy=middle,
        xytext=offset, textcoords="offset points", ha="center", va="center", fontsize=6.3,
        color="#111111", bbox={"boxstyle": "round,pad=0.18", "facecolor": "white",
                                 "edgecolor": color, "linewidth": 0.6, "alpha": 0.94},
        arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.55},
    )


def _place_dimension_labels(ax, figure, entries: list[tuple[dict, tuple[float, float, float]]]) -> None:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    occupied = []
    figure_box = figure.bbox
    directions = (
        (0, 1), (0, -1), (1, 0), (-1, 0),
        (0.707, 0.707), (-0.707, 0.707), (0.707, -0.707), (-0.707, -0.707),
    )
    for record, color in entries:
        start = np.asarray(record["start_world_xz_m"], dtype=float)
        end = np.asarray(record["end_world_xz_m"], dtype=float)
        middle = (start + end) / 2
        tangent = ax.transData.transform(end) - ax.transData.transform(start)
        tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
        normal = np.array([-tangent[1], tangent[0]])
        preferred = [normal, -normal, tangent, -tangent]
        candidate_vectors = preferred + [np.asarray(direction, dtype=float) for direction in directions[4:]]
        best = None
        for radius in (12, 24, 38, 54):
            for direction in candidate_vectors:
                offset = tuple((direction * radius).tolist())
                annotation = _draw_dimension(ax, record, color, offset)
                figure.canvas.draw()
                box = annotation.get_window_extent(renderer).expanded(1.12, 1.18)
                area = max(box.width * box.height, 1.0)
                overlap = sum(
                    max(0.0, min(box.x1, old.x1) - max(box.x0, old.x0))
                    * max(0.0, min(box.y1, old.y1) - max(box.y0, old.y0))
                    for old in occupied
                ) / area
                outside = (max(0.0, figure_box.x0 - box.x0) + max(0.0, box.x1 - figure_box.x1)
                           + max(0.0, figure_box.y0 - box.y0) + max(0.0, box.y1 - figure_box.y1))
                score = overlap * 1000 + outside * 100 + radius * 0.001
                annotation.remove()
                if best is None or score < best[0]:
                    best = (score, offset)
                if overlap == 0 and outside == 0:
                    break
            if best is not None and best[0] < 0.1:
                break
        final = _draw_dimension(ax, record, color, best[1])
        figure.canvas.draw()
        occupied.append(final.get_window_extent(renderer).expanded(1.06, 1.1))


def run(instances_dir: Path) -> Path:
    level_dir = instances_dir / "intermediate_floor_levels"
    records = _collect_floor_records(level_dir)
    cache_path = instances_dir.parent / CACHE_NAME
    changed_areas = apply_saved_openings(records, cache_path)
    for level in records:
        for area in level["areas"]:
            if (level["level"], area["area_id"]) in changed_areas:
                _refresh_floor_boundary_dimensions(area)
            else:
                area.pop("_floor_plane_equation_world", None)
    count = sum(len(level["areas"]) for level in records)
    cmap = plt.get_cmap("tab10" if count <= 10 else "tab20")
    colors = [tuple(float(c) for c in cmap(i % cmap.N)[:3]) for i in range(count)]
    all_points, legend, color_index, displayed_edges, display_labels = [], [], 0, [], []
    figure, ax = plt.subplots(figsize=(13, 11), dpi=170)
    for level in records:
        level_output = {
            "level": level["level"],
            "coordinate_system": "Phase 04 world frame; geometry is stored as X/Z in metres; dimensions are measured on the fitted floor plane",
            "areas": level["areas"],
        }
        output_json = level_dir / f"floor_level_{level['level']:03d}_polygons.json"
        output_json.write_text(json.dumps(level_output, indent=2) + "\n", encoding="utf-8")
        for area in level["areas"]:
            color = colors[color_index]
            color_index += 1
            label = f"Floor {level['level']:03d}, area {area['area_id']:03d}"
            legend.append(Patch(facecolor=(*color, 0.30), edgecolor=color, label=label))
            rings = [np.asarray(ring, dtype=float) for ring in area["floor_polygon_rings_world_xz_m"]]
            ring_areas = [_ring_area(ring) for ring in rings]
            for index, ring in enumerate(rings):
                sample = ring[0]
                depth = sum(
                    1 for other_index, other in enumerate(rings)
                    if ring_areas[other_index] > ring_areas[index] and _inside(sample, other)
                )
                is_hole = depth % 2 == 1
                ax.add_patch(Polygon(
                    ring, closed=True, facecolor="#f7f7f5" if is_hole else (*color, 0.30),
                    edgecolor=color, linewidth=1.8, joinstyle="round", zorder=2 + depth,
                ))
                all_points.append(ring)

            walls = np.asarray(area["regularized_wall_contour_world_xz_m"], dtype=float)
            ax.plot(*np.vstack((walls, walls[0])).T, color="#171717", linestyle=(0, (4, 2)),
                    linewidth=1.5, zorder=8)
            all_points.append(walls)
            for edge_index, item in enumerate(area["dimensions"]):
                start = np.asarray(item["start_world_xz_m"], dtype=float)
                end = np.asarray(item["end_world_xz_m"], dtype=float)
                ax.plot([start[0], end[0]], [start[1], end[1]], color=color,
                        linewidth=1.0 if item["source"] == "floor_polygon_boundary" else 1.6,
                        zorder=5 if item["source"] == "floor_polygon_boundary" else 9)
                fragments, _ = _unshared_floor_fragments(start, end, displayed_edges)
                full_length_xz = max(float(np.linalg.norm(end - start)), 1e-9)
                for fragment_index, (a, b) in enumerate(fragments):
                    fragment_length_m = item["length_m"] * float(np.linalg.norm(b - a)) / full_length_xz
                    if fragment_length_m < MIN_MEASUREMENT_LENGTH_M:
                        continue
                    displayed = {
                        **item,
                        "start_world_xz_m": a.tolist(),
                        "end_world_xz_m": b.tolist(),
                        "length_m": fragment_length_m,
                        "_label": f"F{level['level']:02d} {item['id']}"
                                 f"{chr(97 + fragment_index) if len(fragments) > 1 else ''}",
                    }
                    display_labels.append((displayed, color))
                    displayed_edges.append((a, b))

    all_points_array = np.concatenate(all_points)
    low, high = all_points_array.min(axis=0), all_points_array.max(axis=0)
    padding = max(float(np.max(high - low)) * 0.08, 0.10)
    ax.set_xlim(low[0] - padding, high[0] + padding)
    ax.set_ylim(low[1] - padding, high[1] + padding)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Z (m)")
    ax.set_title("Final floor polygons and regularized wall contours")
    ax.grid(True, color="#cccccc", linewidth=0.5, alpha=0.45)
    legend.extend([
        Patch(facecolor="none", edgecolor="#555555", label="Final floor polygon boundary"),
        Line2D([0], [0], color="#171717", linestyle=(0, (4, 2)), label="Regularized wall contour"),
    ])
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=min(4, len(legend)), fontsize=8, frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    _place_dimension_labels(ax, figure, display_labels)
    output_image = level_dir / "floor_level_all_polygons_topdown.png"
    figure.savefig(output_image, dpi=170, bbox_inches="tight")
    plt.close(figure)

    current_jsons = {f"floor_level_{item['level']:03d}_polygons.json" for item in records}
    for path in level_dir.glob("floor_level_*_polygons.json"):
        if path.name not in current_jsons:
            path.unlink()
    try:
        update_opening_cache(records, cache_path)
    except OSError as exc:
        print(f"Warning: could not update floor opening cache {cache_path}: {exc}", file=sys.stderr)
    return output_image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path, required=True,
                        help="Stage 06 output directory containing floor projection manifests")
    args = parser.parse_args(argv)
    try:
        image_path = run(args.instances_dir)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Error: Could not export final floor polygons: {exc}", file=sys.stderr)
        return 1
    print(f"Floor plan overlay -> {image_path}")
    for path in sorted((args.instances_dir / "intermediate_floor_levels").glob("floor_level_*_polygons.json")):
        print(f"Floor polygon -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
