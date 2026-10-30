#!/usr/bin/env python3
"""Fit finite wall panels to the accepted, aligned floor perimeters."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from shapely.geometry import LineString
from shapely.ops import unary_union

from fit_wall_planes import _plane_quad_geometry, _write

MAX_LINE_OFFSET_M = 0.15
MIN_EDGE_OVERLAP_M = 0.15
MIN_EDGE_OVERLAP_FRACTION = 0.5
GEOMETRY_EPS_M = 1e-6
CHECK_TOLERANCE_M = 0.001


def _regions(plane: dict, span_key: str) -> list[dict]:
    if plane.get("trim_regions"):
        return [
            {"span_m": region["span_m"], "bounds_y": region["bounds_y"]}
            for region in plane["trim_regions"]
        ]
    return [{"span_m": plane[span_key], "bounds_y": plane["bounds_y"]}]


def _edge_match(plane: dict, region: dict, floor: dict) -> dict | None:
    axis = plane["axis"]
    span = np.asarray(region.get("span_m", region["source_span_m"]), dtype=float)
    offset = float(plane["offset_m"])
    points = np.asarray(floor["aligned_perimeter_xz_m"], dtype=float)
    candidates = []
    for edge_index, (first, second) in enumerate(zip(points, np.roll(points, -1, axis=0))):
        delta = second - first
        if axis == "Dir A":
            if abs(delta[0]) > GEOMETRY_EPS_M:
                continue
            edge_offset = float(first[0])
            edge_span = np.sort([first[1], second[1]])
        else:
            if abs(delta[1]) > GEOMETRY_EPS_M:
                continue
            edge_offset = float(first[1])
            edge_span = np.sort([first[0], second[0]])
        offset_error = abs(edge_offset - offset)
        overlap = max(0.0, min(span[1], edge_span[1]) - max(span[0], edge_span[0]))
        shorter = min(span[1] - span[0], edge_span[1] - edge_span[0])
        if (offset_error > MAX_LINE_OFFSET_M or overlap < MIN_EDGE_OVERLAP_M
                or overlap < MIN_EDGE_OVERLAP_FRACTION * shorter):
            continue
        candidates.append((offset_error, -overlap, edge_index, edge_offset, edge_span, overlap / shorter))
    if not candidates:
        return None
    offset_error, neg_overlap, edge_index, edge_offset, edge_span, overlap_fraction = min(candidates)
    return {
        "edge_index": int(edge_index),
        "offset_m": float(edge_offset),
        "span_m": edge_span.tolist(),
        "offset_error_m": float(offset_error),
        "overlap_m": float(-neg_overlap),
        "overlap_fraction_of_shorter": float(overlap_fraction),
    }


def _fitted_group_spans(group: list[dict], edge_span: list[float]) -> dict[int, list[float]]:
    """Partition a supported perimeter edge across its adjacent source panels."""
    edge_lo, edge_hi = map(float, edge_span)
    intervals = []
    for item in group:
        source_lo, source_hi = map(float, item["source_span_m"])
        lo, hi = max(edge_lo, source_lo), min(edge_hi, source_hi)
        if hi - lo >= MIN_EDGE_OVERLAP_M:
            intervals.append((lo, hi, item["region_id"]))
    intervals.sort()
    if not intervals:
        return {}

    result = {region_id: [lo, hi] for lo, hi, region_id in intervals}
    if len(intervals) == 1:
        lo, hi, region_id = intervals[0]
        source_lo, source_hi = map(float, next(x["source_span_m"] for x in group if x["region_id"] == region_id))
        if abs(source_lo - edge_lo) <= MAX_LINE_OFFSET_M and abs(source_hi - edge_hi) <= MAX_LINE_OFFSET_M:
            result[region_id] = [edge_lo, edge_hi]
        else:
            if abs(source_lo - edge_lo) <= MAX_LINE_OFFSET_M:
                result[region_id][0] = edge_lo
            if abs(source_hi - edge_hi) <= MAX_LINE_OFFSET_M:
                result[region_id][1] = edge_hi
        return result

    if abs(intervals[0][0] - edge_lo) <= MAX_LINE_OFFSET_M:
        result[intervals[0][2]][0] = edge_lo
    if abs(intervals[-1][1] - edge_hi) <= MAX_LINE_OFFSET_M:
        result[intervals[-1][2]][1] = edge_hi
    for left, right in zip(intervals, intervals[1:]):
        left_lo, left_hi, left_id = left
        right_lo, right_hi, right_id = right
        if abs(left_hi - right_lo) <= MAX_LINE_OFFSET_M:
            junction = float(np.clip((left_hi + right_lo) / 2.0, edge_lo, edge_hi))
            result[left_id][1] = junction
            result[right_id][0] = junction
    return result


def _line_at_floor(plane: dict, floor_y: float) -> LineString:
    start, end = plane["span_w_m"] if plane["axis"] == "Dir A" else plane["span_u_m"]
    if plane["axis"] == "Dir A":
        return LineString([(plane["offset_m"], start), (plane["offset_m"], end)])
    return LineString([(start, plane["offset_m"]), (end, plane["offset_m"])])


def _plot_fit(path: Path, floor: dict, actual_lines: list[LineString], audit: dict) -> None:
    points = np.asarray(floor["aligned_perimeter_xz_m"], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 8))
    closed = np.vstack([points, points[0]])
    ax.plot(closed[:, 0], closed[:, 1], color="#d32f2f", lw=2.5, label="Final floor perimeter")
    for index, line in enumerate(actual_lines):
        coords = np.asarray(line.coords)
        ax.plot(coords[:, 0], coords[:, 1], color="#1976d2", lw=1.3, alpha=0.9,
                label="Fitted wall planes" if index == 0 else None)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Aligned X (m)")
    ax.set_ylabel("Aligned Z (m)")
    ax.set_title(
        f"Floor {int(floor['floor_index'])}: perimeter / fitted wall planes\n"
        f"uncovered {audit['uncovered_perimeter_m']:.3f} m, outside {audit['wall_outside_perimeter_m']:.3f} m"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fit_wall_planes_to_floor_perimeters(
    floor_fit_manifest: Path,
    wall_manifest_path: Path,
    output_dir: Path,
) -> dict:
    """Create a non-destructive wall-plane candidate using final aligned floor outlines."""
    floor_data = json.loads(floor_fit_manifest.read_text())
    source = json.loads(wall_manifest_path.read_text())
    floors = sorted(floor_data["final_aligned_floor_plans"], key=lambda item: item["floor_index"])
    basis = source["orthogonal_basis"]
    basis_a = np.asarray(basis["dir_A_east_west"], dtype=float)
    basis_b = np.asarray(basis["dir_B_north_south"], dtype=float)
    basis_a /= np.linalg.norm(basis_a)
    basis_b /= np.linalg.norm(basis_b)
    output_dir.mkdir(parents=True, exist_ok=True)
    individual_dir = output_dir / "wall_enclosing_planes_individual"
    connector_dir = output_dir / "wall_closure_connectors"
    individual_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(connector_dir, ignore_errors=True)
    for path in individual_dir.glob("wall_plane_*.ply"):
        path.unlink()

    regions = []
    region_id = 0
    for plane_index, plane in enumerate(source["wall_planes"]):
        span_key = "span_w_m" if plane["axis"] == "Dir A" else "span_u_m"
        for region_index, region in enumerate(_regions(plane, span_key)):
            regions.append({
                "region_id": region_id,
                "plane_index": plane_index,
                "region_index": region_index,
                "plane": plane,
                "span_key": span_key,
                "source_span_m": list(map(float, region["span_m"])),
                "bounds_y": list(map(float, region["bounds_y"])),
                "matches": {},
            })
            region_id += 1

    groups: dict[tuple[int, int], list[dict]] = {}
    for region in regions:
        plane = region["plane"]
        y0, y1 = region["bounds_y"]
        for floor in floors:
            floor_y = float(floor["floor_y_m"])
            if not y0 - GEOMETRY_EPS_M <= floor_y <= y1 + GEOMETRY_EPS_M:
                continue
            match = _edge_match(plane, region, floor)
            if match is None:
                continue
            entry = {**match, "region_id": region["region_id"],
                     "source_span_m": region["source_span_m"]}
            groups.setdefault((int(floor["floor_index"]), match["edge_index"]), []).append(entry)
            region["matches"][int(floor["floor_index"])] = match

    fit_records = {}
    edge_audit = []
    floor_by_id = {int(floor["floor_index"]): floor for floor in floors}
    for (floor_index, edge_index), group in groups.items():
        floor = floor_by_id[floor_index]
        points = np.asarray(floor["aligned_perimeter_xz_m"], dtype=float)
        first, second = points[edge_index], points[(edge_index + 1) % len(points)]
        # The perimeter's axis determines which coordinate is the segment span.
        if group and source["wall_planes"][regions[group[0]["region_id"]]["plane_index"]]["axis"] == "Dir A":
            edge_span = np.sort([first[1], second[1]])
        else:
            edge_span = np.sort([first[0], second[0]])
        group_spans = _fitted_group_spans(group, edge_span.tolist())
        covered = sum(max(0.0, hi - lo) for lo, hi in group_spans.values())
        target_length = float(edge_span[1] - edge_span[0])
        edge_audit.append({
            "floor_index": floor_index,
            "edge_index": edge_index,
            "target_offset_m": group[0]["offset_m"],
            "target_span_m": edge_span.tolist(),
            "target_length_m": target_length,
            "matched_panel_region_count": len(group),
            "supported_length_m": min(covered, target_length),
            "supported_fraction": min(covered, target_length) / target_length if target_length else 0.0,
        })
        for entry in group:
            region_id = entry["region_id"]
            span = group_spans.get(region_id)
            if span and span[1] - span[0] >= MIN_EDGE_OVERLAP_M:
                fit_records.setdefault(region_id, {})[floor_index] = {
                    "axis": regions[region_id]["plane"]["axis"],
                    "offset_m": entry["offset_m"],
                    "span_m": span,
                    "edge_index": edge_index,
                    "offset_error_m": entry["offset_error_m"],
                    "overlap_m": entry["overlap_m"],
                    "overlap_fraction_of_shorter": entry["overlap_fraction_of_shorter"],
                }

    fitted_planes = []
    source_colors: dict[int, np.ndarray] = {}
    all_vertices, all_triangles, all_colors = [], [], []
    vertex_offset = 0
    for region in regions:
        region_id = region["region_id"]
        plane = region["plane"]
        matches = fit_records.get(region_id, {})
        if not matches:
            outputs = [{
                "offset_m": float(plane["offset_m"]),
                "span_m": region["source_span_m"],
                "bounds_y": region["bounds_y"],
                "floor_indices": [],
            }]
        else:
            ordered = sorted(matches.items())
            geoms_same = all(
                first[1]["axis"] == other[1]["axis"]
                and abs(first[1]["offset_m"] - other[1]["offset_m"]) <= GEOMETRY_EPS_M
                and np.max(np.abs(np.asarray(first[1]["span_m"]) - other[1]["span_m"])) <= GEOMETRY_EPS_M
                for first, other in zip(ordered, ordered[1:])
            )
            if len(ordered) == 1 or geoms_same:
                floor_ids = [item[0] for item in ordered]
                chosen = ordered[-1][1]
                outputs = [{
                    "offset_m": chosen["offset_m"],
                    "span_m": chosen["span_m"],
                    "bounds_y": region["bounds_y"],
                    "floor_indices": floor_ids,
                }]
            else:
                outputs = []
                for i, (floor_index, fitted) in enumerate(ordered):
                    low_y = region["bounds_y"][0] if i == 0 else max(
                        region["bounds_y"][0], float(floor_by_id[floor_index]["floor_y_m"])
                    )
                    high_y = region["bounds_y"][1] if i == len(ordered) - 1 else min(
                        region["bounds_y"][1], float(floor_by_id[ordered[i + 1][0]]["floor_y_m"])
                    )
                    if high_y - low_y > GEOMETRY_EPS_M:
                        outputs.append({
                            "offset_m": fitted["offset_m"],
                            "span_m": fitted["span_m"],
                            "bounds_y": [low_y, high_y],
                            "floor_indices": [floor_index],
                        })
        for output_index, output in enumerate(outputs):
            fitted = copy.deepcopy(plane)
            fitted.pop("trim_regions", None)
            fitted["id"] = len(fitted_planes) + 1
            original_name = plane.get("name", f"wall_plane_{region['plane_index'] + 1:03d}")
            if len(outputs) == 1 and len(regions) == 1:
                name = original_name
            else:
                name = f"{original_name}_region_{region['region_index'] + 1:02d}_{output_index + 1:02d}"
            fitted["name"] = name
            fitted["offset_m"] = float(output["offset_m"])
            fitted[region["span_key"]] = list(map(float, output["span_m"]))
            fitted["bounds_y"] = list(map(float, output["bounds_y"]))
            fitted["source_bounds_y"] = list(region["bounds_y"])
            fitted["plane_height_m"] = float(output["bounds_y"][1] - output["bounds_y"][0])
            fitted["enclosing_area_m2"] = float(
                (output["span_m"][1] - output["span_m"][0]) * fitted["plane_height_m"]
            )
            fitted["final_enclosing_area_m2"] = fitted["enclosing_area_m2"]
            fitted["plane_equation"] = [
                *map(float, fitted["normal_unit"]),
                -float(np.dot(np.asarray(fitted["normal_unit"], dtype=float),
                              fitted["offset_m"] * (basis_a if fitted["axis"] == "Dir A" else basis_b))),
            ]
            fitted["floor_perimeter_fit"] = {
                "matched_floor_indices": output["floor_indices"],
                "source_offset_m": float(plane["offset_m"]),
                "source_span_m": region["source_span_m"],
                "source_region_y_m": region["bounds_y"],
            }
            fitted_planes.append(fitted)
            source_index = region["plane_index"]
            if source_index not in source_colors:
                path = Path(plane["individual_ply"])
                mesh = o3d.io.read_triangle_mesh(str(path)) if path.is_file() else o3d.geometry.TriangleMesh()
                colors = np.asarray(mesh.vertex_colors)
                source_colors[source_index] = (
                    np.mean(colors, axis=0) if len(colors) else np.array([0.65, 0.65, 0.65])
                )
            vertices, triangles = _plane_quad_geometry(fitted, basis_a, basis_b)
            colors = np.tile(source_colors[source_index], (len(vertices), 1))
            individual_path = individual_dir / f"{name}.ply"
            _write(individual_path, vertices, triangles, colors)
            fitted["individual_ply"] = str(individual_path)
            all_vertices.append(vertices)
            all_triangles.append(triangles + vertex_offset)
            all_colors.append(colors)
            vertex_offset += len(vertices)

    combined_path = output_dir / "wall_enclosing_planes.ply"
    _write(combined_path, np.concatenate(all_vertices), np.concatenate(all_triangles), np.concatenate(all_colors))

    floor_audits = []
    for floor in floors:
        floor_index = int(floor["floor_index"])
        floor_y = float(floor["floor_y_m"])
        actual_lines = [
            _line_at_floor(plane, floor_y)
            for plane in fitted_planes
            if plane["bounds_y"][0] - GEOMETRY_EPS_M <= floor_y <= plane["bounds_y"][1] + GEOMETRY_EPS_M
        ]
        target_points = np.asarray(floor["aligned_perimeter_xz_m"], dtype=float)
        target = unary_union([
            LineString([first, second])
            for first, second in zip(target_points, np.roll(target_points, -1, axis=0))
        ])
        actual = unary_union(actual_lines)
        tolerant_actual = actual.buffer(CHECK_TOLERANCE_M, cap_style="flat", join_style="mitre")
        audit = {
            "floor_index": floor_index,
            "floor_y_m": floor_y,
            "target_perimeter_m": float(target.length),
            "uncovered_perimeter_m": float(target.difference(tolerant_actual).length),
            "wall_outside_perimeter_m": float(actual.difference(target.buffer(
                CHECK_TOLERANCE_M, cap_style="flat", join_style="mitre"
            )).length),
            "hausdorff_distance_m": float(target.hausdorff_distance(actual)) if not actual.is_empty else None,
            "perimeter_coverage": float(1.0 - target.difference(tolerant_actual).length / target.length),
            "comparison_tolerance_m": CHECK_TOLERANCE_M,
            "image": str(output_dir / f"floor_{floor_index:02d}_wall_fit_check.png"),
        }
        _plot_fit(Path(audit["image"]), floor, actual_lines, audit)
        floor_audits.append(audit)

    output_manifest = copy.deepcopy(source)
    output_manifest["source_wall_manifest"] = str(wall_manifest_path)
    output_manifest["source_floor_fit_manifest"] = str(floor_fit_manifest)
    output_manifest["wall_planes"] = fitted_planes
    output_manifest["total_wall_planes_count"] = len(fitted_planes)
    output_manifest["floor_perimeter_fit"] = {
        "method": "match supported orthogonal wall-plane regions to final aligned floor edges and preserve adjacent panel junctions",
        "maximum_line_offset_m": MAX_LINE_OFFSET_M,
        "minimum_edge_overlap_m": MIN_EDGE_OVERLAP_M,
        "minimum_overlap_fraction_of_shorter": MIN_EDGE_OVERLAP_FRACTION,
        "alignment_frame": "final aligned floor coordinates are applied directly in the wall manifest orthogonal basis",
        "edges": edge_audit,
        "floors": floor_audits,
        "all_perimeters_match_within_tolerance": all(
            audit["uncovered_perimeter_m"] <= 1e-6 and audit["wall_outside_perimeter_m"] <= 1e-6
            for audit in floor_audits
        ),
    }
    output_manifest["artifacts"] = {
        **source.get("artifacts", {}),
        "enclosing_planes": str(combined_path),
        "individual_enclosing_planes": str(individual_dir),
    }
    manifest_path = output_dir / "wall_planes_manifest.json"
    manifest_path.write_text(json.dumps(output_manifest, indent=2) + "\n")
    return {
        "output_dir": str(output_dir),
        "wall_planes_manifest": str(manifest_path),
        "wall_enclosing_planes": str(combined_path),
        "floor_checks": floor_audits,
        "all_perimeters_match_within_tolerance": output_manifest["floor_perimeter_fit"]["all_perimeters_match_within_tolerance"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floor-fit-manifest", type=Path, required=True)
    parser.add_argument("--wall-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = fit_wall_planes_to_floor_perimeters(
        args.floor_fit_manifest, args.wall_manifest, args.output_dir
    )
    floor_data = json.loads(args.floor_fit_manifest.read_text())
    floor_data["wall_planes_fitted_to_floor_perimeters"] = result
    args.floor_fit_manifest.write_text(json.dumps(floor_data, indent=2) + "\n")
    for floor in result["floor_checks"]:
        print(
            f"Floor {floor['floor_index']}: perimeter coverage {floor['perimeter_coverage']:.3%}, "
            f"uncovered {floor['uncovered_perimeter_m']:.4f} m, "
            f"outside {floor['wall_outside_perimeter_m']:.4f} m -> {floor['image']}"
        )
    print(f"Saved wall-plane candidate: {result['wall_enclosing_planes']}")
    print(f"Saved manifest: {result['wall_planes_manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
