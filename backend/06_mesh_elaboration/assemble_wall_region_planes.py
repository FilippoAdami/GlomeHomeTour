#!/usr/bin/env python3
"""Assemble fitted wall-region planes into a supported, closed side shell."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from planar_pattern_regularization import regularize_planar_patches

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_wall_planes as wall_fit  # noqa: E402

JUNCTION_DISTANCE_M = 0.20
PARALLEL_CARVE_DISTANCE_M = 0.30
MIN_CARVE_DIMENSION_M = 0.20
MIN_CARVE_LONG_DIMENSION_M = 1.0
FLOOR_CEILING_SEARCH_DISTANCE_M = 1.0
TOPOLOGY_TOLERANCE_M = 0.01
FLOOR_CONTOUR_HEIGHT_M = 0.30


def _triangle_areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    corners = vertices[triangles]
    return np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0],
                                   corners[:, 2] - corners[:, 0]), axis=1) * 0.5


def _plane_regions(plane: dict) -> list[dict]:
    return plane["trim_regions"]


def _rect_intersection(a: dict, b: dict) -> tuple[float, float, float, float] | None:
    u0, u1 = max(a["span_m"][0], b["span_m"][0]), min(a["span_m"][1], b["span_m"][1])
    y0, y1 = max(a["bounds_y"][0], b["bounds_y"][0]), min(a["bounds_y"][1], b["bounds_y"][1])
    return (u0, u1, y0, y1) if u1 > u0 + 1e-8 and y1 > y0 + 1e-8 else None


def _subtract_rect(rect: dict, cut: tuple[float, float, float, float]) -> list[dict]:
    u0, u1, y0, y1 = rect["span_m"] + rect["bounds_y"]
    overlap = _rect_intersection(rect, {"span_m": list(cut[:2]), "bounds_y": list(cut[2:])})
    if overlap is None:
        return [rect]
    c0, c1, d0, d1 = overlap
    result = []
    for span, height in (([u0, c0], [y0, y1]), ([c1, u1], [y0, y1]),
                         ([c0, c1], [y0, d0]), ([c0, c1], [d1, y1])):
        if span[1] > span[0] + 1e-8 and height[1] > height[0] + 1e-8:
            result.append({"span_m": span, "bounds_y": height})
    return result


def _support_score(plane: dict, faces: np.ndarray, vertices: np.ndarray,
                   triangles: np.ndarray, areas: np.ndarray,
                   rect: tuple[float, float, float, float], tangent: np.ndarray,
                   axis: np.ndarray) -> tuple[float, float]:
    if not len(faces):
        return float("inf"), 0.0
    corners = vertices[triangles[faces]]
    centers = corners.mean(axis=1)
    u, y = centers @ tangent, centers[:, 1]
    selected = ((u >= rect[0]) & (u <= rect[1]) & (y >= rect[2]) & (y <= rect[3]))
    if not selected.any():
        return float("inf"), 0.0
    ids = faces[selected]
    positive_area = areas[ids] > 1e-12
    if not positive_area.any():
        return float("inf"), 0.0
    ids = ids[positive_area]
    distances = wall_fit._triangle_plane_distances(
        vertices, triangles, ids, axis, plane["offset_m"]
    )
    return float(np.average(distances, weights=areas[ids])), float(areas[ids].sum())


def _make_connector(first: dict, second: dict, edge: float,
                    bounds_y: list[float], axes: np.ndarray) -> dict:
    other_family = 1 - int(first["family"])
    axis = axes[other_family]
    normal = np.asarray(first["normal_unit"])
    other_normal = np.asarray(second["normal_unit"])
    if np.dot(other_normal - normal, axis) < 0.0:
        axis = -axis
    key = "span_w_m" if other_family == 0 else "span_u_m"
    return {
        "id": -1, "name": "wall_closure_connector", "axis": "Dir A" if other_family == 0 else "Dir B",
        "family": other_family, "offset_m": edge, "normal_unit": axis.tolist(),
        key: sorted([first["offset_m"], second["offset_m"]]), "bounds_y": list(bounds_y),
        "trim_regions": [{"span_m": sorted([first["offset_m"], second["offset_m"]]),
                          "bounds_y": list(bounds_y)}],
        "source_faces": np.empty(0, dtype=int), "source_region_ids": [], "support_area_m2": 0.0,
        "color_rgb": list(map(int, (np.asarray(first["color_rgb"])
                                    + np.asarray(second["color_rgb"])) // 2)),
        "inferred_connector": True,
        "local_connector": True,
    }


def _make_horizontal_transition(transition: dict, axes: np.ndarray) -> trimesh.Trimesh:
    normal_axis, tangent = axes[transition["family"]], axes[1 - transition["family"]]
    offset0, offset1 = transition["offsets_m"]
    u0, u1 = transition["span_m"]
    y = transition["y_m"]
    vertices = np.asarray([
        offset0 * normal_axis + u0 * tangent + np.array([0., y, 0.]),
        offset0 * normal_axis + u1 * tangent + np.array([0., y, 0.]),
        offset1 * normal_axis + u1 * tangent + np.array([0., y, 0.]),
        offset1 * normal_axis + u0 * tangent + np.array([0., y, 0.]),
    ])
    faces = np.asarray([[0, 1, 2], [0, 2, 3]])
    if np.dot(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]),
              np.array([0., transition["normal_y"], 0.])) < 0.:
        faces = faces[:, ::-1]
    rgba = np.tile([*transition["color_rgb"], 128], (4, 1))
    return trimesh.Trimesh(vertices=vertices, faces=faces,
                           visual=trimesh.visual.ColorVisuals(vertex_colors=rgba),
                           process=False)


def _snap_panel_junctions(planes: list[dict]) -> None:
    """Close near corner gaps without propagating a tall panel's full height to neighbors."""
    changed = True
    while changed:
        changed = False
        for first in (p for p in planes if p["family"] == 0):
            for second in (p for p in planes if p["family"] == 1):
                first_key, second_key = "span_w_m", "span_u_m"
                first_span, second_span = first[first_key], second[second_key]
                first_y, second_y = first["bounds_y"], second["bounds_y"]
                if first_y[0] > second_y[1] and first_y[0] - second_y[1] <= JUNCTION_DISTANCE_M:
                    first_y[0] = second_y[1]; changed = True
                elif second_y[0] > first_y[1] and second_y[0] - first_y[1] <= JUNCTION_DISTANCE_M:
                    second_y[0] = first_y[1]; changed = True
                if first_y[1] < second_y[0] and second_y[0] - first_y[1] <= JUNCTION_DISTANCE_M:
                    first_y[1] = second_y[0]; changed = True
                elif second_y[1] < first_y[0] and first_y[0] - second_y[1] <= JUNCTION_DISTANCE_M:
                    second_y[1] = first_y[0]; changed = True
                if min(first_y[1], second_y[1]) - max(first_y[0], second_y[0]) < -1e-8:
                    continue
                x, z = first["offset_m"], second["offset_m"]
                dist_first = max(first_span[0] - z, 0.0, z - first_span[1])
                dist_second = max(second_span[0] - x, 0.0, x - second_span[1])
                if max(dist_first, dist_second) > JUNCTION_DISTANCE_M + TOPOLOGY_TOLERANCE_M:
                    continue
                new_first = [min(first_span[0], z), max(first_span[1], z)]
                new_second = [min(second_span[0], x), max(second_span[1], x)]
                if new_first != first_span:
                    first[first_key] = new_first; changed = True
                if new_second != second_span:
                    second[second_key] = new_second; changed = True


def _add_parallel_carves(planes: list[dict], owner: np.ndarray, vertices: np.ndarray,
                         triangles: np.ndarray, areas: np.ndarray,
                         axes: np.ndarray, vertical_bounds: tuple[float, float]) -> tuple[list[dict], list[dict]]:
    connectors, transitions = [], []
    for plane in planes:
        plane["base_trim_regions"] = [
            {"span_m": list(region["span_m"]), "bounds_y": list(region["bounds_y"])}
            for region in plane["trim_regions"]
        ]
        plane["carve_rects"] = []
    for i, first in enumerate(planes):
        for j in range(i):
            second = planes[j]
            if (first["family"] != second["family"]
                    or np.dot(first["normal_unit"], second["normal_unit"]) < 0.99
                    or abs(first["offset_m"] - second["offset_m"]) >= PARALLEL_CARVE_DISTANCE_M):
                continue
            tangent = axes[1 - first["family"]]
            cuts: dict[int, list[tuple[float, float, float, float]]] = {i: [], j: []}
            overlaps = []
            for first_region in first["trim_regions"]:
                for second_region in second["trim_regions"]:
                    overlap = _rect_intersection(first_region, second_region)
                    if (overlap is None
                            or min(overlap[1] - overlap[0], overlap[3] - overlap[2])
                            <= MIN_CARVE_DIMENSION_M
                            or max(overlap[1] - overlap[0], overlap[3] - overlap[2])
                            <= MIN_CARVE_LONG_DIMENSION_M):
                        continue
                    first_score = _support_score(
                        first, np.flatnonzero(owner == i), vertices, triangles, areas,
                        overlap, tangent, axes[first["family"]]
                    )
                    second_score = _support_score(
                        second, np.flatnonzero(owner == j), vertices, triangles, areas,
                        overlap, tangent, axes[second["family"]]
                    )
                    # Preserve the surface with more local mesh support; residual breaks ties.
                    first_wins = (first_score[1], -first_score[0], first["support_area_m2"]) >= (
                        second_score[1], -second_score[0], second["support_area_m2"]
                    )
                    winner, loser, winner_id, loser_id = (
                        (first, second, i, j) if first_wins else (second, first, j, i)
                    )
                    winner_region = first_region if winner is first else second_region
                    cut_y = [max(winner_region["bounds_y"][0], vertical_bounds[0]),
                             min(winner_region["bounds_y"][1], vertical_bounds[1])]
                    if cut_y[1] <= cut_y[0]:
                        continue
                    loser_cut = (overlap[0], overlap[1], *cut_y)
                    cuts[loser_id].append(loser_cut)
                    loser.setdefault("carve_rects", []).append(loser_cut)
                    winner["local_support_panel"] = True
                    overlaps.append((loser_cut, winner, loser, cut_y.copy()))
            for loser_id in (i, j):
                if not cuts[loser_id]:
                    continue
                loser = planes[loser_id]
                remaining = []
                for region in loser["trim_regions"]:
                    pieces = [region]
                    for cut in cuts[loser_id]:
                        pieces = [piece for part in pieces for piece in _subtract_rect(part, cut)]
                    remaining.extend(pieces)
                loser["trim_regions"] = remaining
            for overlap, _, _, local_y in overlaps:
                if abs(first["offset_m"] - second["offset_m"]) > 1e-6:
                    for edge in (overlap[0], overlap[1]):
                        connectors.append(_make_connector(
                            first, second, edge, local_y, axes
                        ))
                    color = list(map(int, (np.asarray(first["color_rgb"])
                                           + np.asarray(second["color_rgb"])) // 2))
                    for edge_index, y in enumerate(local_y):
                        if vertical_bounds[0] + 1e-6 < y < vertical_bounds[1] - 1e-6:
                            transitions.append({
                                "plane_ids": [int(winner["id"]), int(loser["id"])],
                                "family": int(first["family"]),
                                "offsets_m": [float(first["offset_m"]), float(second["offset_m"])],
                                "span_m": [float(overlap[0]), float(overlap[1])],
                                "y_m": float(y), "normal_y": -1 if edge_index == 0 else 1,
                                "color_rgb": color,
                            })
    return connectors, transitions


def _cross_section_cycles(
    planes: list[dict], vertical_bounds: tuple[float, float] | None = None,
) -> tuple[list[dict], dict]:
    """Keep only panel intervals that belong to a closed loop at each height band."""
    events = sorted({float(y) for plane in planes for region in _plane_regions(plane)
                     for y in region["bounds_y"]})
    kept: dict[int, list[dict]] = {i: [] for i in range(len(planes))}
    band_reports = []
    for y0, y1 in zip(events, events[1:]):
        if y1 - y0 <= 1e-8:
            continue
        y = (y0 + y1) * 0.5
        segments = []
        for pidx, plane in enumerate(planes):
            spans = sorted((r["span_m"] for r in plane["trim_regions"]
                            if r["bounds_y"][0] <= y <= r["bounds_y"][1]))
            merged = []
            for lo, hi in spans:
                if merged and lo <= merged[-1][1] + TOPOLOGY_TOLERANCE_M:
                    merged[-1][1] = max(merged[-1][1], hi)
                else:
                    merged.append([lo, hi])
            segments.extend({"plane": pidx, "axis": plane["axis"], "offset": plane["offset_m"],
                             "span": span, "crossings": []} for span in merged)

        for idx, first in enumerate(segments):
            for second in segments[:idx]:
                if first["axis"] == second["axis"]:
                    continue
                a, b = (first, second) if first["axis"] == "Dir A" else (second, first)
                x, z = a["offset"], b["offset"]
                if (a["span"][0] - TOPOLOGY_TOLERANCE_M <= z <= a["span"][1] + TOPOLOGY_TOLERANCE_M
                        and b["span"][0] - TOPOLOGY_TOLERANCE_M <= x <= b["span"][1] + TOPOLOGY_TOLERANCE_M):
                    node = (round(x, 5), round(z, 5))
                    a["crossings"].append((z, node)); b["crossings"].append((x, node))

        nodes, edges, seen_edges = set(), [], set()
        for segment in segments:
            crossings = sorted(set(segment["crossings"]))
            for (lo, first), (hi, second) in zip(crossings, crossings[1:]):
                if first == second:
                    continue
                key = tuple(sorted((first, second)))
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                nodes.update((first, second))
                edges.append({"a": first, "b": second, "plane": segment["plane"], "span": [lo, hi]})

        adjacency = {node: [] for node in nodes}
        for edge_id, edge in enumerate(edges):
            adjacency[edge["a"]].append((edge["b"], edge_id))
            adjacency[edge["b"]].append((edge["a"], edge_id))
        cycle_edges = set()
        for edge_id, edge in enumerate(edges):
            seen, stack = {edge["a"]}, [edge["a"]]
            while stack:
                node = stack.pop()
                for neighbor, used in adjacency[node]:
                    if used == edge_id or neighbor in seen:
                        continue
                    seen.add(neighbor); stack.append(neighbor)
            if edge["b"] in seen:
                cycle_edges.add(edge_id)
                kept[edge["plane"]].append({"span_m": edge["span"], "bounds_y": [y0, y1]})

        # Every surviving connected component must contain a cycle; bridge edges were removed above.
        cyclic_nodes = {node for edge_id in cycle_edges
                        for node in (edges[edge_id]["a"], edges[edge_id]["b"])}
        component_count = 0
        remaining = set(cyclic_nodes)
        while remaining:
            component_count += 1
            stack = [remaining.pop()]
            while stack:
                node = stack.pop()
                for neighbor, edge_id in adjacency[node]:
                    if edge_id in cycle_edges and neighbor in remaining:
                        remaining.remove(neighbor); stack.append(neighbor)
        cycle_rank = len(cycle_edges) - len(cyclic_nodes) + component_count
        band_reports.append({"y_bounds": [y0, y1], "cycle_edge_count": len(cycle_edges),
                             "closed_component_count": component_count if cycle_edges else 0,
                             "cycle_rank": max(0, cycle_rank), "is_closed": bool(cycle_edges)})

    if not any(band["cycle_edge_count"] for band in band_reports):
        raise ValueError("Fitted planes do not contain a closed, supported horizontal loop")
    active_bands = [i for i, band in enumerate(band_reports) if band["cycle_edge_count"]]
    participating = [i for i, regions in kept.items() if regions]
    if vertical_bounds is not None:
        floor_y, ceiling_y = map(float, vertical_bounds)
        if ceiling_y <= floor_y:
            raise ValueError("Measured ceiling is not above the measured floor")
        closed_runs = []
        for band in band_reports:
            y0, y1 = map(float, band["y_bounds"])
            if not band["is_closed"]:
                continue
            if closed_runs and abs(y0 - closed_runs[-1][-1][1]) <= 1e-4:
                closed_runs[-1].append((y0, y1))
            else:
                closed_runs.append([(y0, y1)])
        supported_runs = [run for run in closed_runs
                          if run[0][0] <= floor_y + 0.10 and run[-1][1] > floor_y + 1e-4]
        if not supported_runs:
            bands = [(item["y_bounds"], item["is_closed"]) for item in band_reports]
            raise ValueError(
                f"Wall planes do not form a closed loop at the fitted floor height {floor_y:.4f}; "
                f"height bands: {bands}"
            )
        wall_run = max(supported_runs, key=lambda run: run[-1][1] - run[0][0])
        floor_y = max(floor_y, wall_run[0][0])
        extended = []
        for pidx in participating:
            plane = planes[pidx]
            base_regions = plane.get("base_trim_regions", plane["trim_regions"])
            if plane.get("local_support_panel") or plane.get("local_connector"):
                regions = [
                    {"span_m": list(region["span_m"]),
                     "bounds_y": [max(region["bounds_y"][0], floor_y),
                                  min(region["bounds_y"][1], ceiling_y)]}
                    for region in base_regions
                    if min(region["bounds_y"][1], ceiling_y)
                    > max(region["bounds_y"][0], floor_y)
                ]
            else:
                regions = [{"span_m": list(region["span_m"]),
                            "bounds_y": [floor_y, ceiling_y]}
                           for region in base_regions]
            for cut in plane.get("carve_rects", []):
                regions = [piece for region in regions for piece in _subtract_rect(region, cut)]
            plane["trim_regions"] = regions
            if not regions:
                continue
            extended.append(plane)
        final, final_topology = _cross_section_cycles(extended)
        open_bands = [band["y_bounds"] for band in final_topology["height_bands"]
                      if not band["is_closed"]]
        if open_bands:
            raise ValueError(f"Extended wall enclosure has open height bands: {open_bands}")
        return final, {
            "supported_height_bands": band_reports,
            "closed_height_band_count": len(active_bands),
            "extended_to_floor_ceiling": [floor_y, ceiling_y],
            "pruned_panel_count": len(planes) - len(participating),
            "final_topology": final_topology,
        }
    for pidx, plane in enumerate(planes):
        plane["trim_regions"] = kept[pidx]
    return [p for p in planes if p["trim_regions"]], {
        "height_bands": band_reports,
        "closed_height_band_count": len(active_bands),
        "pruned_panel_count": len(planes) - len(participating),
    }


def _copy_planes_for_topology(planes: list[dict]) -> list[dict]:
    return [
        {**plane, "trim_regions": [
            {**region, "span_m": list(region["span_m"]),
             "bounds_y": list(region["bounds_y"])}
            for region in plane["trim_regions"]
        ]}
        for plane in planes
    ]


def _extend_supported_plane_to_close_floor_slice(
    planes: list[dict], floor_y: float,
) -> list[dict]:
    """Close the floor slice with supported panel extensions, preserving overlap cuts."""
    target_y = floor_y + FLOOR_CONTOUR_HEIGHT_M
    try:
        _, topology = _cross_section_cycles(_copy_planes_for_topology(planes))
    except ValueError:
        return []
    if any(band["is_closed"] and band["y_bounds"][0] <= target_y <= band["y_bounds"][1]
           for band in topology["height_bands"]):
        return []

    options = []
    for plane_index, plane in enumerate(planes):
        if plane.get("inferred_connector"):
            continue
        for region_index, region in enumerate(plane["trim_regions"]):
            lower, upper = map(float, region["bounds_y"])
            if lower <= target_y or upper <= target_y:
                continue
            span_length = float(region["span_m"][1] - region["span_m"][0])
            options.append(((lower - floor_y) * span_length, plane_index, region_index, lower))

    def closes_floor(extensions: list[tuple]) -> bool:
        trial = _copy_planes_for_topology(planes)
        for _, plane_index, region_index, _ in extensions:
            trial[plane_index]["trim_regions"][region_index]["bounds_y"][0] = floor_y
        for plane in trial:
            for cut in plane.get("carve_rects", []):
                plane["trim_regions"] = [piece for region in plane["trim_regions"]
                                         for piece in _subtract_rect(region, cut)]
        try:
            _, trial_topology = _cross_section_cycles(trial)
        except ValueError:
            return False
        return any(band["is_closed"] and band["y_bounds"][0] <= target_y <= band["y_bounds"][1]
                   for band in trial_topology["height_bands"])

    candidates = [[option] for option in options if closes_floor([option])]
    if not candidates and options and closes_floor(options):
        # Several missing panels may need to reach the floor together; prune redundant extensions.
        group = options.copy()
        for option in sorted(options, reverse=True):
            without = [item for item in group if item != option]
            if closes_floor(without):
                group = without
        candidates.append(group)

    if not candidates:
        return []
    selected = min(candidates, key=lambda group: sum(item[0] for item in group))
    repairs = []
    for _, plane_index, region_index, original_lower in selected:
        plane = planes[plane_index]
        plane["trim_regions"][region_index]["bounds_y"][0] = floor_y
        plane["bounds_y"][0] = min(float(plane["bounds_y"][0]), floor_y)
        repairs.append({
            "plane_id": int(plane["id"]),
            "source_region_ids": list(map(int, plane.get("source_region_ids", []))),
            "extended_from_y_m": original_lower,
            "extended_to_y_m": floor_y,
            "reason": "required to close the wall contour at the floor projection height",
        })
    for plane_index in {item[1] for item in selected}:
        plane = planes[plane_index]
        for cut in plane.get("carve_rects", []):
            plane["trim_regions"] = [piece for region in plane["trim_regions"]
                                     for piece in _subtract_rect(region, cut)]
        if "base_trim_regions" in plane:
            plane["base_trim_regions"] = [
                {"span_m": list(region["span_m"]), "bounds_y": list(region["bounds_y"])}
                for region in plane["trim_regions"]
            ]
    return repairs


def _split_panels_at_transitions(planes: list[dict], transitions: list[dict]) -> None:
    height_cuts = sorted({float(item["y_m"]) for item in transitions})
    span_cuts: dict[int, set[float]] = {}
    for item in transitions:
        for plane_id in item["plane_ids"]:
            span_cuts.setdefault(plane_id, set()).update(item["span_m"])
    for plane in planes:
        for item in transitions:
            if int(plane["id"]) not in item["plane_ids"] and int(plane["family"]) != item["family"]:
                span_cuts.setdefault(int(plane["id"]), set()).update(item["offsets_m"])
        split_regions = []
        for region in plane["trim_regions"]:
            u0, u1 = region["span_m"]
            y0, y1 = region["bounds_y"]
            us = sorted({u0, u1, *(u for u in span_cuts.get(int(plane["id"]), ())
                                      if u0 + 1e-8 < u < u1 - 1e-8)})
            ys = sorted({y0, y1, *(y for y in height_cuts if y0 + 1e-8 < y < y1 - 1e-8)})
            split_regions.extend(
                {"span_m": [lo, hi], "bounds_y": [bottom, top]}
                for lo, hi in zip(us, us[1:])
                for bottom, top in zip(ys, ys[1:])
            )
        plane["trim_regions"] = split_regions


def _build_planes(regions_dir: Path) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray, dict]:
    source_mesh = regions_dir / "wall_regions_colored.ply"
    region_manifest = json.loads((regions_dir / "wall_regions_manifest.json").read_text())
    fit_manifest = json.loads((regions_dir / "wall_region_planes_manifest.json").read_text())
    label_path = regions_dir / region_manifest.get("triangle_labels_file", "wall_regions_triangle_labels.npy")
    if not source_mesh.is_file() or not label_path.is_file():
        raise ValueError(f"Expected colored PLY and triangle-label sidecar in {regions_dir}")
    mesh = o3d.io.read_triangle_mesh(str(source_mesh))
    vertices, triangles, labels = np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.load(label_path)
    if not len(triangles) or labels.ndim != 1 or len(labels) != len(triangles):
        raise ValueError("Wall-region PLY and triangle-label sidecar are empty or misaligned")
    axes = np.asarray(fit_manifest["orthogonal_horizontal_normal_directions"], dtype=float)
    if axes.shape == (3, 3):
        axes = axes[:2]
    if axes.shape != (2, 3):
        raise ValueError("Plane manifest must contain two horizontal normal directions")
    axis_lengths = np.linalg.norm(axes, axis=1)
    if not np.isfinite(axes).all() or np.any(axis_lengths <= 1e-8):
        raise ValueError("Plane manifest contains invalid horizontal directions")
    axes /= axis_lengths[:, None]
    areas = _triangle_areas(vertices, triangles)
    centers = vertices[triangles].mean(axis=1)
    instances_dir = regions_dir.parent
    floor_y, ceiling_y, _, floor_center_xz, floor_points, floor_weights, ceiling_points, ceiling_weights = (
        wall_fit._vertical_enclosure_bounds(
            instances_dir, vertices, centers, areas, include_ceilings=False
        )
    )
    owner = np.full(len(triangles), -1, dtype=int)
    planes, excluded_small_support, excluded_nonvertical = [], [], []
    for pidx, item in enumerate(fit_manifest.get("planes", [])):
        region_ids = sorted(set(map(int, item["source_region_ids"])))
        if float(item.get("fit_area_scene_units_squared", 0.0)) < wall_fit.MIN_WALL_PLANE_FINAL_SUPPORT_AREA_M2:
            excluded_small_support.append({"region_id": int(item["region_id"]),
                                           "support_area_scene_units_squared": float(
                                               item.get("fit_area_scene_units_squared", 0.0))})
            continue
        faces = np.flatnonzero(np.isin(labels, region_ids))
        if not len(faces):
            continue
        family = int(item["orientation_family"])
        if family not in (0, 1):
            excluded_nonvertical.append({"region_id": int(item["region_id"]),
                                         "orientation_family": family})
            continue
        if np.any(owner[faces] >= 0):
            raise ValueError("A source triangle belongs to more than one fitted plane")
        pidx = len(planes)
        owner[faces] = pidx
        normal = np.asarray(item["normal"], dtype=float)
        normal_length = np.linalg.norm(normal)
        if not np.isfinite(normal).all() or normal_length <= 1e-8:
            raise ValueError(f"Plane {item['region_id']} has an invalid normal")
        normal /= normal_length
        denominator = float(normal @ axes[family])
        if abs(denominator) < 0.99:
            raise ValueError(f"Plane {item['region_id']} is not aligned to its declared direction")
        coordinate = -float(item["offset"]) / denominator
        tangent = axes[1 - family]
        points = vertices[triangles[faces]].reshape(-1, 3)
        # shortcut: support uses a projected bounding interval; preserve holes as polygons if needed.
        span = [float((points @ tangent).min()), float((points @ tangent).max())]
        source_bounds = [float(points[:, 1].min()), float(points[:, 1].max())]
        plane = {
            "id": pidx, "name": f"wall_plane_{int(item['region_id']):03d}",
            "axis": "Dir A" if family == 0 else "Dir B", "family": family,
            "offset_m": coordinate, "normal_unit": normal.tolist(),
            ("span_w_m" if family == 0 else "span_u_m"): span,
            "bounds_y": source_bounds, "source_bounds_y": source_bounds,
            "source_faces": faces, "source_region_ids": region_ids,
            "support_area_m2": float(areas[faces].sum()),
            "color_rgb": list(map(int, item["color_rgb"])), "inferred_connector": False,
        }
        lo, hi, source = wall_fit._local_vertical_bounds(
            plane, floor_points, floor_weights, ceiling_points, ceiling_weights,
            (floor_y, ceiling_y), axes[family], tangent,
            support_distance_m=FLOOR_CEILING_SEARCH_DISTANCE_M,
        )
        plane["bounds_y"] = [float(lo), float(hi)]
        plane["vertical_bounds_source"] = source
        plane["trim_regions"] = [{"span_m": span.copy(), "bounds_y": [float(lo), float(hi)]}]
        planes.append(plane)
    if not planes:
        raise ValueError("Plane manifest contains no supported wall planes")
    # The region labels are already a one-owner partition; retained owners are audited again after carving.
    if len(planes) > 1:
        _snap_panel_junctions(planes)
        for plane in planes:
            span_key = "span_w_m" if plane["family"] == 0 else "span_u_m"
            plane["trim_regions"] = [{"span_m": list(plane[span_key]),
                                      "bounds_y": list(plane["bounds_y"])}]
    return planes, owner, vertices, triangles, labels, areas, {
        "axes": axes, "mesh": mesh, "source_mesh": source_mesh, "centers": centers,
        "floor_y": float(floor_y), "ceiling_y": float(ceiling_y),
        "floor_center_xz": floor_center_xz.tolist(),
        "excluded_small_support": excluded_small_support,
        "excluded_nonvertical": excluded_nonvertical,
        "black_region_ids": [int(region["region_id"]) for region in region_manifest.get("regions", [])
                             if region.get("painted_black", region.get("color_rgb") == [0, 0, 0])],
    }


def _weld_shell(plane_meshes: list[trimesh.Trimesh], floor_y: float,
                ceiling_y: float) -> tuple[trimesh.Trimesh, dict]:
    shell = trimesh.util.concatenate(plane_meshes)
    face_colors = np.asarray(shell.visual.face_colors).copy()
    shell.merge_vertices(digits_vertex=5)
    shell.remove_unreferenced_vertices()
    edge_counts = np.bincount(shell.edges_unique_inverse, minlength=len(shell.edges_unique))
    boundary = shell.edges_unique[edge_counts == 1]
    nonmanifold_count = int(np.count_nonzero(edge_counts > 2))
    if len(boundary):
        y = shell.vertices[boundary, 1]
        at_caps = (((np.abs(y[:, 0] - floor_y) <= 1e-5)
                    & (np.abs(y[:, 1] - floor_y) <= 1e-5))
                   | ((np.abs(y[:, 0] - ceiling_y) <= 1e-5)
                      & (np.abs(y[:, 1] - ceiling_y) <= 1e-5)))
        cap_count = int(at_caps.sum())
    else:
        cap_count = 0
    side_count = int(len(boundary) - cap_count)
    report = {
        "is_closed_between_floor_and_ceiling": side_count == 0 and nonmanifold_count == 0,
        "floor_ceiling_boundary_edge_count": cap_count,
        "side_boundary_edge_count": side_count,
        "nonmanifold_edge_count": nonmanifold_count,
        "welded_vertex_count": int(len(shell.vertices)),
    }
    if not report["is_closed_between_floor_and_ceiling"]:
        side_edges = boundary[~at_caps] if len(boundary) else np.empty((0, 2), dtype=np.int64)
        side_edge_vertices = np.round(shell.vertices[side_edges], 5).tolist()
        raise ValueError(
            f"Assembled shell has {side_count} open side edges and "
            f"{nonmanifold_count} non-manifold edges in [{floor_y:.5f}, {ceiling_y:.5f}]; "
            f"sample vertices: {side_edge_vertices[:16]}"
        )

    # PLY vertex colors interpolate across shared vertices; split color seams after topology validation.
    corner_ids = shell.faces.reshape(-1)
    corner_colors = np.repeat(face_colors, 3, axis=0)
    unique_corners, inverse = np.unique(
        np.column_stack((corner_ids, corner_colors)), axis=0, return_inverse=True
    )
    colored_shell = trimesh.Trimesh(
        vertices=shell.vertices[unique_corners[:, 0]],
        faces=inverse.reshape(-1, 3),
        visual=trimesh.visual.ColorVisuals(
            vertex_colors=unique_corners[:, 1:].astype(np.uint8)
        ),
        process=False,
    )
    report["color_seam_vertex_count"] = int(len(colored_shell.vertices) - len(shell.vertices))
    report["export_vertex_count"] = int(len(colored_shell.vertices))
    return colored_shell, report


def _source_triangle_colors(region_manifest: dict) -> np.ndarray:
    colors = []
    for path in region_manifest.get("wall_meshes", []):
        mesh = o3d.io.read_triangle_mesh(str(path))
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        vertex_colors = np.asarray(mesh.vertex_colors)
        if len(vertex_colors) != len(vertices):
            vertex_colors = np.full((len(vertices), 3), 0.7, dtype=np.float64)
        colors.append(vertex_colors[faces].mean(axis=1))
    return np.vstack(colors) if colors else np.empty((0, 3), dtype=np.float64)


def _assemble(regions_dir: Path) -> tuple[Path, Path, dict]:
    planes, owner, vertices, triangles, labels, areas, info = _build_planes(regions_dir)
    axes, source_mesh = info["axes"], info["source_mesh"]
    connectors, transitions = _add_parallel_carves(
        planes, owner, vertices, triangles, areas, axes,
        (info["floor_y"], info["ceiling_y"]),
    )
    planes.extend(connectors)
    inferred_floor_closures = _extend_supported_plane_to_close_floor_slice(
        planes, info["floor_y"]
    )
    planes, topology = _cross_section_cycles(
        planes, (info["floor_y"], info["ceiling_y"])
    )
    topology["inferred_floor_closure_planes"] = inferred_floor_closures
    info["floor_y"], info["ceiling_y"] = topology["extended_to_floor_ceiling"]
    transitions = [item for item in transitions
                   if info["floor_y"] + 1e-4 < float(item["y_m"])
                   < info["ceiling_y"] - 1e-4]
    surviving_ids = {int(plane["id"]) for plane in planes if not plane["inferred_connector"]}
    transitions = [transition for transition in transitions
                   if set(transition["plane_ids"]).issubset(surviving_ids)]
    _split_panels_at_transitions(planes, transitions)
    # Reindex surviving panels while keeping one source-region owner per triangle.
    owner_map = {int(plane["id"]): i for i, plane in enumerate(planes) if not plane["inferred_connector"]}
    final_owner = np.full(len(triangles), -1, dtype=np.int32)
    for old_id, new_id in owner_map.items():
        final_owner[owner == old_id] = new_id
    if final_owner.ndim != 1 or len(final_owner) != len(triangles):
        raise ValueError("Assembled support ownership lost triangle alignment")

    plane_meshes = []
    manifest_planes = []
    for index, plane in enumerate(planes):
        source_plane_id = int(plane["id"])
        plane["id"] = index + 1
        plane["name"] = f"wall_panel_{index + 1:03d}"
        plane_vertices, plane_faces = wall_fit._plane_quad_geometry(plane, axes[0], axes[1])
        rgba = np.tile([*plane["color_rgb"], 128], (len(plane_vertices), 1))
        plane_meshes.append(trimesh.Trimesh(vertices=plane_vertices, faces=plane_faces,
                                             visual=trimesh.visual.ColorVisuals(vertex_colors=rgba),
                                             process=False))
        source_ids = sorted(map(int, np.unique(labels[final_owner == index]))) if not plane["inferred_connector"] else []
        owned_count = int(np.count_nonzero(final_owner == index)) if not plane["inferred_connector"] else 0
        manifest_planes.append({
            "name": plane["name"], "source_plane_id": source_plane_id,
            "source_region_ids": source_ids, "triangle_count": owned_count,
            "axis": plane["axis"], "normal": plane["normal_unit"],
            "offset_m": float(plane["offset_m"]), "trim_regions": plane["trim_regions"],
            "equation": {
                "normal": axes[plane["family"]].tolist(),
                "constant": -float(plane["offset_m"]),
                "form": "normal dot position + constant = 0",
            },
            "color_rgb": plane["color_rgb"], "inferred_connector": bool(plane["inferred_connector"]),
            "vertical_bounds_source": plane.get("vertical_bounds_source", "parallel overlap connector"),
            "trim_region_count": len(plane["trim_regions"]),
        })

    region_manifest = json.loads((regions_dir / "wall_regions_manifest.json").read_text())
    support_colors = _source_triangle_colors(region_manifest)
    support_centers = info["centers"]
    if len(support_colors) != len(final_owner):
        raise ValueError("Original wall colors are not aligned with assembled support triangles")

    patches = []
    for index, (plane, mesh) in enumerate(zip(planes, plane_meshes)):
        owned = final_owner == index
        equation = manifest_planes[index]["equation"]
        panel_center = np.asarray(mesh.vertices).mean(axis=0)
        floor_center_xz = np.asarray(info["floor_center_xz"], dtype=np.float64)
        wall_normal = np.asarray(equation["normal"], dtype=np.float64)
        inward_hint = np.array([floor_center_xz[0] - panel_center[0], 0.0,
                                floor_center_xz[1] - panel_center[2]])
        relief_direction = wall_normal * (1.0 if inward_hint @ wall_normal >= 0 else -1.0)
        patches.append({
            "name": plane["name"], "mesh": mesh,
            "plane_equation": [*equation["normal"], equation["constant"]],
            "relief_direction": relief_direction.tolist(),
            "support_centers": support_centers[owned],
            "support_areas": areas[owned],
            "support_colors": support_colors[owned],
        })

    transition_meshes = [_make_horizontal_transition(transition, axes) for transition in transitions]
    for index, mesh in enumerate(transition_meshes):
        if len(mesh.faces):
            triangle = np.asarray(mesh.vertices)[np.asarray(mesh.faces)[0]]
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            offset = -float(normal @ triangle[0])
        else:
            normal, offset = np.array([0.0, 1.0, 0.0]), 0.0
        patches.append({
            "name": f"wall_transition_{index + 1:03d}", "mesh": mesh,
            "plane_equation": [*normal, offset],
            "support_centers": np.empty((0, 3)),
            "support_areas": np.empty(0),
            "support_colors": np.empty((0, 3)),
        })

    regularize_planar_patches(patches, neutral_reference_rgb=(0.88, 0.88, 0.86))
    for index, patch in enumerate(patches[:len(planes)]):
        manifest_planes[index]["pattern_classification"] = patch["pattern_classification"]
        manifest_planes[index]["color_split"] = patch["color_split"]
        manifest_planes[index]["regularization"] = patch["regularization"]
    plane_meshes = [patch["mesh"] for patch in patches[:len(planes)]]
    transition_meshes = [patch["mesh"] for patch in patches[len(planes):]]
    base = trimesh.load(source_mesh, force="mesh", process=False)
    combined = trimesh.util.concatenate([base, *plane_meshes, *transition_meshes])
    shell, shell_topology = _weld_shell(
        [*plane_meshes, *transition_meshes], info["floor_y"], info["ceiling_y"]
    )
    output_ply = regions_dir / "wall_regions_assembled.ply"
    output_shell = regions_dir / "wall_regions_assembled_shell.ply"
    output_manifest = regions_dir / "wall_region_assembly_manifest.json"
    combined.export(output_ply)
    shell.export(output_shell)
    assigned_count = int(np.count_nonzero(final_owner >= 0))
    black_mask = np.isin(labels, info["black_region_ids"])
    topology_pruned_mask = ((owner >= 0) & ~np.isin(owner, list(owner_map))
                            & (final_owner < 0))
    trimmed_mask = ((owner >= 0) & np.isin(owner, list(owner_map)) & (final_owner < 0))
    fit_unassigned_mask = (final_owner < 0) & ~black_mask & ~topology_pruned_mask & ~trimmed_mask
    result = {
        "algorithm": "local_support_preserving_carves_with_closed_shell_transitions",
        "input_plane_manifest": str(regions_dir / "wall_region_planes_manifest.json"),
        "input_mesh": str(source_mesh), "triangle_owner_file": "wall_regions_assembled_triangle_owners.npy",
        "triangle_count": int(len(triangles)), "assigned_triangle_count": assigned_count,
        "unassigned_triangle_count": int(len(triangles) - assigned_count),
        "unassigned_triangle_reasons": {
            "black_region": int(black_mask.sum()),
            "plane_removed_for_lack_of_closed_loop": int(topology_pruned_mask.sum()),
            "support_trimmed_outside_closed_panel": int(trimmed_mask.sum()),
            "no_fitted_plane": int(fit_unassigned_mask.sum()),
        },
        "junction_extension_distance_m": JUNCTION_DISTANCE_M,
        "topology_tolerance_m": TOPOLOGY_TOLERANCE_M,
        "parallel_carve_distance_m": PARALLEL_CARVE_DISTANCE_M,
        "floor_ceiling_search_distance_m": FLOOR_CEILING_SEARCH_DISTANCE_M,
        "excluded_small_support_planes": info["excluded_small_support"],
        "excluded_nonvertical_planes": info["excluded_nonvertical"],
        "floor_y": info["floor_y"], "ceiling_y": info["ceiling_y"],
        "topology": topology, "panel_count": len(manifest_planes),
        "connector_count": (sum(p["inferred_connector"] for p in manifest_planes)
                            + len(transitions)),
        "horizontal_transition_count": len(transitions),
        "horizontal_transitions": transitions,
        "shell_topology": shell_topology, "planes": manifest_planes,
        "combined_ply": str(output_ply), "shell_ply": str(output_shell),
    }
    np.save(regions_dir / result["triangle_owner_file"], final_owner)
    output_manifest.write_text(json.dumps(result, indent=2) + "\n")
    return output_ply, output_manifest, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions-dir", type=Path, required=True,
                        help="Directory containing wall-region and fitted-plane artifacts")
    args = parser.parse_args()
    try:
        ply, manifest_path, result = _assemble(args.regions_dir)
    except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Assembled {result['panel_count']} wall panels, including {result['connector_count']} connectors; "
          f"{result['assigned_triangle_count']} / {result['triangle_count']} triangles have one panel owner")
    print(f"Preview PLY: {ply}\nWelded shell PLY: {result['shell_ply']}\nManifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
