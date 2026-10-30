#!/usr/bin/env python3
"""Fit supported Manhattan wall envelopes and enforce horizontal closure."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

DEFAULT_DIR = Path("backend/current_scene/06_mesh_elaboration/extracted_structural_instances")
HORIZONTAL_FACE_TOLERANCE_DEG = 30.0
MIN_WALL_PLANE_SHORT_DIMENSION_M = .20
MIN_WALL_PLANE_LONG_DIMENSION_M = 1.0
MAX_OUTSIDE_WALL_TRIANGLE_FRACTION = .02
# Keep near-minimum supports available until they can be checked as perimeter returns.
MIN_WALL_PLANE_DISCOVERY_DIMENSION_M = .15
MIN_WALL_PLANE_SUPPORT_AREA_M2 = .02
MIN_WALL_PLANE_FINAL_SUPPORT_AREA_M2 = .20
MIN_WALL_PLANE_COVERAGE_RATIO = .10
WALL_PATCH_MERGE_DISTANCE_M = .20
WALL_JUNCTION_SNAP_DISTANCE_M = .35
WALL_COPLANAR_MERGE_DISTANCE_M = .05
MIN_WALL_ENCLOSURE_HEIGHT_M = 1.8
WALL_PLANE_OUTWARD_MARGIN_M = 0.
WALL_PLANE_OUTWARD_COVERAGE = .99
WALL_BOUND_COVERAGE = .99
LOCAL_VERTICAL_SUPPORT_DISTANCE_M = .35
WALL_TOPOLOGY_TOLERANCE_M = .01
WALL_PLANE_COUNT_PENALTY_M = .0005
WALL_DISTANCE_TRUNCATION_M = .10
WALL_FIT_OUTPUTS = (
    "wall_orientation_filtered.ply",
    "wall_cleaned_pieces.ply",
    "wall_enclosing_planes.ply",
    "wall_planes_manifest.json",
    "wall_planes_before_rounding_manifest.json",
    "wall_enclosing_planes_individual",
    "wall_closure_connectors",
)


def _meets_minimum_plane_dimensions(span: list[float], bounds_y: list[float]) -> bool:
    dimensions = sorted((float(span[1] - span[0]), float(bounds_y[1] - bounds_y[0])))
    return (dimensions[0] > MIN_WALL_PLANE_SHORT_DIMENSION_M
            and dimensions[1] > MIN_WALL_PLANE_LONG_DIMENSION_M)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover finite Manhattan wall planes.")
    parser.add_argument("--wall-mesh", type=Path, help="Optional single wall mesh (legacy input).")
    parser.add_argument("--instances-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--plane-dist-thresh", type=float, default=.10)
    parser.add_argument("--filter-only", action="store_true",
                        help="Save the orientation-filtered wall mesh and stop before plane fitting.")
    return parser.parse_args()


def _mesh_paths(args: argparse.Namespace) -> list[Path]:
    if args.wall_mesh:
        return [args.wall_mesh]
    manifest_path = args.instances_dir / "structural_instances_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Wall mesh was not supplied and manifest is missing: {manifest_path}")
    paths = []
    for item in json.loads(manifest_path.read_text()).get("instances", []):
        if str(item.get("role", "")).lower() != "wall":
            continue
        path = Path(item.get("file_path", item.get("file_name", "")))
        paths.append(path if path.is_file() else args.instances_dir / path.name)
    return paths


def _clear_wall_fit_outputs(output_dir: Path, input_paths: list[Path]) -> None:
    outputs = [output_dir / name for name in WALL_FIT_OUTPUTS]
    input_paths = [path.resolve() for path in input_paths]
    conflicts = [output for output in outputs if output.resolve() in input_paths]
    if conflicts:
        raise ValueError(f"Wall-fit output would replace an input mesh: {conflicts[0]}")
    for path in outputs:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def _load_meshes(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices, triangles, colors, base = [], [], [], 0
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Wall mesh not found: {path}")
        mesh = o3d.io.read_triangle_mesh(str(path))
        v, t, c = np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_colors)
        if not len(v) or not len(t):
            continue
        vertices.append(v); triangles.append(t + base)
        colors.append(c if len(c) == len(v) else np.full((len(v), 3), .8))
        base += len(v)
    if not triangles:
        raise ValueError("Wall input contains no triangles")
    return np.vstack(vertices), np.vstack(triangles), np.vstack(colors)


def _remove_horizontal_faces(
    vertices: np.ndarray, triangles: np.ndarray, tolerance_deg: float = HORIZONTAL_FACE_TOLERANCE_DEG
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    corners = vertices[triangles]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1e-10
    normals = np.zeros_like(cross)
    normals[valid] = cross[valid] / double_area[valid, None]
    angle_from_horizontal = np.arctan2(
        np.linalg.norm(normals[:, [0, 2]], axis=1), np.abs(normals[:, 1])
    )
    near_horizontal = valid & (angle_from_horizontal <= np.deg2rad(tolerance_deg) + 1e-12)
    keep = valid & ~near_horizontal
    return (
        triangles[keep],
        normals[keep],
        corners[keep].mean(axis=1),
        double_area[keep] / 2,
        {
            "total_triangles": int(len(triangles)),
            "degenerate_triangles_removed": int((~valid).sum()),
            "horizontal_triangles_removed": int(near_horizontal.sum()),
            "horizontal_area_removed_m2": float(double_area[near_horizontal].sum() / 2),
        },
    )


def _basis(normals: np.ndarray, areas: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    horizontal = normals[:, [0, 2]]
    lengths = np.linalg.norm(horizontal, axis=1)
    keep = lengths > .85
    if not np.any(keep):
        raise ValueError("No sufficiently vertical wall faces")
    angles = np.mod(np.arctan2(horizontal[keep, 1], horizontal[keep, 0]), np.pi / 2)
    edges = np.linspace(0., np.pi / 2, 181)
    hist, _ = np.histogram(angles, bins=edges, weights=areas[keep])
    peak = (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2
    # Axial mean refines the area-weighted histogram peak without random sampling.
    near = np.abs(np.angle(np.exp(4j * (angles - peak)))) <= np.deg2rad(12.)
    theta = float((.25 * np.angle(np.sum(areas[keep][near] * np.exp(4j * angles[near])))) % (np.pi / 2))
    return np.array([np.cos(theta), 0., np.sin(theta)]), np.array([-np.sin(theta), 0., np.cos(theta)]), theta


def _groups(values: np.ndarray, tolerance: float) -> list[np.ndarray]:
    order = np.argsort(values)
    out, group = [], [int(order[0])]
    for index in order[1:]:
        if values[index] - values[group[0]] > tolerance:
            out.append(np.asarray(group, dtype=int)); group = [int(index)]
        else:
            group.append(int(index))
    out.append(np.asarray(group, dtype=int))
    return out


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    index = min(
        int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left")), len(order) - 1
    )
    return float(values[order[index]])


def _supported_bounds(vertices: np.ndarray, triangles: np.ndarray, areas: np.ndarray,
                      faces: np.ndarray, tangent: np.ndarray) -> tuple[list[float], list[float]]:
    """Estimate in-plane span and height from area-weighted supported vertices."""
    points = vertices[triangles[faces].reshape(-1)]
    weights = np.repeat(areas[faces] / 3., 3)
    tail = (1. - WALL_BOUND_COVERAGE) / 2.
    span_values = points @ tangent
    return (
        [_weighted_quantile(span_values, weights, tail),
         _weighted_quantile(span_values, weights, 1. - tail)],
        [_weighted_quantile(points[:, 1], weights, tail),
         _weighted_quantile(points[:, 1], weights, 1. - tail)],
    )


def _outward_plane_offset(projections: np.ndarray, side: float) -> float:
    values = np.sort(side * projections)
    if not len(values):
        raise ValueError("Cannot fit a wall plane without support triangles")
    index = min(math.ceil(WALL_PLANE_OUTWARD_COVERAGE * len(values)) - 1, len(values) - 1)
    outward_projection = float(values[index])
    return float(side * (outward_projection + WALL_PLANE_OUTWARD_MARGIN_M))


def _outward_triangle_projections(
    vertices: np.ndarray, triangles: np.ndarray, faces: np.ndarray,
    normal_axis: np.ndarray, side: float,
) -> np.ndarray:
    projections = vertices[triangles[faces]] @ normal_axis
    return projections.max(axis=1) if side > 0 else projections.min(axis=1)


def _triangle_plane_distances(
    vertices: np.ndarray, triangles: np.ndarray, faces: np.ndarray,
    normal_axis: np.ndarray, offset: float,
) -> np.ndarray:
    """Shortest orthogonal distance from each supported triangle to the plane."""
    projections = vertices[triangles[faces]] @ normal_axis
    return np.maximum(0., np.maximum(projections.min(axis=1) - offset,
                                     offset - projections.max(axis=1)))


def _refine_basis_from_support(vertices: np.ndarray, triangles: np.ndarray, normals: np.ndarray,
                               centers: np.ndarray, areas: np.ndarray, a: np.ndarray, b: np.ndarray,
                               theta: float, offset_tol: float) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    """Refit the shared wall angle from large geometric supports, not noisy face normals."""
    candidates, _ = _discover(vertices, triangles, normals, centers, areas, a, b, offset_tol)
    angles, weights = [], []
    o3d.utility.random.seed(17)
    for plane in candidates:
        span = plane.get("span_w_m", plane.get("span_u_m"))
        height = plane["bounds_y"][1] - plane["bounds_y"][0]
        if plane["support_area_m2"] < .5 or span[1] - span[0] < 1.0 or height < 1.5:
            continue
        points = centers[plane["source_faces"]]
        if len(points) > 20000:
            points = points[np.linspace(0, len(points) - 1, 20000, dtype=int)]
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        try:
            equation, inliers = cloud.segment_plane(.03, 3, 300)
        except RuntimeError:
            continue
        if len(inliers) < 300 or len(inliers) / len(points) < .70:
            continue
        normal = np.asarray(equation[:3], dtype=float)
        normal /= np.linalg.norm(normal)
        if abs(normal[1]) > .2:
            continue
        angles.append(float(np.arctan2(normal[2], normal[0]) % (np.pi / 2)))
        weights.append(min(plane["support_area_m2"], 2.0))
    if len(angles) < 3:
        return a, b, theta, {"method": "area-weighted face normals", "support_plane_count": len(angles)}
    angles, weights = np.asarray(angles), np.asarray(weights)
    center = float((.25 * np.angle(np.sum(weights * np.exp(4j * angles)))) % (np.pi / 2))
    residual = np.abs(np.angle(np.exp(4j * (angles - center)))) / 4
    keep = residual <= np.deg2rad(4.)
    refined = float((.25 * np.angle(np.sum(weights[keep] * np.exp(4j * angles[keep])))) % (np.pi / 2))
    if keep.sum() < 3 or abs(np.angle(np.exp(4j * (refined - theta)))) / 4 > np.deg2rad(5.):
        return a, b, theta, {"method": "area-weighted face normals", "support_plane_count": len(angles)}
    return (np.array([np.cos(refined), 0., np.sin(refined)]),
            np.array([-np.sin(refined), 0., np.cos(refined)]), refined,
            {"method": "RANSAC fit of large wall patches", "support_plane_count": int(keep.sum()),
             "initial_angle_deg": round(float(np.degrees(theta)), 4),
             "fitted_angle_deg": round(float(np.degrees(refined)), 4)})


def _discover(vertices: np.ndarray, triangles: np.ndarray, normals: np.ndarray, centers: np.ndarray,
              areas: np.ndarray, a: np.ndarray, b: np.ndarray,
              offset_tol: float) -> tuple[list[dict[str, Any]], np.ndarray]:
    planes, inlier = [], np.zeros(len(triangles), dtype=bool)
    alignments = np.column_stack((normals @ a, normals @ b))
    best_axis = np.argmax(np.abs(alignments), axis=1)
    for axis_index, (axis, normal, tangent, span_key) in enumerate(
        (("Dir A", a, b, "span_w_m"), ("Dir B", b, a, "span_u_m"))
    ):
        alignment = alignments[:, axis_index]
        for sign in (-1., 1.):
            candidates = np.flatnonzero((best_axis == axis_index) & (alignment * sign >= .85))
            if not len(candidates):
                continue
            for local in _groups(centers[candidates] @ normal, offset_tol):
                faces = candidates[local]
                unique, remap = np.unique(triangles[faces], return_inverse=True)
                component_mesh = o3d.geometry.TriangleMesh()
                component_mesh.vertices = o3d.utility.Vector3dVector(vertices[unique])
                component_mesh.triangles = o3d.utility.Vector3iVector(remap.reshape(-1, 3))
                labels, _, component_areas = component_mesh.cluster_connected_triangles()
                labels, component_areas = np.asarray(labels), np.asarray(component_areas)
                for component in np.flatnonzero(component_areas >= MIN_WALL_PLANE_SUPPORT_AREA_M2):
                    support = faces[labels == component]
                    span, height = _supported_bounds(vertices, triangles, areas, support, tangent)
                    lo, hi = span
                    y0, y1 = height
                    if min(hi - lo, y1 - y0) < MIN_WALL_PLANE_DISCOVERY_DIMENSION_M - 1e-9:
                        continue
                    plane_area = float((hi - lo) * (y1 - y0))
                    projected_support = float(np.sum(areas[support] * np.abs(normals[support] @ normal)))
                    coverage = min(1., projected_support / plane_area)
                    if coverage < MIN_WALL_PLANE_COVERAGE_RATIO:
                        continue
                    offset = float(np.average(centers[support] @ normal, weights=areas[support]))
                    planes.append({
                        "axis": axis,
                        "offset_m": offset,
                        "max_offset_residual_m": float(np.max(np.abs(centers[support] @ normal - offset))),
                        span_key: [float(lo), float(hi)],
                        "normal_unit": sign * normal,
                        "bounds_y": [float(y0), float(y1)],
                        "support_area_m2": float(areas[support].sum()),
                        "enclosing_area_m2": plane_area,
                        "support_coverage_ratio": coverage,
                        "source_faces": support,
                    })
                    inlier[support] = True
    planes.sort(key=lambda p: (p["axis"], p["offset_m"], p.get("span_u_m", p.get("span_w_m"))[0]))
    for index, plane in enumerate(planes, 1):
        plane["id"], plane["name"] = index, f"wall_plane_{index:03d}"
    return planes, inlier


def _merge_nearby_planes(planes: list[dict[str, Any]], distance: float,
                         center: np.ndarray, a: np.ndarray, b: np.ndarray) -> list[dict[str, Any]]:
    clusters = [[plane] for plane in planes]

    def summary(cluster: list[dict[str, Any]]) -> tuple[int, float, float, float, float, float]:
        weights = np.asarray([p["support_area_m2"] for p in cluster])
        offsets = [p["offset_m"] for p in cluster]
        axis_vector = a if cluster[0]["axis"] == "Dir A" else b
        center_offset = float(center[0] * axis_vector[0] + center[1] * axis_vector[2])
        side = 1 if offsets[0] >= center_offset else -1
        offset = max(offsets) if side > 0 else min(offsets)
        spans = [p.get("span_w_m", p.get("span_u_m")) for p in cluster]
        return (side, offset, min(s[0] for s in spans), max(s[1] for s in spans),
                float(weights.sum()), max(offsets) - min(offsets))

    while True:
        best_key: tuple[float, int, int] | None = None
        best_pair: tuple[int, int] | None = None
        for i, left in enumerate(clusters):
            for j, right in enumerate(clusters[:i]):
                if left[0]["axis"] != right[0]["axis"]:
                    continue
                l_side, l_off, l_lo, l_hi, _, _ = summary(left)
                r_side, r_off, r_lo, r_hi, _, _ = summary(right)
                if l_side != r_side:
                    continue
                along_gap = max(0., max(l_lo, r_lo) - min(l_hi, r_hi))
                left_y = [min(p["bounds_y"][0] for p in left), max(p["bounds_y"][1] for p in left)]
                right_y = [min(p["bounds_y"][0] for p in right), max(p["bounds_y"][1] for p in right)]
                height_gap = max(0., max(left_y[0], right_y[0]) - min(left_y[1], right_y[1]))
                cost = math.hypot(abs(l_off - r_off), along_gap)
                combined = left + right
                _, _, _, _, _, offset_spread = summary(combined)
                if (cost > distance or height_gap > distance
                        or offset_spread > WALL_COPLANAR_MERGE_DISTANCE_M):
                    continue
                key = (cost, min(p["id"] for p in combined), max(p["id"] for p in combined))
                if best_key is None or key < best_key:
                    best_key, best_pair = key, (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]

    merged = []
    for cluster in clusters:
        side, offset, lo, hi, support_area, offset_spread = summary(cluster)
        source_faces = np.unique(np.concatenate([p["source_faces"] for p in cluster]))
        y0 = min(p["bounds_y"][0] for p in cluster)
        y1 = max(p["bounds_y"][1] for p in cluster)
        span_key = "span_w_m" if cluster[0]["axis"] == "Dir A" else "span_u_m"
        normal_axis = a if cluster[0]["axis"] == "Dir A" else b
        source_height = y1 - y0
        merged.append({
            "axis": cluster[0]["axis"],
            "offset_m": offset,
            "max_offset_residual_m": max(offset_spread, max(p["max_offset_residual_m"] for p in cluster)),
            span_key: [lo, hi],
            "normal_unit": side * normal_axis,
            "bounds_y": [y0, y1],
            "support_area_m2": support_area,
            "enclosing_area_m2": (hi - lo) * source_height,
            "support_coverage_ratio": min(1., support_area / max((hi - lo) * source_height, 1e-12)),
            "source_faces": source_faces,
            "source_plane_ids": sorted(p["id"] for p in cluster),
        })
    merged.sort(key=lambda p: (p["axis"], p["offset_m"], p.get("span_u_m", p.get("span_w_m"))[0]))
    for index, plane in enumerate(merged, 1):
        plane["id"], plane["name"] = index, f"wall_plane_{index:03d}"
    return merged


def _retain_supported_planes(planes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept = []
    for plane in planes:
        key = "span_w_m" if plane["axis"] == "Dir A" else "span_u_m"
        span = plane[key]
        if (plane["support_area_m2"] < MIN_WALL_PLANE_FINAL_SUPPORT_AREA_M2
                or not _meets_minimum_plane_dimensions(span, plane["bounds_y"])):
            continue
        kept.append(plane)
    kept.sort(key=lambda p: (p["axis"], p["offset_m"], p.get("span_u_m", p.get("span_w_m"))[0]))
    for index, plane in enumerate(kept, 1):
        plane["id"], plane["name"] = index, f"wall_plane_{index:03d}"
    return kept


def _snap_plane_corners(planes: list[dict[str, Any]], distance: float) -> None:
    """Close nearby perpendicular wall corners, including vertical panel bounds."""
    spans = {id(p): list(p.get("span_w_m", p.get("span_u_m"))) for p in planes}
    bounds = {id(p): list(p["bounds_y"]) for p in planes}
    directions_a = [p for p in planes if p["axis"] == "Dir A"]
    directions_b = [p for p in planes if p["axis"] == "Dir B"]
    for plane_a in directions_a:
        for plane_b in directions_b:
            a_span, b_span = spans[id(plane_a)], spans[id(plane_b)]
            intersection_a, intersection_b = plane_a["offset_m"], plane_b["offset_m"]
            dist_a = max(a_span[0] - intersection_b, 0., intersection_b - a_span[1])
            dist_b = max(b_span[0] - intersection_a, 0., intersection_a - b_span[1])
            a_bounds, b_bounds = bounds[id(plane_a)], bounds[id(plane_b)]
            y_overlap = min(a_bounds[1], b_bounds[1]) - max(a_bounds[0], b_bounds[0])
            if y_overlap >= -distance and max(dist_a, dist_b) <= distance:
                a_span[0] = min(a_span[0], intersection_b)
                a_span[1] = max(a_span[1], intersection_b)
                b_span[0] = min(b_span[0], intersection_a)
                b_span[1] = max(b_span[1], intersection_a)

    changed = True
    while changed:
        changed = False
        for plane_a in directions_a:
            for plane_b in directions_b:
                a_span, b_span = spans[id(plane_a)], spans[id(plane_b)]
                intersection_a, intersection_b = plane_a["offset_m"], plane_b["offset_m"]
                dist_a = max(a_span[0] - intersection_b, 0., intersection_b - a_span[1])
                dist_b = max(b_span[0] - intersection_a, 0., intersection_a - b_span[1])
                a_bounds, b_bounds = bounds[id(plane_a)], bounds[id(plane_b)]
                if max(dist_a, dist_b) > distance:
                    continue
                if min(a_bounds[1], b_bounds[1]) - max(a_bounds[0], b_bounds[0]) < -distance:
                    continue
                joined_bounds = [min(a_bounds[0], b_bounds[0]), max(a_bounds[1], b_bounds[1])]
                if joined_bounds != a_bounds or joined_bounds != b_bounds:
                    changed = True
                    bounds[id(plane_a)] = joined_bounds.copy()
                    bounds[id(plane_b)] = joined_bounds.copy()
    for plane in planes:
        key = "span_w_m" if plane["axis"] == "Dir A" else "span_u_m"
        plane[key] = spans[id(plane)]
        plane["bounds_y"] = bounds[id(plane)]
        plane["plane_height_m"] = bounds[id(plane)][1] - bounds[id(plane)][0]


def _subdivide_supported_planes(
    planes: list[dict[str, Any]], vertices: np.ndarray, triangles: np.ndarray,
    centers: np.ndarray, areas: np.ndarray, a: np.ndarray, b: np.ndarray,
) -> list[dict[str, Any]]:
    """Split a supported wall run once where local planes materially reduce fit error."""
    fitted = []
    connectors = []
    for plane in planes:
        split_group = plane["id"]
        if plane.get("inferred_connector") or not len(plane["source_faces"]):
            unchanged = dict(plane)
            unchanged["_split_group"] = split_group
            fitted.append(unchanged)
            continue
        axis_a = plane["axis"] == "Dir A"
        normal_axis, tangent = (a, b) if axis_a else (b, a)
        side = 1. if np.dot(plane["normal_unit"], normal_axis) >= 0. else -1.
        span_key = "span_w_m" if axis_a else "span_u_m"
        span = list(plane[span_key])
        support = np.asarray(plane["source_faces"], dtype=int)
        coords = centers[support] @ tangent
        best = None
        for quantile in np.arange(.1, 1., .1):
            cut = float(np.quantile(coords, quantile))
            left = support[coords < cut]
            right = support[coords >= cut]
            cut = (float(np.max(centers[left] @ tangent))
                   + float(np.min(centers[right] @ tangent))) / 2 if len(left) and len(right) else cut
            left_span, right_span = [span[0], cut], [cut, span[1]]
            if (len(left) < 2 or len(right) < 2
                    or areas[left].sum() < MIN_WALL_PLANE_FINAL_SUPPORT_AREA_M2
                    or areas[right].sum() < MIN_WALL_PLANE_FINAL_SUPPORT_AREA_M2
                    or not _meets_minimum_plane_dimensions(left_span, plane["bounds_y"])
                    or not _meets_minimum_plane_dimensions(right_span, plane["bounds_y"])):
                continue
            offsets = []
            child_loss = 0.
            for faces in (left, right):
                projections = _outward_triangle_projections(
                    vertices, triangles, faces, normal_axis, side
                )
                offset = _outward_plane_offset(projections, side)
                offsets.append(offset)
                distances = _triangle_plane_distances(
                    vertices, triangles, faces, normal_axis, offset
                )
                child_loss += float(np.sum(distances * areas[faces]))
            if abs(offsets[0] - offsets[1]) <= MIN_WALL_PLANE_SHORT_DIMENSION_M:
                continue
            if not _meets_minimum_plane_dimensions(
                [min(offsets), max(offsets)], plane["bounds_y"]
            ):
                continue
            if best is None or child_loss < best[0]:
                best = (child_loss, cut, left, right, offsets)
        if best is None:
            unchanged = dict(plane)
            unchanged["_split_group"] = split_group
            fitted.append(unchanged)
            continue

        _, cut, left, right, offsets = best
        for faces, child_span, offset in zip(
            (left, right), ([span[0], cut], [cut, span[1]]), offsets
        ):
            child = dict(plane)
            child["source_faces"] = faces
            child[span_key] = child_span
            child["offset_m"] = float(offset)
            child["support_plane_offset_m"] = _weighted_quantile(
                centers[faces] @ normal_axis, areas[faces], .5
            )
            child["source_span_m"], child["source_bounds_y"] = _supported_bounds(
                vertices, triangles, areas, faces, tangent
            )
            child["support_area_m2"] = float(areas[faces].sum())
            child["max_offset_residual_m"] = float(np.max(
                _triangle_plane_distances(vertices, triangles, faces, normal_axis, offset)
            ))
            child["_split_group"] = split_group
            fitted.append(child)

        gap = abs(offsets[1] - offsets[0])
        if gap > MIN_WALL_PLANE_SHORT_DIMENSION_M:
            connector_axis = "Dir B" if axis_a else "Dir A"
            connector_normal = tangent * (side if offsets[1] > offsets[0] else -side)
            connector_span_key = "span_u_m" if axis_a else "span_w_m"
            connector_span = [min(offsets), max(offsets)]
            height = list(plane["bounds_y"])
            connectors.append({
                "axis": connector_axis,
                "offset_m": cut,
                "normal_unit": connector_normal,
                connector_span_key: connector_span,
                "bounds_y": height,
                "source_span_m": list(connector_span),
                "source_bounds_y": height.copy(),
                "support_plane_offset_m": None,
                "support_area_m2": 0.,
                "enclosing_area_m2": gap * (height[1] - height[0]),
                "support_coverage_ratio": 0.,
                "source_faces": np.empty(0, dtype=int),
                "plane_height_m": height[1] - height[0],
                "max_offset_residual_m": 0.,
                "vertical_bounds_source": "inferred subdivision connector",
                "inferred_connector": True,
                "_split_group": split_group,
            })

    fitted.extend(connectors)
    fitted.sort(key=lambda p: (p["axis"], p["offset_m"],
                               p.get("span_u_m", p.get("span_w_m"))[0]))
    for index, plane in enumerate(fitted, 1):
        plane["id"], plane["name"] = index, f"wall_plane_{index:03d}"
    return fitted


def _rect(plane: dict[str, Any]) -> tuple[float, float, float, float]:
    span = plane.get("span_w_m", plane.get("span_u_m"))
    return float(span[0]), float(span[1]), float(plane["bounds_y"][0]), float(plane["bounds_y"][1])


def _plane_regions(plane: dict[str, Any]) -> list[dict[str, list[float]]]:
    if plane.get("trim_regions"):
        return plane["trim_regions"]
    span = plane.get("span_w_m", plane.get("span_u_m"))
    return [{"span_m": list(span), "bounds_y": list(plane["bounds_y"])}]


def _plane_quad_geometry(
    plane: dict[str, Any], a: np.ndarray, b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tangent, normal_axis = (b, a) if plane["axis"] == "Dir A" else (a, b)
    vertices, triangles = [], []
    for region in _plane_regions(plane):
        lo, hi = region["span_m"]
        y0, y1 = region["bounds_y"]
        base = plane["offset_m"] * normal_axis
        p0, p1 = base + lo * tangent, base + hi * tangent
        start = len(vertices)
        vertices.extend(([p0[0], y0, p0[2]], [p1[0], y0, p1[2]],
                         [p1[0], y1, p1[2]], [p0[0], y1, p0[2]]))
        cell_triangles = [[0, 1, 2], [0, 2, 3]]
        if np.dot(np.cross(p1 - p0, np.array([0., y1 - y0, 0.])), plane["normal_unit"]) < 0.:
            cell_triangles = [[0, 2, 1], [0, 3, 2]]
        triangles.extend([[start + index for index in tri] for tri in cell_triangles])
    return np.asarray(vertices), np.asarray(triangles)


def _rect_area(rect: tuple[float, float, float, float]) -> float:
    return max(0., rect[1] - rect[0]) * max(0., rect[3] - rect[2])


def _rect_intersection_area(first: tuple[float, float, float, float],
                            second: tuple[float, float, float, float]) -> float:
    return max(0., min(first[1], second[1]) - max(first[0], second[0])) * max(
        0., min(first[3], second[3]) - max(first[2], second[2])
    )


def _triangles_intersect_rect(
    projected: np.ndarray, rect: tuple[float, float, float, float],
) -> np.ndarray:
    """Test projected triangles against a panel rectangle using the separating-axis test."""
    x0, x1, y0, y1 = rect
    rect_min, rect_max = np.array([x0, y0]), np.array([x1, y1])
    intersects = np.all(projected.max(axis=1) >= rect_min, axis=1)
    intersects &= np.all(projected.min(axis=1) <= rect_max, axis=1)
    edges = np.roll(projected, -1, axis=1) - projected
    axes = np.stack((-edges[..., 1], edges[..., 0]), axis=-1)
    projections = np.einsum("nvd,nkd->nkv", projected, axes)
    rect_center = np.array([(x0 + x1) * .5, (y0 + y1) * .5])
    rect_half = np.array([(x1 - x0) * .5, (y1 - y0) * .5])
    center_projection = np.einsum("nkd,d->nk", axes, rect_center)
    radius = np.einsum("nkd,d->nk", np.abs(axes), rect_half)
    separated = ((projections.max(axis=2) < center_projection - radius)
                 | (projections.min(axis=2) > center_projection + radius))
    return intersects & ~np.any(separated, axis=1)


def _assign_support_owners(
    planes: list[dict[str, Any]], vertices: np.ndarray, triangles: np.ndarray,
    a: np.ndarray, b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign projected triangles to one closest panel and retain its distance."""
    owner = np.full(len(triangles), -1, dtype=int)
    closest = np.full(len(triangles), np.inf)
    corners = vertices[triangles]
    for plane in planes:
        if plane.get("inferred_connector"):
            continue
        normal_axis, tangent = (a, b) if plane["axis"] == "Dir A" else (b, a)
        projected = np.stack((corners @ tangent, corners[:, :, 1]), axis=-1)
        for region in _plane_regions(plane):
            span, bounds_y = region["span_m"], region["bounds_y"]
            rect = (span[0], span[1], bounds_y[0], bounds_y[1])
            candidates = np.flatnonzero(
                (projected[:, :, 0].max(axis=1) >= rect[0])
                & (projected[:, :, 0].min(axis=1) <= rect[1])
                & (projected[:, :, 1].max(axis=1) >= rect[2])
                & (projected[:, :, 1].min(axis=1) <= rect[3])
            )
            hits = candidates[_triangles_intersect_rect(projected[candidates], rect)]
            distances = _triangle_plane_distances(
                vertices, triangles, hits, normal_axis, plane["offset_m"]
            )
            closer = distances < closest[hits]
            owner[hits[closer]] = plane["id"]
            closest[hits[closer]] = distances[closer]

    return owner, closest


def _merge_overlapping_parallel_planes(
    planes: list[dict[str, Any]], vertices: np.ndarray, triangles: np.ndarray,
    areas: np.ndarray, a: np.ndarray, b: np.ndarray,
) -> list[dict[str, Any]]:
    """Collapse overlapping, same-facing supports into one outward wall envelope."""
    clusters = [[plane] for plane in planes]
    while True:
        pair = None
        for i, first in enumerate(clusters):
            for j, second in enumerate(clusters[:i]):
                left, right = first[0], second[0]
                if left["axis"] != right["axis"] or np.dot(left["normal_unit"], right["normal_unit"]) < .99:
                    continue
                if _rect_intersection_area(_rect(left), _rect(right)) <= 1e-9:
                    continue
                pair = (i, j)
                break
            if pair:
                break
        if pair is None:
            break
        i, j = pair
        first, second = clusters[i][0], clusters[j][0]
        support_faces = np.unique(np.concatenate([plane["source_faces"] for plane in clusters[i] + clusters[j]]))
        span_key = "span_w_m" if first["axis"] == "Dir A" else "span_u_m"
        normal_axis = a if first["axis"] == "Dir A" else b
        tangent = b if first["axis"] == "Dir A" else a
        side = 1. if np.dot(first["normal_unit"], normal_axis) >= 0. else -1.
        span, supported_height = _supported_bounds(
            vertices, triangles, areas, support_faces, tangent
        )
        projections = _outward_triangle_projections(
            vertices, triangles, support_faces, normal_axis, side
        )
        # Keep the exterior edge of supported mesh so the fitted plane encloses the wall.
        offset = _outward_plane_offset(projections, side)
        y0 = min(plane["bounds_y"][0] for plane in clusters[i] + clusters[j])
        y1 = max(plane["bounds_y"][1] for plane in clusters[i] + clusters[j])
        merged = {**first,
                  "offset_m": offset,
                  "normal_unit": side * normal_axis,
                  span_key: span,
                  "source_span_m": list(span),
                  "bounds_y": [float(y0), float(y1)],
                  "source_bounds_y": supported_height,
                  "vertical_bounds_source": "merged supported panels",
                  "source_faces": support_faces,
                  "source_plane_ids": sorted({source_id for plane in clusters[i] + clusters[j]
                                               for source_id in plane.get("source_plane_ids", [plane["id"]])}),
                  "support_area_m2": float(areas[support_faces].sum()),
                  "plane_height_m": float(y1 - y0),
                  "max_offset_residual_m": float(np.max(_triangle_plane_distances(
                      vertices, triangles, support_faces, normal_axis, offset
                  )))}
        merged["enclosing_area_m2"] = _rect_area(_rect(merged))
        merged["support_coverage_ratio"] = min(
            1., merged["support_area_m2"] / max(merged["enclosing_area_m2"], 1e-12)
        )
        clusters[i] = [merged]
        del clusters[j]
    result = [cluster[0] for cluster in clusters]
    result.sort(key=lambda p: (p["axis"], p["offset_m"], p.get("span_u_m", p.get("span_w_m"))[0]))
    for index, plane in enumerate(result, 1):
        plane["id"], plane["name"] = index, f"wall_plane_{index:03d}"
    return result


def _validate_parallel_overlaps(planes: list[dict[str, Any]]) -> None:
    overlaps = []
    for index, first in enumerate(planes):
        for second in planes[:index]:
            if first.get("inferred_connector") or second.get("inferred_connector"):
                # A seam connector may project over a distant wall while occupying another offset.
                continue
            if (first["axis"] == second["axis"]
                    and np.dot(first["normal_unit"], second["normal_unit"]) >= .99
                    and _rect_intersection_area(_rect(first), _rect(second)) > 1e-6):
                overlaps.append((second["name"], first["name"]))
    if overlaps:
        pairs = ", ".join(f"{left}/{right}" for left, right in overlaps[:5])
        raise ValueError(f"Overlapping same-facing wall planes remain: {pairs}")


def _validate_unique_support(planes: list[dict[str, Any]]) -> None:
    owners: dict[int, str] = {}
    for plane in planes:
        for face in plane["source_faces"]:
            face = int(face)
            if face in owners:
                raise ValueError(
                    f"Wall triangle {face} is assigned to both {owners[face]} and {plane['name']}"
                )
            owners[face] = plane["name"]


def _horizontal_topology(
    planes: list[dict[str, Any]], tolerance: float,
) -> dict[str, Any]:
    """Require each horizontal slice to be one connected network of closed loops."""
    events = sorted({
        float(value)
        for plane in planes
        for region in _plane_regions(plane)
        for value in region["bounds_y"]
    })
    if len(events) < 2:
        raise ValueError("Wall planes have no measurable vertical extent")

    band_count = 0
    minimum_cycle_rank: int | None = None
    max_component_count = 0
    for y0, y1 in zip(events, events[1:]):
        if y1 - y0 <= 1e-8:
            continue
        y = (y0 + y1) / 2
        segments = []
        for plane in planes:
            for region in _plane_regions(plane):
                if region["bounds_y"][0] - tolerance <= y <= region["bounds_y"][1] + tolerance:
                    segments.append({"plane": plane, "span": region["span_m"], "crossings": []})
        if not segments:
            raise ValueError(f"No fitted wall planes cover horizontal slice y={y:.3f} m")

        dir_a = [segment for segment in segments if segment["plane"]["axis"] == "Dir A"]
        dir_b = [segment for segment in segments if segment["plane"]["axis"] == "Dir B"]
        for first in dir_a:
            for second in dir_b:
                first_span, second_span = first["span"], second["span"]
                x, z = float(first["plane"]["offset_m"]), float(second["plane"]["offset_m"])
                if (first_span[0] - tolerance <= z <= first_span[1] + tolerance
                        and second_span[0] - tolerance <= x <= second_span[1] + tolerance):
                    node = (round(x, 4), round(z, 4))
                    first["crossings"].append((z, node))
                    second["crossings"].append((x, node))

        graph: dict[tuple[float, float], set[tuple[float, float]]] = {}
        open_endpoints = []
        for segment in segments:
            lo, hi = map(float, segment["span"])
            crossings = sorted(set(segment["crossings"]))
            coordinates = [value for value, _ in crossings]
            nodes = [node for _, node in crossings]
            name = segment["plane"].get("name", "unnamed panel")
            if not coordinates:
                open_endpoints.extend((name, name))
                continue
            if min(abs(lo - value) for value in coordinates) > tolerance:
                open_endpoints.append(name)
            if min(abs(hi - value) for value in coordinates) > tolerance:
                open_endpoints.append(name)
            for first, second in zip(nodes, nodes[1:]):
                if first == second:
                    continue
                graph.setdefault(first, set()).add(second)
                graph.setdefault(second, set()).add(first)

        if open_endpoints:
            names = ", ".join(sorted(set(open_endpoints))[:5])
            message = (
                f"Wall planes are not horizontally closed at y={y:.3f} m: "
                f"{len(open_endpoints)} open panel endpoints ({names})"
            )
            raise ValueError(message)
        if not graph:
            message = f"Wall planes do not form a closed horizontal loop at y={y:.3f} m"
            raise ValueError(message)

        components, remaining = [], set(graph)
        while remaining:
            component, stack = set(), [remaining.pop()]
            while stack:
                node = stack.pop()
                if node in component:
                    continue
                component.add(node)
                stack.extend(graph[node] - component)
            remaining -= component
            components.append(component)
        max_component_count = max(max_component_count, len(components))
        if len(components) != 1:
            message = (
                f"Wall planes form {len(components)} disconnected horizontal components "
                f"at y={y:.3f} m"
            )
            raise ValueError(message)
        edges = sum(len(neighbors) for neighbors in graph.values()) // 2
        cycle_rank = edges - len(graph) + len(components)
        if cycle_rank < 1:
            message = f"Wall planes do not enclose a horizontal loop at y={y:.3f} m"
            raise ValueError(message)
        band_count += 1
        minimum_cycle_rank = cycle_rank if minimum_cycle_rank is None else min(minimum_cycle_rank, cycle_rank)

    if not band_count:
        raise ValueError("Wall planes do not cover any horizontal height band")
    return {
        "height_band_count": band_count,
        "connected_component_count": max_component_count,
        "minimum_cycle_rank": int(minimum_cycle_rank or 0),
        "open_endpoint_count": 0,
        "is_closed": True,
        "issues": [],
    }


def _outside_panel_triangles(
    planes: list[dict[str, Any]], vertices: np.ndarray, triangles: np.ndarray,
    a: np.ndarray, b: np.ndarray,
) -> np.ndarray:
    """Return triangles not enclosed by any intersected finite wall panel."""
    closest = np.full(len(triangles), np.inf)
    outside_owner = np.ones(len(triangles), dtype=bool)
    triangle_corners = vertices[triangles]
    for plane in planes:
        if plane.get("inferred_connector"):
            continue
        normal_axis, tangent = (a, b) if plane["axis"] == "Dir A" else (b, a)
        side = 1. if np.dot(plane["normal_unit"], normal_axis) >= 0. else -1.
        outside_plane = np.max(
            side * (triangle_corners @ normal_axis - plane["offset_m"]), axis=1
        ) > 1e-8
        projected_all = np.stack((triangle_corners @ tangent,
                                  triangle_corners[:, :, 1]), axis=-1)
        for region in _plane_regions(plane):
            span, bounds_y = region["span_m"], region["bounds_y"]
            rect = (span[0], span[1], bounds_y[0], bounds_y[1])
            candidates = np.flatnonzero(
                (projected_all[:, :, 0].max(axis=1) >= rect[0])
                & (projected_all[:, :, 0].min(axis=1) <= rect[1])
                & (projected_all[:, :, 1].max(axis=1) >= rect[2])
                & (projected_all[:, :, 1].min(axis=1) <= rect[3])
            )
            hits = candidates[_triangles_intersect_rect(projected_all[candidates], rect)]
            distances = _triangle_plane_distances(
                vertices, triangles, hits, normal_axis, plane["offset_m"]
            )
            closer = distances < closest[hits]
            owners = hits[closer]
            closest[owners] = distances[closer]
            outside_owner[owners] = outside_plane[owners]
    return outside_owner


def _trim_split_corner_overshoots(
    planes: list[dict[str, Any]], distance: float,
) -> None:
    """Trim old panel ends to nearby crossings shifted by a local plane split."""
    for plane in planes:
        span_key = "span_w_m" if plane["axis"] == "Dir A" else "span_u_m"
        span = list(plane[span_key])
        crossings = []
        for other in planes:
            if other["axis"] == plane["axis"]:
                continue
            if min(plane["bounds_y"][1], other["bounds_y"][1]) < max(
                plane["bounds_y"][0], other["bounds_y"][0]
            ) - WALL_TOPOLOGY_TOLERANCE_M:
                continue
            coordinate = other["offset_m"]
            other_span = other.get("span_w_m", other.get("span_u_m"))
            if (span[0] - distance <= coordinate <= span[1] + distance
                    and other_span[0] - distance <= plane["offset_m"] <= other_span[1] + distance):
                crossings.append(float(coordinate))
        if len(crossings) < 2:
            continue
        for side in (0, 1):
            endpoint = span[side]
            crossing = min(crossings, key=lambda value: abs(value - endpoint))
            if abs(crossing - endpoint) <= WALL_TOPOLOGY_TOLERANCE_M:
                continue
            if abs(crossing - endpoint) <= distance:
                span[side] = crossing
        plane[span_key] = span


def _layout_objective(
    planes: list[dict[str, Any]], vertices: np.ndarray, triangles: np.ndarray,
    areas: np.ndarray, a: np.ndarray, b: np.ndarray,
) -> float:
    """Area-weighted closest-panel distance plus a per-panel complexity cost."""
    _, distances = _assign_support_owners(planes, vertices, triangles, a, b)
    distances = np.minimum(distances, WALL_DISTANCE_TRUNCATION_M)
    distances[~np.isfinite(distances)] = WALL_DISTANCE_TRUNCATION_M
    mean_distance = float(np.average(distances, weights=areas))
    return mean_distance + WALL_PLANE_COUNT_PENALTY_M * len(planes)


def _topology_safe_subdivisions(
    base: list[dict[str, Any]], proposed: list[dict[str, Any]], tolerance: float,
    vertices: np.ndarray, triangles: np.ndarray, areas: np.ndarray, a: np.ndarray, b: np.ndarray,
) -> list[dict[str, Any]]:
    """Keep only objective-improving splits that preserve closure and the outside cap."""
    current = list(base)
    groups = sorted({int(plane["_split_group"]) for plane in proposed})
    for group in groups:
        replacement = [plane for plane in proposed if plane["_split_group"] == group]
        if len(replacement) == 1 and not replacement[0].get("inferred_connector"):
            continue
        trial = [plane for plane in current if int(plane["id"]) != group]
        trial.extend(replacement)
        _snap_plane_corners(trial, WALL_JUNCTION_SNAP_DISTANCE_M)
        _trim_split_corner_overshoots(trial, WALL_JUNCTION_SNAP_DISTANCE_M)
        if any(not _meets_minimum_plane_dimensions(
            plane.get("span_w_m", plane.get("span_u_m")), plane["bounds_y"]
        ) for plane in trial):
            continue
        try:
            _horizontal_topology(trial, tolerance)
        except ValueError:
            continue
        outside = _outside_panel_triangles(trial, vertices, triangles, a, b)
        if outside.mean() > MAX_OUTSIDE_WALL_TRIANGLE_FRACTION + 1e-12:
            continue
        if _layout_objective(trial, vertices, triangles, areas, a, b) >= (
            _layout_objective(current, vertices, triangles, areas, a, b) - 1e-12
        ):
            continue
        current = trial
    current.sort(key=lambda p: (p["axis"], p["offset_m"],
                                p.get("span_u_m", p.get("span_w_m"))[0]))
    for index, plane in enumerate(current, 1):
        plane["id"], plane["name"] = index, f"wall_plane_{index:03d}"
    return current


def _vertical_enclosure_bounds(
    instances_dir: Path, wall_vertices: np.ndarray, wall_centers: np.ndarray, wall_areas: np.ndarray,
    include_ceilings: bool = True,
) -> tuple[float, float, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    manifest_path = instances_dir / "structural_instances_manifest.json"
    floors, ceilings = [], []
    if manifest_path.is_file():
        for item in json.loads(manifest_path.read_text()).get("instances", []):
            role = str(item.get("role", "")).lower()
            if role not in {"floor", "ceiling"} or (role == "ceiling" and not include_ceilings):
                continue
            raw_path = Path(item.get("file_path", item.get("file_name", "")))
            path = raw_path if raw_path.is_file() else instances_dir / raw_path.name
            if not path.is_file():
                continue
            mesh = o3d.io.read_triangle_mesh(str(path))
            vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
            if not len(vertices) or not len(triangles):
                continue
            corners = vertices[triangles]
            face_areas = np.linalg.norm(np.cross(
                corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
            ), axis=1) * .5
            valid = np.isfinite(face_areas) & (face_areas > 1e-10)
            if not np.any(valid):
                continue
            points = corners[valid].reshape(-1, 3)
            weights = np.repeat(face_areas[valid] / 3., 3)
            (floors if role == "floor" else ceilings).append((points, weights))
    if floors and ceilings:
        floor_points, floor_weights = np.vstack([x[0] for x in floors]), np.concatenate([x[1] for x in floors])
        ceiling_points, ceiling_weights = np.vstack([x[0] for x in ceilings]), np.concatenate([x[1] for x in ceilings])
        floor_y = _weighted_quantile(floor_points[:, 1], floor_weights, .01)
        ceiling_y = _weighted_quantile(ceiling_points[:, 1], ceiling_weights, .99)
        return (floor_y, ceiling_y, "structural floor and ceiling instances",
                np.average(floor_points[:, [0, 2]], axis=0, weights=floor_weights),
                floor_points, floor_weights, ceiling_points, ceiling_weights)
    if floors and not include_ceilings:
        floor_points = np.vstack([item[0] for item in floors])
        floor_weights = np.concatenate([item[1] for item in floors])
        wall_points = np.asarray(wall_vertices, dtype=float)
        valid_wall = np.isfinite(wall_points).all(axis=1)
        wall_points = wall_points[valid_wall]
        wall_weights = np.ones(len(wall_points), dtype=float)
        if len(wall_points):
            floor_y = min(_weighted_quantile(points[:, 1], weights, .5)
                          for points, weights in floors)
            wall_top_y = _weighted_quantile(wall_points[:, 1], wall_weights, .99)
            return (floor_y, wall_top_y, "structural floors and wall mesh top",
                    np.average(floor_points[:, [0, 2]], axis=0, weights=floor_weights),
                    floor_points, floor_weights, wall_points, wall_weights)
    wall_weights = np.ones(len(wall_vertices), dtype=float)
    return (_weighted_quantile(wall_vertices[:, 1], wall_weights, .01),
            _weighted_quantile(wall_vertices[:, 1], wall_weights, .99),
            "wall mesh fallback", np.average(wall_centers[:, [0, 2]], axis=0, weights=wall_areas),
            np.empty((0, 3)), np.empty(0), np.empty((0, 3)), np.empty(0))


def _local_vertical_bounds(
    plane: dict[str, Any], floor_points: np.ndarray, floor_weights: np.ndarray,
    ceiling_points: np.ndarray, ceiling_weights: np.ndarray,
    fallback: tuple[float, float], normal_axis: np.ndarray, tangent: np.ndarray,
    support_distance_m: float | None = None,
) -> tuple[float, float, str]:
    """Use nearby floor/ceiling vertices; optional distance overrides the production search radius."""
    if not len(floor_points) or not len(ceiling_points):
        return fallback[0], fallback[1], "global wall/floor-ceiling fallback"
    span_key = "span_w_m" if plane["axis"] == "Dir A" else "span_u_m"
    span = plane[span_key]
    distance = (LOCAL_VERTICAL_SUPPORT_DISTANCE_M if support_distance_m is None
                else float(support_distance_m))

    source_floor_y, source_ceiling_y = plane["source_bounds_y"]

    def local_height(points: np.ndarray, weights: np.ndarray, quantile: float,
                     near_side: np.ndarray) -> float | None:
        along = points @ tangent
        near_plane = np.abs(points @ normal_axis - plane["offset_m"]) <= distance
        near_span = ((along >= span[0] - distance) & (along <= span[1] + distance))
        selected_mask = near_plane & near_span & near_side
        selected = points[selected_mask, 1]
        if len(selected) < 3:
            return None
        return _weighted_quantile(selected, weights[selected_mask], quantile)

    floor_y = local_height(floor_points, floor_weights, .95,
                           floor_points[:, 1] <= source_floor_y + distance)
    ceiling_y = local_height(ceiling_points, ceiling_weights, .05,
                             ceiling_points[:, 1] >= source_ceiling_y - distance)
    if floor_y is None or ceiling_y is None:
        return fallback[0], fallback[1], "global fallback (insufficient local support)"
    floor_y = min(floor_y, source_floor_y)
    ceiling_y = max(ceiling_y, source_ceiling_y)
    if ceiling_y - floor_y < MIN_WALL_ENCLOSURE_HEIGHT_M:
        return fallback[0], fallback[1], "global fallback (local height below minimum)"
    return floor_y, ceiling_y, "local floor and ceiling support"


def _write(path: Path, vertices: np.ndarray, triangles: np.ndarray, colors: np.ndarray) -> None:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False, compressed=False,
                                        write_vertex_normals=True, write_vertex_colors=True):
        raise OSError(f"Failed to write wall mesh: {path}")


def main() -> int:
    args = parse_args()
    try:
        paths = _mesh_paths(args)
        if not args.filter_only:
            _clear_wall_fit_outputs(args.output_dir, paths)
        if not paths:
            # Open terraces and partial scans still have useful surface geometry.
            direction = np.array([1., 0., 0.])
            longest = 0.
            instances = json.loads((args.instances_dir / "structural_instances_manifest.json").read_text())
            for item in instances.get("instances", []):
                if item.get("role") not in {"floor", "ceiling"}:
                    continue
                mesh = o3d.io.read_triangle_mesh(str(args.instances_dir / item["file_name"]))
                vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
                edges = np.sort(np.concatenate([faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]]), axis=1)
                edges, counts = np.unique(edges, axis=0, return_counts=True)
                for first, second in edges[counts == 1]:
                    vector = vertices[second] - vertices[first]
                    vector[1] = 0.
                    length = np.linalg.norm(vector)
                    if length > longest:
                        direction, longest = vector / length, length
            a = direction
            b = np.array([-a[2], 0., a[0]])
            dom_deg = float(np.rad2deg(np.arctan2(a[2], a[0]))) % 90.0
            args.output_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "dominant_orientation_deg": dom_deg,
                "orthogonal_basis": {"dir_A_east_west": a.tolist(), "dir_B_north_south": b.tolist()},
                "wall_planes": [],
                "artifacts": {},
            }
            (args.output_dir / "wall_planes_manifest.json").write_text(json.dumps(manifest, indent=2))
            return 0
        vertices, triangles, colors = _load_meshes(paths)
        triangles, normal, centers, areas, orientation_filter = _remove_horizontal_faces(vertices, triangles)
        if not len(triangles):
            raise ValueError(
                f"No nondegenerate wall triangles remain after removing faces within "
                f"{HORIZONTAL_FACE_TOLERANCE_DEG:g} degrees of horizontal"
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        filtered_path = args.output_dir / "wall_orientation_filtered.ply"
        filtered_vertices, filtered_remap = np.unique(triangles, return_inverse=True)
        _write(filtered_path, vertices[filtered_vertices], filtered_remap.reshape(-1, 3), colors[filtered_vertices])
        if args.filter_only:
            print(f"Removed {orientation_filter['horizontal_triangles_removed']:,} faces within "
                  f"±{HORIZONTAL_FACE_TOLERANCE_DEG:g}° of horizontal and "
                  f"{orientation_filter['degenerate_triangles_removed']:,} degenerate faces; "
                  f"retained {len(triangles):,}. Saved {filtered_path}; stopped before plane fitting.")
            return 0
        (floor_y, ceiling_y, vertical_source, floor_center, floor_points, floor_weights,
         ceiling_points, ceiling_weights) = _vertical_enclosure_bounds(
             args.instances_dir, vertices[filtered_vertices], centers, areas
         )
        if ceiling_y - floor_y < MIN_WALL_ENCLOSURE_HEIGHT_M:
            raise ValueError(
                f"Wall enclosure height {ceiling_y - floor_y:.3f} m is below the "
                f"{MIN_WALL_ENCLOSURE_HEIGHT_M:.1f} m minimum"
            )
        a, b, theta = _basis(normal, areas)
        scale = float(np.linalg.norm(np.ptp(vertices[filtered_vertices][:, [0, 2]], axis=0)))
        offset_tol = min(args.plane_dist_thresh, max(.025, scale * .02))
        a, b, theta, orientation_fit = _refine_basis_from_support(
            vertices, triangles, normal, centers, areas, a, b, theta, offset_tol
        )
        planes, _ = _discover(vertices, triangles, normal, centers, areas, a, b, offset_tol)
        if not planes:
            raise ValueError("No finite Manhattan wall planes were supported by the input")
        planes = _merge_nearby_planes(
            planes, WALL_PATCH_MERGE_DISTANCE_M, floor_center, a, b
        )
        for plane in planes:
            normal_axis = a if plane["axis"] == "Dir A" else b
            tangent = b if plane["axis"] == "Dir A" else a
            side = 1. if np.dot(plane["normal_unit"], normal_axis) >= 0. else -1.
            support = plane["source_faces"]
            projections = _outward_triangle_projections(
                vertices, triangles, support, normal_axis, side
            )
            # Put each envelope at the nearest offset covering at least 99% of its support triangles.
            plane["offset_m"] = _outward_plane_offset(projections, side)
            plane["normal_unit"] = side * normal_axis
            span, supported_height = _supported_bounds(vertices, triangles, areas, support, tangent)
            span_key = "span_w_m" if plane["axis"] == "Dir A" else "span_u_m"
            plane[span_key] = span
            plane["source_span_m"] = list(span)
            plane["source_bounds_y"] = supported_height
            local_floor, local_ceiling, bounds_source = _local_vertical_bounds(
                plane, floor_points, floor_weights, ceiling_points, ceiling_weights,
                (floor_y, ceiling_y), normal_axis, tangent
            )
            plane["bounds_y"] = [float(local_floor), float(local_ceiling)]
            plane["vertical_bounds_source"] = bounds_source
            plane["plane_height_m"] = float(local_ceiling - local_floor)
        planes = _merge_overlapping_parallel_planes(planes, vertices, triangles, areas, a, b)
        for plane in planes:
            normal_axis = a if plane["axis"] == "Dir A" else b
            support = plane["source_faces"]
            plane["support_plane_offset_m"] = _weighted_quantile(
                centers[support] @ normal_axis, areas[support], .5
            )
        planes = _retain_supported_planes(planes)
        if not planes:
            raise ValueError("No wall planes remain after support and minimum-dimension filtering")
        _snap_plane_corners(planes, WALL_JUNCTION_SNAP_DISTANCE_M)
        planes = _retain_supported_planes(planes)
        if not planes:
            raise ValueError("No wall planes remain after corner closure and minimum-dimension filtering")
        while True:
            proposed_planes = _subdivide_supported_planes(
                planes, vertices, triangles, centers, areas, a, b
            )
            refined_planes = _topology_safe_subdivisions(
                planes, proposed_planes, WALL_TOPOLOGY_TOLERANCE_M,
                vertices, triangles, areas, a, b,
            )
            if len(refined_planes) == len(planes):
                break
            planes = refined_planes
        _validate_parallel_overlaps(planes)
        _validate_unique_support(planes)
        # Track one nearest finite panel per projected triangle without changing
        # the source patches that determine each plane's fitted geometry.
        support_owner, support_distance = _assign_support_owners(
            planes, vertices, triangles, a, b
        )
        horizontal_topology = _horizontal_topology(planes, WALL_TOPOLOGY_TOLERANCE_M)
        local_inlier = np.zeros(len(triangles), dtype=bool)
        for plane in planes:
            local_inlier[plane["source_faces"]] = True
            plane["enclosing_area_m2"] = sum(
                (region["span_m"][1] - region["span_m"][0])
                * (region["bounds_y"][1] - region["bounds_y"][0])
                for region in _plane_regions(plane)
            )
            plane["support_coverage_ratio"] = min(
                1., plane["support_area_m2"] / max(plane["enclosing_area_m2"], 1e-12)
            )
            support = plane["source_faces"]
            normal_axis = a if plane["axis"] == "Dir A" else b
            if len(support):
                distances = _triangle_plane_distances(
                    vertices, triangles, support, normal_axis, plane["offset_m"]
                )
                plane["support_plane_offset_m"] = _weighted_quantile(
                    centers[support] @ normal_axis, areas[support], .5
                )
                normal_deviation = np.degrees(np.arccos(np.clip(
                    np.abs(normal[support] @ plane["normal_unit"]), 0., 1.
                )))
                fit_quality = {
                    "distance_residual_median_m": _weighted_quantile(distances, areas[support], .5),
                    "distance_residual_p95_m": _weighted_quantile(distances, areas[support], .95),
                    "normal_deviation_p95_deg": _weighted_quantile(
                        normal_deviation, areas[support], .95
                    ),
                    "support_coverage_ratio": plane["support_coverage_ratio"],
                    "inferred_enclosure_area_fraction": max(0., 1. - plane["support_coverage_ratio"]),
                }
                observed_support = {
                    "span_m": list(plane["source_span_m"]),
                    "bounds_y": list(plane["source_bounds_y"]),
                    "support_plane_offset_m": plane["support_plane_offset_m"],
                    "support_area_m2": plane["support_area_m2"],
                    "triangle_count": int(len(support)),
                }
            else:
                fit_quality = {
                    "distance_residual_median_m": None,
                    "distance_residual_p95_m": None,
                    "normal_deviation_p95_deg": None,
                    "support_coverage_ratio": 0.,
                    "inferred_enclosure_area_fraction": 1.,
                }
                observed_support = {
                    "span_m": list(plane["source_span_m"]),
                    "bounds_y": list(plane["source_bounds_y"]),
                    "support_plane_offset_m": None,
                    "support_area_m2": 0.,
                    "triangle_count": 0,
                }
            inferred_fraction = max(0., 1. - plane["support_coverage_ratio"])
            plane["fit_quality"] = fit_quality
            plane["observed_surface"] = observed_support
            plane["enclosure"] = {
                "span_m": list(plane.get("span_w_m", plane.get("span_u_m"))),
                "bounds_y": list(plane["bounds_y"]),
                "area_m2": plane["enclosing_area_m2"],
                "inferred_area_fraction": inferred_fraction,
            }
        inlier = local_inlier.copy()
        # The Manhattan frame is a regularization aid, not permission to erase a
        # measured diagonal/curved architectural return.  Retain meaningful
        # vertical source support and explicitly report that it was not flattened.
        horizontal_normal = np.linalg.norm(normal[:, [0, 2]], axis=1)
        unmatched = (horizontal_normal > .85) & (np.maximum(np.abs(normal @ a), np.abs(normal @ b)) < .85)
        retained_unmodeled = np.zeros(len(unmatched), dtype=bool)
        if np.any(unmatched):
            selected = triangles[unmatched]
            unique, remap = np.unique(selected, return_inverse=True)
            subset = o3d.geometry.TriangleMesh()
            subset.vertices = o3d.utility.Vector3dVector(vertices[unique])
            subset.triangles = o3d.utility.Vector3iVector(remap.reshape(-1, 3))
            labels, _, component_areas = subset.cluster_connected_triangles()
            labels = np.asarray(labels)
            supported = []
            for component in np.flatnonzero(np.asarray(component_areas) >= .02):
                points = vertices[np.unique(selected[labels == component])]
                horizontal_span = np.max(np.ptp(points @ np.array([a, b]).T, axis=0))
                covered = np.zeros(len(points), dtype=bool)
                for plane in planes:
                    normal_axis, tangent = (a, b) if plane["axis"] == "Dir A" else (b, a)
                    along = points @ tangent
                    for region in _plane_regions(plane):
                        span, bounds_y = region["span_m"], region["bounds_y"]
                        covered |= ((np.abs(points @ normal_axis-plane["offset_m"]) < .12)
                                    & (along >= span[0]-.12) & (along <= span[1]+.12)
                                    & (points[:, 1] >= bounds_y[0]-.12)
                                    & (points[:, 1] <= bounds_y[1]+.12))
                if horizontal_span >= .12-1e-9 and np.ptp(points[:, 1]) >= .12-1e-9 and not covered.all():
                    supported.append(component)
            retained_unmodeled[np.flatnonzero(unmatched)] = np.isin(np.asarray(labels), supported)
        inlier[retained_unmodeled] = True
        outside_mask = _outside_panel_triangles(planes, vertices, triangles, a, b)
        outside = np.flatnonzero(outside_mask)
        outside_summary = {
            "triangle_count": int(len(outside)),
            "triangle_fraction": float(len(outside) / len(triangles)),
            "area_m2": float(areas[outside].sum()),
            "area_fraction": float(areas[outside].sum() / max(float(areas.sum()), 1e-12)),
            "reference_triangle_count": int(len(triangles)),
            "reference_area_m2": float(areas.sum()),
            "measurement": "vertical-filtered triangles with no finite panel projection or outside the outward side of any overlapping panel",
            "connected_component_count": 0,
            "largest_component_area_m2": 0.,
            "largest_component_extent_m": [0., 0., 0.],
            "significant_component_area_threshold_m2": MIN_WALL_PLANE_FINAL_SUPPORT_AREA_M2,
            "significant_component_count": 0,
            "significant_component_area_m2": 0.,
        }
        if outside_summary["triangle_fraction"] > MAX_OUTSIDE_WALL_TRIANGLE_FRACTION + 1e-12:
            raise ValueError(
                f"{len(outside):,}/{len(triangles):,} wall triangles "
                f"({outside_summary['triangle_fraction']:.2%}) are outside the enclosing planes; "
                f"the maximum is {MAX_OUTSIDE_WALL_TRIANGLE_FRACTION:.0%}"
            )
        if len(outside):
            unique, remap = np.unique(triangles[outside], return_inverse=True)
            residual_mesh = o3d.geometry.TriangleMesh()
            residual_mesh.vertices = o3d.utility.Vector3dVector(vertices[unique])
            residual_mesh.triangles = o3d.utility.Vector3iVector(remap.reshape(-1, 3))
            labels, _, component_areas = residual_mesh.cluster_connected_triangles()
            labels, component_areas = np.asarray(labels), np.asarray(component_areas)
            largest = int(np.argmax(component_areas))
            largest_faces = outside[labels == largest]
            largest_vertices = vertices[np.unique(triangles[largest_faces])]
            outside_summary.update({
                "connected_component_count": int(len(component_areas)),
                "largest_component_area_m2": float(component_areas[largest]),
                "largest_component_extent_m": np.ptp(largest_vertices, axis=0).tolist(),
                "significant_component_count": int(np.count_nonzero(
                    component_areas >= MIN_WALL_PLANE_FINAL_SUPPORT_AREA_M2
                )),
                "significant_component_area_m2": float(component_areas[
                    component_areas >= MIN_WALL_PLANE_FINAL_SUPPORT_AREA_M2
                ].sum()),
            })

        args.output_dir.mkdir(parents=True, exist_ok=True)
        kept = triangles[inlier]; unique, remap = np.unique(kept, return_inverse=True)
        cleaned = args.output_dir / "wall_cleaned_pieces.ply"
        _write(cleaned, vertices[unique], remap.reshape(-1, 3), colors[unique])

        quad_v, quad_t, quad_c, records = [], [], [], []
        individual_dir = args.output_dir / "wall_enclosing_planes_individual"
        individual_dir.mkdir(parents=True, exist_ok=True)
        for plane in planes:
            plane_vertices, plane_triangles = _plane_quad_geometry(plane, a, b)
            start = len(quad_v)
            quad_v.extend(plane_vertices)
            quad_t.extend([[start + index for index in tri] for tri in plane_triangles])
            source = plane.pop("source_faces")
            plane.pop("_inlier_source_faces", None)
            plane.pop("_split_group", None)
            source_vertices = np.unique(triangles[source].reshape(-1))
            rgb = np.asarray(plane.pop("color"), dtype=float) if "color" in plane else (
                np.average(colors[source_vertices], axis=0) if len(source_vertices) else np.full(3, .8)
            )
            quad_c.extend([rgb] * len(plane_vertices))
            individual_path = individual_dir / f"{plane['name']}.ply"
            _write(individual_path, np.asarray(plane_vertices), np.asarray(plane_triangles),
                   np.tile(rgb, (len(plane_vertices), 1)))
            record = {key: value for key, value in plane.items()}
            n = record["normal_unit"]
            record["normal_unit"] = [round(float(v), 8) for v in n]
            normal_axis = a if plane["axis"] == "Dir A" else b
            plane_base = plane["offset_m"] * normal_axis
            record["plane_equation"] = [round(float(v), 8) for v in (*n, -np.dot(n, plane_base))]
            record["bounds_y"] = [round(float(v), 4) for v in record["bounds_y"]]
            record["source_bounds_y"] = [round(float(v), 4) for v in record["source_bounds_y"]]
            record["source_span_m"] = [round(float(v), 4) for v in record["source_span_m"]]
            record["plane_height_m"] = round(float(record["plane_height_m"]), 4)
            record["source"] = {"triangle_count": int(len(source)), "area_m2": round(float(areas[source].sum()), 4)}
            owned = support_owner == plane["id"]
            record["projected_support"] = {
                "triangle_count": int(np.count_nonzero(owned)),
                "area_m2": round(float(areas[owned].sum()), 4),
                "distance_residual_median_m": round(
                    _weighted_quantile(support_distance[owned], areas[owned], .5), 4
                ) if np.any(owned) else None,
                "distance_residual_p95_m": round(
                    _weighted_quantile(support_distance[owned], areas[owned], .95), 4
                ) if np.any(owned) else None,
            }
            record["individual_ply"] = str(individual_path)
            for key in ("offset_m", "span_u_m", "span_w_m"):
                if key in record:
                    record[key] = round(record[key], 4) if isinstance(record[key], float) else [round(v, 4) for v in record[key]]
            for key in ("support_area_m2", "enclosing_area_m2", "support_coverage_ratio",
                        "max_offset_residual_m"):
                record[key] = round(record[key], 4)
            for key, value in record["fit_quality"].items():
                record["fit_quality"][key] = round(float(value), 4) if value is not None else None
            for key in ("span_m", "bounds_y"):
                record["observed_surface"][key] = [
                    round(float(value), 4) for value in record["observed_surface"][key]
                ]
                record["enclosure"][key] = [
                    round(float(value), 4) for value in record["enclosure"][key]
                ]
            record["observed_surface"]["support_area_m2"] = round(
                float(record["observed_surface"]["support_area_m2"]), 4
            )
            if record["observed_surface"]["support_plane_offset_m"] is not None:
                record["observed_surface"]["support_plane_offset_m"] = round(
                    float(record["observed_surface"]["support_plane_offset_m"]), 4
                )
            record["enclosure"]["area_m2"] = round(float(record["enclosure"]["area_m2"]), 4)
            record["final_enclosing_area_m2"] = round(record["enclosing_area_m2"], 4)
            records.append(record)

        enclosing = args.output_dir / "wall_enclosing_planes.ply"
        _write(enclosing, np.asarray(quad_v), np.asarray(quad_t), np.asarray(quad_c))
        manifest = {"source_meshes": [str(p) for p in paths], "dominant_orientation_deg": round(float(np.degrees(theta)), 3),
                    "orientation_refinement": orientation_fit,
                    "orthogonal_basis": {"dir_A_east_west": a.tolist(), "dir_B_north_south": b.tolist()},
                    "plane_constraints": {
                        "nearby_patch_merge_distance_m": WALL_PATCH_MERGE_DISTANCE_M,
                        "nearby_patch_max_offset_spread_m": WALL_COPLANAR_MERGE_DISTANCE_M,
                        "parallel_overlap": "same-facing supported layers with overlapping projections merge into one outward envelope",
                        "horizontal_topology_tolerance_m": WALL_TOPOLOGY_TOLERANCE_M,
                        "minimum_plane_dimensions_m": {
                            "shorter": MIN_WALL_PLANE_SHORT_DIMENSION_M,
                            "longer": MIN_WALL_PLANE_LONG_DIMENSION_M,
                            "strictly_greater_than": True,
                        },
                        "maximum_outside_triangle_fraction": MAX_OUTSIDE_WALL_TRIANGLE_FRACTION,
                        "minimum_plane_support_area_m2": MIN_WALL_PLANE_FINAL_SUPPORT_AREA_M2,
                        "outward_plane_margin_m": WALL_PLANE_OUTWARD_MARGIN_M,
                        "plane_offset_quantile": "99th triangle-count percentile of farthest outward support vertices; no extra margin",
                        "target_supported_wall_triangle_coverage_fraction": WALL_PLANE_OUTWARD_COVERAGE,
                        "plane_layout_objective": {
                            "formula": "area_weighted_mean_closest_panel_distance_truncated_at_tau + lambda * total_plane_count",
                            "plane_count_penalty_m": WALL_PLANE_COUNT_PENALTY_M,
                            "distance_truncation_m": WALL_DISTANCE_TRUNCATION_M,
                            "acceptance": "split must strictly reduce the objective, preserve a closed connected side loop at every height band, and keep outside wall triangles at or below the hard cap",
                        },
                        "plane_distance": "shortest orthogonal distance to a supporting triangle; unsupported projected regions do not contribute",
                        "support_ownership": "each vertical-filtered triangle whose projection intersects a supported panel is assigned only to the closest panel; inferred connectors do not own observed triangles",
                        "adaptive_subdivision": "greedily repeat quality-improving splits; each split is scored by area-weighted closest-panel distance plus the plane-count penalty, and must preserve the closed side topology and 2% outside cap; offset steps use dimension-valid connector panels",
                        "junction_snap_distance_m": WALL_JUNCTION_SNAP_DISTANCE_M,
                        "plane_position": "offsets follow supported mesh and remain fixed during corner closure",
                        "junction_closure": "panel spans extend to supported perpendicular intersections within the configured snap distance",
                        "surface_bounds": "area-weighted 99% quantiles of supported triangle vertices; observed surface bounds are recorded separately from the constructed envelope",
                        "vertical_bounds": "use nearby floor/ceiling vertices within the configured support distance, otherwise fall back to robust global bounds",
                        "local_vertical_support_distance_m": LOCAL_VERTICAL_SUPPORT_DISTANCE_M,
                        "side_topology": "closed connected loop required at every height band; wall surfaces remain uncapped at floor and ceiling",
                        "vertical_enclosure": {
                            "source": vertical_source,
                            "floor_y_m": round(floor_y, 4),
                            "ceiling_y_m": round(ceiling_y, 4),
                            "height_m": round(ceiling_y - floor_y, 4),
                        },
                    },
                    "total_wall_planes_count": len(records), "wall_planes": records,
                    "horizontal_topology": horizontal_topology,
                    "unmodeled_wall_support": {"triangle_count": int(retained_unmodeled.sum()),
                                               "area_m2": round(float(areas[retained_unmodeled].sum()), 4),
                                               "reason": "vertical support outside the dominant Manhattan frame; retained in cleaned source"},
                    "outside_enclosing_planes": {**outside_summary,
                                                  "triangle_fraction": round(outside_summary["triangle_fraction"], 5),
                                                  "area_m2": round(outside_summary["area_m2"], 4),
                                                  "area_fraction": round(outside_summary["area_fraction"], 5),
                                                  "largest_component_area_m2": round(outside_summary["largest_component_area_m2"], 4),
                                                  "significant_component_area_m2": round(outside_summary["significant_component_area_m2"], 4),
                                                  "largest_component_extent_m": [round(v, 4) for v in outside_summary["largest_component_extent_m"]]},
                    "filtering_stats": {**orientation_filter,
                                        "horizontal_tolerance_degrees": HORIZONTAL_FACE_TOLERANCE_DEG,
                                        "triangles_after_orientation_filter": int(len(triangles)),
                                        "uniquely_supported_triangles": int(np.count_nonzero(support_owner >= 0)),
                                        "inlier_triangles": int(inlier.sum()),
                                        "inlier_fraction": round(float(inlier.mean()), 4),
                                        "inlier_area_m2": round(float(areas[inlier].sum()), 3)},
                    "artifacts": {"orientation_filtered_input": str(filtered_path),
                                  "cleaned_wall_pieces": str(cleaned), "enclosing_planes": str(enclosing),
                                  "individual_enclosing_planes": str(individual_dir)}}
        (args.output_dir / "wall_planes_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"Fitted {len(records)} wall panels at {np.degrees(theta):.2f}°; "
              f"closed floor-to-ceiling topology; {outside_summary['triangle_fraction']:.2%} of wall triangles "
              f"outside (maximum {MAX_OUTSIDE_WALL_TRIANGLE_FRACTION:.0%}); "
              f"{outside_summary['significant_component_count']} outside "
              f"components are at least {MIN_WALL_PLANE_FINAL_SUPPORT_AREA_M2:.2f} m² "
              f"(largest {outside_summary['largest_component_area_m2']:.3f} m²).")
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
