#!/usr/bin/env python3
"""Extract the largest clean square from each observed floor texture region."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, median_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_floor_texture_presence import FLOOR_ROLE, _face_property
from planar_pattern_regularization import (
    _plane_frame,
    _remove_thin_color_outliers,
    detect_visual_texture,
)

RASTER_CELL_M = 0.005
RASTER_SUPERSAMPLE = 2
FEATURE_WINDOW_PX = 32
FEATURE_STRIDE_PX = 8
MIN_TILE_SIDE_CELLS = 4
MIN_TEXTURE_REGION_AREA_M2 = 0.5
MIN_TEXTURE_REGION_SEGMENT_M = 0.5
# Chroma distance below this is treated as the same material under scene lighting.
MAX_TEXTURE_REGION_CHROMA_DISTANCE = 0.25
# Ignore coarse-label stair steps when checking the minimum meaningful boundary edge.
TEXTURE_REGION_CONTOUR_SIMPLIFY_M = 0.40
LIGHTING_COARSE_CELL_M = 0.05
LIGHTING_SMOOTHING_SIGMA_M = 0.20


def _rasterize_triangles(triangles_uv: np.ndarray, colors: np.ndarray,
                         cell_size: float = RASTER_CELL_M) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render source triangle RGB into its fitted-plane UV frame at 5 mm/cell."""
    triangles_uv = np.asarray(triangles_uv, dtype=np.float64).reshape(-1, 3, 2)
    colors = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
    if len(triangles_uv) != len(colors) or not len(colors):
        raise ValueError("Texture rasterization needs aligned, non-empty triangles and colors")
    finite = np.isfinite(triangles_uv).all(axis=(1, 2)) & np.isfinite(colors).all(axis=1)
    triangles_uv, colors = triangles_uv[finite], np.clip(colors[finite], 0.0, 1.0)
    if not len(colors):
        raise ValueError("No finite observed floor triangles to rasterize")

    lower = triangles_uv.min(axis=(0, 1))
    upper = triangles_uv.max(axis=(0, 1))
    width, height = np.ceil((upper - lower) / cell_size).astype(int) + 1
    if width < 2 or height < 2:
        raise ValueError("Observed floor support is too small to rasterize")
    sub_width, sub_height = width * RASTER_SUPERSAMPLE, height * RASTER_SUPERSAMPLE
    sub_rgb = np.zeros((sub_height, sub_width, 3), dtype=np.uint8)
    sub_support = np.zeros((sub_height, sub_width), dtype=np.uint8)
    pixels = np.rint((triangles_uv - lower) / (cell_size / RASTER_SUPERSAMPLE)).astype(np.int32)
    pixels[..., 1] = sub_height - 1 - pixels[..., 1]
    triangle_rgb = np.rint(colors * 255).astype(np.uint8)
    for triangle, color in zip(pixels, triangle_rgb):
        cv2.fillConvexPoly(sub_rgb, triangle, tuple(int(channel) for channel in color), lineType=cv2.LINE_8)
        cv2.fillConvexPoly(sub_support, triangle, 1, lineType=cv2.LINE_8)
    sub_support = sub_support.astype(bool).reshape(height, RASTER_SUPERSAMPLE,
                                                   width, RASTER_SUPERSAMPLE)
    support = sub_support.any(axis=(1, 3))
    samples = np.where(sub_support[..., None],
                       sub_rgb.reshape(height, RASTER_SUPERSAMPLE, width,
                                       RASTER_SUPERSAMPLE, 3), 0)
    rgb = samples.sum(axis=(1, 3)) / np.maximum(sub_support.sum(axis=(1, 3))[..., None], 1)
    nearest = distance_transform_edt(~support, return_distances=False, return_indices=True)
    rgb = rgb[tuple(nearest)]
    return rgb.astype(np.float32) / 255.0, support, lower


def _box_mean(values: np.ndarray, window: int) -> np.ndarray:
    return cv2.boxFilter(np.asarray(values, dtype=np.float32), -1, (window, window),
                         normalize=True, borderType=cv2.BORDER_REFLECT)


def _minimum_region_boundary_segment_m(cells: np.ndarray) -> float:
    """Measure the shortest edge after removing coarse-grid contour stair steps."""
    cell_m = FEATURE_STRIDE_PX * RASTER_CELL_M
    contours, _ = cv2.findContours(np.asarray(cells, dtype=np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1:
        return 0.0
    contour = cv2.approxPolyDP(
        contours[0], TEXTURE_REGION_CONTOUR_SIMPLIFY_M / cell_m, True
    ).reshape(-1, 2)
    if len(contour) < 3:
        return 0.0
    lengths = np.linalg.norm(np.roll(contour, -1, axis=0) - contour, axis=1) * cell_m
    return float(lengths.min())


def _regularize_raster_lighting(rgb: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, dict]:
    """Normalize slow luminance changes while retaining local surface detail."""
    rgb = np.asarray(rgb, dtype=np.float32)
    support = np.asarray(support, dtype=bool)
    if rgb.shape != (*support.shape, 3):
        raise ValueError("RGB raster and support mask dimensions do not match")
    if not np.any(support):
        return rgb.copy(), {"processed": True, "corrected": False,
                            "reason": "no observed support"}

    luminance = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    coarse_stride = max(1, int(round(LIGHTING_COARSE_CELL_M / RASTER_CELL_M)))
    coarse_size = (max(1, int(np.ceil(rgb.shape[1] / coarse_stride))),
                   max(1, int(np.ceil(rgb.shape[0] / coarse_stride))))
    support_weight = cv2.resize(support.astype(np.float32), coarse_size,
                                interpolation=cv2.INTER_AREA)
    weighted_luminance = cv2.resize(luminance * support, coarse_size,
                                    interpolation=cv2.INTER_AREA)
    coarse_luminance = weighted_luminance / np.maximum(support_weight, 1e-6)

    sigma = LIGHTING_SMOOTHING_SIGMA_M / LIGHTING_COARSE_CELL_M
    smooth_weight = cv2.GaussianBlur(support_weight, (0, 0), sigmaX=sigma, sigmaY=sigma,
                                     borderType=cv2.BORDER_REFLECT)
    smooth_luminance = cv2.GaussianBlur(coarse_luminance * support_weight, (0, 0),
                                        sigmaX=sigma, sigmaY=sigma,
                                        borderType=cv2.BORDER_REFLECT)
    coarse_field = smooth_luminance / np.maximum(smooth_weight, 1e-4)
    observed_field = coarse_field[support_weight > 0.05]
    reference = float(np.median(observed_field)) if len(observed_field) else 0.0
    field = cv2.resize(coarse_field, (rgb.shape[1], rgb.shape[0]),
                       interpolation=cv2.INTER_LINEAR)
    gain = np.clip(reference / np.maximum(field, 0.04), 0.67, 1.5)
    corrected = np.clip(rgb * gain[..., None], 0.0, 1.0)

    before = luminance[support]
    after = (corrected @ np.array([0.299, 0.587, 0.114], dtype=np.float32))[support]
    before_spread = float(np.percentile(before, 90) - np.percentile(before, 10))
    after_spread = float(np.percentile(after, 90) - np.percentile(after, 10))
    gain_values = gain[support]
    return corrected, {
        "processed": True,
        "corrected": bool(np.max(np.abs(gain_values - 1.0)) > 0.01),
        "method": "bounded_coarse_luminance_field",
        "coarse_cell_size_m": LIGHTING_COARSE_CELL_M,
        "smoothing_sigma_m": LIGHTING_SMOOTHING_SIGMA_M,
        "gain_range": [0.67, 1.5],
        "gain_p05_p95": [float(v) for v in np.percentile(gain_values, [5, 95])],
        "luminance_spread_before": before_spread,
        "luminance_spread_after": after_spread,
    }


def _region_components(rgb: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, list[dict], dict]:
    """Split supported raster into plain regions and locally distinct textures."""
    rgb = np.asarray(rgb, dtype=np.float32)
    support = np.asarray(support, dtype=bool)
    height, width = support.shape
    if rgb.shape != (height, width, 3):
        raise ValueError("RGB raster and support mask dimensions do not match")
    filled = rgb.copy()
    nearest = distance_transform_edt(~support, return_distances=False, return_indices=True)
    filled[~support] = rgb[tuple(nearest)][~support]

    luminance = filled @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    # Match the earlier detector's 6 cm high-pass scale at the 5 mm raster resolution.
    residual = luminance - gaussian_filter(
        luminance, sigma=0.06 / RASTER_CELL_M, mode="nearest"
    )
    window = FEATURE_WINDOW_PX
    coverage = _box_mean(support.astype(np.float32), window)
    energy = _box_mean(np.abs(residual), window)
    variance = np.maximum(_box_mean(residual * residual, window), 1e-8)
    correlations = []
    for axis in (1, 0):
        axis_scores = []
        for lag in (3, 5, 8, 12):
            shifted = np.roll(residual, lag, axis=axis)
            shifted_support = np.roll(support, lag, axis=axis)
            paired = support & shifted_support
            if axis == 1:
                paired[:, :lag] = False
            else:
                paired[:lag, :] = False
            pair_fraction = _box_mean(paired.astype(np.float32), window)
            covariance = _box_mean((residual * shifted) * paired, window)
            score = covariance / variance
            axis_scores.append(np.where(pair_fraction >= 0.55, score, -1.0))
        correlations.append(np.maximum.reduce(axis_scores))
    corr_x, corr_y = correlations

    # Chroma and directional correlation separate material changes from shadows.
    low = gaussian_filter(filled, sigma=(1.5, 1.5, 0), mode="nearest")
    low_luma = np.maximum(low @ np.array([0.299, 0.587, 0.114], dtype=np.float32), 1e-3)
    chroma = low / low_luma[..., None]
    chroma_r = _box_mean(chroma[..., 0], window)
    chroma_b = _box_mean(chroma[..., 2], window)
    log_luminance = np.log(np.maximum(_box_mean(low_luma, window), 1e-3))
    grad_x = cv2.Sobel(residual, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(residual, cv2.CV_32F, 0, 1, ksize=3)
    orientation = np.log((_box_mean(grad_x * grad_x, window) + 1e-5)
                         / (_box_mean(grad_y * grad_y, window) + 1e-5))

    stride = FEATURE_STRIDE_PX
    ys = np.arange(stride // 2, height, stride)
    xs = np.arange(stride // 2, width, stride)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    coarse_coverage = coverage[grid_y, grid_x]
    valid = coarse_coverage >= 0.70
    coarse_energy = energy[grid_y, grid_x]
    coarse_corr_x, coarse_corr_y = corr_x[grid_y, grid_x], corr_y[grid_y, grid_x]
    energy_values = coarse_energy[valid]
    if not len(energy_values):
        raise ValueError("Observed support has no sufficiently covered texture-analysis windows")
    energy_cutoff = max(0.0035, float(np.percentile(energy_values, 35)) * 0.45)
    texture = valid & (coarse_energy >= energy_cutoff) & (
        np.maximum(coarse_corr_x, coarse_corr_y) >= 0.12
    )
    texture = median_filter(texture.astype(np.uint8), size=3, mode="nearest").astype(bool) & valid

    feature_grid = np.stack((
        chroma_r[grid_y, grid_x], chroma_b[grid_y, grid_x],
        np.log(np.maximum(coarse_energy, 1e-5)), coarse_corr_x, coarse_corr_y,
        orientation[grid_y, grid_x], log_luminance[grid_y, grid_x],
    ), axis=-1)
    # Fixed physical feature scales keep region boundaries stable across floors.
    feature_scale = np.array([0.05, 0.05, 0.45, 0.20, 0.20, 0.8, 0.25], dtype=np.float32)
    normalized = feature_grid / feature_scale
    plain_features = feature_grid[..., (0, 1, 6)] / np.array([0.05, 0.05, 0.25])
    node_ids = np.full(valid.shape, -1, dtype=np.int32)
    node_ids[valid] = np.arange(np.count_nonzero(valid), dtype=np.int32)
    rows, cols = [], []
    for dy, dx in ((0, 1), (1, 0)):
        y0, y1 = slice(0, valid.shape[0] - dy), slice(dy, valid.shape[0])
        x0, x1 = slice(0, valid.shape[1] - dx), slice(dx, valid.shape[1])
        first, second = valid[y0, x0], valid[y1, x1]
        same_type = texture[y0, x0] == texture[y1, x1]
        compatible = same_type & first & second
        is_texture = texture[y0, x0] & compatible
        texture_distance = np.linalg.norm(
            normalized[y0, x0] - normalized[y1, x1], axis=-1
        )
        plain_distance = np.linalg.norm(
            plain_features[y0, x0] - plain_features[y1, x1], axis=-1
        )
        compatible &= np.where(is_texture, texture_distance <= 2.5, plain_distance <= 2.5)
        a, b = node_ids[y0, x0][compatible], node_ids[y1, x1][compatible]
        rows.extend((a, b))
        cols.extend((b, a))

    node_count = int(valid.sum())
    graph_rows = np.concatenate(rows) if rows else np.empty(0, dtype=np.int32)
    graph_cols = np.concatenate(cols) if cols else np.empty(0, dtype=np.int32)
    graph = coo_matrix((np.ones(len(graph_rows), dtype=np.uint8), (graph_rows, graph_cols)),
                       shape=(node_count, node_count)).tocsr()
    _, component_labels = connected_components(graph, directed=False)
    coarse_labels = np.zeros(valid.shape, dtype=np.int32)
    coarse_labels[valid] = component_labels + 1
    # Keep texture components distinct and enumerate plain regions after them.
    texture_component_ids = sorted(set(coarse_labels[texture].tolist()) - {0})
    plain_component_ids = sorted(set(coarse_labels[valid & ~texture].tolist()) - {0})
    texture_id_map = {label_id: index + 1 for index, label_id in enumerate(texture_component_ids)}
    plain_id_map = {label_id: len(texture_id_map) + index + 1
                    for index, label_id in enumerate(plain_component_ids)}
    original_labels = coarse_labels.copy()
    for old, new in texture_id_map.items():
        coarse_labels[original_labels == old] = new
    for old, new in plain_id_map.items():
        coarse_labels[original_labels == old] = new

    # Spread each coarse label to the raster for a readable region map and square search.
    full_labels = cv2.resize(coarse_labels.astype(np.int32), (width, height),
                             interpolation=cv2.INTER_NEAREST)
    regions = []
    for texture_id in range(1, len(texture_id_map) + 1):
        cells = coarse_labels == texture_id
        if not np.any(cells):
            continue
        ys0, xs0 = np.where(cells)
        regions.append({
            "texture_id": texture_id,
            "kind": "texture",
            "coarse_cell_count": int(cells.sum()),
            "coarse_grid_bounds": [int(xs0.min()), int(ys0.min()),
                                   int(xs0.max() + 1), int(ys0.max() + 1)],
            "minimum_boundary_segment_m": _minimum_region_boundary_segment_m(cells),
            "median_energy": float(np.median(coarse_energy[cells])),
            "median_correlation_x": float(np.median(coarse_corr_x[cells])),
            "median_correlation_y": float(np.median(coarse_corr_y[cells])),
            "median_grain_orientation": float(np.median(orientation[grid_y, grid_x][cells])),
            "median_chroma": [
                float(np.median(chroma_r[grid_y, grid_x][cells])),
                float(np.median(chroma_b[grid_y, grid_x][cells])),
            ],
        })
    for plain_id in range(len(texture_id_map) + 1, len(texture_id_map) + len(plain_id_map) + 1):
        cells = coarse_labels == plain_id
        if np.any(cells):
            ys0, xs0 = np.where(cells)
            regions.append({
                "region_id": plain_id,
                "kind": "plain",
                "coarse_cell_count": int(cells.sum()),
                "coarse_grid_bounds": [int(xs0.min()), int(ys0.min()),
                                       int(xs0.max() + 1), int(ys0.max() + 1)],
                "minimum_boundary_segment_m": _minimum_region_boundary_segment_m(cells),
                "median_chroma": [
                    float(np.median(chroma_r[grid_y, grid_x][cells])),
                    float(np.median(chroma_b[grid_y, grid_x][cells])),
                ],
            })
    return full_labels, regions, {
        "analysis_cell_size_m": FEATURE_STRIDE_PX * RASTER_CELL_M,
        "feature_window_m": FEATURE_WINDOW_PX * RASTER_CELL_M,
        "valid_window_count": int(valid.sum()),
        "texture_window_count": int(texture.sum()),
        "texture_energy_cutoff": energy_cutoff,
    }


def _largest_square(mask: np.ndarray) -> tuple[int, tuple[int, int] | None]:
    """Return the largest all-true square and its top-left coordinate."""
    mask = np.asarray(mask, dtype=bool)
    dp = np.zeros(mask.shape, dtype=np.int32)
    best, top_left = 0, None
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x]:
                dp[y, x] = 1 + min(
                    dp[y - 1, x] if y else 0,
                    dp[y, x - 1] if x else 0,
                    dp[y - 1, x - 1] if y and x else 0,
                )
                if dp[y, x] > best:
                    best = int(dp[y, x])
                    top_left = (x - best + 1, y - best + 1)
    return best, top_left


def _clean_square_mask(rgb: np.ndarray, support: np.ndarray, labels: np.ndarray,
                       texture_id: int, texture_ids: set[int]) -> np.ndarray:
    """Exclude poorly covered, deeply shadowed, and extreme-outlier coarse cells."""
    stride = FEATURE_STRIDE_PX
    coarse_h, coarse_w = labels.shape[0] // stride, labels.shape[1] // stride
    cropped_support = support[:coarse_h * stride, :coarse_w * stride]
    cropped_rgb = rgb[:coarse_h * stride, :coarse_w * stride]
    support_fraction = cropped_support.reshape(coarse_h, stride, coarse_w, stride).mean(axis=(1, 3))
    gray = cropped_rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    low = gaussian_filter(gray, sigma=10.0, mode="nearest")
    residual = np.linalg.norm(cropped_rgb - gaussian_filter(cropped_rgb, sigma=1.5,
                                                           axes=(0, 1), mode="nearest"), axis=2)
    residual_blocks = residual.reshape(coarse_h, stride, coarse_w, stride)
    residual_p99 = np.percentile(residual_blocks, 99, axis=(1, 3))
    coarse_labels = labels[stride // 2:stride // 2 + coarse_h * stride:stride,
                           stride // 2:stride // 2 + coarse_w * stride:stride]
    component = coarse_labels == texture_id
    neighboring_texture = np.isin(coarse_labels, list(texture_ids - {texture_id}))
    if np.any(component & binary_dilation(neighboring_texture, structure=np.ones((3, 3)))):
        # The analysis window blurs material transitions; leave a two-cell safety margin.
        transition = component & binary_dilation(neighboring_texture, structure=np.ones((3, 3)))
        component &= ~binary_dilation(transition, structure=np.ones((3, 3)))
    component_luma = np.maximum(low.reshape(coarse_h, stride, coarse_w, stride).mean(axis=(1, 3)), 1e-3)
    reference_luma = float(np.percentile(component_luma[component], 75)) if np.any(component) else 0.0
    shadow_free = component_luma >= 0.58 * reference_luma
    component_residual = residual_p99[component]
    outlier_limit = max(0.12, float(np.percentile(component_residual, 95)) * 2.0) if len(component_residual) else 0.12
    artifact_free = residual_p99 <= outlier_limit
    return component & (support_fraction >= 0.92) & shadow_free & artifact_free


def _is_significant_texture_region(region: dict) -> bool:
    cell_m = FEATURE_STRIDE_PX * RASTER_CELL_M
    area_m2 = int(region["coarse_cell_count"]) * cell_m * cell_m
    return (area_m2 >= MIN_TEXTURE_REGION_AREA_M2
            and float(region.get("minimum_boundary_segment_m", 0.0))
            >= MIN_TEXTURE_REGION_SEGMENT_M)


def _region_id(region: dict) -> int:
    return int(region["texture_id"] if region["kind"] == "texture" else region["region_id"])


def _regularize_region_labels(labels: np.ndarray, regions: list[dict],
                              textured_region_ids: set[int]) -> np.ndarray:
    """Keep broad material zones; fold small or same-color labels into the wood texture."""
    labels = np.asarray(labels, dtype=np.int32)
    regions_by_id = {_region_id(region): region for region in regions}
    texture_regions = [regions_by_id[region_id] for region_id in textured_region_ids
                       if region_id in regions_by_id]
    major_ids = {
        region_id for region_id, region in regions_by_id.items()
        if _is_significant_texture_region(region)
    } | set(textured_region_ids)
    if not major_ids and regions:
        major_ids.add(_region_id(max(regions, key=lambda item: item["coarse_cell_count"])))

    original = labels.copy()
    major_mask = np.isin(original, list(major_ids))
    nearest_indices = distance_transform_edt(~major_mask, return_distances=False,
                                             return_indices=True)
    nearest_ids = original[tuple(nearest_indices)]

    def closest_texture(region: dict) -> tuple[int | None, float]:
        chroma = np.asarray(region.get("median_chroma", []), dtype=np.float64)
        if chroma.shape != (2,) or not texture_regions:
            return None, float("inf")
        distances = [
            float(np.linalg.norm(chroma - np.asarray(candidate["median_chroma"], dtype=np.float64)))
            for candidate in texture_regions
        ]
        index = int(np.argmin(distances))
        return _region_id(texture_regions[index]), distances[index]

    major_targets = {}
    for region_id in major_ids:
        target_id, distance = closest_texture(regions_by_id[region_id])
        major_targets[region_id] = (
            target_id if distance <= MAX_TEXTURE_REGION_CHROMA_DISTANCE else region_id
        )
    nearest_targets = nearest_ids.copy()
    for region_id, target_id in major_targets.items():
        nearest_targets[nearest_ids == region_id] = target_id

    result = original.copy()
    for region_id, region in regions_by_id.items():
        if region_id in textured_region_ids:
            continue
        target_id, distance = closest_texture(region)
        if distance <= MAX_TEXTURE_REGION_CHROMA_DISTANCE:
            target = target_id
        elif region_id in major_targets:
            target = major_targets[region_id]
        else:
            nearest = nearest_targets[original == region_id]
            nearest = nearest[np.isin(nearest, list(major_targets.values()))]
            target = (
                int(np.bincount(nearest).argmax())
                if len(nearest) else next(iter(major_targets.values()))
            )
        result[original == region_id] = int(target)
    return result


def _process_area(level_id: int, area: dict, vertices: np.ndarray, faces: np.ndarray,
                  vertex_colors: np.ndarray, selected: np.ndarray) -> tuple[list[dict], dict]:
    equation = np.asarray(area["plane_equation"], dtype=np.float64)
    _, axis_u, axis_v, _ = _plane_frame(equation)
    source_faces = faces[selected]
    triangles = vertices[source_faces]
    uv_triangles = np.stack((triangles @ axis_u, triangles @ axis_v), axis=-1)
    triangle_rgb = vertex_colors[source_faces].mean(axis=1)
    rgb, support, lower = _rasterize_triangles(uv_triangles, triangle_rgb)
    rgb, _ = _regularize_raster_lighting(rgb, support)
    rgb, _ = _remove_thin_color_outliers(
        rgb, support, support.astype(np.float32) * RASTER_CELL_M ** 2
    )
    labels, regions, _ = _region_components(rgb, support)

    area_id = int(area["area_id"])
    triangle_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    ) * 0.5
    has_coherent_texture = detect_visual_texture(
        uv_triangles.mean(axis=1), triangle_rgb, triangle_areas
    )["present"]
    texture_regions = [region for region in regions
                       if has_coherent_texture and region["kind"] == "texture"]
    texture_ids = {int(region["texture_id"]) for region in texture_regions}
    stride = FEATURE_STRIDE_PX
    squares = {}
    for region in texture_regions:
        texture_id = int(region["texture_id"])
        if not _is_significant_texture_region(region):
            continue
        mask = _clean_square_mask(rgb, support, labels, texture_id, texture_ids)
        side, top_left = _largest_square(mask)
        if top_left is None or side < MIN_TILE_SIDE_CELLS:
            continue
        squares[texture_id] = (side, top_left)

    labels = _regularize_region_labels(labels, regions, set(squares))

    area_info = {
        "level": level_id,
        "area": area_id,
        "labels": labels,
        "axis_u": axis_u,
        "axis_v": axis_v,
        "raster_lower_uv": lower,
        "raster_cell_size_m": RASTER_CELL_M,
    }
    candidates = []
    for region in texture_regions:
        texture_id = int(region["texture_id"])
        if texture_id not in squares:
            continue
        side, top_left = squares[texture_id]
        x0, y0 = top_left[0] * stride, top_left[1] * stride
        x1, y1 = x0 + side * stride, y0 + side * stride
        tile = rgb[y0:y1, x0:x1].copy()
        coverage = float(support[y0:y1, x0:x1].mean())
        gray = tile @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        usable = float(np.mean((gray >= 0.04) & (gray <= 0.97)))
        candidates.append({
            "level": level_id,
            "area": area_id,
            "region": texture_id,
            "area_info": area_info,
            "tile_origin_uv": np.array([
                lower[0] + x0 * RASTER_CELL_M,
                lower[1] + (labels.shape[0] - 1 - y0) * RASTER_CELL_M,
            ]),
            "name": f"floor_level_{level_id:03d}_area_{area_id:03d}_texture_{texture_id:03d}.png",
            "tile": tile,
            "side_m": float(side * stride * RASTER_CELL_M),
            "coverage": coverage,
            "quality": float((side * stride * RASTER_CELL_M) ** 2 * coverage * usable),
        })
    return candidates, area_info


def _texture_signature(tile: np.ndarray) -> dict:
    """Describe grain and spatial pattern independently of smooth brightness changes."""
    gray = tile @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    low_frequency = gaussian_filter(gray, sigma=8.0, mode="nearest")
    detail = gray - low_frequency
    center = float(np.median(detail))
    scale = max(1.4826 * float(np.median(np.abs(detail - center))), 0.004)
    detail = np.clip((detail - center) / scale, -4.0, 4.0).astype(np.float32)

    histogram = np.histogram(detail, bins=20, range=(-4.0, 4.0))[0].astype(np.float64)
    histogram /= max(float(histogram.sum()), 1.0)
    gx = cv2.Sobel(detail, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(detail, cv2.CV_32F, 0, 1, ksize=3)
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    orientation = np.histogram(angle, bins=12, range=(0.0, np.pi), weights=np.hypot(gx, gy))[0]
    orientation = orientation.astype(np.float64)
    orientation /= max(float(orientation.sum()), 1e-12)

    correlations = []
    for lag in (2, 4, 8, 16):
        pair = []
        if detail.shape[1] > lag:
            pair.append(float(np.mean(detail[:, :-lag] * detail[:, lag:])))
        if detail.shape[0] > lag:
            pair.append(float(np.mean(detail[:-lag, :] * detail[lag:, :])))
        correlations.extend(sorted(pair))

    mean_rgb = np.maximum(tile.mean(axis=(0, 1)), 1e-4)
    mean_luminance = max(float(mean_rgb @ np.array([0.299, 0.587, 0.114])), 1e-4)
    return {
        "detail_histogram": histogram,
        "orientation_histogram": orientation,
        "correlations": np.asarray(correlations, dtype=np.float64),
        "chromaticity": mean_rgb / mean_luminance,
    }


def _texture_similarity(first: dict, second: dict) -> float:
    """Return a conservative 0..1 material-texture similarity score."""
    hist_distance = float(np.sqrt(np.sum((np.sqrt(first["detail_histogram"])
                                          - np.sqrt(second["detail_histogram"])) ** 2) / 2.0))
    orientation_distance = min(
        float(np.abs(first["orientation_histogram"]
                     - np.roll(second["orientation_histogram"], shift)).sum() / 2.0)
        for shift in range(len(first["orientation_histogram"]))
    )
    correlation_distance = min(1.0, float(np.mean(np.abs(
        first["correlations"] - second["correlations"]
    ))))
    chroma_distance = min(1.0, float(np.linalg.norm(
        first["chromaticity"] - second["chromaticity"]
    ) / 0.60))
    structural_distance = (0.40 * hist_distance + 0.30 * orientation_distance
                           + 0.30 * correlation_distance)
    return float(np.clip(1.0 - (0.85 * structural_distance + 0.15 * chroma_distance), 0.0, 1.0))


def _deduplicate_texture_candidates(candidates: list[dict], threshold: float = 0.80) -> tuple[list[dict], list[dict]]:
    ranked = sorted(candidates, key=lambda item: (item["quality"], item["coverage"], item["side_m"]),
                    reverse=True)
    kept, duplicates = [], []
    for candidate in ranked:
        signature = _texture_signature(candidate["tile"])
        best = max(((_texture_similarity(signature, item["signature"]), item) for item in kept),
                   key=lambda match: match[0], default=(0.0, None))
        if best[0] >= threshold:
            duplicates.append({"removed": candidate, "kept": best[1], "similarity": best[0]})
        else:
            kept.append({**candidate, "signature": signature})
    return [{key: value for key, value in item.items() if key != "signature"} for item in kept], duplicates


def prepare_floor_texture_tiles(instances_dir: Path) -> tuple[list[dict], list[dict]]:
    level_dir = instances_dir / "intermediate_floor_levels"
    source_manifest = level_dir / "floor_level_intermediate_manifest.json"
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    levels = manifest.get("levels") if isinstance(manifest, dict) else None
    if not isinstance(levels, list) or not levels:
        raise ValueError(f"No fitted floor levels found in {source_manifest}")

    candidates = []
    for level in levels:
        level_id, source_name = level.get("level"), level.get("fitted_output_file")
        if (not isinstance(level_id, int) or level_id < 0
                or source_name != f"floor_level_{level_id:03d}_fitted.ply"):
            raise ValueError(f"Invalid fitted floor entry in {source_manifest}: {level!r}")
        source_path = level_dir / source_name
        mesh = o3d.io.read_triangle_mesh(str(source_path))
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.triangles, dtype=np.int64)
        colors = np.asarray(mesh.vertex_colors, dtype=np.float64)
        if not len(faces) or len(colors) != len(vertices):
            raise ValueError(f"Floor PLY needs triangle geometry and vertex RGB: {source_path}")
        roles = _face_property(source_path, "surface_role", len(faces))
        previews = _face_property(source_path, "is_fitted_plane", len(faces))
        fills = _face_property(source_path, "is_hole_fill", len(faces))
        area_ids = _face_property(source_path, "floor_area_id", len(faces))
        observed_floor = (roles == FLOOR_ROLE) & (previews == 0) & (fills == 0)
        level_candidates = []
        processed_areas = 0
        for area in level.get("floor_areas", []):
            area_id = area.get("area_id")
            if not isinstance(area_id, int) or np.asarray(area.get("plane_equation")).shape != (4,):
                raise ValueError(f"Invalid fitted plane metadata for floor level {level_id}")
            selected = observed_floor & (area_ids == area_id)
            if not np.any(selected):
                continue
            processed_areas += 1
            area_candidates, _ = _process_area(
                level_id, area, vertices, faces, colors, selected
            )
            level_candidates.extend(area_candidates)
        if not processed_areas:
            raise ValueError(f"Floor level {level_id} has no observed floor triangles")
        candidates.extend(level_candidates)

    final_candidates, duplicates = _deduplicate_texture_candidates(candidates)
    return final_candidates, duplicates


def extract_floor_texture_tiles(instances_dir: Path) -> tuple[list[dict], list[dict]]:
    """Return clean matched texture samples in memory; create no image files."""
    return prepare_floor_texture_tiles(instances_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path,
                        default=Path("backend/current_scene/scan_0/mesh_elaboration"))
    args = parser.parse_args()
    candidates, duplicates = extract_floor_texture_tiles(args.instances_dir)
    print(f"Found {len(candidates)} distinct texture(s); matched {len(duplicates)} duplicate sample(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
