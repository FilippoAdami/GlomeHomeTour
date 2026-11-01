"""Generic appearance and geometric-pattern regularization for fitted planes."""
from __future__ import annotations

from collections import deque

import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d, label, median_filter
from scipy.signal import find_peaks
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

SCENE_COLOR_ADAPTATION_STRENGTH = 0.85
MAX_SYNTHESIZED_COLOR = 0.95
MIN_REGION_CHROMATICITY_DISTANCE = 0.12
MAX_NEUTRAL_REGION_BRIGHTNESS_RATIO = 2.5
MAX_GEOMETRIC_RELIEF_M = 0.05


def _plane_frame(
    equation: np.ndarray, mesh: trimesh.Trimesh | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    equation = np.asarray(equation, dtype=np.float64).reshape(4)
    length = float(np.linalg.norm(equation[:3]))
    if not np.isfinite(equation).all() or length <= 1e-10:
        raise ValueError("Plane equation must have a finite, nonzero normal")
    normal, offset = equation[:3] / length, float(equation[3] / length)
    axis_u = None
    if mesh is not None and len(mesh.faces):
        # Anchor UV to a real footprint edge so detected patterns follow the fitted plane shape.
        edge_counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
        boundary_edges = mesh.edges_unique[edge_counts == 1]
        if len(boundary_edges):
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            edge_vectors = vertices[boundary_edges[:, 1]] - vertices[boundary_edges[:, 0]]
            edge_vectors -= (edge_vectors @ normal)[:, None] * normal
            lengths = np.linalg.norm(edge_vectors, axis=1)
            if np.max(lengths, initial=0.0) > 1e-10:
                axis_u = edge_vectors[int(np.argmax(lengths))] / np.max(lengths)
                dominant = int(np.argmax(np.abs(axis_u)))
                if axis_u[dominant] < 0:
                    axis_u = -axis_u
    if axis_u is None:
        reference = np.eye(3)[int(np.argmin(np.abs(normal)))]
        axis_u = np.cross(reference, normal)
        axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    return normal, axis_u, axis_v, offset


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    return float(values[min(len(values) - 1, np.searchsorted(np.cumsum(weights), weights.sum() * 0.5))])


def _scene_white_balance(
    patches: list[dict], neutral_reference_rgb: tuple[float, float, float] | None
) -> tuple[np.ndarray, dict]:
    if neutral_reference_rgb is None:
        return np.ones(3), {"method": "none", "applied": False,
                            "reason": "no neutral surface reference supplied"}
    target = np.asarray(neutral_reference_rgb, dtype=np.float64).reshape(3)
    if not np.isfinite(target).all() or np.any((target <= 0) | (target > 1)):
        raise ValueError("Neutral surface reference RGB must be finite values in (0, 1]")

    plane_colors, plane_areas = [], []
    for patch in patches:
        rgb = np.asarray(patch.get("support_colors", []), dtype=np.float64).reshape(-1, 3)
        weight = np.asarray(patch.get("support_areas", []), dtype=np.float64).reshape(-1)
        if len(rgb) != len(weight):
            raise ValueError(f"Support arrays do not align for plane {patch.get('name', '<unnamed>')}")
        valid = np.isfinite(rgb).all(axis=1) & np.isfinite(weight) & (weight > 0)
        if np.any(valid):
            rgb, weight = np.clip(rgb[valid], 0.0, 1.0), weight[valid]
            plane_colors.append(np.array([_weighted_median(rgb[:, channel], weight)
                                          for channel in range(3)]))
            plane_areas.append(float(weight.sum()))
    plain = {"method": "dominant_neutral_surface", "applied": False}
    if not plane_colors:
        return np.ones(3), plain

    colors, areas = np.vstack(plane_colors), np.asarray(plane_areas)
    reference = np.array([_weighted_median(colors[:, channel], areas) for channel in range(3)])
    chroma_spread = float(np.ptp(reference) / max(float(reference.max()), 0.05))
    reference_radius = max(0.12, 0.15 * float(np.linalg.norm(reference)))
    dominant_fraction = float(areas[np.linalg.norm(colors - reference, axis=1) <= reference_radius].sum()
                              / areas.sum())
    info = {
        **plain,
        "source_reference_rgb": reference.tolist(),
        "source_support_area": float(areas.sum()),
        "dominant_chroma_spread": chroma_spread,
        "dominant_color_area_fraction": dominant_fraction,
    }
    # shortcut: assumes dominant supported material matches neutral_reference_rgb; supply it only when known.
    if chroma_spread > 0.45:
        return np.ones(3), {**info, "reason": "dominant surface is too chromatic for a neutral reference"}
    if dominant_fraction < 0.60:
        return np.ones(3), {**info, "reason": "no dominant surface color for a neutral reference"}

    gains = np.clip(target / np.maximum(reference, 0.04), 0.5, 4.0) ** SCENE_COLOR_ADAPTATION_STRENGTH
    return gains, {**info, "applied": True, "target_reference_rgb": target.tolist(),
                   "adaptation_strength": SCENE_COLOR_ADAPTATION_STRENGTH,
                   "gain_rgb": gains.tolist()}


def _preserve_color_highlights(colors: np.ndarray) -> np.ndarray:
    colors = np.maximum(np.asarray(colors, dtype=np.float64), 0.0)
    peak = colors.max(axis=1, keepdims=True)
    scale = np.minimum(1.0, MAX_SYNTHESIZED_COLOR / np.maximum(peak, 1e-12))
    return colors * scale


def _regularize_lighting(colors: np.ndarray, uv: np.ndarray,
                         areas: np.ndarray) -> tuple[np.ndarray, dict]:
    plain = {"corrected": False, "luminance_spread_before": 0.0,
             "luminance_spread_after": 0.0}
    if len(colors) < 12:
        return colors.copy(), plain
    raster = _rasterize_support(uv, colors, areas)
    if _two_color_modes(raster["grid"], raster["occupied"], raster["area_grid"]) is not None:
        return colors.copy(), {**plain, "reason": "distinct supported color regions"}
    center = np.median(uv, axis=0)
    scale = np.maximum(np.percentile(uv, 95, axis=0) - np.percentile(uv, 5, axis=0), 1e-6)
    u, v = ((uv - center) / scale).T
    design = np.column_stack((np.ones(len(u)), u, v, u * u, v * v, u * v))
    luminance = colors @ np.array([0.299, 0.587, 0.114])
    base_weights = np.maximum(areas, 1e-12)
    base_weights /= np.median(base_weights)
    weights = base_weights.copy()
    try:
        for _ in range(5):
            root = np.sqrt(weights)
            coefficients = np.linalg.lstsq(design * root[:, None], luminance * root, rcond=None)[0]
            residual = luminance - design @ coefficients
            residual_center = np.median(residual)
            robust_scale = max(1.4826 * np.median(np.abs(residual - residual_center)), 0.01)
            weights = base_weights * np.minimum(
                1.0, 1.5 * robust_scale / np.maximum(np.abs(residual - residual_center), 1e-12)
            )
        predicted = design @ coefficients
    except np.linalg.LinAlgError:
        return colors.copy(), plain
    inliers = weights >= 0.5 * base_weights
    if np.count_nonzero(inliers) < 12 or np.ptp(predicted[inliers]) < 0.025:
        return colors.copy(), plain
    reference = float(np.median(luminance[inliers]))
    floor = max(0.04, float(np.percentile(luminance[inliers], 10)))
    gain = np.clip(reference / np.maximum(predicted, floor), 0.65, 1.5)
    corrected = _preserve_color_highlights(colors * gain[:, None])
    corrected_luminance = corrected @ np.array([0.299, 0.587, 0.114])
    spread_before = float(np.percentile(luminance[inliers], 90) - np.percentile(luminance[inliers], 10))
    spread_after = float(np.percentile(corrected_luminance[inliers], 90)
                         - np.percentile(corrected_luminance[inliers], 10))
    return corrected, {
        "corrected": True,
        "lighting_inlier_count": int(inliers.sum()),
        "luminance_spread_before": spread_before,
        "luminance_spread_after": spread_after,
    }


def _rasterize_support(uv: np.ndarray, colors: np.ndarray, areas: np.ndarray) -> dict:
    bounds = np.column_stack((uv.min(axis=0), uv.max(axis=0)))
    span = np.maximum(bounds[:, 1] - bounds[:, 0], 0.02)
    cell = max(0.02, float(span.max()) / 192.0)
    nx, ny = np.maximum(2, np.ceil(span / cell).astype(int) + 1)
    ix = np.clip(((uv[:, 0] - bounds[0, 0]) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((uv[:, 1] - bounds[1, 0]) / cell).astype(int), 0, ny - 1)
    flat = iy * nx + ix
    weights = np.maximum(areas, 1e-12)
    cell_weights = np.bincount(flat, weights=weights, minlength=nx * ny)
    rgb = np.column_stack([
        np.bincount(flat, weights=colors[:, channel] * weights, minlength=nx * ny)
        for channel in range(3)
    ])
    occupied = cell_weights > 0
    rgb[occupied] /= cell_weights[occupied, None]
    grid = rgb.reshape(ny, nx, 3)
    occupied = occupied.reshape(ny, nx)
    area_grid = cell_weights.reshape(ny, nx)
    nearest = distance_transform_edt(~occupied, return_distances=False, return_indices=True)
    filled = grid[tuple(nearest)]
    return {"grid": filled, "occupied": occupied, "area_grid": area_grid,
            "bounds": bounds, "cell_size": cell}


def _remove_thin_color_outliers(grid: np.ndarray, occupied: np.ndarray,
                                area_grid: np.ndarray) -> tuple[np.ndarray, int]:
    # A 5-cell window suppresses thin projected clutter without erasing broad regions.
    local = median_filter(grid, size=(5, 5, 1), mode="nearest")
    distance = np.linalg.norm(grid - local, axis=2)
    values = distance[occupied]
    if not len(values):
        return grid.copy(), 0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = max(0.18, median + 4.0 * 1.4826 * mad)
    candidates = occupied & (distance > threshold)
    components, count = label(candidates, structure=np.ones((3, 3), dtype=np.uint8))
    cleaned = grid.copy()
    removed = 0
    total_area = float(area_grid.sum())
    max_component_area = max(0.05, 0.005 * total_area)
    for component_id in range(1, count + 1):
        ys, xs = np.where(components == component_id)
        if not len(xs):
            continue
        component_area = float(area_grid[components == component_id].sum())
        if component_area <= max_component_area:
            cleaned[components == component_id] = local[components == component_id]
            removed += len(xs)
    return np.clip(cleaned, 0.0, 1.0), removed


def _two_color_modes(grid: np.ndarray, occupied: np.ndarray,
                     area_grid: np.ndarray) -> dict | None:
    colors = grid[occupied]
    weights = area_grid[occupied]
    if len(colors) < 50 or weights.sum() <= 0:
        return None
    center = np.average(colors, axis=0, weights=weights)
    centered = colors - center
    covariance = (centered * weights[:, None]).T @ centered / weights.sum()
    _, vectors = np.linalg.eigh(covariance)
    projection = centered @ vectors[:, -1]
    thresholds = np.percentile(projection, np.linspace(5, 95, 37))
    total_error = float(np.sum(weights * np.sum(centered * centered, axis=1)))
    if total_error <= 1e-10:
        return None
    best = None
    for threshold in thresholds:
        low = projection <= threshold
        if low.mean() < 0.025 or (~low).mean() < 0.025:
            continue
        c0 = np.average(colors[low], axis=0, weights=weights[low])
        c1 = np.average(colors[~low], axis=0, weights=weights[~low])
        error = float(np.sum(weights[low] * np.sum((colors[low] - c0) ** 2, axis=1))
                      + np.sum(weights[~low] * np.sum((colors[~low] - c1) ** 2, axis=1)))
        delta = float(np.linalg.norm(c0 - c1))
        within = np.sqrt(error / weights.sum())
        if delta < max(0.12, 3.0 * within) or 1.0 - error / total_error < 0.55:
            continue
        score = error / total_error
        if best is None or score < best[0]:
            best = (score, c0, c1, delta)
    if best is None:
        return None
    _, c0, c1, delta = best
    luminance_weights = np.array([0.299, 0.587, 0.114])
    chroma0 = c0 / max(float(c0 @ luminance_weights), 1e-6)
    chroma1 = c1 / max(float(c1 @ luminance_weights), 1e-6)
    luminance0 = float(c0 @ luminance_weights)
    luminance1 = float(c1 @ luminance_weights)
    brightness_ratio = max(luminance0, luminance1) / max(min(luminance0, luminance1), 1e-6)
    if (np.linalg.norm(chroma0 - chroma1) < MIN_REGION_CHROMATICITY_DISTANCE
            and brightness_ratio <= MAX_NEUTRAL_REGION_BRIGHTNESS_RATIO):
        return None
    return {"colors": np.asarray([c0, c1]), "delta_rgb": delta}


def _straight_color_split(labels: np.ndarray, occupied: np.ndarray,
                          area_grid: np.ndarray) -> np.ndarray | None:
    """Accept two colors only when one supported straight cut separates them."""
    total_area = float(area_grid[occupied].sum())
    if total_area <= 0:
        return None
    ny, nx = labels.shape
    best = None
    for axis, length in ((1, nx), (0, ny)):
        for cut in range(1, length):
            before = [slice(None), slice(None)]
            before[axis] = slice(0, cut)
            after = [slice(None), slice(None)]
            after[axis] = slice(cut, length)
            a, b = tuple(before), tuple(after)
            weights = np.array([
                [area_grid[a][occupied[a] & (labels[a] == color)].sum(),
                 area_grid[b][occupied[b] & (labels[b] == color)].sum()]
                for color in (0, 1)
            ])
            side_area = weights.sum(axis=0)
            if np.any(side_area < 0.15 * total_area):
                continue
            # A real boundary must be observed across most of its length, on both sides.
            if axis == 1:
                paired = np.sum(occupied[:, :cut].any(axis=1) & occupied[:, cut:].any(axis=1))
                boundary_length = np.sum(occupied.any(axis=1))
            else:
                paired = np.sum(occupied[:cut, :].any(axis=0) & occupied[cut:, :].any(axis=0))
                boundary_length = np.sum(occupied.any(axis=0))
            if paired / max(boundary_length, 1) < 0.70:
                continue
            purity = max(weights[0, 0] + weights[1, 1],
                         weights[0, 1] + weights[1, 0]) / total_area
            if purity < 0.90:
                continue
            if best is None or purity > best[0]:
                split_labels = np.zeros_like(labels, dtype=np.uint8)
                if axis == 1:
                    split_labels[:, cut:] = 1
                else:
                    split_labels[cut:, :] = 1
                if weights[0, 1] + weights[1, 0] > weights[0, 0] + weights[1, 1]:
                    split_labels = 1 - split_labels
                best = purity, split_labels
    return None if best is None else best[1]


def _color_split_metadata(visual: dict | None) -> dict:
    split = bool(visual and visual["kind"] == "solid_regions")
    colors = visual["info"].get("region_colors_rgb", []) if split else []
    return {
        "is_split": split,
        "split_type": "regions" if split else "none",
        "split_coord": None,
        "color_1": colors[0] if len(colors) > 0 else None,
        "color_2": colors[1] if len(colors) > 1 else None,
        "delta_E": float(visual["info"].get("color_delta_rgb", 0.0)) if split else 0.0,
    }


def _surface_color_regions_are_connected(
    mesh: trimesh.Trimesh, colors: np.ndarray, palette: np.ndarray,
    minimum_component_fraction: float = 0.80,
) -> bool:
    if len(palette) != 2 or not len(mesh.faces):
        return True
    labels = np.argmin(np.linalg.norm(colors[:, None, :] - palette[None, :, :], axis=2), axis=1)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    for color_id in (0, 1):
        face_ids = np.flatnonzero(labels == color_id)
        if not len(face_ids):
            return False
        local = np.full(len(mesh.faces), -1, dtype=np.int64)
        local[face_ids] = np.arange(len(face_ids))
        pairs = local[adjacency]
        pairs = pairs[(pairs >= 0).all(axis=1)]
        graph = coo_matrix(
            (np.ones(len(pairs) * 2),
             (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
            shape=(len(face_ids), len(face_ids)),
        ).tocsr()
        _, components = connected_components(graph, directed=False)
        component_areas = np.bincount(components, weights=areas[face_ids])
        if component_areas.max() / component_areas.sum() < minimum_component_fraction:
            return False
    return True


def _axis_repeat(
    grid: np.ndarray, occupied: np.ndarray, axis: int
) -> tuple[int | None, float, tuple[int, int, int, int] | None]:
    """Find repetition along one axis and keep only its longest coherent band."""
    colors = grid[occupied]
    if len(colors) < 100:
        return None, 0.0, None
    _, vectors = np.linalg.eigh(np.cov(colors.T))
    signal = grid @ vectors[:, -1]
    signal -= gaussian_filter(signal, sigma=2.0, mode="nearest")
    if axis == 0:
        signal, occupied = signal.T, occupied.T
    gradient = np.diff(signal, axis=1)
    gradient_support = occupied[:, 1:] & occupied[:, :-1]
    if np.count_nonzero(gradient_support) < 50 or float(np.std(gradient[gradient_support])) < 0.015:
        return None, 0.0, None
    dimension, lines = gradient.shape[1], gradient.shape[0]
    candidates = []
    for lag in range(3, dimension // 3 + 1):
        valid = gradient_support[:, :-lag] & gradient_support[:, lag:]
        counts = valid.sum(axis=1)
        first = np.where(valid, gradient[:, :-lag], 0.0)
        second = np.where(valid, gradient[:, lag:], 0.0)
        safe_counts = np.maximum(counts, 1)
        first_mean = first.sum(axis=1) / safe_counts
        second_mean = second.sum(axis=1) / safe_counts
        first -= first_mean[:, None] * valid
        second -= second_mean[:, None] * valid
        first_var = np.sum(first * first, axis=1) / safe_counts
        second_var = np.sum(second * second, axis=1) / safe_counts
        correlations = np.full(lines, -1.0)
        eligible = ((counts >= max(8, int(0.35 * (dimension - lag))))
                    & (first_var >= 1e-4) & (second_var >= 1e-4))
        covariance = np.sum(first * second, axis=1) / safe_counts
        correlations[eligible] = covariance[eligible] / np.sqrt(
            first_var[eligible] * second_var[eligible]
        )
        coherent = correlations >= 0.72
        active = coherent.copy()
        # A single scan dropout may interrupt an otherwise continuous patterned band.
        active[1:-1] |= active[:-2] & active[2:]
        changes = np.diff(np.r_[False, active, False].astype(np.int8))
        runs = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
        for start, stop in runs:
            if stop - start < max(5, int(np.ceil(lines * 0.08))):
                continue
            if np.mean(coherent[start:stop]) < 0.60:
                continue
            run_score = float(np.median(correlations[start:stop][correlations[start:stop] >= 0.72]))
            candidates.append((lag, run_score, start, stop, correlations))
    if not candidates:
        return None, 0.0, None
    best_score = max(score for _, score, _, _, _ in candidates)
    best = min((candidate for candidate in candidates
                if candidate[1] >= best_score - 0.03), key=lambda candidate: candidate[0])
    lag, score, start, stop, correlations = best
    run_gradient = np.abs(gradient[start:stop])
    run_support = gradient_support[start:stop]
    energy = np.where(run_support, run_gradient, 0.0).sum(axis=0)
    threshold = max(0.01 * (stop - start), 0.15 * float(energy.max()))
    active_positions = np.flatnonzero(energy >= threshold)
    if len(active_positions) < 3:
        return None, score, None
    along_start, along_stop = int(active_positions[0]), int(active_positions[-1] + 2)
    if axis == 1:
        region = (along_start, along_stop, max(0, start - 1), min(lines, stop + 1))
    else:
        region = (max(0, start - 1), min(lines, stop + 1), along_start, along_stop)
    return lag, score, region


def _make_seamless_tile(tile: np.ndarray, axes: tuple[bool, bool]) -> np.ndarray:
    """Match opposite tile edges with a narrow, bounded wrap correction."""
    tile = np.asarray(tile, dtype=np.float64).copy()
    for axis, periodic in enumerate(axes):
        if not periodic or tile.shape[axis] < 4:
            continue
        margin = min(8, max(2, tile.shape[axis] // 10))
        first = [slice(None)] * 3
        last = [slice(None)] * 3
        first[axis], last[axis] = 0, -1
        seam = 0.5 * (tile[tuple(first)] + tile[tuple(last)])
        for distance in range(margin):
            weight = 0.5 * (1.0 + np.cos(np.pi * distance / margin))
            first[axis], last[axis] = distance, -1 - distance
            tile[tuple(first)] = (1.0 - weight) * tile[tuple(first)] + weight * seam
            tile[tuple(last)] = (1.0 - weight) * tile[tuple(last)] + weight * seam
    return np.clip(tile, 0.0, 1.0)


def _spatial_texture_score(grid: np.ndarray, occupied: np.ndarray) -> float:
    colors = grid[occupied]
    if len(colors) < 100:
        return 0.0
    _, vectors = np.linalg.eigh(np.cov(colors.T))
    signal = grid @ vectors[:, -1]
    signal -= gaussian_filter(signal, sigma=3.0, mode="nearest")
    scores = []
    for axis in (0, 1):
        gradient = np.diff(signal, axis=axis)
        support = ((occupied[1:, :] & occupied[:-1, :]) if axis == 0
                   else (occupied[:, 1:] & occupied[:, :-1]))
        if axis == 0:
            pairs = support[:-1, :] & support[1:, :]
            first, second = gradient[:-1, :][pairs], gradient[1:, :][pairs]
        else:
            pairs = support[:, :-1] & support[:, 1:]
            first, second = gradient[:, :-1][pairs], gradient[:, 1:][pairs]
        if len(first) >= 50 and np.std(first) > 1e-6 and np.std(second) > 1e-6:
            score = float(np.corrcoef(first, second)[0, 1])
            if np.isfinite(score):
                scores.append(score)
    return max(0.0, max(scores, default=0.0))


def detect_visual_texture(uv: np.ndarray, colors: np.ndarray, areas: np.ndarray) -> dict:
    """Detect repeated fine-scale color structure without requiring a periodic tile."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    colors = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
    areas = np.asarray(areas, dtype=np.float64).reshape(-1)
    if not (len(uv) == len(colors) == len(areas)):
        raise ValueError("Texture support coordinates, colors, and areas must align")
    valid = (np.isfinite(uv).all(axis=1) & np.isfinite(colors).all(axis=1)
             & np.isfinite(areas) & (areas > 0))
    uv, colors, areas = uv[valid], np.clip(colors[valid], 0.0, 1.0), areas[valid]
    if len(uv) < 100:
        return {"present": False, "reason": "insufficient_observed_support", "support_count": int(len(uv))}

    raster = _rasterize_support(uv, colors, areas)
    grid, occupied = raster["grid"], raster["occupied"]
    occupied_colors = grid[occupied]
    if len(occupied_colors) < 100 or occupied.mean() < 0.15:
        return {"present": False, "reason": "insufficient_raster_coverage",
                "support_count": int(len(uv)), "occupied_cell_count": int(occupied.sum()),
                "support_cell_fraction": float(occupied.mean())}

    weights = raster["area_grid"][occupied]
    center = np.average(occupied_colors, axis=0, weights=weights)
    centered = occupied_colors - center
    covariance = (centered * weights[:, None]).T @ centered / weights.sum()
    _, vectors = np.linalg.eigh(covariance)
    signal = (grid - center) @ vectors[:, -1]
    residual = signal - gaussian_filter(signal, sigma=3.0, mode="nearest")
    residual_p90 = float(np.percentile(np.abs(residual[occupied]), 90))

    correlations = []
    for axis in (0, 1):
        for lag in range(3, min(13, (occupied.shape[axis] // 2) + 1)):
            first_slice = [slice(None), slice(None)]
            second_slice = [slice(None), slice(None)]
            first_slice[axis], second_slice[axis] = slice(0, -lag), slice(lag, None)
            first_slice, second_slice = tuple(first_slice), tuple(second_slice)
            paired = occupied[first_slice] & occupied[second_slice]
            first, second = residual[first_slice][paired], residual[second_slice][paired]
            if len(first) < 100 or np.std(first) < 1e-6 or np.std(second) < 1e-6:
                continue
            correlation = float(np.corrcoef(first, second)[0, 1])
            if np.isfinite(correlation):
                correlations.append((correlation, axis, lag))
    best_correlation = max(correlations, default=(0.0, -1, 0))

    axis_results = []
    for axis in (0, 1):
        line_densities = []
        for line_index in range(occupied.shape[1 - axis]):
            support = occupied[:, line_index] if axis == 0 else occupied[line_index, :]
            values = residual[:, line_index] if axis == 0 else residual[line_index, :]
            if support.sum() < 20:
                continue
            changes = np.diff(np.r_[False, support, False].astype(np.int8))
            starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
            event_count = support_length = 0
            for start, stop in zip(starts, stops):
                if stop - start < 12:
                    continue
                support_length += stop - start
                peaks, _ = find_peaks(np.abs(values[start:stop]), prominence=0.012, distance=2)
                event_count += len(peaks)
            length_m = support_length * raster["cell_size"]
            if length_m >= 0.6:
                line_densities.append(event_count / length_m)
        if line_densities:
            density = np.asarray(line_densities)
            axis_results.append({
                "axis": axis,
                "eligible_line_count": int(len(density)),
                "median_texture_events_per_m": float(np.median(density)),
                "fraction_lines_with_at_least_4_events_per_m": float(np.mean(density >= 4.0)),
            })

    best_axis = max(axis_results, key=lambda item: (
        item["fraction_lines_with_at_least_4_events_per_m"],
        item["median_texture_events_per_m"],
    ), default=None)
    present = bool(
        occupied.mean() >= 0.50
        and residual_p90 >= 0.01
        and best_correlation[0] >= 0.25
        and best_axis is not None
        and best_axis["eligible_line_count"] >= 8
        and best_axis["median_texture_events_per_m"] >= 2.0
        and best_axis["fraction_lines_with_at_least_4_events_per_m"] >= 0.20
    )
    return {
        "present": present,
        "reason": "coherent_fine_scale_texture" if present else "no_coherent_fine_scale_texture",
        "support_count": int(len(uv)),
        "occupied_cell_count": int(occupied.sum()),
        "support_cell_fraction": float(occupied.mean()),
        "cell_size_m": float(raster["cell_size"]),
        "highpass_color_residual_p90": residual_p90,
        "best_spatial_autocorrelation": float(best_correlation[0]),
        "best_autocorrelation_axis": int(best_correlation[1]),
        "best_autocorrelation_lag_cells": int(best_correlation[2]),
        "axis_metrics": axis_results,
    }


def _visual_model(uv: np.ndarray, colors: np.ndarray, areas: np.ndarray) -> dict:
    raster = _rasterize_support(uv, colors, areas)
    grid, occupied = raster["grid"], raster["occupied"]
    grid, removed = _remove_thin_color_outliers(
        grid, occupied, raster["area_grid"]
    )
    period_u, score_u, region_u = _axis_repeat(grid, occupied, axis=1)
    period_v, score_v, region_v = _axis_repeat(grid, occupied, axis=0)
    texture_score = _spatial_texture_score(grid, occupied)
    texture_evidence = detect_visual_texture(uv, colors, areas)
    modes = _two_color_modes(grid, occupied, raster["area_grid"])
    ny, nx = grid.shape[:2]
    region_px = (0, nx, 0, ny)
    occupied_colors = grid[occupied]
    base_color = np.median(occupied_colors, axis=0)
    spread = float(np.percentile(np.linalg.norm(occupied_colors - base_color, axis=1), 90))

    periodic_axes = (period_v is not None, period_u is not None)
    if period_u is not None or period_v is not None:
        kind = "periodic_tile"
        detected_regions = [region for region in (region_u, region_v) if region is not None]
        region = (
            max(item[0] for item in detected_regions),
            min(item[1] for item in detected_regions),
            max(item[2] for item in detected_regions),
            min(item[3] for item in detected_regions),
        )
        start_x, end_x, start_y, end_y = region
        if end_x <= start_x or end_y <= start_y:
            start_x, end_x, start_y, end_y = 0, nx, 0, ny
        region_px = (start_x, end_x, start_y, end_y)
        width = period_u or (end_x - start_x)
        height = period_v or (end_y - start_y)
        tile = grid[start_y:start_y + height, start_x:start_x + width].copy()
        tile = _make_seamless_tile(tile, periodic_axes)
        labels = None
        palette = None
    elif modes is not None:
        palette = modes["colors"]
        labels = np.argmin(np.linalg.norm(grid[:, :, None] - palette[None, None, :], axis=3), axis=2)
        straight_labels = _straight_color_split(labels, occupied, raster["area_grid"])
        if straight_labels is not None:
            kind = "solid_regions"
            tile = base_color.reshape(1, 1, 3).copy()
            start_x = start_y = 0
            labels = straight_labels
        else:
            if texture_evidence["present"]:
                kind = "repeating_texture"
                ny, nx = grid.shape[:2]
                width, height = min(nx, 64), min(ny, 64)
                start_x, start_y = (nx - width) // 2, (ny - height) // 2
                tile = grid[start_y:start_y + height, start_x:start_x + width].copy()
                tile = _make_seamless_tile(tile, (True, True))
                periodic_axes = (True, True)
            else:
                kind = "plain_color"
                palette = labels = None
                tile = np.clip(base_color, 0.0, 1.0).reshape(1, 1, 3)
                start_x = start_y = 0
                periodic_axes = (False, False)
            if kind == "repeating_texture":
                labels = palette = None
    elif spread > 0.06 and texture_evidence["present"]:
        kind = "repeating_texture"
        ny, nx = grid.shape[:2]
        width, height = min(nx, 64), min(ny, 64)
        start_x, start_y = (nx - width) // 2, (ny - height) // 2
        tile = grid[start_y:start_y + height, start_x:start_x + width].copy()
        tile = _make_seamless_tile(tile, (True, True))
        periodic_axes = (True, True)
        labels = palette = None
    else:
        kind = "plain_color"
        tile = np.clip(base_color, 0.0, 1.0).reshape(1, 1, 3)
        labels = palette = None
        start_x = start_y = 0
        periodic_axes = (False, False)

    info = {
        "kind": kind,
        "base_color_rgb": np.asarray(base_color).tolist(),
        "support_cell_count": int(occupied.sum()),
        "outlier_cells_replaced": int(removed),
        "color_spread_rgb_p90": spread,
        "period_u_scene_units": None if period_u is None else period_u * raster["cell_size"],
        "period_v_scene_units": None if period_v is None else period_v * raster["cell_size"],
        "period_u_confidence": float(score_u),
        "period_v_confidence": float(score_v),
        "spatial_texture_confidence": float(texture_score),
        "tile_shape_pixels": [int(tile.shape[0]), int(tile.shape[1])],
    }
    if kind == "periodic_tile":
        x0, x1, y0, y1 = region_px
        info["pattern_region_uv_bounds"] = [
            float(raster["bounds"][0, 0] + x0 * raster["cell_size"]),
            float(raster["bounds"][0, 0] + x1 * raster["cell_size"]),
            float(raster["bounds"][1, 0] + y0 * raster["cell_size"]),
            float(raster["bounds"][1, 0] + y1 * raster["cell_size"]),
        ]
    if modes is not None:
        info["region_colors_rgb"] = palette.tolist() if palette is not None else modes["colors"].tolist()
        info["color_delta_rgb"] = float(modes["delta_rgb"])
    return {**raster, "grid": grid, "kind": kind, "info": info,
            "base_color": np.asarray(base_color), "palette": palette, "labels": labels,
            "tile": tile, "tile_start": (start_x, start_y), "period": (period_u, period_v),
            "region_px": region_px,
            "periodic_axes": periodic_axes}


def _sample_visual(model: dict, uv: np.ndarray) -> np.ndarray:
    if not len(uv):
        return np.empty((0, 3), dtype=np.float64)
    if model["kind"] == "plain_color":
        return np.tile(model["base_color"], (len(uv), 1))
    bounds, cell = model["bounds"], model["cell_size"]
    nx, ny = model["grid"].shape[1], model["grid"].shape[0]
    ix = np.clip(((uv[:, 0] - bounds[0, 0]) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((uv[:, 1] - bounds[1, 0]) / cell).astype(int), 0, ny - 1)
    outside_support = np.any((uv < bounds[:, 0]) | (uv > bounds[:, 1]), axis=1)
    if model["kind"] == "solid_regions":
        return model["palette"][model["labels"][iy, ix]]
    tile = model["tile"]
    start_x, start_y = model["tile_start"]
    x0, x1, y0, y1 = model.get("region_px", (0, nx, 0, ny))
    inside_region = (ix >= x0) & (ix < x1) & (iy >= y0) & (iy < y1)
    local_x, local_y = ix - start_x, iy - start_y
    periodic_y, periodic_x = model.get("periodic_axes", (True, True))
    tile_x = local_x % tile.shape[1] if periodic_x else np.clip(local_x, 0, tile.shape[1] - 1)
    tile_y = local_y % tile.shape[0] if periodic_y else np.clip(local_y, 0, tile.shape[0] - 1)
    colors = model["grid"][iy, ix].copy()
    colors[inside_region] = tile[tile_y[inside_region], tile_x[inside_region]]
    colors[outside_support] = model["grid"][iy[outside_support], ix[outside_support]]
    return colors


def _fit_relief_line(coordinate: np.ndarray, residual: np.ndarray,
                     areas: np.ndarray) -> dict | None:
    if len(coordinate) < 20:
        return None
    span = float(np.ptp(coordinate))
    if span < 0.24:
        return None
    count = int(np.clip(round(span / 0.025), 16, 256))
    edges = np.linspace(float(coordinate.min()), float(coordinate.max()) + 1e-9, count + 1)
    bins = np.clip(np.searchsorted(edges, coordinate, side="right") - 1, 0, count - 1)
    profile = np.full(count, np.nan)
    for index in np.unique(bins):
        selected = bins == index
        profile[index] = _weighted_median(residual[selected], areas[selected])
    valid = np.flatnonzero(np.isfinite(profile))
    if len(valid) < 12:
        return None
    profile = np.interp(np.arange(count), valid, profile[valid])
    smooth = gaussian_filter1d(profile, 1.0)
    signal = smooth - gaussian_filter1d(smooth, max(2.0, count / 12.0))
    prominence = max(span * 0.005, 0.2 * float(np.ptp(signal)), 1e-4)
    positive, pos_properties = find_peaks(signal, prominence=prominence, distance=2)
    negative, neg_properties = find_peaks(-signal, prominence=prominence, distance=2)
    peaks, properties = ((positive, pos_properties) if len(positive) >= len(negative)
                         else (negative, neg_properties))
    if len(peaks) < 4:
        return None
    positions = (edges[peaks] + edges[peaks + 1]) * 0.5
    spacing = np.diff(positions)
    period = float(np.median(spacing))
    cv = float(np.std(spacing) / max(period, 1e-9))
    if cv > 0.25 or not 0.08 <= period <= 1.5:
        return None
    return {
        "period": period,
        "cv": cv,
        "positions": positions,
        "score": float(np.median(properties["prominences"]) * (1.0 - cv)),
        "amplitude": float(np.median(np.abs(signal[peaks]))),
    }


def _parallelepiped_dimensions(profile: np.ndarray) -> tuple[float, float, float]:
    observed = gaussian_filter1d(np.asarray(profile, dtype=np.float64), 0.65, mode="wrap")
    baseline, peak = np.percentile(observed, [10, 95])
    height = float(peak - baseline)
    if not np.isfinite(height) or height <= 1e-6:
        raise ValueError("Detected periodic relief has no stable peak height")
    observed = np.clip((observed - baseline) / height, 0.0, 1.0)
    count = len(observed)
    peak_phase = int(np.argmax(observed)) / count
    width = float(np.clip(np.mean(observed >= 0.5), 0.2, 0.75))
    return height, peak_phase, width


def _periodic_relief(centers: np.ndarray, areas: np.ndarray, normal: np.ndarray,
                     offset: float, uv: np.ndarray) -> dict:
    plain = {"pattern_type": "simple_and_plain", "confidence": 1.0, "parameters": {}}
    if len(centers) < 60 or float(areas.sum()) <= 0:
        return plain
    residual = centers @ normal + offset
    band_count = min(32, max(3, int(round(np.sqrt(len(centers)) / 3.0))))
    candidates = []
    for axis in range(2):
        cross_axis = 1 - axis
        cross_min, cross_max = np.percentile(uv[:, cross_axis], [1, 99])
        if cross_max - cross_min < 0.12:
            continue
        edges = np.linspace(float(cross_min), float(cross_max), band_count + 1)
        band_ids = np.clip(np.searchsorted(edges, uv[:, cross_axis], side="right") - 1,
                           0, band_count - 1)
        for band_id in range(band_count):
            selected = band_ids == band_id
            candidate = _fit_relief_line(uv[selected, axis], residual[selected], areas[selected])
            if candidate is not None:
                candidates.append({**candidate, "axis": axis, "band": band_id,
                                   "band_area": float(areas[selected].sum()),
                                   "band_edges": edges})

    best = None
    for seed in candidates:
        members = {}
        for candidate in candidates:
            if candidate["axis"] != seed["axis"]:
                continue
            tolerance = max(0.025, 0.12 * seed["period"])
            if abs(candidate["period"] - seed["period"]) <= tolerance:
                previous = members.get(candidate["band"])
                if previous is None or candidate["score"] > previous["score"]:
                    members[candidate["band"]] = candidate
        active = np.zeros(band_count, dtype=bool)
        active[list(members)] = True
        bridged = active.copy()
        bridged[1:-1] |= active[:-2] & active[2:]
        changes = np.diff(np.r_[False, bridged, False].astype(np.int8))
        runs = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
        for start, stop in runs:
            run_members = [members[i] for i in range(start, stop) if i in members]
            if len(run_members) < max(6, int(np.ceil(band_count * 0.30))):
                continue
            if len(run_members) / max(stop - start, 1) < 0.75:
                continue
            score = sum(item["score"] * item["band_area"] for item in run_members)
            if best is None or score > best[0]:
                best = (score, seed["axis"], start, stop, run_members,
                        seed["period"], seed["band_edges"])
    if best is None:
        return plain

    _, axis, band_start, band_stop, members, _, band_edges = best
    cross_axis = 1 - axis
    cross_min, cross_max = np.percentile(uv[:, cross_axis], [1, 99])
    region_cross = [float(band_edges[band_start]), float(band_edges[band_stop])]
    support = ((uv[:, cross_axis] >= region_cross[0]) & (uv[:, cross_axis] <= region_cross[1]))
    selected_bands = {item["band"] for item in members}
    support &= np.isin(
        np.clip(np.searchsorted(band_edges, uv[:, cross_axis], side="right") - 1,
                0, band_count - 1), list(selected_bands)
    )
    selected_uv, selected_residual, selected_areas = uv[support], residual[support], areas[support]
    period = float(np.median([item["period"] for item in members]))
    phase_angles = np.array([
        2.0 * np.pi * np.mod(item["positions"][0], item["period"]) / item["period"]
        for item in members
    ])
    phase_origin = float(
        (np.angle(np.mean(np.exp(1j * phase_angles))) % (2.0 * np.pi))
        * period / (2.0 * np.pi)
    )
    phase = np.mod(selected_uv[:, axis] - phase_origin, period) / period
    phase_bins = np.minimum((phase * 32).astype(int), 31)
    profile = np.full(32, np.nan)
    for index in np.unique(phase_bins):
        chosen = phase_bins == index
        profile[index] = _weighted_median(selected_residual[chosen], selected_areas[chosen])
    valid_profile = np.flatnonzero(np.isfinite(profile))
    if len(valid_profile) < 8:
        return plain
    profile = np.interp(np.arange(32), np.r_[valid_profile - 32, valid_profile, valid_profile + 32],
                        np.tile(profile[valid_profile], 3))
    profile -= _weighted_median(profile, np.ones(len(profile)))

    evidence_bounds = [0.0, 0.0, 0.0, 0.0]
    evidence_bounds[axis * 2:axis * 2 + 2] = np.percentile(selected_uv[:, axis], [1, 99]).tolist()
    evidence_bounds[cross_axis * 2:cross_axis * 2 + 2] = region_cross
    region_bounds = evidence_bounds
    measured_height, peak_phase, width_fraction = _parallelepiped_dimensions(profile)
    amplitude = min(measured_height, MAX_GEOMETRIC_RELIEF_M, period * 0.08)
    if amplitude <= 1e-6:
        return plain
    phase_origin += peak_phase * period
    axis_min, axis_max = region_bounds[axis * 2:axis * 2 + 2]
    anchor = phase_origin + np.floor((axis_min - phase_origin) / period) * period
    regularized_positions = np.arange(anchor, axis_max + period, period)
    primitive_width_fraction = float(np.clip(width_fraction, 0.2, 0.5))
    primitive_width = period * primitive_width_fraction
    regularized_positions = regularized_positions[
        (regularized_positions >= axis_min + primitive_width * 0.5)
        & (regularized_positions <= axis_max - primitive_width * 0.5)
    ]

    confidence = float(np.average([1.0 - item["cv"] for item in members],
                                  weights=[item["band_area"] for item in members]))
    return {
        "pattern_type": "periodic_relief",
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "parameters": {
            "axis": "u" if axis == 0 else "v",
            "period_scene_units": period,
            "phase_origin_scene_units": phase_origin,
            "amplitude_scene_units": amplitude,
            "primitive_shape": "parallelepiped",
            "primitive_width_fraction": primitive_width_fraction,
            "profile_simplified": True,
            "peak_positions": np.concatenate([item["positions"] for item in members]).tolist(),
            "regularized_peak_positions": regularized_positions.tolist(),
            "spacing_cv": float(np.average([item["cv"] for item in members],
                                            weights=[item["band_area"] for item in members])),
            "pattern_region_uv_bounds": region_bounds,
            "evidence_region_uv_bounds": evidence_bounds,
            "coherent_band_count": len(members),
            "primitive_count": int(len(regularized_positions)),
        },
    }


def _boundary_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    if not len(mesh.faces):
        return np.empty((0, 3), dtype=np.float64)
    counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    edges = mesh.edges_unique[counts == 1]
    return np.asarray(mesh.vertices)[np.unique(edges)] if len(edges) else np.empty((0, 3))


def _subdivide(mesh: trimesh.Trimesh, max_edge: float) -> trimesh.Trimesh:
    if not len(mesh.faces) or max_edge <= 0:
        return mesh.copy()
    vertices, faces = trimesh.remesh.subdivide_to_size(
        np.asarray(mesh.vertices), np.asarray(mesh.faces), max_edge=max_edge
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _add_parallelepiped_relief(
    mesh: trimesh.Trimesh, pattern: dict, normal: np.ndarray, offset: float,
    axis_u: np.ndarray, axis_v: np.ndarray, relief_sign: float = 1.0,
) -> tuple[trimesh.Trimesh, int]:
    if pattern["pattern_type"] != "periodic_relief" or not len(mesh.vertices):
        return mesh.copy(), 0
    parameters = pattern["parameters"]
    axis = 0 if parameters["axis"] == "u" else 1
    period = float(parameters["period_scene_units"])
    bounds = np.asarray(parameters["pattern_region_uv_bounds"], dtype=np.float64)
    along_min, along_max = bounds[axis * 2:axis * 2 + 2]
    cross_min, cross_max = bounds[(1 - axis) * 2:(1 - axis) * 2 + 2]
    length = cross_max - cross_min
    height = float(parameters["amplitude_scene_units"])
    if period <= 0 or length <= 0 or height <= 0:
        return mesh.copy(), 0

    along_axis = axis_u if axis == 0 else axis_v
    relief_normal = relief_sign * normal
    cross_axis = np.cross(relief_normal, along_axis)
    cross_axis /= np.linalg.norm(cross_axis)
    frame = np.column_stack((along_axis, cross_axis, relief_normal))
    cross_center = (cross_min + cross_max) * 0.5
    plane_center = (along_axis * 0.0 + (axis_v if axis == 0 else axis_u) * cross_center
                    - normal * offset)
    base_embed = min(0.005, height * 0.1)
    width = period * float(parameters["primitive_width_fraction"])
    primitives = []
    for along_center in parameters["regularized_peak_positions"]:
        if not along_min <= along_center <= along_max:
            continue
        primitive = trimesh.creation.box(extents=[width, length, height + base_embed])
        transform = np.eye(4)
        transform[:3, :3] = frame
        transform[:3, 3] = (
            plane_center + along_axis * along_center
            + relief_normal * ((height - base_embed) * 0.5)
        )
        primitive.apply_transform(transform)
        primitives.append(primitive)
    if not primitives:
        return mesh.copy(), 0
    return trimesh.util.concatenate([mesh.copy(), *primitives]), len(primitives)


def regularize_planar_patches(
    patches: list[dict],
    *,
    boundary_tolerance: float = 0.05,
    color_match_threshold: float = 0.12,
    max_edge: float = 0.15,
    neutral_reference_rgb: tuple[float, float, float] | None = None,
) -> list[dict]:
    """Regularize geometry metadata and synthesize a repeatable plane appearance.

    Input support colors are analyzed, never copied directly to the output mesh.
    Optional scene-wide color adaptation and per-plane lighting normalization are applied
    before synthesis. Strong periodic relief is represented by separate low-poly parallelepipeds
    inside the observed support bounds; the fitted plane remains unchanged.
    """
    white_balance, scene_lighting = _scene_white_balance(patches, neutral_reference_rgb)
    records = []
    prepared = []
    shared_edge = max_edge
    for patch in patches:
        mesh = patch["mesh"]
        normal, axis_u, axis_v, offset = _plane_frame(patch["plane_equation"], patch["mesh"])
        centers = np.asarray(patch.get("support_centers", []), dtype=np.float64).reshape(-1, 3)
        areas = np.asarray(patch.get("support_areas", []), dtype=np.float64).reshape(-1)
        colors = np.asarray(patch.get("support_colors", []), dtype=np.float64).reshape(-1, 3)
        if not (len(centers) == len(areas) == len(colors)):
            raise ValueError(f"Support arrays do not align for plane {patch.get('name', '<unnamed>')}")
        valid = (np.isfinite(centers).all(axis=1) & np.isfinite(areas)
                 & (areas > 0) & np.isfinite(colors).all(axis=1))
        centers, areas, colors = centers[valid], areas[valid], np.clip(colors[valid], 0.0, 1.0)
        uv = np.column_stack((centers @ axis_u, centers @ axis_v)) if len(centers) else np.empty((0, 2))
        pattern = _periodic_relief(centers, areas, normal, offset, uv)
        prepared.append((patch, mesh, normal, axis_u, axis_v, offset,
                         centers, areas, colors, uv, pattern))

    for patch, mesh, normal, axis_u, axis_v, offset, centers, areas, colors, uv, pattern in prepared:
        colors = _preserve_color_highlights(colors * white_balance)
        reg_colors, lighting = _regularize_lighting(colors, uv, areas) if len(colors) else (colors, {})
        visual = _visual_model(uv, reg_colors, areas) if len(reg_colors) else None
        split = _color_split_metadata(visual)
        requested_direction = np.asarray(
            patch.get("relief_direction", normal), dtype=np.float64
        ).reshape(3)
        direction_length = float(np.linalg.norm(requested_direction))
        if not np.isfinite(requested_direction).all() or direction_length <= 1e-10:
            raise ValueError(f"Invalid geometric-relief direction for plane {patch.get('name', '<unnamed>')}")
        direction_dot = float((requested_direction / direction_length) @ normal)
        if abs(direction_dot) < 0.5:
            raise ValueError(
                f"Geometric-relief direction is not normal to plane {patch.get('name', '<unnamed>')}"
            )
        relief_sign = 1.0 if direction_dot > 0 else -1.0
        base_mesh = _subdivide(mesh, shared_edge)
        regularized_mesh, primitive_count = _add_parallelepiped_relief(
            base_mesh, pattern, normal, offset, axis_u, axis_v, relief_sign
        )
        if pattern["pattern_type"] == "periodic_relief":
            pattern["parameters"]["primitive_count"] = primitive_count
        patch["mesh"] = regularized_mesh
        patch["pattern_classification"] = pattern
        patch["color_split"] = split
        patch["regularization"] = {
            "support_triangle_count": int(len(centers)),
            "support_area": float(areas.sum()),
            "scene_white_balance": scene_lighting,
            "lighting": lighting,
            "lighting_corrected": bool(scene_lighting.get("applied", False)
                                        or lighting.get("corrected", False)),
            "visual_surface": visual["info"] if visual else {"kind": "unsupported"},
            "outlier_cells_replaced": int(visual["info"].get("outlier_cells_replaced", 0)) if visual else 0,
            "geometry": ("repeated_parallelepipeds_in_observed_region"
                         if primitive_count else "fitted_plane_preserved"),
            "primitive_count": primitive_count,
            "shared_mesh_edge_m": float(shared_edge),
            "periodic_layout_regularized": pattern["pattern_type"] == "periodic_relief",
            "relief_direction": (relief_sign * normal).tolist(),
        }
        records.append({
            "patch": patch, "normal": normal, "axis_u": axis_u, "axis_v": axis_v,
            "support_centers": centers, "support_areas": areas, "support_colors": reg_colors,
            "support_uv": uv, "visual": visual, "boundary": _boundary_vertices(base_mesh),
        })
        records[-1]["boundary_tree"] = (
            cKDTree(records[-1]["boundary"]) if len(records[-1]["boundary"]) else None
        )

    # Pool touching, uniform patches only; coherent color regions remain local.
    count = len(records)
    adjacency = [[] for _ in records]
    for i in range(count):
        first = records[i]
        if first["patch"]["color_split"]["is_split"] or not len(first["support_colors"]):
            continue
        for j in range(i + 1, count):
            second = records[j]
            if second["patch"]["color_split"]["is_split"] or not len(second["support_colors"]):
                continue
            if not len(first["boundary"]) or not len(second["boundary"]):
                continue
            distance = min(float(first["boundary_tree"].query(second["boundary"])[0].min()),
                           float(second["boundary_tree"].query(first["boundary"])[0].min()))
            if distance > boundary_tolerance:
                continue
            median_a = np.median(first["support_colors"], axis=0) / white_balance
            median_b = np.median(second["support_colors"], axis=0) / white_balance
            luminance_a = float(median_a @ np.array([0.299, 0.587, 0.114]))
            luminance_b = float(median_b @ np.array([0.299, 0.587, 0.114]))
            chroma_a = median_a / max(luminance_a, 1e-6)
            chroma_b = median_b / max(luminance_b, 1e-6)
            same_observed_color = np.linalg.norm(median_a - median_b) <= color_match_threshold
            same_color_in_shadow = (
                np.linalg.norm(chroma_a - chroma_b) <= 0.12
                and 0.5 <= luminance_a / max(luminance_b, 1e-6) <= 2.0
            )
            if same_observed_color or same_color_in_shadow:
                adjacency[i].append(j)
                adjacency[j].append(i)

    visited = np.zeros(count, dtype=bool)
    for start in range(count):
        if visited[start] or not len(records[start]["support_colors"]):
            continue
        queue = deque([start])
        visited[start] = True
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        pooled = np.median(np.vstack([records[i]["support_colors"] for i in component]), axis=0)
        for index in component:
            record = records[index]
            own = np.median(record["support_colors"], axis=0)
            record["support_colors"] = _preserve_color_highlights(
                record["support_colors"] + pooled - own
            )
            record["patch"]["regularization"]["cross_plane_group"] = [
                records[i]["patch"].get("name") for i in component
            ]
            if record["visual"]["kind"] == "plain_color":
                # Uniform-plane pooling must not invent a new spatial split after fitting.
                base = _preserve_color_highlights((record["visual"]["base_color"] + pooled - own)[None])[0]
                record["visual"]["base_color"] = base
                record["visual"]["info"]["base_color_rgb"] = base.tolist()
            else:
                record["visual"] = _visual_model(
                    record["support_uv"], record["support_colors"], record["support_areas"]
                )
            record["patch"]["color_split"] = _color_split_metadata(record["visual"])
            record["patch"]["regularization"]["visual_surface"] = record["visual"]["info"]

    supported = [record for record in records if len(record["support_centers"])]
    if supported:
        all_support_points = np.vstack([item["support_centers"] for item in supported])
        all_support_owners = np.concatenate([
            np.full(len(item["support_centers"]), index, dtype=np.int32)
            for index, item in enumerate(supported)
        ])
        all_support_tree = cKDTree(all_support_points)
        support_trees = [cKDTree(item["support_centers"]) for item in supported]
    for record in records:
        mesh = record["patch"]["mesh"]
        face_centers = np.asarray(mesh.triangles_center, dtype=np.float64)
        if record["visual"] is not None and len(face_centers):
            query_uv = np.column_stack((face_centers @ record["axis_u"],
                                        face_centers @ record["axis_v"]))
            colors = _sample_visual(record["visual"], query_uv)
            if (record["visual"]["kind"] == "solid_regions"
                    and not _surface_color_regions_are_connected(
                        mesh, colors, record["visual"]["palette"]
                    )):
                visual = record["visual"]
                base = visual["base_color"]
                colors = np.tile(base, (len(face_centers), 1))
                visual.update({
                    "kind": "plain_color", "palette": None, "labels": None,
                    "tile": base.reshape(1, 1, 3),
                })
                visual["info"]["kind"] = "plain_color"
                visual["info"]["regions_collapsed_reason"] = "disconnected fitted-surface patches"
                visual["info"].pop("region_colors_rgb", None)
                visual["info"].pop("color_delta_rgb", None)
                record["patch"]["color_split"] = _color_split_metadata(visual)
                record["patch"]["regularization"]["visual_surface"] = visual["info"]
        elif supported and len(face_centers):
            nearest = all_support_tree.query(face_centers, k=1)[1]
            owner_area = np.bincount(
                all_support_owners[nearest],
                weights=np.maximum(mesh.area_faces, 1e-12),
                minlength=len(supported),
            )
            # A supportless connector must continue one neighboring surface, not mosaic per face.
            source_index = int(np.argmax(owner_area))
            source = supported[source_index]
            source_points = support_trees[source_index].query(face_centers, k=1)[1]
            colors = _sample_visual(source["visual"], source["support_uv"][source_points])
        else:
            colors = np.full((len(face_centers), 3), 0.7)
        rgba = np.column_stack((np.rint(np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8),
                                np.full(len(colors), 255, dtype=np.uint8)))
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=rgba)
        record["patch"]["color_split"] = _color_split_metadata(record["visual"])
        record["patch"]["regularization"]["cross_plane_group"] = record["patch"]["regularization"].get(
            "cross_plane_group", [record["patch"].get("name")]
        )
    return patches
