#!/usr/bin/env python3
"""Select an optimal keyframe budget before the COLMAP step based on ARCore trajectory geometry.

Estimates the room's physical dimensions from the camera path:
  - A person capturing stays at least ~80 cm from walls on average.
  - Therefore, the room's horizontal dimensions extend ~1.6 m beyond the camera path's span
    (0.8 m standoff on both opposite sides).
  - Uses the dilated convex hull of the camera path on the horizontal (X, Z) plane to
    accurately estimate floor area for rectangular, angled, or L-shaped walkthroughs.
  - Derives the optimal keyframe budget N = 50 + (8..11) * A_floor and selects a non-redundant,
    well-spaced keyframe stream before COLMAP runs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from shapely.geometry import MultiPoint

# Empirical indoor keyframe constants
BASE_KEYFRAMES = 50.0
KEYFRAMES_PER_M2_LO = 8.0
KEYFRAMES_PER_M2_HI = 11.0
DEFAULT_STANDOFF_M = 0.80  # 80 cm typical standoff from walls

# Safe dimensional floors (even if person stood stationary in the room center)
MIN_ROOM_DIM_M = 2.5
MIN_ROOM_AREA_M2 = 9.0


def rotation_angle_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    """Relative rotation angle between two 3x3 rotation matrices in degrees."""
    r_rel = np.dot(r_a, r_b.T)
    cos_theta = np.clip((float(np.trace(r_rel)) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def estimate_room_extent_from_arcore(
    frames: Sequence[dict[str, Any]],
    standoff_m: float = DEFAULT_STANDOFF_M,
    up_axis: int = 1,
) -> dict[str, float]:
    """Estimate physical room dimensions and floor area from ARCore camera centers.

    In ARCore/OpenGL coordinate conventions, +Y is gravity-aligned vertical.
    The horizontal floor plane is spanned by X and Z (up_axis = 1).
    """
    if not frames:
        raise ValueError("Cannot estimate room extent from an empty frame list")

    centers = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames], dtype=np.float64)

    # Use 1st and 99th percentiles to guard against single-frame tracking glitch outliers
    p_lo = np.percentile(centers, 1.0, axis=0)
    p_hi = np.percentile(centers, 99.0, axis=0)

    # Horizontal coordinate indices
    h_axes = [i for i in range(3) if i != up_axis]
    span_h0 = float(p_hi[h_axes[0]] - p_lo[h_axes[0]])
    span_h1 = float(p_hi[h_axes[1]] - p_lo[h_axes[1]])
    span_vert = float(p_hi[up_axis] - p_lo[up_axis])

    # Each horizontal dimension extends 2 * standoff_m beyond the path (standoff on both sides)
    dim_h0 = max(span_h0 + 2.0 * standoff_m, MIN_ROOM_DIM_M)
    dim_h1 = max(span_h1 + 2.0 * standoff_m, MIN_ROOM_DIM_M)

    # Dilated 2D convex hull of the camera path in the horizontal plane:
    # Naturally models rectangular, angled, or non-convex walkthrough footprints
    pts_2d = centers[:, h_axes]
    if len(pts_2d) >= 3:
        hull = MultiPoint(pts_2d).convex_hull
        room_poly = hull.buffer(standoff_m)
        poly_area = float(room_poly.area)
    else:
        poly_area = dim_h0 * dim_h1

    floor_area = max(poly_area, MIN_ROOM_AREA_M2)

    # Multi-floor estimation: ceiling height is typically ~2.8m per story
    floors = max(1, int(round(span_vert / 2.8))) if span_vert > 4.2 else 1
    total_floor_area = floor_area * float(floors)

    return {
        "camera_span_x": round(span_h0, 3),
        "camera_span_z": round(span_h1, 3),
        "room_dim_x": round(dim_h0, 3),
        "room_dim_z": round(dim_h1, 3),
        "height_m": round(span_vert, 3),
        "standoff_m": round(standoff_m, 2),
        "area_m2": round(floor_area, 2),
        "floors": floors,
        "floor_area_m2": round(total_floor_area, 2),
    }


def compute_keyframe_budget(extent: dict[str, float], total_frames: int) -> tuple[int, int, int]:
    """Compute (target_lo, target_hi, target) keyframe count from room floor area."""
    floor_area = extent["floor_area_m2"]
    lo = int(math.ceil(BASE_KEYFRAMES + KEYFRAMES_PER_M2_LO * floor_area))
    hi = int(math.ceil(BASE_KEYFRAMES + KEYFRAMES_PER_M2_HI * floor_area))

    lo_clamped = min(lo, total_frames)
    hi_clamped = min(max(hi, lo_clamped), total_frames)
    target = (lo_clamped + hi_clamped) // 2
    return lo_clamped, hi_clamped, target


def select_keyframes_arcore(
    frames: Sequence[dict[str, Any]],
    target_count: int | None = None,
    min_translation_m: float = 0.08,
    min_rotation_deg: float = 4.0,
    standoff_m: float = DEFAULT_STANDOFF_M,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a geometrically budgeted, non-redundant keyframe stream prior to COLMAP.

    Returns:
        (selected_frames, metadata_dict)
    """
    n_total = len(frames)
    if n_total <= 1:
        return list(frames), {"extent": {}, "selected": n_total, "total": n_total}

    # 1. Estimate room extent & keyframe budget
    extent = estimate_room_extent_from_arcore(frames, standoff_m=standoff_m)
    lo, hi, computed_target = compute_keyframe_budget(extent, n_total)
    target = target_count if target_count is not None else computed_target

    if n_total <= target:
        return list(frames), {
            "extent": extent,
            "budget": [lo, hi],
            "target": target,
            "selected": n_total,
            "total": n_total,
            "pruned": False,
        }

    centers = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames], dtype=np.float64)
    rots = np.array([np.array(f["transform_matrix"])[:3, :3] for f in frames], dtype=np.float64)

    # 2. First pass: sequential walk with minimum baseline & angular change
    selected_indices = [0]
    for i in range(1, n_total - 1):
        last = selected_indices[-1]
        dist = float(np.linalg.norm(centers[i] - centers[last]))
        ang = rotation_angle_deg(rots[i], rots[last])
        gap = i - last

        # Accept if moved sufficiently, or if gap reaches safety floor (avoid large jumps)
        if (dist >= min_translation_m or ang >= min_rotation_deg) or dist >= 0.40 or ang >= 18.0 or gap >= 25:
            selected_indices.append(i)

    # Always keep final frame to anchor capture completion
    if selected_indices[-1] != n_total - 1:
        selected_indices.append(n_total - 1)

    # 3. If selection is below target and raw frames are available, top up by bisecting the largest gaps
    while len(selected_indices) < target and len(selected_indices) < n_total:
        c = np.array(selected_indices)
        gaps = c[1:] - c[:-1]
        max_gap_idx = int(np.argmax(gaps))
        if gaps[max_gap_idx] <= 1:
            break
        mid = (c[max_gap_idx] + c[max_gap_idx + 1]) // 2
        selected_indices.insert(max_gap_idx + 1, int(mid))

    # 4. If selection exceeds target, greedily prune frames with smallest neighborhood baseline
    while len(selected_indices) > target:
        c = np.array(selected_indices)
        # Compute baseline distance between previous and next frames (k-1 to k+1)
        step_dists = np.linalg.norm(centers[c[2:]] - centers[c[:-2]], axis=1)
        drop_idx = int(np.argmin(step_dists)) + 1
        selected_indices.pop(drop_idx)

    selected_frames = [frames[i] for i in selected_indices]
    sel_centers = centers[selected_indices]
    inter_dists = np.linalg.norm(np.diff(sel_centers, axis=0), axis=1)

    meta = {
        "extent": extent,
        "budget": [lo, hi],
        "target": target,
        "total": n_total,
        "selected": len(selected_frames),
        "pruned": len(selected_frames) < n_total,
        "mean_baseline_m": round(float(np.mean(inter_dists)), 3),
        "min_baseline_m": round(float(np.min(inter_dists)), 3),
        "max_baseline_m": round(float(np.max(inter_dists)), 3),
    }

    return selected_frames, meta
