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
import shapely
from shapely.geometry import MultiPoint, LineString

from floor_bands import count_floor_bands

# Empirical indoor keyframe constants
BASE_KEYFRAMES = 50.0
KEYFRAMES_PER_M2_LO = 8.0
KEYFRAMES_PER_M2_HI = 11.0
DEFAULT_STANDOFF_M = 0.8  # 0.8m typical standoff from walls

# Safe dimensional floors (even if person stood stationary in the room center)
MIN_ROOM_DIM_M = 2.5
MIN_ROOM_AREA_M2 = 9.0


def rotation_angle_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    """Relative rotation angle between two 3x3 rotation matrices in degrees."""
    r_rel = np.dot(r_a, r_b.T)
    cos_theta = np.clip((float(np.trace(r_rel)) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


from typing import Any, Sequence
import numpy as np
from shapely.geometry import LineString


def estimate_room_extent_from_arcore(
    frames: Sequence[dict[str, Any]],
    standoff_m: float = DEFAULT_STANDOFF_M,
    up_axis: int = 1,
    floor_height_threshold_m: float = 2.5,
) -> dict[str, Any]:
    """Estimate physical room dimensions and floor area per floor from ARCore camera centers.

    Returns per-floor breakdowns (x_span, avg_y, z_span, floor_area) and the
    aggregated total floor area.
    """
    if not frames:
        raise ValueError("Cannot estimate room extent from an empty frame list")

    centers = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames], dtype=np.float64)

    # 1st and 99th percentiles guard against tracking glitches
    p_lo = np.percentile(centers, 1.0, axis=0)
    p_hi = np.percentile(centers, 99.0, axis=0)

    # Horizontal coordinate indices (typically X=0, Z=2 if up_axis=1)
    h_axes = [i for i in range(3) if i != up_axis]
    axis_x, axis_z = h_axes[0], h_axes[1]

    # Floor partitioning along gravity axis
    heights_shifted = centers[:, up_axis] - p_lo[up_axis]
    floors_count = count_floor_bands(heights_shifted)

    floor_indices = np.floor(heights_shifted / floor_height_threshold_m).astype(int)
    floor_indices = np.clip(floor_indices, 0, max(floors_count - 1, 0))

    floors_details: list[dict[str, float]] = []

    for k in range(floors_count):
        pts_floor = centers[floor_indices == k]

        if len(pts_floor) == 0:
            floors_details.append({
                "floor_index": k,
                "x_span": 0.0,
                "avg_y": 0.0,
                "z_span": 0.0,
                "floor_area": 0.0,
            })
            continue

        # Spatial spans and mean elevation for the current floor
        x_pts = pts_floor[:, axis_x]
        y_pts = pts_floor[:, up_axis]
        z_pts = pts_floor[:, axis_z]

        # Use percentiles if sufficient points exist to reject local trajectory outliers
        if len(pts_floor) >= 5:
            p_lo_f = np.percentile(pts_floor, 1.0, axis=0)
            p_hi_f = np.percentile(pts_floor, 99.0, axis=0)
            x_span = float(p_hi_f[axis_x] - p_lo_f[axis_x])
            z_span = float(p_hi_f[axis_z] - p_lo_f[axis_z])
        else:
            x_span = float(np.ptp(x_pts))
            z_span = float(np.ptp(z_pts))

        avg_y = float(np.mean(y_pts))

        # 2D continuous polygon area computation via trajectory dilation
        pts_floor_2d = pts_floor[:, [axis_x, axis_z]]
        if len(pts_floor_2d) >= 2:
            path = LineString(pts_floor_2d)
            room_poly = path.buffer(standoff_m, cap_style="round", join_style="round")
            poly_area = float(room_poly.area)
        else:
            # Single camera position
            poly_area = float(np.pi * (standoff_m ** 2))

        floor_area = max(poly_area, MIN_ROOM_AREA_M2)

        floors_details.append({
            "floor_index": k,
            "x_span": round(x_span, 3),
            "avg_y": round(avg_y, 3),
            "z_span": round(z_span, 3),
            "floor_area": round(floor_area, 2),
        })

    total_floor_area = sum(f["floor_area"] for f in floors_details)
    span_x = max((f["x_span"] for f in floors_details), default=0.0)
    span_z = max((f["z_span"] for f in floors_details), default=0.0)
    dim_x = max(span_x + 2.5 * standoff_m, MIN_ROOM_DIM_M)
    dim_z = max(span_z + 2.5 * standoff_m, MIN_ROOM_DIM_M)

    return {
        "camera_span_x": round(span_x, 3),
        "camera_span_z": round(span_z, 3),
        "room_dim_x": round(dim_x, 3),
        "room_dim_z": round(dim_z, 3),
        "standoff_m": round(standoff_m, 2),
        "floors": floors_count,
        "floors_details": floors_details,
        "floor_area_m2": round(total_floor_area, 2),
        "total_floor_area": round(total_floor_area, 2),
    }

def compute_keyframe_budget(extent: dict[str, float], total_frames: int) -> tuple[int, int, int]:
    """Compute (target_lo, target_hi, target) keyframe count from room floor area."""
    floor_area = extent["total_floor_area"]
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
    sharpness_scores: Sequence[float] | np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a geometrically budgeted, non-redundant keyframe stream prior to COLMAP.

    Optionally accepts per-frame sharpness_scores (e.g. Laplacian variance). If provided,
    performs local neighbor swapping to prefer crisp frames over blurred frames when
    geometric displacement is minimal (< 18cm, < 8 deg).

    Returns:
        (selected_frames, metadata_dict)
    """
    n_total = len(frames)
    if n_total <= 1:
        return list(frames), {"extent": {}, "selected": n_total, "total": n_total}

    # 1. Estimate room extent & keyframe budget
    extent = estimate_room_extent_from_arcore(frames, standoff_m=standoff_m)
    lo, hi, computed_target = compute_keyframe_budget(extent, n_total)
    # print(f"Budgeted keyframes: {lo}..{hi}, computed target={computed_target}, total frames={n_total}")
    # input("Press Enter to continue...")   
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

    # Camera look direction is -Z in OpenGL c2w: -rots[:, :, 2]
    # Pitch angle in degrees (positive = looking upward into ceiling/roof)
    look_dirs = -rots[:, :, 2]
    pitches = np.degrees(np.arcsin(np.clip(look_dirs[:, 1], -1.0, 1.0)))

    # 2. First pass: sequential walk with minimum baseline, angular change & gap floor
    selected_indices = [0]
    max_walk_gap = 8
    for i in range(1, n_total - 1):
        last = selected_indices[-1]
        dist = float(np.linalg.norm(centers[i] - centers[last]))
        ang = rotation_angle_deg(rots[i], rots[last])
        gap = i - last

        is_up = pitches[i] > 2.0
        # If sharpness scores are provided, do not pick severely blurred frames (sharpness < 10.0)
        # unless gap reaches safety floor (avoid large jumps)
        is_blurry = False
        if sharpness_scores is not None and len(sharpness_scores) == n_total:
            if float(sharpness_scores[i]) < 10.0:
                is_blurry = True

        # Accept if moved sufficiently, or if gap reaches safety floor (avoid large jumps),
        # or if camera is looking up at the ceiling, or if large angular change occurs
        if gap >= max_walk_gap or ang >= 15.0 or dist >= 0.35:
            selected_indices.append(i)
        elif not is_blurry:
            if (dist >= min_translation_m or ang >= min_rotation_deg) or (is_up and gap >= 2):
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
    max_prune_gap = 6
    MAX_PRUNE_ROT_DEG = 20.0   # Enforce continuous visual overlap (< 25 deg per step)
    MAX_PRUNE_TRANS_M = 0.65   # Enforce continuous baseline (< 65 cm per step)
    while len(selected_indices) > target:
        c = np.array(selected_indices)
        if len(c) <= 2:
            break

        # Combined SE(3) motion between previous and next frames (k-1 to k+1)
        step_trans = np.linalg.norm(centers[c[2:]] - centers[c[:-2]], axis=1)
        step_rots = np.array([rotation_angle_deg(rots[c[k + 2]], rots[c[k]]) for k in range(len(c) - 2)])
        combined = step_trans + 0.02 * step_rots

        # Hard constraints: resulting gap, turn, and translation must NOT break continuity
        resulting_gaps = c[2:] - c[:-2]
        combined[resulting_gaps > max_prune_gap] = float("inf")
        combined[step_rots > MAX_PRUNE_ROT_DEG] = float("inf")
        combined[step_trans > MAX_PRUNE_TRANS_M] = float("inf")

        # Protect upward-looking frames (looking at ceiling/roof)
        # And if sharpness scores are provided, preferentially prune blurry frames (sharpness < 15)
        for k in range(len(c) - 2):
            idx = c[k + 1]
            if pitches[idx] > 2.0:
                combined[k] += 5.0
            if sharpness_scores is not None and len(sharpness_scores) == n_total:
                # Frames with low sharpness score (< 15) are prioritized for pruning
                s_val = float(sharpness_scores[idx])
                if s_val < 15.0:
                    combined[k] -= (15.0 - s_val) * 0.1

        min_cost = float(np.min(combined))
        if min_cost == float("inf"):
            # Cannot prune any more frames without creating a blind spot gap > max_prune_gap
            break

        drop_idx = int(np.argmin(combined)) + 1
        selected_indices.pop(drop_idx)

    # 5. Optional sharpness-aware neighbor swapping
    # If sharpness scores are provided, check if a direct temporal neighbor (within +/- 3 frames)
    # has significantly higher sharpness while maintaining almost identical camera pose (< 18cm, < 8 deg).
    swapped_count = 0
    if sharpness_scores is not None and len(sharpness_scores) == n_total:
        s_scores = np.asarray(sharpness_scores, dtype=np.float64)
        for _ in range(2):
            any_swap = False
            for idx_in_sel in range(1, len(selected_indices) - 1):
                curr = selected_indices[idx_in_sel]
                prev_sel = selected_indices[idx_in_sel - 1]
                next_sel = selected_indices[idx_in_sel + 1]

                curr_sharpness = s_scores[curr]
                best_cand = curr
                best_sharpness = curr_sharpness

                # Search candidate neighbors between prev_sel and next_sel within +/- 5 frames
                for cand in range(max(prev_sel + 1, curr - 5), min(next_sel, curr + 6)):
                    if cand == curr:
                        continue
                    dist = float(np.linalg.norm(centers[curr] - centers[cand]))
                    ang = rotation_angle_deg(rots[curr], rots[cand])

                    if dist <= 0.18 and ang <= 8.0:
                        cand_sharpness = s_scores[cand]
                        if cand_sharpness > best_sharpness * 1.30 and cand_sharpness > curr_sharpness + 4.0:
                            best_sharpness = cand_sharpness
                            best_cand = cand

                if best_cand != curr:
                    selected_indices[idx_in_sel] = best_cand
                    swapped_count += 1
                    any_swap = True
            if not any_swap:
                break

    selected_frames = [frames[i] for i in selected_indices]
    sel_centers = centers[selected_indices]
    inter_dists = np.linalg.norm(np.diff(sel_centers, axis=0), axis=1)
    sel_gaps = np.diff(selected_indices)

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
        "max_frame_gap": int(np.max(sel_gaps)) if len(sel_gaps) else 0,
        "median_frame_gap": float(np.median(sel_gaps)) if len(sel_gaps) else 0.0,
        "upward_frames_kept": int((pitches[selected_indices] > 2.0).sum()),
        "upward_frames_total": int((pitches > 2.0).sum()),
        "sharpness_swapped": swapped_count,
    }
    # print (f"Selected {len(selected_frames)} keyframes from {n_total} total frames, ")
    # input("Press Enter to continue...")   

    return selected_frames, meta
