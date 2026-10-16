#!/usr/bin/env python3
"""
Recognize and Regularize Geometric Patterns on Architectural Planes.

Analyzes each atomic (instance, plane) unit in the scene:
1. Determines plane orientation (Vertical/Wall vs Horizontal/Inclined/Ceiling/Floor).
2. Applies orientation-relevant pattern recognition:
   - Vertical planes: checks only for genuine periodic horizontal bumps (e.g. siding,
     wainscoting rails) or continuous architectural molding; ignores holes, alcoves,
     and abutting furniture/slab noise.
   - Horizontal / inclined planes: checks for 1D periodic beams (flank face analysis,
     autocorrelation fundamental pitch estimation, beam width, extrusion depth) or 2D tiles;
     otherwise classifies as simple and plain.
3. Regularizes detected patterns across the entire atomic instance/plane extent:
   - For beams: computes the complete regular grid of beams spanning the full boundary,
     extends every beam from boundary wall to boundary wall, synthesizes a continuous
     regularized 3D architectural mesh, snaps raw scan mesh vertices to the canonical
     pattern profile, and transfers authentic scan colors via KDTree.
   - For plain surfaces: regularizes directly to clean boundary-conforming planar geometry.
4. Exports regularized meshes and a comprehensive JSON manifest prior to level grouping.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
from scipy.signal import correlate, find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon, mapping
from surface_footprints import observed_footprint, wall_regions, constrain_footprint, frame_footprint, polygons, snap_footprint_to_walls, clean_footprint
from level_footprints import restore_level_footprints
from structural_geometry import (
    beam_surface, close_slab, pattern_frame, perimeter_pattern_axes, snap_small_excursions,
    surface_wall_sections, polygon_triangles,
)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class PatternClassification:
    pattern_type: str  # "beams", "tiles", "horizontal_bumps", "simple_and_plain"
    confidence: float
    orientation: str  # "vertical", "horizontal", "inclined"
    parameters: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""


# ---------------------------------------------------------------------------
# Mesh Helpers
# ---------------------------------------------------------------------------

def compute_face_centers_and_areas(
    vertices: np.ndarray, triangles: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes face centroids and triangle areas."""
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    centers = (v0 + v1 + v2) / 3.0
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    return centers, areas


def binned_samples(coordinates: np.ndarray, values: np.ndarray, edges: np.ndarray) -> List[np.ndarray]:
    """Group half-open bins once, preserving input order and exact per-bin statistics."""
    bins = np.searchsorted(edges, coordinates, side="right")-1
    valid = (bins >= 0) & (bins < len(edges)-1)
    bins = bins[valid]
    order = np.argsort(bins, kind="stable")
    counts = np.bincount(bins, minlength=len(edges)-1)
    return np.split(values[valid][order], np.cumsum(counts)[:-1])


def weighted_quantile(values, weights, fraction):
    order = np.argsort(values, kind='stable')
    ordered = np.asarray(values)[order]
    cumulative = np.cumsum(np.asarray(weights)[order])
    return float(ordered[min(len(ordered)-1, np.searchsorted(cumulative, fraction*cumulative[-1]))])


def sample_colors_kdtree(
    source_pts: np.ndarray, source_colors: np.ndarray, target_pts: np.ndarray
) -> np.ndarray:
    """Transfers colors from source points to target points via nearest neighbor KDTree."""
    if len(source_pts) == 0 or len(target_pts) == 0:
        return np.ones((len(target_pts), 3), dtype=np.float64) * 0.8
    tree = cKDTree(source_pts)
    _, idxs = tree.query(target_pts, k=1)
    return source_colors[idxs]


def export_ply(
    file_path: Path | str,
    vertices: np.ndarray,
    triangles: np.ndarray,
    vertex_colors: np.ndarray | None = None,
) -> None:
    """Exports little-endian binary PLY mesh with vertex normals and colors."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
    if vertex_colors is not None:
        if vertex_colors.dtype == np.uint8:
            cols = vertex_colors.astype(np.float64) / 255.0
        else:
            cols = np.clip(vertex_colors.astype(np.float64), 0.0, 1.0)
        mesh.vertex_colors = o3d.utility.Vector3dVector(cols)
    mesh.compute_vertex_normals()

    written = o3d.io.write_triangle_mesh(
        str(file_path),
        mesh,
        write_ascii=False,
        compressed=False,
        write_vertex_normals=True,
        write_vertex_colors=vertex_colors is not None,
    )
    if not written:
        raise OSError(f"Failed to write regularized mesh: {file_path}")


# ---------------------------------------------------------------------------
# Pattern Recognition Algorithms
# ---------------------------------------------------------------------------

def analyze_vertical_wall_pattern(
    face_centers: np.ndarray,
    face_areas: np.ndarray,
    tri_normals: np.ndarray,
    plane_normal: np.ndarray,
    plane_offset: float,
    inward_normal: np.ndarray,
    u_horiz: np.ndarray,
    bounds_u: Tuple[float, float],
    bounds_y: Tuple[float, float],
) -> PatternClassification:
    """
    Analyzes vertical wall plane for horizontal patterns (e.g. horizontal ribs, shiplap, moldings).
    Ignores holes (negative residuals), random noise, and abutting furniture/slabs.
    """
    if len(face_centers) < 200:
        return PatternClassification(
            pattern_type="simple_and_plain",
            confidence=0.95,
            orientation="vertical",
            summary="Wall segment has insufficient triangles",
        )

    # Protrusion into room
    protrusion = np.dot(face_centers, inward_normal) - np.dot(-plane_offset * plane_normal, inward_normal)
    y_coords = face_centers[:, 1]
    u_coords = np.dot(face_centers, u_horiz)

    total_u_span = bounds_u[1] - bounds_u[0]
    if total_u_span < 0.3:
        return PatternClassification(
            pattern_type="simple_and_plain",
            confidence=0.95,
            orientation="vertical",
            summary="Wall segment too narrow for horizontal pattern",
        )

    # Discard holes (protrusion < 0) and filter for points near wall plane
    valid_mask = (protrusion >= -0.05) & (protrusion <= 0.25)
    y_sub = y_coords[valid_mask]
    u_sub = u_coords[valid_mask]
    p_sub = protrusion[valid_mask]

    # Analyze median profile along Y in 2cm bins
    y_bins = np.arange(bounds_y[0], bounds_y[1] + 0.02, 0.02)
    y_centers = (y_bins[:-1] + y_bins[1:]) / 2
    med_prot = []
    valid_yc = []

    for i, samples in enumerate(binned_samples(y_sub, p_sub, y_bins)):
        if len(samples) >= 30:
            valid_yc.append(y_centers[i])
            med_prot.append(float(np.median(samples)))

    valid_yc = np.array(valid_yc)
    med_prot = np.array(med_prot)

    if len(valid_yc) < 20:
        return PatternClassification(
            pattern_type="simple_and_plain",
            confidence=0.95,
            orientation="vertical",
            summary="Wall vertical profile too sparse",
        )

    pks, props = find_peaks(med_prot, prominence=0.035, width=(1, 8))

    continuous_bumps = []
    for p in pks:
        y_pk = valid_yc[p]
        m_pk = (y_sub >= y_pk - 0.03) & (y_sub <= y_pk + 0.03) & (p_sub >= 0.025)
        if np.sum(m_pk) < 20:
            continue
        u_pts = u_sub[m_pk]
        u_span_covered = (np.max(u_pts) - np.min(u_pts)) / total_u_span
        u_grid = np.linspace(bounds_u[0], bounds_u[1], 15)
        grid_occupied = sum(
            1 for g in range(len(u_grid) - 1)
            if np.any((u_pts >= u_grid[g]) & (u_pts < u_grid[g + 1]))
        ) / (len(u_grid) - 1)

        if u_span_covered >= 0.80 and grid_occupied >= 0.75:
            if med_prot[p] <= 0.06:
                continuous_bumps.append({
                    "y_center": round(float(y_pk), 3),
                    "protrusion_m": round(float(med_prot[p]), 3),
                    "prominence_m": round(float(props["prominences"][np.where(pks == p)[0][0]]), 3),
                    "horizontal_coverage": round(float(u_span_covered), 3),
                })

    if len(continuous_bumps) >= 2:
        y_pos = [b["y_center"] for b in continuous_bumps]
        dy = np.diff(y_pos)
        cv = float(np.std(dy) / max(np.mean(dy), 1e-4)) if len(dy) > 1 else 0.0
        if cv <= 0.35:
            return PatternClassification(
                pattern_type="horizontal_bumps",
                confidence=0.85,
                orientation="vertical",
                parameters={
                    "bumps_count": len(continuous_bumps),
                    "bumps": continuous_bumps,
                    "mean_spacing_m": round(float(np.mean(dy)), 3),
                },
                summary=f"Detected periodic horizontal bumps ({len(continuous_bumps)} ridges, spacing={np.mean(dy)*100:.1f}cm)",
            )

    return PatternClassification(
        pattern_type="simple_and_plain",
        confidence=0.96,
        orientation="vertical",
        summary="Flat vertical plane with no repeating horizontal pattern",
    )


def analyze_ceiling_floor_pattern(
    face_centers: np.ndarray,
    face_areas: np.ndarray,
    tri_normals: np.ndarray,
    plane_normal: np.ndarray,
    plane_offset: float,
    inward_normal: np.ndarray,
    is_ceiling: bool,
    perimeter_axes: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    profile_axis: Optional[np.ndarray] = None,
) -> PatternClassification:
    """
    Analyzes horizontal or inclined planes (ceilings and floors) for beams or tiles.
    Uses flank face analysis followed by autocorrelation to discover the fundamental period.
    """
    n_plane = plane_normal / np.linalg.norm(plane_normal)
    num_faces = len(face_centers)

    dot_face_plane = np.dot(tri_normals, n_plane)
    flank_mask = np.abs(dot_face_plane) < 0.40

    flank_count = int(np.sum(flank_mask))
    flank_ratio = float(face_areas[flank_mask].sum()/max(face_areas.sum(), 1e-12))

    if flank_ratio < 0.05:
        return PatternClassification(
            pattern_type="simple_and_plain",
            confidence=0.95,
            orientation="horizontal" if abs(n_plane[1]) > 0.8 else "inclined",
            summary=f"Flat plane with negligible flank relief ({flank_ratio*100:.1f}% flank faces)",
        )

    flank_normals = tri_normals[flank_mask]
    flank_proj = flank_normals - np.outer(np.dot(flank_normals, n_plane), n_plane)
    lens = np.linalg.norm(flank_proj, axis=1, keepdims=True)
    valid_lens = lens.ravel() > 1e-4
    if not np.any(valid_lens):
        return PatternClassification(
            pattern_type="simple_and_plain",
            confidence=0.90,
            orientation="horizontal" if abs(n_plane[1]) > 0.8 else "inclined",
            summary="Flank normals have insufficient planar projection",
        )

    flank_proj = flank_proj[valid_lens] / lens[valid_lens]

    ref_dir = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(n_plane, ref_dir)) > 0.9:
        ref_dir = np.array([1.0, 0.0, 0.0])
    u_ref = np.cross(n_plane, ref_dir)
    u_ref /= np.linalg.norm(u_ref)
    v_ref = np.cross(n_plane, u_ref)
    v_ref /= np.linalg.norm(v_ref)

    dot_u = np.dot(flank_proj, u_ref)
    dot_v = np.dot(flank_proj, v_ref)
    angles = np.arctan2(dot_v, dot_u)
    angles_mod = np.mod(angles, np.pi)

    hist, b_edges = np.histogram(angles_mod, bins=180, weights=face_areas[flank_mask][valid_lens])
    peak_idx = int(np.argmax(hist))
    flank_angle = float((b_edges[peak_idx] + b_edges[peak_idx + 1]) / 2.0)

    u_trans = np.cos(flank_angle) * u_ref + np.sin(flank_angle) * v_ref
    u_trans -= np.dot(u_trans, n_plane) * n_plane
    u_trans /= np.linalg.norm(u_trans)

    if profile_axis is not None:
        u_trans = np.array(profile_axis, dtype=float, copy=True)
        u_trans -= (u_trans @ n_plane)*n_plane
        u_trans /= np.linalg.norm(u_trans)
    v_beam = np.cross(n_plane, u_trans)
    v_beam /= np.linalg.norm(v_beam)

    inward_sign = 1.0 if np.dot(inward_normal, n_plane) > 0 else -1.0
    res = inward_sign * (np.dot(face_centers, n_plane) + plane_offset)
    s_coords = np.dot(face_centers, u_trans)

    max_h = 0.35 if not is_ceiling else 0.40
    valid_pts = (np.abs(res) <= max_h) & ((tri_normals @ n_plane > .7))
    s_val = s_coords[valid_pts]
    res_val = res[valid_pts]
    t_val = (face_centers @ v_beam)[valid_pts]
    area_val = face_areas[valid_pts]

    if len(s_val) < 3 or face_areas[valid_pts].sum() < .01:
        return PatternClassification(
            pattern_type="simple_and_plain",
            confidence=0.88,
            orientation="horizontal" if abs(n_plane[1]) > 0.8 else "inclined",
            summary="Insufficient inlier faces for beam profile extraction",
        )

    # 1D Spatial Profile & Autocorrelation
    bins = np.arange(s_val.min(), s_val.max(), 0.01)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    if len(bins) < 4:
        return PatternClassification('simple_and_plain', .9, 'horizontal', summary='Insufficient spatial extent for repetition')
    samples = binned_samples(s_val, res_val, bins)
    weights = binned_samples(s_val, face_areas[valid_pts], bins)
    min_area = max(face_areas[valid_pts].sum()/len(samples)*.02, 1e-10)
    sig = np.array([weighted_quantile(values, area, .5) if area.sum() >= min_area else 0.
                    for values, area in zip(samples, weights)])
    baseline = gaussian_filter1d(sig, sigma=20., mode='nearest')
    sig_centered = sig-baseline

    corr = correlate(sig_centered, sig_centered, mode="full")
    lags = np.arange(-len(sig_centered) + 1, len(sig_centered)) * 0.01

    lag_mask = (lags >= 0.12) & (lags <= 1.50)
    lag_sub = lags[lag_mask]
    corr_sub = corr[lag_mask]

    corr_pks, _ = find_peaks(corr_sub, prominence=0.03 * np.max(corr_sub))
    if len(corr_pks) == 0:
        return PatternClassification(
            pattern_type="simple_and_plain",
            confidence=0.90,
            orientation="horizontal" if abs(n_plane[1]) > 0.8 else "inclined",
            summary="No periodic autocorrelation detected (plain surface)",
        )

    best_idx = corr_pks[np.argmax(corr_sub[corr_pks])]
    best_lag = float(lag_sub[best_idx])
    max_corr = float(corr_sub[best_idx])

    for p in corr_pks:
        lag_cand = float(lag_sub[p])
        if lag_cand < best_lag * 0.75 and corr_sub[p] >= 0.40 * max_corr:
            ratio = round(best_lag / lag_cand)
            if ratio >= 2 and abs(ratio * lag_cand - best_lag) <= 0.06:
                best_lag = lag_cand
                break

    est_period = float(best_lag)

    min_dist_pts = int(round(est_period * 0.60 / 0.01))
    pos_res = sig_centered[sig_centered > 0.01]
    ref_h = float(np.percentile(pos_res, 60)) if len(pos_res) > 0 else 0.03
    peaks, _ = find_peaks(sig_centered, height=max(0.02, ref_h * 0.6), distance=min_dist_pts, prominence=0.012)

    if len(peaks) < 3:
        return PatternClassification(
            pattern_type="simple_and_plain",
            confidence=0.88,
            orientation="horizontal" if abs(n_plane[1]) > 0.8 else "inclined",
            summary=f"Detected only {len(peaks)} peaks, insufficient for repeating beam pattern",
        )

    peak_s = bin_centers[peaks]

    widths = []
    refined_positions = []
    win = int(round(est_period / 0.02))
    for p in peaks:
        h = sig[p]
        base_l = float(np.min(sig[max(0, p - win):p + 1]))
        base_r = float(np.min(sig[p:min(len(sig), p + win + 1)]))
        base = 0.5 * (base_l + base_r)
        half = base + 0.5 * (h - base)
        l, r = p, p
        while l > 0 and sig[l] > half:
            l -= 1
        while r < len(sig) - 1 and sig[r] > half:
            r += 1
        widths.append(float(bin_centers[r] - bin_centers[l]))
        refined_positions.append(float((bin_centers[r]+bin_centers[l])/2))
    peak_s = np.array(refined_positions)

    avg_w = float(np.clip(np.median(widths), 0.05, est_period * 0.60))

    peak_depths = []
    for s_pk in peak_s:
        core_m = np.abs(s_val - s_pk) <= (avg_w / 2.0)
        if np.any(core_m):
            difference = s_val-s_pk
            adjacent = []
            for side in (-1., 1.):
                gap = (side*difference > avg_w*.6) & (side*difference < est_period*.45)
                if not np.any(gap):
                    continue
                # Match along the beam so slope/scan warp is not mistaken for relief depth.
                tree = cKDTree(t_val[gap, None])
                distance, nearest = tree.query(t_val[core_m, None])
                adjacent.append((res_val[gap][nearest], distance < min(.15, est_period*.5)))
            if adjacent:
                baseline = np.mean([value for value, _ in adjacent], axis=0)
                supported = np.logical_and.reduce([valid for _, valid in adjacent])
                if np.any(supported):
                    peak_depths.append(weighted_quantile((res_val[core_m]-baseline)[supported],
                                                       area_val[core_m][supported], .5))

    avg_h = float(np.clip(np.median(peak_depths), .02, max_h)) if peak_depths else float(np.percentile(sig[peaks], 85))

    phases = np.mod(peak_s, est_period)
    anchor_phase = float(np.angle(np.mean(np.exp(2j*np.pi*phases/est_period)))*est_period/(2*np.pi) % est_period)
    gap_m = est_period - avg_w
    if perimeter_axes is not None:
        measured_u = u_trans
        u_trans, v_beam = perimeter_pattern_axes(n_plane, v_beam, perimeter_axes)
        # Rotate around scan support without widening beams by averaging a rotated profile.
        pivot = np.average(face_centers, axis=0, weights=face_areas)
        anchor_phase = float((anchor_phase + pivot @ (u_trans-measured_u)) % est_period)
        peak_s = pivot @ u_trans + (peak_s-pivot @ measured_u)*(measured_u @ u_trans)

    return PatternClassification(
        pattern_type="beams",
        confidence=0.94,
        orientation="horizontal" if abs(n_plane[1]) > 0.8 else "inclined",
        parameters={
            "periodicity_score": float(max_corr/max(corr[len(sig_centered)-1], 1e-12)),
            "period_m": round(est_period, 3),
            "beam_width_m": round(avg_w, 3),
            "beam_depth_m": round(avg_h, 3),
            "space_between_beams_m": round(gap_m, 3),
            "transverse_axis_u": [float(x) for x in u_trans],
            "beam_longitudinal_axis_v": [float(x) for x in v_beam],
            "anchor_phase_s0": round(anchor_phase, 4),
            "detected_peaks_count": int(len(peaks)),
            "detected_peak_positions": [round(float(s), 3) for s in peak_s],
        },
        summary=(
            f"Beams detected: period={est_period*100:.1f}cm, width={avg_w*100:.1f}cm, "
            f"depth={avg_h*100:.1f}cm, gap={gap_m*100:.1f}cm"
        ),
    )


# ---------------------------------------------------------------------------
# Regularization Geometry Synthesis & Full Extension
# ---------------------------------------------------------------------------

def regularize_beams_seamless_plane(
    plane_normal: np.ndarray,
    plane_offset: float,
    inward_normal: np.ndarray,
    u_trans: np.ndarray,
    v_beam: np.ndarray,
    period: float,
    width: float,
    depth: float,
    anchor_s0: float,
    s_min: float,
    s_max: float,
    get_t_bounds_fn: Callable[[float], Tuple[float, float]],
    boundary_polygon_ab: Optional[List[List[float]]] = None,
    nA: Optional[np.ndarray] = None,
    nB: Optional[np.ndarray] = None,
    wall_offsets: Optional[Dict[str, float]] = None,
    source_verts: Optional[np.ndarray] = None,
    source_colors: Optional[np.ndarray] = None,
    default_color: Tuple[float, float, float] = (0.28, 0.16, 0.10),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthesizes a continuous, seamless regularized architectural mesh for the entire plane.
    Extends every regular beam from boundary wall to boundary wall across the full span [s_min, s_max].
    Centers the pattern around the scan plane using duty cycle d_c = width / period.
    Clips substrate and beam soffits to the footprint and connects internal flanks.
    The outer relief boundary remains open until slab skirts or walls are added.
    """
    n_plane, p0, _, u_trans, v_beam = pattern_frame(
        plane_normal, plane_offset, inward_normal, u_trans)
    if boundary_polygon_ab is not None and nA is not None and nB is not None:
        ab = snap_small_excursions(boundary_polygon_ab, wall_offsets or {})
        footprint_xz = ab[:, :1]*nA[[0, 2]] + ab[:, 1:]*nB[[0, 2]]
    else:
        t0, t1 = get_t_bounds_fn(s_min)
        t2, t3 = get_t_bounds_fn(s_max)
        st = np.array([[s_min, t0], [s_max, t2], [s_max, t3], [s_min, t1]])
        xyz = p0 + st[:, :1]*u_trans + st[:, 1:]*v_beam
        footprint_xz = xyz[:, [0, 2]]
    verts_arr, tris_arr = beam_surface(
        plane_normal, plane_offset, inward_normal, u_trans, period, width, depth,
        anchor_s0, footprint_xz)
    if source_verts is not None and source_colors is not None and len(source_colors) == len(source_verts):
        colors_arr = sample_colors_kdtree(source_verts, source_colors, verts_arr)
    else:
        colors_arr = np.tile(default_color, (len(verts_arr), 1))
    return verts_arr, tris_arr, colors_arr


def regularize_beams_geometry(
    plane_normal: np.ndarray,
    plane_offset: float,
    inward_normal: np.ndarray,
    u_trans: np.ndarray,
    v_beam: np.ndarray,
    period: float,
    width: float,
    depth: float,
    anchor_s0: float,
    bounds_s: Tuple[float, float],
    bounds_t: Tuple[float, float],
    boundary_polygon_st: Optional[np.ndarray] = None,
    source_verts: Optional[np.ndarray] = None,
    source_colors: Optional[np.ndarray] = None,
    default_color: Tuple[float, float, float] = (0.28, 0.16, 0.10),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible wrapper for regularize_beams_seamless_plane."""
    return regularize_beams_seamless_plane(
        plane_normal=plane_normal,
        plane_offset=plane_offset,
        inward_normal=inward_normal,
        u_trans=u_trans,
        v_beam=v_beam,
        period=period,
        width=width,
        depth=depth,
        anchor_s0=anchor_s0,
        s_min=bounds_s[0],
        s_max=bounds_s[1],
        get_t_bounds_fn=lambda s: bounds_t,
        source_verts=source_verts,
        source_colors=source_colors,
        default_color=default_color,
    )


def regularize_scanned_mesh_to_pattern(
    raw_verts: np.ndarray,
    raw_tris: np.ndarray,
    plane_normal: np.ndarray,
    plane_offset: float,
    inward_normal: np.ndarray,
    u_trans: np.ndarray,
    v_beam: np.ndarray,
    period: float,
    width: float,
    depth: float,
    anchor_s0: float,
) -> np.ndarray:
    """
    Regularizes every vertex of the scanned instance mesh:
    - Points on beam bottom faces are snapped to the canonical depth h.
    - Points on beam flanks are aligned to the straight flank planes.
    - Points in gaps are flattened to the base plane substrate (r = 0).
    """
    n_plane = plane_normal / np.linalg.norm(plane_normal)
    n_plane, p0, n_inward, u_trans, v_beam = pattern_frame(
        plane_normal, plane_offset, inward_normal, u_trans)

    diff = raw_verts - p0
    s = np.dot(diff, u_trans)
    t = np.dot(diff, v_beam)
    r = np.dot(diff, n_inward)

    k = np.round((s - anchor_s0) / period)
    s_center = anchor_s0 + k * period
    delta_s = s - s_center

    on_bottom = (np.abs(delta_s) <= width / 2.0) & (r > 0.03)
    on_flank = (
        (np.abs(delta_s) > width / 2.0)
        & (np.abs(delta_s) <= width / 2.0 + 0.035)
        & (r > 0.02)
        & (r < depth - 0.01)
    )
    on_base = ~on_bottom & ~on_flank

    s_reg = s.copy()
    r_reg = r.copy()

    d_c = float(np.clip(width / max(period, 1e-4), 0.05, 0.95))
    r_gap = float(-d_c * depth)
    r_beam = float((1.0 - d_c) * depth)

    r_reg[on_bottom] = r_beam
    r_reg[on_base] = r_gap
    s_reg[on_flank] = s_center[on_flank] + np.sign(delta_s[on_flank]) * (width / 2.0)

    reg_verts = p0 + s_reg[:, None] * u_trans + t[:, None] * v_beam + r_reg[:, None] * n_inward
    return reg_verts


def regularize_plain_geometry(
    plane_normal: np.ndarray,
    plane_offset: float,
    quad_corners_3d: Optional[List[List[float]]] = None,
    bounds_u: Optional[Tuple[float, float]] = None,
    bounds_v: Optional[Tuple[float, float]] = None,
    u_axis: Optional[np.ndarray] = None,
    v_axis: Optional[np.ndarray] = None,
    source_verts: Optional[np.ndarray] = None,
    source_colors: Optional[np.ndarray] = None,
    default_color: Tuple[float, float, float] = (0.75, 0.72, 0.68),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates regularized flat planar quad for simple and plain planes."""
    n_plane = plane_normal / np.linalg.norm(plane_normal)
    p0 = -plane_offset * n_plane

    if quad_corners_3d is not None and len(quad_corners_3d) == 4:
        verts_arr = np.array(quad_corners_3d, dtype=np.float64)
    else:
        u_min, u_max = bounds_u
        v_min, v_max = bounds_v
        c0 = p0 + u_min * u_axis + v_min * v_axis
        c1 = p0 + u_max * u_axis + v_min * v_axis
        c2 = p0 + u_max * u_axis + v_max * v_axis
        c3 = p0 + u_min * u_axis + v_max * v_axis
        verts_arr = np.array([c0, c1, c2, c3], dtype=np.float64)

    tris_arr = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    if source_verts is not None and source_colors is not None and len(source_verts) > 0:
        colors_arr = sample_colors_kdtree(source_verts, source_colors, verts_arr)
    else:
        colors_arr = np.tile(default_color, (len(verts_arr), 1))

    return verts_arr, tris_arr, colors_arr


def regularize_floor_polygon_geometry(
    poly_ab: List[List[float]],
    plane_normal: np.ndarray,
    plane_offset: float,
    nA: np.ndarray,
    nB: np.ndarray,
    source_verts: Optional[np.ndarray] = None,
    source_colors: Optional[np.ndarray] = None,
    default_color: Tuple[float, float, float] = (0.75, 0.72, 0.68),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Triangulates an architectural 2D floor polygon in canonical (dA, dB) coordinates,
    projects it onto the fitted 3D floor plane, ensures upward normals, and transfers colors.
    """
    n_plane = plane_normal / np.linalg.norm(plane_normal)
    pts = np.array(poly_ab)
    tris = ear_clip_polygon(pts)

    # Project to 3D on plane:
    # X = dA * nA[0] + dB * nB[0]
    # Z = dA * nA[2] + dB * nB[2]
    # n_plane[0]*X + n_plane[1]*Y + n_plane[2]*Z + plane_offset = 0
    # => Y = -(n_plane[0]*X + n_plane[2]*Z + plane_offset) / n_plane[1]
    dA_v = pts[:, 0]
    dB_v = pts[:, 1]
    x_v = dA_v * nA[0] + dB_v * nB[0]
    z_v = dA_v * nA[2] + dB_v * nB[2]
    denom_y = n_plane[1] if abs(n_plane[1]) > 1e-4 else 1.0
    y_v = -(n_plane[0] * x_v + n_plane[2] * z_v + plane_offset) / denom_y
    verts = np.column_stack([x_v, y_v, z_v])

    # Check normal orientation: floors must face UP (normal_y > 0)
    e1 = verts[tris[:, 1]] - verts[tris[:, 0]]
    e2 = verts[tris[:, 2]] - verts[tris[:, 0]]
    face_norms = np.cross(e1, e2)
    if np.mean(face_norms[:, 1]) < 0:
        tris = tris[:, [0, 2, 1]]

    if source_verts is not None and source_colors is not None and len(source_verts) > 0:
        colors = sample_colors_kdtree(source_verts, source_colors, verts)
    else:
        colors = np.tile(default_color, (len(verts), 1))

    return verts, tris, colors


def ear_clip_polygon(vertices: np.ndarray) -> np.ndarray:
    """Robust ear-clipping triangulation for simple 2D polygons."""
    poly = Polygon(vertices)
    n = len(vertices)
    indices = list(range(n))
    tris = []

    if not poly.exterior.is_ccw:
        indices.reverse()

    def is_ear(i, prev_i, next_i, rem_indices):
        p_prev = vertices[prev_i]
        p_curr = vertices[i]
        p_next = vertices[next_i]
        cross = (p_curr[0] - p_prev[0]) * (p_next[1] - p_curr[1]) - (p_curr[1] - p_prev[1]) * (p_next[0] - p_curr[0])
        if cross <= 1e-9:
            return False
        tri_poly = Polygon([p_prev, p_curr, p_next])
        for idx in rem_indices:
            if idx in (prev_i, i, next_i):
                continue
            if tri_poly.contains(Point(vertices[idx])):
                return False
        return True

    while len(indices) > 2:
        ear_found = False
        m = len(indices)
        for j in range(m):
            prev_j = (j - 1) % m
            next_j = (j + 1) % m
            i = indices[j]
            prev_i = indices[prev_j]
            next_i = indices[next_j]
            if is_ear(i, prev_i, next_i, indices):
                tris.append([prev_i, i, next_i])
                indices.pop(j)
                ear_found = True
                break
        if not ear_found:
            raise ValueError("Cannot triangulate invalid or degenerate structural polygon")

    return np.array(tris, dtype=np.int32)


def parse_args():
    parser = argparse.ArgumentParser(description='Regularize manifest-defined structural surfaces and finite wall junctions.')
    parser.add_argument('--instances-dir', type=Path, default=Path('backend/current_scene/06_mesh_elaboration/extracted_structural_instances'))
    parser.add_argument('--output-dir', type=Path, default=Path('backend/current_scene/06_mesh_elaboration/extracted_structural_instances'))
    parser.add_argument('--scene-mesh', type=Path)
    return parser.parse_args()


def detect_surface_pattern(centers, areas, normals, equation, axes, is_ceiling):
    n, d = equation[:3], equation[3]
    measured = analyze_ceiling_floor_pattern(centers, areas, normals, n, d, n, is_ceiling)
    candidates = []
    for edge in axes:
        u, _ = perimeter_pattern_axes(n, edge, axes)
        classification = analyze_ceiling_floor_pattern(centers, areas, normals, n, d, n, is_ceiling, profile_axis=u)
        if classification.pattern_type == 'beams':
            candidates.append(classification)
    if not candidates:
        return measured
    aligned = max(candidates, key=lambda c: c.parameters['periodicity_score'])
    if measured.pattern_type != 'beams':
        return aligned
    old_v = np.array(measured.parameters['beam_longitudinal_axis_v'])
    new_v = np.array(aligned.parameters['beam_longitudinal_axis_v'])
    near = abs(old_v @ new_v) >= np.cos(np.deg2rad(5.))
    if near or aligned.parameters['periodicity_score'] >= measured.parameters['periodicity_score']:
        return aligned
    return measured


def planar_surface(footprint, equation, upward):
    xy, faces = polygon_triangles(footprint)
    y = -(equation[0]*xy[:, 0]+equation[2]*xy[:, 1]+equation[3])/equation[1]
    vertices = np.column_stack([xy[:, 0], y, xy[:, 1]])
    normals = np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]], vertices[faces[:, 2]]-vertices[faces[:, 0]])
    if (np.mean(normals[:, 1]) > 0) != upward:
        faces = faces[:, ::-1]
    return vertices, faces


def generate_surface(unit, footprint=None):
    footprint = unit['footprint'] if footprint is None else footprint
    equation = unit['equation']
    pattern = unit['classification']
    if unit['preserve_source']:
        return unit['vertices'].copy(), unit['faces'].copy()
    if pattern.pattern_type == 'beams':
        p = pattern.parameters
        return beam_surface(equation[:3], equation[3], equation[:3],
                            np.array(p['transverse_axis_u']), p['period_m'], p['beam_width_m'],
                            p['beam_depth_m'], p['anchor_phase_s0'], footprint)
    return planar_surface(footprint, equation, unit['role'] == 'floor')


def paint_surface(unit, vertices):
    return unit['colors'][unit['color_tree'].query(vertices, k=1)[1]]


def plane_y(equation, x, z):
    return -(equation[0]*x+equation[2]*z+equation[3])/equation[1]


def slab_candidates(units, a, b):
    """Match only close parallel surfaces with real horizontal overlap."""
    pairs = []
    for floor in units:
        if floor['role'] != 'floor' or floor['preserve_source']:
            continue
        for ceiling in units:
            if ceiling['role'] != 'ceiling' or ceiling['preserve_source']:
                continue
            if abs(floor['equation'][:3] @ ceiling['equation'][:3]) < np.cos(np.deg2rad(5.)):
                continue
            overlap = floor['footprint'].intersection(ceiling['footprint'])
            if overlap.area < .01:
                continue
            overlap = frame_footprint(clean_footprint(frame_footprint(overlap, a, b, inverse=True)), a, b)
            point = overlap.representative_point()
            gap = plane_y(floor['equation'], point.x, point.y)-plane_y(ceiling['equation'], point.x, point.y)
            if -.05 <= gap <= .4:
                pairs.append((float(gap), -overlap.area, floor, ceiling, overlap))
    return sorted(pairs, key=lambda pair: pair[:2])


def wall_mesh(wall, units):
    """Match floor/ceiling locally along a finite wall, including split roofs and relief."""
    offset = float(wall['offset_m'])
    if 'Dir A' in wall['axis']:
        normal = units['a']
        along = units['b']
        span = wall['span_w_m']
    else:
        normal = units['b']
        along = units['a']
        span = wall['span_u_m']
    sections = {'floor': [], 'ceiling': []}
    for unit in units['surfaces']:
        v, t = unit['generated']
        for section in surface_wall_sections(v, t, normal, offset, along):
            sections[unit['role']].append((section, unit))
    span = list(span)
    support = [section for role in sections.values() for section, _ in role]
    if support:
        low = min(section[0] for section in support)
        high = max(section[1] for section in support)
        if 0 < low-span[0] < .12:
            span[0] = low
        if 0 < span[1]-high < .12:
            span[1] = high
    cuts = sorted(set([*span] + [float(np.clip(x, *span)) for role in sections.values() for section, _ in role for x in section[:2]]))
    vertices, faces = [], []
    bounds = wall['bounds_y']
    for left, right in zip(cuts[:-1], cuts[1:]):
        if right-left < 1e-8:
            continue
        middle = (left+right)/2
        choices = {}
        for role, fallback in (('floor', bounds[0]), ('ceiling', bounds[1])):
            active = [(section, unit) for section, unit in sections[role] if section[0]-1e-7 <= middle <= section[1]+1e-7]
            def height(section, coordinate):
                return section[2]+(coordinate-section[0])*(section[3]-section[2])/(section[1]-section[0])
            if active:
                # The local source wall extent identifies the storey, not global scene extrema.
                chosen, _ = min(active, key=lambda value: abs(height(value[0], middle)-fallback))
                choices[role] = [height(chosen, x) for x in (left, right)]
            else:
                nearby = [(section, unit) for section, unit in sections[role]
                          if max(section[0]-middle, middle-section[1], 0.) <= .15]
                if nearby:
                    chosen, _ = min(nearby, key=lambda value: (
                        max(value[0][0]-middle, middle-value[0][1], 0.),
                        abs(height(value[0], middle)-fallback)))
                    choices[role] = [height(chosen, x) for x in (left, right)]
                else:
                    choices[role] = [fallback, fallback]
        if min(choices['ceiling']) <= max(choices['floor']):
            continue
        endpoints = np.array([left*along+offset*normal, right*along+offset*normal])
        start = len(vertices)
        vertices.extend([[endpoints[0, 0], choices['floor'][0], endpoints[0, 2]],
                         [endpoints[1, 0], choices['floor'][1], endpoints[1, 2]],
                         [endpoints[1, 0], choices['ceiling'][1], endpoints[1, 2]],
                         [endpoints[0, 0], choices['ceiling'][0], endpoints[0, 2]]])
        faces.extend(np.array([[0, 1, 2], [0, 2, 3]])+start)
    vertices = np.asarray(vertices).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if len(faces):
        face_normal = np.cross(vertices[faces[0, 1]]-vertices[faces[0, 0]], vertices[faces[0, 2]]-vertices[faces[0, 0]])
        if face_normal @ np.array(wall['normal_unit']) < 0:
            faces = faces[:, ::-1]
    return vertices, faces


def main():
    args = parse_args()
    wall_path = args.instances_dir/'wall_planes_manifest.json'
    surface_path = args.instances_dir/'floors_ceilings_planes_manifest.json'
    if not wall_path.is_file() or not surface_path.is_file():
        print('Error: Required fitted geometry manifest not found')
        return 1
    walls = json.loads(wall_path.read_text())
    elements = json.loads(surface_path.read_text()).get('elements', [])
    basis = walls['orthogonal_basis']
    a, b = np.array(basis['dir_A_east_west']), np.array(basis['dir_B_north_south'])
    wall_planes = walls.get('wall_planes', [])
    units = []
    for element in elements:
        source = Path(element['source_file'])
        if not source.is_file():
            raise ValueError(f'Missing structural surface input: {source}')
        mesh = o3d.io.read_triangle_mesh(str(source))
        mesh.compute_triangle_normals()
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        normals = np.asarray(mesh.triangle_normals)
        colors = np.asarray(mesh.vertex_colors)
        if not len(colors):
            colors = np.full((len(vertices), 3), .6)
        centers, areas = compute_face_centers_and_areas(vertices, faces)
        equation = np.array(element['plane_equation'], dtype=float)
        equation /= np.linalg.norm(equation[:3])
        role = element['type']
        classification = detect_surface_pattern(centers, areas, normals, equation, (a, b), role == 'ceiling')
        preserve = bool(element.get('preserve_source')) and classification.pattern_type != 'beams'
        distance = np.abs(centers @ equation[:3]+equation[3])
        band = .4 if classification.pattern_type == 'beams' else max(.08, float(element.get('distance_threshold_m', .1)))
        selected = faces if preserve else faces[distance <= band]
        footprint = observed_footprint(vertices, selected, a, b)
        elevation = float(np.average(centers[:, 1], weights=areas))
        regions = wall_regions(wall_planes, a, b, elevation)
        footprint = snap_footprint_to_walls(footprint, wall_planes, a, b)
        footprint = constrain_footprint(footprint, regions)
        footprint = frame_footprint(clean_footprint(footprint), a, b)
        units.append({'name': element['name'], 'role': role, 'equation': equation, 'classification': classification,
                      'vertices': vertices, 'faces': faces, 'colors': colors, 'footprint': footprint,
                      'preserve_source': preserve, 'elevation': elevation, 'color_tree': cKDTree(vertices)})
        print(f"{element['name']}: {classification.pattern_type}; source_preserved={preserve}; footprint={footprint.area:.3f}m2", flush=True)
    if units:
        extraction = json.loads((args.instances_dir/'structural_instances_manifest.json').read_text())
        scene_path = args.scene_mesh or Path(extraction['source_mesh'])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        restore_level_footprints(units,scene_path,a,b,wall_planes,args.output_dir)
    pairs = slab_candidates(units, a, b)
    for unit in units:
        unit['generated'] = generate_surface(unit)
        unit['consumed'] = Polygon()
    # Lift each whole floor at most once, before exporting surfaces or creating any solid.
    for _, _, floor, ceiling, overlap in pairs:
        bottom, _ = generate_surface(ceiling, overlap)
        points = np.array([[x, z] for poly in polygons(overlap) for x, z in poly.exterior.coords])
        lowest = np.min(plane_y(floor['equation'], points[:, 0], points[:, 1]))
        lift = max(0., float(bottom[:, 1].max()+1e-5-lowest))
        if lift:
            floor['equation'][3] -= lift*floor['equation'][1]
            floor['elevation'] += lift
            floor['lift'] = floor.get('lift', 0.)+lift
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assembled = []
    manifest = {'units': [], 'slabs': [], 'summary': {'total_units_analyzed': len(units), 'patterns_detected': {}}}
    for unit in units:
        unit['generated'] = generate_surface(unit)
        v, t = unit['generated']
        path = args.output_dir/f"{unit['name']}_regularized.ply"
        export_ply(path, v, t, paint_surface(unit, v))
        if unit['classification'].pattern_type == 'beams':
            params = unit['classification'].parameters
            scanned = regularize_scanned_mesh_to_pattern(unit['vertices'], unit['faces'], unit['equation'][:3],
                unit['equation'][3], unit['equation'][:3], np.array(params['transverse_axis_u']),
                np.array(params['beam_longitudinal_axis_v']), params['period_m'], params['beam_width_m'],
                params['beam_depth_m'], params['anchor_phase_s0'])
            export_ply(args.output_dir/f"{unit['name']}_scanned_regularized.ply", scanned, unit['faces'], unit['colors'])
        manifest['units'].append({'unit_name': unit['name'], 'structural_role': unit['role'],
            'plane_equation': unit['equation'].tolist(), 'mean_elevation_y': unit['elevation'],
            'footprint_xz': mapping(unit['footprint']), 'footprint_source': unit.get('footprint_source'), 'regularized_artifact': str(path),
            'pattern_classification': asdict(unit['classification']), 'source_preserved': unit['preserve_source'],
            'slab_clearance_lift_m': unit.get('lift', 0.)})
        pattern = unit['classification'].pattern_type
        manifest['summary']['patterns_detected'][pattern] = manifest['summary']['patterns_detected'].get(pattern, 0)+1
    for _, _, floor, ceiling, overlap in pairs:
        overlap = overlap.difference(floor['consumed']).difference(ceiling['consumed'])
        if overlap.area < .01:
            continue
        bottom, bottom_faces = generate_surface(ceiling, overlap)
        solid, _, extra_lift = close_slab(bottom, bottom_faces, floor['equation'])
        if extra_lift > 2e-5:
            raise ValueError('Slab floor clearance changed after surface export')
        name = f"slab_inst{len(manifest['slabs']):03d}"
        path = args.output_dir/f'{name}.ply'
        solid.visual.vertex_colors = (255*paint_surface(ceiling, solid.vertices)).astype(np.uint8)
        solid.export(path)
        assembled.append((solid.vertices, solid.faces, solid.visual.vertex_colors[:, :3]/255.))
        manifest['slabs'].append({'slab_name': name, 'floor': floor['name'], 'ceiling': ceiling['name'],
            'file_path': str(path), 'footprint_xz': mapping(overlap), 'floor_plane_equation': floor['equation'].tolist(),
            'floor_clearance_lift_m': floor.get('lift', 0.), 'is_watertight': bool(solid.is_watertight),
            'volume_m3': float(solid.volume)})
        floor['consumed'] = floor['consumed'].union(overlap)
        ceiling['consumed'] = ceiling['consumed'].union(overlap)
    for unit in units:
        remaining = unit['footprint'].difference(unit['consumed'])
        if remaining.area > 1e-8:
            v, t = generate_surface(unit, remaining)
            assembled.append((v, t, paint_surface(unit, v)))
    wall_meshes = []
    wall_source_path = walls.get('artifacts', {}).get('cleaned_wall_pieces')
    wall_source = o3d.io.read_triangle_mesh(str(wall_source_path)) if wall_source_path else o3d.geometry.TriangleMesh()
    wall_v, wall_c = np.asarray(wall_source.vertices), np.asarray(wall_source.vertex_colors)
    if not len(wall_c):
        wall_c = np.full((len(wall_v), 3), .6)
    wall_tree = cKDTree(wall_v)
    for wall in wall_planes:
        v, t = wall_mesh(wall, {'a': a, 'b': b, 'surfaces': units})
        if not len(t):
            continue
        colors = wall_c[wall_tree.query(v, k=1)[1]]
        path = args.output_dir/f"{wall['name']}_regularized.ply"
        export_ply(path, v, t, colors)
        wall_meshes.append((v, t, colors))
        manifest['units'].append({'unit_name': wall['name'], 'structural_role': 'wall',
            'regularized_artifact': str(path), 'plane_equation': wall['plane_equation'],
            'pattern_classification': {'pattern_type': 'simple_and_plain', 'confidence': 1., 'parameters': {}}})
    # Preserve significant vertical source support outside the Manhattan plane families.
    wall_source.compute_triangle_normals()
    normals = np.asarray(wall_source.triangle_normals)
    unmodeled = (np.abs(normals[:, 1]) < .35) & (np.max(np.abs(normals @ np.array([a, b]).T), axis=1) < .85)
    if np.any(unmodeled):
        source_faces = np.asarray(wall_source.triangles)[unmodeled]
        patch = o3d.geometry.TriangleMesh()
        patch.vertices = wall_source.vertices
        patch.triangles = o3d.utility.Vector3iVector(source_faces)
        labels, _, areas = patch.cluster_connected_triangles()
        labels = np.asarray(labels)
        kept = np.zeros(len(source_faces), dtype=bool)
        for component in np.flatnonzero(np.asarray(areas) >= .02):
            points = wall_v[np.unique(source_faces[labels == component])]
            if np.max(np.ptp(points @ np.array([a, b]).T, axis=0)) >= .12 and np.ptp(points[:, 1]) >= .12:
                kept |= labels == component
        if np.any(kept):
            wall_meshes.append((wall_v, source_faces[kept], wall_c))
    if wall_meshes:
        export_combined(args.output_dir/'wall_regularized.ply', wall_meshes)
        assembled.extend(wall_meshes)
    extraction_path = args.instances_dir/'structural_instances_manifest.json'
    if extraction_path.is_file():
        for item in json.loads(extraction_path.read_text()).get('instances', []):
            if item['role'] != 'stair':
                continue
            mesh = o3d.io.read_triangle_mesh(str(args.instances_dir/item['file_name']))
            vertices, faces, colors = np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_colors)
            if not len(colors):
                colors = np.full((len(vertices), 3), .6)
            assembled.append((vertices, faces, colors))
            manifest.setdefault('preserved_structural_instances', []).append(item)
    if assembled:
        export_combined(args.output_dir/'all_structural_regularized.ply', assembled)
    (args.output_dir/'plane_patterns_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return 0


def export_combined(path, meshes):
    vertices, faces, colors, start = [], [], [], 0
    for v, t, c in meshes:
        vertices.append(v); faces.append(t+start); colors.append(c); start += len(v)
    export_ply(path, np.vstack(vertices), np.vstack(faces), np.vstack(colors))


if __name__ == '__main__':
    sys.exit(main())
