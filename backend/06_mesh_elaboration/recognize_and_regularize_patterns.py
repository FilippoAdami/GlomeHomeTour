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
import trimesh
from scipy.signal import correlate, find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon, mapping
from shapely.ops import unary_union
from surface_footprints import observed_footprint, wall_regions, constrain_footprint, frame_footprint, polygons, snap_footprint_to_walls, clean_footprint
from level_footprints import restore_level_footprints
from structural_geometry import (
    beam_surface, close_slab, pattern_frame, perimeter_pattern_axes, snap_small_excursions,
    polygon_triangles,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_3DGS_to_mesh"))
from fill_mesh_holes import read_ply_face_flags


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
    parser.add_argument('--wall-manifest', type=Path, help='Optional fitted wall manifest outside instances-dir')
    parser.add_argument('--surface-shape-manifest', type=Path,
                        help='Optional wall_perimeter_fit.json supplying final floor and ceiling outlines')
    parser.add_argument('--skip-texture', action='store_true', help='Skip texture synthesis and export untextured regularized geometry')
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


def _aligned_xz_to_source(points, rotation, translation):
    points = np.asarray(points, dtype=np.float64)
    aligned = np.column_stack((points[:, 0], np.zeros(len(points)), points[:, 1]))
    source = (aligned - translation) @ rotation
    return source[:, [0, 2]]


def _aligned_plane_to_source(equation, rotation, translation):
    equation = np.asarray(equation, dtype=np.float64)
    normal = rotation.T @ equation[:3]
    offset = float(equation[3] + equation[:3] @ translation)
    scale = np.linalg.norm(normal)
    return np.r_[normal / scale, offset / scale]


def _plan_footprint_in_source_frame(perimeter, cutouts, rotation, translation):
    outline = _aligned_xz_to_source(perimeter, rotation, translation)
    footprint = Polygon(outline)
    holes = [Polygon(_aligned_xz_to_source(ring, rotation, translation)) for ring in cutouts]
    holes = [hole.intersection(footprint) for hole in holes if not hole.is_empty]
    if holes:
        footprint = footprint.difference(unary_union(holes))
    if not footprint.is_valid:
        footprint = footprint.buffer(0)
    if footprint.is_empty or footprint.area <= 1e-8:
        raise ValueError('Fitted floor/ceiling plan produced an empty structural footprint')
    return footprint


def _clip_polygon_y(points, colors, y_min, y_max):
    polygon = [(point, color) for point, color in zip(points, colors)]
    for bound, keep_above in ((y_min, True), (y_max, False)):
        if not polygon:
            break
        clipped = []
        previous_point, previous_color = polygon[-1]
        previous_inside = previous_point[1] >= bound if keep_above else previous_point[1] <= bound
        for point, color in polygon:
            inside = point[1] >= bound if keep_above else point[1] <= bound
            if inside != previous_inside:
                fraction = (bound - previous_point[1]) / (point[1] - previous_point[1])
                clipped.append((previous_point + fraction * (point - previous_point),
                                previous_color + fraction * (color - previous_color)))
            if inside:
                clipped.append((point, color))
            previous_point, previous_color, previous_inside = point, color, inside
        polygon = []
        for point, color in clipped:
            if not polygon or np.linalg.norm(point - polygon[-1][0]) > 1e-10:
                polygon.append((point, color))
        if len(polygon) > 1 and np.linalg.norm(polygon[0][0] - polygon[-1][0]) <= 1e-10:
            polygon.pop()
    return polygon


def clip_mesh_y_bounds(vertices, triangles, colors, y_min, y_max):
    """Clip a triangle mesh to two horizontal planes, interpolating new edge vertices."""
    out_v, out_t, out_c = [], [], []
    for face in triangles:
        clipped = _clip_polygon_y(vertices[face], colors[face], y_min, y_max)
        if len(clipped) < 3:
            continue
        start = len(out_v)
        out_v.extend(point for point, _ in clipped)
        out_c.extend(color for _, color in clipped)
        out_t.extend([[start, start + i, start + i + 1] for i in range(1, len(clipped) - 1)])
    return (np.asarray(out_v, dtype=np.float64).reshape(-1, 3),
            np.asarray(out_t, dtype=np.int32).reshape(-1, 3),
            np.asarray(out_c, dtype=np.float64).reshape(-1, 3))


def surface_vertical_bounds(units, wall_planes, fallback_wall_bounds=None):
    bounds = {'floor': [], 'ceiling': []}
    for unit in units:
        if unit['role'] not in bounds or abs(unit['equation'][1]) < 1e-8:
            continue
        points = np.vstack([np.asarray(poly.exterior.coords)[:-1]
                            for poly in polygons(unit['footprint'])])
        heights = -(unit['equation'][0] * points[:, 0]
                    + unit['equation'][2] * points[:, 1]
                    + unit['equation'][3]) / unit['equation'][1]
        bounds[unit['role']].append(heights)
    if not bounds['floor'] or not bounds['ceiling']:
        wall_y = [value for wall in wall_planes for value in wall.get('bounds_y', [])]
        if len(wall_y) >= 2:
            return float(min(wall_y)), float(max(wall_y))
        if fallback_wall_bounds and len(fallback_wall_bounds) == 2:
            return tuple(map(float, fallback_wall_bounds))
        if bounds['floor']:
            floor_min = float(min(values.min() for values in bounds['floor']))
            floor_max = float(max(values.max() for values in bounds['floor']))
            return floor_min, floor_max + 2.8
        if bounds['ceiling']:
            ceil_min = float(min(values.min() for values in bounds['ceiling']))
            ceil_max = float(max(values.max() for values in bounds['ceiling']))
            return ceil_min - 2.8, ceil_max
        raise ValueError('Cannot determine global wall height limits from floors, ceilings, or wall manifest')
    return float(min(values.min() for values in bounds['floor'])), float(max(values.max() for values in bounds['ceiling']))


def subdivide_surface(v: np.ndarray, t: np.ndarray, max_edge: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    """Subdivides a regularized surface mesh to ensure high-fidelity texture rendering."""
    if len(v) == 0 or len(t) == 0:
        return v, t
    try:
        v_sub, t_sub = trimesh.remesh.subdivide_to_size(v, t, max_edge=max_edge)
        return np.asarray(v_sub, dtype=np.float64), np.asarray(t_sub, dtype=np.int32)
    except Exception:
        return v, t


def regularize_wall_lighting_and_outliers(
    pts: np.ndarray,
    cols: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Filters outliers (cables, foreign objects, glare, deep shadow crevices)
    and performs 2D lighting regularization to remove shadows and overexposure.
    """
    if len(pts) < 10:
        return pts, cols, pts @ u_axis, pts @ v_axis, {}

    u = pts @ u_axis
    v = pts @ v_axis
    lum = 0.299 * cols[:, 0] + 0.587 * cols[:, 1] + 0.114 * cols[:, 2]

    # Percentiles for luminance
    p12 = float(np.percentile(lum, 12)) if len(lum) > 50 else 0.0
    p92 = float(np.percentile(lum, 92)) if len(lum) > 50 else 1.0

    # Authentic plaster chromatic signature: warm beige (R > G > B with R - B >= 0.035)
    chroma = cols[:, 0] - cols[:, 2]

    # Handle both calibrated (mean lum > 0.5) and raw uncalibrated camera input
    if float(np.median(lum)) > 0.50:
        not_dark_cable = (lum >= max(0.35, p12)) & (cols[:, 0] >= 0.35)
        not_glare_or_plate = (lum <= p92) & (cols.mean(axis=1) <= 0.98)
        is_warm_plaster = (chroma >= -0.05) & (cols[:, 0] >= cols[:, 1] - 0.05)
    else:
        not_dark_cable = (lum >= max(0.18, p12)) & (cols[:, 0] >= 0.20)
        not_glare_or_plate = (lum <= p92) & (cols.mean(axis=1) <= 0.72) & np.all(cols <= 0.75, axis=1)
        is_warm_plaster = (chroma >= 0.035) & (cols[:, 0] > cols[:, 1])

    inlier_mask = not_dark_cable & not_glare_or_plate & is_warm_plaster

    # Statistical outlier rejection via MAD on chroma
    if np.count_nonzero(inlier_mask) > 50:
        med_chroma = np.median(chroma[inlier_mask])
        mad_chroma = np.median(np.abs(chroma[inlier_mask] - med_chroma))
        if mad_chroma > 1e-4:
            chroma_inliers = np.abs(chroma - med_chroma) <= 3.0 * mad_chroma
            inlier_mask = inlier_mask & chroma_inliers

    if np.count_nonzero(inlier_mask) < 20:
        inlier_mask = (lum >= p12) & (lum <= p92)
        if np.count_nonzero(inlier_mask) < 10:
            inlier_mask = np.ones(len(pts), dtype=bool)

    inl_pts = pts[inlier_mask]
    inl_cols = cols[inlier_mask]
    inl_u = u[inlier_mask]
    inl_v = v[inlier_mask]
    inl_lum = lum[inlier_mask]

    # Fit 2D quadratic lighting polynomial:
    # L(u, v) = c0 + c1*u + c2*v + c3*u^2 + c4*v^2 + c5*u*v
    X = np.column_stack([np.ones_like(inl_u), inl_u, inl_v, inl_u**2, inl_v**2, inl_u * inl_v])
    coeffs, _, _, _ = np.linalg.lstsq(X, inl_lum, rcond=None)
    pred_lum = np.maximum(0.05, X @ coeffs)
    ref_lum = float(np.median(inl_lum))

    # Lighting normalized colors: scale by (ref_lum / pred_lum)
    reg_cols = np.clip(inl_cols * (ref_lum / pred_lum[:, None]), 0.0, 1.0)

    diag = {
        'total_points': len(pts),
        'inlier_points': len(inl_pts),
        'inlier_ratio': len(inl_pts) / max(1, len(pts)),
        'ref_lum': ref_lum,
        'raw_variance': float(np.var(inl_lum)),
        'reg_variance': float(np.var(0.299 * reg_cols[:, 0] + 0.587 * reg_cols[:, 1] + 0.114 * reg_cols[:, 2])),
        'median_color': np.median(reg_cols, axis=0).tolist(),
    }
    return inl_pts, reg_cols, inl_u, inl_v, diag


def detect_wall_color_splits(
    v_coords: np.ndarray,
    u_coords: np.ndarray,
    colors: np.ndarray,
    split_threshold: float = 0.15,
) -> dict:
    """Detects abrupt changes in color across the same wall (e.g. horizontal wainscoting /
    two-tone wall, or vertical accent color split).
    """
    if len(colors) < 50:
        med = np.median(colors, axis=0) if len(colors) > 0 else np.array([0.88, 0.88, 0.86])
        return {
            'is_split': False, 'split_type': 'none', 'split_coord': None,
            'color_1': med, 'color_2': None, 'delta_E': 0.0,
        }

    # 1. Test horizontal split along elevation v
    v_min, v_max = np.percentile(v_coords, 15), np.percentile(v_coords, 85)
    candidates_v = np.linspace(v_min, v_max, 35)
    best_d_horiz = 0.0
    best_v_split = None
    c_lower, c_upper = None, None
    for cand in candidates_v:
        m1 = v_coords < cand
        m2 = v_coords >= cand
        if np.count_nonzero(m1) < 25 or np.count_nonzero(m2) < 25:
            continue
        med1 = np.median(colors[m1], axis=0)
        med2 = np.median(colors[m2], axis=0)
        d = float(np.linalg.norm(med1 - med2))
        if d > best_d_horiz:
            best_d_horiz = d
            best_v_split = cand
            c_lower, c_upper = med1, med2

    # 2. Test vertical split along span u
    u_min, u_max = np.percentile(u_coords, 15), np.percentile(u_coords, 85)
    candidates_u = np.linspace(u_min, u_max, 35)
    best_d_vert = 0.0
    best_u_split = None
    c_left, c_right = None, None
    for cand in candidates_u:
        m1 = u_coords < cand
        m2 = u_coords >= cand
        if np.count_nonzero(m1) < 25 or np.count_nonzero(m2) < 25:
            continue
        med1 = np.median(colors[m1], axis=0)
        med2 = np.median(colors[m2], axis=0)
        d = float(np.linalg.norm(med1 - med2))
        if d > best_d_vert:
            best_d_vert = d
            best_u_split = cand
            c_left, c_right = med1, med2

    if best_d_horiz >= split_threshold and best_d_horiz >= best_d_vert:
        return {
            'is_split': True,
            'split_type': 'horizontal',
            'split_coord': float(best_v_split),
            'color_1': c_lower,
            'color_2': c_upper,
            'delta_E': float(best_d_horiz),
        }
    elif best_d_vert >= split_threshold:
        return {
            'is_split': True,
            'split_type': 'vertical',
            'split_coord': float(best_u_split),
            'color_1': c_left,
            'color_2': c_right,
            'delta_E': float(best_d_vert),
        }
    else:
        return {
            'is_split': False,
            'split_type': 'none',
            'split_coord': None,
            'color_1': np.median(colors, axis=0),
            'color_2': None,
            'delta_E': float(max(best_d_horiz, best_d_vert)),
        }


def unify_adjacent_wall_colors(
    wall_data_list: List[dict],
    border_dist_m: float = 0.40,
    match_threshold: float = 0.10,
) -> List[dict]:
    """Identifies touching adjacent walls at room corners, compares border colors
    near the corner intersections, and unifies touching walls that share the same
    paint color into connected components with a pooled authentic color.
    """
    n_walls = len(wall_data_list)
    adj = np.zeros((n_walls, n_walls), dtype=bool)
    np.fill_diagonal(adj, True)

    for i in range(n_walls):
        w1 = wall_data_list[i]
        eq1 = w1['plane_equation']
        n1 = eq1[:3]
        pts1 = w1['clean_pts']
        cols1 = w1['reg_cols']
        if len(pts1) == 0:
            continue

        for j in range(i + 1, n_walls):
            w2 = wall_data_list[j]
            eq2 = w2['plane_equation']
            n2 = eq2[:3]
            pts2 = w2['clean_pts']
            cols2 = w2['reg_cols']
            if len(pts2) == 0:
                continue

            # Check if wall normals are approximately perpendicular
            if abs(float(np.dot(n1, n2))) > 0.30:
                continue

            # Solve 2D corner line in (X, Z)
            A = np.array([[eq1[0], eq1[2]], [eq2[0], eq2[2]]])
            b = np.array([-eq1[3], -eq2[3]])
            try:
                corner_xz = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue

            d1 = np.linalg.norm(pts1[:, [0, 2]] - corner_xz, axis=1)
            d2 = np.linalg.norm(pts2[:, [0, 2]] - corner_xz, axis=1)

            if np.min(d1) > 0.80 or np.min(d2) > 0.80:
                continue

            near1 = cols1[d1 < border_dist_m]
            near2 = cols2[d2 < border_dist_m]

            med1 = np.median(near1, axis=0) if len(near1) >= 10 else np.median(cols1, axis=0)
            med2 = np.median(near2, axis=0) if len(near2) >= 10 else np.median(cols2, axis=0)

            diff = float(np.linalg.norm(med1 - med2))
            if diff <= match_threshold:
                adj[i, j] = adj[j, i] = True

    visited = np.zeros(n_walls, dtype=bool)
    for i in range(n_walls):
        if visited[i]:
            continue
        cluster = []
        queue = [i]
        visited[i] = True
        while queue:
            curr = queue.pop(0)
            cluster.append(curr)
            for neighbor in range(n_walls):
                if adj[curr, neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        pooled_cols = []
        for idx in cluster:
            if not wall_data_list[idx]['split_info']['is_split']:
                pooled_cols.append(wall_data_list[idx]['reg_cols'])

        if pooled_cols and any(len(c) > 0 for c in pooled_cols):
            all_c = np.vstack([c for c in pooled_cols if len(c) > 0])
            cluster_unified_color = np.median(all_c, axis=0)
        else:
            all_c = np.empty((0, 3), dtype=np.float64)
            cluster_unified_color = np.array([0.88, 0.88, 0.86])

        cluster_names = [wall_data_list[k]['name'] for k in cluster]
        for idx in cluster:
            wall_data_list[idx]['cluster_ids'] = cluster
            wall_data_list[idx]['merged_batch_pixels'] = all_c
            wall_data_list[idx]['merged_wall_names'] = cluster_names
            if not wall_data_list[idx]['split_info']['is_split']:
                wall_data_list[idx]['unified_color'] = cluster_unified_color
            else:
                wall_data_list[idx]['unified_color'] = None

    return wall_data_list


def estimate_scene_chromatic_adaptation(
    wall_inlier_colors: np.ndarray,
    target_white_albedo: np.ndarray = np.array([0.88, 0.88, 0.86]),
) -> np.ndarray:
    """Estimates the scene-wide Von Kries chromatic adaptation gain vector using
    the neutral plaster wall as the color constancy reference.
    """
    if len(wall_inlier_colors) == 0:
        return np.ones(3, dtype=np.float64)
    obs = np.median(wall_inlier_colors, axis=0)
    obs = np.maximum(0.04, obs)
    gain = target_white_albedo / obs
    return np.clip(gain, 0.5, 6.0)


def moisun_periodic_plus_smooth(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Moisan (2011) Periodic plus Smooth Image Decomposition.
    Decomposes an image u into u = p + s, where:
    - p is strictly periodic (zero step discontinuity across opposite boundaries)
    - s is the smooth harmonic component carrying boundary step differences.
    """
    H, W = img.shape[:2]
    is_color = (img.ndim == 3)
    channels = img.shape[2] if is_color else 1

    p = np.zeros_like(img, dtype=np.float32)
    s = np.zeros_like(img, dtype=np.float32)

    for c in range(channels):
        u = img[:, :, c] if is_color else img
        v = np.zeros((H, W), dtype=np.float32)
        v[0, :] = u[0, :] - u[-1, :]
        v[-1, :] = -v[0, :]
        v[:, 0] += u[:, 0] - u[:, -1]
        v[:, -1] += -(u[:, 0] - u[:, -1])

        x = np.arange(W)
        y = np.arange(H)
        X, Y = np.meshgrid(x, y)
        denom = 2.0 * (np.cos(2.0 * np.pi * X / W) + np.cos(2.0 * np.pi * Y / H) - 2.0)
        denom[0, 0] = 1.0

        V = np.fft.fft2(v)
        S = V / denom
        S[0, 0] = 0.0
        s_c = np.fft.ifft2(S).real
        p_c = u - s_c

        if is_color:
            p[:, :, c] = p_c
            s[:, :, c] = s_c
        else:
            p = p_c
            s = s_c

    return np.clip(p, 0.0, 1.0), s


def regularize_tile_boundaries(
    tile: np.ndarray,
    blend_fraction: float = 0.35,
    detrend_linear: bool = True,
) -> np.ndarray:
    """
    Enforces exact C0 and smooth C1 boundary continuity across opposite edges of a tile:
      T(0, y) == T(W-1, y)  and  T(x, 0) == T(x, H-1)

    1. Removes any low-frequency planar illumination gradient across the patch (flat-fielding),
       preventing macro checkerboard / sawtooth luminance jumps between repeated tiles.
    2. Applies a symmetric Hermite/smoothstep (3t^2 - 2t^3) boundary cross-fade over the margin bands.
    3. Guarantees exact floating-point equality at outer borders so seam lines completely vanish
       when repeated in 3x3 or arbitrary planar tessellations.
    """
    img = tile.astype(np.float32).copy()
    if img.max() > 1.0:
        img /= 255.0
    H, W, C = img.shape

    # 1. Low-frequency 2D linear gradient detrending
    if detrend_linear and H >= 16 and W >= 16:
        Y, X = np.indices((H, W))
        X_norm = (X - W / 2.0) / (W / 2.0)
        Y_norm = (Y - H / 2.0) / (H / 2.0)
        A = np.column_stack([np.ones(H * W), X_norm.ravel(), Y_norm.ravel()])
        for c in range(C):
            ch = img[:, :, c].ravel()
            coeffs, _, _, _ = np.linalg.lstsq(A, ch, rcond=None)
            trend = (A @ coeffs).reshape(H, W)
            mean_val = float(np.mean(ch))
            img[:, :, c] = np.clip(img[:, :, c] - trend + mean_val, 0.0, 1.0)

    bw = max(4, int(W * blend_fraction))
    bh = max(4, int(H * blend_fraction))

    t_x = np.linspace(0, 1, bw)
    ramp_x = (3.0 * t_x**2 - 2.0 * t_x**3)[:, None]  # 0 at boundary, 1 at interior
    t_y = np.linspace(0, 1, bh)
    ramp_y = (3.0 * t_y**2 - 2.0 * t_y**3)[:, None]

    out = img.copy()

    # Horizontal cross-fade
    for k in range(bw):
        w_in = ramp_x[k]
        w_edge = 1.0 - w_in
        l_idx = k
        r_idx = W - 1 - k
        l_orig = img[:, l_idx, :].copy()
        r_orig = img[:, r_idx, :].copy()
        mid = 0.5 * (l_orig + r_orig)
        out[:, l_idx, :] = w_in * l_orig + w_edge * mid
        out[:, r_idx, :] = w_in * r_orig + w_edge * mid

    # Vertical cross-fade
    for k in range(bh):
        w_in = ramp_y[k]
        w_edge = 1.0 - w_in
        t_idx = k
        b_idx = H - 1 - k
        t_orig = out[t_idx, :, :].copy()
        b_orig = out[b_idx, :, :].copy()
        mid = 0.5 * (t_orig + b_orig)
        out[t_idx, :, :] = w_in * t_orig + w_edge * mid
        out[b_idx, :, :] = w_in * b_orig + w_edge * mid

    # Exact floating point equality at edge pixels
    out[:, 0] = out[:, -1] = 0.5 * (out[:, 0] + out[:, -1])
    out[0, :] = out[-1, :] = 0.5 * (out[0, :] + out[-1, :])
    return np.clip(out, 0.0, 1.0)


def synthesize_seamless_floor_tile(
    patch: np.ndarray,
    overlap_ratio: float = 0.18,
) -> np.ndarray:
    """Synthesizes a seamless floor tile using:
    1. Structure tensor rotation to align grain with vertical axis (prevents zigzag broken lines).
    2. 2D illumination detrending and smoothstep boundary regularization for zero-discontinuity tiling.
    """
    if patch.size == 0 or patch.shape[0] < 4 or patch.shape[1] < 4:
        return np.full((32, 32, 3), 0.6, dtype=np.float32)

    patch = patch.astype(np.float32)
    if patch.max() > 1.0:
        patch /= 255.0
    H, W, C = patch.shape

    # 1. Structure tensor angle alignment
    gray = cv2.cvtColor((patch * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    Jxx = cv2.GaussianBlur(gx * gx, (21, 21), 5)
    Jyy = cv2.GaussianBlur(gy * gy, (21, 21), 5)
    Jxy = cv2.GaussianBlur(gx * gy, (21, 21), 5)
    theta_grad = 0.5 * np.arctan2(2 * Jxy.mean(), (Jxx - Jyy).mean()) * 180 / np.pi
    theta_lines = theta_grad + 90.0

    rot_deg = 90.0 - theta_lines
    if abs(rot_deg) > 5.0 and abs(rot_deg) < 85.0:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rot_deg, 1.0)
        patch = cv2.warpAffine(patch, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        inset = int(min(W, H) * 0.12)
        patch = patch[inset:-inset, inset:-inset]

    # 2. Perfect boundary regularization (zero edge lines)
    seamless = regularize_tile_boundaries(patch, blend_fraction=0.35, detrend_linear=True)
    return seamless


def synthesize_seamless_beam_tile(
    beam_crop: np.ndarray,
    overlap_y: int = 30,
) -> np.ndarray:
    """Synthesizes a seamless 1-period beam tile with smooth longitudinal (Y) continuity
    and zero step discontinuity across repeating periods.
    """
    if beam_crop.size == 0 or beam_crop.shape[0] < 4 or beam_crop.shape[1] < 4:
        return np.full((32, 32, 3), 0.5, dtype=np.float32)

    b = beam_crop.astype(np.float32).copy()
    if b.max() > 1.0:
        b /= 255.0
    bh_total, bw_total, C = b.shape

    # 1. Detrend longitudinal gradient along Y
    Y = np.arange(bh_total, dtype=np.float32)
    Y_norm = (Y - bh_total / 2.0) / (bh_total / 2.0)
    A_y = np.column_stack([np.ones(bh_total), Y_norm])
    for c_i in range(C):
        col_prof = b[:, :, c_i].mean(axis=1)
        coeff, _, _, _ = np.linalg.lstsq(A_y, col_prof, rcond=None)
        trend = (A_y @ coeff)[:, None]
        mean_v = float(np.mean(col_prof))
        b[:, :, c_i] = np.clip(b[:, :, c_i] - trend + mean_v, 0.0, 1.0)

    # 2. Longitudinal cross-fade along Y (top to bottom)
    by = min(max(10, overlap_y), int(bh_total * 0.35))
    t_y = np.linspace(0, 1, by)
    ramp_y = (3.0 * t_y**2 - 2.0 * t_y**3)[:, None]

    for k in range(by):
        w_in = ramp_y[k]
        w_edge = 1.0 - w_in
        t_idx = k
        b_idx = bh_total - 1 - k
        t_orig = b[t_idx, :, :].copy()
        b_orig = b[b_idx, :, :].copy()
        mid = 0.5 * (t_orig + b_orig)
        b[t_idx, :, :] = w_in * t_orig + w_edge * mid
        b[b_idx, :, :] = w_in * b_orig + w_edge * mid

    # 3. Transverse cross-fade along X (left valley to right valley)
    bx = min(12, bw_total // 4)
    t_x = np.linspace(0, 1, bx)
    ramp_x = (3.0 * t_x**2 - 2.0 * t_x**3)[:, None]
    for k in range(bx):
        w_in = ramp_x[k]
        w_edge = 1.0 - w_in
        l_idx = k
        r_idx = bw_total - 1 - k
        l_orig = b[:, l_idx, :].copy()
        r_orig = b[:, r_idx, :].copy()
        mid = 0.5 * (l_orig + r_orig)
        b[:, l_idx, :] = w_in * l_orig + w_edge * mid
        b[:, r_idx, :] = w_in * r_orig + w_edge * mid

    # Exact floating point equality
    b[0, :] = b[-1, :] = 0.5 * (b[0, :] + b[-1, :])
    b[:, 0] = b[:, -1] = 0.5 * (b[:, 0] + b[:, -1])
    return np.clip(b, 0.0, 1.0)


def synthesize_floor_unit_tile(unit: dict, dir_A: np.ndarray, dir_B: np.ndarray, res_m: float = 0.005) -> np.ndarray:
    """Extracts glare-free inlier floor patch, performs 2D lighting normalization,
    and synthesizes a seamless golden-oak parquet tile with exact C0/C1 boundary continuity.
    """
    v = unit['vertices']
    t = unit['faces']
    c = unit['colors']

    u_coords = v @ dir_A
    v_coords = v @ dir_B
    min_u, max_u = u_coords.min(), u_coords.max()
    min_v, max_v = v_coords.min(), v_coords.max()
    W = int(np.ceil((max_u - min_u) / res_m)) + 1
    H = int(np.ceil((max_v - min_v) / res_m)) + 1

    u_px = np.clip(((u_coords - min_u) / res_m).astype(np.int32), 0, W - 1)
    v_px = np.clip(((max_v - v_coords) / res_m).astype(np.int32), 0, H - 1)
    tri_px = np.column_stack([u_px, v_px])[t]
    tri_cols = c[t].mean(axis=1)

    raw_img = np.zeros((H, W, 3), dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.uint8)
    for i in range(len(t)):
        cv2.fillPoly(raw_img, [tri_px[i].reshape((-1, 1, 2))], tri_cols[i].tolist())
        cv2.fillPoly(mask, [tri_px[i].reshape((-1, 1, 2))], 255)

    ys, xs = np.where(mask > 0)
    if len(ys) > 50:
        pix = raw_img[ys, xs]
        lum = 0.299 * pix[:, 0] + 0.587 * pix[:, 1] + 0.114 * pix[:, 2]
        p10, p90 = float(np.percentile(lum, 10)), float(np.percentile(lum, 90))
        inl = (lum >= max(0.10, p10)) & (lum <= p90)
        if np.count_nonzero(inl) < 20:
            inl = np.ones(len(ys), dtype=bool)
        norm_x = (xs[inl] - W / 2.0) / (W / 2.0)
        norm_y = (ys[inl] - H / 2.0) / (H / 2.0)
        X_inl = np.column_stack([np.ones_like(norm_x), norm_x, norm_y, norm_x**2, norm_y**2, norm_x * norm_y])
        coeffs, _, _, _ = np.linalg.lstsq(X_inl, lum[inl], rcond=None)
        all_norm_x = (xs - W / 2.0) / (W / 2.0)
        all_norm_y = (ys - H / 2.0) / (H / 2.0)
        X_all = np.column_stack([np.ones_like(all_norm_x), all_norm_x, all_norm_y, all_norm_x**2, all_norm_y**2, all_norm_x * all_norm_y])
        pred_lum = np.maximum(0.08, X_all @ coeffs)
        ref_lum = float(np.median(lum[inl]))
        reg_img = raw_img.copy()
        reg_img[ys, xs] = np.clip(pix * (ref_lum / pred_lum)[:, None], 0.0, 1.0)
    else:
        reg_img = raw_img.copy()

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5) * res_m
    r = max(24, int(0.32 / res_m))
    best_cx, best_cy = None, None
    best_score = 1e9
    for y_cand in range(r + 10, H - r - 10, 10):
        for x_cand in range(r + 10, W - r - 10, 10):
            if dist[y_cand, x_cand] < 0.35:
                continue
            p = reg_img[y_cand - r: y_cand + r, x_cand - r: x_cand + r]
            lum_p = 0.299 * p[:, :, 0] + 0.587 * p[:, :, 1] + 0.114 * p[:, :, 2]
            if lum_p.max() > 0.85 or lum_p.min() < 0.20:
                continue
            chroma = p[:, :, 0].mean() - p[:, :, 2].mean()
            if chroma < 0.08:
                continue
            score = lum_p.var() - 0.1 * dist[y_cand, x_cand] - 0.2 * chroma
            if score < best_score:
                best_score = score
                best_cx, best_cy = x_cand, y_cand

    if best_cx is None:
        best_cy, best_cx = np.unravel_index(np.argmax(dist), dist.shape)

    x0 = max(0, best_cx - r)
    x1 = min(W, best_cx + r)
    y0 = max(0, best_cy - r)
    y1 = min(H, best_cy + r)
    crop_floor = reg_img[y0:y1, x0:x1].copy()
    if crop_floor.size == 0 or crop_floor.shape[0] < 4 or crop_floor.shape[1] < 4:
        base_c = c.mean(axis=0) if len(c) > 0 else np.array([0.61, 0.47, 0.38])
        crop_floor = np.full((32, 32, 3), base_c, dtype=np.float32)
    return synthesize_seamless_floor_tile(crop_floor)


def synthesize_ceiling_unit_tile(unit: dict, res_m: float = 0.005) -> Tuple[np.ndarray, float]:
    """Extracts glare-free inlier beam profile, performs 2D lighting normalization,
    and synthesizes a seamless 1-period timber beam tile with smooth longitudinal continuity.
    """
    v = unit['vertices']
    t = unit['faces']
    c = unit['colors']
    params = unit['classification'].parameters
    period = params['period_m']
    u_axis = np.array(params['transverse_axis_u'], dtype=np.float64)
    v_axis = np.array(params['beam_longitudinal_axis_v'], dtype=np.float64)

    u_coords = v @ u_axis
    v_coords = v @ v_axis
    min_u, max_u = u_coords.min(), u_coords.max()
    min_v, max_v = v_coords.min(), v_coords.max()
    W = int(np.ceil((max_u - min_u) / res_m)) + 1
    H = int(np.ceil((max_v - min_v) / res_m)) + 1

    u_px = np.clip(((u_coords - min_u) / res_m).astype(np.int32), 0, W - 1)
    v_px = np.clip(((max_v - v_coords) / res_m).astype(np.int32), 0, H - 1)
    tri_px = np.column_stack([u_px, v_px])[t]
    tri_cols = c[t].mean(axis=1)

    raw_img = np.zeros((H, W, 3), dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.uint8)
    for i in range(len(t)):
        cv2.fillPoly(raw_img, [tri_px[i].reshape((-1, 1, 2))], tri_cols[i].tolist())
        cv2.fillPoly(mask, [tri_px[i].reshape((-1, 1, 2))], 255)

    ys, xs = np.where(mask > 0)
    if len(ys) > 50:
        pix = raw_img[ys, xs]
        lum = 0.299 * pix[:, 0] + 0.587 * pix[:, 1] + 0.114 * pix[:, 2]
        p10, p90 = float(np.percentile(lum, 10)), float(np.percentile(lum, 90))
        inl = (lum >= max(0.10, p10)) & (lum <= p90)
        if np.count_nonzero(inl) < 20:
            inl = np.ones(len(ys), dtype=bool)
        norm_x = (xs[inl] - W / 2.0) / (W / 2.0)
        norm_y = (ys[inl] - H / 2.0) / (H / 2.0)
        X_inl = np.column_stack([np.ones_like(norm_x), norm_x, norm_y, norm_x**2, norm_y**2, norm_x * norm_y])
        coeffs, _, _, _ = np.linalg.lstsq(X_inl, lum[inl], rcond=None)
        all_norm_x = (xs - W / 2.0) / (W / 2.0)
        all_norm_y = (ys - H / 2.0) / (H / 2.0)
        X_all = np.column_stack([np.ones_like(all_norm_x), all_norm_x, all_norm_y, all_norm_x**2, all_norm_y**2, all_norm_x * all_norm_y])
        pred_lum = np.maximum(0.08, X_all @ coeffs)
        ref_lum = float(np.median(lum[inl]))
        reg_img = raw_img.copy()
        reg_img[ys, xs] = np.clip(pix * (ref_lum / pred_lum)[:, None], 0.0, 1.0)
    else:
        reg_img = raw_img.copy()

    p_px = int(round(period / res_m))
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5) * res_m
    cy, cx = np.unravel_index(np.argmax(dist), dist.shape)

    prof_raw = reg_img.mean(axis=0).mean(axis=1)
    prof = cv2.GaussianBlur(prof_raw.reshape(1, -1), (0, 0), sigmaX=3).flatten()
    search_r = min(p_px // 2, cx, len(prof) - cx - 1)
    local_zone = prof[cx - search_r: cx + search_r]
    peak_offset = np.argmax(local_zone) - search_r
    peak_idx = max(0, min(len(prof) - p_px, cx + peak_offset))

    col_valid = np.all(mask[:, peak_idx: peak_idx + p_px] > 0, axis=1)
    valid_rows = np.where(col_valid)[0]
    if len(valid_rows) >= 50:
        y_mid = int(np.clip(cy, valid_rows[0] + 25, valid_rows[-1] - 25))
        half_h = min(int(0.60 / res_m), (valid_rows[-1] - valid_rows[0]) // 2)
        y0 = max(valid_rows[0], y_mid - half_h)
        y1 = min(valid_rows[-1], y_mid + half_h)
    crop_beam = reg_img[y0:y1, peak_idx: peak_idx + p_px].copy()
    if crop_beam.size == 0 or crop_beam.shape[0] < 4 or crop_beam.shape[1] < 4:
        base_c = c.mean(axis=0) if len(c) > 0 else np.array([0.58, 0.38, 0.28])
        crop_beam = np.full((32, 32, 3), base_c, dtype=np.float32)
    tile = synthesize_seamless_beam_tile(crop_beam)
    return tile, peak_idx * res_m


def analyze_and_synthesize_planar_texture(
    unit: dict, target_vertices: np.ndarray, res_m: float = 0.02
) -> np.ndarray:
    """Projects original scanned colors onto the regularized surface plane UV, analyzes pattern,
    and synthesizes a realistic seamless texture across the full surface.
    """
    if len(target_vertices) == 0:
        return np.empty((0, 3), dtype=np.float64)

    role = unit.get("role", "")
    classification = unit.get("classification")
    pattern_type = getattr(classification, "pattern_type", None) if classification else None
    if pattern_type is None:
        pattern_type = "simple_and_plain" if role in ("wall", "floor") else "beams"

    # =========================================================================
    # BRANCH 1: WALLS (Clean White Plaster with Micro-Texture)
    # =========================================================================
    if role == "wall":
        if unit.get("split_info", {}).get("is_split"):
            split_info = unit["split_info"]
            split_type = split_info["split_type"]
            coord = split_info["split_coord"]
            c1, c2 = split_info["color_1"], split_info["color_2"]
            if split_type == "horizontal":
                mask1 = target_vertices[:, 1] < coord
            else:
                u_ax = unit.get("u_axis", np.array([1.0, 0.0, 0.0]))
                mask1 = (target_vertices @ u_ax) < coord
            sampled = np.empty((len(target_vertices), 3), dtype=np.float64)
            sampled[mask1] = c1
            sampled[~mask1] = c2
            return sampled

        uc = unit.get("unified_color")
        base_color = np.array(uc if uc is not None else [0.88, 0.88, 0.86], dtype=np.float64)
        coords_hash = np.sin(target_vertices[:, 0] * 37.0 + target_vertices[:, 1] * 73.0 + target_vertices[:, 2] * 109.0)
        return np.clip(base_color[None, :] + coords_hash[:, None] * 0.006, 0.0, 1.0)

    # =========================================================================
    # BRANCH 2: FLOORS (Seamless Regularized Golden-Oak Parquet)
    # =========================================================================
    if role == "floor":
        if "seamless_tile" in unit:
            tile = unit["seamless_tile"]
            dir_A = unit.get("dir_A", np.array([1.0, 0.0, 0.0]))
            dir_B = unit.get("dir_B", np.array([0.0, 0.0, 1.0]))
            u = target_vertices @ dir_A
            v = target_vertices @ dir_B
            res_tile = unit.get("tile_res_m", 0.005)
            L_tile = tile.shape[1] * res_tile
            u_norm = np.mod(u, L_tile) / L_tile
            v_norm = np.mod(v, L_tile) / L_tile
            px = np.clip((u_norm * tile.shape[1]).astype(np.int32), 0, tile.shape[1] - 1)
            py = np.clip((v_norm * tile.shape[0]).astype(np.int32), 0, tile.shape[0] - 1)
            return tile[py, px]

    # =========================================================================
    # BRANCH 3: CEILINGS / PERIODIC BEAMS (Seamless Continuous Timber Beams)
    # =========================================================================
    if pattern_type == "beams" and "seamless_tile" in unit:
        tile = unit["seamless_tile"]
        params = classification.parameters
        period = params["period_m"]
        u_axis = np.array(params["transverse_axis_u"], dtype=np.float64)
        v_axis = np.array(params["beam_longitudinal_axis_v"], dtype=np.float64)
        s_offset = unit.get("beam_s_offset", 0.0)
        cu = target_vertices @ u_axis
        cv = target_vertices @ v_axis
        res_tile = unit.get("tile_res_m", 0.005)
        L_long = tile.shape[0] * res_tile
        u_norm = np.mod(cu - s_offset, period) / period
        v_norm = np.mod(cv, L_long) / L_long
        b_px = np.clip((u_norm * tile.shape[1]).astype(np.int32), 0, tile.shape[1] - 1)
        b_py = np.clip((v_norm * tile.shape[0]).astype(np.int32), 0, tile.shape[0] - 1)
        return tile[b_py, b_px]

    # Fallback to authentic inlier color if tile synthesis not precomputed
    source_cols = unit.get("colors") if unit.get("colors") is not None else unit.get("orig_colors")
    if source_cols is not None and len(source_cols) > 0:
        base_color = source_cols.mean(axis=0)
    else:
        base_color = np.array([0.70, 0.65, 0.60])
    coords_hash = np.sin(target_vertices[:, 0] * 37.0 + target_vertices[:, 1] * 73.0 + target_vertices[:, 2] * 109.0)
    return np.clip(base_color[None, :] + coords_hash[:, None] * 0.008, 0.0, 1.0)

    # =========================================================================
    # BRANCH 3: GENERAL FALLBACK / TILES
    # =========================================================================
    if abs(n[1]) < 0.9:
        u_axis = np.cross(n, [0.0, 1.0, 0.0])
    else:
        u_axis = np.cross(n, [1.0, 0.0, 0.0])
    u_axis /= np.maximum(1e-12, np.linalg.norm(u_axis))
    v_axis = np.cross(n, u_axis)
    v_axis /= np.maximum(1e-12, np.linalg.norm(v_axis))

    src_u = (source_pts - p0) @ u_axis
    src_v = (source_pts - p0) @ v_axis
    tgt_u = (target_vertices - p0) @ u_axis
    tgt_v = (target_vertices - p0) @ v_axis

    min_u = min(float(tgt_u.min()), float(src_u.min())) - res_m
    max_u = max(float(tgt_u.max()), float(src_u.max())) + res_m
    min_v = min(float(tgt_v.min()), float(src_v.min())) - res_m
    max_v = max(float(tgt_v.max()), float(src_v.max())) + res_m

    width_px = max(4, min(1024, int(np.ceil((max_u - min_u) / res_m)) + 1))
    height_px = max(4, min(1024, int(np.ceil((max_v - min_v) / res_m)) + 1))

    src_px = np.clip(((src_u - min_u) / res_m).astype(np.int32), 0, width_px - 1)
    src_py = np.clip(((src_v - min_v) / res_m).astype(np.int32), 0, height_px - 1)

    img = np.zeros((height_px, width_px, 3), dtype=np.float64)
    counts = np.zeros((height_px, width_px, 1), dtype=np.float64)
    np.add.at(img, (src_py, src_px), source_cols)
    np.add.at(counts, (src_py, src_px), 1.0)

    observed_mask = (counts[:, :, 0] > 0).astype(np.uint8)
    valid_idx = counts[:, :, 0] > 0
    img[valid_idx] /= counts[valid_idx]

    inpaint_mask = (observed_mask == 0).astype(np.uint8)
    u8_img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if np.any(inpaint_mask):
        inpainted = cv2.inpaint(u8_img, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
        completed_img = inpainted.astype(np.float64) / 255.0
    else:
        completed_img = img

    tgt_px = np.clip(((tgt_u - min_u) / res_m).astype(np.int32), 0, width_px - 1)
    tgt_py = np.clip(((tgt_v - min_v) / res_m).astype(np.int32), 0, height_px - 1)
    return completed_img[tgt_py, tgt_px]


def paint_surface(unit, vertices):
    return analyze_and_synthesize_planar_texture(unit, vertices)


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


def wall_mesh(wall, basis_a, basis_b, vertical_bounds):
    """Build fitted wall regions and restrict only their global vertical extent."""
    offset = float(wall['offset_m'])
    if 'Dir A' in wall['axis']:
        normal, along, full_span = basis_a, basis_b, wall['span_w_m']
    else:
        normal, along, full_span = basis_b, basis_a, wall['span_u_m']
    vertices, faces = [], []
    regions = wall.get('trim_regions') or [{'span_m': full_span, 'bounds_y': wall['bounds_y']}]
    for region in regions:
        left, right = sorted(map(float, region['span_m']))
        bounds = region.get('bounds_y', wall['bounds_y'])
        bottom = max(float(bounds[0]), vertical_bounds[0])
        top = min(float(bounds[1]), vertical_bounds[1])
        if right - left < 1e-8 or top - bottom < 1e-8:
            continue
        p0 = left * along + offset * normal
        p1 = right * along + offset * normal
        start = len(vertices)
        vertices.extend(([p0[0], bottom, p0[2]], [p1[0], bottom, p1[2]],
                         [p1[0], top, p1[2]], [p0[0], top, p0[2]]))
        faces.extend(([start, start + 1, start + 2], [start, start + 2, start + 3]))
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if len(faces):
        face_normal = np.cross(vertices[faces[0, 1]]-vertices[faces[0, 0]], vertices[faces[0, 2]]-vertices[faces[0, 0]])
        if face_normal @ np.array(wall['normal_unit']) < 0:
            faces = faces[:, ::-1]
    return vertices, faces

def main():
    args = parse_args()
    wall_path = args.wall_manifest or args.instances_dir/'wall_planes_manifest.json'
    surface_path = args.instances_dir/'floors_ceilings_planes_manifest.json'
    if not wall_path.is_file() or not surface_path.is_file():
        print('Error: Required fitted geometry manifest not found')
        return 1
    walls = json.loads(wall_path.read_text())
    elements = json.loads(surface_path.read_text()).get('elements', [])
    basis = walls['orthogonal_basis']
    a, b = np.array(basis['dir_A_east_west']), np.array(basis['dir_B_north_south'])
    wall_planes = walls.get('wall_planes', [])
    shape_fit = json.loads(args.surface_shape_manifest.read_text()) if args.surface_shape_manifest else None
    alignment = (shape_fit or {}).get('axis_alignment', {})
    shape_rotation = np.asarray(alignment.get('rotation_matrix', np.eye(3)), dtype=np.float64)
    shape_translation = np.asarray(alignment.get('translation_m', np.zeros(3)), dtype=np.float64)
    shape_records = {
        'floor': list((shape_fit or {}).get('final_aligned_floor_plans', [])),
        'ceiling': list((shape_fit or {}).get('ceilings', [])),
    }
    shape_used = {'floor': set(), 'ceiling': set()}

    def take_shape(role, elevation):
        if shape_fit is None:
            return None
        elevation_key = 'floor_y_m' if role == 'floor' else 'mean_y_m'
        available = [(index, record) for index, record in enumerate(shape_records[role])
                     if index not in shape_used[role]]
        if not available:
            raise ValueError(f'No unused fitted {role} plan remains for a surface at Y={elevation:.3f} m')
        index, record = min(available, key=lambda item: abs(float(item[1][elevation_key]) - elevation))
        difference = abs(float(record[elevation_key]) - elevation)
        if difference > .25:
            raise ValueError(f'Closest fitted {role} plan is {difference:.3f} m from its source instance')
        shape_used[role].add(index)
        return record

    # 0. Global Chromatic Adaptation & White Balance Calibration
    wall_source_path = walls.get('artifacts', {}).get('cleaned_wall_pieces')
    wall_source = o3d.io.read_triangle_mesh(str(wall_source_path)) if wall_source_path and Path(wall_source_path).is_file() else o3d.geometry.TriangleMesh()
    wall_v = np.asarray(wall_source.vertices)
    wall_c = np.asarray(wall_source.vertex_colors) if len(wall_source.vertex_colors) == len(wall_v) else np.full((len(wall_v), 3), 0.6)
    scene_gain = estimate_scene_chromatic_adaptation(wall_c)
    print('Scene Chromatic Adaptation Gain (R, G, B):', np.round(scene_gain, 3), flush=True)

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
        colors = np.clip(colors * scene_gain, 0.0, 1.0)
        centers, areas = compute_face_centers_and_areas(vertices, faces)
        equation = np.array(element['plane_equation'], dtype=float)
        equation /= np.linalg.norm(equation[:3])
        role = element['type']
        elevation = float(np.average(centers[:, 1], weights=areas))
        shape_record = take_shape(role, elevation) if role in shape_records else None
        if role == 'ceiling' and shape_record is not None:
            equation = _aligned_plane_to_source(shape_record['plane_equation'], shape_rotation, shape_translation)
        classification = detect_surface_pattern(centers, areas, normals, equation, (a, b), role == 'ceiling')
        preserve = bool(element.get('preserve_source')) and classification.pattern_type != 'beams'
        if shape_record is not None:
            preserve = False
        distance = np.abs(centers @ equation[:3]+equation[3])
        band = .4 if classification.pattern_type == 'beams' else max(.08, float(element.get('distance_threshold_m', .1)))
        selected = faces if preserve else faces[distance <= band]
        footprint = observed_footprint(vertices, selected, a, b)
        if shape_record is not None:
            if role == 'floor':
                perimeter = shape_record['aligned_perimeter_xz_m']
                cutouts = shape_record.get('no_floor_cutouts_xz_m', [])
            else:
                perimeter = shape_record['perimeter_xz_m']
                cutouts = shape_record.get('shape_cutouts_xz_m', [])
            footprint = _plan_footprint_in_source_frame(
                perimeter, cutouts, shape_rotation, shape_translation
            )
        else:
            regions = wall_regions(wall_planes, a, b, elevation)
            footprint = snap_footprint_to_walls(footprint, wall_planes, a, b)
            footprint = constrain_footprint(footprint, regions)
            footprint = frame_footprint(clean_footprint(footprint), a, b)
        # Read face flags to distinguish original scanned triangles from hole-fill triangles
        face_flags = read_ply_face_flags(source, "is_hole_fill")
        if face_flags is not None and len(face_flags) == len(faces):
            orig_face_mask = (face_flags == 0)
        else:
            orig_face_mask = np.ones(len(faces), dtype=bool)

        if np.any(orig_face_mask):
            orig_v_idx = np.unique(faces[orig_face_mask])
            orig_vertices = vertices[orig_v_idx]
            orig_colors = colors[orig_v_idx]
            orig_tree = cKDTree(orig_vertices)
        else:
            orig_vertices = vertices
            orig_colors = colors
            orig_tree = cKDTree(vertices)

        unit_dict = {'name': element['name'], 'role': role, 'equation': equation, 'classification': classification,
                      'vertices': vertices, 'faces': faces, 'colors': colors, 'footprint': footprint,
                      'orig_vertices': orig_vertices, 'orig_colors': orig_colors, 'orig_tree': orig_tree,
                      'preserve_source': preserve, 'elevation': elevation, 'color_tree': cKDTree(vertices),
                      'shape_source': 'fitted_floor_ceiling_plan' if shape_record is not None else None}

        if not args.skip_texture:
            if role == 'floor':
                unit_dict['seamless_tile'] = synthesize_floor_unit_tile(unit_dict, a, b)
                unit_dict['dir_A'] = a
                unit_dict['dir_B'] = b
                unit_dict['tile_res_m'] = 0.005
            elif role == 'ceiling' and classification.pattern_type == 'beams':
                b_tile, s_off = synthesize_ceiling_unit_tile(unit_dict)
                unit_dict['seamless_tile'] = b_tile
                unit_dict['beam_s_offset'] = s_off
                unit_dict['tile_res_m'] = 0.005

        units.append(unit_dict)
        print(f"{element['name']}: {classification.pattern_type}; source_preserved={preserve}; footprint={footprint.area:.3f}m2 (original_verts={len(orig_vertices):,}/{len(vertices):,})", flush=True)
    if units and shape_fit is None:
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
    wall_vertical_bounds = surface_vertical_bounds(
        units, wall_planes, walls.get('wall_vertical_bounds_m')
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assembled = []
    manifest = {'units': [], 'slabs': [], 'wall_vertical_bounds_m': list(wall_vertical_bounds),
                'surface_shape_manifest': str(args.surface_shape_manifest) if shape_fit is not None else None,
                'summary': {'total_units_analyzed': len(units), 'patterns_detected': {}}}
    for unit in units:
        unit['generated'] = generate_surface(unit)
        v, t = unit['generated']
        v_sub, t_sub = subdivide_surface(v, t, max_edge=0.08) if not unit['preserve_source'] else (v, t)
        colors = np.full((len(v_sub), 3), 0.7) if args.skip_texture else paint_surface(unit, v_sub)
        path = args.output_dir/f"{unit['name']}_regularized.ply"
        export_ply(path, v_sub, t_sub, colors)
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
            'slab_clearance_lift_m': unit.get('lift', 0.), 'shape_source': unit.get('shape_source')})
        pattern = unit['classification'].pattern_type
        manifest['summary']['patterns_detected'][pattern] = manifest['summary']['patterns_detected'].get(pattern, 0)+1
    for _, _, floor, ceiling, overlap in pairs:
        overlap = overlap.difference(floor['consumed']).difference(ceiling['consumed'])
        if overlap.area < .01:
            continue
        bottom, bottom_faces = generate_surface(ceiling, overlap)
        try:
            solid, _, extra_lift = close_slab(bottom, bottom_faces, floor['equation'])
        except ValueError as error:
            if shape_fit is None:
                raise
            manifest['slabs'].append({
                'floor': floor['name'], 'ceiling': ceiling['name'],
                'status': 'omitted_nonwatertight', 'reason': str(error),
                'footprint_xz': mapping(overlap),
            })
            continue
        if extra_lift > 2e-5:
            raise ValueError('Slab floor clearance changed after surface export')
        name = f"slab_inst{len(manifest['slabs']):03d}"
        path = args.output_dir/f'{name}.ply'
        # Paint slab top using floor texture and slab bottom using ceiling texture
        slab_v = solid.vertices
        if args.skip_texture:
            slab_c = np.full((len(slab_v), 3), 0.7, dtype=np.float64)
        else:
            slab_c = np.zeros((len(slab_v), 3), dtype=np.float64)
            top_mask = slab_v[:, 1] >= (slab_v[:, 1].min() + slab_v[:, 1].max()) / 2.0
            if np.any(top_mask):
                slab_c[top_mask] = paint_surface(floor, slab_v[top_mask])
            if np.any(~top_mask):
                slab_c[~top_mask] = paint_surface(ceiling, slab_v[~top_mask])

        solid.visual.vertex_colors = (255 * slab_c).astype(np.uint8)
        solid.export(path)
        assembled.append((solid.vertices, solid.faces, slab_c))
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
            v_sub, t_sub = subdivide_surface(v, t, max_edge=0.08) if not unit['preserve_source'] else (v, t)
            col = np.full((len(v_sub), 3), 0.7) if args.skip_texture else paint_surface(unit, v_sub)
            assembled.append((v_sub, t_sub, col))
    wall_meshes = []
    wall_source_path = walls.get('artifacts', {}).get('cleaned_wall_pieces')
    wall_source = o3d.io.read_triangle_mesh(str(wall_source_path)) if wall_source_path and Path(wall_source_path).is_file() else o3d.geometry.TriangleMesh()
    wall_v = np.asarray(wall_source.vertices)
    wall_c = np.asarray(wall_source.vertex_colors) if len(wall_source.vertex_colors) == len(wall_v) else np.full((len(wall_v), 3), 0.6)
    wall_c = np.clip(wall_c * scene_gain, 0.0, 1.0)

    # Collect all available wall meshes
    wall_source_meshes = []
    if len(wall_source.vertices) > 0:
        wall_source_meshes.append(wall_source)
    for p in sorted(args.instances_dir.glob("wall_inst*.ply")):
        wm = o3d.io.read_triangle_mesh(str(p))
        if len(wm.vertices) > 0:
            wall_source_meshes.append(wm)

    all_v = []
    all_c = []
    all_t = []
    base_idx = 0
    for wm in wall_source_meshes:
        mv = np.asarray(wm.vertices)
        mc = np.asarray(wm.vertex_colors)
        mt = np.asarray(wm.triangles)
        if len(mv) == 0:
            continue
        if len(mc) != len(mv):
            mc = np.full((len(mv), 3), 0.6)
        mc = np.clip(mc * scene_gain, 0.0, 1.0)
        all_v.append(mv)
        all_c.append(mc)
        if len(mt) > 0:
            all_t.append(mt + base_idx)
        base_idx += len(mv)

    if all_v:
        wall_full_v = np.vstack(all_v)
        wall_full_c = np.vstack(all_c)
        wall_full_t = np.vstack(all_t) if all_t else np.empty((0, 3), dtype=np.int32)
    else:
        wall_full_v = np.empty((0, 3), dtype=np.float64)
        wall_full_c = np.empty((0, 3), dtype=np.float64)
        wall_full_t = np.empty((0, 3), dtype=np.int32)

    wall_tree = cKDTree(wall_full_v) if len(wall_full_v) > 0 else None

    if len(wall_full_t) > 0:
        wall_v0 = wall_full_v[wall_full_t[:, 0]]
        wall_v1 = wall_full_v[wall_full_t[:, 1]]
        wall_v2 = wall_full_v[wall_full_t[:, 2]]
        wall_centers = (wall_v0 + wall_v1 + wall_v2) / 3.0
        wall_fnorm = np.cross(wall_v1 - wall_v0, wall_v2 - wall_v0)
        wall_fnorm /= np.maximum(1e-12, np.linalg.norm(wall_fnorm, axis=1, keepdims=True))
    else:
        wall_centers = np.empty((0, 3), dtype=np.float64)
        wall_fnorm = np.empty((0, 3), dtype=np.float64)

    prepared_walls = []
    up_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    for wall in wall_planes:
        v, t = wall_mesh(wall, a, b, wall_vertical_bounds)
        if not len(t):
            continue
        v, t = subdivide_surface(v, t, max_edge=0.15)

        wp_eq = np.array(wall['plane_equation'], dtype=np.float64)
        wp_n = wp_eq[:3] / max(1e-12, np.linalg.norm(wp_eq[:3]))
        wp_d = float(wp_eq[3])

        wp_pts = np.empty((0, 3), dtype=np.float64)
        wp_cols = np.empty((0, 3), dtype=np.float64)

        if not args.skip_texture:
            if len(wall_full_t) > 0:
                dists = np.abs(wall_centers @ wp_n + wp_d)
                inward_mask = (dists < 0.15) & (wall_fnorm @ wp_n > 0.15)
                if np.count_nonzero(inward_mask) < 4:
                    inward_mask = (dists < 0.15)
                if np.count_nonzero(inward_mask) >= 4:
                    inward_v_idx = np.unique(wall_full_t[inward_mask])
                    wp_pts = wall_full_v[inward_v_idx]
                    wp_cols = wall_full_c[inward_v_idx]

            if len(wp_pts) < 4 and wall_tree is not None:
                _, near_idx = wall_tree.query(v, k=1)
                wp_pts = wall_full_v[near_idx]
                wp_cols = wall_full_c[near_idx]

            if len(wp_pts) < 4:
                wp_pts = v
                wp_cols = np.full((len(v), 3), 0.6)

            t_axis = np.cross(up_axis, wp_n)
            t_axis = a if np.linalg.norm(t_axis) < 1e-4 else t_axis / np.linalg.norm(t_axis)

            # 1. Outlier removal (cables, foreign objects, glare, deep shadows) & 2D lighting regularization
            clean_pts, reg_cols, inl_u, inl_v, diag = regularize_wall_lighting_and_outliers(
                wp_pts, wp_cols, t_axis, up_axis
            )

            # 2. Check for bi-color / multi-color split on the same wall
            split_info = detect_wall_color_splits(inl_v, inl_u, reg_cols)
        else:
            t_axis = np.cross(up_axis, wp_n)
            t_axis = a if np.linalg.norm(t_axis) < 1e-4 else t_axis / np.linalg.norm(t_axis)
            clean_pts = v
            reg_cols = np.full((len(v), 3), 0.7)
            diag = {'inlier_ratio': 1.0}
            split_info = {'is_split': False, 'split_type': 'none', 'delta_E': 0.0}

        prepared_walls.append({
            'wall_def': wall,
            'name': wall['name'],
            'plane_equation': wp_eq,
            'v': v,
            't': t,
            'wp_n': wp_n,
            't_axis': t_axis,
            'clean_pts': clean_pts,
            'reg_cols': reg_cols,
            'diag': diag,
            'split_info': split_info,
        })

    # 3. Adjacency graph and color unification across touching plain walls
    if not args.skip_texture:
        prepared_walls = unify_adjacent_wall_colors(prepared_walls)
    target_wall_white = np.array([0.88, 0.88, 0.86])
    for pw in prepared_walls:
        if not pw.get('split_info', {}).get('is_split'):
            pw['unified_color'] = target_wall_white

    for pw in prepared_walls:
        wall = pw['wall_def']
        v, t = pw['v'], pw['t']
        wall_unit = {
            'name': pw['name'],
            'role': 'wall',
            'equation': pw['plane_equation'],
            'vertices': pw['clean_pts'],
            'colors': pw['reg_cols'],
            'orig_vertices': pw['clean_pts'],
            'orig_colors': pw['reg_cols'],
            'orig_tree': cKDTree(pw['clean_pts']) if len(pw['clean_pts']) >= 4 else wall_tree,
            'color_tree': cKDTree(pw['clean_pts']) if len(pw['clean_pts']) >= 4 else wall_tree,
            'u_axis': pw['t_axis'],
            'unified_color': pw.get('unified_color') if pw.get('unified_color') is not None else target_wall_white,
            'split_info': pw.get('split_info'),
            'classification': PatternClassification(
                pattern_type='simple_and_plain',
                confidence=1.0,
                orientation='vertical',
                parameters={},
                summary='Regularized vertical wall plane',
            ),
        }

        colors = np.full((len(v), 3), 0.7) if args.skip_texture else analyze_and_synthesize_planar_texture(wall_unit, v)
        path = args.output_dir/f"{wall['name']}_regularized.ply"
        export_ply(path, v, t, colors)
        wall_meshes.append((v, t, colors))

        manifest['units'].append({
            'unit_name': wall['name'],
            'structural_role': 'wall',
            'regularized_artifact': str(path),
            'plane_equation': wall['plane_equation'],
            'pattern_classification': {
                'pattern_type': 'simple_and_plain',
                'confidence': 1.0,
                'parameters': {
                    'unified_color': (pw.get('unified_color') if pw.get('unified_color') is not None else target_wall_white).tolist(),
                    'is_split': pw['split_info']['is_split'],
                    'split_type': pw['split_info']['split_type'],
                    'delta_E': pw['split_info']['delta_E'],
                    'inlier_ratio': pw['diag'].get('inlier_ratio', 1.0),
                }
            }
        })
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
            clipped_v, clipped_t, clipped_c = clip_mesh_y_bounds(
                wall_v, source_faces[kept], wall_c, *wall_vertical_bounds
            )
            if len(clipped_t):
                wall_meshes.append((clipped_v, clipped_t, clipped_c))
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
