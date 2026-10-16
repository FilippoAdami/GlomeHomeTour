#!/usr/bin/env python3
"""
Refines class boundaries on the 3D mesh using:
1. Dihedral normal crease snapping (stops at angles > 20-25 deg)
2. Photometric color edge snapping (stops at color separation > 0.035)
3. Cabinet face propagation: reclaims pieces of cabinet faces swallowed by the wall
4. Fake wall instance absorption: reclaims flat, low-frequency/low-energy drywall patches into Class 5
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import open3d as o3d
from scipy.sparse import csgraph, csr_matrix


def compute_mesh_derivatives(
    vertices: np.ndarray,
    triangles: np.ndarray,
    vertex_colors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes triangle areas, unit normals, centroids, and average RGB colors."""
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    tri_areas = 0.5 * np.linalg.norm(cross, axis=1)
    tri_normals = cross / np.maximum(2.0 * tri_areas[:, None], 1e-8)
    tri_centroids = (v0 + v1 + v2) / 3.0
    tri_colors = np.mean(vertex_colors[triangles], axis=1)
    return tri_areas, tri_normals, tri_centroids, tri_colors


def build_dual_graph(triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray, csr_matrix]:
    """Constructs the undirected dual mesh adjacency graph (shared edge neighbors)."""
    num_tri = len(triangles)
    e0 = np.sort(triangles[:, [0, 1]], axis=1)
    e1 = np.sort(triangles[:, [1, 2]], axis=1)
    e2 = np.sort(triangles[:, [2, 0]], axis=1)
    all_edges = np.vstack([e0, e1, e2])
    tri_ids = np.tile(np.arange(num_tri, dtype=np.int32), 3)

    packed = (all_edges[:, 0].astype(np.int64) << 32) | all_edges[:, 1].astype(np.int64)
    sort_idx = np.argsort(packed)
    packed_sorted = packed[sort_idx]
    tri_sorted = tri_ids[sort_idx]

    same_edge = (packed_sorted[:-1] == packed_sorted[1:]) & (tri_sorted[:-1] != tri_sorted[1:])
    t1 = tri_sorted[:-1][same_edge]
    t2 = tri_sorted[1:][same_edge]

    adj_u = np.concatenate([t1, t2])
    adj_v = np.concatenate([t2, t1])
    adj_csr = csr_matrix((np.ones(len(adj_u), dtype=bool), (adj_u, adj_v)), shape=(num_tri, num_tri))
    return adj_u, adj_v, adj_csr


def reclaim_ceiling_bounding_slabs(
    labels: np.ndarray,
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    tri_normals: np.ndarray,
    tri_centroids: np.ndarray,
    tri_colors: np.ndarray,
    ceiling_cid: int = 1,
    wall_cid: int = 5,
    min_comp_size: int = 50,
    max_slab_thickness_m: float = 0.25,
    margin_m: float = 0.40,
    min_ceiling_y_m: float = 1.05,
    max_color_diff: float | None = None,
) -> Tuple[np.ndarray, int]:
    """
    Reclaims overhead downward-facing ceiling triangles misclassified as Wall.
    Physical rule: Structural room walls are vertical (|nY| <= 0.20).
    A downward-facing surface (nY < -0.25) located overhead (Y >= min_ceiling_y_m)
    cannot physically be a vertical wall.
    1. Reclaims all overhead downward-facing wall triangles (nY < -0.25, Y >= min_ceiling_y_m).
    2. For each ceiling component, fits the plane via 3x3 covariance and reclaims any
       wall triangles within the 3D bounding slab (|dist| <= max_slab_thickness_m, margin_m)
       that face downward (nY < -0.25).
    """
    cleaned = labels.copy()
    num_tris = len(cleaned)

    ceil_mask = (cleaned == ceiling_cid)
    if not np.any(ceil_mask):
        return cleaned, 0

    total_reclaimed = 0
    ce_mask = ceil_mask[adj_u] & ceil_mask[adj_v]
    u_c = adj_u[ce_mask]
    v_c = adj_v[ce_mask]
    c_idx = np.flatnonzero(ceil_mask)
    c_map = {t: i for i, t in enumerate(c_idx)}
    u_loc = np.array([c_map[t] for t in u_c], dtype=np.int32)
    v_loc = np.array([c_map[t] for t in v_c], dtype=np.int32)
    c_adj = csr_matrix((np.ones(len(u_loc), bool), (u_loc, v_loc)), shape=(len(c_idx), len(c_idx)))
    n_comp, comp_lbls = csgraph.connected_components(c_adj, directed=False)
    comp_sizes = np.bincount(comp_lbls)

    for comp_id in np.argsort(-comp_sizes):
        if comp_sizes[comp_id] < min_comp_size:
            continue
        comp_tris = c_idx[comp_lbls == comp_id]
        pts = tri_centroids[comp_tris]
        centroid = pts.mean(axis=0)
        cov = (pts - centroid).T @ (pts - centroid)
        eigvals, eigvecs = np.linalg.eigh(cov)
        plane_n = eigvecs[:, 0]
        if plane_n[1] > 0:
            plane_n = -plane_n
        if plane_n[1] >= -0.25:
            continue

        min_x, max_x = pts[:, 0].min() - margin_m, pts[:, 0].max() + margin_m
        min_z, max_z = pts[:, 2].min() - margin_m, pts[:, 2].max() + margin_m
        min_y, max_y = pts[:, 1].min() - 0.35, pts[:, 1].max() + 0.35

        in_box = (
            (tri_centroids[:, 0] >= min_x) & (tri_centroids[:, 0] <= max_x) &
            (tri_centroids[:, 2] >= min_z) & (tri_centroids[:, 2] <= max_z) &
            (tri_centroids[:, 1] >= min_y) & (tri_centroids[:, 1] <= max_y)
        )
        dist = np.abs((tri_centroids - centroid) @ plane_n)
        in_slab = dist <= max_slab_thickness_m
        downward = tri_normals[:, 1] < -0.25

        reclaim_cand = (cleaned == wall_cid) & in_box & in_slab & downward
        if max_color_diff is not None:
            c_mean = tri_colors[comp_tris].mean(axis=0)
            c_diff = np.linalg.norm(tri_colors - c_mean, axis=1)
            reclaim_cand = reclaim_cand & (c_diff <= max_color_diff)
        n_reclaim = int(reclaim_cand.sum())
        if n_reclaim > 0:
            cleaned[reclaim_cand] = ceiling_cid
            total_reclaimed += n_reclaim

    return cleaned, total_reclaimed


def reclaim_wall_encroached_furniture(
    labels: np.ndarray,
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    tri_normals: np.ndarray,
    tri_centroids: np.ndarray,
    wall_cid: int = 5,
    foreground_cids: List[int] | None = None,
    **kwargs,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Backward compatibility wrapper delegating to reclaim_thin_wall_plates."""
    return reclaim_thin_wall_plates(
        labels=labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_normals=tri_normals,
        tri_centroids=tri_centroids,
        wall_cid=wall_cid,
    )


def reclaim_thin_wall_plates(
    labels: np.ndarray,
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    tri_normals: np.ndarray,
    tri_centroids: np.ndarray,
    wall_cid: int = 5,
    structural_cids: Set[int] = {1, 5, 14},
    max_normal_dev_deg: float = 35.0,
    max_iterations: int = 25,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Generalizable Physical Wall Thickness & Surface Growing (Scene-Agnostic):
    A structural room wall is a massive planar boundary that cannot physically exist
    on thin horizontal surfaces or protruding foreground boards/shelves/dividers.

    1. Identifies non-structural candidate wall triangles:
       - Horizontal surfaces (|n_Y| > 0.60)
       - Downward surfaces (n_Y < -0.30)
    2. Grows touching foreground classes along smooth/coplanar dual-graph edges
       (|n_u . n_v| > cos(max_normal_dev_deg)) into these non-structural wall candidates.
    3. Propagates iteratively, reclaiming complete continuous boards/shelves/furniture
       surfaces until reaching a sharp geometric crease or the true perimeter wall.
    """
    cleaned = labels.copy()
    cos_thresh = float(np.cos(np.radians(max_normal_dev_deg)))

    # Non-structural candidate wall triangles
    is_wall = (cleaned == wall_cid)
    cand_wall = is_wall & (
        (np.abs(tri_normals[:, 1]) > 0.60) |
        (tri_normals[:, 1] < -0.30)
    )

    stats: Dict[int, int] = {}
    if not np.any(cand_wall):
        return cleaned, stats

    # Multi-pass surface growth along smooth edges
    for it in range(max_iterations):
        is_cand = (cleaned == wall_cid) & cand_wall
        if not np.any(is_cand):
            break

        # Dual-graph boundary edges from foreground classes into candidate wall triangles
        b_edges = (~is_cand[adj_u]) & (~np.isin(cleaned[adj_u], list(structural_cids))) & is_cand[adj_v]
        if not np.any(b_edges):
            break

        u_idx = adj_u[b_edges]
        v_idx = adj_v[b_edges]
        norm_dot = np.sum(tri_normals[u_idx] * tri_normals[v_idx], axis=1)
        smooth_e = (norm_dot >= cos_thresh)
        if not np.any(smooth_e):
            break

        u_smooth = u_idx[smooth_e]
        v_smooth = v_idx[smooth_e]

        # Resolve multi-parent conflicts by taking the best normal alignment
        sort_order = np.argsort(-norm_dot[smooth_e])
        v_sorted = v_smooth[sort_order]
        u_sorted = u_smooth[sort_order]

        _, first_occ = np.unique(v_sorted, return_index=True)
        v_unique = v_sorted[first_occ]
        u_unique = u_sorted[first_occ]

        for u_i, v_i in zip(u_unique, v_unique):
            fg_cid = int(cleaned[u_i])
            cleaned[v_i] = fg_cid
            stats[fg_cid] = stats.get(fg_cid, 0) + 1

    return cleaned, stats


def align_structural_planes(
    labels: np.ndarray,
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    tri_normals: np.ndarray,
    tri_centroids: np.ndarray,
    tri_colors: np.ndarray,
    floor_cid: int = 14,
    ceiling_cid: int = 1,
    wall_cid: int = 5,
    max_floor_tilt_deg: float = 45.0,
    max_floor_plane_dist_m: float = 0.25,
    max_floor_reclaim_dist_m: float = 0.030,
    max_floor_reclaim_tilt_deg: float = 45.0,
    max_floor_reclaim_color_diff: float = 0.040,
    min_ceiling_height_m: float = 0.70,
    reclaim_ceiling_slabs: bool = True,
    reclaim_wall_furniture: bool = False,
    foreground_furniture_cids: List[int] | None = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Aligns Floor, Ceiling, and Wall structural boundaries to their true physical planes.
    1. Floor Non-Vertical & Planar Alignment:
       - Floors must face upward. Ceilings and floors cannot be vertical!
       - Any triangle labeled Floor (#14) that tilts > 45 deg away from upward normal
         (including all vertical surfaces at 90 deg) or departs from fitted floor
         planes is reassigned to wall boundaries.
       - Horizontal floor surfaces on ground or loft levels are preserved.
       - Floor can NEVER be reassigned to Bed (#2).
       - Floor plane roughness up to 45 deg is preserved.
       - Reclaims coplanar, color-matched floor paint into Floor (#14).
    2. Ceiling Non-Vertical Alignment:
       - Ceilings must face downward. Ceilings and floors cannot be vertical!
       - Any triangle labeled Ceiling (#1) that is vertical (nY >= -0.25) or is located
         below the ceiling zone (Y < 0.70 m) is a wall/partition leakage and is reassigned
         to its non-ceiling boundary neighbor (predominantly Wall #5).
    3. Ceiling Bounding Slab Reclaim:
       - True ceiling planes reclaim overhead downward-facing triangles matching ceiling color.
    """
    cleaned = labels.copy()
    stats: Dict[str, int] = {}
    num_tris = len(labels)

    # A. Floor Planes Detection (Level 0 and Level 1 if multi-level)
    up_thresh = float(np.cos(np.radians(max_floor_tilt_deg)))
    up_mask = (cleaned == floor_cid) & (tri_normals[:, 1] > up_thresh)
    floor_planes = []
    if np.any(up_mask):
        for g_mask in [up_mask & (tri_centroids[:, 1] < 0.0), up_mask & (tri_centroids[:, 1] > 0.0)]:
            if np.count_nonzero(g_mask) >= 1000:
                pts = tri_centroids[g_mask]
                A = np.column_stack([pts[:, 0], pts[:, 2], np.ones(len(pts))])
                c, _, _, _ = np.linalg.lstsq(A, pts[:, 1], rcond=None)
                floor_planes.append(c)

    if not floor_planes:
        y_ground = float(np.median(tri_centroids[up_mask, 1])) if np.any(up_mask) else -1.22
        floor_planes.append(np.array([0.0, 0.0, y_ground]))

    # Compute distances to nearest floor plane
    floor_dists = np.min([
        np.abs(tri_centroids[:, 1] - (tri_centroids[:, 0] * c[0] + tri_centroids[:, 2] * c[1] + c[2]))
        for c in floor_planes
    ], axis=0)

    tilt_deg = np.degrees(np.arccos(np.clip(tri_normals[:, 1], -1.0, 1.0)))

    # A1: Floor Leakage Reassignment
    # Horizontal surfaces at ground floor level (Y <= -0.80 m) or loft level (0.70 <= Y <= 1.20 m)
    # with upward tilt (tilt <= 45 deg) are PHYSICALLY FLOOR and MUST NEVER be leaked into Bed!
    is_ground_floor_horizontal = (tri_centroids[:, 1] <= -0.80) & (tri_normals[:, 1] > up_thresh)
    is_loft_floor_horizontal = (tri_centroids[:, 1] >= 0.70) & (tri_centroids[:, 1] <= 1.20) & (tri_normals[:, 1] > up_thresh)

    floor_leak = (cleaned == floor_cid) & ((tilt_deg > max_floor_tilt_deg) | (floor_dists > max_floor_plane_dist_m)) & (~is_ground_floor_horizontal) & (~is_loft_floor_horizontal)
    leak_indices = np.where(floor_leak)[0]
    if len(leak_indices) > 0:
        inv_map = np.full(num_tris, -1, dtype=np.int32)
        inv_map[leak_indices] = np.arange(len(leak_indices), dtype=np.int32)
        m = np.zeros(num_tris, dtype=bool)
        m[leak_indices] = True
        internal_e = m[adj_u] & m[adj_v]
        u_loc = inv_map[adj_u[internal_e]]
        v_loc = inv_map[adj_v[internal_e]]
        sub_adj = csr_matrix((np.ones(len(u_loc), dtype=bool), (u_loc, v_loc)), shape=(len(leak_indices), len(leak_indices)))
        n_comp, comp_lbls = csgraph.connected_components(sub_adj, directed=False)

        tri_comp = np.full(num_tris, -1, dtype=np.int32)
        tri_comp[leak_indices] = comp_lbls
        # Ceilings and floors cannot be vertical!
        # Leaking floor cannot become ceiling unless downward, and can NEVER become Bed (Class 2)!
        v_is_ceil = (cleaned[adj_v] == ceiling_cid)
        u_not_down = (tri_normals[adj_u, 1] >= -0.25)
        v_is_bed = (cleaned[adj_v] == 2)
        invalid_trans = (v_is_ceil & u_not_down) | v_is_bed
        boundary_e = (tri_comp[adj_u] >= 0) & (cleaned[adj_v] != floor_cid) & (~m[adj_v]) & (~invalid_trans)
        if np.any(boundary_e):
            comp_u = tri_comp[adj_u[boundary_e]]
            nbr_v = cleaned[adj_v[boundary_e]]
            sort_k = np.lexsort((nbr_v, comp_u))
            c_sorted = comp_u[sort_k]
            nbr_sorted = nbr_v[sort_k]
            split_mask = np.concatenate([[True], (c_sorted[:-1] != c_sorted[1:]) | (nbr_sorted[:-1] != nbr_sorted[1:])])
            pair_c = c_sorted[split_mask]
            pair_nbr = nbr_sorted[split_mask]
            run_lens = np.diff(np.concatenate([np.flatnonzero(split_mask), [len(c_sorted)]]))
            best_nbr = np.full(n_comp, floor_cid, dtype=np.int32)
            best_count = np.zeros(n_comp, dtype=np.int32)
            for c_id, nbr_id, count in zip(pair_c, pair_nbr, run_lens):
                if count > best_count[c_id]:
                    best_count[c_id] = count
                    best_nbr[c_id] = nbr_id
            reassign_mask = (best_nbr != floor_cid)
            if np.any(reassign_mask):
                valid_reassign_c = np.isin(comp_lbls, np.where(reassign_mask)[0])
                cleaned[leak_indices[valid_reassign_c]] = best_nbr[comp_lbls[valid_reassign_c]]
                stats["floor_leakages_reassigned"] = int(np.sum(valid_reassign_c))
                print(f"  [Structural Align] Reassigned {np.sum(valid_reassign_c):,} floor leakage triangles to touching neighbors")
        else:
            stats["floor_leakages_reassigned"] = 0

    # A2: Floor Plane Reclaim (flat on floor, color match)
    is_floor = (cleaned == floor_cid)
    floor_reclaim = (~is_floor) & (tilt_deg <= max_floor_reclaim_tilt_deg) & (floor_dists <= max_floor_reclaim_dist_m)
    if np.any(is_floor & (floor_dists <= max_floor_reclaim_dist_m)):
        c_mean_floor = tri_colors[is_floor & (floor_dists <= max_floor_reclaim_dist_m)].mean(axis=0)
        c_diff_floor = np.linalg.norm(tri_colors - c_mean_floor, axis=1)
        floor_reclaim = floor_reclaim & (c_diff_floor <= max_floor_reclaim_color_diff) & (cleaned != ceiling_cid) & (cleaned != 2)
        reclaim_indices = np.where(floor_reclaim)[0]
        if len(reclaim_indices) > 0:
            cleaned[reclaim_indices] = floor_cid
            stats["floor_reclaimed"] = int(len(reclaim_indices))
            print(f"  [Structural Align] Reclaimed {len(reclaim_indices):,} flat coplanar triangles into Floor")

    # B: Ceiling Non-Vertical & Low-Height Leakage Reassignment
    # Ceilings cannot be vertical (nY >= -0.25) and cannot be near the floor (Y < min_ceiling_height_m)
    ceil_leak = (cleaned == ceiling_cid) & ((tri_normals[:, 1] >= -0.25) | (tri_centroids[:, 1] < min_ceiling_height_m))
    leak_indices_c = np.where(ceil_leak)[0]
    if len(leak_indices_c) > 0:
        inv_map = np.full(num_tris, -1, dtype=np.int32)
        inv_map[leak_indices_c] = np.arange(len(leak_indices_c), dtype=np.int32)
        m = np.zeros(num_tris, dtype=bool)
        m[leak_indices_c] = True
        internal_e = m[adj_u] & m[adj_v]
        u_loc = inv_map[adj_u[internal_e]]
        v_loc = inv_map[adj_v[internal_e]]
        sub_adj = csr_matrix((np.ones(len(u_loc), dtype=bool), (u_loc, v_loc)), shape=(len(leak_indices_c), len(leak_indices_c)))
        n_comp_c, comp_lbls_c = csgraph.connected_components(sub_adj, directed=False)

        tri_comp = np.full(num_tris, -1, dtype=np.int32)
        tri_comp[leak_indices_c] = comp_lbls_c
        # If leaking ceiling is not upward (nY <= 0.0), it cannot become floor.
        v_is_floor = (cleaned[adj_v] == floor_cid)
        u_not_up = (tri_normals[adj_u, 1] <= 0.0)
        invalid_trans_c = v_is_floor & u_not_up
        boundary_e = (tri_comp[adj_u] >= 0) & (cleaned[adj_v] != ceiling_cid) & (~m[adj_v]) & (~invalid_trans_c)
        if np.any(boundary_e):
            comp_u = tri_comp[adj_u[boundary_e]]
            nbr_v = cleaned[adj_v[boundary_e]]
            sort_k = np.lexsort((nbr_v, comp_u))
            c_sorted = comp_u[sort_k]
            nbr_sorted = nbr_v[sort_k]
            split_mask = np.concatenate([[True], (c_sorted[:-1] != c_sorted[1:]) | (nbr_sorted[:-1] != nbr_sorted[1:])])
            pair_c = c_sorted[split_mask]
            pair_nbr = nbr_sorted[split_mask]
            run_lens = np.diff(np.concatenate([np.flatnonzero(split_mask), [len(c_sorted)]]))
            best_nbr = np.full(n_comp_c, wall_cid, dtype=np.int32)
            best_count = np.zeros(n_comp_c, dtype=np.int32)
            for c_id, nbr_id, count in zip(pair_c, pair_nbr, run_lens):
                if count > best_count[c_id]:
                    best_count[c_id] = count
                    best_nbr[c_id] = nbr_id
            cleaned[leak_indices_c] = best_nbr[comp_lbls_c]
        else:
            cleaned[leak_indices_c] = wall_cid
        stats["ceiling_leakages_reassigned"] = int(len(leak_indices_c))
        print(f"  [Structural Align] Reassigned {len(leak_indices_c):,} non-downward/vertical ceiling triangles into walls")

    # C: Ceiling Thick Bounding Slab Reclaim (Downwards, overhead, matching ceiling paint)
    if reclaim_ceiling_slabs:
        cleaned, n_ceil_reclaimed = reclaim_ceiling_bounding_slabs(
            labels=cleaned,
            adj_u=adj_u,
            adj_v=adj_v,
            tri_normals=tri_normals,
            tri_centroids=tri_centroids,
            tri_colors=tri_colors,
            ceiling_cid=ceiling_cid,
            wall_cid=wall_cid,
        )
        if n_ceil_reclaimed > 0:
            stats["ceiling_slabs_reclaimed"] = n_ceil_reclaimed
            print(f"  [Structural Align] Reclaimed {n_ceil_reclaimed:,} overhead triangles in ceiling bounding slabs")

    # D: Wall Thickness & Thin Plate Reclaim (Generalizable)
    if reclaim_wall_furniture:
        cleaned, furn_stats = reclaim_thin_wall_plates(
            labels=cleaned,
            adj_u=adj_u,
            adj_v=adj_v,
            tri_normals=tri_normals,
            tri_centroids=tri_centroids,
            wall_cid=wall_cid,
        )
        for fcid, cnt in furn_stats.items():
            if cnt > 0:
                stats[f"wall_to_furniture_{fcid}_reclaimed"] = cnt
                print(f"  [Structural Align] Reclaimed {cnt:,} thin wall plate triangles into Class #{fcid}")

    # Final invariant enforcement:
    # Strictly zero vertical or downward triangles on floor:
    f_vert = (cleaned == floor_cid) & (tri_normals[:, 1] <= 0.0)
    if np.any(f_vert):
        f_vert_idx = np.flatnonzero(f_vert)
        is_fv = np.zeros(num_tris, dtype=bool)
        is_fv[f_vert_idx] = True
        fv_boundary = is_fv[adj_u] & (~is_fv[adj_v]) & (cleaned[adj_v] != floor_cid)
        if np.any(fv_boundary):
            fv_u = adj_u[fv_boundary]
            fv_nbr = cleaned[adj_v[fv_boundary]]
            u_map = {}
            for u, nbr in zip(fv_u, fv_nbr):
                if u not in u_map:
                    u_map[u] = nbr
            for u, nbr in u_map.items():
                cleaned[u] = nbr
        rem_fv = (cleaned == floor_cid) & (tri_normals[:, 1] <= 0.0)
        cleaned[rem_fv] = wall_cid

    # Strictly zero vertical or upward triangles on ceiling:
    c_vert = (cleaned == ceiling_cid) & (tri_normals[:, 1] >= -0.25)
    cleaned[c_vert] = wall_cid

    return cleaned, stats


def absorb_enclosed_subparts(
    labels: np.ndarray,
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    tri_normals: np.ndarray,
    tri_centroids: np.ndarray,
    tri_colors: np.ndarray,
    structural_cids: Set[int] | None = None,
    min_enclosure_frac: float = 0.60,
    max_boundary_angle_deg: float = 10.0,
    max_plane_residual_m: float = 0.025,
    max_color_diff: float = 0.035,
    max_color_variance: float = 0.005,
    max_color_gradient: float = 0.008,
    min_component_size: int = 15,
    max_subpart_size: int = 25000,
) -> Tuple[np.ndarray, Dict[int, int], List[int]]:
    """
    Generalized Subpart Absorption:
    Absorbs connected components of triangles into an adjacent enclosing class when:
    1. >= min_enclosure_frac (60%) of the component's perimeter touches a single neighbor class.
    2. The border dihedral crease is smooth (median angle <= max_boundary_angle_deg, e.g. 10 deg).
    3. The component's triangles are coplanar to the neighbor's border plane (residual <= max_plane_residual_m, e.g. 2.5 cm).
    4. The color profile matches the neighbor (delta RGB <= max_color_diff, e.g. 0.035).
    5. The patch lacks high-frequency spatial color texture/variance (protecting real objects like wall posters/decals across all classes).
    6. Does not absorb structural room enclosures (Ceiling/Floor/Wall) into one another.
    """
    cleaned_labels = labels.copy()
    if structural_cids is None:
        structural_cids = {1, 5, 14}

    edge_color_grad = np.linalg.norm(tri_colors[adj_u] - tri_colors[adj_v], axis=1)
    active_classes = np.unique(cleaned_labels)
    num_tris = len(cleaned_labels)

    # Compute connected component sizes across the mesh dual graph
    same_lbl = cleaned_labels[adj_u] == cleaned_labels[adj_v]
    u_same = adj_u[same_lbl]
    v_same = adj_v[same_lbl]
    adj_csr = csr_matrix((np.ones(len(u_same), dtype=bool), (u_same, v_same)), shape=(num_tris, num_tris))
    _, comp_ids_all = csgraph.connected_components(adj_csr, directed=False)
    comp_sizes_all = np.bincount(comp_ids_all)
    tri_comp_size = comp_sizes_all[comp_ids_all]

    inv_map = np.full(num_tris, -1, dtype=np.int32)
    absorbed_counts: Dict[int, int] = {}
    absorbed_classes: List[int] = []

    for cid in active_classes:
        if cid in structural_cids:
            # Structural room shell classes (Ceiling, Floor, Wall) cannot be absorbed
            continue
        idx = np.where(cleaned_labels == cid)[0]
        if len(idx) == 0:
            continue
        inv_map[idx] = np.arange(len(idx), dtype=np.int32)
        m = (cleaned_labels == cid)
        internal_e = m[adj_u] & m[adj_v]
        u_loc = inv_map[adj_u[internal_e]]
        v_loc = inv_map[adj_v[internal_e]]
        inv_map[idx] = -1

        sub_adj = csr_matrix((np.ones(len(u_loc), dtype=bool), (u_loc, v_loc)), shape=(len(idx), len(idx)))
        n_comp, comp_lbls = csgraph.connected_components(sub_adj, directed=False)

        for c in range(n_comp):
            c_sub = idx[comp_lbls == c]
            if len(c_sub) < min_component_size or len(c_sub) > max_subpart_size:
                continue
            c_mask = np.zeros(num_tris, dtype=bool)
            c_mask[c_sub] = True

            total_b = c_mask[adj_u] & (~c_mask[adj_v])
            n_total_b = np.count_nonzero(total_b)
            if n_total_b == 0:
                continue

            nbr_cids = np.unique(cleaned_labels[adj_v[total_b]])
            for n_cid in nbr_cids:
                n_b = c_mask[adj_u] & (cleaned_labels[adj_v] == n_cid)
                frac_n = np.count_nonzero(n_b) / n_total_b
                if frac_n < min_enclosure_frac:
                    continue

                v_b = adj_v[n_b]
                local_nbr_size = int(np.max(tri_comp_size[v_b]))
                # Scale asymmetry: if neighbor is non-structural, candidate component must be smaller than the touching neighbor component
                if n_cid not in structural_cids and len(c_sub) >= local_nbr_size:
                    continue

                u_b = adj_u[n_b]
                v_b = adj_v[n_b]
                dots = np.clip(np.sum(tri_normals[u_b] * tri_normals[v_b], axis=1), -1.0, 1.0)
                med_ang = np.median(np.degrees(np.arccos(dots)))
                if med_ang > max_boundary_angle_deg:
                    continue

                n_norm = np.mean(tri_normals[v_b], axis=0)
                norm_len = np.linalg.norm(n_norm)
                if norm_len < 1e-6:
                    continue
                n_norm /= norm_len
                n_cent = np.mean(tri_centroids[v_b], axis=0)

                p_dists = np.abs(np.dot(tri_centroids[c_sub] - n_cent, n_norm))
                med_p_dist = np.median(p_dists)
                if med_p_dist > max_plane_residual_m:
                    continue

                c_mean = np.mean(tri_colors[c_sub], axis=0)
                n_mean = np.mean(tri_colors[v_b], axis=0)
                c_diff = np.linalg.norm(c_mean - n_mean)
                if c_diff > max_color_diff:
                    continue

                # Universal texture / frequency protection for posters, decals, stickers, badges
                color_var = np.var(tri_colors[c_sub], axis=0).mean()
                if color_var > max_color_variance and c_diff > 0.015:
                    continue
                internal_c_edges = c_mask[adj_u] & c_mask[adj_v]
                if np.count_nonzero(internal_c_edges) > 0:
                    mean_edge_grad = np.mean(edge_color_grad[internal_c_edges])
                    if mean_edge_grad > max_color_gradient and c_diff > 0.015:
                        continue

                print(
                    f"  [Absorb Enclosed] Class #{cid:3d} comp ({len(c_sub):,} tris) -> Absorbed into #{n_cid:3d} "
                    f"(%b={frac_n*100:.1f}%, ang={med_ang:.1f}deg, p_dist={med_p_dist*100:.1f}cm, c_diff={c_diff:.4f})"
                )
                cleaned_labels[c_sub] = n_cid
                tri_comp_size[c_sub] = local_nbr_size + len(c_sub)
                absorbed_counts[n_cid] = absorbed_counts.get(n_cid, 0) + len(c_sub)
                absorbed_classes.append(int(cid))
                break

    return cleaned_labels, absorbed_counts, absorbed_classes


def absorb_flush_wall_instances(
    labels: np.ndarray,
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    tri_normals: np.ndarray,
    tri_centroids: np.ndarray,
    tri_colors: np.ndarray,
    wall_cid: int = 5,
    min_wall_boundary_frac: float = 0.60,
    max_boundary_angle_deg: float = 8.0,
    max_plane_dist_m: float = 0.025,
    max_wall_color_dist: float = 0.15,
    max_color_gradient: float = 0.008,
    max_color_variance: float = 0.005,
) -> Tuple[np.ndarray, List[int]]:
    """Backward compatibility wrapper delegating to absorb_enclosed_subparts."""
    cleaned, stats, absorbed_cids = absorb_enclosed_subparts(
        labels=labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_normals=tri_normals,
        tri_centroids=tri_centroids,
        tri_colors=tri_colors,
        structural_cids={1, 14, wall_cid},
        min_enclosure_frac=min_wall_boundary_frac,
        max_boundary_angle_deg=max_boundary_angle_deg,
        max_plane_residual_m=max_plane_dist_m,
        max_color_diff=max_wall_color_dist,
        max_color_variance=max_color_variance,
        max_color_gradient=max_color_gradient,
        min_component_size=1,
    )
    return cleaned, absorbed_cids


def propagate_planar_faces(
    labels: np.ndarray,
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    adj_csr: csr_matrix,
    tri_normals: np.ndarray,
    tri_centroids: np.ndarray,
    tri_colors: np.ndarray,
    tri_contended: np.ndarray | None = None,
    candidate_cids: List[int] | None = None,
    structural_cids: Set[int] | None = None,
    max_crease_deg: float = 20.0,
    max_color_diff: float = 0.035,
    max_plane_residual_m: float = 0.020,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Generalized Planar Face Propagation:
    Propagates planar faces of classes into neighboring bleed triangles that lie on the same plane and color.
    Stops strictly at:
    - Dihedral creases (> max_crease_deg, e.g. 20 deg)
    - Color edges (> max_color_diff, e.g. 0.035)
    - Departure from the face plane (> max_plane_residual_m, e.g. 2.0 cm)
    - Structural boundaries (Ceiling, Floor, Wall never absorb each other)
    """
    cleaned_labels = labels.copy()
    if candidate_cids is None:
        candidate_cids = sorted(np.unique(cleaned_labels).tolist())
    if structural_cids is None:
        structural_cids = {1, 5, 14}

    reclaimed_counts = {}
    class_sizes = {c: np.count_nonzero(cleaned_labels == c) for c in np.unique(cleaned_labels)}

    for cid in candidate_cids:
        m = (cleaned_labels == cid)
        if np.count_nonzero(m) == 0:
            continue

        diff = m[adj_u] & (cleaned_labels[adj_v] != cid)
        if np.count_nonzero(diff) == 0:
            continue

        u_b = adj_u[diff]
        v_b = adj_v[diff]

        dots = np.clip(np.sum(tri_normals[u_b] * tri_normals[v_b], axis=1), -1.0, 1.0)
        smooth = np.degrees(np.arccos(dots)) < 15.0
        if np.count_nonzero(smooth) == 0:
            continue

        u_sm = u_b[smooth]
        v_sm = v_b[smooth]

        claimed = set()
        queue = deque()
        plane_norm = {}
        plane_cent = {}

        for u, v in zip(u_sm, v_sm):
            if v not in claimed:
                nbr_c = cleaned_labels[v]
                if cid in structural_cids and nbr_c in structural_cids:
                    continue
                if class_sizes.get(nbr_c, 0) > class_sizes.get(cid, 0) and tri_contended is not None and not tri_contended[v]:
                    continue

                p_dist = abs(np.dot(tri_normals[u], tri_centroids[v] - tri_centroids[u]))
                c_diff = np.linalg.norm(tri_colors[u] - tri_colors[v])
                if p_dist < 0.015 and c_diff < max_color_diff:
                    claimed.add(v)
                    plane_norm[v] = tri_normals[u]
                    plane_cent[v] = tri_centroids[u]
                    queue.append(v)

        while queue:
            curr = queue.popleft()
            n_ref = plane_norm[curr]
            c_ref = plane_cent[curr]

            for nbr in adj_csr[curr].indices:
                if cleaned_labels[nbr] != cid and nbr not in claimed:
                    nbr_c = cleaned_labels[nbr]
                    if cid in structural_cids and nbr_c in structural_cids:
                        continue
                    if class_sizes.get(nbr_c, 0) > class_sizes.get(cid, 0) and tri_contended is not None and not tri_contended[nbr]:
                        continue

                    # 1. Crease check to parent
                    ang_p = np.degrees(np.arccos(np.clip(np.dot(tri_normals[curr], tri_normals[nbr]), -1.0, 1.0)))
                    if ang_p > max_crease_deg:
                        continue

                    # 2. Color difference to parent
                    if np.linalg.norm(tri_colors[curr] - tri_colors[nbr]) > max_color_diff:
                        continue

                    # 3. Alignment to the reference face plane
                    ang_ref = np.degrees(np.arccos(np.clip(np.dot(tri_normals[nbr], n_ref), -1.0, 1.0)))
                    p_dist = abs(np.dot(n_ref, tri_centroids[nbr] - c_ref))
                    if ang_ref < 15.0 and p_dist < max_plane_residual_m:
                        claimed.add(nbr)
                        plane_norm[nbr] = n_ref
                        plane_cent[nbr] = c_ref
                        queue.append(nbr)

        if claimed:
            claimed_arr = np.array(list(claimed), dtype=np.int32)
            cleaned_labels[claimed_arr] = cid
            reclaimed_counts[cid] = len(claimed)
            print(f"  [Planar Face Reclaim] Class #{cid:3d} reclaimed {len(claimed):,} triangles across planar boundaries")

    return cleaned_labels, reclaimed_counts


def propagate_planar_boundaries(
    labels: np.ndarray,
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    adj_csr: csr_matrix,
    tri_normals: np.ndarray,
    tri_centroids: np.ndarray,
    tri_colors: np.ndarray,
    tri_contended: np.ndarray | None = None,
    min_planar_dominance_frac: float = 0.70,
    min_plane_size: int = 50,
    max_crease_deg: float = 15.0,
    max_color_diff: float = 0.035,
    max_plane_residual_m: float = 0.020,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Backward compatibility wrapper delegating to propagate_planar_faces."""
    return propagate_planar_faces(
        labels=labels,
        adj_u=adj_u,
        adj_v=adj_v,
        adj_csr=adj_csr,
        tri_normals=tri_normals,
        tri_centroids=tri_centroids,
        tri_colors=tri_colors,
        tri_contended=tri_contended,
        max_crease_deg=max_crease_deg,
        max_color_diff=max_color_diff,
        max_plane_residual_m=max_plane_residual_m,
    )


def propagate_cabinet_faces(
    labels: np.ndarray,
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    adj_csr: csr_matrix,
    tri_normals: np.ndarray,
    tri_centroids: np.ndarray,
    tri_colors: np.ndarray,
    tri_contended: np.ndarray | None = None,
    wall_cid: int = 5,
    candidate_cids: List[int] | None = None,
    max_crease_deg: float = 20.0,
    max_color_diff: float = 0.035,
    max_plane_residual_m: float = 0.020,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Backward compatibility wrapper delegating to propagate_planar_faces."""
    return propagate_planar_faces(
        labels=labels,
        adj_u=adj_u,
        adj_v=adj_v,
        adj_csr=adj_csr,
        tri_normals=tri_normals,
        tri_centroids=tri_centroids,
        tri_colors=tri_colors,
        tri_contended=tri_contended,
        candidate_cids=candidate_cids,
        max_crease_deg=max_crease_deg,
        max_color_diff=max_color_diff,
        max_plane_residual_m=max_plane_residual_m,
    )


def absorb_embedded_furniture_subparts(
    labels: np.ndarray,
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    tri_normals: np.ndarray,
    tri_centroids: np.ndarray,
    tri_colors: np.ndarray,
    furniture_cids: List[int] | None = None,
    min_furniture_boundary_frac: float = 0.60,
    max_boundary_angle_deg: float = 10.0,
    max_plane_residual_m: float = 0.025,
    max_color_diff: float = 0.035,
    min_component_size: int = 15,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Backward compatibility wrapper delegating to absorb_enclosed_subparts."""
    cleaned, stats, _ = absorb_enclosed_subparts(
        labels=labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_normals=tri_normals,
        tri_centroids=tri_centroids,
        tri_colors=tri_colors,
        min_enclosure_frac=min_furniture_boundary_frac,
        max_boundary_angle_deg=max_boundary_angle_deg,
        max_plane_residual_m=max_plane_residual_m,
        max_color_diff=max_color_diff,
        min_component_size=min_component_size,
    )
    return cleaned, stats


def refine_all_boundaries(
    mesh_path: Path,
    clean_mesh_path: Path,
    labels: np.ndarray,
    contention_mesh_path: Path | None = None,
) -> Tuple[np.ndarray, List[int], Dict[int, int]]:
    """Runs the generalized, non-scene-specific, non-class-specific boundary refinement."""
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    clean_mesh = o3d.io.read_triangle_mesh(str(clean_mesh_path))

    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    vc = np.asarray(clean_mesh.vertex_colors)

    tri_areas, tri_normals, tri_centroids, tri_colors = compute_mesh_derivatives(verts, tris, vc)
    adj_u, adj_v, adj_csr = build_dual_graph(tris)

    # Contention flags
    tri_contended = None
    if contention_mesh_path is not None and contention_mesh_path.is_file():
        c_mesh = o3d.io.read_triangle_mesh(str(contention_mesh_path))
        vc_c = np.asarray(c_mesh.vertex_colors)
        if len(vc_c) == len(verts):
            tri_contended = np.all(vc_c[tris] == 1.0, axis=(1, 2))

    # Step 1: Structural Plane Alignment (Floor, Ceiling, Wall planes & leakages)
    print("\n--- Refine Boundaries Step 1: Structural Plane Alignment ---")
    labels, plane_stats = align_structural_planes(
        labels=labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_normals=tri_normals,
        tri_centroids=tri_centroids,
        tri_colors=tri_colors,
        floor_cid=14,
        ceiling_cid=1,
        wall_cid=5,
    )

    # Step 2: Universal Texture-Energy-Gated Subpart Absorption
    print("\n--- Refine Boundaries Step 2: Universal Texture-Gated Subpart Absorption ---")
    labels, absorbed_counts, absorbed_cids = absorb_enclosed_subparts(
        labels=labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_normals=tri_normals,
        tri_centroids=tri_centroids,
        tri_colors=tri_colors,
        min_enclosure_frac=0.60,
        max_boundary_angle_deg=10.0,
        max_plane_residual_m=0.025,
        max_color_diff=0.035,
        max_color_variance=0.005,
        max_color_gradient=0.008,
    )

    reclaimed_stats: Dict[int, int] = {}
    for cid, cnt in absorbed_counts.items():
        reclaimed_stats[cid] = reclaimed_stats.get(cid, 0) + cnt
    for k, cnt in plane_stats.items():
        reclaimed_stats[hash(k) % 1000] = cnt

    return labels, absorbed_cids, reclaimed_stats


def main():
    parser = argparse.ArgumentParser(description="Refine class boundaries using normals and photometric colors.")
    parser.add_argument("--mesh", type=Path, default=Path("backend/current_scene/06c_semantic_segmentation/step2_mesh_projection/mesh_projected_refined.ply"))
    parser.add_argument("--clean-mesh", type=Path, default=Path("backend/current_scene/06c_semantic_segmentation/step1_cleaned_mesh/mesh_cleaned.ply"))
    parser.add_argument("--labels", type=Path, default=Path("backend/current_scene/06c_semantic_segmentation/step2_mesh_projection/triangle_class_labels.npy"))
    parser.add_argument("--contention-mesh", type=Path, default=Path("backend/current_scene/06c_semantic_segmentation/step2_mesh_projection/mesh_projected_contention.ply"))
    parser.add_argument("--dry-run", action="store_true", help="Do not overwrite input labels.")
    args = parser.parse_args()

    print(f"Loading mesh: {args.mesh}...")
    t0 = time.perf_counter()
    labels = np.load(args.labels)

    refined_labels, absorbed_cids, reclaimed_stats = refine_all_boundaries(
        mesh_path=args.mesh,
        clean_mesh_path=args.clean_mesh,
        labels=labels,
        contention_mesh_path=args.contention_mesh,
    )
    print(f"Boundary refinement completed in {time.perf_counter() - t0:.2f}s")

    if not args.dry_run:
        np.save(args.labels, refined_labels)
        print(f"\nSaved refined labels to {args.labels}")
    else:
        print("\n[Dry Run] Did not modify disk.")


if __name__ == "__main__":
    main()
