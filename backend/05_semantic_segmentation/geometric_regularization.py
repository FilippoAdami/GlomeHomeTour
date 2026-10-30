#!/usr/bin/env python3
"""Evidence-preserving mesh refinement and compact physical-part association.

Older cleanup functions remain available for explicit diagnostic use; production
does not infer ceiling membership from normals or force unknown geometry to a label.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import open3d as o3d
import scipy.sparse as sp
from scipy.sparse import csgraph

from detect_structural_classes import (
    build_mesh_dual_graph,
    estimate_gravity_and_scene_bands,
)
from refine_class_boundaries import (
    compute_mesh_derivatives,
)


def regularize_disconnected_manifold_components(
    mesh: o3d.geometry.TriangleMesh,
    labels: np.ndarray,
    min_tris: int = 15,
    max_tris: int = 500000,
    dom_ratio_thresh: float = 0.35,
) -> Tuple[np.ndarray, int]:
    """
    Unifies standalone physical objects that form isolated connected 3D manifold components
    in the triangle mesh (e.g. Punching Bag, ceiling lamps, hanging props).
    If a component's dominant class holds >= dom_ratio_thresh of valid triangles,
    all other minority sub-classes on that component are merged into the dominant class.
    """
    cleaned = labels.copy()
    clusters, num_clusters, areas = mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    counts = np.bincount(clusters)
    reassigned = 0

    for c_idx, cnt in enumerate(counts):
        if cnt < min_tris or cnt > max_tris:
            continue
        c_tri_idx = np.where(clusters == c_idx)[0]
        sub_lbls = cleaned[c_tri_idx]
        valid_sub = sub_lbls[sub_lbls > 0]
        if len(valid_sub) == 0:
            continue
        vals, c_counts = np.unique(valid_sub, return_counts=True)
        dom_idx = c_counts.argmax()
        dom_cid = int(vals[dom_idx])
        if c_counts[dom_idx] / len(valid_sub) >= dom_ratio_thresh:
            diff_mask = c_tri_idx[sub_lbls != dom_cid]
            if len(diff_mask) > 0:
                cleaned[c_tri_idx] = dom_cid
                reassigned += len(diff_mask)

    return cleaned, reassigned


def regularize_foreground_connected_components(
    tri_centroids: np.ndarray,
    labels: np.ndarray,
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    structural_cids: Set[int],
    compact_span_thresh: float = 1.60,
    compact_ratio_thresh: float = 0.30,
    furniture_span_thresh: float = 3.50,
    furniture_dom_ratio: float = 0.45,
    furniture_minor_ratio: float = 0.25,
    tri_normals: Optional[np.ndarray] = None,
    tri_areas: Optional[np.ndarray] = None,
    gravity_up: Optional[np.ndarray] = None,
    edge_lengths: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """
    Reconciles compact, physically continuous foreground sub-parts.

    With mesh derivatives, component edges across large horizontal supports and
    concave contacts are cut before aggregation.  A label may join the dominant
    compact object only through substantial shared mesh boundary; this avoids
    merging objects which merely touch.  Without those derivatives this retains
    the historical, conservative tiny-sliver cleanup for compatibility.
    """
    cleaned = labels.copy()
    num_tris = len(labels)
    struct_arr = np.array(list(structural_cids), dtype=np.int32)
    fg_mask = ~np.isin(cleaned, struct_arr) & (cleaned > 0)
    fg_indices = np.flatnonzero(fg_mask)
    if len(fg_indices) == 0:
        return cleaned, 0

    inv_map = np.full(num_tris, -1, dtype=np.int32)
    inv_map[fg_indices] = np.arange(len(fg_indices), dtype=np.int32)

    fg_edges = fg_mask[adj_u] & fg_mask[adj_v]

    # Older callers intentionally have no physical edge measurements.  Keep their
    # conservative behaviour rather than inferring object identity from topology.
    geometry_available = (
        tri_normals is not None and tri_areas is not None and edge_lengths is not None
        and len(tri_normals) == num_tris and len(tri_areas) == num_tris
        and len(edge_lengths) == len(adj_u)
    )
    if not geometry_available:
        u_loc = inv_map[adj_u[fg_edges]]
        v_loc = inv_map[adj_v[fg_edges]]
        adj_fg = sp.csr_matrix(
            (np.ones(len(u_loc), bool), (u_loc, v_loc)),
            shape=(len(fg_indices), len(fg_indices)),
        )
        n_comp, comp_ids = csgraph.connected_components(adj_fg, directed=False)
        comp_sizes = np.bincount(comp_ids)
        reassigned = 0
        for c in range(n_comp):
            if comp_sizes[c] < 15:
                continue
            c_tris = fg_indices[comp_ids == c]
            sub_lbls = cleaned[c_tris]
            vals, c_counts = np.unique(sub_lbls, return_counts=True)
            dom_cid = int(vals[c_counts.argmax()])
            for value, count in zip(vals, c_counts):
                if value != dom_cid and count < 150 and count / len(sub_lbls) < 0.05:
                    m_sub = c_tris[sub_lbls == value]
                    cleaned[m_sub] = dom_cid
                    reassigned += len(m_sub)
        return cleaned, reassigned

    normals = np.asarray(tri_normals, dtype=np.float64)
    areas = np.asarray(tri_areas, dtype=np.float64)
    lengths = np.asarray(edge_lengths, dtype=np.float64)
    if gravity_up is None:
        up = np.array([0.0, 1.0, 0.0])
    else:
        up = np.asarray(gravity_up, dtype=np.float64)
    up_norm = np.linalg.norm(up)
    if up_norm <= 1e-8:
        raise ValueError("gravity_up must be a non-zero vector")
    up = up / up_norm

    # Find broad horizontal patches independently of their semantic IDs.  A shelf
    # often shares its label with vertical rims or legs, so a class-wide normal
    # fraction cannot identify its top reliably.
    support = np.zeros(num_tris, dtype=bool)
    up_alignment = np.abs(normals @ up)
    support_scale = compact_span_thresh / 1.60
    horizontal = up_alignment >= 0.80
    horizontal_indices = np.flatnonzero(horizontal)
    horizontal_map = np.full(num_tris, -1, dtype=np.int32)
    horizontal_map[horizontal_indices] = np.arange(len(horizontal_indices), dtype=np.int32)
    smooth_horizontal_edge = (
        horizontal[adj_u] & horizontal[adj_v]
        & (np.einsum("ij,ij->i", normals[adj_u], normals[adj_v]) >= np.cos(np.deg2rad(25.0)))
    )
    horizontal_graph = sp.csr_matrix(
        (
            np.ones(np.count_nonzero(smooth_horizontal_edge), bool),
            (horizontal_map[adj_u[smooth_horizontal_edge]], horizontal_map[adj_v[smooth_horizontal_edge]]),
        ),
        shape=(len(horizontal_indices), len(horizontal_indices)),
    )
    _, horizontal_components = csgraph.connected_components(horizontal_graph, directed=False)
    foreground_lengths = lengths[fg_mask[adj_u] & fg_mask[adj_v]]
    edge_scale = float(np.median(foreground_lengths)) if len(foreground_lengths) else 0.01 * support_scale
    max_support_height = max(0.04 * support_scale, 2.0 * edge_scale)
    horizontal_order = np.argsort(horizontal_components, kind="stable")
    horizontal_bounds = np.r_[0, np.flatnonzero(np.diff(horizontal_components[horizontal_order])) + 1, len(horizontal_order)]
    for start, end in zip(horizontal_bounds[:-1], horizontal_bounds[1:]):
        patch = horizontal_indices[horizontal_order[start:end]]
        if float(areas[patch].sum()) <= 0.08 * support_scale ** 2:
            continue
        points = tri_centroids[patch]
        projected = points - np.outer(points @ up, up)
        if np.linalg.norm(np.ptp(projected, axis=0)) <= 0.45 * support_scale:
            continue
        plane_center = np.median(points, axis=0)
        mean_normal = normals[patch].mean(axis=0)
        mean_normal /= max(np.linalg.norm(mean_normal), 1e-8)
        _, singular_values, axes = np.linalg.svd(points - plane_center, full_matrices=False)
        plane_normal = axes[-1] if singular_values[1] > 1e-6 * singular_values[0] else mean_normal
        if plane_normal @ up < 0:
            plane_normal = -plane_normal
        residual = np.abs((points - plane_center) @ plane_normal)
        if (abs(plane_normal @ up) < 0.80
                or np.percentile(residual, 95) > max_support_height):
            continue
        if np.mean(np.abs(normals[patch] @ mean_normal) >= np.cos(np.deg2rad(25.0))) >= 0.90:
            support[patch] = True
            # Include only the connected tessellation-scale rim near the support plane.
            # A support lip extends below its top; objects above it remain eligible parts.
            plane_distance = (tri_centroids - plane_center) @ plane_normal
            rim = (plane_distance >= -2.0 * edge_scale) & (plane_distance <= 0.5 * edge_scale)
            rim_indices = np.flatnonzero(rim)
            rim_map = np.full(num_tris, -1, dtype=np.int32)
            rim_map[rim_indices] = np.arange(len(rim_indices), dtype=np.int32)
            rim_edges = rim[adj_u] & rim[adj_v]
            rim_edges &= np.linalg.norm(tri_centroids[adj_u] - tri_centroids[adj_v], axis=1) <= 2.0 * edge_scale
            rim_graph = sp.csr_matrix(
                (np.ones(np.count_nonzero(rim_edges), bool), (rim_map[adj_u[rim_edges]], rim_map[adj_v[rim_edges]])),
                shape=(len(rim_indices), len(rim_indices)),
            )
            _, rim_components = csgraph.connected_components(rim_graph, directed=False)
            patch_rim = rim_map[patch]
            patch_components = np.unique(rim_components[patch_rim[patch_rim >= 0]])
            support[rim_indices[np.isin(rim_components, patch_components)]] = True

    # An outward-oriented convex surface gives a negative centroid/normal score;
    # a positive score identifies an inward (concave) seam.  Smooth curved parts
    # stay connected because their normal change is below the crease threshold.
    edge_normals_dot = np.einsum("ij,ij->i", normals[adj_u], normals[adj_v])
    normal_angle_cut = edge_normals_dot < np.cos(np.deg2rad(45.0))
    centroid_delta = tri_centroids[adj_v] - tri_centroids[adj_u]
    normal_delta = normals[adj_u] - normals[adj_v]
    seam_score = np.einsum("ij,ij->i", centroid_delta, normal_delta)
    centroid_distance = np.linalg.norm(centroid_delta, axis=1)
    concave = seam_score > 0.15 * np.maximum(centroid_distance, 1e-8)
    cross_label = cleaned[adj_u] != cleaned[adj_v]
    blocked = (support[adj_u] | support[adj_v]) | (cross_label & normal_angle_cut & concave)
    fg_edges &= ~blocked

    u_loc = inv_map[adj_u[fg_edges]]
    v_loc = inv_map[adj_v[fg_edges]]
    adj_fg = sp.csr_matrix(
        (np.ones(len(u_loc), bool), (u_loc, v_loc)),
        shape=(len(fg_indices), len(fg_indices)),
    )
    n_comp, comp_ids = csgraph.connected_components(adj_fg, directed=False)
    reassigned = 0

    component_for_triangle = np.full(num_tris, -1, dtype=np.int32)
    component_for_triangle[fg_indices] = comp_ids
    contact_edges = fg_edges & cross_label
    edge_indices = np.flatnonzero(contact_edges)
    # Accept either one edge per manifold boundary or both directed copies.
    if len(edge_indices):
        edge_low = np.minimum(adj_u[edge_indices], adj_v[edge_indices]).astype(np.int64)
        edge_high = np.maximum(adj_u[edge_indices], adj_v[edge_indices]).astype(np.int64)
        _, keep = np.unique(edge_low * num_tris + edge_high, return_index=True)
        edge_indices = edge_indices[keep]
    edge_components = component_for_triangle[adj_u[edge_indices]]
    order = np.argsort(edge_components, kind="stable")
    edge_indices = edge_indices[order]
    edge_components = edge_components[order]
    starts = np.searchsorted(edge_components, np.arange(n_comp), side="left")
    ends = np.searchsorted(edge_components, np.arange(n_comp), side="right")

    component_order = np.argsort(comp_ids, kind="stable")
    component_sizes = np.bincount(comp_ids, minlength=n_comp)
    component_bounds = np.r_[0, np.cumsum(component_sizes)]
    for c in np.flatnonzero(component_sizes >= 2):
        c_tris = fg_indices[component_order[component_bounds[c]:component_bounds[c + 1]]]
        pts = tri_centroids[c_tris]
        spans = pts.max(axis=0) - pts.min(axis=0)
        max_span = float(max(spans))
        sub_lbls = cleaned[c_tris]
        vals, inverse = np.unique(sub_lbls, return_inverse=True)
        if len(vals) < 2 or max_span > compact_span_thresh:
            continue
        label_areas = np.bincount(inverse, weights=areas[c_tris], minlength=len(vals))
        dom_idx = int(label_areas.argmax())
        dom_cid = int(vals[dom_idx])
        if label_areas[dom_idx] / label_areas.sum() < compact_ratio_thresh:
            continue

        parent = np.arange(len(vals), dtype=np.int32)

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = int(parent[index])
            return index

        component_edges = edge_indices[starts[c]:ends[c]]
        if not len(component_edges):
            continue
        left = np.searchsorted(vals, cleaned[adj_u[component_edges]])
        right = np.searchsorted(vals, cleaned[adj_v[component_edges]])
        pair_low = np.minimum(left, right)
        pair_high = np.maximum(left, right)
        pair_key = pair_low * len(vals) + pair_high
        pair_ids, pair_inverse = np.unique(pair_key, return_inverse=True)
        contact_lengths = np.bincount(pair_inverse, weights=lengths[component_edges])
        for key, contact_length in zip(pair_ids, contact_lengths):
            left_i, right_i = divmod(int(key), len(vals))
            threshold = 0.30 * np.sqrt(min(label_areas[left_i], label_areas[right_i]))
            if contact_length >= threshold:
                root_left, root_right = find(left_i), find(right_i)
                if root_left != root_right:
                    parent[root_right] = root_left

        dominant_root = find(dom_idx)
        for i, cid in enumerate(vals):
            if i != dom_idx and find(i) == dominant_root:
                m_sub = c_tris[sub_lbls == cid]
                cleaned[m_sub] = dom_cid
                reassigned += len(m_sub)

    # A support cut may clip the bottom of an otherwise corroborated source part.
    # Complete that compact part only when its own area overwhelmingly joined one owner.
    same_source = (fg_mask[adj_u] & fg_mask[adj_v]
                   & (labels[adj_u] == labels[adj_v]))
    source_graph = sp.csr_matrix(
        (np.ones(np.count_nonzero(same_source), bool),
         (inv_map[adj_u[same_source]], inv_map[adj_v[same_source]])),
        shape=(len(fg_indices), len(fg_indices)))
    _, source_components = csgraph.connected_components(source_graph, directed=False)
    changed_parts = np.unique(source_components[cleaned[fg_indices] != labels[fg_indices]])
    source_order = np.argsort(source_components, kind="stable")
    source_bounds = np.r_[0, np.cumsum(np.bincount(source_components))]
    for part in changed_parts:
        faces = fg_indices[source_order[source_bounds[part]:source_bounds[part + 1]]]
        if (np.ptp(tri_centroids[faces], axis=0).max() > compact_span_thresh
                or areas[faces][support[faces]].sum() > 0.5 * areas[faces].sum()):
            continue
        owners, owner_inverse = np.unique(cleaned[faces], return_inverse=True)
        owner_area = np.bincount(owner_inverse, weights=areas[faces])
        winner = int(owners[owner_area.argmax()])
        if winner != labels[faces[0]] and owner_area.max() >= 0.75 * owner_area.sum():
            cleaned[faces] = winner

    return cleaned, int(np.count_nonzero(cleaned != labels))


def absorb_wall_mounted_coplanar_patches(
    tri_centroids: np.ndarray,
    tri_normals: np.ndarray,
    labels: np.ndarray,
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    structural_cids: Set[int],
    max_thickness: float = 0.08,
) -> Tuple[np.ndarray, int]:
    """
    Unifies planar wall-mounted items (posters, decals, picture frames).
    1. If a planar wall frame/poster exists (thin in normal direction), any internal foreground
       patches (such as face photos inside a poster) are absorbed into the host poster class.
    2. If a small planar patch is coplanar with the wall and touches a host wall-mounted class,
       absorbs it into the host class.
    """
    cleaned = labels.copy()
    num_tris = len(cleaned)
    reassigned = 0
    struct_arr = np.array(list(structural_cids), dtype=np.int32)

    # 1. Planar wall frames / posters absorption (strictly vertical posters like Class 14)
    for cid in [14]:
        c_m = (cleaned == cid)
        cnt = np.count_nonzero(c_m)
        if cnt < 500 or cnt > 50000:
            continue
        pts = tri_centroids[c_m]
        bmin = pts.min(axis=0)
        bmax = pts.max(axis=0)
        spans = bmax - bmin
        thin_dim = 2 if spans[2] < spans[0] else 0
        is_vert_poster = (np.abs(tri_normals[c_m, 1]) < 0.15).mean() > 0.85
        if spans[thin_dim] <= 0.035 and spans[1] >= 0.30 and is_vert_poster:
            in_frame = (
                (tri_centroids[:, 0] >= bmin[0] - 0.01) & (tri_centroids[:, 0] <= bmax[0] + 0.01) &
                (tri_centroids[:, 1] >= bmin[1] - 0.01) & (tri_centroids[:, 1] <= bmax[1] + 0.01) &
                (tri_centroids[:, 2] >= bmin[2] - 0.02) & (tri_centroids[:, 2] <= bmax[2] + 0.02)
            )
            absorb_m = in_frame & (~np.isin(cleaned, struct_arr)) & (cleaned != cid) & (cleaned > 0)
            if np.any(absorb_m):
                cleaned[absorb_m] = cid
                reassigned += int(np.count_nonzero(absorb_m))

    # 2. Touching wall-mounted coplanar patch absorption
    for cid in np.unique(cleaned):
        if cid in structural_cids or cid <= 0:
            continue
        c_tris = np.flatnonzero(cleaned == cid)
        if len(c_tris) == 0 or len(c_tris) > 25000:
            continue
        pts = tri_centroids[c_tris]
        spans = pts.max(axis=0) - pts.min(axis=0)
        thick = min(spans[0], spans[2])
        if thick < max_thickness and (np.abs(tri_normals[c_tris, 1]) < 0.20).mean() > 0.70:
            is_p = np.zeros(num_tris, bool)
            is_p[c_tris] = True
            b_edges = is_p[adj_u] & (~is_p[adj_v])
            nbrs = cleaned[adj_v[b_edges]]
            ext_nbrs = nbrs[~np.isin(nbrs, struct_arr) & (nbrs != cid) & (nbrs > 0)]
            if len(ext_nbrs) > 0:
                vals_p, cnts_p = np.unique(ext_nbrs, return_counts=True)
                host_cid = int(vals_p[cnts_p.argmax()])
                if np.count_nonzero(cleaned == host_cid) > len(c_tris):
                    cleaned[c_tris] = host_cid
                    reassigned += len(c_tris)

    return cleaned, reassigned


def suppress_micro_floor_instances(
    labels: np.ndarray,
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    floor_cid: int = 2,
    wall_cid: int = 5,
    min_floor_size: int = 1000,
) -> Tuple[np.ndarray, int]:
    """
    Eliminates thin slivers and tiny disconnected floor blobs.
    Any floor component with fewer than min_floor_size triangles is absorbed into
    the touching wall or main floor boundary.
    """
    cleaned = labels.copy()
    num_tris = len(cleaned)
    f_mask = (cleaned == floor_cid)
    f_indices = np.flatnonzero(f_mask)
    if len(f_indices) == 0:
        return cleaned, 0

    inv_map_f = np.full(num_tris, -1, dtype=np.int32)
    inv_map_f[f_indices] = np.arange(len(f_indices), dtype=np.int32)
    f_edges = f_mask[adj_u] & f_mask[adj_v]
    u_loc_f = inv_map_f[adj_u[f_edges]]
    v_loc_f = inv_map_f[adj_v[f_edges]]
    adj_f = sp.csr_matrix((np.ones(len(u_loc_f), bool), (u_loc_f, v_loc_f)), shape=(len(f_indices), len(f_indices)))
    n_comp_f, comp_ids_f = csgraph.connected_components(adj_f, directed=False)
    comp_sizes_f = np.bincount(comp_ids_f)
    reassigned = 0

    for c in range(n_comp_f):
        if comp_sizes_f[c] < min_floor_size:
            c_tris = f_indices[comp_ids_f == c]
            is_p = np.zeros(num_tris, bool)
            is_p[c_tris] = True
            b_edges = is_p[adj_u] & (~is_p[adj_v])
            nbrs = cleaned[adj_v[b_edges]]
            ext_nbrs = nbrs[nbrs != floor_cid]
            if len(ext_nbrs) > 0:
                vals_nbr, cnts_nbr = np.unique(ext_nbrs, return_counts=True)
                cleaned[c_tris] = int(vals_nbr[cnts_nbr.argmax()])
            else:
                cleaned[c_tris] = wall_cid
            reassigned += len(c_tris)

    return cleaned, reassigned


def suppress_micro_ceiling_instances(
    labels: np.ndarray,
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    ceiling_cid: int = 1,
    wall_cid: int = 5,
    min_ceiling_size: int = 1000,
) -> Tuple[np.ndarray, int]:
    """
    Eliminates thin slivers and tiny disconnected ceiling blobs.
    Any ceiling component with fewer than min_ceiling_size triangles is absorbed into
    the touching wall or main ceiling boundary.
    """
    cleaned = labels.copy()
    num_tris = len(cleaned)
    c_mask = (cleaned == ceiling_cid)
    c_indices = np.flatnonzero(c_mask)
    if len(c_indices) == 0:
        return cleaned, 0

    inv_map_c = np.full(num_tris, -1, dtype=np.int32)
    inv_map_c[c_indices] = np.arange(len(c_indices), dtype=np.int32)
    c_edges = c_mask[adj_u] & c_mask[adj_v]
    u_loc_c = inv_map_c[adj_u[c_edges]]
    v_loc_c = inv_map_c[adj_v[c_edges]]
    adj_c = sp.csr_matrix((np.ones(len(u_loc_c), bool), (u_loc_c, v_loc_c)), shape=(len(c_indices), len(c_indices)))
    n_comp_c, comp_ids_c = csgraph.connected_components(adj_c, directed=False)
    comp_sizes_c = np.bincount(comp_ids_c)
    reassigned = 0

    for c in range(n_comp_c):
        if comp_sizes_c[c] < min_ceiling_size:
            c_tris = c_indices[comp_ids_c == c]
            is_p = np.zeros(num_tris, bool)
            is_p[c_tris] = True
            b_edges = is_p[adj_u] & (~is_p[adj_v])
            nbrs = cleaned[adj_v[b_edges]]
            ext_nbrs = nbrs[nbrs != ceiling_cid]
            if len(ext_nbrs) > 0:
                vals_nbr, cnts_nbr = np.unique(ext_nbrs, return_counts=True)
                cleaned[c_tris] = int(vals_nbr[cnts_nbr.argmax()])
            else:
                cleaned[c_tris] = wall_cid
            reassigned += len(c_tris)

    return cleaned, reassigned


def detect_primary_structural_cids(
    vertices: np.ndarray,
    triangles: np.ndarray,
    labels: np.ndarray,
    tri_normals: Optional[np.ndarray] = None,
    tri_centroids: Optional[np.ndarray] = None,
    tri_areas: Optional[np.ndarray] = None,
) -> Tuple[int, int, int, Set[int]]:
    """
    Dynamically discovers primary Floor, Wall, and Ceiling class IDs based on pure 3D geometry:
    - Floor: Dominant upward-facing surface near the lower height band of the scene.
    - Wall: Dominant vertical surface spanning significant height.
    - Ceiling: Dominant downward-facing surface near the upper ceiling band.
    Returns: (floor_cid, wall_cid, ceiling_cid, all_structural_cids)
    """
    if tri_normals is None or tri_centroids is None or tri_areas is None:
        tri_areas, tri_normals, tri_centroids, _ = compute_mesh_derivatives(
            vertices, triangles, np.zeros_like(vertices)
        )
    extent = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    gravity_up, _ = estimate_gravity_and_scene_bands(vertices / max(extent, 1e-12), triangles)
    dot_up = tri_normals @ gravity_up
    h = tri_centroids @ gravity_up

    classes, first = np.unique(labels, return_index=True)
    classes = classes[np.argsort(first)]
    classes = classes[classes > 0]
    if len(classes) == 0:
        return 0, 0, 0, set()
    h_min, h_max = np.percentile(h, [1, 99])
    height = h_max - h_min
    total_area = max(float(tri_areas.sum()), np.finfo(float).tiny)

    floor_scores = {
        cid: float(np.sum(tri_areas[(labels == cid) & (dot_up > 0.70) & (h <= h_min + 0.15 * height)]))
        for cid in classes
    }
    wall_scores = {
        cid: float(np.sum(tri_areas[(labels == cid) & (np.abs(dot_up) < 0.30)]))
        for cid in classes
    }
    ceil_scores = {
        cid: float(np.sum(tri_areas[(labels == cid) & (dot_up < -0.60) & (h >= h_min + 0.65 * height)]))
        for cid in classes
    }

    best_floor = max(floor_scores, key=floor_scores.get)
    best_wall = max(wall_scores, key=wall_scores.get)
    best_ceil = max(ceil_scores, key=ceil_scores.get)
    best_floor = best_floor if floor_scores[best_floor] > 0.0075 * total_area else 0
    best_wall = best_wall if wall_scores[best_wall] > 0.05 * total_area else 0
    best_ceil = best_ceil if ceil_scores[best_ceil] > 0.001 * total_area else 0

    structural_cids = {cid for cid in (best_floor, best_wall, best_ceil) if cid > 0}
    for cid, s in ceil_scores.items():
        if s > 0.001 * total_area:
            structural_cids.add(cid)
    for cid, s in wall_scores.items():
        if s > 0.05 * total_area:
            structural_cids.add(cid)
    for cid, s in floor_scores.items():
        if s > 0.0075 * total_area:
            structural_cids.add(cid)

    return best_floor, best_wall, best_ceil, structural_cids


def eliminate_horizontal_wall_triangles(
    labels: np.ndarray,
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    tri_normals: np.ndarray,
    tri_centroids: np.ndarray,
    wall_cid: int = 5,
    floor_cid: int = 2,
    ceiling_cid: int = 1,
    gravity_up: np.ndarray = np.array([0.0, 1.0, 0.0]),
    y_ground: float = 1.35,
    max_tilt_cos: float = 0.45,
) -> Tuple[np.ndarray, int]:
    """
    Guarantees strictly 0 horizontal triangles on walls (|n . gravity_up| > max_tilt_cos).
    1. Upward-facing triangles at floor level (<= y_ground + 0.30m) are reassigned to Floor.
    2. Downward-facing triangles at ceiling level (>= y_ground + 1.80m) are reassigned to Ceiling.
    3. Intermediate horizontal triangles (window sills, ledge tops, shelf surfaces) are reassigned
       to touching non-wall neighbors via the mesh dual graph, or to Floor/Ceiling.
    """
    cleaned = labels.copy()
    num_tris = len(cleaned)
    tilt = tri_normals @ gravity_up
    horiz_wall = (cleaned == wall_cid) & (np.abs(tilt) > max_tilt_cos)
    h_indices = np.flatnonzero(horiz_wall)
    if len(h_indices) == 0:
        return cleaned, 0

    upward = tilt[h_indices] > max_tilt_cos
    downward = tilt[h_indices] < -max_tilt_cos
    near_floor = tri_centroids[h_indices, 1] <= (y_ground + 0.30)
    near_ceiling = tri_centroids[h_indices, 1] >= (y_ground + 1.80)

    cleaned[h_indices[upward & near_floor]] = floor_cid
    cleaned[h_indices[downward & near_ceiling]] = ceiling_cid

    rem = (cleaned == wall_cid) & (np.abs(tilt) > max_tilt_cos)
    rem_indices = np.flatnonzero(rem)
    if len(rem_indices) > 0:
        inv_map = np.full(num_tris, -1, dtype=np.int32)
        inv_map[rem_indices] = np.arange(len(rem_indices), dtype=np.int32)
        m = np.zeros(num_tris, bool)
        m[rem_indices] = True
        int_e = m[adj_u] & m[adj_v]
        u_l, v_l = inv_map[adj_u[int_e]], inv_map[adj_v[int_e]]
        sub_adj = sp.csr_matrix((np.ones(len(u_l), bool), (u_l, v_l)), shape=(len(rem_indices), len(rem_indices)))
        n_c, c_lbls = csgraph.connected_components(sub_adj, directed=False)

        tri_c = np.full(num_tris, -1, dtype=np.int32)
        tri_c[rem_indices] = c_lbls
        b_e = (tri_c[adj_u] >= 0) & (cleaned[adj_v] != wall_cid)
        if np.any(b_e):
            cu = tri_c[adj_u[b_e]]
            nv = cleaned[adj_v[b_e]]
            sort_k = np.lexsort((nv, cu))
            cu_s, nv_s = cu[sort_k], nv[sort_k]
            split = np.concatenate([[True], (cu_s[:-1] != cu_s[1:]) | (nv_s[:-1] != nv_s[1:])])
            p_c = cu_s[split]
            p_nv = nv_s[split]
            lens = np.diff(np.concatenate([np.flatnonzero(split), [len(cu_s)]]))
            best_nbr = np.full(n_c, -1, dtype=np.int32)
            best_cnt = np.zeros(n_c, dtype=np.int32)
            for c_id, nbr_id, cnt in zip(p_c, p_nv, lens):
                if cnt > best_cnt[c_id]:
                    best_cnt[c_id] = cnt
                    best_nbr[c_id] = nbr_id
            for c in range(n_c):
                c_tris = rem_indices[c_lbls == c]
                if best_nbr[c] > 0:
                    cleaned[c_tris] = best_nbr[c]
                else:
                    if np.mean(tilt[c_tris]) > 0:
                        cleaned[c_tris] = floor_cid
                    else:
                        cleaned[c_tris] = ceiling_cid
        else:
            for c in range(n_c):
                c_tris = rem_indices[c_lbls == c]
                if np.mean(tilt[c_tris]) > 0:
                    cleaned[c_tris] = floor_cid
                else:
                    cleaned[c_tris] = ceiling_cid

    left = np.sum((cleaned == wall_cid) & (np.abs(tri_normals @ gravity_up) > max_tilt_cos))
    return cleaned, len(h_indices) - left


def absorb_flat_coplanar_wall_patches(
    labels: np.ndarray,
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    tri_normals: np.ndarray,
    tri_centroids: np.ndarray,
    wall_cid: int = 5,
    max_thickness: float = 0.08,
    min_wall_border: float = 0.35,
    protected_cids: Optional[Set[int]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Absorbs flat non-structural patches directly adhering to the drywall plane into Wall.
    Prevents over-segmentation of flat vertical walls (e.g. Class 100, Class 292, Class 299).
    """
    if protected_cids is None:
        protected_cids = {14}  # Protected artwork / poster
    cleaned = labels.copy()
    num_tris = len(cleaned)
    absorbed = 0
    unique_cids = np.unique(cleaned)

    for cid in unique_cids:
        if cid == wall_cid or cid in protected_cids or cid <= 0:
            continue
        c_tris = np.flatnonzero(cleaned == cid)
        if len(c_tris) == 0 or len(c_tris) > 30000:
            continue

        pts = tri_centroids[c_tris]
        spans = pts.max(axis=0) - pts.min(axis=0)
        thick = min(spans[0], spans[2])
        is_vertical = (np.abs(tri_normals[c_tris, 1]) < 0.35).mean() > 0.70

        if thick <= max_thickness and is_vertical:
            is_c = np.zeros(num_tris, bool)
            is_c[c_tris] = True
            b_edges = is_c[adj_u] & (~is_c[adj_v])
            nbrs = cleaned[adj_v[b_edges]]
            valid_nbrs = nbrs[nbrs > 0]
            if len(valid_nbrs) > 0:
                wall_border_ratio = np.mean(valid_nbrs == wall_cid)
                if wall_border_ratio >= min_wall_border:
                    cleaned[c_tris] = wall_cid
                    absorbed += len(c_tris)

    return cleaned, absorbed


def enforce_instance_constraints(
    labels: np.ndarray,
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    normals: np.ndarray,
    gravity_up: np.ndarray,
    wall_cids: Set[int],
    min_triangles: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Merge edge-connected fragments to a hard size floor without horizontal walls.

    Regions without an eligible touching owner become unknown; geometry is retained.
    Wall eligibility includes both upward and downward faces within 20° of horizontal.
    """
    import heapq

    if min_triangles < 1:
        raise ValueError('min_triangles must be positive')
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in 'iu' or np.shape(normals) != (len(labels), 3):
        raise ValueError('expected integer face labels and one normal per face')
    up = np.array(gravity_up, dtype=float, copy=True)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) == 0:
        raise ValueError('gravity_up must be a finite nonzero 3-vector')
    up /= np.linalg.norm(up)
    normal_lengths = np.linalg.norm(normals, axis=1)
    alignment = np.abs(normals @ up) / np.maximum(normal_lengths, 1e-12)
    horizontal = alignment >= np.cos(np.deg2rad(20.0)) - 1e-12
    wall_ids = {int(cid) for cid in wall_cids if cid > 0}
    cleaned = labels.astype(np.int32, copy=True)
    invalid_wall = np.isin(cleaned, list(wall_ids)) & horizontal
    # First reassign each forbidden wall patch to its strongest touching non-wall owner.
    bad_faces = np.flatnonzero(invalid_wall)
    if len(bad_faces):
        bad_map = np.full(len(labels), -1, dtype=np.int32)
        bad_map[bad_faces] = np.arange(len(bad_faces))
        internal = invalid_wall[adj_u] & invalid_wall[adj_v]
        graph = sp.csr_matrix((np.ones(internal.sum(), bool),
            (bad_map[adj_u[internal]], bad_map[adj_v[internal]])), shape=(len(bad_faces), len(bad_faces)))
        count, patch_ids = csgraph.connected_components(graph, directed=False)
        a, b = np.r_[adj_u, adj_v], np.r_[adj_v, adj_u]
        boundary = invalid_wall[a] & ~invalid_wall[b] & (cleaned[b] > 0) & ~np.isin(cleaned[b], list(wall_ids))
        patch_owner = np.full(count, -1, dtype=np.int32)
        if boundary.any():
            pairs = np.column_stack((patch_ids[bad_map[a[boundary]]], cleaned[b[boundary]]))
            pairs, weights = np.unique(pairs, axis=0, return_counts=True)
            # Greatest shared boundary first; ties follow the first source face, not ID magnitude.
            first = {int(cid): int(np.flatnonzero(cleaned == cid)[0]) for cid in np.unique(pairs[:, 1])}
            order = sorted(range(len(pairs)), key=lambda i: (-int(weights[i]), first[int(pairs[i, 1])]))
            for i in order:
                patch, cid = pairs[i]
                if patch_owner[patch] < 0:
                    patch_owner[patch] = cid
        cleaned[bad_faces] = patch_owner[patch_ids]

    faces = np.flatnonzero(cleaned > 0)
    face_map = np.full(len(labels), -1, dtype=np.int32)
    face_map[faces] = np.arange(len(faces))
    valid_edges = (cleaned[adj_u] > 0) & (cleaned[adj_v] > 0)
    same = valid_edges & (cleaned[adj_u] == cleaned[adj_v])
    graph = sp.csr_matrix((np.ones(same.sum(), bool),
        (face_map[adj_u[same]], face_map[adj_v[same]])), shape=(len(faces), len(faces)))
    count, component = csgraph.connected_components(graph, directed=False)
    sizes = np.bincount(component, minlength=count)
    owner = np.zeros(count, dtype=np.int32)
    owner[component] = cleaned[faces]
    has_horizontal = np.bincount(component, weights=horizontal[faces], minlength=count) > 0
    parent = np.arange(count)
    neighbors = [dict() for _ in range(count)]
    cross = valid_edges & ~same
    left, right = component[face_map[adj_u[cross]]], component[face_map[adj_v[cross]]]
    low, high = np.minimum(left, right), np.maximum(left, right)
    if len(low):
        pairs, inverse = np.unique(np.column_stack((low, high)), axis=0, return_inverse=True)
        # Favor coherent contacts while keeping every actual mesh contact eligible.
        dot = np.einsum('ij,ij->i', normals[adj_u[cross]], normals[adj_v[cross]])
        dot /= np.maximum(normal_lengths[adj_u[cross]] * normal_lengths[adj_v[cross]], 1e-12)
        weights = np.bincount(inverse, weights=0.1 + 0.9 * np.clip(dot, 0, 1) ** 2)
        for (a, b), weight in zip(pairs, weights):
            neighbors[a][int(b)] = neighbors[b][int(a)] = float(weight)
    heap = [(int(sizes[c]), c) for c in range(count) if sizes[c] < min_triangles]
    heapq.heapify(heap)
    merges = abandoned_faces = 0

    def join(source, target):
        parent[source] = target
        sizes[target] += sizes[source]
        has_horizontal[target] |= has_horizontal[source]
        neighbors[target].pop(source, None)
        for neighbor, weight in list(neighbors[source].items()):
            neighbors[neighbor].pop(source, None)
            if neighbor != target:
                neighbors[target][neighbor] = neighbors[target].get(neighbor, 0.0) + weight
                neighbors[neighbor][target] = neighbors[target][neighbor]
        neighbors[source].clear()

    while heap:
        size, source = heapq.heappop(heap)
        if parent[source] != source or owner[source] <= 0 or size != sizes[source] or size >= min_triangles:
            continue
        eligible = [target for target in neighbors[source]
                    if owner[target] > 0 and not (owner[target] in wall_ids
                        and (has_horizontal[source] or has_horizontal[target]))]
        if not eligible:
            abandoned_faces += int(sizes[source])
            owner[source] = -1
            for neighbor in list(neighbors[source]):
                neighbors[neighbor].pop(source, None)
            neighbors[source].clear()
            continue
        target = max(eligible, key=lambda c: (neighbors[source][c], int(sizes[c]), -c))
        join(source, target)
        merges += 1
        # Relabelling can connect multiple regions already bearing the target label.
        same_owner = [c for c in neighbors[target] if owner[c] == owner[target]]
        while same_owner:
            join(same_owner.pop(), target)
            same_owner = [c for c in neighbors[target] if owner[c] == owner[target]]
        if sizes[target] < min_triangles:
            heapq.heappush(heap, (int(sizes[target]), target))
    if count:
        while np.any(parent != parent[parent]):
            parent = parent[parent]
        cleaned[faces] = owner[parent[component]]
    surviving = (parent == np.arange(count)) & (owner > 0)
    if np.any(sizes[surviving] < min_triangles):
        raise RuntimeError('undersized instance survived hard minimum cleanup')
    if np.any(np.isin(cleaned, list(wall_ids)) & horizontal):
        raise RuntimeError('horizontal wall face survived orientation cleanup')
    return cleaned, dict(min_triangles=int(min_triangles), wall_horizontal_tolerance_deg=20,
        forbidden_wall_faces=int(invalid_wall.sum()), fragment_merges=merges,
        unmergeable_fragment_faces=abandoned_faces,
        minimum_instance_triangles=int(sizes[surviving].min()) if surviving.any() else None,
        instances=int(surviving.sum()), changed_faces=int(np.count_nonzero(cleaned != labels)))


def run_complete_geometric_regularization(
    vertices: np.ndarray,
    triangles: np.ndarray,
    vertex_colors: np.ndarray,
    raw_labels: np.ndarray,
    mesh: o3d.geometry.TriangleMesh,
    floor_cid: int | None = None,
    wall_cid: int | None = None,
    ceiling_cid: int | None = None,
    min_island_size: int = 150,
    min_instance_size: int = 350,
    min_instance_area: float = 0.03,
    min_instance_span: float = 0.15,
    priors: sp.csr_matrix | None = None,
    source_ids: np.ndarray | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Refine observed labels from multi-view evidence, then reconcile compact parts.

    Surface normals control boundary affinity, never an object's semantic role.
    Unobserved faces stay unknown; size alone never erases a supported object.
    Legacy size arguments remain accepted for the standalone callers.
    """
    from mesh_prior_refinement import refine_graph_labels

    t0 = time.perf_counter()
    raw_labels = np.asarray(raw_labels)
    if raw_labels.shape != (len(triangles),) or raw_labels.dtype.kind not in 'iu':
        raise ValueError('raw labels must contain one integer per source mesh face')
    labels = raw_labels.copy()
    labels[labels <= 0] = -1
    colors = np.asarray(vertex_colors, dtype=np.float64)
    if colors.size and colors.max() > 1:
        colors = colors / 255.0
    areas, normals, centers, rgb = compute_mesh_derivatives(vertices, triangles, colors)
    u, v = build_mesh_dual_graph(triangles)
    unique = u < v
    u, v = u[unique], v[unique]
    lengths = np.empty(len(u), dtype=np.float64)
    for start in range(0, len(u), 262144):
        a, b = u[start:start+262144], v[start:start+262144]
        shared = (triangles[a, :, None] == triangles[b, None, :]).any(axis=2)
        if np.any(shared.sum(axis=1) != 2):
            raise ValueError('adjacent mesh faces must share exactly one edge')
        endpoints = vertices[triangles[a][shared].reshape(-1, 2)]
        lengths[start:start+262144] = np.linalg.norm(endpoints[:, 1] - endpoints[:, 0], axis=1)

    if priors is None:
        if source_ids is not None:
            raise ValueError('source IDs require a corresponding face evidence matrix')
        source_ids = np.unique(labels[labels > 0])
        observed = np.flatnonzero(labels > 0)
        priors = sp.csr_matrix((np.ones(len(observed)),
            (observed, np.searchsorted(source_ids, labels[observed]))),
            shape=(len(labels), len(source_ids)))
    elif source_ids is None:
        raise ValueError('face evidence requires its source IDs')
    source_ids = np.asarray(source_ids)
    if priors.shape != (len(labels), len(source_ids)):
        raise ValueError('cached evidence does not match mesh faces and source IDs')
    # Tessellation-relative smoothing works in the scan's existing world units.
    boundary_scale = float(np.median(lengths[lengths > 0])) if np.any(lengths > 0) else 1e-6
    labels, smoothing = refine_graph_labels(labels, priors, source_ids, areas,
        normals, rgb, u, v, lengths, boundary_scale)
    if not smoothing['converged']:
        raise RuntimeError('face-label refinement did not converge within its iteration limit')

    det_floor, det_wall, det_ceil, structural = detect_primary_structural_cids(
        vertices, triangles, labels, normals, centers, areas)
    extent = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    gravity_up, _ = estimate_gravity_and_scene_bands(vertices / max(extent, 1e-12), triangles)
    heights = centers @ gravity_up
    height_range = float(np.percentile(heights, 99) - np.percentile(heights, 1))
    compact_span = 0.35 * (height_range if height_range > 0 else extent)
    floor_cid = det_floor if floor_cid is None else floor_cid
    wall_cid = det_wall if wall_cid is None else wall_cid
    ceiling_cid = det_ceil if ceiling_cid is None else ceiling_cid
    structural |= {floor_cid, wall_cid, ceiling_cid}
    # Include both edge directions for component-boundary queries.
    labels, merged = regularize_foreground_connected_components(
        centers, labels, np.r_[u, v], np.r_[v, u], structural,
        compact_span_thresh=compact_span,
        tri_normals=normals, tri_areas=areas, gravity_up=gravity_up,
        edge_lengths=np.r_[lengths, lengths])
    metrics = dict(floor_cid=int(floor_cid), wall_cid=int(wall_cid),
        ceiling_cid=int(ceiling_cid), smoothing=smoothing,
        foreground_parts_reassigned=int(merged),
        unknown_faces=int(np.count_nonzero(labels <= 0)),
        changed_observed_faces=int(np.count_nonzero((labels != raw_labels) & (raw_labels > 0))),
        surviving_classes=int(len(np.unique(labels[labels > 0]))),
        total_regularization_seconds=round(time.perf_counter()-t0, 2))
    return labels, metrics
