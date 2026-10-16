#!/usr/bin/env python3
"""
Step 3: Geometry-Driven Structural Class Detection
Detects and classifies structural architectural classes (walls, ceilings, floors, and stairs)
using pure 3D geometry: extent, surface normal distributions, RANSAC plane fitting,
elevation profiles, piece continuity, and spatial contiguity, without computer vision.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import open3d as o3d
from scipy.sparse import csgraph, csr_matrix


def get_class_color(mid: int) -> Tuple[Tuple[int, int, int], Tuple[float, float, float]]:
    """Deterministic RGB color matching the 2D panoptic segmentation palette."""
    if mid <= 0:
        return (128, 128, 128), (0.5, 0.5, 0.5)
    k = mid - 1
    b = (37 * k + 93) % 255
    g = (109 * k + 71) % 255
    r = (197 * k + 31) % 255
    col_bgr = (int(b), int(g), int(r))
    col_rgb = (r / 255.0, g / 255.0, b / 255.0)
    return col_bgr, col_rgb


def export_colored_ply(
    file_path: Path,
    vertices: np.ndarray,
    triangles: np.ndarray,
    triangle_colors_rgb: np.ndarray | None = None,
    vertex_colors_rgb: np.ndarray | None = None,
) -> None:
    """Exports binary little-endian PLY mesh with vertex colors."""
    num_verts = len(vertices)
    num_faces = len(triangles)

    if vertex_colors_rgb is not None:
        v_colors = vertex_colors_rgb.astype(np.uint8)
    else:
        v_colors = np.zeros((num_verts, 3), dtype=np.float64)
        v_counts = np.zeros((num_verts, 1), dtype=np.float64)

        if triangle_colors_rgb is not None:
            for i in range(3):
                np.add.at(v_colors, triangles[:, i], triangle_colors_rgb)
                np.add.at(v_counts, triangles[:, i], 1.0)

        v_counts = np.maximum(1.0, v_counts)
        v_colors = (v_colors / v_counts).round().astype(np.uint8)

    v_data = np.zeros(
        num_verts,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    v_data["x"] = vertices[:, 0]
    v_data["y"] = vertices[:, 1]
    v_data["z"] = vertices[:, 2]
    v_data["red"] = v_colors[:, 0]
    v_data["green"] = v_colors[:, 1]
    v_data["blue"] = v_colors[:, 2]

    f_data = np.zeros(
        num_faces,
        dtype=[
            ("n_verts", "u1"),
            ("v0", "<i4"),
            ("v1", "<i4"),
            ("v2", "<i4"),
        ],
    )
    f_data["n_verts"] = 3
    f_data["v0"] = triangles[:, 0]
    f_data["v1"] = triangles[:, 1]
    f_data["v2"] = triangles[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {num_verts}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        f"element face {num_faces}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(v_data.tobytes())
        f.write(f_data.tobytes())
    print(f"Exported cleaned PLY mesh: {file_path}")


# Vector stroke font on unit square [0, 1] x [0, 1] for 3D badges
STROKES: Dict[str, List[Tuple[Tuple[float, float], Tuple[float, float]]]] = {
    "#": [
        ((0.28, 0.05), (0.28, 0.95)),
        ((0.72, 0.05), (0.72, 0.95)),
        ((0.08, 0.35), (0.92, 0.35)),
        ((0.08, 0.65), (0.92, 0.65)),
    ],
    "0": [
        ((0.15, 0.05), (0.85, 0.05)),
        ((0.85, 0.05), (0.85, 0.95)),
        ((0.85, 0.95), (0.15, 0.95)),
        ((0.15, 0.95), (0.15, 0.05)),
        ((0.15, 0.05), (0.85, 0.95)),
    ],
    "1": [
        ((0.50, 0.05), (0.50, 0.95)),
        ((0.20, 0.75), (0.50, 0.95)),
        ((0.20, 0.05), (0.80, 0.05)),
    ],
    "2": [
        ((0.15, 0.95), (0.85, 0.95)),
        ((0.85, 0.95), (0.85, 0.55)),
        ((0.85, 0.55), (0.15, 0.55)),
        ((0.15, 0.55), (0.15, 0.05)),
        ((0.15, 0.05), (0.85, 0.05)),
    ],
    "3": [
        ((0.15, 0.95), (0.85, 0.95)),
        ((0.85, 0.95), (0.85, 0.05)),
        ((0.85, 0.05), (0.15, 0.05)),
        ((0.15, 0.55), (0.85, 0.55)),
    ],
    "4": [
        ((0.15, 0.95), (0.15, 0.50)),
        ((0.15, 0.50), (0.85, 0.50)),
        ((0.80, 0.95), (0.80, 0.05)),
    ],
    "5": [
        ((0.85, 0.95), (0.15, 0.95)),
        ((0.15, 0.95), (0.15, 0.55)),
        ((0.15, 0.55), (0.85, 0.55)),
        ((0.85, 0.55), (0.85, 0.05)),
        ((0.85, 0.05), (0.15, 0.05)),
    ],
    "6": [
        ((0.85, 0.95), (0.15, 0.95)),
        ((0.15, 0.95), (0.15, 0.05)),
        ((0.15, 0.05), (0.85, 0.05)),
        ((0.85, 0.05), (0.85, 0.55)),
        ((0.85, 0.55), (0.15, 0.55)),
    ],
    "7": [
        ((0.15, 0.95), (0.85, 0.95)),
        ((0.85, 0.95), (0.35, 0.05)),
    ],
    "8": [
        ((0.15, 0.05), (0.85, 0.05)),
        ((0.85, 0.05), (0.85, 0.95)),
        ((0.85, 0.95), (0.15, 0.95)),
        ((0.15, 0.95), (0.15, 0.05)),
        ((0.15, 0.50), (0.85, 0.50)),
    ],
    "9": [
        ((0.15, 0.05), (0.85, 0.05)),
        ((0.85, 0.05), (0.85, 0.95)),
        ((0.85, 0.95), (0.15, 0.95)),
        ((0.15, 0.95), (0.15, 0.50)),
        ((0.15, 0.50), (0.85, 0.50)),
    ],
    ":": [
        ((0.50, 0.25), (0.50, 0.35)),
        ((0.50, 0.65), (0.50, 0.75)),
    ],
    ".": [
        ((0.40, 0.05), (0.60, 0.05)),
    ],
    "-": [
        ((0.20, 0.50), (0.80, 0.50)),
    ],
    "C": [
        ((0.85, 0.95), (0.15, 0.95)),
        ((0.15, 0.95), (0.15, 0.05)),
        ((0.15, 0.05), (0.85, 0.05)),
    ],
    "I": [
        ((0.20, 0.95), (0.80, 0.95)),
        ((0.50, 0.95), (0.50, 0.05)),
        ((0.20, 0.05), (0.80, 0.05)),
    ],
    "D": [
        ((0.15, 0.05), (0.15, 0.95)),
        ((0.15, 0.95), (0.65, 0.95)),
        ((0.65, 0.95), (0.85, 0.50)),
        ((0.85, 0.50), (0.65, 0.05)),
        ((0.65, 0.05), (0.15, 0.05)),
    ],
}


def create_cylinder_mesh(
    p1: np.ndarray,
    p2: np.ndarray,
    radius: float = 0.003,
    color_rgb: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Creates a 4-sided cylinder (tube) between p1 and p2 with given color."""
    vec = p2 - p1
    length = float(np.linalg.norm(vec))
    if length < 1e-5:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 3), dtype=np.uint8),
        )

    dir_v = vec / length
    up = np.array([0.0, 1.0, 0.0]) if abs(dir_v[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    side1 = np.cross(dir_v, up)
    side1 /= np.linalg.norm(side1)
    side2 = np.cross(dir_v, side1)

    r = radius
    offsets = [r * side1, r * side2, -r * side1, -r * side2]
    v_bottom = [p1 + off for off in offsets]
    v_top = [p2 + off for off in offsets]
    verts = np.array(v_bottom + v_top, dtype=np.float32)

    faces = []
    for i in range(4):
        i_next = (i + 1) % 4
        faces.append([i, i_next, i_next + 4])
        faces.append([i, i_next + 4, i + 4])
    faces = np.array(faces, dtype=np.int32)
    colors = np.tile(np.array(color_rgb, dtype=np.uint8), (len(verts), 1))
    return verts, faces, colors


def build_instance_badge_mesh(
    origin: np.ndarray,
    forward: np.ndarray,
    up: np.ndarray | None = None,
    scale: float = 0.08,
    text: str = "#1",
    border_color_rgb: Tuple[int, int, int] = (255, 255, 255),
    text_color_rgb: Tuple[int, int, int] = (255, 255, 255),
    plate_color_rgb: Tuple[int, int, int] = (20, 20, 25),
    include_plate: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds a 3D badge consisting of a dark backing plate, class-colored rim,
    and 3D white stroke text #<ID>.
    """
    f_len = np.linalg.norm(forward)
    f = forward / f_len if f_len > 1e-6 else np.array([0.0, 0.0, 1.0])

    u_ref = up if up is not None else np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(f, u_ref))) > 0.92:
        u_ref = np.array([0.0, 0.0, -1.0]) if abs(f[1]) > 0.92 else np.array([0.0, 1.0, 0.0])

    r = np.cross(u_ref, f)
    r_len = np.linalg.norm(r)
    if r_len < 1e-6:
        r = np.array([1.0, 0.0, 0.0])
    else:
        r /= r_len
    u = np.cross(f, r)
    u /= np.linalg.norm(u)

    n_chars = len(text)
    char_w = 0.65 * scale
    char_h = 1.00 * scale
    char_adv = 0.80 * scale
    text_w = max(0.0, (n_chars - 1) * char_adv) + char_w
    text_h = char_h

    pad_x = 0.22 * scale
    pad_y = 0.18 * scale
    plate_w = text_w + 2 * pad_x
    plate_h = text_h + 2 * pad_y
    plate_t = 0.005

    all_v: List[np.ndarray] = []
    all_f: List[np.ndarray] = []
    all_c: List[np.ndarray] = []
    v_off = 0

    hw, hh, ht = plate_w / 2.0, plate_h / 2.0, plate_t / 2.0

    if include_plate:
        corners_local = np.array([
            [-hw, -hh, -ht], [hw, -hh, -ht], [hw, hh, -ht], [-hw, hh, -ht],
            [-hw, -hh,  ht], [hw, -hh,  ht], [hw, hh,  ht], [-hw, hh,  ht],
        ])
        corners_world = origin + corners_local[:, 0:1] * r + corners_local[:, 1:2] * u + corners_local[:, 2:3] * f
        box_faces = [
            [0, 2, 1], [0, 3, 2],  # back
            [4, 5, 6], [4, 6, 7],  # front
            [0, 1, 5], [0, 5, 4],  # bottom
            [2, 3, 7], [2, 7, 6],  # top
            [0, 4, 7], [0, 7, 3],  # left
            [1, 2, 6], [1, 6, 5],  # right
        ]
        box_colors = np.zeros((8, 3), dtype=np.uint8)
        box_colors[:4] = plate_color_rgb
        box_colors[4:] = plate_color_rgb
        all_v.append(corners_world)
        all_f.append(np.array(box_faces, dtype=np.int32) + v_off)
        all_c.append(box_colors)
        v_off += 8

        # Border rim around the front and back faces
        rim_rad = max(0.0018, 0.025 * scale)
        rim_front_pts = [
            corners_world[4], corners_world[5], corners_world[6], corners_world[7]
        ]
        rim_back_pts = [
            corners_world[0], corners_world[1], corners_world[2], corners_world[3]
        ]
        for pts in [rim_front_pts, rim_back_pts]:
            for i in range(4):
                p_a, p_b = pts[i], pts[(i + 1) % 4]
                vs, fs, cs = create_cylinder_mesh(p_a, p_b, radius=rim_rad, color_rgb=border_color_rgb)
                if len(vs) > 0:
                    all_v.append(vs)
                    all_f.append(fs + v_off)
                    all_c.append(cs)
                    v_off += len(vs)

    # Text strokes (front and back for 360-degree readability)
    start_x = -text_w / 2.0
    start_y = -text_h / 2.0
    z_stroke = (ht + 0.002) if include_plate else 0.002
    stroke_rad = max(0.002, 0.032 * scale)

    for i, ch in enumerate(text):
        c_x = start_x + i * char_adv
        strokes = STROKES.get(ch, STROKES.get(ch.upper(), []))
        for (x0, y0), (x1, y1) in strokes:
            p0_loc = np.array([c_x + x0 * char_w, start_y + y0 * char_h, z_stroke])
            p1_loc = np.array([c_x + x1 * char_w, start_y + y1 * char_h, z_stroke])
            p0 = origin + p0_loc[0] * r + p0_loc[1] * u + p0_loc[2] * f
            p1 = origin + p1_loc[0] * r + p1_loc[1] * u + p1_loc[2] * f
            vs, fs, cs = create_cylinder_mesh(p0, p1, radius=stroke_rad, color_rgb=text_color_rgb)
            if len(vs) > 0:
                all_v.append(vs)
                all_f.append(fs + v_off)
                all_c.append(cs)
                v_off += len(vs)

            if include_plate:
                # Mirror x and z so back-face text reads left-to-right when viewed from behind
                p0_b_loc = np.array([-p0_loc[0], p0_loc[1], -p0_loc[2]])
                p1_b_loc = np.array([-p1_loc[0], p1_loc[1], -p1_loc[2]])
                p0_b = origin + p0_b_loc[0] * r + p0_b_loc[1] * u + p0_b_loc[2] * f
                p1_b = origin + p1_b_loc[0] * r + p1_b_loc[1] * u + p1_b_loc[2] * f
                vs_b, fs_b, cs_b = create_cylinder_mesh(p0_b, p1_b, radius=stroke_rad, color_rgb=text_color_rgb)
                if len(vs_b) > 0:
                    all_v.append(vs_b)
                    all_f.append(fs_b + v_off)
                    all_c.append(cs_b)
                    v_off += len(vs_b)

    if len(all_v) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 3), dtype=np.uint8),
        )
    return np.vstack(all_v), np.vstack(all_f), np.vstack(all_c)


def read_scene_sizes(scene_size_path: Path | None) -> Dict[str, Any]:
    """Reads floors and room dimensions from scene_size.txt if available."""
    info = {"floors": 1, "total_floor_area_m2": None}
    if scene_size_path is None or not scene_size_path.is_file():
        return info
    try:
        with open(scene_size_path) as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                k, v = line.split(":", 1)
                k = k.strip().lower()
                v = v.strip()
                if k == "floors":
                    info["floors"] = int(v)
                elif k == "total_floor_area_m2":
                    info["total_floor_area_m2"] = float(v)
    except Exception as e:
        print(f"Warning: could not parse scene_size.txt: {e}")
    return info


def load_canonical_dino_embeddings(
    emb_path: Path | None,
    active_classes: List[int],
) -> Dict[int, np.ndarray] | None:
    """Loads canonical DINOv2 384-d unit embeddings for active classes."""
    if emb_path is None or not emb_path.is_file():
        return None
    try:
        data = np.load(emb_path)
        class_ids = data["class_ids"].tolist()
        embs = data["embeddings"]
        res = {}
        for cid in active_classes:
            if cid in class_ids:
                idx = class_ids.index(cid)
                res[cid] = embs[idx].astype(np.float32)
        return res
    except Exception as e:
        print(f"Warning: could not load canonical embeddings from {emb_path}: {e}")
        return None


def estimate_gravity_and_scene_bands(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Estimates the gravity up-vector and scene elevation bounds from mesh geometry.
    Uses Extended Gaussian Image (EGI) normal covariance to find the dominant normal axis.
    """
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid = areas > 1e-8
    areas = areas[valid]
    normals = cross[valid] / (2.0 * areas[:, None])
    centroids = (v0[valid] + v1[valid] + v2[valid]) / 3.0

    # Normal covariance: C = sum_i A_i n_i n_i^T
    # Horizontal surfaces (floors/ceilings) and vertical walls both align with the gravity frame
    weighted_cov = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            weighted_cov[i, j] = np.sum(areas * normals[:, i] * normals[:, j])

    eigvals, eigvecs = np.linalg.eigh(weighted_cov)
    up_candidate = eigvecs[:, 2]

    # Snap to dominant coordinate axis
    abs_up = np.abs(up_candidate)
    max_axis = int(np.argmax(abs_up))
    aligned_up = np.zeros(3, dtype=np.float32)
    aligned_up[max_axis] = 1.0  # Positive along vertical axis (upwards)

    # Project centroids along up-vector
    h_proj = np.dot(centroids, aligned_up)

    h_min = float(np.percentile(h_proj, 1.0))
    h_max = float(np.percentile(h_proj, 99.0))
    h_scene = max(0.5, h_max - h_min)
    h_ceiling_band = h_min + 0.65 * h_scene

    scene_bands = {
        "h_min": h_min,
        "h_max": h_max,
        "h_scene": h_scene,
        "h_ceiling_band": h_ceiling_band,
        "up_axis_index": max_axis,
    }

    return aligned_up, scene_bands


def build_mesh_dual_graph(
    triangles: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs the undirected dual graph connecting adjacent triangles sharing a 3D vertex-vertex edge.
    """
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
    return adj_u, adj_v


def absorb_structural_floaters(
    triangle_labels: np.ndarray,
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    tri_areas: np.ndarray,
    structural_classes: List[int] = [5, 14, 1],
    max_island_tris: int = 5000,
    max_island_area: float = 0.25,
) -> Tuple[np.ndarray, int]:
    """
    Absorbs disconnected structural class islands (e.g. wall/floor patches on furniture)
    that are below max_island_area and max_island_tris into the dominant surrounding neighbor class.
    """
    cleaned = triangle_labels.copy()
    num_tris = len(triangle_labels)
    total_absorbed = 0

    for cid in structural_classes:
        c_indices = np.flatnonzero(cleaned == cid)
        if len(c_indices) == 0:
            continue

        is_c = np.zeros(num_tris, dtype=bool)
        is_c[c_indices] = True
        c_edges = is_c[adj_u] & is_c[adj_v]
        u_c = adj_u[c_edges]
        v_c = adj_v[c_edges]

        if len(u_c) > 0:
            u_loc = np.searchsorted(c_indices, u_c).astype(np.int32)
            v_loc = np.searchsorted(c_indices, v_c).astype(np.int32)
            adj_mat = csr_matrix((np.ones(len(u_loc), bool), (u_loc, v_loc)), shape=(len(c_indices), len(c_indices)))
            n_comp, comp_ids = csgraph.connected_components(adj_mat, directed=False)
        else:
            n_comp = len(c_indices)
            comp_ids = np.arange(n_comp, dtype=np.int32)

        comp_sizes = np.bincount(comp_ids)
        comp_areas = np.bincount(comp_ids, weights=tri_areas[c_indices])

        is_floater_comp = (comp_sizes < max_island_tris) & (comp_areas < max_island_area)
        if not np.any(is_floater_comp):
            continue

        tri_floater_comp = np.full(num_tris, -1, dtype=np.int32)
        floater_mask = is_floater_comp[comp_ids]
        floater_tris = c_indices[floater_mask]
        tri_floater_comp[floater_tris] = comp_ids[floater_mask]

        edge_mask = (tri_floater_comp[adj_u] >= 0) & (cleaned[adj_v] != cid)
        if np.any(edge_mask):
            u_comps = tri_floater_comp[adj_u[edge_mask]]
            v_nbr_lbls = cleaned[adj_v[edge_mask]]

            # Prioritize labeled neighbors over -1
            v_weights = np.where(v_nbr_lbls >= 0, 10.0, 1.0)

            sort_k = np.lexsort((v_nbr_lbls, u_comps))
            u_sorted = u_comps[sort_k]
            v_sorted = v_nbr_lbls[sort_k]
            w_sorted = v_weights[sort_k]

            split_mask = np.concatenate([[True], (u_sorted[:-1] != u_sorted[1:]) | (v_sorted[:-1] != v_sorted[1:])])
            pair_u = u_sorted[split_mask]
            pair_v = v_sorted[split_mask]

            pair_indices = np.cumsum(split_mask) - 1
            pair_weights = np.bincount(pair_indices, weights=w_sorted)

            comp_winners = {}
            comp_best_weights = {}
            for pu, pv, w in zip(pair_u, pair_v, pair_weights):
                if pu not in comp_best_weights or w > comp_best_weights[pu]:
                    comp_best_weights[pu] = w
                    comp_winners[pu] = pv

            for comp_id, winner in comp_winners.items():
                if winner >= 0:
                    t_to_change = c_indices[comp_ids == comp_id]
                    cleaned[t_to_change] = winner
                    total_absorbed += len(t_to_change)

    return cleaned, total_absorbed


def fill_unassigned_triangles(
    triangle_labels: np.ndarray,
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    tri_centroids: np.ndarray,
    max_bfs_iters: int = 50,
) -> np.ndarray:
    """
    Propagates labels across the dual mesh graph using multi-source BFS
    until 100% of triangles are assigned. Uses KDTree fallback for any disconnected islands.
    """
    cleaned = triangle_labels.copy()
    num_unassigned = int(np.count_nonzero(cleaned < 0))
    if num_unassigned == 0:
        return cleaned

    it = 0
    while it < max_bfs_iters:
        unassigned_mask = cleaned < 0
        if not np.any(unassigned_mask):
            break

        edge_mask = unassigned_mask[adj_u] & (~unassigned_mask[adj_v])
        if not np.any(edge_mask):
            break

        u_targets = adj_u[edge_mask]
        v_sources = cleaned[adj_v[edge_mask]]

        sort_idx = np.lexsort((v_sources, u_targets))
        u_sorted = u_targets[sort_idx]
        v_sorted = v_sources[sort_idx]

        split = np.concatenate([[True], (u_sorted[:-1] != u_sorted[1:]) | (v_sorted[:-1] != v_sorted[1:])])
        pair_u = u_sorted[split]
        pair_v = v_sorted[split]
        counts = np.diff(np.concatenate([np.flatnonzero(split), [len(u_sorted)]]))

        best_votes = {}
        best_lbl = {}
        for u, v, cnt in zip(pair_u, pair_v, counts):
            if u not in best_votes or cnt > best_votes[u]:
                best_votes[u] = cnt
                best_lbl[u] = v

        u_arr = np.array(list(best_lbl.keys()), dtype=np.int32)
        v_arr = np.array(list(best_lbl.values()), dtype=np.int32)
        cleaned[u_arr] = v_arr
        it += 1

    remaining_unassigned = int(np.count_nonzero(cleaned < 0))
    if remaining_unassigned > 0:
        assigned_indices = np.flatnonzero(cleaned >= 0)
        unassigned_indices = np.flatnonzero(cleaned < 0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tri_centroids[assigned_indices])
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        for u_idx in unassigned_indices:
            _, idxs, _ = kdtree.search_knn_vector_3d(tri_centroids[u_idx], 1)
            nearest_assigned = assigned_indices[idxs[0]]
            cleaned[u_idx] = cleaned[nearest_assigned]

    return cleaned


def absorb_outlier_floaters(
    triangle_labels: np.ndarray,
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    active_classes: List[int],
    min_island_size: int = 30,
) -> Tuple[np.ndarray, int]:
    """
    Absorbs isolated floaters and small clusters (< min_island_size triangles) of any class
    surrounded by another class into the dominant surrounding neighbor class.
    Uses fast vectorized boundary majority voting.
    """
    cleaned_labels = triangle_labels.copy()
    num_tris = len(triangle_labels)
    total_absorbed = 0

    for cid in active_classes:
        c_indices = np.flatnonzero(cleaned_labels == cid)
        if len(c_indices) == 0:
            continue

        is_c = np.zeros(num_tris, dtype=bool)
        is_c[c_indices] = True
        c_edges = is_c[adj_u] & is_c[adj_v]
        u_c = adj_u[c_edges]
        v_c = adj_v[c_edges]

        if len(u_c) > 0:
            u_loc = np.searchsorted(c_indices, u_c).astype(np.int32)
            v_loc = np.searchsorted(c_indices, v_c).astype(np.int32)
            adj_mat = csr_matrix((np.ones(len(u_loc), bool), (u_loc, v_loc)), shape=(len(c_indices), len(c_indices)))
            n_comp, comp_ids = csgraph.connected_components(adj_mat, directed=False)
        else:
            n_comp = len(c_indices)
            comp_ids = np.arange(n_comp, dtype=np.int32)

        comp_sizes = np.bincount(comp_ids)
        is_small_comp = comp_sizes < min_island_size
        if not np.any(is_small_comp):
            continue

        tri_small_comp = np.full(num_tris, -1, dtype=np.int32)
        small_comp_mask = is_small_comp[comp_ids]
        small_tris = c_indices[small_comp_mask]
        tri_small_comp[small_tris] = comp_ids[small_comp_mask]

        edge_mask = (tri_small_comp[adj_u] >= 0) & (cleaned_labels[adj_v] >= 0) & (cleaned_labels[adj_v] != cid)
        if np.any(edge_mask):
            u_comps = tri_small_comp[adj_u[edge_mask]]
            v_nbr_lbls = cleaned_labels[adj_v[edge_mask]]

            sort_k = np.lexsort((v_nbr_lbls, u_comps))
            u_sorted = u_comps[sort_k]
            v_sorted = v_nbr_lbls[sort_k]

            split_mask = np.concatenate([[True], (u_sorted[:-1] != u_sorted[1:]) | (v_sorted[:-1] != v_sorted[1:])])
            pair_u = u_sorted[split_mask]
            pair_v = v_sorted[split_mask]
            run_lens = np.diff(np.concatenate([np.flatnonzero(split_mask), [len(u_sorted)]]))

            comp_winners = {}
            comp_best_counts = {}
            for pu, pv, cnt in zip(pair_u, pair_v, run_lens):
                if pu not in comp_best_counts or cnt > comp_best_counts[pu]:
                    comp_best_counts[pu] = cnt
                    comp_winners[pu] = pv

            for comp_id, winner in comp_winners.items():
                t_to_change = c_indices[comp_ids == comp_id]
                cleaned_labels[t_to_change] = winner
                total_absorbed += len(t_to_change)

    return cleaned_labels, total_absorbed


def absorb_enclosed_patches(
    labels: np.ndarray,
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    tri_normals: np.ndarray,
    active_classes: List[int],
    max_patch_size: int = 1500,
    encirclement_ratio_thresh: float = 0.70,
) -> Tuple[np.ndarray, int]:
    """
    Absorbs enclosed patches (< max_patch_size triangles) into their enclosing host class
    if >= encirclement_ratio_thresh (default 70%) of their boundary contact on the dual graph
    belongs to a single neighboring class.
    Respects physical orientation invariants:
      - Primarily vertical patches cannot become Floor (14) or Ceiling (1).
      - Downward patches cannot become Floor (14).
    """
    cleaned = labels.copy()
    num_tris = len(cleaned)
    total_absorbed = 0

    for cid in active_classes:
        c_mask = (cleaned == cid)
        c_indices = np.flatnonzero(c_mask)
        if len(c_indices) == 0:
            continue

        c_edges = c_mask[adj_u] & c_mask[adj_v]
        if not np.any(c_edges):
            continue

        inv_map = np.full(num_tris, -1, dtype=np.int32)
        inv_map[c_indices] = np.arange(len(c_indices), dtype=np.int32)
        u_loc = inv_map[adj_u[c_edges]]
        v_loc = inv_map[adj_v[c_edges]]
        sub_adj = csr_matrix((np.ones(len(u_loc), dtype=bool), (u_loc, v_loc)), shape=(len(c_indices), len(c_indices)))
        n_comp, comp_ids = csgraph.connected_components(sub_adj, directed=False)
        comp_sizes = np.bincount(comp_ids)

        small_comps = np.where(comp_sizes < max_patch_size)[0]
        if len(small_comps) == 0:
            continue

        for comp_id in small_comps:
            comp_tris = c_indices[comp_ids == comp_id]
            is_p = np.zeros(num_tris, dtype=bool)
            is_p[comp_tris] = True

            b_edges = is_p[adj_u] & (~is_p[adj_v])
            if not np.any(b_edges):
                continue

            nbr_labels = cleaned[adj_v[b_edges]]
            ext_nbrs = nbr_labels[nbr_labels != cid]
            if len(ext_nbrs) == 0:
                continue

            vals, cnts = np.unique(ext_nbrs, return_counts=True)
            best_idx = cnts.argmax()
            dominant_nbr = int(vals[best_idx])
            dominant_frac = cnts[best_idx] / len(ext_nbrs)

            if dominant_frac >= encirclement_ratio_thresh:
                # Check orientation invariants
                c_norms = tri_normals[comp_tris]
                mean_ny = float(c_norms[:, 1].mean())
                frac_vert = float((np.abs(c_norms[:, 1]) < 0.25).mean())

                # Vertical surfaces cannot become floor or ceiling
                if dominant_nbr in (14, 1) and frac_vert > 0.40:
                    continue
                # Downward surfaces cannot become floor
                if dominant_nbr == 14 and mean_ny < 0.0:
                    continue
                # Upward surfaces cannot become ceiling
                if dominant_nbr == 1 and mean_ny > 0.0:
                    continue
                # Wall is the background substrate; foreground objects mounted on walls (posters, art, shelves)
                # must not be swallowed into the wall if they have substantial size (>= 200 triangles)
                if dominant_nbr == 5 and cid not in (1, 14) and len(comp_tris) >= 200:
                    continue

                cleaned[comp_tris] = dominant_nbr
                total_absorbed += len(comp_tris)

    return cleaned, total_absorbed


def prune_teleported_scattered_crumbs(
    labels: np.ndarray,
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    tri_centroids: np.ndarray,
    tri_normals: np.ndarray,
    active_classes: List[int],
    max_jump_distance: float = 1.00,
    max_grouping_dist: float = 0.75,
    structural_classes: Set[int] = {1, 5, 14},
) -> Tuple[np.ndarray, int]:
    """
    Prevents class scattering and multi-room teleportation (Scene-Agnostic):
    Physical objects (furniture, decorations, appliances, fixtures) exist as compact local clusters.
    Iteratively groups contiguous physical parts of an object (within max_grouping_dist <= 0.75m of the largest component).
    If a secondary disconnected component of a foreground class is separated by > max_jump_distance
    (default 1.0m) from all parts of the primary cluster, it represents 2D projection
    seam bleed / grazing ray noise across the room and is absorbed into its touching local boundary neighbors.
    """
    cleaned = labels.copy()
    num_tris = len(cleaned)
    total_reassigned = 0

    for cid in active_classes:
        if cid in structural_classes:
            continue

        c_mask = (cleaned == cid)
        c_indices = np.flatnonzero(c_mask)
        if len(c_indices) == 0:
            continue

        c_edges = c_mask[adj_u] & c_mask[adj_v]
        if not np.any(c_edges):
            continue

        inv_map = np.full(num_tris, -1, dtype=np.int32)
        inv_map[c_indices] = np.arange(len(c_indices), dtype=np.int32)
        u_loc = inv_map[adj_u[c_edges]]
        v_loc = inv_map[adj_v[c_edges]]
        sub_adj = csr_matrix((np.ones(len(u_loc), dtype=bool), (u_loc, v_loc)), shape=(len(c_indices), len(c_indices)))
        n_comp, comp_ids = csgraph.connected_components(sub_adj, directed=False)
        comp_sizes = np.bincount(comp_ids)

        if n_comp <= 1:
            continue

        root_pid = int(comp_sizes.argmax())
        comp_centers = np.array([
            tri_centroids[c_indices[comp_ids == c]].mean(axis=0)
            for c in range(n_comp)
        ])

        # Grow primary cluster by proximity to root
        primary = {root_pid}
        changed = True
        while changed:
            changed = False
            p_list = list(primary)
            for c in range(n_comp):
                if c not in primary:
                    dist = np.min([np.linalg.norm(comp_centers[c] - comp_centers[p]) for p in p_list])
                    if dist <= max_grouping_dist:
                        primary.add(c)
                        changed = True

        primary_centroids = comp_centers[list(primary)]

        for comp_id in range(n_comp):
            if comp_id in primary:
                continue

            comp_tris = c_indices[comp_ids == comp_id]
            comp_center = comp_centers[comp_id]
            min_dist_to_primary = np.min(np.linalg.norm(primary_centroids - comp_center, axis=1))

            if min_dist_to_primary > max_jump_distance:
                is_p = np.zeros(num_tris, dtype=bool)
                is_p[comp_tris] = True
                b_edges = is_p[adj_u] & (~is_p[adj_v])
                if not np.any(b_edges):
                    continue

                nbr_labels = cleaned[adj_v[b_edges]]
                ext_nbrs = nbr_labels[nbr_labels != cid]
                if len(ext_nbrs) == 0:
                    continue

                c_norms = tri_normals[comp_tris]
                mean_ny = float(c_norms[:, 1].mean())
                frac_vert = float((np.abs(c_norms[:, 1]) < 0.25).mean())

                valid_nbrs = ext_nbrs.copy()
                if frac_vert > 0.40:
                    valid_nbrs = valid_nbrs[(valid_nbrs != 14) & (valid_nbrs != 1)]
                elif mean_ny < 0.0:
                    valid_nbrs = valid_nbrs[valid_nbrs != 14]

                cand_nbrs = valid_nbrs if len(valid_nbrs) > 0 else ext_nbrs
                if len(cand_nbrs) > 0:
                    vals, cnts = np.unique(cand_nbrs, return_counts=True)
                    winner = int(vals[cnts.argmax()])
                    cleaned[comp_tris] = winner
                    total_reassigned += len(comp_tris)

    return cleaned, total_reassigned


def merge_proximate_components(
    valid_comps: List[int],
    c_indices: np.ndarray,
    comp_ids: np.ndarray,
    tri_centroids: np.ndarray,
    max_dist: float = 0.08,
) -> List[np.ndarray]:
    """
    Merges disconnected components of the same class that are within max_dist
    (e.g. 8 cm) of each other into unified instance triangle arrays.
    Uses exact scipy.spatial.cKDTree.query_ball_tree to evaluate 100% of points without downsampling.
    """
    if len(valid_comps) <= 1:
        return [c_indices[comp_ids == c] for c in valid_comps]

    from scipy.spatial import cKDTree

    comp_tris_list = [c_indices[comp_ids == c] for c in valid_comps]
    n_comps = len(valid_comps)

    trees = [cKDTree(tri_centroids[t]) for t in comp_tris_list]
    adj_matrix = np.zeros((n_comps, n_comps), dtype=bool)
    np.fill_diagonal(adj_matrix, True)

    for i in range(n_comps):
        for j in range(i + 1, n_comps):
            matches = trees[i].query_ball_tree(trees[j], r=max_dist)
            if any(len(m) > 0 for m in matches):
                adj_matrix[i, j] = True
                adj_matrix[j, i] = True

    n_groups, group_ids = csgraph.connected_components(csr_matrix(adj_matrix), directed=False)
    merged_tris = []
    for g in range(n_groups):
        member_comps = [i for i in range(n_comps) if group_ids[i] == g]
        all_tris = np.concatenate([comp_tris_list[i] for i in member_comps])
        merged_tris.append(all_tris)

    merged_tris.sort(key=len, reverse=True)
    return merged_tris


def extract_instances(
    triangle_labels: np.ndarray,
    active_classes: List[int],
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    tri_areas: np.ndarray,
    tri_centroids: np.ndarray,
    tri_normals: np.ndarray,
    gravity_up: np.ndarray,
    min_instance_size: int = 150,
    min_instance_area: float = 0.02,
    min_instance_span: float = 0.15,
    max_proximity_merge_dist: float = 0.08,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, int]]:
    """
    Performs 3D mesh instance segmentation across all active classes.
    Groups triangles of each class into spatial connected components >= min_instance_size.
    Enforces physical size checks (area >= min_instance_area, span >= min_instance_span).
    Optionally unifies components of the same class within max_proximity_merge_dist.
    Returns:
      instances: Dict[inst_id, Dict[str, Any]]
      class_instance_counts: Dict[cid, int]
    """
    instances = {}
    class_instance_counts = {}
    num_tris = len(triangle_labels)

    for cid in active_classes:
        c_mask = triangle_labels == cid
        c_indices = np.flatnonzero(c_mask)
        if len(c_indices) == 0:
            class_instance_counts[cid] = 0
            continue

        is_c = np.zeros(num_tris, dtype=bool)
        is_c[c_indices] = True
        c_edges = is_c[adj_u] & is_c[adj_v]
        u_c = adj_u[c_edges]
        v_c = adj_v[c_edges]

        compact_map = {t: i for i, t in enumerate(c_indices)}
        if len(u_c) > 0:
            u_loc = np.array([compact_map[t] for t in u_c], dtype=np.int32)
            v_loc = np.array([compact_map[t] for t in v_c], dtype=np.int32)
            adj_mat = csr_matrix((np.ones(len(u_loc), bool), (u_loc, v_loc)), shape=(len(c_indices), len(c_indices)))
            n_comp, comp_ids = csgraph.connected_components(adj_mat, directed=False)
        else:
            n_comp = len(c_indices)
            comp_ids = np.arange(n_comp, dtype=np.int32)

        comp_sizes = np.bincount(comp_ids)
        valid_comps = [c for c in np.argsort(-comp_sizes) if comp_sizes[c] >= min_instance_size]

        if cid == 5 and len(valid_comps) > 1:
            # Wall is 1 continuous structural perimeter shell. Unify all components into the single primary wall instance.
            merged_comp_tris = [np.concatenate([c_indices[comp_ids == comp_id] for comp_id in valid_comps])]
        elif max_proximity_merge_dist > 0 and len(valid_comps) > 1:
            merged_comp_tris = merge_proximate_components(
                valid_comps, c_indices, comp_ids, tri_centroids, max_dist=max_proximity_merge_dist
            )
        else:
            merged_comp_tris = [c_indices[comp_ids == comp_id] for comp_id in valid_comps]

        # Enforce physical dimension checks on candidate instances
        valid_merged_tris = []
        for p_tris in merged_comp_tris:
            tot_a = float(tri_areas[p_tris].sum())
            p_cents = tri_centroids[p_tris]
            spans = p_cents.max(axis=0) - p_cents.min(axis=0) if len(p_tris) > 0 else np.zeros(3)
            if len(p_tris) < min_instance_size or tot_a < min_instance_area or max(spans) < min_instance_span:
                continue
            valid_merged_tris.append(p_tris)

        class_instance_counts[cid] = len(valid_merged_tris)

        for inst_idx, p_tris in enumerate(valid_merged_tris):
            inst_id = f"inst_{cid}_{inst_idx}"
            p_areas = tri_areas[p_tris]
            p_cents = tri_centroids[p_tris]
            p_norms = tri_normals[p_tris]
            tot_a = float(p_areas.sum())

            mins = p_cents.min(axis=0)
            maxs = p_cents.max(axis=0)
            spans = maxs - mins
            mean_c = p_cents.mean(axis=0)

            p_up = np.dot(p_norms, gravity_up)
            p_h = np.dot(p_cents, gravity_up)
            f_up = float(p_areas[p_up > 0.60].sum() / max(1e-8, tot_a))
            f_dn = float(p_areas[p_up < -0.40].sum() / max(1e-8, tot_a))
            f_vt = float(p_areas[np.abs(p_up) < 0.25].sum() / max(1e-8, tot_a))

            instances[inst_id] = {
                "instance_id": inst_id,
                "class_id": cid,
                "instance_index": inst_idx,
                "triangle_count": int(len(p_tris)),
                "area_m2": round(tot_a, 3),
                "bounds_min": [round(float(x), 3) for x in mins],
                "bounds_max": [round(float(x), 3) for x in maxs],
                "spans": [round(float(x), 3) for x in spans],
                "mean_centroid": [round(float(x), 3) for x in mean_c],
                "frac_up": round(f_up, 2),
                "frac_down": round(f_dn, 2),
                "frac_vert": round(f_vt, 2),
                "h_mean": round(float(p_h.mean()), 3),
                "h_span": round(float(p_h.max() - p_h.min()), 3),
                "triangles": p_tris,
            }

    return instances, class_instance_counts


def converge_class_instances_iteratively(
    triangle_labels: np.ndarray,
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    tri_areas: np.ndarray,
    tri_centroids: np.ndarray,
    tri_normals: np.ndarray,
    gravity_up: np.ndarray,
    active_classes: List[int],
    min_instance_size: int = 150,
    min_instance_area: float = 0.02,
    min_instance_span: float = 0.15,
    max_proximity_merge_dist: float = 0.08,
    max_iterations: int = 10,
    cluster_ratio_thresh: float = 0.75,
    coverage_target_thresh: float = 0.75,
    protect_min_triangles: int = 10000,
    protect_min_area: float = 0.50,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Iteratively maximizes total class area coverage while minimizing the number of instances.
    Enforces strict physical dimensional and surface area checks:
      1. For each active class, extracts components with cKDTree proximity merging.
      2. Prunes sub-scale crumbs (tris < min_instance_size, area < min_instance_area, span < min_instance_span).
      3. If class has <= 1 valid instance or achieves 100% coverage, flags it as CONVERGED.
      4. For multi-instance classes:
         - Greedily groups instances whose sizes are >= cluster_ratio_thresh (0.75) of the cluster leader.
         - Accumulates clusters until cumulative coverage reaches >= coverage_target_thresh (0.75).
         - Protects large architectural components (area >= 0.50 m2 or triangles >= 10,000) from pruning.
         - Unassigns residual long-tail crumbs and reassigns them to enclosing neighbor classes.
      5. Loops until all classes converge or 0 triangles are reassigned.
    """
    num_tris = len(triangle_labels)
    refined_labels = triangle_labels.copy()
    all_reassigned_records = []
    converged_classes: Set[int] = set()

    print(f"\n--- Starting Iterative Instance Convergence (Max {max_iterations} Iterations) ---")
    print(f"  Parameters: min_tris={min_instance_size}, min_area={min_instance_area:.2f}m2, min_span={min_instance_span:.2f}m, cluster_ratio={cluster_ratio_thresh:.2f}, coverage_target={coverage_target_thresh:.2f}, protect_min_area={protect_min_area:.2f}m2, protect_min_tris={protect_min_triangles}")

    for it in range(1, max_iterations + 1):
        t0_it = time.perf_counter()

        # Step A: Extract candidate instances per class with exact proximity merging
        instances_by_class: Dict[int, List[Dict[str, Any]]] = {}
        inst_to_tris: Dict[str, np.ndarray] = {}
        total_valid_instances = 0
        outlier_iids = set()
        same = refined_labels[adj_u] == refined_labels[adj_v]
        same_u, same_v = adj_u[same], adj_v[same]
        edge_labels = refined_labels[same_u]
        edge_order = np.argsort(edge_labels, kind="stable")
        edge_labels, same_u, same_v = edge_labels[edge_order], same_u[edge_order], same_v[edge_order]

        for cid in active_classes:
            c_indices = np.flatnonzero(refined_labels == cid)
            if len(c_indices) == 0:
                converged_classes.add(cid)
                continue

            # Skip classes already converged unless they received new triangles
            if cid in converged_classes:
                continue

            lo = np.searchsorted(edge_labels, cid, side="left")
            hi = np.searchsorted(edge_labels, cid, side="right")
            u_c, v_c = same_u[lo:hi], same_v[lo:hi]
            if len(u_c):
                u_loc = np.searchsorted(c_indices, u_c).astype(np.int32)
                v_loc = np.searchsorted(c_indices, v_c).astype(np.int32)
                adj_mat = csr_matrix((np.ones(len(u_loc), bool), (u_loc, v_loc)), shape=(len(c_indices), len(c_indices)))
                n_comp, comp_ids = csgraph.connected_components(adj_mat, directed=False)
            else:
                n_comp = len(c_indices)
                comp_ids = np.arange(n_comp, dtype=np.int32)

            comp_sizes = np.bincount(comp_ids)
            valid_comps = [c for c in np.argsort(-comp_sizes) if comp_sizes[c] >= min_instance_size]

            # Merge proximate components using exact cKDTree
            if max_proximity_merge_dist > 0 and len(valid_comps) > 1:
                merged_comp_tris = merge_proximate_components(
                    valid_comps, c_indices, comp_ids, tri_centroids, max_dist=max_proximity_merge_dist
                )
            elif len(valid_comps) == 1:
                merged_comp_tris = [c_indices[comp_ids == valid_comps[0]]]
            else:
                merged_comp_tris = []

            total_valid_instances += len(merged_comp_tris)

            # Single-instance classes are immediately converged
            if len(merged_comp_tris) <= 1:
                converged_classes.add(cid)
                continue

            c_inst_list = []
            for inst_idx, p_tris in enumerate(merged_comp_tris):
                iid = f"inst_{cid}_{inst_idx}"
                tot_a = float(tri_areas[p_tris].sum())
                p_cents = tri_centroids[p_tris]
                spans = p_cents.max(axis=0) - p_cents.min(axis=0) if len(p_tris) > 0 else np.zeros(3)
                info = {
                    "instance_id": iid,
                    "class_id": cid,
                    "instance_index": inst_idx,
                    "triangle_count": int(len(p_tris)),
                    "area_m2": tot_a,
                    "max_span": float(max(spans)),
                }
                c_inst_list.append(info)
                inst_to_tris[iid] = p_tris

            # Step B: Dimension clustering, physical size check, and cumulative coverage thresholding
            c_inst_list.sort(key=lambda x: -x["triangle_count"])
            tot_c_tris = sum(x["triangle_count"] for x in c_inst_list)
            selected_indices = set()

            # Automatic outlier rejection for sub-scale crumbs
            subscale_indices = {
                j for j in range(len(c_inst_list))
                if c_inst_list[j]["triangle_count"] < min_instance_size
                or c_inst_list[j]["area_m2"] < min_instance_area
                or c_inst_list[j]["max_span"] < min_instance_span
            }

            for ref_idx in range(len(c_inst_list)):
                if ref_idx in selected_indices or ref_idx in subscale_indices:
                    continue
                ref_t = c_inst_list[ref_idx]["triangle_count"]
                # Cluster instances with at least cluster_ratio_thresh of leader
                cluster = [
                    j for j in range(len(c_inst_list))
                    if j not in selected_indices and j not in subscale_indices and c_inst_list[j]["triangle_count"] >= cluster_ratio_thresh * ref_t
                ]
                for j in cluster:
                    selected_indices.add(j)
                cov = sum(c_inst_list[j]["triangle_count"] for j in selected_indices) / tot_c_tris
                if cov >= coverage_target_thresh:
                    break

            # Architectural protection (protect floor slabs, walls, ceilings)
            is_structural = (cid in {1, 5, 14})
            for j in range(len(c_inst_list)):
                if j not in selected_indices:
                    if cid == 5:
                        # For Wall: The room perimeter wall is 1 continuous structural envelope.
                        # Protect only the primary wall shell or massive sub-structures (>= 50,000 tris).
                        if j == 0 or c_inst_list[j]["triangle_count"] >= 50000:
                            selected_indices.add(j)
                    elif is_structural:
                        # For Floor (14) and Ceiling (1): protect multi-level slabs (>= 1,000 tris and >= 0.15 m2)
                        if (c_inst_list[j]["triangle_count"] >= 1000 and
                            c_inst_list[j]["area_m2"] >= 0.15):
                            selected_indices.add(j)
                    elif (c_inst_list[j]["triangle_count"] >= protect_min_triangles and
                          c_inst_list[j]["area_m2"] >= protect_min_area):
                        selected_indices.add(j)

            # Identify unselected instances as outliers to reassign
            outliers = [c_inst_list[j] for j in range(len(c_inst_list)) if j not in selected_indices]
            if len(outliers) == 0:
                converged_classes.add(cid)
            else:
                for o in outliers:
                    outlier_iids.add(o["instance_id"])

        if not outlier_iids:
            print(f"  Iteration {it}: Total Instances = {total_valid_instances}, Outlier Instances = 0 -> Converged! ({len(converged_classes)}/{len(active_classes)} classes converged)")
            break

        # Step C: Reassign each outlier instance
        outlier_tris_mask = np.zeros(num_tris, dtype=bool)
        for iid in outlier_iids:
            outlier_tris_mask[inst_to_tris[iid]] = True

        non_out_indices = np.flatnonzero(~outlier_tris_mask)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tri_centroids[non_out_indices])
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        sorted_outlier_iids = sorted(outlier_iids, key=lambda x: len(inst_to_tris[x]), reverse=True)
        iter_reassigned = 0
        external_reassignments: Dict[int, int] = {c: 0 for c in active_classes}

        for iid in sorted_outlier_iids:
            cid = int(iid.split("_")[1])
            patch_tris = inst_to_tris[iid]
            is_p = np.zeros(num_tris, dtype=bool)
            is_p[patch_tris] = True
            b_edges = is_p[adj_u] & (~is_p[adj_v])

            winner = None
            method = ""

            # Strategy 1: Enclosing class via dual-graph boundary majority
            if np.any(b_edges):
                nbr_lbls = refined_labels[adj_v[b_edges]]
                ext_nbrs = nbr_lbls[nbr_lbls != cid]
                # Invariants: Floor (14) cannot be reassigned to Bed (2)
                if cid == 14:
                    ext_nbrs = ext_nbrs[ext_nbrs != 2]
                # Invariants: vertical surfaces cannot be assigned to Floor (14) or Ceiling (1)
                p_norms = tri_normals[patch_tris]
                frac_vert = float((np.abs(p_norms[:, 1]) < 0.25).mean()) if len(patch_tris) > 0 else 0.0
                mean_ny = float(p_norms[:, 1].mean()) if len(patch_tris) > 0 else 0.0
                if frac_vert > 0.40:
                    ext_nbrs = ext_nbrs[(ext_nbrs != 14) & (ext_nbrs != 1)]
                elif mean_ny < 0.0:
                    ext_nbrs = ext_nbrs[ext_nbrs != 14]
                if len(ext_nbrs) > 0:
                    vals, cnts = np.unique(ext_nbrs, return_counts=True)
                    winner = int(vals[cnts.argmax()])
                    method = "boundary_enclosing"
                else:
                    winner = cid
                    method = "self_reabsorbed_boundary"

            # Strategy 2: Closest non-outlier class within a small geometric gap (bridging mesh holes or cuts <= max_proximity_merge_dist)
            if winner is None and max_proximity_merge_dist > 0:
                query_pts = tri_centroids[patch_tris]
                if len(query_pts) > 30:
                    step = max(1, len(query_pts) // 30)
                    query_pts = query_pts[::step]
                best_cand = None
                min_sq_dist = max_proximity_merge_dist * max_proximity_merge_dist
                for q_pt in query_pts:
                    k, idxs, dists_sq = kdtree.search_radius_vector_3d(q_pt, max_proximity_merge_dist)
                    if k > 0:
                        for near_idx, d2 in zip(idxs, dists_sq):
                            if d2 < min_sq_dist:
                                cand_lbl = int(refined_labels[non_out_indices[near_idx]])
                                if cid == 14 and cand_lbl == 2:
                                    continue
                                min_sq_dist = d2
                                best_cand = cand_lbl
                if best_cand is not None:
                    winner = best_cand
                    method = "proximity_gap_bridge" if best_cand != cid else "self_reabsorbed_gap"

            if winner is None:
                winner = cid
                method = "self_reabsorbed_isolated"

            if winner == cid:
                refined_labels[patch_tris] = cid
                all_reassigned_records.append({
                    "iteration": it,
                    "instance_id": iid,
                    "original_class": cid,
                    "assigned_class": cid,
                    "triangle_count": len(patch_tris),
                    "method": method,
                })
            else:
                refined_labels[patch_tris] = winner
                iter_reassigned += len(patch_tris)
                external_reassignments[cid] = external_reassignments.get(cid, 0) + len(patch_tris)
                converged_classes.discard(winner) # unflag winner so newly expanded geometry is re-checked
                all_reassigned_records.append({
                    "iteration": it,
                    "instance_id": iid,
                    "original_class": cid,
                    "assigned_class": winner,
                    "triangle_count": len(patch_tris),
                    "method": method,
                })

        # Early convergence for classes whose outliers only reabsorbed into themselves
        classes_with_outliers = {int(iid.split("_")[1]) for iid in outlier_iids}
        for c_out in classes_with_outliers:
            if external_reassignments.get(c_out, 0) == 0:
                converged_classes.add(c_out)
                print(f"    Class {c_out}: all outliers reabsorbed internally into same class -> flagged converged early.")

        # Micro-island cleanup
        refined_labels, n_micro = absorb_outlier_floaters(
            refined_labels, adj_u, adj_v, active_classes, min_island_size=min_instance_size
        )

        outlier_tris_count = sum(len(inst_to_tris[x]) for x in outlier_iids)
        print(
            f"  Iteration {it}: Total Instances = {total_valid_instances}, "
            f"Outliers Absorbed = {len(outlier_iids)} ({outlier_tris_count:,} tris), "
            f"Micro-cleaned = {n_micro:,} tris, Converged Classes = {len(converged_classes)}/{len(active_classes)} ({time.perf_counter() - t0_it:.2f}s)"
        )

        if iter_reassigned == 0:
            print(f"  ==> Reassignment stabilized at iteration {it}.")
            break

    return refined_labels, all_reassigned_records


def absorb_class_outliers(*args, **kwargs):
    """Backwards-compatible wrapper calling converge_class_instances_iteratively."""
    return converge_class_instances_iteratively(*args, **kwargs)



def decompose_class_into_connected_pieces(
    class_tri_indices: np.ndarray,
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    tri_areas: np.ndarray,
    tri_centroids: np.ndarray,
    num_total_triangles: int,
    tri_normals: np.ndarray = None,
    gravity_up: np.ndarray = None,
) -> List[Dict[str, Any]]:
    """
    Decomposes triangles of a single class into spatially connected components (pieces).
    Optionally computes piece-level surface orientation fractions and elevation bounds.
    """
    if len(class_tri_indices) == 0:
        return []

    is_class = np.zeros(num_total_triangles, dtype=bool)
    is_class[class_tri_indices] = True

    class_edges = is_class[adj_u] & is_class[adj_v]
    u_c = adj_u[class_edges]
    v_c = adj_v[class_edges]

    compact_map = {tri: i for i, tri in enumerate(class_tri_indices)}
    if len(u_c) > 0:
        u_local = np.array([compact_map[t] for t in u_c], dtype=np.int32)
        v_local = np.array([compact_map[t] for t in v_c], dtype=np.int32)
        adj_mat = csr_matrix(
            (np.ones(len(u_local), dtype=bool), (u_local, v_local)),
            shape=(len(class_tri_indices), len(class_tri_indices)),
        )
        n_comp, comp_ids = csgraph.connected_components(adj_mat, directed=False)
    else:
        n_comp = len(class_tri_indices)
        comp_ids = np.arange(n_comp, dtype=np.int32)

    pieces = []
    for c_id in range(n_comp):
        c_mask = comp_ids == c_id
        p_tri_indices = class_tri_indices[c_mask]
        p_areas = tri_areas[p_tri_indices]
        p_cents = tri_centroids[p_tri_indices]
        tot_area = float(p_areas.sum())
        mins = p_cents.min(axis=0)
        maxs = p_cents.max(axis=0)
        spans = maxs - mins

        piece_dict = {
            "component_id": int(c_id),
            "triangle_count": int(len(p_tri_indices)),
            "area_m2": float(tot_area),
            "bounds_min": mins.tolist(),
            "bounds_max": maxs.tolist(),
            "spans": spans.tolist(),
            "mean_centroid": p_cents.mean(axis=0).tolist(),
        }

        if tri_normals is not None and gravity_up is not None:
            p_norms = tri_normals[p_tri_indices]
            p_up = np.dot(p_norms, gravity_up)
            p_h = np.dot(p_cents, gravity_up)
            piece_dict["frac_up"] = float(p_areas[p_up > 0.60].sum() / max(1e-8, tot_area))
            piece_dict["frac_down"] = float(p_areas[p_up < -0.40].sum() / max(1e-8, tot_area))
            piece_dict["frac_vert"] = float(p_areas[np.abs(p_up) < 0.25].sum() / max(1e-8, tot_area))
            piece_dict["h_mean"] = float(p_h.mean())
            piece_dict["h_min"] = float(p_h.min())
            piece_dict["h_max"] = float(p_h.max())
            piece_dict["h_span"] = float(p_h.max() - p_h.min())

        pieces.append(piece_dict)

    pieces.sort(key=lambda p: p["area_m2"], reverse=True)
    return pieces


def fit_ransac_plane(
    points: np.ndarray,
    distance_threshold: float = 0.05,
    max_iterations: int = 500,
) -> Tuple[np.ndarray, float, float]:
    """
    Fits a plane Ax + By + Cz + D = 0 using RANSAC.
    Returns (plane_equation [4], inlier_ratio, rms_residual).
    """
    if len(points) < 3:
        return np.array([0, 1, 0, 0], dtype=np.float32), 0.0, 0.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    try:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=max_iterations,
        )
        inlier_ratio = len(inliers) / len(points)
        eq = np.array(plane_model, dtype=np.float32)

        inlier_pts = points[inliers]
        dists = np.abs(np.dot(inlier_pts, eq[:3]) + eq[3]) / np.linalg.norm(eq[:3])
        rmse = float(np.sqrt(np.mean(dists ** 2)))
        return eq, float(inlier_ratio), float(rmse)
    except Exception:
        return np.array([0, 1, 0, 0], dtype=np.float32), 0.0, 0.0


def analyze_class_geometry(
    vertices: np.ndarray,
    triangles: np.ndarray,
    triangle_labels: np.ndarray,
    class_ids: List[int],
    gravity_up: np.ndarray,
    scene_bands: Dict[str, float],
) -> Dict[int, Dict[str, Any]]:
    """
    Computes comprehensive geometric, physical, and topological features for each class.
    """
    num_triangles = len(triangles)
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(2.0 * areas[:, None], 1e-8)
    centroids = (v0 + v1 + v2) / 3.0

    h_vals = np.dot(centroids, gravity_up)
    up_dots = np.dot(normals, gravity_up)

    adj_u, adj_v = build_mesh_dual_graph(triangles)

    class_set = set(class_ids)
    touching_pairs = {}
    valid_edge = (triangle_labels[adj_u] >= 0) & (triangle_labels[adj_v] >= 0)
    diff_edge = valid_edge & (triangle_labels[adj_u] != triangle_labels[adj_v])
    if np.any(diff_edge):
        u_lbl = triangle_labels[adj_u[diff_edge]]
        v_lbl = triangle_labels[adj_v[diff_edge]]
        for ul, vl in zip(u_lbl, v_lbl):
            if ul in class_set and vl in class_set:
                pair = (min(ul, vl), max(ul, vl))
                touching_pairs[pair] = touching_pairs.get(pair, 0) + 1

    features_by_class = {}
    h_min = scene_bands["h_min"]
    h_scene = scene_bands["h_scene"]

    for cid in class_ids:
        tri_mask = triangle_labels == cid
        c_count = int(np.count_nonzero(tri_mask))
        if c_count == 0:
            continue

        c_indices = np.flatnonzero(tri_mask)
        c_areas = areas[tri_mask]
        tot_area = float(c_areas.sum())
        c_cents = centroids[tri_mask]
        c_norms = normals[tri_mask]
        c_h = h_vals[tri_mask]
        c_up = up_dots[tri_mask]

        frac_up = float(c_areas[c_up > 0.60].sum() / max(1e-8, tot_area))
        frac_down = float(c_areas[c_up < -0.40].sum() / max(1e-8, tot_area))
        frac_vert = float(c_areas[np.abs(c_up) < 0.25].sum() / max(1e-8, tot_area))

        c_h_min = float(c_h.min())
        c_h_max = float(c_h.max())
        c_h_span = c_h_max - c_h_min
        c_h_mean = float(c_h.mean())
        norm_h_mean = (c_h_mean - h_min) / h_scene
        norm_h_span = c_h_span / h_scene

        c_mins = c_cents.min(axis=0)
        c_maxs = c_cents.max(axis=0)
        spans_3d = c_maxs - c_mins

        up_ax = scene_bands["up_axis_index"]
        horiz_axes = [i for i in range(3) if i != up_ax]
        horiz_span_1 = float(spans_3d[horiz_axes[0]])
        horiz_span_2 = float(spans_3d[horiz_axes[1]])
        horiz_extent = max(horiz_span_1, horiz_span_2)

        pieces = decompose_class_into_connected_pieces(
            class_tri_indices=c_indices,
            adj_u=adj_u,
            adj_v=adj_v,
            tri_areas=areas,
            tri_centroids=centroids,
            num_total_triangles=num_triangles,
            tri_normals=normals,
            gravity_up=gravity_up,
        )
        largest_piece_area = pieces[0]["area_m2"] if pieces else 0.0
        num_pieces = len(pieces)

        subsample = c_cents if len(c_cents) <= 10000 else c_cents[np.random.choice(len(c_cents), 10000, replace=False)]
        plane_eq, inlier_ratio, rmse = fit_ransac_plane(subsample, distance_threshold=0.05)

        # Stepped stair tread analysis:
        # Check upward-facing surfaces for true circulation flight geometry
        is_tread = c_up > 0.60
        tread_h = c_h[is_tread]
        stair_metrics = {"step_peaks": 0, "mean_step_rise": 0.0, "total_run": 0.0, "total_rise": 0.0, "is_stepped": False}
        if len(tread_h) >= 4 and (tread_h.max() - tread_h.min()) >= 0.35:
            # Cluster tread heights into 5cm increments
            rounded_h = np.round(tread_h * 20.0) / 20.0  # nearest 5cm
            vals, counts = np.unique(rounded_h, return_counts=True)
            # Significant plateaus with at least 5% of tread points or >= 2 points
            sig_mask = counts >= max(2, int(0.05 * len(tread_h)))
            sig_vals = vals[sig_mask]
            if len(sig_vals) >= 3:
                # Merge very close plateaus < 10cm
                merged = [sig_vals[0]]
                for sv in sig_vals[1:]:
                    if sv - merged[-1] >= 0.10:
                        merged.append(sv)
                if len(merged) >= 3:
                    rises = np.diff(merged)
                    if np.all((rises >= 0.10) & (rises <= 0.28)):
                        tread_cents = c_cents[is_tread]
                        tread_cents_h = tread_h
                        step_pos_2d = []
                        for sv in merged:
                            in_plateau = np.abs(tread_cents_h - sv) <= 0.05
                            if np.any(in_plateau):
                                p_pts = tread_cents[in_plateau]
                                p_2d = [p_pts[:, horiz_axes[0]].mean(), p_pts[:, horiz_axes[1]].mean()]
                                step_pos_2d.append(p_2d)
                        if len(step_pos_2d) == len(merged):
                            pos_arr = np.array(step_pos_2d)
                            dists = np.linalg.norm(pos_arr[1:] - pos_arr[:-1], axis=1)
                            tot_run = float(np.sum(dists))
                            tot_rise = float(merged[-1] - merged[0])
                            # Physical circulation flight requirements:
                            # 1. Horizontal run >= 0.80m, total rise >= 0.55m
                            # 2. Individual step tread depth in [0.18, 0.45]m
                            # 3. Maximum pitch angle <= 45 degrees
                            if tot_run >= 0.80 and tot_rise >= 0.55 and np.all((dists >= 0.18) & (dists <= 0.45)):
                                stair_metrics = {
                                    "step_peaks": len(merged),
                                    "mean_step_rise": float(rises.mean()),
                                    "total_run": tot_run,
                                    "total_rise": tot_rise,
                                    "is_stepped": True,
                                }

        neighbors = {}
        for (c_a, c_b), edge_cnt in touching_pairs.items():
            if c_a == cid:
                neighbors[c_b] = edge_cnt
            elif c_b == cid:
                neighbors[c_a] = edge_cnt

        features_by_class[cid] = {
            "class_id": cid,
            "triangle_count": c_count,
            "area_m2": tot_area,
            "frac_up": frac_up,
            "frac_down": frac_down,
            "frac_vert": frac_vert,
            "h_min": c_h_min,
            "h_max": c_h_max,
            "h_span": c_h_span,
            "h_mean": c_h_mean,
            "norm_h_mean": norm_h_mean,
            "norm_h_span": norm_h_span,
            "horiz_extent": horiz_extent,
            "spans_3d": [float(s) for s in spans_3d],
            "pieces_count": num_pieces,
            "largest_piece_area": largest_piece_area,
            "plane_eq": [float(x) for x in plane_eq],
            "inlier_ratio": inlier_ratio,
            "plane_rmse": rmse,
            "stair_metrics": stair_metrics,
            "neighbors": neighbors,
            "pieces": pieces,
        }

    return features_by_class


def classify_structural_classes(
    class_features: Dict[int, Dict[str, Any]],
    scene_bands: Dict[str, float],
    canonical_embs: Dict[int, np.ndarray] | None = None,
    scene_info: Dict[str, Any] | None = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Rule engine classifying each class ID into structural architectural roles:
    - 'floor' (structural floor slabs, with floor_level: 0, 1, ...)
    - 'ceiling' (overhead downward surfaces)
    - 'wall' (massive structural vertical boundaries)
    - 'wall_sottoscala' (under-stair/mezzanine tilted architectural enclosure)
    - 'door' (architectural doorway portals)
    - 'wall_fixture' (posters, boards, wall-mounted planar items)
    - 'furniture_horizontal' (beds, desks, tables)
    - 'furniture_vertical' (wardrobes, cabinets, shelves)
    - 'stairs' (circulation stepped flights)
    - 'object' (props)
    """
    h_min = scene_bands["h_min"]
    h_ceiling_band = scene_bands["h_ceiling_band"]
    if scene_info is None:
        scene_info = {"floors": 1}

    # 1. Identify dominant structural wall candidate (largest vertical area and span)
    vert_candidates = [
        cid for cid, f in class_features.items()
        if f["frac_vert"] >= 0.55 and f["area_m2"] >= 5.0 and f["h_span"] >= 2.0
    ]
    primary_wall_id = max(vert_candidates, key=lambda c: class_features[c]["area_m2"]) if vert_candidates else None

    # 2. Identify dominant structural floor candidates
    floor_candidates = [
        cid for cid, f in class_features.items()
        if f["frac_up"] >= 0.55 and f["area_m2"] >= 2.5 and f["horiz_extent"] >= 2.0
    ]
    floor_candidates.sort(key=lambda c: class_features[c]["area_m2"], reverse=True)
    primary_floor_id = floor_candidates[0] if floor_candidates else None
    floor_base_h = class_features[primary_floor_id]["h_min"] if primary_floor_id else h_min

    classifications = {}

    for cid, f in class_features.items():
        role = "object"
        confidence = 0.70
        details = ""
        level = None

        tot_a = f["area_m2"]
        f_up = f["frac_up"]
        f_dn = f["frac_down"]
        f_vt = f["frac_vert"]
        h_m = f["h_mean"]
        h_sp = f["h_span"]
        h_lo = f["h_min"]
        spans = f["spans_3d"]
        horiz_ext = f["horiz_extent"]
        stair_m = f["stair_metrics"]
        largest_piece = f["largest_piece_area"]

        sim_to_wall = 0.0
        if canonical_embs and primary_wall_id and cid in canonical_embs and primary_wall_id in canonical_embs:
            sim_to_wall = float(np.dot(canonical_embs[cid], canonical_embs[primary_wall_id]))

        # --- A. CEILING & SOTTOSCALA ---
        if f_dn >= 0.40 and h_m >= h_ceiling_band:
            role = "ceiling"
            confidence = min(0.98, 0.70 + 0.30 * f_dn)
            details = f"Overhead downward plane ({f_dn*100:.1f}%) at ceiling band ({h_m:.2f}m >= {h_ceiling_band:.2f}m)"

        # --- A2. ELEVATED HORIZONTAL SHELVES (Top & Bottom Faces) ---
        elif f_up >= 0.20 and f_dn >= 0.20 and tot_a <= 4.0:
            role = "furniture_horizontal"
            confidence = 0.88
            details = f"Elevated horizontal shelving/surface (h_mean={h_m:.2f}m, area={tot_a:.2f}m2)"

        # --- A3. SOTTOSCALA (Tilted under-stair architectural enclosure, downward only) ---
        elif f_dn >= 0.30 and f_up < 0.15 and h_m < h_ceiling_band and tot_a >= 1.0:
            role = "wall_sottoscala"
            confidence = 0.90
            details = f"Tilted architectural under-stair/mezzanine enclosure below ceiling band (area={tot_a:.2f}m2, h={h_m:.2f}m)"

        # --- B. STAIRS (Strict Flight Geometry) ---
        elif stair_m.get("is_stepped", False) and h_sp >= 0.55 and horiz_ext >= 0.80:
            role = "stairs"
            confidence = 0.92
            details = f"Circulation stair flight ({stair_m.get('step_peaks', 0)} steps, span={h_sp:.2f}m, run={stair_m.get('total_run', 0.0):.2f}m)"

        # --- C. STRUCTURAL FLOOR vs ELEVATED HORIZONTAL FURNITURE ---
        elif f_up >= 0.55:
            delta_h_base = h_m - floor_base_h
            is_ground_floor = (cid == primary_floor_id) or (tot_a >= 3.0 and delta_h_base <= 0.25)
            is_upper_floor = (tot_a >= 5.0 and delta_h_base >= 1.80)

            if is_ground_floor or is_upper_floor:
                role = "floor"
                confidence = 0.98
                details = f"Structural floor slab ({tot_a:.2f}m2, extent={horiz_ext:.2f}m)"
            else:
                role = "furniture_horizontal"
                confidence = 0.88
                details = f"Elevated horizontal furniture/surface (h_mean={h_m:.2f}m, elevated {delta_h_base:.2f}m above floor base)"

        # --- D. VERTICAL ELEMENTS: WALL, DOOR, FIXTURE, CABINET ---
        elif f_vt >= 0.55:
            is_door_geom = (1.80 <= h_sp <= 2.50) and (min(spans[0], spans[2]) <= 0.60) and (max(spans[0], spans[2]) <= 1.40) and ((h_lo - floor_base_h) <= 0.25) and (tot_a <= 3.0)
            is_true_wall = (cid == primary_wall_id) or (tot_a >= 8.0 and h_sp >= 2.0 and horiz_ext >= 2.5)
            is_wall_fixture = (canonical_embs is not None and sim_to_wall < 0.50 and tot_a <= 4.0 and (h_lo - floor_base_h) >= 0.30) or (canonical_embs is not None and sim_to_wall < 0.40 and tot_a <= 4.5)
            is_cabinet = tot_a >= 2.0 or (h_sp >= 1.5 and max(spans[0], spans[2]) >= 1.0)

            if is_true_wall:
                role = "wall"
                confidence = 0.98
                details = f"Massive structural wall ({tot_a:.2f}m2, span={h_sp:.2f}m)"
            elif is_door_geom:
                role = "door"
                confidence = 0.95
                details = f"Architectural door portal (H={h_sp:.2f}m, touches floor)"
            elif is_wall_fixture:
                role = "wall_fixture"
                confidence = 0.90
                details = f"Wall-mounted fixture/poster (sim_wall={sim_to_wall:.2f}, area={tot_a:.2f}m2)"
            elif is_cabinet:
                role = "furniture_vertical"
                confidence = 0.85
                details = f"Vertical cabinetry/wardrobe (area={tot_a:.2f}m2, span={h_sp:.2f}m)"
            else:
                role = "furniture_vertical" if tot_a >= 0.5 else "object"
                confidence = 0.80
                details = f"Bounded vertical element (area={tot_a:.2f}m2)"

        # --- E. OBJECT / PROP ---
        else:
            role = "object"
            confidence = 0.75
            details = f"Non-structural prop/object ({tot_a:.2f}m2)"

        # Piece breakdown & sub-component identification
        sub_components = []
        pieces = f.get("pieces", [])
        for p in pieces:
            p_area = p.get("area_m2", 0.0)
            if p_area < 0.35 and (p_area / max(1e-8, tot_a)) < 0.10:
                continue
            p_f_up = p.get("frac_up", 0.0)
            p_f_dn = p.get("frac_down", 0.0)
            p_f_vt = p.get("frac_vert", 0.0)
            p_h = p.get("h_mean", h_m)

            p_role = "object"
            if p_f_dn >= 0.35 and p_h >= h_ceiling_band:
                p_role = "ceiling"
            elif p_f_dn >= 0.30 and p_h < h_ceiling_band and p_area >= 0.5:
                p_role = "wall_sottoscala"
            elif p_f_up >= 0.55:
                p_role = "floor" if (p_area >= 1.5 or (p.get("h_min", p_h) - floor_base_h) <= 0.25) else "furniture_horizontal"
            elif p_f_vt >= 0.60:
                p_role = "wall" if (p_area >= 5.0) else "furniture_vertical"

            sub_components.append({
                "piece_id": p["component_id"],
                "role": p_role,
                "area_m2": round(p_area, 2),
                "h_mean": round(p_h, 2),
                "triangle_count": p["triangle_count"],
            })

        classifications[cid] = {
            "class_id": cid,
            "structural_role": role,
            "confidence": round(confidence, 2),
            "floor_level": level,
            "details": details,
            "piece_breakdown": sub_components,
        }

        # Sub-component flags for sottoscala
        sottoscala_pieces = [sc for sc in sub_components if sc["role"] == "wall_sottoscala"]
        if sottoscala_pieces and role != "wall_sottoscala":
            s_area = sum(sc["area_m2"] for sc in sottoscala_pieces)
            classifications[cid]["sottoscala_subcomponent"] = {
                "total_area_m2": round(s_area, 2),
                "pieces_count": len(sottoscala_pieces),
            }
            classifications[cid]["details"] += f" (includes {s_area:.2f}m2 sottoscala under-stair wall)"

    # Cluster floor elevations into levels
    floor_classes = [cid for cid, c in classifications.items() if c["structural_role"] == "floor"]
    if floor_classes:
        floor_elevations = [(cid, class_features[cid]["h_mean"]) for cid in floor_classes]
        floor_elevations.sort(key=lambda x: x[1])

        current_level = 0
        last_h = floor_elevations[0][1]
        for cid, h in floor_elevations:
            if (h - last_h) >= 1.2:
                current_level += 1
                last_h = h
            classifications[cid]["floor_level"] = current_level
            classifications[cid]["details"] += f" [Level {current_level}]"

        for cid in floor_classes:
            pieces = classifications[cid].get("piece_breakdown", [])
            floor_pieces = [p for p in pieces if p["role"] == "floor" and p["area_m2"] >= 1.0]
            if len(floor_pieces) >= 2:
                heights = [p["h_mean"] for p in floor_pieces]
                if (max(heights) - min(heights)) >= 1.2:
                    levels = [0 if (p["h_mean"] - floor_base_h) <= 0.6 else 1 for p in floor_pieces]
                    classifications[cid]["floor_levels"] = sorted(list(set(levels)))
                    classifications[cid]["details"] += f" (spans Levels {sorted(list(set(levels)))})"

    # Contextual refinement for sottoscala: if classified as wall_sottoscala, verify contiguity
    for cid, c in classifications.items():
        if c["structural_role"] == "wall_sottoscala":
            neighbors = class_features[cid]["neighbors"]
            touches_wall = any(classifications.get(nbr, {}).get("structural_role") in ["wall", "wall_sottoscala", "ceiling"] for nbr in neighbors)
            if touches_wall:
                c["confidence"] = 0.95
                c["details"] += " (contiguous with adjacent wall/ceiling)"

    return classifications


def embed_class_id_badges_on_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    vertex_colors_rgb: np.ndarray,
    instances: Dict[str, Dict[str, Any]],
    tri_centroids: np.ndarray,
    tri_normals: np.ndarray,
    tri_areas: np.ndarray,
    gravity_up: np.ndarray,
    include_instance_idx: bool = False,
    room_center: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Embeds 3D text badges (#<ClassID>) onto each instance directly into the mesh geometry.
    Returns:
      combined_vertices: (N + N_badges, 3) float32
      combined_triangles: (F + F_badges, 3) int32
      combined_vertex_colors: (N + N_badges, 3) uint8
      total_badges_generated: int
    """
    if room_center is None:
        min_b = vertices.min(axis=0)
        max_b = vertices.max(axis=0)
        room_center = (min_b + max_b) / 2.0

    badge_v_list: List[np.ndarray] = []
    badge_f_list: List[np.ndarray] = []
    badge_c_list: List[np.ndarray] = []
    v_offset = len(vertices)
    total_badges = 0

    for inst in instances.values():
        cid = inst["class_id"]
        inst_idx = inst["instance_index"]
        p_tris = inst.get("triangles")
        if p_tris is None or len(p_tris) == 0:
            continue

        text = f"#{cid}.{inst_idx}" if include_instance_idx else f"#{cid}"
        _, col_rgb_float = get_class_color(cid)
        border_color = tuple(int(round(x * 255.0)) for x in col_rgb_float)

        spans = inst.get("spans", [0.5, 0.5, 0.5])
        span_horiz = max(spans[0], spans[2]) if len(spans) >= 3 else 0.5
        tot_a = inst.get("area_m2", 0.1)

        scale = float(np.clip(0.35 * max(span_horiz, 0.10), 0.035, 0.12))
        if tot_a >= 5.0:
            scale = 0.13

        cents = tri_centroids[p_tris]
        norms = tri_normals[p_tris]
        p_up = np.dot(norms, gravity_up)
        role = inst.get("assigned_role", "")
        c_inst = np.array(inst["mean_centroid"])

        # Special multi-facade labeling for large continuous wall enclosing the room
        if cid == 5 and len(p_tris) > 100000:
            cardinal_configs = [
                ("North", cents[:, 2] <= -3.2, np.array([0.0, 0.0, 1.0])),
                ("South", cents[:, 2] >= -0.8, np.array([0.0, 0.0, -1.0])),
                ("West", cents[:, 0] <= -2.6, np.array([1.0, 0.0, 0.0])),
                ("East", cents[:, 0] >= 0.5, np.array([-1.0, 0.0, 0.0])),
            ]
            for _, mask, f_dir in cardinal_configs:
                idxs = np.flatnonzero(mask)
                if len(idxs) == 0:
                    continue
                sub_cents = cents[idxs]
                best_sub = idxs[np.argmin(np.abs(sub_cents[:, 1] - 0.8))]
                anchor = cents[best_sub] + f_dir * (scale * 0.25 + 0.02)
                bv, bf, bc = build_instance_badge_mesh(
                    origin=anchor,
                    forward=f_dir,
                    up=gravity_up,
                    scale=scale,
                    text=text,
                    border_color_rgb=border_color,
                )
                if len(bv) > 0:
                    badge_v_list.append(bv)
                    badge_f_list.append(bf + v_offset)
                    badge_c_list.append(bc)
                    v_offset += len(bv)
                    total_badges += 1
            continue

        # Standard instances
        is_floor = (role.startswith("floor") or p_up.mean() > 0.65)
        is_ceiling = (role == "ceiling" or p_up.mean() < -0.65)

        if is_floor:
            horiz_dist = np.hypot(cents[:, 0] - room_center[0], cents[:, 2] - room_center[2])
            best_idx = int(np.argmin(horiz_dist))
            anchor_pt = cents[best_idx]
            to_center = room_center - anchor_pt
            horiz_to_center = np.array([to_center[0], 0.0, to_center[2]])
            h_len = np.linalg.norm(horiz_to_center)
            forward = horiz_to_center / h_len if h_len > 0.1 else np.array([0.0, 0.0, 1.0])
            anchor = anchor_pt + np.array([0.0, scale * 0.65 + 0.015, 0.0])
        elif is_ceiling:
            dist_c = np.linalg.norm(cents - c_inst, axis=1)
            best_idx = int(np.argmin(dist_c))
            anchor = cents[best_idx] - np.array([0.0, 0.03, 0.0])
            forward = np.array([0.0, -1.0, 0.0])
        else:
            to_center = room_center - cents
            dist_center = np.linalg.norm(to_center, axis=1, keepdims=True)
            dir_center = to_center / np.maximum(dist_center, 1e-6)
            facing_center = np.sum(norms * dir_center, axis=1)

            target_y = inst["bounds_min"][1] + 0.65 * spans[1] if len(spans) >= 2 else c_inst[1]
            dist_y = np.abs(cents[:, 1] - target_y)
            score = facing_center * 2.0 - (dist_y / max(spans[1] if len(spans) >= 2 else 0.5, 0.1))

            if np.any(facing_center > 0):
                valid_idxs = np.flatnonzero(facing_center > 0)
                best_idx = int(valid_idxs[np.argmax(score[valid_idxs])])
            else:
                dist_inst = np.linalg.norm(cents - c_inst, axis=1)
                best_idx = int(np.argmin(dist_inst))

            anchor_pt = cents[best_idx]
            normal_dir = norms[best_idx]
            horiz_norm = np.array([normal_dir[0], 0.0, normal_dir[2]])
            h_len = np.linalg.norm(horiz_norm)
            forward = horiz_norm / h_len if h_len > 0.1 else normal_dir
            hover_offset = forward * (scale * 0.25 + 0.015)
            anchor = anchor_pt + hover_offset

        bv, bf, bc = build_instance_badge_mesh(
            origin=anchor,
            forward=forward,
            up=gravity_up,
            scale=scale,
            text=text,
            border_color_rgb=border_color,
        )
        if len(bv) > 0:
            badge_v_list.append(bv)
            badge_f_list.append(bf + v_offset)
            badge_c_list.append(bc)
            v_offset += len(bv)
            total_badges += 1

    if len(badge_v_list) > 0:
        comb_verts = np.vstack([vertices] + badge_v_list)
        comb_faces = np.vstack([triangles] + badge_f_list)
        comb_colors = np.vstack([vertex_colors_rgb] + badge_c_list)
    else:
        comb_verts = vertices.copy()
        comb_faces = triangles.copy()
        comb_colors = vertex_colors_rgb.copy()

    return comb_verts, comb_faces, comb_colors, total_badges


def export_structural_manifest(
    output_dir: Path,
    classifications: Dict[int, Dict[str, Any]],
    features: Dict[int, Dict[str, Any]],
    scene_bands: Dict[str, float],
    instances: Dict[str, Dict[str, Any]] | None = None,
    class_instance_counts: Dict[int, int] | None = None,
) -> None:
    """
    Exports JSON, CSV, and Markdown summaries of structural classifications and instances.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "structural_classes.json"
    manifest_data = {
        "scene_bands": scene_bands,
        "structural_summary": {
            "num_classes_evaluated": len(classifications),
            "floors": [cid for cid, c in classifications.items() if c["structural_role"] == "floor"],
            "ceilings": [cid for cid, c in classifications.items() if c["structural_role"] == "ceiling"],
            "walls": [cid for cid, c in classifications.items() if c["structural_role"] in ["wall", "wall_sottoscala"]],
            "sottoscala_subcomponents": [cid for cid, c in classifications.items() if "sottoscala_subcomponent" in c],
            "doors": [cid for cid, c in classifications.items() if c["structural_role"] == "door"],
            "wall_fixtures": [cid for cid, c in classifications.items() if c["structural_role"] == "wall_fixture"],
            "stairs": [cid for cid, c in classifications.items() if c["structural_role"] == "stairs"],
            "furniture_horizontal": [cid for cid, c in classifications.items() if c["structural_role"] == "furniture_horizontal"],
            "furniture_vertical": [cid for cid, c in classifications.items() if c["structural_role"] == "furniture_vertical"],
            "objects": [cid for cid, c in classifications.items() if c["structural_role"] == "object"],
            "total_instances_extracted": len(instances) if instances else 0,
        },
        "class_instance_counts": class_instance_counts or {},
        "classifications": classifications,
    }
    with open(json_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Exported JSON manifest: {json_path}")

    csv_path = output_dir / "structural_classes_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Class ID", "Structural Role", "Floor Level", "Confidence",
            "Area (m2)", "Triangle Count", "Instances Count", "Frac Up", "Frac Down", "Frac Vert",
            "H Mean (m)", "H Span (m)", "Largest Piece (m2)", "Details"
        ])
        for cid in sorted(classifications.keys()):
            c = classifications[cid]
            feat = features[cid]
            inst_cnt = (class_instance_counts or {}).get(cid, 0)
            writer.writerow([
                cid,
                c["structural_role"],
                c["floor_level"] if c["floor_level"] is not None else "",
                f"{c['confidence']:.2f}",
                f"{feat['area_m2']:.2f}",
                feat["triangle_count"],
                inst_cnt,
                f"{feat['frac_up']:.2f}",
                f"{feat['frac_down']:.2f}",
                f"{feat['frac_vert']:.2f}",
                f"{feat['h_mean']:.2f}",
                f"{feat['h_span']:.2f}",
                f"{feat['largest_piece_area']:.2f}",
                c["details"],
            ])
    print(f"Exported CSV summary: {csv_path}")

    md_path = output_dir / "structural_classes_summary.md"
    md_lines = [
        "# Structural Architectural Classifications\n",
        f"- **Scene Elevation Span**: {scene_bands['h_scene']:.2f}m (from {scene_bands['h_min']:.2f}m to {scene_bands['h_max']:.2f}m)",
        f"- **Ceiling Level Band**: $\\ge {scene_bands['h_ceiling_band']:.2f}$m\n",
        "## Summary by Structural Category\n",
        f"- **Floors**: {manifest_data['structural_summary']['floors']}",
        f"- **Ceilings**: {manifest_data['structural_summary']['ceilings']}",
        f"- **Walls**: {manifest_data['structural_summary']['walls']}",
        f"- **Doors**: {manifest_data['structural_summary']['doors']}",
        f"- **Wall Fixtures / Posters**: {manifest_data['structural_summary']['wall_fixtures']}",
        f"- **Sottoscala Sub-components**: {manifest_data['structural_summary']['sottoscala_subcomponents']}",
        f"- **Stairs**: {manifest_data['structural_summary']['stairs']}",
        f"- **Furniture (Horizontal)**: {manifest_data['structural_summary']['furniture_horizontal']}",
        f"- **Furniture (Vertical)**: {manifest_data['structural_summary']['furniture_vertical']}",
        f"- **Objects / Props**: {manifest_data['structural_summary']['objects']}\n",
        "## Detailed Class Table\n",
        "| Class ID | Role | Conf | Area (m²) | Instances | Frac Up | Frac Down | Frac Vert | H Mean (m) | Details |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for cid in sorted(classifications.keys()):
        c = classifications[cid]
        feat = features[cid]
        inst_cnt = (class_instance_counts or {}).get(cid, 0)
        md_lines.append(
            f"| **#{cid}** | `{c['structural_role']}` | {c['confidence']:.2f} | "
            f"{feat['area_m2']:.2f} | {inst_cnt} | {feat['frac_up']:.2f} | {feat['frac_down']:.2f} | "
            f"{feat['frac_vert']:.2f} | {feat['h_mean']:.2f} | {c['details']} |"
        )
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Exported Markdown report: {md_path}")

    # Export instances table if instances are provided
    if instances:
        inst_csv_path = output_dir / "instances_summary.csv"
        with open(inst_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Instance ID", "Class ID", "Assigned Role", "Triangle Count", "Area (m2)",
                "Min X", "Min Y", "Min Z", "Max X", "Max Y", "Max Z",
                "Span X", "Span Y", "Span Z", "Centroid X", "Centroid Y", "Centroid Z",
                "Frac Up", "Frac Down", "Frac Vert", "H Mean (m)", "H Span (m)"
            ])
            for iid, inst in sorted(instances.items(), key=lambda x: (x[1]["class_id"], -x[1]["triangle_count"])):
                writer.writerow([
                    inst["instance_id"],
                    inst["class_id"],
                    inst["assigned_role"],
                    inst["triangle_count"],
                    inst["area_m2"],
                    inst["bounds_min"][0], inst["bounds_min"][1], inst["bounds_min"][2],
                    inst["bounds_max"][0], inst["bounds_max"][1], inst["bounds_max"][2],
                    inst["spans"][0], inst["spans"][1], inst["spans"][2],
                    inst["mean_centroid"][0], inst["mean_centroid"][1], inst["mean_centroid"][2],
                    inst["frac_up"], inst["frac_down"], inst["frac_vert"],
                    inst["h_mean"], inst["h_span"],
                ])
        print(f"Exported instances CSV: {inst_csv_path}")

        inst_md_path = output_dir / "instances_summary.md"
        inst_lines = [
            "# Mesh Instance Segmentation Summary\n",
            f"- **Total Active Classes**: {len(classifications)}",
            f"- **Total Filtered Instances (>= 15 triangles)**: {len(instances)}\n",
            "## Instance Counts by Class\n",
            "| Class ID | Role | Instance Count | Total Area (m²) | Largest Instance Area (m²) |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for cid in sorted(classifications.keys()):
            c_insts = [inst for inst in instances.values() if inst["class_id"] == cid]
            max_inst_a = max([inst["area_m2"] for inst in c_insts]) if c_insts else 0.0
            tot_a = features[cid]["area_m2"]
            role = classifications[cid]["structural_role"]
            inst_lines.append(f"| **#{cid}** | `{role}` | {len(c_insts)} | {tot_a:.2f} | {max_inst_a:.2f} |")

        inst_lines.append("\n## Top Instances (Area >= 0.20 m²)\n")
        inst_lines.append("| Instance ID | Class ID | Role | Triangles | Area (m²) | Spans [DX, DY, DZ] (m) | H Mean (m) |")
        inst_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        top_insts = [inst for inst in instances.values() if inst["area_m2"] >= 0.20]
        top_insts.sort(key=lambda x: -x["area_m2"])
        for inst in top_insts[:100]:
            spans_str = f"[{inst['spans'][0]:.2f}, {inst['spans'][1]:.2f}, {inst['spans'][2]:.2f}]"
            inst_lines.append(
                f"| `{inst['instance_id']}` | **#{inst['class_id']}** | `{inst['assigned_role']}` | "
                f"{inst['triangle_count']} | {inst['area_m2']:.2f} | {spans_str} | {inst['h_mean']:.2f} |"
            )

        with open(inst_md_path, "w") as f:
            f.write("\n".join(inst_lines) + "\n")
        print(f"Exported instances Markdown report: {inst_md_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect structural classes and instances from 3D mesh")
    parser.add_argument("--mesh-path", type=Path, required=True, help="Path to resolved colored mesh PLY")
    parser.add_argument("--stats-json", type=Path, required=True, help="Path to projection_stats.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for manifests")
    parser.add_argument("--embeddings-path", type=Path, default=None, help="Path to canonical_class_embeddings.npz")
    parser.add_argument("--scene-size-path", type=Path, default=None, help="Path to scene_size.txt")
    parser.add_argument("--cleaned-mesh-path", type=Path, default=None, help="Path to save cleaned mesh PLY")
    parser.add_argument("--refined-mesh-path", type=Path, default=None, help="Path to save refined mesh PLY with outliers absorbed")
    parser.add_argument("--min-island-size", type=int, default=30, help="Min triangle island size for floater absorption")
    parser.add_argument("--min-instance-size", type=int, default=15, help="Min triangle count for instance segmentation")
    parser.add_argument("--embed-labels", action="store_true", default=True, help="Embed 3D class ID text badges onto mesh instances")
    parser.add_argument("--no-embed-labels", action="store_false", dest="embed_labels", help="Disable embedding 3D class ID badges")
    parser.add_argument("--labeled-mesh-path", type=Path, default=None, help="Path to save labeled mesh PLY with badges")
    parser.add_argument("--include-instance-idx", action="store_true", default=False, help="Include instance index in badge text e.g. #4.0")
    args = parser.parse_args()

    print("=================================================================")
    print("STEP 3: GEOMETRY-DRIVEN STRUCTURAL CLASS DETECTION & INSTANCE SEGMENTATION")
    print(f"Input Mesh:        {args.mesh_path}")
    print(f"Stats JSON:        {args.stats_json}")
    print(f"Output Dir:        {args.output_dir}")
    print("=================================================================")

    t0 = time.perf_counter()
    timings: Dict[str, float] = {}
    mesh = o3d.io.read_triangle_mesh(str(args.mesh_path))
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    v_colors = (np.asarray(mesh.vertex_colors) * 255.0).round().astype(np.uint8)

    with open(args.stats_json) as f:
        stats = json.load(f)
    active_classes = sorted(stats["active_classes"])
    print(f"Loaded mesh: {len(vertices):,} vertices, {len(triangles):,} triangles ({time.perf_counter() - t0:.2f}s)")
    print(f"Active classes to classify: {len(active_classes)} classes: {active_classes}")

    tri_c = (v_colors[triangles[:, 0]].astype(np.float32) + v_colors[triangles[:, 1]].astype(np.float32) + v_colors[triangles[:, 2]].astype(np.float32)) / 3.0
    palette = np.array([get_class_color(c)[1] for c in active_classes], dtype=np.float32) * 255.0
    gray = np.array([128.0, 128.0, 128.0], dtype=np.float32)

    d_palette = np.linalg.norm(tri_c[:, None, :] - palette[None, :, :], axis=2)
    min_d_pal = d_palette.min(axis=1)
    best_c_idx = d_palette.argmin(axis=1)
    d_gray = np.linalg.norm(tri_c - gray, axis=1)

    triangle_labels = np.full(len(triangles), -1, dtype=np.int32)
    assigned_mask = (min_d_pal < d_gray) & (min_d_pal < 60.0)
    for c_idx, cid in enumerate(active_classes):
        m = assigned_mask & (best_c_idx == c_idx)
        triangle_labels[m] = cid
    print(f"Reconstructed triangle labels: {np.count_nonzero(triangle_labels >= 0):,} assigned ({np.count_nonzero(triangle_labels < 0):,} unvoted/neutral)")
    timings["input_load_and_label_decode_seconds"] = time.perf_counter() - t0

    print("\n--- Phase 1: Dual Graph, Enhanced Floater Absorption & 100% Region Growing ---")
    t0_graph = time.perf_counter()
    adj_u, adj_v = build_mesh_dual_graph(triangles)
    timings["dual_graph_seconds"] = time.perf_counter() - t0_graph
    print(f"  Dual mesh graph built: {len(adj_u):,} directed edges ({timings['dual_graph_seconds']:.2f}s)")

    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    tri_areas = 0.5 * np.linalg.norm(cross, axis=1)
    tri_normals = cross / np.maximum(2.0 * tri_areas[:, None], 1e-8)
    tri_centroids = (v0 + v1 + v2) / 3.0

    # Step 1: Structural floater absorption (wall/floor/ceiling bleeds < 5000 tris and < 0.25 m2)
    t0_pass = time.perf_counter()
    cleaned_labels, n_struct_abs = absorb_structural_floaters(
        triangle_labels=triangle_labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_areas=tri_areas,
        structural_classes=[5, 14, 1],
        max_island_tris=5000,
        max_island_area=0.25,
    )
    timings["structural_floater_absorption_seconds"] = time.perf_counter() - t0_pass
    print(f"  Pass 1: Absorbed {n_struct_abs:,} structural floater triangles across Wall, Floor, Ceiling.")

    # Step 2: General floater absorption (islands < 150 tris)
    t0_pass = time.perf_counter()
    cleaned_labels, n_gen_abs = absorb_outlier_floaters(
        triangle_labels=cleaned_labels,
        adj_u=adj_u,
        adj_v=adj_v,
        active_classes=active_classes,
        min_island_size=150,
    )
    timings["general_floater_absorption_seconds"] = time.perf_counter() - t0_pass
    print(f"  Pass 2: Absorbed {n_gen_abs:,} general floater triangles (< 150 tris).")

    # Step 3: Multi-source BFS region growing to fill 100% of unassigned triangles
    initial_unassigned = int(np.count_nonzero(cleaned_labels < 0))
    print(f"  Pass 3: Propagating labels to fill {initial_unassigned:,} unassigned triangles via dual graph...")
    t0_pass = time.perf_counter()
    cleaned_labels = fill_unassigned_triangles(
        triangle_labels=cleaned_labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_centroids=tri_centroids,
    )
    timings["unknown_face_infill_seconds"] = time.perf_counter() - t0_pass
    print(f"  Pass 3: 100% triangle coverage achieved (0 unassigned triangles remaining).")

    # Step 4: Post-filling boundary cleanup on newly filled islands (< 300 tris)
    t0_pass = time.perf_counter()
    cleaned_labels, n_post_abs = absorb_outlier_floaters(
        triangle_labels=cleaned_labels,
        adj_u=adj_u,
        adj_v=adj_v,
        active_classes=active_classes,
        min_island_size=300,
    )
    print(f"  Pass 4: Absorbed {n_post_abs:,} boundary floaters post-filling (< 300 tris).")

    # Step 5: Final micro-island cleanup (< 50 tris)
    cleaned_labels, n_micro_abs = absorb_outlier_floaters(
        triangle_labels=cleaned_labels,
        adj_u=adj_u,
        adj_v=adj_v,
        active_classes=active_classes,
        min_island_size=50,
    )
    timings["post_infill_cleanup_seconds"] = time.perf_counter() - t0_pass
    print(f"  Pass 5: Absorbed {n_micro_abs:,} residual micro-islands (< 50 tris).")

    print("\n--- Phase 2: Gravity, Elevation Bands & Priors ---")
    t0_phase = time.perf_counter()
    gravity_up, scene_bands = estimate_gravity_and_scene_bands(vertices, triangles)
    timings["gravity_and_scene_bands_seconds"] = time.perf_counter() - t0_phase
    print(f"  Gravity Up-Vector:   {gravity_up} (axis {scene_bands['up_axis_index']})")
    print(f"  Elevation Bounds:    [{scene_bands['h_min']:.2f}m, {scene_bands['h_max']:.2f}m] (span {scene_bands['h_scene']:.2f}m)")
    print(f"  Ceiling Level Band:  >= {scene_bands['h_ceiling_band']:.2f}m")

    # Step 6: Class-relative outlier instance absorption
    print("\n--- Phase 2b: Class-Relative Outlier Instance Absorption ---")
    t0_outliers = time.perf_counter()
    cleaned_labels, reassigned_records = converge_class_instances_iteratively(
        triangle_labels=cleaned_labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_areas=tri_areas,
        tri_centroids=tri_centroids,
        tri_normals=tri_normals,
        gravity_up=gravity_up,
        active_classes=active_classes,
        min_instance_size=max(args.min_instance_size, 25),
        max_proximity_merge_dist=0.08,
        max_iterations=10,
    )
    timings["iterative_instance_refinement_seconds"] = time.perf_counter() - t0_outliers
    n_boundary = sum(1 for r in reassigned_records if r["method"] == "boundary_enclosing")
    n_bridge = sum(1 for r in reassigned_records if r["method"] == "proximity_gap_bridge")
    total_reassigned_tris = sum(r["triangle_count"] for r in reassigned_records)
    print(f"  Converged: Absorbed {len(reassigned_records)} outlier instances ({total_reassigned_tris:,} triangles) across iterations in {time.perf_counter() - t0_outliers:.2f}s.")
    print(f"  -> {n_boundary} enclosed outliers reassigned to surrounding boundary class, {n_bridge} small-gap components bridged within 8cm.")

    # Step 7: Final post-outlier micro-island cleanup (< 30 tris)
    t0_pass = time.perf_counter()
    cleaned_labels, n_post_outlier = absorb_outlier_floaters(
        triangle_labels=cleaned_labels,
        adj_u=adj_u,
        adj_v=adj_v,
        active_classes=active_classes,
        min_island_size=30,
    )
    if n_post_outlier > 0:
        print(f"  Pass 7: Absorbed {n_post_outlier:,} residual micro-island triangles post-outlier refinement.")
    timings["final_micro_island_cleanup_seconds"] = time.perf_counter() - t0_pass

    # Save cleaned and refined mesh PLYs with 100% coverage
    cleaned_ply_path = args.cleaned_mesh_path
    if cleaned_ply_path is None:
        cleaned_ply_path = args.mesh_path.parent / "mesh_projected_cleaned.ply"

    refined_ply_path = args.refined_mesh_path
    if refined_ply_path is None:
        refined_ply_path = (
            cleaned_ply_path.parent / "mesh_projected_refined.ply"
            if cleaned_ply_path
            else args.mesh_path.parent / "mesh_projected_refined.ply"
        )

    # Compute cleaned triangle colors
    tri_colors_rgb = np.full((len(triangles), 3), 128, dtype=np.uint8)
    for cid in active_classes:
        m = cleaned_labels == cid
        if np.any(m):
            _, col_rgb = get_class_color(cid)
            tri_colors_rgb[m] = (np.array(col_rgb) * 255.0).round().astype(np.uint8)

    t0_pass = time.perf_counter()
    export_colored_ply(cleaned_ply_path, vertices, triangles, tri_colors_rgb)
    if refined_ply_path != cleaned_ply_path:
        export_colored_ply(refined_ply_path, vertices, triangles, tri_colors_rgb)
    timings["cleaned_mesh_export_seconds"] = time.perf_counter() - t0_pass

    # Load scene_size and canonical embeddings if available
    scene_size_file = args.scene_size_path
    if scene_size_file is None:
        candidate_sz = args.mesh_path.parents[1] / "01_poses_refinment/scene_size.txt"
        if candidate_sz.is_file():
            scene_size_file = candidate_sz
    scene_info = read_scene_sizes(scene_size_file)
    print(f"  Scene metadata:      Floors={scene_info.get('floors')}, FloorArea={scene_info.get('total_floor_area_m2')}m2")

    emb_file = args.embeddings_path
    if emb_file is None:
        candidate_emb = args.mesh_path.parents[1] / "canonical_class_embeddings.npz"
        if candidate_emb.is_file():
            emb_file = candidate_emb
    canonical_embs = load_canonical_dino_embeddings(emb_file, active_classes)
    if canonical_embs:
        print(f"  Loaded canonical DINOv2 embeddings: {len(canonical_embs)} active classes")

    print("\n--- Phase 3: Extracting Geometric Features ---")
    t0_feat = time.perf_counter()
    features = analyze_class_geometry(
        vertices=vertices,
        triangles=triangles,
        triangle_labels=cleaned_labels,
        class_ids=active_classes,
        gravity_up=gravity_up,
        scene_bands=scene_bands,
    )
    timings["geometric_feature_extraction_seconds"] = time.perf_counter() - t0_feat
    print(f"  Geometric features extracted in {timings['geometric_feature_extraction_seconds']:.2f}s for {len(features)} classes.")

    print("\n--- Phase 4: Rule-Based Structural Classification ---")
    t0_phase = time.perf_counter()
    classifications = classify_structural_classes(
        class_features=features,
        scene_bands=scene_bands,
        canonical_embs=canonical_embs,
        scene_info=scene_info,
    )
    timings["structural_classification_seconds"] = time.perf_counter() - t0_phase

    print("\n--- Phase 5: Instance Segmentation Across All Classes ---")
    t0_inst = time.perf_counter()
    instances, class_inst_counts = extract_instances(
        triangle_labels=cleaned_labels,
        active_classes=active_classes,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_areas=tri_areas,
        tri_centroids=tri_centroids,
        tri_normals=tri_normals,
        gravity_up=gravity_up,
        min_instance_size=args.min_instance_size,
    )
    timings["instance_extraction_seconds"] = time.perf_counter() - t0_inst

    # Assign role to each instance
    for inst in instances.values():
        cid = inst["class_id"]
        c_role = classifications.get(cid, {}).get("structural_role", "object")
        # Specialize if sottoscala sub-component
        if c_role == "ceiling" and inst["h_mean"] < scene_bands["h_ceiling_band"] and inst["frac_down"] >= 0.30:
            inst["assigned_role"] = "wall_sottoscala"
        elif c_role == "floor":
            inst["assigned_role"] = f"floor_level_{0 if (inst['h_mean'] - scene_bands['h_min']) <= 0.6 else 1}"
        else:
            inst["assigned_role"] = c_role

    print(f"  Extracted {len(instances)} instances across {len(active_classes)} classes in {time.perf_counter() - t0_inst:.2f}s.")

    print("\nStructural Classification Summary:")
    print(f"{'CID':>4} | {'Role':<22} | {'Conf':>5} | {'Area m2':>7} | {'Inst':>4} | {'Frac Up':>7} | {'Frac Dn':>7} | {'Frac Vt':>7} | {'H Mean':>6}")
    print("-" * 92)
    for cid in sorted(classifications.keys()):
        c = classifications[cid]
        f = features[cid]
        inst_cnt = class_inst_counts.get(cid, 0)
        print(f"{cid:4d} | {c['structural_role']:<22} | {c['confidence']:5.2f} | {f['area_m2']:7.2f} | {inst_cnt:4d} | {f['frac_up']:7.2f} | {f['frac_down']:7.2f} | {f['frac_vert']:7.2f} | {f['h_mean']:6.2f}")

    print("\n--- Phase 6: Exporting Manifests ---")
    t0_phase = time.perf_counter()
    export_structural_manifest(
        output_dir=args.output_dir,
        classifications=classifications,
        features=features,
        scene_bands=scene_bands,
        instances=instances,
        class_instance_counts=class_inst_counts,
    )
    timings["manifest_export_seconds"] = time.perf_counter() - t0_phase

    if args.embed_labels:
        print("\n--- Phase 7: Embedding 3D Class ID Badges on Instances ---")
        t0_label = time.perf_counter()
        labeled_ply_path = args.labeled_mesh_path
        if labeled_ply_path is None:
            labeled_ply_path = (
                refined_ply_path.parent / "mesh_projected_labeled.ply"
                if refined_ply_path
                else args.mesh_path.parent / "mesh_projected_labeled.ply"
            )

        # Compute vertex colors for base mesh
        v_colors = np.zeros((len(vertices), 3), dtype=np.float64)
        v_counts = np.zeros((len(vertices), 1), dtype=np.float64)
        for i in range(3):
            np.add.at(v_colors, triangles[:, i], tri_colors_rgb)
            np.add.at(v_counts, triangles[:, i], 1.0)
        v_counts = np.maximum(1.0, v_counts)
        base_v_colors = (v_colors / v_counts).round().astype(np.uint8)

        comb_v, comb_f, comb_c, n_badges = embed_class_id_badges_on_mesh(
            vertices=vertices,
            triangles=triangles,
            vertex_colors_rgb=base_v_colors,
            instances=instances,
            tri_centroids=tri_centroids,
            tri_normals=tri_normals,
            tri_areas=tri_areas,
            gravity_up=gravity_up,
            include_instance_idx=args.include_instance_idx,
        )
        export_colored_ply(labeled_ply_path, comb_v, comb_f, vertex_colors_rgb=comb_c)
        print(f"  Successfully embedded {n_badges} class ID badges into {labeled_ply_path} ({len(comb_v):,} vertices, {len(comb_f):,} triangles) in {time.perf_counter() - t0_label:.2f}s.")
        timings["instance_badge_export_seconds"] = time.perf_counter() - t0_label

    timings["total_seconds"] = time.perf_counter() - t0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pipeline_timings.json").write_text(json.dumps(timings, indent=2) + "\n")
    print(f"\nAll operations successfully completed in {timings['total_seconds']:.2f}s!")


if __name__ == "__main__":
    main()
