#!/usr/bin/env python3
"""
Step 2: 2D Mask Projection onto 3D Mesh with COLMAP Support Gating & Contention Analysis.

Projects 2D panoptic segmentation masks across all camera viewpoints onto the cleaned 3D mesh.
- Uses COLMAP poses only when frames have >= colmap_threshold triangulated points.
- Skips frames with insufficient COLMAP triangulated points.
- Colors triangles:
  * Pure White [255, 255, 255] if contended between > 1 classes.
  * Class Color if uncontended (voted by exactly 1 class).
  * Neutral Gray [128, 128, 128] if unvoted.
- Outputs:
  * mesh_projected_contention.ply: Colored 3D mesh.
  * contention_matrix.pdf: Ultra-high-resolution vector PDF of the cross-class contention matrix.
  * contention_matrix.png: High-res raster matrix.
  * class_contention_summary.csv: Table of assigned and contended triangles per class.
  * projection_stats.json: Execution metrics and timing breakdown.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import open3d as o3d
import torch

backend_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_dir / "03_FastGS_DNSplatter"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene_inputs

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


def load_colmap_triangulated_points(images_txt_path: Path) -> Dict[str, int]:
    """Compatibility alias for the strict shared COLMAP parser."""
    return scene_inputs.colmap_point_counts(images_txt_path)


def normalized_incidence_cosines(hit_normals: np.ndarray, ray_directions: np.ndarray) -> np.ndarray:
    """Return finite front-facing incidence cosines independent of ray length."""
    normals = np.asarray(hit_normals, dtype=np.float32)
    rays = np.asarray(ray_directions, dtype=np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        normal_length = np.linalg.norm(normals, axis=-1)
        ray_length = np.linalg.norm(rays, axis=-1)
        denominator = normal_length * ray_length
        dot = -np.sum(normals * rays, axis=-1)
    cosine = np.zeros_like(denominator, dtype=np.float32)
    valid = np.isfinite(denominator) & (denominator > 1e-8)
    cosine[valid] = dot[valid] / denominator[valid]
    return np.clip(np.nan_to_num(cosine, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)


def reduce_view_face_votes(
    triangle_ids: np.ndarray,
    class_indices: np.ndarray,
    quality_weights: np.ndarray,
    num_triangles: int,
    num_classes: int,
    all_face_hit_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum per-class quality, normalized by all valid pixels on each face in one view."""
    triangle_ids = np.asarray(triangle_ids, dtype=np.int64)
    class_indices = np.asarray(class_indices, dtype=np.int64)
    quality_weights = np.asarray(quality_weights, dtype=np.float32)
    if num_triangles < 0 or num_classes < 1:
        raise ValueError("num_triangles must be nonnegative and num_classes must be positive")
    if not (triangle_ids.ndim == class_indices.ndim == quality_weights.ndim == 1):
        raise ValueError("per-pixel vote inputs must be one-dimensional")
    if not (len(triangle_ids) == len(class_indices) == len(quality_weights)):
        raise ValueError("per-pixel vote inputs must have equal length")
    if len(triangle_ids) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    if (triangle_ids < 0).any() or (triangle_ids >= num_triangles).any():
        raise ValueError("triangle ids are outside the mesh")
    if (class_indices < 0).any() or (class_indices >= num_classes).any():
        raise ValueError("class indices are outside the active class set")

    if all_face_hit_ids is None:
        all_face_hit_ids = triangle_ids
    else:
        all_face_hit_ids = np.asarray(all_face_hit_ids, dtype=np.int64)
        if all_face_hit_ids.ndim != 1 or (all_face_hit_ids < 0).any() or (all_face_hit_ids >= num_triangles).any():
            raise ValueError("all face hit ids are outside the mesh")
    hits_per_face = np.bincount(all_face_hit_ids, minlength=num_triangles).astype(np.float32)
    flat = triangle_ids * num_classes + class_indices
    unique_flat, inverse = np.unique(flat, return_inverse=True)
    class_quality = np.zeros(len(unique_flat), dtype=np.float32)
    np.add.at(class_quality, inverse, quality_weights)
    return unique_flat, class_quality / hits_per_face[unique_flat // num_classes]


def local_evidence_labels(face_class_votes: np.ndarray) -> np.ndarray:
    """Choose each face's highest locally accumulated class, or -1 without evidence."""
    votes = np.asarray(face_class_votes)
    if votes.ndim != 2:
        raise ValueError("face-class votes must be two-dimensional")
    labels = np.full(votes.shape[0], -1, dtype=np.int64)
    if votes.shape[1]:
        voted = np.any(votes > 0, axis=1)
        labels[voted] = votes[voted].argmax(axis=1)
    return labels


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_mask_distance_weights(labels: np.ndarray) -> np.ndarray:
    """Computes fast boundary distance weights for a 2D label map."""
    H, W = labels.shape
    labels_pad = np.pad(labels, 1, mode="edge")
    diff_r = labels != labels_pad[1:-1, 2:]
    diff_d = labels != labels_pad[2:, 1:-1]
    boundary = (diff_r | diff_d).astype(np.uint8)
    dist = cv2.distanceTransform(1 - boundary, cv2.DIST_L2, 3)
    w_edge = np.clip(dist / 12.0, 0.1, 1.0).astype(np.float32)
    w_edge[labels == 0] = 0.0
    return w_edge


def build_dual_mesh_graph(triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs the undirected dual graph of the mesh where nodes are triangles
    and edges connect adjacent triangles sharing a 3D vertex-vertex edge.
    """
    num_tri = len(triangles)
    e0 = np.sort(triangles[:, [0, 1]], axis=1)
    e1 = np.sort(triangles[:, [1, 2]], axis=1)
    e2 = np.sort(triangles[:, [2, 0]], axis=1)
    all_edges = np.vstack([e0, e1, e2])
    # np.tile maps the first num_tri rows (e0) to 0..N-1, the second (e1) to 0..N-1, and third (e2) to 0..N-1.
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


def absorb_isolated_islands(
    triangle_labels: np.ndarray,
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    num_classes: int,
    min_island_size: int = 15,
) -> Tuple[np.ndarray, int, int]:
    """
    Detects isolated single triangles and small island clusters (< min_island_size)
    of a class surrounded by another class, and absorbs them into the surrounding majority class.
    Operates at connected component level so both perimeter and interior triangles of the island
    are cleanly absorbed into the surrounding majority class.
    """
    from scipy.sparse import csgraph, csr_matrix

    labels_cleaned = triangle_labels.copy()
    num_tri = len(labels_cleaned)

    # 1. Identify same-label edges
    valid_edge = (labels_cleaned[adj_u] >= 0) & (labels_cleaned[adj_v] >= 0)
    same_lbl = valid_edge & (labels_cleaned[adj_u] == labels_cleaned[adj_v])
    u_same = adj_u[same_lbl]
    v_same = adj_v[same_lbl]

    # Degree in same-label subgraph
    deg_same = np.zeros(num_tri, dtype=np.int32)
    np.add.at(deg_same, u_same, 1)

    # Triangles with 0 neighbors of the same class are completely isolated
    isolated_single = (deg_same == 0) & (labels_cleaned >= 0)

    # Connected components on same-label subgraph
    adj_mat = csr_matrix((np.ones(len(u_same), dtype=bool), (u_same, v_same)), shape=(num_tri, num_tri))
    n_comp, comp_ids = csgraph.connected_components(adj_mat, directed=False)
    comp_sizes = np.bincount(comp_ids)

    is_small_cluster = (comp_sizes[comp_ids] < min_island_size) & (labels_cleaned >= 0)
    island_mask = isolated_single | is_small_cluster

    num_isolated = int(np.count_nonzero(isolated_single))
    num_island_tri = int(np.count_nonzero(island_mask))

    if num_island_tri == 0:
        return labels_cleaned, 0, 0

    # Small component IDs that need absorption
    island_comp_ids = np.unique(comp_ids[island_mask])
    comp_to_idx = {cid: i for i, cid in enumerate(island_comp_ids)}

    # Outer edges: u is in an island component, v is a stable neighbor (not an island and label >= 0)
    u_is_island = island_mask[adj_u]
    v_is_stable = (~island_mask[adj_v]) & (labels_cleaned[adj_v] >= 0)
    outer_edges = u_is_island & v_is_stable

    u_edge = adj_u[outer_edges]
    v_lbl = labels_cleaned[adj_v[outer_edges]]

    # For any island components not touching a stable neighbor (e.g. adjacent islands),
    # allow any valid neighbor with a different label as fallback
    u_comps_stable = comp_ids[u_edge] if len(u_edge) > 0 else np.array([], dtype=np.int32)
    touched_comps = set(u_comps_stable)
    untouched_island_comps = [cid for cid in island_comp_ids if cid not in touched_comps]
    if untouched_island_comps:
        v_any_valid = (labels_cleaned[adj_v] >= 0) & (labels_cleaned[adj_v] != labels_cleaned[adj_u])
        outer_any = u_is_island & v_any_valid
        if np.any(outer_any):
            u_any = adj_u[outer_any]
            v_any_lbl = labels_cleaned[adj_v[outer_any]]
            u_any_comps = comp_ids[u_any]
            keep_any = np.isin(u_any_comps, untouched_island_comps)
            if np.any(keep_any):
                if len(u_edge) > 0:
                    u_edge = np.concatenate([u_edge, u_any[keep_any]])
                    v_lbl = np.concatenate([v_lbl, v_any_lbl[keep_any]])
                else:
                    u_edge = u_any[keep_any]
                    v_lbl = v_any_lbl[keep_any]

    if len(u_edge) > 0:
        u_comps = comp_ids[u_edge]
        dense_comp_indices = np.array([comp_to_idx.get(c, -1) for c in u_comps], dtype=np.int32)
        valid_u = dense_comp_indices >= 0
        dense_comp_indices = dense_comp_indices[valid_u]
        v_lbl = v_lbl[valid_u]

        tally = np.zeros((len(island_comp_ids), num_classes), dtype=np.int32)
        np.add.at(tally, (dense_comp_indices, v_lbl), 1)

        has_majority = tally.max(axis=1) > 0
        best_lbl = tally.argmax(axis=1)

        absorbed_comps = island_comp_ids[has_majority]
        absorbed_labels = best_lbl[has_majority]

        comp_lookup = np.full(n_comp, -1, dtype=np.int32)
        comp_lookup[absorbed_comps] = absorbed_labels

        new_lbls = comp_lookup[comp_ids]
        mask_to_update = new_lbls >= 0
        labels_cleaned[mask_to_update] = new_lbls[mask_to_update]

    return labels_cleaned, num_isolated, num_island_tri


def export_colored_ply(
    file_path: Path,
    vertices: np.ndarray,
    triangles: np.ndarray,
    triangle_colors_rgb: np.ndarray,
) -> None:
    """
    Exports binary PLY mesh with per-triangle colors assigned to incident vertices.
    For sharp triangle boundaries, writes vertex colors using triangle assignment.
    """
    num_verts = len(vertices)
    num_faces = len(triangles)

    # Triangle colors are integer RGB values, so bincount preserves the exact sums.
    incident = triangles.T.reshape(-1)
    v_counts = np.maximum(1, np.bincount(incident, minlength=num_verts))[:, None]
    v_colors = np.column_stack([
        np.bincount(incident, weights=np.tile(triangle_colors_rgb[:, channel], 3),
                    minlength=num_verts)
        for channel in range(3)
    ])
    v_colors = (v_colors / v_counts).astype(np.uint8)

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

    with open(file_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(v_data.tobytes())
        f.write(f_data.tobytes())


def render_vector_pdf_matrix(
    pdf_path: Path,
    png_path: Path,
    class_ids: List[int],
    contention_counts: np.ndarray,
    uncontended_counts: np.ndarray,
    total_assigned_counts: np.ndarray,
) -> None:
    """
    Renders an ultra-high-resolution, fully scalable vector PDF of the
    upper-triangular contention matrix with color-coded class headers.
    """
    N = len(class_ids)
    # Figure size scaled with number of classes to guarantee readability
    fig_size = max(24.0, N * 0.35)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=300)

    # Matrix data: upper triangle = contention, diagonal = uncontended
    display_matrix = np.full((N, N), np.nan, dtype=np.float64)
    for i in range(N):
        display_matrix[i, i] = uncontended_counts[i]
        for j in range(i + 1, N):
            c_val = contention_counts[i, j]
            if c_val > 0:
                display_matrix[i, j] = c_val

    # Log colormap for wide dynamic range
    masked_vals = display_matrix[~np.isnan(display_matrix)]
    vmax = float(masked_vals.max()) if len(masked_vals) > 0 else 1.0
    norm = mcolors.LogNorm(vmin=1.0, vmax=max(10.0, vmax))
    cmap = plt.colormaps["YlOrRd"].with_extremes(bad="white")

    im = ax.imshow(display_matrix, cmap=cmap, norm=norm, interpolation="nearest")

    # Grid lines separating every cell
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which="minor", color="#DDDDDD", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    # Major ticks & Labels
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))

    labels = [f"#{cid} ({total_assigned_counts[idx]:,})" for idx, cid in enumerate(class_ids)]
    ax.set_xticklabels(labels, rotation=90, fontsize=8, ha="center")
    ax.set_yticklabels(labels, fontsize=8, va="center")

    from matplotlib.patches import Rectangle

    # Draw colored patch on each diagonal cell (i, i) with exact class mask color
    for i, cid in enumerate(class_ids):
        _, rgb = get_class_color(cid)
        rect = Rectangle(
            (i - 0.5, i - 0.5),
            1.0,
            1.0,
            facecolor=rgb,
            edgecolor="#666666",
            linewidth=0.5,
            zorder=2,
        )
        ax.add_patch(rect)

        diag_val = uncontended_counts[i]
        if diag_val > 0:
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            text_col = "white" if lum < 0.55 else "black"
            ax.text(
                i,
                i,
                f"{diag_val:,}",
                ha="center",
                va="center",
                fontsize=5.5,
                color=text_col,
                fontweight="bold",
                zorder=3,
            )

    # Colorize tick labels with matching class background badges
    for idx, (cid, tick) in enumerate(zip(class_ids, ax.get_xticklabels())):
        _, rgb = get_class_color(cid)
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        text_col = "white" if lum < 0.55 else "black"
        tick.set_bbox(dict(boxstyle="square,pad=0.25", facecolor=rgb, edgecolor="#555555", linewidth=0.5))
        tick.set_color(text_col)
        tick.set_weight("bold")

    for idx, (cid, tick) in enumerate(zip(class_ids, ax.get_yticklabels())):
        _, rgb = get_class_color(cid)
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        text_col = "white" if lum < 0.55 else "black"
        tick.set_bbox(dict(boxstyle="square,pad=0.25", facecolor=rgb, edgecolor="#555555", linewidth=0.5))
        tick.set_color(text_col)
        tick.set_weight("bold")

    # Overlay numeric cell values for upper-triangular contention entries
    for i in range(N):
        for j in range(i + 1, N):
            c_cnt = contention_counts[i, j]
            if c_cnt > 0:
                text_col = "white" if c_cnt > (vmax * 0.25) else "black"
                ax.text(
                    j,
                    i,
                    f"{c_cnt:,}",
                    ha="center",
                    va="center",
                    fontsize=5.0,
                    color=text_col,
                )

    ax.set_title(
        f"Cross-Class Contention Upper-Triangular Matrix ({N} Classes)\n"
        f"Diagonal = Class Mask Color (Uncontended Triangles) | Off-Diagonal = Contended Triangles",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Triangle Count (Log Scale)", fontsize=12)

    plt.tight_layout()

    # 1. Export vector PDF (zoomable to arbitrary scale)
    print(f"Exporting vector PDF matrix: {pdf_path}...")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

    # 2. Export raster PNG preview
    print(f"Exporting PNG matrix preview: {png_path}...")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def project_masks_to_3d(
    workspace: Path,
    mesh_path: Path,
    output_dir: Path,
    npz_dir: Path,
    colmap_threshold: int = 100,
    max_frames: int | None = None,
    min_viewed_ratio: float | None = None,
    min_win_rate: float | None = None,
    baseline_summary_csv: Path | None = None,
) -> Dict[str, Any]:
    t_start = time.perf_counter()
    # A failed recomputation must not leave an old completion marker beside new evidence.
    (output_dir / 'projection_stats.json').unlink(missing_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("STEP 2: 2D MASK PROJECTION & CONTENTION RESOLUTION")
    print(f"Workspace:          {workspace}")
    print(f"Input Mesh:         {mesh_path}")
    print(f"NPZ Masks Dir:      {npz_dir}")
    print(f"Output Dir:         {output_dir}")
    print(f"COLMAP Threshold:   {colmap_threshold} triangulated points")
    if min_viewed_ratio is not None and min_win_rate is not None:
        print(f"Filter Criteria:    viewed >= {min_viewed_ratio*100:.3f}% of mesh, win_rate >= {min_win_rate*100:.1f}%")
    print("=================================================================")

    # 1. Parse COLMAP Triangulated Points
    images_txt_path = scene_inputs.resolve_sparse_dir(workspace) / "images.txt"
    colmap_pts_map = scene_inputs.colmap_point_counts(images_txt_path)

    print(f"Loaded COLMAP point counts for {len(colmap_pts_map)} frames.")

    # 2. Load Clean Mesh & Construct BVH
    t0 = time.perf_counter()
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices_np = np.asarray(mesh.vertices)
    triangles_np = np.asarray(mesh.triangles)
    num_vertices = len(vertices_np)
    num_triangles = len(triangles_np)
    print(f"Loaded clean mesh: {num_vertices:,} vertices, {num_triangles:,} triangles ({time.perf_counter() - t0:.2f}s)")

    print("Building Open3D RaycastingScene (Embree BVH)...")
    t0_bvh = time.perf_counter()
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)
    print(f"BVH built in {time.perf_counter() - t0_bvh:.3f}s")

    # 3. Load Classes Manifest & Apply Optional Quality Filtering
    summary_candidates = [
        npz_dir.parent / "full_pipeline_benchmark_summary.json",
        output_dir / "panoptic_data/full_pipeline_benchmark_summary.json",
        output_dir / "full_pipeline_benchmark_summary.json",
        workspace / "05_semantic_segmentation/panoptic_data/full_pipeline_benchmark_summary.json",
        workspace / "05_semantic_segmentation/full362_final_pipeline/full_pipeline_benchmark_summary.json",
        workspace / "05_semantic_segmentation/full_pipeline_benchmark_summary.json",
    ]
    summary_path = next((p for p in summary_candidates if p.is_file()), None)
    if summary_path is not None:
        with open(summary_path) as f:
            bench_summary = json.load(f)
        initial_classes = sorted(bench_summary.get("final_classes", []))
    else:
        discovered = set()
        for npz_file in npz_dir.glob("*_final.npz"):
            with np.load(npz_file, allow_pickle=False) as data:
                if "labels" in data:
                    discovered.update(np.unique(data["labels"]))
                elif "classes" in data:
                    discovered.update(data["classes"])
        discovered.discard(0)
        initial_classes = sorted(int(cid) for cid in discovered if cid > 0)
    persistent_classes = initial_classes.copy()

    if min_viewed_ratio is not None and min_win_rate is not None:
        if baseline_summary_csv is None or not baseline_summary_csv.exists():
            candidate_csv = output_dir / "class_contention_summary_93classes.csv"
            if not candidate_csv.exists():
                candidate_csv = output_dir / "class_contention_summary.csv"
            baseline_summary_csv = candidate_csv

        if baseline_summary_csv.exists():
            thresh_triangles = min_viewed_ratio * num_triangles
            print(f"\n--- Filtering Classes from Baseline CSV: {baseline_summary_csv.name} ---")
            print(f"  Min viewed triangles: {thresh_triangles:.1f} ({min_viewed_ratio*100:.3f}% of {num_triangles:,})")
            print(f"  Min win rate:         {min_win_rate*100:.1f}%")

            surviving = []
            excluded = []
            with open(baseline_summary_csv) as f:
                reader = csv.DictReader(f)
                for r in reader:
                    cid = int(r["Class ID"])
                    if cid not in initial_classes:
                        continue
                    tot_assigned = int(
                        r.get("Total Assigned Triangles")
                        or r.get("Raw Dominant Assigned Triangles")
                        or r.get("Arbitrated & Cleaned Assigned Triangles")
                        or 0
                    )
                    uncontended = int(r.get("Uncontended Triangles", 0))
                    contended = int(r.get("Total Contended Triangles", 0))
                    viewed = uncontended + contended
                    win_rate = (tot_assigned / viewed) if viewed > 0 else 0.0

                    if viewed >= thresh_triangles and win_rate >= min_win_rate:
                        surviving.append((cid, viewed, win_rate))
                    else:
                        excluded.append((cid, viewed, win_rate))

            persistent_classes = sorted([c[0] for c in surviving])
            print(f"  Excluded {len(excluded)} classes. Retained {len(persistent_classes)} classes:")
            print(f"  Surviving Class IDs: {persistent_classes}")

    num_classes = len(persistent_classes)
    class_to_idx = {cid: idx for idx, cid in enumerate(persistent_classes)}
    idx_to_class = {idx: cid for idx, cid in enumerate(persistent_classes)}
    print(f"\nActive class set: {num_classes} classes configured for 3D projection.")

    # 4. Load Camera Parameters
    registered_cameras = sorted(
        scene_inputs.load_scene_cameras(workspace, resolution=1),
        key=lambda camera: Path(camera["image_path"]).relative_to(workspace / "images").as_posix(),
    )
    all_cameras = [
        camera for camera in registered_cameras
        if colmap_pts_map[Path(camera["image_path"]).relative_to(workspace / "images").as_posix()] >= colmap_threshold
    ]
    skipped_low_colmap_frames = len(registered_cameras) - len(all_cameras)
    if not all_cameras:
        raise ValueError(f"No registered frames have at least {colmap_threshold} triangulated COLMAP points")
    if max_frames is not None and max_frames > 0:
        all_cameras = all_cameras[:max_frames]
    camera_selection_sha256 = hashlib.sha256(
        "\n".join(
            f"{Path(camera['image_path']).relative_to(workspace / 'images').as_posix()}:{colmap_pts_map[Path(camera['image_path']).relative_to(workspace / 'images').as_posix()]}"
            for camera in all_cameras
        ).encode("utf-8")
    ).hexdigest()
    total_cameras = len(all_cameras)
    print(f"Loaded {total_cameras} camera frames.")

    # 5. Set Up GPU Accumulators
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using GPU device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # acc_weights: [num_triangles, num_classes] float32 (~1.01 GB)
    acc_weights = torch.zeros((num_triangles, num_classes), dtype=torch.float32, device=device)
    # Class ID fast index lookup (support up to 65535 IDs safely).
    max_cid = max(persistent_classes, default=0)
    cid_map_tensor = torch.full((max(max_cid + 1, 65536),), -1, dtype=torch.long, device=device)
    class_rgb = np.asarray([
        [int(channel * 255) for channel in get_class_color(cid)[1]]
        for cid in persistent_classes
    ], dtype=np.uint8)
    for cid, idx in class_to_idx.items():
        cid_map_tensor[cid] = idx

    # 6. Raycast & Accumulate Across Frames
    colmap_pose_frames = 0
    total_rays_cast = 0
    total_valid_hits = 0

    print("\n--- Raycasting & Vote Accumulation Across Frames ---")
    t0_ray = time.perf_counter()

    for f_idx, cam in enumerate(all_cameras):
        image_path = Path(cam["image_path"])
        img_name = image_path.relative_to(workspace / "images").as_posix()
        stem = image_path.stem
        npz_file = npz_dir / f"{stem}_final.npz"
        if not npz_file.exists():
            continue

        pts_3d = colmap_pts_map[img_name]

        viewmat = cam["viewmat"]
        colmap_pose_frames += 1

        with np.load(npz_file) as data:
            if "labels" not in data:
                raise ValueError(f"{npz_file} has no labels array")
            labels = scene_inputs.validate_labels(data["labels"], cam, classes=initial_classes)

        # Pinhole rays
        rays = scene.create_rays_pinhole(
            intrinsic_matrix=o3d.core.Tensor(cam["K"], dtype=o3d.core.Dtype.Float64),
            extrinsic_matrix=o3d.core.Tensor(viewmat, dtype=o3d.core.Dtype.Float64),
            width_px=cam["width"],
            height_px=cam["height"],
        )
        ans = scene.cast_rays(rays)

        primitive_ids = ans["primitive_ids"].numpy()  # [H, W] uint32
        hit_normals = ans["primitive_normals"].numpy()  # [H, W, 3] float32
        ray_dirs = rays.numpy()[:, :, 3:]  # [H, W, 3] float32

        # Mask distance transform weights
        w_edge = compute_mask_distance_weights(labels)

        # Grazing angle cosine must not vary with Open3D's ray-vector length.
        cos_theta = normalized_incidence_cosines(hit_normals, ray_dirs)
        surface_hits = (primitive_ids < num_triangles) & (cos_theta > 0.174)  # cos(80 deg)
        valid_hits = surface_hits & (labels > 0)

        if not np.any(valid_hits):
            continue

        tri_indices = primitive_ids[valid_hits]
        all_face_hit_ids = primitive_ids[surface_hits]
        hit_labels = labels[valid_hits]
        cos_vals = cos_theta[valid_hits]
        edge_vals = w_edge[valid_hits]

        weights = (cos_vals ** 1.5) * edge_vals

        # Normalize each face's evidence within this view.  A close-up cannot
        # outweigh another view merely because it covers more pixels.
        if num_classes:
            class_indices = np.full(len(hit_labels), -1, dtype=np.int64)
            in_lookup = (hit_labels >= 0) & (hit_labels < len(cid_map_tensor))
            class_indices[in_lookup] = cid_map_tensor[
                torch.as_tensor(hit_labels[in_lookup], dtype=torch.long, device=device)
            ].cpu().numpy()
            active = class_indices >= 0
            if np.any(active):
                flat_indices, view_votes = reduce_view_face_votes(
                    tri_indices[active], class_indices[active], weights[active], num_triangles, num_classes,
                    all_face_hit_ids=all_face_hit_ids,
                )
                acc_weights.view(-1).index_add_(
                    0,
                    torch.as_tensor(flat_indices, dtype=torch.long, device=device),
                    torch.as_tensor(view_votes, dtype=torch.float32, device=device),
                )

        total_rays_cast += cam["width"] * cam["height"]
        total_valid_hits += len(tri_indices)

        if (f_idx + 1) % 50 == 0 or (f_idx + 1) == total_cameras:
            print(
                f"  [{f_idx + 1:3d}/{total_cameras}] {stem}: "
                f"{len(tri_indices):,} valid ray hits (COLMAP, {pts_3d} 3D pts)"
            )

    t_ray_elapsed = time.perf_counter() - t0_ray
    print(f"\nRaycasting & Accumulation complete in {t_ray_elapsed:.2f}s ({total_valid_hits:,} total hits accumulated)")
    print(
        f"Frames: {colmap_pose_frames} COLMAP poses used, "
        f"{skipped_low_colmap_frames} skipped below the {colmap_threshold}-point threshold."
    )

    # 7. Contention analysis.  Raw labels are the local evidence argmax; the
    # downstream geometric pass owns all spatial regularization.
    print("\n--- Resolving Local Projection Evidence ---")
    t0_resolve = time.perf_counter()

    # Transfer accumulators to CPU
    acc_weights_cpu = acc_weights.cpu().numpy()

    # Number of distinct classes with non-zero votes per triangle
    class_vote_counts = (acc_weights_cpu > 0).sum(axis=1)  # [num_triangles]

    unvoted_mask = class_vote_counts == 0
    single_class_mask = class_vote_counts == 1
    contended_mask = class_vote_counts > 1

    num_unvoted = int(np.count_nonzero(unvoted_mask))
    num_single = int(np.count_nonzero(single_class_mask))
    num_contended = int(np.count_nonzero(contended_mask))

    print(f"Triangle Initial Classification:")
    print(f"  Uncontended (Single Class): {num_single:,} ({num_single / num_triangles * 100:.2f}%)")
    print(f"  Contended (> 1 Classes):     {num_contended:,} ({num_contended / num_triangles * 100:.2f}%)")
    print(f"  Unvoted (0 Classes):         {num_unvoted:,} ({num_unvoted / num_triangles * 100:.2f}%)")

    assigned_labels = local_evidence_labels(acc_weights_cpu)
    local_class_indices = assigned_labels
    cleaned_labels = assigned_labels
    num_underdog_flips = 0
    num_isolated = 0
    num_island_tri = 0

    # Generate Colored Meshes:
    # 1. Contention mesh: uncontended -> class color, contended -> pure white [255, 255, 255]
    triangle_colors_contention = np.zeros((num_triangles, 3), dtype=np.uint8)
    triangle_colors_contention[unvoted_mask] = [128, 128, 128]
    triangle_colors_contention[contended_mask] = [255, 255, 255]

    single_tri_indices = np.flatnonzero(single_class_mask)
    if len(single_tri_indices) > 0:
        single_class_indices = acc_weights_cpu[single_tri_indices].argmax(axis=1)
        triangle_colors_contention[single_tri_indices] = class_rgb[single_class_indices]

    # 2. Resolved mesh: all voted triangles colored by their local evidence winner.
    triangle_colors_resolved = np.zeros((num_triangles, 3), dtype=np.uint8)
    triangle_colors_resolved[unvoted_mask] = [128, 128, 128]

    voted_indices = np.flatnonzero(cleaned_labels >= 0)
    triangle_colors_resolved[voted_indices] = class_rgb[cleaned_labels[voted_indices]]

    t_resolve_elapsed = time.perf_counter() - t0_resolve
    print(f"Color assignment completed in {t_resolve_elapsed:.3f}s")

    # Save raw triangle class IDs (-1 for unvoted) for downstream geometry.
    persistent_classes_np = np.array(persistent_classes, dtype=np.int32)
    raw_triangle_classes = np.full(num_triangles, -1, dtype=np.int32)
    voted = assigned_labels >= 0
    if np.any(voted):
        raw_triangle_classes[voted] = persistent_classes_np[assigned_labels[voted]]
    raw_labels_path = output_dir / "triangle_class_labels_raw.npy"
    np.save(raw_labels_path, raw_triangle_classes)
    print(f"Saved raw triangle class labels: {raw_labels_path}")

    from scipy.sparse import csr_matrix, save_npz
    votes_path = output_dir / "face_class_votes.npz"
    source_ids_path = output_dir / "face_class_source_ids.npy"
    save_npz(votes_path, csr_matrix(acc_weights_cpu))
    np.save(source_ids_path, persistent_classes_np)
    print(f"Saved sparse face-class evidence: {votes_path}")

    # 8. Export Colored 3D Meshes
    t0_exp = time.perf_counter()
    output_mesh_contention_path = output_dir / "mesh_projected_contention.ply"
    print(f"Exporting contention PLY mesh (white disputes): {output_mesh_contention_path}...")
    export_colored_ply(output_mesh_contention_path, vertices_np, triangles_np, triangle_colors_contention)

    output_mesh_resolved_path = output_dir / "mesh_projected_resolved.ply"
    print(f"Exporting arbitrated & cleaned PLY mesh: {output_mesh_resolved_path}...")
    export_colored_ply(output_mesh_resolved_path, vertices_np, triangles_np, triangle_colors_resolved)
    print(f"Exported both PLY meshes in {time.perf_counter() - t0_exp:.2f}s")

    # 9. Compute Cross-Class Contention Matrix
    print("\n--- Computing Cross-Class Contention Upper-Triangular Matrix ---")
    t0_mat = time.perf_counter()

    # contention_matrix[i, j]: number of triangles where both class i and class j have votes > 0
    has_votes = (acc_weights_cpu > 0).astype(np.float32)  # [num_triangles, num_classes]

    # Sparse / Matrix multiplication on GPU for maximum speed: [num_classes, num_classes]
    if num_classes:
        t_has_votes = torch.as_tensor(has_votes, device=device)
        contention_tensor = torch.matmul(t_has_votes.T, t_has_votes).cpu().numpy().astype(np.int64)
    else:
        contention_tensor = np.zeros((0, 0), dtype=np.int64)

    # Diagonal: total triangles that class touched
    total_touched_per_class = np.diag(contention_tensor).copy()

    # Dominant winner per triangle (pre-arbitration argmax over accumulated weights)
    dominant_class_idx = local_class_indices if num_classes else np.full(num_triangles, -1, dtype=np.int64)
    assigned_per_class = np.zeros(num_classes, dtype=np.int64)
    uncontended_per_class = np.zeros(num_classes, dtype=np.int64)
    arbitrated_assigned_per_class = np.zeros(num_classes, dtype=np.int64)

    for c_idx in range(num_classes):
        assigned_per_class[c_idx] = int(np.count_nonzero((dominant_class_idx == c_idx) & ~unvoted_mask))
        uncontended_per_class[c_idx] = int(np.count_nonzero((dominant_class_idx == c_idx) & single_class_mask))
        arbitrated_assigned_per_class[c_idx] = int(np.count_nonzero(cleaned_labels == c_idx))

    # Upper triangular contention matrix (zero out diagonal for pairwise contention)
    pair_contention_matrix = contention_tensor.copy()
    np.fill_diagonal(pair_contention_matrix, 0)

    # Class total contended triangles (triangles where this class had votes and >= 1 other class had votes)
    contended_triangles_per_class = total_touched_per_class - uncontended_per_class

    t_mat_elapsed = time.perf_counter() - t0_mat
    print(f"Contention matrix computed in {t_mat_elapsed:.3f}s")

    # 10. Render Vector PDF & PNG Contention Matrix
    pdf_matrix_path = output_dir / "contention_matrix.pdf"
    png_matrix_path = output_dir / "contention_matrix.png"
    if num_classes:
        render_vector_pdf_matrix(
            pdf_path=pdf_matrix_path,
            png_path=png_matrix_path,
            class_ids=persistent_classes,
            contention_counts=pair_contention_matrix,
            uncontended_counts=uncontended_per_class,
            total_assigned_counts=arbitrated_assigned_per_class,
        )

    # 11. Generate CSV Summary Table
    csv_path = output_dir / "class_contention_summary.csv"
    csv_rows = []
    for c_idx, cid in enumerate(persistent_classes):
        tot_assigned = int(assigned_per_class[c_idx])
        arbitrated_assigned = int(arbitrated_assigned_per_class[c_idx])
        uncont = int(uncontended_per_class[c_idx])
        tot_cont = int(contended_triangles_per_class[c_idx])

        # List of contended classes with triangle counts
        contended_partners = []
        for other_idx, other_cid in enumerate(persistent_classes):
            if c_idx != other_idx:
                c_count = int(contention_tensor[c_idx, other_idx])
                if c_count > 0:
                    contended_partners.append(f"#{other_cid}:{c_count}")

        partner_str = "; ".join(contended_partners)
        csv_rows.append([cid, arbitrated_assigned, tot_assigned, uncont, tot_cont, len(contended_partners), partner_str])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Class ID",
            "Arbitrated & Cleaned Assigned Triangles",
            "Raw Dominant Assigned Triangles",
            "Uncontended Triangles",
            "Total Contended Triangles",
            "Num Contending Classes",
            "Contending Classes Breakdown",
        ])
        writer.writerows(csv_rows)
    print(f"Exported Class Contention Summary CSV: {csv_path}")

    # Top 15 Contended Classes
    sorted_by_cont = sorted(csv_rows, key=lambda r: r[4], reverse=True)
    print("\nTop 15 Classes by Contended Triangles:")
    for r in sorted_by_cont[:15]:
        print(f"  Class #{r[0]:3d}: {r[1]:8,d} arbitrated assigned ({r[2]:8,d} raw) | {r[3]:8,d} uncontended | {r[4]:8,d} contended with {r[5]} classes")

    # 12. Structured JSON Output
    t_total = time.perf_counter() - t_start
    summary_data = {
        "mesh_path": str(mesh_path.resolve()),
        "mesh_sha256": _sha256_file(mesh_path),
        "input_hashes": {str(path): _sha256_file(path) for path in
                         (mesh_path, images_txt_path.parent / 'cameras.txt', images_txt_path)},
        "mask_files": {p.name: [p.stat().st_size, p.stat().st_mtime_ns]
                       for p in sorted(npz_dir.glob('*_final.npz'))},
        "colmap_threshold": colmap_threshold,
        "max_frames": max_frames,
        "camera_selection_sha256": camera_selection_sha256,
        "output_mesh_contention": str(output_mesh_contention_path.resolve()),
        "output_mesh_resolved": str(output_mesh_resolved_path.resolve()),
        "pdf_matrix": str(pdf_matrix_path.resolve()) if num_classes else None,
        "png_matrix": str(png_matrix_path.resolve()) if num_classes else None,
        "csv_summary": str(csv_path.resolve()),
        "min_viewed_ratio": min_viewed_ratio,
        "min_win_rate": min_win_rate,
        "projection_version": "view-normalized-v2",
        "face_class_votes": str(votes_path.resolve()),
        "face_class_source_ids": str(source_ids_path.resolve()),
        "margin_threshold": None,
        "underdog_flips": num_underdog_flips,
        "isolated_triangles_absorbed": num_isolated,
        "island_triangles_absorbed": num_island_tri,
        "num_classes": num_classes,
        "active_classes": persistent_classes,
        "total_triangles": num_triangles,
        "uncontended_triangles": num_single,
        "contended_triangles": num_contended,
        "unvoted_triangles": num_unvoted,
        "coverage_percentage": round((num_single + num_contended) / num_triangles * 100, 2),
        "colmap_pose_frames": colmap_pose_frames,
        "arcore_pose_frames": 0,
        "skipped_low_colmap_frames": skipped_low_colmap_frames,
        "total_ray_hits": total_valid_hits,
        "total_pipeline_seconds": round(t_total, 3),
        "timings": {
            "raycasting_seconds": round(t_ray_elapsed, 3),
            "contention_resolution_seconds": round(t_resolve_elapsed, 3),
            "export_ply_seconds": round(time.perf_counter() - t0_exp, 3),
            "contention_matrix_seconds": round(t_mat_elapsed, 3),
            "total_seconds": round(t_total, 3),
        },
    }

    stats_json_path = output_dir / "projection_stats.json"
    pending_stats = stats_json_path.with_suffix('.work.json')
    pending_stats.write_text(json.dumps(summary_data, indent=2) + '\n')
    pending_stats.replace(stats_json_path)
    print(f"Exported Projection Stats JSON: {stats_json_path}")
    print(f"\n=================================================================")
    print(f"STEP 2 MASK PROJECTION COMPLETED SUCCESSFULLY IN {t_total:.2f}s!")
    print(f"=================================================================")

    return summary_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 2: 2D Mask Projection & Contention Analysis")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=backend_dir / "current_scene",
        help="Workspace directory",
    )
    parser.add_argument(
        "--mesh-path",
        type=Path,
        default=None,
        help="Sealed cleaned 3D mesh PLY",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--npz-dir",
        type=Path,
        default=None,
        help="Directory containing 2D panoptic segmentation NPZ archives",
    )
    parser.add_argument(
        "--colmap-threshold",
        type=int,
        default=100,
        help="Minimum triangulated COLMAP points required to use a frame",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit number of frames for rapid dry-runs",
    )
    parser.add_argument(
        "--min-viewed-ratio",
        type=float,
        default=None,
        help="Minimum ratio of mesh triangles viewed by a class (e.g. 0.0015 for 0.15%)",
    )
    parser.add_argument(
        "--min-win-rate",
        type=float,
        default=None,
        help="Minimum win rate (assigned / viewed) for a class (e.g. 0.20 for 20%)",
    )
    parser.add_argument(
        "--baseline-summary-csv",
        type=Path,
        default=None,
        help="Path to baseline class contention summary CSV for filtering",
    )

    args = parser.parse_args(argv)

    scan = scene_inputs.resolve_scene(args.workspace)
    mesh_path = scene_inputs.resolve_mesh(scan, args.mesh_path)
    stage_out = scene_inputs.resolve_stage_output(scan)
    output_dir = args.output_dir or stage_out
    if args.npz_dir is not None and args.npz_dir.is_dir():
        npz_dir = args.npz_dir
    else:
        candidates = [
            stage_out / "panoptic_data/npz_data",
            scan / "05_semantic_segmentation/full362_final_pipeline/npz_data",
            stage_out / "npz_data",
        ]
        npz_dir = next((p for p in candidates if p.is_dir() and any(p.glob("*.npz"))), stage_out / "panoptic_data/npz_data")

    project_masks_to_3d(
        workspace=scan,
        mesh_path=mesh_path,
        output_dir=output_dir,
        npz_dir=npz_dir,
        colmap_threshold=args.colmap_threshold,
        max_frames=args.max_frames,
        min_viewed_ratio=args.min_viewed_ratio,
        min_win_rate=args.min_win_rate,
        baseline_summary_csv=args.baseline_summary_csv,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
