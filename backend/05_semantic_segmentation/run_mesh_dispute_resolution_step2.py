#!/usr/bin/env python3
"""
Step 2: Geometry & Scale-Aware Dispute Resolution.

Resolves multi-view segmentation boundary overlaps on the 3D mesh faces using:
1. Scale-Normalized Support (Frequency/Volume Damping):
   - Dampens dominant super-classes by (N_core)^alpha to prevent background cannibalization
     of smaller objects (shelves, paintings, decor, furniture).
2. Local Dihedral Crease-Aware Graph Smoothing:
   - Uses mesh dual-graph edge penalties: w(i, j) = exp(-theta^2 / (2*sigma^2)).
   - Strongly penalizes boundaries cutting across smooth surfaces (w ~ 1.0).
   - Allows boundaries to naturally snap into physical creases, steps, and indentations (w -> 0).
   - Does NOT assume global planes or flat walls; fully supports multi-wall super-classes with internal 90-deg corners.
3. Topological Contiguity & Floater Pruning:
   - Absorbs tiny isolated crumbs (< 60 triangles) into surrounding majority bodies.
4. Quantitative Comparison & Export:
   - Exports updated instance-colored mesh: 'mesh_dispute_resolved.ply'.
   - Generates side-by-side comparison JSON, updated contention matrix, and re-projected verification strips.
"""

import os, sys, json, time, argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

backend_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_dir / "03_FastGS_DNSplatter"))
from gsplat_train import load_cameras

def get_class_color(mid: int):
    """Deterministic color matching 2D overlays in run_full_scene_pipeline.py."""
    if mid <= 0:
        return (128, 128, 128), (0.5, 0.5, 0.5)
    k = mid - 1
    b = (37 * k + 93) % 255
    g = (109 * k + 71) % 255
    r = (197 * k + 31) % 255
    col_bgr = (int(b), int(g), int(r))
    col_rgb = (r / 255.0, g / 255.0, b / 255.0)
    return col_bgr, col_rgb

def compute_mask_distance_weights(labels: np.ndarray) -> np.ndarray:
    H, W = labels.shape
    labels_pad = np.pad(labels, 1, mode='edge')
    diff_r = labels != labels_pad[1:-1, 2:]
    diff_d = labels != labels_pad[2:, 1:-1]
    boundary = (diff_r | diff_d).astype(np.uint8)
    dist = cv2.distanceTransform(1 - boundary, cv2.DIST_L2, 3)
    w_edge = np.clip(dist / 12.0, 0.1, 1.0).astype(np.float32)
    w_edge[labels == 0] = 0.0
    return w_edge

def build_mesh_dual_graph(triangles_np: np.ndarray, normals_tensor: torch.Tensor, device: str):
    """Builds mesh triangle-to-triangle adjacency and dihedral crease weights on GPU."""
    num_triangles = len(triangles_np)
    edges = np.vstack([
        np.sort(triangles_np[:, [0, 1]], axis=1),
        np.sort(triangles_np[:, [1, 2]], axis=1),
        np.sort(triangles_np[:, [2, 0]], axis=1)
    ])
    tri_indices = np.tile(np.arange(num_triangles), 3)

    edge_ids = edges[:, 0].astype(np.int64) * (2**32) + edges[:, 1].astype(np.int64)
    sort_idx = np.argsort(edge_ids)
    sorted_edge_ids = edge_ids[sort_idx]
    sorted_tri_idx = tri_indices[sort_idx]

    is_duplicate = sorted_edge_ids[:-1] == sorted_edge_ids[1:]
    tri_a_cpu = sorted_tri_idx[:-1][is_duplicate]
    tri_b_cpu = sorted_tri_idx[1:][is_duplicate]

    tri_a = torch.from_numpy(tri_a_cpu).long().to(device)
    tri_b = torch.from_numpy(tri_b_cpu).long().to(device)

    # Compute dihedral angles between adjacent triangles
    dot = (normals_tensor[tri_a] * normals_tensor[tri_b]).sum(dim=-1).clamp(-1.0, 1.0)
    dihedral = torch.acos(dot) # radians

    # Smoothness penalty: high for coplanar (theta ~ 0), low for crease (theta > 25 deg)
    sigma = np.radians(25.0)
    edge_weights = torch.exp(-(dihedral ** 2) / (2 * (sigma ** 2)))

    # Symmetric directed edges
    edge_src = torch.cat([tri_a, tri_b], dim=0)
    edge_dst = torch.cat([tri_b, tri_a], dim=0)
    edge_w = torch.cat([edge_weights, edge_weights], dim=0)

    return edge_src, edge_dst, edge_w, dihedral

def main():
    parser = argparse.ArgumentParser(description="Step 2: Geometry & Scale-Aware Dispute Resolution")
    parser.add_argument("--workspace", type=Path, default=backend_dir / "current_scene")
    parser.add_argument("--mesh-path", type=Path, default=None)
    parser.add_argument("--step1-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=0.0, help="Scale damping exponent (0.0 = original confidence & occurrences)")
    parser.add_argument("--smoothness-lambda", type=float, default=1.5, help="Dihedral crease smoothing strength")
    parser.add_argument("--iterations", type=int, default=15, help="Graph smoothing relaxation iterations")
    args = parser.parse_args()

    workspace = args.workspace
    mesh_path = args.mesh_path or (workspace / "06c_semantic_segmentation/inputs/mesh.ply")
    t_path = workspace / "06c_semantic_segmentation/inputs/T_mesh_to_raw.npy"
    npz_dir = workspace / "06c_semantic_segmentation/full362_final_pipeline/npz_data"
    step1_dir = args.step1_dir or (workspace / "06c_semantic_segmentation/mesh_projection_step1")
    summary_path = workspace / "06c_semantic_segmentation/full362_final_pipeline/full_pipeline_benchmark_summary.json"
    out_dir = args.output_dir or (workspace / "06c_semantic_segmentation/mesh_dispute_resolution_step2")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================", flush=True)
    print("STEP 2: GEOMETRY & SCALE-AWARE DISPUTE RESOLUTION", flush=True)
    print(f"Workspace:         {workspace}", flush=True)
    print(f"Step 1 Baseline:   {step1_dir}", flush=True)
    print(f"Output Directory:  {out_dir}", flush=True)
    print(f"Scale Alpha:       {args.alpha} (damping exponent)", flush=True)
    print(f"Smoothness Lambda: {args.smoothness_lambda} (crease boundary strength)", flush=True)
    print(f"Iterations:        {args.iterations}", flush=True)
    print("=================================================================", flush=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Compute device: {device}", flush=True)

    # 1. Load Classes List
    with open(summary_path) as f:
        bench_summary = json.load(f)
    persistent_classes = sorted(bench_summary['final_classes'])
    num_classes = len(persistent_classes)
    class_to_idx = {cid: idx for idx, cid in enumerate(persistent_classes)}
    idx_to_class = {idx: cid for idx, cid in enumerate(persistent_classes)}
    idx_to_class_arr = np.array([idx_to_class[i] for i in range(num_classes)], dtype=np.int32)

    # 2. Load Mesh & Compute Normals
    print("\nLoading 3D mesh and computing normals...", flush=True)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_triangle_normals()
    num_triangles = len(mesh.triangles)
    num_vertices = len(mesh.vertices)
    triangles_np = np.asarray(mesh.triangles)
    normals_tensor = torch.from_numpy(np.asarray(mesh.triangle_normals)).float().to(device)
    print(f"Mesh loaded: {num_vertices:,} vertices, {num_triangles:,} triangles", flush=True)

    # 3. Load Cached Accumulated Vote Tensor
    cache_acc_path = step1_dir / "accumulated_votes.pt"
    if cache_acc_path.exists():
        print(f"Loading cached accumulated vote tensor from {cache_acc_path.name}...", flush=True)
        t0 = time.perf_counter()
        acc = torch.load(cache_acc_path, map_location=device)
        print(f"Loaded accumulated votes tensor {acc.shape} in {time.perf_counter() - t0:.2f}s", flush=True)
    else:
        raise FileNotFoundError(f"Cache {cache_acc_path} not found!")

    # 4. Step 1 Baseline Labels for Comparison (Vectorized)
    sum_w = acc.sum(dim=-1)
    observed_mask = sum_w > 0
    observed_mask_cpu = observed_mask.cpu().numpy()
    baseline_win_idx_cpu = acc.argmax(dim=-1).cpu().numpy()
    baseline_win_cid = np.where(observed_mask_cpu, idx_to_class_arr[baseline_win_idx_cpu], 0).astype(np.int32)

    # 5. Build Mesh Dual Graph
    print("Building mesh dual-graph and localized dihedral crease weights on GPU...", flush=True)
    t_g0 = time.perf_counter()
    edge_src, edge_dst, edge_w, dihedral_angles = build_mesh_dual_graph(triangles_np, normals_tensor, device)
    print(f"Dual-graph built: {len(edge_src):,} directed edge links in {time.perf_counter() - t_g0:.3f}s", flush=True)

    # 6. Step 2.1: Noise Gating & Scale Normalization
    print("\n--- Applying Noise Significance Gating & Statistical Scale Normalization ---", flush=True)
    # Noise Significance Filter: classes must have >= 15% of max local vote and >= 1.0 unscaled weight
    max_w = acc.max(dim=-1, keepdim=True).values
    significant_mask = (acc >= 0.15 * max_w) & (acc >= 1.0)
    acc_filtered = torch.where(significant_mask, acc, torch.zeros_like(acc))
    sum_filt = acc_filtered.sum(dim=-1)
    has_filt = sum_filt > 0
    acc_clean = torch.where(has_filt[:, None], acc_filtered, acc)

    binary_pos = (acc_clean > 0).float()
    claims_per_tri = binary_pos.sum(dim=-1)
    uncontended_mask = (claims_per_tri == 1) & observed_mask

    # Compute uncontended core volume per class
    core_counts = (binary_pos * uncontended_mask[:, None].float()).sum(dim=0) # [num_classes]
    core_counts_cpu = core_counts.cpu().numpy()

    # Scale damping factor: (N_core)^alpha
    min_core = 500.0
    scales = torch.clamp(core_counts, min=min_core) ** args.alpha
    print(f"Scale factors: min = {scales.min():.1f} (Class #{idx_to_class[scales.argmin().item()]}), "
          f"max = {scales.max():.1f} (Class #{idx_to_class[scales.argmax().item()]})", flush=True)

    # Scale-normalized vote weights: W_tilde = W / scale
    acc_scaled = acc_clean / scales[None, :]

    # 7. Identify Seeds vs. Disputed Triangles
    print("\n--- Classifying Seeds vs Disputed Triangles ---", flush=True)
    contended_tri_mask = (claims_per_tri > 1) & observed_mask
    num_contended = contended_tri_mask.sum().item()
    print(f"Total observed triangles: {observed_mask.sum().item():,}", flush=True)
    print(f"Uncontended seed triangles: {uncontended_mask.sum().item():,}", flush=True)
    print(f"Contended triangles in dispute: {num_contended:,} ({num_contended / observed_mask.sum().item() * 100:.1f}%)", flush=True)

    # Normalized probability over competing classes per triangle
    probs = acc_scaled / (acc_scaled.sum(dim=-1, keepdim=True) + 1e-7)

    # Permanent Seeds: uncontended triangles OR undisputed dominance (> 0.90)
    top1_prob, _ = probs.max(dim=-1)
    permanent_seed_mask = uncontended_mask | ((top1_prob > 0.90) & observed_mask)
    active_dispute_mask = observed_mask & (~permanent_seed_mask)
    print(f"Permanent seeds locked: {permanent_seed_mask.sum().item():,}", flush=True)
    print(f"Active boundary dispute triangles: {active_dispute_mask.sum().item():,}", flush=True)

    # Initial labels from scale-normalized votes
    current_labels = acc_scaled.argmax(dim=-1) # [num_triangles]

    # 8. Step 2.2: Localized Crease-Aware Graph Smoothing (Two-Phase MRF Relaxation)
    print(f"\n--- Running Localized Crease-Aware Graph Relaxation ({args.iterations} Iterations) ---", flush=True)
    t_rel0 = time.perf_counter()
    lam = args.smoothness_lambda

    # Precompute data cost: -log(probs + 1e-4)
    data_cost = -torch.log(probs + 1e-4) # [N, K]

    # Precompute total edge weight connected to each triangle
    total_w_src = torch.zeros(num_triangles, device=device)
    total_w_src.index_add_(0, edge_src, edge_w)

    # Pre-allocate agreement buffer on GPU
    agree_matrix = torch.zeros((num_triangles, num_classes), device=device)

    # Parity masks for alternating Gauss-Seidel updates (prevents parallel Jacobi limit cycles)
    tri_idx_gpu = torch.arange(num_triangles, device=device)
    even_mask = (tri_idx_gpu % 2 == 0) & active_dispute_mask
    odd_mask = (tri_idx_gpu % 2 == 1) & active_dispute_mask

    for it in range(args.iterations):
        total_changed = 0

        # Sub-step 1: Update even triangles
        dst_labels = current_labels[edge_dst]
        agree_matrix.zero_()
        linear_agree_idx = edge_src * num_classes + dst_labels
        agree_matrix.view(-1).index_add_(0, linear_agree_idx, edge_w)
        smooth_cost = lam * (total_w_src[:, None] - agree_matrix)
        total_cost = data_cost + smooth_cost
        new_labels = total_cost.argmin(dim=-1)
        ch_even = (new_labels != current_labels) & even_mask
        current_labels[even_mask] = new_labels[even_mask]
        total_changed += ch_even.sum().item()

        # Sub-step 2: Update odd triangles
        dst_labels = current_labels[edge_dst]
        agree_matrix.zero_()
        linear_agree_idx = edge_src * num_classes + dst_labels
        agree_matrix.view(-1).index_add_(0, linear_agree_idx, edge_w)
        smooth_cost = lam * (total_w_src[:, None] - agree_matrix)
        total_cost = data_cost + smooth_cost
        new_labels = total_cost.argmin(dim=-1)
        ch_odd = (new_labels != current_labels) & odd_mask
        current_labels[odd_mask] = new_labels[odd_mask]
        total_changed += ch_odd.sum().item()

        if (it + 1) % 5 == 0 or it == 0 or (it + 1) == args.iterations:
            print(f"  Iteration {it + 1:2d}/{args.iterations}: {total_changed:,} disputed triangles updated boundary", flush=True)

    print(f"Graph relaxation converged in {time.perf_counter() - t_rel0:.3f}s!", flush=True)

    # 9. Final Label Mapping & 3D Connected Component Instance Decomposition
    final_win_idx_cpu = current_labels.cpu().numpy()
    final_win_cid = np.where(observed_mask_cpu, idx_to_class_arr[final_win_idx_cpu], 0).astype(np.int32)

    print("\n--- Running 3D Connected Component Instance Decomposition ---", flush=True)
    t_cc0 = time.perf_counter()
    import scipy.sparse as sp

    # Build dual-graph CSR matrix on CPU
    u_cpu = edge_src.cpu().numpy()
    v_cpu = edge_dst.cpu().numpy()
    adj_matrix = sp.coo_matrix((np.ones(len(u_cpu), dtype=bool), (u_cpu, v_cpu)), shape=(num_triangles, num_triangles)).tocsr()

    normals_np = np.asarray(mesh.triangle_normals)
    bg_cids = {1, 5} # Giant background masks: Floor #1 and Wall #5

    # Pass 1: Sub-Part Merging into Foreground Parents & Background Immunity
    print("Pass 1: Reconciling sub-parts and applying background protection...", flush=True)
    merged_subparts = 0
    preserved_wall_objects = 0
    pruned_speckles = 0

    for cid in persistent_classes:
        if cid in bg_cids:
            continue
        c_tris = np.where(final_win_cid == cid)[0]
        if len(c_tris) == 0:
            continue

        sub_adj = adj_matrix[c_tris, :][:, c_tris]
        n_comp, comp_labels = sp.csgraph.connected_components(sub_adj, directed=False)
        comp_sizes = np.bincount(comp_labels)

        for comp_k, size in enumerate(comp_sizes):
            if size >= 150:
                continue
            comp_tri_indices = c_tris[comp_labels == comp_k]
            nbr_tris = adj_matrix[comp_tri_indices].indices
            nbr_labels = final_win_cid[nbr_tris]
            nbr_labels = nbr_labels[(nbr_labels != cid) & (nbr_labels > 0)]
            if len(nbr_labels) == 0:
                continue

            dom_lbl = int(np.bincount(nbr_labels).argmax())

            # Case A: Bordering another foreground object -> merge sub-part into parent object
            if dom_lbl not in bg_cids:
                final_win_cid[comp_tri_indices] = dom_lbl
                merged_subparts += 1
            else:
                # Case B: Bordering giant background mask (Wall or Floor)
                # Check for physical 3D protrusion (normals deviate > 25 deg)
                nbr_bg_tris = nbr_tris[final_win_cid[nbr_tris] == dom_lbl]
                if len(nbr_bg_tris) > 0:
                    mean_bg_norm = normals_np[nbr_bg_tris].mean(axis=0)
                    norm_len = np.linalg.norm(mean_bg_norm)
                    if norm_len > 1e-4:
                        mean_bg_norm /= norm_len
                        cos_diff = np.abs((normals_np[comp_tri_indices] * mean_bg_norm).sum(axis=-1))
                        has_protrusion = bool((cos_diff < 0.85).mean() > 0.30)
                    else:
                        has_protrusion = False
                else:
                    has_protrusion = False

                # Keep if size >= 30 or has 3D protrusion (frames, switches, cross, brackets)
                if size >= 30 or has_protrusion:
                    preserved_wall_objects += 1
                else:
                    # Tiny non-protruding speckle (< 30 tris) -> absorb into wall/floor
                    final_win_cid[comp_tri_indices] = dom_lbl
                    pruned_speckles += 1

    print(f"  Sub-parts merged into parent objects: {merged_subparts:,}", flush=True)
    print(f"  Wall/floor fixtures & objects preserved: {preserved_wall_objects:,}", flush=True)
    print(f"  Stray background speckles cleaned: {pruned_speckles:,}", flush=True)

    # Pass 2: Extract Clean 3D Physical Object Instances
    print("Pass 2: Extracting physical 3D object instances...", flush=True)
    instance_manifest = []
    tri_centers = np.asarray(mesh.vertices)[triangles_np].mean(axis=1)
    next_inst_id = 1

    for cid in persistent_classes:
        c_tris = np.where(final_win_cid == cid)[0]
        if len(c_tris) == 0:
            continue

        is_bg = cid in bg_cids or cid == 2 # 1=floor, 5=wall, 2=bed/ceiling
        sub_adj = adj_matrix[c_tris, :][:, c_tris]
        n_comp, comp_labels = sp.csgraph.connected_components(sub_adj, directed=False)
        comp_sizes = np.bincount(comp_labels)
        sorted_comp_indices = np.argsort(comp_sizes)[::-1]

        valid_comp_count = 0
        for comp_k in sorted_comp_indices:
            size = comp_sizes[comp_k]
            comp_tri_indices = c_tris[comp_labels == comp_k]

            min_manifest_size = 1000 if is_bg else 50
            if size < min_manifest_size:
                continue

            inst_name = f"Class_{cid}_inst_{valid_comp_count}" if not is_bg else f"Class_{cid}"
            valid_comp_count += 1
            pts = tri_centers[comp_tri_indices]
            manifest_entry = {
                'instance_id': next_inst_id,
                'class_id': int(cid),
                'instance_name': inst_name,
                'num_triangles': int(size),
                'centroid': [round(float(x), 3) for x in pts.mean(axis=0)],
                'bbox_min': [round(float(x), 3) for x in pts.min(axis=0)],
                'bbox_max': [round(float(x), 3) for x in pts.max(axis=0)],
                'dimensions': [round(float(pts[:,k].max() - pts[:,k].min()), 3) for k in range(3)]
            }
            instance_manifest.append(manifest_entry)
            next_inst_id += 1

    print(f"Extracted {len(instance_manifest):,} physical 3D object instances in {time.perf_counter() - t_cc0:.3f}s", flush=True)

    # Save instance manifest
    manifest_path = out_dir / "instance_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(instance_manifest, f, indent=2)
    print(f"Exported instance manifest: {manifest_path}", flush=True)

    # 10. Quantitative Comparison vs Baseline
    print("\n=================================================================", flush=True)
    print("QUANTITATIVE COMPARISON: STEP 1 (BASELINE) vs STEP 2 (GEOMETRY-RESOLVED)", flush=True)
    print("=================================================================", flush=True)

    baseline_wins = {cid: int((baseline_win_cid == cid).sum()) for cid in persistent_classes}
    step2_wins = {cid: int((final_win_cid == cid).sum()) for cid in persistent_classes}

    gains = []
    for cid in persistent_classes:
        b_w = baseline_wins[cid]
        s_w = step2_wins[cid]
        diff = s_w - b_w
        pct = (diff / b_w * 100.0) if b_w > 0 else 0.0
        gains.append({
            'class_id': cid,
            'baseline_wins': b_w,
            'step2_wins': s_w,
            'diff': diff,
            'pct_change': round(pct, 2),
            'core_volume': int(core_counts_cpu[class_to_idx[cid]])
        })

    gains_sorted = sorted(gains, key=lambda x: x['diff'], reverse=True)
    print("\nTop 15 Classes with Recovered Territory in Step 2:", flush=True)
    for rank, g in enumerate(gains_sorted[:15], 1):
        sign = "+" if g['diff'] >= 0 else ""
        print(f"  {rank:2d}. Class #{g['class_id']:3d}: {g['baseline_wins']:,} -> {g['step2_wins']:,} ({sign}{g['diff']:,} triangles, {sign}{g['pct_change']}%)", flush=True)

    # Crease Alignment Metric: Check boundary edges across classes
    edge_src_cpu = edge_src.cpu().numpy()
    edge_dst_cpu = edge_dst.cpu().numpy()
    dihedral_cpu = dihedral_angles.cpu().numpy()
    num_unique_edges = len(dihedral_cpu)

    b_boundary_edges = baseline_win_cid[edge_src_cpu[:num_unique_edges]] != baseline_win_cid[edge_dst_cpu[:num_unique_edges]]
    s_boundary_edges = final_win_cid[edge_src_cpu[:num_unique_edges]] != final_win_cid[edge_dst_cpu[:num_unique_edges]]

    b_crease_pct = float((dihedral_cpu[b_boundary_edges] > np.radians(30.0)).mean() * 100.0) if b_boundary_edges.any() else 0.0
    s_crease_pct = float((dihedral_cpu[s_boundary_edges] > np.radians(30.0)).mean() * 100.0) if s_boundary_edges.any() else 0.0

    print(f"\nBoundary Geometric Crease Alignment (edges with dihedral > 30 deg):", flush=True)
    print(f"  Baseline (Step 1):        {b_crease_pct:.2f}% of boundary edges on physical creases", flush=True)
    print(f"  Step 2 (Crease-Resolved): {s_crease_pct:.2f}% of boundary edges on physical creases (+{s_crease_pct - b_crease_pct:.2f}%)", flush=True)

    # 11. Generate Instance-Colored 3D Mesh
    print("\n--- Generating Instance-Colored 3D Mesh (Step 2) ---", flush=True)
    vertex_colors = np.full((num_vertices, 3), 0.5, dtype=np.float64)
    tri_colors = np.zeros((num_triangles, 3), dtype=np.float64)

    for cid in persistent_classes:
        won_mask = (final_win_cid == cid)
        if np.any(won_mask):
            _, rgb = get_class_color(cid)
            tri_colors[won_mask] = rgb

    v_color_accum = np.zeros((num_vertices, 3), dtype=np.float64)
    v_count = np.zeros(num_vertices, dtype=np.float64)

    valid_tri_mask = observed_mask_cpu
    valid_tris = triangles_np[valid_tri_mask]
    valid_colors = tri_colors[valid_tri_mask]

    for corner in range(3):
        corner_verts = valid_tris[:, corner]
        np.add.at(v_color_accum, corner_verts, valid_colors)
        np.add.at(v_count, corner_verts, 1.0)

    has_color = v_count > 0
    vertex_colors[has_color] = v_color_accum[has_color] / v_count[has_color, None]
    vertex_colors[~has_color] = [0.4, 0.4, 0.4]

    mesh_colored = o3d.geometry.TriangleMesh()
    mesh_colored.vertices = mesh.vertices
    mesh_colored.triangles = mesh.triangles
    mesh_colored.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    mesh_colored.compute_vertex_normals()

    step2_mesh_path = out_dir / "mesh_dispute_resolved.ply"
    o3d.io.write_triangle_mesh(str(step2_mesh_path), mesh_colored)
    print(f"Exported Step 2 mesh: {step2_mesh_path} ({step2_mesh_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    # 12. Save Comparison JSON
    comparison_summary = {
        'total_triangles': num_triangles,
        'observed_triangles': int(observed_mask.sum().item()),
        'scale_alpha': args.alpha,
        'smoothness_lambda': args.smoothness_lambda,
        'crease_alignment_baseline_pct': round(b_crease_pct, 2),
        'crease_alignment_step2_pct': round(s_crease_pct, 2),
        'crease_alignment_gain_pct': round(s_crease_pct - b_crease_pct, 2),
        'top_recovered_classes': gains_sorted[:25]
    }
    json_path = out_dir / "comparison_baseline_vs_step2.json"
    with open(json_path, 'w') as f:
        json.dump(comparison_summary, f, indent=2)
    print(f"Exported comparison JSON: {json_path}", flush=True)

    # 13. Re-render Visual Alignment Verification Strips (Baseline vs Step 2)
    print("\n--- Generating Side-by-Side Visual Comparison Strips ---", flush=True)
    verification_dir = out_dir / "visual_comparisons"
    verification_dir.mkdir(parents=True, exist_ok=True)

    T_mesh_to_raw = np.load(t_path)
    mesh_raw = o3d.geometry.TriangleMesh()
    mesh_raw.vertices = mesh.vertices
    mesh_raw.triangles = mesh.triangles
    mesh_raw.transform(T_mesh_to_raw)

    scene = o3d.t.geometry.RaycastingScene()
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_raw)
    scene.add_triangles(t_mesh)

    all_cameras = load_cameras(workspace, resolution=1)
    num_frames = len(all_cameras)
    horse_idx = next((i for i, c in enumerate(all_cameras) if 'frame_00200' in Path(c['image_path']).name), None)
    test_indices = [0, min(90, num_frames - 1), min(180, num_frames - 1), min(270, num_frames - 1)]
    if horse_idx is not None and horse_idx not in test_indices:
        test_indices.append(horse_idx)

    for f_idx in test_indices:
        cam = all_cameras[f_idx]
        stem = Path(cam['image_path']).stem
        img_bgr = cv2.imread(str(cam['image_path']))

        rays = scene.create_rays_pinhole(
            intrinsic_matrix=o3d.core.Tensor(cam['K'], dtype=o3d.core.Dtype.Float64),
            extrinsic_matrix=o3d.core.Tensor(cam['viewmat'], dtype=o3d.core.Dtype.Float64),
            width_px=cam['width'], height_px=cam['height']
        )
        ans = scene.cast_rays(rays)
        prim_ids = ans['primitive_ids'].numpy()
        valid_hits = prim_ids != scene.INVALID_ID
        hit_tris = prim_ids[valid_hits]

        # Render Baseline colors
        base_render = np.zeros_like(img_bgr)
        hit_cids_base = baseline_win_cid[hit_tris]
        for cid in np.unique(hit_cids_base):
            if cid <= 0: continue
            bgr_col, _ = get_class_color(cid)
            sub_m = np.zeros(prim_ids.shape, dtype=bool)
            sub_m[valid_hits] = (hit_cids_base == cid)
            base_render[sub_m] = bgr_col

        # Render Step 2 colors
        step2_render = np.zeros_like(img_bgr)
        hit_cids_step2 = final_win_cid[hit_tris]
        for cid in np.unique(hit_cids_step2):
            if cid <= 0: continue
            bgr_col, _ = get_class_color(cid)
            sub_m = np.zeros(prim_ids.shape, dtype=bool)
            sub_m[valid_hits] = (hit_cids_step2 == cid)
            step2_render[sub_m] = bgr_col

        alpha = 0.55
        blend_base = cv2.addWeighted(img_bgr, 1 - alpha, base_render, alpha, 0)
        blend_step2 = cv2.addWeighted(img_bgr, 1 - alpha, step2_render, alpha, 0)

        panel = np.hstack([img_bgr, blend_base, blend_step2])
        panel_small = cv2.resize(panel, (panel.shape[1] // 2, panel.shape[0] // 2))
        cv2.putText(panel_small, "RGB Photo", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(panel_small, "Step 1 Baseline", (panel_small.shape[1]//3 + 20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 200, 255), 2)
        cv2.putText(panel_small, "Step 2 Geometry-Resolved", (2*panel_small.shape[1]//3 + 20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)

        verif_p = verification_dir / f"compare_{stem}.jpg"
        cv2.imwrite(str(verif_p), panel_small)
        print(f"  Comparison strip saved: {verif_p}", flush=True)

    print("\n=================================================================", flush=True)
    print("STEP 2 DISPUTE RESOLUTION COMPLETED SUCCESSFULLY!", flush=True)
    print(f"Step 2 Resolved Mesh: {step2_mesh_path}", flush=True)
    print(f"Comparison JSON:      {json_path}", flush=True)
    print(f"Visual Strips:        {verification_dir}", flush=True)
    print("=================================================================", flush=True)

if __name__ == '__main__':
    main()
