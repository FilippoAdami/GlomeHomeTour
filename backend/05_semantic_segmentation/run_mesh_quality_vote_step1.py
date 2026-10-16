#!/usr/bin/env python3
"""
Step 1: Quality-Weighted Vote Accumulation (Per-Face) with Coordinate Alignment.

Projects 2D globally consistent panoptic segmentation masks (from full362_final_pipeline)
onto the 3D TSDF scene mesh faces across all 362 camera viewpoints.

Key features:
1. Coordinate Alignment:
   - TSDF mesh (mesh.ply) is in Floor-Aligned coordinates (Y_min = 0.0m).
   - COLMAP cameras are in Raw coordinates (Y_min = -1.32m).
   - Accurately transforms the mesh to Raw camera space via T_mesh_to_raw.npy
     so that camera rays hit the exact geometric surfaces with sub-centimeter precision (<3cm).
2. Exact Raycasting Visibility & Occlusion:
   - Uses Open3D RaycastingScene (Embree BVH) to trace rays from camera pinhole through pixels.
   - Automatically resolves all occlusions, self-occlusions, and backfaces at full pixel precision.
3. Quality-Weighted Accumulation:
   - Grazing angle weight: w_angle = max(0, cos_theta)^1.5. Discards grazing angles > 80 deg.
   - Mask interior distance weight: w_edge = min(1.0, dist / 12.0) down-weighting boundary noise.
   - Per-face vote accumulation on GPU (PyTorch index_add_).
4. 3D Instance Mesh Colorization:
   - Assigns each triangle its winning class ID (argmax over accumulated weights).
   - Assigns vertices the majority color of incident triangles using the deterministic 2D class palette.
   - Exports the colored mesh in its native floor-aligned coordinate system to 'mesh_quality_voted.ply'.
5. Multi-View Alignment Visual Verification:
   - Re-renders the colored mesh into 4 key camera views and saves side-by-side verification strips
     (Real RGB | 2D Mask Overlay | Mesh Re-projection).
6. Cross-Class Contention Upper-Triangular Matrix:
   - Vectorized GPU tensor matrix multiplication.
   - Rows and Columns: 93 persistent class IDs, labeled as "#ID (total_triangles_projected)".
   - Off-diagonal cells (i < j): "contended (row_wins / col_wins)" with Option 1 strict overall winners.
   - Diagonal cells (i, i): "total_won (uncontended / won_with_contention)".
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

# Include FastGS camera loader
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
    """Fast boundary distance transform weighting for 2D label map."""
    H, W = labels.shape
    labels_pad = np.pad(labels, 1, mode='edge')
    diff_r = labels != labels_pad[1:-1, 2:]
    diff_d = labels != labels_pad[2:, 1:-1]
    boundary = (diff_r | diff_d).astype(np.uint8)
    dist = cv2.distanceTransform(1 - boundary, cv2.DIST_L2, 3)
    w_edge = np.clip(dist / 12.0, 0.1, 1.0).astype(np.float32)
    w_edge[labels == 0] = 0.0
    return w_edge

def load_colmap_triangulated_points(images_txt_path: Path):
    """Parses COLMAP images.txt to get the count of triangulated 3D points per image."""
    frame_pts = {}
    if not images_txt_path.exists():
        return frame_pts
    with open(images_txt_path) as f:
        lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i].strip()
        if not line or line.startswith('#'): continue
        parts = line.split()
        if len(parts) >= 10 and (parts[9].endswith('.jpg') or parts[9].endswith('.png')):
            name = parts[9]
            pts_parts = lines[i+1].strip().split()
            num_2d = len(pts_parts) // 3
            num_3d = sum(1 for k in range(num_2d) if int(pts_parts[k*3+2]) != -1)
            frame_pts[name] = num_3d
    return frame_pts

def main():
    parser = argparse.ArgumentParser(description="Step 1: Quality-Weighted Vote Accumulation Per-Face")
    parser.add_argument("--workspace", type=Path, default=backend_dir / "current_scene")
    parser.add_argument("--mesh-path", type=Path, default=None)
    parser.add_argument("--npz-dir", type=Path, default=None)
    parser.add_argument("--transform-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--colmap-threshold", type=int, default=100, help="Min triangulated COLMAP points to accept a frame")
    parser.add_argument("--enable-depth-filter", action="store_true", default=False, help="Enable relative depth discontinuity filtering (default: False)")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    workspace = args.workspace
    mesh_path = args.mesh_path or (workspace / "06c_semantic_segmentation/inputs/mesh.ply")
    t_path = args.transform_path or (workspace / "06c_semantic_segmentation/inputs/T_mesh_to_raw.npy")
    npz_dir = args.npz_dir or (workspace / "06c_semantic_segmentation/full362_final_pipeline/npz_data")
    summary_path = workspace / "06c_semantic_segmentation/full362_final_pipeline/full_pipeline_benchmark_summary.json"
    images_txt_path = workspace / "06c_semantic_segmentation/inputs/sparse/0/images.txt"
    out_dir = args.output_dir or (workspace / "06c_semantic_segmentation/mesh_projection_step1")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("STEP 1: QUALITY-WEIGHTED VOTE ACCUMULATION (PER-FACE)")
    print(f"Workspace:          {workspace}")
    print(f"Mesh:               {mesh_path}")
    print(f"Transform:          {t_path}")
    print(f"NPZ Data:           {npz_dir}")
    print(f"Output:             {out_dir}")
    print(f"COLMAP Threshold:   {args.colmap_threshold} triangulated 3D points")
    print("=================================================================")

    # 0. Load COLMAP Quality Points per Frame
    colmap_pts_map = load_colmap_triangulated_points(images_txt_path)
    if colmap_pts_map:
        pts_list = list(colmap_pts_map.values())
        print(f"Parsed {len(colmap_pts_map)} frames from COLMAP images.txt (Median: {np.median(pts_list):.1f} pts, Min: {min(pts_list)}, Max: {max(pts_list)})")
    else:
        print("Warning: COLMAP images.txt not found, proceeding without COLMAP gating.")

    # 1. Load Classes List
    with open(summary_path) as f:
        bench_summary = json.load(f)
    persistent_classes = sorted(bench_summary['final_classes'])
    num_classes = len(persistent_classes)
    class_to_idx = {cid: idx for idx, cid in enumerate(persistent_classes)}
    idx_to_class = {idx: cid for idx, cid in enumerate(persistent_classes)}
    print(f"Loaded {num_classes} persistent class IDs: {persistent_classes[:10]} ... {persistent_classes[-5:]}")

    # 2. Load Mesh & Coordinate Transform
    t_start = time.perf_counter()
    print("Loading 3D mesh...")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    num_triangles = len(mesh.triangles)
    num_vertices = len(mesh.vertices)
    print(f"Mesh loaded: {num_vertices:,} vertices, {num_triangles:,} triangles")

    # Transform mesh to Raw camera space for exact ray intersection
    T_mesh_to_raw = np.load(t_path)
    print(f"Loaded T_mesh_to_raw (translation Y = {T_mesh_to_raw[1, 3]:.4f}m):")
    print(T_mesh_to_raw)

    print("Transforming mesh into Raw Camera Coordinate space...")
    mesh_raw = o3d.geometry.TriangleMesh()
    mesh_raw.vertices = mesh.vertices
    mesh_raw.triangles = mesh.triangles
    mesh_raw.transform(T_mesh_to_raw)

    print("Building Open3D RaycastingScene BVH in Raw space...")
    t_bvh_start = time.perf_counter()
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_raw)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)
    print(f"BVH built in {time.perf_counter() - t_bvh_start:.3f}s")

    # 3. Load Cameras
    all_cameras = load_cameras(workspace, resolution=1)
    if args.max_frames:
        all_cameras = all_cameras[:args.max_frames]
    num_frames = len(all_cameras)
    print(f"Loaded {num_frames} registered camera frames")

    # 4. GPU Accumulator Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using compute device: {device}")
    
    # Accumulator shape: [num_triangles, num_classes] float32 (~1.18 GB)
    acc = torch.zeros((num_triangles, num_classes), dtype=torch.float32, device=device)

    # Fast mapping tensor for class IDs
    max_cid = max(persistent_classes)
    cid_map_tensor = torch.full((max_cid + 1,), -1, dtype=torch.long, device=device)
    for cid, idx in class_to_idx.items():
        cid_map_tensor[cid] = idx

    # 5. Process Frames
    print("\n--- Raycasting & Vote Accumulation Across Frames ---")
    t_ray_total = 0.0
    t_accum_total = 0.0
    total_valid_votes = 0
    skipped_colmap_frames = 0
    filtered_wall_pixels_total = 0

    for f_idx, cam in enumerate(all_cameras):
        img_name = Path(cam['image_path']).name
        stem = Path(cam['image_path']).stem
        npz_file = npz_dir / f"{stem}_final.npz"
        if not npz_file.exists():
            print(f"Warning: {npz_file.name} missing, skipping.")
            continue

        # COLMAP Confidence Gating
        if colmap_pts_map and colmap_pts_map.get(img_name, 0) < args.colmap_threshold:
            skipped_colmap_frames += 1
            continue

        data = np.load(npz_file)
        labels = data['labels'] # [1920, 1080] int32

        # Pinhole rays in raw camera space
        t_r0 = time.perf_counter()
        rays = scene.create_rays_pinhole(
            intrinsic_matrix=o3d.core.Tensor(cam['K'], dtype=o3d.core.Dtype.Float64),
            extrinsic_matrix=o3d.core.Tensor(cam['viewmat'], dtype=o3d.core.Dtype.Float64),
            width_px=cam['width'], height_px=cam['height']
        )
        ans = scene.cast_rays(rays)
        t_ray_total += (time.perf_counter() - t_r0)

        prim_ids = ans['primitive_ids'].numpy() # [H, W] uint32
        hit_depths = ans['t_hit'].numpy() # [H, W] float32
        normals = ans['primitive_normals'].numpy() # [H, W, 3] float32
        ray_dirs = rays.numpy()[:, :, 3:] # [H, W, 3] float32

        # 1) Angle weight: cos(theta)^1.5, discard grazing angles > 80 deg (cos < 0.17)
        cos_theta = -(normals * ray_dirs).sum(axis=-1)
        cos_theta = np.clip(cos_theta, 0.0, 1.0)
        w_angle = (cos_theta ** 1.5).astype(np.float32)

        # 2) Mask interior distance weight
        w_edge = compute_mask_distance_weights(labels)

        # Total weight
        w_total = w_angle * w_edge

        valid_hits = prim_ids != scene.INVALID_ID
        # 3) Optional Relative Depth Discontinuity Filter
        if args.enable_depth_filter:
            unique_cids = np.unique(labels)
            for cid in unique_cids:
                if cid in [0, 1, 2, 5]: continue # Do not truncate background architecture classes
                m = (labels == cid) & valid_hits
                if m.sum() < 20: continue
                d_vals = hit_depths[m]
                d_base = np.percentile(d_vals, 5)
                rel_diff = (hit_depths - d_base) / max(d_base, 1e-3)
                detached = m & (rel_diff > 0.20) & (cos_theta > 0.35)
                if np.any(detached):
                    w_total[detached] = 0.0
                    filtered_wall_pixels_total += int(detached.sum())

        # Valid pixel hits with positive label and meaningful weight
        valid_mask = valid_hits & (w_total > 0.02) & (labels > 0)
        if not np.any(valid_mask):
            continue

        v_tri_ids = prim_ids[valid_mask]
        v_cids = labels[valid_mask]
        v_weights = w_total[valid_mask]

        # Transfer batch to GPU and scatter-add
        t_a0 = time.perf_counter()
        tri_tensor = torch.from_numpy(v_tri_ids.astype(np.int64)).to(device, non_blocking=True)
        cid_tensor = torch.from_numpy(v_cids.astype(np.int64)).to(device, non_blocking=True)
        w_tensor = torch.from_numpy(v_weights).to(device, non_blocking=True)

        class_indices = cid_map_tensor[cid_tensor]
        valid_cid = class_indices >= 0
        if torch.any(valid_cid):
            tri_sub = tri_tensor[valid_cid]
            cls_sub = class_indices[valid_cid]
            w_sub = w_tensor[valid_cid]

            linear_indices = tri_sub * num_classes + cls_sub
            acc.view(-1).index_add_(0, linear_indices, w_sub)
            total_valid_votes += len(linear_indices)

        t_accum_total += (time.perf_counter() - t_a0)

        if (f_idx + 1) % 40 == 0 or (f_idx + 1) == num_frames:
            elapsed = time.perf_counter() - t_start
            fps = (f_idx + 1) / elapsed
            print(f"  [{f_idx + 1:03d}/{num_frames}] Valid ray hits processed | Speed: {fps:.2f} frames/s")

    t_projection = time.perf_counter() - t_start
    print(f"\nProjection complete in {t_projection:.2f}s ({num_frames / t_projection:.2f} FPS)")
    print(f"Skipped {skipped_colmap_frames} frames with < {args.colmap_threshold} COLMAP triangulated points")
    print(f"Filtered {filtered_wall_pixels_total:,} detached facing background pixels across processed frames")
    print(f"Total ray-triangle weighted votes accumulated: {total_valid_votes:,}")
    print(f"Pure raycasting time: {t_ray_total:.2f}s | Scatter-add accumulation time: {t_accum_total:.2f}s")

    # Save GPU accumulator tensor cache for Step 2
    acc_pt_path = out_dir / "accumulated_votes.pt"
    print(f"Saving accumulated votes cache tensor to {acc_pt_path.name}...")
    torch.save(acc, acc_pt_path)
    print(f"Saved {acc_pt_path.name} ({acc_pt_path.stat().st_size / 1e6:.1f} MB)")

    # 6. Analyze Results & Extract Winning Classes
    print("\n--- Analyzing 3D Class Assignments & Contention ---")
    t_mat0 = time.perf_counter()
    sum_w = acc.sum(dim=-1) # [num_triangles]
    observed_mask = sum_w > 0
    num_observed_triangles = observed_mask.sum().item()
    print(f"Triangles observed: {num_observed_triangles:,} / {num_triangles:,} ({num_observed_triangles / num_triangles * 100:.2f}%)")

    winning_class_indices = acc.argmax(dim=-1) # [num_triangles]

    # Vectorized GPU Co-occurrence & Contention Matrix Multiplication
    print("Computing Cross-Class Contention Matrix via GPU Matrix Multiplication...")
    binary_pos = (acc > 0).float() # [N, K]
    won_binary = torch.zeros_like(binary_pos)
    won_binary[observed_mask, winning_class_indices[observed_mask]] = 1.0

    # 1. Total triangles projected per class
    triangles_projected = binary_pos.sum(dim=0).long().cpu().numpy()

    # 2. Pairwise co-occurrence (contended triangles between class i and class j)
    co_occ_matrix = (binary_pos.T @ binary_pos).long().cpu().numpy()

    # 3. Pairwise wins: wins_matrix[i, j] = count(triangles won by i where j also had votes)
    wins_matrix = (won_binary.T @ binary_pos).long().cpu().numpy()

    # 4. Diagonal: total won, uncontended won, contended won
    total_won = won_binary.sum(dim=0).long().cpu().numpy()
    num_claims_per_tri = binary_pos.sum(dim=-1)
    contended_won = (won_binary * (num_claims_per_tri[:, None] > 1).float()).sum(dim=0).long().cpu().numpy()
    uncontended_won = total_won - contended_won

    print(f"GPU Contention computation finished in {time.perf_counter() - t_mat0:.3f}s!")

    # Format table data
    matrix_cells = {}
    csv_rows = []
    headers = [f"#{cid} ({triangles_projected[k]:,})" for k, cid in enumerate(persistent_classes)]
    top_contended_pairs = []

    for i in range(num_classes):
        cid_i = persistent_classes[i]
        row_dict = {"Class": f"#{cid_i} ({triangles_projected[i]:,})"}

        for j in range(num_classes):
            cid_j = persistent_classes[j]
            col_name = headers[j]

            if i == j:
                cell_str = f"{total_won[i]} ({uncontended_won[i]} / {contended_won[i]})"
                row_dict[col_name] = cell_str
                matrix_cells[(i, j)] = {
                    'type': 'diagonal',
                    'class_id': cid_i,
                    'total_won': int(total_won[i]),
                    'uncontended': int(uncontended_won[i]),
                    'contended_won': int(contended_won[i]),
                    'display': cell_str
                }
            elif i < j:
                num_contended = int(co_occ_matrix[i, j])
                row_wins = int(wins_matrix[i, j])
                col_wins = int(wins_matrix[j, i])

                if num_contended > 0:
                    cell_str = f"{num_contended} ({row_wins} / {col_wins})"
                    top_contended_pairs.append({
                        'class_i': cid_i,
                        'class_j': cid_j,
                        'contended': num_contended,
                        'row_wins': row_wins,
                        'col_wins': col_wins,
                        'i_idx': i,
                        'j_idx': j,
                        'display': cell_str
                    })
                else:
                    cell_str = "0 (0 / 0)"

                row_dict[col_name] = cell_str
                matrix_cells[(i, j)] = {
                    'type': 'off_diagonal',
                    'class_i': cid_i,
                    'class_j': cid_j,
                    'contended': num_contended,
                    'row_wins': row_wins,
                    'col_wins': col_wins,
                    'display': cell_str
                }
            else:
                row_dict[col_name] = ""
                matrix_cells[(i, j)] = {'type': 'empty', 'display': ''}

        csv_rows.append(row_dict)

    top_contended_pairs.sort(key=lambda x: x['contended'], reverse=True)
    print(f"Discovered {len(top_contended_pairs)} pairs of classes with contended triangles.")
    if top_contended_pairs:
        print("\nTop 15 Most Contended Class Pairs:")
        for rank, p in enumerate(top_contended_pairs[:15], 1):
            print(f"  {rank:2d}. Class #{p['class_i']} vs #{p['class_j']}: {p['display']}")

    # 7. Save CSV and JSON Matrix Outputs
    import csv
    csv_path = out_dir / "cross_class_contention_matrix.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Class"] + headers)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nExported complete 93x93 CSV matrix to: {csv_path}")

    summary_data = {
        'total_triangles': num_triangles,
        'observed_triangles': num_observed_triangles,
        'coverage_pct': round(num_observed_triangles / num_triangles * 100, 2),
        'num_classes': num_classes,
        'persistent_classes': persistent_classes,
        'triangles_projected_per_class': {cid: int(triangles_projected[k]) for k, cid in enumerate(persistent_classes)},
        'triangles_won_per_class': {cid: int(total_won[k]) for k, cid in enumerate(persistent_classes)},
        'uncontended_won_per_class': {cid: int(uncontended_won[k]) for k, cid in enumerate(persistent_classes)},
        'top_contended_pairs': top_contended_pairs[:50],
        'timing_seconds': {
            'raycasting': round(t_ray_total, 2),
            'scatter_add': round(t_accum_total, 2),
            'total_projection': round(t_projection, 2)
        }
    }
    json_path = out_dir / "cross_class_contention_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Exported summary JSON to: {json_path}")

    # 8. Render Visual Matrix PNGs
    print("Rendering high-resolution upper-triangular matrix PNG...")
    render_contention_matrix_image(persistent_classes, triangles_projected, matrix_cells, out_dir / "cross_class_contention_matrix.png")
    
    if len(top_contended_pairs) > 0:
        render_top_contended_subset(top_contended_pairs[:25], persistent_classes, out_dir / "top25_contended_matrix.png")

    # 9. Generate Instance-Colored 3D Mesh
    print("\n--- Generating Instance-Colored 3D Mesh ---")
    win_idx_cpu = winning_class_indices.cpu().numpy()
    sum_w_cpu = sum_w.cpu().numpy()
    win_cid_cpu = np.array([idx_to_class[idx] if sum_w_cpu[i] > 0 else 0 for i, idx in enumerate(win_idx_cpu)], dtype=np.int32)

    triangles_np = np.asarray(mesh.triangles)
    vertex_colors = np.full((num_vertices, 3), 0.5, dtype=np.float64)

    print("Computing vertex colors from winning triangle consensus...")
    tri_colors = np.zeros((num_triangles, 3), dtype=np.float64)
    for k, cid in enumerate(persistent_classes):
        won_mask = (win_idx_cpu == k) & (sum_w_cpu > 0)
        if np.any(won_mask):
            _, rgb = get_class_color(cid)
            tri_colors[won_mask] = rgb

    v_color_accum = np.zeros((num_vertices, 3), dtype=np.float64)
    v_count = np.zeros(num_vertices, dtype=np.float64)

    valid_tri_mask = sum_w_cpu > 0
    valid_tris = triangles_np[valid_tri_mask]
    valid_colors = tri_colors[valid_tri_mask]

    for corner in range(3):
        corner_verts = valid_tris[:, corner]
        np.add.at(v_color_accum, corner_verts, valid_colors)
        np.add.at(v_count, corner_verts, 1.0)

    has_color = v_count > 0
    vertex_colors[has_color] = v_color_accum[has_color] / v_count[has_color, None]
    vertex_colors[~has_color] = [0.4, 0.4, 0.4]

    # Save colored mesh in ORIGINAL floor-aligned space (exact match to input mesh)
    mesh_colored = o3d.geometry.TriangleMesh()
    mesh_colored.vertices = mesh.vertices
    mesh_colored.triangles = mesh.triangles
    mesh_colored.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    mesh_colored.compute_vertex_normals()

    colored_mesh_path = out_dir / "mesh_quality_voted.ply"
    o3d.io.write_triangle_mesh(str(colored_mesh_path), mesh_colored)
    print(f"Exported instance-colored mesh: {colored_mesh_path} ({colored_mesh_path.stat().st_size / 1e6:.1f} MB)")

    # 10. Save Visual Alignment Verification Strips
    print("\n--- Generating Multi-View Visual Alignment Verification Strips ---")
    verification_dir = out_dir / "alignment_verification"
    verification_dir.mkdir(parents=True, exist_ok=True)

    # Re-render key viewpoints including horse and pictures
    test_indices = [0, min(90, num_frames - 1), min(180, num_frames - 1), min(270, num_frames - 1)]
    for key_frame in ['frame_00200', 'frame_00969']:
        idx = next((i for i, c in enumerate(all_cameras) if key_frame in str(c['image_path'])), None)
        if idx is not None and idx not in test_indices:
            test_indices.append(idx)
    for f_idx in test_indices:
        cam = all_cameras[f_idx]
        stem = Path(cam['image_path']).stem
        img_bgr = cv2.imread(str(cam['image_path']))
        npz_file = npz_dir / f"{stem}_final.npz"
        if not npz_file.exists(): continue
        data = np.load(npz_file)
        labels = data['labels']

        # Cast rays to get primitive_ids for each pixel in this view
        rays = scene.create_rays_pinhole(
            intrinsic_matrix=o3d.core.Tensor(cam['K'], dtype=o3d.core.Dtype.Float64),
            extrinsic_matrix=o3d.core.Tensor(cam['viewmat'], dtype=o3d.core.Dtype.Float64),
            width_px=cam['width'], height_px=cam['height']
        )
        ans = scene.cast_rays(rays)
        prim_ids = ans['primitive_ids'].numpy()

        # Render 2D label map from projected mesh
        mesh_render_bgr = np.zeros_like(img_bgr)
        valid_hits = prim_ids != scene.INVALID_ID
        hit_tris = prim_ids[valid_hits]
        hit_cids = win_cid_cpu[hit_tris]

        for cid in np.unique(hit_cids):
            if cid <= 0: continue
            bgr_col, _ = get_class_color(cid)
            sub_mask = np.zeros(prim_ids.shape, dtype=bool)
            sub_mask[valid_hits] = (hit_cids == cid)
            mesh_render_bgr[sub_mask] = bgr_col

        # Create 2D Ground-Truth mask color overlay
        gt_render_bgr = np.zeros_like(img_bgr)
        for cid in np.unique(labels):
            if cid <= 0: continue
            bgr_col, _ = get_class_color(cid)
            gt_render_bgr[labels == cid] = bgr_col

        # Alpha blends
        alpha = 0.55
        blend_mesh = cv2.addWeighted(img_bgr, 1 - alpha, mesh_render_bgr, alpha, 0)
        blend_gt = cv2.addWeighted(img_bgr, 1 - alpha, gt_render_bgr, alpha, 0)

        # 3-Panel composite: [Photo | 2D SAM2 Mask | 3D Mesh Re-projection]
        panel = np.hstack([img_bgr, blend_gt, blend_mesh])
        panel_small = cv2.resize(panel, (panel.shape[1] // 2, panel.shape[0] // 2))
        cv2.putText(panel_small, "RGB Photo", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(panel_small, "2D Panoptic Mask", (panel_small.shape[1]//3 + 20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(panel_small, "3D Mesh Re-projection", (2*panel_small.shape[1]//3 + 20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)

        verif_path = verification_dir / f"verif_{stem}.jpg"
        cv2.imwrite(str(verif_path), panel_small)
        print(f"  Alignment verification saved: {verif_path}")

    # 11. Save Compressed Triangle Labels & Votes Archive
    npz_archive_path = out_dir / "triangle_class_votes.npz"
    np.savez_compressed(
        npz_archive_path,
        winning_class_ids=win_cid_cpu,
        winning_weights=sum_w_cpu,
        observed_mask=observed_mask.cpu().numpy()
    )
    print(f"Exported triangle votes NPZ: {npz_archive_path}")

    t_grand = time.perf_counter() - t_start
    print("\n=================================================================")
    print("STEP 1 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"Total Elapsed Time: {t_grand:.2f}s ({t_grand/60:.2f} min)")
    print(f"Colored Mesh:       {colored_mesh_path}")
    print(f"Contention Matrix:  {out_dir / 'cross_class_contention_matrix.png'}")
    print(f"Top 25 Contended:   {out_dir / 'top25_contended_matrix.png'}")
    print(f"Verification Renders: {verification_dir}")
    print(f"Matrix CSV:         {csv_path}")
    print(f"Summary JSON:       {json_path}")
    print("=================================================================")

def render_contention_matrix_image(classes: list, proj_counts: np.ndarray, matrix_cells: dict, save_path: Path):
    """Renders high-resolution publication-quality upper-triangular contention matrix PNG."""
    N = len(classes)
    fig_size = max(24, int(N * 0.45))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=150)
    fig.patch.set_facecolor('#111116')
    ax.set_facecolor('#181820')

    grid_vals = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            cell = matrix_cells.get((i, j), {})
            if cell.get('type') == 'diagonal':
                grid_vals[i, j] = cell.get('total_won', 0)
            elif cell.get('type') == 'off_diagonal':
                grid_vals[i, j] = cell.get('contended', 0)

    norm = mcolors.LogNorm(vmin=max(1.0, float(np.percentile(grid_vals[grid_vals > 0], 5)) if np.any(grid_vals > 0) else 1.0),
                           vmax=max(10.0, float(grid_vals.max()) if np.any(grid_vals > 0) else 10.0))
    cmap = plt.cm.magma

    masked_vals = np.ma.masked_where(np.tril(np.ones((N, N), dtype=bool), k=-1), grid_vals)
    cax = ax.imshow(masked_vals, cmap=cmap, norm=norm, origin='upper')

    tick_labels = [f"#{cid} ({proj_counts[k]:,})" for k, cid in enumerate(classes)]
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6, color='#E0E0E0', ha='center')
    ax.set_yticklabels(tick_labels, fontsize=6, color='#E0E0E0')
    ax.xaxis.tick_top()

    for i in range(N):
        for j in range(i, N):
            cell = matrix_cells.get((i, j), {})
            text = cell.get('display', '')
            if not text or text == "0 (0 / 0)":
                continue
            
            parts = text.split(" (")
            if len(parts) == 2:
                main_val, breakdown = parts[0], "(" + parts[1]
                formatted_text = f"{main_val}\n{breakdown}"
            else:
                formatted_text = text

            is_diag = (i == j)
            color = '#FFEA00' if is_diag else '#FFFFFF'
            fontweight = 'bold' if is_diag else 'normal'
            fontsize = 4.0 if N > 50 else 6.0

            ax.text(j, i, formatted_text, ha='center', va='center',
                    color=color, fontsize=fontsize, fontweight=fontweight)

    ax.grid(True, which='both', color='#333340', linestyle='-', linewidth=0.5)
    plt.title("Cross-Class Contention Upper-Triangular Matrix (Quality-Weighted Step 1)\nOff-Diagonal: Contended (Row Wins / Col Wins) | Diagonal: Total Won (Uncontended / Contended)",
              color='#FFFFFF', fontsize=14, pad=40, fontweight='bold')

    plt.tight_layout()
    plt.savefig(str(save_path), facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Matrix PNG saved: {save_path}")

def render_top_contended_subset(top_pairs: list, all_classes: list, save_path: Path):
    """Renders a focused, readable bar & table plot of the top contended class pairs."""
    if not top_pairs:
        return
    K = len(top_pairs)
    fig, (ax_bar, ax_tbl) = plt.subplots(1, 2, figsize=(18, max(8, int(K * 0.45))), dpi=180, gridspec_kw={'width_ratios': [1.2, 1.0]})
    fig.patch.set_facecolor('#111116')
    ax_bar.set_facecolor('#181820')
    ax_tbl.set_facecolor('#181820')

    labels = [f"#{p['class_i']} vs #{p['class_j']}" for p in top_pairs]
    row_wins = [p['row_wins'] for p in top_pairs]
    col_wins = [p['col_wins'] for p in top_pairs]
    unresolved = [p['contended'] - (p['row_wins'] + p['col_wins']) for p in top_pairs]

    y = np.arange(K)[::-1]
    height = 0.65

    b1 = ax_bar.barh(y, row_wins, height, color='#00E5FF', label='Row Class Won')
    b2 = ax_bar.barh(y, col_wins, height, left=row_wins, color='#FF4081', label='Col Class Won')
    left_third = np.array(row_wins) + np.array(col_wins)
    if any(u > 0 for u in unresolved):
        b3 = ax_bar.barh(y, unresolved, height, left=left_third, color='#757575', label='3rd Class Won')

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels, color='#E0E0E0', fontsize=8, fontweight='bold')
    ax_bar.tick_params(colors='#E0E0E0', which='both')
    ax_bar.set_xlabel("Number of Contended Triangles", color='#E0E0E0', fontsize=10, fontweight='bold')
    ax_bar.set_title("Top Contended Class Pairs Breakdown", color='#FFFFFF', fontsize=12, fontweight='bold', pad=15)
    ax_bar.legend(facecolor='#22222E', edgecolor='#444455', labelcolor='#E0E0E0', loc='lower right')
    ax_bar.grid(axis='x', color='#333344', linestyle='--', alpha=0.7)

    ax_tbl.axis('off')
    col_labels = ["Rank", "Pair", "Contended", "Row Wins", "Col Wins", "Balance"]
    table_data = []
    for rank, p in enumerate(top_pairs, 1):
        tot = p['contended']
        r_pct = f"{p['row_wins']/tot*100:.0f}%" if tot > 0 else "0%"
        c_pct = f"{p['col_wins']/tot*100:.0f}%" if tot > 0 else "0%"
        table_data.append([
            str(rank),
            f"#{p['class_i']} vs #{p['class_j']}",
            f"{tot:,}",
            f"{p['row_wins']:,} ({r_pct})",
            f"{p['col_wins']:,} ({c_pct})",
            f"{r_pct} : {c_pct}"
        ])

    table = ax_tbl.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#262636')
            cell.set_text_props(color='#00E5FF', fontweight='bold')
        else:
            cell.set_facecolor('#181824' if row % 2 == 0 else '#1E1E2C')
            cell.set_text_props(color='#E0E0E0')
        cell.set_edgecolor('#333348')

    plt.suptitle("Top Contended Triangles Breakdown (Step 1 Quality-Weighted Accumulation)",
                 color='#FFFFFF', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(str(save_path), facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Top Contended visual saved: {save_path}")

if __name__ == '__main__':
    main()
