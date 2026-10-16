#!/usr/bin/env python3
"""
Step 3: 3D Object Extraction, Floater Pruning (SOR), and Bounding Box Embedding.

Features:
1. Dynamic Floater & Outlier Pruning:
   - Dynamic size thresholding based on preemptive floater analysis.
   - Filters out non-solid floaters, spatial outliers (>1.8m detached), and wispy strings.
2. 3D Bounding Box Generation & Mesh Embedding:
   - Bounding box contour edges colored with the class color.
   - Bounding box faces semi-transparent (0.2 opacity / alpha=51).
   - 3D class ID (#ID) text embedded at the top-right corner of each box.
   - Exported directly into 'mesh_with_bounding_boxes.ply'.
3. Class-Instance Grid Table:
   - Exports 'class_instance_table.csv' listing all classes with columns for each instance.
4. Individual Object Sub-Mesh Export:
   - Exports clean sub-meshes under 'extracted_instances/'.
"""

import os, sys, json, time, argparse
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import open3d as o3d
import cv2

backend_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_dir / "03_FastGS_DNSplatter"))
from gsplat_train import load_cameras

def get_class_color(mid: int):
    if mid <= 0:
        return (128, 128, 128), (0.5, 0.5, 0.5)
    k = mid - 1
    b = (37 * k + 93) % 255
    g = (109 * k + 71) % 255
    r = (197 * k + 31) % 255
    col_bgr = (int(b), int(g), int(r))
    col_rgb = (r / 255.0, g / 255.0, b / 255.0)
    return col_bgr, col_rgb

# Vector stroke font for '#', '0'-'9' on unit square [0, 1] x [0, 1]
STROKES = {
    '#': [((0.25, 0.05), (0.25, 0.95)), ((0.75, 0.05), (0.75, 0.95)), ((0.05, 0.35), (0.95, 0.35)), ((0.05, 0.65), (0.95, 0.65))],
    '0': [((0.1, 0.05), (0.9, 0.05)), ((0.9, 0.05), (0.9, 0.95)), ((0.9, 0.95), (0.1, 0.95)), ((0.1, 0.95), (0.1, 0.05)), ((0.1, 0.05), (0.9, 0.95))],
    '1': [((0.5, 0.05), (0.5, 0.95)), ((0.2, 0.75), (0.5, 0.95)), ((0.2, 0.05), (0.8, 0.05))],
    '2': [((0.1, 0.95), (0.9, 0.95)), ((0.9, 0.95), (0.9, 0.55)), ((0.9, 0.55), (0.1, 0.55)), ((0.1, 0.55), (0.1, 0.05)), ((0.1, 0.05), (0.9, 0.05))],
    '3': [((0.1, 0.95), (0.9, 0.95)), ((0.9, 0.95), (0.9, 0.05)), ((0.9, 0.05), (0.1, 0.05)), ((0.1, 0.55), (0.9, 0.55))],
    '4': [((0.1, 0.95), (0.1, 0.50)), ((0.1, 0.50), (0.9, 0.50)), ((0.8, 0.95), (0.8, 0.05))],
    '5': [((0.9, 0.95), (0.1, 0.95)), ((0.1, 0.95), (0.1, 0.55)), ((0.1, 0.55), (0.9, 0.55)), ((0.9, 0.55), (0.9, 0.05)), ((0.9, 0.05), (0.1, 0.05))],
    '6': [((0.9, 0.95), (0.1, 0.95)), ((0.1, 0.95), (0.1, 0.05)), ((0.1, 0.05), (0.9, 0.05)), ((0.9, 0.05), (0.9, 0.55)), ((0.9, 0.55), (0.1, 0.55))],
    '7': [((0.1, 0.95), (0.9, 0.95)), ((0.9, 0.95), (0.3, 0.05))],
    '8': [((0.1, 0.05), (0.9, 0.05)), ((0.9, 0.05), (0.9, 0.95)), ((0.9, 0.95), (0.1, 0.95)), ((0.1, 0.95), (0.1, 0.05)), ((0.1, 0.50), (0.9, 0.50))],
    '9': [((0.1, 0.05), (0.9, 0.05)), ((0.9, 0.05), (0.9, 0.95)), ((0.9, 0.95), (0.1, 0.95)), ((0.1, 0.95), (0.1, 0.50)), ((0.1, 0.50), (0.9, 0.50))]
}

def create_cylinder_mesh(p1: np.ndarray, p2: np.ndarray, radius: float = 0.005, color_rgb=(1.0, 1.0, 0.0), alpha_val=255):
    vec = p2 - p1
    length = float(np.linalg.norm(vec))
    if length < 1e-5:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32), np.zeros((0, 4), dtype=np.uint8)
    
    dir_v = vec / length
    up = np.array([0.0, 0.0, 1.0]) if abs(dir_v[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
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

    col_rgba = np.array([int(color_rgb[0] * 255), int(color_rgb[1] * 255), int(color_rgb[2] * 255), int(alpha_val)], dtype=np.uint8)
    colors = np.tile(col_rgba, (len(verts), 1))
    return verts, faces, colors

def build_3d_text_mesh(text: str, origin: np.ndarray, scale: float = 0.08, color_rgb=(1.0, 1.0, 1.0)):
    all_v, all_f, all_c = [], [], []
    v_offset = 0
    cur_x = origin[0]
    y_base = origin[1]
    z_base = origin[2]

    for char in text:
        strokes = STROKES.get(char, [])
        for p_start, p_end in strokes:
            p1 = np.array([cur_x + p_start[0] * scale * 0.7, y_base, z_base + p_start[1] * scale])
            p2 = np.array([cur_x + p_end[0] * scale * 0.7, y_base, z_base + p_end[1] * scale])
            vs, fs, cs = create_cylinder_mesh(p1, p2, radius=0.003, color_rgb=color_rgb, alpha_val=255)
            if len(vs) == 0: continue
            all_v.append(vs)
            all_f.append(fs + v_offset)
            all_c.append(cs)
            v_offset += len(vs)
        cur_x += scale * 0.85

    if len(all_v) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32), np.zeros((0, 4), dtype=np.uint8)
    return np.vstack(all_v), np.vstack(all_f), np.vstack(all_c)

def compute_robust_obb(vertices: np.ndarray):
    pts_clean = vertices.copy()
    cov = np.cov(pts_clean - pts_clean.mean(axis=0), rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    if evals[0] < 1e-5:  # flat / coplanar object (wall picture, calendar, switch)
        normal = evecs[:, 0]
        thick_pts = np.vstack([pts_clean + normal * 0.005, pts_clean - normal * 0.005])
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(thick_pts))
    else:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_clean))

    try:
        obb = pcd.get_minimal_oriented_bounding_box()
        center = np.asarray(obb.center)
        extents = np.asarray(obb.extent)
        R = np.asarray(obb.R)
    except Exception:
        centroid = pts_clean.mean(axis=0)
        pts_centered = pts_clean - centroid
        cov = np.cov(pts_centered, rowvar=False)
        _, evecs = np.linalg.eigh(cov)
        R = evecs
        if np.linalg.det(R) < 0: R[:, 0] = -R[:, 0]
        proj = pts_centered @ R
        min_p = proj.min(axis=0)
        max_p = proj.max(axis=0)
        extents = max_p - min_p
        center = centroid + ((min_p + max_p) / 2.0) @ R.T

    if np.linalg.det(R) < 0:
        R[:, 0] = -R[:, 0]

    half = extents / 2.0
    dx, dy, dz = half
    local_corners = np.array([
        [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz], # Bottom 0..3
        [-dx, -dy, dz],  [dx, -dy, dz],  [dx, dy, dz],  [-dx, dy, dz]   # Top 4..7
    ])
    world_corners = center + local_corners @ R.T
    return world_corners, extents, center, R

def build_bounding_box_geometry(corners: np.ndarray, color_rgb, class_id: int):
    box_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), # bottom loop
        (4, 5), (5, 6), (6, 7), (7, 4), # top loop
        (0, 4), (1, 5), (2, 6), (3, 7)  # vertical struts
    ]

    all_v, all_f, all_c = [], [], []
    v_off = 0

    # 1. Edges (opaque wireframe struts)
    for i_a, i_b in box_edges:
        vs, fs, cs = create_cylinder_mesh(corners[i_a], corners[i_b], radius=0.006, color_rgb=color_rgb, alpha_val=255)
        if len(vs) == 0: continue
        all_v.append(vs)
        all_f.append(fs + v_off)
        all_c.append(cs)
        v_off += len(vs)

    # 2. Semi-transparent Faces (12 triangles, alpha=51 / 0.2 opacity)
    face_quads = [
        (0, 3, 2, 1), # -Z
        (4, 5, 6, 7), # +Z
        (0, 1, 5, 4), # -Y
        (2, 3, 7, 6), # +Y
        (0, 4, 7, 3), # -X
        (1, 2, 6, 5)  # +X
    ]
    face_verts = corners.copy()
    face_tris = []
    for q in face_quads:
        face_tris.append([q[0], q[1], q[2]])
        face_tris.append([q[0], q[2], q[3]])
    face_tris = np.array(face_tris, dtype=np.int32)
    face_rgba = np.array([int(color_rgb[0] * 255), int(color_rgb[1] * 255), int(color_rgb[2] * 255), 51], dtype=np.uint8)
    face_colors = np.tile(face_rgba, (len(face_verts), 1))

    all_v.append(face_verts)
    all_f.append(face_tris + v_off)
    all_c.append(face_colors)
    v_off += len(face_verts)

    # 3. 3D Text Tag "#ID" on highest elevation / top corner
    top_corner_idx = corners[:, 1].argmax()
    tag_pos = corners[top_corner_idx] + np.array([0.01, 0.02, 0.01])
    text_v, text_f, text_c = build_3d_text_mesh(f"#{class_id}", tag_pos, scale=0.08, color_rgb=color_rgb)
    if len(text_v) > 0:
        all_v.append(text_v)
        all_f.append(text_f + v_off)
        all_c.append(text_c)

    return np.vstack(all_v), np.vstack(all_f), np.vstack(all_c)

def export_binary_rgba_ply(file_path: Path, vertices: np.ndarray, faces: np.ndarray, colors_rgba: np.ndarray):
    num_verts = len(vertices)
    num_faces = len(faces)

    v_data = np.zeros(num_verts, dtype=[
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1')
    ])
    v_data['x'] = vertices[:, 0]
    v_data['y'] = vertices[:, 1]
    v_data['z'] = vertices[:, 2]
    v_data['red'] = colors_rgba[:, 0]
    v_data['green'] = colors_rgba[:, 1]
    v_data['blue'] = colors_rgba[:, 2]
    v_data['alpha'] = colors_rgba[:, 3]

    f_data = np.zeros(num_faces, dtype=[
        ('n_verts', 'u1'), ('v0', '<i4'), ('v1', '<i4'), ('v2', '<i4')
    ])
    f_data['n_verts'] = 3
    f_data['v0'] = faces[:, 0]
    f_data['v1'] = faces[:, 1]
    f_data['v2'] = faces[:, 2]

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
        "property uchar alpha\n"
        f"element face {num_faces}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )

    with open(file_path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(v_data.tobytes())
        f.write(f_data.tobytes())

def main():
    parser = argparse.ArgumentParser(description="Step 3: 3D Object Extraction & Filtering")
    parser.add_argument("--workspace", type=Path, default=backend_dir / "current_scene")
    parser.add_argument("--mesh-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    workspace = args.workspace
    mesh_path = args.mesh_path or (workspace / "06c_semantic_segmentation/mesh_dispute_resolution_step2/mesh_dispute_resolved.ply")
    summary_path = workspace / "06c_semantic_segmentation/full362_final_pipeline/full_pipeline_benchmark_summary.json"
    out_dir = args.output_dir or (workspace / "06c_semantic_segmentation/mesh_objects_step3")
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_export_dir = out_dir / "extracted_instances"
    obj_export_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================", flush=True)
    print("STEP 3: 3D OBJECT EXTRACTION, FLOATER PRUNING & BOUNDING BOX GENERATION", flush=True)
    print(f"Workspace:         {workspace}", flush=True)
    print(f"Source Mesh:       {mesh_path}", flush=True)
    print(f"Output Directory:  {out_dir}", flush=True)
    print("=================================================================", flush=True)

    with open(summary_path) as f:
        bench_summary = json.load(f)
    persistent_classes = sorted(bench_summary['final_classes'])

    print("Loading geometry-resolved 3D mesh...", flush=True)
    t0 = time.perf_counter()
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices_np = np.asarray(mesh.vertices)
    triangles_np = np.asarray(mesh.triangles)
    num_triangles = len(triangles_np)
    num_vertices = len(vertices_np)
    vertex_colors_np = np.asarray(mesh.vertex_colors)
    print(f"Loaded mesh: {num_vertices:,} vertices, {num_triangles:,} triangles in {time.perf_counter() - t0:.2f}s", flush=True)

    v0 = vertices_np[triangles_np[:, 0]]
    v1 = vertices_np[triangles_np[:, 1]]
    v2 = vertices_np[triangles_np[:, 2]]
    tri_centers = (v0 + v1 + v2) / 3.0
    cross = np.cross(v1 - v0, v2 - v0)
    tri_areas = 0.5 * np.linalg.norm(cross, axis=1)

    print("Mapping triangle class IDs from mesh vertex colors...", flush=True)
    tri_colors = vertex_colors_np[triangles_np[:, 0]]
    color_to_cid = {}
    for cid in persistent_classes:
        _, rgb = get_class_color(cid)
        color_to_cid[tuple(np.round(rgb, 3))] = cid

    tri_cids = np.zeros(num_triangles, dtype=np.int32)
    rounded_cols = np.round(tri_colors, 3)
    for col_key, cid in color_to_cid.items():
        match = (np.abs(rounded_cols - col_key) < 0.05).all(axis=1)
        tri_cids[match] = cid

    print("Building true geometric dual-graph adjacency...", flush=True)
    edges = np.vstack([
        np.sort(triangles_np[:, [0, 1]], axis=1),
        np.sort(triangles_np[:, [1, 2]], axis=1),
        np.sort(triangles_np[:, [2, 0]], axis=1)
    ])
    tri_indices = np.tile(np.arange(num_triangles), 3)
    edge_ids = edges[:, 0].astype(np.int64) * (2**32) + edges[:, 1].astype(np.int64)
    sort_idx = np.argsort(edge_ids)
    is_duplicate = edge_ids[sort_idx][:-1] == edge_ids[sort_idx][1:]
    tri_a = tri_indices[sort_idx][:-1][is_duplicate]
    tri_b = tri_indices[sort_idx][1:][is_duplicate]
    u_cpu = np.concatenate([tri_a, tri_b])
    v_cpu = np.concatenate([tri_b, tri_a])
    adj_matrix = sp.coo_matrix((np.ones(len(u_cpu), dtype=bool), (u_cpu, v_cpu)), shape=(num_triangles, num_triangles)).tocsr()

    print("\n--- Running Dynamic Preemptive Floaters & Solidity Analysis ---", flush=True)
    bg_cids = {1, 5}
    raw_components = []

    for cid in persistent_classes:
        c_tris = np.where(tri_cids == cid)[0]
        if len(c_tris) == 0: continue

        sub_adj = adj_matrix[c_tris, :][:, c_tris]
        n_comp, comp_labels = sp.csgraph.connected_components(sub_adj, directed=False)
        comp_sizes = np.bincount(comp_labels)
        sorted_indices = np.argsort(comp_sizes)[::-1]

        for k in sorted_indices:
            size = comp_sizes[k]
            idx_list = c_tris[comp_labels == k]
            area = float(tri_areas[idx_list].sum())
            pts = tri_centers[idx_list]
            b_min = pts.min(axis=0)
            b_max = pts.max(axis=0)
            dims = b_max - b_min
            v_obb = float(np.prod(dims))

            raw_components.append({
                'class_id': int(cid),
                'size': int(size),
                'area_m2': round(area, 4),
                'tri_indices': idx_list,
                'centroid': pts.mean(axis=0),
                'bbox_min': b_min,
                'bbox_max': b_max,
                'dimensions': dims,
                'vol_obb': v_obb,
                'is_bg': bool(cid in bg_cids)
            })

    print(f"Discovered {len(raw_components):,} raw topological components across all classes.")

    fg_sizes = [c['size'] for c in raw_components if not c['is_bg']]
    print(f"Foreground Component Size Quantiles: p25={np.percentile(fg_sizes, 25):.0f}, "
          f"p50={np.median(fg_sizes):.0f}, p75={np.percentile(fg_sizes, 75):.0f}, p90={np.percentile(fg_sizes, 90):.0f}")

    valid_instances = []
    pruned_floaters = []
    class_instance_map = {cid: [] for cid in persistent_classes}

    for cid in persistent_classes:
        comps = [c for c in raw_components if c['class_id'] == cid]
        if not comps: continue

        is_bg = comps[0]['is_bg']
        if is_bg:
            # The primary component is the genuine room boundary shell (e.g. 905k triangle wall)
            valid_instances.append(comps[0])
            class_instance_map[cid].append(comps[0])
            # ALL secondary/tertiary detached components of background classes are projection floaters!
            for c in comps[1:]:
                pruned_floaters.append((c, "Secondary detached background floater"))
            continue

        max_size = comps[0]['size']
        ref_centroid = comps[0]['centroid']

        for c in comps:
            size = c['size']
            area = c['area_m2']
            dims = c['dimensions']
            d_sorted = np.sort(dims)
            dist_to_main = float(np.linalg.norm(c['centroid'] - ref_centroid))

            if size < 50 or area < 0.01:
                pruned_floaters.append((c, "Below minimum surface area (<0.01 m2 or <50 faces)"))
                continue

            if d_sorted[1] < 0.02 and d_sorted[0] < 0.02:
                pruned_floaters.append((c, "1D wispy reconstruction string (cross-section < 2cm)"))
                continue

            # Measure thickness via PCA
            pts = tri_centers[c['tri_indices']]
            if len(pts) >= 10:
                centered_pts = pts - pts.mean(axis=0)
                cov_pts = np.cov(centered_pts, rowvar=False)
                evals_pts, _ = np.linalg.eigh(cov_pts)
                pca_thick = float(np.sqrt(max(0, evals_pts[0])))
            else:
                pca_thick = d_sorted[0]

            # Flat 2D projection sheets floating in air (thickness < 2.5cm and not known wall-decor)
            if pca_thick < 0.025 and d_sorted[0] < 0.025 and cid not in [4, 7, 21, 54, 58]:
                pruned_floaters.append((c, f"Flat 2D projection sheet/fan (thickness {pca_thick:.3f}m)"))
                continue

            if max_size > 5000 and dist_to_main > 1.8 and size < 0.02 * max_size:
                pruned_floaters.append((c, f"Detached spatial floater ({dist_to_main:.1f}m from primary instance)"))
                continue

            valid_instances.append(c)
            class_instance_map[cid].append(c)

    print(f"Dynamic Floater Pruning Complete:")
    print(f"  Valid Physical Instances Kept: {len(valid_instances):,}")
    print(f"  Floater / Stray Components Pruned: {len(pruned_floaters):,}")

    print("\n--- Applying Statistical Outlier Removal (SOR) & Floater Deletion to Scene Mesh ---", flush=True)
    # 1. Connected components of mesh geometry (air floaters)
    clusters, num_clusters, areas = mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    counts = np.bincount(clusters)
    main_id = np.argmax(counts)

    air_floater_mask = np.zeros(num_triangles, dtype=bool)
    for c_idx, cnt in enumerate(counts):
        if c_idx == main_id:
            continue
        c_tri_idx = np.where(clusters == c_idx)[0]
        u_v = np.unique(triangles_np[c_tri_idx])
        pts = vertices_np[u_v]
        centered = pts - pts.mean(axis=0)
        cov = np.cov(centered, rowvar=False)
        evals, _ = np.linalg.eigh(cov)
        thickness = float(np.sqrt(max(0, evals[0])))
        cl_cids = tri_cids[c_tri_idx]
        dom_cid = int(np.bincount(cl_cids).argmax())

        # Disconnected clusters that are background classes OR thin (< 0.05m) OR small (< 10,000 tris and not genuine armchair)
        is_genuine_object = (dom_cid not in bg_cids) and (thickness >= 0.05) and (cnt >= 10000)
        if not is_genuine_object:
            air_floater_mask[c_tri_idx] = True

    # 2. SOR on vertices (wisps, spikes, detached crumbs)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices_np))
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    inlier_mask = np.zeros(num_vertices, dtype=bool)
    inlier_mask[ind] = True
    outlier_v = ~inlier_mask
    sor_tri_mask = outlier_v[triangles_np].sum(axis=1) >= 2

    # 3. Pruned Floater Triangles (wisps, distant projections, floor-wall projection artifacts)
    floater_del_mask = np.zeros(num_triangles, dtype=bool)
    
    # Class 5 (Wall) false projection outliers located at floor level near the bed/room interior
    wall_rgb = np.array([0.212, 0.988, 0.945])
    is_wall = np.linalg.norm(vertex_colors_np[triangles_np[:, 0]] - wall_rgb, axis=1) < 0.1
    tri_centers_all = vertices_np[triangles_np].mean(axis=1)
    false_wall_on_floor = is_wall & (tri_centers_all[:, 1] < 0.35) & (tri_centers_all[:, 0] > -1.5) & (tri_centers_all[:, 2] > -3.0)
    floater_del_mask |= false_wall_on_floor

    for comp, reason in pruned_floaters:
        tri_idx = comp['tri_indices']
        if any(k in reason for k in ["1D wispy", "Detached spatial", "Small background", "Secondary detached", "Flat 2D projection"]):
            floater_del_mask[tri_idx] = True

    total_delete_mask = air_floater_mask | sor_tri_mask | floater_del_mask
    print(f"  Air Floaters Deleted:              {air_floater_mask.sum():,} triangles")
    print(f"  SOR Outlier Triangles Deleted:     {sor_tri_mask.sum():,} triangles")
    print(f"  Pruned Floater Triangles Deleted:  {floater_del_mask.sum():,} triangles")
    print(f"  Total Triangles Physically Removed:{total_delete_mask.sum():,} triangles ({total_delete_mask.sum() / num_triangles * 100:.2f}%)")

    # Reassign surface speckles (absorb 2D crumbs into surrounding wall or floor)
    cleaned_colors_np = vertex_colors_np.copy()
    floor_rgb = np.array([0.184, 0.816, 0.290])
    wall_rgb = np.array([0.212, 0.988, 0.945])
    for comp, reason in pruned_floaters:
        if "Below minimum" in reason:
            tri_idx = comp['tri_indices']
            reassign_color = floor_rgb if comp['centroid'][1] < 0.35 else wall_rgb
            for t_i in tri_idx:
                if not total_delete_mask[t_i]:
                    cleaned_colors_np[triangles_np[t_i]] = reassign_color

    # Construct clean base mesh
    clean_base_mesh = o3d.geometry.TriangleMesh()
    clean_base_mesh.vertices = o3d.utility.Vector3dVector(vertices_np)
    clean_base_mesh.vertex_colors = o3d.utility.Vector3dVector(cleaned_colors_np)
    clean_base_mesh.triangles = o3d.utility.Vector3iVector(triangles_np[~total_delete_mask])
    clean_base_mesh.remove_unreferenced_vertices()

    # Secondary post-deletion topological sweep to eliminate severed bridge orphan shards
    sec_clusters, sec_num_clusters, _ = clean_base_mesh.cluster_connected_triangles()
    sec_clusters = np.asarray(sec_clusters)
    sec_counts = np.bincount(sec_clusters)
    sec_main_id = np.argmax(sec_counts)

    clean_tris = np.asarray(clean_base_mesh.triangles)
    sec_del = np.zeros(len(clean_tris), dtype=bool)
    for sc_idx, sc_cnt in enumerate(sec_counts):
        if sc_idx == sec_main_id:
            continue
        # If orphan scrap (< 10,000 tris), delete it! Armchair has > 20,000 tris and is preserved.
        if sc_cnt < 10000:
            sec_del[sec_clusters == sc_idx] = True

    if sec_del.sum() > 0:
        print(f"  Secondary Sweep: deleted {sec_del.sum():,} orphan triangles from severed bridges")
        clean_base_mesh.triangles = o3d.utility.Vector3iVector(clean_tris[~sec_del])
        clean_base_mesh.remove_unreferenced_vertices()

    clean_verts_np = np.asarray(clean_base_mesh.vertices)
    clean_tris_np = np.asarray(clean_base_mesh.triangles)
    clean_colors_np = np.asarray(clean_base_mesh.vertex_colors)
    num_clean_vertices = len(clean_verts_np)
    num_clean_triangles = len(clean_tris_np)

    clean_only_path = out_dir / "mesh_cleaned_no_floaters.ply"
    o3d.io.write_triangle_mesh(str(clean_only_path), clean_base_mesh)
    print(f"Exported Cleaned Base Mesh (no floaters): {clean_only_path} ({num_clean_vertices:,} verts, {num_clean_triangles:,} tris)", flush=True)

    print("\n--- Constructing 3D Bounding Boxes with Contours & Semi-Transparent Faces ---", flush=True)
    all_bbox_verts = []
    all_bbox_faces = []
    all_bbox_colors = []
    v_running_offset = num_clean_vertices

    objects_manifest = []
    inst_counter = 1

    for cid in persistent_classes:
        inst_list = class_instance_map[cid]
        for inst_k, comp in enumerate(inst_list, 1):
            is_bg = comp['is_bg']
            inst_name = f"Class_{cid}_inst_{inst_k}" if not is_bg else f"Class_{cid}"
            color_bgr, color_rgb = get_class_color(cid)

            sub_mesh_path = obj_export_dir / f"inst_{inst_counter:03d}_{inst_name}.ply"
            sub_tri_idx = [t_i for t_i in comp['tri_indices'] if not total_delete_mask[t_i]]
            if len(sub_tri_idx) == 0:
                continue
            sub_triangles = triangles_np[sub_tri_idx]
            sub_v_idx = np.unique(sub_triangles)
            v_remap = {old_i: new_i for new_i, old_i in enumerate(sub_v_idx)}
            remap_fn = np.vectorize(lambda x: v_remap[x])
            sub_tris_remapped = remap_fn(sub_triangles)
            sub_verts = vertices_np[sub_v_idx]

            sub_m = o3d.geometry.TriangleMesh()
            sub_m.vertices = o3d.utility.Vector3dVector(sub_verts)
            sub_m.triangles = o3d.utility.Vector3iVector(sub_tris_remapped)
            sub_m.paint_uniform_color(color_rgb)
            sub_m.compute_vertex_normals()
            o3d.io.write_triangle_mesh(str(sub_mesh_path), sub_m)

            obb_corners, obb_extents, obb_center, obb_R = compute_robust_obb(sub_verts)

            manifest_entry = {
                'instance_id': inst_counter,
                'class_id': int(cid),
                'instance_name': inst_name,
                'instance_index': inst_k,
                'num_triangles': int(comp['size']),
                'surface_area_m2': float(comp['area_m2']),
                'centroid': [round(float(x), 3) for x in comp['centroid']],
                'obb_center': [round(float(x), 3) for x in obb_center],
                'obb_extents': [round(float(x), 3) for x in obb_extents],
                'obb_rotation': [[round(float(x), 3) for x in row] for row in obb_R],
                'obb_volume_m3': round(float(np.prod(obb_extents)), 4),
                'bbox_min': [round(float(x), 3) for x in comp['bbox_min']],
                'bbox_max': [round(float(x), 3) for x in comp['bbox_max']],
                'dimensions': [round(float(x), 3) for x in comp['dimensions']],
                'color_rgb': [round(x, 3) for x in color_rgb]
            }
            objects_manifest.append(manifest_entry)

            if not is_bg and comp['size'] >= 50:
                bv, bf, bc = build_bounding_box_geometry(obb_corners, color_rgb, cid)
                if len(bv) > 0:
                    all_bbox_verts.append(bv)
                    all_bbox_faces.append(bf + v_running_offset)
                    all_bbox_colors.append(bc)
                    v_running_offset += len(bv)

            inst_counter += 1

    print("\n--- Exporting Combined Mesh with 3D Bounding Boxes & Tags ---", flush=True)
    base_rgba = (np.clip(clean_colors_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    base_alpha = np.full((num_clean_vertices, 1), 255, dtype=np.uint8)
    base_rgba = np.hstack([base_rgba, base_alpha])

    combined_verts = [clean_verts_np]
    combined_faces = [clean_tris_np]
    combined_colors = [base_rgba]

    if len(all_bbox_verts) > 0:
        combined_verts.append(np.vstack(all_bbox_verts))
        combined_faces.append(np.vstack(all_bbox_faces))
        combined_colors.append(np.vstack(all_bbox_colors))

    final_verts = np.vstack(combined_verts).astype(np.float32)
    final_faces = np.vstack(combined_faces).astype(np.int32)
    final_colors = np.vstack(combined_colors).astype(np.uint8)

    mesh_boxes_path = out_dir / "mesh_with_bounding_boxes.ply"
    print(f"Writing binary PLY: {mesh_boxes_path.name} ({len(final_verts):,} vertices, {len(final_faces):,} faces)...", flush=True)
    export_binary_rgba_ply(mesh_boxes_path, final_verts, final_faces, final_colors)
    print(f"Exported combined mesh with 3D bounding boxes: {mesh_boxes_path} ({mesh_boxes_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    manifest_json_path = out_dir / "objects_manifest.json"
    with open(manifest_json_path, 'w') as f:
        json.dump(objects_manifest, f, indent=2)
    print(f"Exported metadata manifest: {manifest_json_path} ({len(objects_manifest)} objects)", flush=True)

    print("\n--- Generating Class-Instance Grid Table ---", flush=True)
    max_instances = max(len(class_instance_map[cid]) for cid in persistent_classes)
    headers = ["Class ID", "Total Instances", "Total Triangles", "Total Area (m2)"] + [f"Inst {i}" for i in range(1, max_instances + 1)]

    import csv
    table_csv_path = out_dir / "class_instance_table.csv"
    table_rows = []

    for cid in persistent_classes:
        inst_list = class_instance_map[cid]
        num_inst = len(inst_list)
        total_tris = sum(c['size'] for c in inst_list)
        total_area = sum(c['area_m2'] for c in inst_list)
        row = [cid, num_inst, total_tris, round(total_area, 3)]
        for i in range(max_instances):
            if i < num_inst:
                row.append(inst_list[i]['size'])
            else:
                row.append("")
        table_rows.append(row)

    with open(table_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(table_rows)
    print(f"Exported Class-Instance Table CSV: {table_csv_path}", flush=True)

    print("\nTop 15 Classes by Total Surface Area:")
    sorted_by_area = sorted(table_rows, key=lambda r: r[3], reverse=True)
    for r in sorted_by_area[:15]:
        inst_str = ", ".join(f"#{i+1}:{r[4+i]}" for i in range(min(r[1], 4)))
        if r[1] > 4: inst_str += f", +{r[1]-4} more"
        print(f"  Class #{r[0]:3d}: {r[1]:2d} instances | {r[2]:,d} tris | {r[3]:6.2f} m2 | [{inst_str}]", flush=True)

    print("\n=================================================================", flush=True)
    print("STEP 3 EXTRACTION & FILTERING COMPLETED SUCCESSFULLY!", flush=True)
    print(f"Combined Mesh:   {mesh_boxes_path}", flush=True)
    print(f"Objects Manifest:{manifest_json_path}", flush=True)
    print(f"Instance Table:  {table_csv_path}", flush=True)
    print(f"Extracted PLYs:  {obj_export_dir}", flush=True)
    print("=================================================================", flush=True)

if __name__ == '__main__':
    main()
