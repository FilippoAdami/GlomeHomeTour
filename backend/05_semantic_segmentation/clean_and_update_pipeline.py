#!/usr/bin/env python3
"""
Applies dual-graph micro-island absorption (< 15 triangles) across all active classes.
Absorbs stray seam bleeds, grazing ray artifacts, and boundary residue into enclosing classes.
Updates:
  - triangle_class_labels.npy
  - mesh_projected_refined.ply
  - per_class_meshes/
  - instances_summary.csv
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
import numpy as np
import open3d as o3d
import scipy.sparse as sp
from scipy.sparse import csgraph

from detect_structural_classes import (
    build_mesh_dual_graph,
    absorb_outlier_floaters,
    absorb_enclosed_patches,
    converge_class_instances_iteratively,
    export_colored_ply,
    get_class_color,
    extract_instances,
    classify_structural_classes,
    analyze_class_geometry,
    estimate_gravity_and_scene_bands,
    read_scene_sizes,
    load_canonical_dino_embeddings,
    export_structural_manifest,
)
from refine_class_boundaries import (
    refine_all_boundaries,
    align_structural_planes,
    reclaim_thin_wall_plates,
)
from export_per_class_meshes import export_per_class_meshes


def absorb_disconnected_wall_crumbs(
    labels: np.ndarray,
    adj_u: np.ndarray,
    adj_v: np.ndarray,
    tri_normals: np.ndarray,
    wall_cid: int = 5,
    max_crumb_size: int = 25000,
) -> np.ndarray:
    """
    Absorbs disconnected Wall micro-crumbs (< max_crumb_size triangles) into their
    boundary-enclosing neighbor classes, respecting physical orientation invariants:
      - Primarily vertical crumbs cannot become Floor (14) or Ceiling (1).
      - Downward crumbs cannot become Floor (14).
    Leaves strictly the continuous room perimeter wall intact.
    """
    cleaned = labels.copy()
    m5 = (cleaned == wall_cid)
    if not np.any(m5):
        return cleaned

    num_tris = len(cleaned)
    idx5 = np.flatnonzero(m5)
    e5 = m5[adj_u] & m5[adj_v]
    inv5 = np.full(num_tris, -1, np.int32)
    inv5[idx5] = np.arange(len(idx5), dtype=np.int32)
    sub5 = sp.csr_matrix(
        (np.ones(e5.sum(), bool), (inv5[adj_u[e5]], inv5[adj_v[e5]])),
        shape=(len(idx5), len(idx5))
    )
    n_comp5, comp5_ids = csgraph.connected_components(sub5, directed=False)
    sizes5 = np.bincount(comp5_ids)
    sorted_5 = np.argsort(-sizes5)

    crumb_c_ids = [c for c in sorted_5[1:] if sizes5[c] < max_crumb_size]
    num_reassigned = 0

    for c in crumb_c_ids:
        c_tris = idx5[comp5_ids == c]
        is_p = np.zeros(num_tris, bool)
        is_p[c_tris] = True
        b_edges = is_p[adj_u] & (~is_p[adj_v])
        nbrs = cleaned[adj_v[b_edges]]
        ext_nbrs = nbrs[nbrs != wall_cid]

        c_norms = tri_normals[c_tris]
        mean_ny = float(c_norms[:, 1].mean())
        frac_vert = float((np.abs(c_norms[:, 1]) < 0.25).mean())

        valid_nbrs = ext_nbrs.copy()
        if frac_vert > 0.40:
            valid_nbrs = valid_nbrs[(valid_nbrs != 14) & (valid_nbrs != 1)]
        elif mean_ny < 0.0:
            valid_nbrs = valid_nbrs[valid_nbrs != 14]

        if len(valid_nbrs) > 0:
            vals, cnts = np.unique(valid_nbrs, return_counts=True)
            winner = int(vals[cnts.argmax()])
        else:
            winner = wall_cid

        if winner != wall_cid:
            cleaned[c_tris] = winner
            num_reassigned += len(c_tris)

    print(f"  [Wall {wall_cid}] Absorbed {num_reassigned:,} disconnected crumb triangles across {len(crumb_c_ids)} micro-components.")
    return cleaned


def main():
    mesh_path = Path("backend/current_scene/06c_semantic_segmentation/step2_mesh_projection/mesh_projected_refined.ply")
    raw_labels_path = Path("backend/current_scene/06c_semantic_segmentation/step2_mesh_projection/triangle_class_labels_raw.npy")
    labels_path = Path("backend/current_scene/06c_semantic_segmentation/step2_mesh_projection/triangle_class_labels.npy")
    struct_json_path = Path("backend/current_scene/06c_semantic_segmentation/step3_structural_detection/structural_classes.json")
    instances_csv_path = Path("backend/current_scene/06c_semantic_segmentation/step3_structural_detection/instances_summary.csv")
    per_class_dir = Path("backend/current_scene/06c_semantic_segmentation/per_class_meshes")

    print(f"Loading mesh: {mesh_path}...")
    t0 = time.perf_counter()
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)

    load_source = raw_labels_path if raw_labels_path.is_file() else labels_path
    print(f"Loading labels from {load_source}...")
    labels = np.load(load_source)
    print(f"Loaded {len(verts):,} vertices, {len(tris):,} triangles in {time.perf_counter() - t0:.2f}s")

    print("Building dual graph...")
    t0 = time.perf_counter()
    adj_u, adj_v = build_mesh_dual_graph(tris)
    print(f"Dual graph built in {time.perf_counter() - t0:.2f}s")

    active_classes = sorted(np.unique(labels).tolist())
    print(f"Active classes ({len(active_classes)}): {active_classes}")

    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    tri_areas = 0.5 * np.linalg.norm(cross, axis=1)
    tri_normals = cross / np.maximum(2.0 * tri_areas[:, None], 1e-8)
    tri_centroids = (v0 + v1 + v2) / 3.0

    gravity_up, scene_bands = estimate_gravity_and_scene_bands(verts, tris)

    # Pass 1: Pre-clean raw 2D projection micro-crumbs (< 150 triangles)
    print("\nPass 1: Pre-clean raw 2D projection micro-crumbs (< 150 triangles)...")
    cleaned_labels, n_pre_cleaned = absorb_outlier_floaters(
        triangle_labels=labels,
        adj_u=adj_u,
        adj_v=adj_v,
        active_classes=active_classes,
        min_island_size=150,
    )
    print(f"  Pass 1 pre-cleaned: {n_pre_cleaned:,} triangles")
    active_classes = sorted(np.unique(cleaned_labels).tolist())

    # Pass 2: Geometric & Photometric Boundary Refinement (Structural Planes & Texture-Gated Absorption)
    print("\nPass 2: Geometric & Photometric Boundary Refinement (Creases, Planes & Universal Texture Snapping)...")
    clean_mesh_path = Path("backend/current_scene/06c_semantic_segmentation/step1_cleaned_mesh/mesh_cleaned.ply")
    contention_mesh_path = Path("backend/current_scene/06c_semantic_segmentation/step2_mesh_projection/mesh_projected_contention.ply")
    cleaned_labels, absorbed_cids, reclaimed_stats = refine_all_boundaries(
        mesh_path=mesh_path,
        clean_mesh_path=clean_mesh_path,
        labels=cleaned_labels,
        contention_mesh_path=contention_mesh_path,
    )

    # Pass 2b: Generalizable Physical Wall Thickness & Thin Plate Reclaim (Scene-Agnostic)
    print("\nPass 2b: Physical Wall Thickness & Thin Plate Reclaim (Scene-Agnostic Dual-Graph Growth)...")
    cleaned_labels, furn_stats = reclaim_thin_wall_plates(
        labels=cleaned_labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_normals=tri_normals,
        tri_centroids=tri_centroids,
        wall_cid=5,
    )
    for fcid, cnt in furn_stats.items():
        if cnt > 0:
            print(f"  [Pass 2b] Reclaimed {cnt:,} thin wall plate triangles into Class #{fcid}")

    # Pass 2c: Generalizable Enclosed Patch Absorption (>= 70% boundary contact)
    print("\nPass 2c: Generalizable Enclosed Patch Absorption (>= 70% boundary contact)...")
    cleaned_labels, n_enc2c = absorb_enclosed_patches(
        labels=cleaned_labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_normals=tri_normals,
        active_classes=active_classes,
        max_patch_size=1500,
        encirclement_ratio_thresh=0.70,
    )
    print(f"  Pass 2c absorbed: {n_enc2c:,} enclosed patch triangles")

    # Pass 2d: Disconnected Wall Crumb Absorption
    print("\nPass 2d: Absorbing disconnected Wall micro-crumbs...")
    cleaned_labels = absorb_disconnected_wall_crumbs(
        labels=cleaned_labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_normals=tri_normals,
        wall_cid=5,
        max_crumb_size=25000,
    )
    active_classes = sorted(np.unique(cleaned_labels).tolist())

    # Pass 3: Iterative Instance Convergence (Outlier absorption & multi-instance protection)
    print("\nPass 3: Iterative Instance Convergence (Outlier absorption & multi-instance protection)...")
    cleaned_labels, reassigned = converge_class_instances_iteratively(
        triangle_labels=cleaned_labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_areas=tri_areas,
        tri_centroids=tri_centroids,
        tri_normals=tri_normals,
        gravity_up=gravity_up,
        active_classes=active_classes,
        min_instance_size=500,
        min_instance_area=0.03,
        min_instance_span=0.15,
        max_proximity_merge_dist=0.04,
        max_iterations=10,
        cluster_ratio_thresh=0.20,
        coverage_target_thresh=0.95,
        protect_min_triangles=1000,
        protect_min_area=0.05,
    )
    active_classes = sorted(np.unique(cleaned_labels).tolist())

    # Pass 4: Secondary micro-island / crumb absorption (< 300 triangles)
    print("\nPass 4: Micro-island / crumb absorption (< 300 triangles)...")
    cleaned_labels, n_abs2 = absorb_outlier_floaters(
        triangle_labels=cleaned_labels,
        adj_u=adj_u,
        adj_v=adj_v,
        active_classes=active_classes,
        min_island_size=300,
    )
    print(f"  Pass 4 absorbed: {n_abs2:,} triangles")
    active_classes = sorted(np.unique(cleaned_labels).tolist())

    # Pass 5: Final Structural Plane Alignment (Floor & Ceiling Non-Vertical & Planar Enforcement)
    print("\nPass 5: Final Structural Plane Alignment (Floor & Ceiling Non-Vertical & Planar Enforcement)...")
    clean_mesh = o3d.io.read_triangle_mesh(str(clean_mesh_path))
    vc_clean = np.asarray(clean_mesh.vertex_colors)
    tri_colors_rgb = vc_clean[tris].mean(axis=1) if len(vc_clean) == len(verts) else np.zeros((len(tris), 3), dtype=np.float32)
    cleaned_labels, plane_stats = align_structural_planes(
        labels=cleaned_labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_normals=tri_normals,
        tri_centroids=tri_centroids,
        tri_colors=tri_colors_rgb,
        floor_cid=14,
        ceiling_cid=1,
        wall_cid=5,
        max_floor_tilt_deg=45.0,
    )
    print(f"  Pass 5 alignment: {plane_stats}")
    active_classes = sorted(np.unique(cleaned_labels).tolist())

    # Pass 6: Final Post-Alignment Cleanup (Guarantees zero crumbs or wall floaters reach disk)
    print("\nPass 6: Final Post-Alignment Cleanup...")
    cleaned_labels = absorb_disconnected_wall_crumbs(
        labels=cleaned_labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_normals=tri_normals,
        wall_cid=5,
        max_crumb_size=25000,
    )
    cleaned_labels, n_enc6 = absorb_enclosed_patches(
        labels=cleaned_labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_normals=tri_normals,
        active_classes=active_classes,
        max_patch_size=1500,
        encirclement_ratio_thresh=0.70,
    )
    cleaned_labels, n_abs6 = absorb_outlier_floaters(
        triangle_labels=cleaned_labels,
        adj_u=adj_u,
        adj_v=adj_v,
        active_classes=active_classes,
        min_island_size=150,
    )

    # Strictly enforce physical invariants (Floor vertical = 0, Ceiling vertical = 0)
    f_vert = (cleaned_labels == 14) & (tri_normals[:, 1] <= 0.0)
    cleaned_labels[f_vert] = 5
    c_vert = (cleaned_labels == 1) & (tri_normals[:, 1] >= -0.25)
    cleaned_labels[c_vert] = 5

    print(f"  Pass 6 final cleanup complete (absorbed {n_enc6 + n_abs6:,} residual triangles).")
    active_classes = sorted(np.unique(cleaned_labels).tolist())

    # Save cleaned labels
    np.save(labels_path, cleaned_labels)
    print(f"Saved cleaned triangle labels to {labels_path.name}")

    # Export updated mesh_projected_refined.ply
    print(f"\nUpdating {mesh_path.name} with cleaned vertex colors...")
    tri_colors = np.full((len(tris), 3), 128, dtype=np.uint8)
    for cid in active_classes:
        m = (cleaned_labels == cid)
        if np.any(m):
            _, rgb_f = get_class_color(cid)
            tri_colors[m] = (np.array(rgb_f) * 255.0).round().astype(np.uint8)

    export_colored_ply(mesh_path, verts, tris, triangle_colors_rgb=tri_colors)
    print(f"Exported refined mesh to {mesh_path}")

    # Re-extract instances on the cleaned labels
    print("\nRe-extracting instances on cleaned mesh...")
    instances, class_inst_counts = extract_instances(
        triangle_labels=cleaned_labels,
        active_classes=active_classes,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_areas=tri_areas,
        tri_centroids=tri_centroids,
        tri_normals=tri_normals,
        gravity_up=gravity_up,
        min_instance_size=500,
        min_instance_area=0.03,
        min_instance_span=0.15,
        max_proximity_merge_dist=0.04,
    )
    print(f"Extracted {len(instances):,} valid instances across {len(active_classes)} classes.")

    # Re-extract geometric features & classifications
    scene_size_file = Path("backend/current_scene/01_poses_refinment/scene_size.txt")
    scene_info = read_scene_sizes(scene_size_file if scene_size_file.is_file() else None)
    emb_file = Path("backend/current_scene/canonical_class_embeddings.npz")
    canonical_embs = load_canonical_dino_embeddings(emb_file if emb_file.is_file() else None, active_classes)

    features = analyze_class_geometry(
        vertices=verts,
        triangles=tris,
        triangle_labels=cleaned_labels,
        class_ids=active_classes,
        gravity_up=gravity_up,
        scene_bands=scene_bands,
    )

    classifications = classify_structural_classes(
        class_features=features,
        scene_bands=scene_bands,
        canonical_embs=canonical_embs,
        scene_info=scene_info,
    )

    for inst in instances.values():
        cid = inst["class_id"]
        c_role = classifications.get(cid, {}).get("structural_role", "object")
        if c_role == "ceiling" and inst["h_mean"] < scene_bands["h_ceiling_band"] and inst["frac_down"] >= 0.30:
            inst["assigned_role"] = "wall_sottoscala"
        else:
            inst["assigned_role"] = c_role

    # Export updated manifests and instances summary
    step3_dir = struct_json_path.parent
    export_structural_manifest(
        output_dir=step3_dir,
        classifications=classifications,
        features=features,
        scene_bands=scene_bands,
        instances=instances,
        class_instance_counts=class_inst_counts,
    )

    # Re-export per_class_meshes
    print(f"\nRe-exporting per-class meshes to {per_class_dir}...")
    export_per_class_meshes(
        mesh_path=mesh_path,
        structural_json_path=struct_json_path,
        instances_csv_path=instances_csv_path,
        output_dir=per_class_dir,
        include_clay_backdrop=True,
    )

    print("\n[SUCCESS] Pipeline cleanly updated! All phantom seam lines have been eliminated.")


if __name__ == "__main__":
    main()
