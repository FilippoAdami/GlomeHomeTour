#!/usr/bin/env python3
"""
Step 1: High-Efficiency Mesh Floater Removal for 2D-to-3D Semantic Lifting.

Removes disconnected airborne floaters, crumbs, wisps, and false projection phantoms
while preserving the primary room mesh and genuine detached physical objects (e.g. boxing bag).
The cleaned mesh is preserved for semantic processing; separate colored and red
versions are sealed for web export after all face filtering is complete.

Geometric decision criteria for detached candidate components:
1. Scale: Triangle count >= min_triangles and surface area >= min_area.
2. 3D Volumetric Thickness: Smallest PCA principal axis spread (sqrt(lambda_min)) >= min_thickness.
3. Solidity: Ratio of Convex Hull volume to OBB volume >= min_solidity.
4. Surface Closure: Ratio of open boundary edges to total edges <= max_boundary_openness.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_3DGS_to_mesh"))
from fill_mesh_holes import edge_keys, fill_mesh, red_patches


def profile_component_geometry(
    component_tris: np.ndarray,
    all_vertices: np.ndarray,
    tri_areas: np.ndarray,
    comp_tri_indices: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes rigorous geometric metrics for a mesh component to determine
    whether it is a genuine physical 3D object or an artifact/floater.
    """
    num_triangles = len(comp_tri_indices)
    surface_area = float(tri_areas[comp_tri_indices].sum())

    # Extract unique vertices for this component
    comp_vertex_indices = np.unique(component_tris)
    pts = all_vertices[comp_vertex_indices]

    if len(pts) < 4:
        return {
            "num_triangles": num_triangles,
            "surface_area": surface_area,
            "thickness": 0.0,
            "solidity": 0.0,
            "boundary_openness": 1.0,
            "obb_extents": [0.0, 0.0, 0.0],
            "obb_volume": 0.0,
            "hull_volume": 0.0,
            "centroid": pts.mean(axis=0).tolist() if len(pts) > 0 else [0.0, 0.0, 0.0],
        }

    centroid = pts.mean(axis=0)

    # 1. Thickness via PCA eigenvalue decomposition
    centered = pts - centroid
    cov = np.cov(centered, rowvar=False)
    evals, _ = np.linalg.eigh(cov)
    sqrt_evals = np.sqrt(np.maximum(0.0, evals))
    thickness = float(sqrt_evals[0])

    # 2. OBB Extents & Volume
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    try:
        obb = pcd.get_oriented_bounding_box()
        obb_extents = np.asarray(obb.extent)
        obb_volume = float(np.prod(obb_extents))
    except Exception:
        bmin, bmax = pts.min(axis=0), pts.max(axis=0)
        obb_extents = bmax - bmin
        obb_volume = float(np.prod(obb_extents))

    # 3. Convex Hull Volume & Solidity
    try:
        hull, _ = pcd.compute_convex_hull()
        hull_volume = float(hull.get_volume())
    except Exception:
        hull_volume = 0.0

    solidity = float(hull_volume / max(1e-7, obb_volume))

    # 4. Boundary edge openness ratio
    # Build edge counts across component triangles
    edge_tuples = np.vstack([
        np.sort(component_tris[:, [0, 1]], axis=1),
        np.sort(component_tris[:, [1, 2]], axis=1),
        np.sort(component_tris[:, [2, 0]], axis=1),
    ])
    # Pack edge pairs into 64-bit integers for vectorized uniqueness counting
    edge_keys = edge_tuples[:, 0].astype(np.int64) * (2**32) + edge_tuples[:, 1].astype(np.int64)
    _, counts = np.unique(edge_keys, return_counts=True)
    num_total_edges = len(counts)
    num_boundary_edges = int(np.count_nonzero(counts == 1))
    boundary_openness = float(num_boundary_edges / max(1, num_total_edges))

    return {
        "num_triangles": num_triangles,
        "surface_area": round(surface_area, 4),
        "thickness": round(thickness, 4),
        "solidity": round(solidity, 4),
        "boundary_openness": round(boundary_openness, 4),
        "obb_extents": [round(float(x), 3) for x in obb_extents],
        "obb_volume": round(obb_volume, 6),
        "hull_volume": round(hull_volume, 6),
        "centroid": [round(float(x), 3) for x in centroid],
    }


def clean_mesh_floaters(
    input_mesh_path: Path,
    output_dir: Path,
    min_triangles: int = 4000,
    min_area: float = 0.15,
    min_thickness: float = 0.04,
    min_solidity: float = 0.36,
    max_boundary_openness: float = 0.065,
    enable_sor: bool = True,
    sor_neighbors: int = 30,
    sor_std_ratio: float = 2.0,
) -> Dict[str, Any]:
    """
    Executes efficient mesh floater removal with sub-millisecond phase timing.
    """
    t_start = time.perf_counter()
    timings: Dict[str, float] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=================================================================")
    print(f"STEP 1: HIGH-EFFICIENCY MESH FLOATER REMOVAL")
    print(f"Input Mesh:  {input_mesh_path}")
    print(f"Output Dir:  {output_dir}")
    print(f"=================================================================")

    # -------------------------------------------------------------
    # Phase 1: Load Mesh
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    if not input_mesh_path.is_file():
        raise FileNotFoundError(f"Input mesh not found: {input_mesh_path}")

    mesh = o3d.io.read_triangle_mesh(str(input_mesh_path))
    vertices_np = np.asarray(mesh.vertices)
    triangles_np = np.asarray(mesh.triangles)
    has_colors = mesh.has_vertex_colors()
    vertex_colors_np = np.asarray(mesh.vertex_colors) if has_colors else None

    raw_num_vertices = len(vertices_np)
    raw_num_triangles = len(triangles_np)
    t_load = time.perf_counter() - t0
    timings["load_mesh_seconds"] = round(t_load, 3)
    print(f"[Phase 1/5] Loaded mesh: {raw_num_vertices:,} vertices, {raw_num_triangles:,} triangles ({t_load:.3f}s)")

    if raw_num_triangles == 0:
        raise ValueError("Mesh contains 0 triangles!")

    # Precompute triangle surface areas vectorized
    v0 = vertices_np[triangles_np[:, 0]]
    v1 = vertices_np[triangles_np[:, 1]]
    v2 = vertices_np[triangles_np[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    tri_areas = 0.5 * np.linalg.norm(cross, axis=1)

    # -------------------------------------------------------------
    # Phase 2: Connected Component Clustering (C++ Graph Traversal)
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    clusters, num_clusters, _ = mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    counts = np.bincount(clusters)
    t_cluster = time.perf_counter() - t0
    timings["cluster_triangles_seconds"] = round(t_cluster, 3)

    sorted_cluster_ids = np.argsort(counts)[::-1]
    primary_cluster_id = int(sorted_cluster_ids[0])
    primary_tri_count = int(counts[primary_cluster_id])

    print(f"[Phase 2/5] Partitioned mesh into {len(counts):,} connected components ({t_cluster:.3f}s)")
    print(f"            Primary component (ID {primary_cluster_id}): {primary_tri_count:,} triangles "
          f"({primary_tri_count / raw_num_triangles * 100:.2f}% of mesh)")

    # -------------------------------------------------------------
    # Phase 3: Component Geometric Classification
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    kept_cluster_ids = [primary_cluster_id]
    pruned_cluster_ids = []
    component_records = []

    # Record primary component
    primary_record = {
        "cluster_id": primary_cluster_id,
        "is_primary": True,
        "triangles": primary_tri_count,
        "surface_area": round(float(tri_areas[clusters == primary_cluster_id].sum()), 3),
        "status": "KEPT",
        "reason": "Primary scene mesh body",
    }
    component_records.append(primary_record)

    # Evaluate all other disconnected components
    num_candidates = len(sorted_cluster_ids) - 1
    print(f"[Phase 3/5] Evaluating {num_candidates:,} candidate disconnected components...")

    for rank, cid in enumerate(sorted_cluster_ids[1:], start=1):
        cid = int(cid)
        comp_tri_count = int(counts[cid])
        comp_tri_mask = (clusters == cid)
        comp_tri_idx = np.flatnonzero(comp_tri_mask)

        # Quick size rejection
        comp_area = float(tri_areas[comp_tri_idx].sum())
        if comp_tri_count < min_triangles or comp_area < min_area:
            pruned_cluster_ids.append(cid)
            component_records.append({
                "cluster_id": cid,
                "is_primary": False,
                "triangles": comp_tri_count,
                "surface_area": round(comp_area, 4),
                "status": "PRUNED",
                "reason": f"Below size threshold (<{min_triangles} tris or <{min_area}m2)",
            })
            continue

        # In-depth geometric analysis for large candidate components
        comp_tris = triangles_np[comp_tri_idx]
        geo = profile_component_geometry(comp_tris, vertices_np, tri_areas, comp_tri_idx)

        # Decision gates:
        # 1. Volumetric thickness
        thick_pass = geo["thickness"] >= min_thickness
        # 2. Solidity (convex hull / OBB)
        solidity_pass = geo["solidity"] >= min_solidity
        # 3. Boundary openness (<= max_boundary_openness)
        boundary_pass = geo["boundary_openness"] <= max_boundary_openness

        is_valid_object = thick_pass and solidity_pass and boundary_pass
        if is_valid_object:
            kept_cluster_ids.append(cid)
            status = "KEPT"
            reason = "Valid solid detached 3D object (passes scale, thickness, solidity, closure)"
            print(f"   >>> PRESERVED DETACHED OBJECT: Component {cid} ({comp_tri_count:,} tris, {geo['surface_area']}m2, "
                  f"thickness={geo['thickness']}m, solidity={geo['solidity']}, open_edges={geo['boundary_openness']*100:.1f}%)")
        else:
            pruned_cluster_ids.append(cid)
            status = "PRUNED"
            failures = []
            if not thick_pass:
                failures.append(f"insufficient thickness ({geo['thickness']:.3f}m < {min_thickness}m)")
            if not solidity_pass:
                failures.append(f"low solidity ({geo['solidity']:.3f} < {min_solidity})")
            if not boundary_pass:
                failures.append(f"high boundary openness ({geo['boundary_openness']*100:.1f}% > {max_boundary_openness*100:.1f}%)")
            reason = f"Floater rejected: {', '.join(failures)}"
            print(f"   --- PRUNED FLOATER: Component {cid} ({comp_tri_count:,} tris, {geo['surface_area']}m2) -> {reason}")

        record = {
            "cluster_id": cid,
            "is_primary": False,
            "status": status,
            "reason": reason,
            **geo,
        }
        component_records.append(record)

    t_eval = time.perf_counter() - t0
    timings["classification_seconds"] = round(t_eval, 3)
    print(f"            Classification completed in {t_eval:.3f}s. Kept {len(kept_cluster_ids)} components, pruned {len(pruned_cluster_ids)}.")

    # -------------------------------------------------------------
    # Phase 4: Mesh Filtering & Optional SOR
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    # Mask of triangles to keep based on component classification
    keep_triangle_mask = np.isin(clusters, kept_cluster_ids)
    num_floater_triangles_removed = int(np.count_nonzero(~keep_triangle_mask))

    # Optional Statistical Outlier Removal on vertices
    t_sor = 0.0
    sor_outlier_count = 0
    if enable_sor:
        t_sor_start = time.perf_counter()
        # Find active vertices in the kept triangles
        active_vertex_indices = np.unique(triangles_np[keep_triangle_mask])
        active_pts = vertices_np[active_vertex_indices]

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(active_pts))
        _, inlier_indices = pcd.remove_statistical_outlier(nb_neighbors=sor_neighbors, std_ratio=sor_std_ratio)

        is_inlier_v = np.zeros(raw_num_vertices, dtype=bool)
        is_inlier_v[active_vertex_indices[inlier_indices]] = True
        sor_outlier_count = len(active_pts) - len(inlier_indices)

        # Triangle is kept if all or at least 2 vertices are inliers
        tri_inlier_count = is_inlier_v[triangles_np].sum(axis=1)
        sor_keep_triangles = tri_inlier_count >= 2
        keep_triangle_mask = keep_triangle_mask & sor_keep_triangles
        t_sor = time.perf_counter() - t_sor_start
        print(f"[Phase 4/5] SOR removed {sor_outlier_count:,} vertex outliers ({t_sor:.3f}s)")

    timings["sor_seconds"] = round(t_sor, 3)

    # Build clean Open3D mesh
    filtered_mesh = o3d.geometry.TriangleMesh()
    filtered_mesh.vertices = o3d.utility.Vector3dVector(vertices_np)
    if has_colors:
        filtered_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors_np)
    filtered_mesh.triangles = o3d.utility.Vector3iVector(triangles_np[keep_triangle_mask])
    filtered_mesh.remove_unreferenced_vertices()
    filtered_mesh.compute_vertex_normals()

    clean_num_vertices = len(filtered_mesh.vertices)
    clean_num_triangles = len(filtered_mesh.triangles)
    t_filter = time.perf_counter() - t0 - t_sor
    timings["mesh_filtering_seconds"] = round(t_filter, 3)

    print(f"            Mesh filtered: {clean_num_vertices:,} vertices, {clean_num_triangles:,} triangles "
          f"({num_floater_triangles_removed:,} floater triangles removed, {t_filter:.3f}s)")

    # -------------------------------------------------------------
    # Phase 5: Export Clean Mesh & Statistics
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    raw_cleaned_path = output_dir / "mesh_cleaned.ply"
    if not o3d.io.write_triangle_mesh(str(raw_cleaned_path), filtered_mesh):
        raise RuntimeError(f"Failed to write {raw_cleaned_path}")
    t_seal = time.perf_counter()
    sealed_mesh, records = fill_mesh(filtered_mesh, [], seal_all=True)
    _, edge_counts = np.unique(edge_keys(np.asarray(sealed_mesh.triangles)), return_counts=True)
    if any(record["filled_region"] is None for record in records) or np.any(edge_counts != 2):
        raise RuntimeError("Mesh sealing left boundary or nonmanifold edges; unsealed cleaned mesh is preserved")
    timings["sealing_seconds"] = round(time.perf_counter() - t_seal, 3)
    sealed_mesh_path = output_dir / "mesh_sealed_colored.ply"
    red_mesh_path = output_dir / "mesh_sealed_red.ply"
    sealed_tmp = output_dir / "mesh_sealed_colored.tmp.ply"
    red_tmp = output_dir / "mesh_sealed_red.tmp.ply"
    if not o3d.io.write_triangle_mesh(str(sealed_tmp), sealed_mesh):
        raise RuntimeError(f"Failed to write {sealed_tmp}")
    if not o3d.io.write_triangle_mesh(str(red_tmp), red_patches(sealed_mesh, clean_num_triangles)):
        raise RuntimeError(f"Failed to write {red_tmp}")
    os.replace(red_tmp, red_mesh_path)
    os.replace(sealed_tmp, sealed_mesh_path)
    t_export = time.perf_counter() - t0
    timings["export_ply_seconds"] = round(t_export, 3)

    t_total = time.perf_counter() - t_start
    timings["total_pipeline_seconds"] = round(t_total, 3)
    throughput_triangles_per_sec = int(raw_num_triangles / max(1e-4, t_total))

    # Summary manifest
    summary: Dict[str, Any] = {
        "input_mesh": str(input_mesh_path.resolve()),
        "output_mesh": str(raw_cleaned_path.resolve()),
        "sealed_mesh": str(sealed_mesh_path.resolve()),
        "raw_vertices": raw_num_vertices,
        "raw_triangles": raw_num_triangles,
        "clean_vertices": clean_num_vertices,
        "clean_triangles": clean_num_triangles,
        "sealed_vertices": len(sealed_mesh.vertices),
        "sealed_triangles": len(sealed_mesh.triangles),
        "boundary_loops_filled": len(records),
        "remaining_boundary_edges": 0,
        "red_mesh": str(red_mesh_path.resolve()),
        "removed_triangles": raw_num_triangles - clean_num_triangles,
        "removed_percentage": round((raw_num_triangles - clean_num_triangles) / raw_num_triangles * 100, 2),
        "total_components": len(counts),
        "kept_components_count": len(kept_cluster_ids),
        "pruned_components_count": len(pruned_cluster_ids),
        "kept_component_ids": kept_cluster_ids,
        "throughput_triangles_per_second": throughput_triangles_per_sec,
        "timings_breakdown": timings,
        "components_manifest": component_records,
    }

    stats_path = output_dir / "mesh_floater_removal_stats.json"
    with open(stats_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Phase 5/5] Exported sealed clean mesh: {sealed_mesh_path} ({sealed_mesh_path.stat().st_size / 1e6:.1f} MB, {t_export:.3f}s)")
    print(f"            Exported summary stats: {stats_path}")
    print(f"=================================================================")
    print(f"TOTAL EXECUTION TIME: {t_total:.3f}s ({throughput_triangles_per_sec:,} tri/sec)")
    print(f"Kept: Primary mesh + {len(kept_cluster_ids) - 1} valid objects")
    print(f"Pruned: {len(pruned_cluster_ids)} floater components ({raw_num_triangles - clean_num_triangles:,} triangles)")
    print(f"=================================================================")

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 1: High-Efficiency Mesh Floater Removal")
    parser.add_argument("--workspace", type=Path, help="Pipeline workspace (enables resume and stage metrics)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to input TSDF mesh PLY",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for cleaned mesh and stats",
    )
    parser.add_argument("--min-triangles", type=int, default=4000, help="Minimum triangle count for candidate object")
    parser.add_argument("--min-area", type=float, default=0.15, help="Minimum surface area (m2) for candidate object")
    parser.add_argument("--min-thickness", type=float, default=0.04, help="Minimum PCA thickness (m)")
    parser.add_argument("--min-solidity", type=float, default=0.36, help="Minimum convex hull / OBB volume ratio")
    parser.add_argument("--max-boundary-openness", type=float, default=0.065, help="Maximum open boundary edge ratio")
    parser.add_argument("--disable-sor", action="store_true", help="Disable statistical outlier removal")
    parser.add_argument("--sor-neighbors", type=int, default=30, help="SOR number of neighbors")
    parser.add_argument("--sor-std-ratio", type=float, default=2.0, help="SOR standard deviation ratio")

    args = parser.parse_args(argv)
    workspace = args.workspace or Path(__file__).resolve().parents[1] / "current_scene"
    input_path = args.input or workspace / "04_3DGS_to_mesh/mesh.ply"
    output_dir = args.output_dir or workspace / "06c_semantic_segmentation/step1_cleaned_mesh"
    kwargs = dict(min_triangles=args.min_triangles, min_area=args.min_area,
                  min_thickness=args.min_thickness, min_solidity=args.min_solidity,
                  max_boundary_openness=args.max_boundary_openness,
                  enable_sor=not args.disable_sor, sor_neighbors=args.sor_neighbors,
                  sor_std_ratio=args.sor_std_ratio)
    if args.workspace:
        from Utilities.pipeline_step import StepContext, is_done
        if (not args.force and args.input is None and args.output_dir is None and
                is_done(workspace, "mesh_cleanup", [output_dir / "mesh_cleaned.ply",
                                                   output_dir / "mesh_sealed_colored.ply",
                                                   output_dir / "mesh_sealed_red.ply",
                                                   output_dir / "mesh_floater_removal_stats.json"]) and
                (output_dir / "mesh_sealed_colored.ply").stat().st_mtime_ns >= input_path.stat().st_mtime_ns):
            print("[mesh_cleanup] already done, skipping (use --force to re-run)")
            return 0
        with StepContext("mesh_cleanup", workspace, artifacts_dir=output_dir) as ctx:
            summary = clean_mesh_floaters(input_path, output_dir, **kwargs)
            ctx.metric("boundary_loops_filled", summary["boundary_loops_filled"])
            ctx.metric("mesh_path", summary["sealed_mesh"])
    else:
        clean_mesh_floaters(input_path, output_dir, **kwargs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
