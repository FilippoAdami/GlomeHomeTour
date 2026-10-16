#!/usr/bin/env python3
"""
Extraction of structural classes (walls, floors, ceilings, and stairs if any)
from original colored mesh (original_mesh.ply) into separate instance meshes.
Only major instances are preserved as separate meshes; small components are merged
into the closest major instance if within a proximity threshold, or discarded as floaters.
Uses standard Open3D mesh export with vertex normals and colors for full compatibility
with DCC software including Blender, MeshLab, CloudCompare, etc.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def component_face_indices(triangle_clusters: np.ndarray, num_components: int) -> List[np.ndarray]:
    """Return local face indices for each component, preserving face order."""
    if len(triangle_clusters) == 0:
        return [np.empty(0, dtype=np.int64) for _ in range(num_components)]
    order = np.argsort(triangle_clusters, kind="stable")
    sorted_clusters = triangle_clusters[order]
    starts = np.flatnonzero(np.r_[True, sorted_clusters[1:] != sorted_clusters[:-1]])
    grouped = np.split(order, starts[1:])
    result = [np.empty(0, dtype=np.int64) for _ in range(num_components)]
    for component, faces in zip(sorted_clusters[starts], grouped):
        result[int(component)] = faces
    return result


def closest_major_instance(points: np.ndarray, trees: List[cKDTree]) -> Tuple[int, float]:
    """Choose the closest major instance; ties retain the original first-instance rule."""
    distances = [float(np.min(tree.query(points)[0])) for tree in trees]
    index = int(np.argmin(distances))
    return index, distances[index]


def instance_filename(role: str, instance_index: int) -> str:
    return f"{role}_inst{instance_index:03d}.ply"


def prior_owned_raw_meshes(manifest_path: Path) -> set[str]:
    """Read only safe raw-mesh names that the preceding extraction manifest owned."""
    if not manifest_path.exists():
        return set()
    try:
        with open(manifest_path, "r") as f:
            prior_manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return set()
    instances = prior_manifest.get("instances", []) if isinstance(prior_manifest, dict) else []
    names = set()
    for instance in instances:
        if not isinstance(instance, dict):
            continue
        role = instance.get("role")
        name = instance.get("file_name")
        instance_name = instance.get("instance_name")
        path = Path(name) if isinstance(name, str) else None
        legacy_names = {
            "floor": {"floor_level0.ply", "floor_mezzanine_level1.ply"},
            "ceiling": {"ceiling_main.ply", "ceiling_mezzanine.ply"},
            "wall": {"wall_main.ply"},
        }
        is_indexed = isinstance(role, str) and isinstance(name, str) and (
            name.startswith(f"{role}_inst") and name.removesuffix(".ply").removeprefix(f"{role}_inst").isdigit()
        )
        if (
            role in {"floor", "ceiling", "wall", "stair"}
            and isinstance(name, str)
            and instance_name == path.stem
            and path
            and not path.is_absolute()
            and path.name == name
            and path.suffix == ".ply"
            and (is_indexed or name in legacy_names.get(role, set()))
        ):
            names.add(name)
    return names


def export_colored_ply(
    file_path: Path | str,
    vertices: np.ndarray,
    triangles: np.ndarray,
    vertex_colors: np.ndarray,
) -> bool:
    """Exports binary little-endian PLY mesh with vertex normals and colors compatible with Blender."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))

    if vertex_colors.dtype == np.uint8:
        v_colors_f64 = vertex_colors.astype(np.float64) / 255.0
    else:
        v_colors_f64 = np.clip(vertex_colors.astype(np.float64), 0.0, 1.0)
    mesh.vertex_colors = o3d.utility.Vector3dVector(v_colors_f64)
    mesh.compute_vertex_normals()

    return o3d.io.write_triangle_mesh(
        str(file_path),
        mesh,
        write_ascii=False,
        compressed=False,
        write_vertex_normals=True,
        write_vertex_colors=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structural classes (walls, floors, ceilings) as separate big instances."
    )
    parser.add_argument(
        "--mesh-path",
        type=Path,
        default=Path("backend/current_scene/06_mesh_elaboration/original_mesh.ply"),
        help="Path to original colored mesh PLY",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("backend/current_scene/06_mesh_elaboration/generic_graph_regions/triangle_class_labels.npy"),
        help="Path to triangle class labels array (.npy)",
    )
    parser.add_argument(
        "--structural-json",
        type=Path,
        default=Path("backend/current_scene/06_mesh_elaboration/generic_graph_regions/structural_classes.json"),
        help="Path to structural classes JSON metadata",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backend/current_scene/06_mesh_elaboration/extracted_structural_instances"),
        help="Output directory for extracted meshes and manifest",
    )
    parser.add_argument(
        "--min-big-area",
        type=float,
        default=1.0,
        help="Minimum surface area (m^2) to qualify a component as a separate major instance",
    )
    parser.add_argument(
        "--merge-dist-thresh",
        type=float,
        default=0.25,
        help="Max distance (meters) to merge a small component into the closest big instance",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.mesh_path.exists():
        print(f"Error: Mesh not found at {args.mesh_path}")
        return 1
    if not args.labels_path.exists():
        print(f"Error: Labels not found at {args.labels_path}")
        return 1

    print(f"Loading mesh: {args.mesh_path}...")
    mesh = o3d.io.read_triangle_mesh(str(args.mesh_path))
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertex_colors = np.asarray(mesh.vertex_colors)
    if len(vertex_colors) == 0:
        vertex_colors = np.full((len(vertices), 3), 0.5, dtype=np.float64)
    print(f"Loaded mesh: {len(vertices):,} vertices, {len(triangles):,} triangles")

    print(f"Loading labels: {args.labels_path}...")
    labels = np.load(args.labels_path)
    if len(labels) != len(triangles):
        print(f"Error: Label count {len(labels)} does not match triangle count {len(triangles)}")
        return 1

    structural_classes_by_role: Dict[str, List[int]] = {
        "floor": [],
        "ceiling": [],
        "wall": [],
        "stair": [],
    }

    if not args.structural_json.exists():
        print(f"Error: Structural classification metadata not found at {args.structural_json}")
        return 1
    with open(args.structural_json, "r") as f:
        struct_data = json.load(f)
    classifications = struct_data.get("classifications")
    if not isinstance(classifications, dict):
        print(f"Error: Structural classification metadata at {args.structural_json} has no classifications object")
        return 1
    for cid_str, info in classifications.items():
        cid = int(cid_str)
        role = info.get("structural_role", "").lower()
        if "floor" in role:
            structural_classes_by_role["floor"].append(cid)
        elif "ceiling" in role:
            structural_classes_by_role["ceiling"].append(cid)
        elif "wall" in role:
            structural_classes_by_role["wall"].append(cid)
        elif "stair" in role:
            structural_classes_by_role["stair"].append(cid)

    print("\nIdentified structural classes:")
    for role, cids in structural_classes_by_role.items():
        print(f"  {role.upper()}: classes {cids}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "structural_instances_manifest.json"
    prior_raw_meshes = prior_owned_raw_meshes(manifest_path)
    staging = tempfile.TemporaryDirectory(dir=args.output_dir, prefix=".extract_structural_")
    staging_dir = Path(staging.name)

    manifest: Dict[str, Any] = {
        "source_mesh": str(args.mesh_path),
        "total_source_vertices": len(vertices),
        "total_source_triangles": len(triangles),
        "roles": {},
        "instances": [],
    }

    for role, cids in structural_classes_by_role.items():
        if not cids:
            print(f"\nNo classes found for role: {role}")
            continue

        print(f"\n==================================================")
        print(f"Processing role: {role.upper()} (Class IDs: {cids})")
        print(f"==================================================")

        tri_mask = np.isin(labels, cids)
        role_tri_indices = np.where(tri_mask)[0]
        if len(role_tri_indices) == 0:
            print(f"No triangles found for role {role}")
            continue

        role_triangles = triangles[role_tri_indices]
        unique_v_idx, remapped_triangles = np.unique(role_triangles, return_inverse=True)
        remapped_triangles = remapped_triangles.reshape(role_triangles.shape)
        role_vertices = vertices[unique_v_idx]

        sub_mesh = o3d.geometry.TriangleMesh()
        sub_mesh.vertices = o3d.utility.Vector3dVector(role_vertices)
        sub_mesh.triangles = o3d.utility.Vector3iVector(remapped_triangles)

        triangle_clusters, comp_tri_counts, comp_areas = sub_mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        comp_tri_counts = np.asarray(comp_tri_counts)
        comp_areas = np.asarray(comp_areas)
        component_faces = component_face_indices(triangle_clusters, len(comp_tri_counts))

        num_components = len(comp_tri_counts)
        print(f"Found {num_components} connected components across {len(role_tri_indices):,} triangles")

        big_indices = np.where(comp_areas >= args.min_big_area)[0]
        if len(big_indices) == 0:
            big_indices = np.array([np.argmax(comp_areas)])

        big_indices = big_indices[np.argsort(-comp_areas[big_indices])]
        small_indices = np.setdiff1d(np.arange(num_components), big_indices)

        print(f"  Major instances (area >= {args.min_big_area} m²): {len(big_indices)}")
        for i, b_idx in enumerate(big_indices):
            print(f"    Major #{i}: {comp_tri_counts[b_idx]:,} tris, {comp_areas[b_idx]:.2f} m²")
        print(f"  Small components: {len(small_indices)} (total {np.sum(comp_tri_counts[small_indices]):,} tris, {np.sum(comp_areas[small_indices]):.2f} m²)")

        big_instance_trees = []
        for b_idx in big_indices:
            b_faces = remapped_triangles[component_faces[b_idx]]
            b_v_indices = np.unique(b_faces)
            b_pts = role_vertices[b_v_indices]
            big_instance_trees.append(cKDTree(b_pts))

        assigned_triangles_per_big: List[List[int]] = [
            role_tri_indices[component_faces[b_idx]].tolist() for b_idx in big_indices
        ]
        merged_small_count_per_big = [0] * len(big_indices)
        merged_small_tris_per_big = [0] * len(big_indices)
        discarded_small_count = 0
        discarded_small_tris = 0

        thresh = args.merge_dist_thresh
        if role == "ceiling":
            thresh = max(thresh, 0.30)

        for s_idx in small_indices:
            s_faces = remapped_triangles[component_faces[s_idx]]
            s_v_indices = np.unique(s_faces)
            s_pts = role_vertices[s_v_indices]

            best_big_idx, best_dist = closest_major_instance(s_pts, big_instance_trees)

            s_orig_tris = role_tri_indices[component_faces[s_idx]].tolist()
            if best_dist <= thresh:
                assigned_triangles_per_big[best_big_idx].extend(s_orig_tris)
                merged_small_count_per_big[best_big_idx] += 1
                merged_small_tris_per_big[best_big_idx] += len(s_orig_tris)
            else:
                discarded_small_count += 1
                discarded_small_tris += len(s_orig_tris)

        print(f"  Small components resolution:")
        for i in range(len(big_indices)):
            print(f"    Major #{i}: merged {merged_small_count_per_big[i]} small components ({merged_small_tris_per_big[i]:,} tris)")
        print(f"    Discarded floaters (> {thresh:.2f}m): {discarded_small_count} components ({discarded_small_tris:,} tris)")

        filenames = []
        for i in range(len(big_indices)):
            filenames.append(instance_filename(role, i))

        role_instance_entries = []
        for i, b_idx in enumerate(big_indices):
            inst_tri_indices = np.array(assigned_triangles_per_big[i], dtype=np.int64)
            inst_triangles = triangles[inst_tri_indices]

            inst_unique_v_idx, inst_remapped_tri = np.unique(inst_triangles, return_inverse=True)
            inst_remapped_tri = inst_remapped_tri.reshape(inst_triangles.shape)
            inst_vertices = vertices[inst_unique_v_idx]
            inst_colors = vertex_colors[inst_unique_v_idx]

            mean_y = float(inst_vertices[:, 1].mean())
            y_min = float(inst_vertices[:, 1].min())
            y_max = float(inst_vertices[:, 1].max())

            filename = filenames[i]
            label_suffix = Path(filename).stem[len(role) + 1:]
            out_file = staging_dir / filename
            final_out_file = args.output_dir / filename

            print(f"  Exporting {out_file.name}: {len(inst_vertices):,} verts, {len(inst_triangles):,} tris (mean Y: {mean_y:.2f}m)...")
            if not export_colored_ply(out_file, inst_vertices, inst_remapped_tri, inst_colors) or not out_file.is_file():
                print(f"Error: Failed to export {out_file}")
                staging.cleanup()
                return 1

            inst_m = o3d.geometry.TriangleMesh()
            inst_m.vertices = o3d.utility.Vector3dVector(inst_vertices)
            inst_m.triangles = o3d.utility.Vector3iVector(inst_remapped_tri)
            inst_area = float(inst_m.get_surface_area())

            inst_meta = {
                "instance_name": f"{role}_{label_suffix}",
                "role": role,
                "file_name": filename,
                "file_path": str(final_out_file),
                "num_vertices": int(len(inst_vertices)),
                "num_triangles": int(len(inst_triangles)),
                "surface_area_m2": round(inst_area, 3),
                "elevation": {
                    "y_mean": round(mean_y, 3),
                    "y_min": round(y_min, 3),
                    "y_max": round(y_max, 3),
                },
                "merged_small_components": int(merged_small_count_per_big[i]),
                "merged_small_triangles": int(merged_small_tris_per_big[i]),
            }
            manifest["instances"].append(inst_meta)
            role_instance_entries.append(inst_meta)

        manifest["roles"][role] = {
            "num_major_instances": len(big_indices),
            "merged_small_components": sum(merged_small_count_per_big),
            "merged_small_triangles": sum(merged_small_tris_per_big),
            "discarded_small_components": discarded_small_count,
            "discarded_small_triangles": discarded_small_tris,
            "instances": [e["instance_name"] for e in role_instance_entries],
        }

    staged_manifest = staging_dir / manifest_path.name
    try:
        with open(staged_manifest, "w") as f:
            json.dump(manifest, f, indent=2)
        for instance in manifest["instances"]:
            (staging_dir / instance["file_name"]).replace(args.output_dir / instance["file_name"])
        staged_manifest.replace(manifest_path)
        new_raw_meshes = {instance["file_name"] for instance in manifest["instances"]}
        for filename in prior_raw_meshes - new_raw_meshes:
            stale_output = args.output_dir / filename
            if stale_output.is_file():
                stale_output.unlink()
    except OSError as exc:
        print(f"Error: Failed to publish extracted meshes: {exc}")
        staging.cleanup()
        return 1
    staging.cleanup()
    print(f"\nManifest saved to: {manifest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
