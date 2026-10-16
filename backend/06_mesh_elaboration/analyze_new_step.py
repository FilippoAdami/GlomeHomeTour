#!/usr/bin/env python3
"""Milestone A: relate aligned structural scan triangles to the regularized shell."""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "current_scene/06_mesh_elaboration/new_step"
DEFAULT_REGIONS = ROOT / "current_scene/06_mesh_elaboration/generic_graph_regions"
DEFAULT_OUTPUT = DEFAULT_INPUT / "outputs"
SEGMENTATION = ROOT / "current_scene/05_semantic_segmentation"


def closest_surface(points, triangles, vertices, chunk_size=100_000):
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices.astype(np.float32)),
        o3d.core.Tensor(triangles.astype(np.uint32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    closest, primitive, normals = [], [], []
    for start in range(0, len(points), chunk_size):
        result = scene.compute_closest_points(o3d.core.Tensor(points[start:start + chunk_size].astype(np.float32)))
        closest.append(result["points"].numpy())
        primitive.append(result["primitive_ids"].numpy())
        normals.append(result["primitive_normals"].numpy())
    return np.concatenate(closest), np.concatenate(primitive), np.concatenate(normals)


def ambiguous_candidates(points, target_ids, target_triangles, target_vertices,
                         target_normals, matched_faces, matched_normals, chunk_size=100_000):
    centers = target_vertices[target_triangles[target_ids]].mean(axis=1)
    tree = cKDTree(centers)
    k = min(8, len(target_ids))
    ambiguous = np.zeros(len(points), dtype=bool)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        center_distance, nearby = tree.query(points[start:stop], k=k)
        if nearby.ndim == 1:
            nearby, center_distance = nearby[:, None], center_distance[:, None]
        nearby_normals = target_normals[target_ids[nearby]]
        normal_separation = np.abs(np.einsum(
            "ij,ikj->ik", matched_normals[start:stop], nearby_normals
        )) < np.cos(np.deg2rad(25))
        competing = ((target_ids[nearby] != matched_faces[start:stop, None])
                     & (center_distance <= center_distance[:, :1] + 0.025))
        ambiguous[start:stop] = np.any(competing & normal_separation, axis=1)
    return ambiguous


def analyze(input_dir, regions_dir, output_dir, max_distance=0.25):
    started = time.perf_counter()
    original_path = input_dir / "original_mesh.ply"
    shell_path = input_dir / "all_structural_regularized.ply"
    region_path = regions_dir / "triangle_class_labels.npy"
    metadata_path = regions_dir / "structural_classes.json"
    original = o3d.io.read_triangle_mesh(str(original_path))
    shell = o3d.io.read_triangle_mesh(str(shell_path))
    original.compute_triangle_normals()
    shell.compute_triangle_normals()
    vertices = np.asarray(original.vertices)
    triangles = np.asarray(original.triangles)
    sv, st = np.asarray(shell.vertices), np.asarray(shell.triangles)
    labels = np.load(region_path, mmap_mode="r")
    if len(labels) != len(triangles):
        raise ValueError(f"Aligned region labels ({len(labels)}) do not match original faces ({len(triangles)})")
    metadata = json.loads(metadata_path.read_text())["classifications"]
    legacy_label_path = SEGMENTATION / "step2_mesh_projection/triangle_class_labels.npy"
    legacy_labels = np.load(legacy_label_path, mmap_mode="r") if legacy_label_path.exists() else None
    projected_mesh_path = legacy_label_path.parent / "mesh_projected_cleaned.ply"
    projected_mesh = (o3d.io.read_triangle_mesh(str(projected_mesh_path))
                      if legacy_labels is not None and projected_mesh_path.exists() else None)
    projected_triangles = np.asarray(projected_mesh.triangles) if projected_mesh is not None else None
    projection_prefix_aligned = bool(
        legacy_labels is not None and projected_triangles is not None
        and len(projected_triangles) == len(legacy_labels)
        and len(projected_triangles) <= len(triangles)
        and np.array_equal(projected_triangles, triangles[:len(projected_triangles)])
    )
    mapped_labels_path = output_dir / "source_face_global_class_labels.npy"
    if projection_prefix_aligned:
        source_class_labels = np.full(len(triangles), -1, dtype=legacy_labels.dtype)
        source_class_labels[:len(legacy_labels)] = legacy_labels
        np.save(mapped_labels_path, source_class_labels)
    del projected_mesh, projected_triangles
    global_segmentation = SEGMENTATION / "full362_final_pipeline/full_pipeline_benchmark_summary.json"
    global_summary = json.loads(global_segmentation.read_text()) if global_segmentation.exists() else {}
    archived_instances = ROOT.parent / "agents_files/artifacts/panoptic3d/inspection_checkpoints_2026-10-13/01_v13_evaluated_instances.ply"
    archived_instance_faces = None
    if archived_instances.exists():
        with archived_instances.open("rb") as f:
            header = f.read(8192).split(b"end_header", 1)[0].decode("ascii", errors="ignore")
        face_line = next((line for line in header.splitlines() if line.startswith("element face ")), None)
        archived_instance_faces = int(face_line.split()[-1]) if face_line else None
    contact_report = {
        "status": "unavailable",
        "source_faces": int(len(triangles)),
        "required": "per-face or per-vertex physical instance IDs with provenance to the source mesh",
        "evidence": {
            "aligned_region_labels": {"path": str(region_path), "faces": int(len(labels)),
                                      "aligned": bool(len(labels) == len(triangles)),
                                      "meaning": "structural/semantic region classes; not physical instance IDs"},
            "05_projection_labels": {"path": str(legacy_label_path) if legacy_labels is not None else None,
                                     "faces": int(len(legacy_labels)) if legacy_labels is not None else None,
                                     "aligned": bool(legacy_labels is not None and len(legacy_labels) == len(triangles)),
                                     "mesh_path": str(projected_mesh_path) if projected_mesh_path.exists() else None,
                                     "face_order_prefix_aligned": projection_prefix_aligned,
                                     "mapped_source_faces": int(len(legacy_labels)) if projection_prefix_aligned else 0,
                                     "unmapped_source_faces": int(len(triangles) - len(legacy_labels)) if projection_prefix_aligned else int(len(triangles)),
                                     "source_labels_path": str(mapped_labels_path) if projection_prefix_aligned else None,
                                     "meaning": "global class IDs, not physical instances"},
            "05_global_2d_segmentation": {
                "summary_path": str(global_segmentation) if global_summary else None,
                "frames": global_summary.get("num_frames"),
                "global_classes": global_summary.get("final_classes_count"),
                "meaning": "2D globally consistent class IDs; no source-mesh face provenance"
                          if global_summary else "summary unavailable",
            },
            "archived_panoptic3d_candidate": {
                "path": str(archived_instances) if archived_instances.exists() else None,
                "faces": archived_instance_faces,
                "current_mesh_faces": int(len(triangles)),
                "usable": False,
                "reason": "unaccepted candidate on different face topology; exact provenance to current source is unverified",
            },
            "per_class_meshes": "aggregated by class; no source face-to-instance assignment manifest is present",
            "06c_instance_output": "no current_scene/06c_semantic_segmentation instance result is present",
        },
        "instances": [],
        "reason": "No physical instance IDs aligned to this original mesh were found; proximity-only grouping would invent instance boundaries.",
    }
    roles = {role: [int(cid) for cid, info in metadata.items() if info.get("structural_role") == role]
             for role in ("wall", "floor", "ceiling")}
    tri_normals = np.asarray(original.triangle_normals)
    tri_centers = vertices[triangles].mean(axis=1)
    shell_normals = np.asarray(shell.triangle_normals)
    target_role = {
        "wall": np.abs(shell_normals[:, 1]) < 0.65,
        # The regularized shell is consistently outward oriented: floors point up,
        # ceilings point down. Keeping the signs separate prevents floor matches.
        "floor": shell_normals[:, 1] >= 0.65,
        "ceiling": shell_normals[:, 1] <= -0.65,
    }
    records = []
    structural_ids = sorted({class_id for class_ids in roles.values() for class_id in class_ids})
    structural_faces = np.isin(labels, structural_ids)
    vertical_source_faces = np.abs(tri_normals[:, 1]) < 0.65
    horizontal_source_faces = ~vertical_source_faces
    for role, class_ids in roles.items():
        if role == "wall":
            # Some broad semantic classes contain vertical under-stair wall pieces.
            source_mask = structural_faces & vertical_source_faces
        else:
            source_mask = np.isin(labels, class_ids) & horizontal_source_faces
        source_ids = np.flatnonzero(source_mask)
        target_ids = np.flatnonzero(target_role[role])
        if not len(source_ids) or not len(target_ids):
            continue
        cp, local_primitive, target_normals = closest_surface(
            tri_centers[source_ids], st[target_ids], sv
        )
        target_face = target_ids[local_primitive]
        distance = np.linalg.norm(cp - tri_centers[source_ids], axis=1)
        alignment = np.abs(np.einsum("ij,ij->i", tri_normals[source_ids], target_normals))
        # Competing nearby shell faces with a different normal mark corners/overlaps ambiguous.
        ambiguous = ambiguous_candidates(tri_centers[source_ids], target_ids, st, sv,
                                         shell_normals, target_face, target_normals)
        confidence = np.exp(-distance / max_distance) * np.clip((alignment - 0.35) / 0.65, 0, 1)
        accepted = (distance <= max_distance) & (alignment >= 0.5) & ~ambiguous
        records.append((role, source_ids, target_face, cp, distance, alignment, confidence, accepted, ambiguous))

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    summary = {"inputs": {"original_mesh": str(original_path), "regularized_shell": str(shell_path),
                           "aligned_structural_labels": str(region_path)},
               "parameters": {"max_distance_m": max_distance}, "roles": {},
               "contacts": contact_report}
    viz_points, viz_colors, viz_lines = [], [], []
    offset = 0
    viz_stride = 100
    for role, source_ids, target_face, cp, distance, alignment, confidence, accepted, ambiguous in records:
        summary["roles"][role] = {"source_triangles": int(len(source_ids)),
            "matched": int(accepted.sum()), "unmatched": int((~accepted).sum()),
            "ambiguous_rejected": int(ambiguous.sum()),
            "mean_distance_m": float(distance.mean()), "p95_distance_m": float(np.percentile(distance, 95)),
            "matched_mean_distance_m": float(distance[accepted].mean()) if accepted.any() else None,
            "matched_p95_distance_m": float(np.percentile(distance[accepted], 95)) if accepted.any() else None,
            "mean_normal_alignment": float(alignment.mean()), "mean_confidence": float(confidence.mean())}
        arrays[role + "_source_face"] = source_ids
        arrays[role + "_target_face"] = target_face
        arrays[role + "_target_point"] = cp
        arrays[role + "_displacement"] = cp - tri_centers[source_ids]
        arrays[role + "_distance"] = distance
        arrays[role + "_normal_alignment"] = alignment
        arrays[role + "_confidence"] = confidence
        arrays[role + "_accepted"] = accepted
        arrays[role + "_ambiguous"] = ambiguous
        sample = np.arange(0, len(source_ids), viz_stride)
        viz_points.append(np.stack((tri_centers[source_ids[sample]], cp[sample]), axis=1).reshape(-1, 3))
        viz_lines.append(np.arange(offset, offset + 2 * len(sample), 2, dtype=np.int32)[:, None] + [0, 1])
        colors = np.stack((1.0 - confidence[sample], confidence[sample], np.zeros_like(confidence[sample])), axis=1)
        colors[~accepted[sample]] = (0.55, 0.55, 0.55)
        colors[ambiguous[sample]] = (1.0, 0.0, 1.0)
        viz_colors.append(colors)
        offset += 2 * len(sample)
    np.savez_compressed(output_dir / "structural_correspondence.npz", **arrays)
    if viz_points:
        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(np.concatenate(viz_points))
        lines.lines = o3d.utility.Vector2iVector(np.concatenate(viz_lines))
        lines.colors = o3d.utility.Vector3dVector(np.concatenate(viz_colors))
        o3d.io.write_line_set(str(output_dir / "structural_correspondence.ply"), lines)
    summary["runtime_seconds"] = round(time.perf_counter() - started, 3)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summary["peak_rss_mib"] = round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
    summary["shell_faces"] = int(len(st))
    summary["original_faces"] = int(len(triangles))
    summary["visualization_sample_stride"] = viz_stride
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "contact_analysis.json").write_text(json.dumps(contact_report, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--regions-dir", type=Path, default=DEFAULT_REGIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-distance", type=float, default=0.25, help="Reject farther correspondences (metres)")
    args = parser.parse_args()
    print(json.dumps(analyze(args.input_dir, args.regions_dir, args.output_dir, args.max_distance), indent=2))


if __name__ == "__main__":
    main()
