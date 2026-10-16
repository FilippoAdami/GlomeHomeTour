#!/usr/bin/env python3
"""Estimate provisional object-to-shell contacts from current-mesh instance tracks."""

import argparse
import colorsys
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from analyze_new_step import ROOT, closest_surface


SCENE = ROOT / "current_scene"
INPUT = SCENE / "06_mesh_elaboration/new_step"
OUTPUT = INPUT / "outputs"
RUN = OUTPUT / "panoptic3d_workspace/06c_semantic_segmentation/panoptic3d_current_source_candidate"
ROLE_COLORS = {"wall": (0.2, 0.55, 1.0), "floor": (0.2, 0.9, 0.3), "ceiling": (1.0, 0.65, 0.1)}
MAX_PREVIEW_FACES_PER_INSTANCE = 1200


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def instance_color(instance_id):
    return colorsys.hsv_to_rgb((instance_id * 0.61803398875) % 1, 0.72, 0.92)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact-distance", type=float, default=0.05, help="maximum surface distance in metres")
    parser.add_argument("--normal-alignment", type=float, default=0.7, help="minimum absolute normal dot product")
    parser.add_argument("--minimum-area", type=float, default=0.01, help="minimum view-supported patch area in square metres")
    args = parser.parse_args()
    if args.contact_distance <= 0 or not 0 <= args.normal_alignment <= 1 or args.minimum_area <= 0:
        parser.error("contact distance and minimum area must be positive; normal alignment must be in [0, 1]")

    started = time.perf_counter()
    source_path = INPUT / "original_mesh.ply"
    shell_path = INPUT / "all_structural_regularized.ply"
    source = o3d.io.read_triangle_mesh(str(source_path))
    shell = o3d.io.read_triangle_mesh(str(shell_path))
    source.compute_triangle_normals()
    shell.compute_triangle_normals()
    vertices, triangles = np.asarray(source.vertices), np.asarray(source.triangles)
    source_normals = np.asarray(source.triangle_normals)
    shell_vertices, shell_triangles = np.asarray(shell.vertices), np.asarray(shell.triangles)
    shell_normals = np.asarray(shell.triangle_normals)

    assignments = np.load(RUN / "face_assignments.npz")
    instance_labels = assignments["instance_ids"]
    face_confidence = assignments["confidence"]
    assert len(instance_labels) == len(triangles)
    manifest = json.loads((RUN / "instances_manifest.json").read_text())["instances"]
    observations = json.loads((RUN / "observations.json").read_text())
    structural = json.loads((SCENE / "06_mesh_elaboration/generic_graph_regions/structural_classes.json").read_text())
    structural_ids = {int(cid) for cid, info in structural["classifications"].items()
                      if info.get("structural_role") in ROLE_COLORS}
    region_labels = np.load(SCENE / "06_mesh_elaboration/generic_graph_regions/triangle_class_labels.npy",
                            mmap_mode="r")
    assert len(region_labels) == len(triangles)
    structural_faces = np.isin(region_labels, list(structural_ids))
    candidate_tracks = [entry for entry in manifest
                        if entry["identity_status"] == "multi_view_supported"
                        and entry["pose_reliability"] == "supported"
                        and entry["geometry_triangles"] > 0]
    structural_tracks = [entry for entry in candidate_tracks
                         if structural_ids.intersection(entry["source_ids"])]
    candidate_tracks = [entry for entry in candidate_tracks
                        if not structural_ids.intersection(entry["source_ids"])]
    track_faces = [np.flatnonzero(instance_labels == entry["physical_instance_id"])
                   for entry in candidate_tracks]
    structural_face_counts = [int(structural_faces[ids].sum()) for ids in track_faces]
    eligible_pairs = [(entry, ids[~structural_faces[ids]], structural_count)
                      for entry, ids, structural_count in zip(candidate_tracks, track_faces, structural_face_counts)
                      if np.any(~structural_faces[ids])]
    eligible = [entry for entry, _, _ in eligible_pairs]
    face_groups = [ids for _, ids, _ in eligible_pairs]
    eligible_structural_counts = [count for _, _, count in eligible_pairs]
    face_ids = np.concatenate(face_groups) if face_groups else np.empty(0, np.int64)
    centers = vertices[triangles[face_ids]].mean(axis=1)
    areas = np.linalg.norm(np.cross(vertices[triangles[face_ids, 1]] - vertices[triangles[face_ids, 0]],
                                    vertices[triangles[face_ids, 2]] - vertices[triangles[face_ids, 0]]), axis=1) / 2
    closest, target_face, target_normals = closest_surface(centers, shell_triangles, shell_vertices)
    distances = np.linalg.norm(closest - centers, axis=1)
    normal_dot = np.einsum("ij,ij->i", source_normals[face_ids], target_normals)
    alignment = np.abs(normal_dot)
    role_by_face = np.where(np.abs(target_normals[:, 1]) < 0.65, "wall",
                            np.where(target_normals[:, 1] > 0.65, "floor", "ceiling"))
    close_parallel = (distances <= args.contact_distance) & (alignment >= args.normal_alignment)

    support = np.load(RUN / "observation_support.npz")
    indptr, support_faces = support["indptr"], support["indices"]
    support_shape = tuple(support["shape"])
    assert support_shape == (len(observations), len(triangles))
    source_face_position = np.full(len(triangles), -1, np.int32)
    source_face_position[face_ids] = np.arange(len(face_ids), dtype=np.int32)

    records, preview_points, preview_lines, preview_colors = [], [], [], []
    instance_colors = {}
    contact_preview_faces = [set() for _ in eligible]
    view_group_counts = np.zeros(len(face_ids), np.uint16)
    offset = 0
    for index, entry in enumerate(eligible):
        iid = int(entry["physical_instance_id"])
        color = instance_color(iid)
        instance_colors[iid] = color
        start, stop = offset, offset + len(face_groups[index])
        offset = stop
        local = slice(start, stop)
        groups_seen = {}
        for observation_id in entry["observation_ids"]:
            observation = observations[observation_id]
            groups_seen.setdefault(int(observation["group"]), []).append(int(observation_id))
        view_count = np.zeros(stop - start, np.uint16)
        group_role_counts = {role: 0 for role in ROLE_COLORS}
        for observation_ids in groups_seen.values():
            visible = np.unique(np.concatenate([
                support_faces[indptr[oid]:indptr[oid + 1]] for oid in observation_ids
            ]))
            positions = source_face_position[visible]
            valid = (positions >= start) & (positions < stop)
            positions = positions[valid]
            positions = positions[close_parallel[positions]]
            for role in set(role_by_face[positions]):
                group_role_counts[role] += 1
            view_count[positions - start] += 1
        corroborated = view_count >= 2
        view_group_counts[local] = view_count
        contact_details = []
        total_area = float(areas[local].sum())
        tri_points = vertices[triangles[face_groups[index]]].reshape(-1, 3)
        bounds = [tri_points.min(axis=0).tolist(), tri_points.max(axis=0).tolist()]
        height = bounds[1][1] - bounds[0][1]
        for role in ROLE_COLORS:
            mask = close_parallel[local] & corroborated & (role_by_face[local] == role)
            selected = np.flatnonzero(mask) + start
            if not len(selected):
                continue
            patch_triangles = triangles[face_ids[selected]]
            patch_vertices, compact = np.unique(patch_triangles, return_inverse=True)
            patch_mesh = o3d.geometry.TriangleMesh()
            patch_mesh.vertices = o3d.utility.Vector3dVector(vertices[patch_vertices])
            patch_mesh.triangles = o3d.utility.Vector3iVector(compact.reshape(-1, 3))
            patch_ids, _, _ = patch_mesh.cluster_connected_triangles()
            patch_ids = np.asarray(patch_ids)
            patch_areas = np.bincount(patch_ids, weights=areas[selected])
            largest_patch = int(np.argmax(patch_areas))
            patch_mask = patch_ids == largest_patch
            patch_faces = selected[patch_mask]
            patch_area = float(patch_areas[largest_patch])
            if patch_area < args.minimum_area:
                continue
            weights = areas[patch_faces]
            source_location = np.average(centers[patch_faces], axis=0, weights=weights)
            target_location = np.average(closest[patch_faces], axis=0, weights=weights)
            opposition = float(areas[patch_faces][normal_dot[patch_faces] < 0].sum() / patch_area)
            height_fraction = ((source_location[1] - bounds[0][1]) / height) if height > 1e-8 else 0.5
            if role == "floor":
                relationship = "floor_support_candidate" if height_fraction <= 0.15 else "floor_proximity_unclassified"
            elif role == "ceiling":
                relationship = "ceiling_attachment_candidate" if height_fraction >= 0.85 else "ceiling_proximity_unclassified"
            else:
                relationship = "wall_adjacency_candidate"
            detail = {
                "role": role,
                "status": relationship,
                "largest_connected_patch_area_m2": patch_area,
                "fraction_of_instance_surface": patch_area / total_area if total_area else 0.0,
                "patch_faces": int(len(patch_faces)),
                "near_parallel_faces_before_patch_filter": int(len(selected)),
                "view_groups_observing_any_near_parallel_patch": group_role_counts[role],
                "view_groups_per_patch_face_min_median_max": [int(view_count[patch_faces - start].min()),
                    float(np.median(view_count[patch_faces - start])), int(view_count[patch_faces - start].max())],
                "median_distance_m": float(np.median(distances[patch_faces])),
                "area_weighted_normal_opposition_fraction": opposition,
                "area_weighted_assignment_confidence": float(np.average(
                    face_confidence[face_ids[patch_faces]], weights=weights)),
                "source_location_m": source_location.tolist(),
                "target_location_m": target_location.tolist(),
                "source_height_fraction_of_instance_bounds": float(height_fraction),
                "preview_color_rgb": list(color),
            }
            sample = patch_faces[np.linspace(0, len(patch_faces) - 1, min(12, len(patch_faces)), dtype=int)]
            contact_preview_faces[index].update(map(int, face_ids[sample]))
            points = np.stack((centers[sample], closest[sample]), axis=1).reshape(-1, 3)
            detail["preview_line_start"] = len(preview_colors)
            detail["preview_line_count"] = len(sample)
            detail["preview_source_face_ids"] = face_ids[sample].astype(int).tolist()
            detail["preview_target_face_ids"] = target_face[sample].astype(int).tolist()
            detail["preview_source_points_m"] = centers[sample].tolist()
            detail["preview_target_points_m"] = closest[sample].tolist()
            detail["preview_assignment_confidence"] = face_confidence[face_ids[sample]].tolist()
            detail["preview_view_groups"] = view_count[sample - start].astype(int).tolist()
            preview_points.extend(points)
            preview_lines.extend((np.arange(len(sample), dtype=np.int32)[:, None] * 2
                                  + np.array([[0, 1]], dtype=np.int32) + len(preview_points) - len(points)).tolist())
            preview_colors.extend([color] * len(sample))
            contact_details.append(detail)

        records.append({
            "instance_id": iid,
            "source_ids": entry["source_ids"],
            "identity_status": entry["identity_status"],
            "identity_completeness_status": entry["completeness_status"],
            "independent_views": int(entry["independent_views"]),
            "identity_solver_confidence": float(entry["confidence"]),
            "geometry_components": int(entry["geometry_components"]),
            "assigned_faces_before_structural_filter": int(entry["geometry_triangles"]),
            "structural_faces_excluded": int(eligible_structural_counts[index]),
            "nonstructural_faces_analyzed": int(len(face_groups[index])),
            "structural_face_fraction": float(eligible_structural_counts[index]
                                                  / max(1, entry["geometry_triangles"])),
            "assigned_faces": int(len(face_groups[index])),
            "surface_area_m2": total_area,
            "preview_color_rgb": list(color),
            "bounds_m": bounds,
            "contact_candidates": contact_details,
            "contact_status": ("candidate_pending_visual_review" if any(
                c["status"].endswith("_candidate") for c in contact_details)
                else "proximity_patch_unclassified" if contact_details else "no_corroborated_patch_found"),
        })

    report = {
        "status": "provisional_contact_candidates",
        "candidate_source": str(RUN),
        "source_mesh": str(source_path),
        "regularized_shell": str(shell_path),
        "source_sha256": digest(source_path),
        "shell_sha256": digest(shell_path),
        "parameters": {"contact_distance_m": args.contact_distance,
                       "absolute_normal_alignment_min": args.normal_alignment,
                       "minimum_area_m2": args.minimum_area,
                       "minimum_local_view_groups": 2},
        "normal_role_rule": "shell normals with |Y|<0.65 are walls; +Y are floor-facing; -Y are ceiling-facing",
        "identity_note": "Panoptic tracks are hypotheses from frozen masks/embeddings. Tracks whose source classes are structural were excluded, and source-face-aligned structural triangles were removed from remaining mixed tracks. Contact areas combine distance, parallel normals, and repeated independent-view support; this is not proof of physical attachment.",
        "eligible_instances": len(eligible),
        "candidate_tracks_before_structural_filter": len(candidate_tracks) + len(structural_tracks),
        "tracks_excluded_by_structural_source_class": len(structural_tracks),
        "structural_source_class_ids_excluded": sorted(structural_ids),
        "tracks_with_only_structural_faces": len(candidate_tracks) - len(eligible),
        "identity_quality": {
            "completeness_status_counts": {status: sum(e["completeness_status"] == status for e in eligible)
                                           for status in sorted({e["completeness_status"] for e in eligible})},
            "geometry_components_min_median_max": ([int(min(e["geometry_components"] for e in eligible)),
                float(np.median([e["geometry_components"] for e in eligible])),
                int(max(e["geometry_components"] for e in eligible))] if eligible else [None, None, None]),
            "note": "Geometry components are connected components in the source mesh per track; high counts may indicate fragmented tracks. All eligible tracks remain pending reprojection validation.",
        },
        "instances_with_geometric_patches": sum(bool(row["contact_candidates"]) for row in records),
        "instances_with_supported_relationship_candidates": sum(
            row["contact_status"] == "candidate_pending_visual_review" for row in records),
        "instances": records,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    out_json = OUTPUT / "instance_contact_candidates.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(OUTPUT / "instance_contact_samples.npz", source_face=face_ids[close_parallel],
        instance_id=np.repeat([entry["physical_instance_id"] for entry in eligible], [len(ids) for ids in face_groups])[close_parallel],
        target_face=target_face[close_parallel], source_point=centers[close_parallel],
        target_point=closest[close_parallel], distance=distances[close_parallel],
        normal_dot=normal_dot[close_parallel], shell_role=role_by_face[close_parallel],
        view_groups=view_group_counts[close_parallel])
    preview_face_groups = []
    preview_instances = []
    random = np.random.default_rng(0)
    for index, (group, required) in enumerate(zip(face_groups, contact_preview_faces)):
        if not required:
            continue
        required = np.asarray(sorted(required), dtype=np.int64)
        remaining = np.setdiff1d(group, required, assume_unique=True)
        budget = max(0, MAX_PREVIEW_FACES_PER_INSTANCE - len(required))
        extra = random.choice(remaining, min(budget, len(remaining)), replace=False) if budget else np.empty(0, np.int64)
        preview_face_groups.append(np.sort(np.concatenate((required, extra))))
        preview_instances.append(eligible[index])
    preview_face_ids = np.concatenate(preview_face_groups) if preview_face_groups else np.empty(0, np.int64)
    candidate_triangles = triangles[preview_face_ids]
    if len(candidate_triangles):
        triangle_colors = np.concatenate([
            np.repeat([instance_colors[entry["physical_instance_id"]]], len(group), axis=0)
            for entry, group in zip(preview_instances, preview_face_groups)
        ])
        object_mesh = o3d.geometry.TriangleMesh()
        object_mesh.vertices = o3d.utility.Vector3dVector(vertices[candidate_triangles].reshape(-1, 3))
        object_mesh.triangles = o3d.utility.Vector3iVector(np.arange(len(candidate_triangles) * 3).reshape(-1, 3))
        object_mesh.vertex_colors = o3d.utility.Vector3dVector(np.repeat(triangle_colors, 3, axis=0))
        o3d.io.write_triangle_mesh(str(OUTPUT / "candidate_instances_for_contact_review.ply"), object_mesh)
    if preview_lines:
        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(np.asarray(preview_points))
        lines.lines = o3d.utility.Vector2iVector(np.asarray(preview_lines, dtype=np.int32))
        lines.colors = o3d.utility.Vector3dVector(np.asarray(preview_colors))
        o3d.io.write_line_set(str(OUTPUT / "instance_contact_preview.ply"), lines)
    print(json.dumps({"eligible_instances": len(eligible),
                      "instances_with_geometric_patches": report["instances_with_geometric_patches"],
                      "instances_with_supported_relationship_candidates": report["instances_with_supported_relationship_candidates"],
                      "contact_preview": str(OUTPUT / "instance_contact_preview.ply"),
                      "report": str(out_json), "runtime_seconds": report["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
