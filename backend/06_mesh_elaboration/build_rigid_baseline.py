#!/usr/bin/env python3
"""Align the source structure and apply the same transform to retained objects."""

import hashlib
import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "current_scene/06_mesh_elaboration/new_step"
OUTPUT = INPUT / "outputs"
REGIONS = ROOT / "current_scene/06_mesh_elaboration/generic_graph_regions"
STRUCTURAL_ROLES = {"wall", "floor", "ceiling"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_global_rigid(source, target, weights):
    weights = weights / weights.sum()
    source_center = np.average(source, axis=0, weights=weights)
    target_center = np.average(target, axis=0, weights=weights)
    covariance = ((source - source_center) * weights[:, None]).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def fit_translation(source, target, weights):
    weights = weights / weights.sum()
    return np.eye(3), np.average(target - source, axis=0, weights=weights)


def border_clusters(faces, instance_ids):
    """Connected components of instance IDs that share a source mesh edge."""
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    owners = np.tile(instance_ids, 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owners = edges[order], owners[order]
    shared = np.all(edges[1:] == edges[:-1], axis=1)
    pairs = np.stack((owners[:-1][shared], owners[1:][shared]), axis=1)
    pairs = pairs[(pairs[:, 0] > 0) & (pairs[:, 1] > 0) & (pairs[:, 0] != pairs[:, 1])]
    ids = np.unique(instance_ids[instance_ids > 0])
    parent = {int(i): int(i) for i in ids}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        ra, rb = root(int(a)), root(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for i in ids:
        groups.setdefault(root(int(i)), []).append(int(i))
    return sorted((sorted(group) for group in groups.values()), key=lambda group: group[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-rotation", action="store_true",
                        help="Fit a weighted rigid rotation as well as translation")
    args = parser.parse_args()
    started = time.perf_counter()
    source_path, shell_path = INPUT / "original_mesh.ply", INPUT / "all_structural_regularized.ply"
    source_hash, shell_hash = sha256(source_path), sha256(shell_path)
    source = o3d.io.read_triangle_mesh(str(source_path))
    vertices, faces = np.asarray(source.vertices), np.asarray(source.triangles)
    source.compute_triangle_normals()
    source_normals = np.asarray(source.triangle_normals)
    shell = o3d.io.read_triangle_mesh(str(shell_path))
    shell.compute_triangle_normals()
    shell_vertices, shell_faces = np.asarray(shell.vertices), np.asarray(shell.triangles)
    shell_normals = np.asarray(shell.triangle_normals)
    shell_centers = shell_vertices[shell_faces].mean(axis=1)

    region_labels = np.load(REGIONS / "triangle_class_labels.npy", mmap_mode="r")
    if len(region_labels) != len(faces):
        raise ValueError("Structural labels do not align with the original mesh faces")
    structural = json.loads((REGIONS / "structural_classes.json").read_text())["classifications"]
    structural_ids = [int(class_id) for class_id, info in structural.items()
                      if info.get("structural_role") in STRUCTURAL_ROLES]
    remove = np.isin(region_labels, structural_ids)

    # The regularized shell replaces unlabeled scan fragments that are clearly ceiling.
    floor_top = shell_centers[shell_normals[:, 1] >= 0.65, 1].max()
    roof = ((shell_normals[:, 1] <= -0.65) & (shell_centers[:, 1] > floor_top + 0.15))
    roof_faces = shell_faces[roof]
    roof_vertices, roof_inverse = np.unique(roof_faces, return_inverse=True)
    roof_mesh = o3d.geometry.TriangleMesh()
    roof_mesh.vertices = o3d.utility.Vector3dVector(shell_vertices[roof_vertices])
    roof_mesh.triangles = o3d.utility.Vector3iVector(roof_inverse.reshape(-1, 3))
    roof_scene = o3d.t.geometry.RaycastingScene()
    roof_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(roof_mesh))
    centers = vertices[faces].mean(axis=1)
    extra_ceiling = ((region_labels == -1) & (np.abs(source_normals[:, 1]) >= 0.8)
                     & (centers[:, 1] > floor_top + 0.15))
    candidate_ids = np.flatnonzero(extra_ceiling)
    for start in range(0, len(candidate_ids), 100_000):
        ids = candidate_ids[start:start + 100_000]
        closest = roof_scene.compute_closest_points(o3d.core.Tensor(centers[ids].astype(np.float32)))
        distance = np.linalg.norm(closest["points"].numpy() - centers[ids], axis=1)
        alignment = np.abs(np.einsum("ij,ij->i", source_normals[ids],
                                     closest["primitive_normals"].numpy()))
        extra_ceiling[ids] &= (distance <= 0.10) & (alignment >= 0.7)
    remove |= extra_ceiling

    correspondence = np.load(OUTPUT / "structural_correspondence.npz")
    source_anchors, target_anchors, weights, anchor_roles = [], [], [], []
    tri = vertices[faces]
    for role in ("wall", "floor", "ceiling"):
        accepted = correspondence[f"{role}_accepted"] & ~correspondence[f"{role}_ambiguous"]
        face_ids = correspondence[f"{role}_source_face"][accepted]
        anchors = tri[face_ids]
        area = 0.5 * np.linalg.norm(np.cross(anchors[:, 1] - anchors[:, 0],
                                             anchors[:, 2] - anchors[:, 0]), axis=1)
        confidence = correspondence[f"{role}_confidence"][accepted]
        source_anchors.append(anchors.mean(axis=1))
        target_anchors.append(correspondence[f"{role}_target_point"][accepted])
        weights.append(area * np.maximum(confidence, 0.01))
        anchor_roles.append(np.full(len(face_ids), role))

    source_anchors = np.concatenate(source_anchors)
    target_anchors = np.concatenate(target_anchors)
    weights = np.concatenate(weights)
    anchor_roles = np.concatenate(anchor_roles)
    valid = (np.isfinite(source_anchors).all(axis=1) & np.isfinite(target_anchors).all(axis=1)
             & np.isfinite(weights) & (weights > 0))
    source_anchors, target_anchors = source_anchors[valid], target_anchors[valid]
    weights, anchor_roles = weights[valid], anchor_roles[valid]
    if not len(weights):
        raise ValueError("No finite, positive-weight structural correspondences are available")
    fit = fit_global_rigid if args.allow_rotation else fit_translation
    rotation, translation = fit(source_anchors, target_anchors, weights)
    residual_before = np.linalg.norm(source_anchors - target_anchors, axis=1)
    residual_after = np.linalg.norm(source_anchors @ rotation.T + translation - target_anchors, axis=1)

    def alignment_stats(residual, mask):
        role_weights = weights[mask]
        values = residual[mask]
        return {"anchors": int(mask.sum()),
                "area_confidence_weighted_rms_m": float(np.sqrt(np.average(values**2, weights=role_weights))),
                "p50_m": float(np.percentile(values, 50)),
                "p95_m": float(np.percentile(values, 95))}

    role_stats = {role: {"before": alignment_stats(residual_before, anchor_roles == role),
                         "after": alignment_stats(residual_after, anchor_roles == role)}
                  for role in ("wall", "floor", "ceiling")}

    source_face_ids = np.flatnonzero(~remove)
    used_vertices, inverse = np.unique(faces[source_face_ids], return_inverse=True)
    object_mesh = o3d.geometry.TriangleMesh()
    object_mesh.vertices = o3d.utility.Vector3dVector(vertices[used_vertices] @ rotation.T + translation)
    object_mesh.triangles = o3d.utility.Vector3iVector(inverse.reshape(-1, 3))
    colors = np.asarray(source.vertex_colors)
    if len(colors):
        object_mesh.vertex_colors = o3d.utility.Vector3dVector(colors[used_vertices])
    object_mesh.compute_vertex_normals()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    objects_path = OUTPUT / "rigid_baseline_objects.ply"
    o3d.io.write_triangle_mesh(str(objects_path), object_mesh, write_ascii=False)

    shell_copy = OUTPUT / "rigid_baseline_shell.ply"
    import shutil
    shutil.copyfile(shell_path, shell_copy)
    combined = shell + object_mesh
    o3d.io.write_triangle_mesh(str(OUTPUT / "rigid_baseline_scene.ply"), combined, write_ascii=False)
    assignment_path = OUTPUT / "panoptic3d_workspace/06c_semantic_segmentation/panoptic3d_current_source_candidate/face_assignments.npz"
    assignments = np.load(assignment_path)
    all_instance_ids = assignments["instance_ids"]
    if len(all_instance_ids) != len(faces):
        raise ValueError("Instance assignments do not align with the original mesh")
    instance_ids = all_instance_ids[source_face_ids]
    np.savez_compressed(OUTPUT / "rigid_baseline_provenance.npz",
                        source_face_ids=source_face_ids, instance_ids=instance_ids)

    # Instance PLY face order is the ascending source-face order used by the exporter.
    instance_dir = assignment_path.parent / "instances"
    aligned_dir = OUTPUT / "rigid_baseline_instances"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    cluster_labels = all_instance_ids.copy()
    cluster_labels[remove] = 0
    clusters = border_clusters(faces, cluster_labels)
    cluster_for = {iid: index for index, group in enumerate(clusters) for iid in group}
    written_instances, instance_source_ids = [], []
    for iid in np.unique(instance_ids[instance_ids > 0]):
        source_ids = np.flatnonzero((all_instance_ids == iid) & ~assignments["removed"])
        keep = ~remove[source_ids]
        if not np.any(keep):
            continue
        input_path = instance_dir / f"instance_{iid:04d}.ply"
        instance_mesh = o3d.io.read_triangle_mesh(str(input_path))
        if len(instance_mesh.triangles) != len(source_ids):
            raise ValueError(f"Instance PLY face order/count mismatch: {input_path}")
        instance_faces = np.asarray(instance_mesh.triangles)[keep]
        instance_vertices = np.unique(instance_faces)
        remap = np.full(len(instance_mesh.vertices), -1, dtype=np.int64)
        remap[instance_vertices] = np.arange(len(instance_vertices))
        aligned = o3d.geometry.TriangleMesh()
        aligned.vertices = o3d.utility.Vector3dVector(
            np.asarray(instance_mesh.vertices)[instance_vertices] @ rotation.T + translation)
        aligned.triangles = o3d.utility.Vector3iVector(remap[instance_faces].astype(np.int32))
        colors = np.asarray(instance_mesh.vertex_colors)
        if len(colors):
            aligned.vertex_colors = o3d.utility.Vector3dVector(colors[instance_vertices])
        aligned.compute_vertex_normals()
        path = aligned_dir / f"instance_{iid:04d}.ply"
        o3d.io.write_triangle_mesh(str(path), aligned, write_ascii=False)
        instance_source_ids.append(source_ids[keep])
        written_instances.append({"instance_id": int(iid), "cluster_id": cluster_for[int(iid)],
                                  "source_faces": int(len(source_ids)),
                                  "faces_written": int(keep.sum()), "path": path.name,
                                  "provenance_offset": int(sum(map(len, instance_source_ids[:-1]))),
                                  "provenance_count": int(keep.sum())})
    np.savez_compressed(OUTPUT / "rigid_baseline_instance_provenance.npz",
                        source_face_ids=np.concatenate(instance_source_ids) if instance_source_ids else np.empty(0, np.int64))
    untracked_ids = source_face_ids[instance_ids <= 0]
    untracked_faces = faces[untracked_ids]
    untracked_vertices, untracked_inverse = np.unique(untracked_faces, return_inverse=True)
    untracked = o3d.geometry.TriangleMesh()
    untracked.vertices = o3d.utility.Vector3dVector(vertices[untracked_vertices] @ rotation.T + translation)
    untracked.triangles = o3d.utility.Vector3iVector(untracked_inverse.reshape(-1, 3).astype(np.int32))
    source_colors = np.asarray(source.vertex_colors)
    if len(source_colors):
        untracked.vertex_colors = o3d.utility.Vector3dVector(source_colors[untracked_vertices])
    untracked.compute_vertex_normals()
    if len(untracked_ids):
        o3d.io.write_triangle_mesh(str(aligned_dir / "untracked.ply"), untracked, write_ascii=False)
    np.savez_compressed(OUTPUT / "rigid_baseline_untracked_provenance.npz", source_face_ids=untracked_ids)
    (OUTPUT / "rigid_baseline_clusters.json").write_text(json.dumps({
        "method": "instances sharing a source triangle edge are grouped",
        "clusters": [{"cluster_id": i, "instance_ids": group} for i, group in enumerate(clusters)],
        "untracked_faces": int(len(untracked_ids)),
        "instances": written_instances}, indent=2) + "\n")

    angle = np.rad2deg(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0)))
    report = {"status": "global_structural_alignment_candidate", "source_sha256": source_hash,
              "shell_sha256": shell_hash, "shell_copy_sha256": sha256(shell_copy),
              "global_alignment": {"rotation": rotation.tolist(), "translation_m": translation.tolist(),
                                   "rotation_angle_deg": float(angle),
                                   "rotation_enabled": args.allow_rotation,
                                   "fit_method": "weighted rigid transform" if args.allow_rotation else "weighted translation only",
                                   "fit_weights": "source triangle area × correspondence confidence",
                                   "overall": {"anchors": int(len(source_anchors)),
                                               "area_confidence_weighted_rms_before_m": float(np.sqrt(np.average(residual_before**2, weights=weights))),
                                               "area_confidence_weighted_rms_after_m": float(np.sqrt(np.average(residual_after**2, weights=weights))),
                                               "p50_before_m": float(np.percentile(residual_before, 50)),
                                               "p95_before_m": float(np.percentile(residual_before, 95)),
                                               "p50_after_m": float(np.percentile(residual_after, 50)),
                                               "p95_after_m": float(np.percentile(residual_after, 95))},
                                   "roles": role_stats},
              "source_faces_excluded_as_structural": int(np.isin(region_labels, structural_ids).sum()),
              "unlabeled_ceiling_faces_replaced_by_shell": int(extra_ceiling.sum()),
              "source_faces_replaced_by_shell_total": int(remove.sum()),
              "object_faces_written": int(len(source_face_ids)),
              "object_faces_removed_for_boundary_violations": 0,
              "boundary_status": "one global rigid transform is applied to every retained source object; no per-instance motion or boundary clipping",
              "collision_status": "local shell penetration and object-object collisions not checked",
              "runtime_seconds": time.perf_counter() - started}
    (OUTPUT / "rigid_baseline_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"object_faces_written": report["object_faces_written"],
                      "translation_m": translation.tolist(), "rotation_angle_deg": float(angle),
                      "structural_fit_rms_after_m": float(np.sqrt(np.average(residual_after**2, weights=weights))),
                      "shell_unchanged": report["shell_sha256"] == report["shell_copy_sha256"],
                      "runtime_seconds": report["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
