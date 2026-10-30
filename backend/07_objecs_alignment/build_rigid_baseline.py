#!/usr/bin/env python3
"""Align Stage 05 object instances to the Stage 06 regularized structural shell."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEGMENTATION = ROOT / "current_scene/05_semantic_segmentation/mesh_objects_step3/segmentation_output"
DEFAULT_SHELL = ROOT / "current_scene/06_mesh_elaboration/per_level_collapse_candidate/all_structural_regularized.ply"
DEFAULT_OUTPUT = ROOT / "current_scene/06_mesh_elaboration/new_step/outputs/segmentation_alignment"
FIT_ROLES = ("wall", "floor", "ceiling")
STRUCTURAL_ROLES = {*FIT_ROLES, "stairs", "columns", "column", "ramps", "ramp"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_rigid(source, target, weights, rotation=True):
    weights = weights / weights.sum()
    source_center = np.average(source, axis=0, weights=weights)
    target_center = np.average(target, axis=0, weights=weights)
    if not rotation:
        return np.eye(3), target_center - source_center
    covariance = ((source - source_center) * weights[:, None]).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation_matrix = vt.T @ correction @ u.T
    return rotation_matrix, target_center - rotation_matrix @ source_center


def robust_fit(source, target, weights, normals, allow_rotation, seed=19):
    """Robust point-to-plane translation; optional rigid fit uses deterministic RANSAC."""
    original_count = len(source)
    selected = np.arange(original_count)
    if len(source) > 40_000:
        selected = np.linspace(0, len(source) - 1, 40_000, dtype=np.int64)
        source, target, weights, normals = source[selected], target[selected], weights[selected], normals[selected]
    if not allow_rotation:
        displacement = np.einsum("ij,ij->i", normals, target - source)
        if len(source) < 3:
            matrix = normals * np.sqrt(weights[:, None])
            translation = np.linalg.lstsq(matrix, displacement * np.sqrt(weights), rcond=None)[0]
            inliers = np.ones(original_count, dtype=bool)
            return np.eye(3), translation, inliers
        rng = np.random.default_rng(seed)
        inliers = np.ones(len(source), dtype=bool)
        best = np.zeros(len(source), dtype=bool)
        for _ in range(300):
            sample = rng.choice(len(source), 3, replace=False)
            matrix = normals[sample] * np.sqrt(weights[sample, None])
            if np.linalg.matrix_rank(matrix) < 3:
                continue
            candidate = np.linalg.lstsq(matrix, displacement[sample] * np.sqrt(weights[sample]), rcond=None)[0]
            candidate_inliers = np.abs(normals @ candidate - displacement) <= 0.08
            if weights[candidate_inliers].sum() > weights[best].sum():
                best = candidate_inliers
        if best.sum() >= 3:
            inliers = best
        translation = np.zeros(3)
        for _ in range(8):
            root_weights = np.sqrt(weights[inliers])
            translation = np.linalg.lstsq(normals[inliers] * root_weights[:, None],
                                          displacement[inliers] * root_weights, rcond=None)[0]
            residual = np.abs(normals @ translation - displacement)
            cutoff = min(0.08, max(0.015, float(np.median(residual[inliers]) * 2.5)))
            updated = residual <= cutoff
            if not updated.any() or np.array_equal(updated, inliers):
                break
            inliers = updated
        result = np.zeros(original_count, dtype=bool)
        result[selected[inliers]] = True
        return np.eye(3), translation, result
    rng = np.random.default_rng(seed)
    sample_size = 3
    if len(source) < sample_size:
        raise ValueError(f"Need at least {sample_size} structural correspondences; got {len(source)}")
    best = np.zeros(len(source), dtype=bool)
    for _ in range(300):
        ids = rng.choice(len(source), sample_size, replace=False)
        rotation, translation = fit_rigid(source[ids], target[ids], weights[ids], allow_rotation)
        inliers = np.linalg.norm(source @ rotation.T + translation - target, axis=1) <= 0.08
        if weights[inliers].sum() > weights[best].sum():
            best = inliers
    if best.sum() < sample_size:
        raise ValueError("RANSAC found too few structural inliers to estimate a transform")
    for _ in range(5):
        rotation, translation = fit_rigid(source[best], target[best], weights[best], allow_rotation)
        residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
        cutoff = min(0.08, max(0.015, float(np.median(residual[best]) * 2.5)))
        updated = residual <= cutoff
        if updated.sum() < sample_size or np.array_equal(updated, best):
            break
        best = updated
    rotation, translation = fit_rigid(source[best], target[best], weights[best], allow_rotation)
    inliers = np.zeros(original_count, dtype=bool)
    inliers[selected[best]] = True
    return rotation, translation, inliers


def mesh_scene(mesh):
    triangles = np.asarray(mesh.triangles)
    if not len(triangles):
        raise ValueError("Cannot use an empty target mesh")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-output", type=Path, default=DEFAULT_SEGMENTATION)
    parser.add_argument("--shell", type=Path, default=DEFAULT_SHELL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-rotation", action="store_true", help="Estimate rotation as well as translation")
    args = parser.parse_args()
    started = time.perf_counter()
    package = args.segmentation_output
    metadata_path = package / "class_metadata.json"
    if not metadata_path.is_file():
        metadata_path = package / "segmentation.json"
    classes_dir = package / "3d_masks"
    instances_plys_dir = package / "Instances PLYs"
    for required in (metadata_path, args.shell):
        if not required.is_file():
            raise FileNotFoundError(f"Required Stage 05/06 input is missing: {required}")
    if not classes_dir.is_dir() and not instances_plys_dir.is_dir():
        raise FileNotFoundError(f"Stage 05 instance folder is missing: checked {classes_dir} and {instances_plys_dir}; rerun 3D extraction first")
    raw_meta = json.loads(metadata_path.read_text())
    metadata = raw_meta.get("classes") or raw_meta.get("classifications") or {}
    role_meshes, object_meshes = {role: [] for role in FIT_ROLES}, []
    if classes_dir.is_dir():
        for class_dir in sorted(classes_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_id = int(class_dir.name.split("_")[1])
            info = metadata.get(str(class_id), {})
            role = info.get("structural_role", "object")
            if role not in STRUCTURAL_ROLES | {"object"}:
                raise ValueError(f"Unsupported structural_role {role!r} in {class_dir}")
            instances = sorted(class_dir.glob("instance_*.ply"))
            for path in instances:
                mesh = o3d.io.read_triangle_mesh(str(path))
                if not len(mesh.triangles):
                    continue
                item = {"class_id": class_id, "role": role, "path": path, "mesh": mesh}
                if role in {"object", "columns", "column", "ramps", "ramp"}:
                    object_meshes.append(item)
                elif role in FIT_ROLES:
                    role_meshes[role].append(item)
    elif instances_plys_dir.is_dir():
        role_map = {"Wall": "wall", "Floor": "floor", "Ceiling": "ceiling"}
        for folder_name, role_name in role_map.items():
            s_dir = instances_plys_dir / "Structural" / folder_name
            if s_dir.is_dir():
                for path in sorted(s_dir.glob("instance_*.ply")):
                    mesh = o3d.io.read_triangle_mesh(str(path))
                    if len(mesh.triangles):
                        role_meshes[role_name].append({"class_id": 0, "role": role_name, "path": path, "mesh": mesh})
        for folder_name in ["Columns", "Stairs", "Ramps"]:
            s_dir = instances_plys_dir / "Structural" / folder_name
            if s_dir.is_dir():
                for path in sorted(s_dir.glob("instance_*.ply")):
                    mesh = o3d.io.read_triangle_mesh(str(path))
                    if len(mesh.triangles):
                        cid = 6 if folder_name == "Columns" else 3 if folder_name == "Stairs" else 4
                        object_meshes.append({"class_id": cid, "role": folder_name.lower(), "path": path, "mesh": mesh})
        ns_dir = instances_plys_dir / "Non-Structural"
        if ns_dir.is_dir():
            for c_dir in sorted(ns_dir.iterdir()):
                if not c_dir.is_dir():
                    continue
                try:
                    cid = int(c_dir.name)
                except ValueError:
                    continue
                for path in sorted(c_dir.glob("instance_*.ply")):
                    mesh = o3d.io.read_triangle_mesh(str(path))
                    if len(mesh.triangles):
                        object_meshes.append({"class_id": cid, "role": "object", "path": path, "mesh": mesh})

    shell = o3d.io.read_triangle_mesh(str(args.shell))
    shell.compute_triangle_normals()
    shell_vertices, shell_faces = np.asarray(shell.vertices), np.asarray(shell.triangles)
    shell_normals = np.asarray(shell.triangle_normals)
    role_masks = {"wall": np.abs(shell_normals[:, 1]) < 0.65,
                  "floor": shell_normals[:, 1] >= 0.65,
                  "ceiling": shell_normals[:, 1] <= -0.65}
    role_scenes = {}
    for role, mask in role_masks.items():
        if not role_meshes[role]:
            continue
        ids = np.flatnonzero(mask)
        if not len(ids):
            raise ValueError(f"Regularized shell has no target triangles for role {role}")
        target = o3d.geometry.TriangleMesh()
        used, inverse = np.unique(shell_faces[ids], return_inverse=True)
        target.vertices = o3d.utility.Vector3dVector(shell_vertices[used])
        target.triangles = o3d.utility.Vector3iVector(inverse.reshape(-1, 3))
        role_scenes[role] = mesh_scene(target)

    sources, targets, weights, anchor_normals = [], [], [], []
    for role, items in role_meshes.items():
        for item in items:
            mesh = item["mesh"]
            mesh.compute_triangle_normals()
            vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
            stride = max(1, int(np.ceil(len(faces) / 50_000)))
            sampled = np.arange(0, len(faces), stride)
            tri = vertices[faces[sampled]]
            points = tri.mean(axis=1)
            areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
            normals = np.asarray(mesh.triangle_normals)[sampled]
            scene = role_scenes[role]
            closest_parts, target_normals = [], []
            for start in range(0, len(points), 100_000):
                result = scene.compute_closest_points(o3d.core.Tensor(points[start:start + 100_000].astype(np.float32)))
                closest_parts.append(result["points"].numpy())
                target_normals.append(result["primitive_normals"].numpy())
            closest, target_normals = np.concatenate(closest_parts), np.concatenate(target_normals)
            distance = np.linalg.norm(closest - points, axis=1)
            normal_agreement = np.abs(np.einsum("ij,ij->i", normals, target_normals))
            valid = (np.isfinite(areas) & (areas > 0) & (distance <= 0.25) & (normal_agreement >= 0.5))
            if valid.any():
                sources.append(points[valid])
                targets.append(closest[valid])
                weights.append(areas[valid])
                # Surface normals constrain alignment perpendicular to each fitted plane.
                anchor_normals.append(target_normals[valid])
    if not sources:
        raise ValueError("No wall/floor/ceiling instance faces matched the regularized shell")
    source, target, weight, normals = (np.concatenate(values) for values in
                                       (sources, targets, weights, anchor_normals))
    rotation, translation, inliers = robust_fit(source, target, weight, normals, args.allow_rotation)
    residual_before = np.linalg.norm(source - target, axis=1)
    residual_after = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    plane_before = np.einsum("ij,ij->i", normals, source - target)
    plane_after = np.einsum("ij,ij->i", normals, source @ rotation.T + translation - target)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir = args.output_dir / "rigid_baseline_instances"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    merged_objects = o3d.geometry.TriangleMesh()
    manifest = []
    for item in object_meshes:
        mesh = item["mesh"]
        mesh.transform(np.block([[rotation, translation[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
        relative = Path(f"class_{item['class_id']:03d}") / item["path"].name
        destination = aligned_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_triangle_mesh(str(destination), mesh, write_ascii=False):
            raise IOError(f"Could not write aligned instance {destination}")
        merged_objects += mesh
        manifest.append({"class_id": item["class_id"], "role": item["role"],
                         "source": str(item["path"]), "path": str(relative),
                         "triangles": len(mesh.triangles)})
    objects_path = args.output_dir / "rigid_baseline_objects.ply"
    scene_path = args.output_dir / "rigid_baseline_scene.ply"
    if not o3d.io.write_triangle_mesh(str(objects_path), merged_objects, write_ascii=False):
        raise IOError(f"Could not write aligned object aggregate {objects_path}")
    shutil.copy2(args.shell, args.output_dir / "rigid_baseline_shell.ply")
    if not o3d.io.write_triangle_mesh(str(scene_path), shell + merged_objects, write_ascii=False):
        raise IOError(f"Could not write combined scene {scene_path}")
    angle = np.rad2deg(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0)))
    report = {"status": "segmentation_instance_alignment_candidate",
              "inputs": {"segmentation_output": str(package), "class_metadata": str(metadata_path),
                         "regularized_shell": str(args.shell), "shell_sha256": sha256(args.shell)},
              "transform": {"rotation": rotation.tolist(), "translation_m": translation.tolist(),
                            "rotation_angle_deg": float(angle), "rotation_enabled": args.allow_rotation},
              "fit": {"method": "area-weighted robust point-to-plane translation" if not args.allow_rotation
                      else "deterministic area-weighted point-to-point RANSAC",
                      "inlier_threshold_m": 0.08,
                      "anchors": int(len(source)), "inliers": int(inliers.sum()),
                      "rms_before_m": float(np.sqrt(np.average(plane_before**2, weights=weight))),
                      "rms_after_m": float(np.sqrt(np.average(plane_after[inliers]**2, weights=weight[inliers]))),
                      "point_distance_rms_after_m": float(np.sqrt(np.average(residual_after[inliers]**2, weights=weight[inliers]))),
                      "inlier_fraction": float(inliers.mean())},
              "roles_used_for_fit": list(FIT_ROLES), "fit_structural_instance_meshes_omitted":
                  int(sum(len(items) for items in role_meshes.values())),
              "stairs_omitted_because_shell_retains_them": sum(1 for d in classes_dir.glob("class_*_stairs/instance_*.ply")) if classes_dir.is_dir() else 0,
              "object_instances": manifest, "collision_status": "not evaluated",
              "outputs": {"objects": str(objects_path), "shell": str(args.output_dir / "rigid_baseline_shell.ply"),
                          "combined_scene": str(scene_path), "instances_dir": str(aligned_dir)},
              "runtime_seconds": time.perf_counter() - started}
    (args.output_dir / "rigid_baseline_report.json").write_text(json.dumps(report, indent=2) + "\n")
    from group_touching_instances import build_groups
    groups = build_groups(args.output_dir)
    report["contact_grouping"] = {"manifest": str(args.output_dir / "contact_groups.json"),
                                  "contact_pairs": groups["contact_pair_count"],
                                  "groups": groups["group_count"],
                                  "contact_distance_m": groups["contact_distance_m"]}
    (args.output_dir / "rigid_baseline_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"anchors": len(source), "inliers": int(inliers.sum()),
                      "object_instances": len(manifest), "translation_m": translation.tolist(),
                      "rotation_angle_deg": float(angle), "rms_after_m": report["fit"]["rms_after_m"],
                      "contact_pairs": groups["contact_pair_count"], "contact_groups": groups["group_count"],
                      "scene": str(scene_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
