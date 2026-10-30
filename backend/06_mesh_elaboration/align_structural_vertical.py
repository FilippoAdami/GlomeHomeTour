#!/usr/bin/env python3
"""Level Stage 06 instance meshes from unconstrained wall-plane supports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import open3d as o3d
from plyfile import PlyData

ALIGNMENT_ALGORITHM = "wall_plane_level_v1"


def _rotation_to_up(normal: np.ndarray) -> np.ndarray:
    source = np.asarray(normal, dtype=np.float64)
    source /= np.linalg.norm(source)
    target = np.array([0.0, 1.0, 0.0])
    cross = np.cross(source, target)
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-12:
        return np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])
    skew = np.array([[0.0, -cross[2], cross[1]],
                     [cross[2], 0.0, -cross[0]],
                     [-cross[1], cross[0], 0.0]])
    return np.eye(3) + skew + (skew @ skew) / (1.0 + cosine)


def _estimate_up(normals: np.ndarray, areas: np.ndarray) -> tuple[np.ndarray, float]:
    normals = np.asarray(normals, dtype=np.float64).copy()
    areas = np.asarray(areas, dtype=np.float64)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    normals *= np.where(normals[:, 1] >= 0.0, 1.0, -1.0)[:, None]
    _, inverse = np.unique(np.rint(normals / 0.02).astype(np.int16), axis=0, return_inverse=True)
    bin_weights = np.bincount(inverse, weights=areas)
    mode_members = inverse == int(np.argmax(bin_weights))
    mode = np.average(normals[mode_members], axis=0, weights=areas[mode_members])
    mode /= np.linalg.norm(mode)
    support = normals @ mode >= np.cos(np.deg2rad(30.0))
    up = np.average(normals[support], axis=0, weights=areas[support])
    up /= np.linalg.norm(up)
    return up, float(areas[support].sum() / areas.sum())


def _estimate_wall_up(normals: np.ndarray, areas: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Find the common normal plane of two dominant, unconstrained wall families."""
    normals = np.asarray(normals, dtype=np.float64).copy()
    areas = np.asarray(areas, dtype=np.float64)
    valid = (np.isfinite(normals).all(axis=1) & np.isfinite(areas)
             & (areas > 1e-10) & (np.linalg.norm(normals, axis=1) > 1e-10))
    normals, areas = normals[valid], areas[valid]
    if len(normals) < 2:
        return None
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    first_nonzero = np.argmax(np.abs(normals) > 1e-8, axis=1)
    sign = normals[np.arange(len(normals)), first_nonzero]
    normals *= np.where(sign < 0.0, -1.0, 1.0)[:, None]

    # Quantize the face normals first; candidate scoring then scales with the
    # number of normal directions rather than the potentially million-face mesh.
    _, inverse = np.unique(np.rint(normals / 0.04).astype(np.int16), axis=0, return_inverse=True)
    weights = np.bincount(inverse, weights=areas)
    sums = np.zeros((len(weights), 3), dtype=np.float64)
    np.add.at(sums, inverse, normals * areas[:, None])
    modes = sums / np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-12)
    # shortcut: searches 64 dominant normal bins; raise for scenes with many major wall directions.
    candidates = np.argsort(weights)[-64:]
    best_up, best_support = None, 0.0
    max_normal_component = np.sin(np.deg2rad(10.0))
    for index, first in enumerate(candidates):
        for second in candidates[:index]:
            if abs(float(modes[first] @ modes[second])) > 0.45:
                continue
            up = np.cross(modes[first], modes[second])
            length = float(np.linalg.norm(up))
            if length < 1e-8:
                continue
            up /= length
            support = float(weights[np.abs(modes @ up) <= max_normal_component].sum())
            if support > best_support:
                best_up, best_support = up, support
    if best_up is None:
        return None

    inliers = np.abs(normals @ best_up) <= max_normal_component
    for _ in range(2):
        if areas[inliers].sum() <= 1e-10:
            return None
        covariance = (normals[inliers] * areas[inliers, None]).T @ normals[inliers]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if eigenvalues[1] < 0.12 * eigenvalues.sum() or eigenvalues[1] / max(eigenvalues[2], 1e-12) < 0.35:
            return None
        best_up = eigenvectors[:, 0]
        if best_up[1] < 0.0:
            best_up = -best_up
        inliers = np.abs(normals @ best_up) <= max_normal_component
    support_fraction = float(areas[inliers].sum() / areas.sum())
    if support_fraction < 0.25:
        return None
    best_up /= np.linalg.norm(best_up)
    return best_up, support_fraction


def _transformed_ply(source: Path, destination: Path, rotation: np.ndarray, translation: np.ndarray) -> None:
    ply = PlyData.read(str(source))
    if "vertex" not in ply:
        raise ValueError(f"PLY has no vertex element: {source}")
    vertices = ply["vertex"].data
    names = set(vertices.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"PLY has no x/y/z vertex properties: {source}")
    points = np.column_stack((vertices["x"], vertices["y"], vertices["z"]))
    transformed = points @ rotation.T + translation
    vertices["x"], vertices["y"], vertices["z"] = transformed.T
    if {"nx", "ny", "nz"}.issubset(names):
        normals = np.column_stack((vertices["nx"], vertices["ny"], vertices["nz"])) @ rotation.T
        vertices["nx"], vertices["ny"], vertices["nz"] = normals.T
    ply.write(str(destination))


def align_structural_instances(instances_dir: Path, scene_mesh: Path | None = None) -> dict[str, Any]:
    instances_dir = Path(instances_dir)
    manifest_path = instances_dir / "structural_instances_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    legacy_directory = instances_dir / "vertically_aligned_door_window_instances"
    legacy_changed = legacy_directory.exists()
    if legacy_changed:
        shutil.rmtree(legacy_directory)
    for record in manifest.get("door_window_merges", []):
        legacy_changed |= record.pop("aligned_source_file", None) is not None
    prior = manifest.get("vertical_alignment")
    if isinstance(prior, dict) and prior.get("algorithm") == ALIGNMENT_ALGORITHM and prior.get("status") in {
        "applied", "already_vertical", "skipped_no_horizontal_support"
    }:
        legacy_changed |= prior.pop("aligned_door_window_instances", None) is not None
        if legacy_changed:
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return prior

    previous_rotation = np.eye(3)
    previous_translation = np.zeros(3)
    if isinstance(prior, dict) and prior.get("status") == "applied":
        previous_rotation = np.asarray(prior.get("rotation_matrix", np.eye(3)), dtype=np.float64)
        previous_translation = np.asarray(prior.get("translation_m", np.zeros(3)), dtype=np.float64)
        if previous_rotation.shape != (3, 3) or previous_translation.shape != (3,):
            raise ValueError("Prior vertical-alignment transform is malformed")

    def source_frame(vertices: np.ndarray) -> np.ndarray:
        return (vertices - previous_translation) @ previous_rotation

    meshes: list[tuple[dict[str, Any], Path, np.ndarray, np.ndarray, np.ndarray]] = []
    plane_areas, plane_normals = [], []
    wall_areas, wall_normals = [], []
    for item in manifest.get("instances", []):
        filename = item.get("file_name")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"Invalid structural instance filename: {filename!r}")
        path = instances_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Structural instance PLY is missing: {path}")
        mesh = o3d.io.read_triangle_mesh(str(path))
        vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        if not len(vertices) or not len(triangles):
            raise ValueError(f"Structural instance PLY is empty: {path}")
        raw_vertices = source_frame(vertices)
        meshes.append((item, path, vertices, triangles, raw_vertices))
        if item.get("role") not in {"floor", "ceiling", "wall"}:
            continue
        corners = raw_vertices[triangles]
        cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        double_area = np.linalg.norm(cross, axis=1)
        valid = double_area > 1e-10
        if valid.any():
            areas = double_area[valid] * 0.5
            normals = cross[valid] / double_area[valid, None]
            if item.get("role") == "wall":
                wall_areas.append(areas)
                wall_normals.append(normals)
            else:
                plane_areas.append(areas)
                plane_normals.append(normals)

    wall_estimate = None
    if wall_areas:
        wall_estimate = _estimate_wall_up(np.vstack(wall_normals), np.concatenate(wall_areas))
    if wall_estimate is not None:
        up, support = wall_estimate
        estimate_source = "unconstrained wall-plane normals"
    elif plane_areas:
        up, support = _estimate_up(np.vstack(plane_normals), np.concatenate(plane_areas))
        estimate_source = "floor/ceiling normal mode fallback"
    else:
        result = {"status": "skipped_no_horizontal_support",
                  "algorithm": ALIGNMENT_ALGORITHM,
                  "reason": "no non-degenerate wall, floor, or ceiling triangles were available"}
        (instances_dir / "vertically_aligned_scene_mesh.ply").unlink(missing_ok=True)
        manifest["vertical_alignment"] = result
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print("No wall or floor/ceiling support; leaving structural meshes in their current frame.")
        return result

    rotation = _rotation_to_up(up)
    angle = float(np.rad2deg(np.arccos(np.clip(up[1], -1.0, 1.0))))
    all_vertices = np.vstack([vertices for _, _, _, _, vertices in meshes])
    pivot = (all_vertices.min(axis=0) + all_vertices.max(axis=0)) * 0.5
    translation = pivot - rotation @ pivot
    delta_rotation = rotation @ previous_rotation.T
    delta_translation = translation - delta_rotation @ previous_translation

    status = "applied" if angle >= 1e-5 else "already_vertical"
    if angle < 1e-5 and np.allclose(delta_rotation, np.eye(3), atol=1e-8) \
            and np.allclose(delta_translation, 0.0, atol=1e-8):
        result = {"status": status, "algorithm": ALIGNMENT_ALGORITHM, "tilt_deg": angle,
                  "estimated_up": up.tolist(), "support_fraction": support,
                  "estimate_source": estimate_source,
                  "rotation_matrix": rotation.tolist(), "translation_m": translation.tolist()}
        (instances_dir / "vertically_aligned_scene_mesh.ply").unlink(missing_ok=True)
        manifest["vertical_alignment"] = result
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print("Structural instances are already vertically aligned; no rotation applied.")
        return result

    aligned_scene_mesh = instances_dir / "vertically_aligned_scene_mesh.ply"
    staged: list[tuple[Path, Path]] = []
    with tempfile.TemporaryDirectory(dir=instances_dir, prefix=".vertical_align_") as temp_name:
        temp_dir = Path(temp_name)
        for index, (_, path, _, _, _) in enumerate(meshes):
            temporary = temp_dir / f"instance_{index:04d}.ply"
            _transformed_ply(path, temporary, delta_rotation, delta_translation)
            staged.append((temporary, path))
        if scene_mesh is not None:
            if not Path(scene_mesh).is_file():
                raise FileNotFoundError(f"Source scene mesh is missing: {scene_mesh}")
            temporary = temp_dir / "scene_mesh.ply"
            _transformed_ply(Path(scene_mesh), temporary, rotation, translation)
            staged.append((temporary, aligned_scene_mesh))
        for temporary, destination in staged:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary.replace(destination)

    for item, _, _, _, vertices in meshes:
        transformed_y = (vertices @ rotation.T + translation)[:, 1]
        item["elevation"] = {
            "y_mean": round(float(transformed_y.mean()), 3),
            "y_min": round(float(transformed_y.min()), 3),
            "y_max": round(float(transformed_y.max()), 3),
        }
    result = {
        "status": status,
        "algorithm": ALIGNMENT_ALGORITHM,
        "tilt_deg": angle,
        "estimated_up": up.tolist(),
        "support_fraction": support,
        "estimate_source": estimate_source,
        "pivot_m": pivot.tolist(),
        "rotation_matrix": rotation.tolist(),
        "translation_m": translation.tolist(),
        "aligned_scene_mesh": str(aligned_scene_mesh) if scene_mesh is not None else None,
    }
    manifest["vertical_alignment"] = result
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Applied shared {angle:.3f}° vertical alignment from {estimate_source} to "
          f"{len(meshes)} structural instance PLYs.")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path, required=True)
    parser.add_argument("--scene-mesh", type=Path)
    args = parser.parse_args(argv)
    align_structural_instances(args.instances_dir, args.scene_mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
