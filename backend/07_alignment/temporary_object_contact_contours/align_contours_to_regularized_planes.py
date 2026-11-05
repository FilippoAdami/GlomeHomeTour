#!/usr/bin/env python3
"""Move object contours toward nearby axis-aligned regularized room planes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from detect_large_outside_surface_patches import closest_surface_scene, room_center


BACKEND = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE = BACKEND / "current_scene"
AXIS_NORMAL_COSINE = 0.98
MAX_TRANSLATION_M = 0.15
CONTACT_EPSILON_M = 0.001
BOUNDS_EPSILON_M = 0.001
RAY_ADVANCE_EPSILON_M = 0.0005
MAX_TRANSLATIONS_PER_OBJECT = 2
RED = np.array([255, 0, 0], dtype=np.uint8)


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"Expected a non-empty triangle mesh: {path}")
    if mesh.visual.kind != "vertex" or len(mesh.visual.vertex_colors) != len(mesh.vertices):
        raise ValueError(f"Expected vertex colors on mesh: {path}")
    return mesh


def regularized_source(scan_dir: Path) -> Path:
    pristine = scan_dir / ".raw_unaligned" / "mesh_elaboration" / "complete.ply"
    return pristine if pristine.is_file() else scan_dir / "mesh_elaboration" / "complete.ply"


def instance_relative_path(scan_name: str, path: str) -> Path:
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid contact-contour path: {path!r}")
    return Path(*relative.parts[1:]) if relative.parts and relative.parts[0] == scan_name else relative


def red_faces(mesh: trimesh.Trimesh) -> np.ndarray:
    colors = np.asarray(mesh.visual.vertex_colors)[:, :3]
    return np.all(np.all(colors[mesh.faces] == RED, axis=2), axis=1)


def axis_plane_scene(mesh: trimesh.Trimesh) -> o3d.t.geometry.RaycastingScene:
    return closest_surface_scene(mesh)[0]


def classify_outside_points(points: np.ndarray, scene: o3d.t.geometry.RaycastingScene,
                           midpoint: np.ndarray, face_axes: np.ndarray,
                           face_centers: np.ndarray, face_normals: np.ndarray):
    outside = np.zeros(len(points), dtype=bool)
    signed_distance = np.zeros(len(points), dtype=np.float32)
    axes = np.zeros(len(points), dtype=np.int8)
    outward_sign = np.zeros(len(points), dtype=np.int8)
    for start in range(0, len(points), 100_000):
        stop = min(start + 100_000, len(points))
        closest = scene.compute_closest_points(o3d.core.Tensor(points[start:stop].astype(np.float32)))
        nearest_points = closest["points"].numpy()
        nearest = closest["primitive_ids"].numpy().astype(np.int64)
        local_axes = face_axes[nearest]
        signs = np.sign(face_centers[nearest, local_axes] - midpoint[local_axes])
        ambiguous = signs == 0
        if np.any(ambiguous):
            signs[ambiguous] = np.sign(face_normals[nearest[ambiguous], local_axes[ambiguous]])
        direction = np.zeros_like(nearest_points)
        direction[np.arange(len(nearest_points)), local_axes] = signs
        distance = np.einsum("ij,ij->i", points[start:stop] - nearest_points, direction)
        signed_distance[start:stop] = distance
        axes[start:stop] = local_axes
        outward_sign[start:stop] = signs
        outside[start:stop] = distance > CONTACT_EPSILON_M
    return outside, signed_distance, axes, outward_sign


def contour_outside_samples(mesh: trimesh.Trimesh, contour_faces: np.ndarray,
                            scene: o3d.t.geometry.RaycastingScene,
                            midpoint: np.ndarray, face_axes: np.ndarray,
                            face_centers: np.ndarray, face_normals: np.ndarray):
    triangles = mesh.vertices[mesh.faces[contour_faces]]
    samples = np.concatenate((triangles, triangles.mean(axis=1, keepdims=True)), axis=1)
    flat = samples.reshape(-1, 3)
    outside, signed_distance, axes, outward_sign = classify_outside_points(
        flat, scene, midpoint, face_axes, face_centers, face_normals
    )
    return samples, outside.reshape(-1, 4), signed_distance.reshape(-1, 4), \
        axes.reshape(-1, 4), outward_sign.reshape(-1, 4)


def contour_outside_score(mesh: trimesh.Trimesh, contour_faces: np.ndarray,
                          outside_samples: np.ndarray,
                          signed_distances: np.ndarray) -> tuple[float, float, float, int]:
    face_outside = outside_samples.any(axis=1)
    areas = mesh.area_faces[contour_faces]
    return (float(areas[face_outside].sum()),
            float((areas[:, None] * outside_samples).sum() / outside_samples.shape[1]),
            float((areas[:, None] * np.maximum(signed_distances - CONTACT_EPSILON_M, 0.0)).sum()
                  / signed_distances.shape[1]),
            int(face_outside.sum()))


def minimum_inside_axis_step(signed_distance: np.ndarray, axes: np.ndarray,
                             outward_sign: np.ndarray, outside: np.ndarray) -> list[tuple[int, float]]:
    candidates = []
    for axis in range(3):
        if not np.any(outside & (axes == axis)):
            continue
        on_axis = axes == axis
        positive_side = on_axis & (outward_sign > 0)
        negative_side = on_axis & (outward_sign < 0)
        lower = float(np.max(signed_distance[negative_side] - CONTACT_EPSILON_M)) \
            if negative_side.any() else -np.inf
        upper = float(np.min(CONTACT_EPSILON_M - signed_distance[positive_side])) \
            if positive_side.any() else np.inf
        if lower > upper:
            continue
        delta = lower if lower > 0 else upper if upper < 0 else 0.0
        if abs(delta) <= CONTACT_EPSILON_M:
            continue
        candidates.append((axis, delta))
    return candidates


def fit_outside_contour_inside(mesh: trimesh.Trimesh, contour: np.ndarray,
                               scene: o3d.t.geometry.RaycastingScene,
                               midpoint: np.ndarray, face_axes: np.ndarray,
                               face_centers: np.ndarray, face_normals: np.ndarray) -> dict:
    contour_faces = np.flatnonzero(contour)
    if not len(contour_faces):
        return {"contour_triangle_count": 0, "outside_triangle_count_before": 0,
                "outside_triangle_count_after": 0, "translations": []}

    translations = []
    samples, outside, signed, axes, signs = contour_outside_samples(
        mesh, contour_faces, scene, midpoint, face_axes, face_centers, face_normals
    )
    initial_score = contour_outside_score(mesh, contour_faces, outside, signed)
    current_score = initial_score
    while current_score[3] > 0:
        flat_signed, flat_axes = signed.ravel(), axes.ravel()
        flat_signs, flat_outside = signs.ravel(), outside.ravel()
        candidates = minimum_inside_axis_step(flat_signed, flat_axes, flat_signs, flat_outside)
        evaluated = []
        for axis, amount in candidates:
            delta = np.zeros(3, dtype=np.float64)
            delta[axis] = amount
            moved_samples = samples + delta
            moved_outside, moved_signed, _, _ = classify_outside_points(
                moved_samples.reshape(-1, 3), scene, midpoint, face_axes, face_centers, face_normals
            )
            moved_outside = moved_outside.reshape(-1, 4)
            moved_signed = moved_signed.reshape(-1, 4)
            score = contour_outside_score(mesh, contour_faces, moved_outside, moved_signed)
            if score[:3] < current_score[:3]:
                evaluated.append((score, abs(amount), axis, amount, moved_outside))
        if not evaluated:
            break

        previous_score = current_score
        _, _, axis, amount, outside = min(
            evaluated, key=lambda item: (*item[0], item[1], item[2])
        )
        delta = np.zeros(3, dtype=np.float64)
        delta[axis] = amount
        if np.count_nonzero(delta) != 1 or not np.isfinite(amount):
            raise RuntimeError("Outside-contour pre-fit must be a finite single-axis move")
        mesh.vertices += delta
        samples, outside, signed, axes, signs = contour_outside_samples(
            mesh, contour_faces, scene, midpoint, face_axes, face_centers, face_normals
        )
        current_score = contour_outside_score(mesh, contour_faces, outside, signed)
        if not current_score[:3] < previous_score[:3]:
            raise RuntimeError("Outside-contour pre-fit failed to reduce the outside contour")
        translations.append({
            "axis": "XYZ"[axis],
            "translation_m": delta.tolist(),
            "outside_triangle_count_after": current_score[3],
            "outside_contour_area_after_m2": current_score[0],
        })

    return {
        "contour_triangle_count": int(len(contour_faces)),
        "outside_triangle_count_before": initial_score[3],
        "outside_contour_area_before_m2": initial_score[0],
        "outside_triangle_count_after": current_score[3],
        "outside_contour_area_after_m2": current_score[0],
        "unresolved_outside_contour_area_m2": current_score[0],
        "translations": translations,
    }


def translation_candidates(mesh: trimesh.Trimesh, contour: np.ndarray,
                           scene: o3d.t.geometry.RaycastingScene,
                           already_moved: set[int]) -> list[dict]:
    face_ids = np.flatnonzero(contour)
    face_ids = np.asarray([face for face in face_ids if face not in already_moved], dtype=np.int64)
    if not len(face_ids):
        return []

    triangles = mesh.vertices[mesh.faces[face_ids]]
    samples = np.concatenate((triangles, triangles.mean(axis=1, keepdims=True)), axis=1)
    flat_samples = samples.reshape(-1, 3).astype(np.float32)
    candidates: dict[tuple[int, int, int], dict] = {}
    for axis in range(3):
        for direction in (-1, 1):
            origins = flat_samples.copy()
            travelled = np.zeros(len(origins), dtype=np.float32)
            active = np.ones(len(origins), dtype=bool)
            while active.any():
                sample_ids = np.flatnonzero(active)
                rays = np.zeros((len(sample_ids), 6), dtype=np.float32)
                rays[:, :3] = origins[sample_ids]
                rays[:, 3 + axis] = direction
                distance = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
                finite = np.isfinite(distance)
                total_distance = travelled[sample_ids] + distance
                valid = finite & (total_distance > CONTACT_EPSILON_M) \
                    & (total_distance <= MAX_TRANSLATION_M)
                for ray_idx in np.flatnonzero(valid):
                    sample_idx = int(sample_ids[ray_idx])
                    face_idx = int(face_ids[sample_idx // 4])
                    delta = float(direction * total_distance[ray_idx])
                    coordinate = float(flat_samples[sample_idx, axis] + delta)
                    key = (face_idx, axis, round(coordinate * 1000))
                    candidate = {
                        "face_index": face_idx,
                        "axis": axis,
                        "delta_m": delta,
                        "distance_m": float(total_distance[ray_idx]),
                        "sample_index": int(sample_idx % 4),
                        "plane_coordinate_m": coordinate,
                    }
                    if key not in candidates or candidate["distance_m"] < candidates[key]["distance_m"]:
                        candidates[key] = candidate

                can_continue = finite & (total_distance <= MAX_TRANSLATION_M)
                if not can_continue.any():
                    break
                continuing = sample_ids[can_continue]
                advance = distance[can_continue] + RAY_ADVANCE_EPSILON_M
                origins[continuing, axis] += direction * advance
                travelled[continuing] += advance
                active[:] = False
                active[continuing] = True
    return sorted(candidates.values(), key=lambda item: (
        item["distance_m"], item["face_index"], item["axis"], item["delta_m"]))


def red_vertices_stay_in_bounds(mesh: trimesh.Trimesh, contour: np.ndarray,
                                delta: np.ndarray, bounds: np.ndarray) -> bool:
    face_vertices = np.unique(mesh.faces[contour])
    if not len(face_vertices):
        return True
    moved = mesh.vertices[face_vertices] + delta
    return bool(np.all(moved >= bounds[0] - BOUNDS_EPSILON_M)
                and np.all(moved <= bounds[1] + BOUNDS_EPSILON_M))


def align_one_object(mesh: trimesh.Trimesh, room_bounds: np.ndarray,
                     scene: o3d.t.geometry.RaycastingScene,
                     outside_context: tuple | None = None) -> dict:
    contour = red_faces(mesh)
    outside_prefit = {"contour_triangle_count": int(contour.sum()),
                      "outside_triangle_count_before": 0,
                      "outside_triangle_count_after": 0,
                      "translations": []}
    if outside_context is not None:
        outside_prefit = fit_outside_contour_inside(mesh, contour, scene, *outside_context)
    moved_faces: set[int] = set()
    translations = []
    rejected_by_bounds = 0

    for _ in range(MAX_TRANSLATIONS_PER_OBJECT):
        candidates = translation_candidates(mesh, contour, scene, moved_faces)
        accepted = None
        for candidate in candidates:
            delta = np.zeros(3, dtype=np.float64)
            delta[candidate["axis"]] = candidate["delta_m"]
            if red_vertices_stay_in_bounds(mesh, contour, delta, room_bounds):
                accepted = (candidate, delta)
                break
            rejected_by_bounds += 1
        if accepted is None:
            break

        candidate, delta = accepted
        face_points = mesh.vertices[mesh.faces[candidate["face_index"]]]
        contact_point = face_points[candidate["sample_index"]] \
            if candidate["sample_index"] < 3 else face_points.mean(axis=0)
        mesh.vertices += delta
        contact_residual = abs(float(contact_point[candidate["axis"]]
                                     + delta[candidate["axis"]]
                                     - candidate["plane_coordinate_m"]))
        if contact_residual > 1e-4:
            raise RuntimeError(f"Axis translation did not reach the selected plane: {contact_residual} m")
        moved_faces.add(candidate["face_index"])
        translations.append({
            "contour_face_index": candidate["face_index"],
            "contact_sample_index": candidate["sample_index"],
            "axis": "XYZ"[candidate["axis"]],
            "translation_m": delta.tolist(),
            "distance_to_plane_m": candidate["distance_m"],
            "plane_coordinate_m": candidate["plane_coordinate_m"],
            "contact_residual_m": contact_residual,
        })

    return {
        "red_triangle_count": int(contour.sum()),
        "outside_contour_prefit": outside_prefit,
        "translations": translations,
        "rejected_candidates_outside_room_bounds": rejected_by_bounds,
        "stopped_after_translation_limit": len(translations) == MAX_TRANSLATIONS_PER_OBJECT,
    }


def process(alignment_dir: Path, contour_dir: Path, output_dir: Path) -> dict:
    transforms_path = alignment_dir / "stats" / "aligned_scans_transforms.json"
    contours_path = contour_dir / "contact_contours.json"
    if not transforms_path.is_file() or not contours_path.is_file():
        raise FileNotFoundError(f"Expected {transforms_path} and {contours_path}")
    if output_dir.resolve() == contour_dir.resolve():
        raise ValueError("Output must be separate from source contact contours")

    transforms = json.loads(transforms_path.read_text())["scans"]
    contour_report = json.loads(contours_path.read_text())
    records, objects = [], []

    for scan_record in contour_report["scans"]:
        scan_name = scan_record["scan"]
        scan_dir = alignment_dir.parent / scan_name
        transform = np.asarray(transforms[scan_name]["transform_matrix"], dtype=np.float64)
        regularized_path = regularized_source(scan_dir)
        regularized = load_mesh(regularized_path)
        regularized.apply_transform(transform)
        room_bounds = regularized.bounds.copy()
        scene, face_axes, face_centers, face_normals = closest_surface_scene(regularized)
        outside_context = (room_center(regularized), face_axes, face_centers, face_normals)

        for instance in scan_record["instances"]:
            instance_path = Path(instance["path"])
            if instance_path.is_absolute() or ".." in instance_path.parts:
                raise ValueError(f"Invalid contact-contour path: {instance['path']!r}")
            source_path = contour_dir / instance_path
            mesh = load_mesh(source_path)
            alignment = align_one_object(mesh, room_bounds, scene, outside_context)
            target = output_dir / "objects" / scan_name / instance_relative_path(scan_name, instance["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(str(target))
            alignment.update({
                "scan": scan_name,
                "source_path": str(source_path.relative_to(contour_dir)),
                "output_path": str(target.relative_to(output_dir)),
            })
            records.append(alignment)
            objects.append(mesh)

    if not objects:
        raise ValueError("No contact-contour instances were found")
    combined_objects = trimesh.util.concatenate(objects)
    output_dir.mkdir(parents=True, exist_ok=True)
    objects_path = output_dir / "aligned_objects_contact_contours_translated.ply"
    combined_objects.export(str(objects_path))

    report = {
        "status": "axis_aligned_regularized_plane_contact",
        "plane_rule": "Cast each unprocessed red triangle's vertices and centroid along +/- X, Y, and Z; enumerate axis-aligned face hits within 15 cm and choose the nearest valid translation.",
        "regularized_reference": str(alignment_dir / "aligned_regularized_meshes.ply"),
        "regularized_geometry_exported_or_modified": False,
        "axis_face_normal_cosine_threshold": AXIS_NORMAL_COSINE,
        "maximum_translation_per_step_m": MAX_TRANSLATION_M,
        "maximum_translations_per_object": MAX_TRANSLATIONS_PER_OBJECT,
        "contact_epsilon_m": CONTACT_EPSILON_M,
        "outside_contour_prefit": {
            "rule": "Before plane snapping, classify only red contour triangle vertices and centroids against the nearest axis-aligned regularized face. Apply the smallest feasible single-axis translation that improves contour containment, repeating one axis at a time until inside or no improving shift is possible. There is no distance limit on this pre-fit; the following plane-snap pass retains its 15 cm limit.",
            "outside_epsilon_m": CONTACT_EPSILON_M,
        },
        "outside_check": "All vertices of every red triangle must remain inside the regularized mesh's axis-aligned bounds.",
        "combined_objects_ply": objects_path.name,
        "translated_object_count": sum(bool(item["translations"]) for item in records),
        "translation_count": sum(len(item["translations"]) for item in records),
        "objects": records,
    }
    (output_dir / "axis_plane_alignment_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--alignment-dir", type=Path)
    parser.add_argument("--contour-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    workspace = args.workspace.resolve()
    alignment_dir = (args.alignment_dir or workspace / "multi_room_alignment").resolve()
    contour_dir = (args.contour_dir or alignment_dir / "temporary_object_contact_contours").resolve()
    output_dir = (args.output_dir or contour_dir / "axis_plane_alignment").resolve()
    report = process(alignment_dir, contour_dir, output_dir)
    print(json.dumps({key: report[key] for key in (
        "status", "translated_object_count", "translation_count", "combined_objects_ply")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
