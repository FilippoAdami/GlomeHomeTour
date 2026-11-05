#!/usr/bin/env python3
"""Translate aligned instances until their filtered contour points fit the room shell."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.optimize import minimize
from scipy.spatial import cKDTree


ALIGNMENT_DIR = Path(__file__).resolve().parents[1]
CONTACT_DIR = ALIGNMENT_DIR / "temporary_object_contact_contours"
sys.path.insert(0, str(ALIGNMENT_DIR))
sys.path.insert(0, str(CONTACT_DIR))

from align_contours_to_regularized_planes import CONTACT_EPSILON_M  # noqa: E402
from detect_large_outside_surface_patches import closest_surface_scene, room_center  # noqa: E402
from discover_contact_contours import write_combined_instances, write_instance  # noqa: E402


DEFAULT_ALIGNMENT_DIR = (
    ALIGNMENT_DIR.parent / "current_scene" / "multi_room_alignment"
    / "temporary_alternative_pipeline"
)
RED_THRESHOLD = np.array([0.99, 0.01, 0.01])
INSIDE_MARGIN_M = 0.00002
MAX_PAIRED_SURFACE_GAP_M = 0.30
MAX_PAIRED_SURFACE_PROJECTION_GAP_M = 0.15
MIN_PAIRED_SURFACE_GAP_M = 0.02


def load_colored_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty() or not len(mesh.triangles) or not mesh.has_vertex_colors():
        raise ValueError(f"Expected a non-empty colored triangle mesh: {path}")
    return mesh


def red_face_mask(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    colors = np.asarray(mesh.vertex_colors)[np.asarray(mesh.triangles)]
    return (np.all(colors[:, :, 0] >= RED_THRESHOLD[0], axis=1)
            & np.all(colors[:, :, 1] <= RED_THRESHOLD[1], axis=1)
            & np.all(colors[:, :, 2] <= RED_THRESHOLD[2], axis=1))


def surface_outward_signs(midpoint: np.ndarray, face_axes: np.ndarray,
                          face_centers: np.ndarray,
                          face_normals: np.ndarray) -> tuple[np.ndarray, int]:
    """Use local inward-facing normals for close, opposing room-side surfaces."""
    signs = np.sign(face_centers[np.arange(len(face_axes)), face_axes] - midpoint[face_axes])
    ambiguous = signs == 0
    if np.any(ambiguous):
        signs[ambiguous] = np.sign(
            face_normals[np.flatnonzero(ambiguous), face_axes[ambiguous]]
        )
    signs = signs.astype(np.int8)
    paired_count = 0
    for axis in range(3):
        axis_faces = np.flatnonzero(face_axes == axis)
        projection_axes = [other for other in range(3) if other != axis]
        for normal_sign in (-1, 1):
            faces = axis_faces[np.sign(face_normals[axis_faces, axis]) == normal_sign]
            opposite = axis_faces[np.sign(face_normals[axis_faces, axis]) == -normal_sign]
            if not len(faces) or not len(opposite):
                continue
            projection_distance, opposite_ids = cKDTree(
                face_centers[opposite][:, projection_axes]
            ).query(face_centers[faces][:, projection_axes], k=1)
            plane_gap = np.abs(
                face_centers[faces, axis] - face_centers[opposite[opposite_ids], axis]
            )
            paired = (
                (projection_distance <= MAX_PAIRED_SURFACE_PROJECTION_GAP_M)
                & (plane_gap >= MIN_PAIRED_SURFACE_GAP_M)
                & (plane_gap <= MAX_PAIRED_SURFACE_GAP_M)
            )
            signs[faces[paired]] = -normal_sign
            paired_count += int(paired.sum())
    return signs, paired_count


def classify_shell_points(points: np.ndarray, scene,
                          face_axes: np.ndarray,
                          face_outward_signs: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                                   np.ndarray, np.ndarray]:
    outside = np.zeros(len(points), dtype=bool)
    signed_distance = np.zeros(len(points), dtype=np.float32)
    axes = np.zeros(len(points), dtype=np.int8)
    signs = np.zeros(len(points), dtype=np.int8)
    for start in range(0, len(points), 100_000):
        stop = min(start + 100_000, len(points))
        closest = scene.compute_closest_points(
            o3d.core.Tensor(points[start:stop].astype(np.float32))
        )
        nearest_points = closest["points"].numpy()
        nearest = closest["primitive_ids"].numpy().astype(np.int64)
        local_axes = face_axes[nearest]
        local_signs = face_outward_signs[nearest]
        direction = np.zeros_like(nearest_points)
        direction[np.arange(stop - start), local_axes] = local_signs
        distance = np.einsum(
            "ij,ij->i", points[start:stop] - nearest_points, direction
        )
        signed_distance[start:stop] = distance
        axes[start:stop] = local_axes
        signs[start:stop] = local_signs
        outside[start:stop] = distance > CONTACT_EPSILON_M
    return outside, signed_distance, axes, signs


def minimum_contour_translation(samples, scene=None, axes: np.ndarray | None = None,
                                outward_signs: np.ndarray | None = None,
                                room_bounds: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Find the shortest feasible translation using the piecewise shell classifier."""
    target = CONTACT_EPSILON_M - INSIDE_MARGIN_M
    if isinstance(samples, list):
        shell_constraints = [(
            member["samples"], member["scene"], member["axes"],
            member["outward_signs"], member["room_bounds"],
        ) for member in samples if len(member["samples"])]
    else:
        shell_constraints = [(samples, scene, axes, outward_signs, room_bounds)]
    if not shell_constraints:
        return np.zeros(3), {"feasible": True, "method": "minimum_norm_slsqp",
                             "starts_tried": 0, "solver_success": True}
    joint_samples = np.concatenate([entry[0] for entry in shell_constraints], axis=0)

    def classify(delta: np.ndarray):
        parts = [classify_shell_points(
            entry[0] + delta, entry[1], entry[2], entry[3]
        ) for entry in shell_constraints]
        return tuple(np.concatenate([part[index] for part in parts], axis=0)
                     for index in range(4))

    initial_outside, _, _, _ = classify(np.zeros(3, dtype=np.float64))
    if not initial_outside.any():
        return np.zeros(3), {"feasible": True, "method": "minimum_norm_slsqp",
                             "starts_tried": 0, "solver_success": True}

    lower = np.max(np.stack([
        bounds[0] - member_samples.min(axis=0)
        for member_samples, _, _, _, bounds in shell_constraints
    ]), axis=0)
    upper = np.min(np.stack([
        bounds[1] - member_samples.max(axis=0)
        for member_samples, _, _, _, bounds in shell_constraints
    ]), axis=0)
    if np.any(lower > upper):
        return np.zeros(3), {"feasible": False, "method": "minimum_norm_slsqp",
                             "reason": "contour_exceeds_room_bounds", "starts_tried": 0,
                             "solver_success": False}
    bounds = list(zip(lower, upper))

    def slsqp_constraints(delta: np.ndarray) -> np.ndarray:
        return target - classify(delta)[1]

    def constraint_jacobian(delta: np.ndarray) -> np.ndarray:
        _, _, point_axes, signs = classify(delta)
        jacobian = np.zeros((len(joint_samples), 3), dtype=np.float64)
        jacobian[np.arange(len(joint_samples)), point_axes] = -signs
        return jacobian

    def solve(start: np.ndarray):
        result = minimize(
            lambda delta: 0.5 * float(delta @ delta),
            np.clip(start, lower, upper),
            jac=lambda delta: delta,
            constraints=[{"type": "ineq", "fun": slsqp_constraints,
                          "jac": constraint_jacobian}],
            method="SLSQP",
            bounds=bounds,
            options={"ftol": 1e-10, "maxiter": 100},
        )
        delta = np.asarray(result.x, dtype=np.float64)
        feasible = bool(np.all(slsqp_constraints(delta) >= -1e-6))
        return delta, feasible, bool(result.success)

    candidates = []
    first, first_feasible, first_success = solve(np.zeros(3))
    if first_feasible:
        candidates.append((first, first_success))
    # Nearest-face assignments change at surface boundaries, so large or failed
    # origin solutions get deterministic restarts before choosing a translation.
    starts_tried = 1
    if not first_feasible or np.linalg.norm(first) > 0.5:
        for x in (-0.5, 0.0, 0.5):
            for y in (-0.5, 0.0, 0.5):
                for z in (-0.5, 0.0, 0.5):
                    start = np.array([x, y, z], dtype=np.float64)
                    if np.allclose(start, 0.0):
                        continue
                    candidate, feasible, success = solve(start)
                    if feasible:
                        candidates.append((candidate, success))
                    starts_tried += 1

    if not candidates:
        return np.zeros(3), {"feasible": False, "method": "minimum_norm_slsqp",
                             "reason": "no_feasible_translation_found",
                             "starts_tried": starts_tried,
                             "solver_success": first_success}
    best, success = min(
        candidates, key=lambda item: (float(item[0] @ item[0]), tuple(item[0]))
    )
    return best, {"feasible": True, "method": "minimum_norm_slsqp",
                  "starts_tried": starts_tried, "solver_success": success}


def run(alignment_dir: Path) -> dict:
    alignment_dir = alignment_dir.resolve()
    manifest_path = alignment_dir / "stats" / "aligned_objects_manifest.json"
    contour_dir = alignment_dir / "outside_contact_contours"
    contour_report_path = contour_dir / "outside_contact_contours_report.json"
    if not manifest_path.is_file() or not contour_report_path.is_file():
        raise FileNotFoundError(f"Run joint alignment and contour filtering first under {alignment_dir}")

    manifest = json.loads(manifest_path.read_text())
    contour_report = json.loads(contour_report_path.read_text())
    objects_by_source = {
        Path(item["source_instance_path"]).resolve(): item
        for item in manifest.get("objects", [])
    }
    output_dir = alignment_dir / "contour_inside_fit"
    regularized_by_scan = {}
    output_meshes, records = [], []
    paired_surface_faces_by_scan = {}

    for contour_record in contour_report["objects"]:
        scan_name = contour_record["scan_id"]
        source_path = Path(contour_record["source_instance_path"]).resolve()
        aligned_record = objects_by_source.get(source_path)
        if aligned_record is None:
            raise ValueError(f"Filtered contour has no aligned source instance: {source_path}")
        aligned_path = Path(aligned_record["global_instance_path"])
        aligned_mesh = load_colored_mesh(aligned_path)
        source_faces = np.asarray(aligned_mesh.triangles)

        filtered_relative = Path(contour_record["output_path"])
        if filtered_relative.is_absolute() or ".." in filtered_relative.parts:
            raise ValueError(f"Invalid filtered contour path: {contour_record['output_path']!r}")
        filtered_path = contour_dir / filtered_relative
        filtered_mesh = load_colored_mesh(filtered_path)
        if len(filtered_mesh.triangles) != len(source_faces):
            raise ValueError(f"Filtered contour face count differs from aligned object: {filtered_path}")
        filtered_triangles = np.asarray(filtered_mesh.vertices)[np.asarray(filtered_mesh.triangles)]
        source_triangles = np.asarray(aligned_mesh.vertices)[source_faces]
        if not np.allclose(filtered_triangles, source_triangles, atol=1e-6):
            raise ValueError(f"Filtered contour face order/geometry differs from aligned object: {filtered_path}")
        contour = red_face_mask(filtered_mesh)
        if int(contour.sum()) != contour_record["contact_contour_triangle_count"]:
            raise ValueError(f"Filtered contour colors do not match report: {filtered_path}")

        if scan_name not in regularized_by_scan:
            regularized_path = (alignment_dir / "scans" / scan_name
                                / "mesh_elaboration" / "complete.ply")
            if not regularized_path.is_file():
                raise FileNotFoundError(f"Missing aligned regularized mesh: {regularized_path}")
            regularized = trimesh.load(regularized_path, process=False)
            if not isinstance(regularized, trimesh.Trimesh) or not len(regularized.faces):
                raise ValueError(f"Expected a non-empty aligned regularized mesh: {regularized_path}")
            scene, axes, centers, normals = closest_surface_scene(regularized)
            midpoint = room_center(regularized)
            outward_signs, paired_count = surface_outward_signs(
                midpoint, axes, centers, normals
            )
            paired_surface_faces_by_scan[scan_name] = paired_count
            regularized_by_scan[scan_name] = (
                regularized_path, scene, axes, outward_signs, regularized.bounds.copy()
            )
        regularized_path, scene, axes, outward_signs, room_bounds = \
            regularized_by_scan[scan_name]

        samples = np.empty((0, 3), dtype=np.float64)
        if contour.any():
            contour_faces = np.flatnonzero(contour)
            triangles = np.asarray(aligned_mesh.vertices)[source_faces[contour_faces]]
            samples = np.concatenate((triangles, triangles.mean(axis=1, keepdims=True)), axis=1)
            samples = samples.reshape(-1, 3)
        contour_record["_fit"] = {
            "scan_name": scan_name,
            "contact_group_id": contour_record.get(
                "contact_group_id", f"{scan_name}:{source_path}"
            ),
            "source_path": source_path,
            "aligned_path": aligned_path,
            "regularized_path": regularized_path,
            "aligned_mesh": aligned_mesh,
            "source_faces": source_faces,
            "contour": contour,
            "samples": samples,
            "scene": scene,
            "axes": axes,
            "outward_signs": outward_signs,
            "room_bounds": room_bounds,
        }

    grouped: dict[str, list[dict]] = {}
    for contour_record in contour_report["objects"]:
        fit = contour_record.pop("_fit")
        grouped.setdefault(fit["contact_group_id"], []).append(fit)

    for group_id, members in grouped.items():
        nonempty_samples = [member["samples"] for member in members if len(member["samples"])]
        group_samples = (np.concatenate(nonempty_samples, axis=0) if nonempty_samples
                         else np.empty((0, 3), dtype=np.float64))
        translation, solver = minimum_contour_translation(
            members,
        ) if len(group_samples) else (
            np.zeros(3), {"feasible": True, "starts_tried": 0, "solver_success": True}
        )

        for member in members:
            aligned_mesh = member["aligned_mesh"]
            aligned_path = member["aligned_path"]
            source_path = member["source_path"]
            contour = member["contour"]
            samples = member["samples"]
            source_faces = member["source_faces"]
            original_vertices = np.asarray(aligned_mesh.vertices).copy()
            moved_vertices = original_vertices + translation
            outside_before = outside_after = 0
            outside_triangle_count_before = outside_triangle_count_after = 0
            if len(samples):
                initial_outside, _, _, _ = classify_shell_points(
                    samples, member["scene"], member["axes"], member["outward_signs"]
                )
                outside_before = int(initial_outside.sum())
                outside_triangle_count_before = int(
                    initial_outside.reshape(-1, 4).any(axis=1).sum()
                )
                final_outside, _, _, _ = classify_shell_points(
                    samples + translation, member["scene"], member["axes"],
                    member["outward_signs"],
                )
                outside_after = int(final_outside.sum())
                outside_triangle_count_after = int(
                    final_outside.reshape(-1, 4).any(axis=1).sum()
                )
                if solver["feasible"] and outside_after:
                    raise RuntimeError(
                        f"Shared minimum translation failed contour containment for {aligned_path}: "
                        f"{outside_after} samples remain outside"
                    )

            if not np.allclose(moved_vertices - original_vertices, translation, atol=1e-9):
                raise RuntimeError(f"Contour fitting applied a non-rigid transform to {aligned_path}")
            aligned_mesh.vertices = o3d.utility.Vector3dVector(moved_vertices)
            translation_steps = [
                {"axis": "XYZ"[axis], "translation_m": np.eye(3)[axis] * translation[axis]}
                for axis in range(3) if abs(float(translation[axis])) > 1e-7
            ]
            target = (output_dir / "objects" / member["scan_name"]
                      / aligned_path.parent.name / f"{aligned_path.stem}_inside_shell.ply")
            write_instance(target, aligned_mesh, contour)
            output_meshes.append(target)
            records.append({
                "scan_id": member["scan_name"],
                "contact_group_id": group_id,
                "contact_group_size": len(members),
                "source_instance_path": str(source_path),
                "aligned_instance_path": str(aligned_path),
                "regularized_mesh_path": str(member["regularized_path"]),
                "output_path": str(target.relative_to(output_dir)),
                "contour_triangle_count": int(contour.sum()),
                "outside_triangle_count_before": outside_triangle_count_before,
                "outside_triangle_count_after": outside_triangle_count_after,
                "outside_contour_point_count_before": outside_before,
                "outside_contour_point_count_after": outside_after,
                "translation_m": translation.tolist(),
                "translation_magnitude_m": float(np.linalg.norm(translation)),
                "translation_steps": [
                    {"axis": step["axis"],
                     "translation_m": np.asarray(step["translation_m"]).tolist()}
                    for step in translation_steps
                ],
                "translation_solver": solver,
            })

    if not output_meshes:
        raise ValueError("No filtered object contours were found")

    combined_path = output_dir / "aligned_objects_contours_inside_regularized_mesh.ply"
    combined_triangles = write_combined_instances(output_meshes, combined_path)
    largest_translation = max(records, key=lambda item: item["translation_magnitude_m"])
    report = {
        "status": "filtered_contours_minimum_translation_fit",
        "translation_rule": "Find the minimum-norm translation that keeps every red contour vertex and face centroid inside the regularized mesh within 1 mm. The search is bounded only by the shell's axis-aligned bounds, without an arbitrary distance cap. Close, overlapping, opposing plane pairs use their local face normals to identify each room-facing side. Each nonzero coordinate is applied as a separate rigid axis-only move. Non-contour triangles do not constrain movement. If no feasible translation is found, leave that object unmoved and report it.",
        "rotation_applied": False,
        "regularized_mesh_modified": False,
        "combined_objects_ply": combined_path.name,
        "combined_triangle_count": combined_triangles,
        "object_count": len(records),
        "contact_group_count": len(grouped),
        "touching_contact_group_count": sum(
            len(members) > 1 for members in grouped.values()
        ),
        "paired_internal_surface_face_count_by_scan": paired_surface_faces_by_scan,
        "objects_moved": sum(bool(item["translation_steps"]) for item in records),
        "maximum_translation_m": largest_translation["translation_magnitude_m"],
        "largest_translation_instance": {
            "scan_id": largest_translation["scan_id"],
            "source_instance_path": largest_translation["source_instance_path"],
            "translation_m": largest_translation["translation_m"],
        },
        "objects_with_unresolved_outside_contours": sum(
            item["outside_triangle_count_after"] > 0 for item in records
        ),
        "contour_triangle_count": sum(item["contour_triangle_count"] for item in records),
        "outside_triangle_count_before": sum(item["outside_triangle_count_before"] for item in records),
        "outside_triangle_count_after": sum(item["outside_triangle_count_after"] for item in records),
        "outside_contour_point_count_before": sum(
            item["outside_contour_point_count_before"] for item in records
        ),
        "outside_contour_point_count_after": sum(
            item["outside_contour_point_count_after"] for item in records
        ),
        "objects_without_feasible_translation": sum(
            not item["translation_solver"]["feasible"] for item in records
        ),
        "objects": records,
    }
    (output_dir / "contour_inside_fit_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-dir", type=Path, default=DEFAULT_ALIGNMENT_DIR)
    args = parser.parse_args(argv)
    report = run(args.alignment_dir)
    print(json.dumps({
        "status": report["status"],
        "objects": report["object_count"],
        "objects_moved": report["objects_moved"],
        "maximum_translation_m": report["maximum_translation_m"],
        "objects_with_unresolved_outside_contours": report[
            "objects_with_unresolved_outside_contours"
        ],
        "outside_triangles_before": report["outside_triangle_count_before"],
        "outside_triangles_after": report["outside_triangle_count_after"],
        "outside_contour_points_after": report["outside_contour_point_count_after"],
        "objects_without_feasible_translation": report["objects_without_feasible_translation"],
        "combined_objects_ply": str(args.alignment_dir / "contour_inside_fit"
                                     / report["combined_objects_ply"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
