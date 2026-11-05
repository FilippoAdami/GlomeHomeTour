#!/usr/bin/env python3
"""Color filtered sealed-mesh object contact contours."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


ALIGNMENT_DIR = Path(__file__).resolve().parents[1]
CONTACT_DIR = ALIGNMENT_DIR / "temporary_object_contact_contours"
sys.path.insert(0, str(ALIGNMENT_DIR))
sys.path.insert(0, str(CONTACT_DIR))

from object_alignment_geometry import (  # noqa: E402
    MIN_ELONGATED_ASPECT_RATIO,
)
from align_contours_to_regularized_planes import CONTACT_EPSILON_M  # noqa: E402
from contour_outlier_geometry import (  # noqa: E402
    CLUSTER_DISTANCE_M,
    MAX_CLUSTER_SIZE,
    check_isolated_small_groups,
    isolated_small_groups,
)
from discover_contact_contours import (  # noqa: E402
    instance_face_indices,
    load_sealed_mesh_ownership,
    suppress_object_contacts,
    write_combined_instances,
    write_instance,
)


DEFAULT_ALIGNMENT_DIR = ALIGNMENT_DIR.parent / "current_scene" / "multi_room_alignment" / "temporary_alternative_pipeline"


def load_colored_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty() or not len(mesh.triangles) or not mesh.has_vertex_colors():
        raise ValueError(f"Expected a non-empty colored triangle mesh: {path}")
    return mesh


def source_instance_id(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Cannot read instance id from source mesh name: {path}") from exc


def triangle_areas(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    triangles = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
    return np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1) * 0.5


def triangle_outlier_masks(mesh: o3d.geometry.TriangleMesh) -> tuple[np.ndarray, np.ndarray]:
    """Find per-instance proportion and face-area outliers."""
    triangles = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
    edge_lengths = np.stack((
        np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
        np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
        np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
    ))
    longest = edge_lengths.max(axis=0)
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1) * 0.5
    valid = np.isfinite(areas) & (areas > 1e-12) & np.isfinite(longest)
    aspect = np.divide(longest ** 2, 2.0 * areas, out=np.full(len(areas), np.inf),
                       where=valid)
    shape_outlier = ~valid
    area_outlier = np.zeros(len(areas), dtype=bool)
    if valid.any():
        log_aspect = np.log(aspect[valid])
        median_aspect = np.median(log_aspect)
        mad_aspect = np.median(np.abs(log_aspect - median_aspect))
        aspect_cutoff = np.exp(median_aspect + max(np.log(4.0), 3.0 * 1.4826 * mad_aspect))
        shape_outlier[valid] = ((aspect[valid] >= MIN_ELONGATED_ASPECT_RATIO)
                                | (aspect[valid] >= aspect_cutoff))

        log_area = np.log(areas[valid])
        median_area = np.median(log_area)
        mad_area = np.median(np.abs(log_area - median_area))
        area_cutoff = median_area + max(np.log(8.0), 3.0 * 1.4826 * mad_area)
        area_outlier[valid] = log_area >= area_cutoff
    return shape_outlier, area_outlier


def remove_isolated_faces(mesh: o3d.geometry.TriangleMesh,
                          candidates: np.ndarray) -> tuple[np.ndarray, int]:
    """Drop candidate faces with no edge-adjacent candidate face."""
    if not candidates.any():
        return candidates.copy(), 0
    adjacency = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles), process=False
    ).face_adjacency
    connected = np.zeros(len(mesh.triangles), dtype=bool)
    if len(adjacency):
        red_pairs = candidates[adjacency].all(axis=1)
        connected[adjacency[red_pairs].ravel()] = True
    isolated = candidates & ~connected
    return candidates & connected, int(isolated.sum())


def contact_groups(prepared: list[dict]) -> tuple[list[str], list[dict]]:
    """Connect globally aligned meshes whose surfaces touch within the contour tolerance."""
    parent = list(range(len(prepared)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    bounds = []
    samples = []
    scenes = []
    for item in prepared:
        vertices = np.asarray(item["mesh"].vertices)
        faces = np.asarray(item["mesh"].triangles)
        triangles = vertices[faces]
        bounds.append((vertices.min(axis=0), vertices.max(axis=0)))
        # Shared borders have coincident vertices; face centers also catch overlapping
        # surfaces whose meshes use different border tessellation.
        # shortcut: sparse crossing patches without a sampled point may be missed; upgrade to triangle-pair distance if they appear.
        samples.append(np.concatenate((vertices, triangles.mean(axis=1)), axis=0)
                       .astype(np.float32))
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(item["mesh"]))
        scenes.append(scene)

    def touches(scene, points: np.ndarray) -> bool:
        for start in range(0, len(points), 100_000):
            query = o3d.core.Tensor(points[start:start + 100_000])
            closest = scene.compute_closest_points(query)["points"].numpy()
            if np.any(np.linalg.norm(query.numpy() - closest, axis=1)
                      <= CONTACT_EPSILON_M):
                return True
        return False

    contact_pair_count = 0
    for first in range(len(prepared)):
        low, high = bounds[first]
        for second in range(first + 1, len(prepared)):
            other_low, other_high = bounds[second]
            if not (np.all(low - CONTACT_EPSILON_M <= other_high)
                    and np.all(other_low <= high + CONTACT_EPSILON_M)):
                continue
            if (touches(scenes[second], samples[first])
                    or touches(scenes[first], samples[second])):
                union(first, second)
                contact_pair_count += 1

    members_by_root: dict[int, list[int]] = {}
    for index in range(len(prepared)):
        members_by_root.setdefault(find(index), []).append(index)
    ordered = sorted(members_by_root.values(), key=lambda group: group[0])
    group_ids = [""] * len(prepared)
    groups = []
    for group_number, members in enumerate(ordered):
        group_id = f"touching_group_{group_number:03d}"
        for index in members:
            group_ids[index] = group_id
        groups.append({
            "contact_group_id": group_id,
            "scan_ids": sorted({prepared[index]["scan"] for index in members}),
            "member_count": len(members),
            "member_source_instance_paths": [
                str(prepared[index]["source_path"]) for index in members
            ],
        })
    return group_ids, groups, contact_pair_count


def run(alignment_dir: Path, workspace: Path | None = None) -> dict:
    alignment_dir = alignment_dir.resolve()
    if workspace is None:
        workspace = next((candidate for candidate in
                          (alignment_dir.parent, alignment_dir.parent.parent)
                          if any(candidate.glob("scan_*"))), None)
        if workspace is None:
            raise FileNotFoundError(f"Cannot locate scan_* directories beside {alignment_dir}")
    workspace = workspace.resolve()
    check_isolated_small_groups()
    manifest_path = alignment_dir / "stats" / "aligned_objects_manifest.json"
    transforms_path = alignment_dir / "stats" / "aligned_scans_transforms.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing joint object manifest: {manifest_path}")
    if not transforms_path.is_file():
        raise FileNotFoundError(f"Missing scan transform manifest: {transforms_path}")
    manifest = json.loads(manifest_path.read_text())
    transforms = json.loads(transforms_path.read_text())["scans"]
    output_dir = alignment_dir / "outside_contact_contours"

    sealed_by_scan = {}
    prepared = []
    for item in manifest.get("objects", []):
        scan_name = item["scan_id"]
        scan_dir = workspace / scan_name
        if scan_name not in sealed_by_scan:
            sealed_by_scan[scan_name] = load_sealed_mesh_ownership(scan_dir)
        sealed, class_ids, instance_ids, boundary = sealed_by_scan[scan_name]

        source_path = Path(item["source_instance_path"])
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source instance mesh: {source_path}")
        global_path = Path(item["global_instance_path"])
        if not global_path.is_file():
            raise FileNotFoundError(f"Missing aligned instance mesh: {global_path}")
        face_indices, source_faces = instance_face_indices(
            source_path, sealed, class_ids, instance_ids,
            int(item["class_id"]), source_instance_id(source_path),
        )

        mesh = load_colored_mesh(global_path)
        if not np.array_equal(np.asarray(mesh.triangles), source_faces):
            raise ValueError(f"Aligned instance face order differs from its source: {global_path}")
        transform = np.asarray(transforms[scan_name]["transform_matrix"], dtype=np.float64)
        if not np.array_equal(transform, np.asarray(item["scan_transform_matrix"], dtype=np.float64)):
            raise ValueError(f"Instance transform differs from its scan transform: {global_path}")
        source_mesh = o3d.io.read_triangle_mesh(str(source_path))
        expected_vertices = (np.asarray(source_mesh.vertices) @ transform[:3, :3].T
                             + transform[:3, 3])
        if not np.allclose(np.asarray(mesh.vertices), expected_vertices, atol=1e-5):
            raise ValueError(f"Aligned instance geometry differs from transformed source: {global_path}")
        contour = boundary[face_indices].copy()
        if len(contour) != len(mesh.triangles):
            raise ValueError(f"Contour face count differs from aligned instance: {global_path}")
        prepared.append({
            "scan": scan_name,
            "mesh": mesh,
            "contour": contour,
            "sealed_boundary_count": int(contour.sum()),
            "source_path": source_path,
            "record": item,
        })

    if not prepared:
        raise ValueError("No aligned instances were found")

    group_ids, contact_group_report, contact_pair_count = contact_groups(prepared)
    for item, group_id in zip(prepared, group_ids):
        item["contact_group_id"] = group_id

    bounds = []
    for item in prepared:
        vertices = np.asarray(item["mesh"].vertices)
        bounds.append((vertices.min(axis=0), vertices.max(axis=0)))
    for index, item in enumerate(prepared):
        lower, upper = bounds[index]
        neighbors = [other["mesh"] for other_index, other in enumerate(prepared)
                     if other_index != index
                     and np.all(lower - CONTACT_EPSILON_M <= bounds[other_index][1])
                     and np.all(bounds[other_index][0] <= upper + CONTACT_EPSILON_M)]
        item["contour"], item["suppressed"] = suppress_object_contacts(
            item["mesh"], neighbors, item["contour"], CONTACT_EPSILON_M
        )

    output_meshes, records = [], []
    for item in prepared:
        scan_name = item["scan"]
        mesh = item["mesh"]
        contour_candidates = item["contour"]
        shape_outliers, oversized = triangle_outlier_masks(mesh)
        shape_removed = int((contour_candidates & shape_outliers).sum())
        after_shape_filter = contour_candidates & ~shape_outliers
        oversized_removed = int((after_shape_filter & oversized).sum())
        red_faces, isolated_removed = remove_isolated_faces(
            mesh, after_shape_filter & ~oversized
        )
        red_ids = np.flatnonzero(red_faces)
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        small_groups_removed, small_groups = isolated_small_groups(
            vertices[faces[red_ids]]
        )
        small_group_faces = red_ids[small_groups_removed]
        red_faces[small_group_faces] = False

        aligned_path = Path(item["record"]["global_instance_path"])
        target = (output_dir / "objects" / scan_name / aligned_path.parent.name
                  / f"{aligned_path.stem}_outside_contours.ply")
        write_instance(target, mesh, red_faces)
        output_meshes.append(target)
        areas = triangle_areas(mesh)
        records.append({
            "scan_id": scan_name,
            "contact_group_id": item["contact_group_id"],
            "source_instance_path": str(item["source_path"]),
            "output_path": str(target.relative_to(output_dir)),
            "sealed_boundary_triangle_count": item["sealed_boundary_count"],
            "cross_instance_contact_suppressed_triangle_count": int(item["suppressed"]),
            "contact_contour_triangle_count_before_filters": int(contour_candidates.sum()),
            "shape_outlier_contact_contour_triangles_removed": shape_removed,
            "oversized_contact_contour_triangles_removed": oversized_removed,
            "isolated_contact_contour_triangles_removed": isolated_removed,
            "small_isolated_group_triangles_removed": int(len(small_group_faces)),
            "small_isolated_group_count": len(small_groups),
            "small_isolated_group_sizes": [len(group) for group in small_groups],
            "contact_contour_triangle_count": int(red_faces.sum()),
            "contact_contour_surface_area_m2": float(areas[red_faces].sum()),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "aligned_objects_outside_contact_contours_red.ply"
    combined_triangles = write_combined_instances(output_meshes, combined_path)
    report = {
        "status": "filtered_sealed_mesh_contact_contours",
        "contour_rule": "A source object triangle is a contour when its corresponding sealed-mesh face shares an edge with a different class or instance; triangles touching another aligned object within 1 mm are suppressed.",
        "shape_outlier_filter": (
            f"Exclude degenerate faces, faces with aspect ratio (longest edge squared / "
            f"2 * area) at least {MIN_ELONGATED_ASPECT_RATIO:.0f}, or faces above a "
            "per-instance cutoff exp(median(log(aspect)) + max(log(4), "
            "3 * 1.4826 * MAD(log(aspect))))."
        ),
        "oversized_triangle_filter": (
            "Exclude faces above the per-instance area cutoff exp(median(log(area)) + "
            "max(log(8), 3 * 1.4826 * MAD(log(area))))."
        ),
        "isolated_triangle_filter": "After the shape and area filters, exclude red triangles with no edge-adjacent red triangle on the same object mesh.",
        "small_isolated_group_filter": (
            f"After all other contour filters, remove groups of one to {MAX_CLUSTER_SIZE} red triangles "
            f"only when every triangle pair is within {CLUSTER_DISTANCE_M * 100:.0f} cm surface distance "
            "and no other red triangle is within that distance. Evaluated per instance."
        ),
        "regularized_mesh_used_for_filtering": False,
        "combined_objects_ply": combined_path.name,
        "combined_triangle_count": combined_triangles,
        "contact_group_count": len(contact_group_report),
        "touching_contact_group_count": sum(
            group["member_count"] > 1 for group in contact_group_report
        ),
        "touching_instance_pair_count": contact_pair_count,
        "contact_group_members": contact_group_report,
        "contact_contour_triangle_count_before_filters": sum(
            x["contact_contour_triangle_count_before_filters"] for x in records
        ),
        "shape_outlier_contact_contour_triangles_removed": sum(
            x["shape_outlier_contact_contour_triangles_removed"] for x in records
        ),
        "oversized_contact_contour_triangles_removed": sum(
            x["oversized_contact_contour_triangles_removed"] for x in records
        ),
        "isolated_contact_contour_triangles_removed": sum(
            x["isolated_contact_contour_triangles_removed"] for x in records
        ),
        "small_isolated_group_triangles_removed": sum(
            x["small_isolated_group_triangles_removed"] for x in records
        ),
        "small_isolated_group_count": sum(x["small_isolated_group_count"] for x in records),
        "contact_contour_triangle_count": sum(x["contact_contour_triangle_count"] for x in records),
        "contact_contour_surface_area_m2": sum(x["contact_contour_surface_area_m2"] for x in records),
        "objects_with_red_triangles": sum(x["contact_contour_triangle_count"] > 0 for x in records),
        "objects": records,
    }
    (output_dir / "outside_contact_contours_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-dir", type=Path, default=DEFAULT_ALIGNMENT_DIR)
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args(argv)
    report = run(args.alignment_dir, args.workspace)
    print(json.dumps({
        "status": report["status"],
        "objects": len(report["objects"]),
        "objects_with_red_triangles": report["objects_with_red_triangles"],
        "shape_outlier_triangles_removed": report["shape_outlier_contact_contour_triangles_removed"],
        "oversized_triangles_removed": report["oversized_contact_contour_triangles_removed"],
        "isolated_triangles_removed": report["isolated_contact_contour_triangles_removed"],
        "small_isolated_group_triangles_removed": report[
            "small_isolated_group_triangles_removed"
        ],
        "red_triangles": report["contact_contour_triangle_count"],
        "combined_objects_ply": str(args.alignment_dir / "outside_contact_contours"
                                     / report["combined_objects_ply"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
