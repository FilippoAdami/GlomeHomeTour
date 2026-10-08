#!/usr/bin/env python3
"""Conservative, CPU-only refinement of an extracted triangle mesh."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

DEFAULT_WORKSPACE = Path(__file__).resolve().parents[1] / "current_scene"


def remove_floaters(mesh, min_fraction):
    labels, sizes, _ = mesh.cluster_connected_triangles()
    sizes = np.asarray(sizes)
    if not len(sizes):
        raise RuntimeError("Input mesh has no connected triangles")
    remove = sizes <= min_fraction * len(mesh.triangles)
    remove[np.argmax(sizes)] = False  # always retain the main geometry
    mesh.remove_triangles_by_mask(remove[np.asarray(labels)])
    mesh.remove_unreferenced_vertices()
    return {"components_before": len(sizes), "components_removed": int(remove.sum()),
            "triangles_removed": int(sizes[remove].sum())}


def rotation_to_up(normal):
    up = np.array([0.0, 1.0, 0.0])
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    if normal[1] < 0:
        normal = -normal
    axis = np.cross(normal, up)
    sine = np.linalg.norm(axis)
    if sine < 1e-12:
        return np.eye(3)
    axis /= sine
    angle = math.atan2(sine, np.dot(normal, up))
    import open3d as o3d
    return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)


def detect_patches(mesh, sample_count=200_000, min_points=60):
    import open3d as o3d
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    rng = np.random.default_rng(0)
    indices = rng.choice(len(vertices), min(sample_count, len(vertices)), replace=False)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices[indices]))
    cloud.normals = o3d.utility.Vector3dVector(normals[indices])
    return cloud.detect_planar_patches(
        normal_variance_threshold_deg=20, coplanarity_deg=70, outlier_ratio=0.65,
        min_num_points=min_points, search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30),
    )


def choose_floor(patches, bounds, max_tilt_deg=12):
    lower_limit = bounds[0][1] + 0.3 * (bounds[1][1] - bounds[0][1])
    candidates = [(float(p.extent[0] * p.extent[1]), p) for p in patches
                  if p.center[1] < lower_limit
                  and abs(p.R[1, 2]) >= math.cos(math.radians(max_tilt_deg))]
    if not candidates or max(area for area, _ in candidates) < 0.5:
        raise RuntimeError("No confident low, near-horizontal floor patch; mesh left unchanged")
    return max(candidates, key=lambda item: item[0])[1]


def target_normals(patches, rotation, angle_tol_deg=8):
    """Level horizontal patches; group near-orthogonal vertical patches modulo 90 degrees."""
    normals = np.array([rotation @ p.R[:, 2] for p in patches])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    targets = normals.copy()
    areas = np.array([p.extent[0] * p.extent[1] for p in patches])
    horizontal = np.abs(normals[:, 1]) >= math.cos(math.radians(angle_tol_deg))
    vertical = np.abs(normals[:, 1]) <= math.sin(math.radians(angle_tol_deg))
    targets[horizontal] = np.copysign(1.0, normals[horizontal, 1])[:, None] * np.array([0, 1, 0])
    targets[vertical, 1] = 0
    targets[vertical] /= np.linalg.norm(targets[vertical], axis=1, keepdims=True)

    vertical_ids = np.where(vertical)[0]
    angles = np.arctan2(targets[vertical_ids, 2], targets[vertical_ids, 0])
    period = math.pi / 2
    remaining = set(range(len(vertical_ids)))
    families = 0
    for seed in np.argsort(areas[vertical_ids])[::-1]:
        if seed not in remaining:
            continue
        distance = np.abs((angles - angles[seed] + period / 2) % period - period / 2)
        group = [i for i in remaining if distance[i] <= math.radians(angle_tol_deg)]
        remaining.difference_update(group)
        if len(group) < 2:
            continue
        families += 1
        mean_angle = np.angle(np.sum(areas[vertical_ids[group]] * np.exp(4j * angles[group]))) / 4
        for i in group:
            snapped_angle = mean_angle + round((angles[i] - mean_angle) / period) * period
            targets[vertical_ids[i]] = [math.cos(snapped_angle), 0, math.sin(snapped_angle)]
    return targets, families


def patch_candidates(vertices, normals, patch, rotation, translation, max_distance):
    center = rotation @ patch.center + translation
    basis = rotation @ patch.R
    corners = np.asarray(patch.get_box_points()) @ rotation.T + translation
    margin = max_distance
    nearby = np.flatnonzero(np.all((vertices >= corners.min(0) - margin)
                                   & (vertices <= corners.max(0) + margin), axis=1))
    if not len(nearby):
        return nearby, np.empty(0)
    local = (vertices[nearby] - center) @ basis
    keep = (np.abs(local[:, 0]) <= patch.extent[0] / 2 + margin) & \
           (np.abs(local[:, 1]) <= patch.extent[1] / 2 + margin) & \
           (np.abs(local[:, 2]) <= max_distance) & \
           (np.abs(normals[nearby] @ basis[:, 2]) >= 0.5)
    return nearby[keep], np.abs(local[keep, 2])


def regularize_patches(mesh, patches, rotation, translation, floor, max_distance=0.04,
                       max_shift=0.08, min_patch_area=0.05, snap_angle_deg=8):
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    patches = [p for p in patches if p.extent[0] * p.extent[1] >= min_patch_area]
    targets, families = target_normals(patches, rotation, snap_angle_deg)
    centers = np.array([rotation @ p.center + translation for p in patches])
    floor_index = next(i for i, p in enumerate(patches) if p is floor)
    centers[floor_index, 1] = 0
    first = np.full(len(vertices), -1, np.int16)
    first_score = np.full(len(vertices), np.inf, np.float32)

    for i, patch in enumerate(patches):
        ids, score = patch_candidates(vertices, normals, patch, rotation, translation, max_distance)
        if not len(ids):
            continue
        shift = np.abs((vertices[ids] - centers[i]) @ targets[i])
        use = (score < first_score[ids]) & (shift <= max_shift)
        first[ids[use]] = i
        first_score[ids[use]] = score[use]

    second = np.full(len(vertices), -1, np.int16)
    second_score = np.full(len(vertices), np.inf, np.float32)
    for i, patch in enumerate(patches):
        ids, score = patch_candidates(vertices, normals, patch, rotation, translation, max_distance)
        if not len(ids):
            continue
        assigned = first[ids]
        use = (assigned >= 0) & (assigned != i) & (score < second_score[ids])
        use &= np.abs(targets[np.maximum(assigned, 0)] @ targets[i]) < 1e-6
        use &= np.abs((vertices[ids] - centers[i]) @ targets[i]) <= max_shift
        second[ids[use]] = i
        second_score[ids[use]] = score[use]

    assigned = np.flatnonzero(first >= 0)
    n1, c1 = targets[first[assigned]], centers[first[assigned]]
    delta = -np.sum((vertices[assigned] - c1) * n1, axis=1)[:, None] * n1
    paired = second[assigned] >= 0
    if paired.any():
        ids = assigned[paired]
        n2, c2 = targets[second[ids]], centers[second[ids]]
        delta[paired] -= np.sum((vertices[ids] - c2) * n2, axis=1)[:, None] * n2
    safe = np.linalg.norm(delta, axis=1) <= max_shift
    vertices[assigned[safe]] += delta[safe]
    floor_ids, _ = patch_candidates(vertices, normals, floor, rotation, translation, max_distance=0.12)
    floor_ids = floor_ids[np.abs(normals[floor_ids, 1]) >= 0.75]
    floor_shift = np.abs(vertices[floor_ids, 1])
    vertices[floor_ids, 1] = 0.0
    return {"patches_detected": len(patches), "right_angle_families": families,
            "vertices_regularized": int(safe.sum()), "edge_vertices_snapped": int((paired & safe).sum()),
            "floor_vertices_flattened": len(floor_ids),
            "max_displacement_m": float(max(np.linalg.norm(delta[safe], axis=1).max(initial=0),
                                             floor_shift.max(initial=0)))}


def refine(mesh, min_component_fraction=0.005, snap_angle_deg=8, max_shift=0.08):
    import open3d as o3d
    if not len(mesh.triangles):
        raise RuntimeError("Input mesh has no triangles")
    summary = remove_floaters(mesh, min_component_fraction)
    mesh.compute_vertex_normals()
    patches = detect_patches(mesh)
    bounds = (mesh.get_min_bound(), mesh.get_max_bound())
    floor = choose_floor(patches, bounds)
    normal = floor.R[:, 2].copy()
    if normal[1] < 0:
        normal = -normal
    rotation = rotation_to_up(normal)
    translation = np.array([0.0, -(rotation @ floor.center)[1], 0.0])
    mesh.rotate(rotation, center=(0, 0, 0))
    mesh.translate(translation)
    summary.update(regularize_patches(mesh, patches, rotation, translation, floor,
                                      max_shift=max_shift, snap_angle_deg=snap_angle_deg))
    vertices = np.asarray(mesh.vertices)
    below_floor = vertices[:, 1] < 0
    summary["below_floor_vertices_clamped"] = int(below_floor.sum())
    vertices[below_floor, 1] = 0
    triangles = np.asarray(mesh.triangles)
    degenerate = np.zeros(len(triangles), dtype=bool)
    for start in range(0, len(triangles), 300_000):
        points = vertices[triangles[start:start + 300_000]]
        area = np.linalg.norm(np.cross(points[:, 1] - points[:, 0],
                                       points[:, 2] - points[:, 0]), axis=1) / 2
        degenerate[start:start + 300_000] = area < 1e-10
    summary["degenerate_triangles_removed"] = int(degenerate.sum())
    mesh.remove_triangles_by_mask(degenerate)
    mesh.remove_unreferenced_vertices()
    final_components = remove_floaters(mesh, min_component_fraction)
    summary["components_removed_after_cleanup"] = final_components["components_removed"]
    summary["triangles_removed_after_cleanup"] = final_components["triangles_removed"]
    mesh.compute_vertex_normals()
    summary.update(floor_normal_before=normal.tolist(), floor_height_before=float(floor.center[1]),
                   floor_height_after=0.0, rotation=rotation.tolist(), translation=translation.tolist(),
                   vertices=len(mesh.vertices), triangles=len(mesh.triangles),
                   min_component_fraction=min_component_fraction, snap_angle_deg=snap_angle_deg,
                   max_snap_shift_m=max_shift)
    return summary


def simplify_flat_surfaces(mesh, target_ratio=0.5):
    """Collapse zero-quadric-error edges, which favors coplanar triangles."""
    if target_ratio == 1:
        return mesh
    simplified = mesh.simplify_quadric_decimation(
        max(1, int(len(mesh.triangles) * target_ratio)),
        maximum_error=0.0, boundary_weight=10.0,
    )
    if not len(simplified.triangles):
        raise RuntimeError("Flat-surface simplification produced no triangles")
    simplified.remove_unreferenced_vertices()
    simplified.compute_vertex_normals()
    return simplified


def main(argv=None):
    import open3d as o3d
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--input", type=Path, help="Raw mesh PLY (default: workspace/04_3DGS_to_mesh/mesh.ply)")
    parser.add_argument("--output", type=Path, help="Refined PLY (default: input/refined/mesh.ply)")
    parser.add_argument("--min-component-fraction", type=float, default=0.005, metavar="FRACTION",
                        help="Keep disconnected components only if they exceed this fraction of all triangles (default: 0.005)")
    parser.add_argument("--flat-simplify-ratio", type=float, default=0.5, metavar="RATIO",
                        help="Target triangle fraction with zero-error quadric decimation (default: 0.5; 1 disables)")
    parser.add_argument("--snap-angle", type=float, default=8, metavar="DEG")
    parser.add_argument("--max-shift", type=float, default=0.08, metavar="M")
    args = parser.parse_args(argv)
    if not math.isfinite(args.min_component_fraction) or not 0 <= args.min_component_fraction < 1:
        parser.error("--min-component-fraction must be between 0 and 1")
    if not math.isfinite(args.flat_simplify_ratio) or not 0 < args.flat_simplify_ratio <= 1:
        parser.error("--flat-simplify-ratio must be between 0 and 1")
    if not math.isfinite(args.snap_angle) or not 0 < args.snap_angle < 45:
        parser.error("--snap-angle must be between 0 and 45 degrees")
    if not math.isfinite(args.max_shift) or args.max_shift <= 0:
        parser.error("--max-shift must be positive and finite")
    source = args.input.resolve() if args.input else args.workspace.resolve() / "04_3DGS_to_mesh" / "mesh.ply"
    output = args.output.resolve() if args.output else source.parent / "refined" / "mesh.ply"
    if source == output:
        parser.error("output must differ from input")
    mesh = o3d.io.read_triangle_mesh(str(source))
    if not len(mesh.triangles):
        raise RuntimeError(f"Cannot read nonempty triangle mesh from {source}")
    summary = refine(mesh, args.min_component_fraction, args.snap_angle, args.max_shift)
    summary["triangles_before_simplification"] = len(mesh.triangles)
    mesh = simplify_flat_surfaces(mesh, args.flat_simplify_ratio)
    summary["triangles_removed_by_simplification"] = summary["triangles_before_simplification"] - len(mesh.triangles)
    if args.flat_simplify_ratio < 1:
        cleanup = remove_floaters(mesh, args.min_component_fraction)
        summary["components_removed_after_simplification"] = cleanup["components_removed"]
        summary["triangles_removed_after_simplification"] = cleanup["triangles_removed"]
        mesh.compute_vertex_normals()
    summary["flat_simplify_ratio"] = args.flat_simplify_ratio
    summary["vertices"], summary["triangles"] = len(mesh.vertices), len(mesh.triangles)
    summary["source_mesh"] = str(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output), mesh):
        raise RuntimeError(f"Failed to write {output}")
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Refined mesh: {output} ({summary['triangles']} triangles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
