#!/usr/bin/env python3
"""Snap nearby non-structural object groups to plausible room supports."""

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from shapely import covers, points
from shapely.geometry import Polygon
from shapely.ops import unary_union

from structural_surfaces import load_role_meshes
from translate_instances_inside import triangle_samples
from object_alignment_geometry import fitting_mesh


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALIGNMENT = ROOT / "current_scene/scan_0/objects_alignment"
DEFAULT_OUTPUT = DEFAULT_ALIGNMENT / "structural_fit"
MAX_SUPPORT_GAP_M = 0.05
DISTANCE_EPSILON_M = 1e-4
CONTACT_GAP_M = 0.005
PENETRATION_TOLERANCE_M = 0.002
MIN_INSTANCE_CONTAINED_SURFACE_PERCENT = 98.0
MIN_CONTACT_AREA_M2 = 0.01
MAX_LOCAL_CORRECTION_M = 0.15
WALL_NORMAL_DOT_MIN = float(np.cos(np.deg2rad(30)))
WALL_CATEGORY_PRIORS = {"window", "radiator", "television"}
FLOOR_CATEGORY_PRIORS = {
    "door", "stairs", "ramp", "elevator", "column", "gate", "fountain", "gazebo",
    "table", "cabinet", "chair", "rug", "bookshelf", "sofa", "bed", "cooktop",
    "refrigerator", "oven", "dishwasher", "sink", "wc", "bidet", "shower",
    "washing machine", "car",
}


def as_scene(mesh):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def surface_meshes(shell, shell_path):
    role_surfaces = load_role_meshes(shell, shell_path)
    floor, ceiling, walls = (role_surfaces[name] for name in ("floor", "ceiling", "wall"))
    if not len(floor.triangles) or not len(ceiling.triangles) or not len(walls.triangles):
        raise RuntimeError("Stage 06 complete.ply must contain floor, ceiling, and wall surfaces")
    floor_faces = np.asarray(floor.vertices)[np.asarray(floor.triangles)]
    polygons = [Polygon(triangle[:, (0, 2)]) for triangle in floor_faces
                if np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])) > 1e-10]
    footprint = unary_union(polygons).buffer(0) if polygons else Polygon()
    if footprint.is_empty:
        raise RuntimeError("Could not derive a floor footprint from Stage 06 complete.ply")
    return floor, ceiling, walls, footprint


def vertical_hits(scene, xyz, direction, origin_y):
    rays = np.zeros((len(xyz), 6), dtype=np.float32)
    rays[:, :3] = xyz
    rays[:, 1] = origin_y
    rays[:, 4] = direction
    result = scene.cast_rays(o3d.core.Tensor(rays))
    hit_y = rays[:, 1] + direction * result["t_hit"].numpy()
    return hit_y, np.isfinite(result["t_hit"].numpy())


def weighted_quantile(values, weights, quantile):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order[min(np.searchsorted(cumulative, quantile * cumulative[-1]), len(order) - 1)]])


def floor_candidate(mesh, floor_scene):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    mesh.compute_triangle_normals()
    normals = np.asarray(mesh.triangle_normals)
    centers = triangles.mean(axis=1)
    horizontal = (np.abs(normals[:, 1]) >= 0.65) & (areas > 0)
    ids = np.flatnonzero(horizontal)
    if not len(ids):
        return None
    origin = centers[ids].copy()
    origin[:, 1] += MAX_SUPPORT_GAP_M + 0.01
    rays = np.zeros((len(ids), 6), dtype=np.float32)
    rays[:, :3] = origin
    rays[:, 4] = -1.0
    result = floor_scene.cast_rays(o3d.core.Tensor(rays))
    floor_y = origin[:, 1] - result["t_hit"].numpy()
    gap = centers[ids, 1] - floor_y
    valid = (np.isfinite(floor_y) & (gap >= -MAX_SUPPORT_GAP_M - DISTANCE_EPSILON_M)
             & (gap <= MAX_SUPPORT_GAP_M + DISTANCE_EPSILON_M))
    ids = ids[valid]
    gap = gap[valid]
    if not len(ids):
        return None
    weights = areas[ids]
    area = float(weights.sum())
    if area < MIN_CONTACT_AREA_M2:
        return None
    gap_before = weighted_quantile(gap, weights, 0.1)
    shift = CONTACT_GAP_M - gap_before
    if abs(shift) > MAX_LOCAL_CORRECTION_M:
        return None
    return {"translation_m": [0.0, shift, 0.0], "gap_before_m": gap_before,
            "gap_after_m": CONTACT_GAP_M, "support_area_m2": area}


def wall_candidate(mesh, wall_scene):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    mesh.compute_triangle_normals()
    normals = np.asarray(mesh.triangle_normals)
    centers = triangles.mean(axis=1)
    vertical = (np.abs(normals[:, 1]) < 0.65) & (areas > 0)
    ids = np.flatnonzero(vertical)
    if not len(ids):
        return None
    result = wall_scene.compute_closest_points(o3d.core.Tensor(centers[ids].astype(np.float32)))
    closest = result["points"].numpy()
    target_normals = result["primitive_normals"].numpy()
    source_normals = normals[ids]
    distances = np.linalg.norm(closest - centers[ids], axis=1)
    parallel = np.abs(np.einsum("ij,ij->i", source_normals, target_normals)) >= WALL_NORMAL_DOT_MIN
    valid = (np.isfinite(distances) & (distances <= MAX_SUPPORT_GAP_M + DISTANCE_EPSILON_M) & parallel)
    if not valid.any():
        return None

    ids, closest, target_normals = ids[valid], closest[valid], target_normals[valid]
    weights = areas[ids]
    best_area, best_same_plane, reference_normal = 0.0, None, None
    for i in range(len(ids)):
        same_plane = ((np.abs(target_normals @ target_normals[i]) >= 0.98)
                      & (np.abs((closest - closest[i]) @ target_normals[i]) <= 0.01))
        area = float(weights[same_plane].sum())
        if area > best_area:
            best_area, best_same_plane, reference_normal = area, same_plane, target_normals[i]
    area, same_plane = best_area, best_same_plane
    if area < MIN_CONTACT_AREA_M2:
        return None

    cluster_normals = target_normals[same_plane].copy()
    cluster_normals[np.einsum("ij,j->i", cluster_normals, reference_normal) < 0] *= -1
    normal = np.average(cluster_normals, axis=0, weights=weights[same_plane])
    normal[1] = 0.0
    length = np.linalg.norm(normal)
    if length < 1e-8:
        return None
    normal /= length
    point = np.average(closest[same_plane], axis=0, weights=weights[same_plane])
    if np.dot(vertices.mean(axis=0) - point, normal) < 0:
        normal *= -1
    nearest_gap = float(np.percentile((vertices - point) @ normal, 5))
    shift = CONTACT_GAP_M - nearest_gap
    if abs(shift) > MAX_LOCAL_CORRECTION_M:
        return None
    return {"translation_m": (normal * shift).tolist(), "gap_before_m": nearest_gap,
            "gap_after_m": CONTACT_GAP_M, "support_area_m2": area}


def unique_edges(faces):
    return np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                                             faces[:, [2, 0]])), axis=1), axis=0)


def edge_crossings(edges, vertices, target_scene):
    if not len(edges):
        return 0
    starts, ends = vertices[edges[:, 0]], vertices[edges[:, 1]]
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > 2 * PENETRATION_TOLERANCE_M
    if not valid.any():
        return 0
    starts, vectors, lengths = starts[valid], vectors[valid], lengths[valid]
    directions = vectors / lengths[:, None]
    rays = np.zeros((len(lengths), 6), dtype=np.float32)
    rays[:, :3] = starts + directions * PENETRATION_TOLERANCE_M
    rays[:, 3:] = directions
    result = target_scene.cast_rays(o3d.core.Tensor(rays))
    hit = result["t_hit"].numpy()
    hit_normals = result["primitive_normals"].numpy()
    transverse = np.abs(np.einsum("ij,ij->i", directions, hit_normals)) > 0.1
    return int((np.isfinite(hit) & (hit < lengths - 2 * PENETRATION_TOLERANCE_M) & transverse).sum())


def intersection_edge_hits(mesh, shell_scene, shell_vertices, shell_edges):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    if not len(vertices) or not len(faces):
        return 0
    object_scene = as_scene(mesh)
    forward = edge_crossings(unique_edges(faces), vertices, shell_scene)
    shell_start, shell_end = shell_vertices[shell_edges[:, 0]], shell_vertices[shell_edges[:, 1]]
    lower, upper = vertices.min(axis=0) - 0.01, vertices.max(axis=0) + 0.01
    nearby = (np.all(np.maximum(shell_start, shell_end) >= lower, axis=1)
              & np.all(np.minimum(shell_start, shell_end) <= upper, axis=1))
    reverse = edge_crossings(shell_edges[nearby], shell_vertices, object_scene)
    return forward + reverse


def apply_translation(mesh, translation):
    moved = o3d.geometry.TriangleMesh(mesh)
    moved.translate(np.asarray(translation, dtype=np.float64), relative=True)
    return moved


def contained_surface_percent(mesh, floor_scene, ceiling_scene, footprint, ray_top, ray_bottom,
                              translation=(0.0, 0.0, 0.0)):
    vertices = np.asarray(mesh.vertices) + np.asarray(translation, dtype=np.float64)
    faces = np.asarray(mesh.triangles)
    if not len(faces):
        return 100.0
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    samples = triangle_samples(triangles).reshape(-1, 3)
    weights = np.repeat(areas / 7.0, 7)
    floor_y, floor_ok = vertical_hits(floor_scene, samples, -1.0, samples[:, 1] + 0.001)
    ceiling_y, ceiling_ok = vertical_hits(ceiling_scene, samples, 1.0, samples[:, 1] - 0.001)
    in_plan = covers(footprint, points(samples[:, 0], samples[:, 2]))
    inside = (floor_ok & ceiling_ok & in_plan
              & (samples[:, 1] >= floor_y - PENETRATION_TOLERANCE_M)
              & (samples[:, 1] <= ceiling_y + PENETRATION_TOLERANCE_M))
    return float(100.0 * weights[inside].sum() / max(weights.sum(), 1e-12))


def support_weight(kind, categories):
    if kind == "wall" and categories & WALL_CATEGORY_PRIORS:
        return 1.25
    if kind == "floor" and categories & FLOOR_CATEGORY_PRIORS:
        return 1.25
    return 1.0


def containment_candidate_allowed(percentages, has_support, baseline_percentages=None):
    """Keep supported moves within 2% exterior, or improve a starting violation without regressions."""
    if not has_support or all(value >= MIN_INSTANCE_CONTAINED_SURFACE_PERCENT for value in percentages):
        return True
    if baseline_percentages is None:
        return False
    return (all(value >= baseline - 1e-6 for value, baseline in zip(percentages, baseline_percentages))
            and any(value > baseline + 1e-6 for value, baseline in zip(percentages, baseline_percentages)))


def publish_directory(staging, destination):
    backup = destination.with_name(destination.name + ".previous")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.rename(backup)
    try:
        staging.rename(destination)
    except Exception:
        if backup.exists():
            backup.rename(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def build(alignment_dir=DEFAULT_ALIGNMENT, output_dir=DEFAULT_OUTPUT):
    started = time.perf_counter()
    alignment_dir, output_dir = Path(alignment_dir), Path(output_dir)
    report_path = alignment_dir / "rigid_baseline_report.json"
    groups_path = alignment_dir / "contact_groups.json"
    if not report_path.is_file() or not groups_path.is_file():
        raise FileNotFoundError(f"Missing baseline alignment or contact groups in {alignment_dir}")
    alignment_report = json.loads(report_path.read_text())
    groups_manifest = json.loads(groups_path.read_text())
    shell_path = Path(alignment_report["inputs"]["regularized_shell"])
    shell = o3d.io.read_triangle_mesh(str(shell_path))
    if shell.is_empty():
        raise ValueError(f"Stage 06 shell is empty: {shell_path}")
    floor_mesh, ceiling_mesh, wall_mesh, footprint = surface_meshes(shell, shell_path)
    floor_scene, ceiling_scene, wall_scene = map(as_scene, (floor_mesh, ceiling_mesh, wall_mesh))
    ray_top, ray_bottom = float(shell.get_max_bound()[1] + 1.0), float(shell.get_min_bound()[1] - 1.0)
    shell_vertices, shell_faces = np.asarray(shell.vertices), np.asarray(shell.triangles)
    shell_edges = unique_edges(shell_faces)
    shell_scene = as_scene(shell)

    item_by_path = {item["path"]: item for item in alignment_report["object_instances"]}
    group_dir = output_dir / "contact_groups"
    group_dir.mkdir(parents=True, exist_ok=True)
    for stale in group_dir.glob("group_*.ply"):
        stale.unlink()
    reports, group_items, instance_records = [], [], []
    staging = Path(tempfile.mkdtemp(prefix=".aligned_instances.", dir=alignment_dir))
    try:
        for group in groups_manifest["groups"]:
            source = alignment_dir / group["path"]
            mesh = o3d.io.read_triangle_mesh(str(source))
            if mesh.is_empty() or not len(mesh.triangles):
                raise ValueError(f"Contact group mesh is empty: {source}")
            members = [item_by_path[path] for path in group["instances"]]
            member_meshes = []
            for item in members:
                relative = Path(item["path"])
                instance = o3d.io.read_triangle_mesh(str(alignment_dir / "rigid_baseline_instances" / relative))
                if instance.is_empty() or not len(instance.triangles):
                    raise ValueError(f"Instance mesh is empty or missing: {item['path']}")
                member_meshes.append(instance)
            fit_member_meshes = [fitting_mesh(instance) for instance in member_meshes]
            fit_group_mesh = o3d.geometry.TriangleMesh()
            for instance in fit_member_meshes:
                fit_group_mesh += instance
            categories = {str(item.get("category")) for item in members if item.get("category")}
            baseline_instance_containment = [contained_surface_percent(
                instance, floor_scene, ceiling_scene, footprint, ray_top, ray_bottom)
                for instance in fit_member_meshes]
            floor_fit = floor_candidate(fit_group_mesh, floor_scene)
            wall_fit = wall_candidate(fit_group_mesh, wall_scene)
            options = [(False, False)]
            if floor_fit:
                options.append((True, False))
            if wall_fit:
                options.append((False, True))
            if floor_fit and wall_fit:
                options.append((True, True))

            baseline_hits = intersection_edge_hits(fit_group_mesh, shell_scene, shell_vertices, shell_edges)
            candidates = []
            for use_floor, use_wall in options:
                shift = np.zeros(3, dtype=np.float64)
                support_score = 0.0
                if use_floor:
                    shift += np.asarray(floor_fit["translation_m"])
                    support_score += support_weight("floor", categories)
                if use_wall:
                    shift += np.asarray(wall_fit["translation_m"])
                    support_score += support_weight("wall", categories)
                moved = apply_translation(fit_group_mesh, shift)
                hits = intersection_edge_hits(moved, shell_scene, shell_vertices, shell_edges)
                label = "+".join(name for name, enabled in (("floor", use_floor), ("wall", use_wall)) if enabled) or "none"
                instance_containment = [contained_surface_percent(
                    instance, floor_scene, ceiling_scene, footprint, ray_top, ray_bottom, shift)
                    for instance in fit_member_meshes]
                has_support = use_floor or use_wall
                eligible = containment_candidate_allowed(
                    instance_containment, has_support, baseline_instance_containment
                )
                candidates.append(((hits, -support_score, float(np.linalg.norm(shift))),
                                   shift, moved, hits, use_floor, use_wall, label, support_score,
                                   instance_containment, eligible))
            eligible_candidates = [row for row in candidates if row[9]]
            if min(baseline_instance_containment, default=100.0) >= MIN_INSTANCE_CONTAINED_SURFACE_PERCENT:
                selected_candidate = min(
                    eligible_candidates,
                    key=lambda row: (row[3], -row[7], -sum(row[8]), float(np.linalg.norm(row[1]))),
                )
            else:
                selected_candidate = min(
                    eligible_candidates,
                    key=lambda row: (-sum(row[8]), row[3], -row[7], float(np.linalg.norm(row[1]))),
                )
            _, shift, fit_moved_group, hits_after, use_floor, use_wall, _, _, selected_containment, _ = selected_candidate
            moved_group = apply_translation(mesh, shift)

            group_file = f"group_{int(group['group_id']):03d}.ply"
            if not o3d.io.write_triangle_mesh(str(group_dir / group_file), moved_group, write_ascii=False):
                raise IOError(f"Could not write fitted group: {group_dir / group_file}")
            output_group = {**group, "path": f"contact_groups/{group_file}"}
            group_items.append(output_group)

            for member_index, item in enumerate(members):
                relative = Path(item["path"])
                instance = member_meshes[member_index]
                instance.translate(shift, relative=True)
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not o3d.io.write_triangle_mesh(str(target), instance, write_ascii=False):
                    raise IOError(f"Could not write aligned instance: {target}")
                instance_records.append({"class_id": item["class_id"], "role": "object",
                                         "category": item.get("category"), "source": item["source"],
                                         "path": str(relative), "triangles": len(instance.triangles),
                                         "group_id": int(group["group_id"]),
                                         "placement_translation_m": shift.tolist(),
                                         "baseline_contained_surface_percent": baseline_instance_containment[member_index],
                                         "contained_surface_percent": selected_containment[member_index]})

            floor_gap_after = (CONTACT_GAP_M if use_floor else
                               floor_fit["gap_before_m"] if floor_fit else None)
            wall_gap_after = (CONTACT_GAP_M if use_wall else
                              wall_fit["gap_before_m"] if wall_fit else None)
            before_contained = contained_surface_percent(fit_group_mesh, floor_scene, ceiling_scene,
                                                          footprint, ray_top, ray_bottom)
            after_contained = contained_surface_percent(fit_moved_group, floor_scene, ceiling_scene,
                                                         footprint, ray_top, ray_bottom)
            support_candidates = [row for row in candidates if row[4] or row[5]]
            rejected_by_containment = bool(support_candidates) and not any(row[9] for row in support_candidates)
            status = ("unresolved_intersections" if hits_after else
                      "support_adjusted" if np.linalg.norm(shift) > 1e-6 else
                      "support_candidate_rejected_by_containment" if rejected_by_containment
                      else "support_candidate_rejected_by_intersections" if floor_fit or wall_fit
                      else "no_near_support_candidate")
            reports.append({"group_id": int(group["group_id"]), "instance_count": len(members),
                            "categories": sorted(categories), "translation_m": shift.tolist(),
                            "displacement_rms_m": float(np.linalg.norm(shift)),
                            "support_adjusted": bool(np.linalg.norm(shift) > 1e-6),
                            "floor_support_candidate": floor_fit,
                            "wall_support_candidate": wall_fit,
                            "floor_support_applied": use_floor, "wall_support_applied": use_wall,
                            "floor_gap_after_m": floor_gap_after, "wall_gap_after_m": wall_gap_after,
                            "surface_intersection_edge_hits_before": baseline_hits,
                            "surface_intersection_edge_hits_after": hits_after,
                            "baseline_instance_containment_percent": baseline_instance_containment,
                            "instance_containment_percent": selected_containment,
                            "minimum_instance_contained_surface_percent": min(selected_containment, default=100.0),
                            "candidate_evaluations": [{"candidate": candidate[6],
                                                        "translation_m": candidate[1].tolist(),
                                                        "surface_intersection_edge_hits": candidate[3],
                                                        "support_score": candidate[7],
                                                        "instance_containment_percent": candidate[8],
                                                        "within_containment_tolerance": candidate[9],
                                                        "selected": candidate is selected_candidate}
                                                       for candidate in candidates],
                            "baseline_surface_percent": before_contained,
                            "contained_surface_percent": after_contained,
                            "status": status})

        staged_manifest = {"status": "aligned_non_structural_instances",
                           "source_shell": str(shell_path),
                           "global_transform": alignment_report["transform"],
                           "support_gap_threshold_m": MAX_SUPPORT_GAP_M,
                           "target_contact_gap_m": CONTACT_GAP_M,
                           "minimum_instance_contained_surface_percent": MIN_INSTANCE_CONTAINED_SURFACE_PERCENT,
                           "containment_measure": "area-weighted samples from each instance's non-elongated faces",
                           "instances": instance_records}
        (staging / "manifest.json").write_text(json.dumps(staged_manifest, indent=2) + "\n")
        publish_directory(staging, alignment_dir / "aligned_instances")
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    manifest = {**groups_manifest, "status": "support_fitted_contact_groups",
                "source_shell": str(shell_path), "groups": group_items}
    (output_dir / "contact_groups.json").write_text(json.dumps(manifest, indent=2) + "\n")
    combined = o3d.geometry.TriangleMesh(shell)
    for group in group_items:
        combined += o3d.io.read_triangle_mesh(str(output_dir / group["path"]))
    if not o3d.io.write_triangle_mesh(str(output_dir / "structural_fit_scene.ply"), combined, write_ascii=False):
        raise IOError(f"Could not write fitted scene: {output_dir / 'structural_fit_scene.ply'}")

    report = {"status": "support_contact_fit", "method":
              "fit contact groups to nearby floor and wall surfaces using non-elongated object faces for support, intersection, and containment scores; preserve all source faces in aligned outputs",
              "support_gap_threshold_m": MAX_SUPPORT_GAP_M, "target_contact_gap_m": CONTACT_GAP_M,
              "penetration_tolerance_m": PENETRATION_TOLERANCE_M,
              "internal_structural_surfaces_included": True,
              "group_count": len(reports),
              "transformed_groups": sum(x["support_adjusted"] for x in reports),
              "unresolved_intersection_groups": sum(x["surface_intersection_edge_hits_after"] > 0
                                                      for x in reports),
              "minimum_instance_contained_surface_percent": MIN_INSTANCE_CONTAINED_SURFACE_PERCENT,
              "instances_outside_containment_tolerance": sum(
                  item["contained_surface_percent"] < MIN_INSTANCE_CONTAINED_SURFACE_PERCENT
                  for item in instance_records),
              "partial_groups": sum(x["minimum_instance_contained_surface_percent"] < MIN_INSTANCE_CONTAINED_SURFACE_PERCENT
                                    for x in reports),
              "mean_baseline_surface_percent": float(np.mean([x["baseline_surface_percent"] for x in reports])) if reports else 100.0,
              "mean_contained_surface_percent": float(np.mean([x["contained_surface_percent"] for x in reports])) if reports else 100.0,
              "groups": reports, "final_instances_dir": str(alignment_dir / "aligned_instances"),
              "runtime_seconds": time.perf_counter() - started}
    (output_dir / "structural_fit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-dir", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(alignment_dir=args.alignment_dir, output_dir=args.output_dir)
    print(json.dumps({"groups": result["group_count"], "transformed": result["transformed_groups"],
                      "unresolved_intersection_groups": result["unresolved_intersection_groups"],
                      "mean_contained_surface_percent": result["mean_contained_surface_percent"],
                      "final_instances": result["final_instances_dir"]}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
