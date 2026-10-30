#!/usr/bin/env python3
"""Translate each contact-connected object block into the regularized shell."""

import json
from itertools import product
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from shapely import covers, points
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union
from translate_instances_inside import triangle_samples


import argparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALIGNMENT = ROOT / "current_scene/06_mesh_elaboration/new_step/outputs/segmentation_alignment"
DEFAULT_OUTPUT = DEFAULT_ALIGNMENT / "structural_fit"
CLEARANCE_M = 0.01
MAX_TRANSLATION_M = 0.05
MAX_YAW_DEG = 5.0


def cast_envelope_bounds(xyz, envelope):
    floor_rays = np.zeros((len(xyz), 6), dtype=np.float32)
    floor_rays[:, 0], floor_rays[:, 2] = xyz[:, 0], xyz[:, 2]
    floor_rays[:, 1], floor_rays[:, 4] = envelope["ray_top"], -1.0
    ceiling_rays = np.zeros_like(floor_rays)
    ceiling_rays[:, 0], ceiling_rays[:, 2] = xyz[:, 0], xyz[:, 2]
    ceiling_rays[:, 1], ceiling_rays[:, 4] = envelope["ray_bottom"], 1.0
    floor = envelope["floor_scene"].cast_rays(o3d.core.Tensor(floor_rays))
    ceiling = envelope["ceiling_scene"].cast_rays(o3d.core.Tensor(ceiling_rays))
    floor_y = envelope["ray_top"] - floor["t_hit"].numpy()
    ceiling_y = envelope["ray_bottom"] + ceiling["t_hit"].numpy()
    floor_ok = np.isfinite(floor_y) & (floor["primitive_normals"].numpy()[:, 1] >= 0.65)
    ceiling_ok = np.isfinite(ceiling_y) & (ceiling["primitive_normals"].numpy()[:, 1] <= -0.65)
    return floor_y, ceiling_y, floor_ok, ceiling_ok


def slab_vertical_bounds(xz, slab, ray_top, ray_bottom):
    down = np.zeros((len(xz), 6), dtype=np.float32)
    down[:, 0], down[:, 2] = xz[:, 0], xz[:, 1]
    down[:, 1], down[:, 4] = ray_top, -1.0
    up = np.zeros_like(down)
    up[:, 0], up[:, 2] = xz[:, 0], xz[:, 1]
    up[:, 1], up[:, 4] = ray_bottom, 1.0
    top_hit = slab["scene"].cast_rays(o3d.core.Tensor(down))
    bottom_hit = slab["scene"].cast_rays(o3d.core.Tensor(up))
    top_y = ray_top - top_hit["t_hit"].numpy()
    bottom_y = ray_bottom + bottom_hit["t_hit"].numpy()
    valid = np.isfinite(top_y) & np.isfinite(bottom_y)
    return bottom_y, top_y, valid


def bound_parameters(params):
    result = np.asarray(params, dtype=float).copy()
    result[:3] = np.clip(result[:3], -MAX_TRANSLATION_M, MAX_TRANSLATION_M)
    result[3] = np.clip(result[3], -MAX_YAW_DEG, MAX_YAW_DEG)
    return result


def yaw_matrix(degrees):
    angle = np.deg2rad(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray([[cosine, 0.0, sine],
                       [0.0, 1.0, 0.0],
                       [-sine, 0.0, cosine]])


def required_significant_gain(baseline_percent):
    return 3.0 + 0.08 * max(0.0, 70.0 - baseline_percent)


def fit_group(mesh, envelope, footprint, slabs, clearance):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    center = vertices.mean(axis=0)
    triangles = vertices[faces]
    face_areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                         triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    # Area-stratified probes keep the bounded search fast while representing large surfaces fairly.
    count = min(1500, len(faces))
    cumulative = np.cumsum(face_areas)
    selected = np.searchsorted(cumulative, (np.arange(count) + 0.5) * cumulative[-1] / count)
    search_faces = faces[selected]
    search_triangles = vertices[search_faces]
    search_points = triangle_samples(search_triangles).reshape(-1, 3)
    search_weights = np.repeat(face_areas[selected] / 7.0, 7)

    def evaluate(params, probes=search_points, weights=search_weights):
        rotation = yaw_matrix(params[3])
        moved = (probes - center) @ rotation.T + center + params[:3]
        floor_y, roof_y, floor_ok, roof_ok = cast_envelope_bounds(moved, envelope)
        inside = (floor_ok & roof_ok
                  & (moved[:, 1] >= floor_y + clearance)
                  & (moved[:, 1] <= roof_y - clearance)
                  & covers(footprint, points(moved[:, 0], moved[:, 2])))
        for slab in slabs:
            bottom, top, hit = slab_vertical_bounds(moved[:, (0, 2)], slab,
                                                    envelope["ray_top"], envelope["ray_bottom"])
            inside &= ~(hit & (moved[:, 1] >= bottom - clearance)
                        & (moved[:, 1] <= top + clearance))
        return float(weights[inside].sum() / weights.sum())

    def better(candidate, score, best, best_score):
        return score > best_score + 1e-10 or (
            abs(score - best_score) <= 1e-10
            and np.linalg.norm(candidate / np.asarray([MAX_TRANSLATION_M] * 3 + [MAX_YAW_DEG]))
            < np.linalg.norm(best / np.asarray([MAX_TRANSLATION_M] * 3 + [MAX_YAW_DEG])) - 1e-10)

    full_samples = triangle_samples(triangles).reshape(-1, 3)
    full_weights = np.repeat(face_areas / 7.0, 7)
    baseline_percent = 100.0 * evaluate(np.zeros(4), full_samples, full_weights)
    best = np.zeros(4)
    if baseline_percent > 1e-12:
        best_score = evaluate(best)
        for offset in product((-MAX_TRANSLATION_M, 0.0, MAX_TRANSLATION_M), repeat=3):
            candidate = np.asarray([*offset, 0.0])
            score = evaluate(candidate)
            if better(candidate, score, best, best_score):
                best, best_score = candidate, score
        for yaw in (-MAX_YAW_DEG, MAX_YAW_DEG):
            candidate = best.copy()
            candidate[3] = yaw
            score = evaluate(candidate)
            if better(candidate, score, best, best_score):
                best, best_score = candidate, score

        steps = np.asarray([0.025, 0.025, 0.025, 2.5])
        while np.max(steps[:3]) > 0.001 or steps[3] > 0.1:
            improved = False
            for axis in range(4):
                for sign in (-1.0, 1.0):
                    candidate = best.copy()
                    candidate[axis] += sign * steps[axis]
                    candidate = bound_parameters(candidate)
                    score = evaluate(candidate)
                    if better(candidate, score, best, best_score):
                        best, best_score = candidate, score
                        improved = True
            if not improved:
                steps *= 0.5

    optimized_percent = 100.0 * evaluate(best, full_samples, full_weights)
    required_gain = required_significant_gain(baseline_percent)
    if baseline_percent <= 1e-12:
        best = np.zeros(4)
        optimized_percent = baseline_percent
        status = "unchanged_zero_baseline"
    elif optimized_percent - baseline_percent + 1e-9 < required_gain:
        best = np.zeros(4)
        optimized_percent = baseline_percent
        status = "unchanged_insignificant_gain"
    else:
        status = "transformed" if np.linalg.norm(best) > 1e-8 else "already_inside"

    rotation = yaw_matrix(best[3])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = center - rotation @ center + best[:3]
    contained_percent = optimized_percent
    displacement = (vertices - center) @ rotation.T + center + best[:3] - vertices
    result = {"transform": transform, "translation_m": best[:3].tolist(),
              "rotation_matrix": rotation.tolist(), "rotation_degrees": abs(float(best[3])),
              "yaw_degrees": float(best[3]),
              "displacement_rms_m": float(np.sqrt(np.mean(np.sum(displacement * displacement, axis=1)))),
              "baseline_surface_percent": baseline_percent,
              "contained_surface_percent": contained_percent,
              "gain_percentage_points": contained_percent - baseline_percent,
              "required_gain_percentage_points": required_gain}
    return result, status


def build(alignment_dir=DEFAULT_ALIGNMENT, output_dir=DEFAULT_OUTPUT):
    started = time.perf_counter()
    check = bound_parameters([0.2, -0.2, 0.1, 9.0])
    assert np.allclose(np.abs(check[:3]), MAX_TRANSLATION_M)
    assert check[3] == MAX_YAW_DEG
    check_rotation = yaw_matrix(check[3])
    assert np.allclose(check_rotation @ [0.0, 1.0, 0.0], [0.0, 1.0, 0.0])
    assert np.allclose(check_rotation.T @ check_rotation, np.eye(3), atol=1e-10)
    assert required_significant_gain(80.0) == 3.0
    assert required_significant_gain(43.0) > 2.0
    assert required_significant_gain(10.0) > 5.0
    assert required_significant_gain(0.3) > 0.2
    alignment_dir = Path(alignment_dir)
    output_dir = Path(output_dir)
    report_file = alignment_dir / "rigid_baseline_report.json"
    groups_file = alignment_dir / "contact_groups.json"
    if not report_file.exists() or not groups_file.exists():
        raise FileNotFoundError(f"Missing required alignment inputs in {alignment_dir}")
    alignment_report = json.loads(report_file.read_text())
    groups_manifest = json.loads(groups_file.read_text())
    shell_path = Path(alignment_report["inputs"]["regularized_shell"])
    base = shell_path.parent
    level_manifest_path = base / "structural_levels_manifest.json"
    level_manifest = json.loads(level_manifest_path.read_text()) if level_manifest_path.exists() else {}
    pattern_manifest_path = base / "plane_patterns_manifest.json"
    shell = o3d.io.read_triangle_mesh(str(shell_path))
    ray_top = float(shell.get_max_bound()[1] + 1.0)
    ray_bottom = float(shell.get_min_bound()[1] - 1.0)
    def artifact_path(unit):
        path = Path(unit["regularized_artifact"])
        return path if path.is_absolute() else ROOT.parent / path

    if pattern_manifest_path.exists():
        structural_units = json.loads(pattern_manifest_path.read_text())["units"]
        floors = [item for item in structural_units if item.get("structural_role") == "floor"]
        ceilings = [item for item in structural_units if item.get("structural_role") == "ceiling"]
        if not floors or not ceilings:
            raise RuntimeError(f"No outer floor or ceiling found in {base / 'plane_patterns_manifest.json'}")
        outer_floor = min(floors, key=lambda item: item["mean_elevation_y"])
        outer_ceiling = max(ceilings, key=lambda item: item["mean_elevation_y"])
        floor_mesh = o3d.io.read_triangle_mesh(str(artifact_path(outer_floor)))
        ceiling_mesh = o3d.io.read_triangle_mesh(str(artifact_path(outer_ceiling)))
    else:
        # Fallback: extract floor and ceiling surfaces directly from the regularized shell
        shell.compute_triangle_normals()
        normals = np.asarray(shell.triangle_normals)
        triangles = np.asarray(shell.triangles)
        vertices = np.asarray(shell.vertices)
        f_mask = normals[:, 1] >= 0.65
        c_mask = normals[:, 1] <= -0.65
        floor_mesh = o3d.geometry.TriangleMesh()
        if f_mask.any():
            used, inv = np.unique(triangles[f_mask], return_inverse=True)
            floor_mesh.vertices = o3d.utility.Vector3dVector(vertices[used])
            floor_mesh.triangles = o3d.utility.Vector3iVector(inv.reshape(-1, 3))
        ceiling_mesh = o3d.geometry.TriangleMesh()
        if c_mask.any():
            used, inv = np.unique(triangles[c_mask], return_inverse=True)
            ceiling_mesh.vertices = o3d.utility.Vector3dVector(vertices[used])
            ceiling_mesh.triangles = o3d.utility.Vector3iVector(inv.reshape(-1, 3))
        mean_fy = float(np.mean(vertices[triangles[f_mask]][:, 1])) if f_mask.any() else 0.0
        mean_cy = float(np.mean(vertices[triangles[c_mask]][:, 1])) if c_mask.any() else 3.0
        outer_floor = {"unit_name": "shell_floor", "mean_elevation_y": mean_fy, "footprint_xz": None}
        outer_ceiling = {"unit_name": "shell_ceiling", "mean_elevation_y": mean_cy, "footprint_xz": None}

    ceiling_mesh.compute_triangle_normals()
    ceiling_faces = np.asarray(ceiling_mesh.vertices)[np.asarray(ceiling_mesh.triangles)]
    support_faces = ceiling_faces[np.asarray(ceiling_mesh.triangle_normals)[:, 1] <= -0.65]
    support = unary_union([Polygon(triangle[:, (0, 2)]) for triangle in support_faces]).buffer(0)
    if outer_floor.get("footprint_xz"):
        footprint = shape(outer_floor["footprint_xz"]).buffer(0).intersection(support).buffer(-CLEARANCE_M)
    else:
        floor_mesh.compute_triangle_normals()
        f_faces = np.asarray(floor_mesh.vertices)[np.asarray(floor_mesh.triangles)]
        f_supp = unary_union([Polygon(tri[:, (0, 2)]) for tri in f_faces]).buffer(0)
        footprint = f_supp.intersection(support).buffer(-CLEARANCE_M)
    if footprint.is_empty:
        raise RuntimeError("Outer floor footprint does not overlap the outer roof support")
    floor_scene = o3d.t.geometry.RaycastingScene()
    floor_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(floor_mesh))
    ceiling_scene = o3d.t.geometry.RaycastingScene()
    ceiling_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(ceiling_mesh))
    envelope = {"floor_scene": floor_scene, "ceiling_scene": ceiling_scene,
                "ray_top": ray_top, "ray_bottom": ray_bottom}
    slabs = []
    for slab_item in level_manifest.get("slabs", []):
        slab_path = Path(slab_item["file_path"])
        slab_path = slab_path if slab_path.is_absolute() else ROOT.parent / slab_path
        slab_mesh = o3d.io.read_triangle_mesh(str(slab_path))
        if not slab_mesh.is_watertight():
            raise RuntimeError(f"Internal slab must be watertight: {slab_path}")
        slab_scene = o3d.t.geometry.RaycastingScene()
        slab_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(slab_mesh))
        slabs.append({"scene": slab_scene})
    assert outer_floor["mean_elevation_y"] < outer_ceiling["mean_elevation_y"]

    group_dir = output_dir / "contact_groups"
    instance_dir = output_dir / "instances"
    group_dir.mkdir(parents=True, exist_ok=True)
    instance_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    group_items = []
    for group in groups_manifest["groups"]:
        source = alignment_dir / group["path"]
        mesh = o3d.io.read_triangle_mesh(str(source))
        if not len(mesh.triangles):
            raise ValueError(f"Contact group mesh is empty: {source}")
        fit, status = fit_group(mesh, envelope, footprint, slabs, clearance=CLEARANCE_M)
        transform = fit["transform"]
        moved_group = o3d.geometry.TriangleMesh(mesh)
        moved_group.transform(transform)
        output_group = f"group_{int(group['group_id']):03d}.ply"
        if not o3d.io.write_triangle_mesh(str(group_dir / output_group), moved_group, write_ascii=False):
            raise IOError(f"Could not write fitted group: {group_dir / output_group}")
        group_items.append({**group, "path": f"contact_groups/{output_group}"})

        for source_instance in group["instances"]:
            original_path = alignment_dir / "rigid_baseline_instances" / source_instance
            instance = o3d.io.read_triangle_mesh(str(original_path))
            if not len(instance.triangles):
                raise ValueError(f"Instance mesh is empty or missing: {original_path}")
            target = instance_dir / source_instance
            target.parent.mkdir(parents=True, exist_ok=True)
            instance.transform(transform)
            if not o3d.io.write_triangle_mesh(str(target), instance, write_ascii=False):
                raise IOError(f"Could not write fitted instance: {target}")

        reports.append({"group_id": int(group["group_id"]), "instance_count": len(group["instances"]),
                        "translation_m": fit["translation_m"], "rotation_matrix": fit["rotation_matrix"],
                        "rotation_degrees": fit["rotation_degrees"], "yaw_degrees": fit["yaw_degrees"],
                        "displacement_rms_m": fit["displacement_rms_m"],
                        "baseline_surface_percent": fit["baseline_surface_percent"],
                        "contained_surface_percent": fit["contained_surface_percent"],
                        "gain_percentage_points": fit["gain_percentage_points"],
                        "required_gain_percentage_points": fit["required_gain_percentage_points"],
                        "status": status})

    manifest = {"status": "structural_fit_candidate", "source_groups": str(alignment_dir / "contact_groups.json"),
                "source_shell": str(shell_path), "contact_distance_m": groups_manifest["contact_distance_m"],
                "groups": group_items}
    (output_dir / "contact_groups.json").write_text(json.dumps(manifest, indent=2) + "\n")
    combined = o3d.geometry.TriangleMesh(shell)
    for group in group_items:
        combined += o3d.io.read_triangle_mesh(str(output_dir / group["path"]))
    if not o3d.io.write_triangle_mesh(str(output_dir / "structural_fit_scene.ply"), combined, write_ascii=False):
        raise IOError(f"Could not write fitted scene: {output_dir / 'structural_fit_scene.ply'}")

    report = {"status": "bounded_rigid_block_fit_candidate",
              "method": "deterministic coarse-grid and coordinate search maximizing sampled surface area inside the outer floor-roof envelope using bounded translation and vertical-axis yaw only; internal watertight slabs are excluded",
              "outer_floor": outer_floor["unit_name"], "outer_ceiling": outer_ceiling["unit_name"],
              "internal_slab_count": len(slabs),
              "max_translation_per_axis_m": MAX_TRANSLATION_M,
              "rotation_axis": "vertical Y only", "max_rotation_degrees": MAX_YAW_DEG,
              "significance_rule": "apply only when sampled containment gain meets 3 percentage points + 0.08 times (70 - baseline percentage points), floored at 3; zero-baseline blocks remain unchanged",
              "clearance_m": CLEARANCE_M, "group_count": len(reports),
              "transformed_groups": sum(x["status"] == "transformed" for x in reports),
              "unchanged_insignificant_gain_groups": sum(x["status"] == "unchanged_insignificant_gain" for x in reports),
              "unchanged_zero_baseline_groups": sum(x["status"] == "unchanged_zero_baseline" for x in reports),
              "partial_groups": sum(x["contained_surface_percent"] < 99.99 for x in reports),
              "mean_baseline_surface_percent": float(np.mean([x["baseline_surface_percent"] for x in reports])),
              "mean_contained_surface_percent": float(np.mean([x["contained_surface_percent"] for x in reports])),
              "groups": reports, "runtime_seconds": time.perf_counter() - started}
    (output_dir / "structural_fit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-dir", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(alignment_dir=args.alignment_dir, output_dir=args.output_dir)
    print(json.dumps({"groups": result["group_count"], "transformed": result["transformed_groups"],
                      "unchanged_insignificant_gain": result["unchanged_insignificant_gain_groups"],
                      "unchanged_zero_baseline": result["unchanged_zero_baseline_groups"],
                      "partial": result["partial_groups"],
                      "mean_contained_surface_percent": result["mean_contained_surface_percent"],
                      "output": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
