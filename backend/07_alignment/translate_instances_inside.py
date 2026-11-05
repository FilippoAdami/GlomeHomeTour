#!/usr/bin/env python3
"""Apply minimum per-instance translations into the regularized shell envelope."""

import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from shapely import affinity, covers, linestrings, polygons, area, points
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import nearest_points
from shapely.ops import unary_union



ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "current_scene/06_mesh_elaboration/new_step"
OUTPUT = INPUT / "outputs"
REGIONS = ROOT / "current_scene/06_mesh_elaboration/generic_graph_regions"


def triangles_inside_footprint(triangles, footprint, chunk_size=50_000):
    inside = np.zeros(len(triangles), dtype=bool)
    xz = triangles[..., (0, 2)]
    for start in range(0, len(triangles), chunk_size):
        part = xz[start:start + chunk_size]
        triangle_shapes = polygons(part)
        nondegenerate = area(triangle_shapes) > 1e-12
        result = np.zeros(len(part), dtype=bool)
        if np.any(nondegenerate):
            result[nondegenerate] = covers(footprint, triangle_shapes[nondegenerate])
        if np.any(~nondegenerate):
            result[~nondegenerate] = covers(footprint, linestrings(part[~nondegenerate]))
        inside[start:start + len(part)] = result
    return inside


def minimum_plan_translation(triangles, footprint):
    """Find the shortest translation whose projected triangles are all covered."""
    vertices = np.unique(triangles.reshape(-1, 3), axis=0)
    hull = MultiPoint(vertices[:, (0, 2)]).convex_hull
    constraints = np.asarray(hull.exterior.coords[:-1] if hull.geom_type == "Polygon"
                             else hull.coords, dtype=np.float64)
    projected = triangles[..., (0, 2)]

    for _ in range(10):
        feasible = None
        for x, z in constraints:
            allowed = affinity.translate(footprint, xoff=-x, yoff=-z)
            feasible = allowed if feasible is None else feasible.intersection(allowed)
            if feasible.is_empty:
                return None
        closest = nearest_points(Point(0, 0), feasible)[1]
        shift = np.asarray(closest.coords[0], dtype=np.float64)
        moved = triangles.copy()
        moved[..., 0] += shift[0]
        moved[..., 2] += shift[1]
        inside = triangles_inside_footprint(moved, footprint)
        if np.all(inside):
            return shift

        bad = moved[~inside]
        edge = np.stack(((bad[:, 0] + bad[:, 1]) / 2,
                         (bad[:, 1] + bad[:, 2]) / 2,
                         (bad[:, 2] + bad[:, 0]) / 2), axis=1)
        samples = np.concatenate((bad, edge, bad.mean(axis=1, keepdims=True)), axis=1)
        candidates = samples[..., (0, 2)].reshape(-1, 2)
        outside = ~covers(footprint, points(candidates))
        if not np.any(outside):
            return None
        candidates = np.unique(np.round(candidates[outside], 5), axis=0)
        if len(candidates) > 1024:
            candidates = candidates[np.linspace(0, len(candidates) - 1, 1024, dtype=int)]
        constraints = np.unique(np.round(np.concatenate((constraints, candidates)), 5), axis=0)
    return None


def vertical_limits(xyz, scene):
    down = np.zeros((len(xyz), 6), dtype=np.float32)
    down[:, :3], down[:, 1], down[:, 4] = xyz, xyz[:, 1] + 0.001, -1.0
    up = np.zeros_like(down)
    up[:, :3], up[:, 1], up[:, 4] = xyz, xyz[:, 1] - 0.001, 1.0
    floor_hit = scene.cast_rays(o3d.core.Tensor(down))
    ceiling_hit = scene.cast_rays(o3d.core.Tensor(up))
    floor_y = xyz[:, 1] + 0.001 - floor_hit["t_hit"].numpy()
    ceiling_y = xyz[:, 1] - 0.001 + ceiling_hit["t_hit"].numpy()
    floor_ok = np.isfinite(floor_y) & (floor_hit["primitive_normals"].numpy()[:, 1] >= 0.65)
    ceiling_ok = np.isfinite(ceiling_y) & (ceiling_hit["primitive_normals"].numpy()[:, 1] <= -0.65)
    return floor_y, ceiling_y, floor_ok, ceiling_ok


def faces_inside_shell(samples, scene, chunk_size=20_000):
    inside = np.zeros(len(samples), dtype=bool)
    for start in range(0, len(samples), chunk_size):
        xyz = samples[start:start + chunk_size]
        down = np.zeros((len(xyz), 6), dtype=np.float32)
        down[:, :3], down[:, 1], down[:, 4] = xyz, xyz[:, 1] + 0.001, -1.0
        up = np.zeros_like(down)
        up[:, :3], up[:, 1], up[:, 4] = xyz, xyz[:, 1] - 0.001, 1.0
        floor = scene.cast_rays(o3d.core.Tensor(down))
        ceiling = scene.cast_rays(o3d.core.Tensor(up))
        inside[start:start + len(xyz)] = (
            np.isfinite(floor["t_hit"].numpy()) & np.isfinite(ceiling["t_hit"].numpy())
            & (floor["primitive_normals"].numpy()[:, 1] >= 0.65)
            & (ceiling["primitive_normals"].numpy()[:, 1] <= -0.65))
    return inside


def triangle_samples(triangles):
    edges = np.stack(((triangles[:, 0] + triangles[:, 1]) / 2,
                      (triangles[:, 1] + triangles[:, 2]) / 2,
                      (triangles[:, 2] + triangles[:, 0]) / 2), axis=1)
    return np.concatenate((triangles, edges, triangles.mean(axis=1, keepdims=True)), axis=1)


def main():
    started = time.perf_counter()
    # A small deterministic check protects the minimum-translation solver's core invariant.
    from shapely.geometry import box
    check = np.asarray([[[ -0.2, 0.0, 0.2], [0.4, 0.0, 0.2], [0.4, 0.0, 0.8]]])
    check_shift = minimum_plan_translation(check, box(0, 0, 1, 1))
    assert np.allclose(check_shift, [0.2, 0.0], atol=1e-5)

    shell_path = INPUT / "all_structural_regularized.ply"
    shell = o3d.io.read_triangle_mesh(str(shell_path))
    shell.compute_triangle_normals()
    sv, sf = np.asarray(shell.vertices), np.asarray(shell.triangles)
    sn = np.asarray(shell.triangle_normals)
    floor_faces = sf[sn[:, 1] >= 0.65]
    footprint = unary_union([Polygon(triangle[:, (0, 2)]) for triangle in sv[floor_faces]]).buffer(0).buffer(-0.01)
    if footprint.is_empty:
        raise RuntimeError("A 1 cm inset of the shell footprint is empty")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(shell))

    source_mesh = o3d.io.read_triangle_mesh(str(OUTPUT / "rigid_baseline_objects.ply"))
    vertices, faces = np.asarray(source_mesh.vertices), np.asarray(source_mesh.triangles)
    provenance = np.load(OUTPUT / "rigid_baseline_provenance.npz")
    source_face_ids, instance_ids = provenance["source_face_ids"], provenance["instance_ids"]
    if len(faces) != len(source_face_ids) or len(faces) != len(instance_ids):
        raise ValueError("Baseline faces and provenance are not aligned")
    source_regions = np.load(REGIONS / "triangle_class_labels.npy", mmap_mode="r")[source_face_ids]

    # Candidate ID zero means untracked; keep its connected semantic pieces separate.
    groups = []
    unknown = np.flatnonzero(instance_ids <= 0)
    tracked = np.flatnonzero(instance_ids > 0)
    for instance_id in np.unique(instance_ids[tracked]):
        groups.append((int(instance_id), tracked[instance_ids[tracked] == instance_id], "tracked_instance"))
    for region_id in np.unique(source_regions[unknown]):
        region_faces = unknown[source_regions[unknown] == region_id]
        used, inverse = np.unique(faces[region_faces], return_inverse=True)
        component_mesh = o3d.geometry.TriangleMesh()
        component_mesh.vertices = o3d.utility.Vector3dVector(vertices[used])
        component_mesh.triangles = o3d.utility.Vector3iVector(inverse.reshape(-1, 3))
        components, _, _ = component_mesh.cluster_connected_triangles()
        components = np.asarray(components)
        for component in np.unique(components):
            members = region_faces[components == component]
            if len(members) >= 20:
                groups.append((f"untracked_region_{int(region_id)}_component_{int(component)}",
                               members, "connected_region"))

    grouped_faces = np.zeros(len(faces), dtype=bool)
    output_vertices, output_faces, output_source_ids, output_instance_ids = [], [], [], []
    output_colors, reports = [], []
    for group_id, face_ids, group_type in groups:
        grouped_faces[face_ids] = True
        triangles = vertices[faces[face_ids]]
        inside_before = triangles_inside_footprint(triangles, footprint)
        horizontal = minimum_plan_translation(triangles, footprint) if not np.all(inside_before) else np.zeros(2)
        status = "already_inside" if horizontal is not None and not np.any(horizontal) else "translated"
        if horizontal is None:
            horizontal = np.zeros(2)
            status = "no_feasible_plan_translation"
        plan_feasible = status != "no_feasible_plan_translation"
        moved = triangles.copy()
        moved[..., 0] += horizontal[0]
        moved[..., 2] += horizontal[1]

        used, inverse = np.unique(faces[face_ids], return_inverse=True)
        moved_vertices = vertices[used].copy()
        moved_vertices[:, 0] += horizontal[0]
        moved_vertices[:, 2] += horizontal[1]
        vertical = 0.0
        samples = triangle_samples(moved)
        if plan_feasible and not np.all(faces_inside_shell(samples.reshape(-1, 3), scene)):
            limits = vertical_limits(moved_vertices, scene)
            floor_y, ceiling_y, floor_ok, ceiling_ok = limits
            if np.all(floor_ok & ceiling_ok):
                lower = float(np.max(floor_y + 0.001 - moved_vertices[:, 1]))
                upper = float(np.min(ceiling_y - 0.001 - moved_vertices[:, 1]))
                if lower <= upper:
                    vertical = float(np.clip(0.0, lower, upper))
                    moved_vertices[:, 1] += vertical
                    moved[..., 1] += vertical
                    samples = triangle_samples(moved)
                    if not np.all(faces_inside_shell(samples.reshape(-1, 3), scene)):
                        status = "vertical_sample_conflict"
                else:
                    status = "object_too_tall_for_local_shell"
            else:
                status = "missing_local_floor_or_ceiling"

        after_plan = triangles_inside_footprint(moved, footprint)
        inside_volume = faces_inside_shell(triangle_samples(moved).reshape(-1, 3), scene)
        if not np.all(after_plan) and plan_feasible:
            status = "plan_translation_unresolved"
        elif not np.all(inside_volume) and status in {"already_inside", "translated"}:
            status = "vertical_translation_unresolved"

        offset = sum(len(v) for v in output_vertices)
        output_vertices.append(moved_vertices)
        output_faces.append(inverse.reshape(-1, 3) + offset)
        output_source_ids.append(source_face_ids[face_ids])
        output_instance_ids.append(instance_ids[face_ids])
        colors = np.asarray(source_mesh.vertex_colors)
        output_colors.append(colors[used] if len(colors) else np.zeros((len(used), 3)))
        reports.append({"group_id": group_id, "group_type": group_type, "source_faces": int(len(face_ids)),
                        "faces_outside_plan_before": int((~inside_before).sum()),
                        "faces_outside_plan_after": int((~after_plan).sum()),
                        "triangle_samples_outside_vertical_envelope": int((~inside_volume).sum()),
                        "translation_m": [float(horizontal[0]), float(vertical), float(horizontal[1])],
                        "translation_distance_m": float(np.linalg.norm([horizontal[0], vertical, horizontal[1]])),
                        "status": status})

    untracked_small = np.flatnonzero(~grouped_faces)
    if len(untracked_small):
        used, inverse = np.unique(faces[untracked_small], return_inverse=True)
        offset = sum(len(v) for v in output_vertices)
        output_vertices.append(vertices[used])
        output_faces.append(inverse.reshape(-1, 3) + offset)
        output_source_ids.append(source_face_ids[untracked_small])
        output_instance_ids.append(instance_ids[untracked_small])
        colors = np.asarray(source_mesh.vertex_colors)
        output_colors.append(colors[used] if len(colors) else np.zeros((len(used), 3)))

    result = o3d.geometry.TriangleMesh()
    result.vertices = o3d.utility.Vector3dVector(np.concatenate(output_vertices))
    result.triangles = o3d.utility.Vector3iVector(np.concatenate(output_faces).astype(np.int32))
    result.vertex_colors = o3d.utility.Vector3dVector(np.concatenate(output_colors))
    result.compute_vertex_normals()
    object_path = OUTPUT / "instance_translated_objects.ply"
    o3d.io.write_triangle_mesh(str(object_path), result, write_ascii=False)
    combined = shell + result
    o3d.io.write_triangle_mesh(str(OUTPUT / "instance_translated_scene.ply"), combined, write_ascii=False)
    np.savez_compressed(OUTPUT / "instance_translated_provenance.npz",
                        source_face_ids=np.concatenate(output_source_ids),
                        instance_ids=np.concatenate(output_instance_ids))
    report = {"status": "minimum_instance_translation_candidate", "method": "minimum-norm plan translation by intersecting translated inset footprints, then minimum feasible vertical translation",
              "wall_clearance_m": 0.01, "source_faces": int(len(faces)),
              "faces_written": int(sum(len(x) for x in output_source_ids)),
              "untracked_small_faces_left_at_global_alignment": int(len(untracked_small)),
              "translated_groups": sum(x["translation_distance_m"] > 1e-8 for x in reports),
              "unresolved_groups": sum(x["status"] not in {"translated", "already_inside"} for x in reports),
              "group_faces_outside_plan_after": sum(x["faces_outside_plan_after"] for x in reports),
              "groups_with_vertical_envelope_residuals": sum(x["triangle_samples_outside_vertical_envelope"] > 0 for x in reports),
              "groups": reports, "runtime_seconds": time.perf_counter() - started}
    (OUTPUT / "instance_translation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("faces_written", "translated_groups", "unresolved_groups",
                                               "untracked_small_faces_left_at_global_alignment", "runtime_seconds")}, indent=2))


if __name__ == "__main__":
    main()
