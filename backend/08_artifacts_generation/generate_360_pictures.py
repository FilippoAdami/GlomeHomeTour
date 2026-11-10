#!/usr/bin/env python3
"""Select supported tour locations with adaptive height and render sealed 360 pictures."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import hashlib
import logging
from pathlib import Path
import shutil
import tempfile
import time
import uuid

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra, connected_components

from best_views_visibility import SurfaceVisibility
from extract_best_views import _hash, _json, _rigid, image_quality, load_cameras, sample_surface
from generate_floor_level_meshes import DEFAULT_ALIGNMENT_DIR, _plane, _read_mesh, _selected_ceiling_faces


LOGGER = logging.getLogger("generate_360_pictures")
MACRO = "Generate 360 Pictures"
COVERAGE_TARGET = 0.70
MINIMUM_COVERAGE = 0.70
MIN_NEW_COVERAGE = 0.005
SURFACE_SAMPLES = 12_000
TRIANGLE_SCREEN_SIZE = 32_768
EXACT_CANDIDATE_BATCH = 1


def mesh_scene(vertices, faces):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, np.float32)),
                        o3d.core.Tensor(np.asarray(faces, np.uint32)))
    return scene


def optical_environment(alignment_dir, scan_ids, regions):
    from panorama_rendering import panel_faces

    vertices_all, faces_all, colors_all = [], [], []
    hashes, vertex_offset = {}, 0
    for scan_id in scan_ids:
        path = alignment_dir / "scans" / scan_id / "mesh.ply"
        vertices, faces, colors = _read_mesh(path)
        types, _ = panel_faces(scan_id, vertices, faces, regions)
        vertices_all.append(vertices)
        faces_all.append(faces[types == 0] + vertex_offset)
        colors_all.append(colors)
        vertex_offset += len(vertices)
        hashes[str(path)] = _hash(path)
    vertices = np.concatenate(vertices_all)
    faces = np.concatenate(faces_all)
    colors = np.concatenate(colors_all)
    return (mesh_scene(vertices, faces) if len(faces) else None, faces, colors), hashes


def cast(scene, origins, directions):
    origins, directions = np.broadcast_arrays(np.asarray(origins, np.float32),
                                              np.asarray(directions, np.float32))
    shape = origins.shape[:-1]
    rays = np.concatenate((origins.reshape(-1, 3), directions.reshape(-1, 3)), axis=1)
    hits = []
    for start in range(0, len(rays), 250_000):
        hits.append(scene.cast_rays(o3d.core.Tensor(rays[start:start + 250_000]))["t_hit"].numpy())
    return np.concatenate(hits).reshape(shape) if hits else np.empty(shape, np.float32)


def floor_y(xz, equation):
    equation = np.asarray(equation, np.float64)
    if equation.shape != (4,) or not np.isfinite(equation).all() or equation[1] <= 0.5:
        raise ValueError("Floor must have a finite upward plane equation")
    return -(np.asarray(xz) @ equation[[0, 2]] + equation[3]) / equation[1]


def height_policy(metric, level_heights):
    """Use one nominal height and unit policy throughout every floor in a scan."""
    if metric.get("output_coordinate_units") == "meters":
        return {"height": 1.5, "max_pose_distance": 1.0, "distance_scale": 1.0,
                "units": "meters", "method": "standard_metric_height", "allowed_heights": [step / 10 for step in range(15, 6, -2)]}
    heights = np.asarray(level_heights, np.float64)
    if not len(heights) or not np.isfinite(heights).all() or np.any(heights <= 0):
        raise ValueError("Cannot derive camera height without positive finite level heights")
    height = float(heights.min() / 2)
    return {"height": height, "max_pose_distance": 0.66 * height,
            "distance_scale": height / 1.5, "units": "COLMAP units",
            "method": "half_maximum_height_of_shortest_scan_level",
            "allowed_heights": [height * (step / 15) for step in range(15, 6, -2)]}


def _resolve_paths(alignment_dir, output_dir):
    alignment_dir = Path(alignment_dir).resolve()
    scene_dir = alignment_dir.parent
    output_dir = Path(output_dir or scene_dir / "artifacts" / MACRO).resolve()
    if (scene_dir.is_relative_to(output_dir) or output_dir.is_relative_to(alignment_dir)
            or any(output_dir.is_relative_to(p) for p in scene_dir.glob("scan_*"))):
        raise ValueError("Outputs must be separate from input scan and alignment assets")
    return alignment_dir, scene_dir, output_dir


def _start_timing(scope, name):
    LOGGER.info("%s: starting %s", scope, name)
    return time.perf_counter()


def _finish_timing(timings, scope, name, started, status="complete"):
    elapsed = time.perf_counter() - started
    timings[name] = round(elapsed, 3)
    LOGGER.info("%s: %s %s in %.2f s", scope, name,
                "completed" if status == "complete" else "failed", elapsed)
    return elapsed


def floor_geometry(records, alignment_dir):
    vertices, faces = [], []
    offset = 0
    for record in records:
        v, f, _ = _read_mesh(alignment_dir / record["source_mesh"])
        n, d = _plane(record)
        tri = v[f]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        length = np.linalg.norm(cross, axis=1)
        alignment = np.abs(cross @ n) / np.maximum(length, 1e-20)
        # A slab source includes both sides; retain only the attributed floor plane.
        residual = np.max(np.abs(tri @ n + d), axis=1)
        tolerance = max(float(np.ptp(v, axis=0).max()) * 1e-5, 1e-6)
        selected = (alignment >= np.cos(np.deg2rad(5))) & (residual <= tolerance)
        if not selected.any():
            raise ValueError(f"No attributed floor triangles: {record['plane_id']}")
        vertices.append(v)
        faces.append(f[selected] + offset)
        offset += len(v)
    return np.concatenate(vertices), np.concatenate(faces)


def level_max_height(records, ceiling_records, alignment_dir, sealed_vertices, next_floor=None):
    maximum = []
    for ceiling in ceiling_records:
        v, f, _ = _read_mesh(alignment_dir / ceiling["source_mesh"])
        selected = _selected_ceiling_faces(alignment_dir / ceiling["source_mesh"], v, f, ceiling)
        p = v[f[selected]].reshape(-1, 3)
        if len(p):
            # Each ceiling belongs to this level; use its nearest supporting floor.
            base = np.max([floor_y(p[:, [0, 2]], r["plane_equation"]) for r in records], axis=0)
            maximum.append(float(np.max(p[:, 1] - base)))
    if maximum:
        return max(maximum), "associated_ceiling_geometry"
    p = sealed_vertices
    if next_floor is not None:
        p = p[p[:, 1] <= next_floor]
    if not len(p):
        raise ValueError("Level has no upper geometry for its height fallback")
    base = np.max([floor_y(p[:, [0, 2]], r["plane_equation"]) for r in records], axis=0)
    return float(np.max(p[:, 1] - base)), "sealed_level_upper_extent"


def sample_floor(vertices, faces, step):
    scene = mesh_scene(vertices, faces)
    support = vertices[np.unique(faces)]
    low = np.floor(support[:, [0, 2]].min(0) / step) * step
    high = np.ceil(support[:, [0, 2]].max(0) / step) * step
    x, z = np.meshgrid(np.arange(low[0] + step / 2, high[0], step),
                       np.arange(low[1] + step / 2, high[1], step))
    top = float(support[:, 1].max() + max(np.ptp(support[:, 1]), step) + step)
    origins = np.column_stack((x.ravel(), np.full(x.size, top), z.ravel()))
    distance = cast(scene, origins, [0, -1, 0])
    mask = np.isfinite(distance)
    points = origins[mask].copy()
    points[:, 1] -= distance[mask]
    if not len(points):
        raise ValueError("Attributed floor footprint is empty")
    return points, {"low_xz": low.tolist(), "shape": list(x.shape), "step": step,
                    "cell_indices": np.flatnonzero(mask), "floor_scene": scene, "top": top}


def visible_segments(scene, starts, ends, epsilon):
    directions = np.asarray(ends) - np.asarray(starts)
    distance = np.linalg.norm(directions, axis=-1)
    hit = cast(scene, starts, directions)
    return (hit >= 1 - epsilon / np.maximum(distance, epsilon)) | (distance <= epsilon)


def generate_candidates(points, grid, pose_centers, height, pose_radius, scale, sealed_scene, ceiling_scene=None):
    support = grid["floor_scene"]
    low = points[:, [0, 2]].min(0)
    high = points[:, [0, 2]].max(0)
    spacing = 0.30 * scale
    x, z = np.meshgrid(np.arange(low[0], high[0] + spacing / 2, spacing),
                       np.arange(low[1], high[1] + spacing / 2, spacing))
    xz = np.vstack((np.column_stack((x.ravel(), z.ravel())), pose_centers[:, [0, 2]]))
    xz = np.unique(np.round(xz / (spacing / 20)).astype(np.int64), axis=0) * (spacing / 20)
    origins = np.column_stack((xz[:, 0], np.full(len(xz), grid["top"]), xz[:, 1]))
    distance = cast(support, origins, [0, -1, 0])
    inside = np.isfinite(distance)
    candidates = origins[inside]
    candidates[:, 1] += height - distance[inside]
    if ceiling_scene is not None and len(candidates):
        floor_origins = candidates - [0, height - 0.001 * scale, 0]
        headroom = cast(ceiling_scene, floor_origins, [0, 1, 0])
        # Never place a camera through the local ceiling.
        candidates = candidates[headroom > height - 0.001 * scale]
    if not len(pose_centers):
        return np.empty((0, 3)), [], {"within_pose_radius": 0, "blocked_from_real_poses": 0}
    if len(candidates):
        distance_to_surface = sealed_scene.compute_distance(o3d.core.Tensor(candidates.astype(np.float32))).numpy()
        candidates = candidates[distance_to_surface > 0.001 * scale]
    tree = cKDTree(pose_centers)
    distances, nearest = tree.query(candidates)
    keep = distances <= pose_radius + scale * 1e-7
    candidates, distances, nearest = candidates[keep], distances[keep], nearest[keep]
    initial = len(candidates)
    near_poses = tree.query_ball_point(candidates, pose_radius + scale * 1e-7)
    ci = np.repeat(np.arange(len(candidates)), [len(ids) for ids in near_poses])
    pi = np.asarray([j for ids in near_poses for j in ids], int)
    counts = np.zeros(len(candidates), int)
    nearest_clear = np.full(len(candidates), -1, int)
    if len(ci):
        clear = visible_segments(sealed_scene, pose_centers[pi], candidates[ci], 0.005 * scale)
        np.add.at(counts, ci[clear], 1)
        for i, j in zip(ci[clear], pi[clear]):
            if nearest_clear[i] < 0 or np.linalg.norm(candidates[i] - pose_centers[j]) < np.linalg.norm(candidates[i] - pose_centers[nearest_clear[i]]):
                nearest_clear[i] = j
    keep = counts > 0
    blocked_count = int((~keep).sum())
    candidates, nearest_clear, counts = candidates[keep], nearest_clear[keep], counts[keep]
    if not len(candidates):
        return candidates, [], {"within_pose_radius": initial, "blocked_from_real_poses": initial}
    angle = np.arange(180) * np.pi / 90
    directions = np.column_stack((np.cos(angle), np.zeros(len(angle)), np.sin(angle)))
    hits = cast(sealed_scene, candidates[:, None, :], directions[None])
    close = hits < 0.75 * scale
    stats, keep = [], []
    for i, h in enumerate(hits):
        if h.min() < 0.015 * scale:
            continue
        blocked = close[i]
        longest, current = 0, 0
        for b in np.tile(blocked, 2):
            current = current + 1 if b else 0
            longest = max(longest, current)
        sector = min(longest, len(blocked)) / len(blocked)
        low_clearance = float(np.quantile(np.minimum(h, 10 * scale), 0.1))
        quality = float(np.exp(-2 * blocked.mean() - sector) * min(1, low_clearance / (0.5 * scale)))
        keep.append(i)
        stats.append({"nearest_camera_index": int(nearest_clear[i]),
                      "nearest_camera_distance": float(np.linalg.norm(candidates[i] - pose_centers[nearest_clear[i]])),
                      "supporting_pose_count": int(counts[i]),
                      "horizontal_min_distance": float(np.minimum(h, 10 * scale).min()),
                      "horizontal_p10_distance": low_clearance,
                      "near_horizontal_fraction": float(blocked.mean()),
                      "largest_near_sector_degrees": float(sector * 360),
                      "horizontal_quality": quality})
    return candidates[keep], stats, {"within_pose_radius": initial,
                                    "blocked_from_real_poses": blocked_count,
                                    "horizontal_collision_rejections": int(initial - blocked_count - len(keep))}


def adaptive_candidates(points, grid, pose_centers, policy, scene, ceiling_scene=None):
    """Keep one height per X/Z location, lowering only for useful obstruction relief."""
    scale = policy["distance_scale"]
    options, statistics, reports = [], [], []
    azimuth, elevation = np.meshgrid(np.arange(90) * np.pi / 45, np.deg2rad(np.arange(-15, 56, 5)))
    directions = np.column_stack((np.cos(elevation).ravel() * np.cos(azimuth).ravel(),
                                  np.sin(elevation).ravel(),
                                  np.cos(elevation).ravel() * np.sin(azimuth).ravel()))
    for height in policy["allowed_heights"]:
        candidates, stats, report = generate_candidates(points, grid, pose_centers, height,
            policy["max_pose_distance"], scale, scene, ceiling_scene)
        reports.append({"height": height, **report, "eligible_count": len(candidates)})
        if len(candidates):
            hits = cast(scene, candidates[:, None, :], directions[None])
            weights = np.cos(elevation).ravel()
            upper = elevation.ravel() > 0
            near_fraction = np.average(hits[:, upper] < 0.75 * scale, axis=1, weights=weights[upper])
            band_fraction = np.average(hits < 0.75 * scale, axis=1, weights=weights)
            for row, obstruction, band in zip(stats, near_fraction, band_fraction):
                row.update(height_above_floor=height, upper_near_fraction=float(obstruction),
                           view_band_near_fraction=float(band),
                           obstruction_quality=float(row["horizontal_quality"] * np.exp(-2 * obstruction)))
            options.extend(candidates)
            statistics.extend(stats)
    groups = defaultdict(list)
    for i, point in enumerate(options):
        groups[tuple(np.round(point[[0, 2]] / (0.015 * scale)).astype(int))].append(i)
    chosen = []
    radius_only_rejections = 0
    obstruction_rejections = 0
    for ids in groups.values():
        nominal = next((i for i in ids if statistics[i]["height_above_floor"] == policy["height"]), None)
        eligible = [i for i in ids if statistics[i]["horizontal_min_distance"] >= 0.10 * scale
                    and statistics[i]["horizontal_p10_distance"] >= 0.25 * scale
                    and statistics[i]["view_band_near_fraction"] <= 0.50]
        if not eligible:
            obstruction_rejections += 1
            continue
        # A small height preference prevents unnecessary lowering in unobstructed rooms.
        best = max(eligible, key=lambda i: statistics[i]["obstruction_quality"]
                   - 0.03 * (policy["height"] - statistics[i]["height_above_floor"]) / (0.2 * scale))
        if nominal in eligible:
            base = statistics[nominal]
            obstructed = max(base["near_horizontal_fraction"], base["upper_near_fraction"]) > 0.05
            if not obstructed or statistics[best]["obstruction_quality"] < base["obstruction_quality"] + 0.03:
                best = nominal
        row = statistics[best]
        if nominal is None:
            lower = np.asarray(options[best])
            standard = lower + [0, policy["height"] - row["height_above_floor"], 0]
            blocked = not bool(visible_segments(scene, pose_centers[row["nearest_camera_index"]], standard, 0.005 * scale))
            if ceiling_scene is not None:
                origin = lower - [0, row["height_above_floor"] - 0.001 * scale, 0]
                blocked |= bool(cast(ceiling_scene, origin, [0, 1, 0]) <= policy["height"] - 0.001 * scale)
            angle = np.arange(180) * np.pi / 90
            horizontal = np.column_stack((np.cos(angle), np.zeros(len(angle)), np.sin(angle)))
            near_horizontal = float((cast(scene, standard, horizontal) < 0.75 * scale).mean())
            near_upper = float(np.average(cast(scene, standard, directions[upper]) < 0.75 * scale, weights=np.cos(elevation).ravel()[upper]))
            relieved = max(near_horizontal - row["near_horizontal_fraction"], near_upper - row["upper_near_fraction"]) >= 0.05
            if not blocked and not relieved:
                radius_only_rejections += 1
                continue
        row["height_selection_reason"] = ("standard_height" if best == nominal else
            "nominal_height_ineligible" if nominal not in eligible else
            "upper_obstruction_reduced" if statistics[nominal]["upper_near_fraction"] > row["upper_near_fraction"]
            else "horizontal_obstruction_reduced")
        chosen.append(best)
    candidates = np.asarray(options, np.float64).reshape(-1, 3)[chosen]
    return candidates, [statistics[i] for i in chosen], {"height_trials": reports,
        "retained_locations": len(chosen), "radius_only_lowering_rejections": radius_only_rejections,
        "severe_obstruction_rejections": obstruction_rejections,
        "lowered_locations": sum(statistics[i]["height_above_floor"] < policy["height"] for i in chosen)}


def slice_targets(level, scan_id, alignment_dir, floor_records, scale=1.0):
    """Recover exact per-scan triangle ownership from the macro-one merge order."""
    path = Path(level["sealed_mesh"])
    if not path.is_absolute():
        path = alignment_dir / path
    vertices, faces, _ = _read_mesh(path)
    counts = [int(row["sealed"]["output_triangles"]) for row in level["scans"]]
    if sum(counts) != len(faces) or any(n < 0 for n in counts):
        raise ValueError("Floor-slice face counts disagree with the ownership manifest")
    index = next(i for i, row in enumerate(level["scans"]) if row["scan_id"] == scan_id)
    if level["scans"][index]["output_status"] != "generated_for_floor_level":
        raise ValueError("Carried-forward scans have no new panorama targets")
    start = sum(counts[:index])
    own_faces = faces[start:start + counts[index]]
    centers = vertices[own_faces].mean(axis=1)
    if not floor_records:
        raise ValueError("Triangle coverage requires this scan's attributed floor planes")
    equations = [r["plane_equation"] for r in floor_records]
    base = np.max([floor_y(centers[:, [0, 2]], equation) for equation in equations], axis=0)
    tolerance = 0.0001 * scale
    active = centers[:, 1] >= base - tolerance
    if not active.any():
        raise ValueError("Floor slice contains no triangles at or above its own floor")
    return centers[active], {"mesh": str(path), "mesh_sha256": _hash(path),
        "scan_face_range": [start, start + counts[index]], "scan_triangle_count": counts[index],
        "excluded_below_floor_triangles": int((~active).sum()), "triangle_count": int(active.sum()),
        "floor_plane_equations": equations, "below_floor_tolerance": tolerance,
        "macro_one_greyscale_triangle_count": int(level["scans"][index]["sealed"].get("greyscale_triangles", 0)),
        "level_membership_rule": "final_slice_centroid_at_or_above_own_scan_floor",
        "visibility_rule": "triangle_centroid_first_hit_on_full_sealed_scan", "weighting": "one_per_triangle"}


def triangle_visibility(scene, candidates, targets, scale):
    """Return first-hit visibility for every supplied triangle centroid."""
    matrix = np.zeros((len(candidates), len(targets)), bool)
    for i, point in enumerate(candidates):
        for start in range(0, len(targets), 250_000):
            stop = min(start + 250_000, len(targets))
            directions = targets[start:stop] - point
            hit = cast(scene, point, directions)
            tolerance = 1e-4 + 0.0001 * scale / np.maximum(np.linalg.norm(directions, axis=1), 1e-8 * scale)
            matrix[i, start:stop] = np.abs(hit - 1) <= tolerance
        if (i + 1) % 50 == 0:
            LOGGER.info("Triangle visibility: %d/%d candidates, %d triangles each", i + 1, len(candidates), len(targets))
    return matrix


def source_surface_support(scan_dir, alignment_dir, transform, cameras):
    raw = scan_dir / ".raw_unaligned" / "mesh.ply"
    vertices, faces, _ = _read_mesh(raw)
    samples = sample_surface(vertices, faces, SURFACE_SAMPLES)
    samples["positions"] = samples["positions"].astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]
    samples["normals"] = samples["normals"] @ transform[:3, :3].T
    vertices = vertices.astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]
    renderer = SurfaceVisibility(vertices, faces, samples["positions"], samples["normals"], samples["areas"], "cuda")
    accepted, excluded = image_quality(cameras)
    all_indices = {c["image_id"]: i for i, c in enumerate(cameras)}
    scores = np.zeros((len(cameras), SURFACE_SAMPLES), np.float32)
    sizes = defaultdict(list)
    for c in accepted:
        sizes[(c["width"], c["height"])].append(c)
    for (width, height), group in sizes.items():
        ratio = 512 / max(width, height)
        for start in range(0, len(group), 16):
            batch = group[start:start + 16]
            intrinsic = np.asarray([c["intrinsics"] for c in batch])
            intrinsic[:, :2] *= ratio
            score, _, _ = renderer.evaluate([c["world_to_camera"] for c in batch], intrinsic,
                                            round(width * ratio), round(height * ratio))
            score = score.cpu().numpy()
            for c, row in zip(batch, score):
                scores[all_indices[c["image_id"]]] = row * c["quality"]["factor"]
    images_digest = hashlib.sha256()
    for camera in cameras:
        images_digest.update(camera["image_name"].encode())
        images_digest.update(bytes.fromhex(_hash(Path(camera["source_image"]))))
    return samples, scores, {"accepted_images": len(accepted), "excluded_images": len(excluded),
                             "sample_count": SURFACE_SAMPLES, "ray_count": renderer.ray_count,
                             "acquisition_mesh_sha256": _hash(raw),
                             "acquired_image_set_sha256": images_digest.hexdigest()}


def candidate_source_quality(scene, candidates, samples, scores, cameras, scale):
    centers = np.asarray([c["camera_to_world"] for c in cameras])[:, :3, 3]
    tree = cKDTree(centers)
    points = samples["positions"]
    result = []
    for c in candidates:
        visible = visible_segments(scene, c, points, 0.003 * scale)
        ids = np.flatnonzero(visible)
        if not len(ids):
            result.append(0.0)
            continue
        _, neighbors = tree.query(c, k=min(16, len(centers)))
        neighbors = np.atleast_1d(neighbors)
        target_ray = points[ids] - c
        source_ray = points[ids][None] - centers[neighbors, None]
        dot = np.sum(source_ray * target_ray, axis=-1)
        dot /= np.maximum(np.linalg.norm(source_ray, axis=-1) * np.linalg.norm(target_ray, axis=-1), 1e-12)
        angular = np.clip((dot - 0.5) / 0.5, 0, 1)
        support = (scores[neighbors][:, ids] * angular).max(0)
        result.append(float(support.mean()))
    return np.asarray(result)


def select_points(visibility, quality, target=COVERAGE_TARGET):
    if visibility.ndim != 2 or len(quality) != len(visibility) or visibility.shape[1] == 0:
        raise ValueError("Coverage rows and quality must describe the same nonempty triangle set")
    covered = np.zeros(visibility.shape[1], bool)
    selected = []
    attainable = float(visibility.any(0).mean()) if len(visibility) else 0.0
    effective_target = target
    stopping_reason = "candidate_coverage_exhausted"
    counts = np.count_nonzero(visibility, axis=1)
    single = np.flatnonzero((counts / len(covered) >= target) & (np.asarray(quality) > 0))
    if len(single):
        best = int(single[np.argmax(counts[single] * np.asarray(quality)[single])])
        selected = [best]
        covered |= visibility[best]
        stopping_reason = "single_camera_target_reached"
    while len(selected) < len(visibility) and covered.mean() < effective_target:
        gain = np.count_nonzero(visibility & ~covered, axis=1)
        value = gain * np.asarray(quality)
        value[selected] = -1
        if selected:
            useful = ((gain / len(covered) >= MIN_NEW_COVERAGE)
                      | ((covered.sum() + gain) / len(covered) >= effective_target))
            value[~useful] = -1
        best = int(np.argmax(value))
        if gain[best] == 0 or value[best] <= 0:
            if np.any(gain > 0):
                stopping_reason = "marginal_gain_below_half_percent"
            break
        selected.append(best)
        covered |= visibility[best]
        if covered.mean() >= effective_target:
            stopping_reason = "target_reached"
            break
    for index in selected[::-1]:
        remaining = [i for i in selected if i != index]
        if remaining and visibility[remaining].any(0).mean() >= effective_target:
            selected.remove(index)
    coverage = float(visibility[selected].any(0).mean()) if selected else 0.0
    return selected, {"coverage": coverage, "attainable_coverage": attainable,
                      "covered_triangle_count": int(visibility[selected].any(0).sum()) if selected else 0,
                      "attainable_triangle_count": int(visibility.any(0).sum()),
                      "triangle_count": int(visibility.shape[1]),
                      "target": target, "minimum": MINIMUM_COVERAGE,
                      "effective_selection_target": effective_target, "minimum_new_coverage": MIN_NEW_COVERAGE,
                      "target_reached": coverage >= target,
                      "minimum_reached": coverage >= MINIMUM_COVERAGE,
                      "stopping_reason": stopping_reason}


def height_retry_reason(candidates, selected, visibility, statistics, selection, scale):
    """Nearby weak views indicate an occluder; evaluate a lower camera before adding more."""
    if not selected:
        return "no_usable_camera", 0, 0.0, 1.0
    centers = candidates[selected][:, [0, 2]]
    distance = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    cluster_count = int((distance < 3.0 * scale).sum(axis=1).max())
    single_coverage = float(visibility.mean(axis=1).max())
    near_fraction = max(statistics[i]["view_band_near_fraction"] for i in selected)
    if not selection["minimum_reached"]:
        reason = "low_triangle_coverage"
    elif cluster_count >= 3 and single_coverage < COVERAGE_TARGET:
        reason = "clustered_views_with_low_individual_coverage"
    elif near_fraction > 0.10:
        reason = "nearby_view_obstruction"
    else:
        reason = None
    return reason, cluster_count, single_coverage, near_fraction


def plan_floor_points(points, grid, pose_centers, policy, scene, ceiling_scene, targets,
                      samples, source_scores, cameras):
    """Search in 20 cm steps, preserving the best compact set within the original pose bound."""
    scale = policy["distance_scale"]
    sample_size = min(TRIANGLE_SCREEN_SIZE, len(targets))
    if sample_size == len(targets):
        sample_indices = np.arange(len(targets))
    else:
        sample_indices = np.sort(np.random.default_rng(36008).choice(len(targets), sample_size, replace=False))
    screen_targets = targets[sample_indices]
    best, best_rank, trials = None, None, []
    for height in policy["allowed_heights"]:
        candidates, statistics, candidate_report = adaptive_candidates(
            points, grid, pose_centers, {**policy, "allowed_heights": [height]}, scene, ceiling_scene)
        LOGGER.info("Height %.4f: measuring %d candidates against %d slice triangles", height, len(candidates), len(targets))
        visibility = triangle_visibility(scene, candidates, screen_targets, scale)
        source_quality = candidate_source_quality(scene, candidates, samples, source_scores, cameras, scale)
        quality = np.asarray([(0.15 + 0.85 * row["obstruction_quality"]) * (0.25 + 0.75 * q)
                              for row, q in zip(statistics, source_quality)])
        screen_selected, screen_selection = select_points(visibility, quality)
        screen_counts = np.count_nonzero(visibility, axis=1)
        score_order = np.argsort(-(screen_counts * quality), kind="stable")
        best_coverage = int(np.argmax(screen_counts)) if len(screen_counts) else None
        candidate_order = list(dict.fromkeys(
            screen_selected + ([] if best_coverage is None else [best_coverage]) + score_order.tolist()))
        exact_ids, exact_rows = [], []
        selection = None
        local_selected = []
        # A clearly unattainable sampled pool should trigger a lower-height trial,
        # not exact rays for every candidate at this height.
        search_limit = len(candidate_order)
        if screen_selection["attainable_coverage"] < COVERAGE_TARGET - 0.01:
            search_limit = min(EXACT_CANDIDATE_BATCH, search_limit)
        for start in range(0, search_limit, EXACT_CANDIDATE_BATCH):
            batch_ids = candidate_order[start:start + EXACT_CANDIDATE_BATCH]
            exact_ids.extend(batch_ids)
            exact_rows.extend(triangle_visibility(scene, candidates[batch_ids], targets, scale))
            exact_matrix = np.asarray(exact_rows, bool).reshape(-1, len(targets))
            local_selected, selection = select_points(exact_matrix, quality[exact_ids])
            if selection["target_reached"]:
                break
        if selection is None:
            exact_matrix = np.empty((0, len(targets)), bool)
            local_selected, selection = select_points(exact_matrix, quality)
        evaluated_attainable = selection["attainable_coverage"]
        evaluated_attainable_count = selection["attainable_triangle_count"]
        all_candidates_exact = len(exact_ids) == len(candidates) or sample_size == len(targets)
        pool_attainable = (evaluated_attainable if len(exact_ids) == len(candidates)
                           else float(visibility.any(axis=0).mean()) if sample_size else 0.0)
        pool_attainable_count = (evaluated_attainable_count if len(exact_ids) == len(candidates)
                                 else int(round(pool_attainable * len(targets))))
        selection.update({
            "exact_attainable_coverage_evaluated_candidates": evaluated_attainable,
            "exact_attainable_triangle_count_evaluated_candidates": evaluated_attainable_count,
            "attainable_coverage": pool_attainable,
            "attainable_triangle_count": pool_attainable_count,
            "attainable_sample_triangle_count": int(visibility.any(axis=0).sum()),
            "attainable_sample_size": sample_size,
            "attainable_is_estimate": not all_candidates_exact and sample_size < len(targets),
            "estimated_selected_coverage": screen_selection["coverage"],
            "exact_candidate_count": len(exact_ids),
        })
        selected = [exact_ids[i] for i in local_selected]
        exact_visibility = exact_matrix[local_selected]
        reason, cluster_count, single_coverage, near_fraction = height_retry_reason(
            candidates[exact_ids], local_selected, exact_matrix,
            [statistics[i] for i in exact_ids], selection, scale)
        trials.append({"height": height, "candidate_count": len(candidates), "point_count": len(selected),
            "coverage": selection["coverage"], "attainable_coverage": selection["attainable_coverage"],
            "exact_candidate_count": len(exact_ids),
            "best_single_point_coverage": single_coverage, "max_neighbors_within_3m": cluster_count,
            "max_selected_near_view_fraction": near_fraction, "lowering_reason": reason})
        mean_quality = float(quality[selected].mean()) if selected else 0.0
        rank = (selection["minimum_reached"], -len(selected) if selection["minimum_reached"] else screen_selection["coverage"],
                -near_fraction, mean_quality, selection["coverage"])
        if best is None or rank > best_rank:
            best_rank = rank
            best = {"height": height, "candidates": candidates, "statistics": statistics,
                    "candidate_report": candidate_report, "visibility": visibility,
                    "source_quality": source_quality, "selected": selected, "selection": selection,
                    "selected_visibility": exact_visibility, "exact_candidate_indices": exact_ids,
                    "exact_visibility": exact_matrix, "estimated_coverage": screen_selection["coverage"],
                    "screen_attainable_sample_triangle_count": int(visibility.any(axis=0).sum()),
                    "retry_reason": reason}
        LOGGER.info("Height %.4f: %d views, %.1f%% coverage, cluster %d, near obstruction %.1f%%; %s",
                    height, len(selected), selection["coverage"] * 100, cluster_count, near_fraction * 100,
                    reason or "target reached with clear views")
        if best["retry_reason"] is None:
            break
    if best and not best["selection"]["target_reached"] and len(best["exact_candidate_indices"]) < len(best["candidates"]):
        LOGGER.info("Height %.4f: exact-checking the full candidate pool for the best below-target trial", best["height"])
        exact_matrix = triangle_visibility(scene, best["candidates"], targets, scale)
        selected, selection = select_points(exact_matrix, np.asarray([
            (0.15 + 0.85 * row["obstruction_quality"]) * (0.25 + 0.75 * q)
            for row, q in zip(best["statistics"], best["source_quality"])]))
        selection["estimated_selected_coverage"] = best["estimated_coverage"]
        selection["attainable_sample_triangle_count"] = best["screen_attainable_sample_triangle_count"]
        selection["attainable_sample_size"] = sample_size
        selection["attainable_is_estimate"] = False
        reason, cluster_count, single_coverage, near_fraction = height_retry_reason(
            best["candidates"], selected, exact_matrix, best["statistics"], selection, scale)
        best.update(selected=selected, selection=selection, selected_visibility=exact_matrix[selected],
                    exact_candidate_indices=list(range(len(best["candidates"]))), exact_visibility=exact_matrix,
                    visibility=exact_matrix, retry_reason=reason)
        for trial in trials:
            if trial["height"] == best["height"]:
                trial.update(point_count=len(selected), coverage=selection["coverage"],
                    attainable_coverage=selection["attainable_coverage"],
                    exact_candidate_count=len(best["candidates"]),
                    best_single_point_coverage=single_coverage,
                    max_neighbors_within_3m=cluster_count,
                    max_selected_near_view_fraction=near_fraction,
                    lowering_reason=reason)
    best["height_search"] = {"height_step": 0.20 * scale, "minimum_height": min(policy["allowed_heights"]),
        "cluster_distance": 3.0 * scale, "cluster_min_images": 3, "clear_view_max_near_fraction": 0.10,
        "chosen_height": best["height"], "trials": trials,
        "stopping_reason": "clear_target_reached" if best["retry_reason"] is None else "minimum_height_reached"}
    all_candidates_exact = (len(best["exact_candidate_indices"]) == len(best["candidates"])
                            or sample_size == len(targets))
    best["selection"]["attainable_scope"] = ("all_candidates_exact" if all_candidates_exact
                                               else "all_candidates_uniform_triangle_sample")
    best["candidate_visibility"] = {"method": "exact_all_candidates" if all_candidates_exact
                                    else "uniform_triangle_centroid_sample",
        "sample_triangle_count": sample_size, "total_triangle_count": len(targets),
        "sample_seed": 36008, "exact_candidate_count": len(best["exact_candidate_indices"])}
    return best


def tour_links(scene, candidates, selected, scale, floor_scene=None):
    if len(selected) <= 1:
        return [], len(selected)
    pairs = np.asarray(list(cKDTree(candidates).query_pairs(1.1 * scale)), int).reshape(-1, 2)
    if len(pairs):
        pairs = pairs[visible_segments(scene, candidates[pairs[:, 0]], candidates[pairs[:, 1]], 0.005 * scale)]
    if floor_scene is not None and len(pairs):
        # Check the entire walkway interval, including holes between valid endpoints.
        fraction = np.linspace(0, 1, 25)[None, :, None]
        segment = candidates[pairs[:, 0], None] * (1 - fraction) + candidates[pairs[:, 1], None] * fraction
        support = cast(floor_scene, segment, [0, -1, 0])
        pairs = pairs[np.isfinite(support).all(axis=1)]
    weight = np.linalg.norm(candidates[pairs[:, 0]] - candidates[pairs[:, 1]], axis=1)
    graph = coo_matrix((np.tile(weight, 2), (np.r_[pairs[:, 0], pairs[:, 1]],
                                            np.r_[pairs[:, 1], pairs[:, 0]])),
                       shape=(len(candidates), len(candidates))).tocsr()
    _, components = connected_components(graph)
    distance, predecessor = dijkstra(graph, indices=selected, return_predecessors=True)
    # A minimum spanning forest links panoramas using clear candidate paths.
    edges, sets = [], list(range(len(selected)))
    def root(i):
        while sets[i] != i:
            i = sets[i]
        return i
    for length, i, j in sorted((distance[i, selected[j]], i, j)
                               for i in range(len(selected)) for j in range(i + 1, len(selected))):
        if not np.isfinite(length) or root(i) == root(j):
            continue
        sets[root(i)] = root(j)
        route, k = [], selected[j]
        while k != selected[i]:
            route.append(int(k))
            k = int(predecessor[i, k])
        route.append(selected[i])
        route.reverse()
        edges.append({"from": i + 1, "to": j + 1, "path_length": float(length),
                      "path_world": candidates[route].tolist()})
    return edges, len(set(components[selected]))


def coverage_plot(path, targets, visibility, candidates, selected, camera_centers, statistics, report):
    low = targets[:, [0, 2]].min(0)
    extent = np.maximum(np.ptp(targets[:, [0, 2]], axis=0), 0.01)
    factor = 800 / extent.max()
    width, height = np.maximum(np.ceil(extent * factor).astype(int) + 30, 200)
    canvas = Image.new("RGB", (max(width, 850), height + 80), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f"Visible triangles {report['coverage']:.1%}; attainable {report['attainable_coverage']:.1%}; target {COVERAGE_TARGET:.0%}", fill="black")
    draw.text((12, 30), "Green: visible; red: unseen triangle centroids; blue: acquired poses; yellow: panoramas", fill="black")
    draw.text((12, 50), "+X right; +Z up; all current-level slice triangles have equal weight", fill="black")
    def pixel(p):
        x, z = (p[[0, 2]] - low) * factor
        return int(x + 15), int(height + 65 - z)
    covered = visibility[selected].any(0) if selected else np.zeros(len(targets), bool)
    # Draw all unseen/seen triangles; overlapping top-down projections are diagnostic only.
    projected = (targets[:, [0, 2]] - low) * factor
    xs = (projected[:, 0] + 15).astype(np.int32)
    ys = (height + 65 - projected[:, 1]).astype(np.int32)
    pixels = np.asarray(canvas).copy()
    pixels[ys[~covered], xs[~covered]] = (216, 89, 78)
    pixels[ys[covered], xs[covered]] = (103, 183, 125)
    canvas = Image.fromarray(pixels)
    draw = ImageDraw.Draw(canvas)
    for p in camera_centers:
        x, y = pixel(p)
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill="#3267bc")
    for rank, i in enumerate(selected, 1):
        x, y = pixel(candidates[i])
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#f6ca29", outline="black", width=2)
        draw.text((x + 8, y - 6), f"{rank} ({statistics[i]['height_above_floor']:.2f})", fill="black")
    canvas.save(path)


def publish(staged, target):
    previous = target.with_name(f".{target.name}.previous-{uuid.uuid4().hex}")
    if target.exists():
        marker = target / "selected_points.json"
        if target.is_symlink() or not marker.is_file() or json.loads(marker.read_text()).get("macro_step") != MACRO:
            raise ValueError(f"Refusing to replace unrelated output: {target}")
        target.rename(previous)
    try:
        staged.rename(target)
    except BaseException:
        if previous.exists():
            previous.rename(target)
        raise
    if previous.exists():
        shutil.rmtree(previous)


def process_scan(scan_id, transform, alignment_dir, metadata, slice_manifest, output_dir, width,
                optical_geometry, optical_environment_hashes):
    from panorama_rendering import load_gaussians, render_gaussian_panorama, render_mesh_panorama, inspect_panel_materials, panel_faces

    started = time.perf_counter()
    timings = {}
    stage = "load_scan_inputs_and_height_policy"
    tick = _start_timing(scan_id, stage)
    scan_dir = alignment_dir.parent / scan_id
    aligned_scan = alignment_dir / "scans" / scan_id
    mesh_path = aligned_scan / "mesh.ply"
    splat_path = aligned_scan / "gaussian_splat" / "gaussian_splat_sealed.ply"
    vertices, faces, colors = _read_mesh(mesh_path)
    scene = mesh_scene(vertices, faces)
    cameras, lineage = load_cameras(scan_dir, alignment_dir, transform)
    centers = np.asarray([c["camera_to_world"] for c in cameras])[:, :3, 3]
    records = {r["plane_id"]: r for r in metadata["floor_planes"]}
    groups = []
    for level in slice_manifest["levels"]:
        scan = next((s for s in level["scans"] if s["scan_id"] == scan_id), None)
        if scan and scan["output_status"] == "generated_for_floor_level":
            floors = [records[i] for i in scan["floor_plane_ids"]]
            ceilings = [r for r in metadata["ceiling_planes"] if r["scan_id"] == scan_id
                        and r["floor_level_name"] == level["floor_level_name"]]
            higher = [r["mean_elevation_m"] for r in records.values() if r["scan_id"] == scan_id
                      and r["mean_elevation_m"] > scan["floor_elevation_m"] + 1e-5]
            maximum, method = level_max_height(floors, ceilings, alignment_dir, vertices, min(higher) if higher else None)
            groups.append((level, floors, maximum, method))
    metric_path = scan_dir / "metric_scale.json"
    metric = json.loads(metric_path.read_text()) if metric_path.is_file() else {}
    policy = height_policy(metric, [g[2] for g in groups])
    policy["level_max_heights"] = [{"floor_level": g[0]["floor_level_name"], "max_height": g[2],
                                    "source": g[3]} for g in groups]
    LOGGER.info("%s: %d floors, height %.4f %s, pose radius %.4f", scan_id, len(groups),
                policy["height"], policy["units"], policy["max_pose_distance"])
    _finish_timing(timings, scan_id, stage, tick)

    stage = "prepare_floor_geometry_and_camera_assignment"
    tick = _start_timing(scan_id, stage)
    floor_geometries = [floor_geometry(g[1], alignment_dir) for g in groups]
    all_floor_scene = o3d.t.geometry.RaycastingScene()
    for v, f in floor_geometries:
        all_floor_scene.add_triangles(o3d.core.Tensor(v.astype(np.float32)), o3d.core.Tensor(f.astype(np.uint32)))
    camera_rays = np.column_stack((centers, np.tile([0, -1, 0], (len(centers), 1)))).astype(np.float32)
    camera_floors = all_floor_scene.cast_rays(o3d.core.Tensor(camera_rays))["geometry_ids"].numpy()
    _finish_timing(timings, scan_id, stage, tick)

    stage = "measure_source_surface_support"
    tick = _start_timing(scan_id, stage)
    samples, source_scores, support_report = source_surface_support(scan_dir, alignment_dir, transform, cameras)
    _finish_timing(timings, scan_id, stage, tick)

    stage = "load_render_models_and_provenance"
    tick = _start_timing(scan_id, stage)
    panel_materials = inspect_panel_materials(alignment_dir, scan_id)
    panel_manifest = alignment_dir / "stats" / "phase07_region_replacements.json"
    regions = json.loads(panel_manifest.read_text(encoding="utf-8")).get("regions", []) if panel_manifest.is_file() else []
    panel_types, _ = panel_faces(scan_id, vertices, faces, regions)
    panel_optics = None
    panel_optical_colors = None
    if np.any(panel_types):
        secondary_scene, secondary_faces, panel_optical_colors = optical_geometry
        panel_optics = (panel_types, secondary_scene, secondary_faces, max(0.001 * policy["distance_scale"], 1e-6))
    model = load_gaussians(splat_path, transform)
    provenance = {"mesh": str(mesh_path), "mesh_sha256": _hash(mesh_path),
                  "sealed_gaussians": str(splat_path), "sealed_gaussians_sha256": _hash(splat_path),
                  "pose_lineage": lineage, "panel_materials": panel_materials,
                  "optical_environment_meshes": optical_environment_hashes,
                  "metric_scale_sha256": _hash(metric_path) if metric_path.is_file() else None,
                  "element_hashes": {r["source_mesh"]: _hash(alignment_dir / r["source_mesh"])
                                     for r in metadata["floor_planes"] + metadata["ceiling_planes"]
                                     if r["scan_id"] == scan_id}}
    _finish_timing(timings, scan_id, stage, tick)
    scale = policy["distance_scale"]
    results = []
    floor_processing_tick = _start_timing(scan_id, "process_floors")
    for group_index, (level, floors, maximum, method) in enumerate(groups):
        floor_started = time.perf_counter()
        floor_timings = {}
        active_stage = None
        name = level["floor_level_name"]
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Invalid floor identity: {name}")
        parent = output_dir / scan_id
        if parent.is_symlink():
            raise ValueError(f"Output scan directory must not be a symlink: {parent}")
        parent.mkdir(parents=True, exist_ok=True)
        target = parent / name
        staged = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=parent))
        try:
            active_stage = "sample_floor_and_build_ceiling_scene"
            tick = _start_timing(f"{scan_id}/{name}", active_stage)
            floor_vertices, floor_faces = floor_geometries[group_index]
            points, grid = sample_floor(floor_vertices, floor_faces, 0.05 * scale)
            floor_camera_indices = np.flatnonzero(camera_floors == group_index)
            ceiling_vertices, ceiling_faces, offset = [], [], 0
            for ceiling in metadata["ceiling_planes"]:
                if ceiling["scan_id"] != scan_id or ceiling["floor_level_name"] != name:
                    continue
                v, f, _ = _read_mesh(alignment_dir / ceiling["source_mesh"])
                chosen = _selected_ceiling_faces(alignment_dir / ceiling["source_mesh"], v, f, ceiling)
                ceiling_vertices.append(v)
                ceiling_faces.append(f[chosen] + offset)
                offset += len(v)
            ceiling_scene = (mesh_scene(np.concatenate(ceiling_vertices), np.concatenate(ceiling_faces))
                             if ceiling_faces else None)
            if ceiling_scene is None and min(policy["allowed_heights"]) >= maximum:
                raise ValueError("All allowed camera heights exceed the sealed level's upper extent")

            _finish_timing(floor_timings, f"{scan_id}/{name}", active_stage, tick)
            active_stage = "filter_current_floor_triangles"
            tick = _start_timing(f"{scan_id}/{name}", active_stage)
            targets, target_report = slice_targets(level, scan_id, alignment_dir, floors, scale)

            _finish_timing(floor_timings, f"{scan_id}/{name}", active_stage, tick)
            active_stage = "height_search_and_triangle_visibility"
            tick = _start_timing(f"{scan_id}/{name}", active_stage)
            plan = plan_floor_points(points, grid, centers[floor_camera_indices], policy, scene, ceiling_scene,
                                     targets, samples, source_scores, cameras)
            candidates, statistics = plan["candidates"], plan["statistics"]
            candidate_report, visibility = plan["candidate_report"], plan["visibility"]
            source_quality, selected, selection = plan["source_quality"], plan["selected"], plan["selection"]
            exact_coverage = {candidate: float(row.mean()) for candidate, row in
                              zip(plan["exact_candidate_indices"], plan["exact_visibility"])}
            if plan["candidate_visibility"]["sample_triangle_count"] == len(targets):
                exact_coverage.update({i: float(row.mean()) for i, row in enumerate(visibility)})
            for statistics_row in statistics:
                statistics_row["nearest_camera_index"] = int(floor_camera_indices[statistics_row["nearest_camera_index"]])
            LOGGER.info("%s/%s: %d/%d points, triangle coverage %.1f%% (attainable %.1f%%)",
                        scan_id, name, len(selected), len(candidates), 100 * selection["coverage"],
                        100 * selection["attainable_coverage"])

            _finish_timing(floor_timings, f"{scan_id}/{name}", active_stage, tick)
            active_stage = "build_tour_links_and_coverage_plot"
            tick = _start_timing(f"{scan_id}/{name}", active_stage)
            links, components = tour_links(scene, candidates, selected, scale, grid["floor_scene"])
            coverage_plot(staged / "coverage.png", targets, plan["selected_visibility"],
                          candidates[selected], range(len(selected)), centers[floor_camera_indices],
                          [statistics[i] for i in selected], selection)
            _finish_timing(floor_timings, f"{scan_id}/{name}", active_stage, tick)

            active_stage = "render_panorama_pairs_and_contact_sheet"
            tick = _start_timing(f"{scan_id}/{name}", active_stage)
            records_out, thumbs = [], []
            for rank, i in enumerate(selected, 1):
                render_started = time.perf_counter()
                rgb, alpha = render_gaussian_panorama(model, candidates[i], width)
                optical_stats = {}
                mesh_rgb, hit = render_mesh_panorama(scene, vertices, faces, colors, candidates[i], width,
                                                      panel_optics=panel_optics, optical_stats=optical_stats,
                                                      panel_optical_colors=panel_optical_colors)
                prefix = f"panorama_{rank:03d}"
                Image.fromarray(rgb).save(staged / f"{prefix}_gaussian.jpg", quality=95, subsampling=0)
                Image.fromarray(mesh_rgb).save(staged / f"{prefix}_mesh.jpg", quality=95, subsampling=0)
                thumbnail = Image.fromarray(rgb)
                thumbnail.thumbnail((768, 384))
                thumbs.append(thumbnail.copy())
                nearest = statistics[i]["nearest_camera_index"]
                entry = {"id": rank, "position_world": candidates[i].tolist(),
                         "units": policy["units"],
                         "nearest_acquired_image_id": cameras[nearest]["image_id"],
                         "nearest_acquired_image_name": cameras[nearest]["image_name"],
                         **statistics[i], "source_view_quality": float(source_quality[i]),
                         "single_point_triangle_coverage": exact_coverage[i],
                         "gaussian_image": f"{prefix}_gaussian.jpg", "mesh_image": f"{prefix}_mesh.jpg",
                         "gaussian_alpha_mean": float(alpha.mean()),
                         "gaussian_low_alpha_fraction": float((alpha < 0.5).mean()),
                         "mesh_hit_fraction": float(hit.mean()), "mesh_optical_pixels": optical_stats}
                records_out.append(entry)
                LOGGER.info("%s/%s: rendered panorama %d/%d in %.2f s", scan_id, name, rank,
                            len(selected), time.perf_counter() - render_started)
            if thumbs:
                sheet = Image.new("RGB", (768, len(thumbs) * 408), "white")
                draw = ImageDraw.Draw(sheet)
                for rank, thumbnail in enumerate(thumbs):
                    draw.text((8, rank * 408 + 4), f"{scan_id}/{name} panorama {rank + 1}", fill="black")
                    sheet.paste(thumbnail, (0, rank * 408 + 24))
                sheet.save(staged / "contact_sheet.jpg", quality=90)
            _finish_timing(floor_timings, f"{scan_id}/{name}", active_stage, tick)

            active_stage = "write_and_publish_floor_report"
            tick = _start_timing(f"{scan_id}/{name}", active_stage)
            report = {"schema_version": 5, "macro_step": MACRO, "scan_id": scan_id,
                      "floor_level_name": name,
                      "status": "complete" if selection["minimum_reached"] else "coverage_limited",
                      "coordinate_frame": "phase07_aligned_world", "height_policy": policy,
                      "panorama": {"width": width, "height": width // 2, "projection": "equirectangular",
                                   "longitude_zero": "+X", "positive_longitude": "+Z", "up": "+Y"},
                      "floor_plane_ids": [r["plane_id"] for r in floors],
                      "floor_slice_indicator": level["sealed_mesh"], "inputs": provenance,
                      "coverage_metric": "visible_slice_triangle_count", "triangle_targets": target_report,
                      "triangle_count": len(targets),
                      "coverage": selection, "candidate_summary": candidate_report, "height_search": plan["height_search"],
                      "floor_acquired_camera_count": len(floor_camera_indices),
                      "ceiling_eligibility": ("associated_ceiling_and_clear_acquired_pose_segment" if ceiling_scene
                                               else "sealed_upper_extent_and_clear_acquired_pose_segment"),
                      "source_support": support_report, "points": records_out,
                      "tour_links": links, "tour_component_count": components,
                      "candidate_visibility": plan["candidate_visibility"],
                      "candidates": [{"position_world": c.tolist(), **s,
                                      "source_view_quality": float(source_quality[i]),
                                      "triangle_coverage": exact_coverage.get(i, float(visibility[i].mean())),
                                      "triangle_coverage_estimate": float(visibility[i].mean()),
                                      "triangle_coverage_is_estimate": i not in exact_coverage
                                          and plan["candidate_visibility"]["sample_triangle_count"] < len(targets),
                                      "exact_triangle_coverage": exact_coverage.get(i),
                                      "selected": i in selected}
                                     for i, (c, s) in enumerate(zip(candidates, statistics))],
                      "timings_seconds": dict(floor_timings),
                      "elapsed_seconds": time.perf_counter() - floor_started}
            _json(staged / "selected_points.json", report)
            publish(staged, target)
            _finish_timing(floor_timings, f"{scan_id}/{name}", active_stage, tick)
            active_stage = None
            results.append({"scan_id": scan_id, "floor_level_name": name, "status": report["status"],
                            "point_count": len(selected), "nominal_height": policy["height"],
                            "selected_heights": sorted({statistics[i]["height_above_floor"] for i in selected}),
                            "units": policy["units"],
                            "coverage": selection, "triangle_count": len(targets),
                            "report": str(target / "selected_points.json"), "output_dir": str(target),
                            "timings_seconds": dict(floor_timings),
                            "elapsed_seconds": time.perf_counter() - floor_started})
        except Exception as error:
            if active_stage is not None:
                _finish_timing(floor_timings, f"{scan_id}/{name}", active_stage, tick, "failed")
            LOGGER.exception("360 picture generation failed for %s/%s", scan_id, name)
            results.append({"scan_id": scan_id, "floor_level_name": name, "status": "failed",
                            "error": str(error), "previous_output_preserved": target.exists(),
                            "timings_seconds": dict(floor_timings),
                            "elapsed_seconds": time.perf_counter() - floor_started})
        finally:
            if staged.exists():
                shutil.rmtree(staged)
    del model
    _finish_timing(timings, scan_id, "process_floors", floor_processing_tick)
    return {"scan_id": scan_id, "floors": results, "timings_seconds": timings,
            "elapsed_seconds": time.perf_counter() - started}


def generate_360_pictures(alignment_dir=DEFAULT_ALIGNMENT_DIR, output_dir=None, width=4096):
    started = time.perf_counter()
    import torch

    alignment_dir, scene_dir, output_dir = _resolve_paths(alignment_dir, output_dir)
    if width < 256 or width % 4:
        raise ValueError("Panorama width must be a multiple of four and at least 256")
    if not torch.cuda.is_available():
        raise RuntimeError("Gaussian panorama rendering requires the configured CUDA/ROCm GPU")
    plan_dir = scene_dir / "artifacts" / "Generate Floor Plans"
    plan_manifest = json.loads((plan_dir / "generate_floor_plans_manifest.json").read_text())
    if plan_manifest["macro_step_status"] != "complete":
        raise ValueError("Generate Floor Plans must complete before generating 360 pictures")
    metadata_path = plan_dir / "floor_level_metadata.json"
    slices_path = plan_dir / "floor_level_meshes" / "floor_level_meshes_manifest.json"
    transforms_path = alignment_dir / "stats" / "aligned_scans_transforms.json"
    panel_manifest_path = alignment_dir / "stats" / "phase07_region_replacements.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("source_alignment_dir") and Path(metadata["source_alignment_dir"]).resolve() != alignment_dir:
        raise ValueError("Floor metadata belongs to another Phase 07 alignment; regenerate floor plans")
    slices = json.loads(slices_path.read_text())
    transforms = json.loads(transforms_path.read_text())["scans"]
    if not transforms:
        raise ValueError("Alignment contains no scans")
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_elapsed = round(time.perf_counter() - started, 3)
    LOGGER.info("Starting %s for %d scans at %d px", MACRO, len(transforms), width)
    scan_ids = sorted(transforms)
    for scan_id in scan_ids:
        if Path(scan_id).name != scan_id or not scan_id.startswith("scan_"):
            raise ValueError(f"Invalid scan identity: {scan_id}")
    panel_regions = json.loads(panel_manifest_path.read_text(encoding="utf-8")).get("regions", []) \
        if panel_manifest_path.is_file() else []
    has_panels = any(region.get("class") in {"mirror", "window"} and region.get("sealed_panels")
                     for region in panel_regions)
    if has_panels:
        optical_geometry, optical_hashes = optical_environment(alignment_dir, scan_ids, panel_regions)
    else:
        optical_geometry, optical_hashes = (None, np.empty((0, 3), np.uint32),
                                             np.empty((0, 3), np.uint8)), {}
    scans, failures = [], []
    source_paths = [metadata_path, slices_path, transforms_path]
    if panel_manifest_path.is_file():
        source_paths.append(panel_manifest_path)
    manifest = {"schema_version": 5, "macro_step": MACRO, "phase_status": "in_progress",
                "macro_step_status": "running", "alignment_dir": str(alignment_dir),
                "source_hashes": {str(p): _hash(p) for p in source_paths},
                "coverage_metric": "visible_slice_triangle_count",
                "coverage_target": COVERAGE_TARGET, "minimum_coverage": MINIMUM_COVERAGE,
                "panorama_width": width, "timings_seconds": {"input_validation_and_setup": setup_elapsed},
                "scans": scans, "failures": failures}
    manifest["source_hashes"].update(optical_hashes)
    _json(output_dir / "generate_360_pictures_manifest.json", manifest)
    scan_processing_started = time.perf_counter()
    for scan_id in scan_ids:
        record = transforms[scan_id]
        scan_started = time.perf_counter()
        LOGGER.info("Starting %s for %s", MACRO, scan_id)
        try:
            result = process_scan(scan_id, _rigid(record["transform_matrix"]), alignment_dir,
                                  metadata, slices, output_dir, width, optical_geometry, optical_hashes)
            scans.append(result)
            failures.extend(f for f in result["floors"] if f["status"] == "failed")
            LOGGER.info("%s: %d floors processed in %.2f s", scan_id, len(result["floors"]),
                        result["elapsed_seconds"])
        except Exception as error:
            LOGGER.exception("360 picture generation failed for %s", scan_id)
            failures.append({"scan_id": scan_id, "error": str(error), "previous_outputs_preserved": True,
                             "timings_seconds": {"process_scan": round(time.perf_counter() - scan_started, 3)},
                             "elapsed_seconds": time.perf_counter() - scan_started})
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["timings_seconds"]["scan_processing"] = round(time.perf_counter() - scan_processing_started, 3)
        temporary = output_dir / ".generate_360_pictures_manifest.json.tmp"
        _json(temporary, manifest)
        temporary.replace(output_dir / "generate_360_pictures_manifest.json")

    summary_started = time.perf_counter()
    floors = [f for scan in scans for f in scan["floors"] if f["status"] != "failed"]
    manifest["macro_step_status"] = ("failed" if failures else
                                      "complete" if floors and all(f["status"] == "complete" for f in floors) else "partial")
    manifest["point_count"] = sum(f["point_count"] for f in floors)
    manifest["elapsed_seconds"] = time.perf_counter() - started
    per_level = defaultdict(list)
    for floor in floors:
        per_level[floor["floor_level_name"]].append(floor)
    manifest["floor_level_summary"] = [{"floor_level_name": name,
        "triangle_count_coverage": sum(f["coverage"]["coverage"] * f["triangle_count"] for f in fs)
                                   / sum(f["triangle_count"] for f in fs),
        "triangle_count": sum(f["triangle_count"] for f in fs),
        "point_count": sum(f["point_count"] for f in fs)} for name, fs in per_level.items()]
    manifest["timings_seconds"]["summarize_results"] = round(time.perf_counter() - summary_started, 3)
    manifest["elapsed_seconds"] = time.perf_counter() - started
    _json(temporary, manifest)
    temporary.replace(output_dir / "generate_360_pictures_manifest.json")
    LOGGER.info("Completed %s: %s, %d panoramas at %d scan-floor pairs in %.2f s", MACRO,
                manifest["macro_step_status"], manifest["point_count"], len(floors),
                time.perf_counter() - started)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-dir", type=Path, default=DEFAULT_ALIGNMENT_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--width", type=int, default=4096, help="Equirectangular width; height is half this value")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _, _, output_dir = _resolve_paths(args.alignment_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / "generate_360_pictures.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    try:
        result = generate_360_pictures(args.alignment_dir, output_dir, args.width)
    except Exception:
        LOGGER.exception("%s failed", MACRO)
        raise
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()
    return 0 if result["macro_step_status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
