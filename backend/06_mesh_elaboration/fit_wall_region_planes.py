#!/usr/bin/env python3
"""Fit one robust plane to each non-black cluster from grow_wall_regions.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from trimesh.visual.material import PBRMaterial

MAX_OUTLIER_FRACTION = 0.01
PARALLEL_MERGE_DISTANCE = 0.15
MAX_SNAP_ROTATION_DEGREES = 30.0
FIT_ITERATIONS = 12
TRIM_ITERATIONS = 4


def _weighted_plane(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float]:
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Cannot fit a plane to zero-area triangles")
    center = np.average(points, axis=0, weights=weights)
    centered = points - center
    covariance = (centered * weights[:, None]).T @ centered / total
    normal = np.linalg.eigh(covariance)[1][:, 0]
    normal /= np.linalg.norm(normal)
    return normal, -float(normal @ center)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    index = np.searchsorted(np.cumsum(weights[order]), weights.sum() * 0.5, side="left")
    return float(values[order[min(index, len(order) - 1)]])


def _fit_l1_plane(points: np.ndarray, weights: np.ndarray,
                  initial_normal: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """IRLS approximation to the area-weighted sum of absolute point-plane distances."""
    if initial_normal is None or np.linalg.norm(initial_normal) <= 1e-12:
        normal, offset = _weighted_plane(points, weights)
    else:
        normal = initial_normal / np.linalg.norm(initial_normal)
        offset = -_weighted_median(points @ normal, weights)
    for _ in range(FIT_ITERATIONS):
        residuals = np.abs(points @ normal + offset)
        epsilon = max(float(np.median(residuals)) * 1e-3, 1e-7)
        robust_weights = weights / np.maximum(residuals, epsilon)
        next_normal, next_offset = _weighted_plane(points, robust_weights)
        if abs(float(normal @ next_normal)) > 1.0 - 1e-10 and abs(offset - next_offset) < 1e-8:
            normal, offset = next_normal, next_offset
            break
        normal, offset = next_normal, next_offset
    return normal, offset


def _region_groups(labels: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    order = np.argsort(labels, kind="stable")
    region_ids, starts = np.unique(labels[order], return_index=True)
    return region_ids, list(np.split(order, starts[1:]))


def _fit_regions(
    vertices: np.ndarray,
    triangles: np.ndarray,
    labels: np.ndarray,
    regions: list[dict],
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    if len(triangles) != len(labels):
        raise ValueError(f"Face-label count ({len(labels)}) does not match PLY triangle count ({len(triangles)})")
    region_ids, groups = _region_groups(labels)
    region_by_id = {int(region["region_id"]): region for region in regions}
    if set(map(int, region_ids)) != set(region_by_id):
        raise ValueError("Region manifest IDs do not match the triangle-label sidecar")
    black_ids = {region_id for region_id, region in region_by_id.items()
                 if region.get("painted_black", region.get("color_rgb") == [0, 0, 0])}
    fit_groups = [(int(region_id), face_ids) for region_id, face_ids in zip(region_ids, groups)
                  if int(region_id) not in black_ids]
    black_mask = np.isin(labels, list(black_ids))

    v0 = vertices[triangles[:, 0]]
    edge_a = vertices[triangles[:, 1]] - v0
    edge_b = vertices[triangles[:, 2]] - v0
    areas = np.linalg.norm(np.cross(edge_a, edge_b), axis=1) * 0.5
    inliers = ~black_mask
    outlier_budget = int(len(triangles) * MAX_OUTLIER_FRACTION)
    fits: dict[int, tuple[np.ndarray, float]] = {}

    def refit() -> None:
        fits.clear()
        for region_id, face_ids in fit_groups:
            kept = face_ids[inliers[face_ids]]
            if not len(kept):
                raise ValueError(f"Region {int(region_id)} has no inlier triangles")
            nondegenerate = kept[areas[kept] > 1e-12]
            if len(nondegenerate):
                points = vertices[triangles[nondegenerate]].reshape(-1, 3)
                weights = np.repeat(areas[nondegenerate] / 3.0, 3)
                face_normals = np.cross(
                    vertices[triangles[nondegenerate, 1]] - vertices[triangles[nondegenerate, 0]],
                    vertices[triangles[nondegenerate, 2]] - vertices[triangles[nondegenerate, 0]],
                ) / (2.0 * areas[nondegenerate, None])
                reference = face_normals[0]
                face_normals[np.sum(face_normals * reference, axis=1) < 0] *= -1.0
                initial_normal = np.average(face_normals, axis=0, weights=areas[nondegenerate])
            else:
                # Degenerate geometry has no area or unique normal; its vertices still define a best-effort plane.
                points = vertices[triangles[kept]].reshape(-1, 3)
                weights = np.ones(len(points), dtype=np.float64)
                initial_normal = None
            fits[int(region_id)] = _fit_l1_plane(points, weights, initial_normal)

    def triangle_scores() -> np.ndarray:
        scores = np.zeros(len(triangles), dtype=np.float64)
        for region_id, face_ids in fit_groups:
            normal, offset = fits[int(region_id)]
            points = vertices[triangles[face_ids]]
            scores[face_ids] = np.abs(points @ normal + offset).mean(axis=1)
        return scores

    for _ in range(TRIM_ITERATIONS):
        refit()
        if not outlier_budget:
            break
        scores = triangle_scores()
        remaining = {region_id: len(face_ids) for region_id, face_ids in fit_groups}
        candidates = np.flatnonzero(~black_mask)
        candidates = candidates[np.lexsort((candidates, -scores[candidates]))]
        new_inliers = ~black_mask.copy()
        removed = 0
        for face_id in candidates:
            region_id = int(labels[face_id])
            if removed == outlier_budget:
                break
            if remaining[region_id] > 1:
                new_inliers[face_id] = False
                remaining[region_id] -= 1
                removed += 1
        if np.array_equal(inliers, new_inliers):
            break
        inliers = new_inliers

    refit()
    scores = triangle_scores()
    fitted = []
    for region_id, face_ids in fit_groups:
        normal, offset = fits[region_id]
        keep_ids = face_ids[inliers[face_ids]]
        area = areas[keep_ids]
        positive_area = area > 1e-12
        fit_scores = scores[keep_ids][positive_area]
        fit_area = area[positive_area]
        mean_distance = (
            float(np.average(fit_scores, weights=fit_area))
            if fit_area.sum() > 0 else None
        )
        fitted.append({
            **region_by_id[region_id],
            "normal": normal.tolist(),
            "offset": offset,
            "triangle_count": int(len(face_ids)),
            "inlier_triangle_count": int(len(keep_ids)),
            "outlier_triangle_count": int(len(face_ids) - len(keep_ids)),
            "fit_area_scene_units_squared": float(fit_area.sum()),
            "area_weighted_mean_absolute_distance_scene_units": mean_distance,
            "degenerate_support_only": not bool(positive_area.any()),
            "area_weighted_total_absolute_distance_scene_units_cubed": float(np.sum(fit_scores * fit_area)),
        })
    return fitted, inliers, black_mask


def _plane_preview(vertices: np.ndarray, triangles: np.ndarray, face_ids: np.ndarray,
                   normal: np.ndarray, offset: float, color_rgb: list[int], name: str) -> trimesh.Trimesh:
    reference = np.eye(3)[int(np.argmin(np.abs(normal)))]
    axis_u = np.cross(reference, normal)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    origin = -offset * normal
    points = vertices[triangles[face_ids]].reshape(-1, 3)
    projected = points - origin
    uv = np.column_stack((projected @ axis_u, projected @ axis_v))
    # shortcut: GLB stores finite meshes; exact infinite planes remain in the manifest.
    pad = 0.02 * max(float(np.ptp(uv[:, 0])), float(np.ptp(uv[:, 1])), 1e-3)
    low, high = uv.min(axis=0) - pad, uv.max(axis=0) + pad
    plane_vertices = np.asarray([
        origin + low[0] * axis_u + low[1] * axis_v,
        origin + high[0] * axis_u + low[1] * axis_v,
        origin + high[0] * axis_u + high[1] * axis_v,
        origin + low[0] * axis_u + high[1] * axis_v,
    ])
    plane = trimesh.Trimesh(
        vertices=plane_vertices,
        faces=np.asarray([[0, 1, 2], [0, 2, 3]]),
        process=False,
    )
    plane.visual.material = PBRMaterial(
        name=name,
        baseColorFactor=[*map(int, color_rgb), 128],
        metallicFactor=0.0,
        roughnessFactor=1.0,
        alphaMode="BLEND",
        doubleSided=True,
    )
    return plane


def _enclose_plane(
    vertices: np.ndarray,
    triangles: np.ndarray,
    face_ids: np.ndarray,
    areas: np.ndarray,
    normal: np.ndarray,
    initial_offset: float,
    mesh_centroids: np.ndarray,
    mesh_areas: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, str, float, float]:
    triangle_points = vertices[triangles[face_ids]]
    projections = triangle_points @ normal
    outside_budget = int(len(face_ids) * MAX_OUTLIER_FRACTION)
    high = float(np.partition(projections.max(axis=1), len(face_ids) - outside_budget - 1)[
        len(face_ids) - outside_budget - 1
    ])
    low = float(np.partition(projections.min(axis=1), outside_budget)[outside_budget])
    points = triangle_points.reshape(-1, 3)
    point_weights = np.repeat(areas[face_ids] / 3.0, 3)
    mesh_projections = mesh_centroids @ normal
    total_mesh_area = float(mesh_areas.sum())
    candidates = []
    for boundary, outside, side in (
        (high, projections.max(axis=1) > high, "positive_normal"),
        (low, projections.min(axis=1) < low, "negative_normal"),
    ):
        offset = -boundary
        distance_cost = float(np.sum(np.abs(points @ normal + offset) * point_weights))
        mesh_inside = mesh_projections <= boundary if side == "positive_normal" else mesh_projections >= boundary
        inside_area = float(mesh_areas[mesh_inside].sum())
        inside_fraction = (
            inside_area / total_mesh_area if total_mesh_area > 1e-12
            else float(mesh_inside.mean())
        )
        candidates.append((inside_fraction, distance_cost, abs(offset - initial_offset),
                           offset, outside, side))
    inside_fraction, cost, _, offset, outside, side = max(
        candidates, key=lambda item: (item[0], -item[1], -item[2])
    )
    return offset, face_ids[~outside], face_ids[outside], side, cost, inside_fraction


def _merge_coordinate_anchor(group: list[dict], family_planes: list[dict]) -> dict | None:
    """Pick the group member with the most clearance from its nearest external plane."""
    group_ids = {int(plane["region_id"]) for plane in group}
    external = [plane for plane in family_planes if int(plane["region_id"]) not in group_ids]
    if not external:
        return None
    candidates = []
    for member in group:
        neighbor = min(external, key=lambda plane: (
            abs(float(member["offset"]) - float(plane["offset"])),
            int(plane["region_id"]),
        ))
        clearance = abs(float(member["offset"]) - float(neighbor["offset"]))
        candidates.append((clearance, float(member["area"]), -int(member["region_id"]),
                           member, neighbor))
    clearance, _, _, member, neighbor = max(candidates, key=lambda item: item[:3])
    return {
        "region_id": int(member["region_id"]),
        "offset": float(member["offset"]),
        "nearest_external_region_id": int(neighbor["region_id"]),
        "clearance_scene_units": float(clearance),
    }


def _snap_and_merge_planes(
    vertices: np.ndarray,
    triangles: np.ndarray,
    labels: np.ndarray,
    inliers: np.ndarray,
    fitted: list[dict],
) -> tuple[list[dict], list[dict], dict]:
    """Snap wall normals to two orthogonal horizontal directions and merge close offsets."""
    if not fitted:
        return [], [], {
            "horizontal_direction_yaw_radians": None,
            "orthogonal_horizontal_normal_directions": [],
            "parallel_merge_distance_scene_units": PARALLEL_MERGE_DISTANCE,
            "max_snap_rotation_degrees": MAX_SNAP_ROTATION_DEGREES,
            "excluded_before_snap": [],
            "merge_rejections": [],
        }
    normals = np.asarray([fit["normal"] for fit in fitted], dtype=np.float64)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1e-8):
        raise ValueError("Cannot snap a plane with a zero normal")
    normals /= lengths[:, None]
    horizontal = normals[:, (0, 2)]
    horizontal_lengths = np.linalg.norm(horizontal, axis=1)
    wall_mask = horizontal_lengths >= np.abs(normals[:, 1])
    weights = np.asarray([max(float(fit["fit_area_scene_units_squared"]), 1e-12) for fit in fitted])
    if wall_mask.any():
        wall_horizontal = horizontal[wall_mask] / horizontal_lengths[wall_mask, None]
        angles = np.arctan2(wall_horizontal[:, 1], wall_horizontal[:, 0])
        # Fourfold symmetry makes opposite and perpendicular wall normals equivalent.
        orientation = np.sum(weights[wall_mask] * np.exp(4j * angles))
        if abs(orientation) <= 1e-12:
            reference_index = int(np.argmax(weights[wall_mask]))
            yaw = float((angles[reference_index] + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0)
        else:
            yaw = float(np.angle(orientation) / 4.0)
    else:
        yaw = 0.0
    axes = np.asarray([
        [np.cos(yaw), 0.0, np.sin(yaw)],
        [-np.sin(yaw), 0.0, np.cos(yaw)],
        [0.0, 1.0, 0.0],
    ])

    v0 = vertices[triangles[:, 0]]
    areas = np.linalg.norm(
        np.cross(vertices[triangles[:, 1]] - v0, vertices[triangles[:, 2]] - v0), axis=1
    ) * 0.5
    mesh_centroids = vertices[triangles].mean(axis=1)
    snapped = []
    excluded = []
    for fit, direction in zip(fitted, normals):
        family = int(np.argmax(np.abs(axes @ direction)))
        rotation_degrees = float(np.degrees(np.arccos(np.clip(
            abs(float(axes[family] @ direction)), 0.0, 1.0
        ))))
        if rotation_degrees > MAX_SNAP_ROTATION_DEGREES:
            excluded.append({
                "region_id": int(fit["region_id"]),
                "rotation_degrees": rotation_degrees,
            })
            continue
        normal = axes[family]
        face_ids = np.flatnonzero((labels == int(fit["region_id"])) & inliers)
        positive_area = face_ids[areas[face_ids] > 1e-12]
        if len(positive_area):
            points = vertices[triangles[positive_area]].reshape(-1, 3)
            point_weights = np.repeat(areas[positive_area] / 3.0, 3)
            offset = -_weighted_median(points @ normal, point_weights)
        else:
            points = vertices[triangles[face_ids]].reshape(-1, 3)
            offset = -float(np.median(points @ normal))
        snapped.append({
            "region_id": int(fit["region_id"]),
            "normal": normal.tolist(),
            "offset": offset,
            "family": family,
            "face_ids": face_ids,
            "area": float(areas[positive_area].sum()),
            "color_rgb": fit["color_rgb"],
            "source_normals": [fit["normal"]],
            "rotation_degrees": rotation_degrees,
        })

    merged = []
    merge_rejections = []
    for family in range(len(axes)):
        family_planes = sorted((plane for plane in snapped if plane["family"] == family),
                               key=lambda plane: plane["offset"])
        groups = []
        for plane in family_planes:
            if not groups or plane["offset"] - groups[-1][0]["offset"] >= PARALLEL_MERGE_DISTANCE:
                groups.append([plane])
            else:
                groups[-1].append(plane)

        def group_support(group: list[dict]) -> tuple[np.ndarray, np.ndarray, float]:
            face_ids = np.concatenate([plane["face_ids"] for plane in group])
            positive_area = face_ids[areas[face_ids] > 1e-12]
            normal = axes[family]
            if len(positive_area):
                points = vertices[triangles[positive_area]].reshape(-1, 3)
                point_weights = np.repeat(areas[positive_area] / 3.0, 3)
                offset = -_weighted_median(points @ normal, point_weights)
            else:
                points = vertices[triangles[face_ids]].reshape(-1, 3)
                offset = -float(np.median(points @ normal))
            return face_ids, normal, offset

        safe_groups = []
        for group in groups:
            if len(group) == 1:
                safe_groups.append((group, None))
                continue
            _, _, support_offset = group_support(group)
            anchor = _merge_coordinate_anchor(group, family_planes)
            proposed_offset = anchor["offset"] if anchor else support_offset
            group_ids = {plane["region_id"] for plane in group}
            collision = next((other for other in family_planes
                              if other["region_id"] not in group_ids
                              and abs(other["offset"] - proposed_offset) < PARALLEL_MERGE_DISTANCE
                              and any(abs(other["offset"] - member["offset"]) > PARALLEL_MERGE_DISTANCE
                                      for member in group)), None)
            if collision is None:
                safe_groups.append((group, anchor))
            else:
                merge_rejections.append({
                    "source_region_ids": sorted(group_ids),
                    "nearby_region_id": int(collision["region_id"]),
                    "proposed_offset": proposed_offset,
                    "proposed_distance_scene_units": abs(collision["offset"] - proposed_offset),
                    "previous_distances_scene_units": [
                        {"region_id": int(member["region_id"]),
                         "distance": abs(collision["offset"] - member["offset"])}
                        for member in sorted(group, key=lambda item: item["region_id"])
                    ],
                })
                safe_groups.extend(([plane], None) for plane in group)

        for group, anchor in safe_groups:
            _, normal, support_offset = group_support(group)
            initial_offset = anchor["offset"] if anchor else support_offset
            region_ids = sorted(plane["region_id"] for plane in group)
            support_face_ids = np.flatnonzero(np.isin(labels, region_ids))
            (
                enclosure_offset, face_ids, outside_face_ids, enclosure_side,
                distance_cost, enclosure_inside_fraction,
            ) = _enclose_plane(
                vertices, triangles, support_face_ids, areas, normal, initial_offset,
                mesh_centroids, areas,
            )
            # A nearby parallel plane defines which source plane must survive the merge;
            # whole-mesh enclosure may otherwise pull it onto that neighboring plane.
            offset = anchor["offset"] if anchor else enclosure_offset
            outside_side, mesh_inside_fraction = enclosure_side, enclosure_inside_fraction
            if anchor:
                mesh_projections = mesh_centroids @ normal
                boundary = -offset
                inside_candidates = [
                    (float(areas[mesh_projections <= boundary].sum()), "positive_normal"),
                    (float(areas[mesh_projections >= boundary].sum()), "negative_normal"),
                ]
                inside_area, outside_side = max(
                    inside_candidates,
                    key=lambda item: (item[0], item[1] == enclosure_side),
                )
                mesh_inside_fraction = (
                    inside_area / float(areas.sum()) if areas.sum() > 1e-12
                    else inside_area / max(len(mesh_projections), 1)
                )
                support_points = vertices[triangles[support_face_ids]].reshape(-1, 3)
                support_weights = np.repeat(areas[support_face_ids] / 3.0, 3)
                distance_cost = float(np.sum(
                    np.abs(support_points @ normal + offset) * support_weights
                ))
            support_positive_area = support_face_ids[areas[support_face_ids] > 1e-12]
            support_points = vertices[triangles[support_positive_area]].reshape(-1, 3)
            point_area = np.repeat(areas[support_positive_area] / 3.0, 3)
            distances = np.abs(support_points @ normal + offset)
            fit_area = float(areas[support_positive_area].sum())
            mean_distance = float(np.average(distances, weights=point_area)) if point_area.sum() else None
            anchor_plane = next((plane for plane in group
                                 if anchor and int(plane["region_id"]) == anchor["region_id"]), None)
            primary = anchor_plane or max(group, key=lambda plane: plane["area"])
            merged.append({
                "region_id": int(primary["region_id"]),
                "source_region_ids": region_ids,
                "normal": normal.tolist(),
                "offset": offset,
                "orientation_family": family,
                "triangle_count": int(len(support_face_ids)),
                "enclosed_triangle_count": int(len(face_ids)),
                "outside_triangle_count": int(len(outside_face_ids)),
                "outside_fraction": len(outside_face_ids) / len(support_face_ids),
                "enclosure_candidate_enclosed_triangle_count": int(len(face_ids)),
                "enclosure_candidate_outside_triangle_count": int(len(outside_face_ids)),
                "enclosure_candidate_outside_fraction": len(outside_face_ids) / len(support_face_ids),
                "enclosing_side": outside_side,
                "whole_mesh_interior_area_fraction": mesh_inside_fraction,
                "offset_before_enclosing": initial_offset,
                "enclosure_candidate_offset_scene_units": enclosure_offset,
                "enclosure_candidate_side": enclosure_side,
                "enclosure_candidate_interior_area_fraction": enclosure_inside_fraction,
                "enclosure_translation_scene_units": offset - initial_offset,
                "merge_coordinate_anchor": anchor,
                "fit_area_scene_units_squared": fit_area,
                "area_weighted_mean_absolute_distance_scene_units": mean_distance,
                "area_weighted_total_absolute_distance_scene_units_cubed": distance_cost,
                "color_rgb": primary["color_rgb"],
                "source_normals": [normal for plane in group for normal in plane["source_normals"]],
                "merged_from": [
                    {"region_id": plane["region_id"], "normal": plane["normal"], "offset": plane["offset"]}
                    for plane in group
                ],
                "face_ids": face_ids,
            })
    return snapped, merged, {
        "horizontal_direction_yaw_radians": yaw,
        "orthogonal_horizontal_normal_directions": axes.tolist(),
        "parallel_merge_distance_scene_units": PARALLEL_MERGE_DISTANCE,
        "max_snap_rotation_degrees": MAX_SNAP_ROTATION_DEGREES,
        "excluded_before_snap": excluded,
        "merge_rejections": merge_rejections,
        "merge_coordinate_rule": (
            "For a merged group with external planes in the same snapped parallel family, use "
            "the member with greatest distance to its nearest such plane as the final coordinate; "
            "otherwise use whole-mesh enclosure. The enclosure candidate is retained in diagnostics."
        ),
    }


def _process(regions_dir: Path) -> tuple[Path, Path, dict]:
    mesh_path = regions_dir / "wall_regions_colored.ply"
    manifest_path = regions_dir / "wall_regions_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    label_name = manifest.get("triangle_labels_file", "wall_regions_triangle_labels.npy")
    label_path = regions_dir / label_name
    if not mesh_path.is_file() or not label_path.is_file():
        raise ValueError(f"Expected colored PLY and triangle-label sidecar in {regions_dir}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    labels = np.load(label_path)
    if not len(triangles) or not len(vertices):
        raise ValueError(f"Colored region mesh is empty: {mesh_path}")
    fitted, inliers, black_mask = _fit_regions(
        vertices, triangles, labels, manifest.get("regions", [])
    )
    snapped, final_planes, snap_info = _snap_and_merge_planes(
        vertices, triangles, labels, inliers, fitted
    )

    scene = trimesh.Scene()
    combined_parts = [trimesh.load(mesh_path, force="mesh", process=False)]
    for fit in final_planes:
        face_ids = fit["face_ids"]
        region_ids = fit["source_region_ids"]
        name = (f"wall_region_{region_ids[0]:03d}" if len(region_ids) == 1 else
                "wall_regions_" + "_".join(f"{region_id:03d}" for region_id in region_ids))
        plane = _plane_preview(
            vertices, triangles, face_ids, np.asarray(fit["normal"]), fit["offset"],
            fit["color_rgb"], name,
        )
        scene.add_geometry(plane, geom_name=name)
        ply_plane = plane.copy()
        ply_plane.visual = trimesh.visual.ColorVisuals(
            ply_plane, vertex_colors=np.tile([*fit["color_rgb"], 128], (4, 1))
        )
        combined_parts.append(ply_plane)

    output_glb = regions_dir / "wall_region_planes.glb"
    output_ply = regions_dir / "wall_regions_with_planes.ply"
    output_manifest = regions_dir / "wall_region_planes_manifest.json"
    if final_planes:
        output_glb.write_bytes(scene.export(file_type="glb"))
    trimesh.util.concatenate(combined_parts).export(output_ply)
    outlier_count = int((~black_mask & ~inliers).sum())
    black_count = int(black_mask.sum())
    black_region_ids = set(map(int, np.unique(labels[black_mask])))
    snap_excluded_ids = {int(item["region_id"]) for item in snap_info["excluded_before_snap"]}
    snap_excluded_mask = (
        np.isin(labels, list(snap_excluded_ids)) if snap_excluded_ids
        else np.zeros_like(labels, dtype=bool)
    )
    unassigned_mask = ~inliers | snap_excluded_mask
    enclosure_checked_count = int(sum(plane["triangle_count"] for plane in final_planes))
    enclosure_outside_count = int(sum(plane["outside_triangle_count"] for plane in final_planes))
    enclosure_outside_budget = int(len(triangles) * MAX_OUTLIER_FRACTION)
    if enclosure_outside_count > enclosure_outside_budget:
        raise ValueError(
            f"Enclosing planes leave {enclosure_outside_count} triangles outside; "
            f"the global budget is {enclosure_outside_budget}"
        )
    result = {
        "algorithm": "globally_trimmed_l1_regions_angle_filtered_snapped_and_merged",
        "input_mesh": str(mesh_path),
        "input_triangle_labels": str(label_path),
        "triangle_count": int(len(triangles)),
        "fit_triangle_count": int((~black_mask).sum()),
        "max_outlier_fraction": MAX_OUTLIER_FRACTION,
        "max_enclosure_outside_fraction_of_original_mesh": MAX_OUTLIER_FRACTION,
        "enclosure_checked_triangle_count": enclosure_checked_count,
        "enclosure_outside_triangle_count": enclosure_outside_count,
        "enclosure_outside_fraction_of_original_mesh": enclosure_outside_count / len(triangles),
        "enclosure_outside_triangle_budget": enclosure_outside_budget,
        "outlier_triangle_count": outlier_count,
        "outlier_fraction": outlier_count / len(triangles),
        "black_region_triangle_count": black_count,
        "black_region_fraction": black_count / len(triangles),
        "snap_excluded_region_count": len(snap_excluded_ids),
        "snap_excluded_triangle_count": int(snap_excluded_mask.sum()),
        "unassigned_triangle_count": int(unassigned_mask.sum()),
        "unassigned_fraction": float(unassigned_mask.mean()),
        "excluded_black_regions": [
            {
                "region_id": int(region["region_id"]),
                "triangle_count": int(np.count_nonzero(labels == int(region["region_id"]))),
            }
            for region in manifest["regions"] if int(region["region_id"]) in black_region_ids
        ],
        "distance_objective": "area-weighted sum of absolute orthogonal distances from non-black triangle vertices to their cluster plane; IRLS approximation",
        "preview_note": "GLB and combined PLY contain finite plane previews over inlier bounds; opacity is 0.5. Exact infinite plane equations are in normal and offset.",
        **snap_info,
        "plane_count": len(final_planes),
        "area_weighted_total_absolute_distance_scene_units_cubed": float(sum(
            fit["area_weighted_total_absolute_distance_scene_units_cubed"] for fit in final_planes
        )),
        "fitted_planes_before_snap": fitted,
        "snapped_planes_before_merge": [
            {key: value for key, value in fit.items() if key != "face_ids"} for fit in snapped
        ],
        "planes": [{key: value for key, value in fit.items() if key != "face_ids"}
                   for fit in final_planes],
        "plane_mesh": str(output_glb) if final_planes else None,
        "combined_ply": str(output_ply),
    }
    output_manifest.write_text(json.dumps(result, indent=2) + "\n")
    return output_glb, output_manifest, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions-dir", type=Path, required=True,
                        help="Directory containing wall_regions_colored.ply and its manifest/labels")
    args = parser.parse_args()
    try:
        output_glb, output_manifest, result = _process(args.regions_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Fitted {result['plane_count']} planes; enclosure outside: "
          f"{result['enclosure_outside_triangle_count']} / {result['triangle_count']} original triangles "
          f"({result['enclosure_checked_triangle_count']} assigned; "
          f"{result['enclosure_outside_fraction_of_original_mesh']:.2%} outside, "
          f"budget {result['enclosure_outside_triangle_budget']})")
    print(f"Excluded {result['black_region_triangle_count']} black triangles; trimmed "
          f"{result['outlier_triangle_count']} / {result['triangle_count']} during fit "
          f"({result['outlier_fraction']:.2%})")
    print(f"Excluded {result['snap_excluded_region_count']} regions before snapping "
          f"({result['snap_excluded_triangle_count']} triangles; "
          f"rotation limit {result['max_snap_rotation_degrees']:.0f} degrees)")
    if output_glb.is_file():
        print(f"Planes:   {output_glb}")
    print(f"Combined PLY: {result['combined_ply']}")
    print(f"Manifest: {output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
