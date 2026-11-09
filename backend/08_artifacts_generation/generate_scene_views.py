#!/usr/bin/env python3
"""Render mesh-aligned views and optimize two wall-aligned vertical cut planes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from Utilities.pipeline_paths import bootstrap

bootstrap()
from scene_alignment import _wall_yaw

IMAGE_WIDTH, IMAGE_HEIGHT = 1024, 768
VERTICAL_FOV_DEG = 50.0
PLANE_BALANCE_WEIGHT = 0.2
PLANE_PROXIMITY_WEIGHT = 0.35
PLANE_SUPPORT_SIGMA_FRACTION = 0.02
PLANE_BALANCE_SIGMA_FRACTION = 0.12
PLANE_SURFACE_SIGMA_FRACTION = 0.04


def mesh_face_geometry(vertices: np.ndarray, triangles: np.ndarray):
    vertices = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(triangles, dtype=np.int32)
    p0 = vertices[triangles[:, 0]]
    edge1 = vertices[triangles[:, 1]] - p0
    edge2 = vertices[triangles[:, 2]] - p0
    crosses = np.cross(edge1, edge2)
    double_areas = np.linalg.norm(crosses, axis=1)
    valid = np.isfinite(crosses).all(axis=1) & np.isfinite(double_areas) & (double_areas > 1e-12)
    normals = np.zeros_like(crosses)
    normals[valid] = crosses[valid] / double_areas[valid, None]
    centers = (p0 + vertices[triangles[:, 1]] + vertices[triangles[:, 2]]) / 3.0
    areas = double_areas * 0.5
    return centers, normals, areas


def wall_axes(vertices: np.ndarray, triangles: np.ndarray, face_data=None) -> tuple[np.ndarray, np.ndarray]:
    """Find the two dominant horizontal wall-normal directions in the mesh."""
    if face_data is None:
        _, normals, areas = mesh_face_geometry(vertices, triangles)
    else:
        _, normals, areas = face_data
    valid = np.isfinite(normals).all(axis=1) & np.isfinite(areas) & (areas > 1e-12)
    if not valid.any():
        raise ValueError("Sealed mesh has no nondegenerate faces")
    normals = normals[valid]
    areas = areas[valid]
    if not np.any(np.linalg.norm(normals[:, [0, 2]], axis=1) > 0.85):
        raise ValueError("Sealed mesh has no supported vertical wall faces")
    angle = _wall_yaw(normals, areas)
    axis = np.array([math.cos(angle), math.sin(angle)])
    return axis, np.array([-axis[1], axis[0]])


def camera_views(vertices: np.ndarray, triangles: np.ndarray,
                 extra_points: np.ndarray | None = None,
                 axes: tuple[np.ndarray, np.ndarray] | None = None) -> list[dict]:
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("Sealed mesh has invalid vertex positions")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
        raise ValueError("Sealed mesh has no triangles")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError("Sealed mesh has invalid triangle indices")

    axes = wall_axes(vertices, triangles) if axes is None else axes
    basis = np.stack(axes, axis=1)
    projected = vertices[:, [0, 2]] @ basis
    horizontal_center = (projected.min(axis=0) + projected.max(axis=0)) * 0.5
    horizontal_target = basis @ horizontal_center
    target = np.array([horizontal_target[0], np.median(vertices[:, 1]), horizontal_target[1]])

    points = vertices
    if extra_points is not None and len(extra_points):
        extra_points = np.asarray(extra_points, dtype=np.float64)
        if extra_points.ndim != 2 or extra_points.shape[1] != 3 or not np.isfinite(extra_points).all():
            raise ValueError("Sealed Gaussian model has invalid means")
        points = np.concatenate((vertices, extra_points), axis=0)

    vertical_fov = math.radians(VERTICAL_FOV_DEG)
    tan_v = math.tan(vertical_fov / 2)
    tan_h = tan_v * IMAGE_WIDTH / IMAGE_HEIGHT
    focal = IMAGE_HEIGHT / (2 * tan_v)
    intrinsic = np.array([
        [focal, 0.0, IMAGE_WIDTH / 2],
        [0.0, focal, IMAGE_HEIGHT / 2],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    views = []
    for axis_index, horizontal in enumerate(axes, 1):
        for sign, suffix in ((1.0, "positive"), (-1.0, "negative")):
            side = np.array([horizontal[0], 0.0, horizontal[1]]) * sign
            forward = -side
            right = np.cross([0.0, 1.0, 0.0], forward)
            right /= np.linalg.norm(right)
            up = np.cross(forward, right)
            relative = points - target
            x = relative @ right
            y = relative @ up
            z_offset = relative @ forward
            distance = max(
                float(np.max(np.abs(x) / tan_h - z_offset)),
                float(np.max(np.abs(y) / tan_v - z_offset)),
                float(-z_offset.min() + 0.1),
            ) * 1.08
            eye = target + side * distance
            rotation = np.stack((right, -up, forward)).astype(np.float32)
            viewmat = np.eye(4, dtype=np.float32)
            viewmat[:3, :3] = rotation
            viewmat[:3, 3] = -rotation @ eye.astype(np.float32)
            views.append({
                "name": f"main_{axis_index}_{suffix}",
                "viewmat": viewmat,
                "K": intrinsic,
            })
    return views


def _smooth_profile(values: np.ndarray, sigma_bins: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3.0 * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def estimate_floor_height(face_centers: np.ndarray, face_normals: np.ndarray,
                          face_areas: np.ndarray, vertices: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimate the dominant low floor and room height from horizontal mesh faces."""
    y = face_centers[:, 1]
    horizontal = np.abs(face_normals[:, 1]) >= 0.85
    if not horizontal.any():
        raise ValueError("Cannot score vertical cuts: mesh has no horizontal floor or ceiling faces")

    robust_min, robust_max = np.percentile(vertices[:, 1], [1.0, 99.0])
    vertical_span = max(float(robust_max - robust_min), 0.5)
    floor_step = max(0.02, vertical_span * 0.01)
    floor_search_max = robust_min + max(0.6, min(1.2, vertical_span * 0.3))
    floor_edges = np.arange(robust_min, floor_search_max + floor_step, floor_step)
    if len(floor_edges) < 2:
        floor_edges = np.array([robust_min, robust_min + floor_step])
    low_horizontal = horizontal & (y <= floor_edges[-1])
    floor_hist, floor_edges = np.histogram(
        y[low_horizontal], bins=floor_edges,
        weights=face_areas[low_horizontal] * np.abs(face_normals[low_horizontal, 1]),
    )
    floor_bin = int(np.argmax(floor_hist))
    floor_y = float((floor_edges[floor_bin] + floor_edges[floor_bin + 1]) * 0.5)
    floor_band = max(2.0 * floor_step, vertical_span * 0.025)
    floor_faces = horizontal & (np.abs(y - floor_y) <= floor_band)
    if not floor_faces.any():
        raise ValueError("Cannot score vertical cuts: dominant floor band has no mesh triangles")

    ceiling_min = floor_y + max(0.8, vertical_span * 0.3)
    upper = horizontal & (y >= ceiling_min)
    if upper.any():
        ceiling_step = max(0.04, vertical_span * 0.02)
        ceiling_edges = np.arange(float(y[upper].min()), float(y[upper].max()) + ceiling_step, ceiling_step)
        if len(ceiling_edges) >= 2:
            ceiling_hist, ceiling_edges = np.histogram(
                y[upper], bins=ceiling_edges,
                weights=face_areas[upper] * np.abs(face_normals[upper, 1]),
            )
            ceiling_bin = int(np.argmax(ceiling_hist))
            ceiling_y = float((ceiling_edges[ceiling_bin] + ceiling_edges[ceiling_bin + 1]) * 0.5)
        else:
            ceiling_y = float(np.percentile(vertices[:, 1], 95.0))
    else:
        ceiling_y = float(np.percentile(vertices[:, 1], 95.0))
    room_height = ceiling_y - floor_y
    if not np.isfinite(room_height) or room_height <= 0.5:
        room_height = float(np.percentile(vertices[:, 1], 95.0) - floor_y)
    if not np.isfinite(room_height) or room_height <= 0:
        raise ValueError("Cannot estimate a positive room height from the sealed mesh")
    return floor_faces, room_height


def optimize_cut_planes(vertices: np.ndarray, axes: tuple[np.ndarray, np.ndarray], face_data) -> dict:
    """Score one vertical cut offset per wall-aligned axis using fast area profiles."""
    face_centers, face_normals, face_areas = face_data
    # shortcut: this mesh is open, so floor slices proxy filled sections; use voxel occupancy when closed geometry is available.
    floor_faces, room_height = estimate_floor_height(
        face_centers, face_normals, face_areas, vertices,
    )
    horizontal_points = np.asarray(vertices, dtype=np.float32)[:, [0, 2]]
    horizontal_centers = face_centers[:, [0, 2]]
    floor_projected_area = face_areas[floor_faces] * np.abs(face_normals[floor_faces, 1])
    vertical_surface_area = face_areas * np.linalg.norm(face_normals[:, [0, 2]], axis=1)

    results = []
    for axis_index, normal in enumerate(axes, 1):
        all_s = horizontal_points @ normal
        span = float(all_s.max() - all_s.min())
        if not np.isfinite(span) or span <= 1e-6:
            raise ValueError(f"Scene has no extent along cut axis {axis_index}")
        bin_count = 128
        edges = np.linspace(float(all_s.min()), float(all_s.max()), bin_count + 1)
        step = float(edges[1] - edges[0])
        offsets = (edges[:-1] + edges[1:]) * 0.5

        face_s = horizontal_centers @ normal
        floor_area_hist = np.histogram(
            face_s[floor_faces], bins=edges, weights=floor_projected_area,
        )[0]
        # Floor area per offset gives the footprint chord; multiplying by room height
        # estimates the filled vertical cross-section area without a costly volume fill.
        section_area = _smooth_profile(
            floor_area_hist, max(1.0, PLANE_SUPPORT_SIGMA_FRACTION * span / step),
        ) / step * room_height
        max_area = float(section_area.max())
        if max_area <= 0:
            raise ValueError(f"Could not estimate floor cross-section area along cut axis {axis_index}")

        delta = offsets[None, :] - offsets[:, None]
        balance_sigma = max(PLANE_BALANCE_SIGMA_FRACTION * span, step)
        local_weights = np.exp(-0.5 * (delta / balance_sigma) ** 2)
        left_mass = (local_weights * section_area[None, :] * (delta < 0)).sum(axis=1)
        right_mass = (local_weights * section_area[None, :] * (delta > 0)).sum(axis=1)
        balance_loss = ((left_mass - right_mass) / (left_mass + right_mass + 1e-12)) ** 2

        # shortcut: weighted face-centroid bins approximate triangle distance; use exact triangle-plane distance for highly uneven meshes.
        surface_hist = np.histogram(
            face_s, bins=edges, weights=vertical_surface_area,
        )[0]
        clearance_sigma = max(PLANE_SURFACE_SIGMA_FRACTION * span, step)
        near_surface = _smooth_profile(surface_hist, clearance_sigma / step)
        active = section_area >= max_area * 0.05
        max_near = float(near_surface[active].max()) if active.any() else 0.0
        proximity_loss = near_surface / max_near if max_near > 0 else np.zeros_like(near_surface)
        support = section_area / max_area
        score = support - PLANE_BALANCE_WEIGHT * balance_loss - PLANE_PROXIMITY_WEIGHT * proximity_loss
        candidates = np.flatnonzero(active)
        best = int(candidates[np.argmax(score[candidates])])

        results.append({
            "axis_index": axis_index,
            "plane_normal_xz": [float(normal[0]), float(normal[1])],
            "plane_equation": f"{normal[0]:.8f}*x + {normal[1]:.8f}*z = offset",
            "offset_scene_units": float(offsets[best]),
            "cross_section_area_scene_units2": float(section_area[best]),
            "local_balance_loss": float(balance_loss[best]),
            "near_surface_loss": float(proximity_loss[best]),
            "score": float(score[best]),
            "profile": [
                {
                    "offset_scene_units": float(offsets[i]),
                    "cross_section_area_scene_units2": float(section_area[i]),
                    "local_balance_loss": float(balance_loss[i]),
                    "near_surface_loss": float(proximity_loss[i]),
                    "score": float(score[i]),
                }
                for i in candidates
            ],
        })
    return {
        "method": "floor-footprint section area with local Gaussian balance and area-weighted nearby vertical surfaces",
        "room_height_scene_units": float(room_height),
        "support_smoothing_sigma_fraction_of_axis_span": PLANE_SUPPORT_SIGMA_FRACTION,
        "balance_sigma_fraction_of_axis_span": PLANE_BALANCE_SIGMA_FRACTION,
        "surface_distance_sigma_fraction_of_axis_span": PLANE_SURFACE_SIGMA_FRACTION,
        "score": "normalized_support - 0.20 * local_balance_loss - 0.35 * near_surface_loss",
        "planes": results,
    }


def write_cut_plane_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def render_cut_plane_map(vertices: np.ndarray, axes: tuple[np.ndarray, np.ndarray],
                         face_data, report: dict, output_path: Path) -> None:
    import cv2
    from PIL import Image, ImageDraw

    centers, normals, areas = face_data
    floor_faces, _ = estimate_floor_height(centers, normals, areas, vertices)
    basis = np.stack(axes, axis=1)
    center_map = centers[:, [0, 2]] @ basis
    vertex_map = np.asarray(vertices)[:, [0, 2]] @ basis
    low = vertex_map.min(axis=0)
    high = vertex_map.max(axis=0)
    span = high - low
    max_dim = 384
    bins = np.maximum(2, np.ceil(max_dim * span / span.max()).astype(int))
    floor_weights = areas[floor_faces] * np.abs(normals[floor_faces, 1])
    footprint, _, _ = np.histogram2d(
        center_map[floor_faces, 0], center_map[floor_faces, 1],
        bins=tuple(bins), range=((low[0], high[0]), (low[1], high[1])),
        weights=floor_weights,
    )
    mask = (footprint.T[::-1, :] > 0).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    map_image = np.where(mask > 0, 165, 248).astype(np.uint8)
    image = Image.fromarray(map_image).convert("RGB")
    image.thumbnail((680, 680), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (720, 760), "white")
    left, top = (720 - image.width) // 2, 45 + (680 - image.height) // 2
    canvas.paste(image, (left, top))
    draw = ImageDraw.Draw(canvas)
    plane1, plane2 = report["planes"]
    x = left + (plane1["offset_scene_units"] - low[0]) / (high[0] - low[0]) * image.width
    y = top + (high[1] - plane2["offset_scene_units"]) / (high[1] - low[1]) * image.height
    draw.line((x, top, x, top + image.height), fill=(230, 35, 35), width=3)
    draw.line((left, y, left + image.width, y), fill=(35, 85, 230), width=3)
    draw.text((12, 12), "Top-down floor footprint and selected vertical cuts", fill=(20, 20, 20))
    draw.text((12, 738), "Axis 1: red   Axis 2: blue", fill=(20, 20, 20))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def clip_mesh_halfspace(mesh, normal_xz: np.ndarray, offset: float, keep_positive: bool):
    """Clip triangles to one side of a vertical plane without capping the cut."""
    import open3d as o3d

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    normals = np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None
    signed = (vertices[:, [0, 2]] @ normal_xz - offset) * (1.0 if keep_positive else -1.0)
    inside = signed >= 0
    face_inside = inside[triangles]
    inside_count = face_inside.sum(axis=1)
    retained = triangles[inside_count == 3]
    crossing = triangles[(inside_count > 0) & (inside_count < 3)]
    crossing_count = inside_count[(inside_count > 0) & (inside_count < 3)]

    clipped_vertices = vertices
    clipped_triangles = retained
    clipped_colors = colors
    clipped_normals = normals
    if len(crossing):
        points = vertices[crossing]
        distances = signed[crossing]
        next_points = np.roll(points, -1, axis=1)
        next_distances = np.roll(distances, -1, axis=1)
        edge_inside = distances >= 0
        crosses_plane = edge_inside != (next_distances >= 0)
        denominator = distances - next_distances
        fraction = np.divide(
            distances, denominator, out=np.zeros_like(distances), where=denominator != 0,
        )
        intersections = points + fraction[..., None] * (next_points - points)
        sequence = np.zeros((len(crossing), 6, 3), dtype=vertices.dtype)
        valid = np.zeros((len(crossing), 6), dtype=bool)
        color_sequence = np.zeros_like(sequence) if colors is not None else None
        normal_sequence = np.zeros_like(sequence) if normals is not None else None
        color_points = colors[crossing] if colors is not None else None
        normal_points = normals[crossing] if normals is not None else None
        color_intersections = (
            color_points + fraction[..., None] * (np.roll(color_points, -1, axis=1) - color_points)
            if color_points is not None else None
        )
        normal_intersections = (
            normal_points + fraction[..., None] * (np.roll(normal_points, -1, axis=1) - normal_points)
            if normal_points is not None else None
        )
        for edge in range(3):
            sequence[:, edge * 2] = points[:, edge]
            sequence[:, edge * 2 + 1] = intersections[:, edge]
            valid[:, edge * 2] = edge_inside[:, edge]
            valid[:, edge * 2 + 1] = crosses_plane[:, edge]
            if colors is not None:
                color_sequence[:, edge * 2] = color_points[:, edge]
                color_sequence[:, edge * 2 + 1] = color_intersections[:, edge]
            if normals is not None:
                normal_sequence[:, edge * 2] = normal_points[:, edge]
                normal_sequence[:, edge * 2 + 1] = normal_intersections[:, edge]
        order = np.argsort(~valid, axis=1, kind="stable")
        polygons = np.take_along_axis(sequence, order[..., None], axis=1)[:, :4]
        polygon_colors = (
            np.take_along_axis(color_sequence, order[..., None], axis=1)[:, :4]
            if color_sequence is not None else None
        )
        polygon_normals = (
            np.take_along_axis(normal_sequence, order[..., None], axis=1)[:, :4]
            if normal_sequence is not None else None
        )
        first_index = len(vertices) + np.arange(len(crossing)) * 4
        first_triangles = first_index[:, None] + np.array([0, 1, 2])
        quad_triangles = first_index[crossing_count == 2, None] + np.array([[0, 2, 3]])
        clipped_triangles = np.concatenate((retained, first_triangles, quad_triangles), axis=0)
        clipped_vertices = np.concatenate((vertices, polygons.reshape(-1, 3)), axis=0)
        if colors is not None:
            clipped_colors = np.concatenate((colors, polygon_colors.reshape(-1, 3)), axis=0)
        if normals is not None:
            clipped_normals = np.concatenate((normals, polygon_normals.reshape(-1, 3)), axis=0)

    result = o3d.geometry.TriangleMesh()
    result.vertices = o3d.utility.Vector3dVector(clipped_vertices)
    result.triangles = o3d.utility.Vector3iVector(clipped_triangles.astype(np.int32))
    if clipped_colors is not None:
        result.vertex_colors = o3d.utility.Vector3dVector(clipped_colors)
    if clipped_normals is not None:
        lengths = np.linalg.norm(clipped_normals, axis=1)
        valid_normals = lengths > 1e-12
        clipped_normals[valid_normals] /= lengths[valid_normals, None]
        result.vertex_normals = o3d.utility.Vector3dVector(clipped_normals)
    return result


def render_mesh_cutaways(mesh, views: list[dict], cut_planes: dict,
                         output_dir: Path) -> None:
    import open3d as o3d
    from PIL import Image, ImageOps

    if mesh.is_empty() or not mesh.has_triangles():
        raise ValueError("Sealed mesh is empty or has no triangles")
    renderer = o3d.visualization.rendering.OffscreenRenderer(IMAGE_WIDTH, IMAGE_HEIGHT)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.base_color = [1.0, 1.0, 1.0, 1.0]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        IMAGE_WIDTH, IMAGE_HEIGHT,
        float(views[0]["K"][0, 0]), float(views[0]["K"][1, 1]),
        float(views[0]["K"][0, 2]), float(views[0]["K"][1, 2]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for view in views:
        axis_index = int(view["name"].split("_")[1])
        plane = cut_planes["planes"][axis_index - 1]
        camera_on_positive_side = view["name"].endswith("_positive")
        clipped = clip_mesh_halfspace(
            mesh, np.asarray(plane["plane_normal_xz"]), plane["offset_scene_units"],
            keep_positive=not camera_on_positive_side,
        )
        if clipped.is_empty() or not clipped.has_triangles():
            raise ValueError(f"Selected cut plane removed the entire mesh for {view['name']}")
        if not clipped.has_vertex_normals():
            clipped.compute_vertex_normals()
        geometry_name = "cutaway_mesh"
        renderer.scene.add_geometry(geometry_name, clipped, material)
        renderer.setup_camera(intrinsic, view["viewmat"])
        image = np.asarray(renderer.render_to_image())
        ImageOps.mirror(Image.fromarray(image)).save(output_dir / f"{view['name']}.png")
        renderer.scene.remove_geometry(geometry_name)


def gaussian_keep_mask(means: np.ndarray, view: dict, cut_planes: dict) -> np.ndarray:
    axis_index = int(view["name"].split("_")[1])
    plane = cut_planes["planes"][axis_index - 1]
    camera_on_positive_side = view["name"].endswith("_positive")
    keep_positive = not camera_on_positive_side
    # shortcut: Gaussian splats are indivisible; classify each splat by its center against the exact mesh plane.
    signed = means[:, [0, 2]] @ np.asarray(plane["plane_normal_xz"]) - plane["offset_scene_units"]
    return signed * (1.0 if keep_positive else -1.0) >= 0


def render_gaussians(checkpoint: dict, views: list[dict], output_dir: Path,
                     means_cpu: np.ndarray, cut_planes: dict) -> None:
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageOps
    from gsplat.rendering import rasterization

    raw = checkpoint.get("splats", {})
    required = {"means", "scales", "quats", "opacities", "sh0", "shN"}
    missing = required - raw.keys() if isinstance(raw, dict) else required
    if missing:
        raise ValueError(f"Gaussian checkpoint is missing parameters: {sorted(missing)}")

    device = torch.device("cuda")
    splats = {name: value.to(device) for name, value in raw.items() if name in required}
    colors = torch.cat((splats["sh0"], splats["shN"]), dim=1)
    background = torch.ones((1, 3), dtype=torch.float32, device=device)
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for view in views:
            keep = torch.as_tensor(gaussian_keep_mask(means_cpu, view, cut_planes), device=device)
            if not keep.any():
                Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), "white").save(output_dir / f"{view['name']}.png")
                continue
            rendered, _, _ = rasterization(
                means=splats["means"][keep], quats=F.normalize(splats["quats"][keep], dim=-1),
                scales=splats["scales"][keep].exp(), opacities=splats["opacities"][keep].sigmoid(),
                colors=colors[keep],
                viewmats=torch.as_tensor(view["viewmat"], device=device)[None],
                Ks=torch.as_tensor(view["K"], device=device)[None],
                width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
                sh_degree=int(checkpoint.get("sh_degree", 3)),
                backgrounds=background, packed=False, render_mode="RGB",
            )
            image = (rendered[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            ImageOps.mirror(Image.fromarray(image)).save(output_dir / f"{view['name']}.png")
            print(f"{view['name']}: retained {keep.sum().item():,} splats after the cut")


def resolve_scans(workspace: Path) -> list[Path]:
    scans = [workspace] if (workspace / "mesh.ply").is_file() else sorted(
        path for path in workspace.glob("scan_*") if path.is_dir()
    )
    if not scans:
        raise FileNotFoundError(f"No scan folders found under {workspace}")
    for scan_path in scans:
        if not (scan_path / "mesh.ply").is_file():
            raise FileNotFoundError(f"Sealed mesh missing: {scan_path / 'mesh.ply'}")
        checkpoint = scan_path / "gaussian_splat" / "checkpoint_sealed.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Sealed Gaussian checkpoint missing: {checkpoint}")
    return scans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=BACKEND_DIR / "current_scene",
                        help="current_scene root or one scan directory (default: backend/current_scene)")
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    scan_paths = resolve_scans(workspace)
    scene_root = workspace.parent if workspace.name.startswith("scan_") else workspace
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Gaussian rendering requires the configured CUDA/ROCm GPU")
    import open3d as o3d
    for scan_path in scan_paths:
        checkpoint_path = scan_path / "gaussian_splat" / "checkpoint_sealed.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (not isinstance(checkpoint, dict) or checkpoint.get("format") != "gsplat-stage03-v1"
                or not checkpoint.get("sealed")):
            raise ValueError(f"Expected a sealed stage 03 Gaussian checkpoint: {checkpoint_path}")
        splats = checkpoint.get("splats")
        required = {"means", "scales", "quats", "opacities", "sh0", "shN"}
        if not isinstance(splats, dict) or not required.issubset(splats):
            missing = required - splats.keys() if isinstance(splats, dict) else required
            raise ValueError(f"Gaussian checkpoint is missing parameters: {sorted(missing)}")
        if not torch.is_tensor(splats["means"]) or not torch.is_tensor(splats["scales"]):
            raise ValueError(f"Gaussian checkpoint has invalid means or scales: {checkpoint_path}")
        means_cpu = splats["means"].detach().cpu().numpy().astype(np.float32, copy=False)
        mesh_path = scan_path / "mesh.ply"
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if mesh.is_empty() or not mesh.has_triangles():
            raise ValueError(f"Sealed mesh is empty or has no triangles: {mesh_path}")
        mesh.compute_vertex_normals()
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        face_data = mesh_face_geometry(vertices, triangles)
        axes = wall_axes(vertices, triangles, face_data)
        cut_planes = optimize_cut_planes(vertices, axes, face_data)
        artifacts = scene_root / "artifacts" / "scene_views" / scan_path.name
        write_cut_plane_report(cut_planes, artifacts / "cut_planes.json")
        render_cut_plane_map(
            vertices, axes, face_data, cut_planes,
            artifacts / "cut_plane_selection.png",
        )
        for plane in cut_planes["planes"]:
            print(
                f"cut axis {plane['axis_index']}: offset={plane['offset_scene_units']:.3f}, "
                f"section={plane['cross_section_area_scene_units2']:.3f}, "
                f"balance={plane['local_balance_loss']:.3f}, "
                f"near-surface={plane['near_surface_loss']:.3f}"
            )
        views = camera_views(vertices, triangles, means_cpu, axes)
        render_mesh_cutaways(mesh, views, cut_planes, artifacts / "mesh")
        render_gaussians(checkpoint, views, artifacts / "gaussian_splat", means_cpu, cut_planes)
        print(f"Wrote four matching views per model to {artifacts}")


if __name__ == "__main__":
    main()
