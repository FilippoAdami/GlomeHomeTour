#!/usr/bin/env python3
"""
Extract Unobstructed Cross-Section Views.

Captures direct, unobstructed 2D projection frames from the aligned original scene mesh:
- Applies canonical axis-alignment R_axis (if axis_alignment_transform.json is present) to align
  major planes strictly along the Cartesian axes (Dir A -> +X, Dir B -> +Z).
- Optionally applies a legacy rigid-baseline alignment report when one is supplied.
- Removes highest roof/ceiling using its fitted plane equation (with vertical/normal tilt-aware padding)
  to ensure ZERO ceiling floaters in top-down view.
- Removes mezzanine level and underside ceiling with downward padding to ensure ZERO mezzanine floaters.
- Slices room vertically at the room center along canonical X and Z axes.
- Mathematically positions the camera sufficiently far back so that 100% of all vertices lie strictly
  within the view frustum (with safe padding margin).
- Preserves full authentic vertex colors and all objects inside.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import trimesh
from shapely.geometry import shape
from shapely.ops import unary_union


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-path", type=Path, required=True, help="Path to original mesh.ply")
    parser.add_argument("--instances-dir", type=Path, required=True, help="Directory containing regularized manifests")
    parser.add_argument("--alignment-report", type=Path, help="Optional path to rigid_baseline_report.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save cross-section frames")
    parser.add_argument("--resolution", type=int, default=960, help="Resolution (width/height) of render frames")
    parser.add_argument("--projection-map-dir", type=Path, help="Optional directory for per-pixel source-triangle maps")
    return parser.parse_args()


def load_transforms(instances_dir: Path, report_path: Optional[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Combines axis alignment rotation R_axis and rigid baseline alignment (R_align, t_align)."""
    R_axis = np.eye(3, dtype=np.float64)
    axis_json = instances_dir / "axis_alignment_transform.json"
    if axis_json.is_file():
        try:
            data = json.loads(axis_json.read_text())
            R_axis = np.array(data.get("rotation_matrix", np.eye(3)), dtype=np.float64)
        except Exception as e:
            print(f"Warning: could not read {axis_json}: {e}")

    R_align = np.eye(3, dtype=np.float64)
    t_align = np.zeros(3, dtype=np.float64)
    if report_path and report_path.is_file():
        try:
            data = json.loads(report_path.read_text())
            transform = data.get("transform", {})
            R_align = np.array(transform.get("rotation", np.eye(3)), dtype=np.float64)
            t_align = np.array(transform.get("translation_m", np.zeros(3)), dtype=np.float64)
        except Exception as e:
            print(f"Warning: could not read {report_path}: {e}")

    R_total = R_align @ R_axis
    t_total = t_align
    return R_total, t_total


def slice_open3d_mesh_plane(
    mesh: o3d.geometry.TriangleMesh,
    plane_normal: np.ndarray,
    plane_origin: np.ndarray,
    return_face_indices: bool = False,
) -> o3d.geometry.TriangleMesh | Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """Slices mesh by a plane: keeps face centers where dot(c - plane_origin, plane_normal) >= 0."""
    normal = np.asarray(plane_normal, dtype=np.float64) / np.linalg.norm(plane_normal)
    origin = np.asarray(plane_origin, dtype=np.float64)

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    has_colors = mesh.has_vertex_colors()
    colors = np.asarray(mesh.vertex_colors) if has_colors else None

    face_centers = verts[faces].mean(axis=1)
    keep = ((face_centers - origin) @ normal) >= 0

    if not np.any(keep):
        print("Warning: slice removed all triangles; returning original mesh")
        return (mesh, np.arange(len(faces))) if return_face_indices else mesh

    sub_m = o3d.geometry.TriangleMesh()
    used_indices, inverse = np.unique(faces[keep], return_inverse=True)
    sub_m.vertices = o3d.utility.Vector3dVector(verts[used_indices])
    sub_m.triangles = o3d.utility.Vector3iVector(inverse.reshape(-1, 3))
    if has_colors:
        sub_m.vertex_colors = o3d.utility.Vector3dVector(colors[used_indices])
    sub_m.compute_vertex_normals()
    return (sub_m, np.flatnonzero(keep)) if return_face_indices else sub_m


def classify_outer_shell_faces(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Mark outward sides whose normal-offset sample lies outside the full scene envelope."""
    component_ids, component_counts, _ = mesh.cluster_connected_triangles()
    component_ids = np.asarray(component_ids, dtype=np.int32)
    component_counts = np.asarray(component_counts, dtype=np.int64)
    outer_component = int(np.argmax(component_counts))
    enclosure = o3d.geometry.TriangleMesh()
    enclosure.vertices = mesh.vertices
    enclosure.triangles = o3d.utility.Vector3iVector(
        np.asarray(mesh.triangles)[component_ids == outer_component]
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(enclosure))
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    mesh.compute_triangle_normals()
    centers = vertices[faces].mean(axis=1)
    normals = np.asarray(mesh.triangle_normals)
    offset = max(float(np.linalg.norm(mesh.get_max_bound() - mesh.get_min_bound())) * 1e-5, 1e-4)
    result = np.zeros(len(faces), dtype=bool)
    for start in range(0, len(faces), 200_000):
        stop = min(start + 200_000, len(faces))
        probes = (centers[start:stop] + offset * normals[start:stop]).astype(np.float32)
        occupancy = scene.compute_occupancy(o3d.core.Tensor(probes)).numpy()
        result[start:stop] = occupancy <= 0.5
    return result


def render_cut_view(
    o3d_mesh: o3d.geometry.TriangleMesh,
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    resolution: int,
    fov_deg: float = 60.0,
    background_color: Tuple[float, float, float] = (0.12, 0.12, 0.12),
    orthographic: bool = False,
    outer_shell_faces: Optional[np.ndarray] = None,
    return_projection: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Pass outward sides of the pre-oriented complete mesh through to the nearest inward side."""
    background = np.asarray(background_color, dtype=np.float64) * 255
    image = np.broadcast_to(np.rint(background).astype(np.uint8), (resolution, resolution, 3)).copy()
    projection = None
    if return_projection:
        projection = {
            "triangle_ids": np.full((resolution, resolution), -1, dtype=np.int32),
            "barycentric": np.zeros((resolution, resolution, 3), dtype=np.float32),
            "view_cosine": np.zeros((resolution, resolution), dtype=np.float32),
        }

    def finish():
        return (image, projection) if return_projection else image

    if not len(o3d_mesh.triangles):
        return finish()

    forward = np.asarray(target, dtype=np.float64) - eye
    forward /= np.linalg.norm(forward)
    camera_up = np.asarray(up, dtype=np.float64)
    camera_up /= np.linalg.norm(camera_up)
    right = np.cross(forward, camera_up)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    rotation = np.stack((right, -camera_up, forward))
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = -rotation @ eye
    focal = 0.5 * resolution / np.tan(np.radians(fov_deg) / 2.0)
    intrinsic = np.array([[focal, 0.0, resolution / 2.0],
                          [0.0, focal, resolution / 2.0],
                          [0.0, 0.0, 1.0]], dtype=np.float64)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh))
    if orthographic:
        half_size = np.linalg.norm(target - eye) * np.tan(np.radians(fov_deg) / 2.0)
        pixel_size = 2.0 * half_size / resolution
        x = (np.arange(resolution) + 0.5 - resolution / 2.0) * pixel_size
        y = (resolution / 2.0 - np.arange(resolution) - 0.5) * pixel_size
        xx, yy = np.meshgrid(x, y)
        origins = eye + xx[..., None] * right + yy[..., None] * camera_up
        directions = np.broadcast_to(forward, origins.shape)
        rays = np.concatenate((origins, directions), axis=-1).reshape(-1, 6).astype(np.float32)
    else:
        rays = scene.create_rays_pinhole(
            intrinsic_matrix=o3d.core.Tensor(intrinsic),
            extrinsic_matrix=o3d.core.Tensor(extrinsic),
            width_px=resolution,
            height_px=resolution,
        ).numpy().reshape(-1, 6)
    vertex_colors = np.asarray(o3d_mesh.vertex_colors) if o3d_mesh.has_vertex_colors() else None
    faces = np.asarray(o3d_mesh.triangles)
    o3d_mesh.compute_triangle_normals()
    triangle_normals = np.asarray(o3d_mesh.triangle_normals)
    hits = scene.list_intersections(o3d.core.Tensor(rays))
    ray_ids = hits["ray_ids"].numpy().astype(np.int64)
    ids = hits["primitive_ids"].numpy().astype(np.int64)
    distances = hits["t_hit"].numpy()
    uvs = hits["primitive_uvs"].numpy()
    if not len(ids):
        return finish()

    # Face winding was normalized on the complete sealed mesh before any slicing.
    # Test every hit so transparent outer faces cannot hide another outer face behind them.
    order = np.lexsort((distances, ray_ids))
    ray_directions = rays[ray_ids, 3:].astype(np.float64)
    ray_directions /= np.linalg.norm(ray_directions, axis=1, keepdims=True)
    outward = np.einsum("ij,ij->i", triangle_normals[ids], ray_directions) < -0.05
    if outer_shell_faces is not None:
        outward &= outer_shell_faces[ids]

    eligible = order[~outward[order]]
    if not len(eligible):
        return finish()
    eligible_rays = ray_ids[eligible]
    selected = eligible[np.r_[True, eligible_rays[1:] != eligible_rays[:-1]]]
    selected_rays = ray_ids[selected]
    selected_ids = ids[selected]

    if vertex_colors is None:
        colors = np.ones((len(selected), 3), dtype=np.float64)
    uv = uvs[selected]
    weights = np.column_stack((1.0 - uv.sum(axis=1), uv))
    if vertex_colors is not None:
        face_colors = vertex_colors[faces[selected_ids]]
        colors = np.einsum("ij,ijk->ik", weights, face_colors)
        colors = np.where(colors <= 0.0031308, 12.92 * colors,
                          1.055 * np.power(colors, 1.0 / 2.4) - 0.055)
    image.reshape(-1, 3)[selected_rays] = np.clip(np.rint(colors * 255), 0, 255).astype(np.uint8)

    if return_projection:
        flat_triangles = projection["triangle_ids"].reshape(-1)
        flat_barycentric = projection["barycentric"].reshape(-1, 3)
        flat_cosine = projection["view_cosine"].reshape(-1)
        flat_triangles[selected_rays] = selected_ids
        flat_barycentric[selected_rays] = weights
        flat_cosine[selected_rays] = np.abs(np.einsum(
            "ij,ij->i", triangle_normals[selected_ids], ray_directions[selected]
        )).astype(np.float32)

    return finish()


def compute_framing_camera(
    mesh_part: o3d.geometry.TriangleMesh,
    view_direction: np.ndarray,
    up_vector: np.ndarray,
    fov_deg: float = 60.0,
    margin_factor: float = 1.30,
    frame_points: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes camera target and eye position backed up sufficiently so that 100% of all vertices
    fall strictly inside the camera frustum with a guaranteed safe margin.
    """
    view_dir = np.asarray(view_direction, dtype=np.float64) / np.linalg.norm(view_direction)
    up = np.asarray(up_vector, dtype=np.float64) / np.linalg.norm(up_vector)
    right = np.cross(view_dir, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, view_dir)
    up = up / np.linalg.norm(up)

    verts = np.asarray(mesh_part.vertices)
    if len(verts) == 0:
        return np.zeros(3), -view_dir * 5.0

    fit_points = verts if frame_points is None else np.asarray(frame_points, dtype=np.float64)
    target = (fit_points.min(axis=0) + fit_points.max(axis=0)) / 2.0

    rel_points = fit_points - target
    x_proj = rel_points @ right
    y_proj = rel_points @ up
    z_proj = rel_points @ view_dir

    fov_rad = np.radians(fov_deg)
    tan_half_fov = np.tan(fov_rad / 2.0)

    # For every vertex i, to satisfy |x_i| / (D + z_i) <= tan(fov/2) / margin_factor:
    # D >= |x_i| * margin_factor / tan(fov/2) - z_i
    req_dx = np.abs(x_proj) * margin_factor / tan_half_fov - z_proj
    req_dy = np.abs(y_proj) * margin_factor / tan_half_fov - z_proj
    req_near = -z_proj + 0.8  # near clipping buffer

    dist = float(max(req_dx.max(), req_dy.max(), req_near.max()))
    eye = target - view_dir * dist
    return target, eye


def discover_floors_and_ceilings(instances_dir: Path) -> List[Dict[str, Any]]:
    """
    Discovers levels, floors, and associated ceilings generically from manifests.
    Checks structural_levels_manifest.json, floors_ceilings_planes_manifest.json,
    and plane_patterns_manifest.json in priority order.
    Returns a sorted list of levels with floor elevation and ceiling details.
    """
    levels_manifest_path = instances_dir / "structural_levels_manifest.json"
    fc_manifest_path = instances_dir / "floors_ceilings_planes_manifest.json"
    pattern_manifest_path = instances_dir / "plane_patterns_manifest.json"

    ceilings_pool: List[Dict[str, Any]] = []
    floors_pool: List[Dict[str, Any]] = []

    # 1. Harvest ceilings & floors from floors_ceilings_planes_manifest if available
    if fc_manifest_path.is_file():
        try:
            fc_data = json.loads(fc_manifest_path.read_text())
            for elem in fc_data.get("elements", []):
                if elem.get("type") == "ceiling":
                    ceilings_pool.append(elem)
                elif elem.get("type") == "floor":
                    floors_pool.append(elem)
        except Exception as e:
            print(f"Warning reading {fc_manifest_path}: {e}")

    # Fallback/augment from plane_patterns_manifest
    if pattern_manifest_path.is_file():
        try:
            p_data = json.loads(pattern_manifest_path.read_text())
            for idx, unit in enumerate(p_data.get("units", [])):
                role = unit.get("structural_role")
                name = unit.get("unit_name") or unit.get("instance_name") or f"{role}_{idx}"
                if role == "ceiling" and not any(c.get("name") == name for c in ceilings_pool):
                    ceilings_pool.append({
                        "name": name,
                        "type": "ceiling",
                        "mean_elevation_y": unit.get("mean_elevation_y", 0.0),
                        "plane_equation": unit.get("plane_equation"),
                        "normal_unit": unit.get("plane_equation", [0, -1, 0])[:3],
                    })
                elif role == "floor" and not any(f.get("name") == name for f in floors_pool):
                    floors_pool.append({
                        "name": name,
                        "type": "floor",
                        "mean_elevation_y": unit.get("mean_elevation_y", 0.0),
                        "plane_equation": unit.get("plane_equation"),
                    })
        except Exception as e:
            print(f"Warning reading {pattern_manifest_path}: {e}")

    # Helper: resolve tilt & origin from ceiling entry
    def parse_ceiling_plane(c_entry: Optional[Dict[str, Any]], fallback_y: float, center_xz: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, float]:
        if not c_entry:
            return np.array([0.0, -1.0, 0.0]), np.array([center_xz[0], fallback_y - 0.20, center_xz[1]]), 0.0

        tilt = float(c_entry.get("tilt_from_horizontal_deg", 0.0))
        pe = c_entry.get("plane_equation")
        norm = c_entry.get("normal_unit")

        if norm is not None and len(norm) == 3:
            n = np.array(norm, dtype=np.float64)
            if n[1] > 0:
                n = -n
            n = n / np.linalg.norm(n)
        elif pe is not None and len(pe) == 4:
            n = np.array(pe[:3], dtype=np.float64)
            if n[1] > 0:
                n = -n
            n = n / np.linalg.norm(n)
        else:
            n = np.array([0.0, -1.0, 0.0])

        # Recalculate tilt if not explicitly stated
        if tilt == 0.0 and abs(abs(n[1]) - 1.0) > 0.02:
            tilt = float(np.degrees(np.arccos(np.clip(abs(n[1]), -1.0, 1.0))))

        if tilt > 2.0 and pe is not None and len(pe) == 4:
            pe_arr = np.array(pe, dtype=np.float64)
            d = pe_arr[3]
            p0 = -d * pe_arr[:3]
            # Move farther into the room to clear the complete ceiling layer.
            origin = p0 + n * 0.75
            return n, origin, tilt
        else:
            mean_y = float(c_entry.get("mean_elevation_y", fallback_y))
            # Horizontal cut padded down far enough to remove the ceiling surface.
            origin = np.array([center_xz[0], mean_y - 0.30, center_xz[1]])
            return np.array([0.0, -1.0, 0.0]), origin, tilt

    # 2. Try structural_levels_manifest.json if available
    discovered_levels = []
    if levels_manifest_path.is_file():
        try:
            levels_data = json.loads(levels_manifest_path.read_text())
            for idx, lvl in enumerate(levels_data.get("levels", [])):
                fl_y = lvl.get("mean_floor_elevation_y", lvl.get("floor_elevation_y"))
                if fl_y is None:
                    fu = lvl.get("floor_units", [])
                    if fu:
                        fl_y = fu[0].get("mean_elevation_y", 0.0)
                    else:
                        fl_y = 0.0
                fl_y = float(fl_y)

                # Match ceiling
                c_name = lvl.get("ceiling") or lvl.get("ceiling_soffit")
                c_entry = next((c for c in ceilings_pool if c.get("name") == c_name), None)

                discovered_levels.append({
                    "floor_index": idx,
                    "floor_elevation_y": fl_y,
                    "ceiling_entry": c_entry,
                    "footprint_xz": unary_union([
                        shape(unit["footprint_xz"]) for unit in lvl.get("floor_units", [])
                        if unit.get("footprint_xz")
                    ]) if any(unit.get("footprint_xz") for unit in lvl.get("floor_units", [])) else None,
                })
        except Exception as e:
            print(f"Warning reading {levels_manifest_path}: {e}")

    # Fallback to clustering floors if no levels manifest
    if not discovered_levels and floors_pool:
        # Cluster floors by elevation
        sorted_floors = sorted(floors_pool, key=lambda f: f.get("mean_elevation_y", 0.0))
        cluster_groups: List[List[Dict[str, Any]]] = []
        for fl in sorted_floors:
            y = float(fl.get("mean_elevation_y", 0.0))
            if not cluster_groups or (y - cluster_groups[-1][-1].get("mean_elevation_y", 0.0) > 0.40):
                cluster_groups.append([fl])
            else:
                cluster_groups[-1].append(fl)

        for idx, grp in enumerate(cluster_groups):
            fl_y = float(np.mean([f.get("mean_elevation_y", 0.0) for f in grp]))
            # Find closest ceiling above this floor
            higher_ceilings = [c for c in ceilings_pool if float(c.get("mean_elevation_y", 0.0)) > fl_y + 0.30]
            c_entry = min(higher_ceilings, key=lambda c: float(c.get("mean_elevation_y", 0.0))) if higher_ceilings else None
            discovered_levels.append({
                "floor_index": idx,
                "floor_elevation_y": fl_y,
                "ceiling_entry": c_entry,
            })

    # Sort levels strictly by floor elevation
    discovered_levels.sort(key=lambda x: x["floor_elevation_y"])
    for idx, lvl in enumerate(discovered_levels):
        lvl["floor_index"] = idx

    return discovered_levels, ceilings_pool


def extract_cross_sections(
    mesh_path: Path,
    instances_dir: Path,
    output_dir: Path,
    alignment_report: Optional[Path] = None,
    resolution: int = 960,
    projection_map_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if projection_map_dir:
        projection_map_dir = Path(projection_map_dir).resolve()
        projection_map_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading scene mesh: {mesh_path}...")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not len(mesh.triangles):
        raise ValueError(f"Could not load valid triangle mesh from {mesh_path}")

    # Separate sealed bodies can have opposite winding; normalize each body before classifying faces.
    winding_mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles), process=False
    )
    trimesh.repair.fix_normals(winding_mesh, multibody=True)
    mesh.triangles = o3d.utility.Vector3iVector(winding_mesh.faces)
    outer_shell_faces = classify_outer_shell_faces(mesh)
    print(f"Classified {int(outer_shell_faces.sum())} of {len(outer_shell_faces)} faces as outer-shell faces.")

    # 1. Apply canonical axis-alignment and rigid baseline alignment
    R_total, t_total = load_transforms(instances_dir, alignment_report)
    verts = np.asarray(mesh.vertices)
    mesh.vertices = o3d.utility.Vector3dVector((R_total @ verts.T).T + t_total)
    axis_path = Path(instances_dir) / "axis_alignment_transform.json"
    axis_data = json.loads(axis_path.read_text()) if axis_path.is_file() else {}
    R_axis = np.asarray(axis_data.get("rotation_matrix", np.eye(3)), dtype=np.float64)
    R_align = R_total @ R_axis.T

    # 2. Discover structural level planes and bounds
    bounds_min = mesh.get_min_bound()
    bounds_max = mesh.get_max_bound()
    center = (bounds_min + bounds_max) / 2.0

    levels, ceilings_pool = discover_floors_and_ceilings(instances_dir)
    print(f"Discovered {len(levels)} floor levels for generic top-down cross-sections.")

    cut_definitions = []

    # Build per-floor top-down view definitions
    if not levels:
        # Default single floor if no floors detected in manifests
        levels = [{
            "floor_index": 0,
            "floor_elevation_y": bounds_min[1],
            "ceiling_entry": max(ceilings_pool, key=lambda c: c.get("mean_elevation_y", -999.0)) if ceilings_pool else None,
        }]

    for lvl in levels:
        idx = lvl["floor_index"]
        fl_y = lvl["floor_elevation_y"]
        c_entry = lvl.get("ceiling_entry")

        if not c_entry:
            next_floor_y = levels[idx + 1]["floor_elevation_y"] if idx + 1 < len(levels) else np.inf
            candidates = [
                c for c in ceilings_pool
                if fl_y + 0.30 < float(c.get("mean_elevation_y", -np.inf)) < next_floor_y + 0.20
            ]
            if candidates:
                c_entry = min(candidates, key=lambda c: float(c.get("mean_elevation_y", np.inf)))
            elif idx == len(levels) - 1 and ceilings_pool:
                c_entry = max(ceilings_pool, key=lambda c: c.get("mean_elevation_y", -999.0))

        # Helper to get ceiling cut
        if c_entry:
            tilt = float(c_entry.get("tilt_from_horizontal_deg", 0.0))
            pe = c_entry.get("plane_equation")
            norm = c_entry.get("normal_unit")
            if norm is not None and len(norm) == 3:
                n = np.array(norm, dtype=np.float64)
                if n[1] > 0:
                    n = -n
                n = n / np.linalg.norm(n)
            elif pe is not None and len(pe) == 4:
                n = np.array(pe[:3], dtype=np.float64)
                if n[1] > 0:
                    n = -n
                n = n / np.linalg.norm(n)
            else:
                n = np.array([0.0, -1.0, 0.0])

            if tilt == 0.0 and abs(abs(n[1]) - 1.0) > 0.02:
                tilt = float(np.degrees(np.arccos(np.clip(abs(n[1]), -1.0, 1.0))))

            if tilt > 2.0 and pe is not None and len(pe) == 4:
                pe_arr = np.array(pe, dtype=np.float64)
                d = pe_arr[3]
                p0 = -d * pe_arr[:3]
                top_origin = p0 + n * 0.75
                top_normal = n
                print(f"Floor {idx}: using tilted ceiling cut (tilt={tilt:.1f}°) with 0.75m padding")
            else:
                c_y = float(c_entry.get("mean_elevation_y", bounds_max[1]))
                top_origin = np.array([center[0], c_y - 0.30, center[2]])
                top_normal = np.array([0.0, -1.0, 0.0])
                print(f"Floor {idx}: using horizontal ceiling cut at Y={top_origin[1]:.3f}m")
        else:
            # Fallback for next floor above or top bounds
            if idx + 1 < len(levels):
                next_fl_y = levels[idx + 1]["floor_elevation_y"]
                top_y = next_fl_y - 0.20
            else:
                top_y = bounds_max[1] - 0.25
            top_origin = np.array([center[0], top_y, center[2]])
            top_normal = np.array([0.0, -1.0, 0.0])
            print(f"Floor {idx}: using estimated top cut at Y={top_origin[1]:.3f}m")

        # Bottom cut to isolate this floor (slice away lower floors for idx > 0)
        bottom_plane = None
        if idx > 0:
            # Keep everything above floor elevation minus padding
            bottom_origin = np.array([center[0], fl_y - 0.10, center[2]])
            bottom_normal = np.array([0.0, 1.0, 0.0])
            bottom_plane = {"origin": bottom_origin, "normal": bottom_normal}
            print(f"Floor {idx}: slicing off lower floors below Y={bottom_origin[1]:.3f}m")

        cut_definitions.append({
            "name": f"topdown_floor_{idx}",
            "description": f"Top-down plan view of floor {idx}: ceiling above sliced away and lower floors removed",
            "top_slice_plane_normal": top_normal,
            "top_slice_plane_origin": top_origin,
            "bottom_slice_plane": bottom_plane,
            "footprint_xz": lvl.get("footprint_xz"),
            "floor_elevation_y": fl_y,
            "view_direction": np.array([0.0, -1.0, 0.0]),
            "camera_up": np.array([0.0, 0.0, -1.0]),
            "fov_deg": 60.0,
        })

    # Add 4 orthogonal elevation cross-sections sliced at the room center
    cut_definitions.extend([
        {
            "name": "elevation_west_to_east_x_pos",
            "description": "Left-to-Right (+X) cross-section view: sliced at room center X, looking into East half (+X)",
            "top_slice_plane_normal": np.array([1.0, 0.0, 0.0]),
            "top_slice_plane_origin": np.array([center[0], center[1], center[2]]),
            "bottom_slice_plane": None,
            "view_direction": np.array([1.0, 0.0, 0.0]),
            "camera_up": np.array([0.0, 1.0, 0.0]),
            "fov_deg": 60.0,
        },
        {
            "name": "elevation_east_to_west_x_neg",
            "description": "Right-to-Left (-X) cross-section view: sliced at room center X, looking into West half (-X)",
            "top_slice_plane_normal": np.array([-1.0, 0.0, 0.0]),
            "top_slice_plane_origin": np.array([center[0], center[1], center[2]]),
            "bottom_slice_plane": None,
            "view_direction": np.array([-1.0, 0.0, 0.0]),
            "camera_up": np.array([0.0, 1.0, 0.0]),
            "fov_deg": 60.0,
        },
        {
            "name": "elevation_south_to_north_z_pos",
            "description": "South-to-North (+Z) cross-section view: sliced at room center Z, looking into North half (+Z)",
            "top_slice_plane_normal": np.array([0.0, 0.0, 1.0]),
            "top_slice_plane_origin": np.array([center[0], center[1], center[2]]),
            "bottom_slice_plane": None,
            "view_direction": np.array([0.0, 0.0, 1.0]),
            "camera_up": np.array([0.0, 1.0, 0.0]),
            "fov_deg": 60.0,
        },
        {
            "name": "elevation_north_to_south_z_neg",
            "description": "North-to-South (-Z) cross-section view: sliced at room center Z, looking into South half (-Z)",
            "top_slice_plane_normal": np.array([0.0, 0.0, -1.0]),
            "top_slice_plane_origin": np.array([center[0], center[1], center[2]]),
            "bottom_slice_plane": None,
            "view_direction": np.array([0.0, 0.0, -1.0]),
            "camera_up": np.array([0.0, 1.0, 0.0]),
            "fov_deg": 60.0,
        },
    ])

    manifest_entries = []
    bg_color = (0.12, 0.12, 0.12)
    for cut in cut_definitions:
        name = cut["name"]
        print(f"Generating framed cross-section: {name}...")
        # Apply top slice
        sliced_o3d, sliced_face_ids = slice_open3d_mesh_plane(
            mesh, cut["top_slice_plane_normal"], cut["top_slice_plane_origin"], return_face_indices=True
        )
        sliced_outer_shell_faces = outer_shell_faces[sliced_face_ids]
        # Apply bottom slice if defined
        if cut.get("bottom_slice_plane"):
            bp = cut["bottom_slice_plane"]
            sliced_o3d, kept_face_ids = slice_open3d_mesh_plane(
                sliced_o3d, bp["normal"], bp["origin"], return_face_indices=True
            )
            sliced_face_ids = sliced_face_ids[kept_face_ids]
            sliced_outer_shell_faces = sliced_outer_shell_faces[kept_face_ids]

        footprint = cut.get("footprint_xz")
        frame_points = None
        if footprint is not None:
            polygons = list(footprint.geoms) if hasattr(footprint, "geoms") else [footprint]
            contour = np.vstack([np.asarray(poly.exterior.coords) for poly in polygons])
            frame_points = np.column_stack((contour[:, 0], np.full(len(contour), cut["floor_elevation_y"]), contour[:, 1]))
            frame_points = (R_align @ frame_points.T).T + t_total

        target, eye = compute_framing_camera(
            mesh_part=sliced_o3d,
            view_direction=cut["view_direction"],
            up_vector=cut["camera_up"],
            fov_deg=cut["fov_deg"],
            margin_factor=1.30,
            frame_points=frame_points,
        )

        rendered = render_cut_view(
            sliced_o3d,
            eye=eye,
            target=target,
            up=cut["camera_up"],
            resolution=resolution,
            fov_deg=cut["fov_deg"],
            background_color=bg_color,
            orthographic=name.startswith("topdown_floor_"),
            outer_shell_faces=sliced_outer_shell_faces,
            return_projection=projection_map_dir is not None,
        )
        if projection_map_dir:
            img, projection = rendered
            local_triangles = projection["triangle_ids"]
            valid = local_triangles >= 0
            source_triangles = np.full(local_triangles.shape, -1, dtype=np.int32)
            source_triangles[valid] = sliced_face_ids[local_triangles[valid]]
            projection_path = projection_map_dir / f"{name}.npz"
            np.savez_compressed(
                projection_path,
                triangle_ids=source_triangles,
                barycentric=projection["barycentric"],
                view_cosine=projection["view_cosine"],
            )
        else:
            img = rendered

        img_path = output_dir / f"{name}.png"
        if not cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Could not write cross-section image: {img_path}")

        entry = {
            "view_name": name,
            "description": cut["description"],
            "image_file": str(img_path),
            "slice_plane": {
                "normal": cut["top_slice_plane_normal"].tolist(),
                "origin": cut["top_slice_plane_origin"].tolist(),
            },
            "camera": {
                "eye": [round(float(v), 4) for v in eye],
                "target": [round(float(v), 4) for v in target],
                "up": [round(float(v), 4) for v in cut["camera_up"]],
                "fov_deg": cut["fov_deg"],
            },
            "source_triangles": len(sliced_o3d.triangles),
            "image_resolution": [int(img.shape[1]), int(img.shape[0])],
        }
        if projection_map_dir:
            entry["projection_map_file"] = str(projection_path)
        if cut.get("bottom_slice_plane"):
            entry["bottom_slice_plane"] = {
                "normal": cut["bottom_slice_plane"]["normal"].tolist(),
                "origin": cut["bottom_slice_plane"]["origin"].tolist(),
            }
        manifest_entries.append(entry)

    manifest = {
        "source_mesh": str(mesh_path),
        "axis_aligned": (Path(instances_dir) / "axis_alignment_transform.json").is_file(),
        "aligned_to_regularized_frame": bool(alignment_report and alignment_report.is_file()),
        "bounds_min": bounds_min.tolist(),
        "bounds_max": bounds_max.tolist(),
        "center": center.tolist(),
        "views": manifest_entries,
    }
    manifest_path = output_dir / "cross_sections_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(manifest_entries)} cross-section views and manifest to {output_dir}")
    return manifest


def main():
    args = parse_args()
    manifest = extract_cross_sections(
        mesh_path=args.mesh_path,
        instances_dir=args.instances_dir,
        output_dir=args.output_dir,
        alignment_report=args.alignment_report,
        resolution=args.resolution,
        projection_map_dir=args.projection_map_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
