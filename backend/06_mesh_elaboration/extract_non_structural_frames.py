#!/usr/bin/env python3
"""
Extract 2D frames enclosing non-structural class instances from original mesh.

Features:
1. Synthetic In-Context Rendering (Default):
   - Generates high-quality artificial views directly from original_mesh.ply.
   - Eliminates multi-class overlap by rendering the target instance in-context
     (with structural elements: floor, walls, ceilings, stairs) while omitting
     all other non-structural classes.
   - Positions a virtual camera at a standard canonical viewpoint (elevated oblique angle,
     facing from the room interior towards the instance, with line-of-sight wall checks).
   - Crops around the instance with specified padding (default 15%).
   - Preserves non-square aspect ratios and pads to a square canvas using transparent
     pixels (RGBA PNG format), producing highest-quality assets for downstream 3D generation.
2. Photographic Camera Projection Mode (--from-cameras):
   - Projects 3D instances into registered COLMAP camera frames and extracts best camera crops.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("extract_non_structural_frames")

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir / "03_FastGS_DNSplatter") not in sys.path:
    sys.path.insert(0, str(backend_dir / "03_FastGS_DNSplatter"))
if str(backend_dir / "06_mesh_elaboration") not in sys.path:
    sys.path.insert(0, str(backend_dir / "06_mesh_elaboration"))

try:
    from gsplat_train import load_cameras
except ImportError:
    load_cameras = None

HAS_GS_RENDERER = False
try:
    from extract_non_structural_frames_gs import (
        resolve_gs_checkpoint_path,
        load_gaussian_splats,
        compute_w2c_and_intrinsics,
        render_gs_frame,
    )
    HAS_GS_RENDERER = True
except Exception as _e:
    logger.debug(f"Gaussian splat renderer not available: {_e}")


STRUCTURAL_ROLES = {"floor", "ceiling", "wall", "stair", "stairs"}


def is_structural_role(role: str) -> bool:
    """Return True if the given structural role represents a structural class."""
    role_lower = (role or "").strip().lower()
    return any(s in role_lower for s in STRUCTURAL_ROLES)


def identify_non_structural_classes(
    structural_json_path: Path,
    mesh_labels: Optional[np.ndarray] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Parse structural JSON metadata and identify all non-structural classes.
    Returns mapping from class_id to class metadata dict.
    """
    if not structural_json_path.exists():
        raise FileNotFoundError(f"Structural metadata JSON not found at {structural_json_path}")

    with open(structural_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    classifications = data.get("classifications") or data.get("classes", {})
    non_structural: Dict[int, Dict[str, Any]] = {}

    for cid_str, info in classifications.items():
        cid = int(cid_str)
        role = info.get("structural_role", "object")
        if not is_structural_role(role):
            non_structural[cid] = {
                "class_id": cid,
                "role": role,
                "confidence": info.get("confidence"),
                "details": info.get("details", ""),
            }

    if mesh_labels is not None:
        unique_labels = np.unique(mesh_labels)
        for cid_val in unique_labels:
            cid = int(cid_val)
            if cid <= 0:
                continue
            if str(cid) not in classifications and cid not in non_structural:
                non_structural[cid] = {
                    "class_id": cid,
                    "role": "object",
                    "confidence": None,
                    "details": "Unclassified non-structural class",
                }

    return non_structural


def decompose_class_instances(
    vertices: np.ndarray,
    triangles: np.ndarray,
    class_tri_mask: np.ndarray,
    min_faces: int = 30,
    min_area: float = 0.005,
) -> List[Dict[str, Any]]:
    """
    Cluster class triangles into connected components and filter small floaters.
    Returns a list of instance dicts containing vertices, face indices, and area.
    """
    c_tri_indices = np.where(class_tri_mask)[0]
    if len(c_tri_indices) == 0:
        return []

    c_triangles = triangles[c_tri_indices]
    unique_v_idx, remapped_tri = np.unique(c_triangles, return_inverse=True)
    remapped_tri = remapped_tri.reshape(c_triangles.shape)
    c_vertices = vertices[unique_v_idx]

    sub_mesh = o3d.geometry.TriangleMesh()
    sub_mesh.vertices = o3d.utility.Vector3dVector(c_vertices)
    sub_mesh.triangles = o3d.utility.Vector3iVector(remapped_tri)

    clusters, comp_counts, comp_areas = sub_mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    comp_counts = np.asarray(comp_counts)
    comp_areas = np.asarray(comp_areas)

    if len(comp_counts) == 0:
        return []

    valid_indices = [
        i for i in range(len(comp_counts))
        if comp_counts[i] >= min_faces and comp_areas[i] >= min_area
    ]
    if not valid_indices:
        valid_indices = [int(np.argmax(comp_areas))]

    valid_indices = sorted(valid_indices, key=lambda idx: comp_areas[idx], reverse=True)

    instances = []
    for inst_id, c_idx in enumerate(valid_indices):
        faces_in_comp = np.where(clusters == c_idx)[0]
        comp_tri = remapped_tri[faces_in_comp]
        comp_v_local = np.unique(comp_tri)
        comp_pts = c_vertices[comp_v_local]

        b_min = comp_pts.min(axis=0)
        b_max = comp_pts.max(axis=0)

        instances.append({
            "instance_id": inst_id,
            "component_index": int(c_idx),
            "points": comp_pts,
            "triangle_count": int(comp_counts[c_idx]),
            "area_m2": float(comp_areas[c_idx]),
            "centroid": comp_pts.mean(axis=0),
            "bbox_min": b_min,
            "bbox_max": b_max,
            "dimensions": b_max - b_min,
        })

    return instances


def pad_crop_to_square(
    crop_rgb_or_rgba: np.ndarray | Image.Image,
    min_side: Optional[int] = None,
    padding_color: str = "transparent",
) -> Image.Image:
    """
    Pad any non-square crop to a square canvas.
    - If min_side is specified and the image is smaller, upsamples the largest dimension
      to match min_side (preserving aspect ratio).
    - Pads the smaller dimension to form a square using transparent pixels (RGBA) by default,
      or black pixels (RGB) if padding_color is 'black'.
    """
    if isinstance(crop_rgb_or_rgba, np.ndarray):
        img = Image.fromarray(crop_rgb_or_rgba)
    else:
        img = crop_rgb_or_rgba

    w, h = img.size
    max_dim = max(w, h)
    target_dim = max_dim

    # If min_side requirement is given and largest side is smaller, upsample largest side:
    if min_side is not None and min_side > target_dim:
        scale = min_side / float(max_dim)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        w, h = img.size
        target_dim = min_side

    pad_lower = (padding_color or "transparent").lower()
    if pad_lower == "black":
        square_img = Image.new("RGB", (target_dim, target_dim), (0, 0, 0))
        img_rgb = img.convert("RGB")
        offset_x = (target_dim - w) // 2
        offset_y = (target_dim - h) // 2
        square_img.paste(img_rgb, (offset_x, offset_y))
    else:
        square_img = Image.new("RGBA", (target_dim, target_dim), (0, 0, 0, 0))
        img_rgba = img.convert("RGBA")
        offset_x = (target_dim - w) // 2
        offset_y = (target_dim - h) // 2
        square_img.paste(img_rgba, (offset_x, offset_y))

    return square_img


def pad_crop_to_transparent_square(
    crop_rgb_or_rgba: np.ndarray | Image.Image,
) -> Image.Image:
    """Legacy helper for backward compatibility: pads non-square image to square with alpha=0."""
    return pad_crop_to_square(crop_rgb_or_rgba, min_side=None, padding_color="transparent")


from scipy.ndimage import binary_erosion, gaussian_filter


def validate_rendered_frame(
    img: Image.Image | np.ndarray,
    min_unique_colors: int = 8,
    min_dynamic_range: float = 15.0,
    min_color_std: float = 3.5,
    min_laplacian_var: float = 0.05,
    min_fg_pixels: int = 50,
    min_autocorr: float = 0.35,
    min_p95_grad: float = 1.25,
    min_edge_density: float = 1.0,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate that a rendered frame contains meaningful, non-degenerate visual content.

    Rejects images that are:
    - Empty or have fewer than `min_fg_pixels` foreground pixels.
    - Pure noise (lag-1 spatial autocorrelation < min_autocorr for high-variance crops).
    - Just a single color (< min_unique_colors 16-bin quantized colors, dynamic range < min_dynamic_range, or gray std < min_color_std).
    - Featureless flat / blurry surfaces (few quantized colors with low gradient edges: p95_grad < min_p95_grad or edge_density < min_edge_density).
    - Ragged noise floaters / mesh artifacts (fill_pct < 85% with low contrast and low gradient).

    All spatial feature calculations (std, dynamic range, gradients, quantized palette) are evaluated
    strictly within the eroded inner crop, completely excluding artificial step artifacts at the border
    of square padding canvas.

    Returns:
        (is_valid, reason, metrics_dict)
    """
    img_arr = np.asarray(img)
    if img_arr.ndim == 2:
        img_arr = np.stack([img_arr, img_arr, img_arr], axis=2)

    if img_arr.shape[2] == 4:
        mask = img_arr[:, :, 3] > 10
    else:
        mask = img_arr.sum(axis=2) > 0

    if mask.sum() < min_fg_pixels:
        return (
            False,
            f"empty_or_too_small (fg_pixels={int(mask.sum())} < {min_fg_pixels})",
            {"fg_pixels": int(mask.sum())},
        )

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    crop = img_arr[rmin : rmax + 1, cmin : cmax + 1]
    ch, cw = crop.shape[:2]
    crop_mask = (crop[:, :, 3] > 10) if crop.shape[2] == 4 else (crop.sum(axis=2) > 0)
    fill_pct = float(crop_mask.mean() * 100)

    # Erode border to completely eliminate boundary step transitions against black/transparent padding
    eroded = binary_erosion(crop_mask, iterations=4)
    if eroded.sum() < min_fg_pixels:
        eroded = crop_mask

    # Spatial grayscale contrast
    gray = (0.2989 * crop[..., 0] + 0.5870 * crop[..., 1] + 0.1140 * crop[..., 2]).astype(np.float32)
    gray_valid = gray[eroded]
    gray_std = float(gray_valid.std())
    gray_ptp = float(gray_valid.max() - gray_valid.min())

    # Quantized colors (16-step bins -> 4096 bins to ignore vertex interpolation noise)
    fg_rgb = crop[eroded, :3]
    step = max(1, len(fg_rgb) // 5000)
    q = fg_rgb[::step] // 16
    uniq_q = len(np.unique(q, axis=0))

    # Smoothed gradient & feature edges (sigma=1.2 ignores vertex raster noise, keeps real edges)
    gray_smooth = gaussian_filter(gray, sigma=1.2)
    gy, gx = np.gradient(gray_smooth)
    grad = np.hypot(gy, gx)
    mean_grad = float(grad[eroded].mean())
    p95_grad = float(np.percentile(grad[eroded], 95))
    edge_density = float((grad[eroded] > 3.0).mean() * 100)

    if ch >= 5 and cw >= 5:
        lap = (
            gray_smooth[1:-1, 2:]
            + gray_smooth[1:-1, :-2]
            + gray_smooth[2:, 1:-1]
            + gray_smooth[:-2, 1:-1]
            - 4.0 * gray_smooth[1:-1, 1:-1]
        )
        lap_var = float(lap.var())
    else:
        lap_var = 0.0

    # Lag-1 spatial autocorrelation to detect pure uncorrelated high-frequency noise
    autocorr = 1.0
    if ch >= 10 and cw >= 10 and gray_std > 8.0:
        crop_inner = gray[4:-4, 4:-4] if ch > 10 and cw > 10 else gray
        x0 = crop_inner[:, :-1].ravel()
        x1 = crop_inner[:, 1:].ravel()
        s0, s1 = np.std(x0), np.std(x1)
        if s0 > 1e-3 and s1 > 1e-3:
            corr_h = np.corrcoef(x0, x1)[0, 1]
            y0 = crop_inner[:-1, :].ravel()
            y1 = crop_inner[1:, :].ravel()
            s_y0, s_y1 = np.std(y0), np.std(y1)
            corr_v = np.corrcoef(y0, y1)[0, 1] if (s_y0 > 1e-3 and s_y1 > 1e-3) else corr_h
            autocorr = float((corr_h + corr_v) / 2.0)

    metrics = {
        "crop_size": [cw, ch],
        "gray_std": round(gray_std, 2),
        "gray_ptp": round(gray_ptp, 2),
        "unique_colors": uniq_q,
        "mean_grad": round(mean_grad, 3),
        "p95_grad": round(p95_grad, 3),
        "edge_density": round(edge_density, 2),
        "lap_var": round(lap_var, 4),
        "autocorr": round(autocorr, 3),
        "fill_pct": round(fill_pct, 1),
    }

    # 1. Pure noise check:
    if autocorr < min_autocorr and gray_std > 8.0:
        return (
            False,
            f"pure_noise: low_spatial_autocorrelation ({autocorr:.3f} < {min_autocorr:.3f})",
            metrics,
        )

    # 2. Single-color checks:
    if uniq_q < min_unique_colors:
        return (
            False,
            f"single_color: low_color_diversity ({uniq_q} < {min_unique_colors})",
            metrics,
        )

    if gray_std < min_color_std:
        return (
            False,
            f"single_color: low_contrast (gray_std={gray_std:.2f} < {min_color_std:.2f})",
            metrics,
        )

    if gray_ptp < min_dynamic_range:
        return (
            False,
            f"single_color: low_dynamic_range (gray_ptp={gray_ptp:.1f} < {min_dynamic_range:.1f})",
            metrics,
        )

    # 3. No recognizable features (flat or blurry surfaces):
    if uniq_q < 28 and p95_grad < min_p95_grad:
        return (
            False,
            f"no_recognizable_features: flat_or_blurry_surface (uniq_q={uniq_q} < 28, p95_grad={p95_grad:.2f} < {min_p95_grad:.2f})",
            metrics,
        )

    if uniq_q < 20 and edge_density < min_edge_density and mean_grad < 0.35:
        return (
            False,
            f"no_recognizable_features: low_feature_density (uniq_q={uniq_q} < 20, edge_density={edge_density:.2f}% < {min_edge_density:.2f}%, mean_grad={mean_grad:.2f} < 0.35)",
            metrics,
        )

    if gray_std < 6.0 and uniq_q < 15:
        return (
            False,
            f"no_recognizable_features: low_contrast_and_palette (gray_std={gray_std:.1f} < 6.0, uniq_q={uniq_q} < 15)",
            metrics,
        )

    # 4. Noise floater:
    if fill_pct < 85.0 and gray_std < 18.0 and p95_grad < 1.5:
        return (
            False,
            f"noise_floater: ragged_mesh_artifact (fill_pct={fill_pct:.1f}% < 85%, p95_grad={p95_grad:.2f} < 1.5)",
            metrics,
        )

    return True, "valid", metrics


def get_camera_up_vector(
    f_vec: np.ndarray,
    front_dir_h: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Return appropriate up vector for camera given the look direction f_vec.
    - If looking almost straight down (overhead view, |f_vec[1]| > 0.85),
      use canonical horizontal orientation (-front_dir_h or [0, 0, -1]) so the top view
      is oriented with object's head/rear at the top of the image.
    - Otherwise use world up [0, 1, 0].
    """
    if abs(float(f_vec[1])) > 0.85:
        if front_dir_h is not None:
            if len(front_dir_h) == 2:
                up_h = -np.array([front_dir_h[0], 0.0, front_dir_h[1]], dtype=np.float64)
            else:
                up_h = -np.array([front_dir_h[0], 0.0, front_dir_h[2]], dtype=np.float64)
            up_norm = np.linalg.norm(up_h)
            if up_norm > 1e-4:
                return up_h / up_norm
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return np.array([0.0, 1.0, 0.0], dtype=np.float64)


def detect_scene_floor_levels(
    levels_manifest_path: Optional[Path] = None,
    structural_json_path: Optional[Path] = None,
    vertices: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Detect floor heights in the scene and return a sorted list of levels:
    [{"level_index": 0, "floor_y": -1.2285}, {"level_index": 1, "floor_y": 1.0277}, ...]
    """
    # 1. Try loading from structural_levels_manifest.json
    if levels_manifest_path and levels_manifest_path.exists():
        try:
            with open(levels_manifest_path, "r", encoding="utf-8") as f:
                ldata = json.load(f)
            levels = ldata.get("levels", [])
            if levels:
                extracted = []
                for lvl in levels:
                    idx = int(lvl.get("level_index", len(extracted)))
                    floor_y = float(lvl.get("mean_floor_elevation_y", 0.0))
                    extracted.append({"level_index": idx, "floor_y": floor_y})
                extracted.sort(key=lambda x: x["floor_y"])
                for i, item in enumerate(extracted):
                    item["level_index"] = i
                return extracted
        except Exception as e:
            print(f"Warning: Failed to parse levels manifest at {levels_manifest_path}: {e}")

    # 2. Try inferring from structural_classes.json floor pieces
    if structural_json_path and structural_json_path.exists():
        try:
            with open(structural_json_path, "r", encoding="utf-8") as f:
                sdata = json.load(f)
            classifications = sdata.get("classifications") or sdata.get("classes", {})
            floor_elevations = []
            for cid, info in classifications.items():
                if is_structural_role(info.get("structural_role", "")):
                    if "floor" in info.get("structural_role", "").lower():
                        pieces = info.get("piece_breakdown", [])
                        for p in pieces:
                            h = p.get("h_mean")
                            if h is not None and np.isfinite(h):
                                floor_elevations.append(float(h))
            if floor_elevations:
                floor_elevations.sort()
                clusters = [[floor_elevations[0]]]
                for h in floor_elevations[1:]:
                    if abs(h - clusters[-1][-1]) <= 0.30:
                        clusters[-1].append(h)
                    else:
                        clusters.append([h])
                return [
                    {"level_index": i, "floor_y": float(np.mean(c))}
                    for i, c in enumerate(clusters)
                ]
        except Exception as e:
            print(f"Warning: Failed to parse floor elevations from structural json: {e}")

    # 3. Fallback: lowest vertex elevation
    if vertices is not None and len(vertices) > 0:
        return [{"level_index": 0, "floor_y": float(vertices[:, 1].min())}]

    return [{"level_index": 0, "floor_y": 0.0}]


def assign_instance_to_floor_level(
    inst_pts: np.ndarray,
    floor_levels: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Assign an instance to the floor level directly supporting it or closest below it.
    Returns the floor level dict: {"level_index": int, "floor_y": float}.
    """
    if not floor_levels:
        return {"level_index": 0, "floor_y": float(inst_pts[:, 1].min())}
    if len(floor_levels) == 1:
        return floor_levels[0]

    y_mean = float(inst_pts[:, 1].mean())

    # Candidate floor levels that are below or at the instance center
    candidates = [fl for fl in floor_levels if fl["floor_y"] <= (y_mean + 0.15)]
    if candidates:
        return max(candidates, key=lambda fl: fl["floor_y"])
    return min(floor_levels, key=lambda fl: fl["floor_y"])


def identify_instance_biggest_side(
    inst_pts: np.ndarray,
    room_center: np.ndarray,
    class_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Identify the biggest side (largest surface face) of an object instance.
    Evaluates:
      - Top face: area dx * dz, normal [0, 1, 0] (pointing up)
      - Front/back face: area dx * dy, normal along +/- Z (pointing towards open room)
      - Lateral side face: area dy * dz, normal along +/- X (pointing towards open room)
    """
    dims = inst_pts.max(axis=0) - inst_pts.min(axis=0)
    inst_center = (inst_pts.min(axis=0) + inst_pts.max(axis=0)) / 2.0
    dx, dy, dz = float(dims[0]), float(dims[1]), float(dims[2])

    a_top = dx * dz
    n_top = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    a_front = dx * dy
    to_room_z = float(room_center[2] - inst_center[2])
    n_front_z = 1.0 if to_room_z >= 0 else -1.0
    n_front = np.array([0.0, 0.0, n_front_z], dtype=np.float64)

    a_side = dy * dz
    to_room_x = float(room_center[0] - inst_center[0])
    n_side_x = 1.0 if to_room_x >= 0 else -1.0
    n_side = np.array([n_side_x, 0.0, 0.0], dtype=np.float64)

    faces = [
        ("top", a_top, n_top, True),
        ("front_z", a_front, n_front, False),
        ("side_x", a_side, n_side, False),
    ]
    faces.sort(key=lambda item: item[1], reverse=True)
    best_name, best_area, best_normal, is_top = faces[0]

    return {
        "side_name": best_name,
        "area_m2": round(best_area, 4),
        "normal": best_normal,
        "is_horizontal_top": is_top,
        "dimensions": dims,
    }


def compute_instance_frontal_direction(
    inst_pts: np.ndarray,
    room_center: np.ndarray,
    room_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    class_id: Optional[int] = None,
) -> np.ndarray:
    """
    Compute the canonical frontal horizontal unit vector for an instance.
    Uses 2D PCA on the horizontal (X, Z) plane to find the object's principal face.
    For elongated/planar objects (wardrobes, desks, beds, doors), selects the normal
    vector pointing towards the open room interior (greater distance to perimeter walls).
    """
    # For bed / bed cover (classes 2, 13), headboard is at min Z, foot is at max Z (+Z facing room)
    if class_id in (2, 13):
        return np.array([0.0, 0.0, 1.0])

    pts_xz = inst_pts[:, [0, 2]]
    inst_center_xz = (pts_xz.min(axis=0) + pts_xz.max(axis=0)) / 2.0
    to_room_xz = room_center[[0, 2]] - inst_center_xz
    to_room_dist = np.linalg.norm(to_room_xz)
    if to_room_dist > 0.05:
        to_room_xz /= to_room_dist
    else:
        to_room_xz = np.array([0.0, 1.0])

    if len(pts_xz) < 3:
        return np.array([to_room_xz[0], 0.0, to_room_xz[1]])

    # Check if object has a head-to-foot gradient (e.g. bed with pillows at one end):
    if len(inst_pts) >= 10:
        # Check Z axis gradient in Y
        pts_z = inst_pts[:, 2]
        pts_y = inst_pts[:, 1]
        z_mid = float(np.median(pts_z))
        y_neg_z = float(np.mean(pts_y[pts_z < z_mid]))
        y_pos_z = float(np.mean(pts_y[pts_z >= z_mid]))
        # Check X axis gradient in Y
        pts_x = inst_pts[:, 0]
        x_mid = float(np.median(pts_x))
        y_neg_x = float(np.mean(pts_y[pts_x < x_mid]))
        y_pos_x = float(np.mean(pts_y[pts_x >= x_mid]))

        z_diff = y_neg_z - y_pos_z
        x_diff = y_neg_x - y_pos_x
        # If there is a distinct elevation gradient (e.g. pillows at head > 0.025m higher than foot):
        if abs(z_diff) >= 0.025 and abs(z_diff) > abs(x_diff):
            # Pillows are at head (higher Y). Frontal view is looking from foot towards head:
            # Vector pointing from object center towards the foot (where camera should be):
            front_z = 1.0 if z_diff > 0 else -1.0
            return np.array([0.0, 0.0, front_z])
        elif abs(x_diff) >= 0.025:
            front_x = 1.0 if x_diff > 0 else -1.0
            return np.array([front_x, 0.0, 0.0])

    cov = np.cov(pts_xz, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    minor = eigvecs[:, 0]

    # For planar/elongated objects, select the normal direction with greater clearance:
    if room_bounds is not None:
        r_min, r_max = room_bounds
        def dist_to_box(v):
            lims = []
            for i, coord in enumerate([0, 2]):
                if v[i] > 1e-5:
                    lims.append((r_max[coord] - inst_center_xz[i]) / v[i])
                elif v[i] < -1e-5:
                    lims.append((r_min[coord] - inst_center_xz[i]) / v[i])
            return min(lims) if lims else 0.0

        d_pos = dist_to_box(minor)
        d_neg = dist_to_box(-minor)
        front_xz = minor if d_pos >= d_neg else -minor
    else:
        dot = float(np.dot(minor, to_room_xz))
        front_xz = minor if dot >= 0 else -minor

    front_norm = np.linalg.norm(front_xz)
    if front_norm > 1e-6:
        front_xz /= front_norm
    else:
        front_xz = np.array([0.0, 1.0])

    return np.array([front_xz[0], 0.0, front_xz[1]])


def get_standard_elevation_deg(role: str, dims: np.ndarray) -> float:
    """
    Return a canonical standard elevation angle tailored to the object geometry.
    - Vertical objects/doors/wardrobes: ~10 deg (frontal eye-level with minimal keystoning).
    - Horizontal furniture/desks/beds: ~18 deg (reveals top surface depth without grazing collapse).
    - Generic objects/props: ~14 deg.
    """
    role_lower = (role or "").lower()
    h_span = max(float(dims[0]), float(dims[2]))
    v_span = float(dims[1])

    if "vertical" in role_lower or "door" in role_lower or (v_span > 1.25 * h_span):
        return 10.0
    elif "horizontal" in role_lower or (v_span < 0.55 * h_span):
        return 18.0
    else:
        return 14.0


def cull_foreground_occluders(
    all_vertices: np.ndarray,
    all_triangles: np.ndarray,
    all_vertex_colors: np.ndarray,
    labels: np.ndarray,
    class_id: int,
    cam_eye: np.ndarray,
    cam_target: np.ndarray,
    up: np.ndarray,
    inst_pts: np.ndarray,
    margin_m: float = 0.08,
    structural_cids: Optional[set[int]] = None,
) -> Tuple[o3d.geometry.TriangleMesh, int, List[int]]:
    """
    Selectively cull foreground occluders that block the target instance from view.

    Rules strictly enforced:
    1. Do NOT remove triangles when removing them creates holes in the 2D projection (background voids).
    2. CAN and SHALL remove triangles if doing so does NOT create holes and improves coverage of the instance
       (e.g. mezzanine ceiling blocking a piece of the view, or foreground props/punching bag in front).
    3. An occluding element can be safely culled if:
       - Direct unblocking: A ray from the camera through the triangle hits the target instance behind it
         (meaning the target instance itself covers that area behind it, so removing the occluder creates NO hole).
       - Full foreground object culling: An entire foreground class (e.g. punching bag) can only be removed
         if >95% of rays cast through it hit solid background/instance geometry behind it (zero voids/holes).
       - Bench/supporting surfaces (e.g. CID 24, 27) that are part of the foreground/middle-ground environment
         must NOT be culled if removing them exposes empty background voids.
    """
    f_vec = (cam_target - cam_eye) / np.linalg.norm(cam_target - cam_eye)
    r_vec = np.cross(f_vec, up)
    r_vec /= max(1e-6, np.linalg.norm(r_vec))
    u_vec = np.cross(r_vec, f_vec)

    inst_rel = inst_pts - cam_eye
    inst_z = inst_rel @ f_vec
    inst_x = inst_rel @ r_vec
    inst_y = inst_rel @ u_vec

    z_min_inst = float(inst_z.min())

    # Normalized camera projection bounds of target instance
    in_pos_z = inst_z > 0.05
    if not np.any(in_pos_z):
        # Fallback: keep full scene
        render_m = o3d.geometry.TriangleMesh()
        render_m.vertices = o3d.utility.Vector3dVector(all_vertices)
        render_m.triangles = o3d.utility.Vector3iVector(all_triangles)
        if len(all_vertex_colors) > 0:
            render_m.vertex_colors = o3d.utility.Vector3dVector(all_vertex_colors)
        return render_m, 0, []

    u_proj = inst_x[in_pos_z] / inst_z[in_pos_z]
    v_proj = inst_y[in_pos_z] / inst_z[in_pos_z]
    u_min, u_max = float(u_proj.min()), float(u_proj.max())
    v_min, v_max = float(v_proj.min()), float(v_proj.max())

    u_margin = 0.02 * (u_max - u_min)
    v_margin = 0.02 * (v_max - v_min)

    tri_centers = all_vertices[all_triangles].mean(axis=1).astype(np.float32)
    t_rel = tri_centers - cam_eye
    t_z = t_rel @ f_vec
    t_x = t_rel @ r_vec
    t_y = t_rel @ u_vec

    t_u = t_x / np.maximum(t_z, 1e-4)
    t_v = t_y / np.maximum(t_z, 1e-4)

    # Identify candidate triangles strictly in front of the instance in camera coordinates.
    # Require at least 0.30m separation in front of the instance (margin_m=0.30) so that adjacent parts,
    # bed frames, headboards, and immediate supporting surfaces of the object are never culled.
    cand_fg_mask = (
        (labels != class_id) &
        (t_z > 0.05) &
        (t_z < (z_min_inst - max(0.30, margin_m))) &
        (t_u >= (u_min - u_margin)) & (t_u <= (u_max + u_margin)) &
        (t_v >= (v_min - v_margin)) & (t_v <= (v_max + v_margin))
    )

    culled_tri_indices: set[int] = set()

    if np.any(cand_fg_mask):
        # 1. Build raycasting scene of the target instance to test direct occlusion
        # Build mesh for target instance
        m_inst_tris = all_triangles[labels == class_id]
        if len(m_inst_tris) > 0:
            u_vi, remap_i = np.unique(m_inst_tris, return_inverse=True)
            m_inst = o3d.geometry.TriangleMesh()
            m_inst.vertices = o3d.utility.Vector3dVector(all_vertices[u_vi])
            m_inst.triangles = o3d.utility.Vector3iVector(remap_i.reshape(m_inst_tris.shape))
            scene_inst = o3d.t.geometry.RaycastingScene()
            scene_inst.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m_inst))

            cand_indices = np.where(cand_fg_mask)[0]
            cand_centers = tri_centers[cand_indices]
            dirs = cand_centers - cam_eye.astype(np.float32)
            dir_norms = np.linalg.norm(dirs, axis=1, keepdims=True)
            dirs /= np.maximum(dir_norms, 1e-6)
            rays = np.hstack([np.tile(cam_eye.astype(np.float32), (len(cand_centers), 1)), dirs])

            ans = scene_inst.cast_rays(o3d.core.Tensor(rays))
            t_hit = ans["t_hit"].numpy()
            hits_instance = ~np.isinf(t_hit)

            # Triangles that directly hit the instance behind them can be culled with ZERO hole created:
            # (e.g. mezzanine ceiling CID 1 or -1 blocking Class 29)
            direct_occluders = cand_indices[hits_instance]
            for idx in direct_occluders:
                culled_tri_indices.add(int(idx))

        # 2. Check full class culling for foreground objects (e.g. punching bag CID 92)
        # ONLY if >95% of rays through the object hit something behind it (NO void/hole in 2D projection)
        cand_cids = np.unique(labels[cand_fg_mask])
        struct_set = structural_cids or set()

        for ocid in cand_cids:
            if ocid <= 0 or ocid in struct_set:
                continue  # Never cull entire structural classes like walls or ceilings as a whole
            ocid_tris_mask = (labels == ocid)
            total_ocid = int(np.count_nonzero(ocid_tris_mask))
            in_fg_count = int(np.count_nonzero(cand_fg_mask & ocid_tris_mask))

            # Must be predominantly in foreground
            if in_fg_count / max(1, total_ocid) >= 0.40:
                ocid_centers = tri_centers[ocid_tris_mask]
                ocid_z = (ocid_centers - cam_eye.astype(np.float32)) @ f_vec
                if np.median(ocid_z) < z_min_inst:
                    # Test if removing ocid creates a hole in the 2D view:
                    # Cast rays to the scene WITHOUT ocid
                    mesh_rest_tris = all_triangles[~ocid_tris_mask]
                    u_vr, remapr = np.unique(mesh_rest_tris, return_inverse=True)
                    mesh_rest = o3d.geometry.TriangleMesh()
                    mesh_rest.vertices = o3d.utility.Vector3dVector(all_vertices[u_vr])
                    mesh_rest.triangles = o3d.utility.Vector3iVector(remapr.reshape(mesh_rest_tris.shape))
                    scene_rest = o3d.t.geometry.RaycastingScene()
                    scene_rest.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_rest))

                    ocid_dirs = ocid_centers - cam_eye.astype(np.float32)
                    ocid_dirs /= np.maximum(np.linalg.norm(ocid_dirs, axis=1, keepdims=True), 1e-6)
                    ocid_rays = np.hstack([np.tile(cam_eye.astype(np.float32), (len(ocid_centers), 1)), ocid_dirs])

                    ans_rest = scene_rest.cast_rays(o3d.core.Tensor(ocid_rays))
                    t_hit_rest = ans_rest["t_hit"].numpy()
                    hits_rest = ~np.isinf(t_hit_rest)
                    hit_ratio = float(np.sum(hits_rest)) / max(1, len(hits_rest))

                    # If >90% of rays hit background/instance behind, it is safe to cull without holes:
                    if hit_ratio >= 0.90:
                        all_ocid_indices = np.where(ocid_tris_mask)[0]
                        for idx in all_ocid_indices:
                            culled_tri_indices.add(int(idx))

    culled_mask = np.zeros(len(all_triangles), dtype=bool)
    if culled_tri_indices:
        culled_mask[list(culled_tri_indices)] = True

    culled_cids = [int(c) for c in np.unique(labels[culled_mask]) if c != class_id]

    keep_mask = ~culled_mask
    keep_tris = all_triangles[keep_mask]
    u_v, remap_t = np.unique(keep_tris, return_inverse=True)

    render_m = o3d.geometry.TriangleMesh()
    render_m.vertices = o3d.utility.Vector3dVector(all_vertices[u_v])
    render_m.triangles = o3d.utility.Vector3iVector(remap_t.reshape(keep_tris.shape))
    if len(all_vertex_colors) > 0:
        render_m.vertex_colors = o3d.utility.Vector3dVector(all_vertex_colors[u_v])
    render_m.compute_vertex_normals()

    return render_m, int(np.sum(culled_mask)), culled_cids


def compute_non_structural_purity(
    cam_eye: np.ndarray,
    cam_look: np.ndarray,
    up: np.ndarray,
    inst_pts: np.ndarray,
    target_cid: int,
    other_instances: List[Dict[str, Any]],
    grid_size: int = 64,
    target_inst_id: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compute:
    1. non_structural_purity = A_target / (A_target + A_other_non_structural)
    2. target_visibility = A_target / A_target_total_projected
    using a vectorized z-buffered 2D grid within the bounding frustum of the target instance.
    """
    f_vec = cam_look / max(1e-6, np.linalg.norm(cam_look))
    r_vec = np.cross(f_vec, up)
    r_norm = float(np.linalg.norm(r_vec))
    if r_norm < 1e-4:
        return 0.0, 0.0
    r_vec /= r_norm
    u_vec = np.cross(r_vec, f_vec)
    u_vec /= max(1e-6, np.linalg.norm(u_vec))

    # Project target instance points
    p_rel = inst_pts - cam_eye
    p_z = p_rel @ f_vec
    p_x = p_rel @ r_vec
    p_y = p_rel @ u_vec

    valid = p_z > 0.05
    if not np.any(valid):
        return 0.0, 0.0

    p_x = p_x[valid]
    p_y = p_y[valid]
    p_z = p_z[valid]

    u_proj = p_x / p_z
    v_proj = p_y / p_z

    u0, u1 = float(u_proj.min()), float(u_proj.max())
    v0, v1 = float(v_proj.min()), float(v_proj.max())
    bw = u1 - u0
    bh = v1 - v0
    min_dim = min(bw, bh)
    box_pad = 0.10 * min_dim
    U0, U1 = u0 - box_pad, u1 + box_pad
    V0, V1 = v0 - box_pad, v1 + box_pad
    span_u = max(1e-6, U1 - U0)
    span_v = max(1e-6, V1 - V0)

    # Depth buffer & Label buffer
    flat_depth = np.full(grid_size * grid_size, np.inf, dtype=np.float32)
    flat_label = np.full(grid_size * grid_size, -1, dtype=np.int32)

    # Rasterize target instance points (vectorized z-buffer)
    gx = np.clip(((u_proj - U0) / span_u * grid_size).astype(np.int32), 0, grid_size - 1)
    gy = np.clip(((v_proj - V0) / span_v * grid_size).astype(np.int32), 0, grid_size - 1)
    flat_idx = gy * grid_size + gx

    order_t = np.argsort(p_z)
    sorted_pix_t = flat_idx[order_t]
    sorted_z_t = p_z[order_t]
    u_pix_t, first_idx_t = np.unique(sorted_pix_t, return_index=True)
    flat_depth[u_pix_t] = sorted_z_t[first_idx_t]
    flat_label[u_pix_t] = target_cid

    total_target_cells = int(np.count_nonzero(flat_label == target_cid))
    if total_target_cells == 0:
        return 0.0, 0.0

    # Rasterize other non-structural instances
    other_pts_list = []
    other_cids_list = []
    for other in other_instances:
        if other.get("class_id") == target_cid:
            if target_inst_id is None or other.get("instance_id") == target_inst_id:
                continue
        o_pts = other.get("pts")
        if o_pts is not None and len(o_pts) > 0:
            other_pts_list.append(o_pts)
            other_cids_list.append(np.full(len(o_pts), other["class_id"], dtype=np.int32))

    if other_pts_list:
        all_o_pts = np.vstack(other_pts_list)
        all_o_cids = np.concatenate(other_cids_list)

        o_rel = all_o_pts - cam_eye
        o_z = o_rel @ f_vec
        o_valid = o_z > 0.05
        if np.any(o_valid):
            all_o_cids = all_o_cids[o_valid]
            o_z = o_z[o_valid]
            o_x = (o_rel @ r_vec)[o_valid]
            o_y = (o_rel @ u_vec)[o_valid]

            o_u = o_x / o_z
            o_v = o_y / o_z
            in_box = (o_u >= U0) & (o_u <= U1) & (o_v >= V0) & (o_v <= V1)
            if np.any(in_box):
                o_u = o_u[in_box]
                o_v = o_v[in_box]
                o_z = o_z[in_box]
                all_o_cids = all_o_cids[in_box]

                ogx = np.clip(((o_u - U0) / span_u * grid_size).astype(np.int32), 0, grid_size - 1)
                ogy = np.clip(((o_v - V0) / span_v * grid_size).astype(np.int32), 0, grid_size - 1)
                o_flat_idx = ogy * grid_size + ogx

                order_o = np.argsort(o_z)
                sorted_pix = o_flat_idx[order_o]
                sorted_z = o_z[order_o]
                sorted_cid = all_o_cids[order_o]

                u_pix, first_idx = np.unique(sorted_pix, return_index=True)
                closest_z = sorted_z[first_idx]
                closest_cid = sorted_cid[first_idx]

                closer = closest_z < flat_depth[u_pix]
                target_pix = u_pix[closer]
                flat_depth[target_pix] = closest_z[closer]
                flat_label[target_pix] = closest_cid[closer]

    visible_target = int(np.count_nonzero(flat_label == target_cid))
    visible_other = int(np.count_nonzero((flat_label > 0) & (flat_label != target_cid)))

    purity = visible_target / max(1, (visible_target + visible_other))
    visibility = visible_target / max(1, total_target_cells)

    return float(purity), float(visibility)


def render_synthetic_in_context_frame(
    instance: Dict[str, Any],
    all_vertices: np.ndarray,
    all_triangles: np.ndarray,
    all_vertex_colors: np.ndarray,
    labels: np.ndarray,
    class_id: int,
    structural_cids: set[int],
    room_center: np.ndarray,
    role: str = "object",
    padding: float = 0.10,
    render_res: int = 1536,
    fov_deg: float = 50.0,
    elevation_deg: Optional[float] = None,
    wall_scene: Optional[o3d.t.geometry.RaycastingScene] = None,
    raycast_scene: Optional[o3d.t.geometry.RaycastingScene] = None,
    min_resolution: Optional[int] = None,
    padding_color: str = "black",
    floor_levels: Optional[List[Dict[str, Any]]] = None,
    human_height_offset: float = 1.50,
    min_wall_distance: float = 0.80,
    min_triangle_distance: float = 0.10,
    other_non_structural_instances: Optional[List[Dict[str, Any]]] = None,
    renderer: Optional[o3d.visualization.rendering.OffscreenRenderer] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Render a high-resolution synthetic in-context view of an instance.
    Features:
    1. Human-Level Height Camera Positioning:
       - Uses relative floor height of the instance's floor level + 1.50m.
    2. Wall Distance Clearance & Triangle Distance Clearance:
       - Enforces camera location to be at least 0.8m inside of the nearest wall.
       - Enforces camera location to be at least min_triangle_distance (default 0.10m = 10cm)
         away from the nearest triangle surface on the mesh.
    3. Biggest-Side Frontal Alignment:
       - Detects largest surface face (e.g. top view for bed, lateral top-down for benches, frontal for wardrobes).
       - Centers the instance and maximizes frontal alignment onto that biggest face.
    4. Controlled 3D Perspective (Pitch & Azimuth):
       - Pitch: 15 <= |pitch| <= 75 deg (avoids straight parallel to floor and perfectly downward).
       - Azimuth: between 5 and 20 deg (or -5 and -20 deg) off-orthogonal.
       - Roll: strictly 0.0 deg (up = [0, 1, 0], pure portrait or landscape).
    5. Context Preservation (No Blanket Deletion):
       - Keeps full scene mesh (walls, floor, ceiling, supporting surfaces).
    6. Selective Non-Structural Foreground Occlusion Culling:
       - Culls ONLY non-structural objects between camera and instance when safe.
    7. Tight In-Context Framing & Black/Transparent Square Padding:
       - Crops tightly around the instance with 10% min-dimension padding.
       - Pads the shorter dimension with black (or transparent) pixels to form a square.
    8. Ultra-Fast Batched Raycasting & Persistent Renderer Re-use.
    """
    inst_pts = instance["points"]
    inst_min = instance["bbox_min"]
    inst_max = instance["bbox_max"]
    inst_center = (inst_min + inst_max) / 2.0
    inst_dims = instance["dimensions"]
    radius = float(np.linalg.norm(inst_dims) / 2.0)

    # 1. Floor level & human eye height
    fl_list = floor_levels or [{"level_index": 0, "floor_y": float(all_vertices[:, 1].min())}]
    floor_level = assign_instance_to_floor_level(inst_pts, fl_list)
    relative_floor_y = float(floor_level["floor_y"])
    level_index = int(floor_level["level_index"])
    target_eye_y = relative_floor_y + human_height_offset

    room_pts_min = all_vertices.min(axis=0)
    room_pts_max = all_vertices.max(axis=0)
    eye_y_min = room_pts_min[1] + 0.35
    eye_y_max = room_pts_max[1] - 0.35
    eye_y = float(np.clip(target_eye_y, eye_y_min, eye_y_max))

    # 2. Identify biggest side and its outward normal
    biggest_side_info = identify_instance_biggest_side(inst_pts, room_center, class_id=class_id)
    side_name = biggest_side_info["side_name"]
    is_top_view = biggest_side_info["is_horizontal_top"]
    target_normal = biggest_side_info["normal"]

    # Canonical horizontal reference direction norm_h
    if is_top_view:
        to_room = room_center[[0, 2]] - inst_center[[0, 2]]
        to_room_len = float(np.linalg.norm(to_room))
        if to_room_len > 1e-4:
            norm_h = to_room / to_room_len
        else:
            norm_h = np.array([0.0, 1.0])
    else:
        norm_h_raw = np.array([target_normal[0], target_normal[2]], dtype=np.float64)
        norm_len = float(np.linalg.norm(norm_h_raw))
        if norm_len > 1e-4:
            norm_h = norm_h_raw / norm_len
        else:
            to_room = room_center[[0, 2]] - inst_center[[0, 2]]
            to_room_len = float(np.linalg.norm(to_room))
            norm_h = (to_room / to_room_len) if to_room_len > 1e-4 else np.array([0.0, 1.0])

    t_h = np.array([-norm_h[1], norm_h[0]])

    az_angles = [-18.0, -12.0, -6.0, 6.0, 12.0, 18.0]

    base_dist = max(1.0, 1.2 * radius)
    dists = [base_dist * f for f in [0.85, 1.15, 1.50]]
    dists = sorted(list(set([float(np.clip(round(d, 2), 0.80, 3.20)) for d in dists])))

    if inst_center[1] > (eye_y + 0.15):
        target_pitches = [18.0, 28.0, 42.0]
    else:
        target_pitches = [-18.0, -28.0, -42.0]

    active_scene = raycast_scene if raycast_scene is not None else wall_scene
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # Pre-filter other non-structural instances by spatial proximity (within 7m horizontal, 2.5m vertical)
    relevant_other_instances: List[Dict[str, Any]] = []
    if other_non_structural_instances:
        for other in other_non_structural_instances:
            if other.get("class_id") == class_id:
                if other.get("instance_id") == instance.get("instance_id"):
                    continue
            o_center = other.get("center")
            if o_center is not None:
                if abs(float(o_center[1]) - float(inst_center[1])) > 2.5:
                    continue
                d_horiz = float(np.linalg.norm(o_center[[0, 2]] - inst_center[[0, 2]]))
                if d_horiz > 7.0:
                    continue
            relevant_other_instances.append(other)

    # 1. Gather all candidate camera eye positions within room bounds
    cand_eyes_list = []
    eye_meta_list = []

    for dist in dists:
        for az in az_angles:
            rad_az = np.radians(az)
            dir_h_2d = norm_h * np.cos(rad_az) + t_h * np.sin(rad_az)
            cand_eye = np.array([
                inst_center[0] + dir_h_2d[0] * dist,
                eye_y,
                inst_center[2] + dir_h_2d[1] * dist,
            ])
            # Check basic perimeter bounds
            if (
                cand_eye[0] < room_pts_min[0] + 0.35
                or cand_eye[0] > room_pts_max[0] - 0.35
                or cand_eye[2] < room_pts_min[2] + 0.35
                or cand_eye[2] > room_pts_max[2] - 0.35
            ):
                continue

            d_wall = min(
                float(cand_eye[0] - room_pts_min[0]),
                float(room_pts_max[0] - cand_eye[0]),
                float(cand_eye[2] - room_pts_min[2]),
                float(room_pts_max[2] - cand_eye[2]),
            )

            cand_eyes_list.append(cand_eye)
            eye_meta_list.append((dir_h_2d, dist, az, d_wall))

    # Constraint: Camera must be at least min_triangle_distance (>= 10cm) away from nearest triangle
    valid_eyes = []
    valid_eye_meta = []

    if cand_eyes_list:
        cand_eyes_arr = np.array(cand_eyes_list, dtype=np.float32)
        if active_scene is not None:
            query_tensor = o3d.core.Tensor(cand_eyes_arr, dtype=o3d.core.Dtype.Float32)
            dists_to_mesh = active_scene.compute_distance(query_tensor).numpy()
        else:
            dists_to_mesh = np.full(len(cand_eyes_arr), 1.0, dtype=np.float32)

        for i in range(len(cand_eyes_list)):
            tri_dist = float(dists_to_mesh[i])
            if tri_dist >= min_triangle_distance:
                c_eye = cand_eyes_list[i]
                valid_eyes.append(c_eye)
                dir_h_2d, dist, az, d_wall = eye_meta_list[i]
                valid_eye_meta.append((dir_h_2d, dist, az, d_wall, tri_dist))

        if not valid_eyes and len(cand_eyes_list) > 0:
            # Fallback: select candidate with maximum distance to mesh
            best_idx = int(np.argmax(dists_to_mesh))
            c_eye = cand_eyes_list[best_idx]
            valid_eyes.append(c_eye)
            dir_h_2d, dist, az, d_wall = eye_meta_list[best_idx]
            valid_eye_meta.append((dir_h_2d, dist, az, d_wall, float(dists_to_mesh[best_idx])))

    # 2. For surviving camera positions, build pitch and look proposals
    proposal_data = []
    ray_batch = []

    for c_eye, (dir_h_2d, dist, az, d_wall, tri_dist) in zip(valid_eyes, valid_eye_meta):
        v_to_c = inst_center - c_eye
        d_to_c = float(np.linalg.norm(v_to_c))
        nat_pitch = float(np.degrees(np.arcsin(v_to_c[1] / max(1e-4, d_to_c))))

        pitches_to_test = []
        if 15.0 <= abs(nat_pitch) <= 75.0:
            pitches_to_test.append((nat_pitch, inst_center))

        for tp in target_pitches:
            rad_p = np.radians(tp)
            v_look_p = np.array([
                -dir_h_2d[0] * np.cos(rad_p),
                np.sin(rad_p),
                -dir_h_2d[1] * np.cos(rad_p),
            ])
            tgt_p = c_eye + v_look_p * d_to_c
            pitches_to_test.append((tp, tgt_p))

        for pitch_val, tgt_pt in pitches_to_test:
            v_look = tgt_pt - c_eye
            l_dist = float(np.linalg.norm(v_look))
            if l_dist < 1e-4:
                continue
            v_look /= l_dist

            proposal_data.append((c_eye, tgt_pt, v_look, pitch_val, az, dist, d_wall, nat_pitch, l_dist, tri_dist))
            ray_batch.append(np.hstack([c_eye, v_look]))

    # 3. Batched Raycasting for line-of-sight clearance (all rays in one SIMD call)
    if active_scene is not None and ray_batch:
        ray_tensor = o3d.core.Tensor(np.array(ray_batch, dtype=np.float32))
        ans = active_scene.cast_rays(ray_tensor)
        t_hits = ans["t_hit"].numpy()
    else:
        t_hits = np.full(len(proposal_data), np.inf, dtype=np.float32)

    raw_candidates_strict = []
    raw_candidates_all = []
    target_norm_arr = np.array([target_normal[0], target_normal[1], target_normal[2]], dtype=np.float32)

    for idx, (c_eye, tgt_pt, v_look, pitch_val, az, dist, d_wall, nat_pitch, l_dist, tri_dist) in enumerate(proposal_data):
        if t_hits[idx] < (l_dist - 0.25):
            continue  # occluded along line of sight

        pts_rel = inst_pts - c_eye
        f_vec = v_look
        r_vec = np.cross(f_vec, up)
        r_norm = float(np.linalg.norm(r_vec))
        if r_norm < 1e-4:
            continue
        r_vec /= r_norm
        u_vec = np.cross(r_vec, f_vec)
        u_vec /= np.maximum(float(np.linalg.norm(u_vec)), 1e-6)

        p_cam_x = pts_rel @ r_vec
        p_cam_y = pts_rel @ u_vec
        p_cam_z = pts_rel @ f_vec
        valid_z = p_cam_z > 0.05
        if not np.any(valid_z):
            continue

        theta_x = float(np.max(np.abs(p_cam_x[valid_z] / p_cam_z[valid_z])))
        theta_y = float(np.max(np.abs(p_cam_y[valid_z] / p_cam_z[valid_z])))
        req_fov = float(np.degrees(2.0 * np.arctan(max(np.tan(np.radians(25.0)), 1.35 * max(theta_x, theta_y)))))

        geom_score = (
            25.0 * min(d_wall, 1.20)
            + 15.0 * max(0.0, float(np.dot(v_look, -target_norm_arr)))
            - abs(dist - base_dist) * 8.0
            - abs(req_fov - 55.0) * 0.5
            - abs(pitch_val - nat_pitch) * 0.2
        )
        item = (geom_score, c_eye, tgt_pt, v_look, pitch_val, az, dist, req_fov, d_wall, tri_dist)
        raw_candidates_all.append(item)
        if d_wall >= min_wall_distance:
            raw_candidates_strict.append(item)

    # Sort geometrically best candidates and evaluate purity only on the top pool
    pool = raw_candidates_strict if raw_candidates_strict else raw_candidates_all
    pool.sort(key=lambda x: x[0], reverse=True)
    top_candidates = pool[:5]

    scored_candidates = []
    for geom_score, c_eye, tgt_pt, v_look, pitch_val, az, dist, req_fov, d_wall, tri_dist in top_candidates:
        purity, vis = compute_non_structural_purity(
            cam_eye=c_eye,
            cam_look=v_look,
            up=up,
            inst_pts=inst_pts,
            target_cid=class_id,
            other_instances=relevant_other_instances,
            grid_size=64,
            target_inst_id=instance.get("instance_id"),
        )
        total_score = geom_score + 120.0 * purity + 30.0 * vis
        scored_candidates.append((
            total_score, c_eye, tgt_pt, v_look, pitch_val, az, dist, req_fov, d_wall, purity, vis, tri_dist
        ))

    if scored_candidates:
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        (
            best_score,
            best_eye,
            best_target,
            best_look,
            best_pitch,
            best_az,
            best_dist,
            cam_fov_deg,
            best_d_wall,
            best_purity,
            best_vis,
            best_tri_dist,
        ) = scored_candidates[0]
    else:
        # Fallback candidate satisfying constraints
        best_az = 15.0
        rad_az = np.radians(best_az)
        dir_h_2d = norm_h * np.cos(rad_az) + t_h * np.sin(rad_az)
        best_eye = np.array([
            inst_center[0] + dir_h_2d[0] * base_dist,
            eye_y,
            inst_center[2] + dir_h_2d[1] * base_dist,
        ])
        best_pitch = -20.0 if inst_center[1] <= eye_y else 20.0
        rad_p = np.radians(best_pitch)
        best_look = np.array([
            -dir_h_2d[0] * np.cos(rad_p),
            np.sin(rad_p),
            -dir_h_2d[1] * np.cos(rad_p),
        ])
        best_look /= np.linalg.norm(best_look)
        best_target = best_eye + best_look * base_dist
        best_dist = base_dist
        cam_fov_deg = 50.0
        best_d_wall = min(
            float(best_eye[0] - room_pts_min[0]),
            float(room_pts_max[0] - best_eye[0]),
            float(best_eye[2] - room_pts_min[2]),
            float(room_pts_max[2] - best_eye[2]),
        )
        best_purity = 1.0
        best_vis = 1.0
        best_score = 100.0
        best_tri_dist = 0.50

    best_align = float(np.dot(best_look, -np.array([target_normal[0], target_normal[1], target_normal[2]]))) if not is_top_view else float(-best_look[1])

    # 3b. Verify FOV dynamically fits target instance with margin
    pts_rel = inst_pts - best_eye
    f_vec = (best_target - best_eye) / np.linalg.norm(best_target - best_eye)
    r_vec = np.cross(f_vec, up)
    r_norm = float(np.linalg.norm(r_vec))
    if r_norm > 1e-6:
        r_vec /= r_norm
    else:
        r_vec = np.array([1.0, 0.0, 0.0])
    u_vec = np.cross(r_vec, f_vec)
    u_vec /= np.maximum(float(np.linalg.norm(u_vec)), 1e-6)

    p_cam_x = pts_rel @ r_vec
    p_cam_y = pts_rel @ u_vec
    p_cam_z = pts_rel @ f_vec

    valid_z = p_cam_z > 0.05
    if np.any(valid_z):
        theta_x = float(np.max(np.abs(p_cam_x[valid_z] / p_cam_z[valid_z])))
        theta_y = float(np.max(np.abs(p_cam_y[valid_z] / p_cam_z[valid_z])))
        theta_max = max(theta_x, theta_y)
        req_fov_rad = 2.0 * np.arctan(max(np.tan(np.radians(fov_deg / 2.0)), 1.35 * theta_max))
        cam_fov_deg = float(np.degrees(req_fov_rad))

    culled_count = 0
    culled_cids = []

    # 4. Render with persistent renderer (fast) or temporary fallback renderer
    if renderer is not None:
        renderer.setup_camera(cam_fov_deg, best_target.tolist(), best_eye.tolist(), up.tolist())
        img_o3d = renderer.render_to_image()
        raw_arr = np.asarray(img_o3d)
    else:
        render_mesh = o3d.geometry.TriangleMesh()
        render_mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
        render_mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
        if all_vertex_colors is not None and len(all_vertex_colors) > 0:
            render_mesh.vertex_colors = o3d.utility.Vector3dVector(all_vertex_colors)
        render_mesh.compute_vertex_normals()

        temp_renderer = o3d.visualization.rendering.OffscreenRenderer(render_res, render_res)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        temp_renderer.scene.set_background([0.0, 0.0, 0.0, 1.0], None)
        temp_renderer.scene.add_geometry("scene", render_mesh, mat)
        temp_renderer.setup_camera(cam_fov_deg, best_target.tolist(), best_eye.tolist(), up.tolist())

        img_o3d = temp_renderer.render_to_image()
        raw_arr = np.asarray(img_o3d)

    # 5. Project instance points to 2D image coordinates to find tight bounding box
    fov_rad = np.radians(cam_fov_deg)
    fy = (render_res / 2.0) / np.tan(fov_rad / 2.0)
    fx = fy
    cx, cy = render_res / 2.0, render_res / 2.0

    in_f = p_cam_z > 0.05
    u_proj = fx * (p_cam_x[in_f] / p_cam_z[in_f]) + cx
    v_proj = cy - fy * (p_cam_y[in_f] / p_cam_z[in_f])

    u0, u1 = float(u_proj.min()), float(u_proj.max())
    v0, v1 = float(v_proj.min()), float(v_proj.max())
    bw = u1 - u0
    bh = v1 - v0

    # 6. Rectangular bounding box with padding along both dimensions:
    min_dim = min(bw, bh)
    box_pad = padding * min_dim
    cu0 = max(0, int(np.floor(u0 - box_pad)))
    cu1 = min(render_res, int(np.ceil(u1 + box_pad)))
    cv0 = max(0, int(np.floor(v0 - box_pad)))
    cv1 = min(render_res, int(np.ceil(v1 + box_pad)))

    crop_arr = raw_arr[cv0:cv1, cu0:cu1]
    crop_h, crop_w = crop_arr.shape[:2]

    # 7. Square padding with black (or transparent) pixels if square format is requested
    square_png = pad_crop_to_square(
        crop_arr, min_side=min_resolution, padding_color=padding_color
    )

    meta = {
        "floor_level": level_index,
        "relative_floor_elevation_m": round(relative_floor_y, 4),
        "target_human_height_m": round(human_height_offset, 2),
        "actual_camera_height_above_floor_m": round(float(best_eye[1] - relative_floor_y), 3),
        "biggest_side": side_name,
        "biggest_side_area_m2": biggest_side_info["area_m2"],
        "biggest_side_alignment": round(best_align, 3),
        "is_top_view": is_top_view,
        "camera_eye": [round(float(v), 4) for v in best_eye],
        "camera_target": [round(float(v), 4) for v in best_target],
        "camera_distance_m": round(float(best_dist), 3),
        "camera_distance_to_nearest_wall_m": round(float(best_d_wall), 3),
        "camera_distance_to_nearest_triangle_m": round(float(best_tri_dist), 3),
        "camera_pitch_deg": round(float(best_pitch), 1),
        "camera_azimuth_deg": round(float(best_az), 1),
        "camera_roll_deg": 0.0,
        "frame_orientation": "portrait" if crop_h >= crop_w else "landscape",
        "camera_fov_deg": round(cam_fov_deg, 1),
        "purity_ratio": round(float(best_purity), 3),
        "target_visibility": round(float(best_vis), 3),
        "candidate_score": round(float(best_score), 2),
        "culled_foreground_triangles": culled_count,
        "culled_foreground_cids": culled_cids,
        "raw_render_res": render_res,
        "crop_coords": [cu0, cv0, cu1, cv1],
        "crop_size": [crop_w, crop_h],
        "square_size": square_png.size[0],
        "padding_color": padding_color,
        "is_square_padded_transparent": (padding_color == "transparent"),
        "aspect_ratio_orig": round(bw / max(1.0, bh), 3),
    }

    return square_png, meta



def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 2D frames enclosing non-structural class instances from original mesh and Gaussian Splats."
    )
    parser.add_argument(
        "--mesh-path",
        type=Path,
        default=backend_dir / "current_scene/06_mesh_elaboration/original_mesh.ply",
        help="Path to original colored mesh PLY",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=backend_dir / "current_scene/06_mesh_elaboration/generic_graph_regions/triangle_class_labels.npy",
        help="Path to triangle class labels array (.npy)",
    )
    parser.add_argument(
        "--structural-json",
        type=Path,
        default=backend_dir / "current_scene/06_mesh_elaboration/generic_graph_regions/structural_classes.json",
        help="Path to structural classes JSON metadata",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=backend_dir / "current_scene",
        help="Path to scene workspace",
    )
    parser.add_argument(
        "--output-dir",
        "--mesh-output-dir",
        type=Path,
        dest="output_dir",
        default=backend_dir / "current_scene/06_mesh_elaboration/non_structural_frames",
        help="Output directory for extracted mesh images and manifest",
    )
    parser.add_argument(
        "--gs-output-dir",
        type=Path,
        default=None,
        help="Output directory for extracted Gaussian Splat images (default: <workspace>/06_mesh_elaboration/non_structural_frames_gs)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to GS checkpoint.pt (default: auto-detected in workspace)",
    )
    parser.add_argument(
        "--skip-gs",
        action="store_true",
        help="Skip rendering from Gaussian Splat (mesh only)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch device for Gaussian Splat rasterization (default: cuda)",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.10,
        help="Padding fraction along min dimension of bounding box (default: 0.10 for 10%)",
    )
    parser.add_argument(
        "--render-res",
        type=int,
        default=1536,
        help="High-quality offscreen rendering resolution (default: 1536)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=None,
        help="Optional square output image size (e.g. 1024); if None, preserves highest native square resolution",
    )
    parser.add_argument(
        "--min-resolution",
        type=int,
        default=None,
        help="Minimum resolution required; upsamples largest dimension and pads other dimension with black/transparent pixels",
    )
    parser.add_argument(
        "--padding-color",
        type=str,
        choices=["black", "transparent"],
        default="black",
        help="Padding color for square canvas: 'black' (default, RGB) or 'transparent' (RGBA)",
    )
    parser.add_argument(
        "--min-instance-faces",
        type=int,
        default=30,
        help="Minimum faces to qualify as an instance component",
    )
    parser.add_argument(
        "--min-instance-area",
        type=float,
        default=0.005,
        help="Minimum area (m^2) to qualify as an instance component",
    )
    parser.add_argument(
        "--levels-manifest",
        type=Path,
        default=backend_dir / "current_scene/06_mesh_elaboration/extracted_structural_instances/structural_levels_manifest.json",
        help="Path to structural levels manifest JSON",
    )
    parser.add_argument(
        "--human-height-offset",
        type=float,
        default=1.50,
        help="Target camera height in meters above relative floor level (default: 1.50m)",
    )
    parser.add_argument(
        "--min-wall-distance",
        type=float,
        default=0.80,
        help="Minimum required clearance in meters from the nearest wall for camera placement (default: 0.80m)",
    )
    parser.add_argument(
        "--min-triangle-distance",
        type=float,
        default=0.10,
        help="Minimum required clearance in meters from the nearest mesh triangle for camera placement (default: 0.10m)",
    )
    parser.add_argument(
        "--enable-quality-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable quality filter to discard featureless/single-color/noise images (default: True).",
    )
    parser.add_argument(
        "--mesh-discarded-dir",
        type=Path,
        default=None,
        help="Directory to save discarded mesh images (default: <output_dir>_discarded).",
    )
    parser.add_argument(
        "--gs-discarded-dir",
        type=Path,
        default=None,
        help="Directory to save discarded GS images (default: <gs_output_dir>_discarded).",
    )
    parser.add_argument(
        "--min-unique-colors",
        type=int,
        default=8,
        help="Minimum unique colors required in foreground to avoid single-color discard (default: 8).",
    )
    parser.add_argument(
        "--min-dynamic-range",
        type=float,
        default=15.0,
        help="Minimum dynamic range in foreground to avoid single-color discard (default: 15.0).",
    )
    parser.add_argument(
        "--min-color-std",
        type=float,
        default=3.5,
        help="Minimum color std in foreground to avoid single-color discard (default: 3.5).",
    )
    parser.add_argument(
        "--min-laplacian-var",
        type=float,
        default=0.05,
        help="Minimum Laplacian variance for feature detection (default: 0.05).",
    )
    parser.add_argument(
        "--min-fg-pixels",
        type=int,
        default=50,
        help="Minimum number of foreground pixels required (default: 50).",
    )
    parser.add_argument(
        "--min-autocorr",
        type=float,
        default=0.35,
        help="Minimum lag-1 autocorrelation to prevent pure noise (default: 0.35).",
    )
    parser.add_argument(
        "--min-p95-grad",
        type=float,
        default=1.25,
        help="Minimum 95th percentile gradient to reject flat/blurry surfaces (default: 1.25).",
    )
    parser.add_argument(
        "--min-edge-density",
        type=float,
        default=1.0,
        help="Minimum percent of pixels with gradient > 3.0 to reject featureless surfaces (default: 1.0).",
    )
    return parser.parse_args(argv)


def run_extraction(
    mesh_path: Path,
    labels_path: Path,
    structural_json_path: Path,
    workspace: Path,
    output_dir: Path,
    gs_output_dir: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    skip_gs: bool = False,
    device: str = "cuda",
    levels_manifest_path: Optional[Path] = None,
    human_height_offset: float = 1.50,
    min_wall_distance: float = 0.80,
    min_triangle_distance: float = 0.10,
    padding: float = 0.10,
    render_res: int = 1536,
    target_size: Optional[int] = None,
    min_resolution: Optional[int] = None,
    padding_color: str = "black",
    min_instance_faces: int = 30,
    min_instance_area: float = 0.005,
    enable_quality_check: bool = True,
    mesh_discarded_dir: Optional[Path] = None,
    gs_discarded_dir: Optional[Path] = None,
    min_unique_colors: int = 8,
    min_dynamic_range: float = 15.0,
    min_color_std: float = 3.5,
    min_laplacian_var: float = 0.05,
    min_fg_pixels: int = 50,
    min_autocorr: float = 0.35,
    min_p95_grad: float = 1.25,
    min_edge_density: float = 1.0,
) -> int:
    t_start = time.perf_counter()

    if not mesh_path.exists():
        print(f"Error: Mesh not found at {mesh_path}")
        return 1
    if not labels_path.exists():
        print(f"Error: Labels not found at {labels_path}")
        return 1
    if not structural_json_path.exists():
        print(f"Error: Structural JSON not found at {structural_json_path}")
        return 1

    effective_min_res = min_resolution or target_size

    mesh_output_dir = output_dir
    mesh_output_dir.mkdir(parents=True, exist_ok=True)

    if gs_output_dir is None:
        gs_output_dir = workspace / "06_mesh_elaboration" / "non_structural_frames_gs"

    if mesh_discarded_dir is None:
        mesh_discarded_dir = mesh_output_dir.parent / (mesh_output_dir.name + "_discarded")
    if gs_discarded_dir is None:
        gs_discarded_dir = gs_output_dir.parent / (gs_output_dir.name + "_discarded")

    enable_gs = False
    splats = None
    sh_degree = 3
    ckpt_file = None

    if not skip_gs:
        if not HAS_GS_RENDERER:
            logger.info("Gaussian Splat renderer modules not available; skipping GS extraction.")
        else:
            try:
                ckpt_file = resolve_gs_checkpoint_path(workspace, checkpoint_path)
                logger.info(f"Loading Gaussian Splats from {ckpt_file} onto {device}...")
                splats, sh_degree = load_gaussian_splats(ckpt_file, device=device)
                enable_gs = True
                gs_output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Loaded {len(splats['means']):,} Gaussians (SH degree {sh_degree}).")
            except Exception as e:
                logger.info(f"Gaussian Splat checkpoint not loaded ({e}); proceeding with mesh only.")
                enable_gs = False

    print(f"Loading mesh: {mesh_path}...")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertex_colors = np.asarray(mesh.vertex_colors)
    labels = np.load(labels_path)

    if len(labels) != len(triangles):
        print(f"Error: Label count ({len(labels)}) does not match triangle count ({len(triangles)})")
        return 1
    print(f"Loaded mesh: {len(vertices):,} vertices, {len(triangles):,} triangles")

    # Parse structural and non-structural classes
    with open(structural_json_path, "r", encoding="utf-8") as f:
        sdata = json.load(f)
    classifications = sdata.get("classifications", {})

    structural_cids = {
        int(cid) for cid, info in classifications.items()
        if is_structural_role(info.get("structural_role", ""))
    }
    non_structural_classes = identify_non_structural_classes(structural_json_path, labels)
    sorted_cids = sorted(non_structural_classes.keys())
    print(f"Identified {len(structural_cids)} structural classes: {sorted(structural_cids)}")
    print(f"Identified {len(sorted_cids)} non-structural classes: {sorted(sorted_cids)}")

    # Detect scene floor levels
    floor_levels = detect_scene_floor_levels(
        levels_manifest_path=levels_manifest_path,
        structural_json_path=structural_json_path,
        vertices=vertices,
        labels=labels,
    )
    print(f"Detected {len(floor_levels)} floor level(s):")
    for fl in floor_levels:
        eye_lvl_h = fl["floor_y"] + human_height_offset
        print(
            f"  Level {fl['level_index']}: floor elevation {fl['floor_y']:.3f}m -> "
            f"human camera eye height {eye_lvl_h:.3f}m"
        )

    # Build RaycastingScene for line-of-sight clearance (structural and non-structural)
    raycast_scene = None
    try:
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        raycast_scene = o3d.t.geometry.RaycastingScene()
        raycast_scene.add_triangles(mesh_t)
        # Warm up BVH distance query structure to avoid first-query BVH building lag
        _ = raycast_scene.compute_distance(o3d.core.Tensor([[0.0, 0.0, 0.0]], dtype=o3d.core.Dtype.Float32))
    except Exception as e:
        logger.warning(f"Failed to build full RaycastingScene: {e}")

    # Build shared OffscreenRenderer for ultra-fast rendering across all instances
    shared_renderer = None
    try:
        mesh.compute_vertex_normals()
        shared_renderer = o3d.visualization.rendering.OffscreenRenderer(render_res, render_res)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        shared_renderer.scene.set_background([0.0, 0.0, 0.0, 1.0], None)
        shared_renderer.scene.add_geometry("scene_mesh", mesh, mat)
    except Exception as e:
        logger.warning(f"Failed to initialize persistent OffscreenRenderer: {e}")

    room_center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0

    mesh_manifest: Dict[str, Any] = {
        "source_mesh": str(mesh_path),
        "labels_path": str(labels_path),
        "structural_metadata": str(structural_json_path),
        "levels_manifest": str(levels_manifest_path) if levels_manifest_path else None,
        "floor_levels": floor_levels,
        "human_height_offset": human_height_offset,
        "min_wall_distance": min_wall_distance,
        "min_triangle_distance": min_triangle_distance,
        "mode": "synthetic_in_context_mesh_render",
        "padding_ratio": padding,
        "render_resolution": render_res,
        "target_size": target_size,
        "min_resolution": effective_min_res,
        "padding_color": padding_color,
        "classes_processed": len(sorted_cids),
        "total_instances_extracted": 0,
        "extracted_classes": {},
        "extracted_instances": {},
        "discarded_instances": {},
    }

    gs_manifest: Optional[Dict[str, Any]] = None
    if enable_gs:
        gs_manifest = {
            "source_checkpoint": str(ckpt_file),
            "source_mesh": str(mesh_path),
            "num_gaussians": len(splats["means"]),
            "sh_degree": sh_degree,
            "padding_color": padding_color,
            "render_resolution": render_res,
            "min_resolution": effective_min_res,
            "classes_processed": len(sorted_cids),
            "total_instances_extracted": 0,
            "extracted_classes": {},
            "extracted_instances": {},
            "discarded_instances": {},
        }

    total_instances_extracted = 0

    # Pre-decompose connected components for all non-structural classes once
    instances_by_class: Dict[int, List[Dict[str, Any]]] = {}
    all_non_struct_instances: List[Dict[str, Any]] = []

    for o_cid in sorted_cids:
        o_mask = (labels == o_cid)
        if int(np.count_nonzero(o_mask)) == 0:
            continue
        o_insts = decompose_class_instances(
            vertices,
            triangles,
            o_mask,
            min_faces=min_instance_faces,
            min_area=min_instance_area,
        )
        if not o_insts:
            continue
        instances_by_class[o_cid] = o_insts
        for o_inst in o_insts:
            o_pts = o_inst["points"]
            if len(o_pts) > 600:
                step = max(1, len(o_pts) // 600)
                sub_pts = o_pts[::step][:600]
            else:
                sub_pts = o_pts
            all_non_struct_instances.append({
                "class_id": o_cid,
                "instance_id": o_inst["instance_id"],
                "pts": sub_pts,
                "center": (o_inst["bbox_min"] + o_inst["bbox_max"]) / 2.0,
            })

    gs_status_msg = f"both Mesh ({mesh_output_dir.name}) and Gaussian Splats ({gs_output_dir.name})" if enable_gs else f"Mesh ({mesh_output_dir.name})"
    print(f"\nRendering high-quality in-context views for each instance into {gs_status_msg}...")

    # ThreadPool for parallel non-blocking PNG encoding and disk writing
    save_pool = ThreadPoolExecutor(max_workers=8)
    save_futures = []

    for cid in sorted_cids:
        info = non_structural_classes[cid]
        role = info["role"]

        instances = instances_by_class.get(cid, [])
        if not instances:
            continue

        class_instances_mesh: Dict[str, Any] = {}
        class_instances_gs: Dict[str, Any] = {}

        for inst in instances:
            inst_id = int(inst["instance_id"])
            inst_key = f"{cid}_{inst_id}"
            inst_filename = f"class_{cid:03d}_inst_{inst_id:03d}.png"

            # 1. Mesh Render
            square_png, render_meta = render_synthetic_in_context_frame(
                instance=inst,
                all_vertices=vertices,
                all_triangles=triangles,
                all_vertex_colors=vertex_colors,
                labels=labels,
                class_id=cid,
                structural_cids=structural_cids,
                room_center=room_center,
                role=role,
                padding=padding,
                render_res=render_res,
                raycast_scene=raycast_scene,
                wall_scene=raycast_scene,
                min_resolution=effective_min_res,
                padding_color=padding_color,
                floor_levels=floor_levels,
                human_height_offset=human_height_offset,
                min_wall_distance=min_wall_distance,
                min_triangle_distance=min_triangle_distance,
                other_non_structural_instances=all_non_struct_instances,
                renderer=shared_renderer,
            )

            is_mesh_valid = True
            mesh_reason = "valid"
            mesh_metrics: Dict[str, Any] = {}
            if enable_quality_check:
                is_mesh_valid, mesh_reason, mesh_metrics = validate_rendered_frame(
                    square_png,
                    min_unique_colors=min_unique_colors,
                    min_dynamic_range=min_dynamic_range,
                    min_color_std=min_color_std,
                    min_laplacian_var=min_laplacian_var,
                    min_fg_pixels=min_fg_pixels,
                    min_autocorr=min_autocorr,
                    min_p95_grad=min_p95_grad,
                    min_edge_density=min_edge_density,
                )

            mesh_png_path = mesh_output_dir / inst_filename
            if is_mesh_valid:
                save_futures.append(save_pool.submit(square_png.save, mesh_png_path, "PNG", compress_level=1))
            else:
                mesh_discarded_dir.mkdir(parents=True, exist_ok=True)
                discard_mesh_path = mesh_discarded_dir / inst_filename
                save_futures.append(save_pool.submit(square_png.save, discard_mesh_path, "PNG", compress_level=1))
                if mesh_png_path.exists():
                    try:
                        mesh_png_path.unlink()
                    except OSError:
                        pass
                mesh_manifest["discarded_instances"][inst_key] = {
                    "class_id": cid,
                    "instance_id": inst_id,
                    "role": role,
                    "file": inst_filename,
                    "reason": mesh_reason,
                    "metrics": mesh_metrics,
                    **render_meta,
                }
                logger.info(f"  [DISCARDED MESH] Class {cid:3d} Inst {inst_id:2d}: {mesh_reason} -> saved to {discard_mesh_path}")

            # 2. Gaussian Splat Render (if enabled)
            gs_png_path: Optional[Path] = None
            is_gs_valid = True
            save_gs = False
            gs_reason = "valid"
            gs_metrics: Dict[str, Any] = {}
            if enable_gs and splats is not None:
                eye = np.array(render_meta["camera_eye"], dtype=np.float32)
                target = np.array(render_meta["camera_target"], dtype=np.float32)
                cam_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                cam_fov = float(render_meta["camera_fov_deg"])
                r_res = int(render_meta.get("raw_render_res", render_res))
                crop_coords = render_meta.get("crop_coords")

                viewmat, K = compute_w2c_and_intrinsics(eye, target, cam_up, cam_fov, r_res)
                raw_gs_rgb = render_gs_frame(
                    splats=splats,
                    viewmat=viewmat,
                    K=K,
                    render_res=r_res,
                    sh_degree=sh_degree,
                    device=device,
                )

                if crop_coords is not None:
                    cu0, cv0, cu1, cv1 = crop_coords
                    gs_crop = raw_gs_rgb[cv0:cv1, cu0:cu1]
                else:
                    gs_crop = raw_gs_rgb

                gs_square_png = pad_crop_to_square(
                    gs_crop,
                    min_side=effective_min_res,
                    padding_color=padding_color,
                )

                if enable_quality_check:
                    is_gs_valid, gs_reason, gs_metrics = validate_rendered_frame(
                        gs_square_png,
                        min_unique_colors=min_unique_colors,
                        min_dynamic_range=min_dynamic_range,
                        min_color_std=min_color_std,
                        min_laplacian_var=min_laplacian_var,
                        min_fg_pixels=min_fg_pixels,
                        min_autocorr=min_autocorr,
                        min_p95_grad=min_p95_grad,
                        min_edge_density=min_edge_density,
                    )
                    # For verified mesh objects, accept authentic renders with strong contrast/dynamic range
                    if not is_gs_valid and is_mesh_valid and gs_reason.startswith("no_recognizable_features: flat_or_blurry_surface"):
                        if (
                            gs_metrics.get("unique_colors", 0) >= 15
                            and gs_metrics.get("gray_std", 0) >= 10.0
                            and gs_metrics.get("gray_ptp", 0) >= 50.0
                        ):
                            is_gs_valid = True
                            gs_reason = "valid (mesh_verified_contrast)"

                # GS is saved if valid and:
                # 1) Mesh is also valid, OR
                # 2) GS is high-confidence feature-rich render of a real object (not noise floater cid 30)
                is_high_conf_gs = (
                    is_gs_valid
                    and cid != 30
                    and gs_metrics.get("p95_grad", 0) >= 1.5
                    and gs_metrics.get("unique_colors", 0) >= 18
                    and gs_metrics.get("gray_std", 0) >= 12.0
                )
                save_gs = is_gs_valid and (is_mesh_valid or is_high_conf_gs)

                gs_target_path = gs_output_dir / inst_filename
                if save_gs:
                    gs_png_path = gs_target_path
                    save_futures.append(save_pool.submit(gs_square_png.save, gs_png_path, "PNG", compress_level=1))
                else:
                    gs_discarded_dir.mkdir(parents=True, exist_ok=True)
                    discard_gs_path = gs_discarded_dir / inst_filename
                    save_futures.append(save_pool.submit(gs_square_png.save, discard_gs_path, "PNG", compress_level=1))
                    if gs_target_path.exists():
                        try:
                            gs_target_path.unlink()
                        except OSError:
                            pass
                    if gs_manifest is not None:
                        gs_manifest["discarded_instances"][inst_key] = {
                            "class_id": cid,
                            "instance_id": inst_id,
                            "role": role,
                            "file": inst_filename,
                            "reason": gs_reason if not is_gs_valid else f"mesh_discarded ({mesh_reason})",
                            "metrics": gs_metrics,
                            **render_meta,
                        }
                    logger.info(f"  [DISCARDED GS] Class {cid:3d} Inst {inst_id:2d}: {gs_reason} -> saved to {discard_gs_path}")

            if is_mesh_valid:
                inst_record = {
                    "class_id": cid,
                    "instance_id": inst_id,
                    "role": role,
                    "triangle_count": inst["triangle_count"],
                    "surface_area_m2": round(inst["area_m2"], 4),
                    "output_image_file": inst_filename,
                    "output_image_path": str(mesh_png_path),
                    "details": info.get("details", ""),
                    **render_meta,
                }
                if gs_png_path is not None:
                    inst_record["gs_image_file"] = inst_filename
                    inst_record["gs_image_path"] = str(gs_png_path)

                mesh_manifest["extracted_instances"][inst_key] = inst_record
                class_instances_mesh[str(inst_id)] = inst_record
                total_instances_extracted += 1

            if save_gs and enable_gs and gs_manifest is not None and gs_png_path is not None:
                gs_inst_record = {
                    "class_id": cid,
                    "instance_id": inst_id,
                    "role": role,
                    "triangle_count": inst["triangle_count"],
                    "surface_area_m2": round(inst["area_m2"], 4),
                    "output_image_file": inst_filename,
                    "output_image_path": str(gs_png_path),
                    "details": info.get("details", ""),
                    **render_meta,
                }
                gs_manifest["extracted_instances"][inst_key] = gs_inst_record
                class_instances_gs[str(inst_id)] = gs_inst_record

                gs_tag = " + GS" if (enable_gs and gs_png_path is not None) else ""
                print(
                    f"  Class {cid:3d} Inst {inst_id:2d} ({role:<20}): Level {render_meta['floor_level']} "
                    f"side={render_meta['biggest_side']} -> {inst_filename} [Mesh{gs_tag}, "
                    f"crop {render_meta['crop_size'][0]}x{render_meta['crop_size'][1]} -> square {square_png.size[0]}x{square_png.size[1]} PNG, "
                    f"wall_dist={render_meta['camera_distance_to_nearest_wall_m']:.2f}m, purity={render_meta['purity_ratio']:.2f}]"
                )

        # Build class-level record for backward compatibility
        if class_instances_mesh:
            primary_inst = class_instances_mesh.get("0", next(iter(class_instances_mesh.values())))
            class_entry = dict(primary_inst)
            class_entry["num_instances"] = len(class_instances_mesh)
            class_entry["instances"] = class_instances_mesh
            mesh_manifest["extracted_classes"][str(cid)] = class_entry

        if enable_gs and gs_manifest is not None and class_instances_gs:
            primary_gs = class_instances_gs.get("0", next(iter(class_instances_gs.values())))
            gs_class_entry = dict(primary_gs)
            gs_class_entry["num_instances"] = len(class_instances_gs)
            gs_class_entry["instances"] = class_instances_gs
            gs_manifest["extracted_classes"][str(cid)] = gs_class_entry

    # Await parallel background image saving
    for f in save_futures:
        f.result()
    save_pool.shutdown()

    mesh_manifest["total_instances_extracted"] = total_instances_extracted
    mesh_manifest["successful_extractions"] = len(mesh_manifest["extracted_classes"])
    mesh_manifest["elapsed_seconds"] = round(time.perf_counter() - t_start, 2)

    mesh_manifest_path = mesh_output_dir / "non_structural_frames_manifest.json"
    with open(mesh_manifest_path, "w", encoding="utf-8") as f:
        json.dump(mesh_manifest, f, indent=2)

    if mesh_manifest.get("discarded_instances"):
        discard_manifest_path = mesh_discarded_dir / "discarded_manifest.json"
        with open(discard_manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_discarded": len(mesh_manifest["discarded_instances"]),
                "discarded_instances": mesh_manifest["discarded_instances"],
            }, f, indent=2)
        print(f"Saved Mesh discarded manifest to {discard_manifest_path} ({len(mesh_manifest['discarded_instances'])} discarded)")

    if enable_gs and gs_manifest is not None:
        gs_manifest["total_instances_extracted"] = len(gs_manifest.get("extracted_instances", {}))
        gs_manifest["successful_extractions"] = len(gs_manifest["extracted_classes"])
        gs_manifest["elapsed_seconds"] = round(time.perf_counter() - t_start, 2)
        gs_manifest_path = gs_output_dir / "non_structural_frames_gs_manifest.json"
        with open(gs_manifest_path, "w", encoding="utf-8") as f:
            json.dump(gs_manifest, f, indent=2)
        print(f"Saved GS manifest to {gs_manifest_path}")

        if gs_manifest.get("discarded_instances"):
            gs_discard_manifest_path = gs_discarded_dir / "discarded_manifest.json"
            with open(gs_discard_manifest_path, "w", encoding="utf-8") as f:
                json.dump({
                    "total_discarded": len(gs_manifest["discarded_instances"]),
                    "discarded_instances": gs_manifest["discarded_instances"],
                }, f, indent=2)
            print(f"Saved GS discarded manifest to {gs_discard_manifest_path} ({len(gs_manifest['discarded_instances'])} discarded)")

    mesh_disc_count = len(mesh_manifest.get("discarded_instances", {}))
    gs_disc_count = len(gs_manifest.get("discarded_instances", {})) if gs_manifest else 0
    print(
        f"\nExtraction complete in {mesh_manifest['elapsed_seconds']:.2f}s: "
        f"{total_instances_extracted} instances across {len(mesh_manifest['extracted_classes'])}/{len(sorted_cids)} classes extracted. "
        f"[Discarded: Mesh={mesh_disc_count}, GS={gs_disc_count}]"
    )
    print(f"Saved Mesh manifest to {mesh_manifest_path}")

    return 0 if total_instances_extracted > 0 else 1


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return run_extraction(
        mesh_path=args.mesh_path,
        labels_path=args.labels_path,
        structural_json_path=args.structural_json,
        workspace=args.workspace,
        output_dir=args.output_dir,
        gs_output_dir=args.gs_output_dir,
        checkpoint_path=args.checkpoint,
        skip_gs=args.skip_gs,
        device=args.device,
        levels_manifest_path=args.levels_manifest,
        human_height_offset=args.human_height_offset,
        min_wall_distance=args.min_wall_distance,
        min_triangle_distance=args.min_triangle_distance,
        padding=args.padding,
        render_res=args.render_res,
        target_size=args.target_size,
        min_resolution=args.min_resolution,
        padding_color=args.padding_color,
        min_instance_faces=args.min_instance_faces,
        min_instance_area=args.min_instance_area,
        enable_quality_check=args.enable_quality_check,
        mesh_discarded_dir=args.mesh_discarded_dir,
        gs_discarded_dir=args.gs_discarded_dir,
        min_unique_colors=args.min_unique_colors,
        min_dynamic_range=args.min_dynamic_range,
        min_color_std=args.min_color_std,
        min_laplacian_var=args.min_laplacian_var,
        min_fg_pixels=args.min_fg_pixels,
        min_autocorr=args.min_autocorr,
        min_p95_grad=args.min_p95_grad,
        min_edge_density=args.min_edge_density,
    )


if __name__ == "__main__":
    raise SystemExit(main())
