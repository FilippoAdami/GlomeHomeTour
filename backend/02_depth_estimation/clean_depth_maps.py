#!/usr/bin/env python3
"""Clean depth maps in-place on disk using Sobel Depth Gradient Filtering and GPU PyTorch Free-Space Carving.

Zeroes out invalid pixels directly in the `.npy` depth map files so that:
1. Flying edge pixels at object silhouettes (Sobel depth gradient > max_depth_gradient) are removed.
2. Floating free-space phantom pixels (cross-view epipolar carving violations) are removed via GPU tensor operations.
3. Downstream tools (export_depth_plys.py, surfel initialization, TSDF fusion, etc.) consume clean depth maps.

Usage:
    python 02_depth_estimation/clean_depth_maps.py [--workspace DIR] [--max-depth-gradient 0.15] [--no-freespace]
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
from Utilities.pipeline_paths import bootstrap

bootstrap()

from package_loader import CameraIntrinsics, Keyframe
from Utilities.scene_io import load_scene

from initialization import compute_overexposed_mask, estimate_adaptive_depth_ceiling

DEFAULT_WORKSPACE = _backend_dir / "current_scene"
STAGE_DIRNAME = "02_depth_estimation"
_FLIP_YZ = np.diag([1.0, -1.0, -1.0, 1.0])


def clean_single_depth_gradient(
    depth_map: np.ndarray,
    max_depth_gradient: float = 0.15,
) -> tuple[np.ndarray, int]:
    """Zero out pixels with high relative depth gradients (flying edge pixels)."""
    if max_depth_gradient <= 0.0:
        return depth_map, 0

    d = depth_map.copy()
    valid = np.isfinite(d) & (d > 0.1)

    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3) / np.maximum(d, 1e-3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3) / np.maximum(d, 1e-3)
    grad = np.sqrt(gx**2 + gy**2)

    invalid_grad = valid & (grad > max_depth_gradient)
    n_cleared = int(np.sum(invalid_grad))

    d[invalid_grad] = 0.0
    return d, n_cleared


def clean_saturation_mask(
    depth_map: np.ndarray,
    img_rgb: np.ndarray,
    min_threshold: int = 250,
    max_threshold: int = 255,
    percentile: float = 99.8,
    max_chroma_diff: float = 35.0,
    dilation_radius: int = 1,
) -> tuple[np.ndarray, int]:
    """Zero out overexposed / optical bloom pixels in depth map."""
    sat_mask = compute_overexposed_mask(
        img_rgb,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        percentile=percentile,
        max_chroma_diff=max_chroma_diff,
        dilation_kernel_size=dilation_radius * 2 + 1 if dilation_radius > 0 else 0,
    )
    if not np.any(sat_mask):
        return depth_map, 0
    d = depth_map.copy()
    invalid = sat_mask & (d > 0.0)
    n_cleared = int(np.sum(invalid))
    d[invalid] = 0.0
    return d, n_cleared


def clean_depth_ceiling(
    depth_map: np.ndarray,
    max_depth_ceiling: float,
) -> tuple[np.ndarray, int]:
    """Zero out depth pixels that exceed the room's adaptive depth ceiling."""
    if max_depth_ceiling <= 0.0 or not np.isfinite(max_depth_ceiling):
        return depth_map, 0
    d = depth_map.copy()
    invalid = (d > max_depth_ceiling) & (d > 0.0)
    n_cleared = int(np.sum(invalid))
    d[invalid] = 0.0
    return d, n_cleared


def clean_grazing_angles(
    depth_map: np.ndarray,
    normals_cam: np.ndarray,
    intrinsics: CameraIntrinsics,
    max_grazing_angle_deg: float = 85.0,
    precomputed_ray_dir: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Zero out pixels where camera optical rays strike surface normals at glancing angles."""
    if max_grazing_angle_deg <= 0.0 or max_grazing_angle_deg >= 90.0:
        return depth_map, 0

    h, w = depth_map.shape[:2]
    if normals_cam.shape[:2] != (h, w):
        normals_cam = cv2.resize(normals_cam, (w, h), interpolation=cv2.INTER_NEAREST)

    cos_min = float(np.cos(np.radians(max_grazing_angle_deg)))

    if precomputed_ray_dir is not None and precomputed_ray_dir.shape[:2] == (h, w):
        ray_dir = precomputed_ray_dir
    else:
        fx, fy = float(intrinsics.fl_x), float(intrinsics.fl_y)
        cx, cy = float(intrinsics.cx), float(intrinsics.cy)
        v_grid, u_grid = np.mgrid[0:h, 0:w].astype(np.float32)
        x_cam = (u_grid - cx) / fx
        y_cam = -(v_grid - cy) / fy
        z_cam = -np.ones((h, w), dtype=np.float32)
        ray_dir = np.stack([x_cam, y_cam, z_cam], axis=-1)
        ray_norm = np.linalg.norm(ray_dir, axis=-1, keepdims=True)
        ray_dir /= np.maximum(ray_norm, 1e-6)

    # In OpenGL camera coordinates, normals point towards camera (+Z)
    cos_grazing = np.sum(-normals_cam * ray_dir, axis=-1)

    d = depth_map.copy()
    valid = (d > 0.1) & np.isfinite(d)
    invalid = valid & (cos_grazing < cos_min)
    n_cleared = int(np.sum(invalid))
    d[invalid] = 0.0
    return d, n_cleared


def clean_all_depth_maps(
    depth_maps: list[np.ndarray],
    images: Sequence[np.ndarray] | None,
    c2w_mats: np.ndarray,
    intrinsics: CameraIntrinsics,
    normal_maps: list[np.ndarray] | None = None,
    max_depth_gradient: float = 0.15,
    enable_saturation_mask: bool = True,
    enable_depth_ceiling: bool = True,
    max_grazing_angle_deg: float = 85.0,
    enable_freespace_filter: bool = True,
    freespace_margin_m: float = 0.08,
    max_freespace_violations: int = 0,
    enable_depth_alignment: bool = True,
    alignment_margin_m: float = 0.08,
    min_consensus_views: int = 0,
    min_frustum_views: int = 2,
    device: str = "cuda",
) -> tuple[list[np.ndarray], dict[str, int | float]]:
    """Clean depth maps end-to-end with all masks applied directly to the single depth maps."""
    cleaned = [d.copy() for d in depth_maps]
    stats: dict[str, int | float] = {
        "gradient": 0,
        "saturation": 0,
        "ceiling": 0,
        "grazing": 0,
        "freespace": 0,
        "uncorroborated": 0,
        "aligned_pixels": 0,
        "mean_adjustment_m": 0.0,
    }

    # 1. Depth ceiling
    if enable_depth_ceiling and len(cleaned) > 0:
        depth_ceiling = estimate_adaptive_depth_ceiling(cleaned)
        for i in range(len(cleaned)):
            cleaned[i], n_c = clean_depth_ceiling(cleaned[i], depth_ceiling)
            stats["ceiling"] += n_c

    # 2. Saturation masking, Sobel gradients, and Grazing angles (Multi-threaded & Precomputed Ray Grid)
    if len(cleaned) > 0:
        h, w = cleaned[0].shape[:2]
        fx, fy = float(intrinsics.fl_x), float(intrinsics.fl_y)
        cx, cy = float(intrinsics.cx), float(intrinsics.cy)
        v_grid, u_grid = np.mgrid[0:h, 0:w].astype(np.float32)
        x_cam = (u_grid - cx) / fx
        y_cam = -(v_grid - cy) / fy
        z_cam = -np.ones((h, w), dtype=np.float32)
        cached_ray_dir = np.stack([x_cam, y_cam, z_cam], axis=-1)
        ray_norm = np.linalg.norm(cached_ray_dir, axis=-1, keepdims=True)
        cached_ray_dir /= np.maximum(ray_norm, 1e-6)

        def _clean_frame_substep(i: int):
            d = cleaned[i]
            n_s, n_g, n_gr = 0, 0, 0
            if enable_saturation_mask and images is not None and i < len(images) and images[i] is not None:
                d, n_s = clean_saturation_mask(d, images[i])
            if max_depth_gradient > 0.0:
                d, n_g = clean_single_depth_gradient(d, max_depth_gradient=max_depth_gradient)
            if normal_maps is not None and i < len(normal_maps) and normal_maps[i] is not None and max_grazing_angle_deg > 0.0:
                d, n_gr = clean_grazing_angles(d, normal_maps[i], intrinsics,
                                               max_grazing_angle_deg=max_grazing_angle_deg,
                                               precomputed_ray_dir=cached_ray_dir)
            return i, d, n_s, n_g, n_gr

        max_workers = min(8, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for i, d, n_s, n_g, n_gr in pool.map(_clean_frame_substep, range(len(cleaned))):
                cleaned[i] = d
                stats["saturation"] += n_s
                stats["gradient"] += n_g
                stats["grazing"] += n_gr

    # 3. Cross-view free space carving & epipolar depth consensus alignment (GPU accelerated)
    if (enable_freespace_filter or enable_depth_alignment or min_consensus_views > 0) and len(cleaned) > 1:
        cleaned, mv_stats = gpu_multiview_depth_refine_and_carve(
            depth_maps=cleaned,
            c2w_mats=c2w_mats,
            intrinsics=intrinsics,
            freespace_margin_m=freespace_margin_m,
            max_freespace_violations=max_freespace_violations,
            enable_depth_alignment=enable_depth_alignment,
            alignment_margin_m=alignment_margin_m,
            min_consensus_views=min_consensus_views,
            min_frustum_views=min_frustum_views,
            device=device,
        )
        stats["freespace"] = mv_stats["freespace"]
        stats["uncorroborated"] = mv_stats["uncorroborated"]
        stats["aligned_pixels"] = mv_stats["aligned_pixels"]
        stats["mean_adjustment_m"] = mv_stats["mean_adjustment_m"]

    return cleaned, stats


def gpu_multiview_depth_refine_and_carve(
    depth_maps: list[np.ndarray],
    c2w_mats: np.ndarray,
    intrinsics: CameraIntrinsics,
    max_neighbors: int = 8,
    freespace_margin_m: float = 0.08,
    max_freespace_violations: int = 0,
    enable_depth_alignment: bool = True,
    alignment_margin_m: float = 0.08,
    min_consensus_views: int = 0,
    min_frustum_views: int = 2,
    device: str = "cuda",
) -> tuple[list[np.ndarray], dict[str, int | float]]:
    """GPU PyTorch tensor vectorized Cross-View Epipolar Depth Alignment & Free-Space Carving.

    Performs:
    1. Multi-view epipolar ray alignment: aligns individual depth map surfaces to mutual 3D consensus.
    2. Free-space phantom carving: zeroes out pixels violating unobstructed empty space in other views.
    3. Multi-view corroboration filtering: removes isolated floaters seen by >= min_frustum_views with 0 agreeing views.
    """
    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"

    dev = torch.device(device)
    num_frames = len(depth_maps)
    h, w = depth_maps[0].shape[:2]

    # Convert depth maps to PyTorch GPU tensor shape (N, 1, H, W)
    d_tensor = torch.from_numpy(np.stack(depth_maps)).to(device=dev, dtype=torch.float32).unsqueeze(1)
    c2w_t = torch.from_numpy(c2w_mats).to(device=dev, dtype=torch.float32)  # (N, 4, 4)
    w2c_t = torch.linalg.inv(c2w_t)                                         # (N, 4, 4)

    fx, fy = float(intrinsics.fl_x), float(intrinsics.fl_y)
    cx, cy = float(intrinsics.cx), float(intrinsics.cy)

    v_grid, u_grid = torch.meshgrid(
        torch.arange(h, device=dev, dtype=torch.float32),
        torch.arange(w, device=dev, dtype=torch.float32),
        indexing="ij"
    )

    # Compute neighboring camera indices per frame based on baseline & co-directionality
    centers = c2w_t[:, :3, 3]  # (N, 3)
    dirs = -c2w_t[:, :3, 2]    # (N, 3) viewing direction

    neighbor_indices_per_frame = []
    for i in range(num_frames):
        cur_c = centers[i]
        cur_d = dirs[i]
        dists = torch.norm(centers - cur_c, dim=-1)
        cos_ang = torch.sum(dirs * cur_d, dim=-1)

        valid_cand = (torch.arange(num_frames, device=dev) != i) & (cos_ang > 0.1) & (dists < 3.5)
        cand_indices = torch.where(valid_cand)[0]

        if len(cand_indices) == 0:
            dists_all = torch.norm(centers - cur_c, dim=-1)
            dists_all[i] = 1e9
            _, top_k = torch.topk(dists_all, k=min(max_neighbors, num_frames - 1), largest=False)
            neighbor_indices_per_frame.append(top_k.tolist())
        else:
            scores = dists[cand_indices] / torch.clamp(cos_ang[cand_indices], min=0.2)
            _, top_k_rel = torch.topk(scores, k=min(max_neighbors, len(cand_indices)), largest=False)
            neighbor_indices_per_frame.append(cand_indices[top_k_rel].tolist())

    total_cleared_fs = 0
    total_cleared_uncorr = 0
    total_aligned_pixels = 0
    total_adjustment_mag = 0.0
    cleaned_maps = []

    # Process alignment & carving on GPU using batched neighbor projections
    with torch.inference_mode():
        for i in range(num_frames):
            d_i = d_tensor[i, 0]  # (H, W)
            valid_i = (d_i > 0.2) & torch.isfinite(d_i)

            if not torch.any(valid_i):
                cleaned_maps.append(d_i.cpu().numpy())
                continue

            neighbors = neighbor_indices_per_frame[i]
            K = len(neighbors)
            if K == 0:
                cleaned_maps.append(d_i.cpu().numpy())
                continue

            # Unproject current frame to OpenGL camera space: +X right, +Y up, -Z fwd
            z_cam = -d_i
            x_cam = (u_grid - cx) * d_i / fx
            y_cam = -(v_grid - cy) * d_i / fy
            ones = torch.ones_like(d_i)
            pts_cam_4 = torch.stack([x_cam, y_cam, z_cam, ones], dim=-1)  # (H, W, 4)

            # Transform to world space
            c2w_i = c2w_t[i]
            pts_world_4 = torch.matmul(pts_cam_4, c2w_i.T)  # (H, W, 4)

            # Batched transform of all K neighbors in a single matrix multiply: (K, 4, 4) @ (1, H, W, 4, 1) -> (K, H, W, 4)
            w2c_k = w2c_t[neighbors]  # (K, 4, 4)
            pts_cam_k = torch.matmul(w2c_k.view(K, 1, 1, 4, 4), pts_world_4.view(1, h, w, 4, 1)).squeeze(-1)  # (K, H, W, 4)

            proj_z = -pts_cam_k[..., 2]  # (K, H, W)
            proj_x = pts_cam_k[..., 0]   # (K, H, W)
            proj_y = pts_cam_k[..., 1]   # (K, H, W)

            in_front = proj_z > 0.2
            denom_z = torch.clamp(proj_z, min=1e-4)
            u_k = fx * (proj_x / denom_z) + cx
            v_k = -fy * (proj_y / denom_z) + cy

            valid_uv = in_front & (u_k >= 2) & (u_k < w - 2) & (v_k >= 2) & (v_k < h - 2)

            u_norm = (2.0 * u_k / (w - 1.0)) - 1.0
            v_norm = (2.0 * v_k / (h - 1.0)) - 1.0
            grid_k = torch.stack([u_norm, v_norm], dim=-1)  # (K, H, W, 2)

            # Batched neighbor depth sampling in a single GPU kernel launch
            obs_d_k = F.grid_sample(
                d_tensor[neighbors],
                grid_k,
                mode="nearest",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze(1)  # (K, H, W)

            valid_obs = valid_uv & torch.isfinite(obs_d_k) & (obs_d_k > 0.2)
            in_frustum_count = torch.sum(valid_obs.to(torch.int32), dim=0)  # (H, W)

            # Geometric consistency agreement
            tau_agree = alignment_margin_m + 0.03 * proj_z  # (K, H, W)
            diff_d = torch.abs(proj_z - obs_d_k)
            is_agree = valid_obs & valid_i.unsqueeze(0) & (diff_d <= tau_agree)
            consistent_count = torch.sum(is_agree.to(torch.int32), dim=0)  # (H, W)

            # Free-space violation: candidate point strictly in front of surface and not within agreement tolerance
            violation = valid_obs & valid_i.unsqueeze(0) & (~is_agree) & (proj_z < (obs_d_k - freespace_margin_m))
            violations_count = torch.sum(violation.to(torch.int32), dim=0)  # (H, W)

            # 1. Continuous Depth Consensus Alignment
            if enable_depth_alignment:
                cos_views = torch.clamp(torch.sum(dirs[i].unsqueeze(0) * dirs[neighbors], dim=-1), min=0.1, max=1.0).view(K, 1, 1)
                d_j_to_i = d_i.unsqueeze(0) * (obs_d_k / denom_z)  # (K, H, W)
                w_j = cos_views / (1.0 + diff_d / torch.clamp(tau_agree, min=1e-3))  # (K, H, W)

                w_j_masked = torch.where(is_agree, w_j, torch.zeros_like(w_j))
                d_j_weighted = w_j_masked * d_j_to_i

                depth_accum = d_i + torch.sum(d_j_weighted, dim=0)
                weight_accum = 1.0 + torch.sum(w_j_masked, dim=0)

                d_aligned = depth_accum / torch.clamp(weight_accum, min=1.0)
                diff = torch.abs(d_aligned - d_i)
                aligned_mask = valid_i & (weight_accum > 1.05) & (diff > 1e-4)
                total_aligned_pixels += int(torch.sum(aligned_mask).item())
                total_adjustment_mag += float(torch.sum(diff[aligned_mask]).item()) if torch.any(aligned_mask) else 0.0
                d_cur = torch.where(valid_i, d_aligned, d_i)
            else:
                d_cur = d_i

            # 2. Carve Free-Space Violations
            carve_mask = (violations_count > max_freespace_violations) & valid_i
            n_carved = int(torch.sum(carve_mask).item())
            total_cleared_fs += n_carved
            if n_carved > 0:
                d_cur = d_cur.masked_fill(carve_mask, 0.0)

            # 3. Prune Uncorroborated Floaters
            if min_consensus_views > 0:
                uncorr_mask = valid_i & (in_frustum_count >= min_frustum_views) & (consistent_count < min_consensus_views)
                n_uncorr = int(torch.sum(uncorr_mask).item())
                total_cleared_uncorr += n_uncorr
                if n_uncorr > 0:
                    d_cur = d_cur.masked_fill(uncorr_mask, 0.0)

            cleaned_maps.append(d_cur.cpu().numpy())

            # Periodic VRAM hygiene
            if (i + 1) % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    stats = {
        "freespace": total_cleared_fs,
        "uncorroborated": total_cleared_uncorr,
        "aligned_pixels": total_aligned_pixels,
        "mean_adjustment_m": (total_adjustment_mag / max(1, total_aligned_pixels)),
    }
    return cleaned_maps, stats


def gpu_free_space_carve(
    depth_maps: list[np.ndarray],
    c2w_mats: np.ndarray,
    intrinsics: CameraIntrinsics,
    max_neighbors: int = 8,
    freespace_margin_m: float = 0.08,
    max_freespace_violations: int = 0,
    device: str = "cuda",
) -> tuple[list[np.ndarray], int]:
    """Compatibility wrapper calling gpu_multiview_depth_refine_and_carve without depth alignment."""
    cleaned, stats = gpu_multiview_depth_refine_and_carve(
        depth_maps=depth_maps,
        c2w_mats=c2w_mats,
        intrinsics=intrinsics,
        max_neighbors=max_neighbors,
        freespace_margin_m=freespace_margin_m,
        max_freespace_violations=max_freespace_violations,
        enable_depth_alignment=False,
        device=device,
    )
    return cleaned, int(stats["freespace"])


def clean_depth_maps_in_place(
    workspace: Path,
    max_depth_gradient: float = 0.15,
    enable_saturation_mask: bool = True,
    enable_depth_ceiling: bool = True,
    max_grazing_angle_deg: float = 85.0,
    enable_freespace_filter: bool = True,
    max_freespace_violations: int = 0,
    freespace_margin_m: float = 0.08,
    enable_depth_alignment: bool = True,
    alignment_margin_m: float = 0.08,
    min_consensus_views: int = 0,
    min_frustum_views: int = 2,
) -> None:
    ws = Path(workspace).resolve()
    depth_dir = ws / STAGE_DIRNAME / "depth"
    if not depth_dir.exists():
        depth_dir = ws / "depth"

    maps_dir = depth_dir / "depth_maps"
    normals_dir = depth_dir / "normal_maps"
    if not maps_dir.is_dir():
        raise FileNotFoundError(f"Depth maps directory not found at: {maps_dir}")

    poses_path = depth_dir / "poses_da3.npz"
    if not poses_path.is_file():
        raise FileNotFoundError(f"Camera poses file not found at: {poses_path}")

    poses_data = np.load(poses_path)
    w2c = poses_data["w2c"]
    pose_names = [str(n) for n in poses_data["names"]]

    # Load scene intrinsics and keyframes
    scene = load_scene(ws)
    c2w_gl = np.stack([np.linalg.inv(p.astype(np.float64)) @ _FLIP_YZ for p in w2c])

    npy_paths = []
    depth_maps = []
    images = []
    normal_maps = []

    images_dir = ws / "images"

    for i, n in enumerate(pose_names):
        stem = Path(n).stem
        p = maps_dir / f"{stem}.npy"
        if not p.is_file():
            continue
        depth_maps.append(np.load(p).astype(np.float32))
        npy_paths.append(p)

        # Load RGB image if available
        img_path = images_dir / f"{stem}.jpg"
        if not img_path.is_file():
            img_path = images_dir / n
        if img_path.is_file():
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is not None:
                images.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            else:
                images.append(None)
        else:
            images.append(None)

        # Load Normal map if available
        norm_path = normals_dir / f"{stem}.npy"
        if norm_path.is_file():
            normal_maps.append(np.load(norm_path).astype(np.float32))
        else:
            normal_maps.append(None)

    num_frames = len(depth_maps)
    print(f"[GPU Clean Depth Maps] Loaded {num_frames} depth maps from {maps_dir}.")

    t0 = time.time()
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    cleaned_maps, stats = clean_all_depth_maps(
        depth_maps=depth_maps,
        images=images if any(im is not None for im in images) else None,
        c2w_mats=c2w_gl,
        intrinsics=scene.intrinsics,
        normal_maps=normal_maps if any(nm is not None for nm in normal_maps) else None,
        max_depth_gradient=max_depth_gradient,
        enable_saturation_mask=enable_saturation_mask,
        enable_depth_ceiling=enable_depth_ceiling,
        max_grazing_angle_deg=max_grazing_angle_deg,
        enable_freespace_filter=enable_freespace_filter,
        freespace_margin_m=freespace_margin_m,
        max_freespace_violations=max_freespace_violations,
        enable_depth_alignment=enable_depth_alignment,
        alignment_margin_m=alignment_margin_m,
        min_consensus_views=min_consensus_views,
        min_frustum_views=min_frustum_views,
        device=device_name,
    )

    t_elapsed = time.time() - t0
    print(f"[GPU Clean Depth Maps] Overwriting {len(npy_paths)} .npy files on disk...")
    for path, dmap in zip(npy_paths, cleaned_maps):
        np.save(path, dmap.astype(np.float16))

    print(
        f"[GPU Clean Depth Maps] Success in {t_elapsed:.2f}s across {num_frames} keyframes:\n"
        f"  - Saturation/Bloom pixels cleared: {int(stats['saturation']):,}\n"
        f"  - Depth ceiling outliers cleared: {int(stats['ceiling']):,}\n"
        f"  - Edge-gradient flying pixels cleared: {int(stats['gradient']):,}\n"
        f"  - Grazing angle pixels cleared: {int(stats['grazing']):,}\n"
        f"  - Free-space phantom pixels cleared: {int(stats['freespace']):,}\n"
        f"  - Uncorroborated floater pixels cleared: {int(stats['uncorroborated']):,}\n"
        f"  - Aligned pixels fused: {int(stats['aligned_pixels']):,} (mean adj: {stats['mean_adjustment_m']*100:.2f} cm)"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE), help="Path to workspace directory")
    parser.add_argument("--max-depth-gradient", type=float, default=0.15, help="Max relative depth gradient threshold (default: 0.15)")
    parser.add_argument("--no-saturation-mask", action="store_true", help="Disable dynamic saturation/bloom masking")
    parser.add_argument("--no-depth-ceiling", action="store_true", help="Disable adaptive depth ceiling clipping")
    parser.add_argument("--max-grazing-angle", type=float, default=85.0, help="Max grazing angle threshold in degrees (default: 85.0)")
    parser.add_argument("--no-freespace", action="store_true", help="Disable cross-view free-space carving")
    parser.add_argument("--no-depth-alignment", action="store_true", help="Disable continuous multi-view epipolar depth alignment")
    parser.add_argument("--alignment-margin", type=float, default=0.08, help="Geometric tolerance in meters for cross-view alignment")
    parser.add_argument("--min-consensus-views", type=int, default=0, help="Minimum number of agreeing views required when point is in >= 2 frustums")
    args = parser.parse_args(argv)

    clean_depth_maps_in_place(
        workspace=Path(args.workspace),
        max_depth_gradient=args.max_depth_gradient,
        enable_saturation_mask=not args.no_saturation_mask,
        enable_depth_ceiling=not args.no_depth_ceiling,
        max_grazing_angle_deg=args.max_grazing_angle,
        enable_freespace_filter=not args.no_freespace,
        enable_depth_alignment=not args.no_depth_alignment,
        alignment_margin_m=args.alignment_margin,
        min_consensus_views=args.min_consensus_views,
    )


if __name__ == "__main__":
    main()
