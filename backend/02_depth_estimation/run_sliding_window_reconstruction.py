#!/usr/bin/env python3
"""GlomeHomeTour: End-to-End Multi-View Sliding Window Reconstruction Runner.

Processes captured scene packages with:
1. Dynamic depth-adaptive keyframe filtering (yielding ~149 keyframes for the bedroom).
2. Multi-view Depth Anything 3 sliding-window inference with cross-view attention.
3. Inter-chunk Sim(3) scale-shift alignment and multi-window ensembling (K=2 mean blend or K=3 median consensus).
4. Full 3D 2D-Gaussian surfel cloud unprojection, normal estimation, multi-view consistency filtering, and binary PLY export.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image

# Ensure backend and depth_anything_3 are in sys.path
_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
from pipeline_paths import bootstrap
bootstrap()

_da3_src = _backend_dir / "third_party" / "depth_anything_3" / "src"
if _da3_src.is_dir() and str(_da3_src) not in sys.path:
    sys.path.insert(0, str(_da3_src))

from package_loader import CameraIntrinsics, CapturePackage, Keyframe, PackageLoader
from quality_gate import QualityGate
from pose_aligner import PoseAligner
from depth_priors import DepthPriorEstimator
from initialization import SurfelCloudInitializer


# Anisotropic Portrait Optics constants
HFOV_DEG = 40.8
VFOV_DEG = 67.0


def extract_depth_adaptive_keyframes(
    package: CapturePackage,
    target_count: Optional[int] = None,
) -> tuple[list[Keyframe], list[np.ndarray], CameraIntrinsics, list[Keyframe], list[np.ndarray], list[int]]:
    """Filter keyframes using depth-adaptive velocity, blur quality gating, and anisotropic rotation limits.

    Returns:
        (depth_keyframes, depth_images, intrinsics_portrait, all_filtered_keyframes, all_filtered_images)
        - depth_keyframes / depth_images: Subset selected for optimal depth prior consensus & 3D unprojection.
        - all_filtered_keyframes / all_filtered_images: Complete set of quality-filtered frames for 2DGS training.
    """
    gate = QualityGate()
    gate_result = gate.evaluate(package.keyframes)
    aligner = PoseAligner(package.trajectory)
    synced = aligner.synchronize_keyframes(gate_result.accepted_keyframes)
    if not synced:
        raise ValueError("No synchronized keyframes found in package.")

    intrinsics_raw = package.intrinsics

    # Portrait camera transformation
    # 90° clockwise roll transform for camera extrinsics
    R_roll = np.array([
        [ 0.0, -1.0,  0.0,  0.0],
        [ 1.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0,  0.0],
        [ 0.0,  0.0,  0.0,  1.0],
    ], dtype=np.float64)

    # Portrait intrinsics: W=1080, H=1920
    hl, wl = intrinsics_raw.h, intrinsics_raw.w
    intrinsics_portrait = CameraIntrinsics(
        camera_model=intrinsics_raw.camera_model,
        fl_x=intrinsics_raw.fl_y,
        fl_y=intrinsics_raw.fl_x,
        cx=float(hl - intrinsics_raw.cy),
        cy=float(intrinsics_raw.cx),
        w=hl,  # 1080
        h=wl,  # 1920
        camera_angle_x=math.radians(HFOV_DEG),
        k1=intrinsics_raw.k1,
        k2=intrinsics_raw.k2,
        p1=intrinsics_raw.p1,
        p2=intrinsics_raw.p2,
    )

    # Convert all quality-filtered synced keyframes to upright orientation
    all_upright_keyframes: list[Keyframe] = []
    all_upright_images: list[np.ndarray] = []
    for raw_kf in synced:
        img_raw = raw_kf.load_image_rgb()
        img_up = cv2.rotate(img_raw, cv2.ROTATE_90_CLOCKWISE)
        all_upright_images.append(img_up)

        c2w_upright = np.dot(raw_kf.transform_matrix, R_roll)
        up_kf = Keyframe(
            file_path=raw_kf.file_path,
            timestamp_ns=raw_kf.timestamp_ns,
            fl_x=intrinsics_portrait.fl_x,
            fl_y=intrinsics_portrait.fl_y,
            cx=intrinsics_portrait.cx,
            cy=intrinsics_portrait.cy,
            transform_matrix=c2w_upright,
            image_loader=lambda img=img_up: Image.fromarray(img),
        )
        all_upright_keyframes.append(up_kf)

    sift = cv2.SIFT_create(nfeatures=600)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    def extract_features(img_upright: np.ndarray):
        gray = cv2.cvtColor(img_upright, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (270, 480))
        return sift.detectAndCompute(small, None)

    def compute_inliers(des1, des2, kp1, kp2):
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return 0, 0, 2.0
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if len(matches) > 0 and m.distance < 0.75 * n.distance]
        if len(good) < 8:
            return len(good), len(good), 2.0
        p1 = np.float32([kp1[m.queryIdx].pt for m in good])
        p2 = np.float32([kp2[m.trainIdx].pt for m in good])
        F, mask = cv2.findFundamentalMat(p1.reshape(-1, 1, 2), p2.reshape(-1, 1, 2), cv2.FM_RANSAC, 3.0)
        inl = int(mask.sum()) if mask is not None else 0
        
        if mask is not None and inl >= 6:
            inlier_p1 = p1[mask.ravel() == 1]
            inlier_p2 = p2[mask.ravel() == 1]
            disp = float(np.median(np.linalg.norm(inlier_p2 - inlier_p1, axis=1)))
        else:
            disp = float(np.median(np.linalg.norm(p2 - p1, axis=1)))
            
        return inl, len(good), disp

    def compute_portrait_rotation(r1, r2):
        r_rel = np.dot(r2, r1.T)
        yaw = np.degrees(np.arctan2(r_rel[0, 2], r_rel[2, 2]))
        pitch = np.degrees(np.arctan2(-r_rel[1, 2], np.sqrt(r_rel[1, 0]**2 + r_rel[1, 1]**2)))
        roll = np.degrees(np.arctan2(r_rel[1, 0], r_rel[1, 1]))
        norm_rot = np.sqrt((yaw / HFOV_DEG)**2 + (pitch / VFOV_DEG)**2 + (roll / HFOV_DEG)**2) * HFOV_DEG
        total_rot = np.degrees(np.arccos(np.clip((np.trace(r_rel) - 1.0)/2.0, -1.0, 1.0)))
        return float(norm_rot), float(total_rot)

    selected_indices = [0]
    c2w_last = synced[0].transform_matrix
    kp_last, des_last = extract_features(all_upright_images[0])
    last_idx = 0
    current_Z = 2.2

    while last_idx < len(synced) - 1:
        chosen = None
        z_ratio = np.clip(current_Z / 2.0, 0.6, 2.0)
        d_target = float(0.55 * (z_ratio ** 0.5))
        d_max = float(d_target * 1.35)
        rot_target = float(16.0 * (z_ratio ** 0.35))
        rot_max = float(rot_target * 1.35)

        # Search forward up to 35 frames
        for j in range(last_idx + 1, min(last_idx + 35, len(synced))):
            c2w_curr = synced[j].transform_matrix
            d = float(np.linalg.norm(c2w_curr[:3, 3] - c2w_last[:3, 3]))
            norm_rot, total_rot = compute_portrait_rotation(c2w_last[:3, :3], c2w_curr[:3, :3])

            if d >= d_max or norm_rot >= rot_max or total_rot >= 25.0:
                chosen = j if j == last_idx + 1 else (j - 1)
                break

            if d >= d_target or norm_rot >= rot_target:
                kp_curr, des_curr = extract_features(all_upright_images[j])
                inl, good, disp = compute_inliers(des_last, des_curr, kp_last, kp_curr)
                if disp > 2.0 and d > 0.08:
                    z_est = (360.0 * d) / disp
                    current_Z = float(np.clip(0.7 * current_Z + 0.3 * z_est, 0.8, 4.5))
                chosen = j
                break

        if chosen is None:
            chosen = min(last_idx + 10, len(synced) - 1)

        selected_indices.append(chosen)
        last_idx = chosen
        c2w_last = synced[last_idx].transform_matrix
        kp_last, des_last = extract_features(all_upright_images[last_idx])

        if target_count and len(selected_indices) >= target_count:
            break

    print(f"[KeyframeSelector] Extracted {len(selected_indices)} depth-adaptive portrait keyframes from {len(synced)} quality-filtered frames (total raw: {len(package.keyframes)}).")

    depth_keyframes = [all_upright_keyframes[idx] for idx in selected_indices]
    depth_images = [all_upright_images[idx] for idx in selected_indices]

    return depth_keyframes, depth_images, intrinsics_portrait, all_upright_keyframes, all_upright_images, selected_indices


def main():
    parser = argparse.ArgumentParser(
        description="GlomeHomeTour Multi-View Sliding Window 3D Reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="scenes/bedroom_complete.zip",
        help="Path to scene archive (.zip) or directory",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=6,
        help="Chunk size N (number of views in cross-attention window). Safe range: 3-6 on 16GB VRAM",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=2,
        help="Overlap K between consecutive chunks. K=2 gives Sim(3) alignment; K=3 gives 3-way median consensus",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional custom root output directory. Defaults to <scenes_dir>/<scene_name>_depth_results",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache intermediate chunk depth maps",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit on number of keyframes to process (useful for smoke tests)",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.015,
        help="Voxel downsample grid size in meters (0.015 = 1.5 cm)",
    )
    parser.add_argument(
        "--min-consensus",
        type=int,
        default=1,
        help="Minimum number of overlapping camera frustums that must corroborate a 3D point (0 to disable)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="depth-anything/DA3-BASE",
        help="Depth Anything model checkpoint name or path",
    )
    parser.add_argument(
        "--skip-depth",
        action="store_true",
        help="Skip depth estimation pass and load existing depth maps from output-dir",
    )
    parser.add_argument(
        "--train-2dgs",
        action="store_true",
        help="Automatically launch Phase 5 2DGS training upon staging GS_input",
    )
    parser.add_argument(
        "--iterations-2dgs",
        type=int,
        default=3000,
        help="Number of 2DGS training iterations if --train-2dgs is enabled",
    )

    args = parser.parse_args()

    scene_path = Path(args.scene)
    if not scene_path.is_absolute():
        scene_path = _backend_dir / scene_path

    # Extract clean scene name (e.g., 'bedroom_complete.zip' -> 'bedroom_complete')
    scene_name = scene_path.stem
    scenes_parent = scene_path.parent

    # Standardized output root: <scene_name>_depth_results/
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = scenes_parent / f"{scene_name}_depth_results"

    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir) if args.cache_dir else (out_dir / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Standardized 2DGS training input subfolder: GS_input/ — everything downstream
    # of ingestion (depth maps, depth-selected images, transforms.json,
    # selected_keyframes.json, README.md, the .ply) lives here.
    gs_input_dir = out_dir / "GS_input"
    gs_images_dir = gs_input_dir / "images"
    depth_dir = gs_input_dir / "depth_maps"
    depth_images_dir = gs_input_dir / "images_depth_selected"
    raw_images_dir = gs_input_dir / "images_raw"
    gs_input_dir.mkdir(parents=True, exist_ok=True)
    gs_images_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    depth_images_dir.mkdir(parents=True, exist_ok=True)
    raw_images_dir.mkdir(parents=True, exist_ok=True)

    ply_path = gs_input_dir / f"{scene_name}_surfels.ply"

    print("=" * 70)
    print("  GLOME HOME TOUR: MULTI-VIEW 3D RECONSTRUCTION RUNNER")
    print("=" * 70)
    print(f"Scene:       {scene_path}")
    print(f"Model:       {args.model_name}")
    print(f"Chunk Size:  N = {args.chunk_size}")
    print(f"Overlap:     K = {args.overlap}")
    print(f"Output Root: {out_dir}")
    print(f"GS Input:    {gs_input_dir}")
    print(f"Cache Dir:   {cache_dir}")
    print(f"Voxel Grid:  {args.voxel_size * 100:.1f} cm")
    print("=" * 70)

    # 1. Ingestion & Keyframe Selection
    print("\n[Phase 1] Ingesting package and selecting depth-adaptive keyframes...")
    loader = PackageLoader()
    package = loader.load(scene_path)

    # Reusable 3-tier dataset: raw (pre quality-gate) frames, so this run's
    # frame selection at every stage can be reused without re-parsing the archive.
    for old_file in raw_images_dir.glob("*.jpg"):
        old_file.unlink()
    for i, raw_kf in enumerate(package.keyframes):
        img_bgr = cv2.cvtColor(raw_kf.load_image_rgb(), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(raw_images_dir / f"frame_{i:05d}.jpg"), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"Staged {len(package.keyframes)} raw (pre quality-gate) frames in {raw_images_dir}.")

    (
        depth_keyframes,
        depth_images,
        intrinsics,
        all_keyframes,
        all_images,
        selected_indices,
    ) = extract_depth_adaptive_keyframes(package, target_count=args.max_frames)
    m_depth_frames = len(depth_keyframes)
    m_all_frames = len(all_keyframes)
    print(f"Selected {m_depth_frames} depth-adaptive keyframes for 3D unprojection from {m_all_frames} quality-filtered frames (total raw: {len(package.keyframes)}).")

    # Stage the depth-selected keyframe images (subset of GS_input/images actually
    # fed to depth estimation) alongside a manifest of which indices were picked.
    for old_file in depth_images_dir.glob("*.jpg"):
        old_file.unlink()
    selected_keyframes_manifest = []
    for rank, idx in enumerate(selected_indices):
        rel_path = f"frame_{idx:05d}.jpg"
        img_bgr = cv2.cvtColor(all_images[idx], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(depth_images_dir / rel_path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        selected_keyframes_manifest.append({
            "rank": rank,
            "index": idx,
            "file_path": rel_path,
            "gs_input_file_path": f"images/{rel_path}",
            "depth_map": f"depth_maps/depth_{rank:04d}.npy",
        })
    with open(gs_input_dir / "selected_keyframes.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_filtered_frames": m_all_frames,
            "num_selected_for_depth": m_depth_frames,
            "selected_keyframes": selected_keyframes_manifest,
        }, f, indent=2)
    print(f"Staged {m_depth_frames} depth-selected keyframe images and selected_keyframes.json in {depth_images_dir}.")

    # 2. Multi-view Depth Estimation
    depth_maps: list[np.ndarray] = []
    conf_maps: Optional[list[np.ndarray]] = None

    if args.skip_depth:
        print("\n[Phase 2] Skipping depth estimation (--skip-depth). Loading from disk...")
        for i in range(m_depth_frames):
            fpath = depth_dir / f"depth_{i:04d}.npy"
            if not fpath.exists():
                raise FileNotFoundError(f"Expected depth map missing: {fpath}")
            depth_maps.append(np.load(fpath))
    else:
        print(f"\n[Phase 2] Running Multi-View Sliding Window Depth Estimation ({args.model_name}, N={args.chunk_size}, K={args.overlap})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Compute Device: {device}")
        
        # Build extrinsics (M, 4, 4) and intrinsics (M, 3, 3) for depth keyframes
        exts = np.stack([kf.transform_matrix for kf in depth_keyframes], axis=0)
        K_mat = np.array([
            [intrinsics.fl_x, 0.0, intrinsics.cx],
            [0.0, intrinsics.fl_y, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        ixts = np.repeat(K_mat[None, ...], m_depth_frames, axis=0)

        depth_model = DepthPriorEstimator(model_name=args.model_name, device=device)
        t_start = time.time()
        depth_maps, conf_maps, uncertainties = depth_model.estimate_depth_sliding_window(
            images=depth_images,
            extrinsics=exts,
            intrinsics=ixts,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            cache_dir=cache_dir,
        )
        total_depth_time = time.time() - t_start
        print(f"\n[Phase 2 Complete] Estimated {len(depth_maps)} multi-view consistent depth maps in {total_depth_time:.1f}s ({total_depth_time/60.0:.1f}m).")

        # Save depth maps & visualizations in output root
        print("Saving depth maps and diagnostic heatmaps...")
        for i, d in enumerate(depth_maps):
            np.save(depth_dir / f"depth_{i:04d}.npy", d)
            # Normalize for visualization
            d_min, d_max = np.percentile(d, 2), np.percentile(d, 98)
            d_norm = np.clip((d - d_min) / max(d_max - d_min, 1e-4), 0.0, 1.0)
            vis = cv2.applyColorMap((d_norm * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)
            cv2.imwrite(str(depth_dir / f"vis_{i:04d}.png"), vis)

            if uncertainties and uncertainties[i] is not None:
                np.save(depth_dir / f"unc_{i:04d}.npy", uncertainties[i])

    # 3. 3D Surfel Cloud Initialization & Unprojection
    print("\n[Phase 3] Unprojecting multi-view depth maps into 3D Surfel Cloud...")
    initializer = SurfelCloudInitializer(
        target_surfels=2_000_000,
        min_surfels=50_000,
        max_surfels=3_000_000,
        voxel_downsample_m=args.voxel_size,
        max_depth_m=None,  # Dynamic statistical depth ceiling
        min_consensus=args.min_consensus,
        max_depth_gradient=0.08,   # prunes depth-discontinuity flying pixels (see project_history.md 2026-09-10)
        max_grazing_angle_deg=78.0,  # prunes glancing silhouette-edge rays
        enable_sor=True,
        sor_k=20,
        sor_std_mul=2.2,
    )

    t_init = time.time()
    surfel_cloud = initializer.initialize_from_keyframes(
        keyframes=depth_keyframes,
        depth_maps=depth_maps,
        intrinsics=intrinsics,
        conf_maps=conf_maps,
        min_conf=0.3,
    )
    dt_init = time.time() - t_init

    print(f"Unprojected and filtered {len(surfel_cloud):,} surfels in {dt_init:.1f}s.")

    # 4. Export to Binary PLY directly into GS_input/
    print(f"\n[Phase 4] Exporting 3D Surfel Cloud to {ply_path}...")
    surfel_cloud.to_ply(ply_path)
    file_size_mb = ply_path.stat().st_size / (1024 * 1024)
    print(f"Export Complete: {ply_path} ({file_size_mb:.2f} MB)")

    # 5. Populate GS_input/ with ALL Quality-Filtered Images, transforms.json, and README.md
    print(f"\n[Phase 5] Staging 2DGS training bundle with all {m_all_frames} quality-filtered frames in {gs_input_dir}...")
    frames_json = []
    for i, (kf, img_rgb) in enumerate(zip(all_keyframes, all_images)):
        rel_img_path = f"images/frame_{i:05d}.jpg"
        abs_img_path = gs_input_dir / rel_img_path
        
        # Save high quality JPEG
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(abs_img_path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        frames_json.append({
            "file_path": rel_img_path,
            "timestamp_ns": int(kf.timestamp_ns),
            "fl_x": float(intrinsics.fl_x),
            "fl_y": float(intrinsics.fl_y),
            "cx": float(intrinsics.cx),
            "cy": float(intrinsics.cy),
            "transform_matrix": kf.transform_matrix.tolist(),
        })

    transforms_data = {
        "schema_version": "1.0.0",
        "camera_model": "OPENCV",
        "fl_x": float(intrinsics.fl_x),
        "fl_y": float(intrinsics.fl_y),
        "cx": float(intrinsics.cx),
        "cy": float(intrinsics.cy),
        "w": int(intrinsics.w),
        "h": int(intrinsics.h),
        "camera_angle_x": float(intrinsics.camera_angle_x),
        "k1": float(intrinsics.k1),
        "k2": float(intrinsics.k2),
        "p1": float(intrinsics.p1),
        "p2": float(intrinsics.p2),
        "frames": frames_json,
    }

    transforms_path = gs_input_dir / "transforms.json"
    with open(transforms_path, "w", encoding="utf-8") as f:
        json.dump(transforms_data, f, indent=2)

    # Write standardized README.md for next phase (Phase 3.5 2DGS Training)
    readme_content = f"""# 2DGS Training Input Package (`GS_input`)

Standardized package containing geometric priors and calibrated multi-view keyframes for **Phase 3.5 (2DGS Radiance Field Training & Density Optimization)**.

## Package Nomenclature & Directory Structure

```
GS_input/
├── README.md                      # This specification and contract file
├── transforms.json                # Camera intrinsics, extrinsics, and frame manifest (OpenCV / NeRF format)
├── selected_keyframes.json         # Manifest of which frames were used for depth estimation
├── {scene_name}_surfels.ply       # Initial 3D surfel point cloud with normals, scales, colors, opacities
├── depth_maps/                     # Per-keyframe depth/uncertainty/visualization from DA3 ({m_depth_frames} frames)
│   ├── depth_0000.npy
│   ├── unc_0000.npy
│   └── vis_0000.png
├── images_depth_selected/          # Subset of images/ actually fed to depth estimation
│   └── ... ({m_depth_frames} frames total)
└── images/                        # All quality-filtered training images
    ├── frame_00000.jpg
    ├── frame_00001.jpg
    └── ... ({m_all_frames} frames total)
```

## Asset Specification

1. **3D Surfel Model (`{scene_name}_surfels.ply`):**
   - **Total Surfels:** {len(surfel_cloud):,}
   - **Properties:** `float x, y, z`, `float nx, ny, nz`, `uchar red, green, blue`, `float scale_u, scale_v`, `float opacity`
   - **Coordinate Frame:** OpenGL / ARCore convention (+X right, +Y up, -Z forward, metric meters)
   - **Depth Keyframes Used for 3D Prior:** {m_depth_frames} depth-adaptive keyframes
   - **Usage:** Load directly using `Material2DGSModel.from_ply("{scene_name}_surfels.ply")` or `SurfelCloud.from_ply()`.

2. **Camera Calibration & Manifest (`transforms.json`):**
   - **Schema Compliance:** Validates against `shared/schemas/transforms.schema.json` (version 1.0.0).
   - **Frame Resolution:** {intrinsics.w} × {intrinsics.h} (Portrait)
   - **Focal Length:** fx = {intrinsics.fl_x:.2f}px, fy = {intrinsics.fl_y:.2f}px
   - **Principal Point:** cx = {intrinsics.cx:.2f}px, cy = {intrinsics.cy:.2f}px
   - **Transforms:** 4x4 camera-to-world transformation matrices `transform_matrix`.
   - **Total Supervised Frames:** {m_all_frames}

3. **Training Frames (`images/`):**
   - **Count:** {m_all_frames} quality-filtered, blur-free portrait keyframes.
   - **Format:** JPEG 95% quality RGB.
"""
    readme_path = gs_input_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"Staging Complete: {m_all_frames} images, transforms.json, and README.md written to {gs_input_dir}")

    # 6. Scene Statistics Summary
    pos = surfel_cloud.positions
    min_b = np.min(pos, axis=0)
    max_b = np.max(pos, axis=0)
    dims = max_b - min_b
    print("\n" + "=" * 70)
    print("  RECONSTRUCTION & GS STAGING SUMMARY")
    print("=" * 70)
    print(f"Scene Name:            {scene_name}")
    print(f"Model:                 {args.model_name}")
    print(f"Depth Keyframes:       {m_depth_frames} (from {m_all_frames} quality-filtered frames)")
    print(f"Supervised Frames:     {m_all_frames} staged in GS_input/")
    print(f"Total 3D Surfels:      {len(surfel_cloud):,}")
    print(f"Bounding Box (m):      X:[{min_b[0]:.2f}, {max_b[0]:.2f}]  Y:[{min_b[1]:.2f}, {max_b[1]:.2f}]  Z:[{min_b[2]:.2f}, {max_b[2]:.2f}]")
    print(f"Room Dimensions:       {dims[0]:.2f}m wide x {dims[1]:.2f}m long x {dims[2]:.2f}m high")
    print(f"Output Directory:      {out_dir}")
    print(f"GS Input Package:      {gs_input_dir}")
    print(f"Surfel Cloud PLY:      {ply_path}")
    print("=" * 70)

    # 7. Optional Automatic Phase 5 2DGS Optimization
    if args.train_2dgs:
        print("\n" + "=" * 70)
        print("  LAUNCHING 2DGS RADIANCE FIELD TRAINING")
        print("=" * 70)
        from run_scene_training import train_scene_from_ply_and_frames
        train_scene_from_ply_and_frames(
            scene_dir=gs_input_dir,
            output_dir=out_dir / "2dgs_output",
            iterations=args.iterations_2dgs,
            voxel_size_m=0.035,
            max_surfels=400_000,
            multi_scale=True,
        )


if __name__ == "__main__":
    main()
