#!/usr/bin/env python3
"""GlomeHomeTour Backend: Comprehensive End-to-End Reconstruction Benchmark Runner.

Executes full Phase 3 reconstruction from zip package to 3D surfel PLY model,
measuring execution times, GPU VRAM allocation, and asset file sizes.
"""

import os
import sys
import time
from pathlib import Path

# Set environment before torch imports
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("MIOPEN_USER_DB_PATH", "/tmp/miopen")

import numpy as np
import torch
import cv2

_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
from Utilities.pipeline_paths import bootstrap
bootstrap()

from package_loader import PackageLoader
from run_sliding_window_reconstruction import extract_depth_adaptive_keyframes
from depth_priors import DepthPriorEstimator, arcore_c2w_to_da3_w2c
from initialization import SurfelCloudInitializer


def main():
    scene_path = "scenes/bedroom_complete.zip"
    scene_name = Path(scene_path).stem
    out_dir = Path("scenes") / f"{scene_name}_depth_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = out_dir / "depth_maps"
    depth_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    gs_input_dir = out_dir / "GS_input"
    gs_images_dir = gs_input_dir / "images"
    gs_input_dir.mkdir(parents=True, exist_ok=True)
    gs_images_dir.mkdir(parents=True, exist_ok=True)

    ply_path = gs_input_dir / f"{scene_name}_surfels.ply"

    print("=" * 75)
    print("  GLOME HOME TOUR: FULL PIPELINE PERFORMANCE BENCHMARK REVIEW")
    print("=" * 75)
    print(f"Scene Archive:      {scene_path}")
    print(f"Depth Model:        depth-anything/DA3-BASE (Apache 2.0 Commercial)")
    print(f"Target GPU:         {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0.0
    print(f"Total GPU VRAM:     {total_vram_gb:.2f} GB")
    print(f"Output Directory:   {out_dir}")
    print(f"GS Input Package:   {gs_input_dir}")
    print("=" * 75)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t_global_start = time.perf_counter()

    # --- Phase 1: Ingestion & Keyframe Selection ---
    print("\n[Phase 1] Ingesting Capture Package & Selecting Portrait Keyframes...")
    t1_start = time.perf_counter()
    loader = PackageLoader()
    package = loader.load(scene_path)
    (
        depth_keyframes,
        depth_images,
        intrinsics,
        all_keyframes,
        all_images,
    ) = extract_depth_adaptive_keyframes(package)
    t1_end = time.perf_counter()
    t1_duration = t1_end - t1_start
    m_depth_frames = len(depth_keyframes)
    m_all_frames = len(all_keyframes)
    print(f"-> Phase 1 Complete in {t1_duration:.2f}s (Selected {m_depth_frames} depth keyframes and {m_all_frames} supervised frames from {len(package.keyframes)} raw frames).")

    # --- Phase 2: Multi-View Sliding Window Depth Estimation ---
    print("\n[Phase 2] Multi-View Sliding Window Depth Estimation (DA3-BASE, N=6, K=2)...")
    t2_start = time.perf_counter()

    # These are raw ARCore camera-to-world poses. The estimator takes OpenCV
    # world-to-camera (the pipeline feeds it COLMAP poses, already in that form),
    # so this legacy path has to convert for itself.
    exts = arcore_c2w_to_da3_w2c(np.stack([kf.transform_matrix for kf in depth_keyframes], axis=0))
    K_mat = np.array([
        [intrinsics.fl_x, 0.0, intrinsics.cx],
        [0.0, intrinsics.fl_y, intrinsics.cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    ixts = np.repeat(K_mat[None, ...], m_depth_frames, axis=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    depth_model = DepthPriorEstimator(model_name="depth-anything/DA3-BASE", device=device)

    depth_maps, conf_maps, uncertainties = depth_model.estimate_depth_sliding_window(
        images=depth_images,
        extrinsics=exts,
        intrinsics=ixts,
        chunk_size=6,
        overlap=2,
        cache_dir=cache_dir,
    )
    t2_end = time.perf_counter()
    t2_duration = t2_end - t2_start

    # Save depth maps to disk for inspection
    for i, d in enumerate(depth_maps):
        np.save(depth_dir / f"depth_{i:04d}.npy", d)

    vram_peak_alloc = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    vram_peak_res = torch.cuda.max_memory_reserved() / (1024**3) if torch.cuda.is_available() else 0.0

    print(f"-> Phase 2 Complete in {t2_duration:.2f}s ({t2_duration/60.0:.2f} min).")
    print(f"   Peak GPU VRAM Allocated: {vram_peak_alloc:.2f} GB ({vram_peak_alloc/total_vram_gb*100:.1f}% of {total_vram_gb:.2f} GB)")
    print(f"   Peak GPU VRAM Reserved:  {vram_peak_res:.2f} GB ({vram_peak_res/total_vram_gb*100:.1f}%)")

    # --- Phase 3: Surfel Cloud Initialization & Multi-View Consensus ---
    print("\n[Phase 3] Surfel Cloud Unprojection & Multi-View Geometric Consensus...")
    t3_start = time.perf_counter()

    initializer = SurfelCloudInitializer(
        target_surfels=2_000_000,
        min_surfels=50_000,
        max_surfels=3_000_000,
        voxel_downsample_m=0.015,
        max_depth_m=None,  # Dynamic statistical ceiling!
        min_consensus=1,
        enable_sor=True,
        sor_k=20,
        sor_std_mul=2.2,
    )

    surfel_cloud = initializer.initialize_from_keyframes(
        keyframes=depth_keyframes,
        depth_maps=depth_maps,
        intrinsics=intrinsics,
        conf_maps=conf_maps,
        min_conf=0.3,
    )
    t3_end = time.perf_counter()
    t3_duration = t3_end - t3_start
    num_surfels = len(surfel_cloud)

    print(f"-> Phase 3 Complete in {t3_duration:.2f}s (Unprojected {num_surfels:,} multi-view consistent surfels).")

    # --- Phase 4: Asset Export & GS Staging ---
    print("\n[Phase 4] Exporting Binary PLY Surfel Model & Staging GS Training Bundle...")
    t4_start = time.perf_counter()

    surfel_cloud.to_ply(ply_path)
    ply_size_mb = ply_path.stat().st_size / (1024 * 1024)

    # Save all quality filtered images and transforms.json
    frames_json = []
    for i, (kf, img_rgb) in enumerate(zip(all_keyframes, all_images)):
        rel_img_path = f"images/frame_{i:05d}.jpg"
        abs_img_path = gs_input_dir / rel_img_path
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

    import json
    with open(gs_input_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(transforms_data, f, indent=2)

    readme_content = f"""# 2DGS Training Input Package (`GS_input`)

Standardized package containing geometric priors and calibrated multi-view keyframes for **Phase 3.5 (2DGS Radiance Field Training & Density Optimization)**.

## Package Nomenclature & Directory Structure

```
GS_input/
├── README.md                      # This specification and contract file
├── transforms.json                # Camera intrinsics, extrinsics, and frame manifest (OpenCV / NeRF format)
├── {scene_name}_surfels.ply       # Initial 3D surfel point cloud with normals, scales, colors, opacities
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
    with open(gs_input_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    t4_end = time.perf_counter()
    t4_duration = t4_end - t4_start

    t_global_end = time.perf_counter()
    total_time = t_global_end - t_global_start

    # Scene statistics
    pts = surfel_cloud.positions
    min_b = pts.min(axis=0)
    max_b = pts.max(axis=0)
    dims = max_b - min_b

    print("\n" + "=" * 75)
    print("  GLOME HOME TOUR: RECONSTRUCTION PERFORMANCE REVIEW SUMMARY")
    print("=" * 75)
    print(f"Target Scene:            {scene_path}")
    print(f"Depth Keyframes:         {m_depth_frames} (from {m_all_frames} quality-filtered, {len(package.keyframes)} raw frames)")
    print(f"Supervised Frames:       {m_all_frames} staged in GS_input/")
    print(f"Total 3D Surfels:        {num_surfels:,}")
    print(f"Room Bounding Box (m):   X:[{min_b[0]:.2f}, {max_b[0]:.2f}]  Y:[{min_b[1]:.2f}, {max_b[1]:.2f}]  Z:[{min_b[2]:.2f}, {max_b[2]:.2f}]")
    print(f"Room Dimensions:         {dims[0]:.2f}m wide x {dims[2]:.2f}m long x {dims[1]:.2f}m high")
    print("-" * 75)
    print("EXECUTION TIME BREAKDOWN:")
    print(f"  1. Ingestion & Filtering: {t1_duration:6.2f}s ({t1_duration/total_time*100:4.1f}%)")
    print(f"  2. DA3-BASE Depth Prior:  {t2_duration:6.2f}s ({t2_duration/total_time*100:4.1f}%)  [{t2_duration/60.0:.2f} min]")
    print(f"  3. Surfel Unprojection:   {t3_duration:6.2f}s ({t3_duration/total_time*100:4.1f}%)")
    print(f"  4. PLY & GS Staging:      {t4_duration:6.2f}s ({t4_duration/total_time*100:4.1f}%)")
    print(f"  TOTAL END-TO-END TIME:    {total_time:6.2f}s ({total_time/60.0:.2f} min)")
    print("-" * 75)
    print("HARDWARE & MEMORY UTILIZATION:")
    print(f"  Compute Device:          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Peak VRAM Allocated:     {vram_peak_alloc:.2f} GB ({vram_peak_alloc/total_vram_gb*100:.1f}%)")
    print(f"  Peak VRAM Reserved:      {vram_peak_res:.2f} GB ({vram_peak_res/total_vram_gb*100:.1f}%)")
    print(f"  Total VRAM Available:    {total_vram_gb:.2f} GB")
    print("-" * 75)
    print("GENERATED ASSET DELIVERABLES:")
    print(f"  Binary Surfel PLY:       {ply_path} ({ply_size_mb:.2f} MB)")
    print(f"  GS Input Package:        {gs_input_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
