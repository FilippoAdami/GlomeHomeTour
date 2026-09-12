#!/usr/bin/env python3
"""GlomeHomeTour: Inspection Runner for First Two Chunks + Isolated Frames.

Runs multi-view sliding window depth estimation for keyframes 0-9 (2 chunks, N=6, K=2)
and exports:
1. Isolated frame PLYs: frame_0000_world.ply, frame_0001_world.ply
2. Individual chunk PLYs: chunk_0000.ply (frames 0-5), chunk_0001.ply (frames 4-9)
3. Combined 2-chunk surfel cloud: chunks_0_and_1_combined.ply
4. Diagnostic orthographic snapshot images.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

_backend_dir = Path(__file__).resolve().parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

_da3_src = _backend_dir / "third_party" / "depth_anything_3" / "src"
if _da3_src.is_dir() and str(_da3_src) not in sys.path:
    sys.path.insert(0, str(_da3_src))

from ingestion.package_loader import CameraIntrinsics, PackageLoader
from reconstruction.depth_priors import DepthPriorEstimator, arcore_c2w_to_da3_w2c
from reconstruction.initialization import SurfelCloudInitializer, SurfelCloud
from run_sliding_window_reconstruction import extract_depth_adaptive_keyframes


def write_ply(path: Path, pts: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]
    rgb = rgb[valid]
    with open(path, "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n".encode())
        rec = np.empty(len(pts), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        rec["p"] = pts.astype(np.float32)
        rec["c"] = rgb.astype(np.uint8)
        f.write(rec.tobytes())


def render_ortho_triptych(pts: np.ndarray, colors: np.ndarray, out_path: Path, title: str = "") -> None:
    """Renders top-down (XZ), side (YZ), and front (XY) orthographic projection images."""
    if len(pts) == 0:
        return
    valid = np.isfinite(pts).all(axis=1)
    pts, colors = pts[valid], colors[valid]
    
    img_size = 512
    margin = 0.1
    
    # Compute bounds
    min_b = np.min(pts, axis=0)
    max_b = np.max(pts, axis=0)
    center = (min_b + max_b) / 2.0
    extent = np.maximum(max_b - min_b, 1e-3)
    max_extent = np.max(extent) * (1.0 + margin)
    
    def project_view(axis_u: int, axis_v: int, invert_v: bool = True) -> np.ndarray:
        canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 30
        u = ((pts[:, axis_u] - (center[axis_u] - max_extent / 2.0)) / max_extent * (img_size - 1)).astype(np.int32)
        v = ((pts[:, axis_v] - (center[axis_v] - max_extent / 2.0)) / max_extent * (img_size - 1)).astype(np.int32)
        if invert_v:
            v = (img_size - 1) - v
        
        valid_px = (u >= 0) & (u < img_size) & (v >= 0) & (v < img_size)
        u, v = u[valid_px], v[valid_px]
        c = colors[valid_px]
        
        # Draw points
        canvas[v, u] = c[:, [2, 1, 0]]  # RGB to BGR for cv2
        return canvas

    v_top = project_view(0, 2, invert_v=True)    # XZ: Top-down (X right, Z up)
    v_front = project_view(0, 1, invert_v=True)  # XY: Front (X right, Y up)
    v_side = project_view(2, 1, invert_v=True)   # ZY: Side (Z right, Y up)
    
    # Add view titles
    cv2.putText(v_top, "Top-down (XZ)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(v_front, "Front (XY)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(v_side, "Side (ZY)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    triptych = np.hstack([v_top, v_front, v_side])
    if title:
        header = np.zeros((40, triptych.shape[1], 3), dtype=np.uint8)
        cv2.putText(header, title, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        triptych = np.vstack([header, triptych])
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), triptych)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scenes/bedroom_complete.zip")
    ap.add_argument("--out-dir", default="scenes/inspection_first_chunks")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading scene: {args.scene}...")
    loader = PackageLoader()
    package = loader.load(args.scene)

    print("Extracting keyframes...")
    keyframes_all, images_all, intrinsics = extract_depth_adaptive_keyframes(package)
    
    # Select keyframes for first 2 chunks (10 keyframes)
    num_frames = 10
    keyframes = keyframes_all[:num_frames]
    images = images_all[:num_frames]

    print(f"Selected first {num_frames} keyframes (covers Chunk 0: 0..5, Chunk 1: 4..9).")

    # Run Multi-View Sliding Window Depth Estimation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running DA3 Multi-View Depth Estimation on {device}...")
    
    exts = np.stack([kf.transform_matrix for kf in keyframes], axis=0)
    K_mat = np.array([
        [intrinsics.fl_x, 0.0, intrinsics.cx],
        [0.0, intrinsics.fl_y, intrinsics.cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    ixts = np.repeat(K_mat[None, ...], num_frames, axis=0)

    depth_model = DepthPriorEstimator(device=device)
    t_start = time.time()
    depth_maps, conf_maps, uncertainties = depth_model.estimate_depth_sliding_window(
        images=images,
        extrinsics=exts,
        intrinsics=ixts,
        chunk_size=6,
        overlap=2,
    )
    print(f"Depth estimation completed in {time.time() - t_start:.1f}s.")

    # 1. Export Isolated Single Frame PLYs (Frame 0 and Frame 1)
    print("\n--- Exporting Isolated Single Frames ---")
    initializer = SurfelCloudInitializer(
        target_surfels=500_000,
        min_surfels=1_000,
        max_surfels=1_000_000,
        voxel_downsample_m=0.015,
        min_consensus=0,
        enable_sor=True,
        sor_k=20,
        sor_std_mul=1.5,
    )

    for frame_idx in [0, 1]:
        single_cloud = initializer.initialize_from_keyframes(
            keyframes=[keyframes[frame_idx]],
            depth_maps=[depth_maps[frame_idx]],
            intrinsics=intrinsics,
            conf_maps=[conf_maps[frame_idx]] if conf_maps else None,
        )
        frame_ply_path = out_dir / f"frame_{frame_idx:04d}_world.ply"
        single_cloud.to_ply(frame_ply_path)
        print(f"Exported isolated Frame {frame_idx}: {frame_ply_path} ({len(single_cloud):,} surfels)")
        
        # Render snapshot
        colors_255 = (single_cloud.colors_rgb * 255.0).astype(np.uint8)
        render_ortho_triptych(
            single_cloud.positions, colors_255,
            out_dir / f"frame_{frame_idx:04d}_ortho.png",
            f"Isolated Frame {frame_idx} (World Coords)"
        )

    # 2. Export Individual Chunks (Chunk 0: 0..5, Chunk 1: 4..9)
    print("\n--- Exporting Individual Chunks ---")
    chunk_defs = [
        ("chunk_0000", slice(0, 6)),
        ("chunk_0001", slice(4, 10)),
    ]
    
    chunk_clouds = {}
    for chunk_name, slc in chunk_defs:
        chunk_cloud = initializer.initialize_from_keyframes(
            keyframes=keyframes[slc],
            depth_maps=depth_maps[slc],
            intrinsics=intrinsics,
            conf_maps=conf_maps[slc] if conf_maps else None,
        )
        chunk_ply_path = out_dir / f"{chunk_name}.ply"
        chunk_cloud.to_ply(chunk_ply_path)
        chunk_clouds[chunk_name] = chunk_cloud
        print(f"Exported {chunk_name}: {chunk_ply_path} ({len(chunk_cloud):,} surfels)")

        colors_255 = (chunk_cloud.colors_rgb * 255.0).astype(np.uint8)
        render_ortho_triptych(
            chunk_cloud.positions, colors_255,
            out_dir / f"{chunk_name}_ortho.png",
            f"{chunk_name} (World Coords)"
        )

    # 3. Export Combined 2-Chunk Surfel Cloud
    print("\n--- Exporting Combined 2-Chunk Cloud ---")
    comb_initializer = SurfelCloudInitializer(
        target_surfels=500_000,
        min_surfels=10_000,
        max_surfels=1_000_000,
        voxel_downsample_m=0.015,
        min_consensus=0,
        enable_sor=True,
        sor_k=20,
        sor_std_mul=1.5,
    )
    combined_cloud = comb_initializer.initialize_from_keyframes(
        keyframes=keyframes,
        depth_maps=depth_maps,
        intrinsics=intrinsics,
        conf_maps=conf_maps,
    )
    comb_ply_path = out_dir / "chunks_0_and_1_combined.ply"
    combined_cloud.to_ply(comb_ply_path)
    print(f"Exported combined 2-chunk cloud: {comb_ply_path} ({len(combined_cloud):,} surfels)")

    colors_255 = (combined_cloud.colors_rgb * 255.0).astype(np.uint8)
    render_ortho_triptych(
        combined_cloud.positions, colors_255,
        out_dir / "chunks_0_and_1_combined_ortho.png",
        "Combined Chunks 0 & 1 (World Coords, Solid)"
    )

    # Summary
    print("\n" + "=" * 70)
    print("  INSPECTION RUN COMPLETE")
    print("=" * 70)
    print(f"Output Directory: {out_dir.resolve()}")
    print("Generated Artifacts:")
    print(f" - Isolated Frame 0:  {out_dir / 'frame_0000_world.ply'}")
    print(f" - Isolated Frame 1:  {out_dir / 'frame_0001_world.ply'}")
    print(f" - Chunk 0 (frames 0..5): {out_dir / 'chunk_0000.ply'}")
    print(f" - Chunk 1 (frames 4..9): {out_dir / 'chunk_0001.ply'}")
    print(f" - Combined Chunks 0 & 1: {out_dir / 'chunks_0_and_1_combined.ply'}")
    print("=" * 70)

if __name__ == "__main__":
    main()
