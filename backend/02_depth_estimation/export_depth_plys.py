#!/usr/bin/env python3
"""Export .npy depth maps to .ply point clouds with aligned Blender camera frustums.

For each keyframe depth map (.npy):
1. Unprojects depth map pixels to 3D world coordinates using COLMAP camera poses & intrinsics.
2. Samples corresponding RGB colors from the keyframe image (if available).
3. Adds a standard Blender camera model (wireframe frustum pyramid + top indicator notch)
   located and aligned at the camera pose.
4. Exports binary .ply files to `<workspace>/02_depth_estimation/depth/depth_files/`.

Usage:
    python3 02_depth_estimation/export_depth_plys.py
    python3 02_depth_estimation/export_depth_plys.py [--workspace DIR] [--stride N] [--max-frames N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
from Utilities.pipeline_paths import bootstrap

bootstrap()

DEFAULT_WORKSPACE = _backend_dir / "current_scene"
STAGE_DIRNAME = "02_depth_estimation"


def build_camera_frustum_geom(
    c2w: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    w: int,
    h: int,
    frustum_depth: float = 0.2,
    color_rgb: tuple[int, int, int] = (255, 0, 0),  # Red camera wireframe
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build vertices and edges for a standard Blender camera model wireframe frustum.

    Returns:
        cam_pts_world: (6, 3) float32 3D world positions of camera vertices
        cam_colors: (6, 3) uint8 RGB colors for vertices
        cam_edges: (10, 2) int32 vertex index pairs for wireframe lines
        edge_colors: (10, 3) uint8 RGB colors for edges
    """
    s = frustum_depth
    half_w = (w / 2.0)
    half_h = (h / 2.0)

    # 4 frustum corners in OpenCV camera frame (+X right, +Y down, +Z fwd)
    tl_cam = np.array([-(half_w / fx) * s, -(half_h / fy) * s, s], dtype=np.float32)
    tr_cam = np.array([(half_w / fx) * s, -(half_h / fy) * s, s], dtype=np.float32)
    br_cam = np.array([(half_w / fx) * s, (half_h / fy) * s, s], dtype=np.float32)
    bl_cam = np.array([-(half_w / fx) * s, (half_h / fy) * s, s], dtype=np.float32)
    
    # Apex (camera origin)
    c_cam = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Top notch / direction indicator (Blender camera top notch)
    tn_cam = np.array([0.0, -(half_h / fy) * 1.35 * s, s], dtype=np.float32)

    cam_pts_local = np.stack([c_cam, tl_cam, tr_cam, br_cam, bl_cam, tn_cam], axis=0)  # (6, 3)

    # Transform to world space using camera-to-world (c2w)
    r_c2w = c2w[:3, :3]
    t_c2w = c2w[:3, 3]
    cam_pts_world = (cam_pts_local @ r_c2w.T) + t_c2w

    r, g, b = color_rgb
    cam_colors = np.full((6, 3), [r, g, b], dtype=np.uint8)

    # Wireframe edges:
    # 0: C, 1: TL, 2: TR, 3: BR, 4: BL, 5: TN
    cam_edges = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4],  # Apex to 4 corners
        [1, 2], [2, 3], [3, 4], [4, 1],  # Frustum rectangular frame
        [1, 5], [2, 5],                  # Top notch triangle
    ], dtype=np.int32)

    edge_colors = np.full((len(cam_edges), 3), [r, g, b], dtype=np.uint8)

    return cam_pts_world.astype(np.float32), cam_colors, cam_edges, edge_colors


def save_ply_with_camera(
    out_path: Path,
    pts_world: np.ndarray,
    colors_rgb: np.ndarray,
    cam_pts_world: np.ndarray,
    cam_colors: np.ndarray,
    cam_edges: np.ndarray,
    edge_colors: np.ndarray,
) -> None:
    """Save unprojected point cloud + camera frustum wireframe as a binary PLY file."""
    n_cloud = len(pts_world)
    n_cam = len(cam_pts_world)
    total_vertices = n_cloud + n_cam
    n_edges = len(cam_edges)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {total_vertices}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        f"element edge {n_edges}\n"
        "property int vertex1\n"
        "property int vertex2\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    vertex_dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])

    edge_dtype = np.dtype([
        ("vertex1", "<i4"), ("vertex2", "<i4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])

    v_data = np.empty(total_vertices, dtype=vertex_dtype)
    
    # Cloud vertices
    v_data["x"][:n_cloud] = pts_world[:, 0]
    v_data["y"][:n_cloud] = pts_world[:, 1]
    v_data["z"][:n_cloud] = pts_world[:, 2]
    v_data["red"][:n_cloud] = colors_rgb[:, 0]
    v_data["green"][:n_cloud] = colors_rgb[:, 1]
    v_data["blue"][:n_cloud] = colors_rgb[:, 2]

    # Camera vertices
    v_data["x"][n_cloud:] = cam_pts_world[:, 0]
    v_data["y"][n_cloud:] = cam_pts_world[:, 1]
    v_data["z"][n_cloud:] = cam_pts_world[:, 2]
    v_data["red"][n_cloud:] = cam_colors[:, 0]
    v_data["green"][n_cloud:] = cam_colors[:, 1]
    v_data["blue"][n_cloud:] = cam_colors[:, 2]

    # Camera edges (offset vertex indices by n_cloud)
    e_data = np.empty(n_edges, dtype=edge_dtype)
    e_data["vertex1"] = cam_edges[:, 0] + n_cloud
    e_data["vertex2"] = cam_edges[:, 1] + n_cloud
    e_data["red"] = edge_colors[:, 0]
    e_data["green"] = edge_colors[:, 1]
    e_data["blue"] = edge_colors[:, 2]

    with open(out_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(v_data.tobytes())
        f.write(e_data.tobytes())


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE), help="Path to workspace directory")
    parser.add_argument("--out-dir", default=None, help="Output directory for PLY files (default: <workspace>/02_depth_estimation/depth/depth_files)")
    parser.add_argument("--stride", type=int, default=2, help="Subsampling pixel stride for point cloud generation (default: 2)")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to convert (default: all)")
    parser.add_argument("--frustum-size", type=float, default=0.25, help="Frustum visualizer depth in meters (default: 0.25)")
    parser.add_argument("--max-depth-gradient", type=float, default=0.15, help="Max depth gradient threshold to filter flying edge pixels (default: 0.15, set 0 to disable)")
    parser.add_argument("--clean", action="store_true", help="Run in-place GPU multi-view cleaning and depth alignment before exporting")
    args = parser.parse_args(argv)

    ws = Path(args.workspace).resolve()
    if args.clean:
        from clean_depth_maps import clean_depth_maps_in_place
        clean_depth_maps_in_place(ws)

    depth_dir = ws / STAGE_DIRNAME / "depth"
    if not depth_dir.exists():
        depth_dir = ws / "depth"

    depth_maps_dir = depth_dir / "depth_maps"
    if not depth_maps_dir.is_dir():
        raise FileNotFoundError(f"Depth maps directory not found at: {depth_maps_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else depth_dir / "depth_files"
    out_dir.mkdir(parents=True, exist_ok=True)

    poses_path = depth_dir / "poses_da3.npz"
    if not poses_path.is_file():
        raise FileNotFoundError(f"Camera poses file not found at: {poses_path}")

    poses_data = np.load(poses_path)
    w2c_mats = poses_data["w2c"]  # (N, 4, 4)
    k_mats = poses_data["K"]      # (N, 3, 3)
    pose_names = [str(n) for n in poses_data["names"]]

    images_dir = ws / "images"

    npy_files = sorted(depth_maps_dir.glob("*.npy"))
    if args.max_frames:
        npy_files = npy_files[:args.max_frames]

    print(f"[Export PLYs] Found {len(npy_files)} depth maps. Saving PLYs to: {out_dir}")

    name_to_idx = {name: i for i, name in enumerate(pose_names)}

    converted_count = 0
    stride = max(1, args.stride)

    for npy_path in npy_files:
        stem = npy_path.stem  # e.g. "frame_00000"
        img_name = f"{stem}.jpg"

        if img_name in name_to_idx:
            idx = name_to_idx[img_name]
        elif stem in name_to_idx:
            idx = name_to_idx[stem]
        else:
            print(f"[Warning] Skipping {npy_path.name}: pose not found in poses_da3.npz")
            continue

        depth = np.load(npy_path).astype(np.float32)
        h, w = depth.shape[:2]

        w2c = w2c_mats[idx]
        K = k_mats[idx]
        c2w = np.linalg.inv(w2c)

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        # Load matching RGB image if available
        rgb_path = images_dir / img_name
        if rgb_path.is_file():
            img_bgr = cv2.imread(str(rgb_path))
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                if img_rgb.shape[:2] != (h, w):
                    img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                img_rgb = np.full((h, w, 3), 180, dtype=np.uint8)
        else:
            img_rgb = np.full((h, w, 3), 180, dtype=np.uint8)

        # Unproject pixel grid
        y_sub, x_sub = np.mgrid[0:h:stride, 0:w:stride]
        y_flat = y_sub.flatten()
        x_flat = x_sub.flatten()

        d_flat = depth[y_flat, x_flat]
        valid = (d_flat > 0.1) & np.isfinite(d_flat)

        if not np.any(valid):
            print(f"[Warning] No valid depth points for {stem}")
            continue

        y_valid = y_flat[valid]
        x_valid = x_flat[valid]
        d_valid = d_flat[valid]

        # 3D points in OpenCV camera coordinates (+X right, +Y down, +Z fwd)
        x_cam = (x_valid - cx) * d_valid / fx
        y_cam = (y_valid - cy) * d_valid / fy
        z_cam = d_valid
        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

        # Transform to world space
        r_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]
        pts_world = (pts_cam @ r_c2w.T) + t_c2w
        colors_rgb = img_rgb[y_valid, x_valid]

        # Build Blender camera frustum geometry
        cam_pts_world, cam_colors, cam_edges, edge_colors = build_camera_frustum_geom(
            c2w=c2w,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            w=w,
            h=h,
            frustum_depth=args.frustum_size,
            color_rgb=(255, 0, 0),  # Red wireframe camera
        )

        out_ply_path = out_dir / f"{stem}.ply"
        save_ply_with_camera(
            out_path=out_ply_path,
            pts_world=pts_world,
            colors_rgb=colors_rgb,
            cam_pts_world=cam_pts_world,
            cam_colors=cam_colors,
            cam_edges=cam_edges,
            edge_colors=edge_colors,
        )
        converted_count += 1

    print(f"[Export PLYs] Successfully exported {converted_count} PLY files with camera frustums to {out_dir}")


if __name__ == "__main__":
    main()
