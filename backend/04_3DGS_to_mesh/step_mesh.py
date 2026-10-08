#!/usr/bin/env python3
"""Fuse gsplat training-view depth into an Open3D TSDF mesh."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
from Utilities.pipeline_paths import bootstrap

bootstrap()
from Utilities.pipeline_step import StepContext, is_done

DEFAULT_WORKSPACE = _backend_dir / "current_scene"


def camera_intrinsic(camera, o3d):
    width, height = camera["width"], camera["height"]
    K = camera["K"]
    return o3d.camera.PinholeCameraIntrinsic(
        width, height,
        float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]),
    )


def valid_depth(depth, alpha, min_alpha, depth_trunc):
    valid = np.isfinite(depth) & np.isfinite(alpha) & (depth > 0) & (depth < depth_trunc) & (alpha >= min_alpha)
    return np.where(valid, depth, 0).astype(np.float32), int(valid.sum())


def extract_mesh(workspace, model_path, output, iteration, voxel_size, sdf_trunc,
                 depth_trunc, min_alpha, frame_stride, resolution, ctx):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Open3D is required for mesh extraction; install it in the backend environment: pip install open3d") from exc
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("gsplat mesh extraction requires an accessible CUDA/ROCm GPU")
    from PIL import Image
    from gsplat.rendering import rasterization
    from gsplat_train import load_cameras

    checkpoint_path = model_path / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "gsplat-stage03-v1":
        raise ValueError(f"Unsupported gsplat checkpoint: {checkpoint_path}")
    loaded_iter = int(checkpoint["iteration"])
    if iteration != -1 and iteration != loaded_iter:
        raise ValueError(f"Requested iteration {iteration}, checkpoint contains {loaded_iter}")
    source = model_path / "point_cloud" / f"iteration_{loaded_iter}" / "point_cloud.ply"
    if not source.is_file():
        raise FileNotFoundError(f"gsplat point cloud missing: {source}")
    cameras = load_cameras(workspace, resolution or int(checkpoint["resolution"]))[::frame_stride]
    if not cameras:
        raise RuntimeError(f"No training cameras found in {workspace}")
    splats = {name: value.to("cuda") for name, value in checkpoint["splats"].items()}
    quats = torch.nn.functional.normalize(splats["quats"], dim=-1)
    scales = splats["scales"].exp()
    opacities = splats["opacities"].sigmoid()
    # ED replaces colors with depth before rasterization; RGB comes from source photos.
    colors = torch.empty((len(splats["means"]), 1), device="cuda")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size, sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    valid_pixels = total_pixels = 0
    with torch.no_grad():
        for index, camera in enumerate(cameras, 1):
            rendered, opacity, _ = rasterization(
                means=splats["means"], quats=quats,
                scales=scales, opacities=opacities,
                colors=colors, viewmats=torch.as_tensor(camera["viewmat"], device="cuda")[None],
                Ks=torch.as_tensor(camera["K"], device="cuda")[None],
                width=camera["width"], height=camera["height"],
                render_mode="ED", packed=False,
            )
            depth = rendered[0, :, :, 0].cpu().numpy()
            alpha = opacity[0, :, :, 0].cpu().numpy()
            del rendered, opacity
            depth, count = valid_depth(depth, alpha, min_alpha, depth_trunc)
            valid_pixels += count
            total_pixels += depth.size
            if not count:
                continue
            with Image.open(camera["image_path"]) as image:
                color = np.asarray(image.convert("RGB").resize((camera["width"], camera["height"])))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(color)),
                o3d.geometry.Image(np.ascontiguousarray(depth)),
                depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False,
            )
            volume.integrate(rgbd, camera_intrinsic(camera, o3d), camera["viewmat"].astype(np.float64))
            del rgbd, color, depth, alpha
            if index % 20 == 0 or index == len(cameras):
                ctx.note(f"Fused {index}/{len(cameras)} training cameras")
    if not valid_pixels:
        raise RuntimeError(f"No valid rendered depth in {len(cameras)} training cameras; check alpha threshold and depth range")

    mesh = volume.extract_triangle_mesh()
    mesh.remove_unreferenced_vertices()
    if not len(mesh.triangles):
        raise RuntimeError(f"TSDF produced no triangles from {valid_pixels} valid pixels in {len(cameras)} training cameras; try a coarser voxel size or smaller frame stride")
    mesh.compute_vertex_normals()
    output.mkdir(parents=True, exist_ok=True)
    mesh_path = output / "mesh.ply"
    if not o3d.io.write_triangle_mesh(str(mesh_path), mesh):
        raise RuntimeError(f"Failed to write mesh: {mesh_path}")
    summary = {
        "selected_frames": len(cameras), "valid_pixels": valid_pixels,
        "valid_pixel_rate": valid_pixels / total_pixels,
        "frame_stride": frame_stride, "image_width": cameras[0]["width"],
        "image_height": cameras[0]["height"],
        "voxel_size": voxel_size, "sdf_trunc": sdf_trunc,
        "depth_trunc": depth_trunc, "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles), "checkpoint_iteration": loaded_iter,
        "engine": "gsplat",
        "source_model_path": str(source.resolve()),
    }
    (output / "mesh.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    ctx.metric("frames_processed", len(cameras))
    ctx.metric("mesh_path", str(mesh_path))
    ctx.note(f"Mesh: {mesh_path} ({summary['vertices']} vertices, {summary['triangles']} triangles)")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--model-path", help="gsplat model directory (default: workspace/03_FastGS_DNSplatter)")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--sdf-trunc", type=float, default=0.05)
    parser.add_argument("--depth-trunc", type=float, default=10.0)
    parser.add_argument("--min-alpha", type=float, default=0.5)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--resolution", type=int, choices=(1, 2, 4, 8),
                        help="Optional image downscale for lower-memory extraction; default uses checkpoint resolution")
    parser.add_argument("--output", help="Mesh directory (default: workspace/04_3DGS_to_mesh)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.iteration != -1 and args.iteration <= 0:
        parser.error("--iteration must be -1 or positive")
    if any(not math.isfinite(v) or v <= 0 for v in (args.voxel_size, args.sdf_trunc, args.depth_trunc)):
        parser.error("voxel size and truncation distances must be positive and finite")
    if not 0 <= args.min_alpha <= 1 or not math.isfinite(args.min_alpha):
        parser.error("--min-alpha must be between 0 and 1")
    if args.frame_stride < 1:
        parser.error("--frame-stride must be positive")
    workspace = Path(args.workspace).resolve()
    model_path = Path(args.model_path).resolve() if args.model_path else workspace / "03_FastGS_DNSplatter"
    output = Path(args.output).resolve() if args.output else workspace / "04_3DGS_to_mesh"
    if not args.force and is_done(workspace, "mesh", [output / "mesh.ply", output / "mesh.json"]):
        try:
            if json.loads((output / "mesh.json").read_text(encoding="utf-8")).get("engine") == "gsplat":
                print("[mesh] already done, skipping (use --force to re-run)")
                return 0
        except (ValueError, OSError):
            pass
    if not (model_path / "checkpoint.pt").is_file():
        raise FileNotFoundError(f"No gsplat checkpoint at {model_path / 'checkpoint.pt'}; run train first")
    with StepContext("mesh", workspace, artifacts_dir=output) as ctx:
        extract_mesh(workspace, model_path, output, args.iteration, args.voxel_size,
                     args.sdf_trunc, args.depth_trunc, args.min_alpha, args.frame_stride,
                     args.resolution, ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
