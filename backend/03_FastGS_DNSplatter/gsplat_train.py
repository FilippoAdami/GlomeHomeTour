#!/usr/bin/env python3
"""Train a COLMAP scene with Apache-2.0 gsplat.

This file is intentionally standalone.  It does not import the legacy FastGS
or Inria Gaussian Splatting modules.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


def _rotation(q: np.ndarray) -> np.ndarray:
    """COLMAP quaternion (w, x, y, z) to a world-to-camera rotation."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(((1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
                     (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
                     (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y))), dtype=np.float32)


def load_cameras(workspace: str | Path, resolution: int = 1) -> list[dict]:
    """Read COLMAP text cameras into gsplat-ready camera dictionaries.

    ``viewmat`` is a world-to-camera transform and ``K`` is scaled together
    with the returned image dimensions when ``resolution`` is greater than 1.
    """
    if resolution < 1:
        raise ValueError("resolution must be a positive integer")
    workspace = Path(workspace)
    sparse = workspace / "sparse" / "0"
    cameras_file, images_file = sparse / "cameras.txt", sparse / "images.txt"
    if not cameras_file.is_file() or not images_file.is_file():
        raise FileNotFoundError("gsplat trainer requires sparse/0/cameras.txt and images.txt")
    intrinsics: dict[int, tuple[int, int, float, float, float, float]] = {}
    for line in cameras_file.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        item = line.split()
        camera_id, model, width, height = int(item[0]), item[1], int(item[2]), int(item[3])
        p = [float(x) for x in item[4:]]
        if model == "SIMPLE_PINHOLE":
            fx = fy = p[0]; cx, cy = p[1:3]
        elif model == "PINHOLE":
            fx, fy, cx, cy = p[:4]
        else:
            raise ValueError(f"unsupported COLMAP camera model {model}; undistort to PINHOLE first")
        intrinsics[camera_id] = width, height, fx, fy, cx, cy
    result = []
    lines = images_file.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        if not lines[index] or lines[index].startswith("#"):
            index += 1
            continue
        header = lines[index]
        if index + 1 >= len(lines):
            raise ValueError("malformed COLMAP images.txt: image header has no points line")
        index += 2  # The following line is the (possibly empty) POINTS2D list.
        item = header.split()
        camera_id, image_name = int(item[8]), item[9]
        width, height, fx, fy, cx, cy = intrinsics[camera_id]
        image_path = workspace / "images" / image_name
        if not image_path.is_file():
            raise FileNotFoundError(f"COLMAP image missing: {image_path}")
        new_width, new_height = width // resolution, height // resolution
        if not new_width or not new_height:
            raise ValueError(f"resolution {resolution} is too large for {image_name}")
        K = np.array(((fx / resolution, 0, cx / resolution),
                      (0, fy / resolution, cy / resolution), (0, 0, 1)), dtype=np.float32)
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = _rotation(np.asarray(item[1:5], dtype=np.float64))
        viewmat[:3, 3] = np.asarray(item[5:8], dtype=np.float32)
        result.append({"image_path": image_path, "width": new_width, "height": new_height,
                       "K": K, "viewmat": viewmat})
    if not result:
        raise ValueError("COLMAP reconstruction has no registered images")
    return result


def _load_points(path: Path, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    vertices = PlyData.read(path)["vertex"]
    names = vertices.data.dtype.names or ()
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"{path} has no vertex positions")
    xyz = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float32)
    rgb = (np.column_stack((vertices["red"], vertices["green"], vertices["blue"])).astype(np.float32) / 255
           if {"red", "green", "blue"}.issubset(names) else np.full_like(xyz, .5))
    if not len(xyz):
        raise ValueError(f"{path} has no points")
    if max_points and len(xyz) > max_points:
        chosen = np.random.default_rng(seed).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[chosen], rgb[chosen]
    return xyz, rgb


def _scene_scale(cameras: list[dict]) -> float:
    centers = np.stack([-c["viewmat"][:3, :3].T @ c["viewmat"][:3, 3] for c in cameras])
    return max(1.1 * float(np.linalg.norm(centers - centers.mean(axis=0), axis=1).max()), 1e-3)


def _make_splats(xyz: np.ndarray, rgb: np.ndarray, device: torch.device) -> torch.nn.ParameterDict:
    n = len(xyz)
    if n < 4:
        raise ValueError("at least four initial points are required")
    distances = cKDTree(xyz).query(xyz, k=4, workers=-1)[0][:, 1:]
    scales = np.log(np.maximum(np.sqrt(np.square(distances).mean(axis=1)), 1e-5)).astype(np.float32)
    sh0 = (torch.from_numpy(rgb).float() - .5) / .28209479177387814
    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.from_numpy(xyz).float()),
        "scales": torch.nn.Parameter(torch.from_numpy(np.repeat(scales[:, None], 3, axis=1))),
        "quats": torch.nn.Parameter(torch.tensor([1., 0., 0., 0.]).repeat(n, 1)),
        "opacities": torch.nn.Parameter(torch.full((n,), math.log(.1 / .9))),
        "sh0": torch.nn.Parameter(sh0[:, None]),
        "shN": torch.nn.Parameter(torch.zeros(n, 15, 3)),
    }).to(device)


def _image_tensor(camera: dict, device: torch.device) -> torch.Tensor:
    if "rgb" not in camera:
        with Image.open(camera["image_path"]) as image:
            image = image.convert("RGB").resize((camera["width"], camera["height"]), Image.Resampling.LANCZOS)
            camera["rgb"] = torch.from_numpy(np.asarray(image).copy())
    if device.type == "cuda" and not camera["rgb"].is_pinned():
        camera["rgb"] = _pin_for_transfer(camera["rgb"])
    return camera["rgb"].to(device=device, dtype=torch.float32,
                             non_blocking=device.type == "cuda" and camera["rgb"].is_pinned()).div_(255)


def _pin_for_transfer(value: torch.Tensor) -> torch.Tensor:
    try:
        return value.pin_memory()
    except RuntimeError:
        return value


def _camera_tensor(camera: dict, name: str, device: torch.device) -> torch.Tensor:
    """Cache small, immutable camera tensors on the training device."""
    tensors = camera.setdefault("_tensors", {})
    key = name, str(device)
    if key not in tensors:
        tensors[key] = torch.from_numpy(camera[name])[None].to(device)
    return tensors[key]


def _map_tensor(root: Path, camera: dict, channels: int, device: torch.device,
                cache: dict | None = None) -> torch.Tensor | None:
    """Cache preprocessed CPU maps; training supplies shared memory ceilings."""
    if cache is None:
        cache = camera.setdefault("_maps", {"values": OrderedDict(), "bytes": 0})
    maps = cache["values"]
    key = root, camera["image_path"], camera["height"], camera["width"], channels
    if key in maps:
        value = maps[key]
        maps.move_to_end(key)
    else:
        path = root / f"{camera['image_path'].stem}.npy"
        value = None
        if path.is_file():
            data = np.load(path, allow_pickle=False)
            if channels == 1:
                data = data[..., None]
            value = torch.from_numpy(data)
            # Keep stage 02's float16 storage; conversion to float32 is exact.
            if value.dtype != torch.float16:
                value = value.float()
            size = camera["height"], camera["width"]
            if value.shape[:2] != size:
                value = F.interpolate(value.float().permute(2, 0, 1)[None], size, mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
            value = value.contiguous()
        size_bytes = 0 if value is None else value.numel() * value.element_size()
        limit = cache.get("limit", 8 * 1024**3)
        while maps and cache["bytes"] + size_bytes > limit:
            _, old = maps.popitem(last=False)
            if old is not None:
                old_bytes = old.numel() * old.element_size()
                cache["bytes"] -= old_bytes
        if size_bytes <= limit:
            maps[key] = value
            cache["bytes"] += size_bytes
    if value is None:
        return None
    return value.to(device=device, non_blocking=True).float()


def _depth_loss(pred: torch.Tensor, target: torch.Tensor, mode: str,
                render_valid: torch.Tensor | None = None) -> torch.Tensor:
    valid = torch.isfinite(target) & (target > 0) & torch.isfinite(pred) & (pred > 0)
    if render_valid is not None:
        valid = valid & render_valid
    count = valid.sum().clamp_min(1)
    p, t = torch.where(valid, pred, 0), torch.where(valid, target, 0)
    if mode == "pearson":
        pm, tm = p.sum() / count, t.sum() / count
        pc, tc = torch.where(valid, p - pm, 0), torch.where(valid, t - tm, 0)
        loss = 1 - ((pc * tc).sum() / count) / (
            (pc.square().sum() / count).clamp_min(1e-12).sqrt()
            * (tc.square().sum() / count).clamp_min(1e-12).sqrt() + 1e-6)
    elif mode == "scale_shift":
        pm, tm = p.sum() / count, t.sum() / count
        pc, tc = torch.where(valid, p - pm, 0), torch.where(valid, t - tm, 0)
        scale = ((pc * tc).sum() / count) / (pc.square().sum() / count).clamp_min(1e-6)
        loss = torch.where(valid, scale * pc + tm - t, 0).abs().sum() / count
    elif mode == "log_l1":
        loss = (torch.where(valid, torch.log(p.clamp_min(1e-5)) - torch.log(t.clamp_min(1e-5)), 0).abs().sum() / count)
    else:
        loss = (p - t).abs().sum() / count
    return torch.where(count >= 16, loss, 0)


def _normals_from_depth(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Camera-space +Z normals from central depth differences."""
    h, w = depth.shape[:2]
    y, x = torch.meshgrid(torch.arange(1, h - 1, device=depth.device),
                          torch.arange(1, w - 1, device=depth.device), indexing="ij")
    dx = (depth[1:-1, 2:] - depth[1:-1, :-2]) * .5
    dy = (depth[2:, 1:-1] - depth[:-2, 1:-1]) * .5
    normal = torch.stack((-K[0, 0] * dx, -K[1, 1] * dy,
                          depth[1:-1, 1:-1] + (x - K[0, 2]) * dx + (y - K[1, 2]) * dy), -1)
    return F.normalize(normal, dim=-1, eps=1e-6)


def _training_loss(colors: torch.Tensor, alphas: torch.Tensor, target: torch.Tensor, K: torch.Tensor,
                   depth_target: torch.Tensor | None, normal_target: torch.Tensor | None,
                   lambda_depth: float, lambda_normal: float, depth_loss: str,
                   ssim_score: torch.Tensor) -> torch.Tensor:
    loss = .8 * F.l1_loss(colors[..., :3], target) + .2 * (1 - ssim_score)
    if depth_target is None and normal_target is None:
        return loss
    depth = colors[..., 3] / alphas[..., 0].clamp_min(1e-10)
    valid = (alphas[0, ..., 0] > .05) & torch.isfinite(depth[0]) & (depth[0] > 0)
    depth = torch.where(valid, depth[0], 0)
    if depth_target is not None:
        loss = loss + lambda_depth * _depth_loss(depth, depth_target[..., 0], depth_loss, valid)
    if normal_target is not None:
        normal = _normals_from_depth(depth, K[0])
        raw_target = normal_target[1:-1, 1:-1]
        normal_valid = (valid[1:-1, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:]
                        & valid[:-2, 1:-1] & valid[2:, 1:-1]
                        & torch.isfinite(raw_target).all(-1) & (raw_target.norm(dim=-1) > .5))
        target_normal = F.normalize(torch.where(normal_valid[..., None], raw_target, 0), dim=-1, eps=1e-6)
        normal_loss = torch.where(normal_valid, 1 - (normal * target_normal).sum(-1).clamp(-1, 1), 0)
        loss = loss + lambda_normal * normal_loss.sum() / normal_valid.sum().clamp_min(1)
    return loss


def _export_ply(path: Path, splats: torch.nn.ParameterDict) -> None:
    state = {name: value.detach().cpu().numpy() for name, value in splats.items()}
    n = len(state["means"])
    fields = [(name, "f4") for name in ("x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2")]
    fields += [(f"f_rest_{i}", "f4") for i in range(45)]
    fields += [("opacity", "f4")] + [(f"scale_{i}", "f4") for i in range(3)] + [(f"rot_{i}", "f4") for i in range(4)]
    out = np.empty(n, dtype=fields)
    for i, axis in enumerate("xyz"):
        out[axis] = state["means"][:, i]
    out["nx"], out["ny"], out["nz"] = 0, 0, 0
    dc, rest = state["sh0"][:, 0], state["shN"].transpose(0, 2, 1).reshape(n, -1)
    for i in range(3): out[f"f_dc_{i}"] = dc[:, i]
    for i in range(45): out[f"f_rest_{i}"] = rest[:, i]
    out["opacity"] = state["opacities"]
    for i in range(3): out[f"scale_{i}"] = state["scales"][:, i]
    for i in range(4): out[f"rot_{i}"] = state["quats"][:, i]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(out, "vertex")], text=False).write(path)


def _atomic_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".pt") as file:
        temporary = Path(file.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _render(splats, camera, degree, geometry, absgrad=False, footprint_multiplier=1.0):
    from gsplat import rasterization
    device = splats["means"].device
    return rasterization(means=splats["means"], quats=F.normalize(splats["quats"], dim=-1),
        scales=torch.exp(splats["scales"]), opacities=torch.sigmoid(splats["opacities"]),
        colors=torch.cat((splats["sh0"], splats["shN"]), 1),
        viewmats=_camera_tensor(camera, "viewmat", device), Ks=_camera_tensor(camera, "K", device),
        width=camera["width"], height=camera["height"], sh_degree=degree,
        tile_size=16, packed=False, render_mode="RGB+D" if geometry else "RGB", absgrad=absgrad,
        footprint_multiplier=footprint_multiplier)


@torch.no_grad()
def _fastgs_metrics(splats, cameras, degree, footprint_multiplier=1.0):
    from fused_ssim import fused_ssim
    from gsplat.cuda._backend import _C
    counts = torch.zeros(len(splats["means"]), device=splats["means"].device, dtype=torch.int64)
    score = torch.zeros_like(counts, dtype=torch.float32)
    views = random.sample(cameras, min(10, len(cameras)))
    for camera in views:
        rendered, _, info = _render(splats, camera, degree, False, footprint_multiplier=footprint_multiplier)
        target = _image_tensor(camera, splats["means"].device)[None]
        residual = (rendered - target).abs().mean(-1)
        residual_min, residual_max = residual.amin(), residual.amax()
        flags = ((residual - residual_min) / (residual_max - residual_min).clamp_min(1e-12) > .1).contiguous()
        per_view = _C.rasterize_vcp_counts(info["means2d"].contiguous(), info["conics"].contiguous(),
            info["opacities"].contiguous(), flags, info["isect_offsets"], info["flatten_ids"])
        ssim = fused_ssim(rendered.permute(0, 3, 1, 2).contiguous(),
                          target.permute(0, 3, 1, 2).contiguous(), train=False)
        photometric = .8 * (rendered - target).abs().mean() + .2 * (1 - ssim)
        counts += per_view
        score += photometric * per_view
    score_min, score_max = score.amin(), score.amax()
    return counts.div(len(views), rounding_mode="floor"), (score - score_min) / (score_max - score_min).clamp_min(1e-12)


def _next_camera(cameras: list[dict], pending: list[dict]) -> dict:
    """Visit each camera once before beginning another shuffled epoch."""
    if not pending:
        pending.extend(cameras)
    return pending.pop(random.randrange(len(pending)))


def _optimizer_due(step: int, name: str) -> bool:
    if step <= 15_000:
        return name != "shN" or step % 16 == 0
    return step % (32 if step <= 20_000 else 64) == 0


def train(args: argparse.Namespace) -> Path:
    try:
        import gsplat
        from fused_ssim import fused_ssim
        from fastgs_strategy import FastGSStrategy
    except ImportError as exc:
        raise RuntimeError("gsplat is required; install the Apache-2.0 gsplat package in this environment") from exc
    if getattr(args, "target_cache_mb", 8192) < 0:
        raise ValueError("--target-cache-mb must be nonnegative")
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")
    geometry_until = getattr(args, "geometry_until", None)
    if geometry_until is None:
        geometry_until = args.iterations
    if geometry_until < 1:
        raise ValueError("--geometry-until must be positive")
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("gsplat training requires an accessible CUDA/ROCm GPU")
    device = torch.device("cuda")
    if getattr(args, "tile_size", 16) != 16:
        raise ValueError("FastGS contribution metrics require --tile-size 16")
    from gsplat.cuda._backend import _C
    if not hasattr(_C, "rasterize_vcp_counts"):
        raise RuntimeError("install the updated gsplat_gfx1200 patch for FastGS contribution metrics")
    workspace, model_dir = Path(args.source).resolve(), Path(args.model).resolve()
    cameras = load_cameras(workspace, args.resolution)
    xyz, rgb = _load_points(workspace / "sparse" / "0" / "points3D.ply", args.max_points, args.seed)
    splats = _make_splats(xyz, rgb, device)
    scene_scale = _scene_scale(cameras)
    rates = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": .025, "sh0": 2.5e-3, "shN": .00025}
    optimizers = {key: torch.optim.Adam([value], lr=rates[key], eps=1e-15, fused=True) for key, value in splats.items()}
    strategy = FastGSStrategy(verbose=args.verbose)
    strategy.check_sanity(splats, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)
    depth_root = workspace / "02_depth_estimation" / "depth" / "depth_maps"
    normal_root = workspace / "02_depth_estimation" / "depth" / "normal_maps"
    map_cache = {"values": OrderedDict(), "bytes": 0,
                 "limit": getattr(args, "target_cache_mb", 8192) * 1024**2}
    loss_function = _training_loss if getattr(args, "eager_loss", False) else torch.compile(_training_loss, fullgraph=True)
    pending_cameras = []
    for iteration in range(args.iterations):
        camera = _next_camera(cameras, pending_cameras)
        K = _camera_tensor(camera, "K", device)
        target = _image_tensor(camera, device)[None]
        step = iteration + 1
        degree = min(3, step // 1000)
        geometry = step <= geometry_until and (args.depth_supervision or args.normal_supervision)
        optimizers["means"].param_groups[0]["lr"] = scene_scale * math.exp(
            math.log(1.6e-4) * (1 - min(step / 22_000, 1.0)) + math.log(1.6e-6) * min(step / 22_000, 1.0))
        colors, alphas, info = _render(splats, camera, degree, geometry, absgrad=step < 15_000,
                                        footprint_multiplier=getattr(args, "footprint_multiplier", .5))
        strategy.step_pre_backward(splats, optimizers, strategy_state, step, info)
        depth_target = _map_tensor(depth_root, camera, 1, device, map_cache) if geometry and args.depth_supervision else None
        normal_target = _map_tensor(normal_root, camera, 3, device, map_cache) if geometry and args.normal_supervision else None
        ssim_score = fused_ssim(colors[..., :3].permute(0, 3, 1, 2).contiguous(),
                                target.permute(0, 3, 1, 2).contiguous())
        loss = loss_function(colors, alphas, target, K, depth_target, normal_target,
                             args.lambda_depth, args.lambda_normal, args.depth_loss, ssim_score)
        loss.backward()
        refine = 500 < step < 15_000 and step % 100 == 0
        final_prune = step in (18_000, 21_000, 24_000, 27_000)
        if refine or final_prune:
            importance, pruning = _fastgs_metrics(splats, cameras, degree, getattr(args, "footprint_multiplier", .5))
            info.update(importance_score=importance, pruning_score=pruning)
        strategy.step_post_backward(splats, optimizers, strategy_state, step, info, packed=False)
        for name, optimizer in optimizers.items():
            if _optimizer_due(step, name):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if args.verbose and (iteration + 1) % 100 == 0: print(f"{iteration + 1}/{args.iterations}: loss={loss.item():.5f}, splats={len(splats['means'])}", flush=True)
        if step in getattr(args, "save_iterations", ()):
            _write_model(model_dir / "snapshots" / f"iteration_{step}", splats, step, degree, args.resolution)
            print(f"Saved iteration {step} snapshot", flush=True)
    return _write_model(model_dir, splats, args.iterations, degree, args.resolution)


def _write_model(model_dir: Path, splats, iteration: int, degree: int, resolution: int) -> Path:
    output = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    _export_ply(output, splats)
    _atomic_checkpoint(model_dir / "checkpoint.pt", {"format": "gsplat-stage03-v1",
        "splats": {k: v.detach().cpu() for k, v in splats.items()},
        "iteration": iteration, "sh_degree": degree, "resolution": resolution})
    return output


def _self_check(workspace: Path) -> None:
    cameras = load_cameras(workspace)
    assert cameras and cameras[0]["K"].shape == (3, 3) and cameras[0]["viewmat"].shape == (4, 4)
    assert cameras[0]["image_path"].is_file() and cameras[0]["width"] > 0 and cameras[0]["height"] > 0
    print(f"camera loader OK: {len(cameras)} cameras, {cameras[0]['width']}x{cameras[0]['height']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--source", "--source-path", dest="source", required=True)
    parser.add_argument("-m", "--model", "--model-path", dest="model", required=True)
    parser.add_argument("--iterations", type=int, default=22_000)
    parser.add_argument("--save-iterations", type=int, nargs="*", default=[],
                        help="also export these intermediate steps under model/snapshots/iteration_<N>")
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--tile-size", type=int, choices=(16,), default=16)
    parser.add_argument("--footprint-multiplier", type=float, default=.5,
                        help="FastGS compact ellipse level (0 < value <= 1; 1 keeps full gsplat support)")
    parser.add_argument("--max-points", type=int, default=300_000)
    parser.add_argument("--depth-supervision", action="store_true")
    parser.add_argument("--normal-supervision", action="store_true")
    parser.add_argument("--geometry-until", type=int,
                        help="last step with depth/normal supervision (default: --iterations)")
    parser.add_argument("--lambda-depth", type=float, default=.15)
    parser.add_argument("--lambda-normal", type=float, default=.075)
    parser.add_argument("--depth-loss", choices=("pearson", "l1", "log_l1", "scale_shift"), default="pearson")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--target-cache-mb", type=int, default=8192, help="CPU depth/normal cache ceiling (0 disables caching)")
    parser.add_argument("--eager-loss", action="store_true", help="disable loss fusion for debugging and comparison")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.geometry_until is not None and args.geometry_until < 1:
        parser.error("--geometry-until must be positive")
    if any(step < 1 or step >= args.iterations for step in args.save_iterations):
        parser.error("--save-iterations must be positive and below --iterations")
    if args.self_check: _self_check(Path(args.source).resolve()); return 0
    print(train(args)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
