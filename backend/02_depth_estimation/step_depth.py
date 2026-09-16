#!/usr/bin/env python3
"""Step 4 -- DA3 depth priors, edge snapping, TSDF fusion, and surfel cloud initialization.

In:  ``<workspace>/images/`` + ``transforms.json`` + ``sparse/0/``.
Out: ``depth/depth_maps/*.npy``, ``depth/poses_da3.npz``,
     ``depth/points3D_depth.ply``, ``depth/edge_snapping_vis/*.jpg``,
     ``depth/preview/*.png``.

Two coordinate conventions meet here, in opposite directions, and both are
load-bearing:

* **DA3 wants OpenCV world-to-camera** -- which is exactly how COLMAP stores
  poses, so they go through untouched (see ``colmap_poses_to_da3.py``).
* **SurfelCloudInitializer & TSDF want OpenGL/ARCore camera-to-world** -- unprojects
  OpenGL camera-local rays against ``kf.transform_matrix`` (c2w_gl = inv(w2c) @ flip_yz).

CLI Usage:
    # Full end-to-end pipeline:
    python 02_depth_estimation/step_depth.py [--workspace DIR]

    # Modular substeps:
    python 02_depth_estimation/step_depth.py --substep depth
    python 02_depth_estimation/step_depth.py --substep edge_snapping
    python 02_depth_estimation/step_depth.py --substep tsdf
    python 02_depth_estimation/step_depth.py --substep surfels
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from collections.abc import Sequence as SequenceABC
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
from Utilities.pipeline_paths import bootstrap

bootstrap()

from Utilities.pipeline_step import StepContext, is_done
from Utilities.scene_io import load_scene
from colmap_diagnostics import parse_points3D_txt
from colmap_poses_to_da3 import build_da3_poses, validate_poses
from depth_priors import (
    DepthPriorEstimator,
    GlobalDepthGraphOptimizer,
    anchor_depths_to_sparse_points,
    guided_filter_depth,
)
from initialization import SurfelCloudInitializer
from tsdf_fusion import TSDFVolume

DEFAULT_WORKSPACE = _backend_dir / "current_scene"

# DA3 confidence is not a probability -- it is a relative score, and on textureless
# interior walls the whole frame sits low. 0.5 throws away most of a bedroom;
# 0.01 keeps everything but unwritten pixels and lets downstream filters do the real culling.
MIN_CONF = 0.01
CHUNK_SIZE, OVERLAP = 6, 3
PROCESS_RES = 756  # High-quality DA3 test-time resolution (756x420, 2.25x token density)

# Surfel extraction configuration
VOXEL_DOWNSAMPLE_M = 0.015  # 1.5 cm voxels
TARGET_SURFELS = MAX_SURFELS = 3_000_000

# Boundary-floater filters (tuned hole-safe thresholds)
MAX_DEPTH_GRADIENT = 0.15
MAX_GRAZING_ANGLE_DEG = 82.0

_FLIP_YZ = np.diag([1.0, -1.0, -1.0, 1.0])


def _colorize_depth(dmap: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Depth in meters -> RGB, fixed (vmin, vmax) so color is comparable across frames."""
    norm = np.clip((dmap - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    gray_u8 = (norm * 255.0).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(gray_u8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


class _LazyImages(SequenceABC):
    """Images decoded on access with a small LRU cache for sliding window chunk reuse."""

    def __init__(self, paths: list[Path], cache_size: int = 32):
        self._paths = paths
        self._cache_size = cache_size
        self._cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        if idx in self._cache:
            return self._cache[idx]
        with Image.open(self._paths[idx]) as im:
            arr = np.asarray(im.convert("RGB"))
        if len(self._cache) >= self._cache_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[idx] = arr
        return arr


def _get_vram_info() -> dict[str, float]:
    import torch
    if torch.cuda.is_available():
        alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        res_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        max_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        return {
            "alloc_gb": round(alloc_gb, 3),
            "reserved_gb": round(res_gb, 3),
            "peak_gb": round(max_alloc_gb, 3),
        }
    return {}


def run_depth_estimation_substep(
    workspace: Path,
    ctx: StepContext,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
    process_res: int = PROCESS_RES,
    enable_guided_filter: bool = True,
    enable_sparse_anchor: bool = True,
    enable_global_depth_graph: bool = True,
) -> tuple[list[np.ndarray], list[str], np.ndarray, np.ndarray]:
    """Substep 1/2: Multi-View DA3 sliding window depth estimation & guided filtering."""
    import torch

    scene = load_scene(workspace)
    names = scene.names
    sparse_dir = workspace / "sparse" / "0"
    depth_dir = workspace / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = depth_dir / "depth_maps"
    maps_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx.metric("device", device)
    ctx.metric("process_res", process_res)

    # --- poses: COLMAP -> DA3, unconverted, but checked ---------------------
    with ctx.timer("build_poses"):
        w2c, k_mats, pose_names = build_da3_poses(sparse_dir, names)
        by_name = {Path(f["file_path"]).name: f for f in scene.frames}
        centres = np.array([np.array(by_name[n]["transform_matrix"])[:3, 3] for n in pose_names])
        stats = validate_poses(w2c, centres)
        np.savez(depth_dir / "poses_da3.npz", w2c=w2c, K=k_mats, names=np.array(pose_names))
    ctx.metric("pose_validation", {k: round(v, 6) for k, v in stats.items()})
    ctx.note(f"Poses: {len(pose_names)} world-to-camera, orthonormality err "
             f"{stats['max_ortho_err']:.2e}, centres within "
             f"{stats['centre_offset_median_m'] * 100:.1f} cm (median) of ARCore")

    images = _LazyImages([workspace / "images" / n for n in pose_names])
    vram_substeps: dict[str, dict[str, float]] = {}

    ctx.note(f"Running DA3 on {len(images)} frames (device={device}, "
             f"chunk={chunk_size}, overlap={overlap}, process_res={process_res})")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    estimator = DepthPriorEstimator(device=device, process_res=process_res)
    with ctx.timer("da3_inference"):
        depth_maps, conf_maps, uncertainties = estimator.estimate_depth_sliding_window(
            images=images,
            extrinsics=w2c,
            intrinsics=k_mats,
            chunk_size=chunk_size,
            overlap=overlap,
            cache_dir=depth_dir / "da3_cache",
            process_res=process_res,
        )
    vram_substeps["da3_inference"] = _get_vram_info()

    # --- depth post-processing: RGB-guided filtering -------------------------
    if enable_guided_filter:
        with ctx.timer("guided_filter"):
            depth_maps = [guided_filter_depth(dmap, images[i]) for i, dmap in enumerate(depth_maps)]
        vram_substeps["guided_filter"] = _get_vram_info()
        ctx.note("Applied RGB-guided depth filtering for boundary edge snapping and plane smoothing")

    # --- surfel keyframe setup for anchor ------------------------------------
    c2w_gl = np.stack([np.linalg.inv(p.astype(np.float64)) @ _FLIP_YZ for p in w2c])
    kf_by_name = {Path(k.file_path).name: k for k in scene.keyframes()}
    keyframes = [dataclasses.replace(kf_by_name[n], transform_matrix=c2w_gl[i])
                 for i, n in enumerate(pose_names)]

    points = parse_points3D_txt(sparse_dir / "points3D.txt")
    sparse_xyz = np.array([p["xyz"] for p in points.values()]) if points else None

    # --- depth post-processing: sparse landmark scale anchoring --------------
    if enable_sparse_anchor and sparse_xyz is not None and len(sparse_xyz) >= 8:
        with ctx.timer("sparse_anchor"):
            depth_maps, anchor_stats = anchor_depths_to_sparse_points(
                depth_maps=depth_maps,
                keyframes=keyframes,
                intrinsics=scene.intrinsics,
                sparse_points_3d=sparse_xyz,
            )
            ctx.metric("sparse_anchor_stats", anchor_stats)
            ctx.note(f"Anchored depth maps against COLMAP sparse points: "
                     f"{anchor_stats['anchored_frames']}/{anchor_stats['total_frames']} frames "
                     f"(median scale={anchor_stats['median_scale']}, shift={anchor_stats['median_shift_m']}m)")
        vram_substeps["sparse_anchor"] = _get_vram_info()

    # --- depth post-processing: global multi-view depth graph optimization ----
    if enable_global_depth_graph and len(keyframes) > 2:
        with ctx.timer("global_depth_graph"):
            graph_opt = GlobalDepthGraphOptimizer(
                min_depth=0.2,
                max_depth=15.0,
                max_loop_dist_m=1.8,
                min_loop_separation=12,
                min_cos_angle=0.4,
                max_loop_candidates_per_frame=4,
                anchor_weight=100.0,
                reg_scale_weight=1.5,
                reg_shift_weight=1.5,
            )
            graph_res = graph_opt.optimize(
                keyframes=keyframes,
                raw_depth_maps=depth_maps,
                intrinsics=scene.intrinsics,
            )
            depth_maps = graph_res.aligned_depth_maps
            ctx.metric("depth_graph_rmse_before_m", graph_res.rmse_before_m)
            ctx.metric("depth_graph_rmse_after_m", graph_res.rmse_after_m)
            ctx.metric("depth_graph_constraints", graph_res.num_constraints)
            ctx.note(f"Global Depth Graph: {graph_res.num_temporal_edges} temporal, "
                     f"{graph_res.num_loop_edges} loop closure edges ({graph_res.num_constraints:,} constraints). "
                     f"RMSE {graph_res.rmse_before_m*100:.1f}cm -> {graph_res.rmse_after_m*100:.1f}cm")
        vram_substeps["global_depth_graph"] = _get_vram_info()

    with ctx.timer("write_depth_maps"):
        for name, dmap in zip(pose_names, depth_maps):
            np.save(maps_dir / f"{Path(name).stem}.npy", dmap.astype(np.float32))
    vram_substeps["write_depth_maps"] = _get_vram_info()

    finite = np.concatenate([d[np.isfinite(d)].ravel()[::97] for d in depth_maps])
    depth_p05, depth_p95 = float(np.percentile(finite, 5)), float(np.percentile(finite, 95))
    ctx.metric("depth_m", {
        "median": round(float(np.median(finite)), 3),
        "p05": round(depth_p05, 3),
        "p95": round(depth_p95, 3),
    })

    # RGB | depth side-by-side composite images
    depth_images_dir = depth_dir / "depth_images"
    depth_images_dir.mkdir(exist_ok=True)
    vis_dir = depth_dir / "depth_vis"
    vis_dir.mkdir(exist_ok=True)

    with ctx.timer("write_depth_images"):
        for i, (name, dmap) in enumerate(zip(pose_names, depth_maps)):
            orig_rgb = images[i]
            depth_rgb = _colorize_depth(dmap, depth_p05, depth_p95)
            if depth_rgb.shape[:2] != orig_rgb.shape[:2]:
                depth_rgb = cv2.resize(depth_rgb, (orig_rgb.shape[1], orig_rgb.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
            composite = np.hstack([orig_rgb, depth_rgb])
            comp_img = Image.fromarray(composite)
            stem = Path(name).stem
            comp_img.save(depth_images_dir / f"{stem}.jpg", quality=90)
            if not (vis_dir / f"{stem}.jpg").exists():
                comp_img.save(vis_dir / f"{stem}.jpg", quality=90)
    vram_substeps["write_depth_images"] = _get_vram_info()

    # Diagnostic markdown summary
    summary_md = [
        "# Multi-View Depth Estimation Summary",
        "",
        f"**Workspace:** `{workspace}`  ",
        f"**Model:** `DA3-Base` (Test Resolution: `{process_res}px`, Window: `N={chunk_size}, K={overlap}`)  ",
        f"**Device:** `{device}`  ",
        f"**Total Frames Processed:** `{len(pose_names)}`  ",
        "",
        "## Depth Distribution",
        "",
        f"- **Median Depth:** `{np.median(finite):.2f} m`",
        f"- **5th Percentile (p05):** `{depth_p05:.2f} m`",
        f"- **95th Percentile (p95):** `{depth_p95:.2f} m`",
        "",
        "## Diagnostic Artifacts",
        "",
        f"- **Side-by-Side RGB | Depth Images:** [`{depth_images_dir}`](file://{depth_images_dir})",
        f"- **Dense Float32 Metric Depth Maps:** [`{maps_dir}`](file://{maps_dir})",
        f"- **Camera Extrinsics & Intrinsics:** [`{depth_dir / 'poses_da3.npz'}`](file://{depth_dir / 'poses_da3.npz'})",
        "",
        "## Execution Time & VRAM Profile",
        "",
        "| Substep | Time (s) | VRAM Alloc (GB) | VRAM Reserved (GB) | VRAM Peak (GB) |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for step_name, t_val in ctx.timings.items():
        vr = vram_substeps.get(step_name, {})
        alloc = f"{vr.get('alloc_gb', 0.0):.2f}" if vr else "-"
        res = f"{vr.get('reserved_gb', 0.0):.2f}" if vr else "-"
        peak = f"{vr.get('peak_gb', 0.0):.2f}" if vr else "-"
        summary_md.append(f"| `{step_name}` | **{t_val:.1f}s** | {alloc} | {res} | {peak} |")

    summary_md_path = depth_dir / "depth_inference_summary.md"
    summary_md_path.write_text("\n".join(summary_md), encoding="utf-8")
    ctx.note(f"Human-inspectable depth estimation summary written to {summary_md_path}")

    return depth_maps, pose_names, w2c, k_mats


def run_tsdf_substep(
    workspace: Path,
    ctx: StepContext,
    voxel_size: float = VOXEL_DOWNSAMPLE_M,
    min_weight: float = 2.0,
    max_surfels: int = 500_000,
) -> None:
    """Substep 4d: Volumetric TSDF Fusion & Multi-Scale Zero-Crossing Surfel Extraction."""
    import torch

    scene = load_scene(workspace)
    sparse_dir = workspace / "sparse" / "0"
    depth_dir = workspace / "depth"
    maps_dir = depth_dir / "depth_maps"
    poses_path = depth_dir / "poses_da3.npz"

    if not poses_path.exists():
        raise FileNotFoundError(f"Poses file {poses_path} missing. Run --substep depth first.")

    poses_data = np.load(poses_path, allow_pickle=True)
    pose_names = [str(n) for n in poses_data["names"]]
    w2c = poses_data["w2c"]

    c2w_gl = np.stack([np.linalg.inv(p.astype(np.float64)) @ _FLIP_YZ for p in w2c])
    kf_by_name = {Path(k.file_path).name: k for k in scene.keyframes()}
    keyframes = [dataclasses.replace(kf_by_name[n], transform_matrix=c2w_gl[i])
                 for i, n in enumerate(pose_names)]

    images = _LazyImages([workspace / "images" / n for n in pose_names])
    depth_maps = [np.load(maps_dir / f"{Path(n).stem}.npy") for n in pose_names]

    points = parse_points3D_txt(sparse_dir / "points3D.txt")
    sparse_xyz = np.array([p["xyz"] for p in points.values()]) if points else None

    # Compute bounding box
    if sparse_xyz is not None and len(sparse_xyz) >= 10:
        p01 = np.percentile(sparse_xyz, 0.5, axis=0)
        p99 = np.percentile(sparse_xyz, 99.5, axis=0)
        margin = 0.6
        bbox_min = p01 - margin
        bbox_max = p99 + margin
    else:
        cam_pos = np.array([kf.transform_matrix[:3, 3] for kf in keyframes])
        bbox_min = np.min(cam_pos, axis=0) - 2.5
        bbox_max = np.max(cam_pos, axis=0) + 2.5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx.note(f"Initializing TSDF Volume (voxel_size={voxel_size*100:.1f}cm, device={device}, "
             f"bbox=[{bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f}] -> "
             f"[{bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f}])...")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    volume = TSDFVolume(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        voxel_size=voxel_size,
        device=device,
    )
    vram_init = _get_vram_info()
    ctx.note(f"Allocated TSDF grid ({volume.nx} × {volume.ny} × {volume.nz} voxels, "
             f"{volume.nx * volume.ny * volume.nz / 1e6:.2f}M cells). VRAM: {vram_init.get('alloc_gb', 0.0):.2f} GB")

    with ctx.timer("tsdf_integration"):
        for i, (name, dmap) in enumerate(zip(pose_names, depth_maps)):
            rgb_img = images[i]
            volume.integrate_frame(
                depth_map=dmap,
                rgb_img=rgb_img,
                c2w=keyframes[i].transform_matrix,
                intrinsics=scene.intrinsics,
            )
            if (i + 1) % 100 == 0 or (i + 1) == len(pose_names):
                ctx.note(f"Integrated {i + 1}/{len(pose_names)} keyframes into TSDF volume...")

    vram_integrated = _get_vram_info()

    with ctx.timer("tsdf_multiscale_extraction"):
        cloud = volume.extract_multiscale_surfels(
            min_weight=min_weight,
            max_surfels=max_surfels,
        )

    vram_extracted = _get_vram_info()
    elapsed_total = time.time() - t0

    # Save output surfels
    ply_path = depth_dir / "points3D_depth.ply"
    cloud.to_ply(ply_path)
    # Also save a dedicated TSDF copy
    cloud.to_ply(depth_dir / "points3D_tsdf.ply")

    # Geometry statistics
    pos = cloud.positions
    min_b = np.min(pos, axis=0)
    max_b = np.max(pos, axis=0)
    dims = max_b - min_b
    mean_scale = float(np.mean(cloud.scales_2d))

    # Diagnostic preview renders
    preview_dir = depth_dir / "preview"
    preview_dir.mkdir(exist_ok=True)
    try:
        from preview_cloud import render as render_preview, intrinsics as colmap_intrinsics
        from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary
        extr = read_extrinsics_binary(sparse_dir / "images.bin")
        intr = read_intrinsics_binary(sparse_dir / "cameras.bin")
        im_list = sorted(extr.values(), key=lambda im: im.name)
        step = max(1, len(im_list) // 8)
        rgb_u8 = np.clip(cloud.colors_rgb * 255.0, 0, 255).astype(np.uint8)
        for im in im_list[::step][:8]:
            cam = intr[im.camera_id]
            fx, fy, cx, cy = colmap_intrinsics(cam)
            prev_img = render_preview(pos.astype(np.float64), rgb_u8,
                                      qvec2rotmat(im.qvec), im.tvec,
                                      fx, fy, cx, cy, cam.width, cam.height)
            prev_img.save(preview_dir / f"{Path(im.name).stem}.png")
        ctx.note(f"Rendered diagnostic camera previews to {preview_dir}")
    except Exception as e:
        ctx.note(f"Preview rendering notice: {e}")

    # Write Markdown Summary Report
    tsdf_summary_md = [
        "# Volumetric TSDF Fusion & Multi-Scale Surfel Extraction (Step 4d)",
        "",
        f"**Workspace:** `{workspace}`  ",
        f"**Output PLY:** [`{ply_path}`](file://{ply_path})  ",
        f"**Extracted Surfels:** **`{len(cloud):,}`** (Single-manifold shell, 0 multi-layer slab thickness)  ",
        f"**Bounding Box:** `{dims[0]:.2f}m` (W) × `{dims[1]:.2f}m` (L) × `{dims[2]:.2f}m` (H)  ",
        rf"**Mean Surfel Scale ($\sigma$):** `{mean_scale * 100:.2f} cm`  ",
        f"**Total Execution Time:** `{elapsed_total:.1f}s`  ",
        "",
        "## TSDF Grid & Multi-Scale Parameters",
        "",
        f"- **Base Voxel Resolution:** `{voxel_size * 100:.1f} cm` ({volume.nx} × {volume.ny} × {volume.nz} voxels)",
        f"- **Minimum Integration Weight Floor:** `{min_weight}`",
        "- **Hierarchical Multi-Scale Decimation:**",
        r"  - **Tier 0 (Corners / Edges):** Full 1.5 cm density, $\sigma = 1.1\text{cm}$",
        r"  - **Tier 1 (Curved Geometry):** 3.0 cm stride, $\sigma = 2.2\text{cm}$",
        r"  - **Tier 2 (Planar Walls / Floor):** 6.0 cm stride, $\sigma = 4.5\text{cm}$",
        "",
        "## VRAM & Runtime Breakdown",
        "",
        f"- **Grid Allocation VRAM:** `{vram_init.get('alloc_gb', 0.0):.2f} GB` (Reserved: `{vram_init.get('reserved_gb', 0.0):.2f} GB`)",
        f"- **Peak Integration VRAM:** `{vram_integrated.get('peak_gb', 0.0):.2f} GB`",
        f"- **Post-Extraction VRAM:** `{vram_extracted.get('alloc_gb', 0.0):.2f} GB`",
        "",
        "## Diagnostic Artifacts",
        "",
        f"- **Camera Projections Preview:** [`{preview_dir}`](file://{preview_dir})",
        f"- **Extracted Surfel Cloud:** [`{ply_path}`](file://{ply_path})",
    ]
    summary_file = depth_dir / "tsdf_summary.md"
    summary_file.write_text("\n".join(tsdf_summary_md), encoding="utf-8")
    ctx.note(f"TSDF fusion complete ({len(cloud):,} surfels). Report written to {summary_file}")


def run_surfels_substep(
    workspace: Path,
    ctx: StepContext,
    voxel_downsample_m: float = VOXEL_DOWNSAMPLE_M,
    enable_sor: bool = True,
    sor_k: int = 20,
    sor_std_mul: float = 2.8,
    enable_normal_consensus: bool = True,
    enable_saturation_mask: bool = True,
    enable_freespace_filter: bool = False,
    max_freespace_violations: int = 0,
    enable_tube_collapse: bool = True,
    tube_radius_m: float = 0.03,
    tube_length_m: float = 0.25,
    tube_min_normal_cos: float = 0.75,
    enable_multiscale_pyramid: bool = False,
    enable_hybrid_sampling: bool = True,
    hybrid_energy_threshold: float = 0.08,
    hybrid_coarse_stride: int = 3,
) -> None:
    """Substep 4b: Standard unprojection surfel initializer with geometric filtering."""
    scene = load_scene(workspace)
    sparse_dir = workspace / "sparse" / "0"
    depth_dir = workspace / "depth"
    maps_dir = depth_dir / "depth_maps"
    poses_path = depth_dir / "poses_da3.npz"

    if not poses_path.exists():
        raise FileNotFoundError(f"Poses file {poses_path} missing. Run --substep depth first.")

    poses_data = np.load(poses_path, allow_pickle=True)
    pose_names = [str(n) for n in poses_data["names"]]
    w2c = poses_data["w2c"]

    c2w_gl = np.stack([np.linalg.inv(p.astype(np.float64)) @ _FLIP_YZ for p in w2c])
    kf_by_name = {Path(k.file_path).name: k for k in scene.keyframes()}
    keyframes = [dataclasses.replace(kf_by_name[n], transform_matrix=c2w_gl[i])
                 for i, n in enumerate(pose_names)]

    depth_maps = [np.load(maps_dir / f"{Path(n).stem}.npy", mmap_mode="r") for n in pose_names]
    points = parse_points3D_txt(sparse_dir / "points3D.txt")
    sparse_xyz = np.array([p["xyz"] for p in points.values()]) if points else None

    with ctx.timer("surfel_init"):
        initializer = SurfelCloudInitializer(
            voxel_downsample_m=voxel_downsample_m,
            target_surfels=TARGET_SURFELS,
            max_surfels=MAX_SURFELS,
            max_depth_gradient=MAX_DEPTH_GRADIENT,
            max_grazing_angle_deg=MAX_GRAZING_ANGLE_DEG,
            enable_sor=enable_sor,
            sor_k=sor_k,
            sor_std_mul=sor_std_mul,
            enable_saturation_mask=enable_saturation_mask,
            enable_freespace_filter=enable_freespace_filter,
            max_freespace_violations=max_freespace_violations,
            enable_normal_consensus=enable_normal_consensus,
            enable_global_carving=False,
            enable_tube_collapse=enable_tube_collapse,
            tube_radius_m=tube_radius_m,
            tube_length_m=tube_length_m,
            tube_min_normal_cos=tube_min_normal_cos,
            enable_multiscale_pyramid=enable_multiscale_pyramid,
            enable_hybrid_sampling=enable_hybrid_sampling,
            hybrid_energy_threshold=hybrid_energy_threshold,
            hybrid_coarse_stride=hybrid_coarse_stride,
        )
        cloud = initializer.initialize_from_keyframes(
            keyframes=keyframes,
            depth_maps=depth_maps,
            intrinsics=scene.intrinsics,
            sparse_points_3d=None,
            conf_maps=None,
            min_conf=MIN_CONF,
        )
        ctx.note(f"Initialized continuous refined surfel cloud: {len(cloud):,} surfels")

    # Export main points3D_depth.ply
    depth_ply_path = depth_dir / "points3D_depth.ply"
    cloud.to_ply(depth_ply_path)

    # If tube collapse was enabled, also save dedicated copy for inspection
    if enable_tube_collapse:
        tube_ply_path = depth_dir / "points3D_tube_collapsed.ply"
        cloud.to_ply(tube_ply_path)
        ctx.note(f"Saved tube-collapsed surfel cloud to {tube_ply_path}")

    # Compute bounding box and geometry stats
    pos = cloud.positions
    min_b = np.min(pos, axis=0)
    max_b = np.max(pos, axis=0)
    dims = max_b - min_b
    mean_scale = float(np.mean(cloud.scales_2d))

    preview_dir = depth_dir / "preview"
    preview_dir.mkdir(exist_ok=True)
    try:
        from preview_cloud import render as render_preview, intrinsics as colmap_intrinsics
        from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary
        extr = read_extrinsics_binary(sparse_dir / "images.bin")
        intr = read_intrinsics_binary(sparse_dir / "cameras.bin")
        im_list = sorted(extr.values(), key=lambda im: im.name)
        step = max(1, len(im_list) // 8)
        rgb_u8 = np.clip(cloud.colors_rgb * 255.0, 0, 255).astype(np.uint8)
        for im in im_list[::step][:8]:
            cam = intr[im.camera_id]
            fx, fy, cx, cy = colmap_intrinsics(cam)
            prev_img = render_preview(pos.astype(np.float64), rgb_u8,
                                      qvec2rotmat(im.qvec), im.tvec,
                                      fx, fy, cx, cy, cam.width, cam.height)
            prev_img.save(preview_dir / f"{Path(im.name).stem}.png")
        ctx.note(f"Rendered diagnostic camera previews to {preview_dir}")
    except Exception as e:
        ctx.note(f"Camera preview rendering notice: {e}")

    surfel_summary_md = [
        "# Surfel Cloud Initialization Summary (Step 4b)",
        "",
        f"**Workspace:** `{workspace}`  ",
        f"**Output PLY:** [`{depth_ply_path}`](file://{depth_ply_path})  ",
        f"**Total Surfels:** **`{len(cloud):,}`**  ",
        f"**Bounding Box:** `{dims[0]:.2f}m` (W) × `{dims[1]:.2f}m` (L) × `{dims[2]:.2f}m` (H)  ",
        rf"**Mean Surfel Scale ($\sigma$):** `{mean_scale * 100:.2f} cm`  ",
        "",
        "## Geometric Filtering & Processing Flags",
        "",
        f"- **Hybrid Dual-Guided Adaptive Sampling:** `{'Enabled' if enable_hybrid_sampling else 'Disabled'}` (threshold={hybrid_energy_threshold}, coarse_stride={hybrid_coarse_stride})",
        f"- **Normal Tube Collapse:** `{'Enabled' if enable_tube_collapse else 'Disabled'}` (radius={tube_radius_m*100:.1f}cm, length={tube_length_m*100:.1f}cm)",
        f"- **Multi-Scale Surfel Pyramids:** `{'Enabled' if enable_multiscale_pyramid else 'Disabled'}`",
        f"- **Voxel Grid Size:** `{voxel_downsample_m * 100:.1f} cm` (applied: `{getattr(cloud, 'applied_voxel_size_m', voxel_downsample_m) * 100:.1f} cm`)",
        f"- **Dynamic Saturation / Bloom Masking:** `{'Enabled' if enable_saturation_mask else 'Disabled'}`",
        f"- **Cross-View Free-Space Carving:** `{'Enabled' if enable_freespace_filter else 'Disabled'}`",
        f"- **Multi-View Normal Consensus Regularization:** `{'Enabled' if enable_normal_consensus else 'Disabled'}`",
        f"- **Depth Discontinuity Gradient Filter:** `max_grad = {MAX_DEPTH_GRADIENT}`",
        f"- **Grazing Angle Filter:** `max_angle = {MAX_GRAZING_ANGLE_DEG}°`",
        "",
        "## Diagnostic Artifacts",
        "",
        f"- **Camera Projections Preview:** [`{preview_dir}`](file://{preview_dir})",
        f"- **Surfel Binary PLY:** [`{depth_ply_path}`](file://{depth_ply_path})",
    ]
    summary_file = depth_dir / "surfel_init_summary.md"
    summary_file.write_text("\n".join(surfel_summary_md), encoding="utf-8")
    ctx.note(f"Human-inspectable surfel summary written to {summary_file}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--substep", choices=["all", "depth", "surfels", "tsdf"],
                        default="all", help="Substep to run (default 'all')")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=OVERLAP)
    parser.add_argument("--process-res", type=int, default=PROCESS_RES,
                        help=f"DA3 processing resolution (default {PROCESS_RES}; e.g. 504, 756, 1008)")
    parser.add_argument("--enable-tsdf", action="store_true", default=False,
                        help="Use experimental Volumetric TSDF Fusion instead of SurfelCloudInitializer")
    parser.add_argument("--no-tsdf", action="store_false", dest="enable_tsdf",
                        help="Use classic backprojection unprojection instead of TSDF")
    parser.add_argument("--no-guided-filter", action="store_true",
                        help="Disable RGB-guided edge-preserving depth filter")
    parser.add_argument("--no-sparse-anchor", action="store_true",
                        help="Disable sparse COLMAP landmark scale/shift anchoring")
    parser.add_argument("--no-normal-consensus", action="store_true",
                        help="Disable cross-view surface normal consensus regularization")
    parser.add_argument("--no-saturation-mask", action="store_true",
                        help="Disable dynamic saturation/bloom masking before unprojection")
    parser.add_argument("--no-tube-collapse", action="store_false", dest="enable_tube_collapse",
                        help="Disable normal tube collapse projection")
    parser.add_argument("--enable-tube-collapse", action="store_true", dest="enable_tube_collapse", default=True,
                        help="Enable normal tube collapse projection (default: True)")
    parser.add_argument("--tube-radius", type=float, default=0.03,
                        help="Normal tube cylinder radius in meters (default 0.03 = 3cm)")
    parser.add_argument("--tube-length", type=float, default=0.25,
                        help="Normal tube cylinder search length in meters (default 0.25 = 25cm)")
    parser.add_argument("--enable-multiscale-pyramid", action="store_true", default=False,
                        help="Enable multi-scale surfel decimation for planar regions")
    parser.add_argument("--no-multiscale-pyramid", action="store_false", dest="enable_multiscale_pyramid",
                        help="Disable multi-scale surfel decimation")
    parser.add_argument("--enable-hybrid-sampling", action="store_true", dest="enable_hybrid_sampling", default=True,
                        help="Enable 2D hybrid photometric and geometric adaptive sampling (default: True)")
    parser.add_argument("--no-hybrid-sampling", action="store_false", dest="enable_hybrid_sampling",
                        help="Disable hybrid adaptive sampling and use uniform stride unprojection")
    parser.add_argument("--hybrid-energy-threshold", type=float, default=0.08,
                        help="Energy threshold for fine-stride edge/texture sampling (default 0.08)")
    parser.add_argument("--hybrid-coarse-stride", type=int, default=3,
                        help="Pixel stride for uniform background flat regions (default 3 = ~4.5cm)")
    parser.add_argument("--no-freespace-filter", action="store_true",
                        help="Disable cross-view epipolar/reprojection free-space filter")
    parser.add_argument("--enable-freespace-filter", action="store_true", default=False,
                        help="Enable cross-view epipolar/reprojection free-space filter")
    parser.add_argument("--max-freespace-violations", type=int, default=0,
                        help="Max allowed free-space violations before culling (default 0)")
    # Legacy flags for compatibility
    parser.add_argument("--only-depth", action="store_true",
                        help="Alias for --substep depth")
    parser.add_argument("--only-surfels", action="store_true",
                        help="Alias for --substep surfels")
    parser.add_argument("--force", action="store_true", help="Re-run even if already complete")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    substep = args.substep
    if args.only_depth:
        substep = "depth"
    elif args.only_surfels:
        substep = "surfels"

    outputs = [workspace / "depth" / "poses_da3.npz"]
    if substep in ("all", "surfels", "tsdf"):
        outputs.append(workspace / "depth" / "points3D_depth.ply")

    if not args.force and substep == "all" and is_done(workspace, "depth", outputs):
        print("[depth] already done, skipping (use --force to re-run)")
        return 0

    with StepContext("depth", workspace) as ctx:
        if substep in ("all", "depth"):
            run_depth_estimation_substep(
                workspace=workspace,
                ctx=ctx,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                process_res=args.process_res,
                enable_guided_filter=not args.no_guided_filter,
                enable_sparse_anchor=not args.no_sparse_anchor,
            )

        if substep == "tsdf" or (substep == "all" and args.enable_tsdf):
            run_tsdf_substep(
                workspace=workspace,
                ctx=ctx,
                voxel_size=VOXEL_DOWNSAMPLE_M,
            )
        elif substep == "surfels" or (substep == "all" and not args.enable_tsdf):
            run_surfels_substep(
                workspace=workspace,
                ctx=ctx,
                voxel_downsample_m=VOXEL_DOWNSAMPLE_M,
                enable_normal_consensus=not args.no_normal_consensus,
                enable_saturation_mask=not args.no_saturation_mask,
                enable_freespace_filter=args.enable_freespace_filter and not args.no_freespace_filter,
                max_freespace_violations=args.max_freespace_violations,
                enable_tube_collapse=args.enable_tube_collapse,
                tube_radius_m=args.tube_radius,
                tube_length_m=args.tube_length,
                enable_multiscale_pyramid=args.enable_multiscale_pyramid,
                enable_hybrid_sampling=args.enable_hybrid_sampling,
                hybrid_energy_threshold=args.hybrid_energy_threshold,
                hybrid_coarse_stride=args.hybrid_coarse_stride,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


