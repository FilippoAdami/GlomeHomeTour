#!/usr/bin/env python3
"""GlomeHomeTour: Standalone 2DGS Material & Density Training Pipeline.

Integrates real reconstructed scene assets:
1. Surfel Point Cloud (.ply) with normal-aware voxel downsampling.
2. Calibrated Keyframes & Transforms (transforms.json + images/).
3. Surface Normal & Metric Depth Priors (depth_maps/).
4. Multi-Scale Progressive Resolution Scheduling (e.g. 270x480 -> 540x960 -> 1080x1920).
5. Floating-point Checkpoint & Compressed WebGL Bundle (walkthrough_2dgs.zip <= 25 MB).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure writable cache directories on ROCm
if "MIOPEN_USER_DB_PATH" not in os.environ:
    _miopen_dir = Path.home() / "Desktop" / "GlomeHomeTour" / ".cache" / "miopen"
    _miopen_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MIOPEN_USER_DB_PATH"] = str(_miopen_dir)

import numpy as np
import torch

# Ensure backend is on sys.path
_backend_dir = Path(__file__).resolve().parents[2]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from reconstruction.initialization import SurfelCloud
from reconstruction.training.dataset import GSInputDataset, GSSceneData
from reconstruction.training.density_control import DensityControlConfig
from reconstruction.training.model import Material2DGSModel
from reconstruction.training.trainer import Material2DGSTrainer, TrainerConfig


def resolve_scene_paths(
    scene_dir: Optional[Union[str, Path]] = None,
    ply_path: Optional[Union[str, Path]] = None,
    transforms_path: Optional[Union[str, Path]] = None,
    depth_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path, Optional[Path]]:
    """Resolve paths to the PLY file, GS_input directory, and depth maps directory."""
    resolved_ply: Optional[Path] = None
    resolved_gs_input: Optional[Path] = None
    resolved_depth_dir: Optional[Path] = None

    if scene_dir is not None:
        s_path = Path(scene_dir).resolve()
        # Case 1: scene_dir is directly the GS_input folder
        if s_path.name == "GS_input" or (s_path / "transforms.json").is_file():
            resolved_gs_input = s_path
        # Case 2: scene_dir is the parent folder containing GS_input
        elif (s_path / "GS_input" / "transforms.json").is_file():
            resolved_gs_input = s_path / "GS_input"
        else:
            resolved_gs_input = s_path

        # Locate PLY in GS_input or scene_dir
        if ply_path is None:
            cand_ply = list(resolved_gs_input.glob("*.ply"))
            if not cand_ply and s_path != resolved_gs_input:
                cand_ply = list(s_path.glob("*.ply"))
            if cand_ply:
                resolved_ply = cand_ply[0]

        # Locate depth_maps directory
        if depth_dir is None:
            if (resolved_gs_input.parent / "depth_maps").is_dir():
                resolved_depth_dir = resolved_gs_input.parent / "depth_maps"
            elif (resolved_gs_input / "depth_maps").is_dir():
                resolved_depth_dir = resolved_gs_input / "depth_maps"
            elif (s_path / "depth_maps").is_dir():
                resolved_depth_dir = s_path / "depth_maps"

    if ply_path is not None:
        resolved_ply = Path(ply_path).resolve()

    if transforms_path is not None:
        t_path = Path(transforms_path).resolve()
        resolved_gs_input = t_path.parent if t_path.is_file() else t_path

    if depth_dir is not None:
        resolved_depth_dir = Path(depth_dir).resolve()

    if resolved_ply is None or not resolved_ply.is_file():
        raise FileNotFoundError(f"Surfel PLY file not found: {resolved_ply}")
    if resolved_gs_input is None or not resolved_gs_input.is_dir():
        raise FileNotFoundError(f"GS_input directory not found: {resolved_gs_input}")

    return resolved_ply, resolved_gs_input, resolved_depth_dir


def train_scene_from_ply_and_frames(
    scene_dir: Optional[Union[str, Path]] = None,
    ply_path: Optional[Union[str, Path]] = None,
    transforms_path: Optional[Union[str, Path]] = None,
    depth_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    iterations: int = 3000,
    voxel_size_m: float = 0.02,
    max_surfels: int = 600_000,
    primitive_budget: Optional[int] = None,
    multi_scale: bool = True,
    device: Optional[str] = None,
    max_frames: Optional[int] = None,
    checkpoint_interval: int = 500,
    log_interval: int = 50,
    resume_checkpoint: Optional[Union[str, Path]] = None,
    start_iteration: int = 1,
) -> Dict[str, Any]:
    """Execute complete end-to-end 2DGS training from PLY surfels and keyframe package."""
    # 1. Resolve Paths
    actual_ply, actual_gs_input, actual_depth_dir = resolve_scene_paths(
        scene_dir=scene_dir,
        ply_path=ply_path,
        transforms_path=transforms_path,
        depth_dir=depth_dir,
    )

    if output_dir is None:
        out_path = actual_gs_input.parent / "2dgs_output"
    else:
        out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # 2. Setup Device
    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)
    print(f"[GlomeHomeTour] Initializing 2DGS Reconstruction Engine on: {device_obj}")

    # 3. Load & Downsample Surfel Cloud
    print(f"[1/4] Ingesting Surfel Cloud: {actual_ply}")
    t0 = time.time()
    surfel_cloud = SurfelCloud.from_ply(
        actual_ply,
        voxel_downsample_m=voxel_size_m,
        max_surfels=max_surfels,
    )
    load_time = time.time() - t0
    print(
        f"      Loaded {len(surfel_cloud):,} surfels in {load_time:.2f}s "
        f"(voxel grid: {voxel_size_m*100:.1f}cm, max cap: {max_surfels:,})"
    )

    # 4. Instantiate Material 2DGS Model
    print("[2/4] Instantiating Material2DGSModel with PBR attributes...")
    model = Material2DGSModel.from_surfel_cloud(surfel_cloud, device=device_obj)
    print(f"      Initialized {model.num_gaussians:,} 2D Gaussians.")

    # 5. Initialize Dataset Loader
    print(f"[3/4] Parsing Multi-View Keyframes Manifest: {actual_gs_input}")
    dataset = GSInputDataset(
        gs_input_dir=actual_gs_input,
        depth_maps_dir=actual_depth_dir,
        max_frames=max_frames,
    )
    print(f"      Found {len(dataset.frames_meta)} keyframes. Depth priors: {dataset.depth_maps_dir is not None}")

    # 6. Configure Multi-Scale Progressive Resolution Scheduling
    base_h, base_w = dataset.base_h, dataset.base_w
    if multi_scale and iterations >= 1000:
        it_360 = min(400, int(iterations * (400 / 3000)))
        it_720 = min(1400, int(iterations * (1400 / 3000)))
        stages = [
            {
                "name": "Stage 1: Coarse Radiance Warmup (360p)",
                "iters": (1, it_360),
                "res": (base_h // 3, base_w // 3),  # e.g. (640, 360)
            },
            {
                "name": "Stage 2: Macro Structural Growth (720p)",
                "iters": (it_360 + 1, it_720),
                "res": (int(base_h * 2 / 3), int(base_w * 2 / 3)),  # e.g. (1280, 720)
            },
            {
                "name": "Stage 3: Native 1080p Detail & Topology Freeze",
                "iters": (it_720 + 1, iterations),
                "res": (base_h, base_w),             # (1920, 1080)
            },
        ]
    elif multi_scale and iterations >= 100:
        it_half = iterations // 2
        stages = [
            {
                "name": "Stage 1: Coarse (1/2 scale)",
                "iters": (1, it_half),
                "res": (base_h // 2, base_w // 2),
            },
            {
                "name": "Stage 2: Fine Native",
                "iters": (it_half + 1, iterations),
                "res": (base_h, base_w),
            },
        ]
    else:
        # Single-scale mode (or rapid test)
        target_res = (base_h // 4, base_w // 4) if iterations <= 100 else (base_h, base_w)
        stages = [
            {
                "name": "Full Training",
                "iters": (1, iterations),
                "res": target_res,
            }
        ]

    # 7. Setup Trainer Configuration with Scene-Proportional Budget
    densify_start = min(401, int(iterations * (401 / 3000))) if iterations >= 1000 else 50
    densify_stop = min(2000, int(iterations * (2000 / 3000))) if iterations >= 1000 else int(iterations * 0.7)
    d_cfg = DensityControlConfig.from_initial_surfels(
        num_init_surfels=len(surfel_cloud),
        growth_multiplier=2.5,
        max_hard_cap=primitive_budget if primitive_budget is not None else 1_400_000,
        scale_split_threshold=0.008,
    )
    print(f"      Proportional Primitive Budget: capped at {d_cfg.max_primitives:,} (2.5x of {len(surfel_cloud):,})")

    trainer_cfg = TrainerConfig(
        total_iterations=iterations,
        checkpoint_interval=checkpoint_interval,
        densify_start_iter=densify_start,
        densify_stop_iter=densify_stop,
        densify_interval=200 if iterations >= 1000 else 50,
        lr_opacity=0.005,
        density_config=d_cfg,
    )

    # Determine active starting stage based on start_iteration
    active_stage_idx = 0
    if start_iteration > 1:
        for s_idx, stg in enumerate(stages):
            s_low, s_high = stg["iters"]
            if s_low <= start_iteration <= s_high:
                active_stage_idx = s_idx
                break
            elif start_iteration > s_high and s_idx == len(stages) - 1:
                active_stage_idx = s_idx

    # Load initial stage data into host RAM (device=None) to strictly bound GPU VRAM
    current_stage = stages[active_stage_idx]
    initial_scene_data = dataset.load_scene_data(
        target_resolution=current_stage["res"],
        device=None,
        load_depth_normals=True,
    )

    trainer = Material2DGSTrainer(
        model=model,
        keyframes=initial_scene_data.keyframes,
        intrinsics=initial_scene_data.intrinsics,
        image_size=initial_scene_data.image_size,
        config=trainer_cfg,
    )

    # Resume model weights if requested
    if resume_checkpoint is not None:
        ckpt_p = Path(resume_checkpoint).resolve()
        if ckpt_p.is_file():
            print(f"[GlomeHomeTour] Resuming model weights from checkpoint: {ckpt_p}")
            trainer.load_checkpoint(ckpt_p)
        else:
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_p}")

    # 8. Progressive Training Loop Across Scales
    print(f"\n[4/4] Beginning 2DGS Progressive Multi-Scale Training ({iterations} iterations)...")
    train_start = time.time()
    last_stats: Dict[str, float] = {}

    def save_stage_snapshot(it_num: int, res: Tuple[int, int], stats: Optional[Dict[str, Any]] = None) -> None:
        stage_dir = out_path / "stages" / f"iter_{it_num:04d}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = stage_dir / f"checkpoint_iter_{it_num:04d}.pt"
        trainer.save_checkpoint(ckpt_path)

        from reconstruction.training.export_standard_ply import export_model_to_standard_3dgs_ply
        stage_ply = stage_dir / f"model_3dgs_iter_{it_num:04d}.ply"
        export_model_to_standard_3dgs_ply(trainer.model, stage_ply)

        stage_metrics = {
            "iteration": it_num,
            "resolution": list(res),
            "loss_total": stats.get("loss_total", 0.0) if stats else 0.0,
            "loss_l1": stats.get("loss_l1", 0.0) if stats else 0.0,
            "loss_ssim": stats.get("loss_ssim", 0.0) if stats else 0.0,
            "loss_normal": stats.get("loss_normal", 0.0) if stats else 0.0,
            "loss_depth": stats.get("loss_depth", 0.0) if stats else 0.0,
            "psnr": stats.get("psnr", 0.0) if stats else 0.0,
            "mean_opacity": stats.get("mean_opacity", float(trainer.model.opacity.mean().item())) if stats else float(trainer.model.opacity.mean().item()),
            "num_gaussians": trainer.model.num_gaussians,
            "elapsed_seconds": time.time() - train_start,
        }
        with open(stage_dir / "stage_metrics.json", "w", encoding="utf-8") as f:
            json.dump(stage_metrics, f, indent=2)

        try:
            with torch.no_grad():
                sample_kf = trainer.keyframes[0]
                sample_gb = trainer.rasterizer(
                    trainer.model,
                    extrinsics=sample_kf.w2c.to(trainer.device),
                    intrinsics=trainer.intrinsics,
                    image_size=trainer.image_size,
                )
                sample_rgb = sample_gb.albedo
                from PIL import Image
                sample_np = (sample_rgb[0].detach().cpu().permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
                Image.fromarray(sample_np).save(stage_dir / f"render_sample_iter_{it_num:04d}.png")
        except Exception as e:
            print(f"      [Warning] Could not save stage render sample: {e}")

        print(
            f"      >>> [Stage Log] Saved checkpoint & standard 3DGS PLY -> "
            f"stages/iter_{it_num:04d}/ ({trainer.model.num_gaussians:,} splats)"
        )

    # Save initial snapshot at iter 0 before any training steps
    if start_iteration == 1:
        print("[GlomeHomeTour] Saving initial stage 0 inspection snapshot...")
        save_stage_snapshot(0, current_stage["res"])

    for stage_idx in range(active_stage_idx, len(stages)):
        stage = stages[stage_idx]
        stage_start, stage_end = stage["iters"]
        start_iter = max(stage_start, start_iteration)
        end_iter = min(stage_end, iterations)

        if start_iter > end_iter or start_iter > iterations:
            continue

        stage_res = stage["res"]
        print(
            f"\n>>> Advancing to {stage['name']} | "
            f"Iterations [{start_iter} - {end_iter}] | Resolution: {stage_res[1]}x{stage_res[0]}"
        )

        # Load scene data for this resolution stage if different from active initial stage
        if stage_idx != active_stage_idx:
            del trainer.keyframes
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            stage_data = dataset.load_scene_data(
                target_resolution=stage_res,
                device=None,
                load_depth_normals=True,
            )
            trainer.update_dataset(
                keyframes=stage_data.keyframes,
                intrinsics=stage_data.intrinsics,
                image_size=stage_data.image_size,
            )

        for it in range(start_iter, end_iter + 1):
            it_t0 = time.time()
            step_stats = trainer.train_step(it)
            step_elapsed = time.time() - it_t0
            last_stats = step_stats

            # Logging
            if it % log_interval == 0 or it == end_iter or it == 1:
                norm_loss_str = f", Norm: {step_stats.get('loss_normal', 0.0):.4f}" if "loss_normal" in step_stats else ""
                print(
                    f"  [Iter {it:5d}/{iterations:5d}] "
                    f"Loss: {step_stats['loss_total']:.4f} "
                    f"(L1: {step_stats['loss_l1']:.4f}, SSIM: {step_stats['loss_ssim']:.4f}{norm_loss_str}) | "
                    f"PSNR: {step_stats.get('psnr', 0.0):.2f}dB | "
                    f"alpha_mean: {step_stats.get('mean_opacity', 0.0):.3f} | "
                    f"Gaussians: {int(step_stats['num_gaussians']):,} "
                    f"(+{int(step_stats['cloned'] + step_stats['split'])} -{int(step_stats['pruned'])}) | "
                    f"{step_elapsed*1000:.1f}ms/it"
                )

            # Intermediate Stage Logging & Checkpoints (every checkpoint_interval or at end of each stage)
            if (it % checkpoint_interval == 0 or it == end_iter) and it < iterations:
                save_stage_snapshot(it, stage_res, step_stats)

    total_time = time.time() - train_start

    # 9. Export Checkpoints and Compressed Package
    print("\n" + "=" * 70)
    print("  SERIALIZING FINAL ASSETS")
    print("=" * 70)

    # Uncompressed floating-point checkpoint
    final_ckpt = out_path / "material_2dgs_checkpoint.pt"
    trainer.save_checkpoint(final_ckpt)
    ckpt_size_mb = final_ckpt.stat().st_size / (1024 * 1024)
    print(f"1. FP32 Model Checkpoint:   {final_ckpt} ({ckpt_size_mb:.2f} MB)")

    # LightGaussian Compressed Web Package (<= 25 MB)
    zip_path = out_path / "walkthrough_2dgs.zip"
    zip_size = trainer.compressor.export_package_zip(trainer.model, zip_path)
    zip_size_mb = zip_size / (1024 * 1024)
    print(f"2. WebGL Walkthrough Bundle: {zip_path} ({zip_size_mb:.2f} MB)")
    if zip_size_mb <= 25.0:
        print("   [OK] Bundle size strictly complies with MLS limit (<= 25 MB).")
    else:
        print("   [WARNING] Bundle size exceeds MLS 25 MB limit!")

    # 3. Standard Gaussian Splatting PLY (Compatible with SuperSplat, WebGL, and Blender 3DGS tools)
    from reconstruction.training.export_standard_ply import export_model_to_standard_3dgs_ply
    ply_3dgs_path = out_path / "bedroom_standard_3dgs.ply"
    export_model_to_standard_3dgs_ply(trainer.model, ply_3dgs_path)
    ply_size_mb = ply_3dgs_path.stat().st_size / (1024 * 1024)
    print(f"3. Standard 3DGS PLY:        {ply_3dgs_path} ({ply_size_mb:.2f} MB)")

    summary = {
        "iterations": iterations,
        "total_time_seconds": total_time,
        "final_num_gaussians": trainer.model.num_gaussians,
        "final_loss": last_stats.get("loss_total", 0.0),
        "final_l1": last_stats.get("loss_l1", 0.0),
        "final_ssim": last_stats.get("loss_ssim", 0.0),
        "checkpoint_path": str(final_ckpt),
        "bundle_path": str(zip_path),
        "standard_ply_path": str(ply_3dgs_path),
        "bundle_size_mb": zip_size_mb,
    }
    with open(out_path / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"4. Summary Manifest:         {out_path / 'training_summary.json'}")
    print("=" * 70)
    return summary



def main() -> None:
    parser = argparse.ArgumentParser(
        description="GlomeHomeTour: Train 2DGS Material & Density Model from PLY surfels and keyframes."
    )
    parser.add_argument(
        "--scene-dir",
        type=str,
        default="backend/scenes/bedroom_complete_depth_results/GS_input",
        help="Path to GS_input directory or scene directory.",
    )
    parser.add_argument("--ply", type=str, default=None, help="Explicit path to surfels PLY file.")
    parser.add_argument("--transforms", type=str, default=None, help="Explicit path to transforms.json.")
    parser.add_argument("--depth-dir", type=str, default=None, help="Explicit path to depth_maps directory.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for 2DGS model and bundle.")
    parser.add_argument("--iterations", type=int, default=3000, help="Total training iterations.")
    parser.add_argument("--voxel-size", type=float, default=0.02, help="Voxel size in meters for downsampling.")
    parser.add_argument("--max-surfels", type=int, default=600_000, help="Cap on the INITIAL surfel count.")
    parser.add_argument("--budget", type=int, default=None, help="Hard cap on primitives after densification (defaults to 2.5x initial surfels).")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on number of training frames.")
    parser.add_argument("--device", type=str, default=None, help="PyTorch device ('cuda' or 'cpu').")
    parser.add_argument("--no-multi-scale", action="store_true", help="Disable multi-scale resolution training.")
    parser.add_argument("--checkpoint-interval", type=int, default=500, help="Checkpoint saving interval.")
    parser.add_argument("--log-interval", type=int, default=50, help="Console progress log interval.")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Path to checkpoint.pt to resume model weights from.")
    parser.add_argument("--start-iter", type=int, default=1, help="Iteration to start training from.")

    args = parser.parse_args()

    train_scene_from_ply_and_frames(
        scene_dir=args.scene_dir,
        ply_path=args.ply,
        transforms_path=args.transforms,
        depth_dir=args.depth_dir,
        output_dir=args.output_dir,
        iterations=args.iterations,
        voxel_size_m=args.voxel_size,
        max_surfels=args.max_surfels,
        primitive_budget=args.budget,
        multi_scale=not args.no_multi_scale,
        device=args.device,
        max_frames=args.max_frames,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        resume_checkpoint=args.resume_checkpoint,
        start_iteration=args.start_iter,
    )


if __name__ == "__main__":
    main()
