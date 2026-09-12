"""GlomeHomeTour Backend: 2DGS Radiance & Density Training Pipeline.

Orchestrates multi-view training, Taming density control, checkpoint
serialization, and LightGaussian compression packaging.

Canonical 2DGS (Huang et al. 2024): splats carry view-independent SH-DC
radiance that the rasterizer composites straight into the image. The deferred
Cook-Torrance path is bypassed during training -- an invented sun direction and
sky/ground hemisphere bakes fake shading into every exported colour.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from reconstruction.training.compressor import LightGaussianCompressor
from reconstruction.training.density_control import DensityControlConfig, TamingDensityController
from reconstruction.training.losses import ReconstructionLoss
from reconstruction.training.model import Material2DGSModel
from reconstruction.training.pbr_shader import DeferredCookTorranceShader
from reconstruction.training.rasterizer_interface import Base2DGSRasterizer, get_default_rasterizer


@dataclass
class TrainingKeyframeData:
    """Standardized training frame container (supports real Keyframe or mocks)."""
    w2c: torch.Tensor             # (4, 4) world-to-camera matrix
    image_rgb: torch.Tensor       # (3, H, W) normalized float in [0, 1]
    normal_prior: Optional[torch.Tensor] = None  # (3, H, W) optional Depth Anything normal map
    depth_prior: Optional[torch.Tensor] = None   # (1, H, W) optional metric depth map


@dataclass
class TrainerConfig:
    """Hyperparameters for 2DGS material optimization."""
    total_iterations: int = 3000
    lr_position_init: float = 1.6e-4
    lr_position_final: float = 1.6e-6
    lr_position_delay_mult: float = 0.01
    lr_position_max_steps: int = 3000
    lr_rotation: float = 1e-3
    lr_scaling: float = 5e-3
    lr_opacity: float = 5e-3             # 0.005 (stops opacity decay and preserves solid alpha >= 0.85)
    lr_albedo: float = 2.5e-3
    camera_convention: str = "opengl"  # ARCore/transforms.json cameras are -Z forward
    lambda_ssim: float = 0.2
    gamma_normal: float = 0.05
    gamma_depth: float = 0.1
    densify_start_iter: int = 401       # Start active Taming growth at Stage 2 (720p)
    densify_stop_iter: int = 2000       # Freeze topology after step 2000 for pure photometric convergence
    densify_interval: int = 200         # Densify every 200 steps
    log_interval: int = 50
    checkpoint_interval: int = 500
    max_depth_ceiling: Optional[float] = None
    density_config: DensityControlConfig = field(default_factory=DensityControlConfig)


class Material2DGSTrainer:
    """Orchestrates 2DGS training over keyframe sequences."""

    def __init__(
        self,
        model: Material2DGSModel,
        keyframes: Sequence[TrainingKeyframeData],
        intrinsics: Any,
        image_size: Tuple[int, int],
        config: Optional[TrainerConfig] = None,
        rasterizer: Optional[Base2DGSRasterizer] = None,
        shader: Optional[DeferredCookTorranceShader] = None,
        loss_fn: Optional[ReconstructionLoss] = None,
        density_controller: Optional[TamingDensityController] = None,
    ) -> None:
        self.model = model
        self.keyframes = keyframes
        self.intrinsics = intrinsics
        self.image_size = image_size
        self.config = config or TrainerConfig()

        self.device = model.xyz.device
        self.rasterizer = rasterizer or get_default_rasterizer(
            camera_convention=self.config.camera_convention,
        )
        self.shader = shader or DeferredCookTorranceShader().to(self.device)
        self.loss_fn = loss_fn or ReconstructionLoss(
            lambda_ssim=self.config.lambda_ssim,
            gamma_normal=self.config.gamma_normal,
            gamma_depth=self.config.gamma_depth,
        ).to(self.device)
        self.density_controller = density_controller or TamingDensityController(config=self.config.density_config)
        self.compressor = LightGaussianCompressor()

        # Initialize density controller buffers
        self.density_controller.init_buffers(self.model.num_gaussians, self.device)

        # Setup Adam optimizer with per-parameter learning rates
        self.optimizer = self._create_optimizer()

    def update_dataset(
        self,
        keyframes: Sequence[TrainingKeyframeData],
        intrinsics: Any,
        image_size: Tuple[int, int],
    ) -> None:
        """Update active training keyframes, camera intrinsics, and image resolution."""
        self.keyframes = keyframes
        self.intrinsics = intrinsics
        self.image_size = image_size

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create Adam optimizer with decoupled learning rate groups."""
        param_groups = [
            {"params": [self.model._xyz], "lr": self.config.lr_position_init, "name": "_xyz"},
            {"params": [self.model._rotation], "lr": self.config.lr_rotation, "name": "_rotation"},
            {"params": [self.model._scaling], "lr": self.config.lr_scaling, "name": "_scaling"},
            {"params": [self.model._opacity], "lr": self.config.lr_opacity, "name": "_opacity"},
            {"params": [self.model._albedo], "lr": self.config.lr_albedo, "name": "_albedo"},
        ]
        if hasattr(self.model, "_features_rest") and self.model._features_rest is not None and self.model._features_rest.numel() > 0:
            param_groups.append(
                {"params": [self.model._features_rest], "lr": self.config.lr_albedo / 20.0, "name": "_features_rest"}
            )
        # _roughness / _metallic are deliberately absent: nothing reads them in
        # the direct-radiance path, so optimizing them is pure overhead.
        return optim.Adam(param_groups, eps=1e-15)

    def _update_learning_rates(self, iteration: int) -> None:
        """Exponential decay schedule for positional learning rates."""
        for param_group in self.optimizer.param_groups:
            if param_group.get("name") == "_xyz":
                # Exponential decay schedule
                t = min(1.0, iteration / max(1, self.config.lr_position_max_steps))
                lr = math.exp((1.0 - t) * math.log(self.config.lr_position_init) + t * math.log(self.config.lr_position_final))
                param_group["lr"] = lr

    def train_step(self, iteration: int) -> Dict[str, float]:
        """Execute a single forward-backward training step over a sampled keyframe."""
        self._update_learning_rates(iteration)
        self.optimizer.zero_grad()

        # Sample a keyframe at random: a strict round-robin locks the density
        # control interval to the same few views every time.
        frame_idx = int(torch.randint(len(self.keyframes), (1,)).item())
        kf = self.keyframes[frame_idx]

        w2c = kf.w2c.to(self.device)
        gt_rgb = kf.image_rgb.to(self.device).unsqueeze(0)  # (1, 3, H, W)
        if gt_rgb.dtype == torch.uint8:
            gt_rgb = gt_rgb.float() / 255.0
        gt_norm = None
        if kf.normal_prior is not None:
            gt_norm = kf.normal_prior.to(self.device).unsqueeze(0)
            if gt_norm.dtype == torch.float16:
                gt_norm = gt_norm.float()
        gt_depth = None
        if kf.depth_prior is not None:
            gt_depth = kf.depth_prior.to(self.device).unsqueeze(0).float()

        # 1. Forward pass: Rasterize G-Buffer
        gbuffer = self.rasterizer(
            self.model,
            extrinsics=w2c,
            intrinsics=self.intrinsics,
            image_size=self.image_size,
        )

        # 2. The G-Buffer albedo channel IS the composited radiance.
        rendered_rgb = gbuffer.albedo

        # 3. Loss computation
        coverage = None
        if gbuffer.alpha is not None:
            coverage = (gbuffer.alpha > 0.5).to(dtype=rendered_rgb.dtype)
        total_loss, metrics = self.loss_fn(
            pred_rgb=rendered_rgb,
            gt_rgb=gt_rgb,
            pred_normal=gbuffer.normal,
            gt_normal=gt_norm,
            pred_depth=gbuffer.depth,
            gt_depth=gt_depth,
            coverage_mask=coverage,
        )

        # 4. Backward pass
        total_loss.backward()

        # Accumulate spatial gradients for density control
        self.density_controller.accumulate_gradients(self.model)

        # Optimization step
        self.optimizer.step()

        # 5. Density Control Scheduling
        cloned, split, pruned = 0, 0, 0
        if (
            iteration >= self.config.densify_start_iter
            and iteration <= self.config.densify_stop_iter
            and iteration % self.config.densify_interval == 0
        ):
            cloned, split, pruned = self.density_controller.densify_and_prune(
                self.model,
                optimizer=self.optimizer,
                iteration=iteration,
                max_depth_ceiling=self.config.max_depth_ceiling,
                remaining_intervals=(
                    (self.config.densify_stop_iter - iteration) // self.config.densify_interval + 1
                ),
            )

        # No periodic opacity reset: it is a 3DGS crutch for floaters born of a
        # random SfM init. Here the depth prior gives real surfaces, and
        # flattening every opacity to 0.1 leaves the scene transparent.

        with torch.no_grad():
            mse = torch.mean((rendered_rgb.clamp(0.0, 1.0) - gt_rgb) ** 2)
            psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-10))

        step_stats = {
            "loss_total": float(total_loss.item()),
            "loss_l1": float(metrics["loss_l1"].item()),
            "loss_ssim": float(metrics["loss_ssim"].item()),
            "psnr": float(psnr.item()),
            "mean_opacity": float(self.model.opacity.mean().item()),
            "num_gaussians": float(self.model.num_gaussians),
            "cloned": float(cloned),
            "split": float(split),
            "pruned": float(pruned),
        }
        for key in ("loss_normal", "loss_depth"):
            if key in metrics:
                step_stats[key] = float(metrics[key].item())

        return step_stats

    def train(
        self,
        total_iterations: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Execute full training optimization loop."""
        num_iters = total_iterations or self.config.total_iterations
        out_path = Path(output_dir) if output_dir else None
        if out_path:
            out_path.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        final_stats: Dict[str, float] = {}

        for it in range(1, num_iters + 1):
            stats = self.train_step(it)
            final_stats = stats

            # Periodic checkpointing
            if out_path and it % self.config.checkpoint_interval == 0:
                self.save_checkpoint(out_path / f"checkpoint_iter_{it:05d}.pt")

        elapsed = time.time() - start_time
        summary = {
            "elapsed_seconds": elapsed,
            "final_num_gaussians": self.model.num_gaussians,
            "final_loss": final_stats.get("loss_total", 0.0),
            "final_l1": final_stats.get("loss_l1", 0.0),
            "final_ssim": final_stats.get("loss_ssim", 0.0),
            "final_psnr": final_stats.get("psnr", 0.0),
            "final_mean_opacity": final_stats.get("mean_opacity", 0.0),
        }

        # Export final assets
        if out_path:
            # 1. Floating-point model checkpoint
            self.save_checkpoint(out_path / "material_2dgs_checkpoint.pt")
            # 2. Compressed WebGL bundle (<= 25 MB)
            zip_size = self.compressor.export_package_zip(self.model, out_path / "walkthrough_2dgs.zip")
            summary["compressed_zip_size_bytes"] = zip_size
            summary["compressed_zip_size_mb"] = zip_size / (1024 * 1024)

        return summary

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save model checkpoint to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "num_gaussians": self.model.num_gaussians,
        }, str(p))

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load model checkpoint from disk, updating parameters to match checkpoint primitive count."""
        ckpt = torch.load(str(path), map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt)
        for name, param in state.items():
            if hasattr(self.model, name):
                setattr(self.model, name, nn.Parameter(param.to(self.device)))
        self.optimizer = self._create_optimizer()
        self.density_controller.init_buffers(self.model.num_gaussians, self.device)
