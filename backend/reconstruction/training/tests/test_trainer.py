"""Unit tests for Stage 7: Material2DGSTrainer Multi-View Optimization Loop."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from reconstruction.training.model import Material2DGSModel
from reconstruction.training.trainer import (
    Material2DGSTrainer,
    TrainerConfig,
    TrainingKeyframeData,
)


def _create_synthetic_training_setup() -> Tuple[Material2DGSModel, list[TrainingKeyframeData]]:
    # Create 5 surfels in front of camera
    pos = torch.tensor([
        [0.0, 0.0, -1.5],
        [0.1, 0.0, -1.5],
        [-0.1, 0.0, -1.5],
        [0.0, 0.1, -1.5],
        [0.0, -0.1, -1.5],
    ], dtype=torch.float32)

    quats = torch.zeros((5, 4), dtype=torch.float32)
    quats[:, 0] = 1.0  # Identity

    scales = torch.full((5, 2), float(np.log(0.1)), dtype=torch.float32)
    opacity = torch.full((5, 1), 2.0, dtype=torch.float32)  # High opacity
    albedo = torch.full((5, 3), 0.0, dtype=torch.float32)   # Initial gray
    roughness = torch.full((5, 1), 0.0, dtype=torch.float32)
    metallic = torch.full((5, 1), -2.0, dtype=torch.float32)

    model = Material2DGSModel(
        xyz=pos,
        rotation=quats,
        scaling=scales,
        opacity=opacity,
        albedo=albedo,
        roughness=roughness,
        metallic=metallic,
    )

    # Two synthetic keyframes with target red images
    w2c_1 = torch.eye(4, dtype=torch.float32)
    w2c_2 = torch.eye(4, dtype=torch.float32)
    w2c_2[0, 3] = 0.05  # Slight baseline shift

    # Target: 16x16 red image
    target_img = torch.zeros((3, 16, 16), dtype=torch.float32)
    target_img[0, :, :] = 0.9  # Red

    target_norm = torch.zeros((3, 16, 16), dtype=torch.float32)
    target_norm[2, :, :] = 1.0

    keyframes = [
        TrainingKeyframeData(w2c=w2c_1, image_rgb=target_img, normal_prior=target_norm),
        TrainingKeyframeData(w2c=w2c_2, image_rgb=target_img, normal_prior=target_norm),
    ]

    return model, keyframes


def test_trainer_single_and_multi_step():
    """Verify that Material2DGSTrainer executes training steps and checkpoints."""
    model, keyframes = _create_synthetic_training_setup()
    intrinsics = [16.0, 16.0, 8.0, 8.0]
    image_size = (16, 16)

    cfg = TrainerConfig(
        total_iterations=5,
        densify_start_iter=2,
        densify_stop_iter=4,
        densify_interval=2,
        checkpoint_interval=2,
    )

    trainer = Material2DGSTrainer(
        model=model,
        keyframes=keyframes,
        intrinsics=intrinsics,
        image_size=image_size,
        config=cfg,
    )

    # 1. Single train step
    stats = trainer.train_step(1)
    assert "loss_total" in stats
    assert "loss_l1" in stats
    assert "loss_ssim" in stats
    assert stats["loss_total"] > 0.0

    # 2. Multi-step train loop with checkpoint saving
    with tempfile.TemporaryDirectory() as tmp_dir:
        summary = trainer.train(total_iterations=4, output_dir=tmp_dir)

        assert summary["final_num_gaussians"] > 0
        assert (Path(tmp_dir) / "material_2dgs_checkpoint.pt").exists()
        assert (Path(tmp_dir) / "walkthrough_2dgs.zip").exists()
        assert summary["compressed_zip_size_bytes"] > 0
