"""Unit tests for Stage 7: Material2DGSTrainer Multi-View Optimization Loop."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from model import Material2DGSModel
from trainer import (
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


def test_opacity_sparsity_only_penalizes_non_original_splats():
    """is_original splats (trusted depth-prior init) must be excluded from
    the opacity sparsity pull; only grown (cloned/split) splats are penalized."""
    model, keyframes = _create_synthetic_training_setup()
    intrinsics = [16.0, 16.0, 8.0, 8.0]
    image_size = (16, 16)

    assert bool(model.is_original.all()), "fresh model should start fully 'original'"

    cfg = TrainerConfig(
        total_iterations=1,
        densify_start_iter=10_000,  # disable density control for this check
        densify_stop_iter=10_000,
        gamma_opacity_sparsity=0.05,
    )
    trainer = Material2DGSTrainer(model=model, keyframes=keyframes, intrinsics=intrinsics, image_size=image_size, config=cfg)

    # All-original: sparsity term must be absent from the step metrics.
    stats_all_original = trainer.train_step(1)
    assert "loss_opacity_sparsity" not in stats_all_original

    # Mark one splat as grown (non-original) and confirm the term now appears.
    trainer.model.is_original[0] = False
    stats_with_grown = trainer.train_step(1)
    assert "loss_opacity_sparsity" in stats_with_grown


def test_prune_continues_after_densify_stop_without_growth():
    """After densify_stop_iter, splats should still get pruned (floater cleanup)
    but never cloned/split (topology stays frozen), up to prune_stop_iter."""
    model, keyframes = _create_synthetic_training_setup()
    intrinsics = [16.0, 16.0, 8.0, 8.0]
    image_size = (16, 16)

    cfg = TrainerConfig(
        total_iterations=6,
        densify_start_iter=1,
        densify_stop_iter=2,
        prune_stop_iter=6,
        densify_interval=1,
    )
    trainer = Material2DGSTrainer(model=model, keyframes=keyframes, intrinsics=intrinsics, image_size=image_size, config=cfg)

    num_gaussians_history = []
    for it in range(1, 7):
        trainer.train_step(it)
        num_gaussians_history.append(trainer.model.num_gaussians)

    # No growth allowed once past densify_stop_iter=2: count must be non-increasing from there.
    for prev, cur in zip(num_gaussians_history[2:], num_gaussians_history[3:]):
        assert cur <= prev, "topology grew after densify_stop_iter -- growth freeze is broken"


def test_no_density_control_past_prune_stop_iter():
    """Past prune_stop_iter, densify_and_prune must not run at all."""
    model, keyframes = _create_synthetic_training_setup()
    intrinsics = [16.0, 16.0, 8.0, 8.0]
    image_size = (16, 16)

    cfg = TrainerConfig(
        total_iterations=6,
        densify_start_iter=1,
        densify_stop_iter=2,
        prune_stop_iter=3,
        densify_interval=1,
    )
    trainer = Material2DGSTrainer(model=model, keyframes=keyframes, intrinsics=intrinsics, image_size=image_size, config=cfg)

    for it in range(1, 4):
        trainer.train_step(it)
    count_at_prune_stop = trainer.model.num_gaussians

    for it in range(4, 7):
        trainer.train_step(it)
    assert trainer.model.num_gaussians == count_at_prune_stop
