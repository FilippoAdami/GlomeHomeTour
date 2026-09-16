"""Integration and unit tests for PLY + Keyframes -> 2DGS training pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from initialization import SurfelCloud
from dataset import GSInputDataset
from model import Material2DGSModel
from run_scene_training import (
    resolve_scene_paths,
    train_scene_from_ply_and_frames,
)

SCENE_GS_INPUT = Path("backend/scenes/bedroom_complete_depth_results/GS_input").resolve()
REAL_PLY = SCENE_GS_INPUT / "bedroom_complete_surfels.ply"
DEPTH_MAPS_DIR = Path("backend/scenes/bedroom_complete_depth_results/depth_maps").resolve()


@pytest.mark.skipif(not REAL_PLY.exists(), reason="Real bedroom PLY not found")
def test_surfel_cloud_from_ply_downsampling():
    """Verify fast binary PLY reading and spatial voxel downsampling on real bedroom surfels."""
    # Test loading with 3.5cm voxel downsampling
    surfel_cloud = SurfelCloud.from_ply(
        REAL_PLY,
        voxel_downsample_m=0.035,
        max_surfels=500_000,
    )

    assert len(surfel_cloud) > 100_000
    assert len(surfel_cloud) <= 500_000
    assert surfel_cloud.positions.shape == (len(surfel_cloud), 3)
    assert surfel_cloud.normals.shape == (len(surfel_cloud), 3)
    assert surfel_cloud.tangent_u.shape == (len(surfel_cloud), 3)
    assert surfel_cloud.tangent_v.shape == (len(surfel_cloud), 3)
    assert surfel_cloud.scales_2d.shape == (len(surfel_cloud), 2)
    assert surfel_cloud.colors_rgb.shape == (len(surfel_cloud), 3)

    # Normals must be unit vectors
    norm_len = np.linalg.norm(surfel_cloud.normals, axis=-1)
    assert np.allclose(norm_len, 1.0, atol=1e-3)

    # Tangent vectors u, v must be orthogonal to normal and each other
    dot_un = np.sum(surfel_cloud.tangent_u * surfel_cloud.normals, axis=-1)
    dot_vn = np.sum(surfel_cloud.tangent_v * surfel_cloud.normals, axis=-1)
    dot_uv = np.sum(surfel_cloud.tangent_u * surfel_cloud.tangent_v, axis=-1)
    assert np.all(np.abs(dot_un) < 1e-3)
    assert np.all(np.abs(dot_vn) < 1e-3)
    assert np.all(np.abs(dot_uv) < 1e-3)


@pytest.mark.skipif(not SCENE_GS_INPUT.exists(), reason="GS_input directory not found")
def test_gs_input_dataset_multiscale():
    """Verify GSInputDataset loads keyframes, normal priors, and resizes properly."""
    dataset = GSInputDataset(
        gs_input_dir=SCENE_GS_INPUT,
        depth_maps_dir=DEPTH_MAPS_DIR,
        max_frames=4,
    )

    assert dataset.base_w == 1080
    assert dataset.base_h == 1920
    assert len(dataset.frames_meta) == 4

    # Load 1/4 resolution
    data_small = dataset.load_scene_data(target_resolution=(480, 270), load_depth_normals=True)
    assert data_small.image_size == (480, 270)
    assert data_small.intrinsics.w == 270
    assert data_small.intrinsics.h == 480
    assert len(data_small.keyframes) == 4

    kf0 = data_small.keyframes[0]
    assert kf0.image_rgb.shape == (3, 480, 270)
    assert kf0.w2c.shape == (4, 4)
    if kf0.normal_prior is not None:
        assert kf0.normal_prior.shape == (3, 480, 270)
    if kf0.depth_prior is not None:
        assert kf0.depth_prior.shape == (1, 480, 270)


@pytest.mark.skipif(not REAL_PLY.exists(), reason="Real bedroom PLY not found")
def test_material_2dgs_model_from_ply():
    """Verify Material2DGSModel initializes directly from real PLY with physical parameters."""
    model = Material2DGSModel.from_ply(
        REAL_PLY,
        voxel_downsample_m=0.08,  # Coarse downsample for fast test
        max_surfels=50_000,
        device=torch.device("cpu"),
    )

    assert model.num_gaussians > 0
    assert model.num_gaussians <= 50_000
    assert model.xyz.shape == (model.num_gaussians, 3)
    assert model.rotation.shape == (model.num_gaussians, 4)
    assert model.scaling.shape == (model.num_gaussians, 2)
    assert model.opacity.shape == (model.num_gaussians, 1)
    assert model.albedo.shape == (model.num_gaussians, 3)
    assert model.roughness.shape == (model.num_gaussians, 1)
    assert model.metallic.shape == (model.num_gaussians, 1)

    # All parameter activations should be physically valid
    assert torch.all(model.opacity >= 0.0) and torch.all(model.opacity <= 1.0)
    assert torch.all(model.albedo >= 0.0) and torch.all(model.albedo <= 1.0)
    assert torch.all(model.roughness >= 0.0) and torch.all(model.roughness <= 1.0)
    assert torch.all(model.metallic >= 0.0) and torch.all(model.metallic <= 1.0)


@pytest.mark.skipif(not (REAL_PLY.exists() and SCENE_GS_INPUT.exists()), reason="Scene assets missing")
def test_end_to_end_training_step_real_scene():
    """Execute multi-scale training run on real scene assets and verify loss backpropagation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        summary = train_scene_from_ply_and_frames(
            scene_dir=SCENE_GS_INPUT,
            output_dir=tmp_dir,
            iterations=3,
            voxel_size_m=0.15,  # 15cm grid for rapid unit test
            max_surfels=500,
            multi_scale=True,
            device="cpu",
            max_frames=2,
            checkpoint_interval=2,
            log_interval=1,
        )

        assert summary["iterations"] == 3
        assert summary["final_num_gaussians"] > 0
        assert not np.isnan(summary["final_loss"])
        assert summary["final_loss"] > 0.0
        assert Path(summary["checkpoint_path"]).is_file()
        assert Path(summary["bundle_path"]).is_file()
        assert summary["bundle_size_mb"] <= 25.0
