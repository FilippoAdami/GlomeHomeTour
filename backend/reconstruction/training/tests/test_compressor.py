"""Unit tests for Stage 6: LightGaussian Vector Quantization & Asset Packaging."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from reconstruction.training.compressor import LightGaussianCompressor
from reconstruction.training.model import Material2DGSModel


def _make_random_model(n: int = 500, seed: int = 42) -> Material2DGSModel:
    rng = np.random.RandomState(seed)

    pos = rng.uniform(-5.0, 5.0, size=(n, 3)).astype(np.float32)
    rot = rng.randn(n, 4).astype(np.float32)
    rot /= np.linalg.norm(rot, axis=-1, keepdims=True)

    scales = rng.uniform(0.01, 0.05, size=(n, 2)).astype(np.float32)
    opacities = rng.uniform(0.2, 0.9, size=(n, 1)).astype(np.float32)
    albedo = rng.uniform(0.05, 0.95, size=(n, 3)).astype(np.float32)
    roughness = rng.uniform(0.05, 0.8, size=(n, 1)).astype(np.float32)
    metallic = rng.uniform(0.0, 0.9, size=(n, 1)).astype(np.float32)

    # Invert activations to parameter logits
    logit_op = np.log(opacities / (1.0 - opacities))
    logit_alb = np.log(albedo / (1.0 - albedo))
    logit_rou = np.log(roughness / (1.0 - roughness))
    logit_met = np.log(np.clip(metallic, 1e-4, 1.0 - 1e-4) / (1.0 - np.clip(metallic, 1e-4, 1.0 - 1e-4)))
    log_scales = np.log(scales)

    return Material2DGSModel(
        xyz=torch.from_numpy(pos),
        rotation=torch.from_numpy(rot),
        scaling=torch.from_numpy(log_scales),
        opacity=torch.from_numpy(logit_op),
        albedo=torch.from_numpy(logit_alb),
        roughness=torch.from_numpy(logit_rou),
        metallic=torch.from_numpy(logit_met),
    )


def test_compressor_quantization_and_roundtrip():
    """Verify Vector Quantization, binary serialization, and decompression."""
    compressor = LightGaussianCompressor()
    orig_model = _make_random_model(n=300)

    # 1. Compress to in-memory bundle
    bundle = compressor.compress(orig_model)
    assert bundle.num_primitives == 300
    assert bundle.positions_q16.shape == (300, 3)
    assert bundle.rotations_q8.shape == (300, 4)
    assert bundle.albedo_codebook.shape == (256, 3)
    assert bundle.material_codebook.shape == (64, 2)
    assert bundle.scales_codebook.shape == (256, 2)

    # 2. Serialize to compact binary
    raw_bin = compressor.serialize_to_binary(bundle)
    assert raw_bin.startswith(b"2DGS")
    # Expected size: Header (12B) + Bbox (24B) + Codebooks (3072 + 512 + 2048 = 5632B) + Primitives (300 * 14B = 4200B) = ~9868 B
    assert len(raw_bin) < 15_000

    # 3. Deserialize back to bundle
    deserialized_bundle = compressor.deserialize_from_binary(raw_bin)
    assert deserialized_bundle.num_primitives == 300
    assert np.array_equal(deserialized_bundle.positions_q16, bundle.positions_q16)

    # 4. Decompress back to Material2DGSModel
    rec_model = compressor.decompress(deserialized_bundle)
    assert rec_model.num_gaussians == 300

    # Verify numerical accuracy within quantization limits:
    # Positions within 1 mm error
    pos_err = torch.abs(orig_model.xyz - rec_model.xyz).mean()
    assert pos_err < 0.005, f"Position error too high: {pos_err}"

    # Albedo within VQ tolerance (mean abs error < 0.08)
    alb_err = torch.abs(orig_model.albedo - rec_model.albedo).mean()
    assert alb_err < 0.08, f"Albedo VQ error too high: {alb_err}"


def test_export_package_zip():
    """Verify walkthrough_2dgs.zip packaging and file size constraint."""
    compressor = LightGaussianCompressor()
    model = _make_random_model(n=1000)

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / "walkthrough_2dgs.zip"
        size_bytes = compressor.export_package_zip(model, zip_path)

        assert zip_path.exists()
        assert size_bytes > 0
        # 1000 Gaussians should easily compress to under 50 KB
        assert size_bytes < 50_000
