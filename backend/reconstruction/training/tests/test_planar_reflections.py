"""Unit tests for Stage 5: Planar Mirror Detection & Virtual Reflection Passes."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from reconstruction.training.model import Material2DGSModel
from reconstruction.training.planar_reflections import MirrorPlane, PlanarMirrorDetector


def test_householder_reflection_matrix():
    """Verify mathematical properties of the Householder 4x4 reflection matrix."""
    # Plane at z = 2.0 with normal pointing along +Z: n = [0, 0, 1], d = -2.0
    plane = MirrorPlane(
        normal=torch.tensor([0.0, 0.0, 1.0]),
        d=-2.0,
        inlier_indices=torch.tensor([0, 1]),
        rmse_fit=0.001,
        area_estimate_m2=1.5,
    )

    h_refl = plane.compute_reflection_matrix_4x4()

    # 1. Reflection matrix is involutory: H^2 = I
    h_sq = torch.matmul(h_refl, h_refl)
    assert torch.allclose(h_sq, torch.eye(4), atol=1e-5)

    # 2. Reflecting point [0, 0, 0] across plane z = 2 should give [0, 0, 4]
    pt_orig = torch.tensor([0.0, 0.0, 0.0, 1.0])
    pt_refl = torch.matmul(h_refl, pt_orig)
    assert torch.allclose(pt_refl[:3], torch.tensor([0.0, 0.0, 4.0]), atol=1e-5)

    # 3. Point on plane [1, 1, 2] should remain unchanged
    pt_on_plane = torch.tensor([1.0, 1.0, 2.0, 1.0])
    pt_on_plane_refl = torch.matmul(h_refl, pt_on_plane)
    assert torch.allclose(pt_on_plane_refl[:3], pt_on_plane[:3], atol=1e-5)


def test_planar_mirror_detector():
    """Verify RANSAC discovery of planar mirrors on synthetic scene."""
    detector = PlanarMirrorDetector(
        min_metallic=0.7,
        max_roughness=0.15,
        min_cluster_points=10,
        ransac_distance_threshold=0.02,
    )

    # Synthesize 30 mirror points on plane z = 1.0 (metallic=0.95, roughness=0.05)
    rng = np.random.RandomState(42)
    n_mirror = 30
    x_mir = rng.uniform(-0.5, 0.5, n_mirror)
    y_mir = rng.uniform(-0.5, 0.5, n_mirror)
    z_mir = np.full(n_mirror, 1.0) + rng.normal(0, 0.002, n_mirror)
    pos_mir = np.column_stack([x_mir, y_mir, z_mir]).astype(np.float32)

    # Synthesize 30 diffuse background points (metallic=0.1, roughness=0.6)
    pos_diff = rng.uniform(-2.0, 2.0, size=(30, 3)).astype(np.float32)

    all_pos = np.vstack([pos_mir, pos_diff])
    n_total = len(all_pos)

    # Normals: mirror points have +Z normal
    norms = np.zeros((n_total, 3), dtype=np.float32)
    norms[:n_mirror, 2] = 1.0
    norms[n_mirror:] = rng.randn(30, 3).astype(np.float32)
    norms[n_mirror:] /= np.linalg.norm(norms[n_mirror:], axis=-1, keepdims=True)

    # Materials
    met = np.zeros((n_total, 1), dtype=np.float32)
    met[:n_mirror] = 0.95
    met[n_mirror:] = 0.1

    rou = np.zeros((n_total, 1), dtype=np.float32)
    rou[:n_mirror] = 0.05
    rou[n_mirror:] = 0.6

    # Logit parameters for model
    logit_met = torch.from_numpy(np.log(met / (1.0 - met)))
    logit_rou = torch.from_numpy(np.log(rou / (1.0 - rou)))

    model = Material2DGSModel(
        xyz=torch.from_numpy(all_pos),
        rotation=torch.zeros((n_total, 4)),
        scaling=torch.zeros((n_total, 2)),
        opacity=torch.zeros((n_total, 1)),
        albedo=torch.zeros((n_total, 3)),
        roughness=logit_rou,
        metallic=logit_met,
    )
    # Give unit rotation so normal property is defined
    model._rotation.data[:, 0] = 1.0

    mirrors = detector.detect_mirrors(model)
    assert len(mirrors) >= 1, "Failed to detect synthetic mirror plane"

    best_mirror = mirrors[0]
    assert len(best_mirror.inlier_indices) >= 20, "Should detect majority of synthetic mirror points"
    # Normal should point along Z
    assert abs(best_mirror.normal[2].item()) > 0.9
    # Distance d should be near -1.0
    assert abs(best_mirror.d + 1.0) < 0.1
