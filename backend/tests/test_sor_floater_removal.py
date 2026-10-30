import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

MODULE = Path(__file__).parents[1] / "03_FastGS_DNSplatter" / "gsplat_train.py"
spec = importlib.util.spec_from_file_location("gsplat_train", MODULE)
gsplat_train = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gsplat_train
spec.loader.exec_module(gsplat_train)
remove_cluster_floaters = gsplat_train.remove_cluster_floaters


def _make_dummy_splats(xyz: np.ndarray, device="cpu"):
    n = len(xyz)
    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.from_numpy(xyz).float()),
        "scales": torch.nn.Parameter(torch.zeros(n, 3)),
        "quats": torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.]] * n)),
        "opacities": torch.nn.Parameter(torch.zeros(n)),
        "sh0": torch.nn.Parameter(torch.zeros(n, 1, 3)),
        "shN": torch.nn.Parameter(torch.zeros(n, 15, 3)),
    }).to(device)


def test_remove_cluster_floaters_synthetic():
    np.random.seed(42)
    # 1. Main significant cluster around origin (150 points within radius 0.04m)
    c1 = np.random.randn(150, 3) * 0.015

    # 2. Points near main cluster (10 points at distance ~0.03m from c1)
    near = np.random.randn(10, 3) * 0.005 + np.array([0.03, 0.0, 0.0])

    # 3. Detached floater cluster (15 points at (2.0, 2.0, 2.0))
    c2 = np.random.randn(15, 3) * 0.01 + np.array([2.0, 2.0, 2.0])

    # 4. Isolated floaters
    iso = np.array([
        [5.0, 5.0, 5.0],
        [-3.0, 4.0, 2.0],
        [1.0, -4.0, 3.0],
    ])

    xyz = np.vstack([c1, near, c2, iso])
    splats = _make_dummy_splats(xyz)

    # With min_cluster_size=50, min_distance=0.10:
    # c1 (+ near) has 160 points -> significant cluster (size >= 50)
    # c2 has 15 points (size < 50) and is at distance ~2.8m (> 0.10m) -> floater
    # iso has 3 points (distance > 3m) -> floater
    filtered_splats, stats = remove_cluster_floaters(
        splats,
        min_cluster_size=50,
        min_distance=0.10,
        min_samples=5,
    )

    assert stats["num_clusters_identified"] == 1
    assert stats["num_floaters_detected"] == 18  # 15 from c2 + 3 from iso
    assert stats["surviving_gaussians"] == 160
    assert len(filtered_splats["means"]) == 160

    # Ensure all parameter tensors remain aligned in length
    for key in ("scales", "quats", "opacities", "sh0", "shN"):
        assert len(filtered_splats[key]) == 160


def test_remove_cluster_floaters_empty():
    xyz = np.empty((0, 3), dtype=np.float32)
    splats = _make_dummy_splats(xyz)
    filtered_splats, stats = remove_cluster_floaters(splats, min_cluster_size=50, min_distance=0.10)
    assert stats["total_gaussians"] == 0
    assert stats["surviving_gaussians"] == 0
    assert stats["num_clusters_identified"] == 0
    assert stats["num_floaters_detected"] == 0
    assert len(filtered_splats["means"]) == 0


def test_remove_cluster_floaters_all_inliers():
    np.random.seed(123)
    c1 = np.random.randn(200, 3) * 0.02
    splats = _make_dummy_splats(c1)
    filtered_splats, stats = remove_cluster_floaters(splats, min_cluster_size=50, min_distance=0.10)
    assert stats["total_gaussians"] == 200
    assert stats["num_clusters_identified"] == 1
    assert stats["num_floaters_detected"] == 0
    assert stats["surviving_gaussians"] == 200
    assert len(filtered_splats["means"]) == 200


def test_remove_cluster_floaters_fallback_protection():
    # If all clusters have fewer than min_cluster_size points, the largest cluster is preserved
    np.random.seed(99)
    c1 = np.random.randn(30, 3) * 0.01  # 30 points
    c2 = np.random.randn(5, 3) * 0.01 + np.array([2.0, 0, 0])  # 5 points
    xyz = np.vstack([c1, c2])
    splats = _make_dummy_splats(xyz)

    # min_cluster_size = 100 > 30, so without protection all would be wiped out
    filtered_splats, stats = remove_cluster_floaters(splats, min_cluster_size=100, min_distance=0.10, min_samples=3)
    assert stats["num_clusters_identified"] == 1
    assert stats["surviving_gaussians"] == 30
    assert stats["num_floaters_detected"] == 5
