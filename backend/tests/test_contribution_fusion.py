#!/usr/bin/env python3
"""Unit tests for Stage 06b Step 3: GS Contribution-Aware Fusion."""

import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(backend_dir / "06b_semantic_segmentation_pipeline"))

from Utilities.pipeline_paths import bootstrap
bootstrap()

from contribution_fusion import ContributionObservationGraph


def test_contribution_observation_graph(tmp_path):
    N = 50
    graph = ContributionObservationGraph(num_splats=N)

    # Synthetic A_matrix for frame 1: 3 masks
    A1 = np.zeros((N, 3), dtype=np.float32)
    A1[5, 0] = 12.5
    A1[6, 0] = 45.2
    A1[10, 1] = 8.0
    A1[20, 2] = 100.0

    rec1 = graph.add_view_contributions("frame_001", A1)
    assert rec1 == 3
    assert len(graph.mask_to_gaussians[("frame_001", 0)]) == 2
    assert graph.mask_to_gaussians[("frame_001", 0)][5] == 12.5

    # Synthetic A_matrix for frame 2: 2 masks
    A2 = np.zeros((N, 2), dtype=np.float32)
    A2[6, 0] = 50.0 # overlaps with frame 1 on Gaussian 6!
    A2[15, 1] = 25.0

    rec2 = graph.add_view_contributions("frame_002", A2)
    assert rec2 == 2

    # Check observations on Gaussian 6
    obs6 = graph.gaussian_observations[6]
    assert len(obs6) == 2
    assert obs6[0][:2] == ("frame_001", 0)
    assert np.isclose(obs6[0][2], 45.2)
    assert obs6[1][:2] == ("frame_002", 0)
    assert np.isclose(obs6[1][2], 50.0)

    # Check stats
    stats = graph.compute_summary_stats()
    assert stats["total_gaussians"] == N
    assert stats["covered_gaussians"] == 5 # 5, 6, 10, 15, 20
    assert stats["total_observations"] == 6

    # Test serialization
    graph.save(tmp_path)
    assert (tmp_path / "contribution_observations.npz").is_file()
    assert (tmp_path / "contribution_stats.json").is_file()
    assert (tmp_path / "sample_contribution_observations.json").is_file()

    loaded = np.load(tmp_path / "contribution_observations.npz")
    assert loaded["num_splats"] == N
    assert len(loaded["weights"]) == 6
    assert np.isclose(loaded["weights"][1], 45.2)
