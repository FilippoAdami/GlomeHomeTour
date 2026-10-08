#!/usr/bin/env python3
"""Unit tests for Stage 06b Step 2: Class-Agnostic 2D -> GS Fusion."""

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

from mask_gs_fusion import (
    MaskObservationGraph,
    find_default_checkpoint,
)


def test_mask_observation_graph_construction(tmp_path):
    N = 100
    graph = MaskObservationGraph(num_splats=N)

    # View 1: 3 masks
    vis_gids_v1 = np.array([10, 11, 12, 20, 21, 22], dtype=np.int32)
    # Mask 0 covers 10, 11, 12; Mask 1 covers 20, 21; Mask 2 covers 22
    sampled_v1 = np.array([
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ], dtype=np.uint8)

    rec1 = graph.add_view_observations("frame_001", vis_gids_v1, sampled_v1)
    assert rec1 == 3

    # View 2: 2 masks
    # Overlaps with view 1 on Gaussians 11, 12, 20
    vis_gids_v2 = np.array([11, 12, 13, 20, 30], dtype=np.int32)
    sampled_v2 = np.array([
        [1, 1, 1, 0, 0], # covers 11, 12, 13
        [0, 0, 0, 1, 1], # covers 20, 30
    ], dtype=np.uint8)

    rec2 = graph.add_view_observations("frame_002", vis_gids_v2, sampled_v2)
    assert rec2 == 2

    # Check Gaussian observations
    assert ("frame_001", 0) in graph.gaussian_observations[10]
    assert ("frame_001", 0) in graph.gaussian_observations[11]
    assert ("frame_002", 0) in graph.gaussian_observations[11]
    assert len(graph.gaussian_observations[11]) == 2
    assert len(graph.gaussian_observations[0]) == 0 # unseen

    # Check mask to Gaussians
    np.testing.assert_array_equal(graph.mask_to_gaussians[("frame_001", 0)], [10, 11, 12])
    np.testing.assert_array_equal(graph.mask_to_gaussians[("frame_002", 0)], [11, 12, 13])

    # Check affinities (shared Gaussians)
    affinities = graph.compute_mask_affinity(min_shared_gaussians=1)
    assert len(affinities) >= 2
    # ('frame_001', 0) and ('frame_002', 0) share 11 and 12 (shared=2, union=4, jaccard=0.5)
    match_edge = [e for e in affinities if (e[0] == ("frame_001", 0) and e[1] == ("frame_002", 0)) or
                                           (e[1] == ("frame_001", 0) and e[0] == ("frame_002", 0))]
    assert len(match_edge) == 1
    assert match_edge[0][2] == 2 # shared count
    assert abs(match_edge[0][3] - 0.5) < 1e-5 # jaccard

    # Check serialization
    graph.save(tmp_path)
    assert (tmp_path / "observations.npz").is_file()
    assert (tmp_path / "observation_stats.json").is_file()
    assert (tmp_path / "sample_observations.json").is_file()

    # Verify loaded data
    npz_data = np.load(tmp_path / "observations.npz")
    assert npz_data["num_splats"] == N
    assert len(npz_data["gaussian_ids"]) == len(npz_data["frame_ids"]) == len(npz_data["mask_indices"])

    with open(tmp_path / "observation_stats.json") as f:
        stats = json.load(f)
    assert stats["total_gaussians"] == N
    assert stats["covered_gaussians"] == 8 # 10, 11, 12, 13, 20, 21, 22, 30
    assert stats["total_masks"] == 5


def test_find_default_checkpoint():
    workspace = Path("backend/current_scene")
    if workspace.exists():
        cp = find_default_checkpoint(workspace)
        assert cp.is_file()
        assert cp.suffix == ".pt"
