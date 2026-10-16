import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location("run_dinov2_patches", Path(__file__).with_name("run_dinov2_patches.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_only_neighboring_similar_patches_share_labels():
    features = np.array([[[1, 0], [1, 0], [0, 1]],
                         [[0, 1], [1, 0], [0, 1]]], dtype=np.float32)
    labels = module.group_neighbors(features, 0.9)
    assert labels[0, 0] == labels[0, 1] == labels[1, 1]
    assert labels[0, 2] == labels[1, 2]
    assert labels[0, 0] != labels[0, 2]
    assert labels[1, 0] != labels[0, 2]  # Same descriptor, disconnected region.


def test_running_cluster_mean_rejects_neighbor_that_only_matches_last_patch():
    angles = np.deg2rad([0, 20, 40, 60])
    features = np.stack((np.cos(angles), np.sin(angles)), axis=-1).astype(np.float32)[None]
    labels = module.grow_cluster_means(features, 0.85)[0]
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[2]  # 60° matches 40°, but not the mean of 0°, 20°, 40°.


def test_small_region_merges_with_semantically_closer_touching_region():
    labels = np.array([[0, 0, 2, 1, 1]], dtype=np.uint16)
    features = np.array([[[1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]], dtype=np.float32)
    cleaned = module.merge_small_regions(labels, features, min_patches=2)
    assert np.array_equal(cleaned, [[0, 0, 1, 1, 1]])
    features[0, 2] = [1, 0]
    cleaned = module.merge_small_regions(labels, features, min_patches=2)
    assert np.array_equal(cleaned, [[0, 0, 0, 1, 1]])


def test_grid_rounds_wait_to_update_region_means():
    angles = np.deg2rad([0, 20, 40, 60])
    features = np.stack((np.cos(angles), np.sin(angles)), axis=-1).astype(np.float32)[None]
    labels = module.merge_grid_rounds(features, 0.85)[0]
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_grow_and_merge_grid_seeds_mean_vs_boundary():
    # 2 rows x 2 cols grid test
    features = np.array([
        [[1.0, 0.0], [1.0, 0.0]],
        [[0.0, 1.0], [0.0, 1.0]]
    ], dtype=np.float32)
    # With 2x1 grid, seeds at (0, 0) and (1, 0)
    labels_high = module.grow_and_merge_grid_seeds(features, threshold=0.9, merge_criterion="mean", n_rows=2, n_cols=1)
    assert len(np.unique(labels_high)) == 2
    labels_low = module.grow_and_merge_grid_seeds(features, threshold=-0.1, merge_criterion="mean", n_rows=2, n_cols=1)
    assert len(np.unique(labels_low)) == 1

    labels_b_low = module.grow_and_merge_grid_seeds(features, threshold=-0.1, merge_criterion="boundary", n_rows=2, n_cols=1)
    assert len(np.unique(labels_b_low)) == 1

