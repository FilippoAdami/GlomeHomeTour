import pytest
import numpy as np
from pathlib import Path
import sys

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
sys.path.insert(0, str(_backend_dir / "06b_semantic_segmentation_pipeline"))

from mask_affinity_clustering import (
    AffinityResult,
    cluster_with_affinity_and_geometry,
)


def test_cluster_with_affinity_and_geometry():
    # 3 geometric clusters: 0, 1, 2
    # Boundary 0-1: high affinity (0.8), non-concave -> should merge
    # Boundary 1-2: high affinity (0.9), concave seam -> should cut
    adj_pairs = np.array([
        [0, 1],
        [1, 2],
    ], dtype=np.int64)
    affinity = np.array([0.8, 0.9], dtype=np.float32)
    co_visibility = np.array([5, 5], dtype=np.int32)
    adj_concave = np.array([False, True], dtype=bool)
    base_sp_clusters = np.array([0, 1, 2], dtype=np.int32)
    face_to_sp = np.repeat(np.arange(3), 100)

    sp_clusters, face_clusters, stats = cluster_with_affinity_and_geometry(
        adj_pairs=adj_pairs,
        affinity=affinity,
        co_visibility=co_visibility,
        adj_concave=adj_concave,
        base_sp_clusters=base_sp_clusters,
        face_to_superpoint=face_to_sp,
        tau_affinity=0.5,
        max_concave_ratio=0.35,
        min_covis=2,
    )

    # 0 and 1 should merge into same cluster
    # 2 should remain separate
    assert sp_clusters[0] == sp_clusters[1]
    assert sp_clusters[1] != sp_clusters[2]
    assert stats["num_clusters"] == 2


def test_min_covis_threshold():
    # If co-visibility is below min_covis, boundary should not merge even if affinity is high
    adj_pairs = np.array([[0, 1]], dtype=np.int64)
    affinity = np.array([1.0], dtype=np.float32)
    co_visibility = np.array([1], dtype=np.int32) # covis = 1 < min_covis = 2
    adj_concave = np.array([False], dtype=bool)
    base_sp_clusters = np.array([0, 1], dtype=np.int32)
    face_to_sp = np.repeat(np.arange(2), 100)

    sp_clusters, _, stats = cluster_with_affinity_and_geometry(
        adj_pairs=adj_pairs,
        affinity=affinity,
        co_visibility=co_visibility,
        adj_concave=adj_concave,
        base_sp_clusters=base_sp_clusters,
        face_to_superpoint=face_to_sp,
        tau_affinity=0.5,
        max_concave_ratio=0.35,
        min_covis=2,
    )
    assert sp_clusters[0] != sp_clusters[1]
    assert stats["num_clusters"] == 2
