import pytest
import numpy as np
from pathlib import Path
import sys

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
sys.path.insert(0, str(_backend_dir / "06b_semantic_segmentation_pipeline"))

from progressive_instance_clustering import (
    run_progressive_clustering,
    absorb_small_fragments,
)


def test_progressive_relaxation():
    # 4 clusters in a line: 0 - 1 - 2 - 3
    # Boundary 0-1: affinity 0.85 (very high)
    # Boundary 1-2: affinity 0.55 (medium)
    # Boundary 2-3: affinity 0.90 but CONCAVE contact seam
    adj_pairs = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
    ], dtype=np.int64)

    # 10 co-visible frames for all
    co_visibility = np.array([10, 10, 10], dtype=np.int32)
    # co-occurrence counts: 9/10 = 0.90, 6/10 = 0.60, 9/10 = 0.90
    co_occurrence = np.array([9, 6, 9], dtype=np.int32)
    adj_concave = np.array([False, False, True], dtype=bool)
    base_sp_clusters = np.array([0, 1, 2, 3], dtype=np.int32)

    # Stage 1: tau=0.80 -> 0 and 1 should merge; 2 and 3 do NOT merge (concave); 1 and 2 do NOT merge (0.60 < 0.80)
    sp_inst, hist = run_progressive_clustering(
        adj_pairs=adj_pairs,
        co_visibility=co_visibility,
        co_occurrence=co_occurrence,
        adj_concave=adj_concave,
        base_sp_clusters=base_sp_clusters,
        threshold_schedule=[0.80],
        max_concave_ratio=0.35,
        min_covis=2,
    )
    assert sp_inst[0] == sp_inst[1]
    assert sp_inst[1] != sp_inst[2]
    assert sp_inst[2] != sp_inst[3]

    # Two-stage schedule: [0.80, 0.50]
    # In stage 2 (tau=0.50), 1 and 2 should merge!
    # But 3 should STILL remain separate because concave seam is protected!
    sp_inst2, hist2 = run_progressive_clustering(
        adj_pairs=adj_pairs,
        co_visibility=co_visibility,
        co_occurrence=co_occurrence,
        adj_concave=adj_concave,
        base_sp_clusters=base_sp_clusters,
        threshold_schedule=[0.80, 0.50],
        max_concave_ratio=0.35,
        min_covis=2,
    )
    # 0, 1, 2 united into one instance
    assert sp_inst2[0] == sp_inst2[1] == sp_inst2[2]
    # 3 remains completely distinct!
    assert sp_inst2[3] != sp_inst2[2]
    assert hist2["final_instances"] == 2


def test_absorb_small_fragments():
    # 2 large superpoints (100 faces each) and 1 tiny superpoint (10 faces)
    # sp 0: cluster A (100 faces)
    # sp 1: cluster B (10 faces - tiny!)
    # sp 2: cluster C (100 faces)
    # sp 1 connects to sp 0 (non-concave) and sp 2 (concave)
    sp_instances = np.array([0, 1, 2], dtype=np.int32)
    face_to_sp = np.concatenate([
        np.full(100, 0),
        np.full(10, 1),
        np.full(100, 2),
    ])
    adj_pairs = np.array([
        [0, 1],
        [1, 2],
    ], dtype=np.int64)
    adj_concave = np.array([False, True], dtype=bool)

    cleaned = absorb_small_fragments(
        sp_instances=sp_instances,
        face_to_sp=face_to_sp,
        adj_pairs=adj_pairs,
        adj_concave=adj_concave,
        min_faces=50,
    )
    # sp 1 should be absorbed into sp 0 (non-concave), NOT into sp 2 (concave)
    assert cleaned[1] == cleaned[0]
    assert cleaned[2] != cleaned[0]
