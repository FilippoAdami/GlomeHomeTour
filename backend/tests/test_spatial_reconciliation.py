"""
Unit tests for Stage 06b Step 9: Global Spatial Instance Reconciliation.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
sys.path.insert(0, str(_backend_dir / "06b_semantic_segmentation_pipeline"))

from spatial_reconciliation import (
    InstanceViewStats,
    ReconciliationConfig,
    evaluate_pair_reconciliation,
    run_spatial_reconciliation,
)


def test_conflict_veto():
    """Verify that instances seen together in different masks are strictly vetoed."""
    meta_i = {
        "name": "obj_0",
        "dominant_rgb": [100, 100, 100],
        "surface_area_m2": 0.5,
    }
    meta_j = {
        "name": "obj_1",
        "dominant_rgb": [100, 100, 100],
        "surface_area_m2": 0.5,
    }

    stats_i = InstanceViewStats(
        instance_id=0,
        visible_frames={0, 1, 2},
        frame_masks={0: {1}, 1: {1}, 2: {1}},
        first_frame=0,
        last_frame=2,
    )
    stats_j = InstanceViewStats(
        instance_id=1,
        visible_frames={0, 1, 2},
        frame_masks={0: {2}, 1: {2}, 2: {2}},
        first_frame=0,
        last_frame=2,
    )

    cfg = ReconciliationConfig(max_conflict_count=1, min_prob_threshold=0.45)
    prob, details = evaluate_pair_reconciliation(
        i=0,
        j=1,
        min_dist=0.05,
        meta_i=meta_i,
        meta_j=meta_j,
        stats_i=stats_i,
        stats_j=stats_j,
        cfg=cfg,
    )

    assert prob == 0.0
    assert "VETO" in details["reason"]
    assert details["S_diff"] == 3


def test_cooccurrence_pass():
    """Verify that disconnected instances seen in the same mask are reconciled."""
    meta_i = {
        "name": "bench_pad",
        "dominant_rgb": [50, 50, 50],
        "surface_area_m2": 0.3,
    }
    meta_j = {
        "name": "bench_leg",
        "dominant_rgb": [55, 55, 55],
        "surface_area_m2": 0.1,
    }

    stats_i = InstanceViewStats(
        instance_id=0,
        visible_frames={0, 1, 2},
        frame_masks={0: {5}, 1: {5}, 2: {5}},
        first_frame=0,
        last_frame=2,
    )
    stats_j = InstanceViewStats(
        instance_id=1,
        visible_frames={0, 1, 2},
        frame_masks={0: {5}, 1: {5}, 2: {5}},
        first_frame=0,
        last_frame=2,
    )

    cfg = ReconciliationConfig(min_same_mask_count=1, min_prob_threshold=0.45)
    prob, details = evaluate_pair_reconciliation(
        i=0,
        j=1,
        min_dist=0.05,
        meta_i=meta_i,
        meta_j=meta_j,
        stats_i=stats_i,
        stats_j=stats_j,
        cfg=cfg,
    )

    assert prob >= 0.50
    assert "CO-OCCUR" in details["reason"]
    assert details["S_same"] == 3
    assert details["S_diff"] == 0
