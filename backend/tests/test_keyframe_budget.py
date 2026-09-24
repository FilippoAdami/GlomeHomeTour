#!/usr/bin/env python3
"""Step 3's keyframe budget: scene size -> frame count -> which frames.

The assertions live in `keyframe_budget._demo()` next to the code they pin;
this is the pytest entry point for them.

    .venv/bin/python tests/test_keyframe_budget.py
"""

import sys
from pathlib import Path

_backend = Path(__file__).resolve().parents[1]
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))
from Utilities.pipeline_paths import bootstrap

bootstrap()

import keyframe_budget


def test_budget_and_topup():
    keyframe_budget._demo()


def test_coverage_prune_respects_motion_delta():
    # If frames 0 and 2 have covisibility 1.0, but angular turn between 0 and 2 exceeds threshold,
    # frame 1 must NOT be pruned.
    cov = [{1, 2}, {1, 2}, {1, 2}]
    chain = [0, 1, 2]
    always = lambda a, b: 1.0

    # Motion delta returns (dist_m, rot_deg)
    # Between 0 and 2: rot = 25 deg (exceeds max_rotation_deg 16.0)
    def motion(a, b):
        if (a, b) == (0, 2) or (b, a) == (0, 2):
            return 0.1, 25.0
        return 0.05, 5.0

    pruned = keyframe_budget.coverage_prune(
        cov, chain, max_keyframes=2, covisibility=always, min_covisibility=0.5,
        motion_delta=motion, max_rotation_deg=16.0,
    )
    # Frame 1 is preserved to prevent an unacceptable 25 deg jump
    assert pruned == [0, 1, 2]


if __name__ == "__main__":
    test_budget_and_topup()
    test_coverage_prune_respects_motion_delta()
    print("ok")
