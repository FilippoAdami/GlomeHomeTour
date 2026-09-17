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


if __name__ == "__main__":
    test_budget_and_topup()
    print("ok")
