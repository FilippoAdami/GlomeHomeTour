import sys
from pathlib import Path

import numpy as np

STAGE = Path(__file__).resolve().parents[1] / "08_artifacts_generation"
if str(STAGE) not in sys.path:
    sys.path.insert(0, str(STAGE))

from generate_classified_object_floor_slices import _assign_slice


def test_tall_object_is_assigned_to_slice_with_greatest_vertical_overlap():
    bounds = np.array([[-0.2, 0.1, -0.2], [0.2, 2.42, 0.2]])
    slices = [
        {
            "floor_level_name": "BASE",
            "floor_planes": [{"plane_equation": [0.0, 1.0, 0.0, 0.0]}],
            "ceiling_cut_plane_equation": [0.0, 1.0, 0.0, -2.5],
        },
        {
            "floor_level_name": "BASEa",
            "floor_planes": [{"plane_equation": [0.0, 1.0, 0.0, -2.3]}],
            "ceiling_cut_plane_equation": [0.0, 1.0, 0.0, -3.5],
        },
    ]

    selected, scores = _assign_slice(bounds, slices)

    assert selected["floor_level_name"] == "BASE"
    assert [score["floor_level_name"] for score in scores] == ["BASE", "BASEa"]
    assert np.isclose(scores[0]["vertical_overlap_m"], 2.32)
    assert np.isclose(scores[1]["vertical_overlap_m"], 0.12)
