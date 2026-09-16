#!/usr/bin/env python3
"""Step 3's new logic: scene depth and covisibility read off the COLMAP model.

Everything else in step 3 is the existing `DynamicKeyframeSelector` walk. What
is new is the two measurements fed into it, so that is what is pinned here
against a hand-built model with known geometry.

    .venv/bin/python tests/test_step_filter_depth.py
"""

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

_backend = Path(__file__).resolve().parents[1]
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))
from Utilities.pipeline_paths import bootstrap

bootstrap()

_spec = importlib.util.spec_from_file_location(
    "step_filter_depth", _backend / "02_depth_estimation" / "step_filter_depth.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
colmap_tracks_and_depths = _mod.colmap_tracks_and_depths
TrackCovisibilitySelector = _mod.TrackCovisibilitySelector


def write_model(directory: Path, images: dict[str, list[int]], points: dict[int, tuple]):
    """A minimal COLMAP text model.

    `images` maps name -> the point ids it observes; every camera sits at the
    origin looking down +Z, so a point's Z *is* its depth in that view.
    """
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["# images"]
    for i, (name, p3d_ids) in enumerate(images.items(), start=1):
        lines.append(f"{i} 1 0 0 0 0 0 0 1 {name}")
        # X Y POINT3D_ID triples; the pixel coords are unused by this code path.
        lines.append(" ".join(f"{10.0 + j} {20.0 + j} {pid}" for j, pid in enumerate(p3d_ids)))
    (directory / "images.txt").write_text("\n".join(lines) + "\n")

    plines = ["# points3D"]
    for pid, xyz in points.items():
        plines.append(f"{pid} {xyz[0]} {xyz[1]} {xyz[2]} 200 200 200 0.5 1 0")
    (directory / "points3D.txt").write_text("\n".join(plines) + "\n")


def test_depth_is_median_of_the_frames_own_points():
    with tempfile.TemporaryDirectory() as tmp:
        sparse = Path(tmp) / "sparse"
        write_model(
            sparse,
            images={"a.jpg": [1, 2, 3], "b.jpg": [3, 4]},
            points={1: (0, 0, 1.0), 2: (0, 0, 2.0), 3: (0, 0, 3.0), 4: (0, 0, 9.0)},
        )
        tracks, depths = colmap_tracks_and_depths(sparse, ["a.jpg", "b.jpg"])

        assert tracks == [{1, 2, 3}, {3, 4}], tracks
        assert np.isclose(depths[0], 2.0), depths      # median(1, 2, 3)
        assert np.isclose(depths[1], 6.0), depths      # median(3, 9)


def test_unregistered_frame_falls_back_to_scene_median():
    """One starved frame must not drag the adaptive thresholds to zero."""
    with tempfile.TemporaryDirectory() as tmp:
        sparse = Path(tmp) / "sparse"
        write_model(
            sparse,
            images={"a.jpg": [1], "b.jpg": [2]},
            points={1: (0, 0, 2.0), 2: (0, 0, 4.0)},
        )
        tracks, depths = colmap_tracks_and_depths(sparse, ["a.jpg", "missing.jpg", "b.jpg"])

        assert tracks[1] == set(), tracks
        assert np.isclose(depths[1], 3.0), depths      # median(2, 4), not 0
        assert np.all(depths > 0)


def test_points_behind_the_camera_are_ignored():
    with tempfile.TemporaryDirectory() as tmp:
        sparse = Path(tmp) / "sparse"
        write_model(
            sparse,
            images={"a.jpg": [1, 2, 3]},
            points={1: (0, 0, -5.0), 2: (0, 0, 2.0), 3: (0, 0, 4.0)},
        )
        _, depths = colmap_tracks_and_depths(sparse, ["a.jpg"])
        assert np.isclose(depths[0], 3.0), depths      # median(2, 4); -5 dropped


def test_covisibility_is_the_shared_track_fraction():
    # min(|a|,|b|) in the denominator, not |a|: a frame that sees a few points,
    # all of which the other also sees, is fully covisible with it.
    selector = TrackCovisibilitySelector(
        [{1, 2, 3, 4}, {3, 4, 5, 6}, {1, 2}, set()],
        **vars(_mod.DynamicKeyframeSelector.for_2dgs_training()),
    )
    covis = lambda i, j: selector._mutual_covisibility(i, j, None, None)

    assert covis(0, 1) == 0.5, covis(0, 1)       # {3,4} of min(4,4)
    assert covis(0, 2) == 1.0, covis(0, 2)       # {1,2} of min(4,2)
    assert covis(1, 2) == 0.0, covis(1, 2)       # disjoint
    assert covis(0, 3) == 0.0, covis(0, 3)       # empty track set, no division by zero
    assert covis(0, 1) == covis(1, 0), "covisibility must be symmetric"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} passed")
