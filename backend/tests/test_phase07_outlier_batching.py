import sys
from pathlib import Path

import numpy as np

ALIGNMENT_DIR = Path(__file__).resolve().parents[1] / "07_alignment"
if str(ALIGNMENT_DIR) not in sys.path:
    sys.path.insert(0, str(ALIGNMENT_DIR))

import object_outlier_regions as outliers


def test_batched_outside_distances_match_per_object_queries(monkeypatch):
    triangles = np.asarray([
        [[0, -2, -2], [0, 2, -2], [0, 2, 2]],
        [[0, -2, -2], [0, 2, 2], [0, -2, 2]],
    ], dtype=np.float32)
    surface = {
        "scene": outliers._make_ray_scene(triangles),
        "axes": np.asarray([0, 0], dtype=np.int8),
        "signs": np.asarray([1, 1], dtype=np.int8),
    }
    point_sets = [
        np.asarray([[0.1, -1, -1], [-0.2, 1, 1]], dtype=np.float32),
        np.asarray([[0.5, 0, 0], [-0.75, 1.5, -1.5]], dtype=np.float32),
    ]
    expected = [outliers._signed_outside_distance(points, surface) for points in point_sets]

    original = outliers._signed_outside_distance
    queried_lengths = []

    def counted(points, query_surface):
        queried_lengths.append(len(points))
        return original(points, query_surface)

    monkeypatch.setattr(outliers, "_signed_outside_distance", counted)
    actual = outliers._signed_outside_distances_by_object(point_sets, surface)

    assert queried_lengths == [sum(map(len, point_sets))]
    for actual_points, expected_points in zip(actual, expected):
        np.testing.assert_array_equal(actual_points, expected_points)
