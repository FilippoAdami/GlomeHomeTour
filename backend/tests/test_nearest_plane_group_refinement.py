import importlib.util
from pathlib import Path
import sys

import numpy as np
import open3d as o3d


SCRIPT = (Path(__file__).resolve().parents[1] / "07_alignment"
          / "temporary_alternative_pipeline" / "fit_contours_near_regularized_planes.py")
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("fit_contours_near_regularized_planes", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_group_refines_again_on_a_second_axis_after_first_move(monkeypatch):
    def axis_distance(samples, contour_vertices, candidates, face_count, *_args):
        position = np.asarray(samples).mean(axis=(0, 1))
        result = []
        for axis, direction, distance in candidates:
            moved = position.copy()
            moved[axis] += direction * distance
            result.append({
                "axis": axis,
                "direction": direction,
                "distance": distance,
                "average_distance": abs(moved[0] - 0.10) + abs(moved[1] + 0.14),
                "outside_triangle_count": 0,
                "outside_contour_vertex_count": 0,
                "constraint_outside_count": 0,
                "constraint_sample_count": len(contour_vertices),
                "floor_below_count": 0,
                "floor_point_count": 1,
            })
        return result

    monkeypatch.setattr(MODULE, "candidate_metrics", axis_distance)
    items = [{
        "samples": np.zeros((1, 4, 3)),
        "contour_vertices": np.zeros((3, 3)),
        "contour": np.array([True]),
        "scene": None,
        "axes": None,
        "signs": None,
        "sorted_object_y": np.array([1.0]),
        "floor_y": 0.0,
    }]

    translation, metrics, baseline, selected, steps = MODULE.find_group_translation(
        items, outside_fraction_limit=0.05
    )

    np.testing.assert_allclose(translation, [0.10, -0.14, 0.0], atol=MODULE.REFINE_STEP_M)
    assert [step["axis"] for step in steps] == [1, 0]
    assert metrics["selected_average_distance_m"] < 1e-8
    assert np.linalg.norm(translation) > MODULE.MAX_TRANSLATION_M
    assert all(step["distance_m"] <= MODULE.MAX_TRANSLATION_M for step in steps)
    np.testing.assert_allclose(baseline[0]["average_distance"], 0.24)
    assert selected[0]["average_distance"] < 1e-8


def test_group_can_move_to_a_feasible_pose_when_baseline_is_outside_limit(monkeypatch):
    def narrow_feasible_pose(samples, contour_vertices, candidates, face_count, *_args):
        position = np.asarray(samples).mean(axis=(0, 1))
        result = []
        for axis, direction, distance in candidates:
            moved = position.copy()
            moved[axis] += direction * distance
            inside_window = 0.009 <= moved[0] <= 0.011 and abs(moved[1]) < 1e-8
            outside = 0 if inside_window else 1
            result.append({
                "axis": axis,
                "direction": direction,
                "distance": distance,
                "average_distance": (0.0 if abs(moved[0]) < 1e-8
                                     else 1.0 + abs(moved[0] - 0.01)),
                "outside_triangle_count": outside,
                "outside_contour_vertex_count": outside,
                "constraint_outside_count": outside,
                "constraint_sample_count": len(contour_vertices),
                "floor_below_count": 0,
                "floor_point_count": 1,
            })
        return result

    monkeypatch.setattr(MODULE, "candidate_metrics", narrow_feasible_pose)
    items = [{
        "samples": np.zeros((1, 4, 3)),
        "contour_vertices": np.zeros((3, 3)),
        "contour": np.array([True]),
        "scene": None,
        "axes": None,
        "signs": None,
        "sorted_object_y": np.array([1.0]),
        "floor_y": 0.0,
    }]

    translation, metrics, baseline, selected, _ = MODULE.find_group_translation(
        items, outside_fraction_limit=0.05
    )

    np.testing.assert_allclose(translation, [0.01, 0.0, 0.0], atol=MODULE.REFINE_STEP_M)
    assert not MODULE.feasible(baseline[0], 0.05)
    assert MODULE.feasible(selected[0], 0.05)
    assert metrics["feasible_translation_found"]


def test_five_percent_limit_counts_unique_contour_vertices():
    triangles = np.array([[0, 1, 2], [1, 3, 2]])
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    contour_vertices = MODULE.unique_contour_vertices(
        vertices, triangles, np.array([True, True])
    )

    assert len(contour_vertices) == 4
    one_outside = {
        "constraint_outside_count": 1,
        "constraint_sample_count": 20,
        "floor_below_count": 0,
        "floor_point_count": 100,
    }
    two_outside = {**one_outside, "constraint_outside_count": 2}

    assert MODULE.feasible(one_outside, 0.05)
    assert not MODULE.feasible(two_outside, 0.05)


def test_wall_median_target_uses_directional_ray_even_if_another_axis_is_closer():
    vertices = np.array([
        [1, -1, -1], [1, 1, -1], [1, -1, 1], [1, 1, 1],
        [-1, 0.1, -1], [1, 0.1, -1], [-1, 0.1, 1], [1, 0.1, 1],
    ], dtype=np.float32)
    triangles = np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6], [5, 7, 6]], dtype=np.uint32)
    mesh = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(vertices), o3d.core.Tensor(triangles))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    samples = np.full((1, 4, 3), [0.5, 0.0, 0.0], dtype=np.float64)

    target, coordinate = MODULE.target_wall_plane(
        samples, np.array([0.05, 0.0, 0.0]), scene, np.array([0, 0, 1, 1])
    )

    assert target == (0, 1, 10)
    np.testing.assert_allclose(coordinate, 1.0, atol=1e-6)
