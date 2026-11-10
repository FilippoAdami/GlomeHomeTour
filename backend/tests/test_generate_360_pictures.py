from pathlib import Path
import json
import sys

import numpy as np
import pytest

STAGE = Path(__file__).resolve().parents[1] / "08_artifacts_generation"
sys.path.insert(0, str(STAGE))

from generate_360_pictures import (
    floor_y, height_policy, mesh_scene, sample_floor, generate_candidates,
    triangle_visibility, slice_targets, adaptive_candidates, select_points, tour_links, publish, level_max_height,
    height_retry_reason, plan_floor_points,
)


def floor(width=4, depth=4, y=0):
    v = np.array([[0, y, 0], [width, y, 0], [width, y, depth], [0, y, depth]], np.float32)
    return v, np.array([[0, 2, 1], [0, 3, 2]], np.int32)


def test_metric_and_uncalibrated_height_radius_policies():
    metric = height_policy({"output_coordinate_units": "meters"}, [2, 8])
    assert metric["height"] == 1.5 and metric["max_pose_distance"] == 1
    assert metric["allowed_heights"] == pytest.approx(np.arange(1.5, 0.69, -0.2))
    colmap = height_policy({}, [9, 5, 7])
    assert colmap["height"] == 2.5
    assert colmap["max_pose_distance"] == pytest.approx(0.66 * 2.5)
    assert colmap["distance_scale"] == pytest.approx(2.5 / 1.5)
    assert colmap["allowed_heights"][0] == colmap["height"]
    assert colmap["allowed_heights"][-1] == pytest.approx(2.5 * 7 / 15)
    assert -np.diff(colmap["allowed_heights"]) == pytest.approx(np.full(4, 2 * 2.5 / 15))


def test_floor_height_is_vertical_in_a_tilted_plane():
    xz = np.array([[0, 0], [2, 0], [0, 2]])
    assert np.allclose(floor_y(xz, [-0.1, 1, -0.2, -3]), [3, 3.2, 3.4])


def test_final_height_and_euclidean_real_pose_radius():
    v, f = floor()
    points, grid = sample_floor(v, f, 0.1)
    scene = mesh_scene(v, f)
    poses = np.array([[2, 1.8, 2]])
    candidates, stats, _ = generate_candidates(points, grid, poses, 1.5, 1, 1, scene)
    assert len(candidates) > 0
    assert np.allclose(candidates[:, 1], 1.5)
    assert np.all(np.linalg.norm(candidates - poses[0], axis=1) <= 1 + 1e-7)
    assert all(s["horizontal_quality"] == pytest.approx(1) for s in stats)
    # A horizontal-only test would wrongly accept this camera 1.1 units above target.
    candidates, _, _ = generate_candidates(points, grid, np.array([[2, 2.6, 2]]), 1.5, 1, 1, scene)
    assert len(candidates) == 0


def test_low_ceiling_does_not_penalize_horizontal_clearance():
    v, f = floor()
    top, top_f = floor(y=1.55)
    scene = mesh_scene(np.vstack([v, top]), np.vstack([f, top_f + len(v)]))
    points, grid = sample_floor(v, f, 0.1)
    candidates, stats, _ = generate_candidates(points, grid, np.array([[2, 1.5, 2]]), 1.5, 1, 1, scene,
                                              mesh_scene(top, top_f))
    assert len(candidates) and min(s["horizontal_quality"] for s in stats) == pytest.approx(1)


def test_fixed_height_cannot_cross_a_low_ceiling():
    v, f = floor()
    top, top_f = floor(y=1.4)
    points, grid = sample_floor(v, f, 0.1)
    candidates, _, _ = generate_candidates(points, grid, np.array([[2, 1.3, 2]]), 1.5, 1, 1,
                                            mesh_scene(v, f), mesh_scene(top, top_f))
    assert len(candidates) == 0


def test_source_segment_blocks_ceiling_crossing_without_ceiling_metadata():
    v, f = floor()
    top, top_f = floor(y=1.4)
    points, grid = sample_floor(v, f, 0.1)
    scene = mesh_scene(np.vstack([v, top]), np.vstack([f, top_f + len(v)]))
    candidates, _, _ = generate_candidates(points, grid, np.array([[2, 1.3, 2]]), 1.5, 1, 1, scene)
    assert len(candidates) == 0


def test_triangle_coverage_counts_visible_objects_and_rejects_missing_surface():
    v, f = floor()
    top, top_f = floor(width=2, y=0.7)
    vertices, faces = np.vstack([v, top]), np.vstack([f, top_f + len(v)])
    scene = mesh_scene(vertices, faces)
    targets = np.vstack([vertices[faces].mean(1), [10, 0, 10]])
    visible = triangle_visibility(scene, np.array([[1, 1.5, 2]]), targets, 1)
    assert visible[0, 2:4].all()  # The furniture surface is valid coverage.
    assert not visible[0, 4]  # A miss beyond the mesh cannot count.
    assert not visible[0, 1]  # The tabletop occludes one underlying floor centroid.


def test_height_stays_standard_without_obstruction():
    v, f = floor()
    points, grid = sample_floor(v, f, 0.1)
    policy = height_policy({"output_coordinate_units": "meters"}, [3])
    candidates, stats, _ = adaptive_candidates(points, grid, np.array([[2, 1.5, 2]]), policy, mesh_scene(v, f))
    assert len(candidates) and all(s["height_above_floor"] == 1.5 for s in stats)
    assert all(s["height_selection_reason"] == "standard_height" for s in stats)


def test_height_lowers_for_close_ceiling_but_keeps_pose_radius():
    v, f = floor()
    top, top_f = floor(y=1.55)
    scene = mesh_scene(np.vstack([v, top]), np.vstack([f, top_f + len(v)]))
    points, grid = sample_floor(v, f, 0.1)
    policy = height_policy({"output_coordinate_units": "meters"}, [2])
    candidates, stats, _ = adaptive_candidates(points, grid, np.array([[2, 1.5, 2]]), policy, scene, mesh_scene(top, top_f))
    assert len(candidates) and all(0.7 <= s["height_above_floor"] <= 1.3 for s in stats)
    assert all(s["height_selection_reason"] == "nominal_height_ineligible" for s in stats)
    assert all(s["nearest_camera_distance"] <= 1 + 1e-7 for s in stats)
    assert len(np.unique(candidates[:, [0, 2]], axis=0)) == len(candidates)


def test_severe_near_obstruction_is_ineligible_even_if_it_adds_coverage():
    v, f = floor()
    wall = np.array([[2.05, 0, 0], [2.05, 3, 0], [2.05, 3, 4], [2.05, 0, 4]])
    scene = mesh_scene(np.vstack([v, wall]), np.vstack([f, np.array([[4, 5, 6], [4, 6, 7]])]))
    points, grid = sample_floor(v, f, 0.1)
    policy = height_policy({"output_coordinate_units": "meters"}, [3])
    candidates, stats, report = adaptive_candidates(points, grid, np.array([[2, 1.5, 2]]), policy, scene)
    assert report["severe_obstruction_rejections"] > 0
    assert len(candidates) and all(s["horizontal_min_distance"] >= 0.10 for s in stats)
    assert all(s["horizontal_p10_distance"] >= 0.25 and s["view_band_near_fraction"] <= 0.5 for s in stats)


def test_lower_height_is_tried_when_nominal_fails_hard_clearance():
    v, f = floor()
    obstacle = np.array([[2.05, 1.49, 1.99], [2.05, 1.51, 1.99], [2.05, 1.51, 2.01], [2.05, 1.49, 2.01]])
    scene = mesh_scene(np.vstack([v, obstacle]), np.vstack([f, np.array([[4, 5, 6], [4, 6, 7]])]))
    points, grid = sample_floor(v, f, 0.1)
    policy = height_policy({"output_coordinate_units": "meters"}, [3])
    candidates, stats, _ = adaptive_candidates(points, grid, np.array([[2, 1.5, 2]]), policy, scene)
    nearest = np.argmin(np.linalg.norm(candidates[:, [0, 2]] - [2, 2], axis=1))
    assert np.linalg.norm(candidates[nearest, [0, 2]] - [2, 2]) < 0.02
    assert stats[nearest]["height_above_floor"] == 1.3
    assert stats[nearest]["height_selection_reason"] == "nominal_height_ineligible"


def test_height_does_not_lower_only_to_expand_acquisition_radius():
    v, f = floor(width=6, depth=6)
    points, grid = sample_floor(v, f, 0.1)
    policy = height_policy({"output_coordinate_units": "meters"}, [3])
    poses = np.array([[3, 1, 3]])
    candidates, stats, report = adaptive_candidates(points, grid, poses, policy, mesh_scene(v, f))
    assert report["radius_only_lowering_rejections"] > 0
    assert len(candidates) and all(s["height_above_floor"] == 1.5 for s in stats)


def test_lower_height_remains_available_when_nominal_crosses_ceiling():
    v, f = floor()
    top, top_f = floor(width=0.2, depth=0.2, y=1.35)
    top += [1.9, 0, 1.9]
    points, grid = sample_floor(v, f, 0.1)
    policy = height_policy({"output_coordinate_units": "meters"}, [2])
    scene = mesh_scene(np.vstack([v, top]), np.vstack([f, top_f + len(v)]))
    candidates, stats, _ = adaptive_candidates(points, grid, np.array([[2, 1.3, 2]]), policy, scene, mesh_scene(top, top_f))
    nearest = np.argmin(np.linalg.norm(candidates[:, [0, 2]] - [2, 2], axis=1))
    assert 0.7 <= stats[nearest]["height_above_floor"] <= 1.3
    assert stats[nearest]["height_selection_reason"] == "nominal_height_ineligible"


def test_uncalibrated_reduced_heights_keep_nominal_radius():
    policy = height_policy({}, [3, 7])
    assert policy["allowed_heights"] == pytest.approx(np.arange(1.5, 0.69, -0.2))
    assert policy["max_pose_distance"] == pytest.approx(0.99)


def test_slice_triangle_ownership_excludes_other_scans_and_lower_geometry(tmp_path):
    from generate_floor_level_meshes import _write_mesh
    v, f = floor()
    lower, lower_f = floor(y=-2)
    other, other_f = floor(y=5)
    vertices = np.vstack([v, lower, other])
    faces = np.vstack([f, lower_f + 4, other_f + 8])
    _write_mesh(tmp_path / "slice.ply", vertices, faces, np.full((12, 4), 255, np.uint8))
    level = {"sealed_mesh": "slice.ply", "greyscale_reference_plane_equation": [0, 1, 0, 0],
             "scans": [{"scan_id": "scan_0", "output_status": "generated_for_floor_level", "sealed": {"output_triangles": 4}},
                       {"scan_id": "scan_1", "output_status": "carried_forward", "sealed": {"output_triangles": 2}}]}
    floors = [{"plane_equation": [0, 1, 0, 0]}]
    targets, report = slice_targets(level, "scan_0", tmp_path, floors)
    assert len(targets) == 2 and np.all(targets[:, 1] == 0)
    assert report["excluded_below_floor_triangles"] == 2
    with pytest.raises(ValueError, match="Carried-forward"):
        slice_targets(level, "scan_1", tmp_path, floors)
    level["scans"][0]["sealed"]["output_triangles"] = 3
    with pytest.raises(ValueError, match="counts"):
        slice_targets(level, "scan_0", tmp_path, floors)


def test_slice_excludes_ceiling_just_below_own_floor_and_keeps_visible_floor(tmp_path):
    from generate_floor_level_meshes import _write_mesh
    own_floor, faces = floor(y=2)
    underside = own_floor - [0, 0.05, 0]
    _write_mesh(tmp_path / "slice.ply", np.vstack([own_floor, underside]), np.vstack([faces, faces + 4]), np.full((8, 4), 255, np.uint8))
    level = {"sealed_mesh": "slice.ply", "greyscale_reference_plane_equation": [0, 1, 0, 0],
             "scans": [{"scan_id": "scan_0", "output_status": "generated_for_floor_level", "sealed": {"output_triangles": 4}}]}
    targets, report = slice_targets(level, "scan_0", tmp_path, [{"plane_equation": [0, 1, 0, -2]}])
    assert len(targets) == 2 and np.all(targets[:, 1] == 2)
    assert report["excluded_below_floor_triangles"] == 2


def test_single_camera_is_preferred_when_it_can_meet_target():
    visibility = np.zeros((3, 100), bool)
    visibility[0, :60] = True
    visibility[1, :80] = True
    visibility[2, 60:] = True
    selected, report = select_points(visibility, [1, 0.4, 0.8])
    assert selected == [1] and report["coverage"] == 0.8
    assert report["stopping_reason"] == "single_camera_target_reached"


def test_dense_nearby_views_trigger_lowering_even_if_combined_coverage_reaches_target():
    candidates = np.array([[0, 1.5, 0], [1, 1.5, 0], [2, 1.5, 0]])
    visibility = np.zeros((3, 100), bool)
    for i in range(3):
        visibility[i, i * 30:(i + 1) * 30] = True
    selected, selection = select_points(visibility, [1, 1, 1])
    statistics = [{"view_band_near_fraction": 0.02}] * 3
    reason, cluster_count, single_coverage, _ = height_retry_reason(candidates, selected, visibility, statistics, selection, 1)
    assert reason == "clustered_views_with_low_individual_coverage"
    assert cluster_count == 3 and single_coverage == 0.3
    reason, _, _, _ = height_retry_reason(candidates * [10, 1, 1], selected, visibility, statistics, selection, 1)
    assert reason is None  # Widely separated rooms can legitimately need several views.


def test_height_search_replaces_nearby_weak_views_with_one_lower_view(monkeypatch):
    import generate_360_pictures as module
    called = []
    def candidates(*args):
        height = args[3]["allowed_heights"][0]
        called.append(height)
        count = 3 if height == 1.5 else 1
        xyz = np.column_stack((np.arange(count), np.full(count, height), np.zeros(count)))
        rows = [{"obstruction_quality": .8, "view_band_near_fraction": .2 if height == 1.5 else .02} for _ in range(count)]
        return xyz, rows, {}
    def visibility(scene, xyz, targets, scale):
        matrix = np.zeros((len(xyz), 100), bool)
        for row, point in enumerate(xyz):
            if point[1] == 1.5:
                index = int(point[0])
                matrix[row, index * 20:(index + 1) * 20] = True
            else:
                matrix[row, :80] = True
        return matrix
    monkeypatch.setattr(module, 'adaptive_candidates', candidates)
    monkeypatch.setattr(module, 'triangle_visibility', visibility)
    monkeypatch.setattr(module, 'candidate_source_quality', lambda scene, xyz, *args: np.ones(len(xyz)))
    plan = plan_floor_points(None, None, None, height_policy({'output_coordinate_units': 'meters'}, [3]), None, None, np.zeros((100, 3)), None, None, None)
    assert called == [1.5, 1.3]
    assert plan['height'] == 1.3 and plan['selected'] == [0]
    assert plan['height_search']['trials'][0]['lowering_reason'] == 'low_triangle_coverage'
    assert plan['height_search']['stopping_reason'] == 'clear_target_reached'


def test_height_search_exhausts_empty_heights_without_crashing(monkeypatch):
    import generate_360_pictures as module
    monkeypatch.setattr(module, 'adaptive_candidates', lambda *args: (np.empty((0, 3)), [], {}))
    monkeypatch.setattr(module, 'triangle_visibility', lambda *args: np.empty((0, 100), bool))
    monkeypatch.setattr(module, 'candidate_source_quality', lambda *args: np.array([]))
    plan = plan_floor_points(None, None, None, height_policy({'output_coordinate_units': 'meters'}, [3]), None, None, np.zeros((100, 3)), None, None, None)
    assert plan['selected'] == [] and plan['selection']['coverage'] == 0
    assert len(plan['height_search']['trials']) == 5
    assert plan['height_search']['trials'][-1]['height'] == 0.7
    assert plan['height_search']['stopping_reason'] == 'minimum_height_reached'


def test_best_below_target_height_gets_full_candidate_visibility_check(monkeypatch):
    import generate_360_pictures as module
    candidates = np.array([[0, 1.5, 0], [1, 1.5, 0], [2, 1.5, 0]], float)
    rows = np.zeros((3, 32_768), bool)
    rows[0, :6_500] = True
    rows[1, 6_500:13_000] = True
    rows[2, 13_000:19_500] = True
    calls = []

    monkeypatch.setattr(module, 'adaptive_candidates', lambda *args: (candidates, [
        {'obstruction_quality': .8, 'view_band_near_fraction': .02} for _ in candidates], {}))
    def visibility(scene, xyz, targets, scale):
        calls.append((len(xyz), len(targets)))
        if len(targets) == 32_768:
            return rows[:len(xyz)]
        full = np.zeros((len(xyz), len(targets)), bool)
        for row in range(len(xyz)):
            start = row * 6_500
            full[row, start:start + 6_500] = True
        return full
    monkeypatch.setattr(module, 'triangle_visibility', visibility)
    monkeypatch.setattr(module, 'candidate_source_quality', lambda scene, xyz, *args: np.ones(len(xyz)))

    plan = plan_floor_points(None, None, None,
        {'height': 1.5, 'distance_scale': 1, 'allowed_heights': [1.5]},
        None, None, np.zeros((32_769, 3)), None, None, None)

    assert (3, 32_769) in calls
    assert plan['exact_candidate_indices'] == [0, 1, 2]
    assert plan['selection']['attainable_scope'] == 'all_candidates_exact'
    assert plan['selection']['coverage'] == pytest.approx(19_500 / 32_769)


def test_height_search_keeps_earlier_view_if_lowering_worsens_obstruction(monkeypatch):
    import generate_360_pictures as module
    def candidates(*args):
        height = args[3]['allowed_heights'][0]
        return np.array([[0, height, 0]]), [{'obstruction_quality': .8, 'view_band_near_fraction': .2 if height == 1.5 else .3}], {}
    monkeypatch.setattr(module, 'adaptive_candidates', candidates)
    monkeypatch.setattr(module, 'triangle_visibility', lambda *args: np.array([[1] * 80 + [0] * 20], bool))
    monkeypatch.setattr(module, 'candidate_source_quality', lambda *args: np.ones(1))
    plan = plan_floor_points(None, None, None, height_policy({'output_coordinate_units': 'meters'}, [3]), None, None, np.zeros((100, 3)), None, None, None)
    assert plan['height'] == 1.5 and plan['selected'] == [0]
    assert len(plan['height_search']['trials']) == 5


def test_level_max_height_is_relative_to_its_floor(tmp_path):
    from generate_floor_level_meshes import _write_mesh
    v, f = floor(y=12)
    _write_mesh(tmp_path / "ceiling_group_0.ply", v, f, np.full((4, 4), 255, np.uint8))
    records = [{"plane_equation": [0, 1, 0, -10]}]
    ceilings = [{"source_mesh": "ceiling_group_0.ply"}]
    height, method = level_max_height(records, ceilings, tmp_path, v)
    assert height == 2
    assert height_policy({}, [height, 4])["height"] == 1


def test_horizontal_wall_obstruction_is_penalized():
    v, f = floor()
    wall = np.array([[2.2, 0, 0], [2.2, 3, 0], [2.2, 3, 4], [2.2, 0, 4]])
    wall_f = np.array([[0, 1, 2], [0, 2, 3]]) + len(v)
    scene = mesh_scene(np.vstack([v, wall]), np.vstack([f, wall_f]))
    points, grid = sample_floor(v, f, 0.1)
    candidates, stats, _ = generate_candidates(points, grid, np.array([[2, 1.5, 2]]), 1.5, 1, 1, scene)
    near = min(range(len(candidates)), key=lambda i: np.linalg.norm(candidates[i] - [2, 1.5, 2]))
    assert stats[near]["near_horizontal_fraction"] > 0.2
    assert stats[near]["horizontal_quality"] < 0.5
    assert np.all(candidates[:, 0] < 2.2)  # The source camera cannot see through the wall.


def test_triangle_coverage_accounts_for_wall_occlusion_and_targets_seventy_percent():
    v, f = floor()
    left, left_f = floor(width=2)
    right = left + [2, 0, 0]
    wall = np.array([[2, 0, 0], [2, 3, 0], [2, 3, 4], [2, 0, 4]])
    vertices = np.vstack([left, right, wall])
    faces = np.vstack([left_f, left_f + 4, [[8, 9, 10], [8, 10, 11]]])
    scene = mesh_scene(vertices, faces)
    visibility = triangle_visibility(scene, np.array([[1, 1.5, 2], [3, 1.5, 2]]), vertices[faces[:4]].mean(1), 1)
    assert visibility[0].mean() == pytest.approx(0.5)
    selected, report = select_points(visibility, [1, 1])
    assert len(selected) == 2 and report["coverage"] == 1 and report["target"] == 0.7
    selected, report = select_points(np.array([[1] * 7 + [0] * 3, [0] * 7 + [1] * 3], bool), [1, 1])
    assert selected == [0] and report["coverage"] == 0.7 and report["target_reached"]


def test_unattainable_coverage_is_not_renormalized():
    visibility = np.zeros((2, 100), bool)
    visibility[0, :40] = True
    visibility[1, 20:60] = True
    selected, report = select_points(visibility, [1, 1])
    assert len(selected) == 2
    assert report["coverage"] == report["attainable_coverage"] == 0.6
    assert not report["minimum_reached"]
    empty, report = select_points(np.empty((0, 100), bool), [])
    assert empty == [] and report["coverage"] == 0


def test_tiny_floor_fragments_do_not_demand_extra_panoramas():
    visibility = np.zeros((5, 1000), bool)
    visibility[0, :500] = True
    for i in range(1, 5):
        visibility[i] = visibility[0]
        visibility[i, 500 + i] = True
    selected, report = select_points(visibility, [1, 0.9, 0.9, 0.9, 0.9])
    assert selected == [0]
    assert report["coverage"] == 0.5
    assert report["stopping_reason"] == "marginal_gain_below_half_percent"


def test_useful_supplement_is_considered_before_stopping_on_tiny_best_score():
    visibility = np.zeros((3, 1000), bool)
    visibility[:, :500] = True
    visibility[1, 500:504] = True
    visibility[2, 504:510] = True
    selected, report = select_points(visibility, [1.0, 0.9, 0.5])
    assert selected == [0, 2]
    assert report["coverage"] == 0.506


def test_tour_links_follow_clear_paths():
    v, f = floor()
    scene = mesh_scene(v, f)
    points = np.array([[0.5, 1.5, 2], [1.5, 1.5, 2], [2.5, 1.5, 2], [3.5, 1.5, 2]])
    links, components = tour_links(scene, points, [0, 3], 1)
    assert components == 1
    assert len(links) == 1 and links[0]["path_length"] == 3
    assert len(links[0]["path_world"]) == 4


def test_tour_links_cannot_cross_a_floor_hole():
    v, f = floor(width=1)
    other = v + [1.5, 0, 0]
    scene = mesh_scene(np.vstack([v, other]), np.vstack([f, f + len(v)]))
    candidates = np.array([[0.9, 1.5, 2], [1.6, 1.5, 2]])
    links, components = tour_links(scene, candidates, [0, 1], 1, scene)
    assert components == 2 and links == []


def test_publication_refuses_unrelated_directory(tmp_path):
    target = tmp_path / "result"
    target.mkdir()
    (target / "keep.txt").write_text("original")
    staged = tmp_path / "staged"
    staged.mkdir()
    with pytest.raises(ValueError, match="unrelated"):
        publish(staged, target)
    assert (target / "keep.txt").read_text() == "original"


def test_aggregate_retains_successful_floors_when_another_fails(tmp_path, monkeypatch):
    import generate_360_pictures as module
    import torch

    alignment = tmp_path / "multi_room_alignment"
    plans = tmp_path / "artifacts" / "Generate Floor Plans"
    (alignment / "stats").mkdir(parents=True)
    (plans / "floor_level_meshes").mkdir(parents=True)
    (plans / "generate_floor_plans_manifest.json").write_text(json.dumps({"macro_step_status": "complete"}))
    (plans / "floor_level_metadata.json").write_text("{}")
    (plans / "floor_level_meshes" / "floor_level_meshes_manifest.json").write_text("{}")
    (alignment / "stats" / "aligned_scans_transforms.json").write_text(json.dumps({
        "scans": {"scan_0": {"transform_matrix": np.eye(4).tolist()}}}))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module, "process_scan", lambda *a: {"scan_id": "scan_0", "floors": [
        {"scan_id": "scan_0", "floor_level_name": "BASE", "status": "complete", "point_count": 2,
         "coverage": {"coverage": 0.8}, "triangle_count": 5},
        {"scan_id": "scan_0", "floor_level_name": "BASEa", "status": "failed", "error": "render failure"}]} )
    result = module.generate_360_pictures(alignment, tmp_path / "output", 256)
    assert result["macro_step_status"] == "failed"
    assert result["point_count"] == 2
    assert result["failures"][0]["floor_level_name"] == "BASEa"
