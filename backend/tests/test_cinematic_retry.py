from pathlib import Path
import copy
import sys

import numpy as np
import open3d as o3d
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '08_artifacts_generation'))
import generate_cinematic_flythrough as generator
from cinematic_path_planning import segment_clear


@pytest.mark.parametrize(('linear_fallback', 'reject_framing'), [(False, False), (True, False), (True, True)])
def test_retry_revalidates_alternative_after_smoothed_path_hits_obstacle(monkeypatch, linear_fallback, reject_framing):
    floor = o3d.geometry.TriangleMesh.create_box(width=6, height=.02, depth=6)
    floor.translate([-3, -.02, -3])
    wall = o3d.geometry.TriangleMesh.create_box(width=.2, height=2, depth=1)
    wall.translate([-.1, 0, -.5])
    def scene(mesh):
        result = o3d.t.geometry.RaycastingScene()
        result.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        return result
    geometry, floor_scene = scene(floor + wall), scene(floor)
    poses = np.tile(np.eye(4), (2, 1, 1))
    poses[:, :3, 3] = [[-1, 1, 0], [1, 1, 0]]
    checkpoints = [{'number': i + 1, 'kind': 'photo', 'scan': 'scan_0', 'floor': 'BASE',
                    'position': pose[:3, 3].tolist(), 'camera_to_world': pose.tolist()}
                   for i, pose in enumerate(poses)]
    def piece(z):
        return {'kind': 'local', 'scan': 'scan_0',
                'points': [[-1, 1, 0], [-.5, 1, z], [.5, 1, z], [1, 1, 0]]}
    routes = [piece(1), piece(-1)]
    for route in routes:
        assert all(segment_clear(geometry, floor_scene, a, b)
                   for a, b in zip(route['points'], route['points'][1:]))
    plan = {'checkpoints': checkpoints, 'edges': [{
        'from_checkpoint': 1, 'to_checkpoint': 2, 'pieces': [routes[0]], 'route_quality_score': 1.,
        'route_alternatives': [{'pieces': [routes[1]], 'route_quality_score': 2.}]}]}
    original = copy.deepcopy(plan)
    sampled_choices = []
    def sample(candidate, *_, interpolation):
        choice = candidate['edges'][0]['selected_route_alternative']
        sampled_choices.append((choice, interpolation))
        # Model a smoothing failure on the first spatially safe route.
        if choice == 0 and not (linear_fallback and interpolation == 'linear'):
            waypoints = np.array([poses[0, :3, 3], poses[1, :3, 3]])
        else:
            waypoints = np.array(routes[choice]['points'])
        lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        cumulative = np.r_[0, np.cumsum(lengths)]
        distances = np.linspace(0, cumulative[-1], 121)
        xyz = np.column_stack([np.interp(distances, cumulative, waypoints[:, axis]) for axis in range(3)])
        sampled = np.tile(np.eye(4), (len(xyz), 1, 1))
        sampled[:, :3, 3] = xyz
        candidate['checkpoints'][0]['passage_position'] = xyz[0].tolist()
        return {'camera_to_world': sampled, 'scans': ['scan_0'] * len(xyz), 'fps': 30,
                'checkpoints': [{'number': 1, 'frame': 0}, {'number': 2, 'frame': len(xyz) - 1}],
                'door_crossings': []}
    monkeypatch.setattr(generator, 'sample_drone_trajectory', sample)
    monkeypatch.setattr(generator, 'append_overhead', lambda *_: None)
    if reject_framing:
        def invalid_framing(flight, *_):
            altered = flight['camera_to_world'].copy()
            altered[0, 0, 3] += .1
            return altered, [{'checkpoint': 1}]
        monkeypatch.setattr(generator, 'refine_panorama_framing', invalid_framing)
    selected, flight, validation, search = generator.validated_trajectory(
        plan, geometry, np.asarray((floor + wall).vertices), np.asarray((floor + wall).triangles),
        {('scan_0', 'BASE'): floor_scene}, None)
    expected = [(0, 'pchip'), (0, 'linear')] + ([] if linear_fallback else [(1, 'pchip')])
    assert sampled_choices == expected
    assert search['attempt_count'] == len(expected)
    assert search['failed_attempts'][0]['stage'] == 'validation'
    assert selected['edges'][0]['selected_route_alternative'] == (0 if linear_fallback else 1)
    assert search['interpolation'] == ('linear' if linear_fallback else 'pchip')
    assert search['framing']['status'] == ('rejected' if reject_framing else 'unchanged')
    np.testing.assert_allclose(flight['camera_to_world'][0], poses[0], atol=1e-12, rtol=0)
    assert set(validation['floor']) == {'BASE'}
    assert np.abs(flight['camera_to_world'][:, 2, 3]).max() > .9
    assert plan == original


def test_retry_limit_never_returns_an_unvalidated_candidate(monkeypatch):
    plan = {'edges': [{'pieces': [], 'route_quality_score': 0.,
                       'route_alternatives': [{'pieces': [], 'route_quality_score': i + 1.}
                                              for i in range(4)]}]}
    monkeypatch.setattr(generator, 'sample_drone_trajectory',
                        lambda *_, **__: (_ for _ in ()).throw(ValueError('blocked after smoothing')))
    with pytest.raises(ValueError, match='2 route attempts; search limit 2 reached'):
        generator.validated_trajectory(plan, None, None, None, {}, None, max_attempts=2)


def test_exhausted_retained_routes_reports_spatial_candidate_cap(monkeypatch):
    plan = {'edges': [{'pieces': [], 'route_quality_score': 0., 'route_search_capped': True}]}
    monkeypatch.setattr(generator, 'sample_drone_trajectory',
                        lambda *_, **__: (_ for _ in ()).throw(ValueError('blocked after smoothing')))
    with pytest.raises(ValueError, match='spatial route search was capped'):
        generator.validated_trajectory(plan, None, None, None, {}, None)
