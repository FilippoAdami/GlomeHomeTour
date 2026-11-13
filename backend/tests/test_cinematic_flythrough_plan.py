from pathlib import Path
import json
import sys

import numpy as np
import open3d as o3d
import pytest
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '08_artifacts_generation'))
from cinematic_flythrough_plan import DoorSurfaceAttributor, choose_scan_route, match_pair_panels, route_quality, scan_route
from cinematic_path_planning import validate_door_crossing


def test_panel_identity_and_transit_revisit():
    pair = lambda a, b, x, y: {'scan_from': a, 'scan_to': b,
                              'door_from': {'source_class_id': 1, 'source_instance_id': x},
                              'door_to': {'source_class_id': 1, 'source_instance_id': y}}
    pairs = [pair('scan_0', 'scan_1', 0, 1), pair('scan_0', 'scan_2', 2, 3)]
    panels = [{'panel_index': 10, 'source_instances': [{'scan_id': 'scan_0', 'class_id': 1, 'instance_id': 0},
                                                      {'scan_id': 'scan_1', 'class_id': 1, 'instance_id': 1}]},
              {'panel_index': 20, 'source_instances': [{'scan_id': 'scan_0', 'class_id': 1, 'instance_id': 2},
                                                      {'scan_id': 'scan_2', 'class_id': 1, 'instance_id': 3}]}]
    graph = match_pair_panels(pairs, panels)
    assert [(a, b, p['panel_index']) for a, b, p in scan_route('scan_1', 'scan_2', graph)] == [
        ('scan_1', 'scan_0', 10), ('scan_0', 'scan_2', 20)]
    assert scan_route('scan_0', 'scan_0', graph) == []
    with pytest.raises(ValueError, match='No trajectory-paired'):
        scan_route('scan_1', 'scan_3', graph)
    with pytest.raises(ValueError, match='Expected one aligned panel'):
        match_pair_panels(pairs, panels[:1])


def test_infeasible_direct_door_uses_alternative_route():
    graph = [('a', 'b', {'panel_index': 1}), ('a', 'c', {'panel_index': 2}),
             ('c', 'b', {'panel_index': 3})]
    tried = []
    def build(route):
        doors = [panel['panel_index'] for _, _, panel in route]
        tried.append(doors)
        if doors == [1]:
            raise ValueError('door 1 is blocked')
        return doors
    selected, errors = choose_scan_route('a', 'b', graph, build)
    assert selected == [2, 3]
    assert tried == [[1], [2, 3]]
    assert errors == ['door 1 is blocked']


def test_blocked_door_geometry_uses_second_paired_door():
    doors = []
    for z in (-1., 1.):
        door = o3d.geometry.TriangleMesh.create_box(width=.1, height=2, depth=.8)
        door.translate([-.05, 0, z - .4])
        doors.append(door)
    obstacle = o3d.geometry.TriangleMesh.create_box(width=.05, height=.5, depth=.3)
    obstacle.translate([.25, .75, -1.15])
    mesh = doors[0] + doors[1] + obstacle
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    panels = [{'panel_index': i, 'panel_center_xyz_m': [0, 1, z], 'panel_plane_normal_xyz': [1, 0, 0],
               'panel_dimensions_m': {'depth': .1, 'width': .8, 'height': 2.}}
              for i, z in enumerate((-1., 1.))]
    graph = [('a', 'b', panel) for panel in panels]
    def build(route):
        panel = route[0][2]
        z = panel['panel_center_xyz_m'][2]
        return validate_door_crossing(scene, np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                                      panel, [-.4, 1, z], [.4, 1, z])
    crossing, errors = choose_scan_route('a', 'b', graph, build)
    assert crossing['panel_index'] == 1
    assert len(errors) == 1


def test_route_quality_prefers_clearer_feasible_path():
    floor = o3d.geometry.TriangleMesh.create_box(width=4, height=.02, depth=4)
    floor.translate([-2, -.02, -2])
    wall = o3d.geometry.TriangleMesh.create_box(width=3, height=2, depth=.1)
    wall.translate([-1.5, 0, .15])
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(floor + wall))
    paths = [
        [{'kind': 'local', 'scan': 'a', 'points': [[-1, 1, 0], [1, 1, 0]]}],
        [{'kind': 'local', 'scan': 'a', 'points': [[-1, 1, 1], [1, 1, 1]]}],
    ]
    assert route_quality(paths[1], scene) < route_quality(paths[0], scene)
    detour = [{'kind': 'local', 'scan': 'a', 'points': [[-1, 1, 1], [0, 1, 1.5], [1, 1, 1]]}]
    assert route_quality(paths[1], scene) < route_quality(detour, scene)
    door = [{'kind': 'door', 'points': [[-1, 1, 1], [1, 1, 1]]}]
    assert route_quality(paths[1], scene) < route_quality(door, scene)
    graph = [('a', 'b', {'panel_index': i}) for i in range(2)]
    candidates = []
    selected, notes = choose_scan_route(
        'a', 'b', graph, lambda route: paths[route[0][2]['panel_index']],
        lambda pieces: route_quality(pieces, scene), candidates)
    assert selected == paths[1]
    assert [row[1] for row in sorted(candidates, key=lambda row: row[0])] == [paths[1], paths[0]]
    assert 'Compared 2 feasible routes' in notes[-1]


def test_route_search_limit_keeps_already_verified_candidate():
    graph = [('a', 'b', {'panel_index': i}) for i in range(2)]
    def build(route):
        if route[0][2]['panel_index']:
            raise RuntimeError('Door candidate search limit reached')
        return 'safe'
    selected, notes = choose_scan_route('a', 'b', graph, build, lambda result: 1)
    assert selected == 'safe'
    assert any('search limit' in note for note in notes)


def test_exhausted_route_budget_is_reported_as_a_limit():
    graph = [('a', 'b', {'panel_index': i}) for i in range(65)]
    def blocked(route):
        raise ValueError('blocked')
    with pytest.raises(ValueError, match='search limit reached'):
        choose_scan_route('a', 'b', graph, blocked)


@pytest.mark.parametrize(('size', 'gap_center', 'method'),
                         [(8., 3.2, 'scan_refined_grid'), (50., 1.2, 'local_refined_grid')])
def test_planner_refines_locally_and_retains_scan_wide_fallback(tmp_path, monkeypatch, size, gap_center, method):
    import cinematic_flythrough_plan as planner
    half = size / 2
    floor = o3d.geometry.TriangleMesh.create_box(width=size, height=.02, depth=size)
    floor.translate([-half, -.02, -half])
    lower = o3d.geometry.TriangleMesh.create_box(width=.12, height=3, depth=half + gap_center - .135)
    lower.translate([-.06, 0, -half])
    upper = o3d.geometry.TriangleMesh.create_box(width=.12, height=3, depth=half - gap_center - .135)
    upper.translate([-.06, 0, gap_center + .135])
    mesh = floor + lower + upper
    floor_scene = o3d.t.geometry.RaycastingScene()
    floor_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(floor))
    cameras = [{'number': i + 1, 'kind': 'photo', 'scan': 'scan_0', 'floor': 'BASE',
                'position': [x, 1., 0.]} for i, x in enumerate((-1., 1.))]
    poses = {}
    for camera in cameras:
        pose = np.eye(4)
        pose[:3, 3] = camera['position']
        poses[camera['number']] = pose.tolist()
    for directory in ('artifacts/Generate Interactive Viewer', 'artifacts/Extract Best Views',
                      'artifacts/Generate Floor Plans', 'multi_room_alignment/stats'):
        (tmp_path / directory).mkdir(parents=True)
    (tmp_path / 'artifacts/Generate Interactive Viewer/tour.json').write_text('{}')
    (tmp_path / 'artifacts/Extract Best Views/extract_best_views_manifest.json').write_text('{}')
    (tmp_path / 'artifacts/Generate Floor Plans/floor_level_metadata.json').write_text(
        json.dumps({'floor_planes': [{'scan_id': 'scan_0'}]}))
    (tmp_path / 'multi_room_alignment/stats/aligned_door_panels.json').write_text(json.dumps({'panels': []}))
    (tmp_path / 'multi_room_alignment/stats/aligned_scans_transforms.json').write_text(json.dumps({'scans': {}}))
    monkeypatch.setattr(planner, 'ordered_checkpoints', lambda _: cameras)
    monkeypatch.setattr(planner, 'photo_poses', lambda *_: poses)
    monkeypatch.setattr(planner, 'full_turns', lambda *_: {})
    monkeypatch.setattr(planner, '_read_mesh',
                        lambda _: (np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.triangles), None))
    monkeypatch.setattr(planner, 'floor_geometry', lambda *_: (floor_scene, np.asarray(floor.vertices)))
    plan = planner.build_plan(tmp_path, [])
    piece = plan['edges'][0]['pieces'][0]
    assert piece['routing']['method'] == method
    if method == 'scan_refined_grid':
        assert 'Local fine-grid search exhausted' in piece['routing']['local_failure']
    else:
        assert 'Grid resource limit exceeded' in piece['routing']['coarse_failure']
    assert max(point[2] for point in piece['points']) > gap_center - .2


def test_door_crossing_surface_requires_matching_source_class(tmp_path):
    scan = tmp_path / 'scan_0'
    (scan / 'segmentation').mkdir(parents=True)
    stats = tmp_path / 'multi_room_alignment/stats'
    stats.mkdir(parents=True)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
    source = scan / 'mesh.ply'
    assert o3d.io.write_triangle_mesh(str(source), mesh)
    vertices = np.array([(0, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    faces = np.empty(1, dtype=[('vertex_indices', 'O'), ('class_id', 'i4'), ('instance_id', 'i4')])
    faces[0] = (np.array([0, 1, 2], dtype=np.int32), 7, 1)
    PlyData([PlyElement.describe(vertices, 'vertex'), PlyElement.describe(faces, 'face')]).write(
        scan / 'segmentation/mesh_segmented.ply')
    (stats / 'aligned_scans_transforms.json').write_text(json.dumps({'scans': {'scan_0': {
        'mesh_path': str(source), 'transform_matrix': np.eye(4).tolist()}}}))
    attributor = DoorSurfaceAttributor(tmp_path)
    panel = {'panel_index': 3, 'source_instances': [{'scan_id': 'scan_0', 'class_id': 7}]}
    match = attributor.attribute(panel, [[0, .2, .2]])
    assert match[0][0]['class_id'] == 7
    panel['source_instances'][0]['class_id'] = 8
    with pytest.raises(ValueError, match='without matching source door-class'):
        attributor.attribute(panel, [[0, .2, .2]])
