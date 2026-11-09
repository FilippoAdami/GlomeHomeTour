import sys
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1] / '08_artifacts_generation'
sys.path.insert(0, str(MODULE_DIR))
from generate_floorplan_instance_overlays import (  # noqa: E402
    _fit_inside_scan,
    _floor_coverage_fraction,
    _load_phase07_panels,
    _load_candidates,
    _nearest_wall_rotation,
    _overlaps_xz,
    _resolve_overlaps,
    _symbol_rotation,
    _triangle_density_rotation,
)


def test_only_finally_assigned_instances_with_floor_metadata_become_candidates():
    associations = [
        {
            'scan_id': 'scan_0', 'source_class_id': 1, 'instance_id': 2,
            'category_id': 22, 'category': 'dishwasher',
            'floor_measurement_metadata': {
                'assigned_floor_level': 'BASE', 'floor_plane_id': 'p0',
                'robust_bbox_min_xyz': [0, 0, 0],
                'robust_bbox_max_xyz': [.3, 1, .3],
            },
        },
        {
            'scan_id': 'scan_0', 'source_class_id': 2, 'instance_id': 0,
            'category_id': None,
            'floor_measurement_metadata': {
                'assigned_floor_level': 'BASE', 'floor_plane_id': 'p0',
                'robust_bbox_min_xyz': [1, 0, 1],
                'robust_bbox_max_xyz': [2, 1, 2],
            },
        },
    ]
    levels = [{'floor_level_name': 'BASE'}]
    planes = {'p0': {'scan_level_name': 'BASE'}}

    candidates = _load_candidates(associations, levels, planes)

    assert len(candidates) == 1
    assert candidates[0]['category_id'] == 22
    assert candidates[0]['bbox_max_xyz'][0] == .3


def test_any_positive_area_footprint_intersection_counts_as_overlap():
    a = {'bbox_min_xyz': [0, 0, 0], 'bbox_max_xyz': [1, 1, 1]}
    partial = {'bbox_min_xyz': [.99, 0, .4], 'bbox_max_xyz': [2, 1, 1.2]}
    edge_touch = {'bbox_min_xyz': [1, 0, .2], 'bbox_max_xyz': [2, 1, .8]}

    assert _overlaps_xz(a, partial)
    assert not _overlaps_xz(a, edge_touch)


def test_lower_instance_wins_any_partial_overlap():
    lower = {
        'scan_id': 'scan_0', 'source_class_id': 3, 'instance_id': 0,
        'category': 'table', 'bbox_min_xyz': [0, .1, 0], 'bbox_max_xyz': [1, .8, 1],
    }
    higher = {
        'scan_id': 'scan_0', 'source_class_id': 4, 'instance_id': 0,
        'category': 'chair', 'bbox_min_xyz': [.9, .9, .5], 'bbox_max_xyz': [1.5, 1.5, 1.5],
    }
    separate = {
        'scan_id': 'scan_0', 'source_class_id': 5, 'instance_id': 0,
        'category': 'wc', 'bbox_min_xyz': [2, .2, 2], 'bbox_max_xyz': [3, 1, 3],
    }

    displayed, hidden = _resolve_overlaps([higher, separate, lower])

    assert [item['category'] for item in displayed] == ['table', 'wc']
    assert hidden[0]['category'] == 'chair'
    assert hidden[0]['hidden_by']['source_class_id'] == 3


def test_small_overlap_moves_instance_inside_scan_outline():
    grid = {'x_min': 0.0, 'z_max': 1.0, 'pixel_size': .01}
    footprint = np.zeros((100, 100), dtype=bool)
    footprint[10:90, 10:90] = True
    lower = {
        'scan_id': 'scan_0', 'source_class_id': 3, 'instance_id': 0,
        'category': 'table', 'category_id': 11,
        'bbox_min_xyz': [.30, .1, .30], 'bbox_max_xyz': [.50, .8, .50],
    }
    higher = {
        'scan_id': 'scan_0', 'source_class_id': 4, 'instance_id': 0,
        'category': 'chair', 'category_id': 13,
        'bbox_min_xyz': [.495, .9, .35], 'bbox_max_xyz': [.65, 1.5, .55],
    }

    displayed, hidden = _resolve_overlaps(
        [higher, lower], {'scan_0': footprint}, grid,
    )

    assert not hidden
    assert len(displayed) == 2
    assert displayed[1]['placement_offset_xz_m'] != [0.0, 0.0]
    assert not _overlaps_xz(displayed[0], displayed[1])


def test_outside_instance_moves_only_when_small_translation_fits_contour():
    grid = {'x_min': 0.0, 'z_max': 1.0, 'pixel_size': .01}
    footprint = np.zeros((100, 100), dtype=bool)
    footprint[10:90, 10:90] = True
    candidate = {
        'bbox_min_xyz': [.05, 0, .35],
        'bbox_max_xyz': [.25, 1, .55],
    }

    moved = _fit_inside_scan(candidate, footprint, grid)

    assert moved is not None
    assert moved['placement_offset_xz_m'][0] > 0
    assert _fit_inside_scan({**candidate, 'bbox_min_xyz': [-.4, 0, .3],
                             'bbox_max_xyz': [-.2, 1, .5]}, footprint, grid) is None


def test_semantic_back_edge_faces_nearest_wall():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    candidate = {
        'bbox_min_xyz': [.45, 0, .70],
        'bbox_max_xyz': [.55, 1, .80],
    }
    grid = {'x_min': 0, 'z_max': 1, 'pixel_size': .01}

    assert _nearest_wall_rotation(candidate, mask, grid) == 0


def test_chair_back_edge_uses_the_longer_footprint_dimension():
    chair = {
        'category_id': 13,
        'bbox_min_xyz': [0, 0, 0],
        'bbox_max_xyz': [.4, 1, .8],
    }

    assert _symbol_rotation(chair, None, {}) == 90


def test_phase07_regularized_panels_are_loaded_as_floorplan_geometry(tmp_path):
    stats = tmp_path / 'stats'
    stats.mkdir()
    (stats / 'aligned_door_panels.json').write_text(
        '{"panels":[{"panel_index":3,"scan_id":"scan_0",'
        '"floor_elevation_m":0.1,"panel_bounds_xyz_m":'
        '[[1,0.1,2],[1.1,2.2,3]],"source_instances":[]}]}'
    )
    (stats / 'aligned_window_panels.json').write_text(
        '{"panels":[{"panel_index":1,"floor_elevation_m":0.1,'
        '"orientation":"normal_z","panel_center_xyz_m":[4,1,5],'
        '"panel_dimensions_m":{"width":1.2,"depth":0.1}}]}'
    )
    metadata = {'floor_planes': [
        {'scan_id': 'scan_0', 'floor_level_name': 'BASE', 'scan_level_name': 'BASE',
         'mean_elevation_m': .1},
        {'scan_id': 'scan_0', 'floor_level_name': 'BASEa', 'scan_level_name': 'BASE',
         'mean_elevation_m': 2.3},
    ]}

    panels = _load_phase07_panels(tmp_path, [
        {'floor_level_name': 'BASE'}, {'floor_level_name': 'BASEa'},
    ], metadata)

    assert [(p['category'], p['floor_level_name']) for p in panels] == [
        ('door', 'BASE'), ('window', 'BASE'),
    ]
    assert panels[0]['bbox_min_xyz'][0:3:2] == [1.0, 2.0]
    assert panels[1]['bbox_min_xyz'][0:3:2] == [3.4, 4.95]


def test_stair_arrow_uses_triangle_density_height_gradient():
    rows = []
    for index, z in enumerate(np.linspace(-1, 1, 8)):
        rows.extend([[0, z + 2, z]] * (index + 1))
    assert _triangle_density_rotation(np.asarray(rows, dtype=float)) == 0
    assert _triangle_density_rotation(np.asarray(rows, dtype=float) * [1, 1, -1]) == 180


def test_rug_is_not_suppressed_or_used_as_an_overlap_blocker():
    rug = {
        'scan_id': 'scan_0', 'source_class_id': 1, 'instance_id': 0,
        'category': 'rug', 'category_id': 14,
        'bbox_min_xyz': [0, 0, 0], 'bbox_max_xyz': [2, 1, 2],
    }
    chair = {
        'scan_id': 'scan_0', 'source_class_id': 2, 'instance_id': 0,
        'category': 'chair', 'category_id': 13,
        'bbox_min_xyz': [.5, 1, .5], 'bbox_max_xyz': [1, 2, 1],
    }
    displayed, hidden = _resolve_overlaps([chair, rug])

    assert not hidden
    assert {item['category'] for item in displayed} == {'rug', 'chair'}
    assert _overlaps_xz(rug, chair)


def test_lower_floor_coverage_is_measured_for_the_whole_symbol_footprint():
    grid = {'x_min': 0.0, 'z_max': 1.0, 'pixel_size': .25}
    candidate = {
        'bbox_min_xyz': [.25, 0, .25],
        'bbox_max_xyz': [.5, 1, .5],
    }
    floor_covered = np.zeros((4, 4), dtype=bool)
    floor_covered[2, 1] = True

    assert _floor_coverage_fraction(candidate, floor_covered, grid, 4) == 1.0
    assert _floor_coverage_fraction(candidate, np.zeros_like(floor_covered), grid, 4) == 0.0
    outside = {
        'bbox_min_xyz': [-1, 0, .25],
        'bbox_max_xyz': [-.5, 1, .5],
    }
    assert _floor_coverage_fraction(outside, np.ones_like(floor_covered), grid, 4) == 1.0
