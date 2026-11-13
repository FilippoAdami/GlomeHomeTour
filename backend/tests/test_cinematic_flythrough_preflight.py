import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '08_artifacts_generation'))
from cinematic_flythrough_preflight import full_turns, ordered_checkpoints, paired_panel_route, photo_poses


def camera(number, scan, floor='BASE', kind='photo'):
    return {'number': number, 'scan': scan, 'floor': floor, 'kind': kind,
            'source_id': number, 'position': [number, 1, 0]}


def panel(index, a, b):
    return {'panel_index': index, 'source_instances': [{'scan_id': a}, {'scan_id': b}]}


def test_published_order_is_preserved_and_invalid_order_is_rejected():
    cameras = [camera(1, 'scan_0'), camera(2, 'scan_0', 'UPPER', 'panorama'), camera(3, 'scan_1')]
    assert ordered_checkpoints({'cameras': cameras}) == cameras
    with pytest.raises(ValueError, match='consecutive'):
        ordered_checkpoints({'cameras': cameras[::-1]})
    with pytest.raises(ValueError, match='split'):
        ordered_checkpoints({'cameras': [camera(1, 'scan_0'), camera(2, 'scan_1'), camera(3, 'scan_0')]})


def test_candidate_route_keeps_transit_scans_and_requires_paired_doors():
    panels = [panel(10, 'scan_0', 'scan_1'), panel(20, 'scan_0', 'scan_2')]
    assert paired_panel_route('scan_1', 'scan_2', panels) == {
        'scans': ['scan_1', 'scan_0', 'scan_2'], 'panel_indices': [10, 20]}
    with pytest.raises(ValueError, match='scan_3'):
        paired_panel_route('scan_1', 'scan_3', panels)


def test_panorama_turn_threshold_matches_cinematic_requirement():
    cameras = [camera(1, 'scan_0'), camera(2, 'scan_0'), camera(3, 'scan_0', kind='panorama'),
               camera(4, 'scan_0', 'BASEa', 'panorama'), camera(5, 'scan_1', kind='panorama')]
    assert full_turns(cameras) == {3: 240, 4: 240, 5: 240}


def test_photo_poses_come_from_selection_and_match_viewer_position(tmp_path):
    matrix = np.eye(4)
    matrix[:3, 3] = [1, 1, 0]
    selection = tmp_path / 'selected_images.json'
    selection.write_text(json.dumps({'images': [{'image_id': 1, 'camera_to_world': matrix.tolist()}]}))
    manifest = {'scans': [{'scan_id': 'scan_0', 'selected_images': str(selection)}]}
    assert photo_poses([camera(1, 'scan_0')], manifest)[1] == matrix.tolist()
    inconsistent = camera(1, 'scan_0')
    inconsistent['yaw'] = 0.0
    with pytest.raises(ValueError, match='orientation disagree'):
        photo_poses([inconsistent], manifest)
    matrix[0, 3] += 1
    selection.write_text(json.dumps({'images': [{'image_id': 1, 'camera_to_world': matrix.tolist()}]}))
    with pytest.raises(ValueError, match='checkpoint 1'):
        photo_poses([camera(1, 'scan_0')], manifest)
