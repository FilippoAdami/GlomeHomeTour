import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '07_alignment'))
from step_alignment import (cleanup_temporary_outputs, reusable_trajectory_pairings,
                            trajectory_pairing_input_hash, trajectory_pairing_records)
import recover_trajectory_pairings as recovery


def test_pairing_metadata_keeps_source_identities():
    a = {'source_class_id': 22, 'source_instance_id': 0, 'instance_idx': 1, 'description': 'a'}
    b = {'source_class_id': 5, 'source_instance_id': 5, 'instance_idx': 6, 'description': 'b'}
    record = trajectory_pairing_records([SimpleNamespace(scan_from='scan_0', scan_to='scan_1',
                                          door_from=a, door_to=b, yaw_deg=90)])[0]
    assert record == {'scan_from': 'scan_0', 'scan_to': 'scan_1', 'door_from': a,
                      'door_to': b, 'relative_yaw_deg': 90.0}


def test_cleanup_preserves_report_but_removes_temporary_outputs(tmp_path):
    stats = tmp_path / 'stats'
    stats.mkdir()
    report = stats / 'multi_room_alignment_report.json'
    report.write_text('{"saved": true}')
    (stats / 'aligned_objects_manifest.json').write_text('{}')
    (stats / 'wall_plane_snap_report.json').write_text('{}')
    cleanup_temporary_outputs(tmp_path)
    assert json.loads(report.read_text()) == {'saved': True}
    assert not (stats / 'aligned_objects_manifest.json').exists()
    assert not (stats / 'wall_plane_snap_report.json').exists()


def test_pairings_reused_only_when_transforms_match_and_metadata_is_available(tmp_path):
    path = tmp_path / 'report.json'
    matrix = np.eye(4)
    records = [{'scan_from': 'scan_0'}]
    report = {'transforms': {'scan_0': matrix.tolist()},
              'alignment_metrics': {'trajectory_door_pairs_status': 'computed', 'trajectory_door_pairs': records,
                                    'trajectory_door_pairs_input_sha256': 'inputs-a'}}
    path.write_text(json.dumps(report))
    assert reusable_trajectory_pairings(path, {'scan_0': matrix}, 'inputs-a') == records
    assert reusable_trajectory_pairings(path, {'scan_0': matrix}, 'changed-inputs') is None
    changed = matrix.copy()
    changed[0, 3] = 0.01
    assert reusable_trajectory_pairings(path, {'scan_0': changed}, 'inputs-a') is None
    report['alignment_metrics']['trajectory_door_pairs_status'] = 'not_computed'
    path.write_text(json.dumps(report))
    assert reusable_trajectory_pairings(path, {'scan_0': matrix}, 'inputs-a') is None
    path.write_text('{')
    assert reusable_trajectory_pairings(path, {'scan_0': matrix}, 'inputs-a') is None


def test_pairing_input_hash_changes_with_trajectory_or_door_geometry():
    door = {'instance_idx': 1, 'center': np.zeros(3), 'normal': np.array([1, 0, 0]), 'description': 'door'}
    scan = SimpleNamespace(door_instances=[door], trajectory_poses=None,
                           transition_poses=np.zeros((2, 3)), active_poses=np.zeros((2, 3)))
    original = trajectory_pairing_input_hash({'scan_0': scan})
    scan.active_poses[0, 0] = 1
    assert trajectory_pairing_input_hash({'scan_0': scan}) != original
    scan.active_poses[0, 0] = 0
    door['center'][0] = 1
    assert trajectory_pairing_input_hash({'scan_0': scan}) != original


def test_recovery_refuses_a_split_and_restores_loader_callbacks(tmp_path, monkeypatch):
    stats = tmp_path / 'multi_room_alignment/stats'
    stats.mkdir(parents=True)
    (stats / 'aligned_scans_transforms.json').write_text('{"scans": {}}')
    (stats / 'aligned_door_panels.json').write_text('{"panels": []}')
    original_split = recovery.scan_loader._publish_door_split
    original_metadata = recovery.scan_loader._write_door_scene_metadata
    def simulated_loader(workspace):
        recovery.scan_loader._write_door_scene_metadata(None, None)
        recovery.scan_loader._publish_door_split(None)
    monkeypatch.setattr(recovery.scan_loader, 'ScanDataset', simulated_loader)
    with pytest.raises(ValueError, match='mutating Phase 07 door split'):
        recovery.recover(tmp_path)
    assert recovery.scan_loader._publish_door_split is original_split
    assert recovery.scan_loader._write_door_scene_metadata is original_metadata
