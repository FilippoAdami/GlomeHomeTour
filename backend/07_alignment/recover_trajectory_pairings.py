#!/usr/bin/env python3
"""Recompute missing Phase 07 doorway identities without writing segmentation data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import scan_loader
from door_alignment_engine import DoorAlignmentEngine
from step_alignment import trajectory_pairing_records


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def recover(workspace):
    workspace = Path(workspace).resolve()
    alignment = workspace / 'multi_room_alignment'
    transform_path = alignment / 'stats/aligned_scans_transforms.json'
    panel_path = alignment / 'stats/aligned_door_panels.json'
    saved = json.loads(transform_path.read_text())['scans']
    panels = json.loads(panel_path.read_text())['panels']
    original_metadata = scan_loader._write_door_scene_metadata
    original_split = scan_loader._publish_door_split
    class WriteRequired(BaseException):
        pass
    def refuse_split(*args, **kwargs):
        raise WriteRequired('Source data still requires a mutating Phase 07 door split')
    try:
        scan_loader._write_door_scene_metadata = lambda *args, **kwargs: None
        scan_loader._publish_door_split = refuse_split
        try:
            scans = scan_loader.ScanDataset(workspace).scans
        except WriteRequired as error:
            raise ValueError(str(error)) from error
        pairings = DoorAlignmentEngine().pair_doors_via_trajectories(scans)
    finally:
        scan_loader._write_door_scene_metadata = original_metadata
        scan_loader._publish_door_split = original_split
    if set(scans) != set(saved):
        raise ValueError('Discovered scans differ from saved Phase 07 transforms')
    records = trajectory_pairing_records(pairings)
    for pair, row in zip(pairings, records):
        rotation_from = np.asarray(saved[pair.scan_from]['transform_matrix'])[:3, :3]
        rotation_to = np.asarray(saved[pair.scan_to]['transform_matrix'])[:3, :3]
        if not np.allclose(rotation_from.T @ rotation_to, pair.relative_rotation, atol=1e-6, rtol=0):
            raise ValueError(f'Recovered relative rotation differs from saved transform: {pair.description}')
        required = {(pair.scan_from, pair.door_from.get('source_class_id'), pair.door_from.get('source_instance_id')),
                    (pair.scan_to, pair.door_to.get('source_class_id'), pair.door_to.get('source_instance_id'))}
        matches = [panel['panel_index'] for panel in panels if required <= {
            (item['scan_id'], item['class_id'], item['instance_id']) for item in panel['source_instances']}]
        if len(required) != 2 or any(item[1] is None or item[2] is None for item in required) or len(matches) != 1:
            raise ValueError(f'No unique aligned source panel for {pair.description}: {matches}')
        row['matching_aligned_panel_index'] = matches[0]
        row['saved_relative_rotation_matches'] = True
    paths = {transform_path, panel_path}
    for scan in scans.values():
        paths.update(p for name in ('trajectory_csv_path', 'transforms_json_path', 'colmap_images_txt_path',
                                    'door_instance_metadata_path', 'door_class_metadata_path',
                                    'triangle_labels_path', 'sealed_mesh_path', 'regularized_complete_mesh_path')
                     if (p := getattr(scan.info, name, None)) and p.is_file())
        paths.update((scan.info.scan_dir / 'segmentation/3d_masks').glob('class_*/instance_*.ply'))
    return {'status': 'reconstructed_from_current_inputs_not_original_report',
            'method': 'Phase 07 pairing on current inputs, with segmentation writes disabled; compare saved relative rotations and both panel source identities',
            'pairs': records, 'input_sha256': {str(p): digest(p) for p in sorted(paths)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = recover(args.workspace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as destination:
        json.dump(report, destination, indent=2)
        destination.write('\n')
    print(f'Reconstructed {len(report["pairs"])} pairings: {args.output}')


if __name__ == '__main__':
    main()
