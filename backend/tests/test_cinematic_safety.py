import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '08_artifacts_generation'))
from cinematic_flythrough_preflight import digest
from cinematic_rendering import GaussianRenderer
from generate_cinematic_flythrough import metric_scale_status, preview, verify_inputs, write_frame


def test_metric_scale_must_be_available_and_already_applied(tmp_path):
    stats = tmp_path / 'multi_room_alignment/stats'
    stats.mkdir(parents=True)
    (stats / 'aligned_scans_transforms.json').write_text(json.dumps({'scans': {'scan_0': {}}}))
    scan = tmp_path / 'scan_0'
    scan.mkdir()
    with pytest.raises(ValueError, match='scan_0.*missing'):
        metric_scale_status(tmp_path)
    scale = scan / 'metric_scale.json'
    scale.write_text(json.dumps({'available': True, 'scale_applied_to_colmap_geometry': False,
                                 'output_coordinate_units': 'meters'}))
    with pytest.raises(ValueError, match='scan_0.*uncalibrated'):
        metric_scale_status(tmp_path)
    scale.write_text(json.dumps({'available': True, 'scale_applied_to_colmap_geometry': True,
                                 'output_coordinate_units': 'meters'}))
    assert metric_scale_status(tmp_path)['units'] == 'meters'


def test_gaussian_identity_tracks_missing_and_changed_file(tmp_path):
    path = tmp_path / 'aligned_sealed_gaussians.ply'
    verify_inputs({str(path): None})
    path.write_bytes(b'first')
    with pytest.raises(ValueError, match='Input changed'):
        verify_inputs({str(path): None})
    recorded = {str(path): digest(path)}
    path.write_bytes(b'other')
    with pytest.raises(ValueError, match='Input changed'):
        verify_inputs(recorded)
    path.unlink()
    with pytest.raises(ValueError, match='Input changed'):
        verify_inputs(recorded)


def test_optional_gsplat_binary_appearance_is_a_changed_input(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import generate_cinematic_flythrough as generator
    scene = tmp_path / 'scene'
    files = {
        'artifacts/Generate Interactive Viewer/tour.json': '{}',
        'artifacts/Generate Interactive Viewer/generate_interactive_viewer_manifest.json': '{}',
        'artifacts/Extract Best Views/extract_best_views_manifest.json': '{"scans": []}',
        'artifacts/Generate Floor Plans/floor_level_metadata.json': '{"floor_planes": []}',
        'multi_room_alignment/aligned_sealed_meshes.ply': 'mesh',
        'multi_room_alignment/aligned_sealed_meshes.glb': 'glb',
        'multi_room_alignment/stats/phase07_region_replacements.json': '{}',
        'multi_room_alignment/stats/aligned_door_panels.json': '{}',
        'multi_room_alignment/stats/aligned_scans_transforms.json': '{"scans": {}}',
        'pairings.json': '{}',
    }
    for name, contents in files.items():
        path = scene / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    package = tmp_path / 'gsplat'
    package.mkdir()
    (package / '__init__.py').write_text('')
    monkeypatch.setattr(generator, 'find_spec', lambda _: SimpleNamespace(origin=str(package / '__init__.py')))
    inputs = generator.input_paths(scene, scene / 'pairings.json', {'levels': []})
    binary = package / 'csrc.so'
    assert inputs[str(binary)] is None
    binary.write_bytes(b'new binary')
    with pytest.raises(ValueError, match='Input changed'):
        generator.verify_inputs({str(binary): inputs[str(binary)]})
    present = generator.input_paths(scene, scene / 'pairings.json', {'levels': []})
    binary.unlink()
    with pytest.raises(ValueError, match='Input changed'):
        generator.verify_inputs({str(binary): present[str(binary)]})


def test_gaussian_ignores_changed_partial_and_concurrent_cache(tmp_path):
    import torch
    from concurrent.futures import ThreadPoolExecutor
    path = tmp_path / 'aligned_sealed_gaussians.ply'
    names = ['x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3', 'scale_0', 'scale_1',
             'scale_2', 'opacity', 'f_dc_0', 'f_dc_1', 'f_dc_2']
    vertices = np.zeros(1, dtype=[(name, 'f4') for name in names])
    vertices['rot_0'] = 1
    def save(x):
        vertices['x'] = x
        PlyData([PlyElement.describe(vertices, 'vertex')]).write(str(path))
    save(1)
    cache = path.with_suffix('.cache.pt')
    torch.save({'version': 1, 'source_sha256': digest(path),
                'means': torch.tensor([[99., 0., 0.]]), 'quats': torch.tensor([[1., 0., 0., 0.]]),
                'scales': torch.ones((1, 3)), 'opacities': torch.ones(1),
                'colors': torch.zeros((1, 1, 3)), 'sh_degree': 0}, cache)
    existing = cache.read_bytes()
    assert GaussianRenderer(path, device='cpu').world['means'][0, 0].item() == 1
    assert cache.read_bytes() == existing
    cache.write_bytes(b'partial')
    assert GaussianRenderer(path, device='cpu').world['means'][0, 0].item() == 1
    timestamp = path.stat().st_mtime_ns
    save(2)
    os.utime(path, ns=(timestamp, timestamp))
    assert GaussianRenderer(path, device='cpu').world['means'][0, 0].item() == 2
    cache.unlink()
    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(lambda _: GaussianRenderer(path, device='cpu').world['means'][0, 0].item(), range(2)))
    assert values == [2, 2]
    assert not cache.exists()


def test_gaussian_source_changed_during_loading_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / 'aligned_sealed_gaussians.ply'
    names = ['x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3', 'scale_0', 'scale_1',
             'scale_2', 'opacity', 'f_dc_0', 'f_dc_1', 'f_dc_2']
    vertices = np.zeros(1, dtype=[(name, 'f4') for name in names])
    vertices['rot_0'] = 1
    PlyData([PlyElement.describe(vertices, 'vertex')]).write(str(path))
    vertices['x'] = 1
    replacement = tmp_path / 'replacement.ply'
    PlyData([PlyElement.describe(vertices, 'vertex')]).write(str(replacement))
    original_read = PlyData.read
    def replace_after_read(*args, **kwargs):
        data = original_read(*args, **kwargs)
        os.replace(replacement, path)
        return data
    monkeypatch.setattr(PlyData, 'read', replace_after_read)
    with pytest.raises(ValueError, match='Gaussian source changed while loading'):
        GaussianRenderer(path, device='cpu')


def test_failed_encoder_does_not_block_producer():
    process = subprocess.Popen([sys.executable, '-c', 'pass'], stdin=subprocess.PIPE, bufsize=0)
    process.wait(timeout=5)
    with pytest.raises(BrokenPipeError):
        write_frame(process.stdin, b'x' * 1024)
    process.stdin.close()


def test_stalled_encoder_times_out(tmp_path, monkeypatch):
    import generate_cinematic_flythrough as generator
    ticks = iter([0., 31.])
    monkeypatch.setattr(generator.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(generator.select, 'select', lambda *args: ([], [], []))
    with (tmp_path / 'pipe').open('wb') as stream:
        with pytest.raises(TimeoutError, match='stopped accepting'):
            write_frame(stream, b'frame')


def test_failed_overwrite_keeps_existing_output(tmp_path, monkeypatch):
    import generate_cinematic_flythrough as generator
    output = tmp_path / 'tour'
    output.mkdir()
    (output / 'old.mp4').write_bytes(b'original')
    def fail(*args):
        raise ValueError('invalid scene')
    monkeypatch.setattr(generator, 'pairing_data', fail)
    with pytest.raises(ValueError, match='invalid scene'):
        preview(tmp_path, output, None, overwrite=True)
    assert (output / 'old.mp4').read_bytes() == b'original'


def test_final_rejects_changed_runtime_before_rendering(tmp_path, monkeypatch):
    import generate_cinematic_flythrough as generator
    output = tmp_path / 'tour'
    output.mkdir()
    trajectory = output / 'trajectory.npz'
    trajectory.write_bytes(b'unchanged')
    manifest = {'macro_step': generator.MACRO, 'macro_step_status': 'preview_complete',
                'schema_version': 2, 'trajectory_sha256': digest(trajectory),
                'runtime_versions': {'python': 'old'}}
    (output / 'generate_cinematic_flythrough_manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(generator, 'runtime_versions', lambda: {'python': 'new'})
    with pytest.raises(ValueError, match='package versions changed'):
        generator.final(tmp_path, output)
    assert not (output / 'final').exists()


def test_async_video_writer_roundtrip(tmp_path):
    import generate_cinematic_flythrough as generator
    p1 = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())'],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p2 = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())'],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    log1 = (tmp_path / 'log1.txt').open('w')
    log2 = (tmp_path / 'log2.txt').open('w')
    writer = generator.AsyncVideoWriter([p1, p2], [log1, log2])
    data1 = b'mesh_frame_data_123'
    data2 = b'gaussian_frame_data_456'
    writer.write_frames(data1, data2)
    writer.close('test')
    out1 = p1.stdout.read()
    out2 = p2.stdout.read()
    p1.wait()
    p2.wait()
    p1.stdout.close()
    p2.stdout.close()
    p1.stderr.close()
    p2.stderr.close()
    log1.close()
    log2.close()
    assert out1 == data1
    assert out2 == data2


def test_async_video_writer_propagates_encoder_crash(tmp_path):
    import time
    import generate_cinematic_flythrough as generator
    p1 = subprocess.Popen([sys.executable, '-c', 'import sys; sys.exit(1)'],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p2 = subprocess.Popen([sys.executable, '-c', 'import sys; sys.exit(1)'],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p1.wait()
    p2.wait()
    log1 = (tmp_path / 'log1.txt').open('w')
    log2 = (tmp_path / 'log2.txt').open('w')
    writer = generator.AsyncVideoWriter([p1, p2], [log1, log2])
    time.sleep(0.05)
    with pytest.raises((BrokenPipeError, RuntimeError)):
        for _ in range(20):
            writer.write_frames(b'frame', b'frame')
            time.sleep(0.01)
        writer.close('test')
    log1.close()
    log2.close()


