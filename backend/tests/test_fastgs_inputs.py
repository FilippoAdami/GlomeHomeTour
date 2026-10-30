"""Stage 03 consumes locked stage 01/02 artifacts without rewriting them."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import gsplat_train as trainer
import step_train as step


def _scan(root, names=("frame 0001.png",)):
    (root / "images").mkdir(parents=True)
    sparse = root / "sparse/0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.txt").write_text("1 PINHOLE 7 5 6 4 3 2\n")
    rows = []
    for index, name in enumerate(names, 1):
        (root / "images" / name).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (7, 5)).save(root / "images" / name)
        rows.extend((f"{index} 1 0 0 0 {index} 0 0 1 {name}", ""))
    (sparse / "images.txt").write_text("\n".join(rows) + "\n")
    (sparse / "points3D.ply").write_bytes(b"locked sparse cloud")
    (root / "depth/depth_maps").mkdir(parents=True)
    (root / "depth/normal_maps").mkdir()
    xyz = np.eye(4, 3, dtype=np.float32)
    from scene.dataset_readers import storePly
    storePly(root / "depth/points3D_depth.ply", xyz, np.full_like(xyz, 128))
    for name in names:
        np.save(root / "depth/depth_maps" / f"{Path(name).stem}.npy", np.ones((5, 7), np.float16))
        normals = np.zeros((5, 7, 3), np.float16)
        normals[..., 2] = 1
        np.save(root / "depth/normal_maps" / f"{Path(name).stem}.npy", normals)
    return root


def _upstream_bytes(root):
    return {p.relative_to(root): p.read_bytes() for folder in ("images", "sparse", "depth")
            for p in (root / folder).rglob("*") if p.is_file()}


def test_scan_local_cloud_maps_and_nondivisible_camera_resolution(tmp_path):
    scan = _scan(tmp_path, ("zoom 1/frame 0001.png",))
    before = _upstream_bytes(scan)
    cameras = trainer.load_cameras(scan, resolution=2)
    camera = cameras[0]
    assert (camera["width"], camera["height"]) == (3, 2)
    np.testing.assert_allclose(camera["K"], [[18/7, 0, 9/7], [0, 8/5, 4/5], [0, 0, 1]])
    assert camera["image_path"].name == "frame 0001.png"
    assert camera["viewmat"][0, 3] == 1
    cloud = trainer._training_inputs(scan, cameras, True, True, True)
    assert cloud == scan / "depth/points3D_depth.ply"
    xyz, rgb = trainer._load_points(cloud, 0, 42)
    np.testing.assert_array_equal(xyz, np.eye(4, 3))
    np.testing.assert_allclose(rgb, 128/255)
    assert trainer._map_tensor(scan / "depth/depth_maps", camera, 1, torch.device("cpu")).shape == (2, 3, 1)
    assert trainer._map_tensor(scan / "depth/normal_maps", camera, 3, torch.device("cpu")).shape == (2, 3, 3)
    assert trainer._training_inputs(scan, cameras, False, False, False) == scan / "sparse/0/points3D.ply"
    assert _upstream_bytes(scan) == before


def test_stage03_restores_source_camera_frame_after_phase04_alignment(tmp_path):
    camera = {"viewmat": np.eye(4, dtype=np.float32)}
    camera["viewmat"][:3, 3] = [1., 2., 3.]
    transform = np.eye(4)
    transform[:3, :3] = [[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]
    transform[:3, 3] = [-2., .5, 1.]
    original = camera["viewmat"].copy() @ transform
    (tmp_path / "mesh.json").write_text(json.dumps({"mesh_alignment": {
        "poses_written_in_place": True, "transform_matrix": transform.tolist()}}))

    restored = trainer.restore_source_frame_cameras(tmp_path, [camera])[0]["viewmat"]

    np.testing.assert_allclose(restored, original, atol=1e-6)


@pytest.mark.parametrize("folder", ["depth_maps", "normal_maps"])
def test_requested_missing_supervision_fails_instead_of_silently_disabling(tmp_path, folder):
    scan = _scan(tmp_path)
    cameras = trainer.load_cameras(scan)
    (scan / "depth" / folder / "frame 0001.npy").unlink()
    with pytest.raises(FileNotFoundError, match="Requested supervision map missing"):
        trainer._training_inputs(scan, cameras, True, True, True)
    trainer._training_inputs(scan, cameras, True, folder != "depth_maps", folder != "normal_maps")


def test_missing_depth_cloud_requires_explicit_sparse_opt_out(tmp_path):
    scan = _scan(tmp_path)
    cameras = trainer.load_cameras(scan)
    (scan / "depth/points3D_depth.ply").unlink()
    with pytest.raises(FileNotFoundError, match="Initialization cloud missing"):
        trainer._training_inputs(scan, cameras, True, False, False)
    assert trainer._training_inputs(scan, cameras, False, False, False).name == "points3D.ply"


def test_colliding_map_stems_are_rejected(tmp_path):
    _scan(tmp_path, ("view.png", "view.jpg"))
    cameras = trainer.load_cameras(tmp_path)
    with pytest.raises(ValueError, match="stems must be unique"):
        trainer._training_inputs(tmp_path, cameras, True, True, True)
    assert trainer._training_inputs(tmp_path, cameras, True, False, False).is_file()


@pytest.mark.parametrize("model,params", [
    ("SIMPLE_RADIAL", "6 3 2 0"), ("RADIAL", "6 3 2 0 0"),
    ("OPENCV", "6 4 3 2 0 0 0 0"), ("FULL_OPENCV", "6 4 3 2 0 0 0 0 0 0 0 0"),
])
def test_zero_distortion_upstream_camera_models_are_accepted(tmp_path, model, params):
    _scan(tmp_path)
    (tmp_path / "sparse/0/cameras.txt").write_text(f"1 {model} 7 5 {params}\n")
    camera = trainer.load_cameras(tmp_path)[0]
    assert camera["K"][0, 0] == 6 and camera["K"][0, 2] == 3
    assert camera["K"][1, 1] == (6 if model in {"SIMPLE_RADIAL", "RADIAL"} else 4)
    (tmp_path / "sparse/0/cameras.txt").write_text(f"1 {model} 7 5 {params[:-1]}0.01\n")
    with pytest.raises(ValueError, match="nonzero distortion"):
        trainer.load_cameras(tmp_path)


@pytest.mark.parametrize("data,channels", [(np.zeros((2, 3, 1)), 1),
                                           (np.zeros((2, 3)), 3),
                                           (np.zeros((2, 3, 4)), 3),
                                           (np.zeros((2, 3), dtype=np.uint8), 1)])
def test_invalid_map_shapes_and_types_are_rejected(tmp_path, data, channels):
    np.save(tmp_path / "view.npy", data)
    camera = {"image_path": tmp_path / "view.png", "width": 3, "height": 2}
    with pytest.raises(ValueError, match="expected floating-point"):
        trainer._map_tensor(tmp_path, camera, channels, torch.device("cpu"))


def test_depth_gradient_targets_convert_without_changing_stablenormal_or_disk(tmp_path):
    # Independently fit the oriented normal of a plane in OpenGL camera space.
    K = np.array([[20., 0, 3.], [0, 30., 2.], [0, 0, 1.]], np.float32)
    y, x = np.mgrid[:8, :9].astype(np.float32)
    depth = 2 / (1 + .2 * (x - 3) / 20 + .3 * (y - 2) / 30)
    points_gl = np.stack(((x - 3) * depth / 20, -(y - 2) * depth / 30, -depth), -1)
    normal_gl = -np.cross(points_gl[1:-1, 2:] - points_gl[1:-1, :-2],
                          points_gl[2:, 1:-1] - points_gl[:-2, 1:-1])
    normal_gl /= np.linalg.norm(normal_gl, axis=-1, keepdims=True)
    np.save(tmp_path / "view.npy", normal_gl.astype(np.float16))
    before = (tmp_path / "view.npy").read_bytes()
    camera = {"image_path": tmp_path / "view.png", "width": 7, "height": 6}
    stable = trainer._map_tensor(tmp_path, camera, 3, torch.device("cpu"))
    converted = trainer._map_tensor(tmp_path, camera, 3, torch.device("cpu"), normal_model="depth_gradient")
    torch.testing.assert_close(converted, stable * torch.tensor([-1, 1, 1]))
    rendered = trainer._normals_from_depth(torch.from_numpy(depth), torch.from_numpy(K))
    torch.testing.assert_close(converted, rendered, atol=3e-4, rtol=3e-4)
    assert (tmp_path / "view.npy").read_bytes() == before


@pytest.mark.parametrize("sparse_opt_out", [False, True])
def test_wrapper_passes_cloud_selection_and_preserves_upstream_files(tmp_path, monkeypatch, sparse_opt_out):
    scan = _scan(tmp_path)
    before = _upstream_bytes(scan)
    commands = []

    def run(cmd, **kwargs):
        commands.append(cmd)
        model = Path(cmd[cmd.index("-m") + 1])
        model.mkdir(parents=True)
        (model / "gaussian_splat.ply").touch()
        (model / "checkpoint.pt").touch()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(step.subprocess, "run", run)
    argv = ["--workspace", str(scan), "--iterations", "2"]
    assert step.main(argv + (["--no-depth-cloud"] if sparse_opt_out else [])) == 0
    assert Path(commands[0][commands[0].index("-m") + 1]) == scan / "gaussian_splat"
    assert sorted(path.name for path in (scan / "gaussian_splat").iterdir()) == [
        "checkpoint.pt", "gaussian_splat.ply"
    ]
    assert ("--no-depth-cloud" in commands[0]) == sparse_opt_out
    assert commands[0][commands[0].index("-s") + 1] == str(scan)
    assert _upstream_bytes(scan) == before
    assert not (scan / "sparse/0/points3D_colmap.ply").exists()


def test_wrapper_rejects_missing_final_model(tmp_path, monkeypatch):
    def run(cmd, **kwargs):
        model = Path(cmd[cmd.index("-m") + 1])
        model.mkdir(parents=True)
        (model / "checkpoint.pt").touch()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(step.subprocess, "run", run)
    with pytest.raises(FileNotFoundError, match="gaussian_splat.ply"):
        step.main(["--workspace", str(tmp_path), "--iterations", "2"])


def test_self_check_reads_inputs_without_creating_model(tmp_path, capsys):
    _scan(tmp_path)
    before = _upstream_bytes(tmp_path)
    model = tmp_path / "unused_model"
    assert trainer.main(["-s", str(tmp_path), "-m", str(model), "--self-check",
                         "--depth-supervision", "--normal-supervision"]) == 0
    assert "inputs OK: 1 cameras" in capsys.readouterr().out
    assert not model.exists() and _upstream_bytes(tmp_path) == before


def test_multiple_scans_resume_and_retry_only_failed_scan(tmp_path, monkeypatch):
    (tmp_path / "pipeline_stats.json").write_text('{"depth": {}}')
    for name in ("scan_10", "scan_2"):
        scan = _scan(tmp_path / name)
        (scan / "ingestion_results.json").write_text("{}")
    seen = []

    def train(workspace, **kwargs):
        seen.append(workspace.name)
        if workspace.name == "scan_2" and seen.count("scan_2") == 1:
            raise RuntimeError("test failure")
        model = workspace / "gaussian_splat"
        model.mkdir(parents=True)
        (model / "gaussian_splat.ply").touch()
        (model / "checkpoint.pt").touch()

    monkeypatch.setattr(step, "train", train)
    argv = ["--workspace", str(tmp_path), "--iterations", "2"]
    with pytest.raises(RuntimeError, match="Training failed for scans: scan_2"):
        step.main(argv)
    assert seen == ["scan_2", "scan_10"]
    assert "train" not in json.loads((tmp_path / "pipeline_stats.json").read_text())
    assert step.main(argv) == 0
    assert seen == ["scan_2", "scan_10", "scan_2"]
    assert step.main(argv) == 0
    assert seen == ["scan_2", "scan_10", "scan_2"]
    assert (tmp_path / "scan_10/gaussian_splat/gaussian_splat.ply").is_file()
    assert (tmp_path / "scan_10/gaussian_splat/checkpoint.pt").is_file()
    assert (tmp_path / "scan_10/train_log.txt").is_file()
    assert (tmp_path / "scan_10/train_stats.json").is_file()
    assert not (tmp_path / "03_FastGS_DNSplatter").exists()
    assert not (tmp_path / "train_log.txt").exists()
    assert not (tmp_path / "train_stats.json").exists()
