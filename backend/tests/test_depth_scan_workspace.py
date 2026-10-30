from types import SimpleNamespace

import step_depth
from step_depth import (
    _clean_debug_outputs,
    _discover_scene_workspaces,
    _scene_outputs,
)


def test_discovers_scans_in_numeric_order_and_accepts_single_scan(tmp_path):
    for name in ("scan_10", "scan_2"):
        scan = tmp_path / name
        scan.mkdir()
        (scan / "ingestion_results.json").write_text("{}")

    assert [p.name for p in _discover_scene_workspaces(tmp_path)] == ["scan_2", "scan_10"]
    assert [p.name for p in _discover_scene_workspaces(tmp_path / "scan_2")] == ["scan_2"]


def test_accepts_legacy_single_scene_without_ingestion_result(tmp_path):
    (tmp_path / "00_ingestion").mkdir()
    (tmp_path / "00_ingestion" / "transforms.json").write_text("{}")

    assert _discover_scene_workspaces(tmp_path) == [tmp_path]


def test_outputs_are_scoped_to_each_scan_and_debug_cleanup_preserves_runtime_files(tmp_path):
    scan = tmp_path / "scan_0"
    depth = scan / "depth"
    (depth / "depth_maps").mkdir(parents=True)
    (depth / "normal_maps").mkdir()
    for name in ("poses_da3.npz", "depth_alignment.json", "points3D_depth.ply",
                 "depth_inference_summary.md", "points3D_depth_planes_colored.ply"):
        (depth / name).touch()
    for name in ("depth_images", "normal_images", "preview"):
        (depth / name).mkdir()

    outputs = _scene_outputs(scan, "all")
    assert all(path.is_relative_to(scan) for path in outputs)

    _clean_debug_outputs(depth)

    assert (depth / "poses_da3.npz").exists()
    assert (depth / "depth_alignment.json").exists()
    assert (depth / "points3D_depth.ply").exists()
    assert (depth / "depth_maps").is_dir()
    assert (depth / "normal_maps").is_dir()
    assert not (depth / "depth_inference_summary.md").exists()
    assert not (depth / "points3D_depth_planes_colored.ply").exists()
    assert not any((depth / name).exists() for name in ("depth_images", "normal_images", "preview"))


def test_depth_stage_updates_pipeline_stats_without_log_or_stats_artifacts(tmp_path, monkeypatch):
    scan = tmp_path / "scan_0"
    scan.mkdir()
    (tmp_path / "certificate.json").write_text("{}")
    monkeypatch.setattr(step_depth, "_discover_scene_workspaces", lambda _: [scan])
    monkeypatch.setattr(step_depth, "_scene_outputs", lambda *_: [])
    monkeypatch.setattr(step_depth, "run_surfels_substep", lambda **_: None)
    monkeypatch.setattr(step_depth, "_write_scene_metric_metadata", lambda *_: None)
    monkeypatch.setattr(step_depth, "_scene_manifest", lambda _: SimpleNamespace(names=["frame.png"]))

    assert step_depth.main(["--workspace", str(tmp_path), "--substep", "surfels", "--force"]) == 0

    assert (tmp_path / "pipeline_stats.json").is_file()
    assert (scan / "pipeline_stats.json").is_file()
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "certificate.json", "pipeline_stats.json", "scan_0",
    ]
    for folder in (tmp_path, scan):
        assert not (folder / "depth_log.txt").exists()
        assert not (folder / "depth_stats.json").exists()
