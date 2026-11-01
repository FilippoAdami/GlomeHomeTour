from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "06_mesh_elaboration" / "run_phase06.py"
SPEC = importlib.util.spec_from_file_location("run_phase06", SCRIPT)
run_phase06_module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(run_phase06_module)


def _stub_pipeline(monkeypatch, scan: Path, topology: dict, pipeline_code: int = 0) -> dict:
    mesh = scan / "mesh.ply"
    labels = scan / "labels.npy"
    metadata = scan / "metadata.json"
    scene_size = scan / "scene_size.json"
    for path in (mesh, labels, metadata, scene_size):
        path.write_bytes(b"input")

    monkeypatch.setattr(run_phase06_module, "resolve_scene", lambda _: scan)
    monkeypatch.setattr(run_phase06_module, "resolve_mesh", lambda _: mesh)
    monkeypatch.setattr(run_phase06_module, "resolve_labels", lambda _: labels)
    monkeypatch.setattr(run_phase06_module, "resolve_structural_metadata", lambda _: metadata)
    monkeypatch.setattr(run_phase06_module, "resolve_scene_size_file", lambda _: scene_size)
    monkeypatch.setattr(run_phase06_module, "resolve_segmentation_output", lambda _: None)
    calls = {}

    def rebuild(**kwargs) -> int:
        calls["pipeline"] = kwargs
        kwargs["output_dir"].mkdir(parents=True)
        return pipeline_code

    def finalize(staged_scan: Path) -> None:
        assert (staged_scan / "mesh_elaboration_steps").is_dir()
        calls["finalize"] = staged_scan

    def assemble(staged_scan: Path, elements_dir: Path) -> dict:
        assert elements_dir == staged_scan / "mesh_elaboration_steps" / "elements"
        elements_dir.mkdir()
        (elements_dir / "walls.ply").write_bytes(b"element")
        final = staged_scan / "mesh_elaboration_steps" / "assembled.ply"
        final.write_bytes(b"complete")
        calls["assemble"] = staged_scan
        return {"output_ply": str(final), "topology": topology}

    monkeypatch.setattr(run_phase06_module, "run_pipeline_for_scene", rebuild)
    monkeypatch.setattr(run_phase06_module, "finalize_single_elements", finalize)
    monkeypatch.setattr(run_phase06_module, "assemble_final_regularized_mesh", assemble)
    return calls


def _old_delivery(scan: Path) -> Path:
    delivery = scan / "mesh_elaboration"
    elements = delivery / "elements"
    elements.mkdir(parents=True)
    (delivery / "complete.ply").write_bytes(b"old complete")
    (elements / "walls.ply").write_bytes(b"old element")
    return delivery


def _valid_topology(boundary_edge_count: int = 0, boundary_y: list[float] | None = None) -> dict:
    return {
        "is_watertight": boundary_edge_count == 0,
        "boundary_edge_count": boundary_edge_count,
        "nonmanifold_edge_count": 0,
        "surface_components": 1,
        "boundary_edge_y_range_m": boundary_y,
        "floor_wall_base_y_m": -0.03,
    }


def test_runner_rebuilds_all_steps_and_replaces_managed_delivery(tmp_path: Path, monkeypatch) -> None:
    scan = tmp_path / "scan_0"
    scan.mkdir()
    old = _old_delivery(scan)
    calls = _stub_pipeline(monkeypatch, scan, _valid_topology(21, [-0.03, -0.03]))

    result = run_phase06_module.run_phase06(scan)

    output = scan / "mesh_elaboration"
    assert Path(result["complete_ply"]).read_bytes() == b"complete"
    assert sorted(path.name for path in output.iterdir()) == ["complete.ply", "elements"]
    assert [path.name for path in (output / "elements").iterdir()] == ["walls.ply"]
    assert calls["pipeline"]["scene_mesh"] == scan / "mesh.ply"
    assert calls["pipeline"]["output_dir"].parent != scan
    assert calls["finalize"] == calls["assemble"]
    assert result["topology"]["boundary_edge_count"] == 21
    assert not (scan / "mesh_elaboration_steps").exists()


def test_runner_does_not_publish_nonmanifold_output_or_replace_old_delivery(
    tmp_path: Path, monkeypatch,
) -> None:
    scan = tmp_path / "scan_0"
    scan.mkdir()
    old = _old_delivery(scan)
    old_complete = (old / "complete.ply").read_bytes()
    _stub_pipeline(monkeypatch, scan, {
        **_valid_topology(1, [-0.03, -0.03]),
        "nonmanifold_edge_count": 1,
    })

    with pytest.raises(RuntimeError, match="non-manifold"):
        run_phase06_module.run_phase06(scan)

    assert (old / "complete.ply").read_bytes() == old_complete
    assert not list(scan.glob(".phase06_build_*"))


def test_runner_preserves_separate_finalized_element_boundaries(tmp_path: Path, monkeypatch) -> None:
    scan = tmp_path / "scan_0"
    scan.mkdir()
    _stub_pipeline(monkeypatch, scan, _valid_topology(1, [-0.03, 0.1]))

    result = run_phase06_module.run_phase06(scan)

    assert result["topology"]["boundary_edge_count"] == 1
    assert Path(result["complete_ply"]).is_file()


def test_runner_stages_and_publishes_floor_opening_cache(tmp_path: Path, monkeypatch) -> None:
    scan = tmp_path / "scan_0"
    scan.mkdir()
    cache = scan / run_phase06_module.CACHE_NAME
    cache.write_text("before")
    _stub_pipeline(monkeypatch, scan, _valid_topology())
    rebuild = run_phase06_module.run_pipeline_for_scene

    def rebuild_with_cache(**kwargs) -> int:
        staged_cache = kwargs["output_dir"].parent / run_phase06_module.CACHE_NAME
        assert staged_cache.read_text() == "before"
        staged_cache.write_text("after")
        return rebuild(**kwargs)

    monkeypatch.setattr(run_phase06_module, "run_pipeline_for_scene", rebuild_with_cache)

    run_phase06_module.run_phase06(scan)

    assert cache.read_text() == "after"


def test_runner_preserves_unmanaged_output_directory(tmp_path: Path, monkeypatch) -> None:
    scan = tmp_path / "scan_0"
    scan.mkdir()
    output = scan / "mesh_elaboration"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("user data")
    calls = _stub_pipeline(monkeypatch, scan, _valid_topology())

    with pytest.raises(FileExistsError, match="unmanaged"):
        run_phase06_module.run_phase06(scan)

    assert marker.read_text() == "user data"
    assert "pipeline" not in calls


def test_runner_fails_before_finalization_when_step_1_to_3_fail(tmp_path: Path, monkeypatch) -> None:
    scan = tmp_path / "scan_0"
    scan.mkdir()
    _stub_pipeline(monkeypatch, scan, _valid_topology(), pipeline_code=1)

    with pytest.raises(RuntimeError, match="Steps 1–3 failed"):
        run_phase06_module.run_phase06(scan)

    assert not (scan / "mesh_elaboration").exists()
