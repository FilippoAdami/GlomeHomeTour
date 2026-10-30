"""Stage 04 consumes scan-local artifacts without changing earlier stages."""

import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
import torch
from scipy.spatial.transform import Rotation

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from Utilities.pipeline_paths import bootstrap
from Utilities.pipeline_step import read_pipeline_stats

bootstrap()
import gsplat_train
import infill_gsplat
import mesh_cleanup
import step_mesh
from fill_mesh_holes import read_ply_face_flags


def make_model(scene):
    xyz = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.float32)
    splats = gsplat_train._make_splats(xyz, np.full_like(xyz, .5), torch.device("cpu"))
    gsplat_train._write_model(scene / "gaussian_splat", splats, 3, 1)
    sparse = scene / "sparse/0"
    sparse.mkdir(parents=True, exist_ok=True)
    (sparse / "cameras.txt").write_text("1 PINHOLE 8 6 5 5 4 3\n")
    (sparse / "images.txt").write_text("1 1 0 0 0 0 0 0 1 frame.png\n\n")
    (sparse / "points3D.txt").write_text("1 0 0 0 128 128 128 0.1 1 0 0\n")
    return scene / "gaussian_splat"


def write_test_mesh(path):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]])
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    mesh.compute_vertex_normals()
    assert o3d.io.write_triangle_mesh(str(path), mesh)


@pytest.fixture
def mock_fusion(monkeypatch):
    seen = []

    def extract(scene, model_path, output, *args, **kwargs):
        seen.append(scene.name)
        assert model_path == scene / "gaussian_splat"
        output.mkdir(parents=True, exist_ok=True)
        write_test_mesh(output / "mesh.work.ply")
        return {"vertices": 8, "triangles": 12, "engine": "gsplat"}

    def clean(source, output, **kwargs):
        model = kwargs["gsplat_model_path"]
        assert (model / "checkpoint.pt").is_file()
        assert kwargs["output_mesh_path"] == source
        assert len(o3d.io.read_triangle_mesh(str(source)).triangles) == 2
        staged = kwargs["gsplat_output_dir"]
        checkpoint = torch.load(model / "checkpoint.pt", map_location="cpu", weights_only=True)
        torch.save(checkpoint, staged / "checkpoint_sealed.pt")
        infill_gsplat.export_splats_ply(staged / "gaussian_splat_sealed.ply", checkpoint["splats"])
        return {"sealed_vertices": 8, "sealed_triangles": 12, "clean_triangles": 12,
                "original_triangles_count": 12, "filled_triangles_count": 0,
                "boundary_loops_filled": 0, "remaining_boundary_edges": 0,
                "gsplat_sealing": {"checkpoint_path": str(model / "checkpoint_sealed.pt"),
                                   "ply_path": str(model / "gaussian_splat_sealed.ply"),
                                   "infill_gaussians_added": 0, "total_sealed_gaussians": 4}}

    def regularize(path):
        assert len(o3d.io.read_triangle_mesh(str(path)).triangles) == 2
        return {"mean_absolute_rgb_change": .01}

    monkeypatch.setattr(step_mesh, "extract_mesh", extract)
    monkeypatch.setattr(step_mesh, "clean_mesh_floaters", clean)
    monkeypatch.setattr(step_mesh, "regularize_mesh_rgb", regularize)
    return seen


def test_mesh_processes_all_upstream_scans_in_numeric_order_and_resumes(tmp_path, mock_fusion):
    for name in ("scan_10", "scan_2"):
        scene = tmp_path / name
        make_model(scene)
        (scene / "ingestion_results.json").write_text("{}")
    assert step_mesh.main(["--workspace", str(tmp_path)]) == 0
    assert mock_fusion == ["scan_2", "scan_10"]
    for scene in (tmp_path / "scan_2", tmp_path / "scan_10"):
        assert not (scene / "04_3DGS_to_mesh").exists()
        for name in ("mesh.ply", "mesh.json", "mesh_log.txt", "mesh_stats.json"):
            assert (scene / name).is_file()
    assert step_mesh.main(["--workspace", str(tmp_path)]) == 0
    assert mock_fusion == ["scan_2", "scan_10"]
    for directory in (tmp_path, tmp_path / "scan_2", tmp_path / "scan_10"):
        assert "mesh" in json.loads((directory / "pipeline_stats.json").read_text())


def test_mesh_retains_successful_scans_and_retries_only_failure(tmp_path, mock_fusion):
    for name in ("scan_0", "scan_1"):
        scene = tmp_path / name
        scene.mkdir()
        (scene / "ingestion_results.json").write_text("{}")
    make_model(tmp_path / "scan_1")
    with pytest.raises(RuntimeError, match="scan_0"):
        step_mesh.main(["--workspace", str(tmp_path)])
    assert mock_fusion == ["scan_1"]
    assert "mesh" not in read_pipeline_stats(tmp_path)
    make_model(tmp_path / "scan_0")
    assert step_mesh.main(["--workspace", str(tmp_path)]) == 0
    assert mock_fusion == ["scan_1", "scan_0"]


def test_mesh_requires_infill_and_rgb_outputs_before_resuming(tmp_path, mock_fusion):
    make_model(tmp_path)
    assert step_mesh.main(["--workspace", str(tmp_path)]) == 0
    output = tmp_path
    for path in (tmp_path / "gaussian_splat/checkpoint_sealed.pt", output / "mesh.json"):
        path.unlink()
        assert step_mesh.main(["--workspace", str(tmp_path)]) == 0
        assert path.is_file()
    assert mock_fusion == [tmp_path.name] * 3


@pytest.mark.parametrize("failure", ["rgb", "report"])
def test_failed_processing_preserves_previous_accepted_outputs(tmp_path, mock_fusion, monkeypatch, failure):
    model = make_model(tmp_path)
    (model / "checkpoint_sealed.pt").write_bytes(b"previous sealed checkpoint")
    (model / "gaussian_splat_sealed.ply").write_bytes(b"previous sealed PLY")
    before = {p.name: p.read_bytes() for p in model.iterdir()}
    output = tmp_path
    output.mkdir(exist_ok=True)
    (output / "mesh.ply").write_bytes(b"previous accepted mesh")
    (output / "mesh.json").write_text('{"previous": true}')

    def fail(path):
        path.with_name("mesh.work.rgb.tmp.ply").touch()
        raise RuntimeError("RGB failed")

    if failure == "rgb":
        monkeypatch.setattr(step_mesh, "regularize_mesh_rgb", fail)
    else:
        dumps = json.dumps

        def fail_report(data, **kwargs):
            if isinstance(data, dict) and "cleanup" in data:
                raise RuntimeError("Report failed")
            return dumps(data, **kwargs)

        monkeypatch.setattr(json, "dumps", fail_report)
    with pytest.raises(RuntimeError, match="failed"):
        step_mesh.main(["--workspace", str(tmp_path)])
    assert "mesh" not in read_pipeline_stats(tmp_path)
    assert (output / "mesh.ply").read_bytes() == b"previous accepted mesh"
    assert (output / "mesh.json").read_text() == '{"previous": true}'
    assert not list(output.glob("*.tmp*"))
    assert not (output / "mesh.work.ply").exists()
    assert {p.name: p.read_bytes() for p in model.iterdir()} == before


def test_complete_phase_keeps_one_mesh_and_adds_sibling_gaussians(tmp_path, mock_fusion):
    model = make_model(tmp_path)
    (tmp_path / "depth").mkdir()
    source_w2c = np.eye(4, dtype=np.float32)[None]
    np.savez(tmp_path / "depth/poses_da3.npz", w2c=source_w2c, K=np.eye(3)[None], names=["frame.png"])
    before = {p.name: p.read_bytes() for p in model.iterdir()}
    assert step_mesh.main(["--workspace", str(tmp_path)]) == 0
    output = tmp_path
    assert sorted(p.name for p in output.iterdir()) == ["depth", "gaussian_splat", "mesh.json", "mesh.ply", "mesh_log.txt", "mesh_stats.json", "pipeline_stats.json", "sparse"]
    report = json.loads((output / "mesh.json").read_text())
    assert report["mesh_alignment"]["poses_written_in_place"] is True
    transform = np.asarray(report["mesh_alignment"]["transform_matrix"])
    image_fields = (output / "sparse/0/images.txt").read_text().splitlines()[0].split(maxsplit=9)
    image_rotation = Rotation.from_quat([*map(float, image_fields[2:5]), float(image_fields[1])]).as_matrix()
    image_translation = np.asarray(image_fields[5:8], dtype=float)
    np.testing.assert_allclose(-image_rotation.T @ image_translation, transform[:3, 3], atol=1e-9)
    sparse_point = np.asarray((output / "sparse/0/points3D.txt").read_text().split()[1:4], dtype=float)
    np.testing.assert_allclose(sparse_point, transform[:3, 3], atol=1e-10)
    with np.load(output / "depth/poses_da3.npz") as aligned_depth:
        np.testing.assert_allclose(aligned_depth["w2c"][0], source_w2c[0] @ np.linalg.inv(transform), atol=1e-6)
    assert not (output / "phase04_alignment").exists()
    first_images = (output / "sparse/0/images.txt").read_bytes()
    first_points = (output / "sparse/0/points3D.txt").read_bytes()
    first_depth_poses = (output / "depth/poses_da3.npz").read_bytes()
    assert not (output / "04_3DGS_to_mesh").exists()
    assert sorted(p.name for p in model.iterdir()) == ["checkpoint.pt", "checkpoint_sealed.pt",
                                                    "gaussian_splat.ply", "gaussian_splat_sealed.ply"]
    assert {name: (model / name).read_bytes() for name in before} == before
    assert len(o3d.io.read_triangle_mesh(str(output / "mesh.ply")).triangles) == 2
    legacy = output / "phase04_alignment"
    (legacy / "sparse/0").mkdir(parents=True)
    (legacy / "depth").mkdir()
    for path in (legacy / "alignment.json", legacy / "sparse/0/cameras.txt",
                 legacy / "sparse/0/images.txt", legacy / "sparse/0/points3D.txt",
                 legacy / "depth/poses_da3.npz"):
        path.write_text("legacy sidecar")
    assert step_mesh.main(["--workspace", str(tmp_path), "--force"]) == 0
    assert not legacy.exists()
    assert len(o3d.io.read_triangle_mesh(str(output / "mesh.ply")).triangles) == 2
    assert (output / "sparse/0/images.txt").read_bytes() == first_images
    assert (output / "sparse/0/points3D.txt").read_bytes() == first_points
    assert (output / "depth/poses_da3.npz").read_bytes() == first_depth_poses


def test_completed_run_removes_old_generated_copies_but_preserves_other_files(tmp_path, mock_fusion):
    make_model(tmp_path)
    output = tmp_path / "04_3DGS_to_mesh"
    output.mkdir(exist_ok=True)
    for name in ("mesh_initial.ply", "mesh_cleaned.ply", "mesh_sealed.ply", "mesh_red.ply",
                 "mesh_rgb_regularization.json", "mesh_floater_removal_stats.json"):
        (output / name).write_text("obsolete generated output")
    legacy = output / "gaussian_splat"
    legacy.mkdir()
    for name in ("checkpoint.pt", "gaussian_splat.ply"):
        (legacy / name).write_text("obsolete derivative")
    (output / "notes.txt").write_text("preserve")
    (legacy / "notes.txt").write_text("preserve")
    assert step_mesh.main(["--workspace", str(tmp_path)]) == 0
    assert sorted(p.name for p in output.iterdir()) == ["gaussian_splat", "notes.txt"]
    assert list(legacy.iterdir()) == [legacy / "notes.txt"]


def test_obsolete_cleanup_never_removes_an_explicit_source_model(tmp_path):
    output = tmp_path / "04_3DGS_to_mesh"
    source = make_model(output)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    step_mesh.remove_obsolete_outputs(tmp_path, tmp_path, source)
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before


def test_completed_run_removes_empty_legacy_stage_folder(tmp_path, mock_fusion):
    make_model(tmp_path)
    legacy = tmp_path / "04_3DGS_to_mesh"
    legacy.mkdir()
    for name in ("mesh.ply", "mesh.json", "mesh_log.txt", "mesh_stats.json"):
        (legacy / name).write_text("previous output")
    assert step_mesh.main(["--workspace", str(tmp_path)]) == 0
    assert not legacy.exists()
    assert len(o3d.io.read_triangle_mesh(str(tmp_path / "mesh.ply")).triangles) == 2


def test_obsolete_cleanup_does_not_follow_stage_directory_symlinks(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    mesh = other / "mesh.ply"
    mesh.write_bytes(b"user-owned mesh")
    legacy = tmp_path / "04_3DGS_to_mesh"
    legacy.symlink_to(other, target_is_directory=True)
    step_mesh.remove_obsolete_outputs(tmp_path, tmp_path, tmp_path / "gaussian_splat")
    assert mesh.read_bytes() == b"user-owned mesh"
    assert legacy.is_symlink()


def test_custom_output_keeps_preexisting_models_and_other_files(tmp_path, mock_fusion):
    make_model(tmp_path)
    output = tmp_path / "custom"
    unrelated_model = make_model(output)
    before = {p.name: p.read_bytes() for p in unrelated_model.iterdir()}
    (output / "mesh_cleaned.ply").write_bytes(b"user-owned input")
    assert step_mesh.main(["--workspace", str(tmp_path), "--output", str(output)]) == 0
    assert {p.name: p.read_bytes() for p in unrelated_model.iterdir()} == before
    assert (output / "mesh_cleaned.ply").read_bytes() == b"user-owned input"


def test_obsolete_cleanup_does_not_follow_model_directory_symlinks(tmp_path):
    original = make_model(tmp_path)
    other_model = make_model(tmp_path / "other")
    before = {p.name: p.read_bytes() for p in other_model.iterdir()}
    output = tmp_path / "04_3DGS_to_mesh"
    output.mkdir(exist_ok=True)
    (output / "gaussian_splat").symlink_to(other_model, target_is_directory=True)
    step_mesh.remove_obsolete_outputs(tmp_path, tmp_path, original)
    assert {p.name: p.read_bytes() for p in other_model.iterdir()} == before
    assert (output / "gaussian_splat").is_symlink()


@pytest.mark.parametrize("module", [step_mesh, mesh_cleanup])
def test_gaussian_infill_has_no_cli_bypass(tmp_path, module):
    with pytest.raises(SystemExit):
        module.main(["--workspace", str(tmp_path), "--no-seal-gsplat"])


@pytest.mark.parametrize("flag", ["--output", "--model-path"])
def test_multiple_scans_reject_ambiguous_shared_paths(tmp_path, flag):
    for name in ("scan_0", "scan_1"):
        scene = tmp_path / name
        make_model(scene)
        (scene / "ingestion_results.json").write_text("{}")
    with pytest.raises(SystemExit):
        step_mesh.main(["--workspace", str(tmp_path), flag, str(tmp_path / "shared")])


def test_sealing_adds_sibling_files_and_preserves_original_bytes(tmp_path):
    model = make_model(tmp_path)
    before = {p.name: p.read_bytes() for p in model.iterdir()}
    box = o3d.geometry.TriangleMesh.create_box()
    box.paint_uniform_color([.5, .5, .5])
    stats = infill_gsplat.seal_gsplat_model(model, box, 0)
    assert Path(stats["checkpoint_path"]) == model / "checkpoint_sealed.pt"
    assert Path(stats["ply_path"]) == model / "gaussian_splat_sealed.ply"
    assert stats["infill_gaussians_added"] > 0
    assert {name: (model / name).read_bytes() for name in before} == before
    assert sorted(p.name for p in model.iterdir()) == ["checkpoint.pt", "checkpoint_sealed.pt",
                                                     "gaussian_splat.ply", "gaussian_splat_sealed.ply"]
    output = torch.load(stats["checkpoint_path"], weights_only=True)
    assert len(output["splats"]["means"]) == stats["total_sealed_gaussians"]
    torch.testing.assert_close(output["splats"]["means"][:4],
                               torch.load(model / "checkpoint.pt", weights_only=True)["splats"]["means"])


def test_sealing_without_targets_still_exports_model_and_consistent_stats(tmp_path):
    model = make_model(tmp_path)
    box = o3d.geometry.TriangleMesh.create_box()
    stats = infill_gsplat.seal_gsplat_model(model, box, len(box.triangles),
                                         coverage_deficit_threshold_m=None)
    assert stats["infill_gaussians_added"] == 0
    assert stats["total_sealed_gaussians"] == 4
    assert Path(stats["checkpoint_path"]).is_file()
    assert Path(stats["ply_path"]).is_file()
    checkpoint = torch.load(stats["checkpoint_path"], weights_only=True)
    assert checkpoint["sealed"] is True
    assert checkpoint["original_gaussians"] == 4
    assert checkpoint["infill_gaussians_added"] == 0


def test_sealing_uses_allocated_sh_slots_after_short_upstream_training(tmp_path):
    model = make_model(tmp_path)
    checkpoint = torch.load(model / "checkpoint.pt", weights_only=True)
    checkpoint["sh_degree"] = 0
    torch.save(checkpoint, model / "checkpoint.pt")
    box = o3d.geometry.TriangleMesh.create_box()
    stats = infill_gsplat.seal_gsplat_model(model, box, 0)
    derived = torch.load(stats["checkpoint_path"], weights_only=True)
    assert derived["sh_degree"] == 0
    assert derived["splats"]["shN"].shape == (stats["total_sealed_gaussians"], 15, 3)


def test_ply_face_metadata_accepts_ascii_triangle_inputs(tmp_path):
    source = tmp_path / "ascii.ply"
    source.write_text("ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\n"
                      "property float y\nproperty float z\nelement face 1\n"
                      "property list uchar int vertex_indices\nproperty uchar is_hole_fill\n"
                      "end_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2 1\n")
    np.testing.assert_array_equal(read_ply_face_flags(source), [1])


def test_ply_face_metadata_rejects_truncated_header(tmp_path):
    from plyfile import PlyHeaderParseError

    source = tmp_path / "truncated.ply"
    source.write_text("ply\nformat binary_little_endian 1.0\n")
    with pytest.raises(PlyHeaderParseError):
        read_ply_face_flags(source)


def test_sealing_rejects_overwriting_its_source(tmp_path):
    model = make_model(tmp_path)
    box = o3d.geometry.TriangleMesh.create_box()
    (model / "checkpoint_sealed.pt").symlink_to(model / "checkpoint.pt")
    with pytest.raises(ValueError, match="read-only"):
        infill_gsplat.seal_gsplat_model(model, box, 0)


def test_rgb_preserves_geometry_and_face_metadata_without_sidecars(tmp_path):
    from fill_mesh_holes import export_ply_with_face_attributes
    from regularize_mesh_rgb import regularize_mesh_rgb

    box = o3d.geometry.TriangleMesh.create_box()
    vertices, triangles = np.asarray(box.vertices), np.asarray(box.triangles)
    colors = np.linspace(.1, .9, len(vertices) * 3).reshape(-1, 3)
    flags = np.zeros(len(triangles), dtype=np.uint8)
    flags[-1] = 1
    mesh_path = tmp_path / "mesh.ply"
    export_ply_with_face_attributes(mesh_path, vertices, triangles, colors,
                                   {"is_hole_fill": ("uchar", flags),
                                    **{f"fill_{c}": ("uchar", np.full(len(triangles), value))
                                       for c, value in zip("rgb", (127, 128, 129))}})
    summary = regularize_mesh_rgb(mesh_path)
    written = o3d.io.read_triangle_mesh(str(mesh_path))
    np.testing.assert_array_equal(written.vertices, vertices)
    np.testing.assert_array_equal(written.triangles, triangles)
    np.testing.assert_array_equal(read_ply_face_flags(mesh_path), flags)
    for c, value in zip("rgb", (127, 128, 129)):
        np.testing.assert_array_equal(read_ply_face_flags(mesh_path, f"fill_{c}"),
                                      np.full(len(triangles), value))
    assert summary["mean_absolute_rgb_change"] > 0
    assert list(tmp_path.iterdir()) == [mesh_path]


def test_cleanup_rejects_missing_requested_model(tmp_path):
    box = o3d.geometry.TriangleMesh.create_box()
    source = tmp_path / "input.ply"
    o3d.io.write_triangle_mesh(str(source), box)
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        mesh_cleanup.clean_mesh_floaters(source, tmp_path / "output", enable_sor=False,
                                        gsplat_model_path=tmp_path / "missing")


def test_diagnostic_cleanup_does_not_certify_the_complete_phase(tmp_path, monkeypatch):
    from Utilities.pipeline_step import update_pipeline_stats

    make_model(tmp_path)
    output = tmp_path
    output.mkdir(exist_ok=True)
    for name in ("mesh.ply", "mesh.json"):
        (output / name).touch()
    for step in ("mesh", "mesh_rgb", "mesh_cleanup"):
        update_pipeline_stats(tmp_path, step, 1, 1)
    seen = []

    def clean(*args, **kwargs):
        seen.append(args)
        return {"boundary_loops_filled": 0, "output_mesh": str(output / "mesh.ply"),
                "gsplat_sealing": {"infill_gaussians_added": 0, "total_sealed_gaussians": 4}}

    monkeypatch.setattr(mesh_cleanup, "clean_mesh_floaters", clean)
    assert mesh_cleanup.main(["--workspace", str(tmp_path), "--input", str(output / "mesh.ply")]) == 0
    assert len(seen) == 1
    stats = read_pipeline_stats(tmp_path)
    assert "mesh_cleanup" in stats
    assert "mesh" not in stats and "mesh_rgb" not in stats


def test_web_export_uses_manifest_derivative_with_legacy_source_fallback(tmp_path, monkeypatch):
    from plyfile import PlyData
    from server import pipeline_runner

    model = make_model(tmp_path)
    output = tmp_path
    output.mkdir(exist_ok=True)
    box = o3d.geometry.TriangleMesh.create_box()
    box.paint_uniform_color([.5, .5, .5])
    o3d.io.write_triangle_mesh(str(output / "mesh.ply"), box)
    sealing = infill_gsplat.seal_gsplat_model(model, box, 0)
    manifest = output / "mesh.json"
    manifest.write_text(json.dumps({"gsplat_sealing": sealing}))
    exports = tmp_path / "exports"
    monkeypatch.setattr(pipeline_runner, "OUTPUTS_DIR", exports)
    pipeline_runner.optimize_3d_assets(tmp_path, "sealed")
    assert len(PlyData.read(str(exports / "sealed/splat.ply"))["vertex"]) == sealing["total_sealed_gaussians"]
    # Legacy manifests can predate stage 04 derivatives.
    manifest.write_text("{}")
    pipeline_runner.optimize_3d_assets(tmp_path, "original")
    assert len(PlyData.read(str(exports / "original/splat.ply"))["vertex"]) == 4


@pytest.mark.parametrize("fail_mesh", [False, True])
def test_server_exports_only_after_complete_mesh_phase(tmp_path, monkeypatch, fail_mesh):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from server import pipeline_runner

    connection = Mock()
    seen, exports = [], []

    def run(command, **kwargs):
        step = command[command.index("--only-step") + 1]
        seen.append(step)
        return SimpleNamespace(returncode=int(fail_mesh and step == "mesh"),
                               stderr="Mesh phase failed", stdout="")

    def export(*args):
        assert seen[-1] == "mesh"
        exports.append(args)
        return "/mesh.obj", "/splat.ply"

    monkeypatch.setattr(pipeline_runner, "BACKEND_DIR", tmp_path)
    monkeypatch.setattr(pipeline_runner, "get_connection", lambda: connection)
    monkeypatch.setattr(pipeline_runner.subprocess, "run", run)
    monkeypatch.setattr(pipeline_runner, "optimize_3d_assets", export)
    pipeline_runner.execute_pipeline("test", "capture.zip")
    assert seen[-1] == "mesh" and "mesh_rgb" not in seen
    assert len(exports) == (0 if fail_mesh else 1)
    calls = connection.execute.call_args_list
    initial_stages = json.loads(calls[0].args[1][2])
    assert initial_stages[-1]["name"] == "mesh"
    assert any("SET status = 'error'" in c.args[0] for c in calls) == fail_mesh


@pytest.mark.parametrize("flag,value", [("--resolution", "0"), ("--gsplat-max-ratio", "nan"),
                                         ("--gsplat-max-ratio", "0"),
                                         ("--gsplat-normal-thickness", "-1")])
def test_mesh_rejects_invalid_input_settings(tmp_path, flag, value):
    with pytest.raises(SystemExit):
        step_mesh.main(["--workspace", str(tmp_path), flag, value])
