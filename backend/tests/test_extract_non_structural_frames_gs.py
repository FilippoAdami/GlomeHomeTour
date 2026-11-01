#!/usr/bin/env python3
"""
Unit tests for backend/05_semantic_segmentation/extract_non_structural_frames_gs.py.
"""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "05_semantic_segmentation"))

from extract_non_structural_frames_gs import (
    compute_w2c_and_intrinsics,
    resolve_gs_checkpoint_path,
    render_gs_frame,
    run_gs_extraction,
)


def test_compute_w2c_and_intrinsics():
    eye = np.array([1.0, 1.5, 2.0], dtype=np.float32)
    target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    fov_deg = 60.0
    render_res = 1024

    viewmat, K = compute_w2c_and_intrinsics(eye, target, up, fov_deg, render_res)

    assert viewmat.shape == (4, 4)
    assert K.shape == (3, 3)

    # Check R is orthonormal rotation with det = +1
    R = viewmat[:3, :3]
    t = viewmat[:3, 3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)

    # Check camera center recovers eye
    recovered_eye = -R.T @ t
    np.testing.assert_allclose(recovered_eye, eye, atol=1e-4)

    # Check intrinsics
    expected_f = (render_res / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    assert np.isclose(K[0, 0], expected_f, rtol=1e-4)
    assert np.isclose(K[1, 1], expected_f, rtol=1e-4)
    assert np.isclose(K[0, 2], render_res / 2.0, atol=1e-4)
    assert np.isclose(K[1, 2], render_res / 2.0, atol=1e-4)


def test_resolve_gs_checkpoint_path(tmp_path: Path):
    ckpt = tmp_path / "gaussian_splat" / "checkpoint.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_text("dummy")

    found = resolve_gs_checkpoint_path(tmp_path)
    assert found == ckpt

    explicit = tmp_path / "custom.pt"
    explicit.write_text("dummy")
    assert resolve_gs_checkpoint_path(tmp_path, explicit) == explicit


def test_resolve_checkpoint_from_flat_scan_legacy(tmp_path: Path):
    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_text("legacy")
    assert resolve_gs_checkpoint_path(tmp_path) == ckpt


def test_resolve_checkpoint_from_single_scan_parent(tmp_path: Path):
    ckpt = tmp_path / "scan_0" / "gaussian_splat" / "checkpoint.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_text("current")
    assert resolve_gs_checkpoint_path(tmp_path) == ckpt


def test_resolve_checkpoint_from_mesh_source_path(tmp_path: Path):
    scan = tmp_path / "scan_0"
    model_dir = scan / "gaussian_splat"
    checkpoint = model_dir / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    ply = model_dir / "gaussian_splat.ply"
    ply.touch()
    mesh_dir = scan
    mesh_dir.mkdir(exist_ok=True)
    (mesh_dir / "mesh.json").write_text(json.dumps({"source_model_path": str(ply)}))

    assert resolve_gs_checkpoint_path(scan) == checkpoint


def test_resolve_checkpoint_prefers_stage04_derivative_recorded_in_manifest(tmp_path: Path):
    scan = tmp_path / "scan_0"
    source = scan / "gaussian_splat/checkpoint.pt"
    derived = scan / "gaussian_splat/checkpoint_sealed.pt"
    for checkpoint in (source, derived):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.touch()
    manifest = scan / "mesh.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"gsplat_sealing": {"checkpoint_path": str(derived)},
                                    "source_model_path": str(source.with_name("gaussian_splat.ply"))}))
    assert resolve_gs_checkpoint_path(scan) == derived
    assert resolve_gs_checkpoint_path(tmp_path) == derived
    manifest.write_text("{}")
    assert resolve_gs_checkpoint_path(scan) == source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm GPU required for gsplat rasterization")
def test_render_gs_frame_synthetic():
    N = 50
    device = "cuda"

    splats = {
        "means": torch.randn((N, 3), device=device, dtype=torch.float32),
        "quats": torch.randn((N, 4), device=device, dtype=torch.float32),
        "scales": torch.rand((N, 3), device=device, dtype=torch.float32) * 0.1,
        "opacities": torch.full((N,), 0.8, device=device, dtype=torch.float32),
        "colors": torch.rand((N, 16, 3), device=device, dtype=torch.float32),
    }

    eye = np.array([0.0, 0.0, 3.0], dtype=np.float32)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    viewmat, K = compute_w2c_and_intrinsics(eye, target, up, fov_deg=60.0, render_res=256)

    img = render_gs_frame(splats, viewmat, K, render_res=256, sh_degree=3, tile_size=16, device=device)

    assert isinstance(img, np.ndarray)
    assert img.shape == (256, 256, 3)
    assert img.dtype == np.uint8


def test_run_gs_extraction_discard_routing(monkeypatch, tmp_path: Path):
    manifest_dir = tmp_path / "segmentation" / "non_structural_category_frames" / "mesh"
    manifest_dir.mkdir(parents=True)
    manifest_path = manifest_dir / "non_structural_frames_manifest.json"

    manifest_data = {
        "padding_color": "black",
        "min_resolution": 256,
        "extracted_classes": {"10": {"class_id": 10, "role": "object"}},
        "extracted_instances": {
            "10_0": {
                "class_id": 10,
                "instance_id": 0,
                "role": "object",
                "camera_eye": [0.0, 1.5, 2.0],
                "camera_target": [0.0, 0.5, 0.0],
                "camera_fov_deg": 60.0,
                "raw_render_res": 256,
                "output_image_file": "class_010_inst_000.png",
            }
        },
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f)

    # Monkeypatch checkpoint and loading
    monkeypatch.setattr("extract_non_structural_frames_gs.resolve_gs_checkpoint_path", lambda ws, exp: tmp_path / "dummy.pt")
    monkeypatch.setattr("extract_non_structural_frames_gs.load_gaussian_splats", lambda ckpt, device: ({"means": torch.zeros((10, 3))}, 3))

    # Mock render_gs_frame to return a single flat brown color [48, 19, 12]
    flat_render = np.full((256, 256, 3), [48, 19, 12], dtype=np.uint8)
    monkeypatch.setattr("extract_non_structural_frames_gs.render_gs_frame", lambda **kwargs: flat_render)

    output_dir = tmp_path / "segmentation" / "non_structural_category_frames" / "gs"
    discarded_dir = tmp_path / "segmentation" / "non_structural_category_frames" / "gs_discarded"

    res = run_gs_extraction(
        workspace=tmp_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
        discarded_dir=discarded_dir,
        enable_quality_check=True,
    )

    # Must NOT exist in output_dir
    assert not (output_dir / "class_010_inst_000.png").exists()

    # MUST exist in discarded_dir
    assert (discarded_dir / "class_010_inst_000.png").exists()
    assert (discarded_dir / "discarded_manifest.json").exists()

    assert len(res["extracted_instances"]) == 0
    assert len(res["discarded_instances"]) == 1
    assert "10_0" in res["discarded_instances"]
    assert "single_color" in res["discarded_instances"]["10_0"]["discard_reason"]

