#!/usr/bin/env python3
"""Unit tests for Gaussian Splat infilling and checkpoint sealing."""

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
from plyfile import PlyData
from scipy.spatial.transform import Rotation
import torch

backend_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_dir / "04_3DGS_to_mesh"))
sys.path.insert(0, str(backend_dir))

from infill_gsplat import (
    SH_C0,
    export_splats_ply,
    normal_to_quaternion,
    sample_squeezed_gaussians_from_triangles,
    seal_gsplat_model,
)
from mesh_cleanup import clean_mesh_floaters


def test_normal_to_quaternion_properties():
    test_normals = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0),
        np.array([-0.3, 0.7, -0.65]) / np.linalg.norm([-0.3, 0.7, -0.65]),
    ]

    for n in test_normals:
        q = normal_to_quaternion(n)
        assert q.shape == (4,)
        # Unit quaternion
        norm_q = np.linalg.norm(q)
        assert np.isclose(norm_q, 1.0, atol=1e-6)

        # Reconstruct rotation matrix R from [w, x, y, z] -> scipy [x, y, z, w]
        rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        R = rot.as_matrix()

        # Check det(R) = +1 (right-handed rotation)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)

        # Check that local +Z axis [0, 0, 1] maps to target normal n
        z_transformed = R @ np.array([0.0, 0.0, 1.0])
        assert np.allclose(z_transformed, n, atol=1e-5)


def test_normal_to_quaternion_batch():
    normals = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    quats = normal_to_quaternion(normals)
    assert quats.shape == (3, 4)
    for i in range(3):
        q = quats[i]
        rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        R = rot.as_matrix()
        assert np.allclose(R @ np.array([0.0, 0.0, 1.0]), normals[i], atol=1e-5)


def test_sample_squeezed_gaussians_from_triangles():
    # Create two triangles in XY plane (normal = [0, 0, 1])
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.1, 0.1, 0.0],
    ], dtype=np.float32)
    triangles = np.array([
        [0, 1, 2],
        [1, 3, 2],
    ], dtype=np.int32)
    colors = np.array([
        [0.8, 0.2, 0.1],
        [0.2, 0.8, 0.3],
    ], dtype=np.float32)

    splats = sample_squeezed_gaussians_from_triangles(
        vertices=vertices,
        triangles=triangles,
        triangle_colors=colors,
        sample_density_m=0.02,
        normal_thickness_m=0.001,
        opacity=0.99,
        sh_degree=3,
    )

    assert "means" in splats
    assert "scales" in splats
    assert "quats" in splats
    assert "opacities" in splats
    assert "sh0" in splats
    assert "shN" in splats

    n_splats = len(splats["means"])
    assert n_splats > 0
    assert splats["scales"].shape == (n_splats, 3)
    assert splats["quats"].shape == (n_splats, 4)
    assert splats["opacities"].shape == (n_splats,)
    assert splats["sh0"].shape == (n_splats, 1, 3)
    assert splats["shN"].shape == (n_splats, 15, 3)

    # Validate scale squeeze: normal axis is scale index 2
    log_scales = splats["scales"].numpy()
    scales = np.exp(log_scales)
    tangent_s0 = scales[:, 0]
    tangent_s1 = scales[:, 1]
    normal_s = scales[:, 2]

    # Normal thickness should match normal_thickness_m (0.001 m)
    assert np.allclose(normal_s, 0.001, atol=1e-5)
    # Tangent scales should be at least 5x larger than normal thickness
    assert np.all(tangent_s0 / normal_s >= 5.0)
    assert np.all(tangent_s1 / normal_s >= 5.0)

    # Validate opacities in logit space
    sigmoid_opacities = torch.sigmoid(splats["opacities"]).numpy()
    assert np.allclose(sigmoid_opacities, 0.99, atol=1e-3)

    # Validate colors reconstructed from sh0
    sh0_np = splats["sh0"].numpy()[:, 0, :]
    reconstructed_rgb = sh0_np * SH_C0 + 0.5
    assert np.all(reconstructed_rgb >= 0.0)
    assert np.all(reconstructed_rgb <= 1.0)


def test_sample_squeezed_gaussians_empty():
    vertices = np.empty((0, 3), dtype=np.float32)
    triangles = np.empty((0, 3), dtype=np.int32)
    colors = np.empty((0, 3), dtype=np.float32)

    splats = sample_squeezed_gaussians_from_triangles(
        vertices=vertices,
        triangles=triangles,
        triangle_colors=colors,
    )
    assert len(splats["means"]) == 0
    assert splats["means"].shape == (0, 3)
    assert splats["scales"].shape == (0, 3)
    assert splats["quats"].shape == (0, 4)
    assert splats["sh0"].shape == (0, 1, 3)
    assert splats["shN"].shape == (0, 15, 3)


def test_export_splats_ply(tmp_path):
    n = 20
    splats = {
        "means": torch.randn((n, 3), dtype=torch.float32),
        "scales": torch.randn((n, 3), dtype=torch.float32),
        "quats": torch.randn((n, 4), dtype=torch.float32),
        "opacities": torch.randn((n,), dtype=torch.float32),
        "sh0": torch.randn((n, 1, 3), dtype=torch.float32),
        "shN": torch.zeros((n, 15, 3), dtype=torch.float32),
    }

    ply_path = tmp_path / "test_splats.ply"
    export_splats_ply(ply_path, splats)
    assert ply_path.is_file()

    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]
    assert len(vertex) == n
    assert "x" in vertex
    assert "y" in vertex
    assert "z" in vertex
    assert "f_dc_0" in vertex
    assert "f_rest_0" in vertex
    assert "f_rest_44" in vertex
    assert "scale_0" in vertex
    assert "rot_0" in vertex
    assert "opacity" in vertex


def create_synthetic_checkpoint(path: Path, n_gaussians: int = 50):
    k_sh = 15
    splats = {
        "means": torch.randn((n_gaussians, 3), dtype=torch.float32),
        "scales": torch.randn((n_gaussians, 3), dtype=torch.float32),
        "quats": torch.randn((n_gaussians, 4), dtype=torch.float32),
        "opacities": torch.randn((n_gaussians,), dtype=torch.float32),
        "sh0": torch.randn((n_gaussians, 1, 3), dtype=torch.float32),
        "shN": torch.zeros((n_gaussians, k_sh, 3), dtype=torch.float32),
    }
    payload = {
        "format": "gsplat-stage03-v1",
        "splats": splats,
        "sh_degree": 3,
        "resolution": 1,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    export_splats_ply(path.parent / "gaussian_splat.ply", splats)


def test_seal_gsplat_model(tmp_path):
    model_dir = tmp_path / "gaussian_splat"
    checkpoint_file = model_dir / "checkpoint.pt"
    create_synthetic_checkpoint(checkpoint_file, n_gaussians=60)
    source_bytes = {p.name: p.read_bytes() for p in model_dir.iterdir()}

    # Create mesh: 4 clean triangles, 2 infilled triangles
    v = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
    ], dtype=np.float64)
    # 4 clean triangles, 2 infill triangles (total 6 triangles)
    t = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [1, 4, 5],
        [1, 5, 2],
        # Infill triangles
        [3, 2, 5],
        [3, 5, 0],
    ], dtype=np.int32)
    c = np.full((len(v), 3), 0.5)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(t)
    mesh.vertex_colors = o3d.utility.Vector3dVector(c)

    clean_num_triangles = 4
    infill_colors = np.array([
        [0.9, 0.1, 0.1],
        [0.1, 0.9, 0.1],
    ], dtype=np.float64)

    stats = seal_gsplat_model(
        gsplat_model_path=model_dir,
        sealed_mesh=mesh,
        clean_num_triangles=clean_num_triangles,
        infill_colors=infill_colors,
        sample_density_m=0.1,
    )

    assert stats["original_gaussians"] == 60
    assert stats["infill_triangles"] == 2
    assert stats["infill_gaussians_added"] > 0
    assert stats["total_sealed_gaussians"] == 60 + stats["infill_gaussians_added"]

    # Verify the stage 04 derivative and preserve both locked stage 03 artifacts.
    assert Path(stats["checkpoint_path"]) == model_dir / "checkpoint_sealed.pt"
    assert Path(stats["ply_path"]) == model_dir / "gaussian_splat_sealed.ply"
    assert {name: (model_dir / name).read_bytes() for name in source_bytes} == source_bytes
    updated_loaded = torch.load(stats["checkpoint_path"], map_location="cpu", weights_only=True)
    assert len(updated_loaded["splats"]["means"]) == stats["total_sealed_gaussians"]
    assert updated_loaded["sealed"] is True
    assert updated_loaded["infill_gaussians_added"] == stats["infill_gaussians_added"]

    active_ply = Path(stats["ply_path"])
    assert active_ply.is_file()
    plydata = PlyData.read(str(active_ply))
    assert len(plydata["vertex"]) == stats["total_sealed_gaussians"]
    assert sorted(path.name for path in model_dir.iterdir()) == ["checkpoint.pt", "checkpoint_sealed.pt",
                                                               "gaussian_splat.ply", "gaussian_splat_sealed.ply"]

    repeated = seal_gsplat_model(
        gsplat_model_path=model_dir,
        sealed_mesh=mesh,
        clean_num_triangles=clean_num_triangles,
        infill_colors=infill_colors,
        sample_density_m=0.1,
    )
    assert repeated["total_sealed_gaussians"] == stats["total_sealed_gaussians"]
    assert {name: (model_dir / name).read_bytes() for name in source_bytes} == source_bytes


def test_clean_mesh_floaters_with_gsplat_sealing(tmp_path):
    workspace = tmp_path / "workspace"
    model_dir = workspace / "gaussian_splat"
    mesh_dir = workspace / "04_3DGS_to_mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    create_synthetic_checkpoint(model_dir / "checkpoint.pt", n_gaussians=40)

    # Create closed box mesh (solid watertight object)
    box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    box.paint_uniform_color([0.6, 0.4, 0.2])
    input_mesh = mesh_dir / "mesh_initial.ply"
    o3d.io.write_triangle_mesh(str(input_mesh), box)

    summary = clean_mesh_floaters(
        input_mesh_path=input_mesh,
        output_dir=mesh_dir,
        min_triangles=6,
        min_area=0.01,
        min_thickness=0.01,
        min_solidity=0.1,
        max_boundary_openness=1.0,
        enable_sor=False,
        gsplat_model_path=model_dir,
    )

    assert "gsplat_sealing" in summary
    assert (mesh_dir / "mesh.ply").is_file()
    assert not (mesh_dir / "mesh_floater_removal_stats.json").exists()
    assert (model_dir / "checkpoint_sealed.pt").is_file()


def test_color_scaling_not_black():
    """Verify that colors near 1.0 (with float inaccuracies like 1.0000000000000004) do not become pitch black."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
    ], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    # Target color with float error slightly above 1.0
    colors = np.array([[1.0000000000000004, 0.8, 0.5]], dtype=np.float64)

    splats = sample_squeezed_gaussians_from_triangles(
        vertices=vertices,
        triangles=triangles,
        triangle_colors=colors,
        target_disk_radius_m=0.035,
    )

    # Reconstruct RGB from sh0
    sh0_np = splats["sh0"].numpy()[:, 0, :]
    reconstructed_rgb = sh0_np * SH_C0 + 0.5
    # Must NOT be divided by 255 (which would yield ~0.0039, 0.0031, 0.0019)
    assert np.all(reconstructed_rgb[:, 0] > 0.95)
    assert np.allclose(reconstructed_rgb[:, 1], 0.8, atol=1e-3)
    assert np.allclose(reconstructed_rgb[:, 2], 0.5, atol=1e-3)


def test_strict_gaussian_budget_limit(tmp_path):
    """Verify that sealing strictly respects the max 10% gaussian ceiling even on huge infill areas."""
    model_dir = tmp_path / "gaussian_splat"
    checkpoint_file = model_dir / "checkpoint.pt"
    # 200 original gaussians -> 10% ceiling is 20
    create_synthetic_checkpoint(checkpoint_file, n_gaussians=200)

    # Create a huge infill area: 100m x 100m
    v = np.array([
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
        [100.0, 100.0, 0.0],
        [0.0, 100.0, 0.0],
    ], dtype=np.float64)
    # 1 clean triangle, 1 infill triangle (each 5,000 m2!)
    t = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int32)
    c = np.full((len(v), 3), 0.5)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(t)
    mesh.vertex_colors = o3d.utility.Vector3dVector(c)

    clean_num_triangles = 1
    infill_colors = np.array([[0.7, 0.5, 0.3]], dtype=np.float64)

    stats = seal_gsplat_model(
        gsplat_model_path=model_dir,
        sealed_mesh=mesh,
        clean_num_triangles=clean_num_triangles,
        infill_colors=infill_colors,
        max_gsplat_ratio=0.10,
    )

    assert stats["original_gaussians"] == 200
    # Strict 10% ceiling: added gaussians must be <= 20
    assert stats["infill_gaussians_added"] <= 20
    assert stats["infill_gaussians_added"] > 0
    assert stats["total_sealed_gaussians"] <= 220


def test_seal_gsplat_coverage_deficit(tmp_path):
    """Verify that seal_gsplat_model detects coverage deficits on clean TSDF triangles and seals them."""
    model_dir = tmp_path / "gaussian_splat"
    checkpoint_file = model_dir / "checkpoint.pt"
    # 50 gaussians near origin
    create_synthetic_checkpoint(checkpoint_file, n_gaussians=50)

    # 1 clean triangle far away at [10, 10, 10]
    v = np.array([
        [10.0, 10.0, 10.0],
        [10.5, 10.0, 10.0],
        [10.0, 10.5, 10.0],
    ], dtype=np.float64)
    t = np.array([[0, 1, 2]], dtype=np.int32)
    c = np.full((len(v), 3), 0.3)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(t)
    mesh.vertex_colors = o3d.utility.Vector3dVector(c)

    # All triangles are TSDF clean triangles; 0 boundary infill triangles
    clean_num_triangles = 1

    stats = seal_gsplat_model(
        gsplat_model_path=model_dir,
        sealed_mesh=mesh,
        clean_num_triangles=clean_num_triangles,
        coverage_deficit_threshold_m=0.06,
        max_gsplat_ratio=0.10,
    )

    assert stats["original_gaussians"] == 50
    assert stats["deficit_triangles"] == 1
    assert stats["infill_triangles"] == 0
    assert stats["infill_gaussians_added"] > 0
    assert stats["infill_gaussians_added"] <= 5
    assert stats["total_sealed_gaussians"] == 50 + stats["infill_gaussians_added"]


