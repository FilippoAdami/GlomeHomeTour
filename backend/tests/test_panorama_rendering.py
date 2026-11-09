from pathlib import Path
import sys

import numpy as np
import open3d as o3d
import pytest
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "08_artifacts_generation"))
from panorama_rendering import (
    GaussianPanoramaModel, _cube_to_equirectangular, cube_face_axes,
    equirectangular_directions, inspect_panel_materials, load_gaussians, render_gaussian_panorama,
    render_mesh_panorama,
)


def test_equirectangular_directions_have_requested_world_orientation():
    directions = equirectangular_directions(360, 180)
    # Pixel centres: closest cardinal directions retain the documented order.
    equator = directions[90]
    assert np.argmax(equator @ np.array([1., 0., 0.])) in range(179, 181)
    assert np.argmax(equator @ np.array([0., 0., 1.])) in range(269, 271)
    assert np.argmax(equator @ np.array([-1., 0., 0.])) in list(range(0, 1)) + list(range(359, 360))
    assert np.argmax(equator @ np.array([0., 0., -1.])) in range(89, 91)


def test_cube_faces_are_right_handed_and_point_at_cardinal_directions():
    for right, down, forward in cube_face_axes().values():
        assert np.allclose(np.cross(right, down), forward)
        assert np.isclose(np.linalg.det(np.stack((right, down, forward))), 1.0)


def test_mesh_panorama_interpolates_hit_triangle_colour():
    vertices = np.array([[-1., -1., 2.], [1., -1., 2.], [0., 1., 2.]], np.float32)
    faces = np.array([[0, 1, 2]], np.uint32)
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], np.uint8)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(faces))
    image, valid = render_mesh_panorama(scene, vertices, faces, colors, np.zeros(3), width=128)
    assert valid.any()
    hit_colors = image[valid]
    assert np.any(hit_colors[:, 0] > 0) and np.any(hit_colors[:, 1] > 0) and np.any(hit_colors[:, 2] > 0)


def test_gaussian_loader_restores_aligned_ply_to_source_coordinates(tmp_path):
    dtype = [(name, "f4") for name in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                                         "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")]
    vertex = np.zeros(1, dtype=dtype)
    vertex["x"], vertex["y"], vertex["z"] = 11., 2., 3.
    vertex["rot_0"] = 1.
    path = tmp_path / "aligned.ply"
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)
    transform = np.eye(4)
    transform[0, 3] = 10.
    model = load_gaussians(path, transform, device="cpu")
    assert np.allclose(model.splats["means"].cpu().numpy(), [[1., 2., 3.]])
    assert str(model.splats["means"].dtype) == "torch.float32"


def test_cube_stitch_handles_rgb_alpha_and_large_opencv_maps():
    colors = {"+x": [1., 0., 0.], "-x": [0., 1., 0.], "+z": [0., 0., 1.],
              "-z": [1., 1., 0.], "+y": [1., 0., 1.], "-y": [0., 1., 1.]}
    cube = {name: np.broadcast_to(color, (512, 512, 3)).astype(np.float32) for name, color in colors.items()}
    panorama = _cube_to_equirectangular(cube, 2048)
    assert np.array_equal(panorama[512, 1024], [255, 0, 0])   # +X
    assert np.array_equal(panorama[512, 1536], [0, 0, 255])   # +Z
    assert np.array_equal(panorama[512, 512], [255, 255, 0])  # -Z
    assert np.array_equal(panorama[512, 0], [0, 255, 0])      # -X seam
    assert np.array_equal(panorama[512, -1], [0, 255, 0])
    alpha_cube = {name: np.full((512, 512, 1), value, np.float32)
                  for value, name in enumerate(colors, 1)}
    alpha = _cube_to_equirectangular(alpha_cube, 2048, uint8=False)
    assert alpha.shape == (1024, 2048, 1)
    assert np.isclose(alpha[512, 1024, 0], 1.)
    assert np.isclose(alpha[512, 1536, 0], 3.)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="gsplat smoke test needs CUDA/ROCm")
def test_gaussian_panorama_gpu_smoke_render():
    import torch

    device = "cuda"
    splats = {
        "means": torch.tensor([[2., 0., 0.]], device=device),
        "quats": torch.tensor([[1., 0., 0., 0.]], device=device),
        "scales": torch.tensor([[.25, .25, .25]], device=device),
        "opacities": torch.tensor([.99], device=device),
        "colors": torch.tensor([[[1., 0., 0.]]], device=device),
    }
    model = GaussianPanoramaModel(splats, 0, np.eye(4, dtype=np.float32), device)
    rgb, alpha = render_gaussian_panorama(model, np.zeros(3), width=64)
    assert rgb.shape == (32, 64, 3) and alpha.shape == (32, 64)
    assert np.isfinite(alpha).all() and alpha.max() > 0


def test_panel_material_report_keeps_vertex_colours_when_glb_has_no_material(tmp_path):
    import trimesh

    mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False)
    mesh.visual.vertex_colors = [[192, 192, 192, 255]] * 3
    scene = trimesh.Scene({"sealed_scan_9": mesh})
    scene.export(tmp_path / "aligned_sealed_meshes.glb")
    report = inspect_panel_materials(tmp_path, "scan_9")
    assert report["status"] == "vertex_colors_only"
    assert report["optical_effects"].startswith("unavailable")
    assert len(report["glb_sha256"]) == 64


def test_panel_material_report_handles_missing_glb(tmp_path):
    report = inspect_panel_materials(tmp_path, "scan_0")
    assert report["status"] == "glb_missing"
