import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
import pytest
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "08_artifacts_generation"))
from panorama_rendering import (
    GaussianPanoramaModel, _cube_to_equirectangular, cube_face_axes,
    equirectangular_directions, inspect_panel_materials, load_gaussians, panel_faces, render_gaussian_panorama,
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


def test_mesh_panorama_reflects_mirrors_and_transmits_windows():
    vertices = np.array([
        [2, -1, -1], [2, 1, -1], [2, 1, 1], [2, -1, 1],
        [-2, -4, -4], [-2, 4, -4], [-2, 4, 4], [-2, -4, 4],
        [4, -4, -4], [4, 4, -4], [4, 4, 4], [4, -4, 4],
    ], np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
                      [8, 9, 10], [8, 10, 11]], np.uint32)
    colors = np.array([[160, 160, 160]] * 4 + [[255, 0, 0]] * 4 + [[0, 0, 255]] * 4, np.uint8)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(faces))
    secondary = o3d.t.geometry.RaycastingScene()
    secondary.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(faces[2:]))
    baseline, expected_valid = render_mesh_panorama(scene, vertices, faces, colors, np.zeros(3), width=128)
    directions = equirectangular_directions(128).reshape(-1, 3)
    rays = np.column_stack((np.zeros_like(directions), directions)).astype(np.float32)
    primary = scene.cast_rays(o3d.core.Tensor(rays))["primitive_ids"].numpy().reshape(64, 128)
    for kind, expected in ((1, [255, 0, 0]), (2, [20, 8, 238])):
        types = np.array([kind, kind, 0, 0, 0, 0], np.uint8)
        stats = {}
        image, valid = render_mesh_panorama(scene, vertices, faces, colors, np.zeros(3), width=128,
                                             panel_optics=(types, secondary, faces[2:], .001), optical_stats=stats)
        assert np.array_equal(image[32, 64], expected)
        assert np.array_equal(valid, expected_valid)
        assert np.array_equal(image[primary >= 2], baseline[primary >= 2])
        assert stats["mirror_pixels" if kind == 1 else "window_pixels"] > 0
        assert stats["mirror_reflected_hits" if kind == 1 else "window_transmitted_hits"] > 0
    assert stats["window_reflected_hits"] > 0


def test_window_uses_white_environment_when_transmission_misses_even_if_reflection_hits():
    vertices = np.array([[2, -1, -1], [2, 1, -1], [2, 1, 1], [2, -1, 1]], np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
    colors = np.full((4, 3), 160, np.uint8)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(faces))
    reflection_vertices = np.array([[-2, -1, -1], [-2, 1, -1], [-2, 1, 1], [-2, -1, 1]], np.float32)
    reflection_faces = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
    reflection_scene = o3d.t.geometry.RaycastingScene()
    reflection_scene.add_triangles(o3d.core.Tensor(reflection_vertices), o3d.core.Tensor(reflection_faces))
    reflection_colors = np.tile([255, 0, 0], (4, 1)).astype(np.uint8)
    stats = {}
    image, _ = render_mesh_panorama(scene, vertices, faces, colors, np.zeros(3), width=128,
                                     panel_optics=(np.array([2, 2]), reflection_scene, reflection_faces, .001),
                                     optical_stats=stats, panel_optical_colors=reflection_colors)
    assert np.array_equal(image[32, 64], [250, 238, 238])
    assert stats["window_transmitted_hits"] == 0
    assert stats["window_reflected_hits"] > 0


def test_panel_face_classification_rejects_mismatched_geometry():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
    regions = [{"scan_id": "scan_2", "class": "mirror", "sealed_panels": [{"plane_corners": vertices.tolist()}]}]
    kinds, panel_ids = panel_faces("scan_2", vertices, faces, regions)
    assert np.array_equal(kinds, [1, 1]) and np.array_equal(panel_ids, [0, 0])
    with pytest.raises(ValueError, match="geometry"):
        panel_faces("scan_2", vertices + 1, faces, regions)


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


def test_panel_material_report_identifies_vertex_colours(tmp_path):
    import trimesh

    mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False)
    mesh.visual.vertex_colors = [[192, 192, 192, 255]] * 3
    scene = trimesh.Scene({"sealed_scan_9": mesh})
    scene.export(tmp_path / "aligned_sealed_meshes.glb")
    report = inspect_panel_materials(tmp_path, "scan_9")
    assert report["status"] == "vertex_colors_only"
    assert report["optical_effects"] == "no classified panels"
    assert len(report["glb_sha256"]) == 64


def test_panel_material_report_handles_missing_glb(tmp_path):
    report = inspect_panel_materials(tmp_path, "scan_0")
    assert report["status"] == "glb_missing"


def test_panel_material_report_records_idealized_optics_and_manifest_hash(tmp_path):
    stats = tmp_path / "stats"
    stats.mkdir()
    (stats / "phase07_region_replacements.json").write_text(json.dumps({"regions": [
        {"scan_id": "scan_0", "class": "window", "sealed_panels": [{}]},
    ]}))
    report = inspect_panel_materials(tmp_path, "scan_0")
    assert report["sealed_panel_count"] == 1
    assert report["optical_effects"] == "idealized one-bounce reflection and transmission"
    assert len(report["panel_manifest_sha256"]) == 64
