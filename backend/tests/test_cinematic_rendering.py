from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '08_artifacts_generation'))
from cinematic_rendering import GaussianRenderer, GPUMeshRenderer, MeshRenderer, OpticalMeshRenderer, intrinsics, pinhole_directions


def test_fixed_horizontal_fov_and_camera_pose():
    k = intrinsics(200, 400, 90)
    assert k[0, 0] == pytest.approx(100)
    pose = np.eye(4)
    directions = pinhole_directions(pose, 200, 400, 90)
    assert directions[199, 99] == pytest.approx([-.005, -.005, 1])
    pose[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    assert pinhole_directions(pose, 200, 400, 90)[199, 99] == pytest.approx([1, -.005, .005])


def test_mesh_renderer_uses_same_pinhole_axes():
    mesh = o3d.geometry.TriangleMesh.create_box(width=4, height=4, depth=.1)
    mesh.translate([-2, -2, 2])
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    colors = np.tile(np.array([20, 40, 80], dtype=np.uint8), (len(vertices), 1))
    image = MeshRenderer(vertices, faces, colors).render(np.eye(4), 8, 8, 60)
    np.testing.assert_array_equal(image[4, 4], [20, 40, 80])


def test_gpu_mesh_renderer_matches_cpu_colors_and_camera():
    import torch
    if not torch.cuda.is_available():
        pytest.skip('GPU renderer requires a GPU')
    mesh = o3d.geometry.TriangleMesh.create_box(width=4, height=4, depth=.1)
    mesh.translate([-2, -2, 2])
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    colors = np.tile(np.array([20, 40, 80], dtype=np.uint8), (len(vertices), 1))
    cpu = MeshRenderer(vertices, faces, colors).render(np.eye(4), 64, 64)
    gpu = GPUMeshRenderer(vertices, faces, colors, 64, 64).render(np.eye(4), 64, 64)
    np.testing.assert_allclose(gpu, cpu, atol=1)


def test_gpu_mesh_does_not_clip_a_nearby_door_surface():
    import torch
    if not torch.cuda.is_available():
        pytest.skip('GPU renderer requires a GPU')
    mesh = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=.02)
    mesh.translate([-.5, -.5, .05])
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    colors = np.tile(np.array([80, 40, 20], dtype=np.uint8), (len(vertices), 1))
    image = GPUMeshRenderer(vertices, faces, colors, 32, 32).render(np.eye(4), 32, 32)
    np.testing.assert_allclose(image[16, 16], [80, 40, 20], atol=1)


def test_classified_panels_reflect_and_transmit_scene_geometry():
    import torch
    if not torch.cuda.is_available():
        pytest.skip('GPU renderer requires a GPU')
    vertices = np.array([
        [-2,-2,2], [0,-2,2], [0,2,2], [-2,2,2],
        [0,-2,2], [2,-2,2], [2,2,2], [0,2,2],
        [-5,-5,-2], [5,-5,-2], [5,5,-2], [-5,5,-2],
        [-5,-5,4], [5,-5,4], [5,5,4], [-5,5,4],
    ], dtype=np.float32)
    faces = np.array([[i,i+1,i+2] for i in (0,4,8,12)] +
                     [[i,i+2,i+3] for i in (0,4,8,12)])
    colors = np.array([[100,100,100]]*8 + [[200,20,20]]*4 + [[20,40,200]]*4, np.uint8)
    image = OpticalMeshRenderer(vertices, faces, colors, [1,2,0,0,1,2,0,0], 32, 32).render(np.eye(4), 32, 32)
    np.testing.assert_allclose(image[16, 8], [200, 20, 20], atol=1)
    np.testing.assert_allclose(image[16, 24], [35, 45, 189], atol=2)


def test_gaussian_source_frame_color_matches_gsplat():
    import torch
    if not torch.cuda.is_available():
        pytest.skip('Gaussian renderer requires GPU')
    from gsplat import rasterization
    device = 'cuda'
    means = torch.tensor([[0., 0., 3.], [.4, 0., 3.2]], device=device)
    quats = torch.tensor([[1., 0., 0., 0.]] * 2, device=device)
    scales = torch.full((2, 3), .2, device=device)
    opacity = torch.tensor([.8, .9], device=device)
    sh = torch.tensor([[[.2, .1, -.1]], [[-.1, .2, .3]]], device=device)
    renderer = GaussianRenderer.__new__(GaussianRenderer)
    renderer.device = device
    renderer.models = [SimpleNamespace(world_to_source=np.eye(4, dtype=np.float32), sh_degree=0,
                                       splats={'means': means, 'colors': sh})]
    renderer.world = {'means': means, 'quats': quats, 'scales': scales, 'opacities': opacity}
    actual = renderer.render(np.eye(4, dtype=np.float32), 32, 48)
    k = torch.tensor(intrinsics(32, 48)[None], device=device)
    expected, _, _ = rasterization(**renderer.world, colors=sh, viewmats=torch.eye(4, device=device)[None],
                                    Ks=k, width=32, height=48, sh_degree=0, tile_size=16,
                                    packed=False, render_mode='RGB', backgrounds=torch.ones((1, 3), device=device))
    expected = (expected[0].clamp(0, 1) * 255).round().byte().cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=1)


def test_rotated_scan_keeps_source_frame_gaussian_appearance():
    import torch
    if not torch.cuda.is_available():
        pytest.skip('Gaussian renderer requires GPU')
    from gsplat import rasterization
    device = 'cuda'
    source_means = torch.tensor([[0., 0., 3.], [.4, .1, 3.2]], device=device)
    source_quats = torch.tensor([[1., 0., 0., 0.]] * 2, device=device)
    scales = torch.tensor([[.15, .2, .3], [.2, .1, .3]], device=device)
    opacities = torch.tensor([.8, .9], device=device)
    sh = torch.tensor([[[.2, .1, -.1], [.3, -.2, .1], [.1, .2, .3], [-.2, .1, .2]]] * 2, device=device)
    transform = np.array([[0., 0., 1., 1.], [0., 1., 0., .2], [-1., 0., 0., -.3], [0., 0., 0., 1.]], dtype=np.float32)
    rotation = torch.tensor(transform[:3, :3], device=device)
    world_means = source_means @ rotation.T + torch.tensor(transform[:3, 3], device=device)
    world_quats = torch.tensor([[2**-.5, 0., 2**-.5, 0.]] * 2, device=device)
    renderer = GaussianRenderer.__new__(GaussianRenderer)
    renderer.device = device
    renderer.models = [SimpleNamespace(world_to_source=np.linalg.inv(transform), sh_degree=1,
                                       splats={'means': source_means, 'colors': sh})]
    renderer.world = {'means': world_means, 'quats': world_quats, 'scales': scales, 'opacities': opacities}
    actual = renderer.render(transform, 32, 48)
    reference, _, _ = rasterization(means=source_means, quats=source_quats, scales=scales, opacities=opacities,
                                     colors=sh, viewmats=torch.eye(4, device=device)[None],
                                     Ks=torch.tensor(intrinsics(32, 48)[None], device=device),
                                     width=32, height=48, sh_degree=1, tile_size=16, packed=False,
                                     render_mode='RGB', backgrounds=torch.ones((1, 3), device=device))
    expected = (reference[0].clamp(0, 1) * 255).round().byte().cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=2)


def test_unified_gaussian_renderer_matches_source_frame_appearance():
    import torch
    if not torch.cuda.is_available():
        pytest.skip('Gaussian renderer requires GPU')
    from gsplat import rasterization
    device = 'cuda'
    source_means = torch.tensor([[0., 0., 3.], [.4, .1, 3.2]], device=device)
    source_quats = torch.tensor([[1., 0., 0., 0.]] * 2, device=device)
    scales = torch.tensor([[.15, .2, .3], [.2, .1, .3]], device=device)
    opacities = torch.tensor([.8, .9], device=device)
    sh = torch.tensor([[[.2, .1, -.1], [.3, -.2, .1], [.1, .2, .3], [-.2, .1, .2]]] * 2, device=device)
    transform = np.array([[0., 0., 1., 1.], [0., 1., 0., .2], [-1., 0., 0., -.3], [0., 0., 0., 1.]], dtype=np.float32)
    rotation = torch.tensor(transform[:3, :3], device=device)
    world_means = source_means @ rotation.T + torch.tensor(transform[:3, 3], device=device)
    world_quats = torch.tensor([[2**-.5, 0., 2**-.5, 0.]] * 2, device=device)
    renderer = GaussianRenderer.__new__(GaussianRenderer)
    renderer.device = device
    renderer.world = {'means': world_means, 'quats': world_quats, 'scales': scales, 'opacities': opacities}
    renderer.colors = sh
    renderer.sh_degree = 1
    renderer.scan_slices = [(0, 2, rotation)]
    actual = renderer.render(transform, 32, 48)
    reference, _, _ = rasterization(means=source_means, quats=source_quats, scales=scales, opacities=opacities,
                                     colors=sh, viewmats=torch.eye(4, device=device)[None],
                                     Ks=torch.tensor(intrinsics(32, 48)[None], device=device),
                                     width=32, height=48, sh_degree=1, tile_size=16, packed=False,
                                     render_mode='RGB', backgrounds=torch.ones((1, 3), device=device))
    expected = (reference[0].clamp(0, 1) * 255).round().byte().cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=2)

