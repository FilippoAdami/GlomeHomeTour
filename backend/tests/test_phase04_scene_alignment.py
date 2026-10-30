import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

STAGE04 = Path(__file__).resolve().parents[1] / "04_3DGS_to_mesh"
STAGE03 = Path(__file__).resolve().parents[1] / "03_FastGS_DNSplatter"
sys.path[:0] = [str(STAGE04), str(STAGE03)]
from scene_alignment import (estimate_alignment, transform_mesh_ply, transform_splats,
                             _transform_colmap)


def _box_mesh():
    vertices, triangles = [], []

    def quad(points):
        start = len(vertices)
        vertices.extend(points)
        triangles.extend(((start, start + 1, start + 2), (start, start + 2, start + 3)))

    x0, x1, y0, y1, z0, z1 = 0., 4., 2., 5., 0., 3.
    quad(((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)))
    quad(((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)))
    quad(((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)))
    quad(((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)))
    quad(((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)))
    quad(((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)))
    return np.asarray(vertices), np.asarray(triangles)


def _write_image_pose(path: Path, world_to_camera: np.ndarray):
    q = Rotation.from_matrix(world_to_camera).as_quat()
    path.write_text(f"1 {q[3]} {q[0]} {q[1]} {q[2]} 0 0 0 1 room.png\n1 2 -1\n")


def test_mesh_alignment_levels_walls_and_uses_supported_floor(tmp_path):
    vertices, triangles = _box_mesh()
    source_rotation = Rotation.from_euler("xyz", [6., -4., 27.], degrees=True).as_matrix()
    source_translation = np.array([3., -2., 1.])
    source_vertices = vertices @ source_rotation.T + source_translation
    camera_c2w = source_rotation @ np.diag([1., -1., -1.])
    images = tmp_path / "images.txt"
    _write_image_pose(images, camera_c2w.T)

    result = estimate_alignment(source_vertices, triangles, images)
    transform = np.asarray(result["transform_matrix"])
    aligned = source_vertices @ transform[:3, :3].T + transform[:3, 3]
    aligned_normals = np.cross(
        aligned[triangles[:, 1]] - aligned[triangles[:, 0]],
        aligned[triangles[:, 2]] - aligned[triangles[:, 0]],
    )
    aligned_normals /= np.maximum(np.linalg.norm(aligned_normals, axis=1, keepdims=True), 1e-12)

    assert result["up_source"] == "mesh wall-normal families"
    assert result["floor_y_after_world_units"] == pytest.approx(0., abs=1e-8)
    assert np.min(aligned[:, 1]) == pytest.approx(0., abs=1e-8)
    wall_normals = aligned_normals[[4, 5, 6, 7, 8, 9, 10, 11]]
    assert np.all(np.max(np.abs(wall_normals[:, [0, 2]]), axis=1) > .999)


def test_colmap_pose_and_sparse_points_receive_same_world_transform(tmp_path):
    rotation = Rotation.from_euler("xyz", [8., 3., -17.], degrees=True).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [1.2, -0.8, 2.1]
    old_rotation = Rotation.from_euler("xyz", [-2., 7., 41.], degrees=True).as_matrix()
    old_q = Rotation.from_matrix(old_rotation).as_quat()
    images = tmp_path / "images.txt"
    points = tmp_path / "points3D.txt"
    output = tmp_path / "aligned"
    images.write_text(f"1 {old_q[3]} {old_q[0]} {old_q[1]} {old_q[2]} 3 4 5 2 room image.png\n1 2 -1\n")
    points.write_text("8 2 3 4 20 30 40 0.2 1 0 1\n")

    _transform_colmap(images, points, output, transform)
    fields = (output / "images.txt").read_text().splitlines()[0].split(maxsplit=9)
    new_q = np.asarray(fields[1:5], float)
    new_rotation = Rotation.from_quat([new_q[1], new_q[2], new_q[3], new_q[0]]).as_matrix()
    new_translation = np.asarray(fields[5:8], float)
    np.testing.assert_allclose(new_rotation, old_rotation @ rotation.T, atol=1e-12)
    np.testing.assert_allclose(new_translation, [3, 4, 5] - new_rotation @ transform[:3, 3], atol=1e-12)
    np.testing.assert_allclose(np.asarray((output / "points3D.txt").read_text().split()[1:4], float),
                               rotation @ [2, 3, 4] + transform[:3, 3], atol=1e-12)
    assert (output / "images.txt").read_text().splitlines()[1] == "1 2 -1"


def test_mesh_transform_preserves_face_flags_and_rotates_vertices(tmp_path):
    from fill_mesh_holes import export_ply_with_face_attributes, read_ply_face_flags

    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 0., 1.], [0., 0., 1.]])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    path = tmp_path / "mesh.ply"
    assert export_ply_with_face_attributes(
        path, vertices, triangles, np.full((4, 3), .5),
        {"is_hole_fill": ("uchar", np.array([0, 1], dtype=np.uint8))},
    )
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("z", 12., degrees=True).as_matrix()
    transform[:3, 3] = [1., 2., 3.]

    transform_mesh_ply(path, transform)
    from plyfile import PlyData
    ply = PlyData.read(str(path))
    data = ply["vertex"].data
    np.testing.assert_allclose(np.column_stack([data[name] for name in ("x", "y", "z")]),
                               vertices @ transform[:3, :3].T + transform[:3, 3], atol=1e-6)
    np.testing.assert_array_equal(read_ply_face_flags(path), [0, 1])


def test_sealed_gaussian_transform_preserves_view_dependent_color(tmp_path):
    rng = np.random.default_rng(4)
    splats = {
        "means": torch.tensor(rng.normal(size=(2, 3)), dtype=torch.float32),
        "quats": torch.tensor([[1., 0., 0., 0.], [.8, .1, .2, .3]], dtype=torch.float32),
        "scales": torch.full((2, 3), -2.),
        "opacities": torch.zeros(2),
        "sh0": torch.tensor(rng.normal(size=(2, 1, 3)), dtype=torch.float32),
        "shN": torch.tensor(rng.normal(size=(2, 15, 3)), dtype=torch.float32),
    }
    checkpoint = tmp_path / "checkpoint.pt"
    ply = tmp_path / "gaussians.ply"
    torch.save({"format": "gsplat-stage03-v1", "splats": splats}, checkpoint)
    old_means = splats["means"].clone()
    old_quats = splats["quats"].clone()
    old_sh = torch.cat((splats["sh0"], splats["shN"]), dim=1).numpy()
    rotation = Rotation.from_euler("xyz", [7., -11., 16.], degrees=True).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [2., -1., .5]

    transform_splats(checkpoint, ply, transform)
    updated = torch.load(checkpoint, map_location="cpu", weights_only=True)["splats"]
    np.testing.assert_allclose(updated["means"].numpy(), old_means.numpy() @ rotation.T + transform[:3, 3], atol=1e-6)
    old_rotation = Rotation.from_quat(old_quats[1, [1, 2, 3, 0]].numpy())
    delta = Rotation.from_matrix(rotation)
    expected_quaternion = (delta * old_rotation).as_matrix()
    actual_quaternion = Rotation.from_quat(updated["quats"][1, [1, 2, 3, 0]].numpy()).as_matrix()
    np.testing.assert_allclose(actual_quaternion, expected_quaternion, atol=1e-6)

    dirs = rng.normal(size=(32, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    from gsplat.cuda._torch_impl import _eval_sh_bases_fast
    old_basis = _eval_sh_bases_fast(16, torch.tensor(dirs @ rotation, dtype=torch.float64)).numpy()
    new_basis = _eval_sh_bases_fast(16, torch.tensor(dirs, dtype=torch.float64)).numpy()
    old_color = old_basis @ old_sh[0]
    new_color = new_basis @ torch.cat((updated["sh0"], updated["shN"]), dim=1)[0].numpy()
    np.testing.assert_allclose(new_color, old_color, atol=2e-6)
    assert ply.is_file()
