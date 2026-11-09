"""Text edits must not leave COLMAP readers using old binary or rig poses."""

from pathlib import Path
import sys

import numpy as np
import pycolmap
import pytest
from plyfile import PlyData
from scipy.spatial.transform import Rotation

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from Utilities.pipeline_paths import bootstrap

bootstrap()
import step_colmap
import step_mesh
from scene_alignment import _transform_colmap
from scene.dataset_readers import storePly


def make_model(path):
    path.mkdir(parents=True)
    xyz = np.array([[0., 0., 3.], [.2, .1, 4.]])
    views, lines = [], []
    (path / "cameras.txt").write_text("1 PINHOLE 100 80 100 100 50 40\n")
    for image_id, x in ((1, 0.), (2, -.5)):
        view = np.eye(4)
        view[0, 3] = x
        views.append(view)
        lines.append(f"{image_id} 1 0 0 0 {x} 0 0 1 photo_{image_id}.jpg")
        camera = xyz + view[:3, 3]
        xy = camera[:, :2] / camera[:, 2:] * 100 + [50, 40]
        lines.append(" ".join(f"{u} {v} {i+1}" for i, (u, v) in enumerate(xy)))
    (path / "images.txt").write_text("\n".join(lines) + "\n")
    (path / "points3D.txt").write_text("\n".join(
        f"{i+1} {x} {y} {z} 120 100 80 0 1 {i} 2 {i}" for i, (x, y, z) in enumerate(xyz)) + "\n")
    recon = pycolmap.Reconstruction(str(path))
    recon.write_text(str(path))
    recon.write_binary(str(path))
    storePly(str(path / "points3D.ply"), xyz, np.full_like(xyz, 128))
    return xyz, views


def assert_model(path, xyz, views):
    binary_preferred = pycolmap.Reconstruction(str(path))
    text = pycolmap.Reconstruction()
    text.read_text(str(path))
    for recon in (binary_preferred, text):
        assert set(recon.images) == {1, 2}
        for image_id, view in enumerate(views, 1):
            image = recon.images[image_id]
            np.testing.assert_allclose(image.cam_from_world().matrix(), view[:3], atol=1e-9)
            assert image.name == f"photo_{image_id}.jpg"
            for observation in image.points2D:
                camera = view[:3, :3] @ recon.points3D[observation.point3D_id].xyz + view[:3, 3]
                uv = camera[:2] / camera[2] * 100 + [50, 40]
                np.testing.assert_allclose(uv, observation.xy, atol=1e-8)
        np.testing.assert_allclose([recon.points3D[i].xyz for i in (1, 2)], xyz, atol=1e-10)


def world_transform():
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [17., -28., 9.], degrees=True).as_matrix()
    transform[:3, 3] = [1.2, -.8, 2.1]
    return transform


@pytest.mark.parametrize("scale", [1., 1.7])
def test_phase01_rebuilds_binary_and_frame_poses_from_edited_text(tmp_path, scale):
    sparse = tmp_path / "sparse" / "0"
    xyz, views = make_model(sparse)
    step_colmap.scale_colmap_model(sparse, scale)
    for view in views:
        view[:3, 3] *= scale
    transform = world_transform()
    step_colmap.transform_colmap_model(sparse, transform[:3, :3], transform[:3, 3],
                                      xyz * scale, np.full_like(xyz, 128))
    assert_model(sparse, xyz * scale @ transform[:3, :3].T + transform[:3, 3],
                 [view @ np.linalg.inv(transform) for view in views])


@pytest.mark.parametrize("failure", [None, "depth", "invalidation"])
def test_phase04_invalidates_derived_models_and_preserves_rollback(tmp_path, monkeypatch, failure):
    sparse = tmp_path / "sparse" / "0"
    xyz, views = make_model(sparse)
    (sparse / "keep.dat").write_bytes(b"unrelated")
    before = {p.name: p.read_bytes() for p in sparse.iterdir()}
    staged = tmp_path / "staged"
    transform = world_transform()
    _transform_colmap(sparse / "images.txt", sparse / "points3D.txt", staged, transform)
    if failure:
        def fail(*args):
            raise RuntimeError("publication failure")
        if failure == "depth":
            monkeypatch.setattr(step_mesh, "transform_depth_poses_in_place", fail)
        else:
            unlink = Path.unlink

            def fail_after_first_invalidation(path, *args, **kwargs):
                if path == sparse / "images.bin":
                    fail()
                return unlink(path, *args, **kwargs)
            monkeypatch.setattr(Path, "unlink", fail_after_first_invalidation)
        with pytest.raises(RuntimeError, match="publication failure"):
            step_mesh.publish_aligned_poses(staged, sparse, tmp_path / "depth.npz", transform)
        assert {p.name: p.read_bytes() for p in sparse.iterdir()} == before
        return
    step_mesh.publish_aligned_poses(staged, sparse, tmp_path / "depth.npz", transform)
    moved = xyz @ transform[:3, :3].T + transform[:3, 3]
    assert_model(sparse, moved, [view @ np.linalg.inv(transform) for view in views])
    ply = PlyData.read(sparse / "points3D.ply")["vertex"]
    np.testing.assert_allclose(np.column_stack([ply[axis] for axis in "xyz"]), moved, atol=1e-6)
    assert (sparse / "keep.dat").read_bytes() == b"unrelated"
