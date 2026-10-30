"""Regression checks for the evidence retained by Stage 05 projection."""
from pathlib import Path
import sys

import numpy as np
import pytest


STAGE = Path(__file__).resolve().parents[1] / "05_semantic_segmentation"
sys.path.insert(0, str(STAGE))
from step2_project_masks_to_3d import (  # noqa: E402
    local_evidence_labels,
    normalized_incidence_cosines,
    reduce_view_face_votes,
)
import scene_inputs  # noqa: E402


def test_view_vote_is_invariant_to_pixel_coverage():
    one_pixel, one_weight = reduce_view_face_votes(
        np.array([0]), np.array([1]), np.array([0.75]), num_triangles=2, num_classes=3
    )
    many_pixels, many_weight = reduce_view_face_votes(
        np.zeros(12, dtype=np.int64), np.ones(12, dtype=np.int64),
        np.full(12, 0.75, dtype=np.float32), num_triangles=2, num_classes=3,
    )
    np.testing.assert_array_equal(one_pixel, many_pixels)
    np.testing.assert_allclose(one_weight, many_weight)
    assert one_weight[0] <= 1.0

    # Unknown/background pixels still bound the active class's evidence for
    # this face and view; they add no class evidence of their own.
    _, partial_weight = reduce_view_face_votes(
        np.array([0]), np.array([1]), np.array([0.75]), num_triangles=2, num_classes=3,
        all_face_hit_ids=np.zeros(100, dtype=np.int64),
    )
    np.testing.assert_allclose(partial_weight, [0.0075])


def test_incidence_is_finite_and_independent_of_ray_length():
    normals = np.array([[[0.0, 0.0, 2.0]], [[0.0, 0.0, 0.0]]])
    rays = np.array([[[0.0, 0.0, -10.0]], [[np.inf, 0.0, 0.0]]])
    cosines = normalized_incidence_cosines(normals, rays)
    assert np.isfinite(cosines).all()
    np.testing.assert_allclose(cosines, [[1.0], [0.0]])


def test_raw_labels_remain_the_local_argmax_without_global_class_flip():
    # Class 1 may win many other faces; it must not overturn face 0's local vote.
    votes = np.array([[0.60, 0.55], [0.00, 1.00], [0.00, 1.00]], dtype=np.float32)
    np.testing.assert_array_equal(local_evidence_labels(votes), [0, 1, 1])


def test_local_evidence_handles_one_and_zero_classes():
    np.testing.assert_array_equal(local_evidence_labels(np.array([[0.2], [0.0]])), [0, -1])
    np.testing.assert_array_equal(local_evidence_labels(np.zeros((3, 0))), [-1, -1, -1])


def test_projection_label_archive_requires_camera_shape_and_active_classes():
    camera = {"height": 2, "width": 3}
    archive_labels = np.array([[1, 1, 0], [2, 2, 0]], dtype=np.int32)
    assert scene_inputs.validate_labels(archive_labels, camera, classes=[1, 2]) is archive_labels
    with pytest.raises(ValueError, match="allowed set"):
        scene_inputs.validate_labels(archive_labels, camera, classes=[1])
    with pytest.raises(ValueError, match="height and width"):
        scene_inputs.validate_labels(archive_labels[:, :2], camera, classes=[1, 2])


def test_real_tiny_projection_without_summary_publishes_numpy_source_ids(tmp_path):
    import json
    import open3d as o3d
    from PIL import Image
    from scipy.sparse import load_npz
    from step2_project_masks_to_3d import project_masks_to_3d

    scene = tmp_path / 'scene'
    (scene / 'images').mkdir(parents=True)
    (scene / 'sparse/0').mkdir(parents=True)
    (scene / 'sparse/0/cameras.txt').write_text('1 PINHOLE 8 6 4 4 4 3\n')
    (scene / 'sparse/0/images.txt').write_text('1 1 0 0 0 0 0 0 1 image.png\n\n')
    Image.new('RGB', (8, 6)).save(scene / 'images/image.png')
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(
        [[-3, -3, 2], [-3, 3, 2], [3, -3, 2], [3, 3, 2]]),
        o3d.utility.Vector3iVector([[0, 1, 2], [2, 1, 3]]))
    mesh_path = scene / 'mesh.ply'
    assert o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    npz_dir = tmp_path / 'masks'
    npz_dir.mkdir()
    np.savez_compressed(npz_dir / 'image_final.npz', labels=np.full((6, 8), 7, dtype=np.int32))
    out = tmp_path / 'output'
    result = project_masks_to_3d(scene, mesh_path, out, npz_dir, colmap_threshold=0)
    report = json.loads((out / 'projection_stats.json').read_text())
    assert result['active_classes'] == report['active_classes'] == [7]
    assert report['projection_version'] == 'view-normalized-v2'
    np.testing.assert_array_equal(np.load(out / 'triangle_class_labels_raw.npy'), [7, 7])
    votes = load_npz(out / 'face_class_votes.npz')
    assert votes.shape == (2, 1)
    assert np.all((votes.toarray() > 0) & (votes.toarray() <= 1))
