import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "06_mesh_elaboration" / "build_rigid_baseline.py"
SPEC = importlib.util.spec_from_file_location("build_rigid_baseline", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_translation_fit_does_not_rotate_source():
    source = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    offset = np.array([2., -3., 4.])
    rotation, translation = MODULE.fit_translation(source, source + offset, np.ones(3))
    np.testing.assert_array_equal(rotation, np.eye(3))
    np.testing.assert_allclose(source @ rotation.T + translation, source + offset)


def test_rigid_fit_recovers_rotation_when_enabled():
    source = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    theta = 0.2
    expected_rotation = np.array([[np.cos(theta), -np.sin(theta), 0.],
                                  [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    expected_translation = np.array([2., -3., 4.])
    target = source @ expected_rotation.T + expected_translation
    rotation, translation = MODULE.fit_global_rigid(source, target, np.ones(len(source)))
    np.testing.assert_allclose(rotation, expected_rotation, atol=1e-12)
    np.testing.assert_allclose(translation, expected_translation, atol=1e-12)


def test_border_clusters_join_instances_sharing_a_mesh_edge():
    faces = np.array([[0, 1, 2], [1, 0, 3], [4, 5, 6]])
    assert MODULE.border_clusters(faces, np.array([10, 20, 30])) == [[10, 20], [30]]
