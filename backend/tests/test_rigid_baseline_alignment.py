import importlib.util
from pathlib import Path

import sys
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "07_alignment" / "build_rigid_baseline.py"
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("build_rigid_baseline", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_fit_rigid_translation_only():
    source = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    offset = np.array([2., -3., 4.])
    rotation, translation = MODULE.fit_rigid(source, source + offset, np.ones(3), rotation=False)
    np.testing.assert_array_equal(rotation, np.eye(3))
    np.testing.assert_allclose(source @ rotation.T + translation, source + offset)


def test_fit_rigid_rotation():
    source = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    theta = 0.2
    expected_rotation = np.array([[np.cos(theta), -np.sin(theta), 0.],
                                  [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    expected_translation = np.array([2., -3., 4.])
    target = source @ expected_rotation.T + expected_translation
    rotation, translation = MODULE.fit_rigid(source, target, np.ones(len(source)), rotation=True)
    np.testing.assert_allclose(rotation, expected_rotation, atol=1e-12)
    np.testing.assert_allclose(translation, expected_translation, atol=1e-12)

