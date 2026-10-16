"""Behavior-preserving kernels used by structural mesh elaboration."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'06_mesh_elaboration'))
from recognize_and_regularize_patterns import binned_samples


def test_binned_statistics_match_half_open_reference_exactly():
    rng = np.random.default_rng(120)
    edges = np.arange(-1., 1., .02)
    coordinates = np.r_[rng.uniform(-2, 2, 10000), edges, np.nan, -np.inf, np.inf]
    values = rng.normal(size=len(coordinates))
    groups = binned_samples(coordinates, values, edges)
    assert len(groups) == len(edges)-1
    for lo, hi, actual in zip(edges[:-1], edges[1:], groups):
        expected = values[(coordinates >= lo) & (coordinates < hi)]
        np.testing.assert_array_equal(actual, expected)
        if len(actual):
            assert np.mean(actual) == np.mean(expected)
            assert np.median(actual) == np.median(expected)


def test_binned_samples_empty_support_and_empty_bins():
    groups = binned_samples(np.array([0., 3.]), np.array([1., 2.]), np.array([1., 2., 3.]))
    assert len(groups) == 2 and all(len(group) == 0 for group in groups)


def test_regularization_requires_fitted_manifests_without_inventing_geometry(tmp_path, monkeypatch, capsys):
    import argparse
    import recognize_and_regularize_patterns as regularizer
    output = tmp_path/'output'
    monkeypatch.setattr(regularizer, 'parse_args', lambda: argparse.Namespace(instances_dir=tmp_path, output_dir=output))
    assert regularizer.main() == 1
    assert 'Required fitted geometry manifest not found' in capsys.readouterr().out
    assert not list(output.glob('*.ply'))


def test_occupancy_encoding_keeps_negative_coordinates_and_empty_support():
    from shapely.geometry import Point, Polygon
    from structural_geometry import supported_horizontal_extensions
    footprint = [[-3., -2.], [-2., -2.], [-2., -1.], [-3., -1.]]
    unchanged = supported_horizontal_extensions(footprint, np.empty((0, 2)), {})
    assert Polygon(unchanged).equals(Polygon(footprint))
    a, b = np.meshgrid(np.arange(-2.05, -1., .01), np.arange(-1.9, -1.1, .01))
    observed = np.column_stack([a.ravel(), b.ravel()])
    expanded = Polygon(supported_horizontal_extensions(footprint, observed, {}))
    assert expanded.covers(Point(-1.5, -1.5))
    assert expanded.covers(Polygon(footprint))
