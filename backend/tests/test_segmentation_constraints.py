"""Hard minimum and wall-orientation invariants on the final face labels."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '05_semantic_segmentation'))
from geometric_regularization import enforce_instance_constraints


def run(labels, minimum, normals=None, wall_cids=(), edges=None):
    labels = np.array(labels, dtype=np.int32)
    if normals is None:
        normals = np.tile([0., 0., 1.], (len(labels), 1))
    u, v = edges if edges is not None else (np.arange(len(labels)-1), np.arange(1, len(labels)))
    return enforce_instance_constraints(labels, u, v, np.array(normals), np.array([0., 1., 0.]),
                                        set(wall_cids), minimum)


def assert_minimum(labels, minimum, u, v):
    same = (labels[u] == labels[v]) & (labels[u] > 0)
    _, components = connected_components(csr_matrix((np.ones(same.sum()),
        (u[same], v[same])), shape=(len(labels), len(labels))), directed=False)
    sizes = np.bincount(components)
    assert np.all(sizes[components[labels > 0]] >= minimum)


@pytest.mark.parametrize('minimum', [1000, 3000])
def test_iterative_merging_enforces_both_requested_thresholds(minimum):
    # Several individually undersized adjacent regions must grow together.
    labels = np.repeat([17, 91, 44, 203], minimum // 2)
    result, metrics = run(labels, minimum)
    u, v = np.arange(len(labels)-1), np.arange(1, len(labels))
    assert_minimum(result, minimum, u, v)
    assert np.all(result > 0)
    assert metrics['fragment_merges'] >= 2
    np.testing.assert_array_equal(labels, np.repeat([17, 91, 44, 203], minimum // 2))


def test_relabelled_bridge_coalesces_same_owner_regions_before_size_decision():
    result, metrics = run([10, 10, 20, 10, 10], 4)
    assert np.all(result == 10)
    assert metrics['instances'] == 1
    assert metrics['minimum_instance_triangles'] == 5


def test_disconnected_small_fragment_becomes_unknown_without_touching_large_object():
    u, v = np.array([0, 1, 3]), np.array([1, 2, 4])
    result, metrics = run([10, 10, 10, 20, 20], 3, edges=(u, v))
    np.testing.assert_array_equal(result, [10, 10, 10, -1, -1])
    assert metrics['unmergeable_fragment_faces'] == 2


@pytest.mark.parametrize('angle', [0, 10, 20, -10, -20, 180, 160])
def test_wall_patch_within_twenty_degrees_of_horizontal_is_reassigned(angle):
    r = np.radians(angle)
    normals = [[0., 0., 1.], [np.sin(r), np.cos(r), 0.], [0., 0., 1.]]
    result, metrics = run([5, 5, 40], 1, normals, wall_cids={5})
    np.testing.assert_array_equal(result, [5, 40, 40])
    assert metrics['forbidden_wall_faces'] == 1


def test_near_horizontal_fragment_cannot_merge_back_into_wall():
    normals = [[0., 0., 1.]] * 3 + [[0., 1., 0.]]
    result, _ = run([5, 5, 5, 40], 3, normals, wall_cids={5})
    np.testing.assert_array_equal(result, [5, 5, 5, -1])


def test_wall_without_compatible_neighbor_becomes_unknown_and_original_unknown_stays_unknown():
    result, _ = run([5, -1], 1, [[0., 1., 0.], [0., 0., 1.]], wall_cids={5})
    np.testing.assert_array_equal(result, [-1, -1])


def test_wall_rule_uses_gravity_and_normalizes_normal_magnitudes():
    result, _ = enforce_instance_constraints(np.array([91, 17]), np.array([0]), np.array([1]),
        np.array([[5., 0., 0.], [0., 0., 2.]]), np.array([3., 0., 0.]), {91}, 1)
    np.testing.assert_array_equal(result, [17, 17])


def test_above_twenty_degrees_from_horizontal_is_eligible_as_wall():
    r = np.radians(21)
    result, _ = run([5], 1, [[np.sin(r), np.cos(r), 0.]], wall_cids={5})
    np.testing.assert_array_equal(result, [5])


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError, match='positive'):
        run([1], 0)


def test_random_contact_graphs_satisfy_both_invariants():
    rng = np.random.default_rng(13)
    for _ in range(30):
        labels = rng.choice([-1, 5, 17, 91], 40)
        normals = np.tile([0., 0., 1.], (40, 1))
        normals[rng.random(40) < .3] = [0., 1., 0.]
        u = np.r_[np.arange(39), rng.integers(0, 40, 30)]
        v = np.r_[np.arange(1, 40), rng.integers(0, 40, 30)]
        result, _ = run(labels, 6, normals, {5}, (u, v))
        assert_minimum(result, 6, u, v)
        assert not np.any((result == 5) & (normals[:, 1] > .9))
        assert np.all(result[labels < 0] == -1)
