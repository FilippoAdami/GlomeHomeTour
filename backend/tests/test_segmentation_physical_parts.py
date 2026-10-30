import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "05_semantic_segmentation"))

from geometric_regularization import regularize_foreground_connected_components


def _regularize(labels, normals=None, lengths=None):
    count = len(labels)
    centroids = np.column_stack((np.arange(count) * 0.08, np.zeros(count), np.zeros(count)))
    if normals is None:
        normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (count, 1))
    if lengths is None:
        lengths = np.full(count - 1, 0.10)
    return regularize_foreground_connected_components(
        tri_centroids=centroids,
        labels=np.asarray(labels, dtype=np.int32),
        adj_u=np.arange(count - 1, dtype=np.int32),
        adj_v=np.arange(1, count, dtype=np.int32),
        structural_cids={1, 2, 5},
        tri_normals=np.asarray(normals, dtype=np.float64),
        tri_areas=np.full(count, 0.01),
        gravity_up=np.array([0.0, 1.0, 0.0]),
        edge_lengths=np.asarray(lengths, dtype=np.float64),
    )


def test_compact_curved_subparts_reconcile_and_ids_do_not_matter():
    labels = np.array([218] * 5 + [279] * 2 + [280] * 2, dtype=np.int32)
    cleaned, changed = _regularize(labels)
    assert changed == 4
    assert np.all(cleaned == 218)

    remapped = np.array([901] * 5 + [37] * 2 + [444] * 2, dtype=np.int32)
    remapped_cleaned, remapped_changed = _regularize(remapped)
    assert remapped_changed == changed
    assert np.all(remapped_cleaned == 901)
    assert np.array_equal(labels, np.array([218] * 5 + [279] * 2 + [280] * 2))


def test_concave_contact_and_narrow_bridge_do_not_join_touching_objects():
    labels = np.array([10] * 6 + [20] * 4, dtype=np.int32)
    normals = np.vstack((
        np.tile(np.array([[1.0, 0.0, 0.0]]), (6, 1)),
        np.tile(np.array([[0.0, 1.0, 0.0]]), (4, 1)),
    ))
    # Make the label boundary an inward crease: (c_v - c_u) . (n_u - n_v) > 0.
    centroids = np.column_stack((np.arange(10) * 0.08, np.zeros(10), np.zeros(10)))
    centroids[6, 1] = -0.08
    cleaned, changed = regularize_foreground_connected_components(
        tri_centroids=centroids,
        labels=labels,
        adj_u=np.arange(9, dtype=np.int32),
        adj_v=np.arange(1, 10, dtype=np.int32),
        structural_cids={1, 2, 5},
        tri_normals=normals,
        tri_areas=np.full(10, 0.01),
        gravity_up=np.array([0.0, 1.0, 0.0]),
        edge_lengths=np.full(9, 0.10),
    )
    assert changed == 0
    assert np.array_equal(cleaned, labels)

    cleaned, changed = _regularize(labels, lengths=np.array([0.10] * 5 + [0.001] + [0.10] * 3))
    assert changed == 0
    assert np.array_equal(cleaned, labels)


def test_horizontal_shelf_and_disconnected_objects_remain_separate():
    shelf = np.array([30] * 10 + [40] * 5, dtype=np.int32)
    normals = np.vstack((
        np.tile(np.array([[0.0, 1.0, 0.0]]), (10, 1)),
        np.tile(np.array([[0.0, 0.0, 1.0]]), (5, 1)),
    ))
    cleaned, changed = _regularize(shelf, normals=normals)
    assert changed == 0
    assert np.array_equal(cleaned, shelf)

    labels = np.array([50] * 5 + [60] * 5, dtype=np.int32)
    cleaned, changed = regularize_foreground_connected_components(
        tri_centroids=np.column_stack((np.arange(10), np.zeros(10), np.zeros(10))),
        labels=labels,
        adj_u=np.array([0, 1, 2, 3, 5, 6, 7, 8], dtype=np.int32),
        adj_v=np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=np.int32),
        structural_cids={1, 2, 5},
        tri_normals=np.tile(np.array([[0.0, 0.0, 1.0]]), (10, 1)),
        tri_areas=np.full(10, 0.01),
        gravity_up=np.array([0.0, 1.0, 0.0]),
        edge_lengths=np.full(8, 0.10),
    )
    assert changed == 0
    assert np.array_equal(cleaned, labels)


def test_mixed_normal_support_patch_preserves_and_scales_with_resting_prop():
    """A shelf top must be found even when its rim makes the class non-horizontal."""
    labels = np.array([30] * 14 + [40] * 3 + [41] * 2, dtype=np.int32)
    centroids = np.zeros((len(labels), 3), dtype=np.float64)
    centroids[:9, 0] = np.arange(9) * 0.08  # broad horizontal shelf top
    centroids[9:14] = np.array([[0.65 + i * 0.01, -0.01, 0.0] for i in range(5)])
    centroids[14:17] = np.array([[0.70, 0.04, 0.0], [0.74, 0.04, 0.0], [0.78, 0.04, 0.0]])
    centroids[17:] = np.array([[0.82, 0.06, 0.0], [0.86, 0.06, 0.0]])
    normals = np.vstack((
        np.tile(np.array([[0.0, 1.0, 0.0]]), (9, 1)),
        np.tile(np.array([[0.0, 0.0, 1.0]]), (10, 1)),
    ))
    lengths = np.full(len(labels) - 1, 0.01)
    lengths[16] = 0.10  # substantial body/base boundary

    def run(scale, top_label=30, slope=0.0):
        input_labels = labels.copy()
        input_labels[:9] = top_label
        tilted_normals = normals.copy()
        tilted_normals[:, 0] -= slope * tilted_normals[:, 1]
        tilted_normals /= np.linalg.norm(tilted_normals, axis=1)[:, None]
        return regularize_foreground_connected_components(
            tri_centroids=(centroids + np.outer(centroids[:, 0], [0.0, slope, 0.0])) * scale,
            labels=input_labels,
            adj_u=np.arange(len(labels) - 1, dtype=np.int32),
            adj_v=np.arange(1, len(labels), dtype=np.int32),
            structural_cids={1, 2, 5},
            compact_span_thresh=1.60 * scale,
            tri_normals=tilted_normals,
            tri_areas=np.full(len(labels), 0.01 * scale ** 2),
            gravity_up=np.array([0.0, 1.0, 0.0]),
            edge_lengths=lengths * scale,
        )

    cleaned, changed = run(1.0)
    assert changed == 2
    assert np.all(cleaned[:14] == 30)
    assert np.all(cleaned[14:] == 40)

    scaled, scaled_changed = run(0.5)
    assert scaled_changed == changed
    assert np.array_equal(scaled, cleaned)

    for top_label in (2, -1):
        mixed, mixed_changed = run(1.0, top_label)
        assert mixed_changed == changed
        assert np.all(mixed[:9] == top_label)
        assert np.all(mixed[9:14] == 30)
        assert np.all(mixed[14:] == 40)

    tilted, tilted_changed = run(1.0, slope=0.15)
    assert tilted_changed == changed
    assert np.array_equal(tilted, cleaned)


def test_support_cut_does_not_split_the_bottom_of_an_associated_compact_part():
    labels = np.array([30] * 14 + [40] * 5 + [41] * 4, dtype=np.int32)
    centers = np.zeros((len(labels), 3))
    centers[:9, 0] = np.arange(9) * 0.08
    centers[9:14] = [[0.65 + i * 0.01, -0.01, 0.0] for i in range(5)]
    centers[14:19] = [[0.70 + i * 0.04, 0.04, 0.0] for i in range(5)]
    centers[19:22] = [[0.70 + i * 0.04, 0.06, 0.0] for i in range(3)]
    centers[22] = [0.65, 0.003, 0.0]
    normals = np.tile([0.0, 0.0, 1.0], (len(labels), 1))
    normals[:9] = [0.0, 1.0, 0.0]
    lengths = np.full(len(labels), 0.01)
    lengths[18] = 0.10
    cleaned, changed = regularize_foreground_connected_components(
        centers, labels,
        np.r_[np.arange(len(labels) - 1), 9],
        np.r_[np.arange(1, len(labels)), 22],
        {1, 2, 5}, tri_normals=normals,
        tri_areas=np.full(len(labels), 0.01), gravity_up=np.array([0.0, 1.0, 0.0]),
        edge_lengths=lengths,
    )
    assert changed == 4
    assert np.all(cleaned[:14] == 30)
    assert np.all(cleaned[14:] == 40)
