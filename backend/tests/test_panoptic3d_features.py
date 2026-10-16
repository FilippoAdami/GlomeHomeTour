from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '06c_semantic_segmentation'))
from panoptic3d_features import load_colmap_tracks


def write_sparse(tmp_path, images, points=None):
    sparse = tmp_path / 'sparse' / '0'
    sparse.mkdir(parents=True, exist_ok=True)
    (sparse / 'images.txt').write_text(images)
    if points is not None:
        (sparse / 'points3D.txt').write_text(points)
    return tmp_path


def test_loads_noncontiguous_reliable_tracks_and_xy_columns(tmp_path):
    inputs = write_sparse(tmp_path,
        '1 1 0 0 0 0 0 0 1 trusted_a.jpg\n10 20 10 11 21 20 12 22 -1\n'
        '2 1 0 0 0 0 0 0 1 trusted_b.jpg\n30 40 10\n'
        '3 1 0 0 0 0 0 0 1 ignored.jpg\n50 60 10\n',
        '20 2 3 4 1 2 3 2.1 1 1 2 0 3 0\n'
        '10 5 6 7 1 2 3 1.0 1 0 2 0 3 0\n')
    result = load_colmap_tracks(inputs, {'trusted_a': 0, 'trusted_b': 1})
    assert result['point_ids'].tolist() == [10]
    assert result['points_raw'].tolist() == [[5., 6., 7.]]
    xy, columns = result['observations']['trusted_a']
    assert xy.tolist() == [[10., 20.]] and columns.tolist() == [0]
    assert result['observations']['trusted_b'][1].tolist() == [0]
    assert 'ignored' not in result['observations']


def test_filters_error_track_length_and_independent_trusted_groups(tmp_path):
    inputs = write_sparse(tmp_path,
        '1 1 0 0 0 0 0 0 1 a.jpg\n1 1 1 2 2 2 3 3 3 4 4 4\n'
        '2 1 0 0 0 0 0 0 1 b.jpg\n1 1 1 2 2 2 3 3 3 4 4 4\n'
        '3 1 0 0 0 0 0 0 1 c.jpg\n1 1 1 2 2 2 3 3 3 4 4 4\n',
        '1 0 0 0 0 0 0 2.0 1 0 2 0 3 0\n'
        '2 0 0 0 0 0 0 1.0 1 0 2 0\n'
        '3 0 0 0 0 0 0 1.0 1 0 2 0 3 0\n'
        '4 0 0 0 0 0 0 1.0 1 0 2 0 3 0\n')
    result = load_colmap_tracks(inputs, {'a': 0, 'b': 0, 'c': 1})
    assert result['point_ids'].tolist() == [3, 4]


def test_missing_points_file_returns_none(tmp_path):
    assert load_colmap_tracks(write_sparse(tmp_path, '1 1 0 0 0 0 0 0 1 a.jpg\n\n'), {'a': 0}) is None


def test_rejects_malformed_observation_triples(tmp_path):
    inputs = write_sparse(tmp_path, '1 1 0 0 0 0 0 0 1 a.jpg\n1 2\n', '# empty\n')
    with pytest.raises(ValueError, match='observation triples'):
        load_colmap_tracks(inputs, {'a': 0})


@pytest.mark.parametrize('point', ['1 nan 0 0 0 0 0 1 1 0 2 0 3 0\n',
                                   '1 0 0 0 0 0 0 -1 1 0 2 0 3 0\n'])
def test_rejects_nonfinite_or_negative_point_reliability(tmp_path, point):
    inputs = write_sparse(tmp_path, '1 1 0 0 0 0 0 0 1 a.jpg\n1 2 1\n', point)
    with pytest.raises(ValueError, match='point'):
        load_colmap_tracks(inputs, {'a': 0})


def test_rejects_nonfinite_pose_and_duplicate_stem(tmp_path):
    inputs = write_sparse(tmp_path, '1 nan 0 0 0 0 0 0 1 a.jpg\n\n', '# empty\n')
    with pytest.raises(ValueError, match='pose'):
        load_colmap_tracks(inputs, {'a': 0})
    inputs = write_sparse(tmp_path,
        '1 1 0 0 0 0 0 0 1 same.jpg\n\n2 1 0 0 0 0 0 0 1 same.png\n\n', '# empty\n')
    with pytest.raises(ValueError, match='stem'):
        load_colmap_tracks(inputs, {'same': 0})
