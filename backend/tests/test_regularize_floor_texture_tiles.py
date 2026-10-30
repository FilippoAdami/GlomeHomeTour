import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "06_mesh_elaboration"))
from regularize_floor_texture_tiles import regularize_periodic_tile, regularize_texture_tiles


def test_periodic_tile_regularization_blends_opposite_edges():
    rng = np.random.default_rng(31)
    tile = np.clip(rng.normal(0.48, 0.16, (64, 72, 3)), 0.0, 1.0).astype(np.float32)
    before_vertical = np.abs(tile[:, 1] - tile[:, -1]).mean()
    before_horizontal = np.abs(tile[1] - tile[-1]).mean()

    corrected = regularize_periodic_tile(tile)

    assert np.abs(corrected[:, 0] - corrected[:, -1]).mean() < 0.002
    assert np.abs(corrected[0] - corrected[-1]).mean() < 0.002
    assert np.abs(corrected[:, 1] - corrected[:, -1]).mean() < before_vertical * 0.8
    assert np.abs(corrected[1] - corrected[-1]).mean() < before_horizontal * 0.8


def test_texture_regularization_returns_corrected_tiles_in_memory():
    tile = np.full((16, 16, 3), 0.5, dtype=np.float32)
    candidates = [{"name": "wood-sample", "tile": tile}]

    corrected = regularize_texture_tiles(candidates)

    assert list(corrected) == ["wood-sample"]
    np.testing.assert_array_equal(corrected["wood-sample"], tile)
