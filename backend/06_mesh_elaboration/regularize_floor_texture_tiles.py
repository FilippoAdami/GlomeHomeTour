"""Regularize extracted floor texture tiles in memory."""
from __future__ import annotations

import numpy as np


def _blend_axis(image: np.ndarray, axis: int) -> None:
    work = image if axis == 1 else np.swapaxes(image, 0, 1)
    width = work.shape[1]
    if width < 8:
        return

    band = min(width // 2, max(4, width // 6))
    for distance in range(band):
        weight = 0.5 * (1.0 + np.cos(np.pi * distance / (band - 1)))
        left = work[:, distance].copy()
        right = work[:, -1 - distance].copy()
        average = 0.5 * (left + right)
        work[:, distance] = left + weight * (average - left)
        work[:, -1 - distance] = right + weight * (average - right)


def regularize_periodic_tile(rgb: np.ndarray) -> np.ndarray:
    """Blend opposing edge colors through a cosine-feathered border band."""
    tile = np.asarray(rgb, dtype=np.float32).copy()
    if tile.ndim != 3 or tile.shape[2] != 3 or min(tile.shape[:2]) < 8:
        raise ValueError("Texture tile must be an RGB image at least 8x8 pixels")
    if not np.isfinite(tile).all():
        raise ValueError("Texture tile contains non-finite colors")
    tile = np.clip(tile, 0.0, 1.0)
    _blend_axis(tile, axis=1)
    _blend_axis(tile, axis=0)
    return tile


def regularize_texture_tiles(candidates: list[dict]) -> dict[str, np.ndarray]:
    """Return seam-corrected tiles for the caller to apply to and save floor PLYs."""
    return {candidate["name"]: regularize_periodic_tile(candidate["tile"])
            for candidate in candidates}
