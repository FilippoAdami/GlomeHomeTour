#!/usr/bin/env python3
"""Shared floor-counting rule: threshold-crossing on camera-path height.

Floor k (k>=2) is declared once more than FLOOR_MIN_POINTS camera positions
clear height FLOOR_RISE_M*(k-1) + FLOOR_MARGIN_M above the floor -- floor 2 ->
3.0m, floor 3 -> 5.8m, floor 4 -> 8.6m, etc. Used both pre-COLMAP (raw ARCore
poses, select_keyframes_arcore.py) and post-COLMAP (triangulated poses,
floor_count.py) so both stages agree on floor count from the same rule.
"""

from __future__ import annotations

import numpy as np

FLOOR_RISE_M = 2.8
FLOOR_MARGIN_M = 0.2
FLOOR_MIN_POINTS = 15


def floor_threshold_m(floor_number: int) -> float:
    """Height a camera path must clear for `floor_number` (>=2) to count."""
    return FLOOR_RISE_M * (floor_number - 1) + FLOOR_MARGIN_M


def count_floor_bands(heights: np.ndarray) -> int:
    """1 + the number of consecutive thresholds (floor_threshold_m(2), (3), ...)
    cleared by more than FLOOR_MIN_POINTS camera positions each; stops at the
    first threshold that isn't cleared. `heights` are camera-center heights
    with the floor already shifted to 0."""
    n_floors = 1
    floor_number = 2
    while np.count_nonzero(heights > floor_threshold_m(floor_number)) > FLOOR_MIN_POINTS:
        n_floors = floor_number
        floor_number += 1
    return n_floors


def _demo() -> None:
    assert floor_threshold_m(2) == 3.0
    assert floor_threshold_m(3) == 5.8
    assert abs(floor_threshold_m(4) - 8.6) < 1e-9

    two_floors = np.zeros(60)
    two_floors[30:] = 3.5   # 30 points clear the 3.0m floor-2 threshold
    assert count_floor_bands(two_floors) == 2, "expected 2 floor bands"

    three_floors = np.zeros(90)
    three_floors[30:60] = 3.5   # clears floor 2 (3.0m)
    three_floors[60:] = 6.0     # clears floor 3 (5.8m)
    assert count_floor_bands(three_floors) == 3, "expected 3 floor bands"

    not_enough_points = np.zeros(30)
    not_enough_points[20:] = 3.5   # only 10 points above 3.0m, <= FLOOR_MIN_POINTS
    assert count_floor_bands(not_enough_points) == 1, "expected 1 floor band (not enough points)"

    one_floor = np.random.default_rng(2).normal(1.5, 0.3, 200)
    assert count_floor_bands(one_floor) == 1, "expected 1 floor band"
    print("[floor_bands._demo] OK")


if __name__ == "__main__":
    _demo()
