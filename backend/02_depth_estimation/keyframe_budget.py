#!/usr/bin/env python3
"""How many keyframes a scene needs, and which ones.

* :func:`frame_budget` -- the empirical indoor rule ``N = 50 + (8..11) * A_floor``
  over ``01_poses_refinment/scene_size.txt``. Frame count is driven by the room,
  not by a fraction of however long the operator happened to walk.
* :func:`coverage_topup` / :func:`coverage_prune` -- move a selection into that
  band from below or above, judging every frame by the *places* it observes
  (voxelised COLMAP landmarks) rather than by its position in the walk.

The chain walk in ``step_filter_depth`` guarantees consecutive overlap but has
no notion of a frame budget beyond re-walking at a tighter threshold, which
bottoms out at the overlap floor. These two close the gap to the band.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

# N ~= 50 + (8..11) * A_floor, calibrated for smartphone wide-lens captures of
# 2.5-3 m indoor rooms.
#
# Base constant: the loop-closure minimum -- boundaries, doors, ceiling/floor
# transitions need covering regardless of how small the room is.
# Marginal density: perimeter travel, multi-height passes (chest, eye,
# downward), and furniture occlusions.
EMPIRICAL_BASE = 50.0
EMPIRICAL_PER_M2 = (8.0, 11.0)

# Cell size for coverage counting: well under the smallest thing the floor plan
# stage cares about, well over COLMAP's triangulation noise. At 5 cm the cells
# start splitting one surface patch across several.
COVERAGE_VOXEL_M = 0.10

# Views each occupied cell wants, for multi-view consistency. 3 is reachable:
# the median over the full 652-frame reference capture is 4. Counting raw COLMAP
# track ids instead would not be -- median track length there is 2, so a 3-view
# target per *track* is unreachable no matter which frames are picked, and the
# greedy would be scoring against nothing.
VIEWS_PER_CELL = 3


def read_scene_size(path: Path) -> dict[str, float]:
    """Parse ``scene_extent.py``/``floor_count.py``'s ``key: value`` txt."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = float(v.strip())
    if "floors" not in out:
        out["floors"] = 1.0
    missing = {"x", "y", "z", "area_m2"} - out.keys()
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    return out


def frame_budget(size: dict[str, float], total_frames: int) -> tuple[int, int, dict]:
    """``(min_keyframes, max_keyframes, detail)`` for this scene, from floor area.

    ``area_m2`` is the footprint of the whole capture (stacked floors project
    onto one hull), so total floor area is footprint x floors.
    """
    floor_area = size["area_m2"] * max(1.0, size["floors"])
    lo, hi = (int(np.ceil(EMPIRICAL_BASE + c * floor_area)) for c in EMPIRICAL_PER_M2)
    detail = {"floor_area_m2": round(floor_area, 1), "band": [lo, hi]}
    return min(lo, total_frames), min(max(hi, lo), total_frames), detail


def voxel_coverage(
    tracks: Sequence[set[int]],
    points: dict,
    voxel_m: float = COVERAGE_VOXEL_M,
) -> list[set[tuple[int, int, int]]]:
    """Per-frame set of occupied ``voxel_m`` cells that frame observes."""
    cell = {pid: tuple(np.floor(np.asarray(p["xyz"], dtype=float) / voxel_m).astype(int))
            for pid, p in points.items()}
    return [{cell[t] for t in track if t in cell} for track in tracks]


def coverage_topup(
    coverage: Sequence[set],
    selected: Sequence[int],
    max_keyframes: int,
    views_per_cell: int = VIEWS_PER_CELL,
) -> list[int]:
    """Greedily add frames until every cell has ``views_per_cell`` observers.

    ``selected`` is the connectivity chain and is never dropped from -- coverage
    outranks compactness, and a chain built for consecutive overlap cannot be
    subsampled without re-opening the gaps it exists to close.

    ponytail: plain greedy max-coverage (submodular, so 1-1/e of optimal). An
    ILP would buy a few percent for a solver dependency and minutes of runtime.
    """
    chosen = list(selected)
    if len(chosen) >= max_keyframes:
        return sorted(chosen)

    counts = Counter()
    for i in chosen:
        counts.update(coverage[i])
    taken = set(chosen)

    while len(chosen) < max_keyframes:
        best, best_gain = -1, 0
        for i, cells in enumerate(coverage):
            if i in taken:
                continue
            # Unobserved cells count too: they are needy with a count of zero.
            gain = sum(1 for c in cells if counts[c] < views_per_cell)
            if gain > best_gain:
                best, best_gain = i, gain
        if best < 0:
            break
        chosen.append(best)
        taken.add(best)
        counts.update(coverage[best])

    return sorted(chosen)


def coverage_prune(
    coverage: Sequence[set],
    selected: Sequence[int],
    max_keyframes: int,
    covisibility: Callable[[int, int], float],
    min_covisibility: float,
    views_per_cell: int = VIEWS_PER_CELL,
) -> list[int]:
    """Drop frames down to ``max_keyframes``, cheapest-in-coverage first.

    The dual of :func:`coverage_topup`, and the half that actually decides the
    output when the room's budget sits below what the chain walk produces.

    A frame may only go if its two neighbours in the chain still see each other
    afterwards (``covisibility >= min_covisibility``), which is the invariant the
    walk exists to maintain -- so this thins the chain without ever re-opening
    the gaps it closed. That is also why it can stop above ``max_keyframes``:
    when every remaining frame is load-bearing for connectivity, coverage
    outranks the budget and the stage overshoots, loudly.

    Cost of dropping frame ``i`` is lexicographic: first how many cells it is
    the *only* observer of, then how many would fall below ``views_per_cell``.
    The two must not be summed -- a plain count of "cells this frame is needed
    for" rates sole custody of 50 cells identically to 50 cells already seen
    three times, and a greedy fed that scores worse than evenly subsampling the
    chain, which is the bar any of this has to clear.

    ponytail: recomputes each survivor's cost every round, O(n^2) in the number
    of drops. ~1 s for 250 drops over 460 frames; a lazy-update heap if a
    whole-house capture ever makes that hurt.
    """
    chain = list(selected)
    if len(chain) <= max_keyframes:
        return sorted(chain)

    counts = Counter()
    for i in chain:
        counts.update(coverage[i])

    while len(chain) > max_keyframes:
        best, best_cost = -1, None
        # Endpoints stay: the first frame anchors the walk and the last is the
        # capture's loop closure back at the entry door.
        for k in range(1, len(chain) - 1):
            if covisibility(chain[k - 1], chain[k + 1]) < min_covisibility:
                continue
            unique = sum(1 for c in coverage[chain[k]] if counts[c] <= 1)
            thin = sum(1 for c in coverage[chain[k]] if 1 < counts[c] <= views_per_cell)
            cost = (unique, thin)
            if best_cost is None or cost < best_cost:
                best, best_cost = k, cost
        if best < 0:
            break  # every survivor is holding the chain together
        counts.subtract(coverage[chain[best]])
        chain.pop(best)

    return sorted(chain)


def _demo() -> None:
    size = {"x": 4.37, "y": 3.17, "z": 3.85, "area_m2": 16.23, "floors": 1}
    lo, hi, d = frame_budget(size, total_frames=652)
    assert (lo, hi) == (180, 229), (lo, hi)
    assert d["floor_area_m2"] == 16.2
    # Two floors of the same footprint is twice the floor area.
    assert frame_budget({**size, "floors": 2}, 10_000)[0] == 50 + int(np.ceil(8 * 32.46))
    # Never asks for more frames than were captured.
    assert frame_budget(size, total_frames=100) == (100, 100, d)

    # Top-up: frame 2 is the only observer of cells {9, 10}, so it must be
    # picked before frame 1, which only re-observes what the chain already has.
    cov = [{1, 2, 3}, {1, 2, 3}, {9, 10}, {3, 4}]
    assert coverage_topup(cov, [0], max_keyframes=2, views_per_cell=1) == [0, 2]
    assert coverage_topup(cov, [0], max_keyframes=1, views_per_cell=1) == [0]
    # With a 2-view target, re-observing {1,2,3} is worth more than one new cell.
    assert coverage_topup(cov, [0], max_keyframes=2, views_per_cell=2) == [0, 1]

    # Prune: frame 1 duplicates the chain's cells, frame 3 is the only observer
    # of cell 4, so frame 1 goes first. Frame 2 is pinned as the last frame.
    always = lambda a, b: 1.0
    assert coverage_prune(cov, [0, 1, 3, 2], 3, always, 0.5, views_per_cell=1) == [0, 2, 3]
    # Sole custody outranks thinning: frame 1 holds cell 7 alone, frame 3 shares
    # all of its cells with the chain, so frame 3 goes even though it is larger.
    cov2 = [{1, 2, 3}, {7}, {1, 2}, {1, 2, 3}]
    assert coverage_prune(cov2, [0, 1, 3, 2], 3, always, 0.5) == [0, 1, 2]
    # A frame whose removal would disconnect its neighbours is never dropped.
    never = lambda a, b: 0.0
    assert coverage_prune(cov, [0, 1, 3, 2], 2, never, 0.5) == [0, 1, 2, 3]

    # Landmarks a voxel apart collapse to one cell, two voxels apart do not.
    pts = {1: {"xyz": [0.0, 0.0, 0.0]}, 2: {"xyz": [0.05, 0.0, 0.0]},
           3: {"xyz": [0.25, 0.0, 0.0]}}
    assert [len(c) for c in voxel_coverage([{1, 2}, {1, 3}], pts)] == [1, 2]

    print(f"ok -- demo scene budget {lo}-{hi} of 652 frames")


if __name__ == "__main__":
    _demo()
