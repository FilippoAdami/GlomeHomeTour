from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '08_artifacts_generation'))
from cinematic_minimap import AnimatedMinimap, compose_minimap


def test_floor_state_marker_reveal_and_identical_composition(tmp_path):
    Image.new('RGB', (100, 100), 'white').save(tmp_path / 'plan.png')
    grid = {'x_min': 0, 'z_max': 10, 'size': .1, 'width': 100, 'height': 100}
    levels = [{'name': name, 'styles': {'objects': {'grid': grid, 'image': 'plan.png'}}} for name in ('BASE', 'UP')]
    cameras = [{'number': 1, 'kind': 'photo', 'floor': 'BASE', 'position': [1, 1, 1]},
               {'number': 2, 'kind': 'photo', 'floor': 'UP', 'position': [4, 2, 2]},
               {'number': 3, 'kind': 'panorama', 'floor': 'BASE', 'position': [7, 1, 7]}]
    poses = np.tile(np.eye(4), (5, 1, 1))
    poses[:, :3, 3] = [[1, 1, 1], [4, 1, 1], [4, 2, 2], [5, 2, 2], [2, 1, 1]]
    flight = {'camera_to_world': poses, 'checkpoints': [{'number': 1, 'frame': 1}, {'number': 2, 'frame': 3},
                                                        {'number': 3, 'frame': 4}]}
    minimap = AnimatedMinimap(tmp_path, {'levels': levels, 'cameras': cameras}, flight,
                              ['BASE', 'BASE', 'UP', 'UP', 'BASE'], 200)
    minimap.advance(0)
    assert np.asarray(minimap.markers['BASE'])[:, :, 3].max() == 0
    minimap.advance(1)
    assert np.asarray(minimap.markers['BASE'])[:, :, 3].max() == 255
    base_traveled = np.asarray(minimap.progress['BASE']).copy()
    minimap.advance(3)
    assert np.asarray(minimap.markers['UP'])[:, :, 3].max() == 255
    assert np.array_equal(base_traveled, np.asarray(minimap.progress['BASE']))
    before_panorama = np.asarray(minimap.markers['BASE']).copy()
    overlay = minimap.advance(4)
    assert np.count_nonzero(np.asarray(minimap.markers['BASE'])[:, :, 3]) > np.count_nonzero(before_panorama[:, :, 3])
    assert np.array_equal(base_traveled, np.asarray(minimap.progress['BASE']))
    blank = np.zeros((300, 200, 3), dtype=np.uint8)
    assert np.array_equal(compose_minimap(blank, overlay), compose_minimap(blank.copy(), overlay))
    assert minimap.pixel([2, 1, 3], 'BASE') == (16.8, 25.2)
