"""Read-only COLMAP feature tracks for optional panoptic identity evidence."""
from pathlib import Path

import numpy as np


def _records(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]


def _int(value, context):
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f'malformed {context}') from exc


def _float(value, context):
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f'malformed {context}') from exc
    if not np.isfinite(result):
        raise ValueError(f'malformed {context}')
    return result


def load_colmap_tracks(inputs: Path, trusted_groups: dict[str, int]):
    """Load reliable sparse tracks without changing COLMAP inputs or running inference."""
    sparse = Path(inputs) / 'sparse' / '0'
    points_path = sparse / 'points3D.txt'
    if not points_path.exists():
        return None
    images_path = sparse / 'images.txt'
    # Empty POINTS2D records are valid and still occupy the second line of an image record.
    lines = [line.strip() for line in images_path.read_text().splitlines()
             if not line.lstrip().startswith('#')]
    if len(lines) % 2:
        raise ValueError('malformed images.txt records')
    images, stems = {}, set()
    for header, observations in zip(lines[::2], lines[1::2]):
        tokens = header.split()
        if len(tokens) < 10:
            raise ValueError('malformed image header')
        image_id = _int(tokens[0], 'image ID')
        for value in tokens[1:9]:
            _float(value, 'image pose')
        stem = Path(tokens[9]).stem
        if stem in stems:
            raise ValueError('duplicate image stem')
        triples = observations.split()
        if len(triples) % 3:
            raise ValueError('malformed image observation triples')
        xy, point_ids = [], []
        for x, y, point_id in zip(triples[::3], triples[1::3], triples[2::3]):
            xy.append((_float(x, 'image observation x'), _float(y, 'image observation y')))
            point_ids.append(_int(point_id, 'image observation point ID'))
        if image_id in images:
            raise ValueError('duplicate image ID')
        images[image_id] = (stem, np.asarray(xy, np.float64).reshape(-1, 2), np.asarray(point_ids, np.int64))
        stems.add(stem)

    points = {}
    for line in _records(points_path):
        tokens = line.split()
        if len(tokens) < 8 or (len(tokens) - 8) % 2:
            raise ValueError('malformed points3D record')
        point_id = _int(tokens[0], 'point ID')
        xyz = [_float(value, 'point coordinate') for value in tokens[1:4]]
        for value in tokens[4:7]:
            _int(value, 'point color')
        error = _float(tokens[7], 'point error')
        if error < 0:
            raise ValueError('malformed point error')
        track = [_int(value, 'point track') for value in tokens[8:]]
        if point_id in points:
            raise ValueError('duplicate point ID')
        points[point_id] = (xyz, error, track[::2])

    kept = []
    for point_id, (xyz, error, track_images) in points.items():
        groups = {trusted_groups[images[image_id][0]] for image_id in track_images
                  if image_id in images and images[image_id][0] in trusted_groups}
        if error < 2 and len(track_images) >= 3 and len(groups) >= 2:
            kept.append((point_id, xyz))
    kept.sort(key=lambda item: item[0])
    point_ids = np.asarray([item[0] for item in kept], np.int64)
    points_raw = np.asarray([item[1] for item in kept], np.float64).reshape(-1, 3)
    columns = {point_id: index for index, point_id in enumerate(point_ids)}
    observations = {}
    for stem, xy, observed_ids in images.values():
        if stem not in trusted_groups:
            continue
        valid = np.asarray([point_id in columns for point_id in observed_ids], bool)
        observations[stem] = (xy[valid], np.asarray([columns[point_id] for point_id in observed_ids[valid]], np.int64))
    return dict(point_ids=point_ids, points_raw=points_raw, observations=observations)
