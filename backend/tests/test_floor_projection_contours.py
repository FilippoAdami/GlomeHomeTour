from pathlib import Path
import sys

import numpy as np
from PIL import Image

STAGE = Path(__file__).resolve().parents[1] / "06_mesh_elaboration"
sys.path.insert(0, str(STAGE))

from fit_floor_projection_contours_to_black import (
    MIN_FLOOR_POLYGON_SEGMENT_M,
    _fit_polygon,
    _polygon_mask,
    _remove_short_polygon_edges,
)
from overlay_fitted_floor_wall_contours import _rings_from_segments
from regularize_floor_projection_contours import (
    _clean_aligned_contour,
    _parallel_segment_clearance_errors,
)


def test_final_floor_ring_removes_edges_shorter_than_ten_centimeters():
    points_px = np.array([[0, 0], [100, 0], [100, 100], [94, 100], [0, 100]], dtype=float)

    [simplified] = _remove_short_polygon_edges(points_px, 100.0, 100.0)

    assert simplified is not None
    edges_m = (np.roll(simplified, -1, axis=0) - simplified) / 100.0
    assert np.linalg.norm(edges_m, axis=1).min() >= MIN_FLOOR_POLYGON_SEGMENT_M
    assert len(simplified) == 4


def test_aligned_floor_contour_removes_sub_ten_centimeter_jogs():
    points = np.array([
        [0, 0], [4, 0], [4, 3], [2.01, 3], [2.01, 2],
        [2, 2], [2, 3], [0, 3],
    ], dtype=float)

    cleaned, report = _clean_aligned_contour(points)
    edges = np.roll(cleaned, -1, axis=0) - cleaned

    assert len(cleaned) == 4
    assert np.linalg.norm(edges, axis=1).min() >= MIN_FLOOR_POLYGON_SEGMENT_M
    assert np.all((np.abs(edges[:, 0]) < 1e-8) | (np.abs(edges[:, 1]) < 1e-8))
    assert report["remaining_parallel_segment_pairs"] == []


def test_aligned_contour_removes_narrow_parallel_sliver_and_reports_it():
    points = np.array([
        [0., 0.], [2., 0.], [2., 0.975], [3., 0.975],
        [3., 1.025], [2., 1.025], [2., 2.], [0., 2.],
    ])

    cleaned, report = _clean_aligned_contour(points)

    assert report["flagged_parallel_segment_pairs"]
    assert report["removed_area_m2"] > 0
    assert report["remaining_parallel_segment_pairs"] == []
    assert np.max(cleaned[:, 0]) == 2.0
    assert _parallel_segment_clearance_errors(cleaned) == []


def test_exported_world_polygon_ring_removes_sub_ten_centimeter_edges(tmp_path):
    image_path = tmp_path / "projection.png"
    pixels = np.full((102, 102, 3), [242, 240, 237], dtype=np.uint8)
    pixels[10:90, 10:90] = 0
    Image.fromarray(pixels).save(image_path)
    # 6 px jog becomes about 6.5 cm after world conversion.
    starts = np.array([[0, 0], [100, 0], [100, 100], [94, 100], [94, 50], [0, 50]], dtype=float)
    segments = [
        {"component_id": 1, "start_px": start.tolist(),
         "end_px": starts[(index + 1) % len(starts)].tolist()}
        for index, start in enumerate(starts)
    ]
    area = {
        "expanded_plane_bounds_uv_m": [[0.0, 0.0], [1.0, 1.0]],
        "plane_equation": [0.0, 1.0, 0.0, -0.25],
    }

    [ring] = _rings_from_segments(segments, area, image_path)

    edges_xz = np.roll(ring[:, [0, 2]], -1, axis=0) - ring[:, [0, 2]]
    assert np.linalg.norm(edges_xz, axis=1).min() >= MIN_FLOOR_POLYGON_SEGMENT_M


def test_supplied_regularized_contour_is_preserved_for_final_fill():
    pixels = np.full((120, 120, 3), [242, 240, 237], dtype=np.uint8)
    pixels[35:85, 35:85] = 0
    points = np.array([[10, 10], [110, 10], [110, 110], [10, 110]], dtype=float)

    fitted, report, retained, mask, _ = _fit_polygon(
        Image.fromarray(pixels), [], 100.0, 100.0, contour_points_px=points
    )

    assert np.array_equal(fitted, points)
    assert report["segment_moves"] == []
    assert np.array_equal(mask, _polygon_mask(points, pixels.shape[:2]))
    assert retained.sum() == 50 * 50


def test_hole_rectangle_snaps_to_nearby_wall_contour_and_removes_thin_strip():
    pixels = np.full((120, 120, 3), [242, 240, 237], dtype=np.uint8)
    pixels[10:110, 10:110] = 0
    contour = np.array([[10, 10], [110, 10], [110, 110], [10, 110]], dtype=float)
    rectangles = [{"corners_px": [[11.3, 30], [40, 30], [40, 70], [12.0, 70]]}]

    _, report, _, _, hole_mask = _fit_polygon(
        Image.fromarray(pixels), rectangles, 100.0, 100.0, contour_points_px=contour
    )

    assert len(report["hole_rectangle_wall_snaps"]) == 1
    assert 0.01 < rectangles[0]["wall_contour_snaps"][0]["gap_m"] < 0.03
    assert hole_mask[30:71, 10].all()
