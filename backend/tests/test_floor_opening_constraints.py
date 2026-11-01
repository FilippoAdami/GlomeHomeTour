import json
import sys
from pathlib import Path

from shapely.geometry import Polygon

STAGE = Path(__file__).resolve().parents[1] / "06_mesh_elaboration"
sys.path.insert(0, str(STAGE))

from floor_opening_constraints import _footprints, apply_saved_openings, update_opening_cache
from overlay_fitted_floor_wall_contours import _refresh_floor_boundary_dimensions
from regularize_floor_level_patterns import _polygon_with_holes


def _rectangle(x0, z0, x1, z1):
    return [[x0, z0], [x1, z0], [x1, z1], [x0, z1]]


def _record(rings):
    return [{"level": 1, "areas": [{"area_id": 0, "floor_polygon_rings_world_xz_m": rings}]}]


def test_restores_only_compatible_openings_still_inside_current_floor(tmp_path):
    old_rings = [_rectangle(0, 0, 10, 10), _rectangle(2, 2, 4, 4), _rectangle(8.5, 2, 9.5, 3)]
    cache = tmp_path / "phase06_floor_openings.json"
    cache.write_text(json.dumps({"version": 1, "constraints": [{
        "level": 1, "area_id": 0, "rings_world_xz_m": old_rings,
    }]}))
    records = _record([_rectangle(0, 0, 9, 10)])

    apply_saved_openings(records, cache)

    rings = records[0]["areas"][0]["floor_polygon_rings_world_xz_m"]
    footprint = Polygon(rings[0], [rings[1]])
    assert len(rings) == 2
    assert footprint.area == 86
    assert Polygon(rings[1]).area == 4


def test_incompatible_cached_outline_does_not_change_current_floor(tmp_path):
    cache = tmp_path / "phase06_floor_openings.json"
    cache.write_text(json.dumps({"version": 1, "constraints": [{
        "level": 1, "area_id": 0,
        "rings_world_xz_m": [_rectangle(0, 0, 10, 10), _rectangle(2, 2, 4, 4)],
    }]}))
    current = _rectangle(20, 20, 30, 30)
    records = _record([current])

    apply_saved_openings(records, cache)

    assert records[0]["areas"][0]["floor_polygon_rings_world_xz_m"] == [current]


def test_cached_opening_touching_current_outline_is_not_restored(tmp_path):
    cache = tmp_path / "phase06_floor_openings.json"
    cache.write_text(json.dumps({"version": 1, "constraints": [{
        "level": 1, "area_id": 0,
        "rings_world_xz_m": [_rectangle(0, 0, 10, 10), _rectangle(0.5, 2, 2.5, 4)],
    }]}))
    records = _record([_rectangle(0.5, 0, 10, 10)])

    apply_saved_openings(records, cache)

    rings = records[0]["areas"][0]["floor_polygon_rings_world_xz_m"]
    assert rings == [_rectangle(0.5, 0, 10, 10)]


def test_partially_crossing_ring_is_not_classified_as_a_hole():
    outer = _rectangle(0, 0, 10, 10)
    crossing = _rectangle(-1, 4, 1, 6)

    footprint, outline = _footprints([outer, crossing])
    final_floor = _polygon_with_holes([outer, crossing])

    assert footprint.equals(outline)
    assert footprint.area == 102
    assert final_floor.area == 102


def test_cache_keeps_prior_openings_and_records_new_ones(tmp_path):
    cache = tmp_path / "phase06_floor_openings.json"
    old_rings = [_rectangle(0, 0, 10, 10), _rectangle(2, 2, 4, 4)]
    cache.write_text(json.dumps({"version": 1, "constraints": [{
        "level": 1, "area_id": 0, "rings_world_xz_m": old_rings,
    }]}))
    records = _record([_rectangle(0, 0, 10, 10), _rectangle(2, 2, 4, 4), _rectangle(6, 6, 7, 7)])

    update_opening_cache(records, cache)

    result = json.loads(cache.read_text())
    assert len(result["constraints"]) == 2
    assert result["constraints"][0]["rings_world_xz_m"] == old_rings


def test_cache_does_not_save_an_opening_that_touches_the_outline(tmp_path):
    cache = tmp_path / "phase06_floor_openings.json"
    rings = [_rectangle(0, 0, 10, 10), _rectangle(0, 2, 2, 4)]

    update_opening_cache(_record(rings), cache)

    assert not cache.exists()


def test_restored_ring_refreshes_floor_boundary_dimensions():
    area = {
        "_floor_plane_equation_world": [0, 1, 0, -1],
        "regularized_wall_contour_world_xz_m": _rectangle(10, 10, 14, 14),
        "floor_polygon_rings_world_xz_m": [
            _rectangle(0, 0, 4, 4), _rectangle(1, 1, 2, 2),
        ],
        "dimensions": [
            {"id": "W01", "source": "regularized_wall_contour"},
            {"id": "P01.01", "source": "floor_polygon_boundary"},
        ],
    }

    _refresh_floor_boundary_dimensions(area)

    floor_dimensions = [item for item in area["dimensions"]
                        if item["source"] == "floor_polygon_boundary"]
    assert len(floor_dimensions) == 8
    assert {item["id"] for item in floor_dimensions} >= {"P01.01", "P02.01"}
    assert [item["id"] for item in area["dimensions"] if item["source"] == "regularized_wall_contour"] == ["W01"]

