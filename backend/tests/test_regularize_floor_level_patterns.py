import json
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from plyfile import PlyData

STAGE_DIR = Path(__file__).resolve().parents[1] / "06_mesh_elaboration"
sys.path.insert(0, str(STAGE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_3DGS_to_mesh"))

from fill_mesh_holes import export_ply_with_face_attributes
from regularize_floor_level_patterns import (
    _clear_texture_debug_outputs,
    _sample_periodic_tile,
    _subdivide_and_apply_textures,
    _write_regularized_floor_ply,
    regularize_floor_level_patterns,
)


def test_repeats_seam_regularized_texture_over_planar_mesh():
    mesh = trimesh.Trimesh(
        vertices=np.array([[0, 0.25, 0], [0.1, 0.25, 0],
                           [0.1, 0.25, 0.1], [0, 0.25, 0.1]]),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        process=False,
    )
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh, face_colors=np.full((2, 4), [100, 100, 100, 255], dtype=np.uint8)
    )
    context = {
        "area_info": {
            "labels": np.ones((11, 11), dtype=np.int32),
            "axis_u": np.array([1.0, 0.0, 0.0]),
            "axis_v": np.array([0.0, 0.0, 1.0]),
            "raster_lower_uv": np.array([0.0, 0.0]),
            "raster_cell_size_m": 0.01,
        },
        "filled_labels": np.ones((11, 11), dtype=np.int32),
        "texture_regions": {1: {"tile_name": "wood.png", "tile_origin_uv": np.array([0.0, 0.1])}},
    }
    tile = np.zeros((8, 8, 3), dtype=np.float32)
    tile[..., 0] = 1.0

    vertices, faces, colors, applied = _subdivide_and_apply_textures(
        mesh, context, {"wood.png": tile}
    )

    assert len(faces) > len(mesh.faces)
    assert applied == {"wood.png": len(faces)}
    np.testing.assert_array_equal(colors[:, 0], 255)
    np.testing.assert_array_equal(colors[:, 1:], 0)
    np.testing.assert_allclose(vertices[:, 1], 0.25, atol=1e-7)
    np.testing.assert_allclose(
        [vertices[:, 0].min(), vertices[:, 0].max(),
         vertices[:, 2].min(), vertices[:, 2].max()],
        [0.0, 0.1, 0.0, 0.1], atol=1e-7,
    )


def test_rejected_texture_region_keeps_standard_floor_appearance():
    mesh = trimesh.Trimesh(
        vertices=np.array([[0, 0.25, 0], [0.1, 0.25, 0], [0.1, 0.25, 0.1],
                           [0, 0.25, 0.1]]),
        faces=np.array([[0, 1, 2], [0, 2, 3]]), process=False,
    )
    standard_rgb = np.array([105, 83, 62], dtype=np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh, face_colors=np.tile([*standard_rgb, 255], (2, 1))
    )
    context = {
        "area_info": {
            "labels": np.full((11, 11), 2, dtype=np.int32),
            "axis_u": np.array([1.0, 0.0, 0.0]),
            "axis_v": np.array([0.0, 0.0, 1.0]),
            "raster_lower_uv": np.array([0.0, 0.0]),
            "raster_cell_size_m": 0.01,
        },
        "filled_labels": np.full((11, 11), 2, dtype=np.int32),
        # Region 2 failed the size/segment gate and is deliberately unmapped.
        "texture_regions": {1: {"tile_name": "wood", "tile_origin_uv": np.array([0.0, 0.1])}},
    }
    tile = np.full((8, 8, 3), [0.9, 0.1, 0.1], dtype=np.float32)

    _, _, colors, applied = _subdivide_and_apply_textures(mesh, context, {"wood": tile})

    assert applied == {}
    np.testing.assert_array_equal(colors, np.tile(standard_rgb, (len(colors), 1)))


def test_periodic_tile_sampling_uses_both_floor_plane_axes():
    tile = np.zeros((8, 8, 3), dtype=np.float32)
    tile[..., 0] = np.arange(8, dtype=np.float32)[None, :] / 7
    tile[..., 1] = np.arange(8, dtype=np.float32)[:, None] / 7

    samples = _sample_periodic_tile(
        tile, np.array([[0.0, 0.1], [0.04, 0.06]]),
        np.array([0.0, 0.1]), 0.01,
    )

    np.testing.assert_allclose(samples[0], [0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(samples[1], [4 / 7, 4 / 7, 0], atol=1e-6)


def test_applies_observed_floor_patterns_to_final_planar_polygon(tmp_path):
    level_dir = tmp_path / "intermediate_floor_levels"
    level_dir.mkdir()
    vertices = np.array([
        [10, 0.25, 10], [11, 0.25, 10], [11, 0.25, 11], [10, 0.25, 11],  # observed floor evidence
        [20, 1.0, 20], [21, 1.0, 20], [21, 1.0, 21], [20, 1.0, 21],  # ramp
        [30, 2.0, 30], [31, 2.0, 30], [31, 2.0, 31], [30, 2.0, 31],  # ceiling
        [40, 0.25, 40], [41, 0.25, 40], [41, 0.25, 41], [40, 0.25, 41],  # fitted preview
        [50, 0.25, 50], [51, 0.25, 50], [51, 0.25, 51], [50, 0.25, 51],  # hole fill
        [60, 0.25, 60], [60, 0.25, 61], [60, 0.45, 61], [60, 0.45, 60],  # wrong-facing floor-role contamination
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [8, 9, 10], [8, 10, 11],
        [12, 13, 14], [12, 14, 15],
        [16, 17, 18], [16, 18, 19],
        [20, 21, 22], [20, 22, 23],
    ], dtype=np.int32)
    source = level_dir / "floor_level_000_fitted.ply"
    assert export_ply_with_face_attributes(
        source, vertices, faces, np.full((len(vertices), 3), [150, 80, 45], dtype=np.uint8),
        face_attributes={
            "surface_role": ("uchar", np.array([0, 0, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0], dtype=np.uint8)),
            "is_fitted_plane": ("uchar", np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.uint8)),
            "is_hole_fill": ("uchar", np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], dtype=np.uint8)),
            "floor_area_id": ("int", np.array([0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0], dtype=np.int32)),
        },
    )
    (level_dir / "floor_level_intermediate_manifest.json").write_text(json.dumps({
        "levels": [{
            "level": 0,
            "fitted_output_file": source.name,
            "floor_areas": [{"area_id": 0, "plane_equation": [0, 1, 0, -0.25]}],
        }],
    }))
    outer = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
    hole = [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]
    (level_dir / "floor_level_000_polygons.json").write_text(json.dumps({
        "level": 0,
        "areas": [{"area_id": 0, "floor_polygon_rings_world_xz_m": [outer, hole]}],
    }))

    for folder in ("floor_texture_tiles", "floor_texture_tiles_3x3"):
        stale = level_dir / folder
        stale.mkdir()
        (stale / "debug.png").write_bytes(b"debug")
    (level_dir / "floor_level_patterns_regularization_manifest.json").write_text("{}")

    _clear_texture_debug_outputs(level_dir)
    results = list(regularize_floor_level_patterns(tmp_path))

    assert len(results) == 1
    record = results[0]
    output_path = level_dir / record["output_file"]
    assert not output_path.exists()
    _write_regularized_floor_ply(level_dir, record)
    assert record["observed_floor_support_triangle_count"] == 2
    assert record["observed_floor_support_triangle_count_by_area"] == {"0": 2}
    assert record["output_geometry"] == "final floor polygon with detected relief raised on its upward side"
    output = PlyData.read(str(output_path))
    output_vertices = output["vertex"].data
    output_faces = output["face"].data
    assert len(output_faces) == record["output_triangle_count"]
    assert all(name in output_vertices.dtype.names for name in ("red", "green", "blue"))
    np.testing.assert_allclose(output_vertices["y"], 0.25, atol=1e-6)
    np.testing.assert_allclose(
        [output_vertices["x"].min(), output_vertices["x"].max(),
         output_vertices["z"].min(), output_vertices["z"].max()],
        [0.1, 0.9, 0.1, 0.9], atol=1e-6,
    )
    assert np.all(output_faces["surface_role"] == 0)
    assert np.all(output_faces["floor_area_id"] == 0)
    assert int(output_vertices["red"].max()) > int(output_vertices["blue"].max())
    triangle_indices = np.vstack(output_faces["vertex_indices"])
    triangles = np.column_stack((output_vertices["x"], output_vertices["y"], output_vertices["z"]))[triangle_indices]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1).sum() / 2
    assert area == pytest.approx(0.60, abs=1e-6)
    assert not (level_dir / "floor_texture_tiles").exists()
    assert not (level_dir / "floor_texture_tiles_3x3").exists()
    assert not (level_dir / "floor_level_patterns_regularization_manifest.json").exists()
