import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData


STAGE_DIR = Path(__file__).resolve().parents[1] / "06_mesh_elaboration"
sys.path.insert(0, str(STAGE_DIR))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, STAGE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aggregation = _load("aggregate_fit_ceiling_planes")
floor_fitting = _load("fit_floor_level_planes")


def _write_mesh(path, vertices, triangles):
    assert aggregation.export_ply_with_face_attributes(
        path, np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=np.int32), None,
        face_attributes={"is_hole_fill": ("uchar", np.zeros(len(triangles), dtype=np.uint8))},
    )


def _sloped_patch(x0, x1, intercept, slope):
    return [
        [x0, intercept + slope * x0, 0], [x1, intercept + slope * x1, 0],
        [x1, intercept + slope * x1, 1], [x0, intercept + slope * x0, 1],
    ], [[0, 1, 2], [0, 2, 3]]


def _ceiling_preview_quad(path):
    ply = PlyData.read(str(path))
    vertices = np.column_stack([ply["vertex"][axis] for axis in ("x", "y", "z")])
    faces = ply["face"].data
    ids = np.flatnonzero(faces["is_fitted_plane"] == 1)
    return vertices[np.unique(np.concatenate(faces["vertex_indices"][ids]))]


def test_aggregates_coplanar_ceilings_and_transfers_associated_floor_dimensions(tmp_path):
    levels = tmp_path / "intermediate_floor_levels"
    levels.mkdir()
    floor_vertices = [[0, 0, 0], [2, 0, 0], [2, 0, 3], [0, 0, 3]]
    _write_mesh(levels / "floor_level_000_intermediate.ply", floor_vertices, [[0, 1, 2], [0, 2, 3]])
    (levels / "floor_level_intermediate_manifest.json").write_text(json.dumps({
        "levels": [{
            "level": 0,
            "output_file": "floor_level_000_intermediate.ply",
            "ramps": [],
            "ceilings": [{"file_name": "ceiling_inst003.ply"}],
        }],
    }))
    floor_fitting.fit_floor_level_planes(tmp_path)
    floor_quad = aggregation._floor_preview_quad(tmp_path, {
        "level": 0,
        "fitted_output_file": "floor_level_000_fitted.ply",
        "floor_areas": [{"area_id": 0}],
    })
    expected_floor_dimensions = [np.linalg.norm(floor_quad[1] - floor_quad[0]),
                                 np.linalg.norm(floor_quad[3] - floor_quad[0])]

    slope = np.tan(np.deg2rad(7.0))
    for filename, x0, x1, intercept in [
        ("ceiling_inst000.ply", 0, 1, 2.0),
        ("ceiling_inst001.ply", 1, 2, 2.0),
        ("ceiling_inst002.ply", 2, 3, 2.5),
    ]:
        vertices, triangles = _sloped_patch(x0, x1, intercept, slope)
        _write_mesh(tmp_path / filename, vertices, triangles)
    horizontal = [[0, -0.1, 0], [1, -0.1, 0], [1, -0.1, 1], [0, -0.1, 1]]
    _write_mesh(tmp_path / "ceiling_inst003.ply", horizontal, [[0, 1, 2], [0, 2, 3]])
    (tmp_path / "structural_instances_manifest.json").write_text(json.dumps({
        "instances": [{"role": "ceiling", "file_name": f"ceiling_inst{i:03d}.ply"}
                      for i in range(4)],
    }))

    manifest_path = aggregation.aggregate_fit_ceiling_planes(tmp_path)

    manifest = json.loads(manifest_path.read_text())
    groups = manifest["groups"]
    assert len(groups) == 3
    merged = next(group for group in groups
                  if group["source_ceiling_instances"] == ["ceiling_inst000.ply", "ceiling_inst001.ply"])
    assert merged["associated_floor_level"] is None
    assert merged["snapped_horizontal"] is False
    assert merged["tilt_deg"] == pytest.approx(7.0, abs=0.05)

    associated = next(group for group in groups if group["associated_floor_level"] == 0)
    assert associated["source_ceiling_instances"] == ["ceiling_inst003.ply"]
    assert associated["snapped_horizontal"] is True
    assert associated["tilt_deg"] == 0.0
    np.testing.assert_allclose(associated["dimensions_m"], expected_floor_dimensions, atol=1e-6)
    assert associated["dimensions_source"] == "floor_level_000_fitted.ply"

    output = PlyData.read(str(levels / associated["fitted_output_file"]))
    faces = output["face"].data
    assert int(np.count_nonzero(faces["is_fitted_plane"])) == 2
    assert np.all(faces["surface_role"] == 2)
    ceiling_quad = _ceiling_preview_quad(levels / associated["fitted_output_file"])
    np.testing.assert_allclose(
        [np.linalg.norm(ceiling_quad[1] - ceiling_quad[0]),
         np.linalg.norm(ceiling_quad[3] - ceiling_quad[0])],
        expected_floor_dimensions, atol=1e-6,
    )


def test_joint_fit_groups_noisy_patches_with_different_local_normals():
    target_slope = np.tan(np.deg2rad(25.0))
    items = []
    for index, local_slope in enumerate([target_slope - 0.20, target_slope + 0.20,
                                         target_slope - 0.12, target_slope + 0.12]):
        x0, x1 = index * 0.3, (index + 1) * 0.3
        midpoint = (x0 + x1) * 0.5
        intercept = (target_slope - local_slope) * midpoint
        vertices, triangles = _sloped_patch(x0, x1, intercept, local_slope)
        vertices = np.asarray(vertices, dtype=float)
        triangles = np.asarray(triangles, dtype=np.int32)
        items.append({
            "file_name": f"ceiling_inst{index:03d}.ply",
            "vertices": vertices,
            "triangles": triangles,
            "colors": None,
            "flags": np.zeros(len(triangles), dtype=np.uint8),
            "floor_level": None,
        })

    individual_normals = [aggregation._fit_ceiling_plane(item["vertices"], item["triangles"])["normal"]
                          for item in items]
    assert max(np.rad2deg(np.arccos(np.clip(individual_normals[0] @ normal, -1.0, 1.0)))
               for normal in individual_normals[1:]) > 5.0

    groups = aggregation._ceiling_groups(items)

    assert len(groups) == 1
    assert len(groups[0]) == 4


def test_ceiling_cleanup_keeps_near_plane_vertical_faces_and_removes_far_ones(tmp_path):
    vertices = [
        [0, 2, 0], [2, 2, 0], [2, 2, 2], [0, 2, 2],
        [3, 2, 0], [3, 2.3, 0], [3, 2, 0.3],
        [4, 3, 0], [4, 4, 0], [4, 3, 1],
    ]
    triangles = [[0, 1, 2], [0, 2, 3], [4, 5, 6], [7, 8, 9]]
    _write_mesh(tmp_path / "ceiling_inst000.ply", vertices, triangles)

    item = aggregation._read_mesh(tmp_path, "ceiling_inst000.ply")

    assert len(item["triangles"]) == 3
    assert item["cleanup"]["far_vertical_triangles_removed"] == 1
