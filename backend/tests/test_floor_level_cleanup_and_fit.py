import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
from plyfile import PlyData


STAGE_DIR = Path(__file__).resolve().parents[1] / "06_mesh_elaboration"
sys.path.insert(0, str(STAGE_DIR))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, STAGE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cleaning = _load("clean_floor_level_intermediates")
fitting = _load("fit_floor_level_planes")


def _write_level(path, vertices, triangles, roles, flags=None):
    flags = np.zeros(len(triangles), dtype=np.uint8) if flags is None else np.asarray(flags, dtype=np.uint8)
    assert cleaning.export_ply_with_face_attributes(
        path, np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=np.int32),
        np.full((len(vertices), 3), 160, dtype=np.uint8),
        face_attributes={
            "is_hole_fill": ("uchar", flags),
            "surface_role": ("uchar", np.asarray(roles, dtype=np.uint8)),
        },
    )


def _manifest(output, ramps):
    output.mkdir(exist_ok=True)
    (output / "floor_level_intermediate_manifest.json").write_text(json.dumps({
        "floor_count": 1,
        "levels": [{"level": 0, "output_file": "floor_level_000_intermediate.ply",
                    "instances": [{"file_name": "floor_inst000.ply"}], "ramps": ramps}],
    }))


def test_cleanup_removes_far_verticals_and_outliers_but_keeps_near_plane_vertical_faces(tmp_path):
    output = tmp_path / "intermediate_floor_levels"
    _manifest(output, [])
    vertices = [
        [0, 0, 0], [2, 0, 0], [2, 0, 2], [0, 0, 2],
        [3, 0, 0], [3, 0.3, 0], [3, 0, 0.3],
        [4, 0, 0], [4, 1, 0], [4, 0, 1],
        [5, 0, 0], [5.1, 0, 0], [5, 0, 0.1],
        [7, 1, 0], [7.1, 1, 0], [7, 1, 0.1],
    ]
    triangles = [[0, 1, 2], [0, 2, 3], [4, 5, 6], [7, 8, 9],
                 [10, 11, 12], [13, 14, 15]]
    _write_level(output / "floor_level_000_intermediate.ply", vertices, triangles,
                 [0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0])

    cleaning.clean_floor_level_intermediates(tmp_path)

    path = output / "floor_level_000_intermediate.ply"
    mesh = o3d.io.read_triangle_mesh(str(path))
    assert len(mesh.triangles) == 4
    faces = PlyData.read(str(path))["face"].data
    np.testing.assert_array_equal(faces["is_hole_fill"], [0, 1, 0, 0])
    np.testing.assert_array_equal(faces["surface_role"], [0, 0, 0, 0])
    level = json.loads((output / "floor_level_intermediate_manifest.json").read_text())["levels"][0]
    assert level["vertical_triangles_removed"] == 1
    assert level["other_outlier_triangles_removed"] == 1
    assert level["outlier_triangles_removed"] == 2
    assert level["num_triangles"] == 4


def test_fits_floor_only_regions_and_appends_plane_faces(tmp_path):
    output = tmp_path / "intermediate_floor_levels"
    ramp = [{"file_name": "ramp_inst000.ply"}]
    _manifest(output, ramp)
    slope = np.tan(np.deg2rad(7.0))
    vertices = [
        [0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1],
        [3, 2 * slope, 0], [3, 2 * slope, 1], [4, 3 * slope, 0], [4, 3 * slope, 1],
        [80, 3, 0], [90, 3, 0], [90, 3, 1], [80, 3, 1],
    ]
    triangles = [
        [0, 1, 2], [0, 2, 3],                    # lower floor area
        [1, 4, 5], [1, 5, 2],                    # ramp barrier
        [4, 6, 7], [4, 7, 5],                    # 7 degree floor area
        [8, 9, 10], [8, 10, 11],                 # ceiling, excluded from fitting
    ]
    _write_level(output / "floor_level_000_intermediate.ply", vertices, triangles,
                 [0, 0, 1, 1, 0, 0, 2, 2])

    fitting.fit_floor_level_planes(tmp_path)

    path = output / "floor_level_000_fitted.ply"
    faces = PlyData.read(str(path))["face"].data
    np.testing.assert_array_equal(faces["surface_role"], [0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 0, 0])
    np.testing.assert_array_equal(faces["is_fitted_plane"], [0] * 8 + [1] * 4)
    np.testing.assert_array_equal(faces["floor_area_id"], [0, 0, -1, -1, 1, 1, -1, -1, 0, 0, 1, 1])
    manifest = json.loads((output / "floor_level_intermediate_manifest.json").read_text())
    planes = manifest["levels"][0]["floor_areas"]
    assert len(planes) == 2
    assert planes[0]["snapped_horizontal"] is True
    assert planes[0]["tilt_deg"] == 0.0
    assert planes[1]["snapped_horizontal"] is False
    assert planes[1]["tilt_deg"] == pytest.approx(7.0, abs=0.05)
    fitted_ply = PlyData.read(str(path))
    fitted_vertices = np.column_stack([
        fitted_ply["vertex"].data[name] for name in ("x", "y", "z")
    ])
    preview_faces = np.asarray(fitted_ply["face"].data["is_fitted_plane"], dtype=bool)
    preview_vertices = fitted_vertices[np.unique(
        np.concatenate(fitted_ply["face"].data["vertex_indices"][preview_faces])
    )]
    assert preview_vertices[:, 0].max() <= 4.0 + 1e-6
    assert preview_vertices[:, 2].max() <= 1.0 + 1e-6


def test_no_ramp_means_one_plane_even_for_disconnected_floor_patches(tmp_path):
    output = tmp_path / "intermediate_floor_levels"
    _manifest(output, [])
    vertices = [[0, 0, 0], [1, 0, 0], [0, 0, 1], [3, 0, 0], [4, 0, 0], [3, 0, 1]]
    _write_level(output / "floor_level_000_intermediate.ply", vertices,
                 [[0, 1, 2], [3, 4, 5]], [0, 0])

    fitting.fit_floor_level_planes(tmp_path)

    manifest = json.loads((output / "floor_level_intermediate_manifest.json").read_text())
    assert manifest["levels"][0]["num_fitted_planes"] == 1
