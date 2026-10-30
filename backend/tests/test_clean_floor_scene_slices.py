import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d

STAGE = Path(__file__).resolve().parents[1] / "06_mesh_elaboration"
sys.path.insert(0, str(STAGE))
sys.path.insert(0, str(STAGE.parent / "04_3DGS_to_mesh"))
from fill_mesh_holes import export_ply_with_face_attributes, read_ply_face_flags
import clean_floor_scene_slices as cleaner


def test_cleanup_removes_only_long_elongated_hole_fill_faces_and_is_rerunnable(tmp_path):
    level_dir = tmp_path / "intermediate_floor_levels"
    level_dir.mkdir()
    output_path = level_dir / "floor_level_000_scene_slice.ply"
    vertices = np.array([
        [0., 0., 0.], [1., 0., 0.], [.5, .001, 0.],       # elongated filled face
        [2., 0., 0.], [2.2, 0., 0.], [2.1, .1732, 0.],   # compact filled face
        [4., 0., 0.], [5., 0., 0.], [4.5, .001, 0.],     # elongated observed face
        [6., 0., 0.], [7., 0., 0.], [6.5, .001, 0.],     # fitted preview face
        [8., 0., 0.], [9., 0., 0.], [8.5, .866, 0.],     # long but compact filled face
        [11., 0., 0.], [11.2, 0., 0.], [11.1, .001, 0.], # thin but short filled face
    ], dtype=np.float32)
    triangles = np.arange(18, dtype=np.int32).reshape(-1, 3)
    attributes = {
        "is_hole_fill": ("uchar", np.array([1, 1, 0, 1, 1, 1], dtype=np.uint8)),
        "surface_role": ("uchar", np.array([255, 255, 255, 0, 255, 255], dtype=np.uint8)),
        "is_fitted_plane": ("uchar", np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)),
        "source_class_id": ("int", np.array([41, 41, 41, -1, 41, 41], dtype=np.int32)),
        "floor_area_id": ("int", np.array([-1, -1, -1, 0, -1, -1], dtype=np.int32)),
        "ceiling_group_id": ("int", np.full(6, -1, dtype=np.int32)),
    }
    assert export_ply_with_face_attributes(
        output_path, vertices, triangles, np.full((18, 3), 180, dtype=np.uint8), attributes,
    )
    manifest_path = level_dir / "floor_level_scene_slices_manifest.json"
    manifest_path.write_text(json.dumps({"levels": [{
        "level": 0,
        "output_file": output_path.name,
    }]}))

    cleaner.clean_floor_scene_slices(tmp_path)

    mesh = o3d.io.read_triangle_mesh(str(output_path))
    assert len(mesh.triangles) == 5
    np.testing.assert_array_equal(read_ply_face_flags(output_path, "is_hole_fill"), [1, 0, 1, 1, 1])
    np.testing.assert_array_equal(read_ply_face_flags(output_path, "is_fitted_plane"), [0, 0, 1, 0, 0])
    manifest = json.loads(manifest_path.read_text())
    assert manifest["levels"][0]["elongated_filled_triangles_removed_this_run"] == 1
    assert manifest["elongated_fill_cleanup"]["minimum_longest_edge_m"] == .30

    cleaner.clean_floor_scene_slices(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["levels"][0]["elongated_filled_triangles_removed_this_run"] == 0
    assert len(o3d.io.read_triangle_mesh(str(output_path)).triangles) == 5
