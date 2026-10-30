import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d

STAGE = Path(__file__).resolve().parents[1] / "06_mesh_elaboration"
sys.path.insert(0, str(STAGE))
from fill_mesh_holes import export_ply_with_face_attributes, read_ply_face_flags
import slice_scene_by_floor as slicer


def _write_preview(path, points, role, id_property, plane_id):
    return export_ply_with_face_attributes(
        path, np.asarray(points, dtype=np.float32), np.array([[0, 1, 2]], dtype=np.int32),
        face_attributes={
            "is_hole_fill": ("uchar", np.array([0], dtype=np.uint8)),
            "surface_role": ("uchar", np.array([role], dtype=np.uint8)),
            "is_fitted_plane": ("uchar", np.array([1], dtype=np.uint8)),
            id_property: ("int", np.array([plane_id], dtype=np.int32)),
        },
    )


def _write_scene_mesh(path):
    vertices, triangles, labels = [], [], []

    def face(points, class_id):
        start = len(vertices)
        vertices.extend(points)
        triangles.append([start, start + 1, start + 2])
        labels.append(class_id)

    def horizontal(y, z, class_id):
        face([[0, y, z], [.1, y, z], [0, y, z + .1]], class_id)

    horizontal(0., 0., 17)       # level 0 floor
    horizontal(2., 0., 17)       # level 1 floor
    horizontal(-.25, 0., 41)    # below the lower floor cut
    horizontal(.5, 0., 41)
    horizontal(1.75, 0., 41)    # below level 1 floor cut
    horizontal(1.85, 0., 41)
    horizontal(2.5, 0., 41)
    horizontal(3.5, 0., 41)     # below tilted ceiling cut at z=0
    horizontal(3.5, 1., 41)     # above tilted ceiling cut at z=1
    horizontal(1.9, 0., 23)     # ceiling shared as level 0 top / level 1 lower ceiling

    # A source ceiling face on the tilted level 1 upper plane.
    point = np.array([0., 3.7 / .98, 0.])
    tangent_z = np.array([0., -.2, .98])
    face([point, point + [.1, 0., 0.], point + .1 * tangent_z], 23)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float32))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    mesh.paint_uniform_color([.7, .7, .7])
    assert o3d.io.write_triangle_mesh(str(path), mesh)
    return np.asarray(labels, dtype=np.int32)


def test_scene_slices_use_floor_or_wall_height_bounds_without_ceiling_geometry(tmp_path):
    level_dir = tmp_path / "intermediate_floor_levels"
    level_dir.mkdir()
    mesh_path = tmp_path / "mesh.ply"
    labels = _write_scene_mesh(mesh_path)
    labels_path = tmp_path / "triangle_class_labels.npy"
    np.save(labels_path, labels)
    class_metadata_path = tmp_path / "class_metadata.json"
    class_metadata_path.write_text(json.dumps({"classifications": {
        "17": {"structural_role": "floor"},
        "23": {"structural_role": "ceiling"},
        "41": {"structural_role": "wall"},
    }}))

    floor_equations = [[0., 1., 0., 0.], [0., 1., 0., -2.]]
    floor_levels = []
    for level_id, equation, floor_y in zip((0, 1), floor_equations, (0., 2.)):
        filename = f"floor_level_{level_id:03d}_fitted.ply"
        assert _write_preview(
            level_dir / filename,
            [[0., floor_y, 0.], [.2, floor_y, 0.], [0., floor_y, .2]],
            0, "floor_area_id", 0,
        )
        floor_levels.append({
            "level": level_id,
            "floor_triangle_centroid_y_median": floor_y,
            "floor_areas": [{"plane_equation": equation}],
            "fitted_output_file": filename,
        })
    (level_dir / "floor_level_intermediate_manifest.json").write_text(json.dumps({
        "ceiling_match_max_below_m": .5,
        "levels": floor_levels,
    }))
    (tmp_path / "wall_planes_manifest.json").write_text(json.dumps({
        "wall_vertical_bounds_m": [-.1, 4.0],
    }))

    ceiling_specs = [
        (0, [0., 1., 0., -1.9], 1, 1.9),
        (1, [0., .98, .2, -3.7], 1, 3.75),
    ]
    ceiling_groups = []
    for group_id, equation, associated_level, y_at_zero in ceiling_specs:
        filename = f"ceiling_plane_{group_id:03d}_fitted.ply"
        if group_id == 0:
            points = [[0., y_at_zero, 0.], [.2, y_at_zero, 0.], [0., y_at_zero, .2]]
        else:
            point = np.array([0., 3.7 / .98, 0.])
            points = [point, point + [.2, 0., 0.], point + .2 * np.array([0., -.2, .98])]
        assert _write_preview(level_dir / filename, points, 2, "ceiling_group_id", group_id)
        ceiling_groups.append({
            "group_id": group_id,
            "associated_floor_level": associated_level,
            "fitted_output_file": filename,
            "plane_equation": equation,
        })
    (level_dir / "ceiling_plane_groups_manifest.json").write_text(json.dumps({"groups": ceiling_groups}))

    manifest_path = slicer.slice_scene_by_floor(mesh_path, labels_path, class_metadata_path, tmp_path)
    manifest = json.loads(manifest_path.read_text())
    records = {record["level"]: record for record in manifest["levels"]}
    assert records[0]["upper_boundary_source"] == "next_floor_level"
    assert records[1]["upper_boundary_source"] == "wall_enclosure_top"

    level0_path = level_dir / records[0]["output_file"]
    level1_path = level_dir / records[1]["output_file"]
    level0 = o3d.io.read_triangle_mesh(str(level0_path))
    level1 = o3d.io.read_triangle_mesh(str(level1_path))
    source_ids0 = read_ply_face_flags(level0_path, "source_class_id")
    source_ids1 = read_ply_face_flags(level1_path, "source_class_id")
    fitted0 = read_ply_face_flags(level0_path, "is_fitted_plane")
    fitted1 = read_ply_face_flags(level1_path, "is_fitted_plane")
    vertices0, triangles0 = np.asarray(level0.vertices), np.asarray(level0.triangles)
    vertices1, triangles1 = np.asarray(level1.vertices), np.asarray(level1.triangles)

    # Ceiling-role geometry never enters floor projection slices.
    assert not np.any((source_ids0 == 23) & (fitted0 == 0))
    assert not np.any((source_ids1 == 23) & (fitted1 == 0))
    assert np.count_nonzero(fitted0 == 1) == 1
    assert np.count_nonzero(fitted1 == 1) == 1

    # Level 0 ends at the next floor; the top level uses the wall enclosure height.
    for path, vertices, triangles, lower_y, upper_plane in (
        (level0_path, vertices0, triangles0, -.2, np.array([0., 1., 0., -1.8])),
        (level1_path, vertices1, triangles1, 1.8, np.array([0., 1., 0., -3.8])),
    ):
        source_faces = np.flatnonzero(read_ply_face_flags(path, "is_fitted_plane") == 0)
        faces = triangles[source_faces]
        points = vertices[faces]
        assert np.all(points[:, :, 1] >= lower_y - 1e-5)
        assert np.all(points @ upper_plane[:3] + upper_plane[3] <= 1e-5)

    centers0 = vertices0[triangles0].mean(axis=1)
    centers1 = vertices1[triangles1].mean(axis=1)
    wall0 = centers0[(source_ids0 == 41) & (fitted0 == 0)]
    wall1 = centers1[(source_ids1 == 41) & (fitted1 == 0)]
    assert not np.any(np.isclose(wall0[:, 1], -.25))
    assert np.any(np.isclose(wall1[:, 1], 3.5) & (wall1[:, 2] < .2))
    assert np.any(np.isclose(wall1[:, 1], 3.5) & (wall1[:, 2] > .9))
    assert records[1]["retained_ceiling_class_triangles"] == 0

    stale = level_dir / "floor_level_999_scene_slice.ply"
    stale.touch()
    slicer.slice_scene_by_floor(mesh_path, labels_path, class_metadata_path, tmp_path)
    assert not stale.exists()
