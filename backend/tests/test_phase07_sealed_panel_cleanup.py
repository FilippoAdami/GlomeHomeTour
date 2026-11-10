import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import plyfile
import trimesh


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "07_alignment"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "08_artifacts_generation"))
import step_replace_red_regions_with_panels as replacement
from panorama_rendering import inspect_panel_materials, panel_faces


def _mesh(triangles):
    vertices = np.asarray(triangles, dtype=np.float64).reshape(-1, 3)
    return trimesh.Trimesh(vertices=vertices, faces=np.arange(len(vertices)).reshape(-1, 3), process=False)


def test_panel_projection_removes_intersections_and_preserves_front_and_outside():
    corners = np.array([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]], dtype=float)
    mesh = _mesh([
        [[1, 0, 0], [1, .5, 0], [1, 0, .5]],  # entirely behind
        [[1, 0, 0], [-1, .5, 0], [-1, 0, .5]],  # only one vertex behind
        [[-1, 0, 0], [-1, .5, 0], [-1, 0, .5]],  # in front
        [[1, 2, 0], [1, 2, .5], [1, 3, 0]],  # outside the projection
        [[1, -2, 0], [1, 2, 0], [1, 0, 2]],  # crosses, no vertex inside
        [[1, 1.2, .9], [1, .9, 1.2], [1, 2, 2]],  # bounding boxes overlap only
        [[0, -1, -1], [0, 1, -1], [0, 1, 1]],  # panel itself
        [[1, -2, 0], [1, 2, 0], [-1, 0, 0]],  # crosses both depth and projection
    ])
    actual = replacement._faces_behind_panel(mesh.vertices, mesh.faces, corners, np.array([1, 0, 0]))
    assert actual.tolist() == [True, True, False, False, True, False, False, True]
    for order in ([1, 2, 0], [2, 0, 1]):
        rotated = replacement._faces_behind_panel(
            np.asarray(mesh.vertices)[:, order], mesh.faces, corners[:, order], np.array([1, 0, 0])[order]
        )
        np.testing.assert_array_equal(rotated, actual)
    reverse = replacement._faces_behind_panel(mesh.vertices, mesh.faces, corners, np.array([-1, 0, 0]))
    assert reverse[2] and not reverse[0]


def test_panel_projection_uses_the_placed_tilted_plane():
    corners = np.array([[-.2, -1, -1], [.2, 1, -1], [.2, 1, 1], [-.2, -1, 1]])
    mesh = _mesh([
        [[0, -.5, 0], [0, -.6, 0], [0, -.5, .1]],
        [[0, .5, 0], [0, .6, 0], [0, .5, .1]],
    ])
    assert replacement._faces_behind_panel(mesh.vertices, mesh.faces, corners, np.array([1, 0, 0])).tolist() == [True, False]


def test_panel_projection_ignores_float32_rounding_of_its_own_plane():
    corners = np.array([[2.2851971115, 0, 0], [2.2851971115, 1, 0],
                        [2.2851971115, 1, 1], [2.2851971115, 0, 1]])
    vertices = corners.astype(np.float32).astype(np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    assert not replacement._faces_behind_panel(vertices, faces, corners, np.array([-1, 0, 0])).any()


def test_replacement_cleans_only_the_panel_scan_and_publishes_all_meshes(tmp_path, monkeypatch):
    scene = tmp_path / "scene"
    alignment = scene / "multi_room_alignment"
    stats = alignment / "stats"
    stats.mkdir(parents=True)
    triangles = np.array([
        [[1, .2, 0], [1, .4, .2], [1, .2, .4]],
        [[1, .5, 0], [1, .6, .1], [1, .4, .2]],
        [[1, .7, 0], [1, .8, .1], [1, .7, .2]],
        [[-1, .3, 0], [-1, .4, .2], [-1, .3, .4]],
    ])
    scans = {}
    for scan_id in ("scan_0", "scan_1"):
        source_dir = scene / scan_id
        (source_dir / "segmentation").mkdir(parents=True)
        scan_mesh_dir = alignment / "scans" / scan_id
        scan_mesh_dir.mkdir(parents=True)
        _mesh(triangles).export(scan_mesh_dir / "mesh.ply")
        _mesh(triangles).export(source_dir / "segmentation" / "mesh_segmented.ply")
        scans[scan_id] = SimpleNamespace(info=SimpleNamespace(scan_id=scan_id, scan_dir=source_dir))
    trimesh.util.concatenate([_mesh(triangles), _mesh(triangles)]).export(alignment / "aligned_sealed_meshes.ply")

    objects = []
    for index in range(2):
        object_path = alignment / f"objects/scan_0/class_003_object/instance_{index:03d}.ply"
        object_path.parent.mkdir(parents=True, exist_ok=True)
        _mesh(triangles[[index, index + 2]]).export(object_path)
        objects.append({
            "scan_id": "scan_0", "global_instance_path": str(object_path),
            "source_instance_path": str(object_path), "class_id": 3, "triangles": 2,
            "post_scan_translation_m": [0, 0, 0],
        })
    (stats / "aligned_scans_transforms.json").write_text(json.dumps({
        "scans": {scan_id: {"transform_matrix": np.eye(4).tolist()} for scan_id in scans}
    }))
    (stats / "aligned_objects_manifest.json").write_text(json.dumps({"objects": objects}))
    vertex_data = np.array([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.)],
                           dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    face_data = np.zeros(2, dtype=[("vertex_indices", "i4", (3,)), ("is_phase07_outlier", "u1")])
    face_data["vertex_indices"] = [[0, 1, 2], [0, 1, 2]]
    face_data["is_phase07_outlier"] = 1
    plyfile.PlyData([plyfile.PlyElement.describe(vertex_data, "vertex"),
                     plyfile.PlyElement.describe(face_data, "face")]).write(alignment / "aligned_objects_final_red.ply")

    groups = [[{"scan_id": "scan_0", "source": f"class_003_object/instance_{index:03d}.ply",
                "triangles": triangles[index:index + 1], "face_ids": np.array([0]),
                "plane_offset": index * .5}] for index in range(2)]
    floor_path = alignment / "floor.ply"
    monkeypatch.setattr(replacement, "ScanDataset", lambda _: SimpleNamespace(scans=scans))
    monkeypatch.setattr(replacement, "_source_face_ids", lambda *args: (
        np.array([0, 2]) if args[2].stem == "instance_000" else np.array([1, 3])
    ))
    monkeypatch.setattr(replacement, "_outward_direction", lambda *args: np.array([1., 0., 0.]))
    monkeypatch.setattr(replacement, "_fit_panel_to_mesh_and_floor", lambda corners, *args: (corners, np.zeros(3)))
    monkeypatch.setattr(replacement.contours, "_red_regions", lambda *args: groups)
    monkeypatch.setattr(replacement.contours, "_build_exterior_surface", lambda *args: {})
    monkeypatch.setattr(replacement.contours, "_dominant_plane", lambda group, *_: {
        "normal": np.array([1., 0., 0.]), "offset": group[0]["plane_offset"]
    })
    monkeypatch.setattr(replacement.contours, "_triangle_union", lambda *args: SimpleNamespace(area=1.))
    monkeypatch.setattr(replacement.contours, "_minimum_axis_aligned_box", lambda *args: ((-1., 1., 0., 2.), 1., 1., 0.))
    monkeypatch.setattr(replacement.classification, "_window_classes", lambda *args: ({}, set()))
    monkeypatch.setattr(replacement.classification, "_bbox_low_corner", lambda *args: np.array([0., 2., 0.]))
    monkeypatch.setattr(replacement.classification, "_floor_planes", lambda *args: [{"path": floor_path}])
    monkeypatch.setattr(replacement.classification, "_closest_floor_height", lambda *args: (0., floor_path))
    monkeypatch.setattr(replacement.classification, "_instance_classes", lambda *args: set())

    manifest = replacement.run(scene, alignment)

    assert manifest["removed_sealed_face_count"] == 2
    assert manifest["removed_sealed_behind_panel_face_count_by_scan"] == {"scan_0": 3, "scan_1": 0}
    assert [region["sealed_panels"][0]["retained_face_indices"] for region in manifest["regions"]] == [[0, 1], []]
    scan0 = replacement._mesh(alignment / "scans/scan_0/mesh.ply")
    scan1 = replacement._mesh(alignment / "scans/scan_1/mesh.ply")
    assert len(scan0.faces) == 3  # one front face and two panel faces
    np.testing.assert_allclose(scan0.triangles[0], triangles[3])
    np.testing.assert_allclose(scan1.triangles, triangles)
    kinds, panel_ids = panel_faces("scan_0", scan0.vertices, scan0.faces, manifest["regions"])
    assert kinds.tolist() == [0, 2, 2] and panel_ids.tolist() == [-1, 0, 0]
    assert inspect_panel_materials(alignment, "scan_0")["sealed_panel_count"] == 1
    assert len(replacement._mesh(alignment / "aligned_sealed_meshes.ply").faces) == 7
    scene_glb = trimesh.load(alignment / "aligned_sealed_meshes.glb", force="scene", process=False)
    assert sum(len(mesh.faces) for mesh in scene_glb.geometry.values()) == 7


def test_panel_face_reader_handles_a_partially_removed_panel():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    faces = np.array([[0, 2, 3]])
    regions = [{"scan_id": "scan_0", "class": "mirror", "sealed_panels": [
        {"plane_corners": vertices.tolist(), "retained_face_indices": [1]}
    ]}]
    kinds, panel_ids = panel_faces("scan_0", vertices, faces, regions)
    assert kinds.tolist() == [1] and panel_ids.tolist() == [0]
