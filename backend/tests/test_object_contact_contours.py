import importlib.util
from pathlib import Path
import sys

import numpy as np
import open3d as o3d


SCRIPT = (Path(__file__).resolve().parents[1] / "07_alignment"
          / "temporary_object_contact_contours" / "discover_contact_contours.py")
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("discover_contact_contours", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_only_instance_faces_touching_different_owners_are_boundary_faces():
    vertices = np.zeros((5, 3), dtype=np.float64)
    faces = np.array([[0, 1, 2], [2, 1, 3], [3, 1, 4]])
    class_ids = np.array([7, 7, -1])
    instance_ids = np.array([1, 1, 0])

    boundary = MODULE.sealed_mesh_boundary_faces(vertices, faces, class_ids, instance_ids)

    np.testing.assert_array_equal(boundary, [False, True, True])


def test_split_instance_faces_map_back_to_their_sealed_parent(tmp_path):
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1],
    ], dtype=float)
    sealed = o3d.geometry.TriangleMesh()
    sealed.vertices = o3d.utility.Vector3dVector(vertices)
    sealed.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [3, 4, 5]])
    source = o3d.geometry.TriangleMesh()
    source.vertices = o3d.utility.Vector3dVector(vertices[3:])
    source.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
    source_path = tmp_path / "split-door.ply"
    o3d.io.write_triangle_mesh(str(source_path), source)

    face_indices, source_faces = MODULE.instance_face_indices(
        source_path, sealed, np.array([2, 2]), np.array([1, 1]), 2, 1
    )

    np.testing.assert_array_equal(face_indices, [1])
    np.testing.assert_array_equal(
        np.asarray(sealed.vertices)[np.asarray(sealed.triangles)[face_indices]],
        np.asarray(source.vertices)[source_faces],
    )


def test_red_contour_faces_keep_other_instance_rgb(tmp_path):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float))
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2], [2, 1, 3]]))
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.tile([0.2, 0.4, 0.6], (4, 1)))
    path = tmp_path / "contour.ply"

    MODULE.write_instance(path, mesh, np.array([False, True]))

    result = o3d.io.read_triangle_mesh(str(path))
    faces = np.asarray(result.triangles)
    colors = np.asarray(result.vertex_colors)[faces].mean(axis=1)
    np.testing.assert_allclose(colors[0], [0.2, 0.4, 0.6], atol=1 / 255)
    np.testing.assert_allclose(colors[1], [1.0, 0.0, 0.0], atol=1 / 255)


def test_red_face_touching_another_instance_loses_highlight():
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float))
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]]))
    other = o3d.geometry.TriangleMesh()
    other.vertices = o3d.utility.Vector3dVector(
        np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float))
    other.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]]))

    filtered, removed = MODULE.suppress_object_contacts(mesh, [other], np.array([True]))

    np.testing.assert_array_equal(filtered, [False])
    assert removed == 1


def test_red_face_stays_highlighted_when_other_instance_is_separate():
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float))
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]]))
    other = o3d.geometry.TriangleMesh()
    other.vertices = o3d.utility.Vector3dVector(
        np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=float))
    other.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]]))

    filtered, removed = MODULE.suppress_object_contacts(mesh, [other], np.array([True]))

    np.testing.assert_array_equal(filtered, [True])
    assert removed == 0
