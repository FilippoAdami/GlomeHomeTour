"""Focused geometry checks for the optional mesh-refinement step."""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import open3d as o3d

path = Path(__file__).resolve().parents[1] / "04_3DGS_to_mesh/refine_mesh.py"
spec = importlib.util.spec_from_file_location("refine_mesh", path)
refine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refine)


def test_keep_only_disconnected_components_above_triangle_fraction():
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=2, resolution=30)
    small = o3d.geometry.TriangleMesh.create_box(0.01, 0.01, 0.01)
    small.translate((5, 0, 0))
    large = o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=4)
    large.translate((8, 0, 0))
    mesh += small + large
    original_triangles = len(mesh.triangles)
    summary = refine.remove_floaters(mesh, 0.005)
    assert summary["components_removed"] == 1
    assert summary["triangles_removed"] == 12
    assert len(mesh.triangles) == original_triangles - 12


def test_flat_simplification_preserves_curved_geometry():
    n = 20
    vertices = np.array([[x / (n - 1), 0, z / (n - 1)]
                         for z in range(n) for x in range(n)])
    triangles = []
    for z in range(n - 1):
        for x in range(n - 1):
            i = z * n + x
            triangles.extend(((i, i + 1, i + n), (i + 1, i + n + 1, i + n)))
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                     o3d.utility.Vector3iVector(np.array(triangles)))
    sphere = o3d.geometry.TriangleMesh.create_sphere(0.3, resolution=10)
    sphere.translate((0.5, 2, 0.5))
    mesh += sphere
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile([0.7, 0.5, 0.2], (len(mesh.vertices), 1)))
    simplified = refine.simplify_flat_surfaces(mesh)
    centers = np.asarray(simplified.vertices)[np.asarray(simplified.triangles)].mean(axis=1)
    assert (centers[:, 1] < 1).sum() < len(triangles)
    assert (centers[:, 1] > 1).sum() == len(sphere.triangles)
    assert simplified.has_vertex_colors()
    np.testing.assert_allclose(simplified.get_min_bound(), mesh.get_min_bound())
    np.testing.assert_allclose(simplified.get_max_bound(), mesh.get_max_bound())


def test_floor_and_near_right_angle_snap_exactly():
    floor_normal = np.array([0.02, 0.999, 0.03])
    floor_normal /= np.linalg.norm(floor_normal)
    rotation = refine.rotation_to_up(floor_normal)
    np.testing.assert_allclose(rotation @ floor_normal, [0, 1, 0], atol=1e-12)

    def patch(normal):
        basis = np.eye(3)
        basis[:, 2] = normal
        return SimpleNamespace(R=basis, extent=np.array([1., 1., .02]))

    angle = math.radians(88)
    patches = [patch(np.array([0., 1., 0.])),
               patch(np.array([1., .02, 0.])),
               patch(np.array([math.cos(angle), .01, math.sin(angle)]))]
    normals, families = refine.target_normals(patches, np.eye(3))
    assert families == 1
    np.testing.assert_allclose(normals[0], [0, 1, 0], atol=1e-12)
    assert abs(np.dot(normals[1], normals[2])) < 1e-12
    assert abs(np.dot(normals[0], normals[1])) < 1e-12
