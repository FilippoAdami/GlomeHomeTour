from pathlib import Path
import sys
import json

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_3DGS_to_mesh"))
from fill_mesh_holes import (
    CoplanarTextureSampler,
    boundary_loops,
    derive_infilled_face_textures,
    edge_keys,
    fill_mesh,
    main,
    patch_loop,
)


def test_fills_only_selected_hole_and_preserves_colors():
    mesh = o3d.geometry.TriangleMesh.create_box()
    original = np.asarray(mesh.triangles).copy()
    mesh.triangles = o3d.utility.Vector3iVector(original[[i for i in range(len(original)) if i not in (2, 3, 6, 7)]])
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.ones((len(mesh.vertices), 3)) * [0.2, 0.4, 0.6])
    assert len(boundary_loops(np.asarray(mesh.triangles))) == 2
    region = {"name": "selected", "min": [-0.1, -0.1, -0.1], "max": [0.1, 1.1, 1.1],
              "max_diameter_m": 2, "max_plane_deviation_m": 0.001}
    filled, records = fill_mesh(mesh, [region])
    assert sum(r["filled_region"] is not None for r in records) == 1
    assert len(filled.triangles) == len(mesh.triangles) + 6
    assert len(boundary_loops(np.asarray(filled.triangles))) == 1
    depths = np.abs(np.asarray(filled.vertices)[len(mesh.vertices):, 0])
    assert np.all((depths > 0) & (depths <= 0.015))
    np.testing.assert_array_equal(np.asarray(filled.vertex_colors)[:len(mesh.vertices)], np.asarray(mesh.vertex_colors))
    np.testing.assert_allclose(np.asarray(filled.vertex_colors)[len(mesh.vertices):], [[0.2, 0.4, 0.6]] * 2)
    assert len(np.asarray(mesh.triangles)) == len(original) - 4


def test_exports_colored_and_red_full_meshes(tmp_path):
    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[2:])
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) / 2)
    source = tmp_path / "source.ply"
    colored, red = tmp_path / "colored.ply", tmp_path / "red.ply"
    regions = tmp_path / "regions.json"
    o3d.io.write_triangle_mesh(str(source), mesh)
    regions.write_text(json.dumps([{"name": "cap", "min": [-1, -1, -1], "max": [2, 2, 2],
                                    "max_diameter_m": 2, "max_plane_deviation_m": 0.01}]))
    main(["--input", str(source), "--regions", str(regions), "--output", str(colored),
          "--red-output", str(red), "--report", str(tmp_path / "report.json")])
    a, b = o3d.io.read_triangle_mesh(str(colored)), o3d.io.read_triangle_mesh(str(red))
    assert len(a.triangles) == len(b.triangles) > len(mesh.triangles)
    assert len(a.vertices) == len(b.vertices) > len(mesh.vertices)
    patch_vertices = np.unique(np.asarray(b.triangles)[len(mesh.triangles):])
    np.testing.assert_allclose(np.asarray(b.vertex_colors)[patch_vertices],
                               [[1, 0, 0]] * len(patch_vertices))
    unchanged = np.setdiff1d(np.arange(len(mesh.vertices)), patch_vertices)
    np.testing.assert_allclose(np.asarray(a.vertex_colors)[unchanged],
                               np.asarray(b.vertex_colors)[unchanged])
    assert np.any(np.asarray(a.vertex_colors)[len(mesh.vertices):, 1] > 0)


def test_rejects_nonplanar_or_oversized_hole():
    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[2:])
    vertices = np.asarray(mesh.vertices)
    vertices[0, 0] += 0.1
    region = {"name": "small", "min": [-1, -1, -1], "max": [2, 2, 2],
              "max_diameter_m": 0.5, "max_plane_deviation_m": 0.001}
    filled, records = fill_mesh(mesh, [region])
    assert len(filled.triangles) == len(mesh.triangles)
    assert records[0]["filled_region"] is None


def sheet_with_gap(dx, dy):
    xs, ys = np.meshgrid(np.arange(5) * dx, np.arange(5) * dy)
    vertices = np.column_stack((xs.ravel(), ys.ravel(), np.zeros(25)))
    faces = []
    for y in range(4):
        for x in range(4):
            if (x, y) == (1, 1):
                continue
            a = y * 5 + x
            faces.extend(([a, a + 1, a + 5], [a + 1, a + 6, a + 5]))
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                     o3d.utility.Vector3iVector(np.asarray(faces)))


def test_auto_accepts_rotated_50_cm_square_but_rejects_51_cm():
    mesh = sheet_with_gap(0.5, 0.5)
    mesh.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, np.pi / 4]), center=[0, 0, 0])
    filled, records = fill_mesh(mesh, [], auto=True)
    assert len(boundary_loops(np.asarray(mesh.triangles))) == 2
    assert len(boundary_loops(np.asarray(filled.triangles))) == 1
    assert sum(r["filled_region"] == "auto" for r in records) == 1
    assert any(np.allclose(r.get("planar_extent_m"), [0.5, 0.5]) for r in records if r["filled_region"] == "auto")

    mesh.scale(1.02, center=[0, 0, 0])
    oversized, records = fill_mesh(mesh, [], auto=True)
    assert len(oversized.triangles) == len(mesh.triangles)
    assert all(r["filled_region"] is None for r in records)


def test_auto_fills_minor_hole_despite_noisy_corner_normals():
    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[2:])
    mesh.scale(0.04, center=[0, 0, 0])
    filled, records = fill_mesh(mesh, [], auto=True)
    assert len(filled.triangles) > len(mesh.triangles)
    assert records[0]["filled_region"] == "auto"


def test_auto_fills_long_narrow_gap_but_not_large_opening():
    mesh = sheet_with_gap(1.6, 0.29)
    filled, records = fill_mesh(mesh, [], auto=True)
    assert sum(r["filled_region"] == "auto" for r in records) == 1
    assert len(boundary_loops(np.asarray(filled.triangles))) == 1

    mesh = sheet_with_gap(1.6, 0.31)
    unchanged, records = fill_mesh(mesh, [], auto=True)
    assert len(unchanged.triangles) == len(mesh.triangles)
    assert all(r["filled_region"] is None for r in records)

    mesh = sheet_with_gap(2.1, 0.1)
    unchanged, records = fill_mesh(mesh, [], auto=True)
    assert len(unchanged.triangles) == len(mesh.triangles)


def test_auto_avoids_existing_edge_across_hole():
    mesh = sheet_with_gap(0.2, 0.2)
    vertices = np.asarray(mesh.vertices).copy()
    hole = min(boundary_loops(np.asarray(mesh.triangles)), key=len)
    keys, counts = np.unique(edge_keys(patch_loop(hole, vertices)), return_counts=True)
    diagonal = int(keys[counts == 2][0])
    a, b = diagonal >> 32, diagonal & 0xffffffff
    center = (vertices[a] + vertices[b]) / 2
    mesh.vertices = o3d.utility.Vector3dVector(np.vstack((vertices, center + [0, 0, 0.1], center - [0, 0, 0.1])))
    u, w = len(vertices), len(vertices) + 1
    mesh.triangles = o3d.utility.Vector3iVector(np.vstack((np.asarray(mesh.triangles),
        [[a, b, u], [b, a, w], [a, u, w], [b, w, u]])))
    filled, records = fill_mesh(mesh, [], auto=True)
    assert any(r.get("patch_method") == "split_edges" for r in records)
    _, edge_counts = np.unique(edge_keys(np.asarray(filled.triangles)), return_counts=True)
    assert edge_counts.max() == 2


def test_concave_loop_is_triangulated_without_crossing_void():
    from shapely.geometry import Polygon

    points = np.asarray([(0, 0, 0), (3, 0, 0), (3, 3, 0), (2, 3, 0),
                         (2, 1, 0), (1, 1, 0), (1, 3, 0), (0, 3, 0)], dtype=float)
    faces = patch_loop(np.arange(len(points)), points)
    polygon = Polygon(points[:, :2])
    assert len(faces) == len(points) - 2
    assert all(polygon.covers(Polygon(points[face, :2])) for face in faces)


def test_boundary_loops_split_vertex_junction():
    vertices = np.asarray([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                           (-1, 0, 0), (-1, -1, 0), (0, -1, 0)], dtype=float)
    triangles = np.asarray([[0, 1, 2], [0, 2, 3], [0, 4, 5], [0, 5, 6]])
    loops = boundary_loops(triangles, vertices)
    assert sorted(map(len, loops)) == [4, 4]
    assert sum(0 in loop for loop in loops) == 2


def test_auto_closes_jagged_self_crossing_rim_with_fan():
    rim = np.asarray([(0, 0, 0), (0.2, 0, 0), (0.2, 0.2, 0),
                      (0, 0.2, 0), (0.1, -0.05, 0)])
    vertices = np.vstack((rim, rim - [0, 0, 0.1]))
    triangles = []
    for i in range(len(rim)):
        j = (i + 1) % len(rim)
        triangles.extend(([i, j, j + len(rim)], [i, j + len(rim), i + len(rim)]))
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                     o3d.utility.Vector3iVector(np.asarray(triangles)))
    filled, records = fill_mesh(mesh, [], auto=True)
    assert any(r.get("patch_method") == "fan_fallback" for r in records)
    _, counts = np.unique(edge_keys(np.asarray(filled.triangles)), return_counts=True)
    assert np.all(counts == 2)


def test_seal_all_closes_large_and_tiny_openings():
    large = o3d.geometry.TriangleMesh.create_box(width=3, height=3, depth=3)
    tiny = o3d.geometry.TriangleMesh.create_box(width=1e-5, height=1e-5, depth=1e-5)
    tiny.translate([5, 0, 0])
    mesh = large + tiny
    faces = np.asarray(mesh.triangles)
    mesh.triangles = o3d.utility.Vector3iVector(np.delete(faces, [2, 3, 14, 15], axis=0))
    filled, records = fill_mesh(mesh, [], seal_all=True)
    assert len(records) == 2
    assert all(r["filled_region"] == "seal_all" for r in records)
    _, counts = np.unique(edge_keys(np.asarray(filled.triangles)), return_counts=True)
    assert np.all(counts == 2)


def test_seal_all_repairs_zero_length_boundary_edge():
    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[2:])
    loop = boundary_loops(np.asarray(mesh.triangles))[0]
    vertices = np.asarray(mesh.vertices)
    vertices[loop[1]] = vertices[loop[0]]
    filled, records = fill_mesh(mesh, [], seal_all=True)
    assert records[0]["filled_region"] == "seal_all"
    assert not boundary_loops(np.asarray(filled.triangles))
    assert np.linalg.norm(np.asarray(filled.vertices)[loop[1]] - np.asarray(filled.vertices)[loop[0]]) > 0


def test_coplanar_texture_sampler_logic():
    # Wall in X-Z plane (Y=0), normal [0, 1, 0], pure white [1.0, 1.0, 1.0]
    wall_v = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]], dtype=float)
    wall_t = np.array([[0, 2, 1], [1, 2, 3]])
    wall_c = np.ones((len(wall_v), 3))

    # Floor in X-Y plane (Z=0), normal [0, 0, 1], dark brown [0.2, 0.1, 0.05]
    floor_v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    floor_t = np.array([[0, 1, 2], [1, 3, 2]])
    floor_c = np.full((len(floor_v), 3), [0.2, 0.1, 0.05])

    all_v = np.vstack([wall_v, floor_v + [0, 0, 0]])
    all_t = np.vstack([wall_t, floor_t + len(wall_v)])
    all_c = np.vstack([wall_c, floor_c])

    sampler = CoplanarTextureSampler(all_v, all_t, all_c)

    # Query 1: point on wall plane (Y=0), normal [0, 1, 0]
    wall_sample = sampler.query([0.5, 0.0, 0.5], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(wall_sample, [1.0, 1.0, 1.0], atol=1e-5)

    # Query 2: point on floor plane (Z=0), normal [0, 0, 1]
    floor_sample = sampler.query([0.5, 0.5, 0.0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(floor_sample, [0.2, 0.1, 0.05], atol=1e-5)


def test_coplanar_texture_assignment_avoids_cross_plane_bleeding():
    # 5x5 grid in X-Z (vertical wall, Y=0) with hole at (x=1, z=1)
    dx = 0.1
    xs, zs = np.meshgrid(np.arange(5) * dx, np.arange(5) * dx)
    wall_v = np.column_stack((xs.ravel(), np.zeros(25), zs.ravel()))
    wall_faces = []
    for z in range(4):
        for x in range(4):
            if (x, z) == (1, 1):
                continue
            a = z * 5 + x
            wall_faces.extend([[a, a + 5, a + 1], [a + 1, a + 5, a + 6]])

    # 5x5 grid in X-Y (horizontal floor, Z=0) with hole at (x=1, y=1)
    xs_f, ys_f = np.meshgrid(np.arange(5) * dx, np.arange(1, 5) * dx)
    floor_v = np.column_stack((xs_f.ravel(), ys_f.ravel(), np.zeros(20)))
    floor_faces = []
    for y in range(4):
        for x in range(4):
            if (x, y) == (1, 1):
                continue
            a = x if y == 0 else 25 + (y - 1) * 5 + x
            b = (x + 1) if y == 0 else 25 + (y - 1) * 5 + x + 1
            c = 25 + y * 5 + x
            d = 25 + y * 5 + x + 1
            floor_faces.extend([[a, b, c], [b, d, c]])

    all_v = np.vstack([wall_v, floor_v])
    all_t = np.vstack([wall_faces, floor_faces])

    colors = np.zeros((len(all_v), 3))
    colors[:25] = [1.0, 1.0, 1.0]      # White wall
    colors[25:] = [0.2, 0.1, 0.05]    # Dark brown floor

    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(all_v), o3d.utility.Vector3iVector(all_t))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    filled, records = fill_mesh(mesh, [], auto=True)
    assert sum(r["filled_region"] == "auto" for r in records) == 2

    # Derived infilled triangle textures
    infill_face_colors = derive_infilled_face_textures(filled, len(mesh.triangles))
    assert infill_face_colors is not None
    assert len(infill_face_colors) == len(filled.triangles) - len(mesh.triangles)

    # First 6 infilled triangles correspond to wall hole
    wall_infill_colors = infill_face_colors[:6]
    # Remaining 6 infilled triangles correspond to floor hole
    floor_infill_colors = infill_face_colors[6:]

    # Wall triangles must strictly be pure white (no floor brown contamination)
    np.testing.assert_allclose(wall_infill_colors, np.ones_like(wall_infill_colors), atol=1e-3)

    # Floor triangles must not have white contamination (mean RGB must be well below white)
    assert np.all(floor_infill_colors[:, 0] < 0.4)
    assert np.all(floor_infill_colors[:, 1] < 0.3)
    assert np.all(floor_infill_colors[:, 2] < 0.25)

