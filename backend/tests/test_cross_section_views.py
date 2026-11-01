import json
import sys
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1] / "08_artifacts_generation"
sys.path.insert(0, str(STAGE))

import numpy as np
import open3d as o3d
import pytest
import trimesh

from extract_cross_section_views import (
    classify_outer_shell_faces,
    extract_cross_sections,
    slice_open3d_mesh_plane,
)


def test_slice_mesh_half_space():
    box = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=2.0)
    box.translate([-1.0, -1.0, -1.0])
    normal = np.array([0.0, -1.0, 0.0])
    origin = np.array([0.0, 0.0, 0.0])
    sliced = slice_open3d_mesh_plane(box, normal, origin)
    assert len(sliced.triangles) > 0
    assert len(sliced.triangles) < len(box.triangles)
    sliced_verts = np.asarray(sliced.vertices)
    sliced_faces = np.asarray(sliced.triangles)
    centers_y = sliced_verts[sliced_faces].mean(axis=1)[:, 1]
    assert np.all(centers_y <= 1e-4)


def test_enclosed_disconnected_body_is_not_an_outer_shell():
    outer = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    floater = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    combined = trimesh.util.concatenate([outer, floater])
    trimesh.repair.fix_normals(combined, multibody=True)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(combined.vertices),
        o3d.utility.Vector3iVector(combined.faces),
    )

    outer_faces = classify_outer_shell_faces(mesh)
    assert outer_faces.sum() == len(outer.faces)
    assert not outer_faces[len(outer.faces):].any()


def test_extract_cross_sections_generates_all_views(tmp_path):
    # Create simple box room mesh
    room = trimesh.creation.box(extents=(4.0, 3.0, 4.0))
    mesh_path = tmp_path / "scene_mesh.ply"
    room.export(str(mesh_path))

    # Create dummy plane_patterns_manifest.json with two floors and roof
    manifest_data = {
        "units": [
            {"structural_role": "floor", "mean_elevation_y": -1.5},
            {"structural_role": "floor", "mean_elevation_y": 0.0},
            {"structural_role": "ceiling", "mean_elevation_y": 1.5},
        ]
    }
    instances_dir = tmp_path / "instances"
    instances_dir.mkdir()
    (instances_dir / "plane_patterns_manifest.json").write_text(json.dumps(manifest_data))

    output_dir = tmp_path / "cross_sections"
    result = extract_cross_sections(
        mesh_path=mesh_path,
        instances_dir=instances_dir,
        output_dir=output_dir,
        resolution=256,
    )

    assert (output_dir / "cross_sections_manifest.json").is_file()
    assert len(result["views"]) == 6
    view_names = [v["view_name"] for v in result["views"]]
    assert "topdown_floor_0" in view_names
    assert "topdown_floor_1" in view_names
    assert "elevation_west_to_east_x_pos" in view_names

    # Check that floor 1 has a bottom slice plane defined to isolate it from floor 0
    floor_1_view = next(v for v in result["views"] if v["view_name"] == "topdown_floor_1")
    assert "bottom_slice_plane" in floor_1_view
    assert floor_1_view["bottom_slice_plane"]["normal"] == [0.0, 1.0, 0.0]

    for view in result["views"]:
        img_path = Path(view["image_file"])
        assert img_path.is_file()
        assert img_path.stat().st_size > 0


def test_second_floor_cut_maps_pixels_to_original_mesh_faces(tmp_path):
    room = trimesh.creation.box(extents=(4.0, 3.0, 4.0))
    inner_surface = trimesh.Trimesh(
        vertices=[[-1.5, 0.5, -1.5], [1.5, 0.5, -1.5],
                  [1.5, 0.5, 1.5], [-1.5, 0.5, 1.5]],
        faces=[[0, 2, 1], [0, 3, 2]], process=False,
    )
    mesh = trimesh.util.concatenate([room, inner_surface])
    mesh_path = tmp_path / "scene_mesh.ply"
    mesh.export(str(mesh_path))

    instances_dir = tmp_path / "instances"
    instances_dir.mkdir()
    (instances_dir / "plane_patterns_manifest.json").write_text(json.dumps({
        "units": [
            {"structural_role": "floor", "mean_elevation_y": -1.5},
            {"structural_role": "floor", "mean_elevation_y": 0.0},
            {"structural_role": "ceiling", "mean_elevation_y": 1.5},
        ]
    }))
    output_dir = tmp_path / "cross_sections"
    map_dir = tmp_path / "projection_maps"
    extract_cross_sections(
        mesh_path, instances_dir, output_dir, resolution=64,
        projection_map_dir=map_dir,
    )

    source = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(source.vertices)
    faces = np.asarray(source.triangles)
    with np.load(map_dir / "topdown_floor_1.npz") as projection:
        face_ids = projection["triangle_ids"]
        face_ids = face_ids[face_ids >= 0]
    assert len(face_ids) > 0
    np.testing.assert_array_equal(
        np.unique(face_ids), np.arange(len(room.faces), len(room.faces) + len(inner_surface.faces))
    )
    centers_y = vertices[faces[face_ids]].mean(axis=1)[:, 1]
    assert centers_y.min() >= -0.1 - 1e-6
    assert centers_y.max() <= 1.2 + 1e-6


def test_all_outward_facing_hits_are_transparent_until_an_inward_hit():
    from extract_cross_section_views import render_cut_view

    def quad(x, front_facing=False):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector([
            [x, -1, -1], [x, 1, -1], [x, 1, 1], [x, -1, 1],
        ])
        triangles = [[0, 1, 2], [0, 2, 3]]
        if front_facing:
            triangles = [[0, 2, 1], [0, 3, 2]]
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        return mesh

    camera = (np.array([-4.0, 0.0, 0.0]), np.zeros(3), np.array([0.0, 1.0, 0.0]))

    layered = o3d.geometry.TriangleMesh()
    outside = quad(0.0, front_facing=True)
    outside.vertex_colors = o3d.utility.Vector3dVector(np.tile([0.0, 1.0, 0.0], (4, 1)))
    middle_outside = quad(0.5, front_facing=True)
    middle_outside.vertex_colors = o3d.utility.Vector3dVector(np.tile([1.0, 1.0, 0.0], (4, 1)))
    inside = quad(1.0)
    inside.vertex_colors = o3d.utility.Vector3dVector(np.tile([0.0, 0.0, 1.0], (4, 1)))
    layered += outside
    layered += middle_outside
    layered += inside
    image = render_cut_view(layered, *camera, 64, 60.0)
    assert tuple(image[32, 32]) == (0, 0, 255)
    mapped_image, projection = render_cut_view(layered, *camera, 64, 60.0, return_projection=True)
    assert np.array_equal(mapped_image, image)
    assert projection["triangle_ids"][32, 32] >= 4
    assert np.isclose(projection["barycentric"][32, 32].sum(), 1.0)
    assert projection["view_cosine"][32, 32] > 0.99

    occluded = o3d.geometry.TriangleMesh()
    front = quad(0.0)
    front.vertex_colors = o3d.utility.Vector3dVector(np.tile([0.0, 1.0, 0.0], (4, 1)))
    farther_outside = quad(1.0, front_facing=True)
    farther_outside.vertex_colors = o3d.utility.Vector3dVector(np.tile([0.0, 0.0, 1.0], (4, 1)))
    occluded += front
    occluded += farther_outside
    image = render_cut_view(occluded, *camera, 64, 60.0)
    assert tuple(image[32, 32]) == (0, 255, 0)


def test_tilted_outer_hit_passes_to_inward_side_behind_it():
    from extract_cross_section_views import render_cut_view

    inward = np.array([0.366, -0.931, 0.0])
    inward /= np.linalg.norm(inward)
    outward = -inward
    u = np.cross([0.0, 0.0, 1.0], outward)
    u /= np.linalg.norm(u)
    v = np.cross(outward, u)
    exterior_offset = -0.05 * inward
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array([
        -2 * u - 2 * v, 2 * u - 2 * v, 2 * u + 2 * v, -2 * u + 2 * v,
    ]) + exterior_offset)
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])

    mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile([0.0, 1.0, 0.0], (4, 1)))
    inner_layer = o3d.geometry.TriangleMesh(mesh)
    inner_layer.translate(0.1 * inward, relative=True)
    inner_layer.triangles = o3d.utility.Vector3iVector([[0, 2, 1], [0, 3, 2]])
    inner_layer.vertex_colors = o3d.utility.Vector3dVector(np.tile([0.0, 0.0, 1.0], (4, 1)))
    image = render_cut_view(
        mesh + inner_layer, np.array([-4.0, 0.0, 0.0]), np.zeros(3),
        np.array([0.0, 1.0, 0.0]), 64,
    )
    assert tuple(image[32, 32]) == (0, 0, 255)



