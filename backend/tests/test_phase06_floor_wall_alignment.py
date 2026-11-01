from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import shapely
import trimesh
from shapely.geometry import Polygon

STAGE_DIR = Path(__file__).parents[1] / "06_mesh_elaboration"
sys.path.insert(0, str(STAGE_DIR))
from assemble_final_regularized_mesh import (
    assemble_final_regularized_mesh,
    _ceiling_patch_max_y,
    _clip_wall_to_floor_and_ceiling,
    _extend_wall_top_to_ceiling,
    _clip_surface_to_footprint,
    _wall_face_floor_planes,
    _wall_footprint_at_y,
    _wall_support_over_y_range,
)
from fill_mesh_holes import boundary_loops, export_ply_with_face_attributes, read_ply_face_flags
from structural_geometry import close_slab


def test_upper_ceiling_selection_uses_highest_y_on_tilted_patches() -> None:
    vertices = np.array([
        [-1., 1.2, 0.], [1., .8, 0.], [0., 1., 1.],
        [-1., 1.1, 0.], [1., 1.1, 0.], [0., 1.1, 1.],
        [-1., 2., 0.], [1., 2., 0.], [0., 2., 1.],
        [-1., 3., 0.], [1., 3., 0.], [0., 3., 1.],
    ])
    faces = np.arange(12).reshape(-1, 3)
    ceiling = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    groups = np.array([10, 20, 10, 10])
    associations = np.array([-1, -1, 0, -1])
    areas = np.full(4, -1)
    junctions = np.array([0, 0, 0, 1])

    tilted_patch_y = _ceiling_patch_max_y(
        ceiling, groups, associations, areas, junctions,
        {"group_id": 10, "support_area_id": None},
    )
    flatter_patch_y = _ceiling_patch_max_y(
        ceiling, groups, associations, areas, junctions,
        {"group_id": 20, "support_area_id": None},
    )

    assert vertices[:3, 1].mean() < vertices[3:6, 1].mean()
    assert tilted_patch_y == pytest.approx(1.2)
    assert tilted_patch_y > flatter_patch_y
    assert max((tilted_patch_y, 10), (flatter_patch_y, 20))[1] == 10
    associated_patch_y = _ceiling_patch_max_y(
        ceiling, groups, associations, areas, junctions,
        {"group_id": 10, "support_area_id": None, "associated_floor_level": 0},
    )
    assert associated_patch_y == pytest.approx(2.0)


def test_wall_bottom_uses_local_lowest_floor_planes() -> None:
    wall = trimesh.Trimesh(
        vertices=[[-2., -.5, 0.], [-1., 1., 0.], [-1., 1., 1.],
                  [1., -.5, 0.], [2., 1., 0.], [1., 1., 1.]],
        faces=[[0, 1, 2], [3, 4, 5]],
        process=False,
    )
    supports = [
        {"area_id": 0, "polygon": Polygon([(-3., -1.), (0., -1.), (0., 2.), (-3., 2.)]),
         "equation": np.array([0., 1., 0., 0.])},
        {"area_id": 1, "polygon": Polygon([(0., -1.), (3., -1.), (3., 2.), (0., 2.)]),
         "equation": np.array([0., 1., 0., -.5])},
    ]

    lower_planes, selection = _wall_face_floor_planes(wall, supports, -1.0)
    clipped, _ = _clip_wall_to_floor_and_ceiling(wall, lower_planes, [0., 1., 0., -2.])

    assert selection["area_ids_used"] == [0, 1]
    assert selection["wall_face_count_using_fitted_floor_area"] == 2
    assert clipped.vertices[clipped.vertices[:, 0] < 0, 1].min() == pytest.approx(0.)
    assert clipped.vertices[clipped.vertices[:, 0] > 0, 1].min() == pytest.approx(.5)


def test_wall_face_crossing_floor_area_boundary_uses_lowest_y_fallback() -> None:
    wall = trimesh.Trimesh(
        vertices=[[-1., -.5, 0.], [1., .5, 0.], [0., 1., 1.]],
        faces=[[0, 1, 2]],
        process=False,
    )
    supports = [
        {"area_id": 0, "polygon": Polygon([(-3., -1.), (0., -1.), (0., 2.), (-3., 2.)]),
         "equation": np.array([0., 1., 0., 0.])},
        {"area_id": 1, "polygon": Polygon([(0., -1.), (3., -1.), (3., 2.), (0., 2.)]),
         "equation": np.array([0., 1., 0., -.5])},
    ]

    lower_planes, selection = _wall_face_floor_planes(wall, supports, -1.0)

    np.testing.assert_array_equal(lower_planes[0], [0., 1., 0., 1.])
    assert selection["wall_face_count_using_lowest_y_fallback"] == 1


def test_orphan_ceiling_clip_follows_wall_contour_and_preserves_hole() -> None:
    outer = [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)]
    support = Polygon(outer, holes=[[(-.25, -.25), (.25, -.25), (.25, .25), (-.25, .25)]])
    ring = np.array([[-1.5, -1.5], [1.5, -1.5], [1.5, 1.5], [-1.5, 1.5]])
    roof_vertices = np.column_stack((ring[:, 0], 2.0 + .1 * ring[:, 0], ring[:, 1]))
    roof = trimesh.Trimesh(vertices=roof_vertices, faces=[[0, 1, 2], [0, 2, 3]], process=False)
    roof.visual.vertex_colors = np.tile([120, 90, 70, 255], (4, 1))

    clipped, stats = _clip_surface_to_footprint(
        roof, support, retain_degenerate_inside=True,
    )

    projected = shapely.polygons(np.concatenate((clipped.vertices[clipped.faces][:, :, [0, 2]],
                                                  clipped.vertices[clipped.faces][:, :1, [0, 2]]), axis=1))
    assert stats["removed_projected_area_m2"] > 0
    assert float(shapely.area(projected).sum()) == pytest.approx(support.area)
    assert np.all(shapely.covers(support, projected))
    assert np.allclose(clipped.vertices[:, 1], 2.0 + .1 * clipped.vertices[:, 0], atol=1e-6)


def test_floor_is_lowered_clipped_to_wall_footprint_and_keeps_colors() -> None:
    floor = trimesh.Trimesh(
        vertices=[[-1, 0.25, 0], [2, 0.25, 0], [2, 0.25, 1], [-1, 0.25, 1]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    floor.visual.vertex_colors = np.array([
        [255, 0, 0, 255], [0, 255, 0, 255],
        [0, 0, 255, 255], [255, 255, 0, 255],
    ], dtype=np.uint8)
    footprint = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    aligned, stats = _clip_surface_to_footprint(floor, footprint, 0.0)

    projected = shapely.polygons(np.concatenate((aligned.vertices[aligned.faces][:, :, [0, 2]],
                                                  aligned.vertices[aligned.faces][:, :1, [0, 2]]), axis=1))
    assert np.isclose(stats["vertical_translation_m"], -0.25)
    assert np.allclose(aligned.vertices[:, 1], 0.0)
    assert np.all(shapely.covers(footprint.buffer(1e-8), projected))
    assert np.isclose(float(shapely.area(projected).sum()), 1.0)
    assert stats["removed_projected_area_m2"] > 1.99
    assert len(aligned.visual.vertex_colors) == len(aligned.vertices)
    assert np.all((aligned.visual.vertex_colors[:, :3] >= 0) &
                  (aligned.visual.vertex_colors[:, :3] <= 255))


def test_floor_clipping_drops_triangles_degenerate_at_ply_precision() -> None:
    floor = trimesh.Trimesh(
        vertices=[
            [0., 0., 0.], [1., 0., 0.], [0., 0., 1.],
            [.5, 0., .5], [1.5, 0., .5], [1.5, 0., .50000001],
        ],
        faces=[[0, 1, 2], [3, 4, 5]],
        process=False,
    )
    floor.visual.vertex_colors = np.tile([160, 140, 120, 255], (6, 1))

    clipped, stats = _clip_surface_to_footprint(
        floor, Polygon([(-1., -1.), (2., -1.), (2., 2.), (-1., 2.)]),
    )

    assert len(clipped.faces) == 1
    assert stats["discarded_float32_degenerate_triangle_count"] == 1


def test_wall_clipping_trims_both_elevations_and_interpolates_colors() -> None:
    wall = trimesh.Trimesh(
        vertices=[[0., -1., 0.], [1., 1., 0.], [0., 3., 0.],
                  [2., -2., 0.], [3., -1., 0.], [2., -1.5, 1.]],
        faces=[[0, 1, 2], [3, 4, 5]],
        process=False,
    )
    wall.visual.vertex_colors = np.array([
        [0, 0, 0, 255], [100, 100, 100, 255], [200, 200, 200, 255],
        [1, 2, 3, 255], [4, 5, 6, 255], [7, 8, 9, 255],
    ], dtype=np.uint8)

    clipped, stats = _clip_wall_to_floor_and_ceiling(
        wall, 0., np.array([0., 1., 0., -2.]),
    )

    assert clipped.vertices[:, 1].min() == pytest.approx(0.)
    assert clipped.vertices[:, 1].max() == pytest.approx(2.)
    assert len(clipped.faces) == 3
    assert stats["removed_source_triangle_count"] == 1
    assert stats["clipped_source_triangle_count"] == 1
    assert stats["generated_triangle_count"] == 3
    assert stats["applied"] is True
    assert any(np.array_equal(color[:3], [50, 50, 50]) for color in clipped.visual.vertex_colors)


def open_room_walls_with_window() -> trimesh.Trimesh:
    wall = stepped_walls()
    vertices = list(wall.vertices)
    faces = list(wall.faces)

    def add_panel(x0: float, x1: float, y0: float, y1: float) -> None:
        start = len(vertices)
        vertices.extend(([x0, y0, -1.], [x1, y0, -1.],
                         [x1, y1, -1.], [x0, y1, -1.]))
        faces.extend(([start, start + 1, start + 2], [start, start + 2, start + 3]))

    # This detached wall patch has an interior window loop in a vertical plane.
    add_panel(-.5, .5, .3, .4)
    add_panel(-.5, .5, .7, .8)
    add_panel(-.5, -.3, .4, .7)
    add_panel(.3, .5, .4, .7)
    result = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    result.visual.vertex_colors = np.tile([150, 145, 140, 255], (len(vertices), 1))
    return result


def test_wall_top_extension_follows_sloped_ceiling_and_keeps_window_open() -> None:
    roof = np.array([-.1, 1., 0., -2.])
    wall = open_room_walls_with_window()
    source_vertices = wall.vertices.copy()
    source_faces = wall.faces.copy()
    source_colors = wall.visual.vertex_colors.copy()

    extended, stats = _extend_wall_top_to_ceiling(wall, 0., roof)

    np.testing.assert_array_equal(extended.vertices[:len(source_vertices)], source_vertices)
    np.testing.assert_array_equal(extended.faces[:len(source_faces)], source_faces)
    np.testing.assert_array_equal(extended.visual.vertex_colors[:len(source_colors)], source_colors)
    assert stats["extended_boundary_loop_count"] == 1
    assert stats["extended_boundary_edge_count"] == 4
    assert stats["added_triangle_count"] == 8
    assert stats["maximum_extension_m"] == pytest.approx(1.1)
    new_faces = extended.faces[-stats["added_triangle_count"]:]
    new_triangles = extended.vertices[new_faces]
    plane = roof / np.linalg.norm(roof[:3])
    assert np.max(new_triangles @ plane[:3] + plane[3]) == pytest.approx(0., abs=1e-6)
    centroids = new_triangles.mean(axis=1)
    assert not np.any((np.abs(centroids[:, 2] + 1.) < 1e-6)
                      & (centroids[:, 0] > -.4) & (centroids[:, 0] < .4)
                      & (centroids[:, 1] > .35) & (centroids[:, 1] < .65))


def test_wall_top_extension_drops_triangles_degenerate_at_ply_precision() -> None:
    wall = stepped_walls()
    wall.vertices[8, 1] = 2. - 1e-8

    extended, stats = _extend_wall_top_to_ceiling(wall, 0., [0., 1., 0., -2.])

    triangles = extended.vertices[extended.faces].astype(np.float32)
    twice_area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                         triangles[:, 2] - triangles[:, 0]), axis=1)
    assert stats["added_triangle_count"] > 0
    assert np.all(twice_area > 0.)


def stepped_walls() -> trimesh.Trimesh:
    xz_rings = [
        np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]),
        np.array([[-.8, -.8], [.8, -.8], [.8, .8], [-.8, .8]]),
        np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]),
    ]
    heights = [0., .5, 1.]
    vertices = np.concatenate([
        np.column_stack((ring[:, 0], np.full(4, y), ring[:, 1]))
        for ring, y in zip(xz_rings, heights)
    ])
    faces = []
    for lower in range(2):
        for edge in range(4):
            following = (edge + 1) % 4
            a, b = lower * 4 + edge, lower * 4 + following
            c, d = (lower + 1) * 4 + following, (lower + 1) * 4 + edge
            faces.extend(([a, b, c], [a, c, d]))
    # A horizontal wall transition marks the narrow contour between wider ends.
    vertices = np.vstack((vertices, [[0., .5, 0.], [.05, .5, 0.], [0., .5, .05]]))
    faces.append([12, 13, 14])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def test_wall_support_tracks_each_floor_height_and_horizontal_transitions() -> None:
    walls = stepped_walls()
    upper = _wall_footprint_at_y(walls, .75)
    support = _wall_support_over_y_range(walls, .1, .9)

    assert upper.area == pytest.approx(4 * .9**2)
    assert support.area == pytest.approx(2.56)
    assert upper.buffer(1e-6).covers(support)


def test_paired_slab_is_clipped_to_wall_support_before_being_closed() -> None:
    walls = stepped_walls()
    support = _wall_support_over_y_range(walls, .6, .9)
    ceiling = trimesh.Trimesh(
        vertices=[[-1.2, .6, -1.2], [1.2, .6, -1.2],
                  [1.2, .6, 1.2], [-1.2, .6, 1.2]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    ceiling.visual.vertex_colors = np.tile([120, 90, 70, 255], (4, 1))

    clipped, stats = _clip_surface_to_footprint(ceiling, support)
    slab, _, _ = close_slab(clipped.vertices, clipped.faces, [0., 1., 0., -.9])

    projected = shapely.points(slab.vertices[:, [0, 2]])
    assert stats["removed_projected_area_m2"] > 0
    assert slab.is_watertight
    assert np.all(shapely.covers(support.buffer(1e-6), projected))


@pytest.mark.parametrize("roof_equation", [
    [0., 1., 0., -1.4], [0., 1., 0., -.9], [.1, 1., 0., -.9],
])
def test_final_assembly_trims_walls_and_orphan_roof_to_structural_envelope(tmp_path, roof_equation) -> None:
    ring = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    waist = ring * .7
    wall_vertices = np.concatenate((
        np.column_stack((ring[:, 0], np.zeros(4), ring[:, 1])),
        np.column_stack((waist[:, 0], np.full(4, .5), waist[:, 1])),
        np.column_stack((ring[:, 0], np.ones(4), ring[:, 1])),
    ))
    wall_faces = []
    for lower in range(2):
        for edge in range(4):
            following = (edge + 1) % 4
            a, b = lower * 4 + edge, lower * 4 + following
            c, d = (lower + 1) * 4 + following, (lower + 1) * 4 + edge
            wall_faces.extend(([a, b, c], [a, c, d]))
    floor_vertices = np.column_stack((ring[:, 0], np.zeros(4), ring[:, 1]))
    cap_faces = np.array([[0, 1, 2], [0, 2, 3]])
    original_wall = trimesh.Trimesh(vertices=wall_vertices, faces=wall_faces, process=False)
    original_wall.visual.vertex_colors = np.column_stack((
        np.arange(len(wall_vertices) * 3).reshape(-1, 3) * 5 + 40,
        np.full(len(wall_vertices), 255),
    )).astype(np.uint8)

    for ceiling_y in (.4, .75):
        scene = tmp_path / f"ceiling_{ceiling_y}"
        steps = scene / "mesh_elaboration_steps"
        levels = steps / "intermediate_floor_levels"
        levels.mkdir(parents=True)
        export_ply_with_face_attributes(
            steps / "wall_enclosing_planes_fitted_to_regularized_contours.ply",
            wall_vertices, np.asarray(wall_faces), original_wall.visual.vertex_colors,
        )
        roof_y = -(roof_equation[0] * ring[:, 0] + roof_equation[2] * ring[:, 1]
                   + roof_equation[3]) / roof_equation[1]
        orphan_ring = ring * 1.5
        orphan_y = -(roof_equation[0] * orphan_ring[:, 0] + roof_equation[2] * orphan_ring[:, 1]
                     + roof_equation[3]) / roof_equation[1]
        ceiling_vertices = np.concatenate((
            np.column_stack((ring[:, 0], np.full(4, ceiling_y), ring[:, 1])),
            np.column_stack((orphan_ring[:, 0], orphan_y, orphan_ring[:, 1])),
        ))
        ceiling_faces = np.vstack((cap_faces, cap_faces + 4))
        ceiling_colors = np.column_stack((
            np.arange(24).reshape(8, 3) * 7 + 30, np.full(8, 255),
        )).astype(np.uint8)
        export_ply_with_face_attributes(
            steps / "ceiling_planes_fitted_to_support_contours_regularized.ply",
            ceiling_vertices, ceiling_faces, ceiling_colors,
            {"associated_floor_level": ("int", np.array([0, 0, -1, -1], dtype=np.int32)),
             "ceiling_group_id": ("int", np.array([0, 0, 1, 1], dtype=np.int32)),
             "support_area_id": ("int", np.array([0, 0, -1, -1], dtype=np.int32))},
        )
        export_ply_with_face_attributes(
            levels / "floor_level_000_patterns_regularized.ply",
            floor_vertices, cap_faces, np.tile([100, 80, 60, 255], (4, 1)),
            {"floor_area_id": ("int", np.zeros(2, dtype=np.int32))},
        )
        (steps / "ceiling_planes_fitted_to_support_contours_regularized_manifest.json").write_text(
            json.dumps({"output_planes": [
                {"group_id": 0, "associated_floor_level": 0, "support_area_id": 0,
                 "plane_equation": [0., 1., 0., -ceiling_y]},
                {"group_id": 1, "associated_floor_level": None, "support_area_id": None,
                 "plane_equation": roof_equation, "wall_top_constraint_applied": True},
            ]}),
            encoding="utf-8",
        )
        (levels / "floor_level_intermediate_manifest.json").write_text(json.dumps({
            "levels": [{"level": 0, "elevation_y_median": 0., "floor_areas": [
                {"area_id": 0, "plane_equation": [0., 1., 0., 0.]},
            ]}],
        }), encoding="utf-8")
        (levels / "floor_level_000_polygons.json").write_text(json.dumps({"areas": [{
            "area_id": 0, "regularized_wall_contour_world_xz_m": ring.tolist(),
        }]}), encoding="utf-8")

        elements = scene / "elements"
        result = assemble_final_regularized_mesh(scene, elements_dir=elements)
        final_walls = trimesh.load(elements / "walls.ply", force="mesh", process=False)
        combined = trimesh.load(result["output_ply"], force="mesh", process=False)
        roles = read_ply_face_flags(result["output_ply"], "element_role")
        source_triangles = original_wall.vertices[original_wall.faces]
        final_triangles = final_walls.vertices[final_walls.faces]
        wall_clip = result["wall_ceiling_alignment"]["wall_clipping"]
        wall_gap_fill = result["wall_ceiling_alignment"]["wall_gap_fill"]
        ceiling_source = result["wall_ceiling_alignment"]["wall_ceiling_plane"]
        assert ceiling_source["ceiling_group_id"] == 1
        assert ceiling_source["highest_surface_y_m"] == pytest.approx(float(orphan_y.max()))
        assert wall_clip["lowest_floor_level"] == 0
        assert wall_clip["lowest_floor_surface_y_m"] == pytest.approx(0.)
        roof_plane = np.asarray(roof_equation, dtype=float)
        roof_plane /= np.linalg.norm(roof_plane[:3])
        assert final_triangles[:, :, 1].min() == pytest.approx(wall_clip["lowest_floor_y_m"])
        signed_ceiling_distance = final_triangles @ roof_plane[:3] + roof_plane[3]
        assert np.max(signed_ceiling_distance) <= 1e-6
        if min(roof_y) < 1. or wall_gap_fill["added_triangle_count"]:
            assert np.max(signed_ceiling_distance) == pytest.approx(0., abs=1e-6)
        else:
            assert np.max(signed_ceiling_distance) < 0.
        if min(roof_y) >= 1. and wall_clip["lowest_floor_y_m"] <= 0.:
            np.testing.assert_array_equal(final_triangles[:len(source_triangles)],
                                          source_triangles.astype(np.float32))
            np.testing.assert_array_equal(final_walls.visual.vertex_colors[final_walls.faces][:len(source_triangles)],
                                          original_wall.visual.vertex_colors[original_wall.faces])
        final_roof = trimesh.load(elements / "ceiling_group_001.ply", force="mesh", process=False)
        roof_projection = shapely.polygons(np.concatenate((final_roof.triangles[:, :, [0, 2]],
                                                            final_roof.triangles[:, :1, [0, 2]]), axis=1))
        wall_footprint = Polygon(ring)
        assert np.all(shapely.covers(wall_footprint, roof_projection))
        assert float(shapely.area(roof_projection).sum()) == pytest.approx(wall_footprint.area)
        assert np.allclose(final_roof.vertices[:, 1],
                           -(roof_equation[0] * final_roof.vertices[:, 0]
                             + roof_equation[2] * final_roof.vertices[:, 2]
                             + roof_equation[3]) / roof_equation[1], atol=1e-6)
        assert len(final_roof.visual.vertex_colors) == len(final_roof.vertices)
        np.testing.assert_array_equal(combined.vertices[combined.faces[roles == 2]], final_roof.triangles)
        slab_element = trimesh.load(elements / "slab_level_000_area_000.ply", force="mesh", process=False)
        np.testing.assert_array_equal(combined.vertices[combined.faces[roles == 3]], slab_element.triangles)
        assert result["counts"]["boundary_fill_faces"] == 0
        assert result["slabs"][0]["is_watertight"]
        assert wall_clip["applied"] is (bool(np.min(roof_y) < 1.0))
