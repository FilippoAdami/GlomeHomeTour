from pathlib import Path
import sys

import numpy as np
import pytest
import trimesh
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parents[1] / "06_mesh_elaboration"))
from planar_pattern_regularization import (
    _plane_frame,
    _fit_relief_line,
    _subdivide,
    _sample_visual,
    _straight_color_split,
    _surface_color_regions_are_connected,
    detect_visual_texture,
    _visual_model,
    regularize_planar_patches,
)
import assemble_wall_region_planes


def test_empty_support_band_is_ignored_before_span_reduction():
    assert _fit_relief_line(np.empty(0), np.empty(0), np.empty(0)) is None


def test_pattern_frame_follows_longest_mesh_boundary_edge():
    mesh = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [2, 0, 0], [2, 0, 1], [0, 0, 1]], dtype=float),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32), process=False,
    )
    normal, axis_u, axis_v, _ = _plane_frame([0, 1, 0, 0], mesh)

    np.testing.assert_allclose(normal, [0, 1, 0])
    assert abs(float(axis_u @ [1, 0, 0])) > 0.999
    assert abs(float(axis_v @ [0, 0, 1])) > 0.999


def _patch(name, x0, x1, support_centers, colors):
    mesh = trimesh.Trimesh(
        vertices=np.array([[x0, 0, 0], [x1, 0, 0], [x1, 1, 0], [x0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        process=False,
    )
    count = len(support_centers)
    return {
        "name": name,
        "mesh": mesh,
        "plane_equation": [0, 0, 1, 0],
        "support_centers": np.asarray(support_centers, dtype=float),
        "support_areas": np.ones(count),
        "support_colors": np.asarray(colors, dtype=float),
    }


def test_touching_compatible_planes_share_color_and_splits_are_preserved():
    points_a = np.array([[x, y, 0] for x in np.linspace(0.05, 0.95, 8)
                         for y in np.linspace(0.05, 0.95, 8)])
    points_b = points_a.copy()
    points_b[:, 0] += 1.0
    split_points = np.array([[x + 2.0, y, 0] for x in np.linspace(0.05, 0.95, 10)
                             for y in np.linspace(0.05, 0.95, 10)])
    split_colors = np.where(split_points[:, 0:1] < 2.5, 0.25, 0.75)
    patches = [
        _patch("plain_a", 0, 1, points_a, np.tile([0.45, 0.40, 0.35], (len(points_a), 1))),
        _patch("plain_b", 1, 2, points_b, np.tile([0.50, 0.45, 0.40], (len(points_b), 1))),
        _patch("split", 2, 3, split_points, np.repeat(split_colors, 3, axis=1)),
    ]

    result = regularize_planar_patches(patches, max_edge=0.5)

    np.testing.assert_allclose(result[0]["mesh"].visual.face_colors[0, :3],
                               result[1]["mesh"].visual.face_colors[0, :3], atol=1)
    assert result[0]["regularization"]["cross_plane_group"] == ["plain_a", "plain_b"]
    assert result[2]["color_split"]["is_split"]
    assert result[2]["regularization"]["cross_plane_group"] == ["split"]


def test_repeated_relief_uses_low_poly_prisms_and_keeps_the_panel_surface():
    points = np.array([[x, y, 0.025 * np.sin(2 * np.pi * x / 0.25)]
                       for x in np.linspace(0.01, 0.99, 50)
                       for y in np.linspace(0.01, 0.99, 10)])
    patch = _patch("ribbed", 0, 1, points, np.tile([0.5, 0.5, 0.5], (len(points), 1)))
    base = _subdivide(patch["mesh"], 0.5)
    result = regularize_planar_patches([patch], max_edge=0.5)

    assert result[0]["pattern_classification"]["pattern_type"] == "periodic_relief"
    geometry_pattern = result[0]["pattern_classification"]["parameters"]
    assert geometry_pattern["primitive_shape"] == "parallelepiped"
    assert geometry_pattern["primitive_count"] > 0
    assert geometry_pattern["period_scene_units"] == pytest.approx(0.25, abs=0.04)
    mesh = result[0]["mesh"]
    assert len(mesh.faces) == len(base.faces) + 12 * geometry_pattern["primitive_count"]
    np.testing.assert_allclose(mesh.vertices[:len(base.vertices), 2], 0.0, atol=1e-10)
    assert geometry_pattern["amplitude_scene_units"] <= min(
        0.05, geometry_pattern["period_scene_units"] * 0.08
    )
    assert result[0]["regularization"]["geometry"] == "repeated_parallelepipeds_in_observed_region"


def test_periodic_relief_stays_inside_its_observed_support_band():
    x, y = np.meshgrid(np.linspace(0.01, 0.99, 60), np.linspace(0.01, 0.99, 40))
    z = np.where(y < 0.55, 0.025 * np.sin(2 * np.pi * x / 0.25), 0.0)
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    patch = _patch("partial_relief", 0, 1, points, np.full((len(points), 3), 0.5))
    patch["relief_direction"] = [0.0, 0.0, -1.0]

    result = regularize_planar_patches([patch], max_edge=0.5)[0]
    vertices = result["mesh"].vertices

    assert result["pattern_classification"]["pattern_type"] == "periodic_relief"
    assert np.min(vertices[(vertices[:, 1] < 0.5), 2]) < -0.01
    assert np.allclose(vertices[(vertices[:, 1] > 0.65), 2], 0.0)
    assert np.max(vertices[:, 2]) <= 0.005
    np.testing.assert_allclose(
        result["pattern_classification"]["parameters"]["pattern_region_uv_bounds"],
        [0.0, 1.0, 0.0, 0.55], atol=0.03,
    )
    assert result["pattern_classification"]["parameters"]["primitive_shape"] == "parallelepiped"


def test_relief_primitives_use_each_patch_plane_offset(monkeypatch):
    def repeated_pattern(*args, **kwargs):
        return {
            "pattern_type": "periodic_relief",
            "confidence": 1.0,
            "parameters": {
                "axis": "u",
                "period_scene_units": 0.5,
                "pattern_region_uv_bounds": [-1.0, 1.0, -1.0, 1.0],
                "amplitude_scene_units": 0.02,
                "primitive_width_fraction": 0.4,
                "regularized_peak_positions": [-0.5, 0.0, 0.5],
            },
        }

    monkeypatch.setattr("planar_pattern_regularization._periodic_relief", repeated_pattern)
    xz = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    support_xz = np.array([[x, z] for x in np.linspace(0.05, 0.95, 8)
                           for z in np.linspace(0.05, 0.95, 8)])
    patches = []
    elevations = [-1.0, 1.25, 5.0]
    for index, elevation in enumerate(elevations):
        mesh = trimesh.Trimesh(
            vertices=np.column_stack((xz[:, 0], np.full(len(xz), elevation), xz[:, 1])),
            faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32), process=False,
        )
        centers = np.column_stack((support_xz[:, 0],
                                   np.full(len(support_xz), elevation), support_xz[:, 1]))
        patches.append({
            "name": f"ceiling_{index}",
            "mesh": mesh,
            "plane_equation": [0.0, 1.0, 0.0, -elevation],
            "relief_direction": [0.0, -1.0, 0.0],
            "support_centers": centers,
            "support_areas": np.ones(len(centers)),
            "support_colors": np.full((len(centers), 3), 0.5),
        })

    result = regularize_planar_patches(patches, max_edge=0.5)

    for patch, elevation in zip(result, elevations):
        assert patch["pattern_classification"]["pattern_type"] == "periodic_relief"
        assert patch["mesh"].vertices[:, 1].min() >= elevation - 0.021
        assert patch["mesh"].vertices[:, 1].max() <= elevation + 0.003


def test_sparse_repetition_across_only_six_of_thirty_two_bands_is_rejected():
    x = np.linspace(0.01, 2.4, 500)
    y = np.linspace(0.01, 2.4, 32)
    xx, yy = np.meshgrid(x, y)
    z = np.zeros_like(xx)
    z[:6] = 0.02 * np.sin(2 * np.pi * xx[:6] / 0.3)
    points = np.column_stack((xx.ravel(), yy.ravel(), z.ravel()))
    patch = _patch("sparse_ribs", 0, 2.4, points, np.full((len(points), 3), 0.5))

    result = regularize_planar_patches([patch])[0]

    assert result["pattern_classification"]["pattern_type"] == "simple_and_plain"
    assert result["regularization"]["primitive_count"] == 0


def test_separated_relief_strips_do_not_form_one_geometric_pattern():
    x, y = np.meshgrid(np.linspace(0.01, 0.99, 60), np.linspace(0.01, 0.99, 60))
    strips = ((y > 0.2) & (y < 0.28)) | ((y > 0.72) & (y < 0.8))
    z = np.where(strips, 0.025 * np.sin(2 * np.pi * x / 0.25), 0.0)
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    patch = _patch("separated_relief", 0, 1, points, np.full((len(points), 3), 0.5))

    result = regularize_planar_patches([patch], max_edge=0.5)[0]

    assert result["pattern_classification"]["pattern_type"] == "simple_and_plain"
    np.testing.assert_allclose(result["mesh"].vertices[:, 2], 0.0)


def test_color_change_without_straight_supported_boundary_collapses_to_one_color():
    axis = np.linspace(0.01, 1.99, 100)
    x, y = np.meshgrid(axis, axis)
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    light = 0.78 + 0.12 * x.ravel() / 2.0 + 0.08 * y.ravel() / 2.0
    colors = light[:, None] * np.array([1.0, 0.88, 0.72])
    door = (x.ravel() > 0.45) & (x.ravel() < 0.85) & (y.ravel() > 0.2) & (y.ravel() < 1.8)
    colors[door] *= np.array([0.34, 0.19, 0.10])
    cable = (x.ravel() > 0.15) & (x.ravel() < 0.19) & (y.ravel() > 0.2) & (y.ravel() < 1.8)
    colors[cable] = [0.015, 0.015, 0.015]
    patch = _patch("wall_with_door_and_cable", 0, 2, points, colors)
    patch["mesh"].vertices[:, 1] *= 2.0
    patch["support_areas"][:] = 0.0004

    result = regularize_planar_patches(
        [patch], max_edge=0.05, neutral_reference_rgb=(0.88, 0.88, 0.86)
    )[0]
    face_colors = result["mesh"].visual.face_colors[:, :3].astype(float) / 255.0
    assert result["regularization"]["visual_surface"]["kind"] == "plain_color"
    assert np.max(np.linalg.norm(face_colors - face_colors[0], axis=1)) < 0.02
    assert result["regularization"]["outlier_cells_replaced"] > 0
    white_balance = result["regularization"]["scene_white_balance"]
    assert white_balance["applied"]
    assert white_balance["adaptation_strength"] < 1.0
    assert white_balance["gain_rgb"][2] > white_balance["gain_rgb"][0]


def test_generic_surface_does_not_assume_its_dominant_material_is_white():
    points = np.array([[x, y, 0] for x in np.linspace(0.05, 0.95, 10)
                       for y in np.linspace(0.05, 0.95, 10)])
    color = np.array([0.55, 0.32, 0.18])
    patch = _patch("unreferenced_material", 0, 1, points,
                   np.tile(color, (len(points), 1)))

    result = regularize_planar_patches([patch], max_edge=0.5)[0]

    assert not result["regularization"]["scene_white_balance"]["applied"]
    np.testing.assert_allclose(
        result["regularization"]["visual_surface"]["base_color_rgb"], color, atol=0.01
    )


def test_repeating_visual_pattern_is_synthesized_as_a_tile():
    axis = np.linspace(0.01, 1.99, 100)
    x, y = np.meshgrid(axis, axis)
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    checker = ((np.floor(x / 0.2) + np.floor(y / 0.2)).astype(int) % 2).ravel()
    colors = np.where(checker[:, None] == 0, [0.85, 0.82, 0.72], [0.25, 0.28, 0.30])
    patch = _patch("tile", 0, 2, points, colors)
    patch["mesh"].vertices[:, 1] *= 2.0

    result = regularize_planar_patches([patch], max_edge=0.1)[0]

    assert result["regularization"]["visual_surface"]["kind"] == "periodic_tile"
    assert result["regularization"]["visual_surface"]["tile_shape_pixels"] != [1, 1]
    model = _visual_model(points[:, :2], colors, np.ones(len(colors)))
    tile = model["tile"]
    np.testing.assert_allclose(tile[0], tile[-1], atol=1e-12)
    np.testing.assert_allclose(tile[:, 0], tile[:, -1], atol=1e-12)


def test_repeating_visual_pattern_stays_inside_its_contiguous_linear_region():
    axis = np.linspace(0.01, 1.99, 100)
    x, y = np.meshgrid(axis, axis)
    uv = np.column_stack((x.ravel(), y.ravel()))
    checker = ((np.floor(x / 0.2) + np.floor(y / 0.2)).astype(int) % 2).ravel()
    patterned = y.ravel() < 0.6
    colors = np.tile([0.85, 0.82, 0.72], (len(uv), 1))
    colors[patterned & (checker == 1)] = [0.25, 0.28, 0.30]

    model = _visual_model(uv, colors, np.ones(len(uv)))
    sampled = _sample_visual(model, np.array([[0.1, 0.15], [0.5, 0.15], [0.1, 1.2]]))

    assert model["kind"] == "periodic_tile"
    assert model["info"]["pattern_region_uv_bounds"][3] < 0.9
    np.testing.assert_allclose(sampled[0], sampled[1], atol=0.05)
    np.testing.assert_allclose(sampled[2], [0.85, 0.82, 0.72], atol=0.03)


def test_color_split_requires_one_straight_supported_boundary():
    labels = np.zeros((40, 60), dtype=np.uint8)
    labels[:, 30:] = 1
    occupied = np.ones_like(labels, dtype=bool)
    occupied[::3, 35:45] = False
    split = _straight_color_split(labels, occupied, occupied.astype(float))

    assert split is not None
    np.testing.assert_array_equal(split, labels)


def test_scattered_color_regions_do_not_create_a_plane_split():
    yy, xx = np.indices((40, 60))
    labels = ((xx + yy) % 2).astype(np.uint8)
    occupied = np.ones_like(labels, dtype=bool)

    assert _straight_color_split(labels, occupied, occupied.astype(float)) is None


def test_nonperiodic_woodgrain_is_synthesized_as_texture_not_solid_regions():
    axis = np.linspace(0.01, 1.99, 100)
    x, y = np.meshgrid(axis, axis)
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    noise = gaussian_filter(np.random.default_rng(7).normal(size=(100, 100)), sigma=(2, 10))
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    colors = np.stack([0.38 + 0.32 * noise, 0.22 + 0.2 * noise,
                       0.12 + 0.12 * noise], axis=-1).reshape(-1, 3)
    patch = _patch("woodgrain", 0, 2, points, colors)
    patch["mesh"].vertices[:, 1] *= 2.0

    result = regularize_planar_patches([patch], max_edge=0.1)[0]

    surface = result["regularization"]["visual_surface"]
    assert surface["kind"] == "repeating_texture"
    assert surface["tile_shape_pixels"] != [1, 1]


def test_texture_presence_detects_nonperiodic_woodgrain():
    axis = np.linspace(0.01, 1.99, 100)
    x, y = np.meshgrid(axis, axis)
    uv = np.column_stack((x.ravel(), y.ravel()))
    grain = gaussian_filter(np.random.default_rng(7).normal(size=(100, 100)), sigma=(2, 10))
    grain = (grain - grain.min()) / (grain.max() - grain.min())
    colors = np.stack([0.38 + 0.32 * grain, 0.22 + 0.2 * grain,
                       0.12 + 0.12 * grain], axis=-1).reshape(-1, 3)

    result = detect_visual_texture(uv, colors, np.ones(len(colors)))

    assert result["present"]
    assert result["reason"] == "coherent_fine_scale_texture"


def test_texture_presence_ignores_smooth_shadows_and_unstructured_noise():
    axis = np.linspace(0.01, 1.99, 100)
    x, y = np.meshgrid(axis, axis)
    uv = np.column_stack((x.ravel(), y.ravel()))
    luminance = 0.58 + 0.08 * x - 0.04 * y - 0.20 * (x > 0.9)
    shadow_colors = np.stack([luminance * 0.9, luminance * 0.8,
                              luminance * 0.7], axis=-1).reshape(-1, 3)
    assert not detect_visual_texture(uv, shadow_colors, np.ones(len(shadow_colors)))["present"]

    noise = np.random.default_rng(19).normal(0.0, 0.035, len(uv))
    noisy_luminance = np.clip(0.58 + noise, 0.0, 1.0)
    noisy_colors = np.column_stack((noisy_luminance * 0.9,
                                    noisy_luminance * 0.8,
                                    noisy_luminance * 0.7))
    assert not detect_visual_texture(uv, noisy_colors, np.ones(len(noisy_colors)))["present"]


def test_unstructured_scan_color_noise_collapses_to_a_plain_surface():
    axis = np.linspace(0.01, 1.99, 100)
    x, y = np.meshgrid(axis, axis)
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    noise = np.random.default_rng(3).normal(0.0, 0.10, len(points))
    colors = np.repeat(np.clip(0.85 + noise[:, None], 0.0, 1.0), 3, axis=1)
    patch = _patch("noisy_plain_wall", 0, 2, points, colors)
    patch["mesh"].vertices[:, 1] *= 2.0

    result = regularize_planar_patches([patch], max_edge=0.1)[0]

    assert result["regularization"]["visual_surface"]["kind"] == "plain_color"
    face_colors = result["mesh"].visual.face_colors[:, :3].astype(float) / 255.0
    assert np.max(np.linalg.norm(face_colors - face_colors[0], axis=1)) < 0.02


def test_scattered_compact_color_blobs_are_removed_from_texture():
    axis = np.linspace(0.01, 1.99, 100)
    x, y = np.meshgrid(axis, axis)
    uv = np.column_stack((x.ravel(), y.ravel()))
    colors = np.tile([0.42, 0.31, 0.22], (len(uv), 1))
    for center_x, center_y in [(20, 20), (47, 31), (75, 24), (33, 70), (68, 76)]:
        blob = (np.abs(x - center_x / 50) < 0.07) & (np.abs(y - center_y / 50) < 0.07)
        colors[blob.ravel()] = [0.92, 0.91, 0.88]

    model = _visual_model(uv, colors, np.ones(len(uv)))

    assert model["kind"] == "plain_color"
    assert model["info"]["outlier_cells_replaced"] > 0
    assert model["info"]["tile_shape_pixels"] == [1, 1]


def test_neutral_shadow_regions_are_not_synthesized_as_separate_color_patches():
    axis = np.linspace(0.005, 0.995, 100)
    x, y = np.meshgrid(axis, axis)
    uv = np.column_stack((x.ravel(), y.ravel()))
    colors = np.where((x.ravel() < 0.5)[:, None], [0.8, 0.8, 0.8], [0.35, 0.35, 0.35])

    model = _visual_model(uv, colors, np.ones(len(colors)))

    assert model["kind"] == "plain_color"


def test_disconnected_color_patches_collapse_but_a_single_door_region_survives():
    count = 20
    vertices = np.array([[x / count, 0.0, y / count]
                         for y in range(count + 1) for x in range(count + 1)])
    faces = []
    for y in range(count):
        for x in range(count):
            a = y * (count + 1) + x
            faces.extend(([a, a + 1, a + count + 2], [a, a + count + 2, a + count + 1]))
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)
    centers = mesh.triangles_center
    white, brown = np.array([0.85, 0.83, 0.75]), np.array([0.35, 0.22, 0.16])
    palette = np.array([white, brown])
    patches = (((centers[:, 0] > 0.1) & (centers[:, 0] < 0.35)
                & (centers[:, 2] > 0.1) & (centers[:, 2] < 0.35))
               | ((centers[:, 0] > 0.6) & (centers[:, 0] < 0.85)
                  & (centers[:, 2] > 0.1) & (centers[:, 2] < 0.35))
               | ((centers[:, 0] > 0.3) & (centers[:, 0] < 0.55)
                  & (centers[:, 2] > 0.6) & (centers[:, 2] < 0.85)))
    fragmented_colors = np.where(patches[:, None], brown, white)
    door = ((centers[:, 0] > 0.35) & (centers[:, 0] < 0.65)
            & (centers[:, 2] > 0.1) & (centers[:, 2] < 0.9))
    door_colors = np.where(door[:, None], brown, white)

    assert not _surface_color_regions_are_connected(mesh, fragmented_colors, palette)
    assert _surface_color_regions_are_connected(mesh, door_colors, palette)


def test_texture_outside_support_extends_nearest_edge_color():
    grid = np.array([[[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]],
                     [[0.3, 0.3, 0.3], [0.4, 0.4, 0.4]]])
    model = {
        "kind": "repeating_texture",
        "bounds": np.array([[0.0, 1.0], [0.0, 1.0]]),
        "cell_size": 1.0,
        "grid": grid,
        "tile": np.array([[[0.9, 0.9, 0.9], [0.8, 0.8, 0.8]]]),
        "tile_start": (0, 0),
    }

    sampled = _sample_visual(model, np.array([[1.5, 0.5]]))

    np.testing.assert_allclose(sampled[0], grid[0, 1])


def test_unsupported_connector_inherits_one_nearest_surface_without_mosaicing():
    axis = np.linspace(0.05, 0.95, 20)
    points = np.array([[x, y, 0.0] for x in axis for y in axis])
    dark_points = points.copy()
    light_points = points.copy()
    light_points[:, 0] += 1.0
    dark = _patch("dark_surface", 0.0, 1.0, dark_points,
                  np.tile([0.20, 0.12, 0.16], (len(points), 1)))
    light = _patch("light_surface", 1.0, 2.0, light_points,
                   np.tile([0.80, 0.80, 0.75], (len(points), 1)))
    connector = _patch("unsupported_connector", 0.95, 1.05,
                       np.empty((0, 3)), np.empty((0, 3)))

    result = regularize_planar_patches([dark, light, connector], max_edge=0.05)
    connector_colors = np.unique(result[2]["mesh"].visual.face_colors[:, :3], axis=0)
    supported_colors = [result[i]["mesh"].visual.face_colors[0, :3] for i in (0, 1)]

    assert len(connector_colors) == 1
    assert any(np.array_equal(connector_colors[0], color) for color in supported_colors)


def test_welded_shell_does_not_average_distinct_face_colors():
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    mesh.apply_translation([0.0, 0.5, 0.0])
    red, blue = np.array([255, 0, 0, 255]), np.array([0, 0, 255, 255])
    face_colors = np.tile(red, (len(mesh.faces), 1))
    face_colors[len(face_colors) // 2:] = blue
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=face_colors)

    shell, topology = assemble_wall_region_planes._weld_shell([mesh], 0.0, 1.0)

    assert topology["is_closed_between_floor_and_ceiling"]
    actual = np.unique(shell.visual.vertex_colors, axis=0)
    assert all(np.array_equal(color, red) or np.array_equal(color, blue) for color in actual)
    face_corner_colors = shell.visual.vertex_colors[shell.faces]
    assert np.all(face_corner_colors == face_corner_colors[:, :1, :])
    assert topology["color_seam_vertex_count"] > 0
