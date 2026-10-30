"""Regression coverage for scene-independent wall-plane discovery."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

o3d = pytest.importorskip("open3d")

MODULE = Path(__file__).parents[1] / "06_mesh_elaboration" / "fit_wall_planes.py"
SPEC = importlib.util.spec_from_file_location("fit_wall_planes", MODULE)
fit_wall_planes = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(fit_wall_planes)


def _quad(points: list[np.ndarray], vertices: list[np.ndarray], triangles: list[list[int]]) -> None:
    start = len(vertices); vertices.extend(points)
    triangles.extend(([start, start + 1, start + 2], [start, start + 2, start + 3]))


def _wall_mesh(path: Path, angle: float, translation: tuple[float, float],
               disconnected: bool = False, noise: bool = True) -> None:
    """A rotated room with a recess and optional separate wall, plus small noise."""
    direction = np.array([np.cos(angle), 0., np.sin(angle)])
    normal = np.array([-np.sin(angle), 0., np.cos(angle)])
    origin = np.array([translation[0], 0., translation[1]])
    vertices: list[np.ndarray] = []; triangles: list[list[int]] = []
    def point(t: float, n: float, y: float) -> np.ndarray:
        return origin + t * direction + n * normal + np.array([0., y, 0.])
    # Rectangle sides and a recessed return: all finite, no assumed coordinate range.
    for t0, t1, n0, n1 in ((0., 6., 0., 0.), (0., 6., 4., 4.), (0., 0., 0., 4.), (6., 6., 0., 4.),
                            (2., 3., 1., 1.)):
        if t0 == t1:
            _quad([point(t0, n0, 0.), point(t0, n1, 0.), point(t0, n1, 3.), point(t0, n0, 3.)], vertices, triangles)
        else:
            _quad([point(t0, n0, 0.), point(t1, n1, 0.), point(t1, n1, 3.), point(t0, n0, 3.)], vertices, triangles)
    if disconnected:
        _quad([point(10., 8., 0.), point(13., 8., 0.), point(13., 8., 2.5), point(10., 8., 2.5)], vertices, triangles)
    # Moderate, deterministic vertical diagonal scan noise must not become a plane.
    if noise:
        for index in range(24):
            t, n = 20. + .09 * index, 20. + .07 * (index % 5)
            _quad([point(t, n, 0.), point(t + .05, n + .08, 0.),
                   point(t + .05, n + .08, .05), point(t, n, .05)], vertices, triangles)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile([.3, .5, .7], (len(vertices), 1)))
    o3d.io.write_triangle_mesh(str(path), mesh, write_vertex_colors=True)


def test_removes_horizontal_faces_through_thirty_degrees_only():
    vertices, triangles = [], []
    for index, angle in enumerate((0., 30., 30.01, 90.)):
        offset = 2. * index
        start = len(vertices)
        vertices.extend(([offset, 0., 0.], [offset + 1., 0., 0.],
                         [offset, np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]))
        triangles.append([start, start + 1, start + 2])

    kept, _, _, _, stats = fit_wall_planes._remove_horizontal_faces(
        np.asarray(vertices), np.asarray(triangles)
    )

    assert kept.tolist() == [[6, 7, 8], [9, 10, 11]]
    assert stats["horizontal_triangles_removed"] == 2
    assert stats["degenerate_triangles_removed"] == 0


def test_wall_enclosure_can_ignore_ceiling_support(tmp_path: Path):
    instances = tmp_path / "instances"
    instances.mkdir()
    entries = []
    for role, height in (("floor", 0.0), ("ceiling", 10.0)):
        path = instances / f"{role}_inst000.ply"
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(
            [[0, height, 0], [1, height, 0], [0, height, 1]]
        )
        mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
        assert o3d.io.write_triangle_mesh(str(path), mesh)
        entries.append({"role": role, "file_name": path.name})
    (instances / "structural_instances_manifest.json").write_text(
        json.dumps({"instances": entries})
    )
    wall_vertices = np.array([[0, 0, 0], [1, 1, 1], [2, 3, 2]], dtype=float)
    wall_centers = wall_vertices.copy()
    wall_areas = np.ones(len(wall_vertices), dtype=float)

    with_ceiling = fit_wall_planes._vertical_enclosure_bounds(
        instances, wall_vertices, wall_centers, wall_areas
    )
    wall_only = fit_wall_planes._vertical_enclosure_bounds(
        instances, wall_vertices, wall_centers, wall_areas, include_ceilings=False
    )

    assert with_ceiling[1] == pytest.approx(10.0)
    assert wall_only[0] == pytest.approx(0.0)
    assert wall_only[1] == pytest.approx(3.0)
    assert wall_only[2] == "structural floors and wall mesh top"
    np.testing.assert_array_equal(wall_only[6], wall_centers)


@pytest.mark.parametrize("angle,translation", [(0., (0., 0.)), (np.deg2rad(31.), (123.4, -57.8))])
def test_discovers_finite_planes_without_scene_coordinates(tmp_path: Path, angle: float, translation: tuple[float, float]) -> None:
    mesh_path = tmp_path / "wall_inst001.ply"
    _wall_mesh(mesh_path, angle, translation, noise=False)
    instances = tmp_path / "instances"; instances.mkdir()
    (instances / "structural_instances_manifest.json").write_text(json.dumps({"instances": [
        {"role": "wall", "file_path": str(mesh_path)},
    ]}))
    for name in ("wall_orientation_filtered.ply", "wall_cleaned_pieces.ply",
                 "wall_enclosing_planes.ply", "wall_planes_manifest.json",
                 "wall_planes_before_rounding_manifest.json"):
        (instances / name).write_text("stale output")
    individual = instances / "wall_enclosing_planes_individual"
    individual.mkdir()
    (individual / "wall_plane_999.ply").write_text("stale panel")
    connectors = instances / "wall_closure_connectors"
    connectors.mkdir()
    (connectors / "stale.ply").write_text("stale connector")
    old_argv = sys.argv
    try:
        sys.argv = ["fit_wall_planes.py", "--instances-dir", str(instances), "--output-dir", str(instances)]
        assert fit_wall_planes.main() == 0
    finally:
        sys.argv = old_argv
    result = json.loads((instances / "wall_planes_manifest.json").read_text())
    planes = result["wall_planes"]
    assert len(planes) == 4
    assert "plane_intersection_cleanup" not in result
    assert all(np.isfinite(plane["plane_equation"]).all() for plane in planes)
    assert all(plane["bounds_y"][1] > plane["bounds_y"][0] for plane in planes)
    vertical = result["plane_constraints"]["vertical_enclosure"]
    assert all(plane["bounds_y"] == pytest.approx([vertical["floor_y_m"], vertical["ceiling_y_m"]])
               for plane in planes)
    assert result["horizontal_topology"]["open_endpoint_count"] == 0
    assert result["horizontal_topology"]["minimum_cycle_rank"] >= 1
    assert result["plane_constraints"]["plane_layout_objective"]["plane_count_penalty_m"] == .0005
    assert result["outside_enclosing_planes"]["triangle_fraction"] <= .02
    assert all(max(plane.get("span_u_m", plane.get("span_w_m"))) - min(plane.get("span_u_m", plane.get("span_w_m"))) < 7 for plane in planes)
    a = np.array(result["orthogonal_basis"]["dir_A_east_west"])
    assert abs(np.dot(a[[0, 2]], [np.cos(angle), np.sin(angle)])) > .999
    assert (instances / "wall_cleaned_pieces.ply").is_file()
    assert (instances / "wall_enclosing_planes.ply").is_file()
    assert not (instances / "wall_planes_before_rounding_manifest.json").exists()
    assert not (individual / "wall_plane_999.ply").exists()
    assert not connectors.exists()


def test_fitting_rejects_disconnected_wall_support(tmp_path: Path, monkeypatch, capsys):
    path = tmp_path / "wall.ply"
    _wall_mesh(path, 0., (0., 0.), disconnected=True, noise=False)
    monkeypatch.setattr(sys, "argv", ["fit_wall_planes", "--wall-mesh", str(path),
                                      "--output-dir", str(tmp_path)])

    assert fit_wall_planes.main() == 1
    assert "not horizontally closed" in capsys.readouterr().err
    assert not (tmp_path / "wall_planes_manifest.json").exists()



def test_discovery_rejects_wall_tabs_narrower_than_20cm():
    vertices, triangles = [], []
    _quad([np.array(p) for p in [(0,0,0),(0,0,3),(0,3,3),(0,3,0)]], vertices, triangles)
    _quad([np.array(p) for p in [(1,0,0),(1,0,.08),(1,3,.08),(1,3,0)]], vertices, triangles)
    vertices, triangles = np.array(vertices), np.array(triangles)
    corners = vertices[triangles]
    cross = np.cross(corners[:,1]-corners[:,0],corners[:,2]-corners[:,0])
    area2 = np.linalg.norm(cross,axis=1)
    planes,_ = fit_wall_planes._discover(vertices,triangles,cross/area2[:,None],corners.mean(axis=1),
        area2/2,np.array([1.,0.,0.]),np.array([0.,0.,1.]),.10)
    assert len(planes) == 1
    assert planes[0]['offset_m'] == pytest.approx(0.)


@pytest.mark.parametrize(("span", "bounds_y", "expected"), [
    ([0., .19], [0., 3.], False),
    ([0., .20], [0., 3.], False),
    ([0., 1.1], [0., .20], False),
    ([0., 1.], [0., .21], False),
    ([0., .99], [0., .20], False),
])
def test_final_plane_dimensions_must_strictly_exceed_20cm_and_1m(span, bounds_y, expected):
    assert fit_wall_planes._meets_minimum_plane_dimensions(span, bounds_y) is expected


def test_overlapping_displaced_support_becomes_one_supported_outer_plane(tmp_path, monkeypatch):
    path = tmp_path / "wall.ply"
    vertices, triangles = [], []
    _quad([np.array(p) for p in [(0, 0, 0), (0, 3, 0), (0, 3, 4), (0, 0, 4)]], vertices, triangles)
    # This displaced support patch is geometry, not a semantic door label.
    _quad([np.array(p) for p in [(.15, .1, 1), (.15, 2.5, 1), (.15, 2.5, 2), (.15, .1, 2)]], vertices, triangles)
    _quad([np.array(p) for p in [(4, 0, 0), (4, 0, 4), (4, 3, 4), (4, 3, 0)]], vertices, triangles)
    _quad([np.array(p) for p in [(0, 0, 0), (4, 0, 0), (4, 3, 0), (0, 3, 0)]], vertices, triangles)
    _quad([np.array(p) for p in [(0, 0, 4), (0, 3, 4), (4, 3, 4), (4, 0, 4)]], vertices, triangles)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.paint_uniform_color([.6, .6, .6])
    assert o3d.io.write_triangle_mesh(str(path), mesh)

    monkeypatch.setattr(sys, "argv", ["fit_wall_planes", "--wall-mesh", str(path),
                                        "--output-dir", str(tmp_path)])
    assert fit_wall_planes.main() == 0
    result = json.loads((tmp_path / "wall_planes_manifest.json").read_text())
    planes = result["wall_planes"]

    assert len(planes) == 4
    merged = [plane for plane in planes if plane["axis"] == "Dir A" and plane["source"]["triangle_count"] == 4]
    assert len(merged) == 1
    assert merged[0]["span_w_m"] == pytest.approx([0., 4.])
    assert merged[0]["offset_m"] == pytest.approx(0.)
    assert merged[0]["source"]["triangle_count"] == 4
    assert result["horizontal_topology"]["connected_component_count"] == 1
    assert result["plane_constraints"]["minimum_plane_dimensions_m"] == {
        "shorter": .2, "longer": 1.0, "strictly_greater_than": True,
    }
    assert "door_opening_splits" not in result
    assert "door_closure_connectors" not in result
    assert "rounding" not in result


def test_diagonal_support_outside_enclosure_over_two_percent_is_rejected(tmp_path, monkeypatch, capsys):
    path = tmp_path/'wall.ply'
    _wall_mesh(path,0.,(0.,0.), noise=False)
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices, faces = np.asarray(mesh.vertices).tolist(), np.asarray(mesh.triangles).tolist()
    # An unfit outside fragment contributes triangles to the strict 2% count cap.
    _quad([np.array(p) for p in [(8,1,2),(8,2,2),(8,2,2.1),(8,1,2.1)]], vertices, faces)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.full((len(vertices),3),.6))
    assert o3d.io.write_triangle_mesh(str(path),mesh)
    monkeypatch.setattr(sys,'argv',['fit_wall_planes','--wall-mesh',str(path),'--output-dir',str(tmp_path)])
    assert fit_wall_planes.main() == 1
    assert "maximum is 2%" in capsys.readouterr().err


def test_horizontal_faces_are_removed_before_wall_fitting(tmp_path, monkeypatch):
    path = tmp_path / 'wall.ply'
    _wall_mesh(path, 0., (0., 0.), noise=False)
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices = np.asarray(mesh.vertices).tolist()
    triangles = np.asarray(mesh.triangles).tolist()
    start = len(vertices)
    vertices.extend(([0., -1., 0.], [1., -1., 0.], [1., -1., 1.], [0., -1., 1.]))
    triangles.extend(([start, start + 1, start + 2], [start, start + 2, start + 3]))
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.full((len(vertices), 3), .6))
    assert o3d.io.write_triangle_mesh(str(path), mesh)
    monkeypatch.setattr(sys, 'argv', ['fit_wall_planes', '--wall-mesh', str(path), '--output-dir', str(tmp_path)])

    assert fit_wall_planes.main() == 0

    result = json.loads((tmp_path / 'wall_planes_manifest.json').read_text())
    stats = result['filtering_stats']
    filtered = o3d.io.read_triangle_mesh(result['artifacts']['orientation_filtered_input'])
    assert stats['horizontal_tolerance_degrees'] == 30.
    assert stats['horizontal_triangles_removed'] == 2
    assert stats['triangles_after_orientation_filter'] == len(triangles) - 2
    assert len(filtered.triangles) == len(triangles) - 2


def test_outward_plane_offset_uses_triangle_count_quantile_on_either_room_side():
    projections = np.array([0.] * 99 + [.5])

    for side in (-1., 1.):
        offset = fit_wall_planes._outward_plane_offset(projections, side)
        assert offset == pytest.approx(0.)


def test_overlapping_same_facing_patches_become_one_supported_plane():
    vertices = np.array([[0., 0., 0.], [0., 3., 0.], [0., 3., 2.],
                         [.03, 0., 1.], [.03, 3., 1.], [.03, 3., 3.]])
    triangles = np.array([[0, 1, 2], [3, 4, 5]])
    planes = [{"id": 1, "name": "wall_plane_001", "axis": "Dir A", "offset_m": 0.,
               "normal_unit": np.array([1., 0., 0.]), "span_w_m": [0., 2.], "bounds_y": [0., 3.],
               "source_bounds_y": [0., 3.], "plane_height_m": 3., "support_area_m2": .5,
               "source_faces": np.array([0])},
              {"id": 2, "name": "wall_plane_002", "axis": "Dir A", "offset_m": .03,
               "normal_unit": np.array([1., 0., 0.]), "span_w_m": [1., 3.], "bounds_y": [0., 3.],
               "source_bounds_y": [0., 3.], "plane_height_m": 3., "support_area_m2": .5,
               "source_faces": np.array([1])}]

    merged = fit_wall_planes._merge_overlapping_parallel_planes(
        planes, vertices, triangles, np.array([.5, .5]),
        np.array([1., 0., 0.]), np.array([0., 0., 1.]),
    )

    assert len(merged) == 1
    assert merged[0]["span_w_m"] == pytest.approx([0., 3.])
    assert merged[0]["offset_m"] == pytest.approx(.03)
    assert merged[0]["source_faces"].tolist() == [0, 1]


def test_local_fit_split_has_unique_support_and_dimension_valid_step_connector():
    vertices, triangles = [], []
    _quad([np.array(p) for p in [(0., 0., 0.), (0., 3., 0.), (0., 3., 2.), (0., 0., 2.)]],
          vertices, triangles)
    _quad([np.array(p) for p in [(.5, 0., 2.), (.5, 3., 2.), (.5, 3., 4.), (.5, 0., 4.)]],
          vertices, triangles)
    vertices, triangles = np.asarray(vertices), np.asarray(triangles)
    corners = vertices[triangles]
    double_areas = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0],
                                           corners[:, 2] - corners[:, 0]), axis=1)
    centers = corners.mean(axis=1)
    areas = double_areas / 2
    plane = {"id": 1, "name": "wall_plane_001", "axis": "Dir A", "offset_m": .5,
             "normal_unit": np.array([1., 0., 0.]), "span_w_m": [0., 4.],
             "bounds_y": [0., 3.], "source_span_m": [0., 4.],
             "source_bounds_y": [0., 3.], "plane_height_m": 3.,
             "support_area_m2": float(areas.sum()), "support_plane_offset_m": .25,
             "source_faces": np.arange(len(triangles))}

    split = fit_wall_planes._subdivide_supported_planes(
        [plane], vertices, triangles, centers, areas,
        np.array([1., 0., 0.]), np.array([0., 0., 1.]),
    )

    assert len(split) == 3
    supported = [item for item in split if item["source_faces"].size]
    connector = next(item for item in split if item.get("inferred_connector"))
    assert sorted(np.concatenate([item["source_faces"] for item in supported]).tolist()) == [0, 1, 2, 3]
    fit_wall_planes._validate_unique_support(split)
    assert fit_wall_planes._meets_minimum_plane_dimensions(
        connector["span_u_m"], connector["bounds_y"]
    )


def test_layout_objective_charges_for_each_added_panel():
    vertices = np.array([[0., 0., 0.], [0., 2., 0.], [0., 0., 2.]])
    triangles = np.array([[0, 1, 2]])
    areas = np.array([2.])
    plane = {"id": 1, "axis": "Dir A", "offset_m": 0.,
             "normal_unit": np.array([1., 0., 0.]), "span_w_m": [0., 2.],
             "bounds_y": [0., 2.]}
    connector = {"id": 2, "axis": "Dir B", "offset_m": 1.,
                 "normal_unit": np.array([0., 0., 1.]), "span_u_m": [0., 2.],
                 "bounds_y": [0., 2.], "inferred_connector": True}
    a, b = np.array([1., 0., 0.]), np.array([0., 0., 1.])

    one_plane = fit_wall_planes._layout_objective(
        [plane], vertices, triangles, areas, a, b
    )
    with_connector = fit_wall_planes._layout_objective(
        [plane, connector], vertices, triangles, areas, a, b
    )

    assert with_connector - one_plane == pytest.approx(
        fit_wall_planes.WALL_PLANE_COUNT_PENALTY_M
    )


def test_outside_measure_uses_the_unique_closest_panel_owner():
    vertices = np.array([[.01, 0., 0.], [.01, 2., 0.], [.01, 0., 2.]])
    triangles = np.array([[0, 1, 2]])
    planes = [
        {"id": 1, "axis": "Dir A", "offset_m": 0.,
         "normal_unit": np.array([1., 0., 0.]), "span_w_m": [0., 2.],
         "bounds_y": [0., 2.]},
        {"id": 2, "axis": "Dir B", "offset_m": 0.,
         "normal_unit": np.array([0., 0., -1.]), "span_u_m": [0., .02],
         "bounds_y": [0., 2.]},
    ]

    outside = fit_wall_planes._outside_panel_triangles(
        planes, vertices, triangles, np.array([1., 0., 0.]), np.array([0., 0., 1.])
    )

    assert outside.tolist() == [False]


def test_overlapping_same_facing_wall_layers_collapse_into_one_envelope():
    vertices = np.array([[0., 0., 0.], [0., 3., 0.], [0., 3., 2.],
                         [.15, 0., .05], [.15, 3., .05], [.15, 3., 2.05],
                         [.30, 0., 1.], [.30, 3., 1.], [.30, 3., 3.]])
    triangles = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    planes = []
    for index, (offset, span) in enumerate(((0., [0., 2.]), (.15, [.05, 2.05]),
                                            (.30, [1., 3.])), 1):
        planes.append({"id": index, "name": f"wall_plane_{index:03d}", "axis": "Dir A",
                       "offset_m": offset, "normal_unit": np.array([1., 0., 0.]),
                       "span_w_m": span, "bounds_y": [0., 3.], "source_bounds_y": [0., 3.],
                       "plane_height_m": 3., "support_area_m2": 3.,
                       "source_faces": np.array([index - 1])})

    merged = fit_wall_planes._merge_overlapping_parallel_planes(
        planes, vertices, triangles, np.ones(3),
        np.array([1., 0., 0.]), np.array([0., 0., 1.]),
    )

    assert len(merged) == 1
    assert merged[0]["source_faces"].tolist() == [0, 1, 2]
    assert merged[0]["offset_m"] == pytest.approx(.30)
    assert merged[0]["span_w_m"] == pytest.approx([0., 3.])


def test_local_parallel_feature_does_not_leave_an_overlapping_plane():
    vertices = np.array([[0., 0., 0.], [0., 4., 0.], [0., 4., 4.],
                         [.25, 0., .1], [.25, 3., .1], [.25, 3., 1.1]])
    triangles = np.array([[0, 1, 2], [3, 4, 5]])
    planes = [
        {"id": 1, "name": "wall", "axis": "Dir A", "offset_m": 0.,
         "normal_unit": np.array([1., 0., 0.]), "span_w_m": [0., 4.],
         "bounds_y": [0., 4.], "source_bounds_y": [0., 4.],
         "plane_height_m": 4., "support_area_m2": 16., "source_faces": np.array([0])},
        {"id": 2, "name": "feature", "axis": "Dir A", "offset_m": .25,
         "normal_unit": np.array([1., 0., 0.]), "span_w_m": [.1, 1.1],
         "bounds_y": [0., 3.], "source_bounds_y": [0., 3.],
         "plane_height_m": 3., "support_area_m2": 3., "source_faces": np.array([1])},
    ]

    merged = fit_wall_planes._merge_overlapping_parallel_planes(
        planes, vertices, triangles, np.array([1., 1.]),
        np.array([1., 0., 0.]), np.array([0., 0., 1.]),
    )

    assert len(merged) == 1
    assert merged[0]["source_faces"].tolist() == [0, 1]
    assert merged[0]["offset_m"] == pytest.approx(.25)
    assert merged[0]["span_w_m"] == pytest.approx([0., 4.])


def test_horizontal_topology_requires_one_closed_connected_network():
    planes = [
        {"axis": "Dir A", "offset_m": 0., "span_w_m": [0., 2.], "bounds_y": [0., 3.]},
        {"axis": "Dir A", "offset_m": 4., "span_w_m": [0., 2.], "bounds_y": [0., 3.]},
        {"axis": "Dir B", "offset_m": 0., "span_u_m": [0., 4.], "bounds_y": [0., 3.]},
        {"axis": "Dir B", "offset_m": 2., "span_u_m": [0., 4.], "bounds_y": [0., 3.]},
    ]
    assert fit_wall_planes._horizontal_topology(planes, .01)["minimum_cycle_rank"] == 1
    with pytest.raises(ValueError, match="open panel endpoints"):
        fit_wall_planes._horizontal_topology(planes[:-1], .01)


def test_support_triangle_cannot_belong_to_two_planes():
    planes = [
        {"name": "first", "source_faces": np.array([4, 7])},
        {"name": "second", "source_faces": np.array([7, 9])},
    ]

    with pytest.raises(ValueError, match="assigned to both first and second"):
        fit_wall_planes._validate_unique_support(planes)


def test_projected_triangle_has_one_closest_panel_owner():
    vertices = np.array([
        [.1, .2, .2], [.1, .4, .2], [.1, .2, .4],
        [.9, .2, .2], [.9, .4, .2], [.9, .2, .4],
        [.1, .2, 3.], [.1, .4, 3.], [.1, .2, 3.2],
    ])
    triangles = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    planes = [
        {"id": 1, "axis": "Dir A", "offset_m": 0., "span_w_m": [0., 2.],
         "bounds_y": [0., 2.]},
        {"id": 2, "axis": "Dir A", "offset_m": 1., "span_w_m": [0., 2.],
         "bounds_y": [0., 2.]},
    ]

    owners, distances = fit_wall_planes._assign_support_owners(
        planes, vertices, triangles,
        np.array([1., 0., 0.]), np.array([0., 0., 1.]),
    )

    assert owners.tolist() == [1, 2, -1]
    assert distances.tolist() == pytest.approx([.1, .1, float("inf")])


def test_output_cleanup_refuses_to_delete_an_input_mesh(tmp_path: Path):
    source = tmp_path / "wall_enclosing_planes.ply"
    source.write_text("input mesh")

    with pytest.raises(ValueError, match="would replace an input mesh"):
        fit_wall_planes._clear_wall_fit_outputs(tmp_path, [source])

    assert source.read_text() == "input mesh"


def test_corner_snap_closes_small_endpoint_gaps():
    planes = [
        {"axis": "Dir A", "offset_m": 0., "span_w_m": [-.69, 2.], "bounds_y": [0., 3.]},
        {"axis": "Dir B", "offset_m": -1., "span_u_m": [.31, 1.], "bounds_y": [0., 3.]},
    ]
    offsets_before = [plane["offset_m"] for plane in planes]

    fit_wall_planes._snap_plane_corners(planes, fit_wall_planes.WALL_JUNCTION_SNAP_DISTANCE_M)

    assert planes[0]["span_w_m"] == pytest.approx([-1., 2.])
    assert planes[1]["span_u_m"] == pytest.approx([0., 1.])
    assert [plane["offset_m"] for plane in planes] == offsets_before
