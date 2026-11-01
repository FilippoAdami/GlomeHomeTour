from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
from shapely.geometry import box
from shapely.ops import unary_union

SCRIPT = Path(__file__).parents[1] / "06_mesh_elaboration" / "enclose_wall_instances.py"
sys.path.insert(0, str(SCRIPT.parent))
sys.path.insert(0, str(SCRIPT.parents[1] / "04_3DGS_to_mesh"))
SPEC = importlib.util.spec_from_file_location("enclose_wall_instances", SCRIPT)
enclose_wall_instances = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(enclose_wall_instances)
import extract_structural_meshes


def test_combined_builder_publishes_watertight_room_shell(tmp_path, monkeypatch):
    instances = tmp_path / "instances"
    instances.mkdir()
    entries = []

    def write_quad(name, points, role, color=(0.6, 0.6, 0.6), semantic_categories=None):
        path = instances / f"{name}.ply"
        vertices = np.asarray(points, dtype=float)
        triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        if semantic_categories is not None:
            colors = np.tile(np.asarray(color, dtype=float), (len(vertices), 1))
            assert extract_structural_meshes.export_colored_ply(
                path, vertices, triangles, colors,
                {"semantic_category_id": ("uchar", np.asarray(semantic_categories, dtype=np.uint8))},
            )
        else:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            mesh.paint_uniform_color(color)
            assert o3d.io.write_triangle_mesh(str(path), mesh)
        entries.append({"role": role, "file_name": path.name})

    write_quad("wall_inst000", [[0, 0, 0], [0, 0, 4], [0, 3, 4], [0, 3, 0]],
               "wall", color=(0.1, 0.05, 0.05), semantic_categories=[1, 0])
    write_quad("wall_inst001", [[4, 0, 4], [4, 0, 0], [4, 3, 0], [4, 3, 4]], "wall")
    write_quad("wall_inst002", [[4, 0, 0], [0, 0, 0], [0, 3, 0], [4, 3, 0]], "wall")
    write_quad("wall_inst003", [[0, 0, 4], [4, 0, 4], [4, 2.98, 4], [0, 2.98, 4]], "wall")
    write_quad("floor_inst000", [[0, 0, 0], [4, 0, 0], [4, 0, 4], [0, 0, 4]], "floor")
    write_quad("ceiling_inst000", [[0, 3, 4], [4, 3, 4], [4, 3, 0], [0, 3, 0]], "ceiling")
    (instances / "structural_instances_manifest.json").write_text(json.dumps({
        "instances": entries,
    }))
    output = instances / "wall_enclosing_shell.ply"
    manifest_path = instances / "wall_planes_manifest.json"
    monkeypatch.setattr(enclose_wall_instances.grow, "MIN_REGION_TRIANGLES", 1)
    appearance_samples = []
    regularize = enclose_wall_instances.assemble.regularize_planar_patches

    def capture_appearance_samples(patches, *args, **kwargs):
        appearance_samples.extend(
            np.asarray(patch["support_centers"]) for patch in patches
        )
        return regularize(patches, *args, **kwargs)

    monkeypatch.setattr(
        enclose_wall_instances.assemble, "regularize_planar_patches", capture_appearance_samples
    )
    result = enclose_wall_instances.build_wall_shell(instances, output, manifest_path)

    assert output.is_file()
    saved_shell = o3d.io.read_triangle_mesh(str(output))
    assert len(saved_shell.triangles) > 0
    vertex_colors = np.asarray(saved_shell.vertex_colors)
    assert vertex_colors.shape == (len(saved_shell.vertices), 3)
    assert np.ptp(vertex_colors, axis=0).max() > 0.2
    saved = json.loads(manifest_path.read_text())
    assert saved["semantic_category_color_samples_excluded"] >= 1
    assert "mesh_supported_corner_extensions" in saved
    door_center = np.mean(np.asarray([[0, 0, 0], [0, 0, 4], [0, 3, 4]], dtype=float), axis=0)
    assert not any(
        np.any(np.all(np.isclose(samples, door_center), axis=1))
        for samples in appearance_samples if len(samples)
    )
    assert len(saved["wall_planes"]) >= 4
    assert all(panel["bounds_y"][0] == 0.0 for panel in saved["wall_planes"])
    assert all(panel["bounds_y"][1] == saved["wall_vertical_bounds_m"][1]
               for panel in saved["wall_planes"])
    np.testing.assert_allclose(saved["wall_vertical_bounds_m"], [0.0, 3.0], atol=0.03)
    assert all(len(panel["plane_equation"]) == 4 for panel in saved["wall_planes"])
    assert all(band["is_closed"] for band in saved["horizontal_topology"]["height_bands"])
    np.testing.assert_allclose(saved["orthogonal_basis"]["dir_A_east_west"],
                               [1.0, 0.0, 0.0], atol=1e-12)
    from recognize_and_regularize_patterns import wall_mesh
    axis_a = np.asarray(saved["orthogonal_basis"]["dir_A_east_west"])
    axis_b = np.asarray(saved["orthogonal_basis"]["dir_B_north_south"])
    panel_v, panel_t = wall_mesh(saved["wall_planes"][0], axis_a, axis_b,
                                 tuple(saved["wall_vertical_bounds_m"]))
    assert len(panel_v) and len(panel_t)
    assert result["shell_written"] is True
    assert not list(instances.glob(".wall_shell_*"))
    assert not (instances / "wall_regions_colored.ply").exists()


def test_no_wall_instances_remove_stale_shell_and_write_empty_manifest(tmp_path):
    instances = tmp_path / "instances"
    instances.mkdir()
    (instances / "structural_instances_manifest.json").write_text('{"instances": []}')
    output = instances / "wall_enclosing_shell.ply"
    output.write_bytes(b"stale")
    manifest = instances / "wall_planes_manifest.json"

    result = enclose_wall_instances.build_wall_shell(instances, output, manifest)

    assert not output.exists()
    assert json.loads(manifest.read_text())["wall_planes"] == []
    assert result["shell_written"] is False


def test_supported_upper_wall_closes_an_open_floor_slice():
    planes = [
        {"id": index, "source_region_ids": [index], "axis": axis,
         "offset_m": offset, "bounds_y": list(bounds_y),
         "trim_regions": [{"span_m": span, "bounds_y": list(bounds_y)}]}
        for index, (axis, offset, span, bounds_y) in enumerate((
            ("Dir A", 0.0, [0.0, 4.0], [0.0, 3.0]),
            ("Dir A", 4.0, [0.0, 4.0], [0.0, 3.0]),
            ("Dir B", 0.0, [0.0, 4.0], [0.0, 3.0]),
            ("Dir B", 4.0, [0.0, 4.0], [2.0, 3.0]),
        ))
    ]

    extended = enclose_wall_instances.assemble._extend_supported_plane_to_close_floor_slice(
        planes, floor_y=0.0
    )
    _, topology = enclose_wall_instances.assemble._cross_section_cycles(
        enclose_wall_instances.assemble._copy_planes_for_topology(planes)
    )

    assert [item["plane_id"] for item in extended] == [3]
    assert planes[3]["trim_regions"][0]["bounds_y"] == [0.0, 3.0]
    assert any(band["is_closed"] and band["y_bounds"][0] <= 0.3 <= band["y_bounds"][1]
               for band in topology["height_bands"])


def test_wall_shell_keeps_measured_ceiling_when_one_panel_has_short_support():
    planes = [
        {"id": index, "source_region_ids": [index], "axis": axis,
         "offset_m": offset, "bounds_y": list(bounds),
         "trim_regions": [{"span_m": [0., 4.], "bounds_y": list(bounds)}]}
        for index, (axis, offset, bounds) in enumerate((
            ("Dir A", 0., [0., 3.]), ("Dir A", 4., [0., 3.]),
            ("Dir B", 0., [0., 3.]), ("Dir B", 4., [0., 2.]),
        ))
    ]

    _, topology = enclose_wall_instances.assemble._cross_section_cycles(
        planes, (0., 3.)
    )

    assert topology["extended_to_floor_ceiling"] == [0., 3.]
    assert all(band["is_closed"] for band in topology["final_topology"]["height_bands"])


def test_floor_closure_can_require_two_supported_panels():
    planes = [
        {"id": index, "source_region_ids": [index], "axis": axis,
         "offset_m": offset, "bounds_y": list(bounds),
         "trim_regions": [{"span_m": [0., 4.], "bounds_y": list(bounds)}]}
        for index, (axis, offset, bounds) in enumerate((
            ("Dir A", 0., [0., 3.]), ("Dir A", 4., [0., 3.]),
            ("Dir B", 0., [1., 3.]), ("Dir B", 4., [2., 3.]),
        ))
    ]
    for plane in planes:
        plane["base_trim_regions"] = [
            {"span_m": list(region["span_m"]), "bounds_y": list(region["bounds_y"])}
            for region in plane["trim_regions"]
        ]
        plane["local_support_panel"] = True
    before = [(plane["offset_m"], plane["trim_regions"][0]["span_m"][:]) for plane in planes]

    repairs = enclose_wall_instances.assemble._extend_supported_plane_to_close_floor_slice(planes, 0.)
    _, topology = enclose_wall_instances.assemble._cross_section_cycles(planes, (0., 3.))

    assert {item["plane_id"] for item in repairs} == {2, 3}
    assert topology["extended_to_floor_ceiling"] == [0., 3.]
    assert all(band["is_closed"] for band in topology["final_topology"]["height_bands"])
    assert [(p["offset_m"], p["trim_regions"][0]["span_m"]) for p in planes] == before


def test_floor_closure_does_not_restore_a_carved_away_panel():
    planes = [
        {"id": i, "source_region_ids": [i], "axis": axis, "offset_m": offset,
         "bounds_y": [low, 3.], "trim_regions": [{"span_m": [0., 4.], "bounds_y": [low, 3.]}]}
        for i, (axis, offset, low) in enumerate((
            ("Dir A", 0., 0.), ("Dir A", 4., 0.), ("Dir B", 0., 0.), ("Dir B", 4., 2.),
        ))
    ]
    planes[3]["carve_rects"] = [(0., 4., 0., 2.)]
    original = json.dumps(planes)

    repairs = enclose_wall_instances.assemble._extend_supported_plane_to_close_floor_slice(planes, 0.)

    assert repairs == []
    assert json.dumps(planes) == original


def test_floor_closure_runs_after_parallel_junction_connectors(monkeypatch, tmp_path):
    assembly = enclose_wall_instances.assemble
    planes = [{"id": 1}]
    connector = {"id": -1, "inferred_connector": True}
    monkeypatch.setattr(assembly, "_build_planes", lambda _: (
        planes, None, None, None, None, None,
        {"floor_y": 0., "ceiling_y": 3., "axes": None, "source_mesh": None},
    ))
    monkeypatch.setattr(assembly, "_add_parallel_carves", lambda *args: ([connector], []))

    class ReachedClosure(Exception):
        pass

    def closure(fitted, floor_y):
        assert connector in fitted
        raise ReachedClosure

    monkeypatch.setattr(assembly, "_extend_supported_plane_to_close_floor_slice", closure)
    with pytest.raises(ReachedClosure):
        assembly._assemble(tmp_path)


@pytest.mark.parametrize("cut", [
    (0., 1., 1., 2.), (-2., .5, -1., 2.), (-3., 3., -1., 4.),
    (2., 3., 0., 3.), (-1., 1., 4., 5.), (1., 2., 0., 3.),
])
def test_panel_rectangle_cut_stays_inside_source_and_is_idempotent(cut):
    assembly = enclose_wall_instances.assemble
    source = {"span_m": [-1., 1.], "bounds_y": [0., 3.]}
    expected = box(-1., 0., 1., 3.).difference(box(cut[0], cut[2], cut[1], cut[3]))
    pieces = assembly._subtract_rect(source, cut)
    repeated = [part for piece in pieces for part in assembly._subtract_rect(piece, cut)]
    for result in (pieces, repeated):
        polygons = [box(p["span_m"][0], p["bounds_y"][0],
                        p["span_m"][1], p["bounds_y"][1]) for p in result]
        actual = unary_union(polygons)
        assert actual.symmetric_difference(expected).area == pytest.approx(0., abs=1e-12)
        assert sum(p.area for p in polygons) == pytest.approx(expected.area)


def test_overlapping_collinear_panels_do_not_make_a_closed_room():
    planes = [
        {"axis": axis, "offset_m": offset,
         "trim_regions": [{"span_m": span, "bounds_y": [0., 3.]}]}
        for axis, offset, span in (("Dir B", 0., [0., 4.]),
                                   ("Dir B", 0., [0., 4.]),
                                   ("Dir A", 0., [-1., 1.]),
                                   ("Dir A", 4., [-1., 1.]))
    ]
    with pytest.raises(ValueError, match="closed, supported horizontal loop"):
        enclose_wall_instances.assemble._cross_section_cycles(planes)


def test_junction_closure_accounts_for_existing_topology_tolerance():
    assembly = enclose_wall_instances.assemble
    gap = assembly.JUNCTION_DISTANCE_M + assembly.TOPOLOGY_TOLERANCE_M * .5
    planes = [
        {"family": 0, "offset_m": 0., "span_w_m": [0., 4.], "bounds_y": [0., 3.]},
        {"family": 1, "offset_m": 4., "span_u_m": [gap, 4.], "bounds_y": [0., 3.]},
    ]
    assembly._snap_panel_junctions(planes)
    assert planes[1]["span_u_m"][0] == 0.
    assert not assembly._has_long_corner_gap(planes)

    planes[1]["span_u_m"][0] = assembly.JUNCTION_DISTANCE_M + 2 * assembly.TOPOLOGY_TOLERANCE_M
    assembly._snap_panel_junctions(planes)
    assert planes[1]["span_u_m"][0] > assembly.JUNCTION_DISTANCE_M


def _corner_support_mesh(assembly):
    along = np.repeat((np.arange(5) + 0.5) * 0.1, 6)
    heights = np.tile((np.arange(6) + 0.5) * 0.5, 5)
    first_wall = np.column_stack((np.zeros(30), heights, along))
    second_wall = np.column_stack((along, heights, np.zeros(30)))
    centers = np.vstack((first_wall, second_wall))
    normals = np.vstack((np.tile([1.0, 0.0, 0.0], (30, 1)),
                         np.tile([0.0, 0.0, 1.0], (30, 1))))
    areas = np.full(len(centers), 0.05)
    axes = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return assembly._corner_mesh_support(centers, normals, areas, axes)


def _room_with_open_corner():
    planes = []
    for plane_id, family, offset, span in (
        (0, 0, 0.0, [0.5, 4.0]),
        (1, 0, 4.0, [0.0, 4.0]),
        (2, 1, 0.0, [0.5, 4.0]),
        (3, 1, 4.0, [0.0, 4.0]),
    ):
        span_key = "span_w_m" if family == 0 else "span_u_m"
        planes.append({
            "id": plane_id, "family": family, "offset_m": offset,
            "axis": "Dir A" if family == 0 else "Dir B",
            span_key: span.copy(), "bounds_y": [0.0, 3.0],
            "trim_regions": [{"span_m": span.copy(), "bounds_y": [0.0, 3.0]}],
        })
    return planes


def test_mesh_supported_corner_extension_closes_long_gap():
    assembly = enclose_wall_instances.assemble
    planes = _room_with_open_corner()
    assert assembly._has_long_corner_gap(planes)
    extensions = assembly._snap_panel_junctions(planes, _corner_support_mesh(assembly))
    for plane in planes:
        span_key = "span_w_m" if plane["family"] == 0 else "span_u_m"
        plane["trim_regions"] = [{"span_m": plane[span_key].copy(),
                                  "bounds_y": plane["bounds_y"].copy()}]

    _, topology = assembly._cross_section_cycles(planes)

    assert planes[0]["span_w_m"][0] == 0.0
    assert planes[2]["span_u_m"][0] == 0.0
    assert len(extensions) == 1
    assert all(item["supported"] for item in extensions[0]["mesh_support"])
    assert not assembly._has_long_corner_gap(planes)
    assert topology["height_bands"][0]["is_closed"]


def test_long_corner_gap_without_mesh_evidence_stays_open():
    assembly = enclose_wall_instances.assemble
    planes = _room_with_open_corner()
    original_spans = [plane["span_w_m" if plane["family"] == 0 else "span_u_m"].copy()
                      for plane in planes]

    assert assembly._snap_panel_junctions(planes) == []
    assert [plane["span_w_m" if plane["family"] == 0 else "span_u_m"]
            for plane in planes] == original_spans
    with pytest.raises(ValueError, match="closed, supported horizontal loop"):
        assembly._cross_section_cycles(planes)


def test_wrongly_oriented_mesh_does_not_support_corner_extension():
    assembly = enclose_wall_instances.assemble
    planes = _room_with_open_corner()
    support = _corner_support_mesh(assembly)
    offsets, along, heights, _, areas = support[1]
    support[1] = (offsets, along, heights, np.zeros_like(areas), areas)

    assert assembly._snap_panel_junctions(planes, support) == []
    assert planes[0]["span_w_m"][0] == 0.5
    assert planes[2]["span_u_m"][0] == 0.5
