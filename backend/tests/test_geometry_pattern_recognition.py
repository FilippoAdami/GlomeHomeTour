#!/usr/bin/env python3
"""
Unit and Integration Tests for Geometry Pattern Recognition & Regularization
(backend/06_mesh_elaboration/recognize_and_regularize_patterns.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_dir / "06_mesh_elaboration"))

import json
import numpy as np
import pytest

from recognize_and_regularize_patterns import (
    analyze_ceiling_floor_pattern,
    analyze_vertical_wall_pattern,
    compute_face_centers_and_areas,
    regularize_beams_geometry,
    regularize_plain_geometry,
)


def create_synthetic_beamed_ceiling(
    period: float = 0.50,
    beam_width: float = 0.12,
    beam_depth: float = 0.15,
    span_s: float = 3.0,
    span_t: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates synthetic 3D mesh with periodic rectangular beams along downward plane."""
    # Transverse axis u (across beams) along X, longitudinal v (along beams) along Z
    p0 = np.array([0.0, 2.0, 0.0])
    s_min, s_max = -span_s / 2.0, span_s / 2.0
    t_min, t_max = -span_t / 2.0, span_t / 2.0

    np.random.seed(42)
    N_pts = 6000
    s_pts = np.random.uniform(s_min, s_max, N_pts)
    t_pts = np.random.uniform(t_min, t_max, N_pts)

    centers_list = []
    normals_list = []
    areas_list = []

    for s, t in zip(s_pts, t_pts):
        s_mod = (s + 100 * period) % period
        if s_mod < beam_width:
            # On beam: protrude along -Y
            centers_list.append([s, 2.0 - beam_depth, t])
            normals_list.append([0.0, -1.0, 0.0])
            areas_list.append(0.001)
            # Flank points near edges
            if s_mod < 0.02:
                centers_list.append([s, 2.0 - beam_depth / 2.0, t])
                normals_list.append([-1.0, 0.0, 0.0])
                areas_list.append(0.001)
            elif s_mod > beam_width - 0.02:
                centers_list.append([s, 2.0 - beam_depth / 2.0, t])
                normals_list.append([1.0, 0.0, 0.0])
                areas_list.append(0.001)
        else:
            # Base plane
            centers_list.append([s, 2.0, t])
            normals_list.append([0.0, -1.0, 0.0])
            areas_list.append(0.001)

    centers = np.array(centers_list, dtype=np.float64)
    normals = np.array(normals_list, dtype=np.float64)
    areas = np.array(areas_list, dtype=np.float64)
    return centers, areas, normals


def test_synthetic_beamed_ceiling_detection():
    """Verifies that synthetic beams with known period/width/depth are accurately recognized."""
    true_period = 0.50
    true_width = 0.12
    true_depth = 0.15

    centers, areas, normals = create_synthetic_beamed_ceiling(
        period=true_period, beam_width=true_width, beam_depth=true_depth
    )

    n_plane = np.array([0.0, -1.0, 0.0])
    d_plane = 2.0
    inward = np.array([0.0, -1.0, 0.0])

    cls = analyze_ceiling_floor_pattern(
        centers, areas, normals, n_plane, d_plane, inward, is_ceiling=True
    )

    assert cls.pattern_type == "beams"
    assert cls.confidence >= 0.85
    params = cls.parameters
    # Check within 10% tolerance
    assert abs(params["period_m"] - true_period) <= 0.05
    assert abs(params["beam_width_m"] - true_width) <= 0.03
    assert abs(params["beam_depth_m"] - true_depth) <= 0.03


def test_synthetic_plain_floor_detection():
    """Verifies that a flat surface with random scan noise is classified as simple_and_plain."""
    np.random.seed(42)
    x = np.linspace(-2.0, 2.0, 50)
    z = np.linspace(-2.0, 2.0, 50)
    grid_x, grid_z = np.meshgrid(x, z)
    # Flat floor at Y = -1.0 with noise std 5mm
    grid_y = -1.0 + np.random.normal(0.0, 0.005, size=grid_x.shape)

    centers = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    areas = np.ones(len(centers)) * 0.005
    normals = np.tile([0.0, 1.0, 0.0], (len(centers), 1))

    n_plane = np.array([0.0, 1.0, 0.0])
    d_plane = 1.0
    inward = np.array([0.0, 1.0, 0.0])

    cls = analyze_ceiling_floor_pattern(
        centers, areas, normals, n_plane, d_plane, inward, is_ceiling=False
    )
    assert cls.pattern_type == "simple_and_plain"


def test_vertical_wall_plain_rejection_of_noise_and_holes():
    """Verifies that negative residuals (holes) and random noise are not detected as wall patterns."""
    np.random.seed(42)
    # Wall in X-Y plane at Z = 2.0
    x = np.linspace(-2.0, 2.0, 40)
    y = np.linspace(-1.0, 2.5, 40)
    gx, gy = np.meshgrid(x, y)
    gz = 2.0 + np.random.normal(0.0, 0.005, size=gx.shape)

    # Add artificial holes (negative residuals, going into wall away from room)
    # Inward normal points towards -Z into room
    hole_mask = (np.abs(gx) < 0.5) & (np.abs(gy) < 0.5)
    gz[hole_mask] += 0.08  # moves away from room

    centers = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    areas = np.ones(len(centers)) * 0.01
    normals = np.tile([0.0, 0.0, -1.0], (len(centers), 1))

    n_wall = np.array([0.0, 0.0, 1.0])
    offset = 2.0
    inward = np.array([0.0, 0.0, -1.0])
    u_horiz = np.array([1.0, 0.0, 0.0])

    cls = analyze_vertical_wall_pattern(
        centers, areas, normals, n_wall, -offset, inward, u_horiz, (-2.0, 2.0), (-1.0, 2.5)
    )
    assert cls.pattern_type == "simple_and_plain"


def test_regularize_beams_geometry_synthesis():
    """Verifies that regularize_beams_geometry produces valid non-empty vertices and triangles."""
    n_plane = np.array([0.0, -1.0, 0.0])
    inward = np.array([0.0, -1.0, 0.0])
    u_trans = np.array([1.0, 0.0, 0.0])
    v_beam = np.array([0.0, 0.0, 1.0])

    verts, tris, colors = regularize_beams_geometry(
        plane_normal=n_plane,
        plane_offset=2.5,
        inward_normal=inward,
        u_trans=u_trans,
        v_beam=v_beam,
        period=0.55,
        width=0.14,
        depth=0.18,
        anchor_s0=0.0,
        bounds_s=(-1.5, 1.5),
        bounds_t=(-2.0, 2.0),
    )

    assert len(verts) > 0
    assert len(tris) > 0
    assert len(colors) == len(verts)
    # Check max triangle index is within bounds
    assert np.max(tris) < len(verts)


def test_regularize_plain_geometry_synthesis():
    """Verifies that regularize_plain_geometry produces a clean 4-vertex quad."""
    n_plane = np.array([0.0, 1.0, 0.0])
    u_axis = np.array([1.0, 0.0, 0.0])
    v_axis = np.array([0.0, 0.0, 1.0])

    verts, tris, colors = regularize_plain_geometry(
        plane_normal=n_plane,
        plane_offset=1.2,
        bounds_u=(-1.0, 1.0),
        bounds_v=(-1.5, 1.5),
        u_axis=u_axis,
        v_axis=v_axis,
    )

    assert len(verts) == 4
    assert len(tris) == 2
    assert len(colors) == 4


def test_current_scene_manifest_classifications():
    """Verifies current scene classifications against ground truth expectations."""
    manifest_path = Path("backend/current_scene/06_mesh_elaboration/extracted_structural_instances/plane_patterns_manifest.json")
    if not manifest_path.exists():
        manifest_path = Path("current_scene/06_mesh_elaboration/extracted_structural_instances/plane_patterns_manifest.json")
    if not manifest_path.exists():
        pytest.skip("Scene plane_patterns_manifest.json not found")

    with open(manifest_path, "r") as f:
        data = json.load(f)

    units_by_name = {u["unit_name"]: u for u in data["units"]}

    # Ground truth: ceiling_main and ceiling_sottoscala have beams
    assert "ceiling_main" in units_by_name
    assert units_by_name["ceiling_main"]["pattern_classification"]["pattern_type"] == "beams"
    main_p = units_by_name["ceiling_main"]["pattern_classification"]["parameters"]
    assert 0.45 <= main_p["period_m"] <= 0.65
    assert 0.10 <= main_p["beam_width_m"] <= 0.18

    assert ("ceiling_mezzanine" in units_by_name) or ("ceiling_sottoscala" in units_by_name)
    mezz_unit_name = "ceiling_mezzanine" if "ceiling_mezzanine" in units_by_name else "ceiling_sottoscala"
    assert units_by_name[mezz_unit_name]["pattern_classification"]["pattern_type"] == "beams"
    sotto_p = units_by_name[mezz_unit_name]["pattern_classification"]["parameters"]
    assert 0.20 <= sotto_p["period_m"] <= 0.35
    assert 0.05 <= sotto_p["beam_width_m"] <= 0.16

    # Ground truth: floors are plain
    assert "floor_level0" in units_by_name
    assert units_by_name["floor_level0"]["pattern_classification"]["pattern_type"] == "simple_and_plain"

    assert "floor_mezzanine_level1" in units_by_name
    assert units_by_name["floor_mezzanine_level1"]["pattern_classification"]["pattern_type"] == "simple_and_plain"

    # Ground truth: wall planes are plain
    for name, u in units_by_name.items():
        if "wall_plane" in name:
            assert u["pattern_classification"]["pattern_type"] == "simple_and_plain"


def test_floor_polygon_regularization_wall_alignment():
    """Verifies that regularized floor meshes align perfectly with canonical room walls."""
    instances_dir = Path("backend/current_scene/06_mesh_elaboration/extracted_structural_instances")
    if not instances_dir.exists():
        instances_dir = Path("current_scene/06_mesh_elaboration/extracted_structural_instances")
    f0_path = instances_dir / "floor_level0_regularized.ply"
    f1_path = instances_dir / "floor_mezzanine_level1_regularized.ply"
    wall_manifest = instances_dir / "wall_planes_manifest.json"

    if not f0_path.exists() or not wall_manifest.exists():
        pytest.skip("Required artifacts not found for floor alignment test")

    with open(wall_manifest, "r") as f:
        wdata = json.load(f)

    nA = np.array(wdata["orthogonal_basis"]["dir_A_east_west"])
    nB = np.array(wdata["orthogonal_basis"]["dir_B_north_south"])
    wall_offsets = {wp["name"]: wp["offset_m"] for wp in wdata["wall_planes"]}

    import open3d as o3d
    m0 = o3d.io.read_triangle_mesh(str(f0_path))
    v0 = np.asarray(m0.vertices)
    dA0 = np.dot(v0, nA)
    dB0 = np.dot(v0, nB)

    # Floor Level 0 must reach all four outer walls plus the SE column
    assert abs(dA0.max() - wall_offsets["East Wall"]) <= 0.005
    assert abs(dA0.min() - wall_offsets["West Wall"]) <= 0.005
    assert abs(dB0.max() - wall_offsets["North Wall"]) <= 0.005
    assert abs(dB0.min() - wall_offsets["South Wall"]) <= 0.005
    # Contains SE column corner
    assert any(abs(d - wall_offsets["SE Column Face"]) <= 0.005 for d in dA0)

    # Floor Mezzanine Level 1: 8-vertex notched Manhattan polygon (preserves solid West piece, prunes thin railing noise tail, tightened West edge)
    m1 = o3d.io.read_triangle_mesh(str(f1_path))
    v1 = np.asarray(m1.vertices)
    dA1 = np.dot(v1, nA)
    dB1 = np.dot(v1, nB)

    assert len(v1) == 8
    assert abs(dA1.max() - wall_offsets["East Wall"]) <= 0.005
    # Does NOT extend to West Wall (-3.25m), but stops at the tightened mezzanine open edge (~ -1.42m to -1.50m)
    assert -1.55 <= dA1.min() <= -1.40
    assert abs(dB1.max() - wall_offsets["North Wall"]) <= 0.005
    # East tab reaches the South Wall (-3.686m)
    assert abs(dB1.min() - wall_offsets["South Wall"]) <= 0.005
    # Check the generated perimeter itself rather than old scene-specific corner estimates.
    from shapely.geometry import Polygon
    assert Polygon(np.column_stack([dA1, dB1])).is_valid
    edges = np.roll(np.column_stack([dA1, dB1]), -1, axis=0)-np.column_stack([dA1, dB1])
    assert np.all(np.min(np.abs(edges), axis=1) < 1e-6)
    # Central staircase notch is open (leaving > 40cm opening)
    notch_dBs = [d for d in dB1 if d > wall_offsets["South Wall"] + 0.35 and d < -2.0]
    assert len(notch_dBs) >= 2

    # Ceiling Mezzanine underneath must inherit the exact same footprint
    cs_path = instances_dir / "ceiling_mezzanine_regularized.ply"
    if not cs_path.exists():
        cs_path = instances_dir / "ceiling_sottoscala_regularized.ply"
    assert cs_path.exists()
    mcs = o3d.io.read_triangle_mesh(str(cs_path))
    vcs = np.asarray(mcs.vertices)
    dAcs = np.dot(vcs, nA)
    dBcs = np.dot(vcs, nB)
    assert abs(dAcs.min() - dA1.min()) <= 0.02
    assert abs(dAcs.max() - dA1.max()) <= 0.02
    assert abs(dBcs.min() - dB1.min()) <= 0.02
    assert abs(dBcs.max() - dB1.max()) <= 0.02


def test_mezzanine_slab_watertight_and_thickness():
    """The slab is closed and preserves the complete floor and beam ceiling elevations."""
    import trimesh
    instances_dir = Path("backend/current_scene/06_mesh_elaboration/extracted_structural_instances")
    if not instances_dir.exists():
        instances_dir = Path("current_scene/06_mesh_elaboration/extracted_structural_instances")
    slab_path = instances_dir / "mezzanine_slab_level1.ply"
    if not slab_path.exists():
        pytest.skip("mezzanine_slab_level1.ply not found")

    mesh = trimesh.load(str(slab_path))
    assert mesh.is_watertight, "Mezzanine slab must be a watertight 3D solid (closed 2-manifold)"
    assert mesh.volume > 0.0, "Mezzanine slab solid must have positive volume"

    ceiling = trimesh.load(str(instances_dir / "ceiling_mezzanine_regularized.ply"))
    floor = trimesh.load(str(instances_dir / "floor_mezzanine_level1_regularized.ply"))
    assert floor.vertices[:, 1].min() >= ceiling.vertices[:, 1].max()-1e-7
    assert mesh.bounds[0, 1] == pytest.approx(ceiling.bounds[0, 1], abs=1e-6)
    assert mesh.bounds[1, 1] == pytest.approx(floor.bounds[1, 1], abs=1e-6)


def test_wall_vertical_trimming_between_floor_and_ceiling():
    """Verifies that wall meshes do not protrude below lowest floor or above pitched roof ceiling."""
    import open3d as o3d
    instances_dir = Path("backend/current_scene/06_mesh_elaboration/extracted_structural_instances")
    if not instances_dir.exists():
        instances_dir = Path("current_scene/06_mesh_elaboration/extracted_structural_instances")
    wall_path = instances_dir / "wall_regularized.ply"
    fc_manifest = instances_dir / "floors_ceilings_planes_manifest.json"
    if not wall_path.exists() or not fc_manifest.exists():
        pytest.skip("Required artifacts not found for wall trimming test")

    with open(fc_manifest) as f:
        fc_data = json.load(f)

    floors = [e for e in fc_data["elements"] if e["type"] == "floor"]
    lowest_f = min(floors, key=lambda f: f["mean_elevation_y"])
    f_peq = np.array(lowest_f["plane_equation"])

    ceilings = [e for e in fc_data["elements"] if e["type"] == "ceiling"]
    upper_c = max(ceilings, key=lambda c: c["mean_elevation_y"])
    c_peq = np.array(upper_c["plane_equation"])

    def get_plane_y(peq, x, z):
        return -(peq[0] * x + peq[2] * z + peq[3]) / peq[1]

    mw = o3d.io.read_triangle_mesh(str(wall_path))
    vw = np.asarray(mw.vertices)

    params = json.loads((instances_dir / "plane_patterns_manifest.json").read_text())
    roof_unit = next(u for u in params["units"] if u["unit_name"] == upper_c["name"])
    p = roof_unit["pattern_classification"]["parameters"]
    substrate_raise = p["beam_width_m"]/p["period_m"]*p["beam_depth_m"]/abs(c_peq[1])
    for v in vw:
        x, y, z = v[0], v[1], v[2]
        y_bot = get_plane_y(f_peq, x, z)
        y_top = get_plane_y(c_peq, x, z)
        # Wall vertex must be within [y_bot, y_top]
        assert y >= y_bot - 0.01, f"Wall vertex at Y={y:.3f} protrudes below lowest floor Y={y_bot:.3f}"
        assert y <= y_top + substrate_raise + 1e-6



def test_mezzanine_surfaces_and_slab_stay_inside_recessed_wall_contour():
    import trimesh
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    directory = backend_dir / 'current_scene/06_mesh_elaboration/extracted_structural_instances'
    if not (directory/'mezzanine_slab_watertight.ply').exists():
        pytest.skip('Scene artifacts unavailable')
    wall = json.loads((directory/'wall_planes_manifest.json').read_text())
    a = np.array(wall['orthogonal_basis']['dir_A_east_west'])
    b = np.array(wall['orthogonal_basis']['dir_B_north_south'])
    def footprint(filename):
        mesh = trimesh.load(directory/filename)
        ab = np.column_stack([mesh.vertices @ a, mesh.vertices @ b])
        return unary_union([Polygon(ab[t]) for t in mesh.faces if Polygon(ab[t]).area > 1e-12])
    contour = footprint('floor_level0_regularized.ply').buffer(1e-6)
    for filename in ('floor_mezzanine_level1_regularized.ply',
                     'ceiling_mezzanine_regularized.ply', 'mezzanine_slab_watertight.ply'):
        assert footprint(filename).difference(contour).area < 1e-8, filename


def test_scene_beams_parallel_to_lateral_ceiling_edges():
    directory = backend_dir / 'current_scene/06_mesh_elaboration/extracted_structural_instances'
    if not (directory/'plane_patterns_manifest.json').exists():
        pytest.skip('Scene artifacts unavailable')
    wall = json.loads((directory/'wall_planes_manifest.json').read_text())
    axes = np.array(list(wall['orthogonal_basis'].values()))
    patterns = json.loads((directory/'plane_patterns_manifest.json').read_text())
    for unit in patterns['units']:
        pattern = unit['pattern_classification']
        if pattern['pattern_type'] != 'beams':
            continue
        normal = np.array(unit['plane_equation'][:3])
        edges = axes.copy()
        edges[:, 1] = -(edges @ normal)/normal[1]
        edges /= np.linalg.norm(edges, axis=1)[:, None]
        v = np.array(pattern['parameters']['beam_longitudinal_axis_v'])
        assert np.max(np.abs(edges @ v)) == pytest.approx(1., abs=1e-12), unit['unit_name']
        import trimesh
        mesh = trimesh.load(directory / f"{unit['unit_name']}_regularized.ply")
        flanks = mesh.face_normals[np.abs(mesh.face_normals @ normal) < .4]
        assert len(flanks) > 0
        assert np.allclose(np.max(np.abs(flanks @ axes.T), axis=1), 1., atol=1e-12)


def test_perimeter_alignment_preserves_measured_beam_dimensions():
    centers, areas, normals = create_synthetic_beamed_ceiling()
    normal = np.array([0., -1., 0.])
    original = analyze_ceiling_floor_pattern(centers, areas, normals, normal, 2., normal, True)
    angle = np.deg2rad(2.)
    axes = (np.array([np.cos(angle), 0., np.sin(angle)]), np.array([-np.sin(angle), 0., np.cos(angle)]))
    aligned = analyze_ceiling_floor_pattern(centers, areas, normals, normal, 2., normal, True, perimeter_axes=axes)
    assert aligned.pattern_type == original.pattern_type == 'beams'
    for name in ('period_m', 'beam_width_m', 'beam_depth_m', 'space_between_beams_m'):
        assert aligned.parameters[name] == original.parameters[name]
    v = np.array(aligned.parameters['beam_longitudinal_axis_v'])
    assert max(abs(v @ axis) for axis in axes) == pytest.approx(1., abs=1e-12)


def test_wall_small_end_misalignment_uses_local_roof_height():
    from recognize_and_regularize_patterns import wall_mesh
    a, b = np.array([1.,0.,0.]), np.array([0.,0.,1.])
    faces = np.array([[0,1,2],[0,2,3]])
    floor = np.array([[0.,0.,0.],[2.,0.,0.],[2.,0.,2.],[0.,0.,2.]])
    roof = floor.copy(); roof[:,1] = 2.+roof[:,2]*.2
    wall = {'axis':'Dir A','offset_m':0.,'span_w_m':[-.05,2.05],
            'bounds_y':[0.,3.5],'normal_unit':a.tolist()}
    vertices, _ = wall_mesh(wall, {'a':a,'b':b,'surfaces':[
        {'role':'floor','generated':(floor,faces)}, {'role':'ceiling','generated':(roof,faces)}]})
    assert vertices[:,1].max() < 2.42
    assert vertices[:,1].min() == pytest.approx(0.)


def test_tall_wall_small_horizontal_overshoot_is_trimmed_but_large_extension_survives():
    from recognize_and_regularize_patterns import wall_mesh
    a, b = np.array([1.,0.,0.]), np.array([0.,0.,1.])
    faces = np.array([[0,1,2],[0,2,3]])
    floor = np.array([[0.,0.,0.],[2.,0.,0.],[2.,0.,2.],[0.,0.,2.]])
    roof = floor.copy(); roof[:,1] = 3.
    units = {'a':a,'b':b,'surfaces':[
        {'role':'floor','generated':(floor,faces)}, {'role':'ceiling','generated':(roof,faces)}]}
    wall = {'axis':'Dir A','offset_m':0.,'span_w_m':[-.08,2.08],
            'bounds_y':[0.,3.],'normal_unit':a.tolist()}
    vertices,_ = wall_mesh(wall,units)
    assert vertices[:,2].min() == pytest.approx(0.)
    assert vertices[:,2].max() == pytest.approx(2.)
    wall['span_w_m'] = [-.5,2.5]
    vertices,_ = wall_mesh(wall,units)
    assert vertices[:,2].min() == pytest.approx(-.5)
    assert vertices[:,2].max() == pytest.approx(2.5)
