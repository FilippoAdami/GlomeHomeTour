import pytest
import numpy as np
import open3d as o3d
from pathlib import Path
import tempfile
import sys
import trimesh

backend_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_dir / "06c_semantic_segmentation"))
from step1_remove_mesh_floaters import profile_component_geometry, clean_mesh_floaters


def test_pipeline_orders_cleanup_before_web_export(tmp_path, monkeypatch):
    import run_pipeline
    from server import pipeline_runner

    names = [name for name, _ in run_pipeline.STEPS]
    assert names.index("mesh") + 1 == names.index("mesh_cleanup")
    workspace = tmp_path / "scene"
    raw_dir = workspace / "04_3DGS_to_mesh"
    clean_dir = workspace / "06c_semantic_segmentation/step1_cleaned_mesh"
    splat_dir = workspace / "03_FastGS_DNSplatter"
    for directory in (raw_dir, clean_dir, splat_dir):
        directory.mkdir(parents=True)
    o3d.io.write_triangle_mesh(str(raw_dir / "mesh.ply"), create_flat_sheet())
    o3d.io.write_triangle_mesh(str(clean_dir / "mesh_sealed_colored.ply"), o3d.geometry.TriangleMesh.create_box())
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array([[0., 0., 0.], [1., 1., 1.]])))
    o3d.io.write_point_cloud(str(splat_dir / "point_cloud_INRIA.ply"), cloud)
    monkeypatch.setattr(pipeline_runner, "OUTPUTS_DIR", tmp_path / "exports")

    mesh_url, _ = pipeline_runner.optimize_3d_assets(workspace, "test")
    assert mesh_url.endswith("mesh_web.obj")
    assert len(trimesh.load(tmp_path / "exports/test/mesh_web.obj").faces) == 12


def create_solid_cylinder(center=(0, 0, 0), radius=0.2, height=0.8, resolution=20):
    """Creates a closed solid 3D cylinder."""
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution)
    cylinder.translate(center)
    return cylinder


def create_flat_sheet(center=(0, 0, 0), size=0.5):
    """Creates a flat 2D square sheet with zero volumetric thickness."""
    v = np.array([
        [-size/2, 0.0, -size/2],
        [ size/2, 0.0, -size/2],
        [ size/2, 0.0,  size/2],
        [-size/2, 0.0,  size/2],
    ]) + np.array(center)
    f = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    return mesh


def test_profile_component_geometry_solid_vs_flat():
    # 1. Test solid cylinder
    cyl = create_solid_cylinder(radius=0.15, height=0.7)
    verts = np.asarray(cyl.vertices)
    tris = np.asarray(cyl.triangles)
    
    # Calculate areas
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    
    geo_solid = profile_component_geometry(tris, verts, areas, np.arange(len(tris)))
    assert geo_solid["thickness"] > 0.05
    assert geo_solid["solidity"] > 0.50
    assert geo_solid["boundary_openness"] < 0.01  # watertight
    
    # 2. Test flat sheet
    sheet = create_flat_sheet()
    s_verts = np.asarray(sheet.vertices)
    s_tris = np.asarray(sheet.triangles)
    s_v0, s_v1, s_v2 = s_verts[s_tris[:, 0]], s_verts[s_tris[:, 1]], s_verts[s_tris[:, 2]]
    s_areas = 0.5 * np.linalg.norm(np.cross(s_v1 - s_v0, s_v2 - s_v0), axis=1)
    
    geo_flat = profile_component_geometry(s_tris, s_verts, s_areas, np.arange(len(s_tris)))
    assert geo_flat["thickness"] < 0.01
    assert geo_flat["boundary_openness"] > 0.50  # all perimeter edges open


def test_clean_mesh_floaters_synthetic():
    # Build compound mesh with:
    # 1. Primary main body (large sphere)
    # 2. Detached solid object (cylinder, e.g. boxing bag)
    # 3. Detached flat floater sheet
    # 4. Detached tiny crumb
    main_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.5, resolution=30)
    solid_obj = create_solid_cylinder(center=(3.0, 0, 0), radius=0.2, height=0.9, resolution=30)
    flat_floater = create_flat_sheet(center=(0, 3.0, 0), size=0.4)
    crumb = o3d.geometry.TriangleMesh.create_tetrahedron(radius=0.01)
    crumb.translate((0, 0, 3.0))

    combined = o3d.geometry.TriangleMesh()
    for m in [main_sphere, solid_obj, flat_floater, crumb]:
        v_offset = len(combined.vertices)
        combined.vertices = o3d.utility.Vector3dVector(
            np.vstack([np.asarray(combined.vertices), np.asarray(m.vertices)]) if v_offset > 0 else np.asarray(m.vertices)
        )
        combined.triangles = o3d.utility.Vector3iVector(
            np.vstack([np.asarray(combined.triangles), np.asarray(m.triangles) + v_offset]) if v_offset > 0 else np.asarray(m.triangles)
        )
    combined.triangles = o3d.utility.Vector3iVector(np.asarray(combined.triangles)[1:])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        input_ply = tmp_path / "synthetic_input.ply"
        out_dir = tmp_path / "cleaned_output"
        o3d.io.write_triangle_mesh(str(input_ply), combined)

        # Run floater removal with lowered triangle count for the test
        summary = clean_mesh_floaters(
            input_mesh_path=input_ply,
            output_dir=out_dir,
            min_triangles=50,
            min_area=0.05,
            min_thickness=0.04,
            min_solidity=0.40,
            max_boundary_openness=0.05,
            enable_sor=False,
        )

        assert (out_dir / "mesh_cleaned.ply").is_file()
        assert (out_dir / "mesh_sealed_colored.ply").is_file()
        assert (out_dir / "mesh_sealed_red.ply").is_file()
        assert (out_dir / "mesh_floater_removal_stats.json").is_file()
        raw = o3d.io.read_triangle_mesh(str(out_dir / "mesh_cleaned.ply"))
        sealed = o3d.io.read_triangle_mesh(str(out_dir / "mesh_sealed_colored.ply"))
        red = o3d.io.read_triangle_mesh(str(out_dir / "mesh_sealed_red.ply"))
        assert not raw.is_edge_manifold(allow_boundary_edges=False)
        assert sealed.is_edge_manifold(allow_boundary_edges=False)
        assert len(red.triangles) == len(sealed.triangles)
        assert np.any(np.all(np.asarray(red.vertex_colors) == [1, 0, 0], axis=1))

        # Kept components should be exactly 2: primary sphere + solid cylinder
        assert summary["kept_components_count"] == 2
        assert summary["pruned_components_count"] == 2  # sheet and crumb pruned
        assert summary["clean_triangles"] > 0
        assert summary["removed_triangles"] > 0

        # Verify timings are present and non-negative
        timings = summary["timings_breakdown"]
        assert "total_pipeline_seconds" in timings
        assert timings["total_pipeline_seconds"] >= 0.0
