import sys
from pathlib import Path
import numpy as np
import open3d as o3d
import pytest
import tempfile

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(backend_dir / "06b_semantic_segmentation_pipeline"))

from Utilities.pipeline_paths import bootstrap
bootstrap()

from mesh_superpoints import (
    SuperpointGraph,
    build_superpoints,
    compute_mesh_geometry,
    extract_internal_mesh_edges,
    detect_mesh_boundaries,
)


def test_cube_partition_preserves_orthogonal_faces():
    """A cube has 6 orthogonal faces separated by 90-degree creases.

    At crease_angle_deg = 35.0, each face must be a separate superpoint,
    and no superpoint should wrap around a 90-degree corner.
    """
    # Create a simple box with 12 triangles (2 per face)
    mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    # Subdivide so each face has multiple triangles
    mesh = mesh.subdivide_midpoint(number_of_iterations=2)
    mesh.compute_vertex_normals()

    graph = build_superpoints(
        mesh,
        crease_angle_deg=35.0,
        concavity_thresh=0.005,
        min_patch_faces=4,
        target_patch_faces=50,
    )

    # Exactly 6 orthogonal planar faces
    assert graph.num_superpoints == 6
    assert graph.num_faces == len(mesh.triangles)

    # Verify that each superpoint has a dominant orthogonal normal
    for norm in graph.normals:
        # One component should be ~1.0, others ~0.0
        assert np.max(np.abs(norm)) > 0.98

    # Verify adjacency graph
    # A cube's 6 faces form an octahedron graph with 12 edges
    assert len(graph.adj_pairs) == 12
    # All dihedral angles across adjacent faces must be ~90 degrees (pi/2 radians)
    assert np.all(graph.adj_dihedral > np.radians(80.0))


def test_superpoint_graph_save_and_export(tmp_path):
    """Test saving graph data and exporting colored mesh."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=10)
    mesh.compute_vertex_normals()

    graph = build_superpoints(
        mesh,
        crease_angle_deg=35.0,
        min_patch_faces=5,
        target_patch_faces=20,
    )

    out_dir = tmp_path / "superpoints"
    graph.save(out_dir)

    assert (out_dir / "superpoints.npz").exists()
    assert (out_dir / "superpoints_stats.json").exists()

    ply_path = out_dir / "test_colored.ply"
    graph.export_colored_mesh(mesh, ply_path)
    assert ply_path.exists()
    assert ply_path.stat().st_size > 0
