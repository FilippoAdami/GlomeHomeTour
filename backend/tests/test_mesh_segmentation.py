"""Unit tests for geometry-constrained multi-view self-supervised mesh segmentation."""

import sys
import numpy as np
import open3d as o3d
import pytest
from pathlib import Path

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(backend_dir / "06_semantic_segmentation"))

from super_surfels import build_super_surfels, compute_mesh_geometry, extract_internal_edges, detect_boundaries
from dual_graph_ncut import build_dual_affinity_matrix, solve_spectral_ncut_connected
from export_deliverables import export_segmentation_artifacts


def create_l_junction_mesh():
    """Create a synthetic L-junction mesh (horizontal plane connected to vertical plane).

    Winding order is oriented into the room:
    - Floor normal pointing +Y (up)
    - Wall normal pointing +Z (into room)
    The contact seam at y=0, z=0 is a concave junction.
    """
    vertices = np.array([
        # Floor (z in [0, 1], y=0, x in [0, 1])
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 0.0, 1.0],  # 2
        [0.0, 0.0, 1.0],  # 3
        # Wall (x in [0, 1], z=0, y in [0, 1])
        [0.0, 1.0, 0.0],  # 4
        [1.0, 1.0, 0.0],  # 5
    ], dtype=np.float32)

    triangles = np.array([
        # Floor triangles (normal pointing +Y)
        [0, 2, 1],
        [0, 3, 2],
        # Wall triangles (normal pointing +Z)
        [0, 1, 5],
        [0, 5, 4],
    ], dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh


def test_dihedral_and_concavity_detection():
    """Verify that 90-degree internal junction is correctly detected as crease and concave."""
    mesh = create_l_junction_mesh()
    triangles, vertices, face_normals, areas, face_centroids = compute_mesh_geometry(mesh)
    num_faces = len(triangles)

    t1, t2, opp1, opp2, shared_v0 = extract_internal_edges(triangles, num_faces)
    dihedral_rad, is_crease, is_concave = detect_boundaries(
        vertices, face_normals, t1, t2, opp2, shared_v0, opp1=opp1,
        crease_angle_deg=35.0, concavity_thresh=0.001
    )

    # 90-degree junction should be marked crease and concave
    assert np.any(is_crease), "90-degree junction must be detected as crease"
    assert np.any(is_concave), "Floor-to-wall corner must be detected as concave seam"


def test_super_surfel_boundary_integrity():
    """Verify that super-surfel formation does not bridge across concave junction seams."""
    mesh = create_l_junction_mesh()
    surfel_data = build_super_surfels(
        mesh,
        crease_angle_deg=35.0,
        concavity_thresh=0.001,
        min_patch_faces=1,
    )

    assert surfel_data.num_surfels >= 2, "Floor and wall must be partitioned into separate super-surfels"
    assert np.any(surfel_data.adj_concave), "Adjacency between floor and wall must be marked concave"


def test_spectral_ncut_partitioning():
    """Verify spectral NCut with guaranteed 3D spatial contiguity on synthetic modular graph."""
    num_nodes = 8
    adj_pairs = np.array([
        [0, 1], [1, 2], [2, 3],
        [4, 5], [5, 6], [6, 7],
        [3, 4]  # Cross-component seam
    ], dtype=np.int32)
    adj_concave = np.array([False, False, False, False, False, False, True], dtype=bool)

    normals = np.zeros((num_nodes, 3), dtype=np.float32)
    normals[:4] = [0.0, 1.0, 0.0]   # Component A: floor
    normals[4:] = [0.0, 0.0, 1.0]   # Component B: wall

    features = np.zeros((num_nodes, 768), dtype=np.float32)
    features[:4, 0] = 1.0 # Material A
    features[4:, 1] = 1.0 # Material B
    areas = np.ones(num_nodes, dtype=np.float32)

    W = build_dual_affinity_matrix(
        num_surfels=num_nodes,
        adj_pairs=adj_pairs,
        adj_concave=adj_concave,
        normals=normals,
        features=features,
        concave_penalty=0.001,
    )

    labels, stats = solve_spectral_ncut_connected(
        W,
        adj_pairs=adj_pairs,
        areas=areas,
        num_clusters=2,
        min_object_area=0.1,
    )
    assert len(labels) == num_nodes
    # Component A nodes must share one label, Component B must share the other
    assert labels[0] == labels[1] == labels[2] == labels[3]
    assert labels[4] == labels[5] == labels[6] == labels[7]
    assert labels[0] != labels[4], "Components A and B must receive different cluster labels"


def test_export_deliverables(tmp_path):
    """Verify export creates valid PLY files with normals and manifest JSON."""
    mesh = create_l_junction_mesh()
    surfel_data = build_super_surfels(mesh, min_patch_faces=1)

    labels = np.arange(surfel_data.num_surfels, dtype=np.int32)

    manifest = export_segmentation_artifacts(
        mesh,
        face_to_surfel=surfel_data.face_to_surfel,
        surfel_labels=labels,
        output_dir=tmp_path,
    )

    assert (tmp_path / "mesh_segmented.ply").is_file()
    assert (tmp_path / "mesh_segmented_colored.ply").is_file()
    assert (tmp_path / "segmentation_manifest.json").is_file()

    # Check object files
    objects = manifest["objects"]
    assert len(objects) == surfel_data.num_surfels
    for obj in objects:
        obj_file = tmp_path / obj["file_name"]
        assert obj_file.is_file(), f"Object PLY missing: {obj_file}"
        sub_mesh = o3d.io.read_triangle_mesh(str(obj_file))
        assert sub_mesh.has_vertex_normals()
        assert len(sub_mesh.triangles) > 0
