"""Production regression cases for semantic ownership, uncertainty and physical IDs."""
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '05_semantic_segmentation'))
from geometric_regularization import run_complete_geometric_regularization
from detect_structural_classes import extract_instances


def test_observed_ceiling_beam_sides_and_unknown_geometry_keep_their_ownership():
    vertices = np.array([[0, 2, 0], [1, 2, 0], [0, 2, 1], [1, 2, 1],
                         [0, 1.8, 1], [1, 1.8, 1], [1, 0, 1]], dtype=float)
    triangles = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4], [4, 5, 6]])
    raw = np.array([41, 41, 41, 97, -1])
    evidence = sparse.csr_matrix([[20, 0], [20, 0], [20, 0], [0, 20], [0, 0]])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                    o3d.utility.Vector3iVector(triangles))
    result, metrics = run_complete_geometric_regularization(vertices, triangles,
        np.full_like(vertices, .5), raw, mesh, floor_cid=0, ceiling_cid=41, wall_cid=97,
        priors=evidence, source_ids=np.array([41, 97]))
    np.testing.assert_array_equal(result, raw)
    assert metrics['unknown_faces'] == 1
    assert metrics['smoothing']['converged']
    np.testing.assert_array_equal(raw, [41, 41, 41, 97, -1])


def test_corroboration_cannot_bypass_hard_triangle_minimum():
    labels = np.array([5] * 3 + [5] * 3)
    u, v = np.array([0, 1, 3, 4]), np.array([1, 2, 4, 5])
    centers = np.array([[0, 0, 0], [.01, 0, 0], [.02, 0, 0],
                        [1, 0, 0], [1.01, 0, 0], [1.02, 0, 0]])
    instances, counts = extract_instances(labels, [5], u, v, np.full(6, .001), centers,
        np.tile([0, 0, 1], (6, 1)), np.array([0, 1, 0]), min_instance_size=350,
        max_proximity_merge_dist=0, supported_faces=np.ones(6, bool))
    assert counts[5] == 0
    assert instances == {}


def test_support_for_a_different_source_cannot_certify_a_reassigned_face():
    from mesh_prior_refinement import corroborated_faces
    votes = sparse.csr_matrix([[0, 10], [10, 0], [2, 1], [0, 0]])
    supported = corroborated_faces(np.array([17, 17, 17, -1]), votes, np.array([17, 91]))
    np.testing.assert_array_equal(supported, [False, True, False, False])


def test_missing_structural_role_evidence_never_invents_an_opaque_id():
    from geometric_regularization import detect_primary_structural_cids
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    triangles = np.array([[0, 1, 2], [1, 3, 2]])
    normals = np.tile([1., 0., 0.], (2, 1))
    centers = vertices[triangles].mean(1)
    areas = np.ones(2)
    first = detect_primary_structural_cids(vertices, triangles, np.array([17, 91]), normals, centers, areas)
    renamed = detect_primary_structural_cids(vertices, triangles, np.array([901, 2]), normals, centers, areas)
    assert first[0] == renamed[0] == 0
    assert first[2] == renamed[2] == 0
    assert first[1] == 17 and renamed[1] == 901


def test_production_regularization_is_equivariant_to_upstream_world_scale():
    vertices = np.array([[0, 2, 0], [1, 2, 0], [0, 2, 1], [1, 2, 1],
                         [0, 1.8, 1], [1, 1.8, 1]], dtype=float)
    triangles = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4]])
    labels = np.array([41, 41, 41, 97])
    priors = sparse.csr_matrix([[20, 0], [20, 0], [20, 0], [0, 20]])
    results = []
    for scale in (1., 10.):
        xyz = vertices * scale
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(xyz),
                                        o3d.utility.Vector3iVector(triangles))
        result, metrics = run_complete_geometric_regularization(xyz, triangles,
            np.full_like(xyz, .5), labels, mesh, priors=priors, source_ids=np.array([41, 97]))
        results.append(result)
    np.testing.assert_array_equal(*results)
