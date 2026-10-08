import sys
from pathlib import Path
import numpy as np
import pytest
import tempfile

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(backend_dir / "06b_semantic_segmentation_pipeline"))

from Utilities.pipeline_paths import bootstrap
bootstrap()

from geometry_graph import (
    GeometryGraph,
    build_geometry_graph_from_superpoints,
)


def test_geometry_graph_clustering_separates_concave_and_crease():
    """Test that concave contact seams and orthogonal creases prevent merging."""
    # Synthetic setup:
    # 4 superpoints:
    # SP 0 and SP 1: on the floor, co-planar, normal = [0, 1, 0], no crease
    # SP 2: table leg touching floor at SP 1 (concave seam = True, normal = [1, 0, 0])
    # SP 3: table top above SP 2 (orthogonal crease 90 deg = True, normal = [0, 1, 0])
    num_sp = 4
    num_faces = 40
    face_to_sp = np.repeat(np.arange(num_sp), 10).astype(np.int32)

    centroids = np.array([
        [0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.2, 0.3, 0.0],
        [0.2, 0.6, 0.0],
    ], dtype=np.float32)

    normals = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)

    colors = np.array([
        [0.8, 0.6, 0.4],
        [0.8, 0.6, 0.4],
        [0.1, 0.1, 0.1],
        [0.9, 0.9, 0.9],
    ], dtype=np.float32)

    # 3 edges: (0, 1) floor, (1, 2) floor-to-leg (concave), (2, 3) leg-to-top (90 deg crease)
    adj_pairs = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
    ], dtype=np.int32)

    adj_concave = np.array([False, True, False], dtype=bool)
    adj_dihedral = np.array([0.05, 1.57, 1.57], dtype=np.float32)
    adj_boundary_length = np.array([0.1, 0.05, 0.05], dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        npz_path = Path(tmp_dir) / "superpoints.npz"
        np.savez_compressed(
            npz_path,
            face_to_superpoint=face_to_sp,
            centroids=centroids,
            normals=normals,
            colors=colors,
            adj_pairs=adj_pairs,
            adj_concave=adj_concave,
            adj_dihedral=adj_dihedral,
            adj_boundary_length=adj_boundary_length,
            num_superpoints=num_sp,
        )

        graph = build_geometry_graph_from_superpoints(
            superpoints_npz=npz_path,
            tau_cut=0.40,
            cut_concave=True,
            min_superpoints_per_cluster=1,
        )

        # SP 0 and SP 1 must merge (same floor)
        assert graph.superpoint_clusters[0] == graph.superpoint_clusters[1]

        # SP 2 (leg) must NOT merge with floor (separated by concave seam)
        assert graph.superpoint_clusters[1] != graph.superpoint_clusters[2]

        # SP 3 (top) must NOT merge with leg (separated by 90-deg crease)
        assert graph.superpoint_clusters[2] != graph.superpoint_clusters[3]

        # Total clusters should be 3: [Floor (0,1), Leg (2), Top (3)]
        assert graph.num_clusters == 3
