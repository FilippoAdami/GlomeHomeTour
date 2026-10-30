import numpy as np
import pytest
import open3d as o3d
import scipy.sparse as sp

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "05_semantic_segmentation"))

from geometric_regularization import (
    eliminate_horizontal_wall_triangles,
    absorb_flat_coplanar_wall_patches,
    regularize_foreground_connected_components,
)
from detect_structural_classes import build_mesh_dual_graph, absorb_enclosed_patches


def test_eliminate_horizontal_wall_triangles():
    """Verifies that wall class has strictly zero horizontal triangles after elimination."""
    vertices = np.array([
        # Tri 0: vertical wall in XY plane (normal = [0, 0, 1])
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        # Tri 1: horizontal upward in XZ plane (normal = [0, 1, 0])
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
    ], dtype=np.float64)
    triangles = np.array([
        [0, 1, 2],
        [2, 3, 4],
    ], dtype=np.int32)

    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    tri_normals = cross / np.linalg.norm(cross, axis=1, keepdims=True)
    tri_centroids = (v0 + v1 + v2) / 3.0

    adj_u, adj_v = build_mesh_dual_graph(triangles)
    labels = np.array([5, 5], dtype=np.int32)  # Both labeled Wall (5)
    gravity_up = np.array([0.0, 1.0, 0.0])

    cleaned, n_re = eliminate_horizontal_wall_triangles(
        labels=labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_normals=tri_normals,
        tri_centroids=tri_centroids,
        wall_cid=5,
        floor_cid=2,
        ceiling_cid=1,
        gravity_up=gravity_up,
        y_ground=0.0,
        max_tilt_cos=0.45,
    )

    assert n_re >= 1
    # Wall must have 0 horizontal triangles
    horiz_wall = (cleaned == 5) & (np.abs(tri_normals @ gravity_up) > 0.45)
    assert np.sum(horiz_wall) == 0
    assert cleaned[0] == 5  # Vertical remains wall
    assert cleaned[1] != 5  # Horizontal reassigned away from wall


def test_shelves_do_not_swallow_resting_objects():
    """Verifies that distinct objects on shelves are preserved and not merged into shelf."""
    labels = np.array([227] * 1000 + [275] * 500, dtype=np.int32)  # Shelf (227) and Figurine (275)
    adj_u = np.arange(len(labels) - 1, dtype=np.int32)
    adj_v = np.arange(1, len(labels), dtype=np.int32)
    tri_centroids = np.zeros((len(labels), 3), dtype=np.float64)
    tri_centroids[:1000, 0] = np.linspace(0.0, 1.0, 1000)
    tri_centroids[1000:, 0] = np.linspace(0.2, 0.6, 500)
    tri_centroids[1000:, 1] = 0.5

    cleaned, n_re = regularize_foreground_connected_components(
        tri_centroids=tri_centroids,
        labels=labels,
        adj_u=adj_u,
        adj_v=adj_v,
        structural_cids={1, 2, 5},
    )

    assert np.sum(cleaned == 275) == 500
    assert n_re == 0
