"""Geometry helpers for object placement decisions."""

import numpy as np
import open3d as o3d

MIN_ELONGATED_EDGE_M = 0.30
MIN_ELONGATED_ASPECT_RATIO = 20.0


def fitting_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Drop long, thin faces from placement evidence while preserving source meshes."""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    if not len(faces):
        return o3d.geometry.TriangleMesh()

    triangles = vertices[faces]
    edge_lengths = np.stack((
        np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
        np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
        np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
    ))
    longest = edge_lengths.max(axis=0)
    area = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                         triangles[:, 2] - triangles[:, 0]), axis=1)
    # shortcut: instance PLYs omit fill flags; use source-face provenance when exports retain it.
    aspect = np.divide(longest ** 2, 2.0 * area, out=np.full(len(area), np.inf), where=area > 1e-12)
    keep = ~((longest >= MIN_ELONGATED_EDGE_M) & (aspect >= MIN_ELONGATED_ASPECT_RATIO))
    if not keep.any():
        return o3d.geometry.TriangleMesh()

    selected = faces[keep]
    used, inverse = np.unique(selected, return_inverse=True)
    result = o3d.geometry.TriangleMesh()
    result.vertices = o3d.utility.Vector3dVector(vertices[used])
    result.triangles = o3d.utility.Vector3iVector(inverse.reshape(-1, 3))
    return result
