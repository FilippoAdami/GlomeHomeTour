"""Resolve Phase 06 shell surfaces by exported role files or mesh geometry."""

from pathlib import Path

import numpy as np
import open3d as o3d


def submesh(vertices, faces, mask):
    selected = faces[mask]
    mesh = o3d.geometry.TriangleMesh()
    if len(selected):
        used, inverse = np.unique(selected, return_inverse=True)
        mesh.vertices = o3d.utility.Vector3dVector(vertices[used])
        mesh.triangles = o3d.utility.Vector3iVector(inverse.reshape(-1, 3))
    return mesh


def _combine(paths):
    result = o3d.geometry.TriangleMesh()
    for path in paths:
        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.triangles):
            result += mesh
    return result


def _from_shell_geometry(shell):
    shell.compute_triangle_normals()
    vertices = np.asarray(shell.vertices)
    faces = np.asarray(shell.triangles)
    normals = np.asarray(shell.triangle_normals)
    horizontal = np.abs(normals[:, 1]) >= 0.65
    centers_y = vertices[faces].mean(axis=1)[:, 1]
    positive, negative = normals[:, 1] >= 0.65, normals[:, 1] <= -0.65
    if positive.any() and negative.any():
        floor_mask, ceiling_mask = positive, negative
    elif horizontal.any():
        # Some Phase 06 PLY exports give ceilings the same winding as floors.
        top = centers_y[horizontal].max()
        tolerance = max(0.01, (top - centers_y[horizontal].min()) * 0.01)
        ceiling_mask = horizontal & (centers_y >= top - tolerance)
        floor_mask = horizontal & ~ceiling_mask
    else:
        floor_mask = ceiling_mask = np.zeros(len(faces), dtype=bool)
    return {"floor": submesh(vertices, faces, floor_mask),
            "ceiling": submesh(vertices, faces, ceiling_mask),
            "wall": submesh(vertices, faces, ~horizontal)}


def load_role_meshes(shell, shell_path=None):
    """Use Phase 06 role exports when available, with a geometric fallback."""
    fallback = _from_shell_geometry(shell)
    if shell_path is None:
        return fallback

    elements = Path(shell_path).parent / "elements"
    role_paths = {
        "floor": sorted(elements.glob("floor_level_*.ply")) + sorted(elements.glob("slab_level_*.ply")),
        "ceiling": sorted(elements.glob("ceiling_group_*.ply")),
        "wall": ([elements / "walls.ply"] if (elements / "walls.ply").is_file()
                 else sorted(elements.glob("wall_*.ply"))),
    }
    return {role: _combine(paths) if paths else fallback[role]
            for role, paths in role_paths.items()}
