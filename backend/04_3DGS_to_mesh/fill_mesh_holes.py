#!/usr/bin/env python3
"""Inspect mesh boundary loops and fill selected small, nearly planar acquisition holes.

Regions JSON: [{"name":"wall patch", "min":[x,y,z], "max":[x,y,z],
                "max_diameter_m":0.4, "max_plane_deviation_m":0.02}]
Coordinates are in the input mesh's frame. Only complete loops inside a box are eligible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from shapely import constrained_delaunay_triangles
from shapely.geometry import Polygon


def export_ply_with_face_attributes(
    file_path: Path | str,
    vertices: np.ndarray,
    triangles: np.ndarray,
    vertex_colors: np.ndarray | None = None,
    face_attributes: dict[str, tuple[str, np.ndarray]] | None = None,
) -> bool:
    """Exports binary little-endian PLY mesh with vertex coordinates, vertex colors, and custom face attributes."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    num_verts = len(vertices)
    num_faces = len(triangles)

    has_colors = vertex_colors is not None and len(vertex_colors) == num_verts
    if has_colors:
        if vertex_colors.dtype == np.uint8:
            v_colors_u8 = vertex_colors
        else:
            v_colors_u8 = (np.clip(vertex_colors.astype(np.float64), 0.0, 1.0) * 255.0).round().astype(np.uint8)
    else:
        v_colors_u8 = None

    v_dtype_fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if has_colors:
        v_dtype_fields.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v_data = np.zeros(num_verts, dtype=np.dtype(v_dtype_fields))
    v_data["x"] = vertices[:, 0]
    v_data["y"] = vertices[:, 1]
    v_data["z"] = vertices[:, 2]
    if has_colors:
        v_data["red"] = v_colors_u8[:, 0]
        v_data["green"] = v_colors_u8[:, 1]
        v_data["blue"] = v_colors_u8[:, 2]

    f_dtype_fields = [("n_verts", "u1"), ("v0", "<i4"), ("v1", "<i4"), ("v2", "<i4")]
    face_prop_header_lines = []
    if face_attributes:
        for attr_name, (attr_type, attr_arr) in face_attributes.items():
            if attr_type == "uchar":
                f_dtype_fields.append((attr_name, "u1"))
                face_prop_header_lines.append(f"property uchar {attr_name}")
            elif attr_type == "int":
                f_dtype_fields.append((attr_name, "<i4"))
                face_prop_header_lines.append(f"property int {attr_name}")
            else:
                raise ValueError(f"Unsupported face attribute type: {attr_type}")

    f_data = np.zeros(num_faces, dtype=np.dtype(f_dtype_fields))
    f_data["n_verts"] = 3
    f_data["v0"] = triangles[:, 0]
    f_data["v1"] = triangles[:, 1]
    f_data["v2"] = triangles[:, 2]
    if face_attributes:
        for attr_name, (_, attr_arr) in face_attributes.items():
            f_data[attr_name] = attr_arr

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {num_verts}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_colors:
        header_lines.extend(["property uchar red", "property uchar green", "property uchar blue"])
    header_lines.extend([
        f"element face {num_faces}",
        "property list uchar int vertex_indices",
        *face_prop_header_lines,
        "end_header\n",
    ])
    header = "\n".join(header_lines)

    with open(file_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(v_data.tobytes())
        f.write(f_data.tobytes())
    return True


def read_ply_face_flags(ply_path: Path | str, property_name: str = "is_hole_fill") -> np.ndarray | None:
    """Read a scalar face property from ASCII or binary PLY; reject malformed files."""
    ply_path = Path(ply_path)
    if not ply_path.is_file():
        return None
    ply = PlyData.read(str(ply_path), known_list_len={"face": {"vertex_indices": 3}})
    if "face" not in ply or property_name not in ply["face"].data.dtype.names:
        return None
    return ply["face"][property_name].copy()


read_ply_face_property = read_ply_face_flags


def boundary_loops(triangles, vertices=None):
    """Trace simple loops, splitting boundary junctions at shared vertices."""
    directed = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    undirected = np.sort(directed, axis=1)
    _, first, inverse, counts = np.unique(undirected, axis=0, return_index=True,
                                          return_inverse=True, return_counts=True)
    edges = directed[first[counts == 1]]
    outgoing, incoming = {}, {}
    for index, (a, b) in enumerate(edges):
        outgoing.setdefault(int(a), []).append(index)
        incoming.setdefault(int(b), []).append(index)
    successor = np.full(len(edges), -1, dtype=np.int32)
    for vertex, ins in incoming.items():
        outs = outgoing.get(vertex, [])
        if len(ins) != len(outs):
            continue
        if len(ins) == 1 or vertices is None:
            pairs = zip(ins, outs)
        else:
            before = vertices[edges[ins, 1]] - vertices[edges[ins, 0]]
            after = vertices[edges[outs, 1]] - vertices[edges[outs, 0]]
            before /= np.maximum(np.linalg.norm(before, axis=1, keepdims=True), 1e-12)
            after /= np.maximum(np.linalg.norm(after, axis=1, keepdims=True), 1e-12)
            rows, cols = linear_sum_assignment(-(before @ after.T))
            pairs = ((ins[row], outs[col]) for row, col in zip(rows, cols))
        for edge_in, edge_out in pairs:
            successor[edge_in] = edge_out
    visited = np.zeros(len(edges), dtype=bool)
    loops = []
    for start in range(len(edges)):
        if visited[start]:
            continue
        chain = []
        edge = start
        while edge >= 0 and not visited[edge]:
            visited[edge] = True
            chain.append(int(edges[edge, 0]))
            edge = successor[edge]
        if edge != start:
            continue
        stack, positions = [], {}
        for vertex in chain + [chain[0]]:
            if vertex in positions:
                begin = positions[vertex]
                if len(stack) - begin >= 3:
                    loops.append(np.asarray(stack[begin:], dtype=np.int32))
                for old in stack[begin + 1:]:
                    positions.pop(old, None)
                stack = stack[:begin + 1]
            else:
                positions[vertex] = len(stack)
                stack.append(vertex)
    return loops


def loop_geometry(points):
    center = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points - center, full_matrices=False)
    projected = (points - center) @ axes[:2].T
    return projected, float(np.max(np.abs((points - center) @ axes[2]))), axes


def patch_loop(loop, vertices):
    if len(loop) == 3:
        return np.asarray([[loop[1], loop[0], loop[2]]], dtype=np.int32)
    points = vertices[loop]
    projected, _, _ = loop_geometry(points)
    polygon = Polygon(projected)
    if not polygon.is_valid or polygon.area <= 1e-10:
        return None
    tree = cKDTree(projected)
    faces = []
    for triangle in constrained_delaunay_triangles(polygon).geoms:
        if not polygon.covers(triangle):
            continue
        coords = np.asarray(triangle.exterior.coords)[:3]
        distance, indices = tree.query(coords)
        if np.max(distance) > 1e-7 or len(set(indices)) != 3:
            return None
        face = loop[indices]
        faces.append(face)
    if not faces:
        return None
    faces = np.asarray(faces, dtype=np.int32)
    # The mesh's boundary orientation must oppose the new faces' orientation.
    first, second = int(loop[0]), int(loop[1])
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    if np.any(np.all(directed == [first, second], axis=1)):
        faces = faces[:, [0, 2, 1]]
    if len(faces) != len(loop) - 2:
        return None
    return faces


def edge_keys(triangles):
    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])).astype(np.uint64)
    return (np.minimum(edges[:, 0], edges[:, 1]) << 32) | np.maximum(edges[:, 0], edges[:, 1])


class CoplanarTextureSampler:
    """Samples RGB colors from source mesh faces strictly matching local plane geometry and proximity."""

    def __init__(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        source_colors: np.ndarray | None,
        normal_threshold: float = 0.9063,  # cos(25 degrees)
        plane_tolerance: float = 0.05,     # 5 cm
    ):
        self.has_colors = source_colors is not None and len(source_colors) == len(vertices)
        if not self.has_colors or len(triangles) == 0:
            self.has_colors = False
            return

        self.normal_threshold = float(normal_threshold)
        self.plane_tolerance = float(plane_tolerance)

        if source_colors.dtype == np.uint8:
            colors = source_colors.astype(np.float64) / 255.0
        else:
            colors = source_colors.astype(np.float64)
            if colors.max() > 1.0 + 1e-4:
                colors = colors / 255.0

        p0 = vertices[triangles[:, 0]]
        p1 = vertices[triangles[:, 1]]
        p2 = vertices[triangles[:, 2]]
        cross = np.cross(p1 - p0, p2 - p0)
        lengths = np.linalg.norm(cross, axis=1)
        valid = lengths > 1e-12

        if not np.any(valid):
            self.has_colors = False
            return

        self.src_normals = cross[valid] / lengths[valid, None]
        self.src_centroids = (p0[valid] + p1[valid] + p2[valid]) / 3.0
        self.src_colors = (
            colors[triangles[valid, 0]]
            + colors[triangles[valid, 1]]
            + colors[triangles[valid, 2]]
        ) / 3.0

        self.tree = cKDTree(self.src_centroids)

    def query(
        self,
        positions: np.ndarray,
        normals: np.ndarray,
        k: int = 50,
    ) -> np.ndarray | None:
        """Query coplanar-derived colors for an array of positions (M, 3) and unit normals (M, 3)."""
        if not self.has_colors:
            return None

        positions = np.asarray(positions, dtype=np.float64)
        normals = np.asarray(normals, dtype=np.float64)
        single = (positions.ndim == 1)
        if single:
            positions = positions[None, :]
            normals = normals[None, :]

        if len(positions) == 0:
            return np.empty((0, 3), dtype=np.float64)

        norm_lens = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.where(norm_lens > 1e-12, normals / np.maximum(norm_lens, 1e-12), normals)

        M = len(positions)
        K = min(k, len(self.src_centroids))
        dists, cand_idx = self.tree.query(positions, k=K)
        if K == 1:
            cand_idx = cand_idx[:, None]
            dists = dists[:, None]

        cand_c = self.src_centroids[cand_idx]      # (M, K, 3)
        cand_n = self.src_normals[cand_idx]        # (M, K, 3)
        cand_col = self.src_colors[cand_idx]      # (M, K, 3)

        diff = cand_c - positions[:, None, :]      # (M, K, 3)
        euc_d = np.linalg.norm(diff, axis=-1)      # (M, K)
        alphas = np.sum(cand_n * normals[:, None, :], axis=-1)  # (M, K)
        r_plane = np.abs(np.sum(diff * normals[:, None, :], axis=-1))  # (M, K)

        mask_strict = (alphas >= self.normal_threshold) & (r_plane <= self.plane_tolerance)
        mask_relaxed = (alphas >= 0.866) & (r_plane <= max(0.15, self.plane_tolerance * 3))
        mask_normal = (alphas >= 0.866)
        mask_positive = (alphas > 0.0)

        inv_d = 1.0 / np.maximum(euc_d, 1e-4)

        result_colors = np.zeros((M, 3), dtype=np.float64)
        for i in range(M):
            m_s = mask_strict[i]
            if np.any(m_s):
                w = inv_d[i, m_s]
                result_colors[i] = np.sum(w[:, None] * cand_col[i, m_s], axis=0) / np.sum(w)
                continue

            m_r = mask_relaxed[i]
            if np.any(m_r):
                w = inv_d[i, m_r]
                result_colors[i] = np.sum(w[:, None] * cand_col[i, m_r], axis=0) / np.sum(w)
                continue

            m_n = mask_normal[i]
            if np.any(m_n):
                w = inv_d[i, m_n]
                result_colors[i] = np.sum(w[:, None] * cand_col[i, m_n], axis=0) / np.sum(w)
                continue

            m_p = mask_positive[i]
            if np.any(m_p):
                w = (alphas[i, m_p] ** 2) * inv_d[i, m_p]
                result_colors[i] = np.sum(w[:, None] * cand_col[i, m_p], axis=0) / np.sum(w)
                continue

            result_colors[i] = cand_col[i, 0]

        return result_colors[0] if single else result_colors


def derive_infilled_face_textures(
    mesh: o3d.geometry.TriangleMesh,
    source_triangle_count: int,
    normal_threshold: float = 0.9063,
    plane_tolerance: float = 0.05,
) -> np.ndarray | None:
    """Derives coplanar RGB textures for all infilled triangles (triangles[source_triangle_count:])."""
    if not mesh.has_vertex_colors():
        return None
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(triangles) <= source_triangle_count:
        return np.empty((0, 3), dtype=np.float64)

    orig_triangles = triangles[:source_triangle_count]
    infill_triangles = triangles[source_triangle_count:]
    source_colors = np.asarray(mesh.vertex_colors)

    sampler = CoplanarTextureSampler(
        vertices, orig_triangles, source_colors,
        normal_threshold=normal_threshold,
        plane_tolerance=plane_tolerance,
    )
    p0 = vertices[infill_triangles[:, 0]]
    p1 = vertices[infill_triangles[:, 1]]
    p2 = vertices[infill_triangles[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    lengths = np.linalg.norm(cross, axis=1)
    normals = np.where(lengths[:, None] > 1e-12, cross / np.maximum(lengths[:, None], 1e-12), np.array([0.0, 0.0, 1.0]))
    centroids = (p0 + p1 + p2) / 3.0
    return sampler.query(centroids, normals)


def fill_mesh(mesh, regions, auto=False, seal_all=False, normal_threshold=0.9063, plane_tolerance=0.05):
    vertices = np.asarray(mesh.vertices).copy() if seal_all else np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if not len(triangles):
        raise ValueError("Input mesh has no triangles")
    if auto and not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals) if auto else None
    loops = boundary_loops(triangles, vertices)
    if seal_all:
        for loop in loops:
            points = vertices[loop]
            zero_idx = np.flatnonzero(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1) < 1e-12)
            for k, i in enumerate(zero_idx):
                vertex = int(loop[(i + 1) % len(loop)])
                radial = vertices[vertex] - points.mean(axis=0)
                if np.linalg.norm(radial) < 1e-12:
                    radial = np.array([1.0, 0.0, 0.0])
                axis = np.eye(3)[np.argmin(np.abs(radial))]
                direction = np.cross(radial, axis)
                # A zero-length source edge needs a tiny separation to support a real cap face.
                vertices[vertex] += direction / np.linalg.norm(direction) * ((k + 1) * 1e-5)
    existing_edges = np.unique(edge_keys(triangles)) if auto or seal_all or regions else np.empty(0, dtype=np.uint64)
    added_internal_edges = set()
    extra_vertices = []
    extra_colors = []
    source_colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    sampler = CoplanarTextureSampler(
        vertices, triangles, source_colors,
        normal_threshold=normal_threshold,
        plane_tolerance=plane_tolerance,
    )
    records = []
    patches = []

    def add_patch(patch, loop, geometry):
        local_vertices = []
        local_colors = []
        offset = len(vertices) + len(extra_vertices)
        normal = geometry[2][2]

        def point(index):
            return vertices[index] if index < len(vertices) else (extra_vertices + local_vertices)[index - len(vertices)]

        def color(index):
            return source_colors[index] if index < len(vertices) else (extra_colors + local_colors)[index - len(vertices)]

        if patch is None:
            if not seal_all and abs(Polygon(geometry[0]).area) <= 1e-6:
                return None
            center = vertices[loop].mean(axis=0) - normal * 0.005
            local_vertices.append(center)
            if source_colors is not None:
                aligned_normal = -normal if (normals is not None and np.mean(normals[loop] @ normal) < 0) else normal
                local_colors.append(sampler.query(center, aligned_normal))
            patch = np.column_stack((np.roll(loop, -1), loop, np.full(len(loop), offset, dtype=np.int32)))
            method = "fan_fallback"
        else:
            method = "triangulated"

        faces = patch.tolist()
        internal = np.empty(0, dtype=np.uint64)
        if len(faces) > 1:
            while True:
                keys, counts = np.unique(edge_keys(np.asarray(faces, dtype=np.int32)), return_counts=True)
                if np.any(counts > 2):
                    return None
                internal = keys[counts == 2]
                indices = np.searchsorted(existing_edges, internal)
                valid = indices < len(existing_edges)
                conflicts = set(map(int, internal[valid][existing_edges[indices[valid]] == internal[valid]]))
                conflicts.update(int(key) for key in internal if int(key) in added_internal_edges)
                if not conflicts:
                    break
                method = "split_edges"
                key = conflicts.pop()
                a, b = key >> 32, key & 0xffffffff
                new_id = offset + len(local_vertices)
                pos = (point(a) + point(b)) / 2 - normal * min(0.01, np.linalg.norm(point(a) - point(b)) * 0.02)
                local_vertices.append(pos)
                if source_colors is not None:
                    aligned_normal = -normal if (normals is not None and np.mean(normals[loop] @ normal) < 0) else normal
                    local_colors.append(sampler.query(pos, aligned_normal))
                split_faces = []
                for face in faces:
                    if a in face and b in face:
                        i = next(i for i in range(3) if {face[i], face[(i + 1) % 3]} == {a, b})
                        u, v, w = face[i], face[(i + 1) % 3], face[(i + 2) % 3]
                        split_faces.extend(([u, new_id, w], [new_id, v, w]))
                    else:
                        split_faces.append(face)
                faces = split_faces

        face_points = (np.asarray([[point(index) for index in face] for face in faces])
                       if local_vertices else vertices[np.asarray(faces, dtype=np.int32)])
        areas = np.linalg.norm(np.cross(face_points[:, 1] - face_points[:, 0],
                                        face_points[:, 2] - face_points[:, 0]), axis=1) / 2
        if np.any(areas < 1e-12):
            return None
        extra_vertices.extend(local_vertices)
        extra_colors.extend(local_colors)
        added_internal_edges.update(map(int, internal))
        patches.append(np.asarray(faces, dtype=np.int32))
        return method

    for loop in loops:
        points = vertices[loop]
        bounds_min, bounds_max = points.min(axis=0), points.max(axis=0)
        diameter = float(np.linalg.norm(bounds_max - bounds_min))
        geometry = loop_geometry(points) if seal_all or len(loop) <= 2000 else None
        deviation = geometry[1] if geometry else None
        record = {"id": len(records), "edges": len(loop), "min": bounds_min.tolist(),
                  "max": bounds_max.tolist(), "diameter_m": diameter,
                  "plane_deviation_m": deviation, "filled_region": None}
        if seal_all:
            method = add_patch(patch_loop(loop, vertices) if len(loop) <= 2000 else None,
                               loop, geometry)
            if not method:
                method = add_patch(None, loop, geometry)
            if method:
                record["filled_region"] = "seal_all"
                record["patch_method"] = method
        elif auto and geometry is not None:
            alignment = float(np.mean(np.abs(normals[loop] @ geometry[2][2]) >= 0.866))
            envelope = Polygon(geometry[0]).minimum_rotated_rectangle
            if envelope.geom_type == "Polygon":
                corners = np.asarray(envelope.exterior.coords)[:3]
                planar_extent = np.linalg.norm(np.diff(corners, axis=0), axis=1)
                record["planar_extent_m"] = sorted(map(float, planar_extent))
            else:
                planar_extent = [float("inf")]
            record["rim_alignment"] = alignment
            short, long = sorted(planar_extent)
            minor = long <= 0.10 and deviation <= 0.02
            medium = 0.10 < long <= 0.50 + 1e-9 and deviation <= 0.10
            narrow = 0.50 < long <= 2.00 and short <= 0.30 and deviation <= 0.10
            # shortcut: ambiguous large openings still need view-based evidence.
            if minor or medium or narrow:
                method = add_patch(patch_loop(loop, vertices), loop, geometry)
                if method:
                    record["filled_region"] = "auto"
                    record["patch_method"] = method
        for region in regions:
            if (deviation is None or diameter > region["max_diameter_m"] or
                    deviation > region["max_plane_deviation_m"] or
                    np.any(bounds_min < region["min"]) or np.any(bounds_max > region["max"])):
                continue
            method = add_patch(patch_loop(loop, vertices), loop, geometry)
            if method:
                record["filled_region"] = region["name"]
                record["patch_method"] = method
            break
        records.append(record)
    if patches:
        result = o3d.geometry.TriangleMesh(mesh)
        working_vertices = np.vstack((vertices, extra_vertices)) if extra_vertices else vertices
        patch_faces = np.vstack(patches)
        cross = np.cross(working_vertices[patch_faces[:, 1]] - working_vertices[patch_faces[:, 0]],
                         working_vertices[patch_faces[:, 2]] - working_vertices[patch_faces[:, 0]])
        lengths = np.linalg.norm(cross, axis=1)
        regular = lengths >= 6e-12
        faces_to_recess = patch_faces[regular]
        centers = working_vertices[faces_to_recess].mean(axis=1)
        if len(faces_to_recess):
            edges = working_vertices[faces_to_recess[:, [1, 2, 0]]] - working_vertices[faces_to_recess]
            perimeters = np.linalg.norm(edges, axis=2).sum(axis=1)
            # Recess each triangle center by at most 15 mm; leave tiny triangles unsplit.
            depth = np.minimum(0.015, np.minimum(perimeters * 0.01, lengths[regular] / np.maximum(perimeters, 1e-10)))
            centers -= cross[regular] / lengths[regular, None] * depth[:, None]
            center_ids = np.arange(len(working_vertices), len(working_vertices) + len(centers), dtype=np.int32)
            a, b, c = faces_to_recess.T
            recessed_faces = np.stack((np.column_stack((a, b, center_ids)),
                                       np.column_stack((b, c, center_ids)),
                                       np.column_stack((c, a, center_ids))), axis=1).reshape(-1, 3)
            new_faces = np.vstack((patch_faces[~regular], recessed_faces))
        else:
            new_faces = patch_faces
        result.vertices = o3d.utility.Vector3dVector(np.vstack((working_vertices, centers)))
        result.triangles = o3d.utility.Vector3iVector(np.vstack((triangles, new_faces)))
        if source_colors is not None:
            colors = np.vstack((source_colors, extra_colors)) if extra_colors else source_colors
            if len(faces_to_recess):
                recess_normals = cross[regular] / lengths[regular, None]
                unrecessed_centers = working_vertices[faces_to_recess].mean(axis=1)
                recess_colors = sampler.query(unrecessed_centers, recess_normals)
                result.vertex_colors = o3d.utility.Vector3dVector(np.vstack((colors, recess_colors)))
                recess_face_colors = np.repeat(recess_colors, 3, axis=0)
            else:
                recess_colors = np.empty((0, 3), dtype=np.float64)
                recess_face_colors = np.empty((0, 3), dtype=np.float64)
                result.vertex_colors = o3d.utility.Vector3dVector(colors)

            num_irreg = np.count_nonzero(~regular)
            if num_irreg > 0:
                irreg_faces = patch_faces[~regular]
                p0_irr = working_vertices[irreg_faces[:, 0]]
                p1_irr = working_vertices[irreg_faces[:, 1]]
                p2_irr = working_vertices[irreg_faces[:, 2]]
                irr_cross = np.cross(p1_irr - p0_irr, p2_irr - p0_irr)
                irr_lens = np.linalg.norm(irr_cross, axis=1)
                irr_normals = np.where(irr_lens[:, None] > 1e-12, irr_cross / np.maximum(irr_lens[:, None], 1e-12), np.array([0.0, 0.0, 1.0]))
                irr_centers = (p0_irr + p1_irr + p2_irr) / 3.0
                irreg_colors = sampler.query(irr_centers, irr_normals)
                infill_face_colors = np.vstack((irreg_colors, recess_face_colors))
        result.compute_vertex_normals()
    else:
        result = mesh
    return result, records


def red_patches(mesh, source_triangle_count):
    red = o3d.geometry.TriangleMesh(mesh)
    colors = np.asarray(red.vertex_colors).copy() if red.has_vertex_colors() else np.full((len(red.vertices), 3), 0.7)
    patch_vertex_ids = np.unique(np.asarray(red.triangles)[source_triangle_count:])
    colors[patch_vertex_ids] = [1.0, 0.0, 0.0]
    red.vertex_colors = o3d.utility.Vector3dVector(colors)
    return red


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--regions", type=Path, help="JSON list of regions to fill")
    parser.add_argument("--auto", action="store_true", help="Fill small planar holes with aligned rim normals")
    parser.add_argument("--seal-all", action="store_true", help="Cap every boundary loop, including large openings")
    parser.add_argument("--output", type=Path, help="Output PLY; required with --regions, --auto or --seal-all")
    parser.add_argument("--red-output", type=Path, help="Second full mesh with bright red patch interiors")
    args = parser.parse_args(argv)
    if sum((bool(args.regions), args.auto, args.seal_all)) > 1:
        parser.error("choose --regions, --auto or --seal-all")
    if bool(args.regions or args.auto or args.seal_all) != bool(args.output):
        parser.error("--output is required with --regions, --auto or --seal-all")
    if args.red_output and not args.output:
        parser.error("--red-output requires --output")
    if args.output and args.input.resolve() == args.output.resolve():
        parser.error("--output must differ from --input")
    if args.red_output and args.red_output.resolve() in {args.input.resolve(), args.output.resolve(), args.report.resolve()}:
        parser.error("--red-output must differ from input, output and report")
    if args.report.resolve() in {args.input.resolve(), args.output.resolve() if args.output else None}:
        parser.error("--report must differ from mesh paths")
    regions = json.loads(args.regions.read_text()) if args.regions else []
    if not isinstance(regions, list):
        parser.error("regions must be a JSON list")
    for region in regions:
        if (not isinstance(region, dict) or not isinstance(region.get("name"), str) or
                not all(k in region for k in ("min", "max", "max_diameter_m", "max_plane_deviation_m"))):
            parser.error("each region needs name, min, max, max_diameter_m and max_plane_deviation_m")
        try:
            lo, hi = np.asarray(region["min"], dtype=float), np.asarray(region["max"], dtype=float)
            diameter = float(region["max_diameter_m"])
            deviation = float(region["max_plane_deviation_m"])
        except (TypeError, ValueError):
            parser.error("region coordinates and limits must be numbers")
        if (lo.shape != (3,) or hi.shape != (3,) or not np.isfinite(lo).all() or
                not np.isfinite(hi).all() or np.any(lo >= hi) or
                not np.isfinite(diameter) or diameter <= 0 or
                not np.isfinite(deviation) or deviation < 0):
            parser.error("region coordinates and limits must be finite; min < max; diameter > 0; deviation >= 0")
        region["min"], region["max"] = lo, hi
        region["max_diameter_m"], region["max_plane_deviation_m"] = diameter, deviation
    mesh = o3d.io.read_triangle_mesh(str(args.input))
    result, records = fill_mesh(mesh, regions, auto=args.auto, seal_all=args.seal_all)
    remaining = boundary_loops(np.asarray(result.triangles), np.asarray(result.vertices)) if args.seal_all else []
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"input": str(args.input), "mode": "seal_all" if args.seal_all else "auto" if args.auto else "regions" if regions else "inspect",
                                       "boundary_loops": len(records),
                                       "filled_loops": sum(r["filled_region"] is not None for r in records),
                                       "remaining_boundary_loops": len(remaining) if args.seal_all else None,
                                       "remaining_boundary_edges": sum(map(len, remaining)) if args.seal_all else None,
                                       "loops": records}, indent=2) + "\n")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        res_v = np.asarray(result.vertices)
        res_t = np.asarray(result.triangles)
        res_c = np.asarray(result.vertex_colors) if result.has_vertex_colors() else None
        num_clean_t = len(mesh.triangles)
        is_hole_fill = np.zeros(len(res_t), dtype=np.uint8)
        if len(res_t) > num_clean_t:
            is_hole_fill[num_clean_t:] = 1
        face_attributes = {"is_hole_fill": ("uchar", is_hole_fill)}
        infill_colors = derive_infilled_face_textures(result, num_clean_t)
        if infill_colors is not None and len(infill_colors) == len(res_t) - num_clean_t and len(infill_colors) > 0:
            all_face_colors = np.zeros((len(res_t), 3), dtype=np.uint8)
            if res_c is not None and num_clean_t > 0:
                orig_fc = (res_c[res_t[:num_clean_t, 0]] + res_c[res_t[:num_clean_t, 1]] + res_c[res_t[:num_clean_t, 2]]) / 3.0
                all_face_colors[:num_clean_t] = (np.clip(orig_fc, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            infill_u8 = (np.clip(infill_colors, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            all_face_colors[num_clean_t:] = infill_u8
            face_attributes["fill_r"] = ("uchar", all_face_colors[:, 0])
            face_attributes["fill_g"] = ("uchar", all_face_colors[:, 1])
            face_attributes["fill_b"] = ("uchar", all_face_colors[:, 2])
        if not export_ply_with_face_attributes(
            args.output, res_v, res_t, res_c, face_attributes=face_attributes
        ):
            raise RuntimeError(f"Failed to write {args.output}")
    if args.red_output:
        red = red_patches(result, len(mesh.triangles))
        args.red_output.parent.mkdir(parents=True, exist_ok=True)
        red_v = np.asarray(red.vertices)
        red_t = np.asarray(red.triangles)
        red_c = np.asarray(red.vertex_colors) if red.has_vertex_colors() else None
        num_clean_t = len(mesh.triangles)
        is_hole_fill = np.zeros(len(red_t), dtype=np.uint8)
        if len(red_t) > num_clean_t:
            is_hole_fill[num_clean_t:] = 1
        red_face_attributes = {"is_hole_fill": ("uchar", is_hole_fill)}
        if not export_ply_with_face_attributes(
            args.red_output, red_v, red_t, red_c, face_attributes=red_face_attributes
        ):
            raise RuntimeError(f"Failed to write {args.red_output}")
    print(f"{len(records)} boundary loops; {sum(r['filled_region'] is not None for r in records)} filled; report: {args.report}")


if __name__ == "__main__":
    main()
