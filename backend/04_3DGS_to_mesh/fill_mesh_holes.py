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
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from shapely import constrained_delaunay_triangles
from shapely.geometry import Polygon


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


def fill_mesh(mesh, regions, auto=False, seal_all=False):
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
            for i in np.flatnonzero(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1) < 1e-12):
                vertex = int(loop[(i + 1) % len(loop)])
                radial = vertices[vertex] - points.mean(axis=0)
                if np.linalg.norm(radial) < 1e-12:
                    radial = np.array([1.0, 0.0, 0.0])
                axis = np.eye(3)[np.argmin(np.abs(radial))]
                direction = np.cross(radial, axis)
                # A zero-length source edge needs a tiny separation to support a real cap face.
                vertices[vertex] += direction / np.linalg.norm(direction) * 1e-5
    existing_edges = np.unique(edge_keys(triangles)) if auto or seal_all or regions else np.empty(0, dtype=np.uint64)
    added_internal_edges = set()
    extra_vertices = []
    extra_colors = []
    source_colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
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
                local_colors.append(source_colors[loop].mean(axis=0))
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
                local_vertices.append((point(a) + point(b)) / 2 - normal * min(0.01, np.linalg.norm(point(a) - point(b)) * 0.02))
                if source_colors is not None:
                    local_colors.append((color(a) + color(b)) / 2)
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
            result.vertex_colors = o3d.utility.Vector3dVector(np.vstack((colors, colors[faces_to_recess].mean(axis=1))))
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
        if not o3d.io.write_triangle_mesh(str(args.output), result):
            raise RuntimeError(f"Failed to write {args.output}")
    if args.red_output:
        red = red_patches(result, len(mesh.triangles))
        args.red_output.parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_triangle_mesh(str(args.red_output), red):
            raise RuntimeError(f"Failed to write {args.red_output}")
    print(f"{len(records)} boundary loops; {sum(r['filled_region'] is not None for r in records)} filled; report: {args.report}")


if __name__ == "__main__":
    main()
