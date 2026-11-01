"""Shared geometry constraints for structural regularization (metres, Y up)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from shapely import constrained_delaunay_triangles
from shapely.geometry import Point, Polygon, box
from shapely.affinity import affine_transform
from shapely.ops import unary_union

ALIGNMENT_TOLERANCE = 0.15


def pattern_frame(normal, offset, inward, transverse):
    scale = np.linalg.norm(normal)
    normal = np.asarray(normal, dtype=float) / scale
    transverse = np.array(transverse, dtype=float, copy=True)
    transverse -= transverse.dot(normal) * normal
    transverse /= np.linalg.norm(transverse)
    longitudinal = np.cross(normal, transverse)
    relief = normal if np.dot(normal, inward) > 0 else -normal
    return normal, -float(offset) / scale * normal, relief, transverse, longitudinal


def perimeter_pattern_axes(normal, detected_longitudinal, horizontal_axes):
    """Choose the closest ceiling edge direction and an orthogonal transverse axis."""
    normal = np.array(normal, dtype=float, copy=True)
    normal /= np.linalg.norm(normal)
    edges = np.array(horizontal_axes, dtype=float, copy=True)
    # Lift vertically into the roof plane; orthogonal projection changes the plan direction.
    edges[:, 1] = -(edges[:, [0, 2]] @ normal[[0, 2]])/normal[1]
    edges /= np.linalg.norm(edges, axis=1)[:, None]
    dots = edges @ detected_longitudinal
    index = np.argmax(np.abs(dots))
    longitudinal = edges[index] * (1. if dots[index] >= 0 else -1.)
    transverse = np.cross(longitudinal, normal)
    transverse /= np.linalg.norm(transverse)
    return transverse, longitudinal


def snap_small_excursions(points, wall_offsets, tolerance=ALIGNMENT_TOLERANCE):
    """Snap nearby perimeter coordinates in either direction; keep larger extensions."""
    points = np.array(points, dtype=float, copy=True)
    for axis, names in enumerate((("West Wall", "East Wall"), ("South Wall", "North Wall"))):
        for name in names:
            if name in wall_offsets:
                value = wall_offsets[name]
                close = np.abs(points[:, axis] - value) < tolerance
                points[close, axis] = value
    return points


def supported_horizontal_extensions(perimeter, source_ab, wall_offsets):
    """Keep broad observed extensions; isolated scan crumbs cannot grow the footprint."""
    points = snap_small_excursions(source_ab, wall_offsets)
    # A 5cm occupancy grid requires a connected patch wider than the alignment tolerance.
    cell = 0.05
    indices = np.floor(points / cell).astype(np.int64)
    if len(indices):
        origin = indices.min(axis=0)
        rows = int(np.ptp(indices[:, 1]))+1
        codes = (indices[:, 0]-origin[0])*rows + indices[:, 1]-origin[1]
        codes, counts = np.unique(codes, return_counts=True)
        indices = np.column_stack([codes//rows+origin[0], codes%rows+origin[1]])
    else:
        counts = np.empty(0, dtype=int)
    occupied = unary_union([box(a*cell, b*cell, (a+1)*cell, (b+1)*cell)
                            for (a, b), count in zip(indices, counts) if count >= 5])
    extra = occupied.difference(Polygon(perimeter))
    parts = list(extra.geoms) if hasattr(extra, "geoms") else [extra]
    retained = [Polygon(perimeter)]
    for part in parts:
        core = part.buffer(-ALIGNMENT_TOLERANCE / 2, join_style=2)
        if not core.is_empty and part.distance(retained[0]) < 1e-8:
            retained.append(part)
    result = unary_union(retained)
    if result.geom_type != "Polygon":
        result = max(result.geoms, key=lambda p: p.area)
    return np.asarray(result.exterior.coords[:-1]).tolist()


def cut_small_overhangs(footprint, wall_perimeter):
    """Cut thin exterior pieces at the full wall contour, including recessed corners."""
    surface = Polygon(footprint)
    walls = Polygon(wall_perimeter)
    outside = surface.difference(walls)
    parts = outside.geoms if hasattr(outside, "geoms") else [outside]
    large = [part for part in parts
             if not part.is_empty and not part.buffer(-ALIGNMENT_TOLERANCE/2, join_style=2).is_empty]
    trimmed = unary_union([surface.intersection(walls), *large])
    if trimmed.geom_type != "Polygon" or trimmed.is_empty or not trimmed.is_valid:
        raise ValueError("Wall clipping produced a disconnected or invalid surface footprint")
    return np.asarray(trimmed.exterior.coords[:-1]).tolist()


def polygon_triangles(poly):
    """Constrained triangulation preserves concavities, holes, and boundary vertices."""
    triangles = constrained_delaunay_triangles(poly)
    vertices, faces, lookup = [], [], {}
    for tri in triangles.geoms:
        face = []
        for xy in tri.exterior.coords[:3]:
            key = tuple(xy)
            if key not in lookup:
                lookup[key] = len(vertices)
                vertices.append(xy)
            face.append(lookup[key])
        faces.append(face)
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int32)


def beam_surface(normal, offset, inward, transverse, period, width, depth, phase,
                 footprint_xz):
    """Clip each relief strip to the exact footprint and join only internal flanks."""
    scale = np.linalg.norm(normal)
    normal, origin, relief, u, v = pattern_frame(normal, offset, inward, transverse)
    offset = float(offset) / scale
    footprint = footprint_xz if hasattr(footprint_xz, 'geom_type') else Polygon(footprint_xz)
    cy, cx, cz = -offset/normal[1], -normal[0]/normal[1], -normal[2]/normal[1]
    poly = affine_transform(footprint, [u[0]+u[1]*cx, u[2]+u[1]*cz,
                                      v[0]+v[1]*cx, v[2]+v[1]*cz,
                                      u[1]*cy-origin @ u, v[1]*cy-origin @ v]).buffer(0)
    # Vertical beam sides keep every relief level inside the same horizontal footprint.
    relief = np.array([0., float(relief @ normal)/normal[1], 0.])
    if not poly.is_valid or poly.is_empty:
        raise ValueError("Invalid ceiling footprint")
    s0, t0, s1, t1 = poly.bounds
    duty = np.clip(width/period, 0.05, 0.95)
    base, beam = -duty*depth, (1-duty)*depth
    cuts = [s0, s1]
    for k in range(int(np.floor((s0-phase)/period))-1, int(np.ceil((s1-phase)/period))+2):
        cuts.extend(x for x in (phase+k*period-width/2, phase+k*period+width/2) if s0 < x < s1)
    cuts = sorted(set(cuts))
    vertices, faces = [], []
    for left, right in zip(cuts[:-1], cuts[1:]):
        mid = (left+right)/2
        is_beam = abs((mid-phase+period/2) % period-period/2) < width/2
        height = beam if is_beam else base
        clipped = poly.intersection(box(left, t0-1, right, t1+1))
        parts = clipped.geoms if hasattr(clipped, "geoms") else [clipped]
        for part in parts:
            if part.geom_type != "Polygon" or part.area < 1e-12:
                continue
            xy, triangles = polygon_triangles(part)
            start = len(vertices)
            vertices.extend(origin + xy[:, :1]*u + xy[:, 1:]*v + height*relief)
            faces.extend(triangles + start)
            if is_beam:
                boundary_edges = [(a, b) for ring in [part.exterior, *part.interiors]
                                  for a, b in zip(np.asarray(ring.coords)[:-1], np.asarray(ring.coords)[1:])]
                for a, b in boundary_edges:
                    on_cut = abs(a[0]-b[0]) < 1e-9 and (abs(a[0]-left) < 1e-9 or abs(a[0]-right) < 1e-9)
                    # An edge on the external perimeter is closed later by the slab skirt.
                    if not on_cut or poly.boundary.distance(Point((a+b)/2)) < 1e-8:
                        continue
                    start = len(vertices)
                    vertices.extend([origin+a[0]*u+a[1]*v+base*relief,
                                     origin+b[0]*u+b[1]*v+base*relief,
                                     origin+b[0]*u+b[1]*v+beam*relief,
                                     origin+a[0]*u+a[1]*v+beam*relief])
                    faces.extend(np.array([[0, 1, 2], [0, 2, 3]])+start)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    # All substrate and soffit faces point into the room.
    mesh.fix_normals()
    surface = np.abs(mesh.face_normals @ normal) > 0.9
    if np.any(surface) and np.mean(mesh.face_normals[surface] @ relief) < 0:
        mesh.invert()
    return mesh.vertices.copy(), mesh.faces.copy()


def weld_float32_boundary_edges(mesh: trimesh.Trimesh) -> tuple[int, int]:
    """Collapse one-ULP boundary edges left by float32 clipping intersections."""
    mesh.process()
    if not len(mesh.faces):
        return 0, 0

    edge_counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    boundary = mesh.edges_unique[edge_counts == 1]
    vertices = mesh.vertices.astype(np.float32)
    endpoints = vertices[boundary]
    delta = np.abs(endpoints[:, 0] - endpoints[:, 1])
    resolution = np.maximum(np.spacing(np.abs(endpoints[:, 0])),
                            np.spacing(np.abs(endpoints[:, 1])))
    weld = np.any(delta > 0, axis=1) & np.all(delta <= resolution, axis=1)
    if not np.any(weld):
        return 0, 0

    parent = np.arange(len(vertices), dtype=np.int64)

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for a, b in boundary[weld]:
        root_a, root_b = root(int(a)), root(int(b))
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    roots = np.arange(len(parent))
    while not np.array_equal(roots, parent[roots]):
        roots = parent[roots]
    keep, inverse = np.unique(roots, return_inverse=True)
    mesh.update_vertices(keep, inverse)
    faces = mesh.faces
    noncollapsed = np.all(faces != np.roll(faces, 1, axis=1), axis=1)
    collapsed_face_count = int(np.count_nonzero(~noncollapsed))
    if collapsed_face_count:
        mesh.update_faces(noncollapsed)
        mesh.remove_unreferenced_vertices()
    return int(np.count_nonzero(weld)), collapsed_face_count


def close_slab(bottom_vertices, bottom_faces, floor_equation,
               diagnostics_dir: Path | None = None, diagnostic_context: dict | None = None):
    """Use the actual relief boundary for the skirt and a conforming projected top."""
    # Match PLY's float32 coordinates before topology construction so export cannot collapse seams later.
    bottom = trimesh.Trimesh(vertices=np.asarray(bottom_vertices, dtype=np.float32),
                             faces=bottom_faces, process=False)
    precision_edge_count, precision_face_count = weld_float32_boundary_edges(bottom)
    top = bottom.vertices.copy()
    peq = np.array(floor_equation, dtype=float, copy=True)
    top[:, 1] = -(peq[0]*top[:, 0] + peq[2]*top[:, 2] + peq[3])/peq[1]
    lift = max(0.0, float(bottom.vertices[:, 1].max()+1e-5-top[:, 1].min()))
    top[:, 1] += lift
    peq[3] -= lift*peq[1]
    n = len(top)
    unique, count = np.unique(bottom.edges_sorted, axis=0, return_counts=True)
    boundary = unique[count == 1]
    side_vertices, side_faces = [], []
    skipped_vertical_boundary_edge_count = 0
    boundary_vertices = bottom.vertices[np.unique(boundary)]
    for a, b in boundary:
        pa, pb = bottom.vertices[[a, b]]
        length = np.linalg.norm((pb-pa)[[0, 2]])
        if length < 1e-8:
            skipped_vertical_boundary_edge_count += 1
            continue
        # Include relief corners on vertical skirt edges to avoid T junctions.
        ring = [[0., pa[1]], [length, pb[1]]]
        for endpoint, x, top_y in ((pb, length, top[b, 1]), (pa, 0., top[a, 1])):
            same = np.linalg.norm(boundary_vertices[:, [0, 2]]-endpoint[[0, 2]], axis=1) < 1e-7
            heights = sorted(set(boundary_vertices[same, 1].tolist()+[float(top_y)]))
            heights = [h for h in heights if endpoint[1]+1e-8 < h <= top_y+1e-8]
            ring.extend([[x, h] for h in (heights if x else heights[::-1])])
        xy, triangles = polygon_triangles(Polygon(ring))
        start = 2*n+len(side_vertices)
        direction = (pb-pa)/length
        direction[1] = 0
        xyz = pa + xy[:, :1]*direction
        xyz[:, 1] = xy[:, 1]
        side_vertices.extend(xyz)
        side_faces.extend(triangles+start)
    faces = np.vstack([bottom.faces, bottom.faces[:, ::-1]+n, np.asarray(side_faces)])
    vertices = np.vstack([bottom.vertices, top, side_vertices]).astype(np.float32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    watertight = bool(mesh.is_watertight)
    volume = float(mesh.volume) if watertight else None
    if not watertight or volume <= 0:
        diagnostic_path = None
        if diagnostics_dir is not None:
            try:
                diagnostics_dir = Path(diagnostics_dir)
                diagnostics_dir.mkdir(parents=True, exist_ok=True)
                bottom.export(diagnostics_dir / "clipped_ceiling_surface.ply")
                mesh.export(diagnostics_dir / "attempted_slab.ply")
                input_edge_counts = np.bincount(
                    bottom.edges_unique_inverse, minlength=len(bottom.edges_unique)
                )
                input_boundary_edges = bottom.edges_unique[input_edge_counts == 1]
                input_nonmanifold_edges = bottom.edges_unique[input_edge_counts > 2]
                input_boundary_points = bottom.vertices[input_boundary_edges]
                input_boundary_xz_lengths = np.linalg.norm(
                    (input_boundary_points[:, 1] - input_boundary_points[:, 0])[:, [0, 2]], axis=1
                ) if len(input_boundary_edges) else np.empty(0)
                edge_counts = np.bincount(
                    mesh.edges_unique_inverse, minlength=len(mesh.edges_unique)
                )
                boundary_edges = mesh.edges_unique[edge_counts == 1]
                nonmanifold_edges = mesh.edges_unique[edge_counts > 2]
                boundary_points = mesh.vertices[boundary_edges]
                boundary_xz_lengths = np.linalg.norm(
                    (boundary_points[:, 1] - boundary_points[:, 0])[:, [0, 2]], axis=1
                ) if len(boundary_edges) else np.empty(0)
                np.savez_compressed(
                    diagnostics_dir / "edge_diagnostics.npz",
                    input_boundary_edges_xyz=input_boundary_points,
                    input_nonmanifold_edges_xyz=bottom.vertices[input_nonmanifold_edges],
                    input_nonmanifold_edge_use_counts=input_edge_counts[input_edge_counts > 2],
                    input_vertical_boundary_edges_xyz=(
                        input_boundary_points[input_boundary_xz_lengths < 1e-8]
                    ),
                    output_boundary_edges_xyz=boundary_points,
                    output_nonmanifold_edges_xyz=mesh.vertices[nonmanifold_edges],
                    output_nonmanifold_edge_use_counts=edge_counts[edge_counts > 2],
                    output_vertical_boundary_edges_xyz=boundary_points[boundary_xz_lengths < 1e-8],
                )
                report = {
                    "context": diagnostic_context or {},
                    "floor_equation": np.asarray(floor_equation, dtype=float).tolist(),
                    "clearance_lift_m": lift,
                    "raw_clipped_surface_vertex_count": int(len(bottom_vertices)),
                    "raw_clipped_surface_face_count": int(len(bottom_faces)),
                    "processed_clipped_surface_vertex_count": int(len(bottom.vertices)),
                    "processed_clipped_surface_face_count": int(len(bottom.faces)),
                    "float32_ulp_boundary_edge_candidate_count": int(
                        (diagnostic_context or {}).get("float32_ulp_boundary_edge_candidate_count",
                                                       precision_edge_count)
                    ),
                    "float32_ulp_collapsed_face_count": int(
                        (diagnostic_context or {}).get("float32_ulp_collapsed_face_count",
                                                       precision_face_count)
                    ),
                    "slab_side_face_count": int(len(side_faces)),
                    "skipped_vertical_boundary_edge_count": skipped_vertical_boundary_edge_count,
                    "attempted_slab_vertex_count": int(len(mesh.vertices)),
                    "attempted_slab_face_count": int(len(mesh.faces)),
                    "is_watertight": watertight,
                    "is_winding_consistent": bool(mesh.is_winding_consistent),
                    "volume_m3": volume,
                    "input_boundary_edge_count": int(len(input_boundary_edges)),
                    "input_nonmanifold_edge_count": int(len(input_nonmanifold_edges)),
                    "input_vertical_boundary_edge_count": int(
                        np.count_nonzero(input_boundary_xz_lengths < 1e-8)
                    ),
                    "boundary_edge_count": int(len(boundary_edges)),
                    "nonmanifold_edge_count": int(len(nonmanifold_edges)),
                    "vertical_boundary_edge_count": int(np.count_nonzero(boundary_xz_lengths < 1e-8)),
                    "euler_number": int(mesh.euler_number),
                }
                (diagnostics_dir / "slab_closure_report.json").write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
                diagnostic_path = diagnostics_dir
            except Exception as error:
                print(f"Could not save slab-closure diagnostics: {error}")
        message = "Slab boundary did not produce a watertight positive solid"
        if diagnostic_path is not None:
            message += f"; diagnostics saved to {diagnostic_path}"
        raise ValueError(message)
    return mesh, peq, lift


def surface_wall_sections(vertices, faces, wall_normal, offset, along):
    """Intersect a finite triangulated surface with a vertical wall plane."""
    sections = []
    for tri in np.asarray(vertices)[faces]:
        if abs(np.cross(tri[1]-tri[0], tri[2]-tri[0])[1]) < 1e-12:
            continue
        distance = tri @ wall_normal - offset
        hits = []
        for i in range(3):
            j = (i+1) % 3
            if abs(distance[i]) < 1e-7:
                hits.append(tri[i])
            if distance[i]*distance[j] < 0:
                hits.append(tri[i] + (tri[j]-tri[i])*distance[i]/(distance[i]-distance[j]))
        if len(hits) >= 2:
            hits = np.asarray(hits)
            order = np.argsort(hits @ along)
            a, b = hits[order[0]], hits[order[-1]]
            if (b-a) @ along > 1e-8:
                sections.append((float(a @ along), float(b @ along), a[1], b[1]))
    return sections


def wall_profile_mesh(normal, offset, along, span, floor_peq, roof_peq, sections):
    """Split the wall at every roof relief transition, then trim to the real surface."""
    cuts = sorted(set([*span] + [float(np.clip(x, *span)) for s in sections for x in s[:2]]))
    vertices, faces = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b-a < 1e-8:
            continue
        active = [s for s in sections if s[0]-1e-8 <= (a+b)/2 <= s[1]+1e-8]
        xz = np.array([a*along+offset*normal, b*along+offset*normal])
        bottom = -(xz @ floor_peq[:3]+floor_peq[3])/floor_peq[1]
        top = -(xz @ roof_peq[:3]+roof_peq[3])/roof_peq[1]
        if active:
            top = np.array([min(s[2]+(x-s[0])*(s[3]-s[2])/(s[1]-s[0]) for s in active) for x in (a, b)])
        start = len(vertices)
        vertices.extend([[xz[0,0], bottom[0], xz[0,2]], [xz[1,0], bottom[1], xz[1,2]],
                         [xz[1,0], top[1], xz[1,2]], [xz[0,0], top[0], xz[0,2]]])
        faces.extend(np.array([[0,1,2], [0,2,3]])+start)
    return np.asarray(vertices), np.asarray(faces)
