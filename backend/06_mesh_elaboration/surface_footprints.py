"""Measured horizontal footprints and finite wall contours, in a scene-local frame."""
from __future__ import annotations

import cv2
import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import polygonize, unary_union


def polygons(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == 'Polygon':
        return [geometry]
    if geometry.geom_type in ('MultiPolygon', 'GeometryCollection'):
        return [part for geom in geometry.geoms for part in polygons(geom)]
    return []


def frame_footprint(geometry, a, b, inverse=False):
    matrix = np.array([[a[0], b[0]], [a[2], b[2]]])
    if inverse:
        matrix = matrix.T
    return affine_transform(geometry, [*matrix[0], *matrix[1], 0., 0.])


def orthogonal_ring(points, tolerance):
    """Fit alternating straight horizontal/vertical lines to a measured contour."""
    points = cv2.approxPolyDP(np.asarray(points, np.float32), tolerance, True).reshape(-1, 2)
    lines = []
    for p, q in zip(points, np.roll(points, -1, axis=0)):
        delta = q-p
        length = float(np.linalg.norm(delta))
        if length < 1e-10:
            continue
        axis = 0 if abs(delta[1]) > abs(delta[0]) else 1
        value = float((p[axis]+q[axis])/2)
        if lines and lines[-1][0] == axis:
            _, old, weight = lines[-1]
            lines[-1] = (axis, (old*weight+value*length)/(weight+length), weight+length)
        else:
            lines.append((axis, value, length))
    if len(lines) > 1 and lines[0][0] == lines[-1][0]:
        axis, x, wx = lines.pop()
        _, y, wy = lines[0]
        lines[0] = (axis, (x*wx+y*wy)/(wx+wy), wx+wy)
    if len(lines) < 4:
        return None
    corners = []
    for prev, curr in zip(lines[-1:]+lines[:-1], lines):
        corner = [0., 0.]
        corner[prev[0]], corner[curr[0]] = prev[1], curr[1]
        corners.append(corner)
    return np.asarray(corners)


def observed_footprint(vertices, faces, a, b, cell_size=.02, min_feature=.10, return_contour=False):
    """Rasterize triangle area, independent of vertex/face counts and world translation."""
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    ab = np.column_stack([vertices @ a, vertices @ b])
    triangles = ab[faces]
    first, second = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
    area2 = np.abs(first[:, 0]*second[:, 1]-first[:, 1]*second[:, 0])
    triangles = triangles[area2 > 1e-10]
    if not len(triangles):
        raise ValueError('Surface has no measurable horizontal footprint')
    lo = triangles.reshape(-1, 2).min(axis=0)
    hi = triangles.reshape(-1, 2).max(axis=0)
    cell = max(cell_size, float((hi-lo).max())/4096)
    padding = max(.15, min_feature)*2
    origin = lo-padding
    size = np.ceil((hi-origin+padding)/cell).astype(int)+1
    grid = np.zeros((size[1], size[0]), np.uint8)
    pixel_triangles = np.rint((triangles-origin)/cell).astype(np.int32)
    # Filling each triangle separately implements union even for overlapping relief faces.
    for triangle in pixel_triangles:
        cv2.fillConvexPoly(grid, triangle, 1)
    close_size = max(1, int(round(.15/cell))) | 1
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
    open_size = max(1, int(round(min_feature/cell))) | 1
    opened = cv2.morphologyEx(grid, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8))
    if opened.any():
        grid = opened
    contours, hierarchy = cv2.findContours(grid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        raise ValueError('Surface footprint is empty after noise removal')
    result, raw = [], []
    rings = [orthogonal_ring(c.reshape(-1, 2)*cell, min_feature/3) for c in contours]
    rings = [ring+origin if ring is not None else None for ring in rings]
    for i, ring in enumerate(rings):
        if hierarchy[0, i, 3] != -1 or ring is None:
            continue
        holes = [rings[j] for j in range(len(rings)) if hierarchy[0, j, 3] == i and rings[j] is not None]
        raw_holes = [contours[j].reshape(-1,2)*cell+origin for j in range(len(rings))
                     if hierarchy[0,j,3] == i and rings[j] is not None]
        raw.append(Polygon(contours[i].reshape(-1,2)*cell+origin,raw_holes).buffer(0))
        poly = Polygon(ring, holes).buffer(0)
        result.extend(p for p in polygons(poly) if p.area >= min_feature**2)
    if not result:
        raise ValueError('No supported surface footprint components')
    fitted = unary_union(result)
    return (fitted, unary_union(raw)) if return_contour else fitted


def wall_regions(wall_planes, a, b, elevation=None, tolerance=.15):
    """Close only local endpoint errors; polygonize arbitrary measured wall layouts."""
    segments = []
    for wall in wall_planes:
        normal = np.array(wall['normal_unit'])
        if abs(normal @ a) > .999:
            full_span = wall['span_w_m']
            make_segment = lambda span: np.array([[wall['offset_m'], span[0]], [wall['offset_m'], span[1]]])
        elif abs(normal @ b) > .999:
            full_span = wall['span_u_m']
            make_segment = lambda span: np.array([[span[0], wall['offset_m']], [span[1], wall['offset_m']]])
        else:
            continue
        for region in wall.get('trim_regions') or [{'span_m': full_span, 'bounds_y': wall.get('bounds_y')}]:
            bounds = region.get('bounds_y', wall.get('bounds_y'))
            if (elevation is not None and bounds
                    and not bounds[0]-max(tolerance, .15*(bounds[1]-bounds[0]))
                    <= elevation <= bounds[1]+max(tolerance, .15*(bounds[1]-bounds[0]))):
                continue
            segments.append(make_segment(region['span_m']))
    for i, first in enumerate(segments):
        for second in segments[i+1:]:
            first_axis = int(np.argmax(np.abs(first[1]-first[0])))
            second_axis = int(np.argmax(np.abs(second[1]-second[0])))
            if first_axis == second_axis:
                continue
            crossing = np.zeros(2)
            crossing[1-first_axis] = first[0, 1-first_axis]
            crossing[1-second_axis] = second[0, 1-second_axis]
            point = LineString([crossing, crossing+1e-12]).centroid
            if LineString(first).distance(point) > tolerance or LineString(second).distance(point) > tolerance:
                continue
            for segment, axis in ((first, first_axis), (second, second_axis)):
                for j in range(2):
                    if abs(segment[j, axis]-crossing[axis]) <= tolerance:
                        segment[j, axis] = crossing[axis]
    if not segments:
        return []
    return list(polygonize(unary_union([LineString(segment) for segment in segments])))


def constrain_footprint(observed, regions, tolerance=.15):
    """Complete well-supported room spans and trim thin exterior slivers before slabs."""
    if not regions:
        return observed
    result = observed
    for room in regions:
        overlap = observed.intersection(room)
        if overlap.area < .1*min(observed.area, room.area):
            continue
        # Matching bounds alone mistakes a boundary notch for an incomplete room.
        if (np.max(np.abs(np.array(overlap.bounds)-np.array(room.bounds))) <= tolerance
                and overlap.boundary.hausdorff_distance(room.boundary) <= tolerance):
            result = result.union(room)
    wall_contour = unary_union(regions)
    outside = result.difference(wall_contour)
    broad = [p for p in polygons(outside) if not p.buffer(-tolerance/2, join_style=2).is_empty]
    clipped = unary_union([result.intersection(wall_contour), *broad])
    if clipped.is_empty:
        return observed
    return clipped


def snap_footprint_to_walls(geometry, wall_planes, a, b, tolerance=.15):
    """Align entire supported boundary edges, including finite recessed wall segments."""
    result = []
    for poly in polygons(geometry):
        ring = np.asarray(poly.exterior.coords[:-1]).copy()
        for i in range(len(ring)):
            j = (i+1) % len(ring)
            delta = ring[j]-ring[i]
            axis = 0 if abs(delta[0]) < 1e-8 else 1 if abs(delta[1]) < 1e-8 else None
            if axis is None:
                continue
            low, high = sorted([ring[i, 1-axis], ring[j, 1-axis]])
            choices = []
            for wall in wall_planes:
                normal = np.array(wall['normal_unit'])
                normal_axis = a if axis == 0 else b
                if abs(normal @ normal_axis) < .999:
                    continue
                distance = abs(ring[i, axis]-wall['offset_m'])
                if distance >= tolerance:
                    continue
                span_key = 'span_w_m' if axis == 0 else 'span_u_m'
                spans = [region['span_m'] for region in wall.get('trim_regions', [])] or [wall[span_key]]
                if any(min(high, span[1])-max(low, span[0]) >= .5*(high-low) for span in spans):
                    choices.append((distance, wall['offset_m']))
            if choices:
                ring[[i, j], axis] = min(choices)[1]
        result.extend(polygons(Polygon(ring, [list(hole.coords) for hole in poly.interiors]).buffer(0)))
    return unary_union(result)


def clean_footprint(geometry, min_edge=.10, hole_size=.40):
    """Remove short orthogonal jogs and fill holes small in both local dimensions."""
    result = []
    for poly in polygons(geometry):
        holes = []
        for ring in poly.interiors:
            bounds = Polygon(ring).bounds
            if max(bounds[2]-bounds[0], bounds[3]-bounds[1]) >= hole_size-1e-9:
                holes.append(list(ring.coords))
        poly = Polygon(np.round(np.array(poly.exterior.coords), 8),
                       [np.round(hole, 8) for hole in holes]).simplify(1e-8)
        while True:
            rings = [np.array(r.coords[:-1]) for r in [poly.exterior, *poly.interiors]]
            short = [(r, i) for r, ring in enumerate(rings)
                     for i, length in enumerate(np.linalg.norm(np.roll(ring, -1, axis=0)-ring, axis=1))
                     if length < min_edge-1e-9]
            if not short:
                break
            candidates = []
            for r, i in short:
                ring = rings[r]
                j = (i+1) % len(ring)
                axis = int(np.argmax(np.abs(ring[j]-ring[i])))
                for anchor in (i, j):
                    value = ring[anchor, axis]
                    changed = ring.copy()
                    neighbor = (j+1) % len(ring) if anchor == i else (i-1) % len(ring)
                    changed[[i, j, neighbor], axis] = value
                    replacement = rings.copy(); replacement[r] = changed
                    candidate = Polygon(replacement[0], replacement[1:]).simplify(0)
                    if not candidate.is_valid:
                        candidate = candidate.buffer(0).simplify(1e-8)
                    if not candidate.is_valid or candidate.is_empty or candidate.area < 1e-10:
                        continue
                    if candidate.geom_type != 'Polygon':
                        continue
                    new_count = sum(
                        len(ring.coords)-1
                        for ring in [candidate.exterior, *candidate.interiors]
                    )
                    if new_count >= sum(len(h) for h in rings):
                        continue
                    # Prefer trimming a protrusion to filling a measured wall recess.
                    added = candidate.difference(poly).area
                    candidates.append((added > 1e-9, candidate.symmetric_difference(poly).area, candidate))
            if not candidates:
                break  # Keep measured contour when every simplification would erase surface.
            poly = min(candidates, key=lambda c: c[:2])[2]
        result.append(poly)
    return unary_union(result)


def regularize_orthogonal_footprint(geometry, min_edge=.10, tolerance=.02):
    """Fit axis-aligned polygon rings and enforce right-angle corners and a minimum edge."""
    if not np.isfinite([min_edge, tolerance]).all() or min_edge <= 0 or tolerance < 0:
        raise ValueError('Footprint edge and fit tolerances must be finite and positive')

    tolerance = max(tolerance, min_edge / 2)
    fitted = []
    for source in polygons(geometry):
        rings = []
        for source_ring in [source.exterior, *source.interiors]:
            source_points = np.asarray(source_ring.coords[:-1], dtype=float)
            points = orthogonal_ring(source_points, tolerance)
            if points is None:
                raise ValueError('Could not fit an orthogonal footprint ring')
            source_area = .5 * np.sum(
                source_points[:, 0] * np.roll(source_points[:, 1], -1)
                - source_points[:, 1] * np.roll(source_points[:, 0], -1)
            )
            fitted_area = .5 * np.sum(
                points[:, 0] * np.roll(points[:, 1], -1)
                - points[:, 1] * np.roll(points[:, 0], -1)
            )
            rings.append(points if source_area * fitted_area >= 0 else points[::-1].copy())
        polygon = Polygon(rings[0], rings[1:])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_valid or polygon.is_empty or polygon.area <= 1e-8:
            raise ValueError('Orthogonal footprint fit produced invalid geometry')
        fitted.extend(polygons(clean_footprint(polygon, min_edge, hole_size=0.0)))

    result = unary_union(fitted)
    if result.is_empty:
        raise ValueError('Orthogonal footprint fit produced an empty shape')
    for polygon in polygons(result):
        for ring in [polygon.exterior, *polygon.interiors]:
            points = np.asarray(ring.coords, dtype=float)
            edges = np.diff(points, axis=0)
            lengths = np.linalg.norm(edges, axis=1)
            if np.any(lengths < min_edge - 1e-8):
                raise ValueError(f'Orthogonal footprint contains an edge shorter than {min_edge:.2f} m')
            if np.any((np.abs(edges[:, 0]) > 1e-8) & (np.abs(edges[:, 1]) > 1e-8)):
                raise ValueError('Orthogonal footprint contains a diagonal edge')
            previous = np.roll(edges, 1, axis=0)
            if np.any(np.abs(np.sum(previous * edges, axis=1)) > 1e-8):
                raise ValueError('Orthogonal footprint has neighboring edges that are not perpendicular')
    return result
