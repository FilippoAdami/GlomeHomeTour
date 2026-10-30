#!/usr/bin/env python3
"""Fit axis-aligned enclosing contours to saved per-floor projection meshes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as PlotPath
from matplotlib.ticker import MultipleLocator
import numpy as np
import open3d as o3d
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from surface_footprints import clean_footprint, orthogonal_ring, polygons

MIN_SEGMENT_M = 0.15
MAX_OUTSIDE_FRACTION = 0.0
# Smaller than any omitted 5 mm raster cell; this only absorbs GEOS round-off.
OUTSIDE_NUMERIC_TOLERANCE_FRACTION = 1e-12
GRID_CELL_M = 0.005
MORPHOLOGY = ("raw", "close", "open", "close_open", "open_close")
SIMPLIFICATION_TOLERANCES_M = np.arange(0.02, 0.151, 0.02)


def _mask_geometry(mask: np.ndarray, origin: np.ndarray, cell: float,
                   tolerance_m: float | None = None):
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return Polygon()
    rings = []
    for contour in contours:
        points = contour.reshape(-1, 2)
        if tolerance_m is not None:
            ring = orthogonal_ring(points, tolerance_m / cell)
        else:
            ring = points if len(points) >= 4 else None
        rings.append(ring * cell + origin if ring is not None else None)
    result = []
    for index, ring in enumerate(rings):
        if hierarchy[0, index, 3] != -1 or ring is None:
            continue
        holes = [rings[j] for j in range(len(rings))
                 if hierarchy[0, j, 3] == index and rings[j] is not None]
        result.extend(polygons(Polygon(ring, holes).buffer(0)))
    if not result:
        return Polygon()
    geometry = unary_union(result)
    # OpenCV contours run through pixel centers; this restores the pixel-cell footprint.
    return geometry.buffer(cell / 2, join_style=2)


def _projected_union(mesh: o3d.geometry.TriangleMesh):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    if not len(faces):
        raise ValueError("Projection mesh contains no triangles")
    triangles = vertices[faces][:, :, (0, 2)]
    flat = triangles.reshape(-1, 2)
    low, high = flat.min(axis=0), flat.max(axis=0)
    cell = max(GRID_CELL_M, float(np.max(high - low)) / 4096)
    origin = low - cell
    size = np.ceil((high - origin + cell) / cell).astype(int) + 1
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    pixel_triangles = np.rint((triangles - origin) / cell).astype(np.int32)
    for start in range(0, len(pixel_triangles), 10_000):
        batch = pixel_triangles[start:start + 10_000]
        cv2.fillPoly(mask, [triangle.reshape(-1, 1, 2) for triangle in batch], 1)
    source = _mask_geometry(mask, origin, cell)
    if source.is_empty or source.area <= 0:
        raise ValueError("Projection has no measurable X/Z area")
    return mask, origin, cell, source


def _valid_edges(geometry: Polygon) -> tuple[bool, float, int]:
    lengths = []
    for polygon in polygons(geometry):
        for ring in [polygon.exterior, *polygon.interiors]:
            delta = np.diff(np.asarray(ring.coords), axis=0)
            segment_lengths = np.linalg.norm(delta, axis=1)
            segment_lengths = segment_lengths[segment_lengths > 1e-7]
            if np.any((np.abs(delta[:, 0]) > 1e-7) & (np.abs(delta[:, 1]) > 1e-7)):
                return False, 0.0, 0
            lengths.extend(segment_lengths.tolist())
    if not lengths:
        return False, 0.0, 0
    minimum = float(min(lengths))
    return minimum >= MIN_SEGMENT_M - 1e-7, minimum, len(lengths)


def _outside_fraction(source, geometry: Polygon) -> float:
    fraction = float(source.difference(geometry).area / source.area)
    return 0.0 if fraction <= OUTSIDE_NUMERIC_TOLERANCE_FRACTION else fraction


def _best_buffered_fit(base, source, maximum_radius: float, cell: float):
    outside = _outside_fraction(source, base)
    if outside <= MAX_OUTSIDE_FRACTION:
        first_radius = 0.0
    else:
        low, high = 0.0, cell
        while high < maximum_radius and _outside_fraction(
                source, base.buffer(high, join_style=2)) > MAX_OUTSIDE_FRACTION:
            low, high = high, min(high * 2, maximum_radius)
        if _outside_fraction(source, base.buffer(high, join_style=2)) > MAX_OUTSIDE_FRACTION:
            return None
        for _ in range(14):
            middle = (low + high) / 2
            if _outside_fraction(source, base.buffer(middle, join_style=2)) <= MAX_OUTSIDE_FRACTION:
                high = middle
            else:
                low = middle
        first_radius = high + cell / 10

    best = None
    # The first valid expansion gives the highest fill for this candidate.
    radius = first_radius
    last_radius = min(maximum_radius, first_radius + 0.30)
    while radius <= last_radius + 1e-9:
        candidate = base if radius == 0 else base.buffer(radius, join_style=2)
        valid, minimum, segment_count = _valid_edges(candidate)
        outside = _outside_fraction(source, candidate)
        if valid and outside == MAX_OUTSIDE_FRACTION:
            fill_ratio = source.intersection(candidate).area / candidate.area
            score = (fill_ratio, -outside, -candidate.area)
            if best is None or score > best[0]:
                best = (score, candidate, radius, outside, minimum, segment_count)
        if valid and outside == MAX_OUTSIDE_FRACTION:
            break
        radius += max(cell * 2, 0.01)
    return best


def _candidate_geometries(mask: np.ndarray, origin: np.ndarray, cell: float,
                          kernel: np.ndarray, source: Polygon):
    masks = {
        "raw": mask,
        "close": cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel),
        "open": cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel),
    }
    masks["close_open"] = cv2.morphologyEx(masks["close"], cv2.MORPH_OPEN, kernel)
    masks["open_close"] = cv2.morphologyEx(masks["open"], cv2.MORPH_CLOSE, kernel)
    for method in MORPHOLOGY:
        for tolerance in SIMPLIFICATION_TOLERANCES_M:
            geometry = _mask_geometry(masks[method], origin, cell, float(tolerance))
            if geometry.is_empty:
                continue
            geometry = clean_footprint(geometry, MIN_SEGMENT_M, MIN_SEGMENT_M)
            valid, _, _ = _valid_edges(geometry)
            if valid:
                yield method, float(tolerance), geometry
    yield from _occupancy_grid_candidates(mask, origin, cell)
    bounds = box(*source.bounds)
    if _valid_edges(bounds)[0]:
        yield "projection_bounds", 0.0, bounds


def _occupancy_grid_candidates(mask: np.ndarray, origin: np.ndarray, cell: float):
    block = int(np.ceil(MIN_SEGMENT_M / cell))
    phases = sorted(set(range(0, block, max(1, block // 6))))
    height, width = mask.shape
    for offset_z in phases:
        bottom = (-(height + offset_z)) % block
        for offset_x in phases:
            right = (-(width + offset_x)) % block
            padded = np.pad(mask, ((offset_z, bottom), (offset_x, right)))
            coarse = padded.reshape(
                padded.shape[0] // block, block,
                padded.shape[1] // block, block,
            ).max(axis=(1, 3))
            rectangles = []
            for row, occupied in enumerate(coarse):
                changes = np.diff(np.r_[0, occupied, 0].astype(np.int8))
                starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
                z0 = origin[1] + (-offset_z + row * block - 0.5) * cell
                z1 = origin[1] + (-offset_z + (row + 1) * block - 0.5) * cell
                for start, end in zip(starts, ends):
                    x0 = origin[0] + (-offset_x + start * block - 0.5) * cell
                    x1 = origin[0] + (-offset_x + end * block - 0.5) * cell
                    rectangles.append(box(x0, z0, x1, z1))
            if rectangles:
                yield f"occupancy_grid_x{offset_x}_z{offset_z}", 0.0, unary_union(rectangles)


def _save_plot(path: Path, floor_name: str, mask: np.ndarray, origin: np.ndarray,
               cell: float, geometry, metrics: dict):
    fig, ax = plt.subplots(figsize=(10, 9), dpi=180)
    extent = (origin[0], origin[0] + mask.shape[1] * cell,
              origin[1], origin[1] + mask.shape[0] * cell)
    ax.imshow(mask, origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=1,
              interpolation="nearest", alpha=0.8, zorder=1)
    for polygon in polygons(geometry):
        polygon = orient(polygon, sign=1.0)
        rings = [polygon.exterior, *polygon.interiors]
        paths = []
        for ring in rings:
            points = np.asarray(ring.coords)
            codes = np.full(len(points), PlotPath.LINETO, dtype=np.uint8)
            codes[0], codes[-1] = PlotPath.MOVETO, PlotPath.CLOSEPOLY
            paths.append(PlotPath(points, codes))
        ax.add_patch(PathPatch(PlotPath.make_compound_path(*paths), facecolor="#36a853",
                               edgecolor="none", alpha=0.16, zorder=2))
        x, z = polygon.exterior.xy
        ax.plot(x, z, color="#087b2d", linewidth=2.0, zorder=3)
        for hole in polygon.interiors:
            hx, hz = hole.xy
            ax.plot(hx, hz, color="#087b2d", linewidth=1.5, zorder=3)
    x0, z0, x1, z1 = geometry.bounds
    pad = 0.15
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(z0 - pad, z1 + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(which="major", color="#777", linewidth=0.65, alpha=0.5, zorder=0)
    ax.grid(which="minor", color="#aaa", linewidth=0.35, alpha=0.25, zorder=0)
    ax.set_title(
        f"{floor_name} enclosing contour — fill {metrics['fill_ratio']:.3%}, "
        f"outside {metrics['outside_area_fraction']:.3%}, "
        f"minimum edge {metrics['minimum_segment_m']:.3f} m"
    )
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fit_projection(path: Path, output_dir: Path) -> dict:
    mesh = o3d.io.read_triangle_mesh(str(path))
    mask, origin, cell, source = _projected_union(mesh)
    kernel_size = max(3, int(round(MIN_SEGMENT_M / cell))) | 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    maximum_radius = float(np.hypot(*np.ptp(np.asarray(mesh.vertices)[:, (0, 2)], axis=0))) + MIN_SEGMENT_M

    best = None
    for method, tolerance, base in _candidate_geometries(mask, origin, cell, kernel, source):
        fitted = _best_buffered_fit(base, source, maximum_radius, cell)
        if fitted is None:
            continue
        score, geometry, radius, outside, minimum, segment_count = fitted
        if best is None or score > best[0]:
            best = (score, geometry, method, tolerance, radius, outside, minimum, segment_count)
    if best is None:
        raise ValueError(
            f"No contour satisfies the {MIN_SEGMENT_M * 100:.0f} cm segment "
            f"and zero-omission limits: {path}"
        )

    _, geometry, method, tolerance, radius, outside, minimum, segment_count = best
    fill_ratio = source.intersection(geometry).area / geometry.area
    if outside > MAX_OUTSIDE_FRACTION or minimum < MIN_SEGMENT_M - 1e-7:
        raise AssertionError("Selected contour violates an enforced constraint")
    floor_name = path.name.removesuffix("_pre_collapse.ply").replace("_", " ").title()
    image = output_dir / f"{path.stem.removesuffix('_pre_collapse')}_enclosing_contour.png"
    metrics = {
        "source_mesh": str(path),
        "image": str(image),
        "projected_union_area_m2": float(source.area),
        "contour_area_m2": float(geometry.area),
        "retained_projected_area_m2": float(source.intersection(geometry).area),
        "outside_area_fraction": float(outside),
        "fill_ratio": float(fill_ratio),
        "minimum_segment_m": float(minimum),
        "segment_count": int(segment_count),
        "polygon_count": len(polygons(geometry)),
        "raster_cell_m": float(cell),
        "minimum_segment_constraint_m": MIN_SEGMENT_M,
        "maximum_outside_fraction": MAX_OUTSIDE_FRACTION,
        "selected_candidate": {
            "morphology": method,
            "simplification_tolerance_m": tolerance,
            "outward_buffer_m": float(radius),
        },
        "contours_xz": [
            {
                "exterior": [[float(x), float(z)] for x, z in polygon.exterior.coords],
                "holes": [[[float(x), float(z)] for x, z in ring.coords]
                          for ring in polygon.interiors],
            }
            for polygon in polygons(geometry)
        ],
    }
    _save_plot(image, floor_name, mask, origin, cell, geometry, metrics)
    return metrics


def main() -> int:
    global MIN_SEGMENT_M
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-dir", type=Path, required=True,
                        help="Directory containing floor_*_pre_collapse.ply files")
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to <projection-dir>/enclosing_contours")
    parser.add_argument("--min-segment-m", type=float, default=MIN_SEGMENT_M,
                        help=f"Minimum contour segment length in metres (default: {MIN_SEGMENT_M:.2f})")
    args = parser.parse_args()
    if not np.isfinite(args.min_segment_m) or args.min_segment_m <= 0:
        parser.error("--min-segment-m must be a finite positive number")
    MIN_SEGMENT_M = args.min_segment_m
    output_dir = args.output_dir or args.projection_dir / "enclosing_contours"
    inputs = sorted(args.projection_dir.glob("floor_*_pre_collapse.ply"))
    if not inputs:
        parser.error(f"No floor_*_pre_collapse.ply files found in {args.projection_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [fit_projection(path, output_dir) for path in inputs]
    manifest = output_dir / "enclosing_contours.json"
    manifest.write_text(json.dumps({"floors": results}, indent=2) + "\n")
    for result in results:
        print(f"{Path(result['source_mesh']).name}: fill={result['fill_ratio']:.3%}, "
              f"outside={result['outside_area_fraction']:.3%}, "
              f"segments={result['segment_count']}, "
              f"min={result['minimum_segment_m']:.3f} m -> {result['image']}")
    print(f"Saved contour manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
