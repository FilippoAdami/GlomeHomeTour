"""Project final red regions onto their regularized planes and mark sealed-mesh contours."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon as PolygonPatch, Rectangle
import numpy as np
import plyfile
import shapely
import trimesh
from scipy.optimize import minimize
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import unary_union

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from object_outlier_regions import _build_exterior_surface, _closest, _connected_regions, _region_surface, _visible_groups
from scan_loader import ScanDataset

LOGGER = logging.getLogger("phase07_red_region_contours")
LIME = np.array([0, 255, 0, 255], dtype=np.uint8)
PLANE_DISTANCE_LIMIT_M = 0.15
MIN_CONTOUR_BAND_M = 0.01
MAX_CONTOUR_BAND_M = 0.08
MAX_BBOX_PADDING_M = 0.10
TRIANGLE_CHUNK = 100_000


def _face_matrix(ply: plyfile.PlyData) -> np.ndarray:
    faces = ply["face"].data["vertex_indices"]
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Expected triangular faces")
    return np.asarray(faces, dtype=np.int64)


def _xyz(data: np.ndarray) -> np.ndarray:
    return np.column_stack([data[name] for name in ("x", "y", "z")]).astype(np.float32)


def _projector(normal: np.ndarray, offset: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = np.asarray(normal, dtype=np.float64)
    reference = np.array([0.0, 1.0, 0.0]) if abs(normal[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(reference, normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    return normal * offset, u, v


def _plane_keys(triangles: np.ndarray) -> list[tuple[tuple[float, ...], float]]:
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-15)
    axes = np.abs(normals).argmax(axis=1)
    flip = normals[np.arange(len(normals)), axes] < 0
    normals[flip] *= -1
    offsets = np.einsum("ij,ij->i", normals, triangles.mean(axis=1))
    return [(tuple(np.round(n, 2).tolist()), round(float(d), 2)) for n, d in zip(normals, offsets)]


def _dominant_plane(group: list[dict[str, Any]], surface: dict[str, Any]) -> dict[str, Any]:
    triangles = np.concatenate([item["triangles"] for item in group])
    areas = np.concatenate([item["triangle_areas"] for item in group])
    _, nearest = _closest(surface["scene"], triangles.mean(axis=1))
    keys = _plane_keys(surface["triangles"])
    votes: dict[tuple[tuple[float, ...], float], float] = {}
    for face_id, area in zip(nearest, areas):
        key = keys[face_id]
        votes[key] = votes.get(key, 0.0) + float(area)
    key = sorted(votes, key=lambda item: (-votes[item], item))[0]
    plane_faces = np.fromiter((item == key for item in keys), dtype=bool, count=len(keys))
    plane_triangles = surface["triangles"][plane_faces]
    normals = np.cross(plane_triangles[:, 1] - plane_triangles[:, 0], plane_triangles[:, 2] - plane_triangles[:, 0])
    areas = np.linalg.norm(normals, axis=1) * 0.5
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-15)
    axes = np.abs(normals).argmax(axis=1)
    flip = normals[np.arange(len(normals)), axes] < 0
    normals[flip] *= -1
    normal = np.average(normals, axis=0, weights=np.maximum(areas, 1e-12))
    normal /= np.linalg.norm(normal)
    axis = int(np.argmax(np.abs(normal)))
    if normal[axis] < 0:
        normal *= -1
    offset = float(np.average(plane_triangles.mean(axis=1) @ normal, weights=np.maximum(areas, 1e-12)))
    return {"normal": normal, "offset": offset, "key": key, "triangles": plane_triangles}


def _project_triangles(triangles: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    origin, u, v = _projector(normal, offset)
    return (np.asarray(triangles, dtype=np.float64) - origin) @ np.column_stack((u, v))


def _triangle_union(projected_triangles: np.ndarray):
    polygons = shapely.polygons(projected_triangles)
    valid = shapely.is_valid(polygons) & (shapely.area(polygons) > 1e-12)
    if not np.any(valid):
        raise ValueError("Projected red region contains no non-degenerate triangles")
    geometry = unary_union(polygons[valid])
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    if geometry.is_empty:
        raise ValueError("Projected red region is empty")
    return geometry


def _minimum_axis_aligned_box(geometry) -> tuple[tuple[float, float, float, float], float, float, float]:
    min_x, min_y, max_x, max_y = geometry.bounds
    area = float(geometry.area)
    perimeter = float(geometry.boundary.length)
    target = 0.95

    def coverage(bounds: np.ndarray) -> tuple[float, float]:
        x0, x1, y0, y1 = bounds
        if x1 <= x0 or y1 <= y0:
            return 0.0, 0.0
        rectangle = shapely.box(x0, y0, x1, y1)
        area_fraction = float(geometry.intersection(rectangle).area) / area
        contour_fraction = float(geometry.boundary.intersection(rectangle).length) / perimeter
        return area_fraction, contour_fraction

    def rectangle_area(bounds: np.ndarray) -> float:
        x0, x1, y0, y1 = bounds
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    def empty_area(bounds: np.ndarray) -> float:
        x0, x1, y0, y1 = bounds
        if x1 <= x0 or y1 <= y0:
            return float("inf")
        rectangle = shapely.box(x0, y0, x1, y1)
        return max(0.0, rectangle_area(bounds) - float(geometry.intersection(rectangle).area))

    initial = np.array([min_x, max_x, min_y, max_y], dtype=np.float64)
    starts = [initial]
    width, height = max_x - min_x, max_y - min_y
    for fraction in (0.01, 0.03, 0.05, 0.08):
        for index, direction, extent in ((0, 1, width), (1, -1, width),
                                         (2, 1, height), (3, -1, height)):
            candidate = initial.copy()
            candidate[index] += direction * extent * fraction
            if min(coverage(candidate)) >= target:
                starts.append(candidate)

    constraints = (
        {"type": "ineq", "fun": lambda b: coverage(b)[0] - target},
        {"type": "ineq", "fun": lambda b: coverage(b)[1] - target},
        {"type": "ineq", "fun": lambda b: b[1] - b[0] - 1e-9},
        {"type": "ineq", "fun": lambda b: b[3] - b[2] - 1e-9},
    )
    bounds = ((min_x, max_x), (min_x, max_x), (min_y, max_y), (min_y, max_y))
    best = initial
    best_empty_area = empty_area(initial)
    best_box_area = rectangle_area(initial)
    for start in starts:
        if min(coverage(start)) >= target - 1e-7:
            start_empty_area = empty_area(start)
            start_box_area = rectangle_area(start)
            if start_empty_area < best_empty_area - 1e-10 or (
                    abs(start_empty_area - best_empty_area) <= 1e-10 and start_box_area < best_box_area):
                best = start
                best_empty_area = start_empty_area
                best_box_area = start_box_area
        result = minimize(empty_area, start, method="SLSQP", bounds=bounds,
                          constraints=constraints, options={"ftol": 1e-10, "maxiter": 200})
        candidate = np.asarray(result.x, dtype=np.float64)
        candidate_empty_area = empty_area(candidate)
        candidate_box_area = rectangle_area(candidate)
        if min(coverage(candidate)) >= target - 1e-7 and (
                candidate_empty_area < best_empty_area - 1e-10 or
                (abs(candidate_empty_area - best_empty_area) <= 1e-10 and candidate_box_area < best_box_area)):
            best = candidate
            best_empty_area = candidate_empty_area
            best_box_area = candidate_box_area

    area_coverage, contour_coverage = coverage(best)
    if area_coverage < target - 1e-7 or contour_coverage < target - 1e-7:
        raise AssertionError("Axis-aligned box excludes more than 5% of the projected region")
    return tuple(float(value) for value in best), area_coverage, contour_coverage, float(best_empty_area)


def _find_contour_faces(vertices: np.ndarray, faces: np.ndarray, plane: dict[str, Any],
                        footprint) -> tuple[np.ndarray, float]:
    triangles = vertices[faces]
    centers = triangles.mean(axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-15)
    normal = plane["normal"]
    origin, u, v = _projector(normal, plane["offset"])
    plane_distance = np.abs((centers - origin) @ normal)
    candidate = (plane_distance <= PLANE_DISTANCE_LIMIT_M) & (np.abs(normals @ normal) >= 0.75)
    ids = np.flatnonzero(candidate)
    if not len(ids):
        return ids, MIN_CONTOUR_BAND_M

    projected_centers = (centers[ids] - origin) @ np.column_stack((u, v))
    min_x, min_y, max_x, max_y = footprint.bounds
    padding = MAX_BBOX_PADDING_M
    near = ((projected_centers[:, 0] >= min_x - padding) & (projected_centers[:, 0] <= max_x + padding) &
            (projected_centers[:, 1] >= min_y - padding) & (projected_centers[:, 1] <= max_y + padding))
    ids = ids[near]
    if not len(ids):
        return ids, MIN_CONTOUR_BAND_M

    projected = (triangles[ids] - origin) @ np.column_stack((u, v))
    edge_lengths = np.linalg.norm(projected - np.roll(projected, -1, axis=1), axis=2)
    typical_edge = float(np.median(edge_lengths.max(axis=1)))
    band = float(np.clip(typical_edge * 0.65, MIN_CONTOUR_BAND_M, MAX_CONTOUR_BAND_M))
    # shortcut: use local triangles in a projected 1–8 cm boundary band; refine if exact 3D contact matching is needed.
    zone = footprint.boundary.buffer(band)
    polygons = shapely.polygons(projected)
    max_edge = edge_lengths.max(axis=1)
    centroid = projected.mean(axis=1)
    centroid_distance = shapely.distance(shapely.points(centroid[:, 0], centroid[:, 1]), footprint.boundary)
    selected = (shapely.intersects(polygons, zone) &
                (max_edge <= max(0.05, 3.0 * band)) &
                (centroid_distance <= band + 0.5 * max_edge))
    return ids[selected], band


def _red_regions(alignment_dir: Path, red_ply: plyfile.PlyData, scans: dict[str, Any]) -> list[list[dict[str, Any]]]:
    vertices = _xyz(red_ply["vertex"].data)
    all_faces = _face_matrix(red_ply)
    marker = np.asarray(red_ply["face"].data["is_phase07_outlier"], dtype=bool)
    if marker.shape != (len(all_faces),) or not marker.any():
        raise ValueError("Final red object PLY has no marked outlier faces")

    regions = []
    face_offset = 0
    for scan_id in sorted(scans):
        object_paths = sorted((alignment_dir / "objects" / scan_id).rglob("*.ply"))
        for path in object_paths:
            mesh = trimesh.load(path, process=False)
            if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
                raise ValueError(f"Expected an aligned object mesh: {path}")
            stop = face_offset + len(mesh.faces)
            object_triangles = np.asarray(mesh.triangles, dtype=np.float32)
            red_triangles = vertices[all_faces[face_offset:stop]]
            if not np.array_equal(object_triangles, red_triangles):
                raise ValueError(f"Final red object face order differs from instance mesh: {path}")

            local_red = marker[face_offset:stop]
            adjacency_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
            adjacency = adjacency_mesh.face_adjacency
            for ids in _connected_regions(mesh.faces, local_red, adjacency):
                triangles = object_triangles[ids]
                triangle_areas = 0.5 * np.linalg.norm(
                    np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
                )
                regions.append({
                    "scan_id": scan_id,
                    "source": path.relative_to(alignment_dir / "objects" / scan_id).as_posix(),
                    "triangles": triangles,
                    "triangle_areas": triangle_areas,
                    "face_offset": face_offset,
                    "face_ids": ids,
                    "surface": _region_surface(triangles),
                })
            face_offset = stop

    if face_offset != len(all_faces):
        raise ValueError(f"Object instances cover {face_offset} of {len(all_faces)} final object faces")
    return _visible_groups(regions)


def _draw_geometry(ax, geometry, *, facecolor="none", edgecolor="#333333", linewidth=1.0, alpha=1.0, zorder=1):
    if isinstance(geometry, Polygon):
        parts = [geometry]
    elif isinstance(geometry, (MultiPolygon, GeometryCollection)):
        parts = [item for item in geometry.geoms if isinstance(item, Polygon)]
    else:
        parts = []
    for polygon in parts:
        patch = PolygonPatch(np.asarray(polygon.exterior.coords), closed=True, facecolor=facecolor,
                             edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, zorder=zorder)
        ax.add_patch(patch)
        for hole in polygon.interiors:
            ax.plot(*np.asarray(hole.coords).T, color=edgecolor, linewidth=linewidth, alpha=alpha, zorder=zorder)


def _draw_boundary(ax, geometry, color: str, linewidth: float, zorder: int) -> None:
    if isinstance(geometry, LineString):
        lines = [geometry]
    elif isinstance(geometry, (MultiLineString, GeometryCollection)):
        lines = [item for item in geometry.geoms if isinstance(item, LineString)]
    else:
        lines = []
    for line in lines:
        coordinates = np.asarray(line.coords)
        if len(coordinates) > 1:
            ax.plot(coordinates[:, 0], coordinates[:, 1], color=color, linewidth=linewidth, zorder=zorder)


def _render_region(path: Path, region_id: str, scan_id: str, plane: dict[str, Any], footprint,
                   plane_triangles: np.ndarray, lime_triangles: np.ndarray, contour_band: float,
                   lime_face_count: int, bbox: tuple[float, float, float, float],
                   area_coverage: float, contour_coverage: float, empty_area: float) -> None:
    projected_plane = _project_triangles(plane_triangles, plane["normal"], plane["offset"])
    projected_lime = _project_triangles(lime_triangles, plane["normal"], plane["offset"]) if len(lime_triangles) else np.empty((0, 3, 2))
    figure, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.add_collection(PolyCollection(projected_plane, facecolors="#e8eaed", edgecolors="none", zorder=0))
    _draw_geometry(ax, footprint, facecolor="#ff3b30", edgecolor="#b42318", linewidth=1.0, alpha=0.22, zorder=1)
    if len(projected_lime):
        ax.add_collection(PolyCollection(projected_lime, facecolors="#00ff00", edgecolors="none", alpha=0.9, zorder=2))
    boundary = footprint.boundary
    _draw_boundary(ax, boundary, "#00ff00", 2.2, 3)
    x0, x1, y0, y1 = bbox
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                           edgecolor="#ffae00", linewidth=2.2, linestyle="--", zorder=4))

    min_x, min_y, max_x, max_y = footprint.bounds
    width, height = max_x - min_x, max_y - min_y
    padding = max(0.15, 0.15 * max(width, height))
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    ax.set_aspect("equal", adjustable="box")
    axis = int(np.argmax(np.abs(plane["normal"])))
    axis_name = "XYZ"[axis]
    ax.set_title(f"{region_id} · {scan_id} · regularized {axis_name}={plane['offset']:.3f} m\n"
                 f"red {footprint.area:.3f} m² · lime faces {lime_face_count} · band {contour_band * 100:.1f} cm\n"
                 f"dashed box {((x1 - x0) * (y1 - y0)):.3f} m² · empty interior {empty_area:.3f} m²\n"
                 f"area retained {area_coverage:.1%} · contour retained {contour_coverage:.1%}")
    ax.set_xlabel("plane u (m)")
    ax.set_ylabel("plane v (m)")
    ax.grid(color="#6b7280", alpha=0.2, linewidth=0.5)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def _write_lime_copy(source: trimesh.Trimesh, selected_faces: np.ndarray, output_path: Path) -> None:
    vertices = np.asarray(source.vertices)
    faces = np.asarray(source.faces, dtype=np.int64)
    colors = np.asarray(source.visual.vertex_colors, dtype=np.uint8)
    if colors.shape != (len(vertices), 4):
        raise ValueError("Aligned sealed mesh must have RGBA vertex colors")
    green_used = np.zeros(len(vertices), dtype=bool)
    other_used = np.zeros(len(vertices), dtype=bool)
    green_used[np.unique(faces[selected_faces].reshape(-1))] = True
    other_used[np.unique(faces[~selected_faces].reshape(-1))] = True
    mixed = green_used & other_used
    out_colors = colors.copy()
    out_colors[green_used & ~other_used] = LIME
    duplicates = vertices[mixed].copy()
    duplicate_colors = np.broadcast_to(LIME, (int(mixed.sum()), 4)).copy()
    duplicate_map = np.full(len(vertices), -1, dtype=np.int64)
    duplicate_map[mixed] = len(vertices) + np.arange(int(mixed.sum()))
    out_faces = faces.copy()
    green_faces = out_faces[selected_faces]
    split = mixed[green_faces]
    green_faces[split] = duplicate_map[green_faces[split]]
    out_faces[selected_faces] = green_faces
    result = trimesh.Trimesh(vertices=np.concatenate((vertices, duplicates)), faces=out_faces, process=False)
    result.visual.vertex_colors = np.concatenate((out_colors, duplicate_colors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = output_path.with_name(output_path.stem + ".staging.ply")
    try:
        result.export(str(staging_path))
        check = trimesh.load(staging_path, process=False)
        if len(check.faces) != len(faces) or not np.array_equal(np.asarray(check.triangles), np.asarray(result.triangles)):
            raise AssertionError("Lime contour PLY changed sealed mesh geometry")
        check_colors = np.asarray(check.visual.vertex_colors, dtype=np.uint8)
        if not np.all(np.all(check_colors[np.asarray(check.faces)[selected_faces]] == LIME, axis=2)):
            raise AssertionError("A selected sealed contour face is not lime green")
        if not np.array_equal(check_colors[np.asarray(check.faces)[~selected_faces]], colors[faces[~selected_faces]]):
            raise AssertionError("A non-contour sealed face lost its original color")
        staging_path.replace(output_path)
    finally:
        staging_path.unlink(missing_ok=True)


def run(workspace: Path, alignment_dir: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    alignment_dir = Path(alignment_dir or workspace / "multi_room_alignment").resolve()
    output_dir = Path(output_dir or alignment_dir / "red_region_contours").resolve()
    red_path = alignment_dir / "aligned_objects_final_red.ply"
    sealed_path = alignment_dir / "aligned_sealed_meshes.ply"
    transform_path = alignment_dir / "stats" / "aligned_scans_transforms.json"
    for path in (red_path, sealed_path, transform_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    scans = ScanDataset(workspace).scans
    transforms = json.loads(transform_path.read_text())["scans"]
    red_ply = plyfile.PlyData.read(str(red_path), mmap="r", known_list_len={"face": {"vertex_indices": 3}})
    groups = _red_regions(alignment_dir, red_ply, scans)
    if not groups:
        raise ValueError("Final red object PLY contains no connected regions")

    sealed = trimesh.load(sealed_path, process=False)
    if not isinstance(sealed, trimesh.Trimesh):
        raise ValueError(f"Expected one combined aligned sealed mesh: {sealed_path}")
    sealed_vertices = np.asarray(sealed.vertices, dtype=np.float32)
    sealed_faces = np.asarray(sealed.faces, dtype=np.int64)
    selected_sealed_faces = np.zeros(len(sealed_faces), dtype=bool)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_image in output_dir.glob("red_region_*.png"):
        old_image.unlink()

    transforms_by_scan = {name: np.asarray(item["transform_matrix"], dtype=np.float64)
                          for name, item in transforms.items()}
    surfaces = {scan_id: _build_exterior_surface(scans[scan_id], transforms_by_scan[scan_id])
                for scan_id in {region["scan_id"] for group in groups for region in group}}

    sealed_offsets: dict[str, int] = {}
    scan_meshes: dict[str, trimesh.Trimesh] = {}
    face_offset = 0
    for scan_id in sorted(scans):
        scan_mesh_path = alignment_dir / "scans" / scan_id / "mesh.ply"
        if not scan_mesh_path.is_file():
            raise FileNotFoundError(scan_mesh_path)
        scan_mesh = trimesh.load(scan_mesh_path, process=False)
        if not isinstance(scan_mesh, trimesh.Trimesh):
            raise ValueError(f"Expected one aligned sealed mesh for {scan_id}: {scan_mesh_path}")
        scan_faces = np.asarray(scan_mesh.faces, dtype=np.int64)
        scan_vertices = np.asarray(scan_mesh.vertices, dtype=np.float32)
        scan_meshes[scan_id] = scan_mesh
        stop = face_offset + len(scan_faces)
        for start in range(face_offset, stop, TRIANGLE_CHUNK):
            end = min(start + TRIANGLE_CHUNK, stop)
            global_triangles = sealed_vertices[sealed_faces[start:end]]
            local_start = start - face_offset
            local_triangles = scan_vertices[scan_faces[local_start:local_start + (end - start)]]
            if not np.array_equal(global_triangles, local_triangles):
                raise ValueError(f"Aligned sealed mesh face order differs for {scan_id}")
        sealed_offsets[scan_id] = face_offset
        face_offset = stop
    if face_offset != len(sealed_faces):
        raise ValueError(f"Per-scan sealed meshes cover {face_offset} of {len(sealed_faces)} combined faces")

    region_summaries = []
    for index, group in enumerate(groups, 1):
        scan_id = group[0]["scan_id"]
        if any(region["scan_id"] != scan_id for region in group):
            raise AssertionError("A visible red region contains more than one scan")
        plane = _dominant_plane(group, surfaces[scan_id])
        triangles = np.concatenate([region["triangles"] for region in group])
        footprint = _triangle_union(_project_triangles(triangles, plane["normal"], plane["offset"]))
        bbox, area_coverage, contour_coverage, empty_area = _minimum_axis_aligned_box(footprint)
        scan_mesh = scan_meshes[scan_id]
        shell_face_ids, band = _find_contour_faces(
            np.asarray(scan_mesh.vertices, dtype=np.float32), np.asarray(scan_mesh.faces, dtype=np.int64), plane, footprint
        )
        if not len(shell_face_ids):
            raise ValueError(f"Could not map lime contour for red region {index:03d} ({scan_id}) onto sealed mesh")
        global_face_ids = sealed_offsets[scan_id] + shell_face_ids
        selected_sealed_faces[global_face_ids] = True
        region_id = f"red_region_{index:03d}_{scan_id}"
        image_path = output_dir / f"{region_id}.png"
        associated_plane = _plane_keys(surfaces[scan_id]["triangles"])
        plane_mask = np.fromiter((key == plane["key"] for key in associated_plane), dtype=bool,
                                 count=len(associated_plane))
        _render_region(image_path, region_id, scan_id, plane, footprint,
                       surfaces[scan_id]["triangles"][plane_mask],
                       np.asarray(scan_mesh.vertices)[np.asarray(scan_mesh.faces)[shell_face_ids]],
                       band, len(shell_face_ids), bbox, area_coverage, contour_coverage, empty_area)
        region_summaries.append({"region": region_id, "scan_id": scan_id,
                                 "red_components": len(group), "projected_area_m2": float(footprint.area),
                                 "lime_sealed_faces": int(len(shell_face_ids)), "contour_band_m": band,
                                 "axis_aligned_box_m": list(bbox),
                                 "box_area_m2": float((bbox[1] - bbox[0]) * (bbox[3] - bbox[2])),
                                 "empty_box_area_m2": empty_area,
                                 "area_retained_fraction": area_coverage,
                                 "contour_retained_fraction": contour_coverage,
                                 "image": str(image_path)})
        LOGGER.info("%s: %d object patches, %.3f m² projection, box %.3f m² with %.3f m² empty, %.1f%% area/%.1f%% contour retained, %d lime faces -> %s",
                    region_id, len(group), footprint.area,
                    (bbox[1] - bbox[0]) * (bbox[3] - bbox[2]),
                    empty_area, area_coverage * 100, contour_coverage * 100,
                    len(shell_face_ids), image_path)

    copy_path = output_dir / "aligned_sealed_meshes_lime_contours.ply"
    _write_lime_copy(sealed, selected_sealed_faces, copy_path)
    LOGGER.info("Exported lime-contour sealed mesh: %s (%d faces marked)",
                copy_path, int(selected_sealed_faces.sum()))
    return {"status": "success", "regions": region_summaries,
            "lime_sealed_faces": int(selected_sealed_faces.sum()), "output_mesh": str(copy_path)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("backend/current_scene"))
    parser.add_argument("--alignment-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    run(args.workspace, args.alignment_dir, args.output_dir)


if __name__ == "__main__":
    main()
