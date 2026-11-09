#!/usr/bin/env python3
"""Render top-down projections of every generated regularized and sealed floor mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import distance_transform_edt

from generate_floor_level_meshes import _read_mesh


DEFAULT_MESH_DIR = Path(__file__).resolve().parents[1] / "current_scene" / "artifacts" / "floor_level_meshes"
DEFAULT_RESOLUTION = 1024
RAY_BATCH_ROWS = 128
TRIANGLE_RASTER_BATCH_SIZE = 25_000
MARGIN_FRACTION = 0.03
SUPERSAMPLE_FACTOR = 2
GRAY_NEAR_PLANE = 32
GRAY_FAR_SURFACE = 240
SCAN_OUTLINE_THICKNESS_PX = 5
INNER_SCAN_OUTLINE_THICKNESS_PX = 2
SMALL_FOOTPRINT_HOLE_MAX_AREA_M2 = 0.005
HEIGHT_REGION_EDGE_THRESHOLD = 16
WHITE_FILL_THRESHOLD = 250
INPAINT_RADIUS_PX = 9
FOOTPRINT_CLOSE_KERNEL_PX = 41


def _projection_grid(vertices: np.ndarray, resolution: int) -> dict:
    bounds_min = vertices.min(axis=0).astype(np.float64)
    bounds_max = vertices.max(axis=0).astype(np.float64)
    horizontal_span = max(float(bounds_max[0] - bounds_min[0]), float(bounds_max[2] - bounds_min[2]), 1e-3)
    margin_pixels = max(1, int(round(resolution * MARGIN_FRACTION)))
    pixel_size = horizontal_span / (resolution - 2 * margin_pixels)
    x_center = (bounds_min[0] + bounds_max[0]) * 0.5
    z_center = (bounds_min[2] + bounds_max[2]) * 0.5
    image_span = pixel_size * resolution
    x_min = x_center - image_span * 0.5
    z_max = z_center + image_span * 0.5
    plane_y = float(bounds_max[1] + max(0.05, 0.01 * max(float(bounds_max[1] - bounds_min[1]), 0.1)))
    return {
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "pixel_size": pixel_size,
        "x_min": x_min,
        "z_max": z_max,
        "x_coordinates": x_min + (np.arange(resolution) + 0.5) * pixel_size,
        "z_coordinates": z_max - (np.arange(resolution) + 0.5) * pixel_size,
        "plane_y": plane_y,
        "depth_range": max(plane_y - float(bounds_min[1]), 1e-6),
    }


def _project_topdown(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    mode: str,
    resolution: int,
) -> tuple[np.ndarray, dict, np.ndarray]:
    if mode not in {"color", "height_grayscale"}:
        raise ValueError(f"Unknown projection mode: {mode}")
    if resolution < 64:
        raise ValueError("Projection resolution must be at least 64 pixels")

    grid = _projection_grid(vertices, resolution)
    render_resolution = resolution * SUPERSAMPLE_FACTOR
    subpixel_size = grid["pixel_size"] / SUPERSAMPLE_FACTOR
    x_coordinates = grid["x_min"] + (np.arange(render_resolution) + 0.5) * subpixel_size
    z_coordinates = grid["z_max"] - (np.arange(render_resolution) + 0.5) * subpixel_size
    plane_y = grid["plane_y"]
    depth_range = grid["depth_range"]

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(vertices.astype(np.float32, copy=False), dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(faces.astype(np.uint32, copy=False), dtype=o3d.core.Dtype.UInt32),
    )
    image = np.full((render_resolution, render_resolution, 3), 255, dtype=np.uint8)
    hit_mask = np.zeros((render_resolution, render_resolution), dtype=bool)
    direction = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    for row_start in range(0, render_resolution, RAY_BATCH_ROWS):
        row_stop = min(row_start + RAY_BATCH_ROWS, render_resolution)
        rows = row_stop - row_start
        origins = np.empty((rows, render_resolution, 3), dtype=np.float32)
        origins[:, :, 0] = x_coordinates[None, :]
        origins[:, :, 1] = plane_y
        origins[:, :, 2] = z_coordinates[row_start:row_stop, None]
        rays = np.empty((rows, render_resolution, 6), dtype=np.float32)
        rays[:, :, :3] = origins
        rays[:, :, 3:] = direction
        hits = scene.cast_rays(o3d.core.Tensor(rays))
        primitive_ids = hits["primitive_ids"].numpy().reshape(-1)
        hit_distance = hits["t_hit"].numpy().reshape(-1)
        valid = (primitive_ids < len(faces)) & np.isfinite(hit_distance)
        hit_mask[row_start:row_stop] = valid.reshape(rows, render_resolution)
        if not np.any(valid):
            continue

        output = image[row_start:row_stop].reshape(-1, 3)
        if mode == "height_grayscale":
            depth_fraction = np.clip(hit_distance[valid] / depth_range, 0.0, 1.0)
            shade = np.rint(
                GRAY_NEAR_PLANE + depth_fraction * (GRAY_FAR_SURFACE - GRAY_NEAR_PLANE)
            ).astype(np.uint8)
            output[valid] = shade[:, None]
        else:
            ids = primitive_ids[valid].astype(np.int64)
            uv = hits["primitive_uvs"].numpy().reshape(-1, 2)[valid]
            weights = np.column_stack((1.0 - uv.sum(axis=1), uv))
            face_colors = colors[faces[ids], :3].astype(np.float32)
            rgb = np.einsum("ij,ijk->ik", weights, face_colors)
            output[valid] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)

    sample_colors = image.reshape(
        resolution, SUPERSAMPLE_FACTOR, resolution, SUPERSAMPLE_FACTOR, 3,
    ).transpose(0, 2, 1, 3, 4).reshape(resolution, resolution, -1, 3)
    sample_hits = hit_mask.reshape(
        resolution, SUPERSAMPLE_FACTOR, resolution, SUPERSAMPLE_FACTOR,
    ).transpose(0, 2, 1, 3).reshape(resolution, resolution, -1)
    hit_count = sample_hits.sum(axis=2)
    image = np.full((resolution, resolution, 3), 255, dtype=np.uint8)
    occupied = hit_count > 0
    color_sum = (sample_colors * sample_hits[..., None]).sum(axis=2, dtype=np.uint32)
    image[occupied] = np.rint(color_sum[occupied] / hit_count[occupied, None]).astype(np.uint8)

    return image, {
        "view": "orthographic_top_down_from_positive_y",
        "mode": mode,
        "resolution_px": [resolution, resolution],
        "supersample_factor": SUPERSAMPLE_FACTOR,
        "projection_plane_y_m": plane_y,
        "height_encoding_range_m": depth_range if mode == "height_grayscale" else None,
        "height_encoding_rgb_range": [GRAY_NEAR_PLANE, GRAY_FAR_SURFACE] if mode == "height_grayscale" else None,
        "pixel_size_m": float(grid["pixel_size"]),
        "mesh_bounds_min_xyz": grid["bounds_min"].tolist(),
        "mesh_bounds_max_xyz": grid["bounds_max"].tolist(),
        "background_rgb": [255, 255, 255],
    }, occupied


def _fill_small_footprint_holes(mask: np.ndarray, pixel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Close only tiny enclosed raster holes, leaving open gaps between scans untouched."""
    max_area_px = SMALL_FOOTPRINT_HOLE_MAX_AREA_M2 / max(pixel_size * pixel_size, 1e-12)
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    holes = np.zeros_like(mask)
    for index, contour in enumerate(contours):
        depth, parent = 0, int(hierarchy[0, index, 3])
        while parent >= 0:
            depth += 1
            parent = int(hierarchy[0, parent, 3])
        if depth % 2 and cv2.contourArea(contour) <= max_area_px:
            cv2.drawContours(holes, [contour], -1, 255, cv2.FILLED)
    added = (holes > 0) & (mask == 0)
    return cv2.bitwise_or(mask, holes), added


def _draw_scan_outlines(
    vertices: np.ndarray,
    faces: np.ndarray,
    scan_records: list[dict],
    mesh_type: str,
    resolution: int,
) -> tuple[list[str], np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    grid = _projection_grid(vertices, resolution)
    outlined_scans = []
    footprint = np.zeros((resolution, resolution), dtype=np.uint8)
    regularized_footprint = np.zeros_like(footprint)
    regularized_scan_masks = []
    outer_contours = []
    inner_contours = []
    small_holes = np.zeros_like(footprint)
    face_offset = 0
    close_kernel = None
    if mesh_type == "sealed":
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (FOOTPRINT_CLOSE_KERNEL_PX, FOOTPRINT_CLOSE_KERNEL_PX),
        )
    for scan in scan_records:
        face_count = int(scan[mesh_type]["output_triangles"])
        face_stop = face_offset + face_count
        mask = np.zeros((resolution, resolution), dtype=np.uint8)
        scan_faces = faces[face_offset:face_stop]
        for start in range(0, len(scan_faces), TRIANGLE_RASTER_BATCH_SIZE):
            triangles = vertices[scan_faces[start:start + TRIANGLE_RASTER_BATCH_SIZE]]
            x_pixels = np.rint((triangles[:, :, 0] - grid["x_min"]) / grid["pixel_size"] - 0.5)
            z_pixels = np.rint((grid["z_max"] - triangles[:, :, 2]) / grid["pixel_size"] - 0.5)
            polygons = np.stack((x_pixels, z_pixels), axis=2).astype(np.int32)
            cv2.fillPoly(mask, [polygon.reshape(-1, 1, 2) for polygon in polygons], 255)
        if mesh_type == "regularized":
            mask, filled_holes = _fill_small_footprint_holes(mask, float(grid["pixel_size"]))
            small_holes[filled_holes] = 255
        retrieval = cv2.RETR_TREE if mesh_type == "regularized" else cv2.RETR_EXTERNAL
        contours, hierarchy = cv2.findContours(mask, retrieval, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            if mesh_type == "sealed":
                scan_footprint = np.zeros_like(mask)
                cv2.drawContours(scan_footprint, contours, -1, 255, cv2.FILLED)
                # Closing the combined footprint could bridge scans and color the gap between them.
                scan_footprint = cv2.morphologyEx(scan_footprint, cv2.MORPH_CLOSE, close_kernel)
                cv2.bitwise_or(footprint, scan_footprint, dst=footprint)
                outer_contours.extend(contours)
            else:
                cv2.bitwise_or(regularized_footprint, mask, dst=regularized_footprint)
                regularized_scan_masks.append(mask)
            outlined_scans.append(scan["scan_id"])
        face_offset = face_stop
    if face_offset != len(faces):
        raise ValueError(f"Per-scan {mesh_type} triangle counts do not match the merged PLI")
    scan_interfaces = np.zeros_like(footprint)
    if mesh_type == "regularized":
        owners = np.zeros_like(footprint, dtype=np.uint16)
        for index, mask in enumerate(regularized_scan_masks, start=1):
            owners[(owners == 0) & (mask > 0)] = index
        horizontal = (owners[:, :-1] > 0) & (owners[:, 1:] > 0) & (owners[:, :-1] != owners[:, 1:])
        vertical = (owners[:-1, :] > 0) & (owners[1:, :] > 0) & (owners[:-1, :] != owners[1:, :])
        scan_interfaces[:, :-1] |= horizontal
        scan_interfaces[:, 1:] |= horizontal
        scan_interfaces[:-1, :] |= vertical
        scan_interfaces[1:, :] |= vertical
        contours, hierarchy = cv2.findContours(
            regularized_footprint, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
        )
        for index, contour in enumerate(contours):
            depth = 0
            parent = int(hierarchy[0, index, 3])
            while parent >= 0:
                depth += 1
                parent = int(hierarchy[0, parent, 3])
            if depth % 2 == 0:
                outer_contours.append(contour)
            elif cv2.contourArea(contour) > (
                    SMALL_FOOTPRINT_HOLE_MAX_AREA_M2 / max(grid["pixel_size"] ** 2, 1e-12)):
                inner_contours.append(contour)
    return outlined_scans, footprint > 0, outer_contours, inner_contours, scan_interfaces, small_holes


def _inpaint_contour_gaps(
    image: np.ndarray,
    footprint: np.ndarray,
    occupied: np.ndarray,
) -> dict:
    ray_misses = footprint & ~occupied
    white_pixels = footprint & np.all(image >= WHITE_FILL_THRESHOLD, axis=2)
    repair_mask = ray_misses | white_pixels
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(repair_mask.astype(np.uint8), 8)
    known = footprint & occupied & ~white_pixels
    if repair_mask.any() and known.any():
        nearest = distance_transform_edt(~known, return_distances=False, return_indices=True)
        border_extended = image[nearest[0], nearest[1]]
        filled = cv2.inpaint(
            border_extended,
            repair_mask.astype(np.uint8) * 255,
            INPAINT_RADIUS_PX,
            cv2.INPAINT_TELEA,
        )
        image[repair_mask] = filled[repair_mask]

    return {
        "detected_white_regions": max(0, int(component_count) - 1),
        "detected_white_pixels": int(repair_mask.sum()),
        "ray_miss_pixels": int(ray_misses.sum()),
        "near_white_hit_pixels": int((white_pixels & occupied).sum()),
        "largest_white_region_px": int(stats[1:, cv2.CC_STAT_AREA].max()) if component_count > 1 else 0,
        "inpainted_pixels": int(repair_mask.sum()) if known.any() else 0,
        "near_white_threshold_rgb": WHITE_FILL_THRESHOLD,
        "inpainting_radius_px": INPAINT_RADIUS_PX,
        "footprint_close_kernel_px": FOOTPRINT_CLOSE_KERNEL_PX,
        "inpainting_method": "telea_from_nearest_projected_border_colors",
    }


def generate_floor_level_projections(
    mesh_dir: Path = DEFAULT_MESH_DIR,
    output_dir: Path | None = None,
    resolution: int = DEFAULT_RESOLUTION,
) -> dict:
    mesh_dir = Path(mesh_dir).resolve()
    output_dir = Path(output_dir or mesh_dir.parent / "floor_level_projections").resolve()
    manifest_path = mesh_dir / "floor_level_meshes_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Floor mesh manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    projections = []
    for level in manifest["levels"]:
        for kind, mode in (("sealed", "color"), ("regularized", "height_grayscale")):
            mesh_path = Path(level[f"{kind}_mesh"])
            if not mesh_path.is_file():
                raise FileNotFoundError(f"Floor mesh is missing: {mesh_path}")
            vertices, faces, colors = _read_mesh(mesh_path)
            image, details, occupied = _project_topdown(vertices, faces, colors, mode, resolution)
            (outlined_scans, footprint, outer_contours, inner_contours,
             scan_interfaces, small_holes) = _draw_scan_outlines(
                vertices, faces, level["scans"], kind, resolution,
            )
            gap_filling = (
                _inpaint_contour_gaps(image, footprint, occupied)
                if kind == "sealed" else {"applied": False}
            )
            if kind == "regularized":
                repair = (small_holes > 0) & ~occupied
                if repair.any() and occupied.any():
                    filled = cv2.inpaint(
                        image, repair.astype(np.uint8) * 255,
                        INPAINT_RADIUS_PX, cv2.INPAINT_TELEA,
                    )
                    image[repair] = filled[repair]
                gray = image[:, :, 0].astype(np.int16)
                region_edges = np.zeros_like(occupied)
                horizontal = (
                    (np.abs(gray[:, 1:] - gray[:, :-1]) >= HEIGHT_REGION_EDGE_THRESHOLD)
                    & occupied[:, 1:] & occupied[:, :-1]
                )
                vertical = (
                    (np.abs(gray[1:, :] - gray[:-1, :]) >= HEIGHT_REGION_EDGE_THRESHOLD)
                    & occupied[1:, :] & occupied[:-1, :]
                )
                region_edges[:, 1:] |= horizontal
                region_edges[:, :-1] |= horizontal
                region_edges[1:, :] |= vertical
                region_edges[:-1, :] |= vertical
                image[region_edges] = (0, 0, 0)
                image[scan_interfaces] = (0, 0, 0)
            cv2.drawContours(image, outer_contours, -1, (0, 0, 0), SCAN_OUTLINE_THICKNESS_PX, cv2.LINE_8)
            if inner_contours:
                cv2.drawContours(
                    image, inner_contours, -1, (0, 0, 0), INNER_SCAN_OUTLINE_THICKNESS_PX, cv2.LINE_8,
                )
            output_path = output_dir / f"{mesh_path.stem}_topdown.png"
            if not cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                raise OSError(f"Could not write top-down projection: {output_path}")
            projections.append({
                "floor_level_name": level["floor_level_name"],
                "mesh_type": kind,
                "outlined_scan_ids": outlined_scans,
                "scan_outline_thickness_px": SCAN_OUTLINE_THICKNESS_PX,
                "inner_scan_outline_thickness_px": INNER_SCAN_OUTLINE_THICKNESS_PX if kind == "regularized" else None,
                "contour_gap_filling": gap_filling,
                "source_mesh": str(mesh_path),
                "projection_image": str(output_path),
                **details,
            })

    result = {"schema_version": 1, "source_manifest": str(manifest_path), "projections": projections}
    result_path = output_dir / "floor_level_projections_manifest.json"
    temporary_path = result_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(result_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR,
                        help="Directory containing floor_level_meshes_manifest.json and its PLYs.")
    parser.add_argument("--output-dir", type=Path,
                        help="Output directory; defaults to current_scene/artifacts/floor_level_projections.")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION,
                        help="Square projection resolution in pixels (default: 1024).")
    args = parser.parse_args()
    generate_floor_level_projections(args.mesh_dir, args.output_dir, args.resolution)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
