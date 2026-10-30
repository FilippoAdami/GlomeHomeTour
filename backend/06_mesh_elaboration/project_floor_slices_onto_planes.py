#!/usr/bin/env python3
"""Vertically collapse every floor scene slice onto its fitted plane and render PNGs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.ticker import MultipleLocator
import cv2
import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData
from scipy.ndimage import binary_closing, binary_dilation, binary_opening, distance_transform_edt, find_objects, gaussian_filter, label
import trimesh

FLOOR_ROLE = 0
PLANE_FACES_PER_BATCH = 100_000
VERTICAL_TRIANGLE_TOLERANCE_DEG = 20.0
VERTICAL_NORMAL_Y_THRESHOLD = float(np.sin(np.deg2rad(VERTICAL_TRIANGLE_TOLERANCE_DEG)))
BLUR_SIGMA_PX = 13.0
SHARPEN_TURNS = 7
BLACK_MASK_MAX_CHANNEL = 24
STRIP_OPENING_RADIUS_PX = 6
MIN_BLACK_COMPONENT_AREA_PX = 128
MAX_THIN_COMPONENT_WIDTH_PX = 24
MIN_THIN_COMPONENT_ASPECT_RATIO = 8.0
CLEANUP_EDGE_MARGIN_PX = 8
NEUTRAL_BACKGROUND_MAX_SPREAD = 16
MORPHOLOGY_COMPONENT_AREA_FRACTION = 0.005
MORPHOLOGY_MIN_FEATURE_WIDTH_M = 0.035
MORPHOLOGY_MIN_COMPONENT_AREA_M2 = 0.001
SIGNED_DISTANCE_SMOOTH_SIGMA_M = 0.006
WALL_CONTOUR_HEIGHT_M = 0.30
WALL_CONTOUR_COLOR = (255, 0, 0)
WALL_CONTOUR_WIDTH_PX = 3
HOLE_MIN_SIDE_M = 0.45
HOLE_MIN_LONG_SIDE_M = 0.90
HOLE_MAX_BLACK_FRACTION = 0.05
WALL_CONTOUR_CLOSE_GAP_M = 0.07
HOLE_FILL_RGBA = (20, 100, 255, 90)
HOLE_OUTLINE_RGBA = (0, 65, 220, 235)


def _plane_frame(equation: list[float], description: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plane = np.asarray(equation, dtype=np.float64)
    if plane.shape != (4,) or not np.isfinite(plane).all():
        raise ValueError(f"Invalid {description} plane equation")
    length = float(np.linalg.norm(plane[:3]))
    if length <= 1e-12:
        raise ValueError(f"{description.capitalize()} plane has a zero-length normal")
    plane /= length
    if abs(plane[1]) <= 1e-8:
        raise ValueError(f"{description.capitalize()} plane is vertical and cannot receive a vertical projection")
    if plane[1] < 0:
        plane = -plane
    axis_u = np.array([1.0, -plane[0] / plane[1], 0.0])
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(axis_u, plane[:3])
    axis_v /= np.linalg.norm(axis_v)
    return plane, axis_u, axis_v


def _collapse_to_plane(vertices: np.ndarray, plane: np.ndarray,
                       axis_u: np.ndarray, axis_v: np.ndarray) -> np.ndarray:
    collapsed = np.asarray(vertices, dtype=np.float64).copy()
    collapsed[:, 1] = -(plane[0] * collapsed[:, 0] + plane[2] * collapsed[:, 2] + plane[3]) / plane[1]
    return np.column_stack((collapsed @ axis_u, collapsed @ axis_v))


def _retain_nonvertical_source_faces(vertices: np.ndarray, triangles: np.ndarray,
                                     face_ids: np.ndarray) -> tuple[np.ndarray, int]:
    points = vertices[triangles[face_ids]]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    vertical_fraction = np.divide(
        np.abs(normals[:, 1]), lengths, out=np.zeros_like(lengths), where=lengths > 1e-12
    )
    keep = (lengths <= 1e-12) | (vertical_fraction > VERTICAL_NORMAL_Y_THRESHOLD)
    return face_ids[keep], int((~keep).sum())


def _load_slice(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    ply = PlyData.read(str(path), known_list_len={"face": {"vertex_indices": 3}})
    vertex_data, face_data = ply["vertex"].data, ply["face"].data
    vertices = np.column_stack([vertex_data[name] for name in ("x", "y", "z")]).astype(np.float64)
    triangles = np.asarray(face_data["vertex_indices"], dtype=np.int64).reshape(-1, 3)
    required = ("is_fitted_plane", "surface_role", "floor_area_id")
    if any(name not in face_data.dtype.names for name in required):
        raise ValueError(f"Scene slice is missing projection face attributes: {path}")
    attributes = {name: face_data[name].copy() for name in required}
    if (not len(vertices) or not len(triangles) or not np.isfinite(vertices).all()
            or triangles.min() < 0 or triangles.max() >= len(vertices)
            or any(len(values) != len(triangles) for values in attributes.values())):
        raise ValueError(f"Scene slice is empty, non-finite, or has invalid face attributes: {path}")
    return vertices, triangles, attributes


def _load_wall_shell(instances_dir: Path) -> trimesh.Trimesh | None:
    manifest_path = instances_dir / "wall_planes_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    planes = manifest.get("wall_planes") if isinstance(manifest, dict) else None
    if not isinstance(planes, list):
        raise ValueError("Wall plane manifest must contain a wall_planes list")
    if not planes:
        return None
    shell_path = instances_dir / "wall_enclosing_shell.ply"
    if not shell_path.is_file():
        raise ValueError(f"Fitted wall shell is missing: {shell_path}")
    mesh = trimesh.load(shell_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError(f"Fitted wall shell is empty or invalid: {shell_path}")
    return mesh


def _wall_contour_uv(wall_shell: trimesh.Trimesh | None, plane: np.ndarray,
                     axis_u: np.ndarray, axis_v: np.ndarray) -> np.ndarray:
    if wall_shell is None:
        return np.empty((0, 2, 2), dtype=np.float64)
    offset_plane = plane.copy()
    offset_plane[3] -= WALL_CONTOUR_HEIGHT_M
    origin = -offset_plane[3] * offset_plane[:3]
    segments = trimesh.intersections.mesh_plane(wall_shell, offset_plane[:3], origin)
    if not len(segments):
        return np.empty((0, 2, 2), dtype=np.float64)
    return np.stack((segments @ axis_u, segments @ axis_v), axis=-1)


def _blur_sharpen_plot(image: Image.Image, axis_bbox: tuple[int, int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = axis_bbox
    pixels = np.asarray(image, dtype=np.float32).copy()
    margin = CLEANUP_EDGE_MARGIN_PX
    region = pixels[y0 + margin:y1 - margin, x0 + margin:x1 - margin]
    original = region.copy()
    foreground = np.max(region, axis=2) <= BLACK_MASK_MAX_CHANNEL
    channel_spread = np.ptp(original, axis=2)
    background = (~foreground) & (original.min(axis=2) >= 128) & (
        channel_spread <= NEUTRAL_BACKGROUND_MAX_SPREAD
    )
    refined = gaussian_filter(foreground.astype(np.float32), sigma=BLUR_SIGMA_PX, mode="nearest")
    for _ in range(SHARPEN_TURNS):
        blurred = gaussian_filter(refined, sigma=BLUR_SIGMA_PX, mode="nearest")
        refined = np.clip(refined + (refined - blurred), 0.0, 1.0)

    refined_foreground = (refined >= 0.5) & (foreground | background)
    removed = foreground & ~refined_foreground
    if removed.any():
        if background.any():
            nearest = distance_transform_edt(
                ~background, return_distances=False, return_indices=True
            )
            region[removed] = original[nearest[0][removed], nearest[1][removed]]
    region[refined_foreground] = 0
    return Image.fromarray(np.rint(pixels).astype(np.uint8))


def _regularize_black_mask(image: Image.Image, axis_bbox: tuple[int, int, int, int],
                           pixels_per_meter: float) -> tuple[Image.Image, dict]:
    x0, y0, x1, y1 = axis_bbox
    pixels = np.asarray(image).copy()
    margin = CLEANUP_EDGE_MARGIN_PX
    region = pixels[y0 + margin:y1 - margin, x0 + margin:x1 - margin]
    original = region.copy()
    foreground = np.max(original, axis=2) <= BLACK_MASK_MAX_CHANNEL
    channel_spread = np.ptp(original, axis=2)
    background = (~foreground) & (original.min(axis=2) >= 128) & (
        channel_spread <= NEUTRAL_BACKGROUND_MAX_SPREAD
    )

    radius = max(1, int(np.ceil(MORPHOLOGY_MIN_FEATURE_WIDTH_M * pixels_per_meter / 2.0)))
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    opened = binary_opening(foreground, structure=xx * xx + yy * yy <= radius * radius)
    components, component_count = label(opened, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(components.ravel(), minlength=component_count + 1)
    largest_component = int(sizes[1:].max()) if component_count else 0
    minimum_component_area = max(
        MIN_BLACK_COMPONENT_AREA_PX,
        int(np.ceil(MORPHOLOGY_MIN_COMPONENT_AREA_M2 * pixels_per_meter ** 2)),
        int(np.ceil(largest_component * MORPHOLOGY_COMPONENT_AREA_FRACTION)),
    )
    keep = sizes >= minimum_component_area
    keep[0] = False
    filtered = keep[components]

    if filtered.any():
        signed_distance = distance_transform_edt(filtered) - distance_transform_edt(~filtered)
        sigma_px = max(0.5, SIGNED_DISTANCE_SMOOTH_SIGMA_M * pixels_per_meter)
        smoothed = gaussian_filter(
            signed_distance, sigma=sigma_px, mode="nearest"
        ) > 0
    else:
        sigma_px = max(0.5, SIGNED_DISTANCE_SMOOTH_SIGMA_M * pixels_per_meter)
        smoothed = filtered
    refined = smoothed & (foreground | background)
    removed = foreground & ~refined
    if removed.any():
        if background.any():
            nearest = distance_transform_edt(
                ~background, return_distances=False, return_indices=True
            )
            region[removed] = original[nearest[0][removed], nearest[1][removed]]
        else:
            region[removed] = 255
    region[refined] = 0
    return Image.fromarray(pixels), {
        "black_pixels_before": int(foreground.sum()),
        "black_pixels_after": int(refined.sum()),
        "components_before_area_filter": int(component_count),
        "components_after_area_filter": int(keep[1:].sum()),
        "largest_component_area_px": largest_component,
        "minimum_component_area_px": minimum_component_area,
        "pixels_per_meter": pixels_per_meter,
        "minimum_feature_width_m": MORPHOLOGY_MIN_FEATURE_WIDTH_M,
        "minimum_component_area_m2": MORPHOLOGY_MIN_COMPONENT_AREA_M2,
        "opening_radius_px": radius,
        "signed_distance_sigma_m": SIGNED_DISTANCE_SMOOTH_SIGMA_M,
        "signed_distance_sigma_px": sigma_px,
        "black_pixels_removed": int(removed.sum()),
    }


def _clean_black_artifacts(image: Image.Image, axis_bbox: tuple[int, int, int, int]) -> tuple[Image.Image, int]:
    x0, y0, x1, y1 = axis_bbox
    pixels = np.asarray(image).copy()
    margin = CLEANUP_EDGE_MARGIN_PX
    region = pixels[y0 + margin:y1 - margin, x0 + margin:x1 - margin]
    black = np.max(region, axis=2) <= BLACK_MASK_MAX_CHANNEL
    radius = STRIP_OPENING_RADIUS_PX
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    opened = binary_opening(black, structure=xx * xx + yy * yy <= radius * radius)
    components, count = label(opened, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    keep = sizes >= MIN_BLACK_COMPONENT_AREA_PX
    keep[0] = False
    for component_id, bounds in enumerate(find_objects(components), start=1):
        if bounds is None:
            continue
        height = bounds[0].stop - bounds[0].start
        width = bounds[1].stop - bounds[1].start
        short, long = sorted((height, width))
        if short <= MAX_THIN_COMPONENT_WIDTH_PX and long / short >= MIN_THIN_COMPONENT_ASPECT_RATIO:
            keep[component_id] = False
    clean = keep[components]
    removed = black & ~clean
    removed_count = int(removed.sum())
    if removed_count:
        channel_spread = np.ptp(region, axis=2)
        neutral_background = (~black) & (region.min(axis=2) >= 128) & (
            channel_spread <= NEUTRAL_BACKGROUND_MAX_SPREAD
        )
        if neutral_background.any():
            nearest_background = distance_transform_edt(
                ~neutral_background, return_distances=False, return_indices=True
            )
            region[removed] = region[nearest_background[0][removed], nearest_background[1][removed]]
        else:
            region[removed] = 255
    return Image.fromarray(pixels), removed_count


def _draw_wall_contour(image: Image.Image, contour_uv: np.ndarray, ax,
                       axis_bbox: tuple[int, int, int, int]) -> Image.Image:
    if not len(contour_uv):
        return image
    _, height = image.size
    x0, y0, _, _ = axis_bbox
    crop = image.crop(axis_bbox)
    draw = ImageDraw.Draw(crop)
    for segment in contour_uv:
        points = ax.transData.transform(segment)
        draw.line([(float(x - x0), float(height - y - y0)) for x, y in points],
                  fill=WALL_CONTOUR_COLOR, width=WALL_CONTOUR_WIDTH_PX)
    image.paste(crop, (x0, y0))
    return image


def _largest_true_rectangle(mask: np.ndarray, pixels_per_meter_x: float,
                            pixels_per_meter_y: float) -> tuple[int, int, int, int] | None:
    """Return the largest axis-aligned all-true rectangle as (top, left, bottom, right)."""
    height, width = mask.shape
    heights = np.zeros(width + 1, dtype=np.int32)
    best = None
    best_area = 0
    for row in range(height):
        heights[:-1] = np.where(mask[row], heights[:-1] + 1, 0)
        stack: list[int] = []
        for col in range(width + 1):
            current = int(heights[col])
            while stack and heights[stack[-1]] > current:
                rectangle_height = int(heights[stack.pop()])
                left = stack[-1] + 1 if stack else 0
                rectangle_width = col - left
                valid_dimensions = (
                    rectangle_width / pixels_per_meter_x >= HOLE_MIN_SIDE_M
                    and rectangle_height / pixels_per_meter_y >= HOLE_MIN_LONG_SIDE_M
                ) or (
                    rectangle_width / pixels_per_meter_x >= HOLE_MIN_LONG_SIDE_M
                    and rectangle_height / pixels_per_meter_y >= HOLE_MIN_SIDE_M
                )
                area = rectangle_width * rectangle_height
                if valid_dimensions and area > best_area:
                    best = (row - rectangle_height + 1, left, row + 1, col)
                    best_area = area
            stack.append(col)
    return best


def _integral_sum(integral: np.ndarray, top: int, left: int, bottom: int, right: int) -> int:
    return int(integral[bottom, right] - integral[top, right] - integral[bottom, left] + integral[top, left])


def _wall_contour_masks(image: Image.Image, pixels_per_meter_x: float,
                        pixels_per_meter_y: float) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray(image.convert("RGB"))
    red_contour = (pixels[:, :, 0] >= 180) & (pixels[:, :, 1] <= 120) & (pixels[:, :, 2] <= 120)
    close_size = max(3, int(np.ceil(WALL_CONTOUR_CLOSE_GAP_M * min(
        pixels_per_meter_x, pixels_per_meter_y
    ))) | 1)
    closed_contour = binary_closing(red_contour, structure=np.ones((close_size, close_size), dtype=bool))
    contour_barrier = binary_dilation(closed_contour, structure=np.ones((3, 3), dtype=bool))
    open_pixels = ~contour_barrier
    components, _ = label(open_pixels, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool))
    outside_labels = np.unique(np.concatenate((components[0], components[-1], components[:, 0], components[:, -1])))
    inside = open_pixels & ~np.isin(components, outside_labels)
    return red_contour, inside


def _wall_alignment_transform(inside: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int], float]:
    # shortcut: use one dominant angle per floor area; use per-wall frames for non-orthogonal layouts.
    contours, _ = cv2.findContours(inside.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Wall contour does not enclose an area for hole fitting")
    contour = max(contours, key=cv2.contourArea)
    corners = cv2.approxPolyDP(contour, 2.0, True).reshape(-1, 2).astype(float)
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    orientation = float(np.angle(np.sum(lengths * np.exp(4j * angles))) / 4.0)
    angle_deg = float(np.rad2deg(orientation))

    height, width = inside.shape
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle_deg, 1.0)
    image_corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=float)
    rotated_corners = cv2.transform(image_corners[None, ...].astype(np.float32), matrix)[0]
    minimum = np.floor(rotated_corners.min(axis=0))
    maximum = np.ceil(rotated_corners.max(axis=0))
    matrix[:, 2] -= minimum
    output_size = (int(maximum[0] - minimum[0] + 1), int(maximum[1] - minimum[1] + 1))
    inverse = cv2.invertAffineTransform(matrix)
    return matrix, inverse, output_size, angle_deg


def _expand_rectangle_to_nearby_wall(bounds: tuple[int, int, int, int], available: np.ndarray,
                                     inside: np.ndarray, pixels_per_meter_x: float,
                                     pixels_per_meter_y: float) -> tuple[tuple[int, int, int, int], list[str]]:
    """Close sub-10 cm gaps by extending a fitted rectangle to the contour boundary."""
    top, left, bottom, right = bounds
    snapped = []
    limits = {
        "top": int(np.ceil(0.10 * pixels_per_meter_y)),
        "bottom": int(np.ceil(0.10 * pixels_per_meter_y)),
        "left": int(np.ceil(0.10 * pixels_per_meter_x)),
        "right": int(np.ceil(0.10 * pixels_per_meter_x)),
    }
    for edge in ("top", "bottom", "left", "right"):
        limit = limits[edge]
        last = [top, left, bottom, right]
        hit_wall = False
        for distance in range(1, limit + 2):
            candidate = last.copy()
            candidate[{"top": 0, "left": 1, "bottom": 2, "right": 3}[edge]] += (
                -1 if edge in ("top", "left") else 1
            )
            y0, x0, y1, x1 = candidate
            if not inside[y0:y1, x0:x1].all():
                hit_wall = distance <= limit + 1
                break
            if not available[y0:y1, x0:x1].all():
                break
            last = candidate
        if hit_wall and last != [top, left, bottom, right]:
            top, left, bottom, right = last
            snapped.append(edge)
    return (top, left, bottom, right), snapped


def _fit_wall_gap_rectangles(image: Image.Image, axis_bbox: tuple[int, int, int, int],
                             pixels_per_meter_x: float, pixels_per_meter_y: float,
                             max_black_fraction: float = HOLE_MAX_BLACK_FRACTION) -> tuple[list[dict], dict]:
    """Pack rectangles by net white-pixel gain inside the red wall contour."""
    pixels = np.asarray(image.convert("RGB"))
    _, source_inside = _wall_contour_masks(image, pixels_per_meter_x, pixels_per_meter_y)
    matrix, inverse, aligned_size, alignment_deg = _wall_alignment_transform(source_inside)
    inside = cv2.warpAffine(source_inside.astype(np.uint8), matrix, aligned_size,
                            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT).astype(bool)
    source_black = np.max(pixels, axis=2) <= BLACK_MASK_MAX_CHANNEL
    black = cv2.warpAffine(source_black.astype(np.uint8), matrix, aligned_size,
                           flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT).astype(bool)
    available = inside.copy()
    black_integral = np.pad(black.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    white_integral = np.pad((~black).astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    rectangles = []

    def sum_rect(integral: np.ndarray, top, left, bottom, right):
        return integral[bottom, right] - integral[top, right] - integral[bottom, left] + integral[top, left]

    def rectangle_key(top: int, left: int, bottom: int, right: int,
                      unavailable_integral: np.ndarray) -> tuple[int, int, int]:
        white_count = _integral_sum(white_integral, top, left, bottom, right)
        black_count = _integral_sum(black_integral, top, left, bottom, right)
        if _integral_sum(unavailable_integral, top, left, bottom, right):
            return (-1, -1, -1)
        return white_count - black_count, -black_count, white_count

    def improve_two_edges(bounds: tuple[int, int, int, int], edge_a: str, edge_b: str,
                          unavailable_integral: np.ndarray) -> tuple[int, int, int, int]:
        top, left, bottom, right = bounds
        edge_indices = {"top": 0, "left": 1, "bottom": 2, "right": 3}
        ranges = {
            "top": np.arange(bottom, dtype=np.int32),
            "left": np.arange(right, dtype=np.int32),
            "bottom": np.arange(top + 1, available.shape[0] + 1, dtype=np.int32),
            "right": np.arange(left + 1, available.shape[1] + 1, dtype=np.int32),
        }
        best_bounds = bounds
        best_key = rectangle_key(*bounds, unavailable_integral)
        for value_a in ranges[edge_a]:
            values_b = ranges[edge_b]
            candidate_top = value_a if edge_a == "top" else values_b if edge_b == "top" else top
            candidate_left = value_a if edge_a == "left" else values_b if edge_b == "left" else left
            candidate_bottom = value_a if edge_a == "bottom" else values_b if edge_b == "bottom" else bottom
            candidate_right = value_a if edge_a == "right" else values_b if edge_b == "right" else right
            width_px = candidate_right - candidate_left
            height_px = candidate_bottom - candidate_top
            area_px = width_px * height_px
            black_count = sum_rect(black_integral, candidate_top, candidate_left,
                                   candidate_bottom, candidate_right)
            white_count = sum_rect(white_integral, candidate_top, candidate_left,
                                   candidate_bottom, candidate_right)
            invalid_count = sum_rect(unavailable_integral, candidate_top, candidate_left,
                                     candidate_bottom, candidate_right)
            valid_dimensions = (
                (width_px / pixels_per_meter_x >= HOLE_MIN_SIDE_M)
                & (height_px / pixels_per_meter_y >= HOLE_MIN_LONG_SIDE_M)
            ) | (
                (width_px / pixels_per_meter_x >= HOLE_MIN_LONG_SIDE_M)
                & (height_px / pixels_per_meter_y >= HOLE_MIN_SIDE_M)
            )
            valid = (invalid_count == 0) & valid_dimensions & (
                black_count <= max_black_fraction * area_px
            )
            if not np.any(valid):
                continue
            net_white = white_count - black_count
            highest_net = np.max(net_white[valid])
            tied = valid & (net_white == highest_net)
            fewest_black = np.min(black_count[tied])
            tied &= black_count == fewest_black
            most_white = np.max(white_count[tied])
            index = int(np.flatnonzero(tied & (white_count == most_white))[0])
            candidate_values = list(bounds)
            candidate_values[edge_indices[edge_a]] = int(value_a)
            candidate_values[edge_indices[edge_b]] = int(values_b[index])
            candidate = tuple(candidate_values)
            candidate_key = int(highest_net), -int(fewest_black), int(most_white)
            if candidate_key > best_key:
                best_bounds, best_key = candidate, candidate_key
        return best_bounds

    while True:
        bounds = _largest_true_rectangle(available & ~black, pixels_per_meter_x, pixels_per_meter_y)
        if bounds is None:
            break
        unavailable_integral = np.pad((~available).astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        edge_pairs = (("top", "left"), ("top", "right"), ("bottom", "left"),
                      ("bottom", "right"), ("top", "bottom"), ("left", "right"))
        while True:
            previous = bounds
            for edge_a, edge_b in edge_pairs:
                bounds = improve_two_edges(bounds, edge_a, edge_b, unavailable_integral)
            if bounds == previous:
                break

        (top, left, bottom, right), snapped_edges = _expand_rectangle_to_nearby_wall(
            bounds, available, inside, pixels_per_meter_x, pixels_per_meter_y
        )
        width_px, height_px = right - left, bottom - top
        black_count = _integral_sum(black_integral, top, left, bottom, right)
        white_count = _integral_sum(white_integral, top, left, bottom, right)
        black_fraction = _integral_sum(black_integral, top, left, bottom, right) / (width_px * height_px)
        rectangles.append({
            "bbox_px": None,
            "corners_px": [],
            "orientation_deg": alignment_deg,
            "width_m": width_px / pixels_per_meter_x,
            "height_m": height_px / pixels_per_meter_y,
            "area_m2": width_px * height_px / (pixels_per_meter_x * pixels_per_meter_y),
            "black_fraction": black_fraction,
            "black_pixels": black_count,
            "white_pixels": white_count,
            "snapped_to_wall_edges": snapped_edges,
        })
        aligned_corners = np.array([
            [left, top], [right - 1, top], [right - 1, bottom - 1], [left, bottom - 1],
        ], dtype=np.float32)
        original_corners = cv2.transform(aligned_corners[None, ...], inverse)[0]
        rectangle = rectangles[-1]
        rectangle["corners_px"] = original_corners.tolist()
        rectangle["bbox_px"] = [
            int(np.floor(original_corners[:, 0].min())),
            int(np.floor(original_corners[:, 1].min())),
            int(np.ceil(original_corners[:, 0].max())),
            int(np.ceil(original_corners[:, 1].max())),
        ]
        available[top:bottom, left:right] = False

    metrics = {
        "minimum_side_m": HOLE_MIN_SIDE_M,
        "minimum_long_side_m": HOLE_MIN_LONG_SIDE_M,
        "maximum_black_fraction": max_black_fraction,
        "wall_contour_gap_close_m": WALL_CONTOUR_CLOSE_GAP_M,
        "rectangle_wall_gap_snap_m": 0.10,
        "fit_method": "pairwise rectangle-edge search in dominant wall-contour orientation, maximizing white pixels minus black pixels; black overlap is capped",
        "alignment_angle_deg": alignment_deg,
        "rectangle_count": len(rectangles),
        "rectangles": rectangles,
    }
    return rectangles, metrics


def _draw_hole_rectangles(image: Image.Image, rectangles: list[dict],
                          axis_bbox: tuple[int, int, int, int]) -> Image.Image:
    if not rectangles:
        return image
    x0, y0, x1, y1 = axis_bbox
    crop = image.crop(axis_bbox).convert("RGBA")
    overlay = Image.new("RGBA", crop.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for rectangle in rectangles:
        if rectangle.get("corners_px"):
            points = [(x - x0, y - y0) for x, y in rectangle["corners_px"]]
            draw.polygon(points, fill=HOLE_FILL_RGBA, outline=HOLE_OUTLINE_RGBA, width=2)
        else:
            left, top, right, bottom = rectangle["bbox_px"]
            local_box = (left - x0, top - y0, right - x0, bottom - y0)
            draw.rectangle(local_box, fill=HOLE_FILL_RGBA, outline=HOLE_OUTLINE_RGBA, width=2)
    image.paste(Image.alpha_composite(crop, overlay).convert("RGB"), (x0, y0))
    return image


def _render_area(output_path: Path, original_output_path: Path, level_id: int, area_id: int,
                 plane_equation: list[float],
                 vertices: np.ndarray, triangles: np.ndarray, attributes: dict[str, np.ndarray],
                 wall_shell: trimesh.Trimesh | None, apply_blur: bool = True,
                 morphology_candidate_path: Path | None = None,
                 morphology_holes_path: Path | None = None,
                 max_hole_black_fraction: float = HOLE_MAX_BLACK_FRACTION) -> dict:
    plane, axis_u, axis_v = _plane_frame(plane_equation, f"floor level {level_id} area {area_id}")
    fitted = attributes["is_fitted_plane"] == 1
    floor_preview_faces = fitted & (attributes["surface_role"] == FLOOR_ROLE) & (
        attributes["floor_area_id"] == area_id
    )
    if not floor_preview_faces.any():
        raise ValueError(f"Slice has no fitted floor preview for level {level_id}, area {area_id}")
    source_face_ids, excluded_near_vertical = _retain_nonvertical_source_faces(
        vertices, triangles, np.flatnonzero(~fitted)
    )
    if not len(source_face_ids):
        raise ValueError(f"No non-vertical source triangles remain for floor level {level_id}")

    projected_vertices = _collapse_to_plane(vertices, plane, axis_u, axis_v)
    plane_vertices = np.unique(triangles[floor_preview_faces])
    plane_coords = projected_vertices[plane_vertices]
    source_vertices = np.unique(triangles[source_face_ids])
    source_coords = projected_vertices[source_vertices]
    original_low, original_high = plane_coords.min(axis=0), plane_coords.max(axis=0)
    expanded_low = np.minimum(original_low, source_coords.min(axis=0))
    expanded_high = np.maximum(original_high, source_coords.max(axis=0))
    expanded = bool(np.any(expanded_low < original_low - 1e-8)
                    or np.any(expanded_high > original_high + 1e-8))
    contour_uv = _wall_contour_uv(wall_shell, plane, axis_u, axis_v)
    plot_low, plot_high = expanded_low.copy(), expanded_high.copy()
    if len(contour_uv):
        contour_bounds = contour_uv.reshape(-1, 2)
        plot_low = np.minimum(plot_low, contour_bounds.min(axis=0))
        plot_high = np.maximum(plot_high, contour_bounds.max(axis=0))
    wall_contour_expanded_view = bool(
        np.any(plot_low < expanded_low - 1e-8)
        or np.any(plot_high > expanded_high + 1e-8)
    )
    center_y = vertices[triangles[source_face_ids]].mean(axis=1)[:, 1]
    source_y = vertices[triangles[source_face_ids]][:, :, 1]
    order = np.argsort(center_y, kind="stable")
    source_face_ids = source_face_ids[order]
    plot_span = max(float(np.max(plot_high - plot_low)), 1.0)
    padding = max(plot_span * 0.02, 0.05)
    fig, ax = plt.subplots(figsize=(12, 10), dpi=160)
    ax.set_facecolor("#f2f0ed")

    degenerate_count = 0
    for start in range(0, len(source_face_ids), PLANE_FACES_PER_BATCH):
        face_ids = source_face_ids[start:start + PLANE_FACES_PER_BATCH]
        triangles_2d = projected_vertices[triangles[face_ids]]
        ab = triangles_2d[:, 1] - triangles_2d[:, 0]
        ac = triangles_2d[:, 2] - triangles_2d[:, 0]
        projected_area = np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]) * 0.5
        visible = projected_area > 1e-10
        face_colors = np.tile([0., 0., 0., 1.], (len(face_ids), 1))
        batch_zorder = 2 + start // PLANE_FACES_PER_BATCH
        if visible.any():
            ax.add_collection(PolyCollection(
                triangles_2d[visible], facecolors=face_colors[visible], edgecolors="none",
                linewidths=0, antialiased=False, rasterized=True, zorder=batch_zorder,
            ))

        hidden = ~visible
        if hidden.any():
            degenerate_count += int(hidden.sum())
            flat = triangles_2d[hidden]
            pairs = np.array([[0, 1], [1, 2], [2, 0]])
            endpoints = flat[:, pairs]
            lengths_squared = np.sum((endpoints[:, :, 1] - endpoints[:, :, 0]) ** 2, axis=2)
            longest = np.argmax(lengths_squared, axis=1)
            segments = endpoints[np.arange(len(endpoints)), longest]
            has_length = lengths_squared.max(axis=1) > 1e-12
            if has_length.any():
                ax.add_collection(LineCollection(
                    segments[has_length], colors=face_colors[hidden][has_length],
                    linewidths=0.55, rasterized=True, zorder=batch_zorder,
                ))

    whole_meter = lambda value, _position: f"{round(value):d}" if abs(value - round(value)) < 1e-8 else ""
    ax.set_xlim(plot_low[0] - padding, plot_high[0] + padding)
    ax.set_ylim(plot_low[1] - padding, plot_high[1] + padding)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Floor plane U (m)")
    ax.set_ylabel("Floor plane V (m)")
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(whole_meter))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(whole_meter))
    tilt = float(np.rad2deg(np.arccos(np.clip(plane[1], -1.0, 1.0))))
    ax.set_title(f"Floor level {level_id:03d}, area {area_id:03d} — vertically collapsed onto fitted plane ({tilt:.1f}°)")
    fig.tight_layout()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    axis_bounds = ax.get_window_extent()
    axis_bbox = (
        max(0, int(np.floor(axis_bounds.x0))),
        max(0, int(np.floor(height - axis_bounds.y1))),
        min(width, int(np.ceil(axis_bounds.x1))),
        min(height, int(np.ceil(height - axis_bounds.y0))),
    )
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    image.save(original_output_path)
    morphology_metrics = None
    holes_metrics = None
    if morphology_candidate_path is not None:
        data_origin = ax.transData.transform((0.0, 0.0))
        pixels_per_meter = min(
            np.linalg.norm(ax.transData.transform((1.0, 0.0)) - data_origin),
            np.linalg.norm(ax.transData.transform((0.0, 1.0)) - data_origin),
        )
        candidate, morphology_metrics = _regularize_black_mask(
            image.copy(), axis_bbox, float(pixels_per_meter)
        )
        candidate = _draw_wall_contour(candidate, contour_uv, ax, axis_bbox)
        candidate.save(morphology_candidate_path)
        if morphology_holes_path is not None:
            data_origin = ax.transData.transform((0.0, 0.0))
            pixels_per_meter_x = float(np.linalg.norm(ax.transData.transform((1.0, 0.0)) - data_origin))
            pixels_per_meter_y = float(np.linalg.norm(ax.transData.transform((0.0, 1.0)) - data_origin))
            rectangles, holes_metrics = _fit_wall_gap_rectangles(
                candidate, axis_bbox, pixels_per_meter_x, pixels_per_meter_y,
                max_black_fraction=max_hole_black_fraction,
            )
            _draw_hole_rectangles(candidate, rectangles, axis_bbox).save(morphology_holes_path)
    if apply_blur:
        image = _blur_sharpen_plot(image, axis_bbox)
    image, removed_black_pixels = _clean_black_artifacts(image, axis_bbox)
    image = _draw_wall_contour(image, contour_uv, ax, axis_bbox)
    image.save(output_path)
    plt.close(fig)

    result = {
        "area_id": area_id,
        "image": output_path.name,
        "original_projection_image": original_output_path.name,
        "plane_equation": plane.tolist(),
        "tilt_deg": tilt,
        "source_triangle_count": int(len(source_face_ids)),
        "near_vertical_triangles_excluded": excluded_near_vertical,
        "source_y_bounds_m": [float(source_y.min()), float(source_y.max())],
        "projected_degenerate_triangle_count": degenerate_count,
        "black_artifact_pixels_removed": removed_black_pixels,
        "wall_contour_segment_count": int(len(contour_uv)),
        "wall_contour_height_above_floor_m": WALL_CONTOUR_HEIGHT_M,
        "wall_contour_color_rgb": list(WALL_CONTOUR_COLOR),
        "contour_refinement": {
            "method": "iterative Gaussian unsharp mask on black projection mask" if apply_blur else "none",
            "applied": apply_blur,
            "sigma_px": BLUR_SIGMA_PX if apply_blur else 0.0,
            "turns": SHARPEN_TURNS if apply_blur else 0,
            "unsharp_amount": 1.0 if apply_blur else 0.0,
            "applied_to": "black projection mask; original background preserved" if apply_blur else None,
        },
        "initial_plane_bounds_uv_m": [original_low.tolist(), original_high.tolist()],
        "expanded_plane_bounds_uv_m": [plot_low.tolist(), plot_high.tolist()],
        "expanded_to_fit_slice": expanded,
        "expanded_to_fit_wall_contour": wall_contour_expanded_view,
        "vertical_projection_axis": "scene Y",
    }
    if morphology_candidate_path is not None:
        result["first_refinement"] = {
            "image": morphology_candidate_path.name,
            "method": "disk opening, relative component-area filtering, and signed-distance Gaussian smoothing",
            **morphology_metrics,
        }
    if morphology_holes_path is not None:
        result["holes_overlay_image"] = morphology_holes_path.name
        result["hole_detection"] = holes_metrics
    return result


def project_floor_slices_onto_planes(instances_dir: Path, level_id: int | None = None,
                                     apply_blur: bool = True,
                                     morphology_candidate: bool = False,
                                     detect_holes: bool = False,
                                     max_hole_black_fraction: float = HOLE_MAX_BLACK_FRACTION) -> Path:
    if not np.isfinite(max_hole_black_fraction) or not 0.0 <= max_hole_black_fraction <= 1.0:
        raise ValueError("Maximum hole black-pixel fraction must be between 0 and 1")
    level_dir = instances_dir / "intermediate_floor_levels"
    floor_manifest_path = level_dir / "floor_level_intermediate_manifest.json"
    slice_manifest_path = level_dir / "floor_level_scene_slices_manifest.json"
    floor_manifest = json.loads(floor_manifest_path.read_text(encoding="utf-8"))
    slice_manifest = json.loads(slice_manifest_path.read_text(encoding="utf-8"))
    floor_levels = floor_manifest.get("levels") if isinstance(floor_manifest, dict) else None
    slice_levels = slice_manifest.get("levels") if isinstance(slice_manifest, dict) else None
    if not isinstance(floor_levels, list) or not isinstance(slice_levels, list):
        raise ValueError("Floor and slice manifests must each contain a levels list")
    floor_by_id = {level.get("level"): level for level in floor_levels if isinstance(level, dict)}
    wall_shell = _load_wall_shell(instances_dir)
    selected_level_id = level_id
    selected_slices = slice_levels
    if selected_level_id is not None:
        selected_slices = [record for record in slice_levels
                           if isinstance(record, dict) and record.get("level") == selected_level_id]
        if len(selected_slices) != 1 or selected_level_id not in floor_by_id:
            raise ValueError(f"Floor level {selected_level_id} is missing or duplicated in the slice manifests")

    outputs, level_records = [], []
    morphology_output = morphology_candidate or detect_holes
    output_manifest_path = level_dir / (
        "floor_level_collapsed_first_refinement_holes_manifest.json" if detect_holes else
        "floor_level_first_refinement_manifest.json" if morphology_candidate else
        "floor_level_collapsed_projections_manifest.json"
    )
    with tempfile.TemporaryDirectory(prefix=".floor_plane_projections_", dir=level_dir) as temp_name:
        temp_dir = Path(temp_name)
        for slice_record in selected_slices:
            if not isinstance(slice_record, dict):
                raise ValueError("Invalid floor scene slice manifest entry")
            level_id = slice_record.get("level")
            floor = floor_by_id.get(level_id)
            filename = slice_record.get("output_file")
            if (floor is None or not isinstance(filename, str)
                    or not re.fullmatch(r"floor_level_\d{3}_scene_slice\.ply", filename)):
                raise ValueError(f"Invalid floor/slice pairing for level {level_id!r}")
            areas = floor.get("floor_areas")
            if not isinstance(areas, list) or not areas:
                raise ValueError(f"Floor level {level_id} has no fitted floor areas")
            vertices, triangles, attributes = _load_slice(level_dir / filename)
            area_records = []
            for area_index, area in enumerate(areas):
                if not isinstance(area, dict):
                    raise ValueError(f"Invalid fitted plane for floor level {level_id}")
                area_id = area.get("area_id", area_index)
                if not isinstance(area_id, int) or area_id < 0:
                    raise ValueError(f"Invalid floor area ID: {area_id!r}")
                image_name = f"floor_level_{level_id:03d}_area_{area_id:03d}_collapsed.png"
                original_image_name = f"floor_level_{level_id:03d}_area_{area_id:03d}_collapsed_original.png"
                candidate_image_name = f"floor_level_{level_id:03d}_area_{area_id:03d}_collapsed_first_refinement.png"
                holes_image_name = f"floor_level_{level_id:03d}_area_{area_id:03d}_collapsed_first_refinement_with_holes.png"
                info = _render_area(
                    temp_dir / image_name, temp_dir / original_image_name, level_id, area_id,
                    area.get("plane_equation", []), vertices, triangles, attributes, wall_shell,
                    apply_blur=apply_blur and not morphology_output,
                    morphology_candidate_path=(temp_dir / candidate_image_name) if morphology_output else None,
                    morphology_holes_path=(temp_dir / holes_image_name) if detect_holes else None,
                    max_hole_black_fraction=max_hole_black_fraction,
                )
                if detect_holes:
                    info["first_refinement_image"] = candidate_image_name
                    info["image"] = holes_image_name
                    outputs.extend((original_image_name, candidate_image_name, holes_image_name))
                elif morphology_candidate:
                    info["image"] = candidate_image_name
                    outputs.append(candidate_image_name)
                else:
                    outputs.extend((image_name, original_image_name))
                area_records.append(info)
            level_records.append({"level": level_id, "source_slice": filename, "areas": area_records})

        if morphology_output:
            manifest = {
                "method": "locked morphology first refinement" + (" followed by wall-gap rectangle fitting" if detect_holes else ""),
                "parameters": {
                    "minimum_feature_width_m": MORPHOLOGY_MIN_FEATURE_WIDTH_M,
                    "minimum_component_area_m2": MORPHOLOGY_MIN_COMPONENT_AREA_M2,
                    "minimum_component_area_fraction": MORPHOLOGY_COMPONENT_AREA_FRACTION,
                    "minimum_component_area_px": MIN_BLACK_COMPONENT_AREA_PX,
                    "signed_distance_sigma_m": SIGNED_DISTANCE_SMOOTH_SIGMA_M,
                    "background_preserved": True,
                    "wall_contour_drawn_after_cleanup": True,
                },
                **({"hole_detection": {
                    "minimum_side_m": HOLE_MIN_SIDE_M,
                    "minimum_long_side_m": HOLE_MIN_LONG_SIDE_M,
                    "maximum_black_fraction": max_hole_black_fraction,
                    "wall_contour_gap_close_m": WALL_CONTOUR_CLOSE_GAP_M,
                    "rectangles_non_overlapping": True,
                }} if detect_holes else {}),
                "levels": level_records,
            }
            if selected_level_id is not None:
                previous_manifest = json.loads(output_manifest_path.read_text(encoding="utf-8")) \
                    if output_manifest_path.exists() else {"levels": []}
                previous_levels = previous_manifest.get("levels")
                if not isinstance(previous_levels, list):
                    raise ValueError("Existing morphology manifest has no levels list")
                merged = {record["level"]: record for record in previous_levels
                          if isinstance(record, dict) and record.get("level") != selected_level_id}
                merged.update({record["level"]: record for record in level_records})
                manifest["levels"] = sorted(merged.values(), key=lambda record: record["level"])
                manifest["partial_update"] = {
                    "updated_level_ids": [selected_level_id],
                    "preserved_level_ids": sorted(record["level"] for record in previous_levels
                                                    if isinstance(record, dict) and record.get("level") != selected_level_id),
                }
            temp_manifest = temp_dir / output_manifest_path.name
            temp_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            for filename in outputs:
                os.replace(temp_dir / filename, level_dir / filename)
            os.replace(temp_manifest, output_manifest_path)

            keep = set(outputs)
            cleanup_pattern = (
                f"floor_level_{selected_level_id:03d}_area_*_collapsed_first_refinement*.png"
                if detect_holes and selected_level_id is not None else
                "floor_level_*_area_*_collapsed_first_refinement*.png" if detect_holes else
                f"floor_level_{selected_level_id:03d}_area_*_collapsed_first_refinement.png"
                if selected_level_id is not None else
                "floor_level_*_area_*_collapsed_first_refinement.png"
            )
            for path in level_dir.glob(cleanup_pattern):
                if path.name not in keep:
                    path.unlink()
            for level in level_records:
                for area in level["areas"]:
                    image_name = area.get("image", area.get("first_refinement", {}).get("image"))
                    label_text = "holes overlay" if detect_holes else "first refinement"
                    print(f"Level {level['level']:03d}, area {area['area_id']:03d}: {label_text} -> {image_name}")
            return output_manifest_path

        manifest = {
            "projection": "vertical Y projection onto each fitted floor plane, then orthographic rendering in that plane",
            "source_triangle_filter": {
                "excluded": "source triangle planes vertical within the configured tolerance",
                "vertical_tolerance_deg": VERTICAL_TRIANGLE_TOLERANCE_DEG,
                "equivalent_normal_angle_from_vertical_deg": [
                    90.0 - VERTICAL_TRIANGLE_TOLERANCE_DEG,
                    90.0 + VERTICAL_TRIANGLE_TOLERANCE_DEG,
                ],
            },
            "fitted_plane_expansion": "expand the displayed plane rectangle to include the full collapsed slice bounds",
            "fitted_preview_faces_in_slice": "excluded from source projection; floor preview bounds define the initial plane rectangle",
            "triangle_color_rgb": [0, 0, 0],
            "contour_refinement": {
                "method": "iterative Gaussian unsharp mask on black projection mask",
                "sigma_px": BLUR_SIGMA_PX, "turns": SHARPEN_TURNS, "unsharp_amount": 1.0,
            },
            "black_mask_cleanup": {
                "method": "disk opening, narrow-component rejection, and connected-component area filtering",
                "max_rgb_channel": BLACK_MASK_MAX_CHANNEL,
                "strip_opening_radius_px": STRIP_OPENING_RADIUS_PX,
                "minimum_component_area_px": MIN_BLACK_COMPONENT_AREA_PX,
                "maximum_thin_component_width_px": MAX_THIN_COMPONENT_WIDTH_PX,
                "minimum_thin_component_aspect_ratio": MIN_THIN_COMPONENT_ASPECT_RATIO,
                "plot_edge_margin_px": CLEANUP_EDGE_MARGIN_PX,
                "neutral_background_max_channel_spread": NEUTRAL_BACKGROUND_MAX_SPREAD,
            },
            "wall_contour": {"source": "fitted wall enclosing shell", "height_above_floor_m": WALL_CONTOUR_HEIGHT_M,
                             "color_rgb": list(WALL_CONTOUR_COLOR), "drawn_after_raster_cleanup": True},
            "levels": level_records,
        }
        if selected_level_id is not None:
            if output_manifest_path.exists():
                previous_manifest = json.loads(output_manifest_path.read_text(encoding="utf-8"))
                previous_levels = previous_manifest.get("levels") if isinstance(previous_manifest, dict) else None
                if not isinstance(previous_levels, list):
                    raise ValueError("Existing projection manifest has no levels list")
            else:
                previous_levels = []
            merged = {record["level"]: record for record in previous_levels
                      if isinstance(record, dict) and record.get("level") != selected_level_id}
            merged.update({record["level"]: record for record in level_records})
            manifest["levels"] = sorted(merged.values(), key=lambda record: record["level"])
            manifest["partial_update"] = {
                "updated_level_ids": [selected_level_id],
                "preserved_level_ids": sorted(record["level"] for record in previous_levels
                                                if isinstance(record, dict) and record.get("level") != selected_level_id),
            }
        temp_manifest = temp_dir / output_manifest_path.name
        temp_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for filename in outputs:
            os.replace(temp_dir / filename, level_dir / filename)
        os.replace(temp_manifest, output_manifest_path)

    keep = set(outputs)
    cleanup_pattern = (f"floor_level_{selected_level_id:03d}_area_*_collapsed*.png"
                       if selected_level_id is not None else "floor_level_*_area_*_collapsed*.png")
    for path in level_dir.glob(cleanup_pattern):
        if path.name not in keep:
            path.unlink()
    for level in level_records:
        for area in level["areas"]:
            print(f"Level {level['level']:03d}, area {area['area_id']:03d}: "
                  f"collapsed {area['source_triangle_count']:,} triangles -> {area['image']}")
    return output_manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path, required=True,
                        help="Stage 06 output directory containing fitted floors, walls, and scene slices")
    parser.add_argument("--level", type=int, help="Render only this floor level and preserve other level outputs")
    parser.add_argument("--no-blur", action="store_true", help="Skip Gaussian blur/sharpen refinement")
    parser.add_argument("--first-refinement", "--morphology-candidate", dest="morphology_candidate",
                        action="store_true",
                        help="Write the locked morphology first refinement without replacing baseline projections")
    parser.add_argument("--detect-holes", action="store_true",
                        help="Apply the morphology first refinement and save a separate blue-rectangle hole overlay")
    parser.add_argument("--max-hole-black-fraction", type=float, default=HOLE_MAX_BLACK_FRACTION,
                        help="Maximum black-pixel fraction allowed inside hole rectangles (0.05 = 5%%; default: 0.05)")
    args = parser.parse_args(argv)
    try:
        project_floor_slices_onto_planes(
            args.instances_dir, level_id=args.level, apply_blur=not args.no_blur,
            morphology_candidate=args.morphology_candidate, detect_holes=args.detect_holes,
            max_hole_black_fraction=args.max_hole_black_fraction,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: Could not project floor slices onto fitted planes: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
