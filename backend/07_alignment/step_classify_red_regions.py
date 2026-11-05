"""Label each projected retained red region as a window or mirror."""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.font_manager
import numpy as np
import plyfile
import trimesh
from PIL import Image, ImageDraw, ImageFont

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import step_red_region_contours as contours
from scan_loader import ScanDataset

LOGGER = logging.getLogger("phase07_red_region_classification")
WINDOW_TEXT_CLASS_ID = 2  # Stage 05's fixed category order assigns ID 2 to window.
WINDOW_HEIGHT_ABOVE_FLOOR_M = 1.5


def _window_classes(scans: dict[str, Any]) -> tuple[dict[str, set[int]], set[str]]:
    classes: dict[str, set[int]] = {}
    available = set()
    for scan_id, scan in scans.items():
        path = scan.info.scan_dir / "segmentation" / "non_structural_similarity" / "valid_class_associations.json"
        if not path.is_file():
            classes[scan_id] = set()
            continue
        data = json.loads(path.read_text())
        available.add(scan_id)
        classes[scan_id] = {int(image_id) for image_id, text_id in data.get("associations", [])
                            if int(text_id) == WINDOW_TEXT_CLASS_ID}
    return classes, available


def _floor_planes(alignment_dir: Path, scan_id: str) -> list[dict[str, Any]]:
    element_dir = alignment_dir / "scans" / scan_id / "mesh_elaboration" / "elements"
    paths = sorted(element_dir.glob("floor_level_*.ply"))
    if not paths:
        raise FileNotFoundError(f"No aligned regularized floor planes for {scan_id}: {element_dir}")

    planes = []
    for path in paths:
        mesh = trimesh.load(path, process=False)
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError(f"Expected a non-empty aligned floor mesh: {path}")
        triangles = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        areas = np.linalg.norm(cross, axis=1) * 0.5
        valid = areas > 1e-12
        normals = cross[valid] / (2.0 * areas[valid, None])
        normals[normals[:, 1] < 0] *= -1
        normal = np.average(normals, axis=0, weights=areas[valid])
        normal /= np.linalg.norm(normal)
        if normal[1] < 0.9:
            raise ValueError(f"Regularized floor is not horizontal enough to measure height: {path}")
        centers = triangles[valid].mean(axis=1)
        offset = float(np.average(centers @ normal, weights=areas[valid]))
        planes.append({"path": path, "normal": normal, "offset": offset})
    return planes


def _bbox_low_corner(bbox: tuple[float, float, float, float], plane: dict[str, Any]) -> np.ndarray:
    x0, x1, y0, y1 = bbox
    origin, u, v = contours._projector(plane["normal"], plane["offset"])
    uv = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    corners = origin + uv[:, 0, None] * u + uv[:, 1, None] * v
    return corners[np.argmin(corners[:, 1])]


def _closest_floor_height(floors: list[dict[str, Any]], point: np.ndarray) -> tuple[float, Path]:
    candidates = []
    for floor in floors:
        normal, offset = floor["normal"], floor["offset"]
        height = float((offset - normal[0] * point[0] - normal[2] * point[2]) / normal[1])
        plane_distance = abs(float(point @ normal - offset))
        candidates.append((plane_distance, height, floor["path"]))
    _, height, path = min(candidates, key=lambda item: item[0])
    return height, path


def _instance_classes(group: list[dict[str, Any]]) -> set[int]:
    classes = set()
    for region in group:
        match = re.search(r"class_(\d+)_object(?:/|$)", region["source"])
        if match:
            classes.add(int(match.group(1)))
    return classes


def _annotate_image(source: Path, target: Path, label: str, criteria: list[str]) -> None:
    image = Image.open(source).convert("RGB")
    header_height = 120
    output = Image.new("RGB", (image.width, image.height + header_height), "white")
    output.paste(image, (0, header_height))
    draw = ImageDraw.Draw(output)
    font_path = matplotlib.font_manager.findfont("DejaVu Sans")
    title_font = ImageFont.truetype(font_path, 38)
    criteria_font = ImageFont.truetype(font_path, 21)
    color = "#166534" if label == "WINDOW" else "#1d4ed8"
    draw.text((image.width // 2, 25), label, fill=color, font=title_font, anchor="mm")
    for index, line in enumerate(criteria):
        draw.text((image.width // 2, 58 + index * 27), line, fill="#202124", font=criteria_font, anchor="mm")
    target.parent.mkdir(parents=True, exist_ok=True)
    output.save(target)


def run(workspace: Path, alignment_dir: Path | None = None,
        output_dir: Path | None = None) -> list[dict[str, Any]]:
    workspace = Path(workspace).resolve()
    alignment_dir = Path(alignment_dir or workspace / "multi_room_alignment").resolve()
    image_dir = alignment_dir / "red_region_contours"
    output_dir = Path(output_dir or alignment_dir / "red_region_classification").resolve()
    red_path = alignment_dir / "aligned_objects_final_red.ply"
    transform_path = alignment_dir / "stats" / "aligned_scans_transforms.json"
    for path in (red_path, transform_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    scans = ScanDataset(workspace).scans
    transforms = json.loads(transform_path.read_text())["scans"]
    red_ply = plyfile.PlyData.read(str(red_path), mmap="r", known_list_len={"face": {"vertex_indices": 3}})
    groups = contours._red_regions(alignment_dir, red_ply, scans)
    if not groups:
        raise ValueError("Final red object PLY contains no connected regions")

    region_scans = {group[0]["scan_id"] for group in groups}
    surfaces = {scan_id: contours._build_exterior_surface(
                    scans[scan_id], np.asarray(transforms[scan_id]["transform_matrix"], dtype=np.float64))
                for scan_id in region_scans}
    window_classes, available_associations = _window_classes(scans)
    floor_cache: dict[str, list[dict[str, Any]]] = {}
    source_images = [image_dir / f"red_region_{index:03d}_{group[0]['scan_id']}.png"
                     for index, group in enumerate(groups, 1)]
    missing_images = [path for path in source_images if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing contour projection image(s): {missing_images}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for index, group in enumerate(groups, 1):
        scan_id = group[0]["scan_id"]
        plane = contours._dominant_plane(group, surfaces[scan_id])
        triangles = np.concatenate([region["triangles"] for region in group])
        footprint = contours._triangle_union(
            contours._project_triangles(triangles, plane["normal"], plane["offset"]))
        bbox, _, _, _ = contours._minimum_axis_aligned_box(footprint)
        bottom = _bbox_low_corner(bbox, plane)
        if scan_id not in floor_cache:
            floor_cache[scan_id] = _floor_planes(alignment_dir, scan_id)
        floor_height, floor_path = _closest_floor_height(floor_cache[scan_id], bottom)
        height_above_floor = float(bottom[1] - floor_height)
        height_window = height_above_floor > WINDOW_HEIGHT_ABOVE_FLOOR_M

        instance_classes = _instance_classes(group)
        matched_window_classes = sorted(instance_classes & window_classes.get(scan_id, set()))
        gemma_window = bool(matched_window_classes)
        is_window = height_window or gemma_window
        label = "WINDOW" if is_window else "MIRROR"
        assert is_window == (height_window or gemma_window)

        height_text = (f"Height: {height_above_floor:.2f} m above own floor (> {WINDOW_HEIGHT_ABOVE_FLOOR_M:.2f} m)"
                       if height_window else
                       f"Height: {height_above_floor:.2f} m above own floor (≤ {WINDOW_HEIGHT_ABOVE_FLOOR_M:.2f} m)")
        if gemma_window:
            gemma_text = "Gemma2: window association for " + ", ".join(f"class_{class_id:03d}" for class_id in matched_window_classes)
        elif scan_id in available_associations:
            gemma_text = "Gemma2: no window-associated instance"
        else:
            gemma_text = "Gemma2: class associations unavailable"
        region_id = f"red_region_{index:03d}_{scan_id}"
        source_image = image_dir / f"{region_id}.png"
        output_image = output_dir / source_image.name
        criteria = [height_text, gemma_text]
        _annotate_image(source_image, output_image, label, criteria)
        summaries.append({"region": region_id, "classification": label,
                          "height_above_own_floor_m": height_above_floor,
                          "height_window": height_window, "gemma_window": gemma_window,
                          "window_classes": matched_window_classes,
                          "closest_floor": str(floor_path), "image": str(output_image)})
        LOGGER.info("%s: %s (%s; %s) -> %s", region_id, label,
                    height_text, gemma_text, output_image)
    expected_images = {f"red_region_{index:03d}_{group[0]['scan_id']}.png"
                       for index, group in enumerate(groups, 1)}
    for old_image in output_dir.glob("red_region_*.png"):
        if old_image.name not in expected_images:
            old_image.unlink()
    return summaries


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
