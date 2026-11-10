#!/usr/bin/env python3
"""Build a self-contained local tour and static comparison images from Phase 08 outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont

from generate_floor_level_meshes import _read_mesh
from generate_floorplan_measurements import (
    _render_svg_png,
    _flip_measurements_vertically,
    _write_measurement_overlay_png,
)

STAGE = Path(__file__).resolve().parent
DEFAULT_SCENE = STAGE.parent / "current_scene"
STYLES = ("realistic", "objects", "regularized")
LOG = logging.getLogger("interactive_viewer")


def load_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required viewer input missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def projection_grid(row):
    w, h = row["resolution_px"]
    if w != h or w <= 0:
        raise ValueError("Expected a square floor projection")
    lo, hi, size = row["mesh_bounds_min_xyz"], row["mesh_bounds_max_xyz"], row["pixel_size_m"]
    if not math.isfinite(size) or size <= 0:
        raise ValueError("Invalid projection pixel size")
    return {"x_min": (lo[0] + hi[0] - w * size) / 2,
            "z_max": (lo[2] + hi[2] + h * size) / 2, "size": size, "width": w, "height": h}


def project(position, grid):
    return [(position[0] - grid["x_min"]) / grid["size"],
            grid["height"] - (grid["z_max"] - position[2]) / grid["size"]]


def _flip_floorplan_vertically(image: Image.Image) -> Image.Image:
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def photo_direction(camera):
    matrix = np.asarray(camera["camera_to_world"], dtype=float)
    intrinsic = np.asarray(camera["intrinsics"], dtype=float)
    if matrix.shape != (4, 4) or intrinsic.shape != (3, 3) or not np.isfinite(matrix).all() or intrinsic[0, 0] <= 0:
        raise ValueError("Invalid camera pose or intrinsics")
    forward = matrix[:3, 2]
    if np.linalg.norm(forward[[0, 2]]) < 1e-6:
        raise ValueError("Camera points vertically; horizontal direction is undefined")
    # Principal point may be off-center: project both calibrated horizontal edge rays.
    rays = [matrix[:3, :3] @ np.array([(x - intrinsic[0, 2]) / intrinsic[0, 0], 0, 1]) for x in (0, camera["width"])]
    angles = [math.atan2(ray[2], ray[0]) for ray in rays]
    yaw = math.atan2(forward[2], forward[0])
    delta = [(a - yaw + math.pi) % (2 * math.pi) - math.pi for a in angles]
    return yaw, delta


def floor_geometry(records, alignment):
    triangles = []
    for record in records:
        v, f, _ = _read_mesh(alignment / record["source_mesh"])
        plane = np.asarray(record["plane_equation"], dtype=float)
        norm = np.linalg.norm(plane[:3])
        if not norm:
            raise ValueError(f"Invalid floor plane: {record['plane_id']}")
        n, d = plane[:3] / norm, plane[3] / norm
        tri = v[f]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        lengths = np.linalg.norm(cross, axis=1)
        good = (np.abs(cross @ n) / np.maximum(lengths, 1e-20) >= math.cos(math.radians(5))) & (np.max(np.abs(tri @ n + d), axis=1) <= max(float(np.ptp(v, axis=0).max()) * 1e-5, 1e-6))
        if not good.any():
            raise ValueError(f"No attributed floor triangles: {record['plane_id']}")
        triangles.append(v[f[good]])
    points = np.concatenate(triangles).reshape(-1, 3).astype(np.float32)
    faces = np.arange(len(points), dtype=np.uint32).reshape(-1, 3)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(points), o3d.core.Tensor(faces))
    return scene, points


def assign_floor(position, candidates):
    hits = []
    ray = o3d.core.Tensor([list(position) + [0, -1, 0]], dtype=o3d.core.Dtype.Float32)
    for name, scene, _ in candidates:
        distance = float(scene.cast_rays(ray)["t_hit"].numpy()[0])
        if math.isfinite(distance) and distance > 0:
            hits.append((distance, name))
    if not hits:
        raise ValueError(f"No floor below camera at {position}")
    return min(hits)[1]


def crop_for(points, grid):
    coords = np.array([project(p, grid) for p in points])
    pad = 0.35 / grid["size"]
    left, top = np.maximum(np.floor(coords.min(axis=0) - pad), 0).astype(int)
    right, bottom = np.minimum(np.ceil(coords.max(axis=0) + pad), [grid["width"], grid["height"]]).astype(int)
    if right <= left or bottom <= top:
        raise ValueError("Scan footprint lies outside its floor projection")
    return [int(left), int(top), int(right), int(bottom)]


def versioned_asset_url(relative_path, output):
    digest = hashlib.sha256((Path(output) / relative_path).read_bytes()).hexdigest()[:16]
    return f"{relative_path}?v={digest}"


def marked_plan(image, cameras, style, current=None, crop=None):
    image = image.copy().convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    for camera in cameras:
        x, y = camera["pixels"][style]
        if camera["kind"] == "photo":
            angles = [-(camera["yaw"] + d) for d in camera["edges"]]
            points = [(x, y)] + [(x + 90 * math.cos(a), y - 90 * math.sin(a)) for a in angles]
            draw.polygon(points, fill=(30, 115, 225, 85), outline=(20, 80, 190, 210))
        else:
            draw.ellipse((x-23, y-23, x+23, y+23), outline=(220, 110, 15, 230), width=3)
        color = (205, 85, 0, 255) if camera["kind"] == "panorama" else (10, 80, 180, 255)
        radius = 15 if camera is current else 13
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color, outline="white", width=2)
        draw.text((x, y), str(camera["number"]), font=font, anchor="mm", fill="white", stroke_width=0)
    if crop:
        image = image.crop(tuple(crop))
    return image


def generate(scene_dir):
    started = time.perf_counter()
    scene_dir = Path(scene_dir).resolve()
    artifacts = scene_dir / "artifacts"
    plans, photos, panoramas = [artifacts / name for name in ("Generate Floor Plans", "Extract Best Views", "Generate 360 Pictures")]
    manifests = [plans / "generate_floor_plans_manifest.json", photos / "extract_best_views_manifest.json", panoramas / "generate_360_pictures_manifest.json"]
    plan_manifest, photo_manifest, pano_manifest = map(load_json, manifests)
    for path, manifest in zip(manifests, (plan_manifest, photo_manifest, pano_manifest)):
        if manifest.get("macro_step_status") != "complete":
            raise ValueError(f"Upstream macro step is not complete: {path}")
    alignment = Path(plan_manifest["alignment_dir"])
    if alignment.resolve() != (scene_dir / "multi_room_alignment").resolve() or Path(photo_manifest["alignment_dir"]).resolve() != alignment.resolve() or Path(pano_manifest["alignment_dir"]).resolve() != alignment.resolve():
        raise ValueError("Upstream artifacts refer to different scene alignments")
    transform_path = alignment / "stats/aligned_scans_transforms.json"
    if hashlib.sha256(transform_path.read_bytes()).hexdigest() != photo_manifest["source_transforms_sha256"]:
        raise ValueError("Photo selection refers to stale scan transforms; rerun Extract Best Views")
    for path in (transform_path, plans / "floor_level_metadata.json", plans / "floor_level_meshes/floor_level_meshes_manifest.json"):
        expected = pano_manifest.get("source_hashes", {}).get(str(path))
        if expected is None or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Panoramas refer to missing or stale input: {path}; rerun Generate 360 Pictures")
    panel_path = alignment / "stats/phase07_region_replacements.json"
    panel_hash = pano_manifest.get("source_hashes", {}).get(str(panel_path))
    if ((panel_hash is None and pano_manifest.get("schema_version", 0) >= 5 and panel_path.is_file())
            or (panel_hash is not None and (not panel_path.is_file() or hashlib.sha256(panel_path.read_bytes()).hexdigest() != panel_hash))):
        raise ValueError(f"Panoramas refer to missing or stale panel data: {panel_path}; rerun Generate 360 Pictures")
    metadata = load_json(plans / "floor_level_metadata.json")
    projections = load_json(plans / "floor_level_projections/floor_level_projections_manifest.json")["projections"]
    overlays = load_json(plans / "floorplan_instance_overlays/floorplan_instance_overlays_manifest.json")
    measurements_path = Path(plan_manifest["floorplan_measurements_manifest"])
    measurements = load_json(measurements_path)
    manifests.append(measurements_path)
    if overlays.get("furnished_views_measured"):
        raise ValueError("Unmeasured furnished floorplans are missing; rerun Generate Floor Plans")
    levels = {}
    for row in projections:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", row["floor_level_name"]):
            raise ValueError(f"Invalid floor identity: {row['floor_level_name']}")
        level = levels.setdefault(row["floor_level_name"], {"name": row["floor_level_name"], "styles": {}})
        style = {"sealed": "realistic", "regularized": "regularized"}.get(row["mesh_type"])
        if style:
            if style in level["styles"]:
                raise ValueError(f"Duplicate {style} projection for {level['name']}")
            level["styles"][style] = {"source": row["projection_image"], "grid": projection_grid(row)}
    for row in overlays["floor_levels"]:
        level = levels[row["floor_level_name"]]
        level["styles"]["objects"] = {"source": row["output_svg"], "grid": level["styles"]["regularized"]["grid"]}
    if not levels or any(set(level["styles"]) != set(STYLES) for level in levels.values()):
        raise ValueError("Every floor must have all three plan styles")
    measurement_sources = {}
    for row in measurements["floor_levels"]:
        name = row["floor_level_name"]
        if name not in levels or name in measurement_sources:
            raise ValueError(f"Unexpected or duplicate measurement overlay for floor: {name}")
        grid = row["projection_grid"]
        regularized = levels[name]["styles"]["regularized"]
        if (Path(row["source_regularized_projection_png"]).resolve() != Path(regularized["source"]).resolve()
                or any(not math.isclose(float(grid[key]), float(regularized["grid"][key]), rel_tol=1e-9, abs_tol=1e-9)
                       for key in ("x_min", "z_max", "size", "width", "height"))):
            raise ValueError(f"Measurement overlay grid does not match floor projection: {name}")
        source = Path(row["overlay_png"])
        with Image.open(source) as image:
            if image.mode != "RGBA" or image.size != (grid["width"], grid["height"]):
                raise ValueError(f"Invalid transparent measurement overlay: {source}")
        measurement_sources[name] = row
        levels[name]["measurements"] = {"grid": grid}
    if set(measurement_sources) != set(levels):
        raise ValueError("Every floor must have a measurement overlay")
    floor_groups = defaultdict(list)
    for floor in metadata["floor_planes"]:
        if not all(re.fullmatch(r"[A-Za-z0-9_-]+", floor[key]) for key in ("scan_id", "floor_level_name")):
            raise ValueError(f"Invalid scan or floor identity: {floor['plane_id']}")
        floor_groups[(floor["scan_id"], floor["floor_level_name"])].append(floor)
    geometries = {key: floor_geometry(records, alignment) for key, records in floor_groups.items()}
    cameras = []
    for scan in photo_manifest["scans"]:
        scan_id = scan["scan_id"]
        if scan["status"] != "complete":
            raise ValueError(f"Incomplete photo selection: {scan_id}")
        selection_path = Path(scan["selected_images"])
        selection = load_json(selection_path)
        if selection["scan_id"] != scan_id or selection["coordinate_frame"] != "phase07_aligned_world":
            raise ValueError(f"Inconsistent photo selection: {selection_path}")
        candidates = [(name, *geometry) for (owner, name), geometry in geometries.items() if owner == scan_id]
        for photo in selection["images"]:
            position = np.asarray(photo["camera_to_world"], dtype=float)[:3, 3].tolist()
            floor = assign_floor(position, candidates)
            yaw, edges = photo_direction(photo)
            image_path = selection_path.parent / photo["image"]
            with Image.open(image_path) as image:
                if image.size != (photo["width"], photo["height"]):
                    raise ValueError(f"Photo dimensions do not match calibration: {image_path}")
            cameras.append({"scan": scan_id, "floor": floor, "kind": "photo", "source_id": photo["image_id"],
                            "position": position, "yaw": yaw, "edges": edges,
                            "source": str(image_path)})
    if pano_manifest.get("scans") is None:
        raise ValueError("Missing panorama scan reports")
    for scan in pano_manifest["scans"]:
        for floor in scan["floors"]:
            report_path = Path(floor["report"])
            report = load_json(report_path)
            scan_id, level = scan["scan_id"], floor["floor_level_name"]
            convention = report["panorama"]
            if (report["scan_id"] != scan_id or report["floor_level_name"] != level or report["coordinate_frame"] != "phase07_aligned_world" or
                    any(convention.get(key) != value for key, value in {"projection": "equirectangular", "longitude_zero": "+X", "positive_longitude": "+Z", "up": "+Y"}.items())):
                raise ValueError(f"Inconsistent panorama geometry: {report_path}")
            if (scan_id, level) not in geometries:
                raise ValueError(f"No attributed floor for panorama report: {report_path}")
            for point in report["points"]:
                candidates = [(level, *geometries[(scan_id, level)])]
                if assign_floor(point["position_world"], candidates) != level:
                    raise ValueError(f"Panorama has no support on its stated floor: {report_path}")
                panorama_images = {}
                for kind in ("gaussian", "mesh"):
                    image_path = report_path.parent / point[f"{kind}_image"]
                    with Image.open(image_path) as image:
                        if image.size != (convention["width"], convention["height"]) or image.width != 2 * image.height:
                            raise ValueError(f"Panorama dimensions do not match its metadata: {image_path}")
                    panorama_images[kind] = image_path
                cameras.append({"scan": scan_id, "floor": level, "kind": "panorama", "source_id": point["id"],
                                "position": point["position_world"], "yaw": 0, "edges": [-math.pi/4, math.pi/4],
                                "source": str(panorama_images["gaussian"]),
                                "mesh_source": str(panorama_images["mesh"])})
    if not cameras:
        raise ValueError("No cameras were selected")
    for camera in cameras:
        if (camera["scan"], camera["floor"]) not in geometries or camera["floor"] not in levels or not Path(camera["source"]).is_file():
            raise ValueError(f"Missing or inconsistent camera input: {camera}")
    cameras.sort(key=lambda c: (c["scan"], c["floor"], c["kind"] != "photo", c["source_id"]))
    counts = defaultdict(int)
    for camera in cameras:
        counts[camera["scan"]] += 1
        camera["number"] = counts[camera["scan"]]
        camera["pixels"] = {style: project(camera["position"], levels[camera["floor"]]["styles"][style]["grid"]) for style in STYLES}
    output = artifacts / "Generate Interactive Viewer"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("assets", "static"):
        shutil.rmtree(output / name, ignore_errors=True)
        (output / name).mkdir()
    for file in ("index.html", "viewer.js", "viewer.css"):
        shutil.copy2(STAGE / "interactive_viewer" / file, output / file)
    index_path = output / "index.html"
    index = index_path.read_text(encoding="utf-8")
    script_url = versioned_asset_url("viewer.js", output)
    index_path.write_text(index.replace('src="viewer.js"', f'src="{script_url}"'), encoding="utf-8")
    images = {}
    for level in levels.values():
        dest = output / "assets" / f"{level['name']}_measurements.png"
        row = measurement_sources[level["name"]]
        grid = level["measurements"]["grid"]
        flipped = _flip_measurements_vertically(row["measurements"], grid["height"])
        _write_measurement_overlay_png(dest, flipped, grid["width"])
        level["measurements"]["image"] = dest.relative_to(output).as_posix()
        for style, info in level["styles"].items():
            source = Path(info["source"])
            if not source.is_file():
                raise FileNotFoundError(f"Missing floorplan: {source}")
            dest = output / "assets" / f"{level['name']}_{style}.png"
            if style == "objects":
                image = Image.fromarray(_render_svg_png(source, info["grid"]["width"])[:, :, ::-1])
            else:
                image = Image.open(source)
            _flip_floorplan_vertically(image).save(dest)
            image.close()
            images[(level["name"], style)] = Image.open(dest).convert("RGB")
            info["image"] = dest.relative_to(output).as_posix()
            info["crops"] = {scan: crop_for(points, info["grid"]) for (scan, floor), (_, points) in geometries.items() if floor == level["name"]}
    for camera in cameras:
        source = Path(camera["source"])
        dest = output / "assets" / f"{camera['scan']}_{camera['number']:03d}{source.suffix.lower()}"
        shutil.copy2(source, dest)
        camera["image"] = dest.relative_to(output).as_posix()
        if camera["kind"] == "panorama":
            source = Path(camera["mesh_source"])
            dest = output / "assets" / f"{camera['scan']}_{camera['number']:03d}_mesh{source.suffix.lower()}"
            shutil.copy2(source, dest)
            camera["mesh_image"] = dest.relative_to(output).as_posix()
    for level in levels.values():
        rows = [c for c in cameras if c["floor"] == level["name"]]
        for style in STYLES:
            dest = output / "static" / "plans" / level["name"] / f"{style}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            marked_plan(images[(level["name"], style)], rows, style).save(dest)
    for camera in cameras:
        level = levels[camera["floor"]]
        base = Image.open(output / camera["image"]).convert("RGB")
        if camera["kind"] == "panorama":
            original = output / "static" / camera["scan"] / f"{camera['number']:03d}_panorama" / "original.jpg"
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output / camera["image"], original)
        for style in STYLES:
            crop = level["styles"][style]["crops"][camera["scan"]]
            minimap = marked_plan(images[(camera["floor"], style)], [camera], style, current=camera, crop=crop)
            minimap.thumbnail((max(200, base.width // 4), max(160, base.height // 3)))
            result = base.copy()
            pad = max(10, base.width // 100)
            draw = ImageDraw.Draw(result)
            x, y = base.width - minimap.width - pad, pad * 3
            draw.rectangle((x-4, y-4, x+minimap.width+4, y+minimap.height+4), fill="white")
            result.paste(minimap, (x, y))
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(18, base.width // 65))
            draw.text((pad, pad), f"{camera['scan']}  camera {camera['number']}  |  {camera['floor']}", font=font, fill="white", stroke_width=2, stroke_fill="black")
            dest = output / "static" / camera["scan"] / f"{camera['number']:03d}_{camera['kind']}" / f"{style}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            result.save(dest, quality=94, subsampling=0)
    data = {"levels": [{**level, "styles": {style: {k: v for k, v in info.items() if k != "source"} for style, info in level["styles"].items()}} for level in levels.values()],
            "cameras": [{k: versioned_asset_url(v, output) if k in {"image", "mesh_image"} else v
                         for k, v in camera.items() if k not in {"source", "mesh_source"}} for camera in cameras], "styles": list(STYLES),
            "measurement_warning": measurements["warning_text"]}
    (output / "tour.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    result = {"schema_version": 1, "phase_status": "in_progress", "macro_step": "Generate Interactive Viewer", "macro_step_status": "complete", "source_manifests": [str(p) for p in manifests], "sources": [{"scan": c["scan"], "number": c["number"], "source": c["source"], **({"mesh_source": c["mesh_source"]} if "mesh_source" in c else {})} for c in cameras], "camera_order": "scan ID, floor name, photographs by image ID, panoramas by point ID; numbering restarts per scan", "tour_data": "tour.json", "entry_point": "index.html", "static_dir": "static", "camera_count": len(cameras), "floor_count": len(levels), "elapsed_seconds": round(time.perf_counter()-started, 3)}
    (output / "generate_interactive_viewer_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    LOG.info("Generated %s cameras on %s levels in %.2f s", len(cameras), len(levels), time.perf_counter()-started)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE)
    args = parser.parse_args()
    output = args.scene_dir.resolve() / "artifacts/Generate Interactive Viewer"
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(output / "generate_interactive_viewer.log")])
    manifest = output / "generate_interactive_viewer_manifest.json"
    manifest.write_text(json.dumps({"macro_step": "Generate Interactive Viewer", "macro_step_status": "running"}) + "\n")
    try:
        generate(args.scene_dir)
    except Exception as error:
        manifest.write_text(json.dumps({"macro_step": "Generate Interactive Viewer", "macro_step_status": "failed", "error": str(error)}) + "\n")
        LOG.exception("Generate Interactive Viewer failed")
        raise


if __name__ == "__main__":
    main()
