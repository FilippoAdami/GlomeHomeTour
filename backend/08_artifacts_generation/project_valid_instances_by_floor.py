#!/usr/bin/env python3
"""Project valid segmented object instances over natural refined floors."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import open3d as o3d
from shapely.geometry import MultiPoint, Polygon, mapping
from shapely.ops import unary_union

BACKEND_DIR = Path(__file__).resolve().parents[1]
SCENE_DIR = BACKEND_DIR / "current_scene" / "06_mesh_elaboration"
DEFAULT_CANDIDATE = SCENE_DIR / "per_level_collapse_candidate"
DEFAULT_REFINED_STRUCTURAL_MESH = SCENE_DIR / "new_step" / "all_structural_regularized.ply"
DEFAULT_SEGMENTATION = BACKEND_DIR / "current_scene" / "scan_0" / "segmentation"
DEFAULT_ASSOCIATIONS = DEFAULT_SEGMENTATION / "non_structural_similarity" / "valid_class_associations.json"
DEFAULT_OUTPUT = BACKEND_DIR / "current_scene" / "artifacts" / "floorplan" / "instance_projections"


def _load_segmented_structure(segmentation_dir: Path, metadata: dict):
    class_ids = {role: sorted(int(class_id) for class_id, item in metadata.items()
                              if item.get("structural_role") == role)
                 for role in ("floor", "wall")}
    if not class_ids["wall"]:
        raise ValueError(f"Segmentation package has no wall class: {class_ids}")

    vertices, faces, normals = [], [], []
    offset = 0
    for class_id in class_ids["wall"]:
        class_dir = segmentation_dir / "3d_masks" / f"class_{class_id:03d}_wall"
        instance_paths = sorted(class_dir.glob("instance_*.ply"))
        if not instance_paths:
            raise FileNotFoundError(f"No segmented wall instances found: {class_dir}")
        for path in instance_paths:
            mesh = o3d.io.read_triangle_mesh(str(path))
            if mesh.is_empty() or not len(mesh.triangles):
                raise ValueError(f"Segmented wall instance is empty: {path}")
            mesh.compute_triangle_normals()
            vertices.append(np.asarray(mesh.vertices))
            faces.append(np.asarray(mesh.triangles) + offset)
            normals.append(np.asarray(mesh.triangle_normals))
            offset += len(mesh.vertices)
    return np.concatenate(vertices), np.concatenate(faces), np.concatenate(normals), class_ids


def _horizontal_footprint(mesh: o3d.geometry.TriangleMesh, floor_y: float, path: Path):
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-12)
    selected = (np.abs(triangles[:, :, 1].mean(axis=1) - floor_y) < 0.08) & (np.abs(normals[:, 1]) > 0.9)
    pieces = [Polygon(triangle[:, (0, 2)]) for triangle in triangles[selected]]
    footprint = unary_union([piece for piece in pieces if piece.area > 1e-10])
    if footprint.is_empty or footprint.area <= 0:
        raise ValueError(f"No horizontal floor triangles found near Y={floor_y:.3f} m in {path}")
    components = [footprint] if isinstance(footprint, Polygon) else list(footprint.geoms)
    return max(components, key=lambda polygon: polygon.area)


def _refined_floors(levels: list[dict], refined_mesh_path: Path) -> list[dict]:
    if not refined_mesh_path.exists():
        raise FileNotFoundError(f"Refined structural mesh is missing: {refined_mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(refined_mesh_path))
    if mesh.is_empty() or not len(mesh.triangles):
        raise ValueError(f"Refined structural mesh is empty: {refined_mesh_path}")
    results = []
    for level in levels:
        for unit in level.get("floor_units", []):
            name = unit["unit_name"]
            floor_y = float(unit.get("mean_elevation_y", level["mean_floor_elevation_y"]))
            footprint = _horizontal_footprint(mesh, floor_y, refined_mesh_path)
            results.append({"unit_name": name, "source_path": str(refined_mesh_path), "footprint": footprint})
    return results


def _refined_floor_coverage(instances: list[dict], footprint) -> dict:
    inside = [[item["source_class_id"], item["instance_id"]]
              for item in instances
              if footprint.covers(MultiPoint(np.asarray(item["mesh"].vertices)[:, (0, 2)]).convex_hull)]
    return {"fully_inside_refined_floor": len(inside), "total": len(instances), "instance_ids": inside}


def _instance_floor_score(mesh: o3d.geometry.TriangleMesh, floor_y: float, footprint: object) -> tuple[float, bool]:
    vertices = np.asarray(mesh.vertices)
    xz = vertices[:, (0, 2)]
    bottom_y = float(np.quantile(vertices[:, 1], 0.10))
    footprint_match = footprint.buffer(0.20).intersects(MultiPoint(xz).convex_hull)
    return abs(bottom_y - floor_y), bool(footprint_match)


def _draw_floor_shape(ax, geometry, facecolor: str, alpha: float = 0.3,
                      edgecolor: str | None = None, linewidth: float = 0.8) -> None:
    polygons = [geometry] if isinstance(geometry, Polygon) else list(geometry.geoms)
    for polygon in polygons:
        ax.fill(*polygon.exterior.xy, facecolor=facecolor, edgecolor=edgecolor or facecolor,
                linewidth=linewidth, alpha=alpha, zorder=0)
        for ring in polygon.interiors:
            ax.fill(*ring.xy, facecolor="white", edgecolor="none", zorder=0)


def _add_story_walls(ax, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray,
                     bottom_y: float, top_y: float) -> np.ndarray:
    centers_y = vertices[faces, 1].mean(axis=1)
    mask = (np.abs(normals[:, 1]) < 0.65) & (centers_y >= bottom_y - 0.08) & (centers_y <= top_y + 0.08)
    selected = faces[mask]
    if not len(selected):
        return np.empty((0, 2, 2))
    edges = np.concatenate((selected[:, [0, 1]], selected[:, [1, 2]], selected[:, [2, 0]]))
    segments = vertices[edges][:, :, (0, 2)]
    lengths = np.linalg.norm(segments[:, 0] - segments[:, 1], axis=1)
    segments = segments[lengths > 0.01]
    if len(segments) > 180_000:
        segments = segments[np.linspace(0, len(segments) - 1, 180_000, dtype=np.int64)]
    if len(segments):
        ax.add_collection(LineCollection(segments, colors="#343a40", linewidths=0.35, alpha=0.72, zorder=1))
    return segments


def _render_floor(output_path: Path, level_index: int, floor_y: float, footprint,
                  wall_vertices: np.ndarray, wall_faces: np.ndarray, wall_normals: np.ndarray,
                  instances: list[dict], text_names: dict[int, str], dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(11, 11), dpi=dpi)
    _draw_floor_shape(ax, footprint, "#1f77b4", alpha=0.3, edgecolor="#1f77b4")

    ceilings = [item["ceiling_match"].get("ceiling_elevation_y_at_overlap")
                for instance in instances for item in instance["floor_units"] if item.get("ceiling_match")]
    top_y = max(ceilings) if ceilings else floor_y + 3.0
    wall_segments = _add_story_walls(ax, wall_vertices, wall_faces, wall_normals, floor_y, top_y)

    palette = plt.get_cmap("tab20")
    legend_items = {}
    all_points = []
    bbox_count = 0
    if len(wall_segments):
        all_points.append(wall_segments.reshape(-1, 2))
    for item_index, item in enumerate(instances):
        vertices = np.asarray(item["mesh"].vertices)
        xz = vertices[:, (0, 2)]
        color = palette((item["text_class_id"] - 1) % 20)
        ax.scatter(xz[:, 0], xz[:, 1], s=0.25, color=[color], alpha=0.62, linewidths=0, rasterized=True, zorder=3)
        hull = MultiPoint(xz).convex_hull
        if isinstance(hull, Polygon):
            ax.plot(*hull.exterior.xy, color=color, linewidth=0.8, alpha=0.9, zorder=4)
        centroid = hull.centroid if not hull.is_empty else MultiPoint(xz).centroid
        label = f"{item['source_class_id']}/{item['instance_id']}"
        if item["source_class_id"] != 193:
            box = hull.minimum_rotated_rectangle
            if isinstance(box, Polygon):
                corners = np.asarray(box.exterior.coords)
                edge_lengths = np.linalg.norm(np.diff(corners, axis=0), axis=1)
                long_side, short_side = float(edge_lengths[:4].max()), float(edge_lengths[:4].min())
                ax.plot(*box.exterior.xy, color=color, linewidth=1.1, alpha=0.95, zorder=5)
                label += f"\n{long_side * 100:.0f} × {short_side * 100:.0f} cm"
                centroid = box.centroid
                bbox_count += 1
        ax.annotate(label, (centroid.x, centroid.y),
                    xytext=(2, 2), textcoords="offset points", fontsize=5.5, color="#111111",
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.72, "edgecolor": "none"}, zorder=5)
        legend_items[item["text_class_id"]] = color
        all_points.append(xz)

    if all_points:
        points = np.concatenate(all_points)
        min_x, min_z = points.min(axis=0)
        max_x, max_z = points.max(axis=0)
    else:
        min_x, min_z, max_x, max_z = footprint.bounds
    fx0, fz0, fx1, fz1 = footprint.bounds
    min_x, max_x = min(min_x, fx0), max(max_x, fx1)
    min_z, max_z = min(min_z, fz0), max(max_z, fz1)
    pad = max(max_x - min_x, max_z - min_z, 1.0) * 0.06
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(max_z + pad, min_z - pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Floor {level_index} — Y collapsed to {floor_y:.3f} m", fontsize=14)
    ax.axis("off")
    handles = [Line2D([0], [0], color="#343a40", linewidth=1, label="Segmented walls"),
               Patch(facecolor="#1f77b4", edgecolor="#1f77b4", alpha=0.3,
                     label="Natural refined floor")]
    if bbox_count:
        handles.append(Line2D([0], [0], color="#555555", linewidth=1.1,
                              label="Oriented instance box; dimensions in cm (class 193 omitted)"))
    handles.extend(Line2D([0], [0], marker="o", linestyle="", color=color,
                          label=f"Text {class_id}: {text_names[class_id]}", markersize=6)
                   for class_id, color in sorted(legend_items.items()))
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _render_floor_overlay(output_path: Path, floors: dict[int, dict], dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(13, 13), dpi=dpi)
    colors = {0: "#d62728", 1: "#1f77b4"}
    all_geometries = []
    for floor_index in (0, 1):
        # Remove collinear triangle-boundary vertices so labels describe outline segments.
        geometry = floors[floor_index]["footprint"].simplify(0.005, preserve_topology=True)
        color = colors[floor_index]
        _draw_floor_shape(ax, geometry, color, alpha=0.22, edgecolor=color, linewidth=1.3)
        polygons = [geometry] if isinstance(geometry, Polygon) else list(geometry.geoms)
        for polygon in polygons:
            for ring in (polygon.exterior, *polygon.interiors):
                coordinates = np.asarray(ring.coords)
                for start, end in zip(coordinates[:-1], coordinates[1:]):
                    delta = end - start
                    length = float(np.linalg.norm(delta))
                    if length <= 1e-8:
                        continue
                    outward = np.array([delta[1], -delta[0]]) / length
                    if not ring.is_ccw:
                        outward *= -1
                    midpoint = (start + end) / 2 + outward * (0.035 + 0.045 * floor_index)
                    angle = float(np.degrees(np.arctan2(-delta[1], delta[0])))
                    if angle > 90:
                        angle -= 180
                    elif angle < -90:
                        angle += 180
                    ax.text(midpoint[0], midpoint[1], f"{length * 100:.0f} cm", color=color,
                            fontsize=6.5, ha="center", va="center", rotation=angle,
                            rotation_mode="anchor",
                            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 0.8},
                            zorder=4)
        all_geometries.append(geometry)

    x0 = min(geometry.bounds[0] for geometry in all_geometries)
    z0 = min(geometry.bounds[1] for geometry in all_geometries)
    x1 = max(geometry.bounds[2] for geometry in all_geometries)
    z1 = max(geometry.bounds[3] for geometry in all_geometries)
    pad = max(x1 - x0, z1 - z0, 1.0) * 0.10
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(z1 + pad, z0 - pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Natural refined floor outlines — segment lengths", fontsize=14)
    ax.legend(handles=[Patch(facecolor=colors[0], edgecolor=colors[0], alpha=0.3, label="Floor 0"),
                       Patch(facecolor=colors[1], edgecolor=colors[1], alpha=0.3, label="Floor 1")],
              loc="upper left")
    ax.grid(alpha=0.18)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--refined-structural-mesh", type=Path, default=DEFAULT_REFINED_STRUCTURAL_MESH)
    parser.add_argument("--associations", type=Path, default=DEFAULT_ASSOCIATIONS)
    parser.add_argument("--segmentation-output", type=Path, default=DEFAULT_SEGMENTATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    levels_path = args.candidate_dir / "structural_levels_manifest.json"
    metadata_path = args.segmentation_output / "class_metadata.json"
    for path in (levels_path, args.segmentation_output / "3d_masks", metadata_path, args.associations):
        if not path.exists():
            raise FileNotFoundError(f"Required floor projection input is missing: {path}")

    levels_manifest = json.loads(levels_path.read_text())
    metadata = json.loads(metadata_path.read_text())["classes"]
    wall_vertices, wall_faces, wall_normals, structural_class_ids = _load_segmented_structure(
        args.segmentation_output, metadata)
    levels = levels_manifest.get("levels", [])
    consistency = levels_manifest.get("consistency_report", {})
    floor_count = len(levels)
    for key in ("scene_size_floors", "detected_levels"):
        expected = consistency.get(key)
        if expected is not None and int(expected) != floor_count:
            raise RuntimeError(f"Floor count mismatch: manifest has {floor_count} levels, {key} says {expected}")
    if floor_count == 0:
        raise RuntimeError("No floors were found in structural_levels_manifest.json")

    association_data = json.loads(args.associations.read_text())
    associations = {int(image_id): int(text_id) for image_id, text_id in association_data["associations"]}
    if not associations:
        raise RuntimeError(f"No valid class associations in {args.associations}")
    text_classes = {int(item["id"]): item["name"] for item in json.loads((args.segmentation_output / "non_structural_similarity" / "categories.json").read_text())}

    refined_units = _refined_floors(levels, args.refined_structural_mesh)
    refined_by_name = {unit["unit_name"]: unit for unit in refined_units}
    floor_data = []
    for level in levels:
        units = level.get("floor_units", [])
        floor_y = float(level["mean_floor_elevation_y"])
        level_units = [refined_by_name[unit["unit_name"]] for unit in units]
        if not level_units:
            raise ValueError(f"No refined floor units were found for floor {level['level_index']}")
        footprint = unary_union([unit["footprint"] for unit in level_units])
        floor_data.append({"level_index": int(level["level_index"]), "floor_y": floor_y,
                           "floor_units": units, "refined_units": level_units,
                           "footprint": footprint, "instances": [],
                           "refined_floor_bounds": footprint.bounds, "refined_floor_coverage": None})

    assigned_paths = set()
    assignment_report = []
    for source_class_id, text_class_id in sorted(associations.items()):
        class_info = metadata.get(str(source_class_id))
        if class_info is None or class_info.get("structural_role") != "object":
            raise ValueError(f"Valid image class {source_class_id} is not an object class in class_metadata.json")
        class_dir = args.segmentation_output / "3d_masks" / f"class_{source_class_id:03d}_{class_info['structural_role']}"
        instance_paths = sorted(class_dir.glob("instance_*.ply"))
        if not instance_paths:
            raise FileNotFoundError(f"No segmented instances found for valid image class {source_class_id}: {class_dir}")
        for instance_path in instance_paths:
            match = re.fullmatch(r"instance_(\d+)\.ply", instance_path.name)
            if match is None:
                raise ValueError(f"Unexpected segmented instance path: {instance_path}")
            mesh = o3d.io.read_triangle_mesh(str(instance_path))
            if mesh.is_empty() or not len(mesh.vertices):
                raise ValueError(f"Segmented object instance is empty: {instance_path}")
            scores = [_instance_floor_score(mesh, floor["floor_y"], floor["footprint"]) for floor in floor_data]
            chosen = min(range(floor_count), key=lambda idx: scores[idx][0])
            floor = floor_data[chosen]
            instance_id = int(match.group(1))
            if instance_path in assigned_paths:
                raise RuntimeError(f"Instance assigned more than once: {instance_path}")
            assigned_paths.add(instance_path)
            item = {"mesh": mesh, "path": str(instance_path), "source_class_id": source_class_id,
                    "text_class_id": text_class_id, "instance_id": instance_id, "floor_units": floor["floor_units"]}
            floor["instances"].append(item)
            assignment_report.append({"source_class_id": source_class_id, "text_class_id": text_class_id,
                                      "text_class": text_classes.get(text_class_id, "unknown"),
                                      "instance_id": instance_id, "path": str(instance_path),
                                      "floor_index": floor["level_index"], "floor_elevation_y": floor["floor_y"],
                                      "bottom_to_floor_distance_m": scores[chosen][0],
                                      "footprint_intersection": scores[chosen][1]})
            if not scores[chosen][1] or scores[chosen][0] > 0.75:
                logging.warning("Review floor assignment for %s: distance %.3f m, footprint intersection=%s",
                                instance_path, scores[chosen][0], scores[chosen][1])

    for floor in floor_data:
        floor["refined_floor_coverage"] = _refined_floor_coverage(floor["instances"], floor["footprint"])
        report = floor["refined_floor_coverage"]
        logging.info("Floor %d: %d/%d complete instances fit within natural refined floor; positions preserved",
                     floor["level_index"], report["fully_inside_refined_floor"], report["total"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected_images = {f"floor_{floor['level_index'] + 1:02d}.png" for floor in floor_data}
    for existing in args.output_dir.glob("floor_*.png"):
        if existing.name not in expected_images:
            existing.unlink()
    for existing in args.output_dir.glob("alignment_floor_*.png"):
        existing.unlink()
    for floor in floor_data:
        output_path = args.output_dir / f"floor_{floor['level_index'] + 1:02d}.png"
        _render_floor(output_path, floor["level_index"], floor["floor_y"], floor["footprint"],
                      wall_vertices, wall_faces, wall_normals,
                      floor["instances"], text_classes, args.dpi)
        logging.info("Wrote %d valid instances for floor %d to %s", len(floor["instances"]),
                     floor["level_index"] + 1, output_path)

    floor_by_index = {floor["level_index"]: floor for floor in floor_data}
    overlay_paths = []
    if {0, 1}.issubset(floor_by_index):
        overlay_path = args.output_dir / "floor_0_floor_1_overlay.png"
        _render_floor_overlay(overlay_path, floor_by_index, max(args.dpi, 220))
        overlay_paths = [str(overlay_path), str(overlay_path.with_suffix(".pdf"))]

    report = {"floor_count": floor_count,
              "coordinate_frame": "raw segmented walls/objects and untransformed refined floor outlines",
              "source_structural_instances": str(args.segmentation_output / "3d_masks"),
              "source_refined_floors": str(args.refined_structural_mesh),
              "floor_overlay": overlay_paths,
              "source_object_instances": str(args.segmentation_output / "3d_masks"),
              "structural_class_ids_by_role": structural_class_ids,
              "valid_associations": [[image_id, text_id] for image_id, text_id in sorted(associations.items())],
              "excluded_source_class_ids": sorted(set(map(int, metadata)) - set(associations)),
              "floor_alignment_method": "natural refined floor outlines; no transform",
              "assignment_method": "Assign each raw instance to the closest floor elevation using its 10th-percentile Y; overlap with the natural refined floor is diagnostic only.",
              "floors": [{"floor_index": floor["level_index"], "floor_elevation_y": floor["floor_y"],
                          "instance_count": len(floor["instances"]),
                          "refined_floor_bounds_xz": list(floor["refined_floor_bounds"]),
                          "refined_floor_geometry_xz": mapping(floor["footprint"]),
                          "floor_adjustment": floor.get("floor_adjustment"),
                          "refined_floor_coverage": floor["refined_floor_coverage"],
                          "refined_floor_units": [{"unit_name": unit["unit_name"],
                                                   "source_path": unit["source_path"]}
                                                  for unit in floor["refined_units"]],
                          "image": f"floor_{floor['level_index'] + 1:02d}.png"}
                         for floor in floor_data],
              "instances": assignment_report}
    (args.output_dir / "projection_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"floor_count": floor_count,
                      "instances_per_floor": {f"floor_{f['level_index'] + 1:02d}": len(f["instances"]) for f in floor_data},
                      "images": [str(args.output_dir / f"floor_{f['level_index'] + 1:02d}.png") for f in floor_data],
                      "floor_overlay": overlay_paths}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
