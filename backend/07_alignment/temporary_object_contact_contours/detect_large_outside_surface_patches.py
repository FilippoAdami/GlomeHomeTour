#!/usr/bin/env python3
"""Highlight edge-connected object surface patches outside each scan's room mesh."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


BACKEND = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE = BACKEND / "current_scene"
AXIS_NORMAL_COSINE = 0.98
OUTSIDE_EPSILON_M = 0.001
MIN_PATCH_AREA_M2 = 0.2
QUERY_CHUNK_SIZE = 100_000
RED = np.array([255, 0, 0], dtype=np.uint8)


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"Expected a non-empty triangle mesh: {path}")
    if mesh.visual.kind != "vertex" or len(mesh.visual.vertex_colors) != len(mesh.vertices):
        raise ValueError(f"Expected vertex colors on mesh: {path}")
    return mesh


def regularized_source(scan_dir: Path) -> Path:
    pristine = scan_dir / ".raw_unaligned" / "mesh_elaboration" / "complete.ply"
    return pristine if pristine.is_file() else scan_dir / "mesh_elaboration" / "complete.ply"


def weighted_quantile(values: np.ndarray, weights: np.ndarray, fraction: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(cumulative, fraction * cumulative[-1])]])


def room_center(mesh: trimesh.Trimesh) -> np.ndarray:
    centers = mesh.triangles_center
    areas = mesh.area_faces
    return np.array([
        0.5 * (weighted_quantile(centers[:, axis], areas, 0.02)
               + weighted_quantile(centers[:, axis], areas, 0.98))
        for axis in range(3)
    ])


def closest_surface_scene(mesh: trimesh.Trimesh):
    normals = mesh.face_normals
    axes = np.abs(normals).argmax(axis=1)
    aligned = np.abs(normals[np.arange(len(normals)), axes]) >= AXIS_NORMAL_COSINE
    face_ids = np.flatnonzero(aligned)
    if not len(face_ids):
        raise ValueError("Regularized mesh has no world-axis-aligned faces")
    axis_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(mesh.faces[face_ids], dtype=np.uint32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(axis_mesh)
    return scene, axes[face_ids], mesh.triangles_center[face_ids], normals[face_ids]


def outside_object_faces(mesh: trimesh.Trimesh, scene, room_midpoint: np.ndarray,
                         face_axes: np.ndarray, face_centers: np.ndarray,
                         face_normals: np.ndarray) -> np.ndarray:
    centers = mesh.triangles_center
    outside = np.zeros(len(mesh.faces), dtype=bool)
    for start in range(0, len(centers), QUERY_CHUNK_SIZE):
        stop = min(start + QUERY_CHUNK_SIZE, len(centers))
        query = o3d.core.Tensor(centers[start:stop].astype(np.float32))
        closest = scene.compute_closest_points(query)
        points = closest["points"].numpy()
        nearest = closest["primitive_ids"].numpy().astype(np.int64)
        axes = face_axes[nearest]
        signs = np.sign(face_centers[nearest, axes] - room_midpoint[axes])
        ambiguous = signs == 0
        if np.any(ambiguous):
            signs[ambiguous] = np.sign(face_normals[nearest[ambiguous], axes[ambiguous]])
        direction = np.zeros_like(points)
        direction[np.arange(len(points)), axes] = signs
        distance_out = np.einsum("ij,ij->i", centers[start:stop] - points, direction)
        outside[start:stop] = distance_out > OUTSIDE_EPSILON_M
    return outside


def qualifying_patches(mesh: trimesh.Trimesh, outside: np.ndarray) -> tuple[np.ndarray, list[float]]:
    topology = mesh.copy()
    topology.merge_vertices(digits_vertex=6)
    if len(topology.faces) != len(mesh.faces):
        raise ValueError("Vertex merging changed object face count")
    edges = topology.face_adjacency
    connected = edges[outside[edges].all(axis=1)] if len(edges) else np.empty((0, 2), dtype=np.int64)
    rows = np.concatenate((connected[:, 0], connected[:, 1]))
    columns = np.concatenate((connected[:, 1], connected[:, 0]))
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
        shape=(len(mesh.faces), len(mesh.faces)),
    ).tocsr()
    component_count, labels = connected_components(graph, directed=False)
    component_areas = np.bincount(
        labels[outside], weights=mesh.area_faces[outside], minlength=component_count
    )
    selected = outside & (component_areas[labels] >= MIN_PATCH_AREA_M2)
    patch_labels = np.unique(labels[selected])
    patch_areas = sorted(float(component_areas[label]) for label in patch_labels)
    if np.any(selected & ~outside) or any(area < MIN_PATCH_AREA_M2 for area in patch_areas):
        raise AssertionError("Outside-patch selection violated its area threshold")
    return selected, patch_areas


def original_aligned_object(alignment_dir: Path, scan: str, source_path: str) -> Path:
    relative = Path(source_path)
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 3:
        raise ValueError(f"Invalid contact-contour source path: {source_path!r}")
    if relative.parts[0] == scan:
        relative = Path(*relative.parts[1:])
    instance_name = relative.name.removesuffix("_contact_contour.ply") + ".ply"
    return alignment_dir / "objects" / scan / relative.parent / instance_name


def colored_object(mesh: trimesh.Trimesh, original: trimesh.Trimesh,
                   red_faces: np.ndarray) -> trimesh.Trimesh:
    if len(mesh.faces) != len(original.faces):
        raise ValueError("Translated object face order/count does not match its aligned source")
    triangles = mesh.vertices[mesh.faces]
    colors = np.asarray(original.visual.vertex_colors)[original.faces].copy()
    if colors.shape[1] < 3:
        raise ValueError("Aligned source object has no RGB color channels")
    colors[red_faces] = np.array([*RED, *([255] if colors.shape[2] == 4 else [])], dtype=colors.dtype)
    result = trimesh.Trimesh(
        vertices=triangles.reshape(-1, 3),
        faces=np.arange(len(mesh.faces) * 3, dtype=np.int64).reshape(-1, 3),
        process=False,
    )
    result.visual.vertex_colors = colors.reshape(-1, colors.shape[2])
    return result


def run(alignment_dir: Path, axis_dir: Path, output_dir: Path) -> dict:
    alignment_dir = alignment_dir.resolve()
    axis_dir = axis_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir in (alignment_dir, axis_dir) or output_dir == axis_dir.parent:
        raise ValueError("Output must be a separate temporary directory")
    transforms_path = alignment_dir / "stats" / "aligned_scans_transforms.json"
    axis_report_path = axis_dir / "axis_plane_alignment_report.json"
    if not transforms_path.is_file() or not axis_report_path.is_file():
        raise FileNotFoundError(f"Expected {transforms_path} and {axis_report_path}")

    transforms = json.loads(transforms_path.read_text())["scans"]
    axis_report = json.loads(axis_report_path.read_text())
    grouped: dict[str, list[dict]] = {}
    for record in axis_report["objects"]:
        grouped.setdefault(record["scan"], []).append(record)

    records, output_meshes = [], []
    for scan, instances in sorted(grouped.items()):
        scan_dir = alignment_dir.parent / scan
        regularized_path = regularized_source(scan_dir)
        regularized = load_mesh(regularized_path)
        regularized.apply_transform(np.asarray(transforms[scan]["transform_matrix"], dtype=np.float64))
        scene, face_axes, face_centers, face_normals = closest_surface_scene(regularized)
        midpoint = room_center(regularized)

        for instance in instances:
            input_relative = Path(instance["output_path"])
            if input_relative.is_absolute() or ".." in input_relative.parts:
                raise ValueError(f"Invalid aligned-object path: {instance['output_path']!r}")
            source = axis_dir / input_relative
            mesh = load_mesh(source)
            outside = outside_object_faces(
                mesh, scene, midpoint, face_axes, face_centers, face_normals
            )
            selected, patch_areas = qualifying_patches(mesh, outside)
            original = load_mesh(original_aligned_object(
                alignment_dir, instance["scan"], instance["source_path"]
            ))
            colored = colored_object(mesh, original, selected)

            target = output_dir / "objects" / input_relative.relative_to("objects")
            target.parent.mkdir(parents=True, exist_ok=True)
            colored.export(str(target))
            output_meshes.append(colored)
            records.append({
                "scan": scan,
                "source_path": str(source.relative_to(axis_dir)),
                "output_path": str(target.relative_to(output_dir)),
                "outside_triangle_count_before_area_filter": int(outside.sum()),
                "outside_surface_area_before_area_filter_m2": float(mesh.area_faces[outside].sum()),
                "qualifying_patch_count": len(patch_areas),
                "qualifying_patch_areas_m2": patch_areas,
                "red_triangle_count": int(selected.sum()),
                "red_surface_area_m2": float(mesh.area_faces[selected].sum()),
            })

    if not output_meshes:
        raise ValueError("No translated object instances were found")
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "aligned_objects_large_outside_surface_patches_red.ply"
    trimesh.util.concatenate(output_meshes).export(str(combined_path))
    report = {
        "status": "large_object_surface_patches_outside_parent_regularized_mesh",
        "outside_rule": "Object triangle centroids more than 1 mm beyond their nearest axis-aligned regularized face; outward side is estimated from the midpoint of area-weighted 2nd and 98th percentiles of regularized triangle centers.",
        "patch_connectivity": "Object triangles connected by shared mesh edges.",
        "patch_area_rule": f"Sum of object triangle surface areas >= {MIN_PATCH_AREA_M2:.3f} m^2 per connected outside patch.",
        "regularized_reference": str(alignment_dir / "aligned_regularized_meshes.ply"),
        "regularized_geometry_exported_or_modified": False,
        "objects_only_output": True,
        "combined_objects_ply": combined_path.name,
        "outside_epsilon_m": OUTSIDE_EPSILON_M,
        "axis_face_normal_cosine_threshold": AXIS_NORMAL_COSINE,
        "minimum_patch_surface_area_m2": MIN_PATCH_AREA_M2,
        "objects_with_red_patches": sum(item["red_triangle_count"] > 0 for item in records),
        "red_triangle_count": sum(item["red_triangle_count"] for item in records),
        "red_surface_area_m2": sum(item["red_surface_area_m2"] for item in records),
        "objects": records,
    }
    (output_dir / "outside_surface_detection_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--alignment-dir", type=Path)
    parser.add_argument("--axis-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    alignment_dir = (args.alignment_dir or workspace / "multi_room_alignment").resolve()
    axis_dir = (args.axis_dir or alignment_dir / "temporary_object_contact_contours"
                / "axis_plane_alignment").resolve()
    output_dir = (args.output_dir or alignment_dir / "temporary_object_contact_contours"
                  / "outside_surface_detection").resolve()
    report = run(alignment_dir, axis_dir, output_dir)
    print(json.dumps({
        "status": report["status"],
        "objects": len(report["objects"]),
        "objects_with_red_patches": report["objects_with_red_patches"],
        "red_triangles": report["red_triangle_count"],
        "red_surface_area_m2": report["red_surface_area_m2"],
        "combined_objects_ply": str(output_dir / report["combined_objects_ply"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
