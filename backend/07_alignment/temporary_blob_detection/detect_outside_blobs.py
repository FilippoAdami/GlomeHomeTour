#!/usr/bin/env python3
"""Color large aligned-object regions beyond regularized room walls red."""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np
import trimesh
from plyfile import PlyData, PlyElement
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


MIN_OUTSIDE_DISTANCE_M = 0.15
MIN_SECTION_AREA_M2 = 0.5
QUERY_CHUNK_SIZE = 200_000


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"Expected a non-empty triangle mesh: {path}")
    return mesh


def projected_outlier_faces(source: trimesh.Trimesh, regularized: trimesh.Trimesh) -> np.ndarray:
    """Return regularized wall faces reached by source triangles outside that room."""
    target_centers = regularized.triangles_center
    target_normals = regularized.face_normals
    wall_faces = np.flatnonzero(np.abs(target_normals[:, 1]) < 0.35)
    if not len(wall_faces):
        return np.zeros(len(regularized.faces), dtype=bool)

    room_center = regularized.bounds.mean(axis=0)

    # shortcut: the bounds center approximates outward direction; use room topology for concave rooms.
    outward = target_centers[wall_faces] - room_center
    outward[:, 1] = 0.0
    lengths = np.linalg.norm(outward, axis=1)
    valid = lengths > 1e-6
    wall_faces = wall_faces[valid]
    outward = outward[valid] / lengths[valid, None]
    tree = cKDTree(target_centers[wall_faces])
    supported = np.zeros(len(regularized.faces), dtype=bool)
    outside_source_triangles = 0

    for start in range(0, len(source.faces), QUERY_CHUNK_SIZE):
        stop = min(start + QUERY_CHUNK_SIZE, len(source.faces))
        tri = source.vertices[source.faces[start:stop]]
        centers = tri.mean(axis=1)
        _, nearest_local = tree.query(centers, workers=-1)
        nearest_faces = wall_faces[nearest_local]
        delta = centers - target_centers[nearest_faces]
        outside = np.einsum("ij,ij->i", delta, outward[nearest_local]) > MIN_OUTSIDE_DISTANCE_M
        outside_source_triangles += int(outside.sum())
        supported[nearest_faces[outside]] = True

    regularized_analysis = regularized.copy()
    regularized_analysis.merge_vertices()
    if len(regularized_analysis.faces) != len(regularized.faces):
        raise ValueError("Merging regularized vertices changed face order/count")

    edges = regularized_analysis.face_adjacency
    touching = supported[edges].all(axis=1) if len(edges) else np.zeros(0, dtype=bool)
    edges = edges[touching]
    rows = np.concatenate((edges[:, 0], edges[:, 1])) if len(edges) else np.empty(0, dtype=np.int64)
    cols = np.concatenate((edges[:, 1], edges[:, 0])) if len(edges) else np.empty(0, dtype=np.int64)
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(len(regularized.faces), len(regularized.faces)),
    ).tocsr()
    component_count, labels = connected_components(graph, directed=False)
    face_areas = regularized_analysis.area_faces
    component_areas = np.bincount(
        labels[supported], weights=face_areas[supported], minlength=component_count
    )
    selected = supported & (component_areas[labels] > MIN_SECTION_AREA_M2)
    selected_area = float(face_areas[selected].sum())
    print(
        f"  source outlier triangles={outside_source_triangles:,}; "
        f"supported faces={int(supported.sum()):,}; "
        f"red faces={int(selected.sum()):,}; section area={selected_area:.3f} m^2"
    )
    return selected


def write_red_copy(
    source_ply: Path,
    output_ply: Path,
    red_faces: np.ndarray,
    object_paths: list[Path],
) -> None:
    if source_ply.resolve() == output_ply.resolve():
        raise ValueError("Output must be separate from aligned_regularized_meshes.ply")

    data = PlyData.read(str(source_ply))
    if len(data["face"].data) != len(red_faces):
        raise ValueError("Detected face mask does not match aligned regularized PLY")

    vertices = data["vertex"].data.copy()
    for channel in ("red", "green", "blue"):
        if channel not in vertices.dtype.names:
            raise ValueError(f"Aligned PLY is missing vertex color channel: {channel}")

    face_vertices = data["face"].data["vertex_indices"][red_faces]
    if face_vertices.dtype == object:
        vertex_ids = np.concatenate(face_vertices) if len(face_vertices) else np.empty(0, dtype=np.int64)
    else:
        vertex_ids = face_vertices.reshape(-1)
    vertices["red"][vertex_ids] = 255
    vertices["green"][vertex_ids] = 0
    vertices["blue"][vertex_ids] = 0

    face_indices = [np.asarray(ids, dtype=np.int64).reshape(-1) for ids in data["face"].data["vertex_indices"]]
    all_vertices = [vertices]
    vertex_offset = len(vertices)
    for object_path in object_paths:
        object_data = PlyData.read(str(object_path))
        object_vertices = object_data["vertex"].data
        if object_vertices.dtype.names != vertices.dtype.names:
            raise ValueError(f"Object vertex properties do not match the regularized mesh: {object_path}")
        all_vertices.append(object_vertices)
        face_indices.extend(
            np.asarray(ids, dtype=np.int64).reshape(-1) + vertex_offset
            for ids in object_data["face"].data["vertex_indices"]
        )
        vertex_offset += len(object_vertices)

    combined_vertices = np.concatenate(all_vertices)
    combined_faces = np.empty(len(face_indices), dtype=[("vertex_indices", object)])
    combined_faces["vertex_indices"] = face_indices
    elements = [
        PlyElement.describe(combined_vertices, "vertex", comments=data["vertex"].comments),
        PlyElement.describe(combined_faces, "face", comments=data["face"].comments),
    ]
    colored = PlyData(
        elements,
        text=data.text,
        byte_order=data.byte_order,
        comments=data.comments,
        obj_info=data.obj_info,
    )
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_ply.parent, suffix=".ply", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        colored.write(str(temp_path))
        temp_path.replace(output_ply)
    finally:
        temp_path.unlink(missing_ok=True)


def run(alignment_dir: Path) -> Path:
    alignment_dir = alignment_dir.resolve()
    aligned_ply = alignment_dir / "aligned_regularized_meshes.ply"
    scans_dir = alignment_dir / "scans"
    objects_dir = alignment_dir / "objects"
    if not aligned_ply.is_file() or not scans_dir.is_dir() or not objects_dir.is_dir():
        raise FileNotFoundError(
            f"Expected aligned_regularized_meshes.ply, scans/, and objects/ under {alignment_dir}"
        )

    scan_dirs = sorted(path for path in scans_dir.glob("scan_*") if path.is_dir())
    if not scan_dirs:
        raise FileNotFoundError(f"No aligned scan directories found under {scans_dir}")

    aligned_data = PlyData.read(str(aligned_ply))
    total_faces = len(aligned_data["face"].data)
    red_faces = np.zeros(total_faces, dtype=bool)
    face_offset = 0
    all_object_paths = []

    for scan_dir in scan_dirs:
        regularized_path = scan_dir / "mesh_elaboration" / "complete.ply"
        if not regularized_path.is_file():
            raise FileNotFoundError(f"Missing per-scan regularized mesh: {regularized_path}")

        regularized = load_mesh(regularized_path)
        face_count = len(regularized.faces)
        if face_offset + face_count > total_faces:
            raise ValueError("Per-scan regularized face counts exceed aligned PLY face count")

        scan_objects_dir = objects_dir / scan_dir.name
        source_paths = sorted(scan_objects_dir.rglob("*.ply")) if scan_objects_dir.is_dir() else []
        all_object_paths.extend(source_paths)
        print(f"{scan_dir.name}: {len(source_paths)} aligned object(s) vs {regularized_path.name}")
        for source_path in source_paths:
            source = load_mesh(source_path)
            local_red = projected_outlier_faces(source, regularized)
            red_faces[face_offset : face_offset + face_count] |= local_red
        face_offset += face_count

    if face_offset != total_faces:
        raise ValueError(
            f"Per-scan regularized meshes account for {face_offset:,} faces, "
            f"but aligned PLY has {total_faces:,}"
        )

    output_ply = alignment_dir / "temporary_blob_detection" / "aligned_regularized_meshes_blob_candidates_red.ply"
    write_red_copy(aligned_ply, output_ply, red_faces, all_object_paths)
    print(
        f"Saved colored copy: {output_ply} ({int(red_faces.sum()):,} red faces, "
        f"{len(all_object_paths):,} aligned objects appended)"
    )
    return output_ply


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alignment-dir",
        type=Path,
        required=True,
        help="Phase 07 multi_room_alignment directory",
    )
    args = parser.parse_args()
    run(args.alignment_dir)


if __name__ == "__main__":
    main()
