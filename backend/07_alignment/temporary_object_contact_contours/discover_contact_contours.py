#!/usr/bin/env python3
"""Color aligned object faces that border non-object faces in the sealed mesh."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from plyfile import PlyData


BACKEND = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE = BACKEND / "current_scene"
RED = (1.0, 0.0, 0.0)
OBJECT_CONTACT_DISTANCE_M = 0.001


def sealed_mesh_boundary_faces(vertices: np.ndarray, faces: np.ndarray,
                               class_ids: np.ndarray, instance_ids: np.ndarray) -> np.ndarray:
    """Mark faces edge-connected to a face with a different class/instance owner."""
    adjacency = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).face_adjacency
    boundary = np.zeros(len(faces), dtype=bool)
    if len(adjacency):
        different_owner = (
            (class_ids[adjacency[:, 0]] != class_ids[adjacency[:, 1]])
            | (instance_ids[adjacency[:, 0]] != instance_ids[adjacency[:, 1]])
        )
        boundary[adjacency[different_owner].ravel()] = True
    return boundary


def load_sealed_mesh_ownership(scan_dir: Path) -> tuple[o3d.geometry.TriangleMesh, np.ndarray,
                                                        np.ndarray, np.ndarray]:
    sealed_path = scan_dir / ".raw_unaligned" / "mesh.ply"
    segmented_path = scan_dir / "segmentation" / "mesh_segmented.ply"
    if not sealed_path.is_file() or not segmented_path.is_file():
        raise FileNotFoundError(f"Expected sealed mesh and Stage 05 face ownership in {scan_dir}")

    sealed = o3d.io.read_triangle_mesh(str(sealed_path))
    faces = np.asarray(sealed.triangles)
    if not len(faces):
        raise ValueError(f"Sealed mesh has no triangles: {sealed_path}")

    segmented = PlyData.read(segmented_path)
    face_data = segmented["face"].data
    class_ids = np.asarray(face_data["class_id"], dtype=np.int32)
    instance_ids = np.asarray(face_data["instance_id"], dtype=np.int32)
    if len(class_ids) != len(faces) or len(instance_ids) != len(faces):
        raise ValueError(f"Stage 05 face labels do not match sealed mesh face order: {scan_dir}")

    segmented_faces = np.asarray(face_data["vertex_indices"].tolist(), dtype=np.int64)
    segmented_vertices = np.column_stack([
        segmented["vertex"].data[name] for name in ("x", "y", "z")
    ]).astype(np.float32, copy=False)
    sealed_vertices = np.asarray(sealed.vertices, dtype=np.float32)
    for start in range(0, len(faces), 100_000):
        end = min(start + 100_000, len(faces))
        if not np.array_equal(segmented_vertices[segmented_faces[start:end]],
                              sealed_vertices[faces[start:end]]):
            raise ValueError(f"Stage 05 face order/geometry differs from sealed mesh: {scan_dir}")

    boundary = sealed_mesh_boundary_faces(sealed_vertices, faces, class_ids, instance_ids)
    return sealed, class_ids, instance_ids, boundary


def _triangle_geometry_keys(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = np.asarray(vertices, dtype=np.float32)[faces]
    order = np.lexsort((triangles[:, :, 2], triangles[:, :, 1], triangles[:, :, 0]), axis=1)
    canonical = np.ascontiguousarray(np.take_along_axis(triangles, order[:, :, None], axis=1))
    flattened = canonical.reshape(len(faces), 9)
    return flattened.view(np.dtype((np.void, flattened.dtype.itemsize * 9))).reshape(-1)


def instance_face_indices(source_path: Path, sealed: o3d.geometry.TriangleMesh,
                          class_ids: np.ndarray, instance_ids: np.ndarray,
                          class_id: int, instance_id: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.flatnonzero((class_ids == class_id) & (instance_ids == instance_id))
    source = o3d.io.read_triangle_mesh(str(source_path))
    source_faces = np.asarray(source.triangles)
    sealed_vertices = np.asarray(sealed.vertices, dtype=np.float32)
    sealed_faces = np.asarray(sealed.triangles)
    source_vertices = np.asarray(source.vertices, dtype=np.float32)
    if len(indices) == len(source_faces) and np.array_equal(
        sealed_vertices[sealed_faces[indices]], source_vertices[source_faces]
    ):
        return indices, source_faces.copy()

    candidates = np.flatnonzero(class_ids == class_id)
    candidate_keys = _triangle_geometry_keys(sealed_vertices, sealed_faces[candidates])
    source_keys = _triangle_geometry_keys(source_vertices, source_faces)
    order = np.argsort(candidate_keys)
    sorted_keys = candidate_keys[order]
    positions = np.searchsorted(sorted_keys, source_keys)
    found = positions < len(sorted_keys)
    found[found] &= sorted_keys[positions[found]] == source_keys[found]
    if not np.all(found):
        raise ValueError(f"Could not map all source faces to sealed class {class_id}: {source_path}")
    ambiguous = np.zeros(len(positions), dtype=bool)
    has_previous = positions > 0
    ambiguous[has_previous] |= sorted_keys[positions[has_previous]] == sorted_keys[positions[has_previous] - 1]
    has_next = positions + 1 < len(sorted_keys)
    ambiguous[has_next] |= sorted_keys[positions[has_next]] == sorted_keys[positions[has_next] + 1]
    if np.any(ambiguous):
        raise ValueError(f"Source faces have ambiguous duplicate geometry in sealed class {class_id}: {source_path}")
    return candidates[order[positions]], source_faces.copy()


def write_instance(path: Path, mesh: o3d.geometry.TriangleMesh,
                   red_faces: np.ndarray) -> None:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    if len(colors) != len(vertices) or len(red_faces) != len(faces):
        raise ValueError(f"Expected vertex colors and one contour flag per face: {path}")

    triangle_colors = colors[faces].copy()
    triangle_colors[red_faces] = RED
    output = o3d.geometry.TriangleMesh()
    output.vertices = o3d.utility.Vector3dVector(vertices[faces].reshape(-1, 3))
    output.triangles = o3d.utility.Vector3iVector(
        np.arange(len(faces) * 3, dtype=np.int64).reshape(-1, 3))
    output.vertex_colors = o3d.utility.Vector3dVector(triangle_colors.reshape(-1, 3))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(path), output, write_ascii=False):
        raise OSError(f"Could not write colored instance: {path}")


def suppress_object_contacts(mesh: o3d.geometry.TriangleMesh,
                             other_meshes: list[o3d.geometry.TriangleMesh],
                             red_faces: np.ndarray,
                             max_distance_m: float = OBJECT_CONTACT_DISTANCE_M
                             ) -> tuple[np.ndarray, int]:
    """Restore red faces that touch another aligned instance within 1 mm."""
    candidates = np.flatnonzero(red_faces)
    if not len(candidates) or not other_meshes:
        return red_faces.copy(), 0

    scene = o3d.t.geometry.RaycastingScene()
    added = 0
    for other in other_meshes:
        if len(other.triangles):
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(other))
            added += 1
    if not added:
        return red_faces.copy(), 0

    vertices = np.asarray(mesh.vertices)
    triangles = vertices[np.asarray(mesh.triangles)[candidates]]
    midpoints = (triangles + np.roll(triangles, -1, axis=1)) * 0.5
    centers = triangles.mean(axis=1, keepdims=True)
    samples = np.concatenate((triangles, midpoints, centers), axis=1).astype(np.float32)
    query = o3d.core.Tensor(samples.reshape(-1, 3))
    closest = scene.compute_closest_points(query)["points"].numpy()
    distances = np.linalg.norm(query.numpy() - closest, axis=1).reshape(-1, 7)
    touching = distances.min(axis=1) <= max_distance_m
    filtered = red_faces.copy()
    filtered[candidates[touching]] = False
    return filtered, int(touching.sum())


def write_combined_instances(paths: list[Path], output_path: Path) -> int:
    vertices, faces, colors = [], [], []
    vertex_offset = 0
    for path in paths:
        mesh = o3d.io.read_triangle_mesh(str(path))
        if not len(mesh.triangles) or len(mesh.vertex_colors) != len(mesh.vertices):
            raise ValueError(f"Expected a colored triangle mesh: {path}")
        vertices.append(np.asarray(mesh.vertices))
        faces.append(np.asarray(mesh.triangles) + vertex_offset)
        colors.append(np.asarray(mesh.vertex_colors))
        vertex_offset += len(mesh.vertices)

    if not paths:
        raise ValueError("No aligned object instances were found")
    combined = o3d.geometry.TriangleMesh()
    combined.vertices = o3d.utility.Vector3dVector(np.concatenate(vertices))
    combined.triangles = o3d.utility.Vector3iVector(np.concatenate(faces))
    combined.vertex_colors = o3d.utility.Vector3dVector(np.concatenate(colors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output_path), combined, write_ascii=False):
        raise OSError(f"Could not write combined colored objects: {output_path}")
    return len(combined.triangles)


def process_scan(scan_dir: Path, transforms: dict, output_dir: Path) -> tuple[dict, list[dict]]:
    aligned_dir = scan_dir / "objects_alignment" / "aligned_instances"
    manifest_path = aligned_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run Phase 07 first; missing aligned object manifest: {manifest_path}")

    sealed, class_ids, instance_ids, boundary = load_sealed_mesh_ownership(scan_dir)
    scan_transform = np.asarray(
        transforms["scans"][scan_dir.name]["transform_matrix"], dtype=np.float64)
    manifest = json.loads(manifest_path.read_text())
    records, prepared = [], []

    for item in manifest.get("instances", []):
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid aligned object path in manifest: {item['path']!r}")
        source_path = Path(item["source"])
        if not source_path.is_absolute():
            source_path = scan_dir / source_path
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source instance mesh: {source_path}")
        instance_id = int(relative.stem.rsplit("_", 1)[1])
        face_indices, source_faces = instance_face_indices(
            source_path, sealed, class_ids, instance_ids, int(item["class_id"]), instance_id)

        mesh_path = aligned_dir / relative
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if not len(mesh.triangles) or not mesh.has_vertex_colors():
            raise ValueError(f"Aligned object is empty or has no RGB colors: {mesh_path}")
        if not np.array_equal(np.asarray(mesh.triangles), source_faces):
            raise ValueError(f"Aligned instance face order changed: {mesh_path}")
        mesh.transform(scan_transform)
        red_faces = boundary[face_indices]
        target = output_dir / scan_dir.name / relative.parent / f"{relative.stem}_contact_contour.ply"
        record = {
            "path": str(target.relative_to(output_dir)),
            "source_instance": str(relative),
            "source_face_count": int(len(face_indices)),
            "sealed_boundary_triangle_count": int(red_faces.sum()),
        }
        records.append(record)
        prepared.append({"target": target, "mesh": mesh, "red_faces": red_faces,
                         "record": record})

    return {"scan": scan_dir.name, "instances": records}, prepared


def remove_cross_instance_contacts(instances: list[dict]) -> None:
    bounds = []
    for item in instances:
        vertices = np.asarray(item["mesh"].vertices)
        bounds.append((vertices.min(axis=0), vertices.max(axis=0)))

    for index, item in enumerate(instances):
        lower, upper = bounds[index]
        candidates = np.flatnonzero(item["red_faces"])
        if not len(candidates):
            item["final_red_faces"] = item["red_faces"]
            item["record"]["object_touch_suppressed_triangle_count"] = 0
            item["record"]["red_boundary_triangle_count"] = 0
            continue

        neighbors = [other["mesh"] for other_index, other in enumerate(instances)
                     if other_index != index
                     and np.all(lower - OBJECT_CONTACT_DISTANCE_M <= bounds[other_index][1])
                     and np.all(bounds[other_index][0] <= upper + OBJECT_CONTACT_DISTANCE_M)]
        final_red, removed = suppress_object_contacts(
            item["mesh"], neighbors, item["red_faces"])
        item["final_red_faces"] = final_red
        item["record"]["object_touch_suppressed_triangle_count"] = removed
        item["record"]["red_boundary_triangle_count"] = int(final_red.sum())

    for item in instances:
        write_instance(item["target"], item["mesh"], item["final_red_faces"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--alignment-dir", type=Path,
                        help="Phase 07 output directory; defaults to <workspace>/multi_room_alignment")
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to <alignment-dir>/temporary_object_contact_contours")
    args = parser.parse_args(argv)

    workspace = args.workspace.resolve()
    alignment_dir = (args.alignment_dir or workspace / "multi_room_alignment").resolve()
    transforms_path = alignment_dir / "stats" / "aligned_scans_transforms.json"
    if not transforms_path.is_file():
        parser.error(f"Run Phase 07 first; missing {transforms_path}")
    transforms = json.loads(transforms_path.read_text())
    output_dir = (args.output_dir or alignment_dir / "temporary_object_contact_contours").resolve()

    scans = sorted(path for path in workspace.glob("scan_*") if path.is_dir())
    if not scans:
        parser.error(f"No scan_* directories found under {workspace}")
    records, instances = [], []
    for scan in scans:
        if (scan / "segmentation" / "mesh_segmented.ply").is_file():
            scan_record, scan_instances = process_scan(scan, transforms, output_dir)
            records.append(scan_record)
            instances.extend(scan_instances)
    remove_cross_instance_contacts(instances)
    instances = [item for scan in records for item in scan["instances"]]
    instance_paths = [output_dir / item["path"] for item in instances]
    combined_path = output_dir / "aligned_objects_contact_contours.ply"
    combined_faces = write_combined_instances(instance_paths, combined_path)
    report = {
        "status": "sealed_mesh_edge_contact_contours",
        "contact_rule": "A sealed-mesh face is a contour when it shares an edge with a face whose class_id or instance_id differs.",
        "face_ownership_source": "Stage 05 mesh_segmented.ply face properties, validated against the pristine sealed mesh face geometry and order.",
        "object_contact_filter": f"Restore the original RGB on a contour face if one of its vertices, edge midpoints, or centroid is within {OBJECT_CONTACT_DISTANCE_M:.3f} m of another aligned instance surface.",
        "colors": {"object_boundary": "red", "other_object_faces": "original RGB"},
        "combined_object_mesh": combined_path.name,
        "combined_object_triangle_count": combined_faces,
        "scans": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "contact_contours.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "scans": len(records),
        "instances": len(instances),
        "red_boundary_triangles": sum(item["red_boundary_triangle_count"] for item in instances),
        "suppressed_object_touch_triangles": sum(item["object_touch_suppressed_triangle_count"] for item in instances),
        "combined_mesh": str(combined_path),
        "combined_triangles": combined_faces,
        "output": str(output_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
