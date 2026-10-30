#!/usr/bin/env python3
"""
Extraction of structural classes (walls, floors, ceilings, and stairs if any)
from original colored mesh (original_mesh.ply) into separate instance meshes.
Only major instances are preserved as separate meshes; small components are merged
into the closest major instance if within a proximity threshold, or discarded as floaters.
Uses standard Open3D mesh export with vertex normals and colors for full compatibility
with DCC software including Blender, MeshLab, CloudCompare, etc.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_3DGS_to_mesh"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_mesh_holes import export_ply_with_face_attributes, read_ply_face_flags
from clean_floor_scene_slices import _elongated_fill_faces

ROLE_NORMAL_Y_MIN = 0.5
COPLANAR_BLOB_DISTANCE_M = 0.03
COPLANAR_BLOB_NORMAL_DEG = 25.0


def component_face_indices(triangle_clusters: np.ndarray, num_components: int) -> List[np.ndarray]:
    """Return local face indices for each component, preserving face order."""
    if len(triangle_clusters) == 0:
        return [np.empty(0, dtype=np.int64) for _ in range(num_components)]
    order = np.argsort(triangle_clusters, kind="stable")
    sorted_clusters = triangle_clusters[order]
    starts = np.flatnonzero(np.r_[True, sorted_clusters[1:] != sorted_clusters[:-1]])
    grouped = np.split(order, starts[1:])
    result = [np.empty(0, dtype=np.int64) for _ in range(num_components)]
    for component, faces in zip(sorted_clusters[starts], grouped):
        result[int(component)] = faces
    return result


def _clean_role_triangles(vertices: np.ndarray, triangles: np.ndarray,
                          face_ids: np.ndarray, hole_fill: np.ndarray,
                          role: str) -> tuple[np.ndarray, dict[str, int]]:
    """Remove invalid and elongated fill faces before structural component grouping."""
    selected = triangles[face_ids]
    corners = vertices[selected]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1e-12
    long_fill = _elongated_fill_faces(
        vertices, selected, hole_fill[face_ids], np.zeros(len(face_ids), dtype=np.uint8)
    )
    orientation = np.ones(len(face_ids), dtype=bool)
    normal_y = np.divide(np.abs(cross[:, 1]), double_area,
                         out=np.zeros(len(face_ids), dtype=np.float64), where=valid)
    if role in {"floor", "ceiling"}:
        orientation = normal_y >= ROLE_NORMAL_Y_MIN
    keep = valid & ~long_fill & orientation
    return face_ids[keep], {
        "input_triangles": int(len(face_ids)),
        "degenerate_triangles_removed": int((~valid).sum()),
        "elongated_hole_fill_triangles_removed": int(long_fill.sum()),
        "wrong_orientation_triangles_removed": int((valid & ~long_fill & ~orientation).sum()),
        "retained_triangles": int(keep.sum()),
    }


def closest_major_instance(points: np.ndarray, trees: List[cKDTree]) -> Tuple[int, float]:
    """Choose the closest major instance; ties retain the original first-instance rule."""
    distances = [float(np.min(tree.query(points)[0])) for tree in trees]
    index = int(np.argmin(distances))
    return index, distances[index]


def instance_filename(role: str, instance_index: int) -> str:
    return f"{role}_inst{instance_index:03d}.ply"


def _plane_index(vertices: np.ndarray, faces: np.ndarray):
    corners = vertices[faces]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    valid = lengths > 1e-12
    centers = corners[valid].mean(axis=1)
    normals = cross[valid] / lengths[valid, None]
    return centers, normals, cKDTree(centers)


def _component_plane_support_fraction(
    vertices: np.ndarray, component_faces: np.ndarray, plane_index,
    max_distance: float,
) -> float:
    """Fraction of blob faces supported by a nearby, similarly oriented major face plane."""
    major_centers, major_normals, tree = plane_index
    corners = vertices[component_faces]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    valid = lengths > 1e-12
    blob_centers = corners[valid].mean(axis=1)
    blob_normals = cross[valid] / lengths[valid, None]
    if not len(major_centers) or not len(blob_centers):
        return 0.0
    distances, nearest = tree.query(blob_centers)
    local_normals = major_normals[nearest]
    plane_distance = np.abs(np.einsum(
        "ij,ij->i", blob_centers - major_centers[nearest], local_normals
    ))
    normal_alignment = np.abs(np.einsum("ij,ij->i", blob_normals, local_normals))
    supported = (
        (distances <= max_distance + 0.15)
        & (plane_distance <= COPLANAR_BLOB_DISTANCE_M)
        & (normal_alignment >= np.cos(np.deg2rad(COPLANAR_BLOB_NORMAL_DEG)))
    )
    return float(supported.mean())


def prior_owned_raw_meshes(manifest_path: Path) -> set[str]:
    """Read only safe raw-mesh names that the preceding extraction manifest owned."""
    if not manifest_path.exists():
        return set()
    try:
        with open(manifest_path, "r") as f:
            prior_manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return set()
    instances = prior_manifest.get("instances", []) if isinstance(prior_manifest, dict) else []
    names = set()
    for instance in instances:
        if not isinstance(instance, dict):
            continue
        role = instance.get("role")
        name = instance.get("file_name")
        instance_name = instance.get("instance_name")
        path = Path(name) if isinstance(name, str) else None
        legacy_names = {
            "floor": {"floor_level0.ply", "floor_mezzanine_level1.ply"},
            "ceiling": {"ceiling_main.ply", "ceiling_mezzanine.ply"},
            "wall": {"wall_main.ply"},
        }
        is_indexed = isinstance(role, str) and isinstance(name, str) and (
            name.startswith(f"{role}_inst") and name.removesuffix(".ply").removeprefix(f"{role}_inst").isdigit()
        )
        if (
            role in {"floor", "ceiling", "wall", "stair", "column", "ramp"}
            and isinstance(name, str)
            and instance_name == path.stem
            and path
            and not path.is_absolute()
            and path.name == name
            and path.suffix == ".ply"
            and (is_indexed or name in legacy_names.get(role, set()))
        ):
            names.add(name)
    return names


def export_colored_ply(
    file_path: Path | str,
    vertices: np.ndarray,
    triangles: np.ndarray,
    vertex_colors: np.ndarray,
    face_attributes: dict[str, tuple[str, np.ndarray]] | None = None,
) -> bool:
    """Exports binary little-endian PLY mesh with vertex normals/colors and optional face attributes."""
    if face_attributes:
        return export_ply_with_face_attributes(
            file_path, vertices, triangles, vertex_colors, face_attributes=face_attributes
        )
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))

    if vertex_colors.dtype == np.uint8:
        v_colors_f64 = vertex_colors.astype(np.float64) / 255.0
    else:
        v_colors_f64 = np.clip(vertex_colors.astype(np.float64), 0.0, 1.0)
    mesh.vertex_colors = o3d.utility.Vector3dVector(v_colors_f64)
    mesh.compute_vertex_normals()

    return o3d.io.write_triangle_mesh(
        str(file_path),
        mesh,
        write_ascii=False,
        compressed=False,
        write_vertex_normals=True,
        write_vertex_colors=True,
    )


def clean_extracted_instance(path: Path, instance: dict[str, Any]) -> None:
    """Record cleanup results after component filtering and semantic merges."""
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    flags = read_ply_face_flags(path, "is_hole_fill")
    if not len(vertices) or not len(triangles) or flags is None or len(flags) != len(triangles):
        raise ValueError(f"Extracted instance is empty or missing face flags: {path}")
    used = np.unique(triangles)
    instance.update({
        "num_vertices": int(len(used)),
        "num_triangles": int(len(triangles)),
        "num_original_triangles": int(np.count_nonzero(flags == 0)),
        "num_filled_triangles": int(np.count_nonzero(flags == 1)),
        "surface_area_m2": round(float(mesh.get_surface_area()), 3),
        "elevation": {
            "y_mean": round(float(vertices[used, 1].mean()), 3),
            "y_min": round(float(vertices[used, 1].min()), 3),
            "y_max": round(float(vertices[used, 1].max()), 3),
        },
        "geometric_cleanup": {
            "statistical_outlier_filter": "disabled",
            "non_coplanar_blob_components_removed": instance.get("non_coplanar_blob_components_removed", 0),
            "non_coplanar_blob_triangles_removed": instance.get("non_coplanar_blob_triangles_removed", 0),
        },
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structural classes (walls, floors, ceilings) as separate big instances."
    )
    parser.add_argument(
        "--mesh-path",
        type=Path,
        default=Path("current_scene/scan_0/mesh.ply"),
        help="Path to authoritative sealed mesh PLY",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("current_scene/scan_0/segmentation/triangle_class_labels.npy"),
        help="Path to triangle class labels array (.npy)",
    )
    parser.add_argument(
        "--structural-json",
        type=Path,
        default=Path("current_scene/scan_0/segmentation/class_metadata.json"),
        help="Path to structural classes JSON metadata",
    )
    parser.add_argument(
        "--associations-json",
        type=Path,
        help="Valid image-to-text associations (defaults beside the scene mesh)",
    )
    parser.add_argument(
        "--segmentation-output",
        type=Path,
        help="Stage 05 output with class metadata and door/window instance meshes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("current_scene/scan_0/mesh_elaboration"),
        help="Output directory for extracted meshes and manifest",
    )
    parser.add_argument(
        "--min-big-area",
        type=float,
        default=1.0,
        help="Minimum surface area (m^2) to qualify a component as a separate major instance",
    )
    parser.add_argument(
        "--merge-dist-thresh",
        type=float,
        default=0.25,
        help="Max distance (meters) to merge a small component into the closest big instance",
    )
    return parser.parse_args()


def load_door_window_associations(path: Path) -> dict[int, int]:
    """Return recognized source image classes mapped to door (1) or window (2)."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    pairs = data.get("associations") if isinstance(data, dict) else None
    if not isinstance(pairs, list):
        raise ValueError(f"Invalid valid-class association file: {path}")
    result: dict[int, int] = {}
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"Invalid association tuple in {path}: {pair!r}")
        source_class_id, text_class_id = map(int, pair)
        if text_class_id not in {1, 2}:
            continue
        previous = result.setdefault(source_class_id, text_class_id)
        if previous != text_class_id:
            raise ValueError(f"Source class {source_class_id} maps to both door and window in {path}")
    return result


def load_associated_object_instances(
    associations: dict[int, int], segmentation_dir: Path
) -> list[dict[str, Any]]:
    """Load the segmented meshes selected by the valid door/window associations."""
    if not associations:
        return []
    metadata_path = segmentation_dir / "class_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Segmentation class metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text()).get("classes")
    if not isinstance(metadata, dict):
        raise ValueError(f"Segmentation metadata has no classes object: {metadata_path}")

    result = []
    for source_class_id, text_class_id in sorted(associations.items()):
        class_info = metadata.get(str(source_class_id))
        if class_info is None or class_info.get("structural_role") != "object":
            raise ValueError(f"Associated source class {source_class_id} is not a segmented object")
        class_dir = segmentation_dir / "3d_masks" / f"class_{source_class_id:03d}_object"
        paths = sorted(class_dir.glob("instance_*.ply")) if class_dir.is_dir() else []
        if not paths:
            alt_dir = segmentation_dir / "Instances PLYs" / "Non-Structural" / str(source_class_id)
            if alt_dir.is_dir():
                paths = sorted(alt_dir.glob("instance_*.ply"))
        if not paths:
            raise FileNotFoundError(f"No instances for associated source class {source_class_id}: checked {class_dir} and Instances PLYs")
        for path in paths:
            match = re.fullmatch(r"instance_(\d+)\.ply", path.name)
            if match is None:
                raise ValueError(f"Unexpected segmented instance path: {path}")
            mesh = o3d.io.read_triangle_mesh(str(path))
            vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
            if not len(vertices) or not len(triangles):
                raise ValueError(f"Associated segmented instance is empty: {path}")
            colors = np.asarray(mesh.vertex_colors)
            if len(colors) != len(vertices):
                colors = np.full((len(vertices), 3), 0.5, dtype=np.float64)
            flags = read_ply_face_flags(path, "is_hole_fill")
            if flags is None or len(flags) != len(triangles):
                flags = np.zeros(len(triangles), dtype=np.uint8)
            result.append({
                "source_class_id": source_class_id,
                "text_class_id": text_class_id,
                "instance_id": int(match.group(1)),
                "source_path": str(path),
                "vertices": vertices,
                "triangles": triangles,
                "colors": colors,
                "hole_fill_flags": flags,
            })
    return result


def choose_structural_target(
    vertices: np.ndarray, targets: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, float]:
    """Choose the nearest wall for either recognized doors or windows."""
    scores = []
    for target in targets:
        if target["role"] != "wall":
            continue
        distances = target["tree"].query(vertices, workers=-1)[0]
        scores.append((float(np.mean(distances)), target))
    if not scores:
        return None, float("inf")
    distance, target = min(scores, key=lambda item: item[0])
    return target, distance


def merge_associated_instances(
    staging_dir: Path, manifest: dict[str, Any], associated: list[dict[str, Any]]
) -> None:
    """Append each recognized door/window mesh to its nearest compatible structural instance."""
    targets = []
    for item in manifest["instances"]:
        if item["role"] not in {"wall", "ceiling"}:
            continue
        path = staging_dir / item["file_name"]
        mesh = o3d.io.read_triangle_mesh(str(path))
        vertices = np.asarray(mesh.vertices)
        if len(vertices):
            targets.append({"item": item, "path": path, "vertices": vertices, "tree": cKDTree(vertices), "role": item["role"]})

    merge_records = []
    for source in associated:
        target, mean_distance = choose_structural_target(source["vertices"], targets)
        if target is None:
            print(
                f"  Warning: No compatible structural target for text class {source['text_class_id']} "
                f"instance {source['source_class_id']}:{source['instance_id']}; skipping merge."
            )
            continue
        target_mesh = o3d.io.read_triangle_mesh(str(target["path"]))
        target_vertices = np.asarray(target_mesh.vertices)
        target_triangles = np.asarray(target_mesh.triangles)
        target_colors = np.asarray(target_mesh.vertex_colors)
        if len(target_colors) != len(target_vertices):
            target_colors = np.full((len(target_vertices), 3), 0.5, dtype=np.float64)
        target_flags = read_ply_face_flags(target["path"], "is_hole_fill")
        if target_flags is None or len(target_flags) != len(target_triangles):
            target_flags = np.zeros(len(target_triangles), dtype=np.uint8)

        vertices = np.vstack((target_vertices, source["vertices"]))
        source_face_ids, source_cleanup = _clean_role_triangles(
            source["vertices"], source["triangles"],
            np.arange(len(source["triangles"]), dtype=np.int64),
            source["hole_fill_flags"], target["role"],
        )
        triangles = np.vstack((target_triangles,
                               source["triangles"][source_face_ids] + len(target_vertices)))
        colors = np.vstack((target_colors, source["colors"]))
        flags = np.concatenate((target_flags, source["hole_fill_flags"][source_face_ids]))
        if not export_colored_ply(
            target["path"], vertices, triangles, colors,
            face_attributes={"is_hole_fill": ("uchar", flags)},
        ):
            raise OSError(f"Failed to merge associated instance into {target['path']}")

        merged_mesh = o3d.geometry.TriangleMesh()
        merged_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        merged_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        entry = target["item"]
        entry.update({
            "num_vertices": int(len(vertices)),
            "num_triangles": int(len(triangles)),
            "num_original_triangles": int(np.count_nonzero(flags == 0)),
            "num_filled_triangles": int(np.count_nonzero(flags == 1)),
            "surface_area_m2": round(float(merged_mesh.get_surface_area()), 3),
            "elevation": {
                "y_mean": round(float(vertices[:, 1].mean()), 3),
                "y_min": round(float(vertices[:, 1].min()), 3),
                "y_max": round(float(vertices[:, 1].max()), 3),
            },
        })
        record = {
            "source_class_id": source["source_class_id"],
            "text_class_id": source["text_class_id"],
            "instance_id": source["instance_id"],
            "source_file": source["source_path"],
            "target_role": target["role"],
            "target_instance": entry["instance_name"],
            "mean_nearest_vertex_distance_m": round(mean_distance, 6),
            "triangles_added": int(len(source_face_ids)),
            "source_triangle_cleanup": source_cleanup,
        }
        entry.setdefault("merged_semantic_instances", []).append(record)
        merge_records.append(record)
        print(
            f"  Merged text class {source['text_class_id']} instance "
            f"{source['source_class_id']}:{source['instance_id']} into "
            f"{entry['instance_name']} (mean nearest vertex {mean_distance:.3f}m)"
        )
    manifest["door_window_merges"] = merge_records
    for role, summary in manifest["roles"].items():
        summary["merged_semantic_instances"] = sum(r["target_role"] == role for r in merge_records)


def main() -> int:
    args = parse_args()

    if not args.mesh_path.exists():
        print(f"Error: Mesh not found at {args.mesh_path}")
        return 1
    if not args.labels_path.exists():
        print(f"Error: Labels not found at {args.labels_path}")
        return 1

    print(f"Loading mesh: {args.mesh_path}...")
    mesh = o3d.io.read_triangle_mesh(str(args.mesh_path))
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertex_colors = np.asarray(mesh.vertex_colors)
    if len(vertex_colors) == 0:
        vertex_colors = np.full((len(vertices), 3), 0.5, dtype=np.float64)
    print(f"Loaded mesh: {len(vertices):,} vertices, {len(triangles):,} triangles")

    face_hole_fill_flags = read_ply_face_flags(args.mesh_path, "is_hole_fill")
    if face_hole_fill_flags is None or len(face_hole_fill_flags) != len(triangles):
        face_hole_fill_flags = np.zeros(len(triangles), dtype=np.uint8)
    else:
        num_filled = int(np.count_nonzero(face_hole_fill_flags))
        print(f"Loaded face flags: {num_filled:,} hole-fill triangles identified out of {len(triangles):,}")

    print(f"Loading labels: {args.labels_path}...")
    labels = np.load(args.labels_path)
    if len(labels) != len(triangles):
        print(f"Error: Label count {len(labels)} does not match triangle count {len(triangles)}")
        return 1

    structural_classes_by_role: Dict[str, List[int]] = {
        "floor": [],
        "ceiling": [],
        "wall": [],
        "stair": [],
        "column": [],
        "ramp": [],
    }

    if not args.structural_json.exists():
        print(f"Error: Structural classification metadata not found at {args.structural_json}")
        return 1
    with open(args.structural_json, "r") as f:
        struct_data = json.load(f)
    classifications = struct_data.get("classifications")
    if not isinstance(classifications, dict):
        classifications = struct_data.get("classes")
    if not isinstance(classifications, dict) and isinstance(struct_data.get("structural_summary"), dict):
        classifications = {}
        for role_key, cids in struct_data["structural_summary"].items():
            for c in cids:
                classifications[str(c)] = {"structural_role": role_key}
    if not isinstance(classifications, dict):
        print(f"Error: Structural classification metadata at {args.structural_json} has no classifications or classes object")
        return 1
    for cid_str, info in classifications.items():
        try:
            cid = int(cid_str)
        except ValueError:
            continue
        role = str(info.get("structural_role", "")).lower()
        if "floor" in role:
            structural_classes_by_role["floor"].append(cid)
        elif "ceiling" in role:
            structural_classes_by_role["ceiling"].append(cid)
        elif "wall" in role:
            structural_classes_by_role["wall"].append(cid)
        elif "stair" in role:
            structural_classes_by_role["stair"].append(cid)
        elif "column" in role:
            structural_classes_by_role["column"].append(cid)
        elif "ramp" in role:
            structural_classes_by_role["ramp"].append(cid)

    associations_path = getattr(args, "associations_json", None)
    segmentation_dir = getattr(args, "segmentation_output", None)
    if segmentation_dir is None:
        cand_seg = args.mesh_path.parent / "segmentation"
        if cand_seg.is_dir():
            segmentation_dir = cand_seg
        else:
            segmentation_dir = args.mesh_path.resolve().parents[1] / "05_semantic_segmentation" / "mesh_objects_step3" / "segmentation_output"
    if associations_path is None:
        associations_path = segmentation_dir / "non_structural_similarity" / "valid_class_associations.json"
        if not associations_path.is_file():
            associations_path = args.mesh_path.parent / "non_structural_similarity" / "valid_class_associations.json"
    try:
        door_window_associations = load_door_window_associations(associations_path)
        associated_instances = load_associated_object_instances(door_window_associations, segmentation_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: Could not load associated door/window instances: {exc}")
        return 1
    if associated_instances:
        print(f"\nLoaded {len(associated_instances)} associated door/window instances from {associations_path}")

    print("\nIdentified structural classes:")
    for role, cids in structural_classes_by_role.items():
        print(f"  {role.upper()}: classes {cids}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "structural_instances_manifest.json"
    prior_raw_meshes = prior_owned_raw_meshes(manifest_path)
    staging = tempfile.TemporaryDirectory(dir=args.output_dir, prefix=".extract_structural_")
    staging_dir = Path(staging.name)

    manifest: Dict[str, Any] = {
        "source_mesh": str(args.mesh_path),
        "total_source_vertices": len(vertices),
        "total_source_triangles": len(triangles),
        "cleanup_policy": {
            "statistical_outlier_filter": "disabled",
            "non_coplanar_blob_max_supported_face_fraction": 0.10,
            "coplanar_plane_distance_m": COPLANAR_BLOB_DISTANCE_M,
            "coplanar_normal_tolerance_deg": COPLANAR_BLOB_NORMAL_DEG,
            "blob_merge_distance_extra_m": 0.15,
        },
        "roles": {},
        "instances": [],
    }

    for role, cids in structural_classes_by_role.items():
        if not cids:
            print(f"\nNo classes found for role: {role}")
            continue

        print(f"\n==================================================")
        print(f"Processing role: {role.upper()} (Class IDs: {cids})")
        print(f"==================================================")

        tri_mask = np.isin(labels, cids)
        role_tri_indices = np.where(tri_mask)[0]
        if len(role_tri_indices) == 0:
            print(f"No triangles found for role {role}")
            continue

        role_tri_indices, role_cleanup = _clean_role_triangles(
            vertices, triangles, role_tri_indices, face_hole_fill_flags, role
        )
        print(
            f"Cleaned {role}: removed {role_cleanup['degenerate_triangles_removed']:,} degenerate, "
            f"{role_cleanup['elongated_hole_fill_triangles_removed']:,} elongated hole-fill, "
            f"and {role_cleanup['wrong_orientation_triangles_removed']:,} wrong-orientation triangles; "
            f"retained {role_cleanup['retained_triangles']:,}"
        )
        if not len(role_tri_indices):
            print(f"No clean triangles remain for role {role}")
            continue

        role_triangles = triangles[role_tri_indices]
        unique_v_idx, remapped_triangles = np.unique(role_triangles, return_inverse=True)
        remapped_triangles = remapped_triangles.reshape(role_triangles.shape)
        role_vertices = vertices[unique_v_idx]

        sub_mesh = o3d.geometry.TriangleMesh()
        sub_mesh.vertices = o3d.utility.Vector3dVector(role_vertices)
        sub_mesh.triangles = o3d.utility.Vector3iVector(remapped_triangles)

        triangle_clusters, comp_tri_counts, comp_areas = sub_mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        comp_tri_counts = np.asarray(comp_tri_counts)
        comp_areas = np.asarray(comp_areas)
        component_faces = component_face_indices(triangle_clusters, len(comp_tri_counts))

        num_components = len(comp_tri_counts)
        print(f"Found {num_components} connected components across {len(role_tri_indices):,} triangles")

        min_area = 0.20 if role in {"column", "ramp"} else args.min_big_area
        big_indices = np.where(comp_areas >= min_area)[0]
        if len(big_indices) == 0:
            big_indices = np.array([np.argmax(comp_areas)])

        big_indices = big_indices[np.argsort(-comp_areas[big_indices])]
        small_indices = np.setdiff1d(np.arange(num_components), big_indices)

        print(f"  Major instances (area >= {min_area} m²): {len(big_indices)}")
        for i, b_idx in enumerate(big_indices):
            print(f"    Major #{i}: {comp_tri_counts[b_idx]:,} tris, {comp_areas[b_idx]:.2f} m²")
        print(f"  Small components: {len(small_indices)} (total {np.sum(comp_tri_counts[small_indices]):,} tris, {np.sum(comp_areas[small_indices]):.2f} m²)")

        big_instance_trees = []
        big_plane_surfaces = []
        for b_idx in big_indices:
            b_faces = remapped_triangles[component_faces[b_idx]]
            b_v_indices = np.unique(b_faces)
            b_pts = role_vertices[b_v_indices]
            big_instance_trees.append(cKDTree(b_pts))
            big_plane_surfaces.append(_plane_index(role_vertices, b_faces))

        assigned_triangles_per_big: List[List[int]] = [
            role_tri_indices[component_faces[b_idx]].tolist() for b_idx in big_indices
        ]
        merged_small_count_per_big = [0] * len(big_indices)
        merged_small_tris_per_big = [0] * len(big_indices)
        discarded_non_coplanar_count_per_big = [0] * len(big_indices)
        discarded_non_coplanar_tris_per_big = [0] * len(big_indices)
        discarded_small_count = 0
        discarded_small_tris = 0
        discarded_non_coplanar_count = 0
        discarded_non_coplanar_tris = 0

        thresh = args.merge_dist_thresh
        if role == "ceiling":
            thresh = max(thresh, 0.30)

        for s_idx in small_indices:
            s_faces = remapped_triangles[component_faces[s_idx]]
            s_v_indices = np.unique(s_faces)
            s_pts = role_vertices[s_v_indices]

            best_big_idx, best_dist = closest_major_instance(s_pts, big_instance_trees)

            s_orig_tris = role_tri_indices[component_faces[s_idx]].tolist()
            if best_dist <= thresh:
                support_fraction = _component_plane_support_fraction(
                    role_vertices, s_faces, big_plane_surfaces[best_big_idx], thresh
                )
                if support_fraction < 0.10:
                    discarded_non_coplanar_count += 1
                    discarded_non_coplanar_tris += len(s_orig_tris)
                    discarded_non_coplanar_count_per_big[best_big_idx] += 1
                    discarded_non_coplanar_tris_per_big[best_big_idx] += len(s_orig_tris)
                    continue
                assigned_triangles_per_big[best_big_idx].extend(s_orig_tris)
                merged_small_count_per_big[best_big_idx] += 1
                merged_small_tris_per_big[best_big_idx] += len(s_orig_tris)
            else:
                discarded_small_count += 1
                discarded_small_tris += len(s_orig_tris)

        print(f"  Small components resolution:")
        for i in range(len(big_indices)):
            print(f"    Major #{i}: merged {merged_small_count_per_big[i]} small components ({merged_small_tris_per_big[i]:,} tris)")
        print(f"    Discarded floaters (> {thresh:.2f}m): {discarded_small_count} components ({discarded_small_tris:,} tris)")
        print(
            f"    Discarded non-coplanar blobs: {discarded_non_coplanar_count} components "
            f"({discarded_non_coplanar_tris:,} tris)"
        )

        filenames = []
        for i in range(len(big_indices)):
            filenames.append(instance_filename(role, i))

        role_instance_entries = []
        for i, b_idx in enumerate(big_indices):
            inst_tri_indices = np.array(assigned_triangles_per_big[i], dtype=np.int64)
            inst_triangles = triangles[inst_tri_indices]
            inst_flags = face_hole_fill_flags[inst_tri_indices]

            inst_unique_v_idx, inst_remapped_tri = np.unique(inst_triangles, return_inverse=True)
            inst_remapped_tri = inst_remapped_tri.reshape(inst_triangles.shape)
            inst_vertices = vertices[inst_unique_v_idx]
            inst_colors = vertex_colors[inst_unique_v_idx]

            mean_y = float(inst_vertices[:, 1].mean())
            y_min = float(inst_vertices[:, 1].min())
            y_max = float(inst_vertices[:, 1].max())

            filename = filenames[i]
            label_suffix = Path(filename).stem[len(role) + 1:]
            out_file = staging_dir / filename
            final_out_file = args.output_dir / filename

            print(f"  Exporting {out_file.name}: {len(inst_vertices):,} verts, {len(inst_triangles):,} tris (mean Y: {mean_y:.2f}m)...")
            if not export_colored_ply(
                out_file, inst_vertices, inst_remapped_tri, inst_colors,
                face_attributes={"is_hole_fill": ("uchar", inst_flags)}
            ) or not out_file.is_file():
                print(f"Error: Failed to export {out_file}")
                staging.cleanup()
                return 1

            inst_m = o3d.geometry.TriangleMesh()
            inst_m.vertices = o3d.utility.Vector3dVector(inst_vertices)
            inst_m.triangles = o3d.utility.Vector3iVector(inst_remapped_tri)
            inst_area = float(inst_m.get_surface_area())

            inst_meta = {
                "instance_name": f"{role}_{label_suffix}",
                "role": role,
                "file_name": filename,
                "file_path": str(final_out_file),
                "num_vertices": int(len(inst_vertices)),
                "num_triangles": int(len(inst_triangles)),
                "num_original_triangles": int(np.count_nonzero(inst_flags == 0)),
                "num_filled_triangles": int(np.count_nonzero(inst_flags == 1)),
                "surface_area_m2": round(inst_area, 3),
                "elevation": {
                    "y_mean": round(mean_y, 3),
                    "y_min": round(y_min, 3),
                    "y_max": round(y_max, 3),
                },
                "merged_small_components": int(merged_small_count_per_big[i]),
                "merged_small_triangles": int(merged_small_tris_per_big[i]),
                "non_coplanar_blob_components_removed": int(discarded_non_coplanar_count_per_big[i]),
                "non_coplanar_blob_triangles_removed": int(discarded_non_coplanar_tris_per_big[i]),
            }
            manifest["instances"].append(inst_meta)
            role_instance_entries.append(inst_meta)

        manifest["roles"][role] = {
            "input_cleanup": role_cleanup,
            "num_major_instances": len(big_indices),
            "merged_small_components": sum(merged_small_count_per_big),
            "merged_small_triangles": sum(merged_small_tris_per_big),
            "discarded_small_components": discarded_small_count,
            "discarded_small_triangles": discarded_small_tris,
            "discarded_non_coplanar_blob_components": discarded_non_coplanar_count,
            "discarded_non_coplanar_blob_triangles": discarded_non_coplanar_tris,
            "instances": [e["instance_name"] for e in role_instance_entries],
        }

    try:
        if associated_instances:
            merge_associated_instances(staging_dir, manifest, associated_instances)
        for instance in manifest["instances"]:
            clean_extracted_instance(staging_dir / instance["file_name"], instance)
            print(f"Cleaned {instance['file_name']}: removed "
                  f"{instance['geometric_cleanup']['non_coplanar_blob_triangles_removed']:,} "
                  "non-coplanar blob triangles")
    except (OSError, ValueError) as exc:
        print(f"Error: Could not clean or merge structural instances: {exc}")
        staging.cleanup()
        return 1

    staged_manifest = staging_dir / manifest_path.name
    try:
        with open(staged_manifest, "w") as f:
            json.dump(manifest, f, indent=2)
        for instance in manifest["instances"]:
            (staging_dir / instance["file_name"]).replace(args.output_dir / instance["file_name"])
        staged_manifest.replace(manifest_path)
        new_raw_meshes = {instance["file_name"] for instance in manifest["instances"]}
        for filename in prior_raw_meshes - new_raw_meshes:
            stale_output = args.output_dir / filename
            if stale_output.is_file():
                stale_output.unlink()
    except OSError as exc:
        print(f"Error: Failed to publish extracted meshes: {exc}")
        staging.cleanup()
        return 1
    staging.cleanup()
    print(f"\nManifest saved to: {manifest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
