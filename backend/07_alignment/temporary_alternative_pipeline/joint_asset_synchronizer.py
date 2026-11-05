"""Temporary Phase 07 synchronizer that transforms instances with sealed meshes."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from multi_asset_synchronizer import MultiAssetSynchronizer

logger = logging.getLogger("joint_asset_synchronizer")


def _instance_sources(scan) -> list[tuple[Path, dict[str, Any]]]:
    scan_dir = scan.info.scan_dir
    masks_dir = scan_dir / "segmentation" / "3d_masks"
    aligned_dir = scan_dir / "objects_alignment" / "aligned_instances"
    manifest_path = aligned_dir / "manifest.json"
    sources = []

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        for item in manifest.get("instances", []):
            mesh_path = Path(item.get("source", ""))
            if not mesh_path.is_absolute():
                mesh_path = scan_dir / mesh_path
            mesh_path = mesh_path.absolute()
            if ".." in mesh_path.parts or masks_dir.absolute() not in mesh_path.parents:
                raise ValueError(f"Invalid source-instance path: {mesh_path}")
            if not mesh_path.is_file():
                raise FileNotFoundError(f"Source instance listed in manifest is missing: {mesh_path}")
            # Aligned-instance PLYs contain per-group support snaps; preserve the source geometry here.
            sources.append((mesh_path, {**item, "placement_translation_m": [0.0, 0.0, 0.0]}))

    if not sources:
        masks_dir = scan_dir / "segmentation" / "3d_masks"
        if masks_dir.is_dir():
            for class_dir in sorted(masks_dir.iterdir()):
                if not class_dir.is_dir() or any(
                    name in class_dir.name.lower() for name in ("door", "wall", "floor", "ceiling")
                ):
                    continue
                class_parts = class_dir.name.split("_")
                class_id = int(class_parts[1]) if len(class_parts) > 1 and class_parts[1].isdigit() else 0
                for mesh_path in sorted(class_dir.glob("*.ply")):
                    sources.append((mesh_path, {
                        "class_id": class_id,
                        "category": class_dir.name,
                        "role": "object",
                        "path": f"{class_dir.name}/{mesh_path.name}",
                        "placement_translation_m": [0.0, 0.0, 0.0],
                    }))

    return sources


def _validate_transform(scan_name: str, transform: np.ndarray) -> None:
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"Invalid 4x4 alignment transform for {scan_name}")
    rotation = transform[:3, :3]
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"Invalid homogeneous transform row for {scan_name}")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) \
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"Alignment transform is not a rigid rotation for {scan_name}")


class JointAssetSynchronizer(MultiAssetSynchronizer):
    """Reuse Phase 07 exports while transforming each scan's mesh and instances together."""

    def __init__(self, output_dir: Path):
        super().__init__(output_dir)
        self._joint_object_manifest: dict[str, Any] | None = None

    def _assemble_sealed_meshes(self, scans, transforms, output_path: Path) -> None:
        sealed_meshes = []
        object_meshes = []
        records = []
        object_dir = self.output_dir / "objects"

        for scan_name, scan in sorted(scans.items()):
            transform = np.asarray(transforms[scan_name], dtype=np.float64)
            _validate_transform(scan_name, transform)

            sealed_path = scan.info.sealed_mesh_path
            if sealed_path and sealed_path.is_file():
                sealed = trimesh.load(sealed_path, process=False)
                if not isinstance(sealed, trimesh.Trimesh) or not len(sealed.faces):
                    raise ValueError(f"Expected a non-empty sealed mesh: {sealed_path}")
                sealed.apply_transform(transform)
                sealed_meshes.append(sealed)

            scan_object_dir = object_dir / scan_name
            for source_path, item in _instance_sources(scan):
                instance = trimesh.load(source_path, process=False)
                if not isinstance(instance, trimesh.Trimesh) or not len(instance.faces):
                    raise ValueError(f"Expected a non-empty object instance: {source_path}")
                instance.apply_transform(transform)

                relative_path = Path(item["path"])
                target_path = scan_object_dir / relative_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                instance.export(str(target_path))
                vertices = instance.vertices
                records.append({
                    "scan_id": scan_name,
                    "class_id": item.get("class_id"),
                    "category": item.get("category"),
                    "role": item.get("role", "object"),
                    "source_instance_path": str(source_path),
                    "global_instance_path": str(target_path),
                    "triangles": len(instance.faces),
                    "vertices": len(vertices),
                    "scan_transform_matrix": transform.tolist(),
                    "local_placement_translation_m": item.get("placement_translation_m", [0.0, 0.0, 0.0]),
                    "global_bounding_box": vertices.min(axis=0).tolist() + vertices.max(axis=0).tolist(),
                    "global_centroid": vertices.mean(axis=0).tolist(),
                })
                object_meshes.append(instance)

        if not sealed_meshes:
            raise ValueError("No sealed room meshes were found")
        if not object_meshes:
            raise ValueError("No aligned object instances were found")

        sealed_combined = trimesh.util.concatenate(sealed_meshes)
        sealed_combined.export(str(output_path))
        objects_path = self.output_dir / "aligned_objects.ply"
        trimesh.util.concatenate(object_meshes).export(str(objects_path))
        logger.info(
            "Jointly transformed %d object instances with %d sealed scan meshes",
            len(records), len(sealed_meshes),
        )
        self._joint_object_manifest = {
            "schema_version": "1.0.0",
            "total_objects": len(records),
            "objects": records,
        }

    def _assemble_objects(self, scans, transforms, output_objects_dir: Path,
                          combined_output_ply: Path, manifest_path: Path) -> dict[str, Any]:
        if self._joint_object_manifest is None:
            raise RuntimeError("Instances were not assembled alongside the sealed meshes")
        if combined_output_ply != self.output_dir / "aligned_objects.ply":
            raise ValueError("Unexpected combined object output path")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(self._joint_object_manifest, indent=2) + "\n")
        return self._joint_object_manifest
