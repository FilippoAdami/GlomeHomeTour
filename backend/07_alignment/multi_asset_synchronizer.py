"""Generic Multi-Asset Delivery and Synchronization Engine.

Applies the solved global SE(3) coordinate transformations across all room scans to:
1. Phase 06 Regularized Meshes (aligned_regularized_meshes.ply)
2. Phase 04 Sealed Room Meshes (aligned_sealed_meshes.ply)
3. Phase 04 3D Gaussian Splats (aligned_sealed_gaussians.ply, with full quaternion composition)
4. Stage 05 object instances (aligned_objects.ply and per-instance PLYs)
5. Authoritative Manifests (aligned_scans_transforms.json, multi_room_alignment_report.json)
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import plyfile
import trimesh

try:
    from .scan_loader import ScanData
    from .single_scan_synchronizer import (
        build_mesh_occupancy_scenes,
        transform_gaussian_splat_vertices,
        write_gaussian_splat_vertices,
    )
except (ImportError, ValueError):
    from scan_loader import ScanData
    from single_scan_synchronizer import (
        build_mesh_occupancy_scenes,
        transform_gaussian_splat_vertices,
        write_gaussian_splat_vertices,
    )

logger = logging.getLogger("asset_synchronizer")


class MultiAssetSynchronizer:
    """Synchronizes and exports all multi-room property models and manifests."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gaussian_overlap_scenes: Dict[str, Any] = {}
        self.transformed_gaussian_splat_paths: Dict[str, Dict[Path, Path]] = {}
        self.timings_seconds: Dict[str, float] = {}

    def synchronize_and_export_all(
        self,
        scans: Dict[str, ScanData],
        transforms: Dict[str, np.ndarray],
        report_data: Dict[str, Any],
    ) -> Dict[str, Path]:
        """Assembles and exports all 3D deliverables and manifest files."""
        self.timings_seconds = {}

        def measure(name: str, operation):
            started = time.perf_counter()
            try:
                return operation()
            finally:
                elapsed = time.perf_counter() - started
                self.timings_seconds[name] = elapsed
                logger.info("Timing %-48s %.2f s", f"step2.{name}", elapsed)

        (self.output_dir / "aligned_sealed_meshes_with_doors_red.ply").unlink(missing_ok=True)
        (self.output_dir / "aligned_scene_with_objects.ply").unlink(missing_ok=True)
        scan_keys = sorted(list(scans.keys()))
        anchor_scan = scan_keys[0]

        exported_paths: Dict[str, Path] = {}

        # 1. Aligned Regularized Meshes (Phase 06)
        logger.info("[Export 1/4] Assembling aligned regularized meshes (Phase 06)...")
        p_reg = self.output_dir / "aligned_regularized_meshes.ply"
        measure("assemble_regularized_meshes", lambda: self._assemble_regularized_meshes(scans, transforms, p_reg))
        exported_paths["aligned_regularized_meshes"] = p_reg

        # 2. Aligned Sealed Meshes (Phase 04)
        logger.info("[Export 2/4] Assembling aligned sealed meshes (Phase 04)...")
        p_sealed = self.output_dir / "aligned_sealed_meshes.ply"
        aligned_sealed_meshes = measure(
            "assemble_sealed_meshes", lambda: self._assemble_sealed_meshes(scans, transforms, p_sealed)
        )
        self.gaussian_overlap_scenes = measure(
            "build_mesh_occupancy_scenes", lambda: build_mesh_occupancy_scenes(aligned_sealed_meshes)
        )
        exported_paths["aligned_sealed_meshes"] = p_sealed

        # 3. Aligned 3D Gaussian Splats (Phase 04)
        logger.info("[Export 3/4] Assembling aligned 3D Gaussian Splats with quaternion rotations...")
        p_gs = self.output_dir / "aligned_sealed_gaussians.ply"
        self.transformed_gaussian_splat_paths = measure(
            "assemble_gaussian_splats",
            lambda: self._assemble_gaussian_splats(scans, transforms, p_gs, self.gaussian_overlap_scenes),
        )
        exported_paths["aligned_sealed_gaussians"] = p_gs

        # Stats & Manifests directory
        stats_dir = self.output_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        exported_paths["stats_dir"] = stats_dir

        # 4. Aligned 3D Objects Across All Rooms
        logger.info("[Export 4/5] Transforming and assembling Stage 05 object instances...")
        p_objects_dir = self.output_dir / "objects"
        p_combined_objects = self.output_dir / "aligned_objects.ply"
        p_obj_manifest = stats_dir / "aligned_objects_manifest.json"
        measure(
            "assemble_object_instances",
            lambda: self._assemble_objects(scans, transforms, p_objects_dir, p_combined_objects, p_obj_manifest),
        )
        exported_paths["aligned_objects"] = p_combined_objects
        exported_paths["aligned_objects_manifest"] = p_obj_manifest
        exported_paths["objects_dir"] = p_objects_dir

        # 6. Transform Manifest in stats/
        p_tf = stats_dir / "aligned_scans_transforms.json"
        tf_manifest = {
            "schema_version": "1.1.0",
            "anchor_scan": anchor_scan,
            "scans": {
                sname: {
                    "transform_matrix": transforms[sname].tolist(),
                    "mesh_path": str(scans[sname].info.sealed_mesh_path or ""),
                }
                for sname in scan_keys
            },
        }
        def write_manifests():
            with open(p_tf, "w") as f:
                json.dump(tf_manifest, f, indent=2)

            p_rep = stats_dir / "multi_room_alignment_report.json"
            full_report = {
                "anchor_scan": anchor_scan,
                "scan_count": len(scans),
                "scans": scan_keys,
                "alignment_metrics": report_data,
                "transforms": {sname: transforms[sname].tolist() for sname in scan_keys},
            }
            with open(p_rep, "w") as f:
                json.dump(full_report, f, indent=2)
            return p_rep

        p_rep = measure("write_alignment_manifests", write_manifests)
        logger.info(f"Saved transform manifest to: {p_tf}")
        exported_paths["aligned_scans_transforms"] = p_tf

        # 7. Structured Multi-Room Alignment Report in stats/
        logger.info(f"Saved alignment report to: {p_rep}")
        exported_paths["multi_room_alignment_report"] = p_rep

        return exported_paths

    def _assemble_objects(
        self,
        scans: Dict[str, ScanData],
        transforms: Dict[str, np.ndarray],
        output_objects_dir: Path,
        combined_output_ply: Path,
        manifest_path: Path,
    ) -> Dict[str, Any]:
        """Applies solved SE(3) transformation matrices to each scan's object instances.

        Exports individual aligned object PLYs, an aggregated aligned_objects.ply,
        and an authoritative aligned_objects_manifest.json.
        """
        output_objects_dir.mkdir(parents=True, exist_ok=True)
        all_object_meshes = []
        manifest_records = []

        for sname, scan in sorted(scans.items()):
            T = transforms[sname]
            scan_dir = scan.info.scan_dir
            instances_to_process = []
            masks_dir = scan_dir / "segmentation" / "3d_masks"
            if masks_dir.is_dir():
                for class_dir in sorted(masks_dir.iterdir()):
                    if not class_dir.is_dir() or not class_dir.name.endswith("_object"):
                        continue
                    parts = class_dir.name.split("_")
                    class_id = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                    for inst_ply in sorted(class_dir.glob("instance_*.ply")):
                        instances_to_process.append((inst_ply, {
                            "class_id": class_id,
                            "category": class_dir.name,
                            "role": "object",
                            "path": f"{class_dir.name}/{inst_ply.name}",
                        }))
            else:
                legacy_dir = scan_dir / "segmentation" / "Instances PLYs" / "Non-Structural"
                for class_dir in sorted(legacy_dir.iterdir()) if legacy_dir.is_dir() else []:
                    if not class_dir.is_dir():
                        continue
                    try:
                        class_id = int(class_dir.name)
                    except ValueError:
                        continue
                    for inst_ply in sorted(class_dir.glob("instance_*.ply")):
                        instances_to_process.append((inst_ply, {
                            "class_id": class_id,
                            "category": class_dir.name,
                            "role": "object",
                            "path": f"{class_dir.name}/{inst_ply.name}",
                        }))

            logger.info(f"Processing {len(instances_to_process)} object instances for {sname}...")
            scan_target_dir = output_objects_dir / sname
            scan_target_dir.mkdir(parents=True, exist_ok=True)

            for mesh_p, item in instances_to_process:
                m = trimesh.load(mesh_p, process=False)
                if not isinstance(m, trimesh.Trimesh) or not len(m.faces):
                    raise ValueError(f"Expected a non-empty Stage 05 object mesh: {mesh_p}")
                # Use source instances so every object receives exactly its scan's transform.
                m.apply_transform(T)

                rel_p = Path(item["path"])
                target_ply = scan_target_dir / rel_p
                target_ply.parent.mkdir(parents=True, exist_ok=True)
                m.export(str(target_ply))

                v = m.vertices
                manifest_records.append({
                    "scan_id": sname,
                    "class_id": item["class_id"],
                    "category": item["category"],
                    "role": item["role"],
                    "source_instance_path": str(mesh_p),
                    "global_instance_path": str(target_ply),
                    "triangles": len(m.faces),
                    "vertices": len(m.vertices),
                    "scan_transform_matrix": T.tolist(),
                    "local_placement_translation_m": [0.0, 0.0, 0.0],
                    "global_bounding_box": v.min(axis=0).tolist() + v.max(axis=0).tolist(),
                    "global_centroid": v.mean(axis=0).tolist(),
                })
                all_object_meshes.append(m)

        # Export combined scene of all aligned objects
        if all_object_meshes:
            combined_objects = trimesh.util.concatenate(all_object_meshes)
            combined_objects.export(str(combined_output_ply))
            logger.info(f"Exported combined aligned objects: {combined_output_ply} ({len(combined_objects.vertices)} verts, {len(combined_objects.faces)} faces)")
        else:
            logger.warning("No object instances were found or assembled.")

        # Export manifest
        obj_manifest = {
            "schema_version": "1.0.0",
            "total_objects": len(manifest_records),
            "objects": manifest_records,
        }
        with open(manifest_path, "w") as f:
            json.dump(obj_manifest, f, indent=2)
        logger.info(f"Saved aligned objects manifest to: {manifest_path}")

        return obj_manifest

    def apply_object_placement_translations(
        self,
        inside_fit_report: Dict[str, Any],
        plane_fit_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Publish fitted object positions while retaining their synchronized source colors."""
        stats_dir = self.output_dir / "stats"
        manifest_path = stats_dir / "aligned_objects_manifest.json"
        manifest = json.loads(manifest_path.read_text())

        translations = {}
        for report, field in ((inside_fit_report, "translation_m"),
                              (plane_fit_report, "translation_m")):
            for item in report["objects"]:
                key = str(Path(item["source_instance_path"]).resolve())
                if key not in translations:
                    translations[key] = np.zeros(3, dtype=np.float64)
                translations[key] += np.asarray(item[field], dtype=np.float64)

        records_by_source = {
            str(Path(item["source_instance_path"]).resolve()): item
            for item in manifest["objects"]
        }
        if translations.keys() != records_by_source.keys():
            missing = sorted(records_by_source.keys() - translations.keys())
            extra = sorted(translations.keys() - records_by_source.keys())
            raise ValueError(f"Object fit reports do not match aligned instances: missing={missing}, extra={extra}")

        meshes = []
        for source, record in records_by_source.items():
            mesh_path = Path(record["global_instance_path"])
            mesh = trimesh.load(mesh_path, process=False)
            if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
                raise ValueError(f"Expected an aligned object mesh: {mesh_path}")
            translation = translations[source]
            if translation.shape != (3,) or not np.isfinite(translation).all():
                raise ValueError(f"Invalid fitted translation for {mesh_path}")
            mesh.vertices += translation
            mesh.export(str(mesh_path))
            bounds = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
            record["post_scan_translation_m"] = translation.tolist()
            record["global_bounding_box"] = bounds[0].tolist() + bounds[1].tolist()
            record["global_centroid"] = mesh.vertices.mean(axis=0).tolist()
            meshes.append(mesh)

        combined_path = self.output_dir / "aligned_objects.ply"
        trimesh.util.concatenate(meshes).export(str(combined_path))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        return {
            "aligned_objects": combined_path,
            "aligned_objects_manifest": manifest_path,
        }

    @staticmethod
    def _assemble_regularized_meshes(
        scans: Dict[str, ScanData],
        transforms: Dict[str, np.ndarray],
        output_path: Path,
    ) -> None:
        """Transforms and merges Phase 06 regularized room meshes."""
        meshes = []
        for sname, scan in sorted(scans.items()):
            src_p = scan.info.regularized_complete_mesh_path
            if not src_p or not src_p.is_file():
                src_p = scan.info.sealed_mesh_path
            if not src_p or not src_p.is_file():
                continue

            try:
                m = trimesh.load(src_p, process=False)
                T = transforms[sname]
                m.apply_transform(T)
                meshes.append(m)
            except Exception as e:
                logger.warning(f"Error loading regularized mesh for {sname}: {e}")

        if meshes:
            combined = trimesh.util.concatenate(meshes)
            combined.export(str(output_path))
            logger.info(f"Exported: {output_path} ({len(combined.vertices)} verts, {len(combined.faces)} faces)")

    @staticmethod
    def _assemble_sealed_meshes(
        scans: Dict[str, ScanData],
        transforms: Dict[str, np.ndarray],
        output_path: Path,
    ) -> Dict[str, trimesh.Trimesh]:
        """Transforms and merges Phase 04 sealed room meshes."""
        meshes = []
        aligned_meshes = {}
        for sname, scan in sorted(scans.items()):
            src_p = scan.info.sealed_mesh_path
            if not src_p or not src_p.is_file():
                continue

            try:
                m = trimesh.load(src_p, process=False)
                T = transforms[sname]
                m.apply_transform(T)
                meshes.append(m)
                aligned_meshes[sname] = m
            except Exception as e:
                raise RuntimeError(f"Could not load sealed mesh for Gaussian overlap filtering: {sname}") from e

        if meshes:
            combined = trimesh.util.concatenate(meshes)
            combined.export(str(output_path))
            logger.info(f"Exported: {output_path} ({len(combined.vertices)} verts, {len(combined.faces)} faces)")
        return aligned_meshes

    @staticmethod
    def _assemble_red_doors_mesh(
        scans: Dict[str, ScanData],
        transforms: Dict[str, np.ndarray],
        output_path: Path,
    ) -> None:
        """Transforms sealed meshes, coloring Phase 05 semantic door triangles solid red."""
        all_verts = []
        all_faces = []
        all_colors = []
        vert_offset = 0

        neutral_color = [210, 210, 210, 255]
        door_red_color = [255, 0, 0, 255]

        for sname, scan in sorted(scans.items()):
            src_p = scan.info.sealed_mesh_path
            if not src_p or not src_p.is_file():
                continue

            try:
                m = trimesh.load(src_p, process=False)
                T = transforms[sname]
                v = (T[:3, :3] @ m.vertices.T).T + T[:3, 3]
                f = m.faces + vert_offset

                # Resolve face colors using Phase 05 triangle labels
                face_colors = np.full((len(m.faces), 4), neutral_color, dtype=np.uint8)
                if scan.info.triangle_labels_path and scan.info.triangle_labels_path.is_file() and scan.door_classes:
                    labels = np.load(scan.info.triangle_labels_path)
                    is_door = np.isin(labels, scan.door_classes)
                    face_colors[is_door] = door_red_color

                # Convert face colors to vertex colors
                vert_colors = np.full((len(m.vertices), 4), neutral_color, dtype=np.uint8)
                for f_idx, tri in enumerate(m.faces):
                    if np.array_equal(face_colors[f_idx], door_red_color):
                        vert_colors[tri[0]] = door_red_color
                        vert_colors[tri[1]] = door_red_color
                        vert_colors[tri[2]] = door_red_color

                all_verts.append(v)
                all_faces.append(f)
                all_colors.append(vert_colors)
                vert_offset += len(v)
            except Exception as e:
                logger.warning(f"Error assembling red doors for {sname}: {e}")

        if all_verts:
            combined_v = np.vstack(all_verts)
            combined_f = np.vstack(all_faces)
            combined_c = np.vstack(all_colors)

            final_m = trimesh.Trimesh(vertices=combined_v, faces=combined_f, vertex_colors=combined_c, process=False)
            final_m.export(str(output_path))
            logger.info(f"Exported: {output_path} ({len(combined_v)} verts, {len(combined_f)} faces)")

    @staticmethod
    def _assemble_gaussian_splats(
        scans: Dict[str, ScanData],
        transforms: Dict[str, np.ndarray],
        output_path: Path,
        mesh_scenes: Dict[str, Any],
    ) -> Dict[str, Dict[Path, Path]]:
        """Transforms 3D Gaussian Splats with exact quaternion composition."""
        all_vertex_arrays = []
        per_scan_paths: Dict[str, Dict[Path, Path]] = {}

        for sname, scan in sorted(scans.items()):
            src_p = scan.info.gaussian_splat_path
            if not src_p or not src_p.is_file():
                continue

            try:
                ply_data = plyfile.PlyData.read(str(src_p))
                v = transform_gaussian_splat_vertices(
                    ply_data["vertex"].data, transforms[sname], mesh_scenes, sname
                )
                for root in (
                    scan.info.scan_dir / ".raw_unaligned" / "gaussian_splat",
                    scan.info.scan_dir / "gaussian_splat",
                ):
                    try:
                        relative_path = src_p.resolve().relative_to(root.resolve())
                        break
                    except ValueError:
                        relative_path = None
                if relative_path is not None:
                    per_scan_path = (output_path.parent / "scans" / sname / "gaussian_splat" /
                                     relative_path)
                    write_gaussian_splat_vertices(v, per_scan_path)
                    per_scan_paths.setdefault(sname, {})[relative_path] = per_scan_path

                all_vertex_arrays.append(v)
            except Exception as e:
                if mesh_scenes:
                    raise RuntimeError(f"Gaussian alignment or overlap filtering failed for {sname}") from e
                logger.warning(f"Error transforming Gaussians for {sname}: {e}")

        if all_vertex_arrays:
            merged_data = np.concatenate(all_vertex_arrays)
            merged_el = plyfile.PlyElement.describe(merged_data, "vertex")
            merged_ply = plyfile.PlyData([merged_el], text=False, byte_order="<")
            merged_ply.write(str(output_path))
            logger.info(f"Exported: {output_path} ({len(merged_data)} splats)")
        return per_scan_paths
