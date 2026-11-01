#!/usr/bin/env python3
"""
Authoritative Stage 05 Entry Point: 3D Panoptic Semantic Segmentation & Structural Extraction.

Orchestrates:
1. Multi-scan automatic sequential discovery across scan_N directories under --workspace.
2. Alignment with locked Stage 00-04 upstream outputs:
   - Stage 00: Registered image directory (images/)
   - Stage 01: Refined COLMAP poses (sparse/0/cameras.txt, images.txt, points3D.txt)
   - Stage 02: Optional depth products (depth/depth_maps/, depth/depth_alignment.json)
   - Stage 04: Authoritative sealed 3D mesh (mesh.ply, mesh.json)
3. End-to-end execution:
   - Phase 1: 2D Class-Agnostic Segment Proposals & Streaming Global Memory Bank
   - Phase 2: Ray-Casted 3D Mesh Multi-View Association & Dispute Resolution
   - Phase 3: Geometry-Driven Structural Classification (Wall, Floor, Ceiling, Stairs, Columns, Ramps)
   - Phase 4: Regularized Clean Delivery (mesh_segmented.ply, Instances PLYs/, class_metadata.json, segmentation.json)
   - Phase 5: Rendered non-structural class matching against the fixed 28-category vocabulary.
4. Temporary file scrubbing: removes redundant working meshes to preserve minimal disk footprint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Resolve backend root and bootstrap pipeline paths
_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
_stage05_dir = Path(__file__).resolve().parent
if str(_stage05_dir) not in sys.path:
    sys.path.insert(0, str(_stage05_dir))

from Utilities.pipeline_paths import bootstrap
bootstrap()

from Utilities.pipeline_step import StepContext, is_done
import scene_inputs
from export_delivery import deliver_stage05_outputs


DEFAULT_WORKSPACE = _backend_dir / "current_scene"
ALGORITHM_VERSION = 'evidence-geometry-v5'
MIN_INSTANCE_TRIANGLES = 2000


def run_final_category_classification(scan_dir: Path, mesh_path: Path, stage_out_dir: Path) -> list[tuple[int, int]]:
    """Render delivered objects and classify their source classes for downstream stages."""
    print("\n--- Final Phase: EmbeddingGemma object-category matching ---")
    from extract_non_structural_frames import main as extract_frames
    from run_non_structural_category_similarity import main as classify_frames

    frame_dir = stage_out_dir / "non_structural_category_frames"
    mesh_frames = frame_dir / "mesh"
    gs_frames = frame_dir / "gs"
    output_dir = stage_out_dir / "non_structural_similarity"
    for frames in (mesh_frames, gs_frames):
        frames.mkdir(parents=True, exist_ok=True)
        for image in frames.iterdir():
            if image.is_file() and image.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                image.unlink()
    rc = extract_frames([
        "--workspace", str(scan_dir),
        "--mesh-path", str(mesh_path),
        "--labels-path", str(stage_out_dir / "triangle_class_labels.npy"),
        "--structural-json", str(stage_out_dir / "class_metadata.json"),
        "--output-dir", str(mesh_frames),
        "--gs-output-dir", str(gs_frames),
        "--include-structural",
    ])
    if rc:
        manifest_path = mesh_frames / "non_structural_frames_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Instance frame extraction failed for {scan_dir}: {exc}") from exc
        if manifest.get("total_instances_extracted", 0) != 0:
            raise RuntimeError(f"Instance frame extraction failed for {scan_dir} (exit code {rc})")

    associations = classify_frames([
        "--mesh-frames-dir", str(mesh_frames),
        "--gs-frames-dir", str(gs_frames),
        "--output-dir", str(output_dir),
    ])
    if not (output_dir / "valid_class_associations.json").is_file():
        raise RuntimeError(f"Category matcher did not write associations for {scan_dir}")
    return associations


def run_segmentation_pipeline_for_scan(
    scan_dir: Path,
    args: argparse.Namespace,
    ctx: StepContext,
) -> Dict[str, Any]:
    """
    Executes Stage 05 for a single scan directory.
    """
    t_start = time.perf_counter()
    scan_dir = Path(scan_dir).resolve()
    print(f"\n{'='*70}")
    print(f"STAGE 05: SEMANTIC SEGMENTATION - SCAN: {scan_dir.name}")
    print(f"Scan Directory:    {scan_dir}")
    print(f"{'='*70}")

    # 1. Resolve locked upstream inputs
    sealed_mesh_path = scene_inputs.resolve_mesh(scan_dir, args.mesh_path)
    images_dir = scene_inputs.resolve_images_dir(scan_dir)
    sparse_dir = scene_inputs.resolve_sparse_dir(scan_dir)
    depth_dir = scene_inputs.resolve_depth_dir(scan_dir)
    scene_size_path = depth_dir / "scene_size.txt" if depth_dir else None
    if scene_size_path is not None and not scene_size_path.is_file():
        scene_size_path = None

    stage_out_dir = scene_inputs.resolve_stage_output(scan_dir)
    stage_out_dir.mkdir(parents=True, exist_ok=True)
    input_hashes = {str(path): hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
                   for path in (sealed_mesh_path, sparse_dir / 'cameras.txt', sparse_dir / 'images.txt')}
    print(f"  Input Mesh (Stage 04): {sealed_mesh_path}")
    print(f"  Input Images:          {images_dir}")
    print(f"  Input COLMAP Sparse:   {sparse_dir}")
    print(f"  Stage Output:          {stage_out_dir}")

    # Step outputs to verify completion
    final_segmented_mesh = stage_out_dir / "mesh_segmented.ply"
    final_class_meta = stage_out_dir / "class_metadata.json"
    final_seg_json = stage_out_dir / "segmentation.json"
    final_seg_json.unlink(missing_ok=True)
    (stage_out_dir / "non_structural_similarity" / "valid_class_associations.json").unlink(missing_ok=True)

    # Load sealed mesh geometry
    import open3d as o3d
    base_mesh = o3d.io.read_triangle_mesh(str(sealed_mesh_path))
    num_verts = len(base_mesh.vertices)
    num_faces = len(base_mesh.triangles)
    ctx.note(f"Sealed mesh loaded: {num_verts:,} vertices, {num_faces:,} triangles")

    # -------------------------------------------------------------
    # Phase 1: 2D Panoptic Segmentation & Memory Bank ID Propagation
    # -------------------------------------------------------------
    panoptic_npz_dir = stage_out_dir / "panoptic_data/npz_data"
    panoptic_overlays_dir = stage_out_dir / "panoptic_data/overlays"
    summary_path = stage_out_dir / "full_pipeline_benchmark_summary.json"

    existing_npz = list(panoptic_npz_dir.glob("*.npz")) if panoptic_npz_dir.is_dir() else []
    # Also check legacy cache location
    legacy_npz = scan_dir / "05_semantic_segmentation/full362_final_pipeline/npz_data"
    if not existing_npz and legacy_npz.is_dir() and any(legacy_npz.glob("*.npz")):
        panoptic_npz_dir = legacy_npz
        existing_npz = list(legacy_npz.glob("*.npz"))

    cameras = scene_inputs.load_scene_cameras(scan_dir)
    counts = scene_inputs.colmap_point_counts(sparse_dir / 'images.txt')
    expected = sorted(camera['image_path'] for camera in cameras
                      if counts[camera['image_path'].relative_to(images_dir).as_posix()] >= args.colmap_threshold)
    if args.max_frames:
        expected = expected[:args.max_frames]
    if not expected:
        raise ValueError('No supported registered images are available for segmentation')
    if not {f'{p.stem}_final.npz' for p in expected}.issubset({p.name for p in existing_npz}):
        existing_npz = []
    report = panoptic_npz_dir.parent / 'full_pipeline_benchmark_summary.json'
    if not report.is_file() and existing_npz:
        report = existing_npz[0].resolve().parent.parent / 'full_pipeline_benchmark_summary.json'
    try:
        two_d_cache = json.loads(report.read_text())
    except (ValueError, OSError):
        two_d_cache = {}
    if two_d_cache.get('num_frames', 0) < len(expected):
        existing_npz = []
    if 'image_names' in two_d_cache and not {p.relative_to(images_dir).as_posix() for p in expected}.issubset(two_d_cache['image_names']):
        existing_npz = []
    panoptic_overlays_dir = panoptic_npz_dir.parent / 'overlays'

    if not existing_npz or args.recompute_2d:
        print("\n--- Phase 1: Running SAM 2.1 & DINOv2 2D Panoptic Pipeline ---")
        t0_2d = time.perf_counter()
        from run_full_scene_pipeline import main as run_2d_pipeline
        # Run 2D pipeline
        pipeline_args = [
            "--workspace", str(scan_dir),
            "--output-dir", str(stage_out_dir),
            "--colmap-threshold", str(args.colmap_threshold),
        ]
        if args.max_frames:
            pipeline_args.extend(["--max-frames", str(args.max_frames)])
        # Save sys.argv and invoke
        orig_argv = sys.argv
        try:
            sys.argv = ["run_full_scene_pipeline.py"] + pipeline_args
            run_2d_pipeline(pipeline_args)
        finally:
            sys.argv = orig_argv
        panoptic_npz_dir = stage_out_dir / 'panoptic_data/npz_data'
        panoptic_overlays_dir = stage_out_dir / 'panoptic_data/overlays'
        ctx.metric("stage1_2d_panoptic_seconds", round(time.perf_counter() - t0_2d, 2))
    else:
        print(f"\n--- Phase 1: Found existing 2D panoptic archives ({len(existing_npz)} frames). Skipping 2D inference. ---")

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # Phase 2: Ray-Casted 3D Mesh Multi-View Projection & Dispute Resolution
    # -------------------------------------------------------------
    from step2_project_masks_to_3d import main as project_masks_main
    projection_stats_path = stage_out_dir / "projection_stats.json"
    raw_labels_path = stage_out_dir / "triangle_class_labels_raw.npy"

    projection_cache_valid = False
    if raw_labels_path.is_file() and projection_stats_path.is_file():
        try:
            cached = json.loads(projection_stats_path.read_text())
        except (ValueError, OSError):
            cached = {}
        projection_cache_valid = (cached.get('projection_version') == 'view-normalized-v2'
                                  and cached.get('input_hashes') == input_hashes
                                  and cached.get('colmap_threshold') == args.colmap_threshold
                                  and cached.get('max_frames') == args.max_frames
                                  and cached.get('mask_files') == {p.name: [p.stat().st_size, p.stat().st_mtime_ns]
                                       for p in sorted(panoptic_npz_dir.glob('*_final.npz'))}
                                  and (stage_out_dir / 'face_class_votes.npz').is_file()
                                  and (stage_out_dir / 'face_class_source_ids.npy').is_file())
    if projection_cache_valid and not args.recompute_projection and not args.recompute_2d:
        print(f"\n--- Phase 2: Found existing raw 3D projections ({raw_labels_path.name}). Skipping ray-casting. ---")
    else:
        print("\n--- Phase 2: Projecting 2D Masks to 3D Mesh Faces ---")
        t0_proj = time.perf_counter()
        proj_args = [
            "--workspace", str(scan_dir),
            "--mesh-path", str(sealed_mesh_path),
            "--output-dir", str(stage_out_dir),
            "--npz-dir", str(panoptic_npz_dir),
            "--colmap-threshold", str(args.colmap_threshold),
        ]
        if args.max_frames:
            proj_args.extend(["--max-frames", str(args.max_frames)])
        orig_argv = sys.argv
        try:
            sys.argv = ["step2_project_masks_to_3d.py"] + proj_args
            project_masks_main(proj_args)
        finally:
            sys.argv = orig_argv
        ctx.metric("stage2_projection_seconds", round(time.perf_counter() - t0_proj, 2))

    # -------------------------------------------------------------
    # Phase 3: Geometry-Driven Structural Classification & Instance Segmentation
    # -------------------------------------------------------------
    print("\n--- Phase 3: Geometric Regularization & Structural Classification ---")
    t0_struct = time.perf_counter()
    from geometric_regularization import run_complete_geometric_regularization
    from detect_structural_classes import (
        build_mesh_dual_graph,
        estimate_gravity_and_scene_bands,
        analyze_class_geometry,
        classify_structural_classes,
        extract_instances,
        export_structural_manifest,
        read_scene_sizes,
        load_canonical_dino_embeddings,
    )

    with open(projection_stats_path) as f:
        proj_stats = json.load(f)
    active_classes = sorted(proj_stats["active_classes"])

    # Exact face labels and their competing evidence share the sealed mesh's face order.
    triangles_np = np.asarray(base_mesh.triangles)
    vertices_np = np.asarray(base_mesh.vertices)
    vertex_colors_np = (
        (np.asarray(base_mesh.vertex_colors) * 255.0).round().astype(np.uint8)
        if base_mesh.has_vertex_colors()
        else np.zeros((len(vertices_np), 3), dtype=np.uint8)
    )

    triangle_labels = np.load(raw_labels_path, allow_pickle=False)
    from scipy.sparse import load_npz
    face_priors = load_npz(stage_out_dir / 'face_class_votes.npz')
    source_ids = np.load(stage_out_dir / 'face_class_source_ids.npy', allow_pickle=False)
    from deferred_assignment import read_infill_attributes, assign_deferred_patches
    infill_faces, infill_rgb = read_infill_attributes(sealed_mesh_path, len(triangles_np))
    refinement_labels = triangle_labels.copy()
    refinement_labels[infill_faces] = -1
    face_priors = face_priors.multiply((~infill_faces)[:, None]).tocsr()
    cleaned_labels, reg_metrics = run_complete_geometric_regularization(
        vertices=vertices_np,
        triangles=triangles_np,
        vertex_colors=vertex_colors_np,
        raw_labels=refinement_labels,
        mesh=base_mesh,
        min_island_size=args.min_island_size,
        min_instance_size=MIN_INSTANCE_TRIANGLES,
        priors=face_priors,
        source_ids=source_ids,
    )
    ctx.metric("regularization_seconds", reg_metrics.get("total_regularization_seconds", 0.0))
    ctx.metric("regularization_classes", reg_metrics.get("surviving_classes", 0))
    (stage_out_dir / 'regularization.json').write_text(json.dumps(reg_metrics, indent=2) + '\n')

    np.save(stage_out_dir / "triangle_class_labels.npy", cleaned_labels)
    active_classes = sorted(np.unique(cleaned_labels[cleaned_labels > 0]).tolist())

    adj_u, adj_v = build_mesh_dual_graph(triangles_np)
    v0 = vertices_np[triangles_np[:, 0]]
    v1 = vertices_np[triangles_np[:, 1]]
    v2 = vertices_np[triangles_np[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    tri_areas = 0.5 * np.linalg.norm(cross, axis=1)
    tri_normals = cross / np.maximum(2.0 * tri_areas[:, None], 1e-8)
    tri_centroids = (v0 + v1 + v2) / 3.0

    gravity_up, scene_bands = estimate_gravity_and_scene_bands(vertices_np, triangles_np)
    scene_info = read_scene_sizes(scene_size_path)

    canonical_embs = load_canonical_dino_embeddings(stage_out_dir / "canonical_class_embeddings.npz", active_classes)

    features = analyze_class_geometry(
        vertices=vertices_np,
        triangles=triangles_np,
        triangle_labels=cleaned_labels,
        class_ids=active_classes,
        gravity_up=gravity_up,
        scene_bands=scene_bands,
    )

    classifications = classify_structural_classes(
        class_features=features,
        scene_bands=scene_bands,
        canonical_embs=canonical_embs,
        scene_info=scene_info,
    )

    from geometric_regularization import enforce_instance_constraints
    wall_cids = {int(cid) for cid, info in classifications.items()
                 if info.get('structural_role', '').startswith('wall')}
    if reg_metrics.get('wall_cid', 0) > 0:
        wall_cids.add(reg_metrics['wall_cid'])
    t_constraints = time.perf_counter()
    cleaned_labels, constraints = enforce_instance_constraints(
        cleaned_labels, adj_u, adj_v, tri_normals, gravity_up, wall_cids,
        MIN_INSTANCE_TRIANGLES, tri_areas=tri_areas)
    constraints['seconds'] = round(time.perf_counter() - t_constraints, 3)
    for role, cid in constraints['large_horizontal_class_ids'].items():
        classifications[cid] = {
            'class_id': cid,
            'structural_role': role,
            'confidence': 0.90,
            'floor_level': None,
            'details': 'Reclassified from a connected near-horizontal wall patch with area at least 2 m².',
            'piece_breakdown': [],
        }
    np.save(stage_out_dir / 'triangle_class_labels_observed.npy', cleaned_labels)
    t_deferred = time.perf_counter()
    face_rgb = np.mean(vertex_colors_np[triangles_np], axis=1) / 255.
    if infill_rgb is not None:
        face_rgb[infill_faces] = infill_rgb[infill_faces]
    cleaned_labels, deferred_metrics, instance_links = assign_deferred_patches(
        cleaned_labels, tri_centroids, tri_normals, tri_areas, face_rgb,
        adj_u, adj_v, gravity_up, wall_cids)
    np.save(stage_out_dir / 'deferred_instance_links.npy', instance_links)
    if len(instance_links):
        adj_u = np.r_[adj_u, instance_links[:, 0], instance_links[:, 1]]
        adj_v = np.r_[adj_v, instance_links[:, 1], instance_links[:, 0]]
    deferred_metrics['stage04_infill_faces'] = int(infill_faces.sum())
    deferred_metrics['infill_projection_votes_ignored'] = int(np.count_nonzero(infill_faces & (triangle_labels > 0)))
    reg_metrics['deferred_assignment'] = deferred_metrics
    deferred_metrics['seconds'] = round(time.perf_counter() - t_deferred, 3)
    ctx.note(f"Late geometry/RGB assignment: {deferred_metrics['assigned_faces']:,} faces assigned; "
             f"{deferred_metrics['remaining_faces']:,} unresolved without an eligible seed.")
    constraints['wall_cids'] = sorted(wall_cids)
    reg_metrics['instance_constraints'] = constraints
    reg_metrics['unknown_faces'] = int(np.count_nonzero(cleaned_labels <= 0))
    reg_metrics['surviving_classes'] = int(len(np.unique(cleaned_labels[cleaned_labels > 0])))
    reg_metrics['changed_observed_faces'] = int(np.count_nonzero((cleaned_labels != triangle_labels) & (triangle_labels > 0)))
    np.save(stage_out_dir / 'triangle_class_labels.npy', cleaned_labels)
    (stage_out_dir / 'regularization.json').write_text(json.dumps(reg_metrics, indent=2) + '\n')
    active_classes = sorted(np.unique(cleaned_labels[cleaned_labels > 0]).tolist())
    # Keep inferred roles fixed during the hard cleanup; refresh final geometry metrics.
    classifications = {cid: info for cid, info in classifications.items() if cid in active_classes}
    features = analyze_class_geometry(vertices_np, triangles_np, cleaned_labels, active_classes,
                                      gravity_up, scene_bands)
    instances, class_inst_counts = extract_instances(
        triangle_labels=cleaned_labels,
        active_classes=active_classes,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_areas=tri_areas,
        tri_centroids=tri_centroids,
        tri_normals=tri_normals,
        gravity_up=gravity_up,
        min_instance_size=MIN_INSTANCE_TRIANGLES,
        min_instance_area=0,
        min_instance_span=0,
        max_proximity_merge_dist=0,
    )

    if any(inst['triangle_count'] < MIN_INSTANCE_TRIANGLES for inst in instances.values()):
        raise RuntimeError('final delivery contains an undersized instance')
    if sum(inst['triangle_count'] for inst in instances.values()) != np.count_nonzero(cleaned_labels > 0):
        raise RuntimeError('final assignment produced a disconnected undersized component')

    deferred_metrics['final_instances'] = len(instances)
    deferred_metrics['final_minimum_instance_triangles'] = min((i['triangle_count'] for i in instances.values()), default=None)
    (stage_out_dir / 'regularization.json').write_text(json.dumps(reg_metrics, indent=2) + '\n')

    for inst in instances.values():
        cid = inst["class_id"]
        inst["assigned_role"] = classifications.get(cid, {}).get("structural_role", "object")

    export_structural_manifest(
        output_dir=stage_out_dir,
        classifications=classifications,
        features=features,
        scene_bands=scene_bands,
        instances=instances,
        class_instance_counts=class_inst_counts,
    )
    ctx.metric("stage3_structural_seconds", round(time.perf_counter() - t0_struct, 2))

    # -------------------------------------------------------------
    # Phase 4: Regularized Clean Output Delivery
    # -------------------------------------------------------------
    print("\n--- Phase 4: Delivering Regularized Outputs ---")
    t0_deliv = time.perf_counter()
    delivery_manifest = deliver_stage05_outputs(
        sealed_mesh_path=sealed_mesh_path,
        triangle_labels=cleaned_labels,
        instances=instances,
        classifications=classifications,
        output_dir=stage_out_dir,
        panoptic_npz_dir=panoptic_npz_dir if panoptic_npz_dir.is_dir() else None,
        panoptic_overlays_dir=panoptic_overlays_dir if panoptic_overlays_dir.is_dir() else None,
        create_legacy_symlinks=True,
    )
    delivery_manifest.update(algorithm_version=ALGORITHM_VERSION, input_hashes=input_hashes,
                             unknown_faces=reg_metrics['unknown_faces'],
                             settings=dict(colmap_threshold=args.colmap_threshold, max_frames=args.max_frames,
                                           min_instance_size=MIN_INSTANCE_TRIANGLES),
                             regularization_report='regularization.json')
    completion = stage_out_dir / 'segmentation.work.json'
    completion.write_text(json.dumps(delivery_manifest, indent=2) + '\n')
    completion.replace(stage_out_dir / 'segmentation.json')
    ctx.metric("stage4_delivery_seconds", round(time.perf_counter() - t0_deliv, 2))

    # -------------------------------------------------------------
    # Phase 5: Temporary Scrubbing (Eliminate Redundant Working Meshes)
    # -------------------------------------------------------------
    redundant_patterns = [
        "mesh_projected_contention.ply",
        "mesh_dispute_resolved.ply",
        "mesh_projected_resolved.ply",
        "mesh_projected_cleaned.ply",
        "mesh_projected_refined.ply",
        "mesh_with_bounding_boxes.ply",
        "mesh_cleaned_no_floaters.ply",
    ]
    if not args.keep_temps:
        for fname in redundant_patterns:
            p = stage_out_dir / fname
            if p.is_file():
                p.unlink(missing_ok=True)

    associations = run_final_category_classification(scan_dir, sealed_mesh_path, stage_out_dir)
    ctx.metric("categorized_object_classes", len(associations))

    elapsed_total = time.perf_counter() - t_start
    ctx.metric("total_segmentation_seconds", round(elapsed_total, 2))
    ctx.metric("total_instances_extracted", len(instances))
    ctx.metric("num_classes_discovered", len(active_classes))
    ctx.note(f"Stage 05 completed successfully in {elapsed_total:.2f}s: {len(instances)} instances across {len(active_classes)} classes.")

    return delivery_manifest


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE), help="Root workspace or scan directory")
    parser.add_argument("--mesh-path", default=None, help="Explicit path to Stage 04 sealed mesh.ply")
    parser.add_argument("--colmap-threshold", type=int, default=100, help="Min COLMAP points required per camera frame")
    parser.add_argument("--min-island-size", type=int, default=150, help="Min triangle island size for floater absorption")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames for fast dry-run")
    parser.add_argument("--recompute-2d", action="store_true", default=False, help="Force recomputing 2D panoptic inference")
    parser.add_argument("--recompute-projection", action="store_true", default=False, help="Force recomputing 3D ray-cast projection")
    parser.add_argument("--keep-temps", action="store_true", default=False, help="Keep intermediate working mesh PLYs")
    parser.add_argument("--force", action="store_true", default=False, help="Force execution even if already completed")
    parser.add_argument("--only-category-matching", action="store_true",
                        help="Run only the final category matcher on a current Stage 05 delivery")

    args = parser.parse_args(argv)
    if args.only_category_matching and (args.force or args.recompute_2d or args.recompute_projection):
        parser.error("--only-category-matching cannot be combined with recompute or --force options")
    raw_workspace = Path(args.workspace)
    if not raw_workspace.is_dir() and (_backend_dir / raw_workspace).is_dir():
        workspace = (_backend_dir / raw_workspace).resolve()
    else:
        workspace = raw_workspace.resolve()

    # Automatic multi-scan discovery across scan_N directories
    scans = scene_inputs.discover_scans(workspace)
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('--max-frames must be positive')
    print(f"Discovered {len(scans)} scan(s) under {workspace}: {[s.name for s in scans]}")

    failures = []
    with StepContext("segmentation", workspace, write_artifacts=False) as parent_ctx:
        for scan in scans:
            stage_out = scene_inputs.resolve_stage_output(scan)
            base_outputs = [
                stage_out / "mesh_segmented.ply",
                stage_out / "class_metadata.json",
                stage_out / "segmentation.json",
            ]
            manifest = stage_out / 'segmentation.json'
            current_version = False
            if manifest.is_file():
                try:
                    cached = json.loads(manifest.read_text())
                except (ValueError, OSError):
                    cached = {}
                mesh = scene_inputs.resolve_mesh(scan, args.mesh_path)
                sparse_dir = scene_inputs.resolve_sparse_dir(scan)
                paths = (mesh, sparse_dir / 'cameras.txt', sparse_dir / 'images.txt')
                hashes = {str(path): hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() for path in paths}
                settings = dict(colmap_threshold=args.colmap_threshold, max_frames=args.max_frames,
                                min_instance_size=MIN_INSTANCE_TRIANGLES)
                current_version = (cached.get('algorithm_version') == ALGORITHM_VERSION
                                   and cached.get('input_hashes') == hashes
                                   and cached.get('settings') == settings)
            if args.only_category_matching:
                if not current_version or not all(path.is_file() for path in base_outputs):
                    failures.append(scan.name)
                    parent_ctx.note(f"Scan {scan.name} has no current Stage 05 delivery for category matching.")
                    print(f"ERROR scan {scan.name}: run Stage 05 segmentation first; its delivery is missing or stale.")
                    continue
                try:
                    with StepContext("segmentation", scan, write_artifacts=False) as scan_ctx:
                        associations = run_final_category_classification(
                            scan, scene_inputs.resolve_mesh(scan, args.mesh_path), stage_out
                        )
                        scan_ctx.metric("categorized_object_classes", len(associations))
                except Exception as exc:
                    failures.append(scan.name)
                    parent_ctx.note(f"Scan {scan.name} category matching failed: {exc}")
                    print(f"ERROR category matching scan {scan.name}: {exc}")
                continue
            if not (args.force or args.recompute_2d or args.recompute_projection) and current_version and is_done(scan, "segmentation", base_outputs):
                associations_path = stage_out / "non_structural_similarity" / "valid_class_associations.json"
                if associations_path.is_file():
                    print(f"\nScan {scan.name}: Stage 05 already completed and verified. Skipping (use --force to re-run).")
                    continue
                print(f"\nScan {scan.name}: Segmentation is current; generating the missing final category associations.")
                try:
                    with StepContext("segmentation", scan, write_artifacts=False) as scan_ctx:
                        associations = run_final_category_classification(
                            scan, scene_inputs.resolve_mesh(scan, args.mesh_path), stage_out
                        )
                        scan_ctx.metric("categorized_object_classes", len(associations))
                except Exception as exc:
                    failures.append(scan.name)
                    parent_ctx.note(f"Scan {scan.name} category matching failed: {exc}")
                    print(f"ERROR finalizing categories for scan {scan.name}: {exc}")
                continue

            try:
                with StepContext("segmentation", scan, write_artifacts=False) as scan_ctx:
                    run_segmentation_pipeline_for_scan(scan, args, scan_ctx)
            except Exception as exc:
                failures.append(scan.name)
                parent_ctx.note(f"Scan {scan.name} failed: {exc}")
                print(f"ERROR processing scan {scan.name}: {exc}")

        if failures:
            raise RuntimeError(f"Segmentation failed for scans: {', '.join(failures)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
