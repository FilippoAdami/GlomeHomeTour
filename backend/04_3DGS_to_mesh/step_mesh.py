#!/usr/bin/env python3
"""Run TSDF fusion, cleanup, mesh/Gaussian sealing and RGB refinement in order."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
from Utilities.pipeline_paths import bootstrap

bootstrap()
from Utilities.pipeline_step import StepContext, is_done
from mesh_cleanup import clean_mesh_floaters
from regularize_mesh_rgb import regularize_mesh_rgb
from scene_alignment import estimate_alignment, transform_mesh_ply, transform_splats, _transform_colmap

DEFAULT_WORKSPACE = _backend_dir / "current_scene"


def camera_intrinsic(camera, o3d):
    width, height = camera["width"], camera["height"]
    K = camera["K"]
    return o3d.camera.PinholeCameraIntrinsic(
        width, height,
        float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]),
    )


def valid_depth(depth, alpha, min_alpha, depth_trunc):
    valid = np.isfinite(depth) & np.isfinite(alpha) & (depth > 0) & (depth < depth_trunc) & (alpha >= min_alpha)
    return np.where(valid, depth, 0).astype(np.float32), int(valid.sum())


def extract_mesh(workspace, model_path, output, voxel_size, sdf_trunc,
                 depth_trunc, min_alpha, frame_stride, resolution, ctx, sparse_dir=None):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Open3D is required for mesh extraction; install it in the backend environment: pip install open3d") from exc
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("gsplat mesh extraction requires an accessible CUDA/ROCm GPU")
    from PIL import Image
    from gsplat.rendering import rasterization
    from gsplat_train import load_cameras

    checkpoint_path = model_path / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "gsplat-stage03-v1":
        raise ValueError(f"Unsupported gsplat checkpoint: {checkpoint_path}")
    source = model_path / "gaussian_splat.ply"
    if not source.is_file():
        raise FileNotFoundError(f"gsplat point cloud missing: {source}")
    cameras = load_cameras(workspace, resolution or int(checkpoint["resolution"]),
                           sparse_dir=sparse_dir)[::frame_stride]
    if not cameras:
        raise RuntimeError(f"No training cameras found in {workspace}")
    splats = {name: value.to("cuda") for name, value in checkpoint["splats"].items()}
    quats = torch.nn.functional.normalize(splats["quats"], dim=-1)
    scales = splats["scales"].exp()
    opacities = splats["opacities"].sigmoid()
    # ED replaces colors with depth before rasterization; RGB comes from source photos.
    colors = torch.empty((len(splats["means"]), 1), device="cuda")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size, sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    valid_pixels = total_pixels = 0
    with torch.no_grad():
        for index, camera in enumerate(cameras, 1):
            rendered, opacity, _ = rasterization(
                means=splats["means"], quats=quats,
                scales=scales, opacities=opacities,
                colors=colors, viewmats=torch.as_tensor(camera["viewmat"], device="cuda")[None],
                Ks=torch.as_tensor(camera["K"], device="cuda")[None],
                width=camera["width"], height=camera["height"],
                render_mode="ED", packed=False,
            )
            depth = rendered[0, :, :, 0].cpu().numpy()
            alpha = opacity[0, :, :, 0].cpu().numpy()
            del rendered, opacity
            depth, count = valid_depth(depth, alpha, min_alpha, depth_trunc)
            valid_pixels += count
            total_pixels += depth.size
            if not count:
                continue
            with Image.open(camera["image_path"]) as image:
                color = np.asarray(image.convert("RGB").resize((camera["width"], camera["height"])))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(color)),
                o3d.geometry.Image(np.ascontiguousarray(depth)),
                depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False,
            )
            volume.integrate(rgbd, camera_intrinsic(camera, o3d), camera["viewmat"].astype(np.float64))
            del rgbd, color, depth, alpha
            if index % 20 == 0 or index == len(cameras):
                ctx.note(f"Fused {index}/{len(cameras)} training cameras")
    if not valid_pixels:
        raise RuntimeError(f"No valid rendered depth in {len(cameras)} training cameras; check alpha threshold and depth range")

    mesh = volume.extract_triangle_mesh()
    mesh.remove_unreferenced_vertices()
    if not len(mesh.triangles):
        raise RuntimeError(f"TSDF produced no triangles from {valid_pixels} valid pixels in {len(cameras)} training cameras; try a coarser voxel size or smaller frame stride")
    mesh.compute_vertex_normals()
    output.mkdir(parents=True, exist_ok=True)
    mesh_path = output / "mesh.work.ply"
    if not o3d.io.write_triangle_mesh(str(mesh_path), mesh):
        raise RuntimeError(f"Failed to write mesh: {mesh_path}")
    summary = {
        "selected_frames": len(cameras), "valid_pixels": valid_pixels,
        "valid_pixel_rate": valid_pixels / total_pixels,
        "frame_stride": frame_stride, "image_width": cameras[0]["width"],
        "image_height": cameras[0]["height"],
        "voxel_size": voxel_size, "sdf_trunc": sdf_trunc,
        "depth_trunc": depth_trunc, "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "engine": "gsplat",
        "source_model_path": str(source.resolve()),
    }
    ctx.metric("frames_processed", len(cameras))
    ctx.metric("mesh_path", str(mesh_path))
    ctx.note(f"Mesh: {mesh_path} ({summary['vertices']} vertices, {summary['triangles']} triangles)")
    return summary


def mesh_scenes(workspace):
    """Use the upstream scan-manifest contract, including standalone COLMAP scenes."""
    from step_colmap import discover_scenes

    try:
        return [scene for scene, _ in discover_scenes(workspace)]
    except FileNotFoundError:
        return [workspace]


def remove_obsolete_outputs(workspace, output, model_path):
    """Remove only known stage-generated copies after the compact outputs are complete."""
    if output.resolve() != workspace.resolve():
        return
    legacy_output = workspace / "04_3DGS_to_mesh"
    if not legacy_output.is_dir() or legacy_output.is_symlink():
        return
    for name in ("mesh_initial.ply", "mesh_cleaned.ply", "mesh_sealed.ply", "mesh_red.ply",
                 "mesh_floater_removal_stats.json", "mesh_rgb_regularization.json",
                 "mesh_rgb_log.txt", "mesh_rgb_stats.json", "mesh.ply", "mesh.json",
                 "mesh_log.txt", "mesh_stats.json"):
        (legacy_output / name).unlink(missing_ok=True)
    legacy_model = legacy_output / "gaussian_splat"
    if (legacy_model.is_dir() and not legacy_model.is_symlink()
            and legacy_model.resolve() != model_path.resolve()):
        for name in ("checkpoint.pt", "gaussian_splat.ply"):
            (legacy_model / name).unlink(missing_ok=True)
        if not any(legacy_model.iterdir()):
            legacy_model.rmdir()
    if not any(legacy_output.iterdir()):
        legacy_output.rmdir()


def previous_in_place_transform(report_path: Path) -> np.ndarray:
    """Return the source-to-current transform when the sparse files were aligned in place."""
    if not report_path.is_file():
        return np.eye(4, dtype=np.float64)
    try:
        metadata = json.loads(report_path.read_text(encoding="utf-8")).get("mesh_alignment", {})
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot recover the prior Phase 04 pose frame from {report_path}") from exc
    if not metadata.get("poses_written_in_place"):
        return np.eye(4, dtype=np.float64)
    transform = np.asarray(metadata.get("transform_matrix"), dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"Invalid in-place camera alignment transform in {report_path}")
    return transform


def transform_depth_poses_in_place(path: Path, transform: np.ndarray) -> None:
    if not path.is_file():
        return
    with np.load(path) as source:
        data = {key: source[key] for key in source.files}
    if "w2c" in data:
        data["w2c"] = data["w2c"] @ np.linalg.inv(transform).astype(data["w2c"].dtype)
    temporary = path.with_name(path.name + ".aligned.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_aligned_poses(staged_sparse: Path, sparse_dir: Path,
                          depth_poses: Path, transform: np.ndarray) -> None:
    # These derived records override the edited text poses in COLMAP readers.
    stale_models = [sparse_dir / name for name in (
        "cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin",
        "frames.txt", "rigs.txt",
    ) if (sparse_dir / name).is_file()]
    paths = [sparse_dir / "images.txt", sparse_dir / "points3D.txt"]
    sparse_ply = sparse_dir / "points3D.ply"
    if sparse_ply.is_file():
        paths.append(sparse_ply)
    paths.extend(stale_models)
    if depth_poses.is_file():
        paths.append(depth_poses)
    original = {path: path.read_bytes() for path in paths}
    try:
        (staged_sparse / "images.txt").replace(sparse_dir / "images.txt")
        (staged_sparse / "points3D.txt").replace(sparse_dir / "points3D.txt")
        transform_depth_poses_in_place(depth_poses, transform)
        if sparse_ply.is_file():
            transform_mesh_ply(sparse_ply, transform)
        for path in stale_models:
            path.unlink()
    except BaseException:
        for path, data in original.items():
            temporary = path.with_name(path.name + ".restore.tmp")
            temporary.write_bytes(data)
            temporary.replace(path)
        raise


def remove_old_alignment_directory(output: Path) -> None:
    """Remove files produced by the former sidecar contract, preserving unknown files."""
    alignment = output / "phase04_alignment"
    if not alignment.is_dir() or alignment.is_symlink():
        return
    for name in ("alignment.json", "sparse/0/cameras.txt", "sparse/0/images.txt",
                 "sparse/0/points3D.txt", "depth/poses_da3.npz"):
        path = alignment / name
        if any(parent.is_symlink() for parent in path.parents if parent != alignment.parent):
            continue
        if path.is_file() and not path.is_symlink():
            path.unlink()
    for directory in (alignment / "sparse/0", alignment / "sparse", alignment / "depth", alignment):
        try:
            directory.rmdir()
        except OSError:
            pass


def mesh_scan(workspace, args):
    """Build one final mesh and add a sealed model beside the original Gaussian files."""
    model_path = Path(args.model_path).resolve() if args.model_path else workspace / "gaussian_splat"
    output = Path(args.output).resolve() if args.output else workspace
    final_mesh = output / "mesh.ply"
    sparse_dir = workspace / "sparse" / "0"
    sealed_checkpoint = model_path / "checkpoint_sealed.pt"
    sealed_ply = model_path / "gaussian_splat_sealed.ply"
    outputs = [final_mesh, output / "mesh.json", sealed_checkpoint, sealed_ply,
               sparse_dir / "images.txt", sparse_dir / "points3D.txt"]
    if not args.force and is_done(workspace, "mesh", outputs):
        try:
            info = json.loads((output / "mesh.json").read_text(encoding="utf-8"))
            if (info.get("engine") == "gsplat" and info.get("cleanup")
                    and info.get("rgb_regularization")
                    and info.get("mesh_alignment", {}).get("poses_written_in_place")
                    and info.get("gsplat_sealing", {}).get("ply_path") == str(sealed_ply.resolve())
                    and info.get("gsplat_sealing", {}).get("checkpoint_path") == str(sealed_checkpoint.resolve())):
                remove_old_alignment_directory(output)
                remove_obsolete_outputs(workspace, output, model_path)
                print("[mesh] already done, skipping (use --force to re-run)")
                return 0
        except (ValueError, OSError):
            pass
    if not (model_path / "checkpoint.pt").is_file():
        raise FileNotFoundError(f"No gsplat checkpoint at {model_path / 'checkpoint.pt'}; run train first")
    working_mesh = output / "mesh.work.ply"
    previous_transform = previous_in_place_transform(output / "mesh.json")
    with (StepContext("mesh", workspace, artifacts_dir=output) as ctx,
          tempfile.TemporaryDirectory(prefix=".mesh-infill-", dir=model_path) as staged_model,
          tempfile.TemporaryDirectory(prefix=".phase04-source-", dir=sparse_dir.parent) as source_temp,
          tempfile.TemporaryDirectory(prefix=".phase04-pose-stage-", dir=sparse_dir.parent) as pose_temp):
        from Utilities.pipeline_step import read_pipeline_stats
        stats = read_pipeline_stats(workspace)
        stats.pop("mesh_rgb", None)
        (workspace / "pipeline_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        try:
            source_sparse = sparse_dir
            if not np.allclose(previous_transform, np.eye(4), atol=1e-12):
                source_sparse = Path(source_temp) / "sparse" / "0"
                _transform_colmap(sparse_dir / "images.txt", sparse_dir / "points3D.txt",
                                  source_sparse, np.linalg.inv(previous_transform))
                shutil.copy2(sparse_dir / "cameras.txt", source_sparse / "cameras.txt")
            summary = extract_mesh(workspace, model_path, output, args.voxel_size,
                                   args.sdf_trunc, args.depth_trunc, args.min_alpha, args.frame_stride,
                                   args.resolution, ctx, sparse_dir=source_sparse)
            clean_summary = clean_mesh_floaters(
                working_mesh, output, gsplat_model_path=model_path,
                output_mesh_path=working_mesh,
                gsplat_output_dir=Path(staged_model),
                max_gsplat_ratio=args.gsplat_max_ratio,
                target_disk_radius_m=args.gsplat_target_disk_radius,
                overlap_factor=args.gsplat_overlap_factor,
                sample_density_m=args.gsplat_sample_density,
                normal_thickness_m=args.gsplat_normal_thickness,
                coverage_deficit_threshold_m=args.gsplat_coverage_deficit_threshold,
            )
            rgb_summary = regularize_mesh_rgb(working_mesh)
            rgb_summary["mesh_path"] = str(final_mesh.resolve())
            clean_summary["output_mesh"] = str(final_mesh.resolve())
            sealing = clean_summary.pop("gsplat_sealing")
            sealing.update(checkpoint_path=str(sealed_checkpoint.resolve()),
                           ply_path=str(sealed_ply.resolve()))
            import open3d as o3d
            mesh_for_alignment = o3d.io.read_triangle_mesh(str(working_mesh))
            alignment = estimate_alignment(np.asarray(mesh_for_alignment.vertices),
                                           np.asarray(mesh_for_alignment.triangles),
                                           source_sparse / "images.txt")
            transform = np.asarray(alignment["transform_matrix"], dtype=np.float64)
            transform_mesh_ply(working_mesh, transform)
            transform_splats(Path(staged_model) / "checkpoint_sealed.pt",
                             Path(staged_model) / "gaussian_splat_sealed.ply", transform)
            staged_sparse = Path(pose_temp) / "sparse" / "0"
            _transform_colmap(source_sparse / "images.txt", source_sparse / "points3D.txt",
                              staged_sparse, transform)
            alignment.update({"poses_written_in_place": True,
                              "sparse_dir": str(sparse_dir.resolve()),
                              "depth_poses_path": str((workspace / "depth" / "poses_da3.npz").resolve())
                              if (workspace / "depth" / "poses_da3.npz").is_file() else None})
            summary.update({"raw_vertices": summary["vertices"],
                            "raw_triangles": summary["triangles"],
                            "vertices": clean_summary["sealed_vertices"],
                            "triangles": clean_summary["sealed_triangles"],
                            "mesh_path": str(final_mesh.resolve()),
                            "original_triangles_count": clean_summary["original_triangles_count"],
                            "filled_triangles_count": clean_summary["filled_triangles_count"],
                            "boundary_loops_filled": clean_summary["boundary_loops_filled"],
                            "remaining_boundary_edges": clean_summary["remaining_boundary_edges"],
                            "cleanup": clean_summary,
                            "gsplat_sealing": sealing,
                            "rgb_regularization": rgb_summary,
                            "mesh_alignment": alignment})
            report_tmp = output / "mesh.json.tmp"
            report_tmp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            # shortcut: mesh.json is the only source-frame history; rerun Stage 01 if it is deleted.
            delta = transform @ np.linalg.inv(previous_transform)
            publish_aligned_poses(staged_sparse, sparse_dir,
                                  workspace / "depth" / "poses_da3.npz", delta)
            (Path(staged_model) / "checkpoint_sealed.pt").replace(sealed_checkpoint)
            (Path(staged_model) / "gaussian_splat_sealed.ply").replace(sealed_ply)
            working_mesh.replace(final_mesh)
            report_tmp.replace(output / "mesh.json")
            remove_old_alignment_directory(output)
            remove_obsolete_outputs(workspace, output, model_path)
            ctx.metric("mesh_path", str(final_mesh.resolve()))
            ctx.metric("infill_gaussians_added", sealing["infill_gaussians_added"])
            ctx.metric("total_sealed_gaussians", sealing["total_sealed_gaussians"])
            ctx.metric("rgb_mean_absolute_change", rgb_summary["mean_absolute_rgb_change"])
            ctx.note(f"Final sealed, RGB-refined mesh: {final_mesh}")
        finally:
            for temporary in (working_mesh, output / "mesh.work.seal.tmp.ply",
                              output / "mesh.work.rgb.tmp.ply", output / "mesh.json.tmp"):
                temporary.unlink(missing_ok=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--model-path", help="gsplat model directory (default: workspace/gaussian_splat)")
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--sdf-trunc", type=float, default=0.05)
    parser.add_argument("--depth-trunc", type=float, default=10.0)
    parser.add_argument("--min-alpha", type=float, default=0.5)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--resolution", type=int,
                        help="Optional image downscale for lower-memory extraction; default uses checkpoint resolution")
    parser.add_argument("--output", help="Mesh directory (default: scan workspace)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gsplat-max-ratio", type=float, default=0.10,
                        help="Gaussian addition fraction, rounded with a minimum of one (default: 0.10)")
    parser.add_argument("--gsplat-target-disk-radius", type=float, default=0.035,
                        help="Target radius in scene units for planar Gaussian disks (default: 0.035)")
    parser.add_argument("--gsplat-overlap-factor", type=float, default=1.4,
                        help="Overlap multiplier for tangent scale (default: 1.4)")
    parser.add_argument("--gsplat-sample-density", type=float, default=None,
                        help="Optional explicit point spacing in scene units for sampling Gaussians across infilled triangles")
    parser.add_argument("--gsplat-normal-thickness", type=float, default=0.002,
                        help="Normal thickness in scene units for squeezed planar Gaussians (default: 0.002)")
    parser.add_argument("--gsplat-coverage-deficit-threshold", type=float, default=0.06,
                        help="Distance threshold in scene units to identify TSDF mesh triangles with no nearby Gaussians (default: 0.06)")
    args = parser.parse_args(argv)
    if any(not math.isfinite(v) or v <= 0 for v in (args.voxel_size, args.sdf_trunc, args.depth_trunc)):
        parser.error("voxel size and truncation distances must be positive and finite")
    if not 0 <= args.min_alpha <= 1 or not math.isfinite(args.min_alpha):
        parser.error("--min-alpha must be between 0 and 1")
    if args.frame_stride < 1:
        parser.error("--frame-stride must be positive")
    if args.resolution is not None and args.resolution < 1:
        parser.error("--resolution must be positive")
    if not math.isfinite(args.gsplat_max_ratio) or not 0 < args.gsplat_max_ratio <= 1:
        parser.error("--gsplat-max-ratio must be in (0, 1]")
    for name in ("gsplat_target_disk_radius", "gsplat_overlap_factor", "gsplat_sample_density",
                 "gsplat_normal_thickness", "gsplat_coverage_deficit_threshold"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    workspace = Path(args.workspace).resolve()
    scenes = mesh_scenes(workspace)
    if len(scenes) > 1 and (args.model_path or args.output):
        parser.error("--model-path and --output require a single scan workspace")
    if scenes == [workspace]:
        return mesh_scan(workspace, args)
    failures = []
    with StepContext("mesh", workspace, write_artifacts=False) as ctx:
        for scene in scenes:
            try:
                mesh_scan(scene, args)
            except Exception as exc:
                failures.append(scene.name)
                ctx.note(f"{scene.name}: mesh extraction failed: {exc}")
        if failures:
            raise RuntimeError(f"Mesh extraction failed for scans: {', '.join(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
