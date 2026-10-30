"""Run the native TRELLIS.2 checkpoint with per-stage timing on ROCm/CUDA."""
import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")

import torch
from PIL import Image

from pixal3d.pipelines import Trellis2ImageTo3DPipeline
import o_voxel


class CascadePipeline(Trellis2ImageTo3DPipeline):
    # The 1024 cascade never calls the 512 texture flow model.
    load_on_meta = True
    model_names_to_load = [
        name for name in Trellis2ImageTo3DPipeline.model_names_to_load
        if name != "tex_slat_flow_model_512"
    ]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_methods(pipeline, names):
    timings = {}
    for name in names:
        method = getattr(pipeline, name)

        def measured(*args, _method=method, _name=name, **kwargs):
            sync()
            started = time.perf_counter()
            try:
                return _method(*args, **kwargs)
            finally:
                sync()
                elapsed = time.perf_counter() - started
                timings.setdefault(_name, []).append(elapsed)
                print(f"[TRELLIS_TIMING] {_name}={elapsed:.3f}s", flush=True)

        setattr(pipeline, name, measured)
    return timings


def run_image(pipeline, image_path, output, args, timings):
    timings.clear()
    with Image.open(image_path) as source:
        image = source.copy()
    mark = time.perf_counter()
    meshes = pipeline.run(
        image, seed=args.seed, pipeline_type="1024_cascade",
        max_num_tokens=args.max_num_tokens, fill_holes=False,
        sparse_structure_sampler_params={"steps": args.steps},
        shape_slat_sampler_params={"steps": args.steps},
        tex_slat_sampler_params={"steps": args.steps},
    )
    sync()
    inference_seconds = time.perf_counter() - mark

    mesh = meshes[0]
    resolution = round(1 / mesh.voxel_size)
    attrs = mesh.attrs
    finite = torch.isfinite(attrs)
    if not finite.all():
        defaults = torch.zeros_like(attrs)
        defaults[..., pipeline.pbr_attr_layout["roughness"]] = 1
        defaults[..., pipeline.pbr_attr_layout["alpha"]] = 1
        attrs = torch.where(finite, attrs, defaults)

    mark = time.perf_counter()
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=attrs,
        coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
        grid_size=resolution, aabb=[[-0.5] * 3, [0.5] * 3],
        decimation_target=args.decimation_target, texture_size=args.texture_size,
        remesh=True, remesh_band=1, remesh_project=0.0, use_tqdm=True,
    )
    glb.export(output, extension_webp=True)
    sync()
    export_seconds = time.perf_counter() - mark
    metrics = {
        "model": args.model,
        "image": str(image_path),
        "output": str(output),
        "gpu": torch.cuda.get_device_name(),
        "hip": torch.version.hip,
        "seed": args.seed,
        "steps": args.steps,
        "pipeline_type": "1024_cascade",
        "resolution": resolution,
        "low_vram": pipeline.low_vram,
        "inference_seconds": inference_seconds,
        "glb_export_seconds": export_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "steps_seconds": {name: values.copy() for name, values in timings.items()},
        "glb_bytes": output.stat().st_size,
    }
    print(f"[Metrics] inference={inference_seconds:.2f}s "
          f"glb={export_seconds:.2f}s peak_vram={metrics['peak_allocated_gib']:.2f}GiB")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, nargs="+", required=True,
                        help="One image or multiple images processed with one model load")
    parser.add_argument("--output", type=Path, required=True,
                        help="GLB path for one image; output directory for multiple images")
    parser.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=12,
                        help="Sampling steps per stage (12 is checkpoint default; fewer steps trade quality for speed)")
    parser.add_argument("--max-num-tokens", type=int, default=49152)
    parser.add_argument("--decimation-target", type=int, default=100000)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--keep-models-on-gpu", action="store_true",
                        help="Experimental: keep all models resident; can OOM on 16 GiB GPUs")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if len(args.image) > 1 and (args.output.suffix or len({p.stem for p in args.image}) != len(args.image)):
        parser.error("For multiple images, --output must be a directory and image stems must be unique")
    if len(args.image) > 1:
        args.output.mkdir(parents=True, exist_ok=True)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("A ROCm/CUDA GPU is required")
    torch.cuda.reset_peak_memory_stats()
    print(f"[Device] {torch.cuda.get_device_name()} | HIP={torch.version.hip}", flush=True)

    mark = time.perf_counter()
    pipeline = CascadePipeline.from_pretrained(args.model)
    pipeline._device = torch.device("cuda")
    pipeline.low_vram = not args.keep_models_on_gpu
    gpu_resident_setup = 0.0
    if args.keep_models_on_gpu:
        setup_mark = time.perf_counter()
        pipeline.to(torch.device("cuda"))
        sync()
        gpu_resident_setup = time.perf_counter() - setup_mark
        print(f"[TRELLIS_TIMING] gpu_resident_setup={gpu_resident_setup:.3f}s", flush=True)
    sync()
    model_load = time.perf_counter() - mark
    print(f"[TRELLIS_TIMING] model_load={model_load:.3f}s", flush=True)

    timings = timed_methods(pipeline, (
        "preprocess_image", "get_cond", "sample_sparse_structure",
        "sample_shape_slat_cascade", "sample_tex_slat", "decode_latent",
    ))
    results = []
    for image_path in args.image:
        output = args.output if len(args.image) == 1 else args.output / f"{image_path.stem}.glb"
        print(f"[Image] {image_path}", flush=True)
        metrics = run_image(pipeline, image_path, output, args, timings)
        metrics["model_load_seconds"] = model_load if len(args.image) == 1 else 0.0
        metrics["gpu_resident_setup_seconds"] = gpu_resident_setup if len(args.image) == 1 else 0.0
        metrics["total_seconds"] = metrics["model_load_seconds"] + metrics["inference_seconds"] + metrics["glb_export_seconds"]
        results.append((output, metrics))

    mark = time.perf_counter()
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    sync()
    model_unload = time.perf_counter() - mark
    total_seconds = time.perf_counter() - started
    for output, metrics in results:
        if len(results) == 1:
            metrics["model_unload_seconds"] = model_unload
            metrics["total_seconds"] = total_seconds
        metrics_path = output.with_suffix(".json")
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"[Done] GLB: {output}\n[Done] timings: {metrics_path}", flush=True)
    if len(results) > 1:
        summary = {
            "model_load_seconds": model_load,
            "model_unload_seconds": model_unload,
            "total_seconds": total_seconds,
            "mean_image_seconds": sum(m["total_seconds"] for _, m in results) / len(results),
            "max_image_seconds": max(m["total_seconds"] for _, m in results),
            "images": [str(p) for p in args.image],
        }
        (args.output / "trellis2_batch.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[Batch] mean={summary['mean_image_seconds']:.2f}s max={summary['max_image_seconds']:.2f}s "
              f"total={total_seconds:.2f}s", flush=True)


if __name__ == "__main__":
    main()
