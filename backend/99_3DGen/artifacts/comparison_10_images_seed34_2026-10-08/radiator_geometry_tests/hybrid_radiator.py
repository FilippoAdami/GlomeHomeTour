#!/usr/bin/env python3
"""Proof of concept: TRELLIS.2 shape latent decoded/textured by Pixal3D."""
import gc
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ["PIXAL_TIMING"] = "1"

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
IMAGE = ROOT / "images/radiator.jpg"
OUTPUT = OUT / "trellis_shape_pixal_texture_radiator.glb"
sys.path.insert(0, str(ROOT / "Pixal3D"))
import o_voxel
import inference as pixal_inference
from pixal3d.pipelines import Trellis2ImageTo3DPipeline
from pixal3d.modules.sparse import SparseTensor


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measured(times, name, fn):
    sync(); start = time.perf_counter()
    result = fn()
    sync(); times[name] = time.perf_counter() - start
    print(f"[HYBRID_TIMING] {name}={times[name]:.3f}s", flush=True)
    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm/CUDA GPU required")
    device = torch.device("cuda")
    torch.manual_seed(34)
    timings = {}
    start_all = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    print(f"[Device] {torch.cuda.get_device_name()} | seed=34", flush=True)

    trellis = measured(timings, "trellis_model_load", lambda: Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B"))
    trellis._device = device
    trellis.low_vram = True
    image = Image.open(IMAGE)
    image = measured(timings, "trellis_preprocess", lambda: trellis.preprocess_image(image))
    cond512 = measured(timings, "trellis_condition_512", lambda: trellis.get_cond([image], 512))
    cond1024 = measured(timings, "trellis_condition_1024", lambda: trellis.get_cond([image], 1024))
    torch.manual_seed(34)
    coords = measured(timings, "trellis_sparse_structure", lambda: trellis.sample_sparse_structure(cond512, 32))
    shape_slat, resolution = measured(
        timings, "trellis_shape_cascade",
        lambda: trellis.sample_shape_slat_cascade(
            cond512, cond1024,
            trellis.models["shape_slat_flow_model_512"],
            trellis.models["shape_slat_flow_model_1024"],
            512, 1024, coords, {}, 49152,
        ),
    )
    if resolution != 1024:
        raise RuntimeError(f"Expected 1024 shape latent, got {resolution}")
    # Transfer just the shared shape latent, then release TRELLIS.2 before loading Pixal3D.
    shape_slat_cpu = SparseTensor(
        feats=shape_slat.feats.detach().cpu(),
        coords=shape_slat.coords.detach().cpu(),
        shape=shape_slat.shape,
    )
    del shape_slat, trellis, coords, cond512, cond1024
    gc.collect(); torch.cuda.empty_cache()

    pixal = measured(timings, "pixal_model_load", lambda: pixal_inference.init_pipeline(low_vram=True))
    pixal_image = measured(timings, "pixal_preprocess", lambda: pixal.preprocess_image(Image.open(IMAGE)))
    moge = measured(timings, "moge_model_load", lambda: pixal_inference.load_moge_model(device="cuda"))
    camera = measured(timings, "moge_camera_estimate", lambda: pixal_inference.get_camera_params_wild_moge(pixal_image, moge))
    moge.cpu(); del moge; torch.cuda.empty_cache()
    shape_slat = SparseTensor(
        feats=shape_slat_cpu.feats.to(device),
        coords=shape_slat_cpu.coords.to(device),
        shape=shape_slat_cpu.shape,
    )
    del shape_slat_cpu
    cond_tex = measured(
        timings, "pixal_texture_condition",
        lambda: pixal.get_proj_cond_shape(
            pixal.image_cond_model_tex_1024, [pixal_image], shape_slat.coords,
            camera_angle_x=camera["camera_angle_x"], distance=camera["distance"],
            mesh_scale=camera["mesh_scale"],
        ),
    )
    torch.manual_seed(34)
    tex_slat = measured(
        timings, "pixal_texture_sampling",
        lambda: pixal.sample_tex_slat(
            cond_tex, pixal.models["tex_slat_flow_model_1024"], shape_slat,
            {"steps": 12, "guidance_strength": 1.0,
             "guidance_rescale": 0.0, "rescale_t": 3.0},
        ),
    )
    del cond_tex
    mesh = measured(timings, "pixal_decode_shared_shape_and_texture", lambda: pixal.decode_latent(shape_slat, tex_slat, resolution)[0])
    attrs = mesh.attrs
    attr_layout = pixal.pbr_attr_layout
    if not torch.isfinite(attrs).all():
        defaults = torch.zeros_like(attrs)
        defaults[..., attr_layout["roughness"]] = 1
        defaults[..., attr_layout["alpha"]] = 1
        attrs = torch.where(torch.isfinite(attrs), attrs, defaults)
    del shape_slat, tex_slat, pixal
    gc.collect(); torch.cuda.empty_cache()
    timings["allocated_gib_before_export"] = torch.cuda.memory_allocated() / 2**30
    output_mesh = measured(
        timings, "glb_postprocess_export",
        lambda: o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=attrs,
            coords=mesh.coords, attr_layout=attr_layout,
            grid_size=resolution, aabb=[[-0.5] * 3, [0.5] * 3],
            decimation_target=150000, texture_size=1024,
            remesh=True, remesh_band=1, remesh_project=0.0, use_tqdm=True,
        ),
    )
    rot = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    output_mesh.apply_transform(rot)
    output_mesh.export(OUTPUT, extension_webp=True)
    sync()
    timings["total_seconds"] = time.perf_counter() - start_all
    metrics = {
        "image": str(IMAGE), "output": str(OUTPUT), "seed": 34,
        "shape_source": "microsoft/TRELLIS.2-4B shape_slat_flow_model_512+1024",
        "texture_source": "TencentARC/Pixal3D tex_slat_flow_model_1024 with projection conditioning",
        "resolution": resolution, "camera_params": camera,
        "max_num_tokens": 49152, "texture_steps": 12,
        "decimation_target": 150000, "texture_size": 1024,
        "low_vram": True, "timings_seconds": timings,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "glb_bytes": OUTPUT.stat().st_size,
    }
    (OUT / "hybrid_radiator.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[Done] {OUTPUT} | {timings['total_seconds']:.1f}s | peak {metrics['peak_allocated_gib']:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
