"""Benchmark full pass of StableNormal Turbo on all frames of current_scene/images.

Measures:
1. Model loading time.
2. Total inference time across all images.
3. Per-frame throughput and average latency.
4. Model unloading and VRAM release time.
5. Exact end-to-end elapsed time (start before load -> finish after unload).
"""

from __future__ import annotations

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("MIOPEN_USER_DB_PATH", "/tmp/miopen")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import gc
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
from Utilities.pipeline_paths import bootstrap

bootstrap()

from stable_normal import StableNormalEstimator
from depth_priors import colorize_normals


def get_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def main():
    print("=" * 65)
    print("StableNormal Turbo: Full Pass Benchmark (res=768, full upscale)")
    print("=" * 65)

    images_dir = _backend_dir / "current_scene" / "images"
    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    num_frames = len(image_paths)
    print(f"Total frames found: {num_frames} in {images_dir}")
    if num_frames == 0:
        print("[Error] No images found!")
        return 1

    out_normals_dir = _backend_dir / "current_scene" / "02_depth_estimation" / "depth" / "normal_maps"
    out_normals_dir.mkdir(parents=True, exist_ok=True)
    out_diag_dir = _backend_dir / "current_scene" / "02_depth_estimation" / "depth" / "normal_images"
    out_diag_dir.mkdir(parents=True, exist_ok=True)

    vram_start = get_vram_mb()
    print(f"[VRAM] Initial GPU memory: {vram_start:.2f} MB")

    total_start = time.time()

    # 1. Model Loading
    t_load_0 = time.time()
    estimator = StableNormalEstimator(device="cuda" if torch.cuda.is_available() else "cpu", resolution=768)
    estimator.load()
    t_load_1 = time.time()
    load_duration = t_load_1 - t_load_0
    vram_loaded = get_vram_mb()
    print(f"[Timing] Model load time: {load_duration:.2f}s (VRAM: {vram_loaded:.2f} MB)")

    # 2. Sequential Inference with Background Disk Saving
    def _save_to_disk(stem: str, norm_map: np.ndarray, orig_rgb: np.ndarray):
        np.save(out_normals_dir / f"{stem}.npy", norm_map.astype(np.float16))
        norm_rgb = colorize_normals(norm_map)
        composite = np.hstack([orig_rgb, norm_rgb])
        Image.fromarray(composite).save(out_diag_dir / f"{stem}.jpg", quality=90)

    t_infer_0 = time.time()
    norm_shape = None
    with ThreadPoolExecutor(max_workers=8) as io_pool:
        with torch.inference_mode():
            for idx, img_path in enumerate(image_paths):
                f_t0 = time.time()
                orig_img = Image.open(img_path).convert("RGB")
                orig_arr = np.asarray(orig_img)

                # Inference returns (H, W, 3) normalized float32 array at full input resolution
                norm_map = estimator.estimate_normal(orig_img)
                if norm_shape is None:
                    norm_shape = norm_map.shape

                io_pool.submit(_save_to_disk, img_path.stem, norm_map, orig_arr)

                f_elapsed = time.time() - f_t0
                if (idx + 1) % 50 == 0 or idx == 0 or (idx + 1) == num_frames:
                    vram_curr = get_vram_mb()
                    avg_s = (time.time() - t_infer_0) / (idx + 1)
                    eta_s = avg_s * (num_frames - (idx + 1))
                    print(f"  [{idx + 1:3d}/{num_frames}] Frame: {f_elapsed:.3f}s | "
                          f"Avg: {avg_s:.3f}s/frame ({1.0/avg_s:.1f} FPS) | "
                          f"ETA: {eta_s:.1f}s | VRAM: {vram_curr:.1f} MB", flush=True)

    t_infer_1 = time.time()
    infer_duration = t_infer_1 - t_infer_0

    # 3. Model Unload & Full GPU Release
    t_unload_0 = time.time()
    estimator.unload()
    t_unload_1 = time.time()
    unload_duration = t_unload_1 - t_unload_0

    total_end = time.time()
    total_elapsed = total_end - total_start
    vram_final = get_vram_mb()

    print("\n" + "=" * 65)
    print("BENCHMARK RESULTS")
    print("=" * 65)
    print(f"Total Frames Processed : {num_frames}")
    if norm_shape:
        print(f"Output Resolution      : {norm_shape[1]}x{norm_shape[0]} (Full Image Resolution)")
    print(f"Internal Resolution    : 768px")
    print(f"Model Load Time        : {load_duration:.2f}s")
    print(f"Pure Inference Time    : {infer_duration:.2f}s ({infer_duration/num_frames:.3f}s/frame, {num_frames/infer_duration:.2f} FPS)")
    print(f"Model Unload Time      : {unload_duration:.2f}s")
    print(f"TOTAL END-TO-END TIME  : {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
    print(f"GPU Lingering Memory   : {vram_final - vram_start:.2f} MB")
    print("=" * 65)


if __name__ == "__main__":
    main()
