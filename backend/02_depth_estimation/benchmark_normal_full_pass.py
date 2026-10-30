"""Time the production normal pass; write maps only to an explicit empty directory."""
from __future__ import annotations

import time

_PROCESS_START = time.perf_counter()

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from stable_normal import StableNormalEstimator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 processes all images")
    args = parser.parse_args()
    paths = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        parser.error("no images found")
    if len({p.stem for p in paths}) != len(paths):
        parser.error("image stems must be unique")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be empty (existing maps are never overwritten)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Match the setting inherited from DA3 in step_depth.py.
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    estimator = StableNormalEstimator()
    timings = {}
    held_maps = []  # Production retains float32 normals for depth cleanup.
    start = time.perf_counter()
    try:
        estimator.load()
        sync()
        timings["load_s"] = time.perf_counter() - start
        start = time.perf_counter()
        pending = deque()
        with ThreadPoolExecutor(max_workers=4) as pool:
            for idx, path in enumerate(paths):
                with Image.open(path) as image:
                    rgb = np.asarray(image.convert("RGB"))
                torch.manual_seed(42 + idx)
                normals = estimator.estimate_normal(rgb)
                assert normals.shape == rgb.shape and normals.dtype == np.float32
                assert np.isfinite(normals).all()
                held_maps.append(normals)
                pending.append(pool.submit(np.save, args.output_dir / f"{path.stem}.npy", normals.astype(np.float16)))
                if len(pending) >= 4:
                    pending.popleft().result()
                if (idx + 1) % 50 == 0 or idx == 0:
                    print(f"{idx + 1}/{len(paths)} frames, {time.perf_counter() - start:.2f}s", flush=True)
            for future in pending:
                future.result()
        sync()
        timings["inference_and_save_s"] = time.perf_counter() - start
    finally:
        start = time.perf_counter()
        estimator.unload()
        sync()
        timings["unload_s"] = time.perf_counter() - start
    timings["total_s"] = sum(timings.values())
    timings["process_s_including_imports"] = time.perf_counter() - _PROCESS_START
    report = dict(frames=len(paths), resolution=768, timings=timings,
                  torch=torch.__version__, hip=torch.version.hip,
                  device=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                  peak_vram_mb=torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0,
                  final_vram_mb=torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else 0)
    (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
