#!/usr/bin/env python3
"""Fixed-seed Pixal3D camera-distance ablation at 1024 resolution."""
import contextlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "radiator_camera_distance"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["PIXAL_TIMING"] = "1"
sys.path.insert(0, str(ROOT / "Pixal3D"))
import inference

original_camera = inference.get_camera_params_wild_moge
images = [ROOT / "images" / "radiator.jpg"]
records = []


class LogTee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.log.write(value)
        if any(tag in value for tag in ("[PIXAL_TIMING]", "[Metrics]", "[Done]", "[Inference] Using pipeline_type")):
            self.stream.write(value)
            self.stream.flush()
        return len(value)

    def flush(self):
        self.log.flush()


for image in images:
    for factor in (0.9, 1.1):
        label = f"{image.stem}_distance_{factor:.1f}"
        output = OUT / f"pixal3d_{label}.glb"
        log_path = OUT / f"{label}.log"

        def scaled_camera(*args, _factor=factor, **kwargs):
            params = original_camera(*args, **kwargs)
            params["distance"] *= _factor
            print(f"[CAMERA_ABLATION] distance_scale={_factor:.2f}, params={params}", flush=True)
            return params

        inference.get_camera_params_wild_moge = scaled_camera
        started = time.perf_counter()
        print(f"[START] {label}", flush=True)
        with log_path.open("w") as log:
            tee = LogTee(sys.stdout, log)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                inference.run_inference(
                    image_path=str(image), output_path=str(output), seed=34,
                    ss_sampling_steps=12, shape_slat_sampling_steps=12,
                    tex_slat_sampling_steps=12, max_num_tokens=49152,
                    low_vram=True, resolution=1024,
                    decimation_target=150000, texture_size=1024,
                )
        elapsed = time.perf_counter() - started
        text = log_path.read_text(errors="replace")
        camera = re.search(r"\[CAMERA_ABLATION\] distance_scale=([0-9.]+), params=([^\n]+)", text)
        metrics = re.search(r"\[Metrics\] total=([0-9.]+)s pipeline=([0-9.]+)s glb=([0-9.]+)s peak_vram=([0-9.]+)GiB", text)
        record = {"image": image.name, "seed": 34, "distance_scale": factor,
                  "output": str(output), "log": str(log_path), "wall_seconds": elapsed,
                  "camera_params": camera.group(2) if camera else None,
                  "glb_bytes": output.stat().st_size if output.exists() else None}
        if metrics:
            record.update(dict(zip(("total_seconds", "pipeline_seconds", "glb_export_seconds", "peak_vram_gib"), map(float, metrics.groups()))))
        records.append(record)
        (OUT / "metrics.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"[COMPLETE] {label}: {elapsed:.1f}s", flush=True)

inference.get_camera_params_wild_moge = original_camera
print(f"[DONE] {len(records)} camera variants saved in {OUT}", flush=True)
