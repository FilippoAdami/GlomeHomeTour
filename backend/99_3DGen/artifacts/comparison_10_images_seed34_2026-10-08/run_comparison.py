#!/usr/bin/env python3
"""Run Pixal3D and TRELLIS.2 on every image in ../images with seed 34."""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "images"
PYTHON = ROOT / ".venv/bin/python"
PIXAL_DIR = ROOT / "Pixal3D"
IMAGES = sorted(p for p in IMAGE_DIR.iterdir() if p.is_file())
if len(IMAGES) != 10:
    raise SystemExit(f"Expected 10 image files in {IMAGE_DIR}, found {len(IMAGES)}")

common = {"seed": 34, "resolution": 1024, "steps": 12,
          "max_num_tokens": 49152, "decimation_target": 150000,
          "texture_size": 1024}
manifest = {"image_dir": str(IMAGE_DIR), "images": [p.name for p in IMAGES],
            "seed": 34, "configuration": common,
            "pixal3d_low_vram": True, "trellis2_pipeline_type": "1024_cascade",
            "runs": [], "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

for model in ("pixal3d", "trellis2"):
    for i, image in enumerate(IMAGES, 1):
        stem = image.stem
        glb = OUT / f"{model}_{stem}.glb"
        log = OUT / f"{model}_{stem}.log"
        metric = OUT / f"{model}_{stem}.json"
        if model == "pixal3d":
            cmd = [str(PYTHON), "inference.py", "--image", str(image),
                   "--output", str(glb), "--seed", "34", "--resolution", "1024",
                   "--steps", "12", "--low_vram", "--max_num_tokens", "49152",
                   "--decimation_target", "150000", "--texture_size", "1024"]
            env = dict(os.environ, PIXAL_TIMING="1")
        else:
            cmd = [str(PYTHON), "inference_trellis2.py", "--image", str(image),
                   "--output", str(glb), "--seed", "34", "--max-num-tokens", "49152",
                   "--decimation-target", "150000", "--texture-size", "1024"]
            env = os.environ.copy()
        started = time.perf_counter()
        print(f"[{model} {i}/10] {image.name}: started", flush=True)
        timings = {}
        lines = []
        with log.open("w") as f:
            p = subprocess.Popen(cmd, cwd=PIXAL_DIR, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert p.stdout is not None
            for line in p.stdout:
                f.write(line)
                f.flush()
                lines.append(line)
                if "_TIMING]" in line or "[Metrics]" in line or "[Done]" in line:
                    print(line.rstrip(), flush=True)
            code = p.wait()
        elapsed = time.perf_counter() - started
        run = {"model": model, "image": image.name, "glb": str(glb),
               "log": str(log), "exit_code": code, "wall_seconds": elapsed,
               "glb_bytes": glb.stat().st_size if glb.exists() else None}
        if model == "pixal3d":
            for line in lines:
                match = re.search(r"\[PIXAL_TIMING\] ([^=]+)=([0-9.]+)s", line)
                if match:
                    timings[match.group(1)] = float(match.group(2))
            run["steps_seconds"] = timings
            run["config"] = {**common, "low_vram": True,
                             "pipeline_type": "1024_cascade"}
            if glb.exists():
                metric.write_text(json.dumps(run, indent=2) + "\n")
        else:
            trellis_metric = glb.with_suffix(".json")
            if trellis_metric.exists():
                data = json.loads(trellis_metric.read_text())
                run.update(data)
                run["checkpoint"] = run["model"]
                run["model"] = "trellis2"
                run["wall_seconds"] = elapsed
            run["config"] = {**common, "low_vram": True,
                             "pipeline_type": "1024_cascade"}
        manifest["runs"].append(run)
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if code != 0 or not glb.exists():
            print(f"[{model} {i}/10] FAILED (exit={code}); continuing", flush=True)
        else:
            print(f"[{model} {i}/10] complete in {elapsed:.1f}s; {glb.stat().st_size:,} bytes", flush=True)

manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
ok = sum(bool(r["exit_code"] == 0 and r["glb_bytes"]) for r in manifest["runs"])
print(f"Completed {ok}/20 GLB exports. Manifest: {OUT / 'manifest.json'}", flush=True)
sys.exit(0 if ok == 20 else 1)
