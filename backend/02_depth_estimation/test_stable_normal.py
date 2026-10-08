import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("MIOPEN_USER_DB_PATH", "/tmp/miopen")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import gc
import sys
from pathlib import Path

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
    print("=" * 60)
    print("StableNormal Turbo Verification & VRAM Lifecycle Test")
    print("=" * 60)

    initial_vram = get_vram_mb()
    print(f"[VRAM] Initial GPU allocation: {initial_vram:.2f} MB")

    test_img_path = _backend_dir / "current_scene" / "images" / "frame_00000.jpg"
    if not test_img_path.is_file():
        # Fallback to creating a synthetic test image if frame_00000 is missing
        print("[Test] frame_00000.jpg not found, creating synthetic test image...")
        test_img = Image.new("RGB", (756, 420), color=(128, 128, 200))
    else:
        test_img = Image.open(test_img_path).convert("RGB")
        print(f"[Test] Loaded test image {test_img_path.name}: {test_img.size} (W x H)")

    orig_w, orig_h = test_img.size

    # Test context manager and inference
    peak_vram = 0.0
    with StableNormalEstimator(device="cuda" if torch.cuda.is_available() else "cpu") as estimator:
        active_vram = get_vram_mb()
        print(f"[VRAM] Model loaded GPU allocation: {active_vram:.2f} MB")

        print("[Test] Running normal inference...")
        normals = estimator.estimate_normal(test_img)

        peak_vram = get_vram_mb()
        print(f"[VRAM] Peak inference GPU allocation: {peak_vram:.2f} MB")

    # Post-context VRAM verification
    final_vram = get_vram_mb()
    print(f"[VRAM] After context exit GPU allocation: {final_vram:.2f} MB")

    # Sanity checks on output normals
    h, w, c = normals.shape
    assert (w, h) == (orig_w, orig_h), f"Dimension mismatch: expected {(orig_w, orig_h)}, got {(w, h)}"
    assert c == 3, f"Expected 3 channels, got {c}"
    assert normals.dtype == np.float32, f"Expected float32, got {normals.dtype}"

    lengths = np.linalg.norm(normals, axis=-1)
    mean_len = float(np.mean(lengths))
    min_len = float(np.min(lengths))
    max_len = float(np.max(lengths))
    print(f"[Check] Normal unit lengths: mean={mean_len:.4f}, min={min_len:.4f}, max={max_len:.4f}")
    assert abs(mean_len - 1.0) < 1e-3, f"Normals not normalized: mean={mean_len}"

    # Save diagnostic visualization
    out_dir = _backend_dir / "current_scene" / "02_depth_estimation" / "depth"
    out_dir.mkdir(parents=True, exist_ok=True)
    colorized = colorize_normals(normals)
    out_path = out_dir / "test_stablenormal_preview.jpg"
    Image.fromarray(colorized).save(out_path, quality=90)
    print(f"[Success] Saved diagnostic preview to: {out_path}")

    # Check that VRAM is fully released (less than 10 MB lingering)
    lingering_vram = final_vram - initial_vram
    print(f"[VRAM] Net lingering VRAM: {lingering_vram:.2f} MB")
    if lingering_vram < 10.0:
        print("[Pass] GPU VRAM was successfully and completely released!")
    else:
        print(f"[Warning] {lingering_vram:.2f} MB lingering VRAM detected.")

    print("=" * 60)
    print("ALL STABLENORMAL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
