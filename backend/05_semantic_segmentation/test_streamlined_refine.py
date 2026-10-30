import sys, time, cv2, torch, numpy as np
import pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fast_sam2_generator import FastSAM2Generator


def test_streamlined_refine():
    img_path = Path("backend/current_scene/scan_0/images/frame_00000.jpg")
    if not img_path.is_file():
        img_path = Path("backend/current_scene/05_semantic_segmentation/inputs/images/frame_00000.jpg")
    if not img_path.is_file() or not torch.cuda.is_available():
        pytest.skip("Benchmark image or CUDA device unavailable")

    device = "cuda"
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        pytest.skip("Failed to read image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    total_pixels = H * W
    min_pixels = int(0.005 * total_pixels)

    sam_gen = FastSAM2Generator("small", grid=12, stability_thresh=0.80, compile_model=False, resize_mode="letterbox", nms_iou=0.85)
    masks, metadata = sam_gen.generate(rgb, return_logits=False)
    assert isinstance(masks, np.ndarray)


if __name__ == "__main__":
    test_streamlined_refine()
