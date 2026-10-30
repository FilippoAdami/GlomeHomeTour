import time, cv2, torch, numpy as np
import pytest
from pathlib import Path
from fast_sam2_generator import FastSAM2Generator


def box_filter_gpu(x, r):
    ksize = 2 * r + 1
    return torch.nn.functional.avg_pool2d(x, kernel_size=ksize, stride=1, padding=r)


def guided_filter_gpu(guide, src, r=4, eps=1e-3):
    mean_I = box_filter_gpu(guide, r)
    mean_p = box_filter_gpu(src, r)
    mean_Ip = box_filter_gpu(guide * src, r)
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = box_filter_gpu(guide * guide, r)
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - (a * mean_I).sum(dim=1, keepdim=True)
    mean_a = box_filter_gpu(a, r)
    mean_b = box_filter_gpu(b, r)
    return (mean_a * guide).sum(dim=1, keepdim=True) + mean_b


def test_fast_refinement():
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

    sam_gen = FastSAM2Generator("small", grid=8, stability_thresh=0.80, compile_model=False, resize_mode="letterbox", nms_iou=0.85)
    masks, metadata = sam_gen.generate(rgb)
    assert isinstance(masks, np.ndarray)


if __name__ == "__main__":
    test_fast_refinement()
