import sys, time, cv2, torch, numpy as np
from pathlib import Path
sys.path.insert(0, 'backend/06c_semantic_segmentation')
from fast_sam2_generator import FastSAM2Generator

device = 'cuda'
img_path = Path('backend/current_scene/06c_semantic_segmentation/inputs/images/frame_00000.jpg')
bgr = cv2.imread(str(img_path))
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
H, W = rgb.shape[:2]
total_pixels = H * W
min_pixels = int(0.005 * total_pixels)

sam_gen = FastSAM2Generator('small', grid=12, stability_thresh=0.80, compile_model=True, resize_mode='letterbox', nms_iou=0.85)
_ = sam_gen.generate(rgb, return_logits=False)
torch.cuda.synchronize()

t_sam0 = time.perf_counter()
masks, metadata = sam_gen.generate(rgb, return_logits=False)
torch.cuda.synchronize()
t_sam = time.perf_counter() - t_sam0

print(f"SAM2 (12x12) time: {t_sam*1000:.1f} ms | Raw masks count: {len(masks)}")

# Test new streamlined refinement
t0 = time.perf_counter()

# 1. Filter by IoU >= 0.50 and area >= 0.5%
surv_idx = [i for i, m in enumerate(metadata) if m.get('predicted_iou', 0.0) >= 0.50 and m.get('area_px', 0) >= min_pixels]
if not surv_idx:
    ref_masks = np.zeros((0, H, W), dtype=np.uint8)
    ref_meta = []
else:
    masks_filt = masks[surv_idx]
    meta_filt = [metadata[i] for i in surv_idx]

    # 2. Optimized Fast Filament Filter (at 4x downsample)
    non_fil = []
    for i, m in enumerate(masks_filt):
        pts = cv2.findNonZero(m)
        if pts is None: continue
        _, _, bw, bh = cv2.boundingRect(pts)
        span = max(bw, bh)
        if span < 150:
            non_fil.append(i)
            continue
        m_small = cv2.resize(m, (W // 4, H // 4), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(m_small, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        peri = sum(cv2.arcLength(c, True) for c in cnts) * 4
        area = int(m.sum())
        thickness = 2 * area / max(peri, 1)
        dist = cv2.distanceTransform(m_small, cv2.DIST_L2, 3)
        mean_dist = (dist[m_small > 0].mean() * 4) if m_small.any() else 0
        is_filament = (thickness < 35.0 and span >= 150) or (mean_dist < 15.0 and span >= 200)
        if not is_filament:
            non_fil.append(i)

    masks_clean = masks_filt[non_fil] if non_fil else np.zeros((0, H, W), dtype=np.uint8)
    meta_clean = [meta_filt[i] for i in non_fil]

    # 3. GPU Tensor Hierarchical Overlap Carving
    if len(masks_clean) > 0:
        masks_t = torch.from_numpy(masks_clean).to(device=device, dtype=torch.bool)
        conf_t = torch.tensor([m['predicted_iou'] for m in meta_clean], device=device)
        order = torch.argsort(conf_t, descending=True)
        masks_sorted = masks_t[order]
        meta_sorted = [meta_clean[idx] for idx in order.cpu().numpy()]

        occupied = torch.zeros((H, W), dtype=torch.bool, device=device)
        surv_masks_gpu = []
        surv_meta = []
        for k in range(len(masks_sorted)):
            m_k = masks_sorted[k] & ~occupied
            if m_k.sum() >= min_pixels:
                surv_masks_gpu.append(m_k)
                surv_meta.append(meta_sorted[k])
                occupied |= m_k

        if surv_masks_gpu:
            ref_masks = torch.stack(surv_masks_gpu).to(torch.uint8).cpu().numpy()
            ref_meta = surv_meta
            for mid, item in enumerate(ref_meta, start=1):
                item['mask_id'] = mid
                item['area_px'] = int(ref_masks[mid - 1].sum())
        else:
            ref_masks = np.zeros((0, H, W), dtype=np.uint8)
            ref_meta = []
    else:
        ref_masks = np.zeros((0, H, W), dtype=np.uint8)
        ref_meta = []

torch.cuda.synchronize()
t_ref = time.perf_counter() - t0
print(f"STREAMLINED REFINEMENT TIME: {t_ref*1000:.1f} ms! (Surviving masks: {len(ref_meta)})")
