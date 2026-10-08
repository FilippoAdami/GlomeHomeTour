import time, cv2, torch, numpy as np
from pathlib import Path
from fast_sam2_generator import FastSAM2Generator

device = 'cuda'
img_path = Path('backend/current_scene/06c_semantic_segmentation/inputs/images/frame_00000.jpg')
bgr = cv2.imread(str(img_path))
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
H, W = rgb.shape[:2]
total_pixels = H * W

sam_gen = FastSAM2Generator('small', grid=8, stability_thresh=0.80, compile_model=True, resize_mode='letterbox', nms_iou=0.85)
_ = sam_gen.generate_complete(rgb, return_logits=True)
torch.cuda.synchronize()

masks, metadata, residual, logits = sam_gen.generate_complete(rgb, return_logits=True)
torch.cuda.synchronize()

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

# Fully timed fast refinement
t0 = time.perf_counter()

# Step 1: Pre-prune low IoU and small area
min_pixels = int(0.005 * total_pixels)
valid_idx = [i for i, m in enumerate(metadata) if m.get('predicted_iou', 0.0) >= 0.50 and m.get('area_px', 0) >= min_pixels]

if valid_idx and logits is not None:
    guide_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
    src_t = torch.from_numpy(logits[valid_idx]).unsqueeze(1).to(device=device, dtype=torch.float32)
    q_t = guided_filter_gpu(guide_t, src_t)
    masks_curr = (q_t[:, 0] > 0).to(torch.uint8).cpu().numpy()
    meta_curr = [metadata[i] for i in valid_idx]
else:
    masks_curr = np.zeros((0, H, W), dtype=np.uint8)
    meta_curr = []

torch.cuda.synchronize()
t_gf = time.perf_counter() - t0

t1 = time.perf_counter()
# Step 2: Filament & Contour filter
non_fil = []
for i, m in enumerate(masks_curr):
    pts = cv2.findNonZero(m)
    if pts is None: continue
    _, _, bw, bh = cv2.boundingRect(pts)
    span = max(bw, bh)
    if span < 150:
        non_fil.append(i)
        continue
    # Compute on downsampled 4x for speed
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

masks_curr = masks_curr[non_fil] if non_fil else np.zeros((0, H, W), dtype=np.uint8)
meta_curr = [meta_curr[i] for i in non_fil]
t_fil = time.perf_counter() - t1

t2 = time.perf_counter()
# Step 3: Overlap Subtraction
items = [{'mask': m.copy(), 'info': info} for m, info in zip(masks_curr, meta_curr)]
for i in range(len(items)):
    for j in range(len(items)):
        if i == j: continue
        mi, mj = items[i]['mask'], items[j]['mask']
        overlap = (mi & mj)
        if not overlap.any(): continue
        overlap_area = overlap.sum()
        ci, cj = items[i]['info']['predicted_iou'], items[j]['info']['predicted_iou']
        diff = (max(ci, cj) - min(ci, cj)) / max(min(ci, cj), 1e-6)
        if diff >= 0.33:
            if ci < cj: mi[mj > 0] = 0
            else: mj[mi > 0] = 0
        else:
            area_i = mi.sum()
            area_j = mj.sum()
            if area_i > 0 and area_j > 0:
                if overlap_area / area_i >= 0.80 and area_i < area_j: mi[overlap > 0] = 0
                elif overlap_area / area_j >= 0.80 and area_j < area_i: mj[overlap > 0] = 0
                else:
                    if ci >= cj: mj[mi > 0] = 0
                    else: mi[mj > 0] = 0
t_overlap = time.perf_counter() - t2

t3 = time.perf_counter()
# Step 4: SOR Floater Cleanup
surviving = []
for item in items:
    m = item['mask']
    if m.sum() == 0: continue
    nb_components, output_cc, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    new_m = np.zeros_like(m)
    total_m_area = m.sum()
    sor_thresh = max(1000, int(0.05 * total_m_area))
    for comp_id in range(1, nb_components):
        if stats[comp_id, cv2.CC_STAT_AREA] >= sor_thresh:
            new_m[output_cc == comp_id] = 1
    if new_m.sum() >= min_pixels:
        item['mask'] = new_m
        surviving.append(item)

t_sor = time.perf_counter() - t3
t_total = time.perf_counter() - t0

print(f"FAST REFINEMENT BENCHMARK:")
print(f"  GPU Pre-filter + Guided Filter: {t_gf*1000:.1f} ms")
print(f"  Downsampled Filament filter:    {t_fil*1000:.1f} ms")
print(f"  Hierarchical Overlap Carving:   {t_overlap*1000:.1f} ms")
print(f"  SOR Connected Components:       {t_sor*1000:.1f} ms")
print(f"  --------------------------------------------------")
print(f"  TOTAL REFINEMENT TIME:          {t_total*1000:.1f} ms (was 4378.8 ms!)")
print(f"  Surviving masks count:          {len(surviving)}")
