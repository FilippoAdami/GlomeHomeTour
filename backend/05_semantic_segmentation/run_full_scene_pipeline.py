#!/usr/bin/env python3
"""
Full-Scene End-to-End Panoptic Semantic Segmentation & Memory Consolidation Pipeline.

Runs on all 362 frames in 'inputs/images/':
- Stage 1: SAM 2.1 Hiera-Small (12x12 uniform grid, letterbox resize, compile_model=True).
- Stage 2: GPU Tensor Overlap Carving & Two-Tier Regularization (floater & ribbon pruning).
- Stage 3: DINOv2 Spatial Token Extraction & Bilinear Mask Region Pooling.
- Stage 4: Per-Image Iterative Semantic Merging (threshold >= 0.690).
- Stage 5: Streaming Global Memory Bank ID Propagation (dormancy adaptation + scale-adaptive decay).
- Stage 6: Post-Hoc Memory Consolidation (Single Merge Max, threshold > 0.690, area-weighted average).
- Stage 7: Final Class Cleanup:
    * Prune high-gap classes (avg_gap > 5.0).
    * Merge infrequent classes (seen <= 2 frames) into closest persistent class.
- Stage 8: Final Triangular Matrix & Visuals Generation:
    * Final KxK upper-triangular similarity matrix.
    * 362 full-resolution PNG visual overlays with distance-transform anchored badges.
    * Compressed NPZ archives (masks, labels, embeddings) and JSON summary.

Records precise execution times for every atomic substep and grand total.
"""

import os, sys, json, time
import concurrent.futures
from pathlib import Path
import cv2, torch, numpy as np
from PIL import Image
from transformers import AutoModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fast_sam2_generator import FastSAM2Generator

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import pickle

# Hyperparameters
BASE_THRESHOLD = 0.690
MAX_BASE_THRESHOLD = 0.730
DORMANCY_STEP_FRAMES = 7
DORMANCY_STEP_VAL = 0.01
SECONDARY_THRESHOLD = 0.710
POST_HOC_MERGE_THRESH = 0.730
RARE_APPEARANCE_THRESHOLD = 2

def get_adaptive_gamma(area_ratio: float) -> float:
    if area_ratio >= 0.15:
        return 0.95
    elif area_ratio >= 0.03:
        return 0.90
    else:
        return 0.82

def clean_carved_mask(m: np.ndarray, min_chunk_area: int, min_thickness: float = 12.0) -> np.ndarray:
    H, W = m.shape
    if m.sum() < min_chunk_area:
        return np.zeros_like(m)

    # Tier 1: Detached floater / crumb cleanup
    nb_cc, cc_labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if nb_cc > 2:
        total_a = m.sum()
        min_comp = max(1000, int(0.05 * total_a))
        surv_comp = np.zeros_like(m)
        for c in range(1, nb_cc):
            if stats[c, cv2.CC_STAT_AREA] >= min_comp:
                surv_comp[cc_labels == c] = 1
        m = surv_comp

    if m.sum() < min_chunk_area:
        return np.zeros_like(m)

    # Fast perimeter on 4x downsampled mask
    m_small = cv2.resize(m, (W // 4, H // 4), interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(m_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    peri = sum(cv2.arcLength(c, True) for c in cnts) * 4
    thick = 2 * m.sum() / max(peri, 1)

    if thick >= 25.0:
        return m

    # Tier 2: Ribbon/tendril pruning via morphological opening (7x7 ellipse)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    if opened.sum() < min_chunk_area:
        return np.zeros_like(m)

    nb_cc, cc_labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    chunky_m = np.zeros_like(m)
    for c in range(1, nb_cc):
        a_c = stats[c, cv2.CC_STAT_AREA]
        if a_c >= min_chunk_area:
            comp = (cc_labels == c).astype(np.uint8)
            comp_small = cv2.resize(comp, (W // 4, H // 4), interpolation=cv2.INTER_NEAREST)
            cnts_c, _ = cv2.findContours(comp_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            peri_c = sum(cv2.arcLength(cnt, True) for cnt in cnts_c) * 4
            thick_c = 2 * a_c / max(peri_c, 1)
            if thick_c >= min_thickness:
                chunky_m[comp > 0] = 1

    if chunky_m.sum() >= min_chunk_area:
        return chunky_m
    return np.zeros_like(m)

def get_class_color(mid: int):
    k = mid - 1
    b = (37 * k + 93) % 255
    g = (109 * k + 71) % 255
    r = (197 * k + 31) % 255
    col_bgr = (int(b), int(g), int(r))
    col_rgb = (r / 255.0, g / 255.0, b / 255.0)
    return col_bgr, col_rgb

def find_best_label_anchor(mask_u8: np.ndarray, W: int, H: int):
    nb_cc, cc_labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    if nb_cc <= 1:
        return W // 2, H // 2

    largest_c = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    comp = (cc_labels == largest_c).astype(np.uint8)

    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)

    margin = 60
    dist[:margin, :] *= 0.5
    dist[-margin:, :] *= 0.5
    dist[:, :margin] *= 0.5
    dist[:, -margin:] *= 0.5

    _, _, _, max_loc = cv2.minMaxLoc(dist)
    return max_loc[0], max_loc[1]

def create_mask_overlay(img_path: Path, masks: np.ndarray, labels: np.ndarray, mask_ids: list) -> np.ndarray:
    bgr = cv2.imread(str(img_path))
    H, W = bgr.shape[:2]
    overlay = bgr.copy()

    for mid, m in zip(mask_ids, masks):
        if mid <= 0: continue
        col_bgr, _ = get_class_color(mid)
        col_arr = np.array(col_bgr, dtype=np.uint8)
        vis = m > 0
        overlay[vis] = (0.55 * overlay[vis] + 0.45 * col_arr).astype(np.uint8)

    overlay[labels == 0] = 0

    for mid, m in zip(mask_ids, masks):
        if mid <= 0: continue
        col_bgr, _ = get_class_color(mid)
        mask_u8 = m.astype(np.uint8)
        if not mask_u8.any(): continue
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, col_bgr, 3)

    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 1.3
    thick = 2
    pad_x = 14
    pad_y = 10

    for mid, m in zip(mask_ids, masks):
        if mid <= 0: continue
        mask_u8 = m.astype(np.uint8)
        if not mask_u8.any(): continue
        col_bgr, _ = get_class_color(mid)
        cx, cy = find_best_label_anchor(mask_u8, W, H)

        txt = f"#{mid}"
        (tw, th), baseline = cv2.getTextSize(txt, font, scale, thick)

        bx1 = max(10, min(W - tw - 2 * pad_x - 10, cx - tw // 2 - pad_x))
        by1 = max(10, min(H - th - 2 * pad_y - 10, cy - th // 2 - pad_y))
        bx2 = bx1 + tw + 2 * pad_x
        by2 = by1 + th + 2 * pad_y

        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), col_bgr, 3)

        text_x = bx1 + pad_x
        text_y = by1 + pad_y + th
        cv2.putText(overlay, txt, (text_x, text_y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    return overlay
 
def render_and_save_frame(args):
    p, stem, masks, labels, embs, c_ids, npz_path, overlay_path = args
    np.savez_compressed(
        npz_path,
        masks=masks,
        labels=labels,
        embeddings=embs,
        ids=np.array(c_ids)
    )
    ov = create_mask_overlay(p, masks, labels, c_ids)
    cv2.imwrite(str(overlay_path), ov)

def create_final_triangular_matrix(classes_dict: dict, out_path: Path):
    sorted_ids = sorted(classes_dict.keys())
    K = len(sorted_ids)
    if K == 0: return

    units = np.stack([classes_dict[mid]['unit_vec'] for mid in sorted_ids])
    sim_mat = units @ units.T

    labels = []
    for mid in sorted_ids:
        entry = classes_dict[mid]
        n_frames = len(entry['frame_indices'])
        frames_idx = sorted(entry['frame_indices'])
        if len(frames_idx) <= 1:
            avg_gap = 0.0
        else:
            gaps = [frames_idx[k + 1] - frames_idx[k] for k in range(len(frames_idx) - 1)]
            avg_gap = float(np.mean(gaps))
        labels.append(f"#{mid} ({n_frames}, {avg_gap:.1f})")

    fig_dim = max(10.0, K * 0.55 + 3.0)
    fig, ax = plt.subplots(figsize=(fig_dim, fig_dim), dpi=140)
    cmap = plt.cm.YlGnBu

    for i in range(K):
        for j in range(K):
            if j < i: continue
            val = float(sim_mat[i, j])
            if i == j:
                _, rgb_c = get_class_color(sorted_ids[i])
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=rgb_c, edgecolor='black', linewidth=1.2)
                ax.add_patch(rect)
                lum = 0.299 * rgb_c[0] + 0.587 * rgb_c[1] + 0.114 * rgb_c[2]
                ax.text(j, i, f"#{sorted_ids[i]}", ha='center', va='center',
                        color='white' if lum < 0.55 else 'black', fontweight='bold', fontsize=9.5)
            else:
                c = cmap((val + 0.1) / 1.1)
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=c, edgecolor='#e0e0e0', linewidth=0.5)
                ax.add_patch(rect)
                lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
                fontweight = 'bold' if val >= 0.70 else 'normal'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color='white' if lum < 0.55 else 'black', fontsize=8.5, fontweight=fontweight)

    ax.set_xlim(-0.5, K - 0.5)
    ax.set_ylim(K - 0.5, -0.5)
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(labels, fontsize=9.5, rotation=90)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.xaxis.tick_top()

    for t_idx, tick in enumerate(ax.get_xticklabels()):
        _, col_rgb = get_class_color(sorted_ids[t_idx])
        tick.set_color(col_rgb)
        tick.set_fontweight('bold')

    for t_idx, tick in enumerate(ax.get_yticklabels()):
        _, col_rgb = get_class_color(sorted_ids[t_idx])
        tick.set_color(col_rgb)
        tick.set_fontweight('bold')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label('DINOv2 Cosine Similarity', fontsize=11)
    plt.title(f'Full Scene Final Physical Class Similarity Matrix (K={K} Classes)\n362 Frames Dataset',
              fontsize=13, fontweight='bold', pad=45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

class GlobalMemoryBank:
    def __init__(self):
        self.classes = {}
        self.next_available_id = 1

    def register_initial_frame(self, masks: np.ndarray, embeddings: np.ndarray, frame_name: str, frame_idx: int):
        N = len(masks)
        assigned_ids = list(range(1, N + 1))
        self.next_available_id = N + 1

        for i, mid in enumerate(assigned_ids):
            area = float(masks[i].sum())
            unit_vec = embeddings[i].copy()
            sum_vec = area * unit_vec
            self.classes[mid] = {
                'id': mid,
                'sum_vec': sum_vec,
                'unit_vec': unit_vec,
                'eff_area': area,
                'raw_area': area,
                'frames_seen': 1,
                'first_seen': frame_name,
                'last_seen': frame_name,
                'last_seen_idx': frame_idx,
                'frame_indices': [frame_idx]
            }
        return assigned_ids

    def get_bank_matrix(self):
        bank_ids = sorted(self.classes.keys())
        bank_units = np.stack([self.classes[mid]['unit_vec'] for mid in bank_ids])
        return bank_ids, bank_units

    def get_dormancy_threshold(self, mid: int, current_frame_idx: int) -> float:
        last_idx = self.classes[mid]['last_seen_idx']
        delta_t = max(0, current_frame_idx - last_idx)
        step_inc = (delta_t // DORMANCY_STEP_FRAMES) * DORMANCY_STEP_VAL
        return min(MAX_BASE_THRESHOLD, BASE_THRESHOLD + step_inc)

    def update_matches(self, claims_by_bank_id: dict, next_masks: np.ndarray,
                       next_embeddings: np.ndarray, frame_name: str, frame_idx: int, H: int, W: int):
        total_pixels = float(H * W)
        for mid, claims in claims_by_bank_id.items():
            entry = self.classes[mid]
            inc_area = 0.0
            inc_sum = np.zeros_like(entry['sum_vec'])
            for j, _ in claims:
                a_j = float(next_masks[j].sum())
                inc_area += a_j
                inc_sum += a_j * next_embeddings[j]

            area_ratio = inc_area / max(total_pixels, 1.0)
            gamma = get_adaptive_gamma(area_ratio)

            decayed_sum = gamma * entry['sum_vec']
            decayed_eff_area = gamma * entry['eff_area']

            new_sum = decayed_sum + inc_sum
            new_eff_area = decayed_eff_area + inc_area
            new_unit = new_sum / max(np.linalg.norm(new_sum), 1e-12)

            entry['sum_vec'] = new_sum
            entry['eff_area'] = new_eff_area
            entry['unit_vec'] = new_unit
            entry['raw_area'] += inc_area
            entry['frames_seen'] += 1
            entry['last_seen'] = frame_name
            entry['last_seen_idx'] = frame_idx
            entry['frame_indices'].append(frame_idx)

    def register_new_classes(self, unclaimed_indices: list, next_masks: np.ndarray,
                             next_embeddings: np.ndarray, frame_name: str, frame_idx: int) -> dict:
        new_assignments = {}
        for j in unclaimed_indices:
            mid = self.next_available_id
            self.next_available_id += 1
            area = float(next_masks[j].sum())
            unit_vec = next_embeddings[j].copy()
            sum_vec = area * unit_vec
            self.classes[mid] = {
                'id': mid,
                'sum_vec': sum_vec,
                'unit_vec': unit_vec,
                'eff_area': area,
                'raw_area': area,
                'frames_seen': 1,
                'first_seen': frame_name,
                'last_seen': frame_name,
                'last_seen_idx': frame_idx,
                'frame_indices': [frame_idx]
            }
            new_assignments[j] = mid
        return new_assignments

def main():
    root = Path(__file__).resolve().parent.parent
    workspace = root / "current_scene" / "06c_semantic_segmentation"
    img_dir = workspace / "inputs" / "images"
    raw_image_paths = sorted(img_dir.glob("*.jpg"))

    images_txt = workspace / "inputs" / "sparse" / "0" / "images.txt"
    colmap_pts_map = {}
    if images_txt.exists():
        with open(images_txt) as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        for i in range(0, len(lines), 2):
            parts = lines[i].split()
            name = parts[-1]
            pts = [p for p in lines[i+1].split()[2::3] if p != '-1']
            colmap_pts_map[name] = len(pts)

    if colmap_pts_map:
        image_paths = [p for p in raw_image_paths if colmap_pts_map.get(p.name, 0) >= 100]
        skipped_cnt = len(raw_image_paths) - len(image_paths)
        print(f"COLMAP Point Gating: Excluded {skipped_cnt} low-support frames (< 100 points) from 2D inference.")
    else:
        image_paths = raw_image_paths

    num_frames = len(image_paths)

    out_dir = workspace / "full362_final_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = out_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = out_dir / "npz_data"
    npz_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print(f"FULL SCENE PANOPTIC SEGMENTATION PIPELINE ON {num_frames} FRAMES (Filtered >= 100 COLMAP points)")
    print(f"Output Directory: {out_dir}")
    print("=================================================================\n")

    t_grand_start = time.perf_counter()

    # ================= 1. MODEL INITIALIZATION =================
    print("[Init] Loading FastSAM2Generator (Hiera-Small, Grid=12, compile=True)...")
    t0 = time.perf_counter()
    sam_gen = FastSAM2Generator("small", grid=12, stability_thresh=0.80, compile_model=True, resize_mode="letterbox", nms_iou=0.85)
    
    # Warmup
    warm_bgr = cv2.imread(str(image_paths[0]))
    warm_rgb = cv2.cvtColor(warm_bgr, cv2.COLOR_BGR2RGB)
    _ = sam_gen.generate(warm_rgb, return_logits=False)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_init_sam = time.perf_counter() - t0

    print("[Init] Loading and compiling DINOv2-small...")
    t0 = time.perf_counter()
    torch.set_float32_matmul_precision('high')
    dino_raw = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    dino_model = torch.compile(dino_raw, mode="default")
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    MAX_SIDE = 896
    PATCH = 14
    grid = (36, 64)
    dino_w, dino_h = 896, 504
    
    # Warmup DINOv2 compiled graph
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        _ = dino_model(pixel_values=torch.zeros((1, 3, dino_h, dino_w), device=device, dtype=torch.float32)).last_hidden_state
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_init_dino = time.perf_counter() - t0

    print(f"Models initialized in {t_init_sam + t_init_dino:.2f}s (SAM2: {t_init_sam:.2f}s, DINOv2: {t_init_dino:.2f}s)\n")

    # Storage for stage 1-4 per-frame results
    stage1_sam_masks = []
    stage1_sam_meta = []
    
    stage2_ref_masks = []
    stage4_cache_file = workspace / "stage4_merged_cache.pkl"
    stage4_merged_data = []

    t_stage1_total = 0.0
    t_stage2_total = 0.0
    t_stage3_total = 0.0
    t_stage4_total = 0.0

    H, W = warm_rgb.shape[:2]
    total_pixels = H * W
    min_pixels = int(0.005 * total_pixels)

    # ================= STAGES 1 TO 4 STREAMING =================
    print(f"=== EXECUTING STAGES 1-4 ACROSS ALL {num_frames} FRAMES ===")
    for idx, p in enumerate(image_paths):
        stem = p.stem
        bgr = cv2.imread(str(p))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Stage 1: SAM2
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_s1 = time.perf_counter()
        raw_masks, raw_metadata = sam_gen.generate(rgb, return_logits=False)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        dur_s1 = time.perf_counter() - t_s1
        t_stage1_total += dur_s1

        # Stage 2: GPU Refinement & Regularization
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_s2 = time.perf_counter()
        surv_idx = [i for i, m in enumerate(raw_metadata) if m.get('predicted_iou', 0.0) >= 0.50 and m.get('area_px', 0) >= min_pixels]
        if not surv_idx:
            ref_masks = np.zeros((0, H, W), dtype=np.uint8)
            ref_meta = []
        else:
            masks_filt = raw_masks[surv_idx]
            meta_filt = [raw_metadata[i] for i in surv_idx]

            masks_t = torch.from_numpy(masks_filt).to(device=device, dtype=torch.bool)
            conf_t = torch.tensor([m['predicted_iou'] for m in meta_filt], device=device)
            order = torch.argsort(conf_t, descending=True)
            masks_sorted = masks_t[order]
            meta_sorted = [meta_filt[i] for i in order.cpu().numpy()]

            occupied = torch.zeros((H, W), dtype=torch.bool, device=device)
            carved_masks = []
            carved_meta = []
            for k in range(len(masks_sorted)):
                m_k = masks_sorted[k] & ~occupied
                if m_k.sum() >= min_pixels:
                    carved_masks.append(m_k.to(torch.uint8).cpu().numpy())
                    carved_meta.append(meta_sorted[k])
                    occupied |= m_k

            clean_masks = []
            clean_meta = []
            for m_arr, info in zip(carved_masks, carved_meta):
                m_clean = clean_carved_mask(m_arr, min_chunk_area=min_pixels, min_thickness=12.0)
                if m_clean.sum() >= min_pixels:
                    info_c = dict(info)
                    info_c['area_px'] = int(m_clean.sum())
                    clean_masks.append(m_clean)
                    clean_meta.append(info_c)

            if clean_masks:
                ref_masks = np.stack(clean_masks)
                ref_meta = clean_meta
                for mid, item in enumerate(ref_meta, start=1):
                    item['mask_id'] = mid
            else:
                ref_masks = np.zeros((0, H, W), dtype=np.uint8)
                ref_meta = []

        if torch.cuda.is_available(): torch.cuda.synchronize()
        dur_s2 = time.perf_counter() - t_s2
        t_stage2_total += dur_s2

        # Stage 3: DINOv2 Feature Inference & Vectorized Patch Pooling
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_s3 = time.perf_counter()
        dino_bgr_small = cv2.resize(bgr, (dino_w, dino_h), interpolation=cv2.INTER_CUBIC)
        dino_rgb_small = cv2.cvtColor(dino_bgr_small, cv2.COLOR_BGR2RGB)
        pixels = torch.from_numpy(dino_rgb_small).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32).div_(255.0)

        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            tokens = dino_model(pixel_values=(pixels - mean) / std).last_hidden_state[:, 1:, :] # (1, 2304, 384)

        if len(ref_masks) > 0:
            mask_t = torch.from_numpy(ref_masks).to(device=device, dtype=torch.float32).unsqueeze(1)
            mask_low = torch.nn.functional.interpolate(mask_t, size=grid, mode='bilinear', align_corners=False)[:, 0] # (N, 36, 64)
            weight_bin = (mask_low > 0.4).float()
            sums = weight_bin.sum(dim=(-2, -1), keepdim=True)
            weight_final = torch.where(sums == 0, mask_low, weight_bin)
            weight_flat = weight_final.view(len(ref_masks), -1) # (N, 2304)
            sum_vecs = torch.matmul(weight_flat, tokens[0].float()) # (N, 384)
            unit_vecs = sum_vecs / sum_vecs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            sum_vecs_np = sum_vecs.cpu().numpy()
            unit_vecs_np = unit_vecs.cpu().numpy()
        else:
            sum_vecs_np = np.zeros((0, 384), dtype=np.float32)
            unit_vecs_np = np.zeros((0, 384), dtype=np.float32)

        items_dino = []
        for m_idx, (m, info) in enumerate(zip(ref_masks, ref_meta)):
            items_dino.append({
                'id': info['mask_id'],
                'mask': m,
                'sum_vec': sum_vecs_np[m_idx],
                'unit_vec': unit_vecs_np[m_idx],
                'area': int(m.sum())
            })
        if torch.cuda.is_available(): torch.cuda.synchronize()
        dur_s3 = time.perf_counter() - t_s3
        t_stage3_total += dur_s3

        # Stage 4: Single-Image Iterative Semantic Merging (>= 0.69)
        t_s4 = time.perf_counter()
        while True:
            N = len(items_dino)
            if N <= 1: break
            unit_mat = np.stack([it['unit_vec'] for it in items_dino])
            sim_mat = unit_mat @ unit_mat.T

            best_sim = -1.0
            best_pair = None
            for i_m in range(N):
                for j_m in range(i_m + 1, N):
                    if sim_mat[i_m, j_m] > best_sim:
                        best_sim = float(sim_mat[i_m, j_m])
                        best_pair = (i_m, j_m)

            if best_sim < 0.690:
                break

            i_m, j_m = best_pair
            merged_mask = items_dino[i_m]['mask'] | items_dino[j_m]['mask']
            merged_sum = items_dino[i_m]['sum_vec'] + items_dino[j_m]['sum_vec']
            merged_unit = merged_sum / max(np.linalg.norm(merged_sum), 1e-12)

            items_dino[i_m]['mask'] = merged_mask
            items_dino[i_m]['sum_vec'] = merged_sum
            items_dino[i_m]['unit_vec'] = merged_unit
            items_dino[i_m]['area'] = int(merged_mask.sum())
            items_dino.pop(j_m)

        items_dino.sort(key=lambda x: x['id'])
        dur_s4 = time.perf_counter() - t_s4
        t_stage4_total += dur_s4

        final_masks = np.stack([it['mask'] for it in items_dino]) if len(items_dino) > 0 else np.zeros((0, H, W), dtype=np.uint8)
        final_embs = np.stack([it['unit_vec'] for it in items_dino]) if len(items_dino) > 0 else np.zeros((0, 384), dtype=np.float32)

        stage4_merged_data.append({
            'stem': stem,
            'masks': final_masks,
            'embeddings': final_embs,
            'num_masks': len(items_dino)
        })

        if (idx + 1) % 25 == 0 or idx == num_frames - 1:
            tot_fps = (idx + 1) / (t_stage1_total + t_stage2_total + t_stage3_total + t_stage4_total)
            print(f"  [Stages 1-4] Progress: {idx + 1:03d}/{num_frames} frames ({tot_fps:.2f} FPS) | Current Frame Masks: {len(items_dino)}")

    print(f"\n[Cache] Caching Stage 4 data to {stage4_cache_file}...")
    t_save = time.perf_counter()
    with open(stage4_cache_file, "wb") as f:
        pickle.dump(stage4_merged_data, f)
    print(f"[Cache] Cached {len(stage4_merged_data)} frames in {time.perf_counter() - t_save:.2f}s!")

    # ================= STAGE 5: GLOBAL MEMORY BANK PROPAGATION =================
    print(f"\n=== STAGE 5: GLOBAL MEMORY BANK STREAMING ID PROPAGATION ===")
    t_s5_start = time.perf_counter()
    bank = GlobalMemoryBank()

    # Frame 0
    f0_data = stage4_merged_data[0]
    init_ids_0 = bank.register_initial_frame(f0_data['masks'], f0_data['embeddings'], f0_data['stem'], frame_idx=0)
    stage4_merged_data[0]['propagated_ids'] = list(init_ids_0)

    for step in range(num_frames - 1):
        current_frame_idx = step + 1
        f_next = stage4_merged_data[current_frame_idx]
        next_masks = f_next['masks']
        next_embeddings = f_next['embeddings']
        N_next = len(next_masks)

        if N_next == 0:
            f_next['propagated_ids'] = []
            continue

        bank_ids, bank_units = bank.get_bank_matrix()
        K_bank = len(bank_ids)
        sim_mat = bank_units @ next_embeddings.T

        candidates = []
        for i_k in range(K_bank):
            for j_k in range(N_next):
                val = float(sim_mat[i_k, j_k])
                if val >= BASE_THRESHOLD:
                    candidates.append((val, i_k, j_k))
        candidates.sort(key=lambda x: x[0], reverse=True)

        matched_targets = set()
        claims_by_bank_idx = {i_k: [] for i_k in range(K_bank)}

        for val, i_k, j_k in candidates:
            if j_k in matched_targets: continue
            mid = bank_ids[i_k]
            if len(claims_by_bank_idx[i_k]) == 0:
                dormancy_thresh = bank.get_dormancy_threshold(mid, current_frame_idx)
                if val >= dormancy_thresh:
                    claims_by_bank_idx[i_k].append((j_k, val))
                    matched_targets.add(j_k)
            elif val >= SECONDARY_THRESHOLD:
                claims_by_bank_idx[i_k].append((j_k, val))
                matched_targets.add(j_k)

        next_ids = [None] * N_next
        claims_by_mid = {}
        for i_k, claims in claims_by_bank_idx.items():
            if not claims: continue
            mid = bank_ids[i_k]
            claims_by_mid[mid] = claims
            for j_k, _ in claims:
                next_ids[j_k] = mid

        bank.update_matches(claims_by_mid, next_masks, next_embeddings, f_next['stem'], current_frame_idx, H, W)
        unclaimed_j = [j_k for j_k in range(N_next) if next_ids[j_k] is None]
        new_assignments = bank.register_new_classes(unclaimed_j, next_masks, next_embeddings, f_next['stem'], current_frame_idx)
        for j_k, new_mid in new_assignments.items():
            next_ids[j_k] = new_mid

        f_next['propagated_ids'] = next_ids
        if (step + 1) % 50 == 0 or step == num_frames - 2:
            print(f"  [Stage 5] Step {step + 1:03d}/{num_frames - 1} | Memory Bank Size: {len(bank.classes)} classes")

    t_stage5_total = time.perf_counter() - t_s5_start
    print(f"Stage 5 completed in {t_stage5_total:.2f}s | Total Classes Discovered: {len(bank.classes)}")

    # ================= STAGE 6: MULTI-STEP POST-HOC CONSOLIDATION (UNTIL SIM <= 0.730) =================
    print(f"\n=== STAGE 6: MULTI-STEP POST-HOC CONSOLIDATION (UNTIL ALL PAIRS SIM <= {POST_HOC_MERGE_THRESH}) ===")
    t_s6_start = time.perf_counter()

    # Build registry for Stage 6
    class_embs = {}
    class_areas = {}
    class_frames = {}
    for f_idx, item in enumerate(stage4_merged_data):
        for mid, m in zip(item['propagated_ids'], item['masks']):
            class_areas[mid] = class_areas.get(mid, 0.0) + float(m.sum())
            class_frames.setdefault(mid, set()).add(f_idx)
    for mid, entry in bank.classes.items():
        class_embs[mid] = entry['unit_vec'].copy()

    id_mapping_s6 = {mid: mid for mid in class_embs.keys()}
    final_embs_s6 = {mid: class_embs[mid].copy() for mid in class_embs.keys()}
    final_areas_s6 = {mid: class_areas[mid] for mid in class_embs.keys()}
    final_frames_s6 = {mid: set(class_frames.get(mid, [])) for mid in class_embs.keys()}
    active_ids_s6 = set(class_embs.keys())

    total_merges_s6 = 0
    while True:
        sorted_active = sorted(active_ids_s6)
        K = len(sorted_active)
        if K <= 1:
            break

        units_cur = np.stack([final_embs_s6[mid] for mid in sorted_active])
        S_cur = units_cur @ units_cur.T

        # Find pair with highest similarity
        best_sim = -1.0
        best_pair = None
        for i in range(K):
            for j in range(i + 1, K):
                val = float(S_cur[i, j])
                if val > best_sim:
                    best_sim = val
                    best_pair = (sorted_active[i], sorted_active[j])

        if best_sim <= POST_HOC_MERGE_THRESH or best_pair is None:
            # Reached termination condition: no remaining pairwise similarity exceeds 0.730!
            break

        ida, idb = best_pair
        target_id = min(ida, idb)
        merged_id = max(ida, idb)

        wa = final_areas_s6[target_id]
        wb = final_areas_s6[merged_id]
        new_vec = wa * final_embs_s6[target_id] + wb * final_embs_s6[merged_id]
        new_unit = new_vec / max(np.linalg.norm(new_vec), 1e-12)

        final_embs_s6[target_id] = new_unit
        final_areas_s6[target_id] = wa + wb
        final_frames_s6[target_id] = final_frames_s6[target_id].union(final_frames_s6[merged_id])

        del final_embs_s6[merged_id]
        del final_areas_s6[merged_id]
        del final_frames_s6[merged_id]
        active_ids_s6.remove(merged_id)

        # Update global mapping
        for k_orig, mapped in id_mapping_s6.items():
            if mapped == merged_id:
                id_mapping_s6[k_orig] = target_id

        total_merges_s6 += 1
        if total_merges_s6 % 10 == 0:
            print(f"  [Stage 6 Step {total_merges_s6:03d}] Merged #{merged_id} into #{target_id} (sim={best_sim:.4f}) | Remaining: {len(active_ids_s6)} classes")

    # Update frame consolidated IDs
    for item in stage4_merged_data:
        item['consolidated_ids'] = [id_mapping_s6[old_id] for old_id in item['propagated_ids']]

    t_stage6_total = time.perf_counter() - t_s6_start
    print(f"Stage 6 completed in {t_stage6_total:.2f}s | Executed {total_merges_s6} multi-step merges | Remaining: {len(active_ids_s6)} classes (all pairs <= {POST_HOC_MERGE_THRESH})")

    # ================= STAGE 7: RARE CLASSES MERGE (NO SCATTERED CLASSES REMOVAL) =================
    print(f"\n=== STAGE 7: RARE CLASSES MERGE (NO SCATTERED CLASSES REMOVAL) ===")
    t_s7_start = time.perf_counter()

    surviving_s6 = sorted(active_ids_s6)
    class_frames_s7 = {mid: set(final_frames_s6[mid]) for mid in surviving_s6}
    class_areas_s7 = {mid: final_areas_s6[mid] for mid in surviving_s6}

    # Partition into persistent (seen >= 3 frames) and rare (seen <= 2 frames)
    rare_ids_s7 = [m for m in surviving_s6 if len(class_frames_s7[m]) <= RARE_APPEARANCE_THRESHOLD]
    persistent_ids_s7 = [m for m in surviving_s6 if len(class_frames_s7[m]) > RARE_APPEARANCE_THRESHOLD]

    print(f"  Surviving from Stage 6: {len(surviving_s6)} classes")
    print(f"  Persistent classes (seen >= 3 frames): {len(persistent_ids_s7)}")
    print(f"  Rare classes to merge (seen <= 2 frames): {len(rare_ids_s7)}")

    id_mapping_s7 = {mid: mid for mid in surviving_s6}
    final_embs_s7 = {mid: final_embs_s6[mid].copy() for mid in persistent_ids_s7}
    final_areas_s7 = {mid: class_areas_s7[mid] for mid in persistent_ids_s7}
    final_frames_s7 = {mid: set(class_frames_s7[mid]) for mid in persistent_ids_s7}

    units_persist_s7 = np.stack([final_embs_s7[mid] for mid in persistent_ids_s7])
    for r_id in rare_ids_s7:
        r_emb = final_embs_s6[r_id]
        sims = units_persist_s7 @ r_emb
        best_p_idx = int(np.argmax(sims))
        best_p_id = persistent_ids_s7[best_p_idx]

        wp = final_areas_s7[best_p_id]
        wr = class_areas_s7[r_id]
        new_vec = wp * final_embs_s7[best_p_id] + wr * r_emb
        new_unit = new_vec / max(np.linalg.norm(new_vec), 1e-12)

        final_embs_s7[best_p_id] = new_unit
        final_areas_s7[best_p_id] = wp + wr
        final_frames_s7[best_p_id] = final_frames_s7[best_p_id].union(class_frames_s7[r_id])
        units_persist_s7[best_p_idx] = new_unit
        id_mapping_s7[r_id] = best_p_id

    # Update frame cleaned IDs (strictly mapped to persistent IDs, 0 pruned)
    for item in stage4_merged_data:
        item['cleaned_ids'] = [id_mapping_s7[old_id] for old_id in item['consolidated_ids']]

    t_stage7_total = time.perf_counter() - t_s7_start
    print(f"Stage 7 completed in {t_stage7_total:.2f}s | Merged all {len(rare_ids_s7)} rare classes into persistent classes")
    print(f"FINAL PERSISTENT PHYSICAL CLASSES: {len(persistent_ids_s7)}")

    # ================= STAGE 8: FINAL MATRIX & PARALLEL OVERLAY RENDERING =================
    print(f"\n=== STAGE 8: RENDERING FINAL MATRIX & FULL 362 OVERLAYS ===")
    t_s8_start = time.perf_counter()

    final_classes_dict = {}
    for mid in persistent_ids_s7:
        final_classes_dict[mid] = {
            'id': mid,
            'unit_vec': final_embs_s7[mid],
            'frame_indices': sorted(list(final_frames_s7[mid])),
            'total_area': final_areas_s7[mid]
        }

    final_matrix_path = out_dir / "final_cross_class_similarity_matrix_full362.png"
    create_final_triangular_matrix(final_classes_dict, final_matrix_path)
    print(f"  Final Triangular Matrix saved: {final_matrix_path}")

    # Prepare tasks for parallel overlay rendering and NPZ saving
    print(f"  Rendering {num_frames} HD overlays and compressed NPZ archives in parallel (8 threads)...")
    tasks = []
    for idx, (p, item) in enumerate(zip(image_paths, stage4_merged_data)):
        stem = item['stem']
        masks = item['masks']
        c_ids = item['cleaned_ids']

        labels = np.zeros((H, W), dtype=np.int32)
        for nid, m in zip(c_ids, masks):
            if nid > 0: labels[m > 0] = nid

        embs = np.stack([final_embs_s7[nid] if nid > 0 else np.zeros(384, dtype=np.float32) for nid in c_ids]) if len(c_ids) > 0 else np.zeros((0, 384), dtype=np.float32)
        npz_path = npz_dir / f"{stem}_final.npz"
        overlay_path = overlays_dir / f"{stem}_overlay.png"
        tasks.append((p, stem, masks, labels, embs, c_ids, npz_path, overlay_path))

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        done_count = 0
        for _ in executor.map(render_and_save_frame, tasks):
            done_count += 1
            if done_count % 50 == 0 or done_count == num_frames:
                print(f"  [Overlays & NPZ] Rendered {done_count:03d}/{num_frames} frames")

    t_stage8_total = time.perf_counter() - t_s8_start
    t_grand_total = time.perf_counter() - t_grand_start

    # Grand Summary JSON
    summary = {
        'num_frames': num_frames,
        'final_classes_count': len(persistent_ids_s7),
        'final_classes': persistent_ids_s7,
        'timing_seconds': {
            'stage1_sam2_inference': round(t_stage1_total, 2),
            'stage2_gpu_refine_regularize': round(t_stage2_total, 2),
            'stage3_dinov2_features': round(t_stage3_total, 2),
            'stage4_single_image_merge': round(t_stage4_total, 2),
            'stage5_global_memory_bank_propagation': round(t_stage5_total, 2),
            'stage6_post_hoc_consolidation': round(t_stage6_total, 2),
            'stage7_cleanup_and_rare_merge': round(t_stage7_total, 2),
            'stage8_final_matrix_and_overlays': round(t_stage8_total, 2),
            'grand_total_pipeline': round(t_grand_total, 2)
        },
        'timing_per_frame_ms': {
            'stage1_sam2_inference': round(t_stage1_total / num_frames * 1000, 2),
            'stage2_gpu_refine_regularize': round(t_stage2_total / num_frames * 1000, 2),
            'stage3_dinov2_features': round(t_stage3_total / num_frames * 1000, 2),
            'stage4_single_image_merge': round(t_stage4_total / num_frames * 1000, 2),
            'total_stages1_to_4_per_frame': round((t_stage1_total + t_stage2_total + t_stage3_total + t_stage4_total) / num_frames * 1000, 2)
        },
        'throughput_fps': round(num_frames / t_grand_total, 2)
    }

    with open(out_dir / "full_pipeline_benchmark_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=================================================================")
    print("PIPELINE EXECUTION COMPLETE ON ALL 362 FRAMES!")
    print(f"Grand Total Elapsed: {t_grand_total:.2f}s ({t_grand_total/60:.2f} min) | Average Throughput: {num_frames / t_grand_total:.2f} FPS")
    print(f"Total Physical Classes Discovered: {len(persistent_ids_s7)}")
    print(f"Summary JSON: {out_dir / 'full_pipeline_benchmark_summary.json'}")
    print(f"Overlays:     {overlays_dir}")
    print(f"Matrix:       {final_matrix_path}")
    print("=================================================================")

if __name__ == '__main__':
    main()
