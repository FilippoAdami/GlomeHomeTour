import os, sys, json, time
from pathlib import Path
import cv2, torch, numpy as np
from PIL import Image
from transformers import AutoModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from fast_sam2_generator import FastSAM2Generator

def guided_filter_rgb(guide: np.ndarray, src: np.ndarray, radius: int = 4, eps: float = 1e-3) -> np.ndarray:
    guide = guide.astype(np.float32) / 255.0
    p = src.astype(np.float32)
    ksize = (2 * radius + 1, 2 * radius + 1)

    mean_I = cv2.boxFilter(guide, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)

    mean_Ip = np.zeros_like(mean_I)
    for c in range(3):
        mean_Ip[..., c] = cv2.boxFilter(guide[..., c] * p, cv2.CV_32F, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p[..., None]

    var_I = np.zeros((guide.shape[0], guide.shape[1], 3), dtype=np.float32)
    for c in range(3):
        mean_II_c = cv2.boxFilter(guide[..., c] * guide[..., c], cv2.CV_32F, ksize)
        var_I[..., c] = mean_II_c - mean_I[..., c] * mean_I[..., c]

    a = cov_Ip / (var_I + eps)
    b = mean_p - np.sum(a * mean_I, axis=2)

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

    q = np.sum(mean_a * guide, axis=2) + mean_b
    return q

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    workspace = Path("backend/current_scene/06c_semantic_segmentation")
    img_dir = workspace / "inputs" / "images"
    all_images = sorted(img_dir.glob("*.jpg"))[:50]
    print(f"Found {len(all_images)} images to process.")

    out_dir = workspace / "first50_stab80_iou50_reg05pct_merge069"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Init SAM2
    print("Initializing FastSAM2Generator (small, letterbox, grid=8, stability=0.80)...")
    sam_gen = FastSAM2Generator("small", grid=8, stability_thresh=0.80, compile_model=True, resize_mode="letterbox", nms_iou=0.85)

    # Init DINOv2
    print("Initializing DINOv2 (facebook/dinov2-small)...")
    dino_model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    MAX_SIDE = 896
    PATCH = 14
    MERGE_THRESH = 0.69 # merge if similarity >= 0.69

    # Warmup models
    print("Warming up models on frame 0...")
    warm_bgr = cv2.imread(str(all_images[0]))
    warm_rgb = cv2.cvtColor(warm_bgr, cv2.COLOR_BGR2RGB)
    with torch.inference_mode():
        _ = sam_gen.generate_complete(warm_rgb, 0.6, 0.9, return_logits=True)
        dummy_pixel = torch.zeros((1, 3, 448, 448), device=device)
        _ = dino_model(pixel_values=dummy_pixel)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("Warmup complete. Starting benchmark over 50 images...")

    timings = []
    
    for idx, img_path in enumerate(all_images):
        frame_name = img_path.stem
        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        total_pixels = H * W
        pil_image = Image.fromarray(rgb)

        # ----------------- 1. SAM2 Inference -----------------
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        masks, metadata, residual, logits = sam_gen.generate_complete(
            rgb, gap_stability=0.6, coverage_trigger=0.9, return_logits=True
        )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        t_sam2 = t1 - t0

        # Save pre-refinement visual for comparison (raw SAM2 overlay)
        # ----------------- 2. Initial Masks Refinement -----------------
        # Guided filter, IoU thresh >= 0.50, Outlier >= 0.5%, Filament filter, Hierarchical overlap, SOR
        t0 = time.perf_counter()
        
        # 2a. Guided Filter
        if len(masks) > 0 and logits is not None:
            refined_masks = []
            for i in range(len(masks)):
                refined_logit = guided_filter_rgb(rgb, logits[i], radius=4, eps=1e-3)
                ref_mask = (refined_logit > 0).astype(np.uint8)
                refined_masks.append(ref_mask)
            masks = np.stack(refined_masks)

        # 2b. IoU filter >= 0.50
        if len(masks) > 0:
            keep_iou = [i for i, m in enumerate(metadata) if m.get("predicted_iou", 0.0) >= 0.50]
            masks = masks[keep_iou] if keep_iou else np.zeros((0, H, W), dtype=np.uint8)
            metadata = [metadata[i] for i in keep_iou]

        # 2c. Outlier threshold 0.5%
        min_pixels = int(0.005 * total_pixels)
        if len(masks) > 0:
            survivor_indices = [i for i, m in enumerate(masks) if int(m.sum()) >= min_pixels]
            masks = masks[survivor_indices] if survivor_indices else np.zeros((0, H, W), dtype=np.uint8)
            metadata = [metadata[i] for i in survivor_indices]

        # 2d. Filament filter
        if len(masks) > 0:
            non_filament_indices = []
            for i, m in enumerate(masks):
                pts = cv2.findNonZero(m.astype(np.uint8))
                if pts is None: continue
                _, _, bw, bh = cv2.boundingRect(pts)
                span = max(bw, bh)
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                peri = sum(cv2.arcLength(c, True) for c in cnts)
                thickness = 2 * m.sum() / max(peri, 1)
                dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
                mean_dist = dist[m > 0].mean() if m.any() else 0
                is_filament = (thickness < 35.0 and span >= 150) or (mean_dist < 15.0 and span >= 200)
                if not is_filament:
                    non_filament_indices.append(i)
            masks = masks[non_filament_indices] if non_filament_indices else np.zeros((0, H, W), dtype=np.uint8)
            metadata = [metadata[i] for i in non_filament_indices]

        # 2e. Hierarchical Overlap Subtraction (33% relative confidence)
        items_ref = [{'mask': m.copy(), 'info': info} for m, info in zip(masks, metadata)]
        for i in range(len(items_ref)):
            for j in range(len(items_ref)):
                if i == j: continue
                mi = items_ref[i]['mask']
                mj = items_ref[j]['mask']
                overlap = (mi & mj)
                overlap_area = overlap.sum()
                if overlap_area == 0: continue
                ci = items_ref[i]['info']['predicted_iou']
                cj = items_ref[j]['info']['predicted_iou']
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

        # 2f. SOR Floater Cleanup
        surviving = []
        for item in items_ref:
            m = item['mask']
            if m.sum() == 0: continue
            nb_components, output_cc, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
            new_m = np.zeros_like(m)
            total_m_area = m.sum()
            sor_thresh = max(1000, int(0.05 * total_m_area))
            for comp_id in range(1, nb_components):
                comp_area = stats[comp_id, cv2.CC_STAT_AREA]
                if comp_area >= sor_thresh:
                    new_m[output_cc == comp_id] = 1
            m = new_m
            if m.sum() >= min_pixels:
                item['mask'] = m
                item['info']['area_px'] = int(m.sum())
                surviving.append(item)

        ref_masks = np.array([x['mask'] for x in surviving], dtype=np.uint8) if surviving else np.zeros((0, H, W), dtype=np.uint8)
        ref_metadata = [x['info'] for x in surviving]
        for m_id, info in enumerate(ref_metadata, start=1):
            info['mask_id'] = m_id

        # Hierarchical overlap resolution for non-overlapping visualization
        ref_labels = np.zeros((H, W), dtype=np.int32)
        for index in sorted(range(len(ref_metadata)), key=lambda i: (ref_metadata[i].get("source") != "grid", -ref_metadata[i]["predicted_iou"])):
            ref_labels[(ref_labels == 0) & ref_masks[index].astype(bool)] = ref_metadata[index]['mask_id']

        t1 = time.perf_counter()
        t_refine = t1 - t0

        # ----------------- 3. DINO First Inference -----------------
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        scale = MAX_SIDE / max(W, H)
        grid = (max(1, round(H * scale / PATCH)), max(1, round(W * scale / PATCH)))
        resized_pil = pil_image.resize((grid[1] * PATCH, grid[0] * PATCH), Image.Resampling.BICUBIC)
        pixels = torch.from_numpy(np.array(resized_pil)).permute(2, 0, 1).to(device=device, dtype=torch.float32)[None] / 255.0
        with torch.inference_mode():
            tokens = dino_model(pixel_values=(pixels - mean) / std).last_hidden_state[:, 1:, :]
        features = tokens.reshape(*grid, -1).float().cpu().numpy()

        # Downsample masks to DINO patch grid to compute spatial pooled embedding
        if len(ref_masks) > 0:
            mask_t = torch.from_numpy(ref_masks).float().unsqueeze(1)
            mask_low = torch.nn.functional.interpolate(mask_t, size=grid, mode='bilinear', align_corners=False)[:, 0].numpy()
        else:
            mask_low = np.zeros((0, grid[0], grid[1]), dtype=np.float32)

        items_dino = []
        for m_idx, (m, info) in enumerate(zip(ref_masks, ref_metadata)):
            weight = mask_low[m_idx]
            weight_bin = (weight > 0.4).astype(np.float32)
            if weight_bin.sum() == 0: weight_bin = weight
            sum_vec = (features * weight_bin[..., None]).sum(axis=(0, 1))
            unit_vec = sum_vec / max(np.linalg.norm(sum_vec), 1e-12)
            mid = info['mask_id']

            k = mid - 1
            col_bgr = [(37 * k + 93) % 255, (109 * k + 71) % 255, (197 * k + 31) % 255]
            col_rgb = [col_bgr[2] / 255.0, col_bgr[1] / 255.0, col_bgr[0] / 255.0]

            items_dino.append({
                'id': mid,
                'mask': m,
                'sum_vec': sum_vec,
                'unit_vec': unit_vec,
                'col_bgr': col_bgr,
                'col_rgb': col_rgb,
                'conf': info.get('predicted_iou', 0.0),
                'history': [mid]
            })

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        t_dino_infer = t1 - t0

        # ----------------- 4. DINO Iterative Masks Merging (>= 0.69) -----------------
        t0 = time.perf_counter()

        merge_log = []
        while True:
            N = len(items_dino)
            if N <= 1:
                break
            unit_mat = np.stack([it['unit_vec'] for it in items_dino])
            sim_mat = unit_mat @ unit_mat.T

            best_sim = -1.0
            best_pair = None
            for i in range(N):
                for j in range(i + 1, N):
                    if sim_mat[i, j] > best_sim:
                        best_sim = float(sim_mat[i, j])
                        best_pair = (i, j)

            if best_sim < MERGE_THRESH: # Stop when strictly less than 0.69
                break

            i, j = best_pair
            id_i, id_j = items_dino[i]['id'], items_dino[j]['id']
            keep_id = min(id_i, id_j)
            merge_log.append(f'Merge #{id_i} and #{id_j} (sim={best_sim:.4f}) -> #{keep_id}')

            merged_mask = items_dino[i]['mask'] | items_dino[j]['mask']
            merged_sum = items_dino[i]['sum_vec'] + items_dino[j]['sum_vec']
            merged_unit = merged_sum / max(np.linalg.norm(merged_sum), 1e-12)
            merged_conf = max(items_dino[i]['conf'], items_dino[j]['conf'])
            merged_hist = items_dino[i]['history'] + items_dino[j]['history']

            k = keep_id - 1
            col_bgr = [(37 * k + 93) % 255, (109 * k + 71) % 255, (197 * k + 31) % 255]
            col_rgb = [col_bgr[2] / 255.0, col_bgr[1] / 255.0, col_bgr[0] / 255.0]

            items_dino[i]['id'] = keep_id
            items_dino[i]['mask'] = merged_mask
            items_dino[i]['sum_vec'] = merged_sum
            items_dino[i]['unit_vec'] = merged_unit
            items_dino[i]['conf'] = merged_conf
            items_dino[i]['col_bgr'] = col_bgr
            items_dino[i]['col_rgb'] = col_rgb
            items_dino[i]['history'] = merged_hist

            items_dino.pop(j)

        items_dino.sort(key=lambda x: x['id'])
        N_final = len(items_dino)
        t1 = time.perf_counter()
        t_dino_merge = t1 - t0

        # Save record
        timing_rec = {
            'frame': frame_name,
            'initial_masks': len(ref_metadata),
            'final_masks': N_final,
            'merges_count': len(merge_log),
            't_sam2_sec': round(t_sam2, 4),
            't_refine_sec': round(t_refine, 4),
            't_dino_infer_sec': round(t_dino_infer, 4),
            't_dino_merge_sec': round(t_dino_merge, 4),
            't_total_sec': round(t_sam2 + t_refine + t_dino_infer + t_dino_merge, 4)
        }
        timings.append(timing_rec)

        print(f"[{idx+1:02d}/50] {frame_name}: SAM2={t_sam2*1000:.1f}ms, Refine={t_refine*1000:.1f}ms, DINO_infer={t_dino_infer*1000:.1f}ms, DINO_merge={t_dino_merge*1000:.2f}ms | Masks: {len(ref_metadata)}->{N_final} ({len(merge_log)} merges)")

        # Save final merged data (.npz and .json)
        final_unit_mat = np.stack([it['unit_vec'] for it in items_dino]) if N_final > 0 else np.zeros((0, 384))
        final_sim_mat = final_unit_mat @ final_unit_mat.T if N_final > 0 else np.zeros((0, 0))
        final_masks_arr = np.stack([it['mask'] for it in items_dino]) if N_final > 0 else np.zeros((0, H, W), dtype=np.uint8)
        final_labels = np.zeros((H, W), dtype=np.int32)
        for it in items_dino:
            final_labels[it['mask'] > 0] = it['id']

        np.savez_compressed(
            out_dir / f"{frame_name}_merged.npz",
            masks=final_masks_arr,
            labels=final_labels,
            embeddings=final_unit_mat,
            similarity_matrix=final_sim_mat
        )
        with open(out_dir / f"{frame_name}_merged.json", 'w') as f:
            json.dump({
                'frame': f"{frame_name}.jpg",
                'num_initial_masks': len(ref_metadata),
                'num_merged_masks': N_final,
                'merge_log': merge_log,
                'timings': timing_rec
            }, f, indent=2)

        # Plot 3-panel comparison for key review frames or all
        # To avoid matplotlib memory accumulation or slow I/O, generate comparison images for all 50
        fig_dim = max(7, N_final * 0.8 + 2)
        fig, ax = plt.subplots(figsize=(fig_dim, fig_dim), dpi=120)
        cmap = plt.cm.YlGnBu

        for i in range(N_final):
            for j in range(N_final):
                if j < i: continue
                if i == j:
                    c = items_dino[i]['col_rgb']
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=c, edgecolor='black', linewidth=1.5)
                    ax.add_patch(rect)
                    lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
                    text_col = 'white' if lum < 0.55 else 'black'
                    ax.text(j, i, f"#{items_dino[i]['id']}", ha='center', va='center', color=text_col, fontweight='bold', fontsize=11)
                else:
                    val = float(final_sim_mat[i, j])
                    c = cmap((val + 0.1) / 1.1)
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=c, edgecolor='#cccccc', linewidth=0.8)
                    ax.add_patch(rect)
                    lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
                    text_col = 'white' if lum < 0.55 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_col, fontsize=10, fontweight='medium')

        ax.set_xlim(-0.5, N_final - 0.5)
        ax.set_ylim(N_final - 0.5, -0.5)
        ax.set_xticks(range(N_final))
        ax.set_yticks(range(N_final))
        ax.set_xticklabels([f"#{it['id']}" for it in items_dino], fontsize=11)
        ax.set_yticklabels([f"#{it['id']}" for it in items_dino], fontsize=11)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')

        for t_idx, tick in enumerate(ax.get_xticklabels()):
            tick.set_color(items_dino[t_idx]['col_rgb'])
            tick.set_fontweight('bold')
        for t_idx, tick in enumerate(ax.get_yticklabels()):
            tick.set_color(items_dino[t_idx]['col_rgb'])
            tick.set_fontweight('bold')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('DINOv2 Cosine Similarity', fontsize=11, fontweight='medium')
        plt.title(f'{frame_name}: Final Similarity Matrix (Thresh < 0.69)', fontsize=13, fontweight='bold', pad=25)
        plt.tight_layout()
        matrix_img_path = out_dir / f'{frame_name}_final_similarity_matrix.png'
        plt.savefig(matrix_img_path)
        plt.close(fig)

        # Build Pre-merge and Post-merge overlays
        # Pre-merge overlay
        pre_overlay = bgr.copy()
        for it in ref_metadata:
            mid = it['mask_id']
            k = mid - 1
            col_bgr = np.array([(37 * k + 93) % 255, (109 * k + 71) % 255, (197 * k + 31) % 255], dtype=np.uint8)
            vis = ref_masks[mid - 1] > 0
            pre_overlay[vis] = (0.55 * pre_overlay[vis] + 0.45 * col_bgr).astype(np.uint8)
        pre_overlay[ref_labels == 0] = 0
        for it in ref_metadata:
            mid = it['mask_id']
            k = mid - 1
            col_bgr = tuple(int(c) for c in [(37 * k + 93) % 255, (109 * k + 71) % 255, (197 * k + 31) % 255])
            mask_u8 = (ref_labels == mid).astype(np.uint8)
            if not mask_u8.any(): continue
            cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(pre_overlay, cnts, -1, col_bgr, 2)
            M = cv2.moments(mask_u8)
            cx = int(M['m10'] / M['m00']) if M['m00'] > 0 else int(cv2.findNonZero(mask_u8)[:, 0, 0].mean())
            cy = int(M['m01'] / M['m00']) if M['m00'] > 0 else int(cv2.findNonZero(mask_u8)[:, 0, 1].mean())
            txt = f"#{mid}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            bx1, by1 = max(0, cx - tw // 2 - 6), max(0, cy - th // 2 - 6)
            bx2, by2 = min(W, cx + tw // 2 + 6), min(H, cy + th // 2 + 6)
            cv2.rectangle(pre_overlay, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
            cv2.rectangle(pre_overlay, (bx1, by1), (bx2, by2), col_bgr, 2)
            cv2.putText(pre_overlay, txt, (max(0, cx - tw // 2), min(H - 2, cy + th // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # Post-merge overlay
        post_overlay = bgr.copy()
        for it in items_dino:
            col = np.array(it['col_bgr'], dtype=np.uint8)
            vis = it['mask'] > 0
            post_overlay[vis] = (0.55 * post_overlay[vis] + 0.45 * col).astype(np.uint8)
        post_overlay[final_labels == 0] = 0
        for it in items_dino:
            mask_u8 = (final_labels == it['id']).astype(np.uint8)
            if not mask_u8.any(): continue
            color = tuple(int(c) for c in it['col_bgr'])
            cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(post_overlay, cnts, -1, color, 2)
            M = cv2.moments(mask_u8)
            cx = int(M['m10'] / M['m00']) if M['m00'] > 0 else int(cv2.findNonZero(mask_u8)[:, 0, 0].mean())
            cy = int(M['m01'] / M['m00']) if M['m00'] > 0 else int(cv2.findNonZero(mask_u8)[:, 0, 1].mean())
            txt = f"#{it['id']}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            bx1, by1 = max(0, cx - tw // 2 - 6), max(0, cy - th // 2 - 6)
            bx2, by2 = min(W, cx + tw // 2 + 6), min(H, cy + th // 2 + 6)
            cv2.rectangle(post_overlay, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
            cv2.rectangle(post_overlay, (bx1, by1), (bx2, by2), color, 2)
            cv2.putText(post_overlay, txt, (max(0, cx - tw // 2), min(H - 2, cy + th // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        mat_img = cv2.imread(str(matrix_img_path))
        aspect = mat_img.shape[1] / mat_img.shape[0]
        target_w = int(H * aspect)
        mat_resized = cv2.resize(mat_img, (target_w, H), interpolation=cv2.INTER_AREA)

        banner_h = 70
        p_pre = np.zeros((H + banner_h, W, 3), dtype=np.uint8)
        p_post = np.zeros((H + banner_h, W, 3), dtype=np.uint8)
        p_mat = np.zeros((H + banner_h, target_w, 3), dtype=np.uint8)

        p_pre[:banner_h, :] = (30, 30, 30)
        p_post[:banner_h, :] = (30, 30, 30)
        p_mat[:banner_h, :] = (30, 30, 30)

        p_pre[banner_h:, :] = pre_overlay
        p_post[banner_h:, :] = post_overlay
        p_mat[banner_h:, :] = mat_resized

        cv2.putText(p_pre, f'PRE-MERGE ({len(ref_metadata)} masks)', (40, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 200, 255), 3, cv2.LINE_AA)
        cv2.putText(p_post, f'POST-MERGE (>=0.69 Merged, {N_final} masks)', (40, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100, 255, 150), 3, cv2.LINE_AA)
        cv2.putText(p_mat, f'FINAL TRIANGULAR MATRIX', (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 100), 2, cv2.LINE_AA)

        divider = np.full((H + banner_h, 8, 3), 255, dtype=np.uint8)
        full_comp = np.hstack([p_pre, divider, p_post, divider, p_mat])
        cv2.imwrite(str(out_dir / f'{frame_name}_comparison_with_matrix.png'), full_comp)

    # ----------------- Summary Report -----------------
    sam2_times = [t['t_sam2_sec'] for t in timings]
    refine_times = [t['t_refine_sec'] for t in timings]
    dino_infer_times = [t['t_dino_infer_sec'] for t in timings]
    dino_merge_times = [t['t_dino_merge_sec'] for t in timings]
    total_times = [t['t_total_sec'] for t in timings]

    summary = {
        'num_frames': len(timings),
        'merge_similarity_threshold': MERGE_THRESH,
        'mean_sam2_sec': float(np.mean(sam2_times)),
        'std_sam2_sec': float(np.std(sam2_times)),
        'mean_refine_sec': float(np.mean(refine_times)),
        'std_refine_sec': float(np.std(refine_times)),
        'mean_dino_infer_sec': float(np.mean(dino_infer_times)),
        'std_dino_infer_sec': float(np.std(dino_infer_times)),
        'mean_dino_merge_sec': float(np.mean(dino_merge_times)),
        'std_dino_merge_sec': float(np.std(dino_merge_times)),
        'mean_total_sec': float(np.mean(total_times)),
        'mean_fps': float(1.0 / np.mean(total_times)),
        'timings_per_frame': timings
    }

    with open(out_dir / "benchmark_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n================== BENCHMARK SUMMARY (50 FRAMES) ==================")
    print(f"SAM2 Inference:              {summary['mean_sam2_sec']*1000:7.2f} ms ± {summary['std_sam2_sec']*1000:6.2f} ms")
    print(f"Initial Masks Refinement:    {summary['mean_refine_sec']*1000:7.2f} ms ± {summary['std_refine_sec']*1000:6.2f} ms")
    print(f"DINOv2 First Inference:      {summary['mean_dino_infer_sec']*1000:7.2f} ms ± {summary['std_dino_infer_sec']*1000:6.2f} ms")
    print(f"DINOv2 Iterative Merging:    {summary['mean_dino_merge_sec']*1000:7.2f} ms ± {summary['std_dino_merge_sec']*1000:6.2f} ms")
    print(f"-------------------------------------------------------------------")
    print(f"Total Pipeline per Image:    {summary['mean_total_sec']*1000:7.2f} ms (Overall Throughput: {summary['mean_fps']:.2f} FPS)")
    print("===================================================================\n")

if __name__ == '__main__':
    main()
