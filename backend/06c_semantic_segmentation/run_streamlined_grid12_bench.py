import os, sys, json, time
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

def clean_carved_mask(m: np.ndarray, min_chunk_area: int, min_thickness: float = 12.0) -> np.ndarray:
    """
    Two-tier post-carving regularization:
    1. Component-level floater pruning: Discards small detached satellite pieces, slivers, and edge crumbs
       (< 1,000 px or < 5% of mask mass).
    2. Ribbon/tendril pruning: For surviving components with thickness < 25px, applies morphological
       opening (7x7 ellipse) to strip thin border tendrils, keeping solid chunky bodies (>= min_chunk_area, thickness >= 12px).
    """
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

    # Tier 2: Thick solid mask: keep intact immediately
    if thick >= 25.0:
        return m

    # Thin/carved mask: strip tendrils via morphological opening (7x7 ellipse)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    if opened.sum() < min_chunk_area:
        return np.zeros_like(m)

    # Check surviving components for solid chunky parts
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

def main():
    workspace = Path("backend/current_scene/06c_semantic_segmentation")
    img_dir = workspace / "inputs" / "images"
    image_paths = sorted(img_dir.glob("*.jpg"))[:50]
    out_dir = workspace / "first50_streamlined_grid12_merge069"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"STREAMLINED 12x12 BENCHMARK ON {len(image_paths)} FRAMES")
    print(f"Features: GPU Tensor Overlap Carving + Post-Carving Thread Pruning & Chunk Preservation")
    print(f"Output: {out_dir}")
    print(f"=======================================================\n")

    # 1. Initialize SAM2
    print("Initializing FastSAM2Generator (small, grid=12, no reprompting)...")
    sam_gen = FastSAM2Generator("small", grid=12, stability_thresh=0.80, compile_model=True, resize_mode="letterbox", nms_iou=0.85)

    # Warmup SAM2
    warm_bgr = cv2.imread(str(image_paths[0]))
    warm_rgb = cv2.cvtColor(warm_bgr, cv2.COLOR_BGR2RGB)
    _ = sam_gen.generate(warm_rgb, return_logits=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # ================= STAGE 1: BATCH SAM2 INFERENCE =================
    print(f"\nSTAGE 1: Running SAM2 inference on all {len(image_paths)} images...")
    sam_results = []
    t_sam2_all = []

    for idx, img_path in enumerate(image_paths):
        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        masks, metadata = sam_gen.generate(rgb, return_logits=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        dur = t1 - t0
        t_sam2_all.append(dur)

        sam_results.append({
            'frame_name': img_path.stem,
            'img_path': img_path,
            'rgb': rgb,
            'bgr': bgr,
            'H': H, 'W': W,
            'raw_masks': masks,
            'raw_metadata': metadata
        })
        if (idx + 1) % 10 == 0 or idx == len(image_paths) - 1:
            print(f"  Stage 1 (SAM2) progress: {idx + 1}/{len(image_paths)} frames (current: {dur*1000:.1f}ms, raw masks: {len(masks)})")

    # ================= STAGE 2: STREAMLINED MASK REFINEMENT =================
    print(f"\nSTAGE 2: Running Streamlined Mask Refinement with Post-Carving Thread Pruning & Chunk Preservation...")
    refine_results = []
    t_refine_all = []

    for idx, data in enumerate(sam_results):
        H, W = data['H'], data['W']
        total_pixels = H * W
        min_pixels = int(0.005 * total_pixels)
        metadata = data['raw_metadata']
        masks = data['raw_masks']

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Step 2a: Filter by IoU >= 0.50 and area >= 0.5%
        surv_idx = [i for i, m in enumerate(metadata) if m.get('predicted_iou', 0.0) >= 0.50 and m.get('area_px', 0) >= min_pixels]
        if not surv_idx:
            ref_masks = np.zeros((0, H, W), dtype=np.uint8)
            ref_meta = []
        else:
            masks_filt = masks[surv_idx]
            meta_filt = [metadata[i] for i in surv_idx]

            # Step 2b: GPU Tensor Hierarchical Overlap Carving
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

            # Step 2c: Post-Carving Thread Pruning & Chunk Preservation
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

        ref_labels = np.zeros((H, W), dtype=np.int32)
        for it in ref_meta:
            ref_labels[ref_masks[it['mask_id'] - 1] > 0] = it['mask_id']

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        dur = t1 - t0
        t_refine_all.append(dur)

        refine_results.append({
            'ref_masks': ref_masks,
            'ref_metadata': ref_meta,
            'ref_labels': ref_labels
        })
        if (idx + 1) % 10 == 0 or idx == len(image_paths) - 1:
            print(f"  Stage 2 (Refine) progress: {idx + 1}/{len(image_paths)} frames (current: {dur*1000:.1f}ms, surviving masks: {len(ref_meta)})")

    # ================= STAGE 3: BATCH DINOv2 INFERENCE =================
    print(f"\nSTAGE 3: Running DINOv2 inference on all {len(image_paths)} images...")
    dino_model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    MAX_SIDE = 896
    PATCH = 14

    dino_results = []
    t_dino_infer_all = []

    for idx, (data, ref_data) in enumerate(zip(sam_results, refine_results)):
        rgb = data['rgb']
        H, W = data['H'], data['W']
        ref_masks = ref_data['ref_masks']
        ref_metadata = ref_data['ref_metadata']

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        pil_image = Image.fromarray(rgb)
        scale = MAX_SIDE / max(W, H)
        grid = (max(1, round(H * scale / PATCH)), max(1, round(W * scale / PATCH)))
        resized_pil = pil_image.resize((grid[1] * PATCH, grid[0] * PATCH), Image.Resampling.BICUBIC)
        pixels = torch.from_numpy(np.array(resized_pil)).permute(2, 0, 1).to(device=device, dtype=torch.float32)[None] / 255.0

        with torch.inference_mode():
            tokens = dino_model(pixel_values=(pixels - mean) / std).last_hidden_state[:, 1:, :]
        features = tokens.reshape(*grid, -1).float().cpu().numpy()

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
        dur = t1 - t0
        t_dino_infer_all.append(dur)

        dino_results.append({
            'items_dino': items_dino
        })
        if (idx + 1) % 10 == 0 or idx == len(image_paths) - 1:
            print(f"  Stage 3 (DINOv2) progress: {idx + 1}/{len(image_paths)} frames (current: {dur*1000:.1f}ms)")

    # ================= STAGE 4: PER-IMAGE ITERATIVE MERGING (>= 0.69) =================
    print(f"\nSTAGE 4: Running Iterative Semantic Merging (thresh >= 0.69) on all {len(image_paths)} images...")
    MERGE_THRESH = 0.69
    t_dino_merge_all = []
    final_records = []

    for idx, (data, ref_data, dino_data) in enumerate(zip(sam_results, refine_results, dino_results)):
        frame_name = data['frame_name']
        bgr = data['bgr']
        H, W = data['H'], data['W']
        ref_metadata = ref_data['ref_metadata']
        ref_masks = ref_data['ref_masks']
        ref_labels = ref_data['ref_labels']
        items_dino = dino_data['items_dino']

        t0 = time.perf_counter()
        merge_log = []
        while True:
            N = len(items_dino)
            if N <= 1: break
            unit_mat = np.stack([it['unit_vec'] for it in items_dino])
            sim_mat = unit_mat @ unit_mat.T

            best_sim = -1.0
            best_pair = None
            for i in range(N):
                for j in range(i + 1, N):
                    if sim_mat[i, j] > best_sim:
                        best_sim = float(sim_mat[i, j])
                        best_pair = (i, j)

            if best_sim < MERGE_THRESH:
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
        dur = t1 - t0
        t_dino_merge_all.append(dur)

        t_sam = t_sam2_all[idx]
        t_ref = t_refine_all[idx]
        t_din = t_dino_infer_all[idx]
        t_mrg = dur
        t_tot = t_sam + t_ref + t_din + t_mrg

        final_records.append({
            'frame': frame_name,
            'initial_masks': len(ref_metadata),
            'final_masks': N_final,
            'merges_count': len(merge_log),
            't_sam2_sec': round(t_sam, 4),
            't_refine_sec': round(t_ref, 4),
            't_dino_infer_sec': round(t_din, 4),
            't_dino_merge_sec': round(t_mrg, 4),
            't_total_sec': round(t_tot, 4)
        })

        # Save artifacts (.npz and .json)
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
                'timings': final_records[-1]
            }, f, indent=2)

        # Generate comparison sheet
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
                    ax.text(j, i, f"#{items_dino[i]['id']}", ha='center', va='center', color='white' if lum < 0.55 else 'black', fontweight='bold', fontsize=11)
                else:
                    val = float(final_sim_mat[i, j])
                    c = cmap((val + 0.1) / 1.1)
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=c, edgecolor='#cccccc', linewidth=0.8)
                    ax.add_patch(rect)
                    lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white' if lum < 0.55 else 'black', fontsize=10, fontweight='medium')

        ax.set_xlim(-0.5, N_final - 0.5)
        ax.set_ylim(N_final - 0.5, -0.5)
        ax.set_xticks(range(N_final))
        ax.set_yticks(range(N_final))
        ax.set_xticklabels([f"#{it['id']}" for it in items_dino], fontsize=11)
        ax.set_yticklabels([f"#{it['id']}" for it in items_dino], fontsize=11)
        ax.xaxis.tick_top()
        for t_idx, tick in enumerate(ax.get_xticklabels()):
            tick.set_color(items_dino[t_idx]['col_rgb'])
            tick.set_fontweight('bold')
        for t_idx, tick in enumerate(ax.get_yticklabels()):
            tick.set_color(items_dino[t_idx]['col_rgb'])
            tick.set_fontweight('bold')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('DINOv2 Cosine Similarity', fontsize=11)
        plt.title(f'{frame_name}: Final Similarity Matrix (Thresh < 0.69)', fontsize=13, fontweight='bold', pad=25)
        plt.tight_layout()
        matrix_img_path = out_dir / f'{frame_name}_final_similarity_matrix.png'
        plt.savefig(matrix_img_path)
        plt.close(fig)

        # Overlays
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

        cv2.putText(p_pre, f'PRE-MERGE ({len(ref_meta)} masks)', (40, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 200, 255), 3, cv2.LINE_AA)
        cv2.putText(p_post, f'POST-MERGE (>=0.69 Merged, {N_final} masks)', (40, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100, 255, 150), 3, cv2.LINE_AA)
        cv2.putText(p_mat, f'FINAL TRIANGULAR MATRIX', (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 100), 2, cv2.LINE_AA)

        divider = np.full((H + banner_h, 8, 3), 255, dtype=np.uint8)
        full_comp = np.hstack([p_pre, divider, p_post, divider, p_mat])
        cv2.imwrite(str(out_dir / f'{frame_name}_comparison_with_matrix.png'), full_comp)

    # Summary
    summary = {
        'num_frames': len(image_paths),
        'grid_size': 12,
        'fill_gaps': False,
        'merge_threshold': MERGE_THRESH,
        'mean_sam2_ms': float(np.mean(t_sam2_all) * 1000),
        'std_sam2_ms': float(np.std(t_sam2_all) * 1000),
        'mean_refine_ms': float(np.mean(t_refine_all) * 1000),
        'std_refine_ms': float(np.std(t_refine_all) * 1000),
        'mean_dino_infer_ms': float(np.mean(t_dino_infer_all) * 1000),
        'std_dino_infer_ms': float(np.std(t_dino_infer_all) * 1000),
        'mean_dino_merge_ms': float(np.mean(t_dino_merge_all) * 1000),
        'std_dino_merge_ms': float(np.std(t_dino_merge_all) * 1000),
        'mean_total_ms': float(np.mean([r['t_total_sec'] for r in final_records]) * 1000),
        'mean_fps': float(1.0 / np.mean([r['t_total_sec'] for r in final_records])),
        'timings_per_frame': final_records
    }

    with open(out_dir / "benchmark_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n================ STREAMLINED 12x12 BENCHMARK SUMMARY ================")
    print(f"SAM2 Inference:              {summary['mean_sam2_ms']:7.2f} ms ± {summary['std_sam2_ms']:6.2f} ms")
    print(f"Initial Masks Refinement:    {summary['mean_refine_ms']:7.2f} ms ± {summary['std_refine_ms']:6.2f} ms")
    print(f"DINOv2 First Inference:      {summary['mean_dino_infer_ms']:7.2f} ms ± {summary['std_dino_infer_ms']:6.2f} ms")
    print(f"DINOv2 Iterative Merging:    {summary['mean_dino_merge_ms']:7.2f} ms ± {summary['std_dino_merge_ms']:6.2f} ms")
    print(f"---------------------------------------------------------------------")
    print(f"Total Pipeline per Image:    {summary['mean_total_ms']:7.2f} ms ({summary['mean_fps']:.2f} FPS)")
    print(f"=====================================================================\n")

if __name__ == '__main__':
    main()
