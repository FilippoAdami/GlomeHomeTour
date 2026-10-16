import os, sys, json, time
import cv2, torch, numpy as np
from PIL import Image
from transformers import AutoModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

frames = [f'frame_{i:05d}' for i in [0, 4, 6, 12, 17, 23, 29, 33, 37, 42]]
img_dir = Path('backend/current_scene/06c_semantic_segmentation/inputs/images')
masks_dir = Path('backend/current_scene/06c_semantic_segmentation/sam2_proposals/first10_stab80_iou50_reg05pct')
out_dir = Path('backend/current_scene/06c_semantic_segmentation/dinov2_merged_070/first10_stab80_iou50_reg05pct')
out_dir.mkdir(parents=True, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Loading DINOv2 model on {device}...')
model = AutoModel.from_pretrained('facebook/dinov2-small').to(device).eval()

mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
MAX_SIDE = 896
PATCH = 14

for frame in frames:
    print(f'Processing {frame}...')
    img_path = img_dir / f'{frame}.jpg'
    bgr = cv2.imread(str(img_path))
    image = Image.open(img_path).convert('RGB')
    orig_w, orig_h = image.size
    
    scale = MAX_SIDE / max(orig_w, orig_h)
    grid = (max(1, round(orig_h * scale / PATCH)), max(1, round(orig_w * scale / PATCH)))
    resized = image.resize((grid[1] * PATCH, grid[0] * PATCH), Image.Resampling.BICUBIC)
    
    pixels = torch.from_numpy(np.array(resized)).permute(2, 0, 1).to(device=device, dtype=torch.float32)[None] / 255.0
    with torch.inference_mode():
        tokens = model(pixel_values=(pixels - mean) / std).last_hidden_state[:, 1:, :]
    features = tokens.reshape(*grid, -1).float().cpu().numpy()
    
    npz_data = np.load(masks_dir / f'{frame}.npz')
    masks = [m.copy() for m in npz_data['masks']]
    with open(masks_dir / f'{frame}.json') as f:
        meta = json.load(f)['masks']
        
    mask_t = torch.from_numpy(np.array(masks)).float().unsqueeze(1)
    mask_low = torch.nn.functional.interpolate(mask_t, size=grid, mode='bilinear', align_corners=False)[:, 0].numpy()
    
    items = []
    for idx, (m, info) in enumerate(zip(masks, meta)):
        weight = mask_low[idx]
        weight_bin = (weight > 0.4).astype(np.float32)
        if weight_bin.sum() == 0:
            weight_bin = weight
        sum_vec = (features * weight_bin[..., None]).sum(axis=(0, 1))
        unit_vec = sum_vec / max(np.linalg.norm(sum_vec), 1e-12)
        mid = info['mask_id'] # 1-based original ID
        
        k = mid - 1
        col_bgr = [(37 * k + 93) % 255, (109 * k + 71) % 255, (197 * k + 31) % 255]
        col_rgb = [col_bgr[2] / 255.0, col_bgr[1] / 255.0, col_bgr[0] / 255.0]
        
        items.append({
            'id': mid,
            'mask': m,
            'sum_vec': sum_vec,
            'unit_vec': unit_vec,
            'col_bgr': col_bgr,
            'col_rgb': col_rgb,
            'conf': info.get('predicted_iou', 0.0),
            'history': [mid]
        })
        
    step = 0
    merge_log = []
    while True:
        N = len(items)
        if N <= 1:
            break
        unit_mat = np.stack([it['unit_vec'] for it in items])
        sim_mat = unit_mat @ unit_mat.T
        
        best_sim = -1.0
        best_pair = None
        for i in range(N):
            for j in range(i + 1, N):
                if sim_mat[i, j] > best_sim:
                    best_sim = float(sim_mat[i, j])
                    best_pair = (i, j)
                    
        if best_sim < 0.70:
            break
            
        i, j = best_pair
        id_i, id_j = items[i]['id'], items[j]['id']
        keep_id = min(id_i, id_j)
        step += 1
        merge_log.append(f'Merge #{id_i} and #{id_j} (sim={best_sim:.3f}) -> #{keep_id}')
        
        merged_mask = items[i]['mask'] | items[j]['mask']
        merged_sum = items[i]['sum_vec'] + items[j]['sum_vec']
        merged_unit = merged_sum / max(np.linalg.norm(merged_sum), 1e-12)
        merged_conf = max(items[i]['conf'], items[j]['conf'])
        merged_hist = items[i]['history'] + items[j]['history']
        
        k = keep_id - 1
        col_bgr = [(37 * k + 93) % 255, (109 * k + 71) % 255, (197 * k + 31) % 255]
        col_rgb = [col_bgr[2] / 255.0, col_bgr[1] / 255.0, col_bgr[0] / 255.0]
        
        items[i]['id'] = keep_id
        items[i]['mask'] = merged_mask
        items[i]['sum_vec'] = merged_sum
        items[i]['unit_vec'] = merged_unit
        items[i]['conf'] = merged_conf
        items[i]['col_bgr'] = col_bgr
        items[i]['col_rgb'] = col_rgb
        items[i]['history'] = merged_hist
        
        items.pop(j)
        
    print(f'  {frame}: {len(meta)} -> {len(items)} masks after merging.')
    for l in merge_log:
        print(f'    {l}')
        
    items.sort(key=lambda x: x['id'])
    N_final = len(items)
    
    final_unit_mat = np.stack([it['unit_vec'] for it in items])
    final_sim_mat = final_unit_mat @ final_unit_mat.T
    
    final_masks_arr = np.stack([it['mask'] for it in items])
    final_labels = np.zeros((orig_h, orig_w), dtype=np.int32)
    for idx, it in enumerate(items):
        final_labels[it['mask'] > 0] = it['id']
        
    np.savez_compressed(
        out_dir / f'{frame}_merged.npz',
        masks=final_masks_arr,
        labels=final_labels,
        embeddings=final_unit_mat,
        similarity_matrix=final_sim_mat
    )
    with open(out_dir / f'{frame}_merged.json', 'w') as f:
        json.dump({
            'frame': f'{frame}.jpg',
            'num_initial_masks': len(meta),
            'num_merged_masks': N_final,
            'merge_log': merge_log,
            'masks': [
                {
                    'mask_id': it['id'],
                    'area_px': int(it['mask'].sum()),
                    'merged_from': it['history'],
                    'conf': round(float(it['conf']), 3)
                }
                for it in items
            ]
        }, f, indent=2)
        
    # --- Generate Final Triangular Similarity Matrix Plot ---
    fig_dim = max(7, N_final * 0.8 + 2)
    fig, ax = plt.subplots(figsize=(fig_dim, fig_dim), dpi=150)
    cmap = plt.cm.YlGnBu
    
    for i in range(N_final):
        for j in range(N_final):
            if j < i:
                continue
            if i == j:
                c = items[i]['col_rgb']
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=c, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
                text_col = 'white' if lum < 0.55 else 'black'
                ax.text(j, i, f"#{items[i]['id']}", ha='center', va='center', color=text_col, fontweight='bold', fontsize=11)
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
    ax.set_xticklabels([f"#{it['id']}" for it in items], fontsize=11)
    ax.set_yticklabels([f"#{it['id']}" for it in items], fontsize=11)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    for idx, tick in enumerate(ax.get_xticklabels()):
        tick.set_color(items[idx]['col_rgb'])
        tick.set_fontweight('bold')
    for idx, tick in enumerate(ax.get_yticklabels()):
        tick.set_color(items[idx]['col_rgb'])
        tick.set_fontweight('bold')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('DINOv2 Cosine Similarity', fontsize=11, fontweight='medium')

    plt.title(f'{frame}: Final Post-Merge Similarity Matrix (Threshold < 0.70)', fontsize=13, fontweight='bold', pad=25)
    plt.tight_layout()
    matrix_img_path = out_dir / f'{frame}_final_similarity_matrix.png'
    plt.savefig(matrix_img_path)
    plt.close()

    # --- Generate Final Comparison: Refined (Pre-Merge) vs Final (Post-Merge) with Matrix ---
    pre_overlay = cv2.imread(str(masks_dir / f'{frame}_overlay.png'))
    
    post_overlay = bgr.copy()
    for it in items:
        col = np.array(it['col_bgr'], dtype=np.uint8)
        vis = it['mask'] > 0
        post_overlay[vis] = (0.55 * post_overlay[vis] + 0.45 * col).astype(np.uint8)
    covered = final_labels > 0
    post_overlay[~covered] = 0
    
    for it in items:
        mask_u8 = (final_labels == it['id']).astype(np.uint8)
        if not mask_u8.any():
            continue
        color = tuple(int(c) for c in it['col_bgr'])
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(post_overlay, contours, -1, color, 2)
        M = cv2.moments(mask_u8)
        if M['m00'] > 0:
            cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
        else:
            pts = cv2.findNonZero(mask_u8)
            cx, cy = int(pts[:, 0, 0].mean()), int(pts[:, 0, 1].mean())
        text = f"#{it['id']}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.9, 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        pad = 6
        bx1, by1 = max(0, cx - tw // 2 - pad), max(0, cy - th // 2 - pad)
        bx2, by2 = min(orig_w, cx + tw // 2 + pad), min(orig_h, cy + th // 2 + pad)
        cv2.rectangle(post_overlay, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
        cv2.rectangle(post_overlay, (bx1, by1), (bx2, by2), color, 2)
        tx, ty = max(0, cx - tw // 2), min(orig_h - 2, cy + th // 2)
        cv2.putText(post_overlay, text, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    cv2.imwrite(str(out_dir / f'{frame}_post_merge_overlay.png'), post_overlay)
    
    mat_img = cv2.imread(str(matrix_img_path))
    target_h = orig_h
    aspect = mat_img.shape[1] / mat_img.shape[0]
    target_w = int(target_h * aspect)
    mat_resized = cv2.resize(mat_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    banner_h = 70
    panel_pre = np.zeros((orig_h + banner_h, orig_w, 3), dtype=np.uint8)
    panel_post = np.zeros((orig_h + banner_h, orig_w, 3), dtype=np.uint8)
    panel_mat = np.zeros((orig_h + banner_h, target_w, 3), dtype=np.uint8)
    
    panel_pre[:banner_h, :] = (30, 30, 30)
    panel_post[:banner_h, :] = (30, 30, 30)
    panel_mat[:banner_h, :] = (30, 30, 30)
    
    panel_pre[banner_h:, :] = pre_overlay
    panel_post[banner_h:, :] = post_overlay
    panel_mat[banner_h:, :] = mat_resized

    cv2.putText(panel_pre, f'PRE-MERGE ({len(meta)} masks)', (40, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 200, 255), 3, cv2.LINE_AA)
    cv2.putText(panel_post, f'POST-MERGE (>0.70 Merged, {N_final} masks)', (40, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100, 255, 150), 3, cv2.LINE_AA)
    cv2.putText(panel_mat, f'FINAL TRIANGULAR MATRIX', (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 100), 2, cv2.LINE_AA)

    divider = np.full((orig_h + banner_h, 8, 3), 255, dtype=np.uint8)
    full_comp = np.hstack([panel_pre, divider, panel_post, divider, panel_mat])
    cv2.imwrite(str(out_dir / f'{frame}_comparison_with_matrix.png'), full_comp)

print('ALL 10 FRAMES MERGED AND SAVED!')
