#!/usr/bin/env python3
"""
Final Class Cleanup & Final Global Similarity Matrix.

Logic:
1. Load all current consolidated classes from 'first50_memory_consolidated/'.
2. Remove High-Gap Classes:
   - Calculate average interval between appearances: avg_gap = mean(Δt_appearances).
   - Any class with avg_gap > 5.0 is permanently removed from the active registry.
     (Masks for these classes are re-assigned to class 0 / unassigned).
3. Merge Infrequent Classes (seen 1 or 2 times):
   - For every surviving class that appeared in only 1 or 2 frames:
   - Merge it into the persistent class (seen >= 3 times) that has the highest cosine similarity,
     regardless of similarity value (no threshold).
   - Update canonical embeddings via cumulative area-weighted average.
4. Outputs:
   - 'final_cleaned_cross_class_similarity_matrix.png': Full upper-triangular similarity matrix
     of all final surviving physical classes.
   - 50 full HD visual overlays ('frame_XXXXX_cleaned_overlay.png') with clean distance-transform badges.
   - Updated per-frame NPZ archives and JSON summaries.
"""

import os, sys, json, time
from pathlib import Path
import cv2, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

MAX_AVG_GAP_ALLOWED = 5.0
RARE_APPEARANCE_THRESHOLD = 2

def get_class_color(mid: int):
    """
    Deterministic distinct RGB color for any integer class ID.
    """
    k = mid - 1
    b = (37 * k + 93) % 255
    g = (109 * k + 71) % 255
    r = (197 * k + 31) % 255
    col_bgr = (int(b), int(g), int(r))
    col_rgb = (r / 255.0, g / 255.0, b / 255.0)
    return col_bgr, col_rgb

def find_best_label_anchor(mask_u8: np.ndarray, W: int, H: int):
    """
    Interior anchor via Distance Transform on largest connected component.
    """
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
    """
    Renders high-visibility alpha-blended instance overlays with clear contours
    and crisp, centered ID badges.
    """
    bgr = cv2.imread(str(img_path))
    H, W = bgr.shape[:2]
    overlay = bgr.copy()

    # Mask tint (only for valid class IDs > 0)
    for mid, m in zip(mask_ids, masks):
        if mid == 0:
            continue
        col_bgr, _ = get_class_color(mid)
        col_arr = np.array(col_bgr, dtype=np.uint8)
        vis = m > 0
        overlay[vis] = (0.55 * overlay[vis] + 0.45 * col_arr).astype(np.uint8)

    overlay[labels == 0] = 0

    # Draw contours
    for mid, m in zip(mask_ids, masks):
        if mid == 0:
            continue
        col_bgr, _ = get_class_color(mid)
        mask_u8 = m.astype(np.uint8)
        if not mask_u8.any():
            continue
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, col_bgr, 3)

    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 1.3
    thick = 2
    pad_x = 14
    pad_y = 10

    for mid, m in zip(mask_ids, masks):
        if mid == 0:
            continue
        mask_u8 = m.astype(np.uint8)
        if not mask_u8.any():
            continue
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

def create_final_triangular_matrix(classes_dict: dict, out_path: Path):
    """
    Renders the final upper-triangular similarity matrix across all surviving physical classes.
    """
    sorted_ids = sorted(classes_dict.keys())
    K = len(sorted_ids)
    if K == 0:
        return

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
            if j < i:
                continue
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
    plt.title(f'Final Cleaned Physical Class Similarity Matrix (K={K} Classes)\nRare (<=2 frames) merged into best match; High-gap (>5.0) classes pruned',
              fontsize=13, fontweight='bold', pad=45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def main():
    root = Path(__file__).resolve().parent.parent
    in_dir = root / "current_scene" / "06c_semantic_segmentation" / "first50_memory_consolidated"
    img_dir = root / "current_scene" / "06c_semantic_segmentation" / "inputs" / "images"
    out_dir = root / "current_scene" / "06c_semantic_segmentation" / "first50_final_cleaned"
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(in_dir.glob("frame_*_consolidated.npz"))
    num_frames = len(npz_files)

    print("=================================================================")
    print("FINAL CLASS CLEANUP & FINAL SIMILARITY MATRIX")
    print(f"1. Remove classes with avg_gap > {MAX_AVG_GAP_ALLOWED}")
    print(f"2. Merge infrequent classes (seen <= {RARE_APPEARANCE_THRESHOLD} frames) into nearest persistent class")
    print(f"Input: {in_dir}")
    print(f"Output: {out_dir}")
    print("=================================================================\n")

    # 1. Load initial state from consolidated pass
    class_embs = {}
    class_areas = {}
    class_frames = {}
    per_frame_data = []

    for f_idx, p in enumerate(npz_files):
        stem = p.stem.replace('_consolidated', '')
        d = np.load(p)
        ids = [int(x) for x in d['ids']]
        embs = d['embeddings']
        masks = d['masks']
        H, W = masks.shape[1], masks.shape[2]

        for mid, emb, m in zip(ids, embs, masks):
            class_embs[mid] = emb.copy()
            class_areas[mid] = class_areas.get(mid, 0.0) + float(m.sum())
            class_frames.setdefault(mid, set()).add(f_idx)

        per_frame_data.append({
            'stem': stem,
            'f_idx': f_idx,
            'masks': masks,
            'ids': ids,
            'H': H,
            'W': W
        })

    all_initial_ids = sorted(class_embs.keys())
    print(f"Loaded {len(all_initial_ids)} classes from previous consolidation.")

    # 2. Prune classes with avg_gap > 5.0
    pruned_classes = []
    surviving_after_gap = []
    for mid in all_initial_ids:
        flist = sorted(list(class_frames[mid]))
        if len(flist) > 1:
            avg_gap = float(np.mean([flist[k+1] - flist[k] for k in range(len(flist)-1)]))
        else:
            avg_gap = 0.0

        if avg_gap > MAX_AVG_GAP_ALLOWED:
            pruned_classes.append((mid, len(flist), avg_gap))
        else:
            surviving_after_gap.append(mid)

    print(f"\n[Step 1] Pruned {len(pruned_classes)} classes with avg_gap > {MAX_AVG_GAP_ALLOWED}:")
    for mid, cnt, gap in pruned_classes:
        print(f"  Removed Class #{mid}: seen={cnt} frames, avg_gap={gap:.2f}")

    # 3. Categorize into rare (seen <= 2) and persistent (seen >= 3)
    rare_ids = [m for m in surviving_after_gap if len(class_frames[m]) <= RARE_APPEARANCE_THRESHOLD]
    persistent_ids = [m for m in surviving_after_gap if len(class_frames[m]) > RARE_APPEARANCE_THRESHOLD]

    print(f"\n[Step 2] Surviving Classes Breakdown:")
    print(f"  Persistent classes (seen >= 3 times): {len(persistent_ids)}")
    print(f"  Infrequent classes (seen <= 2 times): {len(rare_ids)}")

    # 4. Merge infrequent classes into nearest persistent class
    id_mapping = {mid: mid for mid in all_initial_ids}
    for mid, _, _ in pruned_classes:
        id_mapping[mid] = 0 # mapped to background / unassigned

    final_embs = {mid: class_embs[mid].copy() for mid in persistent_ids}
    final_areas = {mid: class_areas[mid] for mid in persistent_ids}
    final_frames = {mid: set(class_frames[mid]) for mid in persistent_ids}

    units_persist = np.stack([final_embs[mid] for mid in persistent_ids])
    merge_log = []

    for r_id in rare_ids:
        r_emb = class_embs[r_id]
        sims = units_persist @ r_emb
        best_p_idx = int(np.argmax(sims))
        best_p_id = persistent_ids[best_p_idx]
        best_sim = float(sims[best_p_idx])

        # Area-weighted embedding update
        wp = final_areas[best_p_id]
        wr = class_areas[r_id]
        new_vec = wp * final_embs[best_p_id] + wr * r_emb
        new_unit = new_vec / max(np.linalg.norm(new_vec), 1e-12)

        final_embs[best_p_id] = new_unit
        final_areas[best_p_id] = wp + wr
        final_frames[best_p_id] = final_frames[best_p_id].union(class_frames[r_id])

        # Update persistent matrix units for subsequent merges
        units_persist[best_p_idx] = new_unit

        id_mapping[r_id] = best_p_id
        merge_log.append({
            'rare_class_id': r_id,
            'frames_seen': len(class_frames[r_id]),
            'merged_into_persistent_id': best_p_id,
            'similarity': round(best_sim, 4),
            'rare_area': round(wr, 1),
            'persistent_area_before': round(wp, 1)
        })
        print(f"  Class #{r_id} (seen {len(class_frames[r_id])}) -> merged into Class #{best_p_id} (sim={best_sim:.4f})")

    final_K = len(persistent_ids)
    print(f"\nFinal Physical Class Registry Size: {final_K} classes.")

    # 5. Render Final Triangular Similarity Matrix
    final_classes_dict = {}
    for mid in persistent_ids:
        final_classes_dict[mid] = {
            'id': mid,
            'unit_vec': final_embs[mid],
            'frame_indices': sorted(list(final_frames[mid])),
            'total_area': final_areas[mid]
        }

    final_matrix_path = out_dir / "final_cleaned_cross_class_similarity_matrix.png"
    print(f"\nRendering final {final_K}x{final_K} similarity matrix to:\n  {final_matrix_path}")
    create_final_triangular_matrix(final_classes_dict, final_matrix_path)

    # 6. Re-render per-frame NPZ archives and visual overlays
    print(f"\nRe-rendering overlays and NPZ archives for all {num_frames} frames...")
    for frame_info in per_frame_data:
        stem = frame_info['stem']
        masks = frame_info['masks']
        orig_ids = frame_info['ids']
        H, W = frame_info['H'], frame_info['W']

        new_ids = [id_mapping[old_id] for old_id in orig_ids]
        
        # Build embeddings array for surviving masks
        new_embeddings = []
        for nid in new_ids:
            if nid in final_embs:
                new_embeddings.append(final_embs[nid])
            else:
                new_embeddings.append(np.zeros(384, dtype=np.float32))
        new_embeddings = np.stack(new_embeddings)

        labels = np.zeros((H, W), dtype=np.int32)
        for nid, m in zip(new_ids, masks):
            if nid > 0:
                labels[m > 0] = nid

        np.savez_compressed(
            out_dir / f"{stem}_cleaned.npz",
            masks=masks,
            labels=labels,
            embeddings=new_embeddings,
            ids=np.array(new_ids)
        )

        ov = create_mask_overlay(img_dir / f"{stem}.jpg", masks, labels, new_ids)
        cv2.imwrite(str(out_dir / f"{stem}_cleaned_overlay.png"), ov)

        with open(out_dir / f"{stem}_cleaned.json", 'w') as f:
            json.dump({
                'frame': stem,
                'orig_consolidated_ids': orig_ids,
                'cleaned_ids': new_ids,
                'num_active_masks': sum(1 for nid in new_ids if nid > 0)
            }, f, indent=2)

    # Summary
    summary = {
        'num_frames': num_frames,
        'initial_classes_count': len(all_initial_ids),
        'final_classes_count': final_K,
        'pruned_high_gap_classes': [{'id': m, 'frames_seen': c, 'avg_gap': g} for m, c, g in pruned_classes],
        'rare_merges_executed': merge_log,
        'final_surviving_classes': persistent_ids,
        'id_mapping': {str(k): v for k, v in id_mapping.items() if k != v}
    }
    with open(out_dir / "final_cleanup_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=================================================================")
    print("FINAL CLEANUP COMPLETE!")
    print(f"Final Physical Classes: {final_K}")
    print(f"Final Similarity Matrix: {final_matrix_path}")
    print(f"Outputs written to:      {out_dir}")
    print("=================================================================")

if __name__ == '__main__':
    main()
