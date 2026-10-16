#!/usr/bin/env python3
"""
Post-Hoc Global Bank Merging (Memory Consolidation) - Single Merge Max per Class.

Rules:
1. No co-occurrence restriction: Classes can merge even if they co-occur in the same image.
2. Threshold: Cosine similarity > 0.690.
3. Single Merge Max per Class:
   - Rank all candidate pairs globally in descending order of mutual cosine similarity.
   - Greedily merge pairs (u, v) where neither u nor v has been merged yet.
   - For each merged pair, the lower ID is retained: target_id = min(u, v).
   - In the pre-merge 53x53 matrix, each participating class has exactly ONE red-highlighted cell.
4. Output:
   - 'pre_merge_53x53_highlighted_matrix.png': 53x53 matrix with red background on the 8 selected merge cells.
   - 'consolidated_cross_class_similarity_matrix.png': 45x45 matrix recomputed after the 8 merges.
   - 50 full-resolution visual overlays with updated consolidated IDs.
   - Updated NPZ arrays and metadata JSONs.
"""

import os, sys, json, time
from pathlib import Path
import cv2, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

MERGE_SIM_THRESHOLD = 0.690

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

    for mid, m in zip(mask_ids, masks):
        col_bgr, _ = get_class_color(mid)
        col_arr = np.array(col_bgr, dtype=np.uint8)
        vis = m > 0
        overlay[vis] = (0.55 * overlay[vis] + 0.45 * col_arr).astype(np.uint8)

    overlay[labels == 0] = 0

    for mid, m in zip(mask_ids, masks):
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

def create_pre_merge_highlighted_matrix(classes_dict: dict, merge_pairs: list, out_path: Path):
    """
    Renders initial 53x53 upper-triangular similarity matrix.
    Selected 1-to-1 merge cells are highlighted with red background.
    """
    sorted_ids = sorted(classes_dict.keys())
    K = len(sorted_ids)
    id_to_idx = {mid: idx for idx, mid in enumerate(sorted_ids)}

    merged_cells = set()
    for ida, idb, _ in merge_pairs:
        ia = id_to_idx[ida]
        ib = id_to_idx[idb]
        r = min(ia, ib)
        c = max(ia, ib)
        merged_cells.add((r, c))

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

    fig_dim = max(11.0, K * 0.45 + 3.0)
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
                        color='white' if lum < 0.55 else 'black', fontweight='bold', fontsize=8.5)
            elif (i, j) in merged_cells:
                # Highlighted cell: Red background
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='#ff2a2a', edgecolor='black', linewidth=2.0)
                ax.add_patch(rect)
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color='white', fontsize=8.5, fontweight='bold')
            else:
                c = cmap((val + 0.1) / 1.1)
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=c, edgecolor='#e0e0e0', linewidth=0.5)
                ax.add_patch(rect)
                lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
                fontweight = 'bold' if val >= 0.70 else 'normal'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color='white' if lum < 0.55 else 'black', fontsize=7.5, fontweight=fontweight)

    ax.set_xlim(-0.5, K - 0.5)
    ax.set_ylim(K - 0.5, -0.5)
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(labels, fontsize=8, rotation=90)
    ax.set_yticklabels(labels, fontsize=8)
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
    plt.title(f'Pre-Merge Initial Cross-Class Similarity Matrix (K={K} Classes)\nRED BACKGROUND = 1-to-1 Disjoint Merge Pairs (> {MERGE_SIM_THRESHOLD}, 1 Merge Max per Class)',
              fontsize=13, fontweight='bold', pad=45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def create_consolidated_triangular_matrix(classes_dict: dict, out_path: Path):
    """
    Renders the recomputed upper-triangular similarity matrix across surviving classes.
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

    fig_dim = max(10.0, K * 0.45 + 3.0)
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
                        color='white' if lum < 0.55 else 'black', fontweight='bold', fontsize=8.5)
            else:
                c = cmap((val + 0.1) / 1.1)
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=c, edgecolor='#e0e0e0', linewidth=0.5)
                ax.add_patch(rect)
                lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
                fontweight = 'bold' if val >= 0.70 else 'normal'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color='white' if lum < 0.55 else 'black', fontsize=7.5, fontweight=fontweight)

    ax.set_xlim(-0.5, K - 0.5)
    ax.set_ylim(K - 0.5, -0.5)
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(labels, fontsize=8, rotation=90)
    ax.set_yticklabels(labels, fontsize=8)
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
    plt.title(f'Post-Hoc Consolidated Triangular Similarity Matrix (K={K} Classes)\nSingle-Merge-Max Completed (> {MERGE_SIM_THRESHOLD})',
              fontsize=13, fontweight='bold', pad=45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def main():
    root = Path(__file__).resolve().parent.parent
    in_dir = root / "current_scene" / "06c_semantic_segmentation" / "first50_global_memory_bank"
    img_dir = root / "current_scene" / "06c_semantic_segmentation" / "inputs" / "images"
    out_dir = root / "current_scene" / "06c_semantic_segmentation" / "first50_memory_consolidated"
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(in_dir.glob("frame_*_propagated.npz"))
    num_frames = len(npz_files)

    print("=================================================================")
    print("POST-HOC GLOBAL BANK MERGING (1 MERGE MAX PER CLASS)")
    print(f"Similarity Threshold: > {MERGE_SIM_THRESHOLD} | Constraint: 1 Merge Max per Class")
    print(f"Input: {in_dir}")
    print(f"Output: {out_dir}")
    print("=================================================================\n")

    # 1. Load initial state
    initial_classes_dict = {}
    class_embs = {}
    class_areas = {}
    class_frames = {}
    per_frame_data = []

    for f_idx, p in enumerate(npz_files):
        stem = p.stem.replace('_propagated', '')
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

    initial_K = len(class_embs)
    for mid in class_embs:
        initial_classes_dict[mid] = {
            'id': mid,
            'unit_vec': class_embs[mid].copy(),
            'frame_indices': sorted(list(class_frames[mid])),
            'total_area': class_areas[mid]
        }

    print(f"Loaded {initial_K} global classes across {num_frames} frames.")

    # 2. Compute initial matrix and find greedy 1-merge-max disjoint pairs
    sorted_ids = sorted(class_embs.keys())
    units = np.stack([class_embs[mid] for mid in sorted_ids])
    S = units @ units.T

    candidates = []
    for i in range(len(sorted_ids)):
        for j in range(i + 1, len(sorted_ids)):
            val = float(S[i, j])
            if val > MERGE_SIM_THRESHOLD:
                candidates.append((sorted_ids[i], sorted_ids[j], val))

    candidates.sort(key=lambda x: x[2], reverse=True)

    matched_classes = set()
    selected_merges = []
    for ida, idb, val in candidates:
        if ida not in matched_classes and idb not in matched_classes:
            matched_classes.add(ida)
            matched_classes.add(idb)
            selected_merges.append((ida, idb, val))

    print(f"Identified {len(selected_merges)} disjoint merge pairs (1 merge max per class):")
    for ida, idb, val in selected_merges:
        target = min(ida, idb)
        merged = max(ida, idb)
        print(f"  Class #{merged} -> Class #{target} (sim={val:.4f})")

    # 3. Render Pre-Merge 53x53 matrix with RED BACKGROUND on selected merge cells
    pre_merge_matrix_path = out_dir / "pre_merge_53x53_highlighted_matrix.png"
    print(f"\nRendering 53x53 Pre-Merge Matrix with Red Highlighted Cells to:\n  {pre_merge_matrix_path}")
    create_pre_merge_highlighted_matrix(initial_classes_dict, selected_merges, pre_merge_matrix_path)

    # 4. Execute the merges
    id_mapping = {mid: mid for mid in sorted_ids}
    final_embs = {mid: class_embs[mid].copy() for mid in sorted_ids}
    final_areas = {mid: class_areas[mid] for mid in sorted_ids}
    final_frames = {mid: set(class_frames[mid]) for mid in sorted_ids}
    active_ids = set(sorted_ids)

    merge_log = []
    for step, (ida, idb, val) in enumerate(selected_merges, 1):
        target_id = min(ida, idb)
        merged_id = max(ida, idb)

        wa = final_areas[target_id]
        wb = final_areas[merged_id]
        new_vec = wa * final_embs[target_id] + wb * final_embs[merged_id]
        new_unit = new_vec / max(np.linalg.norm(new_vec), 1e-12)

        final_embs[target_id] = new_unit
        final_areas[target_id] = wa + wb
        final_frames[target_id] = final_frames[target_id].union(final_frames[merged_id])

        del final_embs[merged_id]
        del final_areas[merged_id]
        del final_frames[merged_id]
        active_ids.remove(merged_id)

        id_mapping[merged_id] = target_id

        merge_log.append({
            'merge_idx': step,
            'target_id': target_id,
            'merged_id': merged_id,
            'similarity': round(val, 4)
        })

    final_K = len(active_ids)
    print(f"\nFinal Class Count: {initial_K} -> {final_K} ({len(selected_merges)} classes collapsed).")

    # 5. Render Post-Merge Consolidated Matrix (45x45)
    final_classes_dict = {}
    for mid in sorted(active_ids):
        final_classes_dict[mid] = {
            'id': mid,
            'unit_vec': final_embs[mid],
            'frame_indices': sorted(list(final_frames[mid])),
            'total_area': final_areas[mid]
        }

    post_merge_matrix_path = out_dir / "consolidated_cross_class_similarity_matrix.png"
    print(f"\nRendering {final_K}x{final_K} Post-Merge Consolidated Matrix to:\n  {post_merge_matrix_path}")
    create_consolidated_triangular_matrix(final_classes_dict, post_merge_matrix_path)

    # 6. Re-render per-frame NPZ, JSON, and visual overlays with consolidated IDs
    print(f"\nRe-rendering overlays and NPZ archives for all {num_frames} frames...")
    for frame_info in per_frame_data:
        stem = frame_info['stem']
        masks = frame_info['masks']
        orig_ids = frame_info['ids']
        H, W = frame_info['H'], frame_info['W']

        new_ids = [id_mapping[old_id] for old_id in orig_ids]
        new_embeddings = np.stack([final_embs[nid] for nid in new_ids])

        labels = np.zeros((H, W), dtype=np.int32)
        for nid, m in zip(new_ids, masks):
            labels[m > 0] = nid

        np.savez_compressed(
            out_dir / f"{stem}_consolidated.npz",
            masks=masks,
            labels=labels,
            embeddings=new_embeddings,
            ids=np.array(new_ids)
        )

        ov = create_mask_overlay(img_dir / f"{stem}.jpg", masks, labels, new_ids)
        cv2.imwrite(str(out_dir / f"{stem}_consolidated_overlay.png"), ov)

        with open(out_dir / f"{stem}_consolidated.json", 'w') as f:
            json.dump({
                'frame': stem,
                'orig_ids': orig_ids,
                'consolidated_ids': new_ids,
                'num_masks': len(new_ids)
            }, f, indent=2)

    # Summary
    summary = {
        'num_frames': num_frames,
        'merge_similarity_threshold': MERGE_SIM_THRESHOLD,
        'initial_classes_count': initial_K,
        'final_classes_count': final_K,
        'merges_executed': merge_log,
        'id_mapping': {str(k): v for k, v in id_mapping.items() if k != v},
        'surviving_classes': sorted(list(active_ids))
    }
    with open(out_dir / "consolidation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=================================================================")
    print("CONSOLIDATION PASS COMPLETE!")
    print(f"Pre-Merge 53x53 Red Highlighted Matrix: {pre_merge_matrix_path}")
    print(f"Post-Merge {final_K}x{final_K} Consolidated Matrix:   {post_merge_matrix_path}")
    print(f"Overlays and Archives:                  {out_dir}")
    print("=================================================================")

if __name__ == '__main__':
    main()
