#!/usr/bin/env python3
"""
Compute cross-frame class similarity matrices and 3-column side-by-side comparisons
for consecutive frames in the first 50 frames processed by pipeline 06c.
"""

import os, sys, json, time
from pathlib import Path
import cv2, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_class_color(mid: int):
    k = mid - 1
    b = (37 * k + 93) % 255
    g = (109 * k + 71) % 255
    r = (197 * k + 31) % 255
    col_bgr = (int(b), int(g), int(r))
    col_rgb = (r / 255.0, g / 255.0, b / 255.0)
    return col_bgr, col_rgb

def create_mask_overlay(img_path: Path, masks: np.ndarray, labels: np.ndarray, mask_ids: list) -> np.ndarray:
    bgr = cv2.imread(str(img_path))
    H, W = bgr.shape[:2]
    overlay = bgr.copy()

    # Alpha blend masks
    for mid, m in zip(mask_ids, masks):
        col_bgr, _ = get_class_color(mid)
        col_arr = np.array(col_bgr, dtype=np.uint8)
        vis = m > 0
        overlay[vis] = (0.55 * overlay[vis] + 0.45 * col_arr).astype(np.uint8)

    # Background / unassigned to black
    overlay[labels == 0] = 0

    # Draw contours and text labels
    for mid in mask_ids:
        col_bgr, _ = get_class_color(mid)
        mask_u8 = (labels == mid).astype(np.uint8)
        if not mask_u8.any():
            continue
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, col_bgr, 2)
        M = cv2.moments(mask_u8)
        cx = int(M['m10'] / M['m00']) if M['m00'] > 0 else int(cv2.findNonZero(mask_u8)[:, 0, 0].mean())
        cy = int(M['m01'] / M['m00']) if M['m00'] > 0 else int(cv2.findNonZero(mask_u8)[:, 0, 1].mean())
        txt = f"#{mid}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        bx1, by1 = max(0, cx - tw // 2 - 6), max(0, cy - th // 2 - 6)
        bx2, by2 = min(W, cx + tw // 2 + 6), min(H, cy + th // 2 + 6)
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), col_bgr, 2)
        cv2.putText(overlay, txt, (max(0, cx - tw // 2), min(H - 2, cy + th // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return overlay

def create_cross_matrix_plot(sim_mat: np.ndarray, ids_a: list, ids_b: list,
                            name_a: str, name_b: str, out_path: Path):
    Na = len(ids_a)
    Nb = len(ids_b)

    fig_w = max(7.5, Nb * 0.95 + 3.2)
    fig_h = max(7.0, Na * 0.95 + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=130)
    cmap = plt.cm.YlGnBu

    for i in range(Na):
        for j in range(Nb):
            val = float(sim_mat[i, j])
            c = cmap((val + 0.1) / 1.1)
            rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=c, edgecolor='#dddddd', linewidth=0.8)
            ax.add_patch(rect)
            _, rgb_c = get_class_color(ids_a[i])
            lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
            fontweight = 'bold' if val >= 0.70 else 'medium'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color='white' if lum < 0.55 else 'black', fontsize=11, fontweight=fontweight)

    ax.set_xlim(-0.5, Nb - 0.5)
    ax.set_ylim(Na - 0.5, -0.5)
    ax.set_xticks(range(Nb))
    ax.set_yticks(range(Na))
    ax.set_xticklabels([f"#{mid}" for mid in ids_b], fontsize=12)
    ax.set_yticklabels([f"#{mid}" for mid in ids_a], fontsize=12)
    ax.xaxis.tick_top()
    ax.set_xlabel(f'{name_b} Classes (Cols)', fontsize=13, fontweight='bold', labelpad=14)
    ax.set_ylabel(f'{name_a} Classes (Rows)', fontsize=13, fontweight='bold', labelpad=14)
    ax.xaxis.set_label_position('top')

    for t_idx, tick in enumerate(ax.get_xticklabels()):
        _, col_rgb = get_class_color(ids_b[t_idx])
        tick.set_color(col_rgb)
        tick.set_fontweight('bold')

    for t_idx, tick in enumerate(ax.get_yticklabels()):
        _, col_rgb = get_class_color(ids_a[t_idx])
        tick.set_color(col_rgb)
        tick.set_fontweight('bold')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('DINOv2 Cosine Similarity', fontsize=12)
    plt.title(f'Cross-Frame Cosine Similarity\n{name_a} & {name_b}', fontsize=14, fontweight='bold', pad=36)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "current_scene" / "06c_semantic_segmentation" / "first50_streamlined_grid12_merge069"
    img_dir = root / "current_scene" / "06c_semantic_segmentation" / "inputs" / "images"
    out_dir = root / "current_scene" / "06c_semantic_segmentation" / "cross_frame_similarity_first50"
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(data_dir.glob("frame_*_merged.npz"))
    num_frames = len(npz_files)
    num_pairs = num_frames - 1

    print("=================================================================")
    print(f"CROSS-FRAME SIMILARITY COMPUTATION & 3-COLUMN VISUALIZATION")
    print(f"Total Frames: {num_frames} | Total Consecutive Pairs: {num_pairs}")
    print(f"Input Data: {data_dir}")
    print(f"Output Directory: {out_dir}")
    print("=================================================================\n")

    summary_records = []

    for idx in range(num_pairs):
        t0 = time.perf_counter()
        p_a = npz_files[idx]
        p_b = npz_files[idx + 1]

        stem_a = p_a.stem.replace('_merged', '')
        stem_b = p_b.stem.replace('_merged', '')
        pair_name = f"pair_{idx:02d}_{stem_a}_vs_{stem_b}"

        d_a = np.load(p_a)
        d_b = np.load(p_b)

        masks_a, labels_a, emb_a = d_a['masks'], d_a['labels'], d_a['embeddings']
        masks_b, labels_b, emb_b = d_b['masks'], d_b['labels'], d_b['embeddings']

        ids_a = [int(labels_a[m > 0][0]) for m in masks_a]
        ids_b = [int(labels_b[m > 0][0]) for m in masks_b]

        # Compute cross-frame cosine similarity matrix: shape (Na, Nb)
        sim_mat = emb_a @ emb_b.T

        # Find best matches in both directions
        best_a_to_b = []
        for i, mid_a in enumerate(ids_a):
            best_j = int(np.argmax(sim_mat[i]))
            best_sim = float(sim_mat[i, best_j])
            best_a_to_b.append({
                'class_a': mid_a,
                'best_class_b': ids_b[best_j],
                'similarity': round(best_sim, 4)
            })

        # Save matrix plot
        mat_plot_path = out_dir / f"{pair_name}_similarity_matrix.png"
        create_cross_matrix_plot(sim_mat, ids_a, ids_b, stem_a, stem_b, mat_plot_path)

        # Create overlays
        img_a_path = img_dir / f"{stem_a}.jpg"
        img_b_path = img_dir / f"{stem_b}.jpg"

        ov_a = create_mask_overlay(img_a_path, masks_a, labels_a, ids_a)
        ov_b = create_mask_overlay(img_b_path, masks_b, labels_b, ids_b)

        # Assemble 3-panel composite: [Image A | Image B | Similarity Matrix]
        H, W = ov_a.shape[:2]
        banner_h = 70

        mat_img = cv2.imread(str(mat_plot_path))
        mat_aspect = mat_img.shape[1] / mat_img.shape[0]
        target_w = int(H * mat_aspect)
        mat_resized = cv2.resize(mat_img, (target_w, H), interpolation=cv2.INTER_AREA)

        p_panel_a = np.zeros((H + banner_h, W, 3), dtype=np.uint8)
        p_panel_b = np.zeros((H + banner_h, W, 3), dtype=np.uint8)
        p_panel_mat = np.zeros((H + banner_h, target_w, 3), dtype=np.uint8)

        p_panel_a[:banner_h, :] = (30, 30, 30)
        p_panel_b[:banner_h, :] = (30, 30, 30)
        p_panel_mat[:banner_h, :] = (30, 30, 30)

        p_panel_a[banner_h:, :] = ov_a
        p_panel_b[banner_h:, :] = ov_b
        p_panel_mat[banner_h:, :] = mat_resized

        cv2.putText(p_panel_a, f"{stem_a} ({len(ids_a)} classes)", (40, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100, 200, 255), 3, cv2.LINE_AA)
        cv2.putText(p_panel_b, f"{stem_b} ({len(ids_b)} classes)", (40, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100, 255, 150), 3, cv2.LINE_AA)
        cv2.putText(p_panel_mat, f"CROSS-FRAME SIMILARITY MATRIX", (30, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 100), 2, cv2.LINE_AA)

        divider = np.full((H + banner_h, 8, 3), 255, dtype=np.uint8)
        composite = np.hstack([p_panel_a, divider, p_panel_b, divider, p_panel_mat])

        comp_path = out_dir / f"{pair_name}_comparison_3col.png"
        cv2.imwrite(str(comp_path), composite)

        # Save NPZ and JSON record
        npz_out = out_dir / f"{pair_name}_data.npz"
        np.savez_compressed(
            npz_out,
            similarity_matrix=sim_mat,
            classes_a=np.array(ids_a),
            classes_b=np.array(ids_b)
        )

        dur = time.perf_counter() - t0
        record = {
            'pair_idx': idx,
            'pair_name': pair_name,
            'frame_a': stem_a,
            'frame_b': stem_b,
            'num_classes_a': len(ids_a),
            'num_classes_b': len(ids_b),
            'matrix_shape': [len(ids_a), len(ids_b)],
            'max_similarity': round(float(sim_mat.max()), 4),
            'min_similarity': round(float(sim_mat.min()), 4),
            'mean_similarity': round(float(sim_mat.mean()), 4),
            'best_matches_a_to_b': best_a_to_b,
            'time_sec': round(dur, 3)
        }
        summary_records.append(record)

        with open(out_dir / f"{pair_name}_data.json", 'w') as f:
            json.dump(record, f, indent=2)

        if (idx + 1) % 5 == 0 or idx == num_pairs - 1:
            print(f"Processed pair {idx + 1:02d}/{num_pairs}: {stem_a} vs {stem_b} ({dur:.2f}s) | Max Sim: {sim_mat.max():.4f}")

    # Save grand summary
    with open(out_dir / "summary_cross_frame_pairs.json", 'w') as f:
        json.dump({
            'num_pairs': num_pairs,
            'pairs': summary_records
        }, f, indent=2)

    print("\n=================================================================")
    print(f"SUCCESS! All {num_pairs} pairs rendered and saved to:")
    print(f"{out_dir}")
    print("=================================================================")

if __name__ == '__main__':
    main()
