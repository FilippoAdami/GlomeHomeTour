#!/usr/bin/env python3
"""
2D Multi-Image ID Propagation with Dual-Threshold Matching and Crisp Visual Overlays:
- Base threshold for primary match: >= 0.690
- Multi-claim threshold from 2nd matched class onward: >= 0.705

Visual Enhancements:
- Clean, robust label badge placement via Distance Transform interior anchoring.
- High-contrast badge styling: Dark charcoal pill with class-colored double border and bold white text.
- Guaranteed in-bounds clamping and no contour/badge misalignment.
"""

import os, sys, json, time
from pathlib import Path
import cv2, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_THRESHOLD = 0.690
SECONDARY_THRESHOLD = 0.705

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
    Finds the optimal interior point for badge placement using Distance Transform.
    Prefers the deep interior of the largest connected component and penalizes image borders.
    """
    nb_cc, cc_labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    if nb_cc <= 1:
        return W // 2, H // 2

    largest_c = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    comp = (cc_labels == largest_c).astype(np.uint8)

    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)

    # Soft margin penalty to avoid drawing right against the canvas border
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
    and crisp, perfectly readable centered ID badges.
    """
    bgr = cv2.imread(str(img_path))
    H, W = bgr.shape[:2]
    overlay = bgr.copy()

    # 1. Semi-transparent mask tint
    for mid, m in zip(mask_ids, masks):
        col_bgr, _ = get_class_color(mid)
        col_arr = np.array(col_bgr, dtype=np.uint8)
        vis = m > 0
        overlay[vis] = (0.55 * overlay[vis] + 0.45 * col_arr).astype(np.uint8)

    # Background / unsegmented pixels to black
    overlay[labels == 0] = 0

    # 2. Draw thick crisp contours
    for mid, m in zip(mask_ids, masks):
        col_bgr, _ = get_class_color(mid)
        mask_u8 = m.astype(np.uint8)
        if not mask_u8.any():
            continue
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, col_bgr, 3)

    # 3. Draw high-contrast text badges
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

        # Clamped bounding box
        bx1 = max(10, min(W - tw - 2 * pad_x - 10, cx - tw // 2 - pad_x))
        by1 = max(10, min(H - th - 2 * pad_y - 10, cy - th // 2 - pad_y))
        bx2 = bx1 + tw + 2 * pad_x
        by2 = by1 + th + 2 * pad_y

        # Badge pill background & colored frame
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), col_bgr, 3)

        # Crisp centered text
        text_x = bx1 + pad_x
        text_y = by1 + pad_y + th
        cv2.putText(overlay, txt, (text_x, text_y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    return overlay

def create_cross_matrix_plot(sim_mat: np.ndarray, ids_a: list, ids_b: list,
                             propagated_links: list, name_a: str, name_b: str, out_path: Path):
    Na = len(ids_a)
    Nb = len(ids_b)

    fig_w = max(7.5, Nb * 0.95 + 3.2)
    fig_h = max(7.0, Na * 0.95 + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=130)
    cmap = plt.cm.YlGnBu

    matched_set = {(m['idx_a'], m['idx_b']) for m in propagated_links}

    for i in range(Na):
        for j in range(Nb):
            val = float(sim_mat[i, j])
            c = cmap((val + 0.1) / 1.1)
            is_propagated = (i, j) in matched_set
            edge_col = '#e74c3c' if is_propagated else '#dddddd'
            line_w = 3.2 if is_propagated else 0.8
            rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     facecolor=c, edgecolor=edge_col, linewidth=line_w)
            ax.add_patch(rect)
            lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
            fontweight = 'bold' if (is_propagated or val >= BASE_THRESHOLD) else 'normal'
            txt_color = 'white' if lum < 0.55 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=txt_color, fontsize=11, fontweight=fontweight)

    ax.set_xlim(-0.5, Nb - 0.5)
    ax.set_ylim(Na - 0.5, -0.5)
    ax.set_xticks(range(Nb))
    ax.set_yticks(range(Na))
    ax.set_xticklabels([f"#{mid}" for mid in ids_b], fontsize=12)
    ax.set_yticklabels([f"#{mid}" for mid in ids_a], fontsize=12)
    ax.xaxis.tick_top()
    ax.set_xlabel(f'{name_b} Propagated IDs (Cols)', fontsize=13, fontweight='bold', labelpad=14)
    ax.set_ylabel(f'{name_a} Propagated IDs (Rows)', fontsize=13, fontweight='bold', labelpad=14)
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
    plt.title(f'Cross-Frame Propagation (Red = Propagated; Base >= {BASE_THRESHOLD}, Multi >= {SECONDARY_THRESHOLD})\n{name_a} -> {name_b}',
              fontsize=13, fontweight='bold', pad=36)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "current_scene" / "06c_semantic_segmentation" / "first50_streamlined_grid12_merge069"
    img_dir = root / "current_scene" / "06c_semantic_segmentation" / "inputs" / "images"
    out_dir = root / "current_scene" / "06c_semantic_segmentation" / "first50_id_propagated"
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_vis_dir = out_dir / "pairwise_transitions"
    pair_vis_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(data_dir.glob("frame_*_merged.npz"))
    num_frames = len(npz_files)

    print("=================================================================")
    print("2D MULTI-IMAGE PROPAGATION WITH CRISP VISUALS")
    print(f"Base Threshold: >={BASE_THRESHOLD} | Multi-Claim Threshold: >={SECONDARY_THRESHOLD}")
    print(f"Total Frames: {num_frames}")
    print(f"Input: {data_dir}")
    print(f"Output: {out_dir}")
    print("=================================================================\n")

    # 1. Initialize Frame 0
    p0 = npz_files[0]
    stem0 = p0.stem.replace('_merged', '')
    d0 = np.load(p0)
    masks0 = d0['masks']
    H, W = masks0.shape[1], masks0.shape[2]
    N0 = len(masks0)

    # Frame 0 ordered IDs [1, 2, ..., N0]
    curr_ids = list(range(1, N0 + 1))
    next_available_id = N0 + 1
    curr_embeddings = d0['embeddings'].copy()
    curr_areas = np.array([m.sum() for m in masks0], dtype=np.float32)

    labels0 = np.zeros((H, W), dtype=np.int32)
    for mid, m in zip(curr_ids, masks0):
        labels0[m > 0] = mid

    np.savez_compressed(
        out_dir / f"{stem0}_propagated.npz",
        masks=masks0,
        labels=labels0,
        embeddings=curr_embeddings,
        ids=np.array(curr_ids)
    )

    ov0 = create_mask_overlay(img_dir / f"{stem0}.jpg", masks0, labels0, curr_ids)
    cv2.imwrite(str(out_dir / f"{stem0}_overlay.png"), ov0)

    with open(out_dir / f"{stem0}_propagated.json", 'w') as f:
        json.dump({
            'frame': stem0,
            'frame_idx': 0,
            'num_masks': N0,
            'ids': curr_ids,
            'propagated_from_prev': 0,
            'new_objects': N0,
            'matches': []
        }, f, indent=2)

    prev_stem = stem0
    prev_ov = ov0
    prev_masks = masks0
    prev_labels = labels0

    propagation_history = []
    total_propagated_matches = 0
    total_new_objects = N0

    for step in range(len(npz_files) - 1):
        t0 = time.perf_counter()
        p_next = npz_files[step + 1]
        stem_next = p_next.stem.replace('_merged', '')

        d_next = np.load(p_next)
        next_masks = d_next['masks']
        next_embeddings = d_next['embeddings'].copy()
        next_areas = np.array([m.sum() for m in next_masks], dtype=np.float32)
        N_next = len(next_masks)

        # Cross-frame similarity matrix: (N_curr, N_next)
        sim_mat = curr_embeddings @ next_embeddings.T

        # Gather candidates >= BASE_THRESHOLD (0.690)
        candidates = []
        for i in range(len(curr_ids)):
            for j in range(N_next):
                val = float(sim_mat[i, j])
                if val >= BASE_THRESHOLD:
                    candidates.append((val, i, j))

        # Sort globally descending by similarity
        candidates.sort(key=lambda x: x[0], reverse=True)

        matched_targets = set()
        claims_by_i = {i: [] for i in range(len(curr_ids))}

        for val, i, j in candidates:
            if j in matched_targets:
                continue
            # Primary match: first target for class i (requires val >= 0.690)
            if len(claims_by_i[i]) == 0:
                claims_by_i[i].append((j, val))
                matched_targets.add(j)
            # Secondary+ match: additional targets for class i (requires val >= 0.705)
            elif val >= SECONDARY_THRESHOLD:
                claims_by_i[i].append((j, val))
                matched_targets.add(j)

        # Assign IDs to frame i+1
        next_ids = [None] * N_next
        updated_embeddings = next_embeddings.copy()
        step_matches = []

        for i, claims in claims_by_i.items():
            if not claims:
                continue
            cid = curr_ids[i]
            # Accumulate area-weighted vector for class #cid
            sum_vec = curr_areas[i] * curr_embeddings[i]
            for j, sim in claims:
                next_ids[j] = cid
                sum_vec = sum_vec + next_areas[j] * next_embeddings[j]
                step_matches.append({
                    'idx_a': i,
                    'idx_b': j,
                    'class_id': cid,
                    'similarity': round(sim, 4),
                    'is_primary': (j == claims[0][0])
                })

            # Normalized unit vector
            unit_vec = sum_vec / max(np.linalg.norm(sum_vec), 1e-12)
            for j, _ in claims:
                updated_embeddings[j] = unit_vec

        # Unclaimed masks receive next globally available ID
        new_in_frame = 0
        for j in range(N_next):
            if next_ids[j] is None:
                next_ids[j] = next_available_id
                next_available_id += 1
                new_in_frame += 1

        total_propagated_matches += len(step_matches)
        total_new_objects += new_in_frame

        # Build labels map
        next_labels = np.zeros((H, W), dtype=np.int32)
        for mid, m in zip(next_ids, next_masks):
            next_labels[m > 0] = mid

        # Save NPZ
        np.savez_compressed(
            out_dir / f"{stem_next}_propagated.npz",
            masks=next_masks,
            labels=next_labels,
            embeddings=updated_embeddings,
            ids=np.array(next_ids)
        )

        # Overlay with clean badges
        next_ov = create_mask_overlay(img_dir / f"{stem_next}.jpg", next_masks, next_labels, next_ids)
        cv2.imwrite(str(out_dir / f"{stem_next}_overlay.png"), next_ov)

        # Save Pairwise 3-Column Diagnostic Sheet
        pair_name = f"step_{step:02d}_{prev_stem}_to_{stem_next}"
        mat_plot_path = pair_vis_dir / f"{pair_name}_similarity_matrix.png"
        create_cross_matrix_plot(sim_mat, curr_ids, next_ids, step_matches,
                                 prev_stem, stem_next, mat_plot_path)

        mat_img = cv2.imread(str(mat_plot_path))
        mat_aspect = mat_img.shape[1] / mat_img.shape[0]
        target_w = int(H * mat_aspect)
        mat_resized = cv2.resize(mat_img, (target_w, H), interpolation=cv2.INTER_AREA)

        banner_h = 70
        p_panel_a = np.zeros((H + banner_h, W, 3), dtype=np.uint8)
        p_panel_b = np.zeros((H + banner_h, W, 3), dtype=np.uint8)
        p_panel_mat = np.zeros((H + banner_h, target_w, 3), dtype=np.uint8)

        p_panel_a[:banner_h, :] = (30, 30, 30)
        p_panel_b[:banner_h, :] = (30, 30, 30)
        p_panel_mat[:banner_h, :] = (30, 30, 30)

        p_panel_a[banner_h:, :] = prev_ov
        p_panel_b[banner_h:, :] = next_ov
        p_panel_mat[banner_h:, :] = mat_resized

        cv2.putText(p_panel_a, f"{prev_stem} ({len(curr_ids)} classes)", (40, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100, 200, 255), 3, cv2.LINE_AA)
        cv2.putText(p_panel_b, f"{stem_next} ({len(next_ids)} classes, {len(step_matches)} propagated)", (40, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100, 255, 150), 3, cv2.LINE_AA)
        cv2.putText(p_panel_mat, f"SIMILARITY MATRIX (Red = Propagated)", (30, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 100), 2, cv2.LINE_AA)

        divider = np.full((H + banner_h, 8, 3), 255, dtype=np.uint8)
        composite = np.hstack([p_panel_a, divider, p_panel_b, divider, p_panel_mat])
        cv2.imwrite(str(pair_vis_dir / f"{pair_name}_comparison_3col.png"), composite)

        # JSON record
        dur = time.perf_counter() - t0
        step_record = {
            'step': step,
            'prev_frame': prev_stem,
            'next_frame': stem_next,
            'prev_ids': curr_ids,
            'next_ids': next_ids,
            'num_matches': len(step_matches),
            'num_new_objects': new_in_frame,
            'matches': step_matches,
            'time_sec': round(dur, 3)
        }
        propagation_history.append(step_record)

        with open(out_dir / f"{stem_next}_propagated.json", 'w') as f:
            json.dump({
                'frame': stem_next,
                'frame_idx': step + 1,
                'num_masks': N_next,
                'ids': next_ids,
                'propagated_from_prev': len(step_matches),
                'new_objects': new_in_frame,
                'matches': step_matches
            }, f, indent=2)

        # Advance state
        curr_ids = next_ids
        curr_embeddings = updated_embeddings
        curr_areas = next_areas
        prev_stem = stem_next
        prev_ov = next_ov

        if (step + 1) % 5 == 0 or step == len(npz_files) - 2:
            print(f"Step {step + 1:02d}/49: {prev_stem} -> {stem_next} ({dur:.2f}s) | Matches: {len(step_matches)}/{N_next} | Next Available ID: {next_available_id}")

    # Grand Summary
    summary = {
        'total_frames': num_frames,
        'base_threshold': BASE_THRESHOLD,
        'secondary_threshold': SECONDARY_THRESHOLD,
        'initial_ids_frame0': N0,
        'final_max_id': next_available_id - 1,
        'total_propagated_links': total_propagated_matches,
        'total_new_objects_introduced': total_new_objects,
        'steps': propagation_history
    }
    with open(out_dir / "propagation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=================================================================")
    print("PROPAGATION AND RE-RENDERING COMPLETE!")
    print(f"Total Unique Object IDs: {next_available_id - 1}")
    print(f"Total Propagated Links: {total_propagated_matches}")
    print(f"Outputs written to: {out_dir}")
    print(f"Pairwise 3-Column Diagnostic sheets: {pair_vis_dir}")
    print("=================================================================")

if __name__ == '__main__':
    main()
