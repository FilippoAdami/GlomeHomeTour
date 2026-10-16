#!/usr/bin/env python3
"""
Global Memory Bank ID Propagation with:
1. Dormancy-Adaptive Base Threshold:
   - Base threshold = 0.690 for recently seen classes.
   - Increases by +0.01 for every 7 frames of absence, capped at max 0.730:
     thresh(Δt) = min(0.730, 0.690 + 0.01 * (Δt // 7))
   - Secondary multi-claim threshold is fixed at 0.710 and never increased.
2. Adaptive Decay Gamma by Object Area:
   - Large structure (> 15% image area): gamma = 0.95
   - Mid-size objects (3% - 15% image area): gamma = 0.90
   - Small / compact foreground (< 3% image area): gamma = 0.82
3. Diagnostic 3-Column Transition Sheets (filtered rows >= 0.630).
4. End-of-Pass Triangular Similarity Matrix:
   - Full upper-triangular similarity matrix across all final canonical class embeddings.
   - Labels show: #ID (N_frames, avg_gap) where avg_gap is the mean interval between appearances.
"""

import os, sys, json, time
from pathlib import Path
import cv2, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_THRESHOLD = 0.690
MAX_BASE_THRESHOLD = 0.730
DORMANCY_STEP_FRAMES = 7
DORMANCY_STEP_VAL = 0.01

SECONDARY_THRESHOLD = 0.710
VIS_ROW_THRESHOLD = 0.630

def get_adaptive_gamma(area_ratio: float) -> float:
    """
    Scale-dependent decay gamma based on current mask fraction of the full image.
    """
    if area_ratio >= 0.15:
        return 0.95
    elif area_ratio >= 0.03:
        return 0.90
    else:
        return 0.82

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

    # Mask tint
    for mid, m in zip(mask_ids, masks):
        col_bgr, _ = get_class_color(mid)
        col_arr = np.array(col_bgr, dtype=np.uint8)
        vis = m > 0
        overlay[vis] = (0.55 * overlay[vis] + 0.45 * col_arr).astype(np.uint8)

    # Background black
    overlay[labels == 0] = 0

    # Draw contours
    for mid, m in zip(mask_ids, masks):
        col_bgr, _ = get_class_color(mid)
        mask_u8 = m.astype(np.uint8)
        if not mask_u8.any():
            continue
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, col_bgr, 3)

    # Draw badges
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

def create_filtered_matrix_plot(sim_mat: np.ndarray, bank_ids: list, next_ids: list,
                                bank_stats: dict, propagated_links: list,
                                name_prev: str, name_next: str, out_path: Path):
    """
    Renders similarity matrix filtered to show rows with at least one sim >= VIS_ROW_THRESHOLD.
    """
    K_total, Nb = sim_mat.shape

    matched_row_indices = {m['bank_idx'] for m in propagated_links}
    row_mask = (sim_mat.max(axis=1) >= VIS_ROW_THRESHOLD)
    for r_idx in matched_row_indices:
        row_mask[r_idx] = True

    vis_row_indices = [i for i in range(K_total) if row_mask[i]]
    if len(vis_row_indices) == 0:
        vis_row_indices = list(np.argsort(sim_mat.max(axis=1))[-min(5, K_total):])

    sub_sim = sim_mat[vis_row_indices, :]
    sub_bank_ids = [bank_ids[i] for i in vis_row_indices]
    Na_vis = len(sub_bank_ids)

    fig_w = max(7.5, Nb * 0.95 + 3.2)
    fig_h = max(7.0, Na_vis * 0.95 + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=130)
    cmap = plt.cm.YlGnBu

    matched_pairs = {(m['bank_idx'], m['target_idx']) for m in propagated_links}

    for sub_i, orig_i in enumerate(vis_row_indices):
        for j in range(Nb):
            val = float(sub_sim[sub_i, j])
            c = cmap((val + 0.1) / 1.1)
            is_propagated = (orig_i, j) in matched_pairs
            edge_col = '#e74c3c' if is_propagated else '#dddddd'
            line_w = 3.2 if is_propagated else 0.8
            rect = patches.Rectangle((j - 0.5, sub_i - 0.5), 1, 1,
                                     facecolor=c, edgecolor=edge_col, linewidth=line_w)
            ax.add_patch(rect)
            lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
            fontweight = 'bold' if (is_propagated or val >= BASE_THRESHOLD) else 'normal'
            txt_color = 'white' if lum < 0.55 else 'black'
            ax.text(j, sub_i, f'{val:.2f}', ha='center', va='center',
                    color=txt_color, fontsize=11, fontweight=fontweight)

    ax.set_xlim(-0.5, Nb - 0.5)
    ax.set_ylim(Na_vis - 0.5, -0.5)
    ax.set_xticks(range(Nb))
    ax.set_yticks(range(Na_vis))
    ax.set_xticklabels([f"#{mid}" for mid in next_ids], fontsize=12)

    row_labels = []
    for mid in sub_bank_ids:
        cnt = bank_stats[mid]['frames_seen']
        row_labels.append(f"#{mid} (x{cnt})")
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.xaxis.tick_top()
    ax.set_xlabel(f'{name_next} Proposals (Cols)', fontsize=13, fontweight='bold', labelpad=14)
    ax.set_ylabel(f'Memory Bank Classes (Rows >= {VIS_ROW_THRESHOLD})', fontsize=12, fontweight='bold', labelpad=14)
    ax.xaxis.set_label_position('top')

    for t_idx, tick in enumerate(ax.get_xticklabels()):
        _, col_rgb = get_class_color(next_ids[t_idx])
        tick.set_color(col_rgb)
        tick.set_fontweight('bold')

    for t_idx, tick in enumerate(ax.get_yticklabels()):
        _, col_rgb = get_class_color(sub_bank_ids[t_idx])
        tick.set_color(col_rgb)
        tick.set_fontweight('bold')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('DINOv2 Cosine Similarity', fontsize=12)
    plt.title(f'Global Memory Matching (Red = Propagated)\n{name_prev} -> {name_next} ({Na_vis}/{K_total} classes shown)',
              fontsize=12, fontweight='bold', pad=36)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def create_end_of_pass_triangular_matrix(classes_dict: dict, out_path: Path):
    """
    Renders the full upper-triangular similarity matrix between all final discovered classes.
    Row and column labels format: #ID (N_frames, avg_gap)
    """
    sorted_ids = sorted(classes_dict.keys())
    K = len(sorted_ids)
    if K == 0:
        return

    # Stack final unit embeddings
    units = np.stack([classes_dict[mid]['unit_vec'] for mid in sorted_ids])
    sim_mat = units @ units.T

    # Compute labels with (N_frames, avg_gap)
    labels = []
    for mid in sorted_ids:
        entry = classes_dict[mid]
        n_frames = entry['frames_seen']
        # Calculate gaps
        frames_idx = sorted(entry['frame_indices'])
        if len(frames_idx) <= 1:
            avg_gap = 0.0
        else:
            gaps = [frames_idx[k + 1] - frames_idx[k] for k in range(len(frames_idx) - 1)]
            avg_gap = float(np.mean(gaps))
        labels.append(f"#{mid} ({n_frames}, {avg_gap:.1f})")

    # Figure sizing: dynamically proportional to number of classes
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
    plt.title(f'End-of-Pass Cross-Class Triangular Similarity Matrix (K={K} Classes)\nLabels: #ID (N_frames, avg_gap)',
              fontsize=13, fontweight='bold', pad=45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

class GlobalMemoryBank:
    """
    Maintains persistent canonical class representations with scale-adaptive decay
    and dormancy-adaptive matching thresholds.
    """
    def __init__(self):
        self.classes = {} # class_id -> dict
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
                'frame_indices': [frame_idx],
                'history': [{'frame': frame_name, 'frame_idx': frame_idx, 'area': area}]
            }
        return assigned_ids

    def get_bank_matrix(self):
        bank_ids = sorted(self.classes.keys())
        bank_units = np.stack([self.classes[mid]['unit_vec'] for mid in bank_ids])
        return bank_ids, bank_units

    def get_dormancy_threshold(self, mid: int, current_frame_idx: int) -> float:
        """
        Calculates the primary matching threshold based on dormancy:
        thresh = min(0.730, 0.690 + 0.01 * (delta_t // 7))
        where delta_t is the number of frames since this class was last observed.
        """
        last_idx = self.classes[mid]['last_seen_idx']
        delta_t = max(0, current_frame_idx - last_idx)
        step_inc = (delta_t // DORMANCY_STEP_FRAMES) * DORMANCY_STEP_VAL
        return min(MAX_BASE_THRESHOLD, BASE_THRESHOLD + step_inc)

    def update_matches(self, claims_by_bank_id: dict, next_masks: np.ndarray,
                       next_embeddings: np.ndarray, frame_name: str, frame_idx: int, H: int, W: int):
        total_pixels = float(H * W)
        for mid, claims in claims_by_bank_id.items():
            entry = self.classes[mid]

            # 1. Calculate incoming observation area and determine adaptive gamma
            inc_area = 0.0
            inc_sum = np.zeros_like(entry['sum_vec'])
            for j, _ in claims:
                a_j = float(next_masks[j].sum())
                inc_area += a_j
                inc_sum += a_j * next_embeddings[j]

            area_ratio = inc_area / max(total_pixels, 1.0)
            gamma = get_adaptive_gamma(area_ratio)

            # 2. Scale-adaptive decay
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
            entry['history'].append({'frame': frame_name, 'frame_idx': frame_idx, 'area': inc_area, 'gamma': gamma})

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
                'frame_indices': [frame_idx],
                'history': [{'frame': frame_name, 'frame_idx': frame_idx, 'area': area, 'gamma': 1.0}]
            }
            new_assignments[j] = mid
        return new_assignments

def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "current_scene" / "06c_semantic_segmentation" / "first50_streamlined_grid12_merge069"
    img_dir = root / "current_scene" / "06c_semantic_segmentation" / "inputs" / "images"
    out_dir = root / "current_scene" / "06c_semantic_segmentation" / "first50_global_memory_bank"
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_vis_dir = out_dir / "pairwise_transitions"
    pair_vis_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(data_dir.glob("frame_*_merged.npz"))
    num_frames = len(npz_files)

    print("=================================================================")
    print("GLOBAL MEMORY BANK PROPAGATION (DORMANCY-ADAPTIVE & SCALE-DECAY)")
    print(f"Base Thresh: {BASE_THRESHOLD} (grows +0.01/7 frames up to {MAX_BASE_THRESHOLD})")
    print(f"Fixed Secondary Thresh: {SECONDARY_THRESHOLD} | Row Filter: >= {VIS_ROW_THRESHOLD}")
    print(f"Total Frames: {num_frames}")
    print(f"Input: {data_dir}")
    print(f"Output: {out_dir}")
    print("=================================================================\n")

    bank = GlobalMemoryBank()

    # 1. Initialize Frame 0
    p0 = npz_files[0]
    stem0 = p0.stem.replace('_merged', '')
    d0 = np.load(p0)
    masks0 = d0['masks']
    H, W = masks0.shape[1], masks0.shape[2]
    emb0 = d0['embeddings']

    init_ids_0 = bank.register_initial_frame(masks0, emb0, stem0, frame_idx=0)

    labels0 = np.zeros((H, W), dtype=np.int32)
    for mid, m in zip(init_ids_0, masks0):
        labels0[m > 0] = mid

    np.savez_compressed(
        out_dir / f"{stem0}_propagated.npz",
        masks=masks0,
        labels=labels0,
        embeddings=emb0,
        ids=np.array(init_ids_0)
    )

    ov0 = create_mask_overlay(img_dir / f"{stem0}.jpg", masks0, labels0, init_ids_0)
    cv2.imwrite(str(out_dir / f"{stem0}_overlay.png"), ov0)

    with open(out_dir / f"{stem0}_propagated.json", 'w') as f:
        json.dump({
            'frame': stem0,
            'frame_idx': 0,
            'num_masks': len(init_ids_0),
            'ids': init_ids_0,
            'propagated_from_memory': 0,
            'new_objects': len(init_ids_0),
            'matches': []
        }, f, indent=2)

    prev_stem = stem0
    prev_ov = ov0
    propagation_history = []
    total_propagated_links = 0
    total_new_objects = len(init_ids_0)

    for step in range(len(npz_files) - 1):
        t0 = time.perf_counter()
        current_frame_idx = step + 1
        p_next = npz_files[current_frame_idx]
        stem_next = p_next.stem.replace('_merged', '')

        d_next = np.load(p_next)
        next_masks = d_next['masks']
        next_embeddings = d_next['embeddings'].copy()
        N_next = len(next_masks)

        # 1. Retrieve current global memory bank matrix
        bank_ids, bank_units = bank.get_bank_matrix()
        K_bank = len(bank_ids)

        # 2. Similarity against global memory: (K_bank, N_next)
        sim_mat = bank_units @ next_embeddings.T

        # 3. Gather candidate pairs >= BASE_THRESHOLD (0.690)
        candidates = []
        for i in range(K_bank):
            for j in range(N_next):
                val = float(sim_mat[i, j])
                if val >= BASE_THRESHOLD:
                    candidates.append((val, i, j))

        candidates.sort(key=lambda x: x[0], reverse=True)

        matched_targets = set()
        claims_by_bank_idx = {i: [] for i in range(K_bank)}

        for val, i, j in candidates:
            if j in matched_targets:
                continue
            mid = bank_ids[i]
            # Primary match: depends on dormancy of class mid
            if len(claims_by_bank_idx[i]) == 0:
                dormancy_thresh = bank.get_dormancy_threshold(mid, current_frame_idx)
                if val >= dormancy_thresh:
                    claims_by_bank_idx[i].append((j, val, dormancy_thresh))
                    matched_targets.add(j)
            # Secondary+ match: fixed threshold 0.710
            elif val >= SECONDARY_THRESHOLD:
                claims_by_bank_idx[i].append((j, val, SECONDARY_THRESHOLD))
                matched_targets.add(j)

        # Assign IDs to current frame
        next_ids = [None] * N_next
        step_matches = []
        claims_by_mid = {}

        for i, claims in claims_by_bank_idx.items():
            if not claims:
                continue
            mid = bank_ids[i]
            claims_by_mid[mid] = [(j, sim) for j, sim, _ in claims]
            for j, sim, req_thresh in claims:
                next_ids[j] = mid
                step_matches.append({
                    'bank_idx': i,
                    'target_idx': j,
                    'class_id': mid,
                    'similarity': round(sim, 4),
                    'required_threshold': round(req_thresh, 4),
                    'is_primary': (j == claims[0][0])
                })

        # Update matched classes in memory bank (scale-adaptive decay)
        bank.update_matches(claims_by_mid, next_masks, next_embeddings, stem_next, current_frame_idx, H, W)

        # Register unclaimed proposals as new classes in memory bank
        unclaimed_j = [j for j in range(N_next) if next_ids[j] is None]
        new_assignments = bank.register_new_classes(unclaimed_j, next_masks, next_embeddings, stem_next, current_frame_idx)
        for j, new_mid in new_assignments.items():
            next_ids[j] = new_mid

        total_propagated_links += len(step_matches)
        total_new_objects += len(unclaimed_j)

        # Final embeddings for saving in frame
        frame_embeddings = np.stack([bank.classes[mid]['unit_vec'] for mid in next_ids])

        # Build labels map
        next_labels = np.zeros((H, W), dtype=np.int32)
        for mid, m in zip(next_ids, next_masks):
            next_labels[m > 0] = mid

        # Save NPZ
        np.savez_compressed(
            out_dir / f"{stem_next}_propagated.npz",
            masks=next_masks,
            labels=next_labels,
            embeddings=frame_embeddings,
            ids=np.array(next_ids)
        )

        # Save overlay with crisp badges
        next_ov = create_mask_overlay(img_dir / f"{stem_next}.jpg", next_masks, next_labels, next_ids)
        cv2.imwrite(str(out_dir / f"{stem_next}_overlay.png"), next_ov)

        # Save 3-column diagnostic sheet with filtered similarity matrix
        pair_name = f"step_{step:02d}_{prev_stem}_to_{stem_next}"
        mat_plot_path = pair_vis_dir / f"{pair_name}_similarity_matrix.png"
        create_filtered_matrix_plot(sim_mat, bank_ids, next_ids, bank.classes,
                                    step_matches, prev_stem, stem_next, mat_plot_path)

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

        cv2.putText(p_panel_a, f"{prev_stem}", (40, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100, 200, 255), 3, cv2.LINE_AA)
        cv2.putText(p_panel_b, f"{stem_next} ({len(step_matches)} from Memory Bank)", (40, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100, 255, 150), 3, cv2.LINE_AA)
        cv2.putText(p_panel_mat, f"MEMORY SIMILARITY (Rows >= {VIS_ROW_THRESHOLD})", (30, 48),
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
            'bank_size_before': K_bank,
            'next_ids': next_ids,
            'num_matches': len(step_matches),
            'num_new_objects': len(unclaimed_j),
            'matches': step_matches,
            'time_sec': round(dur, 3)
        }
        propagation_history.append(step_record)

        with open(out_dir / f"{stem_next}_propagated.json", 'w') as f:
            json.dump({
                'frame': stem_next,
                'frame_idx': current_frame_idx,
                'num_masks': N_next,
                'ids': next_ids,
                'propagated_from_memory': len(step_matches),
                'new_objects': len(unclaimed_j),
                'matches': step_matches
            }, f, indent=2)

        prev_stem = stem_next
        prev_ov = next_ov

        if (step + 1) % 5 == 0 or step == len(npz_files) - 2:
            print(f"Step {step + 1:02d}/49: {prev_stem} -> {stem_next} ({dur:.2f}s) | Matches: {len(step_matches)}/{N_next} | Bank Size: {len(bank.classes)}")

    # 4. Generate End-of-Pass Triangular Matrix
    print("\nGenerating End-of-Pass Cross-Class Triangular Similarity Matrix...")
    triangular_out_path = out_dir / "final_cross_class_similarity_matrix.png"
    create_end_of_pass_triangular_matrix(bank.classes, triangular_out_path)
    print(f"Triangular matrix saved to: {triangular_out_path}")

    # Grand Summary & Class Lifespan Statistics
    class_summary = {}
    for mid, entry in bank.classes.items():
        frames_idx = sorted(entry['frame_indices'])
        avg_gap = 0.0 if len(frames_idx) <= 1 else float(np.mean([frames_idx[k+1] - frames_idx[k] for k in range(len(frames_idx)-1)]))
        class_summary[mid] = {
            'id': mid,
            'frames_seen': entry['frames_seen'],
            'raw_area': entry['raw_area'],
            'eff_area': entry['eff_area'],
            'first_seen': entry['first_seen'],
            'last_seen': entry['last_seen'],
            'avg_gap': round(avg_gap, 2)
        }

    summary = {
        'total_frames': num_frames,
        'base_threshold': BASE_THRESHOLD,
        'max_base_threshold': MAX_BASE_THRESHOLD,
        'dormancy_step_frames': DORMANCY_STEP_FRAMES,
        'dormancy_step_val': DORMANCY_STEP_VAL,
        'secondary_threshold': SECONDARY_THRESHOLD,
        'vis_row_threshold': VIS_ROW_THRESHOLD,
        'initial_ids_frame0': len(init_ids_0),
        'final_total_classes': len(bank.classes),
        'total_propagated_links': total_propagated_links,
        'total_new_objects_introduced': total_new_objects,
        'classes': class_summary,
        'steps': propagation_history
    }
    with open(out_dir / "global_memory_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=================================================================")
    print("GLOBAL MEMORY BANK PROPAGATION COMPLETE!")
    print(f"Total Unique Object IDs Discovered Across 50 Frames: {len(bank.classes)}")
    print(f"Total Successful Cross-Frame Propagations: {total_propagated_links}")
    print(f"Outputs written to: {out_dir}")
    print(f"Triangular Matrix: {triangular_out_path}")
    print(f"Pairwise 3-Column Diagnostic sheets: {pair_vis_dir}")
    print("=================================================================")

if __name__ == '__main__':
    main()
