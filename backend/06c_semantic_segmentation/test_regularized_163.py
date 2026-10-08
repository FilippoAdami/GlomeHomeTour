import numpy as np, cv2

npz_163 = np.load('backend/current_scene/06c_semantic_segmentation/first50_streamlined_grid12_merge069/frame_00163_merged.npz')
lbls = npz_163['labels']

def regularize_carved_mask(m: np.ndarray, min_chunk_area: int = 10368, min_thickness: float = 12.0) -> np.ndarray:
    """
    Two-tier regularization:
    1. Component-level floater pruning: Discards small detached satellite pieces / edge crumbs
       (< 1,000 px or < 5% of mask mass).
    2. Ribbon/tendril pruning: For surviving components with thickness < 25px, applies morphological
       opening (7x7 ellipse) to strip thin border tendrils, keeping solid chunky bodies (>= min_chunk_area, thickness >= 12px).
    """
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

    # Tier 2: Check thickness on surviving chunks (downsampled 4x)
    H, W = m.shape
    m_small = cv2.resize(m, (W // 4, H // 4), interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(m_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    peri = sum(cv2.arcLength(c, True) for c in cnts) * 4
    thick = 2 * m.sum() / max(peri, 1)

    # Solid thick mask: keep directly
    if thick >= 25.0:
        return m

    # Thin/fragmented mask: strip tendrils via morphological opening
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

for target in [9, 15]:
    m = (lbls == target).astype(np.uint8)
    reg = regularize_carved_mask(m)
    nb_before = cv2.connectedComponentsWithStats(m, 8)[0] - 1
    nb_after = cv2.connectedComponentsWithStats(reg, 8)[0] - 1
    print(f"Mask #{target}: {nb_before} components -> {nb_after} component(s)! (Area: {m.sum()} -> {reg.sum()})")
