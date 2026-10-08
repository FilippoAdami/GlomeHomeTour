#!/usr/bin/env python3
"""Run and save a reproducible single-frame SAM2 proposal configuration."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from fast_sam2_generator import FastSAM2Generator
import torch


DEFAULT_WORKSPACE = Path(__file__).resolve().parents[1] / "current_scene" / "06c_semantic_segmentation"


def guided_filter_rgb(guide: np.ndarray, src: np.ndarray, radius: int = 4, eps: float = 1e-3) -> np.ndarray:
    """Refine single-channel mask logits using 3-channel RGB guide image."""
    guide = guide.astype(np.float32) / 255.0
    p = src.astype(np.float32)
    ksize = (2 * radius + 1, 2 * radius + 1)

    mean_I = cv2.boxFilter(guide, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)

    # Compute covariances
    mean_Ip = np.zeros_like(mean_I)
    for c in range(3):
        mean_Ip[..., c] = cv2.boxFilter(guide[..., c] * p, cv2.CV_32F, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p[..., None]

    # Variance and a/b calculation per color channel
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--model", choices=["small", "base_plus", "tiny"], default="small")
    parser.add_argument("--resize-mode", choices=["stretch", "letterbox"], default="letterbox")
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--frames", type=int, default=8, help="Evenly spaced review frames; 0 means all")
    parser.add_argument("--first-frames", type=int, help="Process the first N frames in order")
    parser.add_argument("--frame-name", action="append", help="Process an exact input image name (repeatable)")
    parser.add_argument("--fill-gaps", action=argparse.BooleanOptionalAction, default=True,
                        help="Prompt uncovered regions and label the remainder unknown")
    parser.add_argument("--gap-stability", type=float, default=0.6, help="Stability threshold for gap prompts only")
    parser.add_argument("--coverage-trigger", type=float, default=0.9, help="Prompt gaps only below this initial grid coverage")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--nms-iou", type=float, default=0.85, help="NMS IoU threshold (default: 0.85)")
    parser.add_argument("--guided-filter", action=argparse.BooleanOptionalAction, default=True,
                        help="Refine mask boundaries using RGB-guided filter snapping")
    parser.add_argument("--guided-radius", type=int, default=4, help="Radius for guided filter")
    parser.add_argument("--guided-eps", type=float, default=1e-3, help="Regularization epsilon for guided filter")
    parser.add_argument("--stability-thresh", type=float, default=0.80, help="SAM2 mask stability score threshold (default: 0.80)")
    parser.add_argument("--pred-iou-thresh", type=float, default=0.50, help="Minimum predicted IoU threshold (default: 0.50)")
    parser.add_argument("--min-frame-fraction", type=float, default=0.005,
                        help="Remove masks covering less than this fraction of total frame area (e.g. 0.005 for 0.5%)")
    parser.add_argument("--filter-filaments", action=argparse.BooleanOptionalAction, default=True,
                        help="Remove thin filament/contour-like masks")
    parser.add_argument("--min-thickness", type=float, default=35.0,
                        help="Minimum mean hydraulic thickness (2*area/perimeter) for masks with span >= 150px")
    parser.add_argument("--rel-conf-diff", type=float, default=0.33,
                        help="Relative confidence difference threshold for hierarchical overlap carving (default: 0.33)")
    parser.add_argument("--sor-cleanup", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply Statistical Outlier Removal (connected components) on carved masks")
    parser.add_argument("--save-side-by-side", action=argparse.BooleanOptionalAction, default=True,
                        help="Save side-by-side Raw vs Refined visualization with ID and confidence")
    parser.add_argument("--run-name", type=str)
    args = parser.parse_args()

    images = sorted((args.workspace / "inputs" / "images").glob("*.jpg"))
    if not images:
        parser.error("No input JPG images found")
    if args.frames < 0 or args.grid < 1 or not 0 <= args.gap_stability <= 1 or not 0 < args.coverage_trigger <= 1 or (args.first_frames is not None and args.first_frames < 1):
        parser.error("--frames must be >= 0, --first-frames >= 1, and --grid >= 1")
    if args.frame_name:
        selected = set(args.frame_name)
        images = [path for path in images if path.name in selected]
        if len(images) != len(selected):
            parser.error("One or more --frame-name values were not found")
    elif args.first_frames is not None:
        images = images[:args.first_frames]
    elif args.frames:
        images = [images[i] for i in np.linspace(0, len(images) - 1, min(args.frames, len(images)), dtype=int)]
    run_name = args.run_name or f"{args.model}_grid{args.grid}_{args.resize_mode}_{'eager' if args.no_compile else 'compiled'}{'_complete' if args.fill_gaps else ''}"
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        parser.error("--run-name must be a single directory name")
    output = args.workspace / "sam2_proposals" / run_name
    if output.exists() and any(output.iterdir()):
        parser.error(f"Run directory already contains results: {output}")
    output.mkdir(parents=True, exist_ok=True)

    generator = FastSAM2Generator(args.model, grid=args.grid, stability_thresh=args.stability_thresh,
                                  compile_model=not args.no_compile,
                                  resize_mode=args.resize_mode, nms_iou=args.nms_iou)
    frame_metrics = []
    for path in images:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise OSError(f"Cannot read {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        total_pixels = H * W
        torch.cuda.synchronize()
        start = time.perf_counter()
        if args.fill_gaps:
            masks, metadata, residual, logits = generator.generate_complete(
                rgb, args.gap_stability, args.coverage_trigger, return_logits=True
            )
        else:
            masks, metadata, logits = generator.generate(rgb, return_logits=True)
            residual = ~(masks.any(axis=0) if len(masks) else np.zeros((H, W), dtype=bool))
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start

        # Proposal A: RGB-Guided Boundary Snapping on Logits
        if args.guided_filter and len(masks) > 0 and logits is not None:
            refined_masks = []
            for i in range(len(masks)):
                refined_logit = guided_filter_rgb(rgb, logits[i], radius=args.guided_radius, eps=args.guided_eps)
                ref_mask = (refined_logit > 0).astype(np.uint8)
                refined_masks.append(ref_mask)
            masks = np.stack(refined_masks)

        # 1. Filter by predicted IoU threshold
        if args.pred_iou_thresh > 0 and len(masks) > 0:
            keep_iou = [i for i, m in enumerate(metadata) if m.get("predicted_iou", 0.0) >= args.pred_iou_thresh]
            masks = masks[keep_iou] if keep_iou else np.zeros((0, H, W), dtype=np.uint8)
            metadata = [metadata[i] for i in keep_iou]

        # 2. Outlier & Noise Removal: Filter out masks covering less than min_frame_fraction
        if args.min_frame_fraction > 0 and len(masks) > 0:
            min_pixels = int(args.min_frame_fraction * total_pixels)
            survivor_indices = [
                i for i, m in enumerate(masks)
                if int(m.sum()) >= min_pixels
            ]
            if len(survivor_indices) < len(masks):
                masks = masks[survivor_indices] if survivor_indices else np.zeros((0, H, W), dtype=np.uint8)
                metadata = [metadata[i] for i in survivor_indices]
                for item, m in zip(metadata, masks):
                    item["area_px"] = int(m.sum())

        # 3. Filament & Contour-like Mask Removal
        if args.filter_filaments and len(masks) > 0:
            non_filament_indices = []
            for i, m in enumerate(masks):
                pts = cv2.findNonZero(m.astype(np.uint8))
                if pts is None:
                    continue
                _, _, bw, bh = cv2.boundingRect(pts)
                span = max(bw, bh)
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                peri = sum(cv2.arcLength(c, True) for c in cnts)
                thickness = 2 * m.sum() / max(peri, 1)
                dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
                mean_dist = dist[m > 0].mean() if m.any() else 0
                is_filament = (thickness < args.min_thickness and span >= 150) or (mean_dist < 15.0 and span >= 200)
                if not is_filament:
                    non_filament_indices.append(i)

            if len(non_filament_indices) < len(masks):
                masks = masks[non_filament_indices] if non_filament_indices else np.zeros((0, H, W), dtype=np.uint8)
                metadata = [metadata[i] for i in non_filament_indices]

        # 4. Overlap-only hierarchical subtraction
        items = [{'mask': m.copy(), 'info': info} for m, info in zip(masks, metadata)]
        for i in range(len(items)):
            for j in range(len(items)):
                if i == j: continue
                mi = items[i]['mask']
                mj = items[j]['mask']
                overlap = (mi & mj)
                overlap_area = overlap.sum()
                if overlap_area == 0: continue
                ci = items[i]['info']['predicted_iou']
                cj = items[j]['info']['predicted_iou']
                diff = (max(ci, cj) - min(ci, cj)) / max(min(ci, cj), 1e-6)

                if diff >= args.rel_conf_diff:
                    if ci < cj:
                        mi[mj > 0] = 0
                    else:
                        mj[mi > 0] = 0
                else:
                    area_i = mi.sum()
                    area_j = mj.sum()
                    if area_i > 0 and area_j > 0:
                        if overlap_area / area_i >= 0.80 and area_i < area_j:
                            mi[overlap > 0] = 0
                        elif overlap_area / area_j >= 0.80 and area_j < area_i:
                            mj[overlap > 0] = 0
                        else:
                            if ci >= cj:
                                mj[mi > 0] = 0
                            else:
                                mi[mj > 0] = 0

        # 5. SOR Floater Cleanup
        surviving = []
        min_pixels = int(args.min_frame_fraction * total_pixels) if args.min_frame_fraction > 0 else 0
        for item in items:
            m = item['mask']
            if m.sum() == 0: continue
            if args.sor_cleanup:
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

        masks = np.array([x['mask'] for x in surviving], dtype=np.uint8) if surviving else np.zeros((0, H, W), dtype=np.uint8)
        metadata = [x['info'] for x in surviving]

        residual = ~masks.any(axis=0) if len(masks) else np.ones((H, W), dtype=bool)
        covered = ~residual
        grid_count = sum(item.get("source") == "grid" for item in metadata)
        grid_covered = masks[:grid_count].any(axis=0) if grid_count else np.zeros_like(covered)

        # Standard clean hierarchical overlap resolution:
        # Prioritize grid before gap, and sort by descending predicted IoU
        labels = np.zeros((H, W), dtype=np.int32)
        for index in sorted(range(len(metadata)), key=lambda i: (metadata[i].get("source") != "grid", -metadata[i]["predicted_iou"])):
            labels[(labels == 0) & masks[index].astype(bool)] = index + 1

        if args.fill_gaps:
            _, unknown_regions = cv2.connectedComponents(residual.astype(np.uint8), 8)
            labels[residual] = len(masks) + unknown_regions[residual]
            assert np.all(labels > 0) and np.array_equal(residual, ~masks.any(axis=0))
        np.savez_compressed(output / f"{path.stem}.npz", masks=masks, labels=labels, residual=residual)

        # Generate colors for each mask
        palette = [
            [(37 * i + 93) % 255, (109 * i + 71) % 255, (197 * i + 31) % 255]
            for i in range(len(masks))
        ]

        # Standard overlay
        overlay = bgr.copy()
        for index in range(len(masks)):
            color = np.array(palette[index], dtype=np.uint8)
            visible = labels == index + 1
            overlay[visible] = (0.5 * overlay[visible] + 0.5 * color).astype(np.uint8)
        overlay[~covered] = 0
        cv2.imwrite(str(output / f"{path.stem}_overlay.png"), overlay)

        # Side-by-side view with mask ID and confidence written over each mask
        if args.save_side_by_side:
            side_overlay = bgr.copy()
            for index in range(len(masks)):
                color = np.array(palette[index], dtype=np.uint8)
                visible = labels == index + 1
                side_overlay[visible] = (0.55 * side_overlay[visible] + 0.45 * color).astype(np.uint8)
            side_overlay[~covered] = 0

            # Draw contours and text badge (ID & confidence) over each mask region
            for index, item in enumerate(metadata):
                mask_id = index + 1
                mask_u8 = (labels == mask_id).astype(np.uint8)
                if not mask_u8.any():
                    continue
                color = tuple(int(c) for c in palette[index])
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(side_overlay, contours, -1, color, 2)

                # Compute centroid
                moments = cv2.moments(mask_u8)
                if moments["m00"] > 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                else:
                    yy, xx = np.where(mask_u8 > 0)
                    cx, cy = int(np.median(xx)), int(np.median(yy))

                conf = item.get("predicted_iou", 0.0)
                badge_text = f"#{mask_id} ({conf:.2f})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                thickness = 1
                (tw, th), bl = cv2.getTextSize(badge_text, font, font_scale, thickness)
                pad = 4
                bx1 = max(0, min(W - tw - 2 * pad - 2, cx - tw // 2 - pad))
                by1 = max(0, min(H - th - 2 * pad - 2, cy - th // 2 - pad))
                bx2 = bx1 + tw + 2 * pad
                by2 = by1 + th + 2 * pad

                # Dark background box with color border
                cv2.rectangle(side_overlay, (bx1, by1), (bx2, by2), (15, 15, 15), -1)
                cv2.rectangle(side_overlay, (bx1, by1), (bx2, by2), color, 1)
                cv2.putText(side_overlay, badge_text, (bx1 + pad, by2 - pad - 1),
                            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            side_by_side = np.concatenate((bgr, side_overlay), axis=1)
            cv2.imwrite(str(output / f"{path.stem}_side_by_side.png"), side_by_side)
        metric = {"image": path.name, "seconds": round(seconds, 4), "masks": len(metadata),
                  "grid_masks": grid_count, "gap_masks": len(metadata) - grid_count,
                  "grid_coverage": round(float(grid_covered.mean()), 5),
                  "gap_triggered": bool(args.fill_gaps and grid_covered.mean() < args.coverage_trigger),
                  "sam_coverage": round(float(covered.mean()), 5),
                  "unknown_pixels": int(residual.sum()), "pixels": int(residual.size), "metadata": metadata}
        (output / f"{path.stem}.json").write_text(json.dumps(metric, indent=2) + "\n")
        frame_metrics.append(metric)
        print(f"{path.name}: {seconds:.3f}s, {len(metadata)} masks, {grid_covered.mean():.1%} -> {covered.mean():.1%} SAM coverage, {residual.sum()} unknown px", flush=True)

    startup_frames = {0}
    first_gap_frame_index = None
    if args.fill_gaps:
        first_gap_frame_index = next((i for i, frame in enumerate(frame_metrics) if frame["gap_triggered"]), None)
        if first_gap_frame_index is not None:
            startup_frames.add(first_gap_frame_index)
    steady_frames = [frame for i, frame in enumerate(frame_metrics) if i not in startup_frames]
    summary = {"model": args.model, "grid": args.grid, "compiled": not args.no_compile,
               "resize_mode": args.resize_mode,
               "fill_gaps": args.fill_gaps,
               "gap_stability": args.gap_stability if args.fill_gaps else None,
               "coverage_trigger": args.coverage_trigger if args.fill_gaps else None,
               "first_gap_frame_index": first_gap_frame_index,
               "gpu": torch.cuda.get_device_name(0), "frames": len(images),
               "mean_grid_coverage": round(float(np.mean([m["grid_coverage"] for m in frame_metrics])), 5),
               "mean_sam_coverage": round(float(np.mean([m["sam_coverage"] for m in frame_metrics])), 5),
               "mean_unknown_fraction": round(float(np.mean([m["unknown_pixels"] / m["pixels"] for m in frame_metrics])), 5),
               "fps_excluding_first_frame": round((len(images) - 1) / sum(m["seconds"] for m in frame_metrics[1:]), 3)
               if len(images) > 1 else None,
               "steady_fps_excluding_first_grid_and_gap_calls": round(
                   len(steady_frames) / sum(frame["seconds"] for frame in steady_frames), 3
               ) if steady_frames else None,
               "frame_metrics": frame_metrics}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
