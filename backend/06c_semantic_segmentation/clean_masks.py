#!/usr/bin/env python3
"""Remove implausible SAM proposals and render uncovered pixels black."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


DEFAULT_WORKSPACE = Path(__file__).resolve().parents[1] / "current_scene" / "06c_semantic_segmentation"


def shape_reasons(mask: np.ndarray) -> tuple[list[str], dict]:
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None:
        return ["empty"], {}
    x, y, width, height = cv2.boundingRect(points)
    roi = mask[y:y + height, x:x + width].astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    main = 1 + int(np.argmax(areas))
    main_area = int(stats[main, cv2.CC_STAT_AREA])
    total_area = int(areas.sum())
    contours, _ = cv2.findContours((components == main).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    thickness = 2 * main_area / max(perimeter, 1)
    solidity = min(1.0, main_area / max(hull_area, 1))
    fill = main_area / (width * height)
    span = max(width, height)
    fractions = areas / total_area
    reasons = []
    if (fractions >= 0.03).sum() >= 3 and fractions.max() < 0.8:
        reasons.append("scattered")
    if thickness < 25 and span >= 120:
        reasons.append("thin")
    if solidity < 0.55 and fill < 0.25 and thickness < 60 and span >= 120:
        reasons.append("branched")
    return reasons, {"components": count - 1, "largest_fraction": round(float(fractions.max()), 3),
                     "thickness_px": round(float(thickness), 1), "solidity": round(float(solidity), 3),
                     "box_fill": round(float(fill), 3), "span_px": span}


def clean_frame(masks: np.ndarray, metadata: list[dict]) -> tuple[np.ndarray, list[dict], list[dict]]:
    kept, kept_meta, removed = [], [], []
    for index, (mask, item) in enumerate(zip(masks, metadata)):
        reasons, metrics = shape_reasons(mask)
        if reasons:
            removed.append({"original_mask_id": index + 1, "reasons": reasons, **metrics})
        else:
            kept.append(mask)
            kept_meta.append({**item, "original_mask_id": index + 1})
    cleaned = np.stack(kept) if kept else np.zeros((0, *masks.shape[1:]), dtype=np.uint8)
    return cleaned, kept_meta, removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--source", default="first50_grid10_gap060")
    parser.add_argument("--run-name", default="first50_grid10_gap060_clean")
    args = parser.parse_args()
    for value in (args.source, args.run_name):
        if Path(value).name != value or value in {".", ".."}:
            parser.error("--source and --run-name must be directory names")
    root = args.workspace / "sam2_proposals"
    source, output = root / args.source, root / args.run_name
    if source.resolve() == output.resolve():
        parser.error("Source and output must differ")
    if output.exists() and any(output.iterdir()):
        parser.error(f"Run directory already contains results: {output}")
    source_summary = json.loads((source / "summary.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    frame_metrics = []
    for record in source_summary["frame_metrics"]:
        start = time.perf_counter()
        stem = Path(record["image"]).stem
        with np.load(source / f"{stem}.npz") as archive:
            masks = archive["masks"]
        if len(masks) != len(record["metadata"]):
            raise ValueError(f"Mask count differs from metadata in {stem}")
        cleaned, metadata, removed = clean_frame(masks, record["metadata"])
        residual = ~(cleaned.any(axis=0) if len(cleaned) else np.zeros(masks.shape[1:], dtype=bool))
        labels = np.zeros(residual.shape, dtype=np.int32)
        for index in sorted(range(len(metadata)), key=lambda i: (metadata[i].get("source") != "grid", -metadata[i]["predicted_iou"])):
            labels[(labels == 0) & cleaned[index].astype(bool)] = index + 1
        _, regions = cv2.connectedComponents(residual.astype(np.uint8), connectivity=8)
        labels[residual] = len(cleaned) + regions[residual]
        assert np.all(labels > 0) and np.array_equal(residual, ~cleaned.any(axis=0))
        np.savez_compressed(output / f"{stem}.npz", masks=cleaned, labels=labels, residual=residual)

        image = cv2.imread(str(args.workspace / "inputs" / "images" / record["image"]))
        if image is None:
            raise OSError(f"Cannot read image {record['image']}")
        overlay = image.copy()
        for index, item in enumerate(metadata):
            visible = labels == index + 1
            original = item["original_mask_id"] - 1
            color = np.array([(37 * original + 93) % 255, (109 * original + 71) % 255, (197 * original + 31) % 255])
            overlay[visible] = (0.5 * overlay[visible] + 0.5 * color).astype(np.uint8)
        overlay[residual] = 0
        cv2.imwrite(str(output / f"{stem}_overlay.png"), overlay)
        metric = {"image": record["image"], "input_masks": len(masks), "kept_masks": len(cleaned),
                  "removed_masks": removed, "coverage_before": record["sam_coverage"],
                  "coverage_after": round(1 - float(residual.mean()), 5),
                  "unknown_pixels": int(residual.sum()), "pixels": int(residual.size),
                  "cleanup_seconds": round(time.perf_counter() - start, 3), "metadata": metadata}
        (output / f"{stem}.json").write_text(json.dumps(metric, indent=2) + "\n")
        frame_metrics.append(metric)
        print(f"{record['image']}: removed {len(removed)}/{len(masks)}, coverage {metric['coverage_after']:.1%}", flush=True)

    summary = {"source": args.source, "frames": len(frame_metrics),
               "removed_masks": sum(len(item["removed_masks"]) for item in frame_metrics),
               "mean_coverage_before": source_summary["mean_sam_coverage"],
               "mean_coverage_after": round(float(np.mean([item["coverage_after"] for item in frame_metrics])), 5),
               "source_inference_fps": source_summary["fps_excluding_first_frame"],
               "source_steady_inference_fps": source_summary.get("steady_fps_excluding_first_grid_and_gap_calls"),
               "mean_cleanup_seconds": round(float(np.mean([item["cleanup_seconds"] for item in frame_metrics])), 3),
               "frame_metrics": frame_metrics}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
