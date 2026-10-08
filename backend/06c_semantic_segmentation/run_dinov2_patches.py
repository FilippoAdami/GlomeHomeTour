#!/usr/bin/env python3
"""Extract DINOv2 ViT-S/14 patches and sweep spatial grouping thresholds."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel


WORKSPACE = Path(__file__).resolve().parents[1] / "current_scene" / "06c_semantic_segmentation"
MODEL = "facebook/dinov2-small"  # Official conversion of dinov2_vits14, without registers.
PATCH = 14
MAX_SIDE = 896


def group_neighbors(features: np.ndarray, threshold: float) -> np.ndarray:
    """Connected components of four-neighbor edges whose cosine is at least threshold."""
    height, width, _ = features.shape
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    unit = features / np.maximum(norms, 1e-12)
    horizontal = np.sum(unit[:, :-1] * unit[:, 1:], axis=-1) >= threshold
    vertical = np.sum(unit[:-1] * unit[1:], axis=-1) >= threshold
    parent = np.arange(height * width)

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for row in range(height):
        for col in range(width):
            index = row * width + col
            if col and horizontal[row, col - 1]:
                parent[root(index)] = root(index - 1)
            if row and vertical[row - 1, col]:
                parent[root(index)] = root(index - width)

    _, inverse = np.unique([root(index) for index in range(height * width)], return_inverse=True)
    return inverse.reshape(height, width).astype(np.uint16)


def adjacent_similarities(features: np.ndarray) -> np.ndarray:
    unit = features / np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-12)
    return np.concatenate((np.sum(unit[:, :-1] * unit[:, 1:], axis=-1).ravel(),
                           np.sum(unit[:-1] * unit[1:], axis=-1).ravel()))


def grow_cluster_means(features: np.ndarray, threshold: float) -> np.ndarray:
    """Grow each region through four-neighbor candidates compared to its current mean."""
    height, width, channels = features.shape
    flat = features.reshape(-1, channels).astype(np.float32)
    unit = flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)
    labels = np.full(len(flat), -1, dtype=np.int32)

    def neighbors(index: int):
        row, col = divmod(index, width)
        if row:
            yield index - width
        if col:
            yield index - 1
        if col + 1 < width:
            yield index + 1
        if row + 1 < height:
            yield index + width

    region = 0
    for seed in range(len(flat)):
        if labels[seed] != -1:
            continue
        labels[seed] = region
        total = flat[seed].astype(np.float64)
        frontier = {index for index in neighbors(seed) if labels[index] == -1}
        while frontier:
            candidates = np.fromiter(sorted(frontier), dtype=np.int32)
            scores = unit[candidates] @ (total / max(np.linalg.norm(total), 1e-12))
            best = int(np.argmax(scores))
            if scores[best] < threshold:
                break
            index = int(candidates[best])
            labels[index] = region
            total += flat[index]  # Sum / count is the mean; count cancels in cosine.
            frontier.remove(index)
            frontier.update(other for other in neighbors(index) if labels[other] == -1)
        region += 1
    return labels.reshape(height, width).astype(np.uint16)


def grow_and_merge_grid_seeds(
    features: np.ndarray,
    threshold: float,
    merge_criterion: str = "mean",
    n_rows: int = 8,
    n_cols: int = 5,
) -> np.ndarray:
    """Initialize seeds at square grid centers, grow regions greedily, then iteratively merge touching regions."""
    height, width, channels = features.shape
    unit = features / np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-12)

    # 1. 40 seeds at centers of 8x5 grid (square cells on 16:9 portrait)
    row_centers = [int((i + 0.5) * height / n_rows) for i in range(n_rows)]
    col_centers = [int((j + 0.5) * width / n_cols) for j in range(n_cols)]

    labels = np.full((height, width), -1, dtype=np.int32)
    region_totals = {}
    import heapq
    pq = []

    region_id = 0
    for r in row_centers:
        for c in col_centers:
            labels[r, c] = region_id
            region_totals[region_id] = features[r, c].astype(np.float64)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    sim = float(np.dot(unit[nr, nc], unit[r, c]))
                    heapq.heappush(pq, (-sim, nr, nc, region_id))
            region_id += 1

    # 2. Grow regions outward to cover the full image
    while pq:
        neg_sim, r, c, reg = heapq.heappop(pq)
        if labels[r, c] != -1:
            continue
        labels[r, c] = reg
        region_totals[reg] += features[r, c]

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and labels[nr, nc] == -1:
                mean_vec = region_totals[reg] / max(np.linalg.norm(region_totals[reg]), 1e-12)
                sim = float(np.dot(unit[nr, nc], mean_vec))
                heapq.heappush(pq, (-sim, nr, nc, reg))

    # 3. Iterative region merging of touching regions
    while True:
        # Find touching boundaries
        h_left, h_right = labels[:, :-1], labels[:, 1:]
        h_mask = h_left != h_right
        h_u = np.minimum(h_left[h_mask], h_right[h_mask])
        h_v = np.maximum(h_left[h_mask], h_right[h_mask])

        v_top, v_bot = labels[:-1, :], labels[1:, :]
        v_mask = v_top != v_bot
        v_u = np.minimum(v_top[v_mask], v_bot[v_mask])
        v_v = np.maximum(v_top[v_mask], v_bot[v_mask])

        all_u = np.concatenate([h_u, v_u])
        all_v = np.concatenate([h_v, v_v])
        if len(all_u) == 0:
            break

        best_pair = None
        best_sim = -1.0

        if merge_criterion == "mean":
            u_regions = np.unique(labels)
            means = {}
            for reg in u_regions:
                tot = np.sum(features[labels == reg], axis=0)
                means[reg] = tot / max(np.linalg.norm(tot), 1e-12)

            # Unique touching pairs
            pairs = np.unique(np.stack([all_u, all_v], axis=1), axis=0)
            for u, v in pairs:
                sim = float(np.dot(means[u], means[v]))
                if sim > best_sim:
                    best_sim = sim
                    best_pair = (u, v)

        elif merge_criterion == "boundary":
            h_sims = np.sum(unit[:, :-1][h_mask] * unit[:, 1:][h_mask], axis=-1)
            v_sims = np.sum(unit[:-1, :][v_mask] * unit[1:, :][v_mask], axis=-1)
            all_sims = np.concatenate([h_sims, v_sims])

            pairs_dict = {}
            for u, v, s in zip(all_u, all_v, all_sims):
                key = (u, v)
                if key not in pairs_dict or s > pairs_dict[key]:
                    pairs_dict[key] = s

            for (u, v), s in pairs_dict.items():
                if s > best_sim:
                    best_sim = s
                    best_pair = (u, v)

        if best_pair is None or best_sim < threshold:
            break

        u, v = best_pair
        labels[labels == v] = u

    _, compact = np.unique(labels, return_inverse=True)
    return compact.reshape(height, width).astype(np.uint16)


def merge_grid_rounds(features: np.ndarray, threshold: float) -> np.ndarray:
    """Merge mutually preferred adjacent regions once per round using frozen means."""
    height, width, channels = features.shape
    labels = np.arange(height * width, dtype=np.int32).reshape(height, width)
    flat = features.reshape(-1, channels)
    while True:
        edges = np.concatenate((np.stack((labels[:, :-1].ravel(), labels[:, 1:].ravel()), axis=1),
                                np.stack((labels[:-1].ravel(), labels[1:].ravel()), axis=1)))
        edges = edges[edges[:, 0] != edges[:, 1]]
        if not len(edges):
            break
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        totals = np.zeros((int(labels.max()) + 1, channels), dtype=np.float64)
        np.add.at(totals, labels.ravel(), flat)
        unit = totals / np.maximum(np.linalg.norm(totals, axis=1, keepdims=True), 1e-12)
        scores = np.sum(unit[edges[:, 0]] * unit[edges[:, 1]], axis=1)
        best = np.full(len(totals), -1, dtype=np.int32)
        for index in np.argsort(-scores, kind="stable"):
            if scores[index] < threshold:
                break
            left, right = edges[index]
            if best[left] == -1:
                best[left] = right
            if best[right] == -1:
                best[right] = left
        pairs = edges[(best[edges[:, 0]] == edges[:, 1]) & (best[edges[:, 1]] == edges[:, 0])]
        if not len(pairs):
            break
        mapping = np.arange(len(totals), dtype=np.int32)
        mapping[pairs[:, 1]] = pairs[:, 0]
        _, compact = np.unique(mapping[labels], return_inverse=True)
        labels = compact.reshape(height, width).astype(np.int32)
    return labels.astype(np.uint16)


def merge_small_regions(labels: np.ndarray, features: np.ndarray, min_patches: int) -> np.ndarray:
    """Absorb small regions into the closest (touching) region with the best mean cosine."""
    result = labels.copy()
    counts = np.bincount(result.ravel(), minlength=int(result.max()) + 1)
    totals = np.zeros((len(counts), features.shape[-1]), dtype=np.float64)
    np.add.at(totals, result.ravel(), features.reshape(-1, features.shape[-1]))
    while True:
        small = np.flatnonzero((counts > 0) & (counts < min_patches))
        if not len(small):
            break
        source = int(min(small, key=lambda region: (counts[region], region)))
        mask = result == source
        touching = np.unique(np.concatenate((result[:-1][mask[1:]], result[1:][mask[:-1]],
                                             result[:, :-1][mask[:, 1:]], result[:, 1:][mask[:, :-1]])))
        touching = touching[touching != source]
        if not len(touching):  # The entire image is one region.
            break
        source_unit = totals[source] / max(np.linalg.norm(totals[source]), 1e-12)
        target_unit = totals[touching] / np.maximum(np.linalg.norm(totals[touching], axis=1, keepdims=True), 1e-12)
        target = int(touching[np.argmax(target_unit @ source_unit)])
        result[mask] = target
        counts[target] += counts[source]
        counts[source] = 0
        totals[target] += totals[source]
        totals[source] = 0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--first-frames", type=int, default=10)
    parser.add_argument("--run-name", default="first10_vits14_neighbors")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        help="Cosine thresholds; default: 10th, 30th, 50th, 70th, 90th neighbor percentiles")
    parser.add_argument("--grouping", choices=["neighbors", "cluster-mean", "grid-rounds", "grid-seeds-mean", "grid-seeds-boundary"], default="neighbors")
    parser.add_argument("--min-region-patches", type=int, default=0,
                        help="Merge regions smaller than this many patches into a touching region; 0 disables cleanup")
    parser.add_argument("--pca-dims", type=int, default=0,
                        help="Fit one PCA across selected frames and group on this many components; 0 disables PCA")
    parser.add_argument("--pca-from", type=Path, help="Reuse a saved pca.npz projection instead of refitting")
    parser.add_argument("--embeddings-from", type=Path, help="Reuse saved patch embeddings for a grouping-only ablation")
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
        parser.error("Use a single-directory --run-name")
    if args.thresholds and any(not -1 <= value <= 1 for value in args.thresholds):
        parser.error("Cosine thresholds must be between -1 and 1")
    if args.min_region_patches < 0 or args.pca_dims < 0:
        parser.error("--min-region-patches and --pca-dims must be nonnegative")
    if args.pca_from and (not args.pca_from.is_file() or not args.pca_dims):
        parser.error("--pca-from requires an existing pca.npz and --pca-dims")
    images = sorted((args.workspace / "inputs" / "images").glob("*.jpg"))
    if args.first_frames > 0:
        images = images[:args.first_frames]
    if not images:
        parser.error("No input JPG images found")
    output = args.workspace / "dinov2_patches" / args.run_name
    if output.exists() and any(output.iterdir()):
        parser.error(f"Run directory already contains results: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = "cpu" if args.embeddings_from else ("cuda" if torch.cuda.is_available() else "cpu")
    if args.embeddings_from:
        if not args.embeddings_from.is_dir():
            parser.error(f"Embedding directory does not exist: {args.embeddings_from}")
    else:
        model = AutoModel.from_pretrained(MODEL).to(device).eval()
        mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    similarities = []
    frames = []
    frame_grids = {}
    start = time.perf_counter()
    for path in images:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
            if args.embeddings_from:
                source = args.embeddings_from / f"{path.stem}_embeddings.npy"
                if not source.is_file():
                    parser.error(f"Missing embeddings: {source}")
                features = np.load(source).astype(np.float32)
                grid = features.shape[:2]
            else:
                width, height = image.size
                scale = MAX_SIDE / max(width, height)
                grid = (max(1, round(height * scale / PATCH)), max(1, round(width * scale / PATCH)))
                resized = image.convert("RGB").resize((grid[1] * PATCH, grid[0] * PATCH), Image.Resampling.BICUBIC)
        if not args.embeddings_from:
            pixels = torch.from_numpy(np.array(resized)).permute(2, 0, 1).to(device=device, dtype=torch.float32)[None] / 255
            with torch.inference_mode():
                tokens = model(pixel_values=(pixels - mean) / std).last_hidden_state[:, 1:, :]
            features = tokens.reshape(*grid, -1).float().cpu().numpy()
            np.save(output / f"{path.stem}_embeddings.npy", features.astype(np.float16))
            if args.pca_from:
                features = features.astype(np.float16).astype(np.float32)  # Match the cached inputs used to fit PCA.
        frames.append((path, rgb, features))
        frame_grids[path.name] = list(grid)
        print(f"{'Loaded' if args.embeddings_from else 'Extracted'} {path.name}: {features.shape}", flush=True)
    extraction_seconds = time.perf_counter() - start
    pca_explained_variance = None
    if args.pca_dims:
        if args.pca_from:
            with np.load(args.pca_from) as saved:
                pca_mean = saved["mean"]
                components = saved["components"]
                pca_explained_variance = round(float(saved["explained_variance_ratio"].sum()), 6)
            if components.shape != (args.pca_dims, frames[0][2].shape[-1]):
                parser.error(f"PCA components have shape {components.shape}, expected {(args.pca_dims, frames[0][2].shape[-1])}")
            shutil.copyfile(args.pca_from, output / "pca.npz")
            projected_frames = [(features.reshape(-1, features.shape[-1]) - pca_mean) @ components.T
                                for _, _, features in frames]
        else:
            from sklearn.decomposition import PCA

            raw_features = np.concatenate([features.reshape(-1, features.shape[-1]) for _, _, features in frames])
            if args.pca_dims > min(raw_features.shape):
                parser.error(f"--pca-dims must be <= {min(raw_features.shape)}")
            pca = PCA(n_components=args.pca_dims, svd_solver="randomized", random_state=42)
            pca.fit(raw_features)
            projected = pca.transform(raw_features).astype(np.float32)
            pca_explained_variance = round(float(pca.explained_variance_ratio_.sum()), 6)
            np.savez_compressed(output / "pca.npz", mean=pca.mean_, components=pca.components_,
                                explained_variance_ratio=pca.explained_variance_ratio_)
            projected_frames = []
            offset = 0
            for _, _, features in frames:
                count = features.shape[0] * features.shape[1]
                projected_frames.append(projected[offset:offset + count])
                offset += count
        transformed_frames = []
        for (path, rgb, features), projected in zip(frames, projected_frames):
            transformed = projected.reshape(*features.shape[:2], args.pca_dims).astype(np.float32)
            np.save(output / f"{path.stem}_pca{args.pca_dims}_embeddings.npy", transformed.astype(np.float16))
            transformed_frames.append((path, rgb, transformed))
        frames = transformed_frames
    similarities = [adjacent_similarities(features) for _, _, features in frames]
    all_similarities = np.concatenate(similarities)
    thresholds = args.thresholds or np.quantile(all_similarities, [0.1, 0.3, 0.5, 0.7, 0.9]).tolist()
    summary = {"model": MODEL, "model_variant": "dinov2_vits14", "patch_grids": frame_grids,
               "frames": [p.name for p, _, _ in frames],
               "device": device, "feature_loading_seconds" if args.embeddings_from else "extraction_seconds": round(extraction_seconds, 3),
               "embedding_source": str(args.embeddings_from.resolve()) if args.embeddings_from else None,
               "grouping": args.grouping, "min_region_patches": args.min_region_patches,
               "neighbor_cosine_quantiles": {str(q): round(float(v), 6) for q, v in zip(
                   [10, 30, 50, 70, 90], np.quantile(all_similarities, [0.1, 0.3, 0.5, 0.7, 0.9]))},
               "thresholds": [], "pca": bool(args.pca_dims), "pca_dims": args.pca_dims,
               "pca_explained_variance": pca_explained_variance,
               "pca_source": str(args.pca_from.resolve()) if args.pca_from else None}
    overall_start = start
    for threshold in thresholds:
        t0 = time.perf_counter()
        threshold = float(threshold)
        folder = output / f"threshold_{threshold:.6f}"
        folder.mkdir()
        counts = []
        merged_counts = []
        for path, rgb, features in frames:
            grouping = {
                "neighbors": group_neighbors,
                "cluster-mean": grow_cluster_means,
                "grid-rounds": merge_grid_rounds,
                "grid-seeds-mean": lambda feat, thresh: grow_and_merge_grid_seeds(feat, thresh, merge_criterion="mean"),
                "grid-seeds-boundary": lambda feat, thresh: grow_and_merge_grid_seeds(feat, thresh, merge_criterion="boundary"),
            }[args.grouping]
            labels = grouping(features, threshold)
            initial_count = len(np.unique(labels))
            if args.min_region_patches:
                labels = merge_small_regions(labels, features, args.min_region_patches)
            count = len(np.unique(labels))
            counts.append(count)
            merged_counts.append(initial_count - count)
            np.save(folder / f"{path.stem}_labels.npy", labels)
            rng = np.random.default_rng(42)
            colors = rng.integers(48, 256, size=(int(labels.max()) + 1, 3), dtype=np.uint8)
            colored = colors[labels]
            colored = cv2.resize(colored, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            pair = np.concatenate((rgb, colored), axis=1)
            cv2.imwrite(str(folder / f"{path.stem}_rgb_patches.jpg"), cv2.cvtColor(pair, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        threshold_seconds = time.perf_counter() - t0
        threshold_fps = round(len(frames) / max(threshold_seconds, 1e-6), 2)
        summary["thresholds"].append({"cosine": round(threshold, 6), "directory": folder.name,
                                      "components_per_frame": counts, "mean_components": round(float(np.mean(counts)), 2),
                                      "merged_regions_per_frame": merged_counts,
                                      "grouping_seconds": round(threshold_seconds, 3),
                                      "grouping_fps": threshold_fps})
        print(f"{folder.name}: components {counts[:10]}... ({len(frames)} frames in {threshold_seconds:.2f}s, {threshold_fps} FPS)", flush=True)
    total_seconds = time.perf_counter() - overall_start
    total_fps = round((len(frames) * len(thresholds)) / max(total_seconds, 1e-6), 2) if thresholds else 0.0
    summary["total_seconds"] = round(total_seconds, 3)
    summary["overall_fps"] = total_fps
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Total time: {total_seconds:.2f}s | Overall FPS: {total_fps}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
