#!/usr/bin/env python3
"""Rank indoor object categories for extracted mesh and Gaussian Splat frames."""

from __future__ import annotations

import csv
import argparse
import json
import logging
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from step_semantic_similarity import (
    DEFAULT_MODEL_NAME,
    compute_class_embeddings,
    compute_image_embeddings,
    load_embedding_model,
)

BACKEND_DIR = Path(__file__).resolve().parents[1]
SEGMENTATION_DIR = BACKEND_DIR / "current_scene" / "scan_0" / "segmentation"
FRAME_DIR = SEGMENTATION_DIR / "non_structural_category_frames"
OUTPUT_DIR = SEGMENTATION_DIR / "non_structural_similarity"
INPUT_DIRS = {
    "mesh": FRAME_DIR / "mesh",
    "gs": FRAME_DIR / "gs",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TOP_K = 10

CATEGORY_GROUPS = [
    ("door", "door-frame", "doorway"),
    ("window", "curtain"),
    ("stairs", "staircase", "steps", "stairway"),
    ("ramp",),
    ("elevator", "lift"),
    ("column", "pillar"),
    ("radiator", "convector", "heater", "fireplace", "chimney"),
    ("gate",),
    ("fountain",),
    ("gazebo",),
    ("table", "desk", "dining table", "coffee table", "worktop"),
    ("cabinet", "closet", "wardrobe", "nightstand", "cupboard", "drawer"),
    ("chair", "office chair", "bench"),
    ("rug", "carpet", "mat"),
    ("bookshelf", "shelf", "ledge", "rack"),
    ("sofa", "couch", "armchair"),
    ("television", "monitor"),
    ("bed", "crib"),
    ("cooktop",),
    ("refrigerator", "fridge"),
    ("oven",),
    ("dishwasher",),
    ("sink",),
    ("wc", "toilette"),
    ("bidet",),
    ("shower", "jacuzzi", "tub"),
    ("washing machine", "dryer"),
    ("car", "motorbike"),
]
CATEGORIES = [words[0] for words in CATEGORY_GROUPS]
CATEGORY_WORDS = [word for group in CATEGORY_GROUPS for word in group]
WORD_CATEGORY_IDS = [category_id for category_id, group in enumerate(CATEGORY_GROUPS, 1) for _ in group]
ASSOCIATION_EXCLUDED_CLASS_IDS: set[int] = set()
ASSOCIATION_MIN_MARGIN_CLASSES = 5
ASSOCIATION_MARGIN = 0.02
ASSOCIATION_MIN_SPECIFICITY = 0.025


def specificity_matrices(raw_matrices: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Subtract each class's mean cosine across the available render sets."""
    baseline = np.concatenate(list(raw_matrices.values()), axis=1).mean(axis=1, keepdims=True)
    return {name: matrix - baseline for name, matrix in raw_matrices.items()}, baseline


def aggregate_valid_class_associations(
    image_paths: dict[str, list[Path]],
    enhanced_matrices: dict[str, np.ndarray],
) -> list[tuple[int, int]]:
    """Return valid (source image class ID, best text class ID) associations."""
    text_class_ids = [
        class_id for class_id in range(1, len(CATEGORIES) + 1)
        if class_id not in ASSOCIATION_EXCLUDED_CLASS_IDS
    ]
    text_indices = np.asarray([class_id - 1 for class_id in text_class_ids])
    scores_by_image_class: dict[int, list[np.ndarray]] = {}

    for render_set, paths in image_paths.items():
        scores = enhanced_matrices[render_set][text_indices]
        for image_idx, path in enumerate(paths):
            match = re.fullmatch(r"class_(\d+)_inst_\d+", path.stem)
            if match is None:
                raise ValueError(f"Unexpected non-structural frame name: {path.name}")
            image_scores = scores[:, image_idx]
            if np.count_nonzero(image_scores >= image_scores.max() - ASSOCIATION_MARGIN) >= ASSOCIATION_MIN_MARGIN_CLASSES:
                continue
            scores_by_image_class.setdefault(int(match.group(1)), []).append(image_scores)

    associations = []
    for image_class_id, image_scores in sorted(scores_by_image_class.items()):
        mean_scores = np.mean(image_scores, axis=0)
        best_idx = int(np.argmax(mean_scores))
        if mean_scores[best_idx] > ASSOCIATION_MIN_SPECIFICITY:
            associations.append((image_class_id, text_class_ids[best_idx]))
    return associations


def top_rows(
    class_names: list[str],
    class_words: list[tuple[str, ...]],
    image_names: list[str],
    raw: np.ndarray,
    enhanced: np.ndarray,
    top_k: int = TOP_K,
) -> list[dict[str, object]]:
    rows = []
    for image_idx, image_name in enumerate(image_names):
        selected = np.argsort(-enhanced[:, image_idx], kind="stable")[:top_k]
        for rank, class_idx in enumerate(selected, 1):
            rows.append({
                "image": image_name,
                "rank": rank,
                "class_id": int(class_idx + 1),
                "category": class_names[class_idx],
                "words": " | ".join(class_words[class_idx]),
                "enhanced_similarity": float(enhanced[class_idx, image_idx]),
                "average_cosine_similarity": float(raw[class_idx, image_idx]),
            })
    return rows


def _fingerprint(paths: list[Path]) -> str:
    return json.dumps(
        [(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in paths],
        separators=(",", ":"),
    )


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _load_word_embeddings(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as cache:
        required = {"model_name", "words", "word_embeddings"}
        if not required.issubset(cache.files):
            return None
        cached_words = cache["words"].astype(str).tolist()
        cached_embeddings = cache["word_embeddings"]
        if cache["model_name"].item() != DEFAULT_MODEL_NAME or cached_embeddings.shape[0] != len(cached_words):
            return None
        by_word = dict(zip(cached_words, cached_embeddings))
        if any(word not in by_word for word in CATEGORY_WORDS):
            return None
        return np.stack([by_word[word] for word in CATEGORY_WORDS])


def _load_image_embeddings(path: Path, images: list[Path]) -> tuple[np.ndarray, list[str]] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as cache:
        if (
            cache["model_name"].item() != DEFAULT_MODEL_NAME
            or cache["fingerprint"].item() != _fingerprint(images)
        ):
            return None
        paths = cache["paths"].tolist()
        return cache["embeddings"], paths


def _save_panel(
    image_path: Path,
    output_path: Path,
    categories: list[str],
    enhanced_scores: np.ndarray,
    cosine_scores: np.ndarray,
    color_limit: float,
) -> None:
    order = np.argsort(-enhanced_scores, kind="stable")[:TOP_K]
    fig, (image_ax, matrix_ax) = plt.subplots(
        1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1.25, 1]}, layout="constrained"
    )
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image_ax.imshow(image)
    image_ax.set_title(image_path.name)
    image_ax.axis("off")

    values = enhanced_scores[order, None]
    matrix_ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-color_limit, vmax=color_limit)
    matrix_ax.set_yticks(range(len(order)), [categories[idx] for idx in order])
    matrix_ax.set_xticks([0], ["specificity"])
    matrix_ax.set_title("Top 10 class matches")
    matrix_ax.set_xticks(np.arange(-0.5, 1, 1), minor=True)
    matrix_ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    matrix_ax.grid(which="minor", color="white", linewidth=2)
    matrix_ax.tick_params(which="minor", bottom=False, left=False)
    for row, class_idx in enumerate(order):
        matrix_ax.text(
            0, row,
            f"{enhanced_scores[class_idx]:+.3f}\navg cosine {cosine_scores[class_idx]:.3f}",
            ha="center", va="center", fontsize=8,
            color="white" if abs(enhanced_scores[class_idx]) > color_limit * 0.55 else "black",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> list[tuple[int, int]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-frames-dir", type=Path, default=INPUT_DIRS["mesh"])
    parser.add_argument("--gs-frames-dir", type=Path, default=INPUT_DIRS["gs"])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    input_dirs = {"mesh": args.mesh_frames_dir, "gs": args.gs_frames_dir}
    image_paths = {
        name: sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        for name, directory in input_dirs.items() if directory.is_dir()
    }
    image_paths = {name: paths for name, paths in image_paths.items() if paths}
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    class_words = [tuple(group) for group in CATEGORY_GROUPS]
    (output_dir / "categories.json").write_text(json.dumps([
        {"id": index, "name": name, "words": list(words)}
        for index, (name, words) in enumerate(zip(CATEGORIES, CATEGORY_GROUPS), 1)
    ], indent=2) + "\n")
    associations_path = output_dir / "valid_class_associations.json"
    if not any(image_paths.values()):
        (output_dir / "specificity_baseline.json").write_text(json.dumps({
            "definition": "class-average cosine mean across available input folders",
            "mean_average_cosine_by_class": {},
            "num_images": 0,
            "top_k": TOP_K,
        }, indent=2) + "\n")
        associations_path.write_text(json.dumps({
            "tuple_fields": ["image_class_id", "text_class_id"],
            "excluded_text_class_ids": sorted(ASSOCIATION_EXCLUDED_CLASS_IDS),
            "ambiguity_filter": f"discard images with {ASSOCIATION_MIN_MARGIN_CLASSES} or more text classes within {ASSOCIATION_MARGIN:.2f} of the top score",
            "minimum_best_mean_specificity": ASSOCIATION_MIN_SPECIFICITY,
            "associations": [],
        }, indent=2) + "\n")
        logging.info("No non-structural instance frames to classify; saved empty associations.")
        return []

    class_cache = output_dir / "category_embeddings.npz"
    word_embeddings = _load_word_embeddings(class_cache)
    image_data = {
        name: _load_image_embeddings(output_dir / name / "image_embeddings.npz", paths)
        for name, paths in image_paths.items()
    }

    model = None
    if word_embeddings is None or any(data is None for data in image_data.values()):
        model = load_embedding_model()
    if word_embeddings is None:
        embedded, _ = compute_class_embeddings(CATEGORY_WORDS, model, use_templates=False)
        word_embeddings = np.stack([embedding for _, embedding in embedded])
        assert word_embeddings.shape[0] == len(CATEGORY_WORDS)
        assert np.allclose(np.linalg.norm(word_embeddings, axis=1), 1.0, atol=1e-4)
    _save_npz(
        class_cache,
        model_name=np.asarray(DEFAULT_MODEL_NAME),
        classes=np.asarray(CATEGORIES),
        words=np.asarray(CATEGORY_WORDS),
        word_category_ids=np.asarray(WORD_CATEGORY_IDS, dtype=np.int16),
        word_embeddings=word_embeddings,
    )
    for name, paths in image_paths.items():
        if image_data[name] is None:
            assert model is not None
            embeddings, paths_for_cache, _ = compute_image_embeddings(paths, model, batch_size=8)
            image_data[name] = embeddings, paths_for_cache
            _save_npz(
                output_dir / name / "image_embeddings.npz",
                model_name=np.asarray(DEFAULT_MODEL_NAME),
                fingerprint=np.asarray(_fingerprint(paths)),
                paths=np.asarray(paths_for_cache),
                embeddings=embeddings,
            )

    word_similarities = {name: word_embeddings @ image_data[name][0].T for name in input_dirs if name in image_paths}
    offsets = np.cumsum([0, *(len(group) for group in CATEGORY_GROUPS)])
    raw = {
        name: np.stack([
            matrix[offsets[idx]:offsets[idx + 1]].mean(axis=0)
            for idx in range(len(CATEGORY_GROUPS))
        ])
        for name, matrix in word_similarities.items()
    }
    assert all(matrix.shape[0] == len(CATEGORIES) for matrix in raw.values())
    enhanced, baseline = specificity_matrices(raw)
    valid_associations = aggregate_valid_class_associations(image_paths, enhanced)
    color_limit = max(
        float(np.max(np.abs(np.sort(matrix, axis=0)[-TOP_K:])))
        for matrix in enhanced.values()
    ) or 1e-12

    for name, paths in image_paths.items():
        image_names = [path.name for path in paths]
        out_dir = output_dir / name
        rows = top_rows(
            CATEGORIES, class_words, image_names, raw[name], enhanced[name]
        )
        with (out_dir / "top10_similarity.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        (out_dir / "summary.json").write_text(json.dumps({
            "model": DEFAULT_MODEL_NAME,
            "class_similarity": "arithmetic mean of cosine similarities to each word in the class",
            "enhancement": "class-average cosine minus the class mean across available image sets",
            "num_images": len(paths),
            "num_categories": len(CATEGORIES),
            "categories": [
                {"id": index, "name": name, "words": list(words)}
                for index, (name, words) in enumerate(zip(CATEGORIES, CATEGORY_GROUPS), 1)
            ],
            "top_classes_per_image": {
                image: [row for row in rows if row["image"] == image]
                for image in image_names
            },
        }, indent=2) + "\n")
        for idx, path in enumerate(paths):
            _save_panel(
                path,
                out_dir / "side_by_side" / f"{path.stem}_top10.png",
                CATEGORIES,
                enhanced[name][:, idx],
                raw[name][:, idx],
                color_limit,
            )
        logging.info("Saved %d %s image comparisons under %s", len(paths), name, out_dir)

    (output_dir / "specificity_baseline.json").write_text(json.dumps({
        "definition": "class-average cosine mean across available input folders",
        "mean_average_cosine_by_class": dict(zip(CATEGORIES, baseline[:, 0].tolist())),
        "num_images": sum(matrix.shape[1] for matrix in raw.values()),
        "top_k": TOP_K,
    }, indent=2) + "\n")
    associations_path.write_text(json.dumps({
        "tuple_fields": ["image_class_id", "text_class_id"],
        "excluded_text_class_ids": sorted(ASSOCIATION_EXCLUDED_CLASS_IDS),
        "ambiguity_filter": f"discard images with {ASSOCIATION_MIN_MARGIN_CLASSES} or more text classes within {ASSOCIATION_MARGIN:.2f} of the top score",
        "minimum_best_mean_specificity": ASSOCIATION_MIN_SPECIFICITY,
        "associations": valid_associations,
    }, indent=2) + "\n")
    print(f"Valid (image class ID, text class ID) associations: {valid_associations}")
    return valid_associations


if __name__ == "__main__":
    main()
