#!/usr/bin/env python3
"""Rank indoor object categories for extracted mesh and Gaussian Splat frames."""

from __future__ import annotations

import csv
import argparse
import json
import logging
import os
import re
import shutil
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

CATEGORY_GROUPS = {
    1: ("door", "door-frame", "doorway"),
    2: ("window", "curtain"),
    3: ("stairs", "staircase", "steps", "stairway"),
    5: ("elevator", "lift"),
    7: ("radiator", "convector", "heater", "fireplace", "chimney"),
    # 8: ("gate",),  # Disabled while outdoor scenarios are not being acquired.
    # 9: ("fountain",),  # Disabled while outdoor scenarios are not being acquired.
    # 10: ("gazebo",),  # Disabled while outdoor scenarios are not being acquired.
    11: ("table", "desk", "dining table", "coffee table", "worktop"),
    12: ("cabinet", "closet", "wardrobe", "nightstand", "bedside table", "cupboard", "drawer"),
    13: ("chair", "office chair", "bench"),
    14: ("rug", "carpet", "mat"),
    15: ("bookshelf", "shelf", "ledge", "rack"),
    16: ("sofa", "couch", "armchair"),
    # 17: ("television", "monitor"),  # Disabled for the current indoor classification.
    18: ("bed", "crib", "pillow", "blanket", "duvet", "comforter"),
    19: ("cooktop", "hob", "burner", "stovetop", "gas stove", "electric stove", "hotplate"),
    20: ("refrigerator", "fridge"),
    21: ("oven",),
    22: ("dishwasher",),
    23: ("sink",),
    24: ("wc", "toilette", "urinal", "toilet", "squat toilet"),
    25: ("bidet",),
    26: ("shower", "jacuzzi", "tub"),
    27: ("washing machine", "dryer"),
    # 28: ("car", "motorbike"),  # Disabled while outdoor scenarios are not being acquired.
}
CATEGORY_NAMES = {
    **{category_id: words[0] for category_id, words in CATEGORY_GROUPS.items()},
    4: "ramp", 6: "column", 8: "gate", 9: "fountain", 10: "gazebo", 14: "rug",
    17: "television", 28: "car",
}
# Intervals are [min, max] for the complete oriented AABB; base_z is relative to local finished floor.
CATEGORY_GEOMETRY_CONSTRAINTS = {
    1:  {"width_m": (0.55, 2.50), "depth_m": (0.03, 1.00), "height_m": (1.75, 3.20), "base_z_m": (-0.20, 0.15)},
    2:  {"width_m": (0.30, 4.50), "depth_m": (0.03, 0.45), "height_m": (0.30, 3.00), "base_z_m": (-0.20, 1.80)},
    3:  {"width_m": (0.50, 5.00), "depth_m": (0.25, 12.00), "height_m": (0.60, 4.50), "base_z_m": (-0.20, 0.30)},
    4:  {"width_m": (0.60, 5.00), "depth_m": (0.50, 15.00), "height_m": (0.05, 3.00), "base_z_m": (-0.20, 0.40)},
    5:  {"width_m": (0.70, 3.50), "depth_m": (0.70, 3.50), "height_m": (1.90, 3.50), "base_z_m": (-0.20, 0.15)},
    6:  {"width_m": (0.10, 1.50), "depth_m": (0.10, 1.50), "height_m": (1.50, 6.00), "base_z_m": (-0.20, 0.30)},
    7:  {"width_m": (0.25, 2.50), "depth_m": (0.05, 0.90), "height_m": (0.20, 2.50), "base_z_m": (-0.20, 1.20)},
    8:  {"width_m": (0.60, 6.00), "depth_m": (0.03, 0.50), "height_m": (0.80, 3.50), "base_z_m": (-0.20, 0.30)},
    9:  {"width_m": (0.30, 6.00), "depth_m": (0.30, 6.00), "height_m": (0.15, 4.00), "base_z_m": (-0.20, 0.50)},
    10: {"width_m": (1.50, 8.00), "depth_m": (1.50, 8.00), "height_m": (2.00, 5.00), "base_z_m": (-0.20, 0.40)},
    11: {"width_m": (0.35, 4.00), "depth_m": (0.30, 2.00), "height_m": (0.25, 1.20), "base_z_m": (-0.20, 0.90)},
    12: {"width_m": (0.25, 6.00), "depth_m": (0.20, 1.20), "height_m": (0.20, 3.00), "base_z_m": (-0.20, 2.00)},
    13: {"width_m": (0.35, 3.00), "depth_m": (0.35, 1.20), "height_m": (0.35, 1.60), "base_z_m": (-0.20, 0.40)},
    14: {"width_m": (0.50, 6.00), "depth_m": (0.30, 5.00), "height_m": (0.002, 0.50), "base_z_m": (-0.20, 0.20)},
    15: {"width_m": (0.25, 4.00), "depth_m": (0.12, 0.80), "height_m": (0.03, 3.00), "base_z_m": (-0.20, 2.20)},
    16: {"width_m": (0.60, 4.50), "depth_m": (0.60, 2.50), "height_m": (0.50, 1.50), "base_z_m": (-0.20, 0.15)},
    17: {"width_m": (0.30, 3.00), "depth_m": (0.02, 0.35), "height_m": (0.20, 1.90), "base_z_m": (-0.20, 2.20)},
    18: {"width_m": (0.55, 2.60), "depth_m": (1.00, 2.80), "height_m": (0.20, 1.80), "base_z_m": (-0.20, 0.20)},
    19: {"width_m": (0.25, 1.30), "depth_m": (0.25, 0.90), "height_m": (0.01, 0.25), "base_z_m": (-0.20, 1.15)},
    20: {"width_m": (0.40, 1.50), "depth_m": (0.40, 1.00), "height_m": (0.45, 2.40), "base_z_m": (-0.20, 0.20)},
    21: {"width_m": (0.35, 1.00), "depth_m": (0.35, 0.90), "height_m": (0.30, 1.00), "base_z_m": (-0.20, 1.50)},
    22: {"width_m": (0.40, 0.90), "depth_m": (0.40, 0.85), "height_m": (0.40, 0.95), "base_z_m": (-0.20, 0.90)},
    23: {"width_m": (0.25, 2.50), "depth_m": (0.20, 0.90), "height_m": (0.08, 1.20), "base_z_m": (-0.20, 1.15)},
    24: {"width_m": (0.20, 1.00), "depth_m": (0.20, 1.20), "height_m": (0.10, 1.40), "base_z_m": (-0.20, 0.15)},
    25: {"width_m": (0.30, 0.75), "depth_m": (0.40, 0.85), "height_m": (0.25, 0.65), "base_z_m": (-0.20, 0.45)},
    26: {"width_m": (0.65, 2.80), "depth_m": (0.65, 2.80), "height_m": (0.10, 2.80), "base_z_m": (-0.20, 0.30)},
    27: {"width_m": (0.45, 0.90), "depth_m": (0.50, 0.95), "height_m": (0.65, 1.20), "base_z_m": (-0.20, 1.10)},
    28: {"width_m": (0.60, 2.60), "depth_m": (1.20, 6.50), "height_m": (0.80, 2.80), "base_z_m": (-0.20, 0.30)},
}
CATEGORY_IDS = list(CATEGORY_GROUPS)
CATEGORIES = [words[0] for words in CATEGORY_GROUPS.values()]
CATEGORY_WORDS = [word for group in CATEGORY_GROUPS.values() for word in group]
WORD_CATEGORY_IDS = [category_id for category_id, group in CATEGORY_GROUPS.items() for _ in group]
CATEGORY_ROW_BY_ID = {category_id: row for row, category_id in enumerate(CATEGORY_IDS)}
STRUCTURAL_ROLE_TOKENS = ("floor", "ceiling", "wall", "stair", "stairs")
STRUCTURAL_TEXT_CLASS_IDS = {1, 2}
ASSOCIATION_EXCLUDED_CLASS_IDS: set[int] = set()
ASSOCIATION_MIN_MARGIN_CLASSES = 5
ASSOCIATION_MARGIN = 0.02
ASSOCIATION_MIN_POSITIVE_MATCHES = 5
ASSOCIATION_MIN_POSITIVE_SCORE = 0.027
FRAME_PATTERN = re.compile(r"class_(\d+)_inst_(\d+)(?:_(base|up|down|left|right))?$")
SEMANTIC_FRAME_FILTER_RULE = (
    "For instances with positive mean scores, treat the top positive category as tentative; "
    "retain frames whose top positive category matches it and whose score is at least its "
    "pre-filter instance mean."
)


def specificity_matrices(raw_matrices: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Subtract each class's mean cosine across the available render sets."""
    baseline = np.concatenate(list(raw_matrices.values()), axis=1).mean(axis=1, keepdims=True)
    return {name: matrix - baseline for name, matrix in raw_matrices.items()}, baseline


def _structural_instance_keys(mesh_frames_dir: Path) -> set[tuple[int, int]]:
    manifest_path = mesh_frames_dir / "non_structural_frames_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Frame manifest required for structural-role filtering: {manifest_path}")
    instances = json.loads(manifest_path.read_text(encoding="utf-8")).get("extracted_instances", {})
    return {
        (int(record["class_id"]), int(record["instance_id"]))
        for record in instances.values()
        if any(token in str(record.get("role", "")).strip().lower() for token in STRUCTURAL_ROLE_TOKENS)
    }


def aggregate_valid_class_associations(
    image_paths: dict[str, list[Path]],
    enhanced_matrices: dict[str, np.ndarray],
    instance_dimensions: dict[tuple[int, int], tuple[float, float, float]],
) -> list[tuple[int, int]]:
    """Return source-class labels selected only from accepted, dimension-valid matches."""
    instance_scores = aggregate_instance_scores(image_paths, enhanced_matrices)
    instance_scores = {
        key: _filter_instance_scores_by_dimensions(scores, instance_dimensions.get(key))
        for key, scores in instance_scores.items()
    }
    class_assignments, _, _, _ = _class_level_decisions(instance_scores)
    return sorted(class_assignments.items())


def _aggregate_class_scores(
    instance_scores: dict[tuple[int, int], np.ndarray],
) -> dict[int, np.ndarray]:
    scores_by_class: dict[int, list[np.ndarray]] = {}
    for (class_id, _), scores in instance_scores.items():
        scores_by_class.setdefault(class_id, []).append(scores)
    return {
        class_id: np.mean(scores, axis=0)
        for class_id, scores in sorted(scores_by_class.items())
    }


def _class_level_decisions(
    instance_scores: dict[tuple[int, int], np.ndarray],
) -> tuple[dict[int, int], dict[int, np.ndarray], dict[int, int], dict[tuple[int, int], int]]:
    """Choose among accepted instance labels after their category scores were geometry-filtered."""
    class_category_scores: dict[int, dict[int, list[float]]] = {}
    instance_categories: dict[tuple[int, int], int] = {}
    all_class_ids = {class_id for class_id, _ in instance_scores}

    for instance_key, scores in instance_scores.items():
        category_id, _, _ = _association_status(filter_instance_match_scores(scores))
        if category_id is None:
            continue
        instance_categories[instance_key] = category_id
        score = float(scores[CATEGORY_ROW_BY_ID[category_id]])
        class_category_scores.setdefault(instance_key[0], {}).setdefault(category_id, []).append(score)

    class_assignments: dict[int, int] = {}
    class_scores: dict[int, np.ndarray] = {}
    valid_instance_counts: dict[int, int] = {}
    for class_id in sorted(all_class_ids):
        category_scores = class_category_scores.get(class_id, {})
        valid_instance_counts[class_id] = sum(map(len, category_scores.values()))
        scores = np.zeros(len(CATEGORIES), dtype=np.float32)
        if category_scores:
            means = {
                category_id: float(np.mean(values))
                for category_id, values in category_scores.items()
            }
            winner = min(
                means,
                key=lambda category_id: (
                    -len(category_scores[category_id]),
                    -means[category_id],
                    category_id,
                ),
            )
            class_assignments[class_id] = winner
            for category_id, mean_score in means.items():
                scores[CATEGORY_ROW_BY_ID[category_id]] = mean_score
        class_scores[class_id] = scores

    return class_assignments, class_scores, valid_instance_counts, instance_categories


def aggregate_instance_scores(
    image_paths: dict[str, list[Path]],
    enhanced_matrices: dict[str, np.ndarray],
) -> dict[tuple[int, int], np.ndarray]:
    """Average valid mesh/GS frame scores by (source class ID, component instance ID)."""
    row_ids = [category_id for category_id in CATEGORY_IDS if category_id not in ASSOCIATION_EXCLUDED_CLASS_IDS]
    rows = np.asarray([CATEGORY_ROW_BY_ID[category_id] for category_id in row_ids], dtype=np.int64)
    scores_by_instance: dict[tuple[int, int], list[np.ndarray]] = {}
    for render_set, paths in image_paths.items():
        scores = enhanced_matrices[render_set][rows]
        for image_idx, path in enumerate(paths):
            match = FRAME_PATTERN.fullmatch(path.stem)
            if match is None:
                raise ValueError(f"Unexpected category frame name: {path.name}")
            class_id, instance_id = int(match.group(1)), int(match.group(2))
            full_scores = np.full(len(CATEGORIES), -np.inf, dtype=np.float32)
            full_scores[rows] = scores[:, image_idx]
            scores_by_instance.setdefault((class_id, instance_id), []).append(full_scores)
    return {
        instance_key: np.mean(frame_scores, axis=0)
        for instance_key, frame_scores in scores_by_instance.items()
    }


def _accepted_category(scores: np.ndarray) -> int | None:
    """Accept the top positive category unless five positive categories are ambiguous."""
    positive = scores > 0.0
    if not np.any(positive):
        return None
    best_score = float(np.max(scores[positive]))
    if int(np.count_nonzero(positive & (scores >= best_score - ASSOCIATION_MARGIN))) >= ASSOCIATION_MIN_MARGIN_CLASSES:
        return None
    best_row = int(np.argmax(scores))
    return CATEGORY_IDS[best_row]


def _association_status(scores: np.ndarray) -> tuple[int | None, str, int]:
    positive = scores > 0.0
    if not np.any(positive):
        return None, "no_positive_scores", 0
    best_score = float(np.max(scores[positive]))
    close_positive = int(np.count_nonzero(positive & (scores >= best_score - ASSOCIATION_MARGIN)))
    if close_positive >= ASSOCIATION_MIN_MARGIN_CLASSES:
        return None, "ambiguous", close_positive
    best_row = int(np.argmax(scores))
    return CATEGORY_IDS[best_row], "accepted", close_positive


def filter_instance_match_scores(scores: np.ndarray) -> np.ndarray:
    """Drop weak positive instance matches once at least five are available."""
    scores = np.asarray(scores)
    positive = scores > 0.0
    if np.count_nonzero(positive) < ASSOCIATION_MIN_POSITIVE_MATCHES:
        return scores.copy()
    return np.where(positive & (scores >= ASSOCIATION_MIN_POSITIVE_SCORE), scores, -np.inf)


def filter_semantically_inconsistent_frames(
    frames: dict[str, dict[str, dict[str, object]]],
    instance_scores: np.ndarray,
) -> tuple[dict[str, dict[str, dict[str, object]]], list[dict[str, object]]]:
    """Keep frames that support the strongest positive instance category as strongly as its mean."""
    positive_rows = np.flatnonzero(instance_scores > 0.0)
    if not len(positive_rows):
        return frames, []

    # Even an ambiguous initial mean can identify a tentative category; filtering
    # weak frames may resolve that ambiguity before the final association step.
    category_row = int(positive_rows[np.argmax(instance_scores[positive_rows])])
    mean_score = float(instance_scores[category_row])
    kept: dict[str, dict[str, dict[str, object]]] = {}
    rejected: list[dict[str, object]] = []
    for view, render_sets in frames.items():
        for render_set, item in render_sets.items():
            scores = np.asarray(item["scores"])
            positive_rows = np.flatnonzero(scores > 0.0)
            top_row = int(positive_rows[np.argmax(scores[positive_rows])]) if len(positive_rows) else None
            category_score = float(scores[category_row])
            if top_row == category_row and category_score >= mean_score:
                kept.setdefault(view, {})[render_set] = item
                continue
            rejected.append({
                "view": view,
                "render_set": render_set,
                "image": Path(item["path"]).name,
                "candidate_category": CATEGORIES[category_row],
                "candidate_score": category_score,
                "instance_mean_score": mean_score,
                "frame_top_category": CATEGORIES[top_row] if top_row is not None else None,
                "reason": "top_category_mismatch" if top_row != category_row else "below_instance_category_mean",
            })

    # Keep the original evidence if no individual frame can support the decision.
    return (kept, rejected) if kept else (frames, [])


def _positive_top_rows(scores: np.ndarray) -> np.ndarray:
    positive_rows = np.flatnonzero(scores > 0.0)
    return positive_rows[np.argsort(-scores[positive_rows], kind="stable")[:TOP_K]]


def _instance_report_rows(mean_scores: np.ndarray) -> list[list[str]]:
    rows = []
    for row in _positive_top_rows(mean_scores):
        category_id = CATEGORY_IDS[row]
        bounds = CATEGORY_GEOMETRY_CONSTRAINTS[category_id]
        dimensions = " · ".join(
            f"{label} {bounds[key][0]:g}–{bounds[key][1]:g}"
            for label, key in (("W", "width_m"), ("D", "depth_m"),
                               ("H", "height_m"), ("Base Z", "base_z_m"))
        )
        rows.append([CATEGORIES[row], dimensions, f"{mean_scores[row]:+.3f}"])
    return rows


def _instance_bbox_dimensions(output_dir: Path) -> dict[tuple[int, int], tuple[float, float, float]]:
    manifest_path = output_dir.parent / "segmentation.json"
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        (int(item["class_id"]), int(item["instance_index"]) - 1): tuple(
            float(maximum - minimum)
            for minimum, maximum in zip(item["bounds_min"], item["bounds_max"])
        )
        for item in manifest.get("instances", [])
        if len(item.get("bounds_min", [])) == 3
        and len(item.get("bounds_max", [])) == 3
        and "instance_index" in item
    }


def _merge_phase07_split_door_records(
    old_data: dict[str, object],
    associations: list[list[int]],
    instances: list[dict[str, object]],
    door_class_ids: set[int],
) -> tuple[list[list[int]], list[dict[str, object]]]:
    """Keep split-door labels only while their source class is still classified as a door."""
    split_rows = [
        row for row in old_data.get("instances", [])
        if isinstance(row, dict)
        and row.get("classification_source") == "phase07_door_split"
        and int(row["source_class_id"]) in door_class_ids
    ]
    if not split_rows:
        return associations, instances

    split_keys = {
        (int(row["source_class_id"]), int(row["instance_id"]))
        for row in split_rows
    }
    merged_associations = {
        (int(row[0]), int(row[1]), int(row[2]))
        for row in associations
        if (int(row[0]), int(row[1])) not in split_keys
    }
    merged_associations.update(
        (int(row["source_class_id"]), int(row["instance_id"]), int(row["text_class_id"]))
        for row in split_rows
    )
    merged_instances = [
        row for row in instances
        if (int(row["source_class_id"]), int(row["instance_id"])) not in split_keys
    ]
    merged_instances.extend(split_rows)
    return sorted([list(row) for row in merged_associations]), sorted(
        merged_instances,
        key=lambda row: (int(row["source_class_id"]), int(row["instance_id"])),
    )


def _format_instance_bbox(dimensions: tuple[float, float, float] | None) -> str:
    if dimensions is None:
        return "BBox X × Y × Z unavailable"
    return "BBox X × Y × Z: " + " × ".join(f"{value:.2f}" for value in dimensions) + " m"


def _instance_dimensions_match_category(
    dimensions: tuple[float, float, float] | None,
    category_id: int,
) -> bool:
    if dimensions is None or category_id not in CATEGORY_GEOMETRY_CONSTRAINTS:
        return False
    x_dim, height, z_dim = dimensions
    bounds = CATEGORY_GEOMETRY_CONSTRAINTS[category_id]

    def within(value: float, interval: tuple[float, float]) -> bool:
        return interval[0] <= value <= interval[1]

    width = bounds["width_m"]
    depth = bounds["depth_m"]
    valid_height = within(height, bounds["height_m"])
    return valid_height and (
        (within(x_dim, width) and within(z_dim, depth))
        or (within(z_dim, width) and within(x_dim, depth))
    )


def _filter_instance_scores_by_dimensions(
    scores: np.ndarray,
    dimensions: tuple[float, float, float] | None,
) -> np.ndarray:
    """Keep category scores only when the instance AABB fits that category's bounds."""
    filtered = np.asarray(scores).copy()
    for category_id, row in CATEGORY_ROW_BY_ID.items():
        if not _instance_dimensions_match_category(dimensions, category_id):
            filtered[row] = -np.inf
    return filtered


def _save_instance_report(
    output_path: Path,
    class_id: int,
    instance_id: int,
    frames: dict[str, dict[str, dict[str, object]]],
    mean_scores: np.ndarray,
    bbox_dimensions: tuple[float, float, float] | None = None,
) -> None:
    views = ["base", "up", "down", "left", "right"]
    valid_frames = [
        (view, render_set, item)
        for view in views
        for render_set in ("mesh", "gs")
        if (item := frames.get(view, {}).get(render_set)) is not None
    ]

    fig = plt.figure(figsize=(19, 10), layout="constrained")
    outer = fig.add_gridspec(1, 2, width_ratios=(1.25, 1.0))
    collage = outer[0].subgridspec(2, 5)
    for slot in range(10):
        ax = fig.add_subplot(collage[slot // 5, slot % 5])
        if slot < len(valid_frames):
            view, render_set, item = valid_frames[slot]
            with Image.open(item["path"]) as source:
                ax.imshow(source.convert("RGB"))
            ax.set_title(f"{view} · {render_set}", fontsize=9)
        ax.axis("off")

    table_ax = fig.add_subplot(outer[1])
    table_ax.axis("off")
    mean_count = sum(len(by_view) for by_view in frames.values())
    table_ax.set_title(
        f"class {class_id} / instance {instance_id}\n"
        f"{_format_instance_bbox(bbox_dimensions)}\n"
        f"mean over {mean_count} valid frames",
        fontsize=12,
    )
    cells = _instance_report_rows(mean_scores)
    if cells:
        row_category_ids = [CATEGORY_IDS[row] for row in _positive_top_rows(mean_scores)]
        row_colors = [
            "#d9ead3" if _instance_dimensions_match_category(bbox_dimensions, category_id) else "#f4cccc"
            for category_id in row_category_ids
        ]
        table = table_ax.table(
            cellText=cells,
            cellColours=[[color] * 3 for color in row_colors],
            colLabels=["Instance name", "Min–max dimensions (m)", "Final similarity"],
            loc="center",
            cellLoc="center",
            colWidths=[0.23, 0.55, 0.22],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 2.1)
    else:
        table_ax.text(0.5, 0.5, "No positive category averages", ha="center", va="center")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _save_class_report(
    output_path: Path,
    class_id: int,
    mean_scores: np.ndarray,
    category_id: int | None,
    status: str,
    instance_count: int,
) -> None:
    order = _positive_top_rows(mean_scores)
    decision = CATEGORIES[CATEGORY_ROW_BY_ID[category_id]] if category_id is not None else status.replace("_", " ")
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    fig.suptitle(f"Source class {class_id} — final decision: {decision}", fontsize=14, fontweight="bold")
    ax.set_title(f"Top positive categories · equal-weight mean of {instance_count} instances")
    ax.axvline(0.0, color="#555555", linewidth=0.8)
    if len(order):
        values = mean_scores[order]
        labels = [CATEGORIES[row] for row in order]
        colors = ["#2a9d8f" if CATEGORY_IDS[row] == category_id else "#7895b2" for row in order]
        bars = ax.barh(np.arange(len(order)), values, color=colors)
        ax.set_yticks(np.arange(len(order)), labels)
        ax.invert_yaxis()
        ax.set_xlim(0.0, float(values.max()) * 1.2)
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + float(values.max()) * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{value:+.3f}", va="center", fontsize=9)
        ax.grid(axis="x", alpha=0.2)
    else:
        ax.text(0.5, 0.5, "No positive class-average scores", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_yticks([])
    ax.set_xlabel("Mean specificity score")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def top_rows(
    class_names: list[str],
    class_words: list[tuple[str, ...]],
    image_names: list[str],
    raw: np.ndarray,
    enhanced: np.ndarray,
    top_k: int = TOP_K,
    category_ids: list[int] | None = None,
    frame_status: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    category_ids = category_ids or list(range(1, len(class_names) + 1))
    rows = []
    for image_idx, image_name in enumerate(image_names):
        eligible = np.flatnonzero(np.isfinite(enhanced[:, image_idx]))
        selected = eligible[np.argsort(-enhanced[eligible, image_idx], kind="stable")[:top_k]]
        for rank, class_idx in enumerate(selected, 1):
            rows.append({
                "image": image_name,
                "rank": rank,
                "class_id": int(category_ids[class_idx]),
                "category": class_names[class_idx],
                "words": " | ".join(class_words[class_idx]),
                "enhanced_similarity": float(enhanced[class_idx, image_idx]),
                "average_cosine_similarity": float(raw[class_idx, image_idx]),
                "frame_status": (frame_status or {}).get(image_name, "valid"),
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
    previous_instances_path = output_dir / "valid_instance_associations.json"
    previous_instances_data = (
        json.loads(previous_instances_path.read_text(encoding="utf-8"))
        if previous_instances_path.is_file() else {}
    )
    shutil.rmtree(output_dir / "instance_reports", ignore_errors=True)
    shutil.rmtree(output_dir / "class_reports", ignore_errors=True)
    for render_set in ("mesh", "gs"):
        shutil.rmtree(output_dir / render_set / "side_by_side", ignore_errors=True)
    class_words = list(CATEGORY_GROUPS.values())
    excluded_category_ids = sorted((set(range(1, 29)) - set(CATEGORY_IDS)) | ASSOCIATION_EXCLUDED_CLASS_IDS)
    (output_dir / "categories.json").write_text(json.dumps([
        {
            "id": category_id,
            "name": CATEGORY_NAMES[category_id],
            "active": category_id in CATEGORY_GROUPS,
            "words": list(CATEGORY_GROUPS.get(category_id, ())),
            "geometry_constraints": CATEGORY_GEOMETRY_CONSTRAINTS[category_id],
        }
        for category_id in range(1, 29)
    ], indent=2) + "\n")
    associations_path = output_dir / "valid_class_associations.json"
    if not any(image_paths.values()):
        dimensions = _instance_bbox_dimensions(output_dir)
        class_instances: dict[int, list[int]] = {}
        for class_id, instance_id in dimensions:
            class_instances.setdefault(class_id, []).append(instance_id)
        class_decisions = []
        instance_reports = []
        for class_id, instance_ids in sorted(class_instances.items()):
            class_decisions.append({
                "source_class_id": class_id,
                "status": "unassigned_no_valid_instances",
                "valid_instance_count": 0,
                "instance_ids": sorted(instance_ids),
                "propagated_instance_ids": [],
            })
            _save_class_report(
                output_dir / "class_reports" / f"class_{class_id:03d}.png",
                class_id,
                np.zeros(len(CATEGORIES), dtype=np.float32),
                None,
                "unassigned_no_valid_instances",
                len(instance_ids),
            )
            instance_reports.extend({
                "source_class_id": class_id,
                "instance_id": instance_id,
                "status": "no_valid_frames",
                "valid_frame_count": 0,
                "text_class_id": None,
            } for instance_id in sorted(instance_ids))
        for render_set in ("mesh", "gs"):
            for stale_file in ("top10_similarity.csv", "summary.json"):
                (output_dir / render_set / stale_file).unlink(missing_ok=True)
        (output_dir / "specificity_baseline.json").write_text(json.dumps({
            "definition": "class-average cosine mean across available input folders",
            "mean_average_cosine_by_class": {},
            "num_images": 0,
            "top_k": TOP_K,
        }, indent=2) + "\n")
        associations_path.write_text(json.dumps({
            "tuple_fields": ["image_class_id", "text_class_id"],
            "excluded_text_class_ids": excluded_category_ids,
            "structural_category_filter": "structural-role instances are classified only against door (1) and window (2)",
            "ambiguity_filter": f"discard source classes with {ASSOCIATION_MIN_MARGIN_CLASSES} or more positive mean category scores within {ASSOCIATION_MARGIN:.2f} of the top positive score",
            "dimension_filter_rule": "only instance-category matches whose AABB fits the category width, depth, and height bounds are eligible; horizontal axes may swap and floor elevation is ignored",
            "class_assignment_rule": "choose among accepted dimension-valid instance labels; the most frequent label wins, with mean accepted similarity as the tie-breaker, then propagate it to every instance of that source class; leave classes without a valid instance unassigned",
            "associations": [],
            "class_decisions": class_decisions,
        }, indent=2) + "\n")
        empty_associations, instance_reports = _merge_phase07_split_door_records(
            previous_instances_data, [], instance_reports, set()
        )
        (output_dir / "valid_instance_associations.json").write_text(json.dumps({
            "tuple_fields": ["source_class_id", "instance_id", "text_class_id"],
            "excluded_text_class_ids": excluded_category_ids,
            "structural_category_filter": "structural-role instances are classified only against door (1) and window (2)",
            "dimension_filter_rule": "only instance-category matches whose AABB fits the category width, depth, and height bounds are eligible; horizontal axes may swap and floor elevation is ignored",
            "ambiguity_filter": f"discard instances with {ASSOCIATION_MIN_MARGIN_CLASSES} or more positive mean category scores within {ASSOCIATION_MARGIN:.2f} of the top positive score",
            "positive_score_filter": f"when at least {ASSOCIATION_MIN_POSITIVE_MATCHES} positive mean scores exist, retain only scores >= {ASSOCIATION_MIN_POSITIVE_SCORE:.3f}",
            "associations": empty_associations,
            "instances": instance_reports,
        }, indent=2) + "\n")
        (output_dir / "semantic_frame_filter.json").write_text(json.dumps({
            "rule": SEMANTIC_FRAME_FILTER_RULE,
            "discarded_frames": [],
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
    offsets = np.cumsum([0, *(len(group) for group in CATEGORY_GROUPS.values())])
    raw = {
        name: np.stack([
            matrix[offsets[idx]:offsets[idx + 1]].mean(axis=0)
            for idx in range(len(CATEGORY_GROUPS))
        ])
        for name, matrix in word_similarities.items()
    }
    assert all(matrix.shape[0] == len(CATEGORIES) for matrix in raw.values())
    enhanced, baseline = specificity_matrices(raw)
    structural_instance_keys = _structural_instance_keys(args.mesh_frames_dir)
    restricted_rows = [
        CATEGORY_ROW_BY_ID[category_id]
        for category_id in CATEGORY_IDS
        if category_id not in STRUCTURAL_TEXT_CLASS_IDS
    ]
    instance_frames: dict[tuple[int, int], dict[str, dict[str, dict[str, object]]]] = {}
    for render_set, paths in image_paths.items():
        for image_idx, path in enumerate(paths):
            match = FRAME_PATTERN.fullmatch(path.stem)
            if match is None:
                raise ValueError(f"Unexpected category frame name: {path.name}")
            class_id, instance_id = int(match.group(1)), int(match.group(2))
            view = match.group(3) or "base"
            if (class_id, instance_id) in structural_instance_keys:
                enhanced[render_set][restricted_rows, image_idx] = -np.inf
                raw[render_set][restricted_rows, image_idx] = -np.inf
            instance_frames.setdefault((class_id, instance_id), {}).setdefault(view, {})[render_set] = {
                "path": path,
                "scores": enhanced[render_set][:, image_idx],
            }

    initial_instance_means = {
        instance_key: np.mean(
            [item["scores"] for by_view in views.values() for item in by_view.values()], axis=0
        )
        for instance_key, views in instance_frames.items()
    }
    semantic_frame_rejections = []
    rejected_frame_status: dict[str, dict[str, str]] = {name: {} for name in image_paths}
    filtered_instance_frames = {}
    for instance_key, views in instance_frames.items():
        filtered, rejected = filter_semantically_inconsistent_frames(
            views, initial_instance_means[instance_key]
        )
        filtered_instance_frames[instance_key] = filtered
        semantic_frame_rejections.extend({
            "source_class_id": instance_key[0],
            "instance_id": instance_key[1],
            **item,
        } for item in rejected)
        for item in rejected:
            rejected_frame_status[item["render_set"]][item["image"]] = (
                f"discarded: {item['reason']}"
            )
    instance_frames = filtered_instance_frames
    instance_means = {
        instance_key: np.mean(
            [item["scores"] for by_view in views.values() for item in by_view.values()], axis=0
        )
        for instance_key, views in instance_frames.items()
    }
    filtered_instance_means = {
        instance_key: filter_instance_match_scores(scores)
        for instance_key, scores in instance_means.items()
    }
    bbox_dimensions = _instance_bbox_dimensions(output_dir)
    for instance_key in bbox_dimensions:
        filtered_instance_means.setdefault(
            instance_key, np.zeros(len(CATEGORIES), dtype=np.float32)
        )
    dimension_filtered_instance_means = {
        key: _filter_instance_scores_by_dimensions(scores, bbox_dimensions.get(key))
        for key, scores in filtered_instance_means.items()
    }
    class_assignments, class_means, valid_instance_counts, instance_categories = (
        _class_level_decisions(dimension_filtered_instance_means)
    )
    instance_associations = []
    instance_reports = []
    for (class_id, instance_id), mean_scores in sorted(filtered_instance_means.items()):
        decision_scores = dimension_filtered_instance_means[(class_id, instance_id)]
        instance_category_id, status, ambiguity_count = _association_status(decision_scores)
        if status == "no_positive_scores" and _association_status(mean_scores)[0] is not None:
            status = "no_valid_dimensions"
        category_id = class_assignments.get(class_id)
        views = instance_frames.get((class_id, instance_id), {})
        if views:
            _save_instance_report(
                output_dir / "instance_reports" / f"class_{class_id:03d}_inst_{instance_id:03d}.png",
                class_id,
                instance_id,
                views,
                mean_scores,
                bbox_dimensions.get((class_id, instance_id)),
            )
        instance_report = {
            "source_class_id": class_id,
            "instance_id": instance_id,
            "status": status if views else "no_valid_frames",
            "positive_categories_within_margin": ambiguity_count,
            "valid_frame_count": int(sum(len(by_view) for by_view in views.values())),
            "semantic_discarded_frame_count": sum(
                item["source_class_id"] == class_id and item["instance_id"] == instance_id
                for item in semantic_frame_rejections
            ),
        }
        if category_id is not None:
            instance_report["text_class_id"] = category_id
            instance_report["association_source"] = (
                "instance" if instance_category_id == category_id else "class_propagated"
            )
            if instance_category_id is not None and instance_category_id != category_id:
                instance_report["instance_text_class_id"] = instance_category_id
            instance_associations.append([class_id, instance_id, category_id])
        instance_report["category_mean_specificity"] = {
            str(category_id): float(decision_scores[row])
            for row, category_id in enumerate(CATEGORY_IDS)
            if decision_scores[row] > 0.0
        }
        if views:
            instance_report["report_image"] = f"instance_reports/class_{class_id:03d}_inst_{instance_id:03d}.png"
        instance_reports.append(instance_report)

    class_decisions = []
    for class_id, mean_scores in class_means.items():
        category_id = class_assignments.get(class_id)
        status = "accepted" if category_id is not None else "unassigned_no_valid_instances"
        instance_ids = sorted(
            instance_id for source_class_id, instance_id in filtered_instance_means
            if source_class_id == class_id
        )
        class_decision = {
            "source_class_id": class_id,
            "status": status,
            "valid_instance_count": valid_instance_counts[class_id],
            "instance_ids": instance_ids,
            "propagated_instance_ids": [
                instance_id for instance_id in instance_ids
                if category_id is not None and instance_categories.get((class_id, instance_id)) != category_id
            ],
        }
        if category_id is not None:
            class_decision["text_class_id"] = category_id
        class_decisions.append(class_decision)
        _save_class_report(
            output_dir / "class_reports" / f"class_{class_id:03d}.png",
            class_id,
            mean_scores,
            category_id,
            status,
            valid_instance_counts[class_id],
        )

    valid_associations = sorted(class_assignments.items())
    instance_associations, instance_reports = _merge_phase07_split_door_records(
        previous_instances_data,
        instance_associations,
        instance_reports,
        {class_id for class_id, category_id in valid_associations if category_id == 1},
    )
    (output_dir / "valid_instance_associations.json").write_text(json.dumps({
        "tuple_fields": ["source_class_id", "instance_id", "text_class_id"],
        "excluded_text_class_ids": excluded_category_ids,
        "structural_category_filter": "structural-role instances are classified only against door (1) and window (2)",
        "dimension_filter_rule": "only instance-category matches whose AABB fits the category width, depth, and height bounds are eligible; horizontal axes may swap and floor elevation is ignored",
        "ambiguity_filter": f"discard instances with {ASSOCIATION_MIN_MARGIN_CLASSES} or more positive mean category scores within {ASSOCIATION_MARGIN:.2f} of the top positive score",
        "positive_score_filter": f"when at least {ASSOCIATION_MIN_POSITIVE_MATCHES} positive mean scores exist, retain only scores >= {ASSOCIATION_MIN_POSITIVE_SCORE:.3f}",
        "associations": instance_associations,
        "instances": instance_reports,
    }, indent=2) + "\n")
    (output_dir / "semantic_frame_filter.json").write_text(json.dumps({
        "rule": SEMANTIC_FRAME_FILTER_RULE,
        "discarded_frames": semantic_frame_rejections,
    }, indent=2) + "\n")
    for name, paths in image_paths.items():
        image_names = [path.name for path in paths]
        out_dir = output_dir / name
        rows = top_rows(
            CATEGORIES, class_words, image_names, raw[name], enhanced[name],
            category_ids=CATEGORY_IDS, frame_status=rejected_frame_status[name],
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
                {"id": category_id, "name": name, "words": list(words)}
                for category_id, name, words in zip(CATEGORY_IDS, CATEGORIES, CATEGORY_GROUPS.values())
            ],
            "top_classes_per_image": {
                image: [row for row in rows if row["image"] == image]
                for image in image_names
            },
            "semantically_discarded_frames": [
                item for item in semantic_frame_rejections if item["render_set"] == name
            ],
        }, indent=2) + "\n")
        logging.info("Saved %d %s image comparisons under %s", len(paths), name, out_dir)

    (output_dir / "specificity_baseline.json").write_text(json.dumps({
        "definition": "class-average cosine mean across available input folders",
        "mean_average_cosine_by_class": dict(zip(CATEGORIES, baseline[:, 0].tolist())),
        "num_images": sum(matrix.shape[1] for matrix in raw.values()),
        "top_k": TOP_K,
    }, indent=2) + "\n")
    associations_path.write_text(json.dumps({
        "tuple_fields": ["image_class_id", "text_class_id"],
        "excluded_text_class_ids": excluded_category_ids,
        "structural_category_filter": "structural-role instances are classified only against door (1) and window (2)",
        "dimension_filter_rule": "only instance-category matches whose AABB fits the category width, depth, and height bounds are eligible; horizontal axes may swap and floor elevation is ignored",
        "class_assignment_rule": "choose among accepted dimension-valid instance labels; the most frequent label wins, with mean accepted similarity as the tie-breaker, then propagate it to every instance of that source class; leave classes without a valid instance unassigned",
        "associations": valid_associations,
        "class_decisions": class_decisions,
    }, indent=2) + "\n")
    print(f"Valid (image class ID, text class ID) associations: {valid_associations}")
    return valid_associations


if __name__ == "__main__":
    main()
