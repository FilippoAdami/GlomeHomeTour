#!/usr/bin/env python3
"""Compute semantic cosine similarity between class names and images using EmbeddingGemma2.

Supports:
- Direct class name embeddings or 10 fixed template prompt averaging.
- Full 768-dimensional embeddings or PCA dimensionality reduction (e.g. 64 principal components).
- Confidence enhancement postprocessing (Calibrated Softmax, Column Z-score, Class Specificity).
- Maximum-weight bipartite matching (Hungarian algorithm) for optimal 1-to-1 correspondence.
- Automated certainty determination: returns set of (image_path, assigned_class, is_certain) tuples.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Sequence, Union, Optional, Any, Set, Tuple

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
import torch

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("semantic_similarity")

DEFAULT_MODEL_NAME = "google/embeddinggemma-2"

# 10 fixed predefined templates for class prompt averaging
DEFAULT_PROMPT_TEMPLATES: list[str] = [
    "a photo of a {class_name}",
    "an interior picture showing a {class_name}",
    "a clean photo of a {class_name}",
    "a close-up photo of the {class_name}",
    "an indoor shot containing a {class_name}",
    "a view of a {class_name} in a room",
    "a picture of a {class_name} inside a home",
    "a clear depiction of a {class_name}",
    "a scene featuring a {class_name}",
    "an architectural photo with a {class_name}",
]

DEFAULT_TEST_CLASSES: list[str] = [
    "bench", "person", "desk", "office", "drawer", "cabinet", "library",
    "books", "mirror", "appliance", "boxing bag", "puzzle", "poster",
    "frame", "radiator", "termostat", "world", "map", "towel", "woman",
    "wardrobe", "curtain", "door", "stairs", "wall", "ceiling", "wood"
]


@dataclass
class SimilarityResult:
    """Encapsulates the semantic similarity matrix and associated metadata."""
    matrix: np.ndarray  # shape: (n_classes, n_images)
    class_embeddings: list[tuple[str, np.ndarray]]
    image_embeddings: np.ndarray  # shape: (n_images, embedding_dim)
    class_names: list[str]
    image_names: list[str]
    image_paths: list[str] = field(default_factory=list)
    pca_dim: Optional[int] = None
    use_templates: bool = True
    postprocess_mode: str = "raw"
    temperature: Optional[float] = None

    def __post_init__(self):
        if not self.image_paths:
            self.image_paths = list(self.image_names)

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    def __getitem__(self, item: Any) -> Any:
        return self.matrix[item]

    def __array__(self) -> np.ndarray:
        return self.matrix

    def to_dataframe(self):
        """Convert similarity matrix to pandas DataFrame with classes as rows and images as columns."""
        import pandas as pd
        return pd.DataFrame(self.matrix, index=self.class_names, columns=self.image_names)

    def top_classes_per_image(self, top_k: int = 3) -> dict[str, list[tuple[str, float]]]:
        """Return the top-k highest similarity classes for each image."""
        rankings = {}
        for col_idx, img_name in enumerate(self.image_names):
            sims = self.matrix[:, col_idx]
            top_indices = np.argsort(-sims)[:top_k]
            rankings[img_name] = [
                (self.class_names[idx], float(sims[idx])) for idx in top_indices
            ]
        return rankings

    def to_softmax(self, temperature: float = 0.02) -> SimilarityResult:
        """Compute column-wise temperature-scaled Softmax probabilities across classes."""
        shift = self.matrix - np.max(self.matrix, axis=0, keepdims=True)
        exp_m = np.exp(shift / temperature)
        probs = exp_m / np.sum(exp_m, axis=0, keepdims=True)
        return SimilarityResult(
            matrix=probs,
            class_embeddings=self.class_embeddings,
            image_embeddings=self.image_embeddings,
            class_names=self.class_names,
            image_names=self.image_names,
            image_paths=self.image_paths,
            pca_dim=self.pca_dim,
            use_templates=self.use_templates,
            postprocess_mode="softmax",
            temperature=temperature,
        )

    def to_zscore(self) -> SimilarityResult:
        """Compute column-wise Z-scores (standard deviations above column mean)."""
        means = np.mean(self.matrix, axis=0, keepdims=True)
        stds = np.maximum(np.std(self.matrix, axis=0, keepdims=True), 1e-12)
        zscores = (self.matrix - means) / stds
        return SimilarityResult(
            matrix=zscores,
            class_embeddings=self.class_embeddings,
            image_embeddings=self.image_embeddings,
            class_names=self.class_names,
            image_names=self.image_names,
            image_paths=self.image_paths,
            pca_dim=self.pca_dim,
            use_templates=self.use_templates,
            postprocess_mode="zscore",
        )

    def to_centered(self) -> SimilarityResult:
        """Subtract each class's scene average baseline (class specificity)."""
        row_means = np.mean(self.matrix, axis=1, keepdims=True)
        centered = self.matrix - row_means
        return SimilarityResult(
            matrix=centered,
            class_embeddings=self.class_embeddings,
            image_embeddings=self.image_embeddings,
            class_names=self.class_names,
            image_names=self.image_names,
            image_paths=self.image_paths,
            pca_dim=self.pca_dim,
            use_templates=self.use_templates,
            postprocess_mode="centered",
        )

    def one_to_one_correspondence(
        self,
        scoring: str = "specificity",
        temperature: float = 0.02,
        margin_threshold: float = 0.012,
        prob_ratio_threshold: float = 0.40,
        min_confidence: float = 0.30,
        use_full_path: bool = True,
    ) -> Set[Tuple[str, str, bool]]:
        """Compute optimal one-to-one bipartite matching and return as a set of tuples:
        {(image_path, assigned_class, is_certain), ...}

        A result is flagged as uncertain (is_certain=False) if multiple plausible classes
        exhibit close similarity or confidence margins.

        Args:
            scoring: 'specificity' (recommended, removes generic ambient room bias),
                     'softmax' (temperature-scaled probabilities),
                     'raw' (raw cosine similarities).
            temperature: softmax temperature (default 0.02).
            margin_threshold: minimum margin over runner-up to be considered certain (default 0.012).
            prob_ratio_threshold: max allowed runner_up_prob / top_prob ratio for certainty (default 0.40).
            min_confidence: minimum softmax probability required for certainty (default 0.30).
            use_full_path: if True, returns full image file path; if False, returns filename.

        Returns:
            A set of (image_path, assigned_class, is_certain) tuples.
        """
        row_means = np.mean(self.matrix, axis=1, keepdims=True)
        specificity_matrix = self.matrix - row_means

        shift = self.matrix - np.max(self.matrix, axis=0, keepdims=True)
        exp_m = np.exp(shift / temperature)
        prob_matrix = exp_m / np.sum(exp_m, axis=0, keepdims=True)

        if scoring == "specificity":
            cost_matrix = -specificity_matrix
        elif scoring == "softmax":
            cost_matrix = -prob_matrix
        else:
            cost_matrix = -self.matrix

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignments: Set[Tuple[str, str, bool]] = set()
        for r, c in zip(row_ind, col_ind):
            path_or_name = self.image_paths[c] if use_full_path and self.image_paths else self.image_names[c]
            cls = self.class_names[r]
            spec = float(specificity_matrix[r, c])
            raw_sim = float(self.matrix[r, c])
            top_p = float(prob_matrix[r, c])

            # Specificity competitor
            sorted_spec_indices = np.argsort(-specificity_matrix[:, c])
            spec_runner_idx = sorted_spec_indices[1] if sorted_spec_indices[0] == r else sorted_spec_indices[0]
            spec_margin = spec - float(specificity_matrix[spec_runner_idx, c])

            # Raw similarity competitor (checking direct semantic rival on this image)
            sorted_raw_indices = np.argsort(-self.matrix[:, c])
            raw_runner_idx = sorted_raw_indices[1] if sorted_raw_indices[0] == r else sorted_raw_indices[0]
            raw_margin = raw_sim - float(self.matrix[raw_runner_idx, c])
            runner_p = float(prob_matrix[raw_runner_idx, c])
            prob_ratio = runner_p / max(top_p, 1e-12)

            # Certain if wide margin in both specificity and raw similarity, runner up doesn't rival winner, and confidence is solid
            is_certain = bool(
                spec_margin >= margin_threshold
                and raw_margin >= margin_threshold
                and prob_ratio <= prob_ratio_threshold
                and top_p >= min_confidence
            )

            assignments.add((str(path_or_name), cls, is_certain))

        return assignments

    def detailed_one_to_one(
        self,
        scoring: str = "specificity",
        temperature: float = 0.02,
        margin_threshold: float = 0.012,
        prob_ratio_threshold: float = 0.40,
        min_confidence: float = 0.30,
        use_full_path: bool = True,
    ) -> list[dict[str, Any]]:
        """Compute detailed 1-to-1 correspondence with confidence metrics for each pair."""
        row_means = np.mean(self.matrix, axis=1, keepdims=True)
        specificity_matrix = self.matrix - row_means

        shift = self.matrix - np.max(self.matrix, axis=0, keepdims=True)
        exp_m = np.exp(shift / temperature)
        prob_matrix = exp_m / np.sum(exp_m, axis=0, keepdims=True)

        if scoring == "specificity":
            cost_matrix = -specificity_matrix
        elif scoring == "softmax":
            cost_matrix = -prob_matrix
        else:
            cost_matrix = -self.matrix

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        details = []
        for r, c in zip(row_ind, col_ind):
            path_or_name = self.image_paths[c] if use_full_path and self.image_paths else self.image_names[c]
            cls = self.class_names[r]
            spec = float(specificity_matrix[r, c])
            raw_sim = float(self.matrix[r, c])
            top_p = float(prob_matrix[r, c])

            sorted_spec_indices = np.argsort(-specificity_matrix[:, c])
            spec_runner_idx = sorted_spec_indices[1] if sorted_spec_indices[0] == r else sorted_spec_indices[0]
            spec_margin = spec - float(specificity_matrix[spec_runner_idx, c])

            sorted_raw_indices = np.argsort(-self.matrix[:, c])
            raw_runner_idx = sorted_raw_indices[1] if sorted_raw_indices[0] == r else sorted_raw_indices[0]
            raw_runner_cls = self.class_names[raw_runner_idx]
            raw_margin = raw_sim - float(self.matrix[raw_runner_idx, c])
            runner_p = float(prob_matrix[raw_runner_idx, c])
            prob_ratio = runner_p / max(top_p, 1e-12)

            is_certain = bool(
                spec_margin >= margin_threshold
                and raw_margin >= margin_threshold
                and prob_ratio <= prob_ratio_threshold
                and top_p >= min_confidence
            )

            details.append({
                "image_path": str(path_or_name),
                "image_name": self.image_names[c],
                "assigned_class": cls,
                "is_certain": is_certain,
                "raw_cosine": round(raw_sim, 4),
                "raw_margin": round(raw_margin, 4),
                "specificity": round(spec, 4),
                "margin_over_runner_up": round(spec_margin, 4),
                "softmax_confidence": round(top_p, 4),
                "runner_up_class": raw_runner_cls,
                "runner_up_prob": round(runner_p, 4),
            })

        details.sort(key=lambda x: x["image_name"])
        return details


def get_default_device_and_dtype() -> tuple[str, torch.dtype]:
    """Detect appropriate PyTorch device and precision for EmbeddingGemma2."""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    else:
        device = "cpu"
        dtype = torch.float32
    return device, dtype


def load_embedding_model(
    model_name_or_path: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> SentenceTransformer:
    """Load EmbeddingGemma2 model configured for text and vision embeddings."""
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is not installed. Please install via: pip install sentence-transformers"
        )
    auto_device, auto_dtype = get_default_device_and_dtype()
    chosen_device = device or auto_device
    chosen_dtype = dtype or auto_dtype

    logger.info(
        f"Loading '{model_name_or_path}' on device={chosen_device} with dtype={chosen_dtype}..."
    )
    model = SentenceTransformer(
        model_name_or_path,
        model_kwargs={"torch_dtype": chosen_dtype},
        config_kwargs={"audio_config": None},  # Disable audio encoder to save memory
        device=chosen_device,
    )
    return model


def format_template(template: str, class_name: str) -> str:
    """Format template string supporting both {class_name} and [class_name] placeholders."""
    if "{class_name}" in template:
        return template.format(class_name=class_name)
    elif "[class_name]" in template:
        return template.replace("[class_name]", class_name)
    else:
        return f"{template} {class_name}"


def compute_class_embeddings(
    class_names: Sequence[str],
    model: SentenceTransformer,
    templates: Optional[Sequence[str]] = None,
    use_templates: bool = True,
    batch_size: int = 32,
    normalize: bool = True,
) -> tuple[list[tuple[str, np.ndarray]], np.ndarray]:
    """Compute text embeddings for a list of classes."""
    if not use_templates:
        logger.info(f"Computing direct text embeddings for {len(class_names)} classes (no templates)...")
        raw_embeddings = model.encode(
            list(class_names),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        class_embeddings: list[tuple[str, np.ndarray]] = []
        for idx, cls in enumerate(class_names):
            emb = raw_embeddings[idx]
            if normalize:
                norm = np.linalg.norm(emb)
                if norm > 1e-12:
                    emb = emb / norm
            class_embeddings.append((cls, emb))
        return class_embeddings, raw_embeddings

    active_templates = list(templates) if templates is not None else DEFAULT_PROMPT_TEMPLATES
    num_templates = len(active_templates)
    logger.info(
        f"Computing text embeddings for {len(class_names)} classes using {num_templates} templates each..."
    )

    all_texts: list[str] = []
    for cls in class_names:
        for tmpl in active_templates:
            all_texts.append(format_template(tmpl, cls))

    raw_embeddings = model.encode(
        all_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    class_embeddings = []
    for idx, cls in enumerate(class_names):
        start = idx * num_templates
        end = start + num_templates
        cls_tmpl_embs = raw_embeddings[start:end]
        avg_emb = np.mean(cls_tmpl_embs, axis=0)
        if normalize:
            norm = np.linalg.norm(avg_emb)
            if norm > 1e-12:
                avg_emb = avg_emb / norm
        class_embeddings.append((cls, avg_emb))

    return class_embeddings, raw_embeddings


def load_image_rgb(image_input: Union[str, Path, Image.Image]) -> tuple[Image.Image, str, str]:
    """Load image into RGB PIL Image and determine full path and display name."""
    if isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
        name = getattr(image_input, "filename", None) or "image"
        return img, str(name), Path(name).name
    path = Path(image_input).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    img = Image.open(path).convert("RGB")
    return img, str(path), path.name


def compute_image_embeddings(
    images: Sequence[Union[str, Path, Image.Image]],
    model: SentenceTransformer,
    batch_size: int = 1,
    normalize: bool = True,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Compute visual embeddings for input images using EmbeddingGemma2.

    Returns:
        (image_embeddings, image_paths, image_names)
    """
    logger.info(f"Loading and computing visual embeddings for {len(images)} images...")
    pil_images: list[Image.Image] = []
    image_paths: list[str] = []
    image_names: list[str] = []
    for img_item in images:
        img, full_path, name = load_image_rgb(img_item)
        pil_images.append(img)
        image_paths.append(full_path)
        image_names.append(name)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    raw_img_embs = model.encode(
        pil_images,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    if normalize:
        norms = np.linalg.norm(raw_img_embs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        norm_img_embs = raw_img_embs / norms
    else:
        norm_img_embs = raw_img_embs

    return norm_img_embs, image_paths, image_names


def apply_pca_reduction(
    class_matrix: np.ndarray,
    image_matrix: np.ndarray,
    auxiliary_train_embeddings: Optional[np.ndarray] = None,
    pca_dim: int = 64,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """Project class and image embeddings to pca_dim dimensions using PCA and re-normalize."""
    train_pools = [class_matrix, image_matrix]
    if auxiliary_train_embeddings is not None and len(auxiliary_train_embeddings) > 0:
        train_pools.append(auxiliary_train_embeddings)
    pca_train_data = np.vstack(train_pools)

    n_samples = len(pca_train_data)
    actual_dim = min(pca_dim, n_samples - 1 if n_samples <= pca_dim else pca_dim)
    if actual_dim < pca_dim:
        logger.warning(
            f"PCA training pool has {n_samples} samples; adjusting PCA components from {pca_dim} to {actual_dim}."
        )

    logger.info(f"Fitting PCA with {actual_dim} components on {n_samples} embedding samples...")
    pca = PCA(n_components=actual_dim, random_state=42)
    pca.fit(pca_train_data)

    class_proj = pca.transform(class_matrix)
    image_proj = pca.transform(image_matrix)

    class_norms = np.maximum(np.linalg.norm(class_proj, axis=1, keepdims=True), 1e-12)
    image_norms = np.maximum(np.linalg.norm(image_proj, axis=1, keepdims=True), 1e-12)

    class_proj_norm = class_proj / class_norms
    image_proj_norm = image_proj / image_norms

    return class_proj_norm, image_proj_norm, pca


def compute_semantic_similarity(
    class_names: Sequence[str],
    images: Sequence[Union[str, Path, Image.Image]],
    model_name_or_path: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    model: Optional[SentenceTransformer] = None,
    templates: Optional[Sequence[str]] = None,
    use_templates: bool = False,
    pca_dim: Optional[int] = None,
    postprocess: str = "none",
    temperature: float = 0.02,
    precomputed_image_data: Optional[tuple[np.ndarray, list[str], list[str]]] = None,
    return_one_to_one: bool = False,
    margin_threshold: float = 0.012,
    prob_ratio_threshold: float = 0.40,
    min_confidence: float = 0.30,
    normalize: bool = True,
) -> Union[SimilarityResult, Set[Tuple[str, str, bool]]]:
    """Callable function that computes semantic similarity between class names and images.

    If return_one_to_one is True:
        Returns a set of tuples: {(image_path, assigned_class, is_certain), ...}
    Otherwise:
        Returns SimilarityResult containing full matrix and metadata.
    """
    if len(class_names) == 0:
        raise ValueError("class_names list cannot be empty.")
    if len(images) == 0 and precomputed_image_data is None:
        raise ValueError("images list cannot be empty.")

    active_model = model or load_embedding_model(
        model_name_or_path=model_name_or_path, device=device, dtype=dtype
    )

    # 1. Text embeddings for classes
    class_emb_tuples, raw_text_embs = compute_class_embeddings(
        class_names=class_names,
        model=active_model,
        templates=templates,
        use_templates=use_templates,
        normalize=normalize,
    )
    class_matrix = np.stack([emb for _, emb in class_emb_tuples], axis=0)

    # 2. Visual embeddings for images
    if precomputed_image_data is not None:
        image_matrix, image_paths, image_names = precomputed_image_data
    else:
        image_matrix, image_paths, image_names = compute_image_embeddings(
            images=images,
            model=active_model,
            normalize=normalize,
        )

    # If PCA is requested, project embeddings to pca_dim and re-normalize
    if pca_dim is not None and pca_dim > 0:
        aux_data = raw_text_embs
        if not use_templates and len(class_names) * 1 + len(image_names) < pca_dim:
            logger.info("Computing auxiliary template embeddings to enrich PCA subspace...")
            _, aux_data = compute_class_embeddings(
                class_names=class_names,
                model=active_model,
                templates=templates,
                use_templates=True,
                normalize=normalize,
            )

        class_matrix, image_matrix, _ = apply_pca_reduction(
            class_matrix=class_matrix,
            image_matrix=image_matrix,
            auxiliary_train_embeddings=aux_data,
            pca_dim=pca_dim,
        )
        class_emb_tuples = [
            (cls, class_matrix[idx]) for idx, cls in enumerate(class_names)
        ]

    # 3. Cosine similarity: (n_classes, dim) x (dim, n_images) -> (n_classes, n_images)
    sim_matrix = np.matmul(class_matrix, image_matrix.T)

    base_result = SimilarityResult(
        matrix=sim_matrix,
        class_embeddings=class_emb_tuples,
        image_embeddings=image_matrix,
        class_names=list(class_names),
        image_names=image_names,
        image_paths=image_paths,
        pca_dim=pca_dim,
        use_templates=use_templates,
        postprocess_mode="raw",
    )

    if return_one_to_one:
        return base_result.one_to_one_correspondence(
            scoring="specificity",
            temperature=temperature,
            margin_threshold=margin_threshold,
            prob_ratio_threshold=prob_ratio_threshold,
            min_confidence=min_confidence,
        )

    if postprocess == "softmax":
        return base_result.to_softmax(temperature=temperature)
    elif postprocess == "zscore":
        return base_result.to_zscore()
    elif postprocess == "centered":
        return base_result.to_centered()
    return base_result


def compute_one_to_one_correspondence(
    class_names: Sequence[str],
    images: Sequence[Union[str, Path, Image.Image]],
    model_name_or_path: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    model: Optional[SentenceTransformer] = None,
    use_templates: bool = False,
    pca_dim: Optional[int] = None,
    scoring: str = "specificity",
    temperature: float = 0.02,
    margin_threshold: float = 0.012,
    prob_ratio_threshold: float = 0.40,
    min_confidence: float = 0.30,
    use_full_path: bool = True,
) -> Set[Tuple[str, str, bool]]:
    """Callable function that directly returns optimal 1-to-1 image-to-class correspondence as a set of tuples:
    {(image_path, assigned_class, is_certain), ...}
    """
    res = compute_semantic_similarity(
        class_names=class_names,
        images=images,
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=dtype,
        model=model,
        use_templates=use_templates,
        pca_dim=pca_dim,
    )
    if isinstance(res, SimilarityResult):
        return res.one_to_one_correspondence(
            scoring=scoring,
            temperature=temperature,
            margin_threshold=margin_threshold,
            prob_ratio_threshold=prob_ratio_threshold,
            min_confidence=min_confidence,
            use_full_path=use_full_path,
        )
    return res


def save_similarity_visualizations(
    result: SimilarityResult,
    output_csv: Optional[Path] = None,
    output_png: Optional[Path] = None,
    output_json: Optional[Path] = None,
    title_suffix: str = "",
) -> None:
    """Export similarity matrix to CSV, PNG heatmap (with red box around column max), and JSON summary."""
    df = result.to_dataframe()

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv)
        logger.info(f"Saved similarity matrix CSV: {output_csv}")

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "num_classes": len(result.class_names),
            "num_images": len(result.image_names),
            "use_templates": result.use_templates,
            "pca_dim": result.pca_dim,
            "postprocess_mode": result.postprocess_mode,
            "temperature": result.temperature,
            "classes": result.class_names,
            "images": result.image_names,
            "top_classes_per_image": result.top_classes_per_image(top_k=5),
        }
        output_json.write_text(json.dumps(summary, indent=2))
        logger.info(f"Saved similarity summary JSON: {output_json}")

    if output_png:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            import seaborn as sns

            output_png.parent.mkdir(parents=True, exist_ok=True)
            fig_width = max(8, len(result.image_names) * 1.35)
            fig_height = max(10, len(result.class_names) * 0.45)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            mode = getattr(result, "postprocess_mode", "raw")
            if mode == "softmax":
                fmt = ".1%"
                cmap = "YlGnBu"
                cbar_label = f"Softmax Probability (tau={result.temperature or 0.02:.2f})"
            elif mode == "zscore":
                fmt = "+.1f"
                cmap = "coolwarm"
                cbar_label = "Z-Score (standard deviations σ above image mean)"
            elif mode == "centered":
                fmt = "+.2f"
                cmap = "coolwarm"
                cbar_label = "Class Specificity (deviation from class scene mean)"
            else:
                fmt = ".2f"
                cmap = "viridis"
                cbar_label = "Cosine Similarity"

            sns.heatmap(
                df,
                annot=True,
                fmt=fmt,
                cmap=cmap,
                cbar_kws={"label": cbar_label},
                linewidths=0.5,
                ax=ax,
            )

            # Highlight the maximum similarity entry in each column with a bold red rectangle
            for col_idx in range(df.shape[1]):
                max_row_idx = int(np.argmax(df.iloc[:, col_idx].values))
                rect = patches.Rectangle(
                    (col_idx, max_row_idx),
                    1, 1,
                    fill=False,
                    edgecolor="red",
                    linewidth=3.0,
                    clip_on=False,
                )
                ax.add_patch(rect)

            base_title = "EmbeddingGemma 2: Class-Image Semantic Similarity Matrix"
            if title_suffix:
                full_title = f"{base_title}\n({title_suffix})"
            else:
                tmpl_info = "10-template averaged" if result.use_templates else "direct class names"
                pca_info = f", PCA {result.pca_dim}D" if result.pca_dim else ", 768D"
                post_info = f" [{mode.upper()}]" if mode != "raw" else ""
                full_title = f"{base_title}\n({tmpl_info}{pca_info}{post_info})"

            ax.set_title(full_title, fontsize=13, pad=12)
            ax.set_xlabel("Images (red border highlights highest match)", fontsize=11, labelpad=8)
            ax.set_ylabel("Classes", fontsize=11, labelpad=8)
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.yticks(fontsize=9)
            plt.tight_layout()
            plt.savefig(output_png, dpi=200)
            plt.close()
            logger.info(f"Saved similarity matrix PNG heatmap: {output_png}")
        except Exception as e:
            logger.warning(f"Could not save heatmap PNG: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute semantic similarity between classes and images with EmbeddingGemma2"
    )
    backend_root = Path(__file__).resolve().parents[1]
    default_img_dir = backend_root / "current_scene/scan_0/segmentation/non_structural_category_frames/mesh"

    parser.add_argument(
        "--images-dir",
        type=Path,
        default=default_img_dir,
        help="Directory containing images to evaluate",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_TEST_CLASSES,
        help="List of class names to embed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save CSV/PNG/JSON outputs (defaults to images-dir)",
    )
    parser.add_argument(
        "--templates",
        action="store_true",
        help="Use 10-template prompt averaging instead of direct class names",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=None,
        help="Reduce embeddings to N principal dimensions via PCA (e.g. 64)",
    )
    parser.add_argument(
        "--postprocess",
        choices=["none", "softmax", "zscore", "centered", "all"],
        default="all",
        help="Postprocessing confidence enhancement (default: all)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.02,
        help="Temperature parameter for softmax postprocessing (default: 0.02)",
    )
    args = parser.parse_args()

    out_dir = args.output_dir or args.images_dir
    images_dir = args.images_dir

    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return 1

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = sorted([
        p for p in images_dir.iterdir()
        if p.suffix.lower() in valid_extensions and not p.name.startswith("similarity_")
    ])

    if not image_paths:
        logger.error(f"No valid image files found in {images_dir}")
        return 1

    logger.info(f"Found {len(image_paths)} images in {images_dir}: {[p.name for p in image_paths]}")
    logger.info(f"Classes to evaluate ({len(args.classes)}): {args.classes}")

    model = load_embedding_model()
    precomputed_img_data = compute_image_embeddings(image_paths, model=model)

    use_templates = args.templates
    base_result = compute_semantic_similarity(
        class_names=args.classes,
        images=image_paths,
        model=model,
        use_templates=use_templates,
        pca_dim=args.pca_dim,
        precomputed_image_data=precomputed_img_data,
    )

    if args.postprocess == "all":
        modes = [
            ("raw", base_result, "Raw Cosine Similarity (Direct Class Names, 768D)"),
            ("softmax", base_result.to_softmax(temperature=args.temperature), f"Calibrated Softmax Probability (tau={args.temperature})"),
            ("zscore", base_result.to_zscore(), "Column Z-Score (Standard Deviations σ)"),
            ("centered", base_result.to_centered(), "Class Specificity (Row-Centered)"),
        ]
        for mode_name, res, desc in modes:
            save_similarity_visualizations(
                result=res,
                output_csv=out_dir / f"similarity_matrix_{mode_name}.csv",
                output_png=out_dir / f"similarity_matrix_{mode_name}.png",
                output_json=out_dir / f"similarity_summary_{mode_name}.json",
                title_suffix=desc,
            )

        save_similarity_visualizations(
            result=base_result.to_softmax(temperature=args.temperature),
            output_csv=out_dir / "similarity_matrix.csv",
            output_png=out_dir / "similarity_matrix.png",
            output_json=out_dir / "similarity_summary.json",
            title_suffix=f"Direct Classes, Calibrated Softmax Probability (tau={args.temperature})",
        )
    else:
        if args.postprocess == "softmax":
            final_res = base_result.to_softmax(temperature=args.temperature)
        elif args.postprocess == "zscore":
            final_res = base_result.to_zscore()
        elif args.postprocess == "centered":
            final_res = base_result.to_centered()
        else:
            final_res = base_result

        save_similarity_visualizations(
            result=final_res,
            output_csv=out_dir / "similarity_matrix.csv",
            output_png=out_dir / "similarity_matrix.png",
            output_json=out_dir / "similarity_summary.json",
        )

    # Compute optimal one-to-one correspondence as set of tuples
    tuple_set = base_result.one_to_one_correspondence(
        scoring="specificity",
        temperature=args.temperature,
        use_full_path=True,
    )
    detailed_pairs = base_result.detailed_one_to_one(
        scoring="specificity",
        temperature=args.temperature,
        use_full_path=True,
    )

    print("\n" + "=" * 90)
    print("SET OF TUPLES: ONE-TO-ONE IMAGE-TO-CLASS CORRESPONDENCE (path, class, is_certain)")
    print("=" * 90)
    print("{")
    for item in sorted(tuple_set, key=lambda x: x[0]):
        print(f"    {item},")
    print("}")

    print("\nDetailed breakdown:")
    import pandas as pd
    details_df = pd.DataFrame(detailed_pairs)
    print(details_df[["image_name", "assigned_class", "is_certain", "raw_cosine", "specificity", "margin_over_runner_up", "softmax_confidence", "runner_up_class"]].to_string(index=False))
    print("=" * 90 + "\n")

    json_path = out_dir / "one_to_one_correspondence.json"
    csv_path = out_dir / "one_to_one_correspondence.csv"
    json_path.write_text(json.dumps(detailed_pairs, indent=2))
    details_df.to_csv(csv_path, index=False)
    logger.info(f"Saved 1-to-1 correspondence JSON: {json_path}")
    logger.info(f"Saved 1-to-1 correspondence CSV: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
