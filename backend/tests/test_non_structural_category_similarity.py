import sys
import json
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1] / "05_semantic_segmentation"))

from run_non_structural_category_similarity import (
    CATEGORIES,
    CATEGORY_GROUPS,
    CATEGORY_WORDS,
    DEFAULT_MODEL_NAME,
    WORD_CATEGORY_IDS,
    _fingerprint,
    _load_word_embeddings,
    _load_image_embeddings,
    _save_npz,
    aggregate_valid_class_associations,
    main,
    specificity_matrices,
    top_rows,
)


def test_categories_keep_words_grouped_into_28_classes():
    assert len(CATEGORIES) == len(CATEGORY_GROUPS) == 28
    assert CATEGORY_GROUPS[0] == ("door", "door-frame", "doorway")
    assert CATEGORY_GROUPS[12] == ("chair", "office chair", "bench")
    assert "deckchair" not in CATEGORY_WORDS
    assert len(CATEGORY_WORDS) == len(WORD_CATEGORY_IDS) == 66


def test_all_28_categories_can_be_assigned_to_a_source_class(tmp_path):
    frame = tmp_path / "class_71_inst_0.png"
    scores = np.zeros((len(CATEGORIES), 1), dtype=np.float32)
    scores[2, 0] = 0.10

    assert aggregate_valid_class_associations(
        {"mesh": [frame]}, {"mesh": scores}
    ) == [(71, 3)]


def test_top_rows_rank_by_shared_baseline_specificity_and_keep_raw_cosine():
    raw = {
        "mesh": np.array([[0.8, 0.8], [0.7, 0.2]]),
        "gs": np.array([[0.8], [0.2]]),
    }
    enhanced, baseline = specificity_matrices(raw)
    assert np.allclose(baseline[:, 0], [0.8, 1.1 / 3])
    rows = top_rows(
        ["generic", "specific"],
        [("generic",), ("specific",)],
        ["frame.png"],
        raw["mesh"][:, :1],
        enhanced["mesh"][:, :1],
        top_k=1,
    )
    assert rows[0]["category"] == "specific"
    assert rows[0]["rank"] == 1
    assert rows[0]["average_cosine_similarity"] == raw["mesh"][1, 0]
    assert rows[0]["enhanced_similarity"] == enhanced["mesh"][1, 0]


def test_embedding_caches_validate_model_and_image_fingerprint(tmp_path):
    vectors = np.ones((len(CATEGORY_WORDS), 3), dtype=np.float32)
    class_cache = tmp_path / "categories.npz"
    _save_npz(
        class_cache,
        model_name=np.asarray(DEFAULT_MODEL_NAME),
        classes=np.asarray(CATEGORIES),
        words=np.asarray(CATEGORY_WORDS),
        word_category_ids=np.asarray(WORD_CATEGORY_IDS, dtype=np.int16),
        word_embeddings=vectors,
    )
    assert np.array_equal(_load_word_embeddings(class_cache), vectors)

    image = tmp_path / "frame.png"
    image.write_bytes(b"image bytes")
    image_cache = tmp_path / "images.npz"
    paths = [str(image.resolve())]
    embeddings = np.ones((1, 3), dtype=np.float32)
    _save_npz(
        image_cache,
        model_name=np.asarray(DEFAULT_MODEL_NAME),
        fingerprint=np.asarray(_fingerprint([image])),
        paths=np.asarray(paths),
        embeddings=embeddings,
    )
    assert np.array_equal(_load_image_embeddings(image_cache, [image])[0], embeddings)
    image.write_bytes(b"changed image bytes")
    assert _load_image_embeddings(image_cache, [image]) is None


def test_no_rendered_instances_produces_empty_category_associations(tmp_path):
    mesh_frames = tmp_path / "mesh"
    gs_frames = tmp_path / "gs"
    mesh_frames.mkdir()
    output = tmp_path / "similarity"

    assert main([
        "--mesh-frames-dir", str(mesh_frames),
        "--gs-frames-dir", str(gs_frames),
        "--output-dir", str(output),
    ]) == []
    result = json.loads((output / "valid_class_associations.json").read_text())
    assert result["associations"] == []
    assert len(json.loads((output / "categories.json").read_text())) == 28


def test_mesh_only_frames_are_classified_without_loading_model(tmp_path, monkeypatch):
    import run_non_structural_category_similarity as category

    mesh_frames = tmp_path / "mesh"
    mesh_frames.mkdir()
    gs_frames = tmp_path / "missing_gs"
    image = mesh_frames / "class_17_inst_0.png"
    Image.new("RGB", (32, 32), (100, 150, 200)).save(image)
    output = tmp_path / "similarity"
    rng = np.random.default_rng(4)
    words = rng.normal(size=(len(CATEGORY_WORDS), 8)).astype(np.float32)
    words /= np.linalg.norm(words, axis=1, keepdims=True)
    _save_npz(output / "category_embeddings.npz", model_name=np.asarray(DEFAULT_MODEL_NAME),
              classes=np.asarray(CATEGORIES), words=np.asarray(CATEGORY_WORDS),
              word_category_ids=np.asarray(WORD_CATEGORY_IDS, dtype=np.int16), word_embeddings=words)
    image_embedding = np.ones((1, 8), dtype=np.float32) / np.sqrt(8)
    _save_npz(output / "mesh" / "image_embeddings.npz", model_name=np.asarray(DEFAULT_MODEL_NAME),
              fingerprint=np.asarray(_fingerprint([image])), paths=np.asarray([str(image.resolve())]),
              embeddings=image_embedding)
    monkeypatch.setattr(category, "load_embedding_model", lambda: (_ for _ in ()).throw(AssertionError("model should use caches")))

    category.main(["--mesh-frames-dir", str(mesh_frames), "--gs-frames-dir", str(gs_frames),
                   "--output-dir", str(output)])
    summary = json.loads((output / "mesh" / "summary.json").read_text())
    assert summary["num_images"] == 1
    assert not (output / "gs" / "summary.json").exists()
