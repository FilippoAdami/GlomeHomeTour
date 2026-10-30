import pytest
import numpy as np
from PIL import Image
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "06_mesh_elaboration"))

from step_semantic_similarity import (
    DEFAULT_PROMPT_TEMPLATES,
    format_template,
    SimilarityResult,
    compute_semantic_similarity,
    save_similarity_visualizations,
)


def test_prompt_templates_count():
    assert len(DEFAULT_PROMPT_TEMPLATES) == 10
    for tmpl in DEFAULT_PROMPT_TEMPLATES:
        assert "{class_name}" in tmpl or "[class_name]" in tmpl
        formatted = format_template(tmpl, "chair")
        assert "chair" in formatted
        assert "{" not in formatted and "}" not in formatted


def test_similarity_result_structure():
    mock_matrix = np.array([[0.8, 0.2], [0.1, 0.9], [0.5, 0.4]])
    mock_class_embs = [("chair", np.zeros(768)), ("table", np.zeros(768)), ("lamp", np.zeros(768))]
    mock_img_embs = np.zeros((2, 768))
    res = SimilarityResult(
        matrix=mock_matrix,
        class_embeddings=mock_class_embs,
        image_embeddings=mock_img_embs,
        class_names=["chair", "table", "lamp"],
        image_names=["img1.jpg", "img2.jpg"],
    )
    assert res.shape == (3, 2)
    assert res[0, 0] == 0.8
    assert np.asarray(res).shape == (3, 2)

    df = res.to_dataframe()
    assert list(df.index) == ["chair", "table", "lamp"]
    assert list(df.columns) == ["img1.jpg", "img2.jpg"]

    # Test Softmax
    sm = res.to_softmax(temperature=0.02)
    assert np.allclose(sm.matrix.sum(axis=0), 1.0)
    assert sm.matrix[0, 0] > 0.99  # 0.8 vs 0.1, 0.5 at tau=0.02 is virtually 100%

    # Test Z-score
    zs = res.to_zscore()
    assert np.allclose(zs.matrix.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(zs.matrix.std(axis=0), 1.0, atol=1e-5)

    # Test Centered
    cnt = res.to_centered()
    assert np.allclose(cnt.matrix.mean(axis=1), 0.0, atol=1e-5)


def test_compute_semantic_similarity_execution(tmp_path):
    img1 = Image.new("RGB", (64, 64), color=(255, 0, 0))
    img2 = Image.new("RGB", (64, 64), color=(0, 255, 0))
    classes = ["red square", "green square", "blue object"]

    res_direct = compute_semantic_similarity(
        class_names=classes,
        images=[img1, img2],
        use_templates=False,
        postprocess="softmax",
        temperature=0.02,
    )
    assert res_direct.matrix.shape == (3, 2)
    assert np.allclose(res_direct.matrix.sum(axis=0), 1.0)

    # Save heatmap visualization with red boxes
    out_png = tmp_path / "test_softmax_heatmap.png"
    out_csv = tmp_path / "test_softmax_matrix.csv"
    save_similarity_visualizations(res_direct, output_csv=out_csv, output_png=out_png)
    assert out_png.exists() and out_png.stat().st_size > 0


def test_one_to_one_correspondence_return_type():
    from step_semantic_similarity import compute_one_to_one_correspondence

    # Mock matrix: 3 classes, 2 images
    # row 0 (desk): img0 is high (0.8), img1 is low (0.1)
    # row 1 (chair): img0 is low (0.1), img1 is high (0.7)
    # row 2 (wall): generic background across both (0.6, 0.6)
    mock_matrix = np.array([
        [0.80, 0.10],
        [0.10, 0.70],
        [0.60, 0.60],
    ])
    res = SimilarityResult(
        matrix=mock_matrix,
        class_embeddings=[("desk", np.zeros(10)), ("chair", np.zeros(10)), ("wall", np.zeros(10))],
        image_embeddings=np.zeros((2, 10)),
        class_names=["desk", "chair", "wall"],
        image_names=["img_desk.jpg", "img_chair.jpg"],
        image_paths=["/path/to/img_desk.jpg", "/path/to/img_chair.jpg"],
    )

    one_to_one = res.one_to_one_correspondence(use_full_path=True)
    assert isinstance(one_to_one, set)
    assert len(one_to_one) == 2
    for item in one_to_one:
        assert isinstance(item, tuple)
        assert len(item) == 3
        path, cls, certain = item
        assert isinstance(path, str)
        assert isinstance(cls, str)
        assert isinstance(certain, bool)

    # Convert to dict for value assertions
    mapping = {path: (cls, certain) for path, cls, certain in one_to_one}
    assert mapping["/path/to/img_desk.jpg"][0] == "desk"
    assert mapping["/path/to/img_desk.jpg"][1] is True
    assert mapping["/path/to/img_chair.jpg"][0] == "chair"
    assert mapping["/path/to/img_chair.jpg"][1] is True

    # Test uncertain condition: competing close classes
    ambiguous_matrix = np.array([
        [0.55, 0.10],
        [0.54, 0.70],
        [0.10, 0.10],
    ])
    res_ambiguous = SimilarityResult(
        matrix=ambiguous_matrix,
        class_embeddings=[("desk", np.zeros(10)), ("table", np.zeros(10)), ("wall", np.zeros(10))],
        image_embeddings=np.zeros((2, 10)),
        class_names=["desk", "table", "wall"],
        image_names=["img1.jpg", "img2.jpg"],
        image_paths=["/path/img1.jpg", "/path/img2.jpg"],
    )
    ambiguous_result = res_ambiguous.one_to_one_correspondence(use_full_path=True)
    amb_map = {path: (cls, certain) for path, cls, certain in ambiguous_result}
    # For img1, desk (0.55) and table (0.54) are virtually identical -> should be uncertain (False)
    assert amb_map["/path/img1.jpg"][1] is False

