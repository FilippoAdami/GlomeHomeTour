import sys
import json
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1] / "05_semantic_segmentation"))

from run_non_structural_category_similarity import (
    CATEGORIES,
    CATEGORY_GEOMETRY_CONSTRAINTS,
    CATEGORY_GROUPS,
    CATEGORY_WORDS,
    DEFAULT_MODEL_NAME,
    ASSOCIATION_MIN_POSITIVE_SCORE,
    WORD_CATEGORY_IDS,
    _fingerprint,
    _load_word_embeddings,
    _load_image_embeddings,
    _save_npz,
    _aggregate_class_scores,
    _positive_top_rows,
    _save_class_report,
    _save_instance_report,
    _instance_report_rows,
    _format_instance_bbox,
    _instance_dimensions_match_category,
    _filter_instance_scores_by_dimensions,
    _association_status,
    _class_level_decisions,
    _structural_instance_keys,
    aggregate_valid_class_associations,
    filter_semantically_inconsistent_frames,
    filter_instance_match_scores,
    main,
    specificity_matrices,
    top_rows,
)


def test_categories_keep_stable_ids_activate_rug_and_disable_tv_and_outdoor_classes():
    assert len(CATEGORIES) == len(CATEGORY_GROUPS) == 21
    assert CATEGORY_GROUPS[1] == ("door", "door-frame", "doorway")
    assert CATEGORY_GROUPS[13] == ("chair", "office chair", "bench")
    assert CATEGORY_GROUPS[14] == ("rug", "carpet", "mat")
    assert CATEGORY_GROUPS[18] == ("bed", "crib", "pillow", "blanket", "duvet", "comforter")
    assert CATEGORY_GROUPS[19] == (
        "cooktop", "hob", "burner", "stovetop", "gas stove", "electric stove", "hotplate"
    )
    assert [category_id for category_id in CATEGORY_GROUPS] == [
        1, 2, 3, 5, 7, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27
    ]
    assert not {"gate", "fountain", "gazebo", "car"}.intersection(CATEGORIES)
    assert not {"television", "monitor"}.intersection(CATEGORY_WORDS)
    assert "deckchair" not in CATEGORY_WORDS
    assert len(CATEGORY_WORDS) == len(WORD_CATEGORY_IDS) == 67


def test_structural_instances_come_from_frame_manifest_roles(tmp_path):
    (tmp_path / "non_structural_frames_manifest.json").write_text(json.dumps({
        "extracted_instances": {
            "10_0": {"class_id": 10, "instance_id": 0, "role": "wall"},
            "11_0": {"class_id": 11, "instance_id": 0, "role": "object"},
        }
    }))

    assert _structural_instance_keys(tmp_path) == {(10, 0)}


def test_all_fixed_categories_have_hardcoded_geometry_bounds():
    assert set(CATEGORY_GEOMETRY_CONSTRAINTS) == set(range(1, 29))
    assert all(
        set(bounds) == {"width_m", "depth_m", "height_m", "base_z_m"}
        and all(len(interval) == 2 and interval[0] <= interval[1] for interval in bounds.values())
        for bounds in CATEGORY_GEOMETRY_CONSTRAINTS.values()
    )
    assert CATEGORY_GEOMETRY_CONSTRAINTS[19]["base_z_m"] == (-0.20, 1.15)
    assert CATEGORY_GEOMETRY_CONSTRAINTS[27]["width_m"] == (0.45, 0.90)
    assert CATEGORY_GEOMETRY_CONSTRAINTS[27]["depth_m"] == (0.50, 0.95)
    assert CATEGORY_GEOMETRY_CONSTRAINTS[27]["height_m"] == (0.65, 1.20)
    assert CATEGORY_GEOMETRY_CONSTRAINTS[28]["depth_m"] == (1.20, 6.50)
    assert CATEGORY_GEOMETRY_CONSTRAINTS[12]["width_m"] == (0.25, 6.00)
    assert CATEGORY_GEOMETRY_CONSTRAINTS[14]["height_m"] == (0.002, 0.50)


def test_instance_score_floor_applies_only_when_at_least_five_positive_matches_exist():
    scores = np.array([0.003, 0.001, 0.001, -0.001, -0.002] + [0.0] * (len(CATEGORIES) - 5))
    retained = filter_instance_match_scores(scores)
    assert np.array_equal(retained, scores)
    assert _association_status(retained) == (1, "accepted", 3)

    scores[5:7] = [ASSOCIATION_MIN_POSITIVE_SCORE, 0.026]
    filtered = filter_instance_match_scores(scores)
    assert np.count_nonzero(scores > 0.0) == 5
    assert np.flatnonzero(filtered > 0.0).tolist() == [5]
    scores[7] = 0.04
    assert np.flatnonzero(filter_instance_match_scores(scores) > 0.0).tolist() == [5, 7]
    assert np.isneginf(filtered[0]) and np.isneginf(filtered[1]) and np.isneginf(filtered[2])
    assert np.isneginf(filtered[6])


def test_active_categories_can_be_assigned_to_a_source_class(tmp_path):
    frame = tmp_path / "class_71_inst_0.png"
    scores = np.zeros((len(CATEGORIES), 1), dtype=np.float32)
    scores[2, 0] = 0.10

    assert aggregate_valid_class_associations(
        {"mesh": [frame]}, {"mesh": scores}, {(71, 0): (1.0, 1.0, 0.4)}
    ) == [(71, 3)]


def test_dimension_invalid_category_is_removed_before_class_assignment():
    from run_non_structural_category_similarity import CATEGORY_ROW_BY_ID

    window_instance = np.zeros(len(CATEGORIES), dtype=np.float32)
    window_instance[CATEGORY_ROW_BY_ID[1]] = 0.06
    window_instance[CATEGORY_ROW_BY_ID[2]] = 0.023
    invalid_door = np.zeros(len(CATEGORIES), dtype=np.float32)
    invalid_door[CATEGORY_ROW_BY_ID[1]] = 0.05

    filtered = {
        (71, 0): _filter_instance_scores_by_dimensions(window_instance, (0.11, 1.297, 0.338)),
        (71, 1): _filter_instance_scores_by_dimensions(np.zeros(len(CATEGORIES)), None),
        (72, 0): _filter_instance_scores_by_dimensions(invalid_door, (1.531, 2.844, 3.174)),
    }
    assignments, _, valid_counts, instance_categories = _class_level_decisions(filtered)

    assert _association_status(filtered[(71, 0)]) == (2, "accepted", 1)
    assert assignments == {71: 2}
    assert valid_counts == {71: 1, 72: 0}
    assert instance_categories == {(71, 0): 2}


def test_class_decision_uses_valid_instances_and_leaves_empty_classes_unassigned():
    from run_non_structural_category_similarity import CATEGORY_ROW_BY_ID

    valid = np.zeros(len(CATEGORIES), dtype=np.float32)
    valid[CATEGORY_ROW_BY_ID[1]] = 0.06
    invalid = np.zeros(len(CATEGORIES), dtype=np.float32)
    scores = {(71, 0): valid, (71, 1): invalid, (72, 0): invalid}

    assignments, _, valid_counts, instance_categories = _class_level_decisions(scores)

    assert assignments == {71: 1}
    assert valid_counts == {71: 1, 72: 0}
    assert instance_categories == {(71, 0): 1}


def test_classes_with_no_valid_frames_are_written_as_unassigned(tmp_path):
    mesh_dir = tmp_path / "frames/mesh"
    gs_dir = tmp_path / "frames/gs"
    output_dir = tmp_path / "segmentation/non_structural_similarity"
    mesh_dir.mkdir(parents=True)
    gs_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    (output_dir.parent / "segmentation.json").write_text(json.dumps({
        "instances": [
            {"class_id": 71, "instance_index": 1, "bounds_min": [0, 0, 0], "bounds_max": [1, 1, 1]},
            {"class_id": 71, "instance_index": 2, "bounds_min": [1, 0, 0], "bounds_max": [2, 1, 1]},
        ]
    }))
    (output_dir / "valid_instance_associations.json").write_text(json.dumps({
        "associations": [[71, 0, 1]],
        "instances": [{
            "source_class_id": 71,
            "instance_id": 0,
            "status": "accepted",
            "text_class_id": 1,
            "classification_source": "phase07_door_split",
            "door_geometry": {"orientation": "Vertical"},
        }],
    }))

    assert main([
        "--mesh-frames-dir", str(mesh_dir),
        "--gs-frames-dir", str(gs_dir),
        "--output-dir", str(output_dir),
    ]) == []

    classes = json.loads((output_dir / "valid_class_associations.json").read_text())
    instances = json.loads((output_dir / "valid_instance_associations.json").read_text())
    assert classes["associations"] == []
    assert classes["class_decisions"] == [{
        "source_class_id": 71,
        "status": "unassigned_no_valid_instances",
        "valid_instance_count": 0,
        "instance_ids": [0, 1],
        "propagated_instance_ids": [],
    }]
    assert instances["associations"] == []
    assert [row["status"] for row in instances["instances"]] == ["no_valid_frames", "no_valid_frames"]
    assert all(row.get("classification_source") != "phase07_door_split" for row in instances["instances"])
    assert (output_dir / "class_reports/class_071.png").is_file()


def test_class_report_uses_equal_instance_means_and_positive_top_ten(tmp_path):
    first = np.arange(1, len(CATEGORIES) + 1, dtype=np.float32) / 100
    second = first / 2
    class_scores = _aggregate_class_scores({(71, 0): first, (71, 1): second})
    mean_scores = class_scores[71]

    assert np.allclose(mean_scores, (first + second) / 2)
    top_rows = _positive_top_rows(mean_scores)
    assert len(top_rows) == 10
    assert np.all(np.diff(mean_scores[top_rows]) <= 0)

    image = tmp_path / "class_reports/class_071.png"
    category_id = aggregate_valid_class_associations(
        {"mesh": [tmp_path / "class_71_inst_0.png", tmp_path / "class_71_inst_1.png"]},
        {"mesh": np.column_stack((first, second))},
        {(71, 0): (0.5, 0.8, 0.7), (71, 1): (0.5, 0.8, 0.7)},
    )[0][1]
    _save_class_report(image, 71, mean_scores, category_id, "accepted", instance_count=2)
    assert image.is_file() and image.stat().st_size > 0


def test_phase07_split_door_records_survive_similarity_reruns():
    from run_non_structural_category_similarity import _merge_phase07_split_door_records

    split_door = {
        "source_class_id": 5,
        "instance_id": 0,
        "status": "accepted",
        "text_class_id": 1,
        "classification_source": "phase07_door_split",
        "door_geometry": {"orientation": "Vertical"},
    }
    stale_split = {
        **split_door,
        "source_class_id": 2,
        "instance_id": 4,
    }
    associations, instances = _merge_phase07_split_door_records(
        {"instances": [split_door, stale_split]},
        [[5, 0, 14], [8, 2, 3]],
        [
            {"source_class_id": 5, "instance_id": 0, "text_class_id": 14},
            {"source_class_id": 8, "instance_id": 2, "text_class_id": 3},
        ],
        {5},
    )

    assert associations == [[5, 0, 1], [8, 2, 3]]
    assert instances == [split_door, {"source_class_id": 8, "instance_id": 2, "text_class_id": 3}]


def test_instance_report_shows_category_bounds_final_score_and_bbox(tmp_path):
    from run_non_structural_category_similarity import CATEGORY_ROW_BY_ID

    scores = np.full(len(CATEGORIES), -np.inf, dtype=np.float32)
    scores[CATEGORY_ROW_BY_ID[1]] = 0.036
    rows = _instance_report_rows(scores)
    assert rows == [[
        "door",
        "W 0.55–2.5 · D 0.03–1 · H 1.75–3.2 · Base Z -0.2–0.15",
        "+0.036",
    ]]
    assert _format_instance_bbox((0.55, 2.5, 3.2)) == "BBox X × Y × Z: 0.55 × 2.50 × 3.20 m"

    image = tmp_path / "instance_reports/class_071_inst_000.png"
    _save_instance_report(image, 71, 0, {}, scores, (0.55, 2.5, 3.2))
    assert image.is_file() and image.stat().st_size > 0


def test_instance_dimension_compatibility_allows_horizontal_swap_and_ignores_base_height():
    assert _instance_dimensions_match_category((0.9, 2.1, 0.1), 1)
    assert _instance_dimensions_match_category((0.1, 2.1, 0.9), 1)
    assert _instance_dimensions_match_category((0.8, 0.1, 0.7), 19)
    assert not _instance_dimensions_match_category((0.9, 3.21, 0.1), 1)
    assert not _instance_dimensions_match_category((1.1, 2.1, 1.1), 1)


def test_semantic_frame_filter_keeps_only_strong_support_for_instance_category(tmp_path):
    from run_non_structural_category_similarity import CATEGORY_ROW_BY_ID

    bookshelf_row = CATEGORY_ROW_BY_ID[15]
    stairs_row = CATEGORY_ROW_BY_ID[3]
    instance_mean = np.full(len(CATEGORIES), -0.02, dtype=np.float32)
    instance_mean[bookshelf_row] = 0.05
    for row in [row for row in range(len(CATEGORIES)) if row != bookshelf_row][:4]:
        instance_mean[row] = 0.04

    def frame(top_row, category_score):
        scores = np.full(len(CATEGORIES), -0.03, dtype=np.float32)
        scores[bookshelf_row] = category_score
        scores[top_row] = max(float(scores[top_row]), category_score + 0.01)
        return {"path": tmp_path / f"{top_row}_{category_score}.png", "scores": scores}

    frames = {
        "down": {"mesh": frame(bookshelf_row, 0.07)},
        "up": {"mesh": frame(bookshelf_row, 0.02)},
        "left": {"mesh": frame(stairs_row, 0.08)},
    }

    kept, rejected = filter_semantically_inconsistent_frames(frames, instance_mean)

    assert list(kept) == ["down"]
    assert {item["reason"] for item in rejected} == {
        "below_instance_category_mean", "top_category_mismatch"
    }


def test_semantic_frame_filter_preserves_frames_without_accepted_instance_category(tmp_path):
    frame = {"path": tmp_path / "frame.png", "scores": np.zeros(len(CATEGORIES))}
    frames = {"base": {"mesh": frame}}

    kept, rejected = filter_semantically_inconsistent_frames(frames, frame["scores"])

    assert kept is frames
    assert rejected == []


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
        frame_status={"frame.png": "discarded: top_category_mismatch"},
    )
    assert rows[0]["category"] == "specific"
    assert rows[0]["rank"] == 1
    assert rows[0]["average_cosine_similarity"] == raw["mesh"][1, 0]
    assert rows[0]["enhanced_similarity"] == enhanced["mesh"][1, 0]
    assert rows[0]["frame_status"] == "discarded: top_category_mismatch"


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
    categories = json.loads((output / "categories.json").read_text())
    assert len(categories) == 28
    assert [item["id"] for item in categories if item["active"]] == list(CATEGORY_GROUPS)
    assert all("geometry_constraints" in item for item in categories)
    assert categories[3]["name"] == "ramp" and not categories[3]["active"]
    assert categories[13]["name"] == "rug" and categories[13]["active"]
    assert categories[13]["words"] == list(CATEGORY_GROUPS[14])
    assert categories[16]["name"] == "television" and not categories[16]["active"]
    assert categories[16]["words"] == []
    assert categories[13]["geometry_constraints"] == {
        key: list(bounds) for key, bounds in CATEGORY_GEOMETRY_CONSTRAINTS[14].items()
    }
    assert json.loads((output / "semantic_frame_filter.json").read_text())["discarded_frames"] == []


def test_structural_mesh_frames_only_rank_door_and_window_without_loading_model(tmp_path, monkeypatch):
    import run_non_structural_category_similarity as category

    mesh_frames = tmp_path / "mesh"
    mesh_frames.mkdir()
    (mesh_frames / "non_structural_frames_manifest.json").write_text(json.dumps({
        "extracted_instances": {
            "17_0": {"class_id": 17, "instance_id": 0, "role": "wall"},
        }
    }))
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
    top_classes = summary["top_classes_per_image"][image.name]
    assert {item["class_id"] for item in top_classes} == {1, 2}
    assert not (output / "gs" / "summary.json").exists()
