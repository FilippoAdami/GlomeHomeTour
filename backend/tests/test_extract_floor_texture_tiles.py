import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "06_mesh_elaboration"))
from extract_floor_texture_tiles import (
    _clean_square_mask,
    _deduplicate_texture_candidates,
    _is_significant_texture_region,
    _largest_square,
    _minimum_region_boundary_segment_m,
    _regularize_region_labels,
    _regularize_raster_lighting,
    _region_components,
    _texture_signature,
    _texture_similarity,
)


def test_texture_region_requires_area_and_all_boundary_segments_at_least_half_meter():
    blob = {"coarse_cell_count": 58, "minimum_boundary_segment_m": 0.8}
    short_segment = {"coarse_cell_count": 900, "minimum_boundary_segment_m": 0.49}
    broad_region = {"coarse_cell_count": 900, "minimum_boundary_segment_m": 0.5}

    assert not _is_significant_texture_region(blob)
    assert not _is_significant_texture_region(short_segment)
    assert _is_significant_texture_region(broad_region)


def test_texture_region_boundary_measure_uses_simplified_coarse_contour():
    broad = np.zeros((30, 30), dtype=bool)
    broad[4:24, 4:24] = True
    narrow = np.zeros((40, 30), dtype=bool)
    narrow[4:14, 4:24] = True

    assert np.isclose(_minimum_region_boundary_segment_m(broad), 0.76)
    assert _minimum_region_boundary_segment_m(narrow) < 0.5


def test_texture_region_boundary_measure_ignores_small_label_stair_steps():
    region = np.zeros((80, 80), dtype=bool)
    region[10:70, 10:70] = True
    region[10:70:4, 8:10] = True
    region[10:70:4, 70:72] = True

    assert _minimum_region_boundary_segment_m(region) >= 0.5


def test_small_and_same_color_regions_merge_into_texture_but_clear_large_zone_stays():
    labels = np.ones((60, 80), dtype=np.int32)
    labels[:, 20:40] = 4
    labels[:, 40:] = 2
    labels[15:20, 10:15] = 3
    regions = [
        {"kind": "texture", "texture_id": 1, "coarse_cell_count": 1195,
         "minimum_boundary_segment_m": 1.0, "median_chroma": [1.22, 0.92]},
        {"kind": "plain", "region_id": 4, "coarse_cell_count": 1200,
         "minimum_boundary_segment_m": 1.0, "median_chroma": [1.30, 0.88]},
        {"kind": "plain", "region_id": 2, "coarse_cell_count": 2400,
         "minimum_boundary_segment_m": 1.0, "median_chroma": [1.80, 0.40]},
        {"kind": "plain", "region_id": 3, "coarse_cell_count": 25,
         "minimum_boundary_segment_m": 0.0, "median_chroma": [1.80, 0.40]},
    ]

    result = _regularize_region_labels(labels, regions, {1})

    assert np.all(result[:, :40] == 1)
    assert np.all(result[:, 40:] == 2)


def test_region_segmentation_separates_two_grains_from_a_plain_area():
    size = 384
    rng = np.random.default_rng(42)
    horizontal_grain = gaussian_filter(rng.normal(size=(size, size)), sigma=(5, 20))
    vertical_grain = gaussian_filter(rng.normal(size=(size, size)), sigma=(20, 5))
    horizontal_grain = (horizontal_grain - horizontal_grain.mean()) / horizontal_grain.std()
    vertical_grain = (vertical_grain - vertical_grain.mean()) / vertical_grain.std()
    luminance = np.full((size, size), 0.45, dtype=np.float32)
    luminance[:, :128] += 0.09 * horizontal_grain[:, :128]
    luminance[:, 128:256] += 0.09 * vertical_grain[:, 128:256]
    luminance[:, 256:] = 0.42 + np.linspace(0, 0.03, 128, dtype=np.float32)[None, :]
    rgb = np.repeat(luminance[..., None], 3, axis=2) * np.array([1.0, 0.78, 0.58])
    rgb[size // 2:, 256:] = (
        luminance[size // 2:, 256:, None] * np.array([0.62, 0.82, 1.0])
    )

    _, regions, _ = _region_components(rgb, np.ones((size, size), dtype=bool))

    textures = [region for region in regions if region["kind"] == "texture"
                and region["coarse_cell_count"] >= 100]
    plain = [region for region in regions if region["kind"] == "plain"
             and region["coarse_cell_count"] >= 100]
    assert len(textures) == 2
    assert len(plain) == 2
    assert {np.sign(region["median_grain_orientation"]) for region in textures} == {-1.0, 1.0}


def test_largest_square_requires_a_fully_supported_region():
    mask = np.zeros((7, 8), dtype=bool)
    mask[:5, :6] = True
    mask[2, 2] = False

    side, top_left = _largest_square(mask)

    assert side == 3
    assert top_left == (3, 0)


def test_clean_square_mask_rejects_deep_shadow_and_color_outlier():
    rgb = np.full((128, 128, 3), [0.45, 0.32, 0.20], dtype=np.float32)
    rgb[:32, :32] *= 0.25
    rgb[64:72, 64:66] = 1.0

    mask = _clean_square_mask(
        rgb, np.ones((128, 128), dtype=bool), np.ones((128, 128), dtype=np.int32), 1, {1}
    )

    assert not mask[0, 0]
    assert not mask[8, 8]


def test_lighting_correction_runs_on_plain_and_textured_areas_and_preserves_grain():
    size = 256
    x = np.linspace(0.55, 1.0, size, dtype=np.float32)[None, :]
    grain = 0.035 * np.sin(np.arange(size, dtype=np.float32)[:, None] * (2 * np.pi / 7))
    luminance = x * (0.38 + grain)
    luminance[:, 128:] = x[:, 128:] * 0.38
    rgb = luminance[..., None] * np.array([1.0, 0.82, 0.66], dtype=np.float32)

    corrected, info = _regularize_raster_lighting(rgb, np.ones((size, size), dtype=bool))

    before_luma = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    after_luma = corrected @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    assert info["processed"]
    assert info["corrected"]
    assert np.percentile(after_luma, 90) - np.percentile(after_luma, 10) < (
        np.percentile(before_luma, 90) - np.percentile(before_luma, 10)
    ) * 0.6
    before_grain = before_luma[:, 20:110] - gaussian_filter(before_luma[:, 20:110], sigma=2)
    after_grain = after_luma[:, 20:110] - gaussian_filter(after_luma[:, 20:110], sigma=2)
    assert np.std(after_grain) > 0.85 * np.std(before_grain)


def test_plain_even_lighting_runs_and_remains_unchanged():
    rgb = np.full((80, 100, 3), [0.46, 0.38, 0.29], dtype=np.float32)

    corrected, info = _regularize_raster_lighting(rgb, np.ones((80, 100), dtype=bool))

    assert info["processed"]
    assert not info["corrected"]
    np.testing.assert_allclose(corrected, rgb, atol=1e-6)


def test_texture_matching_keeps_best_lighting_variant_and_distinct_pattern():
    rng = np.random.default_rng(19)
    grain = gaussian_filter(rng.normal(size=(128, 128)), sigma=(3, 16))
    grain = (grain - grain.mean()) / grain.std()
    luminance = np.clip(0.44 + 0.07 * grain, 0.05, 0.9)
    original = luminance[..., None] * np.array([1.0, 0.78, 0.58], dtype=np.float32)
    relit = np.clip(original * 0.68 + np.array([0.09, 0.06, 0.03]), 0.0, 1.0)
    checker = (np.indices((128, 128)).sum(axis=0) // 8 % 2).astype(np.float32)
    checker = (0.34 + checker * 0.18)[..., None] * np.array([1.0, 0.78, 0.58])

    same_score = _texture_similarity(_texture_signature(original), _texture_signature(relit))
    different_score = _texture_similarity(_texture_signature(original), _texture_signature(checker))
    candidates = [
        {"name": "best.png", "tile": relit, "quality": 2.0, "coverage": 0.99, "side_m": 0.8},
        {"name": "duplicate.png", "tile": original, "quality": 1.0, "coverage": 0.96, "side_m": 0.6},
        {"name": "other.png", "tile": checker, "quality": 1.5, "coverage": 0.99, "side_m": 0.7},
    ]

    kept, duplicates = _deduplicate_texture_candidates(candidates)

    assert same_score >= 0.82
    assert different_score < 0.80
    assert {item["name"] for item in kept} == {"best.png", "other.png"}
    assert len(duplicates) == 1
    assert duplicates[0]["removed"]["name"] == "duplicate.png"
    assert duplicates[0]["kept"]["name"] == "best.png"


