import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw
import numpy as np
import trimesh

STAGE = Path(__file__).resolve().parents[1] / "06_mesh_elaboration"
sys.path.insert(0, str(STAGE))
sys.path.insert(0, str(STAGE.parent / "04_3DGS_to_mesh"))
from fill_mesh_holes import export_ply_with_face_attributes
import project_floor_slices_onto_planes as projection


def test_projection_includes_below_plane_geometry_expands_bounds_and_overwrites(tmp_path, monkeypatch):
    level_dir = tmp_path / "intermediate_floor_levels"
    level_dir.mkdir()
    slice_path = level_dir / "floor_level_000_scene_slice.ply"

    # Plane y = 0.2x + 1, with a 1 m square fitted preview and source faces outside it.
    vertices = np.array([
        [0., 1., 0.], [1., 1.2, 0.], [1., 1.2, 1.], [0., 1., 1.],
        [-2., 0., -1.], [2., 0., -1.], [0., .3, 2.],  # wholly below the floor plane
        [-1., 3., 2.], [2., 3., 2.], [0., 3., -2.],  # above the floor plane
        [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],  # vertical outlier, excluded from projection
    ], dtype=np.float32)
    triangles = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.int32)
    attributes = {
        "is_hole_fill": ("uchar", np.zeros(5, dtype=np.uint8)),
        "surface_role": ("uchar", np.array([0, 0, 255, 255, 255], dtype=np.uint8)),
        "is_fitted_plane": ("uchar", np.array([1, 1, 0, 0, 0], dtype=np.uint8)),
        "floor_area_id": ("int", np.array([0, 0, -1, -1, -1], dtype=np.int32)),
    }
    assert export_ply_with_face_attributes(slice_path, vertices, triangles, face_attributes=attributes)
    wall_shell = trimesh.Trimesh(
        vertices=np.array([[-1., 0., -2.], [-1., 0., 2.], [-1., 4., 2.], [-1., 4., -2.]]),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        process=False,
    )
    wall_shell.export(tmp_path / "wall_enclosing_shell.ply")
    (tmp_path / "wall_planes_manifest.json").write_text(json.dumps({
        "wall_planes": [{"id": 0, "plane_equation": [1., 0., 0., 1.]}],
    }))
    equation = [-.2, 1., 0., -1.]
    (level_dir / "floor_level_intermediate_manifest.json").write_text(json.dumps({
        "levels": [{"level": 0, "floor_areas": [{"area_id": 0, "plane_equation": equation}]}],
    }))
    (level_dir / "floor_level_scene_slices_manifest.json").write_text(json.dumps({
        "levels": [{"level": 0, "output_file": slice_path.name}],
    }))

    manifest_path = projection.project_floor_slices_onto_planes(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    area = manifest["levels"][0]["areas"][0]
    assert area["source_triangle_count"] == 2
    assert area["source_y_bounds_m"][0] < 1.0 < area["source_y_bounds_m"][1]
    assert area["expanded_to_fit_slice"] is True
    assert area["wall_contour_segment_count"] > 0
    assert area["wall_contour_height_above_floor_m"] == .30
    assert area["contour_refinement"]["sigma_px"] == 13.0
    assert area["contour_refinement"]["turns"] == 7
    assert area["image"] == "floor_level_000_area_000_collapsed.png"
    assert area["original_projection_image"] == "floor_level_000_area_000_collapsed_original.png"
    original_path = level_dir / area["original_projection_image"]
    with Image.open(original_path) as original:
        original_pixels = np.asarray(original.convert("RGB"))
        assert not np.any(np.all(original_pixels == [255, 0, 0], axis=2))
    with Image.open(level_dir / area["image"]) as image:
        assert image.format == "PNG"
        assert image.width > 100 and image.height > 100
        pixels = np.asarray(image.convert("RGB"))
        assert np.count_nonzero(np.all(pixels == [255, 0, 0], axis=2)) > 0

    previous = (level_dir / area["image"]).read_bytes()
    previous_original = original_path.read_bytes()
    projection.project_floor_slices_onto_planes(tmp_path)
    assert (level_dir / area["image"]).read_bytes() == previous
    assert original_path.read_bytes() == previous_original

    preserved_image = level_dir / "floor_level_001_area_000_collapsed.png"
    preserved_image.write_bytes(b"preserve untouched level")
    manifest["levels"].append({
        "level": 1,
        "source_slice": "floor_level_001_scene_slice.ply",
        "areas": [{"area_id": 0, "image": preserved_image.name}],
    })
    manifest_path.write_text(json.dumps(manifest))
    projection.project_floor_slices_onto_planes(tmp_path, level_id=0, apply_blur=False)
    partial_manifest = json.loads(manifest_path.read_text())
    updated_area = partial_manifest["levels"][0]["areas"][0]
    assert partial_manifest["partial_update"] == {
        "updated_level_ids": [0], "preserved_level_ids": [1],
    }
    assert partial_manifest["source_triangle_filter"]["vertical_tolerance_deg"] == 20.0
    assert updated_area["near_vertical_triangles_excluded"] == 1
    assert updated_area["contour_refinement"]["applied"] is False
    assert updated_area["contour_refinement"]["turns"] == 0
    assert preserved_image.read_bytes() == b"preserve untouched level"

    final_path = level_dir / updated_area["image"]
    original_path = level_dir / updated_area["original_projection_image"]
    final_before, original_before = final_path.read_bytes(), original_path.read_bytes()
    candidate_manifest_path = projection.project_floor_slices_onto_planes(
        tmp_path, level_id=0, morphology_candidate=True
    )
    candidate_manifest = json.loads(candidate_manifest_path.read_text())
    candidate_area = candidate_manifest["levels"][0]["areas"][0]
    assert candidate_area["image"].endswith("_collapsed_first_refinement.png")
    assert candidate_area["first_refinement"]["components_after_area_filter"] >= 1
    assert (level_dir / candidate_area["image"]).is_file()
    assert final_path.read_bytes() == final_before
    assert original_path.read_bytes() == original_before

    original_path.unlink()
    monkeypatch.setattr(
        projection, "_fit_wall_gap_rectangles",
        lambda *args, **kwargs: ([], {"rectangle_count": 0}),
    )
    holes_manifest_path = projection.project_floor_slices_onto_planes(
        tmp_path, level_id=0, detect_holes=True
    )
    holes_manifest = json.loads(holes_manifest_path.read_text())
    holes_area = holes_manifest["levels"][0]["areas"][0]
    holes_image = level_dir / holes_area["image"]
    assert holes_area["image"].endswith("_collapsed_first_refinement_with_holes.png")
    assert holes_area["hole_detection"]["rectangle_count"] == 0
    assert holes_image.is_file()
    assert original_path.is_file()
    holes_before = holes_image.read_bytes()

    projection.project_floor_slices_onto_planes(tmp_path, level_id=0, morphology_candidate=True)
    assert holes_image.read_bytes() == holes_before


def test_black_artifact_cleanup_removes_thin_strips_and_small_components():
    pixels = np.full((100, 260, 3), 255, dtype=np.uint8)
    pixels[20:60, 15:65] = 0
    pixels[37:39, 65:100] = 0
    pixels[9:14, 9:14] = [255, 0, 255]
    pixels[10:13, 10:13] = 0
    pixels[70:90, 20:220] = 0

    cleaned, removed = projection._clean_black_artifacts(Image.fromarray(pixels), (0, 0, 260, 100))
    result = np.asarray(cleaned)

    assert removed > 0
    assert np.all(result[35:45, 35:45] == 0)
    assert not np.any(np.max(result[37:39, 70:99], axis=2) <= projection.BLACK_MASK_MAX_CHANNEL)
    assert not np.any(np.max(result[10:13, 10:13], axis=2) <= projection.BLACK_MASK_MAX_CHANNEL)
    assert not np.any(np.max(result[70:90, 20:220], axis=2) <= projection.BLACK_MASK_MAX_CHANNEL)
    assert np.ptp(result[10:13, 10:13], axis=2).max() <= projection.NEUTRAL_BACKGROUND_MAX_SPREAD


def test_blur_sharpen_refines_black_mask_without_creating_background_halos():
    pixels = np.full((120, 120, 3), [242, 240, 237], dtype=np.uint8)
    pixels[28:92, 28:92] = 0
    pixels[28, 28:92] = [177, 61, 184]
    original_black = np.max(pixels, axis=2) <= projection.BLACK_MASK_MAX_CHANNEL

    refined = np.asarray(projection._blur_sharpen_plot(
        Image.fromarray(pixels), (10, 10, 110, 110)
    ))
    changed = np.any(refined != pixels, axis=2)
    changed_background = changed & ~original_black

    assert changed_background.any()
    assert np.all(np.max(refined[changed_background], axis=1) == 0)
    assert np.array_equal(refined[28, 28:92], pixels[28, 28:92])


def test_morphology_candidate_removes_thin_lines_and_small_floaters():
    pixels = np.full((500, 500, 3), [242, 240, 237], dtype=np.uint8)
    pixels[60:400, 60:400] = 0
    pixels[30:33, 80:150] = 0
    pixels[430:450, 430:450] = 0
    original_black = np.max(pixels, axis=2) <= projection.BLACK_MASK_MAX_CHANNEL

    candidate, stats = projection._regularize_black_mask(
        Image.fromarray(pixels), (0, 0, 500, 500), pixels_per_meter=340.0
    )
    result = np.asarray(candidate)
    changed = np.any(result != pixels, axis=2)

    assert np.all(np.max(result[70:390, 70:390], axis=2) == 0)
    assert not np.any(np.max(result[30:33, 80:150], axis=2) <= projection.BLACK_MASK_MAX_CHANNEL)
    assert not np.any(np.max(result[430:450, 430:450], axis=2) <= projection.BLACK_MASK_MAX_CHANNEL)
    assert np.all(np.max(result[changed & ~original_black], axis=1) == 0)
    assert stats["components_after_area_filter"] == 1
    assert stats["opening_radius_px"] == 6
    assert stats["signed_distance_sigma_px"] == 2.04


def test_hole_rectangles_stay_inside_wall_and_allow_at_most_five_percent_black():
    image = Image.new("RGB", (500, 500), (242, 240, 237))
    ImageDraw.Draw(image).rectangle((10, 10, 489, 489), outline=(255, 0, 0), width=3)
    pixels = np.asarray(image).copy()
    pixels[12:488, 110:488] = 0

    rectangles, metrics = projection._fit_wall_gap_rectangles(
        Image.fromarray(pixels), (0, 0, 500, 500), 100.0, 100.0
    )

    assert metrics["rectangle_count"] == 1
    rectangle = rectangles[0]
    assert rectangle["width_m"] >= projection.HOLE_MIN_SIDE_M
    assert rectangle["height_m"] >= projection.HOLE_MIN_LONG_SIDE_M
    assert rectangle["black_fraction"] <= projection.HOLE_MAX_BLACK_FRACTION
    # bbox_px uses inclusive last-pixel coordinates; black support starts at X=110.
    assert 109 <= rectangle["bbox_px"][2] < 116
    left, top, right, bottom = rectangle["bbox_px"]
    sampled = pixels[top:bottom + 1, left:right + 1]
    assert np.mean(np.max(sampled, axis=2) <= projection.BLACK_MASK_MAX_CHANNEL) <= projection.HOLE_MAX_BLACK_FRACTION


def test_hole_rectangle_snaps_to_wall_when_gap_is_under_ten_centimeters():
    inside = np.zeros((100, 100), dtype=bool)
    inside[10:90, 10:90] = True
    bounds, edges = projection._expand_rectangle_to_nearby_wall(
        (15, 15, 85, 85), inside.copy(), inside, 100.0, 100.0
    )

    assert bounds == (10, 10, 90, 90)
    assert set(edges) == {"top", "bottom", "left", "right"}


def test_largest_rectangle_search_returns_a_valid_rectangle():
    mask = np.array([
        [1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 1],
        [1, 1, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ], dtype=bool)

    rectangle = projection._largest_true_rectangle(mask, 1.0, 1.0)

    assert rectangle is not None
    top, left, bottom, right = rectangle
    assert mask[top:bottom, left:right].all()
    assert (bottom - top) * (right - left) == 15


def test_near_vertical_triangle_filter_keeps_horizontal_faces():
    def triangle_with_normal(normal):
        normal = np.asarray(normal, dtype=float)
        first = np.array([1., 0., 0.])
        second = np.cross(normal, first)
        return np.array([[0., 0., 0.], first, second])

    triangles_xyz = np.stack([
        triangle_with_normal([0., 1., 0.]),
        triangle_with_normal([0., 0., 1.]),
        triangle_with_normal([0., np.sin(np.deg2rad(19)), np.cos(np.deg2rad(19))]),
        triangle_with_normal([0., np.sin(np.deg2rad(21)), np.cos(np.deg2rad(21))]),
    ])
    vertices = triangles_xyz.reshape(-1, 3)
    triangles = np.arange(len(vertices)).reshape(-1, 3)

    retained, excluded_count = projection._retain_nonvertical_source_faces(
        vertices, triangles, np.arange(len(triangles))
    )

    assert retained.tolist() == [0, 3]
    assert excluded_count == 2
