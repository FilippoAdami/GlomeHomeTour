"""Regression checks for compact galleries, acquisition visibility and aligned poses."""

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageFilter
from plyfile import PlyData, PlyElement
import pytest
from scipy.spatial.transform import Rotation
import torch

STAGE = Path(__file__).resolve().parents[1] / "08_artifacts_generation"
sys.path.insert(0, str(STAGE))

from best_views_visibility import SurfaceVisibility
from extract_best_views import (
    LOW_PRIORITY_WEIGHT, _publish, camera_pose_quality, extract_best_views, extract_scan, image_quality,
    layout_weights, load_cameras, sample_surface, select_views, utility_matrix, framing_quality,
    check_reprojections, measure_views, object_visible_candidates,
)


def write_model(path, views, names=None):
    path.mkdir(parents=True, exist_ok=True)
    (path / "cameras.txt").write_text("1 PINHOLE 64 48 40 44 29 21\n")
    lines = []
    for index, view in enumerate(views, 1):
        q = np.roll(Rotation.from_matrix(view[:3, :3]).as_quat(), 1)
        name = names[index - 1] if names else f"frame_{index}.jpg"
        numbers = " ".join(str(float(x)) for x in [*q, *view[:3, 3]])
        lines.append(f"{index} {numbers} 1 {name}\n\n")
    (path / "images.txt").write_text("".join(lines))
    (path / "points3D.txt").write_text("# empty test tracks\n")


def write_mesh(path, vertices, faces):
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex = np.empty(len(vertices), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    for axis, name in enumerate(("x", "y", "z")):
        vertex[name] = np.asarray(vertices)[:, axis]
    triangles = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    triangles["vertex_indices"] = faces
    PlyData([PlyElement.describe(vertex, "vertex"), PlyElement.describe(triangles, "face")]).write(path)


def planes():
    vertices = np.array([[-2, -2, 4], [2, -2, 4], [2, 2, 4], [-2, 2, 4],
                         [-0.5, -0.5, 2], [0.5, -0.5, 2], [0.5, 0.5, 2], [-0.5, 0.5, 2]], dtype=np.float32)
    return vertices, np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.int32)


def test_no_photo_cap_for_many_connected_zones():
    matrix = torch.eye(24) / 24
    selected, report = select_views(matrix, matrix.sum(1))
    assert len(selected) == 20
    assert report["weighted_utility"] >= 0.8
    assert report["marginal_gain_threshold"] == pytest.approx(0.1 / 24)


def test_duplicate_views_add_no_utility_and_ties_use_image_identity():
    matrix = torch.tensor([[0.5, 0.2, 0], [0.5, 0.2, 0], [0, 0.2, 0.3]])
    selected, report = select_views(matrix, matrix.sum(1), [10, 2, 5])
    assert selected == [1, 2]
    assert report["weighted_utility"] == pytest.approx(1)
    assert set(report["unique_contributions"]) == set(selected)


def test_small_marginal_contribution_stops_below_target():
    matrix = torch.tensor([[0.60, 0, 0], [0.60, 0.02, 0], [0.60, 0, 0.02]])
    selected, report = select_views(matrix, torch.tensor([1, 0.5, 0.5]))
    assert selected == [0]
    assert report["stopping_reason"] == "low_marginal_gain"


def test_semantic_closeup_cannot_replace_new_room_layout_context():
    matrix = torch.tensor([[0.4, 0, 0], [0.4, 0, 0.5], [0, 0.4, 0]])
    layout = torch.tensor([[0.5, 0], [0.5, 0], [0, 0.5]])
    selected, report = select_views(matrix, layout, target=0.9, relative_gain=0.05)
    assert selected == [0, 2]
    assert report["stopping_reason"] == "insufficient_new_layout_context"


def test_selected_images_require_visible_non_door_window_object_when_available():
    cameras = [{"quality": {}}, {"quality": {}}, {"quality": {}}]
    targets = [{"category": "door"}, {"category": "window"}, {"category": "bed"}]
    features = torch.tensor([[0.9, 0.9, 0.51], [0.9, 0.9, 0.49], [0.1, 0.1, 0.8]])
    candidates, categories = object_visible_candidates(cameras, features, targets)
    assert candidates == [0, 2]
    assert categories == ["bed"]
    assert cameras[0]["quality"]["visible_recognized_categories"] == ["bed"]
    assert cameras[1]["quality"]["visible_recognized_categories"] == []

    fallback, categories = object_visible_candidates(cameras, features[:, :2], targets[:2])
    assert fallback == [0, 1, 2]
    assert categories == []
    with pytest.raises(ValueError, match="No acquired photograph adequately shows"):
        object_visible_candidates(cameras, torch.zeros((3, 3)), targets)


def test_selected_supplements_avoid_excessive_shared_visible_surface():
    matrix = torch.tensor([[0.5, 0., 0.], [0.5, 0.2, 0.], [0., 0., 0.2]])
    layout = torch.tensor([[0.8, 0.], [0.7, 0.09], [0., 0.3]])
    visible = torch.zeros((3, 20), dtype=torch.bool)
    visible[0, :10] = True
    visible[1, :6] = True
    visible[1, 10:14] = True
    visible[2, 10:] = True
    selected, report = select_views(matrix, layout, visible_support=visible)
    assert selected == [0, 2]
    assert report["overlap_blocked_candidates"] == 1
    assert report["maximum_selected_visible_surface_overlap"] == 0


@pytest.mark.parametrize("matrix", [torch.zeros(1, 1), torch.tensor([[float("nan")]]), torch.tensor([[-1.]])])
def test_invalid_or_empty_utility_is_rejected(matrix):
    with pytest.raises(ValueError):
        select_views(matrix, torch.ones(1))


def test_lower_priority_categories_and_repeated_instances_share_budget():
    visibility = torch.ones(3, 1)
    features = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    targets = [{"category": "bed", "category_priority": 1.},
               {"category": "rug", "category_priority": LOW_PRIORITY_WEIGHT},
               {"category": "rug", "category_priority": LOW_PRIORITY_WEIGHT}]
    matrix, _, mix = utility_matrix(visibility, features, np.ones(1), targets, np.zeros(1, dtype=int))
    assert mix["feature_share"] == 0.25
    semantic = matrix[:, 1:].sum(1)
    assert semantic[0] == pytest.approx(8 * float(semantic[1]))
    assert semantic[1] == semantic[2]


def test_area_sampling_and_weights_do_not_reward_tessellation():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0]], dtype=np.float32)
    coarse = sample_surface(vertices, np.array([[0, 1, 2], [0, 2, 3]]), 10_000)
    fine = sample_surface(vertices, np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]), 10_000)
    for samples in (coarse, fine):
        assert samples["surface_area"] == pytest.approx(1)
        assert samples["positions"][:, 0].mean() == pytest.approx(0.5, abs=0.01)
        weights, _ = layout_weights(samples, np.full(10_000, "wall"))
        assert weights.sum() == pytest.approx(1)
    again = sample_surface(vertices, np.array([[0, 1, 2], [0, 2, 3]]), 10_000)
    np.testing.assert_array_equal(again["positions"], coarse["positions"])


def test_one_good_region_view_saturates_without_photographing_every_surface():
    visibility = torch.tensor([[1., 1., 0., 0.], [0., 0., 1., 1.], [1., 0., 1., 0.]])
    matrix, layout, _ = utility_matrix(visibility, torch.empty(3, 0), np.ones(4), [], np.zeros(4, dtype=int))
    selected, _ = select_views(matrix, layout)
    assert selected == [0]


@pytest.mark.parametrize("rotation", [[0, 0, 0], [0, 90, 0], [17, -32, 8]])
def test_camera_lineage_and_projection_are_preserved(tmp_path, rotation):
    scan = tmp_path / "scan_0"
    alignment = tmp_path / "multi_room_alignment"
    view = np.eye(4)
    view[:3, 3] = [0.2, -0.1, 0.5]
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", rotation, degrees=True).as_matrix()
    transform[:3, 3] = [2, 0.3, -4]
    write_model(scan / ".raw_unaligned/sparse/0", [view])
    write_model(alignment / "scans/scan_0/sparse/0", [view @ np.linalg.inv(transform)])
    cameras, lineage = load_cameras(scan, alignment, transform)
    actual = np.array(cameras[0]["world_to_camera"])
    point = np.array([1, 2, 3, 1])
    np.testing.assert_allclose(actual @ transform @ point, view @ point, atol=1e-10)
    assert lineage["checked_images"] == 1
    assert cameras[0]["image_id"] == 1
    assert cameras[0]["camera_model"] == "PINHOLE"
    if np.linalg.norm(transform - np.eye(4)) > 0.1:
        write_model(alignment / "scans/scan_0/sparse/0", [view])
        with pytest.raises(ValueError, match="Stale or doubly transformed"):
            load_cameras(scan, alignment, transform)


def test_visibility_checks_occlusion_camera_axes_and_out_of_frame():
    vertices, faces = planes()
    view = np.eye(4, dtype=np.float32)[None]
    k = np.array([[[40, 0, 29], [0, 44, 21], [0, 0, 1]]], dtype=np.float32)
    points = np.array([[0, 0, 4], [1.2, 0, 4], [0, 0, -2], [30, 0, 4], [0, 0, 2]], dtype=np.float32)
    normal = np.tile([0, 0, -1], (len(points), 1))
    renderer = SurfaceVisibility(vertices, faces, points, normal, np.full(len(points), 0.1), "cpu")
    scores, _, _ = renderer.evaluate(view, k, 64, 48)
    assert scores[0].tolist() == pytest.approx([0, 1, 0, 0, 1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm/CUDA GPU required")
def test_gpu_batches_match_cpu_and_each_other():
    vertices, faces = planes()
    views = np.repeat(np.eye(4, dtype=np.float32)[None], 2, 0)
    views[1, 0, 3] = 0.2
    k = np.repeat(np.array([[[40, 0, 29], [0, 44, 21], [0, 0, 1]]], dtype=np.float32), 2, 0)
    samples = sample_surface(vertices, faces, 2048)
    args = (vertices, faces, samples["positions"], samples["normals"], samples["areas"])
    gpu, cpu = SurfaceVisibility(*args, "cuda"), SurfaceVisibility(*args, "cpu")
    together, _, _ = gpu.evaluate(views, k, 64, 48)
    reference, _, _ = cpu.evaluate(views, k, 64, 48)
    separately = torch.cat([gpu.evaluate(views[i:i + 1], k[i:i + 1], 64, 48)[0] for i in range(2)])
    torch.testing.assert_close(together, separately)
    torch.testing.assert_close(together.cpu(), reference, rtol=1e-4, atol=1e-5)
    # Reversed winding must remain visible from the acquired camera.
    reverse = SurfaceVisibility(vertices, faces[:, ::-1].copy(), *args[2:], "cuda")
    reversed_scores, _, _ = reverse.evaluate(views, k, 64, 48)
    torch.testing.assert_close(together, reversed_scores)


def test_framing_penalizes_floor_ceiling_and_near_obstruction():
    pixels = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    geometry = {"camera_depth": torch.full((4, 4), 2.)}
    geometry["camera_depth"][2, 2] = 0.1
    roles = np.array(["floor", "ceiling", "object", "wall"])
    views = np.repeat(np.eye(4, dtype=np.float32)[None], 4, 0)
    quality, stats = framing_quality(pixels, geometry, roles, views, 5)
    assert float(quality[3]) == pytest.approx(0.25)  # A single-surface crop lacks room context.
    assert bool(stats["excluded_for_unhelpful_framing"][3])
    assert bool((quality[:3] < 0.3).all())


def test_mesh_foreground_occlusion_over_40_percent_is_rejected():
    pixels = torch.tensor([[0.5, 0.3, 0.2], [0.39, 0.41, 0.2]])
    geometry = {"camera_depth": torch.tensor([[0.1, 2., 2.], [0.1, 2., 2.]])}
    roles = np.array(["object", "wall", "floor"])
    views = np.repeat(np.eye(4, dtype=np.float32)[None], 2, 0)
    _, stats = framing_quality(pixels, geometry, roles, views, 5)
    assert stats["near_foreground_fraction"].tolist() == pytest.approx([0.5, 0.39])
    assert stats["excluded_for_unhelpful_framing"].tolist() == [True, False]


def test_single_surface_closeup_over_75_percent_is_rejected():
    pixels = torch.tensor([[0.79, 0.21], [0.74, 0.26]])
    geometry = {"camera_depth": torch.full((2, 2), 2.)}
    roles = np.array(["wall", "object"])
    views = np.repeat(np.eye(4, dtype=np.float32)[None], 2, 0)
    _, stats = framing_quality(pixels, geometry, roles, views, 5)
    assert stats["known_dominant_surface_fraction"].tolist() == pytest.approx([0.79, 0.74])
    assert stats["excluded_for_unhelpful_framing"].tolist() == [True, False]


def test_blur_gate_excludes_soft_focus_instead_of_letting_coverage_compensate(tmp_path):
    rng = np.random.default_rng(4)
    image = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    images = []
    for index in range(8):
        source = Image.fromarray(image)
        if index < 7:
            source = source.filter(ImageFilter.GaussianBlur(5))
        path = tmp_path / f"frame_{index}.jpg"
        source.save(path, quality=95)
        images.append({"image_id": index + 1, "image_name": path.name, "source_image": str(path),
                       "width": 320, "height": 240})
    usable, excluded = image_quality(images)
    assert 1 not in [camera["image_id"] for camera in usable]
    assert any(item["image_id"] == 1 and item["reason"] == "soft_focus_or_blur" for item in excluded)
    assert [camera["image_id"] for camera in usable] == [8]


def test_camera_pose_quality_prefers_level_eye_height_and_handles_stairs():
    from scipy.spatial.transform import Rotation

    level = np.eye(4)
    level[:3, :3] = Rotation.from_euler("x", 180, degrees=True).as_matrix()
    floor_points = np.array([[x, y, z] for y in (0., 2.5)
                             for x in np.linspace(-1, 1, 20) for z in np.linspace(-1, 1, 20)])
    poses = []
    for y, roll, pitch in ((1.45, 0, 0), (3.95, 0, 0), (.45, 0, 0), (1.05, 0, 0),
                           (1.45, 15, 0), (1.45, 0, 42)):
        c2w = level.copy()
        c2w[:3, :3] = level[:3, :3] @ (
            Rotation.from_euler("x", pitch, degrees=True) * Rotation.from_euler("z", roll, degrees=True)
        ).as_matrix()
        c2w[:3, 3] = [0, y, 0]
        poses.append({"image_id": len(poses) + 1, "image_name": f"{len(poses)}.jpg",
                      "camera_to_world": c2w, "quality": {"factor": 1.}})
    accepted, excluded = camera_pose_quality(poses, [0, 1, 0], floor_points, metric_scale=True)
    assert [camera["image_id"] for camera in accepted] == [1, 2, 5]
    assert accepted[0]["quality"]["pose"]["height_above_nearby_floor_m"] == pytest.approx(1.45)
    assert accepted[1]["quality"]["pose"]["height_above_nearby_floor_m"] == pytest.approx(1.45)
    assert accepted[2]["quality"]["pose"]["roll_factor"] < 1
    assert {camera["image_id"] for camera in excluded} == {3, 4, 6}


def test_cpu_scan_exports_unchanged_images_aligned_poses_and_replaces_stale_output(tmp_path):
    scan, alignment = tmp_path / "scan_0", tmp_path / "multi_room_alignment"
    views = [np.diag([-1., -1., 1., 1.]), np.diag([-1., -1., 1., 1.])]
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("y", 30, degrees=True).as_matrix()
    transform[:3, 3] = [3, 0, -4]
    write_model(scan / ".raw_unaligned/sparse/0", views)
    write_model(alignment / "scans/scan_0/sparse/0", [v @ np.linalg.inv(transform) for v in views])
    write_mesh(scan / ".raw_unaligned/mesh.ply", *planes())
    (scan / "images").mkdir()
    image = np.indices((48, 64)).sum(0) % 2 * 180 + 30
    for index in (1, 2):
        Image.fromarray(image.astype(np.uint8)).convert("RGB").save(scan / f"images/frame_{index}.jpg")
    output = tmp_path / "artifacts/listing_images"
    output.mkdir(parents=True)
    result = extract_scan(scan, alignment, output, transform, "cpu", 2, 1024, 64)
    assert result["selected_count"] == 1
    manifest = json.loads((output / "scan_0/selected_images.json").read_text())
    record = manifest["images"][0]
    assert record["image_id"] == 1
    assert (output / "scan_0" / record["image"]).read_bytes() == Path(record["source_image"]).read_bytes()
    assert manifest["coordinate_frame"] == "phase07_aligned_world"
    np.testing.assert_allclose(record["camera_to_world"], transform @ views[0], atol=1e-12)
    stale = output / "scan_0/images/stale.jpg"
    stale.write_bytes(b"stale")
    extract_scan(scan, alignment, output, transform, "cpu", 2, 1024, 64)
    assert not stale.exists()


def test_public_entry_rejects_writing_into_inputs(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        extract_best_views(tmp_path / "multi_room_alignment", tmp_path, "cpu")


def test_reprojection_guard_rejects_internally_broken_colmap_poses(tmp_path):
    import pycolmap

    write_model(tmp_path, [np.eye(4)])
    header = (tmp_path / "images.txt").read_text().splitlines()[0]
    observations, points = [], []
    for j in range(64):
        x, y, z = (j % 8 - 3.5) / 4, (j // 8 - 3.5) / 4, 4
        observations.extend((str(40 * x / z + 29), str(44 * y / z + 21), str(j + 1)))
        points.append(f"{j + 1} {x} {y} {z} 100 100 100 0 1 {j}")
    (tmp_path / "images.txt").write_text(header + "\n" + " ".join(observations) + "\n")
    (tmp_path / "points3D.txt").write_text("\n".join(points) + "\n")
    assert check_reprojections(pycolmap.Reconstruction(str(tmp_path)))["status"] == "passed"
    fields = header.split()
    fields[5] = "2.0"
    (tmp_path / "images.txt").write_text(" ".join(fields) + "\n" + " ".join(observations) + "\n")
    with pytest.raises(ValueError, match="reprojection check failed"):
        check_reprojections(pycolmap.Reconstruction(str(tmp_path)))


def test_public_entry_records_failures_without_deleting_previous_gallery(tmp_path):
    alignment = tmp_path / "multi_room_alignment"
    (alignment / "stats").mkdir(parents=True)
    (alignment / "stats/aligned_scans_transforms.json").write_text(json.dumps({
        "scans": {"scan_0": {"transform_matrix": np.eye(4).tolist()}}}))
    output = tmp_path / "artifacts/listing_images"
    previous = output / "scan_0/selected_images.json"
    previous.parent.mkdir(parents=True)
    previous.write_text("{}")
    result = extract_best_views(alignment, output, "cpu")
    assert result["macro_step_status"] == "failed"
    assert result["scans"][0]["previous_output_preserved"]
    assert previous.read_text() == "{}"


def test_publish_refuses_unrelated_existing_directory(tmp_path):
    staged, target = tmp_path / "staged", tmp_path / "existing"
    staged.mkdir()
    target.mkdir()
    (target / "unrelated.txt").write_text("keep")
    with pytest.raises(ValueError, match="unrelated"):
        _publish(staged, target)
    assert (target / "unrelated.txt").read_text() == "keep"


def test_failed_publication_restores_previous_gallery(tmp_path, monkeypatch):
    staged, target = tmp_path / "staged", tmp_path / "scan_0"
    staged.mkdir()
    target.mkdir()
    (target / "selected_images.json").write_text(json.dumps({"macro_step": "extract_best_views", "sentinel": "previous"}))
    rename = Path.rename
    def failing_rename(path, destination):
        if path == staged:
            raise PermissionError("simulated publication failure")
        return rename(path, destination)
    monkeypatch.setattr(Path, "rename", failing_rename)
    with pytest.raises(PermissionError):
        _publish(staged, target)
    assert json.loads((target / "selected_images.json").read_text())["sentinel"] == "previous"


def test_batch_reduction_handles_only_out_of_memory(monkeypatch):
    vertices, faces = planes()
    samples = sample_surface(vertices, faces, 128)
    renderer = SurfaceVisibility(vertices, faces, samples["positions"], samples["normals"], samples["areas"], "cpu")
    evaluate = renderer.evaluate
    def limited(views, *args):
        if len(views) > 1:
            raise torch.OutOfMemoryError("simulated batch allocation")
        return evaluate(views, *args)
    monkeypatch.setattr(renderer, "evaluate", limited)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    cameras = [{"width": 64, "height": 48, "intrinsics": [[40, 0, 29], [0, 44, 21], [0, 0, 1]],
                "world_to_camera": np.eye(4).tolist(), "quality": {"factor": 1.}} for _ in range(3)]
    args = (renderer, cameras, samples, np.zeros(3), np.full(128, -1), [], np.full(128, "wall"))
    visibility, _, report = measure_views(*args, batch_size=8, long_edge=64)
    assert report["batch_sizes"] == [1]
    torch.testing.assert_close(visibility[0], visibility[1])
    def broken(*args):
        raise RuntimeError("broken visibility backend")
    monkeypatch.setattr(renderer, "evaluate", broken)
    with pytest.raises(RuntimeError, match="broken visibility backend"):
        measure_views(*args, batch_size=8, long_edge=64)
