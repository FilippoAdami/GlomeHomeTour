import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

STAGE = Path(__file__).resolve().parents[1] / "05_semantic_segmentation"
sys.path.insert(0, str(STAGE))
import scene_inputs


def _scan(root: Path, names=("room/frame one.png", "room/frame two.jpg"), size=(7, 5)) -> Path:
    (root / "images").mkdir(parents=True)
    sparse = root / "sparse/0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.txt").write_text(f"1 PINHOLE {size[0]} {size[1]} 6 4 3 2\n")
    records = []
    for index, name in enumerate(names, 1):
        image = root / "images" / name
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size).save(image)
        records.extend((f"{index} 1 0 0 0 0 0 0 1 {name}", "1 2 -1 3 4 7" if index == 1 else ""))
    (sparse / "images.txt").write_text("\n".join(records) + "\n")
    return root


def _hashes(root: Path) -> dict[Path, str]:
    return {path.relative_to(root): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def test_scene_inputs_load_registered_images_without_upstream_writes(tmp_path):
    scan = _scan(tmp_path / "scan_2")
    before = _hashes(scan)
    cameras = scene_inputs.load_scene_cameras(scan)
    assert [camera["image_path"].suffix for camera in cameras] == [".png", ".jpg"]
    assert scene_inputs.registered_images(scan.parent) == [scan / "images/room/frame one.png", scan / "images/room/frame two.jpg"]
    assert scene_inputs.stage_directory(scan) == scan / "segmentation"
    assert _hashes(scan) == before


def test_phase04_aligned_camera_poses_take_precedence(tmp_path):
    scan = _scan(tmp_path / "scan_0", ("frame.png",))
    aligned = scan / "phase04_alignment/sparse/0"
    aligned.mkdir(parents=True)
    (aligned / "cameras.txt").write_text((scan / "sparse/0/cameras.txt").read_text())
    (aligned / "images.txt").write_text("1 1 0 0 0 -2 3 4 1 frame.png\n\n")
    (scan / "phase04_alignment/alignment.json").write_text("{}")

    assert scene_inputs.resolve_sparse_dir(scan) == aligned
    camera = scene_inputs.load_scene_cameras(scan)[0]
    np.testing.assert_allclose(camera["viewmat"][:3, 3], [-2, 3, 4])


def test_in_place_phase04_poses_take_precedence_over_legacy_sidecar(tmp_path):
    scan = _scan(tmp_path / "scan_0", ("frame.png",))
    aligned = scan / "phase04_alignment/sparse/0"
    aligned.mkdir(parents=True)
    (aligned / "cameras.txt").write_text((scan / "sparse/0/cameras.txt").read_text())
    (aligned / "images.txt").write_text("1 1 0 0 0 90 0 0 1 frame.png\n\n")
    (scan / "phase04_alignment/alignment.json").write_text("{}")
    (scan / "mesh.json").write_text(json.dumps({"mesh_alignment": {"poses_written_in_place": True}}))
    (scan / "sparse/0/images.txt").write_text("1 1 0 0 0 -2 3 4 1 frame.png\n\n")

    assert scene_inputs.resolve_sparse_dir(scan) == scan / "sparse/0"
    camera = scene_inputs.load_scene_cameras(scan)[0]
    np.testing.assert_allclose(camera["viewmat"][:3, 3], [-2, 3, 4])


def test_scene_parent_ambiguity_and_dimension_and_stem_validation(tmp_path):
    _scan(tmp_path / "scan_0")
    _scan(tmp_path / "scan_1")
    with pytest.raises(ValueError, match="choose one scan_N"):
        scene_inputs.resolve_scene(tmp_path)
    scan = _scan(tmp_path / "single", ("same.png", "same.jpg"))
    with pytest.raises(ValueError, match="stems must be unique"):
        scene_inputs.load_scene_cameras(scan)
    Image.new("RGB", (8, 5)).save(scan / "images/same.png")
    with pytest.raises(ValueError, match="dimensions"):
        scene_inputs.load_scene_cameras(scan)


def test_colmap_counts_preserve_empty_records_and_space_names(tmp_path):
    path = tmp_path / "images.txt"
    path.write_text("# comment\n1 1 0 0 0 0 0 0 1 nested/name with spaces.jpeg\n0 1 -1 2 3 8\n2 1 0 0 0 0 0 0 1 empty.tiff\n\n")
    assert scene_inputs.colmap_point_counts(path) == {"nested/name with spaces.jpeg": 1, "empty.tiff": 0}
    with pytest.raises(FileNotFoundError):
        scene_inputs.colmap_point_counts(tmp_path / "missing.txt")
    path.write_text("1 1 0 0 0 0 0 0 1 broken.png\n1 2\n")
    with pytest.raises(ValueError, match="POINTS2D"):
        scene_inputs.colmap_point_counts(path)


def test_mesh_transform_requires_explicit_proper_rigid_matrix(tmp_path):
    identity = scene_inputs.load_mesh_transform()
    np.testing.assert_array_equal(identity, np.eye(4))
    path = tmp_path / "transform.npy"
    np.save(path, np.diag([1, 1, -1, 1]))
    with pytest.raises(ValueError, match="proper orthonormal"):
        scene_inputs.load_mesh_transform(path)
    np.save(path, np.eye(3))
    with pytest.raises(ValueError, match="4x4"):
        scene_inputs.load_mesh_transform(path)


def test_labels_and_dino_grid_validate_projection_inputs():
    camera = {"height": 5, "width": 7}
    labels = np.array([[0, 1, 2, 2, 1, 0, 1]] * 5, dtype=np.int32)
    assert scene_inputs.validate_labels(labels, camera, classes={1, 2}) is labels
    with pytest.raises(ValueError, match="height and width"):
        scene_inputs.validate_labels(labels[:, :-1], camera)
    with pytest.raises(ValueError, match="allowed set"):
        scene_inputs.validate_labels(labels, camera, classes={1})
    assert scene_inputs.dino_grid(1920, 1080) == (64, 36)
    assert scene_inputs.dino_grid(1080, 1920) == (36, 64)
    assert scene_inputs.dino_grid(5, 7) == (46, 64)
