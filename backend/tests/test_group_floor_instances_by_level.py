import importlib.util
import json
from pathlib import Path

import numpy as np
import open3d as o3d


MODULE_PATH = Path(__file__).resolve().parents[1] / "06_mesh_elaboration" / "group_floor_instances_by_level.py"
SPEC = importlib.util.spec_from_file_location("group_floor_instances_by_level", MODULE_PATH)
grouping = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(grouping)


def _write_floor(path: Path, height: float, x: float, flag: int, color: tuple[int, int, int]) -> None:
    assert grouping.export_ply_with_face_attributes(
        path,
        np.array([[x, height, 0], [x + 1, height, 0], [x, height, 1]], dtype=float),
        np.array([[0, 1, 2]], dtype=np.int32),
        np.tile(np.array(color, dtype=np.uint8), (3, 1)),
        face_attributes={"is_hole_fill": ("uchar", np.array([flag], dtype=np.uint8))},
    )


def _manifest(instances_dir: Path, files: list[str]) -> None:
    (instances_dir / "structural_instances_manifest.json").write_text(json.dumps({
        "instances": [{"role": "floor", "file_name": name} for name in files],
    }))


def test_groups_fragments_by_height_and_preserves_mesh_attributes(tmp_path):
    files = [f"floor_inst{i:03d}.ply" for i in range(4)]
    for filename, height, x, flag, color in zip(
        files, [0.0, 0.08, 2.0, 2.08], [0.0, 2.0, 4.0, 6.0], [0, 1, 1, 0],
        [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)],
    ):
        _write_floor(tmp_path / filename, height, x, flag, color)
    _manifest(tmp_path, files)
    scene_size = tmp_path / "scene_size.txt"
    scene_size.write_text("floors: 2\n")

    output = grouping.group_floor_instances(tmp_path, scene_size)

    levels = json.loads((output / "floor_level_intermediate_manifest.json").read_text())["levels"]
    assert [[item["file_name"] for item in level["instances"]] for level in levels] == [files[:2], files[2:]]
    for level, expected_flags in zip(levels, [[0, 1], [1, 0]]):
        path = output / level["output_file"]
        mesh = o3d.io.read_triangle_mesh(str(path))
        assert len(mesh.vertices) == 6
        assert len(mesh.triangles) == 2
        np.testing.assert_array_equal(grouping.read_ply_face_flags(path), expected_flags)
        assert len(mesh.vertex_colors) == 6


def test_one_instance_per_level_is_copied_and_stale_level_files_are_removed(tmp_path):
    files = ["floor_inst000.ply", "floor_inst001.ply"]
    for filename, height, x in zip(files, [0.0, 2.0], [0.0, 2.0]):
        _write_floor(tmp_path / filename, height, x, 0, (128, 128, 128))
    _manifest(tmp_path, files)
    scene_size = tmp_path / "scene_size.txt"
    scene_size.write_text("floors: 2\n")

    output = tmp_path / "intermediate_floor_levels"
    output.mkdir()
    for index in range(2):
        (output / f"floor_level_{index:03d}_intermediate.ply").write_bytes(b"stale output")
    stale = output / "floor_level_002_intermediate.ply"
    stale.touch()
    grouping.group_floor_instances(tmp_path, scene_size)

    for index, filename in enumerate(files):
        assert (output / f"floor_level_{index:03d}_intermediate.ply").read_bytes() == (tmp_path / filename).read_bytes()
    assert not stale.exists()


def test_adds_only_ramps_sharing_a_geometric_edge_with_a_floor(tmp_path):
    floors = ["floor_inst000.ply", "floor_inst001.ply"]
    _write_floor(tmp_path / floors[0], 0.0, 0.0, 0, (128, 128, 128))
    _write_floor(tmp_path / floors[1], 2.0, 3.0, 0, (128, 128, 128))
    ramp = tmp_path / "ramp_inst000.ply"
    assert grouping.export_ply_with_face_attributes(
        ramp,
        np.array([[0, 0, 0], [1, 0, 0], [0, 0.3, 1]], dtype=float),
        np.array([[0, 1, 2]], dtype=np.int32),
        np.full((3, 3), 128, dtype=np.uint8),
        face_attributes={"is_hole_fill": ("uchar", np.array([0], dtype=np.uint8))},
    )
    (tmp_path / "structural_instances_manifest.json").write_text(json.dumps({"instances": [
        *[{"role": "floor", "file_name": name} for name in floors],
        {"role": "ramp", "file_name": "ramp_inst000.ply"},
    ]}))
    scene_size = tmp_path / "scene_size.txt"
    scene_size.write_text("floors: 2\n")

    output = grouping.group_floor_instances(tmp_path, scene_size)

    level0 = json.loads((output / "floor_level_intermediate_manifest.json").read_text())["levels"][0]
    assert [item["file_name"] for item in level0["ramps"]] == ["ramp_inst000.ply"]
    ply = output / level0["output_file"]
    faces = grouping.read_ply_face_flags(ply, "surface_role")
    np.testing.assert_array_equal(faces, [0, 1])


def test_rejects_more_declared_levels_than_floor_instances(tmp_path):
    floor = "floor_inst000.ply"
    _write_floor(tmp_path / floor, 0.0, 0.0, 0, (128, 128, 128))
    _manifest(tmp_path, [floor])
    scene_size = tmp_path / "scene_size.txt"
    scene_size.write_text("floors: 2\n")

    try:
        grouping.group_floor_instances(tmp_path, scene_size)
    except ValueError as exc:
        assert "only 1 extracted floor instances" in str(exc)
    else:
        raise AssertionError("Expected floor grouping to reject an impossible floor count")
