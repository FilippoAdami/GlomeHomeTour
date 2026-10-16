"""Manifest-driven structural level reporting checks."""
import argparse
import importlib.util
import json
from pathlib import Path

import open3d as o3d


MODULE_PATH = Path(__file__).resolve().parents[1] / "06_mesh_elaboration" / "group_floors_ceilings_levels.py"
SPEC = importlib.util.spec_from_file_location("group_floors_ceilings_levels", MODULE_PATH)
levels = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(levels)


def square(x, z, size=2):
    return {"type": "Polygon", "coordinates": [[[x, z], [x + size, z], [x + size, z + size], [x, z + size], [x, z]]]}


def unit(name, role, elevation, footprint):
    return {
        "unit_name": name,
        "structural_role": role,
        "plane_equation": [0, 1, 0, -elevation],
        "mean_elevation_y": elevation,
        "footprint_xz": footprint,
        "regularized_artifact": f"{name}.ply",
        "pattern_classification": {"pattern_type": "plain"},
    }


def run(tmp_path, monkeypatch, units, slabs=()):
    instances = tmp_path / "instances"
    output = tmp_path / "output"
    instances.mkdir()
    manifest = {"units": units, "slabs": list(slabs)}
    (instances / "plane_patterns_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(levels, "parse_args", lambda: argparse.Namespace(
        instances_dir=instances, output_dir=output, scene_size_file=tmp_path / "scene_size.txt",
    ))
    return instances, output, levels.main()


def test_one_level_reports_overlap_clear_height_and_copies_valid_slab(tmp_path, monkeypatch):
    slab_source = tmp_path / "valid_slab.ply"
    assert o3d.io.write_triangle_mesh(str(slab_source), o3d.geometry.TriangleMesh.create_box())
    units = [unit("floor_a", "floor", 0, square(0, 0)), unit("ceiling_a", "ceiling", 2.4, square(0, 0))]
    slab = {
        "slab_name": "slab_a", "floor": "floor_a", "ceiling": "ceiling_a", "file_path": str(slab_source),
        "footprint_xz": square(0, 0), "floor_plane_equation": [0, 1, 0, 0],
        "floor_clearance_lift_m": 0.0, "is_watertight": True, "volume_m3": 1.0,
    }
    _, output, status = run(tmp_path, monkeypatch, units, [slab])
    assert status == 0
    report = json.loads((output / "structural_levels_manifest.json").read_text())
    assert len(report["levels"]) == 1
    match = report["levels"][0]["floor_units"][0]["ceiling_match"]
    assert match["ceiling_unit"] == "ceiling_a"
    assert match["clear_height_m"] == 2.4
    assert (output / "slab_a.ply").exists()


def test_groups_arbitrary_floor_elevations_within_fifteen_centimetres(tmp_path, monkeypatch):
    units = [
        unit("floor_0a", "floor", 0.00, square(0, 0)), unit("floor_0b", "floor", 0.14, square(3, 0)),
        unit("floor_1", "floor", 3.0, square(0, 3)), unit("floor_2a", "floor", 6.0, square(3, 3)),
        unit("floor_2b", "floor", 6.08, square(6, 3)),
    ]
    _, output, status = run(tmp_path, monkeypatch, units)
    assert status == 0
    report = json.loads((output / "structural_levels_manifest.json").read_text())
    assert [[floor["unit_name"] for floor in level["floor_units"]] for level in report["levels"]] == [
        ["floor_0a", "floor_0b"], ["floor_1"], ["floor_2a", "floor_2b"],
    ]


def test_disconnected_rooms_match_only_their_overlapping_ceiling(tmp_path, monkeypatch):
    units = [
        unit("floor_left", "floor", 0, square(0, 0)), unit("floor_right", "floor", 0, square(10, 0)),
        unit("ceiling_left", "ceiling", 2, square(0, 0)), unit("ceiling_right", "ceiling", 4, square(10, 0)),
    ]
    _, output, status = run(tmp_path, monkeypatch, units)
    assert status == 0
    floors = json.loads((output / "structural_levels_manifest.json").read_text())["levels"][0]["floor_units"]
    assert {floor["unit_name"]: floor["ceiling_match"]["ceiling_unit"] for floor in floors} == {
        "floor_left": "ceiling_left", "floor_right": "ceiling_right",
    }


def test_missing_manifest_preserves_existing_output(tmp_path, monkeypatch):
    instances, output = tmp_path / "instances", tmp_path / "output"
    instances.mkdir()
    output.mkdir()
    existing = output / "structural_levels_manifest.json"
    existing.write_bytes(b"prior report")
    monkeypatch.setattr(levels, "parse_args", lambda: argparse.Namespace(
        instances_dir=instances, output_dir=output, scene_size_file=tmp_path / "scene_size.txt",
    ))
    assert levels.main() == 1
    assert existing.read_bytes() == b"prior report"


def test_empty_floor_list_reports_no_levels(tmp_path, monkeypatch):
    _, output, status = run(tmp_path, monkeypatch, [unit("ceiling_only", "ceiling", 2, square(0, 0))])
    assert status == 0
    assert json.loads((output / "structural_levels_manifest.json").read_text())["levels"] == []


def test_wall_without_floor_footprint_or_elevation_does_not_block_levels(tmp_path, monkeypatch):
    wall = {
        "unit_name": "wall_line", "structural_role": "wall", "plane_equation": [1, 0, 0, 0],
        "regularized_artifact": "wall_line.ply", "pattern_classification": {"pattern_type": "plain"},
    }
    _, output, status = run(tmp_path, monkeypatch, [unit("floor_a", "floor", 0, square(0, 0)), wall])
    assert status == 0
    report = json.loads((output / "structural_levels_manifest.json").read_text())
    assert [floor["unit_name"] for floor in report["levels"][0]["floor_units"]] == ["floor_a"]


def test_invalid_slab_never_replaces_existing_report(tmp_path, monkeypatch):
    invalid_slab = tmp_path / "open_slab.ply"
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
    assert o3d.io.write_triangle_mesh(str(invalid_slab), mesh)
    units = [unit("floor_a", "floor", 0, square(0, 0)), unit("ceiling_a", "ceiling", 2, square(0, 0))]
    slab = {"slab_name": "broken", "floor": "floor_a", "ceiling": "ceiling_a", "file_path": str(invalid_slab),
            "footprint_xz": square(0, 0), "floor_plane_equation": [0, 1, 0, 0], "is_watertight": True}
    instances, output, _ = run(tmp_path, monkeypatch, units, [slab])
    existing = output / "structural_levels_manifest.json"
    # The first call has already failed during publication; seed the old report and retry.
    output.mkdir(exist_ok=True)
    existing.write_bytes(b"old report")
    monkeypatch.setattr(levels, "parse_args", lambda: argparse.Namespace(
        instances_dir=instances, output_dir=output, scene_size_file=tmp_path / "scene_size.txt",
    ))
    assert levels.main() == 1
    assert existing.read_bytes() == b"old report"
