import importlib.util
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData


MODULE_PATH = Path(__file__).resolve().parents[1] / "06_mesh_elaboration" / "add_ceilings_to_floor_levels.py"
SPEC = importlib.util.spec_from_file_location("add_ceilings_to_floor_levels", MODULE_PATH)
association = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(association)


def _write_instance(path: Path, height: float, x: float, flag: int) -> None:
    assert association.export_ply_with_face_attributes(
        path,
        np.array([[x, height, 0], [x + 1, height, 0], [x, height, 1]], dtype=float),
        np.array([[0, 1, 2]], dtype=np.int32),
        np.full((3, 3), 150, dtype=np.uint8),
        face_attributes={"is_hole_fill": ("uchar", np.array([flag], dtype=np.uint8))},
    )


def _write_manifests(root: Path, ceilings: list[str]) -> Path:
    floors = ["floor_inst000.ply", "floor_inst001.ply"]
    (root / "structural_instances_manifest.json").write_text(json.dumps({
        "instances": ([{"role": "floor", "file_name": name} for name in floors] +
                      [{"role": "ceiling", "file_name": name} for name in ceilings]),
    }))
    output = root / "intermediate_floor_levels"
    output.mkdir(exist_ok=True)
    manifest = {
        "floor_count": 2,
        "levels": [
            {"level": 0, "instances": [{"file_name": floors[0]}],
             "output_file": "floor_level_000_intermediate.ply", "num_triangles": 1},
            {"level": 1, "instances": [{"file_name": floors[1]}],
             "output_file": "floor_level_001_intermediate.ply", "num_triangles": 1},
        ],
    }
    path = output / "floor_level_intermediate_manifest.json"
    path.write_text(json.dumps(manifest))
    for index, name in enumerate(floors):
        _write_instance(root / name, float(index * 2), float(index * 2), 0)
        output_path = output / f"floor_level_{index:03d}_intermediate.ply"
        output_path.write_bytes((root / name).read_bytes())
    return path


def test_adds_all_ceilings_inside_the_below_band_and_labels_each_face(tmp_path):
    eligible = ["ceiling_inst000.ply", "ceiling_inst001.ply"]
    manifest_path = _write_manifests(tmp_path, eligible + ["ceiling_inst002.ply", "ceiling_inst003.ply"])
    _write_instance(tmp_path / eligible[0], 1.5, 10.0, 1)   # Exactly 0.50 m below upper floor.
    _write_instance(tmp_path / eligible[1], 1.75, 12.0, 0)
    _write_instance(tmp_path / "ceiling_inst002.ply", 1.49, 14.0, 0)  # Outside by 1 cm.
    _write_instance(tmp_path / "ceiling_inst003.ply", 2.1, 16.0, 0)   # Above the floor.
    lower_path = tmp_path / "intermediate_floor_levels/floor_level_000_intermediate.ply"
    lower_before = lower_path.read_bytes()

    association.add_ceilings_to_floor_levels(tmp_path)

    lower_after = lower_path.read_bytes()
    assert lower_after == lower_before
    upper_path = tmp_path / "intermediate_floor_levels/floor_level_001_intermediate.ply"
    upper_mesh = o3d.io.read_triangle_mesh(str(upper_path))
    assert len(upper_mesh.triangles) == 3
    assert len(upper_mesh.vertex_colors) == 9
    faces = PlyData.read(str(upper_path))["face"].data
    np.testing.assert_array_equal(faces["surface_role"], [0, 2, 2])
    np.testing.assert_array_equal(faces["is_hole_fill"], [0, 1, 0])
    saved = json.loads(manifest_path.read_text())
    assert [item["file_name"] for item in saved["levels"][1]["ceilings"]] == eligible
    assert saved["surface_role_values"] == {"0": "floor", "1": "ramp", "2": "ceiling"}

    updated_bytes = upper_path.read_bytes()
    association.add_ceilings_to_floor_levels(tmp_path)
    assert upper_path.read_bytes() == updated_bytes
    assert len(o3d.io.read_triangle_mesh(str(upper_path)).triangles) == 3


def test_ceiling_append_keeps_floor_ramp_and_ceiling_roles_distinct(tmp_path):
    manifest_path = _write_manifests(tmp_path, ["ceiling_inst000.ply"])
    _write_instance(tmp_path / "ceiling_inst000.ply", 1.5, 10.0, 0)
    ramp_vertices = np.array([[2, 2, 0], [3, 2, 0], [2, 2.3, 1]], dtype=float)
    assert association.export_ply_with_face_attributes(
        tmp_path / "ramp_inst000.ply", ramp_vertices, np.array([[0, 1, 2]], dtype=np.int32),
        np.full((3, 3), 150, dtype=np.uint8),
        face_attributes={"is_hole_fill": ("uchar", np.array([0], dtype=np.uint8))},
    )
    structural = json.loads((tmp_path / "structural_instances_manifest.json").read_text())
    structural["instances"].append({"role": "ramp", "file_name": "ramp_inst000.ply"})
    (tmp_path / "structural_instances_manifest.json").write_text(json.dumps(structural))
    saved = json.loads(manifest_path.read_text())
    saved["levels"][1]["ramps"] = [{"file_name": "ramp_inst000.ply"}]
    manifest_path.write_text(json.dumps(saved))
    floor = association._read_instance(tmp_path, "floor_inst001.ply", "floor")
    ramp = association._read_instance(tmp_path, "ramp_inst000.ply", "ramp")
    upper_path = tmp_path / "intermediate_floor_levels/floor_level_001_intermediate.ply"
    association._write_mesh(upper_path, [(floor, 0), (ramp, 1)], mark_surface_roles=True)

    association.add_ceilings_to_floor_levels(tmp_path)

    faces = PlyData.read(str(upper_path))["face"].data
    np.testing.assert_array_equal(faces["surface_role"], [0, 1, 2])
    saved = json.loads(manifest_path.read_text())
    assert saved["surface_role_values"] == {"0": "floor", "1": "ramp", "2": "ceiling"}


def test_no_matching_ceiling_leaves_intermediate_plys_and_manifest_unchanged(tmp_path):
    manifest_path = _write_manifests(tmp_path, ["ceiling_inst000.ply"])
    _write_instance(tmp_path / "ceiling_inst000.ply", 1.4, 10.0, 0)
    outputs = sorted((tmp_path / "intermediate_floor_levels").glob("*.ply"))
    before = {path: path.read_bytes() for path in outputs}
    manifest_before = manifest_path.read_bytes()

    association.add_ceilings_to_floor_levels(tmp_path)

    assert {path: path.read_bytes() for path in outputs} == before
    assert manifest_path.read_bytes() == manifest_before


def test_rerun_replaces_previous_ceiling_merge_when_it_no_longer_matches(tmp_path):
    manifest_path = _write_manifests(tmp_path, ["ceiling_inst000.ply"])
    ceiling_path = tmp_path / "ceiling_inst000.ply"
    _write_instance(ceiling_path, 1.5, 10.0, 1)
    association.add_ceilings_to_floor_levels(tmp_path)

    # Move the ceiling outside the 50 cm band; rerun should restore floor-only output.
    _write_instance(ceiling_path, 1.0, 10.0, 1)
    association.add_ceilings_to_floor_levels(tmp_path)

    output = tmp_path / "intermediate_floor_levels/floor_level_001_intermediate.ply"
    assert output.read_bytes() == (tmp_path / "floor_inst001.ply").read_bytes()
    assert list(PlyData.read(str(output))["face"].data.dtype.names) == ["vertex_indices", "is_hole_fill"]
    saved = json.loads(manifest_path.read_text())
    assert saved["levels"][1]["ceilings"] == []
    assert "surface_role_values" not in saved
