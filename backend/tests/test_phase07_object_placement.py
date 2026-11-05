import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import trimesh


SYNCHRONIZER_PATH = Path(__file__).resolve().parents[1] / "07_alignment" / "multi_asset_synchronizer.py"
if str(SYNCHRONIZER_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SYNCHRONIZER_PATH.parent))
SPEC = importlib.util.spec_from_file_location("multi_asset_synchronizer", SYNCHRONIZER_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_fitted_object_publish_sums_stage_translations_and_preserves_colors(tmp_path):
    output_dir = tmp_path / "multi_room_alignment"
    stats_dir = output_dir / "stats"
    object_path = (output_dir / "objects" / "scan_0" / "class_003_object"
                   / "instance_001.ply")
    object_path.parent.mkdir(parents=True)
    original_colors = np.array([[80, 120, 160, 255]] * 3, dtype=np.uint8)
    obj = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]],
        vertex_colors=original_colors,
        process=False,
    )
    obj.export(object_path)
    sealed = trimesh.Trimesh(
        vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        faces=[[0, 1, 2]],
        vertex_colors=[200, 200, 200, 255],
        process=False,
    )
    sealed.export(output_dir / "aligned_sealed_meshes.ply")

    manifest_path = stats_dir / "aligned_objects_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({
        "total_objects": 1,
        "objects": [{
            "source_instance_path": str(tmp_path / "scan_0" / "chair_instance_1.ply"),
            "global_instance_path": str(object_path),
            "global_bounding_box": [0, 0, 0, 1, 1, 0],
            "global_centroid": [1 / 3, 1 / 3, 0],
        }],
    }))
    source = str(tmp_path / "scan_0" / "chair_instance_1.ply")
    inside_report = {"objects": [{"source_instance_path": source, "translation_m": [0.1, 0, 0]}]}
    plane_report = {"objects": [{"source_instance_path": source, "translation_m": [0.2, 0, 0]}]}

    result = MODULE.MultiAssetSynchronizer(output_dir).apply_object_placement_translations(
        inside_report, plane_report
    )

    fitted = trimesh.load(object_path, process=False)
    np.testing.assert_allclose(fitted.vertices, obj.vertices + [0.3, 0, 0])
    np.testing.assert_array_equal(fitted.visual.vertex_colors, original_colors)
    np.testing.assert_array_equal(
        trimesh.load(result["aligned_objects"], process=False).visual.vertex_colors,
        original_colors,
    )
    assert not (output_dir / "aligned_scene_with_objects.ply").exists()
    final_manifest = json.loads(manifest_path.read_text())
    np.testing.assert_allclose(final_manifest["objects"][0]["post_scan_translation_m"], [0.3, 0, 0])


def test_aligned_objects_export_by_scan_class_and_instance(tmp_path):
    scan_dir = tmp_path / "scan_0"
    instance_path = (scan_dir / "segmentation" / "3d_masks" / "class_003_object"
                     / "instance_001.ply")
    instance_path.parent.mkdir(parents=True)
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]],
        vertex_colors=[[80, 120, 160, 255]] * 3,
        process=False,
    )
    mesh.export(instance_path)

    output_dir = tmp_path / "multi_room_alignment"
    objects_dir = output_dir / "objects"
    combined_path = output_dir / "aligned_objects.ply"
    manifest_path = output_dir / "stats" / "aligned_objects_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    scan = type("Scan", (), {"info": type("Info", (), {"scan_dir": scan_dir})()})()

    MODULE.MultiAssetSynchronizer(output_dir)._assemble_objects(
        {"scan_0": scan}, {"scan_0": np.eye(4)}, objects_dir,
        combined_path, manifest_path,
    )

    assert (objects_dir / "scan_0" / "class_003_object" / "instance_001.ply").is_file()
    record = json.loads(manifest_path.read_text())["objects"][0]
    assert Path(record["global_instance_path"]) == (
        objects_dir / "scan_0" / "class_003_object" / "instance_001.ply"
    )
