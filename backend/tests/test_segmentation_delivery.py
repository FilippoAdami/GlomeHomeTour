import json
import os
import sys
from pathlib import Path
import numpy as np
import open3d as o3d
import pytest
from plyfile import PlyData

STAGE = Path(__file__).resolve().parents[1] / "05_semantic_segmentation"
if str(STAGE) not in sys.path:
    sys.path.insert(0, str(STAGE))

import export_delivery


def test_export_delivery_creates_regularized_structure(tmp_path):
    # 1. Create a synthetic test mesh
    num_verts = 6
    num_faces = 4
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
    ], dtype=np.float32)
    tris = np.array([
        [0, 1, 2],
        [1, 3, 2],
        [1, 4, 3],
        [4, 5, 3],
    ], dtype=np.int32)
    v_colors = np.array([
        [200, 10, 20],
        [10, 200, 30],
        [30, 40, 200],
        [150, 150, 20],
        [80, 80, 80],
        [90, 90, 90],
    ], dtype=np.uint8)

    mesh_ply = tmp_path / "mesh.ply"
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(tris)
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(v_colors.astype(np.float64) / 255.0)
    o3d.io.write_triangle_mesh(str(mesh_ply), o3d_mesh)

    triangle_labels = np.array([5, 12, 7, 8], dtype=np.int32)
    instances = {
        "inst_5_1": {
            "class_id": 5,
            "instance_index": 1,
            "triangles": np.array([0], dtype=np.int32),
            "area_m2": 0.5,
            "h_mean": 0.33,
            "h_span": 1.0,
            "bounds_min": [0.0, 0.0, 0.0],
            "bounds_max": [1.0, 1.0, 0.0],
            "spans": [1.0, 1.0, 0.0],
        },
        "inst_12_1": {
            "class_id": 12,
            "instance_index": 1,
            "triangles": np.array([1], dtype=np.int32),
            "area_m2": 0.5,
            "h_mean": 0.66,
            "h_span": 1.0,
            "bounds_min": [0.0, 0.0, 0.0],
            "bounds_max": [1.0, 1.0, 0.0],
            "spans": [1.0, 1.0, 0.0],
        },
        "inst_7_1": {
            "class_id": 7,
            "instance_index": 1,
            "triangles": np.array([2], dtype=np.int32),
            "area_m2": 0.3,
            "h_mean": 1.5,
            "h_span": 2.5,
            "bounds_min": [0.0, 0.0, 0.0],
            "bounds_max": [0.4, 2.5, 0.4],
            "spans": [0.4, 2.5, 0.4],
        },
        "inst_8_1": {
            "class_id": 8,
            "instance_index": 1,
            "triangles": np.array([3], dtype=np.int32),
            "area_m2": 0.4,
            "h_mean": 0.2,
            "h_span": 0.3,
            "bounds_min": [0.0, 0.0, 0.0],
            "bounds_max": [1.0, 0.3, 0.8],
            "spans": [1.0, 0.3, 0.8],
        },
    }
    classifications = {
        5: {"class_id": 5, "structural_role": "wall", "confidence": 0.98, "details": "Structural wall"},
        12: {"class_id": 12, "structural_role": "object", "confidence": 0.85, "details": "Prop object"},
        7: {"class_id": 7, "structural_role": "columns", "confidence": 0.95, "details": "Narrow isolated column"},
        8: {"class_id": 8, "structural_role": "ramps", "confidence": 0.94, "details": "Linear floor ramp"},
    }

    out_dir = tmp_path / "05_semantic_segmentation"
    summary = export_delivery.deliver_stage05_outputs(
        sealed_mesh_path=mesh_ply,
        triangle_labels=triangle_labels,
        instances=instances,
        classifications=classifications,
        output_dir=out_dir,
    )

    # Assert mesh_segmented.ply exists
    segmented_ply = out_dir / "mesh_segmented.ply"
    assert segmented_ply.is_file()

    # Display vertex splitting preserves every face position and its exact palette color.
    exported = PlyData.read(segmented_ply)
    faces_data = exported['face'].data
    np.testing.assert_array_equal(faces_data["class_id"], [5, 12, 7, 8])
    np.testing.assert_array_equal(faces_data["instance_id"], [1, 1, 1, 1])
    positions = np.column_stack([exported['vertex'][axis] for axis in ('x', 'y', 'z')])
    colors = np.column_stack([exported['vertex'][channel] for channel in ('red', 'green', 'blue')])
    for index, face in enumerate(faces_data['vertex_indices']):
        np.testing.assert_array_equal(positions[face], verts[tris[index]])
        expected = np.rint(np.array(export_delivery.get_class_color(triangle_labels[index])[1]) * 255).astype(np.uint8)
        np.testing.assert_array_equal(colors[face], np.tile(expected, (3, 1)))

    # Assert directory structure
    wall_ply = out_dir / "Instances PLYs/Structural/Wall/instance_001.ply"
    assert wall_ply.is_file()
    col_ply = out_dir / "Instances PLYs/Structural/Columns/instance_001.ply"
    assert col_ply.is_file()
    ramp_ply = out_dir / "Instances PLYs/Structural/Ramps/instance_001.ply"
    assert ramp_ply.is_file()
    obj_ply = out_dir / "Instances PLYs/Non-Structural/12/instance_001.ply"
    assert obj_ply.is_file()

    # Verify real colors on instance PLY
    inst_mesh = o3d.io.read_triangle_mesh(str(wall_ply))
    assert len(inst_mesh.vertices) == 3
    assert len(inst_mesh.triangles) == 1
    # Check that colors came from v_colors, not class color
    inst_colors = (np.asarray(inst_mesh.vertex_colors) * 255.0).round().astype(np.uint8)
    np.testing.assert_array_equal(inst_colors, v_colors[:3])

    # Assert class_metadata.json and segmentation.json
    meta_json = out_dir / "class_metadata.json"
    assert meta_json.is_file()
    meta = json.loads(meta_json.read_text())
    assert "classes" in meta
    assert meta["classes"]["5"]["structural_role"] == "wall"
    assert meta["classes"]["12"]["structural_role"] == "object"
    assert meta["classes"]["7"]["structural_role"] == "columns"
    assert meta["classes"]["8"]["structural_role"] == "ramps"

    seg_json = out_dir / "segmentation.json"
    assert seg_json.is_file()
    seg = json.loads(seg_json.read_text())
    assert seg["total_instances"] == 4
    assert "5" in seg["structural_summary"]["wall"]
    assert "12" in seg["structural_summary"]["non_structural"]
    assert "7" in seg["structural_summary"]["columns"]
    assert "8" in seg["structural_summary"]["ramps"]

    # Assert 3d_masks compatibility
    assert (out_dir / "3d_masks/class_005_wall/instance_001.ply").exists()
    assert (out_dir / "3d_masks/class_012_object/instance_001.ply").exists()


def test_structural_classes_share_role_folder_without_overwrite_and_face_ids_match(tmp_path):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                         [2, 0, 0], [3, 0, 0], [2, 1, 0],
                         [4, 0, 0], [5, 0, 0], [4, 1, 0]], dtype=float)
    triangles = np.arange(9).reshape(-1, 3)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                    o3d.utility.Vector3iVector(triangles))
    source = tmp_path / 'mesh.ply'
    assert o3d.io.write_triangle_mesh(str(source), mesh)
    before = source.read_bytes()
    instances = {f'part_{i}': dict(class_id=cid, instance_index=99,
                                 triangles=np.array([i])) for i, cid in enumerate([17, 91, 17])}
    classes = {cid: dict(structural_role='ceiling') for cid in [17, 91]}
    output = tmp_path / 'segmentation'
    manifest = export_delivery.deliver_stage05_outputs(source, np.array([17, 91, 17]), instances, classes, output)
    paths = [item['relative_path'] for item in manifest['instances']]
    assert len(paths) == len(set(paths)) == 3
    assert len(list((output / 'Instances PLYs/Structural/Ceiling').glob('*.ply'))) == 3
    face_ids = PlyData.read(output / 'mesh_segmented.ply')['face']['instance_id']
    np.testing.assert_array_equal(face_ids, [1, 1, 2])
    for item in manifest['instances']:
        cid, index = item['class_id'], item['instance_index']
        assert (output / f'3d_masks/class_{cid:03d}_ceiling/instance_{index:03d}.ply').resolve() == (output / item['relative_path']).resolve()
    assert source.read_bytes() == before


def test_export_rejects_misaligned_labels_instead_of_padding_or_truncating(tmp_path):
    with pytest.raises(ValueError, match='face order'):
        export_delivery.export_segmented_mesh_ply(tmp_path / 'bad.ply', np.zeros((3, 3)),
            np.array([[0, 1, 2]]), np.array([1, 2]), np.array([1]), [1, 2])
    assert not (tmp_path / 'bad.ply').exists()
