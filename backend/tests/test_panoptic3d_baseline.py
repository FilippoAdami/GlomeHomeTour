"""Regression proof for recovery of the saved legacy step-3 baseline."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '06c_semantic_segmentation'))
from panoptic3d_baseline import class_color, legacy_face_sources, recover_baseline_labels


def test_legacy_decoder_uses_per_channel_strict_tolerance_and_keeps_unknown():
    triangles = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    colors = np.zeros((9, 3))
    colors[:3] = class_color(2) + .049
    colors[3:6] = class_color(2) + .050
    colors[6:] = .4
    assert legacy_face_sources(colors, triangles, [2]).tolist() == [2, 0, 0]


def test_exported_instances_override_palette_only_on_exact_surviving_triangles(tmp_path):
    import open3d as o3d
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                         [2, 0, 0], [3, 0, 0], [2, 1, 0],
                         [4, 0, 0], [5, 0, 0], [4, 1, 0]], float)
    triangles = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                     o3d.utility.Vector3iVector(triangles))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(class_color(2), (len(vertices), 1)))
    mesh_path = tmp_path / 'mesh.ply'
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    exports = tmp_path / 'extracted_instances'
    exports.mkdir()
    exported = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices[[2, 0, 1]]),
                                         o3d.utility.Vector3iVector([[0, 1, 2]]))
    o3d.io.write_triangle_mesh(str(exports / 'inst_011_Class_7_inst_1.ply'), exported)
    shifted = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices[[6, 7, 8]] + .001),
                                        o3d.utility.Vector3iVector([[0, 1, 2]]))
    o3d.io.write_triangle_mesh(str(exports / 'inst_012_Class_9_inst_1.ply'), shifted)
    manifest = tmp_path / 'objects_manifest.json'
    manifest.write_text(json.dumps([{'instance_id': 11, 'class_id': 7}, {'instance_id': 12, 'class_id': 9}]))

    recovered = recover_baseline_labels(mesh_path, manifest, exports, [2, 7])
    assert recovered['sources'].tolist() == [7, 2, 2]
    assert recovered['instance_ids'].tolist() == [11, 0, 0]
    assert recovered['provenance']['export_matched_faces'] == 1
    assert recovered['provenance']['palette_fallback_faces'] == 2
    assert recovered['provenance']['unmatched_export_triangles'] == 1
