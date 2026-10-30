import json

import numpy as np
import open3d as o3d

from analyze_new_step import analyze


def test_correspondence_retains_face_provenance_and_reports_missing_instances(tmp_path):
    source = o3d.geometry.TriangleMesh()
    source.vertices = o3d.utility.Vector3dVector([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [3, 0, 0], [4, 0, 0], [3, 1, 0],
    ])
    source.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [3, 4, 5]])
    source.compute_triangle_normals()
    shell = o3d.geometry.TriangleMesh()
    shell.vertices = o3d.utility.Vector3dVector([[0, 0, .1], [1, 0, .1], [0, 1, .1]])
    shell.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
    shell.compute_triangle_normals()

    input_dir, regions_dir, output_dir = (tmp_path / name for name in ("input", "regions", "output"))
    input_dir.mkdir()
    regions_dir.mkdir()
    o3d.io.write_triangle_mesh(str(input_dir / "original_mesh.ply"), source)
    o3d.io.write_triangle_mesh(str(input_dir / "all_structural_regularized.ply"), shell)
    np.save(regions_dir / "triangle_class_labels.npy", np.array([5, 10], dtype=np.int32))
    (regions_dir / "structural_classes.json").write_text(json.dumps({"classifications": {
        "5": {"structural_role": "wall"}, "10": {"structural_role": "object"}
    }}))

    result = analyze(input_dir, regions_dir, output_dir)
    match = np.load(output_dir / "structural_correspondence.npz")
    assert result["roles"]["wall"]["matched"] == 1
    assert match["wall_source_face"].tolist() == [0]
    np.testing.assert_allclose(match["wall_displacement"][0], [0, 0, .1], atol=1e-6)
    assert result["contacts"]["status"] == "unavailable"
