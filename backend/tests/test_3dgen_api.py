import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "07_3DGen"))
from image_to_3d import _reserve_output_dir, _write_ply, generate_3d


def test_numbered_output_directory_skips_existing_paths(tmp_path):
    root = tmp_path / "outputs"
    (root / "1").mkdir(parents=True)
    (root / "2").write_text("reserved")

    assert _reserve_output_dir(root) == root / "3"


def test_numbered_output_directory_uses_maximum_after_a_gap(tmp_path):
    root = tmp_path / "outputs"
    (root / "1").mkdir(parents=True)
    (root / "4").mkdir()

    assert _reserve_output_dir(root) == root / "5"


def test_ply_export_drops_faces_touching_nonfinite_vertices(tmp_path):
    source = SimpleNamespace(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [np.nan, 0, 0]], dtype=np.float32),
        faces=np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64),
    )

    output = _write_ply(source, tmp_path / "mesh.ply")
    mesh = trimesh.load(output, process=False)
    assert mesh.vertices.shape == (3, 3)
    assert mesh.faces.tolist() == [[0, 1, 2]]


@pytest.mark.parametrize(
    ("images", "steps", "model", "error"),
    [([], 1, "trellis", ValueError), (["x.png"], 0, "trellis", ValueError),
     (["x.png"], 1, "other", ValueError), ("x.png", 1, "trellis", TypeError)],
)
def test_generation_rejects_invalid_requests_before_loading_models(images, steps, model, error):
    with pytest.raises(error):
        generate_3d(images, steps=steps, model=model)
