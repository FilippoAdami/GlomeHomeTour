import hashlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

STAGE = Path(__file__).resolve().parents[1] / "05_semantic_segmentation"
if str(STAGE) not in sys.path:
    sys.path.insert(0, str(STAGE))

import step_segmentation


def _create_fake_scan(root: Path) -> Path:
    (root / "images").mkdir(parents=True)
    sparse = root / "sparse/0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.txt").write_text("1 PINHOLE 10 10 5 5 5 5\n")
    (sparse / "images.txt").write_text("1 1 0 0 0 0 0 0 1 frame.jpg\n0 1 -1 1 2 3\n")
    (root / "mesh.ply").write_text("ply\nformat ascii 1.0\nelement vertex 0\nelement face 0\nend_header\n")
    (root / "mesh.json").write_text("{}\n")
    return root


def test_step_segmentation_multi_scan_discovery_and_cli(tmp_path):
    scan_0 = _create_fake_scan(tmp_path / "scan_0")
    scan_1 = _create_fake_scan(tmp_path / "scan_1")

    # Verify discovery finds both scans in order
    scans = step_segmentation.scene_inputs.discover_scans(tmp_path)
    assert len(scans) == 2
    assert scans[0].name == "scan_0"
    assert scans[1].name == "scan_1"

    # Test main with mocked scan runner
    with patch("step_segmentation.run_segmentation_pipeline_for_scan") as mock_run:
        mock_run.return_value = {"num_classes": 2, "total_instances": 3}
        ret = step_segmentation.main(["--workspace", str(tmp_path), "--max-frames", "5"])
        assert ret == 0
        assert mock_run.call_count == 2
        assert step_segmentation.MIN_INSTANCE_TRIANGLES == 2000
        for call in mock_run.call_args_list:
            scan, args, _ = call.args
            assert scan in (scan_0, scan_1)
            assert not hasattr(args, 'output_dir')
            assert not hasattr(args, 'min_instance_size')


@pytest.mark.parametrize('option,value', [('--min-instance-size', '1000'), ('--output-dir', '/tmp/segmentation_min_1000')])
def test_locked_production_cli_rejects_experiment_overrides(option, value):
    with patch('step_segmentation.run_segmentation_pipeline_for_scan') as runner:
        with pytest.raises(SystemExit) as exc:
            step_segmentation.main([option, value])
        assert exc.value.code == 2
        runner.assert_not_called()


def test_final_category_classification_uses_stage05_delivery(tmp_path, monkeypatch):
    scan = tmp_path / "scan_0"
    stage_out = scan / "segmentation"
    stage_out.mkdir(parents=True)
    calls = {}

    def render(argv):
        calls["render"] = argv
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "non_structural_frames_manifest.json").write_text(
            json.dumps({"total_instances_extracted": 0})
        )
        return 1

    def classify(argv):
        calls["classify"] = argv
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "valid_class_associations.json").write_text('{"associations": []}')
        return []

    monkeypatch.setitem(sys.modules, "extract_non_structural_frames", SimpleNamespace(main=render))
    monkeypatch.setitem(sys.modules, "run_non_structural_category_similarity", SimpleNamespace(main=classify))

    assert step_segmentation.run_final_category_classification(scan, scan / "mesh.ply", stage_out) == []
    assert str(stage_out / "triangle_class_labels.npy") in calls["render"]
    assert str(stage_out / "class_metadata.json") in calls["render"]
    assert str(stage_out / "non_structural_similarity") in calls["classify"]


def test_only_category_matching_requires_current_delivery_and_skips_segmentation(tmp_path, monkeypatch):
    scan = _create_fake_scan(tmp_path / "scan_0")
    stage_out = scan / "segmentation"
    stage_out.mkdir()
    (stage_out / "mesh_segmented.ply").write_text("delivery")
    (stage_out / "class_metadata.json").write_text("{}")
    input_paths = (scan / "mesh.ply", scan / "sparse/0/cameras.txt", scan / "sparse/0/images.txt")
    hashes = {
        str(path): hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        for path in input_paths
    }
    (stage_out / "segmentation.json").write_text(json.dumps({
        "algorithm_version": step_segmentation.ALGORITHM_VERSION,
        "input_hashes": hashes,
        "settings": {
            "colmap_threshold": 100,
            "max_frames": None,
            "min_instance_size": step_segmentation.MIN_INSTANCE_TRIANGLES,
        },
    }))
    classified = []
    monkeypatch.setattr(step_segmentation, "run_final_category_classification",
                        lambda *args: classified.append(args) or [(1, 2)])
    monkeypatch.setattr(step_segmentation, "run_segmentation_pipeline_for_scan",
                        lambda *args: pytest.fail("full segmentation must not run"))

    assert step_segmentation.main(["--workspace", str(scan), "--only-category-matching"]) == 0
    assert len(classified) == 1
    assert classified[0][0] == scan
    assert classified[0][2] == stage_out
