import json
from pathlib import Path
import sys

import numpy as np

STAGE = Path(__file__).resolve().parents[1] / "08_artifacts_generation"
sys.path.insert(0, str(STAGE))

import extract_best_views
import generate_floor_plans
import generate_floorplan_measurements


def test_generate_floor_plans_keeps_macro_outputs_under_named_folder(tmp_path, monkeypatch):
    scene = tmp_path / "scene"
    alignment = scene / "multi_room_alignment"
    output_root = scene / "artifacts" / "Generate Floor Plans"
    calls = {}

    monkeypatch.setattr(generate_floor_plans, "build_floor_level_metadata", lambda _: {"floor_planes": []})

    def floor_meshes(alignment_dir, metadata_path, output_dir):
        calls["mesh_dir"] = output_dir
        return {"levels": []}

    def floor_projections(mesh_dir, output_dir, resolution):
        calls["projection_dir"] = output_dir
        return {"projections": []}

    def classified_slices(mesh_dir, output_dir, resolution):
        calls["slice_dir"] = output_dir
        return {"levels": []}

    def floor_checks(scene_dir, output_dir, artifacts_dir=None):
        calls["floor_check_dir"] = output_dir
        calls["floor_check_artifacts"] = artifacts_dir
        return {"rejected_association_count": 0}

    def room_checks(scene_dir, output_dir, artifacts_dir=None):
        calls["room_check_dir"] = output_dir
        calls["room_check_artifacts"] = artifacts_dir
        return {"associations": [], "floor_checked_instance_count": 0,
                "final_instance_reports_written": 0}

    monkeypatch.setattr(generate_floor_plans, "generate_floor_level_meshes", floor_meshes)
    monkeypatch.setattr(generate_floor_plans, "generate_floor_level_projections", floor_projections)
    monkeypatch.setattr(generate_floor_plans, "generate_classified_object_slices", classified_slices)
    monkeypatch.setattr(generate_floor_plans, "check_instance_floor_distances", floor_checks)
    monkeypatch.setattr(generate_floor_plans, "check_instance_room_consistency", room_checks)
    def overlays(artifacts_dir, output_dir):
        calls["overlay_dir"] = output_dir
        intermediate_svg = output_dir / "floor_BASE_regularized_instances.svg"
        intermediate_svg.parent.mkdir(parents=True, exist_ok=True)
        intermediate_svg.write_text("temporary overlay")
        calls["intermediate_svg"] = intermediate_svg
        return {"output_dir": str(output_dir), "floor_levels": [{
            "floor_level_name": "BASE", "output_svg": str(intermediate_svg),
            "displayed_assigned_floor_count": 0, "displayed_visible_lower_floor_count": 0,
            "phase07_panel_count": 0, "door_passage_count": 0,
        }]}

    def measurements(artifacts_dir, output_dir=None, overlay_manifest=None):
        calls["measurement_input_path"] = overlay_manifest["floor_levels"][0]["output_svg"]
        return {
            "measured_view_count": 0, "measured_output_file_count": 0,
            "floor_levels": [{
                "floor_level_name": "BASE",
                "measured_furnished_floorplan_svg": str(output_root / "floorplan_measurements/final.svg"),
            "wall_segment_count": 0, "measured_view_count": 0,
            }],
            "between_scan_segment_count": 0,
        }

    monkeypatch.setattr(generate_floor_plans, "generate_floorplan_instance_overlays", overlays)
    monkeypatch.setattr(generate_floor_plans, "generate_floorplan_measurements", measurements)

    result = generate_floor_plans.generate_floor_plans(alignment)

    assert (output_root / "generate_floor_plans_manifest.json").is_file()
    assert (output_root / "floor_level_metadata.json").is_file()
    output_path_keys = (
        "floor_level_metadata", "floor_level_mesh_manifest", "floor_level_projection_manifest",
        "classified_object_floor_slices_manifest", "instance_floor_distance_manifest",
        "room_category_consistency_manifest", "final_instance_associations",
        "final_instance_report_dir", "floorplan_instance_overlays_manifest",
        "floorplan_measurements_manifest",
    )
    assert all(Path(result[key]).is_relative_to(output_root) for key in output_path_keys)
    assert all(path == output_root / suffix for path, suffix in (
        (calls["mesh_dir"], "floor_level_meshes"),
        (calls["projection_dir"], "floor_level_projections"),
        (calls["slice_dir"], "classified_object_slice_experiment"),
        (calls["floor_check_dir"], "instance_floor_distance_check"),
        (calls["room_check_dir"], "room_category_consistency"),
        (calls["floor_check_artifacts"], ""),
        (calls["room_check_artifacts"], ""),
    ))
    assert calls["overlay_dir"] != output_root / "floorplan_instance_overlays"
    assert not calls["overlay_dir"].exists()
    assert not calls["intermediate_svg"].exists()
    published_overlay_manifest = json.loads(
        (output_root / "floorplan_instance_overlays/floorplan_instance_overlays_manifest.json").read_text()
    )
    assert published_overlay_manifest["output_dir"] == str(output_root / "floorplan_instance_overlays")
    assert published_overlay_manifest["floor_levels"][0]["output_svg"].endswith("floorplan_measurements/final.svg")
    assert calls["measurement_input_path"] == str(calls["intermediate_svg"])
    assert not list((output_root / "floorplan_instance_overlays").glob("*.svg"))


def test_extract_best_views_default_output_uses_named_folder(tmp_path, monkeypatch):
    alignment = tmp_path / "scene" / "multi_room_alignment"
    transforms = alignment / "stats" / "aligned_scans_transforms.json"
    transforms.parent.mkdir(parents=True)
    transforms.write_text(json.dumps({"scans": {
        "scan_0": {"transform_matrix": np.eye(4).tolist()},
    }}))
    output_root = tmp_path / "scene" / "artifacts" / "Extract Best Views"
    captured = {}

    def extract_scan(scan_dir, alignment_dir, output_dir, transform, device, batch_size):
        captured["output_dir"] = output_dir
        return {"scan_id": scan_dir.name, "status": "complete", "selected_count": 1,
                "candidate_count": 1, "weighted_utility": 0.8, "elapsed_seconds": 0.1}

    monkeypatch.setattr(extract_best_views, "extract_scan", extract_scan)
    manifest = extract_best_views.extract_best_views(alignment, device="cpu")

    assert captured["output_dir"] == output_root
    assert manifest["macro_step_status"] == "complete"
    assert (output_root / "extract_best_views_manifest.json").is_file()


def test_measurements_can_consume_staged_overlay_manifest_without_publishing_it(tmp_path):
    artifacts = tmp_path / "artifacts"
    mesh_manifest = artifacts / "floor_level_meshes/floor_level_meshes_manifest.json"
    mesh_manifest.parent.mkdir(parents=True)
    mesh_manifest.write_text(json.dumps({"levels": []}))

    result = generate_floorplan_measurements.generate_floorplan_measurements(
        artifacts,
        overlay_manifest={"floor_levels": []},
    )

    assert result["measured_output_file_count"] == 0
    assert (artifacts / "floorplan_measurements/floorplan_measurements_manifest.json").is_file()
    assert not (artifacts / "floorplan_instance_overlays/floorplan_instance_overlays_manifest.json").exists()
