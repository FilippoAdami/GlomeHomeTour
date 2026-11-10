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
        intermediate_svg.write_text("overlay")
        calls["intermediate_svg"] = intermediate_svg
        (output_dir / "floorplan_instance_overlays_manifest.json").write_text(json.dumps({"output_dir": str(output_dir), "floor_levels": [{"output_svg": str(intermediate_svg)}]}))
        return {"output_dir": str(output_dir), "floor_levels": [{
            "floor_level_name": "BASE", "output_svg": str(intermediate_svg),
            "displayed_assigned_floor_count": 0, "displayed_visible_lower_floor_count": 0,
            "phase07_panel_count": 0, "door_passage_count": 0,
        }]}

    def measurements(artifacts_dir, output_dir=None, overlay_manifest=None):
        calls["measurement_input_path"] = overlay_manifest["floor_levels"][0]["output_svg"]
        return {
            "measurement_overlay_count": 0, "wall_segment_annotation_count": 0,
            "floor_levels": [{
                "floor_level_name": "BASE",
                "wall_segment_count": 0,
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
    assert calls["overlay_dir"] == output_root / "floorplan_instance_overlays"
    assert calls["intermediate_svg"].exists()
    published_overlay_manifest = json.loads(
        (output_root / "floorplan_instance_overlays/floorplan_instance_overlays_manifest.json").read_text()
    )
    assert published_overlay_manifest["output_dir"] == str(output_root / "floorplan_instance_overlays")
    assert published_overlay_manifest["floor_levels"][0]["output_svg"] == str(calls["intermediate_svg"])
    assert calls["measurement_input_path"] == str(calls["intermediate_svg"])
    assert list((output_root / "floorplan_instance_overlays").glob("*.svg"))


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

    assert result["measurement_overlay_count"] == 0
    assert (artifacts / "floorplan_measurements/floorplan_measurements_manifest.json").is_file()
    assert not (artifacts / "floorplan_instance_overlays/floorplan_instance_overlays_manifest.json").exists()


def test_measurements_publish_overlay_and_remove_old_composites(tmp_path, monkeypatch):
    from PIL import Image

    artifacts = tmp_path / "artifacts"
    mesh_manifest = artifacts / "floor_level_meshes/floor_level_meshes_manifest.json"
    mesh_manifest.parent.mkdir(parents=True)
    mesh_manifest.write_text(json.dumps({"levels": [{"floor_level_name": "BASE", "regularized_mesh": "mesh.ply", "scans": []}]}))
    source = artifacts / "floorplan_instance_overlays/unmeasured.svg"
    source.parent.mkdir()
    source.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    obsolete = artifacts / "floorplan_measurements/floor_BASE_regularized_instances_with_measures.svg"
    obsolete.parent.mkdir()
    obsolete.write_text("old measured output")
    monkeypatch.setattr(generate_floorplan_measurements, "_render_svg_png", lambda *_: np.zeros((64, 64, 3), np.uint8))
    monkeypatch.setattr(generate_floorplan_measurements, "_read_mesh", lambda *_: (None, None, None))
    monkeypatch.setattr(generate_floorplan_measurements, "_projection_grid", lambda *_: {"pixel_size": 1, "x_min": 0, "z_max": 64})
    monkeypatch.setattr(generate_floorplan_measurements, "_wall_contours", lambda *_: (None, None, None))
    monkeypatch.setattr(generate_floorplan_measurements, "_measurement_clutter_mask", lambda *_: None)
    monkeypatch.setattr(generate_floorplan_measurements, "_wall_segment_measurements", lambda *_: [])
    result = generate_floorplan_measurements.generate_floorplan_measurements(artifacts, overlay_manifest={"floor_levels": [{"floor_level_name": "BASE", "output_svg": str(source), "source_projection": "plan.png", "projection_resolution_px": [64, 64]}]})
    floor = result["floor_levels"][0]
    with Image.open(floor["overlay_png"]) as overlay:
        assert overlay.mode == "RGBA" and overlay.size == (64, 64)
    assert source.read_text().startswith("<svg")
    assert not list((artifacts / "floorplan_measurements").glob("*.svg"))
    assert result["measurement_overlay_count"] == 1


def test_measurement_overlay_contains_rasterized_labels_and_transparency(tmp_path):
    from PIL import Image

    path = tmp_path / "overlay.png"
    measurement = {"label": "1.25 m", "label_center_px": [128, 50],
                   "extension_spans_px": [[[20, 174], [20, 182]], [[220, 174], [220, 182]]],
                   "dimension_span_px": [[20, 180], [220, 180]]}
    generate_floorplan_measurements._write_measurement_overlay_png(path, [measurement], 256)
    rgba = np.asarray(Image.open(path))
    assert rgba.shape == (256, 256, 4)
    assert rgba[0, 0, 3] == 0 and rgba[180, 128, 3] == 255
    label = rgba[35:65, :, :]
    assert np.any((label[:, :, 0] == 255) & (label[:, :, 1] == 113)
                  & (label[:, :, 2] == 0) & (label[:, :, 3] > 0))
    assert np.any((label[:, :, :3] == 255).all(axis=2) & (label[:, :, 3] > 0))
    measurement.update(label_center_px=[32, 18], extension_spans_px=[], dimension_span_px=[[5, 50], [59, 50]])
    generate_floorplan_measurements._write_measurement_overlay_png(path, [measurement], 64)
    small = np.asarray(Image.open(path))
    assert small[:35, 0, 3].max() == small[:35, -1, 3].max() == 0
    assert small[:35, :, 3].max() == 255
