import json
import sys
from pathlib import Path

import numpy as np
import trimesh


ALIGNMENT_DIR = Path(__file__).resolve().parents[1] / "07_alignment"
sys.path.insert(0, str(ALIGNMENT_DIR))

from window_panel_regularization import generate_aligned_window_panel_artifacts


def _window_mesh(extents, center):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def test_window_pipeline_keeps_stages_associations_and_door_outputs_isolated(tmp_path):
    scene_dir = tmp_path / "scene"
    alignment_dir = scene_dir / "multi_room_alignment"
    stats_dir = alignment_dir / "stats"
    stats_dir.mkdir(parents=True)
    scan_dir = scene_dir / "scan_0"
    association_dir = scan_dir / "segmentation" / "non_structural_similarity"
    association_dir.mkdir(parents=True)

    # A/B overlap in projection but have separate planes; C/D supply the all-panel modes.
    meshes = [
        _window_mesh([0.10, 1.00, 1.00], [0.00, 1.00, 0.00]),
        _window_mesh([0.10, 1.40, 1.20], [0.60, 1.10, 0.10]),
        _window_mesh([1.40, 1.60, 0.10], [10.0, 1.00, 4.00]),
        _window_mesh([1.40, 1.60, 0.10], [14.0, 1.00, 4.50]),
        _window_mesh([0.80, 1.00, 1.00], [20.0, 1.00, 0.00]),  # invalid depth
    ]
    records, associations, instance_details = [], [], []
    for i, mesh in enumerate(meshes, start=1):
        source = scan_dir / "segmentation" / "3d_masks" / "class_010_object" / f"instance_{i:03d}.ply"
        aligned = alignment_dir / "objects" / "scan_0" / "class_010_object" / source.name
        source.parent.mkdir(parents=True, exist_ok=True)
        aligned.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(source)
        mesh.export(aligned)
        records.append({
            "scan_id": "scan_0", "class_id": 10,
            "source_instance_path": str(source), "global_instance_path": str(aligned),
        })
        associations.append([10, i - 1, 2])
        instance_details.append({
            "source_class_id": 10, "instance_id": i - 1,
            "status": "accepted", "association_source": "instance",
        })

    (association_dir / "valid_instance_associations.json").write_text(json.dumps({
        "associations": associations, "instances": instance_details,
    }))
    (stats_dir / "aligned_objects_manifest.json").write_text(json.dumps({"objects": records}))

    # These Phase 07 door/scene assets must remain untouched by the detached window run.
    door_ply = alignment_dir / "aligned_door_panels.ply"
    door_metadata = stats_dir / "aligned_door_panels.json"
    door_ply.write_bytes(b"door-ply-sentinel")
    door_metadata.write_text("door-metadata-sentinel")

    panel_path, metadata_path, panel_count = generate_aligned_window_panel_artifacts(scene_dir, alignment_dir)
    metadata = json.loads(metadata_path.read_text())
    report = json.loads((stats_dir / "window_panel_regularization_report.json").read_text())
    final = {panel["panel_index"]: panel for panel in metadata["panels"]}
    overlap = report["stages"]["final_overlap"]

    assert panel_count == 4
    assert report["stages"]["candidate_filter"]["candidate_count"] == 5
    assert report["stages"]["candidate_filter"]["dimension_valid_count"] == 4
    assert len(final[0]["source_instances"]) == 1
    assert final[0]["panel_dimensions_m"] == {"width": 1.4, "height": 1.6, "depth": 0.1}
    assert final[1]["panel_dimensions_m"] == final[0]["panel_dimensions_m"]
    pair = next(row for row in overlap["pair_checks"] if row["panel_indices"] == [0, 1])
    assert pair["eligible"] and pair["overlap_fraction_before"] >= 0.60
    assert np.isclose(pair["center_plane_separation_m"], 0.60)
    assert np.isclose(final[0]["center_plane_offset_after_m"], 0.00)
    assert np.isclose(final[1]["center_plane_offset_after_m"], 0.60)
    assert panel_path.is_file()
    assert (alignment_dir / "window_panel_pipeline/05_overlap_regularized_panels.ply").is_file()
    assert (alignment_dir / "window_panel_pipeline/00_semantic_window_instances.ply").is_file()
    assert door_ply.read_bytes() == b"door-ply-sentinel"
    assert door_metadata.read_text() == "door-metadata-sentinel"
