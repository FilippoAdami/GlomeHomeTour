import json
from pathlib import Path
import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import trimesh


ALIGNMENT_DIR = Path(__file__).resolve().parents[1] / "07_alignment"
sys.path.insert(0, str(ALIGNMENT_DIR))

from scan_loader import ScanAssetPaths, ScanDataset
from multi_asset_synchronizer import MultiAssetSynchronizer


def _make_scan(tmp_path):
    scan_dir = tmp_path / "scan_1"
    segmentation_dir = scan_dir / "segmentation"
    masks_dir = segmentation_dir / "3d_masks"
    object_dir = masks_dir / "class_005_object"
    wall_dir = masks_dir / "class_005_wall"
    canonical_dir = segmentation_dir / "Instances PLYs" / "Non-Structural" / "5"
    object_dir.mkdir(parents=True)
    wall_dir.mkdir(parents=True)
    canonical_dir.mkdir(parents=True)

    for instance_index in range(1, 6):
        mesh = (
            trimesh.creation.box(extents=[1.3, 2.0, 1.2])
            if instance_index == 1
            else trimesh.creation.box(extents=[0.7, 2.0, 0.2])
        )
        mesh.visual.vertex_colors = np.tile([120, 140, 160, 255], (len(mesh.vertices), 1))
        source = canonical_dir / f"instance_{instance_index:03d}.ply"
        mesh.export(source)
        (object_dir / source.name).symlink_to(Path("../../Instances PLYs/Non-Structural/5") / source.name)
    wall_path = wall_dir / "instance_001.ply"
    trimesh.creation.box(extents=[2.0, 2.5, 0.1]).export(wall_path)

    copies_path = segmentation_dir / "fixed_class_object_copies.json"
    copies_path.write_text(json.dumps({"schema_version": 1, "copies": [{
        "class_id": 5,
        "category_id": 1,
        "category": "door",
        "source_role": "wall",
        "source_directory": "3d_masks/class_005_wall",
        "object_directory": "3d_masks/class_005_object",
        "copied_instances": [f"Instances PLYs/Non-Structural/5/instance_{i:03d}.ply" for i in range(1, 6)],
    }]}))
    association_path = segmentation_dir / "non_structural_similarity" / "valid_instance_associations.json"
    association_path.parent.mkdir()
    association_path.write_text(json.dumps({
        "tuple_fields": ["source_class_id", "instance_id", "text_class_id"],
        "associations": [[5, 0, 1]],
        "instances": [{"source_class_id": 5, "instance_id": 0, "status": "accepted", "text_class_id": 1}],
    }))
    manifest_path = segmentation_dir / "segmentation.json"
    manifest_path.write_text(json.dumps({"total_instances": 1, "instances": [{"instance_key": "inst_5_0"}]}))
    return scan_dir, ScanAssetPaths(
        scan_id="scan_1",
        scan_dir=scan_dir,
        door_instance_metadata_path=association_path,
    )


def test_oversized_door_split_replaces_object_ply_and_updates_scene_metadata(tmp_path):
    scan_dir, paths = _make_scan(tmp_path)
    wall_path = scan_dir / "segmentation/3d_masks/class_005_wall/instance_001.ply"
    wall_hash = hashlib.sha256(wall_path.read_bytes()).hexdigest()
    doors = ScanDataset._extract_door_instances(paths, [5], trimesh.creation.box())

    assert len(doors) == 3
    assert [door["normal"].tolist() for door in doors] == [
        [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
    ]
    object_dir = scan_dir / "segmentation/3d_masks/class_005_object"
    assert all((object_dir / f"instance_{index:03d}.ply").is_file() for index in (1, 6, 7))
    assert hashlib.sha256(wall_path.read_bytes()).hexdigest() == wall_hash
    child = trimesh.load(object_dir / "instance_006.ply", process=False)
    np.testing.assert_array_equal(child.visual.vertex_colors, [[120, 140, 160, 255]] * len(child.vertices))

    matches = json.loads(paths.door_instance_metadata_path.read_text())
    assert matches["associations"] == [[5, 0, 1], [5, 5, 1], [5, 6, 1]]
    assert [item["instance_id"] for item in matches["instances"]] == [0, 5, 6]
    assert all(item["classification_source"] == "phase07_door_split" for item in matches["instances"])

    scene_metadata = json.loads((scan_dir / "segmentation/segmentation.json").read_text())
    assert [item["instance_id"] for item in scene_metadata["door_instances"]] == [0, 5, 6]
    assert all(item["category"] == "door" for item in scene_metadata["door_instances"])

    rerun_doors = ScanDataset._extract_door_instances(paths, [5], trimesh.creation.box())
    assert len(rerun_doors) == 3
    assert [door["normal"].tolist() for door in rerun_doors] == [door["normal"].tolist() for door in doors]

    output_dir = tmp_path / "multi_room_alignment"
    (output_dir / "stats").mkdir(parents=True)
    synchronizer = MultiAssetSynchronizer(output_dir)
    synchronizer._assemble_objects(
        {"scan_1": SimpleNamespace(info=SimpleNamespace(scan_dir=scan_dir))},
        {"scan_1": np.eye(4)},
        output_dir / "objects",
        output_dir / "aligned_objects.ply",
        output_dir / "stats/aligned_objects_manifest.json",
    )
    aligned = json.loads((output_dir / "stats/aligned_objects_manifest.json").read_text())
    door_rows = [row for row in aligned["objects"] if row.get("category") == "door"]
    assert [Path(row["source_instance_path"]).name for row in door_rows] == [
        "instance_001.ply", "instance_006.ply", "instance_007.ply"
    ]
    assert all(row["text_class_id"] == 1 for row in door_rows)


def test_phase07_uses_per_instance_door_matches(tmp_path):
    scan_dir, paths = _make_scan(tmp_path)
    class_path = scan_dir / "segmentation/non_structural_similarity/valid_class_associations.json"
    class_path.parent.mkdir(parents=True, exist_ok=True)
    class_path.write_text(json.dumps({"associations": [[5, 1], [8, 2]]}))
    paths.door_class_metadata_path = class_path

    assert ScanDataset._extract_door_classes(paths) == [5]


def test_empty_phase05_matches_fall_back_to_scan_zero_door_mask(tmp_path):
    scan_dir = tmp_path / "scan_0"
    segmentation_dir = scan_dir / "segmentation"
    object_dir = segmentation_dir / "3d_masks/class_022_object"
    object_dir.mkdir(parents=True)
    trimesh.creation.box(extents=[0.4, 2.2, 1.0]).export(object_dir / "instance_001.ply")

    similarity_dir = segmentation_dir / "non_structural_similarity"
    similarity_dir.mkdir()
    class_path = similarity_dir / "valid_class_associations.json"
    class_path.write_text(json.dumps({"associations": []}))
    instance_path = similarity_dir / "valid_instance_associations.json"
    instance_path.write_text(json.dumps({"associations": [], "instances": []}))
    manifest_path = segmentation_dir / "segmentation.json"
    manifest_path.write_text(json.dumps({"instances": []}))
    paths = ScanAssetPaths(
        scan_id="scan_0",
        scan_dir=scan_dir,
        door_class_metadata_path=class_path,
        door_instance_metadata_path=instance_path,
    )

    door_classes = ScanDataset._extract_door_classes(paths)
    doors = ScanDataset._extract_door_instances(paths, door_classes, trimesh.creation.box())

    assert door_classes == [22]
    assert len(doors) == 1 and doors[0]["source_instance_id"] == 0
    assert json.loads(manifest_path.read_text())["door_instances"][0]["source_class_id"] == 22


def test_phase07_excludes_instances_narrower_than_60cm_from_scene_door_list(tmp_path):
    scan_dir = tmp_path / "scan_1"
    segmentation_dir = scan_dir / "segmentation"
    object_dir = segmentation_dir / "3d_masks/class_005_object"
    object_dir.mkdir(parents=True)
    for index, width in enumerate((0.59, 0.60), start=1):
        trimesh.creation.box(extents=[width, 2.0, 0.3]).export(
            object_dir / f"instance_{index:03d}.ply"
        )

    similarity_dir = segmentation_dir / "non_structural_similarity"
    similarity_dir.mkdir()
    instance_path = similarity_dir / "valid_instance_associations.json"
    instance_path.write_text(json.dumps({
        "associations": [[5, 0, 1], [5, 1, 1]],
        "instances": [
            {"source_class_id": 5, "instance_id": 0, "text_class_id": 1},
            {"source_class_id": 5, "instance_id": 1, "text_class_id": 1},
        ],
    }))
    manifest_path = segmentation_dir / "segmentation.json"
    manifest_path.write_text(json.dumps({"instances": []}))
    paths = ScanAssetPaths(
        scan_id="scan_1",
        scan_dir=scan_dir,
        door_instance_metadata_path=instance_path,
    )

    doors = ScanDataset._extract_door_instances(paths, [5], trimesh.creation.box())

    assert [door["source_instance_id"] for door in doors] == [1]
    manifest_doors = json.loads(manifest_path.read_text())["door_instances"]
    assert [(door["source_class_id"], door["instance_id"]) for door in manifest_doors] == [(5, 1)]


def test_triangle_label_fallback_also_rejects_doors_narrower_than_60cm(tmp_path):
    mesh = trimesh.creation.box(extents=[0.59, 2.0, 0.3])
    labels_path = tmp_path / "triangle_labels.npy"
    np.save(labels_path, np.full(len(mesh.faces), 5, dtype=np.int32))
    paths = ScanAssetPaths(
        scan_id="scan_3",
        scan_dir=tmp_path / "scan_3",
        triangle_labels_path=labels_path,
    )

    doors = ScanDataset._extract_door_instances(paths, [5], mesh)

    assert doors == []


def test_aligned_panels_cover_all_doors_at_final_segment_dimensions(tmp_path):
    from door_plane_regularization import generate_aligned_door_panel_artifacts

    scan_dir = tmp_path / "scan_0"
    source_dir = scan_dir / "segmentation/Instances PLYs/Non-Structural/5"
    source_dir.mkdir(parents=True)
    trimesh.creation.box(extents=[5.0, 4.0, 5.0]).export(scan_dir / "mesh.ply")

    alignment_dir = tmp_path / "multi_room_alignment"
    stats_dir = alignment_dir / "stats"
    aligned_dir = alignment_dir / "objects/scan_0/class_005_object"
    stats_dir.mkdir(parents=True)
    aligned_dir.mkdir(parents=True)
    records = []
    for instance_index, center, height in (
        (1, [-1.0, 1.0, 0.0], 2.0),
        (2, [1.0, 1.1, 0.0], 2.2),
    ):
        source = source_dir / f"instance_{instance_index:03d}.ply"
        aligned = aligned_dir / source.name
        door = trimesh.creation.box(extents=[0.9, height, 0.1])
        door.apply_translation(center)
        if instance_index == 1:
            door.vertices = np.vstack((door.vertices, [center[0], center[1] + 3.0, center[2]]))
        door.export(source)
        door.export(aligned)
        records.append({
            "scan_id": "scan_0",
            "class_id": 5,
            "category": "door",
            "text_class_id": 1,
            "source_instance_path": str(source),
            "global_instance_path": str(aligned),
        })
    non_door = trimesh.creation.box(extents=[0.9, 2.0, 0.1])
    non_door_path = aligned_dir / "instance_003.ply"
    non_door.export(non_door_path)
    records.append({
        "scan_id": "scan_0",
        "class_id": 5,
        "category": "object",
        "text_class_id": 11,
        "source_instance_path": str(source_dir / "instance_003.ply"),
        "global_instance_path": str(non_door_path),
    })
    (stats_dir / "aligned_objects_manifest.json").write_text(json.dumps({"objects": records}))
    floor_dir = alignment_dir / "scans/scan_0/mesh_elaboration/elements"
    floor_dir.mkdir(parents=True)
    trimesh.Trimesh(
        vertices=[[-3, 0.1, -3], [3, 0.1, -3], [3, 0.1, 3], [-3, 0.1, 3]],
        faces=[[0, 1, 2], [0, 2, 3]], process=False,
    ).export(floor_dir / "floor_level_000_area_000.ply")

    panel_path, metadata_path, panel_count = generate_aligned_door_panel_artifacts(
        tmp_path, alignment_dir, {"scan_0": np.eye(4)}
    )

    panels = json.loads(metadata_path.read_text())["panels"]
    combined = trimesh.load(panel_path, force="mesh", process=False)
    report = json.loads((stats_dir / "door_panel_regularization_report.json").read_text())
    assert panel_count == len(panels) == 2
    assert [row["instance_id"] for row in panels] == [0, 1]
    assert [row["panel_dimensions_m"] for row in panels] == [
        {"width": 1.0, "height": 2.2, "depth": 0.1},
        {"width": 1.0, "height": 2.2, "depth": 0.1},
    ]
    assert all(np.isclose(row["floor_elevation_m"], 0.1) for row in panels)
    assert len(report["plane_groups"]) == 1
    assert report["dimension_clusters"]["height"] == [{"dimension_m": 2.2, "panel_indices": [0, 1]}]
    assert len(combined.faces) == 24


def test_panel_export_merges_overlapping_duplicates_but_keeps_stacked_sources_separate(tmp_path):
    from door_plane_regularization import generate_aligned_door_panel_artifacts

    scan_dir = tmp_path / "scan_0"
    scan_dir.mkdir()
    trimesh.creation.box(extents=[8.0, 4.0, 8.0]).apply_translation([0, 2, 0]).export(scan_dir / "mesh.ply")
    alignment_dir = tmp_path / "multi_room_alignment"
    stats_dir = alignment_dir / "stats"
    aligned_dir = alignment_dir / "objects/scan_0/class_005_object"
    source_dir = scan_dir / "segmentation/Instances PLYs/Non-Structural/5"
    stats_dir.mkdir(parents=True)
    aligned_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    records = []
    for instance_index, width, height, center in (
        (1, 0.9, 2.0, [-0.05, 1.1, 0.0]),
        (2, 1.2, 2.1, [0.05, 1.15, -0.32]),
        (3, 0.9, 2.0, [0.0, 3.1, 0.0]),
    ):
        source = source_dir / f"instance_{instance_index:03d}.ply"
        aligned = aligned_dir / source.name
        mesh = trimesh.creation.box(extents=[width, height, 0.1])
        mesh.apply_translation(center)
        mesh.export(source)
        mesh.export(aligned)
        records.append({
            "scan_id": "scan_0", "class_id": 5, "category": "door", "text_class_id": 1,
            "source_instance_path": str(source), "global_instance_path": str(aligned),
        })
    (stats_dir / "aligned_objects_manifest.json").write_text(json.dumps({"objects": records}))
    floor_dir = alignment_dir / "scans/scan_0/mesh_elaboration/elements"
    floor_dir.mkdir(parents=True)
    trimesh.Trimesh(
        vertices=[[-4, 0.1, -4], [4, 0.1, -4], [4, 0.1, 4], [-4, 0.1, 4]],
        faces=[[0, 1, 2], [0, 2, 3]], process=False,
    ).export(floor_dir / "floor_level_000_area_000.ply")

    _, metadata_path, count = generate_aligned_door_panel_artifacts(
        tmp_path, alignment_dir, {"scan_0": np.eye(4)}
    )

    panels = json.loads(metadata_path.read_text())["panels"]
    report = json.loads((stats_dir / "door_panel_regularization_report.json").read_text())
    assert count == len(panels) == 2
    assert sorted(len(group["source_instances"]) for group in report["merge_groups"]) == [1, 2]
    assert sorted(len(panel["source_instances"]) for panel in panels) == [1, 2]
    merged_panel = next(panel for panel in panels if len(panel["source_instances"]) == 2)
    stacked_panel = next(panel for panel in panels if len(panel["source_instances"]) == 1)
    assert merged_panel["panel_dimensions_m"] == {"width": 1.0, "height": 2.0, "depth": 0.1}
    assert stacked_panel["panel_dimensions_m"] == {"width": 1.0, "height": 2.0, "depth": 0.1}
    assert all(np.isclose(panel["floor_elevation_m"], 0.1) for panel in panels)
    assert all(row["panel_dimensions_m"]["depth"] == 0.1 for row in panels)
    assert all(
        abs(row["panel_dimensions_m"]["width"] - row["segment_length_m"]) < 1e-9
        for row in panels
    )
    assert merged_panel["overlap_regularization"]["group_id"] == stacked_panel["overlap_regularization"]["group_id"]
    spanning_pair = next(
        pair for pair in report["pair_checks"]
        if pair["source_a"].endswith("instance_001") and pair["source_b"].endswith("instance_002")
    )
    assert np.isclose(spanning_pair["center_plane_distance_m"], 0.32)
    assert np.isclose(spanning_pair["face_to_face_distance_m"], 0.22)
    assert spanning_pair["same_merge_group"]
    assert stacked_panel["plane_group_id"] != merged_panel["plane_group_id"]
    assert all(np.isclose(row["panel_dimensions_m"]["height"] / 0.20, round(row["panel_dimensions_m"]["height"] / 0.20)) for row in panels)


def test_final_overlap_regularization_uses_global_dimension_modes_and_preserves_plane_gaps(tmp_path):
    from door_plane_regularization import generate_aligned_door_panel_artifacts

    scan_dir = tmp_path / "scan_0"
    scan_dir.mkdir()
    trimesh.creation.box(extents=[20.0, 8.0, 20.0]).apply_translation([0.0, 3.0, 0.0]).export(
        scan_dir / "mesh.ply"
    )
    alignment_dir = tmp_path / "multi_room_alignment"
    stats_dir = alignment_dir / "stats"
    aligned_dir = alignment_dir / "objects/scan_0/class_005_object"
    source_dir = scan_dir / "segmentation/Instances PLYs/Non-Structural/5"
    stats_dir.mkdir(parents=True)
    aligned_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    records = []
    cases = (
        (1, [1.2, 2.0, 0.1], [0.0, 1.1, 0.0]),
        (2, [0.8, 2.0, 0.1], [0.0, 1.1, 2.0]),
        (3, [0.1, 2.4, 1.2], [-3.0, 1.3, -4.0]),
        (4, [0.1, 2.4, 1.2], [0.0, 1.3, 0.0]),
        (5, [0.1, 2.4, 1.2], [3.0, 1.3, 4.0]),
    )
    for instance_index, extents, center in cases:
        source = source_dir / f"instance_{instance_index:03d}.ply"
        aligned = aligned_dir / source.name
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(center)
        mesh.export(source)
        mesh.export(aligned)
        records.append({
            "scan_id": "scan_0", "class_id": 5, "category": "door", "text_class_id": 1,
            "source_instance_path": str(source), "global_instance_path": str(aligned),
        })
    (stats_dir / "aligned_objects_manifest.json").write_text(json.dumps({"objects": records}))
    floor_dir = alignment_dir / "scans/scan_0/mesh_elaboration/elements"
    floor_dir.mkdir(parents=True)
    trimesh.Trimesh(
        vertices=[[-10, 0.1, -10], [10, 0.1, -10], [10, 0.1, 10], [-10, 0.1, 10]],
        faces=[[0, 1, 2], [0, 2, 3]], process=False,
    ).export(floor_dir / "floor_level_000_area_000.ply")

    _, metadata_path, count = generate_aligned_door_panel_artifacts(
        tmp_path, alignment_dir, {"scan_0": np.eye(4)}
    )

    metadata = json.loads(metadata_path.read_text())
    panels = {panel["source_instances"][0]["instance_index"]: panel for panel in metadata["panels"]}
    report = json.loads((stats_dir / "door_panel_regularization_report.json").read_text())
    overlap = report["overlap_regularization"]
    assert count == len(panels) == 5
    assert overlap["global_dimension_modes_m"]["width"]["mode_m"] == 1.2
    assert overlap["global_dimension_modes_m"]["height"]["mode_m"] == 2.4
    assert sum(item["panel_count"] for item in overlap["global_dimension_modes_m"]["height"]["distribution"]) == 5

    first, second = panels[1], panels[2]
    assert first["overlap_regularization"]["group_id"] == second["overlap_regularization"]["group_id"]
    assert first["panel_dimensions_m"] == second["panel_dimensions_m"] == {
        "width": 1.2, "height": 2.4, "depth": 0.1,
    }
    assert np.allclose(first["panel_center_xyz_m"][:2], second["panel_center_xyz_m"][:2])
    assert np.isclose(first["plane_offset_after_m"], 0.0)
    assert np.isclose(second["plane_offset_after_m"], -2.0)
    pair = next(
        item for item in overlap["pair_checks"]
        if {item["panel_index_a"], item["panel_index_b"]} == {0, 1}
    )
    assert pair["qualifies"] and np.isclose(pair["projected_overlap_after_fraction"], 1.0)
    assert pair["center_plane_distance_unchanged"]
    assert np.isclose(pair["center_plane_distance_before_m"], 2.0)


def test_floor_sampling_preserves_the_nearest_of_overlapping_floor_levels(tmp_path):
    from door_plane_regularization import _sample_aligned_floor

    floors = []
    for level, elevation in enumerate((0.1, 2.8)):
        floor = trimesh.Trimesh(
            vertices=[[-2, elevation, -2], [2, elevation, -2], [2, elevation, 2], [-2, elevation, 2]],
            faces=[[0, 1, 2], [0, 2, 3]], process=False,
        )
        floors.append((tmp_path / f"floor_level_{level}.ply", floor))

    lower = _sample_aligned_floor(floors, 0.0, 0.0, 0.1)
    upper = _sample_aligned_floor(floors, 0.0, 0.0, 2.8)

    assert np.isclose(lower["elevation_m"], 0.1)
    assert np.isclose(upper["elevation_m"], 2.8)


def test_segment_proximity_fit_keeps_minimum_door_width_and_reports_conflicts(tmp_path):
    from door_plane_regularization import fit_axis_aligned_door_segments

    doors = [
        {
            "scan": "scan_0",
            "description": name,
            "orientation": "Vertical",
            "wall_coord": wall_x,
            "geom_downproj_min": 0.0,
            "geom_downproj_max": 1.0,
            "regularized_width_m": 1.0,
        }
        for name, wall_x in (("west", 0.0), ("east", 0.05))
    ]

    fit_axis_aligned_door_segments(doors, {"scan_0": np.eye(4)}, tmp_path)

    assert all(door["final_span_m"] >= 0.60 for door in doors)
    assert all(door["proximity_conflicts"] for door in doors)


def test_segment_perimeter_uses_unaligned_mesh_before_applying_scan_transform(tmp_path):
    from door_plane_regularization import fit_axis_aligned_door_segments

    scan_dir = tmp_path / "scan_1"
    raw_dir = scan_dir / ".raw_unaligned"
    raw_dir.mkdir(parents=True)
    transform = np.eye(4)
    transform[2, 3] = -10.0
    raw_mesh = trimesh.creation.box(extents=[4.0, 3.0, 4.0])
    raw_mesh.export(raw_dir / "mesh.ply")
    aligned_mesh = raw_mesh.copy()
    aligned_mesh.apply_transform(transform)
    aligned_mesh.export(scan_dir / "mesh.ply")
    doors = [{
        "scan": "scan_1",
        "description": "door",
        "orientation": "Vertical",
        "wall_coord": 0.0,
        "geom_downproj_min": 9.6,
        "geom_downproj_max": 10.4,
        "regularized_width_m": 0.8,
    }]

    fit_axis_aligned_door_segments(doors, {"scan_1": transform}, tmp_path)

    assert doors[0]["final_span_m"] == 0.8
    assert doors[0]["perimeter_clipping"] == []
