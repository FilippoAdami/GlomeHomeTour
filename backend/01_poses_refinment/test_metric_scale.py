import csv
import json

import numpy as np

from step_colmap import calibrate_metric_scale


def test_colmap_stage_calibrates_scale_before_cloud_processing(tmp_path):
    (tmp_path / "ingestion_results.json").write_text(
        json.dumps({"source_format": "arcore"}), encoding="utf-8")
    transforms = []
    with (tmp_path / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestamp_ns", "tracking", "tx", "ty", "tz"])
        writer.writeheader()
        for index in range(5):
            timestamp = 1000 + index
            writer.writerow({"timestamp_ns": timestamp, "tracking": "TRACKING",
                             "tx": index * 0.25, "ty": 0, "tz": 0})
            transforms.append({"file_path": f"images/frame_{index:03d}.jpg",
                               "timestamp_ns": timestamp,
                               "transform_matrix": np.eye(4).tolist()})
    transforms_path = tmp_path / "transforms.json"
    transforms_path.write_text(json.dumps({"frames": transforms}), encoding="utf-8")

    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.txt").write_text(
        "1 SIMPLE_PINHOLE 640 480 500 320 240\n", encoding="utf-8")
    image_records = []
    for index in range(5):
        center_x = index * 0.5
        image_records.extend((
            f"{index + 1} 1 0 0 0 {-center_x} 0 0 1 frame_{index:03d}.jpg",
            "0 0 -1",
        ))
    images_path = sparse / "images.txt"
    images_path.write_text("\n".join(image_records) + "\n", encoding="utf-8")
    original_model = images_path.read_bytes()

    report = calibrate_metric_scale(
        tmp_path, "arcore", sparse, transforms_path, tmp_path / "calibration")

    assert report["available"]
    assert report["meters_per_colmap_unit"] == 0.5
    assert images_path.read_bytes() == original_model
    assert (tmp_path / "calibration" / "transforms.json").is_file()
