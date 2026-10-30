"""ARCore-to-COLMAP relative metric scale calibration."""

import csv
import json
from pathlib import Path

import numpy as np

ARCORE_SOURCE_FORMATS = {"arcore", "arcore_app_certified"}
# shortcut: use only the earliest ten registered matches; extend this only after checking later ARCore drift.
SCALE_REFERENCE_FRAMES = 10
MIN_ARCORE_BASELINE_M = 0.30
MIN_ARCORE_STEP_M = 0.15
MIN_SCALE_PAIRS = 6
MIN_SCALE_STEPS = 4
MIN_STEP_INLIER_FRACTION = 0.60
MAX_STEP_SCALE_DEVIATION = 0.12
MIN_CALIBRATION_SPAN_M = 0.75
MAX_SCALE_RELATIVE_MAD = 0.05


def estimate_metric_scale(workspace: Path, transforms_path: Path | None = None) -> dict:
    """Estimate metres per COLMAP unit from an early, stable ARCore trajectory segment.

    Distance ratios are invariant to the independent world origins, rotations,
    and translations of the ARCore and COLMAP pose tracks. The result is a
    calibration report; callers may apply it uniformly to points and camera
    translations in the COLMAP world.
    """
    workspace = Path(workspace)
    result_path = workspace / "ingestion_results.json"
    if not result_path.is_file():
        return {"available": False, "reason": "missing ingestion_results.json"}
    try:
        source_format = json.loads(result_path.read_text(encoding="utf-8")).get("source_format")
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "reason": f"cannot read ingestion result: {exc}"}
    if source_format not in ARCORE_SOURCE_FORMATS:
        return {"available": False, "reason": "metric calibration requires an ARCore-origin scan",
                "source_format": source_format}

    trajectory_path = workspace / "trajectory.csv"
    if not trajectory_path.is_file():
        return {"available": False, "reason": "missing trajectory.csv", "source_format": source_format}
    if transforms_path is None:
        transforms_path = workspace / "transforms.json"
    if not transforms_path.is_file():
        return {"available": False, "reason": "missing COLMAP transforms.json", "source_format": source_format}

    try:
        trajectory: dict[int, dict] = {}
        duplicate_timestamps: set[int] = set()
        with trajectory_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                timestamp = int(row["timestamp_ns"])
                if timestamp in trajectory:
                    duplicate_timestamps.add(timestamp)
                trajectory[timestamp] = row
        for timestamp in duplicate_timestamps:
            trajectory.pop(timestamp, None)
        frames = json.loads(transforms_path.read_text(encoding="utf-8")).get("frames", [])
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return {"available": False, "reason": f"cannot read pose tracks: {exc}",
                "source_format": source_format}

    matched = []
    for frame in frames:
        try:
            timestamp = int(frame["timestamp_ns"])
            row = trajectory.get(timestamp)
            if row is None or row.get("tracking") != "TRACKING":
                continue
            arcore_center = np.array([float(row[k]) for k in ("tx", "ty", "tz")], dtype=np.float64)
            colmap_center = np.asarray(frame["transform_matrix"], dtype=np.float64)[:3, 3]
            if not np.isfinite(arcore_center).all() or not np.isfinite(colmap_center).all():
                continue
            matched.append((timestamp, Path(frame["file_path"]).name, arcore_center, colmap_center))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    matched.sort(key=lambda item: (item[0], item[1]))
    reference_frames = matched[:SCALE_REFERENCE_FRAMES]
    if len(reference_frames) < 4:
        return {"available": False, "reason": "fewer than four timestamp-matched TRACKING frames",
                "source_format": source_format, "frames_used": [f[1] for f in reference_frames]}

    steps, step_reports = [], []
    for i, (first, second) in enumerate(zip(reference_frames, reference_frames[1:])):
        arcore_distance = float(np.linalg.norm(first[2] - second[2]))
        colmap_distance = float(np.linalg.norm(first[3] - second[3]))
        report = {
            "from_frame": first[1],
            "to_frame": second[1],
            "delta_time_ns": second[0] - first[0],
            "arcore_distance_m": arcore_distance,
            "colmap_distance_units": colmap_distance,
            "meters_per_colmap_unit": None,
            "status": "too_short" if arcore_distance < MIN_ARCORE_STEP_M else "invalid_colmap_step",
        }
        if arcore_distance >= MIN_ARCORE_STEP_M and colmap_distance > 1e-8:
            report["meters_per_colmap_unit"] = arcore_distance / colmap_distance
            report["status"] = "candidate"
            steps.append({"index": i, "arcore_distance": arcore_distance,
                          "ratio": report["meters_per_colmap_unit"], "report": report})
        step_reports.append(report)

    base = {"source_format": source_format,
            "frames_used": [frame[1] for frame in reference_frames],
            "steps": step_reports}
    if len(steps) < MIN_SCALE_STEPS:
        return {"available": False, "reason": "fewer than four usable adjacent pose steps", **base,
                "usable_steps": len(steps)}

    step_logs = np.log([step["ratio"] for step in steps])
    tolerance_log = float(np.log1p(MAX_STEP_SCALE_DEVIATION))
    # Find the largest log-ratio consensus; this catches a local ARCore/COLMAP
    # disagreement even when pairwise baselines across the rest of the path hide it.
    candidates = [np.abs(step_logs - center) <= tolerance_log for center in step_logs]
    best = max(candidates, key=lambda mask: (int(mask.sum()),
                                             float(sum(s["arcore_distance"] for s, keep in zip(steps, mask) if keep))))
    consensus_log = float(np.median(step_logs[best]))
    inliers = np.abs(step_logs - consensus_log) <= tolerance_log
    consensus_log = float(np.median(step_logs[inliers]))
    inliers = np.abs(step_logs - consensus_log) <= tolerance_log
    consensus_scale = float(np.exp(consensus_log))
    inlier_count = int(inliers.sum())
    inlier_fraction = inlier_count / len(steps)
    for step, keep in zip(steps, inliers):
        report = step["report"]
        report["status"] = "consistent" if keep else "inconsistent_step"
        report["deviation_from_step_consensus_pct"] = round((step["ratio"] / consensus_scale - 1.0) * 100.0, 2)

    base.update({"steps_used": inlier_count,
                 "steps_rejected": [step["report"] for step, keep in zip(steps, inliers) if not keep],
                 "step_consensus_meters_per_colmap_unit": consensus_scale,
                 "step_inlier_fraction": inlier_fraction,
                 "step_consensus_tolerance_pct": MAX_STEP_SCALE_DEVIATION * 100.0})
    if inlier_count < MIN_SCALE_STEPS or inlier_fraction < MIN_STEP_INLIER_FRACTION:
        return {"available": False, "reason": "adjacent pose steps do not have a clear scale consensus", **base}

    # A rejected edge separates two locally valid trajectory segments. Never
    # measure a baseline across it: ARCore may have shifted globally while
    # retaining correct metric motion inside each segment.
    rejected_indices = {step["index"] for step, keep in zip(steps, inliers) if not keep}
    segments, start = [], 0
    for index in sorted(rejected_indices):
        segments.append(reference_frames[start:index + 1])
        start = index + 1
    segments.append(reference_frames[start:])
    segments = [segment for segment in segments if len(segment) >= 2]

    ratios, baselines = [], []
    for segment in segments:
        for i in range(len(segment)):
            for j in range(i + 1, len(segment)):
                arcore_distance = float(np.linalg.norm(segment[i][2] - segment[j][2]))
                colmap_distance = float(np.linalg.norm(segment[i][3] - segment[j][3]))
                if arcore_distance >= MIN_ARCORE_BASELINE_M and colmap_distance > 1e-8:
                    ratios.append(arcore_distance / colmap_distance)
                    baselines.append(arcore_distance)

    step_inlier_ratios = np.asarray([step["ratio"] for step, keep in zip(steps, inliers) if keep])
    measured_ratios = np.asarray(ratios) if len(ratios) >= MIN_SCALE_PAIRS else step_inlier_ratios
    scale = float(np.median(measured_ratios))
    relative_mad = float(np.median(np.abs(measured_ratios - scale)) / scale)
    accepted_motion = float(sum(step["arcore_distance"] for step, keep in zip(steps, inliers) if keep))
    segment_span = max(baselines, default=0.0)
    if not np.isfinite(scale) or scale <= 0:
        reason = "estimated scale is not finite and positive"
    elif max(segment_span, accepted_motion) < MIN_CALIBRATION_SPAN_M:
        reason = "consistent early ARCore motion spans less than 0.75 m"
    elif relative_mad > MAX_SCALE_RELATIVE_MAD:
        reason = f"within-segment scale spread is too high ({relative_mad:.1%} median absolute deviation)"
    else:
        reason = None
    for report in step_reports:
        if report["status"] == "candidate":
            report["status"] = "not_used_short_baseline"
    base.update({"usable_pairs": len(ratios),
                 "usable_motion_m": accepted_motion,
                 "segments_used": [[frame[1] for frame in segment] for segment in segments],
                 "relative_mad": relative_mad,
                 "ratio_p10_p90": [float(v) for v in np.percentile(measured_ratios, [10, 90])]})
    if baselines:
        base["baseline_min_m"] = float(min(baselines))
        base["baseline_max_m"] = float(max(baselines))
    if reason is not None:
        return {"available": False, "reason": reason, **base,
                "meters_per_colmap_unit": scale}
    return {
        "available": True,
        "reason": None,
        **base,
        "method": ("median within-segment pairwise distance ratio after adjacent-step filtering"
                   if len(ratios) >= MIN_SCALE_PAIRS else "median consistent adjacent-step ratio"),
        "meters_per_colmap_unit": scale,
    }


