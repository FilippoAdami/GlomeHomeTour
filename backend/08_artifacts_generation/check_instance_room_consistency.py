#!/usr/bin/env python3
"""Apply WC-anchored room/category consistency to Phase 08 floor-checked instances."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from check_instance_floor_distance import _write_report


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_DIR = BACKEND_DIR / "current_scene"
DEFAULT_OUTPUT_DIR = DEFAULT_SCENE_DIR / "artifacts" / "room_category_consistency"
WC_CATEGORY_ID = 24
BATHROOM_ONLY_CATEGORY_IDS = {24, 25, 26}  # WC, bidet, shower
BATHROOM_INCOMPATIBLE_CATEGORY_IDS = {16, 18, 19, 20, 21, 22}
SINGLETON_CATEGORY_IDS = {20, 21, 22, 24, 25, 26, 27}  # refrigerator, oven, dishwasher, WC, bidet, shower, washer
DOOR_WINDOW_CATEGORY_IDS = {1, 2}
ASSOCIATION_MARGIN = 0.02
MIN_CLOSE_CANDIDATES = 5


def _key(record: dict) -> tuple[str, int, int]:
    return record["scan_id"], int(record["source_class_id"]), int(record["instance_id"])


def _instance_prediction(instance: dict) -> int | None:
    if instance.get("status") != "accepted":
        return None
    category_id = instance.get("instance_text_class_id")
    if category_id is None and instance.get("association_source") == "instance":
        category_id = instance.get("text_class_id")
    return int(category_id) if category_id is not None else None


def _category_conflicts(category_id: int, room_type: str) -> bool:
    if room_type == "bathroom":
        return category_id in BATHROOM_INCOMPATIBLE_CATEGORY_IDS
    return category_id in BATHROOM_ONLY_CATEGORY_IDS


def _score(scores: dict, category_id: int) -> float:
    return float(scores.get(str(category_id), scores.get(category_id, 0.0)))


def _category_owners(rows: list[dict], instances_by_key: dict) -> dict[str, dict[int, int]]:
    claims = {}
    for row in rows:
        category_id = row.get("category_id")
        if category_id is None or int(category_id) not in SINGLETON_CATEGORY_IDS:
            continue
        scan_id, source_class_id, instance_id = _key(row)
        instance = instances_by_key.get((scan_id, source_class_id, instance_id), {})
        score = _score(instance.get("category_mean_specificity", {}), int(category_id))
        class_scores = claims.setdefault(scan_id, {}).setdefault(int(category_id), {})
        class_scores[source_class_id] = max(class_scores.get(source_class_id, float("-inf")), score)
    return {
        scan_id: {
            category_id: min(class_scores, key=lambda class_id: (-class_scores[class_id], class_id))
            for category_id, class_scores in categories.items()
        }
        for scan_id, categories in claims.items()
    }


def _alternative_category(
    original_category_id: int,
    room_type: str,
    scores: dict,
    categories: dict[int, dict],
    floor_record: dict | None,
    category_owners: dict[int, int] | None = None,
    source_class_id: int | None = None,
) -> tuple[int | None, list[dict], str]:
    ranked = sorted(
        ((int(category_id), float(score)) for category_id, score in scores.items()),
        key=lambda item: (-item[1], item[0]),
    )
    eligible = []
    evaluations = []
    for category_id, score in ranked:
        if category_id == original_category_id or score <= 0.0:
            continue
        category = categories.get(category_id)
        if category is None or not category.get("active"):
            evaluations.append({"category_id": category_id, "score": score, "status": "inactive_or_unknown"})
            continue
        if _category_conflicts(category_id, room_type):
            evaluations.append({"category_id": category_id, "name": category["name"], "score": score,
                                "status": "incompatible_with_room"})
            continue
        owner_class_id = (
            (category_owners or {}).get(category_id)
            if category_id in SINGLETON_CATEGORY_IDS else None
        )
        if owner_class_id is not None and owner_class_id != source_class_id:
            evaluations.append({"category_id": category_id, "name": category["name"], "score": score,
                                "status": "owned_by_another_source_class",
                                "owner_source_class_id": owner_class_id})
            continue
        if category_id not in DOOR_WINDOW_CATEGORY_IDS:
            limits = category.get("geometry_constraints", {}).get("base_z_m", [])
            height = floor_record.get("base_height_above_floor_m") if floor_record else None
            if len(limits) != 2 or height is None:
                evaluations.append({"category_id": category_id, "name": category["name"], "score": score,
                                    "status": "floor_height_unavailable"})
                continue
            if not float(limits[0]) <= float(height) <= float(limits[1]):
                evaluations.append({"category_id": category_id, "name": category["name"], "score": score,
                                    "status": "outside_floor_height_range",
                                    "base_height_above_floor_m": float(height),
                                    "allowed_base_height_m": list(limits)})
                continue
            floor_status = "valid"
        else:
            floor_status = "not_checked_door_window"
        candidate = {
            "category_id": category_id,
            "name": category["name"],
            "score": score,
            "floor_status": floor_status,
        }
        eligible.append(candidate)
        evaluations.append({**candidate, "status": "valid_candidate"})

    if not eligible:
        return None, evaluations, "no_valid_alternative"
    top_score = eligible[0]["score"]
    close_count = sum(item["score"] >= top_score - ASSOCIATION_MARGIN for item in eligible)
    if close_count >= MIN_CLOSE_CANDIDATES:
        return None, evaluations, "ambiguous_alternatives"
    return eligible[0]["category_id"], evaluations, "valid_alternative_found"


def _write_final_instance_report(
    scene_dir: Path,
    report_dir: Path,
    row: dict,
    source_instance: dict | None,
    categories: dict[int, dict],
    floor_record: dict | None,
    dimensions: tuple[float, float, float] | None,
    validated_wc_count: int,
) -> bool:
    if not source_instance or not source_instance.get("report_image"):
        return False
    scan_id, source_class_id, instance_id = _key(row)
    phase05_report = scene_dir / scan_id / "segmentation" / "non_structural_similarity" / source_instance["report_image"]
    if not phase05_report.is_file():
        return False

    scores = source_instance.get("category_mean_specificity", {})
    original_id = int(row["original_category_id"])
    final_id = row.get("category_id")
    display_id = int(final_id) if final_id is not None else original_id
    display_category = categories[display_id]
    decisions = []
    if row.get("floor_distance_status") == "rejected_floor_distance":
        decisions.append("Floor-distance check rejected the original category.")
    elif row["room_action"] != "unchanged":
        before = categories[original_id]["name"]
        if row["room_action"] == "reassigned":
            room_id = int(row.get("room_replacement_category_id", display_id))
            decisions.append(
                f"Room rule rejected {before} ({_score(scores, original_id):+.3f}) and selected "
                f"{categories[room_id]['name']} ({_score(scores, room_id):+.3f})."
            )
        else:
            decisions.append(
                f"Room rule rejected {before}; no valid alternative "
                f"({row.get('room_action_reason', 'none').replace('_', ' ')})."
            )
    if row.get("category_class_action") == "reassigned":
        prior_id = int(row["category_before_exclusivity_id"])
        decisions.append(
            f"Category ownership gave {categories[prior_id]['name']} to source class "
            f"{row['category_class_owner_source_class_id']}; reassigned this instance to "
            f"{display_category['name']} ({_score(scores, display_id):+.3f})."
        )
    elif row.get("category_class_action") == "unassigned":
        prior_id = int(row["category_before_exclusivity_id"])
        decisions.append(
            f"Category ownership gave {categories[prior_id]['name']} to source class "
            f"{row['category_class_owner_source_class_id']}; no valid alternative "
            f"({row.get('category_class_action_reason', 'none').replace('_', ' ')})."
        )
    elif final_id is not None:
        decisions.append("Final category retained after room and category checks.")

    if display_id in DOOR_WINDOW_CATEGORY_IDS:
        floor_details = "Floor-height check not applied to door/window categories."
    else:
        limits = categories[display_id].get("geometry_constraints", {}).get("base_z_m", [])
        height = floor_record.get("base_height_above_floor_m") if floor_record else None
        floor_details = (
            f"Base height {float(height):.3f} m (valid {float(limits[0]):.3f}–{float(limits[1]):.3f} m)."
            if height is not None and len(limits) == 2 else "Floor-height check unavailable."
        )
        if final_id is None and row.get("floor_distance_status") != "rejected_floor_distance":
            floor_details += " No valid category assignment remains."

    geometry = categories[display_id].get("geometry_constraints", {})
    if dimensions is None:
        bbox_details = "Instance AABB X × Y × Z: unavailable."
    else:
        bbox_details = (
            "Instance AABB X × Y × Z: "
            + " × ".join(f"{value:.3f}" for value in dimensions)
            + " m."
        )
    width, depth, height = (
        geometry.get("width_m"), geometry.get("depth_m"), geometry.get("height_m")
    )
    if width and depth and height:
        def format_range(bounds: list) -> str:
            return f"{float(bounds[0]):.2f}–{float(bounds[1]):.2f}"

        bbox_details += (
            f"\n{display_category['name']} limits: W {format_range(width)}, "
            f"D {format_range(depth)}, H {format_range(height)} m; X/Z may swap."
        )
        bbox_details += (
            " Candidate passed Phase 05's dimension filter."
            if final_id is not None else (
                " Original category passed Phase 05 dimensions; floor distance rejected it."
                if row.get("floor_distance_status") == "rejected_floor_distance"
                else " Limits shown for the rejected original category."
            )
        )
    else:
        bbox_details += f"\n{display_category['name']} dimension limits unavailable."

    report_path = report_dir / scan_id / (
        f"class_{source_class_id:03d}_inst_{instance_id:03d}_room_consistency.png"
    )
    _write_report(
        report_path,
        phase05_report,
        scores,
        display_category,
        categories,
        (
            "rejected_floor_distance" if row.get("floor_distance_status") == "rejected_floor_distance"
            else "retained_room_reassigned" if final_id is not None and (
                row["room_action"] != "unchanged" or row["category_class_action"] != "unchanged"
            )
            else "retained" if final_id is not None
            else "rejected_room_conflict"
        ),
        "\n" + "\n".join(decisions) + "\n" + bbox_details,
        f"Room {scan_id}: {row['room_type']} · validated WC anchors: {validated_wc_count}\n{floor_details}",
        scan_id,
        source_class_id,
        instance_id,
        check_label="Final category",
    )
    row["report_image"] = str(report_path)
    return True


def check_instance_room_consistency(
    scene_dir: Path = DEFAULT_SCENE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    artifacts_dir: Path | None = None,
) -> dict:
    scene_dir, output_dir = Path(scene_dir).resolve(), Path(output_dir).resolve()
    artifacts_dir = Path(artifacts_dir or scene_dir / "artifacts").resolve()
    floor_manifest_path = artifacts_dir / "instance_floor_distance_check" / "instance_floor_distance_manifest.json"
    if not floor_manifest_path.is_file():
        raise FileNotFoundError(f"Phase 08 floor-distance manifest is missing: {floor_manifest_path}")
    floor_manifest = json.loads(floor_manifest_path.read_text(encoding="utf-8"))

    instances_by_key, categories_by_scan, dimensions_by_key = {}, {}, {}
    for scan_dir in sorted(scene_dir.glob("scan_*/segmentation/non_structural_similarity")):
        scan_id = scan_dir.parents[1].name
        association_path, categories_path = scan_dir / "valid_instance_associations.json", scan_dir / "categories.json"
        if not association_path.is_file() or not categories_path.is_file():
            continue
        categories_by_scan[scan_id] = {
            int(item["id"]): item for item in json.loads(categories_path.read_text(encoding="utf-8"))
        }
        segmentation_path = scan_dir.parent / "segmentation.json"
        if segmentation_path.is_file():
            for item in json.loads(segmentation_path.read_text(encoding="utf-8")).get("instances", []):
                if (
                    "class_id" in item and "instance_index" in item
                    and len(item.get("bounds_min", [])) == len(item.get("bounds_max", [])) == 3
                ):
                    dimensions_by_key[(
                        scan_id,
                        int(item["class_id"]),
                        int(item["instance_index"]) - 1,
                    )] = tuple(
                        float(maximum) - float(minimum)
                        for minimum, maximum in zip(item["bounds_min"], item["bounds_max"])
                    )
        for instance in json.loads(association_path.read_text(encoding="utf-8")).get("instances", []):
            category_id = _instance_prediction(instance)
            if category_id is None:
                continue
            key = (scan_id, int(instance["source_class_id"]), int(instance["instance_id"]))
            if key in instances_by_key:
                raise ValueError(f"Duplicate per-instance category prediction: {key}")
            instances_by_key[key] = {**instance, "_predicted_category_id": category_id}

    floor_records = {_key(item): item for item in floor_manifest.get("associations", [])}
    associations = floor_manifest.get("associations_after_floor_check", [])
    room_groups = sorted({item["scan_id"] for item in associations} | set(categories_by_scan))
    bathroom_anchors = {
        scan_id: [
            _key(item) for item in associations
            if item["scan_id"] == scan_id
            and int(item["category_id"]) == WC_CATEGORY_ID
            and item.get("floor_distance_status") == "retained"
        ]
        for scan_id in room_groups
    }
    room_types = {
        scan_id: "bathroom" if bathroom_anchors[scan_id] else "non_bathroom"
        for scan_id in room_groups
    }

    output_rows, room_counts = [], {
        scan_id: {
            "unchanged": 0,
            "reassigned": 0,
            "unassigned": 0,
            "category_class_reassigned": 0,
            "category_class_unassigned": 0,
        }
        for scan_id in room_groups
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / "instance_reports"
    shutil.rmtree(report_dir, ignore_errors=True)
    for association in associations:
        key = _key(association)
        original_id = int(association["category_id"])
        scan_id = association["scan_id"]
        room_type = room_types[scan_id]
        categories = categories_by_scan[scan_id]
        original_category = categories[original_id]
        result = {
            **association,
            "room_type": room_type,
            "original_category_id": original_id,
            "original_category": original_category["name"],
            "room_action": "unchanged",
            "category_class_action": "unchanged",
        }
        if key in dimensions_by_key:
            result["instance_bbox_xyz_m"] = list(dimensions_by_key[key])
        if not _category_conflicts(original_id, room_type):
            room_counts[scan_id]["unchanged"] += 1
            output_rows.append(result)
            continue

        source_instance = instances_by_key.get(key)
        category_scores = source_instance.get("category_mean_specificity", {}) if source_instance else {}
        replacement_id, candidate_evaluations, replacement_status = _alternative_category(
            original_id, room_type, category_scores, categories, floor_records.get(key),
        )
        result["candidate_evaluations"] = candidate_evaluations
        result["room_action"] = "reassigned" if replacement_id is not None else "unassigned"
        result["room_action_reason"] = replacement_status
        if replacement_id is not None:
            result["replacement_dimension_validation"] = "Phase 05 dimension-filtered instance scores"
            result["room_replacement_category_id"] = replacement_id
            result["category_id"] = replacement_id
            result["category"] = categories[replacement_id]["name"]
            result["replacement_score"] = float(category_scores[str(replacement_id)])
            result["floor_distance_status"] = (
                "not_checked_door_window"
                if replacement_id in DOOR_WINDOW_CATEGORY_IDS
                else "retained"
            )
            room_counts[scan_id]["reassigned"] += 1
        else:
            result["category_id"] = None
            result["category"] = None
            room_counts[scan_id]["unassigned"] += 1
        output_rows.append(result)


    category_owners = _category_owners(output_rows, instances_by_key)
    category_class_conflicts = []
    for scan_id in room_groups:
        owners = category_owners.get(scan_id, {})
        categories = categories_by_scan[scan_id]
        losing_rows = []
        conflict_classes = {}
        for row in output_rows:
            if row["scan_id"] != scan_id or row.get("category_id") is None:
                continue
            source_class_id = int(row["source_class_id"])
            category_id = int(row["category_id"])
            if category_id not in SINGLETON_CATEGORY_IDS:
                continue
            owner_class_id = owners.get(category_id)
            if owner_class_id == source_class_id:
                continue
            key = _key(row)
            scores = instances_by_key.get(key, {}).get("category_mean_specificity", {})
            row["category_before_exclusivity_id"] = category_id
            row["category_before_exclusivity"] = categories[category_id]["name"]
            row["category_class_owner_source_class_id"] = owner_class_id
            row["category_class_conflict_score"] = _score(scores, category_id)
            losing_rows.append(row)
            conflict_classes.setdefault(category_id, {}).setdefault(source_class_id, []).append(row)

        for category_id, losing_classes in conflict_classes.items():
            owner_class_id = owners[category_id]
            owner_rows = [
                row for row in output_rows
                if row["scan_id"] == scan_id
                and row.get("category_id") == category_id
                and int(row["source_class_id"]) == owner_class_id
            ]
            owner_score = max(
                _score(instances_by_key.get(_key(row), {}).get("category_mean_specificity", {}), category_id)
                for row in owner_rows
            )
            category_class_conflicts.append({
                "scan_id": scan_id,
                "category_id": category_id,
                "category": categories[category_id]["name"],
                "owner_source_class_id": owner_class_id,
                "owner_instance_score": owner_score,
                "losing_source_classes": [
                    {
                        "source_class_id": class_id,
                        "best_instance_score": max(
                            row["category_class_conflict_score"] for row in class_rows
                        ),
                        "instances": [
                            {"source_class_id": int(row["source_class_id"]),
                             "instance_id": int(row["instance_id"])}
                            for row in class_rows
                        ],
                    }
                    for class_id, class_rows in sorted(losing_classes.items())
                ],
            })

        pending = list(losing_rows)
        while pending:
            alternatives = []
            for row in pending:
                key = _key(row)
                source_instance = instances_by_key.get(key)
                scores = source_instance.get("category_mean_specificity", {}) if source_instance else {}
                source_class_id = int(row["source_class_id"])
                replacement_id, evaluations, reason = _alternative_category(
                    int(row["category_before_exclusivity_id"]),
                    room_types[scan_id],
                    scores,
                    categories,
                    floor_records.get(key),
                    category_owners=owners,
                    source_class_id=source_class_id,
                )
                row["category_class_candidate_evaluations"] = evaluations
                row["category_class_action_reason"] = reason
                if replacement_id is not None:
                    alternatives.append((
                        _score(scores, replacement_id),
                        source_class_id,
                        int(row["instance_id"]),
                        row,
                        replacement_id,
                    ))
            if not alternatives:
                for row in pending:
                    row["category_class_action"] = "unassigned"
                    row["category_id"] = None
                    row["category"] = None
                    room_counts[scan_id]["category_class_unassigned"] += 1
                break

            alternatives.sort(key=lambda item: (-item[0], item[1], item[2]))
            replacement_score, source_class_id, _, row, replacement_id = alternatives[0]
            row["category_class_action"] = "reassigned"
            row["category_class_action_reason"] = "category_owned_by_higher_confidence_source_class"
            row["category_id"] = replacement_id
            row["category"] = categories[replacement_id]["name"]
            row["category_class_replacement_score"] = replacement_score
            row["replacement_dimension_validation"] = "Phase 05 dimension-filtered instance scores"
            row["floor_distance_status"] = (
                "not_checked_door_window"
                if replacement_id in DOOR_WINDOW_CATEGORY_IDS else "retained"
            )
            if replacement_id in SINGLETON_CATEGORY_IDS:
                owners.setdefault(replacement_id, source_class_id)
            room_counts[scan_id]["category_class_reassigned"] += 1
            pending = [candidate for candidate in pending if candidate is not row]

    final_report_rows = list(output_rows)
    reported_keys = {_key(row) for row in final_report_rows}
    for floor_result in floor_manifest.get("associations", []):
        if floor_result.get("association_status") != "rejected_floor_distance":
            continue
        key = _key(floor_result)
        if key in reported_keys or key not in instances_by_key:
            continue
        original_id = int(floor_result["category_id"])
        final_report_rows.append({
            **floor_result,
            "room_type": room_types.get(key[0], "unknown"),
            "original_category_id": original_id,
            "original_category": categories_by_scan[key[0]][original_id]["name"],
            "category_id": None,
            "category": None,
            "floor_distance_status": "rejected_floor_distance",
            "room_action": "unchanged",
            "category_class_action": "unchanged",
        })

    report_count = 0
    for row in final_report_rows:
        key = _key(row)
        source_instance = instances_by_key.get(key)
        if source_instance:
            row["phase05_instance_metadata"] = {
                name: value for name, value in source_instance.items() if not name.startswith("_")
            }
        floor_record = floor_records.get(key)
        if floor_record:
            row["floor_measurement_metadata"] = {
                name: value for name, value in floor_record.items() if name != "report_image"
            }
            if floor_record.get("report_image"):
                row["floor_distance_report_image"] = floor_record["report_image"]
        if _write_final_instance_report(
            scene_dir,
            report_dir,
            row,
            source_instance,
            categories_by_scan[row["scan_id"]],
            floor_record,
            dimensions_by_key.get(key),
            len(bathroom_anchors[row["scan_id"]]),
        ):
            report_count += 1

    room_summary = []
    for scan_id in room_groups:
        room_rows = [item for item in output_rows if item["scan_id"] == scan_id]
        room_summary.append({
            "room_id": scan_id,
            "room_type": room_types[scan_id],
            "classification_rule": "at_least_one_floor_retained_wc" if bathroom_anchors[scan_id] else "no_floor_retained_wc",
            "validated_wc_anchors": [
                {"source_class_id": key[1], "instance_id": key[2]} for key in bathroom_anchors[scan_id]
            ],
            "floor_checked_instance_count": len(room_rows),
            **room_counts[scan_id],
        })

    associations_path = output_dir / "associations_after_room_check.json"
    manifest_path = output_dir / "room_category_consistency_manifest.json"
    association_payload = {
        "schema_version": 1,
        "source_floor_distance_manifest": str(floor_manifest_path),
        "associations": [item for item in final_report_rows if item.get("category_id") is not None],
        "unassigned_instances": [item for item in final_report_rows if item.get("category_id") is None],
    }
    temporary_path = associations_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(association_payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(associations_path)
    manifest = {
        "schema_version": 1,
        "source_floor_distance_manifest": str(floor_manifest_path),
        "room_grouping": "one source scan per room, matching the Phase 07 room-scan inputs",
        "bathroom_rule": "at least one per-instance WC association retained by the Phase 08 floor-distance check",
        "non_bathroom_rule": "no floor-retained WC; WC, bidet, and shower predictions are context-incompatible",
        "bathroom_incompatible_category_ids": sorted(BATHROOM_INCOMPATIBLE_CATEGORY_IDS),
        "alternative_rule": "highest-scoring positive Phase 05 dimension-valid category that is room-compatible and passes its base-height range; reject if five or more eligible scores fall within 0.02 of the top",
        "door_window_alternatives": "Phase 08 floor-height check remains skipped for door/window categories",
        "category_class_exclusive_category_ids": sorted(SINGLETON_CATEGORY_IDS),
        "category_class_exclusivity_rule": "after floor and room checks, WC, washing machine, refrigerator, oven, dishwasher, shower, and bidet may each belong to at most one source class per room; the class with the highest individual instance score wins, and any number of its instances may retain the category",
        "other_categories": "no source-class exclusivity is applied; instances within a source class may have different category assignments",
        "category_class_fallback_rule": "conflicting instances try their highest-scoring Phase 05 dimension-valid alternative; exclusive categories must be unowned or owned by the same source class",
        "category_class_tie_break": "equal class scores are resolved by the lower source_class_id",
        "room_count": len(room_summary),
        "room_summaries": room_summary,
        "floor_checked_instance_count": len(output_rows),
        "category_class_conflict_count": len(category_class_conflicts),
        "category_class_conflicts": category_class_conflicts,
        "room_context_reassigned_instance_count": sum(counts["reassigned"] for counts in room_counts.values()),
        "room_context_unassigned_instance_count": sum(counts["unassigned"] for counts in room_counts.values()),
        "category_class_reassigned_instance_count": sum(counts["category_class_reassigned"] for counts in room_counts.values()),
        "category_class_unassigned_instance_count": sum(counts["category_class_unassigned"] for counts in room_counts.values()),
        "reassigned_instance_count": sum(
            item.get("category_id") is not None
            and int(item["category_id"]) != int(item["original_category_id"])
            for item in final_report_rows
        ),
        "unassigned_instance_count": sum(item.get("category_id") is None for item in final_report_rows),
        "floor_distance_rejected_instance_count": sum(
            item.get("floor_distance_status") == "rejected_floor_distance" for item in final_report_rows
        ),
        "final_instance_reports_written": report_count,
        "associations_after_room_check": str(associations_path),
        "associations": final_report_rows,
    }
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = check_instance_room_consistency(args.scene_dir, args.output_dir)
    print(
        f"Checked {result['floor_checked_instance_count']} floor-validated associations in "
        f"{result['room_count']} room scans: {result['category_class_conflict_count']} "
        f"category/class conflicts, {result['reassigned_instance_count']} final category changes, "
        f"{result['unassigned_instance_count']} unassigned."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
