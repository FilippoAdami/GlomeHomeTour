#!/usr/bin/env python3
"""Self-test for shared/schemas/ (Phase 2 §6). Five checks, one line each, exit non-zero on
any failure. No pytest/unittest scaffolding -- add a real test framework if this script grows
past a screenful, not before.
"""
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMAS_DIR = Path(__file__).parent
FIXTURES_DIR = SCHEMAS_DIR / "fixtures"

# Kept in sync by hand with mobile/android's DatasetFormat.TRAJECTORY_HEADER (see §6: the CSV
# header string itself is the version marker, not a JSON Schema wrapper).
TRAJECTORY_HEADER = "timestamp_ns,tx,ty,tz,qx,qy,qz,qw,tracking,exported"

FIXTURE_SCHEMA_PAIRS = [
    ("transforms.example.json", "transforms.schema.json"),
    ("coverage_summary.example.json", "coverage_summary.schema.json"),
    ("floorplan.example.json", "floorplan.schema.json"),
    ("panorama_manifest.example.json", "panorama_manifest.schema.json"),
    ("asset_manifest.example.json", "asset_manifest.schema.json"),
    ("asset_manifest.free.example.json", "asset_manifest.schema.json"),
    ("mesh_manifest.example.json", "mesh_manifest.schema.json"),
]


def check_schemas_are_valid() -> list[str]:
    errors = []
    for schema_path in sorted(SCHEMAS_DIR.glob("*.schema.json")):
        schema = json.loads(schema_path.read_text())
        try:
            Draft202012Validator.check_schema(schema)
        except Exception as e:
            errors.append(f"{schema_path.name}: {e}")
    return errors


def check_fixtures_validate() -> list[str]:
    errors = []
    for fixture_name, schema_name in FIXTURE_SCHEMA_PAIRS:
        schema = json.loads((SCHEMAS_DIR / schema_name).read_text())
        instance = json.loads((FIXTURES_DIR / fixture_name).read_text())
        validator = Draft202012Validator(schema)
        for e in validator.iter_errors(instance):
            errors.append(f"{fixture_name} vs {schema_name}: {e.message}")
    return errors


def check_trajectory_header() -> list[str]:
    csv_path = FIXTURES_DIR / "trajectory.example.csv"
    header = csv_path.read_text().splitlines()[0]
    if header != TRAJECTORY_HEADER:
        return [f"trajectory.example.csv header {header!r} != {TRAJECTORY_HEADER!r}"]
    return []


def check_floorplan_panorama_cross_refs() -> list[str]:
    """Constraints §3/§4 call out as schema-must-enforce but that need arithmetic or sibling-
    document lookups plain JSON Schema (no $data extension) can't express: node-id
    cross-references between floorplan.json and panorama_manifest.json, and each wall's
    opening offset_m + width_m <= the wall's own Euclidean length.
    """
    errors = []
    floorplan = json.loads((FIXTURES_DIR / "floorplan.example.json").read_text())
    panorama = json.loads((FIXTURES_DIR / "panorama_manifest.example.json").read_text())
    node_ids = {node["id"] for node in panorama["nodes"]}
    for node_id in floorplan["panorama_node_ids"]:
        if node_id not in node_ids:
            errors.append(f"floorplan panorama_node_ids has {node_id!r} not in panorama_manifest nodes")
    for node in panorama["nodes"]:
        for neighbor in node["neighbors"]:
            if neighbor not in node_ids:
                errors.append(f"panorama_manifest node {node['id']!r} neighbor {neighbor!r} not a known node id")

    for room in floorplan["rooms"]:
        for wall in room.get("walls", []):
            (sx, sy), (ex, ey) = wall["start"], wall["end"]
            length = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
            for opening in wall.get("openings", []):
                extent = opening["offset_m"] + opening["width_m"]
                if extent > length:
                    errors.append(
                        f"wall {wall['id']!r} opening extends to {extent}m, past wall length {length}m"
                    )
    return errors


def main() -> int:
    checks = [
        ("every *.schema.json is valid Draft 2020-12", check_schemas_are_valid),
        ("every fixture validates against its schema", check_fixtures_validate),
        ("trajectory.example.csv header matches DatasetFormat.TRAJECTORY_HEADER", check_trajectory_header),
        ("floorplan <-> panorama_manifest node-id cross-references", check_floorplan_panorama_cross_refs),
    ]
    failed = False
    for label, check in checks:
        errors = check()
        if errors:
            failed = True
            print(f"FAIL: {label}")
            for e in errors:
                print(f"  - {e}")
        else:
            print(f"OK: {label}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
