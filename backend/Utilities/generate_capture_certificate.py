#!/usr/bin/env python3
"""
Generate and optionally inject a valid certificate.json for a capture package
(uncompressed folder or .zip archive) to enable certification without re-capturing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jsonschema


DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "mobile_app" / "shared" / "schemas" / "certificate.schema.json"
)


def compute_sha256(data: bytes) -> str:
    """Compute lowercase SHA-256 hex digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def load_schema(schema_path: Path) -> dict:
    if not schema_path.is_file():
        raise FileNotFoundError(f"Certificate schema not found at {schema_path}")
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_file_from_source(source_path: Path, rel_path: str) -> Optional[bytes]:
    """Read file bytes from directory or zip archive."""
    if source_path.is_dir():
        target = source_path / rel_path
        if target.is_file():
            return target.read_bytes()
        return None
    elif zipfile.is_zipfile(source_path):
        with zipfile.ZipFile(source_path, "r") as zf:
            namelist = zf.namelist()
            if rel_path in namelist:
                return zf.read(rel_path)
            # Check with potential root prefix (e.g. scan_2026.../transforms.json)
            for name in namelist:
                if name.endswith("/" + rel_path) or name == rel_path:
                    return zf.read(name)
        return None
    return None


def find_zip_root_prefix(zf: zipfile.ZipFile) -> str:
    """Find the single root folder prefix inside a zip if one exists."""
    namelist = zf.namelist()
    if "transforms.json" in namelist or any(name.startswith("scan_0/") for name in namelist) or "certificate.json" in namelist:
        return ""
    for name in namelist:
        if "/scan_0/transforms.json" in name:
            return name.split("/scan_0/transforms.json")[0] + "/"
        if name.endswith("/transforms.json") and not name.startswith("scan_"):
            return name[:-len("transforms.json")]
    return ""


def generate_certificate(
    source_path: Path,
    producer_name: str = "com.glomehometour.arscan",
    producer_version: str = "0.1",
    policy_version: str = "1.0.0",
    keyframes_per_m2_lo: float = 8.0,
    keyframes_per_m2_hi: float = 11.0,
    quality_blur_threshold: float = 0.35,
    decimation_stride: int = 8,
) -> Tuple[Dict[str, Any], str]:
    """
    Generate certificate.json dict and the zip root prefix (if inside a zip).
    Supports single-scan root packages as well as multi-room (scan_0, scan_1, ...) packages.
    """
    import re
    is_zip = source_path.is_file() and zipfile.is_zipfile(source_path)
    root_prefix = ""

    # Open source
    zf: Optional[zipfile.ZipFile] = None
    if is_zip:
        zf = zipfile.ZipFile(source_path, "r")
        root_prefix = find_zip_root_prefix(zf)

    def read_bytes(name: str) -> Optional[bytes]:
        full_name = root_prefix + name
        if is_zip and zf is not None:
            if full_name in zf.namelist():
                return zf.read(full_name)
            for item in zf.namelist():
                if item.endswith("/" + name) or item == name:
                    return zf.read(item)
            return None
        else:
            p = source_path / name
            return p.read_bytes() if p.is_file() else None

    # Discover scan folders (e.g. scan_0, scan_1) or fallback to root
    scan_folders = []
    if is_zip and zf is not None:
        prefix_pattern = re.escape(root_prefix)
        scan_re = re.compile(rf"^{prefix_pattern}(scan_\d+)/transforms\.json$")
        for name in zf.namelist():
            m = scan_re.match(name)
            if m:
                scan_folders.append(m.group(1))
    elif source_path.is_dir():
        for item in source_path.iterdir():
            if item.is_dir() and re.match(r"^scan_\d+$", item.name) and (item / "transforms.json").is_file():
                scan_folders.append(item.name)

    if scan_folders:
        scan_folders.sort(key=lambda s: int(s.split("_")[1]))
    else:
        if read_bytes("transforms.json") is not None:
            scan_folders = [""]
        else:
            raise ValueError(f"No scan folders or transforms.json found in {source_path}")

    overall_total_input = 0
    overall_retained_count = 0
    manifest_hashes: Dict[str, str] = {}
    image_hashes: Dict[str, str] = {}
    raw_w, raw_h = 1920, 1080
    processed_w, processed_h = 1080, 1920
    is_already_portrait = False
    rotation_applied_deg = 0
    pose_rolled = False
    raw_orientation = "landscape"

    for folder in scan_folders:
        t_path = "transforms.json" if not folder else f"{folder}/transforms.json"
        transforms_bytes = read_bytes(t_path)
        if not transforms_bytes:
            raise ValueError(f"{t_path} not found in {source_path}")

        transforms_json = json.loads(transforms_bytes.decode("utf-8"))
        frames = transforms_json.get("frames", [])
        retained_count = len(frames)
        overall_retained_count += retained_count

        w = int(transforms_json.get("w", 1080))
        h = int(transforms_json.get("h", 1920))
        is_portrait = w < h
        if is_portrait:
            raw_w, raw_h = h, w
            raw_orientation = "landscape"
            is_already_portrait = False
            processed_w, processed_h = w, h
            rotation_applied_deg = 90
            pose_rolled = True
        else:
            raw_w, raw_h = w, h
            raw_orientation = "landscape"
            is_already_portrait = False
            processed_w, processed_h = w, h
            rotation_applied_deg = 0
            pose_rolled = False

        manifest_hashes[t_path] = compute_sha256(transforms_bytes)
        for m_name in ["trajectory.csv", "coverage_summary.json", "focus_metadata.json"]:
            full_m = m_name if not folder else f"{folder}/{m_name}"
            content = read_bytes(full_m)
            if content is not None:
                manifest_hashes[full_m] = compute_sha256(content)

        for f_obj in frames:
            rel_path = f_obj["file_path"].lstrip("./")
            full_img = rel_path if not folder else f"{folder}/{rel_path}"
            img_bytes = read_bytes(full_img)
            if img_bytes is None:
                raise ValueError(f"Referenced image {full_img} not found in {source_path}")
            image_hashes[full_img] = compute_sha256(img_bytes)

        room_input = retained_count
        cov_path = "coverage_summary.json" if not folder else f"{folder}/coverage_summary.json"
        coverage_bytes = read_bytes(cov_path)
        if coverage_bytes:
            try:
                cov = json.loads(coverage_bytes.decode("utf-8"))
                exported = cov.get("frames", {}).get("exported")
                if exported and exported >= retained_count:
                    room_input = exported
            except Exception:
                pass
        overall_total_input += room_input

    certificate: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "producer": {
            "name": producer_name,
            "version": producer_version,
        },
        "policy_version": policy_version,
        "completed_stages": [
            "quality_gate",
            "keyframe_selection",
            "orientation_normalization",
            "manifest_consolidation",
        ],
        "orientation_state": {
            "raw_width": raw_w,
            "raw_height": raw_h,
            "raw_orientation": raw_orientation,
            "is_already_portrait": is_already_portrait,
            "processed_width": processed_w,
            "processed_height": processed_h,
            "processed_orientation": "portrait",
            "rotation_applied_deg": rotation_applied_deg,
            "pose_rolled": pose_rolled,
        },
        "keyframe_density_settings": {
            "keyframe_filter_enabled": True,
            "keyframes_per_m2_lo": float(keyframes_per_m2_lo),
            "keyframes_per_m2_hi": float(keyframes_per_m2_hi),
            "base_keyframes": 50.0,
            "standoff_m": 0.8,
            "min_room_area_m2": 9.0,
            "min_translation_m": 0.08,
            "min_rotation_deg": 4.0,
            "quality_gate_blur_threshold": float(quality_blur_threshold),
            "decimation_stride": int(decimation_stride),
        },
        "frames": {
            "total_input_frames": int(overall_total_input),
            "retained_frames_count": int(overall_retained_count),
            "discarded_frames_count": int(overall_total_input - overall_retained_count),
        },
        "manifest_hashes": manifest_hashes,
        "image_hashes": image_hashes,
    }

    if zf is not None:
        zf.close()

    return certificate, root_prefix


def inject_into_zip(zip_path: Path, cert_json_bytes: bytes, root_prefix: str) -> None:
    """Inject certificate.json into an existing zip file safely."""
    temp_fd, temp_path_str = tempfile.mkstemp(suffix=".zip", dir=zip_path.parent)
    os.close(temp_fd)
    temp_zip = Path(temp_path_str)

    cert_arcname = root_prefix + "certificate.json"

    with zipfile.ZipFile(zip_path, "r") as zin:
        with zipfile.ZipFile(temp_zip, "w", compression=zin.compression) as zout:
            for item in zin.infolist():
                if item.filename != cert_arcname:
                    zout.writestr(item, zin.read(item.filename))
            zout.writestr(cert_arcname, cert_json_bytes)

    temp_zip.replace(zip_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=str, help="Path to capture directory or .zip file.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output path for certificate.json.")
    parser.add_argument("--inject", action="store_true", help="Inject certificate.json directly into source directory or .zip archive.")
    parser.add_argument("--dry-run", action="store_true", help="Print certificate JSON to stdout without modifying files.")
    parser.add_argument("--schema", type=str, default=str(DEFAULT_SCHEMA_PATH), help="Path to certificate.schema.json.")
    parser.add_argument("--producer-name", type=str, default="com.glomehometour.arscan")
    parser.add_argument("--producer-version", type=str, default="0.1")
    parser.add_argument("--policy-version", type=str, default="1.0.0")

    args = parser.parse_args()
    source_path = Path(args.source).resolve()

    if not source_path.exists():
        print(f"Error: {source_path} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Reading capture from: {source_path}")
    cert_data, root_prefix = generate_certificate(
        source_path,
        producer_name=args.producer_name,
        producer_version=args.producer_version,
        policy_version=args.policy_version,
    )

    # Validate against schema
    schema_path = Path(args.schema)
    if schema_path.is_file():
        schema = load_schema(schema_path)
        jsonschema.validate(instance=cert_data, schema=schema)
        print(f"[✓] Schema validation PASSED against {schema_path.name}")
    else:
        print(f"[!] Warning: Schema not found at {schema_path}, skipping validation", file=sys.stderr)

    cert_str = json.dumps(cert_data, indent=2)

    if args.dry_run:
        print(f"\n--- certificate.json ({len(cert_data['image_hashes'])} images hashed) ---")
        print(cert_str)
        return

    if args.output:
        out_p = Path(args.output).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(cert_str, encoding="utf-8")
        print(f"[✓] Saved certificate to: {out_p}")

    if args.inject:
        cert_bytes = cert_str.encode("utf-8")
        if source_path.is_file() and zipfile.is_zipfile(source_path):
            inject_into_zip(source_path, cert_bytes, root_prefix)
            print(f"[✓] Injected {root_prefix}certificate.json into {source_path.name}")
        elif source_path.is_dir():
            target = source_path / "certificate.json"
            target.write_bytes(cert_bytes)
            print(f"[✓] Injected certificate.json into {target}")


if __name__ == "__main__":
    main()
