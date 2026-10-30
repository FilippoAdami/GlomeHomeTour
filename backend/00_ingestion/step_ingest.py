#!/usr/bin/env python3
"""Standalone ingestion phase for ZIP archives or extracted capture folders."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from Utilities.pipeline_paths import bootstrap

bootstrap()

from Utilities.pipeline_step import StepContext
from ingestion import IngestionError, ingest

DEFAULT_WORKSPACE = _backend_dir / "current_scene"


def _publish(output_root: Path, workspace: Path) -> None:
    backup = output_root / "_backup"
    backup.mkdir()
    package_names = {
        "00_ingestion", "images", "transforms.json", "trajectory.csv", "coverage_summary.json",
        "focus_metadata.json", "certificate.json", "lidar_data", "source_assets", "colmap_model",
        "ingestion_results.json",
    }
    existing_package = {path.name for path in workspace.iterdir()
                        if path.name in package_names or re.fullmatch(r"scan_\d+", path.name)}
    names = sorted({path.name for path in output_root.iterdir() if path != backup} | existing_package)
    moved_old: list[str] = []
    moved_new: list[str] = []
    try:
        for name in names:
            current = workspace / name
            if current.exists():
                current.rename(backup / name)
                moved_old.append(name)
        for name in names:
            staged = output_root / name
            if staged.exists():
                staged.rename(workspace / name)
                moved_new.append(name)
    except OSError:
        for name in reversed(moved_new):
            current = workspace / name
            if current.exists():
                current.rename(output_root / name)
        for name in reversed(moved_old):
            old = backup / name
            if old.exists():
                old.rename(workspace / name)
        raise


def _clear_extract_stats(workspace: Path) -> None:
    path = workspace / "pipeline_stats.json"
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data.pop("extract", None)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except (OSError, json.JSONDecodeError):
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Capture ZIP or extracted folder")
    args = parser.parse_args(argv)

    source = Path(args.source).expanduser().absolute()
    workspace = DEFAULT_WORKSPACE
    result_file = workspace / "ingestion_results.json"
    workspace.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".ingestion-", dir=workspace))
    output_root = temporary / "output"
    started = time.perf_counter()
    try:
        with StepContext("extract", workspace, artifacts_dir=temporary / "logs") as ctx:
            result = ingest(source, output_root, started=started, log=ctx.note)
            ctx.metric("source", str(source))
            ctx.metric("source_format", result["source_format"])
            ctx.metric("poses_format", result["poses_format"])
            ctx.metric("frames", result["retained_images_count"])
            ctx.metric("frames_processed", result["retained_images_count"])
            result_location = (workspace / "scan_*/ingestion_results.json"
                               if result.get("scan_count") else result_file)
            ctx.metric("ingestion_results", str(result_location))
            ctx.note(f"Retained images: {result['retained_images_count']}")
        _publish(output_root, workspace)
    except IngestionError as exc:
        _clear_extract_stats(workspace)
        print(json.dumps({"error": {"code": exc.code, "message": str(exc), "details": exc.details}}, indent=2), file=sys.stderr)
        return 2
    except Exception as exc:
        _clear_extract_stats(workspace)
        print(json.dumps({"error": {"code": "INGESTION_FAILED", "message": str(exc)}}, indent=2), file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    if result.get("scan_count"):
        print(f"[ingestion] ready: {result['scan_count']} scans under {workspace}")
    else:
        print(f"[ingestion] ready: {workspace / 'ingestion_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
