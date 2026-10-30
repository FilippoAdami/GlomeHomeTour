#!/usr/bin/env python3
"""Compatibility entrypoint for the standalone ``00_ingestion`` phase."""

from pathlib import Path
import sys

_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from Utilities.pipeline_paths import bootstrap

bootstrap()

from step_ingest import main


if __name__ == "__main__":
    raise SystemExit(main())
