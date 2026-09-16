#!/usr/bin/env python3
"""The one entry point: a compressed capture in, a trained 2DGS scene out.

    .venv/bin/python run_pipeline.py scenes/Bedroom2.zip

Every step is also a standalone script; this only sequences them and summarises.
Steps run **in-process** (imported and called), so a failure raises here with a
real traceback instead of an exit code.

Nothing is deleted mid-pipeline. Frames a step rejects are *moved* into that
step's own discard folder together with their camera entry, so any step can be
inspected, re-run or resumed in isolation. ``current_scene/`` is removed only
after a fully successful run, and ``--keep-workspace`` keeps even that.

Resuming is the default: each step records itself in ``pipeline_state.json`` and
skips if its outputs are already present. ``--force`` re-runs a step, first
restoring the frames it previously discarded so it sees its original input.

    --from-step colmap      start there, skip everything earlier
    --only-step depth       run exactly one step
    --force                 re-run even if already complete
    --keep-workspace        do not delete current_scene/ on success
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

_backend_dir = Path(__file__).resolve().parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
from Utilities.pipeline_paths import bootstrap

bootstrap()

from Utilities.pipeline_step import read_state

DEFAULT_WORKSPACE = _backend_dir / "current_scene"

# (state name, module path) in execution order. The state name must match the
# one the step passes to StepContext, or resume cannot see its record.
STEPS: tuple[tuple[str, str], ...] = (
    ("extract", "Utilities/step_extract.py"),
    ("filter_quality", "00_ingestion/step_filter_quality.py"),
    ("rotate_upright", "00_ingestion/step_rotate_upright.py"),
    ("colmap", "01_poses_refinment/step_colmap.py"),
    ("filter_depth", "02_depth_estimation/step_filter_depth.py"),
    ("depth", "02_depth_estimation/step_depth.py"),
    ("train", "03_2DGS_training/step_train.py"),
)
STEP_NAMES = [name for name, _ in STEPS]


def _load(module_path: str):
    """Import a step by file path -- the numbered stage folders are not packages."""
    path = _backend_dir / module_path
    spec = importlib.util.spec_from_file_location(f"step_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_summary(workspace: Path, order: list[str]) -> Path:
    """One file answering "what did this run do, and how long did each part take"."""
    state = read_state(workspace)
    lines = ["=" * 70, " PIPELINE SUMMARY", "=" * 70, f"Workspace: {workspace}", ""]

    for name in order:
        entry = state.get(name, {})
        status = entry.get("status", "not run")
        duration = entry.get("duration_s")
        head = f"{name:<16} {status:<10}"
        lines.append(head + (f"{duration:>8.1f}s" if isinstance(duration, (int, float)) else ""))

        stats_path = workspace / f"{name}_stats.json"
        if not stats_path.exists():
            continue
        stats = json.loads(stats_path.read_text())
        for key in ("total_in", "kept", "discarded", "kept_pct", "registered",
                    "unregistered", "sampson_rejects", "frames", "surfels",
                    "rotation_outlier_count", "final_ply_mb"):
            if key in stats:
                lines.append(f"{'':<16}   {key}: {stats[key]}")
        if isinstance(stats.get("reasons"), dict):
            reasons = {k: v for k, v in stats["reasons"].items() if v}
            lines.append(f"{'':<16}   reasons: {reasons}")

    lines += ["", "=" * 70]
    text = "\n".join(lines)
    out = workspace / "pipeline_summary.txt"
    out.write_text(text + "\n")
    print("\n" + text)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", help="Capture .zip or extracted directory")
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--from-step", choices=STEP_NAMES)
    parser.add_argument("--only-step", choices=STEP_NAMES)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-workspace", action="store_true")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)

    if args.only_step:
        selected = [args.only_step]
    else:
        start = STEP_NAMES.index(args.from_step) if args.from_step else 0
        selected = STEP_NAMES[start:]

    if "extract" in selected and not args.source:
        parser.error("source is required when running the extract step "
                     "(use --from-step/--only-step to skip it)")

    print(f"Steps: {' -> '.join(selected)}")
    t0 = time.perf_counter()

    for name, module_path in STEPS:
        if name not in selected:
            continue
        step_args = [args.source] if name == "extract" else []
        step_args += ["--workspace", str(workspace)]
        if args.force:
            step_args.append("--force")

        rc = _load(module_path).main(step_args)
        if rc != 0:
            print(f"\nStep '{name}' returned {rc}; stopping. "
                  f"The workspace is intact at {workspace} -- inspect "
                  f"{name}_log.txt, then re-run to resume from here.")
            write_summary(workspace, selected)
            return rc

    total = time.perf_counter() - t0
    print(f"\nPipeline finished in {total / 60:.1f} min")
    write_summary(workspace, selected)

    ran_everything = selected == STEP_NAMES
    if ran_everything and not args.keep_workspace:
        # Only here, and only after every step succeeded, is deleting safe --
        # and even then the trained scene must already be somewhere else.
        print(f"\nAll steps succeeded. current_scene/ holds the only copy of the "
              f"trained scene; copy it out, then delete {workspace} by hand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
