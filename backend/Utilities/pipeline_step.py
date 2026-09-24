"""Shared contract for pipeline steps: logging, metrics, stats, resume.

Every step wraps its work in a :class:`StepContext`, which owns files in
the workspace:

* ``<name>_log.txt``   -- human-readable, everything the step printed
* ``<name>_stats.json`` -- machine-readable metrics, keyed however the step likes
* ``pipeline_stats.json`` -- high-level aggregated metrics across pipeline steps
"""

from __future__ import annotations

import json
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_pipeline_stats(workspace: Path | str) -> dict[str, Any]:
    path = Path(workspace) / "pipeline_stats.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def is_done(workspace: Path | str, name: str, outputs: Iterable[Path | str] = ()) -> bool:
    """True if the step recorded metrics in pipeline_stats.json *and* its declared outputs still exist."""
    stats = read_pipeline_stats(workspace)
    if name not in stats:
        return False
    return all(Path(p).exists() for p in outputs)


def update_pipeline_stats(
    workspace: Path | str,
    step_name: str,
    total_time: float,
    frames_processed: int,
    time_per_frame: Optional[float] = None,
    extra_metrics: Optional[dict[str, Any]] = None,
) -> Path:
    """Save or update substep metrics into <workspace>/pipeline_stats.json."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    stats_file = workspace / "pipeline_stats.json"

    if time_per_frame is None:
        time_per_frame = round(total_time / max(1, frames_processed), 6) if frames_processed > 0 else 0.0

    data = {}
    if stats_file.exists():
        try:
            data = json.loads(stats_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}

    step_data = {
        "total_time": round(float(total_time), 4),
        "frames_processed": int(frames_processed),
        "time_per_frame": round(float(time_per_frame), 6),
    }
    if extra_metrics:
        step_data.update(extra_metrics)

    data[step_name] = step_data
    stats_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return stats_file


class StepContext:
    """Context manager owning one step's log and metrics.

    ``workspace`` is where the shared ``pipeline_stats.json`` lives (always the
    scene root). ``artifacts_dir`` is where this step's own ``<name>_log.txt``
    and ``<name>_stats.json`` are written -- its per-stage subfolder, if it has
    one, else the same as ``workspace``.
    """

    def __init__(self, name: str, workspace: Path | str, artifacts_dir: Path | str | None = None):
        self.name = name
        self.workspace = Path(workspace)
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else self.workspace
        self.metrics: dict[str, Any] = {}
        self.timings: dict[str, float] = {}
        self._started = 0.0
        self._log = None

    # ---------------------------------------------------------------- recording

    def metric(self, key: str, value: Any) -> None:
        self.metrics[key] = value

    def note(self, text: str = "") -> None:
        """Write a line to both the log file and stdout."""
        print(text)
        if self._log is not None:
            self._log.write(text + "\n")
            self._log.flush()

    @contextmanager
    def timer(self, sub_phase: str):
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.timings[sub_phase] = round(elapsed, 2)
            self.note(f"  [{sub_phase}] {elapsed:.1f}s")

    # ---------------------------------------------------------------- lifecycle

    def __enter__(self) -> "StepContext":
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._started = time.time()
        self._log = open(self.artifacts_dir / f"{self.name}_log.txt", "w", encoding="utf-8")
        self.note("=" * 70)
        self.note(f"  STEP: {self.name}   started {_now()}")
        self.note("=" * 70)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = round(time.time() - self._started, 4)
        self.metrics["elapsed_s"] = elapsed
        if self.timings:
            self.metrics["timings_s"] = self.timings

        frames = self.metrics.get("frames_processed")
        if frames is None:
            for alt_key in ("total_in", "frames", "total_evaluated"):
                if alt_key in self.metrics and isinstance(self.metrics[alt_key], (int, float)):
                    frames = int(self.metrics[alt_key])
                    break
        frames_processed = int(frames) if frames is not None else 0
        t_per_frame = round(elapsed / max(1, frames_processed), 6) if frames_processed > 0 else 0.0
        self.metrics["total_time"] = elapsed
        self.metrics["frames_processed"] = frames_processed
        self.metrics["time_per_frame"] = t_per_frame

        if exc_type is not None:
            self.note("")
            self.note(f"FAILED after {elapsed:.1f}s: {exc_type.__name__}: {exc}")
            self.note("".join(traceback.format_exception(exc_type, exc, tb)))
            # Clear step from pipeline_stats.json on failure
            stats_file = self.workspace / "pipeline_stats.json"
            if stats_file.exists():
                try:
                    data = json.loads(stats_file.read_text(encoding="utf-8"))
                    if self.name in data:
                        del data[self.name]
                        stats_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
                except json.JSONDecodeError:
                    pass
        else:
            self.note("")
            self.note(f"OK in {elapsed:.1f}s")
            update_pipeline_stats(self.workspace, self.name, elapsed, frames_processed, t_per_frame)

        (self.artifacts_dir / f"{self.name}_stats.json").write_text(
            json.dumps(self.metrics, indent=2, default=str), encoding="utf-8")
        if self._log is not None:
            self._log.close()
            self._log = None
        return False  # never swallow the exception
