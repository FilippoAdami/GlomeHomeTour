"""Shared contract for pipeline steps: logging, metrics, state, resume.

Every step wraps its work in a :class:`StepContext`, which owns three files in
the workspace:

* ``<name>_log.txt``   -- human-readable, everything the step printed
* ``<name>_stats.json`` -- machine-readable metrics, keyed however the step likes
* ``pipeline_state.json`` -- ``step -> {status, started, finished, ...}``

On failure the state entry is written as ``failed`` with the traceback in the
log and the workspace is left untouched, so the failing step can be re-run in
isolation against exactly the inputs that broke it.
"""

from __future__ import annotations

import json
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

STATE_FILE = "pipeline_state.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_state(workspace: Path | str) -> dict[str, Any]:
    path = Path(workspace) / STATE_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_state(workspace: Path | str, name: str, entry: dict[str, Any]) -> None:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    state = read_state(workspace)
    state[name] = entry
    (workspace / STATE_FILE).write_text(json.dumps(state, indent=2), encoding="utf-8")


def is_done(workspace: Path | str, name: str, outputs: Iterable[Path | str] = ()) -> bool:
    """True if the step recorded ``ok`` *and* its declared outputs still exist.

    Both halves matter: state alone goes stale the moment someone deletes a
    folder by hand, and outputs alone can't tell a finished step from one that
    crashed after writing its first file.
    """
    entry = read_state(workspace).get(name)
    if not entry or entry.get("status") != "ok":
        return False
    return all(Path(p).exists() for p in outputs)


class StepContext:
    """Context manager owning one step's log, stats and state entry.

    ``workspace`` is where the shared ``pipeline_state.json`` lives (always the
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
        write_state(self.workspace, self.name,
                    {"status": "running", "started": _now(), "finished": None})
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = round(time.time() - self._started, 2)
        self.metrics["elapsed_s"] = elapsed
        if self.timings:
            self.metrics["timings_s"] = self.timings

        if exc_type is not None:
            self.note("")
            self.note(f"FAILED after {elapsed:.1f}s: {exc_type.__name__}: {exc}")
            self.note("".join(traceback.format_exception(exc_type, exc, tb)))
            status = "failed"
        else:
            self.note("")
            self.note(f"OK in {elapsed:.1f}s")
            status = "ok"

        (self.artifacts_dir / f"{self.name}_stats.json").write_text(
            json.dumps(self.metrics, indent=2, default=str), encoding="utf-8")
        write_state(self.workspace, self.name, {
            "status": status,
            "started": _now() if not self._started else
                       datetime.fromtimestamp(self._started, timezone.utc).isoformat(timespec="seconds"),
            "finished": _now(),
            "elapsed_s": elapsed,
        })
        if self._log is not None:
            self._log.close()
            self._log = None
        return False  # never swallow the exception
