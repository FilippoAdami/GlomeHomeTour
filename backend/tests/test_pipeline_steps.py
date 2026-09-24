#!/usr/bin/env python3
"""The pipeline's two structural guarantees, on a synthetic 5-frame scene.

1. **Nothing is deleted.** A rejected frame is moved, with its camera entry, and
   `merge_back` restores the scene exactly -- so `--force` re-runs see their
   original input and a discard folder is always a loadable scene in its own right.
2. **Resume is a no-op.** A completed step skips; a failed one does not.

Both are cheap to get subtly wrong (a renumbered basename, a half-written
manifest) and expensive to discover at step 4 on a real capture.

    .venv/bin/python tests/test_pipeline_steps.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

_backend = Path(__file__).resolve().parents[1]
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))
from Utilities.pipeline_paths import bootstrap

bootstrap()

from Utilities.pipeline_step import StepContext, is_done, read_pipeline_stats
from Utilities.scene_io import load_scene, merge_back, split_scene, write_scene

N_FRAMES = 5


def make_scene(directory: Path, n: int = N_FRAMES) -> list[str]:
    (directory / "images").mkdir(parents=True, exist_ok=True)
    header = {
        # The schema pins this to OPENCV; distortion is carried but must be zero,
        # since step 2 emits a PINHOLE COLMAP model and refuses anything else.
        "schema_version": "1.0.0", "camera_model": "OPENCV",
        "fl_x": 1400.0, "fl_y": 1400.0, "cx": 960.0, "cy": 540.0,
        "w": 1920, "h": 1080, "camera_angle_x": 1.2,
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
    }
    frames, names = [], []
    for i in range(n):
        name = f"frame_{i:05d}.jpg"
        names.append(name)
        Image.fromarray(np.full((8, 8, 3), i * 20, dtype=np.uint8)).save(directory / "images" / name)
        pose = np.eye(4)
        pose[0, 3] = 0.1 * i
        frames.append({
            "file_path": f"images/{name}", "timestamp_ns": 1_000 + i,
            "fl_x": 1400.0, "fl_y": 1400.0, "cx": 960.0, "cy": 540.0,
            "transform_matrix": pose.tolist(),
        })
    write_scene(directory, header, frames)
    return names


def test_split_preserves_every_frame():
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp) / "ws"
        names = make_scene(ws)
        reject = [names[1], names[3]]

        kept, rejected = split_scene(ws, ws / "discarded", reject)
        assert (kept, rejected) == (3, 2), (kept, rejected)

        # Moved, not deleted: both halves are loadable scenes, and together
        # they still account for every original frame.
        survivors = load_scene(ws)
        discarded = load_scene(ws / "discarded")
        assert sorted(survivors.names + discarded.names) == sorted(names)
        for scene in (survivors, discarded):
            for frame in scene.frames:
                assert (scene.directory / frame["file_path"]).exists(), frame["file_path"]
        # The camera entry travels with the image.
        assert discarded.names == reject, discarded.names


def test_merge_back_restores_exactly():
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp) / "ws"
        names = make_scene(ws)
        before = json.loads((ws / "transforms.json").read_text())

        split_scene(ws, ws / "discarded", [names[0], names[4]])
        restored = merge_back(ws / "discarded", ws)
        assert restored == 2, restored

        after = json.loads((ws / "transforms.json").read_text())
        assert after["frames"] == before["frames"], "frame order or content changed"
        assert {k: v for k, v in after.items() if k != "frames"} == \
               {k: v for k, v in before.items() if k != "frames"}
        assert sorted(p.name for p in (ws / "images").iterdir()) == sorted(names)


def test_basenames_never_renumbered():
    """Depth maps, COLMAP image names and stats keys are all keyed by basename."""
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp) / "ws"
        names = make_scene(ws)
        split_scene(ws, ws / "discarded", [names[0], names[1]])
        assert load_scene(ws).names == names[2:], "surviving frames were renumbered"


def test_completed_step_skips_and_failed_step_does_not():
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp) / "ws"
        make_scene(ws)
        marker = ws / "transforms.json"

        assert not is_done(ws, "demo", [marker]), "unrun step reported done"
        with StepContext("demo", ws) as ctx:
            ctx.metric("ok", True)
        assert is_done(ws, "demo", [marker]), "completed step did not record itself"
        assert "demo" in read_pipeline_stats(ws)

        # A missing output invalidates the record: the stats file alone is not
        # enough, or a hand-deleted output would be silently skipped over.
        marker.unlink()
        assert not is_done(ws, "demo", [marker]), "skipped despite a missing output"

        try:
            with StepContext("boom", ws):
                raise RuntimeError("deliberate")
        except RuntimeError:
            pass
        assert "boom" not in read_pipeline_stats(ws)
        assert not is_done(ws, "boom", []), "failed step reported done"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} passed")
