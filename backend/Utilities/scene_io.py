"""Single owner of "read/modify/write a scene folder".

A *scene folder* is the pair the frozen `transforms.schema.json` describes: an
``images/`` directory plus a ``transforms.json`` whose ``frames[].file_path``
entries are ``images/...`` relative to that same folder. The pipeline workspace
and every one of its discard folders are scene folders, which is why a rejected
frame set stays directly loadable by any existing tool.

Generalised from ``00_ingestion/run_staged_filtering.py::write_stage``.

Frame basenames are never renumbered. A split renames nothing, so depth maps,
COLMAP image names and stats keys stay valid across steps.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import jsonschema
from PIL import Image

from package_loader import CameraIntrinsics, Keyframe, find_schemas_dir

_HEADER_KEYS = (
    "schema_version", "camera_model", "fl_x", "fl_y", "cx", "cy", "w", "h",
    "camera_angle_x", "k1", "k2", "p1", "p2",
)

_schema_cache: Optional[dict[str, Any]] = None


def transforms_schema() -> dict[str, Any]:
    global _schema_cache
    if _schema_cache is None:
        path = find_schemas_dir() / "transforms.schema.json"
        _schema_cache = json.loads(path.read_text(encoding="utf-8"))
    return _schema_cache


@dataclass
class Scene:
    """An ``images/`` + ``transforms.json`` pair on disk."""

    directory: Path
    header: dict[str, Any]      # every transforms.json key except "frames"
    frames: list[dict[str, Any]]

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def images_dir(self) -> Path:
        return self.directory / "images"

    @property
    def names(self) -> list[str]:
        """Frame basenames, in file order."""
        return [Path(f["file_path"]).name for f in self.frames]

    @property
    def intrinsics(self) -> CameraIntrinsics:
        h = self.header
        return CameraIntrinsics(
            camera_model=h["camera_model"],
            fl_x=float(h["fl_x"]), fl_y=float(h["fl_y"]),
            cx=float(h["cx"]), cy=float(h["cy"]),
            w=int(h["w"]), h=int(h["h"]),
            camera_angle_x=float(h["camera_angle_x"]),
            k1=float(h["k1"]), k2=float(h["k2"]),
            p1=float(h["p1"]), p2=float(h["p2"]),
        )

    def keyframes(self) -> list[Keyframe]:
        """Frames as :class:`Keyframe` objects with lazy image loaders.

        Poses are handed back exactly as stored (ARCore camera-to-world), since
        that is what ``QualityGate`` and ``DynamicKeyframeSelector`` expect.
        """
        import numpy as np

        out = []
        for frame in self.frames:
            path = self.directory / frame["file_path"]
            out.append(Keyframe(
                file_path=frame["file_path"],
                timestamp_ns=int(frame["timestamp_ns"]),
                fl_x=float(frame["fl_x"]), fl_y=float(frame["fl_y"]),
                cx=float(frame["cx"]), cy=float(frame["cy"]),
                transform_matrix=np.array(frame["transform_matrix"], dtype=np.float64),
                image_loader=(lambda p=path: Image.open(p).convert("RGB")),
            ))
        return out


def load_scene(directory: Path | str, *, validate: bool = True) -> Scene:
    """Read a scene folder, optionally schema-checking it."""
    directory = Path(directory)
    data = json.loads((directory / "transforms.json").read_text(encoding="utf-8"))
    if validate:
        jsonschema.validate(instance=data, schema=transforms_schema())
    header = {k: v for k, v in data.items() if k != "frames"}
    return Scene(directory=directory, header=header, frames=list(data.get("frames", [])))


def write_scene(
    directory: Path | str,
    header: dict[str, Any],
    frames: Sequence[dict[str, Any]],
    *,
    validate: bool = True,
) -> Path:
    """Write ``transforms.json`` describing exactly ``frames``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = {k: header[k] for k in _HEADER_KEYS}
    data["frames"] = list(frames)
    if validate:
        jsonschema.validate(instance=data, schema=transforms_schema())
    out = directory / "transforms.json"
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def split_scene(
    src_dir: Path | str,
    dst_dir: Path | str,
    reject_names: Iterable[str],
) -> tuple[int, int]:
    """Move rejected frames out of ``src_dir`` into their own scene folder.

    ``reject_names`` are basenames. Returns ``(kept, rejected)``.

    Both manifests are built in memory before anything moves, so an interruption
    leaves ``dst_dir`` a partial but still-loadable scene rather than a
    ``transforms.json`` naming files that are not there.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    scene = load_scene(src_dir)
    rejects = set(reject_names)

    keep_frames, drop_frames = [], []
    for frame in scene.frames:
        (drop_frames if Path(frame["file_path"]).name in rejects else keep_frames).append(frame)

    if drop_frames:
        # Merge with whatever is already in dst_dir, so a re-run that rejects a
        # second batch adds to the pile instead of orphaning the first one.
        existing: list[dict[str, Any]] = []
        if (dst_dir / "transforms.json").exists():
            existing = load_scene(dst_dir).frames
        already = {f["file_path"] for f in existing}

        (dst_dir / "images").mkdir(parents=True, exist_ok=True)
        for frame in drop_frames:
            src = src_dir / frame["file_path"]
            if src.exists():
                shutil.move(str(src), str(dst_dir / frame["file_path"]))
        write_scene(dst_dir, scene.header,
                    existing + [f for f in drop_frames if f["file_path"] not in already])

    write_scene(src_dir, scene.header, keep_frames)
    return len(keep_frames), len(drop_frames)


def merge_back(dst_dir: Path | str, src_dir: Path | str) -> int:
    """Inverse of :func:`split_scene`: return every rejected frame to the scene.

    Used by ``--force`` re-runs so a step always sees the same input it saw the
    first time, and by hand when inspecting why a frame was dropped.
    """
    dst_dir, src_dir = Path(dst_dir), Path(src_dir)
    if not (dst_dir / "transforms.json").exists():
        return 0
    rejected = load_scene(dst_dir)
    if not rejected.frames:
        return 0

    scene = load_scene(src_dir)
    present = {f["file_path"] for f in scene.frames}
    restored = []
    for frame in rejected.frames:
        src = dst_dir / frame["file_path"]
        if src.exists():
            shutil.move(str(src), str(src_dir / frame["file_path"]))
        if frame["file_path"] not in present:
            restored.append(frame)

    frames = scene.frames + restored
    frames.sort(key=lambda f: f["file_path"])
    write_scene(src_dir, scene.header, frames)
    shutil.rmtree(dst_dir, ignore_errors=True)
    return len(restored)
