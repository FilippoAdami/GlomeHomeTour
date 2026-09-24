"""GlomeHomeTour Backend: Ingestion Package Loader.

Parses and validates mobile capture packages (ZIP archive or directory)
against shared/schemas/ specifications.
"""

from __future__ import annotations

import csv
import io
import json
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

import jsonschema
import numpy as np
from PIL import Image

TRAJECTORY_HEADER = "timestamp_ns,tx,ty,tz,qx,qy,qz,qw,tracking,exported"
DEFAULT_MIN_KEYFRAMES = 30


def find_schemas_dir() -> Path:
    """Locate the shared/schemas directory relative to this file or repository root."""
    # Try traversing upwards to repository root
    current = Path(__file__).resolve().parent
    for _ in range(5):
        candidate = current / "shared" / "schemas"
        if candidate.is_dir():
            return candidate
        if (current / "AGENTS.md").exists() or (current / ".git").exists():
            candidate = current / "shared" / "schemas"
            if candidate.is_dir():
                return candidate
        current = current.parent
    raise FileNotFoundError("Could not find shared/schemas directory")


@dataclass
class CameraIntrinsics:
    camera_model: str
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    w: int
    h: int
    camera_angle_x: float
    k1: float
    k2: float
    p1: float
    p2: float


@dataclass
class Keyframe:
    file_path: str
    timestamp_ns: int = 0
    fl_x: float = 0.0
    fl_y: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    transform_matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))  # (4, 4) float64
    image_loader: Callable[[], Image.Image] = field(default_factory=lambda: (lambda: Image.new("RGB", (1, 1))))
    compass_heading_deg: Optional[float] = None

    def load_image(self) -> Image.Image:
        """Load and return PIL Image for this keyframe."""
        return self.image_loader()

    def load_image_rgb(self) -> np.ndarray:
        """Load and return RGB numpy array (H, W, 3) in uint8."""
        img = self.load_image()
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img, dtype=np.uint8)


@dataclass
class TrajectorySample:
    timestamp_ns: int
    tx: float
    ty: float
    tz: float
    qx: float
    qy: float
    qz: float
    qw: float
    tracking: str
    exported: int


@dataclass
class CapturePackage:
    intrinsics: CameraIntrinsics
    keyframes: list[Keyframe]
    trajectory: list[TrajectorySample]
    coverage_summary: dict[str, Any]
    focus_metadata: Optional[dict[str, Any]] = None
    package_path: Optional[Path] = None

    def get_trajectory_array(self) -> np.ndarray:
        """Return trajectory as numpy array: [timestamp_ns, tx, ty, tz, qx, qy, qz, qw]."""
        rows = [
            [s.timestamp_ns, s.tx, s.ty, s.tz, s.qx, s.qy, s.qz, s.qw]
            for s in self.trajectory
        ]
        return np.array(rows, dtype=np.float64)


class PackageValidationError(ValueError):
    """Raised when capture package fails contract validation."""
    pass


class PackageLoader:
    def __init__(self, schemas_dir: Optional[Union[str, Path]] = None, min_keyframes: int = DEFAULT_MIN_KEYFRAMES):
        self.schemas_dir = Path(schemas_dir) if schemas_dir else find_schemas_dir()
        self.min_keyframes = min_keyframes

        self._transforms_schema = self._load_schema("transforms.schema.json")
        self._coverage_schema = self._load_schema("coverage_summary.schema.json")

    def _load_schema(self, schema_filename: str) -> dict[str, Any]:
        path = self.schemas_dir / schema_filename
        if not path.is_file():
            raise FileNotFoundError(f"Schema not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load(self, source: Union[str, Path]) -> CapturePackage:
        """Load and validate capture package from directory or zip file."""
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Capture package not found at: {source_path}")

        if source_path.is_file() and (source_path.suffix.lower() == ".zip" or zipfile.is_zipfile(source_path)):
            return self._load_from_zip(source_path)
        elif source_path.is_dir():
            return self._load_from_dir(source_path)
        else:
            raise PackageValidationError(f"Invalid package format: {source_path}. Expected ZIP archive or directory.")

    def _load_from_dir(self, directory: Path) -> CapturePackage:
        transforms_file = directory / "transforms.json"
        trajectory_file = directory / "trajectory.csv"
        coverage_file = directory / "coverage_summary.json"
        focus_file = directory / "focus_metadata.json"

        if not transforms_file.is_file():
            raise PackageValidationError(f"Missing required transforms.json in {directory}")
        if not trajectory_file.is_file():
            raise PackageValidationError(f"Missing required trajectory.csv in {directory}")
        if not coverage_file.is_file():
            raise PackageValidationError(f"Missing required coverage_summary.json in {directory}")

        transforms_data = self._validate_json(transforms_file, self._transforms_schema, "transforms.json")
        coverage_data = self._validate_json(coverage_file, self._coverage_schema, "coverage_summary.json")

        trajectory_samples = self._parse_and_validate_trajectory(trajectory_file.read_text(encoding="utf-8"))

        focus_data = None
        if focus_file.is_file():
            try:
                focus_data = json.loads(focus_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        intrinsics, keyframes = self._build_keyframes(
            transforms_data,
            image_opener=lambda rel_path: Image.open(directory / rel_path)
        )

        return CapturePackage(
            intrinsics=intrinsics,
            keyframes=keyframes,
            trajectory=trajectory_samples,
            coverage_summary=coverage_data,
            focus_metadata=focus_data,
            package_path=directory,
        )

    def _load_from_zip(self, zip_path: Path) -> CapturePackage:
        with zipfile.ZipFile(zip_path, "r") as z:
            namelist = z.namelist()
            # Normalize root if zipped inside a single folder
            root_prefix = ""
            if "transforms.json" not in namelist:
                candidates = [name for name in namelist if name.endswith("transforms.json")]
                if not candidates:
                    raise PackageValidationError(f"transforms.json not found in ZIP archive {zip_path}")
                root_prefix = candidates[0].rsplit("transforms.json", 1)[0]

            transforms_arc = root_prefix + "transforms.json"
            trajectory_arc = root_prefix + "trajectory.csv"
            coverage_arc = root_prefix + "coverage_summary.json"
            focus_arc = root_prefix + "focus_metadata.json"

            if transforms_arc not in namelist:
                raise PackageValidationError("Missing transforms.json in ZIP archive")
            if trajectory_arc not in namelist:
                raise PackageValidationError("Missing trajectory.csv in ZIP archive")
            if coverage_arc not in namelist:
                raise PackageValidationError("Missing coverage_summary.json in ZIP archive")

            transforms_raw = z.read(transforms_arc).decode("utf-8")
            transforms_data = json.loads(transforms_raw)
            transforms_data.setdefault("schema_version", "1.0.0")
            self._validate_schema(transforms_data, self._transforms_schema, "transforms.json")

            coverage_raw = z.read(coverage_arc).decode("utf-8")
            coverage_data = json.loads(coverage_raw)
            coverage_data.setdefault("schema_version", "1.0.0")
            self._validate_schema(coverage_data, self._coverage_schema, "coverage_summary.json")

            trajectory_raw = z.read(trajectory_arc).decode("utf-8")
            trajectory_samples = self._parse_and_validate_trajectory(trajectory_raw)

            focus_data = None
            if focus_arc in namelist:
                try:
                    focus_data = json.loads(z.read(focus_arc).decode("utf-8"))
                except Exception:
                    pass

            # Create image opener that reads bytes from zip archive on demand
            # Cache the zip bytes or re-open archive when requested
            def open_zip_image(rel_path: str) -> Image.Image:
                full_arc = root_prefix + rel_path
                with zipfile.ZipFile(zip_path, "r") as archive:
                    img_bytes = archive.read(full_arc)
                return Image.open(io.BytesIO(img_bytes))

            intrinsics, keyframes = self._build_keyframes(
                transforms_data,
                image_opener=open_zip_image
            )

            return CapturePackage(
                intrinsics=intrinsics,
                keyframes=keyframes,
                trajectory=trajectory_samples,
                coverage_summary=coverage_data,
                focus_metadata=focus_data,
                package_path=zip_path,
            )

    def _validate_json(self, file_path: Path, schema: dict[str, Any], label: str) -> dict[str, Any]:
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise PackageValidationError(f"Invalid JSON in {label}: {e}")
        data.setdefault("schema_version", "1.0.0")
        self._validate_schema(data, schema, label)
        return data

    def _validate_schema(self, instance: dict[str, Any], schema: dict[str, Any], label: str) -> None:
        validator = jsonschema.Draft202012Validator(schema)
        errors = list(validator.iter_errors(instance))
        if errors:
            first_err = errors[0]
            err_msg = f"{first_err.message} at path: {'/'.join(str(p) for p in first_err.absolute_path)}"
            raise PackageValidationError(f"Validation failed for {label}: {err_msg}")

    def _parse_and_validate_trajectory(self, csv_text: str) -> list[TrajectorySample]:
        lines = [line.strip() for line in csv_text.strip().splitlines() if line.strip()]
        if not lines:
            raise PackageValidationError("trajectory.csv is empty")

        header = lines[0]
        if header != TRAJECTORY_HEADER:
            raise PackageValidationError(
                f"trajectory.csv header {header!r} does not match required contract {TRAJECTORY_HEADER!r}"
            )

        reader = csv.DictReader(lines)
        samples = []
        for idx, row in enumerate(reader):
            try:
                sample = TrajectorySample(
                    timestamp_ns=int(row["timestamp_ns"]),
                    tx=float(row["tx"]),
                    ty=float(row["ty"]),
                    tz=float(row["tz"]),
                    qx=float(row["qx"]),
                    qy=float(row["qy"]),
                    qz=float(row["qz"]),
                    qw=float(row["qw"]),
                    tracking=row["tracking"],
                    exported=int(row["exported"]),
                )
                samples.append(sample)
            except (ValueError, KeyError) as e:
                raise PackageValidationError(f"Malformed trajectory.csv row {idx + 1}: {e}")

        if not samples:
            raise PackageValidationError("trajectory.csv contains no data rows")

        return samples

    def _build_keyframes(
        self,
        transforms_data: dict[str, Any],
        image_opener: Callable[[str], Image.Image],
    ) -> tuple[CameraIntrinsics, list[Keyframe]]:
        intrinsics = CameraIntrinsics(
            camera_model=transforms_data["camera_model"],
            fl_x=float(transforms_data["fl_x"]),
            fl_y=float(transforms_data["fl_y"]),
            cx=float(transforms_data["cx"]),
            cy=float(transforms_data["cy"]),
            w=int(transforms_data["w"]),
            h=int(transforms_data["h"]),
            camera_angle_x=float(transforms_data["camera_angle_x"]),
            k1=float(transforms_data["k1"]),
            k2=float(transforms_data["k2"]),
            p1=float(transforms_data["p1"]),
            p2=float(transforms_data["p2"]),
        )

        frames_data = transforms_data.get("frames", [])
        if len(frames_data) < self.min_keyframes:
            raise PackageValidationError(
                f"Capture package contains only {len(frames_data)} keyframes; "
                f"minimum required is {self.min_keyframes}"
            )

        keyframes = []
        for frame_dict in frames_data:
            rel_path = frame_dict["file_path"]
            mat_4x4 = np.array(frame_dict["transform_matrix"], dtype=np.float64)
            if mat_4x4.shape != (4, 4):
                raise PackageValidationError(f"Invalid transform_matrix shape {mat_4x4.shape} for frame {rel_path}")

            # Capture rel_path in closure
            def make_loader(p: str) -> Callable[[], Image.Image]:
                return lambda: image_opener(p)

            kf = Keyframe(
                file_path=rel_path,
                timestamp_ns=int(frame_dict["timestamp_ns"]),
                fl_x=float(frame_dict["fl_x"]),
                fl_y=float(frame_dict["fl_y"]),
                cx=float(frame_dict["cx"]),
                cy=float(frame_dict["cy"]),
                compass_heading_deg=(float(frame_dict["compass_heading_deg"]) if frame_dict.get("compass_heading_deg") is not None else None),
                transform_matrix=mat_4x4,
                image_loader=make_loader(rel_path),
            )
            keyframes.append(kf)

        return intrinsics, keyframes


def load_package(source: Union[str, Path], min_keyframes: int = DEFAULT_MIN_KEYFRAMES) -> CapturePackage:
    """Convenience function to load a capture package."""
    loader = PackageLoader(min_keyframes=min_keyframes)
    return loader.load(source)
