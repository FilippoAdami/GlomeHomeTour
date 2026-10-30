"""Standalone capture ingestion with bounded, allow-listed archive handling."""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import math
import os
import re
import shutil
import stat
import struct
import time
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import jsonschema
import numpy as np
from PIL import Image, ImageOps

from package_loader import PackageLoader, PackageValidationError, find_schemas_dir
from pose_aligner import PoseAligner, matrix_to_quaternion, quaternion_to_matrix
from quality_gate import QualityGate
from keyframe_selector import DynamicKeyframeSelector, estimate_scene_depths
from Utilities.scene_io import write_scene

MAX_ENTRIES = 20_000
MAX_TOTAL_BYTES = 4 * 1024**3
MAX_ARCHIVE_BYTES = 4 * 1024**3
MAX_FILE_BYTES = 512 * 1024**2
MAX_TEXT_BYTES = 128 * 1024**2
MAX_COMPRESSION_RATIO = 200
MAX_IMAGE_PIXELS = 100_000_000
MAX_GRID_CELLS = 20_000_000
MAX_GRID_VISITS = 50_000_000

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
IMAGE_FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP",
                 ".tif": "TIFF", ".tiff": "TIFF", ".bmp": "BMP"}
TEXT_EXTENSIONS = {".json", ".csv", ".txt", ".xml", ".yaml", ".yml", ".md", ".log", ".ini", ".toml"}
DEPTH_EXTENSIONS = {".npy", ".npz", ".pfm", ".exr", ".depth", ".dpt", ".las", ".laz", ".pcd", ".pts", ".xyz", ".e57", ".ply"}
COLMAP_BINARIES = {"cameras.bin", "images.bin", "points3d.bin"}
COLMAP_TEXT = {"cameras.txt", "images.txt", "points3d.txt"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | TEXT_EXTENSIONS | DEPTH_EXTENSIONS
COLMAP_MODELS = {
    "SIMPLE_PINHOLE": 3, "PINHOLE": 4, "SIMPLE_RADIAL": 4, "RADIAL": 5,
    "OPENCV": 8, "OPENCV_FISHEYE": 8, "FULL_OPENCV": 12, "FOV": 5,
    "SIMPLE_RADIAL_FISHEYE": 4, "RADIAL_FISHEYE": 5, "THIN_PRISM_FISHEYE": 12,
}
COLMAP_MODEL_NAMES = tuple(COLMAP_MODELS)


class IngestionError(RuntimeError):
    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


class UnsafeArchiveError(IngestionError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__("UNSAFE_ARCHIVE", message, details=details)


def _safe_parts(name: str) -> tuple[str, ...]:
    if (not name or "\0" in name or "\\" in name or name.startswith("/")
            or any(ord(char) < 32 or ord(char) == 127 for char in name)):
        raise UnsafeArchiveError(f"Unsafe path in input: {name!r}")
    raw = name[:-1] if name.endswith("/") else name
    parts = tuple(raw.split("/"))
    if not parts or any(part in {"", ".", ".."} or ":" in part for part in parts):
        raise UnsafeArchiveError(f"Unsafe path in input: {name!r}")
    return parts


def _allowed_file(name: str) -> bool:
    path = PurePosixPath(name)
    base = path.name.casefold()
    ext = path.suffix.casefold()
    if base in COLMAP_BINARIES or base in COLMAP_TEXT:
        return True
    return ext in ALLOWED_EXTENSIONS


def _check_magic(name: str, prefix: bytes) -> None:
    if prefix.startswith((b"#!", b"\x7fELF", b"MZ", b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe",
                          b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe")):
        raise UnsafeArchiveError(f"Executable or script content is not allowed: {name}")
    if not _allowed_file(name):
        raise UnsafeArchiveError(f"Unrecognized file type is not allowed: {name}")


def _validate_zip(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    infos = archive.infolist()
    if len(infos) > MAX_ENTRIES:
        raise UnsafeArchiveError(f"Archive contains too many entries ({len(infos)} > {MAX_ENTRIES})")
    seen: set[str] = set()
    file_paths: set[str] = set()
    path_spellings: dict[str, str] = {}
    total = 0
    files = []
    for info in infos:
        parts = _safe_parts(info.filename)
        key = unicodedata.normalize("NFC", "/".join(parts)).casefold()
        if key in seen:
            raise UnsafeArchiveError(f"Duplicate archive path: {info.filename}")
        seen.add(key)
        for index in range(1, len(parts) + 1):
            spelling = unicodedata.normalize("NFC", "/".join(parts[:index]))
            folded = spelling.casefold()
            if folded in path_spellings and path_spellings[folded] != spelling:
                raise UnsafeArchiveError(f"Case-insensitive path collision in archive: {info.filename}")
            path_spellings[folded] = spelling
        mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(mode)
        if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise UnsafeArchiveError(f"Links or special files are not allowed: {info.filename}")
        if (info.is_dir() and file_type == stat.S_IFREG) or (not info.is_dir() and file_type == stat.S_IFDIR):
            raise UnsafeArchiveError(f"Inconsistent file type in archive: {info.filename}")
        if info.flag_bits & 1:
            raise UnsafeArchiveError(f"Encrypted archive entries are not allowed: {info.filename}")
        if info.is_dir():
            continue
        if mode & 0o111:
            raise UnsafeArchiveError(f"Executable file permissions are not allowed: {info.filename}")
        if not _allowed_file(info.filename):
            raise UnsafeArchiveError(f"Unrecognized file type is not allowed: {info.filename}")
        file_paths.add(key)
        if info.file_size > MAX_FILE_BYTES:
            raise UnsafeArchiveError(f"Input file is too large: {info.filename}")
        total += info.file_size
        if total > MAX_TOTAL_BYTES:
            raise UnsafeArchiveError("Archive expands beyond the 4 GiB ingestion limit")
        if info.file_size and (not info.compress_size or info.file_size / info.compress_size > MAX_COMPRESSION_RATIO):
            raise UnsafeArchiveError(f"Suspicious compression ratio: {info.filename}")
        if PurePosixPath(info.filename).suffix.casefold() in TEXT_EXTENSIONS and info.file_size > MAX_TEXT_BYTES:
            raise UnsafeArchiveError(f"Metadata/text file is too large: {info.filename}")
        try:
            with archive.open(info) as member:
                _check_magic(info.filename, member.read(16))
        except NotImplementedError as exc:
            raise UnsafeArchiveError(f"Unsupported ZIP compression method: {info.filename}") from exc
        files.append(info)
    for info in infos:
        parts = _safe_parts(info.filename)
        for index in range(1, len(parts)):
            parent = unicodedata.normalize("NFC", "/".join(parts[:index])).casefold()
            if parent in file_paths:
                raise UnsafeArchiveError(f"A file is used as a directory in the archive: {info.filename}")
    return files


def _preflight_zip_kind(infos: list[zipfile.ZipInfo]) -> None:
    groups: dict[str, set[str]] = {}
    has_images = False
    for info in infos:
        path = PurePosixPath(info.filename)
        groups.setdefault(path.parent.as_posix(), set()).add(path.name.casefold())
        has_images |= path.suffix.casefold() in IMAGE_EXTENSIONS
    if not has_images:
        if any(PurePosixPath(info.filename).suffix.casefold() in DEPTH_EXTENSIONS for info in infos):
            raise IngestionError("LIDAR_IN_DEVELOPMENT", "LIDAR input is detected but support is in development")
        raise IngestionError("MISSING_IMAGES", "ZIP archive does not contain image files")
    colmap = [(directory, suffix) for directory, names in groups.items() for suffix in ("txt", "bin")
              if {f"cameras.{suffix}", f"images.{suffix}", f"points3d.{suffix}"}.issubset(names)]
    if colmap:
        if len({directory for directory, _ in colmap}) > 1:
            raise IngestionError("AMBIGUOUS_POSES", "ZIP archive contains multiple complete COLMAP models")
        return
    arcore = [directory for directory, names in groups.items()
              if {"transforms.json", "trajectory.csv", "coverage_summary.json"}.issubset(names)]
    scan_roots = {PurePosixPath(directory).parent.as_posix() for directory in arcore}
    if len(arcore) > 1 and (not all(re.fullmatch(r"scan_\d+", PurePosixPath(path).name) for path in arcore)
                            or len(scan_roots) != 1):
        raise IngestionError("AMBIGUOUS_FORMAT", "ZIP archive contains multiple ARCore capture roots")
    if arcore:
        return
    names = [info.filename.casefold() for info in infos]
    if any("artools" in name or "arkit" in name for name in names):
        raise IngestionError("FORMAT_IN_DEVELOPMENT", "ARKit/ARTools input is detected but support is in development")
    if any(PurePosixPath(info.filename).suffix.casefold() in DEPTH_EXTENSIONS
           for info in infos):
        raise IngestionError("LIDAR_IN_DEVELOPMENT", "LIDAR input is detected but support is in development")
    raise IngestionError("FORMAT_IN_DEVELOPMENT", "Input pose/metadata format is not supported yet")


def _extract_zip(source: Path, destination: Path) -> None:
    try:
        with zipfile.ZipFile(source) as archive:
            infos = _validate_zip(archive)
            _preflight_zip_kind(infos)
            written = 0
            for info in infos:
                target = destination.joinpath(*_safe_parts(info.filename))
                target.parent.mkdir(parents=True, exist_ok=True)
                size = 0
                with archive.open(info) as src, target.open("xb") as dst:
                    while chunk := src.read(1024 * 1024):
                        size += len(chunk)
                        written += len(chunk)
                        if size > info.file_size or written > MAX_TOTAL_BYTES:
                            raise UnsafeArchiveError("Archive expanded beyond its declared safe size")
                        dst.write(chunk)
                if size != info.file_size:
                    raise IngestionError("INVALID_ARCHIVE", f"Truncated archive entry: {info.filename}")
    except zipfile.BadZipFile as exc:
        raise UnsafeArchiveError(f"Invalid or corrupt ZIP archive: {exc}") from exc


def _copy_directory(source: Path, destination: Path) -> None:
    total = 0
    entries = 0
    seen: set[str] = set()
    for current, dirs, files in os.walk(source, followlinks=False):
        current_path = Path(current)
        if destination == current_path or destination in current_path.parents:
            dirs[:] = []
            continue
        entries += len(dirs)
        if entries > MAX_ENTRIES:
            raise IngestionError("UNSAFE_INPUT", "Input directory contains too many entries")
        for name in list(dirs):
            path = current_path / name
            if path == destination or path in destination.parents:
                dirs.remove(name)
                continue
            if path.is_symlink():
                raise IngestionError("UNSAFE_INPUT", f"Symbolic links are not allowed: {path}")
        for name in files:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise IngestionError("UNSAFE_INPUT", f"Special files are not allowed: {path}")
            if path.stat().st_mode & 0o111:
                raise IngestionError("UNSAFE_INPUT", f"Executable files are not allowed: {path}")
            rel = path.relative_to(source).as_posix()
            _safe_parts(rel)
            key = unicodedata.normalize("NFC", rel).casefold()
            if key in seen:
                raise IngestionError("UNSAFE_INPUT", f"Duplicate case-insensitive input path: {rel}")
            seen.add(key)
            if not _allowed_file(rel):
                raise IngestionError("UNSAFE_INPUT", f"Unrecognized file type is not allowed: {rel}")
            size = path.stat().st_size
            entries += 1
            total += size
            if entries > MAX_ENTRIES or size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                raise IngestionError("UNSAFE_INPUT", "Input directory exceeds ingestion size limits")
            if PurePosixPath(rel).suffix.casefold() in TEXT_EXTENSIONS and size > MAX_TEXT_BYTES:
                raise IngestionError("UNSAFE_INPUT", f"Metadata/text file is too large: {rel}")
            with path.open("rb") as stream:
                _check_magic(rel, stream.read(16))
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def _all_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IngestionError("INVALID_METADATA", f"Cannot read {label}: {exc}") from exc
    if not isinstance(data, dict):
        raise IngestionError("INVALID_METADATA", f"{label} must contain a JSON object")
    return data


def _validate_images(root: Path, relative_paths: list[str], expected_size: tuple[int, int] | None = None,
                     *, decode: bool = True) -> None:
    for rel in relative_paths:
        parts = _safe_parts(rel)
        path = root.joinpath(*parts)
        if not path.is_file():
            raise IngestionError("MISSING_IMAGE", f"Metadata references a missing image: {rel}")
        try:
            with Image.open(path) as image:
                width, height = image.size
                if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
                    raise IngestionError("INVALID_IMAGE", f"Image dimensions are outside safe limits: {rel}")
                if expected_size is not None and image.size != expected_size:
                    raise IngestionError("ORIENTATION_MISMATCH", f"Image dimensions {image.size} do not match metadata {expected_size}: {rel}")
                if image.format != IMAGE_FORMATS[path.suffix.casefold()]:
                    raise IngestionError("INVALID_IMAGE", f"Image extension does not match its encoding: {rel}")
                if decode:
                    image.load()
                else:
                    image.verify()
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError("INVALID_IMAGE", f"Cannot decode image {rel}: {exc}") from exc


def _image_dirs(root: Path) -> list[Path]:
    dirs = set()
    for path in _all_files(root):
        if path.suffix.casefold() in IMAGE_EXTENSIONS:
            current = path.parent
            while current == root or root in current.parents:
                dirs.add(current)
                if current == root:
                    break
                current = current.parent
    return sorted(dirs, key=lambda path: (len(path.parts), path.as_posix()))


def _resolve_images(root: Path, paths: list[str], *, relative_to_root: bool = False) -> tuple[Path, dict[str, Path]]:
    """Find the image directory by its contents and ensure every requested basename is unique."""
    clean = [_safe_parts(path) for path in paths]
    basenames = [parts[-1] for parts in clean]
    if len({name.casefold() for name in basenames}) != len(basenames):
        raise IngestionError("AMBIGUOUS_IMAGES", "Image metadata contains duplicate filenames")
    if relative_to_root:
        if len({parts[0] for parts in clean}) != 1:
            raise IngestionError("AMBIGUOUS_IMAGES", "Image paths do not share one image-folder root")
        mapped = {name: root.joinpath(*parts) for name, parts in zip(paths, clean)}
        if all(path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS for path in mapped.values()):
            return root / clean[0][0], mapped
    candidates = []
    for directory in _image_dirs(root):
        mapped = {}
        okay = True
        for original, parts in zip(paths, clean):
            target = directory / parts[-1]
            if not target.is_file() or target.suffix.casefold() not in IMAGE_EXTENSIONS:
                okay = False
                break
            mapped[original] = target
        if okay:
            candidates.append((directory, mapped))
    if not candidates:
        raise IngestionError("MISSING_IMAGES", "No image folder contains every image referenced by metadata")
    # Prefer the folder explicitly named in paths (normally images/), otherwise require one match.
    preferred = [item for item in candidates if any(
        len(_safe_parts(name)) > 1 and item[0] == root.joinpath(*_safe_parts(name)[:-1])
        for name in paths
    )]
    if len(preferred) == 1:
        return preferred[0]
    if len(candidates) != 1:
        raise IngestionError("AMBIGUOUS_IMAGES", "Multiple image folders match the metadata; cannot choose safely")
    return candidates[0]


def _safe_colmap_name(name: str) -> tuple[str, ...]:
    parts = _safe_parts(name)
    if PurePosixPath(name).suffix.casefold() not in IMAGE_EXTENSIONS:
        raise IngestionError("INVALID_COLMAP_MODEL", f"COLMAP references a non-image file: {name}")
    return parts


def _parse_colmap_text(model_dir: Path) -> tuple[list[str], dict[str, tuple[int, int]]]:
    cameras: dict[int, tuple[int, int]] = {}
    try:
        with (model_dir / "cameras.txt").open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 5:
                    raise ValueError("invalid camera row")
                camera_id, model, width, height = int(fields[0]), fields[1], int(fields[2]), int(fields[3])
                params = [float(value) for value in fields[4:]]
                if model not in COLMAP_MODELS or len(params) != COLMAP_MODELS[model]:
                    raise ValueError(f"invalid camera model/parameters: {model}")
                if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS or not np.isfinite(params).all():
                    raise ValueError("invalid camera dimensions or parameters")
                if camera_id <= 0 or camera_id in cameras:
                    raise ValueError("duplicate camera id")
                cameras[camera_id] = (width, height)
        names = []
        image_sizes = {}
        image_ids = set()
        with (model_dir / "images.txt").open(encoding="utf-8") as stream:
            lines = iter(stream)
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split(maxsplit=9)
                if len(fields) != 10:
                    raise ValueError("invalid image record")
                image_id = int(fields[0])
                qvec = np.asarray([float(value) for value in fields[1:5]], dtype=np.float64)
                tvec = np.asarray([float(value) for value in fields[5:8]], dtype=np.float64)
                camera_id = int(fields[8])
                name = fields[9]
                _safe_colmap_name(name)
                if (image_id <= 0 or image_id in image_ids or camera_id not in cameras or not np.isfinite(qvec).all()
                        or not np.isfinite(tvec).all() or np.linalg.norm(qvec) < 1e-9):
                    raise ValueError("invalid image pose or camera reference")
                image_ids.add(image_id)
                observation_line = next(lines).strip()
                observations = observation_line.split()
                if len(observations) % 3:
                    raise ValueError("malformed image observations")
                for index in range(0, len(observations), 3):
                    if not np.isfinite([float(observations[index]), float(observations[index + 1])]).all():
                        raise ValueError("non-finite image observations")
                    int(observations[index + 2])
                names.append(name)
                image_sizes[name] = cameras[camera_id]
        if not cameras or not names or len(set(names)) != len(names):
            raise ValueError("model has no cameras/images or duplicate image names")
        with (model_dir / "points3D.txt").open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 8 or (len(fields) - 8) % 2:
                    raise ValueError("malformed 3D point row")
                values = [float(value) for value in fields[1:4]] + [float(fields[7])]
                if not np.isfinite(values).all():
                    raise ValueError("non-finite 3D point")
        return names, image_sizes
    except (OSError, UnicodeDecodeError, StopIteration, ValueError, OverflowError) as exc:
        raise IngestionError("INVALID_COLMAP_MODEL", f"Invalid COLMAP text model: {exc}") from exc


def _parse_colmap_binary(model_dir: Path) -> tuple[list[str], dict[str, tuple[int, int]]]:
    try:
        cameras: dict[int, tuple[int, int]] = {}
        with (model_dir / "cameras.bin").open("rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError("truncated camera count")
            count, = struct.unpack("<Q", raw)
            if count > 10_000:
                raise ValueError("too many cameras")
            for _ in range(count):
                raw = stream.read(24)
                if len(raw) != 24:
                    raise ValueError("truncated camera record")
                camera_id, model_id, width, height = struct.unpack("<iiQQ", raw)
                # COLMAP IDs follow its stable enum, not the map's insertion order when extended.
                if not 0 <= model_id < len(COLMAP_MODEL_NAMES):
                    raise ValueError("unknown camera model id")
                param_count = COLMAP_MODELS[COLMAP_MODEL_NAMES[model_id]]
                raw = stream.read(8 * param_count)
                if len(raw) != 8 * param_count:
                    raise ValueError("truncated camera parameters")
                params = struct.unpack("<" + "d" * param_count, raw)
                if camera_id <= 0 or camera_id in cameras or width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS or not np.isfinite(params).all():
                    raise ValueError("invalid/duplicate camera record")
                cameras[camera_id] = (width, height)
            if stream.read(1):
                raise ValueError("unexpected trailing camera bytes")
        names = []
        image_sizes = {}
        image_ids = set()
        with (model_dir / "images.bin").open("rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError("truncated image count")
            count, = struct.unpack("<Q", raw)
            if count > 100_000:
                raise ValueError("too many registered images")
            for _ in range(count):
                raw = stream.read(64)
                if len(raw) != 64:
                    raise ValueError("truncated image record")
                values = struct.unpack("<idddddddi", raw)
                image_id, qvec, tvec, camera_id = values[0], np.asarray(values[1:5]), np.asarray(values[5:8]), values[8]
                name_data = stream.read(4097)
                terminator = name_data.find(b"\0")
                if terminator < 0:
                    raise ValueError("image name is too long")
                name = name_data[:terminator].decode("utf-8")
                stream.seek(terminator + 1 - len(name_data), os.SEEK_CUR)
                _safe_colmap_name(name)
                point_count_raw = stream.read(8)
                if len(point_count_raw) != 8:
                    raise ValueError("truncated point observation count")
                point_count, = struct.unpack("<Q", point_count_raw)
                if point_count > 50_000_000 or point_count * 24 > os.fstat(stream.fileno()).st_size - stream.tell():
                    raise ValueError("invalid image observation count")
                stream.seek(point_count * 24, os.SEEK_CUR)
                if (image_id <= 0 or image_id in image_ids or camera_id not in cameras or not np.isfinite(qvec).all()
                        or not np.isfinite(tvec).all() or np.linalg.norm(qvec) < 1e-9):
                    raise ValueError("invalid image pose or camera reference")
                image_ids.add(image_id)
                names.append(name)
                image_sizes[name] = cameras[camera_id]
            if stream.read(1):
                raise ValueError("unexpected trailing image bytes")
        with (model_dir / "points3D.bin").open("rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError("truncated point count")
            count, = struct.unpack("<Q", raw)
            if count > 10_000_000:
                raise ValueError("too many 3D points")
            size = os.fstat(stream.fileno()).st_size
            for _ in range(count):
                record = stream.read(43)
                if len(record) != 43:
                    raise ValueError("truncated 3D point record")
                point_id, x, y, z, red, green, blue, error = struct.unpack("<QdddBBBd", record)
                track_raw = stream.read(8)
                if len(track_raw) != 8:
                    raise ValueError("truncated track count")
                track_count, = struct.unpack("<Q", track_raw)
                if not np.isfinite([x, y, z, error]).all() or track_count * 8 > size - stream.tell():
                    raise ValueError("invalid 3D point record")
                stream.seek(track_count * 8, os.SEEK_CUR)
            if stream.tell() != size:
                raise ValueError("trailing or malformed 3D point bytes")
        if not cameras or not names or len(set(names)) != len(names):
            raise ValueError("model has no cameras/images or duplicate image names")
        return names, image_sizes
    except (OSError, UnicodeDecodeError, ValueError, struct.error, OverflowError) as exc:
        raise IngestionError("INVALID_COLMAP_MODEL", f"Invalid COLMAP binary model: {exc}") from exc


def _find_colmap_models(root: Path) -> list[tuple[Path, str]]:
    models: dict[Path, set[str]] = {}
    for directory in [root, *(path for path in root.rglob("*") if path.is_dir())]:
        for suffix in ("txt", "bin"):
            names = ("cameras." + suffix, "images." + suffix, "points3D." + suffix)
            if all((directory / name).is_file() for name in names):
                models.setdefault(directory, set()).add(suffix)
    return [(directory, "bin" if "bin" in suffixes else "txt") for directory, suffixes in sorted(models.items())]


def _resolve_colmap_images(root: Path, names: list[str]) -> tuple[Path, dict[str, Path]]:
    candidates = []
    for directory in _image_dirs(root):
        mapped = {}
        okay = True
        for name in names:
            path = directory.joinpath(*_safe_colmap_name(name))
            if not path.is_file():
                okay = False
                break
            mapped[name] = path
        if okay:
            candidates.append((directory, mapped))
    if not candidates:
        raise IngestionError("MISSING_IMAGES", "COLMAP model references images that are not present")
    if len(candidates) != 1:
        raise IngestionError("AMBIGUOUS_IMAGES", "Multiple image roots satisfy the COLMAP model")
    return candidates[0]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_certificate(certificate_path: Path, scan_roots: list[Path]) -> tuple[dict[str, Any], dict[Path, int]]:
    certificate = _read_json(certificate_path, "certificate.json")
    schema_path = find_schemas_dir() / "certificate.schema.json"
    if not schema_path.is_file():
        raise IngestionError("CERTIFICATE_SCHEMA_MISSING", f"Missing certificate schema: {schema_path}")
    try:
        jsonschema.validate(certificate, json.loads(schema_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
        raise IngestionError("INVALID_CERTIFICATE", f"certificate.json failed schema validation: {exc}") from exc
    if certificate["producer"]["name"] != "com.glomehometour.arscan":
        raise IngestionError("INVALID_CERTIFICATE", "certificate.json was not produced by the GlomeHomeTour app")
    required_stages = {"quality_gate", "keyframe_selection", "orientation_normalization", "manifest_consolidation"}
    if not required_stages.issubset(set(certificate["completed_stages"])):
        raise IngestionError("INVALID_CERTIFICATE", "certificate.json does not report all required app stages")
    counts = certificate["frames"]
    orientation = certificate["orientation_state"]
    if (orientation["rotation_applied_deg"] == 90) != orientation["pose_rolled"]:
        raise IngestionError("INVALID_CERTIFICATE", "Certificate rotation and camera-pose roll claims disagree")
    manifest_hashes = certificate["manifest_hashes"]
    image_hashes = certificate["image_hashes"]
    required_manifests = {"transforms.json", "trajectory.csv", "coverage_summary.json"}
    cert_root = certificate_path.parent
    room_inputs: dict[Path, int] = {}
    room_retained: dict[Path, int] = {}
    total_retained = 0
    total_input = 0
    expected_manifest_files: dict[str, Path] = {}
    expected_image_files: dict[str, Path] = {}

    for scene_root in scan_roots:
        transforms = _read_json(scene_root / "transforms.json", "transforms.json")
        _validate_transforms(transforms)
        if (orientation["processed_width"], orientation["processed_height"]) != (
                transforms["w"], transforms["h"]):
            raise IngestionError("INVALID_CERTIFICATE", "Certificate orientation dimensions do not match transforms.json")
        try:
            prefix = scene_root.relative_to(cert_root).as_posix()
        except ValueError as exc:
            raise IngestionError("INVALID_CERTIFICATE", "Certificate must be at or above the scan folders") from exc
        if prefix == ".":
            prefix = ""
        if len(scan_roots) == 1 and f"{prefix}/transforms.json" not in manifest_hashes:
            prefix = ""
        for name in required_manifests | ({"focus_metadata.json"} if (scene_root / "focus_metadata.json").is_file() else set()):
            key = f"{prefix}/{name}" if prefix else name
            expected_manifest_files[key] = scene_root / name
        _, image_sources = _resolve_images(
            scene_root, [frame["file_path"] for frame in transforms["frames"]], relative_to_root=True)
        for rel, path in image_sources.items():
            key = f"{prefix}/{rel}" if prefix else rel
            expected_image_files[key] = path

        retained = len(transforms["frames"])
        room_retained[scene_root] = retained
        total_retained += retained
        coverage = _read_json(scene_root / "coverage_summary.json", "coverage_summary.json")
        exported = coverage.get("frames", {}).get("exported")
        room_input = int(exported) if isinstance(exported, (int, float)) and exported >= retained else retained
        room_inputs[scene_root] = room_input
        total_input += room_input

    if set(manifest_hashes) != set(expected_manifest_files):
        raise IngestionError("INVALID_CERTIFICATE", "Certificate manifest list does not match the scan folders")
    if set(image_hashes) != set(expected_image_files):
        raise IngestionError("INVALID_CERTIFICATE", "Certificate image list does not match the scan folders")
    for key, expected in manifest_hashes.items():
        path = expected_manifest_files[key]
        if not hmac.compare_digest(_hash_file(path), expected):
            raise IngestionError("CERTIFICATE_HASH_MISMATCH", f"Certificate hash does not match {key}")
    for key, expected in image_hashes.items():
        path = expected_image_files[key]
        if not hmac.compare_digest(_hash_file(path), expected):
            raise IngestionError("CERTIFICATE_HASH_MISMATCH", f"Certificate hash does not match image {key}")

    if (counts["retained_frames_count"] != total_retained
            or counts["total_input_frames"] != total_input
            or counts["discarded_frames_count"] != total_input - total_retained):
        raise IngestionError("INVALID_CERTIFICATE", "Certificate frame counts do not match the scan folders")
    return certificate, {root: room_inputs[root] - retained
                         for root, retained in room_retained.items()}


def _find_arcore_roots(root: Path) -> list[Path]:
    return sorted({path.parent for path in root.rglob("transforms.json")
                   if (path.parent / "trajectory.csv").is_file()
                   and (path.parent / "coverage_summary.json").is_file()})


def _find_scan_directories(root: Path) -> list[Path]:
    scans = (path for path in root.rglob("*")
             if path.is_dir() and re.fullmatch(r"scan_\d+", path.name))
    return sorted(scans, key=lambda path: int(path.name.removeprefix("scan_")))


def _find_certificate(scene_root: Path, payload: Path) -> Path | None:
    current = scene_root
    while current == payload or payload in current.parents:
        candidate = current / "certificate.json"
        if candidate.is_file():
            return candidate
        if current == payload:
            break
        current = current.parent
    return None


def _matrix_for_rotation(matrix: list[list[float]], orientation: int) -> list[list[float]]:
    value = np.asarray(matrix, dtype=np.float64)
    if orientation == 6:
        roll = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    elif orientation == 8:
        roll = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    elif orientation == 3:
        roll = np.diag([-1.0, -1.0, 1.0, 1.0])
    else:
        return value.tolist()
    return (value @ roll).tolist()


def _rotate_intrinsics(data: dict[str, Any], orientation: int) -> None:
    w, h = int(data["w"]), int(data["h"])
    if orientation == 6:
        data["fl_x"], data["fl_y"] = data["fl_y"], data["fl_x"]
        data["cx"], data["cy"] = h - data["cy"], data["cx"]
        data["w"], data["h"] = h, w
    elif orientation == 8:
        data["fl_x"], data["fl_y"] = data["fl_y"], data["fl_x"]
        data["cx"], data["cy"] = data["cy"], w - data["cx"]
        data["w"], data["h"] = h, w
    elif orientation == 3:
        data["cx"], data["cy"] = w - data["cx"], h - data["cy"]
    data["camera_angle_x"] = 2.0 * math.atan(data["w"] / (2.0 * data["fl_x"]))


def _prepare_arcore_package(
    scene_root: Path,
    normalized_root: Path,
    transforms: dict[str, Any],
    frame_sources: dict[str, Path],
) -> tuple[dict[str, Any], int]:
    orientation_values = []
    for frame in transforms["frames"]:
        path = frame_sources[frame["file_path"]]
        with Image.open(path) as image:
            width, height = image.size
            exif_orientation = int(image.getexif().get(274, 1))
        if exif_orientation not in {1, 3, 6, 8}:
            raise IngestionError("ORIENTATION_IN_DEVELOPMENT", f"Mirrored/unsupported EXIF orientation {exif_orientation}: {path.name}")
        if (width, height) != (int(transforms["w"]), int(transforms["h"])):
            raise IngestionError("ORIENTATION_MISMATCH", f"Image dimensions disagree with transforms.json: {path.name}")
        orientation_values.append(exif_orientation)
    non_identity = {value for value in orientation_values if value != 1}
    if len(non_identity) > 1 or (non_identity and any(value == 1 for value in orientation_values)):
        raise IngestionError("ORIENTATION_MISMATCH", "The package contains mixed image orientation metadata")
    orientation = next(iter(non_identity), 1)

    normalized_root.mkdir(parents=True, exist_ok=True)
    out_images = normalized_root / "images"
    out_images.mkdir(parents=True)
    for frame in transforms["frames"]:
        original = frame["file_path"]
        parts = _safe_parts(original)
        target_rel = PurePosixPath("images", *parts[1:]) if parts[0] == "images" else PurePosixPath("images", parts[-1])
        target = normalized_root.joinpath(*target_rel.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        source = frame_sources[original]
        if orientation == 1:
            shutil.copyfile(source, target)
        else:
            try:
                with Image.open(source) as image:
                    save_format = image.format or "JPEG"
                    image = ImageOps.exif_transpose(image)
                    options = {"quality": 95, "subsampling": 0} if save_format == "JPEG" else {}
                    image.save(target, format=save_format, **options)
            except Exception as exc:
                raise IngestionError("ORIENTATION_ERROR", f"Could not normalize {source.name}: {exc}") from exc
        frame["file_path"] = target_rel.as_posix()
        if orientation != 1:
            for key in ("fl_x", "fl_y", "cx", "cy"):
                frame[key] = float(frame[key])
            if orientation in {6, 8}:
                frame["fl_x"], frame["fl_y"] = frame["fl_y"], frame["fl_x"]
                old_cx, old_cy = frame["cx"], frame["cy"]
                frame["cx"], frame["cy"] = ((int(transforms["h"]) - old_cy, old_cx)
                                              if orientation == 6 else
                                              (old_cy, int(transforms["w"]) - old_cx))
            elif orientation == 3:
                frame["cx"] = int(transforms["w"]) - frame["cx"]
                frame["cy"] = int(transforms["h"]) - frame["cy"]
            frame["transform_matrix"] = _matrix_for_rotation(frame["transform_matrix"], orientation)
    if orientation != 1:
        _rotate_intrinsics(transforms, orientation)

    for name in ("trajectory.csv", "coverage_summary.json", "focus_metadata.json"):
        src = scene_root / name
        if src.is_file():
            if name == "trajectory.csv" and orientation != 1:
                _roll_trajectory(src, normalized_root / name, orientation)
            else:
                shutil.copyfile(src, normalized_root / name)
    (normalized_root / "transforms.json").write_text(json.dumps(transforms, indent=2), encoding="utf-8")
    return transforms, orientation


def _roll_trajectory(source: Path, target: Path, orientation: int) -> None:
    roll = {6: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64),
            8: np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64),
            3: np.diag([-1.0, -1.0, 1.0])}[orientation]
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw", "tracking", "exported"]:
            raise IngestionError("INVALID_METADATA", "trajectory.csv has an unexpected header")
        fields = reader.fieldnames
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in reader:
                quaternion = np.asarray([float(row[name]) for name in ("qx", "qy", "qz", "qw")], dtype=np.float64)
                if not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) < 1e-9:
                    raise IngestionError("INVALID_POSES", "trajectory.csv contains an invalid quaternion")
                row.update(zip(("qx", "qy", "qz", "qw"), map(str, matrix_to_quaternion(quaternion_to_matrix(quaternion) @ roll))))
                writer.writerow(row)


def _extent_from_trajectory(points: np.ndarray) -> dict[str, Any]:
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
        raise IngestionError("INVALID_POSES", "ARCore poses are empty or non-finite")
    if float(np.max(np.abs(points))) > 1000:
        raise IngestionError("INVALID_POSES", "ARCore trajectory exceeds the 1000 m safety bound")
    floor_y = float(np.percentile(points[:, 1], 1, method="lower"))
    floor_count = 1
    floor_num = 2
    while floor_num < 20:
        if int(np.count_nonzero(points[:, 1] - floor_y > 2.8 * (floor_num - 1) + 0.2)) >= 10:
            floor_count = floor_num
            floor_num += 1
        else:
            break
    areas = []
    max_span_x = max_span_z = 0.0
    cell = 0.15
    radius = 0.8
    for floor in range(floor_count):
        indices = np.clip(((points[:, 1] - floor_y) / 2.5).astype(int), 0, floor_count - 1) == floor
        floor_points = points[indices]
        if not len(floor_points):
            continue
        xs, zs = floor_points[:, 0], floor_points[:, 2]
        if len(xs) >= 5:
            x_span = float(np.percentile(xs, 99, method="lower") - np.percentile(xs, 1, method="lower"))
            z_span = float(np.percentile(zs, 99, method="lower") - np.percentile(zs, 1, method="lower"))
        else:
            x_span, z_span = float(np.ptp(xs)), float(np.ptp(zs))
        max_span_x, max_span_z = max(max_span_x, x_span), max(max_span_z, z_span)
        pad = radius + 0.1
        min_x, max_x = float(xs.min() - pad), float(xs.max() + pad)
        min_z, max_z = float(zs.min() - pad), float(zs.max() + pad)
        grid_w = max(1, math.ceil((max_x - min_x) / cell))
        grid_h = max(1, math.ceil((max_z - min_z) / cell))
        if grid_w * grid_h > MAX_GRID_CELLS:
            raise IngestionError("SCENE_TOO_LARGE", "Trajectory extent exceeds the scene-area safety bound")
        occupied = bytearray(grid_w * grid_h)
        occupied_count = 0
        grid_visits = 0
        for a, b in zip(floor_points[:-1], floor_points[1:]):
            x0, z0 = float(a[0]), float(a[2])
            x1, z1 = float(b[0]), float(b[2])
            gx0 = max(0, int((min(x0, x1) - radius - min_x) / cell))
            gx1 = min(grid_w - 1, int((max(x0, x1) + radius - min_x) / cell))
            gz0 = max(0, int((min(z0, z1) - radius - min_z) / cell))
            gz1 = min(grid_h - 1, int((max(z0, z1) + radius - min_z) / cell))
            grid_visits += max(0, gx1 - gx0 + 1) * max(0, gz1 - gz0 + 1)
            if grid_visits > MAX_GRID_VISITS:
                raise IngestionError("SCENE_TOO_LARGE", "Trajectory density exceeds the scene-area processing limit")
            dx, dz = x1 - x0, z1 - z0
            length2 = dx * dx + dz * dz
            for gz in range(gz0, gz1 + 1):
                cz = min_z + (gz + 0.5) * cell
                for gx in range(gx0, gx1 + 1):
                    idx = gz * grid_w + gx
                    if occupied[idx]:
                        continue
                    cx = min_x + (gx + 0.5) * cell
                    t = 0.0 if length2 <= 1e-7 else min(1.0, max(0.0, ((cx - x0) * dx + (cz - z0) * dz) / length2))
                    if (cx - (x0 + t * dx)) ** 2 + (cz - (z0 + t * dz)) ** 2 <= radius * radius:
                        occupied[idx] = 1
                        occupied_count += 1
        areas.append(max(occupied_count * cell * cell, 9.0))
    total_area = max(sum(areas), 9.0)
    height = float(np.percentile(points[:, 1], 99, method="lower") - np.percentile(points[:, 1], 1, method="lower"))
    return {
        "width_m": max_span_x,
        "depth_m": max_span_z,
        "height_m": max(0.0, height),
        "floor_area_m2": total_area,
        "floor_count": floor_count,
        "method": "arcore_trajectory_dilation_0.8m_0.15m_grid",
    }


def _budget(area_m2: float, image_count: int) -> dict[str, int]:
    raw_min = math.ceil(50 + 8 * area_m2)
    raw_max = math.ceil(50 + 11 * area_m2)
    low = min(raw_min, image_count)
    high = min(max(raw_max, low), image_count)
    target = min(image_count, max(2, (low + high) // 2)) if image_count else 0
    return {"base_keyframes": 50, "keyframes_per_m2_min": 8, "keyframes_per_m2_max": 11,
            "minimum": low, "maximum": high, "target": target}


def _safe_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _safe_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def _copy_auxiliary_metadata(scene_root: Path, stage_dir: Path,
                             *, excluded: set[Path] | frozenset[Path] = frozenset()) -> list[str]:
    lidar_assets = []
    lidar_dir = stage_dir / "lidar_data"
    for path in _all_files(scene_root):
        resolved = path.resolve()
        if resolved in excluded or path.suffix.casefold() in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(scene_root)
        if relative.as_posix() in {"ingestion_results.json"}:
            continue
        ext = path.suffix.casefold()
        is_lidar = ext in DEPTH_EXTENSIONS
        if is_lidar:
            dest = lidar_dir / relative
            _safe_copy(path, dest)
            lidar_assets.append(dest.relative_to(stage_dir).as_posix())
        elif path.name not in {"transforms.json", "trajectory.csv", "coverage_summary.json", "focus_metadata.json", "certificate.json"}:
            _safe_copy(path, stage_dir / "source_assets" / relative)
    return sorted(lidar_assets)


def _copy_arcore_images(frame_sources: dict[str, Path], transforms: dict[str, Any],
                        output_images: Path, keep_paths: set[str]) -> None:
    for frame in transforms["frames"]:
        rel = frame["file_path"]
        if rel not in keep_paths:
            continue
        target = output_images / Path(*_safe_parts(rel)[1:])
        _safe_copy(frame_sources[rel], target)


def _write_trajectory(source: Path, target: Path, retained_timestamps: set[int]) -> None:
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw", "tracking", "exported"]:
            raise IngestionError("INVALID_METADATA", "trajectory.csv has an unexpected header")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=reader.fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in reader:
                try:
                    timestamp = int(row["timestamp_ns"])
                except (KeyError, ValueError, TypeError) as exc:
                    raise IngestionError("INVALID_METADATA", "trajectory.csv contains a malformed timestamp") from exc
                row["exported"] = "1" if timestamp in retained_timestamps else "0"
                writer.writerow(row)


def _filter_focus_metadata(source: Path, target: Path, retained_paths: set[str], retained_times: set[int]) -> None:
    data = _read_json(source, "focus_metadata.json")
    if isinstance(data.get("frames"), list):
        def retained(frame: Any) -> bool:
            if not isinstance(frame, dict):
                return False
            if "file_path" in frame:
                return frame["file_path"] in retained_paths
            try:
                return int(frame["timestamp_ns"]) in retained_times
            except (KeyError, ValueError, TypeError) as exc:
                raise IngestionError("INVALID_METADATA", "focus_metadata.json contains a malformed frame") from exc

        data["frames"] = [frame for frame in data["frames"] if retained(frame)]
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _validate_transforms(transforms: dict[str, Any]) -> None:
    schema_path = find_schemas_dir() / "transforms.schema.json"
    try:
        jsonschema.validate(transforms, json.loads(schema_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
        raise IngestionError("INVALID_METADATA", f"transforms.json failed schema validation: {exc}") from exc


def _validate_arcore_package(package_root: Path) -> tuple[Any, list[Any]]:
    try:
        package = PackageLoader(min_keyframes=1).load(package_root)
        if not package.trajectory:
            raise PackageValidationError("trajectory.csv contains no samples")
        trajectory = package.get_trajectory_array()
        if not np.isfinite(trajectory[:, 1:]).all() or np.any(np.diff(trajectory[:, 0]) <= 0):
            raise PackageValidationError("trajectory.csv poses must be finite and timestamps strictly increasing")
        quaternions = trajectory[:, 4:8]
        if np.any(np.linalg.norm(quaternions, axis=1) < 1e-9):
            raise PackageValidationError("trajectory.csv contains a zero quaternion")
        for keyframe in package.keyframes:
            matrix = keyframe.transform_matrix
            if not np.isfinite(matrix).all() or np.max(np.abs(matrix[:3, 3])) > 1000:
                raise PackageValidationError(f"invalid pose for {keyframe.file_path}")
        return package, package.keyframes
    except (PackageValidationError, ValueError, KeyError, TypeError) as exc:
        raise IngestionError("INVALID_ARCORE_PACKAGE", str(exc)) from exc


def _copy_standard_arcore_metadata(scene_root: Path, stage_dir: Path, transforms: dict[str, Any],
                                   selected_frames: list[dict[str, Any]], *, certificate: Path | None = None,
                                   normalized_root: Path | None = None, copy_certificate: bool = True) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    working_root = normalized_root or scene_root
    # App-certified captures are immutable pass-through inputs; preserve manifest bytes exactly.
    if certificate is not None:
        for name in ("transforms.json", "trajectory.csv", "coverage_summary.json", "focus_metadata.json"):
            source = scene_root / name
            if source.is_file():
                _safe_copy(source, stage_dir / name)
        if copy_certificate:
            _safe_copy(certificate, stage_dir / "certificate.json")
        return

    header = {key: value for key, value in transforms.items() if key != "frames"}
    write_scene(stage_dir, header, selected_frames)
    trajectory_source = working_root / "trajectory.csv"
    retained_times = {int(frame["timestamp_ns"]) for frame in selected_frames}
    _write_trajectory(trajectory_source, stage_dir / "trajectory.csv", retained_times)
    coverage_source = working_root / "coverage_summary.json"
    coverage = _read_json(coverage_source, "coverage_summary.json")
    if isinstance(coverage.get("frames"), dict):
        coverage["frames"]["exported"] = len(selected_frames)
        (stage_dir / "coverage_summary.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    else:
        _safe_copy(coverage_source, stage_dir / "coverage_summary.json")
    focus_source = working_root / "focus_metadata.json"
    if focus_source.is_file():
        _filter_focus_metadata(focus_source, stage_dir / "focus_metadata.json",
                               {frame["file_path"] for frame in selected_frames}, retained_times)


def _write_results(output_root: Path, result: dict[str, Any], started: float) -> None:
    result["elapsed_ms"] = max(0, int((time.perf_counter() - started) * 1000))
    schema_path = find_schemas_dir() / "ingestion_results.schema.json"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.validate(result, schema)
    except (OSError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
        raise IngestionError("INVALID_INGESTION_RESULT", f"Generated ingestion result is invalid: {exc}") from exc
    target = output_root / "ingestion_results.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(temporary, target)


def _ingest_colmap(payload: Path, model_dir: Path, model_format: str, output_root: Path,
                   started: float) -> dict[str, Any]:
    names, image_sizes = (_parse_colmap_text(model_dir) if model_format == "txt"
                          else _parse_colmap_binary(model_dir))
    image_root, image_map = _resolve_colmap_images(payload, names)
    for name, path in image_map.items():
        with Image.open(path) as image:
            if image.size != image_sizes[name]:
                raise IngestionError("INVALID_COLMAP_MODEL", f"COLMAP camera dimensions do not match image {name}")
    output_images = output_root / "images"
    for source in sorted(path for path in image_root.rglob("*") if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS):
        relative = source.relative_to(image_root)
        _validate_images(source.parent, [source.name])
        _safe_copy(source, output_images / relative)

    stage_dir = output_root
    model_output = stage_dir / "colmap_model"
    for source in model_dir.rglob("*"):
        if source.is_file():
            if source.suffix.casefold() not in ALLOWED_EXTENSIONS and source.name.casefold() not in COLMAP_BINARIES:
                raise IngestionError("UNSAFE_INPUT", f"Unrecognized file in COLMAP model: {source.name}")
            _safe_copy(source, model_output / source.relative_to(model_dir))
    excluded = {path.resolve() for path in model_dir.rglob("*") if path.is_file()}
    lidar_assets = _copy_auxiliary_metadata(payload, stage_dir, excluded=excluded)
    image_count = sum(1 for p in image_root.rglob("*") if p.is_file() and p.suffix.casefold() in IMAGE_EXTENSIONS)
    result = {
        "schema_version": "1.0.0", "status": "ready", "source_format": "colmap",
        "processing_status": "skipped_colmap", "poses_format": "colmap",
        "images_path": "images", "metadata_path": "colmap_model",
        "transforms_path": None, "colmap_model_path": "colmap_model",
        "scene_dimensions": None, "keyframe_budget": None,
        "input_images_count": image_count,
        "quality_rejected_count": 0, "selection_removed_count": 0,
        "retained_images_count": image_count,
        "app_discarded_count": 0, "orientation": {"rotation_applied_deg": 0, "status": "not_changed"},
        "lidar_data_path": "lidar_data" if lidar_assets else None,
        "lidar_status": "in_development" if lidar_assets else "not_detected",
        "lidar_assets": lidar_assets, "elapsed_ms": 0, "warnings": [],
    }
    _write_results(output_root, result, started)
    return result


def _ingest_arcore(payload: Path, scene_root: Path, output_root: Path, started: float, *,
                   certificate: dict[str, Any] | None = None,
                   certificate_path: Path | None = None,
                   app_discarded_count: int = 0,
                   copy_certificate: bool = True) -> dict[str, Any]:
    transforms_path = scene_root / "transforms.json"
    transforms = _read_json(transforms_path, "transforms.json")
    if transforms.get("camera_model") not in (None, "OPENCV"):
        raise IngestionError("FORMAT_IN_DEVELOPMENT", f"Camera model {transforms['camera_model']!r} is not supported yet")
    _validate_transforms(transforms)
    if any(abs(float(transforms[key])) > 0 for key in ("k1", "k2", "p1", "p2")):
        raise IngestionError("UNSUPPORTED_DISTORTION", "Non-zero lens distortion is unsupported by the current COLMAP handoff")
    paths = [frame["file_path"] for frame in transforms["frames"]]
    if not paths:
        raise IngestionError("INVALID_ARCORE_PACKAGE", "transforms.json contains no images")
    _, sources = _resolve_images(scene_root, paths, relative_to_root=True)
    image_sources = list(sources.values())
    if len({path.resolve() for path in image_sources}) != len(paths):
        raise IngestionError("AMBIGUOUS_IMAGES", "Multiple metadata frames resolve to the same image")
    expected_size = (int(transforms["w"]), int(transforms["h"]))
    for path in image_sources:
        _validate_images(path.parent, [path.name], expected_size, decode=certificate is None)

    stage_dir = output_root
    output_images = output_root / "images"
    stage_dir.mkdir(parents=True, exist_ok=True)
    if certificate is not None:
        # A certificate is a pass-through marker; validate it, then preserve its processed payload.
        _validate_arcore_package(scene_root)
        for frame in transforms["frames"]:
            _safe_link(sources[frame["file_path"]], output_images / Path(*_safe_parts(frame["file_path"])[1:]))
        _copy_standard_arcore_metadata(scene_root, stage_dir, transforms, transforms["frames"],
                                       certificate=certificate_path, copy_certificate=copy_certificate)
        lidar_assets = _copy_auxiliary_metadata(scene_root, stage_dir)
        centers = np.asarray([np.asarray(frame["transform_matrix"], dtype=np.float64)[:3, 3]
                              for frame in transforms["frames"]])
        dimensions = _extent_from_trajectory(centers)
        budget = _budget(dimensions["floor_area_m2"], len(transforms["frames"]))
        result = {
            "schema_version": "1.0.0", "status": "ready", "source_format": "arcore_app_certified",
            "processing_status": "skipped_certified", "poses_format": "arcore",
            "images_path": "images", "metadata_path": ".", "transforms_path": "transforms.json",
            "colmap_model_path": None, "scene_dimensions": dimensions, "keyframe_budget": budget,
            "input_images_count": len(transforms["frames"]), "quality_rejected_count": 0,
            "selection_removed_count": 0, "retained_images_count": len(transforms["frames"]),
            "app_discarded_count": app_discarded_count,
            "orientation": {"rotation_applied_deg": certificate["orientation_state"]["rotation_applied_deg"],
                            "status": "certified_by_app"},
            "lidar_data_path": "lidar_data" if lidar_assets else None,
            "lidar_status": "in_development" if lidar_assets else "not_detected",
            "lidar_assets": lidar_assets, "elapsed_ms": 0, "warnings": [],
        }
        _write_results(output_root, result, started)
        return result

    normalized_root = output_root / ".normalized"
    transforms, orientation = _prepare_arcore_package(scene_root, normalized_root, transforms, sources)
    package, keyframes = _validate_arcore_package(normalized_root)
    _validate_images(normalized_root, [frame["file_path"] for frame in transforms["frames"]])
    try:
        synchronized = PoseAligner(package.trajectory).synchronize_keyframes(keyframes)
    except Exception as exc:
        raise IngestionError("INVALID_POSES", f"Could not align frame poses to trajectory.csv: {exc}") from exc
    for frame, keyframe in zip(transforms["frames"], synchronized):
        frame["transform_matrix"] = keyframe.transform_matrix.tolist()
    # Frame matrices are camera-to-world transforms; their last column gives camera centers.
    centers = np.asarray([np.asarray(frame["transform_matrix"], dtype=np.float64)[:3, 3]
                          for frame in transforms["frames"]])
    dimensions = _extent_from_trajectory(centers)
    budget = _budget(dimensions["floor_area_m2"], len(keyframes))
    filtered = not (budget["minimum"] <= len(keyframes) <= budget["maximum"])
    quality_rejected = 0
    selected = synchronized
    selector_count = len(synchronized)
    warnings = []
    if filtered:
        gate = QualityGate()
        gate_result = gate.evaluate(synchronized)
        quality_rejected = len(gate_result.discarded_indices)
        accepted = gate_result.accepted_keyframes
        if len(accepted) < budget["minimum"]:
            raise IngestionError("NOT_ENOUGH_IMAGES", "not enough images", details={
                "required_minimum": budget["minimum"], "after_quality_gate": len(accepted),
                "input_images": len(synchronized),
            })
        depths = estimate_scene_depths(accepted, package.intrinsics, rotate_to_portrait=False)
        selection = DynamicKeyframeSelector.for_2dgs_training().select_keyframes(
            accepted, package.intrinsics, min_keyframes=budget["minimum"],
            max_keyframes=budget["maximum"], scene_depths=depths,
        )
        selected = selection.selected_keyframes
        selector_count = len(selected)
        if selector_count < budget["minimum"]:
            raise IngestionError("NOT_ENOUGH_IMAGES", "not enough images", details={
                "required_minimum": budget["minimum"], "after_quality_gate": len(accepted),
                "after_selection": selector_count,
            })
        if selector_count > budget["maximum"]:
            warnings.append("Dynamic selection retained more frames than the target maximum to preserve view overlap.")

    selected_paths = {frame.file_path for frame in selected}
    selected_frames = []
    source_frames = {frame["file_path"]: frame for frame in transforms["frames"]}
    for keyframe in selected:
        row = dict(source_frames[keyframe.file_path])
        row["transform_matrix"] = keyframe.transform_matrix.tolist()
        selected_frames.append(row)
    _copy_arcore_images(sources, transforms, output_images, selected_paths)
    _copy_standard_arcore_metadata(scene_root, stage_dir, transforms, selected_frames,
                                   normalized_root=normalized_root)
    shutil.rmtree(normalized_root)
    lidar_assets = _copy_auxiliary_metadata(scene_root, stage_dir)
    result = {
        "schema_version": "1.0.0", "status": "ready", "source_format": "arcore",
        "processing_status": "processed" if filtered or orientation != 1 else "validated_no_filter_needed",
        "poses_format": "arcore", "images_path": "images", "metadata_path": ".",
        "transforms_path": "transforms.json", "colmap_model_path": None,
        "scene_dimensions": dimensions, "keyframe_budget": budget,
        "input_images_count": len(keyframes), "quality_rejected_count": quality_rejected,
        "selection_removed_count": max(0, len(keyframes) - quality_rejected - selector_count),
        "retained_images_count": len(selected), "app_discarded_count": 0,
        "orientation": {"rotation_applied_deg": {1: 0, 3: 180, 6: 90, 8: 270}[orientation],
                        "status": "normalized" if orientation != 1 else "metadata_matches_images"},
        "lidar_data_path": "lidar_data" if lidar_assets else None,
        "lidar_status": "in_development" if lidar_assets else "not_detected",
        "lidar_assets": lidar_assets, "elapsed_ms": 0, "warnings": warnings,
    }
    _write_results(output_root, result, started)
    return result


def _prepare_input(source: Path, destination: Path) -> Path:
    if not source.exists() or source.is_symlink():
        raise IngestionError("INPUT_NOT_FOUND", f"Input does not exist or is a symlink: {source}")
    if source.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        _copy_directory(source, destination)
        return destination
    if not source.is_file():
        raise IngestionError("INVALID_INPUT", "Input must be a ZIP archive or a directory")
    if not zipfile.is_zipfile(source):
        raise IngestionError("INVALID_INPUT", "Input must be a ZIP archive or a directory")
    if source.stat().st_size > MAX_ARCHIVE_BYTES:
        raise UnsafeArchiveError("ZIP archive exceeds the 4 GiB compressed-size limit")
    destination.mkdir(parents=True, exist_ok=True)
    _extract_zip(source, destination)
    return destination


def _ingest(source: Path, output_root: Path, *, started: float, log=None) -> dict[str, Any]:
    """Extract, classify and publish one ingestion phase into an isolated output tree."""
    payload = output_root.parent / "input"
    _prepare_input(source, payload)
    colmap_models = _find_colmap_models(payload)
    arcore_roots = _find_arcore_roots(payload)
    scan_dirs = _find_scan_directories(payload)
    if len(colmap_models) > 1:
        raise IngestionError("AMBIGUOUS_POSES", "Input contains multiple COLMAP models")
    if scan_dirs:
        if (payload / "transforms.json").is_file() or len({path.parent for path in scan_dirs}) != 1:
            raise IngestionError("AMBIGUOUS_FORMAT", "Scan folders must be sibling folders in one session")
        roots_by_dir = {root: root for root in arcore_roots if re.fullmatch(r"scan_\d+", root.name)}
        if colmap_models:
            raise IngestionError("AMBIGUOUS_FORMAT", "Input mixes COLMAP poses with ARCore scan folders")
        if len(roots_by_dir) != len(scan_dirs) or any(
                not all((scan / name).is_file() for name in ("transforms.json", "trajectory.csv", "coverage_summary.json"))
                for scan in scan_dirs):
            raise IngestionError("INVALID_ARCORE_PACKAGE", "Every scan_N folder must contain ARCore transforms, trajectory, and coverage metadata")
        certificate_paths = {_find_certificate(scan, payload) for scan in scan_dirs}
        if len(certificate_paths) > 1:
            raise IngestionError("INVALID_CERTIFICATE", "A multi-scan session must use one root certificate.json")
        certificate_path = next(iter(certificate_paths))
        certificate = None
        discarded_by_scan: dict[Path, int] = {scan: 0 for scan in scan_dirs}
        if certificate_path is not None:
            certificate, discarded_by_scan = _verify_certificate(certificate_path, scan_dirs)
        results = []
        for scan_index, scan in enumerate(scan_dirs, start=1):
            if log:
                log(f"Processing {scan.name} ({scan_index}/{len(scan_dirs)}).")
            results.append(_ingest_arcore(
                payload, scan, output_root / scan.name, started,
                certificate=certificate, certificate_path=certificate_path,
                app_discarded_count=discarded_by_scan[scan], copy_certificate=False,
            ))
        if certificate_path is not None:
            _safe_copy(certificate_path, output_root / "certificate.json")
        return {
            "source_format": results[0]["source_format"],
            "poses_format": "arcore",
            "retained_images_count": sum(result["retained_images_count"] for result in results),
            "scan_count": len(results),
        }
    if colmap_models:
        model_dir, model_format = colmap_models[0]
        if log:
            log(f"Detected COLMAP poses in {model_dir}; copying image and model paths without processing.")
        return _ingest_colmap(payload, model_dir, model_format, output_root, started)
    if len(arcore_roots) > 1:
        raise IngestionError("AMBIGUOUS_FORMAT", "Input contains multiple ARCore capture roots")
    if arcore_roots:
        if log:
            log(f"Detected ARCore metadata in {arcore_roots[0]}.")
        scene_root = arcore_roots[0]
        certificate_path = _find_certificate(scene_root, payload)
        certificate, discarded = _verify_certificate(certificate_path, [scene_root]) if certificate_path else (None, {})
        return _ingest_arcore(
            payload, scene_root, output_root, started,
            certificate=certificate, certificate_path=certificate_path,
            app_discarded_count=discarded.get(scene_root, 0),
        )

    files = _all_files(payload)
    lower = [path.as_posix().casefold() for path in files]
    if any("artools" in name or "arkit" in name for name in lower):
        raise IngestionError("FORMAT_IN_DEVELOPMENT", "ARKit/ARTools input is detected but support is in development")
    if any(path.suffix.casefold() in DEPTH_EXTENSIONS for path in files):
        raise IngestionError("LIDAR_IN_DEVELOPMENT", "LIDAR input is detected but support is in development")
    raise IngestionError("FORMAT_IN_DEVELOPMENT", "Input pose/metadata format is not supported yet")


def ingest(source: Path, output_root: Path, *, started: float, log=None) -> dict[str, Any]:
    """Delete a rejected ZIP if unsafe content is discovered at any validation layer."""
    try:
        return _ingest(source, output_root, started=started, log=log)
    except UnsafeArchiveError as exc:
        if source.is_file() and source.exists():
            try:
                source.unlink()
                exc.details["source_archive_deleted"] = True
            except OSError as unlink_error:
                exc.details["source_archive_deleted"] = False
                exc.details["delete_error"] = str(unlink_error)
        raise
