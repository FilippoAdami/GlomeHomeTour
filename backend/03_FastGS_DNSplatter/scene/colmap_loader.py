"""Small, dependency-free readers for COLMAP sparse reconstruction files."""
from __future__ import annotations
from collections import namedtuple
from pathlib import Path
import struct
import numpy as np

CameraModel = namedtuple("CameraModel", "model_id model_name num_params")
Camera = namedtuple("Camera", "id model width height params")
BaseImage = namedtuple("Image", "id qvec tvec camera_id name xys point3D_ids")
Point3D = namedtuple("Point3D", "id xyz rgb error image_ids point2D_idxs")
_MODEL_SPECS = (
    (0, "SIMPLE_PINHOLE", 3), (1, "PINHOLE", 4), (2, "SIMPLE_RADIAL", 4),
    (3, "RADIAL", 5), (4, "OPENCV", 8), (5, "OPENCV_FISHEYE", 8),
    (6, "FULL_OPENCV", 12), (7, "FOV", 5), (8, "SIMPLE_RADIAL_FISHEYE", 4),
    (9, "RADIAL_FISHEYE", 5), (10, "THIN_PRISM_FISHEYE", 12),
)
CAMERA_MODELS = {CameraModel(*spec) for spec in _MODEL_SPECS}
CAMERA_MODEL_IDS = {model.model_id: model for model in CAMERA_MODELS}
CAMERA_MODEL_NAMES = {model.model_name: model for model in CAMERA_MODELS}


def qvec2rotmat(qvec):
    """Return a matrix for COLMAP's scalar-first quaternion."""
    q = np.asarray(qvec, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError("qvec must contain four values")
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("qvec must be non-zero")
    w, x, y, z = q / norm
    return np.array(((1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
                     (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
                     (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y))))


def rotmat2qvec(rotation):
    """Convert a 3x3 rotation matrix to a canonical scalar-first quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    left, _, right = np.linalg.svd(matrix)
    matrix = left @ right
    if np.linalg.det(matrix) < 0:
        left[:, -1] *= -1
        matrix = left @ right
    trace = np.trace(matrix)
    if trace > 0:
        s = 2 * np.sqrt(trace + 1)
        q = np.array((.25*s, (matrix[2, 1]-matrix[1, 2])/s,
                      (matrix[0, 2]-matrix[2, 0])/s, (matrix[1, 0]-matrix[0, 1])/s))
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            s = 2 * np.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            q = np.array(((matrix[2, 1]-matrix[1, 2])/s, .25*s,
                          (matrix[0, 1]+matrix[1, 0])/s, (matrix[0, 2]+matrix[2, 0])/s))
        elif axis == 1:
            s = 2 * np.sqrt(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            q = np.array(((matrix[0, 2]-matrix[2, 0])/s, (matrix[0, 1]+matrix[1, 0])/s,
                          .25*s, (matrix[1, 2]+matrix[2, 1])/s))
        else:
            s = 2 * np.sqrt(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            q = np.array(((matrix[1, 0]-matrix[0, 1])/s, (matrix[0, 2]+matrix[2, 0])/s,
                          (matrix[1, 2]+matrix[2, 1])/s, .25*s))
    return q if q[0] >= 0 else -q


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def read_next_bytes(file_handle, num_bytes, format_char_sequence, endian_character="<"):
    data = file_handle.read(num_bytes)
    if len(data) != num_bytes:
        raise ValueError("truncated COLMAP binary file")
    return struct.unpack(endian_character + format_char_sequence, data)


def _data_lines(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def read_points3D_text(path):
    records = [line.split() for line in _data_lines(path)]
    xyzs, rgbs = np.empty((len(records), 3)), np.empty((len(records), 3))
    errors = np.empty((len(records), 1))
    for index, fields in enumerate(records):
        if len(fields) < 8:
            raise ValueError(f"invalid point record at index {index}")
        xyzs[index], rgbs[index], errors[index, 0] = fields[1:4], fields[4:7], fields[7]
    return xyzs, rgbs, errors


def read_points3D_binary(path_to_model_file):
    with open(path_to_model_file, "rb") as handle:
        count = read_next_bytes(handle, 8, "Q")[0]
        xyzs, rgbs = np.empty((count, 3)), np.empty((count, 3))
        errors = np.empty((count, 1))
        for index in range(count):
            record = read_next_bytes(handle, 43, "QdddBBBd")
            xyzs[index], rgbs[index], errors[index, 0] = record[1:4], record[4:7], record[7]
            track_length = read_next_bytes(handle, 8, "Q")[0]
            read_next_bytes(handle, 8 * track_length, "ii" * track_length)
    return xyzs, rgbs, errors


def read_intrinsics_text(path):
    cameras = {}
    for line in _data_lines(path):
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"invalid camera record: {line}")
        camera_id, model = int(fields[0]), fields[1]
        spec = CAMERA_MODEL_NAMES.get(model)
        if spec is None:
            raise ValueError(f"unknown COLMAP camera model: {model}")
        params = np.asarray(fields[4:], dtype=np.float64)
        if len(params) != spec.num_params:
            raise ValueError(f"wrong parameter count for {model}")
        cameras[camera_id] = Camera(camera_id, model, int(fields[2]), int(fields[3]), params)
    return cameras


def read_intrinsics_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as handle:
        count = read_next_bytes(handle, 8, "Q")[0]
        for _ in range(count):
            camera_id, model_id, width, height = read_next_bytes(handle, 24, "iiQQ")
            spec = CAMERA_MODEL_IDS.get(model_id)
            if spec is None:
                raise ValueError(f"unknown COLMAP camera model id: {model_id}")
            params = np.asarray(read_next_bytes(handle, 8 * spec.num_params, "d" * spec.num_params))
            cameras[camera_id] = Camera(camera_id, spec.model_name, width, height, params)
    if len(cameras) != count:
        raise ValueError("duplicate camera ids in COLMAP binary file")
    return cameras


def _read_c_string(handle):
    data = bytearray()
    while True:
        byte = handle.read(1)
        if not byte:
            raise ValueError("truncated image name in COLMAP binary file")
        if byte == b"\0":
            return data.decode("utf-8")
        data.extend(byte)


def read_extrinsics_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as handle:
        count = read_next_bytes(handle, 8, "Q")[0]
        for _ in range(count):
            record = read_next_bytes(handle, 64, "idddddddi")
            name = _read_c_string(handle)
            point_count = read_next_bytes(handle, 8, "Q")[0]
            values = np.asarray(read_next_bytes(handle, 24 * point_count, "ddq" * point_count),
                                dtype=object).reshape(-1, 3)
            images[record[0]] = Image(record[0], np.asarray(record[1:5]), np.asarray(record[5:8]),
                                      record[8], name, values[:, :2].astype(np.float64),
                                      values[:, 2].astype(np.int64))
    return images


def read_extrinsics_text(path):
    images = {}
    with open(path, encoding="utf-8") as handle:
        lines = iter(handle)
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split(maxsplit=9)
            if len(fields) != 10:
                raise ValueError(f"invalid image record: {line}")
            try:
                observations = next(lines).split()
            except StopIteration as error:
                raise ValueError("missing image observations line") from error
            if len(observations) % 3:
                raise ValueError("image observations must be x y point-id triples")
            xys = np.asarray([(float(observations[index]), float(observations[index + 1]))
                              for index in range(0, len(observations), 3)], dtype=np.float64).reshape(-1, 2)
            point_ids = np.asarray([int(observations[index]) for index in range(2, len(observations), 3)],
                                   dtype=np.int64)
            image_id = int(fields[0])
            images[image_id] = Image(image_id, np.asarray(fields[1:5], dtype=np.float64),
                                     np.asarray(fields[5:8], dtype=np.float64), int(fields[8]), fields[9],
                                     xys, point_ids)
    return images


def read_colmap_bin_array(path):
    with open(path, "rb") as handle:
        header = bytearray()
        while header.count(b"&") < 3:
            byte = handle.read(1)
            if not byte:
                raise ValueError("invalid COLMAP dense-array header")
            header.extend(byte)
        width, height, channels = (int(value) for value in header[:-1].split(b"&"))
        values = np.fromfile(handle, dtype=np.float32)
    if values.size != width * height * channels:
        raise ValueError("dense-array payload size does not match its header")
    return values.reshape((width, height, channels), order="F").transpose(1, 0, 2).squeeze()


def _self_check(sample_dir=None):
    quaternion = np.array((.5, -.5, .5, -.5))
    matrix = qvec2rotmat(quaternion)
    assert np.allclose(qvec2rotmat(rotmat2qvec(matrix)), matrix)
    root = Path(sample_dir) if sample_dir else Path(__file__).resolve().parents[2] / "current_scene" / "sparse" / "0"
    cameras, images = read_intrinsics_binary(root / "cameras.bin"), read_extrinsics_binary(root / "images.bin")
    points, colors, _ = read_points3D_binary(root / "points3D.bin")
    assert cameras and images and points.shape == colors.shape and points.shape[1] == 3


if __name__ == "__main__":
    import sys
    _self_check(sys.argv[1] if len(sys.argv) > 1 else None)
    print("COLMAP reader self-check passed")
