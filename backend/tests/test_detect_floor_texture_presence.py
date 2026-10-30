import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

STAGE_DIR = Path(__file__).resolve().parents[1] / "06_mesh_elaboration"
sys.path.insert(0, str(STAGE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_3DGS_to_mesh"))

from detect_floor_texture_presence import detect_floor_textures
from fill_mesh_holes import export_ply_with_face_attributes


def test_floor_texture_check_uses_observed_floor_scan_triangles_only(tmp_path):
    level_dir = tmp_path / "intermediate_floor_levels"
    level_dir.mkdir()
    axis = np.linspace(0.01, 1.99, 80)
    x, z = np.meshgrid(axis, axis)
    grain = gaussian_filter(np.random.default_rng(7).normal(size=x.shape), sigma=(2, 9))
    grain = (grain - grain.min()) / (grain.max() - grain.min())
    rgb = np.stack([0.38 + 0.32 * grain, 0.22 + 0.2 * grain,
                    0.12 + 0.12 * grain], axis=-1).reshape(-1, 3)
    vertices = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    count = len(axis)
    faces = []
    for row in range(count - 1):
        for col in range(count - 1):
            a = row * count + col
            faces.extend(([a, a + 1, a + count + 1], [a, a + count + 1, a + count]))
    face_roles = np.zeros(len(faces), dtype=np.uint8)
    preview_flags = np.zeros(len(faces), dtype=np.uint8)
    hole_flags = np.zeros(len(faces), dtype=np.uint8)
    area_ids = np.zeros(len(faces), dtype=np.int32)

    # Add a colored ramp and fitted preview far away; neither is floor-texture evidence.
    extra_vertices = np.array([[10, 1, 10], [11, 1, 10], [11, 1, 11], [10, 1, 11]], dtype=float)
    start = len(vertices)
    vertices = np.vstack((vertices, extra_vertices))
    faces.extend(([start, start + 1, start + 2], [start, start + 2, start + 3]))
    face_roles = np.r_[face_roles, [1, 0]]
    preview_flags = np.r_[preview_flags, [0, 1]]
    hole_flags = np.r_[hole_flags, [0, 0]]
    area_ids = np.r_[area_ids, [-1, 0]]
    colors = np.vstack((rgb, np.tile([0.1, 0.1, 0.1], (4, 1))))

    source = level_dir / "floor_level_000_fitted.ply"
    assert export_ply_with_face_attributes(
        source, vertices, np.asarray(faces, dtype=np.int32),
        np.rint(np.clip(colors, 0, 1) * 255).astype(np.uint8),
        face_attributes={
            "surface_role": ("uchar", face_roles),
            "is_fitted_plane": ("uchar", preview_flags),
            "is_hole_fill": ("uchar", hole_flags),
            "floor_area_id": ("int", area_ids),
        },
    )
    (level_dir / "floor_level_intermediate_manifest.json").write_text(json.dumps({
        "levels": [{
            "level": 0,
            "fitted_output_file": source.name,
            "floor_areas": [{"area_id": 0, "plane_equation": [0, 1, 0, 0]}],
        }],
    }))

    output_path = detect_floor_textures(tmp_path)

    result = json.loads(output_path.read_text())
    assert result["levels"][0]["texture_present"]
    assert result["levels"][0]["areas"][0]["support_count"] == len(faces) - 2
