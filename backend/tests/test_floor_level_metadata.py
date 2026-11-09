from pathlib import Path
import sys

import numpy as np
import open3d as o3d

STAGE = Path(__file__).resolve().parents[1] / "08_artifacts_generation"
sys.path.insert(0, str(STAGE))

from assign_floor_level_metadata import build_floor_level_metadata


def _write_plane(path: Path, y: float) -> None:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([
        [-1, y, -1], [1, y, -1], [1, y, 1], [-1, y, 1],
    ])
    mesh.triangles = o3d.utility.Vector3iVector([[0, 2, 1], [0, 3, 2]])
    assert o3d.io.write_triangle_mesh(str(path), mesh)


def test_assigns_scan_levels_mezzanines_and_ceiling_ownership(tmp_path):
    alignment = tmp_path / "multi_room_alignment"
    scan0 = alignment / "scans" / "scan_0" / "mesh_elaboration" / "elements"
    scan1 = alignment / "scans" / "scan_1" / "mesh_elaboration" / "elements"
    scan2 = alignment / "scans" / "scan_2" / "mesh_elaboration" / "elements"
    scan0.mkdir(parents=True)
    scan1.mkdir(parents=True)
    scan2.mkdir(parents=True)

    _write_plane(scan0 / "floor_level_000_area_000.ply", 0.0)
    slab = o3d.geometry.TriangleMesh.create_box(width=2, height=0.1, depth=2)
    slab.translate([0, 2.0, 0])
    small_upward_patch = o3d.geometry.TriangleMesh()
    small_upward_patch.vertices = o3d.utility.Vector3dVector([
        [0, 2.0, 0], [0.5, 2.0, 0], [0.5, 2.0, 0.5], [0, 2.0, 0.5],
    ])
    small_upward_patch.triangles = o3d.utility.Vector3iVector([[0, 2, 1], [0, 3, 2]])
    slab += small_upward_patch
    assert o3d.io.write_triangle_mesh(str(scan0 / "slab_level_001_area_000.ply"), slab)
    _write_plane(scan0 / "ceiling_group_000.ply", 3.0)

    _write_plane(scan1 / "floor_level_000_area_000.ply", 3.0)
    _write_plane(scan1 / "ceiling_group_000.ply", 5.0)
    _write_plane(scan2 / "floor_level_000_area_000.ply", 0.5)

    metadata = build_floor_level_metadata(alignment)

    assert metadata["scan_level_count"] == 2
    assert metadata["floor_level_count"] == 3
    names = {(plane["scan_id"], plane["role"], Path(plane["source_mesh"]).name): plane["floor_level_name"]
             for plane in metadata["floor_planes"]}
    assert names[("scan_0", "floor", "floor_level_000_area_000.ply")] == "BASE"
    assert names[("scan_0", "floor", "slab_level_001_area_000.ply")] == "BASEa"
    assert names[("scan_1", "floor", "floor_level_000_area_000.ply")] == "+1"
    assert names[("scan_2", "floor", "floor_level_000_area_000.ply")] == "BASE"
    slab_floor = next(plane for plane in metadata["floor_planes"]
                      if plane["plane_id"] == "scan_0:slab_floor:0")
    assert np.isclose(slab_floor["mean_elevation_m"], 2.1)

    ceilings = {plane["plane_id"]: plane["floor_level_name"] for plane in metadata["ceiling_planes"]}
    assert ceilings["scan_0:slab_ceiling:0"] == "BASE"
    assert ceilings["scan_0:ceiling_group:0"] == "BASEa"
    assert ceilings["scan_1:ceiling_group:0"] == "+1"
