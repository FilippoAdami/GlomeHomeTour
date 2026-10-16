# backend/server/pipeline_runner.py
import os
import sys
import json
import time
import uuid
import shutil
import subprocess
from pathlib import Path

from .db import get_connection

BACKEND_DIR = Path(__file__).resolve().parent.parent
PYTHON_EXE = BACKEND_DIR / ".venv" / "bin" / "python"
RUN_PIPELINE_SCRIPT = BACKEND_DIR / "run_pipeline.py"
OUTPUTS_DIR = BACKEND_DIR / "server_storage" / "outputs"


def optimize_3d_assets(workspace: Path, property_id: str) -> tuple[str, str]:
    """
    Decimates the floater-cleaned, sealed mesh for web delivery and preserves the trained Gaussian PLY.
    Returns (mesh_url, splat_url).
    """
    prop_out_dir = OUTPUTS_DIR / property_id
    prop_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Process mesh
    sealed_mesh_ply = workspace / "06c_semantic_segmentation" / "step1_cleaned_mesh" / "mesh_sealed_colored.ply"
    target_obj = prop_out_dir / "mesh_web.obj"
    mesh_url = f"/static/outputs/{property_id}/mesh_web.obj"

    if sealed_mesh_ply.exists():
        try:
            import open3d as o3d
            import trimesh
            mesh = o3d.io.read_triangle_mesh(str(sealed_mesh_ply))
            if len(mesh.triangles) > 150000:
                mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=150000)
            mesh.compute_vertex_normals()
            temp_ply = prop_out_dir / "temp_mesh.ply"
            o3d.io.write_triangle_mesh(str(temp_ply), mesh, write_vertex_colors=True)

            tri_mesh = trimesh.load(str(temp_ply))
            tri_mesh.export(str(target_obj))
            if temp_ply.exists():
                temp_ply.unlink()
        except Exception as e:
            print(f"Error optimizing mesh: {e}", file=sys.stderr)
            # Fallback to serving the full sealed mesh
            mesh_url = f"/static/outputs/{property_id}/mesh.ply"
            shutil.copy2(sealed_mesh_ply, prop_out_dir / "mesh.ply")
    else:
        raise FileNotFoundError(f"No cleaned sealed mesh found at {sealed_mesh_ply}")

    # Keep the Gaussian attributes intact; RGB-only point clouds cannot render as splats.
    splat_dir = workspace / "03_FastGS_DNSplatter"
    iterations = sorted(
        splat_dir.glob("point_cloud/iteration_*/point_cloud.ply"),
        key=lambda path: int(path.parent.name.removeprefix("iteration_")),
    )
    raw_splat_ply = iterations[-1] if iterations else splat_dir / "point_cloud_INRIA.ply"
    target_splat = prop_out_dir / "splat.ply"
    splat_url = f"/static/outputs/{property_id}/splat.ply"

    if raw_splat_ply.is_file():
        from plyfile import PlyData
        import numpy as np

        vertices = PlyData.read(str(raw_splat_ply))["vertex"]
        center = (np.array([vertices[axis].min() for axis in "xyz"]) +
                  np.array([vertices[axis].max() for axis in "xyz"])) / 2
        # Match ThreeViewer's geometry.center() and rotation.x = -PI/2.
        transform = {
            "position": [-float(center[0]), -float(center[2]), float(center[1])],
            "rotation": [-0.7071067811865475, 0, 0, 0.7071067811865476],
        }
        transform_path = prop_out_dir / "splat_transform.json"
        temp_transform = transform_path.with_suffix(".json.tmp")
        temp_transform.write_text(json.dumps(transform))
        os.replace(temp_transform, transform_path)
        temp_splat = target_splat.with_suffix(".ply.tmp")
        shutil.copy2(raw_splat_ply, temp_splat)
        os.replace(temp_splat, target_splat)
    else:
        raise FileNotFoundError(f"No trained Gaussian PLY found under {splat_dir}")

    return mesh_url, splat_url


def execute_pipeline(property_id: str, archive_path: str):
    """
    Executes the real Glome ROCm reconstruction pipeline on the uploaded archive.
    Updates DB stages dynamically during execution.
    """
    workspace = BACKEND_DIR / "workspaces" / property_id
    workspace.mkdir(parents=True, exist_ok=True)

    stages = [
        {"name": "extract", "displayName": "Archive Extraction & Validation", "status": "running"},
        {"name": "colmap", "displayName": "COLMAP Pose Refinement & Alignment", "status": "queued"},
        {"name": "depth", "displayName": "DA3 Monocular Metric Depth & Normals", "status": "queued"},
        {"name": "train", "displayName": "FastGS + DN-Splatter Radiance Training", "status": "queued"},
        {"name": "mesh", "displayName": "TSDF Metric Mesh Reconstruction", "status": "queued"},
        {"name": "mesh_cleanup", "displayName": "Floater Removal & Mesh Sealing", "status": "queued"},
    ]

    job_id = str(uuid.uuid4())
    conn = get_connection()
    conn.execute(
        "INSERT INTO pipeline_jobs (id, property_id, status, stages_json) VALUES (?, ?, 'running', ?)",
        (job_id, property_id, json.dumps(stages))
    )
    conn.commit()
    conn.close()

    step_sequence = ["extract", "colmap", "depth", "train", "mesh", "mesh_cleanup"]

    try:
        for idx, step_name in enumerate(step_sequence):
            # Mark current stage running
            stages[idx]["status"] = "running"
            conn = get_connection()
            conn.execute("UPDATE pipeline_jobs SET stages_json = ? WHERE id = ?", (json.dumps(stages), job_id))
            conn.commit()
            conn.close()

            t_start = time.perf_counter()

            # Execute run_pipeline.py for this specific step
            cmd = [
                str(PYTHON_EXE),
                str(RUN_PIPELINE_SCRIPT),
                archive_path,
                "--workspace", str(workspace),
                "--only-step", step_name,
                "--force"
            ]

            process = subprocess.run(
                cmd,
                cwd=str(BACKEND_DIR),
                capture_output=True,
                text=True
            )

            if process.returncode != 0:
                err_msg = process.stderr.strip() or f"Step {step_name} failed with code {process.returncode}"
                stages[idx]["status"] = "error"
                stages[idx]["description"] = err_msg
                raise RuntimeError(err_msg)

            elapsed = time.perf_counter() - t_start
            stages[idx]["status"] = "done"
            stages[idx]["elapsedSeconds"] = round(elapsed, 2)

            conn = get_connection()
            conn.execute("UPDATE pipeline_jobs SET stages_json = ? WHERE id = ?", (json.dumps(stages), job_id))
            conn.commit()
            conn.close()

        # Decimate mesh and color splats for instant WebGL loading
        mesh_url, splat_url = optimize_3d_assets(workspace, property_id)

        conn = get_connection()
        conn.execute("""
            UPDATE properties
            SET status = 'ready',
                mesh_url = ?,
                splat_url = ?
            WHERE id = ?
        """, (mesh_url, splat_url, property_id))
        conn.execute(
            "UPDATE pipeline_jobs SET status = 'done', completed_at = CURRENT_TIMESTAMP WHERE id = ?",
            (job_id,)
        )
        conn.commit()
        conn.close()

    except Exception as e:
        conn = get_connection()
        conn.execute("UPDATE properties SET status = 'error' WHERE id = ?", (property_id,))
        conn.execute("UPDATE pipeline_jobs SET status = 'error', error = ? WHERE id = ?", (str(e), job_id))
        conn.commit()
        conn.close()
