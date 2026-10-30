#!/usr/bin/env python3
"""Build provisional 3D instance tracks from the current source mesh and frozen masks."""

from pathlib import Path
import subprocess
import sys


BACKEND = Path(__file__).resolve().parents[1]
SCENE = BACKEND / "current_scene"
INPUT = SCENE / "06_mesh_elaboration/new_step"
OUTPUT = INPUT / "outputs"
SEGMENTATION = SCENE / "05_semantic_segmentation"
WORKSPACE = OUTPUT / "panoptic3d_workspace"
RUN = "panoptic3d_current_source_candidate"


def link(source, destination):
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def main():
    stage = WORKSPACE / "06c_semantic_segmentation"
    run_dir = stage / RUN
    if WORKSPACE.exists() or run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite candidate workspace: {WORKSPACE}")

    inputs = stage / "inputs"
    link(INPUT / "original_mesh.ply", inputs / "mesh.ply")
    for name in ("T_mesh_to_raw.npy", "images", "sparse"):
        link(SEGMENTATION / "inputs" / name, inputs / name)
    full = stage / "full362_final_pipeline"
    link(SEGMENTATION / "full362_final_pipeline/npz_data", full / "npz_data")
    link(SEGMENTATION / "full362_final_pipeline/full_pipeline_benchmark_summary.json",
         full / "full_pipeline_benchmark_summary.json")
    link(SCENE / "02_depth_estimation/depth", WORKSPACE / "02_depth_estimation/depth")

    command = [sys.executable, str(BACKEND / "05_semantic_segmentation/run_panoptic3d.py"),
               "--workspace", str(WORKSPACE), "--run-name", RUN, "--stride", "3"]
    for phase in ("evidence", "solve"):
        subprocess.run(command + ["--phase", phase], check=True)
    print(f"Provisional instance tracks: {run_dir}")
    print("This result is for contact review only; do not use it to deform geometry before validation.")


if __name__ == "__main__":
    main()
