# Utilities — Cross-Stage Tooling & Infrastructure

`Utilities` provides shared cross-stage tooling, dynamic import bootstrapping, structured pipeline step orchestration, background queue workers, and benchmarking utilities supporting all reconstruction stages in `backend/`.

---

## Modules & Architecture

| Module / Subdirectory | Role |
| --- | --- |
| [`pipeline_paths.py`](pipeline_paths.py) | **Import Bootstrapper**: Injects all numbered pipeline stage directories (`00_ingestion`, `01_poses_refinment`, etc.) onto `sys.path` so modules can import across stages using flat names without package namespace syntax conflicts. |
| [`pipeline_step.py`](pipeline_step.py) | **Execution Context (`StepContext`)**: Standardizes step execution, timing blocks (`ctx.timer`), JSON metrics recording (`ctx.metric`), step logging (`ctx.note`), and idempotent completion tracking (`is_done`). |
| [`scene_io.py`](scene_io.py) | **Scene Metadata Loader (`load_scene`)**: Unified dataset reader that parses `transforms.json`, images, and camera intrinsics into typed data structures. |
| [`step_extract.py`](step_extract.py) | Utility to unpack and extract mobile captures into standard scene workspaces. |
| [`run_full_benchmark.py`](run_full_benchmark.py) | End-to-end benchmark suite executing and timing all stages on sample scenes with GPU profiling. |
| [`worker/`](worker/) | **Asynchronous Job Worker**: Redis / RQ task consumer (`tasks.py`) running GPU reconstruction jobs in headless background worker daemons. |
| [`third_party/`](third_party/) | Vendored external dependencies, including Depth Anything v3 streaming modules. |

---

## Cross-Stage Imports & Bootstrapping

Because Python syntax does not allow numeric package identifiers (e.g. `import 00_ingestion` is invalid), every entry point script and `conftest.py` invokes:
```python
from Utilities.pipeline_paths import bootstrap
bootstrap()
```
This dynamically prepends every stage directory to `sys.path`, allowing flat module imports such as `from package_loader import CameraIntrinsics` across any stage.

---

## Pipeline Step Context & Idempotency

Stages use `StepContext` to guarantee consistent logs, metrics, and completion checks:
```python
from Utilities.pipeline_step import StepContext, is_done

outputs = [workspace / "depth" / "points3D_depth.ply"]
if not force and is_done(workspace, "depth", outputs):
    print("[depth] already done, skipping")
    return 0

with StepContext("depth", workspace) as ctx:
    with ctx.timer("inference"):
        ...
    ctx.metric("surfels", 500000)
    ctx.note("Completed depth estimation")
```
This writes `<workspace>/depth_log.txt` and `<workspace>/depth_metrics.json` automatically.

---

## Worker Daemon Execution

To run backend processing asynchronously in response to mobile uploads:
```bash
# Start the Redis worker:
python -m Utilities.worker.tasks
```
