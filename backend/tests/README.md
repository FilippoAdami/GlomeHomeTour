# Backend Test Suite

Comprehensive pytest test suite covering ingestion, coordinate conventions, COLMAP pose refinement, Depth Anything v3 inference, surfel cloud initialization, ROCm numerical stability, and 2DGS training.

---

## Test Inventory

| Test Module | Coverage |
| --- | --- |
| [`test_depth_initialization.py`](test_depth_initialization.py) | Metric depth scale-shift alignment, surface normals, surfel cloud tangent frames, dynamic saturation masking, and epipolar free-space carving. |
| [`test_sliding_window_depth.py`](test_sliding_window_depth.py) | Multi-view sliding window DA3 depth inference with cross-chunk overlap blending. |
| [`test_step_filter_depth.py`](test_step_filter_depth.py) | Track covisibility keyframe selection and scene median depth calculation. |
| [`test_colmap_poses_to_da3.py`](test_colmap_poses_to_da3.py) | Pose conversions between COLMAP OpenCV $w2c$ and DA3/ARCore OpenGL $c2w$, verifying determinant, orthonormality, and camera centers. |
| [`test_geometry_conventions.py`](test_geometry_conventions.py) | Verification of camera-to-world ($c2w$) and world-to-camera ($w2c$) transformations, focal length flips, and portrait transpositions. |
| [`test_ingestion.py`](test_ingestion.py) | Package loading, quality gating (Laplacian blur rejection), and sub-millisecond pose alignment. |
| [`test_keyframe_selector.py`](test_keyframe_selector.py) | Depth-adaptive keyframe selection, spatial baseline gating, and SIFT Lowe's ratio tests. |
| [`test_mesh_generation.py`](test_mesh_generation.py) | Segmented PBR mesh extraction, RANSAC planar snapping, and CAD/BIM exporters. |
| [`test_pipeline_steps.py`](test_pipeline_steps.py) | End-to-end multi-step pipeline runner (`run_pipeline.py`) verification and step completion idempotency. |
| [`test_rocm_precision_and_edge_cases.py`](test_rocm_precision_and_edge_cases.py) | Numerical stability tests on ROCm/AMD hardware, zero-division guards, and boundary cases. |

---

## Running the Tests

```bash
# Run all backend tests:
backend/.venv/bin/pytest backend/tests/ -v

# Run specific stage tests:
backend/.venv/bin/pytest backend/tests/test_depth_initialization.py -v
backend/.venv/bin/pytest backend/tests/test_step_filter_depth.py -v
```
