# Project History: Backend Ingestion

## Milestone: Ingestion, Quality Gating & Dynamic Keyframe Selection

- **Package Loader:** Handles ZIP archives and directories; validates against `transforms.json`, `coverage_summary.json`, and `trajectory.csv` schemas.
- **Quality Gate:** Filters motion-blurred, over/underexposed, and redundant frames.
- **Pose Synchronization:** Quaternion SLERP and spline translation matching keyframes to 60 Hz VIO trajectory.
- **Dynamic Keyframe Selection:**
  - Implemented `DynamicKeyframeSelector` in `keyframe_selector.py`.
  - Enforces minimum spatial baseline (>= 0.35m) and angular parallax (>= 18 deg).
  - Evaluates 3D frustum co-visibility overlap (0.35 <= covis <= 0.78) against all historical anchor poses to prune loop-revisit redundancies and prevent double-wall smearing from cumulative VIO drift.
  - Verified on 8.8-min scan: condensed 941 quality-gate keyframes down to 103 optimal anchor views.
