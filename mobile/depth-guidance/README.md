# Depth Guidance (shared spec, per-platform implementation)

Defines how the on-device monocular depth model (see `ml/depth-model/`) integrates with capture
guidance:

- Coverage-grid HUD feedback driven by the live low-quality point cloud
- Rough on-device floor-plan preview used only to guide the operator during capture

This point cloud is **guidance-only** — low quality by design, intentionally kept separate from
the backend reconstruction pipeline (`backend/`) and its point cloud/schema. It is never sent to
the backend as a reconstruction input.

Actual model integration code lives per-platform (`mobile/ios/`, `mobile/android/`) since Core ML
and TFLite/ONNX Runtime Mobile artifacts and APIs differ; this directory holds the shared
behavioral spec and thresholds both platforms must implement identically.
