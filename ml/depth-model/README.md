# On-Device Monocular Depth Model

Training, export, and quantization pipeline for the low-quality, guidance-only monocular depth
model that runs live on the phone during capture. Intentionally kept separate from
`backend/reconstruction/` (different hardware target: edge NPU export vs. ROCm desktop training;
different tooling: quantization-aware export vs. HIP/PyTorch-ROCm).

- `training/` — fine-tuning/distillation of a small depth model (e.g. a distilled Depth Anything
  V2 small / MiDaS-small variant) toward fast, low-quality metric depth suitable for real-time
  mobile inference.
- `export/` — Core ML conversion (`coremltools`) for iOS; TFLite/ONNX export for Android.
- `quantization/` — INT8/FP16 quantization-aware export for on-device latency/thermal budget.

Output of this pipeline is a guidance-only point cloud — not used as a reconstruction prior for
the backend pipeline.
