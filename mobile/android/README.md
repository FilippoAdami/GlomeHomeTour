# Android Capture App

Native Kotlin app using ARCore for VIO pose tracking, decoupled from a dedicated CameraX capture
pipeline for locked-exposure/focus/white-balance video recording.

Runs the on-device depth-guidance model (see `mobile/depth-guidance/`) via TFLite (GPU/NNAPI
delegate) or ONNX Runtime Mobile.
