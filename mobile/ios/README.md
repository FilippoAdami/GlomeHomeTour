# iOS Capture App

Native Swift app using ARKit (ARSession/RealityKit) for VIO pose tracking, decoupled from a
dedicated AVFoundation capture pipeline for locked-exposure/focus/white-balance video recording.

Runs the on-device depth-guidance model (see `mobile/depth-guidance/`) via Core ML.
