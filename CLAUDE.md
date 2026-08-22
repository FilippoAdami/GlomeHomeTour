# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

**Glome Home Tour** is an automated, AI-assisted spatial capture and reconstruction system for the residential real estate market. It converts standard monocular smartphone video (captured by non-technical real estate agents) into an MLS-compliant digital listing bundle:

- An interactive **3D Gaussian Splatting (3DGS) Web Walkthrough**
- An **Automated 2D Floor Plan** with room dimensions
- A set of **Synthetic 4K 360° Equirectangular Panoramas** mapped to the floor plan

The system couples a **Native VIO Mobile Guidance Capture App** with a **Headless Cloud/Desktop 3DGS Reconstruction Engine**, aiming to eliminate common failure modes of mobile spatial reconstruction: operator error, motion blur, incomplete scene coverage, and tracking failure.

### Core Components

1. **Mobile Capture & Guidance Engine** (iOS ARKit / Android ARCore)
   - Kinematic & motion-blur guard (angular velocity, walking speed limits)
   - Persistent spatial memory grid ("floor painting" coverage feedback)
   - VIO tracking protection & blank-wall safeguards
   - Keyframe relocalization engine
   - Enforced loop-closure anchor at the entry door
   - On-device pre-flight validation gate before upload

2. **Spatial Reconstruction & Derivation Engine** (backend, GPU compute)
   - Monocular 3D Gaussian Splatting training from RGB video + VIO pose priors
   - Gravity alignment & ground-plane detection
   - Automated 2D floor plan vectorization (cross-section slicing + RANSAC wall extraction)
   - Semantic viewpoint/node discovery for panorama placement
   - 360° equirectangular panorama synthesis

3. **Distribution & Delivery Interfaces**
   - Interactive 3DGS web viewer (Three.js/WebGL, ≤25 MB compressed scenes)
   - Interactive 2D minimap widget linking floor plan nodes to panoramas
   - Listing portal asset pack (ZIP export: 4K panoramas, dimensioned floor plans, embeddable iframes)

See [README.md](README.md) for full product spec, architecture table, and the Phase 1→2 acceptance checklist.

### Architecture Summary

| Layer | Notes |
| --- | --- |
| Mobile Capture (Client) | iOS/Android, native ARKit/ARCore VIO, 1080p/4K video, decoupled tracking vs. video recording pipeline |
| Backend Compute (Host) | GPU (≥16 GB VRAM, CUDA/ROCm), 3DGS optimization, ≤12–15 min turnaround for 80–120 sqm properties |
| Delivery & Storage | Serverless hosting, CDN, encrypted object storage |

## Tech Stack (locked in)

**Mobile capture app** — native, not cross-platform (ARKit/ARCore VIO internals and precise
camera/AR decoupling are more reliable native than through cross-platform wrappers):
- iOS: Swift, ARKit (ARSession/RealityKit), AVFoundation for locked-exposure video capture
- Android: Kotlin, ARCore, CameraX

**On-device monocular depth (guidance-only)** — deliberately separate from backend reconstruction:
- Small distilled/fine-tuned depth model (e.g. Depth Anything V2 small / MiDaS-small variant),
  intentionally low quality — used only to drive live coverage HUD and a rough on-device
  floor-plan preview, never sent to the backend as a reconstruction prior
- iOS: Core ML (`coremltools` export), Neural Engine
- Android: TFLite (GPU/NNAPI delegate) or ONNX Runtime Mobile

**Backend reconstruction engine** — targets local ROCm hardware (RX 9060 XT / RDNA4, 16GB VRAM):
- Python for orchestration (ingestion, hybrid VIO+SfM pose refinement, floor-plan extraction,
  panorama synthesis)
- Custom 2DGS (surfel-based) model — **not** an off-the-shelf framework (gsplat/reference 2DGS
  ship CUDA-only rasterization kernels, incompatible with ROCm and with a modified splat model)
- Rasterizer: **hand-authored HIP kernels**, written from scratch against RDNA4 characteristics
  (32-wide wavefronts, LDS sizing, occupancy) — not hipify-ported CUDA, not Triton. Chosen over
  Triton because irregular tile-based rasterization with per-tile sorting and atomic blending
  benefits from manual kernel control more than from Triton's compiler; the extra implementation
  difficulty is an accepted trade-off for the performance ceiling.
- Pure-PyTorch fallback rasterizer kept as a correctness oracle / bring-up path
- ROCm/PyTorch-ROCm versions must be pinned; RDNA4 consumer-card ROCm support is newer/less
  mature than CDNA — re-validate the environment after driver or ROCm updates
- Job orchestration: Redis/RQ or Celery queue + GPU worker; Kubernetes/Ray deferred until
  needed beyond a single-machine Phase 1 setup

**Web delivery**
- TypeScript + Three.js (or React Three Fiber) for the 2DGS web viewer and minimap widget
- Static hosting via CDN + object storage (S3/R2/GCS)

## Repo Structure

```
glome-home-tour/
├── mobile/
│   ├── ios/                 # Swift/ARKit app
│   ├── android/             # Kotlin/ARCore app
│   └── depth-guidance/      # shared spec for on-device depth guidance behavior
├── ml/
│   └── depth-model/         # training/export/quantization for the on-device depth model
│                             # (separate env from backend/ — export tooling, not ROCm)
├── backend/
│   ├── ingestion/           # frame extraction, pose/timestamp matching
│   ├── reconstruction/
│   │   ├── rasterizer_hip/            # hand-authored HIP 2DGS rasterizer (ROCm/RDNA4)
│   │   ├── training/                  # 2DGS optimization loop, density control
│   │   └── rasterizer_torch_fallback/ # pure-PyTorch rasterizer, correctness oracle
│   ├── floorplan/           # cross-section slicing, RANSAC, Manhattan regularization
│   ├── panorama/            # viewpoint discovery, equirectangular synthesis
│   ├── api/                 # job submission/status, asset packaging
│   └── worker/              # queue consumer, GPU job runner
├── web/
│   ├── viewer/               # 2DGS Three.js web viewer
│   └── minimap/               # 2D floor plan widget
├── shared/
│   └── schemas/              # pose JSON, floor-plan format, panorama manifest
│                             # (does NOT include the mobile guidance point-cloud format —
│                             # that stays intentionally separate, see mobile/depth-guidance/)
└── docs/
```

Each subsystem directory has its own `README.md` describing its scope; see those for
implementation-level detail as code lands.

## Coding Conventions

_This section will be filled in as conventions are established for this codebase (language/framework choices, formatting, testing, commit style, etc.)._
