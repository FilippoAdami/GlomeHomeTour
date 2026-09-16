# AGENTS.md

Guidance and project rules for Antigravity when working in the GlomeHomeTour codebase.

## 1. Project Overview & Architecture Reference

**Glome Home Tour** is an automated spatial capture and reconstruction system converting standard smartphone video into an MLS-compliant digital listing bundle:
- **Interactive 3D/2D Gaussian Splatting (2DGS) Web Walkthrough** (≤25 MB compressed)
- **Automated 2D Floor Plan** with metric dimensions & room square meterage
- **Synthetic 4K 360° Equirectangular Panoramas** linked to floor plan nodes

### Documentation Map
- [README.md](README.md): System architecture, optical selection, hardware camera overrides, UX guidance, business tiering.
- [CLAUDE.md](CLAUDE.md): Locked tech stack, hardware targets, repo structure.
- [Phase_1.md](Phase_1.md): Mobile capture app specification (`mobile/android/`).
- [Phase_2.md](Phase_2.md): Shared interchange schemas (`shared/schemas/`).
- [Phase_3.md](Phase_3.md): Backend ingestion, 2DGS reconstruction, floor plan vectorization, panorama synthesis, API/worker packaging (`backend/`).
- [ImplementationPlan.md](ImplementationPlan.md): Multi-phase implementation roadmap.

---

## 2. Locked Tech Stack & Environment

1. **Mobile Capture (`mobile/android/`):**
   - Kotlin, ARCore VIO, Camera2 (`SharedCamera` manual overrides: 60 FPS, locked WB, joint shutter/ISO metering, OIS/EIS/distortion disabled).
2. **Shared Schemas (`shared/schemas/`):**
   - JSON Schema Draft 2020-12.
   - Self-validation script: `python shared/schemas/validate.py`.
3. **Backend Compute (`backend/`):**
   - Python 3.11+, PyTorch 2.3+ with ROCm 6.x (Target hardware: AMD Radeon RX 9060 XT / RDNA4, 16GB VRAM).
   - Rasterizer: Hand-authored HIP kernels (`rasterizer_hip/`) optimized for 32-wide wavefronts and LDS memory, with pure-PyTorch fallback oracle (`rasterizer_torch_fallback/`).
   - Depth Prior: Depth Anything v3 (metric depth initialization, separate from on-device guidance depth).
   - Floor Plan Vectorization: Cross-section slicing ($1.0\text{m}-1.5\text{m}$), multi-model RANSAC wall fitting, Manhattan-world regularization, Shapely.
   - Panorama Synthesis: Semantic room centroid discovery ($1.6\text{m}$ eye height, $\ge 0.8\text{m}$ wall clearance), 6-pass cubemap rendering, $4096 \times 2048$ equirectangular reprojection.
   - Orchestration: Redis/RQ queue worker daemon, FastAPI service.

---

## 3. Rules & Development Guidelines

1. **Do Not Touch Claude Configuration Files:**
   - Do not edit or overwrite `CLAUDE.md`, `Claude_Guidelines.md`, `.claudeignore`, or `.claude/`.
2. **Context & Token Optimization:**
   - Obey `.agentignore`: never index or read archived mobile variants (`mobile_0.0/`, `mobile_1.0/`, `mobile_2.0/`), build caches, checkpoints (`*.onnx`, `*.pt`), or `graphify-out/cache/`.
3. **Contract & Schema Integrity:**
   - Any backend output must validate against `shared/schemas/*.schema.json` via `python shared/schemas/validate.py`.
   - Never break field types or naming in `transforms.json`, `trajectory.csv`, `coverage_summary.json`, `floorplan.json`, `panorama_manifest.json`, or `asset_manifest.json`.
4. **Per-Folder Project History:**
   - Actively iterated subfolders (e.g. `backend/03_2DGS_training/`, `mobile/`, `shared/schemas/`) maintain a `project_history.md`. Append entries upon completing macro milestones.
5. **Subagents & Model Routing:**
   - Use `research` subagents for extensive file searches and broad codebase queries.
   - Escalate to higher-reasoning models for HIP kernel wavefront/LDS memory tuning, SfM bundle adjustment math, and 2DGS density control algorithms.
