# AR-Scan (mobile_3.0) — Feasibility Assessment & Refined Specification

Refines `projet.md`. Where this document and `projet.md` disagree, this one is what is built.
Every change is listed in §1 with the reason.

Target device for the first build: Xiaomi Redmi Note 10S (`rosemary`, Helio G95, Mali-G76 MC4,
Android 13 / API 33, ARCore 1.55). Android only for this iteration.

---

## 1. Feasibility assessment

### 1.1 Feasible as written

| Requirement | Verdict | Note |
| --- | --- | --- |
| 6-DoF VIO pose + timestamps + intrinsics + extrinsics logging | Feasible | ARCore gives `Frame.getTimestamp()` (ns, monotonic camera clock), `Camera.getPose()`, `Camera.getImageIntrinsics()`. |
| Photometric gate (`Y_mean`, ΔY, sliding-window variance) | Feasible | Y plane of `acquireCameraImage()` is already luma; sampled at stride 16 it costs well under 1 ms. |
| Kinematic guard (angular rate, walking speed) | Feasible | Both derive from consecutive poses; already proven in `mobile_sphere_capture`. |
| Sparse voxel grid @ 10 cm + ray carving + parallax validation | Feasible | ~300 depth samples/frame at 10 Hz, ≤40 voxel steps per ray = ~120 k updates/s in Kotlin on a worker thread. |
| Coverage overlay registered to the world | Feasible | Same GL pattern as the sibling apps. |
| `transforms.json` / `trajectory.csv` / `coverage_summary.json` export | Feasible | ARCore's camera pose convention (+X right, +Y up, −Z forward) is already the NeRF/OpenGL convention — no basis change needed. |

### 1.2 Infeasible or wrong as written — changed

**a. "1080p frames" is not free.** ARCore's *CPU* image stream is a separate, usually much
smaller stream than the GPU texture; on most mid-range devices the default is 640×480. A 1080p CPU
stream must be selected explicitly via `Session.getSupportedCameraConfigs()` and it costs frame
rate on a Helio G95.
→ **Refined:** pick the largest supported CPU config ≤1920×1080 at 30 fps at session start, log
what was actually obtained, and record the real resolution in `transforms.json`. The pipeline
consumes intrinsics, so 1280×960 or 640×480 is a quality reduction, not a correctness failure.

**b. "≥58 fps sustained (60 fps target)" is not reachable and is not wanted.** ARCore's camera
stream on this class of device runs at 30 fps; the AR frame rate cannot exceed it.
→ **Refined:** 30 fps target, ≥27 fps sustained. The 60 Hz figure is dropped entirely.

**c. Writing every frame to the dataset is not survivable.** 30 fps × 1080p JPEG ≈ 12 MB/s and
~2 GB for a 3-minute scan, and the encoder alone would eat the frame budget. It is also useless:
2DGS/SfM wants well-separated views, not 30 duplicates per second of the same viewpoint.
→ **Refined:** *keyframe* export — a frame is written only when it passes every gate **and** has
moved ≥8 cm or rotated ≥6° since the previous keyframe, capped at 5 Hz. `trajectory.csv` still
logs **every** tracked frame, so the unbroken VIO trajectory required by §8 is preserved.

**d. "Depth Anything v3 monocular metric depth" on-device is out of scope.** It is a backend
stage in `projet.md`'s own pipeline (§1, step 2), not a client feature.
→ **Refined:** client uses ARCore's Depth API (`DepthMode.AUTOMATIC`, motion-stereo, ~160×120)
for the coverage volume where the device has it. Nothing from this path is exported — coverage
guidance only, consistent with the root `CLAUDE.md` rule that on-device depth never becomes a
backend prior.

→ **Revised after on-device measurement (2026-08-27, `rosemary`):** the target device supports
neither `DepthMode.AUTOMATIC` nor `RAW_DEPTH_ONLY`, and its feature point cloud returns **0
points** for seconds at a time (0 points across 285 consecutive tracked frames while stationary).
Both the primary source and the planned fallback are therefore unavailable there, which is why
coverage never left 0% and the overlay had nothing to draw. The fallback is now the project's own
on-device monocular depth model — ZipDepth base, 256×256, ONNX Runtime CPU, ~390 ms/frame — reused
from `mobile_depth_map/`, with its affine-invariant disparity anchored to metres per frame by
least-squares against whatever ARCore feature points exist (`disparity ≈ α/distance + β`), the fit
smoothed and held across the stretches where ARCore publishes none. This is the guidance-only
monocular depth the root `CLAUDE.md` tech stack already calls for; it is still never exported.
Order of preference: ARCore depth → monocular model → raw feature points.

**e. "Coverage mesh shader" implies a meshing stage that earns nothing here.** Marching-cubes
over a live voxel grid is a large amount of code whose only job is to look smoother than the
voxels it is built from.
→ **Refined:** occupied voxels are drawn directly as depth-attenuated GL points, amber→green by
parallax progress. Upgrade path is noted in code if the point cloud reads as too sparse in the
field.

**f. Kotlin Multiplatform shared layer.** One platform is being built. A KMP module for two
files of vector math is scaffolding for a port that does not exist yet.
→ **Refined:** dropped. Pure-Kotlin, Android-free classes (`VoxelGrid`, `PhotometricGate`,
`Nv21`, dataset serialisation) are kept free of Android imports, so extracting them later is a
file move.

**g. iOS.** Not in this iteration (explicitly Android-only). The §3 platform matrix's iOS column
is deferred, not rejected.

**h. `Y_mean < 40` as the only illumination reject.** §7.1's own pass criterion says "no black
**or blown** frames".
→ **Refined:** symmetric guard, reject below 40 and above 250.

**i. "Free / Occupied / Occluded" tri-state.** "Occluded" (behind the hit point) is
indistinguishable from "never seen" for every purpose this app has, and storing it triples the
voxel count.
→ **Refined:** two stored states, FREE and OCCUPIED; absent = unknown. Frontier detection
(§2.3) recovers exactly the guidance signal the third state was for.

### 1.3 Open risks (measured on-device, reported in the HUD)

1. ~~**Depth API support on `rosemary`.**~~ **Resolved, negative:** unsupported, and the point
   cloud is empty whenever the operator stands still. Monocular model path added (§1.2d). The
   remaining risk is its accuracy, not its availability: the metric fit needs feature points, so
   a scan that starts stationary shows `CALIBRATING DEPTH` until the operator walks. HUD shows
   which source is live, the model's latency, and the point count behind the current fit.
2. ~~**CPU image cost.**~~ **Resolved, confirmed:** acquiring and reading a 1080p CPU image
   every frame was the dominant render-thread cost and capture felt laggy in the field. Now
   acquired **every other frame** (`IMAGE_EVERY_N_FRAMES`), the NV21 conversion for keyframes is
   split (bulk row copies on the render thread, interleave + JPEG on the writer thread, queue
   depth 1 so the shared buffers are never rewritten under the encoder), the crop sampler feeding
   the depth model steps a precomputed affine instead of remapping 65k pixels, ONNX runs on 2
   intra-op threads rather than 4, and the HUD is rebuilt at 6 Hz rather than 30.
3. **Grid memory.** Free-space carving dominates the voxel count. The grid grows by doubling to
   a hard cap and reports occupancy in `coverage_summary.json`; hitting the cap in a real home
   means the free-space voxel size needs to be decoupled from the surface voxel size.

---

## 2. Refined functional specification

### 2.1 Capture session

States: `IDLE → SCANNING → DONE`. One continuous walk per session (unlike
`mobile_sphere_capture`'s per-location model). Backgrounding parks the scan (`SCANNING` +
`interrupted`); it resumes on tap, keeping the same grid and dataset.

### 2.2 Logging

Every tracked frame → `trajectory.csv`. Keyframes (§1.2c) → `images/` + `transforms.json`.
Intrinsics are read once per session from the first tracked frame and asserted unchanged.

### 2.3 Volumetric coverage

- Spatial hash grid, 10 cm voxels, open-addressed with parallel primitive arrays (no boxing).
- Grid pitch: 10 cm on ARCore depth, **15 cm on the monocular model** — its several-percent scale
  error puts the same wall in a different 10 cm voxel on each pass, so no voxel is ever seen twice
  and coverage cannot rise. Calibration knob, tighten as the model improves.
- Per depth sample: carve FREE from the camera toward the hit point (stopping 1.5 voxels short),
  mark the hit voxel OCCUPIED. FREE never overwrites OCCUPIED.
- Per OCCUPIED voxel, store the first observation bearing and the widest angle seen since.
  ≥25° ⇒ **parallax-verified**.
- Coverage % = verified OCCUPIED / total OCCUPIED. It turns the finish button green (scan judged
  complete) at ≥85%, and never blocks finishing on its own — see §2.4.
- **Frontier** = FREE voxel with ≥1 unknown 6-neighbour, within 6 m of the camera and inside the
  0.3–2.0 m height band above the session's floor estimate. Guidance target = centroid of the
  frontier cluster nearest the operator (nearest frontier voxel + everything within 1 m of it),
  and **no closer than 1.5 m** — the free space around the operator always has unknown neighbours
  just outside the field of view, so without that floor the target pins itself to the lens.

### 2.4 Guidance

- 3D: amber **wireframe** diamond (octahedron edges) at the guidance target; voxel points
  amber→green. Solid shading was tried first and read in the field as an unexplained orange
  square, because the marker is close, has no depth test, and a filled octahedron seen head-on is
  a square.
- 2D: on-screen arrow rotating toward the target's screen bearing, hidden when the target is
  centred in view. The legend rides on the arrow ("Not scanned yet" / "Hidden — walk around"),
  amber for the frontier and cyan for occlusion, matching the 3D marker colours — a legend printed
  elsewhere on screen doesn't get connected to the shape it names.
- Banners: one slot, highest-priority message only — tracking loss (with the ARCore failure reason
  translated into an instruction), depth calibration, too dark, too bright, lighting transition,
  over-speed, over-rotation. A warning buzzes once when it appears, because the operator is looking
  at the room rather than at the screen.
- Coverage headline + progress bar bound to coverage %; the primary button turns green at ≥85%
  coverage. It is *disabled* only while the tap would be a no-op anyway — pre-flight step 2 before
  ARCore has a pose, and finish before entry-door loop closure or with a validation issue
  outstanding — with the reason (and the distance back to the start point) printed under it. The
  85% coverage target itself never blocks finishing: a scan the operator cannot end is a trap, and
  85% may be unreachable in a room with a mirrored wardrobe.
- Operator-facing HUD carries no ARCore enums, frame counters or camera register values; §1.3's
  diagnostics panel is behind the `i` button in the header.

### 2.5 Export package & Metadata Schema Conformance

All exported JSON manifests are guaranteed to conform to the shared interchange schemas (`shared/schemas/*.schema.json`, schema_version 1.0.0):

```
Documents/GlomeHomeTour/scan_<yyyyMMdd_HHmmss>/
├── images/
│   ├── frame_00000.jpg …           decimation-filtered keyframes, camera-native orientation
├── transforms.json                 NeRF/instant-ngp/2DGS format (schema_version: 1.0.0,
│                                   camera_model: OPENCV, fl_x, fl_y, cx, cy, w, h, camera_angle_x,
│                                   k1..p2 distortion zeroes, per-frame 4×4 row-major camera-to-world
│                                   extrinsics + timestamp_ns)
├── trajectory.csv                  unbroken 60 Hz VIO stream (timestamp_ns,tx,ty,tz,qx,qy,qz,qw,tracking,exported)
├── coverage_summary.json           room coverage %, voxel & landmark metrics, frame drop counts by reason,
│                                   tracking loss event counter, depth_source (schema_version: 1.0.0)
└── focus_metadata.json             per-keyframe focus distances (diopters & metres), AF states/modes, focal length mm
```

Written through `MediaStore.Files` under `Documents/`, so the dataset is visible to the Files
app and to MTP without any storage permission (API 29+; `minSdk` is 30).

## 3. Acceptance criteria (revised §7)

| # | Criterion |
| --- | --- |
| A1 | ≤1 tracking-loss event per 3-minute scan at ≥150 lux |
| A2 | Peak RAM ≤350 MB over 5 minutes of continuous capture |
| A3 | Sustained ≥27 fps (30 fps target) |
| A4 | Zero exported frames with `Y_mean` <40 or >250, or ΔY >30 vs. the previous exported frame |
| A5 | `trajectory.csv` is continuous across every rejected frame and every tracking loss |
| A6 | TC-01/02/03 from `projet.md` §7.2 pass unchanged |

---

## 4. Live Field Evolution: 3DGS Multi-Object Landmarking & Gamified UX (2026-09-02)

### 4.1 Transition from Monocular Plane/Voxel Overlay to Sparse Multi-View Landmarks
- **Issue:** On monocular phones, ARCore plane finding and voxel grids produced phantom geometry floating in mid-air and wall floaters.
- **Solution:** Switched to persistent sparse feature point tracking with live 2-ray geometric triangulation (`FeatureParallaxTracker`):
  - Solves exact line-of-sight intersection across $\ge 12^\circ$ baseline to snap points to true surface depths.
  - Emerald green verified points are stored as permanent 3D landmarks in world space (never deleted when out of frame).
  - Unverified single-view candidate points fade out when out of view.

### 4.2 Gamified 3DGS Multi-Object Capture
- **Target Room Landmarks:** 350–500 points (200 point bare minimum to finish).
- **Orbiting UX:** Dynamically instructs the operator to orbit objects of interest, lighting up green landmarks as parallax confirms 3D surface geometry.
- **2D Top-Down Mini-Map:** Real-time bird's-eye view showing operator position, heading, and room footprint.

### 4.3 5-Second Lighting & Auto White-Balance Calibration
- Automatically determines balanced joint ISO and shutter speeds (ISO 100-800, 1/120s shutter baseline) during an initial 5-second multi-angle sweep.
- Seamlessly calculates and locks grey-world RGB gains, eliminating manual slider friction.
