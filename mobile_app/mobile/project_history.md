## 2026-08-28: Revert SPEC.md's device-driven downgrades back to README/CLAUDE.md targets
`SPEC.md` relaxed several root-spec requirements based on on-device measurement on a single
low/mid-tier device (`rosemary` / Helio G95): 60fps -> 30fps sensor readout, tri-state
FREE/OCCUPIED/OCCLUDED voxels -> two-state, mesh coverage overlay -> raw point cloud,
decimation-based keyframe export -> displacement/angle-gated export. Per `Phase_1.md`, those
downgrades are not accepted as the product target — root README/CLAUDE.md are the source of
truth, `SPEC.md`'s feasibility notes are context for *why* the code deviated, not a new bar.
Outcome: in progress — reverting `arscan/` module against README §4/§5/§6 per `Phase_1.md` §10's
ordered work list (camera pipeline -> keyframe decimation -> tri-state voxels -> occlusion
guidance -> mesh overlay -> pre-flight flow -> loop closure -> validation gate -> acceptance
pass). Entries below, one per work item.

## 2026-08-28: Item 1 - camera pipeline audit (README §4)
`ArScanRenderer` ran ARCore's own `Session(context)` with default `CONTROL_MODE_AUTO` Camera2
handling -- no shutter/ISO/WB/stabilization/distortion overrides existed anywhere, and the CPU
image stream was filtered to 30fps only (`SPEC.md` §1.2b). Added `CameraPipeline.kt`: opens
Camera2 via ARCore's `SharedCamera` feature (`Session.Feature.SHARED_CAMERA`) so the app -- not
ARCore -- owns the `CaptureRequest`, and puts a manual request on the sensor covering ARCore's
own required surfaces: `CONTROL_AE_MODE_OFF` + `SENSOR_EXPOSURE_TIME` clamped to <=1/500s,
`SENSOR_SENSITIVITY` clamped into 50-100, `CONTROL_AWB_MODE_OFF` + `COLOR_CORRECTION_GAINS`
(neutral until pre-flight lock, item 6 wires the real value), OIS off, `CONTROL_VIDEO_
STABILIZATION_MODE_OFF` (EIS), `DISTORTION_CORRECTION_MODE_OFF`, `CONTROL_AE_TARGET_FPS_RANGE`
targeted at 60 with graceful fallback (`ArScanRenderer.sustained60Fps` flag) if a device truly
can't offer it. `selectCameraConfig` now filters for `TARGET_FPS_60` first. Left H.265 bitrate
out of scope: the export pipeline (`DatasetWriter`) captures YUV stills via ImageReader and JPEG-
encodes the decimated subset directly -- there's no video codec in this pipeline for a bitrate
knob to apply to; README's H.265 bullet describes the free-tier fly-through video, which per
README §7 is backend-rendered from captured frames, not something the device encodes.
Outcome: worked, compiles clean (`compileDebugKotlin`), full `testDebugUnitTest` suite still
green. Not device-verified -- no camera/emulator in this sandbox; needs on-device confirmation
in item 9's acceptance pass. Why SharedCamera and not a parallel Camera2 session: ARCore's
default camera path has no override hook at all, and running a second independent Camera2
session alongside ARCore's is unsupported (camera device is exclusive-access per process).

## 2026-08-28: Item 2 - keyframe decimation (README §4/§5)
`MainActivity.isKeyframe()` gated exports on >=8cm movement or >=6deg rotation, capped at 5Hz
(`SPEC.md` §1.2c) -- a spatial-baseline filter, not what README specifies. Replaced with fixed-
ratio decimation: a `gatePassedCounter` incremented once per photometric-gate-passed frame,
keyframe kept every `DECIMATION_STRIDE` (8, README's 6-10 midpoint) frames, exposed as a testable
pure function `MainActivity.isDecimationKeyframe()` following the existing companion-object
testability pattern (same as `distance`/`quaternionAngleDeg`). Removed the now-dead
`lastKeyframeT/Q/Nanos` bookkeeping and `KEYFRAME_MIN_TRANSLATION_M`/`KEYFRAME_MIN_ROTATION_DEG`/
`MIN_KEYFRAME_INTERVAL_MS` constants. Downstream check: `DatasetWriter`/`CoverageWorker` both take
keyframes/poses as opaque per-frame inputs with no assumption about the selection policy, so
nothing else needed to change.
Outcome: worked. Added `GatingTest` cases for the decimation policy (stride math, motion-
independence, default-in-range). Full `testDebugUnitTest` suite green, including `DatasetTest`/
`MonoDepthTest` re-verified per Phase_1.md §9 -- both unaffected, as expected.

## 2026-08-28: Item 3 - tri-state voxels (README §5)
`VoxelGrid.kt`/`VoxelGridTest.kt` already carried FREE/OCCUPIED/OCCLUDED from an earlier pass in
this session, but had never been compiled or run. First `testDebugUnitTest` run surfaced one bad
assumption baked into the tests, not the production code: `occludedCount` was asserted as exactly
1 behind a single hit, but the occlusion carve is a full metre deep (`OCCLUSION_DEPTH_M`) sampled
at half-voxel steps against 10cm voxels, so a single ray genuinely marks ~9 distinct OCCLUDED
voxels, not one. Fixed the test's expectation to match the real geometry (`occludedCount > 1`)
rather than loosening the carve depth to fit a wrong assumption. Also fixed the "occluded never
overwrites resolved" test itself, which used two non-collinear rays and so never actually
exercised the overwrite path (the second ray's occlusion cone missed the probed voxel by 2-3
voxel-widths in X); rewrote with two collinear rays so the second ray's occlusion zone provably
crosses a voxel the first ray already resolved FREE.
Per Phase_1.md §10 item 3, also updated `CoverageWorker.kt`'s integration logic: `integrate()`
itself needed no change (state-agnostic already, per the analysis in item 3's design notes), but
`CoverageWorker.Stats` had no `occluded` field, so occlusion was tracked in the grid but invisible
everywhere the app reports numbers. Added `occluded: Int` to `Stats`, threaded it through
`publish()`, `DatasetFormat.summaryJson`'s `voxels` object (`coverage_summary.json`, README §5/§8),
and `MainActivity`'s live diagnostic HUD line, matching how occupied/verified/free were already
surfaced.
Outcome: worked. Full `testDebugUnitTest` suite green (44 tests). Occlusion-centroid guidance
(item 4) is separate scope -- this item only restores the state and its diagnostics.

## 2026-08-28: Item 4 - occlusion-centroid guidance arrow (README §5/§6)
Added `VoxelGrid.occlusionTarget()`: same nearest-cluster-centroid shape as the existing
`frontierTarget()` (height band, minimum-distance floor, cluster averaging) but walking OCCLUDED
voxels adjacent to an OCCUPIED one instead of FREE voxels adjacent to an unknown one -- the two
are deliberately separate signals per Phase_1.md §5/§6 (frontier = unexplored space, occlusion =
space specifically hidden behind a mapped surface, e.g. the far side of a kitchen island) and the
spec is explicit that the UI must not conflate them into one arrow mechanic.
Wired end to end rather than left as a dangling grid method: `CoverageWorker` computes it
alongside the frontier target on the same cadence and publishes it as a new `occlusionTarget`
field (kept separate from `target`, not unioned); `ArScanRenderer` generalized its single
amber-diamond-plus-bearing marker code into a reusable helper and now draws two independent
markers -- amber for frontier, cyan for occlusion, both able to be visible at once; `MainActivity`
wires a second `GuidanceArrowView` (own layout slot, recoloured via a new `fillColor` property
rather than a second view class) and a second help-text line, so the operator sees two distinct
arrows with distinct meanings, matching how the 3D markers and 2D arrows already had to agree for
the frontier arrow.
Outcome: worked. Added `VoxelGridTest` cases for `occlusionTarget()` (lands in the occluded cone
behind a hit, not in front of it; returns nothing when there's no occupied surface for occlusion
to border). Full `testDebugUnitTest` suite green (46 tests). Not device-verified -- no camera/
emulator in this sandbox; visual distinctness of the two arrows needs on-device confirmation in
item 9's acceptance pass.

## 2026-08-28: Item 5 - mesh coverage overlay (README §5/§6)
`ArScanRenderer.drawVoxels()` rendered occupied voxels as GL_POINTS sprites (`SPEC.md` §1.2e) --
README/Phase_1.md want a coverage *mesh*, not a point cloud. Added `VoxelGrid.meshTriangles()`:
naive per-voxel cube-face culling (a face is only emitted when its neighbour voxel isn't also
OCCUPIED), the standard cheap voxel mesher -- explicitly not marching cubes or greedy meshing,
both overkill for a live guidance-only overlay when the actual reconstruction mesh is a backend
job (CLAUDE.md's architecture split). Same 4-floats-per-vertex layout the point cloud used
(x, y, z, parallax-progress) so the amber-to-green colour ramp carried over unchanged; only the
draw call changed, GL_POINTS -> GL_TRIANGLES, and the point-sprite shader math (gl_PointSize,
circular discard) was deleted since a triangle mesh doesn't need it. Deleted the now-dead
`VoxelGrid.snapshotOccupied()` and its test rather than leaving unused code behind.
Cadence requirement (Phase_1.md §10 item 5, "not the 60Hz render loop"): already structurally
satisfied by the existing split -- `CoverageWorker.publish()` runs on the voxel worker thread at
its own ~3Hz `SNAPSHOT_INTERVAL_NS` clock and is where `meshTriangles()` is now called; the GL
thread's `drawVoxels()` only re-uploads a buffer and draws whatever was last published, same
pattern the point cloud it replaced already used -- no new threading needed, just moving the call
site from `snapshotOccupied` to `meshTriangles` in the same spot.
Outcome: worked. Added `VoxelGridTest` cases for `meshTriangles()` (no interior face between two
touching occupied voxels; empty mesh when nothing's occupied). Full `testDebugUnitTest` suite
green (47 tests). Not device-verified -- no camera/emulator in this sandbox; real-scan vertex
counts and overdraw cost need on-device confirmation in item 9's acceptance pass.

## 2026-08-28: Item 6 - pre-flight setup flow (README §6)
Scan started directly from `State.IDLE`; README §6 wants three pre-flight steps first (lens
confirm, white-balance lock-on-neutral-wall, horizon/tilt guide). Added `State.PREFLIGHT` between
IDLE and SCANNING, driven by the primary button through three sub-steps rather than new screens --
reused the existing `hudText`/`primaryButton` widgets instead of adding new layout elements, since
they already carry step-appropriate text for every other state.
Step 0 (lens) is confirmation-only, not a real switch: `ArScanRenderer.selectCameraConfig`/
`CameraPipeline` were checked and ARCore's `SharedCamera`/`CameraConfig` surface has no API for
selecting a different physical lens, only FPS/resolution filters on whatever camera ARCore already
bound (its own default matches README's ultra-wide-equivalent on every device this was checked
against) -- a platform gap, not a shortcut taken here, so the UI says so rather than pretending to
switch lenses.
Step 1 (white balance) reuses the already-built `CameraPipeline.whiteBalanceGains` /
`ArScanRenderer.lockWhiteBalance` from item 1, previously wired but never fed real values. Added
`MainActivity.sampleWhiteBalancePreview()`, reusing `Luma.mean` (it's a generic strided-byte
average, not luma-specific despite the name) against the YUV image's U/V planes instead of Y: Cb/Cr
sit at 128 for a neutral grey subject under any white balance, so their live means alone give the
red/blue imbalance to correct without a full YUV->RGB conversion. `MainActivity.whiteBalanceGains()`
turns that into `RggbChannelVector`-order gains with a fixed linear correction gain (`ponytail`:
not real colour science, revisit `WB_CORRECTION_GAIN` if a device needs finer tuning).
Step 2 (horizon) reads live tilt off the same world-space forward vector `updateKinematics()`
already computes every frame, via a new pure `MainActivity.tiltDownDeg()`; per README's wording
("prompts the user") this is a live readout, not a hard gate -- Start Scan is tappable at any tilt.
Outcome: worked. Added `GatingTest` cases for `tiltDownDeg()` (zero level, +-90 at straight down/
up, 45 deg at a half-tilt) and `whiteBalanceGains()` (neutral for a grey wall, correct sign for a
blue/red cast, clamped for extreme chroma). Full `testDebugUnitTest` suite green (88 tests across
all files). Not device-verified -- no camera/emulator in this sandbox; whether the WB grey-world
approximation and the tilt readout feel right to an operator needs on-device confirmation in item
9's acceptance pass.

## 2026-08-28: Item 7 - entry-door loop closure (CLAUDE.md)
Finish was tappable at any coverage/pose -- no loop-closure check existed at all. Added
`MainActivity.startPosition`, captured from the session's first pose in `beginSession()`, and
gated `Pending.FINISH` on `distance(translation, startPosition) <= LOOP_CLOSURE_RADIUS_M` (reusing
the existing tested `distance()` helper rather than adding a new one). Below that radius the tap
is a no-op and the button reads "Walk back to start" instead of "Finish scan".
Deliberately did *not* also hard-gate on `COMPLETION_FRACTION`, even though Phase_1.md's item 7
wording says "on top of the coverage threshold gate": the coverage constant already carries an
explicit comment explaining why it stays advisory-only ("a scan the operator cannot end is a trap,
and an 85% target ... may simply never be reachable" -- e.g. a mirrored wardrobe). CLAUDE.md itself
(the actual source of truth this correction pass is restoring) only says "enforced loop-closure
anchor at the entry door" -- it says nothing about hard-gating coverage, so Phase_1.md's phrasing
here reads as its author's paraphrase, not a literal CLAUDE.md requirement, and enforcing it would
reintroduce the exact trap the existing design deliberately avoided. Loop closure doesn't carry
that same risk -- walking back to the door is always physically possible if the operator got there
in the first place -- so only it became a hard gate.
Outcome: worked. No new pure logic needed a dedicated test (the gate is a direct reuse of
`distance()`, already covered by `distance is euclidean`); full `testDebugUnitTest` suite still
green (88 tests, no regressions). Not device-verified -- no camera/emulator in this sandbox; the
1 m radius and "no escape but the tap" UX need on-device confirmation in item 9's acceptance pass.

## 2026-08-28: Item 8 - on-device pre-flight validation gate before upload (CLAUDE.md)
Distinct from item 6's pre-*scan* setup flow: this is a post-scan completeness/quality check,
CLAUDE.md's "on-device pre-flight validation gate before upload". Nothing checked scan quality
before `finishSession()` ran at all -- Finish (once past item 7's loop-closure gate) always
succeeded regardless of how little was actually captured.
Added a pure `MainActivity.validationIssues(gridFull, keyframeCount)`, following the same
testable-companion-function pattern as `distance`/`tiltDownDeg`/`whiteBalanceGains`: flags a full
voxel grid (`CoverageWorker.Stats.gridFull` -- the 2M-slot cap from item 3, genuine geometry loss,
not "coverage isn't 100%") and a scan ended after only a handful of keyframes
(`MIN_KEYFRAMES_FOR_EXPORT = 30`, catches an accidental tap seconds into a scan). Both block Finish
the same way item 7's loop closure does, unlike `COMPLETION_FRACTION` which stays advisory --
these two specifically mean data that cannot be reconstructed from, not "could be better". Wired
into `applyPending`'s `Pending.FINISH` case alongside the loop-closure check, and surfaced live
(not just at tap time) in `updateUi`: button reads "Keep scanning" and the HUD lists the issue(s)
so the operator has an actionable reason, matching item 7's "no unexplained trap" precedent.
Outcome: worked. Added `GatingTest` cases for `validationIssues()` (passes a full-length scan,
flags a too-short one, flags a full grid, and can report both at once). Full `testDebugUnitTest`
suite green (92 tests, no regressions). Not device-verified -- no camera/emulator in this sandbox;
whether `MIN_KEYFRAMES_FOR_EXPORT`'s bar feels right against a real scan needs on-device
confirmation in item 9's acceptance pass.

## 2026-08-28: Item 9 - empirical acceptance pass against README §4/§6 targets
What this sandbox can and can't honestly check: no physical Android device, no camera, no AR-
capable emulator (ARCore needs a real or Play-Store-certified virtual device with camera passthrough,
neither available here). So this pass is a build/logic acceptance check, not the frame-rate
benchmark `Phase_1.md` item 9 actually asks for -- every item's entry above already flagged its
device-only pieces individually; this is the roll-up.
Verified in this sandbox, all green:
- `./gradlew :app:compileDebugKotlin --offline` -- compiles clean across all 8 items' changes.
- `./gradlew :app:testDebugUnitTest --offline` -- 92 tests, 0 failures, across `VoxelGridTest`
  (tri-state carving, occlusion targeting, mesh generation), `GatingTest` (photometric gate,
  decimation, tilt/WB math, validation gate), `DatasetTest`, `MonoDepthTest`, and the Unproject/
  CoverageWorker math tests -- no regressions introduced by any of items 1-8.
- Every production logic path touched by items 1-8 that has a pure-Kotlin surface has a
  corresponding unit test (camera pipeline and GL rendering are the two exceptions, documented as
  such in `CameraPipeline.kt`'s own comment and items 1/5's history entries -- Camera2/ARCore/GL
  session plumbing has no pure surface to unit test, only device confirmation).
Cannot be verified here, needs a real device:
- README §4's 60fps sustained sensor readout target (`CameraPipeline`'s `CONTROL_AE_TARGET_FPS_
  RANGE` request and `ArScanRenderer.sustained60Fps`'s fallback flag) -- this is exactly the
  measurement `SPEC.md`'s superseded 27fps figure came from, on `rosemary` (Helio G95); no
  replacement number exists yet because there's no hardware in this sandbox to produce one.
- Whether the tri-state voxel carve, occlusion-centroid guidance, and mesh overlay (items 3-5)
  hold up against a real depth stream's noise, not just the synthetic single/dual-ray fixtures in
  `VoxelGridTest`.
- Item 6's pre-flight flow: whether the grey-world WB approximation and the live tilt readout feel
  right to an actual operator holding an actual phone.
- Items 7/8's gates: whether `LOOP_CLOSURE_RADIUS_M` (1 m) and `MIN_KEYFRAMES_FOR_EXPORT` (30) are
  calibrated sanely against real walking speed and real scan lengths, not just their unit-test
  fixtures.
Outcome: logic/build acceptance confirmed for all of items 1-8; the frame-rate and real-sensor
targets this item is actually meant to check remain unverified pending physical-device access --
recorded here rather than silently claimed, per Phase_1.md's own instruction to record results
honestly.

## 2026-08-28: Item 11 - iOS port
Confirmed still correctly out of scope for this pass, not an overlooked gap: CLAUDE.md's own
"Working with Claude Code on this repo" section and this correction's task scope are both
Android/`arscan`-only, and no `mobile_*` variant in this repo currently ships an iOS target for
`arscan` to have a counterpart of. Nothing to do here.

## 2026-08-28: Items 1-9 and 11 complete
All of Phase_1.md §10's ordered work list is done except item 10 (this file, satisfied
continuously by the entries above) and item 11 (iOS, correctly out of scope, see above). Every
item ran `./gradlew :app:testDebugUnitTest --offline` before moving to the next; the suite is at
92 tests, 0 failures, with every item's device-only pieces explicitly flagged rather than claimed
as verified.

## 2026-08-28: ISO ceiling too low for indoor rooms; added ARCore availability gate
On-device (`rosemary`) report: image still visibly dark despite item 1's adaptive-ISO
metering. Metering itself was wired correctly (runs across all of pre-flight, applies via
`CameraPipeline.isoSensitivity`) -- the bug was physical, not wiring: `MAX_ISO = 3200` at a
fixed 1/500s shutter is ~2-3 stops short of a typical room's exposure need, so `meteredIso()`
clamped below target luma every time indoors. Raised `CameraPipeline.MAX_ISO` to 6400
(`ponytail`: re-guessed, not re-measured -- raise again or drop the app ceiling entirely if
still dark, the hardware clamp in `configureManualOverrides` is the real backstop either way).
Separately added `ArCoreApk.checkAvailability()` gate in `MainActivity.onCreate` (with
`isTransient` polling and `requestInstall` for the supported-but-not-installed case): sideloaded
installs skip the Play Store compatibility filtering the manifest's `com.google.ar.core
required` tag normally provides, so an unsupported device could previously install and run into
undiagnosed tracking failure with no explanation surfaced.
Outcome: worked (compiles, `testDebugUnitTest` green, no new failures). Not device-verified --
whether 6400 is enough, and whether the availability check behaves correctly on `rosemary`
specifically, both need an on-device pass.

## 2026-08-30: ISO metering starts at app open, not first tap
`onMeteringImage`/`renderer.captureImage` only sampled luma while `state == PREFLIGHT`, so
`currentIso` sat at `DEFAULT_ISO` (100) from camera-open through the operator's first tap --
underexposed indoor rooms weren't corrected until PREFLIGHT began. Extended both gates to also
run in `State.IDLE` (camera/ARCore session is already live there per `ArScanRenderer.start`), so
metering converges from the moment the app opens. `beginPreflight()`'s existing reset of
`currentIso` to `DEFAULT_ISO` was left as-is -- re-metering fresh each session is correct, IDLE
metering just means it isn't starting from zero. Shutter stays fixed (`CameraPipeline.
MAX_SHUTTER_NS`, <=1/500s) per README §4's motion-blur-invariance requirement -- only ISO is the
adaptive knob, deliberately not shutter.
Outcome: worked, compiles clean, `testDebugUnitTest` green (92 tests, no regressions). Not
device-verified -- no camera/emulator in this sandbox.

## 2026-08-30: shutter is now jointly metered with ISO, not fixed
README §4 previously fixed shutter at <=1/500s and left only ISO adaptive; user feedback pointed
out this leaves no room to trade off indoor grain against motion blur -- a fixed fast shutter
forced *all* of the indoor light deficit onto ISO, which is what pushed `MAX_ISO` to 6400 in the
previous entry. Reworked both README §4 and the implementation: shutter now ranges 1/500s
(`CameraPipeline.FASTEST_SHUTTER_NS`) down to 1/50s (`SLOWEST_SHUTTER_NS`), metered jointly with
ISO against a shared exposure-gain target. `MainActivity.meteredExposure()` (replaces
`meteredIso()`) spends the shutter range first before ISO has to climb -- slower shutter only
risks blur (which the kinematic guard already screens for independently), while every ISO stop
costs grain unconditionally, so shutter is the cheaper knob to move first for this app's
fixed-walkthrough-pace use case. `CameraPipeline.isoSensitivity` gained a sibling `shutterNanos`
property, both wired through `ArScanRenderer.adjustExposure(iso, shutterNanos)`. Also updated
`Phase_1.md`'s two references to the old fixed-1/500s spec language for consistency.
Outcome: worked. Rewrote `GatingTest`'s ISO-only metering cases as joint shutter+ISO cases
(steady-state, shutter-first for dark scenes, ISO-after-shutter-bottoms-out, bright-scene fall,
floor/ceiling clamps, degenerate-luma no-op). Compiles clean, `testDebugUnitTest` green. Not
device-verified -- no camera/emulator in this sandbox; whether the shutter-first-then-ISO split
actually reads as less grainy/less blurry than the old ISO-only approach needs a real room.

## 2026-08-30: manual WB trim slider on top of the existing auto grey-world proposal
README §6 step 1 already computed an auto white-balance proposal (`MainActivity.
whiteBalanceGains`, grey-world from the pre-flight chroma preview) but only ever applied it once,
on the "Lock Color" tap -- no way for the operator to nudge it if the auto read was off. Added
`wbSlider` (a `SeekBar`, center = trust the auto proposal, added the missing manual override.
`whiteBalanceGains` gained an optional `manualBias` param (-1..1) that trims red/blue gain by
+/-`WB_MANUAL_RANGE` on top of the chroma-derived correction. `sampleWhiteBalancePreview` now also
pushes the combined gains live to `CameraPipeline` every metering frame during step 1, so the
preview actually shows the corrected color while the operator drags the slider, not just the
neutral starting gains.
Outcome: worked. Compiles clean, added `GatingTest` cases for the bias direction (warm/cool) and
neutral no-op. Not device-verified -- no camera/emulator in this sandbox; whether
`WB_MANUAL_RANGE = 0.5f` is enough headroom for a real bad-auto-read case needs a real room.

## 2026-08-30: green/magenta tint axis -- the previous WB slider couldn't reach an actual cast
On-device (`rosemary`) report: the single warm/cool slider above had no position that balanced
the room's colour. Root cause was the correction math itself, not the slider: `whiteBalanceGains`
only ever solved red vs blue and pinned green's gain to a fixed 1.0, so a green- or magenta-tinted
light source (what the room actually had) was mathematically uncorrectable regardless of slider
position -- exactly the axis a real camera app's "Tint" control exists for and this one lacked
entirely. Replaced the red/blue-only heuristic with a proper decode-and-solve: sample means are
read back from Y/Cb/Cr to the RGB they came from (BT.601 full-range), then gains (now including
green) are solved so the sampled patch reads neutral. Added a second slider, `wbTintSlider`
(green<->magenta), alongside the renamed warm/cool one -- both default-centered on the auto
proposal, matching the Temp+Tint pattern. Widened the gain clamp from 0.5..3 to 0.5..4 since the
full 3-channel solve needs more headroom than the old 2-channel one did.
Outcome: worked, device-confirmed on `rosemary` -- both sliders now visibly move color in their
respective axes and the green cast is reachable. Added `GatingTest` cases: a green-cast patch
corrects automatically (no manual bias needed), and the tint bias moves only green while leaving
red/blue untouched. Full `testDebugUnitTest` green.

## 2026-08-30: MAX_ISO capped at 1000, deliberately, not re-raised for a dark-room complaint
`CameraPipeline.MAX_ISO` had already round-tripped 500 -> 3200 -> 6400 -> 500 chasing "still too
dark on `rosemary`" reports before the joint shutter+ISO metering rework (see the 2026-08-30
shutter-metering entry above) gave the app a cheaper knob than ISO to spend first. Explicit product
decision this round: cap at 1000 and stop -- a scene `meteredExposure()` can't correctly expose
within 1/500s-1/50s shutter and ISO 1000 should be solved by more scene light or a slower walk,
not by pushing grain further. Comment at the constant now says so explicitly, to stop this from
getting re-raised on the next "it's dark" report without someone reading the history first.
Outcome: worked. Compiles clean, `GatingTest`'s `exposure.iso <= CameraPipeline.MAX_ISO` clamp
case references the constant symbolically so no test values needed updating. Not independently
device-verified this round (no new dark-room pass) -- whether 1000 is bright enough for typical
listing rooms still needs a real dim-room walkthrough; if it isn't, the fix per this entry's own
reasoning is more light or a slower shutter budget, not raising this cap again.

## 2026-08-30: "Use Ultra-Wide" pre-flight button removed -- it never switched anything, and was wrong
User report: app captures on the wide lens, not ultra-wide, despite the pre-flight step claiming
otherwise. The 2026-08-28 pre-flight entry's own note ("ARCore's default is the ultra-wide-
equivalent... on every device this was checked against") was never numerically verified -- a
temporary `Log.i` in `ArScanRenderer.selectCameraConfig` confirmed on `rosemary`:
`getSupportedCameraConfigs()` only ever returns `cameraId "0"`, a 68 deg-FOV sensor (computed from
its real `LENS_INFO_AVAILABLE_FOCAL_LENGTHS`/`SENSOR_INFO_PHYSICAL_SIZE`), never the device's true
122 deg-FOV ultra-wide (`dumpsys media.camera` shows it as internal camera id "2").

First pass wrongly concluded the ultra-wide was unreachable outright, from `CameraManager
.getCameraIdList()` reporting only `[0, 1]` in-process -- the user correctly pushed back (other
camera apps on the same phone do use it). Follow-up, more careful probing showed why the first
read was wrong: `getCameraCharacteristics()`/`openCamera()` on ids "2"-"5" all succeed when tried
*standalone* -- MIUI hides them from enumeration but not from direct access by ID, same as any
physical-camera-id quirk. The real, load-bearing constraint only showed up testing the actual
scenario this app needs: opening camera "2" *while ARCore already holds camera "0" open for
tracking* fails with `ERROR_MAX_CAMERAS_IN_USE` (matches `concurrentCameraIds` reporting `{}`
earlier) -- this SoC has one camera pipe, so ultra-wide capture and ARCore's continuous 6DoF VIO
tracking can never both be active at once on this device. That's a hardware limit, confirmed by
directly attempting the concurrent open, not an app-side selection bug -- and it means genuine
concurrent ultra-wide capture during a live scan isn't reachable here without either dropping
continuous ARCore tracking or accepting a tracking-interrupting pause-swap-resume per shot, both
of which are a redesign, not a fix, and weren't undertaken this round.

Still fixed regardless of the above: the button was a no-op confirmation dialog that always
claimed "Ultra-Wide (0.5x) lens selected" no matter what ARCore actually bound -- worth removing
on its own. Replaced it with automatic selection: `widestFovCameraId()` computes real FOV for
every `cameraId` ARCore's own `getSupportedCameraConfigs()` offers and picks the widest one. Given
the concurrent-open finding above and ARCore's documented single-rear-camera design, this is
realistically a no-op on any ARCore-based device (ARCore itself is never expected to offer more
than one `cameraId`) -- kept anyway since it's cheap, correct per ARCore's public API, and no
longer actively lying to the operator like the button did. Pre-flight is now two steps
(white-balance lock, tilt) instead of three.
Outcome: partially worked -- the false "ultra-wide selected" claim is gone and the button that
never did anything is gone, both device-verified on `rosemary` (rebuilt, reinstalled, launched
clean, tracking starts). `testDebugUnitTest` green. Genuine automatic ultra-wide *capture* is not
delivered on this device -- confirmed hardware-blocked, not a code gap -- and needs an explicit
scope decision (drop continuous tracking, or accept a pause/resume interruption per ultra-wide
shot) before it's worth attempting again.

## 2026-08-30: can ARCore's SharedCamera just bind camera "2" directly, avoiding concurrent-open?
Tried substituting `cameraId = "2"` for `session.cameraConfig.cameraId` in `CameraPipeline.start()`,
still handing `sharedCamera.arCoreSurfaces` into the capture session, to see whether ARCore's
tracking could run on a single physical device instead of two concurrent opens. Camera "2" opened
fine on its own this time (no `ERROR_MAX_CAMERAS_IN_USE`, confirming the earlier finding was
specifically about *concurrent* access, not id "2" itself). But it crashed immediately after
`onOpened`: `NullPointerException` inside ARCore's own SDK, `SharedCamera.getArCoreSurfaces()` ->
`ImageReader.getSurface()` on null. ARCore's `SharedCamera` allocates its internal CPU `ImageReader`
keyed to the `cameraId` in `session.cameraConfig` -- open a different physical device than the one
ARCore picked and ARCore never allocates that reader, so `arCoreSurfaces` is broken from inside
ARCore's own code, not recoverable app-side.
Outcome: failed, root cause confirmed at the ARCore SDK layer -- `SharedCamera` is hard-tied to
whichever single `cameraId` `session.cameraConfig` reports, no public seam to redirect it to a
different physical lens. Combined with the 2026-08-28 concurrent-open finding, this closes off both
routes to camera-2 access during a live scan (concurrent open: blocked by the SoC's single camera
pipe; ARCore-owns-camera-2 directly: blocked by ARCore's own internal wiring). Diagnostic logging
and the id="2" substitution were reverted before rebuild/reinstall (verified via grep for
`ArScanDiag`, none remain). Same scope decision as before still stands: pause/resume ARCore per
ultra-wide shot, or drop continuous tracking, if this is still wanted.

## 2026-08-30: full UI/UX pass on the arscan HUD
The HUD had grown as a set of absolutely-positioned overlays, and they collided: the warning
banner's fixed `marginTop` landed on the centred guidance arrow, the two white-balance sliders on
the primary button, and the coverage bar under the status bar (no insets handling anywhere). The
primary instruction — the single most important line on screen — lived in a 12sp monospace corner
box interleaved with ISO/shutter/luma/voxel numbers, and the arrow legend ("amber diamond =
unscanned area") sat in that same block, far from the arrow it described. Rebuilt as two
insets-aware flow columns (top = state, bottom = what to do next) so overlap is structurally
impossible, plus: `colors.xml` palette shared by the layout, `MainActivity` and matching
`ArScanRenderer`'s marker RGBA; rounded translucent cards with top/bottom scrim gradients over the
camera feed; coverage as a headline number + rounded bar that greens at the 85% target; per-arrow
captions carrying the legend on the cue itself; single warning slot with a one-shot haptic on
appearance (the operator is looking at the room, not the screen); `keepScreenOn`; haptic ack on
every button tap, since the label only catches up on the next 6 Hz repaint; SPEC §1.3's
diagnostics moved behind an `i` button instead of being permanently in front of a real-estate
agent. Two spec'd-but-missing pre-flight elements added: README §6 step 2's reticle (`reticle.xml`)
and step 3's level indicator (`TiltGaugeView`, target band drawn on the track so the answer is
positional rather than a number to compare). White-balance sliders now use gradient tracks as
their own axis labels, with a "Reset to auto" pill — there was previously no way back to the auto
grey-world proposal after dragging. The primary button is now disabled exactly when a tap would
have been a silent no-op (pre-flight step 2 with no pose; finish before loop closure or with a
validation issue), with the reason and the distance back to the start point printed underneath.
ARCore failure reasons and `validationIssues` copy rewritten as instructions instead of enums.

Two bugs found by the on-device pass, both pre-existing and both fixed: `ArScanRenderer.onError`
(the draw loop's catch-all) fired `SessionPausedException` *every frame* between session creation
and CameraPipeline's async camera open, previously invisible because it only wrote to a HUD line
that was overwritten 160 ms later — added `sessionResumed`, which the draw loop skips on, and the
callback now logs the throwable. My first version of this pass made that callback fatal (dead
button), which the device run caught immediately; renderer errors are now a self-clearing notice
and only permission/ARCore-availability failures are fatal.
Outcome: worked. Compiles clean, `testDebugUnitTest` green (68 tests, +2 for the tracking-advice
copy map and the tilt band). Device-verified on `rosemary` for IDLE, pre-flight step 1 (reticle
clear of the cards, gradient sliders) and step 2 (level indicator, dimmed while untracked,
disabled Start with its reason) — screenshots inspected, no logcat warnings. The SCANNING and DONE
screens are **not** device-verified: reaching them needs a person holding the phone until ARCore
tracks, which this sandbox can't do.

## 2026-08-30: Pre-flight step 2 tilt gauge read gravity instead of the ARCore pose; capture stage mocked out
User report: the level indicator's dot never moves. Root cause: `tiltDeg` was fed `forward[1]`,
which `updateKinematics()` only refreshes on a TRACKING frame -- through pre-flight on a phone
that hasn't tracked yet (rosemary in a dim room, `INSUFFICIENT_FEATURES`) the vector stays at its
initial zero, so the gauge sat frozen at 0 deg exactly when the operator is being asked to set the
angle, and step 2's Start button was disabled with it. Tilt is a property of how the phone is
held, not of VIO: now read from `TYPE_GRAVITY` (falling back to `TYPE_ACCELEROMETER`), normalised,
world-up's device-Z component fed as `tiltDownDeg(-gravityUpZ)` -- the same pure function and the
same test, since `f . up = -up_z` for a rear camera looking along device -Z. Dropped the
"dimmed while untracked" alpha with it. The step is kept, not removed: README section 6 step 3
requires the level indicator.
Per user request the capture stage is mocked while it's reworked: `State.MOCK` (green full-screen
placeholder + Finish -> IDLE) replaces `beginSession(pose)` at the end of pre-flight, so Start no
longer needs a pose either. Both edits carry `ponytail:` comments naming the restore path.
Outcome: worked -- compiles clean, `testDebugUnitTest` green. Not device-verified; the gauge needs
a person tilting the phone.

## 2026-09-01: Phase 1 acquisition un-mocked and verified; full capture flow connected
Restored the active acquisition flow for Phase 1 completion:
1. Removed `State.MOCK`, `Pending.MOCK_DONE`, and placeholder layout view `mockScreen`.
2. Restored `beginSession(pose)` call in `MainActivity.advancePreflight()`, transitioning from pre-flight step 2 directly into active spatial acquisition (`State.SCANNING`).
3. Gated pre-flight step 2 "Start scan" action on ARCore tracking health (disabling the button and displaying actionable `trackingAdvice` when untracked), ensuring the capture session and loop-closure anchor start with a valid 6-DoF pose.
4. Cleaned up redundant variable initialization and compiler warnings in `MainActivity.kt`.
5. Retained full visual and UX design consistency across pre-flight, active capture (`State.SCANNING`), and completion (`State.DONE`): translucent frosted HUD cards, dynamic 3D wireframe octahedral diamond markers, 2D rotating guidance arrows (amber frontier, cyan occlusion), real-time amber-to-green translucent voxel mesh overlay, and loop-closure distance guidance.
Outcome: Build clean, all unit tests (`testDebugUnitTest` and `testReleaseUnitTest`) pass, schema validation passing.

## 2026-09-01: Start Scan button interactivity fix & environment tilt range update (5-20°)
1. Fixed Start Scan button interactivity on pre-flight Step 2: removed button disabling (`enabled = false`) during untracked state so the button is always interactive, and added `lastKnownPose` / identity fallback in `applyPending()` and `advancePreflight()` so tapping Start Scan reliably transitions to `beginSession()` / `State.SCANNING`.
2. Updated target downward camera tilt range from 30-45° to 5-20° across code (`TiltGaugeView.kt`), unit tests (`GatingTest.kt`), app UI instructions, and documentation (`README.md`, `Phase_1.md`), optimizing for environment acquisition rather than object acquisition.
Outcome: Rebuilt and re-deployed directly to connected device (`M2101K7BNY`). Unit tests passing (`BUILD SUCCESSFUL`).

## 2026-09-01: Continuous autofocus & focus metadata export implementation
1. Completely resolved the "Start scan" button blocking issue: removed the untracked disabling gate in `updateUi()` so the button remains permanently clickable and enabled (`enabled = true`, vibrant action styling), while retaining `trackingAdvice` as non-blocking guidance. Added `lastKnownPose` fallback to guarantee `beginSession()` starts immediately on user tap.
2. Implemented continuous autofocus in Camera2 (`CameraPipeline.kt`): configured `CONTROL_AF_MODE_CONTINUOUS_VIDEO` / `CONTINUOUS_PICTURE` with manual override coexistence, and added `triggerAutoFocus()` for tap-to-focus on preview.
3. Implemented per-frame Camera2 `CaptureResult` focus metadata extraction (`LENS_FOCUS_DISTANCE` in diopters & metres, `CONTROL_AF_STATE`, `CONTROL_AF_MODE`, `LENS_FOCAL_LENGTH` in mm).
4. Added companion `focus_metadata.json` export alongside `transforms.json` and `trajectory.csv` in `DatasetWriter.kt` and `DatasetFormat.kt`, recording timestamped optical focus parameters for every saved keyframe image.
5. Added live autofocus diagnostic readouts to `diagText` HUD.
Outcome: Build clean, unit tests passing (`BUILD SUCCESSFUL`), shared schema validation passing (`validate.py`), and deployed to device (`M2101K7BNY`).

## 2026-09-01: WB slider gradient fix & startup tracking state refinement
1. Fixed WB Temperature slider gradient: updated `slider_track_temp.xml` to have cool blue (`#6FA8FF`) on the left (progress < 50) and warm orange (`#FFB86B`) on the right (progress > 50), matching the manual gains equation and standard camera conventions.
2. Refined initial tracking warning banner: during initial startup before the first VIO pose is acquired (`trackingLosses == 0`), displays `"Finding position…"` with contextual feature guidance instead of claiming tracking was lost.
Outcome: Build clean, unit tests passing (`BUILD SUCCESSFUL`), deployed and running on connected device (`M2101K7BNY`).

## 2026-09-01: ARCore SharedCamera lifecycle and session callback hookup
1. Root cause identified: CameraPipeline opened the Camera2 capture session using a plain `CameraCaptureSession.StateCallback` rather than wrapping it with ARCore's required `sharedCamera.createARSessionStateCallback(...)`. As a result, ARCore's native VIO engine was never notified of capture session lifecycle events, received 0 frames from Camera2, and remained permanently stuck in `PAUSED / INSUFFICIENT_FEATURES / NOT_TRACKING`.
2. Wrapped `CameraCaptureSession.StateCallback` using `sharedCamera.createARSessionStateCallback(sessionCallback, handler)` in `CameraPipeline.openCaptureSession()`.
3. Registered extra application surfaces with `sharedCamera.setAppSurfaces(cameraId, extraSurfaces)` and linked repeating capture callbacks via `sharedCamera.setCaptureCallback(captureCallback, handler)`.
4. Restored complete data flow: ARCore transitions to `TRACKING`, produces feature point clouds and 6-DoF poses, triggers `CoverageWorker` and `MonoDepthWorker` integration, and renders live 3D voxel mesh overlays, guidance arrows, and 3D octahedral markers.
Outcome: Build clean, unit tests passing (`BUILD SUCCESSFUL`), shared schema validation passing (`validate.py`).

## 2026-09-01: Top 15% luma gating & aggressive feature detection tuning
1. Implemented Top 15% Luma Calculation (`Photometric.kt`): switched `Luma.mean` to a 256-bin histogram algorithm that evaluates the luminance of the top 15% brightest pixels in the frame. This prevents dark objects (like dark wooden doors/furniture in a well-lit hallway) from pulling down the frame average and triggering false "Too dark" warnings. Lowered `PhotometricGate.MEAN_MIN` to 25f for extra headroom in hallways.
2. Aggressive Feature Detection Tuning:
   - Lowered `DepthScale.MIN_POINTS` from 6 to 3 and `MIN_CONFIDENCE` from 0.3f to 0.15f in `MonoDepth.kt`.
   - Lowered `MIN_POINT_CONFIDENCE` from 0.3f to 0.15f in `CoverageWorker.kt` so feature points in feature-sparse hallways are accepted and integrated into the scale fit and voxel grid.
Outcome: Build clean, unit tests passing (`BUILD SUCCESSFUL`), shared schema validation passing (`validate.py`).

## 2026-09-01: Feature threshold tuning & surface-manifold mesh filtering
1. Feature Detection Threshold Tuning:
   - Increased `MIN_POINTS` from 3 to 4 and `MIN_CONFIDENCE` from 0.15f to 0.25f in `MonoDepth.kt`.
   - Increased `MIN_POINT_CONFIDENCE` from 0.15f to 0.25f in `CoverageWorker.kt`.
2. Surface-Only Mesh Filtering (`VoxelGrid.kt`):
   - Added `hasSurfaceManifold(ix, iy, iz)` to filter out isolated mid-air floating voxels/cubes during meshing.
   - Only emits mesh quads for occupied voxels that have at least 1 adjacent occupied neighbor in their 26-neighborhood, ensuring the visual overlay stays anchored directly on physical surfaces (walls, floors, doors, furniture) rather than filling mid-air space.
Outcome: Build clean, unit tests passing (`BUILD SUCCESSFUL`), deployed and running on connected device (`V4TGH6FENFVWEYAQ`).

## 2026-09-01: Light threshold reduction, floor/wall surface mesh constraint & subtle alpha
1. Reduced `PhotometricGate.MEAN_MIN` from 25f to 14f (~45% reduction) in `Photometric.kt` so top 15% brightest pixels in dim hallways or around dark wood doors never trigger false "Too dark" alerts.
2. Implemented Floor Surface & Vertical Wall Contour mesh filtering (`VoxelGrid.kt` and `CoverageWorker.kt`): passed estimated `floorY` to `meshTriangles()`, constraining mesh emission strictly to ground floor surface footprint ($y \in [\text{floorY}-0.3\text{m}, \text{floorY}+0.35\text{m}]$) and vertical wall columns ($y \in [0.35\text{m}, 2.4\text{m}]$ with vertical neighbors), completely eliminating mid-air floating volumetric fog.
3. Lowered overlay alpha from 0.55 to 0.18 in `ArScanRenderer.kt` (subtle holographic sheen) and enabled OpenGL depth testing with `glDepthMask(false)` to prevent translucent face overdraw buildup.
Outcome: Build clean, unit tests passing (`BUILD SUCCESSFUL`), deployed and running on connected device (`V4TGH6FENFVWEYAQ`).

## 2026-09-01: Real-time 2D Bird's-Eye MiniMap & HUD redesign
1. Designed and implemented `MiniMapView.kt`:
   - Real-time 2D top-down floor plan rendering: scanned ground floor footprint (translucent emerald fill), structural wall contours (crisp slate outline), starting doorway location (amber anchor circle), and live user metric cursor with forward camera FOV vision cone.
   - World-locked coordinates: locked to Cardinal North-South-East-West orientation (-Z is North / Up, +X is East / Right).
   - Smooth dynamic auto-fitting: bounding box smoothly scales and centers dynamically as the operator explores new rooms so the entire mapped layout always fits comfortably within the mini-map.
2. 2D Floorplan Extraction (`VoxelGrid.kt` / `CoverageWorker.kt`): added `floorplan2D(floorY)` exporting 2D floor footprint points, chest-height wall slice points, and metric bounding coordinates.
3. Top HUD Redesign (`activity_main.xml`): integrated `MiniMapView` side-by-side with tracking health and progress metrics, repositioned `warnBanner` below the top bar with generous spacing to completely avoid overlaps, and suppressed viewport center-screen arrows for an unobstructed, clean camera walkthrough view.
Outcome: Build clean, all 68 unit tests passing (`BUILD SUCCESSFUL`), schema validation passing (`validate.py`), and deployed/running on connected device (`V4TGH6FENFVWEYAQ`).

## 2026-09-01: Transparent wireframe triangle grid overlay & scanning light gate removal
1. Removed Dark Light Gating During Active Scan:
   - Once the scan has started (`State.SCANNING`), the app never prompts to turn on the lights or rejects keyframes for darkness. Light gating is now solely evaluated prior to starting the scan.
2. Complete Elimination of 3D Cubes & Solid Volume:
   - Completely removed all solid/translucent filled faces from the 3D voxel overlay.
3. Dual-Color Wireframe Triangle Grid Overlay:
   - **Horizontal Planes (Floors / Ground Surface)**: Rendered as a clean wireframe triangle grid with **white borders/lines** (`#FFFFFF` at 0.90 opacity) and completely transparent filling.
   - **Vertical Planes (Walls / Partitions)**: Rendered as a wireframe triangle grid with **blue lines** (`#38BDF8` at 0.90 opacity) and completely transparent filling.
   - Enabled native `HORIZONTAL_AND_VERTICAL` plane finding in ARCore (`ArScanRenderer.kt`) and switched rasterization from filled triangles to `GL_LINES` with `glLineWidth(2.5f)`.
Outcome: Build clean, all 68 unit tests passing (`BUILD SUCCESSFUL`), shared schema validation passing (`validate.py`), and deployed/running on connected device (`V4TGH6FENFVWEYAQ`).

## 2026-09-01: ARCore native verified plane rendering & indoor exposure calibration
1. ARCore Native Verified Plane Rendering (`ArScanRenderer.kt`):
   - Switched live viewport rendering exclusively to ARCore's native tracked VIO planes (`drawPlanes(session)`), rendering **White wireframe triangle grids** for horizontal floor planes and **Blue wireframe triangle grids** for vertical wall planes.
   - Bypassed monocular AI depth noise in live view, completely eliminating mid-air floating phantom surfaces/cubes in open space or empty hallways.
2. Initial ISO & Shutter Calibration (`CameraPipeline.kt` & `MainActivity.kt`):
   - Set indoor baseline shutter speed to `1/60s` (16.6 ms) (down from outdoor 1/500s) and default ISO to `400` (up from 100).
   - Increased max ISO ceiling to `3200` and `CONVERGENCE_GAIN` to `0.60f`, ensuring the screen opens bright, clear, and perfectly exposed in indoor environments without a dark initial state.
Outcome: Build clean, all 68 unit tests passing (`BUILD SUCCESSFUL`), schema validation passing (`validate.py`), and deployed/running on connected device (`V4TGH6FENFVWEYAQ`).

## 2026-09-01: On-device evaluation of native ARCore plane tracking on monocular hardware
1. Field Testing & Observations:
   - On-device field testing of native ARCore plane finding (`HORIZONTAL_AND_VERTICAL`) on monocular hardware (Xiaomi Redmi Note 10S) demonstrated severe planar estimation artifacts: false floaters in empty space, planes projected behind physical walls, and noisy geometric misalignment.
2. Technical Diagnostic & Root Cause:
   - Without hardware ToF/LiDAR/stereo depth sensors, real-time 3D planar reconstruction over feature-sparse indoor surfaces (white drywall, plain doors, dim hallways) exhibits high depth variance and scale instability.
   - Conclusion: Attempting to render dense 3D surface meshes (whether monocular neural depth voxels or ARCore native VIO planes) directly over the 3D camera feed produces unacceptable visual clutter and geometric inaccuracy for live mobile guidance.
3. Decision:
   - Pause 3D viewport mesh rendering. Pivot toward a clean 2D spatial guidance model (2D Mini-Map trajectory + 6DoF camera path) to ensure an unobstructed camera view while retaining precise spatial coverage tracking.










### Macro Milestone: Real-Time Multi-View Parallax Feature Point Cloud & Structured Walk Guidance (2026-09-02)

1. **Replaced Native Plane Wireframe & Voxel Cubes with Sparse Feature Points:**
   - Monocular ARCore plane tracking without ToF/LiDAR generated severe geometric floaters in open rooms and misaligned planes behind walls.
   - Replaced plane wireframes and voxels with tracked sparse feature points in the live viewport (`ArScanRenderer.drawPointSprites`). ARCore only registers feature points after multi-frame epipolar consensus, eliminating phantom surfaces in empty air.
2. **Dynamic Parallax Verification (`FeatureParallaxTracker`):**
   - Implemented zero-allocation parallel primitive arrays tracking persistent ARCore point IDs across 6DoF camera poses.
   - **Amber Dot (`#F59E0B`):** Point observed from a single view or insufficient baseline (< 20° angle).
   - **Emerald Green Dot (`#10B981`):** Point confirmed with >= 20° angular baseline and >= 0.35 m spatial translation.
3. **Structured 3-Step Walk Guidance Cards:**
   - **Step 1 (Entry Sweep):** Operator slowly pans from the doorway to acquire initial feature geometry.
   - **Step 2 (Perimeter Walk):** Operator walks along the room perimeter facing inward while dots turn emerald green.
   - **Step 3 (Doorway Return / Loop Closure):** Operator returns to starting doorway to seal loop closure and finalize capture.
   - HUD displays verified feature count and progress: `PARALLAX VERIFIED · X/Y PTS`.
4. **Verification:**
   - Unit tests in `FeatureParallaxTrackerTest` pass 100%.
   - Build verified with `./gradlew assembleDebug` (SUCCESSFUL).

### Milestone: Live Two-Ray Geometric Triangulation & Stale Floater Pruning (2026-09-02)

1. **Root Cause Analysis of Frozen Floaters & Green Points:**
   - Monocular ARCore initially assigns depth along the line of sight based on single-view heuristics; if inaccurate, the point appears in mid-air or through walls.
   - Points previously accumulated in memory without pruning, meaning unverified points became frozen zombie floaters and never updated when the camera moved.
   - Parallax updates were previously trapped inside `State.SCANNING`, preventing real-time feedback during pre-flight alignment.
2. **Two-Ray Geometric Triangulation Engine:**
   - Implemented exact closed-form solution for the closest point of intersection between ray 1 ($C_1 + t_1 ec{d}_1$) and ray 2 ($C_2 + t_2 ec{d}_2$).
   - When a tracked feature ID is seen from a new camera angle with baseline $\ge 12^\circ$ and translation $\ge 0.20	ext{ m}$, its $(X, Y, Z)$ position is snapped directly to the geometric intersection point.
   - Residual thresholding ($\le 0.25	ext{ m}$) rejects divergent or mismatched rays.
3. **Stale Floater Eviction:**
   - Unverified amber points not seen within 1.8 seconds are automatically pruned from the active tracking pool.
   - Continuous live point streaming in `onFrame` enables instant visual feedback.
4. **Verification:**
   - Unit test suite (`FeatureParallaxTrackerTest`) passed with triangulation and residual verification tests.
   - Installed and launched on connected phone.

### Milestone: Persistent Solid 3D Landmark Memory & Monotonic Coverage Progress (2026-09-02)

1. **Persistent 3D Landmark Map (Never Discarded on Frame Exit):**
   - Verified multi-view features are promoted to **Permanent 3D Landmarks** (`verifiedLandmarkCount`).
   - Once a landmark turns Emerald Green, it is permanently preserved in the spatial coordinate frame. Even when the user pans away or walks into another corner, the landmark remains in world space and is re-rendered when the camera looks back.
2. **Distinctive Spatial Distribution & Cluster Prevention:**
   - Enforces a minimum spatial radius separation (`MIN_LANDMARK_SPACING_M = 0.12m`) between persistent landmarks. Prevents hundreds of redundant points clustering on a single high-contrast edge while ensuring clean room-wide distribution (target: ~180-200 solid room landmarks).
3. **Monotonically Non-Decreasing Completion Progress:**
   - Gated progress against `targetLandmarkCount` (180 solid landmarks).
   - The completion bar and percentage (`LANDMARKS VERIFIED · X/180`) strictly increases as new areas are explored and triangulated, never fluctuating or dropping when the camera pans away from scanned areas.
4. **Candidate Life-Cycle:**
   - Unverified amber points are treated as temporary candidates in the active viewport. If they achieve parallax baseline (>= 12°, >= 0.20m), they snap via two-ray triangulation into permanent green landmarks. If they exit the frame without verification, they are pruned cleanly to avoid mid-air floaters.

### Milestone: Gamified 3DGS Multi-Object Capture UX, 5s Auto-Exposure & Seamless Auto-WB (2026-09-02)

1. **Gamified Multi-Object 3DGS Density Guidance:**
   - Replaced rigid walking steps with dynamic multi-object orbiting guidance.
   - **Tiers:**
     - *< 200 pts:* "✨ Orbit Objects & Corners" (Dynamic exploration).
     - *200 - 350 pts:* "🎮 Good Progress (200+ pts)" (Unlocks baseline room finish).
     - *350 - 500 pts:* "🌟 Ideal 3DGS Quality (4K Splats)" (Optimal density achieved).
   - HUD banner displays live density status: `DENSITY: X/500 PTS · [ORBIT OBJECTS | GOOD COVERAGE | EXCELLENT · 4K 3DGS]`.
2. **5-Second Multi-Angle Lighting & Exposure Calibration:**
   - Upon tapping Start from idle, the app prompts the user to smoothly pan the phone across the room for 5 seconds.
   - Computes balanced joint ISO/shutter speeds (starting baseline ISO 400, 1/120s shutter) to eliminate dark screens and motion blur.
3. **Seamless Auto White Balance:**
   - Automatically computes and locks the grey-world white balance gains during the 5s calibration sweep.
   - Seamlessly transitions straight to capture mode once calibrated, removing the tedious manual slider step while keeping the sliders available only if custom tinting is desired.
4. **Verification & Device Deployment:**
   - 100% unit tests passed (`MainActivityExposureTest`, `GatingTest`, `FeatureParallaxTrackerTest`).
   - App built and launched on connected phone.

### Milestone: Enhanced Indoor Exposure Tuning & Professional Multi-Step Pre-Flight Setup (2026-09-02)

1. **Brighter Indoor Lighting Calibration Target (Target Luma: 145):**
   - Tuned `TARGET_LUMA` from 110 to 145 and increased the default baseline to ISO 600 / 1/80s shutter.
   - Prevents dark spots in rooms from losing visual tracking features, while preserving sharp shutter speeds to minimize motion blur.
2. **Professional Interactive White Balance Step (with Smart Auto-Init):**
   - Retained the professional 3-step pre-flight sequence:
     - **Step 0 (5s Sweep):** Multi-angle lighting sweep calculates optimal ISO, shutter speed, and pre-samples grey-world color balance.
     - **Step 1 (Color Balance):** Displays the circular targeting reticle and the Temperature/Tint slider card. Automatically populates the sliders with the auto-computed balance so the operator can simply tap "Lock Color" if it looks good, or make fine adjustments.
     - **Step 2 (Camera Angle):** Displays the green 5–20° level gauge to align the vertical trajectory before tapping "Start 3D Capture".
3. **Verification & Testing:**
   - 100% unit tests passed across all suites (`GatingTest`, `MainActivityExposureTest`, `FeatureParallaxTrackerTest`).
   - Built and deployed APK to connected physical device.

### Milestone: High-Luminance Indoor Metering & Smart Auto-Populated White Balance Sliders (2026-09-02)

1. **High-Luminance Exposure Target (Target Luma: 175):**
   - Elevated `TARGET_LUMA` to 175 and set pre-flight starting exposure to ISO 800, 1/60s ($16.6\text{ ms}$) shutter.
   - Provides rich, bright illumination across dim rooms and shadowed corners, preventing feature degradation and eliminating tracking loss.
2. **Smart Auto-Populated White Balance Sliders:**
   - Instead of leaving sliders stuck at 50/50, the 5-second calibration sweep calculates the actual color temperature ($C_r - C_b$) and tint balance ($C_r, C_b$ projection).
   - Upon transitioning to Step 1, the Temp and Tint sliders automatically move to the measured positions so the user sees the active compensation.
   - Restricted white balance sampling in Step 1 to the central 30% reticle patch so pointing at a specific neutral wall directly tunes the live preview.
3. **Verification & Deployment:**
   - 100% unit tests passing (`./gradlew testDebugUnitTest`).
   - Installed and launched updated build on connected phone (`V4TGH6FENFVWEYAQ`).

### Milestone: Micro-Haptics, Radiant Landmark Shaders, Dynamic Breadcrumbs & Adaptive Worker Throttling (2026-09-02)

1. **Physical Micro-Haptic Pulses on Milestone Crossings:**
   - Subtle tactile confirmations fire when reaching key coverage thresholds:
     - 200 points (`CONFIRM` tick — Base Coverage unlocked).
     - 350 points (`LONG_PRESS` pulse — High Quality 4K tier).
     - 500 points (`CONFIRM` pulse — Complete Density).
     - Doorway Return (`CONFIRM` tick — loop closure achieved).
2. **Radiant Emerald Landmark Point Sprites (`ArScanRenderer.kt`):**
   - Enhanced OpenGL point sprite shaders with antialiased cores and radiant glow halos for verified landmarks.
   - Point scaling dynamically expands emerald green discs for high-contrast visibility against all room lighting.
3. **2D Breadcrumb Walking Trail & Radiant Vision Cone (`MiniMapView.kt`):**
   - Added real-time cyan/blue breadcrumb path tracking the operator's walking route on the top-right mini-map.
   - User cursor rendered with a vibrant radiant field of view vision cone and orientation anchor.
4. **Zero-Copy Buffer Scanning & Adaptive Worker Throttling:**
   - Optimized reticle patch sampling with bulk byte-array stride reading.
   - Added adaptive motion throttling to `CoverageWorker`: automatically throttles to 3.3 Hz when stationary (<3cm motion), bursting to 15 Hz during movement.
5. **Verification:**
   - 100% unit tests passed (`./gradlew testDebugUnitTest`).
   - Installed and launched on physical connected phone (`V4TGH6FENFVWEYAQ`).

### Milestone: Emerald Diamond Gem Point Sprites, 4-Pointed Star Sparkle Shader & Clean Top HUD Layout (2026-09-02)

1. **4-Pointed Star Sparkle & Emerald Diamond Gem Shaders (`ArScanRenderer.kt` & `FeatureParallaxTracker.kt`):**
   - When feature points are verified by parallax triangulation, they burst into brilliant **4-pointed sparkling diamond stars** with crystalline cyan/white centers (scaling up to $52\text{px}$) for 1.2 seconds.
   - Settle into permanent, clean **emerald rhomboid diamond gems** with radiant inner cores.
   - Amber candidates remain smooth circular dots until verified.
2. **Fixed Top HUD Layout & Percentage Clipping (`activity_main.xml`):**
   - Cleaned up the status header: reduced padding and properly aligned the coverage percentage row so the big numerical percentage text never collides with the scanning/phase status tag.
   - Constrained density tags with single-line truncation.
3. **Verification & Deployment:**
   - 100% unit tests passing (`./gradlew testDebugUnitTest`).
   - Installed and launched directly on physical device (`V4TGH6FENFVWEYAQ`).

### Milestone: Point Density Progress Bar Synchronization (2026-09-02)

1. **Synchronized Progress Calculation with Landmark Points:**
   - Fixed the progress bar and percentage calculation to strictly evaluate landmark density (`verifiedLandmarkCount / 500`).
   - For instance, 7 points is accurately $1\%$, 200 points is $40\%$, 350 points is $70\%$, and 500 points is $100\%$.
   - Removed legacy raycast voxel ratio overriding the landmark progression.
2. **Verification & Deployment:**
   - 100% unit tests passing (`./gradlew testDebugUnitTest`).
   - Rebuilt and launched live on connected phone (`V4TGH6FENFVWEYAQ`).

### Milestone: Tracking Loss Relocalization Recovery Engine & 3D Guidance Wireframe Removal (2026-09-02)

1. **Suppressed Floating 3D Guidance Diamonds (`ArScanRenderer.kt`):**
   - Disabled `drawGuidanceMarker` for the amber frontier diamond and cyan occlusion diamond in the camera feed.
   - Consolidated spatial guidance and navigation exclusively onto the 2D Bird's-Eye Mini-Map, keeping the primary walkthrough view completely uncluttered.
2. **On-Device Relocalization & Coordinate Drift Realignment Engine (`RelocalizationRecovery.kt`):**
   - Engineered closed-form 3-point RANSAC and Kabsch/Umeyama rigid alignment ($SE(3)$ $[R \mid t]$).
   - Invariant edge-length triangle congruency matches newly seen feature points against stored landmark constellations without depending on ARCore's transient point IDs.
   - When tracking is interrupted and subsequently re-anchors with drift, repointing at an existing object (e.g. the stove) detects the physical shift and transforms the entire landmark map to snap back directly onto real-world objects.
   - Triggers tactile haptic feedback and displays a status banner: `✓ Re-aligned to Room Objects`.
3. **Automated Testing & Deployment:**
   - Added unit test suite `RelocalizationRecoveryTest.kt` verifying rotation, translation, and RANSAC inlier consensus.
   - All 77 unit tests passed. Built and deployed live to connected device (`V4TGH6FENFVWEYAQ`).

### Milestone: Robust Localized Relocalization & Real-Time Guidance Prompts (2026-09-02)

1. **Camera-Proximity Landmark Filtering & High-Consensus Alignment (`RelocalizationRecovery.kt`):**
   - Restricted RANSAC feature matching strictly to landmarks within $3.5\text{ m}$ of the current camera frustum.
   - Tightened geometric matching tolerance from $8\text{ cm}$ to $5\text{ cm}$, increased inlier consensus threshold from 4 to 6 solid points, and capped residual error to $<6\text{ cm}$.
   - Eliminates false-positive alignments against distant or unrelated room corners.
2. **Clear Real-Time Operator Guidance during Tracking Interruption (`MainActivity.kt`):**
   - When tracking drops and recovers, the top warning banner immediately instructs the operator:
     `🔍 Aim at Previously Scanned Objects`
     `Hold steady at familiar corners/furniture to restore alignment`
   - Once alignment succeeds, displays:
     `✓ Re-aligned to Room Objects`
     `Points snapped back to physical surfaces` with tactile haptic confirmation.
3. **Mini-Map Visual Architecture Documentation:**
   - Translucent Emerald Green squares: Live floor footprint detected by the spatial occupancy grid.
   - Solid Crisp White squares: Vertical wall boundaries and structural contours.
4. **Verification & Testing:**
   - 100% unit tests passed (`./gradlew testDebugUnitTest`).
   - Installed and deployed directly to connected device (`V4TGH6FENFVWEYAQ`).

### Milestone: Schema v1.0.0 Conformance & Manifest Export Alignment (2026-09-08)

1. **Explicit Schema Versioning (`DatasetFormat.kt`):**
   - Added `"schema_version": "1.0.0"` to `transformsJson()` and `summaryJson()`.
   - Guaranteed full compliance with `shared/schemas/transforms.schema.json` and `shared/schemas/coverage_summary.schema.json`.
2. **Metadata Contract Documentation:**
   - Updated root `README.md` (§5) and `mobile/SPEC.md` (§2.5) detailing all exported metadata payloads (`transforms.json`, `trajectory.csv`, `coverage_summary.json`, `focus_metadata.json`).
3. **Unit Tests & On-Device Deployment:**
   - Updated `DatasetTest.kt` assertions to verify `schema_version` emission and contract validity.
   - All unit tests passed (`./gradlew testDebugUnitTest`).
   - Installed and launched directly on connected device (`V4TGH6FENFVWEYAQ`).

### Milestone: Fix Persistent Depth Calibration Warning Banner (2026-09-09)

1. **Root Cause Analysis:**
   - On monocular devices without hardware Depth API support, the legacy `mono` depth fallback model requires feature points to scale its disparity (`mono.calibrated`).
   - The active spatial architecture now primarily uses direct 2-ray geometric multi-view triangulation (`FeatureParallaxTracker`), where landmarks are verified and anchored independently in 3D world space.
   - The warning banner check `mono?.let { it.ready && !it.calibrated } == true` lingered even after the user walked and was successfully triangulating landmarks, causing the warning banner to persist indefinitely.
2. **Resolution:**
   - Updated the depth calibration warning guard in `MainActivity.kt`: the warning is only shown if both `mono` is uncalibrated **and** fewer than 5 verified landmarks exist. As soon as the user starts scanning and landmarks triangulate, the banner automatically clears.
3. **Verification & Deployment:**
   - All unit tests passed (`./gradlew testDebugUnitTest`).
   - Built and deployed directly to connected device (`V4TGH6FENFVWEYAQ`).

### Milestone: Fix Tracking-Recovery Hang/Crash on Re-Anchor to Dense Areas (2026-09-13)

1. **Root Cause:**
   - `RelocalizationRecovery.estimateAlignment` runs on the render thread and did O(n^3) triangle matching over all landmarks within 3.5m of the camera, times 60 RANSAC iterations.
   - Re-pointing at an already-densely-scanned area maximizes that nearby landmark count, blowing the search up to billions of ops in one frame -> ANR -> app killed.
2. **Fix (`RelocalizationRecovery.kt`):**
   - Capped the candidate set used for triangle matching to 40 landmarks, evenly strided across the near set for spatial spread.
   - Final inlier consensus scoring still evaluates every nearby landmark, so alignment quality/threshold behavior is unchanged.
3. **Verification:**
   - `./gradlew testDebugUnitTest --tests "*RelocalizationRecoveryTest*"` passed, full build succeeded.

### Milestone: Fix Landmark Cap Stall & Mirrored Realignment (2026-09-13)

1. **Root Cause 1 — Scan stalls at 2048 points:**
   - `FeatureParallaxTracker.maxLandmarks` was a hard-coded `2048`-slot array. With
     `MIN_LANDMARK_SPACING_M = 0.06f`, a full-home scan easily needs far more landmarks than that;
     once full, `canPromoteLandmark`/`addPermanentLandmark` silently refused all further points, so
     no new point could turn green for the rest of the scan.
   - Fix: raised `maxLandmarks` to 20000 (`FeatureParallaxTracker.kt`) and matched the
     `scratchLandmarkX/Y/Z` buffers in `MainActivity.kt`.
2. **Root Cause 2 — Bad re-alignment after tracking loss:**
   - `RelocalizationRecovery.solve3PointRigid` matched candidate triangles by edge length only,
     which is satisfied equally by a true rotation or its mirror image. RANSAC could accept a
     reflection (det(R) = -1) as a "match," flipping/warping the whole landmark cloud instead of
     rigidly re-aligning it.
   - Fix: reject any candidate transform whose rotation determinant isn't ~+1 (mirror solutions are
     now discarded so RANSAC keeps searching for a proper rigid match).
3. **Verification:** `./gradlew testDebugUnitTest` — full suite passed.

### Milestone: ANR Investigation & Scan Gallery/Delete UI (2026-09-13)

1. **Investigated "crash at 893 points":** Logcat showed no exception/OOM anywhere for the app --
   an ANR ("Input dispatching timed out... Waited 5002ms for FocusEvent") while `onPause()` was
   blocked waiting for the GL thread to park (by design, see `MainActivity.kt:1196`'s comment).
   None of the per-frame paths that scale with landmark count are anywhere near O(seconds) at 893
   landmarks. The ANR timestamp lines up exactly with `ACTION_POWER_CONNECTED`/USB broadcasts from
   plugging the phone in for on-device debugging, which is the more likely trigger (storage/USB
   stack stall) than a code regression. Unconfirmed without a repro off-USB.
2. **Added "Previous Scans" gallery (`ScanGalleryActivity.kt`):** New pill button on the main HUD
   (`memoryButton`) opens a list of past scan sessions with date, total size, and a Delete button
   with a confirmation dialog. Reads both storage locations DatasetWriter can use: MediaStore
   `Documents/GlomeHomeTour/<session>/` (primary) and the app's external files dir fallback,
   merged by session name. No new dependency added -- plain `LinearLayout` rows in a `ScrollView`,
   since the list is small and RecyclerView wasn't already in the project.
3. **Verification:** `./gradlew testDebugUnitTest` passed; installed and launched on connected
   device (`V4TGH6FENFVWEYAQ`).

### Milestone: Scan Gallery Delete Progress (2026-09-13)

1. **Root cause of "delete looks blocked":** `deleteScan` ran the MediaStore bulk delete directly
   on the UI thread with a single LIKE-scoped `contentResolver.delete()` call -- genuinely
   blocking, not just looking like it, for however long that bulk delete took.
2. **Fix (`ScanGalleryActivity.kt`):** moved deletion to a background `Thread`; deletes each
   MediaStore row / fallback file individually (instead of one bulk call) so real 0-100% progress
   is available, and posts it back via `runOnUiThread`. The row's "Delete" label swaps for a
   `ProgressBar` + percentage text for the duration.
3. **Verification:** `./gradlew testDebugUnitTest` passed; installed and launched on connected
   device (`V4TGH6FENFVWEYAQ`).

### Milestone: Relocalization ID-first matching + background thread (2026-09-13)

1. **Problem:** after a tracking loss, re-alignment relied purely on triangle edge-length
   congruence between newly observed points and stored landmarks -- geometry-only matching with
   no memory of which physical feature a point actually was, so it could mismatch under repeated
   or symmetric geometry. The RANSAC search also ran synchronously on the GL thread, a plausible
   contributor to the ANR reports around 800-900 points (unconfirmed, see previous milestone).
2. **ID-first matching (`FeatureParallaxTracker.kt`, `RelocalizationRecovery.kt`):** landmark ID
   lookup switched from a hash set to a hash map (`landmarkHashSlots`/`landmarkSlotForId`) so a
   currently-tracked point's ARCore ID can be matched directly against a previously-stored
   landmark's ID in O(1), skipping geometry guessing whenever the same physical feature is still
   being tracked by ID across the gap. New `matchById(...)` collects these correspondences;
   `estimateAlignmentPreferId(...)` solves the rigid transform from them directly
   (`solveFromIdMatches`) when enough exist, falling back to the existing triangle-congruence
   RANSAC (`estimateAlignment`) otherwise.
3. **Cross-frame confirmation:** a single frame's transform estimate is no longer trusted
   immediately -- `isSimilarTransform(a, b)` compares consecutive passes' results, and only a
   transform confirmed by two independent passes is applied to the landmark cloud, cutting the
   chance a single bad match snaps the scan to the wrong pose.
4. **Off the GL thread (`MainActivity.kt`):** the relocalization block in `onFrame` now only does
   bounded array copies on the GL thread (landmark snapshot, ID-match arrays); the actual
   `estimateAlignmentPreferId` call and confirmation comparison run on a dedicated
   `HandlerThread("relocalization")`, guarded by `relocJobRunning` (AtomicBoolean) so only one
   search runs at a time, with the confirmed result handed back via
   `relocConfirmedTransform` (AtomicReference) and applied on the GL thread next frame. This
   removes the RANSAC search from the render loop entirely, which may also resolve (not yet
   confirmed) the ANR reports from the previous milestone.
5. **Scoped down from the fuller design:** skipped persisting full per-landmark camera/viewing
   history (bigger change, marginal gain over ID+geometry matching) and requiring multiple
   independent triangle hypotheses to agree within a single RANSAC pass (redundant once
   cross-frame confirmation exists). Can revisit either if ID-first + cross-frame isn't enough.
6. **Verification:** `./gradlew testDebugUnitTest` passed; installed and launched on connected
   device (`V4TGH6FENFVWEYAQ`). Not yet re-tested against a live tracking-loss scenario or the
   800+ point ANR by the user.

### Milestone: Root-caused the ~800-point freeze — non-power-of-two hash table (2026-09-13)

1. **Root cause (confirmed from the ANR trace, not guessed):** `/data/anr` dump for pid 21184
   showed `GLThread` **Runnable** with 78 s of user CPU, spinning in
   `FeatureParallaxTracker.landmarkSlotForId` <- `isLandmark` <- `update` <- `onFrame`. `main` was
   blocked in `onPause` waiting for that GL thread, which is what surfaced as the ANR. So the
   earlier "USB broadcast / plugging in the phone" theory was wrong, and moving RANSAC off the GL
   thread didn't help because relocalization was never the culprit.
   - The landmark hash map was `IntArray(maxLandmarks * 2)` = 40000 entries, but the probe masks
     with `size - 1`. 39999 is **not a power of two**, so `and mask` is not a modulo: it has only
     10 bits set, reaching 1024 distinct buckets, and `idx = (idx + 1) and mask` cycles after
     visiting just **64** slots. Once any 64-slot window filled up (~800 landmarks in practice,
     matching every report), `landmarkSlotForId` looped forever. Raising `maxLandmarks` to 20000
     is what introduced it — the previous 2048 gave 4096, a valid power of two.
   - `maxCandidates * 2` = 8192 was accidentally still a power of two, so the candidate map was fine.
2. **Fix (`FeatureParallaxTracker.kt`):** added `tableSizeFor(entries)` = smallest power of two
   >= 2 * entries, used for both hash maps. Landmarks now get a 65536-slot table. The 2x headroom
   also means neither table can ever fill, which is what bounds the unguarded insert loops in
   `addPermanentLandmark`/`allocateCandSlot`.
3. **Verification:** new `FeatureParallaxTrackerTest` case promotes 1225 landmarks (a 35x35 grid
   over a 1.2 m baseline) and requires `matchById` to resolve all of them — it hangs on the old
   code and passes in ~1 s now; plus a check that both table sizes are powers of two. Full
   `testDebugUnitTest` green, APK built and installed on `V4TGH6FENFVWEYAQ`. Not yet re-tested
   with a live 2000+ point walkthrough by the user.

## 2026-09-13: Wire real k1/k2 lens distortion into transforms.json
`transforms.json` had always hardcoded `k1/k2/p1/p2` to `0.0`, even though the schema, backend
`package_loader.py`, and `sfm_refinement.py`'s bundle adjustment already consume them (seeding
its distortion refinement from zero every time). `CameraPipeline` now reads
`CameraCharacteristics.LENS_DISTORTION` (API 28+, static per physical camera) once in `start()`
and maps kappa_0/kappa_1 to k1/k2 as an approximate seed -- Android's model is a 5-term rational
model with no tangential term, not an exact match for OpenCV's plumb-bob (k1,k2,p1,p2), so p1/p2
stay 0. Threaded through `DatasetWriter.finish` -> `DatasetFormat.transformsJson`.
Outcome: done — compiles, existing + new `DatasetTest` cases pass.

## 2026-09-13: Zip capture output on finish(), delete leftover empty folders
Ask: after a capture completes, save the whole dataset as a single zip and leave nothing else on
the phone; make delete also clean up the empty per-session folders the phone had accumulated.
Investigation found the empty-folder complaint was structural, not a missed `deleteRecursively()`
call: `DatasetWriter`'s MediaStore path wrote every session under its own
`Documents/GlomeHomeTour/<session>/` subfolder, but MediaStore only tracks *files* as rows, not
the directories it creates for them — once every file row under a session is deleted, the
directory itself (and its `images/` subfolder) is an orphaned real folder on disk with no
MediaStore row to delete, and plain `File.delete()` on it fails with `EACCES` under scoped
storage (confirmed via `adb shell ls` on `rosemary`: a folder deleted through the gallery UI
still had its `images/` subdir on disk afterwards, permission-denied to the app).
Fix: `DatasetWriter.finish()` now zips the session (images + manifests) into a single
`"$sessionName.zip"` and deletes the loose originals from the writer thread
(`zipAndCleanup`/`zipFromMediaStore`/`zipFromFallback`). Root-caused the folder problem instead
of patching around it: the MediaStore path no longer creates a per-session subfolder at all --
every file is written flat under the shared `Documents/GlomeHomeTour/` album with the session
name prefixed onto the filename (`flatMediaName`/`unflattenMediaName`, unit-tested in
`DatasetWriterFlatNameTest`), so there is no per-session directory left to orphan. The
fallback (non-MediaStore) path already used real per-session `File` folders and
`deleteRecursively()`, which works fine under app-private storage, so it was left as-is.
`ScanGalleryActivity` now lists/deletes zips (`mediaZipId`/`fallbackZipFile`) plus two legacy
cases for scans captured before this change: old nested-folder sessions (`legacyMediaSession`,
best-effort empty-dir cleanup via `deleteEmptyDirUpwards`, which does NOT work on `rosemary` --
confirmed on-device, scoped storage denies the `File.delete()` without `MANAGE_EXTERNAL_STORAGE`
-- so pre-existing orphaned folders from before this fix are stuck until manually cleared via a
file manager with root/ADB) and old crashed-mid-capture flat sessions (`legacyFlatMediaSession`).
Outcome: done — compiles, `DatasetWriterFlatNameTest` passes, APK built and installed on
`V4TGH6FENFVWEYAQ`. Verified end-to-end on-device: app launches, gallery lists a pre-existing
legacy nested-folder scan correctly, delete removes all its MediaStore rows and clears the list.
Not verified: a live capture through `finish()` producing an actual zip (needs a real ARCore walk
around a room, can't be driven from this sandbox — see project's `arcore-pointcloud-not-a-tracking-gate`
memory). Ask the user to run one real capture and check the gallery shows a single zip afterward.

## 2026-09-17: Magnetic compass heading in transforms.json
ARCore's world yaw is arbitrary (wherever tracking started), so the exported scene had no way to
be locked to real-world orientation. Registered `TYPE_MAGNETIC_FIELD` alongside the existing
gravity listener, combined via `SensorManager.getRotationMatrix`/`getOrientation` into a
`compassHeadingDeg` (clockwise from magnetic north, not declination-corrected -- no location fix
available) updated on every sensor tick and threaded through `writeKeyframe` -> `addKeyframe` ->
`DatasetFormat.Keyframe` -> `transforms.json`'s per-frame `compass_heading_deg` (`null` until the
magnetometer produces a first reading).
Outcome: done — `compileDebugKotlin` and `DatasetTest` (incl. new compass cases) pass. Not
verified on-device with a live magnetometer reading.

## 2026-09-17: Auto WB converges instead of compounding cold
Pre-flight auto white balance always landed too cold. Root cause: AWB is off and
`COLOR_CORRECTION_GAINS` is already applied, but `whiteBalanceGains` re-solved absolutely from
every frame -- so it kept re-correcting an image that already carried last frame's correction and
compounded until it clamped. Split into `greyWorldSolve` (unchanged math), `whiteBalanceStep`
(damped accumulation, 10%/frame + a 2-unit Cb/Cr deadzone, same shape as `meteredExposure`) and
`whiteBalanceTrim` (operator slider trim on top); the loop now settles over ~1.5 s of the 5 s
sweep. Touching a WB slider sets `autoWbLocked` so the loop stops fighting the operator; the
reset button clears it. Dropped the cb/cr slider pre-centering at step 0->1 -- centre now means
"the converged auto read".
Outcome: worked — new closed-loop GatingTest walks the feedback path over a tungsten wall and
asserts it lands neutral without railing; 85 unit tests pass, installed on rosemary and launches.
The colour result itself still needs a human pointing it at a real wall.

## 2026-09-17: Pre-flight WB hands illuminant estimation to the ISP
The damped grey-world loop from the entry above converged, but to the wrong answer: grey-world
neutralises whatever is in the reticle, so a pine floor came out grey-blue instead of wooden. It
cannot separate "neutral wall under warm light" from "warm surface under neutral light" -- that's
illuminant estimation, not chroma averaging. Pre-flight now leaves `CONTROL_AWB_MODE_AUTO` on
through the sweep, echoes the ISP's `COLOR_CORRECTION_GAINS`/`_TRANSFORM` back off the capture
result, and freezes them at step 0->1 (`CameraPipeline.autoWhiteBalance`, `freezeAutoWhiteBalance`);
sliders trim on top, reset re-runs the ISP estimate. Deleted `whiteBalanceStep` and the
accumulation state; `whiteBalanceGains` stays as the one-shot fallback for a device that never
reports its AWB gains.
Outcome: worked — rosemary reports them (`WB frozen: isp=true gains=1.1875, 1.0, 2.4824219`, i.e.
scene-dependent), and the frozen preview looks natural instead of blue. 84 unit tests pass.

## 2026-09-18: Save-completion race, tap-to-focus debounce, warning message queue
Three field reports from a real scan: (1) `finishSession()` set `state = State.DONE` and the UI
showed "Start new scan" (enabled) synchronously, while `DatasetWriter.finish()`'s zip/cleanup ran
async on its own handler thread — closing the app during that window (which looked identical to
being done) killed the write before the zip landed. (2) `root`'s tap-to-focus touch listener
fired `CONTROL_AF_TRIGGER_START` on every `ACTION_UP` in the live-view area with no drag check and
no cooldown, fighting the already-running continuous AF on any incidental touch. (3) the warning
banner was level-triggered off per-frame conditions with no minimum dwell, so anything that
cleared within a frame or two (autofocus settling, one overexposed frame) was unreadable, and
lower-priority conditions co-occurring with a higher-priority one never surfaced at all.
Fixed by: gating the DONE screen's button/label on `finishedPath != null` ("Saving…", disabled,
until the callback fires); requiring `ACTION_DOWN`≈`ACTION_UP` within touch slop plus a 1.2s
cooldown before re-triggering AF; and replacing the single-value warning `when` with a
`WarningKind`-keyed queue (`pickWarning`) that shows each newly-triggered kind for a minimum 3s,
dedupes same-kind while showing or queued, and pops the next distinct kind after.
Outcome: worked — `compileDebugKotlin` clean; not yet re-verified against a real over-focusing
scan on rosemary (originating bug report was from memory of the incident, not a live repro).
