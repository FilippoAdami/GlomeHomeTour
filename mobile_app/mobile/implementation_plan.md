# Mobile App Implementation Plan — Account, Gallery, Multi-Room, Settings, Home

Status: proposed, not started. Written 2026-09-20 against the current `mobile_3.0`
(AR-Scan) codebase. Supersedes nothing in `SPEC.md`/`project_history.md` — this plan
is additive product scaffolding around the existing capture pipeline, which is left
untouched except where a phase says otherwise.

## 0. Current state (why this plan is shaped the way it is)

- **Single-Activity, capture-first app.** `MainActivity.kt` (1650 lines) *is* the app:
  launch → straight into ARCore scanning. No home screen, no login, nothing before it.
- **Gallery is real but minimal.** `ScanGalleryActivity` (333 lines) lists/deletes
  session `.zip` files out of `Documents/GlomeHomeTour/` via MediaStore, with legacy
  fallback handling for pre-zip sessions. It has no thumbnails, no coverage %, no
  grouping, no upload state — it is a flat file browser.
- **"One scan" = one continuous walk, not a room, not a property.** Per `SPEC.md`
  §2.1, a session is `IDLE → SCANNING → DONE`, one uninterrupted walk. There is
  currently no concept above that: no room label, no property, no way to associate
  N scans with one listing.
- **No backend wiring from the app at all.** Zips are exported to local storage only.
  Nothing in `mobile/` calls `backend/api/`. Whether `backend/api/` even has an auth
  story for job submission is unconfirmed — this plan flags that as a blocking
  question for Phase 4, not something to assume.
- **No settings surface.** Tuned constants (parallax angle, coverage threshold,
  photometric gate bounds, keyframe spacing) are hardcoded across `CameraPipeline`,
  `VoxelGrid`, `Photometric`, re-tuned per device by editing code (see
  `project_history.md`'s `rosemary`-specific tuning entries).
- **No account/auth anywhere in the repo.**

Net: this is not "add a screen to an app that has screens" — it's introducing the
app's first navigation shell, its first persistent data model, and its first network
layer. The plan sequences work so each phase ships something usable on its own,
rather than one big-bang rewrite.

## 1. Data model (introduced in Phase 1, referenced throughout)

New local Room database (`androidx.room`, already the standard Android persistence
choice — no new dependency category beyond what a settings/gallery/property app
needs). Three tables:

```
Property
  id            UUID (PK)
  address       String
  agentNote     String?
  createdAtMs   Long
  uploadStatus  enum { LOCAL_ONLY, UPLOADING, UPLOADED, UPLOAD_FAILED }

RoomScan
  id                UUID (PK)
  propertyId        UUID (FK -> Property.id)
  label             String            -- "Kitchen", "Bedroom 2", operator-entered
  sessionName       String            -- matches DatasetWriter's session/zip name
  coveragePercent   Float?            -- read back from coverage_summary.json after finish()
  capturedAtMs      Long
  status            enum { CAPTURING, DONE, UPLOADED, UPLOAD_FAILED }

Agent (Phase 4 only)
  id            String   -- server-issued
  email         String
  displayName   String
  tokenEncrypted String  -- via EncryptedSharedPreferences, not stored in Room
```

`RoomScan.sessionName` continues to be exactly what `DatasetWriter`/`ScanGalleryActivity`
already produce and parse — this table is a local index on top of existing files, not
a replacement for the file-based export. If a scan's zip is deleted outside the app
(file manager, ADB), the DB row becomes orphaned; Phase 2's gallery reconciles this the
same way `ScanGalleryActivity` already reconciles MediaStore vs. legacy fallback state
today (read-through, don't trust the DB as sole source of truth).

## 2. Phase 0 — Navigation shell

**Goal:** something other than the capture screen launches first. Everything else
depends on this existing before it can be built.

- Add `HomeActivity` as the new launcher (`MAIN`/`LAUNCHER` intent-filter moves to
  it in `AndroidManifest.xml`).
- Rename `MainActivity` → `CaptureActivity` (it *is* the capture screen; the name
  was only accurate when it was the only screen). Mechanical rename, no logic
  change — update the manifest entry and any internal references.
- `ScanGalleryActivity` loses its own entry point if it had one; it's reached from
  `HomeActivity`/`PropertyDetailActivity` only.
- Stay in the existing **Activity-per-screen** pattern already used by
  `ScanGalleryActivity`. Do not introduce Navigation Component or Compose — the
  current pattern works, is consistent with the codebase, and a framework swap
  isn't part of what's being asked for here.

**Ships:** nothing user-visible yet; this is scaffolding. Merge it together with
Phase 1 in practice so there's a usable home screen at the end of one PR.

**Files touched:** `AndroidManifest.xml`, `MainActivity.kt` → `CaptureActivity.kt`
(rename), new `HomeActivity.kt` + `activity_home.xml`.

## 3. Phase 1 — Property model + multi-room acquisition

**Goal:** the actual product gap. A property groups multiple room scans; capture
becomes "add a room to this property" instead of a standalone act.

- Add the `Property`/`RoomScan` Room tables from §1, plus a `PropertyDao`/
  `RoomScanDao` and a thin `PropertyRepository`.
- `HomeActivity`: list of properties (address + room count + last-updated), "+ New
  Property" action (address text field, minimal — no geocoding/validation beyond
  non-empty).
- `PropertyDetailActivity`: list of that property's rooms (label, coverage %,
  status), "+ Add Room" button, "Export/Upload Property" action (latter wired in
  Phase 4).
- `CaptureActivity` gains two required intent extras: `propertyId`, `roomLabel`
  (prompted via a small dialog before ARCore session start if not already
  supplied). On `finishSession()`, after `DatasetWriter.finish()` completes (respect
  the existing async completion gating from the 2026-09-18 save-completion-race fix
  — don't regress that), insert/update the corresponding `RoomScan` row with the
  produced `sessionName` and, once `coverage_summary.json` is readable, its
  `coveragePercent`.
- **Per-room capture semantics are unchanged.** Per `SPEC.md`, one room scan is
  still one continuous walk with its own loop-closure/coverage rules. "Multi-room"
  is composition at the property level, not a change to how a single walk works.
- Schema question to resolve before writing code: does `coverage_summary.json` (or
  `transforms.json`) need a `room_label`/`property_id` field so the backend pipeline
  can tell rooms apart when it ingests them? Check
  `shared/schemas/coverage_summary.schema.json` and `shared/schemas/transforms.schema.json`.
  If yes, that's a schema version bump coordinated with `backend/00_ingestion/` — per
  this repo's own escalation rule, cross-subsystem schema redesign is an Opus-level
  call, flag it rather than deciding unilaterally in the mobile layer.

**Ships:** a home screen, property creation, multiple labeled room scans per
property, all fully local. Usable standalone even before Phase 4 exists.

**Files touched:** new `data/Property.kt`, `data/RoomScan.kt`, `data/AppDatabase.kt`,
`data/PropertyRepository.kt`; new `HomeActivity.kt`, `PropertyDetailActivity.kt` +
layouts; `CaptureActivity.kt` (extras handling, finish-time DB write).

## 4. Phase 2 — Gallery upgrade

**Goal:** turn the flat file browser into a property-grouped view. Extend
`ScanGalleryActivity`, don't replace it — its MediaStore/legacy-fallback listing and
scoped-storage-safe delete logic (see the zip-and-cleanup fix in
`project_history.md`) is already correct and non-trivial; re-derive grouping from
the DB on top of it rather than rewriting the file-discovery half.

- Reframe as property-grouped: each property row expands to its `RoomScan`s
  (reusing today's per-scan row: date, size, delete-with-progress).
- Thumbnail: decode the first `images/frame_00000.jpg` out of each session's zip on
  a background thread (`ThumbnailFetcher`, simple `ZipFile` + `BitmapFactory`, no new
  dependency) and cache to a small in-memory LRU — skip disk caching, these are a
  handful of images per session and the zip read is already cheap.
- Surface `RoomScan.coveragePercent` as a badge on each room row — the data already
  exists in `coverage_summary.json`/the DB, just wasn't being displayed.
- Upload/export status badge — wire the visual now, leave it as `LOCAL_ONLY`/static
  until Phase 4 lands the actual upload path.
- Orphan reconciliation: if a DB `RoomScan` row has no matching zip (deleted
  out-of-band) or a zip exists with no DB row (pre-Phase-1 capture, or DB reset),
  show it under an "Unassigned" pseudo-property rather than silently dropping it —
  matches the existing app's bias toward never hiding data the operator might need
  to manually clean up.

**Ships:** the gallery becomes the actual browsing surface for the property model,
useful even before any backend upload exists.

**Files touched:** `ScanGalleryActivity.kt` (restructure to grouped list), new
`ThumbnailFetcher.kt`.

## 5. Phase 3 — Settings

**Goal:** expose the handful of constants that have actually needed per-device
retuning in the field, per `project_history.md`'s tuning history — not a general
preferences screen for its own sake.

- `SettingsActivity` using `PreferenceFragmentCompat` (androidx preference library —
  check if already a transitive dependency via `androidx.appcompat` before adding
  it explicitly) backed by plain `SharedPreferences`.
- Expose only the values with a documented history of needing device-specific
  retuning:
  - Parallax verification angle (currently 25° in `FeatureParallaxTracker`)
  - Coverage-complete threshold (currently 85%)
  - Photometric gate bounds (Y_mean 40/250)
  - Keyframe spacing (8cm / 6°)
- Everything else stays a hardcoded constant. Do not add settings for values that
  have never needed changing — that's a knob nobody will turn, and it enlarges the
  support surface (operators fat-fingering a value they don't understand) for no
  benefit.
- Read path: the four consuming classes (`CameraPipeline`, `VoxelGrid`,
  `Photometric`, `FeatureParallaxTracker` or wherever keyframe spacing lives) read
  from `SharedPreferences` with the current hardcoded value as the default, so a
  fresh install behaves identically to today until an operator changes something.

**Ships:** standalone, no dependency on Phases 1/2/4. Can be built any time and
slotted in independently.

**Files touched:** new `SettingsActivity.kt` + `res/xml/preferences.xml`; small
reads added in `CameraPipeline.kt`, `VoxelGrid.kt`, `Photometric.kt`,
`FeatureParallaxTracker.kt`.

## 6. Phase 4 — Account + backend sync (stop here, wait for this phase)

**Goal:** get a property's room scans off the phone and into `backend/api/`.

This is the least-specified phase because **no auth or upload code exists on
either side of this integration today.** Before writing mobile code:

1. Confirm what `backend/api/` actually expects for job submission — auth scheme
   (if any), payload shape, whether it even exists yet as a callable endpoint. If
   it doesn't exist, this phase blocks on backend work, not mobile work — don't
   build the mobile half speculatively against a guessed contract.
2. Assume single-tenant, admin-provisioned agents (email + token issued out of
   band), not self-service registration — real estate agents using a field capture
   tool are very unlikely to be self-signing-up from the phone, and building a full
   account system (password reset flows, org/role management) here would be
   speculative scope no one asked for.

If the backend contract exists or is agreed:

- Login screen: email + token (or password, per whatever the backend expects),
  stored via `EncryptedSharedPreferences` (`androidx.security-crypto`) — never in
  plain `SharedPreferences` or the Room DB.
- `HomeActivity` requires a logged-in `Agent` before showing/creating properties;
  gate behind a simple auth-check on launch.
- `PropertyDetailActivity`'s "Upload Property" action zips the property's rooms
  (or uploads each `RoomScan`'s existing session zip individually, if the backend
  wants per-room jobs) and POSTs to the job-submission endpoint, reusing the
  existing determinate-progress-bar pattern already built for
  `ScanGalleryActivity.deleteScan` rather than inventing a new progress UI.
- Update `Property.uploadStatus`/`RoomScan.status` on response; surface failures
  inline (retry action) rather than silently leaving state stale.

**Ships:** end-to-end capture → property → upload. This is the phase that actually
closes the loop with `backend/`.

**Files touched:** new `AuthManager.kt`, `LoginActivity.kt`, `UploadService.kt` (or
`WorkManager` job if uploads need to survive app backgrounding — likely, given zip
sizes); `PropertyDetailActivity.kt` (upload action); `data/Agent.kt`.

## 7. Suggested build order

1. **Phase 0 + Phase 1 together** (one PR) — scaffolding alone ships nothing, so
   merge it with the property model to land a usable home screen in one step.
2. **Phase 2** — cheap once Phase 1's DB exists, high visible payoff (turns the
   flat file list into the real browsing UI).
3. **Phase 3** — independent, no ordering dependency on anything else; good
   filler/parallel work at any point.
4. **Phase 4** — gate explicitly on confirming the `backend/api/` contract first
   (§6 step 1). Don't start mobile upload code against an assumed API shape.

## 8. Explicitly out of scope (don't build unless asked)

- Navigation Component / Jetpack Compose migration — current Activity-per-screen
  pattern is adequate and consistent with the existing codebase.
- Multi-organization / role-based account system — single-tenant, admin-provisioned
  agents is sufficient for the described use case.
- Geocoding/address validation on property creation — a free-text address field is
  enough; nothing downstream currently consumes a structured address.
- Offline queue / conflict resolution for uploads beyond "retry the whole job" —
  add only if real usage shows it's needed.
- Disk-backed thumbnail cache — in-memory LRU is enough at the expected scan-count
  scale (tens of rooms per property, not thousands).

## 9. Per-phase acceptance checklist

- **Phase 0/1:** fresh install opens to `HomeActivity`; can create a property, add
  ≥2 labeled room scans to it via `CaptureActivity`, and see both listed under the
  property in `PropertyDetailActivity` after each `finishSession()`.
- **Phase 2:** gallery groups by property, shows a thumbnail and coverage % per
  room, delete-with-progress still works exactly as before restructuring, orphaned
  zips/DB rows are visible somewhere rather than silently dropped.
- **Phase 3:** changing a setting takes effect on the next capture session without
  a rebuild; uninstalling/reinstalling resets to today's hardcoded defaults.
- **Phase 4:** a captured property can be uploaded from `PropertyDetailActivity`,
  failure is visible and retryable, and a fresh login is required after clearing
  app data.

Per this repo's `CLAUDE.md` convention, log outcomes for each phase's work in this
folder's `project_history.md` as they land — this plan is the roadmap, not the log.
