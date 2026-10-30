# Capture ingestion

This phase accepts a capture ZIP or extracted folder, validates and safely stages its contents, then handles each room independently. App-certified ARCore rooms are hash-checked and passed through, while COLMAP inputs are validated and preserved; uncertified ARCore rooms have their image orientation checked and corrected, poses synchronized, scene dimensions and image budget estimated, and quality filtering plus overlap-aware keyframe selection applied only when the image count exceeds that budget. Retained images and their matching metadata are written together, and each room receives an `ingestion_results.json` with paths relative to that room; unsafe archives and unsupported formats fail before publication, and this phase never starts later pipeline stages.

`00_ingestion` is a standalone phase. It accepts a capture ZIP or extracted folder and writes the refined package to `backend/current_scene/`. A single capture is written directly there. A multi-room session writes one `scan_N/` folder per room, each with its own `ingestion_results.json`; the session certificate remains at the scene root. Paths in each result are relative to the folder containing that result. This phase never invokes another pipeline phase.

## Run the step

From the backend directory, pass the absolute path to the capture ZIP or extracted folder:

```bash
cd /absolute/path/to/GlomeHomeTour/backend
.venv/bin/python 00_ingestion/step_ingest.py /absolute/path/to/capture.zip
```

The step writes the refined package to `backend/current_scene` by default. For a multi-room session, the output looks like this:

```text
current_scene/
├── certificate.json
├── scan_0/
│   ├── images/
│   ├── transforms.json
│   ├── trajectory.csv
│   ├── coverage_summary.json
│   └── ingestion_results.json
└── scan_1/
    └── ...
```

Step logs are temporary. Errors are printed as JSON and the process exits nonzero.

## Execution flow

| Order | File and functions | What happens |
| --- | --- | --- |
| 1 | [`step_ingest.py`](step_ingest.py): `main()` | Accepts the capture path, creates a temporary staging area, and starts step logging and timing. |
| 2 | [`ingestion.py`](ingestion.py): `ingest()` → `_ingest()` → `_prepare_input()` | Starts ingestion and prepares a private copy of the input. Folders are copied under the same file and size rules used for ZIP contents; ZIP input is handed to `_extract_zip()`. |
| 3 | `ingestion.py`: `_extract_zip()` → `_validate_zip()` → `_preflight_zip_kind()` | For ZIP input, checks paths, entry types, file signatures, sizes, compression ratios, and supported package type before streaming entries into temporary storage. Unsafe or corrupt ZIPs are rejected and deleted; the error reports whether deletion succeeded. |
| 4 | `ingestion.py`: `_find_colmap_models()` → `_find_scan_directories()` → `_find_arcore_roots()` | Detects a complete COLMAP model, a single ARCore package, or sibling `scan_N/` rooms. Each scan is processed independently. Unsupported pose formats return an in-development error. |
| 5a | `ingestion.py`: `_ingest_colmap()` → `_parse_colmap_text()` / `_parse_colmap_binary()` → `_resolve_colmap_images()` | For COLMAP, validates the model and image references, copies the model and its images unchanged, and records `poses_format: colmap`. Processing, image selection, and orientation changes are skipped. Dimensions are `null` because COLMAP scale is not guaranteed to be metric. |
| 5b | `ingestion.py`: `_ingest_arcore()` → `_validate_transforms()` → `_resolve_images()` → `_validate_images()` | For ARCore, validates the manifest and image files, then resolves the image folder from its contents and the paths in the manifest. |
| 6a | `ingestion.py`: `_find_certificate()` → `_verify_certificate()` → `_validate_arcore_package()` | Looks for the session certificate at the root above the scan folders. It validates schema, required stages, aggregate frame counts, and each room's hierarchical manifest and image hashes, then validates every package. Certified images and manifests pass through without backend rotation, quality filtering, or selection. |
| 6b | `ingestion.py`: `_prepare_arcore_package()` → `_validate_arcore_package()` → `PackageLoader.load()` | Without a certificate, checks EXIF orientation and dimensions. It applies only consistent, supported rotations and updates image pixels, intrinsics, camera poses, and trajectory quaternions together. `PackageLoader` then validates the normalized ARCore package and loads its trajectory and frames. |
| 7 | `pose_aligner.py`: `PoseAligner.synchronize_keyframes()` | For uncertified ARCore input, aligns frame poses to trajectory timestamps. ARCore camera centers then feed `_extent_from_trajectory()` and `_budget()`. Certified ARCore uses its already-aligned transform matrices for the same estimates. |
| 8, conditional | `quality_gate.py`: `QualityGate.evaluate()`; `keyframe_selector.py`: `estimate_scene_depths()` and `DynamicKeyframeSelector.select_keyframes()` | Runs only for uncertified ARCore input when the frame count exceeds the available-frame-capped budget. The gate removes low-quality frames; the selector then keeps an overlapping subset. If filtering leaves fewer than the budget minimum, ingestion returns `NOT_ENOUGH_IMAGES` with the message `not enough images`. |
| 9 | `ingestion.py`: `_copy_arcore_images()` → `_copy_standard_arcore_metadata()` → `_copy_auxiliary_metadata()` | Writes the retained images and matching metadata. Transforms and frame-level metadata are filtered together; trajectory rows remain for pose context, with the `exported` flag updated. Detected depth/LIDAR assets are preserved and marked `in_development`. |
| 10 | `ingestion.py`: `_write_results()`; [`ingestion_results.schema.json`](../../mobile_app/shared/schemas/ingestion_results.schema.json) | Validates and atomically writes one `ingestion_results.json` per scan output folder, with paths relative to that folder. |
| 11 | `step_ingest.py`: `_publish()` | Publishes all scan folders and the session certificate to the workspace only after every scan succeeds. No later phase is started. |

For unsupported but safe inputs, the source is preserved and an in-development error is returned. Unsafe ZIPs are deleted. If an error occurs, staged output is discarded and existing workspace outputs are left in place.

## Supported input routes

| Input | Behavior | Result |
| --- | --- | --- |
| ARCore session with sibling `scan_N/` folders and a root `certificate.json` | Validate the session certificate and all room manifests/images, then pass each processed room through unchanged. | One `current_scene/scan_N/ingestion_results.json` per room; session certificate remains at `current_scene/certificate.json`. |
| Single ARCore package with `certificate.json` | Validate the certificate, manifests, hashes, and images; pass the processed package through unchanged. | `source_format: arcore_app_certified`, `processing_status: skipped_certified`. |
| ARCore package without a certificate | Validate metadata; correct supported orientation metadata; align poses; estimate dimensions and budget; filter/select only when outside the budget. | `source_format: arcore`, `processing_status: processed` or `validated_no_filter_needed`. |
| Complete COLMAP text or binary model | Validate and copy the model and images without filtering or rotating them. | `source_format: colmap`, `processing_status: skipped_colmap`, `poses_format: colmap`. |
| ARTools/ARKit, other unknown pose formats, or LIDAR-only input | Return an in-development error. | No success result is written. |

LIDAR/depth assets accompanying a supported ARCore or COLMAP package are copied to `lidar_data/` and marked `in_development`; they are not processed. The app certificate uses hashes but is not digitally signed, so it detects changed package files but cannot prove who created the certificate.

## Scene estimates and image budget

ARCore scene estimates use camera positions from `transforms.json`. For uncertified captures, the backend first synchronizes frame poses with `trajectory.csv`; certified captures already contain app-processed poses. COLMAP inputs skip these estimates because their scale may not be metric.

### Floors and dimensions

The floor estimate is a trajectory heuristic, not architectural floor detection:

- The vertical reference is `y_ref = percentile(camera_y, 1)` using the lower percentile method.
- It starts at one floor. Candidate floor `k` is counted when at least **10** camera poses are more than `2.8 × (k − 1) + 0.2 m` above `y_ref`. The second-floor threshold is therefore **3.0 m**; the third is **5.8 m**. The search stops before candidate 20, so the estimate is capped at 19 floors.
- For area assignment, each pose is placed in a **2.5 m** vertical band using `clip(int((camera_y − y_ref) / 2.5), 0, floor_count − 1)`.
- Scene height is the **99th percentile minus the 1st percentile** of camera Y positions. Per-floor width and depth use the same percentile span when that floor has at least **5** poses; with fewer poses they use `max − min`. Reported width and depth are the largest per-floor spans.

Floor area approximates the horizontal footprint covered by the camera path. On each floor, the path is dilated by a **0.8 m radius**, rasterized into **0.15 m × 0.15 m** cells, and the occupied cells are counted. The grid bounds add **0.9 m** padding around the path. A floor and the total scene area each have a **9 m² minimum**. To bound computation, a floor grid may contain at most **20 million cells**, and the rasterizer may visit at most **50 million cells** while processing path segments.

The keyframe budget for area `A` in square metres is:

```text
raw_min = ceil(50 + 8 × A)
raw_max = ceil(50 + 11 × A)
minimum = min(raw_min, available_images)
maximum = min(max(raw_max, minimum), available_images)
target = min(available_images, max(2, floor((minimum + maximum) / 2)))
```

Thus, the nominal range is **50 base frames plus 8–11 frames per m²**. The range is capped by the number of available frames. If the available count is below the nominal minimum, the capped budget equals the available count and this alone does not trigger filtering. Filtering starts only when the input count exceeds the capped maximum. When filtering does run, fewer than `minimum` frames after quality gating or selection aborts ingestion with `NOT_ENOUGH_IMAGES`. Certified app captures and COLMAP models bypass backend filtering and keyframe selection.

### Quality gate

The backend quality gate runs only on uncertified ARCore input when filtering is triggered. It scores grayscale images after resizing their longest edge to **960 px**, using a **4 × 4** tile grid and the sharpest quarter of tiles. Thresholds are:

| Metric | Reject condition |
| --- | --- |
| Relative sharpness | `max(absolute / scene_median_absolute, normalized / scene_median_normalized) < 0.35` |
| Absolute sharpness safety floor | `absolute / scene_median_absolute < 0.28` |
| Absolute blur threshold | Disabled (`None`) in this ingestion route. |
| Mean grayscale luminance | `< 12` or `> 250` on the 0–255 scale. |
| Saturated pixels | More than **30%** have luminance `>= 250`. |
| Black pixels | More than **50%** have luminance `<= 8`. |
| Texture | Mean intensity variance of the most-textured quarter of tiles `< 60`. |

Blur rejection is capped at `floor(20% × frame_count)`; the sharpest rejected frames are restored if needed to meet that cap. Exposure and texture rejections are not covered by this blur-only cap. The gate also prevents runs longer than **4** consecutive rejected frames by restoring the best-scoring frame in each run.

### Dynamic keyframe selection

After quality gating, the selector estimates scene depth to evaluate mutual camera-frustum overlap. It looks for up to **400** image corners (`qualityLevel=0.01`, `minDistance=12 px`), tries frame gaps of **4, 8, 16, and 32**, and requires at least **20** corners/tracked points plus **25** valid triangulated depths. Optical flow uses a **21 × 21 px** window and pyramid `maxLevel=3`; a camera baseline must be at least **0.04 m**. Triangulated depths outside **0.2–12 m** are discarded before taking the **35th percentile**. The result is clamped to **0.4–8 m** and smoothed with a rolling median of **5** frames; unavailable measurements fall back to the measured median or **4.2 m** if none are available.

Selection uses **10 × 10** sampled rays, initially at **2 m** when no measured depth is available. Consecutive anchors target mutual overlap between **0.50 and 0.80**; the overlap is the smaller of the two directional frustum-visibility ratios. The selector lowers the maximum overlap threshold in **0.03** steps when too many frames are selected, stopping before it would fall below `minimum_overlap + 0.05` (with these settings, the lowest reachable step is **0.56**). It raises the threshold in **0.03** steps when too few are selected, stopping before it would exceed **0.97** (the highest reachable step from 0.80 is **0.95**). It treats motion as sufficient at translation `>= 0.08 m` **or** rotation `>= 4°`, while maintaining the **0.50** overlap floor where possible. It prioritizes overlap continuity, so the requested budget is a target band rather than an unconditional fixed count.

### Orientation and pose constraints

For uncertified ARCore input, every frame must have matching dimensions and one consistent EXIF orientation. EXIF orientations **1, 3, 6, and 8** are supported; mirrored orientations and mixed-orientation packages are rejected. Nonzero `k1`, `k2`, `p1`, or `p2` lens distortion is unsupported. Supported rotations update image pixels, intrinsics, camera poses, and trajectory quaternions together. Position components beyond **1,000 m** are rejected as implausible input.

## ZIP safety limits

ZIP files are checked before extraction and extracted without `extractall`. Ingestion rejects unsafe paths, duplicate or colliding names, links and special files, executable permissions or signatures, encrypted entries, unknown extensions, nested archives, oversized content, suspicious compression ratios, and images that cannot be decoded or whose extension does not match their encoding.

| Limit | Maximum |
| --- | ---: |
| ZIP entries | 20,000 |
| Compressed archive size | 4 GiB |
| Total expanded size | 4 GiB |
| Individual file | 512 MiB |
| Text or metadata file | 128 MiB |
| Compression ratio | 200:1 |
| Image dimensions | 100 million pixels |

The same limits apply to extracted-directory input where relevant. COLMAP validation additionally caps binary models at **10,000 cameras**, **100,000 registered images**, **50 million observations per image**, and **10 million 3D points**. Scene-area calculation has the grid limits listed above.

## Files used, in execution order

| Order | File | Role |
| --- | --- | --- |
| 1 | [`step_ingest.py`](step_ingest.py) | CLI and orchestration entry point; stages temporary output and publishes successful results. |
| 2 | [`pipeline_step.py`](../Utilities/pipeline_step.py) | Provides step logging, timing, and metrics used by the CLI. |
| 3 | [`ingestion.py`](ingestion.py) | Input safety, ZIP extraction, format detection, route-specific processing, metadata synchronization, and result writing. |
| 4 | [`package_loader.py`](package_loader.py) | Loads and validates normalized ARCore transforms, trajectory, coverage metadata, and frames. |
| 5 | [`pose_aligner.py`](pose_aligner.py) | Aligns uncertified ARCore frame poses against trajectory timestamps. |
| 6 | [`quality_gate.py`](quality_gate.py) | Conditionally scores and rejects low-quality ARCore images. |
| 7 | [`keyframe_selector.py`](keyframe_selector.py) | Conditionally estimates scene depths and selects an overlapping keyframe subset. |
| 8 | [`scene_io.py`](../Utilities/scene_io.py) | Writes schema-valid `transforms.json` for refined ARCore output. |
| 9 | [`ingestion_results.schema.json`](../../mobile_app/shared/schemas/ingestion_results.schema.json) | Defines the successful result manifest written by this phase. |

## TODO: input formats in development

- ARTools/ARKit pose and camera metadata.
- Professional LiDAR depth maps and point-cloud packages, including sensor-specific metadata layouts.
- Other custom pose and trajectory formats that need format-specific orientation, coordinate, and scene-dimension handling.
