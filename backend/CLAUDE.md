# CLAUDE.md — backend/

Backend-specific guidance. Read the root `CLAUDE.md` first — this only adds what's
specific to working inside `backend/`.

## Before touching `reconstruction/` or `ingestion/`

Three history files exist here, each with a different scope — read the relevant one(s)
before starting work, don't just skim `README.md`:
- `backend/history.md` — exhaustive chronological engineering/failure log (hardware
  pathology, precision bugs, orientation bug, keyframe selection evolution). This is the
  most detailed record of *why* things are the way they are. Dense/math-heavy; grep for
  the subsystem you're touching rather than reading linearly.
- `backend/project_history.md` — milestone-level summary of the same work.
- `backend/ingestion/project_history.md`, `backend/reconstruction/project_history.md` —
  per-folder logs per the root CLAUDE.md convention. Append an entry here after a macro
  task in that folder.

## Environment

- Python env: `backend/.venv` (already provisioned — use `.venv/bin/python`, not system
  Python or a new venv).
- GPU: AMD Radeon RX 9060 XT, 16GB VRAM, RDNA4 (`gfx1200`), ROCm 7.1, `torch==2.13.0+rocm7.1`.
  Confirmed live and detected (`torch.cuda.is_available() == True`, device name resolves
  correctly) as of 2026-09-10.
- Never manually `.half()`/`.bfloat16()` cast DA3 submodules — this crashes on RDNA4
  (`SIGBUS`/`c10::AcceleratorError`, see `history.md` §4). Use `torch.autocast` /
  `torch.cuda.amp` only; this is already how `DepthPriorEstimator` does it.
- **Never use `torch.matmul`/`torch.mm`/`einsum` to transform a large point set.** On gfx1200 /
  ROCm 7.1 a float32 `(N, 3) @ (3, 3)` silently leaves every output row from index 524,288
  (2**19) onward as zeros — the BLAS kernel's grid only covers the first 2**19 rows. No error,
  no NaN. Use the elementwise column form (`_rotate()` in
  `reconstruction/training/rasterizer_interface.py`); it is exact at any N and faster for k=3.
  This is why primitive counts above ~524k used to render empty frames.
- `MIOPEN_USER_DB_PATH` must point somewhere writable outside the sandboxed home config
  path (currently `/tmp/miopen`) or MIOpen kernel compilation silently aborts.
- Long GPU inference loops need cooperative yields (`time.sleep(0.18)` + periodic
  `torch.cuda.empty_cache()`) or the AMDGPU driver watchdog resets the card and kills the
  process — this pattern is already threaded through `depth_priors.py`, preserve it if you
  touch the inference loops.

## Depth Anything 3 (`third_party/depth_anything_3/`)

This is a vendored upstream clone (has its own `.git/`, own `pyproject.toml`), not code
written for this project — read `third_party/depth_anything_3/src/depth_anything_3/api.py`
directly for ground truth on what `inference()` actually does rather than trusting
`history.md`'s summary. Key fact worth knowing before changing anything in
`reconstruction/depth_priors.py`: DA3 estimates its **own** camera poses internally per
inference call and only uses the extrinsics you pass in to do a post-hoc Umeyama
similarity-transform scale correction (`align_to_input_ext_scale=True` replaces
`prediction.extrinsics` with your input poses and rescales `prediction.depth` by the fitted
scalar — see `_align_to_input_extrinsics_intrinsics` in `api.py`). It does not fuse or
average poses across separate inference calls.

Practical implication: if you run DA3 chunk-by-chunk (sliding window), each chunk already
gets depth rescaled to be consistent with the *same global* VIO poses `initialization.py`
later unprojects with — chunks should already share one coordinate frame without further
correction, provided each chunk's independent scale fit is accurate.

## Resolved 2026-09-10: the "bowled point cloud" (see `project_history.md`)

Multi-chunk reconstruction used to produce a warped, 2-3x oversized cloud. It was **not** a
geometry bug — two unrelated causes, both fixed:

1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (which upstream DA3's `cli.py` and
   `gradio_app.py` set unconditionally) silently corrupts DA3 inference on ROCm 7.1 /
   gfx1200 — wrong-but-plausible depth, then all-NaN, then
   `HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION`. **Never set it in this repo, and don't import
   those DA3 entry points.** `depth_priors.py` carries a comment saying so.
2. DA3 was handed ARCore camera-to-world poses; it needs OpenCV **world-to-camera**.
   `arcore_c2w_to_da3_w2c()` in `depth_priors.py` now does the conversion at the single
   point every caller funnels through. Adjacent-view agreement went 5-30 cm -> 0.7-1.5 cm.

Two things worth carrying forward:
- The unprojection in `reconstruction/initialization.py` is **correct** as written (OpenGL
  camera-local rays against the ARCore c2w). Earlier notes calling this an OpenCV/OpenGL
  mismatch were wrong; don't "fix" it.
- A wrong pinhole intrinsic or a uniform resize **cannot** bend a plane — those are linear
  maps of the camera frame, and linear maps send planes to planes. If reconstructed geometry
  is curved, suspect the depth *values*, not the camera model.

Still open, but minor: floor flatness in the gravity-aligned frame degrades 0.87 cm (3 chunks)
-> 2.81 cm (7 chunks). The inter-chunk median/IQR Sim(3) step in
`DepthPriorEstimator.estimate_depth_sliding_window` is the prime suspect — it is a purely
value-based rescaling with no geometric grounding, and is plausibly redundant with DA3's own
per-chunk Umeyama scale correction. Diff reconstructed geometry with it disabled before
changing it.
