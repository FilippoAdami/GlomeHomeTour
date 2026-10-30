# Multi-level restoration benchmark

Standalone, additive benchmark using official [OSEDiff](https://github.com/cswry/OSEDiff)
and [SeeSR](https://github.com/cswry/SeeSR) sources. Existing mesh pipeline code is untouched.

Current matrix: OSEDiff, SeeSR standard (50 steps), SeeSR Turbo (2 and 4 steps),
each on the same ten images at blur radii **3, 5, 7, 13**. Blur 1 was removed by request.
SeeSR Turbo's source, weights and isolated environments live with this code in
`backend/04_3DGS_to_mesh/`. Inputs, comparison-only models, outputs and reports live
in `backend/current_scene/04_3DGS_to_mesh/`.

From the repository root:

```bash
# Uses backend ROCm torch through a .pth file; never changes the backend environment.
HF_HUB_DISABLE_XET=1 backend/.venv/bin/python backend/04_3DGS_to_mesh/restoration_benchmark/setup.py
backend/04_3DGS_to_mesh/env/bin/python backend/04_3DGS_to_mesh/restoration_benchmark/test_benchmark.py
backend/04_3DGS_to_mesh/env/bin/python backend/04_3DGS_to_mesh/restoration_benchmark/run.py
# Pass the exact batch JSON printed/saved by run.py:
backend/04_3DGS_to_mesh/env/bin/python backend/04_3DGS_to_mesh/restoration_benchmark/report.py backend/current_scene/04_3DGS_to_mesh/images_restored/batch_TIMESTAMP.json
```

`setup.py --blur-only` verifies the exact existing Pillow GaussianBlur/JPEG75 recipe
and creates blur 13 from unblurred originals. It refuses differing existing blur-13
files. `--download-only` resumes weight downloads. Setup requires network access;
all installations are project-local. The SD2.1 mirror is the one linked by OSEDiff.
Published FP16 base weights are hardlinked to the canonical filenames required by
upstream loaders. If `hf_transfer` is available, run `setup.py --download-only` with
the isolated environment and `HF_HUB_ENABLE_HF_TRANSFER=1` for parallel downloads.
The SeeSR Turbo checkpoint, SD-Turbo base, shared RAM/DAPE weights and official
SeeSR source are stored under `backend/04_3DGS_to_mesh/`. SD2/SD2.1 and OSEDiff
comparison assets remain under `backend/current_scene/04_3DGS_to_mesh/`.
The largest RAM/SeeSR assets use validated, resumable 8 MiB ranges and fresh CDN
URLs per request. Completed ranges survive retries; a full SHA256 check precedes
publication. Resume one asset directly with `download_asset.py ram`,
`download_asset.py seesr_control`, or `download_asset.py seesr_unet` using the
backend or isolated Python interpreter. These files total about 17.4 GB together
with the diffusion bases and evaluation weights; RAM's released checkpoint alone
is 5.63 GB.
Two upstream repositories are kept separate because both define top-level `models`
and `ram` packages. Each configuration runs in its own subprocess, loads once,
warms up each distinct input resolution and processes all four levels before unloading. The SeeSR adapter
aliases the obsolete `diffusers.pipeline_utils` import to its current location;
upstream source files are unmodified. Xformers is disabled on AMD.
Workers use eight CPU threads. Only SeeSR Turbo four-step enables ROCm's
experimental AOTriton attention; other configurations keep it disabled.
PyTorch allocations are capped at 70% of device VRAM, leaving room for the
desktop. The SeeSR RAM tagger runs in FP16 on the GPU only when the free-memory
and allocator-budget checks leave room for it; otherwise it stays in FP32 on
the CPU. The PyTorch allocator cache is reused across each batch and released
when the model unloads. An allocation failure stops the configuration
immediately rather than attempting the remaining images.
SeeSR standard and Turbo two-step retain maximum attention slicing.
Turbo four-step uses unsliced PyTorch SDPA with AOTriton, plus 96-pixel latent
tiles; this avoids the quadratic math fallback while keeping the measured tile
working set. The 512-pixel encoder and 64-pixel decoder VAE tiles remain to
bound VRAM use. Tile sizes, attention mode and the AOTriton flag are recorded
per run. An earlier experimental trial coincided with a GPU reset of unknown
cause; the current setting completed the full 40-image run without a reset.
SeeSR workers use `backend/04_3DGS_to_mesh/env_seesr/`, which layers their
documented Diffusers 0.21.0 over the shared isolated dependencies. OSEDiff uses
Diffusers 0.25.0 in `backend/04_3DGS_to_mesh/env/`.
Both inherit the same backend ROCm torch; dependency versions are saved per run.
OSEDiff's hardcoded developer-machine BERT tokenizer path is replaced in memory
with the same public `bert-base-uncased` tokenizer already cached on this machine.
SeeSR Standard uses the official SD2-base configuration; OSEDiff uses SD2.1-base.
Their frozen text encoder/VAE weights have identical published SHA256 hashes and
are shared after verification. Turbo omits the unused image processor, as its base
repository does not ship one and the upstream pipeline disables the safety checker.

Selection is ten evenly spaced filenames in lexicographic order from blur 3,
validated against all four levels and optional matching originals. With these
inputs that selects all ten images. SHA256 hashes and dimensions are recorded.
Use `run.py --prepare-only` to validate without loading a model; `--model NAME`
executes one configuration. Every invocation creates uniquely timestamped run
folders; previous outputs are retained. Report an exact batch, never a mix of runs.
Each batch retains its CSV/Markdown summaries in `comparison/<batch_stem>/`;
the top-level summaries show the most recently reported batch.

Restoration is **1×**, at original dimensions, with FP16 model inference and
the VAE/latent tile settings above. Images are resized to multiples of eight for
inference and returned to original dimensions. This evaluates restoration rather
than 4× super-resolution. All models use the upstream AdaIN color correction.
The callable returned by `load_model` defaults SeeSR Turbo to three denoising
steps and accepts `num_inference_steps=` to override that count; the benchmark
worker explicitly selects two or four steps for its named Turbo configurations.
The same seed (42 + image index) is reused across levels/configurations.
RAM precision is FP16 on GPU for OSEDiff and chosen by the SeeSR memory guard.
Tagging and transfer costs remain in measured inference time.
The device, precision and allocator cap are recorded per run. Report capacity
in GiB (bytes / 2^30): the nominal 16 GB card exposes about 15.92 GiB, not 17 GiB.

Outputs use lossless **PNG contents under original filenames**, even for `.jpg`,
to preserve filenames without JPEG recompression bias. PIL/OpenCV decode by file
signature. Tools that require extension/format agreement need an explicit PNG copy.

Inference time includes preprocessing, RAM tags/embeddings, diffusion, color fix
and transfer to CPU; it excludes reading and saving images. GPU synchronization
bounds every measurement. Processing time adds decode and save. Quality metrics
run only after restoration models have unloaded, and never enter timed regions.
One warmup image per distinct input resolution is separately timed, excluded from
inference statistics and load/unload amortization. This keeps first-use GPU
kernel compilation for new shapes out of the blur-level inference comparison.

For level L, let P(L) be mean processing time and H be measured load + unload:

- inference-only: measured mean and median of ten synchronized restorations;
- 40-image amortization: P(L) + H/40;
- projected ten-image deployment: P(L) + H/10.

Warmup is reported separately; actual cold invocation also incurs warmup,
Python/import startup and benchmark orchestration. Session wall time per image is
reported separately and includes benchmark CSV/trace bookkeeping. Loading includes model transfers
and initialization. Per-image CSVs include peak allocated/reserved PyTorch GPU
memory. The benchmark suppresses SeeSR's internal peak-counter resets during each measured
restoration so the recorded peak covers the complete image, including its tiled VAE.
The 200 ms resource trace samples process RSS/CPU and available sysfs total
GPU VRAM/busy counters; device totals include other processes and may miss brief peaks.

Full-reference PSNR (RGB, range 255), SSIM and AlexNet LPIPS (native resolution)
use only matching **unblurred** originals. Identical images use PSNR=100 as a finite
sentinel. Sharpness is Laplacian variance; edge F1 uses Canny 80/160 with one-pixel
tolerance; mean-color error is RGB mean-vector distance divided by 255. These
proxies cannot establish invented content or geometry correctness. Review the
full/crop model panels and per-model robustness panels before final visual ranking.
Failed images/configurations remain explicit and never become zero timing/quality
measurements. A benchmark is complete only after every feasible configuration
has ten successful outputs at each of the four levels.

## RX 9060 XT Turbo four-step tuning

On the same 40 inputs and seeds, moving SeeSR's RAM tagger to guarded GPU FP16
and retaining the allocator cache for the batch reduced mean inference from
9.59 s to 8.67 s per image (9.6%); median fell from 6.80 s to 5.95 s. Peak
allocated/reserved memory was 7.88/10.14 GiB, and sampled whole-device use
peaked at 13.07 GiB. All 40 lossless outputs passed dimension and decode checks.
Against the previous CPU-tagger outputs, mean PSNR/SSIM was 62.37 dB/0.9985.
The run and paired per-image results are under
`backend/current_scene/04_3DGS_to_mesh/images_restored/`.

The current Turbo four-step result enables AOTriton and unsliced SDPA. Against
the prior cache/tagger-tuned batch, mean inference fell from 8.67 s to 5.45 s
per image (37.1%); median fell from 5.93 s to 4.07 s (31.3%). The complete
40-image session fell from 435 s to 283 s. Peak allocation stayed at 7.88 GiB;
peak reservation was 9.95 GiB, and sampled whole-device use peaked at 12.29 GiB.
All 40 outputs decoded at the expected dimensions. Against the prior tuned
outputs, mean PSNR/SSIM was 63.04 dB/0.99967 (minimum SSIM 0.99937). See
`seesr_turbo_4step_20261018_205850_729093/benchmark.json` and
`output_validation.json` for measurements. Turbo two-step and the other models
keep their previous attention settings.
