# Pixal3D vs TRELLIS.2: 10-image run

Seed: `34`. GPU: AMD Radeon RX 9060 XT (16 GiB). Both: 1024 cascade, 12 sampling steps, 49,152 max tokens, 150,000-face target, 1024 texture. Pixal3D ran with low-VRAM enabled.

GLB validation: all 20 load as GLB scenes containing one mesh geometry. Failed runs: 0.

## Per-image times

| Image | Pixal3D total (s) | TRELLIS.2 total (s) | Pixal3D GLB (MB) | TRELLIS.2 GLB (MB) |
|---|---:|---:|---:|---:|
| bench.jpg | 255.8 | 221.4 | 5.42 | 5.11 |
| desk_scene.jpg | 203.7 | 200.8 | 5.45 | 5.19 |
| drawer.jpg | 180.4 | 192.3 | 5.02 | 4.75 |
| library.jpg | 209.6 | 133.8 | 5.81 | 5.41 |
| mirror.jpg | 179.7 | 133.5 | 5.61 | 5.45 |
| person.jpg | 182.2 | 187.4 | 4.80 | 4.66 |
| punchingbag.jpg | 134.6 | 119.9 | 5.03 | 5.20 |
| puzzle.jpg | 159.0 | 169.9 | 7.28 | 5.15 |
| radiator.jpg | 165.4 | 164.5 | 4.87 | 5.95 |
| worldmap.jpg | 152.6 | 130.6 | 6.63 | 5.84 |

## Aggregate timings

| Model | Mean end-to-end (s) | Mean generation (s) | Mean GLB export (s) | Mean model load (s) | Peak VRAM (GiB) |
|---|---:|---:|---:|---:|---:|
| pixal3d | 182.3 | 97.1 | 10.5 | 58.0 | 13.03 |
| trellis2 | 165.4 | 84.5 | 13.3 | 62.1 | 4.76 |

## Stage timings

`atomic_step_timings.csv` contains every timed stage for every image, preserving both TRELLIS.2 `get_cond` calls separately. It also includes aggregate pipeline/inference, export, end-to-end, and runner wall times; aggregate rows overlap their constituent stage times. Full per-run JSON and console logs are alongside each GLB.

## Files

GLBs use `pixal3d_<image-stem>.glb` and `trellis2_<image-stem>.glb`. See [`manifest.json`](manifest.json) for run settings and all per-run metrics.
