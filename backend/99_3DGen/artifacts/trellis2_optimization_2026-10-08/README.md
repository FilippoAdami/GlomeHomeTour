# TRELLIS.2 RX 9060 XT optimization evidence

Hardware: AMD Radeon RX 9060 XT, 16 GiB; PyTorch 2.10.0+rocm7.0; seed 34; 1024 cascade; 49,152 token cap; 150,000 face target; 1024 texture. Times include GLB export unless specified. Batch per-image times exclude the one shared model load and unload.

From `Pixal3D`, the optimized runner keeps **12 sampling steps as its default**:

```bash
../.venv/bin/python inference_trellis2.py --image ../images/desk_scene.jpg --output ./desk.glb
```

The measured 10-image latency run used 8 steps. Run from `Pixal3D`:

```bash
PYTHONUNBUFFERED=1 ../.venv/bin/python inference_trellis2.py \
  --image ../images/*.jpg \
  --output ../artifacts/trellis2_optimization_2026-10-08/batch_8 \
  --steps 8 --seed 34 --decimation-target 150000
```

The 10-image 8-step batch completed with one checkpoint load. All GLBs load as scenes with one mesh.

| Image | Inference (s) | GLB export (s) | Per-image total (s) |
|---|---:|---:|---:|
| bench | 66.59 | 9.56 | 76.15 |
| desk_scene | 79.71 | 10.24 | 89.95 |
| drawer | 76.95 | 20.63 | 97.59 |
| library | 28.31 | 4.58 | 32.89 |
| mirror | 36.34 | 11.63 | 47.96 |
| person | 66.00 | 10.18 | 76.18 |
| punchingbag | 29.11 | 6.38 | 35.50 |
| puzzle | 44.71 | 19.57 | 64.29 |
| radiator | 48.14 | 10.64 | 58.78 |
| worldmap | 30.50 | 15.62 | 46.12 |

Mean per image: **62.54 s**; maximum: **97.59 s**; peak allocated VRAM: **4.81 GiB**. The batch process took 641.16 s including 15.40 s to load and 0.27 s to unload. The final loader improvement below was applied after this batch; it only changes model load time. See [batch_8/trellis2_batch.json](batch_8/trellis2_batch.json), per-image JSON and GLBs, and [batch_8.log](batch_8.log).

Standalone results including checkpoint load, inference, GLB export, and explicit unload:

| Image | Steps | Earlier total (s) | Optimized total (s) | Measured load (s) |
|---|---:|---:|---:|---:|
| bench | 12 | 215.01 | 98.70 | 15.27 |
| desk_scene | 12 | 194.53 | 142.76 | 4.66 |
| drawer | 8 | 187.00 at 12 steps | 120.78 | 15.45 |

The bench and drawer runs preceded the final load improvement; their inference and export measurements remain valid. The final loader's sparse-structure checkpoint was also checked against ordinary loading: its sampled weights and generated RoPE buffer matched exactly, while that model's load fell from 10.92 s to 1.21 s. The end-to-end desk run uses the final code. Its geometry matches the earlier 12-step GLB closely: volume 0.004490 versus 0.004491 and maximum bounds difference 0.000186 in normalized coordinates. See [desk_single_12.json](desk_single_12.json) and [desk_single_12.glb](desk_single_12.glb).

The 8-step setting trades geometry for speed. Against the earlier 12-step GLBs, mesh volume changed by up to about 31% on these inputs. **12 steps remains the default**; use `--steps 8` when latency takes priority. The original 10-image 12-step runs averaged 97.78 s for inference plus export and had a 146.62 s maximum; these use a different step count and separate model loads, so they are context, not a like-for-like speedup estimate.

Profiling a 12k-token shape pass found fused attention at 49% and matrix multiplication at 27% of GPU time. The optimizations here skip an unused checkpoint, construct loaded models on the meta device with dtype-preserving assignment, avoid duplicate hole filling before the exporter, and reuse one pipeline across batch images. No driver or system package was changed.
