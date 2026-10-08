# 06c: class-agnostic instance pipeline

Inputs are in `../current_scene/06c_semantic_segmentation/inputs/`. Keep each
stage's outputs under `../current_scene/06c_semantic_segmentation/` in a named
run directory. The older `06_semantic_segmentation/` is a reference, not a
source of predefined classes.

## DINOv2 patch-neighbor ablation

```bash
backend/.venv/bin/python backend/06c_semantic_segmentation/run_dinov2_patches.py
```

The first ten input frames are resized without changing their aspect ratio to
a maximum side of 896 pixels and multiples of DINOv2's 14-pixel patch size.
The script uses `facebook/dinov2-small` (the unregistered `dinov2_vits14`
backbone), saves 384-dimensional patch embeddings as float16 `.npy` files,
and colors connected regions formed only by four-neighbor cosine similarities.
The five default thresholds are the 10th, 30th, 50th, 70th, and 90th
percentiles of neighbor similarities across the selected frames. They and
region counts are recorded in `dinov2_patches/<run-name>/summary.json`.
Each threshold directory has one full-resolution RGB-left / colored-patches-right
JPG and one patch-label `.npy` per frame. Colors are local to each frame; these
regions are not yet persistent 3D instances. PCA is omitted from this ablation
so the sweep changes only the cosine threshold. Choose a threshold after
visual review before adding another feature transform.

### Running-mean cluster growth (0.8 and 0.9)

```bash
backend/.venv/bin/python backend/06c_semantic_segmentation/run_dinov2_patches.py \
  --grouping cluster-mean --thresholds 0.8 0.9 \
  --embeddings-from backend/current_scene/06c_semantic_segmentation/dinov2_patches/first10_vits14_neighbors \
  --run-name first10_vits14_cluster_mean_080_090
```

Each region starts at the first unassigned patch. At every step, the most
similar unassigned four-neighbor is compared with the mean embedding of all
current region patches; accepted patches update that mean. Candidates rejected
earlier are reconsidered after growth. The saved embeddings are reused, with
no PCA. Results are under `dinov2_patches/first10_vits14_cluster_mean_080_090/`.
An exact-threshold neighbor-only baseline is saved separately under
`dinov2_patches/first10_vits14_neighbors_080_090/` for atomic comparison.
Mean regions per frame: 76.9 versus 234.4 at 0.8, and 400.8 versus 584.0 at
0.9 (neighbor-only versus running mean). Region count is not an accuracy score;
choose the next configuration by visual inspection.

The same running-mean sweep at 0.7 and 0.6 is saved under
`dinov2_patches/first10_vits14_cluster_mean_070_060/`, with exact-threshold
neighbor-only comparisons under `first10_vits14_neighbors_070_060/`.
Mean regions per frame are 128.0 versus 22.6 at 0.7, and 72.0 versus 9.1 at
0.6 (running mean versus neighbor-only). These remain review candidates.
The 0.5 run is saved under `dinov2_patches/first10_vits14_cluster_mean_050/`
with its exact neighbor-only comparison under `first10_vits14_neighbors_050/`.
Mean regions per frame are 41.6 versus 4.1, respectively.

### Small-region cleanup at 0.55

```bash
backend/.venv/bin/python backend/06c_semantic_segmentation/run_dinov2_patches.py \
  --grouping cluster-mean --thresholds 0.55 --min-region-patches 4 \
  --embeddings-from backend/current_scene/06c_semantic_segmentation/dinov2_patches/first10_vits14_neighbors \
  --run-name first10_vits14_cluster_mean_055_min4
```

The raw 0.55 run is in `first10_vits14_cluster_mean_055_raw/`; the cleaned run
is in `first10_vits14_cluster_mean_055_min4/`. Cleanup absorbs regions of one
to three patches into a four-neighbor touching region. Touching regions are all
one patch away, so ties in proximity are resolved by cosine similarity between
current region means. Merges update those means and preserve surviving color
IDs for visual comparison. Across ten frames, mean region count changes from
53.0 to 12.6. Neither region count nor visual size proves object accuracy;
review the paired images before freezing the cleanup.

### One-percent cleanup sweep

On the 2,304-patch images, `--min-region-patches 24` absorbs every region of
1–23 patches (less than 1% of the image). The same rule was applied at 0.55,
0.60, and 0.65, reusing the saved embeddings and leaving PCA off. Cleaned
results are in `first10_vits14_cluster_mean_055_065_min24/` (0.55 and 0.65)
and `first10_vits14_cluster_mean_060_min24/` (0.60). Exact raw comparisons
are in `first10_vits14_cluster_mean_055_raw/`,
`first10_vits14_cluster_mean_070_060/` (0.60), and
`first10_vits14_cluster_mean_065_raw/`. Mean region counts change from
53.0 to 6.0, 72.0 to 7.7, and 92.5 to 11.6, respectively. The rule and
threshold are not frozen pending visual review.

### PCA ablation

```bash
backend/.venv/bin/python backend/06c_semantic_segmentation/run_dinov2_patches.py \
  --grouping cluster-mean --pca-dims 64 --min-region-patches 24 \
  --thresholds 0.5 0.55 0.6 0.65 0.7 \
  --embeddings-from backend/current_scene/06c_semantic_segmentation/dinov2_patches/first10_vits14_neighbors \
  --run-name first10_vits14_pca64_cluster_mean_min24_050_070
```

One 64-component PCA is fit across the same ten frames so the projection is
consistent across views. The run saves `pca.npz`, projected patch embeddings,
label maps, and RGB/colored-patch pairs under its named folder. The exact
no-PCA comparison uses the prior 0.55/0.60/0.65 cleanup runs plus
`first10_vits14_cluster_mean_050_070_min24/` for 0.50 and 0.70. Mean
surviving regions per frame with PCA versus no PCA are 10.0/5.3 (0.50),
13.1/6.0 (0.55), 16.0/7.7 (0.60), 19.5/11.6 (0.65), and 24.3/15.6
(0.70). These counts describe granularity, not segmentation accuracy.

The wider PCA threshold sweep at 0.55, 0.65, 0.75, 0.85, and 0.95 is in
`first10_vits14_pca64_cluster_mean_min24_055_095/`. It keeps the same
64-component projection, running-mean growth, and 1–23 patch cleanup. Mean
surviving region counts are 13.1, 19.5, 28.9, 44.2, and 52.5, respectively.
Inspect all ten RGB/colored-patch pairs per threshold before selection.

### First 100 frames at 0.50 and 0.55

```bash
backend/.venv/bin/python backend/06c_semantic_segmentation/run_dinov2_patches.py \
  --first-frames 100 --grouping cluster-mean --pca-dims 64 \
  --pca-from backend/current_scene/06c_semantic_segmentation/dinov2_patches/first10_vits14_pca64_cluster_mean_min24_050_070/pca.npz \
  --thresholds 0.5 0.55 --min-region-patches 24 \
  --run-name first100_vits14_pca64_cluster_mean_min24_050_055
```

This reuses the PCA projection fitted for the first-ten-frame review, so its
thresholds remain comparable. The run contains 100 RGB/colored-patch pairs and
label maps per threshold, DINO embeddings, projected embeddings, and a copy
of the PCA coefficients. Each threshold directory also has a
`contact_sheet_100.jpg` overview. The first ten labels exactly reproduce the
earlier PCA run. Mean surviving regions per frame are 6.64 at 0.50 and 8.62
at 0.55. These are local 2D regions, not persistent 3D objects.

### Round-based grid merging ablation

```bash
backend/.venv/bin/python backend/06c_semantic_segmentation/run_dinov2_patches.py \
  --first-frames 100 --grouping grid-rounds \
  --embeddings-from backend/current_scene/06c_semantic_segmentation/dinov2_patches/first100_vits14_pca64_cluster_mean_min24_050_055 \
  --pca-dims 64 \
  --pca-from backend/current_scene/06c_semantic_segmentation/dinov2_patches/first100_vits14_pca64_cluster_mean_min24_050_055/pca.npz \
  --thresholds 0.5 0.6 0.7 --min-region-patches 24 \
  --run-name first100_vits14_pca64_grid_rounds_min24_050_070
```

Every patch begins as a region. In each round, adjacent regions compare their
frozen mean embeddings; mutually preferred pairs above the threshold merge
simultaneously, and means update only between rounds. The same PCA, images,
and 1–23 patch cleanup are used as in the sequential run. The grid results
have mean region counts 6.27/10.57/18.88 at 0.50/0.60/0.70, versus
6.64/11.28/20.67 for sequential growth. Exact sequential 0.60/0.70
references are in `first100_vits14_pca64_cluster_mean_min24_060_070/`;
0.50 is in the prior first-100 run. Each threshold directory has 100 full
pairs, labels, and a `contact_sheet_100.jpg` overview. Counts do not indicate
which method better isolates objects; select by visual review.

## Image coverage step

1. Run the existing fast SAM 2.1 Small proposal generator with aspect-preserving
   letterbox input and an 8x8 prompt grid. Preserve all masks and confidence values;
   measure the union of covered pixels per frame.
2. When the grid covers less than 90% of a frame, place a positive prompt at
   the distance-transform peak of each uncovered connected region. Reuse the
   same frame's SAM2 image features for up to three prompted rounds, accepting
   masks that cover new pixels. The fixed-size 16-point decoder batch avoids
   repeated compiler specializations.
3. Resolve overlaps into a disjoint pixel label map, prioritizing initial grid
   masks before gap masks and sorting each by predicted IoU. Give every
   still-uncovered connected region a **residual/unknown** label. This ensures
   exactly 100% pixel assignment without inventing an object at a weakly
   observed pixel. Keep a separate metric for coverage by accepted SAM2 masks.
4. Lift only reliable object masks through mesh face-ID/depth maps. Keep
   residual pixels and geometry misses explicit; neither is evidence of a 3D
   object. Reconcile instances across views using 3D overlap and visibility,
   then evaluate clone similarity only after instance IDs are stable.

Run the current raw-mask review pass on the first 50 frames with:

```bash
backend/.venv/bin/python backend/06c_semantic_segmentation/run_sam2_proposals.py \
  --first-frames 50 --run-name first50_grid8_letterbox_gap060_trigger90_raw
```

Each run saves per-frame `.npz` archives (`masks` for accepted SAM proposals,
`labels` for complete pixel assignment, `residual` for unknown pixels), JSON
metrics, overlay PNGs, and `summary.json` under
`backend/current_scene/06c_semantic_segmentation/sam2_proposals/<run-name>/`.
Unknown pixels are pure black in new PNG overlays. Earlier uncleaned ablation
JPGs used red for unknown pixels. `labels` is a preview partition; 3D mesh
visibility and instance association are later stages.

The shape-cleanup script was used only in earlier ablations. It is **not** part
of the current run; all accepted SAM masks are retained. Earlier cleaned runs
can still be reproduced with:

```bash
backend/.venv/bin/python backend/06c_semantic_segmentation/clean_masks.py \
  --source first50_grid10_gap060_trigger94 \
  --run-name first50_grid10_gap060_trigger94_clean
```

That filter removes masks with at least three substantial disconnected pieces,
very thin long ribbons, or sparse branched shapes. It preserves two-piece
masks because an occluder can split one object. The cleaned run keeps every
removed mask's ID and reason in its frame JSON, and assigns newly uncovered
pixels unknown. Its PNG overlays show these pixels as exact black.

Review per-frame initial, prompted, and residual overlays; compare proposal
coverage, residual area, proposal count, runtime, and 3D consistency for each
configuration. Save every sweep configuration separately and freeze one only
after manual inspection. Pixel assignment coverage and correct object
recognition are distinct metrics.

The first 50 frames currently have two complete-mask runs for threshold
ablation. With grid 10, the 0.85 gap stability threshold reached 92.2% mean
accepted SAM coverage at 5.22 inference FPS; 0.60 reached 96.2% at 4.30 FPS.
Both label maps assign 100% of pixels, with the remaining 7.8% and 3.8%
respectively labeled unknown. These numbers measure pixel coverage, not
instance correctness. Neither configuration is frozen pending visual review.

Shape cleanup removed 143 of 1,846 masks on these frames. Mean accepted-mask
coverage changed from 96.2% to 95.3%; complete pixel assignment remains 100%
with the remainder unknown. The filter was run offline and does not change
the recorded SAM inference FPS.

In an earlier grid-10 stretch run, the 94% trigger skipped gap prompting on 18 of the first 50 frames. It changed
accepted-mask coverage from 96.2% to 95.3% and steady inference from 4.30 to
5.22 FPS. Shape cleanup of that gated run removes 108 masks and leaves 94.6%
accepted-mask coverage. The other 5.4% remains explicitly unknown.

### Resize comparison

The earlier baseline stretched each 1080x1920 frame to 1024x1024. To preserve aspect
ratio instead, `--resize-mode letterbox` scales it to 576x1024, pads 224 pixels
on each horizontal side with the normalized image mean, maps prompt coordinates
into that rectangle, and crops mask logits before returning to original size.
The user selected letterbox as the new default after visual review. The
stretch and letterbox results below remain available as historical ablations.
With raw masks and letterbox input on the same 50 frames, grid 10 / trigger 94%
reached 95.68% mean accepted-mask coverage and 5.36 steady FPS; grid 8 /
trigger 94% reached 95.56% and 6.05 FPS; grid 8 / trigger 90% reached 93.89%
and 8.04 FPS. The 90% trigger prompts gaps on 15 frames versus 31 at 94%.
These are coverage and inference measurements, not object-quality scores.
Paired sheets for each change are in `sam2_proposals/grid8_letterbox_default_ablation/`.

```bash
backend/.venv/bin/python backend/06c_semantic_segmentation/run_sam2_proposals.py \
  --first-frames 50 --grid 10 --fill-gaps --gap-stability 0.6 \
  --coverage-trigger 0.94 --resize-mode letterbox \
  --run-name first50_grid10_gap060_trigger94_letterbox
backend/.venv/bin/python backend/06c_semantic_segmentation/clean_masks.py \
  --source first50_grid10_gap060_trigger94_letterbox \
  --run-name first50_grid10_gap060_trigger94_letterbox_clean
```

On the same 50 frames, mean accepted-mask coverage after shape cleanup is
94.56% for stretch and 94.86% for letterbox; letterbox gains coverage on 33
frames and loses coverage on 16. Steady inference is 5.22 versus 5.36 FPS,
respectively. Coverage is not object-mask accuracy: frame 00098 loses 18.3
percentage points with letterbox, while frame 00138 gains 7.2. Inspect the
paired contact sheet and full-resolution examples under
`sam2_proposals/resize_comparison_stretch_vs_letterbox/` before choosing one.

## Saved runs

All paths below are inside `backend/current_scene/06c_semantic_segmentation/`.
`inputs/` is the preserved scene input, including source images, supplied
masks, COLMAP poses, mesh, and Gaussian splat. Each run below contains saved
mask archives and per-frame metrics; colors are local proposal IDs, not
persistent objects across frames.

| Folder under `sam2_proposals/` | Purpose and what to inspect |
| --- | --- |
| `first10_stab80_iou50_reg05pct/` | **Validated proposal & refinement candidate.** SAM2 Small with `stability_thresh=0.80`, `pred_iou_thresh=0.50`, outlier threshold $0.5\%$ frame area ($10{,}368\text{ px}$), filament filtering, hierarchical overlap subtraction ($33\%$ relative confidence), and SOR floater cleanup. Averages $90.48\%$ coverage and $11.5$ clean instance proposals per frame without spilling or dropping small furniture items. Includes side-by-side Raw vs Refined comparisons. |

## DINOv2 Mask Embeddings & Iterative Semantic Merging

```bash
backend/.venv/bin/python backend/06c_semantic_segmentation/run_dinov2_merge.py
```

Using the validated refined SAM2 proposals (`first10_stab80_iou50_reg05pct`), DINOv2 (`facebook/dinov2-small`) computes dense spatial feature tokens across each image. For each mask proposal $M$, its semantic embedding is computed by spatially pooling the DINOv2 patch tokens covered by $M$ and L2-normalizing the resulting vector:
$$\mathbf{e}_A = \frac{\sum_{p \in M_A} \mathbf{f}_p}{\left\| \sum_{p \in M_A} \mathbf{f}_p \right\|_2}$$

### Iterative Merge Algorithm
1. Construct the pairwise cosine similarity matrix $\mathbf{S}_{ij} = \mathbf{e}_i^\top \mathbf{e}_j$.
2. Identify the pair $(i, j)$ with maximum cosine similarity $\max_{i < j} \mathbf{S}_{ij}$.
3. If $\max \mathbf{S}_{ij} > 0.70$, merge masks $M_i$ and $M_j$ into a single mask $M_{\text{merged}} = M_i \cup M_j$, retaining the lower ID $\min(i, j)$.
4. The semantic embedding of the merged mask is updated exactly via vector sum pooling $\mathbf{S}_{\text{merged}} = \mathbf{S}_i + \mathbf{S}_j$ (size-weighted normalized average), eliminating redundant neural re-inference.
5. Recompute the reduced similarity matrix and repeat until all pairwise similarities are $\le 0.70$.

Outputs are saved in `dinov2_merged_070/first10_stab80_iou50_reg05pct/`:
- `frame_XXXXX_merged.npz`: Final masks, label map, normalized embeddings, and final similarity matrix.
- `frame_XXXXX_merged.json`: Merge trace, area, confidence, and source mask IDs.
- `frame_XXXXX_final_similarity_matrix.png`: Upper-triangular similarity matrix where diagonal and axis ticks are color-coded to each mask's unique overlay color.
- `frame_XXXXX_comparison_with_matrix.png`: 3-panel composite visualization `[Pre-Merge Overlay | Post-Merge Overlay | Final Triangular Matrix]`.

### First 50 Frames Staged Benchmark & Profiling ($\ge 0.69$ Threshold)

Run script: `backend/06c_semantic_segmentation/run_pipeline_staged_experiments.py`.  
Saved runs:
- `backend/current_scene/06c_semantic_segmentation/first50_opt_grid8_reprompt_merge069/`
- `backend/current_scene/06c_semantic_segmentation/first50_opt_grid12_noreprompt_merge069/`

The pipeline is partitioned into 4 decoupled batch stages across all images:
1. **Stage 1 (Batch SAM2)**: Sequential GPU evaluation caching logits in VRAM.
2. **Stage 2 (Batch GPU Refinement)**: Pre-pruning low IoU/area before refinement, batched GPU tensor guided filtering (`torch.nn.functional.avg_pool2d`), downsampled filament analysis, and hierarchical overlap carving. (Reduced refinement time from $4{,}619\text{ ms} \to 327\text{ ms}$, a **$14.1\times$ speedup**).
3. **Stage 3 (Batch DINOv2)**: Spatial ViT token extraction and patch projection across all images.
4. **Stage 4 (Semantic Merging)**: Pairwise cosine similarity matrix construction and iterative merging ($\ge 0.69$) via exact size-weighted feature sum updates.

#### Comparative Benchmark Results (50 Frames)

| Stage | Config A: $8\times 8$ Grid + Reprompt | Config B: $12\times 12$ Grid (No Reprompt) | Analysis |
| :--- | :---: | :---: | :--- |
| **SAM2 Inference** | $248.8\text{ ms} \pm 250.2\text{ ms}$ | **$287.6\text{ ms} \pm 66.5\text{ ms}$** | Config A has lower mean time but high variance ($\pm 250\text{ ms}$) due to secondary gap passes. Config B has ultra-consistent latency ($\pm 66\text{ ms}$) with no gap passes. |
| **Mask Refinement** | **$327.2\text{ ms} \pm 99.9\text{ ms}$** | $458.5\text{ ms} \pm 184.5\text{ ms}$ | Config B processes $144$ grid points vs $64$, generating more raw candidate masks ($9.44$ vs $8.76$ survivors). |
| **DINOv2 First Inference** | **$165.9\text{ ms} \pm 7.2\text{ ms}$** | $167.8\text{ ms} \pm 6.1\text{ ms}$ | Identical single GPU forward pass. |
| **DINOv2 Iterative Merging** | **$0.64\text{ ms} \pm 0.63\text{ ms}$** | $0.75\text{ ms} \pm 0.55\text{ ms}$ | Microsecond vector additions. |
| **Total Pipeline per Image** | **$742.5\text{ ms}$ ($1.35\text{ FPS}$)** | **$914.7\text{ ms}$ ($1.09\text{ FPS}$)** | **$6.8\times$ faster overall throughput** than unoptimized baseline ($5{,}022\text{ ms}$). |
| **Final Objects & Coverage** | $6.72\text{ objects}$, $89.96\%$ coverage | **$7.00\text{ objects}$, $91.12\%$ coverage** | Config B yields slightly higher pixel coverage ($+1.16\%$) and slightly richer object separation. |

### Streamlined $12\times 12$ Pipeline Benchmark (Two-Tier Regularization: Floater Cleanup & Chunk Preservation)

Script: `backend/06c_semantic_segmentation/run_streamlined_grid12_bench.py`.  
Saved outputs: `backend/current_scene/06c_semantic_segmentation/first50_streamlined_grid12_merge069/`

The refined $12\times 12$ single-pass pipeline incorporates:
1. **GPU Tensor Hierarchical Overlap Carving**: Resolves overlapping proposals directly on GPU boolean tensors.
2. **Two-Tier Regularization Filter**:
   - **Tier 1 (Satellite Floater & Crumb Removal)**: Discards disconnected edge crumbs ($< 1{,}000\text{ px}$ or $< 5\%$ of mask mass), eliminating strings of pearls along high-contrast object seams (e.g. `frame_00163` mask #15 reduced from 45 components to **1 solid body**, mask #9 from 6 components to **1 solid body**).
   - **Tier 2 (Morphological Ribbon Pruning & Chunk Retention)**: Strips border ribbons $< 7\text{ px}$ wide while preserving solid chunky objects ($\ge 0.5\%$ frame area, thickness $\ge 12\text{ px}$, e.g. `frame_00178` mask #3 chunky body retained, `frame_00189` border thread completely pruned).
3. **DINOv2 First Inference & Iterative Semantic Merging ($\ge 0.69$)**.

#### Timing Breakdown Across All 50 Frames:
- **SAM2 Inference ($12\times 12$)**: $198.30\text{ ms} \pm 18.02\text{ ms}$ (steady single-pass, no logit transfers or re-prompting stalls).
- **Initial Mask Refinement**: **$114.75\text{ ms} \pm 27.55\text{ ms}$** (a **$40.3\times$ speedup** over the initial $4{,}619\text{ ms}$ CPU baseline).
- **DINOv2 First Inference**: $173.45\text{ ms} \pm 7.51\text{ ms}$.
- **DINOv2 Iterative Merging**: $1.39\text{ ms} \pm 0.68\text{ ms}$.
- **Total Pipeline per Image**: **$487.87\text{ ms}$ ($\mathbf{2.05\text{ FPS}}$)** (overall pipeline speedup is **$10.3\times$** compared to the original $5{,}022\text{ ms}$ baseline).
- **Coverage & Object Separation**:
  - Mean pixel coverage across all 50 frames: **$91.84\%$**.
  - Averages $12.92$ initial refined proposals converging to **$8.46$ clean physical objects**.






| `first50_grid8_letterbox_gap060_trigger90_raw/` | **Current review candidate.** Fifty raw letterbox/grid-8 overlays with a 90% gap trigger. No shape cleanup is applied; black pixels are uncovered. |
| `first50_grid8_letterbox_gap060_trigger94_raw/` | Same raw letterbox/grid-8 configuration with the earlier 94% trigger; isolates the trigger change. |
| `grid8_letterbox_default_ablation/` | Paired contact sheets: grid 10 versus grid 8 with trigger 94%, then trigger 94% versus 90% with grid 8. `summary.json` lists per-frame coverage and steady FPS. |
| `first50_grid10_gap060_trigger94_clean/` | Earlier stretch/grid-10 candidate with shape cleanup. Fifty PNG overlays with uncovered pixels in pure black; `contact_sheet.png` gives an overview and frame JSON files list removed masks and reasons. |
| `first50_grid10_gap060_trigger94_letterbox_clean/` | Same settings and 50 frames with aspect-preserving padding and the same shape filter; compare directly with the current stretched candidate. |
| `first50_grid10_gap060_trigger94_letterbox/` | Letterbox predictions before shape filtering, retained so removed proposals can be inspected. |
| `resize_comparison_stretch_vs_letterbox/` | Paired overlays: stretch on the left, letterbox on the right. `all50_contact_sheet.png` covers all frames, `frame_*_pair.png` shows selected gains/losses at full resolution, and `comparison.json` lists per-frame coverage. |
| `first50_grid10_gap060_trigger94/` | Same 50 frames before shape filtering. Compare against the clean run to judge removed masks; inspect black gaps and `gate94_comparison.png` for the coverage lost by skipping gap prompts. |
| `first50_grid10_gap060_clean/` | Previous cleanup with gap prompts on every frame. Compare with the gated clean run when black gaps matter more than runtime. |
| `first50_grid10_gap060/` | Raw 0.60 gap-stability run on every frame; retained as the source for the previous cleanup and a coverage ablation. Older JPG overlays mark unknown pixels red. |
| `first50_grid10_complete/` | Raw 0.85 gap-stability run on every frame. Compare with 0.60 to see the effect of accepting less-stable gap proposals; older JPG overlays mark unknown red. |
| `small_grid10_compiled/` | Eight spaced frames from the 10x10 grid alone, without gap prompts. Initial-proposal reference. |
| `small_grid8_compiled_baseline/` | Eight spaced frames from the faster 8x8 compiled grid, without gap prompts. Compare proposal coverage with grid 10. |
| `small_grid8_eager_reference/` | Same 8x8 settings without compilation; speed ablation for the compiled path. |
| `standard_amg_grid8_reference/` | Three frames from SAM2's standard automatic generator; older implementation reference, not a full-scene result. |
| `worst4_gap_stability060/` | Four difficult frames used to check the 0.60 gap threshold before the full run; superseded by the 50-frame runs. |

`sam2_proposals/inference_profile.json` contains per-stage timing for three
difficult frames. Grid 8 compiled reached 10.99 FPS; grid 10 alone reached
9.01 FPS. The extra decoder passes and connected-region work explain the
remaining drop to 4.30 FPS when gap prompting runs on every frame.
