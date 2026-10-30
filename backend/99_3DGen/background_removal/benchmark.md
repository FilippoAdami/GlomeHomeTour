# BiRefNet Lite background removal

Model: [ZhengPeng7/BiRefNet_lite](https://huggingface.co/ZhengPeng7/BiRefNet_lite) at `aa62cd87eafb9cc43056d08ef3615a14628b831d`; GPU: AMD Radeon RX 9060 XT; PyTorch: 2.10.0+rocm7.0.
Model input: 1024×1024 RGB, float16; masks resized to each original image size; outputs are transparent PNGs.
512×512 was tested but produced empty masks for the library and radiator, so this run uses the model's documented input size.
Timing includes model load and ten sequential image reads, preprocessing, inference, GPU synchronization, and PNG writes. Precached model weights; no download time.
The first image includes ROCm kernel warmup.

| Step | Seconds |
| --- | ---: |
| Model load | 0.85 |
| [bench.jpg](../images/bench.jpg) → [bench.png](bench.png) | 3.06 |
| [desk_scene.jpg](../images/desk_scene.jpg) → [desk_scene.png](desk_scene.png) | 0.40 |
| [drawer.jpg](../images/drawer.jpg) → [drawer.png](drawer.png) | 0.23 |
| [library.jpg](../images/library.jpg) → [library.png](library.png) | 0.21 |
| [mirror.jpg](../images/mirror.jpg) → [mirror.png](mirror.png) | 0.26 |
| [person.jpg](../images/person.jpg) → [person.png](person.png) | 0.35 |
| [punchingbag.jpg](../images/punchingbag.jpg) → [punchingbag.png](punchingbag.png) | 0.24 |
| [puzzle.jpg](../images/puzzle.jpg) → [puzzle.png](puzzle.png) | 0.33 |
| [radiator.jpg](../images/radiator.jpg) → [radiator.png](radiator.png) | 0.21 |
| [worldmap.jpg](../images/worldmap.jpg) → [worldmap.png](worldmap.png) | 0.37 |
| **Total** | **6.51** |

## Resources

- Peak device VRAM: 4.75 GiB (total device use, including other processes).
- Peak PyTorch allocated VRAM during processing: 1.75 GiB.
- Peak process RSS: 1.72 GiB.
- Process CPU time: 8.47 s.
- GPU use: 17% mean, 51% peak (0.2 s samples).

Visual check: the drawer and recessed library shelves have incomplete masks; the world map has small holes. These are model segmentation limits on these images.
