# microsoft/Florence-2-base

## Run

- GPU: AMD Radeon RX 9060 XT; PyTorch 2.10.0+rocm7.0; Transformers 4.57.3
- Precision: torch.float16; attention: eager; inference mode; sequential images
- Weights cached before timing; generation deterministic; Florence uses `<CAPTION>`, 3 beams, and no KV cache due to a Transformers 4.57 compatibility issue; Qwen uses the one-sentence prompt in `caption_images.py`.
- Load includes processor/model creation and GPU transfer; caption includes image decode, preprocessing, generation, and GPU synchronization; unload includes garbage collection and clearing PyTorch's GPU cache.
- GPU utilization is mean / peak of 0.5-second `rocm-smi` samples; VRAM is peak total device use and includes other processes; brief unload can finish between samples.

## Timing and GPU usage

| Step | Seconds | GPU use mean / peak | Peak device VRAM |
| --- | ---: | ---: | ---: |
| load | 1.38 | 8% / 15% | 1.88 GiB |
| caption | 8.64 | 61% / 100% | 2.61 GiB |
| unload | 0.12 | 66% / 66% | 2.61 GiB |
| **Total** | **10.15** | | |

Peak PyTorch allocated memory during captioning: **0.67 GiB**.

## Captions

### [bench.jpg](../images/bench.jpg)

a wooden bench with two dumbbells on top of it

Caption step: 3.60 s

### [desk_scene.jpg](../images/desk_scene.jpg)

A desk with a computer and a chair in a room.

Caption step: 0.57 s

### [drawer.jpg](../images/drawer.jpg)

a teddy bear sitting on top of a dresser

Caption step: 0.57 s

### [library.jpg](../images/library.jpg)

A bookshelf filled with lots of books next to a door.

Caption step: 0.57 s

### [mirror.jpg](../images/mirror.jpg)

A mirror hanging on a wall next to a light switch.

Caption step: 0.56 s

### [person.jpg](../images/person.jpg)

A man in a white shirt smiles at the camera.

Caption step: 0.56 s

### [punchingbag.jpg](../images/punchingbag.jpg)

A blue punching bag hanging from a chain in a room.

Caption step: 0.56 s

### [puzzle.jpg](../images/puzzle.jpg)

A picture of three horses running in front of a castle.

Caption step: 0.56 s

### [radiator.jpg](../images/radiator.jpg)

a white radiator sitting on top of a wooden floor

Caption step: 0.55 s

### [worldmap.jpg](../images/worldmap.jpg)

A black and red world map hanging on a wall.

Caption step: 0.55 s
