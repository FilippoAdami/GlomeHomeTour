# microsoft/Florence-2-base

## Run

- GPU: AMD Radeon RX 9060 XT; PyTorch 2.10.0+rocm7.0; Transformers 4.57.3
- Precision: torch.float16; attention: eager; inference mode; sequential images
- Weights cached before timing; generation deterministic; Florence uses `<DETAILED_CAPTION>`, 3 beams, and no KV cache due to a Transformers 4.57 compatibility issue; Qwen uses the one-sentence prompt in `caption_images.py`.
- Load includes processor/model creation and GPU transfer; caption includes image decode, preprocessing, generation, and GPU synchronization; unload includes garbage collection and clearing PyTorch's GPU cache.
- GPU utilization is mean / peak of 0.5-second `rocm-smi` samples; VRAM is peak total device use and includes other processes; brief unload can finish between samples.

## Timing and GPU usage

| Step | Seconds | GPU use mean / peak | Peak device VRAM |
| --- | ---: | ---: | ---: |
| load | 1.91 | 14% / 24% | 1.74 GiB |
| caption | 11.30 | 71% / 97% | 2.48 GiB |
| unload | 0.12 | not sampled | not sampled |
| **Total** | **13.33** | | |

Peak PyTorch allocated memory during captioning: **0.67 GiB**.

## Captions

### [bench.jpg](../images/bench.jpg)

The image shows a wooden bench with two dumbbells sitting on top of it in a room with a wall in the background and a curtain on the right side.

Caption step: 3.83 s

### [desk_scene.jpg](../images/desk_scene.jpg)

The image shows a desk with a computer and a chair in front of it. On the desk there is a monitor, speakers, a globe, and other objects. To the right of the desk is a box with some objects inside. In the background, there are photo frames on the wall and a mirror on the left side.

Caption step: 0.99 s

### [drawer.jpg](../images/drawer.jpg)

The image shows a teddy bear sitting on top of a wooden chest of drawers, with a box underneath it. On the right side of the chest is a bed, and in the background is a wall with a switchboard.

Caption step: 0.76 s

### [library.jpg](../images/library.jpg)

The image shows a white bookcase filled with lots of books sitting on top of a white wall. The books are neatly arranged on the shelves, giving the impression of a well-organized and well-stocked library.

Caption step: 0.74 s

### [mirror.jpg](../images/mirror.jpg)

The image shows a wall with a mirror attached to it, a switchboard, and a cable. The mirror is mounted on the wall, and the switchboard is connected to the wall. The cable is visible in the top right corner of the image.

Caption step: 0.82 s

### [person.jpg](../images/person.jpg)

The image shows a man in a white shirt smiling at the camera. He is standing in front of a wall with a window and a curtain in the background.

Caption step: 0.75 s

### [punchingbag.jpg](../images/punchingbag.jpg)

The image shows a blue punching bag hanging from the ceiling in a room, with a rope attached to it. On the right side of the image, there is a cupboard with drawers, and in the background, there are frames attached to the wall.

Caption step: 0.96 s

### [puzzle.jpg](../images/puzzle.jpg)

The image shows a jigsaw puzzle hanging on a wall in a room. The puzzle pieces are arranged in a way that suggests they are part of a larger puzzle. The colors of the puzzle are vibrant and the details of the horses and the building in the background are intricate.

Caption step: 0.85 s

### [radiator.jpg](../images/radiator.jpg)

The image shows a white radiator sitting on top of a wooden floor, with a wall in the background and a window on the right side.

Caption step: 0.79 s

### [worldmap.jpg](../images/worldmap.jpg)

The image shows a black and red world map hanging on a wall in a room, illuminated by a light at the top. The map is made of metal and is part of a collection of world map wall art, perfect for adding a unique touch to any room.

Caption step: 0.81 s
