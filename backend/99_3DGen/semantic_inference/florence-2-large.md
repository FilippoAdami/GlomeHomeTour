# microsoft/Florence-2-large

## Run

- GPU: AMD Radeon RX 9060 XT; PyTorch 2.10.0+rocm7.0; Transformers 4.57.3
- Precision: torch.float16; attention: eager; inference mode; sequential images
- Weights cached before timing; generation deterministic; Florence uses `<DETAILED_CAPTION>`, 3 beams, and no KV cache due to a Transformers 4.57 compatibility issue; Qwen uses the one-sentence prompt in `caption_images.py`.
- Load includes processor/model creation and GPU transfer; caption includes image decode, preprocessing, generation, and GPU synchronization; unload includes garbage collection and clearing PyTorch's GPU cache.
- GPU utilization is mean / peak of 0.5-second `rocm-smi` samples; VRAM is peak total device use and includes other processes; brief unload can finish between samples.

## Timing and GPU usage

| Step | Seconds | GPU use mean / peak | Peak device VRAM |
| --- | ---: | ---: | ---: |
| load | 2.13 | 12% / 29% | 2.38 GiB |
| caption | 38.01 | 91% / 100% | 3.67 GiB |
| unload | 0.12 | not sampled | not sampled |
| **Total** | **40.26** | | |

Peak PyTorch allocated memory during captioning: **1.83 GiB**.

## Captions

### [bench.jpg](../images/bench.jpg)

The image shows a wooden bench with a white cloth draped over it and a pair of dumbbells resting on top of it. The bench is situated on a wooden floor, with a wall in the background.

Caption step: 17.47 s

### [desk_scene.jpg](../images/desk_scene.jpg)

The image shows a computer desk with a monitor, keyboard, mouse, speakers, globe, and other objects on top of it, a chair in front of the desk, a carpet on the floor, and a wall with photo frames in the background.

Caption step: 2.34 s

### [drawer.jpg](../images/drawer.jpg)

The image shows a teddy bear sitting on top of a yellow and white Ikea chest of drawers, with a pillow on the right side and a wall with a switchboard in the background.

Caption step: 2.31 s

### [library.jpg](../images/library.jpg)

The image shows a bookshelf filled with lots of books next to a door, with a wall in the background. The books are neatly arranged on the shelves, creating a neat and organized look.

Caption step: 2.20 s

### [mirror.jpg](../images/mirror.jpg)

The image shows a wall-mounted magnifying mirror hanging on the wall next to a switchboard. The mirror is round in shape and has a handle on the side for easy portability. The switchboard is located at the bottom of the wall and is likely used to control the brightness of the mirror.

Caption step: 2.55 s

### [person.jpg](../images/person.jpg)

The image shows a man in a white t-shirt smiling at the camera while taking a selfie in a room with a wall, window, curtain, and wooden ceiling.

Caption step: 2.19 s

### [punchingbag.jpg](../images/punchingbag.jpg)

The image shows a blue punching bag hanging from a chain in a room, with a cupboard in the background, a wall with a photo frame, and a floor at the bottom.

Caption step: 2.19 s

### [puzzle.jpg](../images/puzzle.jpg)

The image shows a jigsaw puzzle hanging on the wall in a room. The puzzle is framed and features a painting of three horses running in the water, with a building in the background.

Caption step: 2.15 s

### [radiator.jpg](../images/radiator.jpg)

The image shows a white radiator sitting on top of a wooden floor, with two windows on either side of it. The radiator appears to be in need of repair, as evidenced by the tools and materials scattered around it.

Caption step: 2.38 s

### [worldmap.jpg](../images/worldmap.jpg)

The image shows a black and red world map hanging on a wall in a room, illuminated by a light at the top. The map is made of wood, giving it a unique and rustic look.

Caption step: 2.24 s
