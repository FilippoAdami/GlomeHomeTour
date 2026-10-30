# Qwen/Qwen3-VL-2B-Instruct

## Run

- GPU: AMD Radeon RX 9060 XT; PyTorch 2.10.0+rocm7.0; Transformers 4.57.3
- Precision: torch.bfloat16; attention: sdpa; inference mode; sequential images
- Weights cached before timing; generation deterministic; Florence uses `<DETAILED_CAPTION>`, 3 beams, and no KV cache due to a Transformers 4.57 compatibility issue; Qwen uses the one-sentence prompt in `caption_images.py`.
- Qwen images are resized to at most 512² pixels of area, preserving aspect ratio.
- Load includes processor/model creation and GPU transfer; caption includes image decode, preprocessing, generation, and GPU synchronization; unload includes garbage collection and clearing PyTorch's GPU cache.
- GPU utilization is mean / peak of 0.5-second `rocm-smi` samples; VRAM is peak total device use and includes other processes; brief unload can finish between samples.

## Timing and GPU usage

| Step | Seconds | GPU use mean / peak | Peak device VRAM |
| --- | ---: | ---: | ---: |
| load | 3.27 | 9% / 17% | 4.37 GiB |
| caption | 152.94 | 98% / 100% | 6.21 GiB |
| unload | 0.13 | not sampled | not sampled |
| **Total** | **156.35** | | |

Peak PyTorch allocated memory during captioning: **4.22 GiB**.

## Captions

### [bench.jpg](../images/bench.jpg)

A wooden gym bench with a white towel draped over it, featuring a black logo, is positioned on a light-colored wooden floor, with three black dumbbells resting on the floor in front of it.

Caption step: 29.92 s

### [desk_scene.jpg](../images/desk_scene.jpg)

A computer workstation set up in a cozy, attic-style room with a distinctive orange shag carpet, featuring a black ergonomic office chair, a light-colored desk with a yellow drawer, a computer monitor displaying a dark interface, a black speaker, a small globe, and several framed black-and-white portraits on the wall, all under a sloped ceiling.

Caption step: 30.67 s

### [drawer.jpg](../images/drawer.jpg)

A yellow two-drawer bedside table with a white top, featuring a grey, plush, circular cushion and a small, light-colored teddy bear sitting on it, is placed on a wooden floor next to a bed with a grey comforter, with a white wall and a power outlet visible in the background.

Caption step: 1.87 s

### [library.jpg](../images/library.jpg)

A white bookshelf with three open shelves is filled with a collection of books, including titles like "Atomic Habits" and "The 7 Habits of Highly Effective People," with the top shelf also holding a few books with titles such as "The Complete Guide to the Universe" and "Pulse."

Caption step: 1.88 s

### [mirror.jpg](../images/mirror.jpg)

A chrome-finished, foldable, circular wall-mounted mirror with a telescopic arm, mounted on a textured wall, reflects a television screen and a person's head, with a silver light switch below.

Caption step: 1.29 s

### [person.jpg](../images/person.jpg)

A young man with dark, wavy hair and a broad smile, wearing a light-colored t-shirt with a small blue butterfly graphic on the left chest, is photographed in a room with a prominent wooden ceiling and beams, and a window with a curtain in the background.

Caption step: 28.09 s

### [punchingbag.jpg](../images/punchingbag.jpg)

A vibrant blue, cylindrical punching bag with the brand name "El Jem" and the website "www.hub39.com" printed in white, is suspended by a chain from the ceiling, hanging in a room with a white cabinet and an orange drawer, and a white and yellow dresser in the background.

Caption step: 1.89 s

### [puzzle.jpg](../images/puzzle.jpg)

A completed jigsaw puzzle depicting a fantastical scene of two white horses running through a deep blue, icy landscape, with a grand, illuminated castle and a celestial sky featuring planets and a glowing sun in the background.

Caption step: 28.40 s

### [radiator.jpg](../images/radiator.jpg)

A white, vertical, ribbed radiator is mounted on a light-colored wall, with a wooden floor and a wooden window frame visible in the background.

Caption step: 0.99 s

### [worldmap.jpg](../images/worldmap.jpg)

A large, three-dimensional world map with a dark, textured black background and continents rendered in a vibrant, cracked red-orange material, creating a striking and artistic display.

Caption step: 27.94 s
