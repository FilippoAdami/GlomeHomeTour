# Image caption comparison

Run from the workspace root with the existing ROCm `.venv`. The script requires a GPU, reads the ten `images/*.jpg` files in filename order, and overwrites its model's Markdown report on each run.

Cache the model weights and processors once before timing (the standard HTTP path worked on this machine):

```bash
for model in microsoft/Florence-2-base microsoft/Florence-2-large Qwen/Qwen3-VL-2B-Instruct; do
  HF_HUB_DISABLE_XET=1 hf download "$model" --include '*.json' '*.safetensors' '*.txt' '*.model' '*.py' '*.tiktoken'
done
```

Run the selected Florence-2-base `<CAPTION>` preset:

```bash
.venv/bin/python semantic_inference/caption_images.py
```

The other comparison runs remain available:

```bash
.venv/bin/python semantic_inference/caption_images.py florence-2-base --detailed
.venv/bin/python semantic_inference/caption_images.py florence-2-large
.venv/bin/python semantic_inference/caption_images.py qwen3-vl-2b-instruct
```

`--short` remains an explicit alias for Florence's `<CAPTION>` preset. It writes a separate `-caption.md` report rather than replacing the detailed comparison.

The selected settings keep Florence's 768-pixel input, FP16 weights, and 3-beam decoding. In warm ten-image checks, 512-pixel input cut caption time from 5.57 s to 2.60 s but changed every caption and lost the map's colors; greedy decoding changed four captions with little warm-run speed gain. Batching two images was slightly slower than sequential inference. These alternatives were left out to preserve the selected captions.

The reports separate cached model load, ten-image captioning, and unload. The first image can include GPU kernel startup; initial network download is outside the timed phases. Qwen's input is capped at 512² pixels of area to keep ROCm inference practical; fine text can be missed or invented.

Initial cache transfers on this machine (2026-10-12): Florence-base 172 s, Florence-large 761 s, Qwen 1,636 s. The large and Qwen transfers overlapped; these are network times, not model load times.
