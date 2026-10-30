"""Generate semantic captions in one model load and return one caption per image."""

import argparse
import gc
import os
import time
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
IMAGES = sorted((ROOT / "images").glob("*.jpg"))
MODELS = {
    "florence-2-base": "microsoft/Florence-2-base",
    "florence-2-large": "microsoft/Florence-2-large",
    "qwen3-vl-2b-instruct": "Qwen/Qwen3-VL-2B-Instruct",
}


def _load_image(image):
    if isinstance(image, Image.Image):
        return image.copy()
    with Image.open(image) as source:
        return source.convert("RGB")


def generate_captions(images, detail="minimum", model_name="florence-2-base"):
    """Return one caption per image; detail is ``minimum`` or ``detailed``."""
    if isinstance(images, (str, os.PathLike, Image.Image)):
        raise TypeError("images must be a list of image paths or PIL images")
    images = list(images)
    if not images:
        return []
    if detail not in {"minimum", "detailed"}:
        raise ValueError("detail must be 'minimum' or 'detailed'")
    if model_name not in MODELS:
        raise ValueError(f"model_name must be one of {', '.join(MODELS)}")

    import torch
    from transformers import AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("Semantic captioning requires a ROCm or CUDA GPU")

    florence = model_name.startswith("florence")
    dtype = torch.float16 if florence else torch.bfloat16
    task = "<CAPTION>" if detail == "minimum" else "<DETAILED_CAPTION>"
    processor = model = None
    captions = []
    try:
        if florence:
            from transformers import AutoModelForCausalLM
            model_class = AutoModelForCausalLM
            attention = "eager"
        else:
            from transformers import Qwen3VLForConditionalGeneration
            model_class = Qwen3VLForConditionalGeneration
            attention = "sdpa"

        processor = AutoProcessor.from_pretrained(
            MODELS[model_name], trust_remote_code=florence,
        )
        if not florence:
            # shortcut: limit Qwen image tensors to 512² pixels; raise for text-heavy images.
            processor.image_processor.size["longest_edge"] = 512 * 512
        model = model_class.from_pretrained(
            MODELS[model_name], dtype=dtype, attn_implementation=attention,
            trust_remote_code=florence,
        ).to("cuda").eval()

        for source in images:
            image = _load_image(source)
            with torch.inference_mode():
                if florence:
                    inputs = processor(text=task, images=image, return_tensors="pt").to("cuda", dtype)
                    output = model.generate(
                        **inputs, max_new_tokens=128, num_beams=3,
                        do_sample=False, use_cache=False,
                    )
                    decoded = processor.batch_decode(output, skip_special_tokens=False)[0]
                    caption = processor.post_process_generation(
                        decoded, task=task, image_size=image.size,
                    )[task]
                else:
                    messages = [{"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": (
                            "Describe this image in one short sentence."
                            if detail == "minimum" else
                            "Describe this image in one detailed sentence, focusing on the main objects and their visual appearance."
                        )},
                    ]}]
                    inputs = processor.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True,
                        return_dict=True, return_tensors="pt",
                    ).to("cuda")
                    output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
                    caption = processor.decode(
                        output[0, inputs["input_ids"].shape[-1]:],
                        skip_special_tokens=True,
                    )
            torch.cuda.synchronize()
            captions.append(str(caption).strip())
            del image, inputs, output
        return captions
    finally:
        processor = model = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default="florence-2-base", choices=MODELS)
    parser.add_argument("--detail", choices=("minimum", "detailed"), default="minimum")
    args = parser.parse_args()
    started = time.perf_counter()
    captions = generate_captions(IMAGES, args.detail, args.model)
    for path, caption in zip(IMAGES, captions):
        print(f"{path.name}: {caption}")
    print(f"Captioned {len(captions)} images in {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
