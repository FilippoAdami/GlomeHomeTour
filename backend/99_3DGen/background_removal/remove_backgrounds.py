"""Remove image backgrounds in one model load and return RGBA PIL images."""

import gc
import os
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
IMAGES = sorted((ROOT / "images").glob("*.jpg"))
MODEL = "ZhengPeng7/BiRefNet_lite"
REVISION = "aa62cd87eafb9cc43056d08ef3615a14628b831d"


def _load_image(image):
    if isinstance(image, Image.Image):
        return image.copy()
    with Image.open(image) as source:
        return source.copy()


def remove_backgrounds(images):
    """Return transparent RGBA images, reusing one BiRefNet model for the batch."""
    if isinstance(images, (str, os.PathLike, Image.Image)):
        raise TypeError("images must be a list of image paths or PIL images")
    images = [_load_image(image) for image in images]
    if not images:
        return []

    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    if not torch.cuda.is_available():
        raise RuntimeError("Background removal requires a ROCm or CUDA GPU")

    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    results = []
    model = None
    try:
        model = AutoModelForImageSegmentation.from_pretrained(
            MODEL, revision=REVISION, trust_remote_code=True,
        ).eval().half().to("cuda")
        for image in images:
            rgba = image.convert("RGBA")
            if rgba.getchannel("A").getextrema()[0] < 255:
                results.append(rgba)
                continue
            rgb = image.convert("RGB")
            tensor = transform(rgb).unsqueeze(0).to("cuda", dtype=torch.float16)
            with torch.inference_mode():
                mask = model(tensor)[-1].sigmoid()[0].squeeze().float().cpu()
            alpha = transforms.ToPILImage()(mask).resize(rgb.size, Image.Resampling.BILINEAR)
            rgb.putalpha(alpha)
            if alpha.getbbox() is None:
                raise RuntimeError("Background removal returned an empty mask")
            results.append(rgb)
            del tensor, mask, alpha
        torch.cuda.synchronize()
        return results
    finally:
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    import time

    started = time.perf_counter()
    outputs = remove_backgrounds(IMAGES)
    for path, image in zip(IMAGES, outputs):
        output = Path(__file__).resolve().parent / f"{path.stem}.png"
        image.save(output, format="PNG")
        print(output)
    print(f"Processed {len(outputs)} images in {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
