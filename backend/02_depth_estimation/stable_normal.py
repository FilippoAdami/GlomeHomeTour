"""StableNormal Turbo Monocular Surface Normal Estimator for GlomeHomeTour.

Wraps the high-speed StableNormal Turbo model (Stable-X/StableNormal) with strict GPU
memory lifecycle management, ensuring full release of VRAM after inference.
"""

from __future__ import annotations

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("MIOPEN_USER_DB_PATH", "/tmp/miopen")
os.makedirs("/tmp/miopen", exist_ok=True)

import gc
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

# Diffusers compatibility shim for diffusers >= 0.31 where
# diffusers.models.controlnet was moved to diffusers.models.controlnets.controlnet
try:
    import diffusers.models.controlnets.controlnet as _cnet
    sys.modules.setdefault("diffusers.models.controlnet", _cnet)
except Exception:
    pass


class StableNormalEstimator:
    """Estimates dense unit surface normals from RGB images using StableNormal Turbo.

    Enforces strict GPU lifecycle management:
    - Lazy loading onto GPU on demand or via context manager.
    - Explicit `unload()` method purging model weights and clearing CUDA cache.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        resolution: int = 768,
        yoso_version: str = "yoso-normal-v0-3",
        data_type: str = "indoor",
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.resolution = resolution
        self.yoso_version = yoso_version
        self.data_type = data_type
        self._predictor = None

    def load(self) -> None:
        """Load StableNormal Turbo model onto device."""
        if self._predictor is not None:
            return

        print(f"[StableNormal] Loading StableNormal_turbo ({self.yoso_version}) onto {self.device}...", flush=True)
        self._predictor = torch.hub.load(
            "Stable-X/StableNormal",
            "StableNormal_turbo",
            yoso_version=self.yoso_version,
            device=self.device,
            trust_repo=True,
        )

        # Optimization: Precompute text embeddings and purge CLIP text encoder & tokenizer
        try:
            pipe = self._predictor.model
            device = self.device
            if pipe.empty_text_embedding is None and hasattr(pipe, "tokenizer") and hasattr(pipe, "text_encoder"):
                text_inputs = pipe.tokenizer(
                    "",
                    padding="do_not_pad",
                    max_length=pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(device)
                pipe.empty_text_embedding = pipe.text_encoder(text_input_ids)[0]

            if pipe.prompt_embeds is None and hasattr(pipe, "encode_prompt"):
                p_embeds, neg_embeds = pipe.encode_prompt(
                    pipe.prompt, device, 1, False, None
                )
                pipe.prompt_embeds = p_embeds
                pipe.negative_prompt_embeds = neg_embeds

            if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
                del pipe.text_encoder
                pipe.text_encoder = None
            if hasattr(pipe, "tokenizer") and pipe.tokenizer is not None:
                del pipe.tokenizer
                pipe.tokenizer = None

            # GPU Kernel Warm-up to pre-compile JIT graph
            if self.device != "cpu":
                dummy = Image.new("RGB", (self.resolution, self.resolution), (128, 128, 128))
                with torch.inference_mode():
                    _ = pipe(dummy, processing_resolution=self.resolution, match_input_resolution=False, output_type="pt")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[StableNormal] Note: VRAM optimization step skipped: {e}", flush=True)

        print("[StableNormal] Model successfully loaded and memory-optimized.", flush=True)

    def unload(self) -> None:
        """Completely unload model from GPU memory and reclaim VRAM."""
        if self._predictor is not None:
            del self._predictor
            self._predictor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[StableNormal] Model unloaded from GPU. VRAM reclaimed.", flush=True)

    def __enter__(self) -> "StableNormalEstimator":
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.unload()

    def estimate_normal(
        self,
        image: Union[np.ndarray, Image.Image],
    ) -> np.ndarray:
        """Estimate dense unit surface normals (H, W, 3) in camera coordinates.

        Args:
            image: RGB image as numpy uint8 array (H, W, 3) or PIL Image.

        Returns:
            normals: (H, W, 3) float32 unit vectors in camera coordinates.
                     Range is [-1, 1], with unit length ||n|| = 1.0.
        """
        if self._predictor is None:
            self.load()

        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(image)
        else:
            pil_img = image

        orig_w, orig_h = pil_img.size
        pipe = self._predictor.model

        # StableNormal inference directly producing GPU float32 tensor
        with torch.inference_mode():
            pipe_out = pipe(
                pil_img,
                match_input_resolution=False,
                processing_resolution=self.resolution,
                output_type="pt",
            )
            normal_tensor = pipe_out.prediction  # Shape (1, 3, PH, PW) on GPU in [-1, 1]

            # High-speed GPU bilinear upsampling to original frame resolution (if needed)
            cur_h, cur_w = normal_tensor.shape[-2:]
            if (cur_w, cur_h) != (orig_w, orig_h):
                normal_tensor = torch.nn.functional.interpolate(
                    normal_tensor,
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
                )

            # Normalize to exact unit vectors on GPU
            norm = torch.norm(normal_tensor, dim=1, keepdim=True).clamp(min=1e-8)
            normal_tensor = normal_tensor / norm

            # Convert to numpy (H, W, 3) float32
            normal_unit = (
                normal_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
            )

        return normal_unit
