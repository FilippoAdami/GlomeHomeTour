from typing import *
import os
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image


class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        # The shipped config points at gated briaai/RMBG-2.0; default to open ZhengPeng7/BiRefNet.
        env_model = os.environ.get("REMBG_MODEL")
        if env_model:
            model_name = env_model
        elif model_name == "briaai/RMBG-2.0":
            model_name = "ZhengPeng7/BiRefNet"
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    def to(self, device: str):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        self.model.cuda()
        input_images = self.transform_image(image).unsqueeze(0).to(
            "cuda", dtype=next(self.model.parameters()).dtype,
        )
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        self.model.cpu()
        torch.cuda.empty_cache()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
    
