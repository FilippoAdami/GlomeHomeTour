"""Adapters for the official cswry OSEDiff and SeeSR inference implementations."""
import importlib
import os
import sys
from types import SimpleNamespace

import torch
from PIL import Image
from torchvision import transforms


def load_model(name, artifacts_root, package_root):
    seesr = name.startswith('seesr')
    repo = package_root/'vendor/SeeSR' if seesr else artifacts_root/'vendor/OSEDiff'
    sys.path.insert(0, str(repo))
    # Upstream RAM resolves some assets relative to the repository cwd.
    os.chdir(repo)
    turbo = name.startswith('seesr_turbo')
    turbo_4step = name == 'seesr_turbo_4step'
    common = dict(mixed_precision='fp16', vae_encoder_tiled_size=512,
                  vae_decoder_tiled_size=64,
                  latent_tiled_size=96,
                  latent_tiled_overlap=32 if name == 'osediff' else 4,
                  process_size=512, upscale=1, align_method='adain', prompt='')
    if name == 'osediff':
        module = importlib.import_module('osediff')
        args = SimpleNamespace(**common,
            pretrained_model_name_or_path=str(artifacts_root/'models/sd21'),
            osediff_path=str(repo/'preset/models/osediff.pkl'),
            merge_and_unload_lora=False)
        model = module.OSEDiff_test(args).eval()
        ram_module = importlib.import_module('ram.models.ram_lora')
        # Replace upstream's developer-machine tokenizer path with the same public tokenizer.
        from transformers import BertTokenizer
        def init_tokenizer():
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
            tokenizer.add_special_tokens({'bos_token': '[DEC]'})
            tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
            tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
            return tokenizer
        ram_module.init_tokenizer = init_tokenizer
        tagger = ram_module.ram(pretrained=str(package_root/'models/RAM/ram_swin_large_14m.pth'),
                               pretrained_condition=str(repo/'preset/models/DAPE.pth'),
                               image_size=384, vit='swin_l').eval().to('cuda', dtype=torch.float16)
        from ram import inference_ram
        from my_utils.wavelet_color_fix import adain_color_fix
        def restore(image, seed):
            torch.manual_seed(seed)
            prepared = prepare(image)
            tensor = transforms.ToTensor()(prepared)[None].cuda()
            tagged = transforms.functional.normalize(
                transforms.functional.resize(tensor, [384,384]),
                [0.485,0.456,0.406], [0.229,0.224,0.225]).half()
            prompt = inference_ram(tagged, tagger)[0] + ', ,'
            output = model(tensor*2-1, prompt=prompt)
            output = transforms.ToPILImage()((output[0].float().cpu()*0.5+0.5).clamp(0,1))
            return adain_color_fix(output, prepared).resize(image.size, Image.Resampling.LANCZOS)
        return restore
    # Upstream SeeSR imports the old location of this unchanged module.
    sys.modules['diffusers.pipeline_utils'] = importlib.import_module('diffusers.pipelines.pipeline_utils')
    module = importlib.import_module('test_seesr_turbo' if turbo else 'test_seesr')
    if turbo:
        # sd-turbo omits the unused safety-checker image processor.
        module.CLIPImageProcessor = SimpleNamespace(from_pretrained=lambda path: None)
    args = SimpleNamespace(**common,
        pretrained_model_path=str(package_root/'models/sd-turbo') if turbo else str(artifacts_root/'models/sd2'),
        seesr_model_path=str(package_root/'models/SeeSR/seesr'),
        ram_ft_path=str(package_root/'models/SeeSR/DAPE.pth'),
        guidance_scale=1.0 if turbo else 5.5, conditioning_scale=1.0,
        blending_alpha=1.0, start_point='lr', start_steps=999,
        num_inference_steps=(2 if name.endswith('2step') else 4) if turbo else 50)
    pipeline = module.load_seesr_pipeline(args, SimpleNamespace(mixed_precision='fp16', device=torch.device('cuda')), False)
    if not turbo_4step:
        pipeline.controlnet.set_attention_slice('max')
        pipeline.unet.set_attention_slice('max')
    # Keep RAM on the GPU only when the device and allocator have safe headroom.
    from ram.models.ram_lora import ram
    ram_path = package_root/'models/RAM/ram_swin_large_14m.pth'
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    tagger_device = 'cpu'
    tagger_precision = torch.float32
    tagger_bytes = ram_path.stat().st_size//2 + 512*1024**2
    memory_budget = int(total_bytes*0.7)
    if (total_bytes >= 14*1024**3 and free_bytes >= tagger_bytes+2*1024**3 and
            torch.cuda.memory_allocated()+tagger_bytes+2*1024**3 <= memory_budget):
        tagger_device = 'cuda'
        tagger_precision = torch.float16
    tagger = ram(pretrained=str(ram_path), pretrained_condition=args.ram_ft_path,
                 image_size=384, vit='swin_l').eval().to(tagger_device,dtype=tagger_precision)
    default_steps = 3 if turbo else args.num_inference_steps
    def restore(image, seed, num_inference_steps=default_steps):
        torch.manual_seed(seed)
        if tagger_device == 'cuda':
            with torch.autocast('cuda',dtype=torch.float16):
                prompt, embeds = module.get_validation_prompt(args,image,tagger,device='cuda')
        else:
            prompt, embeds = module.get_validation_prompt(args,image,tagger,device='cpu')
        embeds = embeds.to('cuda',dtype=torch.float16)
        prepared = prepare(image)
        with torch.autocast('cuda', dtype=torch.float16):
            output = pipeline(prompt + 'clean, high-resolution, 8k', prepared,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(device='cuda').manual_seed(seed),
                height=prepared.height, width=prepared.width,
                guidance_scale=args.guidance_scale,
                negative_prompt='dotted, noise, blur, lowres, smooth',
                conditioning_scale=1.0, start_point='lr',
                ram_encoder_hidden_states=embeds,
                latent_tiled_size=args.latent_tiled_size,
                latent_tiled_overlap=args.latent_tiled_overlap, args=args).images[0]
        return module.adain_color_fix(output, prepared).resize(image.size, Image.Resampling.LANCZOS)
    restore.runtime_metadata = dict(tagger_device=tagger_device,
        tagger_precision='fp16' if tagger_device=='cuda' else 'fp32',
        attention_slicing=None if turbo_4step else 'max',
        latent_tiled_size=args.latent_tiled_size)
    return restore


def prepare(image):
    scale = max(1, 512/min(image.size))
    width, height = [max(8, int(s*scale)//8*8) for s in image.size]
    return image.resize((width,height), Image.Resampling.LANCZOS)
