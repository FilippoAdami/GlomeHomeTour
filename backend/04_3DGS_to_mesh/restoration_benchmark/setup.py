"""Prepare project-local sources, isolated dependencies, weights and blur 13."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
PACKAGE=ROOT/'backend/04_3DGS_to_mesh'
ARTIFACTS=ROOT/'backend/current_scene/04_3DGS_to_mesh'
PACKAGES=['diffusers==0.25.0','transformers==4.38.2','huggingface-hub==0.25.2',
          'peft==0.9.0','accelerate==0.28.0','tokenizers==0.15.2','timm==0.9.16',
          'lpips==0.1.4','loralib==0.1.2','fairscale==0.4.13']


def prepare_seesr_env():
    env=PACKAGE/'env_seesr'
    subprocess.run([str(ROOT/'backend/.venv/bin/python'),'-m','venv','--system-site-packages',str(env)],check=True)
    version=f'python{sys.version_info.major}.{sys.version_info.minor}'
    site=env/'lib'/version/'site-packages'
    paths=[PACKAGE/'env/lib'/version/'site-packages',ROOT/'backend/.venv/lib'/version/'site-packages']
    (site/'restoration_environment.pth').write_text(''.join(str(p)+'\n' for p in paths))
    subprocess.run([str(env/'bin/pip'),'install','--no-deps','diffusers==0.21.0'],check=True)


def prepare_blur():
    from PIL import Image,ImageFilter
    import numpy as np
    source=ROOT/'backend/current_scene/04_3DGS_to_mesh'
    target=source/'images blurred/blur_13'
    images=sorted((source/'images blurred/blur_3').glob('*'))
    # Prove the existing degradation recipe before extending the series.
    for file in images:
        for level in (3,5,7):
            with Image.open(source/'images'/file.name) as image:
                buf=io.BytesIO()
                image.convert('RGB').filter(ImageFilter.GaussianBlur(level)).save(buf,format='JPEG',quality=75)
            with Image.open(source/'images blurred'/f'blur_{level}'/file.name) as existing:
                if not np.array_equal(np.asarray(Image.open(buf)),np.asarray(existing)):
                    raise ValueError(f'Cannot reproduce existing blur: {file.name} radius={level}')
    target.mkdir(exist_ok=True)
    for file in images:
        destination=target/file.name
        with Image.open(source/'images'/file.name) as image:
            buf=io.BytesIO()
            image.convert('RGB').filter(ImageFilter.GaussianBlur(13)).save(buf,format='JPEG',quality=75)
        if destination.exists() and destination.read_bytes()!=buf.getvalue():
            raise FileExistsError(f'Refusing to overwrite differing input: {destination}')
        destination.write_bytes(buf.getvalue())
    (ARTIFACTS/'blur_provenance.json').write_text(json.dumps(dict(
        method='Pillow ImageFilter.GaussianBlur(radius), RGB, JPEG quality 75',
        checked_existing_images=len(images)*3,created_radius=13,
        input=str(source/'images'),output=str(target)),indent=2)+'\n')


def download():
    from huggingface_hub import snapshot_download
    from huggingface_hub import constants
    from download_asset import download_asset
    # Large transfer pools exceed this environment's connection permits.
    constants.HF_TRANSFER_CONCURRENCY=3
    jobs=[('Manojb/stable-diffusion-2-1-base','sd21',['*.json','tokenizer/*','text_encoder/model.fp16.safetensors','vae/diffusion_pytorch_model.fp16.safetensors','unet/diffusion_pytorch_model.fp16.safetensors']),
          ('cswry/SeeSR','SeeSR',['DAPE.pth','seesr/*/*.json','seesr/scaler.pt']),
          ('stabilityai/sd-turbo','sd-turbo',['*.json','tokenizer/*','text_encoder/model.fp16.safetensors','vae/diffusion_pytorch_model.fp16.safetensors','unet/diffusion_pytorch_model.fp16.safetensors']),
          ('Manojb/stable-diffusion-2-base','sd2',['*.json','tokenizer/*'])]
    destinations={'sd21':ARTIFACTS/'models/sd21','SeeSR':PACKAGE/'models/SeeSR',
                  'sd-turbo':PACKAGE/'models/sd-turbo','sd2':ARTIFACTS/'models/sd2'}
    def fetch(job):
        repo,name,patterns=job
        print('Downloading',repo,flush=True)
        try:
            snapshot_download(repo,local_dir=destinations[name],allow_patterns=patterns,max_workers=1)
            if name=='SeeSR':
                download_asset('seesr_control')
                download_asset('seesr_unet')
        except Exception as error:
            message=re.sub(r'https?://[^\s)]+','[download URL]',str(error))
            print(f'Download failed for {repo}: {message}',flush=True)
            raise RuntimeError(f'Download failed for {repo}: {message}') from None
        for source in destinations[name].rglob('*.fp16.safetensors'):
            # Upstream loaders expect canonical filenames; hardlinks avoid duplicate weights.
            destination=source.with_name(source.name.replace('.fp16',''))
            if not destination.exists(): os.link(source,destination)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch,jobs))
    # SD2 and SD2.1 publish byte-identical frozen text encoder and VAE weights.
    for component, filename, expected in [
        ('text_encoder','model','681c555376658c81dc273f2d737a2aeb23ddb6d1d8e5b3a7064636d359a22668'),
        ('vae','diffusion_pytorch_model','3e4c08995484ee61270175e9e7a072b66a6e4eeb5f0c266667fe1f45b90daf9a')]:
        source=ARTIFACTS/'models/sd21'/component/f'{filename}.fp16.safetensors'
        if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Unexpected source weights: {source}')
        destination=ARTIFACTS/'models/sd2'/component/f'{filename}.safetensors'
        if not destination.exists(): os.link(source,destination)
    download_asset('ram')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-only',action='store_true')
    parser.add_argument('--blur-only',action='store_true')
    args=parser.parse_args()
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    if args.blur_only:
        prepare_blur();return
    if not args.download_only:
        for repo in ('OSEDiff','SeeSR'):
            destination=ARTIFACTS/'vendor'/repo if repo=='OSEDiff' else PACKAGE/'vendor'/repo
            if not destination.exists(): subprocess.run(['git','clone','--depth','1',f'https://github.com/cswry/{repo}.git',str(destination)],check=True)
        env=PACKAGE/'env'
        subprocess.run([str(ROOT/'backend/.venv/bin/python'),'-m','venv','--system-site-packages',str(env)],check=True)
        site=env/f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
        (site/'backend_environment.pth').write_text(str(ROOT/f'backend/.venv/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages')+'\n')
        subprocess.run([str(env/'bin/pip'),'install','--no-deps',*PACKAGES],check=True)
        prepare_seesr_env()
        prepare_blur()
    download()


if __name__=='__main__': main()
