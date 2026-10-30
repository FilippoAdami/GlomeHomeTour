"""Four-level restoration benchmark. Run with the isolated ROCm environment; see README."""
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import patch

from PIL import Image, ImageFilter

LEVELS = (3, 5, 7, 13)
MODELS = ('osediff', 'seesr_standard', 'seesr_turbo_2step', 'seesr_turbo_4step')
REPO = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO/'backend/04_3DGS_to_mesh'
ARTIFACTS = REPO/'backend/current_scene/04_3DGS_to_mesh'
INPUTS = REPO/'backend/current_scene/04_3DGS_to_mesh/images blurred'
ORIGINALS = INPUTS.parent/'images'
SEESR_ENV = PACKAGE_ROOT/'env_seesr'


def dump(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(inputs, originals):
    files = sorted(p for p in (inputs/'blur_3').iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})
    if len(files) < 10:
        raise ValueError('At least ten images required in blur_3')
    # Evenly spaced over lexicographically sorted names; all ten when exactly ten exist.
    selected = [files[round(i*(len(files)-1)/9)] for i in range(10)]
    records = []
    for source in selected:
        versions = [inputs/f'blur_{level}'/source.name for level in LEVELS]
        for path in versions:
            if not path.is_file():
                raise FileNotFoundError(path)
        sizes = []
        for path in versions:
            with Image.open(path) as image:
                image.load()
                sizes.append(image.size)
        if len(set(sizes)) != 1:
            raise ValueError(f'Dimension mismatch for {source.name}: {sizes}')
        truth = originals/source.name
        if truth.exists():
            with Image.open(truth) as image:
                if image.size != sizes[0]:
                    raise ValueError(f'Ground truth dimensions differ: {truth}')
        records.append(dict(filename=source.name, dimensions=sizes[0],
            inputs={str(level):dict(path=str(path.resolve()),sha256=sha(path)) for level,path in zip(LEVELS,versions)},
            ground_truth=dict(path=str(truth.resolve()),sha256=sha(truth)) if truth.exists() else None))
    return records


class Trace:
    def __init__(self, path):
        self.path, self.rows, self.stop = path, [], threading.Event()
        self.phase = 'load'
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self.sample, daemon=True)
    def sample(self):
        import psutil
        process = psutil.Process()
        devices = list(Path('/sys/class/drm').glob('card*/device/mem_info_vram_used'))
        while not self.stop.is_set():
            row = dict(elapsed_s=time.perf_counter()-self.started, phase=self.phase,
                       rss_bytes=process.memory_info().rss, cpu_percent=process.cpu_percent())
            for path in devices:
                try:
                    row[path.parents[1].name+'_device_vram_bytes'] = int(path.read_text())
                    busy=path.parent/'gpu_busy_percent'
                    if busy.exists(): row[path.parents[1].name+'_busy_percent']=int(busy.read_text())
                except OSError:
                    pass
            self.rows.append(row)
            self.stop.wait(0.2)
    def __enter__(self):
        self.thread.start()
        return self
    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        write_csv(self.path, self.rows)


def amortized(rows, load, unload):
    overhead = load+unload
    return dict(overhead_s=overhead,
        amortized_40_s=statistics.mean(r['processing_s'] for r in rows)+overhead/40,
        projected_10_s=statistics.mean(r['processing_s'] for r in rows)+overhead/10)


def worker(args):
    os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = (
        '1' if args.model == 'seesr_turbo_4step' else '0')
    import torch
    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(0.7)
    from model_adapters import load_model
    torch.manual_seed(42)
    root, run = Path(args.artifacts).resolve(), Path(args.run).resolve()
    manifest = json.loads(Path(args.manifest).read_text())
    rows=[]
    benchmark=dict(model=args.model, status='running', levels=list(LEVELS),
        expected_images=40, seed=42, upscale=1, precision='fp16',
        tagger_precision='fp16' if args.model=='osediff' else 'fp32',
        tagger_device='cuda' if args.model=='osediff' else 'cpu',
        gpu_allocator_fraction=0.7,
        attention_slicing=None if args.model in ('osediff','seesr_turbo_4step') else 'max',
        software={name:importlib.metadata.version(name) for name in ('diffusers','transformers','peft','accelerate','torchvision','Pillow')},
        steps=1 if args.model=='osediff' else 50 if args.model=='seesr_standard' else 2 if args.model.endswith('2step') else 4,
        color_correction='upstream AdaIN', base_weights='published FP16 variants',
        base_repository=('Manojb/stable-diffusion-2-1-base' if args.model=='osediff' else
                         'Manojb/stable-diffusion-2-base' if args.model=='seesr_standard' else 'stabilityai/sd-turbo'),
        selection_manifest=str(Path(args.manifest).resolve()), selection_sha256=sha(Path(args.manifest)),
        torch=torch.__version__, hip=torch.version.hip,
        rocm_aotriton_experimental=os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'],
        vae_encoder_tile_size=512, vae_decoder_tile_size=64,
        latent_tiled_size=96,
        gpu=torch.cuda.get_device_name(0),
        timing='GPU synchronized; inference includes preprocessing, RAM tagging, restoration, color fix and CPU transfer; processing adds image decode and lossless save. Quality evaluation excluded.',
        output_encoding='PNG content with original filenames, including .jpg names, to avoid JPEG quality loss',
        warmup='One blur_3 restoration per distinct input resolution; reported separately from inference statistics and load/unload amortization',
        upstream={name:subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
                  for name,repo in (('OSEDiff',root/'vendor/OSEDiff'),('SeeSR',PACKAGE_ROOT/'vendor/SeeSR'))})
    restore=None
    session_start=time.perf_counter()
    try:
        with Trace(run/'resource_trace.csv') as trace:
            start=time.perf_counter()
            restore=load_model(args.model,root,PACKAGE_ROOT)
            benchmark.update(getattr(restore,'runtime_metadata',{}))
            restore_options = {'num_inference_steps': 2 if args.model.endswith('2step') else 4} if args.model.startswith('seesr_turbo') else {}
            torch.cuda.synchronize()
            benchmark['load_s']=time.perf_counter()-start
            trace.phase='warmup'
            start=time.perf_counter()
            warmed=set()
            for index,record in enumerate(manifest):
                dimensions=tuple(record['dimensions'])
                if dimensions in warmed:
                    continue
                with Image.open(record['inputs']['3']['path']) as image, torch.inference_mode():
                    restore(image.convert('RGB'),42+index,**restore_options)
                warmed.add(dimensions)
            torch.cuda.synchronize()
            benchmark['warmup_s']=time.perf_counter()-start
            benchmark['warmup_resolutions']=list(map(list,sorted(warmed)))
            for level in LEVELS:
                trace.phase=f'blur_{level}'
                for index,record in enumerate(manifest):
                    item=dict(model=args.model,blur=level,filename=record['filename'],status='failed')
                    try:
                        torch.cuda.reset_peak_memory_stats()
                        processing=time.perf_counter()
                        with Image.open(record['inputs'][str(level)]['path']) as image:
                            image=image.convert('RGB')
                        torch.cuda.synchronize()
                        start=time.perf_counter()
                        # SeeSR's tiled VAE resets these counters internally; retain the whole-image peak.
                        with torch.inference_mode(), patch.object(torch.cuda, 'reset_peak_memory_stats', lambda *a, **k: None):
                            output=restore(image,42+index,**restore_options)
                        torch.cuda.synchronize()
                        item['inference_s']=time.perf_counter()-start
                        item['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
                        item['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
                        if output.size != image.size:
                            raise ValueError(f'Output dimensions {output.size} != {image.size}')
                        output.save(run/f'blur_{level}'/record['filename'],format='PNG')
                        item['processing_s']=time.perf_counter()-processing
                        item['status']='ok'
                        print(f"{args.model} blur_{level} {record['filename']}: {item['inference_s']:.3f}s",flush=True)
                    except torch.OutOfMemoryError:
                        raise
                    except Exception:
                        item['error']=traceback.format_exc()
                        torch.cuda.empty_cache()
                    rows.append(item)
                    write_csv(run/'per_image_metrics.csv',rows)
            trace.phase='unload'
            start=time.perf_counter()
            del restore
            restore=None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            benchmark['unload_s']=time.perf_counter()-start
        benchmark['session_wall_s']=time.perf_counter()-session_start
        benchmark['status']='complete' if len(rows)==40 and all(r['status']=='ok' for r in rows) else 'partial'
        benchmark['per_level']={}
        for level in LEVELS:
            good=[r for r in rows if r['blur']==level and r['status']=='ok']
            if good:
                benchmark['per_level'][str(level)] = dict(count=len(good),
                    inference_mean_s=statistics.mean(r['inference_s'] for r in good),
                    inference_median_s=statistics.median(r['inference_s'] for r in good),
                    peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in good),
                    peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in good),
                    **(amortized(good,benchmark['load_s'],benchmark['unload_s'])
                       if benchmark['status']=='complete' else {}))
                device_fields={key for row in trace.rows for key in row if key.endswith('_device_vram_bytes')}
                if len(device_fields)==1:
                    field=device_fields.pop()
                    samples=[row[field] for row in trace.rows if row['phase']==f'blur_{level}' and field in row]
                    if samples:
                        benchmark['per_level'][str(level)]['sampled_peak_device_vram_bytes']=max(samples)
        if benchmark['status']=='complete':
            benchmark['measured_session_wall_s_per_image']=benchmark['session_wall_s']/40
    except Exception:
        benchmark.update(status='blocked',error=traceback.format_exc())
    finally:
        if restore is not None:
            del restore
            gc.collect()
            torch.cuda.empty_cache()
        dump(run/'benchmark.json',benchmark)
    return 0 if benchmark['status']=='complete' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',default=str(ARTIFACTS))
    parser.add_argument('--inputs',default=str(INPUTS))
    parser.add_argument('--originals',default=str(ORIGINALS))
    parser.add_argument('--model',choices=MODELS)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--run')
    parser.add_argument('--manifest')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.worker:
        return worker(args)
    root=Path(args.artifacts).resolve()
    out=root/'images_restored'
    out.mkdir(parents=True,exist_ok=True)
    manifest=select(Path(args.inputs).resolve(),Path(args.originals).resolve())
    stamp=datetime.now(ZoneInfo('Europe/Rome')).strftime('%Y%m%d_%H%M%S_%f')
    manifest_path=out/f'selection_{stamp}.json'
    dump(manifest_path,manifest)
    if args.prepare_only:
        print(manifest_path)
        return 0
    runs=[]
    for name in ([args.model] if args.model else MODELS):
        run=out/f'{name}_{stamp}'
        run.mkdir()
        for level in LEVELS: (run/f'blur_{level}').mkdir()
        runs.append(str(run))
        python=str(SEESR_ENV/'bin/python') if name.startswith('seesr') else sys.executable
        with (run/'execution.log').open('w') as log:
            result=subprocess.run([python,str(Path(__file__).resolve()),'--worker',
                '--model',name,'--artifacts',str(root),'--run',str(run),
                '--manifest',str(manifest_path)],stdout=log,stderr=subprocess.STDOUT)
        if not (run/'benchmark.json').exists():
            dump(run/'benchmark.json',dict(model=name,status='blocked',returncode=result.returncode,error='Worker terminated; see execution.log'))
        print(name,json.loads((run/'benchmark.json').read_text())['status'],flush=True)
    dump(out/f'batch_{stamp}.json',dict(manifest=str(manifest_path),runs=runs))
    return 0 if all(json.loads((Path(r)/'benchmark.json').read_text())['status']=='complete' for r in runs) else 1


if __name__=='__main__':
    sys.exit(main())
