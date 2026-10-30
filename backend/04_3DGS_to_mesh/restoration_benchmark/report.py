"""Evaluate saved images after models unload; generate paired comparisons and plots."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import lpips

from run import LEVELS, MODELS, amortized, dump, write_csv


def metrics(image, reference=None, perceptual=None):
    rgb=np.asarray(image.convert('RGB'))
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    result=dict(sharpness_laplacian=float(cv2.Laplacian(gray,cv2.CV_64F).var()))
    if reference is None:
        return result
    truth=np.asarray(reference.convert('RGB'))
    result['psnr']=float(peak_signal_noise_ratio(truth,rgb,data_range=255)) if not np.array_equal(truth,rgb) else 100.0
    result['ssim']=float(structural_similarity(truth,rgb,channel_axis=2,data_range=255))
    result['color_mean_error']=float(np.linalg.norm(rgb.mean((0,1))-truth.mean((0,1)))/255)
    edges=cv2.Canny(gray,80,160)>0
    ref_edges=cv2.Canny(cv2.cvtColor(truth,cv2.COLOR_RGB2GRAY),80,160)>0
    kernel=np.ones((3,3),np.uint8)
    nearby=cv2.dilate(ref_edges.astype(np.uint8),kernel)>0
    nearby_out=cv2.dilate(edges.astype(np.uint8),kernel)>0
    result['edge_precision']=float((edges&nearby).sum()/max(1,edges.sum()))
    result['edge_recall']=float((ref_edges&nearby_out).sum()/max(1,ref_edges.sum()))
    result['edge_f1']=2*result['edge_precision']*result['edge_recall']/max(1e-12,result['edge_precision']+result['edge_recall'])
    if perceptual is not None:
        # Native resolution, RGB in [-1,1]; metric model runs after restoration.
        tensors=[torch.from_numpy(a.copy()).permute(2,0,1)[None].float()/127.5-1 for a in (rgb,truth)]
        with torch.inference_mode(): result['lpips']=float(perceptual(*tensors).item())
    return result


def panel(path, entries, crop=False):
    width,height=320,360
    canvas=Image.new('RGB',(width*len(entries),height+40),'#202020')
    draw=ImageDraw.Draw(canvas)
    for index,(label,file) in enumerate(entries):
        draw.text((index*width+8,8),label,fill='white')
        if file and Path(file).exists():
            with Image.open(file) as image:
                image=image.convert('RGB')
                if crop:
                    w,h=image.size
                    image=image.crop(((w-256)//2,(h-256)//2,(w+256)//2,(h+256)//2))
                image=ImageOps.contain(image,(width,height))
                canvas.paste(image,(index*width+(width-image.width)//2,40+(height-image.height)//2))
        else:
            draw.text((index*width+8,80),'Unavailable',fill='white')
    canvas.save(path)


def report(batch_path):
    batch=json.loads(batch_path.read_text())
    manifest=json.loads(Path(batch['manifest']).read_text())
    out=batch_path.parent
    comparison=out/'comparison'/batch_path.stem
    comparison.mkdir(parents=True,exist_ok=True)
    # CPU LPIPS keeps metric allocations out of restoration memory and timing.
    torch.set_num_threads(8)
    perceptual=lpips.LPIPS(net='alex').eval().cpu()
    runs={}
    summary=[]
    baseline=[]
    for level in LEVELS:
        for record in manifest:
            truth=Image.open(record['ground_truth']['path']).convert('RGB') if record['ground_truth'] else None
            with Image.open(record['inputs'][str(level)]['path']) as image:
                baseline.append(dict(blur=level,filename=record['filename'],**metrics(image,truth,perceptual)))
    write_csv(comparison/'input_metrics.csv',baseline)
    for path in map(Path,batch['runs']):
        bench=json.loads((path/'benchmark.json').read_text())
        name=bench['model']
        runs[name]=path
        rows=[]
        if (path/'per_image_metrics.csv').exists():
            with (path/'per_image_metrics.csv').open() as source:
                rows=list(csv.DictReader(source))
        for row in rows:
            if row['status']!='ok': continue
            record=next(r for r in manifest if r['filename']==row['filename'])
            truth=Image.open(record['ground_truth']['path']).convert('RGB') if record['ground_truth'] else None
            with Image.open(path/f"blur_{row['blur']}"/row['filename']) as image:
                row.update(metrics(image,truth,perceptual))
        write_csv(path/'per_image_metrics.csv',rows)
        for level in LEVELS:
            good=[r for r in rows if int(r['blur'])==level and r['status']=='ok']
            item=dict(model=name,blur=level,status='complete' if len(good)==10 else bench['status'],count=len(good),run=str(path))
            if good:
                measured=bench['per_level'][str(level)]
                if bench['status']=='complete':
                    measured.update(amortized([dict(processing_s=float(r['processing_s'])) for r in good],
                                              bench['load_s'],bench['unload_s']))
                item.update(measured)
                for key in ['psnr','ssim','lpips','sharpness_laplacian','color_mean_error','edge_precision','edge_recall','edge_f1']:
                    values=[float(r[key]) for r in good if key in r]
                    if values: item[key]=statistics.mean(values)
                for key in ['psnr','ssim','lpips','sharpness_laplacian']:
                    values=[b[key] for b in baseline if b['blur']==level and key in b]
                    if values: item['input_'+key]=statistics.mean(values)
            summary.append(item)
        bench['warmup']='One blur_3 restoration per distinct input resolution; reported separately from inference statistics and load/unload amortization'
        bench['quality_metrics']='per_image_metrics.csv; full-reference metrics use unblurred originals only'
        dump(path/'benchmark.json',bench)
    write_csv(out/'benchmark_summary.csv',summary)
    write_csv(comparison/'benchmark_summary.csv',summary)
    for record in manifest:
        stem=Path(record['filename']).stem
        for level in LEVELS:
            entries=[('Unblurred original',record['ground_truth']['path'] if record['ground_truth'] else None),
                     (f'Input blur {level}',record['inputs'][str(level)]['path'])]
            entries += [(name,runs[name]/f'blur_{level}'/record['filename'] if name in runs else None) for name in MODELS]
            for crop in (False,True):
                panel(comparison/f'{stem}_blur_{level}_{"crop" if crop else "full"}.png',entries,crop)
        for name,path in runs.items():
            entries=[('Original',record['ground_truth']['path'] if record['ground_truth'] else None)]
            entries += [(f'{name} blur {level}',path/f'blur_{level}'/record['filename']) for level in LEVELS]
            panel(comparison/f'{stem}_{name}_robustness.png',entries)
    plots(comparison,summary,baseline)
    lines=['# Restoration benchmark', '', f'Batch: `{batch_path.name}`', '',
           'Four blur radii: 3, 5, 7, 13. Same ten filenames at every level. Restoration at native dimensions (1×), FP16, seed 42 + image index.',
           'Outputs contain lossless PNG data under original filenames. Load/unload overhead is shared over 40 images; projected ten-image deployment uses overhead/10. Warmup is reported separately and excluded from these projections.', '',
           'VRAM is reported in GiB (2^30 bytes). Per-run allocator limits, tagging device and attention slicing are recorded in benchmark.json.', '',
           '| Model | Blur | Images | Mean inference s | 40-image amortized s/image | Projected 10-image s/image | PSNR | SSIM | LPIPS | Peak GPU allocation GiB |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for item in summary:
        values=[item['model'],str(item['blur']),str(item['count'])]
        values += [f'{item[k]:.4f}' if k in item else 'unavailable' for k in ('inference_mean_s','amortized_40_s','projected_10_s','psnr','ssim','lpips')]
        values.append(f"{item['peak_allocated_bytes']/2**30:.3f}" if 'peak_allocated_bytes' in item else 'unavailable')
        lines.append('| '+' | '.join(values)+' |')
    complete=[name for name in MODELS if len([r for r in summary if r['model']==name and r['count']==10])==4]
    lines += ['', '## Conclusions', '']
    if complete:
        aggregates={name:[s for s in summary if s['model']==name] for name in complete}
        fastest=min(complete,key=lambda n:statistics.mean(s['inference_mean_s'] for s in aggregates[n]))
        lines.append(f'Fastest over all four levels: **{fastest}** (equal image counts).')
        if all('lpips' in s for n in complete for s in aggregates[n]):
            groups=[('Mild blur (3)',(3,)),('Intermediate/severe blur (5–7)',(5,7)),('Extreme blur (13)',(13,))]
            for label,levels in groups:
                winner=min(complete,key=lambda n:statistics.mean(s['lpips'] for s in aggregates[n] if s['blur'] in levels))
                lines.append(f'{label}: **{winner}** has lowest mean LPIPS; inspect PSNR/SSIM and geometric comparisons before calling it visually best.')
            consistent=min(complete,key=lambda n:statistics.pstdev(s['lpips'] for s in aggregates[n]))
            lines.append(f'Most consistent LPIPS across levels: **{consistent}**. Low variation alone does not imply high fidelity.')
            pareto=[n for n in complete if not any(m!=n and
                statistics.mean(s['lpips'] for s in aggregates[m]) <= statistics.mean(s['lpips'] for s in aggregates[n]) and
                statistics.mean(s['amortized_40_s'] for s in aggregates[m]) <= statistics.mean(s['amortized_40_s'] for s in aggregates[n]) and
                (statistics.mean(s['lpips'] for s in aggregates[m]) < statistics.mean(s['lpips'] for s in aggregates[n]) or
                 statistics.mean(s['amortized_40_s'] for s in aggregates[m]) < statistics.mean(s['amortized_40_s'] for s in aggregates[n])) for m in complete)]
            lines.append('Quality/time Pareto candidates on this RX 9060 XT: **'+', '.join(pareto)+'**. A single preference requires visual fidelity review.')
    else:
        lines.append('No configuration completed all four levels; no valid overall model winner can be named.')
    for name in MODELS:
        path=runs.get(name)
        if path:
            b=json.loads((path/'benchmark.json').read_text())
            if b['status']!='complete': lines += ['',f"{name}: **{b['status']}**. See `{path.name}/benchmark.json` and execution log."]
    lines += ['', '## Quality interpretation', '',
        'Laplacian variance measures sharpness, not fidelity. PSNR/SSIM/LPIPS use corresponding unblurred originals. Edge F1 (Canny 80/160, 1-pixel tolerance) and mean RGB error flag geometric/color changes; neither proves hallucination or straight-line preservation.',
        'Full-image and center-crop comparisons are provided for every selected image and level. Manual review must identify invented textures, shifted boundaries, bent lines and repetitive patterns. An automatic sharpness score cannot establish a hallucination onset. No visual conclusion is inferred from unavailable outputs.',
        'Input blur radii 3/5/7 reproduce Pillow GaussianBlur(radius) saved at JPEG quality 75 exactly for all 30 source images; blur 13 uses the same recipe on originals.', '',
        f'Completion: {sum(s["count"] for s in summary)}/160 restored images. All four levels are required for each feasible configuration.']
    markdown='\n'.join(lines)+'\n'
    (comparison/'benchmark_summary.md').write_text(markdown)
    (out/'benchmark_summary.md').write_text(markdown)
    print(comparison/'benchmark_summary.md',flush=True)


def plots(directory,summary,baseline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key,label in [('inference_mean_s','Inference time (s/image)'),('peak_allocated_bytes','Peak allocated GPU memory (GiB)'),
                      ('peak_reserved_bytes','Peak reserved GPU memory (GiB)'),
                      ('sampled_peak_device_vram_bytes','Peak device VRAM (GiB, sampled every 200 ms)'),
                      ('sharpness_laplacian','Laplacian sharpness (higher is not necessarily better)'),
                      ('psnr','PSNR (dB)'),('ssim','SSIM'),('lpips','LPIPS (lower is better)'),
                      ('edge_f1','Edge F1 against original'),('color_mean_error','Mean RGB error')]:
        fig,ax=plt.subplots(figsize=(8,5))
        any_values=False
        for name in MODELS:
            values=[s for s in summary if s['model']==name and key in s]
            if values:
                ax.plot([s['blur'] for s in values],[s[key]/2**30 if key.endswith('_bytes') else s[key] for s in values],marker='o',label=name)
                any_values=True
        input_key={'psnr':'psnr','ssim':'ssim','lpips':'lpips','sharpness_laplacian':'sharpness_laplacian'}.get(key)
        if input_key and any(input_key in b for b in baseline):
            ax.plot(LEVELS,[statistics.mean(b[input_key] for b in baseline if b['blur']==level) for level in LEVELS],linestyle='--',color='gray',label='Blurred input baseline')
            any_values=True
        if any_values: ax.legend()
        else: ax.text(0.5,0.5,'No completed measurements',ha='center',transform=ax.transAxes)
        ax.set(xlabel='Gaussian blur radius',ylabel=label,xticks=LEVELS)
        ax.grid(alpha=0.2)
        fig.tight_layout(); fig.savefig(directory/f'{key}.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path)
    report(parser.parse_args().batch.resolve())
