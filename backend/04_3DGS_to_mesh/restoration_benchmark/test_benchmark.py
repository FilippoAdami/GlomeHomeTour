import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import csv
import contextlib
import io
import json
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import types
import sys

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import LEVELS, PACKAGE_ROOT, ARTIFACTS, amortized, select, worker, dump
from report import metrics


class BenchmarkTests(unittest.TestCase):
    def test_model_and_scene_assets_use_separate_roots(self):
        self.assertEqual(PACKAGE_ROOT.name,'04_3DGS_to_mesh')
        self.assertEqual(ARTIFACTS,PACKAGE_ROOT.parent/'current_scene'/'04_3DGS_to_mesh')

    def fixture(self, root, count=12):
        originals=root/'originals';originals.mkdir()
        for level in LEVELS:
            directory=root/f'blur_{level}';directory.mkdir()
            for i in range(count):
                Image.new('RGB',(24,24),(i,i,i)).save(directory/f'{i:02}.png')
        for file in (root/'blur_3').iterdir():
            Image.new('RGB',(24,24)).save(originals/file.name)
        return originals

    def test_selection_is_deterministic_and_shared(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);truth=self.fixture(root)
            selected=select(root,truth)
            self.assertEqual(selected,select(root,truth))
            self.assertEqual(len(selected),10)
            self.assertEqual(len({r['filename'] for r in selected}),10)
            self.assertEqual(selected[0]['filename'],'00.png')
            self.assertEqual(selected[-1]['filename'],'11.png')
            self.assertTrue(all(len(r['inputs'])==4 for r in selected))
            missing=root/'blur_13'/selected[0]['filename'];missing.unlink()
            with self.assertRaises(FileNotFoundError): select(root,truth)

    def test_rejects_dimension_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);truth=self.fixture(root,10)
            Image.new('RGB',(12,24)).save(root/'blur_7/00.png')
            with self.assertRaisesRegex(ValueError,'Dimension mismatch'): select(root,truth)

    def test_amortization_uses_full_session(self):
        result=amortized([dict(processing_s=2),dict(processing_s=4)],30,10)
        self.assertEqual(result['amortized_40_s'],4)
        self.assertEqual(result['projected_10_s'],7)

    def test_metrics_compare_to_unblurred_reference(self):
        rng=np.random.default_rng(42)
        original=Image.fromarray(rng.integers(0,256,(32,32,3),dtype=np.uint8))
        matching=metrics(original,original)
        wrong=metrics(Image.new('RGB',(32,32)),original)
        self.assertEqual(matching['ssim'],1)
        self.assertGreater(matching['psnr'],wrong['psnr'])
        self.assertEqual(matching['color_mean_error'],0)
        self.assertNotIn('psnr',metrics(original))

    def test_worker_loads_once_and_saves_all_four_levels(self):
        import torch
        calls=[]
        class Embeds:
            def to(self,*args,**kwargs): return self
        resets=[]
        memory_limits=[]
        cache_clears=[]
        def restore(image, seed):
            calls.append(seed)
            torch.cuda.reset_peak_memory_stats()
            return image.copy()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);truth=self.fixture(root,10)
            for folder in [truth]+[root/f'blur_{level}' for level in LEVELS]:
                Image.new('RGB',(32,24)).save(folder/'01.png')
            manifest=root/'manifest.json';dump(manifest,select(root,truth))
            run=root/'run';run.mkdir()
            for level in LEVELS: (run/f'blur_{level}').mkdir()
            args=SimpleNamespace(artifacts=str(root),run=str(run),manifest=str(manifest),model='osediff')
            cuda=SimpleNamespace(get_device_name=lambda _: 'test GPU',synchronize=lambda:None,
                set_per_process_memory_fraction=memory_limits.append,
                empty_cache=lambda:cache_clears.append(True),reset_peak_memory_stats=lambda:resets.append(True),
                max_memory_allocated=lambda:1024,max_memory_reserved=lambda:2048)
            with patch('model_adapters.load_model',return_value=restore) as loader, \
                 patch.object(torch,'cuda',cuda), \
                 patch.object(torch,'manual_seed'), \
                 patch('run.subprocess.check_output',return_value='test_revision'), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(worker(args),0)
            loader.assert_called_once()
            self.assertEqual(calls,[42,43]+list(range(42,52))*4)
            self.assertEqual(len(resets),42)  # Two warmup resets and one benchmark reset per image.
            self.assertEqual(memory_limits,[0.7])
            self.assertEqual(len(cache_clears),1)  # Keep allocator cache through the batch; release at unload.
            for level in LEVELS:
                self.assertEqual(len(list((run/f'blur_{level}').glob('*.png'))),10)
            result=json.loads((run/'benchmark.json').read_text())
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['rocm_aotriton_experimental'],'0')
            self.assertEqual(result['warmup_resolutions'],[[24,24],[32,24]])
            self.assertEqual(set(result['per_level']),{'3','5','7','13'})
            self.assertGreater(result['warmup_s'],0)
            for measured in result['per_level'].values():
                self.assertAlmostEqual(measured['overhead_s'],result['load_s']+result['unload_s'])
            self.assertTrue((run/'resource_trace.csv').exists())

    def test_worker_pins_named_turbo_step_counts(self):
        import torch
        calls=[]
        def restore(image, seed, **kwargs):
            calls.append(kwargs.get('num_inference_steps'))
            return image.copy()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);truth=self.fixture(root,10)
            manifest=root/'manifest.json';dump(manifest,select(root,truth))
            run=root/'run';run.mkdir()
            for level in LEVELS: (run/f'blur_{level}').mkdir()
            args=SimpleNamespace(artifacts=str(root),run=str(run),manifest=str(manifest),model='seesr_turbo_2step')
            cuda=SimpleNamespace(get_device_name=lambda _: 'test GPU',synchronize=lambda:None,
                set_per_process_memory_fraction=lambda _:None,empty_cache=lambda:None,
                reset_peak_memory_stats=lambda:None,max_memory_allocated=lambda:1024,
                max_memory_reserved=lambda:2048)
            with patch('model_adapters.load_model',return_value=restore), \
                 patch.object(torch,'cuda',cuda),patch.object(torch,'manual_seed'), \
                 patch('run.subprocess.check_output',return_value='test_revision'), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(worker(args),0)
            self.assertEqual(calls,[2]*41)

    def test_turbo_adapter_defaults_to_three_steps_and_allows_override(self):
        import model_adapters
        from PIL import Image
        class Embeds:
            def to(self,*args,**kwargs): return self
        class Pipeline:
            controlnet=SimpleNamespace(set_attention_slice=lambda _:None)
            unet=SimpleNamespace(set_attention_slice=lambda _:None)
            def __call__(self,*args,**kwargs):
                calls.append(kwargs['num_inference_steps'])
                return SimpleNamespace(images=[Image.new('RGB',(8,8))])
        calls=[]
        module=SimpleNamespace(load_seesr_pipeline=lambda *args:Pipeline(),
            get_validation_prompt=lambda *args,**kwargs:('prompt',Embeds()),
            adain_color_fix=lambda output,prepared:output)
        tagger=SimpleNamespace(eval=lambda:SimpleNamespace(to=lambda *args,**kwargs:None))
        ram_lora=types.ModuleType('ram.models.ram_lora');ram_lora.ram=lambda **kwargs:tagger
        ram_models=types.ModuleType('ram.models');ram=types.ModuleType('ram')
        with patch.dict(sys.modules,{'ram':ram,'ram.models':ram_models,'ram.models.ram_lora':ram_lora}), \
             patch.object(model_adapters.importlib,'import_module',return_value=module), \
             patch.object(model_adapters.os,'chdir'), \
             patch.object(Path,'stat',return_value=SimpleNamespace(st_size=0)), \
             patch.object(model_adapters.torch.cuda,'empty_cache'), \
             patch.object(model_adapters.torch.cuda,'mem_get_info',return_value=(0,1)), \
             patch.object(model_adapters.torch.cuda,'memory_allocated',return_value=0), \
             patch.object(model_adapters.torch,'manual_seed'), \
             patch.object(model_adapters.torch,'autocast'), \
             patch.object(model_adapters.torch,'Generator',return_value=SimpleNamespace(manual_seed=lambda _:None)), \
             patch.object(model_adapters,'prepare',side_effect=lambda image:image):
            restore=model_adapters.load_model('seesr_turbo_2step',Path('/artifacts'),Path('/package'))
            restore(Image.new('RGB',(8,8)),42)
            restore(Image.new('RGB',(8,8)),42,num_inference_steps=2)
        self.assertEqual(calls,[3,2])

    def test_worker_stops_immediately_on_allocation_failure(self):
        import torch
        calls=[]
        def restore(image, seed):
            calls.append(seed)
            if len(calls)>1:
                raise torch.OutOfMemoryError('simulated allocation failure')
            return image.copy()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);truth=self.fixture(root,10)
            manifest=root/'manifest.json';dump(manifest,select(root,truth))
            run=root/'run';run.mkdir()
            for level in LEVELS: (run/f'blur_{level}').mkdir()
            args=SimpleNamespace(artifacts=str(root),run=str(run),manifest=str(manifest),model='osediff')
            cuda=SimpleNamespace(get_device_name=lambda _: 'test GPU',synchronize=lambda:None,
                empty_cache=lambda:None,reset_peak_memory_stats=lambda:None,
                set_per_process_memory_fraction=lambda _:None)
            with patch('model_adapters.load_model',return_value=restore), \
                 patch.object(torch,'cuda',cuda),patch.object(torch,'manual_seed'), \
                 patch('run.subprocess.check_output',return_value='test_revision'):
                self.assertEqual(worker(args),1)
            self.assertEqual(len(calls),2)
            result=json.loads((run/'benchmark.json').read_text())
            self.assertEqual(result['status'],'blocked')
            self.assertIn('simulated allocation failure',result['error'])

    def test_later_report_preserves_previous_batch_summary(self):
        import report
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);truth=self.fixture(root,10)
            manifest=root/'manifest.json';dump(manifest,select(root,truth))
            first=root/'batch_first.json';second=root/'batch_second.json'
            for batch in (first,second): dump(batch,dict(manifest=str(manifest),runs=[]))
            metric=SimpleNamespace(eval=lambda:SimpleNamespace(cpu=lambda:None))
            with patch('report.lpips.LPIPS',return_value=metric), \
                 patch('report.panel'),patch('report.plots'), \
                 contextlib.redirect_stdout(io.StringIO()):
                report.report(first)
                archived=root/'comparison/batch_first/benchmark_summary.md'
                original=archived.read_text()
                report.report(second)
            self.assertEqual(archived.read_text(),original)
            self.assertIn('batch_first.json',original)
            self.assertIn('batch_second.json',(root/'benchmark_summary.md').read_text())
            self.assertTrue((root/'comparison/batch_first/benchmark_summary.csv').is_file())
            self.assertTrue((root/'comparison/batch_second/benchmark_summary.csv').is_file())

    def test_report_corrects_legacy_warmup_amortization(self):
        import report
        from run import write_csv
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);truth=self.fixture(root,10)
            manifest=root/'manifest.json';records=select(root,truth);dump(manifest,records)
            run=root/'run';run.mkdir()
            rows=[]
            for level in LEVELS:
                (run/f'blur_{level}').mkdir()
                for record in records:
                    with Image.open(record['inputs'][str(level)]['path']) as image:
                        image.save(run/f'blur_{level}'/record['filename'])
                    rows.append(dict(blur=level,filename=record['filename'],status='ok',processing_s=3))
            write_csv(run/'per_image_metrics.csv',rows)
            dump(run/'benchmark.json',dict(model='osediff',status='complete',load_s=30,unload_s=10,
                warmup_s=400,per_level={str(level):dict(count=10,inference_mean_s=2,
                amortized_40_s=14,projected_10_s=47) for level in LEVELS}))
            batch=root/'batch.json';dump(batch,dict(manifest=str(manifest),runs=[str(run)]))
            metric=SimpleNamespace(eval=lambda:SimpleNamespace(cpu=lambda:None))
            with patch('report.lpips.LPIPS',return_value=metric),patch('report.metrics',return_value={}), \
                 patch('report.panel'),patch('report.plots'),contextlib.redirect_stdout(io.StringIO()):
                report.report(batch)
            corrected=json.loads((run/'benchmark.json').read_text())
            self.assertEqual(corrected['warmup_s'],400)
            for level in corrected['per_level'].values():
                self.assertEqual(level['amortized_40_s'],4)
                self.assertEqual(level['projected_10_s'],7)
            with (root/'benchmark_summary.csv').open() as source:
                summaries=list(csv.DictReader(source))
            self.assertTrue(all(float(row['amortized_40_s'])==4 for row in summaries))

    def test_download_resumes_complete_ranges_after_failure(self):
        from download_asset import transfer
        payload=bytes(range(40))
        counts={}
        fail=[True]
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                start,end=map(int,self.headers['Range'].removeprefix('bytes=').split('-'))
                counts[start]=counts.get(start,0)+1
                if start==8 and fail[0]:
                    self.send_error(503);return
                block=payload[start:end+1]
                self.send_response(206)
                self.send_header('Content-Range',f'bytes {start}-{end}/{len(payload)}')
                self.send_header('Content-Length',str(len(block)))
                self.end_headers();self.wfile.write(block)
            def log_message(self,*args): pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                destination=Path(directory)/'weights.bin'
                url=f'http://127.0.0.1:{server.server_port}/weights'
                digest=hashlib.sha256(payload).hexdigest()
                with patch('download_asset.time.sleep'),contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError,'five attempts'):
                        transfer(url,destination,len(payload),digest,workers=1,chunk_size=8)
                self.assertFalse(destination.exists())
                self.assertTrue(list(Path(directory).rglob('000000.part')))
                first_requests=counts[0]
                fail[0]=False
                with contextlib.redirect_stdout(io.StringIO()):
                    transfer(url,destination,len(payload),digest,workers=2,chunk_size=8)
                self.assertEqual(destination.read_bytes(),payload)
                self.assertEqual(counts[0],first_requests)
        finally:
            server.shutdown();server.server_close();thread.join()


if __name__=='__main__': unittest.main()
