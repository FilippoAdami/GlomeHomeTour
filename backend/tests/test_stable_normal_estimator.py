"""StableNormal lifecycle and output contracts without downloading model weights."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from stable_normal import StableNormalEstimator


class FakePipeline:
    def __init__(self):
        self.empty_text_embedding = None
        self.prompt_embeds = None
        self.prompt = ""
        self.text_encoder = torch.nn.Embedding(2, 4)
        self.tokenizer = self
        self.model_max_length = 77
        self.images = []

    def __call__(self, image, **kwargs):
        if isinstance(image, str):
            return SimpleNamespace(input_ids=torch.tensor([[0, 1]]))
        self.images.append(image)
        prediction = torch.tensor([[[[1., 0., -1.], [1., 0., -1.]],
                                    [[0., 1., 0.], [0., 1., 0.]],
                                    [[1., 1., 1.], [1., 1., 1.]]]])
        return SimpleNamespace(prediction=prediction)

    def encode_prompt(self, *args):
        assert not torch.is_grad_enabled()
        return self.text_encoder(torch.tensor([[0, 1]]))[0], None


@pytest.fixture
def pipeline(monkeypatch):
    pipe = FakePipeline()
    # Match the text encoder's tuple return contract.
    encoder = pipe.text_encoder
    pipe.text_encoder = lambda ids: (encoder(ids),)
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: SimpleNamespace(model=pipe))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return pipe


def test_load_caches_text_without_autograd_or_unused_warmup(pipeline):
    estimator = StableNormalEstimator(device="cpu")
    estimator.load()
    assert pipeline.images == []
    assert pipeline.empty_text_embedding.grad_fn is None
    assert pipeline.prompt_embeds.grad_fn is None
    assert pipeline.text_encoder is None
    assert pipeline.tokenizer is None
    predictor = estimator._predictor
    estimator.load()
    assert estimator._predictor is predictor
    estimator.unload()
    estimator.unload()
    assert estimator._predictor is None


def test_estimate_preserves_dimensions_unit_normals_and_input_conversion(pipeline):
    with StableNormalEstimator(device="cpu") as estimator:
        normals = estimator.estimate_normal(np.full((5, 7, 3), 300., dtype=np.float32))
        assert np.asarray(pipeline.images[0]).dtype == np.uint8
        assert np.asarray(pipeline.images[0]).max() == 255
        assert normals.shape == (5, 7, 3)
        assert normals.dtype == np.float32
        assert np.isfinite(normals).all()
        np.testing.assert_allclose(np.linalg.norm(normals, axis=-1), 1., atol=1e-6)
        np.testing.assert_array_equal(normals, estimator.estimate_normal(Image.new("RGB", (7, 5))))
    assert estimator._predictor is None


def test_gpu_load_does_not_run_a_synthetic_image(pipeline):
    pipeline.empty_text_embedding = torch.zeros(1, 2, 4)
    pipeline.prompt_embeds = torch.zeros(1, 77, 4)
    with StableNormalEstimator(device="cuda"):
        assert pipeline.images == []


def test_context_releases_predictor_on_failure(pipeline):
    estimator = StableNormalEstimator(device="cpu")
    with pytest.raises(RuntimeError, match="failed save"):
        with estimator:
            raise RuntimeError("failed save")
    assert estimator._predictor is None


def test_rocm_inference_restores_convolution_setting_on_error(pipeline, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)
    estimator = StableNormalEstimator(device="cpu")
    estimator.load()
    original_call = FakePipeline.__call__

    def fail(self, *args, **kwargs):
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.enabled is True
        assert torch.backends.cudnn.deterministic is True
        raise RuntimeError("inference failed")

    monkeypatch.setattr(FakePipeline, "__call__", fail)
    with pytest.raises(RuntimeError, match="inference failed"):
        estimator.estimate_normal(Image.new("RGB", (7, 5)))
    assert torch.backends.cudnn.benchmark is True
    monkeypatch.setattr(FakePipeline, "__call__", original_call)
    estimator.unload()


@pytest.fixture
def benchmark(monkeypatch, tmp_path):
    import benchmark_normal_full_pass as module

    instances = []

    class Estimator:
        def __init__(self):
            self.unloaded = False
            instances.append(self)

        def load(self):
            pass

        def estimate_normal(self, rgb):
            normals = np.zeros(rgb.shape, dtype=np.float32)
            normals[..., 2] = 1
            return normals

        def unload(self):
            self.unloaded = True

    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (7, 5)).save(images / "frame.jpg")
    output = tmp_path / "output"
    monkeypatch.setattr(module, "StableNormalEstimator", Estimator)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    monkeypatch.setattr("sys.argv", ["benchmark", "--images-dir", str(images), "--output-dir", str(output)])
    return module, output, instances


def test_benchmark_saves_maps_and_lifecycle_timings(benchmark):
    import json

    module, output, instances = benchmark
    assert module.main() == 0
    normals = np.load(output / "frame.npy")
    assert normals.shape == (5, 7, 3) and normals.dtype == np.float16
    assert np.all(normals[..., 2] == 1)
    report = json.loads((output / "benchmark.json").read_text())
    assert report["frames"] == 1
    assert {"load_s", "inference_and_save_s", "unload_s", "total_s"} <= report["timings"].keys()
    assert instances[0].unloaded


def test_benchmark_refuses_existing_output(benchmark):
    module, output, instances = benchmark
    output.mkdir()
    original = output / "frame.npy"
    original.write_bytes(b"existing map")
    with pytest.raises(SystemExit):
        module.main()
    assert original.read_bytes() == b"existing map"
    assert instances == []


def test_benchmark_propagates_background_write_failure_and_unloads(benchmark, monkeypatch):
    module, _, instances = benchmark

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(module.np, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        module.main()
    assert instances[0].unloaded
