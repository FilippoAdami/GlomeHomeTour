"""CPU checks for checkpoint loading without model initialization copies."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import save_file

from pixal3d import models


class TinyModel(torch.nn.Module):
    created = 0

    def __init__(self):
        super().__init__()
        TinyModel.created += 1
        self.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.register_buffer("phase", torch.arange(2, dtype=torch.float32))


class TinyRopeModel(TinyModel):
    def __init__(self):
        super().__init__()
        self.register_buffer("rope_phases", torch.arange(2, dtype=torch.float32))

    def _make_rope_phases(self, device):
        return torch.arange(2, dtype=torch.float32, device=device)


class FastLoadTest(unittest.TestCase):
    def test_meta_load_matches_regular_load_and_handles_missing_buffer(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(models, "__getattr__", return_value=TinyModel):
            path = Path(directory) / "tiny"
            path.with_suffix(".json").write_text(json.dumps({"name": "TinyModel", "args": {}}))
            for state in (
                {"weight": torch.tensor([2, 3], dtype=torch.float16), "phase": torch.arange(2, dtype=torch.float32)},
                {"weight": torch.tensor([2, 3], dtype=torch.float16)},
            ):
                save_file(state, path.with_suffix(".safetensors"))
                regular = models.from_pretrained(str(path))
                before = TinyModel.created
                fast = models.from_pretrained(str(path), load_on_meta=True)
                self.assertEqual(TinyModel.created - before, 1 if "phase" in state else 2)
                for name, tensor in regular.state_dict().items():
                    self.assertFalse(fast.state_dict()[name].is_meta)
                    torch.testing.assert_close(fast.state_dict()[name], tensor, rtol=0, atol=0)

    def test_missing_rope_buffer_is_rebuilt_without_model_reinitialization(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(models, "__getattr__", return_value=TinyRopeModel):
            path = Path(directory) / "tiny_rope"
            path.with_suffix(".json").write_text(json.dumps({"name": "TinyRopeModel", "args": {}}))
            save_file({"weight": torch.tensor([2, 3]), "phase": torch.arange(2, dtype=torch.float32)},
                      path.with_suffix(".safetensors"))
            before = TinyModel.created
            fast = models.from_pretrained(str(path), load_on_meta=True)
            self.assertEqual(TinyModel.created - before, 1)
            self.assertTrue(torch.equal(fast.rope_phases, torch.arange(2, dtype=torch.float32)))


if __name__ == "__main__":
    unittest.main()
