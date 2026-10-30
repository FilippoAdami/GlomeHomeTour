import torch
import torch.nn as nn
import torch.nn.functional as F
from . import VarLenTensor

__all__ = [
    'SparseLinear'
]


class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__(in_features, out_features, bias)

    def forward(self, input: VarLenTensor) -> VarLenTensor:
        feats = input.feats
        chunk_size = 131072
        if feats.is_cuda and feats.shape[0] > chunk_size:
            out_chunks = []
            for i in range(0, feats.shape[0], chunk_size):
                out_chunks.append(F.linear(feats[i : i + chunk_size], self.weight, self.bias))
            return input.replace(torch.cat(out_chunks, dim=0))
        return input.replace(super().forward(feats))

