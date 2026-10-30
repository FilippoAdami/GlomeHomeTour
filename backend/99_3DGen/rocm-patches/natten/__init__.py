"""Pure-torch stand-in for NATTEN's na2d (upstream is CUTLASS/NVIDIA-only).

Only what valeoai/NAF calls: na2d(q, k, v, kernel_size, dilation, stride=1, backend=...)
with q/k/v shaped [B, H, W, heads, dim].
"""
import math
import torch

__version__ = "0.0.0+pytorch-shim"

_CHUNK_ROWS = 0  # test hook: force a row-chunk size


def _neighborhood_index(length, kernel, dilation, device):
    """natten border rule: window slides inward instead of padding, per dilation group."""
    i = torch.arange(length, device=device)
    g, p = i % dilation, i // dilation
    group_len = (length - g + dilation - 1) // dilation
    start = (p - kernel // 2).clamp(min=0).minimum(group_len - kernel)
    return g[:, None] + dilation * (start[:, None] + torch.arange(kernel, device=device)[None, :])


def na2d(q, k, v, kernel_size, dilation=1, stride=1, scale=None, **kwargs):
    assert stride == 1, "shim only supports stride=1"
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    dh, dw = (dilation, dilation) if isinstance(dilation, int) else dilation
    B, H, W, N, D = q.shape
    Dv = v.shape[-1]                                       # v may be wider than q/k
    scale = scale or D ** -0.5

    ri = _neighborhood_index(H, kh, dh, q.device)          # [H, kh]
    ci = _neighborhood_index(W, kw, dw, q.device)          # [W, kw]

    # ponytail: materializing every neighborhood at once is B*H*W*N*kh*kw*D and blows
    # past 16GB on real inputs, so walk the rows in chunks of ~64MB worth of gather.
    budget = globals()["_CHUNK_ROWS"] or 2 ** 26 // max(1, q.element_size() * B * W * N * kh * kw * max(D, Dv))
    out = q.new_empty(B, H, W, N, Dv)
    for lo in range(0, H, max(1, budget)):
        hi = min(H, lo + max(1, budget))
        rows = ri[lo:hi]
        # [B,h,W,N,D] -> [B,h,kh,W,kw,N,D] -> [B,h,W,N,kh*kw,D]
        gather = lambda t: t[:, rows][:, :, :, ci].permute(0, 1, 3, 5, 2, 4, 6).reshape(B, hi - lo, W, N, kh * kw, t.shape[-1])
        kn, vn = gather(k), gather(v)
        attn = torch.einsum("bhwnd,bhwnkd->bhwnk", q[:, lo:hi] * scale, kn).softmax(-1)
        out[:, lo:hi] = torch.einsum("bhwnk,bhwnke->bhwne", attn, vn)
        del kn, vn, attn
    return out


# natten.functional.na2d_qk / na2d_av are the legacy API; NAF only needs the modern one.

if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W, N, D, kh, kw, dh, dw = 1, 7, 6, 2, 4, 3, 3, 2, 1
    q, k = (torch.randn(B, H, W, N, D, dtype=torch.float64) for _ in range(2))
    v = torch.randn(B, H, W, N, 2 * D, dtype=torch.float64)   # v wider than q/k
    out = na2d(q, k, v, (kh, kw), (dh, dw))

    ri = _neighborhood_index(H, kh, dh, q.device)
    ci = _neighborhood_index(W, kw, dw, q.device)
    for b, i, j, n in [(0, 0, 0, 0), (0, 3, 2, 1), (0, H - 1, W - 1, 0)]:
        nb = torch.stack([k[b, ri[i, a], ci[j, c], n] for a in range(kh) for c in range(kw)])
        vb = torch.stack([v[b, ri[i, a], ci[j, c], n] for a in range(kh) for c in range(kw)])
        w = ((nb @ q[b, i, j, n]) / math.sqrt(D)).softmax(0)
        assert torch.allclose(out[b, i, j, n], w @ vb), (b, i, j, n)
    # every window must stay in bounds and be strided by the dilation
    assert ri.min() >= 0 and ri.max() < H and (ri.diff(dim=1) == dh).all()
    _CHUNK_ROWS = 2  # chunked path must match the one-shot path
    assert torch.allclose(out, na2d(q, k, v, (kh, kw), (dh, dw)))
    print("na2d shim OK")
