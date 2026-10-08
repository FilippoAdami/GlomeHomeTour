"""CPU oracle and optional gfx1200 HIP check for VCP flagged-pixel counts."""
from __future__ import annotations
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

TILE = 16

def oracle(means, conics, opacities, flags, offsets, ids):
    """Exact scalar compositing reference; offsets are starts of row-major tile lists."""
    _, n, _ = means.shape; _, h, w = flags.shape; th, tw = offsets.shape[1:]
    out = torch.zeros(n, dtype=torch.int32)
    for ty in range(th):
        for tx in range(tw):
            tile = ty * tw + tx; start = int(offsets[0, ty, tx])
            end = int(ids.numel()) if tile + 1 == th * tw else int(offsets.flatten()[tile + 1])
            for y in range(ty * TILE, min((ty + 1) * TILE, h)):
                for x in range(tx * TILE, min((tx + 1) * TILE, w)):
                    transmittance = 1.0
                    for at in range(start, end):
                        g = int(ids[at]); dx, dy = means[0,g,0] - (x + .5), means[0,g,1] - (y + .5)
                        a, b, c = conics[0,g]; sigma = .5 * (a*dx*dx + c*dy*dy) + b*dx*dy
                        alpha = min(.999, float(opacities[0,g]) * float(torch.exp(-sigma)))
                        if sigma < 0 or alpha < 1/255: continue
                        next_t = transmittance * (1-alpha)
                        if next_t <= 1e-4: break
                        if flags[0,y,x]: out[g] += 1
                        transmittance = next_t
    return out

def cases():
    means = torch.tensor([[[.5,.5], [15.5,.5], [16.5,.5]]], dtype=torch.float32)
    conics = torch.tensor([[[1.,0.,1.]] * 3]); opacity = torch.tensor([[.5, 2., .001]])
    flags = torch.zeros(1, 2, 17, dtype=torch.bool); flags[0,0,0] = flags[0,0,15] = flags[0,0,16] = True
    offsets = torch.tensor([[[0, 2]]], dtype=torch.int32); ids = torch.tensor([0, 1, 2], dtype=torch.int32)
    empty = torch.empty(0, dtype=torch.int32)
    return [
        ("partial-border", means, conics, opacity, flags, offsets, ids, torch.tensor([1, 1, 0], dtype=torch.int32)),
        ("flags-false", means, conics, opacity, torch.zeros_like(flags), offsets, ids, torch.zeros(3, dtype=torch.int32)),
        ("empty-list", means, conics, opacity, flags, torch.zeros_like(offsets), empty, torch.zeros(3, dtype=torch.int32)),
        # Fourth .95-alpha contribution crosses 1e-4 and is excluded.
        ("exclusive-terminal", torch.tensor([[[.5, .5]]]), torch.tensor([[[1., 0., 1.]]]),
         torch.tensor([[.95]]), torch.ones(1, 1, 1, dtype=torch.bool), torch.zeros(1, 1, 1, dtype=torch.int32),
         torch.tensor([0, 0, 0, 0], dtype=torch.int32), torch.tensor([3], dtype=torch.int32)),
    ]


def cpu_check():
    for name, *inputs, expected in cases():
        got = oracle(*inputs)
        assert torch.equal(got, expected), (name, got, expected)


def tiled_oracle(means, conics, opacity, flags, offsets, ids):
    """Vectorized CPU oracle for saved captures; preserves front-to-back order."""
    means, conics, opacity = (value.numpy()[0] for value in (means, conics, opacity))
    flags, offsets, ids = flags.numpy()[0], offsets.numpy()[0], ids.numpy()
    h, w = flags.shape; th, tw = offsets.shape; out = np.zeros(len(means), np.int32)
    for ty in range(th):
        for tx in range(tw):
            tile = ty * tw + tx; start = offsets[ty, tx]
            end = len(ids) if tile + 1 == th * tw else offsets.flat[tile + 1]
            yy, xx = np.mgrid[ty*TILE:min((ty+1)*TILE,h), tx*TILE:min((tx+1)*TILE,w)]
            if not xx.size: continue
            px, py = xx.reshape(-1).astype(np.float32) + .5, yy.reshape(-1).astype(np.float32) + .5
            visible, transmittance = flags[yy, xx].reshape(-1), np.ones(len(px), np.float32)
            done = np.zeros(len(px), bool)
            for g in ids[start:end]:
                dx, dy = means[g, 0] - px, means[g, 1] - py
                a, b, c = conics[g]; sigma = .5 * (a*dx*dx + c*dy*dy) + b*dx*dy
                alpha = np.minimum(np.float32(.999), opacity[g] * np.exp(-sigma))
                accepted = ~done & (sigma >= 0) & (alpha >= np.float32(1/255))
                next_t = transmittance * (1 - alpha)
                count = accepted & (next_t > np.float32(1e-4))
                out[g] += np.count_nonzero(count & visible)
                done |= accepted & ~count
                transmittance[count] = next_t[count]
    return torch.from_numpy(out)


def gpu_check(module):
    import time
    for name, means, conics, opacity, flags, offsets, ids, expected in cases():
        got = module.vcp_counts(*(value.cuda() for value in (means, conics, opacity, flags, offsets, ids))).cpu()
        assert torch.equal(got, expected), (name, got, expected)
        torch.cuda.synchronize()
        time.sleep(.2)
    print("HIP VCP counts match CPU oracle: OK")



@torch.no_grad()
def threshold_checks(module):
    """Counts must match native rendering at alpha and termination boundaries."""
    from gsplat.cuda._backend import _C
    means = torch.tensor([[[.5, .5]]], device="cuda")
    conics = torch.tensor([[[1., 0., 1.]]], device="cuda")
    colors = torch.ones(1, 1, 4, device="cuda")
    flags = torch.ones(1, 1, 1, dtype=torch.bool, device="cuda")
    offsets = torch.zeros(1, 1, 1, dtype=torch.int32, device="cuda")
    for sigma in (0., .1, .5, 1., 3., 5.):
        means[0, 0, 0] = .5 + (2 * sigma)**.5
        center = np.float32(np.exp(sigma) / 255)
        low, high = center, center
        values = [center]
        for _ in range(4):
            low = np.nextafter(low, np.float32(0)); high = np.nextafter(high, np.float32(1))
            values.extend((low, high))
        ids = torch.zeros(1, dtype=torch.int32, device="cuda")
        for value in values:
            opacity = torch.tensor([[float(value)]], device="cuda")
            r, _, _ = _C.rasterize_to_pixels_3dgs_fwd(means, conics, colors, opacity, None, None, 1, 1, 16, offsets, ids)
            count = module.vcp_counts(means, conics, opacity, flags, offsets, ids)
            assert int(count[0]) == int(r[0, 0, 0, 0] > 0), (sigma, value, count, r)
        torch.cuda.synchronize(); time.sleep(.1)
    means[0, 0, 0] = .5
    ids = torch.zeros(4, dtype=torch.int32, device="cuda")
    for value in (np.nextafter(np.float32(.99), np.float32(0)), np.float32(.99),
                  np.nextafter(np.float32(.99), np.float32(1))):
        opacity = torch.tensor([[float(value)]], device="cuda")
        _, _, last = _C.rasterize_to_pixels_3dgs_fwd(means, conics, colors, opacity, None, None, 1, 1, 16, offsets, ids)
        count = module.vcp_counts(means, conics, opacity, flags, offsets, ids)
        assert int(count[0]) == int(last[0, 0, 0]) + 1, (value, count, last)
    torch.cuda.synchronize()
    print("VCP threshold-adjacent counts match native renderer: OK")

def saved_checks(module):
    crop = torch.load("/tmp/gsplat-perf-next/crop_backward.pt", map_location="cpu", weights_only=True)
    means, conics, opacity, offsets, ids = crop[0], crop[1], crop[3], crop[9], crop[10]
    for name, flags in (("all-true", torch.ones(1, 64, 64, dtype=torch.bool)),
                        ("patterned", ((torch.arange(64)[:, None] * 5 + torch.arange(64) * 3) % 7 == 0)[None])):
        expected = tiled_oracle(means, conics, opacity, flags, offsets, ids)
        got = module.vcp_counts(*(value.cuda() for value in (means, conics, opacity, flags, offsets, ids))).cpu()
        assert torch.equal(got, expected), (name, got.sum(), expected.sum())
        torch.cuda.synchronize(); time.sleep(.2)
        print(f"crop {name}: {int(got.sum())} pairs")

    real = torch.load("/tmp/gsplat-perf-next/real_backward.pt", map_location="cpu", weights_only=True)
    means, conics, opacity, offsets, ids = real[0], real[1], real[3], real[9], real[10]
    flags = torch.ones(1, 1920, 1080, dtype=torch.bool)
    args = tuple(value.cuda() for value in (means, conics, opacity, flags, offsets, ids))
    torch.cuda.synchronize(); time.sleep(.2)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(); counts = module.vcp_counts(*args); end.record(); torch.cuda.synchronize()
    assert torch.isfinite(counts).all() and counts.dtype == torch.int32
    print(f"full all-true: {start.elapsed_time(end):.3f} ms, {int(counts.sum())} pairs, {int((counts != 0).sum())} gaussians")

if __name__ == "__main__":
    from gsplat.cuda._backend import _C
    cpu_check()
    class Backend:
        vcp_counts = staticmethod(_C.rasterize_vcp_counts)
    gpu_check(Backend)
    threshold_checks(Backend)
