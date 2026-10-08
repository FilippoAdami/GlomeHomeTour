#!/usr/bin/env python3
"""Paced checkpoint-path checks against an independent CPU compositor."""
import time

import torch
from gsplat.cuda._wrapper import rasterize_to_pixels

from check_gsplat_gpu import reference


def check_occluded_long_list():
    # Forces the >1GiB worst-case checkpoint estimate with only32MiB of IDs.
    # Opaque duplicate splats terminate in<32contributors at every pixel.
    from gsplat.cuda._backend import _C
    cpu=[torch.tensor([[6.5,5.5]],requires_grad=True),
         torch.tensor([[.01,0.,.01]],requires_grad=True),
         torch.tensor([[.3,.5,.7,.9]],requires_grad=True),
         torch.tensor([.9],requires_grad=True)]
    bg=torch.tensor([.1,.2,.3,.4])
    want,wa=reference(*(v.repeat(32,1) if v.ndim==2 else v.repeat(32) for v in cpu),bg,13,11)
    expected=torch.autograd.grad(want.sum()+wa.sum(),cpu)
    gpu=[v.detach()[None].cuda().requires_grad_() for v in cpu]
    offsets=torch.zeros(1,1,1,dtype=torch.int32,device="cuda")
    ids=torch.zeros(8_000_000,dtype=torch.int32,device="cuda")
    raw=_C.rasterize_checkpoint_3dgs_fwd(*gpu,bg[None].cuda(),None,13,11,16,offsets,ids)
    assert raw[3].numel()==256*5,raw[3].shape
    r,a=rasterize_to_pixels(*gpu,13,11,16,offsets,ids,backgrounds=bg[None].cuda())
    torch.testing.assert_close(r[0].cpu(),want,atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(a[0].cpu(),wa,atol=2e-5,rtol=2e-5)
    actual=torch.autograd.grad(r.sum()+a.sum(),gpu)
    for got,ref in zip(actual,expected):
        torch.testing.assert_close(got.cpu().reshape_as(ref),ref,atol=4e-4,rtol=5e-4)
    torch.cuda.synchronize();time.sleep(.2)
    print("PASS checkpoint:8M occluded tile entries compact to one checkpoint, CPUvalues/gradients",flush=True)


def main():
    torch.manual_seed(6)
    width, height, tile, n = 19, 17, 16, 96
    for packed, opaque, background, channels in ((False, False, False, 4), (True, True, True, 4),
                                               (False, False, True, 3), (True, True, True, 3)):
        means = (torch.rand(2, n, 2) * min(width, height)).requires_grad_()
        conics = torch.tensor([.012, .001, .014]).repeat(2, n, 1).requires_grad_()
        colors = torch.rand(2, n, channels).requires_grad_()
        opacities = torch.full((2, n), .9 if opaque else .25).requires_grad_()
        bg = torch.rand(2, channels).requires_grad_() if background else None
        mask = torch.ones(2, 2, 2, dtype=torch.bool)
        mask[0, 1, 0] = False
        lists = [[], list(range(17)), list(range(n)), list(range(n)),
                 list(range(n)), [], list(range(29)), []]
        starts, ids, total = [], [], 0
        for index, selected in enumerate(lists):
            starts.append(total)
            ids.extend(g + (index // 4) * n for g in selected)
            total += len(selected)
        offsets = torch.tensor(starts, dtype=torch.int32).reshape(2, 2, 2)
        ids = torch.tensor(ids, dtype=torch.int32)
        rendered = torch.zeros(2, height, width, channels)
        alpha = torch.zeros(2, height, width, 1)
        for image in range(2):
            for row in range(2):
                for col in range(2):
                    selected = lists[image * 4 + row * 2 + col]
                    h, w = min(tile, height - row * tile), min(tile, width - col * tile)
                    b = bg[image] if bg is not None else torch.zeros(channels)
                    if selected and mask[image, row, col]:
                        r, a = reference(means[image, selected] - torch.tensor([col * tile, row * tile]),
                                         conics[image, selected], colors[image, selected],
                                         opacities[image, selected], b, w, h)
                    else:
                        r, a = b.expand(h, w, channels), torch.zeros(h, w, 1)
                    rendered[image, row * tile:row * tile + h, col * tile:col * tile + w] = r
                    alpha[image, row * tile:row * tile + h, col * tile:col * tile + w] = a
        vc, va = torch.randn_like(rendered), torch.randn_like(alpha)
        cpu_inputs = (means, conics, colors, opacities) + ((bg,) if bg is not None else ())
        expected = torch.autograd.grad((rendered * vc).sum() + (alpha * va).sum(), cpu_inputs, retain_graph=True)
        gpu_inputs = tuple(t.detach().reshape(-1, t.shape[-1]).cuda().requires_grad_()
                           if packed and i < 3 else t.detach().flatten().cuda().requires_grad_()
                           if packed and i == 3 else t.detach().cuda().requires_grad_()
                           for i, t in enumerate(cpu_inputs))
        r, a = rasterize_to_pixels(*gpu_inputs[:4], width, height, tile, offsets.cuda(), ids.cuda(),
                                  backgrounds=gpu_inputs[4] if bg is not None else None,
                                  masks=mask.cuda(), packed=packed, absgrad=True)
        torch.testing.assert_close(r.cpu(), rendered, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(a.cpu(), alpha, atol=2e-5, rtol=2e-5)
        actual = torch.autograd.grad((r * vc.cuda()).sum() + (a * va.cuda()).sum(), gpu_inputs, retain_graph=True)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got.cpu().reshape_as(want), want, atol=4e-4, rtol=5e-4)
        rgb_expected = torch.autograd.grad(rendered[..., :3].sum(), cpu_inputs)
        rgb_actual = torch.autograd.grad(r[..., :3].sum(), gpu_inputs)
        for got, want in zip(rgb_actual, rgb_expected):
            torch.testing.assert_close(got.cpu().reshape_as(want), want, atol=4e-4, rtol=5e-4)
        assert torch.isfinite(gpu_inputs[0].absgrad).all()
        with torch.no_grad():
            nr, na = rasterize_to_pixels(*gpu_inputs[:4], width, height, tile, offsets.cuda(), ids.cuda(),
                                        backgrounds=gpu_inputs[4] if bg is not None else None,
                                        masks=mask.cuda(), packed=packed)
        assert not nr.requires_grad and not na.requires_grad
        torch.testing.assert_close(nr.cpu(), rendered, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(na.cpu(), alpha, atol=2e-5, rtol=2e-5)

        torch.cuda.synchronize()
        print(f"PASS checkpoint: packed={packed}, opaque={opaque}, background={background}, channels={channels}, unequal/empty/masked tiles", flush=True)
        time.sleep(.2)

    params = [torch.ones(1, 1, channels, device="cuda", requires_grad=True)
              for channels in (2, 3, 4)]
    opacity = torch.ones(1, 1, device="cuda", requires_grad=True)
    r, a = rasterize_to_pixels(*params, opacity, 19, 17, 16,
                              torch.zeros(1, 2, 2, dtype=torch.int32, device="cuda"),
                              torch.empty(0, dtype=torch.int32, device="cuda"))
    assert r.count_nonzero() == 0 and a.count_nonzero() == 0
    grads = torch.autograd.grad(r.sum() + a.sum(), (*params, opacity))
    assert all(g.count_nonzero() == 0 for g in grads)
    print("PASS checkpoint: zero intersections and fallback backward", flush=True)
    check_occluded_long_list()


if __name__ == "__main__":
    main()
