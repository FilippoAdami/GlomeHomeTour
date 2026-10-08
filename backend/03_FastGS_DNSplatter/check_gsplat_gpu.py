#!/usr/bin/env python3
"""Small, explicit GPU checks against a differentiable CPU reference.

Run with backend/.venv/bin/python. No scene data or long GPU workload is used.
"""
import time

import torch
from gsplat.cuda._backend import _C


def reference(means, conics, colors, opacities, background, width, height):
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    delta = means[:, None, None] - torch.stack((x + .5, y + .5), -1)
    dx, dy = delta.unbind(-1)
    a, b, c = conics.unbind(-1)
    sigma = .5 * (a[:, None, None] * dx.square() + c[:, None, None] * dy.square()) + b[:, None, None] * dx * dy
    alpha = (opacities[:, None, None] * (-sigma).exp()).clamp_max(.999)
    alpha = torch.where((sigma >= 0) & (alpha >= 1 / 255), alpha, 0)
    accepted = (1 - alpha).cumprod(0) > 1e-4
    alpha = torch.where(accepted, alpha, 0)
    transmittance = (1 - alpha).cumprod(0)
    before = torch.cat((torch.ones_like(transmittance[:1]), transmittance[:-1]))
    rendered = ((alpha * before)[..., None] * colors[:, None, None]).sum(0)
    return rendered + transmittance[-1, ..., None] * background, 1 - transmittance[-1, ..., None]


def main():
    torch.manual_seed(12)
    for n, width, height, tile, channels, packed in ((1, 8, 8, 8, 4, False), (37, 13, 11, 8, 4, True),
                                                    (128, 32, 32, 16, 4, False), (31, 19, 17, 32, 3, False)):
        means = (torch.rand(n, 2) * min(width, height)).requires_grad_()
        conics = torch.tensor([.12, .01, .12]).repeat(n, 1).requires_grad_()
        colors = torch.rand(n, channels).requires_grad_()
        opacities = (torch.rand(n) * .25).requires_grad_()
        bg = torch.rand(channels)
        rendered, alpha = reference(means, conics, colors, opacities, bg, width, height)
        vc, va = torch.randn_like(rendered), torch.randn_like(alpha)
        gradients = torch.autograd.grad((rendered * vc).sum() + (alpha * va).sum(), (means, conics, colors, opacities), retain_graph=True)
        tiles_h, tiles_w = (height + tile - 1) // tile, (width + tile - 1) // tile
        offsets = (torch.arange(tiles_h * tiles_w, dtype=torch.int32) * n).reshape(1, tiles_h, tiles_w).cuda()
        ids = torch.arange(n, dtype=torch.int32).repeat(tiles_h * tiles_w).cuda()
        inputs = tuple(t.detach().cuda() if packed else t.detach()[None].cuda() for t in (means, conics, colors, opacities))
        fargs = (*inputs, bg[None].cuda(), None, width, height, tile, offsets, ids)
        r, a, last = _C.rasterize_to_pixels_3dgs_fwd(*fargs)
        if n == 128:
            assert last.unique().numel() > 1, "exercise different backward bounds across lanes"
        torch.testing.assert_close(r[0].cpu(), rendered, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(a[0].cpu(), alpha, atol=2e-5, rtol=2e-5)
        bargs = (*fargs, a, last, vc[None].cuda(), va[None].cuda(), True)
        actual = _C.rasterize_to_pixels_3dgs_bwd(*bargs)
        for expected, got in zip(gradients, actual[1:]):
            torch.testing.assert_close(got.reshape_as(expected).cpu(), expected, atol=4e-4, rtol=5e-4)
        if n == 1:
            abs_means = torch.zeros_like(means)
            for row in range(height):
                for col in range(width):
                    pixel_loss = (rendered[row, col] * vc[row, col]).sum() + (alpha[row, col] * va[row, col]).sum()
                    abs_means += torch.autograd.grad(pixel_loss, means, retain_graph=True)[0].abs()
            torch.testing.assert_close(actual[0].reshape_as(means).cpu(), abs_means, atol=4e-4, rtol=5e-4)
        torch.cuda.synchronize()
        print(f"PASS: {n} Gaussians, {width}x{height}, tile {tile}, channels {channels}, packed={packed}", flush=True)
        time.sleep(.05)


if __name__ == "__main__":
    main()
