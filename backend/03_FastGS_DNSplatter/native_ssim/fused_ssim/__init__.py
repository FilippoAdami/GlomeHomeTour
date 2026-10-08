"""MIT fused 11x11, sigma=1.5 SSIM for fixed-target NCHW images."""
import torch
from fused_ssim_cuda import fusedssim, fusedssim_backward


class _FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img1, img2, padding, train):
        ssim_map, dmu1, dsigma1, dsigma12 = fusedssim(.01 ** 2, .03 ** 2, img1, img2, train)
        ctx.save_for_backward(img1.detach(), img2, dmu1, dsigma1, dsigma12)
        ctx.padding = padding
        return ssim_map if padding == "same" else ssim_map[:, :, 5:-5, 5:-5]

    @staticmethod
    def backward(ctx, grad):
        img1, img2, dmu1, dsigma1, dsigma12 = ctx.saved_tensors
        if ctx.padding == "valid":
            full = torch.zeros_like(img1)
            full[:, :, 5:-5, 5:-5] = grad
            grad = full
        return fusedssim_backward(.01 ** 2, .03 ** 2, img1, img2, grad, dmu1, dsigma1, dsigma12), None, None, None


def fused_ssim(img1, img2, padding="same", train=True):
    if padding not in ("same", "valid"):
        raise ValueError("padding must be 'same' or 'valid'")
    if padding == "valid" and (img1.shape[-2] < 11 or img1.shape[-1] < 11):
        raise ValueError("valid padding requires H and W >= 11")
    if img1.requires_grad and not train:
        raise ValueError("train=False cannot be used when img1 requires gradients")
    return _FusedSSIMMap.apply(img1, img2, padding, train).mean()
