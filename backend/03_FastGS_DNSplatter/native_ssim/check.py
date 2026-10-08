"""Run only in the serialized GPU slot: native fused SSIM vs conv2d reference."""
import torch
import torch.nn.functional as F
from fused_ssim import fused_ssim

G = torch.tensor([.0010283801, .0075987582, .0360007733, .1093606874, .213005528,
                  .266011715, .213005528, .1093606874, .0360007733, .0075987582,
                  .0010283801])

def reference(a, b):
    kernel = (G[:, None] * G[None, :]).to(a).expand(a.shape[1], 1, 11, 11)
    conv = lambda x: F.conv2d(x, kernel, padding=5, groups=x.shape[1])
    mu_a, mu_b = conv(a), conv(b)
    aa, bb, ab = conv(a*a) - mu_a*mu_a, conv(b*b) - mu_b*mu_b, conv(a*b) - mu_a*mu_b
    return (((2*mu_a*mu_b + .01**2) * (2*ab + .03**2)) /
            ((mu_a*mu_a + mu_b*mu_b + .01**2) * (aa + bb + .03**2))).mean()

if __name__ == "__main__":
    import time
    torch.manual_seed(1)
    for h, w in ((7, 13), (35, 37)):
        a = torch.rand(1, 3, h, w, device="cuda", requires_grad=True)
        b = torch.rand_like(a)
        got, want = fused_ssim(a, b), reference(a, b)
        torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-5)
        actual, expected = (torch.autograd.grad(v, a)[0] for v in (got, want))
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)
        assert torch.isfinite(actual).all()
        torch.cuda.synchronize(); time.sleep(.2)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        a = torch.rand(1, 3, 7, 13, device="cuda", requires_grad=True)
        b = torch.rand_like(a)
        got, want = fused_ssim(a, b), reference(a, b)
        torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-5)
        actual, expected = (torch.autograd.grad(v, a)[0] for v in (got, want))
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)
        assert torch.isfinite(actual).all()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    print("fused SSIM values, reference gradients, borders, and current-stream check: OK")
