# Smallest possible check: plane-sweep + backproject must recover a known
# fronto-parallel plane from two synthetic pinhole views. Run: python3 test_densify_pointcloud.py
import torch

from densify_pointcloud import plane_sweep, backproject, device

TRUE_DEPTH = 2.0


def synth_view(K, R, t, H, W):
    """Render a flat textured plane at TRUE_DEPTH as seen by camera (K,R,t)."""
    ys, xs = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                             torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
    pix = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)
    rays = (torch.inverse(K) @ pix.reshape(-1, 3).T).T.reshape(H, W, 3)
    pts_cam = rays * TRUE_DEPTH
    pts_world = torch.einsum("ij,hwj->hwi", R.T, pts_cam) - R.T @ t
    # texture: a smooth function of world xy so different views see matching content
    return torch.sin(pts_world[..., 0] * 5) * torch.cos(pts_world[..., 1] * 5) * 0.5 + 0.5


def main():
    H, W = 64, 64
    K = torch.tensor([[80.0, 0, W / 2], [0, 80.0, H / 2], [0, 0, 1]], device=device)
    ref = {"K": K, "R": torch.eye(3, device=device), "t": torch.zeros(3, device=device)}
    nb = {"K": K, "R": torch.eye(3, device=device), "t": torch.tensor([0.2, 0.0, 0.0], device=device)}

    ref["img"] = synth_view(ref["K"], ref["R"], ref["t"], H, W)
    nb["img"] = synth_view(nb["K"], nb["R"], nb["t"], H, W)

    depths = torch.linspace(0.5, 4.0, 64, device=device)
    depth_map, cost_map, _ = plane_sweep(ref, [nb], depths)

    center = depth_map[H // 2 - 4:H // 2 + 4, W // 2 - 4:W // 2 + 4]
    err = (center - TRUE_DEPTH).abs().mean().item()
    assert err < 0.2, f"plane-sweep depth error too high near image center: {err}"

    xyz, nrm, _ = backproject(ref, depth_map, ref["img"])
    z = xyz[:, 2].reshape(H, W)[H // 2 - 4:H // 2 + 4, W // 2 - 4:W // 2 + 4]
    assert (z - TRUE_DEPTH).abs().mean().item() < 0.2, "backprojection does not match recovered depth"

    print("OK: plane-sweep + backprojection recover a known plane within tolerance"
          f" (depth err={err:.3f})")


if __name__ == "__main__":
    main()
