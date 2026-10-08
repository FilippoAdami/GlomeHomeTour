"""Stage 03/04 entrypoint checks that do not require a GPU."""

import importlib.util
from collections import OrderedDict
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from PIL import Image


_backend = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mesh = _load("step_mesh", _backend / "04_3DGS_to_mesh/step_mesh.py")
train = _load("step_train", _backend / "03_FastGS_DNSplatter/step_train.py")
gsplat_train = _load("gsplat_train", _backend / "03_FastGS_DNSplatter/gsplat_train.py")


def test_masked_depth_losses_match_valid_pixel_reference():
    pred = torch.linspace(1, 3, 30).reshape(5, 6).requires_grad_()
    target = pred.detach().flip(0).clone()
    target[0, 0] = float("nan")
    target[0, 1] = 0
    valid = torch.isfinite(target) & (target > 0)
    p, t = pred[valid], target[valid]
    for mode in ("pearson", "scale_shift", "log_l1", "l1"):
        if mode == "pearson":
            pc, tc = p - p.mean(), t - t.mean()
            expected = 1 - (pc * tc).mean() / (pc.square().mean().sqrt() * tc.square().mean().sqrt() + 1e-6)
        elif mode == "scale_shift":
            pc, tc = p - p.mean(), t - t.mean()
            scale = (pc * tc).mean() / pc.square().mean().clamp_min(1e-6)
            expected = F.l1_loss(scale * pc + t.mean(), t)
        elif mode == "log_l1":
            expected = F.l1_loss(p.log(), t.log())
        else:
            expected = F.l1_loss(p, t)
        actual = gsplat_train._depth_loss(pred, target, mode)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        actual.backward(retain_graph=True)
        assert torch.isfinite(pred.grad).all()
        pred.grad = None
    sparse = torch.zeros_like(target)
    for valid_count in (0, 1):
        if valid_count:
            sparse[0, 0] = 1
        zero_loss = gsplat_train._depth_loss(pred, sparse, "pearson")
        assert zero_loss == 0
        zero_loss.backward(retain_graph=True)
        assert torch.isfinite(pred.grad).all()
        pred.grad = None


def test_image_cache_keeps_pixels_without_redecoding(tmp_path):
    path = tmp_path / "view.png"
    Image.fromarray(np.full((2, 3, 3), 128, dtype=np.uint8)).save(path)
    camera = {"image_path": path, "width": 3, "height": 2}
    first = gsplat_train._image_tensor(camera, torch.device("cpu"))
    path.unlink()
    torch.testing.assert_close(gsplat_train._image_tensor(camera, torch.device("cpu")), first)
    assert camera["rgb"].dtype == torch.uint8


def test_map_cache_keeps_resized_maps_and_missing_files(tmp_path):
    root = tmp_path / "maps"
    root.mkdir()
    path = root / "view.npy"
    np.save(path, np.arange(6, dtype=np.float32).reshape(2, 3))
    camera = {"image_path": tmp_path / "view.png", "width": 6, "height": 4}
    first = gsplat_train._map_tensor(root, camera, 1, torch.device("cpu"))
    path.unlink()
    torch.testing.assert_close(gsplat_train._map_tensor(root, camera, 1, torch.device("cpu")), first)
    assert gsplat_train._map_tensor(root, camera, 3, torch.device("cpu")) is None
    np.save(path, np.zeros((2, 3, 3), dtype=np.float32))
    assert gsplat_train._map_tensor(root, camera, 3, torch.device("cpu")) is None


def test_map_cache_evicts_targets_at_memory_ceiling(tmp_path):
    cache = {"values": OrderedDict(), "bytes": 0, "limit": 24}
    cameras = [{"image_path": tmp_path / f"view{i}.png", "height": 2, "width": 3} for i in range(2)]
    for i, camera in enumerate(cameras):
        np.save(tmp_path / f"view{i}.npy", np.full((2, 3), i, dtype=np.float32))
        gsplat_train._map_tensor(tmp_path, camera, 1, torch.device("cpu"), cache)
    assert cache["bytes"] == 24 and len(cache["values"]) == 1
    assert next(iter(cache["values"]))[1] == cameras[1]["image_path"]
    np.save(tmp_path / "view0.npy", np.full((2, 3), 7, dtype=np.float32))
    torch.testing.assert_close(gsplat_train._map_tensor(tmp_path, cameras[0], 1, torch.device("cpu"), cache),
                              torch.full((2, 3, 1), 7.))


def test_camera_tensor_cache_reuses_device_tensor():
    camera = {"K": np.eye(3, dtype=np.float32)}
    first = gsplat_train._camera_tensor(camera, "K", torch.device("cpu"))
    assert gsplat_train._camera_tensor(camera, "K", torch.device("cpu")).data_ptr() == first.data_ptr()


def test_half_map_cache_preserves_float32_targets(tmp_path):
    data = np.linspace(.1, 4, 24, dtype=np.float16).reshape(4, 6)
    np.save(tmp_path / "view.npy", data)
    camera = {"image_path": tmp_path / "view.png", "height": 4, "width": 6}
    actual = gsplat_train._map_tensor(tmp_path, camera, 1, torch.device("cpu"))
    torch.testing.assert_close(actual, torch.from_numpy(data.astype(np.float32))[..., None], atol=0, rtol=0)
    assert camera["_maps"]["bytes"] == data.nbytes
    camera["height"], camera["width"] = 8, 12
    resized = gsplat_train._map_tensor(tmp_path, camera, 1, torch.device("cpu"))
    expected = F.interpolate(torch.from_numpy(data.astype(np.float32))[None, None], (8, 12), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    torch.testing.assert_close(resized, expected, atol=0, rtol=0)


def test_fused_adam_preserves_dense_updates():
    parameters = [torch.nn.Parameter(torch.tensor([1., 2., 3.])) for _ in range(2)]
    optimizers = [torch.optim.Adam([p], lr=1e-3, eps=1e-15, fused=fused)
                  for p, fused in zip(parameters, (False, True))]
    for gradient in ([.1, -.3, 0.], [0., .2, .1], [-.2, 0., -.1]):
        for p, optimizer in zip(parameters, optimizers):
            p.grad = torch.tensor(gradient)
            optimizer.step()
        torch.testing.assert_close(*parameters)


def test_camera_intrinsic():
    camera = {"width": 640, "height": 480,
              "K": np.array([[320, 0, 320], [0, 400, 240], [0, 0, 1]])}
    intrinsic = mesh.camera_intrinsic(camera, o3d).intrinsic_matrix
    np.testing.assert_allclose(intrinsic, camera["K"])


def test_combined_training_loss_preserves_objective_and_gradients():
    torch.manual_seed(4)
    colors = (torch.rand(1, 8, 9, 4) + 1).requires_grad_()
    alphas = (torch.rand(1, 8, 9, 1) + .1).requires_grad_()
    target = torch.rand(1, 8, 9, 3)
    depth = torch.rand(8, 9, 1) + 1
    normals = F.normalize(torch.randn(8, 9, 3), dim=-1)
    normals[3, 3] = 0
    K = torch.tensor([[[10., 0, 4], [0, 10., 4], [0, 0, 1]]])
    for use_depth, use_normal in ((False, False), (True, False), (False, True), (True, True)):
        expected_depth = colors[..., 3] / alphas[..., 0].clamp_min(1e-10)
        ssim_score = torch.tensor(.7, requires_grad=True)
        expected = .8 * F.l1_loss(colors[..., :3], target) + .2 * (1 - ssim_score)
        if use_depth:
            expected = expected + .1 * gsplat_train._depth_loss(expected_depth[0], depth[..., 0], "pearson")
        if use_normal:
            rendered = gsplat_train._normals_from_depth(expected_depth[0], K[0])
            normal_target = normals[1:-1, 1:-1]
            valid = normal_target.norm(dim=-1) > .5
            expected = expected + .05 * (1 - (rendered[valid] * normal_target[valid]).sum(-1)).mean()
        actual = gsplat_train._training_loss(colors, alphas, target, K, depth if use_depth else None,
                                             normals if use_normal else None, .1, .05, "pearson", ssim_score)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, colors, retain_graph=True)[0],
                                  torch.autograd.grad(expected, colors, retain_graph=True)[0])
        if use_depth or use_normal:
            torch.testing.assert_close(torch.autograd.grad(actual, alphas, retain_graph=True)[0],
                                      torch.autograd.grad(expected, alphas, retain_graph=True)[0])


def test_depth_alpha_mask():
    depth = np.array([[1, 0, np.nan, np.inf], [-1, 9, 2, 3]], dtype=np.float32)
    alpha = np.array([[0.5, 1, 1, 1], [1, 1, 0.49, np.nan]], dtype=np.float32)
    filtered, count = mesh.valid_depth(depth, alpha, 0.5, 10)
    np.testing.assert_array_equal(filtered, [[1, 0, 0, 0], [0, 9, 0, 0]])
    assert count == 2 and np.isfinite(filtered).all()


def test_mesh_uses_new_scene_paths(tmp_path, monkeypatch):
    model = tmp_path / "03_FastGS_DNSplatter"
    model.mkdir()
    (model / "checkpoint.pt").touch()
    seen = {}
    monkeypatch.setattr(mesh, "extract_mesh", lambda workspace, model_path, output, *args: seen.update(
        workspace=workspace, model_path=model_path, output=output))
    assert mesh.main(["--workspace", str(tmp_path)]) == 0
    assert seen == {"workspace": tmp_path, "model_path": model,
                    "output": tmp_path / "04_3DGS_to_mesh"}


def test_force_train_preserves_existing_checkpoints(tmp_path, monkeypatch):
    model = tmp_path / "03_FastGS_DNSplatter"
    checkpoint = model / "checkpoint.pt"
    model.mkdir()
    checkpoint.write_bytes(b"existing checkpoint")
    monkeypatch.setattr(train, "train", lambda **kwargs: None)
    assert train.main(["--workspace", str(tmp_path), "--force"]) == 0
    assert checkpoint.read_bytes() == b"existing checkpoint"


def test_geometry_masks_ignore_uncovered_depth_and_neighbor_normals():
    colors = torch.ones(1, 8, 9, 4, requires_grad=True)
    alphas = torch.zeros(1, 8, 9, 1, requires_grad=True)
    target = colors.detach()[..., :3]
    depth = torch.full((8, 9, 1), 3.)
    normals = torch.zeros(8, 9, 3); normals[..., 2] = 1
    loss = gsplat_train._training_loss(colors, alphas, target, torch.eye(3)[None],
                                     depth, normals, .1, .05, "l1", torch.tensor(1.))
    assert loss == 0
    gc, ga = torch.autograd.grad(loss, (colors, alphas))
    assert torch.isfinite(gc).all() and torch.isfinite(ga).all()
    assert not gc.any() and not ga.any()
    # Verify analytical perspective normals, including off-center intrinsics.
    K = torch.tensor([[20., 0, 3.], [0, 30., 2.], [0, 0, 1.]])
    y, x = torch.meshgrid(torch.arange(8), torch.arange(9), indexing="ij")
    d = 2 + .1 * x + .2 * y
    yy, xx = y[1:-1, 1:-1], x[1:-1, 1:-1]
    want = F.normalize(torch.stack((torch.full_like(xx, -2., dtype=torch.float32),
                                    torch.full_like(yy, -6., dtype=torch.float32),
                                    2 + .2 * xx + .4 * yy - .7), -1), dim=-1)
    torch.testing.assert_close(gsplat_train._normals_from_depth(d, K), want)


def test_fastgs_optimizer_schedule_and_camera_extent():
    due = gsplat_train._optimizer_due
    assert due(15, "means") and not due(15, "shN") and due(16, "shN")
    assert due(15_000, "means") and not due(15_001, "means")
    assert due(15_008, "means") and due(15_008, "shN")
    assert due(20_000, "means") and not due(20_064, "means") and due(20_096, "means")
    cameras = []
    for x in (-2., 2.):
        view = np.eye(4, dtype=np.float32); view[0, 3] = -x
        cameras.append({"viewmat": view})
    assert abs(gsplat_train._scene_scale(cameras) - 2.2) < 1e-6


def test_camera_sampling_visits_each_view_once_per_epoch():
    cameras = [{"id": i} for i in range(17)]
    pending = []
    rng_state = gsplat_train.random.getstate()
    try:
        gsplat_train.random.seed(42)
        epochs = [[gsplat_train._next_camera(cameras, pending) for _ in cameras] for _ in range(3)]
    finally:
        gsplat_train.random.setstate(rng_state)
    for epoch in epochs:
        assert sorted(camera["id"] for camera in epoch) == list(range(17))
        assert all(camera is cameras[camera["id"]] for camera in epoch)
    assert not pending and len(cameras) == 17
    assert [c["id"] for c in epochs[0]] != [c["id"] for c in epochs[1]]


def test_snapshot_uses_standard_stage04_format_and_actual_sh_degree(tmp_path):
    xyz = np.eye(4, 3, dtype=np.float32)
    splats = gsplat_train._make_splats(xyz, np.full_like(xyz, .5), torch.device('cpu'))
    output = gsplat_train._write_model(tmp_path / 'snapshots' / 'iteration_20', splats, 20, 0, 1)
    checkpoint = torch.load(output.parents[2] / 'checkpoint.pt', weights_only=True)
    assert checkpoint['format'] == 'gsplat-stage03-v1'
    assert (checkpoint['iteration'], checkpoint['sh_degree'], checkpoint['resolution']) == (20, 0, 1)
    vertices = gsplat_train.PlyData.read(output)['vertex']
    np.testing.assert_array_equal(np.column_stack([vertices[k] for k in ('x', 'y', 'z')]), xyz)
    for name, parameter in splats.items():
        torch.testing.assert_close(checkpoint['splats'][name], parameter)
