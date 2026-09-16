#
# Dense point cloud initialization for scenes with no CUDA (only ROCm/ HIP GPU).
#
# COLMAP's own dense stereo (patch_match_stereo / stereo_fusion) is a hard CUDA
# dependency baked into its C++/CUDA kernels (texture memory, cub) - hipifying
# that code is a real, brittle undertaking, not worth it here. Instead this
# does classic multi-view plane-sweep stereo directly in PyTorch, which this
# project's .venv already runs on this machine's AMD GPU via ROCm (same GPU
# used for 2DGS training), so no new toolchain is needed.
#
# Usage: python3 densify_pointcloud.py -s <scene_dir> [--images images_keyframes]
#
# Reads camera poses/intrinsics from <scene_dir>/sparse/0 (produced by
# convert_transforms_to_colmap.py) and overwrites <scene_dir>/sparse/0/points3D.ply
# with a denser fused point cloud. 2DGS's dataset_readers.py picks up that ply
# directly, no other changes needed.

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData, PlyElement

from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, qvec2rotmat

device = "cuda" if torch.cuda.is_available() else "cpu"  # ROCm PyTorch exposes HIP as "cuda"


def load_cameras(sparse_dir, images_dir):
    extrinsics = read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))
    intrinsics = read_intrinsics_binary(os.path.join(sparse_dir, "cameras.bin"))

    cams = []
    for image in extrinsics.values():
        cam = intrinsics[image.camera_id]
        fx, fy, cx, cy = cam.params[:4]
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
        R = torch.tensor(qvec2rotmat(image.qvec), dtype=torch.float32)
        t = torch.tensor(image.tvec, dtype=torch.float32)
        img_path = os.path.join(images_dir, image.name)
        center = -R.T @ t
        cams.append({"name": image.name, "K": K, "R": R, "t": t, "center": center,
                      "width": cam.width, "height": cam.height, "path": img_path})
    # Sort by name so neighboring frames in the capture sequence end up adjacent in the list.
    cams.sort(key=lambda c: c["name"])
    return cams


def select_neighbors(cams, i, num_neighbors, search_window, max_angle_deg=20.0):
    """Pick stereo neighbors with similar viewing direction, spread across a range of baselines.

    Index-adjacent frames can have near-zero baseline whenever the capture pauses or slows
    down - at that point the parallax between a hypothesis at 1m and one at 8m is sub-pixel
    and plane-sweep can't tell them apart. But a single *uniform* target baseline doesn't work
    either: a scene has both near (~0.3m) and far (~5m) geometry, and one fixed baseline is only
    in-frame for a narrow slice of that depth range - a 0.7m baseline pair, for instance, puts a
    0.3m-depth hypothesis so far outside the neighbor frame that every neighbor rejects it, and
    only the far end of the depth range keeps any votes at all, no matter the real depth.
    A spread from small to large baseline covers near and far depths simultaneously (whichever
    neighbors don't have the right baseline for a given depth are naturally excluded by the
    in-frame check). Viewing direction match still matters first - a frame that happens to be
    at a good distance while facing a different part of the room shares no real overlap.
    """
    lo, hi = max(0, i - search_window), min(len(cams), i + search_window + 1)
    candidates = [j for j in range(lo, hi) if j != i]

    def angle_deg(j):
        rel = cams[j]["R"] @ cams[i]["R"].T
        return torch.rad2deg(torch.acos(torch.clamp((torch.trace(rel) - 1) / 2, -1, 1))).item()

    overlapping = [j for j in candidates if angle_deg(j) <= max_angle_deg]
    pool = overlapping if overlapping else candidates
    pool.sort(key=lambda j: torch.norm(cams[j]["center"] - cams[i]["center"]).item())
    if len(pool) <= num_neighbors:
        return pool
    spread_idx = np.linspace(0, len(pool) - 1, num_neighbors).round().astype(int)
    return [pool[k] for k in sorted(set(spread_idx))]


def load_gray(cam, max_side=640):
    img = Image.open(cam["path"]).convert("L")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(device), scale


def scaled_K(K, scale):
    K = K.clone()
    K[:2] *= scale
    return K


def apply_mat(pts, M):
    """pts @ M.T for a large-batch (..., 3) tensor.

    torch.matmul/einsum silently return wrong (sometimes zero) results on this
    machine's PyTorch 2.10.0+rocm7.0 build for big broadcasted batches like the
    (64,H,W,3) plane-sweep tensors (confirmed against a manual per-point ground
    truth - this is a rocBLAS/composable-kernel bug, not a logic error). Elementwise
    broadcast+sum avoids the GEMM kernel entirely and matches the ground truth.
    """
    return (pts.unsqueeze(-2) * M.view(*([1] * (pts.dim() - 1)), 3, 3)).sum(-1)


def plane_sweep(ref, neighbors, depths):
    """Winner-take-all plane-sweep depth for one reference frame against its neighbors.
    ref/neighbors: dict with 'img' (H,W), 'K','R','t' (already scaled to img size).
    depths: (D,) inverse-depth-sampled candidate depths.
    Returns depth map (H,W) and per-pixel best cost (H,W).
    """
    H, W = ref["img"].shape
    D = depths.shape[0]
    Kr_inv = torch.inverse(ref["K"])

    ys, xs = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                             torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
    pix = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)  # H,W,3
    rays = (Kr_inv @ pix.reshape(-1, 3).T).T.reshape(H, W, 3)  # camera-space ray dirs at depth=1

    cost_sum = torch.zeros(D, H, W, device=device)
    weight_sum = torch.zeros(D, H, W, device=device)

    Rr, tr = ref["R"], ref["t"]
    for nb in neighbors:
        # relative pose taking ref-camera points into nb-camera space
        R_rel = nb["R"] @ Rr.T
        t_rel = nb["t"] - R_rel @ tr

        pts_ref = rays.unsqueeze(0) * depths.view(D, 1, 1, 1)  # D,H,W,3, camera-space at ref
        pts_nb = apply_mat(pts_ref, R_rel) + t_rel.view(1, 1, 1, 3)
        proj = apply_mat(pts_nb, nb["K"])
        valid = proj[..., 2] > 1e-3
        u = proj[..., 0] / proj[..., 2].clamp_min(1e-3)
        v = proj[..., 1] / proj[..., 2].clamp_min(1e-3)

        Hn, Wn = nb["img"].shape
        grid = torch.stack([u / (Wn - 1) * 2 - 1, v / (Hn - 1) * 2 - 1], dim=-1)  # D,H,W,2
        sampled = F.grid_sample(nb["img"].view(1, 1, Hn, Wn).expand(D, 1, Hn, Wn),
                                 grid, align_corners=True, padding_mode="zeros")[:, 0]  # D,H,W

        in_bounds = valid & (grid[..., 0].abs() <= 1) & (grid[..., 1].abs() <= 1)
        cost = (sampled - ref["img"].unsqueeze(0)).abs()
        cost_sum += torch.where(in_bounds, cost, torch.zeros_like(cost))
        weight_sum += in_bounds.float()

    # A single pixel's raw intensity matches equally well at every depth on any flat,
    # textureless surface (most walls/ceilings/floors indoors) - that's not noise to
    # average away, it's the classic aperture-problem ambiguity of stereo matching, and
    # winner-take-all on it just returns ties/noise. A local patch window turns "does this
    # exact pixel match" into "does this neighborhood's texture pattern match", which is
    # actually discriminative, and still leaves genuinely flat regions with a flat cost
    # curve we can detect and reject below instead of silently keeping a wrong depth.
    patch = 7
    cost_sum = F.avg_pool2d(cost_sum.unsqueeze(1), patch, stride=1, padding=patch // 2).squeeze(1)
    weight_sum = F.avg_pool2d(weight_sum.unsqueeze(1), patch, stride=1, padding=patch // 2).squeeze(1)

    cost_vol = cost_sum / weight_sum.clamp_min(1.0)
    cost_vol = torch.where(weight_sum > 0.5, cost_vol, torch.full_like(cost_vol, float("inf")))

    best_cost, best_idx = cost_vol.min(dim=0)
    depth_map = depths[best_idx]
    views = weight_sum.gather(0, best_idx.unsqueeze(0))[0]

    # Peak-ambiguity test: if the best depth isn't clearly better than the "no idea"
    # baseline (typical cost across all hypotheses), the match is ambiguous - textureless
    # surface, repeating pattern, occlusion - and should be dropped rather than kept at
    # whatever depth WTA happened to land on.
    finite = cost_vol.isfinite()
    background = torch.where(finite, cost_vol, torch.zeros_like(cost_vol)).sum(0) / finite.sum(0).clamp_min(1)
    ambiguous = best_cost > 0.6 * background
    best_cost = torch.where(ambiguous, torch.full_like(best_cost, float("inf")), best_cost)

    return depth_map, best_cost, views


def backproject(cam, depth_map, color_img):
    H, W = depth_map.shape
    K_inv = torch.inverse(cam["K"])
    ys, xs = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                             torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
    pix = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)
    rays = (K_inv @ pix.reshape(-1, 3).T).T.reshape(H, W, 3)
    pts_cam = rays * depth_map.unsqueeze(-1)
    pts_world = apply_mat(pts_cam - cam["t"], cam["R"].T)
    # normal from local depth gradient (camera space), then rotated to world
    dzdx = torch.gradient(depth_map, dim=1)[0]
    dzdy = torch.gradient(depth_map, dim=0)[0]
    normals_cam = torch.stack([-dzdx, -dzdy, torch.ones_like(depth_map)], dim=-1)
    normals_cam = F.normalize(normals_cam, dim=-1)
    normals_world = apply_mat(normals_cam, cam["R"].T)
    return pts_world.reshape(-1, 3), normals_world.reshape(-1, 3), color_img.reshape(-1)


def estimate_scene_depth(sparse_dir, cams):
    """Median/near-far depth of the existing triangulated points, seen from these cameras.

    Used to set the plane-sweep depth range and target stereo baseline to this scene's
    actual scale instead of guessing fixed meters - a hardcoded 0.1-8m range is either
    way too shallow (outdoor courtyard) or, as happened here, way too generous for a
    small room, letting degenerate far-plane hypotheses win by default.
    """
    xyz, _, _ = read_points3D_binary(os.path.join(sparse_dir, "points3D.bin"))
    depths = []
    for cam in cams[:: max(1, len(cams) // 30)]:  # a sample of views is enough
        d = (cam["R"].numpy() @ xyz.T).T[:, 2] + cam["t"].numpy()[2]
        depths.append(d[d > 0])
    depths = np.concatenate(depths)
    return float(np.percentile(depths, 5)), float(np.percentile(depths, 95))


def voxel_dedup(xyz, rgb, nrm, voxel_size):
    keys = np.floor(xyz / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    return xyz[unique_idx], rgb[unique_idx], nrm[unique_idx]


def remove_outliers(xyz, rgb, nrm, nb_neighbors, std_ratio):
    """Statistical outlier removal: drops points whose mean distance to their
    nb_neighbors nearest neighbors is more than std_ratio stds from the global
    mean. Catches strays that drift off the main scene regardless of direction,
    unlike a fixed radius/bounding-box cut."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    _, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    inlier_idx = np.asarray(inlier_idx)
    return xyz[inlier_idx], rgb[inlier_idx], nrm[inlier_idx]


def main():
    parser = argparse.ArgumentParser("Plane-sweep dense point cloud initializer (no CUDA required)")
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--num_planes", type=int, default=64)
    parser.add_argument("--num_neighbors", type=int, default=6)
    parser.add_argument("--max_side", type=int, default=640, help="Working resolution for plane sweep")
    parser.add_argument("--cost_thresh", type=float, default=0.05, help="Max mean-abs-diff to keep a pixel")
    parser.add_argument("--min_views", type=int, default=2, help="Min neighbors a pixel must be visible in")
    parser.add_argument("--frame_stride", type=int, default=1, help="Use every Nth frame as a reference view")
    parser.add_argument("--voxel_size", type=float, default=0.1,
                         help="~1cm voxels leave tens of millions of points for a room-scale scene "
                              "(too many Gaussians to init/train); 0.1m keeps ~1-1.5M, still far denser "
                              "than plain point_triangulator output")
    parser.add_argument("--sor_nb_neighbors", type=int, default=20,
                         help="Statistical outlier removal: neighbors per point (0 disables SOR)")
    parser.add_argument("--sor_std_ratio", type=float, default=1.5,
                         help="Statistical outlier removal: std-dev threshold, lower = more aggressive")
    parser.add_argument("--depth_min", type=float, default=None, help="Default: auto, from the sparse point cloud")
    parser.add_argument("--depth_max", type=float, default=None, help="Default: auto, from the sparse point cloud")
    parser.add_argument("--search_window", type=int, default=40,
                         help="How many frames around each reference view to search for stereo neighbors "
                              "(capture speed varies, so index-adjacent frames are not reliably a good baseline)")
    args = parser.parse_args()

    sparse_dir = os.path.join(args.source_path, "sparse", "0")
    images_dir = os.path.join(args.source_path, args.images)
    cams = load_cameras(sparse_dir, images_dir)
    print(f"Loaded {len(cams)} camera poses.")

    near, far = estimate_scene_depth(sparse_dir, cams)
    depth_min = args.depth_min if args.depth_min is not None else max(0.05, near * 0.7)
    depth_max = args.depth_max if args.depth_max is not None else far * 1.5
    print(f"Scene depth range (5th-95th pct): {near:.2f}-{far:.2f}m -> "
          f"using depth_min={depth_min:.2f} depth_max={depth_max:.2f}")

    depths = 1.0 / torch.linspace(1.0 / depth_max, 1.0 / depth_min, args.num_planes, device=device)

    all_xyz, all_rgb, all_nrm = [], [], []
    for i in range(0, len(cams), args.frame_stride):
        ref_cam = cams[i]
        neighbor_idx = select_neighbors(cams, i, args.num_neighbors, args.search_window)
        if not neighbor_idx:
            continue

        if len(neighbor_idx) < 4:
            continue  # need two independent baseline groups below, so at least 2+2

        ref_img, scale = load_gray(ref_cam, args.max_side)
        ref = {"img": ref_img, "K": scaled_K(ref_cam["K"], scale).to(device),
               "R": ref_cam["R"].to(device), "t": ref_cam["t"].to(device)}

        neighbors = []
        for j in neighbor_idx:
            nb_img, nb_scale = load_gray(cams[j], args.max_side)
            neighbors.append({"img": nb_img, "K": scaled_K(cams[j]["K"], nb_scale).to(device),
                               "R": cams[j]["R"].to(device), "t": cams[j]["t"].to(device)})

        # Winner-take-all on one baseline set can lock onto a degenerate "trivial" depth
        # (see module docstring / commit history) that isn't caught by that set's own
        # ambiguity check - it's confidently wrong, not uncertain. Splitting into two
        # independent baseline groups and requiring their depths to agree kills that: a
        # spurious tie in group A has no reason to reproduce in group B's different
        # baselines, while a real surface point triangulates the same way in both.
        depth_a, cost_a, views_a = plane_sweep(ref, neighbors[0::2], depths)
        depth_b, cost_b, views_b = plane_sweep(ref, neighbors[1::2], depths)
        agree = (depth_a - depth_b).abs() < 0.1 * torch.minimum(depth_a, depth_b)

        depth_map, cost_map, views_map = plane_sweep(ref, neighbors, depths)
        mask = (cost_map < args.cost_thresh) & (views_map >= args.min_views) & agree

        rgb_img = torch.from_numpy(np.asarray(
            Image.open(ref_cam["path"]).convert("RGB").resize(
                (ref_img.shape[1], ref_img.shape[0]), Image.BILINEAR), dtype=np.float32)).to(device)

        xyz, nrm, _ = backproject(ref, depth_map, ref_img)
        xyz, nrm = xyz[mask.reshape(-1)], nrm[mask.reshape(-1)]
        rgb = rgb_img.reshape(-1, 3)[mask.reshape(-1)]

        all_xyz.append(xyz.cpu().numpy())
        all_nrm.append(nrm.cpu().numpy())
        all_rgb.append(rgb.cpu().numpy())
        print(f"[{i + 1}/{len(cams)}] {ref_cam['name']}: kept {mask.sum().item()} / {mask.numel()} pixels")

    xyz = np.concatenate(all_xyz, axis=0)
    nrm = np.concatenate(all_nrm, axis=0)
    rgb = np.concatenate(all_rgb, axis=0).clip(0, 255).astype(np.uint8)
    print(f"Fused {len(xyz)} points before dedup, voxel size {args.voxel_size}")

    xyz, rgb, nrm = voxel_dedup(xyz, rgb, nrm, args.voxel_size)
    print(f"{len(xyz)} points after voxel dedup.")

    if args.sor_nb_neighbors > 0:
        n_before = len(xyz)
        xyz, rgb, nrm = remove_outliers(xyz, rgb, nrm, args.sor_nb_neighbors, args.sor_std_ratio)
        print(f"{len(xyz)} points after statistical outlier removal "
              f"(dropped {n_before - len(xyz)}, nb_neighbors={args.sor_nb_neighbors}, "
              f"std_ratio={args.sor_std_ratio}).")

    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    elements = np.empty(len(xyz), dtype=dtype)
    elements[:] = list(map(tuple, np.concatenate([xyz, nrm, rgb], axis=1)))
    out_path = os.path.join(sparse_dir, "points3D.ply")
    PlyData([PlyElement.describe(elements, "vertex")]).write(out_path)
    print(f"Wrote dense point cloud to {out_path}")


if __name__ == "__main__":
    main()
