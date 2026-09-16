#
# TrackGS-style learnable pose refinement (opt-in, default OFF).
#
# Two pieces:
#   se3_exp()   the exponential map turning a learnable 6-vector into an SE(3)
#               nudge, applied on top of the COLMAP pose by Camera.refresh_pose().
#   TrackLoss   reprojection error of the COLMAP tracks under the *current*
#               poses. Without it, nothing stops the pose deltas from drifting to
#               whatever makes the photometric loss happy, taking the geometry
#               with them.
#
# Scoped down from the paper on purpose: the 3D anchors are COLMAP's triangulated
# points held fixed, not jointly optimised Gaussian centres. That keeps the term
# a pose regulariser (which is all we want here, since COLMAP already gave us a
# well-conditioned starting point) instead of a second structure optimiser.
#
# Only imports torch at module scope - scene.cameras imports this, and reaching
# into scene.* here would close an import cycle.
#

import torch

# ponytail: keypoints are subsampled per view; raise if pose deltas look noisy.
MAX_TRACKS_PER_VIEW = 512


def se3_exp(delta):
    """(6,) [wx, wy, wz, tx, ty, tz] -> (4, 4) SE(3), differentiable at zero.

    Rotation is the Rodrigues exponential map. Translation is used directly
    rather than through the full V(theta) Jacobian: these deltas are meant to
    nudge a pose by millimetres, where the two agree to first order.
    """
    w, t = delta[:3], delta[3:6]
    theta = torch.linalg.norm(w).clamp_min(1e-8)
    k = torch.stack([
        torch.stack([torch.zeros_like(w[0]), -w[2], w[1]]),
        torch.stack([w[2], torch.zeros_like(w[0]), -w[0]]),
        torch.stack([-w[1], w[0], torch.zeros_like(w[0])]),
    ])
    eye = torch.eye(3, dtype=delta.dtype, device=delta.device)
    # sin(t)/t and (1-cos t)/t^2 stay finite as theta -> 0, and k, k@k vanish
    # there, so the gradient at delta = 0 is well defined.
    R = eye + (torch.sin(theta) / theta) * k + ((1 - torch.cos(theta)) / theta ** 2) * (k @ k)

    out = torch.eye(4, dtype=delta.dtype, device=delta.device)
    out = out.clone()
    out[:3, :3] = R
    out[:3, 3] = t
    return out


class TrackLoss:
    """Reprojection error of COLMAP tracks under the currently-optimised poses.

    Reads the sparse model that convert_transforms_to_colmap.py produced and, for
    each training view, keeps the 2D keypoints that have a triangulated 3D point.
    Calling the instance with a camera projects those points with that camera's
    live pose and returns the mean pixel error.
    """

    def __init__(self, source_path, cameras, device="cuda",
                 max_tracks=MAX_TRACKS_PER_VIEW):
        import os
        from scene.colmap_loader import (read_extrinsics_binary, read_extrinsics_text,
                                         read_intrinsics_binary, read_intrinsics_text,
                                         read_points3D_binary, read_points3D_text)

        sparse = os.path.join(source_path, "sparse", "0")
        if not os.path.exists(sparse):
            sparse = os.path.join(source_path, "sparse")
        try:
            extr = read_extrinsics_binary(os.path.join(sparse, "images.bin"))
            intr = read_intrinsics_binary(os.path.join(sparse, "cameras.bin"))
            xyz, _, _ = read_points3D_binary(os.path.join(sparse, "points3D.bin"))
        except Exception:
            extr = read_extrinsics_text(os.path.join(sparse, "images.txt"))
            intr = read_intrinsics_text(os.path.join(sparse, "cameras.txt"))
            xyz, _, _ = read_points3D_text(os.path.join(sparse, "points3D.txt"))

        # read_points3D_* returns a dense array ordered by point id; build the
        # id -> row map the tracks index into.
        point_ids = {}
        try:
            from scene.colmap_loader import read_points3D_text as _  # noqa: F401
        except Exception:
            pass
        ids_sorted = sorted({pid for img in extr.values() for pid in img.point3D_ids if pid != -1})
        for row, pid in enumerate(ids_sorted):
            point_ids[pid] = row

        by_name = {}
        for img in extr.values():
            keep = img.point3D_ids != -1
            if keep.sum() == 0:
                continue
            ids = img.point3D_ids[keep]
            uv = img.xys[keep]
            rows = [point_ids[pid] for pid in ids if pid in point_ids]
            if not rows:
                continue
            rows = rows[:len(uv)]
            uv = uv[:len(rows)]
            if len(rows) > max_tracks:
                step = len(rows) // max_tracks
                rows, uv = rows[::step][:max_tracks], uv[::step][:max_tracks]
            # COLMAP point ids are 1-based and may skip; clamp into range.
            rows = [r for r in rows if r < len(xyz)]
            uv = uv[:len(rows)]
            if not rows:
                continue
            name = os.path.splitext(os.path.basename(img.name))[0]
            by_name[name] = {
                "xyz": torch.tensor(xyz[rows], dtype=torch.float32, device=device),
                "uv": torch.tensor(uv, dtype=torch.float32, device=device),
                "width": intr[img.camera_id].width,
            }

        self.tracks = by_name
        self.device = device
        self.num_views = len(by_name)

    def __call__(self, view):
        entry = self.tracks.get(view.image_name)
        if entry is None:
            return torch.zeros((), device=self.device)

        from utils.graphics_utils import fov2focal
        # COLMAP keypoints are full resolution; training cameras may be downscaled.
        scale = view.image_width / float(entry["width"])
        fx = fov2focal(view.FoVx, view.image_width)
        fy = fov2focal(view.FoVy, view.image_height)

        xyz = entry["xyz"]
        ones = torch.ones((xyz.shape[0], 1), device=xyz.device)
        # world_view_transform is W2C transposed, so right-multiply row vectors.
        cam = torch.cat([xyz, ones], dim=1) @ view.world_view_transform
        z = cam[:, 2].clamp_min(1e-6)
        u = fx * cam[:, 0] / z + view.image_width / 2.0
        v = fy * cam[:, 1] / z + view.image_height / 2.0

        target = entry["uv"] * scale
        in_front = cam[:, 2] > 1e-6
        if in_front.sum() == 0:
            return torch.zeros((), device=self.device)

        err = torch.stack([u, v], dim=1)[in_front] - target[in_front]
        # Huber: a handful of mistriangulated tracks should not dominate.
        return torch.nn.functional.huber_loss(err, torch.zeros_like(err), delta=4.0)
