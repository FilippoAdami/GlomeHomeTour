#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.random_background = True
        self.data_device = "cuda"
        self.eval = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.sparse_adam = False
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0
        self.lambda_normal = 0.05
        # Upstream hardcoded this at 7000, sized for a 30k run. Exposed so a
        # shorter schedule can bring normal consistency forward; the default
        # keeps the original behaviour for anything not passing it.
        self.normal_from_iter = 7_000
        self.opacity_cull = 0.05

        self.densification_interval = 50
        self.opacity_reset_interval = 3000
        self.opacity_reset_until_iter = -1
        self.opacity_reset_iter = -1      # Targeted iteration for opacity reset (e.g. 6,000); -1 uses interval-based
        self.opacity_recovery_iters = 800 # Steps after opacity reset where opacity culling is bypassed
        self.densify_warmup_per_stage = 300 # Grace period after resuming a stage where densification is paused
        self.densify_from_iter = 500
        self.densify_until_iter = 8_000
        self.densify_grad_threshold = 0.0001
        self.cleanup_interval = 250      # prune-only cadence after densify_until_iter
        self.size_prune_from_iter = 1500  # own gate; used to ride on opacity_reset_interval
        self.max_world_size = 0.035       # Max surfel radius in meters (3.5 cm) to strictly prevent bloaters
        # Surfel budget = growth_factor x the initialisation count. The init cloud is a
        # fixed-voxel-size depth fusion, so it already scales with the property; an
        # absolute budget does not.
        self.growth_factor = 1.6
        # -1 = derive from the initialisation point count at train start
        # (see training() in train.py). 16 GB holds ~4.5-6M surfels.
        self.max_gaussians = -1

        # PGSR-style multi-view planar consistency.
        self.lambda_multiview = 0.3
        self.mv_num_neighbors = 2
        self.mv_from_iter = 500
        self.mv_veto_weight = 0.5       # Asymmetric multi-view veto weight (0.0: standard mean, >0: penalize worst mismatch)

        # Optical bloom / sensor saturation loss masking
        self.saturation_threshold = 0.98            # Dynamic floor in [0, 1] (<= 0 to disable)
        self.saturation_percentile = 99.8
        self.saturation_max_chroma_diff = 0.15

        # SAD-2DGS structure-aware densification (2dgs_combined_pipeline.md §4).
        # Needs the Lambda_min maps in the prior cache; without them the
        # gradient-ranked path below runs alone, exactly as before.
        self.sad_densify = False
        self.sad_tau_split = 0.75        # fraction of views that must report eta > 1
        self.sad_min_views = 4           # views needed before a verdict counts
        # Asymmetric free-space ray carving (§5.3).
        self.freespace_carve = False
        self.freespace_margin = 0.04     # metres; band counted as "on the surface"
        self.freespace_min_views = 3     # free-space sightings needed to prune
        self.freespace_ratio = 0.6       # free / (free + surface) above which it goes
        self.freespace_from_iter = 1_000 # needs the rendered depth to mean something
        # §5.2 maximum screen radius, as a fraction of min(H, W). Relative, not
        # the absolute 20 px this used to be: 20 px is 4.8% of a 420p frame and
        # 1.9% of a 1080p one, so the same surfel passed at the resolution where
        # geometry is settled and was culled at the one where it is finalised.
        self.max_screen_frac = 0.15
        # Phase 3 geometry freeze (§3): the last N steps optimise appearance
        # only, so full-resolution pixel residuals polish colour instead of
        # nudging a surface that is already converged. 0 disables.
        self.freeze_geometry_last = 0

        # Multi-view evidence & spatial culling (TIDI-GS style)
        self.view_evidence_from_iter = 1500         # Start after initial coarse geometry (tuned for 10k-30k runs)
        self.view_evidence_frustum_min = 8          # Only cull if in frustum of >= 8 camera views
        self.view_evidence_min_obs = 2              # Cull if actively contributing in <= 2 views

        # Per-pixel monocular depth/normal priors from 02_depth_estimation
        # (utils/depth_prior.py). Off by default so a scene with no prior cache
        # trains exactly as before; step_train.py turns them on.
        self.depth_prior_dir = ""        # "" = <source_path>/03_2DGS_training/depth_priors
        self.lambda_depth_prior = 0.0
        self.lambda_normal_prior = 0.0
        # The prior pins the geometry while the photometric terms are still
        # useless, then hands over. Decaying to a floor rather than to zero
        # keeps textureless walls from drifting once it has.
        self.prior_decay_from_iter = 1_000
        self.prior_decay_until_iter = 2_500
        self.prior_weight_floor = 0.1

        # Unbiased-Depth-for-2DGS surface criterion: pulls expected depth onto the
        # rasteriser's Eq. (9) channel.
        self.lambda_depth_conv = 0.0
        self.depth_conv_from_iter = 500

        # DN-Splatter edge-aware depth smoothness (utils/loss_utils.smooth_loss).
        self.lambda_depth_smooth = 0.0

        # PGSR exposure compensation (per-image affine on the rendered colour).
        self.optimize_exposure = False
        self.exposure_lr = 0.001
        self.lambda_exposure = 0.01     # anchors (gain, bias) near identity

        # TrackGS-style pose refinement (off by default; highest-risk piece).
        self.refine_poses_during_training = False
        self.pose_lr = 0.0001           # ~100x below position_lr_init: poses nudge, not search
        self.lambda_track = 0.1
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
