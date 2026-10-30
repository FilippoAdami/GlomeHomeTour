"""CPU checks for multi-view visibility fusion; run with python -m unittest discover tests."""

import types
import unittest
import sys
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    DinoV3ProjFeatureExtractor,
    DinoV3ProjMultiViewFeatureExtractor,
    ProjGridMV,
)
from pixal3d.trainers.flow_matching.mixins import image_conditioned_proj


class MultiViewProjectionTest(unittest.TestCase):
    def test_view_loader_keeps_alpha_aligned_with_rgb(self):
        # Native pipeline imports initialize HIP; view loading itself is CPU-only.
        pipeline_stub = types.ModuleType('pixal3d.pipelines')
        pipeline_stub.Pixal3DMVImageTo3DPipeline = object
        with mock.patch.dict(sys.modules, {'pixal3d.pipelines': pipeline_stub,
                                           'o_voxel': types.ModuleType('o_voxel')}):
            from inference_mv import load_views
            path = Path(__file__).resolve().parents[1] / 'assets/mv_images/example'
            views = load_views(str(path), num_views=2, image_sizes=(64,))
        sys.modules.pop('inference_mv', None)
        rgb, alpha = views['images'][64], views['masks'][64]
        self.assertEqual(tuple(rgb.shape), (1, 2, 3, 64, 64))
        self.assertEqual(tuple(alpha.shape), (1, 2, 1, 64, 64))
        self.assertTrue((rgb <= alpha).all())
        self.assertEqual(alpha[0, 0, 0, 0, 0], 0)

    def test_backbone_is_loaded_once(self):
        backbone = nn.Identity()
        backbone.config = types.SimpleNamespace(patch_size=2, hidden_size=4)
        with mock.patch.object(image_conditioned_proj.DINOv3ViTModel, 'from_pretrained', return_value=backbone) as load:
            first = DinoV3ProjFeatureExtractor('test', image_size=8, grid_resolution=2)
            second = DinoV3ProjFeatureExtractor('test', image_size=8, grid_resolution=2,
                                                 backbone_model=first.model)
        self.assertIs(first.model, second.model)
        load.assert_called_once_with('test')

    def test_projection_rejects_transparent_pixels(self):
        grid = ProjGridMV(grid_resolution=2, image_resolution=8)
        features = torch.ones(1, 8, 8, 1)
        camera = torch.tensor([0.7])
        distance = torch.tensor([2.0])
        scale = torch.ones(1)
        plain = grid(features, camera, distance, scale)
        projected, visible = grid(
            features, camera, distance, scale,
            visibility_mask=torch.zeros(1, 1, 8, 8), return_valid=True,
        )
        torch.testing.assert_close(projected, plain)
        self.assertTrue((visible == 0).all())
        _, visible = grid(
            features, camera, distance, scale,
            visibility_mask=torch.ones(1, 1, 8, 8), return_valid=True,
        )
        self.assertTrue((visible > 0).any())

    def test_average_uses_visible_views_and_keeps_single_view(self):
        extractor = DinoV3ProjMultiViewFeatureExtractor.__new__(DinoV3ProjMultiViewFeatureExtractor)
        nn.Module.__init__(extractor)
        extractor.model = nn.Identity()
        extractor.model.config = types.SimpleNamespace(num_register_tokens=0)
        extractor.proj_grid = ProjGridMV(grid_resolution=2, image_resolution=8)
        extractor.patch_number = 1
        extractor.naf_target_size = (8, 8)
        extractor.use_naf_upsample = False
        extractor.multiview_fusion = 'average'
        extractor.transform = nn.Identity()
        extractor.extract_features = lambda image: image.mean((1, 2, 3))[:, None, None].expand(-1, 2, -1)

        def project(self, patches, image_for_naf, angle, distance, scale, matrix, mask):
            value = patches[:, 0, 0, :].unsqueeze(1).expand(-1, 8, -1).clone()
            weight = mask.mean((1, 2, 3))[:, None].expand(-1, 8)
            return value, weight

        extractor._project_single_view = types.MethodType(project, extractor)
        images = torch.stack((torch.ones(3, 8, 8), torch.full((3, 8, 8), 3.0)))[None]
        masks = torch.stack((torch.ones(1, 8, 8), torch.zeros(1, 8, 8)))[None]
        cameras = extractor.proj_grid.front_view_transform_matrix[None, None].expand(1, 2, 4, 4).clone()
        args = dict(camera_angle_x=torch.full((1, 2), 0.7), distance=torch.full((1, 2), 2.0),
                    mesh_scale=torch.ones(1), transform_matrix=cameras, visibility_mask=masks)
        _, fused = extractor(images, **args)
        torch.testing.assert_close(fused, torch.ones_like(fused))
        _, fallback = extractor(images, **{**args, 'visibility_mask': torch.zeros_like(masks)})
        torch.testing.assert_close(fallback, torch.ones_like(fallback))
        _, single = extractor(images[:, :1], **{k: v[:, :1] if k != 'mesh_scale' else v
                                                  for k, v in args.items()})
        torch.testing.assert_close(single, torch.ones_like(single))


if __name__ == '__main__':
    unittest.main()
