"""CPU checks for restoring mesh coordinates after texture generation."""

import unittest
from unittest import mock

import numpy as np
from PIL import Image
import trimesh

from pixal3d.pipelines.trellis2_texturing import Trellis2TexturingPipeline


class TexturingTransformTest(unittest.TestCase):
    def test_run_restores_mesh_and_scene_world_coordinates(self):
        mesh = trimesh.creation.box(extents=[2.0, 3.0, 5.0])
        transform = trimesh.transformations.rotation_matrix(np.pi / 3, [1, 2, 3])
        transform[:3, 3] = [10.0, -7.0, 4.0]
        mesh.apply_transform(transform)
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.creation.box(extents=[2.0, 3.0, 5.0]), transform=transform)

        pipeline = Trellis2TexturingPipeline()
        pipeline.models = {'tex_slat_flow_model_1024': object()}
        pipeline.get_cond = mock.Mock(return_value=object())
        pipeline.encode_shape_slat = mock.Mock(return_value=object())
        pipeline.sample_tex_slat = mock.Mock(return_value=object())
        pipeline.decode_tex_slat = mock.Mock(return_value=object())

        def fake_postprocess(normalized, *_args):
            vertices = normalized.vertices.copy()
            vertices[:, [1, 2]] = np.column_stack((vertices[:, 2], -vertices[:, 1]))
            return trimesh.Trimesh(vertices=vertices, faces=normalized.faces, process=False)

        pipeline.postprocess_mesh = mock.Mock(side_effect=fake_postprocess)
        image = Image.new('RGB', (1, 1))
        for input_mesh, expected in ((mesh, mesh), (scene, scene.to_mesh())):
            with self.subTest(input_type=type(input_mesh).__name__):
                output = pipeline.run(input_mesh, image, preprocess_image=False)
                np.testing.assert_allclose(output.vertices, expected.vertices, rtol=0, atol=1e-12)
                np.testing.assert_array_equal(output.faces, expected.faces)

    def test_preprocess_rejects_invalid_geometry(self):
        pipeline = Trellis2TexturingPipeline()
        for vertices in ([], [[0, 0, 0]], [[np.nan, 0, 0]]):
            with self.subTest(vertices=vertices), self.assertRaises(ValueError):
                pipeline.preprocess_mesh(trimesh.Trimesh(vertices=vertices, faces=[], process=False))


if __name__ == '__main__':
    unittest.main()
