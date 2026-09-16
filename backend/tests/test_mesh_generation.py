"""GlomeHomeTour: Unit Tests for Mesh Generation Pipeline (Phase 1).

Covers:
1. Pydantic and dataclass models (BoundingBox3D, PBRMaterial, MeshNode, InstanceCluster, MeshPipelineConfig)
2. Schema contract validation for mesh_manifest.schema.json and fixtures
3. PyTorch ROCm SurfelCloudTorch operations (tangent frame, slicing, bounding box)
4. PLY binary I/O roundtrip and coordinate system adherence (+Y up, -Z forward, metric scale)
5. transforms.json loader, camera projection models, and matrix inversion integrity
"""

import json
import sys
from pathlib import Path
import tempfile

# Ensure both repo root and backend/ are in sys.path
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_DIR.parent
for p in [str(_BACKEND_DIR), str(_REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from jsonschema import Draft202012Validator
import numpy as np
import pytest
import torch

from mesh_types import (
    BoundingBox3D,
    InstanceCluster,
    MeshManifest,
    MeshNode,
    MeshPipelineConfig,
    NodeCategory,
    PBRMaterial,
    SurfelCloudTorch,
)
from io_adapter import (
    CameraIntrinsicsTorch,
    KeyframePoseTorch,
    TransformsDataset,
    create_mock_surfel_cloud,
    create_mock_transforms,
    load_splats_ply,
    load_transforms_json,
    save_splats_ply,
)


SCHEMAS_DIR = Path(__file__).resolve().parent.parent.parent / "shared" / "schemas"


def test_bounding_box_3d():
    """Test 3D bounding box geometry, volume, and containment."""
    pts = np.array([
        [-1.0, 0.0, -2.0],
        [3.0, 2.5, 4.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)

    bbox = BoundingBox3D.from_points(pts)
    assert bbox.min_point == (-1.0, 0.0, -2.0)
    assert bbox.max_point == (3.0, 2.5, 4.0)
    assert bbox.extents == (4.0, 2.5, 6.0)
    assert bbox.center == (1.0, 1.25, 1.0)
    assert pytest.approx(bbox.volume, 1e-5) == 60.0

    assert bbox.contains_point([0.0, 1.0, 0.0])
    assert bbox.contains_point([-1.0, 0.0, -2.0])
    assert not bbox.contains_point([5.0, 1.0, 0.0])


def test_pbr_material_defaults_and_validation():
    """Test PBR material defaults and range constraints."""
    mat = PBRMaterial(
        albedo_texture="textures/albedo.png",
        roughness_factor=0.65,
        metallic_factor=0.1,
    )
    assert mat.albedo_texture == "textures/albedo.png"
    assert mat.roughness_factor == 0.65
    assert mat.metallic_factor == 0.1
    assert mat.base_color_factor == (1.0, 1.0, 1.0, 1.0)
    assert not mat.double_sided

    with pytest.raises(Exception):
        PBRMaterial(roughness_factor=1.5)


def test_mesh_node_hierarchy():
    """Test scene graph node hierarchy and matrix validation."""
    bbox = BoundingBox3D(
        min_point=(-1.0, 0.0, -1.0),
        max_point=(1.0, 2.0, 1.0),
        center=(0.0, 1.0, 0.0),
        extents=(2.0, 2.0, 2.0),
    )

    child_node = MeshNode(
        name="Chair_001",
        category=NodeCategory.FURNITURE,
        instance_id="obj_chair_001",
        polygon_count=1200,
        vertex_count=650,
        bounding_box=bbox,
        cad_layer="FF-FURN",
    )

    root_node = MeshNode(
        name="Architecture",
        category=NodeCategory.ARCHITECTURE,
        polygon_count=4500,
        vertex_count=2300,
        bounding_box=bbox,
        children=[child_node],
    )

    assert len(root_node.children) == 1
    assert root_node.children[0].name == "Chair_001"
    assert root_node.children[0].cad_layer == "FF-FURN"

    # Test invalid 4x4 matrix
    with pytest.raises(ValueError):
        MeshNode(
            name="Invalid",
            category=NodeCategory.UNCLASSIFIED,
            bounding_box=bbox,
            transform_matrix=[[1.0, 0.0], [0.0, 1.0]],
        )


def test_pipeline_config_hardware_constraints():
    """Verify hardware configuration matches RX 9070 XT and ROCm constraints."""
    config = MeshPipelineConfig()
    assert config.target_arch == "gfx1200"
    assert config.wavefront_size == 32
    assert config.lds_max_bytes <= 32768
    assert config.compositor_yield_seconds == 0.18
    assert config.texture_resolution == 2048


def test_mesh_manifest_schema_validation():
    """Validate mesh_manifest.schema.json against Draft 2020-12 and fixture."""
    schema_path = SCHEMAS_DIR / "mesh_manifest.schema.json"
    assert schema_path.exists(), f"Schema not found: {schema_path}"

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)

    fixture_path = SCHEMAS_DIR / "fixtures" / "mesh_manifest.example.json"
    assert fixture_path.exists(), f"Fixture not found: {fixture_path}"

    instance = json.loads(fixture_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(instance))
    assert len(errors) == 0, f"Validation errors: {[e.message for e in errors]}"

    # Verify Pydantic model can parse the fixture directly
    manifest_model = MeshManifest.model_validate(instance)
    assert manifest_model.scene_id == "listing_test_scan_001"
    assert manifest_model.units == "meters"
    assert len(manifest_model.nodes) == 2


def test_surfel_cloud_torch_tangents():
    """Verify right-handed orthonormal tangent frame construction."""
    n = 20
    positions = torch.randn(n, 3, dtype=torch.float32)
    normals = torch.randn(n, 3, dtype=torch.float32)
    normals = normals / torch.linalg.norm(normals, dim=-1, keepdim=True)
    scales = torch.full((n, 2), 0.03, dtype=torch.float32)
    colors = torch.rand(n, 3, dtype=torch.float32)
    opacities = torch.ones(n, 1, dtype=torch.float32)

    cloud = SurfelCloudTorch(
        positions=positions,
        normals=normals,
        scales_2d=scales,
        colors_rgb=colors,
        opacities=opacities,
    )

    u = cloud.tangent_u
    v = cloud.tangent_v
    assert u is not None and v is not None

    # Orthogonality checks: u . n == 0, v . n == 0, u . v == 0
    dot_un = torch.sum(u * normals, dim=-1)
    dot_vn = torch.sum(v * normals, dim=-1)
    dot_uv = torch.sum(u * v, dim=-1)

    assert torch.allclose(dot_un, torch.zeros_like(dot_un), atol=1e-5)
    assert torch.allclose(dot_vn, torch.zeros_like(dot_vn), atol=1e-5)
    assert torch.allclose(dot_uv, torch.zeros_like(dot_uv), atol=1e-5)

    # Unit length checks
    assert torch.allclose(torch.linalg.norm(u, dim=-1), torch.ones(n), atol=1e-5)
    assert torch.allclose(torch.linalg.norm(v, dim=-1), torch.ones(n), atol=1e-5)


def test_surfel_cloud_slice_and_bbox():
    """Verify slicing and bounding box extraction."""
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [-1.0, -1.0, -1.0],
        [0.5, 0.5, 0.5],
    ], dtype=torch.float32)

    normals = torch.tensor([[0.0, 1.0, 0.0]] * 4, dtype=torch.float32)
    scales = torch.full((4, 2), 0.02, dtype=torch.float32)
    colors = torch.full((4, 3), 0.8, dtype=torch.float32)
    opacities = torch.ones(4, 1, dtype=torch.float32)

    cloud = SurfelCloudTorch(
        positions=positions,
        normals=normals,
        scales_2d=scales,
        colors_rgb=colors,
        opacities=opacities,
    )

    sub = cloud.slice([0, 1])
    assert len(sub) == 2
    assert torch.allclose(sub.positions[1], torch.tensor([1.0, 2.0, 3.0]))

    bbox = sub.compute_bounding_box()
    assert bbox.min_point == (0.0, 0.0, 0.0)
    assert bbox.max_point == (1.0, 2.0, 3.0)


def test_mock_surfel_cloud_generation_and_ply_roundtrip():
    """Verify mock surfel generation, PLY writing, and PLY loading roundtrip."""
    cloud = create_mock_surfel_cloud(
        num_surfels=600,
        room_dimensions=(4.0, 2.5, 3.0),
        num_furniture_objects=1,
        device="cpu",
    )
    assert len(cloud) > 0
    assert cloud.positions.shape[1] == 3
    assert cloud.normals.shape[1] == 3

    # Coordinate system checks: +Y up, dimensions within bounds
    bbox = cloud.compute_bounding_box()
    assert bbox.min_point[1] >= -0.05  # Floor is near Y = 0
    assert bbox.max_point[1] <= 2.55   # Ceiling is near Y = 2.5

    with tempfile.TemporaryDirectory() as tmpdir:
        ply_path = Path(tmpdir) / "test_splats.ply"
        save_splats_ply(cloud, ply_path)
        assert ply_path.exists()
        assert ply_path.stat().st_size > 0

        # Load back
        loaded = load_splats_ply(ply_path, device="cpu")
        assert len(loaded) == len(cloud)
        assert torch.allclose(loaded.positions, cloud.positions, atol=1e-4)
        assert torch.allclose(loaded.normals, cloud.normals, atol=1e-3)
        assert torch.allclose(loaded.scales_2d, cloud.scales_2d, atol=1e-4)
        assert torch.allclose(loaded.colors_rgb, cloud.colors_rgb, atol=1.0 / 255.0 + 1e-4)
        assert torch.allclose(loaded.opacities, cloud.opacities, atol=1e-3)


def test_transforms_json_loader_and_camera_model():
    """Verify mock transforms generation, schema conformance, and dataset loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "transforms.json"
        raw_dict = create_mock_transforms(
            num_frames=6,
            room_dimensions=(4.0, 2.6, 3.5),
            output_path=json_path,
        )

        # Validate against transforms.schema.json
        t_schema = json.loads((SCHEMAS_DIR / "transforms.schema.json").read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(t_schema)
        v = Draft202012Validator(t_schema)
        errors = list(v.iter_errors(raw_dict))
        assert len(errors) == 0, f"transforms.json schema errors: {[e.message for e in errors]}"

        # Load into dataset
        ds = load_transforms_json(json_path, device="cpu")
        assert ds.camera_model == "OPENCV"
        assert ds.intrinsics.width == 1080
        assert ds.intrinsics.height == 1920
        assert len(ds.frames) == 6

        c2w_all = ds.get_c2w_batch()
        w2c_all = ds.get_w2c_batch()
        assert c2w_all.shape == (6, 4, 4)
        assert w2c_all.shape == (6, 4, 4)

        # Inversion check: w2c * c2w should be identity
        prod = torch.matmul(w2c_all, c2w_all)
        eye_batch = torch.eye(4).unsqueeze(0).repeat(6, 1, 1)
        assert torch.allclose(prod, eye_batch, atol=1e-4)


def test_device_transfer_and_vram_cleanup():
    """Test device transfer (CPU / ROCm if available) and cache cleanup."""
    cloud = create_mock_surfel_cloud(num_surfels=100, device="cpu")
    assert cloud.device.type == "cpu"

    if torch.cuda.is_available():
        cloud_gpu = cloud.to("cuda")
        assert cloud_gpu.device.type == "cuda"
        torch.cuda.empty_cache()


def test_geometric_architecture_segmentation():
    """Test geometric surface normal filter for separating floor, ceiling, wall, and furniture."""
    from segmentation import GeometricArchitectureFilter

    cloud = create_mock_surfel_cloud(
        num_surfels=1200,
        room_dimensions=(4.0, 2.5, 3.0),
        num_furniture_objects=1,
        device="cpu",
    )
    arch_filter = GeometricArchitectureFilter()
    masks = arch_filter.classify_architecture(cloud)

    assert "floor" in masks
    assert "ceiling" in masks
    assert "wall" in masks
    assert "architecture" in masks
    assert "foreground" in masks

    # Verify counts
    n_total = len(cloud)
    n_floor = int(torch.sum(masks["floor"]).item())
    n_ceiling = int(torch.sum(masks["ceiling"]).item())
    n_wall = int(torch.sum(masks["wall"]).item())
    n_fg = int(torch.sum(masks["foreground"]).item())

    assert n_floor > 50
    assert n_ceiling > 50
    assert n_wall > 100
    assert n_fg > 50
    assert n_floor + n_ceiling + n_wall + n_fg >= n_total - 10


def test_semantic_instance_segmenter_pipeline():
    """Test multi-view projection and foreground instance clustering."""
    import tempfile
    from segmentation import SemanticInstanceSegmenter

    cloud = create_mock_surfel_cloud(
        num_surfels=1500,
        room_dimensions=(4.0, 2.5, 3.0),
        num_furniture_objects=1,
        device="cpu",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "transforms.json"
        create_mock_transforms(
            num_frames=8,
            room_dimensions=(4.0, 2.5, 3.0),
            output_path=json_path,
        )
        dataset = load_transforms_json(json_path, device="cpu")

        config = MeshPipelineConfig(min_cluster_surfels=20)
        segmenter = SemanticInstanceSegmenter(config=config)

        # Cluster into architecture vs foreground instances
        arch_masks, clusters = segmenter.cluster_foreground_instances(
            surfels=cloud,
            transforms=dataset,
        )

        assert torch.sum(arch_masks["architecture"]) > 0
        assert len(clusters) >= 1

        first_cluster = clusters[0]
        assert first_cluster.cluster_id.startswith("cluster_")
        assert len(first_cluster.surfel_indices) >= 20
        assert first_cluster.bounding_box.volume > 0.0


def test_manhattan_planar_snapping():
    """Test orthogonal 90-degree Manhattan plane fitting and zero angular deviation."""
    from planar_snapping import ManhattanPlanarRANSAC

    cloud = create_mock_surfel_cloud(
        num_surfels=1500,
        room_dimensions=(5.0, 2.8, 4.0),
        num_furniture_objects=0,
        device="cpu",
    )
    ransac = ManhattanPlanarRANSAC()

    floor, ceiling = ransac.fit_floor_and_ceiling(cloud)
    assert floor.axis == "Y"
    assert ceiling.axis == "Y"
    assert floor.normal == (0.0, 1.0, 0.0)
    assert ceiling.normal == (0.0, -1.0, 0.0)
    assert pytest.approx(-floor.d, abs=0.1) == 0.0
    assert pytest.approx(ceiling.d, abs=0.1) == 2.8

    walls = ransac.fit_manhattan_walls(cloud)
    assert "wall_neg_x" in walls
    assert "wall_pos_x" in walls
    assert "wall_neg_z" in walls
    assert "wall_pos_z" in walls

    assert walls["wall_neg_x"].normal == (1.0, 0.0, 0.0)
    assert walls["wall_pos_x"].normal == (-1.0, 0.0, 0.0)
    assert walls["wall_neg_z"].normal == (0.0, 0.0, 1.0)
    assert walls["wall_pos_z"].normal == (0.0, 0.0, -1.0)


def test_architectural_infill_and_watertight_shell():
    """Test void detection, inpaint surfels, and watertight manifold shell extraction."""
    from segmentation import GeometricArchitectureFilter
    from architectural_infill import ArchitecturalInfillEngine

    cloud = create_mock_surfel_cloud(
        num_surfels=2000,
        room_dimensions=(4.5, 2.6, 3.8),
        num_furniture_objects=1,
        device="cpu",
    )

    arch_filter = GeometricArchitectureFilter()
    masks = arch_filter.classify_architecture(cloud)

    arch_cloud = cloud.slice(torch.nonzero(masks["architecture"], as_tuple=True)[0])
    fg_cloud = cloud.slice(torch.nonzero(masks["foreground"], as_tuple=True)[0])

    infill_engine = ArchitecturalInfillEngine(grid_resolution=0.08)
    result = infill_engine.process_architectural_infill(
        arch_surfels=arch_cloud,
        foreground_surfels=fg_cloud,
    )

    assert result.num_infilled_points > 0
    assert len(result.infilled_surfels) > len(arch_cloud)
    assert len(result.void_bounds) >= 1

    # Check watertight manifold shell mesh
    shell = result.shell_mesh
    assert shell is not None
    assert shell.is_watertight, "Architectural shell mesh must be strictly watertight"
    assert len(shell.faces) > 0
    assert len(shell.vertices) > 0

    # Verify shell volume matches room dimensions
    expected_vol = 4.5 * 2.6 * 3.8
    assert pytest.approx(shell.volume, rel=0.15) == expected_vol


def test_prototype_cluster_deduplication():
    """Verify that identical furniture items (e.g. 4 chairs + 2 tables) are de-duplicated into 2 prototypes."""
    from prototype_cluster import PrototypeClusterEngine

    # 4 identical blue dining chairs at different room locations
    chair_indices = list(range(0, 100))
    table_indices = list(range(100, 200))

    # Synthetic surfel cloud with blue chairs and brown tables
    n_pts = 200
    pos = torch.randn(n_pts, 3)
    norm = torch.tensor([[0.0, 1.0, 0.0]] * n_pts)
    sc = torch.full((n_pts, 2), 0.03)

    col = torch.zeros(n_pts, 3)
    col[:100] = torch.tensor([0.2, 0.3, 0.8])  # Blue chairs
    col[100:] = torch.tensor([0.6, 0.4, 0.2])  # Brown tables
    op = torch.ones(n_pts, 1)

    cloud = SurfelCloudTorch(positions=pos, normals=norm, scales_2d=sc, colors_rgb=col, opacities=op)

    clusters = []
    # 4 chairs
    for i in range(4):
        bbox = BoundingBox3D(
            min_point=(-0.25 + i * 0.8, 0.0, -0.25),
            max_point=(0.25 + i * 0.8, 0.9, 0.25),
            center=(i * 0.8, 0.45, 0.0),
            extents=(0.5, 0.9, 0.5),
        )
        clusters.append(
            InstanceCluster(
                cluster_id=f"chair_{i+1:03d}",
                category=NodeCategory.FURNITURE,
                surfel_indices=chair_indices,
                bounding_box=bbox,
                centroid=bbox.center,
                confidence=0.95,
            )
        )

    # 2 tables
    for j in range(2):
        bbox = BoundingBox3D(
            min_point=(-0.6, 0.0, -0.6 + j * 2.0),
            max_point=(0.6, 0.75, 0.6 + j * 2.0),
            center=(0.0, 0.375, j * 2.0),
            extents=(1.2, 0.75, 1.2),
        )
        clusters.append(
            InstanceCluster(
                cluster_id=f"table_{j+1:03d}",
                category=NodeCategory.FURNITURE,
                surfel_indices=table_indices,
                bounding_box=bbox,
                centroid=bbox.center,
                confidence=0.92,
            )
        )

    engine = PrototypeClusterEngine()
    prototypes = engine.cluster_prototypes(clusters, cloud)

    # 6 items must be grouped into exactly 2 unique prototypes
    assert len(prototypes) == 2
    # Verify deduction ratio: (6 - 2) / 6 = 66.7% reduction
    reduction = (len(clusters) - len(prototypes)) / len(clusters)
    assert reduction >= 0.65


def test_canonical_view_renderer():
    """Verify that CanonicalViewRenderer outputs clean 4-view projection images."""
    from object_reconstruction import CanonicalViewRenderer
    from prototype_cluster import PrototypeGroup

    cloud = create_mock_surfel_cloud(num_surfels=500, device="cpu")
    bbox = cloud.compute_bounding_box()
    proto = PrototypeGroup(
        prototype_id="proto_test_001",
        category="chair",
        representative_instance_id="inst_001",
        instance_ids=["inst_001"],
        representative_surfel_indices=list(range(200)),
        bounding_box=bbox,
    )

    renderer = CanonicalViewRenderer(image_resolution=256)
    views = renderer.render_prototype_views(cloud, proto)

    assert views.front.size == (256, 256)
    assert views.side.size == (256, 256)
    assert views.top.size == (256, 256)
    assert views.isometric.size == (256, 256)


def test_pixal3d_reconstruction_and_obb_alignment():
    """Test Pixal3D reconstruction engine and OBB instance cloning."""
    from object_reconstruction import (
        CanonicalViewRenderer,
        OBBAligner,
        Pixal3DReconstructionEngine,
    )
    from prototype_cluster import PrototypeGroup

    cloud = create_mock_surfel_cloud(num_surfels=500, device="cpu")
    bbox = BoundingBox3D(
        min_point=(-0.3, 0.0, -0.3),
        max_point=(0.3, 0.85, 0.3),
        center=(0.0, 0.425, 0.0),
        extents=(0.6, 0.85, 0.6),
    )
    proto = PrototypeGroup(
        prototype_id="proto_chair_001",
        category="chair",
        representative_instance_id="chair_001",
        instance_ids=["chair_001", "chair_002"],
        representative_surfel_indices=list(range(150)),
        bounding_box=bbox,
    )

    engine = Pixal3DReconstructionEngine(config=MeshPipelineConfig(compositor_yield_seconds=0.0))
    mesh, material = engine.reconstruct_prototype(proto)

    assert mesh is not None
    assert mesh.is_watertight, "Reconstructed prototype mesh must be watertight"
    assert len(mesh.faces) > 0

    instances = [
        InstanceCluster(
            cluster_id="chair_001",
            category=NodeCategory.FURNITURE,
            surfel_indices=list(range(100)),
            bounding_box=bbox,
            centroid=(0.0, 0.425, 0.0),
            confidence=0.95,
        ),
        InstanceCluster(
            cluster_id="chair_002",
            category=NodeCategory.FURNITURE,
            surfel_indices=list(range(100)),
            bounding_box=BoundingBox3D(
                min_point=(1.0, 0.0, -0.3),
                max_point=(1.6, 0.85, 0.3),
                center=(1.3, 0.425, 0.0),
                extents=(0.6, 0.85, 0.6),
            ),
            centroid=(1.3, 0.425, 0.0),
            confidence=0.95,
        ),
    ]

    nodes = OBBAligner.align_and_clone_instances(proto, mesh, material, instances)
    assert len(nodes) == 2
    assert nodes[0].instance_id == "chair_001"
    assert nodes[1].instance_id == "chair_002"
    assert nodes[1].transform_matrix[0][3] == 1.3  # Translation X matches centroid


def test_uv_unwrapping_and_atlas_bounds():
    """Verify non-overlapping UV atlas unwrap in [0, 1]x[0, 1]."""
    from texture_baking import UVAtlasUnwrapper
    import trimesh

    box = trimesh.creation.box(extents=[1.0, 2.0, 1.5])
    unwrapper = UVAtlasUnwrapper(padding=0.01)
    unwrapped, uvs = unwrapper.unwrap_mesh(box)

    assert unwrapped is not None
    assert len(uvs) == len(unwrapped.vertices)
    # Check all UVs lie within valid normalized atlas space [0.0, 1.0]
    assert np.all(uvs >= 0.0)
    assert np.all(uvs <= 1.0)
    assert uvs.shape[1] == 2


def test_pbr_texture_baking_and_material_export():
    """Verify PBR texture baking (Albedo, Roughness, Metallic, Normal) and map saving."""
    from texture_baking import PBRTextureBaker
    import tempfile
    import trimesh

    box = trimesh.creation.box(extents=[0.8, 0.8, 0.8])
    baker = PBRTextureBaker(texture_resolution=512)

    cloud = create_mock_surfel_cloud(num_surfels=100, device="cpu")
    unwrapped, tex_set = baker.bake_pbr_textures(box, surfels=cloud, roughness=0.75, metallic=0.1)

    assert tex_set.resolution == 512
    assert tex_set.albedo_map.size == (512, 512)
    assert tex_set.roughness_map.size == (512, 512)
    assert tex_set.metallic_map.size == (512, 512)
    assert tex_set.normal_map.size == (512, 512)

    with tempfile.TemporaryDirectory() as tmpdir:
        material = tex_set.save(tmpdir, prefix="chair_001")
        assert material.albedo_texture == "chair_001_albedo.png"
        assert material.roughness_texture == "chair_001_roughness.png"
        assert material.metallic_texture == "chair_001_metallic.png"
        assert material.normal_texture == "chair_001_normal.png"

        assert (Path(tmpdir) / "chair_001_albedo.png").exists()
        assert (Path(tmpdir) / "chair_001_roughness.png").exists()
        assert (Path(tmpdir) / "chair_001_metallic.png").exists()
        assert (Path(tmpdir) / "chair_001_normal.png").exists()


def test_scene_assembler_and_cad_exports():
    """Verify complete SceneAssembler export: .glb, .dxf, .ifc, and mesh_manifest.json validation."""
    from scene_assembler import SceneAssembler
    from jsonschema import Draft202012Validator
    import tempfile
    import trimesh

    # Create synthetic room architecture box
    arch_mesh = trimesh.creation.box(extents=[5.0, 2.8, 4.0])

    # Create 2 synthetic furniture items
    chair_mesh = trimesh.creation.box(extents=[0.5, 0.85, 0.5])
    chair_node = MeshNode(
        name="Furniture/chair_001",
        category=NodeCategory.FURNITURE,
        instance_id="chair_001",
        polygon_count=len(chair_mesh.faces),
        vertex_count=len(chair_mesh.vertices),
        transform_matrix=[
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        bounding_box=BoundingBox3D(
            min_point=(0.75, 0.0, 0.25),
            max_point=(1.25, 0.85, 0.75),
            center=(1.0, 0.425, 0.5),
            extents=(0.5, 0.85, 0.5),
        ),
        cad_layer="FF-FURN",
    )

    object_nodes = [chair_node]
    object_meshes = {"Furniture/chair_001": chair_mesh}

    with tempfile.TemporaryDirectory() as tmpdir:
        assembler = SceneAssembler(scene_id="test_listing_cad_001")
        manifest, paths = assembler.assemble_and_export(
            architecture_mesh=arch_mesh,
            object_nodes=object_nodes,
            object_meshes=object_meshes,
            output_dir=tmpdir,
        )

        assert manifest.scene_id == "test_listing_cad_001"
        assert manifest.total_triangles > 0
        assert len(manifest.nodes) == 2

        # Verify exported files exist and are non-empty
        assert paths["glb"].exists() and paths["glb"].stat().st_size > 500
        assert paths["dxf"].exists() and paths["dxf"].stat().st_size > 200
        assert paths["ifc"].exists() and paths["ifc"].stat().st_size > 200
        assert paths["manifest"].exists() and paths["manifest"].stat().st_size > 100

        # Validate generated mesh_manifest.json against shared/schemas/mesh_manifest.schema.json
        schema_path = SCHEMAS_DIR / "mesh_manifest.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        instance = json.loads(paths["manifest"].read_text(encoding="utf-8"))

        validator = Draft202012Validator(schema)
        errors = list(validator.iter_errors(instance))
        assert len(errors) == 0, f"Validation errors: {[e.message for e in errors]}"


def test_surfel_projection_kernel_oracle():
    """Verify Wave32 surfel projection kernel oracle against camera intrinsics."""
    from kernels.surfel_projection import project_surfels_frustum_torch

    # Point directly in front of camera at Z = -2.0m (OpenGL depth = +2.0m)
    pos = torch.tensor([[0.0, 0.0, -2.0]], dtype=torch.float32)
    w2c = torch.eye(4, dtype=torch.float32)

    uv_depth, in_bounds = project_surfels_frustum_torch(
        positions=pos,
        w2c_matrix=w2c,
        fx=500.0,
        fy=500.0,
        cx=250.0,
        cy=250.0,
        width=500,
        height=500,
        near_plane=0.1,
    )

    assert in_bounds.item() is True
    u, v, d = uv_depth[0].tolist()
    assert pytest.approx(d, abs=1e-4) == 2.0
    assert pytest.approx(u, abs=1e-4) == 250.0
    assert pytest.approx(v, abs=1e-4) == 250.0


def test_end_to_end_mesh_generation_pipeline():
    """Test full end-to-end MeshGenerationPipeline on synthetic room data."""
    from pipeline import MeshGenerationPipeline
    import tempfile

    cloud = create_mock_surfel_cloud(
        num_surfels=1500,
        room_dimensions=(4.0, 2.5, 3.5),
        num_furniture_objects=1,
        device="cpu",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "transforms.json"
        create_mock_transforms(
            num_frames=6,
            room_dimensions=(4.0, 2.5, 3.5),
            output_path=json_path,
        )
        dataset = load_transforms_json(json_path, device="cpu")

        config = MeshPipelineConfig(
            device="cpu",
            texture_resolution=512,
            min_cluster_surfels=15,
            compositor_yield_seconds=0.0,
        )
        pipeline = MeshGenerationPipeline(config=config)

        out_dir = Path(tmpdir) / "cad_export"
        manifest, exported_paths = pipeline.run(
            surfels=cloud,
            transforms=dataset,
            output_dir=out_dir,
            scene_id="e2e_test_scene_001",
        )

        assert manifest.scene_id == "e2e_test_scene_001"
        assert manifest.total_triangles > 0
        assert manifest.total_vertices > 0
        assert len(manifest.nodes) >= 2  # Architecture + at least 1 furniture item

        # Verify all output files were created on disk
        assert exported_paths["glb"].exists() and exported_paths["glb"].stat().st_size > 1000
        assert exported_paths["dxf"].exists() and exported_paths["dxf"].stat().st_size > 500
        assert exported_paths["ifc"].exists() and exported_paths["ifc"].stat().st_size > 500
        assert exported_paths["manifest"].exists() and exported_paths["manifest"].stat().st_size > 200

        # Verify textures directory
        tex_dir = out_dir / "textures"
        assert tex_dir.exists()
        assert len(list(tex_dir.glob("*.png"))) >= 4


def test_worker_task_execution():
    """Verify asynchronous worker task generate_cad_mesh_task."""
    import tempfile
    from Utilities.worker.tasks import generate_cad_mesh_task

    cloud = create_mock_surfel_cloud(num_surfels=800, device="cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        ply_path = Path(tmpdir) / "splats.ply"
        save_splats_ply(cloud, ply_path)

        json_path = Path(tmpdir) / "transforms.json"
        create_mock_transforms(num_frames=4, output_path=json_path)

        out_dir = Path(tmpdir) / "worker_output"

        result = generate_cad_mesh_task(
            scene_id="worker_test_001",
            ply_path=str(ply_path),
            transforms_path=str(json_path),
            output_dir=str(out_dir),
            device="cpu",
        )

        assert result["status"] == "completed"
        assert result["scene_id"] == "worker_test_001"
        assert result["total_triangles"] > 0
        assert Path(result["glb_path"]).exists()
        assert Path(result["manifest_path"]).exists()
