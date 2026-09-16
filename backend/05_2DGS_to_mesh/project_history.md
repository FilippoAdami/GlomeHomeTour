# Project History: backend/mesh_generation/

## 2026-09-10: Phase 1 — Module Skeleton, Data Contracts & Schema Validation

Implemented Phase 1 of the automated 2DGS to Segmented PBR CAD/BIM Mesh pipeline (`backend/mesh_generation/implementation_plan.md`):

1. **Step 1.1 — Data Contracts & Type Definitions (`types.py`):**
   - Implemented frozen `BoundingBox3D` (metric extents, center, volume, containment, and `from_points` computation from NumPy/PyTorch arrays).
   - Implemented `PBRMaterial` adhering to the glTF 2.0 metallic-roughness workflow (albedo, roughness, metallic, normal maps, factors, double-sided support).
   - Implemented `MeshNode` hierarchy with strict 4x4 metric affine transform validation, polygon/vertex counts, bounding boxes, CAD layer mappings (`A-WALL`, `FF-FURN`, etc.), and nested children nodes.
   - Implemented `InstanceCluster` with cluster IDs, semantic labels, surfel index arrays, centroids, confidence scores, and prototype alignment transforms.
   - Implemented `MeshPipelineConfig` encoding RX 9070 XT hardware constraints (target ISA `gfx1200`, native Wave32 execution, LDS capacity $\le 32\text{ KB}$, desktop compositor non-blocking cooperative yield $0.18\text{s}$, PBR texture resolution 2048/4096, polygon decimation budgets).
   - Implemented `MeshManifest` model representing the top-level scene manifest contract.
   - Implemented `SurfelCloudTorch` PyTorch container with right-handed orthonormal tangent frame construction ($u = n \times ref, v = n \times u$, with $u \cdot n = 0, v \cdot n = 0, u \cdot v = 0$), device transfer (`to('cpu')` / `to('cuda')`), slicing, metric bounding box computation, and binary little-endian PLY export.

2. **Step 1.2 — Schema Contract & Validation (`shared/schemas/mesh_manifest.schema.json`):**
   - Authored JSON Schema Draft 2020-12 specifying scene metadata, metric bounding boxes, polygon statistics, PBR texture paths, and 4x4 transform matrices.
   - Created valid fixture `shared/schemas/fixtures/mesh_manifest.example.json`.
   - Registered `mesh_manifest` schema and fixture in `shared/schemas/validate.py`.
   - Verified that `python shared/schemas/validate.py` passes with 100% OK across all schemas and fixtures.

3. **Step 1.3 — Ingestion & 2DGS Adapters (`io_adapter.py`):**
   - Implemented `load_splats_ply` supporting binary little-endian and ASCII PLY formats, metric coordinates (+Y up, -Z forward), normals normalization, 2D scales, normalized RGB colors, and opacities.
   - Implemented `save_splats_ply` for exporting surfels back to binary PLY.
   - Implemented `CameraIntrinsicsTorch`, `KeyframePoseTorch`, and `TransformsDataset` for loading `transforms.json` into PyTorch camera models with validated $c2w$ and inverted $w2c$ matrices.
   - Implemented isolated mock generators `create_mock_surfel_cloud` and `create_mock_transforms` to enable robust, self-contained offline testing without relying on unstable upstream steps.

4. **Test Suite Verification (`backend/tests/test_mesh_generation.py`):**
   - Added 10 comprehensive unit tests covering bounding box arithmetic, PBR material validation, scene graph node hierarchy, pipeline config constraints, JSON schema Draft 2020-12 compliance, PyTorch tangent frame orthogonality, surfel slicing/bounding box, PLY binary roundtrip precision, `transforms.json` matrix inversion integrity, and cooperative GPU VRAM cache flushing.
   - Result: 10/10 tests passed (`pytest backend/tests/test_mesh_generation.py`).

## 2026-09-10: Phase 2 — Hybrid Geometric-Semantic 3D Instance Segmentation

Implemented Phase 2 per user architectural alignment (Option 5: Hybrid Geometric + Grounded-SAM 2):

1. **Geometric Architectural Separation (`GeometricArchitectureFilter`):**
   - Classifies background room shell (Floor, Ceiling, Walls) vs. foreground furniture solids using gravity-aligned surface normals ($\mathbf{n} \cdot [0, 1, 0]$) and normalized vertical heights. Eliminates neural boundary hallucinations on planar surfaces.
2. **Grounded Keyframe Instance Projection (`SemanticInstanceSegmenter`):**
   - Implemented 2D-to-3D multi-view ray-frustum projection to accumulate class votes on 3D surfels across camera trajectories.
   - Built modular detector interface compatible with Florence-2 / SAM 2 with deterministic multi-view mock generators for isolated unit testing.
3. **3D Instance Assembly:**
   - Assembled segmented foreground surfels into validated `InstanceCluster` models with 3D metric bounding boxes (`BoundingBox3D`) and semantic categories.
4. **Verification:**
   - Added unit tests `test_geometric_architecture_segmentation` and `test_semantic_instance_segmenter_pipeline` in `backend/tests/test_mesh_generation.py`.
   - Result: 12/12 tests green (`pytest backend/tests/test_mesh_generation.py`).

## 2026-09-10: Phase 3 — Architectural Background Infill, Manhattan RANSAC & Watertight Mesh Extraction

Implemented Phase 3 (`backend/mesh_generation/planar_snapping.py` and `architectural_infill.py`):

1. **Manhattan-World Planar RANSAC Snapping (`ManhattanPlanarRANSAC`):**
   - Fits orthogonal planes ($90^\circ$ constraints, $0.00^\circ$ angular deviation) to floor, ceiling, and perimeter walls ($+X, -X, +Z, -Z$).
   - Projects inlier surfel positions onto their mathematical planes, eliminating scan noise and depth sensor ripple.
2. **Foreground Void Detection & Planar Infilling (`ArchitecturalInfillEngine`):**
   - Identifies geometric voids and occlusion footprints under removed furniture.
   - Synthesizes planar back-projected surfels to seal floor and wall holes.
3. **Watertight Architectural Shell Mesh Extraction:**
   - Assembles fitted boundary planes into a verified 100% watertight 3D manifold box mesh (`trimesh.Trimesh`, `shell.is_watertight == True`).
4. **Verification:**
   - Added unit tests `test_manhattan_planar_snapping` and `test_architectural_infill_and_watertight_shell`.
   - Result: 14/14 tests green (`pytest backend/tests/test_mesh_generation.py`).

## 2026-09-10: Phase 4 — SOTA Image-to-3D Object Completion (Pixal3D / TRELLIS.2) & Prototype De-duplication

Implemented Phase 4 per user decision on SOTA generative 3D model architecture (`TencentARC/Pixal3D`):

1. **Instance Similarity Clustering & De-duplication (`PrototypeClusterEngine`):**
   - Extracts geometric aspect ratio signatures, bounding box volumes, and 24-bin RGB color histograms.
   - Groups matching furniture instances (e.g. 4 identical chairs + 2 identical tables) into unique `PrototypeGroup` entries, reducing generative inference load by $\ge 65\%$.
2. **Canonical Multi-View Projection Rendering (`CanonicalViewRenderer`):**
   - Projects isolated surfels of each unique prototype into 4 clean, white-background canonical orthographic/perspective views (front, side, top, isometric).
3. **Pixal3D Generative Reconstruction Engine (`Pixal3DReconstructionEngine`):**
   - Implemented wrapper based on Tencent ARC Pixal3D (pixel back-projection conditioning on TRELLIS.2 O-Voxel backbone).
   - Enforces sequential GPU VRAM safety (`torch.cuda.empty_cache()`) and non-blocking desktop compositor cooperative yields.
   - Includes deterministic CAD parametric fallback oracle for instant offline testing.
4. **OBB Alignment & Instance Cloning (`OBBAligner`):**
   - Scales, translates, and aligns prototype 3D meshes to match each instance's target 3D bounding box solid in room world coordinates, cloning instances with their respective $[R \mid \mathbf{t} \mid \mathbf{s}]$ transform matrices into `MeshNode` hierarchies.
5. **Verification:**
   - Added unit tests `test_prototype_cluster_deduplication`, `test_canonical_view_renderer`, and `test_pixal3d_reconstruction_and_obb_alignment` in `backend/tests/test_mesh_generation.py`.
   - Result: 17/17 tests green (`pytest backend/tests/test_mesh_generation.py`).

## 2026-09-10: Phase 5 — UV Parameterization & PBR Texture Baking

Implemented Phase 5 (`backend/mesh_generation/texture_baking.py`):

1. **Conformal & Triplanar UV Atlas Unwrapping (`UVAtlasUnwrapper`):**
   - Implemented non-overlapping UV atlas unwrapping mapping 3D triangle faces to dominant canonical projection planes ($+X, -X, +Y, -Y, +Z, -Z$) packed into a clean $2 \times 3$ grid atlas in $[0, 1] \times [0, 1]$.
2. **PBR Texture Baking (`PBRTextureBaker`):**
   - Synthesizes and exports complete sets of square PBR texture maps per node:
     * Albedo (Base Color sRGB) sampled from nearest 2DGS surfel color distributions.
     * Linear Roughness (grayscale channel).
     * Linear Metallic (grayscale channel).
     * Tangent-space Normal (+Y OpenGL standard tangent space).
   - Saves `.png` texture maps to disk and configures `PBRMaterial` descriptors.
3. **Verification:**
   - Added unit tests `test_uv_unwrapping_and_atlas_bounds` and `test_pbr_texture_baking_and_material_export` in `backend/tests/test_mesh_generation.py`.
   - Result: 19/19 tests green with zero warnings (`pytest backend/tests/test_mesh_generation.py`).

## 2026-09-10: Phase 6 — Scene Graph Assembly & Multi-Format CAD/glTF/BIM Export

Implemented Phase 6 (`backend/mesh_generation/scene_assembler.py`, `exporters/gltf_exporter.py`, and `exporters/cad_exporter.py`):

1. **Hierarchy Scene Assembler (`SceneAssembler`):**
   - Combines the watertight room architectural shell (`Architecture`) and segmented foreground objects into a unified metric coordinate frame (+Y up, -Z forward, metric meters).
   - Computes whole-scene bounding bounds and aggregate polygon/vertex counts.
2. **glTF 2.0 & GLB Binary Packager (`GLTFExporter`):**
   - Packages scene geometry, 4x4 node transformations, and PBR textures into standard standalone binary `.glb` scenes compatible with Blender, Unreal Engine 5, and web viewers.
3. **Parametric CAD & BIM Exporters (`DXFExporter` & `IFCExporter`):**
   - **DXF (AutoCAD R12 compatibility):** Exports standard layered CAD models with `A-WALL`, `A-FLOR`, `A-CEIL`, and `FF-FURN` layers and metric units (`$INSUNITS = 6`).
   - **IFC (Industry Foundation Classes IFC4):** Generates STEP-format BIM files mapping nodes to `IfcWall`, `IfcSlab`, `IfcFurnishingElement`, and `IfcBuildingElementProxy`.
4. **Mesh Manifest Emission & Schema Integrity:**
   - Emits `cad_export/mesh_manifest.json` conforming to `shared/schemas/mesh_manifest.schema.json`.
5. **Verification:**
   - Added unit test `test_scene_assembler_and_cad_exports` in `backend/tests/test_mesh_generation.py`.
   - Result: 20/20 tests green (`pytest backend/tests/test_mesh_generation.py`) and schema validation passes (`shared/schemas/validate.py`).

## 2026-09-10: Phase 7 — ROCm / HIP Wave32 Kernel Optimization, Pipeline & Worker Integration

Implemented Phase 7 (`backend/mesh_generation/pipeline.py`, `kernels/`, and `backend/worker/tasks.py`):

1. **Wave32 Surfel Frustum Projection Kernel (`kernels/hip/surfel_projection.hip` & `kernels/surfel_projection.py`):**
   - Implemented native Wave32 execution (`dim3 block(32, 1, 1)`) on AMD RDNA4 (`gfx1200`).
   - Caches $4 \times 4$ camera transformation matrix into LDS (Local Data Share $\le 32\text{ KB}$ per CU).
   - Authored pure-PyTorch fallback oracle (`project_surfels_frustum_torch`) for CPU and ROCm execution with numerical equivalence.
2. **End-to-End Pipeline Orchestration (`MeshGenerationPipeline`):**
   - Integrates all six phases into a cohesive automated workflow:
     * Ingestion $\to$ Hybrid Segmentation $\to$ Architectural Infill / Watertight Shell $\to$ Prototype De-duplication $\to$ Pixal3D Generative Completion $\to$ UV Unwrapping / PBR Texture Baking $\to$ CAD / BIM / glTF Export $\to$ Manifest validation.
   - Enforces sequential execution, non-blocking compositor yields, and VRAM deallocation (`torch.cuda.empty_cache()`).
3. **Queue Worker Task Integration (`backend/worker/tasks.py`):**
   - Implemented `generate_cad_mesh_task(scene_id, ply_path, transforms_path, output_dir, device)` for background job execution.
4. **Verification:**
   - Added unit tests `test_surfel_projection_kernel_oracle`, `test_end_to_end_mesh_generation_pipeline`, and `test_worker_task_execution`.
   - Result: 23/23 tests green (`pytest backend/tests/test_mesh_generation.py`).
