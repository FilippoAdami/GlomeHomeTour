# Implementation Plan: Automated 2DGS to Segmented PBR CAD/BIM Mesh Pipeline

**Module Target:** `backend/mesh_generation/`  
**System Role:** Converts trained 2D Gaussian Splatting (2DGS) radiance fields into watertight, object-segmented 3D triangle meshes with baked PBR materials, ready for direct CAD (AutoCAD, Revit, SketchUp) and DCC (Blender, Unreal Engine 5) consumption.  
**Hardware Target:** AMD Radeon RX 9070 XT (16 GB VRAM, RDNA4 Architecture, `gfx1200` ISA, ROCm 6.x / 7.x, PyTorch ROCm).

---

## 1. System Inputs and Outputs Contract

### 1.1 Inputs Contract
* **`reconstruction/splats.ply` (or `.pt`):** Trained 2DGS model containing per-surfel parameters:
  * Positions $\mathbf{p}_i \in \mathbb{R}^3$, Normal vectors $\mathbf{n}_i \in \mathbb{R}^3$, Scaling $(s_{u}, s_{v}) \in \mathbb{R}^2$, Rotation quaternion $\mathbf{q}_i \in \mathbb{R}^4$.
  * Material attributes: Base albedo $\mathbf{c}_i \in [0, 1]^3$, Roughness $r_i \in [0, 1]$, Metallic $m_i \in [0, 1]$, Opacity $\alpha_i \in [0, 1]$.
* **`ingestion/transforms.json` & `trajectory.csv`:** Calibrated metric camera intrinsics ($K$), 60 Hz VIO camera poses ($[R \mid t]$), and upright portrait transformations.
* **`ingestion/keyframes/`:** Undistorted RGB keyframe images ($1080 \times 1920$).
* **`reconstruction/depth_maps/`:** Metric dense depth maps from Depth Anything 3 (`DA3NESTED-GIANT`).

### 1.2 Outputs Contract
* **`cad_export/scene.gltf` / `scene.glb`:** Standard glTF 2.0 asset with a clean node hierarchy:
  * Node `Architecture`: Watertight room shell (floors, walls, ceilings).
  * Nodes `Object_001` to `Object_N`: Individual, editable 3D furniture/appliance solids with local coordinate origins and metric transforms.
* **`cad_export/textures/`:** Standard $2048 \times 2048$ PBR texture sets per node:
  * `*_albedo.png` (sRGB Base Color), `*_roughness.png` (Linear), `*_metallic.png` (Linear), `*_normal.png` (Tangent Space $+Y$).
* **`cad_export/floorplan_3d.dxf` / `scene.ifc`:** Metric CAD / BIM vector primitives for direct import into AutoCAD and Revit.
* **`cad_export/mesh_manifest.json`:** Validated JSON metadata conforming to `shared/schemas/mesh_manifest.schema.json`.

---

## 2. Hardware Architecture & ROCm / HIP Kernel Constraints

Because the RX 9070 XT (RDNA4) uses 32-wide wavefronts and shares VRAM with the Linux display compositor, the implementation must adhere to strict hardware rules:
1. **Wave32 Kernel Execution:** All custom HIP kernels (2DGS rasterization, surfel projection, UV sampling) must align block sizes to multiples of 32 (e.g., `dim3 block(32, 1)`) to avoid SIMD lane masking.
2. **Local Data Share (LDS) Caching:** Avoid raw global VRAM accesses in inner loops; cache surfel bounding primitives into LDS ($32\text{ KB}$ per Compute Unit).
3. **Sequential Stage Execution & VRAM Deallocation:** To operate on 150 objects within 16 GB VRAM without OOM or desktop compositor freezing:
   * Explicitly call `torch.cuda.empty_cache()` and `gc.collect()` between stages.
   * Enforce compositor yielding (`time.sleep(0.18)`) between intensive batch inference loops.
4. **Pure-PyTorch Fallback Oracles:** Every custom HIP kernel must have an identical, verified pure-PyTorch ROCm fallback implementation for debugging and regression testing.

---

## 3. Atomic Phased Implementation Roadmap

```
Phase 1: Module Skeleton & Contracts
   │
Phase 2: 3D Segmentation (SAM 2 + 2DGS Feature Distillation)
   │
Phase 3: Architectural Infill & Planar Snapping (3DGIC + SplatFill)
   │
Phase 4: SOTA Image-to-3D Object Completion (TRELLIS / Hunyuan3D)
   │
Phase 5: UV Parameterization & PBR Texture Baking (xatlas + MaterialRefGS)
   │
Phase 6: Scene Graph Assembly & glTF / CAD Export
   │
Phase 7: ROCm / HIP Kernel Optimization & Worker Integration
   │
Phase 8: Comprehensive Test Suite & Benchmark Verification
```

---

### Phase 1: Module Skeleton, Data Contracts & Schema Validation

#### Step 1.1: Directory Setup & Type Definitions
* **File:** `backend/mesh_generation/types.py`
* **Task:** Define dataclasses and Pydantic models for `MeshNode`, `PBRMaterial`, `InstanceCluster`, `BoundingBox3D`, and `MeshPipelineConfig`.
* **Verification:** Type checkers pass with zero errors (`mypy backend/mesh_generation/types.py`).

#### Step 1.2: Schema Contract Creation
* **File:** `shared/schemas/mesh_manifest.schema.json`
* **Task:** Author JSON Schema Draft 2020-12 specifying object node names, bounding boxes, polygon counts, texture atlas paths, and metric transform matrices.
* **Verification:** `python shared/schemas/validate.py` passes.

#### Step 1.3: Ingestion & 2DGS Adapter
* **File:** `backend/mesh_generation/io_adapter.py`
* **Task:** Implement loaders to ingest `splats.ply` and `transforms.json` into PyTorch ROCm tensors.
* **Verification:** Unit test verifies loaded surfels match the expected metric coordinate system ($+Y$ up, $-Z$ forward).

---

### Phase 2: 3D Semantic & Instance Segmentation (SAM 2 + 2DGS Grouping)

#### Step 2.1: Keyframe SAM 2 Multi-View Inference
* **File:** `backend/mesh_generation/segmentation.py`
* **Task:** Implement `extract_keyframe_masks(keyframes_dir, config)`.
  * Runs SAM 2 on keyframes using ROCm BF16 AMP.
  * Outputs 2D instance masks with unique object tracking IDs across views.
* **Verification:** Produces consistent 2D instance mask tensors across sequential keyframes.

#### Step 2.2: 2D-to-3D Gaussian Identity Encoding Projection
* **File:** `backend/mesh_generation/kernels/identity_projection.py` (PyTorch) & `backend/mesh_generation/kernels/hip/identity_proj.hip` (ROCm HIP)
* **Task:** Projects 2D instance masks onto 2DGS surfels via differentiable ray-surfel splatting:
  * Endow each surfel with an identity feature vector $\mathbf{f}_i \in \mathbb{R}^{16}$.
  * Optimize with cross-entropy and 3D spatial smoothness regularization.
* **Verification:** Verify that PyTorch fallback and HIP kernel return identical identity assignments within $\epsilon < 10^{-4}$.

#### Step 2.3: 3D Instance Clustering & Label Assignment
* **File:** `backend/mesh_generation/segmentation.py`
* **Task:** Cluster surfels into discrete instances using 3D DBSCAN and feature affinity graph-cuts.
  * Segregate background architecture (Walls, Floor, Ceiling) from foreground items (150 furniture/appliance objects).
* **Verification:** Produces `InstanceMap` partitioning the scene into `background` and labeled object clusters `obj_001` through `obj_150`.

---

### Phase 3: Architectural Background Infill & Planar Regularization

#### Step 3.1: Foreground Masking & Void Boundary Detection
* **File:** `backend/mesh_generation/architectural_infill.py`
* **Task:** Remove all foreground furniture splats to expose the raw architectural background. Identify geometric voids and boundary edges on floors and walls.
* **Verification:** Void masks accurately locate areas previously occluded by couches, beds, and cabinets.

#### Step 3.2: 3DGIC + SplatFill Epipolar Depth Inpainting
* **File:** `backend/mesh_generation/architectural_infill.py`
* **Task:** Implement depth-guided diffusion inpainting:
  * Inpaint occluded floor/wall keyframes using depth diffusion.
  * Apply 3DGIC multi-view cross-attention to enforce epipolar consistency across camera views.
  * Back-project new 2DGS surfels into the void regions to seal holes.
* **Verification:** Rendered walkthrough of the infilled background shows flat floors and continuous walls without ghosting or tears.

#### Step 3.3: Manhattan-World Planar RANSAC Snapping
* **File:** `backend/mesh_generation/planar_snapping.py`
* **Task:** Fit orthogonal planes ($90^\circ$ Manhattan constraints) to the infilled wall, floor, and ceiling surfels using RANSAC.
  * Project noisy surfel positions onto mathematical planes to eliminate scan ripple.
* **Verification:** Extracted planes exhibit $0.00^\circ$ angular deviation from Manhattan axes.

#### Step 3.4: Watertight Architectural Shell Mesh Extraction
* **File:** `backend/mesh_generation/architectural_infill.py`
* **Task:** Generate a clean 3D triangle mesh of the room shell (`walls_floors.obj`) using 2D-SuGaR / Poisson surface reconstruction regularized by fitted planar boundaries.
* **Verification:** Mesh passes watertight manifold check (`trimesh.is_watertight == True`).

---

### Phase 4: SOTA Image-to-3D Object Completion & Prototype De-duplication

#### Step 4.1: Instance Similarity Clustering (Prototype De-duplication)
* **File:** `backend/mesh_generation/prototype_cluster.py`
* **Task:** Compute geometric point-cloud embeddings and color histograms for the 150 segmented objects.
  * Group matching items (e.g., 8 identical dining chairs, 12 downlights, 4 barstools) into **~40–50 unique object prototypes**.
* **Verification:** Identical objects are clustered with $>95\%$ precision, reducing generative inference load by $\sim 65\%$.

#### Step 4.2: Canonical Multi-View Render Extraction
* **File:** `backend/mesh_generation/object_reconstruction.py`
* **Task:** Render clean, white-background canonical views (front, side, top, isometric) for each unique object prototype directly from the isolated 2DGS surfels.
* **Verification:** Clean multi-view image sets generated for all prototypes.

#### Step 4.3: Batch TRELLIS / Hunyuan3D 2.0 ROCm Engine
* **File:** `backend/mesh_generation/object_reconstruction.py`
* **Task:** Implement the Image-to-3D batch generative reconstruction wrapper:
  * Loads TRELLIS / Hunyuan3D 2.0 in FP16/BF16 on ROCm.
  * Generates watertight 3D meshes with clean quad/triangle topology.
  * Enforces VRAM cleanup (`torch.cuda.empty_cache()`) between batch chunks.
* **Verification:** Generates watertight meshes for all prototypes in $\le 7\text{ seconds}$ per prototype on RX 9070 XT.

#### Step 4.4: 3D Bounding-Box Alignment & Instance Cloning
* **File:** `backend/mesh_generation/object_reconstruction.py`
* **Task:** Register generated prototype meshes back into the 2DGS world coordinate frame:
  * Align to the original 3D oriented bounding box (OBB) via iterative closest point (ICP).
  * Clone prototype meshes to duplicate instances using their respective $[R \mid \mathbf{t} \mid \mathbf{s}]$ transform matrices.
* **Verification:** All 150 objects are precisely positioned inside the room with correct scale and orientation.

---

### Phase 5: UV Parameterization & PBR Texture Baking

#### Step 5.1: Multithreaded Conformal UV Unwrapping
* **File:** `backend/mesh_generation/texture_baking.py`
* **Task:** Integrate `xatlas-python` to compute non-overlapping UV atlas layouts for:
  * The architectural shell.
  * Each of the reconstructed 3D object instances.
* **Verification:** UV charts have zero self-intersections and maintain $>75\%$ UV space packing efficiency.

#### Step 5.2: Surfel-to-UV PBR Projection Kernel
* **File:** `backend/mesh_generation/kernels/pbr_baking.py` (PyTorch) & `backend/mesh_generation/kernels/hip/pbr_baking.hip` (ROCm HIP)
* **Task:** Sample material parameters from 2DGS surfels (or Paint3D inpainting for occluded rear faces) and bake into $2048 \times 2048$ 2D texture maps:
  * **Albedo Map:** Diffuse base color (sRGB).
  * **Roughness Map:** Specular roughness (Linear $R$).
  * **Metallic Map:** Metalness mask (Linear $G$).
  * **Normal Map:** Tangent-space normal vector ($+Y$ OpenGL standard).
* **Verification:** Baked texture maps render photorealistically in standard PBR viewports without seam artifacts.

---

### Phase 6: Scene Graph Assembly & Multi-Format CAD Export

#### Step 6.1: Metric Scene Graph Hierarchy Construction
* **File:** `backend/mesh_generation/scene_assembler.py`
* **Task:** Assemble the root scene graph containing:
  * Metadata node (metric unit: meters, coordinate system: $+Y$ up, $+Z$ forward).
  * Room architecture node with sub-meshes (Floor, Ceiling, Wall_N).
  * Discrete object nodes with semantic labels (e.g. `Furniture/LivingRoom/Sofa_001`).
* **Verification:** Hierarchical structure validated against internal graph specifications.

#### Step 6.2: glTF 2.0 / USDZ Binary Packager
* **File:** `backend/mesh_generation/exporters/gltf_exporter.py`
* **Task:** Export standalone binary `.glb` and `.usdz` packages with embedded PBR textures, buffer views, and node transforms.
* **Verification:** Exported `.glb` loads without warnings in Blender 4.x, Three.js Editor, and Unreal Engine 5.

#### Step 6.3: Parametric CAD / BIM Exporters (DXF / DWG / IFC)
* **File:** `backend/mesh_generation/exporters/cad_exporter.py`
* **Task:** Export 2D/3D polyline DXF/DWG layers and Industry Foundation Classes (IFC 4) files:
  * Walls, doors, windows, and furniture footprints mapped to standard CAD layers (`A-WALL`, `A-FLOR`, `FF-FURN`).
* **Verification:** `.dxf` opens cleanly in AutoCAD with correct layer separations and millimeter/meter units.

---

### Phase 7: ROCm / HIP Optimization, VRAM Lifecycle & Worker Integration

#### Step 7.1: HIP Kernel Wave32 Tuning for RDNA4
* **File:** `backend/mesh_generation/kernels/hip/`
* **Task:** Benchmark and optimize custom HIP operators on RX 9070 XT:
  * Use `#pragma unroll` and 32-thread block configurations.
  * Verify LDS utilization stays under $32\text{ KB}$ per CU.
* **Verification:** Profiling shows zero wavefront stall cycles and $>85\%$ memory bandwidth saturation.

#### Step 7.2: Worker Task & Job Queue Hookup
* **File:** `backend/worker/tasks.py`
* **Task:** Implement `@rq.job` worker task:
  ```python
  @rq.job(timeout="25m")
  def generate_cad_mesh_task(scene_id: str) -> dict:
      pipeline = MeshGenerationPipeline(config=MeshPipelineConfig())
      return pipeline.run(scene_id)
  ```
* **Verification:** Worker processes full scene job from Redis queue, releasing GPU VRAM cleanly upon completion.

---

### Phase 8: Comprehensive Test Suites & Benchmark Verification

#### Step 8.1: Unit & Kernel Test Suite
* **File:** `backend/tests/test_mesh_generation.py`
* **Test Cases:**
  1. `test_types_and_schema_validation`: Checks schema compliance.
  2. `test_sam2_segmentation_clustering`: Validates 3D DBSCAN grouping.
  3. `test_hip_vs_pytorch_projection`: Asserts numerical equivalence between HIP and PyTorch fallback kernels.
  4. `test_infill_manifold_integrity`: Asserts watertightness of infilled architectural shell.
  5. `test_prototype_deduplication`: Verifies clustering efficiency on synthetic 150-object mock data.
  6. `test_uv_and_pbr_baking`: Checks texture dimensions, range $[0, 1]$, and non-zero channels.
  7. `test_gltf_and_dxf_export`: Validates file headers and binary completeness.
* **Verification:** `pytest backend/tests/test_mesh_generation.py` passes with $100\%$ green tests.

#### Step 8.2: End-to-End Benchmark Execution
* **Task:** Run full pipeline on bedroom test scan dataset on RX 9070 XT.
* **Success Criteria:**
  * Total execution time: $\le 10\text{ minutes}$ (with prototype de-duplication).
  * Peak VRAM: $\le 11.0\text{ GB}$ (zero OOM errors).
  * Exported `.glb` and `.dxf` pass structural validation.

---

*This implementation plan is structured to be executed step-by-step in dedicated implementation sessions.*
