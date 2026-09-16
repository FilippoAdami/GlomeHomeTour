# 05_2DGS_to_mesh — Stage 7: 2DGS to Segmented PBR CAD/BIM Mesh Pipeline

`05_2DGS_to_mesh` converts trained 2D Gaussian Splatting radiance fields into watertight, object-segmented 3D triangle meshes with baked PBR materials. It bridges photorealistic neural rendering into engineering-grade CAD (AutoCAD, Revit, SketchUp) and DCC (Blender, Unreal Engine 5) listing assets.

---

## What This Stage Does

1. **Semantic Instance Segmentation:** Decomposes the scene into structural architectural planes (walls, floors, ceilings) and individual furniture/object instances (chairs, tables, beds, cabinets).
2. **Planar Snapping & Manhattan Regularization:** Detects architectural planar surfaces via RANSAC and snaps noisy surfel coordinates to orthogonal, Manhattan-aligned planes.
3. **Architectural Infill (Watertight Completion):** Extrudes and seals unobserved boundary areas (e.g. wall sections occluded by wardrobes or beds) to generate closed room boundaries.
4. **Per-Object Mesh Reconstruction:** Performs Poisson or alpha-shape surface reconstruction on segmented object clusters.
5. **PBR Texture Baking:** Computes UV parameterizations and bakes diffuse albedo and roughness maps from the 2DGS radiance field onto the triangle mesh.
6. **CAD/BIM & glTF Export:** Generates industry-standard `.dxf` (CAD), `.ifc` (BIM), and `.glb` (glTF 2.0 with PBR materials) deliverables.

---

## Modules & Architecture

| Module / Directory | Role |
| --- | --- |
| [`pipeline.py`](pipeline.py) | **Main Stage Orchestrator**: Executes the end-to-end mesh conversion pipeline from trained splats to exported files. |
| [`mesh_types.py`](mesh_types.py) | Data contracts and schema definitions (`SurfelCloudTorch`, `MeshNode`, `PBRMaterial`, `InstanceCluster`, `MeshManifest`). |
| [`io_adapter.py`](io_adapter.py) | High-throughput tensor I/O adapters for loading/saving splats, camera poses, and transforms. |
| [`segmentation.py`](segmentation.py) | Semantic instance segmentation (`SemanticInstanceSegmenter`) and geometric architecture filter (`GeometricArchitectureFilter`). |
| [`planar_snapping.py`](planar_snapping.py) | Multi-model RANSAC plane fitting and orthogonal Manhattan-world regularization. |
| [`architectural_infill.py`](architectural_infill.py) | Geometry completion for occluded architectural backings. |
| [`object_reconstruction.py`](object_reconstruction.py) | Surface meshing and decimation for foreground object clusters. |
| [`texture_baking.py`](texture_baking.py) | Multi-view ray-casting texture atlas generation and UV unwrapping. |
| [`scene_assembler.py`](scene_assembler.py) | Assembles architectural structural elements and object nodes into a coherent scene hierarchy. |
| [`export_surfel_mesh.py`](export_surfel_mesh.py) | Direct surfel-to-mesh conversion utility. |
| [`exporters/cad_exporter.py`](exporters/cad_exporter.py) | Generates layer-structured AutoCAD `.dxf` and standard architectural `.ifc` files. |
| [`exporters/gltf_exporter.py`](exporters/gltf_exporter.py) | Generates web-ready binary `.glb` files with PBR materials. |
| [`kernels/surfel_projection.py`](kernels/surfel_projection.py) | PyTorch/HIP projection compute kernels for surfel rasterization. |

---

## Techniques & Mathematical Specifications

### 1. Geometric Architecture Filtering
Structural elements are classified using normal orientation and height distribution:
- **Floors:** Upward-pointing unit normal ($n_y \approx +1.0$, within $10^\circ$) at minimum room elevation.
- **Ceilings:** Downward-pointing unit normal ($n_y \approx -1.0$) at maximum room elevation.
- **Walls:** Horizontal unit normals ($|n_y| \le 0.15$) clustered into orthogonal Manhattan directions ($0^\circ, 90^\circ, 180^\circ, 270^\circ$).

### 2. Multi-Model RANSAC Planar Snapping
For each architectural cluster, iterative RANSAC identifies dominant plane parameters $\pi = (n_x, n_y, n_z, d)$:
$$n \cdot x + d = 0$$
Surfels within distance threshold $\tau_{\text{plane}} = 2.0\text{ cm}$ are projected onto $\pi$:
$$x_{\text{snapped}} = x - (n \cdot x + d) n$$
Ensuring razor-sharp, flat walls and floors without scanning waviness.

### 3. Texture Baking via Differential Ray Casting
For each triangle face on the reconstructed mesh, texture coordinates $(u, v)$ are mapped to 3D world coordinates $P(u, v)$. The texel color is determined by querying the trained 2DGS model from the camera view $k$ providing the most orthogonal viewing angle:
$$k^* = \arg\max_k \left(n_{\text{face}} \cdot \frac{C_k - P}{\|C_k - P\|}\right)$$

---

## Inputs & Outputs

- **Input:**
  - `<workspace>/output/point_cloud/iteration_30000/point_cloud.ply`: Trained 2DGS radiance field.
  - `<workspace>/transforms.json` + keyframe images.
- **Output:**
  - `<workspace>/mesh/model.glb`: Interactive web-viewable PBR mesh.
  - `<workspace>/mesh/floorplan.dxf`: Layered 2D/3D CAD vector layout.
  - `<workspace>/mesh/building.ifc`: Industry Foundation Classes (BIM) model.
  - `<workspace>/mesh/manifest.json`: Hierarchical scene manifest with room square meterage.
