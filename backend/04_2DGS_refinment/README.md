# 04_2DGS_refinment — Stage 6: 2DGS Model Refinement, PBR & Compression

`04_2DGS_refinment` post-processes trained 2DGS radiance fields into production-grade digital listing assets. It bridges raw radiance field splats into MLS-compliant deliverables by handling specular materials, reflections, standard format conversion, and web compression ($\le 25\text{ MB}$ walkthrough bundle).

---

## What This Stage Does

1. **LightGaussian Web Compression:** Compresses multi-million surfel radiance fields into lightweight binary bundles ($\le 25\text{ MB}$) using prune-and-quantize pipelines, allowing instant loading in mobile web browsers without WebGPU stalls.
2. **PBR Material Learning:** Decomposes view-dependent radiance into physically based rendering (PBR) parameters: base albedo, microfacet roughness, metalness, and deferred Cook-Torrance specular shading.
3. **Planar Mirror & Reflection Handling:** Detects large reflective surfaces (mirrors, polished tiles, glossy wardrobe glass) and synthesizes virtual mirror camera passes to prevent duplicate ghost geometry.
4. **Standard Ecosystem PLY Export:** Converts native internal model checkpoints into canonical 3DGS/2DGS `.ply` files compatible with SuperSplat, gsplat, Nerfstudio, and three.js web viewers.

---

## Modules & Architecture

| Module | Role |
| --- | --- |
| [`compressor.py`](compressor.py) | Vector-quantization & distillation engine implementing LightGaussian-style k-means codebooks and Deflate compression for $\le 25\text{ MB}$ web bundles. |
| [`pbr_shader.py`](pbr_shader.py) | Deferred Cook-Torrance microfacet BRDF shader with neural specular residual MLP for view-dependent lighting. |
| [`planar_reflections.py`](planar_reflections.py) | Planar RANSAC mirror detector that reflects camera centers across mirror planes to resolve reflection geometry. |
| [`export_standard_ply.py`](export_standard_ply.py) | Exports trained surfels to canonical PLY formats with standard Spherical Harmonics property names. |

---

## Techniques & Mathematical Specifications

### 1. LightGaussian Surfel Compression
Standard trained models have $1\text{M} - 3\text{M}$ surfels ($300\text{ MB} - 800\text{ MB}$ uncompressed). The compressor achieves a $> 15\times$ reduction:
- **Opacity / Footprint Pruning:** Removes low-contribution surfels ($\alpha < 0.05$ or tiny projected screen area) without perceptible PSNR drop.
- **Vector Quantization (Codebook K-Means):** Quantizes high-order Spherical Harmonics coefficients into $K = 256$ codebook clusters (8-bit index per splat instead of 45 float32 parameters).
- **Coordinate Half-Float Encoding:** Quantizes scales and rotation quaternions to 16-bit float (`fp16`) or 8-bit log-space intervals.
- **Entropy Coding:** Compresses quantized attribute streams via DEFLATE/gzip into `walkthrough_2dgs.zip`.

### 2. Physically Based Material Decomposition
Instead of purely approximating lighting via Spherical Harmonics:
$$L_o(p, \omega_o) = \int_{\Omega} f_r(p, \omega_i, \omega_o) L_i(p, \omega_i) (\omega_i \cdot n) \, d\omega_i$$
where $f_r$ is split into:
- **Diffuse Component:** Lambertian base color $c_{\text{diffuse}} = \frac{\rho}{\pi}$.
- **Specular Component:** Microfacet Cook-Torrance specular model with GGX normal distribution function $D$, Schlick-GGX geometric shadowing $G$, and Fresnel term $F$:
  $$f_{\text{spec}} = \frac{D(\omega_h) F(\omega_o, \omega_h) G(\omega_i, \omega_o, \omega_h)}{4 (\omega_i \cdot n) (\omega_o \cdot n)}$$

### 3. Planar Reflection Geometry
For mirrors and large glass panels, real-world points reflect across a fitted plane $\Pi = \{x \mid n_\pi \cdot x + d = 0\}$:
$$P_{\text{virtual}} = P - 2 (n_\pi \cdot P + d) n_\pi$$
Camera viewpoints are transformed symmetrically:
$$C_{\text{virtual}} = C - 2 (n_\pi \cdot C + d) n_\pi, \quad R_{\text{virtual}} = (I - 2 n_\pi n_\pi^T) R$$
This renders reflections as virtual scenes behind the plane, eliminating ghost points inside the physical room.

---

## Adapter & Refactoring Guide

These modules were migrated from the backend's earlier 2DGS iteration and are slated for direct integration with `03_2DGS_training`:
- **Model Interface:** Map `Material2DGSModel` to `scene.gaussian_model.GaussianModel`.
- **Render Buffers:** Map `GBufferOutput` to the `allmap` multi-channel tensors returned by `gaussian_renderer.render()`.
- **Checkpoint Layout:** Update `export_standard_ply.py` to read parameter tensors directly from `GaussianModel.state_dict()`.

