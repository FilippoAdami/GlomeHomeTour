# Agent guidance

This stage trains 3D Gaussian splats with Apache-2.0 AMD gsplat on ROCm. The directory name remains for pipeline path compatibility. Read `README.md` and the relevant files in `agents_files/` before substantial changes. Preserve unrelated experiment outputs and user data. Do not reintroduce the original Inria Gaussian Splatting implementation or its native extensions; the source replacement exists to remove their noncommercial restriction. Keep ROCm 7.1/gfx1200 support and verify kernels on the target GPU when changing them.
