"""Build script for custom 2DGS HIP rasterizer extension."""

import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# On ROCm, CUDAExtension routes automatically to hipcc if ROCM_HOME or hipcc is present
setup(
    name="rasterizer_hip",
    ext_modules=[
        CUDAExtension(
            name="rasterizer_hip",
            sources=[
                "ext.cpp",
                "forward.hip",
                "backward.hip",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
