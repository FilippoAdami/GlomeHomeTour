from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(name="glome-native-ssim", version="0.1.0", packages=["fused_ssim"],
      ext_modules=[CUDAExtension("fused_ssim_cuda", ["ssim.cu", "ext.cpp"])],
      cmdclass={"build_ext": BuildExtension})
