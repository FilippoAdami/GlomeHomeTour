#pragma once

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#else
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#endif
