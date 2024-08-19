#pragma once

// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <cuda_texture_types.h>

// includes, cuda
#include <vector_types.h>
#include <driver_functions.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

#include "driver_types.h"
#include "headers/helper_cuda.h"

//#include "cameraparameters.h"

class Managed {
public:
  void *operator new(size_t len) {
    void *ptr;
    checkCudaErrors(cudaMallocManaged(&ptr, len));
    return ptr;
  }

  void operator delete(void *ptr) {
      cudaFree(ptr);
  }
};
