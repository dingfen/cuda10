#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>

#define GPUAssert(x) gpuAssert((x), __FILE__, __LINE__)

// Convenience helper function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}


template <int count, typename... TArgs>
__host__ __device__ void perf_helper_func(const std::string& name, dim3 grid_size, dim3 blk_size, void(*kernel)(TArgs...), TArgs... args) {
    cudaEvent_t startEvent, stopEvent;
    float ms; // elapsed time in milliseconds

    GPUAssert(cudaEventCreate(&startEvent));
    GPUAssert(cudaEventCreate(&stopEvent));
    GPUAssert(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < count; i++) {
        kernel<<<grid_size, blk_size>>>(args);
        GPUAssert(cudaStreamSynchronize(0));
    }
    GPUAssert(cudaEventRecord(stopEvent, 0));
    GPUAssert(cudaEventSynchronize(stopEvent));
    GPUAssert(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("Time for executing %s %d times: %f (ms)\n", name, count, ms);
}
