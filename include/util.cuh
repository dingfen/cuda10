#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>

#define GPUAssert(x) gpuAssert((x), __FILE__, __LINE__)

// Convenience helper function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}


template <int count, typename... TArgs>
void perf_helper_func(const std::string& name, dim3 grid_size, dim3 blk_size, void(*kernel)(TArgs...), TArgs... args) {
  cudaEvent_t startEvent, stopEvent;
  float ms; // elapsed time in milliseconds

  GPUAssert(cudaEventCreate(&startEvent));
  GPUAssert(cudaEventCreate(&stopEvent));
  cudaProfilerStart();
  GPUAssert(cudaEventRecord(startEvent, 0));
  for (int i = 0; i < count; i++) {
    kernel<<<grid_size, blk_size>>>(args...);
    GPUAssert(cudaStreamSynchronize(0));
  }
  GPUAssert(cudaEventRecord(stopEvent, 0));
  GPUAssert(cudaEventSynchronize(stopEvent));
  cudaProfilerStop();
  GPUAssert(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  printf("Time for executing %s %d times: %f (ms)\n", name.c_str(), count, ms);
  GPUAssert(cudaEventDestroy(startEvent));
  GPUAssert(cudaEventDestroy(stopEvent));
}

inline cudaError_t get_grid_size_by_array_size(int64_t n, int block_size, int *num_blocks) {
  constexpr int num_waves = 32;
  int dev;
  GPUAssert(cudaGetDevice(&dev));
  int sm_count;
  GPUAssert(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));
  int max_threads_per_sm;
  GPUAssert(cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, dev));
  *num_blocks = std::max<int>(1, std::min<int64_t>((n + block_size - 1) / block_size,
                                                   max_threads_per_sm * sm_count / block_size * num_waves));
  return cudaSuccess;
}