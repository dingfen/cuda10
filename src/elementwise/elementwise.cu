#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"

#include <stdio.h>
#include <stdlib.h>

#define GPUAssert(x) gpuAssert((x), __FILE__, __LINE__)

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

__global__ void relu_kernel(float *input, float *output) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = input[idx] < 0 ? 0 : input[idx];
}

int main() {
    float *input;
    float *output;
    int32_t elem_cnt = 3 * 224 * 224;
    int loop_times = 100;
    cudaEvent_t startEvent, stopEvent;
    float ms; // elapsed time in milliseconds

    cudaMalloc(&input, sizeof(float) * elem_cnt);
    cudaMalloc(&output, sizeof(float) * elem_cnt);
    GPUAssert(cudaEventCreate(&startEvent));
    GPUAssert(cudaEventCreate(&stopEvent));

    int32_t thread_num = 256;
    int32_t grid_size = (elem_cnt + thread_num - 1) / thread_num;
    
    cudaProfilerStart();
    GPUAssert(cudaEventRecord(startEvent, 0));
    for(int i = 0; i < loop_times; i++) {
        relu_kernel<<<grid_size, thread_num>>>(input, output);
        GPUAssert(cudaStreamSynchronize(0));
    }
    GPUAssert(cudaEventRecord(stopEvent, 0));
    GPUAssert(cudaEventSynchronize(stopEvent));
    cudaProfilerStop();
    GPUAssert(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("Time for sequential transfer and execute (ms): %f\n", ms);

    cudaFree(input);
    cudaFree(output);
    return 0;
}