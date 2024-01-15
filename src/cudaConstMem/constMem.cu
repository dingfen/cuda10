
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define GPUAssert(x) gpuAssert((x), __FILE__, __LINE__)

inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

int main()
{
    float time;
    cudaEvent_t start, stop;
    GPUAssert(cudaEventCreate(&start));
    GPUAssert(cudaEventCreate(&stop));
    GPUAssert(cudaEventRecord(start, 0));

    GPUAssert(cudaEventRecord(stop, 0));
    GPUAssert(cudaEventElapsedTime(&time, start, stop));
}

