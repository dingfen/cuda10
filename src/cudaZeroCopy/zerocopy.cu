#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

#define GPUAssert(x) gpuAssert((x), __FILE__, __LINE__)

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

float maxError(float *a, int n) {
    float maxE = 0.;
    for (int i = 0; i < n; i++) {
    float error = fabs(a[i] - 1.0f);
    if (error > maxE)
        maxE = error;
    }
    return maxE;
}

__global__ void kernel(float *a, int offset) {
    int i = offset + threadIdx.x + blockIdx.x * blockDim.x;
    float x = (float)i;
    float s = sinf(x);
    float c = cosf(x);
    a[i] = a[i] + sqrtf(s * s + c * c);
}

int main(int argc, char **argv) {
    const int nBytes = 4 * 1024 * 1024 * sizeof(float);
    int gridSize = 16;
    int blockSize = 256;
    float *a_h, *a_map;
    float ms;
    cudaDeviceProp prop;
    cudaEvent_t startEvent, stopEvent;

    // initialize
    GPUAssert(cudaGetDeviceProperties(&prop, 0));
    if (!prop.canMapHostMemory) {
        printf("Sorry, but your device is not able to use zero copy.\n");
        exit(0);
    }
    GPUAssert(cudaSetDeviceFlags(cudaDeviceMapHost));
    GPUAssert(cudaHostAlloc(&a_h, nBytes, cudaHostAllocMapped));
    GPUAssert(cudaEventCreate(&startEvent));
    GPUAssert(cudaEventCreate(&stopEvent));

    // sequential execute
    memset(a_h, 0, nBytes);
    GPUAssert(cudaMalloc((void **)&a_map, nBytes));
    GPUAssert(cudaEventRecord(startEvent, 0));
    GPUAssert(cudaMemcpy(a_map, a_h, nBytes, cudaMemcpyHostToDevice));
    kernel<<<gridSize, blockSize>>>(a_map, 0);
    GPUAssert(cudaMemcpy(a_h, a_map, nBytes, cudaMemcpyDeviceToHost));
    GPUAssert(cudaEventRecord(stopEvent, 0));
    GPUAssert(cudaEventSynchronize(stopEvent));
    GPUAssert(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("Time for MemCpy execute (ms): %f\n", ms);
    printf("  max error: %e\n", maxError(a_h, gridSize * blockSize));

    // memcpy async
    int nStreams = 4;
    cudaStream_t stream[nStreams];
    const int streamSize = 4 * 1024 * 1024 / nStreams;
    const int streamBytes = streamSize * sizeof(float);
    memset(a_h, 0, nBytes);
    for (int i = 0; i < nStreams; ++i)
        GPUAssert(cudaStreamCreate(&stream[i]));

    GPUAssert(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < nStreams; ++i) {
        int offset = i * streamSize;
        GPUAssert(cudaMemcpyAsync(&a_map[offset], &a_h[offset],
                                streamBytes, cudaMemcpyHostToDevice,
                                stream[i]));
    }
    for (int i = 0; i < nStreams; ++i) {
        int offset = i * streamSize;
        kernel<<<gridSize, blockSize, 0, stream[i]>>>(a_map, offset);
    }
    for (int i = 0; i < nStreams; ++i) {
        int offset = i * streamSize;
        GPUAssert(cudaMemcpyAsync(&a_h[offset], &a_map[offset],
                                    streamBytes, cudaMemcpyDeviceToHost,
                                    stream[i]));
    }
    GPUAssert(cudaEventRecord(stopEvent, 0));
    GPUAssert(cudaEventSynchronize(stopEvent));
    GPUAssert(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("Time for asynchronous V2 transfer and execute (ms): %f\n", ms);
    printf("  max error: %e\n", maxError(a_h, gridSize * blockSize));

    // zero copy
    memset(a_h, 0, nBytes);
    GPUAssert(cudaHostGetDevicePointer(&a_map, a_h, 0));
    GPUAssert(cudaEventRecord(startEvent, 0));
    kernel<<<gridSize, blockSize>>>(a_map, 0);
    GPUAssert(cudaEventRecord(stopEvent, 0));
    GPUAssert(cudaEventSynchronize(stopEvent));
    GPUAssert(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("Time for zero copy execute (ms): %f\n", ms);
    printf("  max error: %e\n", maxError(a_h, gridSize * blockSize));

    // cleanup
    GPUAssert(cudaEventDestroy(startEvent));
    GPUAssert(cudaEventDestroy(stopEvent));
    cudaFree(a_map);
    cudaFreeHost(a_h);

    return 0;
}