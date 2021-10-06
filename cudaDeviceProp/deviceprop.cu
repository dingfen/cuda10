
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define CUDA_ERROR_WRAP(x) { gpuAssert((x), __FILE__, __LINE__); }
#define N (1 * 1024)

inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    cudaDeviceProp prop;
    int count = 0;

    CUDA_ERROR_WRAP(cudaGetDeviceCount(&count))

    for (int i = 0; i < count; i++) {
        CUDA_ERROR_WRAP(cudaGetDeviceProperties(&prop, i))
        printf("  --- General Information for device %d ---\n", i);
        printf("Name: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Clock rate: %d\n", prop.clockRate);
        printf("Device copy overlap:  ");
        if (prop.deviceOverlap)
            printf("Enabled.\n");
        else
            printf("Disabled.\n");
        printf("Kernel execution timeout:  ");
        if (prop.kernelExecTimeoutEnabled)
            printf("Enabled.\n");
        else
            printf("Disabled.\n");

        printf("  --- Memory Information for device %d ---\n", i);
        printf("Total global mem: %lu\n", prop.totalGlobalMem);
        printf("Total constant mem: %lu\n", prop.totalConstMem);
        printf("Max mem pitch:  %ld\n", prop.memPitch);
        printf("Texture Alignment:  %ld\n", prop.textureAlignment);

        printf("  --- MP Information for device %d ---\n", i);
        printf("Multiprocessor count:  %d\n", prop.multiProcessorCount);
        printf("Shared mem per mp:  %ld\n", prop.sharedMemPerBlock);
        printf("Registers per mp:  %ld\n", prop.sharedMemPerBlock);
        printf("Threads in wrap:  %d\n", prop.warpSize);
        printf("Max threads per block:  %d\n", prop.maxThreadsPerBlock);
        printf("Max threads dimensions:  (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions:  (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
}
