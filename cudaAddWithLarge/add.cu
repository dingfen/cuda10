
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define CUDA_ERROR_WRAP(x) gpuAssert((x), __FILE__, __LINE__)
const int N = 33*1024;

inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

__global__ void addKernel(int* c, int* a, int* b)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

extern "C" void addWithCUDA(int* c, int* a, int* b) {
    addKernel<<<128, 128 >>>(c, a, b);
}

int main()
{
    int a[N];
    int b[N];
    int c[N];
    int* dev_a, * dev_b, * dev_c;

    CUDA_ERROR_WRAP(cudaMalloc(&dev_a, N * sizeof(int)));
    CUDA_ERROR_WRAP(cudaMalloc(&dev_b, N * sizeof(int)));
    CUDA_ERROR_WRAP(cudaMalloc(&dev_c, N * sizeof(int)));

    for (int i = 0; i < N; i++) {
        a[i] = i+2;
        b[i] = i * 11;
    }

    CUDA_ERROR_WRAP(cudaSetDevice(0));

    CUDA_ERROR_WRAP(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_ERROR_WRAP(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

    addWithCUDA(dev_c, dev_a, dev_b);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    CUDA_ERROR_WRAP(cudaDeviceSynchronize());

    CUDA_ERROR_WRAP(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

    bool success = true;
    for (int i = 0; i < N; i++) {
        if (a[i] + b[i] != c[i]) {
            printf("Error: %d + %d != %d at index %d\n", a[i], b[i], c[i], i);
            success = false;
        }
    }

    if (success) {
        printf("We did it!\n");
    }

    CUDA_ERROR_WRAP(cudaDeviceReset());
   
Error:
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}
