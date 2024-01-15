
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


template<typename T>
inline T min(T a, T b) {
    return a < b ? a : b;
}


const int N = 33*1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = min<int>(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(float* c, float* a, float* b) {
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    // gird loops addition, reduce sum in cache for every thread.
    float temp = 0.0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = temp;
    // thread synchronize
    __syncthreads();

    // Reductions parallel as Binary tree in each block
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
    // for not waste computing resource, exit now.
}


extern "C" cudaError_t dotproduct(float* ans, float* a, float* b) {
    float* dev_a = NULL;
    float* dev_b = NULL;
    float* dev_c = NULL;
    float* c = NULL;
    float time;
    float tmp = 0.;
    float kahan = 0.;
    c = (float*)malloc(blocksPerGrid * sizeof(float));

    cudaEvent_t start, stop;
    GPUAssert(cudaEventCreate(&start));
    GPUAssert(cudaEventCreate(&stop));
    GPUAssert(cudaEventRecord(start, 0));

    GPUAssert(cudaMalloc(&dev_a, N * sizeof(float)));
    GPUAssert(cudaMalloc(&dev_b, N * sizeof(float)));
    GPUAssert(cudaMalloc(&dev_c, blocksPerGrid * sizeof(float)));

    GPUAssert(cudaSetDevice(0));

    GPUAssert(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
    GPUAssert(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));

    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_c, dev_a, dev_b);

    GPUAssert(cudaGetLastError());

    GPUAssert(cudaDeviceSynchronize());

    GPUAssert(cudaMemcpy(c, dev_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));

    GPUAssert(cudaEventRecord(stop, 0));
    GPUAssert(cudaEventSynchronize(stop));
    GPUAssert(cudaEventElapsedTime(&time, start, stop));
    // last reduction for cpu
    // use kahan sum to reduce eff
    for (int i = 0; i < blocksPerGrid; i++) {
        float y = c[i] - kahan;
        float t = *ans + y;
        kahan = (t - *ans) - y;
        *ans = t;
    }

    printf("GPU time: %8.4f\n", time);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(c);
    return cudaSuccess;
}


int main() {
    float* a = NULL;
    float* b = NULL;
    float ans = 0.;

    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));
    srand(time(0));
    for (int i = 0; i < N; i++) {
        a[i] = rand() * (4.0 / RAND_MAX) - 2.0;
        b[i] = rand() * (4.0 / RAND_MAX) - 2.0;
    }

    if (dotproduct(&ans, a, b) == cudaSuccess) {
        printf("GPU: %8.4f\n", ans);
    }
    else {
        printf("GPU Assertion failure.\n");
    }

    float cpu_ans = 0.;
    float y = 0.;
    float kahan = 0.;

    // use kahan sum to reduce eff
    for (int i = 0; i < N; i++) {
        float y = (a[i]*b[i]) - kahan;
        float t = cpu_ans + y;
        kahan = (t - cpu_ans) - y;
        cpu_ans = t;
    }
    // for (int i = 0; i < N; i++) {
    //     cpu_ans += a[i] * b[i];
    // }
    printf("CPU: %8.4f\n", cpu_ans);

    free(a);
    free(b);
}