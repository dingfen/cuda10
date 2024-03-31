#include "util.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

__global__ void reduceAdd(float* input, float* output, int size) {
    int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid >= size) return;
    for (int i = 1; i < size; i *= 2) {
        if (threadid % (2 * i) == 0) {
            input[threadid] += input[threadid + i];
        }
        __syncthreads();
    }
    output[0] = input[0];
}

int main() {
    // Initialize input array
    int size = 1024;
    float* input = (float*)malloc(size * sizeof(float));
    float* input_d;
    float* output;
    float* output_d;
    for (int i = 0; i < size; i++) {
        input[i] = 1.0f;
    }
    output = (float*)malloc(sizeof(float));

    // allocate memory on cuda device
    GPUAssert(cudaMalloc((void**)&input_d, size * sizeof(float)));
    GPUAssert(cudaMalloc((void**)&output_d, sizeof(float)));

    // copy input to device
    GPUAssert(cudaMemcpy(input_d, input, size * sizeof(float), cudaMemcpyHostToDevice));
    
    // launch kernel
    dim3 block_dim(16, 1, 1);
    dim3 thread_dim(128, 1, 1);
    reduceAdd<<<block_dim, thread_dim>>>(input_d, output_d, size);

    // copy result back to host
    GPUAssert(cudaMemcpy(output, output_d, sizeof(float), cudaMemcpyDeviceToHost));

    // verify result
    float expected_output = 0.0f;
    for (int i = 0; i < size; i++) {
        expected_output += input[i];
    }
    printf("Expected output: %f\n", expected_output);
    printf("Actual output: %f\n", output[0]);
    printf("Difference: %f\n", fabs(expected_output - output[0]));
    // free device memory
    GPUAssert(cudaFree(input_d));
    GPUAssert(cudaFree(output_d));
    free(input);
    free(output);

    return 0;
}