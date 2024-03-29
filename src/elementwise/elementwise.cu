#include "util.cuh"

#include <stdio.h>
#include <stdlib.h>

__global__ void relu_kernel(float *input, float *output) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = input[idx] < 0 ? 0 : input[idx];
}

int main() {
    float *input;
    float *output;
    int32_t elem_cnt = 32 * 1024 * 1024;
    const int loop_times = 100;

    cudaMalloc(&input, sizeof(float) * elem_cnt);
    cudaMalloc(&output, sizeof(float) * elem_cnt);

    dim3 thread_num = 256;
    dim3 grid_size;
    grid_size.x = (elem_cnt + 256 - 1) / 256;
    grid_size.y = 1;
    grid_size.z = 1;

    printf("grid_size: %d\n", grid_size.x);
    perf_helper_func<loop_times, float*, float*>(
        "relu_kernel", grid_size, thread_num, relu_kernel, input, output
    );

    int grs;
    GPUAssert(get_grid_size_by_array_size(elem_cnt, 256, &grs));

    printf("grid_size: %d\n", grs);
    perf_helper_func<loop_times, float*, float*>(
        "relu_kernel_max_grs", dim3(grs, 1, 1), thread_num, relu_kernel, input, output
    );

    cudaFree(input);
    cudaFree(output);
    return 0;
}