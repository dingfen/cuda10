#include "util.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

void matmul_cpu(float* c, float* a, float* b, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// cuda kernel function
__global__ void matmul_naive_gpu(float* c, float* a, float* b, int m, int n, int k) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.0f;
    for(int i = 0; i < k; i++) {
        v += a[x * k + i] * b[i * n + y];
    }
    c[x * n + y] = v;
}

__global__ void matmul_inner_product_gpu(float* c, float* a, float* b, int m, int n, int k) {
    int thread_num_x = gridDim.x * blockDim.x;
    int thread_num_y = gridDim.y * blockDim.y;
    int km = (m + thread_num_x - 1) / thread_num_x;
    int kn = (n + thread_num_x - 1) / thread_num_x;
    int y_grid_loop = (k + thread_num_y - 1) / thread_num_y;
    for (int mm = 0; mm < y_grid_loop; mm++) {
        int y = mm * thread_num_y + blockIdx.y * blockDim.y + threadIdx.y;
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        float v = 0.0f;
        for(int i = 0; i < thread_num_y; i++) {
            v += a[x * k + i] * b[i * n + y];
        }
        c[x * n + y] = v;
    }
}


template<int BLOCK_SIZE>
__global__ void matmul_outer_product_gpu(float* c, float* a, float* b, int m, int n, int k) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = threadIdx.y;
    int tidx = threadIdx.x;
    float v = 0.0f;
    __shared__ float as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float bs[BLOCK_SIZE][BLOCK_SIZE];

    int tilesx = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for(int i = 0; i < tilesx; i++) {
        // load data from global memory to shared memory
        as[tidx][tidy] = a[x * k + i * BLOCK_SIZE + tidy];
        bs[tidx][tidy] = b[(i * BLOCK_SIZE + tidx) * n + y];

        // sync to wait for all threads in one block to finish loading datas
        __syncthreads();

        // sub-matrix multiply
        for(int l = 0; l < BLOCK_SIZE; l++) {
            v += as[tidx][l] * bs[l][tidy];
        }

        // sync to wait for all threads in one block to finish compute
        __syncthreads();
    }

    // store results into global memory
    c[x * n + y] = v;
}

__global__ void matmul_tile2d_gpu(float* c, float* a, float* b, int m, int n, int k) {
    int thread_num_x = gridDim.x * blockDim.x;
    int thread_num_y = gridDim.y * blockDim.y;
    int x_grid_loop = (m + thread_num_x - 1) / thread_num_x;
    int y_grid_loop = (n + thread_num_y - 1) / thread_num_y;
    assert(x_grid_loop == y_grid_loop);
    for (int mm = 0; mm < x_grid_loop; mm++) {
        for(int nn = 0; nn < y_grid_loop; nn++) {
            int x = mm * thread_num_x + blockIdx.x * blockDim.x + threadIdx.x;
            int y = nn * thread_num_y + blockIdx.y * blockDim.y + threadIdx.y;
            float v = 0.0f;
            for(int i = 0; i < k; i++) {
                v += a[x * k + i] * b[i * n + y];
            }
            c[x * n + y] = v;
        }
    }
}


template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y, int K>
__global__ void matmul_tile2d_shared_gpu(float* c, float* a, float* b, int m, int n, int k) {
    int thread_num_x = gridDim.x * blockDim.x;
    int thread_num_y = gridDim.y * blockDim.y;
    int tidy = threadIdx.y;
    int tidx = threadIdx.x;
    int x_grid_loop = (m + thread_num_x - 1) / thread_num_x;
    int y_grid_loop = (n + thread_num_y - 1) / thread_num_y;
    __shared__ float as[BLOCK_SIZE_X][K];
    __shared__ float bs[K][BLOCK_SIZE_Y];
    assert(x_grid_loop == y_grid_loop);
    for (int mm = 0; mm < x_grid_loop; mm++) {
        for(int nn = 0; nn < y_grid_loop; nn++) {
            int x = mm * thread_num_x + blockIdx.x * blockDim.x + threadIdx.x;
            int y = nn * thread_num_y + blockIdx.y * blockDim.y + threadIdx.y;
            // load data from global memory to shared memory
            for (int i = 0; i < k; i++) {
                as[tidx][i] = a[x * k + i];
                bs[i][tidy] = b[i * n + y];
            }
            // sub-matrix multiply
            float v = 0.0f;
            for(int i = 0; i < k; i++) {
                v += as[tidx][i] * bs[i][tidy];
            }
            c[x * n + y] = v;
        }
    }
}

__global__ void matmul_inner_product_thd_div_gpu(float* c, float* a, float* b, int m, int n, int k) {
    int x_grid_loop = (m + gridDim.x - 1) / gridDim.x;
    int y_grid_loop = (n + gridDim.y - 1) / gridDim.y;
    assert(x_grid_loop == y_grid_loop);
    for (int mm = 0; mm < x_grid_loop; mm++) {
        for(int nn = 0; nn < y_grid_loop; nn++) {
            const int threadid = threadIdx.x + threadIdx.y * blockDim.x;
            const int thread_num = blockDim.x * blockDim.y;
            const int num = (k + thread_num - 1) / thread_num > 1 ? (k + thread_num - 1) / thread_num : 1;
            int row_id = mm * gridDim.x + blockIdx.x;
            int col_id = nn * gridDim.y + blockIdx.y;
            // get A row & B col
            float *row_buf = new float[num+1];
            for (int i = 0; i < num; i++) {
                row_buf[i] = a[row_id * k + num * threadid + i];
            }
            float *col_buf = new float[num+1];
            for (int i = 0; i < num; i++) {
                col_buf[i] = b[n * (num * threadid + i) + col_id];
            }
            // compute inner product
            __shared__ float values[256];
            float v = 0.0f;
            for(int i = 0; i < num; i++) {
                v += row_buf[i] * col_buf[i];
            }
            values[threadid] = v;
            // sync in thread block & reduce the result
            __syncthreads();
            float cas = 0.0f;
            for (int i = 1; i < thread_num; i *= 2) {
                if (threadid % (2 * i) == 0) {
                    float tmp = values[threadid + i] - cas;
                    float add = values[threadid] + tmp;
                    cas = (add - values[threadid]) - tmp;
                    values[threadid] = add;
                }
                __syncthreads();
            }
            if (threadid == 0) {
                c[row_id * n + col_id] = values[0];
            }
            delete[] row_buf;
            delete[] col_buf;
        }
    }
}

// test matmul_naive_gpu
int test_matmul_naive_gpu() {
    int m = 128;
    int n = 128;
    int k = 128;

    float* a = (float*)malloc(m * k * sizeof(float));
    float* b = (float*)malloc(k * n * sizeof(float));
    float* c = (float*)malloc(m * n * sizeof(float));
    float* d_a, *d_b, *d_c;


    // initialize a and b
    for (int i = 0; i < m * k; i++) {
        a[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < k * n; i++) {
        b[i] = rand() / (float)RAND_MAX;
    }
    // initialize c to zero
    for (int i = 0; i < m * n; i++) {
        c[i] = 0.0f;
    }

    // allocate memory on device
    GPUAssert(cudaMalloc((void**)&d_a, m * k * sizeof(float)));
    GPUAssert(cudaMalloc((void**)&d_b, k * n * sizeof(float)));
    GPUAssert(cudaMalloc((void**)&d_c, m * n * sizeof(float)));

    // copy data to device
    GPUAssert(cudaMemcpy(d_a, a, m * k * sizeof(float), cudaMemcpyHostToDevice));
    GPUAssert(cudaMemcpy(d_b, b, k * n * sizeof(float), cudaMemcpyHostToDevice));
    GPUAssert(cudaMemcpy(d_c, c, m * n * sizeof(float), cudaMemcpyHostToDevice));

    // launch kernel
    dim3 block(16, 16);
    const int loop_times = 10;
    // int grid;
    // GPUAssert(get_grid_size_by_array_size(m * n, block.x * block.y, &grid));
    dim3 grid_size = dim3((m + block.x - 1) / block.x,
                          (n + block.y - 1) / block.y, 1);
    perf_helper_func<loop_times, float*, float*, float*, int, int, int>(
                     "matmul_naive_gpu", 
                     grid_size, block,
                     matmul_naive_gpu, d_c, d_a, d_b, m, n, k);

    // copy result back to host
    GPUAssert(cudaMemcpy(c, d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    // check result
    float* c_cpu = (float*)malloc(m * n * sizeof(float));
    matmul_cpu(c_cpu, a, b, m, n, k);
    int err = 0;
    for (int i = 0; i < m * n; i++) {
        if (fabs(c[i] - c_cpu[i]) > 1e-5) {
            printf("error at %d: %f vs %f\n", i, c[i], c_cpu[i]);
            err++;
        }
    }
    if (err == 0) {
        printf("test_matmul_naive_gpu passed\n");
    } else {
        printf("test_matmul_naive_gpu failed\n");
    }

    // free memory
    free(a);
    free(b);
    free(c);
    free(c_cpu);
    GPUAssert(cudaFree(d_a));
    GPUAssert(cudaFree(d_b));
    GPUAssert(cudaFree(d_c));

    return err;
}

// test test_matmul_inner_product_thd_div_gpu
int test_matmul_inner_product_thd_div_gpu() {
    int m = 128;
    int n = 128;
    int k = 128;

    float* a = (float*)malloc(m * k * sizeof(float));
    float* b = (float*)malloc(k * n * sizeof(float));
    float* c = (float*)malloc(m * n * sizeof(float));
    float* d_a, *d_b, *d_c;

    // initialize a and b
    for (int i = 0; i < m * k; i++) {
        a[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < k * n; i++) {
        b[i] = rand() / (float)RAND_MAX;
    }

    // allocate memory on device
    GPUAssert(cudaMalloc((void**)&d_a, m * k * sizeof(float)));
    GPUAssert(cudaMalloc((void**)&d_b, k * n * sizeof(float)));
    GPUAssert(cudaMalloc((void**)&d_c, m * n * sizeof(float)));

    // copy data to device
    GPUAssert(cudaMemcpy(d_a, a, m * k * sizeof(float), cudaMemcpyHostToDevice));
    GPUAssert(cudaMemcpy(d_b, b, k * n * sizeof(float), cudaMemcpyHostToDevice));
    GPUAssert(cudaMemcpy(d_c, c, m * n * sizeof(float), cudaMemcpyHostToDevice));

    // launch kernel
    dim3 block(128);
    const int loop_times = 1;
    // int grid;
    // GPUAssert(get_grid_size_by_array_size(m * n, block.x * block.y, &grid));
    dim3 grid_size = dim3(16, 16, 1);
    perf_helper_func<loop_times, float*, float*, float*, int, int, int>(
                     "matmul_inner_product_thd_div_gpu", 
                     grid_size, block,
                     matmul_inner_product_thd_div_gpu, d_c, d_a, d_b, m, n, k);
    // matmul_inner_product_thd_div_gpu<<<grid_size, block>>>(d_c, d_a, d_b, m, n, k);

    // copy result back to host
    GPUAssert(cudaMemcpy(c, d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    // check result
    float* c_cpu = (float*)malloc(m * n * sizeof(float));
    matmul_cpu(c_cpu, a, b, m, n, k);
    int err = 0;
    for (int i = 0; i < m * n; i++) {
        if (fabs(c[i] - c_cpu[i]) > 1e-2) {
            printf("error at %d: %f vs %f\n", i, c[i], c_cpu[i]);
            err++;
        }
    }
    if (err == 0) {
        printf("test_matmul_inner_product_gpu passed\n");
    } else {
        printf("test_matmul_inner_product_gpu failed\n");
    }

    // free memory
    free(a);
    free(b);
    free(c);
    free(c_cpu);
    GPUAssert(cudaFree(d_a));
    GPUAssert(cudaFree(d_b));
    GPUAssert(cudaFree(d_c));

    return err;
}


int test_matmul_tile2d_gpu() {
    int m = 256;
    int n = 256;
    int k = 256;

    float* a = (float*)malloc(m * k * sizeof(float));
    float* b = (float*)malloc(k * n * sizeof(float));
    float* c = (float*)malloc(m * n * sizeof(float));
    float* d_a, *d_b, *d_c;

    // initialize a and b
    for (int i = 0; i < m * k; i++) {
        a[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < k * n; i++) {
        b[i] = rand() / (float)RAND_MAX;
    }
    // initialize c to zero
    for (int i = 0; i < m * n; i++) {
        c[i] = 0.0f;
    }

    // allocate memory on device
    GPUAssert(cudaMalloc((void**)&d_a, m * k * sizeof(float)));
    GPUAssert(cudaMalloc((void**)&d_b, k * n * sizeof(float)));
    GPUAssert(cudaMalloc((void**)&d_c, m * n * sizeof(float)));

    // copy data to device
    GPUAssert(cudaMemcpy(d_a, a, m * k * sizeof(float), cudaMemcpyHostToDevice));
    GPUAssert(cudaMemcpy(d_b, b, k * n * sizeof(float), cudaMemcpyHostToDevice));
    GPUAssert(cudaMemcpy(d_c, c, m * n * sizeof(float), cudaMemcpyHostToDevice));

    // launch kernel
    dim3 block(16, 16);
    const int loop_times = 10;
    // int grid;
    // GPUAssert(get_grid_size_by_array_size(m * n, block.x * block.y, &grid));
    dim3 grid_size = dim3(8, 8, 1);
    perf_helper_func<loop_times, float*, float*, float*, int, int, int>(
                     "matmul_tile2d_gpu", 
                     grid_size, block,
                     matmul_tile2d_gpu, d_c, d_a, d_b, m, n, k);

    // copy result back to host
    GPUAssert(cudaMemcpy(c, d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    // check result
    float* c_cpu = (float*)malloc(m * n * sizeof(float));
    matmul_cpu(c_cpu, a, b, m, n, k);
    int err = 0;
    for (int i = 0; i < m * n; i++) {
        if (fabs(c[i] - c_cpu[i]) > 1e-2) {
            printf("error at %d: %f vs %f\n", i, c[i], c_cpu[i]);
            err++;
        }
    }
    if (err == 0) {
        printf("test_matmul_tile2d_gpu passed\n");
    } else {
        printf("test_matmul_tile2d_gpu failed\n");
    }

    // free memory
    free(a);
    free(b);
    free(c);
    free(c_cpu);
    GPUAssert(cudaFree(d_a));
    GPUAssert(cudaFree(d_b));
    GPUAssert(cudaFree(d_c));

    return err;
}


int test_matmul_outer_product_gpu() {
    int m = 256;
    int n = 256;
    int k = 256;

    float* a = (float*)malloc(m * k * sizeof(float));
    float* b = (float*)malloc(k * n * sizeof(float));
    float* c = (float*)malloc(m * n * sizeof(float));
    float* d_a, *d_b, *d_c;

    // initialize a and b
    for (int i = 0; i < m * k; i++) {
        a[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < k * n; i++) {
        b[i] = rand() / (float)RAND_MAX;
    }
    // initialize c to zero
    for (int i = 0; i < m * n; i++) {
        c[i] = 0.0f;
    }

    // allocate memory on device
    GPUAssert(cudaMalloc((void**)&d_a, m * k * sizeof(float)));
    GPUAssert(cudaMalloc((void**)&d_b, k * n * sizeof(float)));
    GPUAssert(cudaMalloc((void**)&d_c, m * n * sizeof(float)));

    // copy data to device
    GPUAssert(cudaMemcpy(d_a, a, m * k * sizeof(float), cudaMemcpyHostToDevice));
    GPUAssert(cudaMemcpy(d_b, b, k * n * sizeof(float), cudaMemcpyHostToDevice));
    GPUAssert(cudaMemcpy(d_c, c, m * n * sizeof(float), cudaMemcpyHostToDevice));

    // launch kernel
    dim3 block(16, 16);
    const int loop_times = 10;
    // int grid;
    // GPUAssert(get_grid_size_by_array_size(m * n, block.x * block.y, &grid));
    dim3 grid_size = dim3(16, 16, 1);
    perf_helper_func<loop_times, float*, float*, float*, int, int, int>(
                     "matmul_outer_product_gpu", 
                     grid_size, block,
                     matmul_outer_product_gpu<16>, d_c, d_a, d_b, m, n, k);

    // copy result back to host
    GPUAssert(cudaMemcpy(c, d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    // check result
    float* c_cpu = (float*)malloc(m * n * sizeof(float));
    matmul_cpu(c_cpu, a, b, m, n, k);
    int err = 0;
    for (int i = 0; i < m * n; i++) {
        if (fabs(c[i] - c_cpu[i]) > 1e-2) {
            printf("error at %d: %f vs %f\n", i, c[i], c_cpu[i]);
            err++;
        }
    }
    if (err == 0) {
        printf("test_matmul_outer_product_gpu passed\n");
    } else {
        printf("test_matmul_outer_product_gpu failed\n");
    }

    // free memory
    free(a);
    free(b);
    free(c);
    free(c_cpu);
    GPUAssert(cudaFree(d_a));
    GPUAssert(cudaFree(d_b));
    GPUAssert(cudaFree(d_c));

    return err;
}


int test_matmul_tile2d_shared_gpu() {
    int m = 128;
    int n = 128;
    int k = 128;

    float* a = (float*)malloc(m * k * sizeof(float));
    float* b = (float*)malloc(k * n * sizeof(float));
    float* c = (float*)malloc(m * n * sizeof(float));
    float* d_a, *d_b, *d_c;

    // initialize a and b
    for (int i = 0; i < m * k; i++) {
        a[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < k * n; i++) {
        b[i] = rand() / (float)RAND_MAX;
    }
    // initialize c to zero
    for (int i = 0; i < m * n; i++) {
        c[i] = 0.0f;
    }

    // allocate memory on device
    GPUAssert(cudaMalloc((void**)&d_a, m * k * sizeof(float)));
    GPUAssert(cudaMalloc((void**)&d_b, k * n * sizeof(float)));
    GPUAssert(cudaMalloc((void**)&d_c, m * n * sizeof(float)));

    // copy data to device
    GPUAssert(cudaMemcpy(d_a, a, m * k * sizeof(float), cudaMemcpyHostToDevice));
    GPUAssert(cudaMemcpy(d_b, b, k * n * sizeof(float), cudaMemcpyHostToDevice));
    GPUAssert(cudaMemcpy(d_c, c, m * n * sizeof(float), cudaMemcpyHostToDevice));

    // launch kernel
    dim3 block(4, 4);
    const int loop_times = 10;
    // int grid;
    // GPUAssert(get_grid_size_by_array_size(m * n, block.x * block.y, &grid));
    dim3 grid_size = dim3(8, 8, 1);
    perf_helper_func<loop_times, float*, float*, float*, int, int, int>(
                     "matmul_tile2d_shared_gpu", 
                     grid_size, block,
                     matmul_tile2d_shared_gpu<32, 32, 128>, d_c, d_a, d_b, m, n, k);

    // copy result back to host
    GPUAssert(cudaMemcpy(c, d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    // check result
    float* c_cpu = (float*)malloc(m * n * sizeof(float));
    matmul_cpu(c_cpu, a, b, m, n, k);
    int err = 0;
    for (int i = 0; i < m * n; i++) {
        if (fabs(c[i] - c_cpu[i]) > 1e-2) {
            printf("error at %d: %f vs %f\n", i, c[i], c_cpu[i]);
            err++;
        }
    }
    if (err == 0) {
        printf("test_matmul_tile2d_shared_gpu passed\n");
    } else {
        printf("test_matmul_tile2d_shared_gpu failed\n");
    }

    // free memory
    free(a);
    free(b);
    free(c);
    free(c_cpu);
    GPUAssert(cudaFree(d_a));
    GPUAssert(cudaFree(d_b));
    GPUAssert(cudaFree(d_c));

    return err;
}

int main() {
    test_matmul_naive_gpu();
    test_matmul_tile2d_gpu();
    test_matmul_inner_product_thd_div_gpu();
    test_matmul_outer_product_gpu();
    test_matmul_tile2d_shared_gpu();
    return 0;
}