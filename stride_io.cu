#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <time.h>
#include <unistd.h>
#include <vector>

#define GPUAssert(ans) gpuAssert((ans), __FILE__, __LINE__)

inline void gpuAssert(cudaError_t code, const char* file, int line, bool debug = true) {
  if (debug && code != cudaSuccess) {
    fprintf(stderr, "GPU Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
    assert(code == cudaSuccess);
  }
}

__global__ void load_global_withstride(long long int *result, char *src, unsigned long *offset, unsigned long loop_num,
                                       unsigned long load_num, unsigned long range, char *data) {
  int idx = (threadIdx.x + blockDim.x * blockIdx.x) * loop_num;
  char d;
  long long int start, stop;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start) :: "memory");
  for (int i = 0; i < loop_num; i++) {
    d = src[offset[(i+idx)%load_num]];
    d++;
  }
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stop) :: "memory");
  *result = stop - start;
  *data = d;
}




__global__ void load_global(long long int *result, char *src, unsigned long loop_num,
                            unsigned long load_num, unsigned long range, char *data) {
  int idx = (threadIdx.x + blockDim.x * blockIdx.x) * loop_num;
  char d;
  long long int start, stop;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start) :: "memory");
  for (int i = 0; i < loop_num; i++) {
    d = src[(i+idx)%load_num];
    d++;
  }
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stop) :: "memory");
  *result = stop - start;
  *data = d;
}
