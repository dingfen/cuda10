
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define GPUAssert(x) gpuAssert((x), __FILE__, __LINE__)

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

__global__ void kernel(float *a, int offset)
{
  int i = offset + threadIdx.x + blockIdx.x * blockDim.x;
  float x = (float)i;
  float s = sinf(x);
  float c = cosf(x);
  a[i] = a[i] + sqrtf(s * s + c * c);
}

float maxError(float *a, int n)
{
  float maxE = 0;
  for (int i = 0; i < n; i++)
  {
    float error = fabs(a[i] - 1.0f);
    if (error > maxE)
      maxE = error;
  }
  return maxE;
}

int main(int argc, char **argv)
{
  const int blockSize = 256, nStreams = 4;
  const int n = 4 * 1024 * blockSize * nStreams;
  const int streamSize = n / nStreams;
  const int streamBytes = streamSize * sizeof(float);
  const int bytes = n * sizeof(float);

  int devId = 0;
  if (argc > 1)
    devId = atoi(argv[1]);

  cudaDeviceProp prop;
  GPUAssert(cudaGetDeviceProperties(&prop, devId));
  printf("Device : %s\n", prop.name);
  GPUAssert(cudaSetDevice(devId));

  // allocate pinned host memory and device memory
  float *a, *d_a;
  GPUAssert(cudaMallocHost((void **)&a, bytes)); // host pinned
  GPUAssert(cudaMalloc((void **)&d_a, bytes));   // device

  float ms; // elapsed time in milliseconds

  // create events and streams
  cudaEvent_t startEvent, stopEvent, dummyEvent;
  cudaStream_t stream[nStreams];
  GPUAssert(cudaEventCreate(&startEvent));
  GPUAssert(cudaEventCreate(&stopEvent));
  GPUAssert(cudaEventCreate(&dummyEvent));
  for (int i = 0; i < nStreams; ++i)
    GPUAssert(cudaStreamCreate(&stream[i]));

  // baseline case - sequential transfer and execute
  memset(a, 0, bytes);
  GPUAssert(cudaEventRecord(startEvent, 0));
  GPUAssert(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
  kernel<<<n / blockSize, blockSize>>>(d_a, 0);
  GPUAssert(cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost));
  GPUAssert(cudaEventRecord(stopEvent, 0));
  GPUAssert(cudaEventSynchronize(stopEvent));
  GPUAssert(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  printf("Time for sequential transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));

  // asynchronous version 1: loop over {copy, kernel, copy}
  memset(a, 0, bytes);
  GPUAssert(cudaEventRecord(startEvent, 0));
  for (int i = 0; i < nStreams; ++i)
  {
    int offset = i * streamSize;
    GPUAssert(cudaMemcpyAsync(&d_a[offset], &a[offset],
                              streamBytes, cudaMemcpyHostToDevice,
                              stream[i]));
    kernel<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
    GPUAssert(cudaMemcpyAsync(&a[offset], &d_a[offset],
                              streamBytes, cudaMemcpyDeviceToHost,
                              stream[i]));
  }
  GPUAssert(cudaEventRecord(stopEvent, 0));
  GPUAssert(cudaEventSynchronize(stopEvent));
  GPUAssert(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  printf("Time for asynchronous V1 transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));

  // asynchronous version 2:
  // loop over copy, loop over kernel, loop over copy
  memset(a, 0, bytes);
  GPUAssert(cudaEventRecord(startEvent, 0));
  for (int i = 0; i < nStreams; ++i)
  {
    int offset = i * streamSize;
    GPUAssert(cudaMemcpyAsync(&d_a[offset], &a[offset],
                              streamBytes, cudaMemcpyHostToDevice,
                              stream[i]));
  }
  for (int i = 0; i < nStreams; ++i)
  {
    int offset = i * streamSize;
    kernel<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
  }
  for (int i = 0; i < nStreams; ++i)
  {
    int offset = i * streamSize;
    GPUAssert(cudaMemcpyAsync(&a[offset], &d_a[offset],
                              streamBytes, cudaMemcpyDeviceToHost,
                              stream[i]));
  }
  GPUAssert(cudaEventRecord(stopEvent, 0));
  GPUAssert(cudaEventSynchronize(stopEvent));
  GPUAssert(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  printf("Time for asynchronous V2 transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));

  // cleanup
  GPUAssert(cudaEventDestroy(startEvent));
  GPUAssert(cudaEventDestroy(stopEvent));
  GPUAssert(cudaEventDestroy(dummyEvent));
  for (int i = 0; i < nStreams; ++i)
    GPUAssert(cudaStreamDestroy(stream[i]));
  cudaFree(d_a);
  cudaFreeHost(a);

  return 0;
}