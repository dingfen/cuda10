#include "windows.h"

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda.h"
#include "cuda_gl_interop.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HEIGHT 512
#define WIDTH 512

#define GPUAssert(x) gpuAssert((x), __FILE__, __LINE__)

inline void gpuAssert(cudaError_t code, const char* file, int line) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}

cudaGraphicsResource* cudapbo;
static unsigned ticks = 0;

extern "C" void kernelBindPbo(GLuint pixelBufferObj) {
	GPUAssert(cudaGraphicsGLRegisterBuffer(&cudapbo, pixelBufferObj, cudaGraphicsRegisterFlagsWriteDiscard));
}

extern "C" void kernelExit(GLuint pixelBufferObj) {
	GPUAssert(cudaGLUnregisterBufferObject(pixelBufferObj));
	GPUAssert(cudaGraphicsUnregisterResource(cudapbo));
}

__global__ void kernel(uchar4* map) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int id = x + y * blockDim.x * gridDim.x;
	float fx = x / (float)WIDTH - 0.5f;
	float fy = y / (float)HEIGHT - 0.5f;
	unsigned char green = 128 + 127 * sin(fabs(fx * 100) - fabs(fy * 100));

	map[id].x = 0;
	map[id].y = green;
	map[id].z = 0;
	map[id].w = 255;
}

__global__ void ripple(uchar4* ptr, int ticks) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int off = x + y * blockDim.x * gridDim.x;

	float fx = x - WIDTH / 2;
	float fy = y - HEIGHT / 2;
	float d = sqrtf(fx * fx + fy * fy);
	unsigned char grey = (unsigned char)(128.f + 127.f * cos(d / 10.0f - ticks / 7.0f));

	ptr[off].x = grey;
	ptr[off].y = grey;
	ptr[off].z = grey;
	ptr[off].w = 255;
}

extern "C" void kernelUpdate(int width, int height) {
	uchar4* dev_map;
	ticks++;
	GPUAssert(cudaGraphicsMapResources(1, &cudapbo, NULL));
	GPUAssert(cudaGraphicsResourceGetMappedPointer((void**)&dev_map, NULL, cudapbo));

	dim3 threads(8, 8);
	dim3 grids(width / 8, height / 8);
	//kernel << <grids, threads >> > (dev_map);
	ripple << <grids, threads >> > (dev_map, ticks);

	GPUAssert(cudaDeviceSynchronize());
	GPUAssert(cudaGraphicsUnmapResources(1, &cudapbo, NULL));
}