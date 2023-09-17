#include <stdio.h>
#include <assert.h>

#define GPUAssert(x) gpuAssert((x), __FILE__, __LINE__)

inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}


void profileCopies(float *h_a, float *h_b, float *d, 
                   unsigned int  n, const char *desc) {
    printf("\n%s transfers\n", desc);
    unsigned int bytes = n * sizeof(float);

    // events for timing
    cudaEvent_t startEvent, stopEvent;

    GPUAssert( cudaEventCreate(&startEvent) );
    GPUAssert( cudaEventCreate(&stopEvent) );

    GPUAssert( cudaEventRecord(startEvent, 0) );
    GPUAssert( cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice) );
    GPUAssert( cudaEventRecord(stopEvent, 0) );
    GPUAssert( cudaEventSynchronize(stopEvent) );

    float time;
    GPUAssert( cudaEventElapsedTime(&time, startEvent, stopEvent) );
    printf("  Host to Device bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

    GPUAssert( cudaEventRecord(startEvent, 0) );
    GPUAssert( cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost) );
    GPUAssert( cudaEventRecord(stopEvent, 0) );
    GPUAssert( cudaEventSynchronize(stopEvent) );

    GPUAssert( cudaEventElapsedTime(&time, startEvent, stopEvent) );
    printf("  Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

    for (int i = 0; i < n; ++i) {
      if (h_a[i] != h_b[i]) {
        printf("*** %s transfers failed ***\n", desc);
        break;
      }
    }

    // clean up events
    GPUAssert( cudaEventDestroy(startEvent) );
    GPUAssert( cudaEventDestroy(stopEvent) );
}


int main() {
    unsigned int nElements = 4*1024*1024;
    const unsigned int bytes = nElements * sizeof(float);
    
    // host arrays
    float *h_aPageable, *h_bPageable;   
    float *h_aPinned, *h_bPinned;
    
    // device array
    float *d_a;
    
    // allocate and initialize
    h_aPageable = (float*)malloc(bytes);                    // host pageable
    h_bPageable = (float*)malloc(bytes);                    // host pageable
    GPUAssert( cudaMallocHost((void**)&h_aPinned, bytes) ); // host pinned
    GPUAssert( cudaMallocHost((void**)&h_bPinned, bytes) ); // host pinned
    GPUAssert( cudaMalloc((void**)&d_a, bytes) );           // device
    
    for (int i = 0; i < nElements; ++i) h_aPageable[i] = i;      
    memcpy(h_aPinned, h_aPageable, bytes);
    memset(h_bPageable, 0, bytes);
    memset(h_bPinned, 0, bytes);
    
    // output device info and transfer size
    cudaDeviceProp prop;
    GPUAssert( cudaGetDeviceProperties(&prop, 0) );
    
    printf("\nDevice: %s\n", prop.name);
    printf("Transfer size (MB): %d\n", bytes / (1024 * 1024));
    
    // perform copies and report bandwidth
    profileCopies(h_aPageable, h_bPageable, d_a, nElements, "Pageable");
    profileCopies(h_aPinned, h_bPinned, d_a, nElements, "Pinned");
    
    printf("n");
    
    // cleanup
    cudaFree(d_a);
    cudaFreeHost(h_aPinned);
    cudaFreeHost(h_bPinned);
    free(h_aPageable);
    free(h_bPageable);
    
    return 0;
}