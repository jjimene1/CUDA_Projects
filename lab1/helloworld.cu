// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

__global__ void mykernel(void) {}

int main(void) {
    
    mykernel<<<1, 1>>>();
    printf("Hello World!\n");
    return 0;
}