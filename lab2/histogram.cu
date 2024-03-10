#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define BLOCK_SIZE 256

__global__ void histo_kernel(unsigned int *buffer, long size, unsigned int *histogram, unsigned int BinNum){
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   int stride = blockDim.x * gridDim.x;

    int binIntervals = 1024 / BinNum;

    for (int i = tid; i < size; i += stride){
        atomicAdd(&(histogram[buffer[i] / binIntervals]), 1);
    }
}

__global__ void histogram_privatized_kernel(unsigned int *buffer, long size, unsigned int *histogram, unsigned int BinNum){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    //Privatized Bins initialize to 0
    extern __shared__ unsigned int histo_s[];
    for(unsigned int binIdx = threadIdx.x;binIdx < BinNum; binIdx += blockDim.x){
        histo_s[binIdx] = 0u;
    }
    __syncthreads();

    //Define binIntervals

    int binIntervals = 1024 / BinNum;
    int shiftIntervals = log2f(binIntervals); //find base 2 to make the division in the for loop a simple shift operation

    //Histogram calculations
    for(unsigned int i = tid; i < size; i += stride){
        unsigned int bin = buffer[i] >> shiftIntervals;
        atomicAdd(&(histo_s[bin]), 1);
    }
    __syncthreads();

    //Commit to global memory
    for(unsigned int binIdx = threadIdx.x; binIdx < BinNum;binIdx += blockDim.x){
        atomicAdd(&(histogram[binIdx]), histo_s[binIdx]);
    }
}

__global__ void histogram_privatized_aggregation_kernel(unsigned int *buffer, long size, unsigned int *histogram, unsigned int BinNum){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    //Privatized Bins initialize to 0
    extern __shared__ unsigned int histo_s[];
    for(unsigned int binIdx = threadIdx.x;binIdx < BinNum; binIdx += blockDim.x){
        histo_s[binIdx] = 0u;
    }
    __syncthreads();

    //Aggregation Variables 
    int prev_index = -1;
    int accumulator = 0;

    //Define binIntervals
    int binIntervals = 1024 / BinNum;
    int shiftIntervals = log2f(binIntervals); //find base 2 to make the division in the for loop a simple shift operation

    //Histogram calculations
    for(unsigned int i = tid; i < size; i += stride){
        unsigned int bin = buffer[i] >> shiftIntervals;
        if(bin != prev_index){
            if (prev_index != -1 && accumulator > 0) atomicAdd(&(histo_s[bin]), accumulator);
            accumulator = 1;
            prev_index = bin;
        }
        else{
            accumulator++;
        }
    }
    if (accumulator > 0)
        atomicAdd(&histo_s[prev_index], accumulator);
    __syncthreads();

    //Commit to global memory
    for(unsigned int binIdx = threadIdx.x; binIdx < BinNum;binIdx += blockDim.x){
        atomicAdd(&(histogram[binIdx]), histo_s[binIdx]);
    }
}

void sequential_Histogram(unsigned int *data, int length, unsigned int *histogram, unsigned int BinNum){
    for (int i = 0; i < length; i++){
        unsigned int binIntervals = 1024 / BinNum;
        histogram[data[i] / binIntervals]++;
    }
}

int main(int argc, char **argv){
    unsigned int *h_buffer;
    unsigned int *h_histogram;
    unsigned int *cpu_histogram;
    unsigned int *d_buffer;
    unsigned int *d_histogram;

    unsigned int VecDim;
    unsigned int BinNum;

    if (checkCmdLineFlag(argc, (const char **)argv, "VecDim")) {
        VecDim = getCmdLineArgumentInt(argc, (const char **)argv, "VecDim");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "BinNum")) {
        BinNum = getCmdLineArgumentInt(argc, (const char **)argv, "BinNum");
    }

    if ((BinNum & (BinNum - 1)) || BinNum > 256){
        printf("Error: BinNum must be a number 2^k where k is any integer from 2 to 8\n");
        exit(EXIT_FAILURE);
    }

    int PassFailFlag = 1;

    printf("Initializing data...\n");
    h_buffer = (unsigned int *)malloc(VecDim * sizeof(unsigned int));
    h_histogram = (unsigned int *)malloc(BinNum * sizeof(unsigned int));
    cpu_histogram = (unsigned int *)malloc(BinNum * sizeof(unsigned int));

    memset(cpu_histogram, 0, BinNum * sizeof(unsigned int));

    printf("...generating input data\n");
    srand(time(NULL));
    for (int i = 0; i < VecDim; i++) {
        h_buffer[i] = rand() % 1024;
    }

    printf("...allocating GPU memory and copying input data\n");
    cudaMalloc((void **)&d_buffer, VecDim * sizeof(unsigned int));
    cudaMalloc((void **)&d_histogram, BinNum * sizeof(unsigned int));

    cudaMemcpy(d_buffer, h_buffer, VecDim * sizeof(unsigned int), cudaMemcpyHostToDevice);

    cudaMemset(d_histogram, 0, BinNum * sizeof(unsigned int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    printf("Launching kernel...\n");
    int num_blocks = (VecDim + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int numIter = 300;
    for(int i = 0; i < numIter; i++){
        //histogram_privatized_aggregation_kernel<<<num_blocks, BLOCK_SIZE>>>(d_buffer, VecDim, d_histogram, BinNum);
        histogram_privatized_kernel<<<num_blocks, BLOCK_SIZE>>>(d_buffer, VecDim, d_histogram, BinNum);
        if (i < numIter - 1) cudaMemset(d_histogram, 0, BinNum * sizeof(unsigned int));
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float msecPerHistogram = milliseconds / numIter;
    double gflops = (double)(VecDim) / (msecPerHistogram * 1e6);
    printf("Computation time= %.3f ms\n", msecPerHistogram);
    printf("Performance= %.2f GFlops/s \n", gflops);

    //Copy data from GPU to CPU for analysis
    cudaMemcpy(h_histogram, d_histogram, BinNum * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //Calculate CPU Performance
    
    sequential_Histogram(h_buffer, VecDim, cpu_histogram, BinNum);
    
    //Check for correctness
    for (uint i = 0; i < BinNum; i++){
        if (h_histogram[i] != cpu_histogram[i]) {
            PassFailFlag = 0;
        }
    }
    
    //Print Data from CPU and GPU calculations 
    /*
    for (int i = 0; i < BinNum; i++) {
        printf("Bin %d: %u\n", i, h_histogram[i]);
    }

    for (int i = 0; i < BinNum; i++) {
        printf("Bin %d: %u\n", i, cpu_histogram[i]);
    }
    */

    printf(PassFailFlag ? " ...histograms match\n\n"
                        : " ***histograms do not match!!!***\n\n");

    free(h_buffer);
    free(h_histogram);
    free(cpu_histogram);
    cudaFree(d_buffer);
    cudaFree(d_histogram);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}