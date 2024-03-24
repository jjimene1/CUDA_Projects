#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#define O_TILE_WIDTH 24
//for constant memory
#define MAX_MASK_WIDTH 10

__constant__ float M[MAX_MASK_WIDTH * MAX_MASK_WIDTH];

typedef struct {
    int width;
    int height;
    int pitch;
    int channels;
    float* data;
} * wbImage_t;

//CPU 2D convolution
void convolution2D_CPU(float* N, float *P, float *M, int height, int width, int pitch, int channels, int mask_width){
    for (int ch = 0; ch < channels; ch++) {
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int row_start_point = row - mask_width / 2;
                int col_start_point = col - mask_width / 2;
                float val = 0.f;

                for (int i = 0; i < mask_width; i++) {
                    for (int j = 0; j < mask_width; j++) {
                        int row_idx = row_start_point + i;
                        int col_idx = col_start_point + j;

                        if (row_idx >= 0 && row_idx < height && col_idx >= 0 && col_idx < width) {
                            val += N[ch*width*height + row_idx * pitch + col_idx] * M[i * mask_width + j];
                        }
                    }
                }

                P[ch*width*height + row*width + col] = val;
            }
        }
    }
}

__global__ void convolution2D(float *N, float *P, int height, int width, int pitch, int channels, int Mask_Width){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int ch = blockDim.z * blockIdx.z + tz;
    
    int row_o = blockIdx.y * O_TILE_WIDTH + ty;
    int col_o = blockIdx.x * O_TILE_WIDTH + tx;

    int row_i = row_o - Mask_Width/2;
    int col_i = col_o - Mask_Width/2;

    __shared__ float N_ds[O_TILE_WIDTH + MAX_MASK_WIDTH - 1][O_TILE_WIDTH + MAX_MASK_WIDTH - 1];

    if((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width)){
        N_ds[ty][tx] = N[ch*width*height + row_i * pitch + col_i];
    }
    else{
        N_ds[ty][tx] = 0.0f;
    }
    __syncthreads();

    float output = 0.0f;
    if(ty < O_TILE_WIDTH && tx < O_TILE_WIDTH && channels){
        for (int i = 0; i < Mask_Width; i++){
            for (int j = 0; j < Mask_Width; j++){
                output += M[i * Mask_Width + j] * N_ds[i + ty][j + tx];
            }
        }

        if(row_o < height && col_o < width){
            P[ch*width*height + row_o * width + col_o] = output;
        }
    }
}


int main(int argc, char **argv){
    wbImage_t h_image, h_output;
    float *h_mask;
    float *d_image;
    float *d_output;

    unsigned int dimX;
    unsigned int dimY;
    unsigned int dimK;

    //get x dimension of input array
    if (checkCmdLineFlag(argc, (const char **)argv, "dimX")) {
        dimX = getCmdLineArgumentInt(argc, (const char **)argv, "dimX");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "dimY")) {
        dimY = getCmdLineArgumentInt(argc, (const char **)argv, "dimY");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "dimK")) {
        dimK = getCmdLineArgumentInt(argc, (const char **)argv, "dimK");
    }

    //Initialize and set data
    printf("Initializing data...\n");
    h_image = (wbImage_t)malloc(sizeof(*h_image));
    h_output = (wbImage_t)malloc(sizeof(*h_output));
    h_mask = (float*)malloc(dimK * dimK * sizeof(float));

    h_image->width = dimX;
    h_image->height = dimY;
    h_image->pitch = dimX;
    h_image->channels = 3;
    h_output->width = dimX;
    h_output->height = dimY;
    h_output->pitch = dimX;
    h_output->channels = 3;

    h_image->data = (float*)malloc(h_image->width * h_image->height * h_image->channels * sizeof(float));
    h_output->data = (float*)malloc(h_image->width * h_image->height * h_image->channels * sizeof(float));

    printf("...generating input mask and image");
    srand(time(NULL));
    for (int c = 0; c < h_image->channels; c++)
        for (int i = 0; i < h_image->height; i++)
            for (int j = 0; j < h_image->width; j++)
                h_image->data[c*h_image->height * h_image->width + i * h_image->width + j] = (rand() / (float)RAND_MAX) * 15.0f;
    for (int i = 0; i < dimK; i++)
        for (int j = 0; j < dimK; j++)
            h_mask[i*dimK + j] = (rand() / (float)RAND_MAX) * 15.0f;

    printf("...allocating GPU memory and copying input data\n");
    cudaMalloc((void **)&d_image, dimX * dimY * h_image->channels * sizeof(float));
    cudaMalloc((void **)&d_output, dimX * dimY * h_image->channels * sizeof(float));

    cudaMemcpy(d_image, h_image->data, dimX * dimY * h_image->channels * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(M, h_mask, dimK * dimK * sizeof(float));

    cudaMemset(d_output, 0 , dimX * dimY * h_image->channels * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    printf("Launching kernel...\n");
    const int I_TILE_WIDTH = O_TILE_WIDTH + dimK - 1; 
    dim3 dimBlock(I_TILE_WIDTH, I_TILE_WIDTH, 1);
    dim3 dimGrid((h_image->width  + O_TILE_WIDTH - 1) / O_TILE_WIDTH, (h_image->height + O_TILE_WIDTH - 1) / O_TILE_WIDTH, h_image->channels);

    int numIter = 300;
    for(int i = 0; i < numIter; i++){
        convolution2D<<<dimGrid, dimBlock>>>(d_image, d_output, h_image->height, h_image->width, h_image->pitch, h_image->channels, dimK);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float msecPerConvolution = milliseconds / numIter;
    double numOps = 2.0 * h_image->height * h_image->width * dimK * dimK * h_image->channels;
    double gflops = numOps / (msecPerConvolution * 1e6);
    printf("Computation time in GPU= %.3f ms\n", msecPerConvolution);
    printf("Performance on GPU= %.2f GFlops/s \n", gflops);

    //Copy data from GPU to CPU for analysis
    cudaMemcpy(h_output->data, d_output, dimX * dimY * h_image->channels * sizeof(float), cudaMemcpyDeviceToHost);

    //Calculate CPU Data and compare for correctness
    float* CPU_out = (float*)malloc(dimX * dimY * h_image->channels * sizeof(float));
     // Declare variables for CPU timing
    clock_t start_cpu, end_cpu;
    double cpu_time_used;
    printf("Calculating in CPU...\n");
    start_cpu = clock();
    convolution2D_CPU(h_image->data, CPU_out, h_mask, h_image->height, h_image->width, h_image->pitch, h_image->channels, dimK);
    end_cpu = clock();

    cpu_time_used = ((double) (end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000;

    // Print CPU execution time
    printf("Computation time in CPU = %.3f ms\n", cpu_time_used);
    double cpu_gflops = numOps / (cpu_time_used * 1e6);
    printf("Performance on CPU = %.2f GFlops/s \n", cpu_gflops);
    
    int precision = 8;
    double threshold = 1e-8 * h_image->channels * h_image->width * h_image->height;
    double diff = 0.0;
    for (int i = 0; i < h_image->channels * h_image->width * h_image->height; i++) {
        diff += fabs((double)CPU_out[i] - (double)h_output->data[i]);
    }
    diff /= (double)h_image->channels * h_image->width * h_image->height;
    printf("Error : %.*f (threshold: %f)\n", precision, (double)diff, threshold);

    //Print data from convolution calculation
    /*
    printf("GPU output:\n");
    for (int i = 0; i < dimX * dimY * h_image->channels; i++){
        printf("%f ", h_output->data[i]);
    }
    printf("\n");
    printf("CPU output:\n");
    for (int i = 0; i < dimX * dimY * h_image->channels; i++){
        printf("%f ", CPU_out[i]);
    }
    printf("\n");
    */

    printf("Completed Convolution\n");
    cudaFree(h_image->data);
    cudaFree(h_output->data);
    cudaFree(d_image);
    cudaFree(d_output);
    free(h_mask);
    free(h_image);
    free(h_output);
    free(CPU_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}