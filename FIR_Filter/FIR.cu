#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

//Program defines 
#define MAX_SAMPLES 10000
#define MAX_TAPS 2048
#define TILE_SIZE 32

__constant__ float filter_coeffs[MAX_TAPS];

//CPU FIR IMPLEMENTATION
struct FIRFilter{
    float buf[MAX_TAPS];
    uint16_t bufIndex;
    float out;
};

void FIRFilter_Init(FIRFilter *fir){
    for (uint16_t i = 0; i < MAX_TAPS; i++){
        fir->buf[i] = 0.0f;
    }

    fir->out = 0.0f;
}

float FIRFilter_Update(FIRFilter *fir, float inp, float *filter_coeffs){
    // Store the latest sample in the buffer
    fir->buf[fir->bufIndex] = inp;
    //Increment buffer index and wrap around if necessary (circular buffer)
    fir->bufIndex++;

    if (fir->bufIndex == MAX_TAPS){
        fir->bufIndex = 0;
    }

    //Computing new output sample (convolution)
    fir->out = 0.0f;

    uint16_t sumIndex = fir->bufIndex;

    //decrement index and wrap if necessary 
    for(uint16_t i = 0; i < MAX_TAPS; i++){
        if (sumIndex > 0){
            sumIndex--;
        }
        else {
            sumIndex = MAX_TAPS - 1;
        }

        fir->out += filter_coeffs[i] * fir->buf[sumIndex];
    }


    return fir->out;
} 

void convolution1D_CPU(float* h_N, float* h_M, float* h_P, int Kernel_Width, int Width)
{
    for (int i = 0; i < Width; i++) {
        float Pvalue = 0.f;
        int N_start_point = i - (Kernel_Width);
        for (int j = 0; j < Kernel_Width; j++) {
            if (N_start_point + j >= 0 && N_start_point + j < Width)
                Pvalue += h_N[N_start_point + j] * h_M[j];
        }
        h_P[i] = Pvalue;
    }
}


//I/O helper func
void read_signal_from_file(const char *filename, float *signal, int num_samples) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Read signal data
    for (int i = 0; i < num_samples; i++) {
        fscanf(file, "%e", &signal[i]);
    }

    fclose(file);
}

void write_signal_to_file(const char *filename, float *signal, int num_samples) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Write signal data
    for (int i = 0; i < num_samples; i++) {
        fprintf(file, "%f\n", signal[i]);
    }

    fclose(file);
}

//GPU Func
__global__ void FIR_GPU(float *raw_signal, float *filtered_signal, int taps, int samples){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float raw_signal_ds[TILE_SIZE];

    __syncthreads();
    int current_tile_start_point = blockIdx.x * blockDim.x;
    int next_tile_start_point = (blockIdx.x + 1) * blockDim.x;
    int samples_start_point = i - (taps);
    float filtered_signal_value = 0;
    for (int j = 0; j < taps; j++){
        int sample_index = samples_start_point + j;
        if (sample_index >= 0 && sample_index < samples) {
            if((sample_index >= current_tile_start_point) && (sample_index < next_tile_start_point)){
                filtered_signal_value += raw_signal_ds[threadIdx.x + j - (taps)] * filter_coeffs[j];
            }
            else {
                filtered_signal_value += raw_signal[sample_index] * filter_coeffs[j];
            }
        } 
    }
    filtered_signal[i] = filtered_signal_value;
}

__global__ void convolution1D_tiled(float* d_N, float* d_P, int Kernel_Width, int Width)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    __shared__ float d_Nds[TILE_SIZE + MAX_TAPS - 1];
    int n = Kernel_Width / 2;

    if (threadIdx.x >= blockDim.x - n) {
        int halo_index_left = (blockIdx.x - 1)*blockDim.x + threadIdx.x;
        d_Nds[threadIdx.x - (blockDim.x - n)] = (halo_index_left < 0) ? 0 : d_N[halo_index_left];
    }

    d_Nds[n + threadIdx.x] = d_N[i];
    
    if (threadIdx.x < n) {
        int halo_index_right = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
        d_Nds[n + blockDim.x + threadIdx.x] = (halo_index_right >= Width) ? 0 : d_N[halo_index_right];
    }

    __syncthreads();

    float Pvalue = 0;
    for (int j = 0; j < Kernel_Width; j++)
        Pvalue += d_Nds[threadIdx.x + j]*filter_coeffs[j];
    d_P[i] = Pvalue;
}

int main(int argc, char **argv){
    //host memory
    float h_raw_signal[MAX_SAMPLES];
    float h_filtered_GPU_signal[MAX_SAMPLES];
    float h_filter_coeffs[MAX_TAPS];
    //CPU Filter Struct
    FIRFilter filter_data;
    float CPU_filtered_signal[MAX_SAMPLES];
    //device memory 
    float *d_raw_signal;
    float *d_filtered_signal;
    //Define file names
    const char *signal_filename = "signal_data.txt";
    const char *filter_coeffs_filename = "filter_coeffs.txt";
    const char *filtered_signal_GPU_filename = "filtered_signal_GPU.txt";
    const char *filtered_signal_CPU_filename = "filtered_signal_CPU.txt";

    printf("Initializing data...\n");
    // Read data for input into the kernel
    read_signal_from_file(signal_filename, h_raw_signal, MAX_SAMPLES);
    read_signal_from_file(filter_coeffs_filename, h_filter_coeffs, MAX_SAMPLES);

    printf("...allocating GPU memory and copying input data\n");
    cudaMalloc((void **)&d_raw_signal, MAX_SAMPLES * sizeof(float));
    cudaMalloc((void **)&d_filtered_signal, MAX_SAMPLES * sizeof(float));

    cudaMemcpy(d_raw_signal, h_raw_signal, MAX_SAMPLES * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(filter_coeffs, h_filter_coeffs, MAX_TAPS * sizeof(float));

    cudaMemset(d_filtered_signal, 0, MAX_SAMPLES * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    printf("Launching kernel...\n");
    const int threadsPerBlock = TILE_SIZE;
    const int numBlocks = (MAX_SAMPLES + threadsPerBlock - 1) / threadsPerBlock;

    int numIter = 300;
    for (int i = 0; i < numIter; i++){
        FIR_GPU<<<numBlocks, threadsPerBlock>>>(d_raw_signal, d_filtered_signal, MAX_TAPS, MAX_SAMPLES);
    }
    //Measuring time of execution of the kernel
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float msecPerConvolution = milliseconds / numIter;
    printf("Computation time in GPU= %.3f ms\n", msecPerConvolution);

    //Copy data from GPU to CPU for analysis
    cudaMemcpy(h_filtered_GPU_signal, d_filtered_signal, MAX_SAMPLES * sizeof(float), cudaMemcpyDeviceToHost);

    //Declare variables for CPU timing
    clock_t start_cpu, end_cpu;
    double cpu_time_used;
    printf("Carrying out filtering in CPU...\n");
    start_cpu = clock();

    //Carry out filtering for the synthethic data
    convolution1D_CPU(h_raw_signal, h_filter_coeffs, CPU_filtered_signal, MAX_TAPS, MAX_SAMPLES);
    /*
    for(int i = 0; i < MAX_SAMPLES; i++){
        FIRFilter_Update(&filter_data, h_raw_signal[i], h_filter_coeffs);
        CPU_filtered_signal[i] = filter_data.out;
    }
    */
    end_cpu = clock();

    cpu_time_used = ((double) (end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000; //measuring in ms

    // Print CPU execution time
    printf("Computation time in CPU = %.3f ms\n", cpu_time_used);


    write_signal_to_file(filtered_signal_CPU_filename, CPU_filtered_signal, MAX_SAMPLES);
    write_signal_to_file(filtered_signal_GPU_filename, h_filtered_GPU_signal, MAX_SAMPLES);

    printf("Completed Convolution\n");
    cudaFree(d_filtered_signal);
    cudaFree(d_raw_signal);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}