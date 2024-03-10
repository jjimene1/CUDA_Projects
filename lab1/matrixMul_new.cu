// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#define TILE_WIDTH 32

__global__ void MatrixMulKernel(float* M, float* N, float* P, uint j, uint k, uint l){
    
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    //Calculate the row index of the P element and M
    int Row = by * TILE_WIDTH + ty;
    // Calculate the column index of P and N
    int Col = bx * TILE_WIDTH + tx;
   
   float Pvalue = 0;
   //loop over the tiles of M and N required to compute the output matrix
   for (int m = 0; m < ceil(k/(float)TILE_WIDTH); ++m){
    //Loading of of M and N tiles into shared memory
    if ((Row < j) && (m*TILE_WIDTH+tx)<= k)
      Mds[ty][tx] = M[Row*k + m*TILE_WIDTH + tx];
    else 
      Mds[ty][tx] = 0.0; //0 for out of bounds elements
    if ((m*TILE_WIDTH+ty)<k && Col<l)
      Nds[ty][tx] = N[(m*TILE_WIDTH + ty) * l + Col];
    else 
      Nds[ty][tx] = 0.0;
    __syncthreads();

    if ((Row<j) && (Col<l)){
      for (int c = 0; c < TILE_WIDTH; ++c){
        Pvalue += Mds[ty][c] * Nds[c][tx];
      }
    }
    __syncthreads();
   }
   if ((Row<j) && (Col<l)) P[Row*l + Col] = Pvalue;
}

void ConstantInit(float *data, int size, float val) {
  //function for adding a constant to an array
  for (int i = 0; i < size; ++i) {
    data[i] = val;
  }
}

int MatrixMultiply(int argc, char **argv,
                   const dim3 &dimsA,
                   const dim3 &dimsB) {
  // Allocate host memory for matrices A and B
  unsigned int size_A = dimsA.x * dimsA.y;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float *h_A;
  checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
  unsigned int size_B = dimsB.x * dimsB.y;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float *h_B;
  checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
  cudaStream_t stream;

  // Initialize host memory
  const float valB = 0.01f;
  ConstantInit(h_A, size_A, 1.0f);
  ConstantInit(h_B, size_B, valB);

  // Allocate device memory
  float *d_A, *d_B, *d_C;

  // Allocate host matrix C
  dim3 dimsC(dimsB.x, dimsA.y, 1);
  unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
  float *h_C;
  checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));

  if (h_C == NULL) {
    fprintf(stderr, "Failed to allocate host matrix C!\n");
    exit(EXIT_FAILURE);
  }

  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));
  // Allocate CUDA events that we'll use for timing
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // copy host memory to device
  checkCudaErrors(
      cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(
      cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

  // Setup execution parameters
  int NumBlocksX = (dimsB.x + TILE_WIDTH - 1) / TILE_WIDTH; // Number of blocks in x-dimension
  int NumBlocksY = (dimsA.y + TILE_WIDTH - 1) / TILE_WIDTH; // Number of blocks in y-dimension
  dim3 grid(NumBlocksX, NumBlocksY);
  dim3 threads(TILE_WIDTH, TILE_WIDTH);

  //Rows of A
  uint j = dimsA.y; 
  //Rows of B/Columns of A
  uint k = dimsB.y; 
  //Columns of B
  uint l = dimsB.x; 

  // Create and start timer
  printf("Computing result using CUDA Kernel...\n");

  checkCudaErrors(cudaStreamSynchronize(stream));

  // Record the start event
  checkCudaErrors(cudaEventRecord(start, stream));

  // Execute the kernel
  int nIter = 300;

  for (int f = 0; f < nIter; f++) {
      MatrixMulKernel
          <<<grid, threads, 0, stream>>>(d_A, d_B, d_C, j, k, l);
  }

  // Record the stop event
  checkCudaErrors(cudaEventRecord(stop, stream));

  // Wait for the stop event to complete
  checkCudaErrors(cudaEventSynchronize(stop));

  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  // Compute and print the performance
  float msecPerMatrixMul = msecTotal / nIter;
  double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
                             static_cast<double>(dimsA.y) *
                             static_cast<double>(dimsB.x);
  double gigaFlops =
      (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
  printf(
      "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
      " WorkgroupSize= %u threads/block\n",
      gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);

  // Copy result from device to host
  checkCudaErrors(
      cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));

  printf("Checking computed result for correctness: \n");
  bool correct = true;

  // test relative error by the formula
  //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
  double eps = 1.e-6;  // machine zero

  for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
    double abs_err = fabs(h_C[i] - (dimsA.x * valB));
    double dot_length = dimsA.x;
    double abs_val = fabs(h_C[i]);
    double rel_err = abs_err / abs_val / dot_length;

    if (rel_err > eps) {
      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
             i, h_C[i], dimsA.x * valB, eps);
      correct = false;
    }
  }
  
  //printed result of matrix
  /*
  for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
    if (i % TILE_WIDTH == 0){
        printf("%f \n", h_C[i]);
    }
    else{
        printf("%f ", h_C[i]);
    }
  }
  */
  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
  //
  // Clean up memory
  checkCudaErrors(cudaFreeHost(h_A));
  checkCudaErrors(cudaFreeHost(h_B));
  checkCudaErrors(cudaFreeHost(h_C));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  printf(
      "\nNOTE: The CUDA Samples are not meant for performance "
      "measurements. Results may vary when GPU Boost is enabled.\n");

  if (correct) {
    return EXIT_SUCCESS;
  } else {
    return EXIT_FAILURE;
  }
}

int main(int argc, char **argv){
  printf("[Matrix Multiply Using CUDA] - Starting...\n");

  if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
      checkCmdLineFlag(argc, (const char **)argv, "?")) {
    printf("Usage -device=n (n >= 0 for deviceID)\n");
    printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
    printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
    printf("  Note: Outer matrix dimensions of A & B matrices" \
           " must be equal.\n");

    exit(EXIT_SUCCESS);
  }

  int dev = findCudaDevice(argc, (const char **)argv);

  //Initialize input matrices
  dim3 dimsA(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimsB(TILE_WIDTH, TILE_WIDTH, 1);
  
  // width of Matrix A
  if (checkCmdLineFlag(argc, (const char **)argv, "wA")) {
    dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
  }

  // height of Matrix A
  if (checkCmdLineFlag(argc, (const char **)argv, "hA")) {
    dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
  }

 // width of Matrix B
  if (checkCmdLineFlag(argc, (const char **)argv, "wB")) {
    dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
  }

  // Since outer dimensions must be equal
  dimsB.y = dimsA.x;

  if (dimsA.x != dimsB.y) {
    printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
           dimsA.x, dimsB.y);
    exit(EXIT_FAILURE);
  }

  printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.y, dimsA.x,
         dimsB.y, dimsB.x);

  checkCudaErrors(cudaProfilerStart());
  int matrix_result = MatrixMultiply(argc, argv, dimsA, dimsB);
  printf("The value of the multiplication is %d", matrix_result);
  checkCudaErrors(cudaProfilerStop());

  exit(matrix_result);

}