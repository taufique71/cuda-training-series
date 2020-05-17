#include <stdio.h>
#include <stdlib.h>

// these are just for timing measurments
#include <time.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


const int DSIZE = 1024;
const int block_size = 16;  // CUDA maximum is 1024 *total* threads in block
const float A_val = 1.0f;
const float B_val = 2.0f;

// matrix multiply (naive) kernel: C = A * B
__global__ void mmul(const float *A, const float *B, float *C, int ds) {

  int idx = threadIdx.x+blockDim.x*blockIdx.x; // create thread x index
  int idy = threadIdx.y+blockDim.y*blockIdx.y; // create thread y index

  if ((idx < ds) && (idy < ds)){
    float temp = 0;
    for (int i = 0; i < ds; i++)
      temp += A[idy*ds+i] * B[i*ds+idx];   // dot product of row and column
    C[idy*ds+idx] = temp;
  }
}

int main(){
  
  printf("Matrix dimension: [%d x %d]\n", DSIZE, DSIZE);

  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
  float *ground_truth_C;

  // these are just for timing
  clock_t t0, t1, t2, t3;
  double t1sum=0.0;
  double t2sum=0.0;
  double t3sum=0.0;

  // start timing
  t0 = clock();

  h_A = new float[DSIZE*DSIZE];
  h_B = new float[DSIZE*DSIZE];
  h_C = new float[DSIZE*DSIZE];
  ground_truth_C = new float[DSIZE*DSIZE];
  for (int i = 0; i < DSIZE*DSIZE; i++){
    h_A[i] = (float) (rand() % 10); // random value between 0 and 9 inclusive
    h_B[i] = (float) (rand() % 10); // random value between 0 and 9 inclusive
    h_C[i] = 0;
    ground_truth_C[i] = 0;
  }
  /*for (int i = 0; i < DSIZE*DSIZE; i++){*/
    /*h_A[i] = A_val;*/
    /*h_B[i] = B_val;*/
    /*h_C[i] = 0;*/
    /*ground_truth_C[i] = 0;*/
  /*}*/

  // Initialization timing
  t1 = clock();
  t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
  printf("Init took %f seconds.  Begin compute\n", t1sum);

  // Allocate device memory and copy input data over to GPU
  cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));
  cudaMalloc(&d_B, DSIZE*DSIZE*sizeof(float));
  cudaMalloc(&d_C, DSIZE*DSIZE*sizeof(float));
  cudaCheckErrors("cudaMalloc failure");
  cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  // Cuda processing sequence step 1 is complete

  // Launch kernel
  dim3 block(block_size, block_size);  // dim3 variable holds 3 dimensions
  dim3 grid((DSIZE+block.x-1)/block.x, (DSIZE+block.y-1)/block.y);
  mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
  cudaCheckErrors("kernel launch failure");

  // Cuda processing sequence step 2 is complete

  // Copy results back to host
  cudaMemcpy(h_C, d_C, DSIZE*DSIZE*sizeof(float), cudaMemcpyDeviceToHost);

  // Cuda processing sequence step 3 is complete

  // Verify results
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

  // GPU timing
  t2 = clock();
  t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
  printf ("Done. Compute in GPU took %f seconds\n", t2sum);

  // Calculate traditional sequential matrix multiplication
  for(int i = 0; i < DSIZE; i++){
    for(int j = 0; j < DSIZE; j++){
      for(int k = 0; k < DSIZE; k++){
        int idx_A = i * DSIZE + k;
        int idx_B = k * DSIZE + j;
        int idx_C = i * DSIZE + j;
        ground_truth_C[idx_C] += h_A[idx_A] * h_B[idx_B];
      }
    }
  }

  // Sequential CPU timing
  t3 = clock();
  t3sum = ((double)(t3-t2))/CLOCKS_PER_SEC;
  printf ("Done. Sequential compute in CPU took %f seconds\n", t3sum);

  // Compare two results
  /*for (int i = 0; i < DSIZE*DSIZE; i++) if (h_C[i] != A_val*B_val*DSIZE) {printf("mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], A_val*B_val*DSIZE); return -1;}*/
  for (int i = 0; i < DSIZE*DSIZE; i++) {
    if (h_C[i] != ground_truth_C[i]) {
      printf("mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], ground_truth_C[i]); 
      return -1;
    }
  }
  printf("Success!\n"); 

  return 0;
}
  
