#include <stdio.h>

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


const int DSIZE = 8192;
const int block_size = 32;  // CUDA maximum is 1024 *total* threads in block
const float A_val = 3.0f;
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

// matrix multiply (naive) kernel: C = A * B
// using shared memory
__global__ void mmul_sm(const float *A, const float *B, float *C, int ds) {

  // declare cache in shared memory
  __shared__ float As[block_size][block_size];
  __shared__ float Bs[block_size][block_size];

  int idx = threadIdx.x+blockDim.x*blockIdx.x; // create thread x index
  int idy = threadIdx.y+blockDim.y*blockIdx.y; // create thread y index

  if ((idx < ds) && (idy < ds)){
    float temp = 0;
    for (int i = 0; i < ds/block_size; i++) {

      // Load data into shared memory
      As[threadIdx.y][threadIdx.x] = A[idy * ds + (i * block_size + threadIdx.x)];
      Bs[threadIdx.y][threadIdx.x] = B[(i * block_size + threadIdx.y) * ds + idx];

      // Synchronize
      __syncthreads();

      // Keep track of the running sum
      for (int k = 0; k < block_size; k++)
      	temp += As[threadIdx.y][k] * Bs[k][threadIdx.x]; // dot product of row and column
      __syncthreads();

    }

    // Write to global memory
    C[idy*ds+idx] = temp;
  }
}

int main(){

  float *h_A, *h_B, *h_C, *h_C_sm, *d_A, *d_B, *d_C;

  dim3 block(block_size, block_size);  // dim3 variable holds 3 dimensions
  dim3 grid((DSIZE+block.x-1)/block.x, (DSIZE+block.y-1)/block.y);

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
  h_C_sm = new float[DSIZE*DSIZE];
  for (int i = 0; i < DSIZE*DSIZE; i++){
    h_A[i] = (float) (rand() % 10); // random value between 0 and 9 inclusive
    h_B[i] = (float) (rand() % 10); // random value between 0 and 9 inclusive
    h_C[i] = 0;
    h_C_sm[i] = 0;
  }

  // Initialization timing
  t1 = clock();
  t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
  printf("Init took %f seconds.  Begin compute\n", t1sum);


  // Start non shared memory multiplication //

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
  mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
  cudaCheckErrors("kernel launch failure");

  // Cuda processing sequence step 2 is complete

  // Copy results back to host
  cudaMemcpy(h_C, d_C, DSIZE*DSIZE*sizeof(float), cudaMemcpyDeviceToHost);

  // GPU timing
  t2 = clock();
  t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
  printf ("Done without shared memory. Compute took %f seconds\n", t2sum);

  // Cuda processing sequence step 3 is complete
  // Free all memory in allocated in GPU
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
    
  // Start shared memory multiplication //

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
  mmul_sm<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
  cudaCheckErrors("kernel launch failure");

  // Cuda processing sequence step 2 is complete

  // Copy results back to host
  cudaMemcpy(h_C_sm, d_C, DSIZE*DSIZE*sizeof(float), cudaMemcpyDeviceToHost);

  // Cuda processing sequence step 3 is complete
  // Free all memory in allocated in GPU
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // GPU timing
  t3 = clock();
  t3sum = ((double)(t3-t2))/CLOCKS_PER_SEC;
  printf ("Done with shared memory. Compute took %f seconds\n", t3sum);

  // Verify results
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  for (int i = 0; i < DSIZE*DSIZE; i++) if (h_C[i] != h_C_sm[i]) {printf("mismatch at index %d, was: %f, should be: %f\n", i, h_C_sm[i], h_C[i]); return -1;}
  printf("Success!\n"); 
  return 0;
}
  
