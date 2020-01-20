#include <stdio.h>

int N = 4096;

__global__ void vadd(float* A, float* B, float* C){
    C[blockIdx.x] = A[blockIdx.x] + B[blockIdx.x];
}

int main(){
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    h_A = new float[N];
    h_B = new float[N];
    h_C = new float[N];

    cudaMalloc((void **)&d_A, N*sizeof(float));
    cudaMalloc((void **)&d_B, N*sizeof(float));
    cudaMalloc((void **)&d_C, N*sizeof(float));

    for(int i = 0; i < N; i++){
        h_A[i] = i; h_B[i] = i;
    }
    cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*sizeof(float), cudaMemcpyHostToDevice);

    vadd<<<N,1>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);

    bool flag = true;
    for(int i = 0; i < N; i++){
        if(h_C[i] != h_A[i]+h_B[i]){
            flag = false;
            break;
        }
    }

    if(flag) printf("Correct\n");
    else printf("Incorrect\n");
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    return 0;
}
  
