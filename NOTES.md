# Introduction to CUDA C++
- What is CUDA?
    - Compute Unified Device Architecture
    - Refers to one kind of GPU architecture that is programmable
    - Introduced by NVIDIA
    - Also refers to programming environment to write programs for that kind of GPU architecture
- GPU accelerated systems means heterogeneous architecture - multiple processors of different types in single computer architecture
    - Different processors handle different aspects of the workload
    - GPU and its memory is called "Device" - designed to work best for compute intensive workload
    - CPU and its memory is called "Host" - designed to work best for other sequential of workload i.e., user i/o, disk i/o, network activity
    - GPU is not a standalone entity like CPU, it always is next to a CPU
    - These two processing systems are connected with high performance bus
    - Normally programs are written keeping CPU systems in mind, then when the program comes to some compute intensive portion, that portion is offloaded to the GPU system
- Offloading workload from host to device consists of three steps
    - Copy data from host memory to device memory
    - Do computation on device on device memory and store output on device memory
    - Copy data from device memory to host memory
- Typically all of these steps are done by calling some functions from code running in host system
    - The function that is called with from host system to perform the second step is called GPU kernel


#### GPU Kernel: Device code
```
__global__ void mykernel(void){
    // Some computation
}
```
- Code needs to be compiled with `nvcc` compiler
    - Any function with `__global__` infront is compiled by under the hood GPU compiler of `nvcc`
    - Other codes are compiled by host compiler - gcc, microsoft visual c++ etc. (nvcc takes care of that)
- GPU kernel needs to be launched by syntax `mykernel<<<1,1>>>()`
    - Any function call having `<<<x,y>>>` is compiled by under the hood GPU compiler` of `nvcc`
    - Other function calls are compiled by host compiler (again taken care by `nvcc`)

#### GPU Memory Handling
- `cudaMalloc`
- `cudaMemcpy`
- `cudaFree`
