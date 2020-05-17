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

#### GPU Memory Handling
- `cudaMalloc` -  allocates memory in device
- `cudaMemcpy` - copies data from host to device as well as device to host
- `cudaFree` - frees allocated memory in device

#### GPU Kernel: Device code
```
__global__ void mykernel(void){
    // Some computation
}
```
- `__global__` denotes the function to be GPU kernel
- `nvcc` is the compiler to be used
    - Any function with `__global__` infront is compiled by under the hood GPU compiler of `nvcc`
    - Other codes are compiled by host compiler - gcc, microsoft visual c++ etc. (nvcc takes care of that as well)
- GPU kernel needs to be launched by syntax `mykernel<<<x,y>>>()`
    - Any function call having `<<<x,y>>>` is compiled by under the hood GPU compiler` of `nvcc`
    - Other function calls are compiled by host compiler (similarly)
    - `<<<x,y>>>` is called kernel launch configuration
        - denotes how many GPU threads to be launched - `x` blocks each having `y` threads - total `xy` threads
        - `x` and `y` can be of type `int` or `dim3` 
            - `int` for 1D thread blocks and grids
            - `dim3` for 2D and 3D thread blocks and grids

#### Thread hierarchy
- Whole set of threads are organized as a grid
- A grid is made of thread blocks
    - Grid can be 1D, 2D or 3D
- A thread block is made of threads
    - A thread block can be 1D, 2D or 3D
- Threads in the same block can synchronize with each other
- Threads in different block can't synchronize and must execute completely in parallel

