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
- Two canonical type of problems
    - Transformation: input and output is of same size
        - example: vector add, stencil operation
        - common strategy is to spawn one thread per output/input element
    - Reduction: output size is smaller than input size
        - example: vector dot product


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
    - Size/dimension of thread block is accessible with variable named `blockDims`
    - Index of the thread block inside the grid is accessible with variable named `blockIdx`
    - Index of the thread inside a thread block is accessible with variable named `threadIdx`
    - Combination of `blockDims.x`, `blockDims.y` and `blockDims.z` refers to the dimension/size of the block
    - Combination of `blockIdx.x`, `blockIdx.y` and `blockIdx.z` refers to the index of the block within the grid
    - Combination of `threadIdx.x`, `threadIdx.y` and `threadIdx.z` refers to the index of the thread within a block
- Threads in the same block can synchronize with each other
- Threads in different block can't synchronize and must execute completely in parallel

#### Motivating problem
- Vector addition
    - Spawn one thread for each element of the vector
    - From a thread add correspoding elements and write to the corresponding location
    - Get index of corresponding element: `idx = blockIdx.x * blockDims.x + threadIdx.x`
    - Each thread operating on corresponding element: `w[idx] = u[idx] + v[idx]`



# CUDA Shared Memory
#### Shared memory vs global memory
- Global memory
    - GPU DRAM memory, not on-chip
    - Memory allocated with `cudaMalloc` is global memory
- Shared memory
    - Implemented on GPU die itself - on-chip memory
    - Declared with `__shared__`, allocated per thread block
    - Per block resource, threads in one blocks can't access the changes made by threads in other blocks
- Because of being on-chip memory shared memory is faster than global memory
    - Bandwidth of shared memory is more than bandwidth of global memory
    - Latency if shared memory is lower than latency of global memory
    - roughly order of 5x difference
#### Data race in using shared memory
- As threads in a block can read and write to shared memory data race is possible
- `__syncthreads()` is used to put a barrier
    - All threads in the same block have to reach the point before any thread is allowed to proceed
#### Motivating problem
- 1D stencil
    - input: array of n numbers
    - output: array of n numbers, each number calculated from some numbers around it (radius)
    - some numbers are needed from multiple threads to calculate its corresponding position
        - solution 1: each thread reads all the values it needs from global memory
            - performance problem: same data is being read from memory multiple times
        - solution 2: read all data items that are needed by the threads inside the block only once from global memory to shared memory
        
# Fundamental CUDA Optimization
## Expose a lot of parallelism
- Launch configuration (lots of threads for latency hiding)
## Make efficient use of memory subsystem
- Global memory throughput (try to achieve coalesced memory access)
- Shared memory access (allocate shared mem in such a way so that each thread in the warp access from different bank)

# Atomics, Reductions, Warp Shuffle

# Managed Memory

# CUDA Concurrency

