# Fundamental CUDA Optimization (Part 2) HW Report

## 1. Matrix Row/Column Sums
In this case each thread should be computing each element of the row sum or column sum output
- Row sum or column sum output array should be  allocated in GPU in following way - `cudaMalloc(&d_sums, DSIZE*sizeof(float))`
- Thread id can be found in following way - `int idx = blockIdx.x * blockDim.x + threadIdx.x`
- For row sum each thread should do the following in each of its iteration - `sum += A[ds * idx + i]`
- For column sum each thread should do the following in each of its iteration - `sum += A[ds * i + idx]`

## 2. Profiling

### 2a. Basic Nsight Compute Profiling
Output of `nv-nsight-cu-cli ./matrix_sums` -
```
==PROF== Connected to process 73376 (/global/u1/t/taufique/Codes/study-notes/CUDA_Training/exercises/hw4/matrix_sums)
==PROF== Profiling "row_sums" - 1: 0%....50%....100% - 17 passes
==PROF== Profiling "column_sums" - 2: 0%....50%....100%row sums correct!
column sums correct!
 - 17 passes
==PROF== Disconnected from process 73376
[73376] matrix_sums@127.0.0.1
  row_sums(float const *,float*,unsigned long), 2020-Jul-28 06:48:36, Context 1, Stream 7
    Section: GPU Speed Of Light
    ---------------------------------------------------------------------- --------------- ------------------------------
    Memory Frequency                                                         cycle/usecond                         874.41
    SOL FB                                                                               %                          36.25
    Elapsed Cycles                                                                   cycle                        4341767
    SM Frequency                                                             cycle/nsecond                           1.31
    Memory [%]                                                                           %                          77.32
    Duration                                                                       msecond                           3.32
    SOL L2                                                                               %                          12.66
    SM Active Cycles                                                                 cycle                     3430442.94
    SM [%]                                                                               %                           2.42
    SOL TEX                                                                              %                          97.85
    ---------------------------------------------------------------------- --------------- ------------------------------

    Section: Launch Statistics
    ---------------------------------------------------------------------- --------------- ------------------------------
    Block Size                                                                                                        256
    Grid Size                                                                                                          64
    Registers Per Thread                                                   register/thread                             31
    Shared Memory Configuration Size                                                  byte                              0
    Dynamic Shared Memory Per Block                                             byte/block                              0
    Static Shared Memory Per Block                                              byte/block                              0
    Threads                                                                         thread                          16384
    Waves Per SM                                                                                                     0.10
    ---------------------------------------------------------------------- --------------- ------------------------------

    Section: Occupancy
    ---------------------------------------------------------------------- --------------- ------------------------------
    Block Limit SM                                                                   block                             32
    Block Limit Registers                                                            block                              8
    Block Limit Shared Mem                                                           block                             32
    Block Limit Warps                                                                block                              8
    Achieved Active Warps Per SM                                                      warp                           8.00
    Achieved Occupancy                                                                   %                          12.50
    Theoretical Active Warps per SM                                             warp/cycle                             64
    Theoretical Occupancy                                                                %                            100
    ---------------------------------------------------------------------- --------------- ------------------------------

  column_sums(float const *,float*,unsigned long), 2020-Jul-28 06:48:36, Context 1, Stream 7
    Section: GPU Speed Of Light
    ---------------------------------------------------------------------- --------------- ------------------------------
    Memory Frequency                                                         cycle/usecond                         838.25
    SOL FB                                                                               %                          88.20
    Elapsed Cycles                                                                   cycle                        1784040
    SM Frequency                                                             cycle/nsecond                           1.25
    Memory [%]                                                                           %                          88.20
    Duration                                                                       msecond                           1.42
    SOL L2                                                                               %                          30.80
    SM Active Cycles                                                                 cycle                     1375717.71
    SM [%]                                                                               %                           6.34
    SOL TEX                                                                              %                          30.49
    ---------------------------------------------------------------------- --------------- ------------------------------

    Section: Launch Statistics
    ---------------------------------------------------------------------- --------------- ------------------------------
    Block Size                                                                                                        256
    Grid Size                                                                                                          64
    Registers Per Thread                                                   register/thread                             32
    Shared Memory Configuration Size                                                  byte                              0
    Dynamic Shared Memory Per Block                                             byte/block                              0
    Static Shared Memory Per Block                                              byte/block                              0
    Threads                                                                         thread                          16384
    Waves Per SM                                                                                                     0.10
    ---------------------------------------------------------------------- --------------- ------------------------------

    Section: Occupancy
    ---------------------------------------------------------------------- --------------- ------------------------------
    Block Limit SM                                                                   block                             32
    Block Limit Registers                                                            block                              8
    Block Limit Shared Mem                                                           block                             32
    Block Limit Warps                                                                block                              8
    Achieved Active Warps Per SM                                                      warp                           7.98
    Achieved Occupancy                                                                   %                          12.47
    Theoretical Active Warps per SM                                             warp/cycle                             64
    Theoretical Occupancy                                                                %                            100
    ---------------------------------------------------------------------- --------------- ------------------------------
```
Row sum kernel is taking 3.32ms while column sum kernel is taking 1.42ms.
As column sum memory accesses are more coalesced than the row sum accesses, this variation of kernel time probably is coming from difference in global memory
throughput due to caching load from global memory.

### 2b. Measure the global memory load efficiency
Output of `nv-nsight-cu-cli --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./matrix_sums` -
```
==PROF== Connected to process 64237 (/global/u1/t/taufique/Codes/study-notes/CUDA_Training/exercises/hw4/matrix_sums)
==PROF== Profiling "row_sums" - 1: 0%....50%....100% - 4 passes
==PROF== Profiling "column_sums" - 2: 0%....50%....100%row sums correct!
column sums correct!
 - 4 passes
==PROF== Disconnected from process 64237
[64237] matrix_sums@127.0.0.1
  row_sums(float const *,float*,unsigned long), 2020-Jul-28 06:42:00, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                                request                        8388608
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                                  sector                      268435451
    ---------------------------------------------------------------------- --------------- ------------------------------

  column_sums(float const *,float*,unsigned long), 2020-Jul-28 06:42:01, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                                request                        8388608
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                                  sector                       33554432
    ---------------------------------------------------------------------- --------------- ------------------------------
```
Transactions per request for row sum `268435451/8388608 = 31.99`, transactions per request for column sum `33554432/8388608 = 4`.
Which means row sum kernel is accessing more global memory sectors per request than the column sum kernel.
This probably is a numerical proof why row sum kernel is taking more time than column sum kernel.
