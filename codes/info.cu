// nvcc -o info info.cu
// ./info
#include <stdio.h>
#include <cuda_runtime.h>

int main()
{
    int device_id = 0;
    cudaSetDevice(device_id);

    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id);

    printf("Compute capability of device %d: %d.%d\n", device_id, major, minor);

    int shared_mem_per_block;
    cudaDeviceGetAttribute(&shared_mem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device_id);
    printf("Shared memory per block of device %d: %d bytes\n", device_id, shared_mem_per_block);

    int regs_per_block;
    cudaDeviceGetAttribute(&regs_per_block, cudaDevAttrMaxRegistersPerBlock, device_id);
    printf("Registers per block of device %d: %d\n", device_id, regs_per_block);

    int warp_size;
    cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, device_id);
    printf("Warp size of device %d: %d threads\n", device_id, warp_size);

    int max_threads_per_block;
    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id);
    printf("Max threads per block of device %d: %d threads\n", device_id, max_threads_per_block);

    int max_block_dim_x, max_block_dim_y, max_block_dim_z;
    cudaDeviceGetAttribute(&max_block_dim_x, cudaDevAttrMaxBlockDimX, device_id);
    cudaDeviceGetAttribute(&max_block_dim_y, cudaDevAttrMaxBlockDimY, device_id);
    cudaDeviceGetAttribute(&max_block_dim_z, cudaDevAttrMaxBlockDimZ, device_id);
    printf("Max thread dimensions of device %d: %d x %d x %d\n", device_id, max_block_dim_x, max_block_dim_y, max_block_dim_z);

    int total_constant_mem;
    cudaDeviceGetAttribute(&total_constant_mem, cudaDevAttrTotalConstantMemory, device_id);
    printf("Total constant memory of device %d: %d bytes\n", device_id, total_constant_mem);

    int mp_count;
    cudaDeviceGetAttribute(&mp_count, cudaDevAttrMultiProcessorCount, device_id);
    printf("Multiprocessor count of device %d: %d\n", device_id, mp_count);

    int memory_bus_width;
    cudaDeviceGetAttribute(&memory_bus_width, cudaDevAttrGlobalMemoryBusWidth, device_id);
    printf("Memory bus width of device %d: %d bits\n", device_id, memory_bus_width);

    int l2_cache_size;
    cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, device_id);
    printf("L2 cache size of device %d: %d bytes\n", device_id, l2_cache_size);

    int max_threads_per_sm;
    cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device_id);
    printf("Max threads per SM of device %d: %d\n", device_id, max_threads_per_sm);

    return 0;
}