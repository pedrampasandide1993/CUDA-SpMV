#include <cuda.h>


template <typename T, int THREADS_PER_VECTOR, int MAX_NUM_VECTORS_PER_BLOCK>
__global__ void spmv_light_kernel(int *cudaRowCounter, int *d_ptr, int *d_cols, T *d_val, T *d_vector, T *d_out, int N)
{
    int i;
    T sum;
    int row;
    int rowStart, rowEnd;
    int laneId = threadIdx.x % THREADS_PER_VECTOR;      // lane index in the vector
    int vectorId = threadIdx.x / THREADS_PER_VECTOR;    // vector index in the thread block
    int warpLaneId = threadIdx.x & 31;                  // lane index in the warp
    int warpVectorId = warpLaneId / THREADS_PER_VECTOR; // vector index in the warp

    __shared__ volatile int space[MAX_NUM_VECTORS_PER_BLOCK][2];

    // Get the row index
    if (warpLaneId == 0)
    {
        row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
    }
    // Broadcast the value to other threads in the same warp and compute the row index of each vector
    row = __shfl_sync(0xffffffff, row, 0) + warpVectorId;

    while (row < N)
    {

        // Use two threads to fetch the row offset
        if (laneId < 2)
        {
            space[vectorId][laneId] = d_ptr[row + laneId];
        }
        rowStart = space[vectorId][0];
        rowEnd = space[vectorId][1];

        sum = 0;
        // Compute dot product
        if (THREADS_PER_VECTOR == 32)
        {

            // Ensure aligned memory access
            i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

            // Process the unaligned part
            if (i >= rowStart && i < rowEnd)
            {
                sum += d_val[i] * d_vector[d_cols[i]];
            }

            // Process the aligned part
            for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR)
            {
                sum += d_val[i] * d_vector[d_cols[i]];
            }
        }
        else
        {
            for (i = rowStart + laneId; i < rowEnd; i +=
                                                    THREADS_PER_VECTOR)
            {
                sum += d_val[i] * d_vector[d_cols[i]];
            }
        }
        // Intra-vector reduction
        for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1)
        {
            sum += __shfl_down_sync(0xffffffff, sum, i);
        }

        // Save the results
        if (laneId == 0)
        {
            d_out[row] = sum;
        }

        // Get a new row index
        if (warpLaneId == 0)
        {
            row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
        }
        // Broadcast the row index to the other threads in the same warp and compute the row index of each vector
        row = __shfl_sync(0xffffffff, row, 0) + warpVectorId;
    }
}

template <typename T>
void spmv_light(int m, int n, T *values, int *rowptr, int *colidx, T *vector, T *out, int nnz, int ITER)
{
    // T is the precision of the input and outputs
    // m is number of rows
    // n is number of cols
    // nnz number of non-zero
    // rowptr is row pointers
    // colidx is col indices
    // Acsr->data is values
    // vector is x as the y = A*x
    // out is y in y = A*x
    // ITER is the number of time algorithm will do the A*x

    T *d_vector, *d_val, *d_out;
    int *d_cols, *d_ptr;
    float time_taken;
    double gflop = 2 * (double)nnz / 1e9;
    float milliseconds = 0;
    int meanElementsPerRow = nnz / m;
    int *cudaRowCounter;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory on device
    cudaMalloc(&d_vector, n * sizeof(T));
    cudaMalloc(&d_val, nnz * sizeof(T));
    cudaMalloc(&d_out, m * sizeof(T));
    cudaMalloc(&d_cols, nnz * sizeof(int));
    cudaMalloc(&d_ptr, (m + 1) * sizeof(int));
    cudaMalloc(&cudaRowCounter, sizeof(int));

    // Copy from host memory to device memory
    cudaMemcpy(d_vector, vector, n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, values, nnz * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, colidx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ptr, rowptr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, m * sizeof(T));
    cudaMemset(cudaRowCounter, 0, sizeof(int));

    // Choose the vector size depending on the NNZ/Row, run the kernel and time it
    cudaEventRecord(start);
    if (meanElementsPerRow <= 2)
    {
        for (int i = 0; i < ITER; i++)
        {
            spmv_light_kernel<T, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<ceil(m / (float)BlockDim), BlockDim>>>(
                cudaRowCounter, d_ptr, d_cols, d_val, d_vector, d_out, m);
            cudaMemset(cudaRowCounter, 0, sizeof(int));
        }
    }
    else if (meanElementsPerRow <= 4)
    {
        for (int i = 0; i < ITER; i++)
        {
            spmv_light_kernel<T, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<ceil(m / (float)BlockDim), BlockDim>>>(
                cudaRowCounter, d_ptr, d_cols, d_val, d_vector, d_out, m);
            cudaMemset(cudaRowCounter, 0, sizeof(int));
        }
    }
    else if (meanElementsPerRow <= 64)
    {
        for (int i = 0; i < ITER; i++)
        {
            spmv_light_kernel<T, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<ceil(m / (float)BlockDim), BlockDim>>>(
                cudaRowCounter, d_ptr, d_cols, d_val, d_vector, d_out, m);
            cudaMemset(cudaRowCounter, 0, sizeof(int));
        }
    }
    else
    {
        for (int i = 0; i < ITER; i++)
        {
            spmv_light_kernel<T, 32, MAX_NUM_THREADS_PER_BLOCK / 32><<<ceil(m / (float)BlockDim), BlockDim>>>(
                cudaRowCounter, d_ptr, d_cols, d_val, d_vector, d_out, m);
            cudaMemset(cudaRowCounter, 0, sizeof(int));
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy from device memory to host memory
    cudaMemcpy(out, d_out, m * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_vector);
    cudaFree(d_val);
    cudaFree(d_cols);
    cudaFree(d_ptr);
    cudaFree(d_out);

    // Calculate and print out GFLOPs and GB/s
    double gbs = ((n * sizeof(T)) + (nnz * sizeof(T)) + (m * sizeof(int)) + (nnz * sizeof(int)) + (m * sizeof(T))) / (milliseconds / ITER) / 1e6;
    time_taken = (milliseconds / ITER) / 1000.0;
    printf("Average time taken for %s is %f\n", "SpMV by GPU CSR LightSpMV Algorithm", time_taken);
    printf("Average GFLOP/s is %lf\n", gflop / time_taken);
    printf("Average GB/s is %lf\n\n", gbs);
}
