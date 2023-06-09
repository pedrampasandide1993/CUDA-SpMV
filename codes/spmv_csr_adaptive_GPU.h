#include <cuda.h>

template <typename T>
__global__ void spmv_csr_adaptive_kernel(T *d_val, T *d_vector, int *d_cols, int *d_ptr, int N, int *d_rowBlocks, T *d_out)
{
    int startRow = d_rowBlocks[blockIdx.x];
    int nextStartRow = d_rowBlocks[blockIdx.x + 1];
    int num_rows = nextStartRow - startRow;
    int i = threadIdx.x;
    __shared__ volatile T LDS[BlockDim];
    // If the block consists of more than one row then run CSR Stream
    if (num_rows > 1)
    {
        int nnz = d_ptr[nextStartRow] - d_ptr[startRow];
        int first_col = d_ptr[startRow];

        // Each thread writes to shared memory
        if (i < nnz)
        {
            LDS[i] = d_val[first_col + i] * d_vector[d_cols[first_col + i]];
        }
        __syncthreads();

        // Threads that fall within a range sum up the partial results
        for (int k = startRow + i; k < nextStartRow; k += blockDim.x)
        {
            T temp = 0;
            for (int j = (d_ptr[k] - first_col); j < (d_ptr[k + 1] - first_col); j++)
            {
                temp = temp + LDS[j];
            }
            d_out[k] = temp;
        }
    }
    // If the block consists of only one row then run CSR Vector
    else
    {
        // Thread ID in warp
        int rowStart = d_ptr[startRow];
        int rowEnd = d_ptr[nextStartRow];

        T sum = 0;

        // Use all threads in a warp to accumulate multiplied elements
        for (int j = rowStart + i; j < rowEnd; j += BlockDim)
        {
            int col = d_cols[j];
            sum += d_val[j] * d_vector[col];
        }

        LDS[i] = sum;
        __syncthreads();

        // Reduce partial sums
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
        {
            __syncthreads();
            if (i < stride)
                LDS[i] += LDS[i + stride];
        }
        // Write result
        if (i == 0)
            d_out[startRow] = LDS[i];
    }
}

int spmv_csr_adaptive_rowblocks(int *ptr, int totalRows, int *rowBlocks)
{
    rowBlocks[0] = 0;
    int sum = 0;
    int last_i = 0;
    int ctr = 1;
    for (int i = 1; i < totalRows; i++)
    {
        // Count non-zeroes in this row
        sum += ptr[i] - ptr[i - 1];
        if (sum == BlockDim)
        {
            // This row fills up LOCAL_SIZE
            last_i = i;
            rowBlocks[ctr++] = i;
            sum = 0;
        }
        else if (sum > BlockDim)
        {
            if (i - last_i > 1)
            {
                // This extra row will not fit
                rowBlocks[ctr++] = i - 1;
                i--;
            }
            else if (i - last_i == 1)
                // This one row is too large
                rowBlocks[ctr++] = i;
            last_i = i;
            sum = 0;
        }
    }
    rowBlocks[ctr++] = totalRows;
    return ctr;
}

template <typename T>
void spmv_csr_adaptive(int m, int n, T *values, int *rowptr, int *colidx, T *vector, T *out, int nnz, int ITER)
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
    int *d_cols, *d_ptr, *rowBlocks, *d_rowBlocks;
    float time_taken;
    double gflop = 2 * (double)nnz / 1e9;
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    rowBlocks = (int *)malloc(m * sizeof(int));

    // Allocate memory on device
    cudaMalloc(&d_vector, n * sizeof(T));
    cudaMalloc(&d_val, nnz * sizeof(T));
    cudaMalloc(&d_out, m * sizeof(T));
    cudaMalloc(&d_cols, nnz * sizeof(int));
    cudaMalloc(&d_ptr, (m + 1) * sizeof(int));

    // Copy from host memory to device memory
    cudaMemcpy(d_vector, vector, n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, values, nnz * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, colidx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ptr, rowptr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, m * sizeof(T));

    int countRowBlocks = spmv_csr_adaptive_rowblocks(rowptr, m, rowBlocks);
    cudaMalloc(&d_rowBlocks, countRowBlocks * sizeof(int));
    cudaMemcpy(d_rowBlocks, rowBlocks, countRowBlocks * sizeof(int), cudaMemcpyHostToDevice);

    // Run the kernel and time it
    cudaEventRecord(start);
    for (int i = 0; i < ITER; i++)
        spmv_csr_adaptive_kernel<T><<<(countRowBlocks - 1), BlockDim>>>(d_val, d_vector, d_cols, d_ptr, m, d_rowBlocks, d_out);
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
    cudaFree(d_rowBlocks);
    free(rowBlocks);

    // Calculate and print out GFLOPs and GB/s
    double gbs = ((n * sizeof(T)) + (nnz * sizeof(T)) + (m * sizeof(int)) + (nnz * sizeof(int)) + (m * sizeof(T)) + countRowBlocks * sizeof(int)) / (milliseconds / ITER) / 1e6;
    time_taken = (milliseconds / ITER) / 1000.0;
    printf("Average time taken for %s is %f\n", "SpMV by GPU CSR Adaptive Algorithm", time_taken);
    printf("Average GFLOP/s is %lf\n", gflop / time_taken);
    printf("Average GB/s is %lf\n\n", gbs);
}
