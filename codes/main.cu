// nvcc -arch=sm_61 -o main main.cu -lgsl -lgslcblas -Xcompiler -fopenmp
// OMP_NUM_THREADS=8 ./main
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <iostream>

#define BlockDim 512
#define MAX_NUM_THREADS_PER_BLOCK 1024
#include "spmv_light_GPU.h"
#include "spmv_csr_adaptive_GPU.h"
#include "read_csr.h"
#include "spmv.h"

template <typename T>
int num_diff_elems(int n, const T *a, const T *b, double *max)
{
    int diff = 0;
    *max = 0;
    for (int i = 0; i < n; i++)
    {
        T tmp = fabs(a[i] - b[i]) / fabs(a[i]);
        if (tmp > *max)
            *max = double(tmp);
        if (tmp > 1e-4)
        {
            // printf("%e  %e \n", a[i], b[i]);
            diff++;
        }
    }
    return diff;
}

template <typename T>
void init_xy(int n, T *x, T *y)
{
    for (int i = 0; i < n; i++)
        x[i] = 1;
    for (int i = 0; i < n; i++)
        y[i] = 1;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "\n Usage: %s  file.mtx num_repetitions\n", argv[0]);
        return 1;
    }

    int device_id = 0;
    cudaDeviceProp props;

    // Get the device properties
    cudaGetDeviceProperties(&props, device_id);

    // Get the maximum threads per block supported by the device
    int ped;
    cudaDeviceGetAttribute(&ped, cudaDevAttrMaxThreadsPerBlock, device_id);

    printf("Device Specifications:\n");
    printf("-------------------------------------------------------\n");

    printf("Maximum threads per block supported by device %d: %d\n", device_id, ped);
    printf("Current threads per block : %d\n\n", BlockDim);

    char *fname = argv[1];
    int repeats = atoi(argv[2]);

    int num_procs;
    char *var = std::getenv("OMP_NUM_THREADS");
    if (var != NULL)
        sscanf(var, "%d", &num_procs);
    else
    {
        printf("Set environment OMP_NUM_THREADS\n");
        return 2;
    }

    // ############################# reading sparse CSR matrix #############################
    // n is number of cols
    // nnz number of non-zero
    // ia is row pointers
    // ja is col indices
    // Acsr->data values

    gsl_spmatrix *Acsr = ReadMMtoCSR(fname);

    int m = Acsr->size1;
    int n = Acsr->size2;
    int nnz = Acsr->nz;

    double avg_nnz_per_row = (double)nnz / m; // average number of non-zeros per row

    printf("Matrix Specifications:\n");
    printf("-------------------------------------------------------\n");
    printf("Dimensions: %d x %d\n", m, n);
    printf("Number of non-zeros: %d\n", nnz);
    printf("Average number of non-zeros per row: %.2f\n\n", avg_nnz_per_row);

    // copy row pointers and column indices to integer vectors in case GSL reads
    // them into long int arrays.
    int *ia = new int[n + 1], *ja = new int[nnz];
    for (int i = 0; i < n + 1; i++)
        ia[i] = Acsr->p[i];
    for (int i = 0; i < nnz; i++)
        ja[i] = Acsr->i[i];

    double max;

    // #################################### double ############################################
    printf("Double precision results:\n");
    printf("-------------------------------------------------------\n");

    // using GPU for adaptive CSR
    double *xg_d = new double[2 * n], *yg_d = xg_d + n;
    init_xy(n, xg_d, yg_d);

    spmv_csr_adaptive<double>(m, n, Acsr->data, ia, ja, xg_d, yg_d, nnz, repeats);

    // using GPU for light CSR
    double *xl_d = new double[2 * n], *yl_d = xl_d + n;
    init_xy(n, xl_d, yl_d);

    spmv_light<double>(m, n, Acsr->data, ia, ja, xl_d, yl_d, nnz, repeats);

    // using CPU for CSR format
    double *xd_d = new double[2 * n], *yd_d = xd_d + n;
    init_xy(n, xd_d, yd_d);
    SpMVcsr<double, 0, int, double, double, double>(m, Acsr->data, ia, ja, xd_d, yd_d, repeats);

    // Check if results are OK.
    int diff_elems1 = num_diff_elems<double>(n, yd_d, yg_d, &max);
    if (diff_elems1)
    {
        // fprintf(stderr, "\n--- %d different elements in double precision result for adaptive CSR", diff_elems1);
        fprintf(stderr, "\n---largest diff in double precision for adaptive CSR %e\n", max);
    }

    int diff_elems2 = num_diff_elems<double>(n, yd_d, yl_d, &max);
    if (diff_elems2)
    {
        // fprintf(stderr, "\n--- %d different elements in double precision result for lightSpMV", diff_elems2);
        fprintf(stderr, "\n---largest diff in double precision for lightSpMV %e\n", max);
    }

    // #################################### float ############################################
    printf("\nSingle precision results:\n");
    printf("-------------------------------------------------------\n");
    // using GPU for adaptive CSR
    float *xg_f = new float[2 * n], *yg_f = xg_f + n, *Af = new float[nnz];
    init_xy(n, xg_f, yg_f);
    for (int i = 0; i < nnz; i++)
        Af[i] = Acsr->data[i];

    spmv_csr_adaptive<float>(m, n, Af, ia, ja, xg_f, yg_f, nnz, repeats);

    // using GPU for light CSR
    float *xl_f = new float[2 * n], *yl_f = xl_f + n;
    init_xy(n, xl_f, yl_f);

    spmv_light<float>(m, n, Af, ia, ja, xl_f, yl_f, nnz, repeats);

    // using CPU for CSR format
    float *xd_f = new float[2 * n], *yd_f = xd_f + n;
    init_xy(n, xd_f, yd_f);
    SpMVcsr<float, 0, int, double, float, float>(m, Acsr->data, ia, ja, xd_f, yd_f, repeats);

    // Check if results are OK.
    int diff_elems_f1 = num_diff_elems<float>(n, yd_f, yg_f, &max);
    if (diff_elems_f1)
    {
        // fprintf(stderr, "\n--- %d different elements in single precision result for adaptive CSR", diff_elems_f1);
        fprintf(stderr, "\n---largest diff in single precision for adaptive CSR %e\n", max);
    }

    int diff_elems_f2 = num_diff_elems<float>(n, yd_f, yl_f, &max);
    if (diff_elems_f2)
    {
        // fprintf(stderr, "\n--- %d different elements in single precision result for lightSpMV ", diff_elems_f2);
        fprintf(stderr, "\n---largest diff in single precision for lightSpMV %e\n", max);
    }
    // #######################################################################################

    free(Acsr->p);
    free(Acsr->i);
    free(Acsr->data);
    free(Acsr);

    return 0;
}
