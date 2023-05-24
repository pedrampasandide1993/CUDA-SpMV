#include <omp.h>

template <typename PREC, int onebased, typename INT, typename AT, typename XT, typename YT>
void SpMVcsr(INT m, const AT *values, const INT *rowptr, const INT *colidx, const XT *x, YT *y, int ITER)
{
    // YT is the precision of the outputs
    // PREC is the precision of intermediate results
    // m is number of rows
    // nnz number of non-zero
    // rowptr is row pointers
    // colidx is col indices
    // Acsr->data is values
    // x is x as the y = A*x 
    // y is y in y = A*x 
    // ITER is the number of time algorithm will do the A*x

    // check input parameters
    if (m <= 0 || values == NULL || rowptr == NULL || colidx == NULL || x == NULL || y == NULL)
    {
        fprintf(stderr, "Error: invalid input parameters.\n");
        return;
    }

    double t = omp_get_wtime();
    for (int jj = 1; jj <= ITER; jj++)
    {
        
        // Compute the product y = Ax
        #pragma omp parallel for
        for (INT i = 0; i < m; i++)
        {
            PREC sum = 0;
            for (INT j = rowptr[i] - INT(onebased); j < rowptr[i + 1] - INT(onebased); j++)
            {
                sum += (PREC)values[j] * (PREC)x[colidx[j] - INT(onebased)];
            }
            y[i] = (YT)sum;
        }
    }


    double time_taken = (omp_get_wtime() - t) / ITER;
    double gflop = 2 * (float)(rowptr[m] - rowptr[0]) / 1e9;
    double gbs = ((m * sizeof(YT)) + (rowptr[m] - rowptr[0]) * sizeof(AT) + (rowptr[m] + 1) * sizeof(INT)) / ((double)(ITER)*1e6);
    printf("Average time taken for %s is %f\n", "SpMV by CPU using OpenMP", time_taken);
    printf("Average GFLOP/s is %lf\n", gflop / time_taken);
    printf("Average GB/s is %lf\n\n", gbs);
}
