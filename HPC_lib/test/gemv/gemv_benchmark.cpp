#include <cstddef>
#include <iostream>
#include <iomanip>
#include <mkl.h>
#include <HPC_lib/BLAS/init.hpp>
#include <HPC_lib/BLAS/asumDiffMatrix.hpp>
#include <HPC_lib/BLAS/copy.hpp>
#include <HPC_lib/utils/runtime.hpp>
#include <HPC_lib/BLAS/gemv.hpp>

//------------------------------------------------------------------------------
// benchmark parameters

#ifndef MIN_M
#define MIN_M 100
#endif

#ifndef MIN_N
#define MIN_N 100
#endif

#ifndef MAX_M
#define MAX_M 1500
#endif

#ifndef MAX_N
#define MAX_N 1500
#endif

#ifndef INCX
#define INCX 1
#endif

#ifndef INCY
#define INCY 1
#endif

#ifndef ALPHA
#define ALPHA 1.5
#endif

#ifndef BETA
#define BETA 1.5
#endif

#ifndef T_MIN
#define T_MIN 5
#endif

#ifndef COLMAJOR
//#define COLMAJOR 1
#define COLMAJOR 0
#endif

//------------------------------------------------------------------------------


double A[MAX_M*MAX_N];
double X[MAX_N*INCX];
double Y[MAX_M*INCY];
double Y1[MAX_M*INCY];
double Y2[MAX_M*INCY];
double Y3[MAX_M*INCY];
double Y4[MAX_M*INCY];
double Y5[MAX_M*INCY];

// wrapper for calling Intel MKL gemv method
void dgemv_mkl(MKL_INT m, MKL_INT n,
               double alpha,
               const double *A, MKL_INT incRowA, MKL_INT incColA,
               const double *x, MKL_INT incX,
               double beta,
               double *y, MKL_INT incY)
{
    MKL_INT ldA = (incRowA == 1) ? incColA : incRowA;   // leading dimension of A
    //char trans = (incRowA == 1) ? 'N' : 'T';            // transpose?
    CBLAS_TRANSPOSE trans = (incRowA == 1) ? CblasNoTrans : CblasTrans;
    MKL_INT M = (incRowA == 1) ? m : n;
    MKL_INT N = (incRowA == 1) ? n : m;

    cblas_dgemv(CblasColMajor, trans, M, N, alpha, A, ldA, x, incX, beta, y, incY);
}

// main benchmark function
int main() {
    std::size_t runs, incRowA, incColA;
    double t0, t1, t2;
    double diff2;
    double alpha = ALPHA;
    double beta = BETA;

    hpc::blas::initMatrix(MAX_M, MAX_N, A, 1, MAX_M, false);
    hpc::blas::initMatrix(MAX_N, 1, X, INCX, 1, false);
    hpc::blas::initMatrix(MAX_M, 1, Y, INCY, 1, false);

    std::cout << "# COLMAJOR    = " << COLMAJOR << "\n";
    std::cout << "# T_MIN       = " << T_MIN << "\n";
    std::cout << "#RUN    M     N  INCROW  INCCOL";
    std::cout << "    GEMV_MKL    GEMV_OWN";
    std::cout << "    GEMV_MKL    GEMV_OWN";
    std::cout << "       DIFF2";
    std::cout << "\n";
    std::cout << "#                              ";
    std::cout << "    (t in s)    (t in s)";
    std::cout << "    (MFLOPS)    (MFLOPS)";
    std::cout << "           ";
    std::cout << "\n";

    // Increase m, n +100 in each run
    for (size_t i=0, m=MIN_M, n=MIN_N; m<=MAX_M && n<=MAX_N;
     ++i, m+=100, n+=100) {

        if (COLMAJOR) {
            incRowA = 1;
            incColA = m;
        } else {
            incRowA = n;
            incColA = 1;
        }

        // run intel MKL for T_MIN time 
        t1   = 0;
        runs = 0;
        do {
            hpc::blas::copy(MAX_M, false, Y, INCY, Y1, INCY);
            t0 = hpc::utils::get_walltime();
            dgemv_mkl(
                    m, n, 
                    alpha,
                    A, incRowA, incColA,
                    X, INCX,
                    beta,
                    Y1, INCY
            );
            t1 += hpc::utils::get_walltime() - t0;
            ++runs;
        } while (t1<T_MIN);
        t1 /= runs;

        // run own implementation for T_MIN time
        t2 = 0;
        runs = 0;
        do {
            hpc::blas::copy(MAX_M, false, Y, INCY, Y2, INCY);
            t0 = hpc::utils::get_walltime();
            hpc::blas::gemv_fused(
                    m, n,
                    alpha,
                    false, A, incRowA, incColA,
                    false, X, INCX,
                    beta,
                    false, Y2, INCY
            );
            t2 += hpc::utils::get_walltime() - t0;
            ++runs;
        } while (t2 < T_MIN);
        t2 /= runs;
        diff2 = hpc::blas::asumDiffMatrix(m, 1, Y1, INCY, 1, Y2, INCY, 1);

        // print results to table
        std::cout << std::setw(3) << i << std::setw(5) << m << std::setw(5) << n << std::setw(7) << incRowA << std::setw(7) << incColA << " ";
        std::cout << std::setw(11) << std::fixed << std::setprecision(4) << t1 << " ";
        std::cout << std::setw(11) << std::fixed << std::setprecision(4) << t2 << " ";
        std::cout << std::setw(11) << std::fixed << std::setprecision(4) << 2 * (m / 1000.0) * (n / 1000.0) / t1 << " ";
        std::cout << std::setw(11) << std::fixed << std::setprecision(4) << 2 * (m / 1000.0) * (n / 1000.0) / t2 << " ";
        std::cout << std::setw(11) << std::fixed << std::setprecision(4) << diff2 << " ";
        std::cout << "\n";
     } 
}