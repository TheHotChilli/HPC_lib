#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip> // For std::setw
#include <float.h>
#include <array> // for std::array

#include <mkl.h>    

#include <HPC_lib/BLAS/axpy.hpp>
#include <HPC_lib/BLAS/copy.hpp>
#include <HPC_lib/BLAS/gemm/gemm.hpp>
#include <HPC_lib/BLAS/gemm/config.hpp>

#include <HPC_lib/utils/buffer.hpp>
#include <HPC_lib/utils/initMatrix.hpp>
#include <HPC_lib/utils/runtime.hpp>

using namespace hpc;

//------------------------------------------------------------------------------

#ifndef ALPHA
#define ALPHA   1
#endif

#ifndef BETA
#define BETA    1
#endif

#ifndef DIM_MAX_M
#define DIM_MAX_M   2000
#endif

#ifndef DIM_MAX_N
#define DIM_MAX_N   2000
#endif

#ifndef DIM_MAX_K
#define DIM_MAX_K   2000
#endif

#ifndef COLMAJOR_C
#define COLMAJOR_C 1
#endif

#ifndef COLMAJOR_A
#define COLMAJOR_A 1
#endif

#ifndef COLMAJOR_B
#define COLMAJOR_B 1
#endif

// constexpr std::array<blas::config::GemmConfigType, 1> allConfigs = {
//     blas::config::GemmConfigType::Default
// };

// constexpr std::array<blas::config::GemmConfigType, 5> allConfigs = {
//     blas::config::GemmConfigType::Default,
//     blas::config::GemmConfigType::SSE,
//     blas::config::GemmConfigType::AVX,
//     blas::config::GemmConfigType::AVX_BLIS,
//     blas::config::GemmConfigType::AVX_512
// };

//------------------------------------------------------------------------------

double genorm_inf(std::size_t m, std::size_t n,
                  const double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA)
{
    double res = 0;
    for (std::size_t i=0; i<m; ++i) {
        double asum = 0;
        for (std::size_t j=0; j<n; ++j) {
            asum += std::fabs(A[i*incRowA+j*incColA]);
        }
        if (std::isnan(asum)) {
            return asum;
        }
        if (asum>res) {
            res = asum;
        }
    }
    return res;
}

#define MAX(x,y)    ((x)>(y)) ? (x) : (y)

double gemm_err_est(std::size_t m, std::size_t n, std::size_t k,
                    double alpha,
                    const double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
                    const double *B, std::ptrdiff_t incRowB, std::ptrdiff_t incColB,
                    const double *C0, std::ptrdiff_t incRowC0, std::ptrdiff_t incColC0,
                    double beta,
                    const double *C_, std::ptrdiff_t incRowC_, std::ptrdiff_t incColC_,
                    double *C, std::ptrdiff_t incRowC, std::ptrdiff_t incColC)
{
    blas::geaxpy(m, n, 
                 -1, 
                 false, C_, incRowC_, incColC_, 
                 C, incRowC, incColC);

    double normD  = genorm_inf(m, n, C, incRowC, incColC);
    std::size_t N = MAX(m, MAX(n, k));

    if (std::isnan(normD)) {
        return normD;
    }

    if (normD==0) {
        return 0;
    }

    double normA = 0;
    double normB = 0;

    if (alpha!=0) {
        normB  = genorm_inf(k, n, B, incRowB, incColB);
        normA  = genorm_inf(m, k, A, incRowA, incColA);
        normA  *= fabs(alpha);
    }

    double normC0 = 0;
    if (beta!=0) {
        normC0 = genorm_inf(m, n, C0, incRowC0, incColC0);
        normC0 *= fabs(beta);
    }

    return normD/(DBL_EPSILON*(N*normA*normB+normC0));
}

// wrapper for intel MKL GEMM
void dgemm_mkl(int m, int n, int k,
               double alpha,
               const double *A, int incRowA, int incColA,
               const double *B, int incRowB, int incColB,
               double beta,
               double *C, int incRowC, int incColC)
{
    CBLAS_LAYOUT layout = (incRowC == 1) ? CblasColMajor : CblasRowMajor;
    MKL_INT ldC = (incRowC == 1) ? incColC : incRowC;

    MKL_INT ldA = (incRowA == 1) ? incColA : incRowA; // m : k
    MKL_INT ldB = (incRowB == 1) ? incColB : incRowB; // k : n

    CBLAS_TRANSPOSE transA, transB;
    if (layout == CblasColMajor) {
        transA = (incRowA == 1) ? CblasNoTrans : CblasTrans;
        transB = (incRowB == 1) ? CblasNoTrans : CblasTrans;
    } else {
        transA = (incRowA == 1) ? CblasTrans : CblasNoTrans;
        transB = (incRowB == 1) ? CblasTrans : CblasNoTrans;
    }

    // Call Intel MKL's GEMM function directly
    cblas_dgemm(layout, transA, transB,
                m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
    // void cblas_dgemm (const CBLAS_LAYOUT Layout, 
    //                   const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, 
    //                   const MKL_INT m, const MKL_INT n, const MKL_INT k, 
    //                   const double alpha, 
    //                   const double *a, const MKL_INT lda, 
    //                   const double *b, const MKL_INT ldb, 
    //                   const double beta, 
    //                   double *c, const MKL_INT ldc);>
}

void print_header() {
    // load config settings
    blas::config::GemmConfig<double> gemm_config;

    // print config settings
    std::cout << "# Configuration:\n";
    std::cout << "#\tMC = " << std::setw(5) << gemm_config.MC << "\n";
    std::cout << "#\tNC = " << std::setw(5) << gemm_config.NC << "\n";
    std::cout << "#\tKC = " << std::setw(5) << gemm_config.KC << "\n";
    std::cout << "#\tMR = " << std::setw(5) << gemm_config.MR << "\n";
    std::cout << "#\tNR = " << std::setw(5) << gemm_config.NR << "\n";

    // print results table head
    std::cout << "#\n";
    std::cout << "# Benchmark:\n";
    std::cout << "#" << std::setw(8) << "MR" << std::setw(8) << "NR" << std::setw(8) << "k"
              << std::setw(8) << "incRowC" << std::setw(8) << "incColC" << std::setw(8) << "incRowA"
              << std::setw(8) << "incColA" << std::setw(8) << "incRowB" << std::setw(8) << "incColB"
              << std::setw(8) << "alpha" << std::setw(8) << "beta"
              //<< " | "
              << std::setw(12) << "error"
              << std::setw(12) << "tRef[s]" << std::setw(12) << "tTest[s]"
              << std::setw(12) << "mflopsRef" << std::setw(12) << "mflopsTest"
              << std::endl;
}

//------------------------------------------------------------------------------

int main() 
{
    print_header();

    // for measuring runtimes
    double t0, t1;

    // loop different matrix sizes in 100-steps
    for (std::size_t m=300, n=300, k=300;
         m <= DIM_MAX_M && n <= DIM_MAX_N && k <= DIM_MAX_K;
         m += 100, n +=100, k += 100) 
    {
        // number of FLOP using this m,n,k
        double mflop = 2. * m/1000 * n/1000 * k;

        // Matrix storage format
        std::size_t incRowC = (COLMAJOR_C) ? 1 : n;
        std::size_t incColC = (COLMAJOR_C) ? m : 1;

        std::size_t incRowA = (COLMAJOR_A) ? 1 : k;
        std::size_t incColA = (COLMAJOR_A) ? m : 1;

        std::size_t incRowB = (COLMAJOR_B) ? 1 : n;
        std::size_t incColB = (COLMAJOR_B) ? k : 1;

        // print matrix dims
        std::cout << " " 
                  // matrix dims
                  << std::setw(8) << m << std::setw(8) << n << std::setw(8) << k 
                  << std::setw(8) << incRowC << std::setw(8) << incColC << std::setw(8) 
                  << incRowA << std::setw(8) << incColA 
                  << std::setw(8) << incRowB << std::setw(8) << incColB 
                  << std::setw(8) << ALPHA << std::setw(8) << BETA;

        // allocate memory
        double* A      = new double[m*n];
        double* B      = new double[k*n];
        double* C_0    = new double[m*n];
        double* C_ref  = new double[m*n];
        double* C_test = new double[m*n];


        // init matrices
        utils::initMatrix(m, k, A, incRowA, incColA, false);
        utils::initMatrix(k, n, B, incRowA, incColA, false);
        utils::initMatrix(m, n, C_0, incRowC, incColC, false);

        // call reference implementation (intel MKL)                // print reference results
        int     runs = 0;
        double  tRef = 0;
        do {
            blas::gecopy(m, n, 
                        false, C_0, incRowC, incColC, 
                        C_ref, incRowC, incColC);
            t0 = utils::get_walltime();
            dgemm_mkl(
                m, n, k,
                ALPHA,
                A, incRowA, incColA,
                B, incRowB, incColB,
                BETA,
                C_ref, incRowC, incColC
            );
            t1 = utils::get_walltime();
            tRef += t1 - t0;
            ++runs;
        } while (tRef < 1);
        tRef /= runs;

        // call own implementations
        double tTest = 0; 
        runs = 0;
        do {
            blas::gecopy(m, n, 
                        false, C_0, incRowC, incColC, 
                        C_test, incRowC, incColC);
            t0 = utils::get_walltime();
            blas::gemm<double>(
                m, n, k,
                ALPHA,
                false, A, incRowA, incColA,
                false, B, incRowB, incColB,
                BETA,
                C_test, incRowC, incColC
            );
            t1 = utils::get_walltime();
            tTest += t1 - t0;
            ++runs;
        } while (tTest < 1);
        tTest = tTest / runs;
        double error = gemm_err_est(m, n, k,
                                    ALPHA,
                                    A, incRowA, incColA, 
                                    B, incRowB, incColB,
                                    C_0, incRowC, incColC, 
                                    BETA,
                                    C_ref, incRowC, incColC,
                                    C_test, incRowC, incColC);
        // print results
        // std::cout << " | " s
        std::cout << std::scientific << std::setw(12) << std::setprecision(2) << error;
        std::cout << std::fixed 
                  << std::setw(12) << std::setprecision(2) << tRef
                  << std::setw(12) << std::setprecision(2) << tTest;
        std::cout << std::fixed 
                  << std::setw(12) << std::setprecision(2) << mflop / tRef
                  << std::setw(12) << std::setprecision(2) << mflop / tTest
                  << std::endl;

        // deallocate memory
        delete[] A;
        delete[] B;
        delete[] C_0;
        delete[] C_ref;
        delete[] C_test;

    }
}
