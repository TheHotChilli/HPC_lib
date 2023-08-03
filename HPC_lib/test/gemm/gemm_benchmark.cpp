#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip> // For std::setw
#include <float.h>

#include <HPC_lib/BLAS/axpy.hpp>
#include <HPC_lib/BLAS/gemm/gemm.hpp>
#include <HPC_lib/BLAS/gemm/config.hpp>


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
    geaxpy(m, n, -1, C_, incRowC_, incColC_, C, incRowC, incColC);

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

int main() 
{
    std::cout << "#Configuration:\n";
    std::cout << "#\tMC = " << std::setw(5) << ulmblas::dgemm_parameter::MC << "\n";
    std::cout << "#\tNC = " << std::setw(5) << ulmblas::dgemm_parameter::NC << "\n";
    std::cout << "#\tKC = " << std::setw(5) << ulmblas::dgemm_parameter::KC << "\n";
    std::cout << "#\tMR = " << std::setw(5) << ulmblas::dugemm_parameter::MR << "\n";
    std::cout << "#\tNR = " << std::setw(5) << ulmblas::dugemm_parameter::NR << "\n";

    std::cout << "#\n";
    std::cout << "#Benchmark:\n";
    std::cout << std::left << std::setw(7) << "MR" << std::setw(7) << "NR" << std::setw(7) << "k"
              << std::setw(7) << "incRowC" << std::setw(7) << "incColC" << std::setw(7) << "incRowA"
              << std::setw(7) << "incColA" << std::setw(7) << "incRowB" << std::setw(7) << "incColB"
              << std::setw(7) << "alpha" << std::setw(7) << "beta" << std::setw(12) << "error"
              << std::setw(7) << "tRef" << std::setw(7) << "tTst" << std::setw(12) << "mflops: ref"
              << "tst" << "\n";
}
