#ifndef HPC_BLAS_ASUMDIFFMATRIX_HPP
#define HPC_BLAS_ASUMDIFFMATRIX_HPP

#include <cstddef>      // for size_t, ptrdiff_t
#include <cmath>        // for fabs
#include <type_traits>  // for std::common_type

namespace hpc { namespace blas { 

//find common return type
template<typename TX, typename TY>
using TRET = typename std::common_type<TX, TY>::type;

template<typename TX, typename TY>
TRET<TX, TY> asumDiffMatrix(
    std::size_t m, std::size_t n,
    const TX *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
    const TY *B, std::ptrdiff_t incRowB, std::ptrdiff_t incColB)
{
    TRET<TX, TY> asum = 0;

    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < m; ++i) {
            asum += std::fabs(static_cast<TRET<TX, TY>>(B[i * incRowB + j * incColB]) -
                              static_cast<TRET<TX, TY>>(A[i * incRowA + j * incColA]));
        }
    }

    return asum;
}


} } // end namespace hpc, blas

#endif // end HPC_BLAS_ASUMDIFFMATRIX_HPP