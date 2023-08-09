/**
 * @file asumDiffMatrix.hpp
 * @brief Definition of the hpc::blas::asumDiffMatrix function for calculating the absolute sum of element-wise differences between matrices.
 *
 * This file contains the definition of the hpc::blas::asumDiffMatrix function, which calculates the absolute sum of differences
 * between corresponding elements of two matrices A and B. The function takes into account specified increments for row and column traversals.
 */

#ifndef HPC_BLAS_ASUMDIFFMATRIX_HPP
#define HPC_BLAS_ASUMDIFFMATRIX_HPP

#include <cstddef>      // for size_t, ptrdiff_t
#include <cmath>        // for fabs
#include <type_traits>  // for std::common_type

namespace hpc { namespace blas { 


//find common return type
template<typename TX, typename TY>
using TRET = typename std::common_type<TX, TY>::type;

/**
 * @brief Calculates the absolute sum of element-wise differences between two matrices.
 *
 * Given two matrices A and B, this function calculates the absolute sum of the differences
 * between corresponding elements of A and B, considering specified increments for row and column traversals.
 *
 * @tparam TX Type of elements in matrix A.
 * @tparam TY Type of elements in matrix B.
 * @param m Number of rows in the matrices.
 * @param n Number of columns in the matrices.
 * @param A Pointer to the first matrix (A).
 * @param incRowA Increment value for moving to the next row in matrix A.
 * @param incColA Increment value for moving to the next column in matrix A.
 * @param B Pointer to the second matrix (B).
 * @param incRowB Increment value for moving to the next row in matrix B.
 * @param incColB Increment value for moving to the next column in matrix B.
 * @return The absolute sum of element-wise differences between matrices A and B.
 */
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