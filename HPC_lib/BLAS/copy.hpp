#ifndef HPC_BLAS_COPY_HPP
#define HPC_BLAS_COPY_HPP

#include <cstddef>
#include <HPC_lib/utils/conjugate.hpp>

namespace hpc { namespace blas { 


/**
 * @brief Copies elements from one array to another with optional conjugation.
 *
 * This function copies elements from the source array `x` to the destination array `y`.
 * The number of elements to copy is determined by the parameter `n`. Array elements can
 * have different types. 
 *
 * @tparam TX   The type of elements in the source array `x`.
 * @tparam TY   The type of elements in the destination array `y`.
 * @param n     The number of elements to copy.
 * @param conjX Whether to apply conjugation to elements from array `x`.
 * @param x     Pointer to the source array.
 * @param incX  Increment for indexing elements in array `x`.
 * @param y     Pointer to the destination array.
 * @param incY  Increment for indexing elements in array `y`.
 */
template <typename TX, typename TY>
void copy(std::size_t n,
          bool conjX, const TX *x, std::ptrdiff_t incX, 
          TY *y, std::ptrdiff_t incY)
{
    for (std::size_t i=0; i<n; ++i) {
        y[i*incY] = utils::conjugate(x[i*incX], conjX);
    }
}



/**
 * @brief Copies elements from a source matrix to a destination matrix with optional conjugation.
 *
 * This function copies elements from the source matrix `A` to the destination matrix `B`.
 * The dimensions of the matrices are specified by `m` and `n`.
 *
 * @tparam TA      The type of elements in the source matrix `A`.
 * @tparam TB      The type of elements in the destination matrix `B`.
 * @param m        The number of rows in the matrices.
 * @param n        The number of columns in the matrices.
 * @param conjA    Whether to apply conjugation to elements from matrix `A`.
 * @param A        Pointer to the source matrix.
 * @param incRowA  Increment for indexing rows in matrix `A`.
 * @param incColA  Increment for indexing columns in matrix `A`.
 * @param B        Pointer to the destination matrix.
 * @param incRowB  Increment for indexing rows in matrix `B`.
 * @param incColB  Increment for indexing columns in matrix `B`.
 */
template <typename TA, typename TB>
void
gecopy(std::size_t m, size_t n,
       bool conjA, const TA *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
       TB *B, std::ptrdiff_t incRowB, std::ptrdiff_t incColB)
{
    if (m==0 || n==0) {
        return;
    }
    // B is row major:   B^T <- A^T
    if (incRowB>incColB) {
        gecopy(n, m, conjA, A, incColA, incRowA, B, incColB, incRowB);
        return;
    }
    // B is col major:
    for (std::size_t j=0; j<n; ++j) {
        for (std::size_t i=0; i<m; ++i) {
            B[i*incRowB+j*incColB]
                = utils::conjugate(A[i*incRowA+j*incColA], conjA);
        }

    }
}


} } // end namespace hpc, blas

#endif //end HPC_BLAS_COPY_HPP