/**
 * @file axpy.hpp
 * @brief Definitions of AXPY operations (BLAS) for scaling and adding vectors and matrices.
 */

#ifndef HPC_BLAS_AXPY_HPP
#define HPC_BLAS_AXPY_HPP

#include <cstddef>
#include <HPC_lib/utils/conjugate.hpp>

namespace hpc { namespace blas { 


/**
 * @brief Scales a vector and adds the result to another vector.
 *
 * This function scales the elements of the input vector `x` by the scalar `alpha`,
 * conjugates them if specified, and then adds the scaled vector to the elements of
 * the vector `y`. The operation is performed in-place, meaning the input vector `y`
 * is modified directly. The operation is defined as:
 * \f[ 
 *      y \leftarrow \alpha x
 * ]\f
 *
 * @tparam TALPHA Data type of the scalar `alpha`.
 * @tparam TX Data type of the elements in the vector `x`.
 * @tparam TY Data type of the elements in the vector `y`.
 * @param n Number of elements in the vectors `x` and `y`.
 * @param alpha Scalar value to scale the elements of the vector `x`.
 * @param x Pointer to the data of the input vector `x`.
 * @param incX Increment between consecutive elements in the vector `x`.
 * @param conjX Boolean indicating whether to conjugate the elements of the vector `x`.
 * @param y Pointer to the data of the input/output vector `y`.
 * @param incY Increment between consecutive elements in the vector `y`.
 */
template <typename TALPHA, typename TX, typename TY>
void axpy(std::size_t n, const TALPHA &alpha, 
          const TX *x, std::ptrdiff_t incX, bool conjX,
          TY *y, std::ptrdiff_t incY)
{
    if (alpha==TALPHA(0)) {
        return;
    }
    if (alpha!=TALPHA(1)) {
        for (std::size_t i=0; i<n; ++i) {
            y[i*incY] += TY(alpha) * TY(utils::conjugate(x[i*incX], conjX));
        }
    }
    else {
        for (std::size_t i=0; i<n; ++i) {
            y[i*incY] += TY(utils::conjugate(x[i*incX], conjX));
        }
    }
}

/**
 * @brief Scales a matrix and adds the result to another marix.
 *
 * This function scales the elements of the input matrix `A` by the scalar `alpha`,
 * conjugates them if specified, and then adds the scaled matrix to the elements
 * of the matrix `B`. The operation is performed in-place, meaning the input matrix `B`
 * is modified directly. The operation is defined as:
 * \f[ 
 *      Y \leftarrow \alpha X
 * ]\f
 *
 * @tparam TALPHA Data type of the scalar `alpha`.
 * @tparam TA Data type of the elements in the matrix `A`.
 * @tparam TB Data type of the elements in the matrix `B`.
 * @param m Number of rows in the matrices `A` and `B`.
 * @param n Number of columns in the matrices `A` and `B`.
 * @param alpha Scalar value to scale the elements of the matrix `A`.
 * @param conjA Boolean indicating whether to conjugate the elements of the matrix `A`.
 * @param A Pointer to the data of the input matrix `A`.
 * @param incRowA Increment between consecutive rows in the matrix `A`.
 * @param incColA Increment between consecutive columns in the matrix `A`.
 * @param B Pointer to the data of the input/output matrix `B`.
 * @param incRowB Increment between consecutive rows in the matrix `B`.
 * @param incColB Increment between consecutive columns in the matrix `B`.
 */
template <typename TALPHA, typename TA, typename TB>
void geaxpy(std::size_t m, size_t n, const TALPHA &alpha,
            bool conjA, const TA *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
            TB *B, std::ptrdiff_t incRowB, std::ptrdiff_t incColB)
{
    if (m==0 || n==0 || alpha==TALPHA(0)) {
        return;
    }
    // row major:   B^T <- alpha*A^T + B^T
    if (incRowB>incColB) {
        geaxpy(n, m, alpha, conjA, A, incColA, incRowA, B, incColB, incRowB);
        return;
    }
    // col major
    for (std::size_t j=0; j<n; ++j) {
        for (std::size_t i=0; i<m; ++i) {
            B[i*incRowB+j*incColB]
                += TB(alpha)*TB(utils::conjugate(A[i*incRowA+j*incColA],conjA));
        }
    }
}


} } // end namespace blas, hpc

#endif //end HPC_BLAS_AXPY_HPP