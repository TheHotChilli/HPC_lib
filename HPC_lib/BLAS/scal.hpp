#ifndef HPC_BLAS_SCAL_HPP
#define HPC_BLAS_SCAL_HPP

#include <cstddef>

namespace hpc { namespace blas { 

/**
 * @brief Scale the elements of a vector by a scalar value.
 *
 * This function multiplies each element of the input vector `x` by the scalar `alpha`.
 * The operation is performed in-place, meaning the input vector `x` is modified directly.
 * This operation is defined as: \f[ x \rightarrow \alpha x ]\f
 *
 * @tparam TALPHA Data type of the scalar `alpha`.
 * @tparam TX Data type of the elements in the vector `x`.
 * @param n Number of elements in the vector `x`.
 * @param alpha Scalar value to scale the elements of the vector `x`.
 * @param x Pointer to the data of the input vector `x`.
 * @param incX Increment between consecutive elements in the vector `x`.
 */
template <typename TALPHA, typename TX>
void scal(std::size_t n, const TALPHA & alpha,
          TX *x, std::ptrdiff_t incX)
{
    if (alpha==TALPHA(1)) {
        return;
    }
    if (alpha!=TALPHA(0)) {
        for (std::size_t i=0; i<n; ++i) {
            x[i*incX] *= TX(alpha);
        }
    }
    else {
        for (std::size_t i=0; i<n; ++i) {
            x[i*incX] = TALPHA(0);
        }
    }
}

/**
 * @brief Scale the elements of a matrix by a scalar value.
 *
 * This function multiplies each element of the input matrix `A` by the scalar `alpha`.
 * The operation is performed in-place, meaning the input matrix `A` is modified directly.
 * This operation is defined as: \f[ A \rightarrow \alpha A ]\f
 *
 * @tparam TALPHA Data type of the scalar `alpha`.
 * @tparam T Data type of the elements in the matrix `A`.
 * @param m Number of rows in the matrix `A`.
 * @param n Number of columns in the matrix `A`.
 * @param alpha Scalar value to scale the elements of the matrix `A`.
 * @param A Pointer to the data of the input matrix `A`.
 * @param incRowA Increment between consecutive rows in the matrix `A`.
 * @param incColA Increment between consecutive columns in the matrix `A`.
 */
template <typename TALPHA, typename TA>
void gescal(std::size_t m, std::size_t n,
           const TALPHA & alpha,
           TA *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA)
{
    if (m==0 || n==0 || alpha==TALPHA(1)) {
        return;
    }
    // row major -> scale A^T
    if (incRowA>incColA) {
        gescal(n, m, alpha, A, incColA, incRowA);
        return;
    }
    // col major
    if (alpha!=TALPHA(0)) {
        for (std::size_t j=0; j<n; ++j) {
            for (std::size_t i=0; i<m; ++i) {
                A[i*incRowA+j*incColA] *= TA(alpha);
            }
        }
    } else {
        for (std::size_t j=0; j<n; ++j) {
            for (std::size_t i=0; i<m; ++i) {
                A[i*incRowA+j*incColA] = TA(0);
            }
        }
    }
}


} } //end namespace ulmblas, hpc

#endif //end HPC_BLAS_SCAL_HPP