/**
 * @file gemv.hpp
 * @brief Definitions of GEMV operations (BLAS) for matrix-vector multiplication.
 */

#ifndef HPC_BLAS_GEMV_HPP
#define HPC_BLAS_GEMV_HPP

#include <cstddef>
#include <HPC_lib/BLAS/scal.hpp>
#include <HPC_lib/BLAS/dot.hpp>

namespace hpc { namespace blas {


/**
 * @brief Computes a scaled matrix-vector product and adds the result to another scaled vector.
 *
 * This function performs the GEMV operation, multiplying a matrix `A` by a vector `x`, and adding the result
 * to a vector `y`. The operation is controlled by parameters such as alpha, beta, and conjugation flags.
 * The operation is performed in-place, meaning the vector `y` is modified directly.
 * The operation is defined as: \f[ y\leftarrow \alpha A x + \beta y ]\f
 *
 * @tparam ALPHA Data type of the scalar `alpha`.
 * @tparam BETA Data type of the scalar `beta`.
 * @tparam TX Data type of the elements in vector `x`.
 * @tparam TY Data type of the elements in vector `y`.
 * @tparam TA Data type of the elements in matrix `A`.
 * @param m Number of rows in matrix `A`.
 * @param n Number of columns in matrix `A`.
 * @param alpha Scalar value to scale the matrix-vector product.
 * @param conjA Boolean indicating whether to conjugate the elements of matrix `A`.
 * @param A Pointer to the data of matrix `A`.
 * @param incRowA Increment between consecutive rows in matrix `A`.
 * @param incColA Increment between consecutive columns in matrix `A`.
 * @param conjX Boolean indicating whether to conjugate the elements of vector `x`.
 * @param x Pointer to the data of vector `x`.
 * @param incX Increment between consecutive elements in vector `x`.
 * @param beta Scalar value to scale vector `y` before addition.
 * @param conjY Boolean indicating whether to conjugate the elements of vector `y`.
 * @param y Pointer to the data of vector `y`.
 * @param incY Increment between consecutive elements in vector `y`.
 */
template <typename ALPHA, typename BETA,
          typename TX, typename TY, typename TA>
void gemv(std::size_t m, std::size_t n, 
          const ALPHA &alpha, 
          bool conjA, const TA *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
          bool conjX, const TX *x, std::ptrdiff_t incX,
          const BETA &beta, 
          bool conjY, TY *y, std::ptrdiff_t incY)
{
    // SCAL y: y <- beta * y
    if (beta != BETA(1)) {
        // for (std::size_t i=0; i<m; ++i) {
        //     y[i*incY] *= TY(beta); 
        // }
        scal(m, beta, y, incY);
    }

    // alpha==0 -> no action
    if (alpha == ALPHA(0)) {
        return;
    }

    //row major
    if (incRowA > incColA) { 
        for (std::size_t i=0; i<m; ++i) {
            // y[i*incY] += TY(alpha)*TY(dot(n, &A[i*incRowA], incColA, x, incX));
            for (std::size_t j=0; j<n; ++j) {
                y[i*incY] += TY(alpha) * TY(A[i*incRowA + j*incColA]) * TY(x[j*incX]);  
            }
        }
    } 
    //col major
    else { 
        for (std::size_t j=0; j<n; ++j) {
            // axpy(m, TX(alpha)*x[j*incX], &A[j*incColA], incRowA, y, incY);
            for (std::size_t i=0; i<m; ++i) {
                y[i*incY] += TY(alpha) * TY(A[i*incRowA + j*incColA]) * TY(x[j*incX]);
            }
        }
    }

}


//------------------------------------------------------------------------------

// Fuse factors - cache dependent
#ifndef DGEMV_DOTF_FUSE
#define DGEMV_DOTF_FUSE  6
#endif

#ifndef DGEMV_AXPYF_FUSE
#define DGEMV_AXPYF_FUSE  6
#endif


/**
 * @brief Perform the GEMV operation with fusion optimization.
 *
 * This function performs the GEMV operation with fusion optimization, where the matrix-vector product is
 * split into blocks for improved cache performance. The operation is controlled by parameters such as alpha,
 * beta, and conjugation flags. The operation is performed in-place, meaning the vector `y` is modified directly.
 * The operation is defined as: \f[ y\leftarrow \alpha A x + \beta y ]\f
 *
 * @tparam ALPHA Data type of the scalar `alpha`.
 * @tparam BETA Data type of the scalar `beta`.
 * @tparam TX Data type of the elements in vector `x`.
 * @tparam TY Data type of the elements in vector `y`.
 * @tparam TA Data type of the elements in matrix `A`.
 * @param m Number of rows in matrix `A`.
 * @param n Number of columns in matrix `A`.
 * @param alpha Scalar value to scale the matrix-vector product.
 * @param conjA Boolean indicating whether to conjugate the elements of matrix `A`.
 * @param A Pointer to the data of matrix `A`.
 * @param incRowA Increment between consecutive rows in matrix `A`.
 * @param incColA Increment between consecutive columns in matrix `A`.
 * @param conjX Boolean indicating whether to conjugate the elements of vector `x`.
 * @param x Pointer to the data of vector `x`.
 * @param incX Increment between consecutive elements in vector `x`.
 * @param beta Scalar value to scale vector `y` before addition.
 * @param conjY Boolean indicating whether to conjugate the elements of vector `y`.
 * @param y Pointer to the data of vector `y`.
 * @param incY Increment between consecutive elements in vector `y`.
 */
template <typename ALPHA, typename BETA,
          typename TX, typename TY, typename TA>
void gemv_fused(std::size_t m, std::size_t n, 
          const ALPHA &alpha, 
          bool conjA, const TA *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
          bool conjX, const TX *x, std::ptrdiff_t incX, 
          const BETA &beta, 
          bool conjY, TY *y, std::ptrdiff_t incY)
{
    // SCAL y: y <- beta * y
    if (beta != BETA(1)) {
        // for (std::size_t i=0; i<m; ++i) {
        //     y[i*incY] *= TY(beta); 
        // }
        scal(m, beta, y, incY);
    }

    // alpha==0 -> no action
    if (alpha == ALPHA(0)) {
        return;
    }

    // row major
    if (incRowA > incColA) {
        // number of blocks
        std::size_t m_b = m / DGEMV_DOTF_FUSE;
        // loop first m_b even blocks 
        for (std::size_t i_b=0; i_b<m_b; ++i_b) {
            // get current block
            const TA *A_b = &A[i_b*DGEMV_DOTF_FUSE*incRowA];
            TY *y_b = &y[i_b*DGEMV_DOTF_FUSE*incY];
            // perform fused dot product on block (zick zack pattern)
            for (std::size_t j=0; j<n; ++j) {
                for (std::size_t i=0; i<DGEMV_DOTF_FUSE; ++i) {
                    y_b[i*incY] += TY(alpha) * TY(A_b[i*incRowA+j*incColA]) * TY(x[j*incX]);
                }
            }
        }
        // last odd block (if present)
        for (std::size_t i=m_b*DGEMV_DOTF_FUSE; i<m; ++i) {
            // y[i*incY] += TY(alpha)*TY(dot(n, &A[i*incRowA], incColA, x, incX));
            for (std::size_t j=0; j<n; ++j) {
                y[i*incY] += TY(alpha) * TY(A[i*incRowA + j*incColA]) * TY(x[j*incX]);  
            }
        }
    }
    // col major
    else {
        // number of blocks
        std::size_t n_b = n / DGEMV_AXPYF_FUSE;
        // loop first n_b even blocks
        for (std::size_t j_b=0; j_b<n_b; ++j_b) {
            // get current block
            const TA *A_b = &A[j_b*DGEMV_AXPYF_FUSE*incColA];
            const TX *x_b = &x[j_b*DGEMV_AXPYF_FUSE*incX];
            // perform fused axpy operation on block (zick zack pattern)
            for (std::size_t i=0; i<m; ++i) {
                for (std::size_t j=0; j<DGEMV_AXPYF_FUSE; ++j) {
                    y[i*incY] += TY(alpha) * TY(A_b[i*incRowA + j*incColA]) * TY(x_b[j*incX]);
                }
            }
        }
        // last odd block (if present)
        for (std::size_t j=n_b*DGEMV_AXPYF_FUSE; j<n; ++j) {
            // axpy(m, alpha*x[j*incX], &A[j*incColA], incRowA, y, incY);
            for (std::size_t i=0; i<m; ++i) {
                y[i*incY] += TY(alpha) * TY(A[i*incRowA + j*incColA]) * TY(x[j*incX]);
            }
        }
    }
}  


} } // end namespace hpc, blas

#endif // end HPC_BLAS_GEMV_HPP