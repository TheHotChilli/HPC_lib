#ifndef HPC_BLAS_UGEMM_REF_HPP
#define HPC_BLAS_UGEMM_REF_HPP

#include <cstddef>

namespace hpc { namespace blas { 


/**
 * @brief GEMM Microkernel: Compute the matrix-matrix multiplication C = alpha * A * B + beta * C.
 * 
 * This function computes the matrix-matrix multiplication C = alpha * A * B + beta * C.
 * The input matrices A, B, and C represent blocks from their corresponding parent matrices 
 * and are provided as pointers. A and B are stored in a cache optimized mannor in 1D arrays (packed). 
 * The function performs the multiplication with the given scaling factors alpha and beta. 
 * The result is stored in the output matrix C. 
 * 
 * @tparam T Data type for the matrices (e.g., float, double, etc.).
 * @tparam MR Row blocking factor for matrix A.
 * @tparam NR Column blocking factor for matrix B.
 * @param[in] k Number of columns in matrix A and the number of rows in matrix B.
 * @param[in] alpha Scaling factor for matrices A and B.
 * @param[in] A Pointer to the packed input matrix block A.
 * @param[in] B Pointer to the packed input matrix block B.
 * @param[in] beta Scaling factor for matrix C.
 * @param[in,out] C Pointer to the output matrix C.
 * @param[in] incRowC Increment value for the row index when accessing elements in C.
 * @param[in] incColC Increment value for the column index when accessing elements in C.
 * @param[in] a_next Optional parameter (not used in this implementation).
 * @param[in] b_next Optional parameter (not used in this implementation). 
 */
template <typename T, std::size_t MR, std::size_t NR>
void ugemm_ref(std::size_t k, T alpha,
               const T *A, const T *B,
               T beta,
               T *C std::ptrdiff_t incRowC, std::ptrdiff_t incColC,
               const T* /*a_next*/,
               const T* /*b_next*/)
{
    // zero init array for result A*B
    T AB[MR*NR];
    for (std::size_t i=0; i<MR*NR; ++i) {
        AB[i] = 0;
    }

    // compute AB = alpha*A*B
    if (alpha != T(0)) {
        // AB = A*B
        for (std::size_t l=0; l<k; ++l) {
            for (std::size_t i=0; i<MR; ++i) {
                for (std::size_t j=0; j<NR; ++j) {
                    AB[i*NR + j] += A[i] * B[j];
                }
            }
            A += MR;
            B += NR;
        }
        // compute AB = alpha * AB
        for (std::size_t i=0; i<MR*NR; ++i) {
            AB[i] *= alpha; 
        }
    }

    // compute C = beta * C + AB
    if (beta!=T(0)) {
        for (std::size_t j=0; j<NR; ++j) {
            for (std::size_t i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] *= beta;
                C[i*incRowC+j*incColC] += AB[i*NR+j];
            }
        }
    } else {
        for (std::size_t j=0; j<NR; ++j) {
            for (std::size_t i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] = AB[i*NR+j];
            }
        }
    }
}



} } // end namespace blas, hpc

#endif // end HPC_BLAS_UGEMM_REF_HPP