#ifndef HPC_BLAS_GEMM_GCCVEC_HPP
#define HPC_BLAS_GEMM_GCCVEC_HPP

#include <cstddef>

namespace hpc { namespace blas { 

/**
 * @brief GEMM micro kernel using GNU vector extensions (GCC). 
 * 
 * This function computes the matrix-matrix multiplication C = alpha * A * B + beta * C.
 * The input matrices A, B, and C represent blocks from their corresponding parent matrices 
 * and are provided as pointers. A and B are stored in a cache optimized mannor in 1D arrays (packed). 
 * The function uses the GNU Compiler Collection (GCC) to utilize SIMD registers and instructions to 
 * perform vector operations in parallel. 
 * 
 * @tparam T Data type of the matrix elements.
 * @tparam MR Number of rows in the micro-panel of matrix A.
 * @tparam NR Number of columns in the micro-panel of matrix B.
 * @tparam vec_bits Register size in bits.
 *                  Should be chosen based on the target architecture and the size of the vector registers
 *                  supported by the CPU. Common Values: 128 bits (for SSE on x86), 256 bits (for AVX on x86), 
 *                  or 512 bits (for AVX-512 on x86). 
 * @param k The size of the inner dimension for the GEMM operation.
 * @param alpha The scaling factor for the product of A and B.
 * @param A Pointer to the input matrix A.
 * @param B Pointer to the input matrix B.
 * @param beta The scaling factor for the matrix C.
 * @param C Pointer to the output matrix C.
 * @param incRowC The increment between consecutive rows of C.
 * @param incColC The increment between consecutive columns of C.
 * @param ptrA Not used in this function (placeholder for future use).
 * @param ptrB Not used in this function (placeholder for future use).
 */
template <typename T, std::size_t MR, std::size_t NR, std::size_t vec_bits>
void ugemm_gccvec(std::size_t k, T alpha,
             const T *A, const T *B,
             T beta,
             T *C, std::ptrdiff_t incRowC, std::ptrdiff_t incColC,
             const T *, const T *)
{
    // register size in bytes (one vector element per register)
    constexpr std::size_t vec_bytes = vec_bits/8;
    // number of elements of type T that can fit in a single vector register
    constexpr std::size_t vec_dbls  = vec_bytes/sizeof(T);
    // number of panels 
    constexpr std::size_t NR_       = NR / vec_dbls;
    // Define GCC vector type (name vec)
    typedef T vec __attribute__((vector_size (vec_bytes)));

    // align A and B
    A = (const T*) __builtin_assume_aligned (A, vec_bytes);
    B = (const T*) __builtin_assume_aligned (B, vec_bytes);

    // allocate array of type 'vec'
    vec AB[MR*NR_] = {};

    // GEMM
    for (std::size_t l=0; l<k; ++l) {
        const vec *b = (const vec *)B;
        for (std::size_t i=0; i<MR; ++i) {
            for (std::size_t j=0; j<NR_; ++j) {
                AB[i*NR_ + j] += A[i]*b[j];
            }
        }
        A += MR;
        B += NR;
    }
    for (std::size_t i=0; i<MR; ++i) {
        for (std::size_t j=0; j<NR_; ++j) {
            AB[i*NR_+j] *= alpha;
        }
    }
    if (beta!=T(0)) {
        for (std::size_t i=0; i<MR; ++i) {
            for (std::size_t j=0; j<NR_; ++j) {
                const T *p = (const T *) &AB[i*NR_+j];
                for (std::size_t j0=0; j0<vec_dbls; ++j0) {
                    C[i*incRowC+(j*vec_dbls+j0)*incColC] *= beta;
                    C[i*incRowC+(j*vec_dbls+j0)*incColC] += p[j0];
                }
            }
        }
    } else {
        for (std::size_t i=0; i<MR; ++i) {
            for (std::size_t j=0; j<NR_; ++j) {
                const T *p = (const T *) &AB[i*NR_+j];
                for (std::size_t j0=0; j0<vec_dbls; ++j0) {
                    C[i*incRowC+(j*vec_dbls+j0)*incColC] = p[j0];
                }
            }
        }
    }
}



} } // end namespace blas, hpc 

#endif // end HPC_BLAS_GEMM_GCCVEC_HPP