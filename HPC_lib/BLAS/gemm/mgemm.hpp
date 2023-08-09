#ifndef HPC_BLAS_MGEMM_HPP
#define HPC_BLAS_MGEMM_HPP

#include <cstddef>
#include <HPC_lib/BLAS/axpy.hpp>
#include <HPC_lib/BLAS/scal.hpp>

namespace hpc { namespace blas { 


// declare type alias UGEMM<T> for micro kernel (ugemm) function 
// (via function pointer to micro Kernel function)
template <typename T>
using UGEMM = void (*)(std::size_t, T,
                       const T *, const T *,
                       T,
                       T *, std::ptrdiff_t, std::ptrdiff_t,
                       const T *, const T *);

/**
 * @brief GEMM Macrokernel 
 * 
 * This function takes as inputs pointers to matrices A and B, which represent blocks from 
 * larger parent matrices, and performs the GEMM operation on these blocks. The blocks 
 * are stored in a cache optimized pattern in buffers (packed). The blocks are further divided into 
 * smaller sub-blocks (panels). The actual calculation of the GEMM operation takes place in the 
 * microkernel (ugemm) at the panel level. The macrokernel iteratively passes the panels in A, B and C 
 * that belong together for a calculation step to the microkernel. 
 * 
 * @tparam T Data type for the matrices (e.g., float, double, etc.).
 * @tparam MR Row blocking factor for matrix A.
 * @tparam NR Column blocking factor for matrix B.
 * @tparam ugemm Micro-kernel function alias of type `UGEMM<T>` for performing the actual matrix multiplication.
 * @param[in] M Number of rows in matrix A and the number of rows in matrix C.
 * @param[in] N Number of columns in matrix B and the number of columns in matrix C.
 * @param[in] K Number of columns in matrix A and the number of rows in matrix B.
 * @param[in] alpha Scaling factor for matrices A and B.
 * @param[in] A Pointer to the input matrix A.
 * @param[in] B Pointer to the input matrix B.
 * @param[in] beta Scaling factor for matrix C.
 * @param[in,out] C Pointer to the output matrix C.
 * @param[in] incRowC Increment value for the row index when accessing elements in C.
 * @param[in] incColC Increment value for the column index when accessing elements in C.
 */
template<typename T, std::size_t MR, std::size_t NR, UGEMM<T> ugemm>
void mgemm(std::size_t M, std::size_t N, std::size_t K,
           T alpha, 
           const T *A, const T *B,
           T beta,
           T *C, std::ptrdiff_t incRowC, std::ptrdiff_t incColC)
{
    // number of panels 
    std::size_t mp = (M + MR -1) / MR; // in matrix block A
    std::size_t np = (N + NR -1) / NR; // in matrix block B

    // remainder (nof rows/cols) in last panel
    std::size_t mr_ = M % MR; // A
    std::size_t nr_ = N % NR; // B

    // Buffer for enlarged C matrix block (with potential zero padding)
    T C_[MR*NR];

    // pointers on next panels in A and B (can improve performance in mugemm)
    const T *a_next = A;
    const T *b_next = nullptr;

    // loop panels in B (i.e. rows)
    for (std::size_t j=0; j<np; ++j) {

        // compute number of cols in current panel in B
        std::size_t nr = (j<np-1 || nr_==0) ? NR : nr_;
        //starting address of next panel in B
        b_next = &B[j*K*NR];

        // loop panels in A (i.e. cols)
        for (std::size_t i=0; i<mp; ++i) {

            // compute number of rows in current panel in A
            std::size_t mr = (i<mp-1 || mr_==0) ? MR : mr_;
            // starting address of next panel in A
            a_next = &A[(i+1)*MR*K];
            // handle case of last/outer panel in A
            if (i==mp-1) {
                a_next = A;                 // jump to first panel
                b_next = &B[(j+1)*K*NR];    // jump to next panel
                // handle case of last/outer panel in B
                if (j==np-1) {
                    b_next = B;
                }
            }

            // panel without zero padding -> just call ugemm
            if (mr==MR && nr==NR) {
                ugemm(K, alpha,
                      &A[i*MR*K], &B[j*K*NR],
                      beta,
                      &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC,
                      a_next, b_next);
            // panel with zero padding -> ugemm on C_ + scal + axpy
            } else {
                // perform ugemm on zero padded A and B
                ugemm(K, alpha,
                      &A[i*MR*K], &B[j*K*NR],
                      0,
                      C_, 1, MR,
                      a_next, b_next);
                // Scal non zerp padded C 
                gescal(mr, nr, beta,
                       &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);
                // only copy relevant part from zero padded result to C
                geaxpy(mr, nr, T(1),
                       false, C_, 1, MR,
                       &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);
            }
        }
    }
}


} } // end namespace blas, hpc

#endif //end HPC_BLAS_MGEMM_HPP
