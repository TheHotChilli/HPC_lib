/**
 * @file gemm.hpp
 * @brief Definitions of frame algorithm for GEMM (General Matrix-Matrix multiplication) operation.
 */

#ifndef HPC_BLAS_GEMM_HPP
#define HPC_BLAS_GEMM_HPP

#include <cstddef>

#include <HPC_lib/BLAS/axpy.hpp>
#include <HPC_lib/BLAS/scal.hpp>
#include <HPC_lib/BLAS/gemm/config.hpp>
#include <HPC_lib/utils/buffer.hpp>

namespace hpc { namespace blas { 


/**
 * @brief Perform the GEMM operation (General Matrix-Matrix multiplication).
 *
 * This function performs the GEMM operation, multiplying two matrices `A` and `B`, and adding the result
 * to a matrix `C`. The operation is controlled by parameters such as alpha, beta, and conjugation flags.
 * The implementation is optimized with various configurations for better performance. 
 * The GEMM operation is defined as: \f[ C \rightarrow \alpha A B + \beta C ]\f
 *
 * @tparam T Data type of the matrices and scalars.
 * @param m Number of rows in matrix `C`.
 * @param n Number of columns in matrix `C`.
 * @param k Number of columns in matrix `A` (and number of rows in matrix `B`).
 * @param alpha Scalar value to scale the matrix-matrix product.
 * @param conjA Boolean indicating whether to conjugate the elements of matrix `A`.
 * @param A Pointer to the data of matrix `A`.
 * @param incRowA Increment between consecutive rows in matrix `A`.
 * @param incColA Increment between consecutive columns in matrix `A`.
 * @param conjB Boolean indicating whether to conjugate the elements of matrix `B`.
 * @param B Pointer to the data of matrix `B`.
 * @param incRowB Increment between consecutive rows in matrix `B`.
 * @param incColB Increment between consecutive columns in matrix `B`.
 * @param beta Scalar value to scale matrix `C` before addition.
 * @param C Pointer to the data of matrix `C`.
 * @param incRowC Increment between consecutive rows in matrix `C`.
 * @param incColC Increment between consecutive columns in matrix `C`.
 */
template <typename T>
void gemm(std::size_t m, std::size_t n, std::size_t k,
          T alpha, 
          bool conjA, const T *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
          bool conjB, const T *B, std::ptrdiff_t incRowB, std::ptrdiff_t incColB,
          T beta, 
          T *C, std::ptrdiff_t incRowC, std::ptrdiff_t incColC)
{
    if (alpha==T(0) || k==0) {
        gescal(m, n, beta, C, incRowC, incColC);
        return;
    }

    // load config
    config::GemmConfig<T> gemm_conf;

    // block sizes
    std::size_t MC = gemm_conf.MC;
    std::size_t NC = gemm_conf.NC;
    std::size_t KC = gemm_conf.KC;

    // number of blocks
    std::size_t mb = (m + MC-1)/MC;
    std::size_t nb = (n + NC-1)/NC;
    std::size_t kb = (k + KC-1)/KC;

    // remainder (number of cols/rows) in last blocks
    std::size_t mc_ = m % MC;
    std::size_t nc_ = n % NC;
    std::size_t kc_ = k % KC;

    // Buffer for blocks 
    utils::Buffer<T> A_(MC*KC+gemm_conf.extra_A);
    utils::Buffer<T> B_(KC*NC+gemm_conf.extra_B); 

    // loop j blocks
    for (std::size_t j=0; j<nb; ++j) {
 
        std::size_t N = (j<nb-1 || nc_==0) ? NC : nc_; 

        // loop l blocks
        for (std::size_t l=0; l<kb; ++l) {

            std::size_t K = (l<kb-1 || kc_==0) ? KC : kc_; 

            T beta_ = (l==0) ? beta : 1;

            gemm_conf.pack_B(K, N, conjB, 
                             &B[l*KC*incRowB+j*NC*incColB], incRowB, incColB,
                             B_.data());

            // loop i blocks
            for(std::size_t i=0; i<mb; ++i) {

                std::size_t M = (i<mb-1 || mc_==0) ? MC : mc_;

                gemm_conf.pack_A(M, K, conjA,
                                 &A[i*MC*incRowA+l*KC*incColA], incRowA, incColA,
                                 A_.data());

                gemm_conf.mgemm(M, N, K, alpha, A_.data(), B_.data(), beta_,
                                &C[i*MC*incRowC+j*NC*incColC], incRowC, incColC);
            }
        }
    }
}


} } // namespace hpc, blas

#endif // HPC_BLAS_GEMM_HPP
