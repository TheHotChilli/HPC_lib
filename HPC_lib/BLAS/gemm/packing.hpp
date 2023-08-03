#ifndef HPC_BLAS_GEMM_PACKING_HPP
#define HPC_BLAS_GEMM_PACKING_HPP

#include <cstddef>

#include <HPC_lib/utils/conjugate.hpp>

namespace hpc { namespace blas { 


/**
 * @brief Panelizes and packs a matrix (block) A into a flat array p using a specified block size MR. 
 * 
 * This function is a utility function that packs the matrix block A (i.e. block from a bigger matrix) 
 * into a flat array p. Before the packing the matrix block A is penalized (i.e. divided into smaller
 * blocks) using a specified block size NR. The array p will be filled in a way that optimizes cache
 * usage and memory access patterns for later computations (by going through the individual panals in 
 * a special zig-zag pattern). The packed array p will have a compact representation of the matrix 
 * block A. The panel size MR is dependendent on the target hardware/cache and should be chosen#
 * carefully. If the last panel has less then MR rows, additional rows will be zero padded.
 * 
 * @warning The function assumes that the size of the array p is large enough to store the
 *          packed representation of matrix A. Make sure to allocate a sufficiently large
 *          memory block for p to avoid buffer overflows.
 * 
 * @tparam T Data type of the matrix elements.
 * @tparam MR Number of rows in a block for the packed representation.
 *         Should be specified at compile time for optimization.
 * @param[in] M Number of rows in the original matrix (block) A.
 * @param[in] K Number of columns in the original matrix (block) A.
 * @param[in] conjA Boolean indicating whether the elements of A should be conjugated during packing.
 * @param[in] A Pointer to the original matrix (block) data (input matrix).
 * @param[in] incRowA Increment between consecutive rows in the original matrix (block) A.
 * @param[in] incColA Increment between consecutive columns in the original matrix (block) A.
 * @param[out] p Pointer to the packed matrix (output array).
 */
template <typename T, std::size_t MR> //make MR template parameter in order to be specified at compile time
void pack_A(std::size_t M, std::size_t K, 
            bool conjA, const T *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
            T *p)
{
    // number of panels
    std::size_t mp = (M + MR - 1) / MR;

    // col major
    if (incRowA < incColA) {
        for (std::size_t J=0; J<K; ++J) {
            for (std::size_t I=0; I<mp*MR; ++I) {
                // Position in buffer for a_IJ (when zig-zag pattern is applied)
                std::size_t mu = MR*K*(I/MR) + J*MR + (I % MR);
                // Fill Buffer (zero pad if neccessary)
                p[mu] = (I<M) ? utils::conjugate(A[I*incRowA+J*incColA], conjA)
                        : T(0);
            }
        }
    // row major
    } else {
        for (std::size_t I=0; I<mp*MR; ++I) {
            for (std::size_t J=0; J<K; ++J) {
                // Position in buffer for a_IJ (when zig-zag pattern is applied)
                std::size_t mu = MR*K*(I/MR) + J*MR + (I % MR);
                // Fill Buffer (zero pad if neccessary)
                p[mu] = (I<M) ? utils::conjugate(A[I*incRowA+J*incColA], conjA)
                        : T(0);
            }
        }
    }
}

/**
 * @brief Panelizes and packs a matrix (block) A into a flat array p using a specified block size NR. 
 *
 * Internally calls pack_A function with the parameters rearranged.
 * 
 * @see pack_A
 * 
 * @warning The function assumes that the size of the array p is large enough to store the
 *          packed representation of matrix B. Make sure to allocate a sufficiently large
 *          memory block for p to avoid buffer overflows.
 * 
 * @tparam T Data type of the matrix elements.
 * @tparam NR Number of rows in a block for the packed representation.
 *         Should be specified at compile time for optimization.
 * @param[in] K Number of rows in the original matrix (block) B.
 * @param[in] N Number of columns in the original matrix (block) B.
 * @param[in] conjB Boolean indicating whether the elements of B should be conjugated during packing.
 * @param[in] B Pointer to the original matrix (block) data (input matrix).
 * @param[in] incRowB Increment between consecutive rows in the original matrix (block) B.
 * @param[in] incColB Increment between consecutive columns in the original matrix (block) B.
 * @param[out] p Pointer to the packed matrix (output array).
 */
template <typename T, std::size_t NR>
void
pack_B(std::size_t K, std::size_t N, bool conjB,
       const T *B, std::ptrdiff_t incRowB, std::ptrdiff_t incColB,
       T *p)
{
    // Internally calls pack_A function with the parameters rearranged
    pack_A<T, NR>(N, K, conjB, B, incColB, incRowB, p);
}


} } // end namespace blas, hpc

#endif // end HPC_BLAS_GEMM_PACKING_HPP