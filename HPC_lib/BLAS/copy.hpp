#ifndef HPC_BLAS_COPY_HPP
#define HPC_BLAS_COPY_HPP

#include <cstddef>
#include <HPC_lib/utils/conjugate.hpp>

namespace hpc { namespace blas { 


/// @brief copy function for vectors (with different element types)
template <typename TX, typename TY>
void copy(std::size_t n,
          bool conjX, const TX *x, std::ptrdiff_t incX, 
          TY *y, std::ptrdiff_t incY)
{
    for (std::size_t i=0; i<n; ++i) {
        y[i*incY] = utils::conjugate(x[i*incX], conjX);
    }
}


/// @brief copy function for matrices (with different element types)
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