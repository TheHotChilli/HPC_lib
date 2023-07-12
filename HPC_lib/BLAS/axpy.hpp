#ifndef HPC_BLAS_AXPY_HPP
#define HPC_BLAS_AXPY_HPP

#include <cstddef>
#include <HPC_lib/utils/conjugate.hpp>

namespace hpc { namespace blas { 


template <typename ALPHA, typename TX, typename TY>
void axpy(std::size_t n, const ALPHA &alpha, 
          const TX *x, std::ptrdiff_t incX, bool conjX,
          TY *y, std::ptrdiff_t incY)
{
    if (alpha==ALPHA(0)) {
        return;
    }
    else if (alpha==ALPHA(1)) {
        for (std::size_t i=0; i<n; ++i) {
            y[i*incY] += TY(utils::conjugate(x[i*incX], conjX));
        }
    }
    else {
        for (std::size_t i=0; i<n; ++i) {
            y[i*incY] += TY(alpha) * TY(utils::conjugate(x[i*incX], conjX));
        }
    }
}


} } // end namespace blas, hpc

#endif //end HPC_BLAS_AXPY_HPP