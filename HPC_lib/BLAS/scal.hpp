#ifndef HPC_BLAS_SCAL_HPP
#define HPC_BLAS_SCAL_HPP

#include <cstddef>

namespace hpc { namespace blas { 


template <typename TALPHA, typename TX>
void scal(std::size_t n, const TALPHA &alpha,
          TX *x, std::ptrdiff_t incX)
{
    if (alpha==TALPHA(1)) {
        return;
    } else if (alpha==TALPHA(0)) {
        for (std::size_t i=0; i<n; ++i) {
            x[i*incX] = TALPHA(0);
        }
    } else {
        for (std::size_t i=0; i<n; ++i) {
            x[i*incX] *= TX(alpha);
        }
    }
}


} } //end namespace ulmblas, hpc

#endif //end HPC_BLAS_SCAL_HPP