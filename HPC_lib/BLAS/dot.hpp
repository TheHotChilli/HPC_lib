#ifndef HPC_BLAS_DOT_HPP
#define HPC_BLAS_DOT_HPP

#include <cstddef>
#include <type_traits>
#include <HPC_lib/utils/conjugate.hpp>

namespace hpc { namespace blas {

template <typename TX, typename TY>
typename std::common_type<TX, TY>::type dot(
    std::size_t n,
    const TX *x, std::ptrdiff_t incX, bool conjX,
    const TY *y, std::ptrdiff_t incY, bool conjY)
{
    using TRET = typename std::common_type<TX, TY>::type; //find common type between TX and TY
    TRET result = TRET(0);

    for (std::size_t i=0; i<n; ++i) {
        result += T(utils::conjugate(x[i*incX], conjX)),
                  * T(utils::conjugate(y[i*incY], conjY));
    }
    
    return result;
}

//------------------------------------------------------------------------------

#ifndef DGEMV_DOTF_FUSE
#define DGEMV_DOTF_FUSE  6
#endif

#ifndef DGEMV_AXPYF_FUSE
#define DGEMV_AXPYF_FUSE  6
#endif

template <typename TX, typename TY>
typename std::common_type<TX, TY>::type 
dot_fused(
    std::size_t n,
    const TX *x, std::ptrdiff_t incX, bool conjX,
    const TY *y, std::ptrdiff_t incY, bool conjY)
{
    using TRET = typename std::common_type<TX, TY>::type; //find common type between TX and TY
    TRET result = TRET(0);

    for (std::size_t i=0; i<n; ++i) {
        result += T(utils::conjugate(x[i*incX], conjX)),
                  * T(utils::conjugate(y[i*incY], conjY));
    }
    
    return result;
}


} } // end nanmespace hpc, blas


#endif // end HPC_BLAS_DOT_HPP