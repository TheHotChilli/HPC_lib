/**
 * @file dot.hpp
 * @brief Definitions of dot product operations for vectors.
 */

#ifndef HPC_BLAS_DOT_HPP
#define HPC_BLAS_DOT_HPP

#include <cstddef>
#include <type_traits>
#include <HPC_lib/utils/conjugate.hpp>

namespace hpc { namespace blas {


/**
 * @brief Calculate the dot product of two vectors.
 *
 * This function calculates the dot product of two input vectors `x` and `y`. The elements of the vectors are optionally
 * conjugated based on the `conjX` and `conjY` parameters. 
 * The operation is defined as: \f[ res \leftarrow \langle x, y \rangle = x^T y ]\f
 *
 * @tparam TX Data type of the elements in vector `x`.
 * @tparam TY Data type of the elements in vector `y`.
 * @param n Number of elements in the vectors `x` and `y`.
 * @param x Pointer to the data of the input vector `x`.
 * @param incX Increment between consecutive elements in the vector `x`.
 * @param conjX Boolean indicating whether to conjugate the elements of the vector `x`.
 * @param y Pointer to the data of the input vector `y`.
 * @param incY Increment between consecutive elements in the vector `y`.
 * @param conjY Boolean indicating whether to conjugate the elements of the vector `y`.
 * @return The dot product of the vectors `x` and `y`.
 */
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

/**
 * @brief Calculate the dot product of two vectors with fusion optimization.
 *
 * This function calculates the dot product of two input vectors `x` and `y`, while applying fusion optimization.
 * The elements of the vectors are optionally conjugated based on the `conjX` and `conjY` parameters.
 * The operation is defined as: \f[ res \leftarrow \langle x, y \rangle = x^T y ]\f
 *
 * @tparam TX Data type of the elements in vector `x`.
 * @tparam TY Data type of the elements in vector `y`.
 * @param n Number of elements in the vectors `x` and `y`.
 * @param x Pointer to the data of the input vector `x`.
 * @param incX Increment between consecutive elements in the vector `x`.
 * @param conjX Boolean indicating whether to conjugate the elements of the vector `x`.
 * @param y Pointer to the data of the input vector `y`.
 * @param incY Increment between consecutive elements in the vector `y`.
 * @param conjY Boolean indicating whether to conjugate the elements of the vector `y`.
 * @return The dot product of the vectors `x` and `y`.
 */
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