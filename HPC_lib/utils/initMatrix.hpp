/**
 * @file init.hpp
 * @brief Definitions for initializing matrices.
 */

#ifndef HPC_UTILS_INIT_HPP
#define HPC_UTILS_INIT_HPP

#include <cstddef>
#include <limits>

#define MY_ABS(x)   ((x)<0 ? -(x) : (x))

namespace hpc { namespace blas {


/**
 * @brief Initialize a matrix with NaN values or random numbers.
 *
 * This function initializes a matrix with either NaN values or random numbers based on the given parameters.
 *
 * @tparam TA Data type of the matrix elements.
 * @param m Number of rows in the matrix.
 * @param n Number of columns in the matrix.
 * @param A Pointer to the data of the matrix.
 * @param incRowA Increment between consecutive rows in the matrix.
 * @param incColA Increment between consecutive columns in the matrix.
 * @param isNaN Boolean indicating whether to initialize with NaN values.
 */
template <typename TA>
void initMatrix(std::size_t m, std::size_t n, 
                TA *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
                bool isNaN) 
{
    // A is row-major -> init A^T
    if (MY_ABS(incRowA) > MY_ABS(incColA)) {
        initMatrix(n, m, A, incColA, incRowA, isNaN);
        return;
    }
    // A is col-major
    if (isNaN) {
        // NaN init
        for (std::size_t j=0; j<n; ++j) {
            for (std::size_t i=0; i<m; ++i) {
                A[i*incRowA + j*incColA] = std::numeric_limits<TA>::quiet_NaN();
            }
        }
    } else {
        // random number init
        for (std::size_t j=0; j<n; ++j) {
            for (std::size_t i=0; i<m; ++i) {
                double randValue = ((double)rand() - RAND_MAX/2)*2/RAND_MAX;
                A[i*incRowA + j*incColA] = TA(randValue);
            }
        }
    }
}


} } // end namespace hpc, utils

#endif // end HPC_UTILS_INIT_HPP