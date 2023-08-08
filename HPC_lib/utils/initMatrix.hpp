#ifndef HPC_UTILS_INITMATRIX_HPP
#define HPC_UTILS_INITMATRIX_HPP

#include <cstddef>
#include <limits>

#define MY_ABS(x)   ((x)<0 ? -(x) : (x))

namespace hpc { namespace utils {


template <typename TA>
void initMatrix(std::size_t m, std::size_t n, 
                TA *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
                bool NaN) 
{
    // A is row-major -> init A^T
    if (MY_ABS(incRowA) > MY_ABS(incColA)) {
        initMatrix(n, m, A, incColA, incRowA, NaN);
        return;
    }
    // A is col-major
    if (NaN) {
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

#endif // end HPC_UTILS_INITMATRIX_HPP