#ifndef HPC_UTILS_CONJUGATE_HPP
#define HPC_UTILS_CONJUGATE_HPP

#include <complex>

namespace hpc { namespace utils {

// fallback for non complex types -> simply return input
template <typename T>
T conjugate(T &&x, bool conj)
{
    return x;
}

// for complex types
template <typename T>
std::complex<T> conjugate(const std::complex<T> &x, bool conj)
{
    return conj ? std::conj(x) : x;
}

} } // end namespace utils, hpc

#endif // end HPC_UTILS_CONJUGATE_HPP