#ifndef HPC_BLAS_GEMM_CONFIG_HPP
#define HPC_BLAS_GEMM_CONFIG_HPP

#include <cstdlib>
#include <complex>
#include <map>
#include <string>

#include <HPC_lib/BLAS/gemm/gemm.hpp>
#include <HPC_lib/BLAS/gemm/mgemm.hpp>
#include <HPC_lib/BLAS/gemm/packing.hpp>
#include <HPC_lib/BLAS/gemm/ugemm_ref.hpp>
#include <HPC_lib/BLAS/gemm/ugemm_gccvec.hpp>

// Default block sizes
#ifndef MC_GEMM
#define MC_GEMM = 256
#endif
#ifndef NC_GEMM
#define NC_GEMM = 2048
#endif
#ifndef KC_GEMM 
#define KC_GEMM = 256
#endif

// Default panel sizes
#ifndef MR_GEMM
#define MR_GEMM = 4
#endif
#ifndef NR_GEMM
#define NR_GEMM = 8
#endif



namespace hpc { namespace blas { 

namespace config {


// Possible micro kernel configurations
enum ugemm_config {
    Default = 0,
    SSE = 1,
    AVX = 2
};
// ugemm_config ugemm_variant = Defualt;

// Function alias for macro kernel
template <typename T>
using MGemm = void (*)(std::size_t, std::size_t, std::size_t,
                       T, const T *, const T *, T,
                       T *, std::ptrdiff_t, std::ptrdiff_t);
// Function alias for packing algos
template <typename T>
using Pack  = void (*)(std::size_t, std::size_t, bool,
                       const T *, std::ptrdiff_t, std::ptrdiff_t,
                       T *);

// Generic Case
template<typename T, ugemm_config ugemm_variant = Default>
struct GemmConfig 
{
    std::size_t MC, NC, KC; // block sizes
    // std::size_t MR, NR;     // panel sizes
    std::size_t alignment;
    std::size_t extra_A, extra_B;
    
    MGemm<T> mgemm;
    Pack<T> pack_A, pack_B;

    GemmConfig()
    {
        MC = 256;
        NC = 2048;
        KC = 256;
        // MR = 4; 
        // NR = 64;
        alignment =0; 
        extra_A = 0;
        extra_B = 0;

        mgemm  = blas::mgemm<T, 4, 64, blas::ugemm_ref<T, 4, 64>>;
        pack_A = blas::pack_A<T, 4>;
        pack_B = blas::pack_B<T, 64>;
    }
};

// float case
template <>
struct GemmConfig<float, ugemm_config ugemm_variant = Default>
{
    std::size_t MC, NC, KC; // block sizes
    // std::size_t MR, NR;     // panel sizes
    std::size_t alignment;
    std::size_t extra_A, extra_B;
    
    MGemm<float> mgemm;
    Pack<float> pack_A, pack_B;

    GemmConfig()
    {
        extra_A   = 0;
        extra_B   = 0;

        switch (ugemm_variant) {
            case SSE:
                MC = 256;
                NC = 2048;
                KC = 256;
                //MR = 4;
                //NR = 4;
                alignment = 16;

                mgemm  = blas::mgemm<float, 4, 4, ugemm_gccvec<float, 4, 4, 128>>;
                pack_A = blas::pack_A<float, 4>;
                pack_B = blas::pack_B<float, 4>;
                break;

            case AVX:
                MC = 128;
                NC = 2048;
                KC = 384;
                //MR = 8;
                //NR = 8;
                alignment = 32;
                extra_A   = 8;
                extra_B   = 8;

                mgemm  = blas::mgemm<float, 8, 8, sugemm_asm_8x8>;
                pack_A = blas::pack_A<float, 8>;
                pack_B = blas::pack_B<float, 8>;
                break;

            default:
                MC = 256;
                NC = 2048;
                KC = 256;
                // MR = 4;
                // NR = 64;
                alignment = 0;

                mgemm  = blas::mgemm<float, 4, 64, ugemm_ref<float, 4, 64> >;
                pack_A = blas::pack_A<float, 4>;
                pack_B = blas::pack_B<float, 64>;
        }
    }
};

// double case
template <>
struct GemmConfig<double>
{
    std::size_t MC, NC, KC; // block sizes
    // std::size_t MR, NR;     // panel sizes
    std::size_t alignment;
    std::size_t extra_A, extra_B;
    
    MGemm<double> mgemm;
    Pack<double> pack_A, pack_B;

    GemmConfig()
    {
        extra_A   = 0;
        extra_B   = 0;

        switch (ugemm_variant) {
            case SSE:
                MC = 256;
                NC = 2048;
                KC = 256;
                //MR = 4;
                //NR = 4;
                alignment = 16;

                mgemm  = blas::mgemm<double, 4, 4, ugemm_gccvec<double, 4, 4, 128>>;
                pack_A = blas::pack_A<double, 4>;
                pack_B = blas::pack_B<double, 4>;
                break;

            case AVX:
                MC = 256;
                NC = 2048;
                KC = 256;
                //MR = 8;
                //NR = 4;
                alignment = 32;
                extra_A   = 8;
                extra_B   = 4;

                mgemm  = blas::mgemm<double, 8, 4, dugemm_asm_8x4>;
                pack_A = blas::pack_A<double, 8>;
                pack_B = blas::pack_B<double, 4>;
                break;

            default:
                MC = 256;
                NC = 2048;
                KC = 256;
                // MR = 4;
                // NR = 64;
                alignment = 0;

                mgemm  = ulmblas::mgemm<double, 4, 64, ugemm_ref<double, 4, 64> >;
                pack_A = ulmblas::pack_A<double, 4>;
                pack_B = ulmblas::pack_B<double, 64>;
        }
    }
};

// complex float case
template <>
struct GemmConfig<std::complex<float>>
{
    std::size_t MC, NC, KC; // block sizes
    // std::size_t MR, NR;     // panel sizes
    std::size_t alignment;
    std::size_t extra_A, extra_B;
    
    MGemm<std::complex<float>> mgemm;
    Pack<std::complex<float>> pack_A, pack_B;

    GemmConfig()
    {
        extra_A   = 0;
        extra_B   = 0;

        switch (ugemm_variant) {
            // case SSE:
                // to be implemented

            case AVX:
                MC = 96;
                NC = 4096;
                KC = 256;
                //MR = 8;
                //NR = 4;
                alignment = 32;
                extra_A   = 8;
                extra_B   = 4;

                mgemm  = blas::mgemm<std::complex<float>, 8, 4, cugemm_asm_8x4>;
                pack_A = blas::pack_A<std::complex<float>, 8>;
                pack_B = blas::pack_B<std::complex<float>, 4>;
                break;

            default:
                MC = 384;
                NC = 4096;
                KC = 384;
                // MR = 4;
                // NR = 2;
                alignment = 0;

                mgemm  = blas::mgemm<std::complex<float>, 4, 2, ugemm_ref<std::complex<float>, 4, 2> >;
                pack_A = blas::pack_A<std::complex<float>, 4>;
                pack_B = blas::pack_B<std::complex<float>, 64>;
        }
    }
};

// complex double case
template <>
struct GemmConfig<double>
{
    std::size_t MC, NC, KC; // block sizes
    // std::size_t MR, NR;     // panel sizes
    std::size_t alignment;
    std::size_t extra_A, extra_B;
    
    MGemm<std::complex<double>> mgemm;
    Pack<std::complex<double>> pack_A, pack_B;

    GemmConfig()
    {
        extra_A   = 0;
        extra_B   = 0;

        switch (ugemm_variant) {
            // case SSE:
            //     // to be implemented

            case AVX:
                MC = 64;
                NC = 4096;
                KC = 192;
                //MR = 4;
                //NR = 4;
                alignment = 32;
                extra_A   = 4;
                extra_B   = 4;

                mgemm  = blas::mgemm<std::complex<double>, 4, 4, zugemm_asm_4x4>;
                pack_A = blas::pack_A<std::complex<double>, 4>;
                pack_B = blas::pack_B<std::complex<double>, 4>;
                break;

            default:
                MC = 384;
                NC = 4096;
                KC = 384;
                // MR = 4;
                // NR = 2;
                alignment = 0;

                mgemm  = ulmblas::mgemm<std::complex<double>, 4, 2, ugemm_ref<std::complex<double>, 4, 2> >;
                pack_A = ulmblas::pack_A<std::complex<double>, 4>;
                pack_B = ulmblas::pack_B<std::complex<double>, 2>;
        }
    }
};


} //end namespace config

} } // end namespace blas, hpc

#endif // end HPC_BLAS_GEMM_CONFIG_HPP