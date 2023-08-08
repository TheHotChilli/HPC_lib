#ifndef HPC_BLAS_GEMM_CONFIG_HPP
#define HPC_BLAS_GEMM_CONFIG_HPP

#include <cstdlib>
#include <complex>
#include <map>
#include <string>

// Include macro kernel and packing algos
#include <HPC_lib/BLAS/gemm/mgemm.hpp>
#include <HPC_lib/BLAS/gemm/packing.hpp>

// Include micro kernel variants
#include <HPC_lib/BLAS/gemm/ugemm_ref.hpp>
#include <HPC_lib/BLAS/gemm/ugemm_gccvec.hpp>
#include <HPC_lib/BLAS/gemm/avx_cugemm_8x4.hpp>
#include <HPC_lib/BLAS/gemm/avx_dugemm_8x4.hpp>
#include <HPC_lib/BLAS/gemm/avx_sugemm_8x8.hpp>
#include <HPC_lib/BLAS/gemm/avx_zugemm_4x4.hpp>

namespace hpc { namespace blas { 

namespace config {


// Potential configurations
enum ConfigMode {
    Undefined   = 0,
    Reference   = 1,
    SSE         = 2,
    AVX         = 3,
    AVX_BLIS    = 4,
    AVX_512     = 5
};

ConfigMode getConfigMode()
{
    static ConfigMode config = ConfigMode::Undefined;

    static std::map<std::string, ConfigMode> configMap = {
        {"Reference",   Reference},
        {"SSE",         SSE},
        {"AVX",         AVX},
        {"AVX_BLIS",    AVX_BLIS},
        {"AVX_512",     AVX_512}
    };

    if (config==ConfigMode::Undefined) {
        const char *env = std::getenv("HPC_BLAS_GEMM_CONFIG_MODE");
        std::string sel_env = (env) ? std::string(env) : std::string();
        if (configMap.count(sel_env)>0) {
            config = configMap[sel_env];
        } else {
            std::cout << "WARNING: Invalid ConfigMode provided. Will switch to Reference implementation.";
            config = ConfigMode::Reference;
        }
    }
    return config;
}

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

// Default block dimensions
constexpr std::size_t MC_DEFAULT = 256;
constexpr std::size_t NC_DEFAULT = 2048;
constexpr std::size_t KC_DEFAULT = 256;


// Generic Case
template<typename T>
struct GemmConfig 
{
    std::size_t MC, NC, KC;                 // block sizes
    static constexpr std::size_t MR = 8;    // panels sizes
    static constexpr std::size_t NR = 4;
    std::size_t alignment;                  // custom memory alignment
    std::size_t extra_A, extra_B;           // extra space in packing buffers
    
    MGemm<T> mgemm;
    Pack<T> pack_A, pack_B;

    GemmConfig()
    {
        switch (getConfigMode()) {
            case ConfigMode::SSE:
                MC = MC_DEFAULT;
                NC = NC_DEFAULT;
                KC = KC_DEFAULT;
                alignment = 0; 
                extra_A   = 0;
                extra_B   = 0;

                mgemm  = blas::mgemm<T, MR, NR, blas::ugemm_gccvec<T, MR, NR, 128>>;
                pack_A = blas::pack_A<T, MR>;
                pack_B = blas::pack_B<T, NR>;
            
            case ConfigMode::AVX:
                MC = MC_DEFAULT;
                NC = NC_DEFAULT;
                KC = KC_DEFAULT;
                alignment = 0; // 16
                extra_A   = 0;
                extra_B   = 0;

                mgemm  = blas::mgemm<T, MR, NR, blas::ugemm_gccvec<T, MR, NR, 256>>;
                pack_A = blas::pack_A<T, MR>;
                pack_B = blas::pack_B<T, NR>;

            case ConfigMode::AVX_512:
                MC = MC_DEFAULT;
                NC = NC_DEFAULT;
                KC = KC_DEFAULT;
                alignment = 0; // 16
                extra_A   = 0;
                extra_B   = 0;

                mgemm  = blas::mgemm<T, MR, NR, blas::ugemm_gccvec<T, MR, NR, 512>>;
                pack_A = blas::pack_A<T, MR>;
                pack_B = blas::pack_B<T, NR>;

            default:
                MC = MC_DEFAULT;
                NC = NC_DEFAULT;
                KC = KC_DEFAULT;
                alignment = 0; 
                extra_A   = 0;
                extra_B   = 0;

                mgemm  = blas::mgemm<T, MR, NR, blas::ugemm_ref<T, MR, NR>>;
                pack_A = blas::pack_A<T, MR>;
                pack_B = blas::pack_B<T, NR>;
        }
    }
};


template <>
struct GemmConfig<float>
{
    std::size_t MC, NC, KC;                 // block sizes
    static constexpr std::size_t MR = 8;    // panels sizes
    static constexpr std::size_t NR = 8;
    std::size_t alignment;                  // custom memory alignment
    std::size_t extra_A, extra_B;           // extra space in packing buffers
    
    MGemm<float> mgemm;
    Pack<float> pack_A, pack_B;

    GemmConfig()
    {
        switch (getConfigMode()) {
            case ConfigMode::AVX_BLIS:
                MC = MC_DEFAULT;
                NC = NC_DEFAULT;
                KC = KC_DEFAULT;
                alignment = 32;
                extra_A   = 8;
                extra_B   = 8;

                mgemm  = blas::mgemm<float, MR, NR, sugemm_asm_8x8>;
                pack_A = blas::pack_A<float, MR>;
                pack_B = blas::pack_B<float, NR>;

            case ConfigMode::SSE:
                MC = MC_DEFAULT;
                NC = NC_DEFAULT;
                KC = KC_DEFAULT;
                alignment = 0; 
                extra_A   = 0;
                extra_B   = 0;

                mgemm  = blas::mgemm<float, MR, NR, blas::ugemm_gccvec<float, MR, NR, 128>>;
                pack_A = blas::pack_A<float, MR>;
                pack_B = blas::pack_B<float, NR>;
            
            case ConfigMode::AVX:
                MC = MC_DEFAULT;
                NC = NC_DEFAULT;
                KC = KC_DEFAULT;
                alignment = 0; // 16
                extra_A   = 0;
                extra_B   = 0;

                mgemm  = blas::mgemm<float, MR, NR, blas::ugemm_gccvec<float, MR, NR, 256>>;
                pack_A = blas::pack_A<float, MR>;
                pack_B = blas::pack_B<float, NR>;

            case ConfigMode::AVX_512:
                MC = MC_DEFAULT;
                NC = NC_DEFAULT;
                KC = KC_DEFAULT;
                alignment = 0; // 16
                extra_A   = 0;
                extra_B   = 0;

                mgemm  = blas::mgemm<float, MR, NR, blas::ugemm_gccvec<float, MR, NR, 512>>;
                pack_A = blas::pack_A<float, MR>;
                pack_B = blas::pack_B<float, NR>;

            default:
                MC = MC_DEFAULT;
                NC = NC_DEFAULT;
                KC = KC_DEFAULT;
                alignment = 0; 
                extra_A   = 0;
                extra_B   = 0;

                mgemm  = blas::mgemm<float, MR, NR, blas::ugemm_ref<float, MR, NR>>;
                pack_A = blas::pack_A<float, MR>;
                pack_B = blas::pack_B<float, NR>;
        }
    }
};


template <>
struct GemmConfig<double>
{
    std::size_t MC, NC, KC;                 // block sizes
    static constexpr std::size_t MR = 8;    // panels sizes
    static constexpr std::size_t NR = 4;
    std::size_t alignment;                  // custom memory alignment
    std::size_t extra_A, extra_B;           // extra space in packing buffers
    
    MGemm<double> mgemm;
    Pack<double> pack_A, pack_B;

    GemmConfig()
    {
        switch (getConfigMode()) {
            case ConfigMode::AVX_BLIS:
                MC = MC_DEFAULT;
                NC = NC_DEFAULT;
                KC = KC_DEFAULT;
                alignment = 32;
                extra_A   = 8;
                extra_B   = 4;

                mgemm  = blas::mgemm<double, MR, NR, dugemm_asm_8x4>;
                pack_A = blas::pack_A<double, MR>;
                pack_B = blas::pack_B<double, NR>;

            case ConfigMode::SSE:
                MC = MC_DEFAULT;
                NC = NC_DEFAULT;
                KC = KC_DEFAULT;
                alignment = 0; 
                extra_A   = 0;
                extra_B   = 0;

                mgemm  = blas::mgemm<double, MR, NR, blas::ugemm_gccvec<double, MR, NR, 128>>;
                pack_A = blas::pack_A<double, MR>;
                pack_B = blas::pack_B<double, NR>;
            
            case ConfigMode::AVX:
                MC = MC_DEFAULT;
                NC = NC_DEFAULT;
                KC = KC_DEFAULT;
                alignment = 0; // 16
                extra_A   = 0;
                extra_B   = 0;

                mgemm  = blas::mgemm<double, MR, NR, blas::ugemm_gccvec<double, MR, NR, 256>>;
                pack_A = blas::pack_A<double, MR>;
                pack_B = blas::pack_B<double, NR>;

            case ConfigMode::AVX_512:
                MC = MC_DEFAULT;
                NC = NC_DEFAULT;
                KC = KC_DEFAULT;
                alignment = 0; // 16
                extra_A   = 0;
                extra_B   = 0;

                mgemm  = blas::mgemm<double, MR, NR, blas::ugemm_gccvec<double, MR, NR, 512>>;
                pack_A = blas::pack_A<double, MR>;
                pack_B = blas::pack_B<double, NR>;

            default:
                MC = MC_DEFAULT;
                NC = NC_DEFAULT;
                KC = KC_DEFAULT;
                alignment = 0; 
                extra_A   = 0;
                extra_B   = 0;

                mgemm  = blas::mgemm<double, MR, NR, blas::ugemm_ref<double, MR, NR>>;
                pack_A = blas::pack_A<double, MR>;
                pack_B = blas::pack_B<double, NR>;
        }
    }
};


template <>
struct GemmConfig<std::complex<float>>
{
    std::size_t MC, NC, KC;                 // block sizes
    static constexpr std::size_t MR = 8;    // panels sizes
    static constexpr std::size_t NR = 4;
    std::size_t alignment;                  // custom memory alignment
    std::size_t extra_A, extra_B;           // extra space in packing buffers
    
    MGemm<std::complex<float>> mgemm;
    Pack<std::complex<float>> pack_A, pack_B;

    GemmConfig()
    {
        switch (getConfigMode()) {
            case ConfigMode::AVX_BLIS:
                MC = MC_DEFAULT;
                NC = NC_DEFAULT;
                KC = KC_DEFAULT;
                alignment = 32;
                extra_A   = 8;
                extra_B   = 4;

                mgemm  = blas::mgemm<std::complex<float>, MR, NR, cugemm_asm_8x4>;
                pack_A = blas::pack_A<std::complex<float>, MR>;
                pack_B = blas::pack_B<std::complex<float>, NR>;

            // case ConfigMode::SSE:
            //     MC = MC_DEFAULT;
            //     NC = NC_DEFAULT;
            //     KC = KC_DEFAULT;
            //     alignment = 0; 
            //     extra_A   = 0;
            //     extra_B   = 0;

            //     mgemm  = blas::mgemm<std::complex<float>, MR, NR, blas::ugemm_gccvec<std::complex<float>, MR, NR, 128>>;
            //     pack_A = blas::pack_A<std::complex<float>, MR>;
            //     pack_B = blas::pack_B<std::complex<float>, NR>;
            
            // case ConfigMode::AVX:
            //     MC = MC_DEFAULT;
            //     NC = NC_DEFAULT;
            //     KC = KC_DEFAULT;
            //     alignment = 0; // 16
            //     extra_A   = 0;
            //     extra_B   = 0;

            //     mgemm  = blas::mgemm<std::complex<float>, MR, NR, blas::ugemm_gccvec<std::complex<float>, MR, NR, 256>>;
            //     pack_A = blas::pack_A<std::complex<float>, MR>;
            //     pack_B = blas::pack_B<std::complex<float>, NR>;

            // case ConfigMode::AVX_512:
            //     MC = MC_DEFAULT;
            //     NC = NC_DEFAULT;
            //     KC = KC_DEFAULT;
            //     alignment = 0; // 16
            //     extra_A   = 0;
            //     extra_B   = 0;

            //     mgemm  = blas::mgemm<std::complex<float>, MR, NR, blas::ugemm_gccvec<std::complex<float>, MR, NR, 512>>;
            //     pack_A = blas::pack_A<std::complex<float>, MR>;
            //     pack_B = blas::pack_B<std::complex<float>, NR>;

            default:
                MC = MC_DEFAULT;
                NC = NC_DEFAULT;
                KC = KC_DEFAULT;
                alignment = 0; 
                extra_A   = 0;
                extra_B   = 0;

                mgemm  = blas::mgemm<std::complex<float>, MR, NR, blas::ugemm_ref<std::complex<float>, MR, NR>>;
                pack_A = blas::pack_A<std::complex<float>, MR>;
                pack_B = blas::pack_B<std::complex<float>, NR>;
        }
    }
};


template <>
struct GemmConfig<std::complex<double>>
{
    std::size_t MC, NC, KC;                 // block sizes
    static constexpr std::size_t MR = 4;    // panels sizes
    static constexpr std::size_t NR = 4;
    std::size_t alignment;                  // custom memory alignment
    std::size_t extra_A, extra_B;           // extra space in packing buffers
    
    MGemm<std::complex<double>> mgemm;
    Pack<std::complex<double>> pack_A, pack_B;

    GemmConfig()
    {
        switch (getConfigMode()) {
            case ConfigMode::AVX_BLIS:
                MC = MC_DEFAULT;
                NC = NC_DEFAULT;
                KC = KC_DEFAULT;
                alignment = 32;
                extra_A   = 4;
                extra_B   = 4;

                mgemm  = blas::mgemm<std::complex<double>, MR, NR, zugemm_asm_4x4>;
                pack_A = blas::pack_A<std::complex<double>, MR>;
                pack_B = blas::pack_B<std::complex<double>, NR>;

            // case ConfigMode::SSE:
            //     MC = MC_DEFAULT;
            //     NC = NC_DEFAULT;
            //     KC = KC_DEFAULT;
            //     alignment = 0; 
            //     extra_A   = 0;
            //     extra_B   = 0;

            //     mgemm  = blas::mgemm<std::complex<double>, MR, NR, blas::ugemm_gccvec<std::complex<double>, MR, NR, 128>>;
            //     pack_A = blas::pack_A<std::complex<double>, MR>;
            //     pack_B = blas::pack_B<std::complex<double>, NR>;
            
            // case ConfigMode::AVX:
            //     MC = MC_DEFAULT;
            //     NC = NC_DEFAULT;
            //     KC = KC_DEFAULT;
            //     alignment = 0; // 16
            //     extra_A   = 0;
            //     extra_B   = 0;

            //     mgemm  = blas::mgemm<std::complex<double>, MR, NR, blas::ugemm_gccvec<std::complex<double>, MR, NR, 256>>;
            //     pack_A = blas::pack_A<std::complex<double>, MR>;
            //     pack_B = blas::pack_B<std::complex<double>, NR>;

            // case ConfigMode::AVX_512:
            //     MC = MC_DEFAULT;
            //     NC = NC_DEFAULT;
            //     KC = KC_DEFAULT;
            //     alignment = 0; // 16
            //     extra_A   = 0;
            //     extra_B   = 0;

            //     mgemm  = blas::mgemm<std::complex<double>, MR, NR, blas::ugemm_gccvec<std::complex<double>, MR, NR, 512>>;
            //     pack_A = blas::pack_A<std::complex<double>, MR>;
            //     pack_B = blas::pack_B<std::complex<double>, NR>;

            default:
                MC = MC_DEFAULT;
                NC = NC_DEFAULT;
                KC = KC_DEFAULT;
                alignment = 0; 
                extra_A   = 0;
                extra_B   = 0;

                mgemm  = blas::mgemm<std::complex<double>, MR, NR, blas::ugemm_ref<std::complex<double>, MR, NR>>;
                pack_A = blas::pack_A<std::complex<double>, MR>;
                pack_B = blas::pack_B<std::complex<double>, NR>;
        }
    }
};

} //end namespace config

} } // end namespace blas, hpc

#endif // end HPC_BLAS_GEMM_CONFIG_HPP