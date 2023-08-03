#ifndef HPC_BLAS_GEMM_AVX_ZUGEMM_4X4_HPP
#define HPC_BLAS_GEMM_AVX_ZUGEMM_4X4_HPP

//
//  BLIS
//  An object-based framework for developing high-performance BLAS-like
//  libraries.
//
// Copyright (C) 2014, The University of Texas at Austin
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//  - Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  - Neither the name of The University of Texas at Austin nor the names
//    of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <cstddef>
#include <cstdint>

namespace hpc { namespace blas {

void
zugemm_asm_4x4(std::size_t kc, std::complex<double> alpha,
               const std::complex<double> *A, const std::complex<double> *B,
               std::complex<double> beta,
               std::complex<double> *C,
               std::ptrdiff_t incRowC, std::ptrdiff_t incColC,
               const std::complex<double> *,
               const std::complex<double> *)
{
    std::int64_t   k_iter = kc / 4;
    std::int64_t   k_left = kc % 4;

    std::int64_t rs_c = incRowC;
    std::int64_t cs_c = incColC;

    double *pAlpha  = (double *)&alpha;
    double *pBeta   = (double *)&beta;

    __asm__ volatile
    (
    "                                            \n\t"
    "                                            \n\t"
    "movq                %2, %%rax               \n\t" // load address of a.
    "movq                %3, %%rbx               \n\t" // load address of b.
    //"movq                %9, %%r15               \n\t" // load address of b_next.
    //"movq               %10, %%r14               \n\t" // load address of a_next.
    "                                            \n\t"
    "vmovapd        0 * 32(%%rax), %%ymm0        \n\t" // initialize loop by pre-loading
    "vmovddup   0 + 0 * 32(%%rbx), %%ymm2        \n\t"
    "vmovddup   0 + 1 * 32(%%rbx), %%ymm3        \n\t"
    "                                            \n\t"
    "movq                %6, %%rcx               \n\t" // load address of c
    "movq                %8, %%rdi               \n\t" // load cs_c
    "leaq        (,%%rdi,8), %%rdi               \n\t" // cs_c *= sizeof(dcomplex)
    "leaq        (,%%rdi,2), %%rdi               \n\t"
    "leaq   (%%rcx,%%rdi,2), %%r10               \n\t" // load address of c + 2*cs_c;
    "                                            \n\t"
    "prefetcht0   3 * 8(%%rcx)                   \n\t" // prefetch c + 0*cs_c
    "prefetcht0   3 * 8(%%rcx,%%rdi)             \n\t" // prefetch c + 1*cs_c
    "prefetcht0   3 * 8(%%r10)                   \n\t" // prefetch c + 2*cs_c
    "prefetcht0   3 * 8(%%r10,%%rdi)             \n\t" // prefetch c + 3*cs_c
    "                                            \n\t"
    "vxorpd    %%ymm8,  %%ymm8,  %%ymm8          \n\t"
    "vxorpd    %%ymm9,  %%ymm9,  %%ymm9          \n\t"
    "vxorpd    %%ymm10, %%ymm10, %%ymm10         \n\t"
    "vxorpd    %%ymm11, %%ymm11, %%ymm11         \n\t"
    "vxorpd    %%ymm12, %%ymm12, %%ymm12         \n\t"
    "vxorpd    %%ymm13, %%ymm13, %%ymm13         \n\t"
    "vxorpd    %%ymm14, %%ymm14, %%ymm14         \n\t"
    "vxorpd    %%ymm15, %%ymm15, %%ymm15         \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "movq      %0, %%rsi                         \n\t" // i = k_iter;
    "testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
    "je     .ZCONSIDKLEFT%=                      \n\t" // if i == 0, jump to code that
    "                                            \n\t" // contains the k_left loop.
    "                                            \n\t"
    "                                            \n\t"
    ".ZLOOPKITER%=:                              \n\t" // MAIN LOOP
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t" // iteration 0
    "vmovapd        1 * 32(%%rax),      %%ymm1   \n\t"
    "vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
    "vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
    "vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
    "vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
    "vaddpd           %%ymm6,  %%ymm15, %%ymm15  \n\t"
    "vaddpd           %%ymm7,  %%ymm11, %%ymm11  \n\t"
    "                                            \n\t"
    "prefetcht0    16 * 32(%%rax)                \n\t"
    "vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
    "vmovddup   8 + 0 * 32(%%rbx),      %%ymm2   \n\t"
    "vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
    "vmovddup   8 + 1 * 32(%%rbx),      %%ymm3   \n\t"
    "vaddpd           %%ymm6,  %%ymm14, %%ymm14  \n\t"
    "vaddpd           %%ymm7,  %%ymm10, %%ymm10  \n\t"
    "                                            \n\t"
    "vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
    "vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
    "vpermilpd  $0x5, %%ymm0,  %%ymm0            \n\t"
    "vaddpd           %%ymm6,  %%ymm13, %%ymm13  \n\t"
    "vaddpd           %%ymm7,  %%ymm9,  %%ymm9   \n\t"
    "                                            \n\t"
    "vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
    "vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
    "vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
    "vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
    "vaddpd           %%ymm6,  %%ymm12, %%ymm12  \n\t"
    "vaddpd           %%ymm7,  %%ymm8,  %%ymm8   \n\t"
    "                                            \n\t"
    "vpermilpd  $0x5, %%ymm1,  %%ymm1            \n\t"
    "vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
    "vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
    "vaddsubpd        %%ymm6,  %%ymm15, %%ymm15  \n\t"
    "vaddsubpd        %%ymm7,  %%ymm11, %%ymm11  \n\t"
    "                                            \n\t"
    "vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
    "vmovddup   0 + 2 * 32(%%rbx),      %%ymm2   \n\t"
    "vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
    "vmovddup   0 + 3 * 32(%%rbx),      %%ymm3   \n\t"
    "vaddsubpd        %%ymm6,  %%ymm14, %%ymm14  \n\t"
    "vaddsubpd        %%ymm7,  %%ymm10, %%ymm10  \n\t"
    "                                            \n\t"
    "vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
    "vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
    "vmovapd        2 * 32(%%rax),      %%ymm0   \n\t"
    "vaddsubpd        %%ymm6,  %%ymm13, %%ymm13  \n\t"
    "vaddsubpd        %%ymm7,  %%ymm9,  %%ymm9   \n\t"
    "                                            \n\t"
    "vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
    "vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
    "vaddsubpd        %%ymm6,  %%ymm12, %%ymm12  \n\t"
    "vaddsubpd        %%ymm7,  %%ymm8,  %%ymm8   \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t" // iteration 1
    "vmovapd        3 * 32(%%rax),      %%ymm1   \n\t"
    "vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
    "vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
    "vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
    "vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
    "vaddpd           %%ymm6,  %%ymm15, %%ymm15  \n\t"
    "vaddpd           %%ymm7,  %%ymm11, %%ymm11  \n\t"
    "                                            \n\t"
    "prefetcht0    18 * 32(%%rax)                \n\t"
    "vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
    "vmovddup   8 + 2 * 32(%%rbx),      %%ymm2   \n\t"
    "vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
    "vmovddup   8 + 3 * 32(%%rbx),      %%ymm3   \n\t"
    "vaddpd           %%ymm6,  %%ymm14, %%ymm14  \n\t"
    "vaddpd           %%ymm7,  %%ymm10, %%ymm10  \n\t"
    "                                            \n\t"
    "vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
    "vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
    "vpermilpd  $0x5, %%ymm0,  %%ymm0            \n\t"
    "vaddpd           %%ymm6,  %%ymm13, %%ymm13  \n\t"
    "vaddpd           %%ymm7,  %%ymm9,  %%ymm9   \n\t"
    "                                            \n\t"
    "vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
    "vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
    "vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
    "vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
    "vaddpd           %%ymm6,  %%ymm12, %%ymm12  \n\t"
    "vaddpd           %%ymm7,  %%ymm8,  %%ymm8   \n\t"
    "                                            \n\t"
    "vpermilpd  $0x5, %%ymm1,  %%ymm1            \n\t"
    "vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
    "vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
    "vaddsubpd        %%ymm6,  %%ymm15, %%ymm15  \n\t"
    "vaddsubpd        %%ymm7,  %%ymm11, %%ymm11  \n\t"
    "                                            \n\t"
    "vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
    "vmovddup   0 + 4 * 32(%%rbx),      %%ymm2   \n\t"
    "vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
    "vmovddup   0 + 5 * 32(%%rbx),      %%ymm3   \n\t"
    "vaddsubpd        %%ymm6,  %%ymm14, %%ymm14  \n\t"
    "vaddsubpd        %%ymm7,  %%ymm10, %%ymm10  \n\t"
    "                                            \n\t"
    "vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
    "vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
    "vmovapd        4 * 32(%%rax),      %%ymm0   \n\t"
    "vaddsubpd        %%ymm6,  %%ymm13, %%ymm13  \n\t"
    "vaddsubpd        %%ymm7,  %%ymm9,  %%ymm9   \n\t"
    "                                            \n\t"
    "vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
    "vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
    "vaddsubpd        %%ymm6,  %%ymm12, %%ymm12  \n\t"
    "vaddsubpd        %%ymm7,  %%ymm8,  %%ymm8   \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t" // iteration 2
    "vmovapd        5 * 32(%%rax),      %%ymm1   \n\t"
    "vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
    "vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
    "vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
    "vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
    "vaddpd           %%ymm6,  %%ymm15, %%ymm15  \n\t"
    "vaddpd           %%ymm7,  %%ymm11, %%ymm11  \n\t"
    "                                            \n\t"
    "prefetcht0    20 * 32(%%rax)                \n\t"
    "vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
    "vmovddup   8 + 4 * 32(%%rbx),      %%ymm2   \n\t"
    "vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
    "vmovddup   8 + 5 * 32(%%rbx),      %%ymm3   \n\t"
    "vaddpd           %%ymm6,  %%ymm14, %%ymm14  \n\t"
    "vaddpd           %%ymm7,  %%ymm10, %%ymm10  \n\t"
    "                                            \n\t"
    "vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
    "vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
    "vpermilpd  $0x5, %%ymm0,  %%ymm0            \n\t"
    "vaddpd           %%ymm6,  %%ymm13, %%ymm13  \n\t"
    "vaddpd           %%ymm7,  %%ymm9,  %%ymm9   \n\t"
    "                                            \n\t"
    "vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
    "vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
    "vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
    "vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
    "vaddpd           %%ymm6,  %%ymm12, %%ymm12  \n\t"
    "vaddpd           %%ymm7,  %%ymm8,  %%ymm8   \n\t"
    "                                            \n\t"
    "vpermilpd  $0x5, %%ymm1,  %%ymm1            \n\t"
    "vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
    "vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
    "vaddsubpd        %%ymm6,  %%ymm15, %%ymm15  \n\t"
    "vaddsubpd        %%ymm7,  %%ymm11, %%ymm11  \n\t"
    "                                            \n\t"
    "vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
    "vmovddup   0 + 6 * 32(%%rbx),      %%ymm2   \n\t"
    "vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
    "vmovddup   0 + 7 * 32(%%rbx),      %%ymm3   \n\t"
    "vaddsubpd        %%ymm6,  %%ymm14, %%ymm14  \n\t"
    "vaddsubpd        %%ymm7,  %%ymm10, %%ymm10  \n\t"
    "                                            \n\t"
    "vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
    "vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
    "vmovapd        6 * 32(%%rax),      %%ymm0   \n\t"
    "vaddsubpd        %%ymm6,  %%ymm13, %%ymm13  \n\t"
    "vaddsubpd        %%ymm7,  %%ymm9,  %%ymm9   \n\t"
    "                                            \n\t"
    "vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
    "vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
    "vaddsubpd        %%ymm6,  %%ymm12, %%ymm12  \n\t"
    "vaddsubpd        %%ymm7,  %%ymm8,  %%ymm8   \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t" // iteration 3
    "vmovapd        7 * 32(%%rax),      %%ymm1   \n\t"
    "vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
    "vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
    "vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
    "vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
    "vaddpd           %%ymm6,  %%ymm15, %%ymm15  \n\t"
    "vaddpd           %%ymm7,  %%ymm11, %%ymm11  \n\t"
    "                                            \n\t"
    "prefetcht0    22 * 32(%%rax)                \n\t"
    "vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
    "vmovddup   8 + 6 * 32(%%rbx),      %%ymm2   \n\t"
    "vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
    "vmovddup   8 + 7 * 32(%%rbx),      %%ymm3   \n\t"
    "vaddpd           %%ymm6,  %%ymm14, %%ymm14  \n\t"
    "vaddpd           %%ymm7,  %%ymm10, %%ymm10  \n\t"
    "                                            \n\t"
    "vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
    "vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
    "vpermilpd  $0x5, %%ymm0,  %%ymm0            \n\t"
    "vaddpd           %%ymm6,  %%ymm13, %%ymm13  \n\t"
    "vaddpd           %%ymm7,  %%ymm9,  %%ymm9   \n\t"
    "                                            \n\t"
    "vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
    "vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
    "vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
    "vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
    "vaddpd           %%ymm6,  %%ymm12, %%ymm12  \n\t"
    "vaddpd           %%ymm7,  %%ymm8,  %%ymm8   \n\t"
    "                                            \n\t"
    "vpermilpd  $0x5, %%ymm1,  %%ymm1            \n\t"
    "vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
    "vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
    "vaddsubpd        %%ymm6,  %%ymm15, %%ymm15  \n\t"
    "vaddsubpd        %%ymm7,  %%ymm11, %%ymm11  \n\t"
    "                                            \n\t"
    "vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
    "vmovddup   0 + 8 * 32(%%rbx),      %%ymm2   \n\t"
    "vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
    "vmovddup   0 + 9 * 32(%%rbx),      %%ymm3   \n\t"
    "vaddsubpd        %%ymm6,  %%ymm14, %%ymm14  \n\t"
    "vaddsubpd        %%ymm7,  %%ymm10, %%ymm10  \n\t"
    "                                            \n\t"
    "vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
    "vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
    "vmovapd        8 * 32(%%rax),      %%ymm0   \n\t"
    "vaddsubpd        %%ymm6,  %%ymm13, %%ymm13  \n\t"
    "vaddsubpd        %%ymm7,  %%ymm9,  %%ymm9   \n\t"
    "                                            \n\t"
    "vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
    "vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
    "vaddsubpd        %%ymm6,  %%ymm12, %%ymm12  \n\t"
    "vaddsubpd        %%ymm7,  %%ymm8,  %%ymm8   \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "addq         $4 * 4 * 16, %%rbx             \n\t" // b += 4*4 (unroll x nr)
    "addq         $4 * 4 * 16, %%rax             \n\t" // a += 4*4 (unroll x mr)
    "                                            \n\t"
    "                                            \n\t"
    "decq   %%rsi                                \n\t" // i -= 1;
    "jne    .ZLOOPKITER%=                        \n\t" // iterate again if i != 0.
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".ZCONSIDKLEFT%=:                            \n\t"
    "                                            \n\t"
    "movq      %1, %%rsi                         \n\t" // i = k_left;
    "testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
    "je     .ZPOSTACCUM%=                        \n\t" // if i == 0, we're done; jump to end.
    "                                            \n\t" // else, we prepare to enter k_left loop.
    "                                            \n\t"
    "                                            \n\t"
    ".ZLOOPKLEFT%=:                              \n\t" // EDGE LOOP
    "                                            \n\t"
    "                                            \n\t" // iteration 0
    "vmovapd        1 * 32(%%rax),      %%ymm1   \n\t"
    "vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
    "vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
    "vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
    "vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
    "vaddpd           %%ymm6,  %%ymm15, %%ymm15  \n\t"
    "vaddpd           %%ymm7,  %%ymm11, %%ymm11  \n\t"
    "                                            \n\t"
    "prefetcht0    16 * 32(%%rax)                \n\t"
    "vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
    "vmovddup   8 + 0 * 32(%%rbx),      %%ymm2   \n\t"
    "vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
    "vmovddup   8 + 1 * 32(%%rbx),      %%ymm3   \n\t"
    "vaddpd           %%ymm6,  %%ymm14, %%ymm14  \n\t"
    "vaddpd           %%ymm7,  %%ymm10, %%ymm10  \n\t"
    "                                            \n\t"
    "vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
    "vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
    "vpermilpd  $0x5, %%ymm0,  %%ymm0            \n\t"
    "vaddpd           %%ymm6,  %%ymm13, %%ymm13  \n\t"
    "vaddpd           %%ymm7,  %%ymm9,  %%ymm9   \n\t"
    "                                            \n\t"
    "vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
    "vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
    "vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
    "vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
    "vaddpd           %%ymm6,  %%ymm12, %%ymm12  \n\t"
    "vaddpd           %%ymm7,  %%ymm8,  %%ymm8   \n\t"
    "                                            \n\t"
    "vpermilpd  $0x5, %%ymm1,  %%ymm1            \n\t"
    "vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
    "vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
    "vaddsubpd        %%ymm6,  %%ymm15, %%ymm15  \n\t"
    "vaddsubpd        %%ymm7,  %%ymm11, %%ymm11  \n\t"
    "                                            \n\t"
    "vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
    "vmovddup   0 + 2 * 32(%%rbx),      %%ymm2   \n\t"
    "vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
    "vmovddup   0 + 3 * 32(%%rbx),      %%ymm3   \n\t"
    "vaddsubpd        %%ymm6,  %%ymm14, %%ymm14  \n\t"
    "vaddsubpd        %%ymm7,  %%ymm10, %%ymm10  \n\t"
    "                                            \n\t"
    "vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
    "vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
    "vmovapd        2 * 32(%%rax),      %%ymm0   \n\t"
    "vaddsubpd        %%ymm6,  %%ymm13, %%ymm13  \n\t"
    "vaddsubpd        %%ymm7,  %%ymm9,  %%ymm9   \n\t"
    "                                            \n\t"
    "vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
    "vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
    "vaddsubpd        %%ymm6,  %%ymm12, %%ymm12  \n\t"
    "vaddsubpd        %%ymm7,  %%ymm8,  %%ymm8   \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "addq         $4 * 1 * 16, %%rax             \n\t" // a += 4 (1 x mr)
    "addq         $4 * 1 * 16, %%rbx             \n\t" // b += 4 (1 x nr)
    "                                            \n\t"
    "                                            \n\t"
    "decq   %%rsi                                \n\t" // i -= 1;
    "jne    .ZLOOPKLEFT%=                        \n\t" // iterate again if i != 0.
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".ZPOSTACCUM%=:                              \n\t"
    "                                            \n\t"
    "                                            \n\t" // ymm15:  ymm13:  ymm11:  ymm9:
    "                                            \n\t" // ( ab00  ( ab01  ( ab02  ( ab03
    "                                            \n\t" //   ab10    ab11    ab12    ab13  
    "                                            \n\t" //   ab21    ab20    ab23    ab22
    "                                            \n\t" //   ab31 )  ab30 )  ab33 )  ab32 )
    "                                            \n\t"
    "                                            \n\t" // ymm14:  ymm12:  ymm10:  ymm8:
    "                                            \n\t" // ( ab40  ( ab41  ( ab42  ( ab43
    "                                            \n\t" //   ab50    ab51    ab52    ab53  
    "                                            \n\t" //   ab61    ab60    ab63    ab62
    "                                            \n\t" //   ab71 )  ab70 )  ab73 )  ab72 )
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd           %%ymm15, %%ymm7           \n\t"
    "vperm2f128 $0x12, %%ymm15, %%ymm13, %%ymm15 \n\t"
    "vperm2f128 $0x30, %%ymm7,  %%ymm13, %%ymm13 \n\t"
    "                                            \n\t"
    "vmovapd           %%ymm11, %%ymm7           \n\t"
    "vperm2f128 $0x12, %%ymm11, %%ymm9,  %%ymm11 \n\t"
    "vperm2f128 $0x30, %%ymm7,  %%ymm9,  %%ymm9  \n\t"
    "                                            \n\t"
    "vmovapd           %%ymm14, %%ymm7           \n\t"
    "vperm2f128 $0x12, %%ymm14, %%ymm12, %%ymm14 \n\t"
    "vperm2f128 $0x30, %%ymm7,  %%ymm12, %%ymm12 \n\t"
    "                                            \n\t"
    "vmovapd           %%ymm10, %%ymm7           \n\t"
    "vperm2f128 $0x12, %%ymm10, %%ymm8,  %%ymm10 \n\t"
    "vperm2f128 $0x30, %%ymm7,  %%ymm8,  %%ymm8  \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t" // ymm15:  ymm13:  ymm11:  ymm9:
    "                                            \n\t" // ( ab00  ( ab01  ( ab02  ( ab03
    "                                            \n\t" //   ab10    ab11    ab12    ab13  
    "                                            \n\t" //   ab20    ab21    ab22    ab23
    "                                            \n\t" //   ab30 )  ab31 )  ab32 )  ab33 )
    "                                            \n\t"
    "                                            \n\t" // ymm14:  ymm12:  ymm10:  ymm8:
    "                                            \n\t" // ( ab40  ( ab41  ( ab42  ( ab43
    "                                            \n\t" //   ab50    ab51    ab52    ab53  
    "                                            \n\t" //   ab60    ab61    ab62    ab63
    "                                            \n\t" //   ab70 )  ab71 )  ab72 )  ab73 )
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t" // scale by alpha
    "                                            \n\t"
    "movq         %4, %%rax                      \n\t" // load address of alpha
    "vbroadcastsd    (%%rax), %%ymm7             \n\t" // load alpha_r and duplicate
    "vbroadcastsd   8(%%rax), %%ymm6             \n\t" // load alpha_i and duplicate
    "                                            \n\t"
    "vpermilpd  $0x5, %%ymm15, %%ymm3            \n\t"
    "vmulpd           %%ymm7,  %%ymm15, %%ymm15  \n\t"
    "vmulpd           %%ymm6,  %%ymm3,  %%ymm3   \n\t"
    "vaddsubpd        %%ymm3,  %%ymm15, %%ymm15  \n\t"
    "                                            \n\t"
    "vpermilpd  $0x5, %%ymm14, %%ymm2            \n\t"
    "vmulpd           %%ymm7,  %%ymm14, %%ymm14  \n\t"
    "vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubpd        %%ymm2,  %%ymm14, %%ymm14  \n\t"
    "                                            \n\t"
    "vpermilpd  $0x5, %%ymm13, %%ymm1            \n\t"
    "vmulpd           %%ymm7,  %%ymm13, %%ymm13  \n\t"
    "vmulpd           %%ymm6,  %%ymm1,  %%ymm1   \n\t"
    "vaddsubpd        %%ymm1,  %%ymm13, %%ymm13  \n\t"
    "                                            \n\t"
    "vpermilpd  $0x5, %%ymm12, %%ymm0            \n\t"
    "vmulpd           %%ymm7,  %%ymm12, %%ymm12  \n\t"
    "vmulpd           %%ymm6,  %%ymm0,  %%ymm0   \n\t"
    "vaddsubpd        %%ymm0,  %%ymm12, %%ymm12  \n\t"
    "                                            \n\t"
    "vpermilpd  $0x5, %%ymm11, %%ymm3            \n\t"
    "vmulpd           %%ymm7,  %%ymm11, %%ymm11  \n\t"
    "vmulpd           %%ymm6,  %%ymm3,  %%ymm3   \n\t"
    "vaddsubpd        %%ymm3,  %%ymm11, %%ymm11  \n\t"
    "                                            \n\t"
    "vpermilpd  $0x5, %%ymm10, %%ymm2            \n\t"
    "vmulpd           %%ymm7,  %%ymm10, %%ymm10  \n\t"
    "vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubpd        %%ymm2,  %%ymm10, %%ymm10  \n\t"
    "                                            \n\t"
    "vpermilpd  $0x5, %%ymm9,  %%ymm1            \n\t"
    "vmulpd           %%ymm7,  %%ymm9,  %%ymm9   \n\t"
    "vmulpd           %%ymm6,  %%ymm1,  %%ymm1   \n\t"
    "vaddsubpd        %%ymm1,  %%ymm9,  %%ymm9   \n\t"
    "                                            \n\t"
    "vpermilpd  $0x5, %%ymm8,  %%ymm0            \n\t"
    "vmulpd           %%ymm7,  %%ymm8,  %%ymm8   \n\t"
    "vmulpd           %%ymm6,  %%ymm0,  %%ymm0   \n\t"
    "vaddsubpd        %%ymm0,  %%ymm8,  %%ymm8   \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "movq         %5, %%rbx                      \n\t" // load address of beta 
    "vbroadcastsd    (%%rbx), %%ymm7             \n\t" // load beta_r and duplicate
    "vbroadcastsd   8(%%rbx), %%ymm6             \n\t" // load beta_i and duplicate
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "movq                %7, %%rsi               \n\t" // load rs_c
    "leaq        (,%%rsi,8), %%rsi               \n\t" // rsi = rs_c * sizeof(dcomplex)
    "leaq        (,%%rsi,2), %%rsi               \n\t"
    "leaq   (%%rcx,%%rsi,2), %%rdx               \n\t" // load address of c + 2*rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t" // determine if
    "                                            \n\t" //    c    % 32 == 0, AND
    "                                            \n\t" // 16*cs_c % 32 == 0, AND
    "                                            \n\t" //    rs_c      == 1
    "                                            \n\t" // ie: aligned, ldim aligned, and
    "                                            \n\t" // column-stored
    "                                            \n\t"
    "cmpq      $16, %%rsi                        \n\t" // set ZF if (16*rs_c) == 16.
    "sete           %%bl                         \n\t" // bl = ( ZF == 1 ? 1 : 0 );
    "testq     $31, %%rcx                        \n\t" // set ZF if c & 32 is zero.
    "setz           %%bh                         \n\t" // bh = ( ZF == 0 ? 1 : 0 );
    "testq     $31, %%rdi                        \n\t" // set ZF if (16*cs_c) & 32 is zero.
    "setz           %%al                         \n\t" // al = ( ZF == 0 ? 1 : 0 );
    "                                            \n\t" // and(bl,bh) followed by
    "                                            \n\t" // and(bh,al) will reveal result
    "                                            \n\t"
    "                                            \n\t" // now avoid loading C if beta == 0
    "                                            \n\t"
    "vxorpd    %%ymm0,  %%ymm0,  %%ymm0          \n\t" // set ymm0 to zero.
    "vucomisd  %%xmm0,  %%xmm7                   \n\t" // set ZF if beta_r == 0.
    "sete       %%r8b                            \n\t" // r8b = ( ZF == 1 ? 1 : 0 );
    "vucomisd  %%xmm0,  %%xmm6                   \n\t" // set ZF if beta_i == 0.
    "sete       %%r9b                            \n\t" // r9b = ( ZF == 1 ? 1 : 0 );
    "andb       %%r8b, %%r9b                     \n\t" // set ZF if r8b & r9b == 1.
    "jne      .ZBETAZERO%=                       \n\t" // if ZF = 0, jump to beta == 0 case
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t" // check if aligned/column-stored
    "andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
    "andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
    "jne     .ZCOLSTORED%=                       \n\t" // jump to column storage case
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".ZGENSTORED%=:                              \n\t"
    "                                            \n\t" // update c00:c30
    "                                            \n\t"
    "vmovupd    (%%rcx),       %%xmm0            \n\t" // load (c00,c10) into xmm0
    "vmovupd    (%%rcx,%%rsi), %%xmm2            \n\t" // load (c20,c30) into xmm2
    "vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:1],xmm2)
    "vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddpd           %%ymm15, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[2:3]
    "vmovupd          %%xmm0,  (%%rcx)           \n\t" // store (c00,c10)
    "vmovupd          %%xmm2,  (%%rcx,%%rsi)     \n\t" // store (c20,c30)
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c40:c70
    "                                            \n\t"
    "vmovupd    (%%rdx),       %%xmm0            \n\t" // load (c40,c50) into xmm0
    "vmovupd    (%%rdx,%%rsi), %%xmm2            \n\t" // load (c60,c70) into xmm2
    "vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:1],xmm2)
    "vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddpd           %%ymm14, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[2:3]
    "vmovupd          %%xmm0,  (%%rdx)           \n\t" // store (c40,c50)
    "vmovupd          %%xmm2,  (%%rdx,%%rsi)     \n\t" // store (c60,c70)
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c01:c31
    "                                            \n\t"
    "vmovupd    (%%rcx),       %%xmm0            \n\t" // load (c01,c11) into xmm0
    "vmovupd    (%%rcx,%%rsi), %%xmm2            \n\t" // load (c21,c31) into xmm2
    "vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:1],xmm2)
    "vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddpd           %%ymm13, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[2:3]
    "vmovupd          %%xmm0,  (%%rcx)           \n\t" // store (c01,c11)
    "vmovupd          %%xmm2,  (%%rcx,%%rsi)     \n\t" // store (c21,c31)
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c41:c71
    "                                            \n\t"
    "vmovupd    (%%rdx),       %%xmm0            \n\t" // load (c41,c51) into xmm0
    "vmovupd    (%%rdx,%%rsi), %%xmm2            \n\t" // load (c61,c71) into xmm2
    "vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:1],xmm2)
    "vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddpd           %%ymm12, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[2:3]
    "vmovupd          %%xmm0,  (%%rdx)           \n\t" // store (c41,c51)
    "vmovupd          %%xmm2,  (%%rdx,%%rsi)     \n\t" // store (c61,c71)
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c02:c32
    "                                            \n\t"
    "vmovupd    (%%rcx),       %%xmm0            \n\t" // load (c02,c12) into xmm0
    "vmovupd    (%%rcx,%%rsi), %%xmm2            \n\t" // load (c22,c32) into xmm2
    "vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:1],xmm2)
    "vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddpd           %%ymm11, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[2:3]
    "vmovupd          %%xmm0,  (%%rcx)           \n\t" // store (c02,c12)
    "vmovupd          %%xmm2,  (%%rcx,%%rsi)     \n\t" // store (c22,c32)
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c42:c72
    "                                            \n\t"
    "vmovupd    (%%rdx),       %%xmm0            \n\t" // load (c42,c52) into xmm0
    "vmovupd    (%%rdx,%%rsi), %%xmm2            \n\t" // load (c62,c72) into xmm2
    "vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:1],xmm2)
    "vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddpd           %%ymm10, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[2:3]
    "vmovupd          %%xmm0,  (%%rdx)           \n\t" // store (c42,c52)
    "vmovupd          %%xmm2,  (%%rdx,%%rsi)     \n\t" // store (c62,c72)
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c03:c33
    "                                            \n\t"
    "vmovupd    (%%rcx),       %%xmm0            \n\t" // load (c03,c13) into xmm0
    "vmovupd    (%%rcx,%%rsi), %%xmm2            \n\t" // load (c23,c33) into xmm2
    "vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:1],xmm2)
    "vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddpd           %%ymm9,  %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[2:3]
    "vmovupd          %%xmm0,  (%%rcx)           \n\t" // store (c03,c13)
    "vmovupd          %%xmm2,  (%%rcx,%%rsi)     \n\t" // store (c23,c33)
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c43:c73
    "                                            \n\t"
    "vmovupd    (%%rdx),       %%xmm0            \n\t" // load (c43,c53) into xmm0
    "vmovupd    (%%rdx,%%rsi), %%xmm2            \n\t" // load (c63,c73) into xmm2
    "vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:1],xmm2)
    "vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddpd           %%ymm8,  %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[2:3]
    "vmovupd          %%xmm0,  (%%rdx)           \n\t" // store (c43,c53)
    "vmovupd          %%xmm2,  (%%rdx,%%rsi)     \n\t" // store (c63,c73)
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "jmp    .ZDONE%=                             \n\t" // jump to end.
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".ZCOLSTORED%=:                              \n\t"
    "                                            \n\t" // update c00:c30
    "                                            \n\t"
    "vmovapd    (%%rcx),       %%ymm0            \n\t" // load c00:c30 into ymm0
    "vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddpd           %%ymm15, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vmovapd          %%ymm0,  (%%rcx)           \n\t" // store c00:c30
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c40:c70
    "                                            \n\t"
    "vmovapd    (%%rdx),       %%ymm0            \n\t" // load c40:c70 into ymm0
    "vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddpd           %%ymm14, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vmovapd          %%ymm0,  (%%rdx)           \n\t" // store c40:c70
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c01:c31
    "                                            \n\t"
    "vmovapd    (%%rcx),       %%ymm0            \n\t" // load c01:c31 into ymm0
    "vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddpd           %%ymm13, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vmovapd          %%ymm0,  (%%rcx)           \n\t" // store c01:c31
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c41:c71
    "                                            \n\t"
    "vmovapd    (%%rdx),       %%ymm0            \n\t" // load c41:c71 into ymm0
    "vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddpd           %%ymm12, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vmovapd          %%ymm0,  (%%rdx)           \n\t" // store c41:c71
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c02:c32
    "                                            \n\t"
    "vmovapd    (%%rcx),       %%ymm0            \n\t" // load c02:c32 into ymm0
    "vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddpd           %%ymm11, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vmovapd          %%ymm0,  (%%rcx)           \n\t" // store c02:c32
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c42:c72
    "                                            \n\t"
    "vmovapd    (%%rdx),       %%ymm0            \n\t" // load c42:c72 into ymm0
    "vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddpd           %%ymm10, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vmovapd          %%ymm0,  (%%rdx)           \n\t" // store c42:c72
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c03:c33
    "                                            \n\t"
    "vmovapd    (%%rcx),       %%ymm0            \n\t" // load c03:c33 into ymm0
    "vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddpd           %%ymm9,  %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vmovapd          %%ymm0,  (%%rcx)           \n\t" // store c03:c33
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c43:c73
    "                                            \n\t"
    "vmovapd    (%%rdx),       %%ymm0            \n\t" // load c43:c73 into ymm0
    "vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddpd           %%ymm8,  %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vmovapd          %%ymm0,  (%%rdx)           \n\t" // store c43:c73
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "jmp    .ZDONE%=                             \n\t" // jump to end.
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".ZBETAZERO%=:                               \n\t"
    "                                            \n\t" // check if aligned/column-stored
    "                                            \n\t" // check if aligned/column-stored
    "andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
    "andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
    "jne     .ZCOLSTORBZ%=                       \n\t" // jump to column storage case
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".ZGENSTORBZ%=:                              \n\t"
    "                                            \n\t" // update c00:c30
    "                                            \n\t"
    "vextractf128 $1, %%ymm15, %%xmm2            \n\t"
    "vmovupd          %%xmm15, (%%rcx)           \n\t" // store (c00,c10)
    "vmovupd          %%xmm2,  (%%rcx,%%rsi)     \n\t" // store (c20,c30)
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c40:c70
    "                                            \n\t"
    "vextractf128 $1, %%ymm14, %%xmm2            \n\t"
    "vmovupd          %%xmm14, (%%rdx)           \n\t" // store (c40,c50)
    "vmovupd          %%xmm2,  (%%rdx,%%rsi)     \n\t" // store (c60,c70)
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c01:c31
    "                                            \n\t"
    "vextractf128 $1, %%ymm13, %%xmm2            \n\t"
    "vmovupd          %%xmm13, (%%rcx)           \n\t" // store (c01,c11)
    "vmovupd          %%xmm2,  (%%rcx,%%rsi)     \n\t" // store (c21,c31)
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c41:c71
    "                                            \n\t"
    "vextractf128 $1, %%ymm12, %%xmm2            \n\t"
    "vmovupd          %%xmm12, (%%rdx)           \n\t" // store (c41,c51)
    "vmovupd          %%xmm2,  (%%rdx,%%rsi)     \n\t" // store (c61,c71)
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c02:c32
    "                                            \n\t"
    "vextractf128 $1, %%ymm11, %%xmm2            \n\t"
    "vmovupd          %%xmm11, (%%rcx)           \n\t" // store (c02,c12)
    "vmovupd          %%xmm2,  (%%rcx,%%rsi)     \n\t" // store (c22,c32)
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c42:c72
    "                                            \n\t"
    "vextractf128 $1, %%ymm10, %%xmm2            \n\t"
    "vmovupd          %%xmm10, (%%rdx)           \n\t" // store (c42,c52)
    "vmovupd          %%xmm2,  (%%rdx,%%rsi)     \n\t" // store (c62,c72)
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c03:c33
    "                                            \n\t"
    "vextractf128 $1, %%ymm9,  %%xmm2            \n\t"
    "vmovupd          %%xmm9,  (%%rcx)           \n\t" // store (c03,c13)
    "vmovupd          %%xmm2,  (%%rcx,%%rsi)     \n\t" // store (c23,c33)
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c43:c73
    "                                            \n\t"
    "vextractf128 $1, %%ymm8,  %%xmm2            \n\t"
    "vmovupd          %%xmm8,  (%%rdx)           \n\t" // store (c43,c53)
    "vmovupd          %%xmm2,  (%%rdx,%%rsi)     \n\t" // store (c63,c73)
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "jmp    .ZDONE%=                             \n\t" // jump to end.
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".ZCOLSTORBZ%=:                              \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd          %%ymm15, (%%rcx)           \n\t" // store c00:c30
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "vmovapd          %%ymm14, (%%rdx)           \n\t" // store c40:c70
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "vmovapd          %%ymm13, (%%rcx)           \n\t" // store c01:c31
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "vmovapd          %%ymm12, (%%rdx)           \n\t" // store c41:c71
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "vmovapd          %%ymm11, (%%rcx)           \n\t" // store c02:c32
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "vmovapd          %%ymm10, (%%rdx)           \n\t" // store c42:c72
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "vmovapd          %%ymm9,  (%%rcx)           \n\t" // store c03:c33
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "vmovapd          %%ymm8,  (%%rdx)           \n\t" // store c43:c73
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".ZDONE%=:                                   \n\t"
    "                                            \n\t"

    : // output operands (none)
    : // input operands
      "m" (k_iter), // 0
      "m" (k_left), // 1
      "m" (A),      // 2
      "m" (B),      // 3
      "m" (pAlpha),  // 4
      "m" (pBeta),   // 5
      "m" (C),      // 6
      "m" (rs_c),   // 7
      "m" (cs_c)/*,   // 8
      "m" (b_next), // 9
      "m" (a_next)*/  // 10
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi", 
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3",
      "xmm4", "xmm5", "xmm6", "xmm7",
      "xmm8", "xmm9", "xmm10", "xmm11",
      "xmm12", "xmm13", "xmm14", "xmm15",
      "memory"
    );
}

} } // end namespace blas, hpc

#endif // end HPC_BLAS_GEMM_AVX_ZUGEMM_4X4_HPP
