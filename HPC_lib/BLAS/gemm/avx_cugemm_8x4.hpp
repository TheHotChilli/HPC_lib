#ifndef HPC_BLAS_GEMM_AVX_CUGEMM_8X4_HPP
#define HPC_BLAS_GEMM_AVX_CUGEMM_8X4_HPP

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
#include <complex>

namespace hpc { namespace blas {


void
cugemm_asm_8x4(std::size_t kc, std::complex<float> alpha,
               const std::complex<float> *A, const std::complex<float> *B,
               std::complex<float> beta,
               std::complex<float> *C,
               std::ptrdiff_t incRowC, std::ptrdiff_t incColC,
               const std::complex<float> *,
               const std::complex<float> *b_next)
{
    std::int64_t k_iter = kc / 4;
    std::int64_t k_left = kc % 4;

    std::int64_t rs_c = incRowC;
    std::int64_t cs_c = incColC;

    float *pAlpha  = (float *)&alpha;
    float *pBeta   = (float *)&beta;

    __asm__ volatile
    (
    "                                            \n\t"
    "                                            \n\t"
    "movq                %2, %%rax               \n\t" // load address of a.
    "movq                %3, %%rbx               \n\t" // load address of b.
    "movq                %9, %%r15               \n\t" // load address of b_next.
    //"movq               %10, %%r14               \n\t" // load address of a_next.
    "addq          $-4 * 64, %%r15               \n\t"
    "                                            \n\t"
    "vmovaps        0 * 32(%%rax), %%ymm0        \n\t" // initialize loop by pre-loading
    "vmovsldup      0 * 32(%%rbx), %%ymm2        \n\t"
    "vpermilps     $0x4e, %%ymm2,  %%ymm3        \n\t"
    "                                            \n\t"
    "movq                %6, %%rcx               \n\t" // load address of c
    "movq                %8, %%rdi               \n\t" // load cs_c
    "leaq        (,%%rdi,8), %%rdi               \n\t" // cs_c *= sizeof(scomplex)
    "leaq   (%%rcx,%%rdi,2), %%r10               \n\t" // load address of c + 2*cs_c;
    "                                            \n\t"
    "prefetcht0   3 * 8(%%rcx)                   \n\t" // prefetch c + 0*cs_c
    "prefetcht0   3 * 8(%%rcx,%%rdi)             \n\t" // prefetch c + 1*cs_c
    "prefetcht0   3 * 8(%%r10)                   \n\t" // prefetch c + 2*cs_c
    "prefetcht0   3 * 8(%%r10,%%rdi)             \n\t" // prefetch c + 3*cs_c
    "                                            \n\t"
    "vxorps    %%ymm8,  %%ymm8,  %%ymm8          \n\t"
    "vxorps    %%ymm9,  %%ymm9,  %%ymm9          \n\t"
    "vxorps    %%ymm10, %%ymm10, %%ymm10         \n\t"
    "vxorps    %%ymm11, %%ymm11, %%ymm11         \n\t"
    "vxorps    %%ymm12, %%ymm12, %%ymm12         \n\t"
    "vxorps    %%ymm13, %%ymm13, %%ymm13         \n\t"
    "vxorps    %%ymm14, %%ymm14, %%ymm14         \n\t"
    "vxorps    %%ymm15, %%ymm15, %%ymm15         \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "movq      %0, %%rsi                         \n\t" // i = k_iter;
    "testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
    "je     .CCONSIDKLEFT%=                      \n\t" // if i == 0, jump to code that
    "                                            \n\t" // contains the k_left loop.
    "                                            \n\t"
    "                                            \n\t"
    ".CLOOPKITER%=:                              \n\t" // MAIN LOOP
    "                                            \n\t"
    "addq         $4 * 4 * 8,  %%r15             \n\t" // b_next += 4*4 (unroll x nr)
    "                                            \n\t"
    "                                            \n\t" // iteration 0
    "prefetcht0     8 * 32(%%rax)                \n\t"
    "vmovaps        1 * 32(%%rax),      %%ymm1   \n\t"
    "vmulps           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
    "vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
    "vmulps           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
    "vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
    "vaddps           %%ymm6,  %%ymm15, %%ymm15  \n\t"
    "vaddps           %%ymm7,  %%ymm13, %%ymm13  \n\t"
    "                                            \n\t"
    "vmulps           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
    "vmovshdup      0 * 32(%%rbx),      %%ymm2   \n\t"
    "vmulps           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
    "vpermilps $0x4e, %%ymm2,  %%ymm3            \n\t"
    "vaddps           %%ymm6,  %%ymm14, %%ymm14  \n\t"
    "vaddps           %%ymm7,  %%ymm12, %%ymm12  \n\t"
    "                                            \n\t"
    "vmulps           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
    "vmulps           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
    "vpermilps $0xb1, %%ymm0,  %%ymm0            \n\t"
    "vaddps           %%ymm6,  %%ymm11, %%ymm11  \n\t"
    "vaddps           %%ymm7,  %%ymm9,  %%ymm9   \n\t"
    "                                            \n\t"
    "vmulps           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
    "vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
    "vmulps           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
    "vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
    "vaddps           %%ymm6,  %%ymm10, %%ymm10  \n\t"
    "vaddps           %%ymm7,  %%ymm8,  %%ymm8   \n\t"
    "prefetcht0   0 * 32(%%r15)                  \n\t" // prefetch b_next[0*4]
    "                                            \n\t"
    "vpermilps $0xb1, %%ymm1,  %%ymm1            \n\t"
    "vmulps           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
    "vmulps           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
    "vaddsubps        %%ymm6,  %%ymm15, %%ymm15  \n\t"
    "vaddsubps        %%ymm7,  %%ymm13, %%ymm13  \n\t"
    "                                            \n\t"
    "vmulps           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
    "vmovsldup      1 * 32(%%rbx),      %%ymm2   \n\t"
    "vmulps           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
    "vpermilps $0x4e, %%ymm2,  %%ymm3            \n\t"
    "vaddsubps        %%ymm6,  %%ymm14, %%ymm14  \n\t"
    "vaddsubps        %%ymm7,  %%ymm12, %%ymm12  \n\t"
    "                                            \n\t"
    "vmulps           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
    "vmulps           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
    "vmovaps        2 * 32(%%rax),      %%ymm0   \n\t"
    "vaddsubps        %%ymm6,  %%ymm11, %%ymm11  \n\t"
    "vaddsubps        %%ymm7,  %%ymm9,  %%ymm9   \n\t"
    "                                            \n\t"
    "vmulps           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
    "vmulps           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
    "vaddsubps        %%ymm6,  %%ymm10, %%ymm10  \n\t"
    "vaddsubps        %%ymm7,  %%ymm8,  %%ymm8   \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t" // iteration 1
    "prefetcht0    10 * 32(%%rax)                \n\t"
    "vmovaps        3 * 32(%%rax),      %%ymm1   \n\t"
    "vmulps           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
    "vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
    "vmulps           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
    "vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
    "vaddps           %%ymm6,  %%ymm15, %%ymm15  \n\t"
    "vaddps           %%ymm7,  %%ymm13, %%ymm13  \n\t"
    "                                            \n\t"
    "vmulps           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
    "vmovshdup      1 * 32(%%rbx),      %%ymm2   \n\t"
    "vmulps           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
    "vpermilps $0x4e, %%ymm2,  %%ymm3            \n\t"
    "vaddps           %%ymm6,  %%ymm14, %%ymm14  \n\t"
    "vaddps           %%ymm7,  %%ymm12, %%ymm12  \n\t"
    "                                            \n\t"
    "vmulps           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
    "vmulps           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
    "vpermilps $0xb1, %%ymm0,  %%ymm0            \n\t"
    "vaddps           %%ymm6,  %%ymm11, %%ymm11  \n\t"
    "vaddps           %%ymm7,  %%ymm9,  %%ymm9   \n\t"
    "                                            \n\t"
    "vmulps           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
    "vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
    "vmulps           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
    "vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
    "vaddps           %%ymm6,  %%ymm10, %%ymm10  \n\t"
    "vaddps           %%ymm7,  %%ymm8,  %%ymm8   \n\t"
    "                                            \n\t"
    "vpermilps $0xb1, %%ymm1,  %%ymm1            \n\t"
    "vmulps           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
    "vmulps           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
    "vaddsubps        %%ymm6,  %%ymm15, %%ymm15  \n\t"
    "vaddsubps        %%ymm7,  %%ymm13, %%ymm13  \n\t"
    "                                            \n\t"
    "vmulps           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
    "vmovsldup      2 * 32(%%rbx),      %%ymm2   \n\t"
    "vmulps           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
    "vpermilps $0x4e, %%ymm2,  %%ymm3            \n\t"
    "vaddsubps        %%ymm6,  %%ymm14, %%ymm14  \n\t"
    "vaddsubps        %%ymm7,  %%ymm12, %%ymm12  \n\t"
    "                                            \n\t"
    "vmulps           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
    "vmulps           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
    "vmovaps        4 * 32(%%rax),      %%ymm0   \n\t"
    "vaddsubps        %%ymm6,  %%ymm11, %%ymm11  \n\t"
    "vaddsubps        %%ymm7,  %%ymm9,  %%ymm9   \n\t"
    "                                            \n\t"
    "vmulps           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
    "vmulps           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
    "vaddsubps        %%ymm6,  %%ymm10, %%ymm10  \n\t"
    "vaddsubps        %%ymm7,  %%ymm8,  %%ymm8   \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t" // iteration 2
    "prefetcht0    12 * 32(%%rax)                \n\t"
    "vmovaps        5 * 32(%%rax),      %%ymm1   \n\t"
    "vmulps           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
    "vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
    "vmulps           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
    "vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
    "vaddps           %%ymm6,  %%ymm15, %%ymm15  \n\t"
    "vaddps           %%ymm7,  %%ymm13, %%ymm13  \n\t"
    "                                            \n\t"
    "vmulps           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
    "vmovshdup      2 * 32(%%rbx),      %%ymm2   \n\t"
    "vmulps           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
    "vpermilps $0x4e, %%ymm2,  %%ymm3            \n\t"
    "vaddps           %%ymm6,  %%ymm14, %%ymm14  \n\t"
    "vaddps           %%ymm7,  %%ymm12, %%ymm12  \n\t"
    "                                            \n\t"
    "vmulps           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
    "vmulps           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
    "vpermilps $0xb1, %%ymm0,  %%ymm0            \n\t"
    "vaddps           %%ymm6,  %%ymm11, %%ymm11  \n\t"
    "vaddps           %%ymm7,  %%ymm9,  %%ymm9   \n\t"
    "                                            \n\t"
    "vmulps           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
    "vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
    "vmulps           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
    "vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
    "vaddps           %%ymm6,  %%ymm10, %%ymm10  \n\t"
    "vaddps           %%ymm7,  %%ymm8,  %%ymm8   \n\t"
    "prefetcht0   2 * 32(%%r15)                  \n\t" // prefetch b_next[2*4]
    "                                            \n\t"
    "vpermilps $0xb1, %%ymm1,  %%ymm1            \n\t"
    "vmulps           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
    "vmulps           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
    "vaddsubps        %%ymm6,  %%ymm15, %%ymm15  \n\t"
    "vaddsubps        %%ymm7,  %%ymm13, %%ymm13  \n\t"
    "                                            \n\t"
    "vmulps           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
    "vmovsldup      3 * 32(%%rbx),      %%ymm2   \n\t"
    "vmulps           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
    "vpermilps $0x4e, %%ymm2,  %%ymm3            \n\t"
    "vaddsubps        %%ymm6,  %%ymm14, %%ymm14  \n\t"
    "vaddsubps        %%ymm7,  %%ymm12, %%ymm12  \n\t"
    "                                            \n\t"
    "vmulps           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
    "vmulps           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
    "vmovaps        6 * 32(%%rax),      %%ymm0   \n\t"
    "vaddsubps        %%ymm6,  %%ymm11, %%ymm11  \n\t"
    "vaddsubps        %%ymm7,  %%ymm9,  %%ymm9   \n\t"
    "                                            \n\t"
    "vmulps           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
    "vmulps           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
    "vaddsubps        %%ymm6,  %%ymm10, %%ymm10  \n\t"
    "vaddsubps        %%ymm7,  %%ymm8,  %%ymm8   \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t" // iteration 3
    "prefetcht0    14 * 32(%%rax)                \n\t"
    "vmovaps        7 * 32(%%rax),      %%ymm1   \n\t"
    "vmulps           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
    "vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
    "vmulps           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
    "vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
    "vaddps           %%ymm6,  %%ymm15, %%ymm15  \n\t"
    "vaddps           %%ymm7,  %%ymm13, %%ymm13  \n\t"
    "                                            \n\t"
    "vmulps           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
    "vmovshdup      3 * 32(%%rbx),      %%ymm2   \n\t"
    "vmulps           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
    "vpermilps $0x4e, %%ymm2,  %%ymm3            \n\t"
    "vaddps           %%ymm6,  %%ymm14, %%ymm14  \n\t"
    "vaddps           %%ymm7,  %%ymm12, %%ymm12  \n\t"
    "                                            \n\t"
    "vmulps           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
    "vmulps           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
    "vpermilps $0xb1, %%ymm0,  %%ymm0            \n\t"
    "vaddps           %%ymm6,  %%ymm11, %%ymm11  \n\t"
    "vaddps           %%ymm7,  %%ymm9,  %%ymm9   \n\t"
    "                                            \n\t"
    "vmulps           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
    "vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
    "vmulps           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
    "vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
    "vaddps           %%ymm6,  %%ymm10, %%ymm10  \n\t"
    "vaddps           %%ymm7,  %%ymm8,  %%ymm8   \n\t"
    "                                            \n\t"
    "vpermilps $0xb1, %%ymm1,  %%ymm1            \n\t"
    "vmulps           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
    "vmulps           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
    "vaddsubps        %%ymm6,  %%ymm15, %%ymm15  \n\t"
    "vaddsubps        %%ymm7,  %%ymm13, %%ymm13  \n\t"
    "                                            \n\t"
    "vmulps           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
    "vmovsldup      4 * 32(%%rbx),      %%ymm2   \n\t"
    "vmulps           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
    "vpermilps $0x4e, %%ymm2,  %%ymm3            \n\t"
    "vaddsubps        %%ymm6,  %%ymm14, %%ymm14  \n\t"
    "vaddsubps        %%ymm7,  %%ymm12, %%ymm12  \n\t"
    "                                            \n\t"
    "vmulps           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
    "vmulps           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
    "vmovaps        8 * 32(%%rax),      %%ymm0   \n\t"
    "vaddsubps        %%ymm6,  %%ymm11, %%ymm11  \n\t"
    "vaddsubps        %%ymm7,  %%ymm9,  %%ymm9   \n\t"
    "                                            \n\t"
    "vmulps           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
    "vmulps           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
    "vaddsubps        %%ymm6,  %%ymm10, %%ymm10  \n\t"
    "vaddsubps        %%ymm7,  %%ymm8,  %%ymm8   \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "addq          $8 * 4 * 8, %%rax             \n\t" // a += 8*4 (unroll x mr)
    "addq          $4 * 4 * 8, %%rbx             \n\t" // b += 4*4 (unroll x nr)
    "                                            \n\t"
    "                                            \n\t"
    "decq   %%rsi                                \n\t" // i -= 1;
    "jne    .CLOOPKITER%=                        \n\t" // iterate again if i != 0.
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".CCONSIDKLEFT%=:                            \n\t"
    "                                            \n\t"
    "movq      %1, %%rsi                         \n\t" // i = k_left;
    "testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
    "je     .CPOSTACCUM%=                        \n\t" // if i == 0, we're done; jump to end.
    "                                            \n\t" // else, we prepare to enter k_left loop.
    "                                            \n\t"
    "                                            \n\t"
    ".CLOOPKLEFT%=:                              \n\t" // EDGE LOOP
    "                                            \n\t"
    "                                            \n\t" // iteration 0
    "prefetcht0     8 * 32(%%rax)                \n\t"
    "vmovaps        1 * 32(%%rax),      %%ymm1   \n\t"
    "vmulps           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
    "vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
    "vmulps           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
    "vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
    "vaddps           %%ymm6,  %%ymm15, %%ymm15  \n\t"
    "vaddps           %%ymm7,  %%ymm13, %%ymm13  \n\t"
    "                                            \n\t"
    "vmulps           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
    "vmovshdup      0 * 32(%%rbx),      %%ymm2   \n\t"
    "vmulps           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
    "vpermilps $0x4e, %%ymm2,  %%ymm3            \n\t"
    "vaddps           %%ymm6,  %%ymm14, %%ymm14  \n\t"
    "vaddps           %%ymm7,  %%ymm12, %%ymm12  \n\t"
    "                                            \n\t"
    "vmulps           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
    "vmulps           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
    "vpermilps $0xb1, %%ymm0,  %%ymm0            \n\t"
    "vaddps           %%ymm6,  %%ymm11, %%ymm11  \n\t"
    "vaddps           %%ymm7,  %%ymm9,  %%ymm9   \n\t"
    "                                            \n\t"
    "vmulps           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
    "vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
    "vmulps           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
    "vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
    "vaddps           %%ymm6,  %%ymm10, %%ymm10  \n\t"
    "vaddps           %%ymm7,  %%ymm8,  %%ymm8   \n\t"
    "                                            \n\t"
    "vpermilps $0xb1, %%ymm1,  %%ymm1            \n\t"
    "vmulps           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
    "vmulps           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
    "vaddsubps        %%ymm6,  %%ymm15, %%ymm15  \n\t"
    "vaddsubps        %%ymm7,  %%ymm13, %%ymm13  \n\t"
    "                                            \n\t"
    "vmulps           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
    "vmovsldup      1 * 32(%%rbx),      %%ymm2   \n\t"
    "vmulps           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
    "vpermilps $0x4e, %%ymm2,  %%ymm3            \n\t"
    "vaddsubps        %%ymm6,  %%ymm14, %%ymm14  \n\t"
    "vaddsubps        %%ymm7,  %%ymm12, %%ymm12  \n\t"
    "                                            \n\t"
    "vmulps           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
    "vmulps           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
    "vmovaps        2 * 32(%%rax),      %%ymm0   \n\t"
    "vaddsubps        %%ymm6,  %%ymm11, %%ymm11  \n\t"
    "vaddsubps        %%ymm7,  %%ymm9,  %%ymm9   \n\t"
    "                                            \n\t"
    "vmulps           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
    "vmulps           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
    "vaddsubps        %%ymm6,  %%ymm10, %%ymm10  \n\t"
    "vaddsubps        %%ymm7,  %%ymm8,  %%ymm8   \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "addq          $8 * 1 * 8, %%rax             \n\t" // a += 8 (1 x mr)
    "addq          $4 * 1 * 8, %%rbx             \n\t" // b += 4 (1 x nr)
    "                                            \n\t"
    "                                            \n\t"
    "decq   %%rsi                                \n\t" // i -= 1;
    "jne    .CLOOPKLEFT%=                        \n\t" // iterate again if i != 0.
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".CPOSTACCUM%=:                              \n\t"
    "                                            \n\t"
    "                                            \n\t" // ymm15:  ymm13:  ymm11:  ymm9:
    "                                            \n\t" // ( ab00  ( ab01  ( ab02  ( ab03 
    "                                            \n\t" //   ab10    ab11    ab12    ab13 
    "                                            \n\t" //   ab21    ab20    ab23    ab22 
    "                                            \n\t" //   ab31    ab30    ab33    ab32 
    "                                            \n\t" //   ab42    ab43    ab40    ab41 
    "                                            \n\t" //   ab52    ab53    ab50    ab51 
    "                                            \n\t" //   ab63    ab62    ab61    ab60 
    "                                            \n\t" //   ab73 )  ab72 )  ab71 )  ab70 )
    "                                            \n\t"
    "                                            \n\t" // ymm14:  ymm12:  ymm10:  ymm8:
    "                                            \n\t" // ( ab80  ( ab81  ( ab82  ( ab83 
    "                                            \n\t" //   ab90    ab91    ab92    ab93 
    "                                            \n\t" //   aba1    aba0    aba3    aba2 
    "                                            \n\t" //   abb1    abb0    abb3    abb2 
    "                                            \n\t" //   abc2    abc3    abc0    abc1 
    "                                            \n\t" //   abd2    abd3    abd0    abd1 
    "                                            \n\t" //   abe3    abe2    abe1    abe0 
    "                                            \n\t" //   abf3    abf2    abf1    abf0 )
    "                                            \n\t"
    "vmovaps          %%ymm15, %%ymm7            \n\t"
    "vshufps   $0xe4, %%ymm13, %%ymm15, %%ymm15  \n\t"
    "vshufps   $0xe4, %%ymm7,  %%ymm13, %%ymm13  \n\t"
    "                                            \n\t"
    "vmovaps          %%ymm11, %%ymm7            \n\t"
    "vshufps   $0xe4, %%ymm9,  %%ymm11, %%ymm11  \n\t"
    "vshufps   $0xe4, %%ymm7,  %%ymm9,  %%ymm9   \n\t"
    "                                            \n\t"
    "vmovaps          %%ymm14, %%ymm7            \n\t"
    "vshufps   $0xe4, %%ymm12, %%ymm14, %%ymm14  \n\t"
    "vshufps   $0xe4, %%ymm7,  %%ymm12, %%ymm12  \n\t"
    "                                            \n\t"
    "vmovaps          %%ymm10, %%ymm7            \n\t"
    "vshufps   $0xe4, %%ymm8,  %%ymm10, %%ymm10  \n\t"
    "vshufps   $0xe4, %%ymm7,  %%ymm8,  %%ymm8   \n\t"
    "                                            \n\t"
    "                                            \n\t" // ymm15:  ymm13:  ymm11:  ymm9:
    "                                            \n\t" // ( ab00  ( ab01  ( ab02  ( ab03 
    "                                            \n\t" //   ab10    ab11    ab12    ab13 
    "                                            \n\t" //   ab20    ab21    ab22    ab23 
    "                                            \n\t" //   ab30    ab31    ab32    ab33 
    "                                            \n\t" //   ab42    ab43    ab40    ab41 
    "                                            \n\t" //   ab52    ab53    ab50    ab51 
    "                                            \n\t" //   ab62    ab63    ab60    ab61 
    "                                            \n\t" //   ab72 )  ab73 )  ab70 )  ab71 )
    "                                            \n\t"
    "                                            \n\t" // ymm14:  ymm12:  ymm10:  ymm8:
    "                                            \n\t" // ( ab80  ( ab81  ( ab82  ( ab83 
    "                                            \n\t" //   ab90    ab91    ab92    ab93 
    "                                            \n\t" //   aba0    aba1    aba2    aba3 
    "                                            \n\t" //   abb0    abb1    abb2    abb3 
    "                                            \n\t" //   abc2    abc3    abc0    abc1 
    "                                            \n\t" //   abd2    abd3    abd0    abd1 
    "                                            \n\t" //   abe2    abe3    abe0    abe1 
    "                                            \n\t" //   abf2 )  abf3 )  abf0 )  abf1 )
    "                                            \n\t"
    "vmovaps           %%ymm15, %%ymm7           \n\t"
    "vperm2f128 $0x12, %%ymm15, %%ymm11, %%ymm15 \n\t"
    "vperm2f128 $0x30, %%ymm7,  %%ymm11, %%ymm11 \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm13, %%ymm7           \n\t"
    "vperm2f128 $0x12, %%ymm13, %%ymm9,  %%ymm13 \n\t"
    "vperm2f128 $0x30, %%ymm7,  %%ymm9,  %%ymm9  \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm14, %%ymm7           \n\t"
    "vperm2f128 $0x12, %%ymm14, %%ymm10, %%ymm14 \n\t"
    "vperm2f128 $0x30, %%ymm7,  %%ymm10, %%ymm10 \n\t"
    "                                            \n\t"
    "vmovaps           %%ymm12, %%ymm7           \n\t"
    "vperm2f128 $0x12, %%ymm12, %%ymm8,  %%ymm12 \n\t"
    "vperm2f128 $0x30, %%ymm7,  %%ymm8,  %%ymm8  \n\t"
    "                                            \n\t"
    "                                            \n\t" // ymm15:  ymm13:  ymm11:  ymm9:
    "                                            \n\t" // ( ab00  ( ab01  ( ab02  ( ab03 
    "                                            \n\t" //   ab10    ab11    ab12    ab13 
    "                                            \n\t" //   ab20    ab21    ab22    ab23 
    "                                            \n\t" //   ab30    ab31    ab32    ab33 
    "                                            \n\t" //   ab40    ab41    ab42    ab43 
    "                                            \n\t" //   ab50    ab51    ab52    ab53 
    "                                            \n\t" //   ab60    ab61    ab62    ab63 
    "                                            \n\t" //   ab70 )  ab71 )  ab72 )  ab73 )
    "                                            \n\t"
    "                                            \n\t" // ymm14:  ymm12:  ymm10:  ymm8:
    "                                            \n\t" // ( ab80  ( ab81  ( ab82  ( ab83 
    "                                            \n\t" //   ab90    ab91    ab92    ab93 
    "                                            \n\t" //   aba0    aba1    aba2    aba3 
    "                                            \n\t" //   abb0    abb1    abb2    abb3 
    "                                            \n\t" //   abc0    abc1    abc2    abc3 
    "                                            \n\t" //   abd0    abd1    abd2    abd3 
    "                                            \n\t" //   abe0    abe1    abe2    abe3 
    "                                            \n\t" //   abf0 )  abf1 )  abf2 )  abf3 )
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t" // scale by alpha
    "                                            \n\t"
    "movq         %4, %%rax                      \n\t" // load address of alpha
    "vbroadcastss    (%%rax), %%ymm7             \n\t" // load alpha_r and duplicate
    "vbroadcastss   4(%%rax), %%ymm6             \n\t" // load alpha_i and duplicate
    "                                            \n\t"
    "vpermilps $0xb1, %%ymm15, %%ymm3            \n\t"
    "vmulps           %%ymm7,  %%ymm15, %%ymm15  \n\t"
    "vmulps           %%ymm6,  %%ymm3,  %%ymm3   \n\t"
    "vaddsubps        %%ymm3,  %%ymm15, %%ymm15  \n\t"
    "                                            \n\t"
    "vpermilps $0xb1, %%ymm14, %%ymm2            \n\t"
    "vmulps           %%ymm7,  %%ymm14, %%ymm14  \n\t"
    "vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubps        %%ymm2,  %%ymm14, %%ymm14  \n\t"
    "                                            \n\t"
    "vpermilps $0xb1, %%ymm13, %%ymm1            \n\t"
    "vmulps           %%ymm7,  %%ymm13, %%ymm13  \n\t"
    "vmulps           %%ymm6,  %%ymm1,  %%ymm1   \n\t"
    "vaddsubps        %%ymm1,  %%ymm13, %%ymm13  \n\t"
    "                                            \n\t"
    "vpermilps $0xb1, %%ymm12, %%ymm0            \n\t"
    "vmulps           %%ymm7,  %%ymm12, %%ymm12  \n\t"
    "vmulps           %%ymm6,  %%ymm0,  %%ymm0   \n\t"
    "vaddsubps        %%ymm0,  %%ymm12, %%ymm12  \n\t"
    "                                            \n\t"
    "vpermilps $0xb1, %%ymm11, %%ymm3            \n\t"
    "vmulps           %%ymm7,  %%ymm11, %%ymm11  \n\t"
    "vmulps           %%ymm6,  %%ymm3,  %%ymm3   \n\t"
    "vaddsubps        %%ymm3,  %%ymm11, %%ymm11  \n\t"
    "                                            \n\t"
    "vpermilps $0xb1, %%ymm10, %%ymm2            \n\t"
    "vmulps           %%ymm7,  %%ymm10, %%ymm10  \n\t"
    "vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubps        %%ymm2,  %%ymm10, %%ymm10  \n\t"
    "                                            \n\t"
    "vpermilps $0xb1, %%ymm9,  %%ymm1            \n\t"
    "vmulps           %%ymm7,  %%ymm9,  %%ymm9   \n\t"
    "vmulps           %%ymm6,  %%ymm1,  %%ymm1   \n\t"
    "vaddsubps        %%ymm1,  %%ymm9,  %%ymm9   \n\t"
    "                                            \n\t"
    "vpermilps $0xb1, %%ymm8,  %%ymm0            \n\t"
    "vmulps           %%ymm7,  %%ymm8,  %%ymm8   \n\t"
    "vmulps           %%ymm6,  %%ymm0,  %%ymm0   \n\t"
    "vaddsubps        %%ymm0,  %%ymm8,  %%ymm8   \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "movq         %5, %%rbx                      \n\t" // load address of beta 
    "vbroadcastss    (%%rbx), %%ymm7             \n\t" // load beta_r and duplicate
    "vbroadcastss   4(%%rbx), %%ymm6             \n\t" // load beta_i and duplicate
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "movq                %7, %%rsi               \n\t" // load rs_c
    "leaq        (,%%rsi,8), %%rsi               \n\t" // rsi = rs_c * sizeof(scomplex)
    "                                            \n\t"
    "leaq   (%%rcx,%%rsi,4), %%rdx               \n\t" // load address of c + 4*rs_c;
    "                                            \n\t"
    "leaq        (,%%rsi,2), %%r12               \n\t" // r12 = 2*rs_c;
    "leaq   (%%r12,%%rsi,1), %%r13               \n\t" // r13 = 3*rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t" // determine if
    "                                            \n\t" //    c    % 32 == 0, AND
    "                                            \n\t" //  8*cs_c % 32 == 0, AND
    "                                            \n\t" //    rs_c      == 1
    "                                            \n\t" // ie: aligned, ldim aligned, and
    "                                            \n\t" // column-stored
    "                                            \n\t"
    "cmpq       $8, %%rsi                        \n\t" // set ZF if (8*rs_c) == 8.
    "sete           %%bl                         \n\t" // bl = ( ZF == 1 ? 1 : 0 );
    "testq     $31, %%rcx                        \n\t" // set ZF if c & 32 is zero.
    "setz           %%bh                         \n\t" // bh = ( ZF == 0 ? 1 : 0 );
    "testq     $31, %%rdi                        \n\t" // set ZF if (8*cs_c) & 32 is zero.
    "setz           %%al                         \n\t" // al = ( ZF == 0 ? 1 : 0 );
    "                                            \n\t" // and(bl,bh) followed by
    "                                            \n\t" // and(bh,al) will reveal result
    "                                            \n\t"
    "                                            \n\t" // now avoid loading C if beta == 0
    "                                            \n\t"
    "vxorps    %%ymm0,  %%ymm0,  %%ymm0          \n\t" // set ymm0 to zero.
    "vucomiss  %%xmm0,  %%xmm7                   \n\t" // set ZF if beta_r == 0.
    "sete       %%r8b                            \n\t" // r8b = ( ZF == 1 ? 1 : 0 );
    "vucomiss  %%xmm0,  %%xmm6                   \n\t" // set ZF if beta_i == 0.
    "sete       %%r9b                            \n\t" // r9b = ( ZF == 1 ? 1 : 0 );
    "andb       %%r8b, %%r9b                     \n\t" // set ZF if r8b & r9b == 1.
    "jne      .CBETAZERO%=                       \n\t" // if ZF = 0, jump to beta == 0 case
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t" // check if aligned/column-stored
    "andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
    "andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
    "jne     .CCOLSTORED%=                       \n\t" // jump to column storage case
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".CGENSTORED%=:                              \n\t"
    "                                            \n\t"
    "                                            \n\t" // update c00:c70
    "                                            \n\t"
    "vmovlpd    (%%rcx),       %%xmm0,  %%xmm0   \n\t" // load (c00,10) into xmm0[0:1]
    "vmovhpd    (%%rcx,%%rsi), %%xmm0,  %%xmm0   \n\t" // load (c20,30) into xmm0[2:3]
    "vmovlpd    (%%rcx,%%r12), %%xmm2,  %%xmm2   \n\t" // load (c40,50) into xmm2[0:1]
    "vmovhpd    (%%rcx,%%r13), %%xmm2,  %%xmm2   \n\t" // load (c60,70) into xmm2[2:3]
    "vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:3],xmm2)
    "vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddps           %%ymm15, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[4:7]
    "vmovlpd          %%xmm0,  (%%rcx)           \n\t" // store (c00,c10)
    "vmovhpd          %%xmm0,  (%%rcx,%%rsi)     \n\t" // store (c20,c30)
    "vmovlpd          %%xmm2,  (%%rcx,%%r12)     \n\t" // store (c40,c50)
    "vmovhpd          %%xmm2,  (%%rcx,%%r13)     \n\t" // store (c60,c70)
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c80:cf0
    "                                            \n\t"
    "vmovlpd    (%%rdx),       %%xmm0,  %%xmm0   \n\t" // load (c80,90) into xmm0[0:1]
    "vmovhpd    (%%rdx,%%rsi), %%xmm0,  %%xmm0   \n\t" // load (ca0,b0) into xmm0[2:3]
    "vmovlpd    (%%rdx,%%r12), %%xmm2,  %%xmm2   \n\t" // load (cc0,d0) into xmm2[0:1]
    "vmovhpd    (%%rdx,%%r13), %%xmm2,  %%xmm2   \n\t" // load (ce0,f0) into xmm2[2:3]
    "vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:3],xmm2)
    "vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddps           %%ymm14, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[4:7]
    "vmovlpd          %%xmm0,  (%%rdx)           \n\t" // store (c80,c90)
    "vmovhpd          %%xmm0,  (%%rdx,%%rsi)     \n\t" // store (ca0,cb0)
    "vmovlpd          %%xmm2,  (%%rdx,%%r12)     \n\t" // store (cc0,cd0)
    "vmovhpd          %%xmm2,  (%%rdx,%%r13)     \n\t" // store (ce0,cf0)
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c01:c71
    "                                            \n\t"
    "vmovlpd    (%%rcx),       %%xmm0,  %%xmm0   \n\t" // load (c01,11) into xmm0[0:1]
    "vmovhpd    (%%rcx,%%rsi), %%xmm0,  %%xmm0   \n\t" // load (c21,31) into xmm0[2:3]
    "vmovlpd    (%%rcx,%%r12), %%xmm2,  %%xmm2   \n\t" // load (c41,51) into xmm2[0:1]
    "vmovhpd    (%%rcx,%%r13), %%xmm2,  %%xmm2   \n\t" // load (c61,71) into xmm2[2:3]
    "vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:3],xmm2)
    "vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddps           %%ymm13, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[4:7]
    "vmovlpd          %%xmm0,  (%%rcx)           \n\t" // store (c01,c11)
    "vmovhpd          %%xmm0,  (%%rcx,%%rsi)     \n\t" // store (c21,c31)
    "vmovlpd          %%xmm2,  (%%rcx,%%r12)     \n\t" // store (c41,c51)
    "vmovhpd          %%xmm2,  (%%rcx,%%r13)     \n\t" // store (c61,c71)
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c81:cf1
    "                                            \n\t"
    "vmovlpd    (%%rdx),       %%xmm0,  %%xmm0   \n\t" // load (c81,91) into xmm0[0:1]
    "vmovhpd    (%%rdx,%%rsi), %%xmm0,  %%xmm0   \n\t" // load (ca1,b1) into xmm0[2:3]
    "vmovlpd    (%%rdx,%%r12), %%xmm2,  %%xmm2   \n\t" // load (cc1,d1) into xmm2[0:1]
    "vmovhpd    (%%rdx,%%r13), %%xmm2,  %%xmm2   \n\t" // load (ce1,f1) into xmm2[2:3]
    "vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:3],xmm2)
    "vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddps           %%ymm12, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[4:7]
    "vmovlpd          %%xmm0,  (%%rdx)           \n\t" // store (c81,c91)
    "vmovhpd          %%xmm0,  (%%rdx,%%rsi)     \n\t" // store (ca1,cb1)
    "vmovlpd          %%xmm2,  (%%rdx,%%r12)     \n\t" // store (cc1,cd1)
    "vmovhpd          %%xmm2,  (%%rdx,%%r13)     \n\t" // store (ce1,cf1)
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c02:c72
    "                                            \n\t"
    "vmovlpd    (%%rcx),       %%xmm0,  %%xmm0   \n\t" // load (c02,12) into xmm0[0:1]
    "vmovhpd    (%%rcx,%%rsi), %%xmm0,  %%xmm0   \n\t" // load (c22,32) into xmm0[2:3]
    "vmovlpd    (%%rcx,%%r12), %%xmm2,  %%xmm2   \n\t" // load (c42,52) into xmm2[0:1]
    "vmovhpd    (%%rcx,%%r13), %%xmm2,  %%xmm2   \n\t" // load (c62,72) into xmm2[2:3]
    "vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:3],xmm2)
    "vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddps           %%ymm11, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[4:7]
    "vmovlpd          %%xmm0,  (%%rcx)           \n\t" // store (c02,c12)
    "vmovhpd          %%xmm0,  (%%rcx,%%rsi)     \n\t" // store (c22,c32)
    "vmovlpd          %%xmm2,  (%%rcx,%%r12)     \n\t" // store (c42,c52)
    "vmovhpd          %%xmm2,  (%%rcx,%%r13)     \n\t" // store (c62,c72)
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c82:cf2
    "                                            \n\t"
    "vmovlpd    (%%rdx),       %%xmm0,  %%xmm0   \n\t" // load (c82,92) into xmm0[0:1]
    "vmovhpd    (%%rdx,%%rsi), %%xmm0,  %%xmm0   \n\t" // load (ca2,b2) into xmm0[2:3]
    "vmovlpd    (%%rdx,%%r12), %%xmm2,  %%xmm2   \n\t" // load (cc2,d2) into xmm2[0:1]
    "vmovhpd    (%%rdx,%%r13), %%xmm2,  %%xmm2   \n\t" // load (ce2,f2) into xmm2[2:3]
    "vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:3],xmm2)
    "vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddps           %%ymm10, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[4:7]
    "vmovlpd          %%xmm0,  (%%rdx)           \n\t" // store (c82,c92)
    "vmovhpd          %%xmm0,  (%%rdx,%%rsi)     \n\t" // store (ca2,cb2)
    "vmovlpd          %%xmm2,  (%%rdx,%%r12)     \n\t" // store (cc2,cd2)
    "vmovhpd          %%xmm2,  (%%rdx,%%r13)     \n\t" // store (ce2,cf2)
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c03:c73
    "                                            \n\t"
    "vmovlpd    (%%rcx),       %%xmm0,  %%xmm0   \n\t" // load (c03,13) into xmm0[0:1]
    "vmovhpd    (%%rcx,%%rsi), %%xmm0,  %%xmm0   \n\t" // load (c23,33) into xmm0[2:3]
    "vmovlpd    (%%rcx,%%r12), %%xmm2,  %%xmm2   \n\t" // load (c43,53) into xmm2[0:1]
    "vmovhpd    (%%rcx,%%r13), %%xmm2,  %%xmm2   \n\t" // load (c63,73) into xmm2[2:3]
    "vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:3],xmm2)
    "vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddps           %%ymm9,  %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[4:7]
    "vmovlpd          %%xmm0,  (%%rcx)           \n\t" // store (c03,c13)
    "vmovhpd          %%xmm0,  (%%rcx,%%rsi)     \n\t" // store (c23,c33)
    "vmovlpd          %%xmm2,  (%%rcx,%%r12)     \n\t" // store (c43,c53)
    "vmovhpd          %%xmm2,  (%%rcx,%%r13)     \n\t" // store (c63,c73)
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c83:cf3
    "                                            \n\t"
    "vmovlpd    (%%rdx),       %%xmm0,  %%xmm0   \n\t" // load (c83,93) into xmm0[0:1]
    "vmovhpd    (%%rdx,%%rsi), %%xmm0,  %%xmm0   \n\t" // load (ca3,b3) into xmm0[2:3]
    "vmovlpd    (%%rdx,%%r12), %%xmm2,  %%xmm2   \n\t" // load (cc3,d3) into xmm2[0:1]
    "vmovhpd    (%%rdx,%%r13), %%xmm2,  %%xmm2   \n\t" // load (ce3,f3) into xmm2[2:3]
    "vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:3],xmm2)
    "vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddps           %%ymm8,  %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[4:7]
    "vmovlpd          %%xmm0,  (%%rdx)           \n\t" // store (c83,c93)
    "vmovhpd          %%xmm0,  (%%rdx,%%rsi)     \n\t" // store (ca3,cb3)
    "vmovlpd          %%xmm2,  (%%rdx,%%r12)     \n\t" // store (cc3,cd3)
    "vmovhpd          %%xmm2,  (%%rdx,%%r13)     \n\t" // store (ce3,cf3)
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "jmp    .CDONE%=                             \n\t" // jump to end.
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".CCOLSTORED%=:                              \n\t"
    "                                            \n\t"
    "                                            \n\t" // update c00:c70
    "                                            \n\t"
    "vmovaps    (%%rcx),       %%ymm0            \n\t" // load c00:c70 into ymm0
    "vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddps           %%ymm15, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vmovaps          %%ymm0,  (%%rcx)           \n\t" // store c00:c70
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c80:cf0
    "                                            \n\t"
    "vmovaps    (%%rdx),       %%ymm0            \n\t" // load c80:f0 into ymm0
    "vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddps           %%ymm14, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vmovaps          %%ymm0,  (%%rdx)           \n\t" // store c80:cf0
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c00:c70
    "                                            \n\t"
    "vmovaps    (%%rcx),       %%ymm0            \n\t" // load c01:c71 into ymm0
    "vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddps           %%ymm13, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vmovaps          %%ymm0,  (%%rcx)           \n\t" // store c01:c71
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c81:cf1
    "                                            \n\t"
    "vmovaps    (%%rdx),       %%ymm0            \n\t" // load c81:f1 into ymm0
    "vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddps           %%ymm12, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vmovaps          %%ymm0,  (%%rdx)           \n\t" // store c81:cf1
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c02:c72
    "                                            \n\t"
    "vmovaps    (%%rcx),       %%ymm0            \n\t" // load c02:c72 into ymm0
    "vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddps           %%ymm11, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vmovaps          %%ymm0,  (%%rcx)           \n\t" // store c02:c72
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c82:cf2
    "                                            \n\t"
    "vmovaps    (%%rdx),       %%ymm0            \n\t" // load c82:f2 into ymm0
    "vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddps           %%ymm10, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vmovaps          %%ymm0,  (%%rdx)           \n\t" // store c82:cf2
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c03:c73
    "                                            \n\t"
    "vmovaps    (%%rcx),       %%ymm0            \n\t" // load c03:c73 into ymm0
    "vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddps           %%ymm9,  %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vmovaps          %%ymm0,  (%%rcx)           \n\t" // store c03:c73
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c83:cf3
    "                                            \n\t"
    "vmovaps    (%%rdx),       %%ymm0            \n\t" // load c83:f3 into ymm0
    "vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
    "vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
    "vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
    "vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
    "vaddps           %%ymm8,  %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
    "vmovaps          %%ymm0,  (%%rdx)           \n\t" // store c83:cf3
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "jmp    .CDONE%=                             \n\t" // jump to end.
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".CBETAZERO%=:                               \n\t"
    "                                            \n\t" // check if aligned/column-stored
    "                                            \n\t" // check if aligned/column-stored
    "andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
    "andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
    "jne     .CCOLSTORBZ%=                       \n\t" // jump to column storage case
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".CGENSTORBZ%=:                              \n\t"
    "                                            \n\t"
    "                                            \n\t" // update c00:c70
    "                                            \n\t"
    "vextractf128 $1, %%ymm15, %%xmm2            \n\t" // xmm2 := ymm0[4:7]
    "vmovlpd          %%xmm15, (%%rcx)           \n\t" // store (c00,c10)
    "vmovhpd          %%xmm15, (%%rcx,%%rsi)     \n\t" // store (c20,c30)
    "vmovlpd          %%xmm2,  (%%rcx,%%r12)     \n\t" // store (c40,c50)
    "vmovhpd          %%xmm2,  (%%rcx,%%r13)     \n\t" // store (c60,c70)
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c80:cf0
    "                                            \n\t"
    "vextractf128 $1, %%ymm14, %%xmm2            \n\t" // xmm2 := ymm0[4:7]
    "vmovlpd          %%xmm14, (%%rdx)           \n\t" // store (c80,c90)
    "vmovhpd          %%xmm14, (%%rdx,%%rsi)     \n\t" // store (ca0,cb0)
    "vmovlpd          %%xmm2,  (%%rdx,%%r12)     \n\t" // store (cc0,cd0)
    "vmovhpd          %%xmm2,  (%%rdx,%%r13)     \n\t" // store (ce0,cf0)
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c01:c71
    "                                            \n\t"
    "vextractf128 $1, %%ymm13, %%xmm2            \n\t" // xmm2 := ymm0[4:7]
    "vmovlpd          %%xmm13, (%%rcx)           \n\t" // store (c01,c11)
    "vmovhpd          %%xmm13, (%%rcx,%%rsi)     \n\t" // store (c21,c31)
    "vmovlpd          %%xmm2,  (%%rcx,%%r12)     \n\t" // store (c41,c51)
    "vmovhpd          %%xmm2,  (%%rcx,%%r13)     \n\t" // store (c61,c71)
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c81:cf1
    "                                            \n\t"
    "vextractf128 $1, %%ymm12, %%xmm2            \n\t" // xmm2 := ymm0[4:7]
    "vmovlpd          %%xmm12, (%%rdx)           \n\t" // store (c81,c91)
    "vmovhpd          %%xmm12, (%%rdx,%%rsi)     \n\t" // store (ca1,cb1)
    "vmovlpd          %%xmm2,  (%%rdx,%%r12)     \n\t" // store (cc1,cd1)
    "vmovhpd          %%xmm2,  (%%rdx,%%r13)     \n\t" // store (ce1,cf1)
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c02:c72
    "                                            \n\t"
    "vextractf128 $1, %%ymm11, %%xmm2            \n\t" // xmm2 := ymm0[4:7]
    "vmovlpd          %%xmm11, (%%rcx)           \n\t" // store (c02,c12)
    "vmovhpd          %%xmm11, (%%rcx,%%rsi)     \n\t" // store (c22,c32)
    "vmovlpd          %%xmm2,  (%%rcx,%%r12)     \n\t" // store (c42,c52)
    "vmovhpd          %%xmm2,  (%%rcx,%%r13)     \n\t" // store (c62,c72)
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c82:cf2
    "                                            \n\t"
    "vextractf128 $1, %%ymm10, %%xmm2            \n\t" // xmm2 := ymm0[4:7]
    "vmovlpd          %%xmm10, (%%rdx)           \n\t" // store (c82,c92)
    "vmovhpd          %%xmm10, (%%rdx,%%rsi)     \n\t" // store (ca2,cb2)
    "vmovlpd          %%xmm2,  (%%rdx,%%r12)     \n\t" // store (cc2,cd2)
    "vmovhpd          %%xmm2,  (%%rdx,%%r13)     \n\t" // store (ce2,cf2)
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c03:c73
    "                                            \n\t"
    "vextractf128 $1, %%ymm9,  %%xmm2            \n\t" // xmm2 := ymm0[4:7]
    "vmovlpd          %%xmm9,  (%%rcx)           \n\t" // store (c03,c13)
    "vmovhpd          %%xmm9,  (%%rcx,%%rsi)     \n\t" // store (c23,c33)
    "vmovlpd          %%xmm2,  (%%rcx,%%r12)     \n\t" // store (c43,c53)
    "vmovhpd          %%xmm2,  (%%rcx,%%r13)     \n\t" // store (c63,c73)
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t" // update c83:cf3
    "                                            \n\t"
    "vextractf128 $1, %%ymm8,  %%xmm2            \n\t" // xmm2 := ymm0[4:7]
    "vmovlpd          %%xmm8,  (%%rdx)           \n\t" // store (c83,c93)
    "vmovhpd          %%xmm8,  (%%rdx,%%rsi)     \n\t" // store (ca3,cb3)
    "vmovlpd          %%xmm2,  (%%rdx,%%r12)     \n\t" // store (cc3,cd3)
    "vmovhpd          %%xmm2,  (%%rdx,%%r13)     \n\t" // store (ce3,cf3)
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "jmp    .CDONE%=                             \n\t" // jump to end.
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".CCOLSTORBZ%=:                              \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "vmovaps          %%ymm15, (%%rcx)           \n\t" // store c00:c70
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "vmovaps          %%ymm14, (%%rdx)           \n\t" // store c80:cf0
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "vmovaps          %%ymm13, (%%rcx)           \n\t" // store c01:c71
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "vmovaps          %%ymm12, (%%rdx)           \n\t" // store c81:cf1
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "vmovaps          %%ymm11, (%%rcx)           \n\t" // store c02:c72
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "vmovaps          %%ymm10, (%%rdx)           \n\t" // store c82:cf2
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "vmovaps          %%ymm9,  (%%rcx)           \n\t" // store c03:c73
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "vmovaps          %%ymm8,  (%%rdx)           \n\t" // store c83:cf3
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".CDONE%=:                                   \n\t"
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
      "m" (cs_c),   // 8
      "m" (b_next)/*, // 9
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


} } // namespace blas, hpc

#endif // HPC_BLAS_GEMM_AVX_CUGEMM_8X4_HPP
