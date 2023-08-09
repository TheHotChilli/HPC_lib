set terminal svg noenhanced size 900, 500
set output "gemm_colmajor.svg"
set xlabel "Matrix dimensions: M=N=K"
set ylabel "MFLOPS"
set title "GEMM (colmajor, double)"
set key outside
# set yrange [0:26000]
plot "gemm_colmajor_Reference.dat" using 1:15 with linespoints lt 1 lw 2 title "Intel MKL", \
     "gemm_colmajor_Reference.dat" using 1:16 with linespoints lt 2 lw 2 title "Reference", \
     "gemm_colmajor_SSE.dat"       using 1:16 with linespoints lt 3 lw 2 title "SSE", \
     "gemm_colmajor_AVX.dat"       using 1:16 with linespoints lt 4 lw 2 title "AVX", \
     "gemm_colmajor_AVX_BLIS.dat"  using 1:16 with linespoints lt 5 lw 2 title "AVX_BLIS", \
     "gemm_colmajor_AVX_512.dat"   using 1:16 with linespoints lt 6 lw 2 title "AVX_512"
