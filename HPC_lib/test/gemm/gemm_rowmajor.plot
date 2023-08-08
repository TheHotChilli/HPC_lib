set terminal svg noenhanced size 900, 500
set output "gemm_rowmajor.svg"
set xlabel "Matrix dimensions: M=N=K"
set ylabel "MFLOPS"
set title "GEMM (rownmajor, double)"
set key outside
# set yrange [0:26000]
plot "gemm_rowmajor_Reference.dat" using 1:15 with linespoints lt 1 lw 2 title "Intel MKL", \
     "gemm_rowmajor_Reference.dat" using 1:16 with linespoints lt 2 lw 2 title "Reference", \
     "gemm_rowmajor_SSE.dat"       using 1:16 with linespoints lt 3 lw 2 title "SSE", \
     "gemm_rowmajor_AVX.dat"       using 1:16 with linespoints lt 4 lw 2 title "AVX", \
     "gemm_rowmajor_AVX_BLIS.dat"  using 1:16 with linespoints lt 5 lw 2 title "AVX_BLIS", \
     "gemm_rowmajor_AVX_512.dat"   using 1:16 with linespoints lt 6 lw 2 title "AVX_512"
