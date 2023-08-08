# set terminal svg size 900, 500
# set output "gemv_rowmajor.svg"
set terminal pdf size 900, 500
set output "gemv_rowmajor.pdf"
set xlabel "Matrix dim A: M=N"
set ylabel "MFLOPS"
set yrange [0:11000]
set title "GEMV: Row major"
set key outside
set pointsize 0.5
plot "gemv_rowmajor.dat" using 2:8 with linespoints lt 2 lw 3 title "MKL", \
     "gemv_rowmajor.dat" using 2:9 with linespoints lt 3 lw 3 title "Custom"