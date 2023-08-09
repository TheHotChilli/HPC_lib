# High Performance Computing (HPC) lib

This repo emerged from the High Performance Computing 1 lecture at the University of Ulm, which I listened to in the winter semester 2020/2021 as part of my Master's program in Computational Science and Engineering. The lecture is held by [Dr. Michael Lehn](https://www.uni-ulm.de/mawi/institut-fuer-numerische-mathematik/institut/mitarbeiter/mlehn/) and [Dr. Andreas F. Borchert](https://www.uni-ulm.de/mawi/institut-fuer-numerische-mathematik/institut/mitarbeiter/dr-andreas-f-borchert/). During the lecture, numerical algorithms were optimized and parallelized step by step and implemented in C/C++. This resulted in a small HPC computing library, which is made available in this repo. 

## Structure of the project 

This project follows a modern canocial structure and is structured as follows:

```
└── HPC_lib
    ├── BLAS
    ├── test
    └── utils
```
The 

### BLAS (Basic Linear Algebra Subprograms)
This subfolder contains generic C++ implementations (using template parameters) for some operations from the Basic Linear Algebra Subprograms (BLAS). 

BLAS is a set of ~50 low-level linear algebra operations that form the basis for more sophisticated mathematical applications. A good overview of all BLAS operations can be found [here](https://www.netlib.org/blas/blasqr.pdf). The BLAS operations are divided into 3 subsets, so-called "levels", which differ in terms of the involved computational complexity:
<center>

| Level   | Operation Style      | Elements | FLOP |
| ------- | --------------| ---------| ---------- |
|  1 | Vector-Vector | $\mathcal{O}(n)$  | $\mathcal{O}(n)$ |
|  2 | Matrix-Vector | $\mathcal{O}(n²)$ | $\mathcal{O}(n²)$ |
|  3 | Matrix-Matrix | $\mathcal{O}(n²)$ | $\mathcal{O}(n³)$ |

</center>

This project implements the level 1 operations __SCAL__, __COPY__, __DOT__ and __AXPY__. Further we extend these vector operations to matrices and introduce their generalized variants __GESCAL__, __GECOPY__, and __GEAXPY__. In addition to these relatively simple operations  cache optimized variants of the level 2 __GEMV__ operation (General Matrix-Vector Product) and the level 3 __GEMM__ operation (General Matrix-Matrix Product) are realized as part of this project. 

__GEMV__ and __GEMM__ operations form the basis for solving a wide range of more complex applications and are therefore very important in the field of high-performance computing. 

__GEMV__ is defined as 
  $$ y \leftarrow \alpha A x + \beta y,$$
with $a \in \mathbb{C}$, $b \in \mathbb{C}$, $x \in \mathbb{C}^n$ ,$y \in \mathbb{C}^n$, $A \in \mathbb{C}^{m \times k}$.

__GEMM__ is defined as 
  $$C \leftarrow \alpha A B + \beta C,$$
whereby $a \in \mathbb{C}$, $b \in \mathbb{C}$, $A \in \mathbb{C}^{m \times k}$, $B \in \mathbb{C}^{k \times n}$ and $C \in \mathbb{C}^{m \times n}$.

### test
This subfolder contains tests for our implmentations of the __GEMV__ and __GEMM__ BLAS operations. For benchmarking we compare to the I[ntel Math Kernel Library (MKL)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html). All tests were run on a UNIX based system. 

### utils
Contains several functions that are used at various points within the project. 

## License