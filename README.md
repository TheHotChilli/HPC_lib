# High Performance Computing (HPC) lib

This repo emerged from the High Performance Computing 1 lecture at the University of Ulm, which I listened to in the winter semester 2020/2021 as part of my Master's program in Computational Science and Engineering. The lecture is held by [Dr. Michael Lehn](https://www.uni-ulm.de/mawi/institut-fuer-numerische-mathematik/institut/mitarbeiter/mlehn/) and [Dr. Andreas F. Borchert](https://www.uni-ulm.de/mawi/institut-fuer-numerische-mathematik/institut/mitarbeiter/dr-andreas-f-borchert/). In the course of the lecture, numerical algorithms were optimized and parallelized step by step and implemented in C++. In the process, a small HPC computing library was created, which is made available in this repo.

## Structure of the project 

This project follows a modern canocial structure for C/C++ projects and is structured as follows:

```
└── HPC_lib
    ├── BLAS
    ├── test
    │   └── gemv
    └── utils
```
The overall project is seperated into several sub projects, which are described in the following.

### BLAS (Basic Linear Algebra Subprograms)
This subfolder contains generic C++ implementations (using template parameters) for some functions from the Basic Linear Algebra Subprograms (BLAS). BLAS is a set of ~50 low-level linear algebra operations that form the basis for more sophisticated mathematical applications. A good overview of all BLAS operations can be found [here](https://www.netlib.org/blas/blasqr.pdf). The BLAS operations are divided into 3 subsets, called "levels": 
- Level 1: Vector-vector (Elements: O(n),  Operations O(n))
- Level 2: Matrix-vector (Elements: O(n²), Operations O(n²))
- Level 3: Matrix-matrix (Elements: O(n²), Operations O(n³))

 This projects implements the following operations:
- Level 1:
  - (SWAP)
  - SCAL
  - COPY
  - DOT
- Level 2:
  - GEMV
- Level 3:
  - (GEMM)

### test
This subfolder contains tests for our implmentations of BLAS routines. For benchmarking we compare to the Intel Math Kernel Library (MKL). 

### utils
Contains several usefull functions that are used at various points within the project. 

## Parallelizations/Optimization
- GEMV
  - row major vs col major
- GEMM
  - row major vs col major
  - partioned/blocks (session 5.5)