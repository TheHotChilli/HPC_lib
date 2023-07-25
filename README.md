# High Performance Computing (HPC) lib

This repo has emerged from the HPC lectures in the Master's program in Computational Science and Engineering at the University of Ulm. In the course of the lecture, numerical algorithms were parallelized step by step and implemented in C++. In the process, a small HPC computing library was created, which is made available in this repo.

## Structure of the project

This project follows a modern canocial structure for C/C++ projects, as presented by [Boris Kolpackov](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1204r0.html). The Project is structured as follows:

- HPC_lib 
  - BLAS
- doc 
- test

Source code in detail

- HPC_lib 
  - BLAS
    - GEMM
    - GEMV


## Basic Linear Algebra System (BLAS)
~50 Operations 

- Level 1: Vector-vector Operations
- Level 2: Matrix-vector Operations
- Level 3: Matrix-matrix Operations

There exist different implementations for BLAS. A famous one is the [Intel Math Kernel Library (MKL)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html). 

### Implemented functions:

- Level 1:
  - SWAP
  - SCAL
  - AXPY
  - DOT
  - ASUM
- Level 2:
  - GEMV
- Level 3:
  - GEMM

## Parallelizations/Optimization
- GEMV
  - row major vs col major
- GEMM
  - row major vs col major
  - partioned/blocks (session 5.5)