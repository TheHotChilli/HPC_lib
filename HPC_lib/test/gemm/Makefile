# Specify Matrix Storage Format
COLMAJOR = -DCOLMAJOR_C=1 -DCOLMAJOR_A=1 -DCOLMAJOR_B=1
ROWMAJOR = -DCOLMAJOR_C=0 -DCOLMAJOR_A=0 -DCOLMAJOR_B=0

# HPC-Lib Dir
HPC_ROOT = /home/morph/Projekte/HPC_lib

# Intel MKL Configuration (replace with appropriate paths)
# Also see Intel link line advisor:
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html
MKL_ARCH := intel64
MKL_ROOT := /opt/intel/oneapi/mkl/latest
MKL_INCLUDE_DIR := $(MKL_ROOT)/include
MKL_LIB_DIR := $(MKL_ROOT)/lib/$(MKL_ARCH)
#MKL_LIB_DIRS := $(MKL_ROOT)/lib/$(MKL_ARCH) $(MKL_ROOT)/mkl/lib/$(MKL_ARCH)

# Compiler and Flags
CXX := g++
CXXFLAGS := -std=c++17 -Wall -Ofast
CC := $(CXX)
CFLAGS := $(CXXFLAGS)

# Preprocessor Flags (where should we look for files to include)
CPPFLAGS += -I$(MKL_INCLUDE_DIR)
CPPFLAGS += -I$(HPC_ROOT)
CPPFLAGS += -m64 -DMKL_ILP64 

# Linker Flags (dynmaic linking)
LDFLAGS := -L$(MKL_LIB_DIR) -Wl,--no-as-needed
# Linker Libraries (static linking)
LDLIBS := -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

TARGETS = gemm_rowmajor gemm_colmajor

gemm_rowmajor : gemm_benchmark.cpp
	$(CXX) $(ROWMAJOR) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS) -o $@ $<

gemm_colmajor : gemm_benchmark.cpp
	$(CXX) $(COLMAJOR) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS) -o $@ $<

run: run_rowmajor run_colmajor

run_rowmajor: gemm_rowmajor
	@echo "ROWMAJOR Benchmark"
	LD_LIBRARY_PATH=$(MKL_LIB_DIR):$(LD_LIBRARY_PATH) ./gemm_rowmajor | tee gemm_rowmajor.dat

run_colmajor: gemm_colmajor
	@echo "COLMAJOR Benchmark"
	LD_LIBRARY_PATH=$(MKL_LIB_DIR):$(LD_LIBRARY_PATH) ./gemm_colmajor | tee gemm_colmajor.dat

plot: plot_rowmajor plot_colmajor

plot_rowmajor: gemm_rowmajor.dat
	gnuplot gemm_rowmajor.plot

plot_colmajor: gemm_colmajor.dat
	gnuplot gemm_colmajor.plot

clean:
	$(RM) $(TARGETS) $(wildcard *.o) $(addsuffix .dat, $(TARGETS)) $(addsuffix .svg, $(TARGETS))