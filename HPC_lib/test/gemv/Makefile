# Specify Matrix Storage Format
COLMAJOR  = -DCOLMAJOR=1
ROWMAJOR  = -DCOLMAJOR=0

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

# Linker Flags (static linking)
# LDFLAGS := -Wl,--start-group
# LDFLAGS += $(MKL_LIB_DIR)/libmkl_intel_ilp64.a
# LDFLAGS += $(MKL_LIB_DIR)/libmkl_sequential.a
# LDFLAGS += $(MKL_LIB_DIR)/libmkl_core.a
# LDFLAGS += -Wl,--end-group
# Linked Libraries (static linking)
# LDLIBS := -lpthread -lm -ldl

# Linker Flags (dynmaic linking)
LDFLAGS := -L$(MKL_LIB_DIR) -Wl,--no-as-needed
# Linker Libraries (static linking)
LDLIBS := -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

TARGETS = gemv_rowmajor gemv_colmajor

all: $(TARGETS)

gemv_rowmajor : gemv_benchmark.cpp
	$(CXX) $(ROWMAJOR) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS) -o $@ $<

gemv_colmajor : gemv_benchmark.cpp
	$(CXX) $(COLMAJOR) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS) -o $@ $<

run: run_rowmajor run_colmajor

run_rowmajor: gemv_rowmajor
	@echo "ROWMAJOR Benchmark"
	LD_LIBRARY_PATH=$(MKL_LIB_DIR):$(LD_LIBRARY_PATH) ./gemv_rowmajor | tee gemv_rowmajor.dat

run_colmajor: gemv_colmajor
	@echo "COLMAJOR Benchmark"
	LD_LIBRARY_PATH=$(MKL_LIB_DIR):$(LD_LIBRARY_PATH) ./gemv_colmajor | tee gemv_colmajor.dat

plot: plot_rowmajor plot_colmajor

plot_rowmajor: gemv_rowmajor.dat
	gnuplot gemv_rowmajor.plot

plot_colmajor: gemv_colmajor.dat
	gnuplot gemv_colmajor.plot

clean:
	$(RM) $(TARGETS) $(wildcard *.o) $(addsuffix .dat, $(TARGETS)) $(addsuffix .svg, $(TARGETS))