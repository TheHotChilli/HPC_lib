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
CXXFLAGS := -std=c++17 -Wall -O3 -mavx # --mavx512f
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

CONFIGURATIONS = Reference SSE AVX AVX_BLIS AVX_512
TARGETS_ROWMAJOR = $(patsubst %, gemm_rowmajor_%, $(CONFIGURATIONS)) # = gemm_rowmajor_Reference, ...
TARGETS_COLMAJOR = $(patsubst %, gemm_colmajor_%, $(CONFIGURATIONS)) # = gemm_colmajor_Reference, ...
TARGETS = $(TARGETS_ROWMAJOR) $(TARGETS_COLMAJOR)


# Compile targets
all : $(TARGETS)
# Rowmajor compile targets
$(TARGETS_ROWMAJOR): gemm_rowmajor_% : gemm_benchmark.cpp
	@echo "Building $@ (Config: $*)"
	HPC_BLAS_GEMM_CONFIG_MODE=$* \
		$(CXX) $(ROWMAJOR) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS) -o $@ $<
# Colmajor compile targets
$(TARGETS_COLMAJOR): gemm_colmajor_% : gemm_benchmark.cpp
	@echo "Building $@ (Config: $*)"
	HPC_BLAS_GEMM_CONFIG_MODE=$* \
		$(CXX) $(COLMAJOR) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS) -o $@ $<

# run targets
run_colmajor_Reference : gemm_colmajor_Reference 
	@echo "Running $< benchmark"
	HPC_BLAS_GEMM_CONFIG_MODE=Reference \
	LD_LIBRARY_PATH=$(MKL_LIB_DIR):$(LD_LIBRARY_PATH) ./$< | tee $<.dat

run_colmajor_SSE : gemm_colmajor_SSE
	@echo "Running $< benchmark"
	HPC_BLAS_GEMM_CONFIG_MODE=SSE \
	LD_LIBRARY_PATH=$(MKL_LIB_DIR):$(LD_LIBRARY_PATH) ./$< | tee $<.dat

run_colmajor_AVX : gemm_colmajor_AVX
	@echo "Running $< benchmark"
	HPC_BLAS_GEMM_CONFIG_MODE=AVX \
	LD_LIBRARY_PATH=$(MKL_LIB_DIR):$(LD_LIBRARY_PATH) ./$< | tee $<.dat

run_colmajor_AVX_BLIS : gemm_colmajor_AVX_BLIS
	@echo "Running $< benchmark"
	HPC_BLAS_GEMM_CONFIG_MODE=AVX_BLIS \
	LD_LIBRARY_PATH=$(MKL_LIB_DIR):$(LD_LIBRARY_PATH) ./$< | tee $<.dat

run_colmajor_AVX_512 : gemm_colmajor_AVX_512
	@echo "Running $< benchmark"
	HPC_BLAS_GEMM_CONFIG_MODE=AVX_512 \
	LD_LIBRARY_PATH=$(MKL_LIB_DIR):$(LD_LIBRARY_PATH) ./$< | tee $<.dat

run_rowmajor_Reference : gemm_rowmajor_Reference
	@echo "Running $< benchmark"
	HPC_BLAS_GEMM_CONFIG_MODE=Reference \
	LD_LIBRARY_PATH=$(MKL_LIB_DIR):$(LD_LIBRARY_PATH) ./$< | tee $<.dat

run_rowmajor_SSE : gemm_rowmajor_SSE
	@echo "Running $< benchmark"
	HPC_BLAS_GEMM_CONFIG_MODE=SSE \
	LD_LIBRARY_PATH=$(MKL_LIB_DIR):$(LD_LIBRARY_PATH) ./$< | tee $<.dat

run_rowmajor_AVX : gemm_rowmajor_AVX
	@echo "Running $< benchmark"
	HPC_BLAS_GEMM_CONFIG_MODE=AVX \
	LD_LIBRARY_PATH=$(MKL_LIB_DIR):$(LD_LIBRARY_PATH) ./$< | tee $<.dat

run_rowmajor_AVX_BLIS : gemm_rowmajor_AVX_BLIS
	@echo "Running $< benchmark"
	HPC_BLAS_GEMM_CONFIG_MODE=AVX_BLIS \
	LD_LIBRARY_PATH=$(MKL_LIB_DIR):$(LD_LIBRARY_PATH) ./$< | tee $<.dat

run_rowmajor_AVX_512 : gemm_rowmajor_AVX_512
	@echo "Running $< benchmark"
	HPC_BLAS_GEMM_CONFIG_MODE=AVX_512 \
	LD_LIBRARY_PATH=$(MKL_LIB_DIR):$(LD_LIBRARY_PATH) ./$< | tee $<.dat

run_colmajor: $(addprefix run_colmajor_, $(CONFIGURATIONS))

run_rowmajor: $(addprefix run_rowmajor_, $(CONFIGURATIONS))

run_all: run_colmajor run_rowmajor

# plot targets
plot_rowmajor: $(addsuffix .dat, $(TARGETS_ROWMAJOR))
	gnuplot gemm_rowmajor.plot

plot_colmajor: $(addsuffix .dat, $(TARGETS_COLMAJOR))
	gnuplot gemm_colmajor.plot

plot: plot_rowmajor plot_colmajor

print: 
	@echo $(TARGETS)

# clean targets
.PHONY: clean
clean:
	$(RM) $(TARGETS) $(wildcard *.o) $(addsuffix .dat, $(TARGETS)) $(addsuffix .svg, $(TARGETS))

.PHONY: clean_dat
clean_dat: 
	$(RM) $(addsuffix .dat, $(TARGETS)) $(addsuffix .svg, $(TARGETS))