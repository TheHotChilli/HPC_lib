# Matrix Storage Format
COLMAJOR  = -DCOLMAJOR=1
ROWMAJOR  = -DCOLMAJOR=0

# Compiler and Compiler Flags
CC_LINUX       := g++
CC_WINDOWS     := g++
CFLAGS_LINUX  := -std=c++17 -Wall -Ofast
CFLAGS_WINDOWS := -std=c++17 -Wall -Ofast
# Determine Platform
ifeq ($(shell uname -s), Linux)
    CC       := $(CC_LINUX)
    CFLAGS  := $(CFLAGS_LINUX)
else
    CC       := $(CC_WINDOWS)
    CFLAGS  := $(CFLAGS_WINDOWS)
endif

# Intel MKL Configuration (replace with appropriate paths)
# 
# Intel link line advisor:
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html
MKL_ARCH := intel64				                                            #Processor Architecture
MKL_ROOT := /opt/intel/oneapi/mkl/latest
MKL_INCLUDE_DIR  := $(MKL_ROOT)/include
MKL_LIB_DIRS := $(MKL_ROOT)/lib/$(MKL_ARCH) $(MKL_ROOT)/mkl/lib/$(MKL_ARCH)

# Linker 
LinkerFlag := -Wl,
Rpath := $(patsubst %,$(LinkerFlag)-rpath $(LinkerFlag)%,$(MKL_LIB_DIRS))	# Runtime Lib (Execution)
Lpath := $(patsubst %,-L%,$(MKL_LIB_DIRS))									# Lib (Linking)
LDLIBS := -lmkl_intel_ilp64 -lmkl_core -lmkl_sequential -lm -lpthread
LDFLAGS := -L$(MKL_LIBS) -Wl,-rpath,$(MKL_LIBS)

TARGETS = gemv_rowmajor gemv_colmajor

all: $(TARGETS)

%_rowmajor : %.cpp
	$(CPP) $(ROWMAJOR) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS) -o $@ $<

%_colmajor : %.cpp
	$(CXX) $(COLMAJOR) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS) -o $@ $<

clean:
	$(RM) $(TARGETS) $(wildcard *.o)