
# # Compiler and flags
# MPICXX = mpicxx
# NVCC = nvcc

# # Flags
# OMPFLAG = -fopenmp
# CUDA_PATH = /usr/local/cuda
# CUDA_INCLUDES = -I$(CUDA_PATH)/include
# CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart

# # C++ flags
# CXXFLAGS = -O2 -std=c++17 $(OMPFLAG) $(CUDA_INCLUDES)
# NVCCFLAGS = -std=c++14
# # Target executable
# TARGET = a4

# # Source files
# SRC_CPP = main.cpp
# SRC_CU  = cuda_kernels.cu

# # Object files
# OBJS = main.o cuda_kernels.o

# # Default build
# all: $(TARGET)

# # Link
# $(TARGET): $(OBJS)
# 	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(CUDA_LIBS) -pthread

# # Compile main.cpp with MPI + OpenMP
# main.o: main.cpp
# 	$(MPICXX) $(CXXFLAGS) -c $< -o $@

# # Compile cuda_kernels.cu with NVCC
# cuda_kernels.o: cuda_kernels.cu
# 	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# # Run the program (with input folder name, e.g., small_test)
# run: $(TARGET)
# 	mpirun -np 1 perf stat -e instructions,cycles ./$(TARGET) Passed_new/pos_large_test

# # Clean up
# clean:
# 	rm -f $(TARGET) $(OBJS) *.txt
# 	find . -type f -executable -exec rm -f {} +

# .PHONY: all run clean


# Updated Compiler and Flags section with optimizations
MPICXX = mpicxx
NVCC = nvcc



# Modified Compiler Flags (ODR-safe)
OPT_FLAGS = -O3 -march=native -funroll-loops -ffast-math
CUDA_OPT_FLAGS = -Xptxas -O3 --fmad=true --use_fast_math

# Keep these unchanged
OMPFLAG = -fopenmp
CUDA_PATH = /usr/local/cuda
CUDA_INCLUDES = -I$(CUDA_PATH)/include
CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart

# Unified C++ standard
CXXFLAGS = -std=c++14 $(OPT_FLAGS) $(OMPFLAG) $(CUDA_INCLUDES)
NVCCFLAGS = -std=c++14 $(CUDA_OPT_FLAGS) -Xcompiler "$(OPT_FLAGS) $(OMPFLAG)"  # <-- Added $(OMPFLAG) here
LDFLAGS = $(OPT_FLAGS) $(OMPFLAG) $(CUDA_LIBS) -pthread  # <-- Ensure $(OMPFLAG) is present here too

# Target and objects remain same
TARGET = a4
SRC_CPP = main.cpp
SRC_CU  = cuda_kernels.cu
OBJS = main.o cuda_kernels.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

main.o: main.cpp
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

cuda_kernels.o: cuda_kernels.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Rest remains same...

# Run the program (with input folder name, e.g., small_test) Passed_new/pos_large_test
run: $(TARGET)
	mpirun -np 4 perf stat -e instructions,cycles ./$(TARGET) Passed_new/medium_test

# Clean up
clean:
	rm -f $(TARGET) $(OBJS) *.txt
	find . -type f -executable -exec rm -f {} +

.PHONY: all run clean
