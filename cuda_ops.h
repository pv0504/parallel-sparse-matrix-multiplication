#ifndef CUDA_OPS_H
#define CUDA_OPS_H

#include "matrix.h"
#include <vector>

// Remove extern "C" as it's not needed for C++ functions
Matrix multiply_pair_cuda(Matrix A, Matrix B, int k);
Matrix multiply_sequential_cuda(std::vector<Matrix> matrices, int k, int rank);

#endif // CUDA_OPS_H