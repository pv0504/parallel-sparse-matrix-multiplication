#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <fstream>
#include <memory>
#include <string>
#include <iostream>
#include <climits>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <mpi.h>
#include <map>
using namespace std; 

struct Matrix {
    int height = 0, width = 0, k = 0;
    // std::vector<std::pair<std::pair<int, int>, std::unique_ptr<uint64_t[]>>> blocks;
    std::unique_ptr<uint64_t[]> blocks;
    int num_blocks;
    Matrix() = default;
    Matrix(int h, int w, int k_val) : height(h), width(w), k(k_val) {}
    Matrix(Matrix&& other) noexcept = default;
    Matrix& operator=(Matrix&& other) noexcept = default;
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;
};

std::vector<char> serialize_matrix(const Matrix& mat);
Matrix deserialize_matrix(const std::vector<char>& buffer);

Matrix read_single_matrix(std::ifstream& file, int k);
std::vector<Matrix> read_matrices_parallel(
    const std::string& folderPath, 
    int start_idx, 
    int end_idx, 
    int k
);
Matrix do_internal_mul(Matrix A, Matrix B, int k);
// Block dimension calculation
inline int compute_block_dim(int pos, int matrix_dim, int k) {
    return std::min(k, matrix_dim - pos);
}

#endif // MATRIX_H