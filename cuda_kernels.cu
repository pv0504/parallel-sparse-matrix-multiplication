#include "matrix.h"
#include <cuda_runtime.h>
#include "cuda_ops.h"

using namespace std;

#include <cstdio>  // Add this include

void write_matrix_to_file(const Matrix& mat, int rank) {
    if (mat.height == 0 || mat.width == 0 || mat.num_blocks == 0) {
        std::cerr << "Matrix is empty. Nothing to write for rank " << rank << "." << std::endl;
        return;
    }

    std::string filename = "result_matrix_k" + std::to_string(rank);
    std::ofstream outfile(filename);

    if (!outfile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    outfile << mat.height << " " << mat.width << "\n";
    outfile << mat.num_blocks << "\n";

    for (int i = 0; i < mat.num_blocks; ++i) {
        int offset = i * (2 + mat.k * mat.k);
        int r = mat.blocks[offset];
        int c = mat.blocks[offset + 1];
        uint64_t* data = mat.blocks.get() + offset + 2;

        int actual_rows = compute_block_dim(r, mat.height, mat.k);
        int actual_cols = compute_block_dim(c, mat.width, mat.k);
        int k = mat.k;
        outfile << r << " " << c << "\n";
        for (int row = 0; row < k; ++row) {
            for (int col = 0; col <  k; ++col) {
                outfile << data[row * k + col];
                if (col < k - 1) {
                    outfile << " ";
                }
            }
            outfile << "\n";
        }
    }

    outfile.close();
    std::cout << "Matrix written to " << filename << std::endl;
}

/*

__global__ void batchedCudaMatMul(
    const uint64_t* A, const uint64_t* B, uint64_t* C,
    int batch_size, int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    int A_off = idx * k * k;
    int B_off = idx * k * k;
    int C_off = idx * k * k;

    for (int row = 0; row < k; ++row) {
        for (int col = 0; col < k; ++col) {
            uint64_t sum = 0;
            for (int t = 0; t < k; ++t) {
                sum += A[A_off + row * k + t] * B[B_off + t * k + col];
            }
            C[C_off + row * k + col] = sum;
        }
    }
}



struct BatchEntry {
    pair<int, int> c_pos;  // Position of the result block
};


Matrix multiply_pair_cuda(Matrix A, Matrix B, int k) {
    map<pair<int, int>, unique_ptr<uint64_t[]>> result_blocks;
    map<int, vector<pair<int, int>>> m1, m2;

    for(int i = 0; i < A.num_blocks; i++) {
        int offset = i * (2 + k * k);
        int row = A.blocks[offset];
        int col = A.blocks[offset + 1];
        m1[col].emplace_back(i, row);  
    }

    for(int i = 0; i < B.num_blocks; i++) {
        int offset = i * (2 + k * k);
        int row = B.blocks[offset];
        int col = B.blocks[offset + 1];
        m2[row].emplace_back(i, col);  
    }

    vector<uint64_t> h_A, h_B;
    vector<pair<int, int>> result_positions;

    for (const auto& entry : m1) {
        int z = entry.first;
        const auto& a_blocks = entry.second;

        if(!m2.count(z)) continue;
        
        for (const auto& ab : a_blocks) {
            int a_idx = ab.first;
            int x = ab.second;

            const uint64_t* a_data = A.blocks.get() + a_idx * (2 + k*k) + 2;
            
            for (const auto& bb : m2[z]) {
                int b_idx = bb.first;
                int y = bb.second;
                const uint64_t* b_data = B.blocks.get() + b_idx * (2 + k*k) + 2;
                
                // Add to batch
                h_A.insert(h_A.end(), a_data, a_data + k*k);
                h_B.insert(h_B.end(), b_data, b_data + k*k);
                result_positions.emplace_back(x, y);
            }
        }
    }

    if(h_A.empty()) return Matrix(A.height, B.width, k);

    uint64_t *d_A, *d_B, *d_C;
    const size_t batch_size = h_A.size() / (k*k);
    const size_t buf_size = batch_size * k*k * sizeof(uint64_t);

    cudaMalloc(&d_A, buf_size);
    cudaMalloc(&d_B, buf_size);
    cudaMalloc(&d_C, buf_size);

    cudaMemcpy(d_A, h_A.data(), buf_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), buf_size, cudaMemcpyHostToDevice);

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    batchedCudaMatMul<<<blocks, threads>>>(d_A, d_B, d_C, batch_size, k);
    (cudaDeviceSynchronize());

    vector<uint64_t> h_C(batch_size * k*k);
    cudaMemcpy(h_C.data(), d_C, buf_size, cudaMemcpyDeviceToHost);

    #pragma omp parallel for
    for(size_t i = 0; i < batch_size; i++) {
        int x = result_positions[i].first;
        int y = result_positions[i].second;
        const uint64_t* block_result = h_C.data() + i * k * k;

        auto iter = result_blocks.end();
        bool exists = false;

        #pragma omp critical
        {
            iter = result_blocks.find({x, y});
            if (iter == result_blocks.end()) {
                auto temp = std::make_unique<uint64_t[]>(k * k);
                std::copy(block_result, block_result + k * k, temp.get());
                result_blocks[{x, y}] = std::move(temp);
                exists = false;
            } else {
                exists = true;
            }
        }

        if (exists) {
            for(int j = 0; j < k * k; j++) {
                #pragma omp atomic
                iter->second[j] += block_result[j];
            }
        }
    }

    // for(size_t i = 0; i < batch_size; i++) {
    //     int x = result_positions[i].first;
    //     int y = result_positions[i].second;
    //     const uint64_t* block_result = h_C.data() + i * k * k;

    //     auto& block_ptr = result_blocks[{x, y}];
    //     if (!block_ptr) {
    //         block_ptr = std::make_unique<uint64_t[]>(k * k);
    //         std::copy(block_result, block_result + k * k, block_ptr.get());
    //     } else {
    //         std::transform(block_result,
    //                     block_result + k * k,
    //                     block_ptr.get(),
    //                     block_ptr.get(),
    //                     [](uint64_t a, uint64_t b) {
    //                         return a + b;
    //                     });
    //     }
    // }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    Matrix result;
    result.height = A.height;
    result.width = B.width;
    result.k = k;
    result.num_blocks = result_blocks.size();
    result.blocks = make_unique<uint64_t[]>(result.num_blocks * (2 + k*k));

    int idx = 0;
    for(const auto& entry : result_blocks) {
        int offset = idx * (2 + k*k);
        result.blocks[offset] = entry.first.first;     // row
        result.blocks[offset + 1] = entry.first.second; // column
        // memcpy(result.blocks.get() + offset + 2, entry.second.get(), k * k * sizeof(uint64_t));
        copy(entry.second.get(), entry.second.get() + k*k, result.blocks.get() + offset + 2);
        idx++;
    }

    return result;
}

*/

__global__ void batchedCudaMatMul(
    const uint64_t* A, const uint64_t* B, uint64_t* C,int A_height, int A_width, int B_width,
    int batch_size, int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const int A_offset = idx * (2 + k*k);
    const int B_offset = idx * (2 + k*k);
    const int C_offset = idx * (2 + k*k);

    int a_row = A[A_offset];
    int a_col = A[A_offset + 1];
    int b_col = B[B_offset + 1];

    int a_rows = min(k, A_height - a_row);
    int a_cols = min(k, A_width - a_col);
    int b_cols = min(k, B_width - b_col);

    for(int i = 0; i < a_rows; ++i) {
        for(int j = 0; j < b_cols; ++j) {
            uint64_t sum = 0;
            for(int t = 0; t < a_cols; ++t) {
                sum += A[A_offset + 2 + i*k + t] * B[B_offset + 2 + t*k + j];
            }
            C[C_offset + 2 + i*k + j] = sum;
        }
    }
}



struct BatchEntry {
    pair<int, int> c_pos;  // Position of the result block
};


Matrix multiply_pair_cuda(Matrix A, Matrix B, int k) {
    map<pair<int, int>, unique_ptr<uint64_t[]>> result_blocks;
    map<int, vector<pair<int, int>>> m1, m2;

    for(int i = 0; i < A.num_blocks; i++) {
        int offset = i * (2 + k * k);
        int row = A.blocks[offset];
        int col = A.blocks[offset + 1];
        m1[col].emplace_back(i, row);  
    }

    for(int i = 0; i < B.num_blocks; i++) {
        int offset = i * (2 + k * k);
        int row = B.blocks[offset];
        int col = B.blocks[offset + 1];
        m2[row].emplace_back(i, col);  
    }

    vector<uint64_t> h_A, h_B;
    vector<pair<int, int>> result_positions;

    for (const auto& entry : m1) {
        int z = entry.first;
        const auto& a_blocks = entry.second;

        if(!m2.count(z)) continue;
        
        for (const auto& ab : a_blocks) {
            int a_idx = ab.first;
            int x = ab.second;

            const uint64_t* a_data = A.blocks.get() + a_idx * (2 + k*k) + 2;
            
            for (const auto& bb : m2[z]) {
                int b_idx = bb.first;
                int y = bb.second;
                const uint64_t* b_data = B.blocks.get() + b_idx * (2 + k*k) + 2;
                
                // Add to batch
                h_A.push_back((uint64_t)x);
                h_A.push_back((uint64_t)z);
                h_A.insert(h_A.end(), a_data, a_data + k*k);

                h_B.push_back((uint64_t)z);
                h_B.push_back((uint64_t)y);
                h_B.insert(h_B.end(), b_data, b_data + k*k);
                result_positions.emplace_back(x, y);
            }
        }
    }

    if(h_A.empty()) return Matrix(A.height, B.width, k);

    uint64_t *d_A, *d_B, *d_C;
    const size_t batch_size = h_A.size() / (k*k + 2);
    const size_t buf_size =  h_A.size() * sizeof(uint64_t);

    cudaMalloc(&d_A, buf_size);
    cudaMalloc(&d_B, buf_size);
    cudaMalloc(&d_C, buf_size);
    cudaMemset(d_C, 0, buf_size);

    cudaMemcpy(d_A, h_A.data(), buf_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), buf_size, cudaMemcpyHostToDevice);

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    batchedCudaMatMul<<<blocks, threads>>>(d_A, d_B, d_C, A.height,A.width,B.width, batch_size, k);
    (cudaDeviceSynchronize());

    vector<uint64_t> h_C(batch_size * (k*k+2));
    cudaMemcpy(h_C.data(), d_C, buf_size, cudaMemcpyDeviceToHost);

    #pragma omp parallel for
    for(size_t i = 0; i < batch_size; i++) {
        int x = result_positions[i].first;
        int y = result_positions[i].second;
        const uint64_t* block_result = h_C.data() + i * (2 + k * k) + 2;

        auto iter = result_blocks.end();
        bool exists = false;

        #pragma omp critical
        {
            iter = result_blocks.find({x, y});
            if (iter == result_blocks.end()) {
                auto temp = std::make_unique<uint64_t[]>(k * k);
                std::copy(block_result, block_result + k * k, temp.get());
                result_blocks[{x, y}] = std::move(temp);
                exists = false;
            } else {
                exists = true;
            }
        }

        if (exists) {
            for(int j = 0; j < k * k; j++) {
                #pragma omp atomic
                iter->second[j] += block_result[j];
            }
        }
    }


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    Matrix result;
    result.height = A.height;
    result.width = B.width;
    result.k = k;
    result.num_blocks = result_blocks.size();
    result.blocks = make_unique<uint64_t[]>(result.num_blocks * (2 + k*k));

    int idx = 0;
    for(const auto& entry : result_blocks) {
        int offset = idx * (2 + k*k);
        result.blocks[offset] = entry.first.first;     // row
        result.blocks[offset + 1] = entry.first.second; // column
        // memcpy(result.blocks.get() + offset + 2, entry.second.get(), k * k * sizeof(uint64_t));
        copy(entry.second.get(), entry.second.get() + k*k, result.blocks.get() + offset + 2);
        idx++;
    }

    return result;
}


bool is_zero_matrix(const Matrix& mat) {
    if (mat.num_blocks == 0) return true;

    int elements_per_block = 2 + mat.k * mat.k; // Skip row/col metadata
    for (int i = 0; i < mat.num_blocks; ++i) {
        int offset = i * elements_per_block + 2; // Skip position metadata
        for (int j = 0; j < mat.k * mat.k; ++j) {
            if (mat.blocks[offset + j] != 0) {
                return false;
            }
        }
    }
    return true;
}


Matrix multiply_sequential_cuda(std::vector<Matrix> matrices, int k, int rank) {
    if (matrices.empty()) return Matrix(0, 0, k);
    if (matrices.size() == 1) return std::move(matrices[0]);

    // Initialize CPU path with a deep copy of matrices[0]
    Matrix cpu_path;
    cpu_path.height = matrices[0].height;
    cpu_path.width = matrices[0].width;
    cpu_path.k = matrices[0].k;
    cpu_path.num_blocks = matrices[0].num_blocks;
    if (matrices[0].num_blocks > 0) {
        size_t block_size = matrices[0].num_blocks * (2 + k * k);
        cpu_path.blocks = std::make_unique<uint64_t[]>(block_size);
        std::memcpy(cpu_path.blocks.get(), matrices[0].blocks.get(), block_size * sizeof(uint64_t));
    }

    Matrix gpu_path = std::move(matrices[0]);
    Matrix matrice_1;
    if (matrices.size() >= 2) {
        matrice_1.height = matrices[1].height;
        matrice_1.width = matrices[1].width;
        matrice_1.k = matrices[1].k;
        matrice_1.num_blocks = matrices[1].num_blocks;
        if (matrices[1].num_blocks > 0) {
            size_t block_size = matrices[1].num_blocks * (2 + k * k);
            matrice_1.blocks = std::make_unique<uint64_t[]>(block_size);
            std::memcpy(matrice_1.blocks.get(), matrices[1].blocks.get(), block_size * sizeof(uint64_t));
        }
    }

    if (matrices.size() >= 2) {
        gpu_path = multiply_pair_cuda(std::move(gpu_path), std::move(matrice_1), k);
    }

    bool use_gpu = !is_zero_matrix(gpu_path);
    cout<<rank<<" "<<use_gpu<<endl;
    if (use_gpu) {
        
        for (size_t i = 2; i < matrices.size(); ++i) {
            gpu_path = do_internal_mul(std::move(gpu_path), std::move(matrices[i]), k);
        }
        std::cout << rank << " completed" << std::endl;
        return gpu_path;
    } else {
        for (size_t i = 1; i < matrices.size(); ++i) {
            cpu_path = do_internal_mul(std::move(cpu_path), std::move(matrices[i]), k);
        }
        std::cout << rank << " completed (CPU fallback)" << std::endl;
        return cpu_path;
    }
}


/*
Matrix multiply_sequential_cuda(std::vector<Matrix> matrices, int k, int rank) {
    if (matrices.empty()) return Matrix(0, 0, k);

    Matrix result = std::move(matrices[0]);
    const size_t cuda_limit = std::min((size_t)2, matrices.size());
    for (size_t i = 1; i < cuda_limit; ++i) {
        result = multiply_pair_cuda(
            std::move(result), 
            std::move(matrices[i]), 
            k
        );
    }

    for (size_t i = cuda_limit; i < matrices.size(); ++i) {
        result = do_internal_mul(
            std::move(result), 
            std::move(matrices[i]), 
            k
        );
    }
    cout<<rank<<" "<<"completed"<<endl;
    

    return result;
}
*/
