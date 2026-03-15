
#include "matrix.h"
#include "cuda_ops.h"
#include <mpi.h>
#include <set>
using namespace std;


void initialize_omp() {
    int num_procs = omp_get_num_procs(); 
    omp_set_num_threads(num_procs);  
    // cout<<num_procs<<" : procs"<<endl;
    // cout<<omp_get_num_threads<<" : threads" <<endl;
}

vector<char> serialize_matrix(const Matrix& mat) {
    const int elements_per_block = 2 + mat.k * mat.k; 
    const size_t header_size = 4 * sizeof(int); 
    const size_t data_size = mat.num_blocks * elements_per_block * sizeof(uint64_t);

    vector<char> buffer(header_size + data_size);
    char* ptr = buffer.data();

    memcpy(ptr, &mat.height, sizeof(int)); ptr += sizeof(int);
    memcpy(ptr, &mat.width, sizeof(int)); ptr += sizeof(int);
    memcpy(ptr, &mat.k, sizeof(int)); ptr += sizeof(int);
    memcpy(ptr, &mat.num_blocks, sizeof(int)); ptr += sizeof(int);

    memcpy(ptr, mat.blocks.get(), data_size);

    return buffer;
}

Matrix deserialize_matrix(const vector<char>& buffer) {
    Matrix mat;
    const char* ptr = buffer.data();

    auto read = [&](void* dest, size_t size) {
        memcpy(dest, ptr, size);
        ptr += size;
    };

    read(&mat.height, sizeof(int));
    read(&mat.width, sizeof(int));
    read(&mat.k, sizeof(int));
    read(&mat.num_blocks, sizeof(int));

    const int elements_per_block = 2 + mat.k * mat.k;
    mat.blocks.reset(new uint64_t[mat.num_blocks * elements_per_block]);
    read(mat.blocks.get(), mat.num_blocks * elements_per_block * sizeof(uint64_t));

    return mat;
}


void tree_reduction(Matrix& rank_result, int rank, int size, int k, int total_matrices) {
    // cout<<" i came here "<<rank<<endl;
    int steps = ceil(log2(size));
    int mask = 1;
    bool active = (rank < total_matrices);

    for (int step = 0; step < steps; ++step) {
        int partner = rank ^ mask;
        bool valid_partner = (partner < size) & (partner < total_matrices);

        if (active && valid_partner) {
            if (rank & mask) { // Sender
                if (rank_result.height > 0) {
                    vector<char> buffer = serialize_matrix(rank_result);
                    int buffer_size = buffer.size();
                    MPI_Send(&buffer_size, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
                    MPI_Send(buffer.data(), buffer_size, MPI_BYTE, partner, 0, MPI_COMM_WORLD);
                }
                active = false;
            } else { // Receiver
                int buffer_size;
                MPI_Status status;
                MPI_Recv(&buffer_size, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, &status);
                
                if (buffer_size > 0) {
                    vector<char> buffer(buffer_size);
                    MPI_Recv(buffer.data(), buffer_size, MPI_BYTE, partner, 0, MPI_COMM_WORLD, &status);
                    Matrix received = deserialize_matrix(buffer);

                    if (rank_result.height > 0) {
                        rank_result = do_internal_mul(
                            std::move(rank_result), 
                            std::move(received), 
                            k
                        );
                    } else {
                        rank_result = std::move(received);
                    }
                }
            }
        }

        mask <<= 1;
        MPI_Barrier(MPI_COMM_WORLD);
    }
}


Matrix do_internal_mul(Matrix A, Matrix B, int k){
    map<pair<int, int>, unique_ptr<uint64_t[]>> result_blocks;

    map<int,vector<pair<int,int>>>m1;
    map<int,vector<pair<int,int>>>m2;

    int ele_A = 2 + k*k;
    int ele_B = 2 + k*k;
    for(int i=0;i<A.num_blocks;i++){
        int a_offset = i * (2 + k * k);
        int l = A.blocks[a_offset];
        int r = A.blocks[a_offset + 1];
        m1[r].push_back({i,l});
    }

    for(int i=0;i<B.num_blocks;i++){
        int b_offset = i * (2 + k * k);
        int l = B.blocks[b_offset];
        int r = B.blocks[b_offset + 1];
        m2[l].push_back({i,r});
    }
    // cout<<m1.size()<<" "<<m2.size()<<endl;

    #pragma omp parallel
    {
        // cout<<"Total Threads " << omp_get_num_threads()<<endl;
        #pragma omp single 
        {
            for(const auto &e1 : m1){
                int z = e1.first;
                if (m2.find(z) == m2.end()) continue;
                for(const auto &v1 : e1.second){
                    int a_idx = v1.first;
                    int x = v1.second;
                    #pragma omp task
                    {
                        // cout<<" Task number "<<omp_get_thread_num()<<endl;
                        for(const auto &v2 : m2[z]){
                            int b_idx = v2.first;
                            int y = v2.second;
                            int a_offset = a_idx * ele_A + 2;
                            int b_offset = b_idx * ele_B + 2;
                            const uint64_t* a_blk = A.blocks.get() + a_offset;
                            const uint64_t* b_blk = B.blocks.get() + b_offset;

                            int a_rows = compute_block_dim(x, A.height, k);
                            int a_cols = compute_block_dim(z, A.width, k);
                            int b_cols = compute_block_dim(y, B.width, k);
                            unique_ptr<uint64_t[]> temp(new uint64_t[k * k]());
                            for (int i = 0; i < a_rows; i++) {
                                for (int kk = 0; kk < a_cols; kk++) {
                                    for (int j = 0; j < b_cols; j++) {
                                        temp[i * k + j] += a_blk[i * k + kk] * b_blk[kk * k + j];
                                    }
                                }
                            }

                            auto iter = result_blocks.end();
                            bool ok = true;

                            #pragma omp critical
                            {
                                iter = result_blocks.find({x, y});
                                if (iter == result_blocks.end()) {
                                    ok = false;
                                    result_blocks[{x, y}] = std::move(temp);
                                }
                            }

                            if (ok) {
                                for (int i = 0; i < k * k; i++) {
                                    #pragma omp atomic
                                    iter->second[i] += temp[i];
                                }
                            }
                        }


                    }
                }
            }
        }
    }

    Matrix result;
    result.height = A.height;
    result.width = B.width;
    result.k = k;
    result.num_blocks = result_blocks.size();
    int elements_per_block = 2 + k * k;
    result.blocks = make_unique<uint64_t[]>(result.num_blocks * elements_per_block);

    int block_idx = 0;
    for (const auto& entry : result_blocks) {
        int offset = block_idx * elements_per_block;
        result.blocks[offset] = entry.first.first;     // row
        result.blocks[offset + 1] = entry.first.second; // column
        memcpy(result.blocks.get() + offset + 2, entry.second.get(), k * k * sizeof(uint64_t));
        ++block_idx;
    }

    return result;

}


vector<Matrix> read_matrices_parallel(const string& folderPath, 
                                     int start_idx, int end_idx, int k) {
    vector<Matrix> matrices(end_idx - start_idx + 1);

    #pragma omp parallel for schedule(dynamic)
    for (int i = start_idx; i <= end_idx; ++i) {
        string path = folderPath + "/matrix" + to_string(i + 1);
        ifstream file;
        #pragma omp critical
        {
            file.open(path);
        }

        if (file.is_open()) {
            matrices[i - start_idx] = read_single_matrix(file, k);
        }
    }

    return matrices;
}



int main(int argc, char** argv) {
    // omp_set_num_threads(4);
    // MPI_Init(&argc, &argv);
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
      MPI_Abort(MPI_COMM_WORLD,1);
      return 1;
    }
    initialize_omp();
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // cout<<omp_get_num_threads()<<endl;
    string folderPath = argv[1];
    int N = 0, k = 0;

    if (rank == 0) {
        ifstream sizeFile(folderPath + "/size");
        sizeFile >> N >> k;
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int files_per_rank = N / size;
    int remainder = N % size;
    int start_file = (rank < remainder) ? rank * (files_per_rank + 1) 
                                      : remainder * (files_per_rank + 1) + (rank - remainder) * files_per_rank;
    int end_file = start_file + files_per_rank + (rank < remainder ? 1 : 0) - 1;


    vector<Matrix> local_matrices = read_matrices_parallel(folderPath, start_file, end_file, k);

    Matrix final_result;
    if(!local_matrices.empty()){
        final_result = multiply_sequential_cuda(std::move(local_matrices), k,rank);
    }


    local_matrices.clear();
    MPI_Barrier(MPI_COMM_WORLD); // Sync before communication


    tree_reduction(final_result, rank, size, k, N);

    if(rank == 0 && final_result.height>0){
        ofstream outfile("matrix");
        outfile << final_result.height<<" "<<final_result.width<<"\n";
        outfile<<final_result.num_blocks<<"\n";

        for(int i=0;i<final_result.num_blocks;i++) {
            int offset = i * (2 + k*k);
            int r = final_result.blocks[offset];
            int c = final_result.blocks[offset+1];
            uint64_t* data = final_result.blocks.get() + offset + 2;

            int actual_rows = compute_block_dim(r, final_result.height, final_result.k);
            int actual_cols = compute_block_dim(c, final_result.width, final_result.k);
            int element_count = actual_rows * actual_cols;
            
            outfile << r << " " << c << "\n";
            for (int i = 0; i < k; ++i) {
                for (int j = 0; j < k; ++j) {
                    if (i < actual_rows && j < actual_cols) {
                        outfile << data[i * k + j];
                    } else {
                        outfile << 0;  // padding zero
                    }

                    if (j < k - 1) {  // <-- fix here
                        outfile << " ";
                    }
                }
                outfile << "\n";
            }
        }
        outfile.close();
    }

    MPI_Finalize();
    return 0;
}