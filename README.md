# Parallel Sparse Matrix Multiplication

<div align="center">

![Language](https://img.shields.io/badge/Language-C%2B%2B%20%7C%20CUDA-blue)
![Parallel](https://img.shields.io/badge/Parallelism-MPI%20%7C%20OpenMP%20%7C%20CUDA-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A high-performance distributed sparse matrix multiplication implementation using hybrid parallelization with MPI, OpenMP, and CUDA.

</div>

---

## 📋 Table of Contents

- [Overview](##overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Performance](#performance)
- [File Format](#file-format)
- [Algorithm Details](#algorithm-details)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements an efficient distributed sparse matrix multiplication algorithm that leverages:

- **GPU Acceleration**: CUDA kernels for parallel computation
- **Shared Memory Parallelism**: OpenMP for multi-threaded execution
- **Distributed Computing**: MPI for inter-process communication across multiple nodes

The implementation uses a **block-sparse matrix format** to minimize memory overhead while maximizing computational efficiency.

---

## ✨ Features

### Core Capabilities
- ✅ **Hybrid Parallelization**: Combines MPI, OpenMP, and CUDA
- ✅ **Block-Sparse Format**: Memory-efficient sparse matrix representation
- ✅ **Distributed Architecture**: Scales across multiple compute nodes
- ✅ **Dynamic Load Balancing**: Intelligent work distribution among processes
- ✅ **Parallel I/O**: Concurrent matrix file reading
- ✅ **Tree Reduction**: Binary tree-based result aggregation
- ✅ **Thread-Safe**: Critical sections and atomic operations for correctness

### Performance Optimizations
- GPU acceleration via CUDA kernels
- OpenMP task-based parallelism
- MPI binary tree reduction for minimal communication overhead
- Block-wise computation for cache efficiency

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│         MPI Master (Rank 0)                         │
│  ┌─────────────────────────────────────────────┐   │
│  │ 1. Read matrix metadata & broadcast         │   │
│  │ 2. Distribute matrices to worker processes  │   │
│  │ 3. Collect and write final result           │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
                        │
         ┌──────────────┼──────────────┐
         │              │              │
    ┌────▼──┐      ┌────▼──┐      ┌────▼──┐
    │Rank 1 │      │Rank 2 │      │Rank N │  (Worker Processes)
    │┌──────┼┐     │┌──────┼┐     │┌──────┼┐
    ││OpenMP││     ││OpenMP││     ││OpenMP││ (Shared Memory)
    ││ ┌──┐ ││     ││ ┌──┐ ││     ││ ┌──┐ ││
    ││ │CUDA│ ││  ││ │CUDA│ ││  ││ │CUDA│ ││ (GPU Acceleration)
    │└──────┘     └──────┘     └──────┘
    └────┬──┘      └────┬──┘      └────┬──┘
         │              │              │
         └──────────────┼──────────────┘
                Tree Reduction
                        │
                   ┌────▼────┐
                   │ Result  │
                   └─────────┘
```

### Execution Pipeline

1. **Initialization**
   - MPI initialization with thread support
   - OpenMP thread pool setup
   - Matrix size broadcasting

2. **Local Processing**
   - Parallel matrix file I/O (OpenMP)
   - GPU-accelerated sparse matrix multiplication (CUDA)

3. **Result Aggregation**
   - Binary tree reduction via MPI
   - Matrix serialization for communication

4. **Output**
   - Final result written by master process

---

## 📁 Project Structure

```
parallel-sparse-matrix-multiplication/
│
├── README.md                    # This file
├── Makefile                     # Build configuration
├── Assignment_4.pdf             # Problem statement
├── report.pdf                   # Analysis and results
│
├── main.cpp                     # Main program entry point
│                                # - MPI/OpenMP orchestration
│                                # - Matrix serialization
│                                # - Tree reduction logic
│
├── cuda_kernels.cu              # CUDA GPU kernels
│                                # - GPU-accelerated computations
│                                # - Kernel launches
│
├── cuda_ops.h                   # CUDA operations interface
│                                # - GPU function declarations
│
└── matrix.h                     # Data structures
                                 # - Sparse matrix definition
                                 # - Block representation
```

### Language Composition
- **CUDA**: 50.9% (GPU acceleration)
- **C++**: 41.0% (CPU logic and main implementation)
- **Makefile**: 8.1% (Build system)

---

## 📦 Prerequisites

### System Requirements
- **GPU**: NVIDIA CUDA-capable GPU (Compute Capability 3.0+)
- **CPU**: Multi-core processor
- **Memory**: Sufficient RAM for matrix data + GPU VRAM

### Software Dependencies

| Tool/Library | Version | Purpose |
|---|---|---|
| **CUDA Toolkit** | 10.0+ | GPU programming |
| **GCC/G++** | 7.0+ | C++ compilation |
| **OpenMPI** | 3.0+ | Distributed computing |
| **OpenMP** | 4.5+ | Shared memory parallelism |
| **Make** | 4.0+ | Build automation |

### Installation

#### Ubuntu/Debian
```bash
# CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-get update && sudo apt-get install cuda

# OpenMPI
sudo apt-get install libopenmpi-dev openmpi-bin

# Build tools
sudo apt-get install build-essential
```

#### macOS (with Homebrew)
```bash
# OpenMPI
brew install open-mpi

# GCC
brew install gcc

# Note: CUDA support for macOS is limited; check NVIDIA documentation
```

---

## 🚀 Usage

### Building the Project

```bash
# Clone the repository
git clone <repository-url>
cd parallel-sparse-matrix-multiplication

# Build the project
make

# Build with optimizations
make CXXFLAGS="-O3 -march=native"

# Clean build artifacts
make clean
```

### Running the Program

#### Single Node (Multiple Processes)
```bash
mpirun -np 4 ./main <matrix_folder_path>
```

#### Multiple Nodes (Distributed)
```bash
mpirun -np 8 -hostfile hostfile ./main <matrix_folder_path>
```

#### Example with Full Output
```bash
mpirun -np 4 ./main ./matrices/
# Output: matrix (result file in current directory)
```

### Input Format

**Directory Structure:**
```
matrices/
├── size              # Contains: N k (number of matrices and block size)
├── matrix1           # First sparse matrix
├── matrix2           # Second sparse matrix
└── matrixN           # Nth sparse matrix
```

**Matrix File Format:**
```
height width
num_blocks
[for each block]:
row col
data[0,0] data[0,1] ... data[0,k-1]
data[1,0] data[1,1] ... data[1,k-1]
...
data[k-1,0] ...
