# Experiment 5: Neural Network Layers in CUDA

## Overview

This experiment implements basic neural network layers using **CUDA C++** and compares their execution performance on the CPU and GPU.

The program performs a two-layer neural network forward pass using:
- Matrix multiplication with bias
- ReLU activation function
- Sigmoid activation function

The implementation is tested on an NVIDIA GeForce RTX 5060 Ti GPU.

## Objectives

- Understand CUDA programming and GPU parallel computing.
- Implement matrix multiplication with bias using CUDA.
- Implement ReLU and sigmoid activation functions.
- Understand CUDA kernels, threads and blocks.
- Compare CPU and GPU execution times.
- Verify CPU and GPU output correctness.

## Neural Network Architecture

| Layer | Dimensions |
|---|---:|
| Input | 256 × 128 |
| Hidden | 256 × 64 |
| Output | 256 × 10 |

**Forward-pass workflow:**

```text
Input Data
    |
    v
Matrix Multiplication + Bias
    |
    v
ReLU Activation
    |
    v
Hidden Layer
    |
    v
Matrix Multiplication + Bias
    |
    v
Sigmoid Activation
    |
    v
Output
```

## Technologies Used

- C++
- NVIDIA CUDA
- CUDA Toolkit 13.1
- NVIDIA GeForce RTX 5060 Ti
- Windows 11
- Visual Studio C++ Build Tools

## Project Structure

```text
Experiment5_CUDA/
│
├── neural_network.cu
├── README.md
├── results.md
├── experiment-report.md
├── .gitignore
│
└── screenshots/
    ├── gpu-detection.png
    ├── kernel-tests.png
    └── final-output.png
```

## Implementation Details

The experiment is implemented in eight phases:

1. Header files, CUDA error checking and constants.
2. GPU kernels for matrix multiplication, ReLU and sigmoid.
3. CPU implementation of neural network operations.
4. GPU forward-pass execution.
5. Individual CUDA kernel testing.
6. Main function, GPU detection and CPU timing.
7. GPU memory allocation, data transfer and timing.
8. CPU/GPU output comparison, speedup calculation and memory cleanup.

## How to Run

### 1. Check CUDA installation

Open the Visual Studio Developer Command Prompt and run:

```cmd
nvcc --version
```

Check GPU information:

```cmd
nvidia-smi
```

### 2. Navigate to the project folder

```cmd
cd path\to\Experiment5_CUDA
```

### 3. Compile the CUDA program

```cmd
nvcc -O2 -gencode arch=compute_120,code=sm_120 neural_network.cu -o neural_network.exe
```

### 4. Execute the program

```cmd
neural_network.exe
```

## Experimental Results

The program was successfully executed on an NVIDIA GeForce RTX 5060 Ti.

| Performance Metric | Result |
|---|---:|
| CPU compute time | 1.198105 ms |
| GPU kernel-only time | 0.045510 ms |
| GPU end-to-end time | 0.254215 ms |
| Kernel-only speedup | 26.325961× |
| End-to-end speedup | 4.712960× |

### Kernel Testing

| CUDA Kernel | Status |
|---|---|
| Matrix multiplication + bias | PASSED |
| ReLU activation | PASSED |
| Sigmoid activation | PASSED |

### Output Validation

- CPU and GPU outputs: **MATCH**
- Maximum absolute output difference: **0.000000**
- Comparison tolerance: **0.0001**

For detailed measurements and calculations, see [results.md](results.md).

## Screenshots

Add your experiment screenshots to the `screenshots/` folder:

1. GPU detection
2. Individual CUDA kernel test results
3. Final CPU and GPU performance output

## Learning Outcomes

- Understood how CUDA executes parallel operations on an NVIDIA GPU.
- Implemented neural network operations using CUDA kernels.
- Learned GPU memory allocation and host-device data transfer.
- Compared CPU and GPU performance.
- Verified output correctness between CPU and GPU implementations.

## Conclusion

The experiment successfully demonstrates the implementation of neural network layers using CUDA. All three CUDA kernels passed their tests, and the GPU implementation achieved a kernel-only speedup of approximately **26.33×** and an end-to-end speedup of approximately **4.71×** compared with CPU execution.

**Experiment Status: Successfully Completed**
