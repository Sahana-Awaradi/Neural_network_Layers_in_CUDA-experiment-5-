# Experiment 5: Neural Network Layers in CUDA

## 1. Aim

To implement basic neural network layers using CUDA and compare their execution performance on CPU and GPU.

The experiment implements:
- Matrix multiplication with bias
- ReLU activation function
- Sigmoid activation function

## 2. Introduction

CUDA (Compute Unified Device Architecture) is a parallel computing platform developed by NVIDIA. It allows programmers to execute general-purpose computations on NVIDIA GPUs.

In this experiment, we implement a simple two-layer neural network using CUDA. The same computations are performed on CPU and GPU to compare execution time and verify output correctness.

## 3. Objectives

1. Understand GPU parallel computing using CUDA.
2. Implement matrix multiplication with bias.
3. Implement ReLU and sigmoid activation functions.
4. Understand CUDA kernels, threads and blocks.
5. Compare CPU and GPU execution performance.
6. Calculate speedup and validate the outputs.

## 4. Hardware and Software Requirements

### Hardware
- NVIDIA GeForce RTX 5060 Ti GPU
- Computer with Windows operating system

### Software
- Windows 11
- NVIDIA CUDA Toolkit 13.4
- NVIDIA GPU Driver
- Visual Studio 2026 Build Tools
- C++ Compiler (MSVC)
- CUDA Compiler (NVCC)

## 5. Neural Network Architecture

The experiment uses a two-layer feed-forward neural network.

### Network dimensions

| Parameter | Value |
|---|---:|
| Batch size | 256 |
| Input features | 128 |
| Hidden layer size | 64 |
| Output size | 10 |
| Repetitions | 20 |

### Architecture

Input (256 × 128)

↓

Matrix Multiplication + Bias

↓

ReLU Activation

↓

Hidden Layer (256 × 64)

↓

Matrix Multiplication + Bias

↓

Sigmoid Activation

↓

Output (256 × 10)

## 6. Mathematical Operations

### A. Matrix Multiplication with Bias

Formula:

C = A × B + bias

Each output element is calculated by multiplying corresponding matrix elements, summing them and adding the bias.

### B. ReLU Activation

Formula:

ReLU(x) = max(0, x)

It converts negative values into zero and keeps positive values unchanged.

Example:

Input: -2, 0, 3, -4, 5

Output: 0, 0, 3, 0, 5

### C. Sigmoid Activation

Formula:

Sigmoid(x) = 1 / (1 + e^(-x))

It converts values into a range between 0 and 1.

## 7. Implementation

The program is implemented in eight phases.

| Phase | Description |
|---|---|
| Phase 1 | Header files, CUDA error checking and constants |
| Phase 2 | GPU kernels for matrix multiplication, ReLU and sigmoid |
| Phase 3 | CPU implementation of neural network operations |
| Phase 4 | GPU forward-pass execution |
| Phase 5 | Individual CUDA kernel testing |
| Phase 6 | Main function, GPU detection and CPU timing |
| Phase 7 | GPU memory allocation, data transfer and timing |
| Phase 8 | CPU/GPU comparison, speedup calculation and memory cleanup |

## 8. Execution Procedure

1. Install NVIDIA CUDA Toolkit and Visual Studio Build Tools.
2. Verify GPU availability using `nvidia-smi`.
3. Verify CUDA compiler using `nvcc --version`.
4. Create the CUDA source file `neural_network.cu`.
5. Implement all eight phases.
6. Open Visual Studio Developer Command Prompt.
7. Navigate to the project directory.
8. Compile the CUDA program.

Compilation command:

```cmd
nvcc -O2 -gencode arch=compute_120,code=sm_120 neural_network.cu -o neural_network.exe
```

9. Execute the program.

```cmd
neural_network.exe
```

10. Observe CPU and GPU execution times.
11. Compare CPU and GPU outputs.
12. Calculate speedup.

## 9. Experimental Results

### GPU Information

GPU detected: NVIDIA GeForce RTX 5060 Ti

### Individual Kernel Testing

| CUDA Kernel | Result |
|---|---|
| Matrix multiplication + bias | PASSED |
| ReLU | PASSED |
| Sigmoid | PASSED |

All three individual CUDA kernel tests passed successfully.

### Forward-Pass Dimensions

| Layer | Dimensions |
|---|---|
| Input | 256 × 128 |
| Hidden layer | 256 × 64 |
| Output | 256 × 10 |

### Performance Comparison

| Measurement | Observed Result |
|---|---:|
| Average CPU compute time | 1.198105 ms |
| Average GPU kernel-only time | 0.045510 ms |
| Average GPU end-to-end time | 0.254215 ms |
| Kernel-only speedup | 26.325961× |
| End-to-end speedup | 4.712960× |
| Maximum absolute output difference | 0.000000 |

### Output Validation

CPU and GPU outputs:

MATCH (within tolerance)

The maximum absolute difference between CPU and GPU outputs was 0.000000 at the displayed precision.

### First 10 GPU Output Values

```text
0.498196 0.503101 0.508006
0.512911 0.517814 0.497729
0.502638 0.507548 0.512457 0.517364
```

## 10. Speedup Calculation

Speedup is calculated as:

Speedup = CPU execution time / GPU execution time

### Kernel-only speedup

= 1.198105 / 0.045510

= 26.325961×

### End-to-end speedup

= 1.198105 / 0.254215

= 4.712960×

The kernel-only speedup measures GPU computation performance, whereas end-to-end speedup includes data transfer and synchronization overhead.

## 11. Observations

1. All three CUDA kernels executed successfully.
2. GPU parallel processing reduced computation time.
3. GPU kernel-only execution was faster than CPU execution.
4. End-to-end execution took longer than kernel-only execution because of data transfer and synchronization.
5. CPU and GPU outputs matched within the specified tolerance.
6. The experiment demonstrated the performance benefits of GPU parallel computing for this workload.

## 12. Conclusion

The neural network layers were successfully implemented using CUDA. Matrix multiplication with bias, ReLU and sigmoid activation functions were executed successfully on the NVIDIA GeForce RTX 5060 Ti GPU.

The CPU execution time was 1.198105 ms, while GPU kernel-only execution took 0.045510 ms. The kernel-only speedup was approximately 26.33×, and end-to-end speedup was approximately 4.71×.

The maximum absolute output difference was 0.000000, confirming that CPU and GPU outputs matched at the displayed precision.

Thus, the experiment demonstrated successful CUDA implementation, output correctness and improved execution performance through GPU parallel computing.

---

**Experiment Status:** Successfully Completed

**GPU:** NVIDIA GeForce RTX 5060 Ti

**Programming Language:** C++ / CUDA
