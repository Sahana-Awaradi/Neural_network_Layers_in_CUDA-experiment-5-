# Experiment 5 – CUDA Neural Network Layers: Results

## 1. GPU Information

| Parameter | Details |
|---|---|
| GPU Model | NVIDIA GeForce RTX 5060 Ti |
| CUDA Toolkit | 13.1 |
| NVIDIA Driver | 610.88 |
| Programming Language | C++ / CUDA |
| Batch Size | 256 |
| Input Size | 128 |
| Hidden Layer Size | 64 |
| Output Size | 10 |
| Repetitions | 20 |

## 2. Individual CUDA Kernel Testing

All three CUDA kernels were tested individually before executing the complete neural network.

| Test | Result |
|---|---|
| Matrix multiplication + bias | PASSED |
| ReLU activation | PASSED |
| Sigmoid activation | PASSED |

**Observation:** All three CUDA kernels executed successfully and produced the expected results.

## 3. Neural Network Dimensions

| Layer | Dimensions |
|---|---|
| Input layer | 256 × 128 |
| Hidden layer | 256 × 64 |
| Output layer | 256 × 10 |

The neural network performs two matrix multiplication operations, followed by ReLU and sigmoid activation functions.

## 4. CPU and GPU Performance Results

The following values were obtained from the successful execution on NVIDIA GeForce RTX 5060 Ti.

| Performance Metric | Observed Value |
|---|---:|
| Average CPU compute time | 1.198105 ms |
| Average GPU kernel-only time | 0.045510 ms |
| Average GPU end-to-end time | 0.254215 ms |
| Kernel-only speedup | 26.325961× |
| End-to-end speedup | 4.712960× |

## 5. Speedup Calculations

### A. Kernel-only Speedup

Formula:

Speedup = CPU Execution Time / GPU Kernel-only Time

Calculation:

= 1.198105 / 0.045510

= **26.325961×**

### B. End-to-end Speedup

Formula:

Speedup = CPU Execution Time / GPU End-to-end Time

Calculation:

= 1.198105 / 0.254215

= **4.712960×**

**Interpretation:**

- Kernel-only speedup measures the performance of GPU computation.
- End-to-end speedup includes data transfers and synchronization overhead.
- GPU execution was faster than CPU execution for this workload.

## 6. CPU and GPU Output Validation

| Validation Metric | Result |
|---|---|
| Maximum absolute output difference | 0.000000 |
| CPU and GPU comparison | MATCH |
| Comparison tolerance | 0.0001 |

The maximum absolute difference between corresponding CPU and GPU output elements was zero at the displayed precision.

**Result:** CPU and GPU outputs matched within the specified tolerance.

## 7. First 10 GPU Output Values

```text
0.498196
0.503101
0.508006
0.512911
0.517814
0.497729
0.502638
0.507548
0.512457
0.517364
```

These values are generated after the final sigmoid activation function.

## 8. Performance Analysis

The experiment demonstrates that GPU parallel processing reduces computation time for the implemented neural network.

The GPU kernel-only execution time was 0.045510 ms, compared with the CPU execution time of 1.198105 ms.

The end-to-end GPU execution time was 0.254215 ms because it includes additional memory transfer and synchronization overhead.

## 9. Final Observation

1. All three CUDA kernels passed their individual tests.
2. The complete neural network forward pass executed successfully.
3. GPU kernel-only execution was approximately 26.33 times faster than CPU execution.
4. End-to-end GPU execution was approximately 4.71 times faster.
5. CPU and GPU outputs matched within the specified tolerance.
6. The maximum absolute output difference was 0.000000.

## 10. Conclusion

The CUDA implementation of matrix multiplication with bias, ReLU and sigmoid activation functions was successfully executed.

The NVIDIA GeForce RTX 5060 Ti GPU achieved a kernel-only speedup of 26.33× and an end-to-end speedup of 4.71× compared with CPU execution.

The output comparison confirmed that the CPU and GPU implementations produced matching results at the displayed precision.

**Final Status: SUCCESS**

All kernel tests passed and the CPU/GPU output validation was successful.
