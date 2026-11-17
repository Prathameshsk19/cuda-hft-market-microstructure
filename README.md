# CUDA-Accelerated Market Microstructure Analytics

A C++ and NVIDIA CUDA project exploring GPU acceleration for financial market microstructure analytics through parallel processing of synthetic market data.

The project implements market analytics on both CPU and GPU, benchmarks their performance at different event volumes, and investigates the trade-off between GPU computation speed and CPU–GPU data-transfer overhead.

## Project Overview

Financial market systems process large volumes of market data, including bid prices, ask prices, order quantities and executed trades.

This project explores how NVIDIA CUDA can accelerate computationally intensive market analytics by distributing calculations across GPU threads.

The implementation currently supports:

- Bid-ask spread calculation
- Mid-price calculation
- Order Book Imbalance (OBI)
- Market signal classification
- Volume Weighted Average Price (VWAP)
- CPU vs GPU performance benchmarking

**Project objective:** Evaluate GPU parallelisation for market microstructure analytics and understand the impact of memory transfers on end-to-end performance.

## System Architecture

```text
             Synthetic Market Data
                       |
                       v
              Market Data Generator
                       |
                       v
                 Input Arrays
                       |
              +--------+--------+
              |                 |
              v                 v
         CPU Analytics      GPU Analytics
              |                 |
              |            CUDA Kernels
              |                 |
              |            Parallel OBI
              |            Spread & Mid
              |            Signal Analysis
              |            VWAP Reduction
              |                 |
              +--------+--------+
                       |
                       v
               Results Comparison
                       |
                       v
               Performance Analysis
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | C++ |
| GPU programming | NVIDIA CUDA C++ |
| Compiler | NVCC |
| Parallel processing | CUDA kernels |
| CPU benchmarking | C++ chrono |
| Data generation | C++ random library |
| Operating system | Ubuntu 24.04 |
| CUDA Toolkit | 12.0 |
| GPU architecture | NVIDIA Maxwell (sm_50) |

## Hardware Configuration

The initial experiments were performed on:

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce 940MX |
| GPU Compute Capability | 5.0 |
| GPU Memory | 4 GB |
| CPU | Intel processor |
| Integrated GPU | Intel UHD Graphics 620 |
| Operating System | Ubuntu 24.04 |

The CUDA implementation is compiled specifically for the Maxwell GPU architecture using `-arch=sm_50`.

## Project Structure

```text
cuda-hft-market-microstructure/
│
├── src/
│   ├── hello_cuda.cu
│   ├── market_generator.cpp
│   └── hft_analytics.cu
│
├── include/
│
├── benchmarks/
│
├── python/
│
├── results/
│
├── .gitignore
└── README.md
```

## Market Analytics

### 1. Bid-Ask Spread

Measures the difference between the best ask and best bid prices.

```text
Spread = Ask Price - Bid Price
```

### 2. Mid-Price

Calculates the midpoint between the best bid and ask.

```text
Mid Price = (Bid Price + Ask Price) / 2
```

### 3. Order Book Imbalance (OBI)

Measures the relative difference between bid and ask quantities.

```text
OBI = (Bid Quantity - Ask Quantity)
      / (Bid Quantity + Ask Quantity)
```

Positive values indicate relatively greater bid-side quantity, while negative values indicate relatively greater ask-side quantity.

### 4. Market Signal Classification

Classifies market conditions into:

- Buy Pressure
- Sell Pressure
- Neutral

The classification is based on the implemented market analytics logic.

### 5. Volume Weighted Average Price (VWAP)

Calculates the volume-weighted average execution price.

```text
VWAP = Sum(Trade Price × Trade Quantity)
       / Sum(Trade Quantity)
```

The GPU implementation uses parallel partial reductions followed by a final reduction.

## CUDA Implementation

The project uses a Structure of Arrays (SoA) memory layout for market data.

Input arrays include:

- `bid[]`
- `ask[]`
- `bidQty[]`
- `askQty[]`
- `tradePrice[]`
- `tradeQty[]`

The main CUDA analytics kernel distributes market events across GPU threads.

The implementation also uses shared memory for partial VWAP reduction.

**Current configuration:**

- CUDA threads per block: 256
- Default workload: 1,048,576 events
- Data processing: Parallel GPU execution
- VWAP: Two-stage reduction

## Compilation and Execution

### Prerequisites

- Ubuntu/Linux
- NVIDIA GPU with CUDA support
- NVIDIA driver
- CUDA Toolkit
- C++ compiler

### Compile CUDA Analytics

```bash
nvcc -O2 -std=c++17 -arch=sm_50 \
src/hft_analytics.cu -o hft_analytics
```

### Run with Default Workload

```bash
./hft_analytics
```

### Run with Custom Workload Sizes

```bash
./hft_analytics 100000
./hft_analytics 500000
./hft_analytics 1048576
./hft_analytics 2097152
./hft_analytics 5242880
```

The workload can be adjusted through the command-line argument.

## Initial Benchmark Results

The following results were collected on the NVIDIA GeForce 940MX.

| Events | CPU Time (ms) | GPU Kernel (ms) | GPU End-to-End (ms) | Kernel Speedup |
|---:|---:|---:|---:|---:|
| 100,000 | 3.747 | 0.249 | 2.845 | 15.07× |
| 500,000 | 8.122 | 0.991 | 9.550 | 8.20× |
| 1,048,576 | 17.207 | 1.972 | 18.953 | 8.72× |
| 2,097,152 | 34.069 | 3.833 | 35.707 | 8.89× |
| 5,242,880 | 83.901 | 9.497 | 87.048 | 8.83× |

*Note: These are initial single-run measurements, not statistically averaged results.*

### Performance Observations

1. GPU kernel execution achieved approximately 8.8× speedup at larger workloads.
2. GPU execution time scaled effectively as the event count increased.
3. End-to-end execution was slower than CPU execution for most tested workloads.
4. CPU–GPU data transfers and associated overhead reduced the overall performance benefit.

This demonstrates an important distinction between kernel-level acceleration and application-level performance.

### Kernel Speedup vs End-to-End Speedup

**Kernel-only speedup:**

```text
CPU Execution Time / GPU Kernel Execution Time
```

**End-to-end speedup:**

```text
CPU Execution Time / GPU End-to-End Execution Time
```

The two measurements are reported separately because GPU kernel execution excludes data-transfer overhead, whereas end-to-end execution includes it.

## Verification

The implementation compares CPU and GPU analytics results.

Current test runs report:

```text
Verification: PASSED
```

This indicates that the implemented verification checks passed for the tested workloads.

## Development Roadmap

### Phase 1 — Core CUDA Analytics
- [x] CUDA environment setup
- [x] Synthetic market data generation
- [x] CPU analytics implementation
- [x] GPU analytics implementation
- [x] Parallel VWAP reduction
- [x] Variable workload support
- [x] Initial CPU vs GPU benchmarking

### Phase 2 — Benchmarking and Optimisation
- [ ] Repeated benchmark measurements
- [ ] Average, minimum and maximum execution time
- [ ] Throughput measurement (million events/second)
- [ ] CPU-to-GPU transfer time analysis
- [ ] CUDA block-size experimentation
- [ ] Memory access optimisation
- [ ] Pinned host memory experiments

### Phase 3 — Advanced Market Microstructure
- [ ] Order Flow Imbalance (OFI)
- [ ] Rolling VWAP
- [ ] Trade imbalance
- [ ] Price momentum
- [ ] Spread dynamics

### Phase 4 — Performance Visualisation
- [ ] CPU vs GPU execution-time graphs
- [ ] Throughput comparison
- [ ] Kernel speedup visualisation
- [ ] Memory-transfer overhead analysis
- [ ] Benchmark result exports

## Project Scope

This is an educational and experimental CUDA market analytics project.

It explores GPU parallelisation and performance characteristics using synthetic market data. It is not an exchange-grade trading system, production execution engine or validated trading strategy.

The objective is to develop practical experience in GPU programming, parallel algorithms, memory management and performance benchmarking.

## Author

**Prathamesh Kulkarni**

Master of Electrical and Electronics Engineering  
University of Melbourne

Interested in GPU Computing, Computer Architecture, FPGA Design, Low-Latency Systems and Hardware Acceleration.
