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

The implementation also
