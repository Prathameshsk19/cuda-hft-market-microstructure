#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); } } while (0)

constexpr int THREADS = 256;

enum Signal {
    SELL_PRESSURE = -1,
    NEUTRAL = 0,
    BUY_PRESSURE = 1
};

// Structure of Arrays (SoA)
struct MarketData {
    std::vector<float> bid;
    std::vector<float> ask;
    std::vector<uint32_t> bidQty;
    std::vector<uint32_t> askQty;
    std::vector<float> tradePrice;
    std::vector<uint32_t> tradeQty;
};

struct Results {
    std::vector<float> spread;
    std::vector<float> mid;
    std::vector<float> obi;
    std::vector<int> signal;

    float vwap = 0.0f;
};

// --------------------------------------------------
// 1. Generate synthetic market data
// --------------------------------------------------

MarketData generateMarketData(int N)
{
    MarketData data;

    data.bid.resize(N);
    data.ask.resize(N);
    data.bidQty.resize(N);
    data.askQty.resize(N);
    data.tradePrice.resize(N);
    data.tradeQty.resize(N);

    std::mt19937 rng(42);

    std::uniform_real_distribution<float> price(99.0f, 101.0f);
    std::uniform_int_distribution<uint32_t> qty(100, 1000);

    for (int i = 0; i < N; i++)
    {
        float mid = price(rng);

        data.bid[i] = mid - 0.01f;
        data.ask[i] = mid + 0.01f;

        data.bidQty[i] = qty(rng);
        data.askQty[i] = qty(rng);

        data.tradePrice[i] = mid;
        data.tradeQty[i] = qty(rng);
    }

    return data;
}

// --------------------------------------------------
// 2. CPU reference implementation
// --------------------------------------------------

Results runCPU(const MarketData& data, int N)
{
    Results r;

    r.spread.resize(N);
    r.mid.resize(N);
    r.obi.resize(N);
    r.signal.resize(N);

    double totalNotional = 0.0;
    double totalVolume = 0.0;

    for (int i = 0; i < N; i++)
    {
        r.spread[i] = data.ask[i] - data.bid[i];

        r.mid[i] = (data.ask[i] + data.bid[i]) / 2.0f;

        float denominator =
            static_cast<float>(data.bidQty[i]) +
            static_cast<float>(data.askQty[i]);

        r.obi[i] =
            (static_cast<float>(data.bidQty[i]) -
             static_cast<float>(data.askQty[i])) /
            denominator;

        if (r.obi[i] > 0.5f && r.spread[i] < 0.03f)
            r.signal[i] = BUY_PRESSURE;

        else if (r.obi[i] < -0.5f && r.spread[i] < 0.03f)
            r.signal[i] = SELL_PRESSURE;

        else
            r.signal[i] = NEUTRAL;

        totalNotional +=
            static_cast<double>(data.tradePrice[i]) *
            data.tradeQty[i];

        totalVolume += data.tradeQty[i];
    }

    r.vwap = totalNotional / totalVolume;

    return r;
}

// --------------------------------------------------
// 3. GPU analytics kernel
// --------------------------------------------------

__global__ void marketAnalytics(
    const float* bid,
    const float* ask,
    const uint32_t* bidQty,
    const uint32_t* askQty,
    float* spread,
    float* mid,
    float* obi,
    int* signal,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    spread[idx] = ask[idx] - bid[idx];

    mid[idx] = (ask[idx] + bid[idx]) / 2.0f;

    float denominator =
        static_cast<float>(bidQty[idx]) +
        static_cast<float>(askQty[idx]);

    obi[idx] =
        (static_cast<float>(bidQty[idx]) -
         static_cast<float>(askQty[idx])) / denominator;

    if (obi[idx] > 0.5f && spread[idx] < 0.03f)
        signal[idx] = BUY_PRESSURE;

    else if (obi[idx] < -0.5f && spread[idx] < 0.03f)
        signal[idx] = SELL_PRESSURE;

    else
        signal[idx] = NEUTRAL;
}

// --------------------------------------------------
// 4. Parallel VWAP reduction
// --------------------------------------------------

__global__ void partialVWAP(
    const float* price,
    const uint32_t* volume,
    float* partialNotional,
    float* partialVolume,
    int N)
{
    __shared__ float sharedNotional[THREADS];
    __shared__ float sharedVolume[THREADS];

    int tid = threadIdx.x;

    int idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    float localNotional = 0.0f;
    float localVolume = 0.0f;

    for (int i = idx; i < N; i += stride)
    {
        localNotional += price[i] * volume[i];
        localVolume += volume[i];
    }

    sharedNotional[tid] = localNotional;
    sharedVolume[tid] = localVolume;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sharedNotional[tid] += sharedNotional[tid + s];
            sharedVolume[tid] += sharedVolume[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        partialNotional[blockIdx.x] = sharedNotional[0];
        partialVolume[blockIdx.x] = sharedVolume[0];
    }
}

// Final reduction of block-level partial sums
__global__ void finalVWAP(
    const float* partialNotional,
    const float* partialVolume,
    float* totalNotional,
    float* totalVolume,
    int numBlocks)
{
    __shared__ float sumN[THREADS];
    __shared__ float sumV[THREADS];

    int tid = threadIdx.x;

    float n = 0.0f;
    float v = 0.0f;

    for (int i = tid; i < numBlocks; i += blockDim.x)
    {
        n += partialNotional[i];
        v += partialVolume[i];
    }

    sumN[tid] = n;
    sumV[tid] = v;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sumN[tid] += sumN[tid + s];
            sumV[tid] += sumV[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        *totalNotional = sumN[0];
        *totalVolume = sumV[0];
    }
}

// --------------------------------------------------
// 5. GPU execution and benchmarking
// --------------------------------------------------

int main(int argc, char* argv[])
{
    // Default: 1,048,576 events
    const int N = (argc > 1)
        ? std::atoi(argv[1])
        : (1 << 20);

    // Validate event count
    if (N <= 0 || N > 10000000)
    {
        std::cerr << "Error: Event count must be between "
                  << "1 and 10,000,000.\n";
        return 1;
    }

    std::cout << "\nHFT CUDA Market Analytics\n";
    std::cout << "Events: " << N << "\n\n";

    MarketData data = generateMarketData(N);

    // CPU benchmark
    auto cpuStart = std::chrono::high_resolution_clock::now();

    Results cpuResults = runCPU(data, N);

    auto cpuEnd = std::chrono::high_resolution_clock::now();

    double cpuTime =
        std::chrono::duration<double, std::milli>(
            cpuEnd - cpuStart).count();

    // Allocate GPU memory
    float *d_bid, *d_ask;
    float *d_price;
    float *d_spread, *d_mid, *d_obi;
    float *d_partialN, *d_partialV;
    float *d_totalN, *d_totalV;

    uint32_t *d_bidQty, *d_askQty, *d_volume;
    int *d_signal;

    CUDA_CHECK(cudaMalloc((void**)&d_bid, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_ask, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_price, N * sizeof(float)));

    CUDA_CHECK(cudaMalloc((void**)&d_bidQty, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_askQty, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_volume, N * sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc((void**)&d_spread, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_mid, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_obi, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_signal, N * sizeof(int)));

    int blocks = (N + THREADS - 1) / THREADS;

    CUDA_CHECK(cudaMalloc((void**)&d_partialN, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_partialV, blocks * sizeof(float)));

    CUDA_CHECK(cudaMalloc((void**)&d_totalN, sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_totalV, sizeof(float)));

    Results gpuResults;

    gpuResults.spread.resize(N);
    gpuResults.mid.resize(N);
    gpuResults.obi.resize(N);
    gpuResults.signal.resize(N);

    float totalN, totalV;

    // GPU timing events
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto gpuStart = std::chrono::high_resolution_clock::now();

    // Transfer market data to GPU
    CUDA_CHECK(cudaMemcpy(d_bid, data.bid.data(),
        N * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_ask, data.ask.data(),
        N * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_bidQty, data.bidQty.data(),
        N * sizeof(uint32_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_askQty, data.askQty.data(),
        N * sizeof(uint32_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_price, data.tradePrice.data(),
        N * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_volume, data.tradeQty.data(),
        N * sizeof(uint32_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start));

    // Launch analytics kernels
    marketAnalytics<<<blocks, THREADS>>>(
        d_bid, d_ask,
        d_bidQty, d_askQty,
        d_spread, d_mid, d_obi, d_signal, N);

    partialVWAP<<<blocks, THREADS>>>(
        d_price, d_volume,
        d_partialN, d_partialV, N);

    finalVWAP<<<1, THREADS>>>(
        d_partialN, d_partialV,
        d_totalN, d_totalV, blocks);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernelTime = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &kernelTime, start, stop));

    // Copy results back to CPU
    CUDA_CHECK(cudaMemcpy(gpuResults.spread.data(), d_spread,
        N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(gpuResults.mid.data(), d_mid,
        N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(gpuResults.obi.data(), d_obi,
        N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(gpuResults.signal.data(), d_signal,
        N * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(&totalN, d_totalN,
        sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(&totalV, d_totalV,
        sizeof(float), cudaMemcpyDeviceToHost));

    gpuResults.vwap = totalN / totalV;

    auto gpuEnd = std::chrono::high_resolution_clock::now();

    double gpuTotalTime =
        std::chrono::duration<double, std::milli>(
            gpuEnd - gpuStart).count();

    // --------------------------------------------------
    // 6. Results and verification
    // --------------------------------------------------

    double avgSpread = 0.0;
    double avgOBI = 0.0;

    int buy = 0, sell = 0, neutral = 0;

    bool correct = true;

    for (int i = 0; i < N; i++)
    {
        avgSpread += gpuResults.spread[i];
        avgOBI += gpuResults.obi[i];

        if (gpuResults.signal[i] == BUY_PRESSURE) buy++;
        else if (gpuResults.signal[i] == SELL_PRESSURE) sell++;
        else neutral++;

        if (std::fabs(gpuResults.spread[i] -
                      cpuResults.spread[i]) > 1e-5f ||
            std::fabs(gpuResults.obi[i] -
                      cpuResults.obi[i]) > 1e-5f ||
            gpuResults.signal[i] != cpuResults.signal[i])
        {
            correct = false;
            break;
        }
    }

    avgSpread /= N;
    avgOBI /= N;

    std::cout << "\n========== MARKET ANALYTICS ==========\n";

    std::cout << "Average Spread: " << avgSpread << "\n";
    std::cout << "Average OBI: " << avgOBI << "\n";
    std::cout << "VWAP (GPU): " << gpuResults.vwap << "\n";
    std::cout << "VWAP (CPU): " << cpuResults.vwap << "\n";

    std::cout << "\nMarket Signals:\n";
    std::cout << "Buy Pressure: " << buy << "\n";
    std::cout << "Sell Pressure: " << sell << "\n";
    std::cout << "Neutral: " << neutral << "\n";

    std::cout << "\n========== BENCHMARK ==========\n";

    std::cout << "CPU Time: " << cpuTime << " ms\n";
    std::cout << "GPU Kernel Time: " << kernelTime << " ms\n";
    std::cout << "GPU End-to-End Time: "
              << gpuTotalTime << " ms\n";

    std::cout << "Kernel-only Speedup: "
              << cpuTime / kernelTime << "x\n";

    std::cout << "End-to-End Speedup: "
              << cpuTime / gpuTotalTime << "x\n";

    std::cout << "\nVerification: "
              << (correct ? "PASSED" : "FAILED") << "\n";

    // Cleanup
    cudaFree(d_bid);
    cudaFree(d_ask);
    cudaFree(d_price);
    cudaFree(d_bidQty);
    cudaFree(d_askQty);
    cudaFree(d_volume);
    cudaFree(d_spread);
    cudaFree(d_mid);
    cudaFree(d_obi);
    cudaFree(d_signal);
    cudaFree(d_partialN);
    cudaFree(d_partialV);
    cudaFree(d_totalN);
    cudaFree(d_totalV);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return correct ? 0 : 1;
}
