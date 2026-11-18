#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <string>

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
// 5. Automated benchmarking utilities
// --------------------------------------------------

constexpr int WARMUP_ITERATIONS = 3;
constexpr int BENCHMARK_ITERATIONS = 20;

struct DeviceBuffers {
    float *bid = nullptr, *ask = nullptr, *price = nullptr;
    uint32_t *bidQty = nullptr, *askQty = nullptr, *volume = nullptr;

    float *spread = nullptr, *mid = nullptr, *obi = nullptr;
    int *signal = nullptr;

    float *partialN = nullptr, *partialV = nullptr;
    float *totalN = nullptr, *totalV = nullptr;
};

struct Timing {
    double h2d = 0.0;
    double kernel = 0.0;
    double d2h = 0.0;
    double e2e = 0.0;
};

struct Statistics {
    double mean;
    double median;
    double minimum;
    double maximum;
};

Statistics calculateStatistics(const std::vector<double>& values)
{
    Statistics s{};

    if (values.empty())
        return s;

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    double sum = 0.0;

    for (double value : values)
        sum += value;

    s.mean = sum / values.size();
    s.minimum = sorted.front();
    s.maximum = sorted.back();

    size_t n = sorted.size();

    if (n % 2 == 0)
        s.median = (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    else
        s.median = sorted[n / 2];

    return s;
}

void allocateDeviceMemory(DeviceBuffers& d, int N, int blocks)
{
    CUDA_CHECK(cudaMalloc((void**)&d.bid, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d.ask, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d.price, N * sizeof(float)));

    CUDA_CHECK(cudaMalloc((void**)&d.bidQty, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&d.askQty, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&d.volume, N * sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc((void**)&d.spread, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d.mid, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d.obi, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d.signal, N * sizeof(int)));

    CUDA_CHECK(cudaMalloc((void**)&d.partialN, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d.partialV, blocks * sizeof(float)));

    CUDA_CHECK(cudaMalloc((void**)&d.totalN, sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d.totalV, sizeof(float)));
}

void freeDeviceMemory(DeviceBuffers& d)
{
    cudaFree(d.bid);
    cudaFree(d.ask);
    cudaFree(d.price);

    cudaFree(d.bidQty);
    cudaFree(d.askQty);
    cudaFree(d.volume);

    cudaFree(d.spread);
    cudaFree(d.mid);
    cudaFree(d.obi);
    cudaFree(d.signal);

    cudaFree(d.partialN);
    cudaFree(d.partialV);

    cudaFree(d.totalN);
    cudaFree(d.totalV);
}

// One complete GPU iteration.
// Allocation is deliberately excluded from this function.
Timing runGPUIteration(
    const MarketData& data,
    Results& results,
    DeviceBuffers& d,
    int N,
    int blocks,
    cudaEvent_t start,
    cudaEvent_t stop)
{
    Timing timing;

    auto e2eStart = std::chrono::steady_clock::now();

    // -------------------------------
    // Host-to-device transfer
    // -------------------------------

    auto h2dStart = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaMemcpy(d.bid, data.bid.data(),
        N * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d.ask, data.ask.data(),
        N * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d.bidQty, data.bidQty.data(),
        N * sizeof(uint32_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d.askQty, data.askQty.data(),
        N * sizeof(uint32_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d.price, data.tradePrice.data(),
        N * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d.volume, data.tradeQty.data(),
        N * sizeof(uint32_t), cudaMemcpyHostToDevice));

    auto h2dEnd = std::chrono::steady_clock::now();

    timing.h2d =
        std::chrono::duration<double, std::milli>(
            h2dEnd - h2dStart).count();

    // -------------------------------
    // GPU kernel execution
    // -------------------------------

    CUDA_CHECK(cudaEventRecord(start));

    marketAnalytics<<<blocks, THREADS>>>(
        d.bid, d.ask,
        d.bidQty, d.askQty,
        d.spread, d.mid, d.obi, d.signal,
        N
    );

    partialVWAP<<<blocks, THREADS>>>(
        d.price, d.volume,
        d.partialN, d.partialV,
        N
    );

    finalVWAP<<<1, THREADS>>>(
        d.partialN, d.partialV,
        d.totalN, d.totalV,
        blocks
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernelMs = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &kernelMs, start, stop));

    timing.kernel = kernelMs;

    // -------------------------------
    // Device-to-host transfer
    // -------------------------------

    auto d2hStart = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaMemcpy(results.spread.data(), d.spread,
        N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(results.mid.data(), d.mid,
        N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(results.obi.data(), d.obi,
        N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(results.signal.data(), d.signal,
        N * sizeof(int), cudaMemcpyDeviceToHost));

    float totalN = 0.0f;
    float totalV = 0.0f;

    CUDA_CHECK(cudaMemcpy(&totalN, d.totalN,
        sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(&totalV, d.totalV,
        sizeof(float), cudaMemcpyDeviceToHost));

    results.vwap = totalN / totalV;

    auto d2hEnd = std::chrono::steady_clock::now();

    timing.d2h =
        std::chrono::duration<double, std::milli>(
            d2hEnd - d2hStart).count();

    auto e2eEnd = std::chrono::steady_clock::now();

    timing.e2e =
        std::chrono::duration<double, std::milli>(
            e2eEnd - e2eStart).count();

    return timing;
}

bool verifyResults(
    const Results& cpu,
    const Results& gpu,
    int N)
{
    constexpr float ARRAY_TOLERANCE = 1e-5f;
    constexpr float VWAP_TOLERANCE = 0.01f;

    for (int i = 0; i < N; i++)
    {
        if (std::fabs(cpu.spread[i] - gpu.spread[i])
                > ARRAY_TOLERANCE ||
            std::fabs(cpu.mid[i] - gpu.mid[i])
                > ARRAY_TOLERANCE ||
            std::fabs(cpu.obi[i] - gpu.obi[i])
                > ARRAY_TOLERANCE ||
            cpu.signal[i] != gpu.signal[i])
        {
            std::cerr << "Verification failed at event "
                      << i << "\n";
            return false;
        }
    }

    if (std::fabs(cpu.vwap - gpu.vwap) > VWAP_TOLERANCE)
    {
        std::cerr << "VWAP verification failed. CPU: "
                  << cpu.vwap << ", GPU: " << gpu.vwap << "\n";
        return false;
    }

    return true;
}

// --------------------------------------------------
// 6. Benchmark one workload
// --------------------------------------------------

bool benchmarkWorkload(int N, std::ofstream& csv)
{
    std::cout << "\n========================================\n";
    std::cout << "Benchmarking " << N << " events\n";
    std::cout << "========================================\n";

    MarketData data = generateMarketData(N);

    Results cpuResults;
    Results gpuResults;

    // -------------------------------
    // GPU setup and allocation
    // -------------------------------

    int blocks = (N + THREADS - 1) / THREADS;

    DeviceBuffers d;

    gpuResults.spread.resize(N);
    gpuResults.mid.resize(N);
    gpuResults.obi.resize(N);
    gpuResults.signal.resize(N);

    cudaEvent_t start, stop;

    auto setupStart = std::chrono::steady_clock::now();

    allocateDeviceMemory(d, N, blocks);

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto setupEnd = std::chrono::steady_clock::now();

    double setupMs =
        std::chrono::duration<double, std::milli>(
            setupEnd - setupStart).count();

    // -------------------------------
    // Warm-up iterations
    // -------------------------------

    std::cout << "Running " << WARMUP_ITERATIONS
              << " warm-up iterations...\n";

    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        Results warmupCPU = runCPU(data, N);

        runGPUIteration(
            data, gpuResults, d, N, blocks, start, stop
        );
    }

    // -------------------------------
    // Measured CPU iterations
    // -------------------------------

    std::vector<double> cpuTimes;

    std::cout << "Running " << BENCHMARK_ITERATIONS
              << " CPU iterations...\n";

    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        auto cpuStart = std::chrono::steady_clock::now();

        cpuResults = runCPU(data, N);

        auto cpuEnd = std::chrono::steady_clock::now();

        double elapsed =
            std::chrono::duration<double, std::milli>(
                cpuEnd - cpuStart).count();

        cpuTimes.push_back(elapsed);
    }

    // -------------------------------
    // Measured GPU iterations
    // -------------------------------

    std::vector<double> h2dTimes;
    std::vector<double> kernelTimes;
    std::vector<double> d2hTimes;
    std::vector<double> e2eTimes;

    std::cout << "Running " << BENCHMARK_ITERATIONS
              << " GPU iterations...\n";

    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        Timing t = runGPUIteration(
            data, gpuResults, d, N, blocks, start, stop
        );

        h2dTimes.push_back(t.h2d);
        kernelTimes.push_back(t.kernel);
        d2hTimes.push_back(t.d2h);
        e2eTimes.push_back(t.e2e);
    }

    // -------------------------------
    // Statistics
    // -------------------------------

    Statistics cpu = calculateStatistics(cpuTimes);
    Statistics h2d = calculateStatistics(h2dTimes);
    Statistics kernel = calculateStatistics(kernelTimes);
    Statistics d2h = calculateStatistics(d2hTimes);
    Statistics e2e = calculateStatistics(e2eTimes);

    // -------------------------------
    // Correctness verification
    // -------------------------------

    bool correct = verifyResults(cpuResults, gpuResults, N);

    // -------------------------------
    // Throughput and speedup
    // -------------------------------

    double cpuThroughput = N / (cpu.mean * 1000.0);
    double kernelThroughput = N / (kernel.mean * 1000.0);
    double e2eThroughput = N / (e2e.mean * 1000.0);

    double kernelSpeedup = cpu.mean / kernel.mean;
    double e2eSpeedup = cpu.mean / e2e.mean;

    // -------------------------------
    // Market analytics summary
    // -------------------------------

    double avgSpread = 0.0;
    double avgOBI = 0.0;

    int buy = 0;
    int sell = 0;
    int neutral = 0;

    for (int i = 0; i < N; i++)
    {
        avgSpread += gpuResults.spread[i];
        avgOBI += gpuResults.obi[i];

        if (gpuResults.signal[i] == BUY_PRESSURE)
            buy++;
        else if (gpuResults.signal[i] == SELL_PRESSURE)
            sell++;
        else
            neutral++;
    }

    avgSpread /= N;
    avgOBI /= N;

    // -------------------------------
    // Print results
    // -------------------------------

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "\n--- Timing statistics (ms) ---\n";

    std::cout << "CPU mean:       " << cpu.mean << "\n";
    std::cout << "CPU median:     " << cpu.median << "\n";
    std::cout << "CPU min/max:    " << cpu.minimum
              << " / " << cpu.maximum << "\n";

    std::cout << "\nH2D mean:       " << h2d.mean << "\n";
    std::cout << "Kernel mean:    " << kernel.mean << "\n";
    std::cout << "Kernel median:  " << kernel.median << "\n";
    std::cout << "Kernel min/max: " << kernel.minimum
              << " / " << kernel.maximum << "\n";
    std::cout << "D2H mean:       " << d2h.mean << "\n";

    std::cout << "\nGPU E2E mean:   " << e2e.mean << "\n";
    std::cout << "GPU E2E median: " << e2e.median << "\n";
    std::cout << "GPU E2E min/max:" << e2e.minimum
              << " / " << e2e.maximum << "\n";

    std::cout << "GPU setup:      " << setupMs << " ms\n";

    std::cout << "\n--- Performance ---\n";
    std::cout << "CPU throughput:          "
              << cpuThroughput << " M events/s\n";
    std::cout << "GPU kernel throughput:   "
              << kernelThroughput << " M events/s\n";
    std::cout << "GPU E2E throughput:      "
              << e2eThroughput << " M events/s\n";

    std::cout << "Kernel-only speedup:     "
              << kernelSpeedup << "x\n";
    std::cout << "End-to-end speedup:      "
              << e2eSpeedup << "x\n";

    std::cout << "\n--- Market analytics ---\n";
    std::cout << "Average spread: " << avgSpread << "\n";
    std::cout << "Average OBI:    " << avgOBI << "\n";
    std::cout << "VWAP CPU:       " << cpuResults.vwap << "\n";
    std::cout << "VWAP GPU:       " << gpuResults.vwap << "\n";

    std::cout << "Buy pressure:   " << buy << "\n";
    std::cout << "Sell pressure:  " << sell << "\n";
    std::cout << "Neutral:        " << neutral << "\n";

    std::cout << "\nVerification: "
              << (correct ? "PASSED" : "FAILED") << "\n";

    // -------------------------------
    // CSV export
    // -------------------------------

    csv << N << ","
        << cpu.mean << "," << cpu.median << ","
        << cpu.minimum << "," << cpu.maximum << ","
        << h2d.mean << "," << h2d.median << ","
        << kernel.mean << "," << kernel.median << ","
        << kernel.minimum << "," << kernel.maximum << ","
        << d2h.mean << "," << d2h.median << ","
        << e2e.mean << "," << e2e.median << ","
        << e2e.minimum << "," << e2e.maximum << ","
        << setupMs << ","
        << cpuThroughput << ","
        << kernelThroughput << ","
        << e2eThroughput << ","
        << kernelSpeedup << ","
        << e2eSpeedup << ","
        << (correct ? "PASSED" : "FAILED")
        << "\n";

    csv.flush();

    // -------------------------------
    // Cleanup
    // -------------------------------

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    freeDeviceMemory(d);

    return correct;
}

// --------------------------------------------------
// 7. Main
// --------------------------------------------------

int main(int argc, char* argv[])
{
    std::vector<int> workloads = {
        100000,
        500000,
        1000000,
        2000000,
        5000000
    };

    // Optional: run only one workload.
    if (argc == 2)
    {
        int N = std::atoi(argv[1]);

        if (N <= 0 || N > 10000000)
        {
            std::cerr << "Event count must be between 1 and "
                      << "10,000,000.\n";
            return 1;
        }

        workloads = {N};
    }
    else if (argc > 2)
    {
        std::cerr << "Usage: ./hft_analytics [event_count]\n";
        return 1;
    }

    std::ofstream csv(
        "benchmarks/automated_results.csv"
    );

    if (!csv.is_open())
    {
        std::cerr << "Could not open benchmarks/automated_results.csv\n";
        return 1;
    }

    csv << std::fixed << std::setprecision(6);

    csv << "events,"
        << "cpu_mean_ms,cpu_median_ms,cpu_min_ms,cpu_max_ms,"
        << "h2d_mean_ms,h2d_median_ms,"
        << "kernel_mean_ms,kernel_median_ms,kernel_min_ms,kernel_max_ms,"
        << "d2h_mean_ms,d2h_median_ms,"
        << "gpu_e2e_mean_ms,gpu_e2e_median_ms,"
        << "gpu_e2e_min_ms,gpu_e2e_max_ms,"
        << "setup_ms,"
        << "cpu_throughput_mops,"
        << "gpu_kernel_throughput_mops,"
        << "gpu_e2e_throughput_mops,"
        << "kernel_speedup,e2e_speedup,verification\n";

    bool allPassed = true;

    std::cout << "\nCUDA HFT Automated Benchmark\n";
    std::cout << "Warm-up iterations: " << WARMUP_ITERATIONS << "\n";
    std::cout << "Measured iterations: " << BENCHMARK_ITERATIONS << "\n";

    for (int N : workloads)
    {
        bool passed = benchmarkWorkload(N, csv);

        if (!passed)
            allPassed = false;
    }

    csv.close();

    std::cout << "\n========================================\n";
    std::cout << "Benchmark suite completed.\n";
    std::cout << "CSV: benchmarks/automated_results.csv\n";
    std::cout << "Overall verification: "
              << (allPassed ? "PASSED" : "FAILED") << "\n";
    std::cout << "========================================\n";

    return allPassed ? 0 : 1;
}
    
   
