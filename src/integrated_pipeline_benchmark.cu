#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::cerr << "CUDA error at line " << __LINE__ << ": " \
                  << cudaGetErrorString(e) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

constexpr int N = 5000000;
constexpr int THREADS = 256;
constexpr int WARMUPS = 3;
constexpr int ITERATIONS = 20;
constexpr int NUM_STREAMS = 2;
const int CHUNK_SIZES[] = {50000, 100000, 250000, 500000, 1000000, 2500000};
constexpr int NUM_CHUNKS = sizeof(CHUNK_SIZES) / sizeof(CHUNK_SIZES[0]);
using Clock = std::chrono::steady_clock;

struct Config {
    const char* name;
    int streams;
    bool async;
    bool pinned;
};

const Config CONFIGS[] = {
    {"pageable_sync_1_stream", 1, false, false},
    {"pinned_sync_1_stream", 1, false, true},
    {"pinned_async_1_stream", 1, true, true},
    {"pinned_async_2_streams", 2, true, true}
};
constexpr int NUM_CONFIGS = sizeof(CONFIGS) / sizeof(CONFIGS[0]);

struct HostData {
    std::vector<float> bid, ask, spread, midpoint;
};

struct DeviceBuffer {
    float *bid = nullptr, *ask = nullptr, *spread = nullptr, *midpoint = nullptr;
    cudaStream_t stream = nullptr;
};

double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double average(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

double stddev(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    const double m = average(v);
    double sum = 0.0;
    for (double x : v) sum += (x - m) * (x - m);
    return std::sqrt(sum / (v.size() - 1));
}

double ci95(const std::vector<double>& v) {
    // Student-t critical value, two-sided 95%, df=19.
    return v.size() == 20 ? 2.093 * stddev(v) / std::sqrt(20.0) : 0.0;
}

__global__ void marketKernel(
    const float* bid, const float* ask,
    float* spread, float* midpoint, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        spread[i] = ask[i] - bid[i];
        midpoint[i] = (ask[i] + bid[i]) * 0.5f;
    }
}

void initialise(HostData& h) {
    h.bid.resize(N); h.ask.resize(N);
    h.spread.resize(N); h.midpoint.resize(N);
    for (int i = 0; i < N; ++i) {
        h.bid[i] = 100.0f + (i % 1000) * 0.001f;
        h.ask[i] = h.bid[i] + 0.02f + (i % 10) * 0.001f;
    }
}

void allocateBuffers(DeviceBuffer b[NUM_STREAMS], int maxChunk) {
    size_t bytes = static_cast<size_t>(maxChunk) * sizeof(float);
    for (int s = 0; s < NUM_STREAMS; ++s) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&b[s].bid), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&b[s].ask), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&b[s].spread), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&b[s].midpoint), bytes));
        CUDA_CHECK(cudaStreamCreateWithFlags(&b[s].stream, cudaStreamNonBlocking));
    }
}

void freeBuffers(DeviceBuffer b[NUM_STREAMS]) {
    for (int s = 0; s < NUM_STREAMS; ++s) {
        if (b[s].stream) CUDA_CHECK(cudaStreamDestroy(b[s].stream));
        if (b[s].bid) CUDA_CHECK(cudaFree(b[s].bid));
        if (b[s].ask) CUDA_CHECK(cudaFree(b[s].ask));
        if (b[s].spread) CUDA_CHECK(cudaFree(b[s].spread));
        if (b[s].midpoint) CUDA_CHECK(cudaFree(b[s].midpoint));
    }
}

void pinHost(HostData& h) {
    CUDA_CHECK(cudaHostRegister(h.bid.data(), h.bid.size() * sizeof(float), cudaHostRegisterDefault));
    CUDA_CHECK(cudaHostRegister(h.ask.data(), h.ask.size() * sizeof(float), cudaHostRegisterDefault));
    CUDA_CHECK(cudaHostRegister(h.spread.data(), h.spread.size() * sizeof(float), cudaHostRegisterDefault));
    CUDA_CHECK(cudaHostRegister(h.midpoint.data(), h.midpoint.size() * sizeof(float), cudaHostRegisterDefault));
}

void unpinHost(HostData& h) {
    CUDA_CHECK(cudaHostUnregister(h.bid.data()));
    CUDA_CHECK(cudaHostUnregister(h.ask.data()));
    CUDA_CHECK(cudaHostUnregister(h.spread.data()));
    CUDA_CHECK(cudaHostUnregister(h.midpoint.data()));
}

void processSync(HostData& h, DeviceBuffer& b, int offset, int count) {
    size_t bytes = static_cast<size_t>(count) * sizeof(float);
    CUDA_CHECK(cudaMemcpy(b.bid, h.bid.data() + offset, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.ask, h.ask.data() + offset, bytes, cudaMemcpyHostToDevice));
    int blocks = (count + THREADS - 1) / THREADS;
    marketKernel<<<blocks, THREADS>>>(b.bid, b.ask, b.spread, b.midpoint, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h.spread.data() + offset, b.spread, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.midpoint.data() + offset, b.midpoint, bytes, cudaMemcpyDeviceToHost));
}

void processAsync(HostData& h, DeviceBuffer& b, int offset, int count) {
    size_t bytes = static_cast<size_t>(count) * sizeof(float);
    cudaStream_t s = b.stream;
    CUDA_CHECK(cudaMemcpyAsync(b.bid, h.bid.data() + offset, bytes, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(b.ask, h.ask.data() + offset, bytes, cudaMemcpyHostToDevice, s));
    int blocks = (count + THREADS - 1) / THREADS;
    marketKernel<<<blocks, THREADS, 0, s>>>(b.bid, b.ask, b.spread, b.midpoint, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(h.spread.data() + offset, b.spread, bytes, cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(h.midpoint.data() + offset, b.midpoint, bytes, cudaMemcpyDeviceToHost, s));
}

double run(HostData& h, DeviceBuffer b[NUM_STREAMS], const Config& cfg, int chunkSize) {
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = Clock::now();
    int chunk = 0;
    for (int offset = 0; offset < N; offset += chunkSize, ++chunk) {
        int count = std::min(chunkSize, N - offset);
        DeviceBuffer& current = b[chunk % cfg.streams];
        if (cfg.async) processAsync(h, current, offset, count);
        else processSync(h, current, offset, count);
    }
    for (int s = 0; s < cfg.streams; ++s)
        CUDA_CHECK(cudaStreamSynchronize(b[s].stream));
    CUDA_CHECK(cudaDeviceSynchronize());
    return ms(start, Clock::now());
}

bool verify(const HostData& h) {
    for (int i = 0; i < N; ++i) {
        float expectedSpread = h.ask[i] - h.bid[i];
        float expectedMid = (h.ask[i] + h.bid[i]) * 0.5f;
        if (std::fabs(h.spread[i] - expectedSpread) > 1e-5f ||
            std::fabs(h.midpoint[i] - expectedMid) > 1e-5f) {
            std::cerr << "Verification failed at index " << i << '\n';
            return false;
        }
    }
    return true;
}

int main() {
    std::cout << std::fixed << std::setprecision(5);
    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "Integrated pipeline benchmark\nGPU: " << prop.name
              << "\nCompute capability: " << prop.major << "." << prop.minor
              << "\nEvents: " << N << "\nWarmups: " << WARMUPS
              << "\nMeasured iterations: " << ITERATIONS << '\n';

    // Keep separate host allocations: configuration A must remain pageable,
    // while configurations B-D use the separately registered pinned buffers.
    HostData pageable;
    HostData pinned;
    initialise(pageable);
    initialise(pinned);
    DeviceBuffer buffers[NUM_STREAMS]{};

    auto allocStart = Clock::now();
    allocateBuffers(buffers, CHUNK_SIZES[NUM_CHUNKS - 1]);
    double allocationMs = ms(allocStart, Clock::now());

    auto pinStart = Clock::now();
    pinHost(pinned);
    double registrationMs = ms(pinStart, Clock::now());

    std::ofstream raw("benchmarks/integrated_pipeline_raw.csv");
    std::ofstream summary("benchmarks/integrated_pipeline_summary.csv");
    std::ofstream stats("benchmarks/integrated_pipeline_statistics.csv");
    std::ofstream setup("benchmarks/integrated_pipeline_setup.csv");
    if (!raw || !summary || !stats || !setup) {
        std::cerr << "Cannot open benchmark output CSV files.\n";
        unpinHost(pinned); freeBuffers(buffers);
        return 1;
    }

    raw << "configuration,chunk_size,iteration,execution_order,elapsed_ms\n";
    summary << "configuration,chunk_size,iterations,mean_ms,stddev_ms,ci95_margin_ms,events_per_second\n";
    stats << "configuration,chunk_size,mean_ms,stddev_ms,ci95_margin_ms\n";
    setup << "host_registration_ms,device_allocation_ms\n"
          << registrationMs << "," << allocationMs << "\n";

    bool allCorrect = true;
    for (int ci = 0; ci < NUM_CHUNKS && allCorrect; ++ci) {
        int chunkSize = CHUNK_SIZES[ci];
        std::cout << "\nChunk size: " << chunkSize << '\n';

        for (int c = 0; c < NUM_CONFIGS; ++c) {
            HostData& selectedHost = CONFIGS[c].pinned ? pinned : pageable;
            for (int w = 0; w < WARMUPS; ++w)
                run(selectedHost, buffers, CONFIGS[c], chunkSize);
        }

        std::vector<double> timings[NUM_CONFIGS];
        for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
            // Rotate order across iterations to reduce systematic order bias.
            for (int position = 0; position < NUM_CONFIGS; ++position) {
                int c = (position + iteration) % NUM_CONFIGS;
                HostData& selectedHost = CONFIGS[c].pinned ? pinned : pageable;
                double elapsed = run(selectedHost, buffers, CONFIGS[c], chunkSize);
                timings[c].push_back(elapsed);
                raw << CONFIGS[c].name << "," << chunkSize << ","
                    << iteration + 1 << "," << position + 1 << ","
                    << elapsed << "\n";
            }
            raw.flush();
        }

        // Correctness checks are outside measured intervals.
        for (int c = 0; c < NUM_CONFIGS; ++c) {
            HostData& selectedHost = CONFIGS[c].pinned ? pinned : pageable;
            run(selectedHost, buffers, CONFIGS[c], chunkSize);
            if (!verify(selectedHost)) {
                std::cerr << "Failed: " << CONFIGS[c].name
                          << ", chunk " << chunkSize << '\n';
                allCorrect = false;
                break;
            }
        }

        for (int c = 0; c < NUM_CONFIGS; ++c) {
            double m = average(timings[c]);
            double sd = stddev(timings[c]);
            double ci = ci95(timings[c]);
            double eps = static_cast<double>(N) / (m / 1000.0);
            summary << CONFIGS[c].name << "," << chunkSize << ","
                    << timings[c].size() << "," << m << "," << sd << ","
                    << ci << "," << eps << "\n";
            stats << CONFIGS[c].name << "," << chunkSize << ","
                  << m << "," << sd << "," << ci << "\n";
            std::cout << CONFIGS[c].name << ": " << m << " +/- "
                      << ci << " ms; " << eps << " events/s\n";
        }
    }

    raw.close(); summary.close(); stats.close(); setup.close();
    unpinHost(pinned);
    freeBuffers(buffers);

    std::cout << "\nCorrectness verification: "
              << (allCorrect ? "PASSED" : "FAILED") << '\n'
              << "Results written to benchmarks/integrated_pipeline_*.csv\n";
    return allCorrect ? 0 : 1;
}
