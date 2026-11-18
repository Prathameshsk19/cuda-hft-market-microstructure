#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA Error: "                            \
                  << cudaGetErrorString(err)                  \
                  << " at line " << __LINE__ << std::endl;    \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)

const int N = 5000000;
const int CHUNK_SIZE = 500000;
const int NUM_CHUNKS = N / CHUNK_SIZE;
const int THREADS = 256;
const int WARMUP = 3;
const int ITERATIONS = 10;

__global__ void marketKernel(
    const float* bid,
    const float* ask,
    float* spread,
    float* midpoint,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        spread[i] = ask[i] - bid[i];
        midpoint[i] = (ask[i] + bid[i]) * 0.5f;
    }
}

using Clock = std::chrono::steady_clock;

struct StreamBuffer {
    float *d_bid, *d_ask;
    float *d_spread, *d_midpoint;
    cudaStream_t stream;
};

double runPipeline(
    const std::vector<float>& bid,
    const std::vector<float>& ask,
    std::vector<float>& spread,
    std::vector<float>& midpoint,
    StreamBuffer* buffers,
    int numStreams)
{
    auto start = Clock::now();

    for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {

        int streamIndex = chunk % numStreams;
        int offset = chunk * CHUNK_SIZE;

        StreamBuffer& b = buffers[streamIndex];

        CUDA_CHECK(cudaMemcpyAsync(
            b.d_bid,
            bid.data() + offset,
            CHUNK_SIZE * sizeof(float),
            cudaMemcpyHostToDevice,
            b.stream
        ));

        CUDA_CHECK(cudaMemcpyAsync(
            b.d_ask,
            ask.data() + offset,
            CHUNK_SIZE * sizeof(float),
            cudaMemcpyHostToDevice,
            b.stream
        ));

        int blocks = (CHUNK_SIZE + THREADS - 1) / THREADS;

        marketKernel<<<blocks, THREADS, 0, b.stream>>>(
            b.d_bid,
            b.d_ask,
            b.d_spread,
            b.d_midpoint,
            CHUNK_SIZE
        );

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(
            spread.data() + offset,
            b.d_spread,
            CHUNK_SIZE * sizeof(float),
            cudaMemcpyDeviceToHost,
            b.stream
        ));

        CUDA_CHECK(cudaMemcpyAsync(
            midpoint.data() + offset,
            b.d_midpoint,
            CHUNK_SIZE * sizeof(float),
            cudaMemcpyDeviceToHost,
            b.stream
        ));
    }

    // Wait until all streams have completed.
    for (int i = 0; i < numStreams; i++) {
        CUDA_CHECK(cudaStreamSynchronize(buffers[i].stream));
    }

    auto stop = Clock::now();

    return std::chrono::duration<double, std::milli>(
        stop - start
    ).count();
}

bool verifyResults(
    const std::vector<float>& bid,
    const std::vector<float>& ask,
    const std::vector<float>& spread,
    const std::vector<float>& midpoint)
{
    for (int i = 0; i < N; i++) {

        float expectedSpread = ask[i] - bid[i];
        float expectedMid = (ask[i] + bid[i]) * 0.5f;

        if (std::fabs(spread[i] - expectedSpread) > 1e-5f ||
            std::fabs(midpoint[i] - expectedMid) > 1e-5f) {
            std::cerr << "Verification failed at index "
                      << i << "\n";
            return false;
        }
    }

    return true;
}

int main()
{
    std::cout << std::fixed << std::setprecision(4);

    // Initialize CUDA context before benchmark timing.
    CUDA_CHECK(cudaFree(0));

    std::vector<float> bid(N);
    std::vector<float> ask(N);

    std::vector<float> spread(N);
    std::vector<float> midpoint(N);

    // Generate deterministic market data.
    for (int i = 0; i < N; i++) {
        bid[i] = 100.0f + (i % 1000) * 0.001f;
        ask[i] = bid[i] + 0.02f + (i % 10) * 0.001f;
    }

    // Allocate pinned host memory.
    CUDA_CHECK(cudaHostRegister(
        bid.data(), N * sizeof(float), cudaHostRegisterDefault
    ));

    CUDA_CHECK(cudaHostRegister(
        ask.data(), N * sizeof(float), cudaHostRegisterDefault
    ));

    CUDA_CHECK(cudaHostRegister(
        spread.data(), N * sizeof(float), cudaHostRegisterDefault
    ));

    CUDA_CHECK(cudaHostRegister(
        midpoint.data(), N * sizeof(float), cudaHostRegisterDefault
    ));

    // Allocate two independent stream buffers.
    StreamBuffer buffers[2]{};

    for (int i = 0; i < 2; i++) {

        CUDA_CHECK(cudaMalloc(
            &buffers[i].d_bid,
            CHUNK_SIZE * sizeof(float)
        ));

        CUDA_CHECK(cudaMalloc(
            &buffers[i].d_ask,
            CHUNK_SIZE * sizeof(float)
        ));

        CUDA_CHECK(cudaMalloc(
            &buffers[i].d_spread,
            CHUNK_SIZE * sizeof(float)
        ));

        CUDA_CHECK(cudaMalloc(
            &buffers[i].d_midpoint,
            CHUNK_SIZE * sizeof(float)
        ));

        CUDA_CHECK(cudaStreamCreate(&buffers[i].stream));
    }

    std::cout << "\n===== CUDA PIPELINE BENCHMARK =====\n";
    std::cout << "GPU: NVIDIA GeForce 940MX\n";
    std::cout << "Total events: " << N << "\n";
    std::cout << "Chunk size: " << CHUNK_SIZE << "\n";
    std::cout << "Number of chunks: " << NUM_CHUNKS << "\n";

    // Warm up both configurations.
    runPipeline(bid, ask, spread, midpoint, buffers, 1);
    runPipeline(bid, ask, spread, midpoint, buffers, 2);

    double sequentialTotal = 0.0;
    double pipelineTotal = 0.0;

    for (int i = 0; i < ITERATIONS; i++) {

        sequentialTotal += runPipeline(
            bid, ask, spread, midpoint, buffers, 1
        );

        pipelineTotal += runPipeline(
            bid, ask, spread, midpoint, buffers, 2
        );
    }

    double sequentialMean = sequentialTotal / ITERATIONS;
    double pipelineMean = pipelineTotal / ITERATIONS;

    std::cout << "\n----- RESULTS -----\n";

    std::cout << "Single-stream pipeline: "
              << sequentialMean << " ms\n";

    std::cout << "Two-stream pipeline:    "
              << pipelineMean << " ms\n";

    std::cout << "Pipeline speedup:       "
              << sequentialMean / pipelineMean << "x\n";

    std::cout << "-------------------\n";

    if (verifyResults(bid, ask, spread, midpoint)) {
        std::cout << "Verification: PASSED\n";
    } else {
        std::cout << "Verification: FAILED\n";
    }

    std::ofstream csv("benchmarks/pipeline_results.csv");

    csv << "events,chunk_size,sequential_ms,two_stream_ms,speedup\n";
    csv << N << ","
        << CHUNK_SIZE << ","
        << sequentialMean << ","
        << pipelineMean << ","
        << sequentialMean / pipelineMean << "\n";

    csv.close();

    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaStreamDestroy(buffers[i].stream));

        CUDA_CHECK(cudaFree(buffers[i].d_bid));
        CUDA_CHECK(cudaFree(buffers[i].d_ask));
        CUDA_CHECK(cudaFree(buffers[i].d_spread));
        CUDA_CHECK(cudaFree(buffers[i].d_midpoint));
    }

    CUDA_CHECK(cudaHostUnregister(bid.data()));
    CUDA_CHECK(cudaHostUnregister(ask.data()));
    CUDA_CHECK(cudaHostUnregister(spread.data()));
    CUDA_CHECK(cudaHostUnregister(midpoint.data()));

    std::cout << "\nResults saved to benchmarks/pipeline_results.csv\n";

    return 0;
}
