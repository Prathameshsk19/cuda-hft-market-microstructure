#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
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

const int WARMUP = 3;
const int ITERATIONS = 20;

struct TransferResult {
    double h2d_ms;
    double d2h_ms;
    double total_ms;
};

double elapsedMs(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

TransferResult benchmarkSync(
    const std::vector<float>& input,
    std::vector<float>& output,
    float* d_input,
    float* d_output,
    cudaMemcpyKind direction)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double total = 0.0;

    for (int i = 0; i < WARMUP + ITERATIONS; i++) {
        CUDA_CHECK(cudaEventRecord(start));

        CUDA_CHECK(cudaMemcpy(
            direction == cudaMemcpyHostToDevice
                ? static_cast<void*>(d_input)
                : static_cast<void*>(output.data()),

            direction == cudaMemcpyHostToDevice
                ? static_cast<const void*>(input.data())
                : static_cast<const void*>(d_output),

            input.size() * sizeof(float),
            direction
        ));

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        if (i >= WARMUP)
            total += elapsedMs(start, stop);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    double mean = total / ITERATIONS;

    return {
        direction == cudaMemcpyHostToDevice ? mean : 0.0,
        direction == cudaMemcpyDeviceToHost ? mean : 0.0,
        mean
    };
}

double benchmarkAsync(
    const std::vector<float>& input,
    std::vector<float>& output,
    float* d_input,
    float* d_output,
    cudaMemcpyKind direction,
    cudaStream_t stream)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double total = 0.0;

    for (int i = 0; i < WARMUP + ITERATIONS; i++) {

        CUDA_CHECK(cudaEventRecord(start, stream));

        CUDA_CHECK(cudaMemcpyAsync(
            direction == cudaMemcpyHostToDevice
                ? static_cast<void*>(d_input)
                : static_cast<void*>(output.data()),

            direction == cudaMemcpyHostToDevice
                ? static_cast<const void*>(input.data())
                : static_cast<const void*>(d_output),

            input.size() * sizeof(float),
            direction,
            stream
        ));

        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        if (i >= WARMUP)
            total += elapsedMs(start, stop);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total / ITERATIONS;
}

int main()
{
    std::vector<int> sizes = {
        100000, 500000, 1000000, 2000000, 5000000
    };

    std::ofstream csv("benchmarks/async_transfer_results.csv");

    csv << "events,"
        << "pageable_h2d_ms,pageable_d2h_ms,"
        << "pinned_sync_h2d_ms,pinned_sync_d2h_ms,"
        << "pinned_async_h2d_ms,pinned_async_d2h_ms\n";

    std::cout << std::fixed << std::setprecision(4);

    for (int N : sizes) {

        std::cout << "\n========================================\n";
        std::cout << "Events: " << N << "\n";
        std::cout << "========================================\n";

        std::vector<float> input(N, 1.25f);
        std::vector<float> output(N, 0.0f);

        float *d_input, *d_output;

        CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));

        // Initialize device output for D2H tests.
        CUDA_CHECK(cudaMemset(d_output, 1, N * sizeof(float)));

        // 1. Pageable synchronous baseline.
        auto pageable_h2d = benchmarkSync(
            input, output, d_input, d_output,
            cudaMemcpyHostToDevice
        );

        auto pageable_d2h = benchmarkSync(
            input, output, d_input, d_output,
            cudaMemcpyDeviceToHost
        );

        // Register host buffers as pinned memory.
        CUDA_CHECK(cudaHostRegister(
            input.data(), N * sizeof(float),
            cudaHostRegisterDefault
        ));

        CUDA_CHECK(cudaHostRegister(
            output.data(), N * sizeof(float),
            cudaHostRegisterDefault
        ));

        // 2. Pinned synchronous transfers.
        auto pinned_h2d = benchmarkSync(
            input, output, d_input, d_output,
            cudaMemcpyHostToDevice
        );

        auto pinned_d2h = benchmarkSync(
            input, output, d_input, d_output,
            cudaMemcpyDeviceToHost
        );

        // Create explicit CUDA stream.
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // 3. Pinned asynchronous transfers.
        double async_h2d = benchmarkAsync(
            input, output, d_input, d_output,
            cudaMemcpyHostToDevice, stream
        );

        double async_d2h = benchmarkAsync(
            input, output, d_input, d_output,
            cudaMemcpyDeviceToHost, stream
        );

        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::cout << "Pageable H2D:    " << pageable_h2d.h2d_ms << " ms\n";
        std::cout << "Pageable D2H:    " << pageable_d2h.d2h_ms << " ms\n";

        std::cout << "Pinned Sync H2D: " << pinned_h2d.h2d_ms << " ms\n";
        std::cout << "Pinned Sync D2H: " << pinned_d2h.d2h_ms << " ms\n";

        std::cout << "Pinned Async H2D:" << async_h2d << " ms\n";
        std::cout << "Pinned Async D2H:" << async_d2h << " ms\n";

        csv << N << ","
            << pageable_h2d.h2d_ms << ","
            << pageable_d2h.d2h_ms << ","
            << pinned_h2d.h2d_ms << ","
            << pinned_d2h.d2h_ms << ","
            << async_h2d << ","
            << async_d2h << "\n";

        CUDA_CHECK(cudaStreamDestroy(stream));

        CUDA_CHECK(cudaHostUnregister(input.data()));
        CUDA_CHECK(cudaHostUnregister(output.data()));

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
    }

    csv.close();

    std::cout << "\nResults saved to "
              << "benchmarks/async_transfer_results.csv\n";

    return 0;
}
