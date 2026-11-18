#include <iostream>
#include <cuda_runtime.h>

int main()
{
    int device = 0;

    cudaDeviceProp prop{};

    cudaError_t err = cudaGetDeviceProperties(&prop, device);

    if (err != cudaSuccess)
    {
        std::cerr << cudaGetErrorString(err) << "\n";
        return 1;
    }

    std::cout << "\n===== GPU CAPABILITIES =====\n";

    std::cout << "GPU: " << prop.name << "\n";

    std::cout << "Compute capability: "
              << prop.major << "." << prop.minor << "\n";

    std::cout << "Async engine count: "
              << prop.asyncEngineCount << "\n";

    std::cout << "Concurrent kernels: "
              << prop.concurrentKernels << "\n";

    std::cout << "Can map host memory: "
              << prop.canMapHostMemory << "\n";

    std::cout << "Unified addressing: "
              << prop.unifiedAddressing << "\n";

    std::cout << "Global memory: "
              << prop.totalGlobalMem / (1024.0 * 1024.0)
              << " MB\n";

    std::cout << "============================\n";

    return 0;
}
