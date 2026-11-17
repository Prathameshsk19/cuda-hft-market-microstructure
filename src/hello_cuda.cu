#include <iostream>
#include <cuda_runtime.h>

__global__ void processMarketEvents(int *data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        data[idx] = data[idx] * 2;
    }
}

int main()
{
    const int N = 1024;
    const int SIZE = N * sizeof(int);

    int *h_data = new int[N];
    int *d_data;

    for (int i = 0; i < N; i++)
        h_data[i] = i;

    cudaMalloc(&d_data, SIZE);

    cudaMemcpy(d_data, h_data, SIZE,
               cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1)
                        / threadsPerBlock;

    processMarketEvents<<<blocksPerGrid,
                          threadsPerBlock>>>(d_data, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, SIZE,
               cudaMemcpyDeviceToHost);

    std::cout << "CUDA Market Event Processing\n";

    for (int i = 0; i < 10; i++)
        std::cout << "Event " << i
                  << " : " << h_data[i] << "\n";

    cudaFree(d_data);
    delete[] h_data;

    return 0;
}
