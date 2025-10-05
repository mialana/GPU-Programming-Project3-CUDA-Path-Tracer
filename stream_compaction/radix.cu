#include "radix.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "efficient.h"
#include "shared.h"

using StreamCompaction::Common::PerformanceTimer;

PerformanceTimer& StreamCompaction::Radix::timer()
{
    static PerformanceTimer timer;
    return timer;
}

__device__ __host__ int StreamCompaction::Radix::_isolateBit(const int num, const int tgtBit)
{
    return (num >> tgtBit) & 1;
}

__global__ void StreamCompaction::Radix::_split(int n, int* data, int* notBit, const int tgtBit)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index >= n)
    {
        return;
    }

    notBit[index] = _isolateBit(data[index], tgtBit) ^ 1;  // not(target bit)
}

__global__ void StreamCompaction::Radix::_computeScatterIndices(
    int n, int* indices, const int* idata, const int* scan, const int tgtBit)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index >= n)
    {
        return;
    }

    __shared__ int totalFalses;
    if (threadIdx.x == 0)
    {
        totalFalses = (_isolateBit(idata[n - 1], tgtBit) ^ 1) + scan[n - 1];
    }

    __syncthreads();  // wait for totalFalses

    // if value is 1, we shift right by total falses minus falses before current index
    // if value is 0, we set to position based on how many other falses / 0s come before it
    indices[index] = _isolateBit(idata[index], tgtBit) ? index + (totalFalses - scan[index])
                                                       : scan[index];
}

__global__ void StreamCompaction::Radix::_scatter(int n,
                                                  int* odata,
                                                  const int* idata,
                                                  const int* indices)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index >= n)
    {
        return;
    }

    int address = indices[index];
    odata[address] = idata[index];  // Scatter the value to its new position
}

void StreamCompaction::Radix::sort(int n,
                                   int* dev_dataA,
                                   int* dev_dataB,
                                   int* dev_blockSums,
                                   int* dev_indices,
                                   const int maxBitLength,
                                   const int blockSize)
{
    for (int tgtBit = 0; tgtBit < maxBitLength; tgtBit++)
    {
        unsigned blocks = divup(n, blockSize);

        // Split data into 0s and 1s based on the target bit
        _split<<<blocks, blockSize>>>(n, dev_dataA, dev_dataB, tgtBit);

        // Perform scan on the split results. use same array as we need
        Shared::scan(n, dev_dataB, dev_dataB, dev_blockSums, blockSize);

        // Scatter data based on the split results
        _computeScatterIndices<<<blocks, blockSize>>>(n, dev_indices, dev_dataA, dev_dataB, tgtBit);

        _scatter<<<blocks, blockSize>>>(n, dev_dataB, dev_dataA, dev_indices);

        // Swap buffers (ping-pong)
        int* temp = dev_dataA;
        dev_dataA = dev_dataB;
        dev_dataB = temp;
    }
}

void StreamCompaction::Radix::sortWrapper(
    int n, int* odata, const int* idata, const int maxBitLength, const int blockSize)
{
    const unsigned numLayers = ilog2ceil(n);
    const unsigned long long paddedN = 1 << ilog2ceil(n);
    const unsigned blockSums = divup(paddedN, 2 * blockSize);

    // Allocate device memory for input/output data and scan
    int* dev_dataA;
    int* dev_dataB;
    int* dev_blockSums;
    int* dev_indices;

    cudaMalloc((void**)&dev_dataA, sizeof(int) * paddedN);
    checkCUDAError("CUDA malloc for device idata array failed.");

    cudaMalloc((void**)&dev_dataB, sizeof(int) * paddedN);
    checkCUDAError("CUDA malloc for device odata array failed.");

    cudaMalloc((void**)&dev_blockSums, sizeof(int) * blockSums);
    checkCUDAError("CUDA malloc for device block sums array failed.");

    cudaMalloc((void**)&dev_indices, sizeof(int) * n);
    checkCUDAError("CUDA malloc for device indices array failed.");

    // Copy input data to device
    cudaMemcpy(dev_dataA, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
    checkCUDAError("Memory copy from host data to device array failed.");

    bool usingTimer = false;
    if (!timer().gpu_timer_started)
    {
        timer().startGpuTimer();
        usingTimer = true;
    }

    StreamCompaction::Radix::sort(n,
                                  dev_dataA,
                                  dev_dataB,
                                  dev_blockSums,
                                  dev_indices,
                                  maxBitLength,
                                  blockSize);

    if (usingTimer)
    {
        timer().endGpuTimer();
    }

    // Copy sorted data back to host
    cudaMemcpy(odata, dev_dataA, sizeof(int) * n, cudaMemcpyDeviceToHost);
    checkCUDAError("Memory copy from device array to host data failed.");

    // Free device memory
    cudaFree(dev_dataA);
    cudaFree(dev_dataB);
    cudaFree(dev_blockSums);
    cudaFree(dev_indices);
}

void StreamCompaction::Radix::sortByKey(int n,
                                        int* dev_keysA,
                                        int* dev_keysB,
                                        int* dev_valuesA,
                                        int* dev_valuesB,
                                        int* dev_blockSums,
                                        int* dev_indices,
                                        const int maxBitLength,
                                        const int blockSize)
{
    for (int tgtBit = 0; tgtBit < maxBitLength; tgtBit++)
    {
        unsigned blocks = divup(n, blockSize);

        // Split data into 0s and 1s based on the target bit
        _split<<<blocks, blockSize>>>(n, dev_keysA, dev_keysB, tgtBit);

        // Perform scan on the split results
        Shared::scan(n, dev_keysB, dev_keysB, dev_blockSums, blockSize);

        // Scatter data based on the split results
        _computeScatterIndices<<<blocks, blockSize>>>(n, dev_indices, dev_keysA, dev_keysB, tgtBit);

        _scatter<<<blocks, blockSize>>>(n, dev_keysB, dev_keysA, dev_indices);
        _scatter<<<blocks, blockSize>>>(n, dev_valuesB, dev_valuesA, dev_indices);

        // Swap buffers (ping-pong)
        int* temp = dev_keysA;
        dev_keysA = dev_keysB;
        dev_keysB = temp;

        temp = dev_valuesA;
        dev_valuesA = dev_valuesB;
        dev_valuesB = temp;
    }
}

void StreamCompaction::Radix::sortByKeyWrapper(int n,
                                               int* okeys,
                                               const int* ikeys,
                                               int* ovalues,
                                               const int* ivalues,
                                               const int maxBitLength,
                                               const int blockSize)
{
    const unsigned numLayers = ilog2ceil(n);
    const unsigned long long paddedN = 1 << ilog2ceil(n);
    const unsigned blockSums = divup(paddedN, 2 * blockSize);

    // Allocate device memory for input/output data and scan
    int* dev_keysA;
    int* dev_keysB;
    int* dev_valuesA;
    int* dev_valuesB;
    int* dev_blockSums;
    int* dev_indices;

    cudaMalloc((void**)&dev_keysA, sizeof(int) * paddedN);
    checkCUDAError("CUDA malloc for device ikeys array failed.");

    cudaMalloc((void**)&dev_keysB, sizeof(int) * paddedN);
    checkCUDAError("CUDA malloc for device okeys array failed.");

    cudaMalloc((void**)&dev_valuesA, sizeof(int) * paddedN);
    checkCUDAError("CUDA malloc for device ivalues array failed.");

    cudaMalloc((void**)&dev_valuesB, sizeof(int) * paddedN);
    checkCUDAError("CUDA malloc for device ovalues array failed.");

    cudaMalloc((void**)&dev_blockSums, sizeof(int) * blockSums);
    checkCUDAError("CUDA malloc for device block sums array failed.");

    cudaMalloc((void**)&dev_indices, sizeof(int) * n);
    checkCUDAError("CUDA malloc for device indices array failed.");

    // Copy input data to device
    cudaMemcpy(dev_keysA, ikeys, sizeof(int) * n, cudaMemcpyHostToDevice);
    checkCUDAError("Memory copy from host ikeys to device array failed.");

    cudaMemcpy(dev_valuesA, ivalues, sizeof(int) * n, cudaMemcpyHostToDevice);
    checkCUDAError("Memory copy from host values to device array failed.");

    bool usingTimer = false;
    if (!timer().gpu_timer_started)
    {
        timer().startGpuTimer();
        usingTimer = true;
    }

    StreamCompaction::Radix::sortByKey(n,
                                       dev_keysA,
                                       dev_keysB,
                                       dev_valuesA,
                                       dev_valuesB,
                                       dev_blockSums,
                                       dev_indices,
                                       maxBitLength,
                                       blockSize);

    if (usingTimer)
    {
        timer().endGpuTimer();
    }

    // Copy sorted data back to host
    cudaMemcpy(okeys, dev_keysA, sizeof(int) * n, cudaMemcpyDeviceToHost);
    checkCUDAError("Memory copy from device array to host data failed.");

    cudaMemcpy(ovalues, dev_valuesA, sizeof(int) * n, cudaMemcpyDeviceToHost);
    checkCUDAError("Memory copy from device array to host data failed.");

    // Free device memory
    cudaFree(dev_keysA);
    cudaFree(dev_keysB);
    cudaFree(dev_valuesA);
    cudaFree(dev_valuesB);
    cudaFree(dev_blockSums);
    cudaFree(dev_indices);
}
