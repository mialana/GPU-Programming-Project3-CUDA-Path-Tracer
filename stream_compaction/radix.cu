#include "radix.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "efficient.h"

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

template<typename T>
__global__ void StreamCompaction::Radix::_scatter(int n,
                                                  T* odata,
                                                  const T* idata,
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

void StreamCompaction::Radix::sort(int n, int* odata, const int* idata, const int maxBitLength)
{
    const unsigned numLayers = ilog2ceil(n);
    const unsigned paddedN = 1 << ilog2ceil(n);

    // Allocate device memory for input/output data and scan
    int* dev_data[2];
    int* dev_scan;
    int* dev_indices;

    cudaMalloc((void**)&dev_data[0], sizeof(int) * n);
    checkCUDAError("CUDA malloc for device data array failed.");

    cudaMalloc((void**)&dev_data[1], sizeof(int) * n);
    checkCUDAError("CUDA malloc for temporary data array failed.");

    cudaMalloc((void**)&dev_scan, sizeof(int) * paddedN);
    checkCUDAError("CUDA malloc for device scan array failed.");

    cudaMalloc((void**)&dev_indices, sizeof(int) * n);
    checkCUDAError("CUDA malloc for device indices array failed.");

    // Copy input data to device
    cudaMemcpy(dev_data[0], idata, sizeof(int) * n, cudaMemcpyHostToDevice);
    checkCUDAError("Memory copy from host data to device array failed.");

    bool usingTimer = false;
    if (!timer().gpu_timer_started)
    {
        timer().startGpuTimer();
        usingTimer = true;
    }

    int current = 0;  // Index to track the current buffer (ping-pong)

    for (int tgtBit = 0; tgtBit < maxBitLength; tgtBit++)
    {
        unsigned blocks = divup(n, BLOCK_SIZE);

        // Split data into 0s and 1s based on the target bit
        _split<<<blocks, BLOCK_SIZE>>>(n, dev_data[current], dev_scan, tgtBit);

        // Perform scan on the split results
        Efficient::scanHelper(numLayers, paddedN, dev_scan);

        // Scatter data based on the split results
        _computeScatterIndices<<<blocks, BLOCK_SIZE>>>(n,
                                                       dev_indices,
                                                       dev_data[current],
                                                       dev_scan,
                                                       tgtBit);

        _scatter<<<blocks, BLOCK_SIZE>>>(n, dev_data[1 - current], dev_data[current], dev_indices);

        // Swap buffers (ping-pong)
        current = 1 - current;
    }

    if (usingTimer)
    {
        timer().endGpuTimer();
    }

    // Copy sorted data back to host
    cudaMemcpy(odata, dev_data[current], sizeof(int) * n, cudaMemcpyDeviceToHost);
    checkCUDAError("Memory copy from device array to host data failed.");

    // Free device memory
    cudaFree(dev_data[0]);
    cudaFree(dev_data[1]);
    cudaFree(dev_scan);
}

template<typename T>
void StreamCompaction::Radix::sortByKey(
    int n, int* outKeys, T* outValues, const int* keys, const T* values, const int maxBitLength)
{
    const unsigned numLayers = ilog2ceil(n);
    const unsigned paddedN = 1 << ilog2ceil(n);

    // Allocate device memory for keys, objects, and scan
    int* dev_keys[2];
    T* dev_values[2];
    int* dev_scan;
    int* dev_indices;

    cudaMalloc((void**)&dev_keys[0], sizeof(int) * n);
    checkCUDAError("CUDA malloc for device keys array failed.");

    cudaMalloc((void**)&dev_keys[1], sizeof(int) * n);
    checkCUDAError("CUDA malloc for temporary keys array failed.");

    cudaMalloc((void**)&dev_values[0], sizeof(T) * n);
    checkCUDAError("CUDA malloc for device objects array failed.");

    cudaMalloc((void**)&dev_values[1], sizeof(T) * n);
    checkCUDAError("CUDA malloc for temporary objects array failed.");

    cudaMalloc((void**)&dev_scan, sizeof(int) * paddedN);
    checkCUDAError("CUDA malloc for device scan array failed.");

    cudaMalloc((void**)&dev_indices, sizeof(int) * n);
    checkCUDAError("CUDA malloc for device indices array failed.");

    // Copy keys and objects to device
    cudaMemcpy(dev_keys[0], keys, sizeof(int) * n, cudaMemcpyHostToDevice);
    checkCUDAError("Memory copy from host keys to device array failed.");

    cudaMemcpy(dev_values[0], values, sizeof(T) * n, cudaMemcpyHostToDevice);
    checkCUDAError("Memory copy from host values to device array failed.");

    int current = 0;  // Index to track the current buffer (ping-pong)

    for (int tgtBit = 0; tgtBit < maxBitLength; tgtBit++)
    {
        unsigned blocks = divup(n, BLOCK_SIZE);

        // Split keys into 0s and 1s based on the target bit
        _split<<<blocks, BLOCK_SIZE>>>(n, dev_keys[current], dev_scan, tgtBit);

        // Perform scan on the split results
        Efficient::scanHelper(numLayers, paddedN, dev_scan);

        // Scatter keys and rearrange objects based on the sorted keys
        _computeScatterIndices<<<blocks, BLOCK_SIZE>>>(n,
                                                       dev_indices,
                                                       dev_keys[current],
                                                       dev_scan,
                                                       tgtBit);

        // Scatter keys based on the computed indices
        _scatter<<<blocks, BLOCK_SIZE>>>(n, dev_keys[1 - current], dev_keys[current], dev_indices);

        // Scatter values based on the computed indices
        _scatter<<<blocks, BLOCK_SIZE>>>(n, dev_values[1 - current], dev_values[current], dev_indices);


        // Swap buffers (ping-pong)
        current = 1 - current;
    }

    cudaMemcpy(outKeys, dev_keys[current], sizeof(int) * n, cudaMemcpyDeviceToHost);

    // Copy sorted values back to host
    cudaMemcpy(outValues, dev_values[current], sizeof(T) * n, cudaMemcpyDeviceToHost);
    checkCUDAError("Memory copy from device objects array to host failed.");

    // Free device memory
    cudaFree(dev_keys[0]);
    cudaFree(dev_keys[1]);
    cudaFree(dev_values[0]);
    cudaFree(dev_values[1]);
    cudaFree(dev_scan);
}

template void StreamCompaction::Radix::sortByKey<int>(
    int n, int* outKeys, int* outValues, const int* keys, const int* values, const int maxBitLength);

template void StreamCompaction::Radix::sortByKey<float>(
    int n, int* outKeys, float* outValues, const int* keys, const float* values, const int maxBitLength);
