#include "radix.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "efficient.h"

namespace StreamCompaction
{
namespace Radix
{
using StreamCompaction::Common::PerformanceTimer;

PerformanceTimer& timer()
{
    static PerformanceTimer timer;
    return timer;
}

__device__ __host__ int _isolateBit(const int num, const int tgtBit)
{
    return (num >> tgtBit) & 1;
}

__global__ void _split(int n, int* data, int* notBit, const int tgtBit)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index >= n)
    {
        return;
    }

    notBit[index] = _isolateBit(data[index], tgtBit) ^ 1;  // not(target bit)
}

__global__ void _scatter(int n, int* odata, const int* idata, const int* scan, const int tgtBit)
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
    int address = _isolateBit(idata[index], tgtBit) ? index + (totalFalses - scan[index])
                                                    : scan[index];

    __syncthreads();

    odata[address] = idata[index];
}

void sort(int n, int* odata, const int* idata, const int maxBitLength)
{
    const unsigned numLayers = ilog2ceil(n);
    const unsigned paddedN = 1 << ilog2ceil(n);

    // create device arrays
    int* dev_idata;
    int* dev_odata;
    int* dev_scan;

    cudaMalloc((void**)&dev_idata, sizeof(int) * n);
    checkCUDAError("CUDA malloc for device in data array failed.");

    cudaMalloc((void**)&dev_odata, sizeof(int) * n);
    checkCUDAError("CUDA malloc for device out data array failed.");

    cudaMalloc((void**)&dev_scan, sizeof(int) * paddedN);
    checkCUDAError("CUDA malloc for device scan array failed.");

    cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
    checkCUDAError("Memory copy from host idata to device array failed.");

    bool usingTimer = false;
    if (!timer().gpu_timer_started)
    {
        timer().startGpuTimer();
        usingTimer = true;
    }

    for (int tgtBit = 0; tgtBit < maxBitLength; tgtBit++)
    {
        unsigned blocks = divup(n, BLOCK_SIZE);
        _split<<<blocks, BLOCK_SIZE>>>(n, dev_idata, dev_scan, tgtBit);

        Efficient::scanHelper(numLayers, paddedN, dev_scan);

        _scatter<<<blocks, BLOCK_SIZE>>>(n, dev_odata, dev_idata, dev_scan, tgtBit);

        int* temp = dev_idata;
        dev_idata = dev_odata;
        dev_odata = temp;
    }

    if (usingTimer)
    {
        timer().endGpuTimer();
    }

    cudaMemcpy(odata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);
    checkCUDAError("Memory copy from device array to host odata failed.");

    cudaFree(dev_idata);
    cudaFree(dev_odata);
    cudaFree(dev_scan);
}

}  // namespace Radix
}  // namespace StreamCompaction
