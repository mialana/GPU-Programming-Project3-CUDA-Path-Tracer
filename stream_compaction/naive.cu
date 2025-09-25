#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction
{
namespace Naive
{
using StreamCompaction::Common::PerformanceTimer;

PerformanceTimer& timer()
{
    static PerformanceTimer timer;
    return timer;
}

// scanA is input and scanB is output for this iteration
__global__ void kernel_performNaiveScanIteration(const int n,
                                                 const int iter,
                                                 const int* scanA,
                                                 int* scanB)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int iter_startIdx = exp2f(iter - 1);
    if (index >= n)
    {
        return;
    }

    if (index < iter_startIdx)
    {
        scanB[index] = scanA[index];
    } else
    {
        scanB[index] = scanA[index - iter_startIdx] + scanA[index];
    }

    // profile time efficiency
    // scanB[index] = index < iter_startIdx ? scanA[index]
    //                                      : scanA[index - iter_startIdx] + scanA[index];
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int* odata, const int* idata)
{
    // create two device arrays to ping-pong between
    int* dev_scanA;
    int* dev_scanB;

    cudaMalloc((void**)&dev_scanA, sizeof(int) * n);
    checkCUDAError("CUDA malloc for scan array A failed.");

    cudaMalloc((void**)&dev_scanB, sizeof(int) * n);
    checkCUDAError("CUDA malloc for scan array B failed.");

    cudaMemcpy(dev_scanA, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
    checkCUDAError("Memory copy from input data to scan array A failed.");
    cudaMemcpy(dev_scanB, odata, sizeof(int) * n, cudaMemcpyHostToDevice);
    checkCUDAError("Memory copy from output data to scan array B failed.");

    cudaDeviceSynchronize();

    timer().startGpuTimer();

    int blocks = divup(n, BLOCK_SIZE);

    for (int i = 1; i <= ilog2ceil(n); i++)
    {
        kernel_performNaiveScanIteration<<<blocks, BLOCK_SIZE>>>(n, i, dev_scanA, dev_scanB);
        checkCUDAError("Perform Naive Scan Iteration CUDA kernel failed.");

        // ping-pong
        int* temp = dev_scanA;
        dev_scanA = dev_scanB;
        dev_scanB = temp;
    }

    // result ends up in dev_scanA
    Common::kernel_inclusiveToExclusive<<<blocks, BLOCK_SIZE>>>(n, 0, dev_scanA, dev_scanB);
    checkCUDAError("Inclusive to Exclusive CUDA kernel failed.");

    timer().endGpuTimer();

    cudaMemcpy(odata,
               dev_scanB,
               sizeof(int) * n,
               cudaMemcpyDeviceToHost);  // result ends up in scanB

    cudaFree(dev_scanA);
    cudaFree(dev_scanB);  // can't forget memory leaks!
}
}  // namespace Naive
}  // namespace StreamCompaction
