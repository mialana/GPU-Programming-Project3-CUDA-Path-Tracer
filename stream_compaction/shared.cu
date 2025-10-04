#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "shared.h"

using StreamCompaction::Common::PerformanceTimer;

PerformanceTimer& StreamCompaction::Shared::timer()
{
    static PerformanceTimer timer;
    return timer;
}

__device__ __host__ int _offset(int idx)
{
    return idx + CONFLICT_FREE_OFFSET(idx);
}

__global__ void kernel_scanIntraBlockShared(const unsigned long long paddedN,
                                            const int* idata,
                                            int* odata,
                                            int* blockSums)
{
    extern __shared__ int mat[];

    const int tileSize = blockDim.x * 2;

    const int tid = threadIdx.x;

    unsigned long long blockOffset = (blockIdx.x * blockDim.x) * 2;
    int threadOffset = 2 * tid;  // first index this thread is responsible for

    unsigned long long globalThreadIdx = blockOffset + threadOffset;

    // global memory is read from in coalesced fashion
    // ensure some threads do not return early without zero-padding the shared matrix
    mat[_offset(threadOffset)] = (globalThreadIdx < paddedN) ? idata[blockOffset + threadOffset]
                                                             : 0;
    mat[_offset(threadOffset + 1)] = (globalThreadIdx + 1 < paddedN)
                                         ? idata[blockOffset + threadOffset + 1]
                                         : 0;

    // which stride each child is reponsible for -- constant per thread
    // in reality, it is one stride higher than expected, but that's due to -1
    const int strideIdx_firstChild = threadOffset + 1;
    const int strideIdx_secondChild = threadOffset + 2;

    int STRIDE = 1;  // 1, 2, 4, 8, 16, 32, ... tileSize
    // activeThreads: n/2, n/4, n/8, ... 1
    for (int activeThreads = tileSize >> 1; activeThreads > 0; activeThreads >>= 1)
    {
        __syncthreads();

        if (tid < activeThreads)
        {
            int firstIdx = strideIdx_firstChild * STRIDE - 1;
            int secondIdx = strideIdx_secondChild * STRIDE - 1;

            mat[_offset(secondIdx)] += mat[_offset(firstIdx)];
        }

        STRIDE *= 2;
    }

    __syncthreads();

    if (tid == 0)
    {
        blockSums[blockIdx.x] = mat[_offset(tileSize - 1)];  // write accumulated val of block
        mat[_offset(tileSize - 1)] = 0;                      // clear last element
    }

    for (int activeThreads = 1; activeThreads < tileSize; activeThreads <<= 1)
    {
        STRIDE >>= 1;  // STRIDE ended at tileSize
        __syncthreads();

        if (tid < activeThreads)
        {
            int firstIdx = strideIdx_firstChild * STRIDE - 1;
            int secondIdx = strideIdx_secondChild * STRIDE - 1;

            firstIdx = _offset(firstIdx);
            secondIdx = _offset(secondIdx);
            int temp = mat[firstIdx];
            mat[firstIdx] = mat[secondIdx];
            mat[secondIdx] += temp;
        }
    }

    __syncthreads();  // this time the last __syncthreads() wasn't called

    if (globalThreadIdx < paddedN)
    {
        odata[blockOffset + threadOffset] = mat[_offset(threadOffset)];
    }
    if (globalThreadIdx + 1 < paddedN)
    {
        odata[blockOffset + threadOffset + 1] = mat[_offset(threadOffset + 1)];
    }
}

__global__ void kernel_addBlockSums(int n, int* dev_data, const int* dev_blockSums)
{
    __shared__ int blockOffset;

    if (threadIdx.x == 0)
    {
        blockOffset = dev_blockSums[blockIdx.x];
    }

    __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n)
    {  // should be safe to return now
        return;
    }

    dev_data[index] += blockOffset;
}

/*
    the inner operation of scan without timers and allocation.
    note: dev_scan should be pre-allocated to the padded power of two size
*/
void StreamCompaction::Shared::scan(
    int n, const int* dev_idata, int* dev_odata, int* dev_blockSums, const int blockSize)
{
    unsigned long long paddedN = 1 << ilog2ceil(n);  // pad to nearest power of 2

    const int blockSpan = blockSize * 2;
    // perform scan on the block level
    int numBlocks = divup(paddedN, blockSpan);

    // numBlocks, numThreads, shared mem size
    kernel_scanIntraBlockShared<<<numBlocks, blockSize, _offset(blockSpan) * sizeof(int)>>>(
        paddedN, dev_idata, dev_odata, dev_blockSums);

    if (numBlocks > 1)
    {
        // Allocate temporary buffer for recursive scan of block sums
        int* dev_newOData;
        cudaMalloc((void**)&dev_newOData, sizeof(int) * numBlocks);
        checkCUDAError("CUDA malloc for recursive block sums failed.");

        // Recursively scan the block sums
        scan(numBlocks, dev_blockSums, dev_newOData, dev_blockSums, blockSize);

        // Add the recursively scanned block sums to the output
        kernel_addBlockSums<<<numBlocks, blockSpan>>>(paddedN, dev_odata, dev_newOData);

        // Free the temporary buffer
        cudaFree(dev_newOData);
    }
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void StreamCompaction::Shared::scanWrapper(int n, int* odata, const int* idata)
{
    int numLayers = ilog2ceil(n);
    unsigned long long paddedN = 1 << ilog2ceil(n);

    int totalBlocks = divup(paddedN, 2 * BLOCK_SIZE);

    // create two device arrays
    int* dev_idata;
    int* dev_odata;
    int* dev_blockSums;

    cudaMalloc((void**)&dev_idata, sizeof(int) * paddedN);
    checkCUDAError("CUDA malloc for scan array failed.");

    cudaMalloc((void**)&dev_odata, sizeof(int) * paddedN);
    checkCUDAError("CUDA malloc for device out data failed.");

    // create new array to store total sum of each block
    cudaMalloc((void**)&dev_blockSums, sizeof(int) * totalBlocks);
    checkCUDAError("CUDA malloc for block sums array failed.");

    cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
    checkCUDAError("Memory copy from input data to device idata array failed.");

    cudaDeviceSynchronize();

    bool usingTimer = false;
    if (!timer().gpu_timer_started)  // added in order to call `scan` from other functions.
    {
        timer().startGpuTimer();
        usingTimer = true;
    }

    StreamCompaction::Shared::scan(n, dev_idata, dev_odata, dev_blockSums, BLOCK_SIZE);

    if (usingTimer)
    {
        timer().endGpuTimer();
    }

    cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);  // only copy n elements

    cudaFree(dev_idata);  // can't forget memory leaks!
    cudaFree(dev_odata);
    cudaFree(dev_blockSums);
}

/************************************************************************************************ */
