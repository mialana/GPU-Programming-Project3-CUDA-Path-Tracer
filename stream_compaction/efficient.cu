#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

using StreamCompaction::Common::PerformanceTimer;

PerformanceTimer& StreamCompaction::Efficient::timer()
{
    static PerformanceTimer timer;
    return timer;
}

__global__ void kernel_efficientUpSweep(const unsigned long long paddedN,
                                        const int stride,
                                        const int prevStride,
                                        int* scan)
{
    int strideIdx = blockIdx.x * blockDim.x + threadIdx.x;  // 0, 1, 2, 3... (like normal)
                                                            // but this is not target elem index

    unsigned long long strideStart = strideIdx * stride;    // index where this stride starts

    // last index in stride. accumulated value of stride always goes here
    unsigned long long accumulatorIdx = strideStart + stride - 1;

    if (accumulatorIdx >= paddedN)
    {
        return;
    }

    int accumulator = scan[accumulatorIdx];  // pre-fetch accumulator's value

    // this new stride has swallowed two strides total
    // siblingIdx is the index of the other stride that now no longer exists
    unsigned long long siblingIdx = strideStart + prevStride - 1;  // doesn't depend on accumulator

    scan[accumulatorIdx] = accumulator + scan[siblingIdx];
}

__global__ void kernel_efficientDownSweep(const unsigned long long paddedN,
                                          const int STRIDE,
                                          const int nextSTRIDE,  // nextStride == (stride / 2)
                                          int* scan)
{
    int strideIdx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long strideStart = strideIdx * STRIDE;

    unsigned long long rightChildIdx = strideStart + STRIDE - 1;
    if (rightChildIdx >= paddedN)
    {
        return;
    }

    int rightChild = scan[rightChildIdx];

    // leftChild and rightChild are nextSTRIDE indices apart
    unsigned long long leftChildIdx = strideStart + nextSTRIDE - 1;
    int leftChild = scan[leftChildIdx];  // does not depend on first memory read

    // give left child right child's value
    // its value has not changed since the end of upsweep
    // it has it easier than right child.
    // on this update it now has accumulated vals of all strides of size nextSTRIDE, besides its own
    scan[leftChildIdx] = rightChild;  // depends on first read, but not second

    // right child currently contains accumulated vals of all strides of size STRIDE besodes its own
    // adding the left child, which only contains values of one stride of size nextSTRIDE
    // means that right child now also has accumulated vals of all strides of size nextSTRIDE
    // besides its own (same status as leftChild)
    scan[rightChildIdx] = rightChild + leftChild;  // memory writes do not depend on each other

    // summary: at each layer, the updated elements get the value of all strides of size nextSTRIDE
    // besides its own.
    // so when nextSTRIDE == 1, then this element is done, and so are our iterations
}

/*
    the inner operation of scan without timers and allocation.
    note: dev_scan should be pre-allocated to the padded power of two size
*/
void StreamCompaction::Efficient::scan(int n, int* dev_scan, const int blockSize)
{
    // unsigned long long numLayers = ilog2ceil(n);
    int numLayers = ilog2ceil(n);
    unsigned long long paddedN = 1 << numLayers;  // pad to nearest power of 2

    int prevStride = 1;                           // 1, 2, 4, 8, ... n/2
    int stride = 2;  // essentially the amount of indices that are accumulated into 1 at this iter
                     // 2, 4, 8, ... n
    for (int iter = 0; iter < numLayers; iter++)
    {
        // paddedN >> (iter + 1) == paddedN / (iter + 2) = the number of active threads in this iter
        // n/2, n/4, n/8, ... 1
        int blocks = divup(paddedN >> (iter + 1), BLOCK_SIZE);
        kernel_efficientUpSweep<<<blocks, BLOCK_SIZE>>>(paddedN, stride, prevStride, dev_scan);
        checkCUDAError("Perform Work-Efficient Scan Up Sweep Iteration CUDA kernel failed.");

        prevStride = stride;
        stride = stride <<= 1;
    }

    // set last value of dev_scan to 0
    int replacement = 0;
    cudaMemcpy(&dev_scan[paddedN - 1], &replacement, sizeof(int), cudaMemcpyHostToDevice);

    stride = paddedN;               // n, n/2, n/4, ... 2
    int nextStride = paddedN >> 1;  // n/2, n/4, ... 1
    for (int iter = numLayers; iter > 0; iter--)
    {
        // paddedN >> iter == number of active threads in this iter. 1, 2, 4, 8, ... n/2
        int blocks = divup(paddedN >> iter, BLOCK_SIZE);
        kernel_efficientDownSweep<<<blocks, BLOCK_SIZE>>>(paddedN, stride, nextStride, dev_scan);
        checkCUDAError("Perform Work-Efficient Scan Down Sweep Iteration CUDA kernel failed.");

        stride = nextStride;
        nextStride >>= 1;  // n/2, n/4, n/8, n/16, ...
    }
}

/************************************************************************************************ */

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void StreamCompaction::Efficient::scanWrapper(int n, int* odata, const int* idata)
{
    unsigned long long numLayers = ilog2ceil(n);
    unsigned long long paddedN = 1 << ilog2ceil(n);

    // create two device arrays
    int* dev_scan;

    cudaMalloc((void**)&dev_scan, sizeof(int) * paddedN);
    checkCUDAError("CUDA malloc for scan array failed.");

    cudaMemcpy(dev_scan, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
    checkCUDAError("Memory copy from input data to scan array failed.");

    cudaDeviceSynchronize();

    bool usingTimer = false;
    if (!timer().gpu_timer_started)  // added in order to call `scan` from other functions.
    {
        timer().startGpuTimer();
        usingTimer = true;
    }

    scan(n, dev_scan, BLOCK_SIZE);

    if (usingTimer)
    {
        timer().endGpuTimer();
    }

    cudaMemcpy(odata, dev_scan, sizeof(int) * n, cudaMemcpyDeviceToHost);

    cudaFree(dev_scan);  // can't forget memory leaks!
}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int StreamCompaction::Efficient::compact(int n, int* odata, const int* idata)
{
    // TODO: these arrays are unnecessary. will optimize soon.

    // create device arrays
    int* dev_idata;
    int* dev_odata;

    int* dev_bools;
    int* dev_indices;

    cudaMalloc((void**)&dev_idata, sizeof(int) * n);
    checkCUDAError("CUDA malloc for idata array failed.");

    cudaMalloc((void**)&dev_odata, sizeof(int) * n);
    checkCUDAError("CUDA malloc for odata array failed.");

    cudaMalloc((void**)&dev_bools, sizeof(int) * n);
    checkCUDAError("CUDA malloc for bools array failed.");

    cudaMalloc((void**)&dev_indices, sizeof(int) * n);
    checkCUDAError("CUDA malloc for indices array failed.");

    cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
    checkCUDAError("Memory copy from input data to idata array failed.");
    cudaMemcpy(dev_bools, odata, sizeof(int) * n, cudaMemcpyHostToDevice);
    checkCUDAError("Memory copy from output data to odata array failed.");

    cudaDeviceSynchronize();

    int* indices = new int[n];  // create cpu side indices array
    int* bools = new int[n];

    StreamCompaction::Efficient::timer().startGpuTimer();

    int blocks = divup(n, BLOCK_SIZE);

    // reuse dev_idata for bools
    Common::kernMapToBoolean<<<blocks, BLOCK_SIZE>>>(n, dev_bools, dev_idata);

    cudaMemcpy(bools, dev_bools, sizeof(int) * n, cudaMemcpyDeviceToHost);
    checkCUDAError("Memory copy from device bools to indices array failed.");

    scanWrapper(n, indices, bools);

    cudaMemcpy(dev_indices, indices, sizeof(int) * n, cudaMemcpyHostToDevice);
    checkCUDAError("Memory copy from indices to device indices array failed.");

    Common::kernScatter<<<blocks, BLOCK_SIZE>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);

    StreamCompaction::Efficient::timer().endGpuTimer();

    cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);

    cudaFree(dev_idata);
    cudaFree(dev_odata);
    cudaFree(dev_bools);
    cudaFree(dev_indices);

    return indices[n - 1] + bools[n - 1];
}
