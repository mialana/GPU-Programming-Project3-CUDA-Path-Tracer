#pragma once

#include "common.h"

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

namespace StreamCompaction
{
namespace Shared
{

StreamCompaction::Common::PerformanceTimer& timer();

void scan(int n, const int* dev_idata, int* dev_odata, int* dev_blockSums, const int blockSize);

void scanWrapper(int n, int* odata, const int* idata);

int compact(int n,
            const int* dev_idata,
            int* dev_odata,
            int* dev_bools,
            int* dev_indices,
            int* dev_blockSums,
            int blockSize);

int compactWithBools(int n,
                     const int* dev_idata,
                     int* dev_odata,
                     const int* dev_bools,
                     int* dev_indices,
                     int* dev_blockSums,
                     int blockSize);

int compactWrapper(int n, int* odata, const int* idata);

template<typename T>
inline int compactByKey(int n,
                        const T* dev_ivalues,
                        T* dev_ovalues,
                        const int* dev_idata,
                        int* dev_odata,
                        int* dev_bools,
                        int* dev_indices,
                        int* dev_blockSums,
                        int blockSize)
{
    int blocks = divup(n, blockSize);

    Common::kernMapToBoolean<<<blocks, blockSize>>>(n, dev_bools, dev_idata);

    scan(n, dev_bools, dev_indices, dev_blockSums, blockSize);

    Common::kernScatter<int>
        <<<blocks, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);

    cudaDeviceSynchronize();

    Common::kernScatter<T>
        <<<blocks, blockSize>>>(n, dev_ovalues, dev_ivalues, dev_bools, dev_indices);

    int lastIndex;
    int lastBool;

    cudaMemcpy(&lastIndex, &dev_indices[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lastBool, &dev_bools[n - 1], sizeof(int), cudaMemcpyDeviceToHost);

    return lastIndex + lastBool;
}

// Host wrapper for compactByKey. It accepts host arrays for values and keys.
// For example, h_ivalues and h_ikeys are input arrays,
// h_ovalues and h_okeys will receive the compacted results.
// The function returns the number of surviving (compacted) elements.
template<typename T>
inline int compactByKeyWrapper(int n,
                               T* h_ovalues,        // output values (host)
                               int* h_okeys,        // output keys (host)
                               const T* h_ivalues,  // input values (host)
                               const int* h_ikeys)  // input keys (host)
{
    T *d_ivalues, *d_ovalues;
    int *d_ikeys, *d_okeys, *d_bools, *d_indices, *d_blockSums;

    // Allocate device memory
    cudaMalloc((void**)&d_ivalues, n * sizeof(T));
    cudaMalloc((void**)&d_ovalues, n * sizeof(T));
    cudaMalloc((void**)&d_ikeys, n * sizeof(int));
    cudaMalloc((void**)&d_okeys, n * sizeof(int));
    cudaMalloc((void**)&d_bools, n * sizeof(int));
    cudaMalloc((void**)&d_indices, n * sizeof(int));
    int blocks = divup(n, BLOCK_SIZE);
    cudaMalloc((void**)&d_blockSums, blocks * sizeof(int));

    // Copy input data from host to device.
    cudaMemcpy(d_ivalues, h_ivalues, n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ikeys, h_ikeys, n * sizeof(int), cudaMemcpyHostToDevice);

    // Call the templated device function from shared.h.
    // (This kernel launches both key and value scatter.)
    int count
        = StreamCompaction::Shared::compactByKey<T>(n,
                                                    d_ivalues,    // device input values
                                                    d_ovalues,    // device output values
                                                    d_ikeys,      // device input keys
                                                    d_okeys,      // device output keys
                                                    d_bools,      // temporary device bool array
                                                    d_indices,    // temporary device indices array
                                                    d_blockSums,  // temporary device blockSums array
                                                    BLOCK_SIZE);

    // Copy compacted results back to host.
    cudaMemcpy(h_ovalues, d_ovalues, count * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_okeys, d_okeys, count * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory.
    cudaFree(d_ivalues);
    cudaFree(d_ovalues);
    cudaFree(d_ikeys);
    cudaFree(d_okeys);
    cudaFree(d_bools);
    cudaFree(d_indices);
    cudaFree(d_blockSums);

    return count;
}

}  // namespace Shared
}  // namespace StreamCompaction
