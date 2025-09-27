#pragma once

#include "common.h"

namespace StreamCompaction
{
namespace Radix
{
StreamCompaction::Common::PerformanceTimer& timer();

__device__ __host__ int _isolateBit(const int num, const int tgtBit);

__global__ void _split(int n, int* data, int* notLSB, const int bit);

__global__ void _computeScatterIndices(int n, int* odata, const int* idata, const int* scan, const int tgtBit);

template<typename T>
__global__ void _scatter(int n, T* odata, const T* idata, const int* addresses);

void sort(int n, int* odata, const int* idata, const int maxBitLength);

template<typename T>
void sortByKey(int n, int* dev_keys[2], T* dev_values[2], int* dev_scan, int* dev_indices, const int maxBitLength);

template<typename T>
void sortByKeyWrapper(
    int n, int* outKeys, T* outValues, const int* keys, const T* values, const int maxBitLength);

}  // namespace Radix
}  // namespace StreamCompaction
