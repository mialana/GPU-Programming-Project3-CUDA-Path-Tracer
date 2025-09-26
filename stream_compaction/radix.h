#pragma once

#include "common.h"

namespace StreamCompaction
{
namespace Radix
{
StreamCompaction::Common::PerformanceTimer& timer();

__device__ __host__ int _isolateBit(const int num, const int tgtBit);

__global__ void _split(int n, int* data, int* notLSB, const int bit);

__global__ void _scatter(int n, int* odata, const int* idata, const int* scan, const int bit);

void sort(int n, int* odata, const int* idata, const int maxBitLength);
}  // namespace Radix
}  // namespace StreamCompaction
