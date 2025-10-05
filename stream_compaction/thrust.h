#pragma once

#include "common.h"

namespace StreamCompaction
{
namespace Thrust
{

struct IsNonZero
{
    __host__ __device__ bool operator()(const int x) const
    {
        return x != 0;
    }
};

StreamCompaction::Common::PerformanceTimer& timer();

void scan(int n, int* odata, const int* idata);

void radixSort(int n, int* o_data, const int* i_data);

void radixSortByKey(int n, int* out_keys, int* out_values, const int* in_keys, const int* in_values);

int compactByKey(int n, int* out_keys, float* out_values, const int* in_keys, const float* in_values);
}  // namespace Thrust
}  // namespace StreamCompaction
