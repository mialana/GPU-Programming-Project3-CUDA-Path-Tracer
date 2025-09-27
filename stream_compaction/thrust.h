#pragma once

#include "common.h"

namespace StreamCompaction
{
namespace Thrust
{
StreamCompaction::Common::PerformanceTimer& timer();

void scan(int n, int* odata, const int* idata);

void radixSort(int n, int* o_data, const int* i_data);

void radixSortByKey(int n, int* out_keys, int* out_values, const int* in_keys, const int* in_values);
}  // namespace Thrust
}  // namespace StreamCompaction
