#pragma once

#include "common.h"

namespace StreamCompaction
{
namespace Thrust
{
StreamCompaction::Common::PerformanceTimer& timer();

void scan(int n, int* odata, const int* idata);

void radixSort(int n, int* o_data, const int* i_data);
void radixSortByKey(int* d_keys, int* d_vals, int N);

}  // namespace Thrust
}  // namespace StreamCompaction
