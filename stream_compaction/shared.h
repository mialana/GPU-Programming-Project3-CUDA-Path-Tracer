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

int compact(int n, int* odata, const int* idata);
}  // namespace Shared
}  // namespace StreamCompaction
