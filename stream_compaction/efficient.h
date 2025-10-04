#pragma once

#include "common.h"

namespace StreamCompaction
{
namespace Efficient
{
StreamCompaction::Common::PerformanceTimer& timer();

void scan(int n, int* dev_scan, const int blockSize);

void scanWrapper(int n, int* odata, const int* idata);

int compact(int n, int* odata, const int* idata);
}  // namespace Efficient
}  // namespace StreamCompaction
