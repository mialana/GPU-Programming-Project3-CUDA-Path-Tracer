#pragma once

#include "common.h"

namespace StreamCompaction
{
namespace Efficient
{
StreamCompaction::Common::PerformanceTimer& timer();

void scanHelper(int numLayers, int paddedN, int* dev_scan);

void scan(int n, int* odata, const int* idata);

int compact(int n, int* odata, const int* idata);
}  // namespace Efficient
}  // namespace StreamCompaction
