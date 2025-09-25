#include "cpu.h"

#include "common.h"

namespace StreamCompaction
{
namespace CPU
{
using StreamCompaction::Common::PerformanceTimer;

PerformanceTimer& timer()
{
    static PerformanceTimer timer;
    return timer;
}

/**
 * CPU scan (prefix sum).
 * For performance analysis, this is supposed to be a simple for loop.
 * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan
 * in this function first.
 */
void scan(int n, int* odata, const int* idata)
{
    bool usingTimer = false;
    if (!timer().cpu_timer_started)  // added in order to call `scan` from other functions.
    {
        timer().startCpuTimer();
        usingTimer = true;
    }

    odata[0] = 0;             // identity is 0

    int prev_sum = idata[0];  // save prev sum for access ease
    for (int j = 1; j < n; j++)
    {
        odata[j] = prev_sum;
        prev_sum += idata[j];
    }

    if (usingTimer)
    {
        timer().endCpuTimer();
    }
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int* odata, const int* idata)
{
    timer().startCpuTimer();

    int outIndex = 0;  // pointer to current progress in out array

    for (int i = 0; i < n; i++)
    {
        int inVal = idata[i];
        if (inVal != 0)
        {
            odata[outIndex] = inVal;
            outIndex++;
        }
    }

    timer().endCpuTimer();
    return outIndex;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int* odata, const int* idata)
{
    timer().startCpuTimer();

    int* isNotZero = new int[n];
    int* scan_isNotZero = new int[n];

    for (int i = 0; i < n; i++)
    {
        isNotZero[i] = idata[i] != 0 ? 1 : 0;  // val is 1 at i if idata[i] != 0, else 0
    }

    scan(n, scan_isNotZero, isNotZero);  // scan result is index in final array

    for (int i = 0; i < n; i++)
    {
        if (isNotZero[i])
        {
            odata[scan_isNotZero[i]] = idata[i];
        }
    }

    timer().endCpuTimer();

    return scan_isNotZero[n - 1] + isNotZero[n - 1];  // due to exclusive scan
}
}  // namespace CPU
}  // namespace StreamCompaction
