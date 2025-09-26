#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction
{
namespace Thrust
{
using StreamCompaction::Common::PerformanceTimer;

PerformanceTimer& timer()
{
    static PerformanceTimer timer;
    return timer;
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int* odata, const int* idata)
{
    // Copy data from host to device
    thrust::host_vector<int> host_idata(idata, idata + n);  // thrust host vector
    thrust::device_vector<int> dev_idata = host_idata;      // built-in assignment conversion
    thrust::device_vector<int> dev_odata(n);                // for output

    timer().startGpuTimer();

    thrust::exclusive_scan(dev_idata.begin(), dev_idata.end(), dev_odata.begin());

    timer().endGpuTimer();

    // copy result back to host
    thrust::copy(dev_odata.begin(), dev_odata.end(), odata);
}

void radixSort(int n, int* o_data, const int* i_data)
{
    thrust::device_vector<int> d_copy(i_data, i_data + n);

    bool usingTimer = false;
    if (!timer().gpu_timer_started)
    {
        timer().startGpuTimer();
        usingTimer = true;
    }

    thrust::sort(d_copy.begin(), d_copy.end());

    if (usingTimer)
    {
        timer().endGpuTimer();
    }

    thrust::copy(d_copy.begin(), d_copy.end(), o_data);
}

void radixSortByKey(int* d_keys, int* d_vals, int N)
{
    thrust::device_ptr<int> keys(d_keys);
    thrust::device_ptr<int> vals(d_vals);

    thrust::sort_by_key(keys, keys + N, vals);
}
}  // namespace Thrust
}  // namespace StreamCompaction
