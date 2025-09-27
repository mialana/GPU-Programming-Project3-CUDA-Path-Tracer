#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/detail/vector_base.h>
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

using thrust::host_vector;

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

void radixSortByKey(int n, int* out_keys, int* out_values, const int* in_keys, const int* in_values)
{
    // Wrap raw pointers with Thrust device pointers
    thrust::device_vector<int> d_keys(in_keys, in_keys + n);
    thrust::device_vector<int> d_values(in_values, in_values + n);

    bool usingTimer = false;
    if (!timer().gpu_timer_started)
    {
        timer().startGpuTimer();
        usingTimer = true;
    }

    // Sort keys and reorder values accordingly
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());

    if (usingTimer)
    {
        timer().endGpuTimer();
    }

    // Copy sorted keys and values back to host
    thrust::copy(d_keys.begin(), d_keys.end(), out_keys);
    thrust::copy(d_values.begin(), d_values.end(), out_values);
}
}  // namespace Thrust
}  // namespace StreamCompaction
