#include "common.h"

void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}

namespace StreamCompaction
{
namespace Common
{

/**
 * Maps an array to an array of 0s and 1s for stream compaction. Elements
 * which map to 0 will be removed, and elements which map to 1 will be kept.
 */
__global__ void kernMapToBoolean(int n, int* bools, const int* idata)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n)
    {
        return;
    }

    bools[index] = idata[index] == 0 ? 0 : 1;
}

template __global__ void kernScatter<int>(
    int n, int* odata, const int* idata, const int* bools, const int* indices);

__global__ void kernel_inclusiveToExclusive(int n, int identity, const int* iData, int* oData)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n)
    {
        return;
    } else if (index == 0)
    {
        oData[index] = identity;
    } else
    {
        oData[index] = iData[index - 1];
    }
}

__global__ void kernel_setDeviceArrayValue(int* arr, const int index, const int value)
{
    arr[index] = value;  // round up to nearest power of two
}

}  // namespace Common
}  // namespace StreamCompaction
