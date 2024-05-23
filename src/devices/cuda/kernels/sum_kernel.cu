extern "C" __global__ void sum_kernel(float *data, int size, float *result)
{
    int i = threadIdx.x;
    if (i < size)
    {
        atomicAdd(result, data[i]);
    }
}