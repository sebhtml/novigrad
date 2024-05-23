extern "C" __global__ void mul_kernel(float *a, float *b, float *c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n)
    {
        c[tid] = a[tid] * b[tid];
    }
}
