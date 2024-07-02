extern "C" __global__ void pow_kernel(float *a, float *b, float *c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n)
    {
        c[tid] = powf(a[tid], b[tid]);
    }
}
