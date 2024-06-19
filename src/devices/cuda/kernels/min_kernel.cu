extern "C" __global__ void min_kernel(float *a, float *b, float *c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n)
    {
        c[tid] = fminf(a[tid], b[tid]);
    }
}
