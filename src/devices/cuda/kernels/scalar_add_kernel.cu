extern "C" __global__ void scalar_add_kernel(int n, float *vec, float *scalar)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        vec[i] = *scalar + vec[i];
    }
}