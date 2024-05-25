/// https://emre-avci.medium.com/dot-product-in-cuda-c-which-might-outperform-cublas-t-dot-732047aa5ec5
extern "C" __global__ void dot_kernel(float *lhs, float *rhs, float *c, int n)
{
    const int threadsPerBlock = 1024;
    __shared__ float cache[threadsPerBlock];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    float partial = 0.0;
    for (int i = idx; i < n; i += stride)
    {
        partial += lhs[i] * rhs[i];
    }
    cache[threadIdx.x] = partial;
    __syncthreads();

    if (idx == 0.0)
    {
        float sum = 0.0;
        for (int i = 0; i < threadsPerBlock; i++)
        {
            sum += cache[i];
        }
        *c = sum;
    }
}
