/// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
/// https://emre-avci.medium.com/dot-product-in-cuda-c-which-might-outperform-cublas-t-dot-732047aa5ec5
/// https://www.reddit.com/r/CUDA/comments/194cdhn/reference_implementation_for_optimized_sum_reduce/
extern "C" __global__ void dot_kernel(float *lhs, float *rhs, float *c, int n)
{
    const int block_dim = 1024;
    __shared__ float shared_mem[block_dim];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0)
    {
        *c = 0;
    }

    // Each thread in a block do a partial sum.
    int stride = blockDim.x * gridDim.x;
    float partial = 0.0;
    for (int i = idx; i < n; i += stride)
    {
        partial += lhs[i] * rhs[i];
    }
    shared_mem[threadIdx.x] = partial;
    __syncthreads();

    // Parallel reduction.
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + s];
        }
        __syncthreads();
    }

    // The first thread of each block adds its block sum.
    if (threadIdx.x == 0)
    {
        atomicAdd(c, shared_mem[0]);
    }
}
