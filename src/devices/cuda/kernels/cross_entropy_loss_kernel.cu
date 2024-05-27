/// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
extern "C" __global__ void cross_entropy_loss_kernel(float *expected, float *actual, float *loss, int n, float epsilon)
{
    const int block_dim = 1024;
    __shared__ float shared_mem[block_dim];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    float partial = 0.0;
    if (idx == 0)
    {
        for (int i = 0; i < n; i++)
        {
            partial += expected[i] * logf(actual[i] + epsilon);
        }
        *loss = partial;
    }
/*
    return;

    if (idx == 0)
    {
        *loss = 0.0;
    }

    // Each thread in a block do a partial sum.
    int stride = blockDim.x * gridDim.x;
    float partial = 0.0;
    for (int i = idx; i < n; i += stride)
    {
        partial += expected[i] * logf(actual[i] + epsilon);
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
        atomicAdd(loss, -shared_mem[0]);
    }
    */
}
