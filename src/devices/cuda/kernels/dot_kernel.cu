#define THREADS_PER_BLOCK 1024

extern "C" __global__ void dot_kernel(float *a, float *b, float *c, int n)
{
    // Thread index within the block
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Shared memory to store partial sums
    __shared__ float cache[THREADS_PER_BLOCK];

    // Initialize partial sum for this thread
    float temp = 0.0f;

    // Loop until we've processed all elements (assuming vectors are same size)
    while (tid < n)
    {
        // Multiply corresponding elements and accumulate in temp
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    // Store partial sum in shared memory
    cache[threadIdx.x] = temp;

    // Synchronize threads within the block
    __syncthreads();

    // Parallel reduction to sum partial sums in shared memory
    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (threadIdx.x < i)
        {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Write final result to output array if this is the first thread
    if (threadIdx.x == 0)
    {
        c[blockIdx.x] = cache[0];
    }
}