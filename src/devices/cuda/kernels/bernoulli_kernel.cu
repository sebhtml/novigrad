__global__ void readAddress(int *a, int *b)
{
    int expected;
    int new_value;
    int swapped;

    do
    {
        expected = *a;
        new_value = *a;
        swapped = atomicCAS(a, expected, new_value);
        // CAS failed if swapped != expected_a (another thread modified a)
    } while (swapped != expected);
}

__global__ void swap(int *a, int *b)
{
    int expected;
    int new_value;
    int swapped;

    do
    {
        //readAddress(a, &expected);
        //readAddress(b, &new_value);
        expected = *a;
        new_value = *b;
        swapped = atomicCAS(a, expected, new_value);
        // CAS failed if swapped != expected_a (another thread modified a)
    } while (swapped != expected);
}

#define uint32_t unsigned int

extern "C" __global__ void bernoulli_kernel2(float *input, float *output, uint32_t n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }

    // Assign a value to each output index
    float probability = input[i];
    uint32_t count_of_ones = n * probability;
    float value = i < count_of_ones ? 1.0 : 0.0;
    output[i] = value;
    __syncthreads();

    // shuffle using atomicCAS
    uint32_t rounds = 32;
    for (uint32_t round = 0; round < rounds; round++)
    {
        for (uint32_t stride = 0; stride < n; stride++)
        {
            // swap with another output index
            uint32_t j = (i + i + stride) % n;
            swap((int *)(output + i), (int *)(output + j));
            __syncthreads();
        }
    }
}

__global__ void clock_block(clock_t clock_count)
{
    clock_t start_clock = clock();
    clock_t clock_offset = 0;
    while (clock_offset < clock_count)
    {
        clock_offset = clock() - start_clock;
    }
}

__device__ int global_output_index = 0;

extern "C" __global__ void bernoulli_kernel(float *input, float *output, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }

    float probability = input[i];
    int count_of_ones = n * probability;
    float value = 0.0;// i < count_of_ones ? 1.0 : 0.0;
    //clock_block(i * 1024);
    int output_index = atomicAdd(&global_output_index, 1);
    output[output_index] = value;
}