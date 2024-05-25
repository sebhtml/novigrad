extern "C" __global__ void sqrt_kernel(float *input, float *output, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        output[i] = sqrtf(input[i]);
    }
}