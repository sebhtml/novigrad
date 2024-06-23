extern "C" __global__ void gelu_kernel(float *input, float *output, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    // GELU(x) ≈ 0.5 * x * (1 + tanh(x * sqrt(2 / 5)))
    float x = input[i];
    output[i] = 0.5 * x * (1.0 + tanh(x * sqrt(2.0 / 5.0)));
}