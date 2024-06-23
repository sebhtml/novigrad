extern "C" __global__ void gelu_derivative_kernel(float *input, float *output, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    // GELU'(x) ≈ 0.5 * (1 + (4 * x) / (5 * (1 + tanh^2(sqrt(2/5) * x))))
    float x = input[i];
    output[i] = 0.5 * (1.0 + (4.0 * x) / 5.0 * (1.0 + powf(tanh(sqrt(2.0 / 5.0) * x), 2.0)));
}