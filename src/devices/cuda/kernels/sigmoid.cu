extern "C" __global__ void sigmoid(float *input, float *output, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        output[i] = 1.0f / (1.0f + expf(-input[i]));
    }
}