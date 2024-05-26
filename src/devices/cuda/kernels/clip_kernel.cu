extern "C" __global__ void clip_kernel(float *min, float *max, float *input, float *output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }

    float x = input[idx];
    x = fmaxf(x, *min);
    x = fminf(x, *max);
    output[idx] = x;
}