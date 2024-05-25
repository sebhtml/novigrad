extern "C" __global__ void dot_kernel(float *a, float *b, float *c, int n)
{
    const int threadsPerBlock = 1024;
    __shared__ float cache[threadsPerBlock];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        float sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += a[i] * b[i];
        }
        *c =sum;
    }
}
