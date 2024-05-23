extern "C" __global__ void scalar_mul_kernel(float scalar, float *A, int rows, int cols)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows && j < cols)
    {
        A[i * cols + j] *= scalar;
    }
}