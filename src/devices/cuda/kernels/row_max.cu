#define BLOCK_SIZE 32 * 32

extern "C" __global__ void row_max_kernel(const float *input, float *output, int rows, int cols)
{
    // Thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = threadIdx.y;

    // Thread-local maximum for each thread
    float row_max = 0.0f;

    // Loop through columns to find row-wise maximum
    for (int j = 0; j < cols; ++j)
    {
        if (i < rows && j < cols)
        {
            row_max = fmaxf(row_max, input[i * cols + j]);
        }
    }

    // Only thread 0 of each block writes to output (avoid race conditions)
    if (j == 0  && i < rows)
    {
        output[i] = row_max;
    }
}
