// TODO use a smarter reduce instead of doing the same reduction in every thread !

extern "C" __global__ void softmax_kernel(float *input, float *output, int rows, int cols)
{
    // Get thread and block indices
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check for boundary conditions
    if (idx >= rows * cols)
    {
        return;
    }

    // Calculate element index based on row and column
    int row = idx / cols;
    int col = idx % cols;

    // Find the maximum value in the current row
    float max_val = input[row * cols + 0];
    for (int j = 1; j < cols; j++)
    {
        max_val = fmaxf(max_val, input[row * cols + j]);
    }

    // Subtract the maximum value for numerical stability
    float exp_val = expf(input[row * cols + col] - max_val);

    // Normalize the values using the calculated sum
    float exp_sum = 0.0f;
    for (int j = 0; j < cols; j++)
    {
        float exp_val = expf(input[row * cols + j] - max_val);
        exp_sum += exp_val;
    }

    output[row * cols + col] = exp_val / exp_sum;
}
