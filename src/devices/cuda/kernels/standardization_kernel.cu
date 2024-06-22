// TODO use a smarter reduce instead of doing the same reduction in every thread !
// TODO pass EPSILON as argument.
extern "C" __global__ void standardization_kernel(float *input, float *output, int rows, int cols)
{
    const float EPSILON = 1e-8;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * cols;
    if (idx >= n)
    {
        return;
    }

    int row = idx / cols;
    int col = idx % cols;

    float row_sum = 0.0;
    for (int j = 0; j < cols; j++)
    {
        row_sum += input[row * cols + j];
    }
    float row_mean = row_sum / cols;

    float row_diff_sum = 0.0;
    for (int j = 0; j < cols; j++)
    {
        float diff = input[row * cols + j] - row_mean;
        row_diff_sum += diff * diff;
    }
    float row_stddev = sqrt(row_diff_sum / cols);

    float x = input[row * cols + col];
    output[row * cols + col] = (x - row_mean) / (row_stddev + EPSILON);
}
