#define uint64_t unsigned long long
#define uint32_t unsigned int

/// https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
extern "C" __global__ void bernoulli_kernel(float *input, float *output, int n, uint64_t *state)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;


    if (i >= n) {
        return;
    }

    float probability = input[i];
    uint64_t local_state = *state;
    float random_value;

    for (int j = 0; j < i + 1; j++) {
        local_state ^= local_state >> 12;
        local_state ^= local_state << 25;
        local_state ^= local_state >> 27;

        if (i == j) {
            uint64_t random_u64 = local_state * 0x2545F4914F6CDD1Dull;
            random_value = (random_u64 >> 32) / 4294967295.0;
        }
    }

    if (i == n - 1) {
        *state = local_state;
    }

    output[i] = random_value < probability ? 1.0 : 0.0;
}