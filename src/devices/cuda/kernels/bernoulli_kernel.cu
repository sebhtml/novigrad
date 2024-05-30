#define uint64_t unsigned long long

/// https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
/// With ideas from Alexandre Desilets-Benoit @AlexDesBen on GitHub.
extern "C" __global__ void bernoulli_kernel(float *input, float *output, int n, uint64_t *state)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    uint64_t local_state = *state;

    for (int j = 0; j < i + 1; j++) {
        local_state ^= local_state >> 12;
        local_state ^= local_state << 25;
        local_state ^= local_state >> 27;
    }

    uint64_t random_u64 = local_state * 0x2545F4914F6CDD1Dull;
    float random_value = (random_u64 >> 32) / 4294967295.0;
    float probability = input[i];
    output[i] = random_value < probability ? 1.0 : 0.0;

    if (i == n - 1) {
        *state = local_state;
    }
}