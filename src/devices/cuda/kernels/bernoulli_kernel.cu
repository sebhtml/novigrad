#define uint64_t unsigned long long

/// @brief Compute <n> Bernoulli trials
/// @param input    vector with <n> Probabilities
/// @param output   vector with <n> Bernoulli trial outputs
/// @param n        number of Bernoulli trials
/// @param state    RNG state for the CUDA stream used for this kernel launch.
/// @return   void
/// @note It uses Xorshift from George Marsaglia.
/// @note With ideas from Alexandre Desilets-Benoit @AlexDesBen on GitHub.
/// @see https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
extern "C" __global__ void bernoulli_kernel(float *input, float *output, int n, uint64_t *state)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    // Each thread has its own private RNG state.
    uint64_t local_state = *state;
    for (int j = 0; j < i + 1; j++) {
        local_state ^= local_state >> 12;
        local_state ^= local_state << 25;
        local_state ^= local_state >> 27;
    }

    // Generate a random value between 0.0 and 1.0 from the state.
    uint64_t random_u64 = local_state * 0x2545F4914F6CDD1Dull;
    float random_value = (random_u64 >> 32) / 4294967295.0;
    // Do a Bernoulli trial
    float probability = input[i];
    output[i] = random_value < probability ? 1.0 : 0.0;

    // Write back the new RNG state to the device memory.
    // It will be used for the next kernel launch.
    if (i == n - 1) {
        *state = local_state;
    }
}