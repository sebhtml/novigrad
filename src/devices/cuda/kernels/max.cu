extern "C" __global__ void max_kernel(float* data, int size, float* result) {
  // Thread index within the block
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Check for boundary conditions (avoid out-of-bounds access)
  if (tid < size) {
    // Initialize local maximum for each thread
    float local_max = data[tid];

    // Shared memory for efficient reduction within a block (optional)
    __shared__ float block_max[32 * 32];

    // Loop for parallel reduction within a block
    for (int stride = blockDim.x; stride > 1; stride /= 2) {
      if (tid < stride) {
        local_max = fmaxf(local_max, data[tid + stride]);
      }
      __syncthreads();
    }

    // Store block maximum in shared memory
    block_max[tid] = local_max;
    __syncthreads();

    // Reduce block maximums across threads (only thread 0)
    if (tid == 0) {
      local_max = block_max[0];
      for (int i = 1; i < blockDim.x; i++) {
        local_max = fmaxf(local_max, block_max[i]);
      }
      result[0] = local_max;
    }
  }
}
