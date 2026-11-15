// Teaching GEMV: A[M,K] times x[K] -> y[M], contiguous row-major float.
// Four complete warps per block; each warp owns one output row.
#include <cuda_runtime.h>

constexpr int ROWS_PER_BLOCK = 4;
constexpr int WARP_K = 32;

__global__ void warp_gemv(const float* A, const float* x, float* y, int M, int K) {
  const int warp = threadIdx.x / WARP_K;
  const int lane = threadIdx.x % WARP_K;
  const int row = blockIdx.x * ROWS_PER_BLOCK + warp;
  float sum = 0.0f;
  for (int k0 = 0; k0 < K; k0 += WARP_K) {
    const int k = k0 + lane;
    if (row < M && k < K) sum += A[row * K + k] * x[k];
  }
  for (int delta = WARP_K / 2; delta > 0; delta /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, delta);
  }
  if (lane == 0 && row < M) y[row] = sum;
}

void launch_warp_gemv(const float* A, const float* x, float* y, int M, int K) {
  dim3 block(ROWS_PER_BLOCK * WARP_K);
  dim3 grid((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
  warp_gemv<<<grid, block>>>(A, x, y, M, K);
}
