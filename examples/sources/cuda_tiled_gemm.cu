// Demonstration source: one CUDA block owns one output tile.
#include <cuda_runtime.h>

constexpr int BM = 4;
constexpr int BN = 4;
constexpr int BK = 4;

__global__ void tiled_gemm(const float* A, const float* B, float* C,
                           int M, int N, int K) {
  const int tile_m = blockIdx.y;
  const int tile_n = blockIdx.x;
  const int local_m = threadIdx.y;
  const int local_n = threadIdx.x;
  const int row = tile_m * BM + local_m;
  const int col = tile_n * BN + local_n;
  __shared__ float a_tile[BM][BK];
  __shared__ float b_tile[BK][BN];
  float acc = 0.0f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    a_tile[local_m][local_n] =
        (row < M && k0 + local_n < K) ? A[row * K + k0 + local_n] : 0.0f;
    b_tile[local_m][local_n] =
        (k0 + local_m < K && col < N) ? B[(k0 + local_m) * N + col] : 0.0f;
    __syncthreads();
    for (int k = 0; k < BK; ++k) {
      acc += a_tile[local_m][k] * b_tile[k][local_n];
    }
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

void launch(const float* A, const float* B, float* C, int M, int N, int K) {
  dim3 block(BN, BM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  tiled_gemm<<<grid, block>>>(A, B, C, M, N, K);
}
