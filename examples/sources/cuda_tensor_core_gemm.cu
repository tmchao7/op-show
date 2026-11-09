// Educational WMMA GEMM: one 32-lane CUDA block owns one 16x16 C tile.
// Inputs A[M,K] and B[K,N] are contiguous row-major half; C[M,N] is float.
// Compile for a Tensor Core target, for example: nvcc -arch=sm_70 -c cuda_tensor_core_gemm.cu
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;
constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 16;

__global__ void tensor_core_gemm(const half* A, const half* B, float* C,
                                 int M, int N, int K) {
  const int tile_m = blockIdx.y;
  const int tile_n = blockIdx.x;
  const int lane = threadIdx.x;  // Launch contract: exactly one 32-lane warp.
  const int row0 = tile_m * TILE_M;
  const int col0 = tile_n * TILE_N;

  // WMMA matrix pointers need 32-byte alignment and ldm=16 for half/float.
  __shared__ __align__(32) half a_shared[TILE_M * TILE_K];
  __shared__ __align__(32) half b_shared[TILE_K * TILE_N];
  __shared__ __align__(32) float c_shared[TILE_M * TILE_N];
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  for (int k0 = 0; k0 < K; k0 += TILE_K) {
    // Stage a full 16x16 K slice; out-of-bounds elements become zero.
    for (int i = lane; i < TILE_M * TILE_K; i += 32) {
      const int r = i / TILE_K;
      const int c = i % TILE_K;
      const int a_row = row0 + r;
      const int a_col = k0 + c;
      const int b_row = k0 + r;
      const int b_col = col0 + c;
      a_shared[i] = (a_row < M && a_col < K)
                        ? A[a_row * K + a_col] : __float2half(0.0f);
      b_shared[i] = (b_row < K && b_col < N)
                        ? B[b_row * N + b_col] : __float2half(0.0f);
    }
    __syncthreads();
    wmma::load_matrix_sync(a_frag, a_shared, TILE_K);
    wmma::load_matrix_sync(b_frag, b_shared, TILE_N);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
    __syncthreads();  // No lane overwrites shared A/B before this warp finishes.
  }

  wmma::store_matrix_sync(c_shared, acc, TILE_N, wmma::mem_row_major);
  __syncthreads();
  for (int i = lane; i < TILE_M * TILE_N; i += 32) {
    const int row = row0 + i / TILE_N;
    const int col = col0 + i % TILE_N;
    if (row < M && col < N) C[row * N + col] = c_shared[i];
  }
}

void launch_tensor_core_gemm(const half* A, const half* B, float* C,
                             int M, int N, int K) {
  dim3 block(32);
  dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
  tensor_core_gemm<<<grid, block>>>(A, B, C, M, N, K);
}
