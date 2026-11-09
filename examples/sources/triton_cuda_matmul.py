"""Demonstration source: each Triton program owns one output tile."""

import triton
import triton.language as tl


@triton.jit
def matmul_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    tiles_n = tl.cdiv(N, BN)
    tile_m = pid // tiles_n
    tile_n = pid % tiles_n
    rows = tile_m * BM + tl.arange(0, BM)
    cols = tile_n * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)

    for k0 in range(tl.cdiv(K, BK)):
        a_ptrs = A + rows[:, None] * K + (k0 * BK + ks[None, :])
        b_ptrs = B + (k0 * BK + ks[:, None]) * N + cols[None, :]
        a = tl.load(a_ptrs, mask=(rows[:, None] < M) &
                    (k0 * BK + ks[None, :] < K), other=0)
        b = tl.load(b_ptrs, mask=(k0 * BK + ks[:, None] < K) &
                    (cols[None, :] < N), other=0)
        acc += tl.dot(a, b)

    c_ptrs = C + rows[:, None] * N + cols[None, :]
    tl.store(c_ptrs, acc, mask=(rows[:, None] < M) & (cols[None, :] < N))


def launch(A, B, C, M, N, K):
    BM, BN, BK = 16, 16, 16
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    matmul_kernel[grid](A, B, C, M, N, K, BM, BN, BK)
