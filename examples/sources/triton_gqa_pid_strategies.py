"""Small GQA forward kernels used to compare two source-defined PID strategies.

Q[H_Q, M, D], K/V[H_KV, N, D], O[H_Q, M, D] are contiguous.
This is an illustrative noncausal fp16 forward with D=16, not a tuned kernel.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gqa_output_tile(Q, K, V, O, q_block, q_head, kv_head,
                     M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, SCALE: tl.constexpr):
    q_rows = q_block * BM + tl.arange(0, BM)
    d = tl.arange(0, D)
    q = tl.load(Q + q_head * M * D + q_rows[:, None] * D + d[None, :],
                mask=q_rows[:, None] < M, other=0)
    row_max = tl.full((BM,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    for key_block in range(tl.cdiv(N, BN)):
        key_rows = key_block * BN + tl.arange(0, BN)
        k = tl.load(K + kv_head * N * D + key_rows[:, None] * D + d[None, :],
                    mask=key_rows[:, None] < N, other=0)
        v = tl.load(V + kv_head * N * D + key_rows[:, None] * D + d[None, :],
                    mask=key_rows[:, None] < N, other=0)
        score = tl.dot(q, tl.trans(k)).to(tl.float32) * SCALE
        score = tl.where(key_rows[None, :] < N, score, -float("inf"))
        next_max = tl.maximum(row_max, tl.max(score, 1))
        scale_old = tl.exp(row_max - next_max)
        prob = tl.exp(score - next_max[:, None])
        acc = acc * scale_old[:, None]
        acc = tl.dot(prob.to(tl.float16), v, acc)
        row_sum = row_sum * scale_old + tl.sum(prob, 1)
        row_max = next_max
    result = acc / row_sum[:, None]
    tl.store(O + q_head * M * D + q_rows[:, None] * D + d[None, :],
             result, mask=q_rows[:, None] < M)


@triton.jit
def gqa_q_owned(Q, K, V, O, M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
                H_Q: tl.constexpr, H_KV: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, SCALE: tl.constexpr):
    q_block = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // (H_Q // H_KV)
    _gqa_output_tile(Q, K, V, O, q_block, q_head, kv_head, M, N, D, BM, BN, SCALE)


@triton.jit
def gqa_kv_owned(Q, K, V, O, M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
                 H_Q: tl.constexpr, H_KV: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, SCALE: tl.constexpr):
    kv_head = tl.program_id(0)
    group_size = H_Q // H_KV
    for q_head in range(kv_head * group_size, (kv_head + 1) * group_size):
        for q_block in range(tl.cdiv(M, BM)):
            _gqa_output_tile(Q, K, V, O, q_block, q_head, kv_head, M, N, D, BM, BN, SCALE)


def launch_gqa(q, k, v, out, strategy="q"):
    h_q, m, d = q.shape
    h_kv, n, kd = k.shape
    assert h_q % h_kv == 0 and h_q > h_kv
    assert d == kd == v.shape[2] == out.shape[2] == 16
    assert v.shape[:2] == (h_kv, n) and out.shape[:2] == (h_q, m)
    assert q.dtype == k.dtype == v.dtype == torch.float16 and out.dtype == torch.float32
    assert all(t.is_contiguous() and t.is_cuda for t in (q, k, v, out))
    bm, bn = 16, 16
    common = (q, k, v, out, m, n, d, h_q, h_kv, bm, bn, d ** -0.5)
    if strategy == "q":
        grid = (triton.cdiv(m, bm), h_q)
        gqa_q_owned[grid](*common, num_warps=4)
    elif strategy == "kv":
        grid = (h_kv,)
        gqa_kv_owned[grid](*common, num_warps=4)
    else:
        raise ValueError("strategy must be 'q' or 'kv'")
