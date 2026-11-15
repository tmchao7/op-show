"""Single-head, noncausal FlashAttention-2 style forward teaching kernel.

Q[M, D], K[N, D], V[N, D] are contiguous fp16; O[M, D] is float32.
Only D=16 is supported here. This is a logical example, not a tuned FA2 kernel.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention2_forward_demo(Q, K, V, O,
                                  M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
                                  SCALE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    q_idx = pid_m * BM + tl.arange(0, BM)
    d_idx = tl.arange(0, D)
    q = tl.load(Q + q_idx[:, None] * D + d_idx[None, :],
                mask=(q_idx[:, None] < M) & (d_idx[None, :] < D), other=0)
    m_i = tl.full((BM,), -float("inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)

    for n0 in range(tl.cdiv(N, BN)):
        k_idx = n0 * BN + tl.arange(0, BN)
        k = tl.load(K + k_idx[:, None] * D + d_idx[None, :],
                    mask=(k_idx[:, None] < N) & (d_idx[None, :] < D), other=0)
        v = tl.load(V + k_idx[:, None] * D + d_idx[None, :],
                    mask=(k_idx[:, None] < N) & (d_idx[None, :] < D), other=0)
        score = tl.dot(q, tl.trans(k)).to(tl.float32) * SCALE
        score = tl.where(k_idx[None, :] < N, score, -float("inf"))
        m_next = tl.maximum(m_i, tl.max(score, 1))
        alpha = tl.exp(m_i - m_next)
        p = tl.exp(score - m_next[:, None])
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.float16), v, acc)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_next

    out = acc / l_i[:, None]
    tl.store(O + q_idx[:, None] * D + d_idx[None, :], out,
             mask=(q_idx[:, None] < M) & (d_idx[None, :] < D))


def launch_flash_attention2_forward_demo(q, k, v, out):
    m, d = q.shape
    n, kd = k.shape
    assert m > 0 and n > 0
    assert d == kd == v.shape[1] == out.shape[1] == 16
    assert v.shape[0] == n and out.shape[0] == m
    assert q.dtype == k.dtype == v.dtype == torch.float16 and out.dtype == torch.float32
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and out.is_contiguous()
    assert q.is_cuda and q.device == k.device == v.device == out.device
    bm, bn = 16, 16
    grid = (triton.cdiv(m, bm),)
    flash_attention2_forward_demo[grid](q, k, v, out, m, n, d,
                                        d ** -0.5, bm, bn, num_warps=4)
