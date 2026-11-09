"""Demonstration source for Triton-Ascend's Triton-compatible language surface."""

import triton
import triton.language as tl


@triton.jit
def add_kernel(X, Y, Z, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask, other=0)
    y = tl.load(Y + offsets, mask=mask, other=0)
    z = x + y
    tl.store(Z + offsets, z, mask=mask)


def launch(X, Y, Z, N):
    BLOCK = 8
    grid = (triton.cdiv(N, BLOCK),)
    add_kernel[grid](X, Y, Z, N, BLOCK)
