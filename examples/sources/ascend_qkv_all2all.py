# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""QKV All2All with contiguous head-major peer tiles on every path.

For one to four local heads, all ``(target rank, S block)`` tasks are flattened
over half of the available AIV cores.  One task moves Q, K, and V together,
writes the local rank directly to the ordinary outputs, and uses exactly one
remote fence/notify for a remote rank.  Remote staging stores each B112xD
source/fused-head tile contiguously before the matching consumer reads it.
Five to seven local heads retain V0's B112/W4 head-grouped task schedule while
using the same contiguous source/fused-head staging layout.  The public
launcher, output layout, workspace API,
and 189-shape HCCL benchmark contract are unchanged.
"""

import math
import os

import shmem as ash
import torch
import torch_npu
import torch.distributed as dist
import triton
import triton.language as tl
import triton_dist.language as dl
from triton.backends.ascend.driver import NPUUtils
from triton_dist.language.extra import libshmem_device


g_ash_size = 1024 * 1024 * 1024 # ash 是 Ascend SHMEM（共享内存通信库）。这个值是每个 rank 分配的 SHMEM 本地内存大小 1G
G_IP_PORT = os.environ.get("G_IP_PORT", "tcp://127.0.0.1:8666") # SHMEM 初始化时用于建链的 IP 和端口
COMM_BLOCK_S = 112 # 序列方向（S 维）的 tile 大小，即一个通信 tile 覆盖 112 个 token
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
COMM_BLOCK_D = 128 # head_dim 方向的 tile 大小，也就是 head_dim 本身，固定 128。
FLAT_HEAD_LIMIT = 4 # block-flat 路径和 head-grouped 路径的分界线。

# dl.notify requires an int32/uint32 signal pointer on Ascend.
# aclshmem_wait walks barriers 64 bytes apart: 64 / sizeof(int32) = 16 elements.
SIGNAL_SLOT_STRIDE = 16 # 每个信号槽在 signal_mem 里占的 int32 元素个数。

# 划分aiv的生产者-消费者 vector角色
def _aiv_role_counts(total_aiv_cores):
    """Split every available AIV between concurrent producers and consumers."""
    if total_aiv_cores < 2:
        raise RuntimeError(
            "QKV All2All requires at least two AIV cores; "
            f"device reports {total_aiv_cores}"
        )
    producer_cores = (total_aiv_cores + 1) // 2 # (n+1)//2 让 producer 在奇数时多一个
    consumer_cores = total_aiv_cores - producer_cores
    return producer_cores, consumer_cores

# dsp_size：总 rank 数 rank_size
def qkv_fuse_a2a_golden(q, k, v, dsp_size, group=None):
    """Return the standard torch QKV All2All reference result."""
    q_transposed = torch.cat(torch.chunk(q, dsp_size, dim=1), dim=0)
    k_transposed = torch.cat(torch.chunk(k, dsp_size, dim=1), dim=0)
    v_transposed = torch.cat(torch.chunk(v, dsp_size, dim=1), dim=0)
    heads_per_rank = q_transposed.shape[1]
    qkv = torch.cat([q_transposed, k_transposed, v_transposed], dim=1).contiguous()
    recv = torch.empty_like(qkv)
    dist.all_to_all_single(recv, qkv, group=group)
    return recv.split([heads_per_rank] * 3, dim=1)

#什么时候走 block-flat？什么时候走 head-grouped？ 每组几个头？
# 读完你应该知道： ≤4：block-flat，一组就是全部头； 5~16：每组 2 头； >16：每组 4 头。
def _head_group_size(heads_per_rank):
    """Bound head-loop expansion while keeping enough producer tasks."""
    if heads_per_rank <= 4:
        return 0  # Block-flat path: one notify covers Q/K/V for the block.
    if heads_per_rank <= 16:
        return 2
    return 4

# 需要多少个 信号组  这个依赖于 _head_group_size
def _signal_groups(heads_per_rank):
    group_size = _head_group_size(heads_per_rank)
    return 1 if group_size == 0 else math.ceil(heads_per_rank / group_size)

# 通信缓冲区大小的计算器
def workspace_sizes(
    sequence_length,
    n_head,
    rank_size,
    head_dim=COMM_BLOCK_D,
    comm_block_s=None,
):
    """Return required ``(peer_elements, signal_elements, buffer_num)``."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if rank_size <= 0:
        raise ValueError("rank_size must be positive")
    if n_head <= 0 or n_head % rank_size != 0:
        raise ValueError("n_head must be positive and divisible by rank_size")
    if head_dim != COMM_BLOCK_D:
        raise ValueError(f"head_dim must be {COMM_BLOCK_D}")

    if comm_block_s is None:
        comm_block_s = COMM_BLOCK_S
    if comm_block_s != COMM_BLOCK_S:
        raise ValueError(f"comm_block_s must be {COMM_BLOCK_S}")

    buffer_num = math.ceil(sequence_length / comm_block_s)
    heads_per_rank = n_head // rank_size
    fused_heads = 3 * heads_per_rank
    # 单个rank peer缓冲区的 元素个数 依赖于：token数 rank数量 融合头数 hidden
    #
    peer_elements = (
        buffer_num * comm_block_s * rank_size * fused_heads * head_dim
    )
    # SHMEM signal addressing uses element offsets on an int32 tensor.
    # 信号量 元素个数（存的是int32 一个信号槽 16个元素）
    signal_elements = (
        buffer_num * rank_size * _signal_groups(heads_per_rank) * SIGNAL_SLOT_STRIDE
    )
    return peer_elements, signal_elements, buffer_num


@triton.jit
def kernel_qkv_fuse_a2a_block_flat(
    q_ptr,
    k_ptr,
    v_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    peer_mem_ptr,
    signal_mem_ptr,
    rank,
    rank_size,
    buffer_num,
    sequence_length,
    stride_a_s,
    stride_a_h,
    stride_a_d,
    stride_out_s,
    stride_out_h,
    stride_out_d,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEADS_PER_RANK: tl.constexpr,
    FUSED_HEADS: tl.constexpr,
    PRODUCER_CORES: tl.constexpr,
    CONSUMER_CORES: tl.constexpr,
    SIGNAL_STRIDE: tl.constexpr,
):
    """Flatten every small-head ``(target rank, S block)`` QKV task."""
    pid = tl.program_id(axis=0) # [vecs,1,1]
    role = pid % 2 # 生产者 消费者 的id划分
    logical_core_id = pid // 2

    num_blocks = tl.cdiv(sequence_length, BLOCK_S) # block数量 沿着token维度
    total_tasks = rank_size * num_blocks # 每个rank要处理的任务数量
    peer_head_size = BLOCK_S * BLOCK_D # 一个tile里面的元素数量
    block_slot_size = rank_size * FUSED_HEADS * peer_head_size # 一个完整 Sequence Block 为所有来源 Rank、所有 QKV Head 位置预留的总空间
    offs_d = tl.arange(0, BLOCK_D)

    # Even AIVs produce.  Each task completes Q/K/V before its sole notify.
    if role == 0: # 生产者- 负责往目标rank缓冲区写数据 并通知
        for flat_idx in range(logical_core_id, total_tasks, PRODUCER_CORES):
            target_rank = flat_idx // num_blocks
            block_id = flat_idx % num_blocks
            offs_s = block_id * BLOCK_S + tl.arange(0, BLOCK_S)
            row_mask = offs_s < sequence_length
            io_mask = row_mask[:, None]
            input_base = (
                offs_s[:, None] * stride_a_s
                + offs_d[None, :] * stride_a_d
            )
            input_head_base = target_rank * HEADS_PER_RANK
            # 本地rank 直接往out写数据 (无需 notify-wait)
            if target_rank == rank:
                output_row = rank * sequence_length + offs_s
                output_base = (
                    output_row[:, None] * stride_out_s
                    + offs_d[None, :] * stride_out_d
                )
                for head in range(0, HEADS_PER_RANK):
                    input_offs = input_base + (
                        input_head_base + head
                    ) * stride_a_h
                    output_offs = output_base + head * stride_out_h
                    # qkv_raw_data -> qkv_out （【S，H，D】-> 【S*P，H/P，D】）
                    q_data = tl.load(
                        q_ptr + input_offs, mask=io_mask, other=0.0
                    )
                    k_data = tl.load(
                        k_ptr + input_offs, mask=io_mask, other=0.0
                    )
                    v_data = tl.load(
                        v_ptr + input_offs, mask=io_mask, other=0.0
                    )
                    tl.store(q_out_ptr + output_offs, q_data, mask=io_mask)
                    tl.store(k_out_ptr + output_offs, k_data, mask=io_mask)
                    tl.store(v_out_ptr + output_offs, v_data, mask=io_mask)
            # 往 非己 rank中写 数据 并 fence + notify
            else:
                remote_peer_ptr = dl.symm_at(peer_mem_ptr, target_rank)
                peer_block_base = block_id * block_slot_size
                peer_source_base = rank * FUSED_HEADS * peer_head_size
                peer_row_base = (
                    peer_block_base
                    + tl.arange(0, BLOCK_S)[:, None] * BLOCK_D
                    + offs_d[None, :]
                )
                # write loop
                for head in range(0, HEADS_PER_RANK):
                    input_offs = input_base + (
                        input_head_base + head
                    ) * stride_a_h
                    q_data = tl.load(
                        q_ptr + input_offs, mask=io_mask, other=0.0
                    )
                    k_data = tl.load(
                        k_ptr + input_offs, mask=io_mask, other=0.0
                    )
                    v_data = tl.load(
                        v_ptr + input_offs, mask=io_mask, other=0.0
                    )
                    peer_head = peer_source_base + head * peer_head_size
                    tl.store(
                        remote_peer_ptr + peer_row_base + peer_head,
                        q_data,
                        mask=io_mask,
                    )
                    tl.store(
                        remote_peer_ptr
                        + peer_row_base
                        + peer_head
                        + HEADS_PER_RANK * peer_head_size,
                        k_data,
                        mask=io_mask,
                    )
                    tl.store(
                        remote_peer_ptr
                        + peer_row_base
                        + peer_head
                        + 2 * HEADS_PER_RANK * peer_head_size,
                        v_data,
                        mask=io_mask,
                    )
                # write loop  done
                libshmem_device.fence() # fence确保 store延迟
                signal_slot = block_id * rank_size + rank
                dl.notify(
                    signal_mem_ptr + signal_slot * SIGNAL_STRIDE,
                    target_rank,
                    1,
                )

    # Odd AIVs consume remote sources only.  Q/K/V share one acquire wait.
    if role == 1:
        for flat_idx in range(logical_core_id, total_tasks, CONSUMER_CORES):
            source_rank = flat_idx // num_blocks
            block_id = flat_idx % num_blocks
            # 只对非本地rank的signal信号
            if source_rank != rank:
                signal_slot = block_id * rank_size + source_rank
                token = dl.wait(
                    signal_mem_ptr + signal_slot * SIGNAL_STRIDE,
                    1,
                    "gpu",
                    "acquire",
                    waitValue=1,
                )
                local_peer_ptr = dl.consume_token(peer_mem_ptr, token) # consume阻塞
                offs_s = block_id * BLOCK_S + tl.arange(0, BLOCK_S)
                row_mask = offs_s < sequence_length
                io_mask = row_mask[:, None]
                output_row = source_rank * sequence_length + offs_s
                output_base = (
                    output_row[:, None] * stride_out_s
                    + offs_d[None, :] * stride_out_d
                )
                peer_block_base = block_id * block_slot_size
                peer_source_base = source_rank * FUSED_HEADS * peer_head_size
                peer_row_base = (
                    peer_block_base
                    + tl.arange(0, BLOCK_S)[:, None] * BLOCK_D
                    + offs_d[None, :]
                )
                for head in range(0, HEADS_PER_RANK):
                    peer_head = peer_source_base + head * peer_head_size
                    # 消费上面的consume_token 放置 load 垃圾值
                    q_data = tl.load(
                        local_peer_ptr + peer_row_base + peer_head,
                        mask=io_mask,
                        other=0.0,
                    )
                    k_data = tl.load(
                        local_peer_ptr
                        + peer_row_base
                        + peer_head
                        + HEADS_PER_RANK * peer_head_size,
                        mask=io_mask,
                        other=0.0,
                    )
                    v_data = tl.load(
                        local_peer_ptr
                        + peer_row_base
                        + peer_head
                        + 2 * HEADS_PER_RANK * peer_head_size,
                        mask=io_mask,
                        other=0.0,
                    )
                    output_offs = output_base + head * stride_out_h
                    tl.store(q_out_ptr + output_offs, q_data, mask=io_mask)
                    tl.store(k_out_ptr + output_offs, k_data, mask=io_mask)
                    tl.store(v_out_ptr + output_offs, v_data, mask=io_mask)


@triton.jit
def kernel_qkv_fuse_a2a_rank_batched(
    q_ptr,
    k_ptr,
    v_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    peer_mem_ptr,
    signal_mem_ptr,
    rank,
    rank_size,
    buffer_num,
    sequence_length,
    stride_a_s,
    stride_a_h,
    stride_a_d,
    stride_out_s,
    stride_out_h,
    stride_out_d,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    S_WAVE: tl.constexpr,
    HEADS_PER_RANK: tl.constexpr,
    FUSED_HEADS: tl.constexpr,
    TOTAL_CORES: tl.constexpr,
    PRODUCER_CORES: tl.constexpr,
    CONSUMER_CORES: tl.constexpr,
    SIGNAL_STRIDE: tl.constexpr,
):
    """Low-overhead path for at most four heads per destination rank."""
    pid = tl.program_id(axis=0)
    producers_before = (
        pid * PRODUCER_CORES + TOTAL_CORES - 1
    ) // TOTAL_CORES
    producers_through = (
        (pid + 1) * PRODUCER_CORES + TOTAL_CORES - 1
    ) // TOTAL_CORES

    num_blocks = tl.cdiv(sequence_length, BLOCK_S)
    stride_peer_s = rank_size * FUSED_HEADS * BLOCK_D
    stride_peer_h = BLOCK_D
    block_slot_size = BLOCK_S * stride_peer_s

    for wave_start in range(0, num_blocks, S_WAVE):
        total_wave_tasks = rank_size * S_WAVE

        if producers_through > producers_before:
            logical_core_id = producers_before
            for flat_idx in range(
                logical_core_id, total_wave_tasks, PRODUCER_CORES
            ):
                lane = flat_idx % S_WAVE
                target_rank = flat_idx // S_WAVE
                block_id = wave_start + lane

                if block_id < num_blocks:
                    offs_s = block_id * BLOCK_S + tl.arange(0, BLOCK_S)
                    offs_d = tl.arange(0, BLOCK_D)
                    row_mask = offs_s < sequence_length
                    io_mask = row_mask[:, None]
                    input_base = (
                        offs_s[:, None] * stride_a_s
                        + offs_d[None, :] * stride_a_d
                    )
                    output_row = rank * sequence_length + offs_s
                    output_base = (
                        output_row[:, None] * stride_out_s
                        + offs_d[None, :] * stride_out_d
                    )
                    input_head_base = target_rank * HEADS_PER_RANK
                    peer_block_base = (
                        (block_id % buffer_num) * block_slot_size
                    )

                    for fused_head in range(0, FUSED_HEADS):
                        if fused_head < HEADS_PER_RANK:
                            output_head = fused_head
                            data = tl.load(
                                q_ptr
                                + input_base
                                + (input_head_base + output_head) * stride_a_h,
                                mask=io_mask,
                                other=0.0,
                            )
                        elif fused_head < 2 * HEADS_PER_RANK:
                            output_head = fused_head - HEADS_PER_RANK
                            data = tl.load(
                                k_ptr
                                + input_base
                                + (input_head_base + output_head) * stride_a_h,
                                mask=io_mask,
                                other=0.0,
                            )
                        else:
                            output_head = fused_head - 2 * HEADS_PER_RANK
                            data = tl.load(
                                v_ptr
                                + input_base
                                + (input_head_base + output_head) * stride_a_h,
                                mask=io_mask,
                                other=0.0,
                            )

                        if target_rank == rank:
                            if fused_head < HEADS_PER_RANK:
                                tl.store(
                                    q_out_ptr
                                    + output_base
                                    + output_head * stride_out_h,
                                    data,
                                    mask=io_mask,
                                )
                            elif fused_head < 2 * HEADS_PER_RANK:
                                tl.store(
                                    k_out_ptr
                                    + output_base
                                    + output_head * stride_out_h,
                                    data,
                                    mask=io_mask,
                                )
                            else:
                                tl.store(
                                    v_out_ptr
                                    + output_base
                                    + output_head * stride_out_h,
                                    data,
                                    mask=io_mask,
                                )
                        else:
                            remote_peer_ptr = dl.symm_at(
                                peer_mem_ptr, target_rank
                            )
                            peer_offs = (
                                peer_block_base
                                + tl.arange(0, BLOCK_S)[:, None]
                                * stride_peer_s
                                + (rank * FUSED_HEADS + fused_head)
                                * stride_peer_h
                                + offs_d[None, :]
                            )
                            tl.store(
                                remote_peer_ptr + peer_offs,
                                data,
                                mask=io_mask,
                            )

            libshmem_device.fence()
            for flat_idx in range(
                logical_core_id, total_wave_tasks, PRODUCER_CORES
            ):
                lane = flat_idx % S_WAVE
                target_rank = flat_idx // S_WAVE
                block_id = wave_start + lane
                if block_id < num_blocks and target_rank != rank:
                    signal_slot = (
                        (block_id % buffer_num) * rank_size + rank
                    )
                    dl.notify(
                        signal_mem_ptr + signal_slot * SIGNAL_STRIDE,
                        target_rank,
                        1,
                    )

        if producers_through == producers_before:
            logical_core_id = pid - producers_before
            for flat_idx in range(
                logical_core_id, total_wave_tasks, CONSUMER_CORES
            ):
                lane = flat_idx % S_WAVE
                source_rank = flat_idx // S_WAVE
                block_id = wave_start + lane

                if block_id < num_blocks and source_rank != rank:
                    offs_s = block_id * BLOCK_S + tl.arange(0, BLOCK_S)
                    offs_d = tl.arange(0, BLOCK_D)
                    row_mask = offs_s < sequence_length
                    io_mask = row_mask[:, None]
                    output_row = source_rank * sequence_length + offs_s
                    output_base = (
                        output_row[:, None] * stride_out_s
                        + offs_d[None, :] * stride_out_d
                    )
                    peer_block_base = (
                        (block_id % buffer_num) * block_slot_size
                    )
                    signal_slot = (
                        (block_id % buffer_num) * rank_size + source_rank
                    )
                    token = dl.wait(
                        signal_mem_ptr + signal_slot * SIGNAL_STRIDE,
                        1,
                        "gpu",
                        "acquire",
                        waitValue=1,
                    )
                    local_peer_ptr = dl.consume_token(peer_mem_ptr, token)

                    for fused_head in range(0, FUSED_HEADS):
                        peer_offs = (
                            peer_block_base
                            + tl.arange(0, BLOCK_S)[:, None] * stride_peer_s
                            + (source_rank * FUSED_HEADS + fused_head)
                            * stride_peer_h
                            + offs_d[None, :]
                        )
                        data = tl.load(
                            local_peer_ptr + peer_offs,
                            mask=io_mask,
                            other=0.0,
                        )

                        if fused_head < HEADS_PER_RANK:
                            tl.store(
                                q_out_ptr
                                + output_base
                                + fused_head * stride_out_h,
                                data,
                                mask=io_mask,
                            )
                        elif fused_head < 2 * HEADS_PER_RANK:
                            output_head = fused_head - HEADS_PER_RANK
                            tl.store(
                                k_out_ptr
                                + output_base
                                + output_head * stride_out_h,
                                data,
                                mask=io_mask,
                            )
                        else:
                            output_head = fused_head - 2 * HEADS_PER_RANK
                            tl.store(
                                v_out_ptr
                                + output_base
                                + output_head * stride_out_h,
                                data,
                                mask=io_mask,
                            )


@triton.jit
def kernel_qkv_fuse_a2a_head_grouped(
    q_ptr,
    k_ptr,
    v_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    peer_mem_ptr,
    signal_mem_ptr,
    rank,
    rank_size,
    buffer_num,
    sequence_length,
    stride_a_s,
    stride_a_h,
    stride_a_d,
    stride_out_s,
    stride_out_h,
    stride_out_d,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    S_WAVE: tl.constexpr,
    HEADS_PER_RANK: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    HEAD_GROUPS: tl.constexpr,
    TOTAL_CORES: tl.constexpr,
    PRODUCER_CORES: tl.constexpr,
    CONSUMER_CORES: tl.constexpr,
    SIGNAL_STRIDE: tl.constexpr,
):
    """Parallelize bounded non-interleaved head groups across AIV cores."""
    pid = tl.program_id(axis=0)
    # producers_before 和 producers_through 用于在 vec_nums不是偶数时候的划分
    producers_before = (
        pid *       PRODUCER_CORES + TOTAL_CORES - 1
    ) // TOTAL_CORES
    producers_through = (
        (pid + 1) * PRODUCER_CORES + TOTAL_CORES - 1
    ) // TOTAL_CORES

    fused_heads = 3 * HEADS_PER_RANK
    num_blocks = tl.cdiv(sequence_length, BLOCK_S)
    peer_head_size = BLOCK_S * BLOCK_D
    block_slot_size = rank_size * fused_heads * peer_head_size

    for wave_start in range(0, num_blocks, S_WAVE):
        # 这里把 heads_per_rank 分成 HEAD_GROUPS 组 导致每次任务会 搬运一组 所以 总组数需要*HEAD_GROUPS
        total_wave_tasks = rank_size * HEAD_GROUPS * S_WAVE
        # （这里差值为 1）pid 是 producer
        if producers_through > producers_before:
            logical_core_id = producers_before
            # PRODUCER_CORES 并行处理 total_wave_tasks
            for flat_idx in range(
                logical_core_id, total_wave_tasks, PRODUCER_CORES
            ):
                lane = flat_idx % S_WAVE
                task_idx = flat_idx // S_WAVE
                target_rank = task_idx % rank_size
                head_group = task_idx // rank_size
                block_id = wave_start + lane

                if block_id < num_blocks:# 防止 sequence序列方向 越界

                    offs_s = block_id * BLOCK_S + tl.arange(0, BLOCK_S)
                    offs_d = tl.arange(0, BLOCK_D)
                    row_mask = offs_s < sequence_length
                    io_mask = row_mask[:, None]
                    input_base = (
                        offs_s[:, None] * stride_a_s
                        + offs_d[None, :] * stride_a_d
                    )
                    output_row = rank * sequence_length + offs_s
                    output_base = (
                        output_row[:, None] * stride_out_s
                        + offs_d[None, :] * stride_out_d
                    )
                    input_head_base = target_rank * HEADS_PER_RANK
                    peer_block_base = (
                        (block_id % buffer_num) * block_slot_size
                    )
                    # 遍历当前要搬运的 heads组
                    for local_head in range(0, HEADS_PER_GROUP):
                        head = head_group * HEADS_PER_GROUP + local_head
                        if head < HEADS_PER_RANK:
                            q_data = tl.load(
                                q_ptr
                                + input_base
                                + (input_head_base + head) * stride_a_h,
                                mask=io_mask,
                                other=0.0,
                            )
                            k_data = tl.load(
                                k_ptr
                                + input_base
                                + (input_head_base + head) * stride_a_h,
                                mask=io_mask,
                                other=0.0,
                            )
                            v_data = tl.load(
                                v_ptr
                                + input_base
                                + (input_head_base + head) * stride_a_h,
                                mask=io_mask,
                                other=0.0,
                            )
                            # 本 rank 寄存器直写本rank HBM的qkv_out地址
                            if target_rank == rank:
                                output_offs = (
                                    output_base + head * stride_out_h
                                )
                                tl.store(
                                    q_out_ptr + output_offs,
                                    q_data,
                                    mask=io_mask,
                                )
                                tl.store(
                                    k_out_ptr + output_offs,
                                    k_data,
                                    mask=io_mask,
                                )
                                tl.store(
                                    v_out_ptr + output_offs,
                                    v_data,
                                    mask=io_mask,
                                )
                            # 跨卡写 需要 fence+notify
                            else:
                                remote_peer_ptr = dl.symm_at(
                                    peer_mem_ptr, target_rank
                                )
                                peer_base = (
                                    peer_block_base
                                    + rank * fused_heads * peer_head_size
                                    + head * peer_head_size
                                    + tl.arange(0, BLOCK_S)[:, None]
                                    * BLOCK_D
                                    + offs_d[None, :]
                                )
                                tl.store(
                                    remote_peer_ptr + peer_base,
                                    q_data,
                                    mask=io_mask,
                                )
                                tl.store(
                                    remote_peer_ptr
                                    + peer_base
                                    + HEADS_PER_RANK * peer_head_size,
                                    k_data,
                                    mask=io_mask,
                                )
                                tl.store(
                                    remote_peer_ptr
                                    + peer_base
                                    + 2 * HEADS_PER_RANK * peer_head_size,
                                    v_data,
                                    mask=io_mask,
                                )
            # producer write done
            # fence
            libshmem_device.fence()
            # 再次遍历刚刚的total_wave_tasks任务；填充 signal slot
            for flat_idx in range(
                logical_core_id, total_wave_tasks, PRODUCER_CORES
            ):
                lane = flat_idx % S_WAVE
                task_idx = flat_idx // S_WAVE
                target_rank = task_idx % rank_size
                head_group = task_idx // rank_size
                block_id = wave_start + lane
                if block_id < num_blocks and target_rank != rank:
                    signal_slot = (
                        ((block_id % buffer_num) * rank_size + rank)
                        * HEAD_GROUPS
                        + head_group
                    )
                    # notify
                    dl.notify(
                        signal_mem_ptr + signal_slot * SIGNAL_STRIDE,
                        target_rank,
                        1,
                    )
        # （这里差值为 0）pid 是 consume
        if producers_through == producers_before:
            logical_core_id = pid - producers_before

            for flat_idx in range(
                logical_core_id, total_wave_tasks, CONSUMER_CORES
            ):
                lane = flat_idx % S_WAVE
                task_idx = flat_idx // S_WAVE
                source_rank = task_idx % rank_size
                head_group = task_idx // rank_size
                block_id = wave_start + lane

                if block_id < num_blocks and source_rank != rank:# 等待对应 rank写来的 信号
                    offs_s = block_id * BLOCK_S + tl.arange(0, BLOCK_S)
                    offs_d = tl.arange(0, BLOCK_D)
                    row_mask = offs_s < sequence_length
                    io_mask = row_mask[:, None]
                    output_row = source_rank * sequence_length + offs_s
                    output_base = (
                        output_row[:, None] * stride_out_s
                        + offs_d[None, :] * stride_out_d
                    )
                    peer_block_base = (
                        (block_id % buffer_num) * block_slot_size
                    )
                    signal_slot = (
                        ((block_id % buffer_num) * rank_size + source_rank)
                        * HEAD_GROUPS
                        + head_group
                    )
                    # wait
                    token = dl.wait(
                        signal_mem_ptr + signal_slot * SIGNAL_STRIDE,
                        1,
                        "gpu",
                        "acquire",
                        waitValue=1,
                    )
                    # consume_token
                    local_peer_ptr = dl.consume_token(peer_mem_ptr, token)

                    for local_head in range(0, HEADS_PER_GROUP):
                        head = head_group * HEADS_PER_GROUP + local_head
                        if head < HEADS_PER_RANK:
                            peer_base = (
                                peer_block_base
                                + source_rank * fused_heads * peer_head_size
                                + head * peer_head_size
                                + tl.arange(0, BLOCK_S)[:, None]
                                * BLOCK_D
                                + offs_d[None, :]
                            )
                            # peer_mem -> qkv_out 地址
                            q_data = tl.load(
                                local_peer_ptr + peer_base,
                                mask=io_mask,
                                other=0.0,
                            )
                            k_data = tl.load(
                                local_peer_ptr
                                + peer_base
                                + HEADS_PER_RANK * peer_head_size,
                                mask=io_mask,
                                other=0.0,
                            )
                            v_data = tl.load(
                                local_peer_ptr
                                + peer_base
                                + 2 * HEADS_PER_RANK * peer_head_size,
                                mask=io_mask,
                                other=0.0,
                            )
                            output_offs = output_base + head * stride_out_h
                            tl.store(
                                q_out_ptr + output_offs,
                                q_data,
                                mask=io_mask,
                            )
                            tl.store(
                                k_out_ptr + output_offs,
                                k_data,
                                mask=io_mask,
                            )
                            tl.store(
                                v_out_ptr + output_offs,
                                v_data,
                                mask=io_mask,
                            )

# 算子 launcher
def qkv_fuse_a2a(
    q,
    k,
    v,
    q_out,
    k_out,
    v_out,
    peer_mem,
    signal_mem,
    rank,
    rank_size,
    buffer_num=None,
    comm_block_s=None,
    comm_block_d=COMM_BLOCK_D,
):
    """Launch the shape-adaptive non-interleaved QKV All2All kernel."""
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have identical shapes")
    if q.stride() != k.stride() or q.stride() != v.stride():
        raise ValueError("q, k, and v must have identical strides")
    if q.ndim != 3:
        raise ValueError("q, k, and v must have shape (S, n_head, head_dim)")

    sequence_length, n_head, head_dim = q.shape
    if n_head % rank_size != 0:
        raise ValueError("n_head must be divisible by rank_size")
    heads_per_rank = n_head // rank_size
    if comm_block_s is None:
        comm_block_s = COMM_BLOCK_S
    if comm_block_s != COMM_BLOCK_S or comm_block_d != COMM_BLOCK_D:
        raise ValueError(
            f"this implementation requires tiles ({COMM_BLOCK_S}, {COMM_BLOCK_D})"
        )

    peer_elements, signal_elements, required_buffers = workspace_sizes(
        sequence_length, n_head, rank_size, head_dim, comm_block_s
    )
    if buffer_num is not None and buffer_num < required_buffers:
        raise ValueError(
            f"buffer_num={buffer_num} is smaller than required {required_buffers}"
        )
    if peer_mem.numel() < peer_elements:
        raise ValueError(
            f"peer_mem has {peer_mem.numel()} elements; {peer_elements} required"
        )
    if signal_mem.numel() < signal_elements:
        raise ValueError(
            f"signal_mem has {signal_mem.numel()} elements; "
            f"{signal_elements} required"
        )

    expected_output_shape = (
        sequence_length * rank_size,
        heads_per_rank,
        head_dim,
    )
    for name, output in (("q_out", q_out), ("k_out", k_out), ("v_out", v_out)):
        if tuple(output.shape) != expected_output_shape:
            raise ValueError(
                f"{name} has shape {tuple(output.shape)}; "
                f"expected {expected_output_shape}"
            )
    if q_out.stride() != k_out.stride() or q_out.stride() != v_out.stride():
        raise ValueError("q_out, k_out, and v_out must have identical strides")

    vec_num = NPUUtils().get_aivector_core_num()
    producer_cores, consumer_cores = _aiv_role_counts(vec_num)
    # s_wave
    s_wave = 4 if rank_size <= 6 else (2 if rank_size < 24 else 1)
    group_size = _head_group_size(heads_per_rank)
    common_args = (
        q,
        k,
        v,
        q_out,
        k_out,
        v_out,
        peer_mem,
        signal_mem,
        rank,
        rank_size,
        required_buffers,
        sequence_length,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q_out.stride(0),
        q_out.stride(1),
        q_out.stride(2),
    )

    if group_size == 0:
        kernel_qkv_fuse_a2a_block_flat[vec_num, 1, 1](
            *common_args,
            BLOCK_S=comm_block_s,
            BLOCK_D=comm_block_d,
            HEADS_PER_RANK=heads_per_rank,
            FUSED_HEADS=3 * heads_per_rank,
            PRODUCER_CORES=producer_cores,
            CONSUMER_CORES=consumer_cores,
            SIGNAL_STRIDE=SIGNAL_SLOT_STRIDE,
        )
    else:
        head_groups = math.ceil(heads_per_rank / group_size)
        kernel_qkv_fuse_a2a_head_grouped[vec_num, 1, 1](
            *common_args,
            BLOCK_S=comm_block_s,
            BLOCK_D=comm_block_d,
            S_WAVE=s_wave,
            HEADS_PER_RANK=heads_per_rank,
            HEADS_PER_GROUP=group_size,
            HEAD_GROUPS=head_groups,
            TOTAL_CORES=vec_num,
            PRODUCER_CORES=producer_cores,
            CONSUMER_CORES=consumer_cores,
            SIGNAL_STRIDE=SIGNAL_SLOT_STRIDE,
        )



def run_hccl_benchmark():
    # 环境初始化
    """Small correctness/performance harness; shape lists are configurable."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    rank_size = dist.get_world_size()
    # rank0 打印配置
    if rank == 0:
        total_aiv_cores = NPUUtils().get_aivector_core_num()
        # 打印 producer consumer
        producer_cores, consumer_cores = _aiv_role_counts(total_aiv_cores)
        print(
            "CONFIG "
            f"total_aiv_cores={total_aiv_cores} "
            f"producer_cores={producer_cores} "
            f"consumer_cores={consumer_cores} "
            f"block_s={COMM_BLOCK_S} s_wave=4"
        )
    device = torch.npu.current_device()
    warmup = int(os.environ.get("QKV_PROFILE_WARMUP", "5")) # 预热5次
    bench_iters = int(os.environ.get("QKV_PROFILE_ITERS", "50")) # 正式跑50次
    # 生成序列 待测试的长度 （依赖于2次幂来生成）
    default_sequence_lengths = (
        list(range(2040, 2049))
        + list(range(4088, 4097))
        + list(range(8184, 8193))
    )
    sequence_lengths = [
        int(value)
        for value in os.environ.get(
            "QKV_BENCH_S",
            ",".join(str(value) for value in default_sequence_lengths),
        ).split(",")
    ]
    # rank本地头数量的设置
    local_heads = [
        int(value)
        for value in os.environ.get(
            "QKV_BENCH_H", "1,2,3,4,5,6,7"
        ).split(",")
    ]

    dist.barrier()
    if ash.set_conf_store_tls(False, "") != 0:
        raise RuntimeError("set_conf_store_tls failed")
    attributes = ash.InitAttr()
    attributes.my_rank = rank
    attributes.n_ranks = rank_size
    attributes.local_mem_size = g_ash_size
    attributes.ip_port = G_IP_PORT
    attributes.option_attr.data_op_engine_type = ash.OpEngineType.MTE
    if ash.aclshmem_init(attributes) != 0:
        raise RuntimeError("aclshmem_init failed")

    try:
        for sequence_length in sequence_lengths:
            for heads_per_rank in local_heads:
                #  sequence_length  heads_per_rank
                n_head = heads_per_rank * rank_size # qkv 总头数

                torch.manual_seed(42 + rank)
                # 这里是 MHA : qkv [S,H,hidden]
                q = torch.randn(
                    sequence_length,
                    n_head,
                    COMM_BLOCK_D,
                    dtype=torch.bfloat16,
                    device=device,
                )
                k = torch.randn_like(q)
                v = torch.randn_like(q)
                # 输出形状 从 [S,H,hidden]->[ S*P , H/P ,hidden ]
                output_shape = (
                    sequence_length * rank_size,
                    heads_per_rank,
                    COMM_BLOCK_D,
                )
                q_out = torch.empty(output_shape, dtype=q.dtype, device=device)
                k_out = torch.empty_like(q_out)
                v_out = torch.empty_like(q_out)

                # Warmup golden
                for _ in range(warmup):
                    dist.barrier()
                    qkv_fuse_a2a_golden(q, k, v, rank_size)
                    torch.npu.synchronize()
                    dist.barrier()

                # Benchmark golden (msprof captures these)
                for _ in range(bench_iters):
                    dist.barrier()
                    q_golden, k_golden, v_golden = qkv_fuse_a2a_golden(
                        q, k, v, rank_size
                    )
                    torch.npu.synchronize()
                    dist.barrier()
                # 计算 缓冲区 信号量 迭代num
                peer_elements, signal_elements, buffer_num = workspace_sizes(
                    sequence_length, n_head, rank_size
                )
                peer_mem = ash.aclshmem_create_tensor(
                    [peer_elements], dtype=q.dtype, device_id=rank
                )
                signal_mem = ash.aclshmem_create_tensor(
                    [signal_elements], dtype=torch.int32, device_id=rank
                )
                peer_mem.zero_()
                signal_mem.zero_()
                dist.barrier()

                # Warmup kernel
                for _ in range(warmup):
                    signal_mem.zero_()
                    dist.barrier()
                    qkv_fuse_a2a(
                        q, k, v,
                        q_out, k_out, v_out,
                        peer_mem, signal_mem,
                        rank, rank_size, buffer_num,
                    )
                    torch.npu.synchronize()
                    dist.barrier()

                # Benchmark kernel (msprof captures these)
                for _ in range(bench_iters):
                    signal_mem.zero_()
                    dist.barrier()
                    qkv_fuse_a2a(
                        q, k, v,
                        q_out, k_out, v_out,
                        peer_mem, signal_mem,
                        rank, rank_size, buffer_num,
                    )
                    torch.npu.synchronize()
                    dist.barrier()

                # Accuracy check
                passed = torch.tensor([1], dtype=torch.int32, device=device)
                error_msg = ""
                try:
                    torch.testing.assert_close(
                        q_out, q_golden, rtol=1e-3, atol=1e-3
                    )
                    torch.testing.assert_close(
                        k_out, k_golden, rtol=1e-3, atol=1e-3
                    )
                    torch.testing.assert_close(
                        v_out, v_golden, rtol=1e-3, atol=1e-3
                    )
                except AssertionError as e:
                    passed[0] = 0
                    error_msg = str(e)

                all_passed = [
                    torch.zeros(1, dtype=torch.int32, device=device)
                    for _ in range(rank_size)
                ]
                dist.all_gather(all_passed, passed)

                dist.barrier()
                for rank_id in range(rank_size):
                    if rank == rank_id:
                        if all_passed[rank_id].item() == 1:
                            print(
                                f"  {GREEN}[PASS]{RESET} Rank {rank}: "
                                f"S={sequence_length}, H={n_head} matches golden.",
                                flush=True,
                            )
                        else:
                            print(
                                f"  {RED}[FAIL]{RESET} Rank {rank}: "
                                f"S={sequence_length}, H={n_head} failed. "
                                f"Details:\n{error_msg}",
                                flush=True,
                            )
                    dist.barrier()

                ash.aclshmem_free_tensor(peer_mem)
                ash.aclshmem_free_tensor(signal_mem)
                dist.barrier()

                if any(r.item() == 0 for r in all_passed):
                    raise AssertionError(
                        f"Parameter sweep failed at case: "
                        f"S={sequence_length}, H={n_head}"
                    )
    finally:
        ash.aclshmem_finalize()
        dist.destroy_process_group()


if __name__ == "__main__":
    run_hccl_benchmark()
