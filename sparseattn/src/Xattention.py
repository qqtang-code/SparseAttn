import torch
import math
import torch.nn.functional as F
from block_sparse_attn import block_sparse_attn_func
import triton
import triton.language as tl


@triton.jit
def softmax_fuse_block_sum_kernel_causal(
    In,
    Out,
    scale,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    real_q_len,
    k_len,  # we assume k_len is divisible by chunk size
    chunk_start,
    chunk_end,
    segment_size: tl.constexpr,
    block_size: tl.constexpr,
):
    block_id = tl.program_id(0)
    head_id = tl.program_id(1)
    batch_id = tl.program_id(2)

    offs_q = tl.arange(0, block_size) + chunk_start + block_id * block_size
    offs_k = tl.arange(0, segment_size)

    num_iters = k_len // segment_size
    num_iters_before_causal = (
        chunk_start + (block_id + 1) * block_size - 1
    ) // segment_size

    m_i = tl.zeros([block_size], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([block_size], dtype=tl.float32) + 1.0

    input_ptr = (
        In
        + batch_id * input_stride_0
        + head_id * input_stride_1
        + block_id * block_size * input_stride_2
    )
    input_ptr = (
        input_ptr
        + tl.arange(0, segment_size)
        + tl.arange(0, block_size)[:, None] * input_stride_2
    )

    output_ptr = (
        Out
        + batch_id * output_stride_0
        + head_id * output_stride_1
        + block_id * output_stride_2
    )
    output_ptr = output_ptr + tl.arange(0, segment_size // block_size)

    for iter in range(0, num_iters_before_causal):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        m_local = tl.max(X, 1)
        m_new = tl.maximum(m_i, m_local)
        alpha = tl.math.exp2(m_i - m_new)

        X = X - m_new[:, None]
        l_local = tl.sum(tl.math.exp2(X), 1)
        l_i = l_i * alpha + l_local

        m_i = m_new

    for iter in range(num_iters_before_causal, num_iters_before_causal + 1):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        mask = offs_q[:, None] >= (offs_k[None, :] + iter * segment_size)
        X = tl.where(mask, X, -1.0e6)
        m_local = tl.max(X, 1)
        m_new = tl.maximum(m_i, m_local)
        alpha = tl.math.exp2(m_i - m_new)

        X = X - m_new[:, None]
        l_local = tl.sum(tl.math.exp2(X), 1)
        l_i = l_i * alpha + l_local

        m_i = m_new

    l_i_inv = 1.0 / l_i

    sum_mask = offs_q[:, None] < real_q_len

    for iter in range(0, num_iters_before_causal):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
        X = tl.where(sum_mask, X, 0)
        X = tl.reshape(X, (block_size, segment_size // block_size, block_size))
        X = tl.sum(X, 2)
        X = tl.sum(X, 0)
        tl.store(
            output_ptr + iter * segment_size // block_size, X.to(Out.type.element_ty)
        )

    for iter in range(num_iters_before_causal, num_iters_before_causal + 1):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        mask = offs_q[:, None] >= (offs_k[None, :] + iter * segment_size)
        X = tl.where(mask, X, -1.0e6)
        X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
        X = tl.where(sum_mask, X, 0)
        X = tl.reshape(X, (block_size, segment_size // block_size, block_size))
        X = tl.sum(X, 2)
        X = tl.sum(X, 0)
        tl.store(
            output_ptr + iter * segment_size // block_size, X.to(Out.type.element_ty)
        )

    for iter in range(num_iters_before_causal + 1, num_iters):
        X = tl.zeros([segment_size // block_size], dtype=tl.float32)
        tl.store(
            output_ptr + iter * segment_size // block_size, X.to(Out.type.element_ty)
        )


@triton.jit
def softmax_fuse_block_sum_kernel_non_causal(
    In,
    Out,
    scale,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    real_q_len,
    k_len,  # we assume k_len is divisible by chunk size
    chunk_start,
    chunk_end,
    segment_size: tl.constexpr,
    block_size: tl.constexpr,
):
    block_id = tl.program_id(0)
    head_id = tl.program_id(1)
    batch_id = tl.program_id(2)

    offs_q = tl.arange(0, block_size) + chunk_start + block_id * block_size
    offs_k = tl.arange(0, segment_size)

    num_iters = k_len // segment_size

    m_i = tl.zeros([block_size], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([block_size], dtype=tl.float32) + 1.0

    input_ptr = (
        In
        + batch_id * input_stride_0
        + head_id * input_stride_1
        + block_id * block_size * input_stride_2
    )
    input_ptr = (
        input_ptr
        + tl.arange(0, segment_size)
        + tl.arange(0, block_size)[:, None] * input_stride_2
    )

    output_ptr = (
        Out
        + batch_id * output_stride_0
        + head_id * output_stride_1
        + block_id * output_stride_2
    )
    output_ptr = output_ptr + tl.arange(0, segment_size // block_size)

    for iter in range(0, num_iters):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        m_local = tl.max(X, 1)
        m_new = tl.maximum(m_i, m_local)
        alpha = tl.math.exp2(m_i - m_new)

        X = X - m_new[:, None]
        l_local = tl.sum(tl.math.exp2(X), 1)
        l_i = l_i * alpha + l_local

        m_i = m_new

    l_i_inv = 1.0 / l_i

    sum_mask = offs_q[:, None] < real_q_len

    for iter in range(0, num_iters):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
        X = tl.where(sum_mask, X, 0)
        X = tl.reshape(X, (block_size, segment_size // block_size, block_size))
        X = tl.sum(X, 2)
        X = tl.sum(X, 0)
        tl.store(
            output_ptr + iter * segment_size // block_size, X.to(Out.type.element_ty)
        )


@triton.jit
def flat_group_gemm_kernel(
    Q,
    K,
    Out,
    stride_qz,
    stride_qh,
    stride_qn,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_oz,
    stride_oh,
    stride_on,
    chunk_start,
    chunk_end,
    H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    block_m = tl.program_id(0).to(tl.int64)
    block_n = tl.program_id(1).to(tl.int64)
    batch_id = tl.program_id(2).to(tl.int64) // H
    head_id = tl.program_id(2).to(tl.int64) % H

    if chunk_start + (block_m + 1) * BLOCK_M <= block_n * BLOCK_N:
        return

    Q_ptrs = (
        Q + batch_id * stride_qz + head_id * stride_qh + block_m * BLOCK_M * stride_qn
    )
    K_ptrs = (
        K + batch_id * stride_kz + head_id * stride_kh + block_n * BLOCK_N * stride_kn
    )

    Q_ptrs = (
        Q_ptrs
        + tl.arange(0, BLOCK_M)[:, None] * stride_qn
        + tl.arange(0, BLOCK_K)[None, :]
    )
    K_ptrs = (
        K_ptrs
        + tl.arange(0, BLOCK_N)[None, :] * stride_kn
        + tl.arange(0, BLOCK_K)[:, None]
    )

    num_iters = HEAD_DIM // BLOCK_K
    o = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for iter in range(num_iters):
        q = tl.load(Q_ptrs + iter * BLOCK_K)
        k = tl.load(K_ptrs + iter * BLOCK_K)
        o += tl.dot(q, k)

    O_ptrs = (
        Out
        + batch_id * stride_oz
        + head_id * stride_oh
        + block_m * BLOCK_M * stride_on
        + block_n * BLOCK_N
    )
    O_ptrs = (
        O_ptrs
        + tl.arange(0, BLOCK_M)[:, None] * stride_on
        + tl.arange(0, BLOCK_N)[None, :]
    )

    tl.store(O_ptrs, o.to(Out.type.element_ty))


@triton.jit
def flat_group_gemm_fuse_reshape_kernel(
    Q,
    K,
    Out,
    stride_qz,
    stride_qh,
    stride_qn,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_oz,
    stride_oh,
    stride_on,
    chunk_start,
    chunk_end,
    H: tl.constexpr,
    STRIDE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    is_causal: tl.constexpr,
):
    block_m = tl.program_id(0).to(tl.int64)
    block_n = tl.program_id(1).to(tl.int64)
    batch_id = tl.program_id(2).to(tl.int64) // H
    head_id = tl.program_id(2).to(tl.int64) % H

    if is_causal:
        if chunk_start + (block_m + 1) * BLOCK_M <= block_n * BLOCK_N:
            return

    Q_ptrs = (
        Q
        + batch_id * stride_qz
        + head_id * stride_qh
        + block_m * BLOCK_M * STRIDE * stride_qn
    )
    K_ptrs = (
        K
        + batch_id * stride_kz
        + head_id * stride_kh
        + block_n * BLOCK_N * STRIDE * stride_kn
    )

    Q_ptrs = (
        Q_ptrs
        + tl.arange(0, BLOCK_M)[:, None] * (stride_qn * STRIDE)
        + tl.arange(0, HEAD_DIM)[None, :]
        + stride_qn * (STRIDE - 1)
    )
    K_ptrs = (
        K_ptrs
        + tl.arange(0, BLOCK_N)[None, :] * (stride_kn * STRIDE)
        + tl.arange(0, HEAD_DIM)[:, None]
    )

    o = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for iter in range(STRIDE):
        q = tl.load(Q_ptrs - iter * stride_qn)
        k = tl.load(K_ptrs + iter * stride_kn)
        o += tl.dot(q, k)

    O_ptrs = (
        Out
        + batch_id * stride_oz
        + head_id * stride_oh
        + block_m * BLOCK_M * stride_on
        + block_n * BLOCK_N
    )
    O_ptrs = (
        O_ptrs
        + tl.arange(0, BLOCK_M)[:, None] * stride_on
        + tl.arange(0, BLOCK_N)[None, :]
    )

    tl.store(O_ptrs, o.to(Out.type.element_ty))


def softmax_fuse_block_sum(
    attn_weights_slice,
    reshaped_block_size,
    segment_size,
    chunk_start,
    chunk_end,
    real_q_len,
    scale,
    is_causal=True,
):
    batch_size, num_heads, q_len, k_len = attn_weights_slice.shape
    assert q_len % reshaped_block_size == 0
    try:
        assert k_len % segment_size == 0
    except:
        breakpoint()
    assert segment_size % reshaped_block_size == 0
    assert attn_weights_slice.stride(-1) == 1

    output = torch.empty(
        (
            batch_size,
            num_heads,
            q_len // reshaped_block_size,
            k_len // reshaped_block_size,
        ),
        dtype=attn_weights_slice.dtype,
        device=attn_weights_slice.device,
    )

    grid = (q_len // reshaped_block_size, num_heads, batch_size)

    if is_causal:
        softmax_fuse_block_sum_kernel_causal[grid](
            attn_weights_slice,
            output,
            scale,
            attn_weights_slice.stride(0),
            attn_weights_slice.stride(1),
            attn_weights_slice.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            real_q_len,
            k_len,
            chunk_start,
            chunk_end,
            segment_size,
            reshaped_block_size,
        )
    else:
        softmax_fuse_block_sum_kernel_non_causal[grid](
            attn_weights_slice,
            output,
            scale,
            attn_weights_slice.stride(0),
            attn_weights_slice.stride(1),
            attn_weights_slice.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            real_q_len,
            k_len,
            chunk_start,
            chunk_end,
            segment_size,
            reshaped_block_size,
        )

    return output


def flat_group_gemm(query_states, key_states, chunk_start, chunk_end):
    batch_size, num_heads, q_len, head_dim = query_states.shape
    kv_len = key_states.shape[2]

    output = torch.empty(
        (batch_size, num_heads, q_len, kv_len),
        dtype=query_states.dtype,
        device=query_states.device,
    )
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    grid = (q_len // BLOCK_M, kv_len // BLOCK_N, batch_size * num_heads)
    flat_group_gemm_kernel[grid](
        query_states,
        key_states,
        output,
        query_states.stride(0),
        query_states.stride(1),
        query_states.stride(2),
        key_states.stride(0),
        key_states.stride(1),
        key_states.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        chunk_start,
        chunk_end,
        num_heads,
        head_dim,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )

    return output


def flat_group_gemm_fuse_reshape(
    query_states, key_states, stride, chunk_start, chunk_end, is_causal=True
):
    batch_size, num_heads, q_len, head_dim = query_states.shape
    kv_len = key_states.shape[2]

    assert key_states.shape[0] == batch_size
    assert key_states.shape[1] == num_heads
    assert key_states.shape[3] == head_dim

    output = torch.empty(
        (batch_size, num_heads, q_len // stride, kv_len // stride),
        dtype=query_states.dtype,
        device=query_states.device,
    )
    # BLOCK_M = 128
    # BLOCK_N = 128
    # H20
    BLOCK_M = 64
    BLOCK_N = 64
    assert q_len % (stride * BLOCK_M) == 0
    assert kv_len % (stride * BLOCK_N) == 0

    grid = (
        q_len // stride // BLOCK_M,
        kv_len // stride // BLOCK_N,
        batch_size * num_heads,
    )
    flat_group_gemm_fuse_reshape_kernel[grid](
        query_states,
        key_states,
        output,
        query_states.stride(0),
        query_states.stride(1),
        query_states.stride(2),
        key_states.stride(0),
        key_states.stride(1),
        key_states.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        chunk_start,
        chunk_end,
        num_heads,
        stride,
        head_dim,
        BLOCK_M,
        BLOCK_N,
        is_causal,
    )

    return output


def xattn_estimate(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    block_size,
    stride,
    norm=1,
    softmax=True,
    threshold=0.9,
    chunk_size=16384,
    select_mode="inverse",
    use_triton=True,
    causal=True,
    kdb: int = 1,
    keep_sink=False,
    keep_recent=False,
) -> torch.Tensor:
    batch_size, num_kv_head, k_len, head_dim = key_states.shape
    batch_size, num_q_head, q_len, head_dim = query_states.shape
    assert num_q_head == num_kv_head

    k_num_to_pad = ((k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len
    q_num_to_pad = ((q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len
    k_chunk_num = (k_len + k_num_to_pad) // chunk_size
    k_block_num = (k_len + k_num_to_pad) // block_size
    q_chunk_num = (q_len + q_num_to_pad) // chunk_size
    q_block_num = (q_len + q_num_to_pad) // block_size
    # [BugFix] fix chunked_prefill_underperforming_issue with use_triton=False
    assert k_chunk_num >= q_chunk_num
    offset_token_chunk_num = k_chunk_num - q_chunk_num

    if k_num_to_pad > 0:
        pad_key_states = F.pad(key_states, (0, 0, 0, k_num_to_pad), value=0).to("cuda")
    else:
        pad_key_states = key_states
    if q_num_to_pad > 0:
        pad_query_states = F.pad(query_states, (0, 0, 0, q_num_to_pad), value=0).to(
            "cuda"
        )
    else:
        pad_query_states = query_states

    assert num_kv_head == num_q_head
    attn_sum_list = []
    simple_mask_list = []

    # if use_triton and (
    #     "100" not in torch.cuda.get_device_properties(torch.cuda.current_device()).name
    # ):
    #     use_triton = False
    #     print(
    #         "setting use triton to false. Triton kernel not surpported on this device"
    #     )

    reshaped_chunk_size = chunk_size // stride
    reshaped_block_size = block_size // stride
    k_reshaped_num_to_pad = k_num_to_pad // stride
    k_reshaped_seq_len = (k_len + k_num_to_pad) // stride
    q_reshaped_num_to_pad = q_num_to_pad // stride
    num_blocks_per_chunk = reshaped_chunk_size // reshaped_block_size
    if not use_triton:
        if select_mode == "random":
            perm_idx = torch.randperm(stride)
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_query = torch.cat(
                [
                    pad_query_states[:, :, perm_idx[i] :: stride, :]
                    for i in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "inverse" or select_mode == "":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_query = torch.cat(
                [
                    (pad_query_states[:, :, (stride - 1 - q) :: (stride * kdb), :])
                    for q in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "slash":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_query = torch.cat(
                [(pad_query_states[:, :, q::stride, :]) for q in range(stride)], dim=-1
            )
        elif select_mode == "double":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_key = reshaped_key + torch.cat(
                [reshaped_key[:, :, :, head_dim:], reshaped_key[:, :, :, 0:head_dim]],
                dim=-1,
            )
            reshaped_query = torch.cat(
                [
                    (pad_query_states[:, :, (stride - 1 - q) :: stride, :])
                    for q in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "triple":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_key = reshaped_key + torch.cat(
                [reshaped_key[:, :, :, head_dim:], reshaped_key[:, :, :, 0:head_dim]],
                dim=-1,
            )
            reshaped_key = reshaped_key + torch.cat(
                [reshaped_key[:, :, :, -head_dim:], reshaped_key[:, :, :, 0:-head_dim]],
                dim=-1,
            )
            reshaped_query = torch.cat(
                [
                    (pad_query_states[:, :, (stride - 1 - q) :: stride, :])
                    for q in range(stride)
                ],
                dim=-1,
            )
        assert reshaped_key.shape[-2] == k_reshaped_seq_len

    for chunk_idx in range(q_chunk_num):
        if use_triton:
            if kdb != 1:
                raise ValueError("use_triton and kdb cannot be used together")
            attn_weights_slice = flat_group_gemm_fuse_reshape(
                pad_query_states[
                    :,
                    :,
                    (chunk_idx * reshaped_chunk_size) * stride : (
                        chunk_idx * reshaped_chunk_size + reshaped_chunk_size
                    )
                    * stride,
                    :,
                ],
                pad_key_states,
                stride,
                (k_block_num - q_block_num) * reshaped_block_size
                + chunk_idx * reshaped_chunk_size,
                (k_block_num - q_block_num) * reshaped_block_size
                + chunk_idx * reshaped_chunk_size
                + reshaped_chunk_size,
                is_causal=causal,
            )
            attn_sum = softmax_fuse_block_sum(
                attn_weights_slice,
                reshaped_block_size,
                min(4096, reshaped_block_size),
                (k_block_num - q_block_num) * reshaped_block_size
                + chunk_idx * reshaped_chunk_size,
                (k_block_num - q_block_num) * reshaped_block_size
                + chunk_idx * reshaped_chunk_size
                + reshaped_chunk_size,
                k_reshaped_seq_len - k_reshaped_num_to_pad,
                1.4426950408889634 / math.sqrt(head_dim) / stride / norm,
                is_causal=causal,
            )
        else:
            chunked_query = reshaped_query[
                :,
                :,
                (chunk_idx * reshaped_chunk_size) // kdb : (
                    chunk_idx * reshaped_chunk_size + reshaped_chunk_size
                )
                // kdb,
                :,
            ]
            attn_weights_slice = torch.matmul(
                chunked_query,
                reshaped_key.transpose(2, 3),
            ).to("cuda")

            attn_weights_slice = (
                attn_weights_slice / math.sqrt(head_dim) / stride / norm
            )

            if causal:
                causal_mask = torch.zeros(
                    (
                        batch_size,
                        num_q_head,
                        reshaped_chunk_size,
                        reshaped_chunk_size * k_chunk_num,
                    ),
                    device=key_states.device,
                )
                causal_mask[:, :, :, (-k_reshaped_num_to_pad):] = float("-inf")
                chunk_start = (chunk_idx + offset_token_chunk_num) * reshaped_chunk_size
                chunk_end = chunk_start + reshaped_chunk_size
                causal_mask[:, :, :, chunk_start:chunk_end] = torch.triu(
                    torch.ones(
                        1,
                        num_q_head,
                        reshaped_chunk_size,
                        reshaped_chunk_size,
                        device=key_states.device,
                    )
                    * float("-inf"),
                    diagonal=1,
                )

                if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
                    causal_mask[:, :, (-(q_reshaped_num_to_pad // kdb)) :, :] = float(
                        "-inf"
                    )

                causal_mask[:, :, :, chunk_end:] = float("-inf")
                causal_mask = causal_mask[:, :, kdb - 1 :: kdb, :]
                attn_weights_slice = attn_weights_slice + causal_mask.to(
                    attn_weights_slice.device
                )

            if softmax:
                attn_weights_slice = F.softmax(
                    attn_weights_slice, dim=-1, dtype=torch.float32
                ).to(pad_query_states.dtype)
            else:
                attn_weights_slice = torch.exp(attn_weights_slice).to(
                    pad_query_states.dtype
                )
            attn_weights_slice = F.dropout(attn_weights_slice, p=0, training=False)

            if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
                attn_weights_slice[:, :, (-(q_reshaped_num_to_pad // kdb)) :, :] = 0

            attn_sum = (
                attn_weights_slice.view(
                    batch_size,
                    num_kv_head,
                    num_blocks_per_chunk,
                    reshaped_block_size // kdb,
                    -1,
                    reshaped_block_size,
                )
                .sum(dim=-1)
                .sum(dim=-2)
                .to("cuda")
            )
            del chunked_query

        simple_mask = find_blocks_chunked(
            attn_sum,
            k_block_num - q_block_num + chunk_idx * num_blocks_per_chunk,
            threshold,
            None,
            decoding=False,
            mode="prefill",
            causal=causal,
        )

        attn_sum_list.append(attn_sum)
        simple_mask_list.append(simple_mask)

        del attn_weights_slice

    if not use_triton:
        del reshaped_query, reshaped_key
    attn_sums = torch.cat(attn_sum_list, dim=-2)
    simple_masks = torch.cat(simple_mask_list, dim=-2)

    if causal:
        # Apply causal mask in-place to avoid creating large intermediate tensors
        # Create upper triangular mask more efficiently
        mask_size = min(q_block_num, simple_masks.shape[-1])
        if mask_size > 0:
            causal_block_mask = ~torch.triu(
                torch.ones(
                    mask_size, mask_size, device=simple_masks.device, dtype=torch.bool
                ),
                diagonal=1,
            )
            # Apply the mask to the relevant portion
            simple_masks[:, :, -mask_size:, -mask_size:] &= causal_block_mask
    if keep_sink:
        simple_masks[:, :, 0, :] = True
    if keep_recent:
        eye_matrix = torch.eye(q_block_num, device=simple_masks.device, dtype=bool)
        eye_matrix_expanded = (
            eye_matrix.unsqueeze(0)
            .unsqueeze(0)
            .expand(1, num_kv_head, q_block_num, q_block_num)
        )
        simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
            eye_matrix_expanded, True, simple_masks[:, :, -q_block_num:, -q_block_num:]
        )

    return attn_sums, simple_masks


def Xattention_prefill_dim4(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    stride,
    cu_seq_lens,
    norm=1,
    threshold=0.8,
    block_size=128,
    use_triton=True,
    causal=True,
    kdb=1,
    chunk_size=None,
    keep_sink=False,
    keep_recent=False,
    head_mask_type=None,
    sink_num=1,
    local_num=16,
):
    batch_size, num_heads, max_q_len, head_dim = query_states.shape
    _, _, max_k_len, _ = key_states.shape
    
    # 计算每个batch的有效长度
    valid_lengths = cu_seq_lens[1:] - cu_seq_lens[:-1]
    
    max_q_blocks_of_return = 0#表示返回的mask.shape[2]
    max_k_blocks_of_return = 0#表示返回的mask.shape[3]

    # 存储每个batch的结果
    approx_simple_mask_list = []

    # 对每个batch单独处理
    for i in range(batch_size):
        valid_len = valid_lengths[i]
        
        # 截取当前batch的有效token部分
        current_query = query_states[i:i+1, :, :valid_len, :]  # [1, num_heads, valid_len, head_dim]
        current_key = key_states[i:i+1, :, :valid_len, :]      # [1, num_heads, valid_len, head_dim]
        
        _, _, current_k_len, _ = current_key.shape
        current_q_len = current_k_len
        #这个chunk_size是什么意思？为什么是16384？16384=128*128
        if chunk_size is None:
            chunk_size = int(
                max(
                    min(
                        max(2048, 1 << (current_k_len - 1).bit_length()),
                        128 * 1024 * 2048 // (1 << (current_k_len - 1).bit_length()),
                    ),
                    2048,
                )
            )
        
        # 调用xattn_estimate
        attn_sum, approx_mask = xattn_estimate(
            current_query,
            current_key,
            block_size=block_size,
            stride=stride,
            norm=norm,
            threshold=threshold,
            select_mode="inverse",
            use_triton=use_triton,
            causal=causal,
            chunk_size=chunk_size,
            kdb=kdb,
            keep_sink=keep_sink,
            keep_recent=keep_recent,
        )
        
        #用于后续填充
        _, _, q_blocks_of_return, k_blocks_of_return = approx_mask.shape
        
        max_q_blocks_of_return = max(max_q_blocks_of_return, q_blocks_of_return)
        max_k_blocks_of_return = max(max_k_blocks_of_return, k_blocks_of_return)
  
        # 计算有效的block数量
        valid_q_blocks = (valid_len + block_size - 1) // block_size
        valid_k_blocks = (valid_len + block_size - 1) // block_size
        
        approx_mask[:, :, valid_q_blocks:, :] = False
        approx_mask[:, :, :, valid_k_blocks:] = False
        
        
        approx_simple_mask_list.append(approx_mask)
    # 填充approx_simple_mask_list中的approx_mask到[1, num_heads, max_q_blocks_of_return, max_k_blocks_of_return]
    padded_mask_list = []
    for approx_mask in approx_simple_mask_list:
        _, _, current_q_blocks, current_k_blocks = approx_mask.shape
        
        # 创建填充后的mask
        padded_mask = torch.zeros(
            (1, num_heads, max_q_blocks_of_return, max_k_blocks_of_return),
            dtype=approx_mask.dtype,
            device=approx_mask.device
        )
        
        # 将原始mask数据复制到填充后的mask中
        padded_mask[:, :, :current_q_blocks, :current_k_blocks] = approx_mask
        
        padded_mask_list.append(padded_mask)
    
    # 合并所有batch的结果
    approx_simple_mask = torch.cat(padded_mask_list, dim=0)  # [batch_size, num_heads, max_q_blocks, max_k_blocks]
    
    if query_states.device != key_states.device:
        key_states = key_states.to(query_states.device)
    if query_states.device != value_states.device:
        value_states = value_states.to(query_states.device)
    if approx_simple_mask.device != query_states.device:
        approx_simple_mask = approx_simple_mask.to(query_states.device)

    ####################
    # 根据 cu_seq_lens 转换为去填充格式
    total_seq_len = cu_seq_lens[-1].item()  # 总的有效token数
    
    # 创建去填充的张量
    unpadded_query_states = torch.zeros(
        (total_seq_len, num_heads, head_dim), 
        dtype=query_states.dtype, 
        device=query_states.device
    )
    unpadded_key_states = torch.zeros(
        (total_seq_len, num_heads, head_dim), 
        dtype=key_states.dtype, 
        device=key_states.device
    )
    unpadded_value_states = torch.zeros(
        (total_seq_len, num_heads, head_dim), 
        dtype=value_states.dtype, 
        device=value_states.device
    )
    
    # 填充数据
    for i in range(batch_size):
        start_idx = cu_seq_lens[i].item()
        end_idx = cu_seq_lens[i + 1].item()
        seq_len_i = end_idx - start_idx
        
        # 获取当前batch的有效token数
        actual_seq_len = seq_len_i
        
        # 转换维度并复制数据
        # query_states形状: [batch_size, num_heads, q_len, head_dim]
        unpadded_query_states[start_idx:start_idx + actual_seq_len] = (
            query_states[i, :, :actual_seq_len, :].transpose(0, 1)  # [actual_seq_len, num_heads, head_dim]
        )
        unpadded_key_states[start_idx:start_idx + actual_seq_len] = (
            key_states[i, :, :actual_seq_len, :].transpose(0, 1)    # [actual_seq_len, num_heads, head_dim]
        )
        unpadded_value_states[start_idx:start_idx + actual_seq_len] = (
            value_states[i, :, :actual_seq_len, :].transpose(0, 1)  # [actual_seq_len, num_heads, head_dim]
        )
    

    query_states = unpadded_query_states
    key_states = unpadded_key_states
    value_states = unpadded_value_states
    
    head_mask_type = head_mask_type if head_mask_type is not None else torch.tensor(
        [1 for _ in range(num_heads)], device=query_states.device, dtype=torch.int32
    )
    assert head_mask_type.device == query_states.device
    assert cu_seq_lens.device == query_states.device
    assert key_states.device == query_states.device
    assert value_states.device == query_states.device
    assert approx_simple_mask.device == query_states.device
    
    max_q_block_num = (max_q_len + block_size - 1) // block_size
    max_k_block_num = (max_k_len + block_size - 1) // block_size
    
    # head_mask_type
    mask = (head_mask_type == 1)
    blockmask = approx_simple_mask[:, mask, :max_q_block_num, :max_k_block_num].contiguous()
    streaming_info = torch.tensor([sink_num, local_num] * num_heads, device=query_states.device, dtype=torch.int32)
    
    attn_output = block_sparse_attn_func(
        query_states,
        key_states,
        value_states,
        cu_seq_lens,
        cu_seq_lens,
        head_mask_type,
        streaming_info,
        blockmask,
        max_q_len,
        max_k_len, 
        p_dropout=0.0,
        deterministic=True,
        is_causal=causal,
    )
    
    # 将输出转换回批处理格式
    attn_output_batch = []
    for i in range(batch_size):
        start_idx = cu_seq_lens[i].item()
        end_idx = cu_seq_lens[i + 1].item()
        seq_len_i = end_idx - start_idx
        actual_seq_len = min(seq_len_i, max_q_len)
        
        # 获取当前batch的输出并转换维度
        batch_output = attn_output[start_idx:start_idx + actual_seq_len]  # [actual_seq_len, num_heads, head_dim]
        batch_output = batch_output.transpose(0, 1).unsqueeze(0)  # [1, num_heads, actual_seq_len, head_dim]
        
        # 如果需要填充到原始长度
        if actual_seq_len < max_q_len:
            pad_size = max_q_len - actual_seq_len
            batch_output = F.pad(batch_output, (0, 0, 0, pad_size, 0, 0, 0, 0))
        
        attn_output_batch.append(batch_output)
    
    attn_output = torch.cat(attn_output_batch, dim=0)
    ################################

    del query_states
    # num_to_compute = (k_block_num + 1) * k_block_num / 2 * num_heads

    # print(f"approximated prefilling Computation: {approx_simple_mask.sum() / num_to_compute}")
    del approx_simple_mask
    return attn_output

#此时处理的是unpadded的query_states
def Xattention_prefill_dim3(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    stride,
    cu_seq_lens,
    norm=1,
    threshold=0.8,
    block_size=128,
    use_triton=True,
    causal=True,
    kdb=1,
    chunk_size=None,
    keep_sink=False,
    keep_recent=False,
):  
    num_heads, total_len, head_dim = query_states.shape
    # 计算每个batch的有效长度
    valid_lengths = cu_seq_lens[1:] - cu_seq_lens[:-1]
    batch_size = valid_lengths.shape[-1]
    
    max_q_blocks_of_return = 0#表示返回的mask.shape[2]
    max_k_blocks_of_return = 0#表示返回的mask.shape[3]

    # 存储每个batch的结果
    approx_simple_mask_list = []

    # 对每个batch单独处理
    for i in range(batch_size):
        
        start = cu_seq_lens[i].item()
        end = cu_seq_lens[i+1].item()
        valid_len = end - start
        
        # 截取当前batch的有效token部分
        current_query = query_states[:, start: end, :].unsqueeze(0).contiguous() #[1, num_heads, seq_len, head_dim]
        current_key = key_states[:, start: end, :].unsqueeze(0).contiguous() #[1, num_heads, seq_len, head_dim]
        
        _, _, current_k_len, _ = current_key.shape
        current_q_len = current_k_len
        #这个chunk_size是什么意思？为什么是16384？16384=128*128
        if chunk_size is None:
            chunk_size = int(
                max(
                    min(
                        max(2048, 1 << (current_k_len - 1).bit_length()),
                        128 * 1024 * 2048 // (1 << (current_k_len - 1).bit_length()),
                    ),
                    2048,
                )
            )
        
        # 调用xattn_estimate
        attn_sum, approx_mask = xattn_estimate(
            current_query,
            current_key,
            block_size=block_size,
            stride=stride,
            norm=norm,
            threshold=threshold,
            select_mode="inverse",
            use_triton=use_triton,
            causal=causal,
            chunk_size=chunk_size,
            kdb=kdb,
            keep_sink=keep_sink,
            keep_recent=keep_recent,
        )
        
        #用于后续填充
        _, _, q_blocks_of_return, k_blocks_of_return = approx_mask.shape
        
        max_q_blocks_of_return = max(max_q_blocks_of_return, q_blocks_of_return)
        max_k_blocks_of_return = max(max_k_blocks_of_return, k_blocks_of_return)
  
        # 计算有效的block数量
        valid_q_blocks = (valid_len + block_size - 1) // block_size
        valid_k_blocks = (valid_len + block_size - 1) // block_size
        
        approx_mask[:, :, valid_q_blocks:, :] = False
        approx_mask[:, :, :, valid_k_blocks:] = False
        
        
        approx_simple_mask_list.append(approx_mask)
    # 填充approx_simple_mask_list中的approx_mask到[1, num_heads, max_q_blocks_of_return, max_k_blocks_of_return]
    padded_mask_list = []
    for approx_mask in approx_simple_mask_list:
        _, _, current_q_blocks, current_k_blocks = approx_mask.shape
        
        # 创建填充后的mask
        padded_mask = torch.zeros(
            (1, num_heads, max_q_blocks_of_return, max_k_blocks_of_return),
            dtype=approx_mask.dtype,
            device=approx_mask.device
        )
        
        # 将原始mask数据复制到填充后的mask中
        padded_mask[:, :, :current_q_blocks, :current_k_blocks] = approx_mask
        
        padded_mask_list.append(padded_mask)
    # 合并所有batch的结果
    approx_simple_mask = torch.cat(padded_mask_list, dim=0)  # [batch_size, num_heads, max_q_blocks, max_k_blocks]
    
    if query_states.device != key_states.device:
        key_states = key_states.to(query_states.device)
    if query_states.device != value_states.device:
        value_states = value_states.to(query_states.device)
    if approx_simple_mask.device != query_states.device:
        approx_simple_mask = approx_simple_mask.to(query_states.device)

    ####################
    head_mask_type = torch.tensor(
        [1 for _ in range(num_heads)], device=query_states.device, dtype=torch.int32
    )
    assert head_mask_type.device == query_states.device
    assert cu_seq_lens.device == query_states.device
    assert key_states.device == query_states.device
    assert value_states.device == query_states.device
    assert approx_simple_mask.device == query_states.device
    
    max_q_len = valid_lengths.max().item()
    max_k_len = max_q_len
    max_q_block_num = (max_q_len + block_size - 1) // block_size
    max_k_block_num = (max_k_len + block_size - 1) // block_size

    attn_output = block_sparse_attn_func(
        query_states.transpose(0, 1).contiguous(),
        key_states.transpose(0, 1).contiguous(),
        value_states.transpose(0, 1).contiguous(),
        cu_seq_lens,
        cu_seq_lens,
        head_mask_type,
        None,
        approx_simple_mask[:, :, :max_q_block_num, :max_k_block_num].contiguous(),
        max_q_len,
        max_k_len, 
        p_dropout=0.0,
        deterministic=True,
        is_causal=causal,
    )

    # 将输出转换回批处理格式
    # attn_output_batch = []
    # for i in range(batch_size):
    #     start_idx = cu_seq_lens[i].item()
    #     end_idx = cu_seq_lens[i + 1].item()
    #     seq_len_i = end_idx - start_idx
    #     actual_seq_len = min(seq_len_i, max_q_len)
        
    #     # 获取当前batch的输出并转换维度
    #     batch_output = attn_output[start_idx:start_idx + actual_seq_len]  # [actual_seq_len, num_heads, head_dim]
    #     batch_output = batch_output.transpose(0, 1).unsqueeze(0)  # [1, num_heads, actual_seq_len, head_dim]
        
    #     # 如果需要填充到原始长度
    #     if actual_seq_len < max_q_len:
    #         pad_size = max_q_len - actual_seq_len
    #         batch_output = F.pad(batch_output, (0, 0, 0, pad_size, 0, 0, 0, 0))
        
    #     attn_output_batch.append(batch_output)
    
    # attn_output = torch.cat(attn_output_batch, dim=0)
    ################################

    del query_states
    del approx_simple_mask
    return attn_output

from sparseattn.src.utils import *
import torch
import math
import torch.nn.functional as F
from block_sparse_attn import block_sparse_attn_func
import triton
import triton.language as tl

# ... (previous Triton kernels and functions are expected to be above in the same file) 

if __name__ == "__main__":
    
    #case1
    batch_size = 2
    num_heads = 32
    q_len = 32768
    kv_len = 32768
    head_dim = 128


    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("警告：当前未检测到 CUDA。Triton 内核将无法运行，测试可能失败或速度极慢。")

    # 随机输入（注意 dtype 用 float16/float32 以匹配实际模型）
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    cu_seq_lens = torch.tensor([0, 1024, 33792], device=device, dtype=torch.int32)
    query = torch.randn((batch_size, num_heads, q_len, head_dim), dtype=dtype, device=device)
    key = torch.randn((batch_size, num_heads, kv_len, head_dim), dtype=dtype, device=device)
    value = torch.randn((batch_size, num_heads, kv_len, head_dim), dtype=dtype, device=device)

    # Xattention_prefill 参数
    stride = 16
    norm = 1
    threshold = 0.9
    block_size = 128
    use_triton = True if device.type == "cuda" else False
    causal = True
    kdb = 1
    chunk_size = 16384
    keep_sink = False
    keep_recent = False

    print(f"Running Xattention_prefill on device={device}, use_triton={use_triton}")

    try:
        out = Xattention_prefill_dim4(
            query_states=query,
            key_states=key,
            value_states=value,
            stride=stride,
            cu_seq_lens=cu_seq_lens,
            norm=norm,
            threshold=threshold,
            block_size=block_size,
            use_triton=use_triton,
            causal=causal,
            kdb=kdb,
            chunk_size=chunk_size,
            keep_sink=keep_sink,
            keep_recent=keep_recent,
        )

        print("Xattention_prefill 返回: ")
        print(f"  类型: {type(out)}")
        try:
            print(f"  形状: {out.shape}")
            print(f"  dtype: {out.dtype}, device: {out.device}")
            # 打印少量元素以作 sanity check
            flat = out.detach().cpu()
            print(f"  少量元素 (前 8): {flat.flatten()[:8]}")
        except Exception:
            print("无法读取返回张量的详细信息（可能为自定义对象或未成功计算）")

    except Exception as e:
        import traceback

        print("调用 Xattention_prefill 出现异常:")
        traceback.print_exc()
        print("请检查 Triton/CUDA 环境、block_size 与输入长度是否匹配，以及可能的显存限制。")

    print("测试结束。")
    #case2
    num_heads = 32
    head_dim = 128
    total_len = 1024+32768

    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("警告：当前未检测到 CUDA。Triton 内核将无法运行，测试可能失败或速度极慢。")

    # 随机输入（注意 dtype 用 float16/float32 以匹配实际模型）
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    cu_seq_lens = torch.tensor([0, 1024, 33792], device=device, dtype=torch.int32)
    query = torch.randn((num_heads, total_len, head_dim), dtype=dtype, device=device)
    key = torch.randn((num_heads, total_len, head_dim), dtype=dtype, device=device)
    value = torch.randn((num_heads, total_len, head_dim), dtype=dtype, device=device)

    # Xattention_prefill 参数
    stride = 16
    norm = 1
    threshold = 0.9
    block_size = 128
    use_triton = True if device.type == "cuda" else False
    causal = True
    kdb = 1
    chunk_size = 16384
    keep_sink = False
    keep_recent = False

    print(f"Running Xattention_prefill on device={device}, use_triton={use_triton}")

    try:
        out = Xattention_prefill_dim3(
            query_states=query,
            key_states=key,
            value_states=value,
            stride=stride,
            cu_seq_lens=cu_seq_lens,
            norm=norm,
            threshold=threshold,
            block_size=block_size,
            use_triton=use_triton,
            causal=causal,
            kdb=kdb,
            chunk_size=chunk_size,
            keep_sink=keep_sink,
            keep_recent=keep_recent,
        )

        print("Xattention_prefill 返回: ")
        print(f"  类型: {type(out)}")
        try:
            print(f"  形状: {out.shape}")
            print(f"  dtype: {out.dtype}, device: {out.device}")
            # 打印少量元素以作 sanity check
            flat = out.detach().cpu()
            print(f"  少量元素 (前 8): {flat.flatten()[:8]}")
        except Exception:
            print("无法读取返回张量的详细信息（可能为自定义对象或未成功计算）")

    except Exception as e:
        import traceback

        print("调用 Xattention_prefill 出现异常:")
        traceback.print_exc()
        print("请检查 Triton/CUDA 环境、block_size 与输入长度是否匹配，以及可能的显存限制。")

    print("测试结束。")