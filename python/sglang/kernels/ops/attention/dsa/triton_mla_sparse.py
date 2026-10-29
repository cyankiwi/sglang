# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 sparse MLA attention, adapted from vLLM's Triton sparse MLA kernel.

Source: vllm/v1/attention/ops/triton_mla_sparse.py (vLLM commit 5d84ab20ec).
The online softmax, BF16 probability dot, and FP32 split/LSE merge follow the
Apache-2.0 donor. SGLang adds validation, safe empty-tile handling, and a narrow
GB10 decode split policy. Indices are already physical KV slots; do not remap.

Launch configurations are static, not runtime-autotuned: no first-use benchmark
can run during CUDA graph capture, and prefill cannot seed a decode tuning cache.
Warm each shape/stride/dtype/split specialization before capture as with other
Triton kernels. The fixed four-warp/four-stage donor configurations were measured
on SM121: final uses <=54,016 B shared memory, split <=89,600 B, both below its
99 KiB per-block limit, including the 576-wide RoPE path (Triton 3.7.1).
"""

import functools
import math
from numbers import Real

import torch
import triton
import triton.language as tl

LOG2E = math.log2(math.e)
LOGE2 = math.log(2.0)

# DeepSeek-V3.2 / GLM-5 sparse MLA shape constants.
_BLOCK_DMODEL = 512
_BLOCK_DPE = 64
_BLOCK_DV = 512
_DIM_QK = _BLOCK_DMODEL + _BLOCK_DPE  # 576

_BLOCK_H = 16
# The backend pads the complete selection table (including KPool live tails).
_MIN_BLOCK_N = 16

# Merge kernel grid is spread across heads and DV tiles to avoid a (1,1)
# launch starving the SMs (pattern from FlashMLA's combine kernel).
_MERGE_BLOCK_H = 1
_MERGE_BLOCK_DV_TILE = 128
assert _BLOCK_DV % _MERGE_BLOCK_DV_TILE == 0
_NUM_MERGE_DV_TILES = _BLOCK_DV // _MERGE_BLOCK_DV_TILE

# Fixed donor configurations, qualified for resources and timing on SM121.
# Four stages beat two in the local NoPE H32 T1/T8/T128 sweep without spills.
# Separate final/split tiles retain the donor's numerical operation order.
_FINAL_BLOCK_N = 16
_SPLIT_BLOCK_N = 32
KV_SPLITS_CANDIDATES = (1, 2, 4, 8, 16)

_MIN_TOPK_PER_SPLIT = 128  # below this, per-split work is too small to amortize
_SPLIT_MAX_OCCUPANCY = 4  # skip split when baseline grid fills >=1/4 of SMs


@triton.jit
def _sparse_mla_compute_tile(
    q_buffer,
    k_buffer,  # V is the first BLOCK_DV lanes of each row of k_buffer.
    indices_ptr,
    cur_q,
    cur_head,
    cur_kv_head_id,
    mask_h,
    split_start,
    split_end,
    seq_kv,
    stride_q_token,
    stride_q_head,
    stride_kv_token,
    stride_kv_head,
    stride_indices_token,
    stride_indices_head,
    sm_scale,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
):
    """Shared stage-1 body: load Q, run the sparse online-softmax loop over
    `[split_start, split_end)` of the topk axis, return accumulators."""
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)

    q = tl.load(
        q_buffer
        + cur_q * stride_q_token
        + cur_head[:, None] * stride_q_head
        + offs_d[None, :],
        mask=mask_h[:, None],
        other=0.0,
    )
    # NoPE MLA (glm5_next: qk_rope_head_dim=0) has no rope lanes; skip the
    # pe load/dot entirely. BLOCK_DPE is constexpr so this prunes at compile.
    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < BLOCK_DMODEL + BLOCK_DPE
        qpe = tl.load(
            q_buffer
            + cur_q * stride_q_token
            + cur_head[:, None] * stride_q_head
            + offs_dpe[None, :],
            mask=(mask_h[:, None]) & (mask_dpe[None, :]),
            other=0.0,
        )

    # Use -inf, but guard empty-tile subtraction below. A finite sentinel can
    # give padding softmax mass, or dominate real logits smaller than -1e30.
    e_max = tl.full([BLOCK_H], -float("inf"), dtype=tl.float32)
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    for start_indice in range(split_start, split_end, BLOCK_N):
        offs_indice = start_indice + tl.arange(0, BLOCK_N)
        mask_indice = offs_indice < split_end
        indices = tl.load(
            indices_ptr
            + cur_q * stride_indices_token
            + cur_kv_head_id * stride_indices_head
            + offs_indice,
            mask=mask_indice,
            other=-1,
        )
        mask_kv = mask_indice & (indices >= 0) & (indices < seq_kv)
        # Physical cache row offsets can exceed int32 even with int32 indices.
        # Also avoid forming arbitrary OOB pointers for masked int64 indices.
        indices = tl.where(mask_kv, indices, 0).to(tl.int64)

        offs_k = (
            indices[None, :] * stride_kv_token
            + cur_kv_head_id * stride_kv_head
            + offs_d[:, None]
        )
        k = tl.load(k_buffer + offs_k, mask=mask_kv[None, :], other=0.0)
        qk = tl.dot(q, k.to(q.dtype))

        if BLOCK_DPE > 0:
            offs_kpe = (
                indices[None, :] * stride_kv_token
                + cur_kv_head_id * stride_kv_head
                + offs_dpe[:, None]
            )
            kpe = tl.load(
                k_buffer + offs_kpe,
                mask=(mask_kv[None, :]) & (mask_dpe[:, None]),
                other=0.0,
            )
            qk += tl.dot(qpe, kpe.to(q.dtype))

        qk *= sm_scale
        qk = tl.where(mask_h[:, None] & mask_kv[None, :], qk, -float("inf"))

        offs_v = (
            indices[:, None] * stride_kv_token
            + cur_kv_head_id * stride_kv_head
            + offs_dv[None, :]
        )
        v = tl.load(k_buffer + offs_v, mask=mask_kv[:, None], other=0.0)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        safe_max = tl.where(n_e_max == -float("inf"), 0.0, n_e_max)
        re_scale = tl.exp2(e_max - safe_max)
        p = tl.exp2(qk - safe_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    return acc, e_max, e_sum


@triton.jit
def _sparse_mla_kernel_final(
    q_buffer,
    k_buffer,
    indices_ptr,
    out_ptr,
    seq_kv,
    h_q,
    stride_q_token,
    stride_q_head,
    stride_kv_token,
    stride_kv_head,
    stride_out_token,
    stride_out_head,
    stride_indices_token,
    stride_indices_head,
    sm_scale,
    index_topk: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
):
    """Single-pass fast path: full topk, write final bf16 output directly."""
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head_id = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    acc, e_max, e_sum = _sparse_mla_compute_tile(
        q_buffer,
        k_buffer,
        indices_ptr,
        cur_q,
        cur_head,
        cur_kv_head_id,
        mask_h,
        0,
        index_topk,
        seq_kv,
        stride_q_token,
        stride_q_head,
        stride_kv_token,
        stride_kv_head,
        stride_indices_token,
        stride_indices_head,
        sm_scale,
        BLOCK_H,
        BLOCK_N,
        BLOCK_DV,
        BLOCK_DMODEL,
        BLOCK_DPE,
    )

    # Guard against queries with zero valid KV (e_sum == 0 → NaN from 0/0).
    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    offs_dv = tl.arange(0, BLOCK_DV)
    tl.store(
        out_ptr
        + cur_q * stride_out_token
        + cur_head[:, None] * stride_out_head
        + offs_dv[None, :],
        (acc / e_sum_safe[:, None]).to(tl.bfloat16),
        mask=mask_h[:, None],
    )


@triton.jit
def _sparse_mla_kernel_split(
    q_buffer,
    k_buffer,
    indices_ptr,
    mid_out_ptr,
    seq_kv,
    h_q,
    stride_q_token,
    stride_q_head,
    stride_kv_token,
    stride_kv_head,
    stride_mid_token,
    stride_mid_head,
    stride_mid_split,
    stride_indices_token,
    stride_indices_head,
    sm_scale,
    index_topk: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    LOGE2: tl.constexpr,
):
    """Stage 1 of split-KV: process one slice of the topk axis and write
    its `(out_partial, lse_partial)` into the mid buffer."""
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)
    cur_kv_head_id = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    split_topk: tl.constexpr = tl.cdiv(index_topk, NUM_KV_SPLITS)
    split_start = split_kv_id * split_topk
    split_end = tl.minimum(split_start + split_topk, index_topk)

    acc, e_max, e_sum = _sparse_mla_compute_tile(
        q_buffer,
        k_buffer,
        indices_ptr,
        cur_q,
        cur_head,
        cur_kv_head_id,
        mask_h,
        split_start,
        split_end,
        seq_kv,
        stride_q_token,
        stride_q_head,
        stride_kv_token,
        stride_kv_head,
        stride_indices_token,
        stride_indices_head,
        sm_scale,
        BLOCK_H,
        BLOCK_N,
        BLOCK_DV,
        BLOCK_DMODEL,
        BLOCK_DPE,
    )

    # Partial output and natural-log LSE for stage-2 merge.
    # When a split has no valid KV (`e_sum == 0`), guard the divide so the
    # mid buffer holds 0 instead of NaN; otherwise the `0 * NaN = NaN` term
    # in stage 2 would poison every other split.
    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    offs_dv = tl.arange(0, BLOCK_DV)
    mid_base_2d = (
        mid_out_ptr
        + cur_q * stride_mid_token
        + cur_head[:, None] * stride_mid_head
        + split_kv_id * stride_mid_split
    )
    tl.store(
        mid_base_2d + offs_dv[None, :],
        acc / e_sum_safe[:, None],
        mask=mask_h[:, None],
    )
    mid_lse_ptr = (
        mid_out_ptr
        + cur_q * stride_mid_token
        + cur_head * stride_mid_head
        + split_kv_id * stride_mid_split
        + BLOCK_DV
    )
    tl.store(mid_lse_ptr, (e_max + tl.log2(e_sum)) * LOGE2, mask=mask_h)


@triton.jit
def _sparse_mla_merge_kernel(
    mid_out_ptr,
    out_ptr,
    h_q,
    stride_mid_token,
    stride_mid_head,
    stride_mid_split,
    stride_out_token,
    stride_out_head,
    NUM_KV_SPLITS: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DV_TILE: tl.constexpr,
):
    """Stage 2: N-way online-softmax merge of per-split `(out, lse)` tiles.

    Grid is `(num_tokens, num_head_groups, num_dv_tiles)`. Each program handles
    `BLOCK_H` heads × `BLOCK_DV_TILE` output-dim lanes. The LSE reduction is
    identical across DV tiles for the same (token, head) — each program
    recomputes it locally, which is cheap (O(NUM_KV_SPLITS) scalars) and
    avoids inter-CTA synchronization.
    """
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_dv_tile = tl.program_id(2)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    offs_dv = cur_dv_tile * BLOCK_DV_TILE + tl.arange(0, BLOCK_DV_TILE)
    mask_dv = offs_dv < BLOCK_DV
    e_max = tl.full([BLOCK_H], -float("inf"), dtype=tl.float32)
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV_TILE], dtype=tl.float32)

    mid_base_2d = (
        mid_out_ptr + cur_q * stride_mid_token + cur_head[:, None] * stride_mid_head
    )
    mid_lse_1d = (
        mid_out_ptr + cur_q * stride_mid_token + cur_head * stride_mid_head + BLOCK_DV
    )

    for split_kv_id in range(NUM_KV_SPLITS):
        tv = tl.load(
            mid_base_2d + split_kv_id * stride_mid_split + offs_dv[None, :],
            mask=mask_h[:, None] & mask_dv[None, :],
            other=0.0,
        )
        tlogic = tl.load(
            mid_lse_1d + split_kv_id * stride_mid_split,
            mask=mask_h,
            other=-float("inf"),
        )
        n_e_max = tl.maximum(tlogic, e_max)
        safe_max = tl.where(n_e_max == -float("inf"), 0.0, n_e_max)
        old_scale = tl.exp(e_max - safe_max)
        exp_logic = tl.exp(tlogic - safe_max)
        acc = acc * old_scale[:, None] + exp_logic[:, None] * tv
        e_sum = e_sum * old_scale + exp_logic
        e_max = n_e_max

    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    tl.store(
        out_ptr
        + cur_q * stride_out_token
        + cur_head[:, None] * stride_out_head
        + offs_dv[None, :],
        (acc / e_sum_safe[:, None]).to(tl.bfloat16),
        mask=mask_h[:, None] & mask_dv[None, :],
    )


@functools.lru_cache(maxsize=256)
def _choose_num_kv_splits(
    num_tokens: int,
    num_heads: int,
    dim_qk: int,
    index_topk: int,
    sm_count: int,
    capability: tuple[int, int],
) -> int:
    """Donor occupancy policy, with a narrow measured SM121/48-SM exception.

    NoPE H32, K2048/2064: T1 split16 measured ~18 us; T8 split8 ~74 us
    versus split1 ~165 us. Interpolate only T2..7, not larger batches, RoPE,
    other head counts, K widths or devices. T128 remains single-pass (and
    TileLang is faster there). All workload/device fields are in this cache key.
    """
    if (
        capability == (12, 1)
        and sm_count == 48
        and num_heads == 32
        and dim_qk == 512
        and index_topk in (2048, 2064)
        and 1 <= num_tokens <= 8
    ):
        return 16 if num_tokens == 1 else 8

    num_head_groups = triton.cdiv(num_heads, _BLOCK_H)
    baseline = num_tokens * num_head_groups
    if baseline == 0 or baseline * _SPLIT_MAX_OCCUPANCY >= sm_count:
        return 1
    ideal = triton.next_power_of_2(max(1, index_topk // _MIN_TOPK_PER_SPLIT))
    max_splits = max(1, sm_count // baseline)
    max_splits = 1 << (max_splits.bit_length() - 1)  # floor to power of 2
    num_kv_splits = min(ideal, max_splits, max(KV_SPLITS_CANDIDATES))
    while num_kv_splits > 1 and index_topk % num_kv_splits != 0:
        num_kv_splits //= 2
    return max(1, num_kv_splits)


@functools.lru_cache(maxsize=None)
def _device_info(device_index: int) -> tuple[int, tuple[int, int]]:
    # Metadata only; no GPU-to-CPU tensor transfers or synchronization. Warmup
    # populates this per-device cache before graph capture.
    props = torch.cuda.get_device_properties(device_index)
    return props.multi_processor_count, (props.major, props.minor)


def _validate_inputs(q, kv, indices, sm_scale, num_kv_splits, sm_count):
    for name, tensor in (("q", q), ("kv", kv), ("indices", indices)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.layout != torch.strided or tensor.ndim != 3:
            raise ValueError(f"{name} must be a strided rank-3 tensor")
        if not tensor.is_cuda or torch.version.hip is not None:
            raise ValueError(f"{name} must be on an NVIDIA CUDA device")
        if tensor.device != q.device:
            raise ValueError("q, kv and indices must be on the same CUDA device")
        if tensor.stride(-1) != 1 or any(s < 0 for s in tensor.stride()):
            raise ValueError(
                f"{name} must have last-dimension stride 1 and nonnegative strides"
            )
        if tensor.requires_grad:
            raise ValueError(f"{name} must not require gradients (inference only)")
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise TypeError("q and kv must have dtype torch.bfloat16")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("indices must have dtype torch.int32 or torch.int64")
    if q.shape[1] < 1 or q.shape[2] not in (_BLOCK_DMODEL, _DIM_QK):
        raise ValueError("q must have shape [T, H, 512 or 576] with H > 0")
    if kv.shape[1:] != (1, q.shape[2]):
        raise ValueError("kv must have shape [slots, 1, q.shape[2]]")
    if indices.shape[:2] != (q.shape[0], 1):
        raise ValueError("indices must have shape [T, 1, K], matching q's T")
    if indices.shape[2] % _MIN_BLOCK_N:
        raise ValueError("indices K must be a multiple of 16; pad with -1")
    if isinstance(sm_scale, bool) or not isinstance(sm_scale, Real):
        raise TypeError("sm_scale must be a finite host real scalar, not a tensor")
    if not math.isfinite(sm_scale):
        raise ValueError("sm_scale must be finite")
    if num_kv_splits is not None and (
        type(num_kv_splits) is not int or num_kv_splits not in KV_SPLITS_CANDIDATES
    ):
        raise ValueError("num_kv_splits must be None or one of 1, 2, 4, 8, 16")
    if sm_count is not None and (type(sm_count) is not int or sm_count <= 0):
        raise ValueError("sm_count must be None or a positive integer")


def triton_mla_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    num_kv_splits: int | None = None,
    sm_count: int | None = None,
) -> torch.Tensor:
    """Inference-only sparse MLA over the entire physical selection table.

    Args:
        q: [T, H, D] BF16, D=512 (NoPE) or 576 (512 latent + 64 RoPE).
        kv: [slots, 1, D] BF16; the first 512 lanes are also the values.
        indices: [T, 1, K] int32/int64, K a multiple of 16. Negative and
            out-of-range physical slots are ignored. Duplicate slots retain
            their multiplicity. KPool tail columns beyond 2048 are NOT trimmed.
        sm_scale: Finite host softmax scale (not recomputed from D).
        num_kv_splits: None for automatic choice, or one of 1/2/4/8/16.
        sm_count: Optional positive SM count for automatic split selection.

    All tensors must be on the same NVIDIA CUDA device with last-dimension
    stride 1; arbitrary nonnegative outer strides are supported. Returns BF16
    [T, H, 512]. Empty selections/cache and all-invalid rows return zeros; T=0
    returns an empty output without a kernel launch. Scratch is invocation-local
    (no global buffers), so independent streams and graph replays cannot alias.
    """
    _validate_inputs(q, kv, indices, sm_scale, num_kv_splits, sm_count)
    num_tokens, num_heads_q, dim_qk = q.shape
    block_dpe = dim_qk - _BLOCK_DMODEL
    index_topk = indices.shape[2]
    kv_group_num = num_heads_q
    num_head_groups = triton.cdiv(num_heads_q, _BLOCK_H)

    out = torch.empty(
        (num_tokens, num_heads_q, _BLOCK_DV),
        dtype=torch.bfloat16,
        device=q.device,
    )
    if num_tokens == 0:
        return out
    if index_topk == 0 or kv.shape[0] == 0:
        return out.zero_()

    if num_kv_splits is None:
        device_sms, capability = _device_info(q.device.index)
        num_kv_splits = _choose_num_kv_splits(
            num_tokens,
            num_heads_q,
            dim_qk,
            index_topk,
            device_sms if sm_count is None else sm_count,
            capability,
        )
    sm_scale = float(sm_scale)

    if num_kv_splits == 1:
        _sparse_mla_kernel_final[(num_tokens, num_head_groups)](
            q_buffer=q,
            k_buffer=kv,
            indices_ptr=indices,
            out_ptr=out,
            seq_kv=kv.shape[0],
            h_q=num_heads_q,
            stride_q_token=q.stride(0),
            stride_q_head=q.stride(1),
            stride_kv_token=kv.stride(0),
            stride_kv_head=kv.stride(1),
            stride_out_token=out.stride(0),
            stride_out_head=out.stride(1),
            stride_indices_token=indices.stride(0),
            stride_indices_head=indices.stride(1),
            sm_scale=sm_scale * LOG2E,
            index_topk=index_topk,
            kv_group_num=kv_group_num,
            BLOCK_H=_BLOCK_H,
            BLOCK_DV=_BLOCK_DV,
            BLOCK_DMODEL=_BLOCK_DMODEL,
            BLOCK_DPE=block_dpe,
            BLOCK_N=_FINAL_BLOCK_N,
            num_warps=4,
            num_stages=4,
        )
        return out

    # Split-KV: partial fp32 output + LSE per (token, head, split).
    mid_out = torch.empty(
        (num_tokens, num_heads_q, num_kv_splits, _BLOCK_DV + 1),
        dtype=torch.float32,
        device=q.device,
    )
    _sparse_mla_kernel_split[(num_tokens, num_head_groups, num_kv_splits)](
        q_buffer=q,
        k_buffer=kv,
        indices_ptr=indices,
        mid_out_ptr=mid_out,
        seq_kv=kv.shape[0],
        h_q=num_heads_q,
        stride_q_token=q.stride(0),
        stride_q_head=q.stride(1),
        stride_kv_token=kv.stride(0),
        stride_kv_head=kv.stride(1),
        stride_mid_token=mid_out.stride(0),
        stride_mid_head=mid_out.stride(1),
        stride_mid_split=mid_out.stride(2),
        stride_indices_token=indices.stride(0),
        stride_indices_head=indices.stride(1),
        sm_scale=sm_scale * LOG2E,
        index_topk=index_topk,
        NUM_KV_SPLITS=num_kv_splits,
        kv_group_num=kv_group_num,
        BLOCK_H=_BLOCK_H,
        BLOCK_DV=_BLOCK_DV,
        BLOCK_DMODEL=_BLOCK_DMODEL,
        # Derive from dim_qk (0 for NoPE / glm5_next, 64 for rope MLA) — must
        # match the single-pass launch above; hardcoding _BLOCK_DPE (64) here
        # reads 64 nonexistent rope lanes past the 512-wide NoPE row.
        BLOCK_DPE=block_dpe,
        LOGE2=LOGE2,
        BLOCK_N=_SPLIT_BLOCK_N,
        num_warps=4,
        num_stages=4,
    )

    _sparse_mla_merge_kernel[(num_tokens, num_heads_q, _NUM_MERGE_DV_TILES)](
        mid_out_ptr=mid_out,
        out_ptr=out,
        h_q=num_heads_q,
        stride_mid_token=mid_out.stride(0),
        stride_mid_head=mid_out.stride(1),
        stride_mid_split=mid_out.stride(2),
        stride_out_token=out.stride(0),
        stride_out_head=out.stride(1),
        NUM_KV_SPLITS=num_kv_splits,
        kv_group_num=kv_group_num,
        BLOCK_H=_MERGE_BLOCK_H,
        BLOCK_DV=_BLOCK_DV,
        BLOCK_DV_TILE=_MERGE_BLOCK_DV_TILE,
        num_warps=2,
    )
    return out
