# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gated MLA training attention (Kimi-K3 full-attention layers).

Dense attention over per-head keys and values expanded from a shared latent, with no
rotary on the attention path (NoPE) and an elementwise sigmoid output gate. Per token
``t`` and head ``h``, with ``d_nope = q.shape[-1] - k_rot.shape[-1]``::

    k_nope[t, h], v[t, h] = split((kv_latent[t] @ kv_b_weight.T)[h], [d_nope, d_v])
    k[t, h]               = cat(k_nope[t, h], k_rot[t])
    o[t, h]               = softmax(q[t, h] . k[:, h] * softmax_scale) @ v[:, h]
    out[t]                = o[t].flatten() * sigmoid(gate[t])

``out`` is the ``o_proj`` input. The q / kv down-projections, their RMSNorms,
``q_b_proj``, ``g_proj`` and ``o_proj`` stay with the caller.

The forward runs on the gfx950 OPUS D_QK=192 / D_V=128 kernel: it reads ``k_nope`` and
``v`` straight out of the ``kv_b_proj`` output and ``k_rot`` from its own buffer, so K is
never concatenated, and it applies the gate before the O store. Only ``kv_latent`` is kept
for the backward, which recomputes the expansion.

The backward is OPUS too. A pre-pass turns the gate into the attention's ``dO`` and the
gate gradient, and forms ``D = rowsum(dout * out)``, which is the ungated attention's
``rowsum(dO * O)`` without keeping the ungated output. The main pass reads K the way the
forward does and writes ``[dk_nope | dv]`` in the ``kv_b_proj`` output layout, so the two
``kv_b_proj`` GEMMs take it as is.
"""

import torch
from torch import Tensor

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx


def _gated_mla_fwd_bf16_opus_fake(
    q: Tensor,
    k_nope: Tensor,
    v: Tensor,
    k_rot: Tensor,
    gate: Tensor,
    out: Tensor,
    lse: Tensor | None,
    seqstart_q: Tensor,
    seqstart_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    softmax_scale: float,
) -> None:
    return None


# Writes `out` (and `lse`, when given) in place.
@compile_ops(
    "module_fmha_fwd_bf16_opus",
    fc_name="gated_mla_fwd_bf16_opus",
    gen_fake=_gated_mla_fwd_bf16_opus_fake,
)
def _gated_mla_fwd_bf16_opus(
    q: Tensor,
    k_nope: Tensor,
    v: Tensor,
    k_rot: Tensor,
    gate: Tensor,
    out: Tensor,
    lse: Tensor | None,
    seqstart_q: Tensor,
    seqstart_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    softmax_scale: float,
) -> None: ...


def _gated_mla_bwd_pre_bf16_opus_fake(
    dout: Tensor, out: Tensor, gate: Tensor, d_o: Tensor, dgate: Tensor, d: Tensor
) -> None:
    return None


# Writes `d_o`, `dgate` and `d` in place.
@compile_ops(
    "module_gated_mla_bwd_opus",
    fc_name="gated_mla_bwd_pre_bf16_opus",
    gen_fake=_gated_mla_bwd_pre_bf16_opus_fake,
)
def _gated_mla_bwd_pre_bf16_opus(
    dout: Tensor, out: Tensor, gate: Tensor, d_o: Tensor, dgate: Tensor, d: Tensor
) -> None: ...


def _gated_mla_bwd_dot_bf16_opus_fake(dout: Tensor, out: Tensor, d: Tensor) -> None:
    return None


# The pre-pass without a gate: writes `d` in place.
@compile_ops(
    "module_gated_mla_bwd_opus",
    fc_name="gated_mla_bwd_dot_bf16_opus",
    gen_fake=_gated_mla_bwd_dot_bf16_opus_fake,
)
def _gated_mla_bwd_dot_bf16_opus(dout: Tensor, out: Tensor, d: Tensor) -> None: ...


def _gated_mla_bwd_pre(
    dout: Tensor, out: Tensor, gate: Tensor | None
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Backward pre-pass on ``[T, H, d_v]`` rows: ``(dO, dgate, D)``.

    ``dO = dout * sigmoid(gate)`` and ``dgate = dout * out * (1 - sigmoid(gate))`` in
    bf16; ``D = rowsum(dout * out)`` ``[H, T]`` fp32, the ``rowsum(dO * O)`` of the
    ungated attention. Without a gate (``gate`` None, ``out`` the attention's output)
    ``dO`` is ``dout``, ``dgate`` None, and only ``D`` is computed.
    """
    t, h, _ = out.shape
    dout = _kernel_ready(dout.reshape(out.shape))
    d = torch.empty(h, t, dtype=torch.float32, device=out.device)
    if gate is None:
        _gated_mla_bwd_dot_bf16_opus(dout, _kernel_ready(out), d)
        return dout, None, d
    gate = _kernel_ready(gate.reshape(out.shape))
    d_o, dgate = torch.empty_like(out), torch.empty_like(out)
    _gated_mla_bwd_pre_bf16_opus(dout, _kernel_ready(out), gate, d_o, dgate, d)
    return d_o, dgate, d


def _gated_mla_bwd_bf16_opus_fake(
    q: Tensor,
    k_nope: Tensor,
    v: Tensor,
    k_rot: Tensor,
    d_o: Tensor,
    lse: Tensor,
    d: Tensor,
    dq_acc: Tensor,
    dkv: Tensor,
    dk_rope: Tensor,
    seqstart_q: Tensor,
    seqstart_k: Tensor,
    max_seqlen_k: int,
    causal: bool,
    softmax_scale: float,
) -> None:
    return None


# Adds into `dq_acc`, writes `dkv` and `dk_rope` in place.
@compile_ops(
    "module_gated_mla_bwd_opus",
    fc_name="gated_mla_bwd_bf16_opus",
    gen_fake=_gated_mla_bwd_bf16_opus_fake,
)
def _gated_mla_bwd_bf16_opus(
    q: Tensor,
    k_nope: Tensor,
    v: Tensor,
    k_rot: Tensor,
    d_o: Tensor,
    lse: Tensor,
    d: Tensor,
    dq_acc: Tensor,
    dkv: Tensor,
    dk_rope: Tensor,
    seqstart_q: Tensor,
    seqstart_k: Tensor,
    max_seqlen_k: int,
    causal: bool,
    softmax_scale: float,
) -> None: ...


def _gated_mla_bwd_dkdv_bf16_opus_fake(
    q: Tensor,
    k_nope: Tensor,
    v: Tensor,
    k_rot: Tensor,
    d_o: Tensor,
    lse: Tensor,
    d: Tensor,
    dkv: Tensor,
    dk_rope: Tensor,
    seqstart_q: Tensor,
    seqstart_k: Tensor,
    max_seqlen_k: int,
    causal: bool,
    softmax_scale: float,
) -> None:
    return None


# The main pass without dQ: writes `dkv` and `dk_rope` in place.
@compile_ops(
    "module_gated_mla_bwd_opus",
    fc_name="gated_mla_bwd_dkdv_bf16_opus",
    gen_fake=_gated_mla_bwd_dkdv_bf16_opus_fake,
)
def _gated_mla_bwd_dkdv_bf16_opus(
    q: Tensor,
    k_nope: Tensor,
    v: Tensor,
    k_rot: Tensor,
    d_o: Tensor,
    lse: Tensor,
    d: Tensor,
    dkv: Tensor,
    dk_rope: Tensor,
    seqstart_q: Tensor,
    seqstart_k: Tensor,
    max_seqlen_k: int,
    causal: bool,
    softmax_scale: float,
) -> None: ...


# As _gated_mla_bwd_dkdv_bf16_opus, on the 8-wave kernel whose wave pairs split by dependency chain.
@compile_ops(
    "module_gated_mla_bwd_opus",
    fc_name="gated_mla_bwd_dkdv_pair_bf16_opus",
    gen_fake=_gated_mla_bwd_dkdv_bf16_opus_fake,
)
def _gated_mla_bwd_dkdv_pair_bf16_opus(
    q: Tensor,
    k_nope: Tensor,
    v: Tensor,
    k_rot: Tensor,
    d_o: Tensor,
    lse: Tensor,
    d: Tensor,
    dkv: Tensor,
    dk_rope: Tensor,
    seqstart_q: Tensor,
    seqstart_k: Tensor,
    max_seqlen_k: int,
    causal: bool,
    softmax_scale: float,
) -> None: ...


def _gated_mla_bwd_dkdv_pair_ds_bf16_opus_fake(
    q: Tensor,
    k_nope: Tensor,
    v: Tensor,
    k_rot: Tensor,
    d_o: Tensor,
    lse: Tensor,
    d: Tensor,
    dkv: Tensor,
    dk_rope: Tensor,
    seqstart_q: Tensor,
    seqstart_k: Tensor,
    max_seqlen_k: int,
    causal: bool,
    softmax_scale: float,
    ds: Tensor,
    ds_seq: Tensor,
) -> None:
    return None


# As _gated_mla_bwd_dkdv_pair_bf16_opus, also writing dS to `ds` (bf16 tiles, see _ds_tiles_per_head)
# for _gated_mla_bwd_dq_ds_bf16_opus; `ds_seq` [num_seqs] int64 is each sequence's first tile.
@compile_ops(
    "module_gated_mla_bwd_opus",
    fc_name="gated_mla_bwd_dkdv_pair_ds_bf16_opus",
    gen_fake=_gated_mla_bwd_dkdv_pair_ds_bf16_opus_fake,
)
def _gated_mla_bwd_dkdv_pair_ds_bf16_opus(
    q: Tensor,
    k_nope: Tensor,
    v: Tensor,
    k_rot: Tensor,
    d_o: Tensor,
    lse: Tensor,
    d: Tensor,
    dkv: Tensor,
    dk_rope: Tensor,
    seqstart_q: Tensor,
    seqstart_k: Tensor,
    max_seqlen_k: int,
    causal: bool,
    softmax_scale: float,
    ds: Tensor,
    ds_seq: Tensor,
) -> None: ...


def _gated_mla_bwd_dq_ds_bf16_opus_fake(
    k_nope: Tensor,
    k_rot: Tensor,
    ds: Tensor,
    ds_seq: Tensor,
    dq: Tensor,
    seqstart_q: Tensor,
    seqstart_k: Tensor,
    max_seqlen_q: int,
    causal: bool,
    softmax_scale: float,
) -> None:
    return None


# dQ = dS K softmax_scale from the dS _gated_mla_bwd_dkdv_pair_ds_bf16_opus wrote: writes `dq` (bf16).
@compile_ops(
    "module_gated_mla_bwd_opus",
    fc_name="gated_mla_bwd_dq_ds_bf16_opus",
    gen_fake=_gated_mla_bwd_dq_ds_bf16_opus_fake,
)
def _gated_mla_bwd_dq_ds_bf16_opus(
    k_nope: Tensor,
    k_rot: Tensor,
    ds: Tensor,
    ds_seq: Tensor,
    dq: Tensor,
    seqstart_q: Tensor,
    seqstart_k: Tensor,
    max_seqlen_q: int,
    causal: bool,
    softmax_scale: float,
) -> None: ...


def _gated_mla_bwd_dq_bf16_opus_fake(
    q: Tensor,
    k_nope: Tensor,
    v: Tensor,
    k_rot: Tensor,
    d_o: Tensor,
    lse: Tensor,
    d: Tensor,
    dq: Tensor,
    seqstart_q: Tensor,
    seqstart_k: Tensor,
    max_seqlen_q: int,
    causal: bool,
    softmax_scale: float,
) -> None:
    return None


# The dQ pass: writes `dq` (bf16) in place, no atomics.
@compile_ops(
    "module_gated_mla_bwd_opus",
    fc_name="gated_mla_bwd_dq_bf16_opus",
    gen_fake=_gated_mla_bwd_dq_bf16_opus_fake,
)
def _gated_mla_bwd_dq_bf16_opus(
    q: Tensor,
    k_nope: Tensor,
    v: Tensor,
    k_rot: Tensor,
    d_o: Tensor,
    lse: Tensor,
    d: Tensor,
    dq: Tensor,
    seqstart_q: Tensor,
    seqstart_k: Tensor,
    max_seqlen_q: int,
    causal: bool,
    softmax_scale: float,
) -> None: ...


# How the backward computes dQ: "separate" runs a dQ pass that recomputes S and dP after the dK/dV
# pass (the pair kernel); "ds" has the dK/dV pass also write dS, from which a GEMM makes dQ, where a
# launch's dS buffer can hold enough heads (_ds_head_group), and falls back to "separate"
# elsewhere; "atomic" adds dQ into an fp32 buffer from the dK/dV pass.
# "ds" trades the recomputation for memory traffic (dS written and read back, 2 bytes per query-key
# pair and head each way): 4 to 6% faster than "separate" on an MI350X from 2K to 32K tokens, slower
# on small problems, and up to _BWD_DS_BUDGET more memory.
_BWD_DQ_MODE = "separate"
# Most bytes the "ds" mode's dS buffer may take; heads go in groups that fit.
_BWD_DS_BUDGET = 16 << 30
_DS_TILE_BYTES = 32 * 32 * 2


def _clamp_sum(n: int, t: int) -> int:
    """sum(clamp(u, 0, t) for u in range(n))."""
    if n <= 0:
        return 0
    if n <= t + 1:
        return n * (n - 1) // 2
    return t * (t + 1) // 2 + (n - t - 1) * t


def _ds_tiles_per_head(seqlen_q: int, seqlen_k: int, causal: bool) -> int:
    """32 x 32 dS tiles one head of a sequence keeps (gated_mla_bwd_ds in gated_mla_bwd_opus_defs.h):
    per key tile, its query tiles from the first one causal masking (bottom-right) lets see it."""
    t, nk = (seqlen_q + 31) // 32, (seqlen_k + 31) // 32
    if not causal:
        return nk * t
    s = (seqlen_q - seqlen_k) // 32
    return nk * t - (_clamp_sum(nk + s, t) - _clamp_sum(s, t))


def _ds_head_group(h: int, tiles_per_head: int) -> int:
    """Heads per launch of the "ds" mode, such that a launch's dS fits _BWD_DS_BUDGET, or 0 where
    recomputing dQ is the faster. Workgroups reach the 8 XCDs round-robin and the kernels deal a
    launch's heads over them in turn, so a launch holds a multiple of 8 heads (16 if they fit),
    each XCD then keeping to one or two heads at a time, the heads spread evenly over the fewest
    such launches; or 4 heads. Against recomputing dQ (96 heads, 32K and 64K): launches of 8 to 48
    heads 3 to 4% faster, of 4 heads 1 to 2% faster, of 14, 3 or 2 heads up to 2% slower."""
    most = min(h, _BWD_DS_BUDGET // (max(1, tiles_per_head) * _DS_TILE_BYTES))
    if most >= h:
        return h
    if most < 4:
        return 0
    if most < 8:
        return 4
    unit = 16 if most >= 16 else 8
    launches = -(-h // (most // unit * unit))
    return -(-h // launches // unit) * unit


def _gated_mla_bwd(
    q: Tensor,
    kv: Tensor,
    k_rot: Tensor,
    gate: Tensor | None,
    out: Tensor,
    lse: Tensor,
    dout: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
    softmax_scale: float,
    causal: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    """Attention and gate gradients on the expanded ``kv`` ``[T, H, d_nope + d_v]``.

    Returns ``dq [T, H, 192]``, ``dkv [T, H, 256]`` (each head's row ``[dk_nope | dv]``,
    the ``kv_b_proj`` output layout), ``dk_rot [T, 64]`` and ``dgate [T, H, 128]``.
    ``gate`` None is the backward of the attention alone, ``out`` its ungated output and
    ``dgate`` None.
    """
    t, h, _ = q.shape
    d_nope = q.shape[-1] - k_rot.shape[-1]
    d_o, dgate, d = _gated_mla_bwd_pre(dout, out, gate)
    q_k, k_rot_k = _kernel_ready(q), _kernel_ready(k_rot)
    k_nope, v = kv[..., :d_nope], kv[..., d_nope:]
    dkv = torch.empty(t, h, kv.shape[-1], dtype=q.dtype, device=q.device)
    dk_rope = torch.empty(t, h, k_rot.shape[-1], dtype=torch.float32, device=q.device)
    args = (int(max_seqlen), bool(causal), float(softmax_scale))
    if _BWD_DQ_MODE == "atomic":
        dq_acc = torch.zeros(t, h, q.shape[-1], dtype=torch.float32, device=q.device)
        _gated_mla_bwd_bf16_opus(
            q_k, k_nope, v, k_rot_k, d_o, lse, d, dq_acc, dkv, dk_rope, cu_seqlens, cu_seqlens, *args
        )
        dq = dq_acc.to(q.dtype)
        return dq, dkv, dk_rope.sum(1).to(k_rot.dtype), dgate
    dq = torch.empty(t, h, q.shape[-1], dtype=q.dtype, device=q.device)
    group = 0
    if _BWD_DQ_MODE == "ds":
        lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        tiles = [_ds_tiles_per_head(n, n, bool(causal)) for n in lens]
        per_head = max(1, sum(tiles))
        group = _ds_head_group(h, per_head)
    if group:
        ds = torch.empty(group * per_head * _DS_TILE_BYTES // 2, dtype=q.dtype, device=q.device)
        ds_seq = None
        for h0 in range(0, h, group):
            hs = slice(h0, min(h, h0 + group))
            n_h = hs.stop - h0
            if ds_seq is None or n_h != group:
                firsts = [0]
                for n in tiles[:-1]:
                    firsts.append(firsts[-1] + n_h * n)
                ds_seq = torch.tensor(firsts, dtype=torch.int64, device=q.device)
            _gated_mla_bwd_dkdv_pair_ds_bf16_opus(
                q_k[:, hs], k_nope[:, hs], v[:, hs], k_rot_k, d_o[:, hs], lse[hs], d[hs],
                dkv[:, hs], dk_rope[:, hs], cu_seqlens, cu_seqlens, *args, ds, ds_seq,
            )
            _gated_mla_bwd_dq_ds_bf16_opus(
                k_nope[:, hs], k_rot_k, ds, ds_seq, dq[:, hs], cu_seqlens, cu_seqlens, *args
            )
    else:
        _gated_mla_bwd_dkdv_pair_bf16_opus(
            q_k, k_nope, v, k_rot_k, d_o, lse, d, dkv, dk_rope, cu_seqlens, cu_seqlens, *args
        )
        _gated_mla_bwd_dq_bf16_opus(
            q_k, k_nope, v, k_rot_k, d_o, lse, d, dq, cu_seqlens, cu_seqlens, *args
        )
    return dq, dkv, dk_rope.sum(1).to(k_rot.dtype), dgate


def _kernel_ready(x: Tensor) -> Tensor:
    """`x` if the kernel's 16-byte row loads can read it as is, else a contiguous copy."""
    aligned = x.stride(-1) == 1 and x.data_ptr() % 16 == 0
    aligned = aligned and all(s % 8 == 0 for s in x.stride()[:-1])
    return x if aligned else x.contiguous()


def _gated_mla_fwd(
    q: Tensor,
    kv: Tensor,
    k_rot: Tensor,
    gate: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
    softmax_scale: float,
    causal: bool,
    return_lse: bool,
) -> tuple[Tensor, Tensor | None]:
    """Fused attention + gate on the expanded ``kv`` ``[T, H, d_nope + d_v]``.

    Returns ``out [T, H, d_v]`` and, when ``return_lse``, the ungated attention's
    log-sum-exp ``[H, T]`` (fp32, natural log).
    """
    t, h, d_qk = q.shape
    d_nope = d_qk - k_rot.shape[-1]
    d_v = kv.shape[-1] - d_nope
    out = torch.empty(t, h, d_v, dtype=q.dtype, device=q.device)
    lse = (
        torch.empty(h, t, dtype=torch.float32, device=q.device) if return_lse else None
    )
    _gated_mla_fwd_bf16_opus(
        _kernel_ready(q),
        kv[..., :d_nope],
        kv[..., d_nope:],
        _kernel_ready(k_rot),
        _kernel_ready(gate.reshape(t, h, d_v)),
        out,
        lse,
        cu_seqlens,
        cu_seqlens,
        int(max_seqlen),
        int(max_seqlen),
        bool(causal),
        float(softmax_scale),
    )
    return out, lse


class _GatedMLA(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        kv_latent,
        k_rot,
        kv_b_weight,
        gate,
        cu_seqlens,
        max_seqlen,
        scale,
        causal,
    ):
        t, h, _ = q.shape
        cu_seqlens = cu_seqlens.to(torch.int32).contiguous()
        kv = torch.mm(kv_latent, kv_b_weight.t()).view(t, h, -1)
        out, lse = _gated_mla_fwd(
            q,
            kv,
            k_rot,
            gate,
            cu_seqlens,
            max_seqlen,
            scale,
            causal,
            return_lse=any(ctx.needs_input_grad),
        )
        ctx.save_for_backward(
            q, kv_latent, k_rot, kv_b_weight, gate, out, lse, cu_seqlens
        )
        ctx.max_seqlen, ctx.scale, ctx.causal = max_seqlen, scale, causal
        return out.view(t, -1)

    @staticmethod
    def backward(ctx, dout):
        q, kv_latent, k_rot, kv_b_weight, gate, out, lse, cu_seqlens = ctx.saved_tensors
        t, h, _ = q.shape
        kv = torch.mm(kv_latent, kv_b_weight.t()).view(t, h, -1)
        dq, dkv, dk_rot, dgate = _gated_mla_bwd(
            q,
            kv,
            k_rot,
            gate,
            out,
            lse,
            dout,
            cu_seqlens,
            ctx.max_seqlen,
            ctx.scale,
            ctx.causal,
        )
        dkv = dkv.view(t, -1)
        d_latent = torch.mm(dkv, kv_b_weight)
        d_weight = torch.mm(dkv.t(), kv_latent)
        return (
            dq,
            d_latent,
            dk_rot,
            d_weight,
            dgate.view_as(gate),
            None,
            None,
            None,
            None,
        )


def gated_mla_train(
    q: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rot: torch.Tensor,
    kv_b_weight: torch.Tensor,
    gate: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    softmax_scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Gated MLA attention forward; differentiable in every tensor argument.

    q           : [T, H, d_nope + d_rot], the ``q_b_proj`` output.
    kv_latent   : [T, d_latent], the ``kv_a_layernorm`` output.
    k_rot       : [T, d_rot], the companion key from ``kv_a_proj_with_mqa``, shared
                  by all heads.
    kv_b_weight : [H * (d_nope + d_v), d_latent], ``kv_b_proj.weight``; each head's
                  rows are ``[k_nope | v]``.
    gate        : [T, H * d_v], the ``g_proj`` output before the sigmoid.
    cu_seqlens  : [B + 1] int32 packed-sequence boundaries, shared by queries and keys.
    max_seqlen  : longest sequence in the batch.
    softmax_scale : defaults to ``(d_nope + d_rot) ** -0.5``.

    BF16 on gfx950, with (d_nope, d_rot, d_v) = (128, 64, 128). Returns
    ``[T, H * d_v]`` in ``q.dtype``.
    """
    t, h, d_qk = q.shape
    d_rot = k_rot.shape[-1]
    d_nope = d_qk - d_rot
    if kv_b_weight.shape[0] % h:
        raise ValueError(
            f"kv_b_weight rows {kv_b_weight.shape[0]} are not a multiple of H={h}"
        )
    d_v = kv_b_weight.shape[0] // h - d_nope
    if kv_latent.shape != (t, kv_b_weight.shape[1]):
        raise ValueError(
            f"kv_latent shape {tuple(kv_latent.shape)} != {(t, kv_b_weight.shape[1])}"
        )
    if k_rot.shape != (t, d_rot):
        raise ValueError(f"k_rot shape {tuple(k_rot.shape)} != {(t, d_rot)}")
    if gate.shape != (t, h * d_v):
        raise ValueError(f"gate shape {tuple(gate.shape)} != {(t, h * d_v)}")
    if (d_nope, d_rot, d_v) != (128, 64, 128):
        raise ValueError(
            f"(d_nope, d_rot, d_v) = {(d_nope, d_rot, d_v)}, only (128, 64, 128) is supported"
        )
    if get_gfx() != "gfx950":
        raise NotImplementedError(f"gated_mla_train needs gfx950, got {get_gfx()}")
    if softmax_scale is None:
        softmax_scale = d_qk**-0.5
    return _GatedMLA.apply(
        q,
        kv_latent,
        k_rot,
        kv_b_weight,
        gate,
        cu_seqlens,
        max_seqlen,
        softmax_scale,
        causal,
    )
