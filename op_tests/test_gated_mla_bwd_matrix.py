# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Backward matrix of the gated MLA attention kernels (`_gated_mla_bwd`, Kimi-K3 gated MLA).

Every case runs the backward (gate pre-pass, dK / dV pass, dQ) on the fused forward's
bf16 output and LSE in each dQ mode, "separate" and "ds", and compares dq, d(kv) as
[dk_nope | dv], dk_rot and dgate with fp32 autograd (`_ref_bwd` of test_gated_mla_train): the
relative L2 error must stay within PARITY_REL_L2. That is the requirement on either mode; the
two need not agree bit for bit ("separate" starts dP's accumulators at -D by default, "ds"
subtracts D, which differ in the order of fp32 sums).

The cases cover sequence lengths from one token, around the kernels' tile and block edges,
to 65536 tokens and batches of 131072 at the K3 head count, equal-length batches and varlen
mixes, head counts around the scheduling groups of 16 heads that the kernels deal over the
8 XCDs, causal and unmasked attention, "ds" launches of 16, 8 and 4 heads and its fall back
to "separate", strided and unaligned inputs, KV heads shared by several query heads (GQA /
MQA), run-to-run determinism and the module-level op.
"""

import itertools
import os
import sys

import pytest
import torch

import aiter.ops.gated_mla_train as gmt

try:
    from op_tests import test_gated_mla_train as T
except ImportError:  # run from op_tests itself
    import test_gated_mla_train as T

pytestmark = pytest.mark.skipif(not T._is_gfx950(), reason="gfx950 only")

_OUTPUTS = ("dq", "dkv", "dk_rot", "dgate")
# The dQ mode `_fwd_bwd` runs, set by the `dq_mode` fixture.
_DQ_MODE = "separate"
# Largest relative L2 error of each output over the module's cases, with its case.
_worst = {}


@pytest.fixture(autouse=True)
def _module_globals():
    """Restores the dQ mode, the dS budget and the helpers' head count after each test."""
    saved = gmt._BWD_DQ_MODE, gmt._BWD_DS_BUDGET, T.HEADS
    yield
    gmt._BWD_DQ_MODE, gmt._BWD_DS_BUDGET, T.HEADS = saved


@pytest.fixture(params=["separate", "ds"])
def dq_mode(request):
    """Runs a test once per dQ mode: `_fwd_bwd` computes the gradients in it, and the test holds
    them to fp32 autograd."""
    global _DQ_MODE
    _DQ_MODE = request.param
    yield request.param
    _DQ_MODE = "separate"


@pytest.fixture(autouse=True, scope="module")
def _error_summary():
    yield
    print(f"\naiter: {os.path.dirname(gmt.__file__)}")
    print("largest relative L2 errors against fp32:")
    for name, (err, case) in _worst.items():
        print(f"  {name:22s} {err:.3e}  {case}")


def _case_ids(cases):
    """b<sequences>_s<length>_h<heads>, or the shortest and longest length of a varlen mix."""
    ids = []
    for seqlens, heads in cases:
        lo, hi = min(seqlens), max(seqlens)
        span = f"s{hi}" if lo == hi else f"min{lo}_max{hi}"
        ids.append(f"b{len(seqlens)}_{span}_h{heads}")
    return ids


def _inputs(seqlens, heads, seed=0):
    """`T._fwd_inputs` at `heads` heads (the helpers of T read T.HEADS) and a random dout."""
    T.HEADS = heads
    cu, smax, q, kv, k_rot, gate = T._fwd_inputs(seqlens, seed)
    dout = torch.randn(q.shape[0], heads * T.D_V, device="cuda", dtype=q.dtype)
    return cu, smax, q, kv, k_rot, gate, dout


def _bwd(mode, q, kv, k_rot, gate, out, lse, dout, cu, smax, causal):
    gmt._BWD_DQ_MODE = mode
    return gmt._gated_mla_bwd(
        q, kv, k_rot, gate, out, lse, dout, cu, smax, T.SCALE, causal
    )


def _assert_equal(a, b, what):
    differ = [n for n, x, y in zip(_OUTPUTS, a, b) if not torch.equal(x, y)]
    assert not differ, f"{what}: {differ} differ"


def _fwd_bwd(q, kv, k_rot, gate, dout, cu, smax, causal):
    """Fused forward, then the backward in the test's dQ mode (the `dq_mode` fixture)."""
    out, lse = gmt._gated_mla_fwd(q, kv, k_rot, gate, cu, smax, T.SCALE, causal, True)
    return _bwd(_DQ_MODE, q, kv, k_rot, gate, out, lse, dout, cu, smax, causal)


def _assert_parity(ours, ref, names=_OUTPUTS, scales=None):
    """Relative L2 error of each of `ours` against `ref` within PARITY_REL_L2; `scales`
    replaces the reference norms as the errors' denominators."""
    case = os.environ.get("PYTEST_CURRENT_TEST", "").split("::")[-1].split(" ")[0]
    failed = []
    for i, (name, a, r) in enumerate(zip(names, ours, ref)):
        if scales is None:
            err = T._rel_err(a.reshape(r.shape), r)
        else:
            err = ((a.reshape(r.shape).float() - r.float()).norm() / scales[i]).item()
        print(f"  {name:22s} rel_err {err:.3e}")
        if err > _worst.get(name, (0.0, ""))[0]:
            _worst[name] = (err, case)
        # Not `err > bound`, which a NaN would pass.
        if not err <= T.PARITY_REL_L2:
            failed.append(name)
    assert not failed, f"relative L2 error above {T.PARITY_REL_L2}: {failed}"


def _check(seqlens, heads, causal):
    cu, smax, q, kv, k_rot, gate, dout = _inputs(seqlens, heads)
    ours = _fwd_bwd(q, kv, k_rot, gate, dout, cu, smax, causal)
    _assert_parity(ours, T._ref_bwd(q, kv, k_rot, gate, dout, cu, causal))


# Around the 32-row tiles of every pass; the 48 rows of a wave and the 192-row blocks of the 4-wave
# dK / dV and dQ passes ("separate"); the pair pass's 128-key blocks and the dS GEMM's 256-query
# blocks ("ds").
_EDGE_LENS = [
    31,
    32,
    33,
    47,
    48,
    49,
    63,
    64,
    65,
    95,
    96,
    97,
    127,
    128,
    129,
    191,
    192,
    193,
    255,
    256,
    257,
    383,
    385,
    511,
    513,
]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("heads", [1, 9, 33])
@pytest.mark.parametrize("seqlen", _EDGE_LENS)
@pytest.mark.usefixtures("dq_mode")
def test_gated_mla_bwd_edge_lengths(seqlen, heads, causal):
    """One short sequence, its length at or next to a tile or block edge."""
    _check([seqlen], heads, causal)


def _dp_term_norms(q, kv, k_rot, gate, dout, cu, causal):
    """Norms of dq, dk_nope and dk_rot in fp32 with dS = P dP for dS = P (dP - D): the
    half of each gradient that its D half cancels."""
    k, v = T._expanded_kv(kv, k_rot)
    q, k, v = q.float(), k.float(), v.float()
    d_o = dout.float().view(v.shape) * gate.float().view(v.shape).sigmoid()
    dq, dk = torch.empty_like(q), torch.empty_like(k)
    for b in range(cu.numel() - 1):
        rows = slice(int(cu[b]), int(cu[b + 1]))
        s = rows.stop - rows.start
        scores = torch.einsum("qhd,khd->hqk", q[rows], k[rows]) * T.SCALE
        if causal:
            mask = torch.ones(s, s, dtype=torch.bool, device=q.device).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
        ds = scores.softmax(-1) * torch.einsum("qhd,khd->hqk", d_o[rows], v[rows])
        dq[rows] = torch.einsum("hqk,khd->qhd", ds, k[rows]) * T.SCALE
        dk[rows] = torch.einsum("hqk,qhd->khd", ds, q[rows]) * T.SCALE
    return dq.norm(), dk[..., : T.D_NOPE].norm(), dk[..., T.D_NOPE :].sum(1).norm()


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("batch", [1, 4, 16])
@pytest.mark.parametrize("seqlen", [1, 2])
@pytest.mark.usefixtures("dq_mode")
def test_gated_mla_bwd_one_and_two_tokens(seqlen, batch, causal):
    """Sequences of one or two tokens. A query that sees one key has P = 1 and
    dS = P (dP - D) = 0, and two keys leave dS = +-P0 P1 (dP0 - dP1): dq, dk_nope and dk_rot
    are zero or what is left of two terms that cancel, while the kernels' D, the rowsum of
    dout times the bf16 `out`, carries that output's rounding. Their errors against fp32
    autograd are therefore measured against the dP half of each gradient instead of the
    reference's own norm; dv and dgate do not cancel and are compared as usual."""
    cu, smax, q, kv, k_rot, gate, dout = _inputs([seqlen] * batch, 9)
    dq, dkv, dk_rot, dgate = _fwd_bwd(q, kv, k_rot, gate, dout, cu, smax, causal)
    ref_dq, ref_dkv, ref_dk_rot, ref_dgate = T._ref_bwd(
        q, kv, k_rot, gate, dout, cu, causal
    )
    _assert_parity(
        (dq, dkv[..., : T.D_NOPE], dk_rot),
        (ref_dq, ref_dkv[..., : T.D_NOPE], ref_dk_rot),
        ("dq / dP half", "dk_nope / dP half", "dk_rot / dP half"),
        _dp_term_norms(q, kv, k_rot, gate, dout, cu, causal),
    )
    _assert_parity(
        (dkv[..., T.D_NOPE :], dgate),
        (ref_dkv[..., T.D_NOPE :], ref_dgate),
        ("dv", "dgate"),
    )


# Up to 20 dQ blocks and 40 pair-pass blocks a head, lengths off the 32-row tiles included,
# from a partial scheduling group of heads to several whole ones.
_MEDIUM = [
    ([1000], 24),
    ([2048], 24),
    ([3000], 128),
    ([4096], 96),
    ([4097], 24),
    ([5000], 24),
]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("seqlens,heads", _MEDIUM, ids=_case_ids(_MEDIUM))
@pytest.mark.usefixtures("dq_mode")
def test_gated_mla_bwd_medium_lengths(seqlens, heads, causal):
    _check(seqlens, heads, causal)


# Few heads keep the fp32 reference affordable: H * s * s * 4 bytes a temporary.
_LARGE = [
    ([8192], 1),
    ([8192], 4),
    ([8192], 16),
    ([8192, 8192], 3),
    ([16384], 1),
    ([16384], 2),
    ([16384], 4),
    ([16384, 8192, 4096, 2048], 2),
]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("seqlens,heads", _LARGE, ids=_case_ids(_LARGE))
@pytest.mark.usefixtures("dq_mode")
def test_gated_mla_bwd_large(seqlens, heads, causal):
    _check(seqlens, heads, causal)


def _ref_bwd_per_head(q, kv, k_rot, gate, dout, cu, causal, heads):
    """`T._ref_bwd` one sequence and one head at a time, which bounds its temporaries to
    s * s * 4 bytes each: d(q), d(kv) and d(gate) of `heads` as [T, len(heads), ...] and
    those heads' share of d(k_rot)."""
    t, h = q.shape[:2]
    gate, dout = gate.view(t, h, -1), dout.view(t, h, -1)
    bounds = cu.tolist()
    saved, T.HEADS = T.HEADS, 1
    per_seq = []
    for start, stop in itertools.pairwise(bounds):
        rows = slice(start, stop)
        cu_seq = torch.tensor([0, stop - start], dtype=cu.dtype, device=cu.device)
        dq, dkv, dk_rot, dgate = zip(
            *(
                T._ref_bwd(
                    q[rows, i : i + 1],
                    kv[rows, i : i + 1],
                    k_rot[rows],
                    gate[rows, i],
                    dout[rows, i],
                    cu_seq,
                    causal,
                )
                for i in heads
            )
        )
        per_seq.append(
            (torch.cat(dq, 1), torch.cat(dkv, 1), sum(dk_rot), torch.stack(dgate, 1))
        )
    T.HEADS = saved
    return tuple(torch.cat(x) for x in zip(*per_seq))


# The first and last heads of scheduling groups and a few in between.
_SOME_HEADS = (0, 1, 15, 16, 47, 64, 80, 95)
# (seqlens, heads, heads the fp32 reference is computed for; None: all, which adds dk_rot,
# the sum of every head's). The K3 head count from 8192 tokens to batches of 131072, where
# q, kv and their gradients are 4.5 and 6 GiB.
_FULL_SIZE = [
    ([8192], 96, None),
    ([16384], 96, None),
    ([8192] * 4, 64, None),
    ([32768], 96, None),
    ([65536], 96, _SOME_HEADS),
    ([32768] * 4, 96, _SOME_HEADS),
    ([65536] * 2, 96, _SOME_HEADS),
]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "seqlens,heads,ref_heads",
    _FULL_SIZE,
    ids=_case_ids([case[:2] for case in _FULL_SIZE]),
)
@pytest.mark.usefixtures("dq_mode")
def test_gated_mla_bwd_full_size(seqlens, heads, ref_heads, causal):
    """Production-size problems against fp32 autograd taken one head at a time, on all
    heads or, for the largest, on `_SOME_HEADS`."""
    cu, smax, q, kv, k_rot, gate, dout = _inputs(seqlens, heads)
    dq, dkv, dk_rot, dgate = _fwd_bwd(q, kv, k_rot, gate, dout, cu, smax, causal)
    dgate = dgate.view(q.shape[0], heads, -1)
    if ref_heads is None:
        ref = _ref_bwd_per_head(q, kv, k_rot, gate, dout, cu, causal, range(heads))
        _assert_parity((dq, dkv, dk_rot, dgate), ref)
    else:
        idx = list(ref_heads)
        ref = _ref_bwd_per_head(q, kv, k_rot, gate, dout, cu, causal, idx)
        _assert_parity(
            (dq[:, idx], dkv[:, idx], dgate[:, idx]),
            (ref[0], ref[1], ref[3]),
            ("dq", "dkv", "dgate"),
        )


# Equal-length batches and varlen mixes of 2, 4 and 16 sequences: one-token sequences
# between longer ones, every edge length at an unaligned token offset, and one long sequence
# with many short ones (its blocks are dispatched with theirs).
_BATCHES = [
    ([513] * 2, 96),
    ([1000] * 4, 96),
    ([257] * 16, 96),
    ([1, 1000], 96),
    ([700, 1, 1900, 300], 96),
    ([1, 2, *_EDGE_LENS], 96),
    ([4096] + [17] * 15, 96),
    ([1] * 7 + [2500] + [1] * 8, 24),
    ([100, 3000] + [64] * 14, 24),
]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("seqlens,heads", _BATCHES, ids=_case_ids(_BATCHES))
@pytest.mark.usefixtures("dq_mode")
def test_gated_mla_bwd_batches(seqlens, heads, causal):
    _check(seqlens, heads, causal)


# Whole and partial scheduling groups of 16 heads, and heads per XCD from none to several.
_HEAD_COUNTS = [1, 2, 3, 7, 8, 9, 12, 16, 17, 24, 33, 96, 128]
# Several blocks of both passes a sequence, with a one-token sequence in between.
_SEQLENS = [600, 1, 257, 1000]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "seqlens", [_SEQLENS, [1024] * 2], ids=["b4_min1_max1000", "b2_s1024"]
)
@pytest.mark.parametrize("heads", _HEAD_COUNTS)
@pytest.mark.usefixtures("dq_mode")
def test_gated_mla_bwd_head_counts(heads, seqlens, causal):
    _check(seqlens, heads, causal)


# (heads, heads whose dS the budget holds, heads of each "ds" launch; none: dQ recomputed)
_DS_LAUNCHES = [
    (96, 96, [96]),
    (96, 40, [32, 32, 32]),
    (128, 100, [64, 64]),
    (40, 16, [16, 16, 8]),
    (33, 16, [16, 16, 1]),
    (40, 15, [8, 8, 8, 8, 8]),
    (17, 8, [8, 8, 1]),
    (17, 7, [4, 4, 4, 4, 1]),
    (9, 4, [4, 4, 1]),
    (40, 3, []),
    (2, 1, []),
]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "heads,fit,launches",
    _DS_LAUNCHES,
    ids=[f"h{h}_fit{n}" for h, n, _ in _DS_LAUNCHES],
)
def test_gated_mla_bwd_ds_launches(heads, fit, launches, causal, monkeypatch):
    """The "ds" mode under a dS budget of `fit` heads' tiles: launches of 16, 8 or 4 heads
    (a multiple of 16 where more fit) with the heads that are left in the last one, or dQ
    recomputed where fewer than 4 fit. Every layout is held to fp32 autograd, as "separate" is.
    """
    cu, smax, q, kv, k_rot, gate, dout = _inputs(_SEQLENS, heads)
    per_head = sum(gmt._ds_tiles_per_head(n, n, causal) for n in _SEQLENS)
    gmt._BWD_DS_BUDGET = fit * per_head * gmt._DS_TILE_BYTES

    def launch_heads(name):
        """Wraps gmt's launcher `name` to list the heads of each of its launches."""
        launcher, heads_of = getattr(gmt, name), []

        def counted(first, *args):
            heads_of.append(first.shape[1])
            return launcher(first, *args)

        monkeypatch.setattr(gmt, name, counted)
        return heads_of

    # The two "ds" passes and the launch that recomputes dQ ("separate": dK / dV and dQ in one).
    pair_ds = launch_heads("_gated_mla_bwd_dkdv_pair_ds_bf16_opus")
    dq_ds = launch_heads("_gated_mla_bwd_dq_ds_bf16_opus")
    dq_recomputed = launch_heads("_gated_mla_bwd_dkdvdq_bf16_opus")

    out, lse = gmt._gated_mla_fwd(q, kv, k_rot, gate, cu, smax, T.SCALE, causal, True)
    args = (q, kv, k_rot, gate, out, lse, dout, cu, smax, causal)
    ds = _bwd("ds", *args)
    print(f"  ds launches {pair_ds or 'none, dQ recomputed'}")
    assert pair_ds == dq_ds == launches
    assert dq_recomputed == ([] if launches else [heads])
    assert all(n <= fit for n in pair_ds)
    ref = T._ref_bwd(q, kv, k_rot, gate, dout, cu, causal)
    _assert_parity(ds, ref)
    _assert_parity(_bwd("separate", *args), ref)


def _within(x, front):
    """A view equal to `x`: the last dim of a wider tensor from element `front` on."""
    n = x.shape[-1]
    wide = torch.randn(*x.shape[:-1], front + n + 8, device=x.device, dtype=x.dtype)
    wide[..., front : front + n] = x
    return wide[..., front : front + n]


def _every_other(x, dim):
    """A view equal to `x`: every other index along `dim` of a tensor twice as long."""
    shape = list(x.shape)
    shape[dim] *= 2
    index = [slice(None)] * x.dim()
    index[dim] = slice(1, None, 2)
    wide = torch.randn(shape, device=x.device, dtype=x.dtype)
    wide[tuple(index)] = x
    return wide[tuple(index)]


# Views of the contiguous inputs, by argument. 16-byte aligned rows with strides that are
# multiples of 8 elements are read in place; `_kernel_ready` copies the unaligned ones (an
# odd element offset) of q, k_rot, gate and dout. kv always goes to the kernels as it is.
_LAYOUTS = {
    "k_rot_of_kv_a_proj": {"k_rot": lambda x: _within(x, T.KV_LORA)},
    "k_rot_unaligned": {"k_rot": lambda x: _within(x, 1)},
    "q_slice": {"q": lambda x: _within(x, 8)},
    "q_slice_unaligned": {"q": lambda x: _within(x, 1)},
    "q_every_other_head": {"q": lambda x: _every_other(x, 1)},
    "q_every_other_token": {"q": lambda x: _every_other(x, 0)},
    "kv_slice": {"kv": lambda x: _within(x, 32)},
    "kv_every_other_head": {"kv": lambda x: _every_other(x, 1)},
    "kv_every_other_token": {"kv": lambda x: _every_other(x, 0)},
    "gate_dout_slices": {
        "gate": lambda x: _within(x, 8),
        "dout": lambda x: _within(x, 8),
    },
    "gate_dout_unaligned": {
        "gate": lambda x: _within(x, 1),
        "dout": lambda x: _within(x, 3),
    },
    "all_views": {
        "q": lambda x: _every_other(x, 1),
        "kv": lambda x: _within(x, 32),
        "k_rot": lambda x: _within(x, T.KV_LORA),
        "gate": lambda x: _within(x, 8),
        "dout": lambda x: _within(x, 1),
    },
}


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.usefixtures("dq_mode")
def test_gated_mla_bwd_input_layouts(layout, causal):
    """Strided, sliced and unaligned views of the inputs give the gradients of contiguous
    copies bit for bit, in each dQ mode."""
    seqlens, heads = [300, 1, 700], 17
    cu, smax, q, kv, k_rot, gate, dout = _inputs(seqlens, heads)
    x = {"q": q, "kv": kv, "k_rot": k_rot.contiguous(), "gate": gate, "dout": dout}
    assert all(v.is_contiguous() for v in x.values())
    plain = _fwd_bwd(**x, cu=cu, smax=smax, causal=causal)
    views = {name: make(x[name]) for name, make in _LAYOUTS[layout].items()}
    for name, view in views.items():
        assert not view.is_contiguous() and torch.equal(view, x[name]), name
    ours = _fwd_bwd(**{**x, **views}, cu=cu, smax=smax, causal=causal)
    _assert_equal(ours, plain, f"{layout} against contiguous inputs")
    _assert_parity(ours, T._ref_bwd(q, kv, k_rot, gate, dout, cu, causal))


def _ref_bwd_shared_kv(q, kv, k_rot, gate, dout, cu, causal):
    """fp32 autograd as `T._ref_bwd`, each head of `kv` shared by the next
    q.shape[1] / kv.shape[1] query heads: d(q), d(kv) [T, KV heads, 256], d(k_rot), d(gate).
    """
    qf, kvf, krf, gf = (
        x.detach().float().requires_grad_() for x in (q, kv, k_rot, gate)
    )
    kv_q = kvf.repeat_interleave(q.shape[1] // kv.shape[1], dim=1)
    k = torch.cat(
        [kv_q[..., : T.D_NOPE], krf.unsqueeze(1).expand(-1, T.HEADS, T.D_ROT)], -1
    )
    out, _ = T._ref_attention(qf, k, kv_q[..., T.D_NOPE :], gf, cu, causal)
    out.backward(dout.float().view(out.shape))
    return qf.grad, kvf.grad, krf.grad, gf.grad


# (query heads, KV heads)
_SHARED_KV = [
    (16, 4),
    (24, 8),
    (33, 11),
    (64, 16),
    (96, 24),
    (12, 1),
    (96, 1),
    (128, 1),
]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "heads,kv_heads", _SHARED_KV, ids=[f"hq{h}_hk{k}" for h, k in _SHARED_KV]
)
@pytest.mark.usefixtures("dq_mode")
def test_gated_mla_bwd_shared_kv(heads, kv_heads, causal):
    """GQA / MQA on kernels that take a KV row per query head: the shared KV is expanded to
    the query heads, as a stride-0 view when one KV head serves them all (MQA) and as a copy
    otherwise, and d(kv) is summed over each KV head's query heads."""
    cu, smax, q, _, k_rot, gate, dout = _inputs(_SEQLENS, heads)
    t, group = q.shape[0], heads // kv_heads
    kv = torch.randn(t, kv_heads, T.D_NOPE + T.D_V, device="cuda", dtype=q.dtype)
    if kv_heads == 1:
        kv_q = kv.expand(t, heads, -1)
        assert kv_q.stride(1) == 0
    else:
        kv_q = kv.repeat_interleave(group, dim=1)
    ours = _fwd_bwd(q, kv_q, k_rot, gate, dout, cu, smax, causal)
    if kv_heads == 1:
        copied = _fwd_bwd(q, kv_q.contiguous(), k_rot, gate, dout, cu, smax, causal)
        _assert_equal(ours, copied, "stride-0 KV view against its copy")
    dq, dkv, dk_rot, dgate = ours
    dkv = dkv.view(t, kv_heads, group, -1).sum(2, dtype=torch.float32)
    _assert_parity(
        (dq, dkv, dk_rot, dgate),
        _ref_bwd_shared_kv(q, kv, k_rot, gate, dout, cu, causal),
    )


_DETERMINISM = [([1000, 3000, 513], 96), ([8192], 33)]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("mode", ["separate", "ds"])
@pytest.mark.parametrize("seqlens,heads", _DETERMINISM, ids=_case_ids(_DETERMINISM))
def test_gated_mla_bwd_deterministic(seqlens, heads, mode, causal):
    """No atomics in either dQ mode: repeated runs give the same bits."""
    cu, smax, q, kv, k_rot, gate, dout = _inputs(seqlens, heads)
    out, lse = gmt._gated_mla_fwd(q, kv, k_rot, gate, cu, smax, T.SCALE, causal, True)
    args = (q, kv, k_rot, gate, out, lse, dout, cu, smax, causal)
    first = _bwd(mode, *args)
    for _ in range(3):
        _assert_equal(_bwd(mode, *args), first, f'"{mode}" run to run')


@pytest.mark.parametrize("causal", [True, False])
def test_gated_mla_train_dq_modes(causal):
    """Module-level op: the gradients of q, kv_latent, k_rot, kv_b_weight and gate match fp32
    autograd through the kv_b_proj expansion in each dQ mode.
    """
    seqlens, heads = [300, 1, 700], 17
    cu, smax, q, _, k_rot, gate, dout = _inputs(seqlens, heads)
    t = q.shape[0]
    kv_latent = torch.randn(t, T.KV_LORA, device="cuda", dtype=q.dtype)
    kv_b_weight = torch.randn(
        heads * (T.D_NOPE + T.D_V), T.KV_LORA, device="cuda", dtype=q.dtype
    )
    kv_b_weight = kv_b_weight * T.KV_LORA**-0.5
    names = ("dq", "d_kv_latent", "dk_rot", "d_kv_b_weight", "dgate")

    def grads(mode):
        gmt._BWD_DQ_MODE = mode
        xs = [
            x.detach().requires_grad_()
            for x in (q, kv_latent, k_rot, kv_b_weight, gate)
        ]
        out = gmt.gated_mla_train(*xs, cu, smax, causal=causal)
        return torch.autograd.grad(out, xs, dout)

    def ref_grads():
        xs = [
            x.detach().float().requires_grad_()
            for x in (q, kv_latent, k_rot, kv_b_weight, gate)
        ]
        qf, latent, krf, weight, gf = xs
        kv = torch.mm(latent, weight.t()).view(t, heads, -1)
        k = torch.cat(
            [kv[..., : T.D_NOPE], krf.unsqueeze(1).expand(-1, heads, T.D_ROT)], -1
        )
        out, _ = T._ref_attention(qf, k, kv[..., T.D_NOPE :], gf, cu, causal)
        return torch.autograd.grad(out, xs, dout.float().view(out.shape))

    ref = ref_grads()
    _assert_parity(grads("separate"), ref, names)
    _assert_parity(grads("ds"), ref, names)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
