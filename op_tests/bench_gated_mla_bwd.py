# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark matrix of the MLA attention backward (OPUS, `_gated_mla_bwd` without its gate)
against the ASM v3 backward, Kimi-K3 MLA head dims (Q / K 192, V 128), bf16, gfx950.

Every row times, on one set of inputs:

  a32    ASM v3 backward with fp32 dQ atomics (batch mode, equal-length sequences)
  a16    ASM v3 backward with bf16 dQ atomics (batch mode), the baseline of the ratios;
         for varlen mixes (group mode) the ASM v3 backward has no D_QK 192 / D_V 128
         kernel and `flash_attn_varlen_func` runs CK, which is then the baseline ("ck")
  sep    OPUS backward, dQ mode "separate" (the default): the D pass, then one launch of
         the dK / dV pass and a dQ pass that recomputes S and dP
  ds     OPUS backward, dQ mode "ds": the pair pass also writes dS, a GEMM pass makes dQ

The conditions are ASM's: no gate. OPUS runs the backward of the attention alone (gate None:
dO is dout as it is, and its pre-pass makes only D = rowsum(dO * O), which ASM makes in a
kernel of its own too), so all of them solve the same problem on the same q, k, v and dO;
ASM gets K materialized as [k_nope | k_rot]. The D pass's share of OPUS's kernel time is
reported (pre%). The forward that provides `out` and `lse` runs with gate = 100, whose
sigmoid is exactly 1 in bf16: its output is the attention's.

GQA / MQA rows (fewer KV heads than query heads): the OPUS kernels read a KV row per query
head, so the shared KV is expanded to the query heads (a stride-0 view when one KV head
serves all of them, a `repeat_interleave` copy otherwise) and d(kv) is summed over each KV
head's query heads; both steps are inside OPUS's timed call.

Timing is interleaved: the functions are timed round robin, each timed run after a warm-up
call of the same function, and a row reports the median over the rounds.
TF = 2.5 * 2 * pairs * Hq * (192 + 128) / ms / 1e9 (2.5 x the forward's FLOPs), pairs the
sum over sequences of s (s + 1) / 2 causal or s * s unmasked.

Checks per row, on gradients computed after the timing:
  valid  OPUS "separate" against ASM a32 (CK for varlen mixes): relative L2 of dq, dk_nope
         and dv, about 1e-4 where both are right (2e-3 with shared KV heads). Above 1e-2
         the row is flagged MISMATCH and its ASM timings are not of a correct backward:
         the ASM v3 backward is known to return wrong gradients on very large tensors.
  ds==   "ds" equals "separate" bit for bit on dq, d(kv) and dk_rot
  a16/a32 relative L2 of ASM a16's gradients against a32's (CSV only)
  --precision adds each implementation's relative L2 error against an exact fp32
         reference on the first and the last KV head (sequences up to 16384 tokens)

    bench_gated_mla_bwd.py --groups size,batch --csv out.csv
    bench_gated_mla_bwd.py --case 4x8192@64/16 --case 1000,3000,513@96 --causal causal
"""

import argparse
import csv
import gc
import itertools
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile

import aiter
import aiter.ops.gated_mla_train as gmt
from aiter.ops.mha import FlashAttnFunc, flash_attn_varlen_func

D_NOPE, D_ROT, D_V = 128, 64, 128
D_QK = D_NOPE + D_ROT
SCALE = D_QK**-0.5
BF16 = torch.bfloat16
MISMATCH_REL_L2 = 1e-2
PRECISION_MAX_SEQLEN = 16384

_VARLEN_MIXES = [
    [1000, 3000, 513],
    [
        8192,
        4096,
        4096,
        2048,
        2048,
        2048,
        1024,
        1024,
        1024,
        1024,
        512,
        512,
        512,
        512,
        768,
        1280,
    ],
    [700, 1900, 4500, 300, 2600, 6000, 1100, 2900],
    [16384, 8192, 4096, 2048],
]


def _batch(b, s, hq=96, hk=None):
    return [s] * b, hq, hk or hq


# Preset groups: (seqlens, query heads, KV heads) each.
PRESETS = {
    "size": [_batch(1, s) for s in (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)],
    "batch": [
        _batch(b, s)
        for b, s in (
            (2, 512),
            (8, 512),
            (32, 512),
            (2, 1024),
            (4, 1024),
            (16, 1024),
            (2, 2048),
            (8, 2048),
            (16, 2048),
            (4, 4096),
            (8, 4096),
            (2, 8192),
            (4, 8192),
            (4, 16384),
            (4, 32768),
        )
    ],
    "heads": [
        _batch(1, s, h) for s in (4096, 16384) for h in (12, 16, 24, 48, 64, 96, 128)
    ]
    + [_batch(4, 8192, 64)],
    "odd": [_batch(1, s) for s in (1000, 3000, 5000, 10000, 20000)],
    "gqa": [
        _batch(4, 8192, 64, 16),
        _batch(4, 8192, 64, 4),
        _batch(1, 8192, 96, 24),
        _batch(1, 16384, 96, 12),
        _batch(4, 8192, 64, 1),
        _batch(1, 8192, 96, 1),
    ],
    "group": [(mix, 96, 96) for mix in _VARLEN_MIXES],
}

COLUMNS = [
    "group",
    "shape",
    "mode",
    "seqs",
    "tokens",
    "max_seqlen",
    "hq",
    "hk",
    "causal",
    "pairs",
    "a32_ms",
    "a32_tf",
    "base",
    "base_ms",
    "base_tf",
    "sep_ms",
    "sep_tf",
    "ds_ms",
    "ds_tf",
    "sep_speedup",
    "ds_speedup",
    "ds_vs_sep",
    "ds_layout",
    "ds_eq_sep",
    "a32_spread",
    "base_spread",
    "sep_spread",
    "ds_spread",
    "valid",
    "val_dq",
    "val_dk",
    "val_dv",
    "val_dk_rot",
    "a16_dq",
    "a16_dk",
    "a16_dv",
    "pre_pct",
    "pre_ms",
    "pair_ms",
    "dq_ms",
    "other_ms",
    "prec_opus_dq",
    "prec_opus_dk",
    "prec_opus_dv",
    "prec_a32_dq",
    "prec_a32_dk",
    "prec_a32_dv",
    "prec_base_dq",
    "prec_base_dk",
    "prec_base_dv",
    "a32_kernel",
    "base_kernel",
    "rounds",
    "gpu",
    "notes",
    "seqlens",
]


def once(fn):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def rr_timed(fns, rounds, target_ms, max_iters):
    """Each of `fns` timed round robin: a timed run is as many calls as fit `target_ms` (1 to
    `max_iters`), after a warm-up call of the same function. Returns the median ms per call
    over the rounds and the rounds' spread, (max - min) / median in percent."""
    iters = {}
    for name, fn in fns.items():
        fn()
        iters[name] = max(1, min(max_iters, int(target_ms / max(once(fn), 1e-3))))
    res = {name: [] for name in fns}
    for _ in range(rounds):
        for name, fn in fns.items():
            fn()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(iters[name]):
                fn()
            end.record()
            torch.cuda.synchronize()
            res[name].append(start.elapsed_time(end) / iters[name])
    median = {name: sorted(v)[len(v) // 2] for name, v in res.items()}
    spread = {name: 100.0 * (max(v) - min(v)) / median[name] for name, v in res.items()}
    return median, spread


def kernel_times(fn):
    """GPU time in ms of one call of `fn` by kernel name, from the torch profiler."""
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    return {
        e.key: e.device_time_total / 1e3
        for e in p.key_averages()
        if e.device_time_total > 0
    }


def top_kernel(fn):
    times = kernel_times(fn)
    return max(times, key=times.get) if times else "?"


def rel(a, ref):
    """Relative L2 of `a` against `ref`, head by head to bound the fp32 copies."""
    num = den = 0.0
    for h in range(a.shape[-2]) if a.dim() >= 3 else [None]:
        x, y = (a, ref) if h is None else (a[..., h, :], ref[..., h, :])
        num += (x.float() - y.float()).pow(2).sum().item()
        den += y.float().pow(2).sum().item()
    return (num / max(den, 1e-30)) ** 0.5


def ds_layout(seqlens, heads, causal):
    """The "ds" mode's launches for this problem, as `_gated_mla_bwd` lays them out."""
    per_head = max(1, sum(gmt._ds_tiles_per_head(n, n, causal) for n in seqlens))
    group = gmt._ds_head_group(heads, per_head)
    if not group:
        return "recomputes dQ"
    launches = [min(group, heads - h0) for h0 in range(0, heads, group)]
    text = f"{len(launches)}x{group}"
    if launches[-1] != group:
        text = f"{len(launches) - 1}x{group}+{launches[-1]}"
    return f"{text} heads {group * per_head * gmt._DS_TILE_BYTES / 2**30:.1f} GiB"


def exact_reference(q, kv, k_rot, dout, bounds, causal, kv_head, chunk=2048):
    """Exact fp32 gradients of KV head `kv_head` and its query heads, a chunk of queries at a
    time: dQ [T, heads per KV head, 192], dK nope [T, 128] and dV [T, 128]."""
    t, hq = q.shape[:2]
    ratio = hq // kv.shape[1]
    dout = dout.view(t, hq, D_V)
    dq = torch.empty(t, ratio, D_QK, device=q.device)
    dk = torch.zeros(t, D_QK, device=q.device)
    dv = torch.zeros(t, D_V, device=q.device)
    for start, stop in itertools.pairwise(bounds):
        s = stop - start
        ar = torch.arange(s, device=q.device)
        kh = torch.cat(
            [kv[start:stop, kv_head, :D_NOPE], k_rot[start:stop]], dim=-1
        ).float()
        vh = kv[start:stop, kv_head, D_NOPE:].float()
        for j in range(ratio):
            qh = q[start:stop, kv_head * ratio + j].float()
            doh = dout[start:stop, kv_head * ratio + j].float()
            for c0 in range(0, s, chunk):
                c1 = min(s, c0 + chunk)
                nk = c1 if causal else s
                sc = (qh[c0:c1] @ kh[:nk].T) * SCALE
                if causal:
                    sc.masked_fill_(ar[None, :nk] > ar[c0:c1, None], float("-inf"))
                p = torch.softmax(sc, dim=-1)
                del sc
                o = p @ vh[:nk]
                ds = p * (
                    doh[c0:c1] @ vh[:nk].T - (doh[c0:c1] * o).sum(-1, keepdim=True)
                )
                dq[start + c0 : start + c1, j] = (ds @ kh[:nk]) * SCALE
                dk[start : start + nk] += (ds.T @ qh[c0:c1]) * SCALE
                dv[start : start + nk] += p.T @ doh[c0:c1]
                del p, ds, o
    return dq, dk[:, :D_NOPE], dv


def run_case(group, seqlens, hq, hk, causal, opt):
    b, t, smax = len(seqlens), sum(seqlens), max(seqlens)
    ratio = hq // hk
    batch_mode = len(set(seqlens)) == 1
    pairs = sum(s * (s + 1) // 2 if causal else s * s for s in seqlens)
    rounds = (
        opt.rounds
        if sum(s * s for s in seqlens) * hq < 4e9 * 96
        else max(3, opt.rounds - 2)
    )
    row = dict.fromkeys(COLUMNS, "")
    row.update(
        group=group,
        shape=f"{b}x{smax}" if batch_mode else f"mix{b}/max{smax}",
        mode="batch" if batch_mode else "group",
        seqs=b,
        tokens=t,
        max_seqlen=smax,
        hq=hq,
        hk=hk,
        causal=int(causal),
        pairs=pairs,
        base="a16" if batch_mode else "ck",
        rounds=rounds,
        gpu=opt.gpu,
        seqlens=" ".join(map(str, seqlens)),
    )
    notes = []
    bounds = [0, *itertools.accumulate(seqlens)]
    cu = torch.tensor(bounds, dtype=torch.int32, device="cuda")
    torch.manual_seed(0)
    q = torch.randn(t, hq, D_QK, device="cuda", dtype=BF16)
    kv = torch.randn(t, hk, D_NOPE + D_V, device="cuda", dtype=BF16)
    k_rot = torch.randn(t, D_ROT, device="cuda", dtype=BF16)
    dout = torch.randn(t, hq * D_V, device="cuda", dtype=BF16)
    gate = torch.full((t, hq * D_V), 100.0, device="cuda", dtype=BF16)

    def kv_per_query_head():
        if ratio == 1:
            return kv
        return kv.expand(t, hq, -1) if hk == 1 else kv.repeat_interleave(ratio, dim=1)

    out, lse = gmt._gated_mla_fwd(
        q, kv_per_query_head(), k_rot, gate, cu, smax, SCALE, causal, True
    )

    def opus(mode):
        def fn():
            gmt._BWD_DQ_MODE = mode
            dq, dkv, dk_rot, _ = gmt._gated_mla_bwd(
                q,
                kv_per_query_head(),
                k_rot,
                None,  # no gate, as ASM: `out` is the attention's output
                out,
                lse,
                dout,
                cu,
                smax,
                SCALE,
                causal,
            )
            if ratio > 1:
                dkv = dkv.view(t, hk, ratio, -1).sum(2, dtype=torch.float32).to(BF16)
            return dq, dkv, dk_rot

        return fn

    # ASM / CK on the materialized K: q, k [.., heads, 192], v and dO [.., heads, 128].
    k_asm = torch.cat([kv[..., :D_NOPE], k_rot[:, None].expand(t, hk, D_ROT)], dim=-1)
    v_asm = kv[..., D_NOPE:].contiguous()
    lead = (b, smax) if batch_mode else (t,)
    qa = q.view(*lead, hq, D_QK).detach().requires_grad_()
    ka = k_asm.view(*lead, hk, D_QK).requires_grad_()
    va = v_asm.view(*lead, hk, D_V).requires_grad_()
    do = dout.view(*lead, hq, D_V)

    def baseline(a32):
        if batch_mode:
            o = FlashAttnFunc.apply(
                qa,
                ka,
                va,
                0.0,
                None,
                causal,
                (-1, -1, 0),
                None,
                None,
                False,
                True,
                False,
                True,
                a32,
                1,
                None,
                None,
                None,
                0,
            )[0]
        else:
            o = flash_attn_varlen_func(
                qa, ka, va, cu, cu, smax, smax, causal=causal, return_lse=True
            )[0]
        return lambda: torch.autograd.grad(o, (qa, ka, va), do, retain_graph=True)

    fns = {}
    wanted = (["a32"] if batch_mode and not opt.no_a32 else []) + [row["base"]]
    for name in wanted:
        try:
            fns[name] = baseline(name == "a32")
            fns[name]()
        except Exception as ex:  # noqa: BLE001
            fns.pop(name, None)
            notes.append(
                f"{name} failed: {type(ex).__name__}: {str(ex).splitlines()[0][:120]}"
            )
    fns["sep"], fns["ds"] = opus("separate"), opus("ds")

    ms, spread = rr_timed(fns, rounds, opt.target_ms, opt.max_iters)
    flops = 2.5 * 2 * pairs * hq * (D_QK + D_V)
    for name, key in (
        ("a32", "a32"),
        (row["base"], "base"),
        ("sep", "sep"),
        ("ds", "ds"),
    ):
        if name in ms:
            row[f"{key}_ms"], row[f"{key}_tf"] = ms[name], flops / ms[name] / 1e9
            row[f"{key}_spread"] = spread[name]
    if row["base"] in ms:
        row["sep_speedup"] = ms[row["base"]] / ms["sep"]
        row["ds_speedup"] = ms[row["base"]] / ms["ds"]
    row["ds_vs_sep"] = ms["sep"] / ms["ds"]
    row["ds_layout"] = ds_layout(seqlens, hq, causal)

    # Checks, on gradients as [T, heads, d].
    g_sep, g_ds = fns["sep"](), fns["ds"]()
    row["ds_eq_sep"] = int(all(torch.equal(x, y) for x, y in zip(g_ds, g_sep)))
    del g_ds

    def flat(grads):
        return [x.reshape(t, x.shape[-2], -1) for x in grads]

    g_ref = g_base = None
    ref_name = "a32" if batch_mode else row["base"]
    if ref_name in fns:
        g_ref = flat(fns[ref_name]())
        dq, dk, dv = g_ref
        dk_rot = torch.zeros(t, D_ROT, device="cuda")
        for h in range(hk):
            dk_rot += dk[:, h, D_NOPE:].float()
        row["val_dq"] = rel(g_sep[0], dq)
        row["val_dk"] = rel(g_sep[1][..., :D_NOPE], dk[..., :D_NOPE])
        row["val_dv"] = rel(g_sep[1][..., D_NOPE:], dv)
        row["val_dk_rot"] = rel(g_sep[2], dk_rot)
        worst = max(row["val_dq"], row["val_dk"], row["val_dv"])
        row["valid"] = "ok" if worst <= MISMATCH_REL_L2 else "MISMATCH"
        del dk_rot
    if batch_mode and "a16" in fns:
        g_base = flat(fns["a16"]())
        if g_ref is not None:
            row["a16_dq"], row["a16_dk"], row["a16_dv"] = (
                rel(x, y) for x, y in zip(g_base, g_ref)
            )
    elif not batch_mode:
        g_base, g_ref = g_ref, None

    if opt.precision and smax <= PRECISION_MAX_SEQLEN:
        err = {name: [0.0] * 6 for name in ("opus", "a32", "base")}

        def add(name, grads, kv_head, ref):
            heads = slice(kv_head * ratio, (kv_head + 1) * ratio)
            mine = (
                grads[0][:, heads],
                grads[1][:, kv_head, :D_NOPE],
                grads[2][:, kv_head],
            )
            for i, (x, y) in enumerate(zip(mine, ref)):
                err[name][2 * i] += (x.float() - y).pow(2).sum().item()
                err[name][2 * i + 1] += y.pow(2).sum().item()

        for kv_head in sorted({0, hk - 1}):
            ref = exact_reference(q, kv, k_rot, dout, bounds, causal, kv_head)
            add(
                "opus",
                (g_sep[0], g_sep[1][..., :D_NOPE], g_sep[1][..., D_NOPE:]),
                kv_head,
                ref,
            )
            for name, grads in (("a32", g_ref), ("base", g_base)):
                if grads is not None:
                    add(
                        name, (grads[0], grads[1][..., :D_NOPE], grads[2]), kv_head, ref
                    )
            del ref
        for name, e in err.items():
            if e[1]:
                for i, part in enumerate(("dq", "dk", "dv")):
                    row[f"prec_{name}_{part}"] = (e[2 * i] / e[2 * i + 1]) ** 0.5
    del g_sep, g_ref, g_base

    if not opt.no_profile:
        for name, key in (("a32", "a32_kernel"), (row["base"], "base_kernel")):
            if name in fns:
                row[key] = top_kernel(fns[name])
        if (
            row["base"] == "a16"
            and row["base_kernel"]
            and "_a16" not in row["base_kernel"]
        ):
            notes.append("a16 request did not run an a16 kernel")
        times = kernel_times(fns["sep"])
        parts = {"pre": 0.0, "pair": 0.0, "dq": 0.0, "other": 0.0}
        for key, value in times.items():
            part = "other"
            if "gated_mla_bwd_pre" in key:
                part = "pre"
            elif "gated_mla_bwd_dkdv" in key or "gated_mla_bwd_4w" in key:
                part = "pair"  # "separate": the one launch of the dK / dV and dQ passes
            elif "gated_mla_bwd_dq" in key:
                part = "dq"
            parts[part] += value
        for part, value in parts.items():
            row[f"{part}_ms"] = value
        row["pre_pct"] = 100.0 * parts["pre"] / max(sum(parts.values()), 1e-9)
    row["notes"] = "; ".join(notes)
    return row


def _cell(row, key):
    ms = row[f"{key}_ms"]
    return f"{ms:9.2f} {row[f'{key}_tf']:4.0f}" if ms != "" else f"{'-':>14s}"


def _num(value, fmt):
    return format(value, fmt) if value != "" else "-"


HEADER = (
    f"{'group':6s} {'shape':15s} {'mode':5s} {'tokens':>7s} {'hq/hk':>7s} {'mask':6s} | "
    f"{'a32 ms   TF':>14s} | {'a16|ck ms   TF':>14s} | {'sep ms   TF':>14s} | {'ds ms   TF':>14s} | "
    f"{'sep/base':>8s} {'ds/base':>8s} {'ds/sep':>7s} | {'pre%':>5s} | ds launches"
)


def print_row(row):
    mark = "*" if row["valid"] == "MISMATCH" else " "
    print(
        f"{row['group']:6s} {row['shape']:15s} {row['mode']:5s} {row['tokens']:7d} "
        f"{str(row['hq']) + '/' + str(row['hk']):>7s} {'causal' if row['causal'] else 'full':6s} | "
        f"{_cell(row, 'a32')} | {_cell(row, 'base')} | {_cell(row, 'sep')} | {_cell(row, 'ds')} | "
        f"{_num(row['sep_speedup'], '7.3f')}{mark} {_num(row['ds_speedup'], '7.3f')}{mark} "
        f"{_num(row['ds_vs_sep'], '7.3f')} | {_num(row['pre_pct'], '5.1f')} | {row['ds_layout']}"
    )
    ref = "a32" if row["mode"] == "batch" else "ck"
    line = (
        f"       valid {row['valid'] or '-'}: sep vs {ref} dq {_num(row['val_dq'], '.2e')} "
        f"dk {_num(row['val_dk'], '.2e')} dv {_num(row['val_dv'], '.2e')} "
        f"dk_rot {_num(row['val_dk_rot'], '.2e')} | ds==sep {row['ds_eq_sep']}"
    )
    if row["a16_dq"] != "":
        line += f" | a16 vs a32 dq {row['a16_dq']:.2e} dk {row['a16_dk']:.2e} dv {row['a16_dv']:.2e}"
    print(line + f" | {row['base']} kernel {str(row['base_kernel'])[:52]}")
    spreads = [
        f"{k} {row[f'{k}_spread']:.1f}"
        for k in ("a32", "base", "sep", "ds")
        if row[f"{k}_spread"] != ""
    ]
    print(f"       spread of the rounds, % of the median: {' '.join(spreads)}")
    if row["prec_opus_dq"] != "":
        line = "       fp32 ref dq / dk_nope / dv:"
        for name, label in (("opus", "sep"), ("a32", "a32"), ("base", row["base"])):
            if row[f"prec_{name}_dq"] != "":
                line += (
                    f" {label} "
                    + " ".join(
                        f"{row[f'prec_{name}_{p}']:.2e}" for p in ("dq", "dk", "dv")
                    )
                    + " |"
                )
        print(line.rstrip(" |"))
    if row["notes"]:
        print(f"       note: {row['notes']}")
    sys.stdout.flush()


def append_csv(path, row):
    """Appends `row` to the CSV at `path`, after the header if the file is new."""
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(
            {k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()}
        )


def parse_case(text):
    """`SEQS[@HQ[/HK]]`, SEQS as `BxS` or a comma-separated list of lengths; 96 heads by default."""
    seqs, _, heads = text.partition("@")
    if "x" in seqs:
        b, s = map(int, seqs.split("x"))
        seqlens = [s] * b
    else:
        seqlens = [int(x) for x in seqs.split(",")]
    hq, _, hk = (heads or "96").partition("/")
    return seqlens, int(hq), int(hk or hq)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--groups",
        default="",
        help=f"comma-separated preset groups, or all: {','.join(PRESETS)}",
    )
    ap.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="SEQS[@HQ[/HK]]",
        help="an extra case: 4x8192@64/16 (batch x length @ query heads / KV heads) "
        "or 1000,3000,513@96 (a varlen mix); repeatable",
    )
    ap.add_argument(
        "--causal",
        choices=("both", "causal", "full"),
        default="both",
        help="masks to run: causal, full (unmasked) or both (default)",
    )
    ap.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="timing rounds (2 fewer, at least 3, on the largest)",
    )
    ap.add_argument(
        "--target-ms", type=float, default=400.0, help="duration of a timed run"
    )
    ap.add_argument(
        "--max-iters", type=int, default=1000, help="most calls in a timed run"
    )
    ap.add_argument("--csv", help="append the rows to this CSV file")
    ap.add_argument(
        "--precision",
        action="store_true",
        help=f"errors against an exact fp32 reference (sequences up to {PRECISION_MAX_SEQLEN})",
    )
    ap.add_argument(
        "--no-a32",
        action="store_true",
        help="skip the ASM a32 timing and the check against it",
    )
    ap.add_argument(
        "--no-profile",
        action="store_true",
        help="skip the kernel names and the pre-pass share",
    )
    ap.add_argument(
        "--ds-budget",
        type=float,
        metavar="GiB",
        help='dS buffer budget of the "ds" mode',
    )
    ap.add_argument(
        "--expect-aiter",
        metavar="DIR",
        help="fail unless aiter is imported from under DIR",
    )
    ap.add_argument(
        "--no-build",
        action="store_true",
        help="fail instead of JIT-building a module that is not built yet",
    )
    ap.add_argument(
        "--list", action="store_true", help="list the selected cases and exit"
    )
    opt = ap.parse_args()

    print(f"aiter: {aiter.__file__}")
    if opt.expect_aiter and not aiter.__file__.startswith(
        os.path.join(opt.expect_aiter, "")
    ):
        sys.exit(f"aiter is not imported from {opt.expect_aiter}")
    if opt.no_build:
        import aiter.jit.core as jit_core

        def no_build(md_name, *args, **kwargs):
            raise RuntimeError(f"JIT build of {md_name} refused (--no-build)")

        jit_core.build_module = no_build
    if opt.ds_budget is not None:
        gmt._BWD_DS_BUDGET = int(opt.ds_budget * 2**30)

    names = (
        list(PRESETS)
        if opt.groups == "all"
        else [g for g in opt.groups.split(",") if g]
    )
    unknown = [g for g in names if g not in PRESETS]
    if unknown or not (names or opt.case):
        ap.error(
            f"--groups takes {','.join(PRESETS)} or all (got {unknown}); or give --case"
        )
    cases = [(g, *case) for g in names for case in PRESETS[g]]
    cases += [("case", *parse_case(text)) for text in opt.case]
    masks = {"both": (True, False), "causal": (True,), "full": (False,)}[opt.causal]
    if opt.list:
        for g, seqlens, hq, hk in cases:
            print(
                g,
                f"{len(seqlens)} seqs max {max(seqlens)} tokens {sum(seqlens)} hq {hq} hk {hk}",
            )
        return

    opt.gpu = os.environ.get(
        "HIP_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    )
    free, total = torch.cuda.mem_get_info()
    print(
        f"GPU {opt.gpu}: {torch.cuda.get_device_name(0)}, {free / 2**30:.0f} of {total / 2**30:.0f} GiB free; "
        f"dS budget {gmt._BWD_DS_BUDGET / 2**30:.1f} GiB; rounds {opt.rounds}, {opt.target_ms:.0f} ms a run"
    )
    print(
        "ms: median per call; TF: 2.5 x forward FLOPs / time; sep/base, ds/base: baseline ms / OPUS ms "
        "(above 1: OPUS is faster), baseline = ASM a16, or CK in group mode; * = baseline failed the check"
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    saved = gmt._BWD_DQ_MODE
    for causal in masks:
        print(HEADER)
        for g, seqlens, hq, hk in cases:
            try:
                row = run_case(g, seqlens, hq, hk, causal, opt)
            except Exception as ex:  # noqa: BLE001
                row = dict.fromkeys(COLUMNS, "")
                row.update(
                    group=g,
                    shape=f"{len(seqlens)}x{max(seqlens)}",
                    seqs=len(seqlens),
                    tokens=sum(seqlens),
                    max_seqlen=max(seqlens),
                    hq=hq,
                    hk=hk,
                    causal=int(causal),
                    gpu=opt.gpu,
                    seqlens=" ".join(map(str, seqlens)),
                    notes=f"FAILED {type(ex).__name__}: {str(ex).splitlines()[0][:200]}",
                )
                print(
                    f"{g:6s} {row['shape']:15s} hq {hq} hk {hk} {'causal' if causal else 'full'}: {row['notes']}",
                    flush=True,
                )
            else:
                print_row(row)
            finally:
                gmt._BWD_DQ_MODE = saved
            if opt.csv:
                append_csv(opt.csv, row)
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
