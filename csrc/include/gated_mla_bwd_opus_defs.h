// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Kernel arguments of the gated MLA backward (Kimi-K3 full-attention layers, D_V = 128).
#pragma once

// Pre-pass, one row per (token, head) of D_V = 128:
//   s = sigmoid(gate), dO = dout * s, dgate = dout * out * (1 - s), D = rowsum(dout * out)
// D equals the usual rowsum(dO * O) of the ungated attention, since out = O * s.
// Strides are in elements; D is [H, total] fp32 like the forward's LSE.
struct gated_mla_bwd_pre_kargs {
    const void* __restrict__ ptr_dout;
    const void* __restrict__ ptr_out;
    const void* __restrict__ ptr_gate;
    void* __restrict__ ptr_do;
    void* __restrict__ ptr_dgate;
    float* __restrict__ ptr_d;
    int total;
    int H;
    int stride_dout_n, stride_dout_h;
    int stride_out_n, stride_out_h;
    int stride_gate_n, stride_gate_h;
    int stride_do_n, stride_do_h;
    int stride_dgate_n, stride_dgate_h;
    int stride_d_h;
};

static constexpr int GATED_MLA_BWD_PRE_BLOCK = 256;

// Main pass, one workgroup per (KV block of 192 keys, sequence, head), group / varlen only.
// Per query tile it recomputes S^T = K Q^T and dP^T = V dO^T, forms P and dS from the
// forward's LSE and the pre-pass D, accumulates dK and dV in registers and adds its share of
// dQ into dq_acc with fp32 atomics. K is read as the forward reads it: dims 0..127 from
// k_nope, dims 128..191 from k_rot, which every head shares.
// Outputs: dkv [total_k, H, 256] bf16 with each head's row [dK nope | dV] (the kv_b_proj
// output layout), dk_rope [total_k, H, 64] fp32 (the per-head rope part of dK, summed over
// heads by the caller), dq_acc [total_q, H, 192] fp32 (zeroed by the caller). dK and dQ
// include softmax_scale.
struct gated_mla_bwd_kargs {
    const void* __restrict__ ptr_q;
    const void* __restrict__ ptr_k_nope;
    const void* __restrict__ ptr_v;
    const void* __restrict__ ptr_k_rot;
    const void* __restrict__ ptr_do;
    const float* __restrict__ ptr_lse;
    const float* __restrict__ ptr_d;
    float* __restrict__ ptr_dq_acc;
    void* __restrict__ ptr_dkv;
    float* __restrict__ ptr_dk_rope;
    const int* __restrict__ ptr_seqstart_q;
    const int* __restrict__ ptr_seqstart_k;
    int H;
    int num_seqs;
    int n_kv_blocks;  // per sequence, from max_seqlen_k
    int stride_q_n, stride_q_h;
    int stride_k_n, stride_k_h;
    int stride_v_n, stride_v_h;
    int stride_k_rot_n;
    int stride_do_n, stride_do_h;
    int stride_lse_h;  // LSE and D share [H, total_q]
    int stride_dq_n, stride_dq_h;
    int stride_dkv_n, stride_dkv_h;
    int stride_dkr_n, stride_dkr_h;
    float softmax_scale;
    int causal;
};

namespace gated_mla_bwd {
constexpr int BK = 192;  // keys per workgroup; fewer, larger blocks mean fewer dQ atomics
constexpr int NW = 4;    // waves, one per SIMD; wave w owns keys [48 w, 48 w + 48)
constexpr int WR = BK / NW;
constexpr int BQ = 16;   // queries per tile
constexpr int D_QK = 192, D_V = 128, D_NOPE = 128, D_ROT = 64;
constexpr int DQ_M = D_QK / NW;  // dQ: each wave owns 48 dims of the tile
constexpr int BLOCK = NW * 64;
// Heads per scheduling group, a multiple of the 8 XCDs that workgroups reach round-robin by
// linear id; see the kernel's id map.
constexpr int HG = 16;

// LDS, bf16 elements, all row-major with 8 elements of row padding. Per-tile operands are
// stored once and read transposed with ds_read_b64_tr_b16 where needed; the key block, staged
// once per workgroup, is stored as K^T. sQ, sDO and sDST are double-buffered by query-tile
// parity.
constexpr int LD_Q  = D_QK + 8;  // sQ  [BQ][D_QK]  B of S^T; B of dK (transposed)
constexpr int LD_DO = D_V + 8;   // sDO [BQ][D_V]   B of dP^T; B of dV (transposed)
constexpr int LD_T  = BQ + 8;    // sDST [BK][BQ]   B of dQ^T (transposed)
constexpr int LD_KT = BK + 8;    // sKT [D_QK][BK]  A of dQ^T
constexpr int OFF_Q   = 0;
constexpr int OFF_DO  = OFF_Q + 2 * BQ * LD_Q;
constexpr int OFF_DST = OFF_DO + 2 * BQ * LD_DO;
constexpr int OFF_KT  = OFF_DST + 2 * BK * LD_T;
constexpr int SMEM_ELEMS = OFF_KT + D_QK * LD_KT;
} // namespace gated_mla_bwd

// dK / dV pass, 4 waves: each wave (one per SIMD) owns 48 keys with their K, V, dK and dV
// (accumulators in AGPRs), on 32-query tiles so that every GEMM is a full-rate 16x16x32 MFMA.
// Takes gated_mla_bwd_kargs (n_kv_blocks counted in BK = 192); ptr_dq_acc is unused.
namespace gated_mla_bwd_4w {
constexpr int BK = 192;  // keys per workgroup
constexpr int NW = 4;    // waves; wave w owns keys [48 w, 48 w + 48)
constexpr int WR = BK / NW;
constexpr int BQ = 32;   // queries per tile
constexpr int D_QK = 192, D_V = 128, D_NOPE = 128, D_ROT = 64;
constexpr int BLOCK = NW * 64;
constexpr int HG = 16;
// LDS, bf16, double-buffered by query-tile parity: sQ [BQ][D_QK] (B of S^T; B of dK,
// transposed) and sDO [BQ][D_V] (B of dP^T; B of dV, transposed), rows padded by 16 elements
// (conflict-free, see gated_mla_bwd_pair). The epilogue then transposes each wave's
// [48][dK nope | dV] through the same LDS.
constexpr int LD_Q = D_QK + 16, LD_DO = D_V + 16, LD_O = D_NOPE + D_V + 8;
constexpr int OFF_Q = 0, OFF_DO = OFF_Q + 2 * BQ * LD_Q;
constexpr int SMEM_MAIN = OFF_DO + 2 * BQ * LD_DO, SMEM_EPI = NW * WR * LD_O;
constexpr int SMEM_ELEMS = SMEM_MAIN > SMEM_EPI ? SMEM_MAIN : SMEM_EPI;
} // namespace gated_mla_bwd_4w

// dK / dV pass, 8 waves: the two waves of a SIMD (w, w + 4) share 32 keys and split the work by
// dependency chain (S wave: S^T, P, dV; dP wave: dP^T, dS, dK). Takes gated_mla_bwd_kargs
// (n_kv_blocks counted in BK = 128); ptr_dq_acc is unused.
namespace gated_mla_bwd_pair {
constexpr int BK = 128;  // keys per workgroup
constexpr int NW = 8;    // waves
constexpr int NP = 4;    // wave pairs (w, w + 4), one per SIMD; pair p owns keys [32 p, 32 p + 32)
constexpr int PR = BK / NP;
constexpr int BQ = 32;   // queries per tile
constexpr int D_QK = 192, D_V = 128, D_NOPE = 128, D_ROT = 64;
constexpr int BLOCK = NW * 64;
constexpr int HG = 16;
// LDS, in bf16 elements: NBUF query tiles, each sQ [BQ][D_QK] (B of S^T; B of dK, transposed),
// sDO [BQ][D_V] (B of dP^T; B of dV, transposed) and the tile's LSE and D in fp32 (256 bytes
// each, one DMA load). Rows padded by 16 elements: with 8, both the B-fragment ds_read_b128 and
// the ds_read_b64_tr_b16 reads conflict 2-way. After them sP [NP][PR x BQ] in fp32: P from each
// pair's S wave to its dP wave.
constexpr int NBUF = 5;
constexpr int LD_Q = D_QK + 16, LD_DO = D_V + 16;
constexpr int OFF_Q = 0, OFF_DO = OFF_Q + BQ * LD_Q, OFF_LSE = OFF_DO + BQ * LD_DO, OFF_D = OFF_LSE + 128;
constexpr int SZ_BUF     = OFF_D + 128;
constexpr int OFF_P      = NBUF * SZ_BUF;
constexpr int SMEM_ELEMS = OFF_P + NP * PR * BQ * 2;
} // namespace gated_mla_bwd_pair

// dQ pass without atomics, one workgroup per (query block of 256, sequence, head). Each wave
// keeps its 32 queries' Q, dO and dQ in registers and walks the keys in tiles of 32,
// recomputing S = Q K^T, P, dP = dO V^T and dS = P (dP - D) to accumulate dQ = dS K. Writes
// dq [total_q, H, 192] bf16 once, softmax_scale included; with it the main pass skips dQ.
struct gated_mla_bwd_dq_kargs {
    const void* __restrict__ ptr_q;
    const void* __restrict__ ptr_k_nope;
    const void* __restrict__ ptr_v;
    const void* __restrict__ ptr_k_rot;
    const void* __restrict__ ptr_do;
    const float* __restrict__ ptr_lse;
    const float* __restrict__ ptr_d;
    void* __restrict__ ptr_dq;
    const int* __restrict__ ptr_seqstart_q;
    const int* __restrict__ ptr_seqstart_k;
    int H;
    int num_seqs;
    int n_q_blocks;  // per sequence, from max_seqlen_q
    int stride_q_n, stride_q_h;
    int stride_k_n, stride_k_h;
    int stride_v_n, stride_v_h;
    int stride_k_rot_n;
    int stride_do_n, stride_do_h;
    int stride_lse_h;  // LSE and D share [H, total_q]
    int stride_dq_n, stride_dq_h;
    float softmax_scale;
    int causal;
};

namespace gated_mla_bwd_dq {
constexpr int NW = 8;          // waves
constexpr int WQ = 32;         // queries per wave
constexpr int BQ = NW * WQ;    // queries per workgroup
constexpr int KC = 32;         // keys per tile
constexpr int D_QK = 192, D_V = 128, D_NOPE = 128, D_ROT = 64;
constexpr int BLOCK = NW * 64;
constexpr int HG = 16;         // heads per scheduling group, as in the main pass
// LDS, bf16: NBUF key tiles, each sKN [KC][D_NOPE] and sKR [KC][D_ROT] (K; B of S, and of dQ
// transposed) and sV [KC][D_V] (B of dP). K's two parts come from two tensors, so each part is
// one DMA region. Rows padded by 16 elements (conflict-free, see gated_mla_bwd_pair).
// After them, sQR [BQ][D_ROT]: the rope part of each wave's Q (A of S's last K steps), kept in
// LDS to save its registers.
constexpr int NBUF = 4;
constexpr int LD_KN = D_NOPE + 16, LD_KR = D_ROT + 16, LD_V = D_V + 16, LD_QR = D_ROT + 16;
constexpr int OFF_KN = 0, OFF_KR = OFF_KN + KC * LD_KN, OFF_V = OFF_KR + KC * LD_KR;
constexpr int SZ_BUF     = OFF_V + KC * LD_V;
constexpr int OFF_QR     = NBUF * SZ_BUF;
constexpr int SMEM_ELEMS = OFF_QR + BQ * LD_QR;
} // namespace gated_mla_bwd_dq
