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
    int hg;           // heads per scheduling group (gated_mla_bwd::wg_heads)
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
    // Pair pass only: dS tiles out (gated_mla_bwd_ds), or none if null; each sequence's first tile.
    void* __restrict__ ptr_ds;
    const long long* __restrict__ ptr_ds_seq;
};

namespace gated_mla_bwd {
constexpr int BK = 192;  // keys per workgroup; fewer, larger blocks mean fewer dQ atomics
constexpr int NW = 4;    // waves, one per SIMD; wave w owns keys [48 w, 48 w + 48)
constexpr int WR = BK / NW;
constexpr int BQ = 16;   // queries per tile
constexpr int D_QK = 192, D_V = 128, D_NOPE = 128, D_ROT = 64;
constexpr int DQ_M = D_QK / NW;  // dQ: each wave owns 48 dims of the tile
constexpr int BLOCK = NW * 64;
// Workgroup id -> (head, sequence, block in dispatch order), for every pass. Ids run: groups of
// `hg` heads one after another (only about hg heads' tensors are live in the caches at a time);
// within a group its sequences in order; within a sequence its blocks in order (the long causal
// ones first); the head fastest. The last group holds the heads that are left and has no idle ids.
//
// The GPU deals workgroups by id to 32 lanes (id % 32: its 8 XCDs by id % 8, 4 lanes of 8 CUs in
// each), a round of 32 ids at a time and in step: a lane gets its next workgroup only once every
// lane has taken the round before, so in a round whose workgroups differ in length the lanes with
// the shorter ones idle by the difference. With hg = 16 each XCD keeps to 2 heads of a group, whose
// consecutive blocks run together and read the same tiles close in time, from L2 (which a batch's
// other sequences in between, or more heads, undo); a round then holds two consecutive blocks,
// and half the lanes idle one block's step per workgroup, little against long workgroups. With
// hg = 32 a round is 32 heads' same block and no lane idles, at 4 heads on an XCD. Measured on an
// MI350X, 96 heads, causal: hg = 32 is 4 to 5% faster on batches of 512 and 1024 tokens, the same
// from 2K to 8K, 2.5% slower at 16K; hg = 64 is no better anywhere; in non-causal attention the
// workgroups of a sequence are all as long, and no lane idles.
constexpr int HG       = 16;    // heads per scheduling group
constexpr int HG_SHORT = 32;    // for causal sequences of at most HG_SHORT_MAX tokens
#ifndef GMLA_HGS
#define GMLA_HGS 2048
#endif
constexpr int HG_SHORT_MAX = GMLA_HGS;
struct wg_ids {
    int h, g, blk;
};
__host__ __device__ inline int wg_heads(int max_seqlen, int causal)
{
    return causal && max_seqlen <= HG_SHORT_MAX ? HG_SHORT : HG;
}
__host__ __device__ inline long long wg_count(int H, int num_seqs, int n_blocks)
{
    return static_cast<long long>(H) * num_seqs * n_blocks;
}
// `hg`: wg_heads' power of two.
__host__ __device__ inline wg_ids wg_map(int wg, int H, int num_seqs, int n_blocks, int hg)
{
    const int per  = num_seqs * n_blocks;  // workgroups of a head
    const int full = H & -hg;              // heads in whole groups
    int h0 = wg / (hg * per) * hg, n = hg, local = wg % (hg * per);
    if(wg >= full * per)
    {
        h0    = full;
        n     = H - full;
        local = wg - full * per;
    }
    const int sb = local / n;
    int blk      = sb % n_blocks;
#ifndef GMLA_ZIG
#define GMLA_ZIG 1
#endif
#if GMLA_ZIG
    if((blk & 2) && (blk | 1) < n_blocks)
        blk ^= 1;
#endif
    return {h0 + local % n, sb / n_blocks, blk};
}

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

// The pair pass on 32x32x16 MFMAs (gated_mla_bwd_dkdv_pair32_kernel): as gated_mla_bwd_pair, query
// tiles laid out for that MFMA's fragments as in gated_mla_bwd_dq32 (rows of Q 224 and of dO 160
// elements apart, 16-byte chunk c of row r stored at c ^ swz(r)).
namespace gated_mla_bwd_pair32 {
using gated_mla_bwd_pair::BK;
using gated_mla_bwd_pair::NW;
using gated_mla_bwd_pair::NP;
using gated_mla_bwd_pair::PR;
using gated_mla_bwd_pair::BQ;
using gated_mla_bwd_pair::D_QK;
using gated_mla_bwd_pair::D_V;
using gated_mla_bwd_pair::D_NOPE;
using gated_mla_bwd_pair::D_ROT;
using gated_mla_bwd_pair::BLOCK;
constexpr int NBUF = 5;
constexpr int LD_Q = D_QK + 32, LD_DO = D_V + 32;
constexpr int OFF_Q = 0, OFF_DO = OFF_Q + BQ * LD_Q, OFF_LSE = OFF_DO + BQ * LD_DO, OFF_D = OFF_LSE + 128;
constexpr int SZ_BUF     = OFF_D + 128;
constexpr int OFF_P      = NBUF * SZ_BUF;
constexpr int SMEM_ELEMS = OFF_P + NP * PR * BQ * 2;
} // namespace gated_mla_bwd_pair32

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
    int hg;          // heads per scheduling group (gated_mla_bwd::wg_heads)
    int stride_q_n, stride_q_h;
    int stride_k_n, stride_k_h;
    int stride_v_n, stride_v_h;
    int stride_k_rot_n;
    int stride_do_n, stride_do_h;
    int stride_lse_h;  // LSE and D share [H, total_q]
    int stride_dq_n, stride_dq_h;
    float softmax_scale;
    int causal;
    // dQ from dS only (gated_mla_bwd_dq_ds_kernel): the pair pass's dS tiles, each sequence's first.
    const void* __restrict__ ptr_ds;
    const long long* __restrict__ ptr_ds_seq;
};

// dS as the pair pass keeps it for the dQ pass: per sequence, then per head of the launch, then per
// key tile j of 32 keys, the 32 x 32 tiles of query tiles first_q_tile(j) .. T - 1 in order (the
// ones causal masking leaves). A tile is 32 key rows of 64 bytes; row r holds key r's dS against the
// tile's queries, query 16 a + 4 b + c at element 8 b + 4 a + c (how the pair pass's registers hold
// them). Bottom-right causal: query qp sees keys kv <= qp + (seqlen_k - seqlen_q).
namespace gated_mla_bwd_ds {
constexpr int TILE = 32 * 32;  // bf16 elements per tile
__host__ __device__ constexpr int floor_div32(int a) { return a >= 0 ? a / 32 : -((31 - a) / 32); }
// Key tile j's first query tile, of T.
__host__ __device__ inline int first_q_tile(int j, int T, int co, int causal)
{
    if(!causal)
        return 0;
    const int i = j + floor_div32(-co);
    return i < 0 ? 0 : (i > T ? T : i);
}
// sum over u = 0 .. n - 1 of clamp(u, 0, T).
__host__ __device__ inline long long clamp_sum(long long n, long long T)
{
    if(n <= 0)
        return 0;
    if(n <= T + 1)
        return n * (n - 1) / 2;
    return T * (T + 1) / 2 + (n - T - 1) * T;
}
// Key tile j's first tile within its (sequence, head) block: sum over j' < j of T - first_q_tile(j').
__host__ __device__ inline long long col_offset(int j, int T, int co, int causal)
{
    if(!causal)
        return static_cast<long long>(j) * T;
    const long long s = floor_div32(-co);
    return static_cast<long long>(j) * T - (clamp_sum(j + s, T) - clamp_sum(s, T));
}
__host__ __device__ inline long long tiles_per_head(int seqlen_q, int seqlen_k, int causal)
{
    return col_offset((seqlen_k + 31) / 32, (seqlen_q + 31) / 32, seqlen_k - seqlen_q, causal);
}
} // namespace gated_mla_bwd_ds

namespace gated_mla_bwd_dq {
constexpr int NW = 8;          // waves
constexpr int WQ = 32;         // queries per wave
constexpr int BQ = NW * WQ;    // queries per workgroup
constexpr int KC = 32;         // keys per tile
constexpr int D_QK = 192, D_V = 128, D_NOPE = 128, D_ROT = 64;
constexpr int BLOCK = NW * 64;
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

// dQ = dS K from the pair pass's dS (gated_mla_bwd_dq_ds_kernel): the dQ pass's query blocks and
// waves; per key tile, K as in gated_mla_bwd_dq (B of dQ, transposed reads) and each wave's dS
// tile, its 16-byte chunks swizzled within a row (chunk c of row r at c ^ (r / 4 % 4)) so that the
// transposed reads that make dS the A operand are conflict-free.
namespace gated_mla_bwd_dqds {
using gated_mla_bwd_dq::NW;
using gated_mla_bwd_dq::WQ;
using gated_mla_bwd_dq::BQ;
using gated_mla_bwd_dq::KC;
using gated_mla_bwd_dq::D_QK;
using gated_mla_bwd_dq::D_NOPE;
using gated_mla_bwd_dq::D_ROT;
using gated_mla_bwd_dq::BLOCK;
using gated_mla_bwd_dq::LD_KN;
using gated_mla_bwd_dq::LD_KR;
constexpr int NBUF   = 5;
constexpr int OFF_KN = 0, OFF_KR = OFF_KN + KC * LD_KN, OFF_DS = OFF_KR + KC * LD_KR;
constexpr int SZ_BUF     = OFF_DS + NW * gated_mla_bwd_ds::TILE;
constexpr int SMEM_ELEMS = NBUF * SZ_BUF;
} // namespace gated_mla_bwd_dqds

// The dQ pass on 32x32x16 MFMAs (gated_mla_bwd_dq32_kernel): tiles and stages as
// gated_mla_bwd_dq, LDS laid out for that MFMA's fragments, conflict-free. Its A fragment reads
// (lane: row l % 32, one 16-byte chunk) and its transposed K reads (a 16-lane group: 4 rows x
// 32 bytes, two groups the same rows) conflict on every padding that whole 1 KB DMA loads allow,
// so key tile rows are 160 / 96 elements apart (the transposed reads' rows 64 bytes apart in
// the banks) and the 16-byte chunks of row r are swizzled: chunk c is stored at c ^ swz(r),
// swz(r) = bit 2 of r | bit 4 of r << 1. sQR, read by A-fragment reads only, needs neither.
namespace gated_mla_bwd_dq32 {
using gated_mla_bwd_dq::NW;
using gated_mla_bwd_dq::WQ;
using gated_mla_bwd_dq::BQ;
using gated_mla_bwd_dq::KC;
using gated_mla_bwd_dq::D_QK;
using gated_mla_bwd_dq::D_V;
using gated_mla_bwd_dq::D_NOPE;
using gated_mla_bwd_dq::D_ROT;
using gated_mla_bwd_dq::BLOCK;
constexpr int NBUF = 4;
constexpr int LD_KN = D_NOPE + 32, LD_KR = D_ROT + 32, LD_V = D_V + 32, LD_QR = D_ROT + 8;
constexpr int OFF_KN = 0, OFF_KR = OFF_KN + KC * LD_KN, OFF_V = OFF_KR + KC * LD_KR;
constexpr int SZ_BUF     = OFF_V + KC * LD_V;
constexpr int OFF_QR     = NBUF * SZ_BUF;
constexpr int SMEM_ELEMS = OFF_QR + BQ * LD_QR;
} // namespace gated_mla_bwd_dq32
