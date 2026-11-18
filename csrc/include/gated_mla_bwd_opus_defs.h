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

// Main pass, one workgroup per (KV block of 128 keys, head, sequence), group / varlen only.
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
constexpr int BK = 128;  // keys per workgroup
constexpr int NW = 8;    // waves; wave w owns keys [16 w, 16 w + 16)
constexpr int WR = BK / NW;
constexpr int BQ = 32;   // queries per tile
constexpr int D_QK = 192, D_V = 128, D_NOPE = 128, D_ROT = 64;
constexpr int BLOCK = NW * 64;

// LDS, bf16 elements. Row strides carry 8 elements of padding.
constexpr int LD_Q  = D_QK + 8;  // sQ   [BQ][D_QK]   B of S^T
constexpr int LD_T  = BQ + 8;    // sQT  [D_QK][BQ]   B of dK;  sDOT [D_V][BQ] B of dV
                                 // sPT, sDST [BK][BQ] A of dV, dK
constexpr int LD_DO = D_V + 8;   // sDO  [BQ][D_V]    B of dP^T
constexpr int LD_KV = BK + 8;    // sDS  [BQ][BK]     A of dQ;  sKT [D_QK][BK] B of dQ
constexpr int OFF_Q   = 0;
constexpr int OFF_QT  = OFF_Q + BQ * LD_Q;
constexpr int OFF_DO  = OFF_QT + D_QK * LD_T;
constexpr int OFF_DOT = OFF_DO + BQ * LD_DO;
constexpr int OFF_PT  = OFF_DOT + D_V * LD_T;
constexpr int OFF_DST = OFF_PT + BK * LD_T;
constexpr int OFF_DS  = OFF_DST + BK * LD_T;
constexpr int OFF_KT  = OFF_DS + BQ * LD_KV;
constexpr int SMEM_ELEMS = OFF_KT + D_QK * LD_KV;
} // namespace gated_mla_bwd
