// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Gated MLA backward (Kimi-K3 full-attention layers) on gfx950, bf16, D_QK = 192 / D_V = 128.
#pragma once
#include <torch/extension.h>

// Pre-pass of the backward, per (token, head) row of 128:
//   s = sigmoid(gate)            (the forward epilogue's expression)
//   d_o   = dout * s             [total, H, 128] bf16, the attention output gradient
//   dgate = dout * out * (1 - s) [total, H, 128] bf16, the gate gradient
//   d     = rowsum(dout * out)   [H, total] fp32, equal to rowsum(dO * O) of the ungated O
// `out` is the gated forward output. All bf16 tensors are contiguous along the head dim with
// 16-byte aligned rows; d is contiguous along total.
void gated_mla_bwd_pre_bf16_opus(at::Tensor& dout,
                                 at::Tensor& out,
                                 at::Tensor& gate,
                                 at::Tensor& d_o,
                                 at::Tensor& dgate,
                                 at::Tensor& d);

// Main pass, group / varlen, H_KV == H. K is read like the forward reads it: dims 0..127
// from k_nope, dims 128..191 from k_rot, shared by every head.
//   q      [total_q, H, 192]   k_nope, v [total_k, H, 128] (may be views of one kv_b_proj
//   k_rot  [total_k, 64]       output)   d_o [total_q, H, 128]   lse, d [H, total_q] fp32
// Outputs:
//   dq_acc  [total_q, H, 192] fp32, accumulated into with atomics, so zero it first
//   dkv     [total_k, H, 256] bf16, each head's row is [dK nope | dV]
//   dk_rope [total_k, H, 64]  fp32, the per-head rope part of dK; dk_rot is its sum over H
// dq_acc and dK include softmax_scale.
void gated_mla_bwd_bf16_opus(at::Tensor& q,
                             at::Tensor& k_nope,
                             at::Tensor& v,
                             at::Tensor& k_rot,
                             at::Tensor& d_o,
                             at::Tensor& lse,
                             at::Tensor& d,
                             at::Tensor& dq_acc,
                             at::Tensor& dkv,
                             at::Tensor& dk_rope,
                             at::Tensor& seqstart_q,
                             at::Tensor& seqstart_k,
                             int max_seqlen_k,
                             bool causal,
                             float softmax_scale);

// dK / dV only (4 waves x 48 keys on 32-query tiles): same arguments and outputs as the main
// pass minus dq_acc. Pair it with the dQ pass.
void gated_mla_bwd_dkdv_bf16_opus(at::Tensor& q,
                                  at::Tensor& k_nope,
                                  at::Tensor& v,
                                  at::Tensor& k_rot,
                                  at::Tensor& d_o,
                                  at::Tensor& lse,
                                  at::Tensor& d,
                                  at::Tensor& dkv,
                                  at::Tensor& dk_rope,
                                  at::Tensor& seqstart_q,
                                  at::Tensor& seqstart_k,
                                  int max_seqlen_k,
                                  bool causal,
                                  float softmax_scale);

// As gated_mla_bwd_dkdv_bf16_opus, on the 8-wave kernel whose wave pairs split by dependency chain.
void gated_mla_bwd_dkdv_pair_bf16_opus(at::Tensor& q,
                                       at::Tensor& k_nope,
                                       at::Tensor& v,
                                       at::Tensor& k_rot,
                                       at::Tensor& d_o,
                                       at::Tensor& lse,
                                       at::Tensor& d,
                                       at::Tensor& dkv,
                                       at::Tensor& dk_rope,
                                       at::Tensor& seqstart_q,
                                       at::Tensor& seqstart_k,
                                       int max_seqlen_k,
                                       bool causal,
                                       float softmax_scale);

// dQ pass, no atomics: inputs as the main pass, output dq [total_q, H, 192] bf16 (written,
// not accumulated), including softmax_scale. max_seqlen_q sizes the grid.
void gated_mla_bwd_dq_bf16_opus(at::Tensor& q,
                                at::Tensor& k_nope,
                                at::Tensor& v,
                                at::Tensor& k_rot,
                                at::Tensor& d_o,
                                at::Tensor& lse,
                                at::Tensor& d,
                                at::Tensor& dq,
                                at::Tensor& seqstart_q,
                                at::Tensor& seqstart_k,
                                int max_seqlen_q,
                                bool causal,
                                float softmax_scale);
