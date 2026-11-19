// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Torch entry points of the gated MLA backward (gfx950, bf16, D_QK = 192 / D_V = 128).

#include "gated_mla_bwd_opus_kernel.hpp"
#include "torch/gated_mla_bwd_opus.h"
#include "aiter_hip_common.h"

#include <ATen/hip/HIPContext.h>

namespace {

void check_rows(const at::Tensor& t, const at::Tensor& like, const char* name)
{
    TORCH_CHECK(t.device() == like.device(), name, " must be on dout's device");
    TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be bf16");
    TORCH_CHECK(t.dim() == 3 && t.sizes() == like.sizes(), name, " must be [total, H, 128]");
    TORCH_CHECK(t.stride(-1) == 1 && t.stride(0) % 8 == 0 && t.stride(1) % 8 == 0 &&
                    reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                name, " rows must be contiguous and 16-byte aligned");
}

} // namespace

void gated_mla_bwd_pre_bf16_opus(at::Tensor& dout,
                                 at::Tensor& out,
                                 at::Tensor& gate,
                                 at::Tensor& d_o,
                                 at::Tensor& dgate,
                                 at::Tensor& d)
{
    TORCH_CHECK(dout.is_cuda(), "dout must be a GPU tensor");
    TORCH_CHECK(dout.dim() == 3 && dout.size(2) == 128, "dout must be [total, H, 128]");
    for(auto [t, name] : {std::pair<at::Tensor*, const char*>{&dout, "dout"},
                          {&out, "out"},
                          {&gate, "gate"},
                          {&d_o, "d_o"},
                          {&dgate, "dgate"}})
        check_rows(*t, dout, name);
    const int total = static_cast<int>(dout.size(0));
    const int H     = static_cast<int>(dout.size(1));
    TORCH_CHECK(d.device() == dout.device() && d.scalar_type() == at::kFloat && d.dim() == 2 &&
                    d.size(0) == H && d.size(1) == total && d.stride(1) == 1,
                "d must be float32 [H, total], contiguous along total");

    gated_mla_bwd_pre_kargs k{};
    k.ptr_dout  = dout.data_ptr();
    k.ptr_out   = out.data_ptr();
    k.ptr_gate  = gate.data_ptr();
    k.ptr_do    = d_o.data_ptr();
    k.ptr_dgate = dgate.data_ptr();
    k.ptr_d     = d.data_ptr<float>();
    k.total     = total;
    k.H         = H;
    k.stride_dout_n  = static_cast<int>(dout.stride(0));  k.stride_dout_h  = static_cast<int>(dout.stride(1));
    k.stride_out_n   = static_cast<int>(out.stride(0));   k.stride_out_h   = static_cast<int>(out.stride(1));
    k.stride_gate_n  = static_cast<int>(gate.stride(0));  k.stride_gate_h  = static_cast<int>(gate.stride(1));
    k.stride_do_n    = static_cast<int>(d_o.stride(0));   k.stride_do_h    = static_cast<int>(d_o.stride(1));
    k.stride_dgate_n = static_cast<int>(dgate.stride(0)); k.stride_dgate_h = static_cast<int>(dgate.stride(1));
    k.stride_d_h     = static_cast<int>(d.stride(0));
    if(total == 0 || H == 0)
        return;

    constexpr int ROWS = GATED_MLA_BWD_PRE_BLOCK / (128 / 8);
    const long long rows = static_cast<long long>(total) * H;
    HipDeviceGuard guard(dout.device().index());
    gated_mla_bwd_pre_kernel<<<dim3(static_cast<unsigned>((rows + ROWS - 1) / ROWS)),
                               dim3(GATED_MLA_BWD_PRE_BLOCK),
                               0,
                               at::hip::getCurrentHIPStream()>>>(k);
    HIP_CALL_LAUNCH(hipGetLastError());
}

namespace {

// Main pass; without dq_acc it skips dQ, which the dQ pass then computes.
void launch_main(at::Tensor& q,
                 at::Tensor& k_nope,
                 at::Tensor& v,
                 at::Tensor& k_rot,
                 at::Tensor& d_o,
                 at::Tensor& lse,
                 at::Tensor& d,
                 at::Tensor* dq_acc,
                 at::Tensor& dkv,
                 at::Tensor& dk_rope,
                 at::Tensor& seqstart_q,
                 at::Tensor& seqstart_k,
                 int max_seqlen_k,
                 bool causal,
                 float softmax_scale)
{
    using namespace gated_mla_bwd;
    TORCH_CHECK(q.is_cuda(), "q must be a GPU tensor");
    const int total_q = static_cast<int>(q.size(0));
    const int H       = static_cast<int>(q.size(1));
    const int total_k = static_cast<int>(k_nope.size(0));
    // `vec` is the widest access, in elements: rows and strides must keep it aligned.
    auto check = [&](const at::Tensor& t, at::ScalarType dt, std::initializer_list<int64_t> shape,
                     int vec, const char* name) {
        TORCH_CHECK(t.device() == q.device(), name, " must be on q's device");
        TORCH_CHECK(t.scalar_type() == dt, name, " has the wrong dtype");
        TORCH_CHECK(t.sizes() == at::IntArrayRef(shape), name, " has the wrong shape");
        TORCH_CHECK(t.stride(-1) == 1, name, " must be contiguous along its last dim");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % (vec * t.element_size()) == 0,
                    name, " is not aligned for its vector accesses");
        for(int i = 0; i + 1 < t.dim(); i++)
            TORCH_CHECK(t.stride(i) % vec == 0, name, " strides must be multiples of ", vec);
    };
    check(q, at::kBFloat16, {total_q, H, D_QK}, 8, "q");
    check(k_nope, at::kBFloat16, {total_k, H, D_NOPE}, 8, "k_nope");
    check(v, at::kBFloat16, {total_k, H, D_V}, 8, "v");
    check(k_rot, at::kBFloat16, {total_k, D_ROT}, 8, "k_rot");
    check(d_o, at::kBFloat16, {total_q, H, D_V}, 8, "d_o");
    check(lse, at::kFloat, {H, total_q}, 1, "lse");
    check(d, at::kFloat, {H, total_q}, 1, "d");
    TORCH_CHECK(lse.strides() == d.strides(), "lse and d must share strides");
    if(dq_acc)
        check(*dq_acc, at::kFloat, {total_q, H, D_QK}, 1, "dq_acc");
    check(dkv, at::kBFloat16, {total_k, H, D_NOPE + D_V}, 4, "dkv");
    check(dk_rope, at::kFloat, {total_k, H, D_ROT}, 4, "dk_rope");
    const int B = static_cast<int>(seqstart_q.numel()) - 1;
    TORCH_CHECK(B > 0 && seqstart_k.numel() == B + 1, "seqstart_q / seqstart_k must be [num_seqs + 1]");
    for(const at::Tensor* s : {&seqstart_q, &seqstart_k})
        TORCH_CHECK(s->device() == q.device() && s->scalar_type() == at::kInt && s->is_contiguous(),
                    "seqstart_q / seqstart_k must be contiguous int32 on q's device");
    TORCH_CHECK(max_seqlen_k > 0, "max_seqlen_k must be > 0");

    gated_mla_bwd_kargs a{};
    a.ptr_q          = q.data_ptr();
    a.ptr_k_nope     = k_nope.data_ptr();
    a.ptr_v          = v.data_ptr();
    a.ptr_k_rot      = k_rot.data_ptr();
    a.ptr_do         = d_o.data_ptr();
    a.ptr_lse        = lse.data_ptr<float>();
    a.ptr_d          = d.data_ptr<float>();
    a.ptr_dq_acc     = dq_acc ? dq_acc->data_ptr<float>() : nullptr;
    a.ptr_dkv        = dkv.data_ptr();
    a.ptr_dk_rope    = dk_rope.data_ptr<float>();
    a.ptr_seqstart_q = seqstart_q.data_ptr<int>();
    a.ptr_seqstart_k = seqstart_k.data_ptr<int>();
    a.H              = H;
    a.stride_q_n     = static_cast<int>(q.stride(0));       a.stride_q_h   = static_cast<int>(q.stride(1));
    a.stride_k_n     = static_cast<int>(k_nope.stride(0));  a.stride_k_h   = static_cast<int>(k_nope.stride(1));
    a.stride_v_n     = static_cast<int>(v.stride(0));       a.stride_v_h   = static_cast<int>(v.stride(1));
    a.stride_k_rot_n = static_cast<int>(k_rot.stride(0));
    a.stride_do_n    = static_cast<int>(d_o.stride(0));     a.stride_do_h  = static_cast<int>(d_o.stride(1));
    a.stride_lse_h   = static_cast<int>(lse.stride(0));
    if(dq_acc)
    {
        a.stride_dq_n = static_cast<int>(dq_acc->stride(0));
        a.stride_dq_h = static_cast<int>(dq_acc->stride(1));
    }
    a.stride_dkv_n   = static_cast<int>(dkv.stride(0));     a.stride_dkv_h = static_cast<int>(dkv.stride(1));
    a.stride_dkr_n   = static_cast<int>(dk_rope.stride(0)); a.stride_dkr_h = static_cast<int>(dk_rope.stride(1));
    a.softmax_scale  = softmax_scale;
    a.causal         = causal ? 1 : 0;
    if(total_q == 0 || total_k == 0 || H == 0)
        return;
    a.num_seqs    = B;
    a.n_kv_blocks = (max_seqlen_k + BK - 1) / BK;
    const long long n_wg = static_cast<long long>(HG) * a.n_kv_blocks * B * ((H + HG - 1) / HG);
    TORCH_CHECK(n_wg <= 0x7fffffffLL, "too many workgroups");

    HipDeviceGuard guard(q.device().index());
    if(dq_acc)
        gated_mla_bwd_kernel<true>
            <<<dim3(static_cast<unsigned>(n_wg)), dim3(BLOCK), 0, at::hip::getCurrentHIPStream()>>>(a);
    else
        gated_mla_bwd_kernel<false>
            <<<dim3(static_cast<unsigned>(n_wg)), dim3(BLOCK), 0, at::hip::getCurrentHIPStream()>>>(a);
    HIP_CALL_LAUNCH(hipGetLastError());
}

} // namespace

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
                             float softmax_scale)
{
    launch_main(q, k_nope, v, k_rot, d_o, lse, d, &dq_acc, dkv, dk_rope, seqstart_q, seqstart_k,
                max_seqlen_k, causal, softmax_scale);
}

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
                                  float softmax_scale)
{
    launch_main(q, k_nope, v, k_rot, d_o, lse, d, nullptr, dkv, dk_rope, seqstart_q, seqstart_k,
                max_seqlen_k, causal, softmax_scale);
}

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
                                float softmax_scale)
{
    using namespace gated_mla_bwd_dq;
    TORCH_CHECK(q.is_cuda(), "q must be a GPU tensor");
    const int total_q = static_cast<int>(q.size(0));
    const int H       = static_cast<int>(q.size(1));
    const int total_k = static_cast<int>(k_nope.size(0));
    auto check = [&](const at::Tensor& t, at::ScalarType dt, std::initializer_list<int64_t> shape,
                     int vec, const char* name) {
        TORCH_CHECK(t.device() == q.device(), name, " must be on q's device");
        TORCH_CHECK(t.scalar_type() == dt, name, " has the wrong dtype");
        TORCH_CHECK(t.sizes() == at::IntArrayRef(shape), name, " has the wrong shape");
        TORCH_CHECK(t.stride(-1) == 1, name, " must be contiguous along its last dim");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % (vec * t.element_size()) == 0,
                    name, " is not aligned for its vector accesses");
        for(int i = 0; i + 1 < t.dim(); i++)
            TORCH_CHECK(t.stride(i) % vec == 0, name, " strides must be multiples of ", vec);
    };
    check(q, at::kBFloat16, {total_q, H, D_QK}, 8, "q");
    check(k_nope, at::kBFloat16, {total_k, H, D_NOPE}, 8, "k_nope");
    check(v, at::kBFloat16, {total_k, H, D_V}, 8, "v");
    check(k_rot, at::kBFloat16, {total_k, D_ROT}, 8, "k_rot");
    check(d_o, at::kBFloat16, {total_q, H, D_V}, 8, "d_o");
    check(lse, at::kFloat, {H, total_q}, 1, "lse");
    check(d, at::kFloat, {H, total_q}, 1, "d");
    TORCH_CHECK(lse.strides() == d.strides(), "lse and d must share strides");
    check(dq, at::kBFloat16, {total_q, H, D_QK}, 4, "dq");
    const int B = static_cast<int>(seqstart_q.numel()) - 1;
    TORCH_CHECK(B > 0 && seqstart_k.numel() == B + 1, "seqstart_q / seqstart_k must be [num_seqs + 1]");
    for(const at::Tensor* s : {&seqstart_q, &seqstart_k})
        TORCH_CHECK(s->device() == q.device() && s->scalar_type() == at::kInt && s->is_contiguous(),
                    "seqstart_q / seqstart_k must be contiguous int32 on q's device");
    TORCH_CHECK(max_seqlen_q > 0, "max_seqlen_q must be > 0");

    gated_mla_bwd_dq_kargs a{};
    a.ptr_q          = q.data_ptr();
    a.ptr_k_nope     = k_nope.data_ptr();
    a.ptr_v          = v.data_ptr();
    a.ptr_k_rot      = k_rot.data_ptr();
    a.ptr_do         = d_o.data_ptr();
    a.ptr_lse        = lse.data_ptr<float>();
    a.ptr_d          = d.data_ptr<float>();
    a.ptr_dq         = dq.data_ptr();
    a.ptr_seqstart_q = seqstart_q.data_ptr<int>();
    a.ptr_seqstart_k = seqstart_k.data_ptr<int>();
    a.H              = H;
    a.stride_q_n     = static_cast<int>(q.stride(0));       a.stride_q_h  = static_cast<int>(q.stride(1));
    a.stride_k_n     = static_cast<int>(k_nope.stride(0));  a.stride_k_h  = static_cast<int>(k_nope.stride(1));
    a.stride_v_n     = static_cast<int>(v.stride(0));       a.stride_v_h  = static_cast<int>(v.stride(1));
    a.stride_k_rot_n = static_cast<int>(k_rot.stride(0));
    a.stride_do_n    = static_cast<int>(d_o.stride(0));     a.stride_do_h = static_cast<int>(d_o.stride(1));
    a.stride_lse_h   = static_cast<int>(lse.stride(0));
    a.stride_dq_n    = static_cast<int>(dq.stride(0));      a.stride_dq_h = static_cast<int>(dq.stride(1));
    a.softmax_scale  = softmax_scale;
    a.causal         = causal ? 1 : 0;
    if(total_q == 0 || H == 0)
        return;
    a.num_seqs   = B;
    a.n_q_blocks = (max_seqlen_q + BQ - 1) / BQ;
    const long long n_wg = static_cast<long long>(HG) * a.n_q_blocks * B * ((H + HG - 1) / HG);
    TORCH_CHECK(n_wg <= 0x7fffffffLL, "too many workgroups");

    HipDeviceGuard guard(q.device().index());
    gated_mla_bwd_dq_kernel<<<dim3(static_cast<unsigned>(n_wg)),
                              dim3(BLOCK),
                              0,
                              at::hip::getCurrentHIPStream()>>>(a);
    HIP_CALL_LAUNCH(hipGetLastError());
}
