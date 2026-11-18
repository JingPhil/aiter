// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Device code of the gated MLA backward on gfx950 (bf16, D_QK = 192, D_V = 128).
#pragma once

#include <hip/hip_runtime.h>

#include "gated_mla_bwd_opus_defs.h"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#include <opus/opus.hpp>

namespace gated_mla_bwd {

// Same expression as the forward epilogue, so the backward differentiates the forward's
// own rounding of the gate.
__device__ inline float sigmoid(float g)
{
    constexpr float LOG2_E = 1.44269504089f;
    return __builtin_amdgcn_rcpf(1.0f + __builtin_amdgcn_exp2f(-LOG2_E * g));
}

} // namespace gated_mla_bwd

// One row of D_V = 128 bf16 per 16 lanes (8 per lane, one 16-byte access each).
__global__ __launch_bounds__(GATED_MLA_BWD_PRE_BLOCK) void
gated_mla_bwd_pre_kernel(gated_mla_bwd_pre_kargs k)
{
    using bf16_t = opus::bf16_t;
    constexpr int D_V   = 128;
    constexpr int VEC   = 8;
    constexpr int LANES = D_V / VEC;
    constexpr int ROWS  = GATED_MLA_BWD_PRE_BLOCK / LANES;

    const int tid = static_cast<int>(opus::thread_id_x());
    const long long row = static_cast<long long>(opus::block_id_x()) * ROWS + tid / LANES;
    const int col = (tid % LANES) * VEC;
    const bool valid = row < static_cast<long long>(k.total) * k.H;
    const int t = static_cast<int>(valid ? row / k.H : 0);
    const int h = static_cast<int>(valid ? row % k.H : 0);

    float acc = 0.0f;
    if(valid)
    {
        auto at = [&](const void* p, int sn, int sh) {
            return reinterpret_cast<const bf16_t*>(p) + static_cast<long long>(t) * sn +
                   static_cast<long long>(h) * sh + col;
        };
        using vec_t = opus::vector_t<bf16_t, VEC>;
        auto dout = opus::cast<float>(*reinterpret_cast<const vec_t*>(at(k.ptr_dout, k.stride_dout_n, k.stride_dout_h)));
        auto out  = opus::cast<float>(*reinterpret_cast<const vec_t*>(at(k.ptr_out, k.stride_out_n, k.stride_out_h)));
        auto gate = opus::cast<float>(*reinterpret_cast<const vec_t*>(at(k.ptr_gate, k.stride_gate_n, k.stride_gate_h)));
        opus::vector_t<float, VEC> v_do, v_dgate;
        for(int i = 0; i < VEC; i++)
        {
            const float s    = gated_mla_bwd::sigmoid(gate[i]);
            const float d_o  = dout[i] * out[i];
            v_do[i]    = dout[i] * s;
            v_dgate[i] = d_o * (1.0f - s);
            acc += d_o;
        }
        auto wr = [&](void* p, int sn, int sh) {
            return reinterpret_cast<bf16_t*>(p) + static_cast<long long>(t) * sn +
                   static_cast<long long>(h) * sh + col;
        };
        *reinterpret_cast<vec_t*>(wr(k.ptr_do, k.stride_do_n, k.stride_do_h)) = opus::cast<bf16_t>(v_do);
        *reinterpret_cast<vec_t*>(wr(k.ptr_dgate, k.stride_dgate_n, k.stride_dgate_h)) =
            opus::cast<bf16_t>(v_dgate);
    }
    // A row's 16 lanes are aligned within the wave, so xor offsets below 16 stay in the row.
    const int lane = tid % opus::get_warp_size();
    for(int m = LANES / 2; m > 0; m >>= 1)
        acc += opus::shfl(acc, lane ^ m);
    if(valid && tid % LANES == 0)
        k.ptr_d[static_cast<long long>(h) * k.stride_d_h + t] = acc;
}

// Every GEMM is per wave with 16x16x32 bf16 MFMAs, C = A * B^T. P^T, dS^T and dS go through
// LDS between GEMMs, and Q^T, dO^T and K^T are kept as transposed LDS copies, so each
// operand is read in its natural layout.
__global__ __launch_bounds__(gated_mla_bwd::BLOCK, 1) void gated_mla_bwd_kernel(gated_mla_bwd_kargs k)
{
    using namespace gated_mla_bwd;
    using opus::operator""_I;
    using bf16_t = opus::bf16_t;
    using vec8_t = opus::vector_t<bf16_t, 8>;
    constexpr float LOG2_E = 1.44269504089f;
    __shared__ __attribute__((aligned(16))) bf16_t smem[SMEM_ELEMS];

    const int tid  = static_cast<int>(opus::thread_id_x());
    const int lane = tid % 64;
    const int w    = __builtin_amdgcn_readfirstlane(tid / 64);
    const int j    = static_cast<int>(opus::block_id_x());
    const int h    = static_cast<int>(opus::block_id_y());
    const int g    = static_cast<int>(opus::block_id_z());
    auto sc        = [](int x) { return __builtin_amdgcn_readfirstlane(x); };
    const int q0       = sc(k.ptr_seqstart_q[g]);
    const int k0       = sc(k.ptr_seqstart_k[g]);
    const int seqlen_q = sc(k.ptr_seqstart_q[g + 1]) - q0;
    const int seqlen_k = sc(k.ptr_seqstart_k[g + 1]) - k0;
    const int kv0      = j * BK;
    if(kv0 >= seqlen_k)
        return;
    // Bottom-right causal: query qp sees keys kv <= qp + causal_offset.
    const int causal_offset = seqlen_k - seqlen_q;
    const int n_q_tiles     = (seqlen_q + BQ - 1) / BQ;
    int i_begin             = 0;
    if(k.causal && kv0 - causal_offset > 0)
        i_begin = (kv0 - causal_offset) / BQ;

    auto make_mma = [](auto e_n, auto e_k) {
        return opus::make_tiled_mma<bf16_t, bf16_t, float>(
            opus::seq<1, decltype(e_n)::value, decltype(e_k)::value>{},
            opus::seq<1, 1, 1>{},
            opus::seq<16, 16, 32>{},
            opus::mfma_adaptor_swap_ab{});
    };
    constexpr int DQ_N = D_QK / 4;  // dQ: the 8 waves split the tile into 2 x 16 queries by 4 x 48 dims
    auto mma_s   = make_mma(opus::number<BQ / 16>{}, opus::number<D_QK / 32>{});   // S^T  [16 x BQ]
    auto mma_kn  = make_mma(opus::number<BQ / 16>{}, opus::number<D_NOPE / 32>{}); // A of S^T, nope part
    auto mma_kr  = make_mma(opus::number<BQ / 16>{}, opus::number<D_ROT / 32>{});  // A of S^T, rope part
    auto mma_dp  = make_mma(opus::number<BQ / 16>{}, opus::number<D_V / 32>{});    // dP^T [16 x BQ]
    auto mma_dv  = make_mma(opus::number<D_V / 16>{}, opus::number<BQ / 32>{});    // dV   [16 x 128]
    auto mma_dk  = make_mma(opus::number<D_QK / 16>{}, opus::number<BQ / 32>{});   // dK   [16 x 192]
    auto mma_dkr = make_mma(opus::number<D_ROT / 16>{}, opus::number<BQ / 32>{});  // C of dK, rope part
    auto mma_dq  = make_mma(opus::number<DQ_N / 16>{}, opus::number<BK / 32>{});   // dQ   [16 x 48]
    const auto c_ab = opus::make_tuple(0_I, lane % mma_s.grpm_a, 0_I, lane / mma_s.grpm_a);
    const auto c_c  = opus::make_tuple(0_I, lane % mma_s.grpn_c, 0_I, lane / mma_s.grpn_c);

    auto rec = [](long long elems, int elem_bytes) -> unsigned int {
        const long long b = elems > 0 ? elems * elem_bytes : 0;
        return b >= 0xffffffffLL ? 0xffffffffu : static_cast<unsigned int>(b);
    };

    // This wave's 16 keys as the A operand of S^T and dP^T; rows past seqlen_k read as 0.
    const int row_w  = kv0 + w * WR;
    const int rows_w = seqlen_k - row_w < WR ? (seqlen_k - row_w > 0 ? seqlen_k - row_w : 0) : WR;
    const bf16_t* k_nope = reinterpret_cast<const bf16_t*>(k.ptr_k_nope);
    const bf16_t* k_rot  = reinterpret_cast<const bf16_t*>(k.ptr_k_rot);
    auto g_kn = opus::make_gmem(k_nope + static_cast<long long>(k0 + row_w) * k.stride_k_n +
                                    static_cast<long long>(h) * k.stride_k_h,
                                rec(static_cast<long long>(rows_w) * k.stride_k_n, 2));
    auto g_kr = opus::make_gmem(k_rot + static_cast<long long>(k0 + row_w) * k.stride_k_rot_n,
                                rec(static_cast<long long>(rows_w) * k.stride_k_rot_n, 2));
    auto g_v  = opus::make_gmem(reinterpret_cast<const bf16_t*>(k.ptr_v) +
                                    static_cast<long long>(k0 + row_w) * k.stride_v_n +
                                    static_cast<long long>(h) * k.stride_v_h,
                                rec(static_cast<long long>(rows_w) * k.stride_v_n, 2));
    auto v_kn = g_kn.template load<8>(opus::partition_layout_a<8>(mma_kn, opus::make_tuple(k.stride_k_n, 1_I), c_ab));
    auto v_kr = g_kr.template load<8>(opus::partition_layout_a<8>(mma_kr, opus::make_tuple(k.stride_k_rot_n, 1_I), c_ab));
    auto v_vw = g_v.template load<8>(opus::partition_layout_a<8>(mma_dp, opus::make_tuple(k.stride_v_n, 1_I), c_ab));
    typename decltype(mma_s)::vtype_a v_kw;
    constexpr int KN_LEN = opus::vector_traits<decltype(v_kn)>::size();
    constexpr int KR_LEN = opus::vector_traits<decltype(v_kr)>::size();
    for(int e = 0; e < KN_LEN; e++)
        v_kw[e] = v_kn[e];
    for(int e = 0; e < KR_LEN; e++)
        v_kw[KN_LEN + e] = v_kr[e];

    // K^T of the whole block, the B operand of dQ = dS K.
    for(int c = tid; c < BK * D_QK / 8; c += BLOCK)
    {
        const int r = c / (D_QK / 8), c8 = (c % (D_QK / 8)) * 8, kv = kv0 + r;
        vec8_t v;
        if(kv < seqlen_k)
            v = *reinterpret_cast<const vec8_t*>(
                c8 < D_NOPE ? k_nope + static_cast<long long>(k0 + kv) * k.stride_k_n +
                                  static_cast<long long>(h) * k.stride_k_h + c8
                            : k_rot + static_cast<long long>(k0 + kv) * k.stride_k_rot_n + c8 - D_NOPE);
        else
            for(int e = 0; e < 8; e++)
                v[e] = bf16_t(0.0f);
        for(int e = 0; e < 8; e++)
            smem[OFF_KT + (c8 + e) * LD_KV + r] = v[e];
    }

    auto s_q     = opus::make_smem(smem + OFF_Q);
    auto s_qt    = opus::make_smem(smem + OFF_QT);
    auto s_do    = opus::make_smem(smem + OFF_DO);
    auto s_dot   = opus::make_smem(smem + OFF_DOT);
    auto s_pt_w  = opus::make_smem(smem + OFF_PT + w * WR * LD_T);
    auto s_dst_w = opus::make_smem(smem + OFF_DST + w * WR * LD_T);
    auto s_ds_w  = opus::make_smem(smem + OFF_DS + w * WR);
    auto s_ds_q  = opus::make_smem(smem + OFF_DS + (w / 4) * 16 * LD_KV);
    auto s_kt_q  = opus::make_smem(smem + OFF_KT + (w % 4) * DQ_N * LD_KV);

    const auto u_q    = opus::partition_layout_b<8>(mma_s, opus::make_tuple(opus::number<LD_Q>{}, 1_I), c_ab);
    const auto u_do   = opus::partition_layout_b<8>(mma_dp, opus::make_tuple(opus::number<LD_DO>{}, 1_I), c_ab);
    const auto u_ct   = opus::partition_layout_c(mma_s, opus::make_tuple(opus::number<LD_T>{}, 1_I), c_c);
    const auto u_ctt  = opus::partition_layout_c(mma_s, opus::make_tuple(1_I, opus::number<LD_KV>{}), c_c);
    const auto u_a_dv = opus::partition_layout_a<8>(mma_dv, opus::make_tuple(opus::number<LD_T>{}, 1_I), c_ab);
    const auto u_a_dk = opus::partition_layout_a<8>(mma_dk, opus::make_tuple(opus::number<LD_T>{}, 1_I), c_ab);
    const auto u_b_dv = opus::partition_layout_b<8>(mma_dv, opus::make_tuple(opus::number<LD_T>{}, 1_I), c_ab);
    const auto u_b_dk = opus::partition_layout_b<8>(mma_dk, opus::make_tuple(opus::number<LD_T>{}, 1_I), c_ab);
    const auto u_a_dq = opus::partition_layout_a<8>(mma_dq, opus::make_tuple(opus::number<LD_KV>{}, 1_I), c_ab);
    const auto u_b_dq = opus::partition_layout_b<8>(mma_dq, opus::make_tuple(opus::number<LD_KV>{}, 1_I), c_ab);
    // (row, col) of each C element, encoded as row * 256 + col.
    const auto pos_s  = opus::layout_to_offsets<1>(
        opus::partition_layout_c(mma_s, opus::make_tuple(opus::number<256>{}, 1_I), c_c));
    const auto pos_dq = opus::layout_to_offsets<1>(
        opus::partition_layout_c(mma_dq, opus::make_tuple(opus::number<256>{}, 1_I), c_c));

    typename decltype(mma_dv)::vtype_c v_dv;
    typename decltype(mma_dk)::vtype_c v_dk;
    opus::clear(v_dv);
    opus::clear(v_dk);
    const float scale_log2 = k.softmax_scale * LOG2_E;
    const bf16_t* q_ptr  = reinterpret_cast<const bf16_t*>(k.ptr_q);
    const bf16_t* do_ptr = reinterpret_cast<const bf16_t*>(k.ptr_do);

    for(int i = i_begin; i < n_q_tiles; i++)
    {
        const int qt0 = i * BQ;
        __syncthreads();
        for(int c = tid; c < BQ * D_QK / 8; c += BLOCK)
        {
            const int r = c / (D_QK / 8), c8 = (c % (D_QK / 8)) * 8;
            vec8_t v;
            if(qt0 + r < seqlen_q)
                v = *reinterpret_cast<const vec8_t*>(q_ptr + static_cast<long long>(q0 + qt0 + r) * k.stride_q_n +
                                                     static_cast<long long>(h) * k.stride_q_h + c8);
            else
                for(int e = 0; e < 8; e++)
                    v[e] = bf16_t(0.0f);
            *reinterpret_cast<vec8_t*>(smem + OFF_Q + r * LD_Q + c8) = v;
            for(int e = 0; e < 8; e++)
                smem[OFF_QT + (c8 + e) * LD_T + r] = v[e];
        }
        for(int c = tid; c < BQ * D_V / 8; c += BLOCK)
        {
            const int r = c / (D_V / 8), c8 = (c % (D_V / 8)) * 8;
            vec8_t v;
            if(qt0 + r < seqlen_q)
                v = *reinterpret_cast<const vec8_t*>(do_ptr + static_cast<long long>(q0 + qt0 + r) * k.stride_do_n +
                                                     static_cast<long long>(h) * k.stride_do_h + c8);
            else
                for(int e = 0; e < 8; e++)
                    v[e] = bf16_t(0.0f);
            *reinterpret_cast<vec8_t*>(smem + OFF_DO + r * LD_DO + c8) = v;
            for(int e = 0; e < 8; e++)
                smem[OFF_DOT + (c8 + e) * LD_T + r] = v[e];
        }
        __syncthreads();

        auto v_st  = mma_s(v_kw, s_q.template load<8>(u_q));
        auto v_dpt = mma_dp(v_vw, s_do.template load<8>(u_do));
        constexpr int NS = opus::vector_traits<decltype(v_st)>::size();
        opus::vector_t<float, NS> v_p, v_ds;
        for(int e = 0; e < NS; e++)
        {
            const int kv = row_w + pos_s[e] / 256, qp = qt0 + pos_s[e] % 256;
            const bool ok = kv < seqlen_k && qp < seqlen_q && (!k.causal || qp + causal_offset >= kv);
            float p = 0.0f, ds = 0.0f;
            if(ok)
            {
                const long long li = static_cast<long long>(h) * k.stride_lse_h + q0 + qp;
                p  = __builtin_amdgcn_exp2f(v_st[e] * scale_log2 - k.ptr_lse[li] * LOG2_E);
                ds = p * (v_dpt[e] - k.ptr_d[li]) * k.softmax_scale;
            }
            v_p[e]  = p;
            v_ds[e] = ds;
        }
        const auto v_ds_bf = opus::cast<bf16_t>(v_ds);
        s_pt_w.template store<4>(opus::cast<bf16_t>(v_p), u_ct);
        s_dst_w.template store<4>(v_ds_bf, u_ct);
        s_ds_w.template store<1>(v_ds_bf, u_ctt);
        __syncthreads();

        v_dv = mma_dv(s_pt_w.template load<8>(u_a_dv), s_dot.template load<8>(u_b_dv), v_dv);
        v_dk = mma_dk(s_dst_w.template load<8>(u_a_dk), s_qt.template load<8>(u_b_dk), v_dk);
        auto v_dq = mma_dq(s_ds_q.template load<8>(u_a_dq), s_kt_q.template load<8>(u_b_dq));
        constexpr int NDQ = opus::vector_traits<decltype(v_dq)>::size();
        for(int e = 0; e < NDQ; e++)
        {
            const int qp = qt0 + (w / 4) * 16 + pos_dq[e] / 256;
            if(qp < seqlen_q)
                unsafeAtomicAdd(k.ptr_dq_acc + static_cast<long long>(q0 + qp) * k.stride_dq_n +
                                    static_cast<long long>(h) * k.stride_dq_h + (w % 4) * DQ_N + pos_dq[e] % 256,
                                v_dq[e]);
        }
    }

    // dK nope and dV into the kv_b_proj layout, dK rope per head in fp32; rows past seqlen_k
    // fall outside the descriptors and are dropped.
    bf16_t* dkv = reinterpret_cast<bf16_t*>(k.ptr_dkv) + static_cast<long long>(k0 + row_w) * k.stride_dkv_n +
                  static_cast<long long>(h) * k.stride_dkv_h;
    const long long dkv_elems = static_cast<long long>(rows_w) * k.stride_dkv_n;
    auto g_dkn = opus::make_gmem(dkv, rec(dkv_elems, 2));
    auto g_dv  = opus::make_gmem(dkv + D_NOPE, rec(dkv_elems - D_NOPE, 2));
    auto g_dkr = opus::make_gmem(k.ptr_dk_rope + static_cast<long long>(k0 + row_w) * k.stride_dkr_n +
                                     static_cast<long long>(h) * k.stride_dkr_h,
                                 rec(static_cast<long long>(rows_w) * k.stride_dkr_n, 4));
    const auto u_dkv = opus::partition_layout_c(mma_dv, opus::make_tuple(k.stride_dkv_n, 1_I), c_c);
    const auto u_dkr = opus::partition_layout_c(mma_dkr, opus::make_tuple(k.stride_dkr_n, 1_I), c_c);
    constexpr int DK_LEN  = opus::vector_traits<decltype(v_dk)>::size();
    constexpr int DKN_LEN = DK_LEN * D_NOPE / D_QK;
    g_dkn.template store<4>(opus::cast<bf16_t>(opus::slice(v_dk, opus::number<0>{}, opus::number<DKN_LEN>{})), u_dkv);
    g_dv.template store<4>(opus::cast<bf16_t>(v_dv), u_dkv);
    g_dkr.template store<4>(opus::slice(v_dk, opus::number<DKN_LEN>{}, opus::number<DK_LEN>{}), u_dkr);
}

#else
__global__ void gated_mla_bwd_pre_kernel(gated_mla_bwd_pre_kargs) {}
__global__ void gated_mla_bwd_kernel(gated_mla_bwd_kargs) {}
#endif
