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

using bf16_t = opus::bf16_t;
template <int N>
using bvec = opus::vector_t<bf16_t, N>;

// Same expression as the forward epilogue, so the backward differentiates the forward's
// own rounding of the gate.
__device__ inline float sigmoid(float g)
{
    constexpr float LOG2_E = 1.44269504089f;
    return __builtin_amdgcn_rcpf(1.0f + __builtin_amdgcn_exp2f(-LOG2_E * g));
}

// Transposed operand reads for 16x16x32 MFMAs from a row-major LDS tile whose rows run
// along the GEMM's K. ds_read_b64_tr_b16: each 16-lane group addresses a 4 x 16 block (lane
// i: row i / 4, columns 4 (i % 4) .. + 3) and lane i receives column i, 4 rows deep.
// These use the compiler builtin, not opus' inline-asm smem::tr_load: the compiler cannot see
// inline-asm LDS reads, so it neither waits for them nor keeps them under the 15 outstanding
// LDS operations lgkmcnt can count.
using u32x2_t = opus::vector_t<opus::u32_t, 2>;
using i16x4_t = opus::vector_t<short, 4>;

__device__ inline u32x2_t tr_read(opus::smem<bf16_t>& s, int elem_off)
{
    return __builtin_bit_cast(
        u32x2_t,
        __builtin_amdgcn_ds_read_tr16_b64_v4i16(
            reinterpret_cast<OPUS_LDS_ADDR i16x4_t*>(s.ptr + elem_off * static_cast<int>(sizeof(bf16_t)))));
}

template <int N>
__device__ inline bvec<N * 4> concat_tr_loads(const u32x2_t (&raw)[N])
{
    opus::vector_t<opus::u32_t, N * 2> r;
#pragma unroll
    for(int x = 0; x < N; x++)
    {
        r[2 * x]     = raw[x][0];
        r[2 * x + 1] = raw[x][1];
    }
    return __builtin_bit_cast(bvec<N * 4>, r);
}

// B fragments (N-tiles of 16 columns, K = 32 rows) in the K order of a swap_ab C fragment
// reused as the A operand: slot 4 h + e of lane group g is row 16 h + 4 g + e.
template <int NT, int LD>
__device__ inline bvec<NT * 8> tr_load_b_c_order(opus::smem<bf16_t>& s, int lane)
{
    const int g = lane / 16, i = lane % 16;
    const int base = (g * 4 + i / 4) * LD + (i % 4) * 4;
    u32x2_t raw[NT * 2];
#pragma unroll
    for(int tn = 0; tn < NT; tn++)
#pragma unroll
        for(int hh = 0; hh < 2; hh++)
            raw[tn * 2 + hh] = tr_read(s, base + hh * 16 * LD + tn * 16);
    return concat_tr_loads(raw);
}

// A or B fragments in the MFMA's own K order (slot j of lane group g is row 8 g + j) over
// NKS steps of 32 rows, for NT tiles of 16 columns starting at col0.
template <int NT, int NKS, int LD>
__device__ inline bvec<NT * NKS * 8> tr_load_std(opus::smem<bf16_t>& s, int lane, int col0)
{
    const int g = lane / 16, i = lane % 16;
    const int base = (g * 8 + i / 4) * LD + col0 + (i % 4) * 4;
    u32x2_t raw[NT * NKS * 2];
#pragma unroll
    for(int tn = 0; tn < NT; tn++)
#pragma unroll
        for(int ks = 0; ks < NKS; ks++)
#pragma unroll
            for(int hh = 0; hh < 2; hh++)
                raw[(tn * NKS + ks) * 2 + hh] = tr_read(s, base + (ks * 32 + hh * 4) * LD + tn * 16);
    return concat_tr_loads(raw);
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

// Every GEMM is per wave with 16x16x32 bf16 MFMAs, C = A * B^T. P^T and dS^T stay in
// registers as the A operand of dV and dK; their B operands (dO^T, Q^T) are transposed reads
// of the natural Q and dO tiles. dS^T is the one tile that goes through LDS, since dQ sums it
// over all 128 keys of the workgroup. The next query tile is fetched into registers during
// the current one and staged into the other LDS buffer, so each tile has one barrier.
__global__ __launch_bounds__(gated_mla_bwd::BLOCK, 1) void gated_mla_bwd_kernel(gated_mla_bwd_kargs k)
{
    using namespace gated_mla_bwd;
    using opus::operator""_I;
    using vec8_t = bvec<8>;
    constexpr float LOG2_E = 1.44269504089f;
    __shared__ __attribute__((aligned(16))) bf16_t smem[SMEM_ELEMS];
    // LSE and D of a query tile, [lse | D], double-buffered like sQ.
    __shared__ float s_ld[2][2 * BQ];

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
    // dQ^T [48 x 16] = K^T dS^T: lanes run along d, so each atomic instruction covers
    // contiguous dq_acc rows instead of 16 scattered ones.
    auto mma_dq = opus::make_tiled_mma<bf16_t, bf16_t, float>(
        opus::seq<DQ_N / 16, 1, BK / 32>{}, opus::seq<1, 1, 1>{}, opus::seq<16, 16, 32>{},
        opus::mfma_adaptor_swap_ab{});
    static_assert(BQ == 32, "the C -> A reuse of P^T / dS^T needs the query tile to be one K step");
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

    // The whole key block in its natural layout; dQ reads it transposed as K^T.
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
        *reinterpret_cast<vec8_t*>(smem + OFF_K + r * LD_K + c8) = v;
    }

    // Query tiles, with their LSE and D, go global -> registers -> LDS one tile ahead; rows
    // past seqlen_q are 0. A tile's global reads are all issued before the previous tile's
    // dQ atomics, so waiting for them never waits for the atomics.
    constexpr int QC = BQ * D_QK / 8, QR = (QC + BLOCK - 1) / BLOCK;
    static_assert(BQ * D_V / 8 == BLOCK, "one dO chunk per thread");
    const bf16_t* q_ptr  = reinterpret_cast<const bf16_t*>(k.ptr_q);
    const bf16_t* do_ptr = reinterpret_cast<const bf16_t*>(k.ptr_do);
    vec8_t r_q[QR], r_do;
    float r_ld = 0.0f;
    auto fetch = [&](int i) {
        const int qt0 = i * BQ;
#pragma unroll
        for(int u = 0; u < QR; u++)
        {
            const int c = tid + u * BLOCK, r = c / (D_QK / 8), c8 = (c % (D_QK / 8)) * 8;
            if(c < QC && qt0 + r < seqlen_q)
                r_q[u] = *reinterpret_cast<const vec8_t*>(q_ptr + static_cast<long long>(q0 + qt0 + r) * k.stride_q_n +
                                                          static_cast<long long>(h) * k.stride_q_h + c8);
            else
                for(int e = 0; e < 8; e++)
                    r_q[u][e] = bf16_t(0.0f);
        }
        const int r = tid / (D_V / 8), c8 = (tid % (D_V / 8)) * 8;
        if(qt0 + r < seqlen_q)
            r_do = *reinterpret_cast<const vec8_t*>(do_ptr + static_cast<long long>(q0 + qt0 + r) * k.stride_do_n +
                                                    static_cast<long long>(h) * k.stride_do_h + c8);
        else
            for(int e = 0; e < 8; e++)
                r_do[e] = bf16_t(0.0f);
        if(tid < 2 * BQ)
        {
            const int qp = qt0 + tid % BQ;
            r_ld = qp < seqlen_q ? (tid < BQ ? k.ptr_lse : k.ptr_d)[static_cast<long long>(h) * k.stride_lse_h + q0 + qp]
                                 : 0.0f;
        }
    };
    auto stage = [&](int buf) {
        if(tid < 2 * BQ)
            s_ld[buf][tid] = r_ld;
#pragma unroll
        for(int u = 0; u < QR; u++)
        {
            const int c = tid + u * BLOCK, r = c / (D_QK / 8), c8 = (c % (D_QK / 8)) * 8;
            if(c < QC)
                *reinterpret_cast<vec8_t*>(smem + OFF_Q + buf * BQ * LD_Q + r * LD_Q + c8) = r_q[u];
        }
        const int r = tid / (D_V / 8), c8 = (tid % (D_V / 8)) * 8;
        *reinterpret_cast<vec8_t*>(smem + OFF_DO + buf * BQ * LD_DO + r * LD_DO + c8) = r_do;
    };

    opus::smem<bf16_t> s_q[2]     = {opus::make_smem(smem + OFF_Q), opus::make_smem(smem + OFF_Q + BQ * LD_Q)};
    opus::smem<bf16_t> s_do[2]    = {opus::make_smem(smem + OFF_DO), opus::make_smem(smem + OFF_DO + BQ * LD_DO)};
    opus::smem<bf16_t> s_dst[2]   = {opus::make_smem(smem + OFF_DST), opus::make_smem(smem + OFF_DST + BK * LD_T)};
    opus::smem<bf16_t> s_dst_w[2] = {opus::make_smem(smem + OFF_DST + w * WR * LD_T),
                                     opus::make_smem(smem + OFF_DST + BK * LD_T + w * WR * LD_T)};
    auto s_k                      = opus::make_smem(smem + OFF_K);

    const auto u_q  = opus::partition_layout_b<8>(mma_s, opus::make_tuple(opus::number<LD_Q>{}, 1_I), c_ab);
    const auto u_do = opus::partition_layout_b<8>(mma_dp, opus::make_tuple(opus::number<LD_DO>{}, 1_I), c_ab);
    const auto u_ct = opus::partition_layout_c(mma_s, opus::make_tuple(opus::number<LD_T>{}, 1_I), c_c);
    // (row, col) of each C element of S^T, encoded as row * 256 + col.
    const auto pos_s = opus::layout_to_offsets<1>(
        opus::partition_layout_c(mma_s, opus::make_tuple(opus::number<256>{}, 1_I), c_c));
    // dQ: this wave's 48 dims x 16 queries of the tile, query rows from dq_base.
    float* dq_base  = k.ptr_dq_acc + static_cast<long long>(q0) * k.stride_dq_n +
                     static_cast<long long>(h) * k.stride_dq_h + (w % 4) * DQ_N;
    const auto u_dq = opus::partition_layout_c(mma_dq, opus::make_tuple(1_I, k.stride_dq_n), c_c);

    typename decltype(mma_dv)::vtype_c v_dv;
    typename decltype(mma_dk)::vtype_c v_dk;
    opus::clear(v_dv);
    opus::clear(v_dk);
    const float scale_log2 = k.softmax_scale * LOG2_E;

    if(i_begin < n_q_tiles)
    {
        fetch(i_begin);
        stage(0);
        if(i_begin + 1 < n_q_tiles)
            fetch(i_begin + 1);
    }
    __syncthreads();

    for(int i = i_begin; i < n_q_tiles; i++)
    {
        const int b   = (i - i_begin) & 1;
        const int qt0 = i * BQ;
        constexpr int NS = decltype(mma_s)::tile_c_len;
        float lse_e[NS], d_e[NS];
#pragma unroll
        for(int e = 0; e < NS; e++)
        {
            lse_e[e] = s_ld[b][pos_s[e] % 256] * LOG2_E;
            d_e[e]   = s_ld[b][BQ + pos_s[e] % 256];
        }

        const auto v_st  = mma_s(v_kw, s_q[b].template load<8>(u_q));
        const auto v_dpt = mma_dp(v_vw, s_do[b].template load<8>(u_do));
        opus::vector_t<float, NS> v_p, v_ds;
#pragma unroll
        for(int e = 0; e < NS; e++)
        {
            const int kv = row_w + pos_s[e] / 256, qp = qt0 + pos_s[e] % 256;
            const bool ok = kv < seqlen_k && qp < seqlen_q && (!k.causal || qp + causal_offset >= kv);
            // Select the exponent, not the result, so the exp needs no branch.
            const float p = __builtin_amdgcn_exp2f(ok ? v_st[e] * scale_log2 - lse_e[e] : -1e30f);
            v_p[e]  = p;
            v_ds[e] = p * (v_dpt[e] - d_e[e]) * k.softmax_scale;
        }
        // C -> A: the K order of these fragments is what tr_load_b_c_order supplies.
        const auto a_p  = opus::cast<bf16_t>(v_p);
        const auto a_ds = opus::cast<bf16_t>(v_ds);
        v_dv = mma_dv(a_p, tr_load_b_c_order<D_V / 16, LD_DO>(s_do[b], lane), v_dv);
        v_dk = mma_dk(a_ds, tr_load_b_c_order<D_QK / 16, LD_Q>(s_q[b], lane), v_dk);
        s_dst_w[b].template store<4>(a_ds, u_ct);

        if(i + 1 < n_q_tiles)
            stage(b ^ 1);
        if(i + 2 < n_q_tiles)
            fetch(i + 2);
        __syncthreads();

        const auto v_dq = mma_dq(tr_load_std<DQ_N / 16, BK / 32, LD_K>(s_k, lane, (w % 4) * DQ_N),
                                 tr_load_std<1, BK / 32, LD_T>(s_dst[b], lane, (w / 4) * 16));
        // Buffer atomics bounded to this wave's valid query rows: rows past seqlen_q are
        // dropped by the descriptor instead of a branch per element.
        const int dq_row = qt0 + (w / 4) * 16;
        auto g_dq        = opus::make_gmem(dq_base + static_cast<long long>(dq_row) * k.stride_dq_n,
                                    rec(static_cast<long long>(seqlen_q - dq_row) * k.stride_dq_n, 4));
        g_dq.template atomic_add<1>(v_dq, u_dq);
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
    constexpr int DK_LEN  = decltype(mma_dk)::tile_c_len;
    constexpr int DKN_LEN = DK_LEN * D_NOPE / D_QK;
    g_dkn.template store<4>(opus::cast<bf16_t>(opus::slice(v_dk, opus::number<0>{}, opus::number<DKN_LEN>{})), u_dkv);
    g_dv.template store<4>(opus::cast<bf16_t>(v_dv), u_dkv);
    g_dkr.template store<4>(opus::slice(v_dk, opus::number<DKN_LEN>{}, opus::number<DK_LEN>{}), u_dkr);
}

#else
__global__ void gated_mla_bwd_pre_kernel(gated_mla_bwd_pre_kargs) {}
__global__ void gated_mla_bwd_kernel(gated_mla_bwd_kargs) {}
#endif
