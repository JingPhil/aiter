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

__device__ inline u32x2_t tr_read(const opus::smem<bf16_t>& s, int elem_off)
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
__device__ inline bvec<NT * 8> tr_load_b_c_order(const opus::smem<bf16_t>& s, int lane, int col0)
{
    const int g = lane / 16, i = lane % 16;
    const int base = (g * 4 + i / 4) * LD + col0 + (i % 4) * 4;
    u32x2_t raw[NT * 2];
#pragma unroll
    for(int tn = 0; tn < NT; tn++)
#pragma unroll
        for(int hh = 0; hh < 2; hh++)
            raw[tn * 2 + hh] = tr_read(s, base + hh * 16 * LD + tn * 16);
    return concat_tr_loads(raw);
}

// A or B fragments in the MFMA's own K order (slot j of lane group g is row 8 g + j) over
// NKS steps of 32 rows from row0, for NT tiles of 16 columns from col0.
template <int NT, int NKS, int LD>
__device__ inline bvec<NT * NKS * 8> tr_load_std(const opus::smem<bf16_t>& s, int lane, int col0, int row0)
{
    const int g = lane / 16, i = lane % 16;
    const int base = (row0 + g * 8 + i / 4) * LD + col0 + (i % 4) * 4;
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

// Every GEMM is per wave with 16x16x32 bf16 MFMAs, C = A * B^T. The two waves of a SIMD
// (w, w + 4) share 32 keys and split the work along its two dependency chains, so each LDS
// fragment of Q and dO feeds two M tiles:
//   S wave  (w < 4):  S^T = K Q^T, P, dV += P^T dO, and dQ^T of the previous query tile
//   dP wave (w >= 4): dP^T = V dO^T, dS = P (dP - D), dK += dS^T Q, and the query-tile fetch
// P crosses from the S wave to the dP wave through LDS in fp32. P^T and dS^T are reused in
// registers as the A operand of dV and dK; their B operands (dO^T, Q^T) are transposed reads
// of the natural Q and dO tiles. dS^T goes through LDS since dQ sums it over all 128 keys.
// Each role runs its own loop so that a wave only keeps its own role's registers live; both
// loops pass the same two barriers per tile, after P is written and after dS^T is written.
// dQ atomics come only from S waves and global loads only from dP waves, so neither waits
// on the other's vmcnt.
__global__ __launch_bounds__(gated_mla_bwd::BLOCK, 1) void gated_mla_bwd_kernel(gated_mla_bwd_kargs k)
{
    using namespace gated_mla_bwd;
    using opus::number;
    using opus::operator""_I;
    using vec8_t  = bvec<8>;
    using f32x4_t = opus::vector_t<float, 4>;
    constexpr float LOG2_E = 1.44269504089f;
    constexpr int MR       = PR / 16;          // M tiles per wave
    constexpr int P_CHUNKS = PR * BQ / 64 / 4;  // float4 chunks of P per lane
    __shared__ __attribute__((aligned(16))) bf16_t smem[SMEM_ELEMS];
    // LSE and D of a query tile, [lse | D], double-buffered like sQ.
    __shared__ __attribute__((aligned(16))) float s_ld[2][2 * BQ];
    // P of each pair in its C layout; chunk-major so that each float4 access is lane-contiguous.
    __shared__ f32x4_t s_p[NP][P_CHUNKS][64];

    const int tid     = static_cast<int>(opus::thread_id_x());
    const int lane    = tid % 64;
    const int w       = __builtin_amdgcn_readfirstlane(tid / 64);
    const int pr      = w % NP;
    const bool s_wave = w < NP;
    const int j       = static_cast<int>(opus::block_id_x());
    const int h       = static_cast<int>(opus::block_id_y());
    const int g       = static_cast<int>(opus::block_id_z());
    auto sc           = [](int x) { return __builtin_amdgcn_readfirstlane(x); };
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
    const bool any_tile = i_begin < n_q_tiles;

    auto make_mma = [](auto e_m, auto e_n, auto e_k) {
        return opus::make_tiled_mma<bf16_t, bf16_t, float>(
            opus::seq<decltype(e_m)::value, decltype(e_n)::value, decltype(e_k)::value>{},
            opus::seq<1, 1, 1>{},
            opus::seq<16, 16, 32>{},
            opus::mfma_adaptor_swap_ab{});
    };
    static_assert(D_V == D_NOPE, "dV and the nope part of dK share one MMA shape");
    auto mma_s   = make_mma(number<MR>{}, number<BQ / 16>{}, number<D_QK / 32>{});   // S^T  [32 x BQ]
    auto mma_kn  = make_mma(number<MR>{}, number<BQ / 16>{}, number<D_NOPE / 32>{}); // A of S^T, nope part
    auto mma_kr  = make_mma(number<MR>{}, number<BQ / 16>{}, number<D_ROT / 32>{});  // A of S^T, rope part
    auto mma_dp  = make_mma(number<MR>{}, number<BQ / 16>{}, number<D_V / 32>{});    // dP^T [32 x BQ]
    auto mma_dv  = make_mma(number<MR>{}, number<D_V / 16>{}, number<BQ / 32>{});    // dV, dK nope [32 x 128]
    auto mma_dkr = make_mma(number<MR>{}, number<D_ROT / 16>{}, number<BQ / 32>{});  // dK rope [32 x 64]
    // dQ^T [48 x BQ] = K^T dS^T over half of the keys per call: lanes run along d, so each
    // atomic instruction covers contiguous dq_acc rows.
    auto mma_dq = make_mma(number<DQ_M / 16>{}, number<BQ / 16>{}, number<BK / 64>{});
    static_assert(BQ == 32, "the C -> A reuse of P^T / dS^T needs the query tile to be one K step");
    const auto c_ab = opus::make_tuple(0_I, lane % mma_s.grpm_a, 0_I, lane / mma_s.grpm_a);
    const auto c_c  = opus::make_tuple(0_I, lane % mma_s.grpn_c, 0_I, lane / mma_s.grpn_c);

    auto rec = [](long long elems, int elem_bytes) -> unsigned int {
        const long long b = elems > 0 ? elems * elem_bytes : 0;
        return b >= 0xffffffffLL ? 0xffffffffu : static_cast<unsigned int>(b);
    };

    // This pair's 32 keys; rows past seqlen_k read as 0 and are dropped on store.
    const int row_w  = kv0 + pr * PR;
    const int rows_w = seqlen_k - row_w < PR ? (seqlen_k - row_w > 0 ? seqlen_k - row_w : 0) : PR;
    const bf16_t* k_nope = reinterpret_cast<const bf16_t*>(k.ptr_k_nope);
    const bf16_t* k_rot  = reinterpret_cast<const bf16_t*>(k.ptr_k_rot);

    // The whole key block as K^T [d][kv], the A operand of dQ^T. Consecutive threads take
    // consecutive keys, so each 2-byte store instruction covers one contiguous K^T row segment.
    for(int c = tid; c < BK * D_QK / 8; c += BLOCK)
    {
        const int r = c % BK, c8 = (c / BK) * 8, kv = kv0 + r;
        vec8_t v;
        if(kv < seqlen_k)
            v = *reinterpret_cast<const vec8_t*>(
                c8 < D_NOPE ? k_nope + static_cast<long long>(k0 + kv) * k.stride_k_n +
                                  static_cast<long long>(h) * k.stride_k_h + c8
                            : k_rot + static_cast<long long>(k0 + kv) * k.stride_k_rot_n + c8 - D_NOPE);
        else
            for(int e = 0; e < 8; e++)
                v[e] = bf16_t(0.0f);
#pragma unroll
        for(int e = 0; e < 8; e++)
            smem[OFF_KT + (c8 + e) * LD_KT + r] = v[e];
    }

    // Buffers by query-tile parity. These are computed, not indexed from an array of smem views:
    // a runtime-indexed array would be promoted to LDS and cost a pointer load per use.
    auto s_q   = [&](int b) { return opus::make_smem(smem + OFF_Q + b * BQ * LD_Q); };
    auto s_do  = [&](int b) { return opus::make_smem(smem + OFF_DO + b * BQ * LD_DO); };
    auto s_dst = [&](int b) { return opus::make_smem(smem + OFF_DST + b * BK * LD_T); };

    const auto u_q  = opus::partition_layout_b<8>(mma_s, opus::make_tuple(number<LD_Q>{}, 1_I), c_ab);
    const auto u_do = opus::partition_layout_b<8>(mma_dp, opus::make_tuple(number<LD_DO>{}, 1_I), c_ab);
    // (row, col) of each C element of S^T and dP^T, encoded as row * 256 + col.
    const auto pos_s = opus::layout_to_offsets<1>(
        opus::partition_layout_c(mma_s, opus::make_tuple(number<256>{}, 1_I), c_c));
    constexpr int NS = decltype(mma_s)::tile_c_len;
    const float scale_log2 = k.softmax_scale * LOG2_E;
    // LSE or D at this lane's C elements of S^T / dP^T: element (tm, tn, p) is query
    // 16 tn + 4 (lane / 16) + p, so each N tile is one float4 run of s_ld.
    auto ld_c = [&](int b, int off, float mul, float(&out)[NS]) {
        f32x4_t run[BQ / 16];
#pragma unroll
        for(int tn = 0; tn < BQ / 16; tn++)
            run[tn] = *reinterpret_cast<const f32x4_t*>(&s_ld[b][off + 16 * tn + 4 * (lane / 16)]);
#pragma unroll
        for(int e = 0; e < NS; e++)
            out[e] = run[(e / 4) % (BQ / 16)][e % 4] * mul;
    };
    // This pair's rows of dkv [.., H, dK nope | dV]; rows past seqlen_k fall outside the
    // descriptors and are dropped.
    auto dkv_part = [&](int col0) {
        bf16_t* dkv = reinterpret_cast<bf16_t*>(k.ptr_dkv) + static_cast<long long>(k0 + row_w) * k.stride_dkv_n +
                      static_cast<long long>(h) * k.stride_dkv_h + col0;
        return opus::make_gmem(dkv, rec(static_cast<long long>(rows_w) * k.stride_dkv_n - col0, 2));
    };

    if(s_wave)
    {
        auto g_kn = opus::make_gmem(k_nope + static_cast<long long>(k0 + row_w) * k.stride_k_n +
                                        static_cast<long long>(h) * k.stride_k_h,
                                    rec(static_cast<long long>(rows_w) * k.stride_k_n, 2));
        auto g_kr = opus::make_gmem(k_rot + static_cast<long long>(k0 + row_w) * k.stride_k_rot_n,
                                    rec(static_cast<long long>(rows_w) * k.stride_k_rot_n, 2));
        const auto v_kn = g_kn.template load<8>(opus::partition_layout_a<8>(mma_kn, opus::make_tuple(k.stride_k_n, 1_I), c_ab));
        const auto v_kr = g_kr.template load<8>(opus::partition_layout_a<8>(mma_kr, opus::make_tuple(k.stride_k_rot_n, 1_I), c_ab));
        // A of S^T is [M tile][K step][8]; nope and rope interleave per M tile.
        constexpr int KS = D_QK / 32, KS_N = D_NOPE / 32, KS_R = D_ROT / 32;
        typename decltype(mma_s)::vtype_a v_kw;
#pragma unroll
        for(int m = 0; m < MR; m++)
#pragma unroll
            for(int ks = 0; ks < KS; ks++)
#pragma unroll
                for(int e = 0; e < 8; e++)
                {
                    if(ks < KS_N)
                        v_kw[(m * KS + ks) * 8 + e] = v_kn[(m * KS_N + ks) * 8 + e];
                    else
                        v_kw[(m * KS + ks) * 8 + e] = v_kr[(m * KS_R + ks - KS_N) * 8 + e];
                }

        __syncthreads();

        typename decltype(mma_dv)::vtype_c v_dv;
        typename decltype(mma_dq)::vtype_c v_dq;
        opus::clear(v_dv);
        opus::clear(v_dq);
        // dQ: this wave's 48 dims x the tile's 32 queries, query rows from dq_base.
        float* dq_base = k.ptr_dq_acc + static_cast<long long>(q0) * k.stride_dq_n +
                         static_cast<long long>(h) * k.stride_dq_h + pr * DQ_M;
        auto s_kt = [&](int half) { return opus::make_smem(smem + OFF_KT + pr * DQ_M * LD_KT + half * (BK / 2)); };
        const auto u_kt = opus::partition_layout_a<8>(mma_dq, opus::make_tuple(number<LD_KT>{}, 1_I), c_ab);
        auto dq_half    = [&](int buf, int half) {
            v_dq = mma_dq(s_kt(half).template load<8>(u_kt),
                          tr_load_std<BQ / 16, BK / 64, LD_T>(s_dst(buf), lane, 0, half * (BK / 2)),
                          v_dq);
        };
        // C element (tm, tn, p) of dQ^T is dim 16 tm + lane % 16 of query 16 tn + 4 (lane / 16) + p.
        // One descriptor per query row offset (16 tn + p) keeps the row stride in SGPRs, so a lane
        // needs a single offset VGPR; rows past seqlen_q fall outside the descriptor and drop.
        const int dq_vo = lane % 16 + (lane / 16) * 4 * k.stride_dq_n;
        auto dq_flush   = [&](int qt0) {
            opus::static_for<BQ / 16>([&](auto tn) {
                opus::static_for<4>([&](auto p) {
                    const int qr = qt0 + 16 * tn.value + p.value;
                    auto g_dq    = opus::make_gmem(dq_base + static_cast<long long>(qr) * k.stride_dq_n,
                                                rec(static_cast<long long>(seqlen_q - qr) * k.stride_dq_n, 4));
                    opus::static_for<DQ_M / 16>([&](auto tm) {
                        constexpr int e = (tm.value * (BQ / 16) + tn.value) * 4 + p.value;
                        g_dq.template atomic_add<1>(v_dq[e], dq_vo, 0, number<16 * tm.value>{});
                    });
                });
            });
            opus::clear(v_dq);
        };

        for(int i = i_begin; i < n_q_tiles; i++)
        {
            const int b   = (i - i_begin) & 1;
            const int qt0 = i * BQ;
            float lse_e[NS];
            ld_c(b, 0, LOG2_E, lse_e);
            const auto v_st = mma_s(v_kw, s_q(b).template load<8>(u_q));
            if(i > i_begin)
                dq_half(b ^ 1, 0);
            opus::vector_t<float, NS> v_p;
#pragma unroll
            for(int e = 0; e < NS; e++)
            {
                const int kv = row_w + pos_s[e] / 256, qp = qt0 + pos_s[e] % 256;
                // Non-short-circuit tests and a selected exponent keep the masked exp branch-free.
                const bool ok = (kv < seqlen_k) & (qp < seqlen_q) & (!k.causal | (qp + causal_offset >= kv));
                const float x = v_st[e] * scale_log2 - lse_e[e];
                v_p[e]        = __builtin_amdgcn_exp2f(ok ? x : -1e30f);
            }
#pragma unroll
            for(int c = 0; c < P_CHUNKS; c++)
            {
                f32x4_t t;
#pragma unroll
                for(int e = 0; e < 4; e++)
                    t[e] = v_p[c * 4 + e];
                s_p[pr][c][lane] = t;
            }
            // C -> A: the K order of this fragment is what tr_load_b_c_order supplies.
            const auto a_p = opus::cast<bf16_t>(v_p);
            __syncthreads();

            v_dv = mma_dv(a_p, tr_load_b_c_order<D_V / 16, LD_DO>(s_do(b), lane, 0), v_dv);
            if(i > i_begin)
            {
                dq_half(b ^ 1, 1);
                dq_flush(qt0 - BQ);
            }
            __syncthreads();
        }
        if(any_tile)
        {
            const int b = (n_q_tiles - 1 - i_begin) & 1;
            dq_half(b, 0);
            dq_half(b, 1);
            dq_flush((n_q_tiles - 1) * BQ);
        }

        auto g_dv = dkv_part(D_NOPE);
        g_dv.template store<4>(opus::cast<bf16_t>(v_dv),
                               opus::partition_layout_c(mma_dv, opus::make_tuple(k.stride_dkv_n, 1_I), c_c));
    }
    else
    {
        // Query tiles, with their LSE and D, go global -> registers -> LDS one tile ahead, all by
        // the dP waves. Thread ft owns row ft / 8 of the tile and every 8th chunk of 8 columns
        // from ft % 8, so one offset VGPR per tensor serves all its chunks; rows past seqlen_q
        // fall outside the descriptors and read as 0.
        constexpr int FT = (NW - NP) * 64, FC = 8;
        static_assert(FT == BQ * FC, "one tile row per 8 fetching threads");
        constexpr int QR = D_QK / 8 / FC, DR = D_V / 8 / FC;
        const int ft = tid - NP * 64, f_row = ft / FC, f_col = (ft % FC) * 8;
        const int vo_q = f_row * k.stride_q_n + f_col, vo_do = f_row * k.stride_do_n + f_col;
        const int so_q = f_row * LD_Q + f_col, so_do = f_row * LD_DO + f_col;
        const bf16_t* q_ptr  = reinterpret_cast<const bf16_t*>(k.ptr_q) + static_cast<long long>(q0) * k.stride_q_n +
                              static_cast<long long>(h) * k.stride_q_h;
        const bf16_t* do_ptr = reinterpret_cast<const bf16_t*>(k.ptr_do) + static_cast<long long>(q0) * k.stride_do_n +
                               static_cast<long long>(h) * k.stride_do_h;
        const float* ld_ptr = (ft < BQ ? k.ptr_lse : k.ptr_d) + static_cast<long long>(h) * k.stride_lse_h + q0;
        vec8_t r_q[QR], r_do[DR];
        float r_ld = 0.0f;
        auto fetch = [&](int i) {
            const int qt0 = i * BQ;
            auto g_q  = opus::make_gmem(q_ptr + static_cast<long long>(qt0) * k.stride_q_n,
                                       rec(static_cast<long long>(seqlen_q - qt0) * k.stride_q_n, 2));
            auto g_do = opus::make_gmem(do_ptr + static_cast<long long>(qt0) * k.stride_do_n,
                                        rec(static_cast<long long>(seqlen_q - qt0) * k.stride_do_n, 2));
#pragma unroll
            for(int u = 0; u < QR; u++)
                r_q[u] = g_q.template load<8>(vo_q + u * FC * 8);
#pragma unroll
            for(int u = 0; u < DR; u++)
                r_do[u] = g_do.template load<8>(vo_do + u * FC * 8);
            if(ft < 2 * BQ)
            {
                auto g_ld = opus::make_gmem(ld_ptr + qt0, rec(seqlen_q - qt0, 4));
                r_ld      = g_ld.template load<1>(ft % BQ)[0];
            }
        };
        auto stage = [&](int buf) {
            if(ft < 2 * BQ)
                s_ld[buf][ft] = r_ld;
#pragma unroll
            for(int u = 0; u < QR; u++)
                *reinterpret_cast<vec8_t*>(smem + OFF_Q + buf * BQ * LD_Q + so_q + u * FC * 8) = r_q[u];
#pragma unroll
            for(int u = 0; u < DR; u++)
                *reinterpret_cast<vec8_t*>(smem + OFF_DO + buf * BQ * LD_DO + so_do + u * FC * 8) = r_do[u];
        };
        if(any_tile)
        {
            fetch(i_begin);
            stage(0);
            if(i_begin + 1 < n_q_tiles)
                fetch(i_begin + 1);
        }
        auto g_v = opus::make_gmem(reinterpret_cast<const bf16_t*>(k.ptr_v) +
                                       static_cast<long long>(k0 + row_w) * k.stride_v_n +
                                       static_cast<long long>(h) * k.stride_v_h,
                                   rec(static_cast<long long>(rows_w) * k.stride_v_n, 2));
        const auto v_vw = g_v.template load<8>(opus::partition_layout_a<8>(mma_dp, opus::make_tuple(k.stride_v_n, 1_I), c_ab));
        auto s_dst_w = [&](int b) { return opus::make_smem(smem + OFF_DST + b * BK * LD_T + pr * PR * LD_T); };
        const auto u_ct = opus::partition_layout_c(mma_s, opus::make_tuple(number<LD_T>{}, 1_I), c_c);

        __syncthreads();

        typename decltype(mma_dv)::vtype_c v_dkn;
        typename decltype(mma_dkr)::vtype_c v_dkr;
        opus::clear(v_dkn);
        opus::clear(v_dkr);

        for(int i = i_begin; i < n_q_tiles; i++)
        {
            const int b = (i - i_begin) & 1;
            float d_e[NS];
            ld_c(b, BQ, 1.0f, d_e);
            const auto v_dpt = mma_dp(v_vw, s_do(b).template load<8>(u_do));
            if(i + 1 < n_q_tiles)
                stage(b ^ 1);
            if(i + 2 < n_q_tiles)
                fetch(i + 2);
            __syncthreads();

            opus::vector_t<float, NS> v_ds;
#pragma unroll
            for(int c = 0; c < P_CHUNKS; c++)
            {
                const f32x4_t p = s_p[pr][c][lane];
#pragma unroll
                for(int e = 0; e < 4; e++)
                    v_ds[c * 4 + e] = p[e] * (v_dpt[c * 4 + e] - d_e[c * 4 + e]) * k.softmax_scale;
            }
            const auto a_ds = opus::cast<bf16_t>(v_ds);
            s_dst_w(b).template store<4>(a_ds, u_ct);
            v_dkn = mma_dv(a_ds, tr_load_b_c_order<D_NOPE / 16, LD_Q>(s_q(b), lane, 0), v_dkn);
            v_dkr = mma_dkr(a_ds, tr_load_b_c_order<D_ROT / 16, LD_Q>(s_q(b), lane, D_NOPE), v_dkr);
            __syncthreads();
        }

        // dK nope into the kv_b_proj layout, dK rope per head in fp32.
        auto g_dkn = dkv_part(0);
        auto g_dkr = opus::make_gmem(k.ptr_dk_rope + static_cast<long long>(k0 + row_w) * k.stride_dkr_n +
                                         static_cast<long long>(h) * k.stride_dkr_h,
                                     rec(static_cast<long long>(rows_w) * k.stride_dkr_n, 4));
        g_dkn.template store<4>(opus::cast<bf16_t>(v_dkn),
                                opus::partition_layout_c(mma_dv, opus::make_tuple(k.stride_dkv_n, 1_I), c_c));
        g_dkr.template store<4>(v_dkr, opus::partition_layout_c(mma_dkr, opus::make_tuple(k.stride_dkr_n, 1_I), c_c));
    }
}

#else
__global__ void gated_mla_bwd_pre_kernel(gated_mla_bwd_pre_kargs) {}
__global__ void gated_mla_bwd_kernel(gated_mla_bwd_kargs) {}
#endif
