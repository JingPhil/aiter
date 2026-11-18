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

// B fragments of 16x16x16 MFMAs (N-tiles of 16 columns from col0, K = 16 rows): slot j of
// lane group g is row 4 g + j, one transposed read per N tile. This K order is also the one of
// a swap_ab 16x16 C fragment, so such a C fragment is directly the matching A operand.
template <int NT, int LD>
__device__ inline bvec<NT * 4> tr_load_b16(const opus::smem<bf16_t>& s, int lane, int col0)
{
    const int g = lane / 16, i = lane % 16;
    const int base = (g * 4 + i / 4) * LD + col0 + (i % 4) * 4;
    u32x2_t raw[NT];
#pragma unroll
    for(int tn = 0; tn < NT; tn++)
        raw[tn] = tr_read(s, base + tn * 16);
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

// acc += A * B^T on one 16x16x16 bf16 MFMA, C in the swap_ab layout of opus (lane l holds
// row l % 16, columns 4 (l / 16) .. + 3), so the hardware's first operand is B. acc stays in
// AGPRs: the module is built with -amdgpu-mfma-vgpr-form, so the compiler's own MFMAs
// accumulate in VGPRs and the AGPR file holds only these long-lived dK / dV tiles. The
// compiler does not model this asm as an MFMA: s_nop 1 gives the 2 wait states a VALU write of
// A or B needs before the MFMA reads it, and readers of acc wait with mfma_acc_drain.
using f32x4_t = opus::vector_t<float, 4>;
__device__ inline void mfma_acc_16x16x16(f32x4_t& acc, const u32x2_t& a, const u32x2_t& b)
{
    asm volatile("s_nop 1\n\tv_mfma_f32_16x16x16_bf16 %0, %2, %1, %0" : "+a"(acc) : "v"(a), "v"(b));
}

// Covers the latency of the last mfma_acc_16x16x16 before its accumulators are read.
__device__ inline void mfma_acc_drain() { asm volatile("s_nop 7\n\ts_nop 7\n\ts_nop 7"); }

// 4 bf16 at element 4 i of a fragment, as the 64-bit MFMA operand.
template <typename V>
__device__ inline u32x2_t frag4(const V& v, int i)
{
    opus::vector_t<bf16_t, 4> r;
#pragma unroll
    for(int e = 0; e < 4; e++)
        r[e] = v[4 * i + e];
    return __builtin_bit_cast(u32x2_t, r);
}

// acc[m][n] += A_m * B_n^T over the M tiles of a (4 bf16 each) and the N tiles of b.
template <int MT, int NT, typename VA, typename VB>
__device__ inline void acc_gemm(f32x4_t (&acc)[MT][NT], const VA& a, const VB& b)
{
    static_assert(opus::vector_traits<VA>::size() == MT * 4 && opus::vector_traits<VB>::size() == NT * 4,
                  "one 16x16x16 fragment per tile");
#pragma unroll
    for(int m = 0; m < MT; m++)
#pragma unroll
        for(int n = 0; n < NT; n++)
            mfma_acc_16x16x16(acc[m][n], frag4(a, m), frag4(b, n));
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

// Each wave runs at one per SIMD so that its 48 keys' K, V, dK and dV stay in registers
// (accumulators in AGPRs): every LDS fragment of Q and dO feeds three M tiles. GEMMs are
// per wave, C = A * B^T: S^T, dP^T and dQ^T on 16x16x32 MFMAs, dV and dK on 16x16x16 MFMAs
// with one 16-query tile as K, so that P^T and dS^T, the C fragments of S^T and dP^T, are
// their A operands as they are. Their B operands (dO^T, Q^T) are transposed reads of the
// natural Q and dO tiles. dS^T goes through LDS since dQ sums it over all 192 keys of the
// workgroup. The next query tile is fetched into registers during the current one and staged
// into the other LDS buffer, so each tile has one barrier.
__global__ __launch_bounds__(gated_mla_bwd::BLOCK, 1) void gated_mla_bwd_kernel(gated_mla_bwd_kargs k)
{
    using namespace gated_mla_bwd;
    using opus::number;
    using opus::operator""_I;
    using vec8_t = bvec<8>;
    constexpr float LOG2_E = 1.44269504089f;
    constexpr int MR       = WR / 16;  // M tiles per wave
    __shared__ __attribute__((aligned(16))) bf16_t smem[SMEM_ELEMS];
    // LSE and D of a query tile, [lse | D], double-buffered like sQ.
    __shared__ __attribute__((aligned(16))) float s_ld[2][2 * BQ];

    const int tid  = static_cast<int>(opus::thread_id_x());
    const int lane = tid % 64;
    const int w    = __builtin_amdgcn_readfirstlane(tid / 64);
    // Workgroup id -> (head h, sequence g, KV block j), dispatched in id order: groups of HG heads
    // one after another; within a group the head varies fastest, then the sequence, then the
    // block. Workgroups reach the XCDs round-robin by id, so each XCD gets the same heads of every
    // block (balanced causal work); every (h, g) starts with its long early blocks; and only about
    // HG heads' Q / dO are live in the caches at a time.
    const int wg   = static_cast<int>(opus::block_id_x());
    const int h    = wg % HG + HG * (wg / (HG * k.num_seqs * k.n_kv_blocks));
    const int g    = wg / HG % k.num_seqs;
    const int j    = wg / (HG * k.num_seqs) % k.n_kv_blocks;
    if(h >= k.H)
        return;
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

    auto make_mma = [](auto e_m, auto e_n, auto e_k, auto t_k) {
        return opus::make_tiled_mma<bf16_t, bf16_t, float>(
            opus::seq<decltype(e_m)::value, decltype(e_n)::value, decltype(e_k)::value>{},
            opus::seq<1, 1, 1>{},
            opus::seq<16, 16, decltype(t_k)::value>{},
            opus::mfma_adaptor_swap_ab{});
    };
    static_assert(D_V == D_NOPE, "dV and the nope part of dK share one MMA shape");
    static_assert(BQ == 16, "P^T / dS^T are the A operand of a 16x16x16 MFMA as they are");
    constexpr auto K32 = number<32>{};
    constexpr auto K16 = number<16>{};
    auto mma_s   = make_mma(number<MR>{}, number<1>{}, number<D_QK / 32>{}, K32);    // S^T  [48 x 16]
    auto mma_kn  = make_mma(number<MR>{}, number<1>{}, number<D_NOPE / 32>{}, K32);  // A of S^T, nope part
    auto mma_kr  = make_mma(number<MR>{}, number<1>{}, number<D_ROT / 32>{}, K32);   // A of S^T, rope part
    auto mma_dp  = make_mma(number<MR>{}, number<1>{}, number<D_V / 32>{}, K32);     // dP^T [48 x 16]
    auto mma_dv  = make_mma(number<MR>{}, number<D_V / 16>{}, number<1>{}, K16);     // dV, dK nope [48 x 128]
    auto mma_dkr = make_mma(number<MR>{}, number<D_ROT / 16>{}, number<1>{}, K16);   // dK rope [48 x 64]
    // dQ^T [48 x 16] = K^T dS^T over 64 keys per call: lanes run along d, so each atomic
    // instruction covers contiguous dq_acc rows.
    constexpr int DQ_KC = 64;
    auto mma_dq = make_mma(number<DQ_M / 16>{}, number<1>{}, number<DQ_KC / 32>{}, K32);
    const auto c_ab = opus::make_tuple(0_I, lane % mma_s.grpm_a, 0_I, lane / mma_s.grpm_a);
    const auto c_c  = opus::make_tuple(0_I, lane % mma_s.grpn_c, 0_I, lane / mma_s.grpn_c);

    auto rec = [](long long elems, int elem_bytes) -> unsigned int {
        const long long b = elems > 0 ? elems * elem_bytes : 0;
        return b >= 0xffffffffLL ? 0xffffffffu : static_cast<unsigned int>(b);
    };

    // This wave's 48 keys as the A operand of S^T and dP^T; rows past seqlen_k read as 0.
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
    const auto v_kn = g_kn.template load<8>(opus::partition_layout_a<8>(mma_kn, opus::make_tuple(k.stride_k_n, 1_I), c_ab));
    const auto v_kr = g_kr.template load<8>(opus::partition_layout_a<8>(mma_kr, opus::make_tuple(k.stride_k_rot_n, 1_I), c_ab));
    const auto v_vw = g_v.template load<8>(opus::partition_layout_a<8>(mma_dp, opus::make_tuple(k.stride_v_n, 1_I), c_ab));
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

    // Query tiles, with their LSE and D, go global -> registers -> LDS one tile ahead. Waves 0-1
    // fetch Q and waves 2-3 dO; a thread owns row ft / 8 of its tensor and every 8th chunk of 8
    // columns from ft % 8, so one offset VGPR serves all its chunks. Rows past seqlen_q fall
    // outside the descriptors and read as 0.
    constexpr int FC = 8, QR = D_QK / 8 / FC, DR = D_V / 8 / FC;
    static_assert(BLOCK == 2 * BQ * FC, "half of the workgroup per tensor, one tile row per 8 threads");
    const bool f_q       = w < NW / 2;
    const int ft         = f_q ? tid : tid - BQ * FC;
    const int f_rows     = f_q ? k.stride_q_n : k.stride_do_n;
    const int f_chunks   = f_q ? QR : DR;
    const int f_vo       = (ft / FC) * f_rows + (ft % FC) * 8;
    const int f_so       = f_q ? OFF_Q + (ft / FC) * LD_Q + (ft % FC) * 8 : OFF_DO + (ft / FC) * LD_DO + (ft % FC) * 8;
    const int f_buf      = f_q ? BQ * LD_Q : BQ * LD_DO;
    const bf16_t* f_ptr  = f_q ? reinterpret_cast<const bf16_t*>(k.ptr_q) + static_cast<long long>(q0) * k.stride_q_n +
                                    static_cast<long long>(h) * k.stride_q_h
                               : reinterpret_cast<const bf16_t*>(k.ptr_do) + static_cast<long long>(q0) * k.stride_do_n +
                                    static_cast<long long>(h) * k.stride_do_h;
    const float* ld_ptr  = (tid < BQ ? k.ptr_lse : k.ptr_d) + static_cast<long long>(h) * k.stride_lse_h + q0;
    vec8_t r_f[QR];
    float r_ld = 0.0f;
    auto fetch = [&](int i) {
        const int qt0 = i * BQ;
        auto g_f = opus::make_gmem(f_ptr + static_cast<long long>(qt0) * f_rows,
                                   rec(static_cast<long long>(seqlen_q - qt0) * f_rows, 2));
#pragma unroll
        for(int u = 0; u < QR; u++)
            if(u < f_chunks)
                r_f[u] = g_f.template load<8>(f_vo + u * FC * 8);
        if(tid < 2 * BQ)
        {
            auto g_ld = opus::make_gmem(ld_ptr + qt0, rec(seqlen_q - qt0, 4));
            r_ld      = g_ld.template load<1>(tid % BQ)[0];
        }
    };
    auto stage = [&](int buf) {
        if(tid < 2 * BQ)
            s_ld[buf][tid] = r_ld;
#pragma unroll
        for(int u = 0; u < QR; u++)
            if(u < f_chunks)
                *reinterpret_cast<vec8_t*>(smem + f_so + buf * f_buf + u * FC * 8) = r_f[u];
    };

    // Buffers by query-tile parity. These are computed, not indexed from an array of smem views:
    // a runtime-indexed array would be promoted to LDS and cost a pointer load per use.
    auto s_q     = [&](int b) { return opus::make_smem(smem + OFF_Q + b * BQ * LD_Q); };
    auto s_do    = [&](int b) { return opus::make_smem(smem + OFF_DO + b * BQ * LD_DO); };
    auto s_dst   = [&](int b) { return opus::make_smem(smem + OFF_DST + b * BK * LD_T); };
    auto s_dst_w = [&](int b) { return opus::make_smem(smem + OFF_DST + b * BK * LD_T + w * WR * LD_T); };
    auto s_kt    = [&](int kc) { return opus::make_smem(smem + OFF_KT + w * DQ_M * LD_KT + kc * DQ_KC); };

    const auto u_q  = opus::partition_layout_b<8>(mma_s, opus::make_tuple(number<LD_Q>{}, 1_I), c_ab);
    const auto u_do = opus::partition_layout_b<8>(mma_dp, opus::make_tuple(number<LD_DO>{}, 1_I), c_ab);
    const auto u_ct = opus::partition_layout_c(mma_s, opus::make_tuple(number<LD_T>{}, 1_I), c_c);
    const auto u_kt = opus::partition_layout_a<8>(mma_dq, opus::make_tuple(number<LD_KT>{}, 1_I), c_ab);
    // (row, col) of each C element of S^T and dP^T, encoded as row * 256 + col.
    const auto pos_s = opus::layout_to_offsets<1>(
        opus::partition_layout_c(mma_s, opus::make_tuple(number<256>{}, 1_I), c_c));
    constexpr int NS = decltype(mma_s)::tile_c_len;
    const float scale_log2 = k.softmax_scale * LOG2_E;

    // dQ: this wave's 48 dims x the tile's 16 queries. C element (tm, p) of dQ^T is dim
    // 16 tm + lane % 16 of query 4 (lane / 16) + p. One descriptor per query offset p keeps the
    // row stride in SGPRs, so a lane needs a single offset VGPR; rows past seqlen_q fall outside
    // the descriptor and are dropped.
    float* dq_base  = k.ptr_dq_acc + static_cast<long long>(q0) * k.stride_dq_n +
                     static_cast<long long>(h) * k.stride_dq_h + w * DQ_M;
    const int dq_vo = lane % 16 + (lane / 16) * 4 * k.stride_dq_n;
    auto dq_tile    = [&](int b, int qt0) {
        typename decltype(mma_dq)::vtype_c v_dq;
        opus::clear(v_dq);
#pragma unroll
        for(int kc = 0; kc < BK / DQ_KC; kc++)
        {
            v_dq = mma_dq(s_kt(kc).template load<8>(u_kt),
                          tr_load_std<1, DQ_KC / 32, LD_T>(s_dst(b), lane, 0, kc * DQ_KC),
                          v_dq);
        }
        opus::static_for<4>([&](auto p) {
            const int qr = qt0 + p.value;
            auto g_dq    = opus::make_gmem(dq_base + static_cast<long long>(qr) * k.stride_dq_n,
                                        rec(static_cast<long long>(seqlen_q - qr) * k.stride_dq_n, 4));
            opus::static_for<DQ_M / 16>([&](auto tm) {
                g_dq.template atomic_add<1>(v_dq[tm.value * 4 + p.value], dq_vo, 0, number<16 * tm.value>{});
            });
        });
    };

    // dK and dV accumulators, [M tile][N tile] in the C layout of mma_dv / mma_dkr, in AGPRs.
    constexpr int NV = D_V / 16, NR = D_ROT / 16;
    f32x4_t acc_dv[MR][NV], acc_dkn[MR][NV], acc_dkr[MR][NR];
#pragma unroll
    for(int m = 0; m < MR; m++)
    {
#pragma unroll
        for(int n = 0; n < NV; n++)
            acc_dv[m][n] = acc_dkn[m][n] = f32x4_t{0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for(int n = 0; n < NR; n++)
            acc_dkr[m][n] = f32x4_t{0.0f, 0.0f, 0.0f, 0.0f};
    }

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
        // LSE and D at this lane's C elements: query 4 (lane / 16) + p, one float4 run each.
        const f32x4_t lse4 = *reinterpret_cast<const f32x4_t*>(&s_ld[b][4 * (lane / 16)]);
        const f32x4_t d4   = *reinterpret_cast<const f32x4_t*>(&s_ld[b][BQ + 4 * (lane / 16)]);

        const auto v_st = mma_s(v_kw, s_q(b).template load<8>(u_q));
        opus::vector_t<float, NS> v_p;
#pragma unroll
        for(int e = 0; e < NS; e++)
        {
            const int kv = row_w + pos_s[e] / 256, qp = qt0 + pos_s[e] % 256;
            // Non-short-circuit tests and a selected exponent keep the masked exp branch-free.
            const bool ok = (kv < seqlen_k) & (qp < seqlen_q) & (!k.causal | (qp + causal_offset >= kv));
            const float x = v_st[e] * scale_log2 - lse4[e % 4] * LOG2_E;
            v_p[e]        = __builtin_amdgcn_exp2f(ok ? x : -1e30f);
        }
        const auto v_dpt = mma_dp(v_vw, s_do(b).template load<8>(u_do));
        opus::vector_t<float, NS> v_ds;
#pragma unroll
        for(int e = 0; e < NS; e++)
            v_ds[e] = v_p[e] * (v_dpt[e] - d4[e % 4]) * k.softmax_scale;
        const auto a_p  = opus::cast<bf16_t>(v_p);
        const auto a_ds = opus::cast<bf16_t>(v_ds);
        s_dst_w(b).template store<4>(a_ds, u_ct);
        acc_gemm(acc_dv, a_p, tr_load_b16<D_V / 16, LD_DO>(s_do(b), lane, 0));
        acc_gemm(acc_dkn, a_ds, tr_load_b16<D_NOPE / 16, LD_Q>(s_q(b), lane, 0));
        acc_gemm(acc_dkr, a_ds, tr_load_b16<D_ROT / 16, LD_Q>(s_q(b), lane, D_NOPE));

        if(i + 1 < n_q_tiles)
            stage(b ^ 1);
        if(i + 2 < n_q_tiles)
            fetch(i + 2);
        __syncthreads();

        dq_tile(b, qt0);
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
    mfma_acc_drain();
    typename decltype(mma_dv)::vtype_c v_dv, v_dkn;
    typename decltype(mma_dkr)::vtype_c v_dkr;
#pragma unroll
    for(int m = 0; m < MR; m++)
#pragma unroll
        for(int e = 0; e < 4; e++)
        {
#pragma unroll
            for(int n = 0; n < NV; n++)
            {
                v_dv[(m * NV + n) * 4 + e]  = acc_dv[m][n][e];
                v_dkn[(m * NV + n) * 4 + e] = acc_dkn[m][n][e];
            }
#pragma unroll
            for(int n = 0; n < NR; n++)
                v_dkr[(m * NR + n) * 4 + e] = acc_dkr[m][n][e];
        }
    const auto u_dkv = opus::partition_layout_c(mma_dv, opus::make_tuple(k.stride_dkv_n, 1_I), c_c);
    g_dkn.template store<4>(opus::cast<bf16_t>(v_dkn), u_dkv);
    g_dv.template store<4>(opus::cast<bf16_t>(v_dv), u_dkv);
    g_dkr.template store<4>(v_dkr, opus::partition_layout_c(mma_dkr, opus::make_tuple(k.stride_dkr_n, 1_I), c_c));
}

#else
__global__ void gated_mla_bwd_pre_kernel(gated_mla_bwd_pre_kargs) {}
__global__ void gated_mla_bwd_kernel(gated_mla_bwd_kargs) {}
#endif
