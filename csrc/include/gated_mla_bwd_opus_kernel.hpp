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

// B fragments of 16x16x32 MFMAs (N tiles of 16 columns from col0, K = 32 rows) in the K order of
// two swap_ab 16x16 C tiles reused as the A operand: slot 4 h + e of lane group g is row
// 16 h + 4 g + e.
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

// tr_load_b_c_order with this lane's part of the address, tr_c_order_lane, already in s.
template <int LD>
__device__ inline int tr_c_order_lane(int lane)
{
    const int g = lane / 16, i = lane % 16;
    return (g * 4 + i / 4) * LD + (i % 4) * 4;
}
template <int NT, int LD>
__device__ inline bvec<NT * 8> tr_load_b_c_order_at(const opus::smem<bf16_t>& s, int col0)
{
    u32x2_t raw[NT * 2];
#pragma unroll
    for(int tn = 0; tn < NT; tn++)
#pragma unroll
        for(int hh = 0; hh < 2; hh++)
            raw[tn * 2 + hh] = tr_read(s, col0 + hh * 16 * LD + tn * 16);
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

// The 16x16x32 form of mfma_acc_16x16x16 (same layouts, hazards and drain).
using u32x4_t = opus::vector_t<opus::u32_t, 4>;
__device__ inline void mfma_acc_16x16x32(f32x4_t& acc, const u32x4_t& a, const u32x4_t& b)
{
    asm volatile("s_nop 1\n\tv_mfma_f32_16x16x32_bf16 %0, %2, %1, %0" : "+a"(acc) : "v"(a), "v"(b));
}

// 8 bf16 at element 8 i of a fragment, as the 128-bit MFMA operand.
template <typename V>
__device__ inline u32x4_t frag8(const V& v, int i)
{
    opus::vector_t<bf16_t, 8> r;
#pragma unroll
    for(int e = 0; e < 8; e++)
        r[e] = v[8 * i + e];
    return __builtin_bit_cast(u32x4_t, r);
}

template <int MT, int NT, typename VA, typename VB>
__device__ inline void acc_gemm32(f32x4_t (&acc)[MT][NT], const VA& a, const VB& b)
{
    static_assert(opus::vector_traits<VA>::size() == MT * 8 && opus::vector_traits<VB>::size() == NT * 8,
                  "one 16x16x32 fragment per tile");
#pragma unroll
    for(int m = 0; m < MT; m++)
#pragma unroll
        for(int n = 0; n < NT; n++)
            mfma_acc_16x16x32(acc[m][n], frag8(a, m), frag8(b, n));
}

// Global -> LDS copy of 16 bytes per lane (buffer_load_dwordx4 ... lds): lane l's 16 bytes at
// byte offset voff of the buffer (past its size: zeros) land at LDS byte lds + 16 l. Inline asm,
// so the compiler neither sees the LDS write nor waits for it (it would wait before every LDS
// read it cannot tell apart); the caller waits vmcnt before anyone reads the data, which
// s_barrier alone does not do.
using i32x4_t = opus::vector_t<int, 4>;
__device__ inline i32x4_t dma_rsrc(const void* p, unsigned int bytes)
{
    const unsigned long long a = reinterpret_cast<unsigned long long>(p);
    return i32x4_t{__builtin_amdgcn_readfirstlane(static_cast<int>(a)),
                   __builtin_amdgcn_readfirstlane(static_cast<int>(a >> 32) & 0xffff),
                   __builtin_amdgcn_readfirstlane(static_cast<int>(bytes)),
                   0x00020000};
}
__device__ inline unsigned int lds_addr(const void* p)
{
    return static_cast<unsigned int>(reinterpret_cast<__UINTPTR_TYPE__>(p));
}
__device__ inline void dma16(const i32x4_t& rsrc, int voff, unsigned int lds)
{
    asm volatile("s_mov_b32 m0, %2\n\tbuffer_load_dwordx4 %0, %1, 0 offen lds" : : "v"(voff), "s"(rsrc), "s"(lds) : "m0");
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
// DQ = false skips dQ (no K^T staging, no dS^T in LDS, no atomics) for a separate dQ pass.
template <bool DQ>
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
    // Bytes of `rows` rows clamped to [0, cap], for the per-tile descriptors: 32-bit math, where
    // rec's 64-bit compares go to the VALU. readfirstlane marks the size uniform; a size the
    // compiler cannot prove uniform turns every access through the descriptor into a waterfall loop.
    auto rec_rows = [](int rows, int cap, int row_bytes) -> unsigned int {
        return static_cast<unsigned int>(
            __builtin_amdgcn_readfirstlane((rows < 0 ? 0 : (rows > cap ? cap : rows)) * row_bytes));
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

    // The whole key block as K^T [d][kv], the A operand of dQ^T. Chunk c = it BLOCK + tid is key
    // c % BK, columns 8 (c / BK) .. + 7: consecutive threads take consecutive keys, so each 2-byte
    // LDS store covers one contiguous K^T row segment, and the first KT_NOPE rounds read k_nope,
    // the rest k_rot. All loads are issued before the transposing stores; keys past seqlen_k
    // fall outside the descriptors and read as 0.
    if constexpr(DQ)
    {
        constexpr int KT_ROUNDS = BK * D_QK / 8 / BLOCK, KT_NOPE = BK * D_NOPE / 8 / BLOCK;
        static_assert(KT_ROUNDS * BLOCK * 8 == BK * D_QK && KT_NOPE * BLOCK * 8 == BK * D_NOPE,
                      "whole rounds, each within one of k_nope / k_rot");
        auto g_kb = opus::make_gmem(k_nope + static_cast<long long>(k0 + kv0) * k.stride_k_n +
                                        static_cast<long long>(h) * k.stride_k_h,
                                    rec(static_cast<long long>(seqlen_k - kv0) * k.stride_k_n, 2));
        auto g_rb = opus::make_gmem(k_rot + static_cast<long long>(k0 + kv0) * k.stride_k_rot_n,
                                    rec(static_cast<long long>(seqlen_k - kv0) * k.stride_k_rot_n, 2));
        vec8_t kt[KT_ROUNDS];
#pragma unroll
        for(int it = 0; it < KT_ROUNDS; it++)
        {
            const int c = it * BLOCK + tid, r = c % BK, c8 = (c / BK) * 8;
            if(it < KT_NOPE)
                kt[it] = g_kb.template load<8>(r * k.stride_k_n + c8);
            else
                kt[it] = g_rb.template load<8>(r * k.stride_k_rot_n + c8 - D_NOPE);
        }
#pragma unroll
        for(int it = 0; it < KT_ROUNDS; it++)
        {
            const int c = it * BLOCK + tid, r = c % BK, c8 = (c / BK) * 8;
#pragma unroll
            for(int e = 0; e < 8; e++)
                smem[OFF_KT + (c8 + e) * LD_KT + r] = kt[it][e];
        }
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
    // LSE through lanes 0-15 of wave 0 and D through those of wave 1: choosing the tensor per wave
    // keeps each load's descriptor uniform, where a per-lane choice makes it a waterfall loop.
    const bool f_ld     = w < 2 && lane < BQ;
    const int ld_slot   = (w == 0 ? 0 : BQ) + lane;
    const float* ld_ptr = (w == 0 ? k.ptr_lse : k.ptr_d) + static_cast<long long>(h) * k.stride_lse_h + q0;
    vec8_t r_f[QR];
    float r_ld = 0.0f;
    auto fetch = [&](int i) {
        const int qt0 = i * BQ;
        auto g_f = opus::make_gmem(f_ptr + static_cast<long long>(qt0) * f_rows, rec_rows(seqlen_q - qt0, BQ, f_rows * 2));
#pragma unroll
        for(int u = 0; u < QR; u++)
            if(u < f_chunks)
                r_f[u] = g_f.template load<8>(f_vo + u * FC * 8);
        if(f_ld)
            r_ld = opus::make_gmem(ld_ptr + qt0, rec_rows(seqlen_q - qt0, BQ, 4)).template load<1>(lane)[0];
    };
    auto stage = [&](int buf) {
        if(f_ld)
            s_ld[buf][ld_slot] = r_ld;
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
                                        rec_rows(seqlen_q - qr, BQ, k.stride_dq_n * 4));
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
        // Most tiles need no mask for this wave's keys: all queries valid and, if causal, every
        // query at or past the wave's last key. The test is wave-uniform, so it is a scalar branch.
        const bool full = (qt0 + BQ <= seqlen_q) & (row_w + WR <= seqlen_k) &
                          (!k.causal | (qt0 + causal_offset >= row_w + WR - 1));
        if(full)
        {
#pragma unroll
            for(int e = 0; e < NS; e++)
                v_p[e] = __builtin_amdgcn_exp2f(v_st[e] * scale_log2 - lse4[e % 4] * LOG2_E);
        }
        else
        {
#pragma unroll
            for(int e = 0; e < NS; e++)
            {
                const int kv = row_w + pos_s[e] / 256, qp = qt0 + pos_s[e] % 256;
                // Non-short-circuit tests and a selected exponent keep the masked exp branch-free.
                const bool ok = (kv < seqlen_k) & (qp < seqlen_q) & (!k.causal | (qp + causal_offset >= kv));
                const float x = v_st[e] * scale_log2 - lse4[e % 4] * LOG2_E;
                v_p[e]        = __builtin_amdgcn_exp2f(ok ? x : -1e30f);
            }
        }
        const auto v_dpt = mma_dp(v_vw, s_do(b).template load<8>(u_do));
        opus::vector_t<float, NS> v_ds;
#pragma unroll
        for(int e = 0; e < NS; e++)
            v_ds[e] = v_p[e] * (v_dpt[e] - d4[e % 4]) * k.softmax_scale;
        const auto a_p  = opus::cast<bf16_t>(v_p);
        const auto a_ds = opus::cast<bf16_t>(v_ds);
        if constexpr(DQ)
            s_dst_w(b).template store<4>(a_ds, u_ct);
        acc_gemm(acc_dv, a_p, tr_load_b16<D_V / 16, LD_DO>(s_do(b), lane, 0));
        acc_gemm(acc_dkn, a_ds, tr_load_b16<D_NOPE / 16, LD_Q>(s_q(b), lane, 0));
        acc_gemm(acc_dkr, a_ds, tr_load_b16<D_ROT / 16, LD_Q>(s_q(b), lane, D_NOPE));

        if(i + 1 < n_q_tiles)
            stage(b ^ 1);
        if(i + 2 < n_q_tiles)
            fetch(i + 2);
        __syncthreads();

        if constexpr(DQ)
            dq_tile(b, qt0);
    }

    // dK nope and dV into the kv_b_proj layout, dK rope per head in fp32. The C fragments hold
    // 4 columns of 16 rows per instruction, so each wave transposes its 48 rows through LDS first
    // and stores whole rows; rows past seqlen_k fall outside the descriptors and are dropped.
    mfma_acc_drain();
    __syncthreads();
    {
        constexpr int LD_O = D_NOPE + D_V + 8;  // [48][dK nope | dV] bf16
        constexpr int LD_R = D_ROT + 4;         // [48][dK rope] fp32, over the same LDS
        static_assert(NW * WR * LD_O <= SMEM_ELEMS && WR * LD_R * 4 <= WR * LD_O * 2, "staging fits");
        bf16_t* s_o = smem + w * WR * LD_O;
        float* s_r  = reinterpret_cast<float*>(s_o);
        // C element (m, n, e): row 16 m + lane % 16, column 16 n + 4 (lane / 16) + e.
        const int c_row = lane % 16, c_col = 4 * (lane / 16);
#pragma unroll
        for(int m = 0; m < MR; m++)
#pragma unroll
            for(int n = 0; n < NV; n++)
            {
                bf16_t* o = s_o + (16 * m + c_row) * LD_O + 16 * n + c_col;
                *reinterpret_cast<bvec<4>*>(o)          = opus::cast<bf16_t>(acc_dkn[m][n]);
                *reinterpret_cast<bvec<4>*>(o + D_NOPE) = opus::cast<bf16_t>(acc_dv[m][n]);
            }
        bf16_t* dkv = reinterpret_cast<bf16_t*>(k.ptr_dkv) + static_cast<long long>(k0 + row_w) * k.stride_dkv_n +
                      static_cast<long long>(h) * k.stride_dkv_h;
        auto g_dkv = opus::make_gmem(dkv, rec(static_cast<long long>(rows_w) * k.stride_dkv_n, 2));
        // 32 lanes x 8 bf16 per row, two rows per instruction.
        constexpr int OC = (D_NOPE + D_V) / 8;
#pragma unroll
        for(int it = 0; it < WR * OC / 64; it++)
        {
            const int row = (it * 64 + lane) / OC, col = (lane % OC) * 8;
            g_dkv.template store<8>(*reinterpret_cast<const vec8_t*>(s_o + row * LD_O + col), row * k.stride_dkv_n + col);
        }
#pragma unroll
        for(int m = 0; m < MR; m++)
#pragma unroll
            for(int n = 0; n < NR; n++)
                *reinterpret_cast<f32x4_t*>(s_r + (16 * m + c_row) * LD_R + 16 * n + c_col) = acc_dkr[m][n];
        auto g_dkr = opus::make_gmem(k.ptr_dk_rope + static_cast<long long>(k0 + row_w) * k.stride_dkr_n +
                                         static_cast<long long>(h) * k.stride_dkr_h,
                                     rec(static_cast<long long>(rows_w) * k.stride_dkr_n, 4));
        // 16 lanes x 4 fp32 per row, four rows per instruction.
        constexpr int RC = D_ROT / 4;
#pragma unroll
        for(int it = 0; it < WR * RC / 64; it++)
        {
            const int row = (it * 64 + lane) / RC, col = (lane % RC) * 4;
            g_dkr.template store<4>(*reinterpret_cast<const f32x4_t*>(s_r + row * LD_R + col), row * k.stride_dkr_n + col);
        }
    }
}

// dQ pass without atomics. Each wave keeps its 32 queries' Q (the rope part in LDS) and dO, the
// A operands of S and dP, and dQ in registers and walks the keys in tiles of 32, which go global -> LDS directly, two
// tiles ahead, into NBUF buffers. All GEMMs are 16x16x32 MFMAs, C = A * B^T: dS has the C
// layout of S, and as the A operand of dQ = dS K it takes the tile's 32 keys as K, with the
// matching transposed reads of the K tile.
//
// Each tile is four stages, each closed by a workgroup barrier, alternating MFMA and VALU:
//   C1: S(t)                      (24 MFMA) and dQ(t - 1) on columns 0 .. 63    (8 MFMA);
//       DMA of tile t + 2 between the MFMAs
//   N1: P = exp(S(t) - LSE), kept in bf16
//   C2: dP(t)                     (16 MFMA) and dQ(t - 1) on columns 64 .. 191 (16 MFMA)
//   N2: dS(t) = P (dP(t) - D), in bf16
// Waves 4-7 run one stage behind waves 0-3 on the same SIMDs, so in every stage one wave of a
// SIMD issues 32 MFMAs while the other's VALU work fills the cycles beside them. dQ(t) runs a
// tile late, on dS(t) and tile t's K, so that both C stages carry 32 MFMAs; the last tile's dQ
// runs after the loop. In C stages the B fragments of each 4-MFMA step stream from LDS LA steps
// ahead, the first LA steps (and the stage's LDS addresses) made in the N stage before: C stages
// keep VALU out, whose lone instructions would wait behind the other wave's.
__global__ __launch_bounds__(gated_mla_bwd_dq::BLOCK, 1) void gated_mla_bwd_dq_kernel(gated_mla_bwd_dq_kargs k)
{
    using namespace gated_mla_bwd_dq;
    using gated_mla_bwd::bf16_t;
    using gated_mla_bwd::bvec;
    using gated_mla_bwd::tr_load_b_c_order;
    using opus::number;
    using opus::operator""_I;
    constexpr float LOG2_E = 1.44269504089f;
    constexpr int MR       = WQ / 16;  // M tiles per wave
    __shared__ __attribute__((aligned(16))) bf16_t smem[SMEM_ELEMS];

    const int tid  = static_cast<int>(opus::thread_id_x());
    const int lane = tid % 64;
    const int w    = __builtin_amdgcn_readfirstlane(tid / 64);
    // Workgroup id -> (head, sequence, query block) as in the main pass: groups of HG heads, the
    // head fastest within a group; query blocks run last-first, the causal long ones first.
    const int wg = static_cast<int>(opus::block_id_x());
    const int h  = wg % HG + HG * (wg / (HG * k.num_seqs * k.n_q_blocks));
    const int g  = wg / HG % k.num_seqs;
    const int qb = k.n_q_blocks - 1 - wg / (HG * k.num_seqs) % k.n_q_blocks;
    if(h >= k.H)
        return;
    auto sc            = [](int x) { return __builtin_amdgcn_readfirstlane(x); };
    const int q0       = sc(k.ptr_seqstart_q[g]);
    const int k0       = sc(k.ptr_seqstart_k[g]);
    const int seqlen_q = sc(k.ptr_seqstart_q[g + 1]) - q0;
    const int seqlen_k = sc(k.ptr_seqstart_k[g + 1]) - k0;
    const int qb0      = qb * BQ;
    if(qb0 >= seqlen_q)
        return;
    // Bottom-right causal: query qp sees keys kv <= qp + causal_offset. The block needs the keys
    // up to its last query's limit.
    const int causal_offset = seqlen_k - seqlen_q;
    const int q_last        = (qb0 + BQ < seqlen_q ? qb0 + BQ : seqlen_q) - 1;
    int kv_end              = seqlen_k;
    if(k.causal && q_last + causal_offset + 1 < kv_end)
        kv_end = q_last + causal_offset + 1;
    const int n_kv_tiles = kv_end > 0 ? (kv_end + KC - 1) / KC : 0;

    auto make_mma = [](auto e_m, auto e_n, auto e_k) {
        return opus::make_tiled_mma<bf16_t, bf16_t, float>(
            opus::seq<decltype(e_m)::value, decltype(e_n)::value, decltype(e_k)::value>{},
            opus::seq<1, 1, 1>{},
            opus::seq<16, 16, 32>{},
            opus::mfma_adaptor_swap_ab{});
    };
    static_assert(KC == 32, "dS is the A operand of dQ with one K step of 32 keys");
    auto mma_qn = make_mma(number<MR>{}, number<KC / 16>{}, number<D_NOPE / 32>{});  // for Q nope's A layout
    // Every MFMA step: [32 x 32] += A [32 x 32] B^T, 4 MFMAs. S and dP: one K step of 32 dims;
    // dQ: one chunk of 32 columns over the tile's 32 keys.
    auto mma_k1 = make_mma(number<MR>{}, number<KC / 16>{}, number<1>{});
    auto mma_dp = make_mma(number<MR>{}, number<KC / 16>{}, number<D_V / 32>{});   // for dO's A layout
    const auto c_ab = opus::make_tuple(0_I, lane % mma_k1.grpm_a, 0_I, lane / mma_k1.grpm_a);
    const auto c_c  = opus::make_tuple(0_I, lane % mma_k1.grpn_c, 0_I, lane / mma_k1.grpn_c);

    auto rec = [](long long elems, int elem_bytes) -> unsigned int {
        const long long b = elems > 0 ? elems * elem_bytes : 0;
        return b >= 0xffffffffLL ? 0xffffffffu : static_cast<unsigned int>(b);
    };

    // This wave's 32 queries: Q and dO as A operands, LSE (log2 domain) and D at its C rows
    // 16 tm + lane % 16. Rows past seqlen_q read as 0 and are dropped on store.
    const int qw0    = qb0 + w * WQ;
    const int rows_q = seqlen_q - qw0 < WQ ? (seqlen_q - qw0 > 0 ? seqlen_q - qw0 : 0) : WQ;
    auto g_q  = opus::make_gmem(reinterpret_cast<const bf16_t*>(k.ptr_q) + static_cast<long long>(q0 + qw0) * k.stride_q_n +
                                   static_cast<long long>(h) * k.stride_q_h,
                               rec(static_cast<long long>(rows_q) * k.stride_q_n, 2));
    auto g_do = opus::make_gmem(reinterpret_cast<const bf16_t*>(k.ptr_do) + static_cast<long long>(q0 + qw0) * k.stride_do_n +
                                    static_cast<long long>(h) * k.stride_do_h,
                                rec(static_cast<long long>(rows_q) * k.stride_do_n, 2));
    const auto v_qw  = g_q.template load<8>(opus::partition_layout_a<8>(mma_qn, opus::make_tuple(k.stride_q_n, 1_I), c_ab));
    // Q rope rows of this wave -> sQR (lane l: row l / 2, chunks 4 (l % 2) .. + 3).
    {
        using vec8_t = bvec<8>;
        const int r = lane / 2, c0 = (lane % 2) * (D_ROT / 2);
        vec8_t t_[D_ROT / 2 / 8];
#pragma unroll
        for(int u = 0; u < D_ROT / 2 / 8; u++)
            t_[u] = g_q.template load<8>(r * k.stride_q_n + D_NOPE + c0 + u * 8);
#pragma unroll
        for(int u = 0; u < D_ROT / 2 / 8; u++)
            *reinterpret_cast<vec8_t*>(smem + OFF_QR + (w * WQ + r) * LD_QR + c0 + u * 8) = t_[u];
    }
    const auto v_dow = g_do.template load<8>(opus::partition_layout_a<8>(mma_dp, opus::make_tuple(k.stride_do_n, 1_I), c_ab));
    float lse2[MR], dd[MR];
    {
        auto g_lse = opus::make_gmem(k.ptr_lse + static_cast<long long>(h) * k.stride_lse_h + q0 + qw0, rec(rows_q, 4));
        auto g_d   = opus::make_gmem(k.ptr_d + static_cast<long long>(h) * k.stride_lse_h + q0 + qw0, rec(rows_q, 4));
#pragma unroll
        for(int tm = 0; tm < MR; tm++)
        {
            lse2[tm] = g_lse.template load<1>(16 * tm + lane % 16)[0] * LOG2_E;
            dd[tm]   = g_d.template load<1>(16 * tm + lane % 16)[0];
        }
    }

    // DMA of key tile t into buffer t % NBUF (gated_mla_bwd::dma16). Load j of a region writes its
    // 16-byte slots 64 j .. + 63; slot s is row s / SPR, chunk s % SPR, the last two chunks of a
    // row being padding. Padding and keys past seqlen_k read out of bounds, as 0. The tile's
    // J_N + J_R + J_V loads go to wave w as loads w, w + NW, w + 2 NW.
    constexpr int SPR_N = LD_KN * 2 / 16, SPR_R = LD_KR * 2 / 16, SPR_V = LD_V * 2 / 16;
    constexpr int J_N = KC * SPR_N / 64, J_R = KC * SPR_R / 64, J_V = KC * SPR_V / 64;
    constexpr int JU  = (J_N + J_R + J_V + NW - 1) / NW;
    static_assert(KC * SPR_N % 64 == 0 && KC * SPR_R % 64 == 0 && KC * SPR_V % 64 == 0, "whole loads per region");
    constexpr int OOB = 1 << 30;  // byte offset past any tile
    const bf16_t* kn_ptr = reinterpret_cast<const bf16_t*>(k.ptr_k_nope) + static_cast<long long>(k0) * k.stride_k_n +
                           static_cast<long long>(h) * k.stride_k_h;
    const bf16_t* kr_ptr = reinterpret_cast<const bf16_t*>(k.ptr_k_rot) + static_cast<long long>(k0) * k.stride_k_rot_n;
    const bf16_t* v_ptr  = reinterpret_cast<const bf16_t*>(k.ptr_v) + static_cast<long long>(k0) * k.stride_v_n +
                          static_cast<long long>(h) * k.stride_v_h;
    // Per load round u, wave-uniform: source pointer, row bytes, place in the buffer; and this
    // lane's source offset within the tile. A wave without a load in its last round repeats its
    // previous one, the same bytes to the same place, so that no round takes a branch.
    const bf16_t* d_ptr[JU];
    int d_rb[JU], d_lds[JU], d_os[JU];
#pragma unroll
    for(int u = 0; u < JU; u++)
    {
        const int j   = w + NW * u < J_N + J_R + J_V ? w + NW * u : w + NW * (u - 1);
        const int reg = j < J_N ? 0 : (j < J_N + J_R ? 1 : 2);
        const int jj  = reg == 0 ? j : (reg == 1 ? j - J_N : j - J_N - J_R);
        const int rs  = reg == 0 ? k.stride_k_n : (reg == 1 ? k.stride_k_rot_n : k.stride_v_n);
        d_rb[u]       = rs * 2;
        d_ptr[u]      = reg == 0 ? kn_ptr : (reg == 1 ? kr_ptr : v_ptr);
        d_lds[u]      = (reg == 0 ? OFF_KN : (reg == 1 ? OFF_KR : OFF_V)) * 2 + jj * 1024;
        const int spr = reg == 0 ? SPR_N : (reg == 1 ? SPR_R : SPR_V);
        const int dch = reg == 0 ? D_NOPE / 8 : (reg == 1 ? D_ROT / 8 : D_V / 8);
        const int sl = 64 * jj + lane, c = sl % spr;
        d_os[u]      = c < dch ? sl / spr * d_rb[u] + c * 16 : OOB;
    }
    // Round u of a tile's DMA: `rows` keys into the buffer at LDS address lb. Tiles go in order, so
    // the source pointer just advances. Rounds past the last tile fill a buffer no one reads
    // anymore (with zeros past seqlen_k).
    // Key rows of tile t, in SALU: the compiler would clamp with v_med3 and move the result back
    // with a readfirstlane, which waits behind the other wave's VALU work.
    auto dma_rows  = [&](int t) {
        int r = seqlen_k - t * KC;
        asm("s_min_i32 %0, %0, %1\n\ts_max_i32 %0, %0, 0" : "+s"(r) : "n"(KC));
        return r;
    };
    auto dma_lds   = [&](int t) { return gated_mla_bwd::lds_addr(smem + t % NBUF * SZ_BUF); };
    auto dma_round = [&](int rows, unsigned int lb, int u) {
        auto rs = gated_mla_bwd::dma_rsrc(d_ptr[u], 0);
        asm("s_mul_i32 %0, %1, %2" : "=s"(rs[2]) : "s"(rows), "s"(d_rb[u]));
        gated_mla_bwd::dma16(rs, d_os[u], lb + d_lds[u]);
        d_ptr[u] += KC * d_rb[u] / 2;
    };
    auto dma = [&](int t) {
#pragma unroll
        for(int u = 0; u < JU; u++)
            dma_round(dma_rows(t), dma_lds(t), u);
    };
    // The DMA writes land in LDS when vmcnt says so, which s_barrier does not wait for.
    auto dma_wait = [] { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); };
    auto stage_end = [] {
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    };
    // sched_barrier only fences one basic block: values a stage produces are pinned at its end (an
    // empty asm that reads and writes them), so the compiler cannot sink their computation into
    // the next stage and keep the inputs live across it.
    auto pin = [](auto& x) { asm volatile("" : "+v"(x)); };

    // Key tile fragments are read at a per-lane LDS address plus immediates. The address of an
    // operand in a tile's buffer, one VGPR add, is made in the N stage before the C stage that
    // reads it and pinned there: in a C stage a lone VALU instruction waits behind the other
    // wave's VALU work. Layouts with lane coordinate 0 give the immediates.
    auto lay_b       = [&](auto ld, const auto& c) {
        return opus::partition_layout_b<8>(mma_k1, opus::make_tuple(ld, 1_I), c);
    };
    const auto c_0   = opus::make_tuple(0_I, 0_I, 0_I, 0_I);
    const auto u_kn0 = lay_b(number<LD_KN>{}, c_0);
    const auto u_kr0 = lay_b(number<LD_KR>{}, c_0);
    const auto u_v0  = lay_b(number<LD_V>{}, c_0);
    auto lane_b      = [&](auto ld) {
        return opus::layout_to_offsets<8>(lay_b(ld, c_ab))[0] - opus::layout_to_offsets<8>(lay_b(ld, c_0))[0];
    };
    const int o_kn = lane_b(number<LD_KN>{}), o_kr = lane_b(number<LD_KR>{}), o_v = lane_b(number<LD_V>{});
    const int o_tn = gated_mla_bwd::tr_c_order_lane<LD_KN>(lane), o_tr = gated_mla_bwd::tr_c_order_lane<LD_KR>(lane);
    auto addr = [&](int t, int off, int o) {
        unsigned int a = gated_mla_bwd::lds_addr(smem + t % NBUF * SZ_BUF + off) + o * 2;
        pin(a);
        return a;
    };
    auto at = [](unsigned int a) { return opus::make_smem(reinterpret_cast<bf16_t*>(static_cast<__UINTPTR_TYPE__>(a))); };
    // S(t): K nope, K rope; dP(t): V; dQ(t - 1): K nope, K rope, transposed.
    unsigned int a_kn, a_kr, a_v, a_tn, a_tr;
    using frag_t  = bvec<16>;
    using afrag_t = opus::vector_t<bf16_t, decltype(mma_k1)::tile_a_len>;
    using acc_t   = typename decltype(mma_k1)::vtype_c;
    constexpr int KS_S = D_QK / 32, KS_N = D_NOPE / 32, KS_R = D_ROT / 32, KS_P = D_V / 32, NQC = D_QK / 32;
    constexpr int LA_  = 2;  // steps of fragments read ahead of their MFMAs (see c_stage)
    // B fragments of one step: S K step ks, dP K step ks, dQ column chunk c.
    auto ld_s = [&](int ks) -> frag_t {
        return ks < KS_N ? at(a_kn).template load<8>(u_kn0, ks * 32) : at(a_kr).template load<8>(u_kr0, (ks - KS_N) * 32);
    };
    auto ld_dp = [&](int ks) -> frag_t { return at(a_v).template load<8>(u_v0, ks * 32); };
    auto ld_dq = [&](int c) -> frag_t {
        return c < KS_N ? gated_mla_bwd::tr_load_b_c_order_at<2, LD_KN>(at(a_tn), 32 * c)
                        : gated_mla_bwd::tr_load_b_c_order_at<2, LD_KR>(at(a_tr), 32 * (c - KS_N));
    };
    // A fragments of Q (nope part, from registers; rope part, from sQR) and dO for one K step.
    auto a_q = [&](int ks) {
        afrag_t a;
#pragma unroll
        for(int m = 0; m < MR; m++)
#pragma unroll
            for(int x = 0; x < 8; x++)
                a[m * 8 + x] = v_qw[(m * KS_N + ks) * 8 + x];
        return a;
    };
    auto s_qr       = opus::make_smem(smem + OFF_QR + w * WQ * LD_QR);
    const auto u_qr = opus::partition_layout_a<8>(mma_k1, opus::make_tuple(number<LD_QR>{}, 1_I), c_ab);
    // The rope K steps run last; their A fragments are read LA steps ahead, like the B ones.
    afrag_t a_qr[KS_R];
    static_assert(KS_R <= LA_ && KS_N >= LA_, "rope A fragments read within S's nope steps");
    auto a_do = [&](int ks) {
        afrag_t a;
#pragma unroll
        for(int m = 0; m < MR; m++)
#pragma unroll
            for(int x = 0; x < 8; x++)
                a[m * 8 + x] = v_dow[(m * KS_P + ks) * 8 + x];
        return a;
    };

    constexpr int NS_ = decltype(mma_k1)::tile_c_len;
    acc_t v_s, v_dp, v_dq[NQC];
#pragma unroll
    for(int c = 0; c < NQC; c++)
        opus::clear(v_dq[c]);
    // dS of the previous tile, the A operand of its dQ. Zero before tile 0, whose dQ(t - 1) steps
    // then add nothing (they read tile 0's K, which is finite), so that tile 0 runs the loop body.
    afrag_t a_ds;
    opus::clear(a_ds);
    // P in bf16 from N1 to N2: it waits out C2 there, where the registers are tightest, and dS is
    // rounded to bf16 right after anyway.
    opus::vector_t<bf16_t, NS_> p_bf;
    // B fragments of LA steps in flight, in a ring of LA slots: a step issues its MFMAs, then reads
    // the fragments of step + LA into the slot they have just consumed.
    constexpr int LA = LA_;
    frag_t ring[LA];
    auto c_stage = [&](auto n_steps, auto ld, auto ex) {
        constexpr int N = decltype(n_steps)::value;
#pragma unroll
        for(int st = 0; st < N; st++)
        {
            ex(st, ring[st % LA]);
            if(st + LA < N)
                ring[st % LA] = ld(st + LA);
            __builtin_amdgcn_sched_barrier(0);
        }
    };
    auto prefetch = [&](auto ld) {
#pragma unroll
        for(int st = 0; st < LA; st++)
            ring[st] = ld(st);
    };
    const acc_t zero{};
    auto ex_s = [&](int ks, const frag_t& b) {
        v_s = mma_k1(ks < KS_N ? a_q(ks) : a_qr[(ks - KS_N) % KS_R], b, ks == 0 ? zero : v_s);
        if(ks + LA_ >= KS_N && ks + LA_ < KS_S)
            a_qr[ks + LA_ - KS_N] = s_qr.template load<8>(u_qr, (ks + LA_ - KS_N) * 32);
    };
    auto ex_dp = [&](int ks, const frag_t& b) { v_dp = mma_k1(a_do(ks), b, ks == 0 ? zero : v_dp); };
    auto ex_dq = [&](int c, const frag_t& b) { v_dq[c] = mma_k1(a_ds, b, v_dq[c]); };

    // C element e = (tm 2 + tn) 4 + p of S and dP is query 16 tm + lane % 16, key 16 tn +
    // 4 (lane / 16) + p of the wave's tile.
    constexpr int NS = decltype(mma_k1)::tile_c_len;
    static_assert(NS == MR * 2 * 4);
    const float scale_log2 = k.softmax_scale * LOG2_E;
    const int T            = n_kv_tiles;

    // N1: the first steps of C2 (and its addresses); P = exp(S(t) - LSE), to p_bf.
    auto n1 = [&](int t, int tp) {
        a_v  = addr(t, OFF_V, o_v);
        a_tr = addr(tp, OFF_KR, o_tr);
        prefetch(ld_dp);
        const int kv0 = t * KC;
        // As in the main pass: tiles this wave sees in full take an unmasked exp.
        const bool full = (kv0 + KC <= seqlen_k) & (qw0 + WQ <= seqlen_q) &
                          (!k.causal | (kv0 + KC - 1 <= qw0 + causal_offset));
        if(full)
        {
#pragma unroll
            for(int e = 0; e < NS; e++)
                v_s[e] = __builtin_amdgcn_exp2f(v_s[e] * scale_log2 - lse2[e / (NS / MR)]);
        }
        else
        {
#pragma unroll
            for(int e = 0; e < NS; e++)
            {
                const int qp = qw0 + lane % 16 + 16 * (e / 8), kv = kv0 + 4 * (lane / 16) + 16 * (e / 4 % 2) + e % 4;
                const bool ok = (kv < seqlen_k) & (qp < seqlen_q) & (!k.causal | (qp + causal_offset >= kv));
                const float x = v_s[e] * scale_log2 - lse2[e / (NS / MR)];
                v_s[e]        = __builtin_amdgcn_exp2f(ok ? x : -1e30f);
            }
        }
        p_bf = opus::cast<bf16_t>(v_s);
        pin(p_bf);
#pragma unroll
        for(int r = 0; r < LA; r++)
            pin(ring[r]);
    };
    // N2: the first steps of the next C stage (C1 of tile t + 1, or the tail's dQ of tile t);
    // dS(t) = P (dP(t) - D) in bf16.
    auto n2 = [&](int t) {
        a_tn = addr(t, OFF_KN, o_tn);
        if(t + 1 < T)
        {
            a_kn = addr(t + 1, OFF_KN, o_kn);
            a_kr = addr(t + 1, OFF_KR, o_kr);
            prefetch(ld_s);
        }
        else
        {
            a_tr = addr(t, OFF_KR, o_tr);
            prefetch(ld_dq);
        }
        opus::vector_t<float, NS> v_ds;
#pragma unroll
        for(int e = 0; e < NS; e++)
            v_ds[e] = static_cast<float>(p_bf[e]) * (v_dp[e] - dd[e / (NS / MR)]);
        a_ds = opus::cast<bf16_t>(v_ds);
        pin(a_ds);
#pragma unroll
        for(int r = 0; r < LA; r++)
            pin(ring[r]);
    };

    // Tiles 0 and 1 in LDS, and Q, dO, LSE, D in registers, before the loop (0x0f70 is
    // s_waitcnt vmcnt(0)): otherwise the compiler waits for the latter inside the loop with
    // vmcnt counts that the DMA then makes block.
    if(T > 0)
        dma(0);
    if(T > 1)
        dma(1);
    __builtin_amdgcn_s_waitcnt(0x0f70);
    dma_wait();
    stage_end();
    const bool lag = w >= NW / 2;
    if(lag)
        stage_end();
    if(T > 0)
    {
        a_kn = addr(0, OFF_KN, o_kn);
        a_kr = addr(0, OFF_KR, o_kr);
        a_tn = addr(0, OFF_KN, o_tn);
        prefetch(ld_s);
        for(int t = 0; t < T; t++)
        {
            const int tp = t > 0 ? t - 1 : 0;
            // C1: S(t), then dQ(t - 1) on chunks 0, 1. Steps 1 .. JU also issue the DMA of tile
            // t + 2, a round each, beside their MFMAs; tile t + 1's, issued a tile ago, is waited
            // for first. All waves have waited for their share of tile t + 1 by the barrier that
            // ends the lagging waves' C1, before any N2 reads it. Buffer (t + 2) % NBUF last held
            // tile t - 2, whose dQ every wave finished in the stage before this one.
            const int r2          = dma_rows(t + 2);
            const unsigned int l2 = dma_lds(t + 2);
            c_stage(number<KS_S + 2>{},
                    [&](int st) { return st < KS_S ? ld_s(st) : ld_dq(st - KS_S); },
                    [&](int st, const frag_t& b) {
                        if(st < KS_S)
                            ex_s(st, b);
                        else
                            ex_dq(st - KS_S, b);
                        if(st == 1)
                            dma_wait();
                        if(st >= 1 && st <= JU)
                            dma_round(r2, l2, st - 1);
                    });
            pin(v_s);
            stage_end();
            n1(t, tp);
            stage_end();
            // C2: dP(t), then dQ(t - 1) on chunks 2 .. 5.
            c_stage(number<KS_P + NQC - 2>{},
                    [&](int st) { return st < KS_P ? ld_dp(st) : ld_dq(st - KS_P + 2); },
                    [&](int st, const frag_t& b) {
                        if(st < KS_P)
                            ex_dp(st, b);
                        else
                            ex_dq(st - KS_P + 2, b);
                    });
            pin(v_dp);
            stage_end();
            n2(t);
            stage_end();
        }
        // Tail: dQ of the last tile.
        c_stage(number<NQC>{}, ld_dq, ex_dq);
        stage_end();
    }
    if(!lag)
        stage_end();

    auto g_dq = opus::make_gmem(reinterpret_cast<bf16_t*>(k.ptr_dq) + static_cast<long long>(q0 + qw0) * k.stride_dq_n +
                                    static_cast<long long>(h) * k.stride_dq_h,
                                rec(static_cast<long long>(rows_q) * k.stride_dq_n, 2));
    const auto u_dq = opus::partition_layout_c(mma_k1, opus::make_tuple(k.stride_dq_n, 1_I), c_c);
#pragma unroll
    for(int c = 0; c < NQC; c++)
    {
#pragma unroll
        for(int e = 0; e < NS; e++)
            v_dq[c][e] *= k.softmax_scale;
        g_dq.template store<4>(opus::cast<bf16_t>(v_dq[c]), u_dq, c * 32);
    }
}

// dK / dV pass, 8 waves. The two waves of a SIMD (w, w + 4) share 32 keys and split the work
// along its two dependency chains, so each LDS fragment of Q and dO feeds two M tiles:
//   S wave  (w < 4):  S^T = K Q^T, P, dV += P^T dO
//   dP wave (w >= 4): dP^T = V dO^T, dS = P (dP - D), dK += dS^T Q, and the query-tile fetch
// P crosses from the S wave to the dP wave through LDS in fp32. Each half of a tile is 40 MFMAs
// per SIMD (S^T and dP^T, then dV and dK), with the exp and the dS math beside the other wave's
// MFMAs. Each role runs its own loop so that a wave only keeps its own role's registers live;
// both loops pass the same two barriers per tile.
__global__ __launch_bounds__(gated_mla_bwd_pair::BLOCK, 1) void gated_mla_bwd_dkdv_pair_kernel(gated_mla_bwd_kargs k)
{
    using namespace gated_mla_bwd_pair;
    using gated_mla_bwd::bf16_t;
    using gated_mla_bwd::bvec;
    using gated_mla_bwd::tr_load_b_c_order;
    using opus::number;
    using opus::operator""_I;
    using vec8_t  = bvec<8>;
    using f32x4_t = opus::vector_t<float, 4>;
    constexpr float LOG2_E = 1.44269504089f;
    constexpr int MR       = PR / 16;          // M tiles per wave
    constexpr int NT       = BQ / 16;          // N tiles of S^T / dP^T
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
    // Workgroup id -> (head, sequence, key block) as in the 4-wave pass.
    const int wg = static_cast<int>(opus::block_id_x());
    const int h  = wg % HG + HG * (wg / (HG * k.num_seqs * k.n_kv_blocks));
    const int g  = wg / HG % k.num_seqs;
    const int j  = wg / (HG * k.num_seqs) % k.n_kv_blocks;
    if(h >= k.H)
        return;
    auto sc            = [](int x) { return __builtin_amdgcn_readfirstlane(x); };
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
    static_assert(BQ == 32, "the C -> A reuse of P^T / dS^T needs the query tile to be one K step");
    auto mma_s   = make_mma(number<MR>{}, number<NT>{}, number<D_QK / 32>{});      // S^T  [32 x BQ]
    auto mma_kn  = make_mma(number<MR>{}, number<NT>{}, number<D_NOPE / 32>{});    // A of S^T, nope part
    auto mma_kr  = make_mma(number<MR>{}, number<NT>{}, number<D_ROT / 32>{});     // A of S^T, rope part
    auto mma_dp  = make_mma(number<MR>{}, number<NT>{}, number<D_V / 32>{});       // dP^T [32 x BQ]
    auto mma_dv  = make_mma(number<MR>{}, number<D_V / 16>{}, number<BQ / 32>{});   // dV, dK nope [32 x 128]
    auto mma_dkr = make_mma(number<MR>{}, number<D_ROT / 16>{}, number<BQ / 32>{}); // dK rope [32 x 64]
    const auto c_ab = opus::make_tuple(0_I, lane % mma_s.grpm_a, 0_I, lane / mma_s.grpm_a);
    const auto c_c  = opus::make_tuple(0_I, lane % mma_s.grpn_c, 0_I, lane / mma_s.grpn_c);

    auto rec = [](long long elems, int elem_bytes) -> unsigned int {
        const long long b = elems > 0 ? elems * elem_bytes : 0;
        return b >= 0xffffffffLL ? 0xffffffffu : static_cast<unsigned int>(b);
    };
    // Bytes of `rows` rows clamped to [0, cap], uniform (see the 4-wave pass).
    auto rec_rows = [](int rows, int cap, int row_bytes) -> unsigned int {
        return static_cast<unsigned int>(
            __builtin_amdgcn_readfirstlane((rows < 0 ? 0 : (rows > cap ? cap : rows)) * row_bytes));
    };

    // This pair's 32 keys; rows past seqlen_k read as 0 and are dropped on store.
    const int row_w  = kv0 + pr * PR;
    const int rows_w = seqlen_k - row_w < PR ? (seqlen_k - row_w > 0 ? seqlen_k - row_w : 0) : PR;
    const bf16_t* k_nope = reinterpret_cast<const bf16_t*>(k.ptr_k_nope);
    const bf16_t* k_rot  = reinterpret_cast<const bf16_t*>(k.ptr_k_rot);

    auto s_q  = [&](int b) { return opus::make_smem(smem + OFF_Q + b * BQ * LD_Q); };
    auto s_do = [&](int b) { return opus::make_smem(smem + OFF_DO + b * BQ * LD_DO); };
    const auto u_q  = opus::partition_layout_b<8>(mma_s, opus::make_tuple(number<LD_Q>{}, 1_I), c_ab);
    const auto u_do = opus::partition_layout_b<8>(mma_dp, opus::make_tuple(number<LD_DO>{}, 1_I), c_ab);
    constexpr int NS = decltype(mma_s)::tile_c_len;
    static_assert(NS == MR * NT * 4);
    const float scale_log2 = k.softmax_scale * LOG2_E;
    // C element e = (tm NT + tn) 4 + p of S^T / dP^T is key 16 tm + lane % 16 of the pair and
    // query 16 tn + 4 (lane / 16) + p of the tile, so each N tile is one float4 run of s_ld.
    auto ld_c = [&](int b, int off, float mul, float(&out)[NS]) {
        f32x4_t run[NT];
#pragma unroll
        for(int tn = 0; tn < NT; tn++)
            run[tn] = *reinterpret_cast<const f32x4_t*>(&s_ld[b][off + 16 * tn + 4 * (lane / 16)]);
#pragma unroll
        for(int e = 0; e < NS; e++)
            out[e] = run[(e / 4) % NT][e % 4] * mul;
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
        const int c_kv = row_w + lane % 16, c_q = 4 * (lane / 16);

        __syncthreads();

        typename decltype(mma_dv)::vtype_c v_dv;
        opus::clear(v_dv);
        for(int i = i_begin; i < n_q_tiles; i++)
        {
            const int b   = (i - i_begin) & 1;
            const int qt0 = i * BQ;
            float lse_e[NS];
            ld_c(b, 0, LOG2_E, lse_e);
            const auto v_st = mma_s(v_kw, s_q(b).template load<8>(u_q));
            opus::vector_t<float, NS> v_p;
            // Tiles this pair sees in full take an unmasked exp (a scalar branch).
            const bool full = (qt0 + BQ <= seqlen_q) & (row_w + PR <= seqlen_k) &
                              (!k.causal | (qt0 + causal_offset >= row_w + PR - 1));
            if(full)
            {
#pragma unroll
                for(int e = 0; e < NS; e++)
                    v_p[e] = __builtin_amdgcn_exp2f(v_st[e] * scale_log2 - lse_e[e]);
            }
            else
            {
#pragma unroll
                for(int e = 0; e < NS; e++)
                {
                    const int kv = c_kv + 16 * (e / (4 * NT)), qp = qt0 + c_q + 16 * (e / 4 % NT) + e % 4;
                    const bool ok = (kv < seqlen_k) & (qp < seqlen_q) & (!k.causal | (qp + causal_offset >= kv));
                    const float x = v_st[e] * scale_log2 - lse_e[e];
                    v_p[e]        = __builtin_amdgcn_exp2f(ok ? x : -1e30f);
                }
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
            __syncthreads();
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
        // fall outside the descriptors and read as 0. LSE comes from wave NP and D from wave
        // NP + 1, lanes 0 .. BQ - 1: a per-wave base keeps the descriptor uniform.
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
        const bool f_ld     = w < NP + 2;
        const float* ld_ptr = (w == NP ? k.ptr_lse : k.ptr_d) + static_cast<long long>(h) * k.stride_lse_h + q0;
        const int ld_slot   = (w == NP ? 0 : BQ) + lane;
        vec8_t r_q[QR], r_do[DR];
        float r_ld = 0.0f;
        auto fetch = [&](int i) {
            const int qt0 = i * BQ;
            auto g_q  = opus::make_gmem(q_ptr + static_cast<long long>(qt0) * k.stride_q_n,
                                       rec_rows(seqlen_q - qt0, BQ, k.stride_q_n * 2));
            auto g_do = opus::make_gmem(do_ptr + static_cast<long long>(qt0) * k.stride_do_n,
                                        rec_rows(seqlen_q - qt0, BQ, k.stride_do_n * 2));
#pragma unroll
            for(int u = 0; u < QR; u++)
                r_q[u] = g_q.template load<8>(vo_q + u * FC * 8);
#pragma unroll
            for(int u = 0; u < DR; u++)
                r_do[u] = g_do.template load<8>(vo_do + u * FC * 8);
            if(f_ld)
            {
                auto g_ld = opus::make_gmem(ld_ptr + qt0, rec_rows(seqlen_q - qt0, BQ, 4));
                r_ld      = g_ld.template load<1>(lane % BQ)[0];
            }
        };
        auto stage = [&](int buf) {
            if(f_ld && lane < BQ)
                s_ld[buf][ld_slot] = r_ld;
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

// dK / dV pass, 4 waves. Each wave runs at one per SIMD so that its 48 keys' K and V (A of
// S^T and dP^T) stay in VGPRs and dK and dV in AGPRs: every LDS fragment of Q and dO feeds
// three M tiles. GEMMs are per wave, C = A * B^T, all 16x16x32 MFMAs: S^T and dP^T over d; dV
// and dK over the tile's 32 queries, with P^T and dS^T (the C fragments of S^T and dP^T, two N
// tiles per M tile) as their A operands and transposed reads of the dO and Q tiles as B. Query
// tiles go global -> LDS one tile ahead into the other buffer, one barrier per tile.
__global__ __launch_bounds__(gated_mla_bwd_4w::BLOCK, 1) void gated_mla_bwd_dkdv_4w_kernel(gated_mla_bwd_kargs k)
{
    using namespace gated_mla_bwd_4w;
    using gated_mla_bwd::acc_gemm32;
    using gated_mla_bwd::bf16_t;
    using gated_mla_bwd::bvec;
    using gated_mla_bwd::f32x4_t;
    using gated_mla_bwd::mfma_acc_drain;
    using gated_mla_bwd::tr_load_b_c_order;
    using opus::number;
    using opus::operator""_I;
    using vec8_t = bvec<8>;
    constexpr float LOG2_E = 1.44269504089f;
    constexpr int MR       = WR / 16;  // M tiles per wave
    constexpr int NT       = BQ / 16;  // N tiles of S^T / dP^T
    __shared__ __attribute__((aligned(16))) bf16_t smem[SMEM_ELEMS];
    // LSE and D of a query tile, [lse | D], double-buffered like sQ.
    __shared__ __attribute__((aligned(16))) float s_ld[2][2 * BQ];

    const int tid  = static_cast<int>(opus::thread_id_x());
    const int lane = tid % 64;
    const int w    = __builtin_amdgcn_readfirstlane(tid / 64);
    // Workgroup id -> (head, sequence, key block) as in gated_mla_bwd_kernel.
    const int wg = static_cast<int>(opus::block_id_x());
    const int h  = wg % HG + HG * (wg / (HG * k.num_seqs * k.n_kv_blocks));
    const int g  = wg / HG % k.num_seqs;
    const int j  = wg / (HG * k.num_seqs) % k.n_kv_blocks;
    if(h >= k.H)
        return;
    auto sc            = [](int x) { return __builtin_amdgcn_readfirstlane(x); };
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

    auto make_mma = [](auto e_m, auto e_n, auto e_k) {
        return opus::make_tiled_mma<bf16_t, bf16_t, float>(
            opus::seq<decltype(e_m)::value, decltype(e_n)::value, decltype(e_k)::value>{},
            opus::seq<1, 1, 1>{},
            opus::seq<16, 16, 32>{},
            opus::mfma_adaptor_swap_ab{});
    };
    static_assert(D_V == D_NOPE, "dV and the nope part of dK share one tile shape");
    static_assert(BQ == 32, "P^T / dS^T are the A operand of one 16x16x32 K step");
    auto mma_s  = make_mma(number<MR>{}, number<NT>{}, number<D_QK / 32>{});    // S^T  [48 x 32]
    auto mma_kn = make_mma(number<MR>{}, number<NT>{}, number<D_NOPE / 32>{});  // A of S^T, nope part
    auto mma_kr = make_mma(number<MR>{}, number<NT>{}, number<D_ROT / 32>{});   // A of S^T, rope part
    auto mma_dp = make_mma(number<MR>{}, number<NT>{}, number<D_V / 32>{});     // dP^T [48 x 32]
    auto mma1   = make_mma(number<1>{}, number<1>{}, number<1>{});
    const auto c_ab = opus::make_tuple(0_I, lane % mma_s.grpm_a, 0_I, lane / mma_s.grpm_a);

    auto rec = [](long long elems, int elem_bytes) -> unsigned int {
        const long long b = elems > 0 ? elems * elem_bytes : 0;
        return b >= 0xffffffffLL ? 0xffffffffu : static_cast<unsigned int>(b);
    };
    // Bytes of `rows` rows clamped to [0, cap], uniform (see gated_mla_bwd_kernel).
    auto rec_rows = [](int rows, int cap, int row_bytes) -> unsigned int {
        return static_cast<unsigned int>(
            __builtin_amdgcn_readfirstlane((rows < 0 ? 0 : (rows > cap ? cap : rows)) * row_bytes));
    };

    // This wave's 48 keys as the A operand of S^T and dP^T; rows past seqlen_k read as 0.
    const int row_w  = kv0 + w * WR;
    const int rows_w = seqlen_k - row_w < WR ? (seqlen_k - row_w > 0 ? seqlen_k - row_w : 0) : WR;
    auto g_kn = opus::make_gmem(reinterpret_cast<const bf16_t*>(k.ptr_k_nope) + static_cast<long long>(k0 + row_w) * k.stride_k_n +
                                    static_cast<long long>(h) * k.stride_k_h,
                                rec(static_cast<long long>(rows_w) * k.stride_k_n, 2));
    auto g_kr = opus::make_gmem(reinterpret_cast<const bf16_t*>(k.ptr_k_rot) + static_cast<long long>(k0 + row_w) * k.stride_k_rot_n,
                                rec(static_cast<long long>(rows_w) * k.stride_k_rot_n, 2));
    auto g_v  = opus::make_gmem(reinterpret_cast<const bf16_t*>(k.ptr_v) + static_cast<long long>(k0 + row_w) * k.stride_v_n +
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

    // Q and dO tiles go global -> LDS directly (gated_mla_bwd::dma16), one tile ahead, so they take
    // no staging registers. Each such load writes 64 lanes x 16 B contiguously from a wave-uniform
    // LDS address, so load j of a tensor covers 16-byte slots 64 j .. + 63 of its tile laid out
    // with the row padding: slot s of Q is row s / 26, chunk s % 26 (chunks 24, 25 are padding),
    // of dO row s / 18, chunk s % 18 (16, 17 padding). Padding slots and rows past seqlen_q read
    // out of bounds, as 0. Of a tile's 13 Q and 9 dO loads, wave w issues Q loads w, w + 4, w + 8
    // and dO loads w, w + 4; wave 0 also Q load 12, wave 1 dO load 8.
    constexpr int QS = LD_Q * 2 / 16, DS = LD_DO * 2 / 16;  // slots per padded row
    static_assert(BQ * QS == 13 * 64 && BQ * DS == 9 * 64, "the load split below");
    constexpr int OOB = 1 << 30;                            // byte offset past any tile
    auto slot_os = [&](int j, int slots, int chunks, int row_bytes) {
        const int sl = 64 * j + lane, c = sl % slots;
        return c < chunks ? sl / slots * row_bytes + c * 16 : OOB;
    };
    int os_q[4], os_d[3];
#pragma unroll
    for(int u = 0; u < 3; u++)
        os_q[u] = slot_os(w + 4 * u, QS, D_QK / 8, k.stride_q_n * 2);
    os_q[3] = slot_os(12, QS, D_QK / 8, k.stride_q_n * 2);
#pragma unroll
    for(int u = 0; u < 2; u++)
        os_d[u] = slot_os(w + 4 * u, DS, D_V / 8, k.stride_do_n * 2);
    os_d[2] = slot_os(8, DS, D_V / 8, k.stride_do_n * 2);
    const bf16_t* q_ptr  = reinterpret_cast<const bf16_t*>(k.ptr_q) + static_cast<long long>(q0) * k.stride_q_n +
                          static_cast<long long>(h) * k.stride_q_h;
    const bf16_t* do_ptr = reinterpret_cast<const bf16_t*>(k.ptr_do) + static_cast<long long>(q0) * k.stride_do_n +
                           static_cast<long long>(h) * k.stride_do_h;
    auto dma = [&](int i, int buf) {
        const int qt0 = i * BQ;
        const auto rq = gated_mla_bwd::dma_rsrc(q_ptr + static_cast<long long>(qt0) * k.stride_q_n,
                                                rec_rows(seqlen_q - qt0, BQ, k.stride_q_n * 2));
        const auto rd = gated_mla_bwd::dma_rsrc(do_ptr + static_cast<long long>(qt0) * k.stride_do_n,
                                                rec_rows(seqlen_q - qt0, BQ, k.stride_do_n * 2));
        const unsigned int lq = gated_mla_bwd::lds_addr(smem + OFF_Q + buf * BQ * LD_Q);
        const unsigned int ld = gated_mla_bwd::lds_addr(smem + OFF_DO + buf * BQ * LD_DO);
#pragma unroll
        for(int u = 0; u < 3; u++)
            gated_mla_bwd::dma16(rq, os_q[u], lq + (w + 4 * u) * 1024);
#pragma unroll
        for(int u = 0; u < 2; u++)
            gated_mla_bwd::dma16(rd, os_d[u], ld + (w + 4 * u) * 1024);
        if(w == 0)
            gated_mla_bwd::dma16(rq, os_q[3], lq + 12 * 1024);
        if(w == 1)
            gated_mla_bwd::dma16(rd, os_d[2], ld + 8 * 1024);
    };
    const bool f_ld     = w < 2 && lane < BQ;
    const int ld_slot   = (w == 0 ? 0 : BQ) + lane;
    const float* ld_ptr = (w == 0 ? k.ptr_lse : k.ptr_d) + static_cast<long long>(h) * k.stride_lse_h + q0;
    float r_ld = 0.0f;
    auto fetch_ld = [&](int i) {
        if(f_ld)
            r_ld = opus::make_gmem(ld_ptr + i * BQ, rec_rows(seqlen_q - i * BQ, BQ, 4)).template load<1>(lane)[0];
    };
    auto stage_ld = [&](int buf) {
        if(f_ld)
            s_ld[buf][ld_slot] = r_ld;
    };
    // The DMA writes land in LDS when vmcnt says so, which s_barrier does not wait for.
    auto dma_wait = [] { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); };

    auto s_q  = [&](int b) { return opus::make_smem(smem + OFF_Q + b * BQ * LD_Q); };
    auto s_do = [&](int b) { return opus::make_smem(smem + OFF_DO + b * BQ * LD_DO); };
    const auto u_q  = opus::partition_layout_b<8>(mma_s, opus::make_tuple(number<LD_Q>{}, 1_I), c_ab);
    const auto u_do = opus::partition_layout_b<8>(mma_dp, opus::make_tuple(number<LD_DO>{}, 1_I), c_ab);
    // C element e = (m NT + tn) 4 + p of S^T / dP^T is key 16 m + lane % 16 of the wave and
    // query 16 tn + 4 (lane / 16) + p of the tile.
    constexpr int NS = decltype(mma_s)::tile_c_len;
    static_assert(NS == MR * NT * 4);
    const int c_kv = row_w + lane % 16, c_q = 4 * (lane / 16);
    const float scale_log2 = k.softmax_scale * LOG2_E;

    // dK and dV accumulators, [M tile][N tile] in the swap_ab C layout, in AGPRs.
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

    // K and V must be in registers before the loop: otherwise the compiler waits for them inside
    // it, with vmcnt counts that the DMA loads then make block. 0x0f70 is s_waitcnt vmcnt(0).
    __builtin_amdgcn_s_waitcnt(0x0f70);
    if(i_begin < n_q_tiles)
    {
        dma(i_begin, 0);
        fetch_ld(i_begin);
        stage_ld(0);
        if(i_begin + 1 < n_q_tiles)
            fetch_ld(i_begin + 1);
    }
    dma_wait();
    __syncthreads();

    // One tile per iteration: issue the next tile's DMA, compute, wait for the DMA, barrier. The
    // tiles this wave sees in full (no mask: every query valid, at or past its last key) form one
    // contiguous range [i_full0, i_full1) and run the interleaved code; the masked ones before and
    // after it run the plain code. Every wave passes one barrier per tile either way.
    auto iter = [&](int i, auto full_c) {
        const int b   = (i - i_begin) & 1;
        const int qt0 = i * BQ;
        // The other buffer held tile i - 1, read before the last barrier. LSE / D of tile i + 1
        // are stored and those of tile i + 2 loaded before the DMA is issued: the compiler waits
        // for its own loads with vmcnt(0), which would otherwise also wait for the DMA.
        if(i + 1 < n_q_tiles)
            stage_ld(b ^ 1);
        if(i + 2 < n_q_tiles)
            fetch_ld(i + 2);
        if(i + 1 < n_q_tiles)
            dma(i + 1, b ^ 1);
        // With one wave per SIMD, VALU work overlaps MFMAs only where it sits between them in
        // issue order (the wave stalls on an MFMA while the pipe is busy). The tile runs as four
        // blocks fenced by sched_barrier(0): S^T; dP^T one MFMA at a time with the exp of one P
        // element after each (S^T is complete); dV likewise with one dS element after each (dP^T
        // is complete); dK. Inside a block, sched_barrier(SB_MEM_SALU) keeps that MFMA / VALU
        // order and lets LDS, VMEM and SALU work move; the fences keep each block's LDS reads
        // from all being hoisted to the top of the tile, which would exceed the VGPRs. Only tiles
        // that need no mask run this way.
        constexpr int SB_MEM_SALU = 0x4 | 0x10 | 0x80;  // SALU, all VMEM, all DS may cross
        constexpr int KSV         = D_V / 32;
        static_assert(KSV * MR * NT == NS && MR * NV == NS, "one P / dS element per MFMA");
        auto tile_full = [&] {
            f32x4_t lse4[NT];
#pragma unroll
            for(int tn = 0; tn < NT; tn++)
            {
                lse4[tn] = *reinterpret_cast<const f32x4_t*>(&s_ld[b][16 * tn + c_q]);
#pragma unroll
                for(int p = 0; p < 4; p++)
                    lse4[tn][p] *= LOG2_E;
            }
            auto v_sp       = mma_s(v_kw, s_q(b).template load<8>(u_q));
            const auto b_dp = s_do(b).template load<8>(u_do);
            __builtin_amdgcn_sched_barrier(0);

            opus::vector_t<float, NS> v_dpt;
#pragma unroll
            for(int ks = 0; ks < KSV; ks++)
#pragma unroll
                for(int m = 0; m < MR; m++)
#pragma unroll
                    for(int tn = 0; tn < NT; tn++)
                    {
                        const int t = m * NT + tn;
                        f32x4_t c;
#pragma unroll
                        for(int p = 0; p < 4; p++)
                            c[p] = ks == 0 ? 0.0f : v_dpt[t * 4 + p];
                        bvec<8> fa, fb;
#pragma unroll
                        for(int x = 0; x < 8; x++)
                        {
                            fa[x] = v_vw[(m * KSV + ks) * 8 + x];
                            fb[x] = b_dp[(tn * KSV + ks) * 8 + x];
                        }
                        c = mma1(fa, fb, c);
#pragma unroll
                        for(int p = 0; p < 4; p++)
                            v_dpt[t * 4 + p] = c[p];
                        const int e   = (ks * MR + m) * NT + tn;
                        const float x = v_sp[e] * scale_log2 - lse4[e / 4 % NT][e % 4];
                        v_sp[e]       = __builtin_amdgcn_exp2f(x);
                        __builtin_amdgcn_sched_barrier(SB_MEM_SALU);
                    }
            // C -> A: an M tile's 8 elements (both N tiles) are in the K order of tr_load_b_c_order.
            const auto a_p = opus::cast<bf16_t>(v_sp);
            __builtin_amdgcn_sched_barrier(0);

            f32x4_t d4[NT];
#pragma unroll
            for(int tn = 0; tn < NT; tn++)
                d4[tn] = *reinterpret_cast<const f32x4_t*>(&s_ld[b][BQ + 16 * tn + c_q]);
            const auto b_dv = tr_load_b_c_order<NV, LD_DO>(s_do(b), lane, 0);
#pragma unroll
            for(int m = 0; m < MR; m++)
#pragma unroll
                for(int n = 0; n < NV; n++)
                {
                    gated_mla_bwd::mfma_acc_16x16x32(acc_dv[m][n], gated_mla_bwd::frag8(a_p, m), gated_mla_bwd::frag8(b_dv, n));
                    const int e = m * NV + n;
                    v_sp[e]     = v_sp[e] * (v_dpt[e] - d4[e / 4 % NT][e % 4]) * k.softmax_scale;
                    __builtin_amdgcn_sched_barrier(SB_MEM_SALU);
                }
            const auto a_ds = opus::cast<bf16_t>(v_sp);
            __builtin_amdgcn_sched_barrier(0);

            acc_gemm32(acc_dkn, a_ds, tr_load_b_c_order<NV, LD_Q>(s_q(b), lane, 0));
            acc_gemm32(acc_dkr, a_ds, tr_load_b_c_order<NR, LD_Q>(s_q(b), lane, D_NOPE));
        };
        // Tiles that need the mask (few: the diagonal and sequence ends) take the plain order.
        auto tile_masked = [&] {
            f32x4_t lse4[NT], d4[NT];
#pragma unroll
            for(int tn = 0; tn < NT; tn++)
            {
                lse4[tn] = *reinterpret_cast<const f32x4_t*>(&s_ld[b][16 * tn + c_q]);
                d4[tn]   = *reinterpret_cast<const f32x4_t*>(&s_ld[b][BQ + 16 * tn + c_q]);
#pragma unroll
                for(int p = 0; p < 4; p++)
                    lse4[tn][p] *= LOG2_E;
            }
            const auto v_st = mma_s(v_kw, s_q(b).template load<8>(u_q));
            opus::vector_t<float, NS> v_p;
#pragma unroll
            for(int e = 0; e < NS; e++)
            {
                const int kv = c_kv + 16 * (e / (4 * NT)), qp = qt0 + c_q + 16 * (e / 4 % NT) + e % 4;
                const bool ok = (kv < seqlen_k) & (qp < seqlen_q) & (!k.causal | (qp + causal_offset >= kv));
                const float x = v_st[e] * scale_log2 - lse4[e / 4 % NT][e % 4];
                v_p[e]        = __builtin_amdgcn_exp2f(ok ? x : -1e30f);
            }
            const auto v_dpt = mma_dp(v_vw, s_do(b).template load<8>(u_do));
            opus::vector_t<float, NS> v_ds;
#pragma unroll
            for(int e = 0; e < NS; e++)
                v_ds[e] = v_p[e] * (v_dpt[e] - d4[e / 4 % NT][e % 4]) * k.softmax_scale;
            const auto a_p  = opus::cast<bf16_t>(v_p);
            const auto a_ds = opus::cast<bf16_t>(v_ds);
            acc_gemm32(acc_dv, a_p, tr_load_b_c_order<NV, LD_DO>(s_do(b), lane, 0));
            acc_gemm32(acc_dkn, a_ds, tr_load_b_c_order<NV, LD_Q>(s_q(b), lane, 0));
            acc_gemm32(acc_dkr, a_ds, tr_load_b_c_order<NR, LD_Q>(s_q(b), lane, D_NOPE));
        };

        if constexpr(decltype(full_c)::value)
        tile_full();
        else
        tile_masked();
        dma_wait();
        __syncthreads();
    };
    int i_full0 = n_q_tiles, i_full1 = n_q_tiles;
    if(row_w + WR <= seqlen_k)
    {
        // causal: qt0 + causal_offset >= row_w + WR - 1; all: qt0 + BQ <= seqlen_q
        const int need = row_w + WR - 1 - causal_offset;
        i_full0 = k.causal && need > 0 ? (need + BQ - 1) / BQ : 0;
        i_full1 = seqlen_q / BQ;
        i_full0 = i_full0 < i_begin ? i_begin : (i_full0 > n_q_tiles ? n_q_tiles : i_full0);
        i_full1 = i_full1 < i_full0 ? i_full0 : (i_full1 > n_q_tiles ? n_q_tiles : i_full1);
    }
    int i = i_begin;
    for(; i < i_full0; i++)
        iter(i, std::false_type{});
    for(; i < i_full1; i++)
        iter(i, std::true_type{});
    for(; i < n_q_tiles; i++)
        iter(i, std::false_type{});

    // dK nope and dV into the kv_b_proj layout, dK rope per head in fp32. The C fragments hold
    // 4 columns of 16 rows per instruction, so each wave transposes its 48 rows through LDS first
    // and stores whole rows; rows past seqlen_k fall outside the descriptors and are dropped.
    mfma_acc_drain();
    __syncthreads();
    {
        constexpr int LD_R = D_ROT + 4;  // [48][dK rope] fp32, over the same LDS
        static_assert(WR * LD_R * 4 <= WR * LD_O * 2, "staging fits");
        bf16_t* s_o = smem + w * WR * LD_O;
        float* s_r  = reinterpret_cast<float*>(s_o);
        // C element (m, n, e): row 16 m + lane % 16, column 16 n + 4 (lane / 16) + e.
        const int c_row = lane % 16, c_col = 4 * (lane / 16);
#pragma unroll
        for(int m = 0; m < MR; m++)
#pragma unroll
            for(int n = 0; n < NV; n++)
            {
                bf16_t* o = s_o + (16 * m + c_row) * LD_O + 16 * n + c_col;
                *reinterpret_cast<bvec<4>*>(o)          = opus::cast<bf16_t>(acc_dkn[m][n]);
                *reinterpret_cast<bvec<4>*>(o + D_NOPE) = opus::cast<bf16_t>(acc_dv[m][n]);
            }
        bf16_t* dkv = reinterpret_cast<bf16_t*>(k.ptr_dkv) + static_cast<long long>(k0 + row_w) * k.stride_dkv_n +
                      static_cast<long long>(h) * k.stride_dkv_h;
        auto g_dkv = opus::make_gmem(dkv, rec(static_cast<long long>(rows_w) * k.stride_dkv_n, 2));
        // 32 lanes x 8 bf16 per row, two rows per instruction.
        constexpr int OC = (D_NOPE + D_V) / 8;
#pragma unroll
        for(int it = 0; it < WR * OC / 64; it++)
        {
            const int row = (it * 64 + lane) / OC, col = (lane % OC) * 8;
            g_dkv.template store<8>(*reinterpret_cast<const vec8_t*>(s_o + row * LD_O + col), row * k.stride_dkv_n + col);
        }
#pragma unroll
        for(int m = 0; m < MR; m++)
#pragma unroll
            for(int n = 0; n < NR; n++)
                *reinterpret_cast<f32x4_t*>(s_r + (16 * m + c_row) * LD_R + 16 * n + c_col) = acc_dkr[m][n];
        auto g_dkr = opus::make_gmem(k.ptr_dk_rope + static_cast<long long>(k0 + row_w) * k.stride_dkr_n +
                                         static_cast<long long>(h) * k.stride_dkr_h,
                                     rec(static_cast<long long>(rows_w) * k.stride_dkr_n, 4));
        // 16 lanes x 4 fp32 per row, four rows per instruction.
        constexpr int RC = D_ROT / 4;
#pragma unroll
        for(int it = 0; it < WR * RC / 64; it++)
        {
            const int row = (it * 64 + lane) / RC, col = (lane % RC) * 4;
            g_dkr.template store<4>(*reinterpret_cast<const f32x4_t*>(s_r + row * LD_R + col), row * k.stride_dkr_n + col);
        }
    }
}

#else
__global__ void gated_mla_bwd_pre_kernel(gated_mla_bwd_pre_kargs) {}
template <bool DQ>
__global__ void gated_mla_bwd_kernel(gated_mla_bwd_kargs) {}
__global__ void gated_mla_bwd_dkdv_pair_kernel(gated_mla_bwd_kargs) {}
__global__ void gated_mla_bwd_dkdv_4w_kernel(gated_mla_bwd_kargs) {}
__global__ void gated_mla_bwd_dq_kernel(gated_mla_bwd_dq_kargs) {}
#endif
