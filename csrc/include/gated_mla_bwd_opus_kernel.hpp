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
// wg_map with its results in SGPRs: its divisions run in VALU, and what derives from them (tile
// and buffer indices, LDS addresses) must be scalar for the DMA's descriptors and m0.
__device__ inline wg_ids wg_map_sgpr(int wg, int H, int num_seqs, int n_blocks, int hg)
{
    const wg_ids r = wg_map(wg, H, num_seqs, n_blocks, hg);
    return {__builtin_amdgcn_readfirstlane(r.h), __builtin_amdgcn_readfirstlane(r.g), __builtin_amdgcn_readfirstlane(r.blk)};
}
// dQ pass: load phases and compute phases, the VALU work behind the wave's own MFMAs (1), or the
// MFMA stages C1, C2 and the VALU stages N1, N2 (0).
#ifndef GMLA_X_PFIRST
#define GMLA_X_PFIRST 0
#endif
#ifndef GMLA_DQ_PP
#define GMLA_DQ_PP 0
#endif
// The 4-wave backward's dS = P (dP - D): dP's accumulators start at -D, so that dP's MFMAs leave
// dP - D and a tile's 24 subtractions go (1, the default), or they start at 0 and D is subtracted,
// as the pair pass and the ASM kernels do it (0: "separate" then equals "ds" bit for bit). The two
// differ in the order of dP's fp32 sums, an ulp of a bf16 result here and there: against gradients
// computed in fp32 the error is the same to four digits (2.4e-3 relative L2 on dQ, dK and dV, as
// the ASM backward's with fp32 dQ atomics).
#ifndef GMLA_DP_MINUS_D
#define GMLA_DP_MINUS_D 1
#endif
// 4-wave dK / dV pass: every read, VALU instruction and DMA round behind an MFMA at a unit of a
// schedule (1, see gated_mla_bwd_4ws), or a tile as four blocks that each load their operands
// first (0).
#ifndef GMLA_4W_PP
#define GMLA_4W_PP 1
#endif
// Pair pass: load phases and compute phases as the dQ pass's (1), or the stages A and B (0).
#ifndef GMLA_PAIR_PP
#define GMLA_PAIR_PP 0
#endif
// Timing experiments on the phased pair pass: a switch off its default leaves one kind of work out
// of the loop (the results are then wrong), or moves it, to see what it costs in the phases' lengths.
#ifndef GMLA_X_PW
#define GMLA_X_PW 1  // the S wave writes P to LDS
#endif
#ifndef GMLA_X_PWPOS
#define GMLA_X_PWPOS 0  // P's float4s 0 .. 3 in the S wave's load phases 1, 2, 3, 4 (0); 1, 2, 3, 3 (1); 1, 2, 2, 3 (2); 1, 2, 3, 5 (3)
#endif
#ifndef GMLA_X_PWMID
#define GMLA_X_PWMID 0  // those writes right behind the phase's reads (1) or at its end (0)
#endif
#ifndef GMLA_X_PR
#define GMLA_X_PR 1  // the dP wave reads P from LDS (0: from a register)
#endif
#ifndef GMLA_X_PRLAST
#define GMLA_X_PRLAST 0  // that read after its phase's operand reads (1) or before them (0)
#endif
#ifndef GMLA_X_DMA
#define GMLA_X_DMA 1  // the DMA rounds of the loop and their descriptors
#endif
#ifndef GMLA_X_RPH
#define GMLA_X_RPH 3, 3, 3, 4, 1, 2, 4  // the dP wave's load phase (0 .. 4: a .. e) of each DMA round: 0 .. 2 Q, 3 Q or dO, 4, 5 dO, 6 LSE or D
#endif
#ifndef GMLA_X_D
#define GMLA_X_D 1  // the dP wave reads D from LDS (0: its registers stay 0 and the subtractions go; 2: they stay, against registers the compiler knows nothing of)
#endif
#ifndef GMLA_X_RMASK
#define GMLA_X_RMASK 127  // the DMA rounds that are issued (bit u: round u)
#endif
#ifndef GMLA_X_DSV
#define GMLA_X_DSV 0  // the dP wave's VALU work: (dP - D) behind 8 MFMAs and P (dP - D) behind 16 (0), one instruction behind each MFMA (1), or as 1 with (dP - D) behind the second 8 MFMAs alone (2)
#endif
#ifndef GMLA_X_LSE
#define GMLA_X_LSE 1  // the S wave reads LSE from LDS (0: from a register)
#endif
#ifndef GMLA_X_LOOPPIN
#define GMLA_X_LOOPPIN 1  // the loop-carried scalars pinned ahead of the loop test
#endif
#if GMLA_DQ_PP && GMLA_DQ_QR_REG
#error "GMLA_DQ_PP reads Q rope from sQR"
#endif
// A compute phase's order: each of its 8 MFMAs, then CNT instructions of class MASK
// (__builtin_amdgcn_sched_group_barrier's: 0x002 VALU, 0x400 transcendental).
#define GMLA_SCHED8(MASK, CNT)                                 \
    _Pragma("unroll") for(int i_ = 0; i_ < 8; i_++)            \
    {                                                          \
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);     \
        __builtin_amdgcn_sched_group_barrier(MASK, CNT, 0);    \
    }
// As GMLA_SCHED8 for the 4 MFMAs of one step.
#define GMLA_SCHED4(MASK, CNT)                                 \
    _Pragma("unroll") for(int i_ = 0; i_ < 4; i_++)            \
    {                                                          \
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);     \
        __builtin_amdgcn_sched_group_barrier(MASK, CNT, 0);    \
    }
// dQ pass: an N stage's LDS reads (the next C stage's first steps) after its VALU work (1) or before (0).
#ifndef GMLA_DQ_LPF
#define GMLA_DQ_LPF 0
#endif
// dQ pass: a C stage's LDS reads one after each MFMA (1) or after each step's MFMAs (0).
#ifndef GMLA_DQ_IL
#define GMLA_DQ_IL 0
#endif
// dQ pass: this wave's Q rope A operands in registers (1) or re-read from LDS every tile (0).
#ifndef GMLA_DQ_QR_REG
#define GMLA_DQ_QR_REG 0
#endif
// Cache policy of the dS stores and loads (written once, read once): non-temporal.
#ifndef GMLA_DS_NT
#define GMLA_DS_NT 1
#endif
#if GMLA_DS_NT
#define GMLA_DS_CACHE " nt"
#else
#define GMLA_DS_CACHE ""
#endif
__device__ inline void dma16(const i32x4_t& rsrc, int voff, unsigned int lds)
{
    asm volatile("s_mov_b32 m0, %2\n\tbuffer_load_dwordx4 %0, %1, 0 offen lds" : : "v"(voff), "s"(rsrc), "s"(lds) : "m0");
}
// As dma16 with 4 bytes per lane, landing at LDS byte lds + 4 l.
__device__ inline void dma4(const i32x4_t& rsrc, int voff, unsigned int lds)
{
    asm volatile("s_mov_b32 m0, %2\n\tbuffer_load_dword %0, %1, 0 offen lds" : : "v"(voff), "s"(rsrc), "s"(lds) : "m0");
}

} // namespace gated_mla_bwd

// One row of D_V = 128 bf16 per 16 lanes (8 per lane, one 16-byte access each). With ptr_gate
// null it makes D alone (gated_mla_bwd_dot_bf16_opus).
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
        if(k.ptr_gate == nullptr)
        {
            // No gate: dO is dout itself, and only D is made.
            for(int i = 0; i < VEC; i++)
            {
                const float d_o = dout[i] * out[i];
                acc += d_o;
            }
        }
        else
        {
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
    // Workgroup id -> (head h, sequence g, KV block j): gated_mla_bwd::wg_map.
    const int wg   = static_cast<int>(opus::block_id_x());
    const auto ids = gated_mla_bwd::wg_map_sgpr(wg, k.H, k.num_seqs, k.n_kv_blocks, k.hg);
    const int h = ids.h, g = ids.g, j = ids.blk;
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
//   N1: P = exp(S(t) - LSE), in fp32 in S's registers until dS
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
    // Workgroup id -> (head, sequence, query block): gated_mla_bwd::wg_map, the query blocks
    // last-first (the causal long ones first).
    const int wg   = static_cast<int>(opus::block_id_x());
    const auto ids = gated_mla_bwd::wg_map_sgpr(wg, k.H, k.num_seqs, k.n_q_blocks, k.hg);
    const int h = ids.h, g = ids.g, qb = k.n_q_blocks - 1 - ids.blk;
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
#if GMLA_DQ_QR_REG
    auto mma_qr = make_mma(number<MR>{}, number<KC / 16>{}, number<D_ROT / 32>{});  // for Q rope's A layout
    const auto v_qrw =
        g_q.template load<8>(opus::partition_layout_a<8>(mma_qr, opus::make_tuple(k.stride_q_n, 1_I), c_ab), D_NOPE);
#else
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
#endif
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
    // Per load round u, wave-uniform: buffer descriptor of the next tile to load (its base and
    // size), row bytes, place in the buffer and m0 for the next tile; and this lane's source offset
    // within the tile. A wave without a load in its last round repeats its previous one, the same
    // bytes to the same place, so that no round takes a branch.
    gated_mla_bwd::i32x4_t d_rs[JU];
    int d_rb[JU], d_lds[JU], d_os[JU];
    unsigned int d_m0[JU];
#pragma unroll
    for(int u = 0; u < JU; u++)
    {
        const int j   = w + NW * u < J_N + J_R + J_V ? w + NW * u : w + NW * (u - 1);
        const int reg = j < J_N ? 0 : (j < J_N + J_R ? 1 : 2);
        const int jj  = reg == 0 ? j : (reg == 1 ? j - J_N : j - J_N - J_R);
        const int rs  = reg == 0 ? k.stride_k_n : (reg == 1 ? k.stride_k_rot_n : k.stride_v_n);
        d_rb[u]       = rs * 2;
        d_rs[u]       = gated_mla_bwd::dma_rsrc(reg == 0 ? kn_ptr : (reg == 1 ? kr_ptr : v_ptr), 0);
        d_lds[u]      = (reg == 0 ? OFF_KN : (reg == 1 ? OFF_KR : OFF_V)) * 2 + jj * 1024;
        const int spr = reg == 0 ? SPR_N : (reg == 1 ? SPR_R : SPR_V);
        const int dch = reg == 0 ? D_NOPE / 8 : (reg == 1 ? D_ROT / 8 : D_V / 8);
        const int sl = 64 * jj + lane, c = sl % spr;
        d_os[u]      = c < dch ? sl / spr * d_rb[u] + c * 16 : OOB;
    }
    // Key rows of tile t, in SALU: the compiler would clamp with v_med3 and move the result back
    // with a readfirstlane, which waits behind the other wave's VALU work.
    auto dma_rows = [&](int t) {
        int r = seqlen_k - t * KC;
        asm("s_min_i32 %0, %0, %1\n\ts_max_i32 %0, %0, 0" : "+s"(r) : "n"(KC) : "scc");
        return r;
    };
    auto dma_lds = [&](int t) { return gated_mla_bwd::lds_addr(smem + static_cast<unsigned int>(t) % NBUF * SZ_BUF); };
    // Point the descriptors and m0 of every round at tile t (`adv`: from tile t - 1, whose base
    // they hold). Tiles go in order, so the base just advances; tiles past the last one get size
    // 0 and fill a buffer no one reads anymore with zeros, as do keys past seqlen_k. Made in an N
    // stage and pinned, so that a C stage's round is only m0 and the load.
    auto dma_next = [&](int t, bool adv) {
        const int rows        = dma_rows(t);
        const unsigned int lb = dma_lds(t);
#pragma unroll
        for(int u = 0; u < JU; u++)
        {
            auto& r = d_rs[u];
            unsigned long long p =
                static_cast<unsigned int>(r[0]) | static_cast<unsigned long long>(static_cast<unsigned int>(r[1])) << 32;
            if(adv)
                p += static_cast<unsigned long long>(static_cast<unsigned int>(KC * d_rb[u]));
            int n;
            asm("s_mul_i32 %0, %1, %2" : "=s"(n) : "s"(rows), "s"(d_rb[u]));
            r       = gated_mla_bwd::i32x4_t{static_cast<int>(p), static_cast<int>(p >> 32), n, 0x00020000};
            d_m0[u] = lb + d_lds[u];
            asm volatile("" : "+s"(r), "+s"(d_m0[u]));
        }
    };
    auto dma_round = [&](int u) { gated_mla_bwd::dma16(d_rs[u], d_os[u], d_m0[u]); };
    auto dma       = [&] {
#pragma unroll
        for(int u = 0; u < JU; u++)
            dma_round(u);
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
        unsigned int a = gated_mla_bwd::lds_addr(smem + static_cast<unsigned int>(t) % NBUF * SZ_BUF + off) + o * 2;
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
#if GMLA_DQ_QR_REG
    auto a_qrope = [&](int kr) {
        afrag_t a;
#pragma unroll
        for(int m = 0; m < MR; m++)
#pragma unroll
            for(int x = 0; x < 8; x++)
                a[m * 8 + x] = v_qrw[(m * KS_R + kr) * 8 + x];
        return a;
    };
#else
    auto s_qr       = opus::make_smem(smem + OFF_QR + w * WQ * LD_QR);
    const auto u_qr = opus::partition_layout_a<8>(mma_k1, opus::make_tuple(number<LD_QR>{}, 1_I), c_ab);
    // The rope K steps run last; their A fragments are read LA steps ahead, like the B ones.
    afrag_t a_qr[KS_R];
    static_assert(KS_R <= LA_ && KS_N >= LA_, "rope A fragments read within S's nope steps");
#endif
    auto a_do = [&](int ks) {
        afrag_t a;
#pragma unroll
        for(int m = 0; m < MR; m++)
#pragma unroll
            for(int x = 0; x < 8; x++)
                a[m * 8 + x] = v_dow[(m * KS_P + ks) * 8 + x];
        return a;
    };

    acc_t v_s, v_dp, v_dq[NQC];
#pragma unroll
    for(int c = 0; c < NQC; c++)
        opus::clear(v_dq[c]);
    // dS of the previous tile, the A operand of its dQ. Zero before tile 0, whose dQ(t - 1) steps
    // then add nothing (they read tile 0's K, which is finite), so that tile 0 runs the loop body.
    afrag_t a_ds;
    opus::clear(a_ds);
#if GMLA_DQ_IL
    // B fragments of LA steps in flight, in a ring of LA + 1 slots: a step's reads (of step + LA)
    // go to the slot step - 1 consumed, so that they issue between the step's own MFMAs, one after
    // each: an MFMA holds the SIMD's vector issue for the first 8 of its 16 cycles only, and the
    // wave's own LDS reads go into the other 8.
    constexpr int LA = LA_, RS = LA + 1;
    frag_t ring[RS];
    auto c_stage = [&](auto n_steps, auto ld, auto ex) {
        constexpr int N = decltype(n_steps)::value;
#pragma unroll
        for(int st = 0; st < N; st++)
        {
            ex(st, ring[st % RS]);
            if(st + LA < N)
                ring[(st + LA) % RS] = ld(st + LA);
#pragma unroll
            for(int m = 0; m < MR * KC / 16; m++)
            {
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);  // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);  // DS read
                __builtin_amdgcn_sched_group_barrier(0x004, 2, 0);  // SALU
            }
            __builtin_amdgcn_sched_barrier(0);
        }
    };
#else
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
#endif
    auto prefetch = [&](auto ld) {
#pragma unroll
        for(int st = 0; st < LA; st++)
            ring[st] = ld(st);
    };
    const acc_t zero{};
    auto ex_s = [&](int ks, const frag_t& b) {
#if GMLA_DQ_QR_REG
        v_s = mma_k1(ks < KS_N ? a_q(ks) : a_qrope((ks - KS_N) % KS_R), b, ks == 0 ? zero : v_s);
#else
        v_s = mma_k1(ks < KS_N ? a_q(ks) : a_qr[(ks - KS_N) % KS_R], b, ks == 0 ? zero : v_s);
        if(ks + LA_ >= KS_N && ks + LA_ < KS_S)
            a_qr[ks + LA_ - KS_N] = s_qr.template load<8>(u_qr, (ks + LA_ - KS_N) * 32);
#endif
    };
    auto ex_dp = [&](int ks, const frag_t& b) { v_dp = mma_k1(a_do(ks), b, ks == 0 ? zero : v_dp); };
    auto ex_dq = [&](int c, const frag_t& b) { v_dq[c] = mma_k1(a_ds, b, v_dq[c]); };

    // C element e = (tm 2 + tn) 4 + p of S and dP is query 16 tm + lane % 16, key 16 tn +
    // 4 (lane / 16) + p of the wave's tile.
    constexpr int NS = decltype(mma_k1)::tile_c_len;
    static_assert(NS == MR * 2 * 4);
    const float scale_log2 = k.softmax_scale * LOG2_E;
    const int T            = n_kv_tiles;

    // N1: the first steps of C2 (and its addresses); P = exp(S(t) - LSE), in place in v_s.
    auto n1 = [&](int t) {
        int tp = t;  // max(t - 1, 0) in SALU, not a clamped VALU subtract
        asm volatile("s_sub_i32 %0, %0, 1\n\ts_max_i32 %0, %0, 0" : "+s"(tp) : : "scc");
        a_v  = addr(t, OFF_V, o_v);
        a_tr = addr(tp, OFF_KR, o_tr);
#if !GMLA_DQ_LPF
        prefetch(ld_dp);
#endif
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
        pin(v_s);
#if GMLA_DQ_LPF
        // The stage's LDS reads go last: the other wave of the SIMD, in its C stage, reads LDS after
        // each MFMA but those of its last LA steps, and its reads would hold these back.
        __builtin_amdgcn_sched_barrier(0);
        prefetch(ld_dp);
#endif
#pragma unroll
        for(int r = 0; r < LA; r++)
            pin(ring[r]);
    };
    // N2: the first steps of the next C stage (C1 of tile t + 1, or the tail's dQ of tile t);
    // dS(t) = P (dP(t) - D) in bf16.
    auto n2 = [&](int t) {
        a_tn = addr(t, OFF_KN, o_tn);
        auto pf = [&] {
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
        };
#if !GMLA_DQ_LPF
        pf();
#endif
        opus::vector_t<float, NS> v_ds;
#pragma unroll
        for(int e = 0; e < NS; e++)
            v_ds[e] = v_s[e] * (v_dp[e] - dd[e / (NS / MR)]);
        a_ds = opus::cast<bf16_t>(v_ds);
        pin(a_ds);
#if GMLA_DQ_LPF
        __builtin_amdgcn_sched_barrier(0);  // as in N1: the LDS reads last
        pf();
#endif
#pragma unroll
        for(int r = 0; r < LA; r++)
            pin(ring[r]);
        dma_next(t + 3, true);
    };

    // Tiles 0 and 1 in LDS, and Q, dO, LSE, D in registers, before the loop (0x0f70 is
    // s_waitcnt vmcnt(0)): otherwise the compiler waits for the latter inside the loop with
    // vmcnt counts that the DMA then makes block.
    dma_next(0, false);
    if(T > 0)
        dma();
    dma_next(1, true);
    if(T > 1)
        dma();
    dma_next(2, true);
    __builtin_amdgcn_s_waitcnt(0x0f70);
    dma_wait();
    stage_end();
    const bool lag = w >= NW / 2;
    if(lag)
        stage_end();
#if GMLA_DQ_PP
    // Load and compute phases, as the forward has them (fmha_fwd_hd192_v128_bf16_opus_kernel.hpp).
    // A tile is 8 pairs of a load phase and a compute phase, each closed by a workgroup barrier:
    //   load     the B fragments of the next 8 MFMAs (two steps) into registers, and the scalar
    //            work, the DMA rounds and their descriptors
    //   compute  those 8 MFMAs, with this wave's own VALU work in the 8 of an MFMA's 16 cycles
    //            that take other vector instructions
    // Waves 4-7 run one phase behind waves 0-3 on the same SIMDs: a wave reads LDS while the other
    // one of its SIMD computes, so that their LDS reads never meet and each read has a phase to
    // come back. Per tile t:
    //   1 .. 3  S(t), K steps 0 .. 5        behind 1: dS(t - 1) to bf16 (the A operand of dQ(t - 1))
    //   4, 5    dP(t), K steps 0 .. 3       behind 4: S scale - LSE; behind 5: exp, first half
    //   6 .. 8  dQ(t - 1), chunks 0 .. 5    behind 6: exp, second half (P in S's registers);
    //                                       behind 7: dP - D; behind 8: P (dP - D), fp32, in dP's
    //
    // Where a load phase's scalar work sits decides whether the compute phases beside it run their
    // 8 MFMAs in 150 cycles or in 230 (ATT, 1 x 16384 keys: the SIMD's other wave stops issuing
    // VALU and MFMA for 60 to 80 cycles). What the traces show, without the mechanism:
    //   - a branch on VCC or an s_mul_i32 in the reading wave does it: the test for a tile this
    //     wave does not see in full is a scalar compare whose branch is not taken, and the DMA
    //     descriptors go from tile to tile by adds and selects;
    //   - so does scalar work behind the 8 transposed reads of phases 6 .. 8, which fill their
    //     phase, or in phase 4: it goes behind the 4 reads of phases 1, 2 and 5;
    //   - s_setprio (the computing wave above the reading one) makes the tile's first phases slower.
    // The tile takes 2442 cycles this way, 2805 with all three wrong (2048 are MFMAs).
    if(T > 0)
    {
        frag_t op[2];
        // A load phase ends without waiting for its reads (an empty asm that clobbers memory keeps
        // them on its side of the barrier; the compiler waits for each in front of its MFMA).
        auto l_end = [&] {
            asm volatile("" ::: "memory");
            stage_end();
        };
        auto c_end = [&] { stage_end(); };
        auto xs = [&](int ks, const frag_t& b) {
            v_s = mma_k1(ks < KS_N ? a_q(ks) : a_qr[ks - KS_N], b, ks == 0 ? zero : v_s);
        };
        static_assert(JU <= 3 && KS_S == 6 && KS_P == 4 && NQC == 6 && KS_R == 2, "phases as laid out below");
        // One MFMA of S's nope K step ks: M tile tm against key half tn (mma_k1 runs the four in one
        // call, which leaves their order, and what goes between them, to the compiler).
        auto mma_11 = make_mma(number<1>{}, number<1>{}, number<1>{});
        static_assert(decltype(mma_11)::tile_a_len == 8 && decltype(mma_11)::tile_c_len == 4);
        auto xs1 = [&](int ks, int tm, int tn, const frag_t& b) {
            opus::vector_t<bf16_t, 8> a1;
            bvec<8> b1;
            typename decltype(mma_11)::vtype_c c1{};
#pragma unroll
            for(int x = 0; x < 8; x++)
            {
                a1[x] = v_qw[(tm * KS_N + ks) * 8 + x];
                b1[x] = b[tn * 8 + x];
            }
            if(ks != 0)
            {
#pragma unroll
                for(int p_ = 0; p_ < 4; p_++)
                    c1[p_] = v_s[(tm * 2 + tn) * 4 + p_];
            }
            c1 = mma_11(a1, b1, c1);
#pragma unroll
            for(int p_ = 0; p_ < 4; p_++)
                v_s[(tm * 2 + tn) * 4 + p_] = c1[p_];
        };

        // The descriptors hold tile 2 here. Per round: the bytes of a full tile and of the
        // sequence's last, partial one. For all rounds: the key rows from the descriptors' tile
        // on, and its buffer.
        int d_kb[JU], d_np[JU];
#pragma unroll
        for(int u = 0; u < JU; u++)
        {
            d_kb[u] = KC * d_rb[u];
            d_np[u] = seqlen_k % KC * d_rb[u];
            asm volatile("" : "+s"(d_kb[u]), "+s"(d_np[u]));
        }
        int rl               = seqlen_k - 2 * KC;
        unsigned int lb      = dma_lds(2);
        unsigned int lb_end  = gated_mla_bwd::lds_addr(smem + NBUF * SZ_BUF);
        asm volatile("" : "+s"(rl), "+s"(lb), "+s"(lb_end));
        // Rounds 1 and 2 are moved on in the next tile's load phases 1 and 2, tile 0's included:
        // their descriptors start a tile back.
#pragma unroll
        for(int u = 1; u < JU; u++)
        {
            auto& r = d_rs[u];
            unsigned long long p =
                static_cast<unsigned int>(r[0]) | static_cast<unsigned long long>(static_cast<unsigned int>(r[1])) << 32;
            p -= static_cast<unsigned long long>(static_cast<unsigned int>(d_kb[u]));
            r = gated_mla_bwd::i32x4_t{static_cast<int>(p), static_cast<int>(p >> 32), r[2], 0x00020000};
            asm volatile("" : "+s"(r));
        }
        // The next tile's rows and buffer, once a tile, before its first round's descriptor.
        auto adv_tile = [&] {
            rl -= KC;
            lb += SZ_BUF * 2;
            unsigned int lw = lb - NBUF * SZ_BUF * 2;
            asm volatile("s_cmp_ge_u32 %0, %2\n\ts_cselect_b32 %0, %1, %0" : "+s"(lb) : "s"(lw), "s"(lb_end) : "scc");
            asm volatile("" : "+s"(rl));
        };
        // Round u's descriptor and m0 to that tile: the base a tile further; the size a full tile's,
        // the last tile's, or 0 past the sequence (those tiles fill a buffer no one reads with zeros).
        auto adv_round = [&](int u) {
            auto& r = d_rs[u];
            unsigned long long p =
                static_cast<unsigned int>(r[0]) | static_cast<unsigned long long>(static_cast<unsigned int>(r[1])) << 32;
            p += static_cast<unsigned long long>(static_cast<unsigned int>(d_kb[u]));
            int n = d_kb[u];
            asm volatile("s_cmp_ge_i32 %1, %3\n\ts_cselect_b32 %0, %0, %2\n\ts_cmp_gt_i32 %1, 0\n\ts_cselect_b32 %0, %0, 0"
                         : "+s"(n)
                         : "s"(rl), "s"(d_np[u]), "n"(KC)
                         : "scc");
            r       = gated_mla_bwd::i32x4_t{static_cast<int>(p), static_cast<int>(p >> 32), n, 0x00020000};
            d_m0[u] = lb + d_lds[u];
            asm volatile("" : "+s"(r), "+s"(d_m0[u]));
        };
        // The first key of the last tile this wave sees in full (as in the main pass: those tiles
        // take an unmasked exp), -1 if it sees none.
        int kfm = seqlen_k - KC;
        {
            int lim_c = qw0 + causal_offset + 1 - KC, cz = k.causal, qover = qw0 + WQ - seqlen_q;
            asm volatile("s_cmp_lg_i32 %2, 0\n\ts_cselect_b32 %1, %1, %0\n\ts_min_i32 %0, %0, %1\n\t"
                         "s_cmp_gt_i32 %3, 0\n\ts_cselect_b32 %0, -1, %0"
                         : "+s"(kfm), "+s"(lim_c)
                         : "s"(cz), "s"(qover)
                         : "scc");
        }

        a_kn = addr(0, OFF_KN, o_kn);
        a_kr = addr(0, OFF_KR, o_kr);
        // Tile 0's phase 1 packs this into dS(-1) = 0, with which its dQ(-1) phases add nothing.
        opus::clear(v_dp);
        pin(v_dp);
        for(int t = 0;;)
        {
            // 1: S steps 0, 1. Round 1's descriptor moves on to tile t + 2.
            op[0] = ld_s(0);
            op[1] = ld_s(1);
            __builtin_amdgcn_sched_barrier(0);
            if(JU > 1)
                adv_round(1);
            l_end();
            // Behind each MFMA, pinned and fenced there, one pack of dS(t - 1) to bf16: left to the
            // compiler, the eight packs come after the sixth MFMA, into registers it had lent to S.
            static_assert(NS == 16 && MR == 2);
#pragma unroll
            for(int i = 0; i < 8; i++)
            {
                xs1(i / 4, i / 2 % 2, i % 2, op[i / 4]);
                const opus::vector_t<float, 2> f2{v_dp[2 * i], v_dp[2 * i + 1]};
                auto h2 = opus::cast<bf16_t>(f2);
                pin(h2);  // or the packs sink below the last MFMA
                a_ds[2 * i]     = h2[0];
                a_ds[2 * i + 1] = h2[1];
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                __builtin_amdgcn_sched_group_barrier(0x002, 1, 0);
                __builtin_amdgcn_sched_barrier(0);
            }
            pin(v_s);
            pin(a_ds);
            c_end();
            // 2: S steps 2, 3. DMA of tile t + 2, a round in each of the load phases 2 .. 4: buffer
            // (t + 2) % NBUF last held tile t - 2, whose dQ every wave finished a phase ago. Behind
            // the MFMAs, the LDS addresses (a VALU add each; lone VALU instructions in a load phase
            // wait behind the other wave's): K nope of tile t + 1, V of tile t, K of tile t - 1 for
            // dQ (max(t - 1, 0) in SALU).
            // Round 2's descriptor moves on to tile t + 2 after round 0's DMA.
            op[0] = ld_s(2);
            op[1] = ld_s(3);
            __builtin_amdgcn_sched_barrier(0);
            dma_round(0);
            if(JU > 2)
                adv_round(2);
            l_end();
            xs(2, op[0]);
            xs(3, op[1]);
            {
                int tp = t;
                asm volatile("s_sub_i32 %0, %0, 1\n\ts_max_i32 %0, %0, 0" : "+s"(tp) : : "scc");
                a_kn = addr(t + 1, OFF_KN, o_kn);
                a_v  = addr(t, OFF_V, o_v);
                a_tn = addr(tp, OFF_KN, o_tn);
                a_tr = addr(tp, OFF_KR, o_tr);
            }
            GMLA_SCHED8(0x002, 1)
            pin(v_s);
            c_end();
            // 3: S steps 4, 5, on K rope and this wave's Q rope from sQR.
            op[0] = ld_s(4);
            op[1] = ld_s(5);
#pragma unroll
            for(int kr = 0; kr < KS_R; kr++)
                a_qr[kr] = s_qr.template load<8>(u_qr, kr * 32);
            __builtin_amdgcn_sched_barrier(0);
            if(JU > 1)
                dma_round(1);
            l_end();
            xs(4, op[0]);
            xs(5, op[1]);
            a_kr = addr(t + 1, OFF_KR, o_kr);
            pin(v_s);
            c_end();
            // 4: dP steps 0, 1; behind them S scale - LSE.
            op[0] = ld_dp(0);
            op[1] = ld_dp(1);
            __builtin_amdgcn_sched_barrier(0);
            if(JU > 2)
                dma_round(2);
            l_end();
            ex_dp(0, op[0]);
            ex_dp(1, op[1]);
#pragma unroll
            for(int e = 0; e < NS; e++)
                v_s[e] = v_s[e] * scale_log2 - lse2[e / (NS / MR)];
            GMLA_SCHED8(0x002, 2)
            pin(v_dp);
            pin(v_s);
            c_end();
            // 5: dP steps 2, 3; behind them the first half of P = exp. The rows and the buffer of
            // tile t + 3, and round 0's descriptor to it. A tile this wave does not see in full
            // (the causal diagonal, the ends of the sequences) is masked first.
            op[0] = ld_dp(2);
            op[1] = ld_dp(3);
            __builtin_amdgcn_sched_barrier(0);
            adv_tile();
            adv_round(0);
            {
                int kvp = t * KC;
                asm volatile("" : "+s"(kvp));
                if(__builtin_expect(kvp > kfm, 0))
                {
                    const int kv0 = t * KC;
#pragma unroll
                    for(int e = 0; e < NS; e++)
                    {
                        const int qp = qw0 + lane % 16 + 16 * (e / 8), kv = kv0 + 4 * (lane / 16) + 16 * (e / 4 % 2) + e % 4;
                        const bool ok = (kv < seqlen_k) & (qp < seqlen_q) & (!k.causal | (qp + causal_offset >= kv));
                        v_s[e]        = ok ? v_s[e] : -1e30f;
                    }
                    pin(v_s);
                }
            }
            l_end();
            ex_dp(2, op[0]);
            ex_dp(3, op[1]);
#pragma unroll
            for(int e = 0; e < NS / 2; e++)
                v_s[e] = __builtin_amdgcn_exp2f(v_s[e]);
            GMLA_SCHED8(0x400, 1)
            pin(v_dp);
            pin(v_s);
            c_end();
            // 6: dQ(t - 1) chunks 0, 1; behind them the second half of P.
            op[0] = ld_dq(0);
            op[1] = ld_dq(1);
            l_end();
            ex_dq(0, op[0]);
            ex_dq(1, op[1]);
#pragma unroll
            for(int e = NS / 2; e < NS; e++)
                v_s[e] = __builtin_amdgcn_exp2f(v_s[e]);
            GMLA_SCHED8(0x400, 1)
            pin(v_s);
            c_end();
            // 7: dQ(t - 1) chunks 2, 3; behind them dP - D.
            op[0] = ld_dq(2);
            op[1] = ld_dq(3);
            l_end();
            ex_dq(2, op[0]);
            ex_dq(3, op[1]);
#pragma unroll
            for(int e = 0; e < NS; e++)
                v_dp[e] = v_dp[e] - dd[e / (NS / MR)];
            GMLA_SCHED8(0x002, 2)
            pin(v_dp);
            c_end();
            // 8: dQ(t - 1) chunks 4, 5; behind them dS(t) = P (dP - D), fp32, left in dP's registers
            // for the next tile's phase 1 (or the tail). The loop test is made before the barriers.
            op[0] = ld_dq(4);
            op[1] = ld_dq(5);
            // The DMA of tile t + 1, issued a tile ago, is waited for here, in the last phase
            // before any wave reads it, and by count: this wave's JU loads of tile t + 2 are
            // still out. The memory path now and then stops for 1000 to 3600 cycles (ATT, in
            // every kernel that loads from global memory); the loads have had 3600 by now.
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt vmcnt(%0)" : : "n"(JU) : "memory");
            int tn = t + 1;
            asm volatile("" : "+s"(tn));
            const bool more = tn < T;
            l_end();
            ex_dq(4, op[0]);
            ex_dq(5, op[1]);
#pragma unroll
            for(int e = 0; e < NS; e++)
                v_dp[e] = v_dp[e] * v_s[e];
            GMLA_SCHED8(0x002, 2)
            pin(v_dp);
            c_end();
            if(!more)
                break;
            t = tn;
        }
        // Tail: dS of the last tile to bf16, then its dQ.
        a_ds = opus::cast<bf16_t>(v_dp);
        pin(a_ds);
        a_tn = addr(T - 1, OFF_KN, o_tn);
        a_tr = addr(T - 1, OFF_KR, o_tr);
#pragma unroll
        for(int c = 0; c < NQC; c += 2)
        {
            op[0] = ld_dq(c);
            op[1] = ld_dq(c + 1);
            l_end();
            ex_dq(c, op[0]);
            ex_dq(c + 1, op[1]);
            c_end();
        }
    }
#else
    if(T > 0)
    {
        a_kn = addr(0, OFF_KN, o_kn);
        a_kr = addr(0, OFF_KR, o_kr);
        a_tn = addr(0, OFF_KN, o_tn);
        prefetch(ld_s);
        // The tile index goes through an empty asm and the loop test is made before the N2 barrier:
        // otherwise the compiler derives counters from it and updates them, and tests, after that
        // barrier, in front of C1's first MFMA.
        for(int t = 0;;)
        {
            // C1: S(t), then dQ(t - 1) on chunks 0, 1. Steps 1 .. JU also issue the DMA of tile
            // t + 2 (set up in the N2 stage before), a round each, beside their MFMAs; tile t + 1's,
            // issued a tile ago, is waited for first. All waves have waited for their share of
            // tile t + 1 by the barrier that ends the lagging waves' C1, before any N2 reads it.
            // Buffer (t + 2) % NBUF last held tile t - 2, whose dQ every wave finished in the stage
            // before this one.
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
                            dma_round(st - 1);
                    });
            pin(v_s);
            stage_end();
            n1(t);
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
            int tn = t + 1;
            asm volatile("" : "+s"(tn));
            const bool more = tn < T;
            stage_end();
            if(!more)
                break;
            t = tn;
        }
        // Tail: dQ of the last tile.
        c_stage(number<NQC>{}, ld_dq, ex_dq);
        stage_end();
    }
#endif
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

// dQ from the pair pass's dS (gated_mla_bwd_ds): dQ = dS K softmax_scale, without recomputing S and
// dP. Workgroups, waves and key tiles as gated_mla_bwd_dq_kernel. Per key tile, K goes global -> LDS
// by DMA as there, and each wave's dS tile beside it (a tile causal masking drops reads as zeros);
// the dS tile becomes the A operand through transposed reads in the key order of K's transposed
// reads (tr_load_b_c_order). Tiles go NBUF - 1 ahead, each waited for by count; a wave whose queries
// all precede a tile's keys skips that tile's MFMAs.
__global__ __launch_bounds__(gated_mla_bwd_dqds::BLOCK, 1) void gated_mla_bwd_dq_ds_kernel(gated_mla_bwd_dq_kargs k)
{
    using namespace gated_mla_bwd_dqds;
    namespace dsl = gated_mla_bwd_ds;
    using gated_mla_bwd::bf16_t;
    using gated_mla_bwd::bvec;
    using opus::number;
    using opus::operator""_I;
    constexpr int MR = WQ / 16;  // M tiles per wave
    __shared__ __attribute__((aligned(16))) bf16_t smem[SMEM_ELEMS];

    const int tid  = static_cast<int>(opus::thread_id_x());
    const int lane = tid % 64;
    const int w    = __builtin_amdgcn_readfirstlane(tid / 64);
    // Workgroup id -> (head, sequence, query block) as in gated_mla_bwd_dq_kernel.
    const int wg   = static_cast<int>(opus::block_id_x());
    const auto ids = gated_mla_bwd::wg_map_sgpr(wg, k.H, k.num_seqs, k.n_q_blocks, k.hg);
    const int h = ids.h, g = ids.g, qb = k.n_q_blocks - 1 - ids.blk;
    auto sc            = [](int x) { return __builtin_amdgcn_readfirstlane(x); };
    const int q0       = sc(k.ptr_seqstart_q[g]);
    const int k0       = sc(k.ptr_seqstart_k[g]);
    const int seqlen_q = sc(k.ptr_seqstart_q[g + 1]) - q0;
    const int seqlen_k = sc(k.ptr_seqstart_k[g + 1]) - k0;
    const int qb0      = qb * BQ;
    if(qb0 >= seqlen_q)
        return;
    const int causal_offset = seqlen_k - seqlen_q;
    const int q_last        = (qb0 + BQ < seqlen_q ? qb0 + BQ : seqlen_q) - 1;
    int kv_end              = seqlen_k;
    if(k.causal && q_last + causal_offset + 1 < kv_end)
        kv_end = q_last + causal_offset + 1;
    const int T   = kv_end > 0 ? (kv_end + KC - 1) / KC : 0;  // key tiles of this block
    const int Tq  = (seqlen_q + 31) / 32;                        // query tiles of the sequence
    const int NKT = (seqlen_k + 31) / 32;                        // key tiles of the sequence
    const int iq  = qb0 / 32 + w;                                // this wave's query tile
    const long long seq0 = k.ptr_ds_seq[g];
    const long long head0 =
        (static_cast<long long>(sc(static_cast<int>(seq0 >> 32))) << 32 | static_cast<unsigned int>(sc(static_cast<int>(seq0)))) +
        h * dsl::tiles_per_head(seqlen_q, seqlen_k, k.causal);

    auto make_mma = [](auto e_m, auto e_n, auto e_k) {
        return opus::make_tiled_mma<bf16_t, bf16_t, float>(
            opus::seq<decltype(e_m)::value, decltype(e_n)::value, decltype(e_k)::value>{},
            opus::seq<1, 1, 1>{},
            opus::seq<16, 16, 32>{},
            opus::mfma_adaptor_swap_ab{});
    };
    // Every MFMA step: [32 queries x 32 columns] += dS [32 x 32 keys] K [32 keys x 32 columns].
    auto mma_k1     = make_mma(number<MR>{}, number<KC / 16>{}, number<1>{});
    const auto c_c  = opus::make_tuple(0_I, lane % mma_k1.grpn_c, 0_I, lane / mma_k1.grpn_c);
    using frag_t    = bvec<16>;
    using afrag_t   = opus::vector_t<bf16_t, decltype(mma_k1)::tile_a_len>;
    using acc_t     = typename decltype(mma_k1)::vtype_c;
    constexpr int NS = decltype(mma_k1)::tile_c_len, NQC = D_QK / 32, KS_N = D_NOPE / 32;
    static_assert(MR == 2 && KC == 32 && NS == 16, "one 32 x 32 dS tile per wave and key tile");

    auto rec = [](long long elems, int elem_bytes) -> unsigned int {
        const long long b = elems > 0 ? elems * elem_bytes : 0;
        return b >= 0xffffffffLL ? 0xffffffffu : static_cast<unsigned int>(b);
    };

    // DMA per key tile kt into buffer kt % NBUF: K nope and rope as in gated_mla_bwd_dq_kernel
    // (J_N + J_R loads of 1 KB to waves w, w + NW; a wave without a second repeats its first), then
    // this wave's dS tile, two loads of 1 KB with 16-byte chunks swizzled within rows.
    constexpr int SPR_N = LD_KN * 2 / 16, SPR_R = LD_KR * 2 / 16;
    constexpr int J_N = KC * SPR_N / 64, J_R = KC * SPR_R / 64, JK = (J_N + J_R + NW - 1) / NW, R = JK + 2;
    static_assert(KC * SPR_N % 64 == 0 && KC * SPR_R % 64 == 0 && JK == 2, "whole loads per region");
    constexpr int OOB = 1 << 30;
    const bf16_t* d_ptr[JK];
    int d_rb[JK], d_lds[JK], d_os[JK];
#pragma unroll
    for(int u = 0; u < JK; u++)
    {
        const int j   = w + NW * u < J_N + J_R ? w + NW * u : w + NW * (u - 1);
        const bool nope = j < J_N;
        const int jj  = nope ? j : j - J_N;
        d_rb[u]       = (nope ? k.stride_k_n : k.stride_k_rot_n) * 2;
        d_ptr[u]      = nope ? reinterpret_cast<const bf16_t*>(k.ptr_k_nope) + static_cast<long long>(k0) * k.stride_k_n +
                              static_cast<long long>(h) * k.stride_k_h
                             : reinterpret_cast<const bf16_t*>(k.ptr_k_rot) + static_cast<long long>(k0) * k.stride_k_rot_n;
        d_lds[u]      = (nope ? OFF_KN : OFF_KR) * 2 + jj * 1024;
        const int spr = nope ? SPR_N : SPR_R, dch = (nope ? D_NOPE : D_ROT) / 8;
        const int sl = 64 * jj + lane, c = sl % spr;
        d_os[u]      = c < dch ? sl / spr * d_rb[u] + c * 16 : OOB;
    }
    int ds_os[2];
#pragma unroll
    for(int hh = 0; hh < 2; hh++)
    {
        const int row = hh * 16 + lane / 4;
        ds_os[hh]     = row * 64 + ((lane % 4) ^ (row / 4 % 4)) * 16;
    }
    auto dma_lds = [&](int b) { return gated_mla_bwd::lds_addr(smem + b * SZ_BUF); };
    auto dma     = [&](int kt, int b) {
        int rows = seqlen_k - kt * KC;
        asm("s_min_i32 %0, %0, %1\n\ts_max_i32 %0, %0, 0" : "+s"(rows) : "n"(KC) : "scc");
        const unsigned int lb = dma_lds(b);
#pragma unroll
        for(int u = 0; u < JK; u++)
        {
            auto rs = gated_mla_bwd::dma_rsrc(d_ptr[u], 0);
            asm("s_mul_i32 %0, %1, %2" : "=s"(rs[2]) : "s"(rows), "s"(d_rb[u]));
            gated_mla_bwd::dma16(rs, d_os[u], lb + d_lds[u]);
            d_ptr[u] += KC * d_rb[u] / 2;
        }
        const int i0      = dsl::first_q_tile(kt, Tq, causal_offset, k.causal);
        const bool valid  = kt < NKT && iq >= i0 && iq < Tq;
        const long long t = head0 + dsl::col_offset(kt, Tq, causal_offset, k.causal) + (valid ? iq - i0 : 0);
        const auto rs     = gated_mla_bwd::dma_rsrc(reinterpret_cast<const bf16_t*>(k.ptr_ds) + t * dsl::TILE,
                                                valid ? dsl::TILE * 2 : 0);
#pragma unroll
        for(int hh = 0; hh < 2; hh++)
            asm volatile("s_mov_b32 m0, %2\n\tbuffer_load_dwordx4 %0, %1, 0 offen" GMLA_DS_CACHE " lds"
                         : : "v"(ds_os[hh]), "s"(rs), "s"(lb + (OFF_DS + w * dsl::TILE) * 2 + hh * 1024) : "m0");
    };

    // K transposed (B of dQ) as in gated_mla_bwd_dq_kernel; dS (A of dQ): lane i of 16-lane group G
    // reads rows (keys) 16 n + 4 G + i / 4, the 4 queries 16 m + 4 (i % 4) .. at element
    // 8 (i % 4) + 4 m of the row (chunk i % 4, swizzled by G), and gets keys 16 n + 4 G + 0 .. 3 of
    // query 16 m + i: slot 4 n + e of M tile m, as tr_load_b_c_order's key order.
    const int o_tn = gated_mla_bwd::tr_c_order_lane<LD_KN>(lane), o_tr = gated_mla_bwd::tr_c_order_lane<LD_KR>(lane);
    const int G = lane / 16, i16 = lane % 16;
    const int o_ds = (4 * G + i16 / 4) * 32 + 8 * ((i16 % 4) ^ G);
    auto at = [](unsigned int a) { return opus::make_smem(reinterpret_cast<bf16_t*>(static_cast<__UINTPTR_TYPE__>(a))); };

    acc_t v_dq[NQC];
#pragma unroll
    for(int c = 0; c < NQC; c++)
        opus::clear(v_dq[c]);

    auto stage_end = [] {
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    };
#pragma unroll
    for(int x = 0; x < NBUF - 1; x++)
        dma(x, x);
    static_assert((NBUF - 2) * R == 12, "the wait count below");
    for(int kt = 0; kt < T; kt++)
    {
        // This wave's loads of tile kt are done once at most those of the NBUF - 2 tiles after it
        // remain; the barrier then has every wave's, and every wave done with tile kt - 1, whose
        // buffer the next DMA fills.
        asm volatile("s_waitcnt vmcnt(12)" ::: "memory");
        stage_end();
        dma(kt + NBUF - 1, (kt + NBUF - 1) % NBUF);
        if(iq >= dsl::first_q_tile(kt, Tq, causal_offset, k.causal) && iq < Tq)
        {
            const int b          = kt % NBUF;
            const unsigned int a_tn = gated_mla_bwd::lds_addr(smem + b * SZ_BUF + OFF_KN) + o_tn * 2;
            const unsigned int a_tr = gated_mla_bwd::lds_addr(smem + b * SZ_BUF + OFF_KR) + o_tr * 2;
            const unsigned int a_ds = gated_mla_bwd::lds_addr(smem + b * SZ_BUF + OFF_DS + w * dsl::TILE) + o_ds * 2;
            opus::vector_t<opus::u32_t, 8> r;
#pragma unroll
            for(int m = 0; m < MR; m++)
#pragma unroll
                for(int n = 0; n < 2; n++)
                {
                    const auto x     = gated_mla_bwd::tr_read(at(a_ds), n * 16 * 32 + m * 4);
                    r[m * 4 + n * 2]     = x[0];
                    r[m * 4 + n * 2 + 1] = x[1];
                }
            const afrag_t a = __builtin_bit_cast(afrag_t, r);
#pragma unroll
            for(int c = 0; c < NQC; c++)
            {
                const frag_t bf = c < KS_N ? gated_mla_bwd::tr_load_b_c_order_at<2, LD_KN>(at(a_tn), 32 * c)
                                           : gated_mla_bwd::tr_load_b_c_order_at<2, LD_KR>(at(a_tr), 32 * (c - KS_N));
                v_dq[c]         = mma_k1(a, bf, v_dq[c]);
            }
        }
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

    const int qw0    = qb0 + w * WQ;
    const int rows_q = seqlen_q - qw0 < WQ ? (seqlen_q - qw0 > 0 ? seqlen_q - qw0 : 0) : WQ;
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

// dQ pass on 32x32x16 MFMAs, as gated_mla_bwd_dq_kernel (same tiles, stages and stagger): each
// MFMA here takes 32 cycles of which 8 issue it, leaving three times a 16x16x32 MFMA's issue
// cycles to the LDS reads and SALU work beside it. S^T = K Q^T and dP^T = V dO^T, so that a lane
// holds one query's column (its LSE and D are one value each) and 16 keys of it: rows
// 8 (i / 4) + 4 (lane / 32) + i % 4 of the C fragment. Elements 8 b .. 8 b + 7 of dS^T are then
// directly the B operand of key block b of dQ^T = K^T dS^T, whose A operand is read transposed
// from the K tile in the same key order (two ds_read_b64_tr_b16 per lane: rows 16 b + 4 h + 0 .. 3
// and 16 b + 8 + 4 h + 0 .. 3, h = lane / 32).
//   C1: S^T(t)                    (12 MFMA) and dQ^T(t - 1) on columns 0 .. 63    (4 MFMA);
//       DMA of tile t + 2 between the MFMAs
//   N1: P^T = exp(S^T(t) - LSE), in fp32 in S^T's registers until dS
//   C2: dP^T(t)                   ( 8 MFMA) and dQ^T(t - 1) on columns 64 .. 191 (8 MFMA)
//   N2: dS^T(t) = P^T (dP^T(t) - D), in bf16
// A step is one MFMA; its A fragment (8 bf16) is read from LDS LA steps ahead through a ring.
// Key tiles are laid out as gated_mla_bwd_dq32 describes.
__global__ __launch_bounds__(gated_mla_bwd_dq32::BLOCK, 1) void gated_mla_bwd_dq32_kernel(gated_mla_bwd_dq_kargs k)
{
    using namespace gated_mla_bwd_dq32;
    using gated_mla_bwd::bf16_t;
    using gated_mla_bwd::bvec;
    using opus::number;
    constexpr float LOG2_E = 1.44269504089f;
    __shared__ __attribute__((aligned(16))) bf16_t smem[SMEM_ELEMS];

    const int tid  = static_cast<int>(opus::thread_id_x());
    const int lane = tid % 64;
    const int w    = __builtin_amdgcn_readfirstlane(tid / 64);
    const int lq = lane % 32, lh = lane / 32;  // this lane's query (C column) and K half
    // Workgroup id -> (head, sequence, query block): gated_mla_bwd::wg_map, the query blocks
    // last-first (the causal long ones first).
    const int wg   = static_cast<int>(opus::block_id_x());
    const auto ids = gated_mla_bwd::wg_map_sgpr(wg, k.H, k.num_seqs, k.n_q_blocks, k.hg);
    const int h = ids.h, g = ids.g, qb = k.n_q_blocks - 1 - ids.blk;
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
    const int T = kv_end > 0 ? (kv_end + KC - 1) / KC : 0;

    static_assert(KC == 32 && WQ == 32, "one 32x32 C fragment per tile and wave");
    using frag_t = bvec<8>;                      // an MFMA operand: 8 bf16 per lane
    using acc_t  = opus::vector_t<float, 16>;    // a 32 x 32 C fragment
    auto mfma    = [](const frag_t& a, const frag_t& b, const acc_t& c) -> acc_t {
        return __builtin_amdgcn_mfma_f32_32x32x16_bf16(a, b, c, 0, 0, 0);
    };
    constexpr int KS_S = D_QK / 16, KS_N = D_NOPE / 16, KS_R = D_ROT / 16, KS_P = D_V / 16, NQC = D_QK / 32;

    auto rec = [](long long elems, int elem_bytes) -> unsigned int {
        const long long b = elems > 0 ? elems * elem_bytes : 0;
        return b >= 0xffffffffLL ? 0xffffffffu : static_cast<unsigned int>(b);
    };

    // This wave's 32 queries: Q nope and dO as the B operands of S^T and dP^T (lane: row lq,
    // columns 16 s + 8 lh .. + 7), Q rope in LDS; LSE (log2 domain) and D of query lq. Rows past
    // seqlen_q read as 0 and are dropped on store.
    const int qw0    = qb0 + w * WQ;
    const int rows_q = seqlen_q - qw0 < WQ ? (seqlen_q - qw0 > 0 ? seqlen_q - qw0 : 0) : WQ;
    auto g_q  = opus::make_gmem(reinterpret_cast<const bf16_t*>(k.ptr_q) + static_cast<long long>(q0 + qw0) * k.stride_q_n +
                                   static_cast<long long>(h) * k.stride_q_h,
                               rec(static_cast<long long>(rows_q) * k.stride_q_n, 2));
    auto g_do = opus::make_gmem(reinterpret_cast<const bf16_t*>(k.ptr_do) + static_cast<long long>(q0 + qw0) * k.stride_do_n +
                                    static_cast<long long>(h) * k.stride_do_h,
                                rec(static_cast<long long>(rows_q) * k.stride_do_n, 2));
    frag_t v_q[KS_N], v_do[KS_P];
#pragma unroll
    for(int s = 0; s < KS_N; s++)
        v_q[s] = g_q.template load<8>(lq * k.stride_q_n + 16 * s + 8 * lh);
    // Q rope rows of this wave -> sQR (lane l: row l / 2, chunks 4 (l % 2) .. + 3).
    {
        const int r = lane / 2, c0 = (lane % 2) * (D_ROT / 2);
        frag_t t_[D_ROT / 2 / 8];
#pragma unroll
        for(int u = 0; u < D_ROT / 2 / 8; u++)
            t_[u] = g_q.template load<8>(r * k.stride_q_n + D_NOPE + c0 + u * 8);
#pragma unroll
        for(int u = 0; u < D_ROT / 2 / 8; u++)
            *reinterpret_cast<frag_t*>(smem + OFF_QR + (w * WQ + r) * LD_QR + c0 + u * 8) = t_[u];
    }
#pragma unroll
    for(int s = 0; s < KS_P; s++)
        v_do[s] = g_do.template load<8>(lq * k.stride_do_n + 16 * s + 8 * lh);
    float lse2, dd;
    {
        auto g_lse = opus::make_gmem(k.ptr_lse + static_cast<long long>(h) * k.stride_lse_h + q0 + qw0, rec(rows_q, 4));
        auto g_d   = opus::make_gmem(k.ptr_d + static_cast<long long>(h) * k.stride_lse_h + q0 + qw0, rec(rows_q, 4));
        lse2       = g_lse.template load<1>(lq)[0] * LOG2_E;
        dd         = g_d.template load<1>(lq)[0];
    }

    // DMA of key tile t into buffer t % NBUF (gated_mla_bwd::dma16), as in gated_mla_bwd_dq_kernel,
    // into the swizzled rows: slot s, row s / SPR, physical chunk p = s % SPR, holds chunk p ^ swz.
    auto swz = [](int row) { return ((row >> 2) & 1) | (((row >> 4) & 1) << 1); };
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
        const int sl = 64 * jj + lane, c = sl % spr, row = sl / spr;
        d_os[u]      = c < dch ? row * d_rb[u] + (c ^ swz(row)) * 16 : OOB;
    }
    auto dma_rows = [&](int t) {
        int r = seqlen_k - t * KC;
        asm("s_min_i32 %0, %0, %1\n\ts_max_i32 %0, %0, 0" : "+s"(r) : "n"(KC) : "scc");
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
    auto dma_wait  = [] { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); };
    auto stage_end = [] {
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    };
    auto pin = [](auto& x) { asm volatile("" : "+v"(x)); };

    // LDS reads at a per-lane address (made and pinned in the N stage before the C stage that
    // reads it) plus immediates. A fragments of K and V: row lq, logical chunk 2 s + lh, stored at
    // 2 (s ^ (swz >> 1)) + (lh ^ (swz & 1)): K steps 2 s and 2 s + 1 swap for rows 16 .. 31, so
    // even and odd steps have their own lane offset (o_*[0], o_*[1]). K^T for dQ^T: the transposed
    // reads (group G = lane / 16 reads rows 4 (G / 2) .. + 3, columns 16 (G % 2) .. + 15 of 32 c ..;
    // its lane i receives column i) have rows 16 b + 8 r and chunks 4 c as immediates; their swz is
    // G / 2 | b << 1, so key blocks b = 0, 1 have their own lane offset (o_t*[b]). Q rope as the B
    // operand of S^T's last steps: row w WQ + lq of sQR, not swizzled.
    const int G = lane / 16, gi = lane % 16;
    auto o_frag = [&](int ld, int odd) {
        const int f = swz(lq);
        return lq * ld + 8 * (lh ^ (f & 1)) + (odd ? -16 : 16) * (f >> 1);
    };
    auto o_trans = [&](int ld, int b) {
        const int pc = (2 * (G % 2) + (gi % 4) / 2) ^ (G / 2) ^ (b << 1);
        return (4 * (G / 2) + gi / 4) * ld + 8 * pc + 4 * (gi % 2);
    };
    const int o_kn[2] = {o_frag(LD_KN, 0), o_frag(LD_KN, 1)}, o_kr[2] = {o_frag(LD_KR, 0), o_frag(LD_KR, 1)};
    const int o_v[2]  = {o_frag(LD_V, 0), o_frag(LD_V, 1)};
    const int o_tn[2] = {o_trans(LD_KN, 0), o_trans(LD_KN, 1)}, o_tr[2] = {o_trans(LD_KR, 0), o_trans(LD_KR, 1)};
    auto addr = [&](int t, int off, int o) {
        unsigned int a = gated_mla_bwd::lds_addr(smem + t % NBUF * SZ_BUF + off) + o * 2;
        pin(a);
        return a;
    };
    auto lds16 = [](unsigned int a) -> frag_t {
        return *reinterpret_cast<const OPUS_LDS_ADDR frag_t*>(static_cast<__UINTPTR_TYPE__>(a));
    };
    auto at = [](unsigned int a) { return opus::make_smem(reinterpret_cast<bf16_t*>(static_cast<__UINTPTR_TYPE__>(a))); };
    auto ld_kt = [&](unsigned int a, int ld, int col, int b) -> frag_t {
        const auto r0 = gated_mla_bwd::tr_read(at(a), 16 * b * ld + col);
        const auto r1 = gated_mla_bwd::tr_read(at(a), (16 * b + 8) * ld + col);
        return __builtin_bit_cast(frag_t, opus::vector_t<opus::u32_t, 4>{r0[0], r0[1], r1[0], r1[1]});
    };
    const unsigned int a_qr0 = gated_mla_bwd::lds_addr(smem + OFF_QR) + ((w * WQ + lq) * LD_QR + 8 * lh) * 2;
    // S^T(t): K nope, K rope; dP^T(t): V; dQ^T(t - 1): K nope, K rope, transposed; each a pair
    // (even / odd K steps, or key blocks 0 / 1).
    unsigned int a_kn[2], a_kr[2], a_v[2], a_tn[2], a_tr[2];
    auto addr2 = [&](unsigned int(&a)[2], int t, int off, const int(&o)[2]) {
        a[0] = addr(t, off, o[0]);
        a[1] = addr(t, off, o[1]);
    };
    // A fragment of one step: S^T K step s, dP^T K step s, dQ^T step j (column chunk j / 2, key
    // block j % 2).
    auto ld_s = [&](int s) -> frag_t {
        return s < KS_N ? lds16(a_kn[s % 2] + 32 * s) : lds16(a_kr[(s - KS_N) % 2] + 32 * (s - KS_N));
    };
    auto ld_dp = [&](int s) -> frag_t { return lds16(a_v[s % 2] + 32 * s); };
    auto ld_dq = [&](int j) -> frag_t {
        const int c = j / 2, b = j % 2;
        return c < D_NOPE / 32 ? ld_kt(a_tn[b], LD_KN, 32 * c, b) : ld_kt(a_tr[b], LD_KR, 32 * (c - D_NOPE / 32), b);
    };

    acc_t v_s, v_dp, v_dq[NQC];
#pragma unroll
    for(int c = 0; c < NQC; c++)
        opus::clear(v_dq[c]);
    // dS^T of the previous tile per key block, the B operand of its dQ^T. Zero before tile 0,
    // whose dQ^T(t - 1) steps then add nothing (they read tile 0's K, which is finite).
    frag_t b_ds[2];
    opus::clear(b_ds[0]);
    opus::clear(b_ds[1]);
    // Q rope's B fragments of S^T's last KS_R steps, read QA steps ahead into QA slots.
    constexpr int QA = 2;
    frag_t b_qr[QA];
    static_assert(KS_R % QA == 0 && KS_N >= QA, "Q rope fragments read within S^T's nope steps");

    // A fragments of LA steps in flight, in a ring of LA + 1 slots, so that a step's reads (of step
    // + LA) go to the slot step - 1 consumed and can issue beside the step's MFMA: it leaves 24 of
    // its 32 cycles to other instructions.
#ifndef GMLA_DQ32_LA
#define GMLA_DQ32_LA 3
#endif
    constexpr int LA = GMLA_DQ32_LA, RS = LA + 1;
    frag_t ring[RS];
    auto c_stage = [&](auto n_steps, auto ld, auto ex) {
        constexpr int N = decltype(n_steps)::value;
#pragma unroll
        for(int st = 0; st < N; st++)
        {
            ex(st, ring[st % RS]);
            if(st + LA < N)
                ring[(st + LA) % RS] = ld(st + LA);
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);  // MFMA
            __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);  // DS read
            __builtin_amdgcn_sched_group_barrier(0x004, 3, 0);  // SALU
            __builtin_amdgcn_sched_barrier(0);
        }
    };
    auto prefetch = [&](auto ld) {
#pragma unroll
        for(int st = 0; st < LA; st++)
            ring[st] = ld(st);
    };
    auto pin_ring = [&] {
#pragma unroll
        for(int r = 0; r < LA; r++)
            pin(ring[r]);
    };
    const acc_t zero{};
    auto ex_s = [&](int s, const frag_t& a) {
        v_s = mfma(a, s < KS_N ? v_q[s] : b_qr[(s - KS_N) % QA], s == 0 ? zero : v_s);
        if(s + QA >= KS_N && s + QA < KS_S)
            b_qr[(s + QA - KS_N) % QA] = lds16(a_qr0 + 32 * (s + QA - KS_N));
    };
    auto ex_dp = [&](int s, const frag_t& a) { v_dp = mfma(a, v_do[s], s == 0 ? zero : v_dp); };
    auto ex_dq = [&](int j, const frag_t& a) { v_dq[j / 2] = mfma(a, b_ds[j % 2], v_dq[j / 2]); };

    const float scale_log2 = k.softmax_scale * LOG2_E;
    // C element i of S^T / dP^T: key 8 (i / 4) + 4 lh + i % 4 of the tile, query qw0 + lq.
    constexpr int NS = 16;

    // N1: the first steps of C2 (and its addresses); P^T = exp(S^T(t) - LSE), in place in v_s.
    auto n1 = [&](int t, int tp) {
        addr2(a_v, t, OFF_V, o_v);
        addr2(a_tr, tp, OFF_KR, o_tr);
        prefetch(ld_dp);
        const int kv0 = t * KC;
        // As in the main pass: tiles this wave sees in full take an unmasked exp.
        const bool full = (kv0 + KC <= seqlen_k) & (qw0 + WQ <= seqlen_q) &
                          (!k.causal | (kv0 + KC - 1 <= qw0 + causal_offset));
        if(full)
        {
#pragma unroll
            for(int i = 0; i < NS; i++)
                v_s[i] = __builtin_amdgcn_exp2f(v_s[i] * scale_log2 - lse2);
        }
        else
        {
            const int qp = qw0 + lq;
#pragma unroll
            for(int i = 0; i < NS; i++)
            {
                const int kv  = kv0 + 8 * (i / 4) + 4 * lh + i % 4;
                const bool ok = (kv < seqlen_k) & (qp < seqlen_q) & (!k.causal | (qp + causal_offset >= kv));
                const float x = v_s[i] * scale_log2 - lse2;
                v_s[i]        = __builtin_amdgcn_exp2f(ok ? x : -1e30f);
            }
        }
        pin(v_s);
        pin_ring();
    };
    // N2: the first steps of the next C stage (C1 of tile t + 1, or the tail's dQ^T of tile t);
    // dS^T(t) = P^T (dP^T(t) - D) in bf16.
    auto n2 = [&](int t) {
        addr2(a_tn, t, OFF_KN, o_tn);
        if(t + 1 < T)
        {
            addr2(a_kn, t + 1, OFF_KN, o_kn);
            addr2(a_kr, t + 1, OFF_KR, o_kr);
            prefetch(ld_s);
        }
        else
        {
            addr2(a_tr, t, OFF_KR, o_tr);
            prefetch(ld_dq);
        }
#pragma unroll
        for(int b = 0; b < 2; b++)
        {
            opus::vector_t<float, 8> v_ds;
#pragma unroll
            for(int x = 0; x < 8; x++)
                v_ds[x] = v_s[8 * b + x] * (v_dp[8 * b + x] - dd);
            b_ds[b] = opus::cast<bf16_t>(v_ds);
            pin(b_ds[b]);
        }
        pin_ring();
    };

    // Tiles 0 and 1 in LDS, and Q, dO, LSE, D in registers, before the loop (0x0f70 is
    // s_waitcnt vmcnt(0)): otherwise the compiler waits for the latter inside the loop with
    // vmcnt counts that the DMA then makes block.
    dma(0);
    dma(1);
    __builtin_amdgcn_s_waitcnt(0x0f70);
    dma_wait();
    stage_end();
    const bool lag = w >= NW / 2;
    if(lag)
        stage_end();
    if(T > 0)
    {
        addr2(a_kn, 0, OFF_KN, o_kn);
        addr2(a_kr, 0, OFF_KR, o_kr);
        addr2(a_tn, 0, OFF_KN, o_tn);
        prefetch(ld_s);
        for(int t = 0; t < T; t++)
        {
            const int tp = t > 0 ? t - 1 : 0;
            // C1: S^T(t), then dQ^T(t - 1) on chunks 0, 1. Steps 1 .. JU also issue the DMA of tile
            // t + 2, a round each, beside their MFMAs; tile t + 1's, issued a tile ago, is waited
            // for first. All waves have waited for their share of tile t + 1 by the barrier that
            // ends the lagging waves' C1, before any N2 reads it. Buffer (t + 2) % NBUF last held
            // tile t - 2, whose dQ^T every wave finished in the stage before this one.
            const int r2          = dma_rows(t + 2);
            const unsigned int l2 = dma_lds(t + 2);
            c_stage(number<KS_S + 4>{},
                    [&](int st) { return st < KS_S ? ld_s(st) : ld_dq(st - KS_S); },
                    [&](int st, const frag_t& a) {
                        if(st < KS_S)
                            ex_s(st, a);
                        else
                            ex_dq(st - KS_S, a);
                        if(st == 1)
                            dma_wait();
                        if(st >= 1 && st <= JU)
                            dma_round(r2, l2, st - 1);
                    });
            pin(v_s);
            stage_end();
            n1(t, tp);
            stage_end();
            // C2: dP^T(t), then dQ^T(t - 1) on chunks 2 .. 5.
            c_stage(number<KS_P + 2 * NQC - 4>{},
                    [&](int st) { return st < KS_P ? ld_dp(st) : ld_dq(st - KS_P + 4); },
                    [&](int st, const frag_t& a) {
                        if(st < KS_P)
                            ex_dp(st, a);
                        else
                            ex_dq(st - KS_P + 4, a);
                    });
            pin(v_dp);
            stage_end();
            n2(t);
            stage_end();
        }
        // Tail: dQ^T of the last tile.
        c_stage(number<2 * NQC>{}, ld_dq, ex_dq);
        stage_end();
    }
    if(!lag)
        stage_end();

    // dQ^T C element i of chunk c is column 32 c + 8 (i / 4) + 4 lh + i % 4 of query qw0 + lq:
    // four runs of 4 columns, one 8-byte store each.
    auto g_dq = opus::make_gmem(reinterpret_cast<bf16_t*>(k.ptr_dq) + static_cast<long long>(q0 + qw0) * k.stride_dq_n +
                                    static_cast<long long>(h) * k.stride_dq_h,
                                rec(static_cast<long long>(rows_q) * k.stride_dq_n, 2));
#pragma unroll
    for(int c = 0; c < NQC; c++)
    {
#pragma unroll
        for(int i = 0; i < NS; i++)
            v_dq[c][i] *= k.softmax_scale;
        const auto o = opus::cast<bf16_t>(v_dq[c]);
#pragma unroll
        for(int r = 0; r < 4; r++)
            g_dq.template store<4>(opus::vector_t<bf16_t, 4>{o[4 * r], o[4 * r + 1], o[4 * r + 2], o[4 * r + 3]},
                                   lq * k.stride_dq_n + 32 * c + 8 * r + 4 * lh);
    }
}

// dK / dV pass, 8 waves. The two waves of a SIMD (w, w + 4) share 32 keys and split the work
// along its two dependency chains, so each LDS fragment of Q and dO feeds two M tiles:
//   S wave  (w < 4):  S^T = K Q^T, P, dV += P^T dO
//   dP wave (w >= 4): dP^T = V dO^T, dS = P (dP - D), dK += dS^T Q
// Each query tile is two stages, each closed by a workgroup barrier. In each, one wave of a SIMD
// issues 40 MFMAs while the other's VALU work fills the cycles beside them; the dP wave's dS and
// dK run a tile late:
//   A(t): S wave  S^T(t) (24 MFMA), then dV(t - 1) (16 MFMA)
//         dP wave dS(t - 1), from P(t - 1) (fp32, through LDS) and its own dP^T(t - 1)
//   B(t): S wave  P(t) = exp(S^T(t) - LSE), to LDS in fp32 and kept in bf16 for dV(t)
//         dP wave dP^T(t) (16 MFMA), then dK(t - 1) (24 MFMA)
// Each MFMA run starts with the result the next stage needs. Query tiles (Q, dO, LSE, D) go
// global -> LDS by DMA into NBUF buffers, issued beside the MFMAs: the S waves' share of tile t + 2
// in A(t), the dP waves' share of tile t + 3 in B(t). Every GEMM is steps of 4 16x16x32
// MFMAs, [32 x 32] += [32 x 32] [32 x 32]^T: S^T and dP^T over a K step of 32 dims, dV and dK
// over a chunk of 32 columns, with P^T and dS^T (C fragments of S^T and dP^T) as A and
// transposed reads of the dO and Q tiles as B. In MFMA stages the B fragments stream from LDS LA
// steps ahead, the first LA steps (and the stage's LDS addresses) made in the wave's VALU stage
// before, so that MFMA stages carry no lone VALU instructions.
template <bool DS>
__global__ __launch_bounds__(gated_mla_bwd_pair::BLOCK, 1) void gated_mla_bwd_dkdv_pair_kernel(gated_mla_bwd_kargs k)
{
    using namespace gated_mla_bwd_pair;
    using gated_mla_bwd::bf16_t;
    using gated_mla_bwd::bvec;
    using gated_mla_bwd::f32x4_t;
    using opus::number;
    using opus::operator""_I;
    constexpr float LOG2_E = 1.44269504089f;
    constexpr int MR       = PR / 16;  // M tiles per wave
    constexpr int NT       = BQ / 16;  // N tiles of S^T / dP^T
    __shared__ __attribute__((aligned(16))) bf16_t smem[SMEM_ELEMS];

    const int tid     = static_cast<int>(opus::thread_id_x());
    const int lane    = tid % 64;
    const int w       = __builtin_amdgcn_readfirstlane(tid / 64);
    const int pr      = w % NP;
    const bool s_wave = w < NP;
    // Workgroup id -> (head, sequence, key block): gated_mla_bwd::wg_map.
    const int wg   = static_cast<int>(opus::block_id_x());
    const auto ids = gated_mla_bwd::wg_map_sgpr(wg, k.H, k.num_seqs, k.n_kv_blocks, k.hg);
    const int h = ids.h, g = ids.g, j = ids.blk;
    auto sc            = [](int x) { return __builtin_amdgcn_readfirstlane(x); };
    const int q0       = sc(k.ptr_seqstart_q[g]);
    const int k0       = sc(k.ptr_seqstart_k[g]);
    const int seqlen_q = sc(k.ptr_seqstart_q[g + 1]) - q0;
    const int seqlen_k = sc(k.ptr_seqstart_k[g + 1]) - k0;
    const int kv0      = j * BK;
    if(kv0 >= seqlen_k)
        return;
    // Bottom-right causal: query qp sees keys kv <= qp + causal_offset. Query tiles i0 .. T - 1.
    const int causal_offset = seqlen_k - seqlen_q;
    const int T             = (seqlen_q + BQ - 1) / BQ;
    int i0                  = 0;
    if(k.causal && kv0 - causal_offset > 0)
        i0 = (kv0 - causal_offset) / BQ;

    auto make_mma = [](auto e_m, auto e_n, auto e_k) {
        return opus::make_tiled_mma<bf16_t, bf16_t, float>(
            opus::seq<decltype(e_m)::value, decltype(e_n)::value, decltype(e_k)::value>{},
            opus::seq<1, 1, 1>{},
            opus::seq<16, 16, 32>{},
            opus::mfma_adaptor_swap_ab{});
    };
    static_assert(BQ == 32, "P^T / dS^T are the A operand of one K step");
    auto mma_k1 = make_mma(number<MR>{}, number<NT>{}, number<1>{});             // every step
    auto mma_kn = make_mma(number<MR>{}, number<NT>{}, number<D_NOPE / 32>{});   // for K's A layout
    auto mma_kr = make_mma(number<MR>{}, number<NT>{}, number<D_ROT / 32>{});
    auto mma_v  = make_mma(number<MR>{}, number<NT>{}, number<D_V / 32>{});      // for V's A layout
    const auto c_ab = opus::make_tuple(0_I, lane % mma_k1.grpm_a, 0_I, lane / mma_k1.grpm_a);
    const auto c_c  = opus::make_tuple(0_I, lane % mma_k1.grpn_c, 0_I, lane / mma_k1.grpn_c);
    using frag_t  = bvec<16>;
    using afrag_t = opus::vector_t<bf16_t, decltype(mma_k1)::tile_a_len>;
    using acc_t   = typename decltype(mma_k1)::vtype_c;
    constexpr int NS = decltype(mma_k1)::tile_c_len;
    static_assert(NS == MR * NT * 4);
    constexpr int KS = D_QK / 32, KS_N = D_NOPE / 32, KS_R = D_ROT / 32, KS_P = D_V / 32;
    constexpr int NV = D_V / 32, NK = D_QK / 32, NK_N = D_NOPE / 32;  // 32-column chunks of dV, dK

    auto rec = [](long long elems, int elem_bytes) -> unsigned int {
        const long long b = elems > 0 ? elems * elem_bytes : 0;
        return b >= 0xffffffffLL ? 0xffffffffu : static_cast<unsigned int>(b);
    };
    // This pair's 32 keys; rows past seqlen_k read as 0 and are dropped on store.
    const int row_w  = kv0 + pr * PR;
    const int rows_w = seqlen_k - row_w < PR ? (seqlen_k - row_w > 0 ? seqlen_k - row_w : 0) : PR;
    // The tiles this pair sees in full (no mask: every key and query valid, the tile's first query
    // at or past its last key) form one range [i_full0, i_full1).
    int i_full0 = T, i_full1 = T;
    if(row_w + PR <= seqlen_k)
    {
        const int need = row_w + PR - 1 - causal_offset;
        i_full0        = k.causal && need > 0 ? (need + BQ - 1) / BQ : 0;
        i_full1        = seqlen_q / BQ;
        i_full1        = i_full1 < i_full0 ? i_full0 : i_full1;
    }

    // DMA of a query tile (gated_mla_bwd::dma16, dma4): 13 Q and 9 dO loads of 1 KB, then one
    // 256-byte load each of LSE and D. Load j of Q or dO writes its 16-byte slots 64 j .. + 63 of
    // the padded tile, slot s being row s / SPR, chunk s % SPR (the last two chunks padding).
    // The S waves issue 16-byte loads w + NP u in JS rounds, the dP waves 16-byte loads
    // NP JS + w - NP + NP u in JP - 1 rounds and LSE or D in their last. Loads past the tile's
    // repeat the wave's previous one, the same bytes to the same place, so that no round takes a
    // branch. Padding, queries past seqlen_q and lanes past the tile read out of bounds, as 0.
    constexpr int SPR_Q = LD_Q * 2 / 16, SPR_D = LD_DO * 2 / 16;
    constexpr int J_Q = BQ * SPR_Q / 64, J_D = BQ * SPR_D / 64, JS = 4, JP = 3, JU = JS;
    static_assert(BQ * SPR_Q % 64 == 0 && BQ * SPR_D % 64 == 0, "whole loads per tensor");
    static_assert(NP * (JS + JP - 1) >= J_Q + J_D && NP * (JS + JP - 2) < J_Q + J_D && NP >= 2, "the load split");
    // MFMA steps between a wave's rounds: back to back, each m0 write waits for the previous load
    // to leave the CU's queue, which the other SIMDs' rounds fill at the same time.
    constexpr int SP_S = 2, SP_P = 3;
    constexpr int OOB = 1 << 30;  // byte offset past any tile
    const char* q_src  = reinterpret_cast<const char*>(reinterpret_cast<const bf16_t*>(k.ptr_q) +
                                                      static_cast<long long>(q0) * k.stride_q_n +
                                                      static_cast<long long>(h) * k.stride_q_h);
    const char* do_src = reinterpret_cast<const char*>(reinterpret_cast<const bf16_t*>(k.ptr_do) +
                                                       static_cast<long long>(q0) * k.stride_do_n +
                                                       static_cast<long long>(h) * k.stride_do_h);
    const char* l_src  = reinterpret_cast<const char*>(k.ptr_lse + static_cast<long long>(h) * k.stride_lse_h + q0);
    const char* d_src  = reinterpret_cast<const char*>(k.ptr_d + static_cast<long long>(h) * k.stride_lse_h + q0);
    // Per load round u, wave-uniform: buffer descriptor of the next tile to load (its base and
    // size), row bytes, whether a 4-byte load, place in the buffer and m0 for the next tile; and
    // this lane's source offset within the tile.
    gated_mla_bwd::i32x4_t d_rs[JU];
    int d_rb[JU], d_lds[JU], d_os[JU];
    unsigned int d_m0[JU];
#pragma unroll
    for(int u = 0; u < JU; u++)
    {
        int jl = s_wave ? w + NP * u : NP * JS + w - NP + NP * u;
        if(jl >= J_Q + J_D)
            jl -= NP;
        const int reg = !s_wave && u == JP - 1 ? 2 + (w - NP) % 2 : (jl < J_Q ? 0 : 1);
        const int jj  = reg == 0 ? jl : jl - J_Q;
        d_rb[u]       = reg == 0 ? k.stride_q_n * 2 : (reg == 1 ? k.stride_do_n * 2 : 4);
        d_rs[u]       = gated_mla_bwd::dma_rsrc((reg == 0 ? q_src : (reg == 1 ? do_src : (reg == 2 ? l_src : d_src))) +
                                                    static_cast<long long>(i0) * BQ * d_rb[u],
                                                0);
        d_lds[u] = reg == 0 ? OFF_Q * 2 + jj * 1024 : (reg == 1 ? OFF_DO * 2 + jj * 1024 : (reg == 2 ? OFF_LSE : OFF_D) * 2);
        const int spr = reg == 0 ? SPR_Q : SPR_D, dch = reg == 0 ? D_QK / 8 : D_V / 8;
        const int sl = 64 * jj + lane, c = sl % spr;
        d_os[u]      = reg >= 2 ? lane * 4 : (c < dch ? sl / spr * d_rb[u] + c * 16 : OOB);
    }
    // Query rows of tile t, in SALU: the compiler would clamp with v_med3 and move the result back
    // with a readfirstlane, which waits behind the other wave's VALU work.
    auto dma_rows = [&](int t) {
        int r = seqlen_q - t * BQ;
        asm("s_min_i32 %0, %0, %1\n\ts_max_i32 %0, %0, 0" : "+s"(r) : "n"(BQ) : "scc");
        return r;
    };
    // Buffers go round in tile order; b is a tile's buffer, named by its offset in smem (elements):
    // an index would take an s_mul_i32 for every address made from it, and a scalar multiply in
    // one wave holds back the MFMA and VALU issue of the SIMD's other wave (see the dQ pass's
    // phased loop). In SALU: the compiler would step back with a VALU subtract and a readfirstlane.
    auto bnext = [](int b) {
        int r;
        asm("s_add_i32 %0, %1, %2\n\ts_cmp_eq_u32 %0, %3\n\ts_cselect_b32 %0, 0, %0"
            : "=&s"(r)
            : "s"(b), "n"(SZ_BUF), "n"(NBUF * SZ_BUF)
            : "scc");
        return r;
    };
    auto bprev = [](int b) {
        int r;
        asm("s_add_i32 %0, %1, %2\n\ts_cmp_lt_i32 %0, 0\n\ts_cselect_b32 %0, %3, %0"
            : "=&s"(r)
            : "s"(b), "n"(-SZ_BUF), "n"((NBUF - 1) * SZ_BUF)
            : "scc");
        return r;
    };
    auto dma_lds = [&](int b) { return gated_mla_bwd::lds_addr(smem + b); };
    // Point this wave's rounds (dp_c: the dP waves' JP, else JS) at tile t in buffer b (`adv`:
    // from tile t - 1, whose base they hold). Tiles go in order, so the base just advances; tiles
    // past the last one fill a buffer no one reads anymore, with zeros, as do queries past
    // seqlen_q. Made in the wave's VALU stage and pinned, so that a round in its MFMA stage is
    // only m0 and the load.
    auto dma_next = [&](int t, int b, bool adv, auto dp_c) {
        constexpr int n       = decltype(dp_c)::value ? JP : JS;
        const int rows        = dma_rows(t);
        const unsigned int lb = dma_lds(b);
        // The sizes: a full tile's by a shift; only the sequence's last, partial tile and the
        // tiles past it multiply.
        int part = rows - BQ;
        asm volatile("" : "+s"(part));
        int nbs[JU];
        if(__builtin_expect(part == 0, 1))
        {
#pragma unroll
            for(int u = 0; u < n; u++)
                nbs[u] = BQ * d_rb[u];
        }
        else
        {
#pragma unroll
            for(int u = 0; u < n; u++)
                asm("s_mul_i32 %0, %1, %2" : "=s"(nbs[u]) : "s"(rows), "s"(d_rb[u]));
        }
#pragma unroll
        for(int u = 0; u < n; u++)
        {
            auto& r = d_rs[u];
            unsigned long long p =
                static_cast<unsigned int>(r[0]) | static_cast<unsigned long long>(static_cast<unsigned int>(r[1])) << 32;
            if(adv)
                p += static_cast<unsigned long long>(static_cast<unsigned int>(BQ * d_rb[u]));
            const int nb = nbs[u];
            r       = gated_mla_bwd::i32x4_t{static_cast<int>(p), static_cast<int>(p >> 32), nb, 0x00020000};
            d_m0[u] = lb + d_lds[u];
            asm volatile("" : "+s"(r), "+s"(d_m0[u]));
        }
    };
    // The role is a type (dp_c) and u is a constant once unrolled, so no round takes a branch.
    auto dma_round = [&](int u, auto dp_c) {
        if(decltype(dp_c)::value && u == JP - 1)
            gated_mla_bwd::dma4(d_rs[u], d_os[u], d_m0[u]);
        else
            gated_mla_bwd::dma16(d_rs[u], d_os[u], d_m0[u]);
    };
    auto dma = [&](auto dp_c) {
        constexpr int n = decltype(dp_c)::value ? JP : JS;
#pragma unroll
        for(int u = 0; u < n; u++)
            dma_round(u, dp_c);
    };
    // The DMA writes land in LDS when vmcnt says so, which s_barrier does not wait for.
    auto dma_wait  = [] { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); };
    auto stage_end = [] {
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    };
    // sched_barrier only fences one basic block: values a stage produces are pinned at its end (an
    // empty asm that reads and writes them), so the compiler cannot sink their computation into
    // the next stage and keep the inputs live across it.
    auto pin = [](auto& x) { asm volatile("" : "+v"(x)); };
    // Each wave's loop: the tile index goes through an empty asm and the loop test is made before
    // the barrier that closes a tile, so that the compiler derives no counters from it to update
    // and test after that barrier, in front of the next stage.
    auto next_tile = [&](int t) {
        int tn = t + 1;
        asm volatile("" : "+s"(tn));
        return tn;
    };

    // Query tile fragments are read at a per-lane LDS address plus immediates. The address of an
    // operand in a tile's buffer, one VGPR add, is made in the wave's VALU stage before the MFMA
    // stage that reads it and pinned there: in an MFMA stage a lone VALU instruction waits behind
    // the other wave's VALU work. Layouts with lane coordinate 0 give the immediates.
    auto lay_b     = [&](auto ld, const auto& c) {
        return opus::partition_layout_b<8>(mma_k1, opus::make_tuple(ld, 1_I), c);
    };
    const auto c_0 = opus::make_tuple(0_I, 0_I, 0_I, 0_I);
    auto lane_b    = [&](auto ld) {
        return opus::layout_to_offsets<8>(lay_b(ld, c_ab))[0] - opus::layout_to_offsets<8>(lay_b(ld, c_0))[0];
    };
    auto addr = [&](int b, int off, int o) {
        unsigned int a = gated_mla_bwd::lds_addr(smem + b + off) + o * 2;
        pin(a);
        return a;
    };
    auto at = [](unsigned int a) { return opus::make_smem(reinterpret_cast<bf16_t*>(static_cast<__UINTPTR_TYPE__>(a))); };
    // C element e = (m NT + tn) 4 + p of S^T / dP^T is key 16 m + lane % 16 of the pair and query
    // 16 tn + 4 (lane / 16) + p of the tile, so each N tile is one float4 run of LSE or D, and P
    // goes to the dP wave in that layout, chunk-major so that each float4 access is lane-contiguous.
    auto ld_runs = [&](int b, int off, float mul, float(&out)[NS]) {
        const float* f = reinterpret_cast<const float*>(smem + b + off);
        f32x4_t run[NT];
#pragma unroll
        for(int tn = 0; tn < NT; tn++)
            run[tn] = *reinterpret_cast<const f32x4_t*>(f + 16 * tn + 4 * (lane / 16));
#pragma unroll
        for(int e = 0; e < NS; e++)
            out[e] = run[(e / 4) % NT][e % 4] * mul;
    };
    float* s_p = reinterpret_cast<float*>(smem + OFF_P) + pr * PR * BQ;

    // B fragments of LA steps in flight, in a ring of LA + 1 slots: a step's reads (of step + LA)
    // go to the slot step - 1 consumed, so that they can issue between the step's own MFMAs, one
    // after each: an MFMA leaves 8 of its 16 cycles to issue other instructions, as much as one
    // ds_read_b64_tr_b16 or two SALU instructions take.
    constexpr int LA = 2, RS = LA + 1;
    frag_t ring[RS];
    auto c_stage = [&](auto n_steps, auto ld, auto ex) {
        constexpr int N = decltype(n_steps)::value;
#pragma unroll
        for(int st = 0; st < N; st++)
        {
            ex(st, ring[st % RS]);
            if(st + LA < N)
                ring[(st + LA) % RS] = ld(st + LA);
#pragma unroll
            for(int m = 0; m < MR * NT; m++)
            {
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);  // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);  // DS read
                __builtin_amdgcn_sched_group_barrier(0x004, 2, 0);  // SALU
            }
            __builtin_amdgcn_sched_barrier(0);
        }
    };
    auto prefetch = [&](auto ld) {
#pragma unroll
        for(int st = 0; st < LA; st++)
            ring[st] = ld(st);
    };
    auto pin_ring = [&] {
#pragma unroll
        for(int r = 0; r < LA; r++)
            pin(ring[r]);
    };
    const acc_t zero{};
    const float scale_log2 = k.softmax_scale * LOG2_E;
    // This pair's rows of dkv [.., H, dK nope | dV]; rows past seqlen_k fall outside the
    // descriptors and are dropped.
    auto dkv_part = [&](int col0) {
        bf16_t* dkv = reinterpret_cast<bf16_t*>(k.ptr_dkv) + static_cast<long long>(k0 + row_w) * k.stride_dkv_n +
                      static_cast<long long>(h) * k.stride_dkv_h + col0;
        return opus::make_gmem(dkv, rec(static_cast<long long>(rows_w) * k.stride_dkv_n - col0, 2));
    };
    const auto u_dkv = opus::partition_layout_c(mma_k1, opus::make_tuple(k.stride_dkv_n, 1_I), c_c);
    // The first tiles in LDS (the dP waves' shares up to i0 + 2, as they issue tile t + 3 from
    // B(t)) and this wave's K or V in registers before the loop (0x0f70 is s_waitcnt vmcnt(0)):
    // otherwise the compiler waits for the latter inside the loop with vmcnt counts that the DMA
    // then makes block. The S waves' rounds are then pointed at tile i0 + 2 for A(i0).
    const int b0 = i0 % NBUF * SZ_BUF;
    auto prologue = [&](auto dp_c) {
        constexpr int n = decltype(dp_c)::value ? 3 : 2;
        int b           = b0;
#pragma unroll
        for(int x = 0; x < n; x++, b = bnext(b))
        {
            dma_next(i0 + x, b, x > 0, dp_c);
            if(i0 < T)
                dma(dp_c);
        }
#if !GMLA_PAIR_PP
        if(!decltype(dp_c)::value)
            dma_next(i0 + n, b, true, dp_c);
#endif
        __builtin_amdgcn_s_waitcnt(0x0f70);
        dma_wait();
        stage_end();
    };

#if GMLA_PAIR_PP
    // One MFMA of a step, M tile m against N tile n (mma_k1 runs the four in one call, which
    // leaves their order, and what goes between them, to the compiler): for the phases that put
    // one pack to bf16 behind each MFMA. Left to the compiler, the eight packs come after the
    // sixth MFMA, into registers it had lent to the step's operands.
    auto mma_11 = make_mma(number<1>{}, number<1>{}, number<1>{});
    static_assert(decltype(mma_11)::tile_a_len == 8 && decltype(mma_11)::tile_c_len == 4 && NS == 16 && MR == 2);
    auto mfma1 = [&](const afrag_t& fa, const frag_t& fb, acc_t& c, int m, int n, bool first = false) {
        opus::vector_t<bf16_t, 8> a1;
        bvec<8> b1;
        typename decltype(mma_11)::vtype_c c1{};
#pragma unroll
        for(int x = 0; x < 8; x++)
        {
            a1[x] = fa[m * 8 + x];
            b1[x] = fb[n * 8 + x];
        }
        if(!first)
        {
#pragma unroll
            for(int p_ = 0; p_ < 4; p_++)
                c1[p_] = c[(m * NT + n) * 4 + p_];
        }
        c1 = mma_11(a1, b1, c1);
        // The result through an empty asm: the MFMA is no later than here. Without it the compiler
        // may put a phase's MFMAs after all of its other work, at the pin that ends the phase (ATT).
        asm volatile("" : "+v"(c1));
#pragma unroll
        for(int p_ = 0; p_ < 4; p_++)
            c[(m * NT + n) * 4 + p_] = c1[p_];
    };
    // A compute phase as 8 units: MFMA i (mf), then what goes behind it (f: at most two VALU
    // instructions and a few scalar ones), fenced so that it stays behind that MFMA.
    auto units = [&](auto mf, auto f) {
#pragma unroll
        for(int i = 0; i < 8; i++)
        {
            mf(i);
            f(i);
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
            __builtin_amdgcn_sched_group_barrier(0x002, 2, 0);
            __builtin_amdgcn_sched_barrier(0);
        }
    };
    // The float4 runs of LSE or D (ld_runs) from an address made a phase before: a lone VALU
    // instruction in a load phase waits behind the other wave's, and holds it back.
    auto at_f      = [](unsigned int a) { return opus::make_smem(reinterpret_cast<float*>(static_cast<__UINTPTR_TYPE__>(a))); };
    const int o_run = 8 * (lane / 16);
    // addr from this lane's byte offset, the scalar part in an SGPR: one v_add_u32. Left to the
    // compiler it is a v_lshl_add_u32, which waits for the other wave's scalar run to end (ATT).
    auto addrb = [&](int b, int off, int ob) {
        unsigned int sb = gated_mla_bwd::lds_addr(smem + b + off);
        asm volatile("" : "+s"(sb));
        unsigned int a = sb + static_cast<unsigned int>(ob);
        pin(a);
        return a;
    };
#endif
    if(s_wave)
    {
        auto g_kn = opus::make_gmem(reinterpret_cast<const bf16_t*>(k.ptr_k_nope) +
                                        static_cast<long long>(k0 + row_w) * k.stride_k_n +
                                        static_cast<long long>(h) * k.stride_k_h,
                                    rec(static_cast<long long>(rows_w) * k.stride_k_n, 2));
        auto g_kr = opus::make_gmem(reinterpret_cast<const bf16_t*>(k.ptr_k_rot) +
                                        static_cast<long long>(k0 + row_w) * k.stride_k_rot_n,
                                    rec(static_cast<long long>(rows_w) * k.stride_k_rot_n, 2));
        const auto v_kn = g_kn.template load<8>(opus::partition_layout_a<8>(mma_kn, opus::make_tuple(k.stride_k_n, 1_I), c_ab));
        const auto v_kr = g_kr.template load<8>(opus::partition_layout_a<8>(mma_kr, opus::make_tuple(k.stride_k_rot_n, 1_I), c_ab));
        // A of S^T per K step: [M tile][8], nope steps then rope.
        afrag_t a_k[KS];
#pragma unroll
        for(int ks = 0; ks < KS; ks++)
#pragma unroll
            for(int m = 0; m < MR; m++)
#pragma unroll
                for(int x = 0; x < 8; x++)
                    a_k[ks][m * 8 + x] = ks < KS_N ? v_kn[(m * KS_N + ks) * 8 + x]
                                                   : v_kr[(m * KS_R + (ks < KS_N ? 0 : ks - KS_N)) * 8 + x];
        const int c_kv = row_w + lane % 16, c_q = 4 * (lane / 16);

        acc_t v_s, v_dv[NV];
#pragma unroll
        for(int c = 0; c < NV; c++)
            opus::clear(v_dv[c]);
        // P^T(t - 1), the A operand of dV(t - 1): zero before the first tile, whose dV(t - 1) steps
        // then add nothing (they read the first tile's dO, which is finite).
        afrag_t p_bf;
        opus::clear(p_bf);
        unsigned int a_q, a_tdo;  // S^T(t): Q; dV(t - 1): dO, transposed
        const int o_q = lane_b(number<LD_Q>{}), o_tdo = gated_mla_bwd::tr_c_order_lane<LD_DO>(lane);
        const auto u_q0 = lay_b(number<LD_Q>{}, c_0);
        auto ld_q   = [&](int ks) -> frag_t { return at(a_q).template load<8>(u_q0, ks * 32); };
        auto ld_tdo = [&](int c) -> frag_t { return gated_mla_bwd::tr_load_b_c_order_at<2, LD_DO>(at(a_tdo), 32 * c); };
        auto ex_s   = [&](int ks, const frag_t& b) { v_s = mma_k1(a_k[ks], b, ks == 0 ? zero : v_s); };
        auto ex_dv  = [&](int c, const frag_t& b) { v_dv[c] = mma_k1(p_bf, b, v_dv[c]); };

#if GMLA_PAIR_PP
        // Load and compute phases, as the dQ pass has them (GMLA_DQ_PP). A tile is, per wave, 5 pairs
        // of a load phase and a compute phase, each closed by a workgroup barrier:
        //   load     the B fragments of the next 8 MFMAs (two steps) into registers and one more
        //            thing: a float4 of P to or from LDS, a DMA round
        //   compute  those 8 MFMAs, with this wave's own VALU work in the 8 of an MFMA's 16 cycles
        //            that take other vector instructions, at most two behind each
        // The dP waves run one phase behind the S waves of their SIMDs: a wave reads LDS while the
        // other one computes. Per tile t of the S waves, phase by phase (S wave | dP wave):
        //   load  Q(t) steps 0, 1; P(t - 1) chunk 0 out       | dP^T(t - 1) steps 2, 3    dS^T(t - 2) to bf16, 2nd half
        //   S^T(t) steps 0, 1    exp, P(t - 1) 2nd half        | load Q^T(t - 2) chunks 0, 1; P(t - 1) chunk 0 in
        //   load  Q(t) steps 2, 3; chunk 1 out                 | dK(t - 2) chunks 0, 1     (dP - D)(t - 1)
        //   S^T(t) steps 2, 3    P^T(t - 1) to bf16            | load Q^T(t - 2) chunks 2, 3; chunk 1 in
        //   load  Q(t) steps 4, 5, LSE(t); chunk 2 out         | dK(t - 2) chunks 2, 3     dS(t - 1) chunks 0, 1
        //   S^T(t) steps 4, 5    LSE log2 e, LDS addresses     | load Q^T(t - 2) chunks 4, 5; chunk 2 in
        //   load  dO^T(t - 1) chunks 0, 1; chunk 3 out         | dK(t - 2) chunks 4, 5     dS(t - 1) chunk 2
        //   dV(t - 1) chunks 0, 1    S^T(t) scale - LSE        | load dO(t) steps 0, 1; chunk 3 in
        //   load  dO^T(t - 1) chunks 2, 3                      | dP^T(t) steps 0, 1        dS(t - 1) chunk 3, dS^T(t - 1) to bf16, 1st half
        //   dV(t - 1) chunks 2, 3    exp, P(t) 1st half        | load dO(t) steps 2, 3, D(t)
        // P goes over a float4 a phase: a ds_write_b128 holds its wave for some 60 cycles, which a
        // load phase of four reads has to spare and one of eight transposed reads, or two writes,
        // has not (ATT). dV runs a tile behind S^T and dK two, so that P, dS and their bf16 forms
        // are made behind MFMAs.
        // DMA is the dP waves' alone (see their part): the S wave's phases are full on both sides
        // (48 VALU instructions behind its 40 MFMAs; its reads and P's writes), theirs are not.
        acc_t px;  // P on its way from S^T scale - LSE through exp, beside the next tile's S^T
#pragma unroll
        for(int e = 0; e < NS; e++)
            px[e] = e < NS / 2 ? 0.0f : -1e30f;  // P(i0 - 1) = 0: its halves after and before exp
        pin(px);
        frag_t op[2];
        // A load phase ends without waiting for its reads (see the dQ pass).
        auto l_end = [&] {
            asm volatile("" ::: "memory");
            stage_end();
        };
        auto write_px = [&](int c) {
            *reinterpret_cast<f32x4_t*>(s_p + (c * 64 + lane) * 4) =
                f32x4_t{px[c * 4], px[c * 4 + 1], px[c * 4 + 2], px[c * 4 + 3]};
        };
        // The float4s of P that load phase ph sends (GMLA_X_PWPOS), right behind its reads (early,
        // GMLA_X_PWMID 1) or at its end.
        auto wr_px = [&](int ph, bool early) {
            if(!GMLA_X_PW || early != (GMLA_X_PWMID != 0))
                return;
#pragma unroll
            for(int c = 0; c < NS / 4; c++)
            {
                const int at = GMLA_X_PWPOS == 0 ? c + 1
                               : (GMLA_X_PWPOS == 1 ? (c < 3 ? c + 1 : 3) : (GMLA_X_PWPOS == 2 ? (c < 2 ? c + 1 : c) : (c < 3 ? c + 1 : 5)));
                if(at == ph)
                    write_px(c);
            }
        };
        unsigned int a_lse;
        opus::vector_t<float, 4> lz{0.0f, 0.0f, 0.0f, 0.0f};  // GMLA_X_LSE 0: stands in for LSE
        pin(lz);
        int ob_q = o_q * 2, ob_tdo = o_tdo * 2, ob_lse = o_run * 2;
        pin(ob_q);
        pin(ob_tdo);
        pin(ob_lse);
        // This wave's K in registers (0x0f70 is s_waitcnt vmcnt(0)); the first tiles are in LDS
        // past the barrier, which the dP waves reach when their loads have landed.
        __builtin_amdgcn_s_waitcnt(0x0f70);
        stage_end();
        int n_full = i_full1 - i_full0;
        asm volatile("" : "+s"(n_full));
        if(i0 < T)
        {
            a_q   = addrb(b0, OFF_Q, ob_q);
            a_tdo = addrb(b0, OFF_DO, ob_tdo);
            a_lse = addrb(b0, OFF_LSE, ob_lse);
            int bt = b0, bp = b0;  // tile t's buffer, tile t - 1's (tile t's for the first)
            // The loop closes with one conditional branch: the buffers move on before the test.
            int t = i0;
            bool more;
            do
            {
                const int qt0 = t * BQ;
                // 1: S^T(t) steps 0, 1; behind them the second half of P(t - 1)'s exp. P(t - 1)
                // chunk 0.
                op[0] = ld_q(0);
                op[1] = ld_q(1);
                __builtin_amdgcn_sched_barrier(0);
                wr_px(1, true);
                const int b1 = bnext(bt);
                wr_px(1, false);
                l_end();
                ex_s(0, op[0]);
                ex_s(1, op[1]);
#pragma unroll
                for(int e = NS / 2; e < NS; e++)
                    px[e] = __builtin_amdgcn_exp2f(px[e]);
                GMLA_SCHED8(0x400, 1)
                pin(v_s);
                pin(px);
                stage_end();
                // 2: S^T(t) steps 2, 3; behind each MFMA one pack of P^T(t - 1) to bf16 (C -> A: the
                // K order of this fragment is what tr_load_b_c_order supplies). P(t - 1) chunk 1.
                op[0] = ld_q(2);
                op[1] = ld_q(3);
                __builtin_amdgcn_sched_barrier(0);
                wr_px(2, true);
                wr_px(2, false);
                l_end();
                units([&](int i) { mfma1(a_k[2 + i / 4], op[i / 4], v_s, i / 2 % 2, i % 2); },
                      [&](int i) {
                          const opus::vector_t<float, 2> f2{px[2 * i], px[2 * i + 1]};
                          auto h2 = opus::cast<bf16_t>(f2);
                          pin(h2);  // or the packs sink below the last MFMA
                          p_bf[2 * i]     = h2[0];
                          p_bf[2 * i + 1] = h2[1];
                      });
                pin(v_s);
                pin(p_bf);
                stage_end();
                // 3: S^T(t) steps 4, 5. LSE(t). Behind the MFMAs, LSE log2 e and the LDS addresses
                // of dO of tile t - 1 (dV) and of Q and LSE of tile t + 1. P(t - 1) chunk 2.
                op[0] = ld_q(4);
                op[1] = ld_q(5);
                opus::vector_t<float, 4> lse_r[NT];
#pragma unroll
                for(int n = 0; n < NT; n++)
                    lse_r[n] = GMLA_X_LSE ? at_f(a_lse).template load<4>(16 * n) : lz;
                __builtin_amdgcn_sched_barrier(0);
                wr_px(3, true);
                wr_px(3, false);
                l_end();
                ex_s(4, op[0]);
                ex_s(5, op[1]);
                opus::vector_t<float, 4 * NT> lse2;
#pragma unroll
                for(int i = 0; i < 4 * NT; i++)
                    lse2[i] = lse_r[i / 4][i % 4] * LOG2_E;
                a_tdo = addrb(bp, OFF_DO, ob_tdo);
                a_q   = addrb(b1, OFF_Q, ob_q);
                a_lse = addrb(b1, OFF_LSE, ob_lse);
                GMLA_SCHED8(0x002, 2)
                pin(v_s);
                pin(lse2);
                stage_end();
                // 4: dV(t - 1) chunks 0, 1; behind them S^T(t) scale - LSE, into P's registers
                // (P(t - 1)'s last chunk goes out before).
                op[0] = ld_tdo(0);
                op[1] = ld_tdo(1);
                __builtin_amdgcn_sched_barrier(0);
                wr_px(4, true);
                wr_px(4, false);
                l_end();
                ex_dv(0, op[0]);
                ex_dv(1, op[1]);
#pragma unroll
                for(int e = 0; e < NS; e++)
                    px[e] = v_s[e] * scale_log2 - lse2[e / 4 % NT * 4 + e % 4];
                GMLA_SCHED8(0x002, 2)
                pin(px);
                stage_end();
                // 5: dV(t - 1) chunks 2, 3; behind them the first half of P(t). A tile this pair does
                // not see in full is masked before its exp (as in the main pass); the test is a
                // scalar compare whose branch is not taken. The loop test, before the barriers.
                op[0] = ld_tdo(2);
                op[1] = ld_tdo(3);
                __builtin_amdgcn_sched_barrier(0);
                wr_px(5, true);
                {
                    int fd = t - i_full0;
                    asm volatile("" : "+s"(fd));
                    if(__builtin_expect(static_cast<unsigned int>(fd) >= static_cast<unsigned int>(n_full), 0))
                    {
#pragma unroll
                        for(int e = 0; e < NS; e++)
                        {
                            const int kv = c_kv + 16 * (e / (4 * NT)), qp = qt0 + c_q + 16 * (e / 4 % NT) + e % 4;
                            const bool ok = (kv < seqlen_k) & (qp < seqlen_q) & (!k.causal | (qp + causal_offset >= kv));
                            px[e]         = ok ? px[e] : -1e30f;
                        }
                        pin(px);
                    }
                }
                const int tn = next_tile(t);
                more         = tn < T;
                wr_px(5, false);
                l_end();
                ex_dv(2, op[0]);
                ex_dv(3, op[1]);
#pragma unroll
                for(int e = 0; e < NS / 2; e++)
                    px[e] = __builtin_amdgcn_exp2f(px[e]);
                GMLA_SCHED8(0x400, 1)
                pin(px);
                stage_end();
                t  = tn;
                bp = bt;
                bt = b1;
                if(GMLA_X_LOOPPIN)
                    asm volatile("" : "+s"(t), "+s"(bp), "+s"(bt));
            } while(__builtin_expect(more, 1));
            // Tail: P of the last tile (the second half of its exp; whole to LDS, for the dP wave's
            // next phase, and to bf16), then its dV.
#pragma unroll
            for(int e = NS / 2; e < NS; e++)
                px[e] = __builtin_amdgcn_exp2f(px[e]);
#pragma unroll
            for(int c = 0; c < NS / 4; c++)
                write_px(c);
            p_bf  = opus::cast<bf16_t>(px);
            a_tdo = addrb(bp, OFF_DO, ob_tdo);
#pragma unroll
            for(int c = 0; c < NV; c += 2)
            {
                op[0] = ld_tdo(c);
                op[1] = ld_tdo(c + 1);
                l_end();
                ex_dv(c, op[0]);
                ex_dv(c + 1, op[1]);
                stage_end();
            }
            // The dP waves end 12 phases after this wave's loop.
#pragma unroll
            for(int x = 0; x < 12 - NV; x++)
                stage_end();
        }
        // The dP waves run one phase behind.
        stage_end();
#else
        prologue(std::false_type{});
        int n_full = i_full1 - i_full0;
        asm volatile("" : "+s"(n_full));
        if(i0 < T)
        {
            a_q   = addr(b0, OFF_Q, o_q);
            a_tdo = addr(b0, OFF_DO, o_tdo);
            prefetch(ld_q);
            int bt = b0;  // tile t's buffer
            for(int t = i0;;)
            {
                const int qt0 = t * BQ;
                // A(t): S^T(t), then dV(t - 1). Steps 1, 1 + SP_S, .. also issue this wave's share
                // of tile t + 2 (set up in B(t - 1)), a round each, beside their MFMAs; its share
                // of tile t + 1, issued a tile ago, is waited for first, before B(t) reads it.
                // Buffer (t + 2) % NBUF last held tile t - 3.
                static_assert(1 + SP_S * (JS - 1) < KS + NV, "every round in the stage");
                c_stage(number<KS + NV>{},
                        [&](int st) { return st < KS ? ld_q(st) : ld_tdo(st - KS); },
                        [&](int st, const frag_t& b) {
                            if(st < KS)
                                ex_s(st, b);
                            else
                                ex_dv(st - KS, b);
                            if(st == 1)
                                dma_wait();
                            if(st >= 1 && (st - 1) % SP_S == 0 && (st - 1) / SP_S < JS)
                                dma_round((st - 1) / SP_S, std::false_type{});
                        });
                pin(v_s);
                stage_end();
                // B(t): LSE, read ahead of the next A stage's first steps (S^T(t + 1); past the
                // last tile they read a buffer the tail does not use) so as not to wait behind
                // them; P(t), as in the main pass: tiles this pair sees in full take an unmasked exp.
                // Also this wave's rounds of tile t + 3, in buffer (t + 3) % NBUF = (t - 2) % NBUF.
                const int b1 = bnext(bt);
                float lse_e[NS];
                ld_runs(bt, OFF_LSE, LOG2_E, lse_e);
                a_tdo = addr(bt, OFF_DO, o_tdo);
                a_q   = addr(b1, OFF_Q, o_q);
                prefetch(ld_q);
#if !GMLA_X_PFIRST
                dma_next(t + 3, bprev(bprev(bt)), true, std::false_type{});
#endif
                // Each float4 of P goes to LDS as soon as its exps are done, the next exps issuing
                // while it waits in the LDS queue: four writes back to back wait there in turn.
                auto write_p = [&](int c) {
                    *reinterpret_cast<f32x4_t*>(s_p + (c * 64 + lane) * 4) =
                        f32x4_t{v_s[c * 4], v_s[c * 4 + 1], v_s[c * 4 + 2], v_s[c * 4 + 3]};
                    __builtin_amdgcn_sched_barrier(0);
                };
                // A scalar compare and a branch on SCC: the compiler's own form goes through a
                // lane mask and VCC, which holds back the other wave as a scalar multiply does.
                int fd = t - i_full0;
                asm volatile("" : "+s"(fd));
                if(__builtin_expect(static_cast<unsigned int>(fd) < static_cast<unsigned int>(n_full), 1))
                {
#pragma unroll
                    for(int c = 0; c < NS / 4; c++)
                    {
#pragma unroll
                        for(int e = c * 4; e < c * 4 + 4; e++)
                            v_s[e] = __builtin_amdgcn_exp2f(v_s[e] * scale_log2 - lse_e[e]);
                        write_p(c);
                    }
                }
                else
                {
#pragma unroll
                    for(int c = 0; c < NS / 4; c++)
                    {
#pragma unroll
                        for(int e = c * 4; e < c * 4 + 4; e++)
                        {
                            const int kv = c_kv + 16 * (e / (4 * NT)), qp = qt0 + c_q + 16 * (e / 4 % NT) + e % 4;
                            const bool ok = (kv < seqlen_k) & (qp < seqlen_q) & (!k.causal | (qp + causal_offset >= kv));
                            const float x = v_s[e] * scale_log2 - lse_e[e];
                            v_s[e]        = __builtin_amdgcn_exp2f(ok ? x : -1e30f);
                        }
                        write_p(c);
                    }
                }
                // C -> A: the K order of this fragment is what tr_load_b_c_order supplies.
                p_bf = opus::cast<bf16_t>(v_s);
                pin(p_bf);
#if GMLA_X_PFIRST
                // The scalar work last: P's VALU work then runs beside the dP wave's dP^T
                // steps, which read LDS half as often as its dK steps.
                __builtin_amdgcn_sched_barrier(0);
                dma_next(t + 3, bprev(bprev(bt)), true, std::false_type{});
#endif
                pin_ring();
                const int tn    = next_tile(t);
                const bool more = tn < T;
                stage_end();
                if(!more)
                    break;
                t  = tn;
                bt = b1;
            }
            // A': dV of the last tile. B': the dP waves' dK of it.
            prefetch(ld_tdo);
            c_stage(number<NV>{}, ld_tdo, ex_dv);
            stage_end();
            stage_end();
        }
#endif

        auto g_dv = dkv_part(D_NOPE);
#pragma unroll
        for(int c = 0; c < NV; c++)
            g_dv.template store<4>(opus::cast<bf16_t>(v_dv[c]), u_dkv, c * 32);
    }
    else
    {
        auto g_v = opus::make_gmem(reinterpret_cast<const bf16_t*>(k.ptr_v) +
                                       static_cast<long long>(k0 + row_w) * k.stride_v_n +
                                       static_cast<long long>(h) * k.stride_v_h,
                                   rec(static_cast<long long>(rows_w) * k.stride_v_n, 2));
        const auto v_vw = g_v.template load<8>(opus::partition_layout_a<8>(mma_v, opus::make_tuple(k.stride_v_n, 1_I), c_ab));
        auto a_v = [&](int ks) {
            afrag_t a;
#pragma unroll
            for(int m = 0; m < MR; m++)
#pragma unroll
                for(int x = 0; x < 8; x++)
                    a[m * 8 + x] = v_vw[(m * KS_P + ks) * 8 + x];
            return a;
        };

        acc_t v_dp, v_dk[NK];
#pragma unroll
        for(int c = 0; c < NK; c++)
            opus::clear(v_dk[c]);
        // dS^T(t - 1), the A operand of dK(t - 1): zero before the first tile, as P^T in the S wave.
        afrag_t a_ds;
        opus::clear(a_ds);
        unsigned int a_do, a_tq;  // dP^T(t): dO; dK(t - 1): Q, transposed
        const int o_do = lane_b(number<LD_DO>{}), o_tq = gated_mla_bwd::tr_c_order_lane<LD_Q>(lane);
        const auto u_do0 = lay_b(number<LD_DO>{}, c_0);
        auto ld_do = [&](int ks) -> frag_t { return at(a_do).template load<8>(u_do0, ks * 32); };
        auto ld_tq = [&](int c) -> frag_t { return gated_mla_bwd::tr_load_b_c_order_at<2, LD_Q>(at(a_tq), 32 * c); };
        auto ex_dp = [&](int ks, const frag_t& b) { v_dp = mma_k1(a_v(ks), b, ks == 0 ? zero : v_dp); };
        auto ex_dk = [&](int c, const frag_t& b) { v_dk[c] = mma_k1(a_ds, b, v_dk[c]); };
        // dS^T = P (dP^T - D) of the tile in buffer b, in bf16 (softmax_scale goes onto dK at the
        // end); `then` issues the next stage's first reads after these.
        auto ds = [&](int b, auto then) {
            float d_e[NS];
            ld_runs(b, OFF_D, 1.0f, d_e);
            f32x4_t p[NS / 4];
#pragma unroll
            for(int c = 0; c < NS / 4; c++)
                p[c] = *reinterpret_cast<const f32x4_t*>(s_p + (c * 64 + lane) * 4);
            then();
            opus::vector_t<float, NS> v_ds;
#pragma unroll
            for(int c = 0; c < NS / 4; c++)
#pragma unroll
                for(int e = 0; e < 4; e++)
                    v_ds[c * 4 + e] = p[c][e] * (v_dp[c * 4 + e] - d_e[c * 4 + e]);
            a_ds = opus::cast<bf16_t>(v_ds);
        };
        // DS: dS^T of query tile tt (in a_ds) to its tile in this pair's key tile's column
        // (gated_mla_bwd_ds), for the dQ pass. Lane l holds key 16 m + l % 16, queries
        // 16 n + 4 (l / 16) + 0 .. 3 at a_ds[8 m + 4 n ..]: per m, 16 bytes at element 8 (l / 16) of
        // the key's row. The tile's descriptor is made in the VALU stage (size 0 for a tile causal
        // masking drops, so that its stores do nothing), the stores issued in the MFMA stage after its
        // DMA wait, one half per step: s_waitcnt vmcnt counts stores too, and a store just before that
        // wait would make it wait out the store's latency.
        int ds_i0             = T;
        const bf16_t* ds_col  = nullptr;
        gated_mla_bwd::i32x4_t ds_rs{};
        const int ds_voff = (lane % 16 * 32 + lane / 16 * 8) * 2;
        if constexpr(DS)
        {
            namespace dsl = gated_mla_bwd_ds;
            const int jt         = row_w / 32;
            const long long seq0 = k.ptr_ds_seq[g];
            const long long tile0 =
                (static_cast<long long>(sc(static_cast<int>(seq0 >> 32))) << 32 | static_cast<unsigned int>(sc(static_cast<int>(seq0)))) +
                h * dsl::tiles_per_head(seqlen_q, seqlen_k, k.causal) + dsl::col_offset(jt, T, causal_offset, k.causal);
            ds_i0  = rows_w > 0 ? dsl::first_q_tile(jt, T, causal_offset, k.causal) : T;
            ds_col = reinterpret_cast<const bf16_t*>(k.ptr_ds) + tile0 * dsl::TILE;
        }
        auto ds_tile = [&](int tt) {
            if constexpr(DS)
            {
                const bool keep = tt >= ds_i0;
                ds_rs = gated_mla_bwd::dma_rsrc(ds_col + static_cast<long long>(keep ? tt - ds_i0 : 0) * gated_mla_bwd_ds::TILE,
                                                keep ? gated_mla_bwd_ds::TILE * 2 : 0);
                asm volatile("" : "+s"(ds_rs));
            }
        };
        auto store_ds = [&](int m) {
            if constexpr(DS)
            {
                const auto v = __builtin_bit_cast(opus::vector_t<opus::u32_t, 8>, a_ds);
                const opus::vector_t<opus::u32_t, 4> h4{v[m * 4], v[m * 4 + 1], v[m * 4 + 2], v[m * 4 + 3]};
                if(m == 0)
                    asm volatile("buffer_store_dwordx4 %0, %1, %2, 0 offen" GMLA_DS_CACHE : : "v"(h4), "v"(ds_voff), "s"(ds_rs) : "memory");
                else
                    asm volatile("buffer_store_dwordx4 %0, %1, %2, 0 offen offset:1024" GMLA_DS_CACHE : : "v"(h4), "v"(ds_voff), "s"(ds_rs) : "memory");
            }
        };
        constexpr int ST_DS0 = 2, ST_DS1 = 5;  // B steps of the two halves, between DMA rounds

        // P = 0 and dP^T = 0 before the first tile, so that its dS(t - 1) is 0 without a branch.
#pragma unroll
        for(int c = 0; c < NS / 4; c++)
            *reinterpret_cast<f32x4_t*>(s_p + (c * 64 + lane) * 4) = f32x4_t{0.0f, 0.0f, 0.0f, 0.0f};
        opus::clear(v_dp);
#if GMLA_PAIR_PP
        // The phases are laid out in the S wave's part; this wave runs one behind it. Per tile t it
        // has dK of tile t - 2, dS of tile t - 1 (as P(t - 1) comes, a float4 a phase) and dP^T of
        // tile t.
        acc_t dpd;  // (dP^T - D), then dS^T in fp32
        opus::clear(dpd);
        frag_t op[2];
        auto l_end = [&] {
            asm volatile("" ::: "memory");
            stage_end();
        };
        // P's float4 c, as the S wave wrote it.
        auto ld_p = [&](int c) { return *reinterpret_cast<const f32x4_t*>(s_p + (c * 64 + lane) * 4); };
        f32x4_t pc[NS / 4];
        // GMLA_X_PR 0: a register stands in for P.
        f32x4_t pz{0.0f, 0.0f, 0.0f, 0.0f};
        pin(pz);
        auto ld_pc = [&](int c) { return GMLA_X_PR ? ld_p(c) : pz; };
        // D of no tile before the first: with dP^T = 0 and P = 0, dS of the tile before the first is 0.
        opus::vector_t<float, 4> d_r[NT];
#pragma unroll
        for(int n = 0; n < NT; n++)
#pragma unroll
            for(int x = 0; x < 4; x++)
                d_r[n][x] = 0.0f;
        if(GMLA_X_D == 2)
        {
            pin(d_r[0]);
            pin(d_r[1]);
        }
        // One pack of dS^T to bf16, pinned (or the packs sink below the phase's last MFMA).
        auto pack_ds = [&](int j) {
            const opus::vector_t<float, 2> f2{dpd[2 * j], dpd[2 * j + 1]};
            auto h2 = opus::cast<bf16_t>(f2);
            pin(h2);
            a_ds[2 * j]     = h2[0];
            a_ds[2 * j + 1] = h2[1];
        };
        unsigned int a_d;
        int ob_tq = o_tq * 2, ob_do = o_do * 2, ob_d = o_run * 2;
        pin(ob_tq);
        pin(ob_do);
        pin(ob_d);

        // DMA in the phased loop: the dP waves issue every load of a query tile. dP wave wp issues
        // the tile's 16-byte loads wp + NP u, u = 0 .. 5 (the 22 loads of Q then dO; a load past
        // them repeats the wave's last one), and as round 6 LSE (even wp) or D. Rounds 0 .. 2 are
        // loads of Q and 4, 5 of dO for every wave; round 3 is Q's last load for wave 0 and a load
        // of dO for the others. So four descriptors a wave: Q's, round 3's, dO's and LSE's or D's,
        // each the tile they point at (its base, and its size: a full tile's, or what x_sizes sets
        // for the sequence's last, partial tile and the tiles past it). A round is then two
        // instructions: m0 from the buffer's LDS address and the round's place in it, and the load.
        constexpr int XR = 7;
        static_assert(J_Q == 13 && J_D == 9 && NP == 4, "rounds 0 .. 2 load Q, rounds 4, 5 dO");
        const int wp = w - NP;
        int x_lds[XR], x_os[XR];
#pragma unroll
        for(int u = 0; u < XR - 1; u++)
        {
            int jl = wp + NP * u;
            if(jl >= J_Q + J_D)
                jl -= NP;
            const bool isq = jl < J_Q;
            const int jj   = isq ? jl : jl - J_Q;
            const int rb   = isq ? k.stride_q_n * 2 : k.stride_do_n * 2;
            // readfirstlane: the compiler makes these selects in VALU, with the lanes' offsets'.
            x_lds[u]       = __builtin_amdgcn_readfirstlane((isq ? OFF_Q : OFF_DO) * 2 + jj * 1024);
            const int spr = isq ? SPR_Q : SPR_D, dch = isq ? D_QK / 8 : D_V / 8;
            const int sl = 64 * jj + lane, c = sl % spr;
            x_os[u]      = c < dch ? sl / spr * rb + c * 16 : OOB;
        }
        x_lds[XR - 1] = __builtin_amdgcn_readfirstlane((wp % 2 == 0 ? OFF_LSE : OFF_D) * 2);
        x_os[XR - 1]  = lane * 4;
        const bool r3q = wp == 0;
        // Row bytes and a full tile's bytes per descriptor.
        const int x_rb[4] = {k.stride_q_n * 2, (r3q ? k.stride_q_n : k.stride_do_n) * 2, k.stride_do_n * 2, 4};
        int x_kb[4];
        gated_mla_bwd::i32x4_t x_rs[4];
#pragma unroll
        for(int d = 0; d < 4; d++)
        {
            const char* src = d == 0 ? q_src : (d == 1 ? (r3q ? q_src : do_src) : (d == 2 ? do_src : (wp % 2 == 0 ? l_src : d_src)));
            x_kb[d]         = BQ * x_rb[d];
            x_rs[d]         = gated_mla_bwd::dma_rsrc(src + static_cast<long long>(i0) * BQ * x_rb[d], 0);
            asm volatile("" : "+s"(x_kb[d]));
        }
#pragma unroll
        for(int u = 0; u < XR; u++)
        {
            asm volatile("" : "+s"(x_lds[u]));
            pin(x_os[u]);
        }
        // The load phase (0 .. 4: a .. e) each round goes out in. The rounds of dP iteration t go to
        // tile t + 3, in buffer (t + 3) % NBUF = (t - 2) % NBUF: its dO, LSE and D were last read a
        // tile ago, its Q is read until this iteration's phase c, so Q's rounds wait for phase d.
        constexpr int x_ph[XR] = {GMLA_X_RPH};
        static_assert(x_ph[0] >= 3 && x_ph[1] >= 3 && x_ph[2] >= 3 && x_ph[3] >= 3, "Q's rounds after the buffer's last reads");
        // The rounds of a tile that are out before phase e's wait.
        constexpr int x_early = (x_ph[0] < 4) + (x_ph[1] < 4) + (x_ph[2] < 4) + (x_ph[3] < 4) + (x_ph[4] < 4) + (x_ph[5] < 4) + (x_ph[6] < 4);
        // Tile t's sizes: its rows of the sequence times each descriptor's row bytes.
        auto x_sizes = [&](int t) {
            const int rows = dma_rows(t);
#pragma unroll
            for(int d = 0; d < 4; d++)
            {
                int nb;
                asm("s_mul_i32 %0, %1, %2" : "=s"(nb) : "s"(rows), "s"(x_rb[d]));
                x_rs[d] = gated_mla_bwd::i32x4_t{x_rs[d][0], x_rs[d][1], nb, 0x00020000};
                asm volatile("" : "+s"(x_rs[d]));
            }
        };
        // Descriptor d a tile on.
        auto x_adv = [&](int d) {
            auto& r = x_rs[d];
            asm volatile("" : "+s"(r));  // made here, not where r was last set
            unsigned long long a =
                static_cast<unsigned int>(r[0]) | static_cast<unsigned long long>(static_cast<unsigned int>(r[1])) << 32;
            a += static_cast<unsigned long long>(static_cast<unsigned int>(x_kb[d]));
            r = gated_mla_bwd::i32x4_t{static_cast<int>(a), static_cast<int>(a >> 32), r[2], 0x00020000};
            asm volatile("" : "+s"(r));
        };
        // Round u into the buffer at LDS address lb.
        auto x_go = [&](int u, unsigned int lb) {
            const int d = u < 3 ? 0 : (u == 3 ? 1 : (u < XR - 1 ? 2 : 3));
            if(u == XR - 1)
                asm volatile("s_add_u32 m0, %2, %3\n\tbuffer_load_dword %0, %1, 0 offen lds"
                             :
                             : "v"(x_os[u]), "s"(x_rs[d]), "s"(lb), "s"(x_lds[u])
                             : "m0", "scc");
            else
                asm volatile("s_add_u32 m0, %2, %3\n\tbuffer_load_dwordx4 %0, %1, 0 offen lds"
                             :
                             : "v"(x_os[u]), "s"(x_rs[d]), "s"(lb), "s"(x_lds[u])
                             : "m0", "scc");
        };
        // The rounds of load phase ph.
        auto x_rounds = [&](int ph, unsigned int lb) {
#pragma unroll
            for(int u = 0; u < XR; u++)
                if(GMLA_X_DMA && (GMLA_X_RMASK >> u & 1) && x_ph[u] == ph)
                    x_go(u, lb);
        };
        (void)prologue;
        (void)bprev;
        // The first three tiles in LDS and this wave's V in registers before the loop (0x0f70 is
        // s_waitcnt vmcnt(0)); the S waves, which load nothing, wait at this barrier.
        {
            int b = b0;
#pragma unroll
            for(int x = 0; x < 3; x++, b = bnext(b))
            {
                if(x > 0)
                {
#pragma unroll
                    for(int d = 0; d < 4; d++)
                        x_adv(d);
                }
                x_sizes(i0 + x);
                const unsigned int lbx = dma_lds(b);
                if(i0 < T)
                {
#pragma unroll
                    for(int u = 0; u < XR; u++)
                        x_go(u, lbx);
                }
            }
            __builtin_amdgcn_s_waitcnt(0x0f70);
            dma_wait();
            stage_end();
        }
        // The dP waves run one phase behind.
        stage_end();
        if(i0 < T)
        {
            a_tq = addrb(b0, OFF_Q, ob_tq);
            int bt = b0, bp = b0;  // tile t's buffer, tile t - 1's (tile t's for the first)
            // The buffer this iteration's rounds go to (tile t + 3's) and its LDS address.
            int bd = bnext(bnext(bnext(b0)));
            unsigned int lb = dma_lds(bd);
            asm volatile("" : "+s"(lb));
#pragma unroll
            for(int d = 0; d < 4; d++)
                x_adv(d);
            x_sizes(i0 + 3);
            // A scalar made behind an MFMA: its input goes through an empty asm in that unit. The
            // compiler makes a scalar where its inputs are ready, which for these is the top of the
            // phase: 20 of them in a row there held the MFMAs back 80 cycles (ATT).
            auto here = [](auto& x) { asm volatile("" : "+s"(x)); };
            // The LDS addresses of this tile's dO and D (for its phases d and e) and of Q of the
            // tile before (for the next tile's phases a to c), one behind an MFMA: unit i of
            // compute phase ph.
            auto addrs = [&](int ph, int i) {
                const int k = GMLA_X_DSV ? (i == 7 ? ph : -1) : (ph == 2 ? i - 4 : -1);
                if(k == 0)
                {
                    here(bt);
                    a_do = addrb(bt, OFF_DO, ob_do);
                }
                if(k == 1)
                {
                    here(bt);
                    a_d = addrb(bt, OFF_D, ob_d);
                }
                if(k == 2)
                {
                    here(bp);
                    a_tq = addrb(bp, OFF_Q, ob_tq);
                }
            };
            // The loop closes with one conditional branch: the buffers move on before the test.
            int t = i0;
            bool more;
            do
            {
                // a: dK(t - 2) chunks 0, 1 (dS^T(t - 2) = 0 for the first two tiles, whose steps
                // read the first tile's Q and add nothing); behind them (dP^T - D)(t - 1). P(t - 1)
                // chunk 0, which the S wave wrote a phase ago.
                if(!GMLA_X_PRLAST)
                    pc[0] = ld_pc(0);
                op[0] = ld_tq(0);
                op[1] = ld_tq(1);
                if(GMLA_X_PRLAST)
                    pc[0] = ld_pc(0);
                __builtin_amdgcn_sched_barrier(0);
                store_ds(0);
                x_rounds(0, lb);
                l_end();
                if(GMLA_X_DSV)
                {
                    units([&](int i) { mfma1(a_ds, op[i / 4], v_dk[i / 4], i / 2 % 2, i % 2); },
                          [&](int i) {
                              if(GMLA_X_DSV == 1)
                                  dpd[i] = v_dp[i] - d_r[i / 4 % NT][i % 4];
                              addrs(0, i);
                          });
                }
                else
                {
                    ex_dk(0, op[0]);
                    ex_dk(1, op[1]);
#pragma unroll
                    for(int e = 0; e < NS; e++)
                        dpd[e] = v_dp[e] - d_r[e / 4 % NT][e % 4];
                    GMLA_SCHED8(0x002, 2)
                }
                pin(dpd);
                stage_end();
                // b: dK(t - 2) chunks 2, 3; behind them dS(t - 1) chunks 0, 1 = P (dP^T - D).
                // P(t - 1) chunk 1.
                if(!GMLA_X_PRLAST)
                    pc[1] = ld_pc(1);
                op[0] = ld_tq(2);
                op[1] = ld_tq(3);
                if(GMLA_X_PRLAST)
                    pc[1] = ld_pc(1);
                __builtin_amdgcn_sched_barrier(0);
                store_ds(1);
                x_rounds(1, lb);
                l_end();
                units([&](int i) { mfma1(a_ds, op[i / 4], v_dk[2 + i / 4], i / 2 % 2, i % 2); },
                      [&](int i) {
                          if(GMLA_X_DSV == 1)
                              dpd[8 + i] = v_dp[8 + i] - d_r[i / 4 % NT][i % 4];
                          else if(GMLA_X_DSV == 2)
                          {
#pragma unroll
                              for(int e = 2 * i; e < 2 * i + 2; e++)
                                  dpd[e] = v_dp[e] - d_r[e / 4 % NT][e % 4];
                          }
                          else
                              dpd[i] = pc[i / 4][i % 4] * dpd[i];
                          addrs(1, i);
                      });
                pin(dpd);
                stage_end();
                // c: dK(t - 2) chunks 4, 5; behind them dS(t - 1) chunk 2 and the LDS addresses of
                // the operands to come. P(t - 1) chunk 2.
                if(!GMLA_X_PRLAST)
                    pc[2] = ld_pc(2);
                op[0] = ld_tq(4);
                op[1] = ld_tq(5);
                if(GMLA_X_PRLAST)
                    pc[2] = ld_pc(2);
                __builtin_amdgcn_sched_barrier(0);
                x_rounds(2, lb);
                l_end();
                units([&](int i) { mfma1(a_ds, op[i / 4], v_dk[4 + i / 4], i / 2 % 2, i % 2); },
                      [&](int i) {
                          if(GMLA_X_DSV)
                              dpd[i] = pc[i / 4][i % 4] * dpd[i];
                          else if(i < 4)
                              dpd[8 + i] = pc[2][i] * dpd[8 + i];
                          addrs(2, i);
                      });
                pin(dpd);
                stage_end();
                // d: dP^T(t) steps 0, 1; behind them dS(t - 1) chunk 3 and the first half of
                // dS^T(t - 1) to bf16 (dK(t - 2) is done). P(t - 1) chunk 3.
                if(!GMLA_X_PRLAST)
                    pc[3] = ld_pc(3);
                op[0] = ld_do(0);
                op[1] = ld_do(1);
                if(GMLA_X_PRLAST)
                    pc[3] = ld_pc(3);
                __builtin_amdgcn_sched_barrier(0);
                x_rounds(3, lb);
                l_end();
                units([&](int i) { mfma1(a_v(i / 4), op[i / 4], v_dp, i / 2 % 2, i % 2, i < 4); },
                      [&](int i) {
                          if(i < 4)
                          {
                              const int e = (GMLA_X_DSV ? 8 : 12) + i;
                              dpd[e]      = pc[e / 4][i] * dpd[e];
                          }
                          else
                              pack_ds(i - 4);
                      });
                pin(v_dp);
                pin(dpd);
                pin(a_ds);
                stage_end();
                // e: dP^T(t) steps 2, 3; behind them the second half of dS^T(t - 1) to bf16, the
                // buffer of the next tile's rounds and the descriptors a tile on. D(t).
                // This wave's loads of tile t + 2, issued a tile ago, are waited for here, by
                // count: x_early loads of tile t + 3 are out (loads return in order; the stores of
                // dS(t - 2), counted too, only make the wait longer if they are still out). Then
                // this phase's rounds, and the sizes of the next tile's if it is not a full one.
                // The descriptor of dS(t - 1)'s tile; the loop test, before the barriers.
                op[0] = ld_do(2);
                op[1] = ld_do(3);
                if(GMLA_X_D == 1)
                {
#pragma unroll
                    for(int n = 0; n < NT; n++)
                        d_r[n] = at_f(a_d).template load<4>(16 * n);
                }
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_waitcnt vmcnt(%0)" : : "n"(GMLA_X_DMA ? x_early : 0) : "memory");
                x_rounds(4, lb);
                if(GMLA_X_DMA)
                {
                    int rl = seqlen_q - (t + 4) * BQ;
                    asm volatile("" : "+s"(rl));
                    if(__builtin_expect(rl < BQ, 0))
                        x_sizes(t + 4);
                }
                ds_tile(t - 1);
                const int bn = bnext(bt);
                const int tn = next_tile(t);
                more         = tn < T;
                l_end();
                units([&](int i) { mfma1(a_v(2 + i / 4), op[i / 4], v_dp, i / 2 % 2, i % 2); },
                      [&](int i) {
                          if(GMLA_X_DSV)
                          {
                              if(i < 4)
                                  dpd[12 + i] = pc[3][i] * dpd[12 + i];
                              if(i < 2)
                                  pack_ds(4 + i);
                              if(i == 4 || i == 5)
                                  pack_ds(2 + i);
                          }
                          else if(i < 4)
                              pack_ds(4 + i);
                          if(GMLA_X_DMA && i == (GMLA_X_DSV ? 2 : 0))
                              bd = bnext(bd);
                          if(GMLA_X_DMA && i == (GMLA_X_DSV ? 3 : 1))
                          {
                              here(bd);
                              lb = dma_lds(bd);
                              asm volatile("" : "+s"(lb));
                          }
                          if(GMLA_X_DMA && i >= 4)
                              x_adv(i - 4);
                      });
                pin(v_dp);
                pin(dpd);
                pin(a_ds);
                stage_end();
                t  = tn;
                bp = bt;
                bt = bn;
                if(GMLA_X_LOOPPIN)
                    asm volatile("" : "+s"(t), "+s"(bp), "+s"(bt));
            } while(__builtin_expect(more, 1));
            // Tail. The S wave wrote P of the last tile whole a phase ago. dK(T - 2), with dS(T - 1)
            // behind its first steps and to bf16 after its last; then dK(T - 1).
            op[0] = ld_tq(0);
            op[1] = ld_tq(1);
#pragma unroll
            for(int c = 0; c < NS / 4; c++)
                pc[c] = ld_p(c);
            __builtin_amdgcn_sched_barrier(0);
            store_ds(0);
            l_end();
            ex_dk(0, op[0]);
            ex_dk(1, op[1]);
#pragma unroll
            for(int e = 0; e < NS; e++)
                dpd[e] = pc[e / 4][e % 4] * (v_dp[e] - d_r[e / 4 % NT][e % 4]);
            pin(dpd);
            stage_end();
            op[0] = ld_tq(2);
            op[1] = ld_tq(3);
            __builtin_amdgcn_sched_barrier(0);
            store_ds(1);
            l_end();
            ex_dk(2, op[0]);
            ex_dk(3, op[1]);
            stage_end();
            op[0] = ld_tq(4);
            op[1] = ld_tq(5);
            l_end();
            ex_dk(4, op[0]);
            ex_dk(5, op[1]);
#pragma unroll
            for(int c = 0; c < NK; c++)
                pin(v_dk[c]);
            a_ds = opus::cast<bf16_t>(dpd);
            pin(a_ds);
            ds_tile(T - 1);
            a_tq = addrb(bp, OFF_Q, ob_tq);
            stage_end();
#pragma unroll
            for(int c = 0; c < NK; c += 2)
            {
                op[0] = ld_tq(c);
                op[1] = ld_tq(c + 1);
                if(c == 0)
                    store_ds(0);
                if(c == 2)
                    store_ds(1);
                l_end();
                ex_dk(c, op[0]);
                ex_dk(c + 1, op[1]);
                stage_end();
            }
        }
#else
        prologue(std::true_type{});
        if(i0 < T)
        {
            int bt = b0, bp = b0;  // tile t's buffer, tile t - 1's (tile t's for the first)
            for(int t = i0;;)
            {
                // A(t): dS(t - 1), its LDS reads ahead of the first steps of B(t). Also this
                // wave's rounds of tile t + 3, in buffer (t + 3) % NBUF = (t - 2) % NBUF, and the
                // next tile's buffer.
                ds(bp, [&] {
                    a_do = addr(bt, OFF_DO, o_do);
                    a_tq = addr(bp, OFF_Q, o_tq);
                    prefetch(ld_do);
                });
                dma_next(t + 3, bprev(bprev(bt)), true, std::true_type{});
                ds_tile(t - 1);
                const int bn = bnext(bt);
                pin(a_ds);
                pin_ring();
                stage_end();
                // B(t): dP^T(t), then dK(t - 1). Steps 1, 1 + SP_P, .. also issue this wave's
                // share of tile t + 3, as the S waves do theirs of tile t + 2 in A(t); its share of
                // tile t + 2, issued a tile ago, is waited for first, before B(t + 1) reads it.
                // Buffer (t + 3) % NBUF last held tile t - 2, done with in A(t - 1) and B(t - 1).
                static_assert(1 + SP_P * (JP - 1) < KS_P + NK, "every round in the stage");
                c_stage(number<KS_P + NK>{},
                        [&](int st) { return st < KS_P ? ld_do(st) : ld_tq(st - KS_P); },
                        [&](int st, const frag_t& b) {
                            if(st < KS_P)
                                ex_dp(st, b);
                            else
                                ex_dk(st - KS_P, b);
                            if(st == 1)
                                dma_wait();
                            if(st >= 1 && (st - 1) % SP_P == 0 && (st - 1) / SP_P < JP)
                                dma_round((st - 1) / SP_P, std::true_type{});
                            if(st == ST_DS0 || st == ST_DS1)
                                store_ds(st == ST_DS1);
                        });
                pin(v_dp);
                const int tn    = next_tile(t);
                const bool more = tn < T;
                stage_end();
                if(!more)
                    break;
                t  = tn;
                bp = bt;
                bt = bn;
            }
            // A': dS of the last tile and the first steps of B'. B': dK of it.
            const int bl = (T - 1) % NBUF * SZ_BUF;
            ds(bl, [&] {
                a_tq = addr(bl, OFF_Q, o_tq);
                prefetch(ld_tq);
            });
            ds_tile(T - 1);
            pin(a_ds);
            pin_ring();
            stage_end();
            c_stage(number<NK>{}, ld_tq, [&](int st, const frag_t& b) {
                ex_dk(st, b);
                if(st == 0 || st == 1)
                    store_ds(st);
            });
            stage_end();
        }
#endif

        // dK nope into the kv_b_proj layout, dK rope per head in fp32.
#pragma unroll
        for(int c = 0; c < NK; c++)
#pragma unroll
            for(int e = 0; e < NS; e++)
                v_dk[c][e] *= k.softmax_scale;
        auto g_dkn = dkv_part(0);
        auto g_dkr = opus::make_gmem(k.ptr_dk_rope + static_cast<long long>(k0 + row_w) * k.stride_dkr_n +
                                         static_cast<long long>(h) * k.stride_dkr_h,
                                     rec(static_cast<long long>(rows_w) * k.stride_dkr_n, 4));
        const auto u_dkr = opus::partition_layout_c(mma_k1, opus::make_tuple(k.stride_dkr_n, 1_I), c_c);
#pragma unroll
        for(int c = 0; c < NK_N; c++)
            g_dkn.template store<4>(opus::cast<bf16_t>(v_dk[c]), u_dkv, c * 32);
#pragma unroll
        for(int c = NK_N; c < NK; c++)
            g_dkr.template store<4>(v_dk[c], u_dkr, (c - NK_N) * 32);
    }
}

// dK / dV pass on 32x32x16 MFMAs, as gated_mla_bwd_dkdv_pair_kernel (same roles, stages, DMA and
// stagger) with the fragments of gated_mla_bwd_dq32_kernel. S = Q K^T and dP = dO V^T, so that a
// lane holds one key's column of 16 queries, rows 8 (i / 4) + 4 (lane / 32) + i % 4 of the C
// fragment. Elements 8 b .. 8 b + 7 of P and dS are then directly the B operand of query block b
// of dV^T = dO^T P and dK^T = Q^T dS, whose A operand is read transposed from the dO and Q tiles in
// the same query order; their C fragments give each lane one key and four runs of 4 columns.
//   A(t): S wave  S(t) (12 MFMA), then dV^T(t - 1) (8 MFMA)
//         dP wave dS(t - 1), from P(t - 1) (fp32, through LDS) and its own dP(t - 1)
//   B(t): S wave  P(t) = exp(S(t) - LSE), to LDS in fp32 and kept in bf16 for dV^T(t)
//         dP wave dP(t) (8 MFMA), then dK^T(t - 1) (12 MFMA)
// A step is one MFMA, its LDS operand (8 bf16) read LA steps ahead through a ring. Query tiles are
// laid out as gated_mla_bwd_pair32 describes.
__global__ __launch_bounds__(gated_mla_bwd_pair32::BLOCK, 1) void gated_mla_bwd_dkdv_pair32_kernel(gated_mla_bwd_kargs k)
{
    using namespace gated_mla_bwd_pair32;
    using gated_mla_bwd::bf16_t;
    using gated_mla_bwd::bvec;
    using gated_mla_bwd::f32x4_t;
    constexpr float LOG2_E = 1.44269504089f;
    __shared__ __attribute__((aligned(16))) bf16_t smem[SMEM_ELEMS];

    const int tid     = static_cast<int>(opus::thread_id_x());
    const int lane    = tid % 64;
    const int w       = __builtin_amdgcn_readfirstlane(tid / 64);
    const int pr      = w % NP;
    const bool s_wave = w < NP;
    const int lq = lane % 32, lh = lane / 32;  // this lane's key (C column) and K half
    // Workgroup id -> (head, sequence, key block): gated_mla_bwd::wg_map.
    const int wg   = static_cast<int>(opus::block_id_x());
    const auto ids = gated_mla_bwd::wg_map_sgpr(wg, k.H, k.num_seqs, k.n_kv_blocks, k.hg);
    const int h = ids.h, g = ids.g, j = ids.blk;
    auto sc            = [](int x) { return __builtin_amdgcn_readfirstlane(x); };
    const int q0       = sc(k.ptr_seqstart_q[g]);
    const int k0       = sc(k.ptr_seqstart_k[g]);
    const int seqlen_q = sc(k.ptr_seqstart_q[g + 1]) - q0;
    const int seqlen_k = sc(k.ptr_seqstart_k[g + 1]) - k0;
    const int kv0      = j * BK;
    if(kv0 >= seqlen_k)
        return;
    // Bottom-right causal: query qp sees keys kv <= qp + causal_offset. Query tiles i0 .. T - 1.
    const int causal_offset = seqlen_k - seqlen_q;
    const int T             = (seqlen_q + BQ - 1) / BQ;
    int i0                  = 0;
    if(k.causal && kv0 - causal_offset > 0)
        i0 = (kv0 - causal_offset) / BQ;

    static_assert(PR == 32 && BQ == 32, "one 32x32 C fragment per tile and pair");
    using frag_t = bvec<8>;                    // an MFMA operand: 8 bf16 per lane
    using acc_t  = opus::vector_t<float, 16>;  // a 32 x 32 C fragment
    auto mfma    = [](const frag_t& a, const frag_t& b, const acc_t& c) -> acc_t {
        return __builtin_amdgcn_mfma_f32_32x32x16_bf16(a, b, c, 0, 0, 0);
    };
    constexpr int NS = 16;
    constexpr int KS = D_QK / 16, KS_N = D_NOPE / 16, KS_P = D_V / 16;
    constexpr int NV = D_V / 32, NK = D_QK / 32, NK_N = D_NOPE / 32;  // 32-column chunks of dV, dK

    auto rec = [](long long elems, int elem_bytes) -> unsigned int {
        const long long b = elems > 0 ? elems * elem_bytes : 0;
        return b >= 0xffffffffLL ? 0xffffffffu : static_cast<unsigned int>(b);
    };
    // This pair's 32 keys; rows past seqlen_k read as 0 and are dropped on store.
    const int row_w  = kv0 + pr * PR;
    const int rows_w = seqlen_k - row_w < PR ? (seqlen_k - row_w > 0 ? seqlen_k - row_w : 0) : PR;
    // The tiles this pair sees in full (no mask: every key and query valid, the tile's first query
    // at or past its last key) form one range [i_full0, i_full1).
    int i_full0 = T, i_full1 = T;
    if(row_w + PR <= seqlen_k)
    {
        const int need = row_w + PR - 1 - causal_offset;
        i_full0        = k.causal && need > 0 ? (need + BQ - 1) / BQ : 0;
        i_full1        = seqlen_q / BQ;
        i_full1        = i_full1 < i_full0 ? i_full0 : i_full1;
    }

    // DMA of a query tile as in gated_mla_bwd_dkdv_pair_kernel (14 Q and 10 dO loads of 1 KB,
    // then LSE and D), into the swizzled rows: slot s, row s / SPR, physical chunk p = s % SPR,
    // holds chunk p ^ swz(row).
    auto swz = [](int row) { return ((row >> 2) & 1) | (((row >> 4) & 1) << 1); };
    constexpr int SPR_Q = LD_Q * 2 / 16, SPR_D = LD_DO * 2 / 16;
    constexpr int J_Q = BQ * SPR_Q / 64, J_D = BQ * SPR_D / 64, JS = 4, JP = 3, JU = JS;
    static_assert(BQ * SPR_Q % 64 == 0 && BQ * SPR_D % 64 == 0, "whole loads per tensor");
    static_assert(NP * (JS + JP - 1) >= J_Q + J_D && NP * (JS + JP - 2) < J_Q + J_D && NP >= 2, "the load split");
    constexpr int OOB = 1 << 30;  // byte offset past any tile
    const char* q_src  = reinterpret_cast<const char*>(reinterpret_cast<const bf16_t*>(k.ptr_q) +
                                                      static_cast<long long>(q0) * k.stride_q_n +
                                                      static_cast<long long>(h) * k.stride_q_h);
    const char* do_src = reinterpret_cast<const char*>(reinterpret_cast<const bf16_t*>(k.ptr_do) +
                                                       static_cast<long long>(q0) * k.stride_do_n +
                                                       static_cast<long long>(h) * k.stride_do_h);
    const char* l_src  = reinterpret_cast<const char*>(k.ptr_lse + static_cast<long long>(h) * k.stride_lse_h + q0);
    const char* d_src  = reinterpret_cast<const char*>(k.ptr_d + static_cast<long long>(h) * k.stride_lse_h + q0);
    const char* d_ptr[JU];
    int d_rb[JU], d_lds[JU], d_os[JU];
#pragma unroll
    for(int u = 0; u < JU; u++)
    {
        int jl = s_wave ? w + NP * u : NP * JS + w - NP + NP * u;
        if(jl >= J_Q + J_D)
            jl -= NP;
        const int reg = !s_wave && u == JP - 1 ? 2 + (w - NP) % 2 : (jl < J_Q ? 0 : 1);
        const int jj  = reg == 0 ? jl : jl - J_Q;
        d_rb[u]       = reg == 0 ? k.stride_q_n * 2 : (reg == 1 ? k.stride_do_n * 2 : 4);
        d_ptr[u]      = (reg == 0 ? q_src : (reg == 1 ? do_src : (reg == 2 ? l_src : d_src))) +
                   static_cast<long long>(i0) * BQ * d_rb[u];
        d_lds[u] = reg == 0 ? OFF_Q * 2 + jj * 1024 : (reg == 1 ? OFF_DO * 2 + jj * 1024 : (reg == 2 ? OFF_LSE : OFF_D) * 2);
        const int spr = reg == 0 ? SPR_Q : SPR_D, dch = reg == 0 ? D_QK / 8 : D_V / 8;
        const int sl = 64 * jj + lane, c = sl % spr, row = sl / spr;
        d_os[u]      = reg >= 2 ? lane * 4 : (c < dch ? row * d_rb[u] + (c ^ swz(row)) * 16 : OOB);
    }
    auto dma_rows = [&](int t) {
        int r = seqlen_q - t * BQ;
        asm("s_min_i32 %0, %0, %1\n\ts_max_i32 %0, %0, 0" : "+s"(r) : "n"(BQ) : "scc");
        return r;
    };
    auto bnext     = [](int b) { return b + 1 == NBUF ? 0 : b + 1; };
    auto bprev     = [](int b) { return b == 0 ? NBUF - 1 : b - 1; };
    auto dma_lds   = [&](int b) { return gated_mla_bwd::lds_addr(smem + b * SZ_BUF); };
    auto dma_round = [&](int rows, unsigned int lb, int u, auto dp_c) {
        auto rs = gated_mla_bwd::dma_rsrc(d_ptr[u], 0);
        asm("s_mul_i32 %0, %1, %2" : "=s"(rs[2]) : "s"(rows), "s"(d_rb[u]));
        if(decltype(dp_c)::value && u == JP - 1)
            gated_mla_bwd::dma4(rs, d_os[u], lb + d_lds[u]);
        else
            gated_mla_bwd::dma16(rs, d_os[u], lb + d_lds[u]);
        d_ptr[u] += BQ * d_rb[u];
    };
    auto dma = [&](int t, int b, auto dp_c) {
        constexpr int n = decltype(dp_c)::value ? JP : JS;
#pragma unroll
        for(int u = 0; u < n; u++)
            dma_round(dma_rows(t), dma_lds(b), u, dp_c);
    };
    auto dma_wait  = [] { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); };
    auto stage_end = [] {
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    };
    auto pin     = [](auto& x) { asm volatile("" : "+v"(x)); };
    auto pin_dma = [](const char* (&ptr)[JU], int n_rounds) {
#pragma unroll
        for(int u = 0; u < JU; u++)
            if(u < n_rounds)
                asm volatile("" : "+s"(ptr[u]));
    };

    // LDS reads at a per-lane address (made and pinned in the wave's VALU stage before the MFMA
    // stage that reads it) plus immediates, as in gated_mla_bwd_dq32_kernel: A fragments of Q and
    // dO (row lq, logical chunk 2 s + lh; even and odd K steps have their own lane offset) and the
    // transposed reads of dO and Q (query blocks b = 0, 1 have their own lane offset).
    const int G = lane / 16, gi = lane % 16;
    auto o_frag = [&](int ld, int odd) {
        const int f = swz(lq);
        return lq * ld + 8 * (lh ^ (f & 1)) + (odd ? -16 : 16) * (f >> 1);
    };
    auto o_trans = [&](int ld, int b) {
        const int pc = (2 * (G % 2) + (gi % 4) / 2) ^ (G / 2) ^ (b << 1);
        return (4 * (G / 2) + gi / 4) * ld + 8 * pc + 4 * (gi % 2);
    };
    auto addr = [&](int b, int off, int o) {
        unsigned int a = gated_mla_bwd::lds_addr(smem + b * SZ_BUF + off) + o * 2;
        pin(a);
        return a;
    };
    auto addr2 = [&](unsigned int(&a)[2], int b, int off, const int(&o)[2]) {
        a[0] = addr(b, off, o[0]);
        a[1] = addr(b, off, o[1]);
    };
    auto lds16 = [](unsigned int a) -> frag_t {
        return *reinterpret_cast<const OPUS_LDS_ADDR frag_t*>(static_cast<__UINTPTR_TYPE__>(a));
    };
    auto at    = [](unsigned int a) { return opus::make_smem(reinterpret_cast<bf16_t*>(static_cast<__UINTPTR_TYPE__>(a))); };
    auto ld_tr = [&](unsigned int a, int ld, int col, int b) -> frag_t {
        const auto r0 = gated_mla_bwd::tr_read(at(a), 16 * b * ld + col);
        const auto r1 = gated_mla_bwd::tr_read(at(a), (16 * b + 8) * ld + col);
        return __builtin_bit_cast(frag_t, opus::vector_t<opus::u32_t, 4>{r0[0], r0[1], r1[0], r1[1]});
    };
    // C element i of S / dP is query 8 (i / 4) + 4 lh + i % 4 of the tile, key row_w + lq: run i / 4
    // of 4 queries is one float4 of LSE or D. P goes to the dP wave in that layout, chunk-major so
    // that each float4 access is lane-contiguous.
    auto ld_runs = [&](int b, int off, float mul, float(&out)[NS]) {
        const float* f = reinterpret_cast<const float*>(smem + b * SZ_BUF + off);
        f32x4_t run[4];
#pragma unroll
        for(int r = 0; r < 4; r++)
            run[r] = *reinterpret_cast<const f32x4_t*>(f + 8 * r + 4 * lh);
#pragma unroll
        for(int i = 0; i < NS; i++)
            out[i] = run[i / 4][i % 4] * mul;
    };
    float* s_p = reinterpret_cast<float*>(smem + OFF_P) + pr * PR * BQ;

    // LDS operands of LA steps in flight, in a ring of LA + 1 slots: a step's read (of step + LA)
    // goes to the slot step - 1 consumed and can issue beside the step's MFMA, which leaves 24 of
    // its 32 cycles to other instructions.
    constexpr int LA = 3, RS = LA + 1;
    frag_t ring[RS];
    auto c_stage = [&](auto n_steps, auto ld, auto ex) {
        constexpr int N = decltype(n_steps)::value;
#pragma unroll
        for(int st = 0; st < N; st++)
        {
            ex(st, ring[st % RS]);
            if(st + LA < N)
                ring[(st + LA) % RS] = ld(st + LA);
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);  // MFMA
            __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);  // DS read
            __builtin_amdgcn_sched_group_barrier(0x004, 3, 0);  // SALU
            __builtin_amdgcn_sched_barrier(0);
        }
    };
    auto prefetch = [&](auto ld) {
#pragma unroll
        for(int st = 0; st < LA; st++)
            ring[st] = ld(st);
    };
    auto pin_ring = [&] {
#pragma unroll
        for(int r = 0; r < LA; r++)
            pin(ring[r]);
    };
    const acc_t zero{};
    const float scale_log2 = k.softmax_scale * LOG2_E;
    // This pair's rows of dkv [.., H, dK nope | dV]; rows past seqlen_k fall outside the
    // descriptors and are dropped. A transposed C fragment's element i of chunk c is column
    // 32 c + 8 (i / 4) + 4 lh + i % 4 of key lq: four runs of 4 columns.
    auto dkv_part = [&](int col0) {
        bf16_t* dkv = reinterpret_cast<bf16_t*>(k.ptr_dkv) + static_cast<long long>(k0 + row_w) * k.stride_dkv_n +
                      static_cast<long long>(h) * k.stride_dkv_h + col0;
        return opus::make_gmem(dkv, rec(static_cast<long long>(rows_w) * k.stride_dkv_n - col0, 2));
    };
    auto store_bf16 = [&](auto& g, const acc_t& v, int stride, int c) {
        const auto o = opus::cast<bf16_t>(v);
#pragma unroll
        for(int r = 0; r < 4; r++)
            g.template store<4>(opus::vector_t<bf16_t, 4>{o[4 * r], o[4 * r + 1], o[4 * r + 2], o[4 * r + 3]},
                                lq * stride + 32 * c + 8 * r + 4 * lh);
    };
    const int b0 = i0 % NBUF;
    auto prologue = [&](auto dp_c) {
        if(i0 < T)
        {
            int b = b0;
            for(int x = 0; x < (decltype(dp_c)::value ? 3 : 2); x++, b = bnext(b))
                dma(i0 + x, b, dp_c);
        }
        __builtin_amdgcn_s_waitcnt(0x0f70);
        dma_wait();
        stage_end();
    };

    if(s_wave)
    {
        // K as the B operand of S: lane lq's key, columns 16 s + 8 lh .. + 7, nope then rope.
        auto g_kn = opus::make_gmem(reinterpret_cast<const bf16_t*>(k.ptr_k_nope) +
                                        static_cast<long long>(k0 + row_w) * k.stride_k_n +
                                        static_cast<long long>(h) * k.stride_k_h,
                                    rec(static_cast<long long>(rows_w) * k.stride_k_n, 2));
        auto g_kr = opus::make_gmem(reinterpret_cast<const bf16_t*>(k.ptr_k_rot) +
                                        static_cast<long long>(k0 + row_w) * k.stride_k_rot_n,
                                    rec(static_cast<long long>(rows_w) * k.stride_k_rot_n, 2));
        frag_t v_k[KS];
#pragma unroll
        for(int s = 0; s < KS; s++)
            v_k[s] = s < KS_N ? g_kn.template load<8>(lq * k.stride_k_n + 16 * s + 8 * lh)
                              : g_kr.template load<8>(lq * k.stride_k_rot_n + 16 * (s - KS_N) + 8 * lh);
        const int c_kv = row_w + lq;

        acc_t v_s, v_dv[NV];
#pragma unroll
        for(int c = 0; c < NV; c++)
            opus::clear(v_dv[c]);
        // P(t - 1) per query block, the B operand of dV^T(t - 1): zero before the first tile, whose
        // dV^T(t - 1) steps then add nothing (they read the first tile's dO, which is finite).
        frag_t p_bf[2];
        opus::clear(p_bf[0]);
        opus::clear(p_bf[1]);
        unsigned int a_q[2], a_tdo[2];  // S(t): Q; dV^T(t - 1): dO, transposed
        const int o_q[2] = {o_frag(LD_Q, 0), o_frag(LD_Q, 1)}, o_tdo[2] = {o_trans(LD_DO, 0), o_trans(LD_DO, 1)};
        auto ld_q   = [&](int s) -> frag_t { return lds16(a_q[s % 2] + 32 * s); };
        auto ld_tdo = [&](int jj) -> frag_t { return ld_tr(a_tdo[jj % 2], LD_DO, 32 * (jj / 2), jj % 2); };
        auto ex_s   = [&](int s, const frag_t& a) { v_s = mfma(a, v_k[s], s == 0 ? zero : v_s); };
        auto ex_dv  = [&](int jj, const frag_t& a) { v_dv[jj / 2] = mfma(a, p_bf[jj % 2], v_dv[jj / 2]); };

        prologue(std::false_type{});
        if(i0 < T)
        {
            addr2(a_q, b0, OFF_Q, o_q);
            addr2(a_tdo, b0, OFF_DO, o_tdo);
            prefetch(ld_q);
            int bt = b0;  // tile t's buffer
            for(int t = i0; t < T; t++, bt = bnext(bt))
            {
                const int qt0         = t * BQ;
                const int b1          = bnext(bt);
                const int r2          = dma_rows(t + 2);
                const unsigned int l2 = dma_lds(bnext(b1));
                // A(t): S(t), then dV^T(t - 1); steps 1 .. JS also issue this wave's share of tile
                // t + 2, as in gated_mla_bwd_dkdv_pair_kernel.
                c_stage(opus::number<KS + 2 * NV>{},
                        [&](int st) { return st < KS ? ld_q(st) : ld_tdo(st - KS); },
                        [&](int st, const frag_t& a) {
                            if(st < KS)
                                ex_s(st, a);
                            else
                                ex_dv(st - KS, a);
                            if(st == 1)
                                dma_wait();
                            if(st >= 1 && st <= JS)
                                dma_round(r2, l2, st - 1, std::false_type{});
                        });
                pin(v_s);
                pin_dma(d_ptr, JS);
                stage_end();
                // B(t): LSE, then the next A stage's first steps (S(t + 1); past the last tile they
                // read a buffer the tail does not use); P(t), each float4 to LDS as soon as its
                // exps are done.
                float lse_e[NS];
                ld_runs(bt, OFF_LSE, LOG2_E, lse_e);
                addr2(a_tdo, bt, OFF_DO, o_tdo);
                addr2(a_q, b1, OFF_Q, o_q);
                prefetch(ld_q);
                auto write_p = [&](int c) {
                    *reinterpret_cast<f32x4_t*>(s_p + (c * 64 + lane) * 4) =
                        f32x4_t{v_s[c * 4], v_s[c * 4 + 1], v_s[c * 4 + 2], v_s[c * 4 + 3]};
                    __builtin_amdgcn_sched_barrier(0);
                };
                if(static_cast<unsigned int>(t - i_full0) < static_cast<unsigned int>(i_full1 - i_full0))
                {
#pragma unroll
                    for(int c = 0; c < NS / 4; c++)
                    {
#pragma unroll
                        for(int i = c * 4; i < c * 4 + 4; i++)
                            v_s[i] = __builtin_amdgcn_exp2f(v_s[i] * scale_log2 - lse_e[i]);
                        write_p(c);
                    }
                }
                else
                {
#pragma unroll
                    for(int c = 0; c < NS / 4; c++)
                    {
#pragma unroll
                        for(int i = c * 4; i < c * 4 + 4; i++)
                        {
                            const int qp  = qt0 + 8 * (i / 4) + 4 * lh + i % 4;
                            const bool ok = (c_kv < seqlen_k) & (qp < seqlen_q) & (!k.causal | (qp + causal_offset >= c_kv));
                            const float x = v_s[i] * scale_log2 - lse_e[i];
                            v_s[i]        = __builtin_amdgcn_exp2f(ok ? x : -1e30f);
                        }
                        write_p(c);
                    }
                }
#pragma unroll
                for(int b = 0; b < 2; b++)
                {
                    opus::vector_t<float, 8> v;
#pragma unroll
                    for(int x = 0; x < 8; x++)
                        v[x] = v_s[8 * b + x];
                    p_bf[b] = opus::cast<bf16_t>(v);
                    pin(p_bf[b]);
                }
                pin_ring();
                stage_end();
            }
            // A': dV^T of the last tile. B': the dP waves' dK^T of it.
            prefetch(ld_tdo);
            c_stage(opus::number<2 * NV>{}, ld_tdo, ex_dv);
            stage_end();
            stage_end();
        }

        auto g_dv = dkv_part(D_NOPE);
#pragma unroll
        for(int c = 0; c < NV; c++)
            store_bf16(g_dv, v_dv[c], k.stride_dkv_n, c);
    }
    else
    {
        // V as the B operand of dP: lane lq's key, columns 16 s + 8 lh .. + 7.
        auto g_v = opus::make_gmem(reinterpret_cast<const bf16_t*>(k.ptr_v) +
                                       static_cast<long long>(k0 + row_w) * k.stride_v_n +
                                       static_cast<long long>(h) * k.stride_v_h,
                                   rec(static_cast<long long>(rows_w) * k.stride_v_n, 2));
        frag_t v_v[KS_P];
#pragma unroll
        for(int s = 0; s < KS_P; s++)
            v_v[s] = g_v.template load<8>(lq * k.stride_v_n + 16 * s + 8 * lh);

        acc_t v_dp, v_dk[NK];
#pragma unroll
        for(int c = 0; c < NK; c++)
            opus::clear(v_dk[c]);
        // dS(t - 1) per query block, the B operand of dK^T(t - 1): zero before the first tile.
        frag_t b_ds[2];
        opus::clear(b_ds[0]);
        opus::clear(b_ds[1]);
        unsigned int a_do[2], a_tq[2];  // dP(t): dO; dK^T(t - 1): Q, transposed
        const int o_do[2] = {o_frag(LD_DO, 0), o_frag(LD_DO, 1)}, o_tq[2] = {o_trans(LD_Q, 0), o_trans(LD_Q, 1)};
        auto ld_do = [&](int s) -> frag_t { return lds16(a_do[s % 2] + 32 * s); };
        auto ld_tq = [&](int jj) -> frag_t { return ld_tr(a_tq[jj % 2], LD_Q, 32 * (jj / 2), jj % 2); };
        auto ex_dp = [&](int s, const frag_t& a) { v_dp = mfma(a, v_v[s], s == 0 ? zero : v_dp); };
        auto ex_dk = [&](int jj, const frag_t& a) { v_dk[jj / 2] = mfma(a, b_ds[jj % 2], v_dk[jj / 2]); };
        // dS = P (dP - D) of the tile in buffer b, in bf16 per query block (softmax_scale goes onto
        // dK at the end); `then` issues the next stage's first reads after these.
        auto ds = [&](int b, auto then) {
            float d_e[NS];
            ld_runs(b, OFF_D, 1.0f, d_e);
            f32x4_t p[NS / 4];
#pragma unroll
            for(int c = 0; c < NS / 4; c++)
                p[c] = *reinterpret_cast<const f32x4_t*>(s_p + (c * 64 + lane) * 4);
            then();
#pragma unroll
            for(int blk = 0; blk < 2; blk++)
            {
                opus::vector_t<float, 8> v;
#pragma unroll
                for(int x = 0; x < 8; x++)
                {
                    const int i = 8 * blk + x;
                    v[x]        = p[i / 4][i % 4] * (v_dp[i] - d_e[i]);
                }
                b_ds[blk] = opus::cast<bf16_t>(v);
            }
        };

        // P = 0 and dP = 0 before the first tile, so that its dS(t - 1) is 0 without a branch.
#pragma unroll
        for(int c = 0; c < NS / 4; c++)
            *reinterpret_cast<f32x4_t*>(s_p + (c * 64 + lane) * 4) = f32x4_t{0.0f, 0.0f, 0.0f, 0.0f};
        opus::clear(v_dp);
        prologue(std::true_type{});
        if(i0 < T)
        {
            int bt = b0;  // tile t's buffer
            for(int t = i0; t < T; t++, bt = bnext(bt))
            {
                const int bp = t > i0 ? bprev(bt) : bt;  // tile t - 1's (tile t's for the first)
                // A(t): dS(t - 1), its LDS reads ahead of the first steps of B(t).
                ds(bp, [&] {
                    addr2(a_do, bt, OFF_DO, o_do);
                    addr2(a_tq, bp, OFF_Q, o_tq);
                    prefetch(ld_do);
                });
                pin(b_ds[0]);
                pin(b_ds[1]);
                pin_ring();
                stage_end();
                // B(t): dP(t), then dK^T(t - 1); steps 1 .. JP also issue this wave's share of tile
                // t + 3, as in gated_mla_bwd_dkdv_pair_kernel.
                const int r3          = dma_rows(t + 3);
                const unsigned int l3 = dma_lds(bprev(bprev(bt)));
                c_stage(opus::number<KS_P + 2 * NK>{},
                        [&](int st) { return st < KS_P ? ld_do(st) : ld_tq(st - KS_P); },
                        [&](int st, const frag_t& a) {
                            if(st < KS_P)
                                ex_dp(st, a);
                            else
                                ex_dk(st - KS_P, a);
                            if(st == 1)
                                dma_wait();
                            if(st >= 1 && st <= JP)
                                dma_round(r3, l3, st - 1, std::true_type{});
                        });
                pin(v_dp);
                pin_dma(d_ptr, JP);
                stage_end();
            }
            // A': dS of the last tile and the first steps of B'. B': dK^T of it.
            const int bl = (T - 1) % NBUF;
            ds(bl, [&] {
                addr2(a_tq, bl, OFF_Q, o_tq);
                prefetch(ld_tq);
            });
            pin(b_ds[0]);
            pin(b_ds[1]);
            pin_ring();
            stage_end();
            c_stage(opus::number<2 * NK>{}, ld_tq, ex_dk);
            stage_end();
        }

        // dK nope into the kv_b_proj layout, dK rope per head in fp32.
#pragma unroll
        for(int c = 0; c < NK; c++)
#pragma unroll
            for(int i = 0; i < NS; i++)
                v_dk[c][i] *= k.softmax_scale;
        auto g_dkn = dkv_part(0);
#pragma unroll
        for(int c = 0; c < NK_N; c++)
            store_bf16(g_dkn, v_dk[c], k.stride_dkv_n, c);
        auto g_dkr = opus::make_gmem(k.ptr_dk_rope + static_cast<long long>(k0 + row_w) * k.stride_dkr_n +
                                         static_cast<long long>(h) * k.stride_dkr_h,
                                     rec(static_cast<long long>(rows_w) * k.stride_dkr_n, 4));
#pragma unroll
        for(int c = NK_N; c < NK; c++)
#pragma unroll
            for(int r = 0; r < 4; r++)
                g_dkr.template store<4>(f32x4_t{v_dk[c][4 * r], v_dk[c][4 * r + 1], v_dk[c][4 * r + 2], v_dk[c][4 * r + 3]},
                                        lq * k.stride_dkr_n + 32 * (c - NK_N) + 8 * r + 4 * lh);
    }
}

// The 4-wave dK / dV loop's schedule (GMLA_4W_PP): what goes behind the MFMA of each unit of its four blocks
// (S^T 36 units, dP^T 24, dK 36, dV 24). Element e of a tile's 24 S^T / dP^T values per lane is C tile (M tile
// e / 8, N tile e / 4 % 2); its query's LSE and D have index (e / 4 % 2) 4 + e % 4.
//   V_LSE x    LSE log2 e           V_FMA e    S^T scale - LSE      V_EXP e   its exp, P (fills a unit)
//   V_SUB e    dP^T - D             V_MUL e    P (dP^T - D), dS in fp32
//   V_NEG x    -D, index x (GMLA_DP_MINUS_D: dP^T starts at -D and there is no V_SUB; D is then read a tile
//              ahead, in the dV units of V_DD, behind V_ALD)
//   V_PP j     P's elements 2 j, 2 j + 1 to bf16 (A of dV)          V_PD j    dS's of the tile before (A of dK)
//   V_AQ, V_ADO, V_ATDO, V_ALD   an LDS address a tile on; V_ATQ   Q^T's address from Q's
//   R_Q f, R_O f   S^T's / dP^T's B fragment f (N tile, K step in N-tile-major order); R_TQ g, h / R_TO g, h
//   Q^T's / dO^T's N tile g, half h; R_LS n, R_DD n   LSE / D of N tile n;   S_X  the DMA round of an S^T unit
// A unit has room for two of VALU and read behind its MFMA (an exp is two, a DMA round both); the S^T block's
// first 7 units take no VALU (it stands still behind the barrier). The dK and dV blocks' own reads are regular
// and not in a table. S_EXPD: bit e set if element e's exp is in the S^T block (the mask of a tile not seen in
// full is applied after it: those take P = 0, the others an S^T that exps to 0). The kernel reads the tables in
// constant expressions only. Made and checked by a script (the dependencies and the units' room).
namespace gated_mla_bwd_4ws {
enum : signed char { V_NONE, V_LSE, V_FMA, V_EXP, V_SUB, V_MUL, V_PP, V_PD, V_AQ, V_ATQ, V_ADO, V_ATDO, V_ALD, V_NEG };
enum : signed char { R_NONE, R_Q, R_O, R_TQ, R_TO, R_LS, R_DD };
struct vop
{
    signed char k, e;
};
struct rop
{
    signed char k, a, b;
};
#if GMLA_DP_MINUS_D
inline constexpr vop S_V[36][3] = {
    {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_LSE, 0}, {V_LSE, 1}, {V_NONE, 0}},
    {{V_LSE, 2}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_LSE, 3}, {V_PD, 0}, {V_NONE, 0}},
    {{V_PD, 1}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PD, 2}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_PD, 3}, {V_PD, 4}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PD, 5}, {V_PD, 6}, {V_NONE, 0}},
    {{V_FMA, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PD, 7}, {V_PD, 8}, {V_NONE, 0}},
    {{V_FMA, 1}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PD, 9}, {V_PD, 10}, {V_NONE, 0}}, {{V_FMA, 2}, {V_FMA, 3}, {V_NONE, 0}},
    {{V_FMA, 8}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PD, 11}, {V_LSE, 4}, {V_NONE, 0}}, {{V_FMA, 9}, {V_FMA, 10}, {V_NONE, 0}},
    {{V_FMA, 11}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 16}, {V_FMA, 17}, {V_NONE, 0}}, {{V_FMA, 18}, {V_FMA, 19}, {V_NONE, 0}},
    {{V_LSE, 5}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_LSE, 6}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_LSE, 7}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 1}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}},
};
inline constexpr rop S_R[36] = {
    {R_Q, 2, 0}, {R_NONE, 0, 0}, {R_LS, 0, 0}, {R_Q, 3, 0}, {R_NONE, 0, 0}, {R_LS, 1, 0},
    {R_Q, 4, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_Q, 5, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
    {R_Q, 6, 0}, {R_NONE, 0, 0}, {R_Q, 7, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
    {R_Q, 8, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_Q, 9, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
    {R_Q, 10, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_Q, 11, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
    {R_O, 0, 0}, {R_NONE, 0, 0}, {R_O, 1, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
};
inline constexpr vop P_V[24][3] = {
    {{V_EXP, 2}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 4}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 3}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_FMA, 5}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 8}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 9}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_FMA, 6}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 10}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 7}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_EXP, 11}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 16}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 17}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_FMA, 12}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 18}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 13}, {V_MUL, 0}, {V_NONE, 0}},
    {{V_FMA, 14}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 15}, {V_MUL, 1}, {V_NONE, 0}}, {{V_FMA, 20}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_FMA, 21}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 22}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 23}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_EXP, 19}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 4}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 5}, {V_NONE, 0}, {V_NONE, 0}},
};
inline constexpr rop P_R[24] = {
    {R_NONE, 0, 0}, {R_O, 2, 0}, {R_NONE, 0, 0}, {R_O, 3, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
    {R_O, 4, 0}, {R_NONE, 0, 0}, {R_O, 5, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
    {R_O, 6, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_O, 7, 0}, {R_NONE, 0, 0}, {R_TQ, 0, 0},
    {R_TQ, 0, 1}, {R_TQ, 1, 0}, {R_TQ, 1, 1}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
};
inline constexpr vop K_V[36][3] = {
    {{V_MUL, 2}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 3}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 6}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_MUL, 8}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 9}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 7}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_MUL, 10}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 11}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 12}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_PP, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PP, 1}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 13}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_PP, 4}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PP, 5}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 14}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_PP, 8}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PP, 9}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 15}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_PP, 2}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PP, 3}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 20}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_PP, 6}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PP, 7}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 21}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_PP, 10}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 16}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 22}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_MUL, 17}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 18}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 23}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_PP, 11}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 19}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 4}, {V_MUL, 5}, {V_NONE, 0}},
    {{V_MUL, 6}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 7}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 12}, {V_MUL, 13}, {V_NONE, 0}},
};
inline constexpr vop V_V[24][3] = {
    {{V_ALD, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 14}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 15}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_MUL, 20}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 21}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 22}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_NEG, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NEG, 1}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NEG, 2}, {V_NEG, 3}, {V_NONE, 0}},
    {{V_NEG, 4}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NEG, 5}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NEG, 6}, {V_NEG, 7}, {V_NONE, 0}},
    {{V_ATQ, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_AQ, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_ADO, 0}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_MUL, 23}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_ATDO, 0}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}},
};
inline constexpr signed char S_X[36] = {-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5, -1, -1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
inline constexpr unsigned int S_EXPD = 0x000003;
inline constexpr signed char V_DD[24] = {-1, -1, 0, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
#else
inline constexpr vop S_V[36][3] = {
    {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_LSE, 0}, {V_LSE, 1}, {V_NONE, 0}},
    {{V_LSE, 2}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_LSE, 3}, {V_PD, 0}, {V_NONE, 0}},
    {{V_PD, 1}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PD, 2}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_PD, 3}, {V_PD, 4}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PD, 5}, {V_PD, 6}, {V_NONE, 0}},
    {{V_FMA, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PD, 7}, {V_PD, 8}, {V_NONE, 0}},
    {{V_FMA, 1}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PD, 9}, {V_PD, 10}, {V_NONE, 0}}, {{V_FMA, 2}, {V_FMA, 3}, {V_NONE, 0}},
    {{V_FMA, 8}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PD, 11}, {V_LSE, 4}, {V_NONE, 0}}, {{V_FMA, 9}, {V_FMA, 10}, {V_NONE, 0}},
    {{V_FMA, 11}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 16}, {V_FMA, 17}, {V_NONE, 0}}, {{V_FMA, 18}, {V_FMA, 19}, {V_NONE, 0}},
    {{V_LSE, 5}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_LSE, 6}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_LSE, 7}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 1}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}},
};
inline constexpr rop S_R[36] = {
    {R_Q, 2, 0}, {R_NONE, 0, 0}, {R_LS, 0, 0}, {R_Q, 3, 0}, {R_NONE, 0, 0}, {R_LS, 1, 0},
    {R_Q, 4, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_Q, 5, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
    {R_Q, 6, 0}, {R_NONE, 0, 0}, {R_Q, 7, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
    {R_Q, 8, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_Q, 9, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
    {R_Q, 10, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_Q, 11, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
    {R_O, 0, 0}, {R_NONE, 0, 0}, {R_O, 1, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
};
inline constexpr vop P_V[24][3] = {
    {{V_EXP, 2}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 4}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 3}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_FMA, 5}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 8}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 9}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_FMA, 6}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 10}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 7}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_EXP, 11}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 16}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 17}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_FMA, 12}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 18}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 13}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_FMA, 14}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 15}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 20}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_FMA, 21}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 22}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 23}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_EXP, 19}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 4}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 5}, {V_NONE, 0}, {V_NONE, 0}},
};
inline constexpr rop P_R[24] = {
    {R_NONE, 0, 0}, {R_O, 2, 0}, {R_NONE, 0, 0}, {R_O, 3, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
    {R_O, 4, 0}, {R_NONE, 0, 0}, {R_O, 5, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
    {R_O, 6, 0}, {R_NONE, 0, 0}, {R_DD, 0, 0}, {R_O, 7, 0}, {R_DD, 1, 0}, {R_TQ, 0, 0},
    {R_TQ, 0, 1}, {R_TQ, 1, 0}, {R_TQ, 1, 1}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
};
inline constexpr vop K_V[36][3] = {
    {{V_SUB, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_SUB, 1}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 6}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_SUB, 2}, {V_NONE, 0}, {V_NONE, 0}}, {{V_SUB, 3}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 7}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_SUB, 8}, {V_NONE, 0}, {V_NONE, 0}}, {{V_SUB, 9}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 12}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_SUB, 10}, {V_NONE, 0}, {V_NONE, 0}}, {{V_SUB, 11}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 13}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_SUB, 16}, {V_NONE, 0}, {V_NONE, 0}}, {{V_SUB, 17}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 14}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_SUB, 18}, {V_NONE, 0}, {V_NONE, 0}}, {{V_SUB, 19}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 15}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_PP, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PP, 1}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 20}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_PP, 4}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PP, 5}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 21}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_PP, 8}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PP, 9}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 22}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_PP, 2}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PP, 3}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 23}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_PP, 6}, {V_SUB, 12}, {V_NONE, 0}}, {{V_PP, 7}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PP, 10}, {V_SUB, 4}, {V_NONE, 0}},
    {{V_SUB, 5}, {V_SUB, 13}, {V_NONE, 0}}, {{V_SUB, 6}, {V_SUB, 14}, {V_NONE, 0}}, {{V_PP, 11}, {V_SUB, 7}, {V_NONE, 0}},
};
inline constexpr vop V_V[24][3] = {
    {{V_MUL, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 1}, {V_NONE, 0}, {V_NONE, 0}}, {{V_SUB, 15}, {V_SUB, 20}, {V_NONE, 0}},
    {{V_MUL, 2}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 3}, {V_NONE, 0}, {V_NONE, 0}}, {{V_SUB, 21}, {V_SUB, 22}, {V_NONE, 0}},
    {{V_MUL, 8}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 9}, {V_NONE, 0}, {V_NONE, 0}}, {{V_SUB, 23}, {V_ATQ, 0}, {V_NONE, 0}},
    {{V_MUL, 10}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 11}, {V_NONE, 0}, {V_NONE, 0}}, {{V_AQ, 0}, {V_ADO, 0}, {V_NONE, 0}},
    {{V_MUL, 16}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 17}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 18}, {V_MUL, 19}, {V_NONE, 0}},
    {{V_MUL, 4}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 5}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 6}, {V_MUL, 7}, {V_NONE, 0}},
    {{V_MUL, 12}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 13}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 14}, {V_MUL, 15}, {V_NONE, 0}},
    {{V_MUL, 20}, {V_MUL, 21}, {V_NONE, 0}}, {{V_MUL, 22}, {V_MUL, 23}, {V_NONE, 0}}, {{V_ATDO, 0}, {V_ALD, 0}, {V_NONE, 0}},
};
inline constexpr signed char S_X[36] = {-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5, -1, -1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
inline constexpr unsigned int S_EXPD = 0x000003;
inline constexpr signed char V_DD[24] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
#endif
}  // namespace gated_mla_bwd_4ws

// dK / dV pass of the 4-wave backward. Each wave runs at one per SIMD so that its 48 keys' K and V (A of
// S^T and dP^T) stay in VGPRs and dK and dV in AGPRs: every LDS fragment of Q and dO feeds
// three M tiles. GEMMs are per wave, C = A * B^T, all 16x16x32 MFMAs: S^T and dP^T over d; dV
// and dK over the tile's 32 queries, with P^T and dS^T (the C fragments of S^T and dP^T, two N
// tiles per M tile) as their A operands and transposed reads of the dO and Q tiles as B. Query
// tiles go global -> LDS by DMA two tiles ahead into a ring of four buffers, one barrier per tile.
__device__ __attribute__((always_inline)) inline void gated_mla_bwd_dkdv_4w_body(
    const gated_mla_bwd_4w_kargs& k, gated_mla_bwd::bf16_t* smem, const int lane, const int w, const int h, const int j,
    unsigned long long bnd_q, unsigned long long bnd_k, gated_mla_bwd::f32x4_t (&z)[3][12])
{
    using namespace gated_mla_bwd_4w;
    using gated_mla_bwd::bf16_t;
    using gated_mla_bwd::bvec;
    using gated_mla_bwd::f32x4_t;
    using gated_mla_bwd::mfma_acc_drain;
    using opus::number;
    using opus::operator""_I;
    using vec8_t = bvec<8>;
    constexpr float LOG2_E = 1.44269504089f;
    constexpr int MR       = WR / 16;  // M tiles per wave
    constexpr int NT       = BQ / 16;  // N tiles of S^T / dP^T
    // dK's accumulators, [M tile][N tile] in the swap_ab C layout, in AGPRs: the kernel's zeros.
    constexpr int NV = D_V / 16, NR = D_ROT / 16;
    static_assert(MR == 3 && NV + NR == 12, "the kernel's zeroed tiles");
    f32x4_t acc_dkn[MR][NV], acc_dkr[MR][NR];
#pragma unroll
    for(int m = 0; m < MR; m++)
    {
#pragma unroll
        for(int n = 0; n < NV; n++)
            acc_dkn[m][n] = z[m][n];
#pragma unroll
        for(int n = 0; n < NR; n++)
            acc_dkr[m][n] = z[m][NV + n];
    }

    // The tensors' bases at this head.
    const bf16_t* q_h  = reinterpret_cast<const bf16_t*>(k.ptr_q) + static_cast<long long>(h) * k.stride_q_h;
    const bf16_t* do_h = reinterpret_cast<const bf16_t*>(k.ptr_do) + static_cast<long long>(h) * k.stride_do_h;
    const bf16_t* kn_h = reinterpret_cast<const bf16_t*>(k.ptr_k_nope) + static_cast<long long>(h) * k.stride_k_h;
    const bf16_t* v_h  = reinterpret_cast<const bf16_t*>(k.ptr_v) + static_cast<long long>(h) * k.stride_v_h;
    const float* ld_h  = (w % 2 == 0 ? k.ptr_lse : k.ptr_d) + static_cast<long long>(h) * k.stride_lse_h;
    asm volatile("" : "+s"(q_h), "+s"(do_h), "+s"(kn_h), "+s"(v_h), "+s"(ld_h));

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

    // C element e = (m NT + tn) 4 + p of S^T / dP^T is key 16 m + lane % 16 of the wave and
    // query 16 tn + 4 (lane / 16) + p of the tile.
    constexpr int NS = decltype(mma_s)::tile_c_len;
    static_assert(NS == MR * NT * 4);
    constexpr int KS = D_QK / 32, KS_N = D_NOPE / 32, KS_R = D_ROT / 32;
    const int c_q = 4 * (lane / 16);

    // One wave per SIMD: behind an MFMA the wave issues one scalar (or wait) and two of LDS read
    // and VALU (an exp is two, a DMA round takes both). An iteration (S^T tile ts) is 120 such
    // units; what each does is in gated_mla_bwd_4ws:
    //   barrier               every wave is past its reads of tile ts - 2, tile ts + 1 is whole
    //   S^T(ts)      36 MFMA  the 7 DMA rounds of tile ts + 2 into tile ts - 2's buffer, Q's
    //                         fragments two ahead, LSE and LSE log2 e, dS of tile ts - 1 to bf16,
    //                         the loop's scalars, S^T scale - LSE of the C tiles done mid block
    //   dP^T(ts)     24 MFMA  the other S^T scale - LSE, half of the exps, D
    //   dK(ts - 1)   36 MFMA  Q^T's fragments two N tiles ahead; the other exps, dP^T - D, P to
    //                         bf16 for dV
    //   dV(ts)       24 MFMA  dO^T's fragments two N tiles ahead; dS = P (dP^T - D) in fp32, the
    //                         LDS addresses a tile on
    // S^T and dP^T go N tile by N tile (a B fragment serves three consecutive MFMAs, one per M
    // tile), so half their C tiles are done mid block. dK runs a tile behind so that the VALU
    // work between S^T and dS has the MFMAs of dP^T, dK and dV to go behind. No MFMA waits for
    // LDS: every fragment is read two fragments or two N tiles ahead into registers the ones
    // before left free. dS = P (dP^T - D): dP^T's accumulators start at -D (GMLA_DP_MINUS_D; D is
    // then read and negated a tile ahead, in the dV block), or at 0 with D subtracted as the pair
    // pass does it, bit for bit. Query tiles (Q, dO, LSE, D) go global -> LDS by DMA into 4
    // buffers.
    constexpr int XB_Q = 0, XB_DO = BQ * LD_Q, XB_LSE = XB_DO + BQ * LD_DO, XB_D = XB_LSE + 128;
    constexpr int XB_SZ = XB_D + 128, XNB = 4;
    static_assert(XNB * XB_SZ <= SMEM_ELEMS, "four tile buffers");
    auto pin    = [](auto& x) { asm volatile("" : "+v"(x)); };
    auto fence  = [] { __builtin_amdgcn_sched_barrier(0); };
    float sc2   = k.softmax_scale * LOG2_E;  // the scale in a VGPR, for the asm fma
    pin(sc2);
    // A fragment the next MFMA reads, named right behind this one: the compiler puts its wait
    // for the LDS read (s_waitcnt lgkmcnt) before the first use, which is then here, in the
    // cycles the MFMA leaves to scalars, not in front of the next MFMA.
    auto touch  = [](const auto& x) { asm volatile("" : : "v"(x)); };
    auto touch2 = [](const auto& x, const auto& y) { asm volatile("" : : "v"(x), "v"(y)); };
    auto lds_of = [&](int b) { return gated_mla_bwd::lds_addr(smem + b); };

    // DMA: a load is 64 lanes x 16 B from a tile laid out with the row padding, to a wave-uniform
    // LDS address: load jj of a tensor covers its 16-byte slots 64 jj .. + 63, slot s being row
    // s / QS, chunk s % QS of Q (s / DS, s % DS of dO); padding chunks and rows past seqlen_q read
    // out of bounds, as 0. Wave w issues the tile's loads w + NW u, u = 0 .. 5 (13 of Q then 9 of
    // dO; a load past them repeats the wave's last one) and, as round 6, LSE (even w) or D.
    // Rounds 0 .. 2 load Q, 4 and 5 dO, round 3 Q's last load for wave 0 and dO for the others:
    // four descriptors, each pointing at the tile to load (base; size a full tile's, or what
    // x_sizes sets for the sequence's last, partial tile and the tiles past it).
    constexpr int XR = 7, J_Q = 13, J_D = 9;
    constexpr int QS = LD_Q * 2 / 16, DS = LD_DO * 2 / 16;  // 16-byte slots per padded row
    constexpr int OOB = 1 << 30;                            // byte offset past any tile
    static_assert(BQ * QS == J_Q * 64 && BQ * DS == J_D * 64 && NW == 4, "the rounds below");
    // The lanes' byte offsets of load jj (jj wave-uniform; the divisors are constants).
    auto os_q = [&](int jj) {
        const unsigned int sl = 64u * jj + lane, c = sl % QS;
        const int in = -static_cast<int>(c < D_QK / 8);  // all ones in a chunk of the row, 0 in its padding
        return (static_cast<int>(sl / QS * (k.stride_q_n * 2) + c * 16) & in) | (OOB & ~in);
    };
    auto os_d = [&](int jj) {
        const unsigned int sl = 64u * jj + lane, c = sl % DS;
        const int in = -static_cast<int>(c < D_V / 8);  // all ones in a chunk of the row, 0 in its padding
        return (static_cast<int>(sl / DS * (k.stride_do_n * 2) + c * 16) & in) | (OOB & ~in);
    };
    const bool r3q = w == 0;
    const int j5   = w < 2 ? w + 7 : w + 3;
    // readfirstlane: the compiler makes these selects in VALU, with the lanes' offsets'.
    auto u_lds = [](int x) { return __builtin_amdgcn_readfirstlane(x); };
    int x_lds[XR], x_os[XR];
#pragma unroll
    for(int u = 0; u < 3; u++)
    {
        x_lds[u] = u_lds(XB_Q * 2 + (w + NW * u) * 1024);
        x_os[u]  = os_q(w + NW * u);
    }
    x_lds[3] = u_lds(r3q ? XB_Q * 2 + 12 * 1024 : XB_DO * 2 + (w - 1) * 1024);
    x_os[3]  = r3q ? os_q(12) : os_d(w - 1);
    x_lds[4] = u_lds(XB_DO * 2 + (w + 3) * 1024);
    x_os[4]  = os_d(w + 3);
    x_lds[5] = u_lds(XB_DO * 2 + j5 * 1024);
    x_os[5]  = os_d(j5);
    x_lds[6] = u_lds((w % 2 == 0 ? XB_LSE : XB_D) * 2);
    x_os[6]  = lane * 4;
#pragma unroll
    for(int u = 0; u < XR; u++)
    {
        asm volatile("" : "+s"(x_lds[u]));
        pin(x_os[u]);
    }

    // Fragments of one K step or N tile, at a per-lane LDS address plus an immediate.
    auto mma_k1    = make_mma(number<MR>{}, number<NT>{}, number<1>{});
    auto lay_b     = [&](auto ld, const auto& c) { return opus::partition_layout_b<8>(mma_k1, opus::make_tuple(ld, 1_I), c); };
    const auto c_0 = opus::make_tuple(0_I, 0_I, 0_I, 0_I);
    const auto of_q  = opus::layout_to_offsets<8>(lay_b(number<LD_Q>{}, c_0));
    const auto of_do = opus::layout_to_offsets<8>(lay_b(number<LD_DO>{}, c_0));
    const int ob_q   = (opus::layout_to_offsets<8>(lay_b(number<LD_Q>{}, c_ab))[0] - of_q[0]) * 2;
    const int ob_do  = (opus::layout_to_offsets<8>(lay_b(number<LD_DO>{}, c_ab))[0] - of_do[0]) * 2;
    const int ob_tq  = gated_mla_bwd::tr_c_order_lane<LD_Q>(lane) * 2;
    const int ob_tdo = gated_mla_bwd::tr_c_order_lane<LD_DO>(lane) * 2;
    const int ob_ld  = 16 * (lane / 16);  // this lane's float4 run of LSE or D
    int c_tq         = ob_tq - ob_q;      // Q^T's per-lane address is Q's plus this
    pin(c_tq);
    auto at   = [](unsigned int a) { return opus::make_smem(reinterpret_cast<bf16_t*>(static_cast<__UINTPTR_TYPE__>(a))); };
    auto at_f = [](unsigned int a) { return opus::make_smem(reinterpret_cast<float*>(static_cast<__UINTPTR_TYPE__>(a))); };
    // An operand's address in buffer b: the scalar part in SALU, one v_add.
    auto addr = [&](int b, int off, int ob) {
        unsigned int sb = gated_mla_bwd::lds_addr(smem + b + off);
        asm volatile("" : "+s"(sb));
        unsigned int a = sb + static_cast<unsigned int>(ob);
        pin(a);
        return a;
    };
    // The first tile is in buffer 0. The first dK block is of no tile (its dS is 0, but what it
    // reads must be finite): it reads the first tile's Q^T too.
    unsigned int a_q = addr(0, XB_Q, ob_q), a_do = addr(0, XB_DO, ob_do), a_tdo = addr(0, XB_DO, ob_tdo);
    unsigned int a_ld = addr(0, XB_LSE, ob_ld), a_tq = addr(0, XB_Q, ob_tq);
    auto rd_q  = [&](int tn, int ks) __attribute__((always_inline)) -> bvec<8> { return at(a_q).template load<8>(of_q[tn] + ks * 32); };
    auto rd_do = [&](int tn, int ks) __attribute__((always_inline)) -> bvec<8> { return at(a_do).template load<8>(of_do[tn] + ks * 32); };
    // One transposed read of N tile gn (16 columns) of Q or dO, half hh of the tile's 32 queries.
    auto rd_tq = [&](int gn, int hh) __attribute__((always_inline)) { return gated_mla_bwd::tr_read(at(a_tq), gn * 16 + hh * 16 * LD_Q); };
    auto rd_to = [&](int gn, int hh) __attribute__((always_inline)) { return gated_mla_bwd::tr_read(at(a_tdo), gn * 16 + hh * 16 * LD_DO); };
    using u32x2_t = gated_mla_bwd::u32x2_t;
    using u32x4_t = gated_mla_bwd::u32x4_t;
    auto b_tr = [](const u32x2_t (&hv)[2]) __attribute__((always_inline)) { return u32x4_t{hv[0][0], hv[0][1], hv[1][0], hv[1][1]}; };
    // An MFMA of dV or dK, its accumulator in AGPRs.
    auto acc1 = [](f32x4_t& acc, const u32x4_t& a, const u32x4_t& b) __attribute__((always_inline)) {
        asm volatile("v_mfma_f32_16x16x32_bf16 %0, %2, %1, %0" : "+a"(acc) : "v"(a), "v"(b));
    };
    auto barrier = [&] {
        fence();
        __builtin_amdgcn_s_barrier();
        fence();
    };
    constexpr int KSV = D_V / 32, NG = NV + NR;
    static_assert(MR == 3 && NT == 2 && KS == 6 && KSV == 4 && NV == 8 && NG == 12 && NS == 24, "the schedule's shape");

    bvec<8> bq[3], bdo[3];  // B fragments of S^T and dP^T: the one in use and the next two
    u32x2_t tb[3][2];       // B of dK and dV: this N tile's two reads and the next two's
    opus::vector_t<float, NS> v_sp, v_dp;
    opus::vector_t<bf16_t, MR * 8> a_p, a_ds;
    f32x4_t lse4[NT], d4[NT];
    f32x4_t c_old{0.0f, 0.0f, 0.0f, 0.0f};  // what a C tile's last MFMA accumulates onto
#pragma unroll
    for(int e = 0; e < NS; e++)
    {
        v_sp[e] = 0.0f;  // P and dS of the tile before the first are 0
        v_dp[e] = 0.0f;
    }
    // One MFMA of S^T (va is K, vacc S^T) or dP^T (V, dP^T) onto C tile tl. The first K step
    // (step 0, from 0) and the steps between (1, in place) are asm; the last (2) is the
    // compiler's MFMA. The compiler's may put its result in other registers and hand the old ones
    // to the next instruction that needs some (an LDS read), which then stands s_nop behind the
    // MFMA that reads them: the old ones (c_old) are kept to the end of the unit after.
    auto mm = [&](const auto& va, int ia, const bvec<8>& fb, auto& vacc, int tl, int step, const f32x4_t* c0 = nullptr) __attribute__((always_inline)) {
        bvec<8> fa;
        f32x4_t cc;
#pragma unroll
        for(int x = 0; x < 8; x++)
            fa[x] = va[ia * 8 + x];
        if(step == 0)
        {
            // From 0, or from what the accumulator starts at (c0: -D for dP).
            if(c0)
                asm volatile("v_mfma_f32_16x16x32_bf16 %0, %2, %1, %3" : "=&v"(cc) : "v"(fa), "v"(fb), "v"(*c0));
            else
                asm volatile("v_mfma_f32_16x16x32_bf16 %0, %2, %1, 0" : "=&v"(cc) : "v"(fa), "v"(fb));
        }
        else
        {
#pragma unroll
            for(int p_ = 0; p_ < 4; p_++)
                cc[p_] = vacc[tl * 4 + p_];
            if(step == 1)
                asm volatile("v_mfma_f32_16x16x32_bf16 %0, %2, %1, %0" : "+v"(cc) : "v"(fa), "v"(fb));
            else
            {
                c_old = cc;
                cc    = mma1(fa, fb, cc);
                asm volatile("" : "+v"(cc));
            }
        }
#pragma unroll
        for(int p_ = 0; p_ < 4; p_++)
            vacc[tl * 4 + p_] = cc[p_];
    };
    // The VALU of a unit is asm, as its first MFMAs are: a unit's instructions then stand in the
    // order they are written (the compiler's own VALU goes where ISel and the scheduler put it:
    // a unit early, or in front of the unit's read, which then issues 12 cycles late). What is
    // left to the compiler in the loop: the LDS reads (chained between the asm around them), the
    // last K step's MFMA, the descriptors' 64-bit steps and the waits. It does not know these
    // VALU read what an MFMA wrote: the schedule keeps them 3 units behind that MFMA. They are
    // the instructions the compiler makes of the pair pass's expressions.
    auto pack = [&](const opus::vector_t<float, NS>& v, opus::vector_t<bf16_t, MR * 8>& o, int j) __attribute__((always_inline)) {
        opus::vector_t<bf16_t, 2> h2;
        asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(h2) : "v"(v[2 * j]), "v"(v[2 * j + 1]));
        o[2 * j]     = h2[0];
        o[2 * j + 1] = h2[1];
    };
    auto lse1 = [&](int x) __attribute__((always_inline)) {  // LSE log2 e (0x3fb8aa3b)
        float l = lse4[x / 4][x % 4];
        asm volatile("v_mul_f32 %0, 0x3fb8aa3b, %0" : "+v"(l));
        lse4[x / 4][x % 4] = l;
    };
    auto fma1 = [&](int e) __attribute__((always_inline)) {
        float x = v_sp[e];
        asm volatile("v_fma_f32 %0, %0, %1, -%2" : "+v"(x) : "v"(sc2), "v"(lse4[e / 4 % NT][e % 4]));
        v_sp[e] = x;
    };
    auto expa = [&](int e) __attribute__((always_inline)) {
        float x = v_sp[e];
        asm volatile("v_exp_f32_e32 %0, %0" : "+v"(x));
        v_sp[e] = x;
    };
    auto sub1 = [&](int e) __attribute__((always_inline)) {
        float x = v_dp[e];
        asm volatile("v_sub_f32 %0, %0, %1" : "+v"(x) : "v"(d4[e / 4 % NT][e % 4]));
        v_dp[e] = x;
    };
    // -D of query index x, in place: D's sign bit (GMLA_DP_MINUS_D).
    auto neg1 = [&](int x) __attribute__((always_inline)) {
        float nd = d4[x / 4][x % 4];
        asm volatile("v_xor_b32 %0, 0x80000000, %0" : "+v"(nd));
        d4[x / 4][x % 4] = nd;
    };
    auto mul1 = [&](int e) __attribute__((always_inline)) {
        float x = v_dp[e];
        asm volatile("v_mul_f32 %0, %1, %0" : "+v"(x) : "v"(v_sp[e]));
        v_dp[e] = x;
    };

    // The bounds are in.
    asm volatile("s_waitcnt lgkmcnt(0)" : "+s"(bnd_q), "+s"(bnd_k));
    const int q0 = static_cast<int>(bnd_q), k0 = static_cast<int>(bnd_k);
    const int seqlen_q = static_cast<int>(bnd_q >> 32) - q0, seqlen_k = static_cast<int>(bnd_k >> 32) - k0;
    const int kv0      = j * BK;
    // A block past the sequence's keys has nothing to do. (No early return: the two bodies are
    // then a plain if / else, with nothing of one live across the other.)
    if(kv0 < seqlen_k)
    {
    // Bottom-right causal: query qp sees keys kv <= qp + causal_offset. The block's keys see
    // (part of) query tiles i0 .. T - 1; with no query at all the loop runs one tile of zeros.
    const int causal_offset = seqlen_k - seqlen_q;
    const int T             = (seqlen_q + BQ - 1) / BQ;
    int i0                  = 0;
    if(k.causal && kv0 - causal_offset > 0)
        i0 = (kv0 - causal_offset) / BQ;
    // This wave's 48 keys; rows past seqlen_k read as 0.
    const int row_w  = kv0 + w * WR;
    const int rows_w = seqlen_k - row_w < WR ? (seqlen_k - row_w > 0 ? seqlen_k - row_w : 0) : WR;
    const int c_kv   = row_w + lane % 16;

    // The tiles from if0 on are those this wave's keys all see under the causal mask; the ones
    // before take a mask. Nothing else does: rows past seqlen_k and past seqlen_q read as 0 (K, V;
    // Q, dO, LSE, D), and what they make stays in dK / dV rows that are not stored (keys) or is 0
    // (queries: dO and D - dP^T).
    int if0 = 0;
    if(k.causal && rows_w > 0)
    {
        const int need = row_w + rows_w - 1 - causal_offset;
        if0 = need > 0 ? (need + BQ - 1) / BQ : 0;
    }
    asm volatile("" : "+s"(if0));

    const bf16_t* q_ptr  = q_h + static_cast<long long>(q0) * k.stride_q_n;
    const bf16_t* do_ptr = do_h + static_cast<long long>(q0) * k.stride_do_n;
    // readfirstlane: as above (the lanes' offsets use the same products).
    const int x_rb[4] = {__builtin_amdgcn_readfirstlane(k.stride_q_n * 2),
                         __builtin_amdgcn_readfirstlane((r3q ? k.stride_q_n : k.stride_do_n) * 2),
                         __builtin_amdgcn_readfirstlane(k.stride_do_n * 2), 4};
    const char* x_src[4] = {reinterpret_cast<const char*>(q_ptr),
                            r3q ? reinterpret_cast<const char*>(q_ptr) : reinterpret_cast<const char*>(do_ptr),
                            reinterpret_cast<const char*>(do_ptr),
                            reinterpret_cast<const char*>(ld_h + q0)};
    int x_kb[4];
    gated_mla_bwd::i32x4_t x_rs[4];
#pragma unroll
    for(int d = 0; d < 4; d++)
    {
        x_kb[d] = BQ * x_rb[d];
        x_rs[d] = gated_mla_bwd::dma_rsrc(x_src[d] + static_cast<long long>(i0) * BQ * x_rb[d], 0);
        asm volatile("" : "+s"(x_kb[d]), "+s"(x_rs[d]));
    }
    auto x_sizes = [&](int t) {
        int rows = seqlen_q - t * BQ;
        asm("s_min_i32 %0, %0, %1\n\ts_max_i32 %0, %0, 0" : "+s"(rows) : "n"(BQ) : "scc");
#pragma unroll
        for(int d = 0; d < 4; d++)
        {
            int nb;
            asm("s_mul_i32 %0, %1, %2" : "=s"(nb) : "s"(rows), "s"(x_rb[d]));
            x_rs[d] = gated_mla_bwd::i32x4_t{x_rs[d][0], x_rs[d][1], nb, 0x00020000};
            asm volatile("" : "+s"(x_rs[d]));
        }
    };
    auto x_adv = [&](int d) __attribute__((always_inline)) {
        auto& r = x_rs[d];
        asm volatile("" : "+s"(r));
        unsigned long long a =
            static_cast<unsigned int>(r[0]) | static_cast<unsigned long long>(static_cast<unsigned int>(r[1])) << 32;
        a += static_cast<unsigned long long>(static_cast<unsigned int>(x_kb[d]));
        r = gated_mla_bwd::i32x4_t{static_cast<int>(a), static_cast<int>(a >> 32), r[2], r[3]};
        asm volatile("" : "+s"(r));
    };
    auto x_go = [&](int u, unsigned int lb) __attribute__((always_inline)) {
        const int d = u < 3 ? 0 : (u == 3 ? 1 : (u < XR - 1 ? 2 : 3));
        if(u == XR - 1)
            asm volatile("s_add_u32 m0, %2, %3\n\tbuffer_load_dword %0, %1, 0 offen lds"
                         :
                         : "v"(x_os[u]), "s"(x_rs[d]), "s"(lb), "s"(x_lds[u])
                         : "m0", "scc");
        else
            asm volatile("s_add_u32 m0, %2, %3\n\tbuffer_load_dwordx4 %0, %1, 0 offen lds"
                         :
                         : "v"(x_os[u]), "s"(x_rs[d]), "s"(lb), "s"(x_lds[u])
                         : "m0", "scc");
    };

    // The first two tiles to LDS (buffers 0 and 1), ahead of K and V: the memory pipeline is at
    // them while the 30 loads of K and V are issued.
    {
        unsigned int lb0 = lds_of(0), lb1 = lds_of(XB_SZ);
        asm volatile("" : "+s"(lb0), "+s"(lb1));
        x_sizes(i0);
#pragma unroll
        for(int u = 0; u < XR; u++)
            x_go(u, lb0);
#pragma unroll
        for(int d = 0; d < 4; d++)
            x_adv(d);
        x_sizes(i0 + 1);
#pragma unroll
        for(int u = 0; u < XR; u++)
            x_go(u, lb1);
        // Descriptor 0 goes a tile on past its rounds, the others in front of theirs.
        x_adv(0);
        x_sizes(i0 + 2);
    }
    // This wave's keys as the A operand of S^T and dP^T, in registers.
    auto g_kn = opus::make_gmem(kn_h + static_cast<long long>(k0 + row_w) * k.stride_k_n,
                                rec(static_cast<long long>(rows_w) * k.stride_k_n, 2));
    auto g_kr = opus::make_gmem(reinterpret_cast<const bf16_t*>(k.ptr_k_rot) + static_cast<long long>(k0 + row_w) * k.stride_k_rot_n,
                                rec(static_cast<long long>(rows_w) * k.stride_k_rot_n, 2));
    auto g_v  = opus::make_gmem(v_h + static_cast<long long>(k0 + row_w) * k.stride_v_n,
                                rec(static_cast<long long>(rows_w) * k.stride_v_n, 2));
    const auto v_kn = g_kn.template load<8>(opus::partition_layout_a<8>(mma_kn, opus::make_tuple(k.stride_k_n, 1_I), c_ab));
    const auto v_kr = g_kr.template load<8>(opus::partition_layout_a<8>(mma_kr, opus::make_tuple(k.stride_k_rot_n, 1_I), c_ab));
    const auto v_vw = g_v.template load<8>(opus::partition_layout_a<8>(mma_dp, opus::make_tuple(k.stride_v_n, 1_I), c_ab));
    // A of S^T is [M tile][K step][8]; nope and rope interleave per M tile.
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

    // dV's accumulators, zeroed behind the loads of K and V (dK's come zeroed from the kernel).
    f32x4_t acc_dv[MR][NV];
#pragma unroll
    for(int m = 0; m < MR; m++)
#pragma unroll
        for(int n = 0; n < NV; n++)
        {
            acc_dv[m][n] = f32x4_t{0.0f, 0.0f, 0.0f, 0.0f};
            asm volatile("" : "+a"(acc_dv[m][n]));
        }
    {
        // Loop-carried scalars, at the top of the iteration of S^T tile ts: lbc, lbp the LDS
        // address of tile ts - 1's buffer and of tile ts - 2's (the DMA's target, tile ts + 2
        // goes there); dlt the LDS addresses' step from tile ts - 1's buffer to tile ts's (from
        // tile ts's to the next past S^T unit 24); rc tile ts's buffer in the ring (0 .. 3), dn
        // the step from it to the next (-3 buffers past the last, without flags: single scalars
        // go anywhere); ra = seqlen_q - (ts + 4) BQ (negative: tile ts + 3 is not a full one),
        // fd = ts - if0 (negative: mask), z = ts - T.
        constexpr int SZB = XB_SZ * 2;
        static_assert(XNB == 4, "the ring's wrap below");
        int rc = 0, dn = 0;
        int dlt = -(XNB - 1) * SZB;
        unsigned int lbc = lds_of((XNB - 1) * XB_SZ), lbp = lds_of((XNB - 2) * XB_SZ);
        int ra = seqlen_q - (i0 + 4) * BQ, fd = i0 - if0, z = i0 - T;
        asm volatile("" : "+s"(rc), "+s"(dlt), "+s"(lbc), "+s"(lbp));
        asm volatile("" : "+s"(ra), "+s"(fd), "+s"(z));

        namespace sch = gated_mla_bwd_4ws;
        // A unit's VALU and read from its row of the schedule. The rows are read in constant
        // expressions and passed by value (the tables are globals of the device module
        // otherwise, whose loads do not fold).
        auto v_run = [&](sch::vop o0, sch::vop o1, sch::vop o2) __attribute__((always_inline)) {
            const sch::vop os[3] = {o0, o1, o2};
#pragma unroll
            for(int i = 0; i < 3; i++)
            {
                const int kd = os[i].k, e = os[i].e;
                if(kd == sch::V_LSE)
                    lse1(e);
                if(kd == sch::V_FMA)
                    fma1(e);
                if(kd == sch::V_EXP)
                    expa(e);
                if(kd == sch::V_SUB)
                    sub1(e);
                if(kd == sch::V_NEG)
                    neg1(e);
                if(kd == sch::V_MUL)
                    mul1(e);
                if(kd == sch::V_PP)
                    pack(v_sp, a_p, e);
                if(kd == sch::V_PD)
                    pack(v_dp, a_ds, e);
                if(kd == sch::V_AQ)
                    asm volatile("v_add_u32 %0, %1, %0" : "+v"(a_q) : "s"(dlt));
                if(kd == sch::V_ATQ)
                    asm volatile("v_add_u32 %0, %1, %2" : "=v"(a_tq) : "v"(a_q), "v"(c_tq));
                if(kd == sch::V_ADO)
                    asm volatile("v_add_u32 %0, %1, %0" : "+v"(a_do) : "s"(dlt));
                if(kd == sch::V_ATDO)
                    asm volatile("v_add_u32 %0, %1, %0" : "+v"(a_tdo) : "s"(dlt));
                if(kd == sch::V_ALD)
                    asm volatile("v_add_u32 %0, %1, %0" : "+v"(a_ld) : "s"(dlt));
            }
        };
        auto r_run = [&](sch::rop r) __attribute__((always_inline)) {
            if(r.k == sch::R_Q)
                bq[r.a % 3] = rd_q(r.a / KS, r.a % KS);
            if(r.k == sch::R_O)
                bdo[r.a % 3] = rd_do(r.a / KSV, r.a % KSV);
            if(r.k == sch::R_TQ)
                tb[r.a % 3][r.b] = rd_tq(r.a, r.b);
            if(r.k == sch::R_TO)
                tb[r.a % 3][r.b] = rd_to(r.a, r.b);
            if(r.k == sch::R_LS)
                lse4[r.a] = at_f(a_ld).template load<4>(16 * r.a);
            if(r.k == sch::R_DD)
                d4[r.a] = at_f(a_ld).template load<4>(64 + 16 * r.a);
        };

        // S^T of tile ts, 36 units: fragment f = u / 3 (N tile f / 6, K step f % 6) for M tile
        // u % 3. It follows the barrier, behind which VALU stands still for some 100 cycles while
        // MFMA, LDS reads, DMA and scalars run on: its first 7 units carry reads, two DMA rounds
        // and scalar pairs (the DMA descriptors' 64-bit steps). Single scalars: one to a unit
        // that has no wait. Returns a negative number if the sizes of tile ts + 3 are to be set
        // or S^T is to be masked (rare_fix).
        auto s_block = [&]() __attribute__((always_inline)) {
            int rare = 0;
            opus::static_for<NT * KS * MR>([&](auto U) {
                constexpr int u = decltype(U)::value, f = u / MR, j = u % MR, tn = f / KS, ks = f % KS;
                constexpr sch::vop o0 = sch::S_V[u][0], o1 = sch::S_V[u][1], o2 = sch::S_V[u][2];
                constexpr sch::rop rs = sch::S_R[u];
                constexpr int xr = sch::S_X[u];
                mm(v_kw, j * KS + ks, bq[f % 3], v_sp, j * NT + tn, ks == 0 ? 0 : (ks == KS - 1 ? 2 : 1));
                if(u % 3 == 2 && u <= 32)
                    touch(bq[(u + 1) / 3 % 3]);
                if(u == 34)
                    touch(bdo[0]);
                // The DMA's descriptors a tile on: 1 .. 3 in front of their rounds, 0 past its.
                if(u == 0)
                    x_adv(1);
                if(u == 3)
                    x_adv(2);
                if(u == 6)
                    x_adv(3);
                if(u == 33)
                    x_adv(0);
                // The ring a buffer on: rc = (rc + 1) & 3, dn = rc == 0 ? -3 buffers : 1 buffer.
                if(u == 2)
                    asm volatile("s_add_i32 %0, %0, 1" : "+s"(rc) : : "scc");
                if(u == 5)
                    asm volatile("s_and_b32 %0, %0, 3" : "+s"(rc) : : "scc");
                if(u == 9)
                    asm volatile("s_sub_i32 %0, %1, 1" : "=s"(dn) : "s"(rc) : "scc");
                if(u == 12)
                    asm volatile("s_ashr_i32 %0, %0, 31" : "+s"(dn) : : "scc");
                if(u == 15)
                    asm volatile("s_and_b32 %0, %0, %1" : "+s"(dn) : "n"(-XNB * SZB) : "scc");
                if(u == 18)
                    asm volatile("s_add_i32 %0, %0, %1" : "+s"(dn) : "n"(SZB) : "scc");
                // Past the last round: the DMA's target and the address step a tile on.
                if(u == 21)
                    asm volatile("s_mov_b32 %0, %1" : "=s"(lbp) : "s"(lbc));
                if(u == 22)
                    asm volatile("s_add_u32 %0, %0, %1" : "+s"(lbc) : "s"(dlt) : "scc");
                if(u == 24)
                    asm volatile("s_mov_b32 %0, %1" : "=s"(dlt) : "s"(dn));
                // The counters on; the rare cases' sign from the S^T tile's ra and fd first.
                if(u == 25)
                    asm volatile("s_add_i32 %0, %0, 1" : "+s"(z) : : "scc");
                if(u == 27)
                    asm volatile("s_or_b32 %0, %1, %2" : "=s"(rare) : "s"(ra), "s"(fd) : "scc");
                if(u == 30)
                    asm volatile("s_sub_i32 %0, %0, %1" : "+s"(ra) : "n"(BQ) : "scc");
                if(u == 31)
                    asm volatile("s_add_i32 %0, %0, 1" : "+s"(fd) : : "scc");
                r_run(rs);
                if(xr >= 0)
                    x_go(xr >= 0 ? xr : 0, lbp);
                v_run(o0, o1, o2);
                if(ks == KS - 1 || u == 18)
                    touch(c_old);
                fence();
            });
            return rare;
        };
        // The rare cases: tile ts + 3 is not a full one (its sizes), this wave's keys do not all
        // see tile ts under the causal mask: the elements masked take P = 0 where the exp is made
        // already, else an S^T (or S^T scale - LSE) that exps to 0.
        auto rare_fix = [&](int ts) {
            if(seqlen_q - (ts + 4) * BQ < 0)
                x_sizes(ts + 3);
            if(ts < if0)
            {
                // Element e is key row_w + 16 (e / 8) + lane % 16 and query BQ ts + 16 (e / 4 % 2)
                // + 4 (lane / 16) + e % 4, seen if query + causal_offset >= key: xm[M tile] is
                // the first query of the tile the lane's key sees, less the lane's 4 (lane / 16).
                int xm[MR];
#pragma unroll
                for(int m = 0; m < MR; m++)
                    xm[m] = c_kv + 16 * m - causal_offset - ts * BQ - c_q;
                // The mask, four elements (one C tile) at a time: four compares into scalar pairs, then
                // the four selects. A select then stands three instructions behind its compare; left
                // to the compiler, each pair takes two s_nop between them (a rare iteration was 400
                // cycles of masking).
                float m_z = 0.0f, m_n = -1e30f;
                pin(m_z);
                pin(m_n);
                opus::static_for<NS / 4>([&](auto G) {
                    constexpr int gt = decltype(G)::value, c0 = 16 * (gt % NT), e0 = 4 * gt;
                    float x0 = v_sp[e0], x1 = v_sp[e0 + 1], x2 = v_sp[e0 + 2], x3 = v_sp[e0 + 3];
                    unsigned long long t0, t1, t2, t3;
                    asm volatile("v_cmp_le_i32 %4, %8, %13\n\tv_cmp_le_i32 %5, %8, %14\n\tv_cmp_le_i32 %6, %8, %15\n\tv_cmp_le_i32 %7, %8, %16\n\t"
                                 "v_cndmask_b32 %0, %9, %0, %4\n\tv_cndmask_b32 %1, %10, %1, %5\n\tv_cndmask_b32 %2, %11, %2, %6\n\tv_cndmask_b32 %3, %12, %3, %7"
                                 : "+v"(x0), "+v"(x1), "+v"(x2), "+v"(x3), "=&s"(t0), "=&s"(t1), "=&s"(t2), "=&s"(t3)
                                 : "v"(xm[gt / NT]), "v"((sch::S_EXPD >> e0 & 1) ? m_z : m_n), "v"((sch::S_EXPD >> (e0 + 1) & 1) ? m_z : m_n),
                                   "v"((sch::S_EXPD >> (e0 + 2) & 1) ? m_z : m_n), "v"((sch::S_EXPD >> (e0 + 3) & 1) ? m_z : m_n),
                                   "n"(c0), "n"(c0 + 1), "n"(c0 + 2), "n"(c0 + 3));
                    v_sp[e0]     = x0;
                    v_sp[e0 + 1] = x1;
                    v_sp[e0 + 2] = x2;
                    v_sp[e0 + 3] = x3;
                });
            }
        };
        // dP^T(ts), 24 units: fragment f = u / 3 (N tile f / 4, K step f % 4) for M tile u % 3.
        auto dp_block = [&]() __attribute__((always_inline)) {
            opus::static_for<NT * KSV * MR>([&](auto U) {
                constexpr int u = decltype(U)::value, f = u / MR, j = u % MR, tn = f / KSV, ks = f % KSV;
                constexpr sch::vop o0 = sch::P_V[u][0], o1 = sch::P_V[u][1], o2 = sch::P_V[u][2];
                constexpr sch::rop rs = sch::P_R[u];
                mm(v_vw, j * KSV + ks, bdo[f % 3], v_dp, j * NT + tn, ks == 0 ? 0 : (ks == KSV - 1 ? 2 : 1),
                   GMLA_DP_MINUS_D && ks == 0 ? &d4[tn] : nullptr);
                if(u % 3 == 2 && u <= 20)
                    touch(bdo[(u + 1) / 3 % 3]);
                if(u == 23)
                    touch2(tb[0][0], tb[0][1]);
                r_run(rs);
                v_run(o0, o1, o2);
                if(ks == KSV - 1 || u == 0 || u == 12)
                    touch(c_old);
                fence();
            });
        };
        // dK of the tile before (of the last tile in the tail), 36 units of 3 per N tile; the
        // first two of an N tile read Q^T two N tiles ahead (the last two N tiles' dO^T's first
        // two, for the dV that follows).
        auto dk_block = [&](bool tail) __attribute__((always_inline)) {
            opus::static_for<NG * MR>([&](auto U) {
                constexpr int q = decltype(U)::value, gn = q / MR, m = q % MR;
                constexpr sch::vop o0 = sch::K_V[q][0], o1 = sch::K_V[q][1], o2 = sch::K_V[q][2];
                if(gn < NV)
                    acc1(acc_dkn[m][gn < NV ? gn : 0], gated_mla_bwd::frag8(a_ds, m), b_tr(tb[gn % 3]));
                else
                    acc1(acc_dkr[m][gn >= NV ? gn - NV : 0], gated_mla_bwd::frag8(a_ds, m), b_tr(tb[gn % 3]));
                if(m == 2 && gn + 1 < NG)
                    touch2(tb[(gn + 1) % 3][0], tb[(gn + 1) % 3][1]);
                if(!tail && m == 2 && gn == NG - 1)
                    touch2(tb[0][0], tb[0][1]);
                if(m < 2 && gn + 2 < NG)
                    tb[(gn + 2) % 3][m] = rd_tq(gn + 2, m);
                if(!tail && m < 2 && gn + 2 >= NG)
                    tb[(gn + 2 - NG) % 3][m] = rd_to(gn + 2 - NG, m);
                if(!tail)
                    v_run(o0, o1, o2);
                if(!tail && q == 0)
                    touch(c_old);
                fence();
            });
        };
        // dV(ts), 24 units of 3 per N tile; the first two of an N tile read dO^T two N tiles
        // ahead (N tile 6's the next S^T's first two fragments). The waits before the barrier:
        // this wave's DMA loads, issued in this iteration's S^T block, are in.
        auto dv_block = [&]() __attribute__((always_inline)) {
            opus::static_for<NV * MR>([&](auto U) {
                constexpr int v = decltype(U)::value, gn = v / MR, m = v % MR;
                constexpr sch::vop o0 = sch::V_V[v][0], o1 = sch::V_V[v][1], o2 = sch::V_V[v][2];
                acc1(acc_dv[m][gn], gated_mla_bwd::frag8(a_p, m), b_tr(tb[gn % 3]));
                if(m == 2 && gn + 1 < NV)
                    touch2(tb[(gn + 1) % 3][0], tb[(gn + 1) % 3][1]);
                if(v == 19)
                    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                if(v == NV * MR - 1)
                    touch(bq[0]);
                if(m < 2 && gn + 2 < NV)
                    tb[(gn + 2) % 3][m] = rd_to(gn + 2, m);
                if(m < 2 && gn == NV - 2)
                    bq[m] = rd_q(0, m);  // the next S^T's first two fragments
                // D of the next tile's N tile vdd (GMLA_DP_MINUS_D): LSE's and D's address is on it.
                constexpr int vdd = sch::V_DD[v];
                if(vdd >= 0)
                    d4[vdd >= 0 ? vdd : 0] = at_f(a_ld).template load<4>(64 + 16 * vdd);
                v_run(o0, o1, o2);
                fence();
            });
        };

        // K and V are in registers and this wave's loads of the first two tiles in LDS (0x0f70 is
        // s_waitcnt vmcnt(0), as an instruction the compiler sees: it then waits for none of them
        // inside the loop); every wave's are behind the barrier.
        __builtin_amdgcn_s_waitcnt(0x0f70);
        barrier();
        bq[0] = rd_q(0, 0);
        bq[1] = rd_q(0, 1);
        touch(bq[0]);
#if GMLA_DP_MINUS_D
        // -D of the first tile's queries, where its dP^T's accumulators start; the loop reads each
        // next tile's D in its dV block.
#pragma unroll
        for(int tn = 0; tn < NT; tn++)
            d4[tn] = at_f(a_ld).template load<4>(64 + 16 * tn);
#pragma unroll
        for(int x = 0; x < 4 * NT; x++)
            neg1(x);
#endif
        // The loop starts on a 64-byte boundary: where its code lies decides whether the branch
        // of the rare cases (not taken) costs 4 cycles or 20.
        asm volatile(".p2align 6");
        do
        {
            // Every wave is past dK's reads of tile ts - 2, whose buffer this S^T block's DMA
            // overwrites, and has its loads of tile ts + 1 in.
            barrier();
            const int rare = s_block();
            if(__builtin_expect(rare < 0, 0))
                rare_fix(z - 1 + T);
            dp_block();
            dk_block(false);
            dv_block();
        } while(__builtin_expect(z < 0, 1));
        // The last tile's dK: its dS to bf16, its first two N tiles' fragments of Q^T (whose
        // address the last iteration left on it).
#pragma unroll
        for(int jp = 0; jp < NS / 2; jp++)
            pack(v_dp, a_ds, jp);
#pragma unroll
        for(int x = 0; x < 4; x++)
            tb[x / 2][x % 2] = rd_tq(x / 2, x % 2);
        dk_block(true);
    }
    // The loads still out go to LDS, which the epilogue uses.
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

    // dK nope and dV into the kv_b_proj layout, dK rope per head in fp32. The C fragments hold
    // 4 columns of 16 rows per instruction, so each wave transposes its 48 rows through LDS first
    // and stores whole rows; rows past seqlen_k fall outside the descriptors and are dropped.
    mfma_acc_drain();
    // The accumulators are read from here on: the compiler does not know the asm MFMAs' latency
    // and would read one right behind the MFMA that writes it.
#pragma unroll
    for(int m = 0; m < MR; m++)
    {
#pragma unroll
        for(int n = 0; n < NV; n++)
            asm volatile("" : "+a"(acc_dv[m][n]), "+a"(acc_dkn[m][n]));
#pragma unroll
        for(int n = 0; n < NR; n++)
            asm volatile("" : "+a"(acc_dkr[m][n]));
    }
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
                *reinterpret_cast<bvec<4>*>(o)          = opus::cast<bf16_t>(acc_dkn[m][n] * k.softmax_scale);
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
                *reinterpret_cast<f32x4_t*>(s_r + (16 * m + c_row) * LD_R + 16 * n + c_col) = acc_dkr[m][n] * k.softmax_scale;
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
}

// The 4-wave dQ kernel's schedule: what goes behind the MFMA of each unit of its three blocks (S 36 units,
// dP 24, dQ 36). Element e of a tile's 24 S / dP values per lane is C tile (M tile e / 8, N tile e / 4 % 2).
//   V_PACK j   elements 2 j, 2 j + 1 of dS of the tile before to bf16 (the A operand of the dQ block)
//   V_FMA e    S scale - LSE        V_EXP e   its exp, P (fills a unit)
//   V_SUB e    dP - D               V_MUL e   P (dP - D), dS in fp32 (GMLA_DP_MINUS_D: dP starts at -D, no V_SUB)
//   V_AKN, V_AKR, V_AV   K nope's, K rope's, V's LDS address a tile on; V_ATKN, V_ATKR  K^T's from V's
//   R_K f, R_V f   S's / dP's B fragment f (N tile, K step in N-tile-major order); R_T g, h   K^T's N tile g, half h
//   S_X        the DMA round of an S unit, or -1
// A unit has room for two of VALU and read behind its MFMA (an exp is two, a DMA round both); the S block's
// first 7 units take no VALU (it stands still behind the barrier). The dQ block's own reads are regular and
// not in a table. S_EXPD: bit e set if element e's exp is in the S block (the mask of a tile not seen in full
// is applied after it: those take P = 0, the others an S that exps to 0). The kernel reads the tables in
// constant expressions only.
namespace gated_mla_bwd_dq4s {
enum : signed char { V_NONE, V_PACK, V_FMA, V_EXP, V_SUB, V_MUL, V_AKN, V_AKR, V_AV, V_ATKN, V_ATKR };
enum : signed char { R_NONE, R_K, R_V, R_T };
struct vop
{
    signed char k, e;
};
struct rop
{
    signed char k, a, b;
};
#if GMLA_DP_MINUS_D
inline constexpr vop S_V[36][3] = {
    {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PACK, 0}, {V_PACK, 1}, {V_NONE, 0}},
    {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PACK, 2}, {V_PACK, 3}, {V_NONE, 0}},
    {{V_PACK, 4}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PACK, 5}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_PACK, 6}, {V_PACK, 7}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PACK, 8}, {V_PACK, 9}, {V_NONE, 0}},
    {{V_FMA, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 1}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 2}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_FMA, 3}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 8}, {V_FMA, 9}, {V_NONE, 0}}, {{V_EXP, 0}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_FMA, 10}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 1}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 2}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_FMA, 11}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 16}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 3}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_FMA, 17}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 8}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 18}, {V_FMA, 19}, {V_NONE, 0}},
    {{V_EXP, 9}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 10}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}},
};
inline constexpr rop S_R[36] = {
    {R_K, 2, 0}, {R_NONE, 0, 0}, {R_T, 0, 0}, {R_K, 3, 0}, {R_NONE, 0, 0}, {R_T, 0, 1},
    {R_K, 4, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_K, 5, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
    {R_K, 6, 0}, {R_NONE, 0, 0}, {R_K, 7, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
    {R_K, 8, 0}, {R_T, 1, 0}, {R_T, 1, 1}, {R_K, 9, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
    {R_K, 10, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_K, 11, 0}, {R_V, 0, 0}, {R_NONE, 0, 0},
    {R_V, 1, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
};
inline constexpr signed char S_X[36] = {-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
inline constexpr vop P_V[24][3] = {
    {{V_PACK, 11}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 4}, {V_FMA, 5}, {V_NONE, 0}}, {{V_FMA, 6}, {V_FMA, 7}, {V_NONE, 0}},
    {{V_FMA, 12}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 13}, {V_FMA, 14}, {V_NONE, 0}}, {{V_FMA, 15}, {V_FMA, 20}, {V_NONE, 0}},
    {{V_FMA, 21}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 22}, {V_FMA, 23}, {V_NONE, 0}}, {{V_AKN, 0}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_EXP, 11}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 16}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 17}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_PACK, 10}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 0}, {V_MUL, 1}, {V_NONE, 0}}, {{V_MUL, 2}, {V_MUL, 3}, {V_NONE, 0}},
    {{V_MUL, 8}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 9}, {V_MUL, 10}, {V_NONE, 0}}, {{V_MUL, 11}, {V_MUL, 16}, {V_NONE, 0}},
    {{V_MUL, 17}, {V_AKR, 0}, {V_NONE, 0}}, {{V_EXP, 18}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 19}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_EXP, 4}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 5}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 6}, {V_NONE, 0}, {V_NONE, 0}},
};
inline constexpr rop P_R[24] = {
    {R_V, 2, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_V, 3, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
    {R_V, 4, 0}, {R_NONE, 0, 0}, {R_V, 5, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
    {R_V, 6, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_V, 7, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
    {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
};
inline constexpr vop Q_V[36][3] = {
    {{V_MUL, 18}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 19}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 7}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_MUL, 4}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 5}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 12}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_MUL, 6}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 7}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 13}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_MUL, 12}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 13}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 14}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_MUL, 14}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 15}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_MUL, 15}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 20}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_MUL, 20}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 21}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_MUL, 21}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 22}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_MUL, 22}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 23}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_MUL, 23}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_ATKN, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_ATKR, 0}, {V_AV, 0}, {V_NONE, 0}},
};
inline constexpr unsigned int S_EXPD = 0x00070f;
#else
inline constexpr vop S_V[36][3] = {
    {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PACK, 0}, {V_PACK, 1}, {V_NONE, 0}},
    {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PACK, 2}, {V_PACK, 3}, {V_NONE, 0}},
    {{V_PACK, 4}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PACK, 5}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_PACK, 6}, {V_PACK, 7}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_PACK, 8}, {V_PACK, 9}, {V_NONE, 0}},
    {{V_FMA, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 1}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 2}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_FMA, 3}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 8}, {V_FMA, 9}, {V_NONE, 0}}, {{V_EXP, 0}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_FMA, 10}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 1}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 2}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_FMA, 11}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 16}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 3}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_FMA, 17}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 8}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 18}, {V_FMA, 19}, {V_NONE, 0}},
    {{V_EXP, 9}, {V_AKR, 0}, {V_NONE, 0}}, {{V_EXP, 10}, {V_NONE, 0}, {V_NONE, 0}}, {{V_NONE, 0}, {V_NONE, 0}, {V_NONE, 0}},
};
inline constexpr rop S_R[36] = {
    {R_K, 2, 0}, {R_NONE, 0, 0}, {R_T, 0, 0}, {R_K, 3, 0}, {R_NONE, 0, 0}, {R_T, 0, 1},
    {R_K, 4, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_K, 5, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
    {R_K, 6, 0}, {R_NONE, 0, 0}, {R_K, 7, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
    {R_K, 8, 0}, {R_T, 1, 0}, {R_T, 1, 1}, {R_K, 9, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
    {R_K, 10, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_K, 11, 0}, {R_V, 0, 0}, {R_NONE, 0, 0},
    {R_V, 1, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
};
inline constexpr signed char S_X[36] = {-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
inline constexpr vop P_V[24][3] = {
    {{V_PACK, 11}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 4}, {V_FMA, 5}, {V_NONE, 0}}, {{V_FMA, 6}, {V_FMA, 7}, {V_NONE, 0}},
    {{V_FMA, 12}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 13}, {V_FMA, 14}, {V_NONE, 0}}, {{V_FMA, 15}, {V_FMA, 20}, {V_NONE, 0}},
    {{V_FMA, 21}, {V_NONE, 0}, {V_NONE, 0}}, {{V_FMA, 22}, {V_FMA, 23}, {V_NONE, 0}}, {{V_AKN, 0}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_EXP, 11}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 16}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 17}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_PACK, 10}, {V_NONE, 0}, {V_NONE, 0}}, {{V_SUB, 0}, {V_SUB, 1}, {V_NONE, 0}}, {{V_SUB, 2}, {V_SUB, 3}, {V_NONE, 0}},
    {{V_SUB, 8}, {V_NONE, 0}, {V_NONE, 0}}, {{V_SUB, 9}, {V_SUB, 10}, {V_NONE, 0}}, {{V_SUB, 11}, {V_SUB, 16}, {V_NONE, 0}},
    {{V_SUB, 17}, {V_SUB, 18}, {V_NONE, 0}}, {{V_SUB, 19}, {V_EXP, 18}, {V_NONE, 0}}, {{V_EXP, 19}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_EXP, 4}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 5}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 6}, {V_NONE, 0}, {V_NONE, 0}},
};
inline constexpr rop P_R[24] = {
    {R_V, 2, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_V, 3, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
    {R_V, 4, 0}, {R_NONE, 0, 0}, {R_V, 5, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
    {R_V, 6, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_V, 7, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
    {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0}, {R_NONE, 0, 0},
};
inline constexpr vop Q_V[36][3] = {
    {{V_MUL, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 1}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 7}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_MUL, 2}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 3}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 12}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_MUL, 8}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 9}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 13}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_MUL, 10}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 11}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 14}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_MUL, 16}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 17}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 15}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_MUL, 18}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 19}, {V_NONE, 0}, {V_NONE, 0}}, {{V_EXP, 20}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_SUB, 4}, {V_SUB, 5}, {V_NONE, 0}}, {{V_SUB, 6}, {V_SUB, 7}, {V_NONE, 0}}, {{V_EXP, 21}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_SUB, 12}, {V_SUB, 13}, {V_NONE, 0}}, {{V_SUB, 14}, {V_SUB, 15}, {V_NONE, 0}}, {{V_EXP, 22}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_SUB, 20}, {V_SUB, 21}, {V_NONE, 0}}, {{V_SUB, 22}, {V_SUB, 23}, {V_NONE, 0}}, {{V_EXP, 23}, {V_NONE, 0}, {V_NONE, 0}},
    {{V_MUL, 4}, {V_MUL, 5}, {V_NONE, 0}}, {{V_MUL, 6}, {V_MUL, 7}, {V_NONE, 0}}, {{V_MUL, 12}, {V_MUL, 13}, {V_NONE, 0}},
    {{V_MUL, 14}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 15}, {V_NONE, 0}, {V_NONE, 0}}, {{V_MUL, 20}, {V_MUL, 21}, {V_NONE, 0}},
    {{V_MUL, 22}, {V_MUL, 23}, {V_NONE, 0}}, {{V_ATKN, 0}, {V_NONE, 0}, {V_NONE, 0}}, {{V_ATKR, 0}, {V_AV, 0}, {V_NONE, 0}},
};
inline constexpr unsigned int S_EXPD = 0x00070f;
#endif
}  // namespace gated_mla_bwd_dq4s

// dQ pass of the 4-wave backward, one wave per SIMD, on the recipe of the dK / dV pass's loop: wave w
// owns 48 queries; their Q and dO (A of S and dP) stay in VGPRs, LSE and D with them, dQ in AGPRs.
// Per tile of 32 keys (K nope, K rope and V go global -> LDS by DMA into 4 buffers) an iteration is
// 96 units of one MFMA and what fits behind it (a scalar or wait and two of LDS read or VALU):
//   barrier             every wave is past its reads of tile ts - 2, tile ts + 1 is whole
//   S(ts)      36 MFMA  the 6 DMA rounds of tile ts + 2 into tile ts - 2's buffer, K's fragments
//                       two ahead, dS of tile ts - 1 to bf16, the loop's scalars, S scale - LSE
//                       and the exp of elements whose C tiles are done mid block
//   dP(ts)     24 MFMA  the other S scale - LSE, exps, dP - D of the C tiles done mid block
//   dQ(ts - 1) 36 MFMA  K^T's fragments two N tiles ahead; the other exps and dP - D, and
//                       dS = P (dP - D) in fp32
// S and dP go N tile by N tile (a B fragment serves three consecutive MFMAs, one per M tile), so
// half their C tiles are done mid block. dQ runs a tile behind so that the VALU work of dS has
// dP's and dQ's MFMAs to go behind. dS = P (dP - D): dP's accumulators start at -D
// (GMLA_DP_MINUS_D), or at 0 with D subtracted as the pair pass and the 8-wave dQ kernel do it,
// bit for bit.
__device__ __attribute__((always_inline)) inline void gated_mla_bwd_dq_4w_body(
    const gated_mla_bwd_4w_kargs& k, gated_mla_bwd::bf16_t* smem, const int lane, const int w, const int h, const int blk,
    unsigned long long bnd_q, unsigned long long bnd_k, gated_mla_bwd::f32x4_t (&z)[3][12])
{
    using namespace gated_mla_bwd_dq4;
    using gated_mla_bwd::bf16_t;
    using gated_mla_bwd::bvec;
    using gated_mla_bwd::f32x4_t;
    using opus::number;
    using opus::operator""_I;
    using u32x2_t = gated_mla_bwd::u32x2_t;
    using u32x4_t = gated_mla_bwd::u32x4_t;
    constexpr int MR = WQ / 16, NT = KC / 16;
    constexpr int KS = D_QK / 32, KS_N = D_NOPE / 32, KSV = D_V / 32, NG = D_QK / 16, NGN = D_NOPE / 16;
    // dQ, [M tile][N tile of 16 columns], in AGPRs: the kernel's zeros. The long query blocks go
    // first.
    static_assert(MR == 3 && NG == 12, "the kernel's zeroed tiles");
    f32x4_t acc[MR][NG];
#pragma unroll
    for(int m = 0; m < MR; m++)
#pragma unroll
        for(int n = 0; n < NG; n++)
            acc[m][n] = z[m][n];
    const int qb = k.n_q_blocks - 1 - blk;

    // The tensors' bases at this head.
    const bf16_t* q_h  = reinterpret_cast<const bf16_t*>(k.ptr_q) + static_cast<long long>(h) * k.stride_q_h;
    const bf16_t* do_h = reinterpret_cast<const bf16_t*>(k.ptr_do) + static_cast<long long>(h) * k.stride_do_h;
    const bf16_t* kn_h = reinterpret_cast<const bf16_t*>(k.ptr_k_nope) + static_cast<long long>(h) * k.stride_k_h;
    const bf16_t* v_h  = reinterpret_cast<const bf16_t*>(k.ptr_v) + static_cast<long long>(h) * k.stride_v_h;
    bf16_t* dq_h       = reinterpret_cast<bf16_t*>(k.ptr_dq) + static_cast<long long>(h) * k.stride_dq_h;
    const float* lse_h = k.ptr_lse + static_cast<long long>(h) * k.stride_lse_h;
    const float* d_h   = k.ptr_d + static_cast<long long>(h) * k.stride_lse_h;
    asm volatile("" : "+s"(q_h), "+s"(do_h), "+s"(kn_h), "+s"(v_h), "+s"(dq_h), "+s"(lse_h), "+s"(d_h));

    auto make_mma = [](auto e_m, auto e_n, auto e_k) {
        return opus::make_tiled_mma<bf16_t, bf16_t, float>(
            opus::seq<decltype(e_m)::value, decltype(e_n)::value, decltype(e_k)::value>{},
            opus::seq<1, 1, 1>{},
            opus::seq<16, 16, 32>{},
            opus::mfma_adaptor_swap_ab{});
    };
    auto mma_s  = make_mma(number<MR>{}, number<NT>{}, number<KS>{});   // Q's A layout: [M tile][K step][8]
    auto mma_dp = make_mma(number<MR>{}, number<NT>{}, number<KSV>{});  // dO's A layout
    auto mma_k1 = make_mma(number<MR>{}, number<NT>{}, number<1>{});
    auto mma1   = make_mma(number<1>{}, number<1>{}, number<1>{});
    const auto c_ab = opus::make_tuple(0_I, lane % mma_k1.grpm_a, 0_I, lane / mma_k1.grpm_a);
    constexpr int NS = decltype(mma_k1)::tile_c_len;
    static_assert(NS == MR * NT * 4 && NS == 24 && MR * NG == 36 && NG % 3 == 0, "the units below");
    auto rec = [](long long elems, int elem_bytes) -> unsigned int {
        const long long b = elems > 0 ? elems * elem_bytes : 0;
        return b >= 0xffffffffLL ? 0xffffffffu : static_cast<unsigned int>(b);
    };
    auto pin   = [](auto& x) { asm volatile("" : "+v"(x)); };
    auto here  = [](auto& x) { asm volatile("" : "+s"(x)); };
    auto fence = [] { __builtin_amdgcn_sched_barrier(0); };
    auto touch  = [](const auto& x) { asm volatile("" : : "v"(x)); };
    auto touch2 = [](const auto& x, const auto& y) { asm volatile("" : : "v"(x), "v"(y)); };
    auto read_first = [] {
        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
        __builtin_amdgcn_sched_group_barrier(0x002, 1, 0);
    };
    auto barrier = [&] {
        fence();
        __builtin_amdgcn_s_barrier();
        fence();
    };

    float sc2 = k.softmax_scale * 1.44269504089f;  // the scale times log2 e, in a VGPR for the asm fma
    pin(sc2);
    auto lds_of = [&](int b) { return gated_mla_bwd::lds_addr(smem + b); };

    // DMA: a tile is 9 loads of 1 KB of K nope, 5 of K rope, 9 of V (slot s of a region is row
    // s / SPR, chunk s % SPR; the padding chunks read out of bounds, as 0). Wave w issues loads
    // w + NW u, u = 0 .. 5 (wave 3's last repeats its fifth). Rounds 0, 1 load K nope, 4, 5 V;
    // round 2 is K nope for wave 0 and K rope for the others, round 3 K rope for waves 0, 1 and V
    // for 2, 3: four descriptors, each pointing at the tile to load.
    constexpr int SPR_N = LD_KN * 2 / 16, SPR_R = LD_KR * 2 / 16, SPR_V = LD_V * 2 / 16;
    constexpr int J_N = KC * SPR_N / 64, J_R = KC * SPR_R / 64, J_V = KC * SPR_V / 64, XR = 6;
    static_assert(KC * SPR_N % 64 == 0 && KC * SPR_R % 64 == 0 && KC * SPR_V % 64 == 0 && J_N == 9 && J_R == 5 && J_V == 9 && NW == 4,
                  "the rounds below");
    constexpr int OOB = 1 << 30;
    // The lanes' byte offsets of load jj of a region (jj wave-uniform; the divisors are constants).
    auto os_n = [&](int jj) {
        const unsigned int sl = 64u * jj + lane, c = sl % SPR_N;
        const int in = -static_cast<int>(c < D_NOPE / 8);  // all ones in a chunk of the row, 0 in its padding
        return (static_cast<int>(sl / SPR_N * (k.stride_k_n * 2) + c * 16) & in) | (OOB & ~in);
    };
    auto os_r = [&](int jj) {
        const unsigned int sl = 64u * jj + lane, c = sl % SPR_R;
        const int in = -static_cast<int>(c < D_ROT / 8);  // all ones in a chunk of the row, 0 in its padding
        return (static_cast<int>(sl / SPR_R * (k.stride_k_rot_n * 2) + c * 16) & in) | (OOB & ~in);
    };
    auto os_v = [&](int jj) {
        const unsigned int sl = 64u * jj + lane, c = sl % SPR_V;
        const int in = -static_cast<int>(c < D_V / 8);  // all ones in a chunk of the row, 0 in its padding
        return (static_cast<int>(sl / SPR_V * (k.stride_v_n * 2) + c * 16) & in) | (OOB & ~in);
    };
    // readfirstlane: the compiler makes these selects in VALU, with the lanes' offsets'.
    auto u_lds = [](int x) { return __builtin_amdgcn_readfirstlane(x); };
    const int j5 = w == 3 ? 5 : w + 6;
    int x_lds[XR], x_os[XR];
    x_lds[0] = u_lds(OFF_KN * 2 + w * 1024);
    x_os[0]  = os_n(w);
    x_lds[1] = u_lds(OFF_KN * 2 + (w + 4) * 1024);
    x_os[1]  = os_n(w + 4);
    x_lds[2] = u_lds(w == 0 ? OFF_KN * 2 + 8 * 1024 : OFF_KR * 2 + (w - 1) * 1024);
    x_os[2]  = w == 0 ? os_n(8) : os_r(w - 1);
    x_lds[3] = u_lds(w < 2 ? OFF_KR * 2 + (w + 3) * 1024 : OFF_V * 2 + (w - 2) * 1024);
    x_os[3]  = w < 2 ? os_r(w + 3) : os_v(w - 2);
    x_lds[4] = u_lds(OFF_V * 2 + (w + 2) * 1024);
    x_os[4]  = os_v(w + 2);
    x_lds[5] = u_lds(OFF_V * 2 + j5 * 1024);
    x_os[5]  = os_v(j5);
#pragma unroll
    for(int u = 0; u < XR; u++)
    {
        asm volatile("" : "+s"(x_lds[u]));
        pin(x_os[u]);
    }

    // Fragments of one K step or N tile, at a per-lane LDS address plus an immediate.
    auto lay_b     = [&](auto ld, const auto& c) { return opus::partition_layout_b<8>(mma_k1, opus::make_tuple(ld, 1_I), c); };
    const auto c_0 = opus::make_tuple(0_I, 0_I, 0_I, 0_I);
    const auto of_kn = opus::layout_to_offsets<8>(lay_b(number<LD_KN>{}, c_0));
    const auto of_kr = opus::layout_to_offsets<8>(lay_b(number<LD_KR>{}, c_0));
    const auto of_v  = opus::layout_to_offsets<8>(lay_b(number<LD_V>{}, c_0));
    int ob_kn  = (opus::layout_to_offsets<8>(lay_b(number<LD_KN>{}, c_ab))[0] - of_kn[0]) * 2;
    int ob_kr  = (opus::layout_to_offsets<8>(lay_b(number<LD_KR>{}, c_ab))[0] - of_kr[0]) * 2;
    int ob_v   = (opus::layout_to_offsets<8>(lay_b(number<LD_V>{}, c_ab))[0] - of_v[0]) * 2;
    int ob_tkn = gated_mla_bwd::tr_c_order_lane<LD_KN>(lane) * 2;
    int ob_tkr = gated_mla_bwd::tr_c_order_lane<LD_KR>(lane) * 2;
    pin(ob_kn);
    pin(ob_kr);
    pin(ob_v);
    // K^T's per-lane addresses are V's plus cv_n, cv_r.
    int cv_n = (OFF_KN - OFF_V) * 2 + ob_tkn - ob_v, cv_r = (OFF_KR - OFF_V) * 2 + ob_tkr - ob_v;
    pin(cv_n);
    pin(cv_r);
    auto at   = [](unsigned int a) { return opus::make_smem(reinterpret_cast<bf16_t*>(static_cast<__UINTPTR_TYPE__>(a))); };
    auto addr = [&](int b, int off, int ob) {
        unsigned int sb = gated_mla_bwd::lds_addr(smem + b + off);
        asm volatile("" : "+s"(sb));
        unsigned int a = sb + static_cast<unsigned int>(ob);
        pin(a);
        return a;
    };
    // Tile 0's K and V, and K^T: the loop's first dQ block is of no tile (its dS is 0, but what it
    // reads must be finite) and reads tile 0's. At the end of an iteration K^T's addresses are
    // taken from V's, then V's go a tile on.
    unsigned int a_kn = addr(0, OFF_KN, ob_kn), a_kr = addr(0, OFF_KR, ob_kr), a_v = addr(0, OFF_V, ob_v);
    unsigned int a_tkn = addr(0, OFF_KN, ob_tkn), a_tkr = addr(0, OFF_KR, ob_tkr);
    auto rd_k = [&](int tn, int ks) -> bvec<8> {
        return ks < KS_N ? at(a_kn).template load<8>(of_kn[tn] + ks * 32) : at(a_kr).template load<8>(of_kr[tn] + (ks - KS_N) * 32);
    };
    auto rd_v = [&](int tn, int ks) -> bvec<8> { return at(a_v).template load<8>(of_v[tn] + ks * 32); };
    // One transposed read of K's N tile gn (16 columns), half hh of the tile's 32 keys.
    auto rd_t = [&](int gn, int hh) {
        return gn < NGN ? gated_mla_bwd::tr_read(at(a_tkn), gn * 16 + hh * 16 * LD_KN)
                        : gated_mla_bwd::tr_read(at(a_tkr), (gn - NGN) * 16 + hh * 16 * LD_KR);
    };
    auto b_tr = [](const u32x2_t (&hv)[2]) { return u32x4_t{hv[0][0], hv[0][1], hv[1][0], hv[1][1]}; };
    auto acc1 = [](f32x4_t& ac, const u32x4_t& a, const u32x4_t& b) {
        asm volatile("v_mfma_f32_16x16x32_bf16 %0, %2, %1, %0" : "+a"(ac) : "v"(a), "v"(b));
    };

    float lse2[MR], dd[MR];        // this wave's LSE log2 e and D, per M tile
    f32x4_t nd4[MR];               // -D four times over: where dP's accumulators start (GMLA_DP_MINUS_D)
    bvec<8> bq[3], bv[3];          // B fragments of S and dP: the one in use and the next two
    u32x2_t tb[3][2];              // B of dQ: this N tile's two reads and the next two's
    opus::vector_t<float, NS> v_sp, v_dp;
    opus::vector_t<bf16_t, MR * 8> a_ds;
    f32x4_t c_old{0.0f, 0.0f, 0.0f, 0.0f};  // what a C tile's last MFMA accumulates onto
#pragma unroll
    for(int e = 0; e < NS; e++)
    {
        v_sp[e] = 0.0f;  // P and dS of the tile before the first are 0
        v_dp[e] = 0.0f;
    }
#pragma unroll
    for(int e = 0; e < MR * 8; e++)
        a_ds[e] = static_cast<bf16_t>(0.0f);
    // One MFMA of S (va is Q, vacc S) or dP (dO, dP) onto C tile tl. The first K step (step 0,
    // from 0) and the steps between (1, in place) are asm; the last (2) is the compiler's MFMA.
    // The compiler's may put its result in other registers and hand the old ones to the unit's
    // next instruction, which then stands three s_nop behind the MFMA that reads them: the last
    // step keeps the old ones (c_old) to the end of its unit.
    auto mm = [&](const auto& va, int ia, const bvec<8>& fb, auto& vacc, int tl, int step, const f32x4_t* c0 = nullptr) __attribute__((always_inline)) {
        bvec<8> fa;
        f32x4_t cc;
#pragma unroll
        for(int x = 0; x < 8; x++)
            fa[x] = va[ia * 8 + x];
        if(step == 0)
        {
            // From 0, or from what the accumulator starts at (c0: -D for dP).
            if(c0)
                asm volatile("v_mfma_f32_16x16x32_bf16 %0, %2, %1, %3" : "=&v"(cc) : "v"(fa), "v"(fb), "v"(*c0));
            else
                asm volatile("v_mfma_f32_16x16x32_bf16 %0, %2, %1, 0" : "=&v"(cc) : "v"(fa), "v"(fb));
        }
        else
        {
#pragma unroll
            for(int p_ = 0; p_ < 4; p_++)
                cc[p_] = vacc[tl * 4 + p_];
            if(step == 1)
                asm volatile("v_mfma_f32_16x16x32_bf16 %0, %2, %1, %0" : "+v"(cc) : "v"(fa), "v"(fb));
            else
            {
                c_old = cc;
                cc    = mma1(fa, fb, cc);
                asm volatile("" : "+v"(cc));
            }
        }
#pragma unroll
        for(int p_ = 0; p_ < 4; p_++)
            vacc[tl * 4 + p_] = cc[p_];
    };
    // The VALU of a unit is asm, as its first MFMAs are: a unit's instructions then stand in the
    // order they are written (the compiler's own VALU goes where ISel and the scheduler put it:
    // a unit early, or in front of the unit's read, which then issues 12 cycles late; anchors
    // and scheduling groups held it only where the groups happened to match). What is left to
    // the compiler in the loop: the LDS reads (chained between the asm around them), the last K
    // step's MFMA, the descriptors' 64-bit steps and the waits. It does not know these VALU
    // read what an MFMA wrote: the schedule keeps them 3 units behind that MFMA (it wants 11
    // instructions between; an MFMA is done in 16 cycles).
    auto pack = [&](int j) __attribute__((always_inline)) {
        opus::vector_t<bf16_t, 2> h2;
        asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(h2) : "v"(v_dp[2 * j]), "v"(v_dp[2 * j + 1]));
        a_ds[2 * j]     = h2[0];
        a_ds[2 * j + 1] = h2[1];
    };
    // S scale - LSE of one element, the exp (P), dP - D, dS = P (dP - D): the instructions the
    // compiler makes of the other kernels' expressions.
    auto fma1 = [&](int e) __attribute__((always_inline)) {
        float x = v_sp[e];
        asm volatile("v_fma_f32 %0, %0, %1, -%2" : "+v"(x) : "v"(sc2), "v"(lse2[e / (NS / MR)]));
        v_sp[e] = x;
    };
    auto expa = [&](int e) __attribute__((always_inline)) {
        float x = v_sp[e];
        asm volatile("v_exp_f32_e32 %0, %0" : "+v"(x));
        v_sp[e] = x;
    };
    auto sub1 = [&](int e) __attribute__((always_inline)) {
        float x = v_dp[e];
        asm volatile("v_sub_f32 %0, %0, %1" : "+v"(x) : "v"(dd[e / (NS / MR)]));
        v_dp[e] = x;
    };
    auto mul1 = [&](int e) __attribute__((always_inline)) {
        float x = v_dp[e];
        asm volatile("v_mul_f32 %0, %1, %0" : "+v"(x) : "v"(v_sp[e]));
        v_dp[e] = x;
    };

    // The bounds are in.
    asm volatile("s_waitcnt lgkmcnt(0)" : "+s"(bnd_q), "+s"(bnd_k));
    const int q0 = static_cast<int>(bnd_q), k0 = static_cast<int>(bnd_k);
    const int seqlen_q = static_cast<int>(bnd_q >> 32) - q0, seqlen_k = static_cast<int>(bnd_k >> 32) - k0;
    const int qb0      = qb * BQ;
    // A block past the sequence's queries has nothing to do (no early return, as in the dK / dV body).
    if(qb0 < seqlen_q)
    {
    // Bottom-right causal: query qp sees keys kv <= qp + causal_offset. The workgroup's tiles are
    // those its last query sees, 0 .. T - 1; if it sees none the loop runs tile 0, all masked.
    const int causal_offset = seqlen_k - seqlen_q;
    const int q_last        = (qb0 + BQ < seqlen_q ? qb0 + BQ : seqlen_q) - 1;
    int kv_end              = seqlen_k;
    if(k.causal && q_last + causal_offset + 1 < kv_end)
        kv_end = q_last + causal_offset + 1;
    const int T = kv_end > 0 ? (kv_end + KC - 1) / KC : 0;
    // This wave's 48 queries; rows past seqlen_q read as 0.
    const int qw0    = qb0 + w * WQ;
    const int rows_q = seqlen_q - qw0 < WQ ? (seqlen_q - qw0 > 0 ? seqlen_q - qw0 : 0) : WQ;

    // The tiles this wave sees in full: [0, n_full). Rows past seqlen_q count as rows: Q, dO, LSE
    // and D read as 0 there, so P is 1 and dP - D 0, and they add nothing whatever they see.
    int n_full = seqlen_k / KC;
    if(k.causal)
    {
        const int lim = qw0 + causal_offset + 1 > 0 ? (qw0 + causal_offset + 1) / KC : 0;
        n_full        = lim < n_full ? lim : n_full;
    }
    here(n_full);

    // LSE and D of the wave's queries first: six small loads, issued while no other is out.
    {
        const float* lse_p = lse_h + (q0 + qw0);
        const float* d_p   = d_h + (q0 + qw0);
        unsigned int ld_sz = static_cast<unsigned int>(__builtin_amdgcn_readfirstlane(rows_q * 4));
        asm volatile("" : "+s"(lse_p), "+s"(d_p), "+s"(ld_sz));
        auto g_lse = opus::make_gmem(lse_p, ld_sz);
        auto g_d   = opus::make_gmem(d_p, ld_sz);
#pragma unroll
        for(int tm = 0; tm < MR; tm++)
        {
            lse2[tm] = g_lse.template load<1>(16 * tm + lane % 16)[0];
            dd[tm]   = g_d.template load<1>(16 * tm + lane % 16)[0];
        }
    }
    const char* kn_p = reinterpret_cast<const char*>(kn_h + static_cast<long long>(k0) * k.stride_k_n);
    const char* kr_p = reinterpret_cast<const char*>(reinterpret_cast<const bf16_t*>(k.ptr_k_rot) + static_cast<long long>(k0) * k.stride_k_rot_n);
    const char* v_p  = reinterpret_cast<const char*>(v_h + static_cast<long long>(k0) * k.stride_v_n);
    const int rb_n = __builtin_amdgcn_readfirstlane(k.stride_k_n * 2), rb_r = __builtin_amdgcn_readfirstlane(k.stride_k_rot_n * 2);
    const int rb_v = __builtin_amdgcn_readfirstlane(k.stride_v_n * 2);
    const int x_rb[4] = {rb_n, __builtin_amdgcn_readfirstlane(w == 0 ? rb_n : rb_r), __builtin_amdgcn_readfirstlane(w < 2 ? rb_r : rb_v), rb_v};
    const char* x_src[4] = {kn_p, w == 0 ? kn_p : kr_p, w < 2 ? kr_p : v_p, v_p};
    int x_kb[4];
    gated_mla_bwd::i32x4_t x_rs[4];
#pragma unroll
    for(int d = 0; d < 4; d++)
    {
        x_kb[d] = KC * x_rb[d];
        x_rs[d] = gated_mla_bwd::dma_rsrc(x_src[d], 0);
        asm volatile("" : "+s"(x_kb[d]), "+s"(x_rs[d]));
    }
    auto x_sizes = [&](int t) {
        int rows = seqlen_k - t * KC;
        asm("s_min_i32 %0, %0, %1\n\ts_max_i32 %0, %0, 0" : "+s"(rows) : "n"(KC) : "scc");
#pragma unroll
        for(int d = 0; d < 4; d++)
        {
            int nb;
            asm("s_mul_i32 %0, %1, %2" : "=s"(nb) : "s"(rows), "s"(x_rb[d]));
            x_rs[d] = gated_mla_bwd::i32x4_t{x_rs[d][0], x_rs[d][1], nb, 0x00020000};
            asm volatile("" : "+s"(x_rs[d]));
        }
    };
    auto x_adv = [&](int d) {
        auto& r = x_rs[d];
        asm volatile("" : "+s"(r));
        unsigned long long a =
            static_cast<unsigned int>(r[0]) | static_cast<unsigned long long>(static_cast<unsigned int>(r[1])) << 32;
        a += static_cast<unsigned long long>(static_cast<unsigned int>(x_kb[d]));
        r = gated_mla_bwd::i32x4_t{static_cast<int>(a), static_cast<int>(a >> 32), r[2], r[3]};
        asm volatile("" : "+s"(r));
    };
    auto x_go = [&](int u, unsigned int lb) {
        const int d = u < 2 ? 0 : (u - 1 < 3 ? u - 1 : 3);
        asm volatile("s_add_u32 m0, %2, %3\n\tbuffer_load_dwordx4 %0, %1, 0 offen lds"
                     :
                     : "v"(x_os[u]), "s"(x_rs[d]), "s"(lb), "s"(x_lds[u])
                     : "m0", "scc");
    };

    // Tiles 0 and 1 to LDS (buffers 0 and 1), ahead of Q and dO: the memory pipeline is at them
    // while the 30 loads of Q and dO are issued.
    {
        unsigned int lb0 = lds_of(0), lb1 = lds_of(SZ_BUF);
        asm volatile("" : "+s"(lb0), "+s"(lb1));
        x_sizes(0);
#pragma unroll
        for(int u = 0; u < XR; u++)
            x_go(u, lb0);
#pragma unroll
        for(int d = 0; d < 4; d++)
            x_adv(d);
        x_sizes(1);
#pragma unroll
        for(int u = 0; u < XR; u++)
            x_go(u, lb1);
        // Descriptor 0 goes a tile on past its rounds, the others in front of theirs.
        x_adv(0);
        x_sizes(2);
    }
    // This wave's queries as the A operand of S and dP, in registers.
    auto g_q  = opus::make_gmem(q_h + static_cast<long long>(q0 + qw0) * k.stride_q_n, rec(static_cast<long long>(rows_q) * k.stride_q_n, 2));
    auto g_do = opus::make_gmem(do_h + static_cast<long long>(q0 + qw0) * k.stride_do_n, rec(static_cast<long long>(rows_q) * k.stride_do_n, 2));
    const auto v_qw  = g_q.template load<8>(opus::partition_layout_a<8>(mma_s, opus::make_tuple(k.stride_q_n, 1_I), c_ab));
    const auto v_dow = g_do.template load<8>(opus::partition_layout_a<8>(mma_dp, opus::make_tuple(k.stride_do_n, 1_I), c_ab));
    {
        // Loop-carried scalars, at the top of the iteration of S tile ts: lbc, lbp the LDS address
        // of tile ts - 1's buffer and of tile ts - 2's (the DMA's target, tile ts + 2 goes there);
        // dlt the LDS addresses' step from tile ts - 1's buffer to tile ts's (from tile ts's to the
        // next past S unit 25); rc tile ts's buffer in the ring (0 .. 3), dn the step from it to
        // the next (-3 buffers past the last, without flags: single scalars go anywhere);
        // ra = seqlen_k - (ts + 4) KC (negative: tile ts + 3 is not a full one), nf = n_full - 1
        // - ts (negative: mask), z = ts - T.
        constexpr int SZB = SZ_BUF * 2;
        static_assert(NBUF == 4, "the ring's wrap below");
        int rc = 0, dn = 0;
        int dlt = -(NBUF - 1) * SZB;
        unsigned int lbc = lds_of((NBUF - 1) * SZ_BUF), lbp = lds_of((NBUF - 2) * SZ_BUF);
        int ra = seqlen_k - 4 * KC, nf = n_full - 1, z = -T;
        asm volatile("" : "+s"(rc), "+s"(dlt), "+s"(lbc), "+s"(lbp));
        asm volatile("" : "+s"(ra), "+s"(nf), "+s"(z));

        namespace sch = gated_mla_bwd_dq4s;
        // A unit's VALU from its row of the schedule. The rows are read in constant expressions
        // and passed by value (the tables are globals of the device module otherwise, whose loads
        // do not fold).
        auto v_run = [&](sch::vop o0, sch::vop o1, sch::vop o2) __attribute__((always_inline)) {
            const sch::vop os[3] = {o0, o1, o2};
#pragma unroll
            for(int i = 0; i < 3; i++)
            {
                const int kd = os[i].k, e = os[i].e;
                if(kd == sch::V_PACK)
                    pack(e);
                if(kd == sch::V_FMA)
                    fma1(e);
                if(kd == sch::V_EXP)
                    expa(e);
                if(kd == sch::V_SUB)
                    sub1(e);
                if(kd == sch::V_MUL)
                    mul1(e);
                if(kd == sch::V_AKN)
                    asm volatile("v_add_u32 %0, %1, %0" : "+v"(a_kn) : "s"(dlt));
                if(kd == sch::V_AKR)
                    asm volatile("v_add_u32 %0, %1, %0" : "+v"(a_kr) : "s"(dlt));
                if(kd == sch::V_AV)
                    asm volatile("v_add_u32 %0, %1, %0" : "+v"(a_v) : "s"(dlt));
                if(kd == sch::V_ATKN)
                    asm volatile("v_add_u32 %0, %1, %2" : "=v"(a_tkn) : "v"(a_v), "v"(cv_n));
                if(kd == sch::V_ATKR)
                    asm volatile("v_add_u32 %0, %1, %2" : "=v"(a_tkr) : "v"(a_v), "v"(cv_r));
            }
        };
        auto r_run = [&](sch::rop r) __attribute__((always_inline)) {
            if(r.k == sch::R_K)
                bq[r.a % 3] = rd_k(r.a / KS, r.a % KS);
            if(r.k == sch::R_V)
                bv[r.a % 3] = rd_v(r.a / KSV, r.a % KSV);
            if(r.k == sch::R_T)
                tb[r.a % 3][r.b] = rd_t(r.a, r.b);
        };

        // S of tile ts, 36 units: fragment f = u / 3 (N tile f / 6, K step f % 6) for M tile
        // u % 3. It follows the barrier, behind which VALU stands still for some 100 cycles while
        // MFMA, LDS reads, DMA and scalars run on: its first 7 units carry reads, two DMA rounds
        // and scalar pairs (the DMA descriptors' 64-bit steps). A unit takes one scalar (or wait,
        // or the s_nop of an anchor) and two of read and VALU behind its MFMA, a second scalar
        // for one of those; a DMA round takes a unit's two.
        auto s_block = [&]() __attribute__((always_inline)) {
            int rare = 0;
            opus::static_for<NT * KS * MR>([&](auto U) {
                constexpr int u = decltype(U)::value, f = u / MR, j = u % MR, tn = f / KS, ks = f % KS;
                constexpr sch::vop o0 = sch::S_V[u][0], o1 = sch::S_V[u][1], o2 = sch::S_V[u][2];
                constexpr sch::rop rs = sch::S_R[u];
                constexpr int xr = sch::S_X[u];
                mm(v_qw, j * KS + ks, bq[f % 3], v_sp, NT * j + tn, ks == 0 ? 0 : (ks == KS - 1 ? 2 : 1));
                if(u % 3 == 2 && u <= 32)
                    touch(bq[(u + 1) / 3 % 3]);
                if(u == 34)
                    touch(bv[0]);
                // The DMA's descriptors a tile on: 1 .. 3 in front of their rounds, 0 past its.
                if(u == 0)
                    x_adv(2);
                if(u == 3)
                    x_adv(1);
                if(u == 6)
                    x_adv(0);
                if(u == 9)
                    x_adv(3);
                // The ring a buffer on: rc = (rc + 1) & 3, dn = rc == 0 ? -3 buffers : 1 buffer.
                // Single scalars: one to a unit that has no wait.
                if(u == 2)
                    asm volatile("s_add_i32 %0, %0, 1" : "+s"(rc) : : "scc");
                if(u == 5)
                    asm volatile("s_and_b32 %0, %0, 3" : "+s"(rc) : : "scc");
                if(u == 12)
                    asm volatile("s_sub_i32 %0, %1, 1" : "=s"(dn) : "s"(rc) : "scc");
                if(u == 15)
                    asm volatile("s_ashr_i32 %0, %0, 31" : "+s"(dn) : : "scc");
                if(u == 18)
                    asm volatile("s_and_b32 %0, %0, %1" : "+s"(dn) : "n"(-NBUF * SZB) : "scc");
                if(u == 21)
                    asm volatile("s_add_i32 %0, %0, %1" : "+s"(dn) : "n"(SZB) : "scc");
                // Past the last round: the DMA's target and the address step a tile on.
                if(u == 22)
                    asm volatile("s_mov_b32 %0, %1" : "=s"(lbp) : "s"(lbc));
                if(u == 24)
                    asm volatile("s_add_u32 %0, %0, %1" : "+s"(lbc) : "s"(dlt) : "scc");
                if(u == 25)
                    asm volatile("s_mov_b32 %0, %1" : "=s"(dlt) : "s"(dn));
                // The counters on; the rare cases' sign from the S tile's ra and nf first.
                if(u == 27)
                    asm volatile("s_add_i32 %0, %0, 1" : "+s"(z) : : "scc");
                if(u == 28)
                {
                    rare = ra | nf;
                    here(rare);
                }
                if(u == 30)
                    asm volatile("s_sub_i32 %0, %0, %1" : "+s"(ra) : "n"(KC) : "scc");
                if(u == 31)
                    asm volatile("s_sub_i32 %0, %0, 1" : "+s"(nf) : : "scc");
                r_run(rs);
                if(xr >= 0)
                    x_go(xr >= 0 ? xr : 0, lbp);
                v_run(o0, o1, o2);
                if(ks == KS - 1)
                    touch(c_old);
                fence();
            });
            return rare;
        };
        // The rare cases: tile ts + 3 is not a full one (its sizes), this wave does not see tile
        // ts in full: the elements it does not see take P = 0 where the exp is made already,
        // else an S (or S scale - LSE) that exps to 0.
        auto rare_fix = [&](int ts) {
            if(seqlen_k - (ts + 4) * KC < 0)
                x_sizes(ts + 3);
            if(ts >= n_full)
            {
                // Element e is row 16 (e / 8) + lane % 16 of the wave, key 16 (e / 4 % 2) +
                // 4 (lane / 16) + e % 4 of the tile; xm[M tile] is the last key of the tile that
                // row sees less the lane's 4 (lane / 16) (causal: a key is seen if kv <= qp +
                // causal_offset, which is below seqlen_k for a row below seqlen_q).
                const int kv0 = ts * KC;
                int xm[MR];
#pragma unroll
                for(int m = 0; m < MR; m++)
                    xm[m] = (k.causal ? qw0 + causal_offset + 16 * m + lane % 16 : seqlen_k - 1) - kv0 - 4 * (lane / 16);
                // The mask, four elements (one C tile) at a time: four compares into scalar pairs, then
                // the four selects. A select then stands three instructions behind its compare; left
                // to the compiler, each pair takes two s_nop between them (a rare iteration was 400
                // cycles of masking).
                float m_z = 0.0f, m_n = -1e30f;
                pin(m_z);
                pin(m_n);
                opus::static_for<NS / 4>([&](auto G) {
                    constexpr int gt = decltype(G)::value, c0 = 16 * (gt % NT), e0 = 4 * gt;
                    float x0 = v_sp[e0], x1 = v_sp[e0 + 1], x2 = v_sp[e0 + 2], x3 = v_sp[e0 + 3];
                    unsigned long long t0, t1, t2, t3;
                    asm volatile("v_cmp_ge_i32 %4, %8, %13\n\tv_cmp_ge_i32 %5, %8, %14\n\tv_cmp_ge_i32 %6, %8, %15\n\tv_cmp_ge_i32 %7, %8, %16\n\t"
                                 "v_cndmask_b32 %0, %9, %0, %4\n\tv_cndmask_b32 %1, %10, %1, %5\n\tv_cndmask_b32 %2, %11, %2, %6\n\tv_cndmask_b32 %3, %12, %3, %7"
                                 : "+v"(x0), "+v"(x1), "+v"(x2), "+v"(x3), "=&s"(t0), "=&s"(t1), "=&s"(t2), "=&s"(t3)
                                 : "v"(xm[gt / NT]), "v"((sch::S_EXPD >> e0 & 1) ? m_z : m_n), "v"((sch::S_EXPD >> (e0 + 1) & 1) ? m_z : m_n),
                                   "v"((sch::S_EXPD >> (e0 + 2) & 1) ? m_z : m_n), "v"((sch::S_EXPD >> (e0 + 3) & 1) ? m_z : m_n),
                                   "n"(c0), "n"(c0 + 1), "n"(c0 + 2), "n"(c0 + 3));
                    v_sp[e0]     = x0;
                    v_sp[e0 + 1] = x1;
                    v_sp[e0 + 2] = x2;
                    v_sp[e0 + 3] = x3;
                });
            }
        };
        // dP(ts), 24 units: fragment f = u / 3 (N tile f / 4, K step f % 4) for M tile u % 3.
        auto dp_block = [&]() __attribute__((always_inline)) {
            opus::static_for<NT * KSV * MR>([&](auto U) {
                constexpr int u = decltype(U)::value, f = u / MR, j = u % MR, tn = f / KSV, ks = f % KSV;
                constexpr sch::vop o0 = sch::P_V[u][0], o1 = sch::P_V[u][1], o2 = sch::P_V[u][2];
                constexpr sch::rop rs = sch::P_R[u];
                mm(v_dow, j * KSV + ks, bv[f % 3], v_dp, NT * j + tn, ks == 0 ? 0 : (ks == KSV - 1 ? 2 : 1),
                   GMLA_DP_MINUS_D && ks == 0 ? &nd4[j] : nullptr);
                if(u % 3 == 2 && u <= 20)
                    touch(bv[(u + 1) / 3 % 3]);
                if(u == 23)
                    touch2(tb[0][0], tb[0][1]);
                r_run(rs);
                v_run(o0, o1, o2);
                if(ks == KSV - 1)
                    touch(c_old);
                fence();
            });
        };
        // dQ of the tile before (of the last tile in the tail), 36 units of 3 per N tile; the
        // first two of an N tile read K^T two N tiles ahead (the last N tiles' the next S's first
        // fragments, then nothing). The waits before the barrier: this wave's DMA loads, issued
        // in this iteration's S block, are in.
        auto dq_block = [&](bool tail) __attribute__((always_inline)) {
            opus::static_for<NG * MR>([&](auto U) {
                constexpr int q = decltype(U)::value, gn = q / MR, m = q % MR;
                constexpr sch::vop o0 = sch::Q_V[q][0], o1 = sch::Q_V[q][1], o2 = sch::Q_V[q][2];
                acc1(acc[m][gn], gated_mla_bwd::frag8(a_ds, m), b_tr(tb[gn % 3]));
                if(m == 2 && gn + 1 < NG)
                    touch2(tb[(gn + 1) % 3][0], tb[(gn + 1) % 3][1]);
                if(!tail && gn == NG - 1 && m == 1)
                {
                    touch(bq[0]);
                    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                }
                if(m < 2 && gn + 2 < NG)
                    tb[(gn + 2) % 3][m] = rd_t(gn + 2, m);
                if(!tail && m < 2 && gn == NG - 2)
                    bq[m] = rd_k(0, m);
                if(!tail)
                    v_run(o0, o1, o2);
                fence();
            });
        };

        // Q, dO, LSE and D are in registers and this wave's loads of tiles 0 and 1 in LDS (0x0f70 is
        // s_waitcnt vmcnt(0), as an instruction the compiler sees: it then waits for none of them
        // inside the loop); every wave's are behind the barrier. LSE log2 e (0x3fb8aa3b).
        __builtin_amdgcn_s_waitcnt(0x0f70);
#pragma unroll
        for(int tm = 0; tm < MR; tm++)
            asm volatile("v_mul_f32 %0, 0x3fb8aa3b, %0" : "+v"(lse2[tm]));
#if GMLA_DP_MINUS_D
#pragma unroll
        for(int tm = 0; tm < MR; tm++)
        {
            float nd = dd[tm];
            asm volatile("v_xor_b32 %0, 0x80000000, %0" : "+v"(nd));
            nd4[tm] = f32x4_t{nd, nd, nd, nd};
            asm volatile("" : "+v"(nd4[tm]));
        }
#endif
        barrier();
        bq[0] = rd_k(0, 0);
        bq[1] = rd_k(0, 1);
        touch(bq[0]);
        // The loop starts on a 64-byte boundary: where its code lies decides whether the branch
        // of the rare cases (not taken) costs 4 cycles or 20, and with the kernel's code in front
        // of it shifting by 4 or 8 bytes the iteration went from 1628 cycles to 1644.
        asm volatile(".p2align 6");
        do
        {
            // Every wave is past dQ's reads of tile ts - 2, whose buffer this S block's DMA
            // overwrites, and has its loads of tile ts + 1 in.
            barrier();
            const int rare = s_block();
            if(__builtin_expect(rare < 0, 0))
                rare_fix(z - 1 + T);
            dp_block();
            dq_block(false);
        } while(__builtin_expect(z < 0, 1));
        // The last tile's dQ: its dS to bf16, its first two N tiles' fragments of K^T (whose
        // addresses the last iteration left on it).
#pragma unroll
        for(int jp = 0; jp < NS / 2; jp++)
            pack(jp);
#pragma unroll
        for(int x = 0; x < 4; x++)
            tb[x / 2][x % 2] = rd_t(x / 2, x % 2);
        dq_block(true);
    }
    // The loads still out go to LDS; the accumulators' last MFMAs finish.
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    gated_mla_bwd::mfma_acc_drain();
    // The accumulators are read from here on: the compiler does not know the asm MFMAs' latency
    // and would read one right behind the MFMA that writes it.
#pragma unroll
    for(int m = 0; m < MR; m++)
#pragma unroll
        for(int n = 0; n < NG; n++)
            asm volatile("" : "+a"(acc[m][n]));
    auto g_dq = opus::make_gmem(dq_h + static_cast<long long>(q0 + qw0) * k.stride_dq_n,
                                rec(static_cast<long long>(rows_q) * k.stride_dq_n, 2));
    // The C fragments hold 4 columns of 16 rows per instruction; stored like that, every lane is
    // a memory request of its own (a store of 64 lanes then takes 256 cycles to issue). Each
    // wave transposes its 48 rows through LDS instead (free once every wave's DMA writes are in:
    // the barrier) and stores whole rows, 64 lanes x 8 bf16 on end; rows past seqlen_q fall
    // outside the descriptor and are dropped.
    __syncthreads();
    {
        constexpr int LD_O = D_QK + 16;
        static_assert(NW * WQ * LD_O <= SMEM_ELEMS && WQ * (D_QK / 8) % 64 == 0, "staging fits; whole instructions");
        bf16_t* s_o = smem + w * WQ * LD_O;
        // C tile (m, n): row 16 m + lane % 16, columns 16 n + 4 (lane / 16) + 0 .. 3.
        const int c_row = lane % 16, c_col = 4 * (lane / 16);
#pragma unroll
        for(int m = 0; m < MR; m++)
#pragma unroll
            for(int n = 0; n < NG; n++)
                *reinterpret_cast<bvec<4>*>(s_o + (16 * m + c_row) * LD_O + 16 * n + c_col) = opus::cast<bf16_t>(acc[m][n] * k.softmax_scale);
        constexpr int OC = D_QK / 8;  // 8-element chunks of a row
#pragma unroll
        for(int it = 0; it < WQ * OC / 64; it++)
        {
            const int ch = it * 64 + lane, row = ch / OC, col = ch % OC * 8;
            g_dq.template store<8>(*reinterpret_cast<const bvec<8>*>(s_o + row * LD_O + col), row * k.stride_dq_n + col);
        }
    }
    }
}


// The 4-wave backward: the dK / dV pass and the dQ pass in one launch, a workgroup id being of
// one or the other by its place in the grid. 4 waves, one per SIMD; a pass with no blocks takes
// no workgroups. The kernel reads its arguments (three scalar loads, the accumulators' zeros
// behind them: the 36 tiles both passes have) and hands over to the pass's body.
__global__ __launch_bounds__(gated_mla_bwd_4w::BLOCK, 1) void gated_mla_bwd_4w_kernel(gated_mla_bwd_4w_kargs k_)
{
    using gated_mla_bwd::bf16_t;
    using gated_mla_bwd::f32x4_t;
    constexpr int SMEM = gated_mla_bwd_4w::SMEM_ELEMS > gated_mla_bwd_dq4::SMEM_ELEMS ? gated_mla_bwd_4w::SMEM_ELEMS : gated_mla_bwd_dq4::SMEM_ELEMS;
    static_assert(gated_mla_bwd_4w::BLOCK == gated_mla_bwd_dq4::BLOCK, "one workgroup shape");
    __shared__ __attribute__((aligned(16))) bf16_t smem[SMEM];

    const int tid  = static_cast<int>(opus::thread_id_x());
    const int lane = tid % 64;
    const int w    = __builtin_amdgcn_readfirstlane(tid / 64);
    gated_mla_bwd_4w_kargs k = k_;

    // The asm keeps the zeros here, as values of registers and not constants to make again
    // where they are used.
    f32x4_t z[3][12];
#pragma unroll
    for(int m = 0; m < 3; m++)
#pragma unroll
        for(int n = 0; n < 12; n++)
        {
            z[m][n] = f32x4_t{0.0f, 0.0f, 0.0f, 0.0f};
            asm volatile("" : "+a"(z[m][n]));
        }
    // Every argument goes through here: one batch of loads, one wait behind the zeros. (In and
    // out: as inputs only, the compiler moves some of them to other registers in front of the
    // last load, with a wait of its own.)
    asm volatile("" : "+s"(k.ptr_q), "+s"(k.ptr_k_nope), "+s"(k.ptr_v), "+s"(k.ptr_k_rot), "+s"(k.ptr_do), "+s"(k.ptr_lse));
    asm volatile("" : "+s"(k.ptr_d), "+s"(k.ptr_seqstart_q), "+s"(k.ptr_seqstart_k), "+s"(k.ptr_dq), "+s"(k.ptr_dkv), "+s"(k.ptr_dk_rope));
    asm volatile("" : "+s"(k.H), "+s"(k.num_seqs), "+s"(k.n_kv_blocks), "+s"(k.n_q_blocks), "+s"(k.hg), "+s"(k.causal), "+s"(k.softmax_scale));
    asm volatile("" : "+s"(k.stride_k_rot_n), "+s"(k.stride_q_n), "+s"(k.stride_q_h), "+s"(k.stride_k_n), "+s"(k.stride_k_h), "+s"(k.stride_v_n));
    asm volatile("" : "+s"(k.stride_v_h), "+s"(k.stride_do_n), "+s"(k.stride_do_h), "+s"(k.stride_lse_h), "+s"(k.stride_dq_n), "+s"(k.stride_dq_h));
    asm volatile("" : "+s"(k.stride_dkv_n), "+s"(k.stride_dkv_h), "+s"(k.stride_dkr_n), "+s"(k.stride_dkr_h), "+s"(k.div_magic));

    // The workgroup's place is its place in the grid (gated_mla_bwd_4w_kargs). With a causal mask
    // (by_group) z is the group and the sequence, y the block of the dK / dV pass, then of the dQ
    // pass; without, z is the sequence and the pass, y the group and the block. x is the head in
    // its group.
    // readfirstlane: these are scalars, and must stay so for the DMA's descriptors and m0.
    auto sc = [](int x) { return __builtin_amdgcn_readfirstlane(x); };
    const int gy = static_cast<int>(opus::block_id_y()), gz = static_cast<int>(opus::block_id_z());
    const int both     = (k.n_kv_blocks > 0 && k.n_q_blocks > 0) ? 1 : 0;
    const int nbm      = k.n_kv_blocks > k.n_q_blocks ? k.n_kv_blocks : k.n_q_blocks;
    const int by_group = k.causal ? 1 : 0;
    // The coordinate that holds the group, over what it counts there (div_magic).
    const int dv   = by_group ? gz : gy;
    const int grp  = static_cast<int>((static_cast<unsigned long long>(static_cast<unsigned int>(dv)) * static_cast<unsigned int>(k.div_magic)) >> 31);
    const int rem  = dv - grp * (by_group ? k.num_seqs : nbm);
    const int g    = sc(by_group ? rem : (both ? gz >> 1 : gz));
    // The sequence's bounds, seqstart[g] and [g + 1] of the queries and of the keys: two scalar
    // loads that go out here; a pass waits for them past its lanes' constants.
    unsigned long long bnd_q, bnd_k;
    {
        const int* pq = k.ptr_seqstart_q + g;
        const int* pk = k.ptr_seqstart_k + g;
        asm volatile("s_load_dwordx2 %0, %2, 0\n\ts_load_dwordx2 %1, %3, 0" : "=&s"(bnd_q), "=&s"(bnd_k) : "s"(pq), "s"(pk));
    }
    const int is_q = sc(by_group ? (gy >= k.n_kv_blocks ? 1 : 0) : (both ? gz & 1 : (k.n_kv_blocks == 0 ? 1 : 0)));
    const int nb   = is_q ? k.n_q_blocks : k.n_kv_blocks;
    const int blk0 = by_group ? (is_q ? gy - k.n_kv_blocks : gy) : rem;
    const int h    = sc(grp * k.hg + static_cast<int>(opus::block_id_x()));
    // Blocks 2 and 3 of every four change places (gated_mla_bwd::wg_map's GMLA_ZIG).
    const int blk  = sc((blk0 & 2) && (blk0 | 1) < nb ? blk0 ^ 1 : blk0);
    // The last group's ids past H heads, the shorter pass's past its blocks.
    if(h >= k.H || blk0 >= nb)
    {
        asm volatile("s_endpgm");
        __builtin_unreachable();
    }
    // Each pass ends the kernel itself: with the two ends joined, the compiler turns the if / else
    // into "if(!is_q) dK / dV; if(is_q) dQ" and keeps what the second needs (the arguments, the
    // zeroed accumulators) alive across the first, out of registers.
    if(is_q)
    {
        gated_mla_bwd_dq_4w_body(k, smem, lane, w, h, blk, bnd_q, bnd_k, z);
        asm volatile("s_endpgm");
        __builtin_unreachable();
    }
    else
    {
        gated_mla_bwd_dkdv_4w_body(k, smem, lane, w, h, blk, bnd_q, bnd_k, z);
        asm volatile("s_endpgm");
        __builtin_unreachable();
    }
}

#else
__global__ void gated_mla_bwd_pre_kernel(gated_mla_bwd_pre_kargs) {}
template <bool DQ>
__global__ void gated_mla_bwd_kernel(gated_mla_bwd_kargs) {}
template <bool DS>
__global__ void gated_mla_bwd_dkdv_pair_kernel(gated_mla_bwd_kargs) {}
__global__ void gated_mla_bwd_dkdv_pair32_kernel(gated_mla_bwd_kargs) {}
__global__ void gated_mla_bwd_dq_kernel(gated_mla_bwd_dq_kargs) {}
__global__ void gated_mla_bwd_4w_kernel(gated_mla_bwd_4w_kargs) {}
__global__ void gated_mla_bwd_dq_ds_kernel(gated_mla_bwd_dq_kargs) {}
__global__ void gated_mla_bwd_dq32_kernel(gated_mla_bwd_dq_kargs) {}
#endif
