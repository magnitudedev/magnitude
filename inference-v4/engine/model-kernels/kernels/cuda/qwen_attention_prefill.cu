// qwen_attention_prefill (M >= 16): flash attention on tensor cores.
//
// L1 `qwen_attention_prefill_prepare`, one warp per (row, query or kv head):
// the query (norm_rotary, rounded to the activation element) goes to scratch
// as the 16-bit MMA operand (scores are scaled into the exp2 domain after
// Q K^T); the prepared key goes to scratch in the activation
// element (as history stores it), and the row's K/V is appended at its
// destination.
// L2 `qwen_attention_prefill_attend`, one block per (row tile, kv head): the
// tile's matrix rows are QT tokens x G query heads (16 per warp). Each span in
// order (the R visible history spans, then the fresh span of batch rows) is
// scanned over the union of the tile rows' intervals in KEYS-key K/V tiles,
// double-buffered through `cp.async`. S = Q K^T and O += P V run as
// m16n8k16 MMAs with F32 accumulation; P stays in registers (FA2). Per-row
// interval masks apply only to K/V tiles outside the rows' common interval,
// and tiles past every row's interval (the causal tail) are never loaded.
// The sigmoid gate is fused into the store.

#include "common/attention.cuh"

namespace {

using attn::G;
using attn::KV;
using attn::W;
using mx::u16;
using mx::u32;
using mx::u64;
using mx::u8;

constexpr int WARPS = SEISMIC_TUNE_WARPS;
constexpr int ROWS = WARPS * 16;
constexpr int QT = ROWS / G;
constexpr int KEYS = 32;
constexpr int CHUNKS = W / 8;  // 16-byte chunks per 16-bit row
static_assert(16 % G == 0, "a warp's 16 matrix rows hold whole tokens");

// MMA operand element: bf16 for bf16 activations, f16 otherwise.
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
__device__ __forceinline__ u16 operand(float value) { return seismic_f32_to_bf16(value); }
__device__ __forceinline__ u32 operand_pair(float lo, float hi) {
    return seismic_pack_bf16x2(lo, hi);
}
__device__ __forceinline__ void mma(float (&acc)[4], const u32 (&a)[4], const u32 (&b)[2]) {
    seismic_mma_m16n8k16_bf16(acc, a, b);
}
#else
__device__ __forceinline__ u16 operand(float value) { return seismic_f32_to_f16(value); }
__device__ __forceinline__ u32 operand_pair(float lo, float hi) {
    return seismic_pack_f16x2(lo, hi);
}
__device__ __forceinline__ void mma(float (&acc)[4], const u32 (&a)[4], const u32 (&b)[2]) {
    seismic_mma_m16n8k16_f16(acc, a, b);
}
#endif

// Swizzled shared layout of a 16-bit [rows][W] tile: 16-byte chunk c of row r
// sits at chunk c ^ (r % SWIZZLE), so ldmatrix row sets are bank-conflict
// free (at W = 32 two rows share a bank set).
constexpr int SWIZZLE = CHUNKS < 8 ? CHUNKS : 8;
__device__ __forceinline__ u32 swizzled(int row, int chunk) {
    return static_cast<u32>(row * CHUNKS + (chunk ^ (row & (SWIZZLE - 1)))) * 16;
}

// Stage `rows` rows of a [.., W] source into a swizzled 16-bit tile.
// `source(r)` is the element offset of row r, or -1 for a zero row. A source
// of activation elements converts f32 activations to the operand element; a
// 16-bit source (activation or operand) is copied with `cp.async`.
template <bool F32_SOURCE, class Source>
__device__ __forceinline__ void stage(u8 *tile, const u8 *base, int rows, Source source) {
    for (int index = threadIdx.x; index < rows * CHUNKS; index += WARPS * 32) {
        const int row = index / CHUNKS;
        const int chunk = index % CHUNKS;
        const long long at = source(row);
        u8 *destination = tile + swizzled(row, chunk);
        if constexpr (F32_SOURCE) {
            uint4 packed = make_uint4(0, 0, 0, 0);
            if (at >= 0) {
                const float4 *from = reinterpret_cast<const float4 *>(
                    base + (static_cast<u64>(at) + chunk * 8) * 4);
                const float4 a = from[0];
                const float4 b = from[1];
                packed = make_uint4(operand_pair(a.x, a.y), operand_pair(a.z, a.w),
                                    operand_pair(b.x, b.y), operand_pair(b.z, b.w));
            }
            *reinterpret_cast<uint4 *>(destination) = packed;
        } else {
            const u8 *from = base + (at >= 0 ? static_cast<u64>(at) + chunk * 8 : 0) * 2;
            seismic_cp_async_16_zfill(destination, from, at >= 0 ? 16u : 0u);
        }
    }
}

constexpr bool F32_ACTIVATION = MX_ACT_BYTES == 4;

struct Tile {
    int span;
    int first;
};

}  // namespace

extern "C" __global__ void __launch_bounds__(256) qwen_attention_prefill_prepare(SEISMIC_KERNEL_PARAMS) {
    const attn::Inputs in = ATTN_INPUTS();
    u16 *queries = reinterpret_cast<u16 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_QUERIES));
    u8 *keys = SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_KEYS);
    __shared__ float exchange[8][W];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const u64 item = static_cast<u64>(blockIdx.x) * 8 + warp;
    const u64 heads = KV * (G + 1);
    if (item >= SEISMIC_DIM_M * heads) return;
    const u64 row = item / heads;
    const int head = static_cast<int>(item % heads);
    if (head < KV * G) {
        float x[attn::DPL];
        mx::act_span<attn::DPL, true>(in.query_gate, ATTN_QUERY_AT(row, head) + lane * attn::DPL, x);
        attn::norm_rotary(x, in.query_norm, SEISMIC_QUERY_NORM_STRIDE_0, in, row, exchange[warp],
                          lane);
        u16 *to = queries + (row * KV * G + head) * W + lane * attn::DPL;
#pragma unroll
        for (int d = 0; d < attn::DPL; ++d) to[d] = operand(mx::act_round(x[d]));
        return;
    }
    const int kv = head - KV * G;
    float k[attn::DPL];
    attn::prepared_key(in, row, kv, k, exchange[warp], lane);
    const u64 at = (row * KV + kv) * W + lane * attn::DPL;
#pragma unroll
    for (int d = 0; d < attn::DPL; ++d) mx::act_store(keys, at + d, k[d]);
    attn::append(in, row, kv, k, lane);
}

extern "C" __global__ void __launch_bounds__(WARPS * 32, 1)
    qwen_attention_prefill_attend(SEISMIC_KERNEL_PARAMS) {
    const attn::Inputs in = ATTN_INPUTS();
    const u8 *queries = SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_QUERIES);
    const u8 *keys = SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_KEYS);
    u8 *gated = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    const int kv = blockIdx.y;
    const long long first_token = static_cast<long long>(blockIdx.x) * QT;
    const long long rows_total = static_cast<long long>(SEISMIC_DIM_M);
    const int spans = static_cast<int>(SEISMIC_DIM_R);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int g = lane / 4;
    const int t = lane % 4;
    const float query_scale = in.scale * attn::LOG2E;

    extern __shared__ __align__(16) u8 shared[];
    u8 *q_tile = shared;
    // Two K/V stages after the query tile: stage b holds K then V.
    u8 *kv_tiles = q_tile + ROWS * W * 2;
    auto k_tile = [&](int b) { return kv_tiles + b * 2 * KEYS * W * 2; };
    auto v_tile = [&](int b) { return kv_tiles + b * 2 * KEYS * W * 2 + KEYS * W * 2; };
    int *table = reinterpret_cast<int *>(kv_tiles + 4 * KEYS * W * 2);  // [R + 1][4]

    // Query tile: matrix row i is token first_token + i / G, head kv * G + i % G.
    stage<false>(q_tile, queries, ROWS, [&](int i) -> long long {
        const long long token = first_token + i / G;
        if (token >= rows_total) return -1;
        return (token * KV * G + kv * G + i % G) * W;
    });
    seismic_cp_async_commit();

    // Span table: union and common interval of the tile's valid tokens.
    for (int span = threadIdx.x; span <= spans; span += WARPS * 32) {
        int union_lo = 0x7fffffff, union_hi = -0x7fffffff, common_lo = -0x7fffffff,
            common_hi = 0x7fffffff;
        for (long long token = first_token; token < min(first_token + QT, rows_total); ++token) {
            const attn::Span s = attn::row_span(in, token, span, spans);
            common_lo = max(common_lo, s.lo);
            common_hi = min(common_hi, s.hi);
            if (s.hi > s.lo) {
                union_lo = min(union_lo, s.lo);
                union_hi = max(union_hi, s.hi);
            }
        }
        if (union_hi <= union_lo) union_lo = union_hi = 0;
        table[span * 4 + 0] = union_lo;
        table[span * 4 + 1] = union_hi;
        table[span * 4 + 2] = common_lo;
        table[span * 4 + 3] = common_hi;
    }
    __syncthreads();

    auto settle = [&](Tile tile) {
        while (tile.span <= spans && tile.first >= table[tile.span * 4 + 1]) {
            ++tile.span;
            if (tile.span <= spans) tile.first = table[tile.span * 4 + 0];
        }
        return tile;
    };
    auto issue = [&](Tile tile, int buffer) {
        const int limit = table[tile.span * 4 + 1];
        if (tile.span < spans) {
            stage<F32_ACTIVATION>(k_tile(buffer), in.history_key, KEYS, [&](int r) -> long long {
                const int token = tile.first + r;
                return token < limit ? static_cast<long long>(ATTN_HISTORY_KEY_AT(token, kv)) : -1;
            });
            stage<F32_ACTIVATION>(v_tile(buffer), in.history_value, KEYS, [&](int r) -> long long {
                const int token = tile.first + r;
                return token < limit ? static_cast<long long>(ATTN_HISTORY_VALUE_AT(token, kv)) : -1;
            });
        } else {
            stage<F32_ACTIVATION>(k_tile(buffer), keys, KEYS, [&](int r) -> long long {
                const int token = tile.first + r;
                return token < limit ? (static_cast<long long>(token) * KV + kv) * W : -1;
            });
            stage<F32_ACTIVATION>(v_tile(buffer), in.value, KEYS, [&](int r) -> long long {
                const int token = tile.first + r;
                return token < limit ? static_cast<long long>(ATTN_VALUE_AT(token, kv)) : -1;
            });
        }
        seismic_cp_async_commit();
    };

    // This lane's two matrix rows (g and g + 8 of the warp's 16).
    const long long token_a = first_token + (warp * 16 + g) / G;
    const long long token_b = first_token + (warp * 16 + g + 8) / G;
    const bool valid_a = token_a < rows_total;
    const bool valid_b = token_b < rows_total;

    const float NEG_INF = -__int_as_float(0x7f800000);
    float o[W / 8][4];
#pragma unroll
    for (int n = 0; n < W / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float maximum[2] = {NEG_INF, NEG_INF};
    float denominator[2] = {0.0f, 0.0f};

    Tile current = settle(Tile{0, table[0]});
    int buffer = 0;
    int cached_span = -1;
    attn::Span interval_a{0, 0}, interval_b{0, 0};
    if (current.span <= spans) issue(current, 0);
    while (current.span <= spans) {
        const Tile next = settle(Tile{current.span, current.first + KEYS});
        if (next.span <= spans) {
            issue(next, buffer ^ 1);
            seismic_cp_async_wait<1>();
        } else {
            seismic_cp_async_wait<0>();
        }
        __syncthreads();

        if (current.span != cached_span) {
            cached_span = current.span;
            interval_a = valid_a ? attn::row_span(in, token_a, current.span, spans) : attn::Span{0, 0};
            interval_b = valid_b ? attn::row_span(in, token_b, current.span, spans) : attn::Span{0, 0};
        }
        const bool full = current.first >= table[current.span * 4 + 2] &&
                          current.first + KEYS <= table[current.span * 4 + 3];

        // S = Q K^T over W.
        float s[KEYS / 8][4];
#pragma unroll
        for (int n = 0; n < KEYS / 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
        const u8 *k_base = k_tile(buffer);
#pragma unroll
        for (int step = 0; step < W / 16; ++step) {
            u32 a[4];
            {
                const int row = warp * 16 + (lane % 8) + 8 * ((lane / 8) % 2);
                const int chunk = 2 * step + lane / 16;
                seismic_ldmatrix_x4(a, q_tile + swizzled(row, chunk));
            }
#pragma unroll
            for (int n = 0; n < KEYS / 8; n += 2) {
                u32 b4[4];
                const int row = 8 * (n + lane / 16) + (lane % 8);
                const int chunk = 2 * step + (lane / 8) % 2;
                seismic_ldmatrix_x4(b4, k_base + swizzled(row, chunk));
                const u32 b0[2] = {b4[0], b4[1]};
                const u32 b1[2] = {b4[2], b4[3]};
                mma(s[n], a, b0);
                mma(s[n + 1], a, b1);
            }
        }

        // Scale into the exp2 domain, mask, then the online softmax of both rows.
#pragma unroll
        for (int n = 0; n < KEYS / 8; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) s[n][e] *= query_scale;
        if (!full) {
#pragma unroll
            for (int n = 0; n < KEYS / 8; ++n) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int key = current.first + 8 * n + 2 * t + (e & 1);
                    const attn::Span &interval = e < 2 ? interval_a : interval_b;
                    if (key < interval.lo || key >= interval.hi) s[n][e] = NEG_INF;
                }
            }
        }
        float carry[2];
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            float row_max = maximum[half];
#pragma unroll
            for (int n = 0; n < KEYS / 8; ++n)
                row_max = fmaxf(row_max, fmaxf(s[n][2 * half], s[n][2 * half + 1]));
            row_max = fmaxf(row_max, seismic_shfl_xor_f32(row_max, 1));
            row_max = fmaxf(row_max, seismic_shfl_xor_f32(row_max, 2));
            const float base = row_max == NEG_INF ? 0.0f : row_max;
            carry[half] = seismic_ex2_approx(maximum[half] - base);
            maximum[half] = row_max;
            float sum = 0.0f;
#pragma unroll
            for (int n = 0; n < KEYS / 8; ++n) {
                s[n][2 * half] = seismic_ex2_approx(s[n][2 * half] - base);
                s[n][2 * half + 1] = seismic_ex2_approx(s[n][2 * half + 1] - base);
                sum += s[n][2 * half] + s[n][2 * half + 1];
            }
            denominator[half] = __fmaf_rn(denominator[half], carry[half], sum);
        }
#pragma unroll
        for (int n = 0; n < W / 8; ++n) {
            o[n][0] *= carry[0];
            o[n][1] *= carry[0];
            o[n][2] *= carry[1];
            o[n][3] *= carry[1];
        }

        // O += P V, P from the S registers.
        const u8 *v_base = v_tile(buffer);
#pragma unroll
        for (int step = 0; step < KEYS / 16; ++step) {
            const u32 p[4] = {operand_pair(s[2 * step][0], s[2 * step][1]),
                              operand_pair(s[2 * step][2], s[2 * step][3]),
                              operand_pair(s[2 * step + 1][0], s[2 * step + 1][1]),
                              operand_pair(s[2 * step + 1][2], s[2 * step + 1][3])};
#pragma unroll
            for (int n = 0; n < W / 8; n += 2) {
                u32 b4[4];
                const int row = 16 * step + (lane % 8) + 8 * ((lane / 8) % 2);
                const int chunk = n + lane / 16;
                seismic_ldmatrix_x4_trans(b4, v_base + swizzled(row, chunk));
                const u32 b0[2] = {b4[0], b4[1]};
                const u32 b1[2] = {b4[2], b4[3]};
                mma(o[n], p, b0);
                mma(o[n + 1], p, b1);
            }
        }
        __syncthreads();
        current = next;
        buffer ^= 1;
    }

    // Normalize, gate and store.
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        float l = denominator[half];
        l += seismic_shfl_xor_f32(l, 1);
        l += seismic_shfl_xor_f32(l, 2);
        denominator[half] = fmaxf(l, 1e-30f);
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const long long token = half == 0 ? token_a : token_b;
        if (token >= rows_total) continue;
        const int query_head = kv * G + (warp * 16 + g + 8 * half) % G;
        const u64 gate_at = ATTN_GATE_AT(token, query_head);
        const u64 out_at = static_cast<u64>(token) * SEISMIC_RESULT_0_STRIDE_0 +
                           static_cast<u64>(query_head) * SEISMIC_RESULT_0_STRIDE_1;
#pragma unroll
        for (int n = 0; n < W / 8; ++n) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int column = 8 * n + 2 * t + e;
                const float gate = mx::act_load(in.query_gate, gate_at + column);
                mx::act_store(gated, out_at + column * SEISMIC_RESULT_0_STRIDE_2,
                              o[n][2 * half + e] / denominator[half] / (1.0f + expf(-gate)));
            }
        }
    }
}
