// The prefill attention entries' bodies (`qwen_attention_prefill`,
// `qwen_attention_prefill_k8v4`, M >= 16): flash attention on tensor cores
// over a history policy (`attention::DenseHistory`,
// `attention::AffineHistory`) that appends a row's key and value.
//
// L1 `prepare`, one warp per (row, query or kv head): the query (norm_rotary,
// rounded to the activation element) goes to scratch as the policy's 16-bit
// MMA operand (`Operands`; scores are scaled into the exp2 domain after
// Q K^T); the prepared key (rounded to the activation element, as history
// stores it) and the value go to scratch as operands too, and the row's K/V
// is appended at its destination through the policy.
// L2 `attend`, one block per (row tile, kv head): the tile's matrix rows are
// QT tokens x G query heads (16 per warp). Each span in order (the R visible
// history spans, then the fresh span of batch rows) is scanned over the union
// of the tile rows' intervals in KEYS-key K/V tiles held in two operand
// stages. S = Q K^T and O += P V run as m16n8k16 MMAs with F32 accumulation;
// P stays in registers (FA2). Per-row interval masks apply only to K/V tiles
// outside the rows' common interval, and tiles past every row's interval (the
// causal tail) are never loaded. The sigmoid gate is fused into the store.
//
// Dense history and fresh tiles are copied with `cp.async`, the next tile's
// copy overlapping the current tile's products. Affine history is
// warp-specialized: WARPS producer warps beside the WARPS MMA warps load a
// tile's codes and group (scale, zero) pairs and store the decoded values
// code * scale + zero, rounded to the operand element, while the MMA warps run
// the previous tile; named barriers hand the two stages back and forth. The
// products are then the dense ones over the decoded history (the Metal and
// Vulkan staging).

#include "common/attention.cuh"

namespace attention {
namespace prefill {

constexpr int WARPS = SEISMIC_TUNE_WARPS;
constexpr int ROWS = WARPS * 16;
constexpr int QT = ROWS / G;
constexpr int KEYS = 32;
constexpr int CHUNKS = W / 8;  // 16-byte chunks per 16-bit row
static_assert(16 % G == 0, "a warp's 16 matrix rows hold whole tokens");

// MMA operand elements (16-bit floats): the element, a packed pair, and the
// m16n8k16 product with F32 accumulation.
struct Bf16Operands {
    __device__ static __forceinline__ u16 operand(float value) { return seismic_f32_to_bf16(value); }
    __device__ static __forceinline__ u32 pair(float lo, float hi) {
        return seismic_pack_bf16x2(lo, hi);
    }
    __device__ static __forceinline__ void mma(float (&acc)[4], const u32 (&a)[4], const u32 (&b)[2]) {
        seismic_mma_m16n8k16_bf16(acc, a, b);
    }
};
struct F16Operands {
    __device__ static __forceinline__ u16 operand(float value) { return seismic_f32_to_f16(value); }
    __device__ static __forceinline__ u32 pair(float lo, float hi) {
        return seismic_pack_f16x2(lo, hi);
    }
    __device__ static __forceinline__ void mma(float (&acc)[4], const u32 (&a)[4], const u32 (&b)[2]) {
        seismic_mma_m16n8k16_f16(acc, a, b);
    }
};

// The products' operands over a history policy. Dense history: the
// activation's element (bf16 for bf16 activations, f16 otherwise). Affine
// history: f16, the codec's coefficient element, so decoded 8-bit keys keep
// their precision (a bf16 rounding costs up to a code step); activation
// values enter exactly within the codec's range.
template <class History> struct Operands;
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
template <> struct Operands<DenseHistory> : Bf16Operands {};
#else
template <> struct Operands<DenseHistory> : F16Operands {};
#endif
template <> struct Operands<AffineHistory> : F16Operands {};

// Swizzled shared layout of a 16-bit [rows][W] tile: 16-byte chunk c of row r
// sits at chunk c ^ (r % SWIZZLE), so ldmatrix row sets are bank-conflict
// free (at W = 32 two rows share a bank set).
constexpr int SWIZZLE = CHUNKS < 8 ? CHUNKS : 8;
__device__ __forceinline__ u32 swizzled(int row, int chunk) {
    return static_cast<u32>(row * CHUNKS + (chunk ^ (row & (SWIZZLE - 1)))) * 16;
}

// Stage `rows` rows of a [.., W] source into a swizzled 16-bit tile, by
// `threads` threads of which this is `thread`. `source(r)` is the element
// offset of row r, or -1 for a zero row. An f32 source (dense history of f32
// activations) converts to the dense operand element; a 16-bit source
// (activation or operand) is copied with `cp.async`.
template <bool F32_SOURCE, class Source>
__device__ __forceinline__ void stage(u8 *tile, const u8 *base, int rows, Source source,
                                      int thread, int threads) {
    typedef Operands<DenseHistory> Ops;
    for (int index = thread; index < rows * CHUNKS; index += threads) {
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
                packed = make_uint4(Ops::pair(a.x, a.y), Ops::pair(a.z, a.w), Ops::pair(b.x, b.y),
                                    Ops::pair(b.z, b.w));
            }
            *reinterpret_cast<uint4 *>(destination) = packed;
        } else {
            const u8 *from = base + (at >= 0 ? static_cast<u64>(at) + chunk * 8 : 0) * 2;
            seismic_cp_async_16_zfill(destination, from, at >= 0 ? 16u : 0u);
        }
    }
}

constexpr bool F32_ACTIVATION = Act::bytes == 4;

// Dense history K/V tiles of tokens [first, first + KEYS) (zero at or past
// `limit`), copied by the block's WARPS warps.
__device__ __forceinline__ void stage_history(const DenseHistory &history, u8 *k_tile,
                                              u8 *v_tile, int first, int limit, int kv) {
    stage<F32_ACTIVATION>(k_tile, history.key, KEYS, [&](int r) -> long long {
        const int token = first + r;
        return token < limit ? static_cast<long long>(history.key_at(token, kv)) : -1;
    }, threadIdx.x, WARPS * 32);
    stage<F32_ACTIVATION>(v_tile, history.value, KEYS, [&](int r) -> long long {
        const int token = first + r;
        return token < limit ? static_cast<long long>(history.value_at(token, kv)) : -1;
    }, threadIdx.x, WARPS * 32);
}

// ---------------------------------------------------------------------------
// Affine history tiles, produced by the producer warps.

// Producer threads, the 16-byte code pieces of a key and a value row, and the
// codes of one piece (all in one group).
constexpr int PRODUCERS = WARPS * 32;
constexpr int KEY_PIECES = W * KEY_BITS / 128;
constexpr int VALUE_PIECES = W * VALUE_BITS / 128;
constexpr int KEY_ITEMS = (KEYS * KEY_PIECES + PRODUCERS - 1) / PRODUCERS;
constexpr int VALUE_ITEMS = (KEYS * VALUE_PIECES + PRODUCERS - 1) / PRODUCERS;
constexpr int KEY_PIECE_CODES = 128 / KEY_BITS;
constexpr int VALUE_PIECE_CODES = 128 / VALUE_BITS;
static_assert(GROUP % KEY_PIECE_CODES == 0 && GROUP % VALUE_PIECE_CODES == 0,
              "a code piece lies in one group");
constexpr float MAGIC = 8388608.0f;  // 2^23

// Eight 8-bit codes (two words) decoded with their group's (scale, zero)
// `pair` into one operand chunk. A code in a float's mantissa (no conversion
// instruction) is exact.
__device__ __forceinline__ uint4 key_chunk(u32 low, u32 high, float2 pair) {
    float f[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        f[i] = __fmaf_rn(__uint_as_float(__byte_perm(i < 4 ? low : high, 0x4B000000u,
                                                     0x7440u | static_cast<u32>(i % 4))) -
                             MAGIC,
                         pair.x, pair.y);
    typedef Operands<AffineHistory> Ops;
    return make_uint4(Ops::pair(f[0], f[1]), Ops::pair(f[2], f[3]), Ops::pair(f[4], f[5]),
                      Ops::pair(f[6], f[7]));
}

// Eight 4-bit codes (one word) decoded with their group's pair into one
// operand chunk.
__device__ __forceinline__ uint4 value_chunk(u32 word, float2 pair) {
    float f[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        f[i] = __fmaf_rn(__uint_as_float(0x4B000000u | ((word >> (4 * i)) & 0xFu)) - MAGIC, pair.x,
                         pair.y);
    typedef Operands<AffineHistory> Ops;
    return make_uint4(Ops::pair(f[0], f[1]), Ops::pair(f[2], f[3]), Ops::pair(f[4], f[5]),
                      Ops::pair(f[6], f[7]));
}

// The affine tile of tokens [first, first + KEYS) decoded into the swizzled K
// and V operand tiles by producer thread `thread`, each piece with its group's
// (scale, zero) pair. Rows at or past `limit` get zero codes and zero pairs,
// so they decode to exact zeros. Every load is issued before any conversion,
// so they are in flight together.
__device__ __forceinline__ void produce(const AffineHistory &history, u8 *k_tile, u8 *v_tile,
                                        int first, int limit, int kv, int thread) {
    uint4 key_bits[KEY_ITEMS];
    u32 key_pairs[KEY_ITEMS];
    uint4 value_bits[VALUE_ITEMS];
    u32 value_pairs[VALUE_ITEMS];
#pragma unroll
    for (int k = 0; k < KEY_ITEMS; ++k) {
        const int index = thread + k * PRODUCERS;
        const int token = first + index / KEY_PIECES;
        const int piece = index % KEY_PIECES;
        const bool live = index < KEYS * KEY_PIECES && token < limit;
        key_bits[k] = live ? seismic_ld_nc_v4(history.key_row(token, kv) + piece * 4)
                           : make_uint4(0, 0, 0, 0);
        key_pairs[k] = live ? seismic_ld_nc_u32(history.key_pair(token, kv) +
                                                piece * KEY_PIECE_CODES / GROUP)
                            : 0u;
    }
#pragma unroll
    for (int k = 0; k < VALUE_ITEMS; ++k) {
        const int index = thread + k * PRODUCERS;
        const int token = first + index / VALUE_PIECES;
        const int piece = index % VALUE_PIECES;
        const bool live = index < KEYS * VALUE_PIECES && token < limit;
        value_bits[k] = live ? seismic_ld_nc_v4(history.value_row(token, kv) + piece * 4)
                             : make_uint4(0, 0, 0, 0);
        value_pairs[k] = live ? seismic_ld_nc_u32(history.value_pair(token, kv) +
                                                  piece * VALUE_PIECE_CODES / GROUP)
                              : 0u;
    }
#pragma unroll
    for (int k = 0; k < KEY_ITEMS; ++k) {
        const int index = thread + k * PRODUCERS;
        if (index < KEYS * KEY_PIECES) {
            // A 16-byte key piece: 16 codes, two operand chunks.
            const int r = index / KEY_PIECES;
            const int chunk = (index % KEY_PIECES) * 2;
            const float2 pair = seismic_unpack_f16x2(key_pairs[k]);
            *reinterpret_cast<uint4 *>(k_tile + swizzled(r, chunk)) =
                key_chunk(key_bits[k].x, key_bits[k].y, pair);
            *reinterpret_cast<uint4 *>(k_tile + swizzled(r, chunk + 1)) =
                key_chunk(key_bits[k].z, key_bits[k].w, pair);
        }
    }
#pragma unroll
    for (int k = 0; k < VALUE_ITEMS; ++k) {
        const int index = thread + k * PRODUCERS;
        if (index < KEYS * VALUE_PIECES) {
            // A 16-byte value piece: 32 codes, four operand chunks.
            const int r = index / VALUE_PIECES;
            const int chunk = (index % VALUE_PIECES) * 4;
            const float2 pair = seismic_unpack_f16x2(value_pairs[k]);
            *reinterpret_cast<uint4 *>(v_tile + swizzled(r, chunk)) = value_chunk(value_bits[k].x, pair);
            *reinterpret_cast<uint4 *>(v_tile + swizzled(r, chunk + 1)) =
                value_chunk(value_bits[k].y, pair);
            *reinterpret_cast<uint4 *>(v_tile + swizzled(r, chunk + 2)) =
                value_chunk(value_bits[k].z, pair);
            *reinterpret_cast<uint4 *>(v_tile + swizzled(r, chunk + 3)) =
                value_chunk(value_bits[k].w, pair);
        }
    }
}

// Named barriers between the MMA and the producer warps (0 is
// `__syncthreads`): stage b is full (produced) at FULL + b and empty
// (consumed) at EMPTY + b; the MMA warps' own barrier is MMA.
constexpr u32 FULL = 1;
constexpr u32 EMPTY = 3;
constexpr u32 MMA_WARPS = 5;
__device__ __forceinline__ void named_sync(u32 id, u32 threads) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}
__device__ __forceinline__ void named_arrive(u32 id, u32 threads) {
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

struct Tile {
    int span;
    int first;
};

// L1 over `M` rows: 8 warps per block (launch with 256 threads). Queries,
// keys and values go to scratch as the history policy's operands ([M, KV * G,
// W], [M, KV, W] and [M, KV, W]).
template <class History>
__device__ __forceinline__ void prepare(const Inputs &in, const History &history, u16 *queries,
                                        u16 *keys, u16 *values) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    typedef Operands<History> Ops;
    __shared__ float exchange[8][W];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const u64 item = static_cast<u64>(blockIdx.x) * 8 + warp;
    const u64 heads = KV * (G + 1);
    if (item >= SEISMIC_DIM_M * heads) return;
    const u64 row = item / heads;
    const int head = static_cast<int>(item % heads);
    if (head < KV * G) {
        float x[DPL];
        element::span<Act, DPL, true>(in.query_gate, ATTENTION_QUERY_AT(row, head) + lane * DPL, x);
        norm_rotary(x, in.query_norm, SEISMIC_QUERY_NORM_STRIDE_0, in, row, exchange[warp], lane);
        u16 *to = queries + (row * KV * G + head) * W + lane * DPL;
#pragma unroll
        for (int d = 0; d < DPL; ++d) to[d] = Ops::operand(Act::round(x[d]));
        return;
    }
    const int kv = head - KV * G;
    float k[DPL];
    float v[DPL];
    prepared_key(in, row, kv, k, exchange[warp], lane);
    fresh_value(in, row, kv, v, lane);
    const u64 at = (row * KV + kv) * W + lane * DPL;
#pragma unroll
    for (int d = 0; d < DPL; ++d) {
        keys[at + d] = Ops::operand(k[d]);
        values[at + d] = Ops::operand(v[d]);
    }
    const int destination = ATTENTION_DESTINATION(in, row);
    if (destination >= 0) history.append(destination, kv, k, v, lane);
}

// L2: block (row tile, kv head): WARPS MMA warps, plus WARPS producer warps
// for affine history (launch with WARPS * 32 threads for dense history,
// WARPS * 64 for affine). Shared memory: the query tile [ROWS][W], two
// operand stages each K [KEYS][W] then V [KEYS][W] 16-bit, then the span
// table [R + 1][4].
template <class History>
__device__ __forceinline__ void attend(const Inputs &in, const History &history, const u8 *queries,
                                       const u8 *keys, const u8 *values, u8 *gated) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    typedef Operands<History> Ops;
    constexpr bool CODED = History::CODED;
    constexpr int THREADS = CODED ? WARPS * 64 : WARPS * 32;
    const int kv = blockIdx.y;
    const long long first_token = static_cast<long long>(blockIdx.x) * QT;
    const long long rows_total = static_cast<long long>(SEISMIC_DIM_M);
    const int spans = static_cast<int>(SEISMIC_DIM_R);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int g = lane / 4;
    const int t = lane % 4;
    const float query_scale = in.scale * LOG2E;

    extern __shared__ __align__(16) u8 shared[];
    u8 *q_tile = shared;
    u8 *kv_tiles = q_tile + ROWS * W * 2;
    auto k_tile = [&](int b) { return kv_tiles + b * 2 * KEYS * W * 2; };
    auto v_tile = [&](int b) { return kv_tiles + b * 2 * KEYS * W * 2 + KEYS * W * 2; };
    int *table = reinterpret_cast<int *>(kv_tiles + 4 * KEYS * W * 2);  // [R + 1][4]

    // Query tile (MMA warps): matrix row i is token first_token + i / G, head
    // kv * G + i % G.
    if (warp < WARPS) {
        stage<false>(q_tile, queries, ROWS, [&](int i) -> long long {
            const long long token = first_token + i / G;
            if (token >= rows_total) return -1;
            return (token * KV * G + kv * G + i % G) * W;
        }, threadIdx.x, WARPS * 32);
        seismic_cp_async_commit();
    }

    // Span table: union and common interval of the tile's valid tokens.
    for (int index = threadIdx.x; index <= spans; index += THREADS) {
        int union_lo = 0x7fffffff, union_hi = -0x7fffffff, common_lo = -0x7fffffff,
            common_hi = 0x7fffffff;
        for (long long token = first_token; token < min(first_token + QT, rows_total); ++token) {
            const Span s = span(in, token, index, spans);
            common_lo = max(common_lo, s.lo);
            common_hi = min(common_hi, s.hi);
            if (s.hi > s.lo) {
                union_lo = min(union_lo, s.lo);
                union_hi = max(union_hi, s.hi);
            }
        }
        if (union_hi <= union_lo) union_lo = union_hi = 0;
        table[index * 4 + 0] = union_lo;
        table[index * 4 + 1] = union_hi;
        table[index * 4 + 2] = common_lo;
        table[index * 4 + 3] = common_hi;
    }
    __syncthreads();

    auto settle = [&](Tile tile) {
        while (tile.span <= spans && tile.first >= table[tile.span * 4 + 1]) {
            ++tile.span;
            if (tile.span <= spans) tile.first = table[tile.span * 4 + 0];
        }
        return tile;
    };
    // Fresh tiles (the batch's prepared keys and values, operands in scratch)
    // into operand stage b, by `threads` threads of which this is `thread`.
    auto stage_fresh = [&](Tile tile, int buffer, int thread, int threads) {
        const int limit = table[tile.span * 4 + 1];
        auto row = [&](int r) -> long long {
            const int token = tile.first + r;
            return token < limit ? (static_cast<long long>(token) * KV + kv) * W : -1;
        };
        stage<false>(k_tile(buffer), keys, KEYS, row, thread, threads);
        stage<false>(v_tile(buffer), values, KEYS, row, thread, threads);
    };

    if constexpr (CODED) {
        if (warp >= WARPS) {
            // Producer warps: every tile in order into stage i % 2, once its
            // previous occupant (tile i - 2) is consumed.
            const int thread = threadIdx.x - WARPS * 32;
            Tile tile = settle(Tile{0, table[0]});
            for (int i = 0; tile.span <= spans; ++i) {
                const int b = i % 2;
                if (i >= 2) named_sync(EMPTY + b, THREADS);
                if (tile.span < spans) {
                    produce(history, k_tile(b), v_tile(b), tile.first, table[tile.span * 4 + 1],
                            kv, thread);
                } else {
                    stage_fresh(tile, b, thread, PRODUCERS);
                    seismic_cp_async_commit();
                    seismic_cp_async_wait<0>();
                }
                __threadfence_block();
                named_arrive(FULL + b, THREADS);
                tile = settle(Tile{tile.span, tile.first + KEYS});
            }
            return;
        }
    }

    // MMA warps. This lane's two matrix rows (g and g + 8 of the warp's 16).
    const long long token_a = first_token + (warp * 16 + g) / G;
    const long long token_b = first_token + (warp * 16 + g + 8) / G;
    const bool valid_a = token_a < rows_total;
    const bool valid_b = token_b < rows_total;

    // Affine: the producers stage every K/V tile, so the MMA warps complete
    // their query tile copy here.
    if constexpr (CODED) {
        seismic_cp_async_wait<0>();
        named_sync(MMA_WARPS, WARPS * 32);
    }

    const float NEG_INF = -__int_as_float(0x7f800000);
    float o[W / 8][4];
#pragma unroll
    for (int n = 0; n < W / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float maximum[2] = {NEG_INF, NEG_INF};
    float denominator[2] = {0.0f, 0.0f};
    int cached_span = -1;
    Span interval_a{0, 0}, interval_b{0, 0};

    // The products of one staged tile.
    auto absorb_tile = [&](Tile current, const u8 *k_base, const u8 *v_base) {
        if (current.span != cached_span) {
            cached_span = current.span;
            interval_a = valid_a ? span(in, token_a, current.span, spans) : Span{0, 0};
            interval_b = valid_b ? span(in, token_b, current.span, spans) : Span{0, 0};
        }
        const bool full = current.first >= table[current.span * 4 + 2] &&
                          current.first + KEYS <= table[current.span * 4 + 3];

        // S = Q K^T over W.
        float s[KEYS / 8][4];
#pragma unroll
        for (int n = 0; n < KEYS / 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
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
                Ops::mma(s[n], a, b0);
                Ops::mma(s[n + 1], a, b1);
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
                    const Span &interval = e < 2 ? interval_a : interval_b;
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
#pragma unroll
        for (int step = 0; step < KEYS / 16; ++step) {
            const u32 p[4] = {Ops::pair(s[2 * step][0], s[2 * step][1]),
                              Ops::pair(s[2 * step][2], s[2 * step][3]),
                              Ops::pair(s[2 * step + 1][0], s[2 * step + 1][1]),
                              Ops::pair(s[2 * step + 1][2], s[2 * step + 1][3])};
#pragma unroll
            for (int n = 0; n < W / 8; n += 2) {
                u32 b4[4];
                const int row = 16 * step + (lane % 8) + 8 * ((lane / 8) % 2);
                const int chunk = n + lane / 16;
                seismic_ldmatrix_x4_trans(b4, v_base + swizzled(row, chunk));
                const u32 b0[2] = {b4[0], b4[1]};
                const u32 b1[2] = {b4[2], b4[3]};
                Ops::mma(o[n], p, b0);
                Ops::mma(o[n + 1], p, b1);
            }
        }
    };

    Tile current = settle(Tile{0, table[0]});
    if constexpr (CODED) {
        // Tile i waits for stage i % 2 to be full; once its products are
        // issued, the stage is released for tile i + 2 when that tile exists.
        for (int i = 0; current.span <= spans; ++i) {
            const int b = i % 2;
            named_sync(FULL + b, THREADS);
            absorb_tile(current, k_tile(b), v_tile(b));
            const Tile next = settle(Tile{current.span, current.first + KEYS});
            if (next.span <= spans && settle(Tile{next.span, next.first + KEYS}).span <= spans)
                named_arrive(EMPTY + b, THREADS);
            current = next;
        }
    } else {
        auto issue = [&](Tile tile, int buffer) {
            if (tile.span < spans) {
                stage_history(history, k_tile(buffer), v_tile(buffer), tile.first,
                              table[tile.span * 4 + 1], kv);
            } else {
                stage_fresh(tile, buffer, threadIdx.x, WARPS * 32);
            }
            seismic_cp_async_commit();
        };
        int buffer = 0;
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
            absorb_tile(current, k_tile(buffer), v_tile(buffer));
            __syncthreads();
            current = next;
            buffer ^= 1;
        }
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
        const u64 gate_at = ATTENTION_GATE_AT(token, query_head);
        const u64 out_at = static_cast<u64>(token) * SEISMIC_RESULT_0_STRIDE_0 +
                           static_cast<u64>(query_head) * SEISMIC_RESULT_0_STRIDE_1;
#pragma unroll
        for (int n = 0; n < W / 8; ++n) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int column = 8 * n + 2 * t + e;
                const float gate = element::at<Act>(in.query_gate, gate_at + column);
                element::put<Act>(gated, out_at + column * SEISMIC_RESULT_0_STRIDE_2,
                                  o[n][2 * half + e] / denominator[half] / (1.0f + expf(-gate)));
            }
        }
    }
}

}  // namespace prefill
}  // namespace attention
