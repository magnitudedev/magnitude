// The prefill attention entries' bodies (`attention_prefill`,
// `attention_prefill_k8v4`, M >= 16): flash attention on tensor cores
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
// causal tail) are never loaded. The output gate is fused into the store. A
// head wider than 256 makes one such scan per 256-column output window.
//
// Dense history and fresh tiles are copied with `cp.async`, the next tile's
// copy overlapping the current tile's products. Affine history is
// warp-specialized: WARPS producer warps beside the WARPS MMA warps load a
// tile's codes and group (scale, zero) pairs and store the decoded values
// code * scale + zero, rounded to the operand element, while the MMA warps run
// the previous tile; named barriers hand the two stages back and forth. The
// products are then the dense ones over the decoded history (the Metal and
// Vulkan staging).

#include "attention.cuh"

namespace attention {
namespace prefill {

constexpr int WARPS = SEISMIC_TUNE_WARPS;
constexpr int ROWS = WARPS * 16;
// Tokens of a block: its matrix rows hold QT whole tokens of G query heads;
// the last ROWS - QT * G rows (none when G divides ROWS) are padding.
constexpr int QT = ROWS / G;
// Keys per K/V tile, and output columns per pass over the keys: a head wider
// than 256 takes 16-key tiles and one pass per 256-column output window, each
// recomputing the scores (so every pass's softmax statistics are identical)
// and staging only its window's V columns, which keeps the F32 outputs in
// registers and the stages within shared memory. Its key tiles also stage in
// 256-column pieces, one per operand stage, the scores accumulating over the
// pieces (the query tile is the only whole-head operand in shared memory).
constexpr int KEYS = W > 256 ? 16 : 32;
constexpr int WINDOW = W > 256 ? 256 : W;
constexpr int WINDOWS = W / WINDOW;
constexpr int PIECE = WINDOW;
constexpr int PIECES = W / PIECE;
constexpr int CHUNKS = W / 8;  // 16-byte chunks per 16-bit row
constexpr int WINDOW_CHUNKS = WINDOW / 8;
constexpr int PIECE_CHUNKS = PIECE / 8;
static_assert(QT >= 1, "a block holds at least one token's query heads");
static_assert(W % WINDOW == 0, "output windows tile the head");

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

// Swizzled shared layout of a 16-bit [rows][ROW_CHUNKS * 8] tile (the head,
// or an output window of it): 16-byte chunk c of row r sits at chunk c ^ (r %
// swizzle), so ldmatrix row sets are bank-conflict free (at 32 columns two
// rows share a bank set).
template <int ROW_CHUNKS = CHUNKS>
__device__ __forceinline__ u32 swizzled(int row, int chunk) {
    constexpr int SWIZZLE = ROW_CHUNKS < 8 ? ROW_CHUNKS : 8;
    return static_cast<u32>(row * ROW_CHUNKS + (chunk ^ (row & (SWIZZLE - 1)))) * 16;
}

// Stage `rows` rows of ROW_CHUNKS 16-byte chunks into a swizzled 16-bit tile,
// by `threads` threads of which this is `thread`. `source(r)` is the element
// offset of row r's first column, or -1 for a zero row. An f32 source (dense
// history of f32 activations) converts to the dense operand element; a 16-bit
// source (activation or operand) is copied with `cp.async`.
template <bool F32_SOURCE, int ROW_CHUNKS, class Source>
__device__ __forceinline__ void stage(u8 *tile, const u8 *base, int rows, Source source,
                                      int thread, int threads) {
    typedef Operands<DenseHistory> Ops;
    for (int index = thread; index < rows * ROW_CHUNKS; index += threads) {
        const int row = index / ROW_CHUNKS;
        const int chunk = index % ROW_CHUNKS;
        const long long at = source(row);
        u8 *destination = tile + swizzled<ROW_CHUNKS>(row, chunk);
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

// Slab history supplies a row pointer (at the row's first staged column) for
// each tile row; rows can cross a slab boundary within one key tile.
template <bool F32_SOURCE, int ROW_CHUNKS, class Source>
__device__ __forceinline__ void stage_slab(u8 *tile, int rows, Source source, int thread, int threads) {
    typedef Operands<DenseHistory> Ops;
    for (int index = thread; index < rows * ROW_CHUNKS; index += threads) {
        const int row = index / ROW_CHUNKS;
        const int chunk = index % ROW_CHUNKS;
        const u8 *base = source(row);
        u8 *destination = tile + swizzled<ROW_CHUNKS>(row, chunk);
        if constexpr (F32_SOURCE) {
            uint4 packed = make_uint4(0, 0, 0, 0);
            if (base != nullptr) {
                const float4 *from = reinterpret_cast<const float4 *>(base + chunk * 8 * 4);
                const float4 a = from[0];
                const float4 b = from[1];
                packed = make_uint4(Ops::pair(a.x, a.y), Ops::pair(a.z, a.w), Ops::pair(b.x, b.y),
                                    Ops::pair(b.z, b.w));
            }
            *reinterpret_cast<uint4 *>(destination) = packed;
        } else {
            const u8 *from = base != nullptr ? base + chunk * 16 : source(0);
            seismic_cp_async_16_zfill(destination, from, base != nullptr ? 16u : 0u);
        }
    }
}

constexpr bool F32_ACTIVATION = Act::bytes == 4;

// Dense history K piece tile (columns [piece * PIECE, (piece + 1) * PIECE)) of
// tokens [first, first + KEYS) (zero at or past `limit`), and with the last
// piece the V window tile (columns [column0, column0 + WINDOW)), copied by the
// block's WARPS warps.
__device__ __forceinline__ void stage_history(const DenseHistory &history, u8 *k_tile,
                                              u8 *v_tile, int first, int limit, int kv, int piece,
                                              int column0) {
    stage_slab<F32_ACTIVATION, PIECE_CHUNKS>(k_tile, KEYS, [&](int r) -> const u8 * {
        const int token = first + r;
        return token < limit ? history.key_vector(token, kv) + piece * PIECE * Act::bytes : nullptr;
    }, threadIdx.x, WARPS * 32);
    if (piece + 1 < PIECES) return;
    stage_slab<F32_ACTIVATION, WINDOW_CHUNKS>(v_tile, KEYS, [&](int r) -> const u8 * {
        const int token = first + r;
        return token < limit ? history.value_vector(token, kv) + column0 * Act::bytes : nullptr;
    }, threadIdx.x, WARPS * 32);
}

// ---------------------------------------------------------------------------
// Affine history tiles, produced by the producer warps.

// Producer threads, the 16-byte code pieces of a key and a value row, and the
// codes of one piece (all in one group).
constexpr int PRODUCERS = WARPS * 32;
// Key code pieces of one staged key piece.
constexpr int KEY_PIECES = PIECE * KEY_BITS / 128;
// Value pieces of one output window.
constexpr int VALUE_PIECES = WINDOW * VALUE_BITS / 128;
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
// operand tile (key piece `piece`) and, with the last piece, the V operand
// tile (the output window's columns from column0) by producer thread
// `thread`, each code piece with its group's (scale, zero) pair. Rows at or
// past `limit` get zero codes and zero pairs, so they decode to exact zeros.
// Every load is issued before any conversion, so they are in flight together.
__device__ __forceinline__ void produce(const AffineHistory &history, u8 *k_tile, u8 *v_tile,
                                        int first, int limit, int kv, int piece, int column0,
                                        int thread) {
    const int key_piece0 = piece * KEY_PIECES;
    const int value_piece0 = column0 * VALUE_BITS / 128;
    const bool values = piece + 1 == PIECES;
    uint4 key_bits[KEY_ITEMS];
    u32 key_pairs[KEY_ITEMS];
    uint4 value_bits[VALUE_ITEMS];
    u32 value_pairs[VALUE_ITEMS];
#pragma unroll
    for (int k = 0; k < KEY_ITEMS; ++k) {
        const int index = thread + k * PRODUCERS;
        const int token = first + index / KEY_PIECES;
        const int code = key_piece0 + index % KEY_PIECES;
        const bool live = index < KEYS * KEY_PIECES && token < limit;
        key_bits[k] = live ? seismic_ld_nc_v4(history.key_row(token, kv) + code * 4)
                           : make_uint4(0, 0, 0, 0);
        key_pairs[k] = live ? seismic_ld_nc_u32(history.key_pair(token, kv) +
                                                code * KEY_PIECE_CODES / GROUP)
                            : 0u;
    }
#pragma unroll
    for (int k = 0; k < VALUE_ITEMS; ++k) {
        const int index = thread + k * PRODUCERS;
        const int token = first + index / VALUE_PIECES;
        const int code = value_piece0 + index % VALUE_PIECES;
        const bool live = values && index < KEYS * VALUE_PIECES && token < limit;
        value_bits[k] = live ? seismic_ld_nc_v4(history.value_row(token, kv) + code * 4)
                             : make_uint4(0, 0, 0, 0);
        value_pairs[k] = live ? seismic_ld_nc_u32(history.value_pair(token, kv) +
                                                  code * VALUE_PIECE_CODES / GROUP)
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
            *reinterpret_cast<uint4 *>(k_tile + swizzled<PIECE_CHUNKS>(r, chunk)) =
                key_chunk(key_bits[k].x, key_bits[k].y, pair);
            *reinterpret_cast<uint4 *>(k_tile + swizzled<PIECE_CHUNKS>(r, chunk + 1)) =
                key_chunk(key_bits[k].z, key_bits[k].w, pair);
        }
    }
#pragma unroll
    for (int k = 0; k < VALUE_ITEMS; ++k) {
        const int index = thread + k * PRODUCERS;
        if (values && index < KEYS * VALUE_PIECES) {
            // A 16-byte value piece: 32 codes, four operand chunks.
            const int r = index / VALUE_PIECES;
            const int chunk = (index % VALUE_PIECES) * 4;
            const float2 pair = seismic_unpack_f16x2(value_pairs[k]);
            *reinterpret_cast<uint4 *>(v_tile + swizzled<WINDOW_CHUNKS>(r, chunk)) =
                value_chunk(value_bits[k].x, pair);
            *reinterpret_cast<uint4 *>(v_tile + swizzled<WINDOW_CHUNKS>(r, chunk + 1)) =
                value_chunk(value_bits[k].y, pair);
            *reinterpret_cast<uint4 *>(v_tile + swizzled<WINDOW_CHUNKS>(r, chunk + 2)) =
                value_chunk(value_bits[k].z, pair);
            *reinterpret_cast<uint4 *>(v_tile + swizzled<WINDOW_CHUNKS>(r, chunk + 3)) =
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

// Key partitions of a query tile: its key tiles (each span's union interval
// in KEYS steps, spans then fresh) split into runs of at least MIN_TILES over
// the grid's z extent. A tile served by one partition stores its gated output
// directly; otherwise each partition stores (partial output, maximum,
// denominator) per (row, query head) and `merge` combines them. `counts`
// holds each query tile's partition count. Entries without key partitions
// (a grid z extent of 1) pass null scratch.
constexpr int MIN_TILES = 16;
struct Split {
    float *partials;    // [parts][M][KV * G][W]
    float *statistics;  // [parts][M][KV * G][2]
    u32 *counts;        // [query tiles]
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
        prepared_query(in, row, head, x, exchange[warp], lane);
        u16 *to = queries + (row * KV * G + head) * W + lane * DPL;
#pragma unroll
        for (int d = 0; d < DPL; ++d) to[d] = Ops::operand(x[d]);
        return;
    }
    if (!FRESH) return;
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

// L2: block (row tile, kv head, key partition): WARPS MMA warps, plus WARPS
// producer warps for affine history (launch with WARPS * 32 threads for dense
// history, WARPS * 64 for affine). Shared memory: the query tile [ROWS][W],
// two operand stages each K [KEYS][PIECE] then V [KEYS][WINDOW] 16-bit, then
// the span table [R + 1][4]. The MMA warps make one pass over the partition's
// key tiles per output window; a key tile is PIECES staged units, the last
// carrying the window's values.
template <class History>
__device__ __forceinline__ void attend(const Inputs &in, const History &history, const u8 *queries,
                                       const u8 *keys, const u8 *values, u8 *gated, const Split &split) {
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
    constexpr int STAGE_BYTES = KEYS * (PIECE + WINDOW) * 2;
    auto k_tile = [&](int b) { return kv_tiles + b * STAGE_BYTES; };
    auto v_tile = [&](int b) { return kv_tiles + b * STAGE_BYTES + KEYS * PIECE * 2; };
    int *table = reinterpret_cast<int *>(kv_tiles + 2 * STAGE_BYTES);  // [R + 1][4]

    // Query tile (MMA warps): matrix row i is token first_token + i / G, head
    // kv * G + i % G; padding rows are zero.
    if (warp < WARPS) {
        stage<false, CHUNKS>(q_tile, queries, ROWS, [&](int i) -> long long {
            const long long token = first_token + i / G;
            if (i >= QT * G || token >= rows_total) return -1;
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

    // This block's key partition: tiles [tiles_lo, tiles_lo + count) of the
    // query tile's sequence.
    auto span_tiles = [&](int index) {
        const int lo = table[index * 4 + 0], hi = table[index * 4 + 1];
        return hi > lo ? (hi - lo + KEYS - 1) / KEYS : 0;
    };
    int total_tiles = 0;
    for (int index = 0; index <= spans; ++index) total_tiles += span_tiles(index);
    const int parts = static_cast<int>(gridDim.z);
    const int per = max(MIN_TILES, (total_tiles + parts - 1) / parts);
    const int active = max(1, (total_tiles + per - 1) / per);
    const int partition = static_cast<int>(blockIdx.z);
    if (partition >= active) {
        seismic_cp_async_wait<0>();
        return;
    }
    if (parts > 1 && partition == 0 && kv == 0 && threadIdx.x == 0) split.counts[blockIdx.x] = active;
    const int tiles_lo = partition * per;
    const int count = max(0, min(per, total_tiles - tiles_lo));
    // Tile `index` of the query tile's sequence.
    auto nth = [&](int index) {
        for (int span = 0; span <= spans; ++span) {
            const int n = span_tiles(span);
            if (index < n) return Tile{span, table[span * 4 + 0] + index * KEYS};
            index -= n;
        }
        return Tile{spans + 1, 0};
    };
    // Fresh tiles (the batch's prepared keys and values, operands in scratch:
    // the keys' piece, and with the last piece the values' window from
    // column0) into operand stage b, by `threads` threads of which this is
    // `thread`.
    auto stage_fresh = [&](Tile tile, int buffer, int piece, int column0, int thread, int threads) {
        const int limit = table[tile.span * 4 + 1];
        auto row = [&](int r) -> long long {
            const int token = tile.first + r;
            return token < limit ? (static_cast<long long>(token) * KV + kv) * W : -1;
        };
        stage<false, PIECE_CHUNKS>(k_tile(buffer), keys, KEYS, [&](int r) -> long long {
            const long long at = row(r);
            return at >= 0 ? at + piece * PIECE : -1;
        }, thread, threads);
        if (piece + 1 < PIECES) return;
        stage<false, WINDOW_CHUNKS>(v_tile(buffer), values, KEYS, [&](int r) -> long long {
            const long long at = row(r);
            return at >= 0 ? at + column0 : -1;
        }, thread, threads);
    };
    // The unit after (tile, piece): the tile's next key piece, else the next
    // tile's first.
    auto advance = [&](Tile &tile, int &piece) {
        if (++piece == PIECES) {
            piece = 0;
            tile = settle(Tile{tile.span, tile.first + KEYS});
        }
    };
    // Staged units per output window.
    const int units = count * PIECES;

    if constexpr (CODED) {
        if (warp >= WARPS) {
            // Producer warps: per output window, the partition's units in
            // order; staged unit i (counted over every window) goes into stage
            // i % 2 once its previous occupant (unit i - 2) is consumed.
            const int thread = threadIdx.x - WARPS * 32;
            for (int window = 0; window < WINDOWS; ++window) {
                const int column0 = window * WINDOW;
                Tile tile = nth(tiles_lo);
                int piece = 0;
                for (int j = 0; j < units; ++j) {
                    const int i = window * units + j;
                    const int b = i % 2;
                    if (i >= 2) named_sync(EMPTY + b, THREADS);
                    if (tile.span < spans) {
                        produce(history, k_tile(b), v_tile(b), tile.first, table[tile.span * 4 + 1],
                                kv, piece, column0, thread);
                    } else {
                        stage_fresh(tile, b, piece, column0, thread, PRODUCERS);
                        seismic_cp_async_commit();
                        seismic_cp_async_wait<0>();
                    }
                    __threadfence_block();
                    named_arrive(FULL + b, THREADS);
                    advance(tile, piece);
                }
            }
            return;
        }
    }

    // MMA warps. This lane's two matrix rows (g and g + 8 of the warp's 16).
    const long long token_a = first_token + (warp * 16 + g) / G;
    const long long token_b = first_token + (warp * 16 + g + 8) / G;
    const bool valid_a = warp * 16 + g < QT * G && token_a < rows_total;
    const bool valid_b = warp * 16 + g + 8 < QT * G && token_b < rows_total;

    // Affine: the producers stage every K/V tile, so the MMA warps complete
    // their query tile copy here.
    if constexpr (CODED) {
        seismic_cp_async_wait<0>();
        named_sync(MMA_WARPS, WARPS * 32);
    }

    const float NEG_INF = -__int_as_float(0x7f800000);
    // The output window's accumulators and both rows' softmax states, reset
    // per pass.
    float o[WINDOW / 8][4];
    float maximum[2];
    float denominator[2];
    int cached_span = -1;
    Span interval_a{0, 0}, interval_b{0, 0};
    // A key tile's scores, accumulated over its pieces.
    float s[KEYS / 8][4];

    // The products of one staged unit: S = Q K^T over the key piece's
    // columns; after the tile's last piece, the softmax and O += P V.
    auto absorb_unit = [&](Tile current, int piece, const u8 *k_base, const u8 *v_base) {
        if (piece == 0) {
#pragma unroll
            for (int n = 0; n < KEYS / 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
        }
#pragma unroll
        for (int step = 0; step < PIECE / 16; ++step) {
            u32 a[4];
            {
                const int row = warp * 16 + (lane % 8) + 8 * ((lane / 8) % 2);
                const int chunk = piece * PIECE_CHUNKS + 2 * step + lane / 16;
                seismic_ldmatrix_x4(a, q_tile + swizzled(row, chunk));
            }
#pragma unroll
            for (int n = 0; n < KEYS / 8; n += 2) {
                u32 b4[4];
                const int row = 8 * (n + lane / 16) + (lane % 8);
                const int chunk = 2 * step + (lane / 8) % 2;
                seismic_ldmatrix_x4(b4, k_base + swizzled<PIECE_CHUNKS>(row, chunk));
                const u32 b0[2] = {b4[0], b4[1]};
                const u32 b1[2] = {b4[2], b4[3]};
                Ops::mma(s[n], a, b0);
                Ops::mma(s[n + 1], a, b1);
            }
        }
        if (piece + 1 < PIECES) return;

        if (current.span != cached_span) {
            cached_span = current.span;
            interval_a = valid_a ? span(in, token_a, current.span, spans) : Span{0, 0};
            interval_b = valid_b ? span(in, token_b, current.span, spans) : Span{0, 0};
        }
        const bool full = current.first >= table[current.span * 4 + 2] &&
                          current.first + KEYS <= table[current.span * 4 + 3];

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
        for (int n = 0; n < WINDOW / 8; ++n) {
            o[n][0] *= carry[0];
            o[n][1] *= carry[0];
            o[n][2] *= carry[1];
            o[n][3] *= carry[1];
        }

        // O += P V over the window, P from the S registers.
#pragma unroll
        for (int step = 0; step < KEYS / 16; ++step) {
            const u32 p[4] = {Ops::pair(s[2 * step][0], s[2 * step][1]),
                              Ops::pair(s[2 * step][2], s[2 * step][3]),
                              Ops::pair(s[2 * step + 1][0], s[2 * step + 1][1]),
                              Ops::pair(s[2 * step + 1][2], s[2 * step + 1][3])};
#pragma unroll
            for (int n = 0; n < WINDOW / 8; n += 2) {
                u32 b4[4];
                const int row = 16 * step + (lane % 8) + 8 * ((lane / 8) % 2);
                const int chunk = n + lane / 16;
                seismic_ldmatrix_x4_trans(b4, v_base + swizzled<WINDOW_CHUNKS>(row, chunk));
                const u32 b0[2] = {b4[0], b4[1]};
                const u32 b1[2] = {b4[2], b4[3]};
                Ops::mma(o[n], p, b0);
                Ops::mma(o[n + 1], p, b1);
            }
        }
    };

    for (int column0 = 0; column0 < W; column0 += WINDOW) {
#pragma unroll
        for (int n = 0; n < WINDOW / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
        maximum[0] = maximum[1] = NEG_INF;
        denominator[0] = denominator[1] = 0.0f;

        Tile current = nth(tiles_lo);
        int piece = 0;
        if constexpr (CODED) {
            // Staged unit i (counted over every window) waits for stage i % 2
            // to be full; once its products are issued, the stage is released
            // for unit i + 2 when that unit exists.
            const int staged = WINDOWS * units;
            for (int j = 0; j < units; ++j) {
                const int i = (column0 / WINDOW) * units + j;
                const int b = i % 2;
                named_sync(FULL + b, THREADS);
                absorb_unit(current, piece, k_tile(b), v_tile(b));
                if (i + 2 < staged) named_arrive(EMPTY + b, THREADS);
                advance(current, piece);
            }
        } else {
            auto issue = [&](Tile tile, int tile_piece, int buffer) {
                if (tile.span < spans) {
                    stage_history(history, k_tile(buffer), v_tile(buffer), tile.first,
                                  table[tile.span * 4 + 1], kv, tile_piece, column0);
                } else {
                    stage_fresh(tile, buffer, tile_piece, column0, threadIdx.x, WARPS * 32);
                }
                seismic_cp_async_commit();
            };
            int buffer = 0;
            if (units > 0) issue(current, 0, 0);
            for (int i = 0; i < units; ++i) {
                Tile next = current;
                int next_piece = piece;
                advance(next, next_piece);
                if (i + 1 < units) {
                    issue(next, next_piece, buffer ^ 1);
                    seismic_cp_async_wait<1>();
                } else {
                    seismic_cp_async_wait<0>();
                }
                __syncthreads();
                absorb_unit(current, piece, k_tile(buffer), v_tile(buffer));
                __syncthreads();
                current = next;
                piece = next_piece;
                buffer ^= 1;
            }
        }

        // Both rows' whole denominators (their four lanes' shares).
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            float l = denominator[half];
            l += seismic_shfl_xor_f32(l, 1);
            l += seismic_shfl_xor_f32(l, 2);
            denominator[half] = l;
        }
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const long long token = half == 0 ? token_a : token_b;
            if (!(half == 0 ? valid_a : valid_b)) continue;
            const int query_head = kv * G + (warp * 16 + g + 8 * half) % G;
            if (active > 1) {
                // A partition's partial: the output relative to its row
                // maximum, and (maximum, denominator).
                const u64 slot = (static_cast<u64>(partition) * rows_total + token) * (KV * G) + query_head;
#pragma unroll
                for (int n = 0; n < WINDOW / 8; ++n) {
#pragma unroll
                    for (int e = 0; e < 2; ++e)
                        split.partials[slot * W + column0 + 8 * n + 2 * t + e] = o[n][2 * half + e];
                }
                if (column0 == 0 && t == 0) {
                    split.statistics[slot * 2 + 0] = maximum[half];
                    split.statistics[slot * 2 + 1] = denominator[half];
                }
                continue;
            }
            const float whole = fmaxf(denominator[half], 1e-30f);
            const u64 out_at = static_cast<u64>(token) * SEISMIC_RESULT_0_STRIDE_0 +
                               static_cast<u64>(query_head) * SEISMIC_RESULT_0_STRIDE_1;
#pragma unroll
            for (int n = 0; n < WINDOW / 8; ++n) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int column = column0 + 8 * n + 2 * t + e;
                    element::put<Act>(gated, out_at + column * SEISMIC_RESULT_0_STRIDE_2,
                                      attention::gated(in, token, query_head, column,
                                                       o[n][2 * half + e] / whole));
                }
            }
        }
    }
}

// L3: block (query tile, query head), one thread per column. A tile whose
// keys took several partitions merges each of its rows' partitions in
// partition order and applies the gate; other tiles were stored by L2.
__device__ __forceinline__ void merge_partitions(const Inputs &in, const Split &split, u8 *gated) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const int tile = static_cast<int>(blockIdx.x);
    const int query_head = static_cast<int>(blockIdx.y);
    const int column = static_cast<int>(threadIdx.x);
    const int count = static_cast<int>(split.counts[tile]);
    if (count <= 1) return;
    const long long rows_total = static_cast<long long>(SEISMIC_DIM_M);
    const u64 heads = KV * G;
    for (long long token = static_cast<long long>(tile) * QT;
         token < min(static_cast<long long>(tile + 1) * QT, rows_total); ++token) {
        const u64 slot = static_cast<u64>(token) * heads + query_head;
        float denominator, accumulated;
        merge(split.statistics + slot * 2, static_cast<u64>(rows_total) * heads * 2,
              split.partials + slot * W + column, static_cast<u64>(rows_total) * heads * W, count,
              denominator, accumulated);
        element::put<Act>(gated,
                          static_cast<u64>(token) * SEISMIC_RESULT_0_STRIDE_0 +
                              static_cast<u64>(query_head) * SEISMIC_RESULT_0_STRIDE_1 +
                              static_cast<u64>(column) * SEISMIC_RESULT_0_STRIDE_2,
                          attention::gated(in, token, query_head, column, accumulated / fmaxf(denominator, 1e-30f)));
    }
}

}  // namespace prefill
}  // namespace attention
