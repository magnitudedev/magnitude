// Flash attention on tensor cores over a history policy
// (`attention::DenseHistory`, `attention::AffineHistory`), the body of the
// prefill entries (`attention_prefill`, `attention_prefill_k8v4`, M >= 16)
// and of the decode entries' grouped-query matrix form (`attention_decode`,
// `attention_decode_k8v4` with MATRIX = 1, M <= 8).
//
// A block's matrix rows are QT tokens x G query heads (16 per warp) of one
// kv head. `attend` scans one key partition of its query tile: each span in
// order (the R visible history spans, then the fresh span of batch rows) is
// scanned over the union of the tile rows' intervals in KEYS-key K/V tiles
// held in STAGES operand stages. S = Q K^T and O += P V run as m16n8k16 MMAs
// with F32 accumulation; P stays in registers (FA2). Per-row interval masks
// apply only to K/V tiles outside the rows' common interval, and tiles past
// every row's interval (the causal tail) are never loaded. A head wider than
// 256 makes one such scan per 256-column output window, or (split scores, two
// column warps over a 512-column head) one scan over whole key tiles in which
// each column warp computes its own 256 query columns' scores and the pair adds
// them.
//
// Dense history tiles are copied with `cp.async`, the next tile's copy
// overlapping the current tile's products. Affine history is
// warp-specialized: the entry's producer warps beside the MMA warps load a
// tile's codes and group (scale, zero) pairs one tile ahead (the next tile's
// loads in flight while they decode) and store the decoded values
// code * scale + zero, rounded to the operand element, while the MMA warps run
// the previous tile; named barriers hand the two stages back and forth. Without
// producer warps (ATTENTION_PRODUCER_WARPS = 0) the MMA warps copy the codes
// with `cp.async` as dense tiles are copied and decode each tile once it lands
// (`AffineTiles` in `history.cuh`). The products are then the dense ones over
// the decoded history (the Metal and Vulkan staging).
//
// The rows policy supplies the query tile and fresh tiles and stores the
// partition's result. Prefill rows (`PrefillRows`): L1 `prepare`, one warp
// per (row, query or kv head), puts the queries (norm_rotary, rounded to the
// activation element) and the batch's keys and values as history holds them
// (`hold`: over affine history their encoding, decoded as a history tile is)
// in scratch as the policy's 16-bit MMA operands (`Operands`; scores are
// scaled into the exp2 domain after Q K^T) and appends the row's K/V at its
// destination from the same encoding; L2 `attend` copies them and stores its
// gated output directly when one partition serves the tile, else partials
// that L3 `merge_partitions` combines. Decode rows (`DecodeRows`): the block
// prepares its own query tile and fresh tiles (keys and values as history
// holds them, `hold`), partition 0 appends the tile's K/V, its key tiles sit
// at fixed positions of each row's own key sequence, spans and fresh rows
// alike, each partition taking a contiguous run of a row's tiles sized by
// that row's own count (so a row's result does not
// depend on the rows sharing its tile or launch), and every partition (an
// empty one too) publishes its partial in the decode layout for the decode
// merge (`attention::decode_gate`).

#include "attention.cuh"

namespace attention {
namespace prefill {

template <bool B, typename T, typename F> struct SelectType { using Type = T; };
template <typename T, typename F> struct SelectType<false, T, F> { using Type = F; };

// Warps per 16-row block of the query tile (its row blocks), and warps per
// row block, the entry's ATTENTION_COLUMNS (1 for prefill): a row block's
// column warps compute the same scores from the shared K tile and split the
// output window's columns, so decode rows (one row block) keep the block's
// warps in the products and each warp's accumulators small.
constexpr int WARPS = SEISMIC_TUNE_WARPS;
constexpr int COLUMNS = ATTENTION_COLUMNS;
constexpr int MMA_THREADS = WARPS * COLUMNS * 32;
// K/V operand stages in flight: the entry's ATTENTION_STAGES (2 for prefill;
// decode, with few blocks per SM, tunes deeper pipelines).
constexpr int STAGES = ATTENTION_STAGES;
static_assert(STAGES >= 2 && 1 + 2 * STAGES < 16, "stages fit the named barriers");
// The entry's ATTENTION_Q_REGISTERS: the query fragments stay in registers
// (heads up to 256 columns), so the query tile's shared memory is free for the
// K/V stages and blocks of more warps fit.
constexpr bool Q_REGISTERS = ATTENTION_Q_REGISTERS != 0;
constexpr int ROWS = WARPS * 16;
// Tokens of a block: its matrix rows hold QT whole tokens of G query heads;
// the last ROWS - QT * G rows (none when G divides ROWS) are padding.
constexpr int QT = ROWS / G;
// Keys per K/V tile, and output columns per pass over the keys: a head wider
// than 256 takes 16-key tiles and one pass per output window of 256 columns
// per column warp, each
// recomputing the scores (so every pass's softmax statistics are identical)
// and staging only its window's V columns, which keeps the F32 outputs in
// registers and the stages within shared memory. Its key tiles also stage in
// 256-column pieces, one per operand stage, the scores accumulating over the
// pieces (the query tile is the only whole-head operand in shared memory).
// Split scores (an entry defining ATTENTION_SPLIT_SCORES nonzero, with two or
// four column warps over a head wider than 256): the key tile stages whole
// (one piece) and column warp c computes the scores of its own SPAN query
// columns only (columns [c SPAN, (c + 1) SPAN)); the row block's column warps
// add their partial score tiles through a shared exchange, so Q K^T is
// computed once per tile instead of once per column warp, by every column warp
// at once. Every column warp adds the partials in column order, so their
// softmax states agree exactly.
#ifndef ATTENTION_SPLIT_SCORES
#define ATTENTION_SPLIT_SCORES 0
#endif
constexpr bool SPLIT_SCORES = ATTENTION_SPLIT_SCORES != 0 && COLUMNS >= 2 && W > 256;
constexpr int KEYS = W > 256 ? 16 : 32;
constexpr int WINDOW = W > 256 * COLUMNS ? 256 * COLUMNS : W;
constexpr int WINDOWS = W / WINDOW;
constexpr int PIECE = W > 256 && !SPLIT_SCORES ? 256 : W;
constexpr int PIECES = W / PIECE;
// Stored-key values (VK = 1) over a whole-head stage (one piece, one window):
// keys and values are identical, so each stage holds one [KEYS][W] tile and
// every stager copies it once.
constexpr bool SHARED_KV = STORED_KEY && PIECES == 1 && WINDOWS == 1;
// 8-column output fragments per warp: its column warp's share of the window.
constexpr int OUT = WINDOW / 8 / COLUMNS;
constexpr int CHUNKS = W / 8;  // 16-byte chunks per 16-bit row
constexpr int WINDOW_CHUNKS = WINDOW / 8;
constexpr int PIECE_CHUNKS = PIECE / 8;
static_assert(QT >= 1, "a block holds at least one token's query heads");
static_assert(W % WINDOW == 0, "output windows tile the head");
static_assert(OUT % 2 == 0 && OUT * 8 * COLUMNS == WINDOW, "column warps take fragment pairs");
constexpr int SPAN = SPLIT_SCORES ? W / COLUMNS : PIECE;
static_assert(!SPLIT_SCORES || (PIECE % SPAN == 0 && SPAN % 16 == 0), "a column warp's span lies in one piece");
// The split's exchange: [PARITIES][WARPS row blocks][COLUMNS column warps]
// [32 lanes][KEYS / 8 * 4] F32, after the span table (the entry's shared bytes
// add PARITIES * COLUMNS * WARPS * 1024 for it). With producer warps a column
// warp may publish its next tile's scores while the others still read this
// tile's, so the exchange alternates two parities; the MMA warps' own staging
// (no producers) synchronizes the block after every unit, so one suffices.
constexpr int SPLIT_LANE_FLOATS = KEYS / 8 * 4;
constexpr int PARITIES = ATTENTION_PRODUCER_WARPS > 0 ? 2 : 1;
static_assert(!SPLIT_SCORES || PARITIES * WARPS * COLUMNS * 32 * SPLIT_LANE_FLOATS * 4 == PARITIES * COLUMNS * WARPS * 1024,
              "the split exchange is the declared PARITIES * COLUMNS * WARPS * 1024 bytes");
// Register queries hold one piece: heads up to 256 columns, or with split
// scores each column warp's own span of the head.
static_assert(!Q_REGISTERS || PIECES == 1 || SPLIT_SCORES,
              "register queries hold one piece: heads up to 256 columns, or split scores");

// MMA operand elements (16-bit floats): the element, a packed pair, and the
// m16n8k16 product with F32 accumulation.
struct Bf16Operands {
    __device__ static __forceinline__ u16 operand(float value) { return seismic_f32_to_bf16(value); }
    __device__ static __forceinline__ float value(u16 operand) { return seismic_bf16_to_f32(operand); }
    __device__ static __forceinline__ u32 pair(float lo, float hi) {
        return seismic_pack_bf16x2(lo, hi);
    }
    __device__ static __forceinline__ void mma(float (&acc)[4], const u32 (&a)[4], const u32 (&b)[2]) {
        seismic_mma_m16n8k16_bf16(acc, a, b);
    }
};
struct F16Operands {
    __device__ static __forceinline__ u16 operand(float value) { return seismic_f32_to_f16(value); }
    __device__ static __forceinline__ float value(u16 operand) { return seismic_f16_to_f32(operand); }
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
#if ELEMENT_KIND(KERNEL_A) == ELEMENT_BF16
template <> struct Operands<DenseReads> : Bf16Operands {};
template <> struct Operands<DenseHistory> : Bf16Operands {};
#else
template <> struct Operands<DenseReads> : F16Operands {};
template <> struct Operands<DenseHistory> : F16Operands {};
#endif
#ifdef ATTENTION_AFFINE_HISTORY
template <> struct Operands<AffineReads> : F16Operands {};
template <> struct Operands<AffineHistory> : F16Operands {};
#ifdef ATTENTION_HISTORY_STAGING
template <> struct Operands<ListedF16History> : F16Operands {};
#endif
#endif

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
// by `threads` threads of which this is `thread`, rows [from, rows) only.
// `source(r)` is the element offset of row r's first column, or -1 for a zero
// row. An f32 source (dense
// history of f32 activations) converts to the dense operand element; a 16-bit
// source (activation or operand) is copied with `cp.async`.
template <bool F32_SOURCE, int ROW_CHUNKS, class Source>
__device__ __forceinline__ void stage(u8 *tile, const u8 *base, int rows, Source source,
                                      int thread, int threads, int from = 0) {
    typedef Operands<DenseHistory> Ops;
    for (int index = from * ROW_CHUNKS + thread; index < rows * ROW_CHUNKS; index += threads) {
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
    const u8 *zero_row = F32_SOURCE || rows <= 0 ? nullptr : source(0);
    for (int base_idx = 0; base_idx < rows * ROW_CHUNKS; base_idx += threads) {
        const int index = base_idx + thread;
        const int row = min(index / ROW_CHUNKS, KEYS - 1);
        const u8 *base = source(row);
        if (index >= rows * ROW_CHUNKS) continue;
        const int chunk = index % ROW_CHUNKS;
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
            const u8 *from = base != nullptr ? base + chunk * 16 : zero_row;
            seismic_cp_async_16_zfill(destination, from, base != nullptr ? 16u : 0u);
        }
    }
}

constexpr bool F32_ACTIVATION = Act::bytes == 4;

// The history rows of a key tile: `keys(r)` is the history token of tile row
// r, or -1 for a zero row; the stagers write rows [0, rows) and leave the
// tile's other rows to the fresh rows' stager.
//
// Dense history K piece tile (columns [piece * PIECE, (piece + 1) * PIECE)) of
// rows [0, rows), and with the last piece the V window tile (columns
// [column0, column0 + WINDOW)), copied by the block's WARPS warps.
template <class Keys>
__device__ __forceinline__ void stage_history(const DenseReads &history, u8 *k_tile,
                                              u8 *v_tile, const Keys &keys, int rows, int kv, int piece,
                                              int column0) {
    stage_slab<F32_ACTIVATION, PIECE_CHUNKS>(k_tile, rows, [&](int r) -> const u8 * {
        const int token = keys(r);
        return token >= 0 ? history.key_vector(token, kv) + piece * PIECE * Act::bytes : nullptr;
    }, threadIdx.x, MMA_THREADS);
    if (SHARED_KV || piece + 1 < PIECES) return;
    stage_slab<F32_ACTIVATION, WINDOW_CHUNKS>(v_tile, rows, [&](int r) -> const u8 * {
        const int token = keys(r);
        return token >= 0 ? history.value_vector(token, kv) + column0 * Act::bytes : nullptr;
    }, threadIdx.x, MMA_THREADS);
}

// ---------------------------------------------------------------------------
#ifdef ATTENTION_AFFINE_HISTORY
// Affine history tiles, decoded by the producer warps or, without them, by
// the MMA warps from code stages (`AffineTiles`, `history.cuh`).

// Producer threads (the entry's ATTENTION_PRODUCER_WARPS warps; 0 for dense
// history, which has none). Fewer producer warps than MMA warps leave the MMA
// warps more registers (a block's registers split evenly over its threads).
constexpr int PRODUCERS = ATTENTION_PRODUCER_WARPS * 32;

// The operand tiles affine history decodes into: KEYS-row K piece and V
// window tiles in the swizzled layout.
struct OperandTiles {
    static constexpr int KEYS = prefill::KEYS;
    static constexpr int PIECE = prefill::PIECE;
    static constexpr int PIECES = prefill::PIECES;
    static constexpr int PIECE_CHUNKS = prefill::PIECE_CHUNKS;
    static constexpr int WINDOW = prefill::WINDOW;
    static constexpr int WINDOW_CHUNKS = prefill::WINDOW_CHUNKS;
    static constexpr int PRODUCERS = prefill::PRODUCERS;
    template <int ROW_CHUNKS>
    __device__ static __forceinline__ u32 swizzled(int row, int chunk) {
        return prefill::swizzled<ROW_CHUNKS>(row, chunk);
    }
};
typedef AffineTiles<OperandTiles> Affine;

#endif

// A batch row's prepared key `k` and value `v` (one lane's share; the whole
// warp calls it) made the operands every read of the row takes, the values
// history holds: dense history holds them as they are, affine history their
// encoding (`AffineHistory::quantize`), decoded as a history tile is. Stored
// at `destination` when it is >= 0, from the same encoding.
template <class History>
__device__ __forceinline__ void hold(const History &history, int destination, int kv,
                                     float (&k)[DPL], float (&v)[DPL], int lane) {
#ifdef ATTENTION_AFFINE_HISTORY
    if constexpr (History::CODED) {
        const AffineHistory::Encoding e = AffineHistory::quantize(k, v);
        if (destination >= 0) history.store(destination, kv, e, lane);
        decoded(e.key, k);
        decoded(e.value, v);
    } else
#endif
    {
        if (destination >= 0) history.append(destination, kv, k, v, lane);
    }
}

// Named barriers between the MMA and the producer warps (0 is
// `__syncthreads`): stage b is full (produced) at FULL + b and empty
// (consumed) at EMPTY + b; the MMA warps' own barrier is MMA.
constexpr u32 FULL = 1;
constexpr u32 EMPTY = FULL + STAGES;
constexpr u32 MMA_WARPS = EMPTY + STAGES;
// Split scores: row block r's column warps meet at PAIR + r.
constexpr u32 PAIR = MMA_WARPS + 1;
static_assert(!SPLIT_SCORES || PAIR + WARPS <= 16, "the pairs' barriers fit the named barriers");
__device__ __forceinline__ void named_sync(u32 id, u32 threads) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}
__device__ __forceinline__ void named_arrive(u32 id, u32 threads) {
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

struct Tile {
    int span;
    int first;
    int index = 0;  // its place in a WorkList's list
};

// A block of `attend`: query tile `tile` (tokens [tile * QT, (tile + 1) *
// QT)), kv head `kv`, and key partition `partition` of `parts`. Prefill
// rows: the query tile's key tiles (each span's union interval in KEYS steps,
// spans then fresh) split into runs of at least MIN_TILES over the
// partitions; partitions past the last run are empty. Decode rows: a run of
// each row's position tiles (`attend`).
struct Block {
    int tile;
    int kv;
    int partition;
    int parts;
};

// This lane's DPL columns of a W-vector, as operands, into row `row` of a
// swizzled 16-bit tile of ROW_CHUNKS chunks holding columns [first, first +
// ROW_CHUNKS * 8).
template <class Ops, int ROW_CHUNKS>
__device__ __forceinline__ void put_row(u8 *tile, int row, int first, const float (&x)[DPL],
                                        int lane) {
#pragma unroll
    for (int d = 0; d < DPL; ++d) {
        const int column = lane * DPL + d - first;
        if (column >= 0 && column < ROW_CHUNKS * 8)
            *reinterpret_cast<u16 *>(tile + swizzled<ROW_CHUNKS>(row, column / 8) + (column % 8) * 2) =
                Ops::operand(x[d]);
    }
}

// Prefill split scratch: a tile served by one partition stores its gated
// output directly; otherwise each partition stores (partial output, maximum,
// denominator) per (row, query head) and `merge_partitions` combines them.
// `counts` holds each query tile's partition count. Entries without key
// partitions (a grid z extent of 1) pass null scratch.
//
struct Split {
    float *partials;    // [parts][M][KV * G][W]
    float *statistics;  // [parts][M][KV * G][2]
    u32 *counts;        // [query tiles]
};

// An index-list source (an entry defining ATTENTION_PICKS, its KS picks a
// row, nonzero): row `token`'s picks (`rows`, [M][KS] ordinals of the
// selected plane, -1 for none) each name a selected vector (`selected`, a
// slab plane [TS][KV][W] of the activation element), read as key and value.
// `attend` scans a query tile's picks as one more span after the fresh span,
// its tokens' lists in token order (key ordinal j * KS + pick is token j's
// pick), so a staged tile of picked rows serves every query head of its
// token, as history tiles serve the tile's.
#ifndef ATTENTION_PICKS
#define ATTENTION_PICKS 0
#endif
constexpr int PICK_COUNT = ATTENTION_PICKS;
constexpr bool PICKS = PICK_COUNT > 0;
struct Picked {
    const int *rows;
    u64 row_stride, pick_stride;
    const u32 *selected;
    u64 slab_rows, vector_stride, head_stride;

    __device__ __forceinline__ int at(long long token, int pick) const {
        return rows[static_cast<u64>(token) * row_stride + static_cast<u64>(pick) * pick_stride];
    }
    __device__ __forceinline__ const u8 *vector(int row, int kv) const {
        return slab::row(selected, row, slab_rows, vector_stride * Act::bytes) +
               static_cast<u64>(kv) * head_stride * Act::bytes;
    }
};

// A prefill launch's WorkList selection (`unit_reads`' per-row block
// bitmap, `words` words a row, `block_rows` keys a block; none when `words`
// is 0), with what its list launch (`list`) leaves: each row's key ordinal at
// the start of each of its spans (`bases`, [M][R + 1]) and each query tile's
// listed key tiles (`lists`, [tiles][capacity] (span, first) pairs; `listed`
// counts them).
struct Selection {
    const u32 *bitmap;
    const int *bases;
    const int *lists;
    const u32 *listed;
    u32 words;
    int block_rows;
    u32 capacity;

    // Whether row `token` reads key `key` of span `index` (whose bounds
    // begin at `lo`): the bit of the key's block (a block past the bitmap is
    // not read).
    __device__ __forceinline__ bool reads(long long token, int index, int spans, int key, int lo) const {
        const int ordinal = bases[token * (spans + 1) + index] + key - lo;
        const u32 block = static_cast<u32>(ordinal) / static_cast<u32>(block_rows);
        return block / 32 < words && ((bitmap[token * words + block / 32] >> (block % 32)) & 1u) != 0u;
    }
};

// Prefill rows: L1 put the queries and the batch's keys and values in scratch
// as operands ([M, KV * G, W], [M, KV, W] and [M, KV, W]); `attend` copies
// them. Rows past the decode rows take each row's tiles from the union of its
// tile's spans, in runs of at least MIN_TILES a partition; the decode row
// classes (DECODE) take their keys on the position grid (POSITIONS, `attend`)
// and publish a partial per partition that holds any, the merge reading
// `counts` of them.
template <bool DECODE>
struct PrefillRows {
    static constexpr bool POSITIONS = DECODE;
    static constexpr int MIN_TILES = 16;
    const u8 *queries;
    const u8 *keys;
    const u8 *values;
    element::Out<Act> gated;
    Split split;

    // The query tile (MMA warps): matrix row i is token first_token + i / G,
    // head kv * G + i % G; padding rows are zero.
    template <class History>
    __device__ __forceinline__ void stage_queries(const Inputs &in, const History &, u8 *q_tile,
                                                  float *, Block block, long long first_token,
                                                  int, int) const {
        [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
        const long long rows_total = static_cast<long long>(SEISMIC_DIM_M);
        stage<false, CHUNKS>(q_tile, queries, ROWS, [&](int i) -> long long {
            const long long token = first_token + i / G;
            if (i >= QT * G || token >= rows_total) return -1;
            return (token * KV * G + block.kv * G + i % G) * W;
        }, threadIdx.x, MMA_THREADS);
        seismic_cp_async_commit();
    }

    __device__ __forceinline__ void partitioned(Block block, int active) const {
        if (block.parts > 1 && block.partition == 0 && block.kv == 0 && threadIdx.x == 0)
            split.counts[block.tile] = active;
    }

    __device__ __forceinline__ void vacant(const Inputs &, Block, long long) const {}

    // The fresh tile's rows [r0, KEYS), row r the token first + r (zero at or
    // past `limit`): the keys' piece, and with the last piece the values'
    // window from column0, by `threads` threads of which this is `thread`.
    template <class History>
    __device__ __forceinline__ void stage_fresh(const Inputs &, const History &, u8 *k_tile,
                                                u8 *v_tile, int first, int limit, int r0, int kv,
                                                int piece, int column0, int thread, int threads,
                                                float *) const {
        auto row = [&](int r) -> long long {
            const int token = first + r;
            return token < limit ? (static_cast<long long>(token) * KV + kv) * W : -1;
        };
        stage<false, PIECE_CHUNKS>(k_tile, keys, KEYS, [&](int r) -> long long {
            const long long at = row(r);
            return at >= 0 ? at + piece * PIECE : -1;
        }, thread, threads, r0);
        if (SHARED_KV || piece + 1 < PIECES) return;
        stage<false, WINDOW_CHUNKS>(v_tile, values, KEYS, [&](int r) -> long long {
            const long long at = row(r);
            return at >= 0 ? at + column0 : -1;
        }, thread, threads, r0);
    }

    // One matrix row's output window (this lane's columns of `o`, row half
    // `half`), relative to its maximum, with the row's whole denominator.
    template <int N>
    __device__ __forceinline__ void store(const Inputs &in, Block block, int active,
                                          long long token, int query_head, int column0,
                                          const float (&o)[N][4], int half, int t, float maximum,
                                          float denominator) const {
        [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
        const long long rows_total = static_cast<long long>(SEISMIC_DIM_M);
        if (active > 1) {
            // A partition's partial: the output relative to its row maximum,
            // and (maximum, denominator).
            const u64 slot =
                (static_cast<u64>(block.partition) * rows_total + token) * (KV * G) + query_head;
#pragma unroll
            for (int n = 0; n < N; ++n) {
#pragma unroll
                for (int e = 0; e < 2; ++e)
                    split.partials[slot * W + column0 + 8 * n + 2 * t + e] = o[n][2 * half + e];
            }
            if (column0 == 0 && t == 0) {
                split.statistics[slot * 2 + 0] = maximum;
                split.statistics[slot * 2 + 1] = denominator;
            }
            return;
        }
        const float whole = normalizer(in, query_head, maximum, denominator);
        const u64 out_at = static_cast<u64>(token) * SEISMIC_RESULT_0_STRIDE_0 +
                           static_cast<u64>(query_head) * SEISMIC_RESULT_0_STRIDE_1;
#pragma unroll
        for (int n = 0; n < N; ++n) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int column = column0 + 8 * n + 2 * t + e;
                gated.put(out_at + column * SEISMIC_RESULT_0_STRIDE_2,
                          attention::gated(in, token, query_head, column, o[n][2 * half + e] / whole));
            }
        }
    }
};

// Decode rows: the block prepares its query tile and the fresh tiles itself
// (a warp per row, through a [WARPS][W] F32 exchange; fresh keys and values
// as history holds them, `hold`), takes its keys on the position grid
// (POSITIONS, `attend`), and publishes every partition in the decode layout
// (`attention::publish`): slot = (row * KV * G + query head) * PARTS +
// partition, an empty partition with denominator 0.
template <int PARTS>
struct DecodeRows {
    static constexpr bool POSITIONS = true;
    static constexpr int MIN_TILES = 1;
    float *partials;
    float *statistics;

    // The query tile (MMA warp `warp`, rows warp, warp + WARPS, ...), and in
    // partition 0 the tile's rows appended to history.
    template <class History>
    __device__ __forceinline__ void stage_queries(const Inputs &in, const History &history,
                                                  u8 *q_tile, float *exchange, Block block,
                                                  long long first_token, int warp,
                                                  int lane) const {
        [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
        typedef Operands<History> Ops;
        const long long rows_total = static_cast<long long>(SEISMIC_DIM_M);
        float *own = exchange + warp * W;
        for (int i = warp; i < ROWS; i += WARPS * COLUMNS) {
            const long long token = first_token + i / G;
            float x[DPL];
            if (i < QT * G && token < rows_total) {
                prepared_query(in, token, block.kv * G + i % G, x, own, lane);
            } else {
#pragma unroll
                for (int d = 0; d < DPL; ++d) x[d] = 0.0f;
            }
            put_row<Ops, CHUNKS>(q_tile, i, 0, x, lane);
        }
        if (FRESH && block.partition == 0) {
            for (long long token = first_token + warp;
                 token < min(first_token + QT, rows_total); token += WARPS * COLUMNS) {
                if (ATTENTION_DESTINATION(in, token) < 0) continue;
                float k[DPL];
                prepared_key(in, token, block.kv, k, own, lane);
                append(in, history, token, block.kv, k, lane);
            }
        }
    }

    __device__ __forceinline__ void partitioned(Block, int) const {}

    __device__ __forceinline__ void vacant(const Inputs &in, Block block, long long first_token) const {
        [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
        const long long rows_total = static_cast<long long>(SEISMIC_DIM_M);
        for (int i = threadIdx.x; i < QT * G; i += blockDim.x) {
            const long long token = first_token + i / G;
            if (token >= rows_total) continue;
            const u64 slot =
                (static_cast<u64>(token) * KV * G + block.kv * G + i % G) * PARTS + block.partition;
            statistics[slot * 2 + 0] = -__int_as_float(0x7f800000);
            statistics[slot * 2 + 1] = 0.0f;
        }
    }

    // Rows [r0, KEYS) of a key tile whose row r is fresh row `first` + r
    // (zero at or past `limit`), prepared by the warps of `threads` threads
    // of which this is `thread`, a warp per row.
    template <class History>
    __device__ __forceinline__ void stage_fresh(const Inputs &in, const History &history,
                                                u8 *k_tile, u8 *v_tile, int first, int limit, int r0,
                                                int kv, int piece, int column0, int thread,
                                                int threads, float *exchange) const {
        typedef Operands<History> Ops;
        const int lane = thread % 32;
        float *own = exchange + (thread / 32) * W;
        for (int r = r0 + thread / 32; r < KEYS; r += threads / 32) {
            const int token = first + r;
            float k[DPL];
            float v[DPL];
            if (token < limit) {
                prepared_key(in, token, kv, k, own, lane);
                fresh_value(in, token, kv, v, lane);
                hold(history, -1, kv, k, v, lane);
            } else {
#pragma unroll
                for (int d = 0; d < DPL; ++d) k[d] = v[d] = 0.0f;
            }
            put_row<Ops, PIECE_CHUNKS>(k_tile, r, piece * PIECE, k, lane);
            if (!SHARED_KV && piece + 1 == PIECES) put_row<Ops, WINDOW_CHUNKS>(v_tile, r, column0, v, lane);
        }
    }

    template <int N>
    __device__ __forceinline__ void store(const Inputs &, Block block, int, long long token,
                                          int query_head, int column0, const float (&o)[N][4],
                                          int half, int t, float maximum,
                                          float denominator) const {
        const u64 slot =
            (static_cast<u64>(token) * KV * G + query_head) * PARTS + block.partition;
#pragma unroll
        for (int n = 0; n < N; ++n) {
#pragma unroll
            for (int e = 0; e < 2; ++e)
                partials[slot * W + column0 + 8 * n + 2 * t + e] = o[n][2 * half + e];
        }
        if (column0 == 0 && t == 0) {
            statistics[slot * 2 + 0] = maximum;
            statistics[slot * 2 + 1] = denominator;
        }
    }
};

// Store a lane's DPL operand elements to `dst` using the widest aligned vector
// store (`uint4`, `uint2`, `u32`) when DPL is even, with scalar fallback.
template <class Ops>
__device__ __forceinline__ void store_operands(u16 *dst, const float (&x)[DPL]) {
    if constexpr (DPL % 2 == 0) {
        const u64 addr = reinterpret_cast<u64>(dst);
        if constexpr (DPL % 8 == 0) {
            if ((addr & 15u) == 0) {
                uint4 *out = reinterpret_cast<uint4 *>(dst);
#pragma unroll
                for (int d = 0; d < DPL; d += 8) {
                    out[d / 8] = make_uint4(Ops::pair(x[d + 0], x[d + 1]),
                                            Ops::pair(x[d + 2], x[d + 3]),
                                            Ops::pair(x[d + 4], x[d + 5]),
                                            Ops::pair(x[d + 6], x[d + 7]));
                }
                return;
            }
        }
        if constexpr (DPL % 4 == 0) {
            if ((addr & 7u) == 0) {
                uint2 *out = reinterpret_cast<uint2 *>(dst);
#pragma unroll
                for (int d = 0; d < DPL; d += 4) {
                    out[d / 4] = make_uint2(Ops::pair(x[d + 0], x[d + 1]),
                                            Ops::pair(x[d + 2], x[d + 3]));
                }
                return;
            }
        }
        if ((addr & 3u) == 0) {
            u32 *out = reinterpret_cast<u32 *>(dst);
#pragma unroll
            for (int d = 0; d < DPL; d += 2) {
                out[d / 2] = Ops::pair(x[d + 0], x[d + 1]);
            }
            return;
        }
    }
#pragma unroll
    for (int d = 0; d < DPL; ++d) dst[d] = Ops::operand(x[d]);
}

// Prefill L1 over `M` rows: 8 warps per block (launch with 256 threads). Queries,
// and keys and values as history holds them (`hold`), go to scratch as the
// history policy's operands ([M, KV * G, W], [M, KV, W] and [M, KV, W]); a row
// with a destination is appended there.
template <class History>
__device__ __forceinline__ void prepare(const Inputs &in, const History &history, u16 *queries,
                                        u16 *keys, u16 *values, rounding::Rounding trained) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    typedef Operands<History> Ops;
    __shared__ float exchange[head_layout::NEEDS_EXCHANGE ? 8 * W : 1];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    float *own = head_layout::NEEDS_EXCHANGE ? exchange + warp * W : exchange;
    const u64 item = static_cast<u64>(blockIdx.x) * 8 + warp;
    constexpr u64 heads = static_cast<u64>(KV) * (G + 1);
    if (item >= SEISMIC_DIM_M * heads) return;
    u64 row;
    int head;
    if (item <= 0xffffffffull) {
        const u32 item32 = static_cast<u32>(item);
        constexpr u32 heads32 = static_cast<u32>(KV * (G + 1));
        row = item32 / heads32;
        head = static_cast<int>(item32 % heads32);
    } else {
        row = item / heads;
        head = static_cast<int>(item % heads);
    }
    if (head < KV * G) {
        float x[DPL];
        prepared_query(in, row, head, x, own, lane);
        store_operands<Ops>(queries + (row * KV * G + head) * W + lane * DPL, x);
        return;
    }
    if (!FRESH) return;
    const int kv = head - KV * G;
    float k[DPL];
    float v[DPL];
    prepared_key(in, row, kv, k, own, lane, trained);
    // A stored-key value is the prepared key.
    if constexpr (STORED_KEY) {
#pragma unroll
        for (int d = 0; d < DPL; ++d) v[d] = k[d];
    } else {
        fresh_value(in, row, kv, v, lane);
    }
    hold(history, ATTENTION_DESTINATION(in, row), kv, k, v, lane);
    const u64 at = (row * KV + kv) * W + lane * DPL;
    store_operands<Ops>(keys + at, k);
    store_operands<Ops>(values + at, v);
}

// Block `block`: WARPS MMA warps, plus WARPS producer warps for affine
// history (launch with WARPS * COLUMNS * 32 threads for dense history, twice that for
// affine). Shared memory: the query tile [ROWS][W], STAGES operand stages each K
// [KEYS][PIECE] then V [KEYS][WINDOW] 16-bit, the span table [R + 1][4] (or for
// decode position tiles `PositionPlan`), then for decode rows the exchange
// [WARPS * COLUMNS][W] F32. The MMA warps make one pass
// over the partition's key tiles per output window; a key tile is PIECES
// staged units, the last carrying the window's values.
#ifdef ATTENTION_HISTORY_STAGING
// Each row is looked up separately. An operand tile may cross compact-page
// or visibility gaps without changing its original span-based KEY grouping.
template <class Keys>
__device__ __forceinline__ void stage_history(const ListedF16History &history, u8 *k_tile,
                                               u8 *v_tile, const Keys &keys, int rows, int kv, int piece,
                                               int column0) {
    stage_slab<false, PIECE_CHUNKS>(k_tile, rows, [&](int r) -> const u8 * {
        const int token = keys(r);
        const u8 *row = token >= 0 ? history.key_vector(token, kv) : nullptr;
        return row ? row + piece * PIECE * 2 : nullptr;
    }, threadIdx.x, MMA_THREADS);
    if (piece + 1 < PIECES) return;
    stage_slab<false, WINDOW_CHUNKS>(v_tile, rows, [&](int r) -> const u8 * {
        const int token = keys(r);
        const u8 *row = token >= 0 ? history.value_vector(token, kv) : nullptr;
        return row ? row + column0 * 2 : nullptr;
    }, threadIdx.x, MMA_THREADS);
}
#endif

// The list launch of a WorkList prefill: block per query tile of QT rows,
// 256 threads. It writes each tile row's key ordinal at the start of each of
// its spans (`bases`), then lists, in `attend`'s tile order (each span's
// union interval in KEYS-key tiles, spans then fresh), the key tiles some row
// of the tile reads (a row reads a key of its span interval when the bit of
// the key's block is set). `attend` walks exactly these.
__device__ __forceinline__ void list(const Inputs &in, int *bases, int *lists, u32 *listed, const u32 *bitmap,
                                     u32 words, int block_rows, u32 capacity, int tile) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const int spans = static_cast<int>(SEISMIC_DIM_R);
    const long long first_token = static_cast<long long>(tile) * QT;
    const long long rows_total = static_cast<long long>(SEISMIC_DIM_M);
    extern __shared__ __align__(16) u8 shared[];
    int *table = reinterpret_cast<int *>(shared);  // [R + 1][2] union intervals
    u32 *sums = reinterpret_cast<u32 *>(table + (spans + 1) * 2);  // [8], then the count
    const int thread = threadIdx.x;
    const int warp = thread / 32;
    const int lane = thread % 32;
    if (thread < QT && first_token + thread < rows_total) {
        int base = 0;
        for (int index = 0; index <= spans; ++index) {
            const Span s = span(in, first_token + thread, index, spans);
            bases[(first_token + thread) * (spans + 1) + index] = base;
            base += max(s.hi - s.lo, 0);
        }
    }
    for (int index = thread; index <= spans; index += 256) {
        int union_lo = 0x7fffffff, union_hi = -0x7fffffff;
        for (long long token = first_token; token < min(first_token + QT, rows_total); ++token) {
            const Span s = span(in, token, index, spans);
            if (s.hi > s.lo) {
                union_lo = min(union_lo, s.lo);
                union_hi = max(union_hi, s.hi);
            }
        }
        if (union_hi <= union_lo) union_lo = union_hi = 0;
        table[index * 2] = union_lo;
        table[index * 2 + 1] = union_hi;
    }
    if (thread == 0) sums[8] = 0;
    __syncthreads();
    int total = 0;
    for (int index = 0; index <= spans; ++index)
        if (table[index * 2 + 1] > table[index * 2]) total += (table[index * 2 + 1] - table[index * 2] + KEYS - 1) / KEYS;
    for (int round = 0; round < total; round += 256) {
        const int candidate = round + thread;
        bool read = false;
        int entry_span = 0, entry_first = 0;
        if (candidate < total) {
            int before = 0;
            for (int index = 0; index <= spans; ++index) {
                const int lo = table[index * 2], hi = table[index * 2 + 1];
                if (hi <= lo) continue;
                const int n = (hi - lo + KEYS - 1) / KEYS;
                if (candidate < before + n) {
                    entry_span = index;
                    entry_first = lo + (candidate - before) * KEYS;
                    break;
                }
                before += n;
            }
            const int end = min(entry_first + KEYS, table[entry_span * 2 + 1]);
            for (long long token = first_token; token < min(first_token + QT, rows_total) && !read; ++token) {
                const Span s = span(in, token, entry_span, spans);
                const int first_key = max(entry_first, s.lo);
                const int last_key = min(end, s.hi);
                if (first_key >= last_key) continue;
                const int base = bases[token * (spans + 1) + entry_span];
                const u32 first_block = static_cast<u32>(base + first_key - s.lo) / static_cast<u32>(block_rows);
                const u32 last_block = static_cast<u32>(base + last_key - 1 - s.lo) / static_cast<u32>(block_rows);
                for (u32 b = first_block; b <= last_block && !read; ++b)
                    read = b / 32 < words && ((bitmap[token * words + b / 32] >> (b % 32)) & 1u) != 0u;
            }
        }
        // Compact this round's listed tiles in tile order.
        const u32 votes = __ballot_sync(0xffffffffu, read);
        const u32 before_lane = __popc(votes & ((1u << lane) - 1u));
        __syncthreads();
        if (lane == 0) sums[warp] = __popc(votes);
        __syncthreads();
        u32 offset = sums[8];
        for (int other = 0; other < warp; ++other) offset += sums[other];
        if (read && offset + before_lane < capacity) {
            int *slot = lists + (static_cast<long long>(tile) * capacity + offset + before_lane) * 2;
            slot[0] = entry_span;
            slot[1] = entry_first;
        }
        __syncthreads();
        if (thread == 0) {
            u32 added = 0;
            for (int other = 0; other < 8; ++other) added += sums[other];
            sums[8] += added;
        }
        __syncthreads();
    }
    if (thread == 0) listed[tile] = min(sums[8], capacity);
}

struct SpanPlacement {
    int shift;
    int end;
};

// Decode key ordinal `first` of row `row` placed on the span holding it
// (`visible_locate`; the whole warp calls it).
__device__ __forceinline__ SpanPlacement position_locate(const Inputs &in, int first, long long row, int spans,
                                                         int lane) {
    const reduce::Located at = visible_locate(in, static_cast<u64>(row), static_cast<u64>(spans), first, lane);
    const Span s = span(in, static_cast<u64>(row), at.span, static_cast<u64>(spans));
    return SpanPlacement{s.lo - static_cast<int>(at.offset),
                         static_cast<int>(at.offset) + max(s.hi - s.lo, 0)};
}

// Decode row `row`'s figures and the span placement of key ordinal
// `at(total)`: in `total` its key count, in `historical` its history keys and
// in `fresh` the fresh row of its ordinal 0. Where its spans fit one warp
// (R < 32), one load of each span gives them all; else its count
// (`visible_total`), its fresh span, then `position_locate`. An ordinal at or
// past `total` is left unplaced (the whole warp calls it).
template <class At>
__device__ __forceinline__ SpanPlacement position_place(const Inputs &in, long long row, int spans, const At &at,
                                                        int lane, int &total, int &historical, int &fresh) {
    if (spans < 32) {
        const Span own = lane <= spans ? span(in, static_cast<u64>(row), static_cast<u64>(lane), static_cast<u64>(spans))
                                       : Span{0, 0};
        const int length = max(own.hi - own.lo, 0);
        const int inclusive = reduce::lanes_inclusive_add(length, lane);
        total = __shfl_sync(0xffffffffu, inclusive, 31);
        historical = total - __shfl_sync(0xffffffffu, length, spans);
        fresh = __shfl_sync(0xffffffffu, own.lo, spans) - historical;
        const int tile_first = at(total);
        const unsigned hits = __ballot_sync(0xffffffffu, lane <= spans && inclusive > tile_first);
        if (hits != 0u) {
            const int first = __ffs(hits) - 1;
            return SpanPlacement{__shfl_sync(0xffffffffu, own.lo - (inclusive - length), first),
                                 __shfl_sync(0xffffffffu, inclusive, first)};
        }
        return SpanPlacement{0, 0};
    }
    total = static_cast<int>(visible_total(in, static_cast<u64>(row), static_cast<u64>(spans), lane));
    const Span f = span(in, static_cast<u64>(row), static_cast<u64>(spans), static_cast<u64>(spans));
    historical = total - max(f.hi - f.lo, 0);
    fresh = f.lo - historical;
    const int tile_first = at(total);
    return tile_first < total ? position_locate(in, tile_first, row, spans, lane) : SpanPlacement{0, 0};
}

template <class History, class Rows>
__device__ __forceinline__ void attend(const Inputs &in, const History &history, const Rows &rows,
                                       Block block, const Selection &selection = Selection{},
                                       const Picked &picked = Picked{}) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    typedef Operands<History> Ops;
    constexpr bool CODED = History::CODED;
    // Affine history is decoded by producer warps, or without them by the
    // MMA warps from code stages (`AffineTiles::stage_codes`, `decode_codes`).
#ifdef ATTENTION_AFFINE_HISTORY
    constexpr bool PRODUCED = CODED && PRODUCERS > 0;
    constexpr bool INLINE = CODED && PRODUCERS == 0;
    constexpr int THREADS = MMA_THREADS + (CODED ? PRODUCERS : 0);
#else
    constexpr bool PRODUCED = false;
    constexpr bool INLINE = false;
    constexpr int THREADS = MMA_THREADS;
#endif
    static_assert(!INLINE || (W >= 128 && (W <= 256 || W == 512) && MMA_THREADS % KEYS == 0),
                  "inline decoding takes whole 16-byte pair rows and whole rows per thread group");
    static_assert(!(PICKS && CODED), "picked rows are activation vectors");
    const int kv = block.kv;
    const long long first_token = static_cast<long long>(block.tile) * QT;
    const long long rows_total = static_cast<long long>(SEISMIC_DIM_M);
    const int spans = static_cast<int>(SEISMIC_DIM_R);
    // The key sequence's last span: the fresh span (index `spans`), or with
    // an index-list source the tile's picks (index `spans` + 1), whose keys
    // are pick ordinals.
    const int last = spans + (PICKS ? 1 : 0);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    // Picks span key `ordinal`'s selected row: its token's pick, -1 for none.
    auto pick_row = [&](int ordinal) {
        constexpr int LIST = PICKS ? PICK_COUNT : 1;
        const long long token = first_token + ordinal / LIST;
        return token < rows_total ? picked.at(token, ordinal % LIST) : -1;
    };

    extern __shared__ __align__(16) u8 shared[];
    // With Q_REGISTERS the query tile is staged in the K/V stages' region and
    // read into registers before the first K/V tile arrives, so the stages
    // and the query tile share the region.
    constexpr int Q_BYTES = ROWS * W * 2;
    constexpr int STAGE_BYTES = KEYS * (PIECE + (SHARED_KV ? 0 : WINDOW)) * 2;
    // Inline decoding adds STAGES code stages after the operand stages.
    // Split scores decode inline from a single code stage: a unit's codes
    // are copied once the previous unit's are decoded, while its products run.
    constexpr int CODE_STAGES = SPLIT_SCORES ? 1 : STAGES;
#ifdef ATTENTION_AFFINE_HISTORY
    constexpr int STAGED_BYTES = STAGES * STAGE_BYTES + (INLINE ? CODE_STAGES * Affine::CODE_STAGE_BYTES : 0);
#else
    constexpr int STAGED_BYTES = STAGES * STAGE_BYTES;
#endif
    constexpr int REGION = Q_REGISTERS ? (Q_BYTES > STAGED_BYTES ? Q_BYTES : STAGED_BYTES)
                                       : Q_BYTES + STAGED_BYTES;
    u8 *q_tile = shared;
    u8 *kv_tiles = shared + (Q_REGISTERS ? 0 : Q_BYTES);
    auto k_tile = [&](int b) { return kv_tiles + b * STAGE_BYTES; };
    auto v_tile = [&](int b) { return SHARED_KV ? k_tile(b) : kv_tiles + b * STAGE_BYTES + KEYS * PIECE * 2; };
#ifdef ATTENTION_AFFINE_HISTORY
    auto code_tile = [&](int b) { return kv_tiles + STAGES * STAGE_BYTES + (b % CODE_STAGES) * Affine::CODE_STAGE_BYTES; };
#endif
    constexpr int POSITION_ROWS = static_cast<int>(SEISMIC_CONST_VERIFY_ROWS);
    static_assert(POSITION_ROWS <= 32, "a verify launch's rows fit one warp's lanes");
    constexpr int MAX_POSITION_WALKS = POSITION_ROWS * (PICKS ? 2 : 1);
    struct PositionWalk {
        int row;
        int first;
        int end_key;
        int historical;
        int fresh;
        int shift0;
        int span0_end;
    };
    struct PositionPlan {
        const int *visible;
        u64 vis_s0;
        u64 vis_s1;
        int first_token;
        int spans;
        int active;
        int units;
        int walks;
        int prefixed;
        int full_start;
        int full_end;
        int row_end[POSITION_ROWS];
        union {
            PositionWalk walk[MAX_POSITION_WALKS];
            struct {
                PositionWalk walk[1 + (PICKS ? POSITION_ROWS : 0)];
                int row_start[POSITION_ROWS];
                int tail_row;
            } single;
        };
    };
    constexpr int PLAN_INTS = PICKS ? 144 : 80;
    static_assert(sizeof(PositionPlan) <= PLAN_INTS * 4, "PositionPlan fits the declared 16-byte slots");

    int *table = reinterpret_cast<int *>(shared + REGION);  // [last + 1][4], or PositionPlan for POSITIONS
    PositionPlan *plan = reinterpret_cast<PositionPlan *>(table);
    float *exchange = reinterpret_cast<float *>(
        Rows::POSITIONS ? table + PLAN_INTS : table + (last + 1) * 4);  // [WARPS * COLUMNS][W]
    // Decode rows with producer warps concurrently use `exchange` in
    // `stage_fresh_at`, so the split-scores exchange sits after it.
    [[maybe_unused]] float *split_exchange =
        (Rows::POSITIONS && PRODUCED) ? exchange + WARPS * COLUMNS * W : exchange;

    // Span table: union and common interval of the tile's valid tokens. Decode
    // rows (POSITIONS) place their tiles on each row's own grid instead.
    for (int index = threadIdx.x; !Rows::POSITIONS && index <= spans; index += THREADS) {
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
    // The picks span: ordinals up to the tile's last named pick (an indexer
    // pads a short list with -1), each key masked by its own pick, so it has
    // no common interval.
    if constexpr (PICKS && !Rows::POSITIONS) {
        if (threadIdx.x == 0) {
            table[last * 4 + 0] = table[last * 4 + 1] = 0;
            table[last * 4 + 2] = table[last * 4 + 3] = 0;
        }
        __syncthreads();
        int named = 0;
        for (int ordinal = threadIdx.x; ordinal < QT * PICK_COUNT; ordinal += THREADS)
            if (pick_row(ordinal) >= 0) named = ordinal + 1;
        if (named > 0) atomicMax(&table[last * 4 + 1], named);
    }

    // A WorkList's key tiles are its query tile's listed ones.
    const bool listing = selection.words > 0;
    const int parts = block.parts;
    const int partition = block.partition;
    constexpr int PICK_TILES = PICKS ? (PICK_COUNT + KEYS - 1) / KEYS : 0;
    // Decode rows (POSITIONS) take their keys on a grid of positions in each
    // row's own key sequence (its visible spans, then its fresh keys), the
    // decode key rule (`attention.seismic`): warp 0 places the partition's
    // walks once in shared `plan` so neither the producer warps nor the MMA
    // loop carry per-row state or placement loops in registers.
    if constexpr (Rows::POSITIONS) {
        if (listing || rows_total > POSITION_ROWS) __trap();
        if (warp == 0) {
            const auto tiles_of = [](int total) { return (total + KEYS - 1) / KEYS; };
            const auto run_of = [parts, tiles_of](int total) {
                return static_cast<int>(partition_run(tiles_of(total) + PICK_TILES, Rows::MIN_TILES, parts));
            };
            const int valid = max(0, static_cast<int>(min(static_cast<long long>(QT), rows_total - first_token)));
            int row_totals = 0, row_historicals = 0, row_freshes = 0, row_runs = 0;
            int row_shift0 = 0, row_end0 = 0;
            int row_starts = 0, row_ends = 0;
            int row_hist_starts = 0, row_hist_ends = 0;
            int u_total = 0, u_hist = 0, u_fresh = 0, u_run = 1;
            int u_shift0 = 0, u_end0 = 0;
            int head_row = 0, tail_row = 0;
            int hist_head_row = 0, hist_tail_row = 0;
            int max_start = 0, min_end = valid > 0 ? 0x7fffffff : 0;
            int max_hist_start = 0, min_hist_end = valid > 0 ? 0x7fffffff : 0;
            int active_parts = 0;
            int pref_mode = 0;
            if (POSITION_ROWS <= 8 && spans < 4) {
                const int sub_r = lane >> 2;
                const int sub_s = lane & 3;
                const Span own = (sub_r < valid && sub_s <= spans)
                    ? span(in, static_cast<u64>(first_token + sub_r), static_cast<u64>(sub_s), static_cast<u64>(spans))
                    : Span{0, 0};
                const int length = max(own.hi - own.lo, 0);
                int inclusive = length;
                int n = __shfl_up_sync(0xffffffffu, inclusive, 1, 4);
                if (sub_s >= 1) inclusive += n;
                n = __shfl_up_sync(0xffffffffu, inclusive, 2, 4);
                if (sub_s >= 2) inclusive += n;
                const int s_total = __shfl_sync(0xffffffffu, inclusive, 3, 4);
                const int s_hist = s_total - __shfl_sync(0xffffffffu, length, spans, 4);
                const int s_fresh = __shfl_sync(0xffffffffu, own.lo, spans, 4) - s_hist;
                const int s_run = run_of(s_total);
                const int s_first = partition * s_run * KEYS;
                const u32 all_hits = __ballot_sync(0xffffffffu, sub_s <= spans && inclusive > s_first);
                const u32 sub_hits = (all_hits >> (sub_r * 4)) & 0xfu;
                const int hit = sub_hits != 0u ? (__ffs(sub_hits) - 1) : 0;
                const int hit_shift = __shfl_sync(0xffffffffu, own.lo - (inclusive - length), hit, 4);
                const int hit_end = __shfl_sync(0xffffffffu, inclusive, hit, 4);
                const int s_shift0 = sub_hits != 0u ? hit_shift : 0;
                const int s_end0 = sub_hits != 0u ? hit_end : 0;

                // Union span [min_lo, max_hi) across all valid rows for each span index sub_s.
                int u_lo = length > 0 ? own.lo : 0x7fffffff;
                int u_hi = length > 0 ? own.hi : -0x7fffffff;
                u_lo = min(u_lo, __shfl_xor_sync(0xffffffffu, u_lo, 4));
                u_hi = max(u_hi, __shfl_xor_sync(0xffffffffu, u_hi, 4));
                u_lo = min(u_lo, __shfl_xor_sync(0xffffffffu, u_lo, 8));
                u_hi = max(u_hi, __shfl_xor_sync(0xffffffffu, u_hi, 8));
                u_lo = min(u_lo, __shfl_xor_sync(0xffffffffu, u_lo, 16));
                u_hi = max(u_hi, __shfl_xor_sync(0xffffffffu, u_hi, 16));
                if (u_hi <= u_lo) u_lo = u_hi = 0;
                const int u_len = max(u_hi - u_lo, 0);
                int u_inc = u_len;
                int un = __shfl_up_sync(0xffffffffu, u_inc, 1, 4);
                if (sub_s >= 1) u_inc += un;
                un = __shfl_up_sync(0xffffffffu, u_inc, 2, 4);
                if (sub_s >= 2) u_inc += un;
                const int u_base = u_inc - u_len;
                u_total = __shfl_sync(0xffffffffu, u_inc, 3, 4);
                u_hist = u_total - __shfl_sync(0xffffffffu, u_len, spans, 4);
                u_fresh = __shfl_sync(0xffffffffu, u_lo, spans, 4) - u_hist;
                u_run = run_of(u_total);
                const int u_first = partition * u_run * KEYS;
                const u32 u_hits = __ballot_sync(0xffffffffu, sub_r == 0 && sub_s <= spans && u_inc > u_first) & 0xfu;
                const int u_hit = u_hits != 0u ? (__ffs(u_hits) - 1) : 0;
                u_shift0 = u_hits != 0u ? __shfl_sync(0xffffffffu, u_lo - u_base, u_hit, 4) : 0;
                u_end0 = u_hits != 0u ? __shfl_sync(0xffffffffu, u_inc, u_hit, 4) : 0;

                // Each row is a contiguous subsequence [s_start, s_end) of the union when its
                // leading non-empty span reaches u_hi, its interior spans match the union, and
                // its trailing non-empty span begins at u_lo. Check both all spans 0..=spans
                // (standard causal) and history spans 0..spans + 1-token self fresh (cached causal).
                const u32 all_ne = __ballot_sync(0xffffffffu, sub_r < valid && sub_s <= spans && length > 0);
                const u32 sub_ne = (all_ne >> (sub_r * 4)) & 0xfu;
                const int first_ne = sub_ne != 0u ? (__ffs(sub_ne) - 1) : -1;
                const int last_ne = sub_ne != 0u ? (31 - __clz(sub_ne)) : -1;
                const bool span_ok =
                    sub_r >= valid || sub_s > spans || first_ne < 0 || sub_s < first_ne || sub_s > last_ne ||
                    (first_ne == last_ne) ||
                    (sub_s == first_ne ? own.hi == u_hi
                                       : (sub_s == last_ne
                                              ? own.lo == u_lo
                                              : ((u_len == 0 && length == 0) ||
                                                 (length > 0 && own.lo == u_lo && own.hi == u_hi))));
                const bool spans_ok = __all_sync(0xffffffffu, span_ok) != 0;
                const int first_offset = __shfl_sync(0xffffffffu, u_base + (own.lo - u_lo), max(first_ne, 0), 4);
                const int s_start = first_ne >= 0 ? first_offset : 0;
                const int s_end = s_start + s_total;

                const u32 sub_hist_ne = sub_ne & ((1u << spans) - 1u);
                const int first_hist_ne = sub_hist_ne != 0u ? (__ffs(sub_hist_ne) - 1) : -1;
                const int last_hist_ne = sub_hist_ne != 0u ? (31 - __clz(sub_hist_ne)) : -1;
                const bool hist_span_ok =
                    sub_r >= valid || sub_s >= spans || first_hist_ne < 0 || sub_s < first_hist_ne ||
                    sub_s > last_hist_ne || (first_hist_ne == last_hist_ne) ||
                    (sub_s == first_hist_ne ? own.hi == u_hi
                                            : (sub_s == last_hist_ne
                                                   ? own.lo == u_lo
                                                   : ((u_len == 0 && length == 0) ||
                                                      (length > 0 && own.lo == u_lo && own.hi == u_hi))));
                const bool hist_spans_ok = __all_sync(0xffffffffu, hist_span_ok) != 0;
                const bool cached_fresh_ok =
                    FRESH &&
                    __all_sync(0xffffffffu, sub_r >= valid || sub_s != spans ||
                                                (own.lo == u_lo + sub_r && own.hi == u_lo + sub_r + 1)) != 0;
                const int first_hist_offset =
                    __shfl_sync(0xffffffffu, u_base + (own.lo - u_lo), max(first_hist_ne, 0), 4);
                const int s_hist_start = first_hist_ne >= 0 ? first_hist_offset : 0;
                const int s_hist_end = s_hist_start + s_hist;

                const int src = (lane & 7) * 4;
                row_totals = __shfl_sync(0xffffffffu, s_total, src);
                row_historicals = __shfl_sync(0xffffffffu, s_hist, src);
                row_freshes = __shfl_sync(0xffffffffu, s_fresh, src);
                row_runs = __shfl_sync(0xffffffffu, s_run, src);
                row_shift0 = __shfl_sync(0xffffffffu, s_shift0, src);
                row_end0 = __shfl_sync(0xffffffffu, s_end0, src);
                row_starts = __shfl_sync(0xffffffffu, s_start, src);
                row_ends = __shfl_sync(0xffffffffu, s_end, src);
                row_hist_starts = __shfl_sync(0xffffffffu, s_hist_start, src);
                row_hist_ends = __shfl_sync(0xffffffffu, s_hist_end, src);

                bool same_run = true;
                for (int i = 0; i < valid; ++i) {
                    const int total = __shfl_sync(0xffffffffu, row_totals, i);
                    const int run = __shfl_sync(0xffffffffu, row_runs, i);
                    const int r_start = __shfl_sync(0xffffffffu, row_starts, i);
                    const int r_end = __shfl_sync(0xffffffffu, row_ends, i);
                    const int r_hstart = __shfl_sync(0xffffffffu, row_hist_starts, i);
                    const int r_hend = __shfl_sync(0xffffffffu, row_hist_ends, i);
                    if (r_start == 0) head_row = i;
                    if (r_end == u_total) tail_row = i;
                    if (r_hstart == 0) hist_head_row = i;
                    if (r_hend == u_hist) hist_tail_row = i;
                    max_start = max(max_start, r_start);
                    min_end = min(min_end, r_end);
                    max_hist_start = max(max_hist_start, r_hstart);
                    min_hist_end = min(min_hist_end, r_hend);
                    if (run != u_run) same_run = false;
                    active_parts = max(active_parts, (tiles_of(total) + PICK_TILES + run - 1) / run);
                }
                const bool std_pref =
                    valid > 0 && spans_ok && max_start < min_end && (same_run || (!PICKS && max_start > 0));
                const bool cached_pref =
                    !std_pref && valid > 0 && hist_spans_ok && cached_fresh_ok &&
                    max_hist_start < min_hist_end && (!PICKS || same_run);
                pref_mode = std_pref ? 1 : (cached_pref ? 2 : 0);
            } else {
                for (int i = 0; i < valid; ++i) {
                    int total, historical, fresh;
                    const SpanPlacement p0 = position_place(
                        in, first_token + i, spans,
                        [&](int rows) { return partition * run_of(rows) * KEYS; }, lane, total, historical, fresh);
                    const int run = run_of(total);
                    if (lane == i) {
                        row_totals = total;
                        row_historicals = historical;
                        row_freshes = fresh;
                        row_runs = run;
                        row_shift0 = p0.shift;
                        row_end0 = p0.end;
                    }
                    active_parts = max(active_parts, (tiles_of(total) + PICK_TILES + run - 1) / run);
                }
                int u_base = 0;
                bool started = false, ended = false, row_ok = true;
                int r_start_lane = 0;
                bool hist_row_ok = true, hist_started = false;
                int r_hist_start_lane = 0;
                bool cached_fresh_lane = true;
                for (int index = 0; index <= spans; ++index) {
                    const Span own = lane < valid
                        ? span(in, static_cast<u64>(first_token + lane), static_cast<u64>(index), static_cast<u64>(spans))
                        : Span{0, 0};
                    const int len = max(own.hi - own.lo, 0);
                    int u_lo = len > 0 ? own.lo : 0x7fffffff;
                    int u_hi = len > 0 ? own.hi : -0x7fffffff;
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        u_lo = min(u_lo, __shfl_xor_sync(0xffffffffu, u_lo, offset));
                        u_hi = max(u_hi, __shfl_xor_sync(0xffffffffu, u_hi, offset));
                    }
                    if (u_hi <= u_lo) u_lo = u_hi = 0;
                    const int u_len = max(u_hi - u_lo, 0);
                    if (index == spans) {
                        hist_row_ok = row_ok;
                        hist_started = started;
                        r_hist_start_lane = r_start_lane;
                        u_hist = u_base;
                        u_fresh = u_lo - u_hist;
                        if (lane < valid) {
                            cached_fresh_lane = (own.lo == u_lo + lane && own.hi == u_lo + lane + 1);
                        }
                    }
                    if (lane < valid) {
                        if (len == 0) {
                            if (started && u_len > 0) ended = true;
                        } else {
                            if (ended) row_ok = false;
                            if (!started) {
                                started = true;
                                r_start_lane = u_base + (own.lo - u_lo);
                            } else if (own.lo != u_lo) {
                                row_ok = false;
                            }
                            if (own.hi < u_hi) ended = true;
                        }
                    }
                    u_base += u_len;
                }
                u_total = u_base;
                u_run = run_of(u_total);
                row_starts = r_start_lane;
                row_ends = started ? r_start_lane + row_totals : 0;
                row_hist_starts = r_hist_start_lane;
                row_hist_ends = hist_started ? r_hist_start_lane + row_historicals : 0;
                const bool spans_ok = __all_sync(0xffffffffu, lane >= valid || row_ok) != 0;
                const bool hist_spans_ok = __all_sync(0xffffffffu, lane >= valid || hist_row_ok) != 0;
                const bool cached_fresh_ok = FRESH && __all_sync(0xffffffffu, lane >= valid || cached_fresh_lane) != 0;
                bool same_run = true;
                for (int i = 0; i < valid; ++i) {
                    const int run = __shfl_sync(0xffffffffu, row_runs, i);
                    const int r_start = __shfl_sync(0xffffffffu, row_starts, i);
                    const int r_end = __shfl_sync(0xffffffffu, row_ends, i);
                    const int r_hstart = __shfl_sync(0xffffffffu, row_hist_starts, i);
                    const int r_hend = __shfl_sync(0xffffffffu, row_hist_ends, i);
                    if (r_start == 0) head_row = i;
                    if (r_end == u_total) tail_row = i;
                    if (r_hstart == 0) hist_head_row = i;
                    if (r_hend == u_hist) hist_tail_row = i;
                    max_start = max(max_start, r_start);
                    min_end = min(min_end, r_end);
                    max_hist_start = max(max_hist_start, r_hstart);
                    min_hist_end = min(min_hist_end, r_hend);
                    if (run != u_run) same_run = false;
                }
                const bool std_pref =
                    valid > 0 && spans_ok && max_start < min_end && (same_run || (!PICKS && max_start > 0));
                const bool cached_pref =
                    !std_pref && valid > 0 && hist_spans_ok && cached_fresh_ok &&
                    max_hist_start < min_hist_end && (!PICKS || same_run);
                pref_mode = std_pref ? 1 : (cached_pref ? 2 : 0);
                if (pref_mode != 0) {
                    const int u_head = pref_mode == 2 ? hist_head_row : head_row;
                    const int u_tail = pref_mode == 2 ? hist_tail_row : tail_row;
                    const int u_first = partition * u_run * KEYS;
                    if (u_first < u_total) {
                        int base = 0;
                        for (int index = 0; index <= spans; ++index) {
                            Span s = span(in, static_cast<u64>(first_token + u_head), static_cast<u64>(index),
                                          static_cast<u64>(spans));
                            if (u_tail != u_head) {
                                const Span s1 = span(in, static_cast<u64>(first_token + u_tail),
                                                     static_cast<u64>(index), static_cast<u64>(spans));
                                if (s1.hi > s1.lo) {
                                    if (s.hi > s.lo) {
                                        s.lo = min(s.lo, s1.lo);
                                        s.hi = max(s.hi, s1.hi);
                                    } else {
                                        s = s1;
                                    }
                                }
                            }
                            const int len = max(s.hi - s.lo, 0);
                            if (u_first < base + len) {
                                u_shift0 = s.lo - base;
                                u_end0 = base + len;
                                break;
                            }
                            base += len;
                        }
                    }
                }
            }
            const bool pref = pref_mode != 0;
            const bool cached_fresh = pref_mode == 2;
            const int u_head = cached_fresh ? hist_head_row : head_row;
            const int u_tail = cached_fresh ? hist_tail_row : tail_row;
            if (pref) {
                const int u_parts = (tiles_of(u_total) + u_run - 1) / u_run;
                if constexpr (!PICKS) {
                    active_parts = u_parts;
                } else {
                    active_parts = max(active_parts, u_parts);
                }
            }
            if (lane < POSITION_ROWS) {
                plan->row_end[lane] = lane < valid ? (cached_fresh ? row_hist_ends : (pref ? row_ends : row_totals)) : 0;
                if (pref) plan->single.row_start[lane] = lane < valid ? (cached_fresh ? row_hist_starts : row_starts) : 0;
            }
            if (lane == 0 && pref) plan->single.tail_row = u_tail;
            const int vis_walks = pref ? 1 : valid;
            const int all_walks = vis_walks + (PICKS ? valid : 0);
            int out_walks = 0;
            int units_sum = 0;
            if (partition < active_parts) {
                for (int w = 0; w < all_walks; ++w) {
                    const bool is_union = pref && w == 0;
                    const int r = w < vis_walks ? (pref ? u_head : w) : w - vis_walks;
                    const int r_total = is_union ? u_total : __shfl_sync(0xffffffffu, row_totals, r);
                    const int r_hist = is_union ? u_hist : __shfl_sync(0xffffffffu, row_historicals, r);
                    const int r_fresh = is_union ? u_fresh : __shfl_sync(0xffffffffu, row_freshes, r);
                    const int r_run = is_union ? u_run : __shfl_sync(0xffffffffu, row_runs, r);
                    const int r_shift0 = is_union ? u_shift0 : __shfl_sync(0xffffffffu, row_shift0, r);
                    const int r_end0 = is_union ? u_end0 : __shfl_sync(0xffffffffu, row_end0, r);
                    const int lo = partition * r_run;
                    const int vis = tiles_of(r_total);
                    const int2 tiles = w < vis_walks ? int2{lo, min(lo + r_run, vis)}
                                                     : int2{max(lo, vis), min(lo + r_run, vis + PICK_TILES)};
                    if (tiles.x >= tiles.y) continue;
                    units_sum += tiles.y - tiles.x;
                    if (lane == 0) {
                        if (PICKS && w >= vis_walks) {
                            const int base = r * PICK_COUNT;
                            plan->walk[out_walks] = PositionWalk{
                                r, base + (tiles.x - vis) * KEYS, base + (tiles.y - vis) * KEYS,
                                -1, 0, 0, 0};
                        } else {
                            const int walk_first = tiles.x * KEYS;
                            const int walk_end = min(tiles.y * KEYS, r_total);
                            plan->walk[out_walks] = PositionWalk{
                                r, walk_first, walk_end,
                                r_hist, r_fresh, r_shift0, r_end0};
                        }
                    }
                    ++out_walks;
                }
            }
            if (lane == 0) {
                plan->visible = in.visible;
                plan->vis_s0 = SEISMIC_VISIBLE_STRIDE_0;
                plan->vis_s1 = SEISMIC_VISIBLE_STRIDE_1;
                plan->first_token = static_cast<int>(first_token);
                plan->spans = spans;
                plan->active = active_parts;
                plan->units = units_sum * PIECES;
                plan->walks = out_walks;
                plan->prefixed = pref_mode;
                plan->full_start = pref ? (cached_fresh ? max_hist_start : max_start) : 1;
                plan->full_end = pref ? (cached_fresh ? min_hist_end : min_end) : 0;
            }
        }
    }
    __syncthreads();

    auto settle = [&](Tile tile) {
        while (tile.span <= last && tile.first >= table[tile.span * 4 + 1]) {
            ++tile.span;
            if (tile.span <= last) tile.first = table[tile.span * 4 + 0];
        }
        return tile;
    };
    auto span_tiles = [&](int index) {
        const int lo = table[index * 4 + 0], hi = table[index * 4 + 1];
        return hi > lo ? (hi - lo + KEYS - 1) / KEYS : 0;
    };

    int active = 1;
    int total_tiles = 0;
    int tiles_lo = 0;
    int units = 0;
    if constexpr (Rows::POSITIONS) {
        active = plan->active;
        if (partition >= active) {
            rows.vacant(in, block, first_token);
            return;
        }
        units = plan->units;
        if (units == 0) __trap();
    } else {
        total_tiles = listing
            ? static_cast<int>(selection.listed[block.tile])
            : static_cast<int>(reduce::span_total(static_cast<u64>(last) + 1, span_tiles, lane));
        const int per = static_cast<int>(partition_run(total_tiles, Rows::MIN_TILES, parts));
        active = max(1, (total_tiles + per - 1) / per);
        if (partition >= active) {
            rows.vacant(in, block, first_token);
            return;
        }
        tiles_lo = partition * per;
        units = max(0, min(per, total_tiles - tiles_lo)) * PIECES;
    }
    rows.partitioned(block, active);

    if (warp < WARPS * COLUMNS)
        rows.stage_queries(in, history, q_tile, exchange, block, first_token, warp, lane);

    // The warp's query fragments (row block warp % WARPS, every 16-column
    // step of the head), when they stay in registers (loaded after partition
    // bounds so vacant partitions skip them and setup never pins `qa`).
    u32 qa[Q_REGISTERS ? SPAN / 16 : 1][4];
    if constexpr (Q_REGISTERS) {
        seismic_cp_async_wait<0>();
        __syncthreads();
        if (warp < WARPS * COLUMNS) {
#pragma unroll
            for (int step = 0; step < SPAN / 16; ++step) {
                const int row = (warp % WARPS) * 16 + (lane % 8) + 8 * ((lane / 8) % 2);
                // With split scores, the column warp's own span.
                const int span0 = SPLIT_SCORES ? (warp / WARPS) * SPAN / 8 : 0;
                seismic_ldmatrix_x4(qa[step], q_tile + swizzled(row, span0 + 2 * step + lane / 16));
            }
        }
        // The region is the K/V stages' from here on.
        __syncthreads();
    } else if constexpr (PRODUCED) {
        // Order `DecodeRows::stage_queries`'s use of `exchange` before producer
        // warps can touch `exchange` in `stage_fresh_at`.
        __syncthreads();
    }

    // Tile `index` of the query tile's sequence (`reduce::span_locate` over
    // the span tiles; the whole warp calls it).
    auto nth = [&](int index) {
        if (listing) {
            if (index >= total_tiles) return Tile{last + 1, 0, index};
            const int *entry = selection.lists + (static_cast<long long>(block.tile) * selection.capacity + index) * 2;
            return Tile{entry[0], entry[1], index};
        }
        const reduce::Located at = reduce::span_locate(static_cast<u64>(last) + 1, index, span_tiles, lane);
        const int span = static_cast<int>(at.span);
        if (span > last) return Tile{last + 1, 0};
        return Tile{span, table[span * 4 + 0] + (index - static_cast<int>(at.offset)) * KEYS};
    };
    struct PositionStep {
        int index;
        int first;
    };
    using Step = typename SelectType<Rows::POSITIONS, PositionStep, Tile>::Type;
    auto first_step = [&]() -> Step {
        if constexpr (Rows::POSITIONS) return PositionStep{0, plan->walk[0].first};
        else return nth(tiles_lo);
    };
    auto is_picks = [&](const Step &step) -> bool {
        if constexpr (!PICKS) return false;
        else if constexpr (Rows::POSITIONS) return plan->walk[step.index].historical < 0;
        else return step.span > spans;
    };
    auto history_rows = [&](const Step &step) -> int {
        if constexpr (Rows::POSITIONS)
            return step.index >= plan->walks
                ? 0
                : min(max(plan->walk[step.index].historical - step.first, 0), KEYS);
        else return step.span < spans ? KEYS : 0;
    };
    // Rare multi-span straddling history tile (`step.first + held > w.span0_end`):
    // lane `lane_r` resolves row `lane_r`'s history token once per tile (`KEYS <= 32`)
    // from `plan` behind a value barrier so the hot loop carries no span state.
    static_assert(KEYS <= 32, "a key tile's rows fit one warp's lanes");
    auto cold_lane_token = [&](int s_idx, int s_first, int &s_held, int lane_r) -> int {
        u8 *p_base = reinterpret_cast<u8 *>(plan);
        asm volatile("" : "+r"(s_idx), "+r"(s_first), "+r"(s_held), "+l"(p_base));
        const PositionPlan *lp = reinterpret_cast<const PositionPlan *>(p_base);
        const PositionWalk &w = lp->walk[s_idx];
        int st = -1;
        if (lane_r < s_held) {
            const int ordinal = s_first + lane_r;
            if (ordinal < w.span0_end) {
                st = ordinal + w.shift0;
            } else {
                const int *vis = lp->visible;
                const u64 s0 = lp->vis_s0;
                const u64 s1 = lp->vis_s1;
                const int *vis_head = vis + static_cast<u64>(lp->first_token + w.row) * s0;
                const int tail_row = lp->prefixed ? lp->single.tail_row : w.row;
                const int *vis_tail = vis + static_cast<u64>(lp->first_token + tail_row) * s0;
                const int nspans = lp->spans;
                int base = 0;
                for (int index = 0; index < nspans; ++index) {
                    const int *sp0 = vis_head + static_cast<u64>(index) * s1;
                    int lo = sp0[0];
                    int hi = sp0[SEISMIC_VISIBLE_STRIDE_2];
                    if (tail_row != w.row) {
                        const int *sp1 = vis_tail + static_cast<u64>(index) * s1;
                        const int lo1 = sp1[0];
                        const int hi1 = sp1[SEISMIC_VISIBLE_STRIDE_2];
                        if (hi1 > lo1) {
                            if (hi > lo) {
                                lo = min(lo, lo1);
                                hi = max(hi, hi1);
                            } else {
                                lo = lo1;
                                hi = hi1;
                            }
                        }
                    }
                    const int length = max(hi - lo, 0);
                    if (ordinal < base + length) {
                        st = lo + ordinal - base;
                        break;
                    }
                    base += length;
                }
                if (st < 0) __trap();
            }
        }
        return st;
    };
    auto tokens_at = [](int row_tokens) {
        return [row_tokens](int r) -> int { return __shfl_sync(0xffffffffu, row_tokens, r & 31); };
    };
    auto stage_fresh_at = [&](const Step &step, int held, int buffer, int piece, int column0, int thread,
                              int threads) {
        if constexpr (Rows::POSITIONS) {
            const PositionWalk &w = plan->walk[step.index];
            rows.stage_fresh(in, history, k_tile(buffer), v_tile(buffer), w.fresh + step.first,
                             w.fresh + w.end_key, held, kv, piece, column0, thread, threads,
                             exchange);
        } else
            rows.stage_fresh(in, history, k_tile(buffer), v_tile(buffer), step.first,
                             table[step.span * 4 + 1], 0, kv, piece, column0, thread, threads, exchange);
    };
    auto stage_picked = [&](const Step &step, int buffer, int piece, int column0) {
        int end = 0;
        if constexpr (Rows::POSITIONS) end = (plan->walk[step.index].row + 1) * PICK_COUNT;
        else end = table[step.span * 4 + 1];
        auto vector = [&](int r) -> const u8 * {
            const int ordinal = step.first + r;
            const int row = ordinal < end ? pick_row(ordinal) : -1;
            return row >= 0 ? picked.vector(row, kv) : nullptr;
        };
        stage_slab<F32_ACTIVATION, PIECE_CHUNKS>(k_tile(buffer), KEYS, [&](int r) -> const u8 * {
            const u8 *at = vector(r);
            return at != nullptr ? at + piece * PIECE * Act::bytes : nullptr;
        }, threadIdx.x, MMA_THREADS);
        if (SHARED_KV || piece + 1 < PIECES) return;
        stage_slab<F32_ACTIVATION, WINDOW_CHUNKS>(v_tile(buffer), KEYS, [&](int r) -> const u8 * {
            const u8 *at = vector(r);
            return at != nullptr ? at + column0 * Act::bytes : nullptr;
        }, threadIdx.x, MMA_THREADS);
    };
    auto advance = [&](Step &step, int &piece) {
        if (++piece == PIECES) {
            piece = 0;
            if constexpr (Rows::POSITIONS) {
                step.first += KEYS;
                if (step.first >= plan->walk[step.index].end_key && ++step.index < plan->walks)
                    step.first = plan->walk[step.index].first;
            } else {
                step = listing ? nth(step.index + 1) : settle(Tile{step.span, step.first + KEYS});
            }
        }
    };

#ifdef ATTENTION_AFFINE_HISTORY
    if constexpr (PRODUCED) {
        if (warp >= WARPS * COLUMNS) {
            const int thread = threadIdx.x - MMA_THREADS;
            auto is_straddle = [&](const Step &s, int held) -> bool {
                if constexpr (Rows::POSITIONS) return held > 0 && s.first + held > plan->walk[s.index].span0_end;
                else return false;
            };
            auto issue_fetch = [&](const Step &s, int s_piece, int held, int column0) {
                int f0 = 0, rows_cnt = 0;
                if constexpr (Rows::POSITIONS) {
                    f0 = s.first + plan->walk[s.index].shift0;
                    rows_cnt = held;
                } else {
                    f0 = s.first;
                    rows_cnt = min(KEYS, max(table[s.span * 4 + 1] - f0, 0));
                }
                return Affine::fetch_produced(history, f0, rows_cnt, kv, s_piece, column0, thread);
            };
            auto produce_straddled = [&](const Step &s, int held, int buffer, int s_piece, int column0) {
                if constexpr (Rows::POSITIONS) {
                    int s_held = held;
                    const int st = cold_lane_token(s.index, s.first, s_held, thread & 31);
                    int step_p = PRODUCERS;
                    asm("" : "+r"(step_p));
                    const int k_total = s_held * Affine::KEY_PIECES;
                    const int k_span = ((k_total + PRODUCERS - 1) / PRODUCERS) * PRODUCERS;
                    const int v_total = s_piece + 1 == PIECES ? s_held * Affine::VALUE_PIECES : 0;
                    for (int base_idx = 0; base_idx < k_span + v_total; base_idx += step_p) {
                        const bool is_v = base_idx >= k_span;
                        const int local = (is_v ? base_idx - k_span : base_idx) + thread;
                        const int limit = is_v ? v_total : k_total;
                        const int pieces = is_v ? Affine::VALUE_PIECES : Affine::KEY_PIECES;
                        const int r = min(local / pieces, KEYS - 1);
                        const int tok = __shfl_sync(0xffffffffu, st, r);
                        if (local < limit && tok >= 0)
                            Affine::produce_piece(history.vectors(tok, kv), is_v, r, local % pieces, s_piece, column0,
                                                  k_tile(buffer), v_tile(buffer));
                    }
                }
            };
            for (int window = 0; window < WINDOWS; ++window) {
                const int column0 = window * WINDOW;
                Step step = first_step();
                int piece = 0;
                Affine::ProducedCodes ahead{};
                const int first_held = units > 0 ? history_rows(step) : 0;
                if (first_held > 0 && !is_straddle(step, first_held))
                    ahead = issue_fetch(step, piece, first_held, column0);
                for (int j = 0; j < units; ++j) {
                    const int i = window * units + j;
                    const int b = i % STAGES;
                    const int cur_held = history_rows(step);
                    const bool cur_straddle = is_straddle(step, cur_held);
                    Step next = step;
                    int next_piece = piece;
                    advance(next, next_piece);
                    const int next_held = j + 1 < units ? history_rows(next) : 0;
                    const bool next_prefetched = next_held > 0 && !is_straddle(next, next_held);
                    if (cur_held == KEYS && !cur_straddle) {
                        const Affine::ProducedCodes current = ahead;
                        ahead = next_prefetched ? issue_fetch(next, next_piece, next_held, column0) : Affine::ProducedCodes{};
                        if (i >= STAGES) named_sync(EMPTY + b, THREADS);
                        Affine::store_produced(current, k_tile(b), v_tile(b), KEYS, piece, thread);
                    } else {
                        if (i >= STAGES) named_sync(EMPTY + b, THREADS);
                        if (cur_straddle) produce_straddled(step, cur_held, b, piece, column0);
                        else if (cur_held > 0) Affine::store_produced(ahead, k_tile(b), v_tile(b), cur_held, piece, thread);
                        if (cur_held < KEYS) {
                            stage_fresh_at(step, cur_held, b, piece, column0, thread, PRODUCERS);
                            seismic_cp_async_commit();
                            seismic_cp_async_wait<0>();
                        }
                        ahead = next_prefetched ? issue_fetch(next, next_piece, next_held, column0) : Affine::ProducedCodes{};
                    }
                    __threadfence_block();
                    named_arrive(FULL + b, THREADS);
                    step = next;
                    piece = next_piece;
                }
            }
            return;
        }
    }
#endif

    // MMA warps: row block `row_block` (rows 16 row_block ..), output
    // fragments [n0, n0 + OUT) of the window. This lane's two matrix rows
    // (g and g + 8 of the block's 16).
    const int g = lane / 4;
    const int t = lane % 4;
    const float query_scale = in.scale * LOG2E;
    const int row_block = warp % WARPS;
    const int n0 = (warp / WARPS) * OUT;
    const long long token_a = first_token + (row_block * 16 + g) / G;
    const long long token_b = first_token + (row_block * 16 + g + 8) / G;
    const bool valid_a = row_block * 16 + g < QT * G && token_a < rows_total;
    const bool valid_b = row_block * 16 + g + 8 < QT * G && token_b < rows_total;
    // A block of padding rows only (the tile's last rows, or with decode rows
    // most blocks) makes every call of the others and skips the products.
    const bool computes = row_block * 16 < QT * G && first_token + row_block * 16 / G < rows_total;

    // Produced affine history: the producers stage every K/V tile, so the MMA
    // warps complete their query tile copy here.
#ifdef ATTENTION_AFFINE_HISTORY
    if constexpr (PRODUCED) {
        seismic_cp_async_wait<0>();
        named_sync(MMA_WARPS, MMA_THREADS);
    }
#endif

    const float NEG_INF = -__int_as_float(0x7f800000);
    // The output window's accumulators and both rows' softmax states, reset
    // per pass.
    float o[OUT][4];
    float maximum[2];
    float denominator[2];
    int cached_span = -1;
    Span interval_a{0, 0}, interval_b{0, 0};
    // The rows' raw span bounds and key ordinals, for a WorkList mask.
    Span raw_a{0, 0}, raw_b{0, 0};
    // A key tile's scores, accumulated over its pieces.
    float s[KEYS / 8][4];
    // Split scores: this warp's column, the key piece and first 16-column step
    // of its span, and the parity of its next exchanged tile.
    const int column = warp / WARPS;
    const int span_piece = column * SPAN / PIECE;
    const int span_step = column * SPAN % PIECE / 16;
    int parity = 0;

    // The products of one staged unit: S = Q K^T over the key piece's
    // columns; after the tile's last piece, the score exchange, the softmax
    // and O += P V.
    auto absorb_unit = [&](const Step &current, int piece, const u8 *k_base, const u8 *v_base) {
        if (piece == 0) {
#pragma unroll
            for (int n = 0; n < KEYS / 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
        }
        if (computes && (!SPLIT_SCORES || piece == span_piece)) {
#pragma unroll
        for (int index = 0; index < SPAN / 16; ++index) {
            const int step = SPLIT_SCORES ? span_step + index : index;
            u32 a[4];
            if constexpr (Q_REGISTERS) {
#pragma unroll
                for (int e = 0; e < 4; ++e) a[e] = qa[index][e];
            } else {
                const int row = row_block * 16 + (lane % 8) + 8 * ((lane / 8) % 2);
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
        }
        if (piece + 1 < PIECES) return;

        if constexpr (SPLIT_SCORES) {
            // Publish this warp's span scores, meet the row block's other
            // column warps, and add every span's scores in column order.
            float *group = split_exchange + ((parity * WARPS + row_block) * COLUMNS) * 32 * SPLIT_LANE_FLOATS;
            float *own = group + (column * 32 + lane) * SPLIT_LANE_FLOATS;
#pragma unroll
            for (int n = 0; n < KEYS / 8; ++n)
                *reinterpret_cast<float4 *>(own + 4 * n) = make_float4(s[n][0], s[n][1], s[n][2], s[n][3]);
            named_sync(PAIR + row_block, COLUMNS * 32);
#pragma unroll
            for (int n = 0; n < KEYS / 8; ++n) {
                const float4 mine = make_float4(s[n][0], s[n][1], s[n][2], s[n][3]);
#pragma unroll
                for (int c = 0; c < COLUMNS; ++c) {
                    const float4 x = c == column ? mine
                                                 : *reinterpret_cast<const float4 *>(
                                                       group + (c * 32 + lane) * SPLIT_LANE_FLOATS + 4 * n);
                    if (c == 0) {
                        s[n][0] = x.x, s[n][1] = x.y, s[n][2] = x.z, s[n][3] = x.w;
                    } else {
                        s[n][0] += x.x, s[n][1] += x.y, s[n][2] += x.z, s[n][3] += x.w;
                    }
                }
            }
            parity = (parity + 1) % PARITIES;
        }
        if (!computes) return;

        const bool picks = is_picks(current);
        bool full;
        if constexpr (Rows::POSITIONS) {
            full = !picks && current.first >= plan->full_start && current.first + KEYS <= plan->full_end;
        } else {
            if (current.span != cached_span) {
                cached_span = current.span;
                // A row reads its own token's picks, each masked by its pick.
                auto interval = [&](bool valid, long long token) {
                    if (!valid) return Span{0, 0};
                    const int own = static_cast<int>(token - first_token) * PICK_COUNT;
                    return picks ? Span{own, own + PICK_COUNT} : span(in, token, current.span, spans);
                };
                interval_a = interval(valid_a, token_a);
                interval_b = interval(valid_b, token_b);
                raw_a = interval_a;
                raw_b = interval_b;
            }
            full = !listing && current.first >= table[current.span * 4 + 2] &&
                   current.first + KEYS <= table[current.span * 4 + 3];
        }

        // Scale into the exp2 domain, mask, then the online softmax of both rows.
#pragma unroll
        for (int n = 0; n < KEYS / 8; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) s[n][e] *= query_scale;
        if (!full) {
            int walked_row = 0, start_a = 0, start_b = 0, end_a = 0, end_b = 0;
            int row_a = -1, row_b = -1, hist_limit = 0;
            bool cached_fresh = false;
            if constexpr (Rows::POSITIONS) {
                const PositionWalk &w = plan->walk[current.index];
                walked_row = w.row;
                const int pref_mode = plan->prefixed;
                const bool pref = pref_mode != 0;
                cached_fresh = pref_mode == 2;
                hist_limit = w.historical;
                const int ra = static_cast<int>(token_a - first_token);
                const int rb = static_cast<int>(token_b - first_token);
                if (valid_a && (pref || ra == walked_row)) {
                    row_a = ra;
                    start_a = pref ? plan->single.row_start[ra] : 0;
                    end_a = plan->row_end[ra];
                }
                if (valid_b && (pref || rb == walked_row)) {
                    row_b = rb;
                    start_b = pref ? plan->single.row_start[rb] : 0;
                    end_b = plan->row_end[rb];
                }
            }
#pragma unroll
            for (int n = 0; n < KEYS / 8; ++n) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int key = current.first + 8 * n + 2 * t + (e & 1);
                    if constexpr (Rows::POSITIONS) {
                        if constexpr (PICKS) {
                            if (picks) {
                                if ((e < 2 ? token_a : token_b) != first_token + walked_row ||
                                    key >= (walked_row + 1) * PICK_COUNT ||
                                    pick_row(key) < 0)
                                    s[n][e] = NEG_INF;
                                continue;
                            }
                        }
                        if (cached_fresh && key >= hist_limit) {
                            if (key != hist_limit + (e < 2 ? row_a : row_b)) s[n][e] = NEG_INF;
                        } else if (key < (e < 2 ? start_a : start_b) || key >= (e < 2 ? end_a : end_b)) {
                            s[n][e] = NEG_INF;
                        }
                    } else {
                        const Span &interval = e < 2 ? interval_a : interval_b;
                        if (key < interval.lo || key >= interval.hi) {
                            s[n][e] = NEG_INF;
                        } else if (picks && pick_row(key) < 0) {
                            s[n][e] = NEG_INF;
                        } else if (listing &&
                                   !selection.reads(e < 2 ? token_a : token_b, current.span, spans, key,
                                                    (e < 2 ? raw_a : raw_b).lo)) {
                            s[n][e] = NEG_INF;
                        }
                    }
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
        for (int n = 0; n < OUT; ++n) {
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
            for (int n = 0; n < OUT; n += 2) {
                u32 b4[4];
                const int row = 16 * step + (lane % 8) + 8 * ((lane / 8) % 2);
                const int chunk = n0 + n + lane / 16;
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
        for (int n = 0; n < OUT; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
        maximum[0] = maximum[1] = NEG_INF;
        denominator[0] = denominator[1] = 0.0f;

        Step current = first_step();
        int piece = 0;
#ifdef ATTENTION_AFFINE_HISTORY
        if constexpr (PRODUCED) {
            // Staged unit i (counted over every window) waits for stage
            // i % STAGES to be full; once its products are issued, the stage
            // is released for unit i + STAGES when that unit exists.
            const int staged = WINDOWS * units;
            for (int j = 0; j < units; ++j) {
                const int i = (column0 / WINDOW) * units + j;
                const int b = i % STAGES;
                named_sync(FULL + b, THREADS);
                absorb_unit(current, piece, k_tile(b), v_tile(b));
                if (i + STAGES < staged) named_arrive(EMPTY + b, THREADS);
                advance(current, piece);
            }
        } else
#endif
        {
            // A history unit goes to its operand stage, or with inline
            // decoding as codes to its code stage.
            auto issue = [&](const Step &step, int tile_piece, int buffer) {
                const int held = history_rows(step);
                if (held > 0) {
                    bool straddle = false;
                    int f0 = 0, lim = 0, st = -1;
                    if constexpr (Rows::POSITIONS) {
                        const PositionWalk &w = plan->walk[step.index];
                        straddle = step.first + held > w.span0_end;
                        if (straddle) {
                            int s_held = held;
                            st = cold_lane_token(step.index, step.first, s_held, lane);
                        }
                        f0 = step.first + w.shift0;
                        lim = f0 + held;
                    } else {
                        f0 = step.first;
                        lim = table[step.span * 4 + 1];
                    }
#ifdef ATTENTION_AFFINE_HISTORY
                    if constexpr (INLINE) {
                        if (Rows::POSITIONS && straddle) {
                            Affine::stage_codes(history, code_tile(buffer), tokens_at(st), held, kv, tile_piece, column0,
                                        threadIdx.x, MMA_THREADS);
                        } else {
                            Affine::stage_codes(history, code_tile(buffer), [f0, lim](int r) { return f0 + r < lim ? f0 + r : -1; },
                                        held, kv, tile_piece, column0, threadIdx.x, MMA_THREADS);
                        }
                    } else
#endif
                    {
                        if (Rows::POSITIONS && straddle) {
                            stage_history(history, k_tile(buffer), v_tile(buffer), tokens_at(st), held, kv,
                                          tile_piece, column0);
                        } else {
                            stage_history(history, k_tile(buffer), v_tile(buffer),
                                          [f0, lim](int r) { return f0 + r < lim ? f0 + r : -1; },
                                          held, kv, tile_piece, column0);
                        }
                    }
                }
                if (is_picks(step)) {
                    stage_picked(step, buffer, tile_piece, column0);
                } else if (held < KEYS) {
                    stage_fresh_at(step, held, buffer, tile_piece, column0, threadIdx.x, MMA_THREADS);
                }
            };
            Step fetch = first_step();
            int fetch_piece = 0;
#ifdef ATTENTION_AFFINE_HISTORY
            if constexpr (INLINE && SPLIT_SCORES) {
                // One code stage: unit i's codes land, are decoded into operand
                // stage i % STAGES, and unit i + 1's copy is issued into the
                // freed code stage before unit i's products. The block barrier
                // opening each unit also orders the score exchange and the
                // operand stages' reuse.
                if (units > 0) {
                    issue(fetch, fetch_piece, 0);
                    advance(fetch, fetch_piece);
                }
                seismic_cp_async_commit();
                for (int i = 0; i < units; ++i) {
                    seismic_cp_async_wait<0>();
                    __syncthreads();
                    const int held = history_rows(current);
                    if (held > 0) {
                        Affine::decode_codes(code_tile(0), k_tile(i % STAGES), v_tile(i % STAGES), true, held,
                                     threadIdx.x, MMA_THREADS);
                    }
                    __syncthreads();
                    if (i + 1 < units) {
                        issue(fetch, fetch_piece, (i + 1) % STAGES);
                        advance(fetch, fetch_piece);
                    }
                    seismic_cp_async_commit();
                    absorb_unit(current, piece, k_tile(i % STAGES), v_tile(i % STAGES));
                    advance(current, piece);
                }
                __syncthreads();
            } else
#endif
            {
                // Unit i goes into stage i % STAGES, STAGES - 1 units ahead of
                // the products; one copy group is committed per unit slot (empty
                // past the last unit), so the wait for unit i is uniform.
                for (int ahead = 0; ahead < STAGES - 1; ++ahead) {
                    if (ahead < units) {
                        issue(fetch, fetch_piece, ahead);
                        advance(fetch, fetch_piece);
                    }
                    seismic_cp_async_commit();
                }
                for (int i = 0; i < units; ++i) {
                    const int ahead = i + STAGES - 1;
                    if (ahead < units) {
                        issue(fetch, fetch_piece, ahead % STAGES);
                        advance(fetch, fetch_piece);
                    }
                    seismic_cp_async_commit();
                    seismic_cp_async_wait<STAGES - 1>();
                    __syncthreads();
#ifdef ATTENTION_AFFINE_HISTORY
                    if constexpr (INLINE) {
                        const int held = history_rows(current);
                        if (held > 0) {
                            Affine::decode_codes(code_tile(i % STAGES), k_tile(i % STAGES), v_tile(i % STAGES),
                                         piece + 1 == PIECES, held, threadIdx.x, MMA_THREADS);
                            __syncthreads();
                        }
                    }
#endif

                    absorb_unit(current, piece, k_tile(i % STAGES), v_tile(i % STAGES));
                    __syncthreads();
                    advance(current, piece);
                }
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
            const int query_head = kv * G + (row_block * 16 + g + 8 * half) % G;
            rows.store(in, block, active, token, query_head, column0 + 8 * n0, o, half, t, maximum[half],
                       denominator[half]);
        }
    }
}

// Prefill L3: block (query tile, query head), one thread per column. A tile whose
// keys took several partitions merges each of its rows' partitions in
// partition order and applies the gate; other tiles were stored by L2.
__device__ __forceinline__ void merge_partitions(const Inputs &in, const Split &split, element::Out<Act> gated) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const int tile = static_cast<int>(blockIdx.x);
    const int query_head = static_cast<int>(blockIdx.y);
    const int column = static_cast<int>(threadIdx.x);
    const int count = static_cast<int>(split.counts[tile]);
    if (count <= 1) return;
    const long long rows_total = static_cast<long long>(SEISMIC_DIM_M);
    const u64 heads = KV * G;
    const u64 stat_stride = static_cast<u64>(rows_total) * heads;
    const u64 partial_stride = stat_stride * W;
    const long long start = static_cast<long long>(tile) * QT;
    const long long stop = min(static_cast<long long>(tile + 1) * QT, rows_total);
    constexpr int MAX_SHARED_PARTS = 256;
    if (count <= MAX_SHARED_PARTS) {
        __shared__ float shared_weights[MAX_SHARED_PARTS];
        __shared__ float shared_inv_norm;
        for (long long token = start; token < stop; ++token) {
            if (token > start) __syncthreads();
            const u64 slot = static_cast<u64>(token) * heads + query_head;
            if (column < 32) {
                constexpr int MAX_PER_LANE = MAX_SHARED_PARTS / 32;
                float2 stat[MAX_PER_LANE];
                const float2 *stats = reinterpret_cast<const float2 *>(split.statistics) + slot;
                float local_max = -__int_as_float(0x7f800000);
#pragma unroll
                for (int i = 0; i < MAX_PER_LANE; ++i) {
                    if (i * 32 >= count) break;
                    const int p = column + i * 32;
                    if (p < count) {
                        const float2 s = stats[static_cast<u64>(p) * stat_stride];
                        stat[i] = s;
                        if (s.y > 0.0f) local_max = fmaxf(local_max, s.x);
                    }
                }
                const float maximum = seismic_warp_max_f32(local_max);
                float local_denom = 0.0f;
#pragma unroll
                for (int i = 0; i < MAX_PER_LANE; ++i) {
                    if (i * 32 >= count) break;
                    const int p = column + i * 32;
                    if (p < count) {
                        const float l = stat[i].y;
                        float weight = 0.0f;
                        if (l > 0.0f) {
                            weight = seismic_ex2_approx(stat[i].x - maximum);
                            local_denom = __fmaf_rn(l, weight, local_denom);
                        }
                        shared_weights[p] = weight;
                    }
                }
                const float denominator = seismic_warp_sum_f32(local_denom);
                if (column == 0) {
                    shared_inv_norm = 1.0f / normalizer(in, query_head, maximum, denominator);
                }
            }
            __syncthreads();
            float accumulated = 0.0f;
            const float *values = split.partials + slot * W + column;
            for (int p = 0; p < count; ++p) {
                const float weight = shared_weights[p];
                if (weight > 0.0f) {
                    accumulated = __fmaf_rn(weight, values[static_cast<u64>(p) * partial_stride], accumulated);
                }
            }
            gated.put(static_cast<u64>(token) * SEISMIC_RESULT_0_STRIDE_0 +
                          static_cast<u64>(query_head) * SEISMIC_RESULT_0_STRIDE_1 +
                          static_cast<u64>(column) * SEISMIC_RESULT_0_STRIDE_2,
                      attention::gated(in, token, query_head, column, accumulated * shared_inv_norm));
        }
        return;
    }
    for (long long token = start; token < stop; ++token) {
        const u64 slot = static_cast<u64>(token) * heads + query_head;
        float denominator, accumulated;
        const float maximum = merge(split.statistics + slot * 2, stat_stride * 2,
                                    split.partials + slot * W + column, partial_stride, count,
                                    denominator, accumulated);
        gated.put(static_cast<u64>(token) * SEISMIC_RESULT_0_STRIDE_0 +
                      static_cast<u64>(query_head) * SEISMIC_RESULT_0_STRIDE_1 +
                      static_cast<u64>(column) * SEISMIC_RESULT_0_STRIDE_2,
                  attention::gated(in, token, query_head, column,
                                   accumulated / normalizer(in, query_head, maximum, denominator)));
    }
}

// The output rotary (RO = 1) of one published output column pair, run as its
// own launch after the entry has stored every query head's output: block
// (row, query head), thread `pair` turns its pair back by the row's negated
// angles (`output_rotary`), in F32, and publishes it again.
__device__ __forceinline__ void output_rotary(const Inputs &in, element::Out<Act> gated, u64 row, int head, int pair) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    if (pair >= P) return;
    float cosine, sine;
    row_table(in, row).turn(pair, cosine, sine);
    const int lo = head_layout::leading(pair);
    const int hi = head_layout::partner(lo);
    const u64 out_at = row * SEISMIC_RESULT_0_STRIDE_0 + static_cast<u64>(head) * SEISMIC_RESULT_0_STRIDE_1;
    const u64 lo_at = out_at + static_cast<u64>(lo) * SEISMIC_RESULT_0_STRIDE_2;
    const u64 hi_at = out_at + static_cast<u64>(hi) * SEISMIC_RESULT_0_STRIDE_2;
    const float low = gated.at(lo_at);
    const float high = gated.at(hi_at);
    gated.put(lo_at, low * cosine + high * sine);
    gated.put(hi_at, high * cosine - low * sine);
}

}  // namespace prefill
}  // namespace attention
