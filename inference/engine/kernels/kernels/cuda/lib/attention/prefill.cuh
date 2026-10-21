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
// (`stage_codes`, `decode_codes`). The products are then the dense ones over
// the decoded history (the Metal and Vulkan staging).
//
// The rows policy supplies the query tile and fresh tiles and stores the
// partition's result. Prefill rows (`PrefillRows`): L1 `prepare`, one warp
// per (row, query or kv head), puts the queries (norm_rotary, rounded to the
// activation element) and the batch's prepared keys and values in scratch as
// the policy's 16-bit MMA operands (`Operands`; scores are scaled into the
// exp2 domain after Q K^T) and appends the row's K/V at its destination; L2
// `attend` copies them and stores its gated output directly when one
// partition serves the tile, else partials that L3 `merge_partitions`
// combines. Decode rows (`DecodeRows`): the block prepares its own query tile
// and fresh tiles, partition 0 appends the tile's K/V, and every partition
// (an empty one too) publishes its partial in the decode layout for the
// decode merge (`attention::decode_gate`).

#include "attention.cuh"

namespace attention {
namespace prefill {

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
#ifdef ATTENTION_AFFINE_HISTORY
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
    }, threadIdx.x, MMA_THREADS);
    if (piece + 1 < PIECES) return;
    stage_slab<F32_ACTIVATION, WINDOW_CHUNKS>(v_tile, KEYS, [&](int r) -> const u8 * {
        const int token = first + r;
        return token < limit ? history.value_vector(token, kv) + column0 * Act::bytes : nullptr;
    }, threadIdx.x, MMA_THREADS);
}

// ---------------------------------------------------------------------------
#ifdef ATTENTION_AFFINE_HISTORY
// Affine history tiles, produced by the producer warps.

// Producer threads (the entry's ATTENTION_PRODUCER_WARPS warps; 0 for dense
// history, which has none), the 16-byte code pieces of a key and a value row,
// and the codes of one piece (all in one group). Fewer producer warps than MMA
// warps leave the MMA warps more registers (a block's registers split evenly
// over its threads).
constexpr int PRODUCERS = ATTENTION_PRODUCER_WARPS * 32;
constexpr int PRODUCER_STRIDE = PRODUCERS > 0 ? PRODUCERS : 1;
// Key code pieces of one staged key piece.
constexpr int KEY_PIECES = PIECE * KEY_BITS / 128;
// Value pieces of one output window.
constexpr int VALUE_PIECES = WINDOW * VALUE_BITS / 128;
constexpr int KEY_ITEMS = (KEYS * KEY_PIECES + PRODUCER_STRIDE - 1) / PRODUCER_STRIDE;
constexpr int VALUE_ITEMS = (KEYS * VALUE_PIECES + PRODUCER_STRIDE - 1) / PRODUCER_STRIDE;
constexpr int KEY_PIECE_CODES = 128 / KEY_BITS;
constexpr int VALUE_PIECE_CODES = 128 / VALUE_BITS;
static_assert(GROUP % KEY_PIECE_CODES == 0 && GROUP % VALUE_PIECE_CODES == 0,
              "a code piece lies in one group");
// Codes decode in packed F16 (the affine operands' element): code c sits in
// the mantissa of the F16 1024 (0x6400 | c, exact), 1024 is subtracted
// exactly, and one F16x2 fma applies the group's (scale, zero), rounding once
// (the F32 fma rounded to F16 gives the same operands on the goldens), two
// dimensions per instruction.
__device__ __forceinline__ u32 half_decode(u32 biased, u32 scales, u32 zeros) {
    u32 code, value;
    asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(code) : "r"(biased), "r"(0x64006400u));
    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(value) : "r"(code), "r"(scales), "r"(zeros));
    return value;
}

// Eight 8-bit codes (two words) decoded with their group's (scale, zero)
// `pair` into one operand chunk.
__device__ __forceinline__ uint4 key_chunk(u32 low, u32 high, u32 pair) {
    const u32 scales = __byte_perm(pair, 0, 0x1010);
    const u32 zeros = __byte_perm(pair, 0, 0x3232);
    return make_uint4(half_decode(__byte_perm(low, 0x64646464u, 0x4140), scales, zeros),
                      half_decode(__byte_perm(low, 0x64646464u, 0x4342), scales, zeros),
                      half_decode(__byte_perm(high, 0x64646464u, 0x4140), scales, zeros),
                      half_decode(__byte_perm(high, 0x64646464u, 0x4342), scales, zeros));
}

// Eight 4-bit codes (one word) decoded with their group's pair into one
// operand chunk: dimension pair j is byte j's low and high nibbles.
__device__ __forceinline__ uint4 value_chunk(u32 word, u32 pair) {
    const u32 scales = __byte_perm(pair, 0, 0x1010);
    const u32 zeros = __byte_perm(pair, 0, 0x3232);
    // Byte j of `low` holds nibble 2j, of `high` nibble 2j + 1; pair j
    // gathers both into bytes 0 and 1, then under the F16 1024's exponent.
    const u32 low = word & 0x0F0F0F0Fu;
    const u32 high = (word >> 4) & 0x0F0F0F0Fu;
    u32 biased[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
        biased[j] = __byte_perm(__byte_perm(low, high, static_cast<u32>(j) | (static_cast<u32>(4 + j) << 4)),
                                0x64646464u, 0x4140);
    return make_uint4(half_decode(biased[0], scales, zeros), half_decode(biased[1], scales, zeros),
                      half_decode(biased[2], scales, zeros), half_decode(biased[3], scales, zeros));
}

// A producer thread's codes and (scale, zero) pairs of one affine unit, held
// in registers between their loads and their decode.
struct ProducedCodes {
    uint4 key_bits[KEY_ITEMS];
    u32 key_pairs[KEY_ITEMS];
    uint4 value_bits[VALUE_ITEMS];
    u32 value_pairs[VALUE_ITEMS];
};

// Producer thread `thread`'s loads of the affine tile of tokens [first, first
// + KEYS): key piece `piece` and, with the last piece, the output window's
// values from column0. Rows at or past `limit` get zero codes and zero pairs,
// so they decode to exact zeros. Every load is issued before any is used.
__device__ __forceinline__ ProducedCodes fetch_produced(const AffineHistory &history, int first, int limit, int kv,
                                                        int piece, int column0, int thread) {
    const int key_piece0 = piece * KEY_PIECES;
    const int value_piece0 = column0 * VALUE_BITS / 128;
    const bool values = piece + 1 == PIECES;
    ProducedCodes codes;
#pragma unroll
    for (int k = 0; k < KEY_ITEMS; ++k) {
        const int index = thread + k * PRODUCERS;
        const int token = first + index / KEY_PIECES;
        const int code = key_piece0 + index % KEY_PIECES;
        codes.key_bits[k] = make_uint4(0, 0, 0, 0);
        codes.key_pairs[k] = 0u;
        if (index < KEYS * KEY_PIECES && token < limit) {
            const AffineHistory::Vectors v = history.vectors(token, kv);
            codes.key_bits[k] = seismic_ld_nc_v4(v.key_codes + code * 4);
            codes.key_pairs[k] = seismic_ld_nc_u32(v.key_pairs + code * KEY_PIECE_CODES / GROUP);
        }
    }
#pragma unroll
    for (int k = 0; k < VALUE_ITEMS; ++k) {
        const int index = thread + k * PRODUCERS;
        const int token = first + index / VALUE_PIECES;
        const int code = value_piece0 + index % VALUE_PIECES;
        codes.value_bits[k] = make_uint4(0, 0, 0, 0);
        codes.value_pairs[k] = 0u;
        if (values && index < KEYS * VALUE_PIECES && token < limit) {
            const AffineHistory::Vectors v = history.vectors(token, kv);
            codes.value_bits[k] = seismic_ld_nc_v4(v.value_codes + code * 4);
            codes.value_pairs[k] = seismic_ld_nc_u32(v.value_pairs + code * VALUE_PIECE_CODES / GROUP);
        }
    }
    return codes;
}

// Producer thread `thread`'s fetched codes decoded into the swizzled K operand
// tile (key piece `piece`) and, with the last piece, the V operand tile, each
// code piece with its group's (scale, zero) pair.
__device__ __forceinline__ void store_produced(const ProducedCodes &codes, u8 *k_tile, u8 *v_tile, int piece,
                                               int thread) {
    const bool values = piece + 1 == PIECES;
#pragma unroll
    for (int k = 0; k < KEY_ITEMS; ++k) {
        const int index = thread + k * PRODUCERS;
        if (index < KEYS * KEY_PIECES) {
            // A 16-byte key piece: 16 codes, two operand chunks.
            const int r = index / KEY_PIECES;
            const int chunk = (index % KEY_PIECES) * 2;
            const u32 pair = codes.key_pairs[k];
            *reinterpret_cast<uint4 *>(k_tile + swizzled<PIECE_CHUNKS>(r, chunk)) =
                key_chunk(codes.key_bits[k].x, codes.key_bits[k].y, pair);
            *reinterpret_cast<uint4 *>(k_tile + swizzled<PIECE_CHUNKS>(r, chunk + 1)) =
                key_chunk(codes.key_bits[k].z, codes.key_bits[k].w, pair);
        }
    }
#pragma unroll
    for (int k = 0; k < VALUE_ITEMS; ++k) {
        const int index = thread + k * PRODUCERS;
        if (values && index < KEYS * VALUE_PIECES) {
            // A 16-byte value piece: 32 codes, four operand chunks.
            const int r = index / VALUE_PIECES;
            const int chunk = (index % VALUE_PIECES) * 4;
            const u32 pair = codes.value_pairs[k];
            *reinterpret_cast<uint4 *>(v_tile + swizzled<WINDOW_CHUNKS>(r, chunk)) =
                value_chunk(codes.value_bits[k].x, pair);
            *reinterpret_cast<uint4 *>(v_tile + swizzled<WINDOW_CHUNKS>(r, chunk + 1)) =
                value_chunk(codes.value_bits[k].y, pair);
            *reinterpret_cast<uint4 *>(v_tile + swizzled<WINDOW_CHUNKS>(r, chunk + 2)) =
                value_chunk(codes.value_bits[k].z, pair);
            *reinterpret_cast<uint4 *>(v_tile + swizzled<WINDOW_CHUNKS>(r, chunk + 3)) =
                value_chunk(codes.value_bits[k].w, pair);
        }
    }
}

// ---------------------------------------------------------------------------
// Affine history decoded by the MMA warps themselves (no producer warps): a
// unit's code rows and (scale, zero) pairs are copied with `cp.async` into a
// code stage beside its operand stage, as dense history is copied, and
// decoded into the operand stage once they land. The MMA warps keep the whole
// register file (dense's budget); the decode is a few ALU instructions per
// operand beside the tensor-core products. Heads of 128 to 256 columns: every
// code and pair row is whole 16-byte chunks.
//
// A code stage: key codes [KEYS][PIECE] u8, key pairs [KEYS][PIECE / GROUP]
// u32, value codes [KEYS][WINDOW / 2] u8, value pairs [KEYS][WINDOW / GROUP]
// u32.
constexpr int KEY_PAIR_CHUNKS = PIECE / GROUP * 4 / 16;
constexpr int VALUE_PAIR_CHUNKS = WINDOW / GROUP * 4 / 16;
constexpr int KEY_ROW_CHUNKS = KEY_PIECES + KEY_PAIR_CHUNKS;
constexpr int CODE_ROW_CHUNKS = KEY_ROW_CHUNKS + VALUE_PIECES + VALUE_PAIR_CHUNKS;
constexpr int CODE_STAGE_BYTES = KEYS * CODE_ROW_CHUNKS * 16;
constexpr int KEY_PAIRS_AT = KEYS * KEY_PIECES * 16;
constexpr int VALUE_CODES_AT = KEY_PAIRS_AT + KEYS * KEY_PAIR_CHUNKS * 16;
constexpr int VALUE_PAIRS_AT = VALUE_CODES_AT + KEYS * VALUE_PIECES * 16;

// Copy the affine unit of tokens [first, first + KEYS) (key piece `piece`,
// and with the last piece the output window's values from column0) into code
// stage `codes` by `threads` threads of which this is `thread`: threads / KEYS
// consecutive threads per row, so each locates its row's slab once and a
// row's chunks are copied contiguously. Rows at or past `limit` are zero
// codes and pairs (they decode to exact zeros).
__device__ __forceinline__ void stage_codes(const AffineHistory &history, u8 *codes, int first,
                                            int limit, int kv, int piece, int column0, int thread,
                                            int threads) {
    const int per_row = threads / KEYS;
    const int r = thread / per_row;
    const int token = first + r;
    const bool live = token < limit;
    // A dead row still names a valid source (the tile's first row).
    const AffineHistory::Vectors v = history.vectors(live ? token : first, kv);
    const u32 bytes = live ? 16u : 0u;
    const u8 *key_codes = reinterpret_cast<const u8 *>(v.key_codes) + piece * PIECE;
    const u8 *key_pairs = reinterpret_cast<const u8 *>(v.key_pairs) + piece * PIECE / GROUP * 4;
    const u8 *value_codes = reinterpret_cast<const u8 *>(v.value_codes) + column0 / 2;
    const u8 *value_pairs = reinterpret_cast<const u8 *>(v.value_pairs) + column0 / GROUP * 4;
    const int chunks = piece + 1 == PIECES ? CODE_ROW_CHUNKS : KEY_ROW_CHUNKS;
    for (int c = thread % per_row; c < chunks; c += per_row) {
        u8 *to;
        const u8 *from;
        if (c < KEY_PIECES) {
            to = codes + (r * KEY_PIECES + c) * 16;
            from = key_codes + c * 16;
        } else if (c < KEY_ROW_CHUNKS) {
            const int d = c - KEY_PIECES;
            to = codes + KEY_PAIRS_AT + (r * KEY_PAIR_CHUNKS + d) * 16;
            from = key_pairs + d * 16;
        } else if (c < KEY_ROW_CHUNKS + VALUE_PIECES) {
            const int d = c - KEY_ROW_CHUNKS;
            to = codes + VALUE_CODES_AT + (r * VALUE_PIECES + d) * 16;
            from = value_codes + d * 16;
        } else {
            const int d = c - KEY_ROW_CHUNKS - VALUE_PIECES;
            to = codes + VALUE_PAIRS_AT + (r * VALUE_PAIR_CHUNKS + d) * 16;
            from = value_pairs + d * 16;
        }
        seismic_cp_async_16_zfill(to, from, bytes);
    }
}

// Decode code stage `codes` into the swizzled K operand tile and, with the
// last piece, the V operand tile, by `threads` threads of which this is
// `thread`.
__device__ __forceinline__ void decode_codes(const u8 *codes, u8 *k_tile, u8 *v_tile, bool values,
                                             int thread, int threads) {
    for (int index = thread; index < KEYS * KEY_PIECES; index += threads) {
        // A 16-byte key piece: 16 codes, two operand chunks.
        const int r = index / KEY_PIECES;
        const int p = index % KEY_PIECES;
        const uint4 bits = *reinterpret_cast<const uint4 *>(codes + index * 16);
        const u32 pair = *reinterpret_cast<const u32 *>(
            codes + KEY_PAIRS_AT + r * KEY_PAIR_CHUNKS * 16 + p * KEY_PIECE_CODES / GROUP * 4);
        *reinterpret_cast<uint4 *>(k_tile + swizzled<PIECE_CHUNKS>(r, 2 * p)) =
            key_chunk(bits.x, bits.y, pair);
        *reinterpret_cast<uint4 *>(k_tile + swizzled<PIECE_CHUNKS>(r, 2 * p + 1)) =
            key_chunk(bits.z, bits.w, pair);
    }
    if (!values) return;
    for (int index = thread; index < KEYS * VALUE_PIECES; index += threads) {
        // A 16-byte value piece: 32 codes, four operand chunks.
        const int r = index / VALUE_PIECES;
        const int p = index % VALUE_PIECES;
        const uint4 bits = *reinterpret_cast<const uint4 *>(codes + VALUE_CODES_AT + index * 16);
        const u32 pair = *reinterpret_cast<const u32 *>(
            codes + VALUE_PAIRS_AT + r * VALUE_PAIR_CHUNKS * 16 + p * VALUE_PIECE_CODES / GROUP * 4);
        *reinterpret_cast<uint4 *>(v_tile + swizzled<WINDOW_CHUNKS>(r, 4 * p)) =
            value_chunk(bits.x, pair);
        *reinterpret_cast<uint4 *>(v_tile + swizzled<WINDOW_CHUNKS>(r, 4 * p + 1)) =
            value_chunk(bits.y, pair);
        *reinterpret_cast<uint4 *>(v_tile + swizzled<WINDOW_CHUNKS>(r, 4 * p + 2)) =
            value_chunk(bits.z, pair);
        *reinterpret_cast<uint4 *>(v_tile + swizzled<WINDOW_CHUNKS>(r, 4 * p + 3)) =
            value_chunk(bits.w, pair);
    }
}

#endif

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
    // Its place in a WorkList's list (unused otherwise).
    int index;
};

// A block of `attend`: query tile `tile` (tokens [tile * QT, (tile + 1) *
// QT)), kv head `kv`, and key partition `partition` of `parts`. The query
// tile's key tiles (each span's union interval in KEYS steps, spans then
// fresh) split into runs of at least the rows policy's MIN_TILES over the
// partitions; partitions past the last run are empty.
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
struct Split {
    float *partials;    // [parts][M][KV * G][W]
    float *statistics;  // [parts][M][KV * G][2]
    u32 *counts;        // [query tiles]
};

// A prefill launch's WorkList selection (`indexer_select`'s per-row block
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
// them.
struct PrefillRows {
    static constexpr int MIN_TILES = 16;
    const u8 *queries;
    const u8 *keys;
    const u8 *values;
    u8 *gated;
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

    // The fresh tile of tokens [first, first + KEYS) (zero at or past
    // `limit`): the keys' piece, and with the last piece the values' window
    // from column0, by `threads` threads of which this is `thread`.
    template <class History>
    __device__ __forceinline__ void stage_fresh(const Inputs &, u8 *k_tile, u8 *v_tile, int first,
                                                int limit, int kv, int piece, int column0,
                                                int thread, int threads, float *) const {
        auto row = [&](int r) -> long long {
            const int token = first + r;
            return token < limit ? (static_cast<long long>(token) * KV + kv) * W : -1;
        };
        stage<false, PIECE_CHUNKS>(k_tile, keys, KEYS, [&](int r) -> long long {
            const long long at = row(r);
            return at >= 0 ? at + piece * PIECE : -1;
        }, thread, threads);
        if (piece + 1 < PIECES) return;
        stage<false, WINDOW_CHUNKS>(v_tile, values, KEYS, [&](int r) -> long long {
            const long long at = row(r);
            return at >= 0 ? at + column0 : -1;
        }, thread, threads);
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
                element::put<Act>(gated, out_at + column * SEISMIC_RESULT_0_STRIDE_2,
                                  attention::gated(in, token, query_head, column,
                                                   o[n][2 * half + e] / whole));
            }
        }
    }
};

// Decode rows: the block prepares its query tile and the fresh tiles itself
// (a warp per row, through a [WARPS][W] F32 exchange) and publishes every
// partition in the decode layout (`attention::publish`): slot = (row * KV * G
// + query head) * PARTS + partition, an empty partition with denominator 0.
template <int PARTS>
struct DecodeRows {
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

    // The fresh tile of tokens [first, first + KEYS) (zero at or past
    // `limit`), prepared by the warps of `threads` threads of which this is
    // `thread`, a warp per token.
    template <class History>
    __device__ __forceinline__ void stage_fresh(const Inputs &in, u8 *k_tile, u8 *v_tile, int first,
                                                int limit, int kv, int piece, int column0,
                                                int thread, int threads, float *exchange) const {
        typedef Operands<History> Ops;
        const int lane = thread % 32;
        float *own = exchange + (thread / 32) * W;
        for (int r = thread / 32; r < KEYS; r += threads / 32) {
            const int token = first + r;
            float k[DPL];
            float v[DPL];
            if (token < limit) {
                prepared_key(in, token, kv, k, own, lane);
                fresh_value(in, token, kv, v, lane);
            } else {
#pragma unroll
                for (int d = 0; d < DPL; ++d) k[d] = v[d] = 0.0f;
            }
            put_row<Ops, PIECE_CHUNKS>(k_tile, r, piece * PIECE, k, lane);
            if (piece + 1 == PIECES) put_row<Ops, WINDOW_CHUNKS>(v_tile, r, column0, v, lane);
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

// Prefill L1 over `M` rows: 8 warps per block (launch with 256 threads). Queries,
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

// Block `block`: WARPS MMA warps, plus WARPS producer warps for affine
// history (launch with WARPS * COLUMNS * 32 threads for dense history, twice that for
// affine). Shared memory: the query tile [ROWS][W], STAGES operand stages each K
// [KEYS][PIECE] then V [KEYS][WINDOW] 16-bit, the span table [R + 1][4], then
// for decode rows the exchange [WARPS][W] F32. The MMA warps make one pass
// over the partition's key tiles per output window; a key tile is PIECES
// staged units, the last carrying the window's values.
#ifdef ATTENTION_HISTORY_STAGING
// Each row is looked up separately. An operand tile may cross compact-page
// or visibility gaps without changing its original span-based KEY grouping.
__device__ __forceinline__ void stage_history(const ListedF16History &history, u8 *k_tile,
                                               u8 *v_tile, int first, int limit, int kv, int piece,
                                               int column0) {
    stage_slab<false, PIECE_CHUNKS>(k_tile, KEYS, [&](int r) -> const u8 * {
        const int token = first + r;
        const u8 *row = token < limit ? history.key_vector(token, kv) : nullptr;
        return row ? row + piece * PIECE * 2 : nullptr;
    }, threadIdx.x, MMA_THREADS);
    if (piece + 1 < PIECES) return;
    stage_slab<false, WINDOW_CHUNKS>(v_tile, KEYS, [&](int r) -> const u8 * {
        const int token = first + r;
        const u8 *row = token < limit ? history.value_vector(token, kv) : nullptr;
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

template <class History, class Rows>
__device__ __forceinline__ void attend(const Inputs &in, const History &history, const Rows &rows,
                                       Block block, const Selection &selection = Selection{}) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    typedef Operands<History> Ops;
    constexpr bool CODED = History::CODED;
    // Affine history is decoded by producer warps, or without them by the
    // MMA warps from code stages (`stage_codes`, `decode_codes`).
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
    const int kv = block.kv;
    const long long first_token = static_cast<long long>(block.tile) * QT;
    const long long rows_total = static_cast<long long>(SEISMIC_DIM_M);
    const int spans = static_cast<int>(SEISMIC_DIM_R);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int g = lane / 4;
    const int t = lane % 4;
    const float query_scale = in.scale * LOG2E;

    extern __shared__ __align__(16) u8 shared[];
    // With Q_REGISTERS the query tile is staged in the K/V stages' region and
    // read into registers before the first K/V tile arrives, so the stages
    // and the query tile share the region.
    constexpr int Q_BYTES = ROWS * W * 2;
    constexpr int STAGE_BYTES = KEYS * (PIECE + WINDOW) * 2;
    // Inline decoding adds STAGES code stages after the operand stages.
    // Split scores decode inline from a single code stage: a unit's codes
    // are copied once the previous unit's are decoded, while its products run.
    constexpr int CODE_STAGES = SPLIT_SCORES ? 1 : STAGES;
#ifdef ATTENTION_AFFINE_HISTORY
    constexpr int STAGED_BYTES = STAGES * STAGE_BYTES + (INLINE ? CODE_STAGES * CODE_STAGE_BYTES : 0);
#else
    constexpr int STAGED_BYTES = STAGES * STAGE_BYTES;
#endif
    constexpr int REGION = Q_REGISTERS ? (Q_BYTES > STAGED_BYTES ? Q_BYTES : STAGED_BYTES)
                                       : Q_BYTES + STAGED_BYTES;
    u8 *q_tile = shared;
    u8 *kv_tiles = shared + (Q_REGISTERS ? 0 : Q_BYTES);
    auto k_tile = [&](int b) { return kv_tiles + b * STAGE_BYTES; };
    auto v_tile = [&](int b) { return kv_tiles + b * STAGE_BYTES + KEYS * PIECE * 2; };
#ifdef ATTENTION_AFFINE_HISTORY
    auto code_tile = [&](int b) { return kv_tiles + STAGES * STAGE_BYTES + (b % CODE_STAGES) * CODE_STAGE_BYTES; };
#endif
    int *table = reinterpret_cast<int *>(shared + REGION);  // [R + 1][4]
    float *exchange = reinterpret_cast<float *>(table + (spans + 1) * 4);  // [WARPS * COLUMNS][W]

    if (warp < WARPS * COLUMNS)
        rows.stage_queries(in, history, q_tile, exchange, block, first_token, warp, lane);

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

    // The warp's query fragments (row block warp % WARPS, every 16-column
    // step of the head), when they stay in registers.
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
    }

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
    // A WorkList's key tiles are its query tile's listed ones.
    const bool listing = selection.words > 0;
    int total_tiles = 0;
    for (int index = 0; index <= spans; ++index) total_tiles += span_tiles(index);
    if (listing) total_tiles = static_cast<int>(selection.listed[block.tile]);
    const int parts = block.parts;
    const int per = max(Rows::MIN_TILES, (total_tiles + parts - 1) / parts);
    const int active = max(1, (total_tiles + per - 1) / per);
    const int partition = block.partition;
    if (partition >= active) {
        seismic_cp_async_wait<0>();
        rows.vacant(in, block, first_token);
        return;
    }
    rows.partitioned(block, active);
    const int tiles_lo = partition * per;
    const int count = max(0, min(per, total_tiles - tiles_lo));
    // Tile `index` of the query tile's sequence.
    auto nth = [&](int index) {
        if (listing) {
            if (index >= total_tiles) return Tile{spans + 1, 0, index};
            const int *entry = selection.lists + (static_cast<long long>(block.tile) * selection.capacity + index) * 2;
            return Tile{entry[0], entry[1], index};
        }
        for (int span = 0; span <= spans; ++span) {
            const int n = span_tiles(span);
            if (index < n) return Tile{span, table[span * 4 + 0] + index * KEYS};
            index -= n;
        }
        return Tile{spans + 1, 0};
    };
    // Fresh tile `tile` into operand stage b (the keys' piece, and with the
    // last piece the values' window from column0), by `threads` threads of
    // which this is `thread`.
    auto stage_fresh = [&](Tile tile, int buffer, int piece, int column0, int thread, int threads) {
        rows.template stage_fresh<History>(in, k_tile(buffer), v_tile(buffer), tile.first,
                                           table[tile.span * 4 + 1], kv, piece, column0, thread,
                                           threads, exchange);
    };
    // The unit after (tile, piece): the tile's next key piece, else the next
    // tile's first.
    auto advance = [&](Tile &tile, int &piece) {
        if (++piece == PIECES) {
            piece = 0;
            tile = listing ? nth(tile.index + 1) : settle(Tile{tile.span, tile.first + KEYS});
        }
    };
    // Staged units per output window.
    const int units = count * PIECES;

#ifdef ATTENTION_AFFINE_HISTORY
    if constexpr (PRODUCED) {
        if (warp >= WARPS * COLUMNS) {
            // Producer warps: per output window, the partition's units in
            // order; staged unit i (counted over every window) goes into stage
            // i % STAGES once its previous occupant (unit i - STAGES) is
            // consumed.
            // A history unit's codes are loaded one unit ahead: unit j + 1's
            // loads are in flight while unit j waits for its stage and
            // decodes.
            const int thread = threadIdx.x - MMA_THREADS;
            for (int window = 0; window < WINDOWS; ++window) {
                const int column0 = window * WINDOW;
                auto fetch = [&](Tile tile, int piece) {
                    return fetch_produced(history, tile.first, table[tile.span * 4 + 1], kv, piece, column0, thread);
                };
                Tile tile = nth(tiles_lo);
                int piece = 0;
                ProducedCodes ahead;
                if (units > 0 && tile.span < spans) ahead = fetch(tile, piece);
                for (int j = 0; j < units; ++j) {
                    const int i = window * units + j;
                    const int b = i % STAGES;
                    Tile next = tile;
                    int next_piece = piece;
                    advance(next, next_piece);
                    const ProducedCodes current = ahead;
                    if (j + 1 < units && next.span < spans) ahead = fetch(next, next_piece);
                    if (i >= STAGES) named_sync(EMPTY + b, THREADS);
                    if (tile.span < spans) {
                        store_produced(current, k_tile(b), v_tile(b), piece, thread);
                    } else {
                        stage_fresh(tile, b, piece, column0, thread, PRODUCERS);
                        seismic_cp_async_commit();
                        seismic_cp_async_wait<0>();
                    }
                    __threadfence_block();
                    named_arrive(FULL + b, THREADS);
                    tile = next;
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
    const int row_block = warp % WARPS;
    const int n0 = (warp / WARPS) * OUT;
    const long long token_a = first_token + (row_block * 16 + g) / G;
    const long long token_b = first_token + (row_block * 16 + g + 8) / G;
    const bool valid_a = row_block * 16 + g < QT * G && token_a < rows_total;
    const bool valid_b = row_block * 16 + g + 8 < QT * G && token_b < rows_total;
    // A block of padding rows only (the tile's last rows, or with decode rows
    // most blocks) stages and synchronizes but skips the products.
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
    // columns; after the tile's last piece, the softmax and O += P V.
    auto absorb_unit = [&](Tile current, int piece, const u8 *k_base, const u8 *v_base) {
        if (piece == 0) {
#pragma unroll
            for (int n = 0; n < KEYS / 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
        }
        if (!SPLIT_SCORES || piece == span_piece) {
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
            float *group = exchange + ((parity * WARPS + row_block) * COLUMNS) * 32 * SPLIT_LANE_FLOATS;
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

        if (current.span != cached_span) {
            cached_span = current.span;
            interval_a = valid_a ? span(in, token_a, current.span, spans) : Span{0, 0};
            interval_b = valid_b ? span(in, token_b, current.span, spans) : Span{0, 0};
            raw_a = interval_a;
            raw_b = interval_b;
        }
        const bool full = !listing && current.first >= table[current.span * 4 + 2] &&
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
                    if (key < interval.lo || key >= interval.hi) {
                        s[n][e] = NEG_INF;
                    } else if (listing &&
                               !selection.reads(e < 2 ? token_a : token_b, current.span, spans, key,
                                                (e < 2 ? raw_a : raw_b).lo)) {
                        s[n][e] = NEG_INF;
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

        Tile current = nth(tiles_lo);
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
                if (computes) absorb_unit(current, piece, k_tile(b), v_tile(b));
                if (i + STAGES < staged) named_arrive(EMPTY + b, THREADS);
                advance(current, piece);
            }
        } else
#endif
        {
            // A history unit goes to its operand stage, or with inline
            // decoding as codes to its code stage.
            auto issue = [&](Tile tile, int tile_piece, int buffer) {
                if (tile.span < spans) {
#ifdef ATTENTION_AFFINE_HISTORY
                    if constexpr (INLINE) {
                        stage_codes(history, code_tile(buffer), tile.first, table[tile.span * 4 + 1],
                                    kv, tile_piece, column0, threadIdx.x, MMA_THREADS);
                    } else
#endif
                    {
                        stage_history(history, k_tile(buffer), v_tile(buffer), tile.first,
                                      table[tile.span * 4 + 1], kv, tile_piece, column0);
                    }
                } else {
                    stage_fresh(tile, buffer, tile_piece, column0, threadIdx.x, MMA_THREADS);
                }
            };
            Tile fetch = current;
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
                    if (current.span < spans) {
                        decode_codes(code_tile(0), k_tile(i % STAGES), v_tile(i % STAGES), true,
                                     threadIdx.x, MMA_THREADS);
                    }
                    __syncthreads();
                    if (i + 1 < units) {
                        issue(fetch, fetch_piece, (i + 1) % STAGES);
                        advance(fetch, fetch_piece);
                    }
                    seismic_cp_async_commit();
                    if (computes) absorb_unit(current, piece, k_tile(i % STAGES), v_tile(i % STAGES));
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
                        if (current.span < spans) {
                            decode_codes(code_tile(i % STAGES), k_tile(i % STAGES), v_tile(i % STAGES),
                                         piece + 1 == PIECES, threadIdx.x, MMA_THREADS);
                            __syncthreads();
                        }
                    }
#endif

                    if (computes) absorb_unit(current, piece, k_tile(i % STAGES), v_tile(i % STAGES));
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
        const float maximum =
            merge(split.statistics + slot * 2, static_cast<u64>(rows_total) * heads * 2,
                  split.partials + slot * W + column, static_cast<u64>(rows_total) * heads * W, count,
                  denominator, accumulated);
        element::put<Act>(gated,
                          static_cast<u64>(token) * SEISMIC_RESULT_0_STRIDE_0 +
                              static_cast<u64>(query_head) * SEISMIC_RESULT_0_STRIDE_1 +
                              static_cast<u64>(column) * SEISMIC_RESULT_0_STRIDE_2,
                          attention::gated(in, token, query_head, column,
                                           accumulated / normalizer(in, query_head, maximum, denominator)));
    }
}

}  // namespace prefill
}  // namespace attention
