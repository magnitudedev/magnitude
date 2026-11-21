// history shared attention mechanisms; included at the original declaration point.
// ---------------------------------------------------------------------------
// History planes. Each policy appends a row's prepared key and raw value at
// its destination (one warp, the caller skips rows without one) and locates a
// (history row, kv head) vector in its planes. An entry source builds its
// policy from its own ABI (`HISTORY()`): a header may name only macros every
// including entry generates.

// Read-only view of dense history: key and value planes [T, KV, W] of the
// activation element.
struct DenseReads {
    static constexpr bool CODED = false;
    const u32 *key;
    const u32 *value;
    u64 key_row, key_head, value_row, value_head;
    u64 slab_rows;

    // The end of the run of rows [token, hi) within history row `token`'s
    // slab.
    __device__ __forceinline__ int slab_end(int token, int hi) const {
        return static_cast<int>(min(static_cast<u64>(hi), (static_cast<u64>(token) / slab_rows + 1) * slab_rows));
    }

    // Byte addresses of a (history row, kv head) vector, for reading.
    __device__ __forceinline__ const u8 *key_vector(int token, int kv_head) const {
        return slab::row(key, static_cast<u64>(token), slab_rows, key_row * Act::bytes) +
               static_cast<u64>(kv_head) * key_head * Act::bytes;
    }
    // A stored-key history's values are its key plane (it has no value plane).
    __device__ __forceinline__ const u8 *value_vector(int token, int kv_head) const {
        if constexpr (STORED_KEY) return key_vector(token, kv_head);
        return slab::row(value, static_cast<u64>(token), slab_rows, value_row * Act::bytes) +
               static_cast<u64>(kv_head) * value_head * Act::bytes;
    }
    // A slab's first (history row, kv head) vectors; later rows of the slab
    // are `key_row` / `value_row` elements past them.
    __device__ __forceinline__ const u8 *key_slab(u32 slab_index, int kv_head) const {
        return slab::region(key, slab_index) + static_cast<u64>(kv_head) * key_head * Act::bytes;
    }
    __device__ __forceinline__ const u8 *value_slab(u32 slab_index, int kv_head) const {
        if constexpr (STORED_KEY) return key_slab(slab_index, kv_head);
        return slab::region(value, slab_index) + static_cast<u64>(kv_head) * value_head * Act::bytes;
    }
};

// Dense history: key and value planes [T, KV, W] of the activation element.
struct DenseHistory {
    static constexpr bool CODED = false;
    element::SlabOut<Act> key;
    element::SlabOut<Act> value;
    u64 key_row, key_head, value_row, value_head;
    u64 slab_rows;

    __device__ __forceinline__ DenseReads reads() const {
        return DenseReads{key.reads(), value.reads(), key_row, key_head, value_row, value_head, slab_rows};
    }
    __device__ __forceinline__ operator DenseReads() const { return reads(); }

    // The end of the run of rows [token, hi) within history row `token`'s
    // slab.
    __device__ __forceinline__ int slab_end(int token, int hi) const { return reads().slab_end(token, hi); }

    // The handles of a (history row, kv head) vector.
    __device__ __forceinline__ element::Out<Act> key_out(int token, int kv_head) const {
        return key.row(static_cast<u64>(token), slab_rows, key_row).offset(static_cast<u64>(kv_head) * key_head);
    }
    // A stored-key history's values are its key plane (it has no value plane).
    __device__ __forceinline__ element::Out<Act> value_out(int token, int kv_head) const {
        if constexpr (STORED_KEY) return key_out(token, kv_head);
        return value.row(static_cast<u64>(token), slab_rows, value_row)
            .offset(static_cast<u64>(kv_head) * value_head);
    }
    // Byte addresses of a (history row, kv head) vector, for reading.
    __device__ __forceinline__ const u8 *key_vector(int token, int kv_head) const {
        return reads().key_vector(token, kv_head);
    }
    __device__ __forceinline__ const u8 *value_vector(int token, int kv_head) const {
        return reads().value_vector(token, kv_head);
    }
    __device__ __forceinline__ const u8 *key_slab(u32 slab_index, int kv_head) const {
        return reads().key_slab(slab_index, kv_head);
    }
    __device__ __forceinline__ const u8 *value_slab(u32 slab_index, int kv_head) const {
        return reads().value_slab(slab_index, kv_head);
    }

    __device__ __forceinline__ void append(int destination, int kv_head, const float (&k)[DPL],
                                           const float (&v)[DPL], int lane) const {
        const element::Out<Act> k_row = key_out(destination, kv_head);
        const u64 offset = static_cast<u64>(lane) * DPL;
        auto store = [&](const element::Out<Act> &row, const float (&values)[DPL]) {
            if constexpr (DPL % 2 == 0) {
                constexpr u64 align =
                    (Act::bytes == 4 && DPL % 4 == 0) ? 16u : (DPL % 4 == 0 ? 8u : 2u * Act::bytes);
                if ((reinterpret_cast<u64>(row.stored()) & (align - 1u)) == 0) {
                    row.put_span(offset, values);
                    return;
                }
            }
#pragma unroll
            for (int d = 0; d < DPL; ++d) row.put(offset + d, values[d]);
        };
        store(k_row, k);
        if constexpr (!STORED_KEY) store(value_out(destination, kv_head), v);
    }
};

#ifdef ATTENTION_AFFINE_HISTORY
// Affine K8/V4 history (the `affine_k8_uniform_v4` codec,
// `attention_*_k8v4`). A (history row, kv head) vector is a code row of
// W * B / 32 u32 words (code i at bits B * (i % (32 / B)) of word
// i / (32 / B)) plus one F16 (scale, zero) pair per group of GROUP
// consecutive dimensions, pairs in group order; decoded value =
// code * scale + zero with its group's pair. Keys use B = 8, values B = 4
// (`attention.seismic`'s AFFINE_KEY_BITS and AFFINE_VALUE_BITS). A lane's DPL
// dimensions lie in one group, shared by PAIR_LANES lanes.
constexpr int KEY_BITS = static_cast<int>(SEISMIC_CONST_AFFINE_KEY_BITS);
constexpr int VALUE_BITS = static_cast<int>(SEISMIC_CONST_AFFINE_VALUE_BITS);
// Dimensions per (scale, zero) pair: the codec's group (`attention.seismic`'s
// AFFINE_GROUP), and the pairs per vector.
constexpr int GROUP = static_cast<int>(SEISMIC_CONST_AFFINE_GROUP);
constexpr int PAIRS = W / GROUP;
constexpr int PAIR_LANES = GROUP / DPL;
static_assert(W % GROUP == 0 && GROUP % DPL == 0, "a lane's dimensions lie in one group");

// A lane's share of one code row: its DPL codes are DPL * B bits, whole
// words or a power-of-two part of one word shared with its neighbours.
template <int B> struct LaneCodes {
    static constexpr int bits = DPL * B;
    static constexpr int words = (bits + 31) / 32;
    static constexpr int row_words = W * B / 32;
    static constexpr u32 levels = (1u << B) - 1u;
    static_assert(bits % 32 == 0 || 32 % bits == 0,
                  "a lane's codes are whole words or a power-of-two part of one");

    // This lane's codes of a code row, shifted so its code i sits at bits
    // B * i of word i * B / 32.
    __device__ static __forceinline__ void load(const u32 *row, int lane, u32 (&w)[words]) {
        if constexpr (bits == 128) {
            const uint4 quad = *reinterpret_cast<const uint4 *>(row + lane * 4);
            w[0] = quad.x;
            w[1] = quad.y;
            w[2] = quad.z;
            w[3] = quad.w;
        } else if constexpr (bits == 64) {
            const uint2 pair = *reinterpret_cast<const uint2 *>(row + lane * 2);
            w[0] = pair.x;
            w[1] = pair.y;
        } else if constexpr (bits >= 32) {
#pragma unroll
            for (int j = 0; j < words; ++j) w[j] = row[lane * words + j];
        } else {
            w[0] = row[lane * bits / 32] >> ((lane * bits) % 32);
        }
    }

    // Code i of this lane's dimensions.
    __device__ static __forceinline__ u32 at(const u32 (&w)[words], int i) {
        return (w[i * B / 32] >> ((i * B) % 32)) & levels;
    }

    // Code i of this lane's dimensions, as F32 (exact): the code in the
    // mantissa of 2^23, minus 2^23 (no conversion instruction).
    __device__ static __forceinline__ float code(const u32 (&w)[words], int i) {
        return __uint_as_float(0x4B000000u | at(w, i)) - 8388608.0f;
    }
};

// A lane's share of an encoded vector: its DPL codes, laid out as
// `LaneCodes::load` returns a code row's, and its group's F16-rounded
// (scale, zero) pair.
template <int B> struct LaneEncoding {
    u32 codes[LaneCodes<B>::words];
    float scale;
    float zero;
};

// This lane's share of the B-bit encoding of one vector held by a warp (each
// lane owns its DPL consecutive dimensions; the whole warp calls it): per
// group, zero = f16(min), scale = f16((max - min) / L), code = min(L,
// u32(fma(x - zero, 1 / scale, 0.5))), 0 when scale is 0.
template <int B>
__device__ __forceinline__ LaneEncoding<B> quantize(const float (&x)[DPL]) {
    typedef LaneCodes<B> Codes;
    float low = x[0];
    float high = x[0];
#pragma unroll
    for (int d = 1; d < DPL; ++d) {
        low = fminf(low, x[d]);
        high = fmaxf(high, x[d]);
    }
    // The group's range, over its PAIR_LANES neighbouring lanes.
#pragma unroll
    for (int offset = 1; offset < PAIR_LANES; offset *= 2) {
        low = fminf(low, seismic_shfl_xor_f32(low, offset));
        high = fmaxf(high, seismic_shfl_xor_f32(high, offset));
    }
    const float zero = element::F16::round(low);
    const float scale = element::F16::round((high - low) / static_cast<float>(Codes::levels));
    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    LaneEncoding<B> e;
#pragma unroll
    for (int j = 0; j < Codes::words; ++j) e.codes[j] = 0u;
#pragma unroll
    for (int d = 0; d < DPL; ++d) {
        const float t = __fmaf_rn(x[d] - zero, inverse, 0.5f);
        const u32 c = min(static_cast<u32>(fmaxf(t, 0.0f)), Codes::levels);
        e.codes[d * B / 32] |= c << ((d * B) % 32);
    }
    e.scale = scale;
    e.zero = zero;
    return e;
}

// Stores a warp's encoding of one vector: its codes at `row` (its code row)
// and its group pairs at `pairs` (the whole warp calls it).
template <int B>
__device__ __forceinline__ void store_encoding(const LaneEncoding<B> &e, u32 *row,
                                               const element::Out<element::F16> &pairs, int lane) {
    typedef LaneCodes<B> Codes;
    if constexpr (Codes::bits >= 32) {
        u32 *dst = row + lane * Codes::words;
        if constexpr (Codes::words % 4 == 0) {
            if ((reinterpret_cast<u64>(dst) & 15u) == 0) {
#pragma unroll
                for (int j = 0; j < Codes::words; j += 4) {
                    *reinterpret_cast<uint4 *>(dst + j) =
                        make_uint4(e.codes[j], e.codes[j + 1], e.codes[j + 2], e.codes[j + 3]);
                }
            } else {
#pragma unroll
                for (int j = 0; j < Codes::words; ++j) dst[j] = e.codes[j];
            }
        } else if constexpr (Codes::words % 2 == 0) {
            if ((reinterpret_cast<u64>(dst) & 7u) == 0) {
#pragma unroll
                for (int j = 0; j < Codes::words; j += 2) {
                    *reinterpret_cast<uint2 *>(dst + j) = make_uint2(e.codes[j], e.codes[j + 1]);
                }
            } else {
#pragma unroll
                for (int j = 0; j < Codes::words; ++j) dst[j] = e.codes[j];
            }
        } else {
#pragma unroll
            for (int j = 0; j < Codes::words; ++j) dst[j] = e.codes[j];
        }
    } else {
        // 32 / bits neighbouring lanes share a word: join their parts.
        constexpr int sharing = 32 / Codes::bits;
        u32 joined = e.codes[0] << ((lane % sharing) * Codes::bits);
#pragma unroll
        for (int offset = 1; offset < sharing; offset *= 2)
            joined |= seismic_shfl_xor_u32(joined, offset);
        if (lane % sharing == 0) row[lane / sharing] = joined;
    }
    // Pair p is elements 2p (scale) and 2p + 1 (zero): one 32-bit store.
    if (lane % PAIR_LANES == 0) {
        pairs.put2(2 * static_cast<u64>(lane / PAIR_LANES), e.scale, e.zero);
    }
}

// A (scale, zero) pair, as F32.
__device__ __forceinline__ float2 coefficients(u32 pair) { return seismic_unpack_f16x2(pair); }

// Read-only view of affine history: code planes [T, KV, W * B / 32] u32 and
// (scale, zero) planes [T, KV, 2 * PAIRS] f16 per vector kind.
struct AffineReads {
    static constexpr bool CODED = true;
    const u32 *key_codes;
    const u32 *key_pairs;
    const u32 *value_codes;
    const u32 *value_pairs;
    u64 key_codes_row, key_codes_head, key_pairs_row, key_pairs_head;
    u64 value_codes_row, value_codes_head, value_pairs_row, value_pairs_head;
    u64 slab_rows;

    // The end of the run of rows [token, hi) within history row `token`'s
    // slab.
    __device__ __forceinline__ int slab_end(int token, int hi) const {
        return static_cast<int>(min(static_cast<u64>(hi), (static_cast<u64>(token) / slab_rows + 1) * slab_rows));
    }

    // A (history row, kv head)'s vectors in the four planes, its slab located
    // once (32-bit: rows and slab_rows, the entry's u32 SLAB_ROWS, fit).
    struct Vectors {
        const u32 *key_codes;
        const u32 *key_pairs;
        const u32 *value_codes;
        const u32 *value_pairs;
        // Row `within` of the slab these are the first vectors of (row
        // strides in words: codes 4-byte, pairs 2-byte elements).
        __device__ __forceinline__ Vectors at(const AffineReads &h, u32 within) const {
            return Vectors{key_codes + within * h.key_codes_row, key_pairs + within * (h.key_pairs_row / 2),
                           value_codes + within * h.value_codes_row, value_pairs + within * (h.value_pairs_row / 2)};
        }
    };
    __device__ __forceinline__ Vectors vectors(int token, int kv_head) const {
        const u32 rows = static_cast<u32>(slab_rows);
        return slab_vectors(static_cast<u32>(token) / rows, kv_head).at(*this, static_cast<u32>(token) % rows);
    }
    // A slab's first (history row, kv head) vectors; `at` steps them to a
    // later row of the slab.
    __device__ __forceinline__ Vectors slab_vectors(u32 slab_index, int kv_head) const {
        auto at = [&](const u32 *table, u64 head_words) {
            return reinterpret_cast<const u32 *>(slab::region(table, slab_index)) +
                   static_cast<u64>(kv_head) * head_words;
        };
        auto pairs = [&](const u32 *table, u64 head_elements) {
            return reinterpret_cast<const u32 *>(
                slab::region(table, slab_index) + static_cast<u64>(kv_head) * head_elements * 2);
        };
        return Vectors{at(key_codes, key_codes_head), pairs(key_pairs, key_pairs_head),
                       at(value_codes, value_codes_head), pairs(value_pairs, value_pairs_head)};
    }
};

// Affine history: code planes [T, KV, W * B / 32] u32 and (scale, zero)
// planes [T, KV, 2 * PAIRS] f16 per vector kind. Every plane is canonical: a
// vector's code row and its pairs are contiguous.
struct AffineHistory {
    static constexpr bool CODED = true;
    const u32 *key_codes;
    element::SlabOut<element::F16> key_pairs;
    const u32 *value_codes;
    element::SlabOut<element::F16> value_pairs;
    u64 key_codes_row, key_codes_head, key_pairs_row, key_pairs_head;
    u64 value_codes_row, value_codes_head, value_pairs_row, value_pairs_head;
    u64 slab_rows;

    __device__ __forceinline__ AffineReads reads() const {
        return AffineReads{key_codes,       key_pairs.reads(), value_codes,     value_pairs.reads(),
                           key_codes_row,   key_codes_head,    key_pairs_row,   key_pairs_head,
                           value_codes_row, value_codes_head,  value_pairs_row, value_pairs_head,
                           slab_rows};
    }
    __device__ __forceinline__ operator AffineReads() const { return reads(); }

    // The end of the run of rows [token, hi) within history row `token`'s
    // slab.
    __device__ __forceinline__ int slab_end(int token, int hi) const { return reads().slab_end(token, hi); }

    // A vector's code row and (scale, zero) pairs (one u32 each).
    __device__ __forceinline__ u32 *key_row(int token, int kv_head) const {
        return reinterpret_cast<u32 *>(slab::row(key_codes, token, slab_rows, key_codes_row * 4))
            + static_cast<u64>(kv_head) * key_codes_head;
    }
    __device__ __forceinline__ element::Out<element::F16> key_pair(int token, int kv_head) const {
        return key_pairs.row(static_cast<u64>(token), slab_rows, key_pairs_row)
            .offset(static_cast<u64>(kv_head) * key_pairs_head);
    }
    __device__ __forceinline__ u32 *value_row(int token, int kv_head) const {
        return reinterpret_cast<u32 *>(slab::row(value_codes, token, slab_rows, value_codes_row * 4))
            + static_cast<u64>(kv_head) * value_codes_head;
    }
    __device__ __forceinline__ element::Out<element::F16> value_pair(int token, int kv_head) const {
        return value_pairs.row(static_cast<u64>(token), slab_rows, value_pairs_row)
            .offset(static_cast<u64>(kv_head) * value_pairs_head);
    }

    using Vectors = AffineReads::Vectors;
    __device__ __forceinline__ Vectors vectors(int token, int kv_head) const { return reads().vectors(token, kv_head); }
    __device__ __forceinline__ Vectors slab_vectors(u32 slab_index, int kv_head) const {
        return reads().slab_vectors(slab_index, kv_head);
    }

    // A lane's share of a key's and a value's encodings.
    struct Encoding {
        LaneEncoding<KEY_BITS> key;
        LaneEncoding<VALUE_BITS> value;
    };
    // The encodings of a key and a value held by a warp (the whole warp calls
    // it): the key is the prepared key rounded to the activation element (as
    // `k` holds it), the value the projected value. Every read of a batch row
    // reads this encoding, as history holds it.
    __device__ static __forceinline__ Encoding quantize(const float (&k)[DPL], const float (&v)[DPL]) {
        return Encoding{attention::quantize<KEY_BITS>(k), attention::quantize<VALUE_BITS>(v)};
    }
    __device__ __forceinline__ void store(int destination, int kv_head, const Encoding &e,
                                          int lane) const {
        store_encoding(e.key, key_row(destination, kv_head), key_pair(destination, kv_head), lane);
        store_encoding(e.value, value_row(destination, kv_head), value_pair(destination, kv_head),
                       lane);
    }
    __device__ __forceinline__ void append(int destination, int kv_head, const float (&k)[DPL],
                                           const float (&v)[DPL], int lane) const {
        store(destination, kv_head, quantize(k, v), lane);
    }
};


// ---------------------------------------------------------------------------
// Affine history decoded into 16-bit MMA operands (the flash attention body,
// `prefill.cuh`). Codes decode in packed F16 (the affine operands' element):
// code c sits in the mantissa of the F16 1024 (0x6400 | c, exact), 1024 is
// subtracted exactly, and one F16x2 fma applies the group's (scale, zero)
// `pair`, rounding once (the F32 fma rounded to F16 gives the same operands on
// the goldens), two dimensions per instruction. Every affine operand, history
// or fresh, is decoded here.
static_assert(KEY_BITS == 8 && VALUE_BITS == 4, "the operand decode reads 8-bit key and 4-bit value codes");

__device__ __forceinline__ u32 half_decode(u32 biased, u32 pair) {
    const u32 scales = __byte_perm(pair, 0, 0x1010);
    const u32 zeros = __byte_perm(pair, 0, 0x3232);
    u32 code, value;
    asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(code) : "r"(biased), "r"(0x64006400u));
    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(value) : "r"(code), "r"(scales), "r"(zeros));
    return value;
}

// Eight 8-bit codes (two words) decoded with their group's (scale, zero)
// `pair` into one operand chunk.
__device__ __forceinline__ uint4 key_chunk(u32 low, u32 high, u32 pair) {
    return make_uint4(half_decode(__byte_perm(low, 0x64646464u, 0x4140), pair),
                      half_decode(__byte_perm(low, 0x64646464u, 0x4342), pair),
                      half_decode(__byte_perm(high, 0x64646464u, 0x4140), pair),
                      half_decode(__byte_perm(high, 0x64646464u, 0x4342), pair));
}

// Eight 4-bit codes (one word) decoded with their group's pair into one
// operand chunk: dimension pair j is byte j's low and high nibbles.
__device__ __forceinline__ uint4 value_chunk(u32 word, u32 pair) {
    // Byte j of `low` holds nibble 2j, of `high` nibble 2j + 1; pair j
    // gathers both into bytes 0 and 1, then under the F16 1024's exponent.
    const u32 low = word & 0x0F0F0F0Fu;
    const u32 high = (word >> 4) & 0x0F0F0F0Fu;
    u32 biased[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
        biased[j] = __byte_perm(__byte_perm(low, high, static_cast<u32>(j) | (static_cast<u32>(4 + j) << 4)),
                                0x64646464u, 0x4140);
    return make_uint4(half_decode(biased[0], pair), half_decode(biased[1], pair),
                      half_decode(biased[2], pair), half_decode(biased[3], pair));
}

// A lane's share of an encoded vector decoded into its DPL operands, as F32
// (exact): dimension pairs decode as a history tile's (`half_decode`).
template <int B>
__device__ __forceinline__ void decoded(const LaneEncoding<B> &e, float (&x)[DPL]) {
    typedef LaneCodes<B> Codes;
    const u32 pair = element::F16::pack2(e.scale, e.zero);
#pragma unroll
    for (int d = 0; d < DPL; d += 2) {
        const u32 high = d + 1 < DPL ? Codes::at(e.codes, d + 1) : 0u;
        const u32 biased = (0x6400u | Codes::at(e.codes, d)) | ((0x6400u | high) << 16);
        const float2 values = seismic_unpack_f16x2(half_decode(biased, pair));
        x[d] = values.x;
        if (d + 1 < DPL) x[d + 1] = values.y;
    }
}

// Affine history tiles decoded into the swizzled operand tiles of a flash
// attention body, whose layout `Tile` supplies: KEYS rows a tile; key pieces
// of PIECE columns (PIECES per head, PIECE_CHUNKS 16-byte chunks each); value
// windows of WINDOW columns (WINDOW_CHUNKS chunks); PRODUCERS producer threads
// (0 without producer warps); and `swizzled<ROW_CHUNKS>(row, chunk)`, a
// chunk's byte offset in a tile. A code piece is 16 bytes of one row's codes,
// all in one group.
template <class Tile> struct AffineTiles {
    static constexpr int PRODUCER_STRIDE = Tile::PRODUCERS > 0 ? Tile::PRODUCERS : 1;
    // Key code pieces of one staged key piece.
    static constexpr int KEY_PIECES = Tile::PIECE * KEY_BITS / 128;
    // Value pieces of one output window.
    static constexpr int VALUE_PIECES = Tile::WINDOW * VALUE_BITS / 128;
    static constexpr int KEY_ITEMS = (Tile::KEYS * KEY_PIECES + PRODUCER_STRIDE - 1) / PRODUCER_STRIDE;
    static constexpr int VALUE_ITEMS = (Tile::KEYS * VALUE_PIECES + PRODUCER_STRIDE - 1) / PRODUCER_STRIDE;
    static constexpr int KEY_PIECE_CODES = 128 / KEY_BITS;
    static constexpr int VALUE_PIECE_CODES = 128 / VALUE_BITS;
    static_assert(GROUP % KEY_PIECE_CODES == 0 && GROUP % VALUE_PIECE_CODES == 0,
                  "a code piece lies in one group");

    // Key piece p of tile row r (16 codes, `bits`) decoded with its group's
    // `pair` into its two operand chunks of key tile `k_tile`.
    __device__ static __forceinline__ void put_key(u8 *k_tile, int r, int p, const uint4 &bits, u32 pair) {
        *reinterpret_cast<uint4 *>(k_tile + Tile::template swizzled<Tile::PIECE_CHUNKS>(r, 2 * p)) =
            key_chunk(bits.x, bits.y, pair);
        *reinterpret_cast<uint4 *>(k_tile + Tile::template swizzled<Tile::PIECE_CHUNKS>(r, 2 * p + 1)) =
            key_chunk(bits.z, bits.w, pair);
    }

    // Value piece p of tile row r (32 codes, `bits`) decoded with its group's
    // `pair` into its four operand chunks of value tile `v_tile`.
    __device__ static __forceinline__ void put_value(u8 *v_tile, int r, int p, const uint4 &bits, u32 pair) {
        *reinterpret_cast<uint4 *>(v_tile + Tile::template swizzled<Tile::WINDOW_CHUNKS>(r, 4 * p)) =
            value_chunk(bits.x, pair);
        *reinterpret_cast<uint4 *>(v_tile + Tile::template swizzled<Tile::WINDOW_CHUNKS>(r, 4 * p + 1)) =
            value_chunk(bits.y, pair);
        *reinterpret_cast<uint4 *>(v_tile + Tile::template swizzled<Tile::WINDOW_CHUNKS>(r, 4 * p + 2)) =
            value_chunk(bits.z, pair);
        *reinterpret_cast<uint4 *>(v_tile + Tile::template swizzled<Tile::WINDOW_CHUNKS>(r, 4 * p + 3)) =
            value_chunk(bits.w, pair);
    }

    // Piece p of tile row r of history vectors `v` decoded into the operand
    // tiles: of key piece `piece`, or of the value window from `column0`.
    __device__ static __forceinline__ void produce_piece(const AffineReads::Vectors &v, bool value, int r, int p,
                                                         int piece, int column0, u8 *k_tile, u8 *v_tile) {
        if (!value) {
            const int code = piece * KEY_PIECES + p;
            const uint4 bits = seismic_ld_nc_v4(v.key_codes + code * 4);
            const u32 pair = seismic_ld_nc_u32(v.key_pairs + code * KEY_PIECE_CODES / GROUP);
            put_key(k_tile, r, p, bits, pair);
        } else {
            const int code = column0 * VALUE_BITS / 128 + p;
            const uint4 bits = seismic_ld_nc_v4(v.value_codes + code * 4);
            const u32 pair = seismic_ld_nc_u32(v.value_pairs + code * VALUE_PIECE_CODES / GROUP);
            put_value(v_tile, r, p, bits, pair);
        }
    }

    // A producer thread's codes and (scale, zero) pairs of one affine unit,
    // held in registers between their loads and their decode.
    struct ProducedCodes {
        uint4 key_bits[KEY_ITEMS];
        u32 key_pairs[KEY_ITEMS];
        uint4 value_bits[VALUE_ITEMS];
        u32 value_pairs[VALUE_ITEMS];
    };

    // Producer thread `thread`'s loads of a contiguous affine history slice
    // `[first, first + rows)`. Every load is issued before any is used.
    __device__ static __forceinline__ ProducedCodes fetch_produced(const AffineReads &history, int first, int rows,
                                                                   int kv, int piece, int column0, int thread) {
        const int key_piece0 = piece * KEY_PIECES;
        const int value_piece0 = column0 * VALUE_BITS / 128;
        const bool values = piece + 1 == Tile::PIECES;
        ProducedCodes codes;
#pragma unroll
        for (int k = 0; k < KEY_ITEMS; ++k) {
            const int index = thread + k * Tile::PRODUCERS;
            const int token = first + index / KEY_PIECES;
            const int code = key_piece0 + index % KEY_PIECES;
            codes.key_bits[k] = make_uint4(0, 0, 0, 0);
            codes.key_pairs[k] = 0u;
            if (index < rows * KEY_PIECES) {
                const AffineReads::Vectors v = history.vectors(token, kv);
                codes.key_bits[k] = seismic_ld_nc_v4(v.key_codes + code * 4);
                codes.key_pairs[k] = seismic_ld_nc_u32(v.key_pairs + code * KEY_PIECE_CODES / GROUP);
            }
        }
#pragma unroll
        for (int k = 0; k < VALUE_ITEMS; ++k) {
            const int index = thread + k * Tile::PRODUCERS;
            const int token = first + index / VALUE_PIECES;
            const int code = value_piece0 + index % VALUE_PIECES;
            codes.value_bits[k] = make_uint4(0, 0, 0, 0);
            codes.value_pairs[k] = 0u;
            if (values && index < rows * VALUE_PIECES) {
                const AffineReads::Vectors v = history.vectors(token, kv);
                codes.value_bits[k] = seismic_ld_nc_v4(v.value_codes + code * 4);
                codes.value_pairs[k] = seismic_ld_nc_u32(v.value_pairs + code * VALUE_PIECE_CODES / GROUP);
            }
        }
        return codes;
    }

    // Producer thread `thread`'s fetched codes of rows [0, rows) decoded into
    // the K operand tile (key piece `piece`) and, with the last piece, the V
    // operand tile, each code piece with its group's (scale, zero) pair.
    __device__ static __forceinline__ void store_produced(const ProducedCodes &codes, u8 *k_tile, u8 *v_tile,
                                                          int rows, int piece, int thread) {
        const bool values = piece + 1 == Tile::PIECES;
#pragma unroll
        for (int k = 0; k < KEY_ITEMS; ++k) {
            const int index = thread + k * Tile::PRODUCERS;
            if (index < rows * KEY_PIECES)
                put_key(k_tile, index / KEY_PIECES, index % KEY_PIECES, codes.key_bits[k], codes.key_pairs[k]);
        }
#pragma unroll
        for (int k = 0; k < VALUE_ITEMS; ++k) {
            const int index = thread + k * Tile::PRODUCERS;
            if (values && index < rows * VALUE_PIECES)
                put_value(v_tile, index / VALUE_PIECES, index % VALUE_PIECES, codes.value_bits[k],
                          codes.value_pairs[k]);
        }
    }

    // Affine history decoded by the MMA warps themselves (no producer warps):
    // a unit's code rows and (scale, zero) pairs are copied with `cp.async`
    // into a code stage beside its operand stage, as dense history is copied,
    // and decoded into the operand stage once they land. The MMA warps keep
    // the whole register file (dense's budget); the decode is a few ALU
    // instructions per operand beside the tensor-core products. Heads of 128
    // to 256 columns: every code and pair row is whole 16-byte chunks.
    //
    // A code stage: key codes [KEYS][PIECE] u8, key pairs [KEYS][PIECE /
    // GROUP] u32, value codes [KEYS][WINDOW / 2] u8, value pairs [KEYS][WINDOW
    // / GROUP] u32.
    static constexpr int KEY_PAIR_CHUNKS = Tile::PIECE / GROUP * 4 / 16;
    static constexpr int VALUE_PAIR_CHUNKS = Tile::WINDOW / GROUP * 4 / 16;
    static constexpr int KEY_ROW_CHUNKS = KEY_PIECES + KEY_PAIR_CHUNKS;
    static constexpr int CODE_ROW_CHUNKS = KEY_ROW_CHUNKS + VALUE_PIECES + VALUE_PAIR_CHUNKS;
    static constexpr int CODE_STAGE_BYTES = Tile::KEYS * CODE_ROW_CHUNKS * 16;
    static constexpr int KEY_PAIRS_AT = Tile::KEYS * KEY_PIECES * 16;
    static constexpr int VALUE_CODES_AT = KEY_PAIRS_AT + Tile::KEYS * KEY_PAIR_CHUNKS * 16;
    static constexpr int VALUE_PAIRS_AT = VALUE_CODES_AT + Tile::KEYS * VALUE_PIECES * 16;

    // Copy the affine unit's history rows [0, rows) (`keys(r)` the history
    // token of tile row r, or -1 for a zero row; key piece `piece`, and with
    // the last piece the output window's values from column0) into code stage
    // `codes` by `threads` threads of which this is `thread`: threads / KEYS
    // consecutive threads per row, so each locates its row's slab once and a
    // row's chunks are copied contiguously. A zero row, or one past `rows`, is
    // zero codes and pairs (they decode to exact zeros).
    template <class Keys>
    __device__ static __forceinline__ void stage_codes(const AffineReads &history, u8 *codes, const Keys &keys,
                                                       int rows, int kv, int piece, int column0, int thread,
                                                       int threads) {
        const int per_row = threads / Tile::KEYS;
        const int r = thread / per_row;
        const int raw = keys(min(r, Tile::KEYS - 1));
        const int token = r < rows ? raw : -1;
        const bool live = token >= 0;
        // A dead row still names a valid source (the tile's first row).
        const int zero = keys(0);
        const AffineReads::Vectors v = history.vectors(live ? token : zero, kv);
        const u32 bytes = live ? 16u : 0u;
        const u8 *key_codes = reinterpret_cast<const u8 *>(v.key_codes) + piece * Tile::PIECE;
        const u8 *key_pairs = reinterpret_cast<const u8 *>(v.key_pairs) + piece * Tile::PIECE / GROUP * 4;
        const u8 *value_codes = reinterpret_cast<const u8 *>(v.value_codes) + column0 / 2;
        const u8 *value_pairs = reinterpret_cast<const u8 *>(v.value_pairs) + column0 / GROUP * 4;
        const int chunks = piece + 1 == Tile::PIECES ? CODE_ROW_CHUNKS : KEY_ROW_CHUNKS;
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

    // Decode rows [0, rows) of code stage `codes` into the K operand tile
    // and, with the last piece (`values`), the V operand tile, by `threads`
    // threads of which this is `thread`.
    __device__ static __forceinline__ void decode_codes(const u8 *codes, u8 *k_tile, u8 *v_tile, bool values,
                                                        int rows, int thread, int threads) {
        for (int index = thread; index < rows * KEY_PIECES; index += threads) {
            const int r = index / KEY_PIECES;
            const int p = index % KEY_PIECES;
            const uint4 bits = *reinterpret_cast<const uint4 *>(codes + index * 16);
            const u32 pair = *reinterpret_cast<const u32 *>(
                codes + KEY_PAIRS_AT + r * KEY_PAIR_CHUNKS * 16 + p * KEY_PIECE_CODES / GROUP * 4);
            put_key(k_tile, r, p, bits, pair);
        }
        if (!values) return;
        for (int index = thread; index < rows * VALUE_PIECES; index += threads) {
            const int r = index / VALUE_PIECES;
            const int p = index % VALUE_PIECES;
            const uint4 bits = *reinterpret_cast<const uint4 *>(codes + VALUE_CODES_AT + index * 16);
            const u32 pair = *reinterpret_cast<const u32 *>(
                codes + VALUE_PAIRS_AT + r * VALUE_PAIR_CHUNKS * 16 + p * VALUE_PIECE_CODES / GROUP * 4);
            put_value(v_tile, r, p, bits, pair);
        }
    }
};

#endif

// Append row `row`'s prepared key `k` and raw value for `kv_head` at its
// destination through `history`, when it has one: the key alone for
// stored-key values. One warp.
template <class History>
__device__ __forceinline__ void append(const Inputs &in, const History &history, u64 row,
                                       int kv_head, const float (&k)[DPL], int lane) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const int destination = ATTENTION_DESTINATION(in, row);
    if (destination < 0) return;
    float v[DPL];
    if constexpr (!STORED_KEY) fresh_value(in, row, kv_head, v, lane);
    history.append(destination, kv_head, k, v, lane);
}
#ifdef ATTENTION_HISTORY_STAGING
// Ephemeral operands for exactly the call's union of visible rows. The
// stored planes and append policy remain affine. Every vector is located
// independently: consumer KEY tiles retain the original span-based grouping.
struct ListedF16History {
    static constexpr bool CODED = false;
    const int *tiles;
    const u32 *valid;
    const u16 *keys;
    const u16 *values;
    u64 count, tile_stride, slab_rows;

    __device__ __forceinline__ int listed(int token) const {
        if (token < 0) return -1;
        const int page = token / 256;
        u64 lo = 0, hi = count;
        while (lo < hi) {
            const u64 mid = lo + (hi - lo) / 2;
            const int held = tiles[mid * tile_stride];
            if (held >= 0 && held < page) lo = mid + 1;
            else hi = mid;
        }
        return lo < count && tiles[lo * tile_stride] == page ? static_cast<int>(lo) : -1;
    }
    __device__ __forceinline__ int slab_end(int token, int hi) const {
        return static_cast<int>(min(static_cast<u64>(hi), (static_cast<u64>(token) / slab_rows + 1) * slab_rows));
    }
    __device__ __forceinline__ const u8 *vector(const u16 *plane, int token, int kv) const {
        const int slot = listed(token);
        const int row = token & 255;
        if (slot < 0 || !(valid[static_cast<u64>(slot) * 8 + row / 32] & (1u << (row & 31)))) return nullptr;
        return reinterpret_cast<const u8 *>(plane + ((static_cast<u64>(slot) * 256 + row) * KV + kv) * W);
    }
    __device__ __forceinline__ const u8 *key_vector(int token, int kv) const { return vector(keys, token, kv); }
    __device__ __forceinline__ const u8 *value_vector(int token, int kv) const { return vector(values, token, kv); }
};
#endif
