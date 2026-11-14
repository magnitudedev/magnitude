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
        const element::Out<Act> v_row = value_out(destination, kv_head);
#pragma unroll
        for (int d = 0; d < DPL; ++d) {
            k_row.put(lane * DPL + d, k[d]);
            if constexpr (!STORED_KEY) v_row.put(lane * DPL + d, v[d]);
        }
    }
};

#ifdef ATTENTION_AFFINE_HISTORY
// Affine K8/V4 history (the `affine_k8_uniform_v4` codec,
// `attention_*_k8v4`). A (history row, kv head) vector is a code row of
// W * B / 32 u32 words (code i at bits B * (i % (32 / B)) of word
// i / (32 / B)) plus one F16 (scale, zero) pair per group of GROUP
// consecutive dimensions, pairs in group order; decoded value =
// code * scale + zero with its group's pair. Keys use B = 8, values B = 4. A
// lane's DPL dimensions lie in one group, shared by PAIR_LANES lanes.
constexpr int KEY_BITS = 32 * ATTENTION_KEY_CODE_WORDS / W;
constexpr int VALUE_BITS = 32 * ATTENTION_VALUE_CODE_WORDS / W;
// Dimensions per (scale, zero) pair: the codec's group (model-state
// `AFFINE_GROUP`), and the pairs per vector.
constexpr int GROUP = 2 * W / ATTENTION_KEY_COEFFICIENTS;
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
        if constexpr (bits == 64) {
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
#pragma unroll
        for (int j = 0; j < Codes::words; ++j) row[lane * Codes::words + j] = e.codes[j];
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
