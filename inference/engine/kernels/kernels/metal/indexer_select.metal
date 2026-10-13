// indexer_select on Metal (contract and portable body in indexer.seismic).
//
// append:   threadgroup per row; its raw key (widened to F32) and
//           coordinates to the window planes.
// complete: one simdgroup per row that completes a block; the mean of its SB
//           raw keys (F32, members in order: the last SB - 1 rows of its
//           window spans, then its own), RMS-normed with the key norm and
//           rotated at the coordinates stored with the first, stored in the
//           F32 block plane.
// score:    threadgroup per (row, tile of 8 * 32 blocks); the row's query
//           heads are prepared once into threadgroup memory, then each
//           simdgroup scores 32 blocks, one block at a time over its lanes.
// select:   threadgroup per row; radix select over the scores' bits (scores
//           are >= 0, so their bits order as the scores do) finds the K-th
//           best score T, then blocks above T and the first equal ones in
//           block order are marked, with every block from the row's tail
//           on, as the row's bitmap.
// segments: one simdgroup per row (C > 0); the marked blocks' keys mapped
//           through the row's spans, as runs within slabs.
//
// Dynamic extents and parameters read `seismic_words`, so every helper that
// needs one takes its value as an argument.

#include "lib/core/activation.h"
#include <seismic/slab.h>

#define INDEXER_W uint(2 * SEISMIC_DIM_P + SEISMIC_DIM_S)
#define INDEXER_E (INDEXER_W / 32)
#define INDEXER_H uint(SEISMIC_DIM_H)
#define INDEXER_SB uint(SEISMIC_DIM_SB)
#define INDEXER_UNROLL _Pragma("clang loop unroll(full)")

static_assert(INDEXER_W % 32 == 0, "indexer head width must be a multiple of 32");
static_assert(SEISMIC_DIM_P % INDEXER_E == 0, "each rotary half must cover whole lanes");

typedef element::Act::native Scalar;

namespace indexer {

// The rotary angle's sine and cosine, reduced as the attention entries reduce
// it (`attention::sincos`).
inline float sincos(float angle, thread float &cosine) {
    const float turns = metal::rint(angle * 0.15915494309189535f);
    float reduced = metal::fma(-turns, 6.28125f, angle);
    reduced = metal::fma(-turns, 0.0019354820251464844f, reduced);
    reduced = metal::fma(-turns, -1.7484555314695172e-07f, reduced);
    cosine = metal::fast::cos(reduced);
    return metal::fast::sin(reduced);
}

// Lane `lane`'s E columns of one head row `x` (already in registers, F32)
// RMS-normed with `norm` and its first 2P columns rotated at `coordinates`:
// the portable `head_rotary`, unrounded. The whole simdgroup calls it.
inline void norm_rotary(thread float (&x)[INDEXER_E], device const float *norm, device const int *coordinates,
    device const int *components, device const float *frequencies, device const float *amplitudes, float epsilon,
    uint lane) {
    float squares = 0.0f;
    INDEXER_UNROLL
    for (uint i = 0; i < INDEXER_E; ++i)
        squares = metal::fma(x[i], x[i], squares);
    squares = simd_sum(squares);
    const float inverse = metal::rsqrt(squares / float(INDEXER_W) + epsilon);
    INDEXER_UNROLL
    for (uint i = 0; i < INDEXER_E; ++i)
        x[i] = x[i] * inverse * norm[lane * INDEXER_E + i];
    if (SEISMIC_DIM_P == 0)
        return;
    constexpr uint half_lanes = SEISMIC_DIM_P / INDEXER_E;
    const uint partner_lane = lane < half_lanes ? lane + half_lanes
        : (lane < 2 * half_lanes ? lane - half_lanes : lane);
    float partner[INDEXER_E];
    INDEXER_UNROLL
    for (uint i = 0; i < INDEXER_E; ++i)
        partner[i] = simd_shuffle(x[i], ushort(partner_lane));
    if (lane >= 2 * half_lanes)
        return;
    INDEXER_UNROLL
    for (uint i = 0; i < INDEXER_E; ++i) {
        const uint column = lane * INDEXER_E + i;
        const uint pair = column % SEISMIC_DIM_P;
        float c;
        const float s = sincos(float(coordinates[components[pair]]) * frequencies[pair], c);
        c *= amplitudes[pair];
        const float sn = s * amplitudes[pair];
        x[i] = column < SEISMIC_DIM_P ? x[i] * c - partner[i] * sn : x[i] * c + partner[i] * sn;
    }
}

// The rows of a row's spans, in order.
inline uint span_rows(device const int *spans, uint count) {
    uint total = 0;
    for (uint span = 0; span < count; ++span) {
        const int lo = spans[span * 2];
        const int hi = spans[span * 2 + 1];
        total += hi > lo ? uint(hi - lo) : 0u;
    }
    return total;
}

// The physical row of row `ordinal` of a row's spans, then of `own` (the
// row's own, after them).
inline int span_row(device const int *spans, uint count, uint ordinal, int own) {
    uint before = 0;
    for (uint span = 0; span < count; ++span) {
        const int lo = spans[span * 2];
        const int hi = spans[span * 2 + 1];
        const uint length = hi > lo ? uint(hi - lo) : 0u;
        if (ordinal < before + length)
            return lo + int(ordinal - before);
        before += length;
    }
    return own;
}

// A row's complete blocks: those of its block spans, then its own.
inline uint complete_blocks(device const int *spans, uint count, int own) {
    return span_rows(spans, count) + (own >= 0 ? 1u : 0u);
}

} // namespace indexer

kernel void indexer_select_append(
    device const Scalar *key [[buffer(SEISMIC_BUFFER_KEY)]],
    device const int *coordinates [[buffer(SEISMIC_BUFFER_COORDINATES)]],
    device const int *window_destinations [[buffer(SEISMIC_BUFFER_WINDOW_DESTINATIONS)]],
    device const ulong *window_keys [[buffer(SEISMIC_BUFFER_WINDOW_KEYS)]],
    device const ulong *window_coordinates [[buffer(SEISMIC_BUFFER_WINDOW_COORDINATES)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const ulong row = group.x;
    const int destination = window_destinations[row];
    if (destination < 0)
        return;
    const ulong slab_rows = ulong(SEISMIC_PARAM_WINDOW_SLAB_ROWS);
    device float *target = slab::row<float>(window_keys, ulong(destination), slab_rows, INDEXER_W);
    for (uint column = thread_index; column < INDEXER_W; column += 128)
        target[column] = float(key[row * INDEXER_W + column]);
    if (thread_index < 4)
        slab::row<int>(window_coordinates, ulong(destination), slab_rows, 4)[thread_index] =
            coordinates[row * 4 + thread_index];
}

kernel void indexer_select_complete(
    device const float *key_norm [[buffer(SEISMIC_BUFFER_KEY_NORM)]],
    device const int *rotary_components [[buffer(SEISMIC_BUFFER_ROTARY_COMPONENTS)]],
    device const float *rotary_frequencies [[buffer(SEISMIC_BUFFER_ROTARY_FREQUENCIES)]],
    device const float *rotary_amplitudes [[buffer(SEISMIC_BUFFER_ROTARY_AMPLITUDES)]],
    device const int *window_visible [[buffer(SEISMIC_BUFFER_WINDOW_VISIBLE)]],
    device const int *window_destinations [[buffer(SEISMIC_BUFFER_WINDOW_DESTINATIONS)]],
    device const int *block_destinations [[buffer(SEISMIC_BUFFER_BLOCK_DESTINATIONS)]],
    device const ulong *window_keys [[buffer(SEISMIC_BUFFER_WINDOW_KEYS)]],
    device const ulong *window_coordinates [[buffer(SEISMIC_BUFFER_WINDOW_COORDINATES)]],
    device const ulong *block_keys [[buffer(SEISMIC_BUFFER_BLOCK_KEYS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const ulong row = group.x;
    const int destination = block_destinations[row];
    if (destination < 0)
        return;
    const ulong window_slab_rows = ulong(SEISMIC_PARAM_WINDOW_SLAB_ROWS);
    // The block's raw keys: the last SB - 1 rows of the window spans, then
    // the row's own.
    const uint spans = uint(SEISMIC_DIM_R);
    device const int *row_spans = window_visible + row * spans * 2;
    const uint earlier = indexer::span_rows(row_spans, spans);
    const uint skipped = earlier > INDEXER_SB - 1 ? earlier - (INDEXER_SB - 1) : 0u;
    const int own = window_destinations[row];
    int first_member = own;
    float sum[INDEXER_E];
    INDEXER_UNROLL
    for (uint i = 0; i < INDEXER_E; ++i)
        sum[i] = 0.0f;
    for (uint member = 0; member < INDEXER_SB; ++member) {
        const int at = member + 1 < INDEXER_SB ? indexer::span_row(row_spans, spans, skipped + member, own) : own;
        if (member == 0)
            first_member = at;
        device const float *raw = slab::row<float>(window_keys, ulong(at), window_slab_rows, INDEXER_W);
        INDEXER_UNROLL
        for (uint i = 0; i < INDEXER_E; ++i)
            sum[i] += raw[lane * INDEXER_E + i];
    }
    float x[INDEXER_E];
    INDEXER_UNROLL
    for (uint i = 0; i < INDEXER_E; ++i)
        x[i] = sum[i] / float(INDEXER_SB);
    device const int *first_coordinates = reinterpret_cast<device const int *>(
        slab::row<int>(window_coordinates, ulong(first_member), window_slab_rows, 4));
    indexer::norm_rotary(x, key_norm, first_coordinates, rotary_components, rotary_frequencies,
        rotary_amplitudes, as_type<float>(uint(SEISMIC_PARAM_EPSILON)), lane);
    device float *target =
        slab::row<float>(block_keys, ulong(destination), ulong(SEISMIC_PARAM_BLOCK_SLAB_ROWS), INDEXER_W);
    INDEXER_UNROLL
    for (uint i = 0; i < INDEXER_E; ++i)
        target[lane * INDEXER_E + i] = x[i];
}

kernel void indexer_select_score(
    device const Scalar *query [[buffer(SEISMIC_BUFFER_QUERY)]],
    device const float *query_norm [[buffer(SEISMIC_BUFFER_QUERY_NORM)]],
    device const int *rotary_components [[buffer(SEISMIC_BUFFER_ROTARY_COMPONENTS)]],
    device const float *rotary_frequencies [[buffer(SEISMIC_BUFFER_ROTARY_FREQUENCIES)]],
    device const float *rotary_amplitudes [[buffer(SEISMIC_BUFFER_ROTARY_AMPLITUDES)]],
    device const int *coordinates [[buffer(SEISMIC_BUFFER_COORDINATES)]],
    device const int *block_visible [[buffer(SEISMIC_BUFFER_BLOCK_VISIBLE)]],
    device const int *block_destinations [[buffer(SEISMIC_BUFFER_BLOCK_DESTINATIONS)]],
    device const ulong *block_keys [[buffer(SEISMIC_BUFFER_BLOCK_KEYS)]],
    device float *scores [[buffer(SEISMIC_BUFFER_SCRATCH_SCORES)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup float *prepared [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const ulong row = group.y;
    const uint spans = uint(SEISMIC_DIM_R);
    const ulong nb = 32 * ulong(SEISMIC_DIM_B);
    device const int *row_spans = block_visible + row * spans * 2;
    const int own = block_destinations[row];
    const uint complete = indexer::complete_blocks(row_spans, spans, own);
    const uint first = group.x * 256;
    if (first >= complete)
        return;
    // The row's query heads, prepared in F32.
    for (uint head = simd; head < INDEXER_H; head += 8) {
        float x[INDEXER_E];
        device const Scalar *raw = query + (row * INDEXER_H + head) * INDEXER_W;
        INDEXER_UNROLL
        for (uint i = 0; i < INDEXER_E; ++i)
            x[i] = float(raw[lane * INDEXER_E + i]);
        indexer::norm_rotary(x, query_norm, coordinates + row * 4, rotary_components, rotary_frequencies,
            rotary_amplitudes, as_type<float>(uint(SEISMIC_PARAM_EPSILON)), lane);
        INDEXER_UNROLL
        for (uint i = 0; i < INDEXER_E; ++i)
            prepared[head * INDEXER_W + lane * INDEXER_E + i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q[INDEXER_H][INDEXER_E];
    INDEXER_UNROLL
    for (uint head = 0; head < INDEXER_H; ++head)
        INDEXER_UNROLL
        for (uint i = 0; i < INDEXER_E; ++i)
            q[head][i] = prepared[head * INDEXER_W + lane * INDEXER_E + i];
    const ulong slab_rows = ulong(SEISMIC_PARAM_BLOCK_SLAB_ROWS);
    const float root = metal::sqrt(float(INDEXER_W));
    const uint last = metal::min(first + 256, complete);
    for (uint block = first + simd * 32; block < metal::min(first + simd * 32 + 32, last); ++block) {
        const int at = indexer::span_row(row_spans, spans, block, own);
        device const float *k = slab::row<float>(block_keys, ulong(at), slab_rows, INDEXER_W);
        float kx[INDEXER_E];
        INDEXER_UNROLL
        for (uint i = 0; i < INDEXER_E; ++i)
            kx[i] = k[lane * INDEXER_E + i];
        float score = 0.0f;
        INDEXER_UNROLL
        for (uint head = 0; head < INDEXER_H; ++head) {
            float dot = 0.0f;
            INDEXER_UNROLL
            for (uint i = 0; i < INDEXER_E; ++i)
                dot = metal::fma(q[head][i], kx[i], dot);
            score += metal::max(simd_sum(dot), 0.0f);
        }
        if (lane == 0)
            scores[row * nb + block] = score / root + 0.0f;
    }
}

kernel void indexer_select_select(
    device const int *block_visible [[buffer(SEISMIC_BUFFER_BLOCK_VISIBLE)]],
    device const int *block_destinations [[buffer(SEISMIC_BUFFER_BLOCK_DESTINATIONS)]],
    device uint *selection [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    device const float *scores [[buffer(SEISMIC_BUFFER_SCRATCH_SCORES)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup uint *shared [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint THREADS = 256;
    const ulong row = group.x;
    const uint spans = uint(SEISMIC_DIM_R);
    const uint written = uint(SEISMIC_DIM_B);
    const ulong nb = 32 * ulong(written);
    device uint *bitmap = selection + row * written;
    const uint selected = uint(metal::max(int(uint(SEISMIC_PARAM_SELECTED)), 0));
    const uint complete =
        indexer::complete_blocks(block_visible + row * spans * 2, spans, block_destinations[row]);
    device const uint *bits = reinterpret_cast<device const uint *>(scores + row * nb);
    threadgroup uint *histogram = shared;             // [256]
    threadgroup uint *counts = shared + 256;          // [THREADS]
    threadgroup uint *state = shared + 256 + THREADS; // prefix, remaining
    // Dense: every block.
    if (complete <= selected) {
        for (uint word = thread_index; word < written; word += THREADS)
            bitmap[word] = 0xffffffffu;
        return;
    }
    if (thread_index == 0) {
        state[0] = 0;
        state[1] = selected;
    }
    uint mask = 0;
    for (int digit = 3; digit >= 0; --digit) {
        for (uint bin = thread_index; bin < 256; bin += THREADS)
            histogram[bin] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint prefix = state[0];
        const uint shift = uint(digit) * 8;
        for (uint block = thread_index; block < complete; block += THREADS) {
            const uint u = bits[block];
            if ((u & mask) == prefix)
                atomic_fetch_add_explicit(reinterpret_cast<threadgroup atomic_uint *>(&histogram[(u >> shift) & 255u]),
                    1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (thread_index == 0) {
            uint remaining = state[1];
            uint value = 0;
            for (int bin = 255; bin >= 0; --bin) {
                const uint count = histogram[bin];
                if (count >= remaining) {
                    value = uint(bin);
                    break;
                }
                remaining -= count;
            }
            state[0] = prefix | (value << shift);
            state[1] = remaining;
        }
        mask |= 255u << shift;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // T = state[0]; take every block above it and the first state[1] equal
    // ones in block order. Each thread owns a run of whole words.
    const uint threshold = state[0];
    const uint take = state[1];
    const uint per = (written + THREADS - 1) / THREADS;
    const uint word_lo = metal::min(thread_index * per, written);
    const uint word_hi = metal::min(word_lo + per, written);
    uint equal = 0;
    for (uint block = word_lo * 32; block < metal::min(word_hi * 32, complete); ++block)
        equal += bits[block] == threshold ? 1u : 0u;
    counts[thread_index] = equal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint rank = 0;
    for (uint other = 0; other < thread_index; ++other)
        rank += counts[other];
    for (uint word = word_lo; word < word_hi; ++word) {
        uint value = 0;
        for (uint bit = 0; bit < 32; ++bit) {
            const uint block = word * 32 + bit;
            bool read = block >= complete;
            if (!read) {
                const uint u = bits[block];
                if (u > threshold) {
                    read = true;
                } else if (u == threshold) {
                    read = rank < take;
                    ++rank;
                }
            }
            value |= read ? (1u << bit) : 0u;
        }
        bitmap[word] = value;
    }
}

kernel void indexer_select_segments(
    device const int *visible [[buffer(SEISMIC_BUFFER_VISIBLE)]],
    device int *segments [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device const uint *selection [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const ulong row = group.x;
    const uint r = uint(SEISMIC_DIM_R);
    const uint c = uint(SEISMIC_DIM_C);
    const uint words = uint(SEISMIC_DIM_B);
    const int slab_rows = int(uint(SEISMIC_PARAM_SLAB_ROWS));
    device const uint *bitmap = selection + row * words;
    device int *out = segments + row * c * 2;
    // The row's keys in its R visible spans (its fresh rows, its tail,
    // attention reads itself).
    uint total = indexer::span_rows(visible + row * r * 2, r);
    // Lane 0 walks the marked blocks' keys with a cursor over the spans;
    // the lanes find the next nonzero bitmap word together.
    uint count = 0;
    int start = 0, end = 0;
    bool open = false;
    uint last_key = 0;
    uint span = 0, span_first = 0;
    int span_lo = 0, span_hi = 0;
    if (lane == 0 && r > 0) {
        span_lo = visible[row * r * 2];
        span_hi = visible[row * r * 2 + 1];
    }
    const uint last_word = metal::min(words, (total + 32 * INDEXER_SB - 1) / (32 * INDEXER_SB));
    for (uint base = 0; base < last_word; base += 32) {
        const uint word_index = base + lane;
        const uint word = word_index < last_word ? bitmap[word_index] : 0u;
        for (uint pending = uint(uint64_t(simd_ballot(word != 0u))); pending != 0u; pending &= pending - 1u) {
            const uint at = metal::ctz(pending);
            const uint value = simd_shuffle(word, ushort(at));
            if (lane != 0)
                continue;
            for (uint bit = 0; bit < 32; ++bit) {
                if ((value & (1u << bit)) == 0u)
                    continue;
                const uint block = (base + at) * 32 + bit;
                for (uint key = block * INDEXER_SB; key < metal::min((block + 1) * INDEXER_SB, total); ++key) {
                    // Advance the cursor to the span holding `key`.
                    while (key >= span_first + uint(metal::max(span_hi - span_lo, 0))) {
                        span_first += uint(metal::max(span_hi - span_lo, 0));
                        ++span;
                        span_lo = visible[(row * r + span) * 2];
                        span_hi = visible[(row * r + span) * 2 + 1];
                    }
                    const int stored = span_lo + int(key - span_first);
                    // A run continues over consecutive read keys stored
                    // consecutively within one slab.
                    if (open && key == last_key + 1 && stored == end && stored % slab_rows != 0) {
                        ++end;
                    } else {
                        if (open && count < c) {
                            out[count * 2] = start;
                            out[count * 2 + 1] = end;
                            ++count;
                        }
                        open = true;
                        start = stored;
                        end = stored + 1;
                    }
                    last_key = key;
                }
            }
        }
    }
    if (lane == 0) {
        if (open && count < c) {
            out[count * 2] = start;
            out[count * 2 + 1] = end;
            ++count;
        }
        for (; count < c; ++count) {
            out[count * 2] = 0;
            out[count * 2 + 1] = 0;
        }
    }
}
