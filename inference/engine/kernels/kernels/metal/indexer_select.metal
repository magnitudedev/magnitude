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
// select:   threadgroup of 256 threads per row; a radix select over the
//           scores' bits (scores are >= 0, so their bits order as the scores
//           do) finds the K-th best score T: each pass's histogram takes one
//           atomic per distinct digit of a simdgroup, and its digit is found
//           by a scan of the bins. Blocks above T and the first equal ones
//           in block order (ranked by a scan of each thread's count) are
//           marked, with every block from the row's tail on, as the row's
//           bitmap. With C > 0 it then emits the runs: the row's span offsets
//           are scanned into scratch, each thread counts the runs
//           starting in its share of the keys, and a scan of the counts
//           places each run.
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

// The rows of a row's spans: every lane of a simdgroup calls it, each
// taking the spans lane, lane + 32, ...
inline uint simd_span_rows(device const int *spans, uint count, uint lane) {
    uint total = 0;
    for (uint span = lane; span < count; span += 32) {
        const int lo = spans[span * 2];
        const int hi = spans[span * 2 + 1];
        total += hi > lo ? uint(hi - lo) : 0u;
    }
    return simd_sum(total);
}

// A forward walk over a row's spans: the physical row of each ordinal of
// its spans, then of `own` (the row's own, after them). Ordinals are asked
// in ascending order, so a launch walks each span once.
struct span_cursor {
    uint span = 0;
    uint before = 0;

    int row(device const int *spans, uint count, uint ordinal, int own) {
        for (; span < count; ++span) {
            const int lo = spans[span * 2];
            const int hi = spans[span * 2 + 1];
            const uint length = hi > lo ? uint(hi - lo) : 0u;
            if (ordinal < before + length)
                return lo + int(ordinal - before);
            before += length;
        }
        return own;
    }
};

// The exclusive prefix of `value` over the 256 threads of a threadgroup in
// thread order, and in `total` their sum. Every thread calls it; `totals`
// holds the 8 simdgroups' sums.
inline uint group_exclusive(uint value, threadgroup uint *totals, uint simd, uint lane, thread uint &total) {
    const uint inclusive = simd_prefix_inclusive_sum(value);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 31)
        totals[simd] = inclusive;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint before = 0;
    total = 0;
    for (uint other = 0; other < 8; ++other) {
        const uint sum = totals[other];
        before += other < simd ? sum : 0u;
        total += sum;
    }
    return before + inclusive - value;
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
    const uint earlier = indexer::simd_span_rows(row_spans, spans, lane);
    const uint skipped = earlier > INDEXER_SB - 1 ? earlier - (INDEXER_SB - 1) : 0u;
    const int own = window_destinations[row];
    int first_member = own;
    float sum[INDEXER_E];
    INDEXER_UNROLL
    for (uint i = 0; i < INDEXER_E; ++i)
        sum[i] = 0.0f;
    indexer::span_cursor cursor;
    for (uint member = 0; member < INDEXER_SB; ++member) {
        const int at = member + 1 < INDEXER_SB ? cursor.row(row_spans, spans, skipped + member, own) : own;
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
    const uint complete = indexer::simd_span_rows(row_spans, spans, lane) + (own >= 0 ? 1u : 0u);
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
    indexer::span_cursor cursor;
    for (uint block = first + simd * 32; block < metal::min(first + simd * 32 + 32, last); ++block) {
        const int at = cursor.row(row_spans, spans, block, own);
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
    device const int *visible [[buffer(SEISMIC_BUFFER_VISIBLE)]],
    device int *segments [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device uint *selection [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    device const float *scores [[buffer(SEISMIC_BUFFER_SCRATCH_SCORES)]],
    device uint *offsets [[buffer(SEISMIC_BUFFER_SCRATCH_SPANS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup uint *shared [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint THREADS = 256;
    const ulong row = group.x;
    const uint spans = uint(SEISMIC_DIM_R);
    const uint written = uint(SEISMIC_DIM_B);
    const ulong nb = 32 * ulong(written);
    device uint *bitmap = selection + row * written;
    const uint selected = uint(metal::max(int(uint(SEISMIC_PARAM_SELECTED)), 0));
    const uint complete = indexer::simd_span_rows(block_visible + row * spans * 2, spans, lane)
        + (block_destinations[row] >= 0 ? 1u : 0u);
    device const uint *bits = reinterpret_cast<device const uint *>(scores + row * nb);
    threadgroup uint *bins = shared;                 // [256]
    threadgroup uint *totals = shared + 256;         // [8]: the scans' simdgroup sums
    threadgroup uint *state = totals + 8;            // prefix, remaining
    uint total = 0;
    if (complete <= selected) {
        // Dense: every block.
        for (uint word = thread_index; word < written; word += THREADS)
            bitmap[word] = 0xffffffffu;
    } else {
        if (thread_index == 0) {
            state[0] = 0;
            state[1] = selected;
        }
        uint mask = 0;
        for (int digit = 3; digit >= 0; --digit) {
            bins[thread_index] = 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint prefix = state[0];
            const uint remaining = state[1];
            const uint shift = uint(digit) * 8;
            // One atomic per distinct digit of a simdgroup: its lanes holding
            // the first pending lane's digit add their count together.
            for (uint base = 0; base < complete; base += THREADS) {
                const uint block = base + thread_index;
                const uint u = block < complete ? bits[block] : 0u;
                const uint value = (u >> shift) & 255u;
                bool pending = block < complete && (u & mask) == prefix;
                for (uint waiting = uint(uint64_t(simd_ballot(pending))); waiting != 0u;
                     waiting = uint(uint64_t(simd_ballot(pending)))) {
                    const uint leader = metal::ctz(waiting);
                    const uint leading = simd_shuffle(value, ushort(leader));
                    const bool same = pending && value == leading;
                    const uint count = metal::popcount(uint(uint64_t(simd_ballot(same))));
                    if (lane == leader)
                        atomic_fetch_add_explicit(reinterpret_cast<threadgroup atomic_uint *>(&bins[leading]), count,
                            memory_order_relaxed);
                    pending = pending && !same;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Thread t holds bin 255 - t: the candidates above it are the
            // scan of the bins before it. The bin holding the `remaining`-th
            // best takes the digit (bin 255 when nothing remains).
            const uint bin = 255 - thread_index;
            const uint count = bins[bin];
            const uint above = indexer::group_exclusive(count, totals, simd, lane, total);
            const bool pick = remaining == 0 ? thread_index == 0 : above < remaining && remaining <= above + count;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (pick) {
                state[0] = prefix | (bin << shift);
                state[1] = remaining - above;
            }
            mask |= 255u << shift;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // T = state[0]; take every block above it and the first state[1]
        // equal ones in block order. Each thread owns a run of whole words.
        const uint threshold = state[0];
        const uint take = state[1];
        const uint per = (written + THREADS - 1) / THREADS;
        const uint word_lo = metal::min(thread_index * per, written);
        const uint word_hi = metal::min(word_lo + per, written);
        uint equal = 0;
        for (uint block = word_lo * 32; block < metal::min(word_hi * 32, complete); ++block)
            equal += bits[block] == threshold ? 1u : 0u;
        uint rank = indexer::group_exclusive(equal, totals, simd, lane, total);
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
    if (SEISMIC_DIM_C == 0)
        return;

    // Segments: the row's keys (its R visible spans; its fresh rows, its
    // tail, attention reads itself) in span order, read where their block is
    // marked (and past the bitmap), as runs of consecutive stored rows within
    // one slab.
    const uint c = uint(SEISMIC_DIM_C);
    const uint slab_rows = uint(SEISMIC_PARAM_SLAB_ROWS);
    device const int *row_visible = visible + row * spans * 2;
    device int *out = segments + row * c * 2;
    // The span offsets [R + 1]: the keys before each span, then the row's.
    device uint *span_first = offsets + row * (spans + 1);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint keys = 0;
    for (uint base = 0; base < spans; base += THREADS) {
        const uint span = base + thread_index;
        const int lo = span < spans ? row_visible[span * 2] : 0;
        const int hi = span < spans ? row_visible[span * 2 + 1] : 0;
        uint chunk = 0;
        const uint before = indexer::group_exclusive(hi > lo ? uint(hi - lo) : 0u, totals, simd, lane, chunk);
        if (span < spans)
            span_first[span] = keys + before;
        keys += chunk;
    }
    if (thread_index == 0)
        span_first[spans] = keys;
    threadgroup_barrier(mem_flags::mem_device);
    // Key k's read flag and stored row: a cursor on the span holding the
    // last key asked, searched again when a key lies outside it.
    uint span = 0;
    auto seek = [&](uint key) {
        if (key < span_first[span] || key >= span_first[span + 1]) {
            // The last span starting at or before `key`: the one holding it.
            uint lo = 0, hi = spans;
            while (hi - lo > 1) {
                const uint middle = (lo + hi) / 2;
                if (span_first[middle] <= key)
                    lo = middle;
                else
                    hi = middle;
            }
            span = lo;
        }
    };
    auto stored = [&](uint key) {
        seek(key);
        return row_visible[span * 2] + int(key - span_first[span]);
    };
    auto read = [&](uint key) {
        const uint block = key / INDEXER_SB;
        return block >= 32 * written || ((bitmap[block / 32] >> (block % 32)) & 1u) != 0u;
    };
    // A read key continues the run of the key before it.
    auto continues = [&](uint key, int at) {
        return key > 0 && at % int(slab_rows) != 0 && read(key - 1) && stored(key - 1) + 1 == at;
    };
    const uint share = (keys + THREADS - 1) / THREADS;
    const uint key_lo = metal::min(thread_index * share, keys);
    const uint key_hi = metal::min(key_lo + share, keys);
    // The runs starting in this thread's keys, then their places.
    uint starts = 0;
    for (uint key = key_lo; key < key_hi;) {
        if (!read(key)) {
            key = metal::min((key / INDEXER_SB + 1) * INDEXER_SB, key_hi);
            continue;
        }
        starts += continues(key, stored(key)) ? 0u : 1u;
        ++key;
    }
    uint runs = 0;
    int run = int(indexer::group_exclusive(starts, totals, simd, lane, runs)) - 1;
    for (uint key = key_lo; key < key_hi;) {
        if (!read(key)) {
            key = metal::min((key / INDEXER_SB + 1) * INDEXER_SB, key_hi);
            continue;
        }
        const int at = stored(key);
        if (!continues(key, at)) {
            ++run;
            if (uint(run) < c)
                out[run * 2] = at;
        }
        const bool ends = key + 1 >= keys || !read(key + 1) || !continues(key + 1, stored(key + 1));
        if (ends && run >= 0 && uint(run) < c)
            out[run * 2 + 1] = at + 1;
        ++key;
    }
    for (uint entry = runs + thread_index; entry < c; entry += THREADS) {
        out[entry * 2] = 0;
        out[entry * 2 + 1] = 0;
    }
}
