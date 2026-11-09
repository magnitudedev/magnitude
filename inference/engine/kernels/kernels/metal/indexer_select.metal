// indexer_select on Metal (contract and portable body in indexer.seismic).
//
// append:   threadgroup per row; its raw key (widened to F32) and
//           coordinates to the window planes.
// complete: one simdgroup per row that completes a block; the mean of its SB
//           raw keys (F32, members in order: the last SB - 1 rows of its
//           window spans, then its own), RMS-normed with the key norm and
//           rotated at the coordinates stored with the first, stored in the
//           F32 block plane.
// score:    threadgroup per (row, tile of 8 * INDEXER_BATCH blocks); the
//           row's query heads are prepared into threadgroup memory, then each
//           simdgroup scores INDEXER_BATCH blocks over its lanes, their keys'
//           loads issued together.
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
#include "lib/core/head.h"
#include "lib/core/rotary.h"
#include "lib/indexer/indexer.h"
#include <seismic/slab.h>

#define INDEXER_W uint(2 * SEISMIC_DIM_P + SEISMIC_DIM_S)
#define INDEXER_E (INDEXER_W / 32)
#define INDEXER_H uint(SEISMIC_DIM_H)
#define INDEXER_SB uint(SEISMIC_DIM_SB)
#define INDEXER_UNROLL _Pragma("clang loop unroll(full)")
// Blocks a score simdgroup takes, their key loads in flight together.
#define INDEXER_BATCH 4u

typedef element::Act::native Scalar;
// A key or query head row's rotary pairs (p, p + P) over its first 2P
// columns: the portable `head_rotary`, unrounded.
typedef rotary::layout<INDEXER_W, SEISMIC_DIM_P, 0, false> key_layout;

namespace indexer {

// The rows of a row's spans: every lane of a simdgroup calls it, each
// taking the spans lane, lane + 32, ...
inline uint simd_span_rows(device const int *spans, uint count, uint lane) {
    uint total = 0;
    for (uint span = lane; span < count; span += 32)
        total += span_length(spans[span * 2], spans[span * 2 + 1]);
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
            const uint length = span_length(lo, spans[span * 2 + 1]);
            if (ordinal < before + length)
                return lo + int(ordinal - before);
            before += length;
        }
        return own;
    }
};

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
    head::rms_norm(x, key_norm, as_type<float>(uint(SEISMIC_PARAM_EPSILON)), lane);
    const key_layout::table table(first_coordinates, rotary_components, rotary_frequencies, rotary_amplitudes, lane);
    key_layout::rotate(table, lane, x);
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
    const uint complete = indexer::simd_span_rows(row_spans, spans, lane);
    const uint first = group.x * (8 * INDEXER_BATCH);
    if (first >= complete)
        return;
    // The row's query heads, prepared in F32; one table rotates them all.
    const key_layout::table table(coordinates + row * 4, rotary_components, rotary_frequencies, rotary_amplitudes,
        lane);
    for (uint head = simd; head < INDEXER_H; head += 8) {
        float x[INDEXER_E];
        device const Scalar *raw = query + (row * INDEXER_H + head) * INDEXER_W;
        INDEXER_UNROLL
        for (uint i = 0; i < INDEXER_E; ++i)
            x[i] = float(raw[lane * INDEXER_E + i]);
        head::rms_norm(x, query_norm, as_type<float>(uint(SEISMIC_PARAM_EPSILON)), lane);
        key_layout::rotate(table, lane, x);
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
    const uint begin = first + simd * INDEXER_BATCH;
    const uint end = metal::min(begin + INDEXER_BATCH, complete);
    if (begin >= end)
        return;
    indexer::span_cursor cursor;
    float kx[INDEXER_BATCH][INDEXER_E];
    INDEXER_UNROLL
    for (uint j = 0; j < INDEXER_BATCH; ++j) {
        // A block past the row's last repeats the last: its score is not stored.
        const int at = cursor.row(row_spans, spans, metal::min(begin + j, end - 1), -1);
        device const float *k = slab::row<float>(block_keys, ulong(at), slab_rows, INDEXER_W);
        INDEXER_UNROLL
        for (uint i = 0; i < INDEXER_E; ++i)
            kx[j][i] = k[lane * INDEXER_E + i];
    }
    INDEXER_UNROLL
    for (uint j = 0; j < INDEXER_BATCH; ++j) {
        float score = 0.0f;
        INDEXER_UNROLL
        for (uint head = 0; head < INDEXER_H; ++head) {
            float dot = 0.0f;
            INDEXER_UNROLL
            for (uint i = 0; i < INDEXER_E; ++i)
                dot = metal::fma(q[head][i], kx[j][i], dot);
            score += metal::max(simd_sum(dot), 0.0f);
        }
        if (lane == 0 && begin + j < end)
            scores[row * nb + begin + j] = score / root + 0.0f;
    }
}

kernel void indexer_select_select(
    device const int *block_visible [[buffer(SEISMIC_BUFFER_BLOCK_VISIBLE)]],
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
    const uint complete = indexer::simd_span_rows(block_visible + row * spans * 2, spans, lane);
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
        const uint before = indexer::group_exclusive(indexer::span_length(lo, hi), totals, simd, lane, chunk);
        if (span < spans)
            span_first[span] = keys + before;
        keys += chunk;
    }
    if (thread_index == 0)
        span_first[spans] = keys;
    threadgroup_barrier(mem_flags::mem_device);
    // Thread t's keys [key_lo, key_hi). A key is read when its block is
    // marked (or lies past the bitmap) and is stored at its span's first row
    // plus its place in the span. A run starts at a read key unless the key
    // before is read, stored one row before it, and the key does not begin a
    // slab. The walk starts at the key before the share (whether a run
    // continues into it) and ends at the key after (whether the share's last
    // run ends in it). It reads each block's bit once and skips the rest of
    // an unread block, and finds its first span by bisection, then advances
    // span by span. With `emit` it writes the share's runs from index
    // `opened` + 1 (`opened` is the run open at key_lo, begun before it);
    // it returns the runs starting in the share.
    const uint share = (keys + THREADS - 1) / THREADS;
    const uint key_lo = metal::min(thread_index * share, keys);
    const uint key_hi = metal::min(key_lo + share, keys);
    auto walk = [&](bool emit, int opened) {
        uint starts = 0;
        if (key_lo >= key_hi)
            return starts;
        const uint kstart = key_lo > 0 ? key_lo - 1 : 0;
        uint span = 0;
        for (uint high = spans - 1; span < high;) {
            const uint middle = (span + high + 1) / 2;
            if (span_first[middle] <= kstart)
                span = middle;
            else
                high = middle - 1;
        }
        uint first = span_first[span];
        uint next = span_first[span + 1];
        int lo = row_visible[span * 2];
        uint block = 0xffffffffu;
        bool block_read = false;
        bool previous_read = false;
        int previous_stored = 0;
        int index = opened;
        const uint kend = metal::min(key_hi + 1, keys);
        for (uint key = kstart; key < kend; ++key) {
            if (key / INDEXER_SB != block) {
                block = key / INDEXER_SB;
                block_read = block >= 32 * written || ((bitmap[block / 32] >> (block % 32)) & 1u) != 0u;
            }
            if (!block_read) {
                // The run through the previous key ends there; the block's
                // other keys are unread too.
                if (emit && key > key_lo && previous_read && uint(index) < c)
                    out[index * 2 + 1] = previous_stored + 1;
                previous_read = false;
                key = metal::min((block + 1) * INDEXER_SB, kend) - 1;
                continue;
            }
            while (key >= next) {
                ++span;
                first = next;
                next = span_first[span + 1];
                lo = row_visible[span * 2];
            }
            const int at = lo + int(key - first);
            const bool continues = previous_read && previous_stored + 1 == at && at % int(slab_rows) != 0;
            if (emit && key > key_lo && previous_read && !continues && uint(index) < c)
                out[index * 2 + 1] = previous_stored + 1;
            if (key >= key_lo && key < key_hi && !continues) {
                ++starts;
                ++index;
                if (emit && uint(index) < c)
                    out[index * 2] = at;
            }
            previous_read = true;
            previous_stored = at;
        }
        // The row's last key ends its run.
        if (emit && key_hi == keys && previous_read && uint(index) < c)
            out[index * 2 + 1] = previous_stored + 1;
        return starts;
    };
    uint runs = 0;
    const int opened = int(indexer::group_exclusive(walk(false, 0), totals, simd, lane, runs)) - 1;
    walk(true, opened);
    for (uint entry = runs + thread_index; entry < c; entry += THREADS) {
        out[entry * 2] = 0;
        out[entry * 2 + 1] = 0;
    }
}
