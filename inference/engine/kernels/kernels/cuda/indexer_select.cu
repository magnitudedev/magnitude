// indexer_select on CUDA (contract and portable body in indexer.seismic),
// the Metal launches (`metal/indexer_select.metal`):
//
// append:   block per row; its raw key (widened to F32) and coordinates to
//           the window planes.
// complete: one warp per row that completes a block; the mean of its SB raw
//           keys (F32, members in order: the last SB - 1 rows of its window
//           spans, then its own), RMS-normed and rotated at the coordinates
//           stored with the first, stored in the F32 block plane.
// score:    block per (tile of 8 * kScoreBatch blocks, row); the query heads
//           prepared into shared memory, each warp scores kScoreBatch blocks,
//           their key loads in flight together, walking the block spans from
//           its first block's.
// select:   block of 256 threads per row; a radix select over the scores'
//           bits finds the K-th best score T (each pass's histogram takes one
//           atomic per distinct digit of a warp, the digit is found by a scan
//           of the bins), blocks above T and the first equal ones in block
//           order (ranked by a scan of each thread's count) are marked, with
//           every block from the tail on, as the row's bitmap. With C > 0 it
//           then emits the row's runs: the span offsets are scanned into
//           scratch, each thread counts the runs starting in its share
//           of the keys, and a scan of the counts places each run.
//
// The warp-cooperative span walks (`reduce::span_total`,
// `reduce::span_locate`) and the block scans are lib/core/reduce.cuh's.
//
// Dynamic extents, strides and parameters read `seismic_words`: every helper
// using one takes `seismic_words_value`.

#include "lib/core/activation.cuh"
#include "lib/core/reduce.cuh"
#include "lib/indexer/indexer.cuh"
#include <seismic/slab.cuh>

namespace indexer {

typedef unsigned int u32;
typedef unsigned long long u64;
typedef element::Act Act;
typedef Act::storage Storage;

constexpr int W = 2 * SEISMIC_DIM_P + SEISMIC_DIM_S;
constexpr int E = W / 32;
constexpr int P = SEISMIC_DIM_P;
constexpr int H = SEISMIC_DIM_H;
constexpr int SB = SEISMIC_DIM_SB;
static_assert(W % 32 == 0, "indexer head width must be a multiple of 32");
static_assert(P % E == 0, "each rotary half must cover whole lanes");

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    return value;
}

// Lane `lane`'s E columns of one head row, RMS-normed with `norm` and its
// first 2P columns rotated at `coordinates`: the portable `head_rotary`,
// unrounded. The whole warp calls it.
__device__ __forceinline__ void norm_rotary(float (&x)[E], const float *norm, const int *coordinates,
                                            const int *components, const float *frequencies,
                                            const float *amplitudes, float epsilon, int lane) {
    float squares = 0.0f;
#pragma unroll
    for (int i = 0; i < E; ++i) squares = fmaf(x[i], x[i], squares);
    squares = warp_sum(squares);
    const float inverse = rsqrtf(squares / static_cast<float>(W) + epsilon);
#pragma unroll
    for (int i = 0; i < E; ++i) x[i] = x[i] * inverse * norm[lane * E + i];
    if (P == 0) return;
    constexpr int half_lanes = P / E;
    const int partner_lane = lane < half_lanes ? lane + half_lanes : (lane < 2 * half_lanes ? lane - half_lanes : lane);
    float partner[E];
#pragma unroll
    for (int i = 0; i < E; ++i) partner[i] = __shfl_sync(0xffffffffu, x[i], partner_lane);
    if (lane >= 2 * half_lanes) return;
#pragma unroll
    for (int i = 0; i < E; ++i) {
        const int column = lane * E + i;
        const int pair = column % (P > 0 ? P : 1);
        float sine, cosine;
        sincosf(static_cast<float>(coordinates[components[pair]]) * frequencies[pair], &sine, &cosine);
        cosine *= amplitudes[pair];
        sine *= amplitudes[pair];
        x[i] = column < P ? x[i] * cosine - partner[i] * sine : x[i] * cosine + partner[i] * sine;
    }
}

// A row's spans: `count` [lo, hi) pairs from `spans`, `span_stride` and
// `bound_stride` apart.
struct Spans {
    const int *spans;
    u64 count, span_stride, bound_stride;

    __device__ __forceinline__ int lo(u64 span) const { return spans[span * span_stride]; }
    __device__ __forceinline__ u32 length(u64 span) const {
        return indexer::span_length(lo(span), spans[span * span_stride + bound_stride]);
    }

    // The rows of the spans (`reduce::span_total`; the whole warp calls it).
    __device__ __forceinline__ u32 rows(int lane) const {
        return static_cast<u32>(reduce::span_total(count, [&](u64 span) { return length(span); }, lane));
    }
};

// A walk over the rows of a row's spans in ordinal order: the span holding
// the current rows, its first row's ordinal and the ordinal past its rows.
struct Cursor {
    Spans spans;
    u64 span;
    int lo;
    u32 first, end;

    // At the span holding row `ordinal` (`reduce::span_locate`; the whole warp
    // calls it).
    __device__ __forceinline__ Cursor(const Spans &walked, u32 ordinal, int lane) : spans(walked) {
        const reduce::Located at =
            reduce::span_locate(spans.count, ordinal, [&](u64 index) { return spans.length(index); }, lane);
        enter(at.span, static_cast<u32>(at.offset));
    }

    __device__ __forceinline__ void enter(u64 index, u32 offset) {
        span = index;
        first = offset;
        lo = span < spans.count ? spans.lo(span) : 0;
        end = span < spans.count ? first + spans.length(span) : first;
    }

    // The physical row of row `ordinal` of the spans (at or past the rows
    // asked for before), then `own` (the row's own, after them).
    __device__ __forceinline__ int row(u32 ordinal, int own) {
        while (span < spans.count && ordinal >= end) enter(span + 1, end);
        return span < spans.count ? lo + static_cast<int>(ordinal - first) : own;
    }
};

// Row `row`'s complete blocks: its block spans, its own block among them.
__device__ __forceinline__ Spans block_spans(const int *block_visible, u64 row,
                                             const seismic_words_t &seismic_words_value) {
    return Spans{block_visible + row * SEISMIC_BLOCK_VISIBLE_STRIDE_0, SEISMIC_DIM_R,
                 SEISMIC_BLOCK_VISIBLE_STRIDE_1, SEISMIC_BLOCK_VISIBLE_STRIDE_2};
}

// Bitmap words per row (32 blocks each), and the blocks they hold.
__device__ __forceinline__ u32 bitmap_words(const seismic_words_t &seismic_words_value) {
    return static_cast<u32>(SEISMIC_DIM_B);
}

constexpr u32 THREADS = 256;

// Marks the bitmap words [0, words) of a row with more complete blocks than
// `selected`, from the bits of its scores `bits` (which order as the
// non-negative scores do). Shared memory: `bins` [256], `partials` the block
// scans' warp totals, `state` (prefix, count still to take). The whole block
// calls it.
__device__ __forceinline__ void mark(const u32 *bits, u32 complete, u32 selected, u32 *bitmap, u32 words,
                                     u32 *bins, u32 *partials, u32 *state) {
    const u32 thread = threadIdx.x;
    const int lane = static_cast<int>(thread % 32);
    if (thread == 0) {
        state[0] = 0;
        state[1] = selected;
    }
    // Four 8-bit passes, most significant first: the K-th best score's
    // digit is the bin holding the remaining-th best of the scores matching
    // the digits found so far.
    u32 mask = 0;
    for (int digit = 3; digit >= 0; --digit) {
        const u32 shift = static_cast<u32>(digit) * 8u;
        bins[thread] = 0;
        __syncthreads();
        const u32 prefix = state[0];
        const u32 remaining = state[1];
        // One atomic per distinct digit of a warp: its lowest lane adds the
        // count of the warp's lanes holding it.
        for (u32 base = 0; base < complete; base += THREADS) {
            const u32 block = base + thread;
            const u32 u = block < complete ? bits[block] : 0u;
            const bool counted = block < complete && (u & mask) == prefix;
            const u32 value = counted ? (u >> shift) & 255u : 256u;
            const u32 peers = __match_any_sync(0xffffffffu, value);
            if (counted && lane == __ffs(peers) - 1) atomicAdd(&bins[value], static_cast<u32>(__popc(peers)));
        }
        __syncthreads();
        // Thread t holds bin 255 - t: the scan counts the matching scores in
        // the bins above it, and the digit is the highest bin whose scores
        // reach the remaining-th.
        const u32 bin = 255u - thread;
        const u32 count = bins[bin];
        u32 matching;
        const u32 above = reduce::group_exclusive_add(count, partials, matching);
        if (remaining <= above + count && (thread == 0 || above < remaining)) {
            state[0] = prefix | (bin << shift);
            state[1] = remaining - above;
        }
        mask |= 255u << shift;
    }
    __syncthreads();
    const u32 threshold = state[0];
    const u32 take = state[1];
    // Each thread marks whole words; the scores equal to the threshold are
    // ranked in block order by a scan of each thread's count.
    const u32 per = (words + THREADS - 1) / THREADS;
    const u32 word_lo = min(thread * per, words);
    const u32 word_hi = min(word_lo + per, words);
    u32 equal = 0;
    for (u32 block = word_lo * 32; block < min(word_hi * 32, complete); ++block) equal += bits[block] == threshold;
    u32 equals;
    u32 rank = reduce::group_exclusive_add(equal, partials, equals);
    for (u32 word = word_lo; word < word_hi; ++word) {
        u32 value = 0;
        for (u32 bit = 0; bit < 32; ++bit) {
            const u32 block = word * 32 + bit;
            bool read = block >= complete;
            if (!read) {
                const u32 u = bits[block];
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

// A row's runs: entry e's [start, end) at `out` + e * `entry_stride`, its end
// `bound_stride` past its start; runs past `capacity` are dropped.
struct Runs {
    int *out;
    u64 entry_stride, bound_stride;
    u32 capacity;

    __device__ __forceinline__ void start(u32 entry, int stored) const {
        if (entry < capacity) out[entry * entry_stride] = stored;
    }
    __device__ __forceinline__ void end(u32 entry, int stored) const {
        if (entry < capacity) out[entry * entry_stride + bound_stride] = stored;
    }
};

// The stored rows of a row's keys in key order, from the span offsets
// `span_first` [count + 1]: at the span holding the first key asked for (the
// last span starting at or before it: an empty span ends where it starts),
// then past each span's end.
struct Keys {
    const Spans &spans;
    const u32 *span_first;
    u64 span;
    int lo;
    u32 first, end;

    // `key` is below the total, span_first[count].
    __device__ __forceinline__ Keys(const Spans &walked, const u32 *offsets, u32 key)
        : spans(walked), span_first(offsets) {
        u64 low = 0, high = spans.count;
        while (high - low > 1) {
            const u64 middle = (low + high) / 2;
            if (span_first[middle] <= key) low = middle;
            else high = middle;
        }
        enter(low);
    }

    __device__ __forceinline__ void enter(u64 index) {
        span = index;
        first = span_first[span];
        end = span_first[span + 1];
        lo = spans.lo(span);
    }

    // The stored row of `key` (at or past the keys asked for before).
    __device__ __forceinline__ int stored(u32 key) {
        while (key >= end) enter(span + 1);
        return lo + static_cast<int>(key - first);
    }
};

// Emits a row's runs: the stored rows of its keys `keys` (in key order, the
// key k in block k / SB) where the row reads them, as [start, end) runs each
// within one slab of `slab_rows` rows, then empty runs up to the capacity. A
// key is read when its block's bit is set in the row's `bitmap` [words], or
// past the bitmap's blocks. Shared memory: `partials` the block scans' warp
// totals, `span_first` [count + 1]. The whole block calls it.
__device__ __forceinline__ void emit(const Spans &keys, const u32 *bitmap, u32 words, const Runs &runs,
                                     int slab_rows, u32 *partials, u32 *span_first) {
    const u32 thread = threadIdx.x;
    // span_first[s]: the keys of the spans before s; span_first[count]: the
    // total.
    u32 total = 0;
    for (u64 base = 0; base < keys.count; base += THREADS) {
        const u64 span = base + thread;
        u32 chunk;
        const u32 before = reduce::group_exclusive_add(span < keys.count ? keys.length(span) : 0u, partials, chunk);
        if (span < keys.count) span_first[span] = total + before;
        total += chunk;
    }
    if (thread == 0) span_first[keys.count] = total;
    __syncthreads();
    const u32 blocks = 32u * words;
    const u32 block_rows = static_cast<u32>(SB);
    auto read = [&](u32 key) {
        const u32 block = key / block_rows;
        return block >= blocks || ((bitmap[block / 32] >> (block % 32)) & 1u) != 0u;
    };
    // Thread t's keys [key_lo, key_hi). A run starts at a read key unless the
    // key before is read, stored one row before it, and the key does not
    // begin a slab; it ends at a read key the key after does not continue.
    // The walk reads the key before the share (whether a run continues into
    // it) and the key after (whether the share's last run ends in it), and
    // skips the rest of an unread block.
    const u32 share = (total + THREADS - 1) / THREADS;
    const u32 key_lo = min(thread * share, total);
    const u32 key_hi = min(key_lo + share, total);
    // The first walk counts the share's run starts, the second (`write`)
    // places its runs from `entry`.
    u32 starts = 0;
    u32 entry = 0;
    auto walk = [&](bool write) {
        auto start = [&](int stored) {
            if (write)
                runs.start(++entry, stored);
            else
                ++starts;
        };
        auto end = [&](int stored) {
            if (write)
                runs.end(entry, stored);
        };
        if (key_lo >= key_hi) return;
        Keys cursor(keys, span_first, key_lo > 0 ? key_lo - 1 : key_lo);
        bool previous = key_lo > 0 && read(key_lo - 1);
        int previous_stored = previous ? cursor.stored(key_lo - 1) : 0;
        for (u32 key = key_lo;;) {
            if (key >= total || !read(key)) {
                if (previous && key > key_lo) end(previous_stored + 1);
                if (key == key_hi) return;
                previous = false;
                key = min((key / block_rows + 1) * block_rows, key_hi);
                continue;
            }
            const int stored = cursor.stored(key);
            const bool continues = previous && previous_stored + 1 == stored && stored % slab_rows != 0;
            if (!continues && previous && key > key_lo) end(previous_stored + 1);
            if (key == key_hi) return;
            if (!continues) start(stored);
            previous = true;
            previous_stored = stored;
            ++key;
        }
    };
    walk(false);
    u32 count;
    // The run open at the share's first key, if any, began in an earlier
    // share: its index is the share's first minus one.
    entry = reduce::group_exclusive_add(starts, partials, count) - 1u;
    walk(true);
    for (u32 empty = count + thread; empty < runs.capacity; empty += THREADS) {
        runs.start(empty, 0);
        runs.end(empty, 0);
    }
}

} // namespace indexer

extern "C" __global__ void indexer_select_append(SEISMIC_KERNEL_PARAMS) {
    using namespace indexer;
    const u64 row = blockIdx.x;
    const int *destinations = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_WINDOW_DESTINATIONS));
    const int destination = destinations[row * SEISMIC_WINDOW_DESTINATIONS_STRIDE_0];
    if (destination < 0) return;
    const Storage *key = reinterpret_cast<const Storage *>(SEISMIC_PTR(SEISMIC_BUFFER_KEY));
    float *target = reinterpret_cast<float *>(
        slab::row(reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_WINDOW_KEYS)),
                  static_cast<u64>(destination), SEISMIC_PARAM_WINDOW_SLAB_ROWS, W * sizeof(float)));
    for (int column = threadIdx.x; column < W; column += blockDim.x)
        target[column] = Act::load(key[row * SEISMIC_KEY_STRIDE_0 + column * SEISMIC_KEY_STRIDE_1]);
    if (threadIdx.x < 4) {
        const int *coordinates = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_COORDINATES));
        int *stored = reinterpret_cast<int *>(
            slab::row(reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_WINDOW_COORDINATES)),
                      static_cast<u64>(destination), SEISMIC_PARAM_WINDOW_SLAB_ROWS, 4 * sizeof(int)));
        stored[threadIdx.x] =
            coordinates[row * SEISMIC_COORDINATES_STRIDE_0 + threadIdx.x * SEISMIC_COORDINATES_STRIDE_1];
    }
}

extern "C" __global__ void indexer_select_complete(SEISMIC_KERNEL_PARAMS) {
    using namespace indexer;
    const u64 row = blockIdx.x;
    const int lane = threadIdx.x % 32;
    const int *block_destinations = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_BLOCK_DESTINATIONS));
    const int destination = block_destinations[row * SEISMIC_BLOCK_DESTINATIONS_STRIDE_0];
    if (destination < 0) return;
    // The block's raw keys: the last SB - 1 rows of the window spans, then
    // the row's own.
    const Spans window_spans{reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_WINDOW_VISIBLE)) +
                                 row * SEISMIC_WINDOW_VISIBLE_STRIDE_0,
                             SEISMIC_DIM_R, SEISMIC_WINDOW_VISIBLE_STRIDE_1, SEISMIC_WINDOW_VISIBLE_STRIDE_2};
    const int own = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_WINDOW_DESTINATIONS))
        [row * SEISMIC_WINDOW_DESTINATIONS_STRIDE_0];
    const u32 earlier = window_spans.rows(lane);
    const u32 skipped = earlier > static_cast<u32>(SB - 1) ? earlier - static_cast<u32>(SB - 1) : 0u;
    Cursor members(window_spans, skipped, lane);
    const u32 *window = reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_WINDOW_KEYS));
    int first_member = own;
    float sum[E];
#pragma unroll
    for (int i = 0; i < E; ++i) sum[i] = 0.0f;
    for (int member = 0; member < SB; ++member) {
        const int at = member + 1 < SB ? members.row(skipped + static_cast<u32>(member), own) : own;
        if (member == 0) first_member = at;
        const float *raw = reinterpret_cast<const float *>(
            slab::row(window, static_cast<u64>(at), SEISMIC_PARAM_WINDOW_SLAB_ROWS, W * sizeof(float)));
#pragma unroll
        for (int i = 0; i < E; ++i) sum[i] += raw[lane * E + i];
    }
    float x[E];
#pragma unroll
    for (int i = 0; i < E; ++i) x[i] = sum[i] / static_cast<float>(SB);
    const int *coordinates = reinterpret_cast<const int *>(
        slab::row(reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_WINDOW_COORDINATES)),
                  static_cast<u64>(first_member), SEISMIC_PARAM_WINDOW_SLAB_ROWS, 4 * sizeof(int)));
    norm_rotary(x, reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_KEY_NORM)), coordinates,
                reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_COMPONENTS)),
                reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_FREQUENCIES)),
                reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_AMPLITUDES)),
                element::word_f32(SEISMIC_PARAM_EPSILON), lane);
    float *target = reinterpret_cast<float *>(
        slab::row(reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_BLOCK_KEYS)), static_cast<u64>(destination),
                  SEISMIC_PARAM_BLOCK_SLAB_ROWS, W * sizeof(float)));
#pragma unroll
    for (int i = 0; i < E; ++i) target[lane * E + i] = x[i];
}

extern "C" __global__ void indexer_select_score(SEISMIC_KERNEL_PARAMS) {
    using namespace indexer;
    // Blocks a warp takes, their key loads in flight together.
    constexpr u32 kScoreBatch = 4;
    extern __shared__ float prepared[];  // [H][W]
    const u64 row = blockIdx.y;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const Spans blocks =
        block_spans(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_BLOCK_VISIBLE)), row, seismic_words_value);
    const u32 complete = blocks.rows(lane);
    const u32 first = blockIdx.x * (8u * kScoreBatch);
    if (first >= complete) return;
    const Storage *query = reinterpret_cast<const Storage *>(SEISMIC_PTR(SEISMIC_BUFFER_QUERY));
    const int *coordinates = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_COORDINATES));
    for (int head = warp; head < H; head += 8) {
        float x[E];
#pragma unroll
        for (int i = 0; i < E; ++i)
            x[i] = Act::load(query[row * SEISMIC_QUERY_STRIDE_0 + head * SEISMIC_QUERY_STRIDE_1 +
                                   static_cast<u64>(lane * E + i) * SEISMIC_QUERY_STRIDE_2]);
        norm_rotary(x, reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_QUERY_NORM)),
                    coordinates + row * SEISMIC_COORDINATES_STRIDE_0,
                    reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_COMPONENTS)),
                    reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_FREQUENCIES)),
                    reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_AMPLITUDES)),
                    element::word_f32(SEISMIC_PARAM_EPSILON), lane);
#pragma unroll
        for (int i = 0; i < E; ++i) prepared[head * W + lane * E + i] = x[i];
    }
    __syncthreads();
    float q[H][E];
#pragma unroll
    for (int head = 0; head < H; ++head)
#pragma unroll
        for (int i = 0; i < E; ++i) q[head][i] = prepared[head * W + lane * E + i];
    const u32 *block_keys = reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_BLOCK_KEYS));
    float *scores = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_SCORES));
    const float root = sqrtf(static_cast<float>(W));
    const u32 begin = first + static_cast<u32>(warp) * kScoreBatch;
    const u32 end = min(begin + kScoreBatch, complete);
    if (begin >= end) return;
    Cursor cursor(blocks, begin, lane);
    float kx[kScoreBatch][E];
#pragma unroll
    for (u32 j = 0; j < kScoreBatch; ++j) {
        // A block past the row's last repeats the last: its score is not stored.
        const int at = cursor.row(min(begin + j, end - 1u), -1);
        const float *k = reinterpret_cast<const float *>(
            slab::row(block_keys, static_cast<u64>(at), SEISMIC_PARAM_BLOCK_SLAB_ROWS, W * sizeof(float)));
#pragma unroll
        for (int i = 0; i < E; ++i) kx[j][i] = k[lane * E + i];
    }
#pragma unroll
    for (u32 j = 0; j < kScoreBatch; ++j) {
        float score = 0.0f;
#pragma unroll
        for (int head = 0; head < H; ++head) {
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < E; ++i) dot = fmaf(q[head][i], kx[j][i], dot);
            score += fmaxf(warp_sum(dot), 0.0f);
        }
        // Scores are >= 0; a zero is stored as +0 so its bits order first.
        const float value = score / root;
        if (lane == 0 && begin + j < end) scores[row * 32 * SEISMIC_DIM_B + begin + j] = value == 0.0f ? 0.0f : value;
    }
}


extern "C" __global__ void indexer_select_select(SEISMIC_KERNEL_PARAMS) {
    using namespace indexer;
    extern __shared__ u32 select_shared[];
    u32 *bins = select_shared;          // [256]
    u32 *partials = bins + 256;         // [8]: the scans' warp sums
    u32 *state = partials + 8;          // [4]
    const u64 row = blockIdx.x;
    const int lane = static_cast<int>(threadIdx.x % 32);
    const Spans blocks =
        block_spans(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_BLOCK_VISIBLE)), row, seismic_words_value);
    const u32 complete = blocks.rows(lane);
    const u32 words = bitmap_words(seismic_words_value);
    u32 *bitmap = reinterpret_cast<u32 *>(SEISMIC_PTR(SEISMIC_RESULT_1_BUFFER)) + row * SEISMIC_RESULT_1_STRIDE_0;
    const int requested = static_cast<int>(static_cast<u32>(SEISMIC_PARAM_SELECTED));
    const u32 selected = requested > 0 ? static_cast<u32>(requested) : 0u;
    if (complete <= selected) {
        for (u32 word = threadIdx.x; word < words; word += THREADS) bitmap[word] = 0xffffffffu;
    } else {
        mark(reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_SCORES)) + row * 32 * SEISMIC_DIM_B,
             complete, selected, bitmap, words, bins, partials, state);
    }
    const u32 capacity = static_cast<u32>(SEISMIC_DIM_C);
    if (capacity == 0) return;
    // The row's keys in its R visible spans (its fresh rows, its tail,
    // attention reads itself), read where the bitmap this block wrote marks
    // their blocks.
    __syncthreads();
    const Spans keys{reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_VISIBLE)) + row * SEISMIC_VISIBLE_STRIDE_0,
                     SEISMIC_DIM_R, SEISMIC_VISIBLE_STRIDE_1, SEISMIC_VISIBLE_STRIDE_2};
    emit(keys, bitmap, words,
         Runs{reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)) + row * SEISMIC_RESULT_0_STRIDE_0,
              SEISMIC_RESULT_0_STRIDE_1, SEISMIC_RESULT_0_STRIDE_2, capacity},
         static_cast<int>(SEISMIC_PARAM_SLAB_ROWS), partials,
         reinterpret_cast<u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_SPANS)) + row * (SEISMIC_DIM_R + 1));
}
