// indexer_select on CUDA (contract and portable body in indexer.seismic),
// the Metal launches (`metal/indexer_select.metal`):
//
// append:   block per row; its raw key (widened to F32) and coordinates to
//           the window planes.
// complete: one warp per row that completes a block; the mean of its SB raw
//           keys (F32, members in order: the last SB - 1 rows of its window
//           spans, then its own), RMS-normed and rotated at the coordinates
//           stored with the first, stored in the F32 block plane.
// score:    block per (tile of 8 * 32 blocks, row); the query heads prepared
//           once into shared memory, each warp scores 32 blocks.
// select:   block per row; radix select over the scores' bits finds the K-th
//           best score T, blocks above T and the first equal ones in block
//           order are marked, with every block from the tail on, as the
//           row's bitmap.
// segments: one warp per row (C > 0); the marked blocks' keys mapped through
//           the row's spans, as runs within slabs.
//
// Dynamic extents, strides and parameters read `seismic_words`: every helper
// using one takes `seismic_words_value`.

#include "lib/core/activation.cuh"
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

    __device__ __forceinline__ u32 rows() const {
        u32 total = 0;
        for (u64 span = 0; span < count; ++span) {
            const int lo = spans[span * span_stride];
            const int hi = spans[span * span_stride + bound_stride];
            total += hi > lo ? static_cast<u32>(hi - lo) : 0u;
        }
        return total;
    }

    // The physical row of row `ordinal` of the spans, then of `own` (the
    // row's own, after them).
    __device__ __forceinline__ int row(u32 ordinal, int own) const {
        u32 before = 0;
        for (u64 span = 0; span < count; ++span) {
            const int lo = spans[span * span_stride];
            const int hi = spans[span * span_stride + bound_stride];
            const u32 length = hi > lo ? static_cast<u32>(hi - lo) : 0u;
            if (ordinal < before + length) return lo + static_cast<int>(ordinal - before);
            before += length;
        }
        return own;
    }
};

// Row `row`'s complete blocks: its block spans, then its own block.
__device__ __forceinline__ Spans block_spans(const int *block_visible, u64 row,
                                             const seismic_words_t &seismic_words_value) {
    return Spans{block_visible + row * SEISMIC_BLOCK_VISIBLE_STRIDE_0, SEISMIC_DIM_R,
                 SEISMIC_BLOCK_VISIBLE_STRIDE_1, SEISMIC_BLOCK_VISIBLE_STRIDE_2};
}

// Bitmap words per row (32 blocks each), and the blocks they hold.
__device__ __forceinline__ u32 bitmap_words(const seismic_words_t &seismic_words_value) {
    return static_cast<u32>(SEISMIC_DIM_B);
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
    const u32 earlier = window_spans.rows();
    const u32 skipped = earlier > static_cast<u32>(SB - 1) ? earlier - static_cast<u32>(SB - 1) : 0u;
    const u32 *window = reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_WINDOW_KEYS));
    int first_member = own;
    float sum[E];
#pragma unroll
    for (int i = 0; i < E; ++i) sum[i] = 0.0f;
    for (int member = 0; member < SB; ++member) {
        const int at = member + 1 < SB ? window_spans.row(skipped + static_cast<u32>(member), own) : own;
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
    extern __shared__ float prepared[];  // [H][W]
    const u64 row = blockIdx.y;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const Spans blocks =
        block_spans(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_BLOCK_VISIBLE)), row, seismic_words_value);
    const int own = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_BLOCK_DESTINATIONS))
        [row * SEISMIC_BLOCK_DESTINATIONS_STRIDE_0];
    const u32 complete = blocks.rows() + (own >= 0 ? 1u : 0u);
    const u32 first = blockIdx.x * 256u;
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
    const u32 begin = first + static_cast<u32>(warp) * 32u;
    const u32 last = min(min(first + 256u, complete), begin + 32u);
    for (u32 block = begin; block < last; ++block) {
        const int at = blocks.row(block, own);
        const float *k = reinterpret_cast<const float *>(
            slab::row(block_keys, static_cast<u64>(at), SEISMIC_PARAM_BLOCK_SLAB_ROWS, W * sizeof(float)));
        float kx[E];
#pragma unroll
        for (int i = 0; i < E; ++i) kx[i] = k[lane * E + i];
        float score = 0.0f;
#pragma unroll
        for (int head = 0; head < H; ++head) {
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < E; ++i) dot = fmaf(q[head][i], kx[i], dot);
            score += fmaxf(warp_sum(dot), 0.0f);
        }
        // Scores are >= 0; a zero is stored as +0 so its bits order first.
        const float value = score / root;
        if (lane == 0) scores[row * 32 * SEISMIC_DIM_B + block] = value == 0.0f ? 0.0f : value;
    }
}

extern "C" __global__ void indexer_select_select(SEISMIC_KERNEL_PARAMS) {
    using namespace indexer;
    constexpr u32 THREADS = 256;
    __shared__ u32 histogram[256];
    __shared__ u32 counts[THREADS];
    __shared__ u32 state[2];
    const u64 row = blockIdx.x;
    const u32 thread = threadIdx.x;
    const Spans blocks =
        block_spans(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_BLOCK_VISIBLE)), row, seismic_words_value);
    const int own = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_BLOCK_DESTINATIONS))
        [row * SEISMIC_BLOCK_DESTINATIONS_STRIDE_0];
    const u32 complete = blocks.rows() + (own >= 0 ? 1u : 0u);
    const u32 written = bitmap_words(seismic_words_value);
    u32 *bitmap = reinterpret_cast<u32 *>(SEISMIC_PTR(SEISMIC_RESULT_1_BUFFER)) + row * SEISMIC_RESULT_1_STRIDE_0;
    const u32 *bits = reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_SCORES)) +
                      row * 32 * SEISMIC_DIM_B;
    const int requested = static_cast<int>(static_cast<u32>(SEISMIC_PARAM_SELECTED));
    const u32 selected = requested > 0 ? static_cast<u32>(requested) : 0u;
    if (complete <= selected) {
        for (u32 word = thread; word < written; word += THREADS) bitmap[word] = 0xffffffffu;
        return;
    }
    if (thread == 0) {
        state[0] = 0;
        state[1] = selected;
    }
    u32 mask = 0;
    for (int digit = 3; digit >= 0; --digit) {
        for (u32 bin = thread; bin < 256; bin += THREADS) histogram[bin] = 0;
        __syncthreads();
        const u32 prefix = state[0];
        const u32 shift = static_cast<u32>(digit) * 8u;
        for (u32 block = thread; block < complete; block += THREADS) {
            const u32 u = bits[block];
            if ((u & mask) == prefix) atomicAdd(&histogram[(u >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (thread == 0) {
            u32 remaining = state[1];
            u32 value = 0;
            for (int bin = 255; bin >= 0; --bin) {
                const u32 count = histogram[bin];
                if (count >= remaining) {
                    value = static_cast<u32>(bin);
                    break;
                }
                remaining -= count;
            }
            state[0] = prefix | (value << shift);
            state[1] = remaining;
        }
        mask |= 255u << shift;
        __syncthreads();
    }
    const u32 threshold = state[0];
    const u32 take = state[1];
    const u32 per = (written + THREADS - 1) / THREADS;
    const u32 word_lo = min(thread * per, written);
    const u32 word_hi = min(word_lo + per, written);
    u32 equal = 0;
    for (u32 block = word_lo * 32; block < min(word_hi * 32, complete); ++block) equal += bits[block] == threshold;
    counts[thread] = equal;
    __syncthreads();
    u32 rank = 0;
    for (u32 other = 0; other < thread; ++other) rank += counts[other];
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

extern "C" __global__ void indexer_select_segments(SEISMIC_KERNEL_PARAMS) {
    using namespace indexer;
    const u64 row = blockIdx.x;
    const int lane = threadIdx.x % 32;
    // The row's keys in its R visible spans (its fresh rows, its tail,
    // attention reads itself).
    const Spans keys{reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_VISIBLE)) +
                         row * SEISMIC_VISIBLE_STRIDE_0,
                     SEISMIC_DIM_R, SEISMIC_VISIBLE_STRIDE_1, SEISMIC_VISIBLE_STRIDE_2};
    auto bound = [&](u64 span, int which) {
        return span < keys.count ? keys.spans[span * keys.span_stride + which * keys.bound_stride] : 0;
    };
    int *segments = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const u32 words = bitmap_words(seismic_words_value);
    const u32 *bitmap = reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_RESULT_1_BUFFER)) + row * SEISMIC_RESULT_1_STRIDE_0;
    const u32 c = static_cast<u32>(SEISMIC_DIM_C);
    const int slab_rows = static_cast<int>(SEISMIC_PARAM_SLAB_ROWS);
    auto store = [&](u32 entry, int start, int end) {
        const u64 at = row * SEISMIC_RESULT_0_STRIDE_0 + entry * SEISMIC_RESULT_0_STRIDE_1;
        segments[at] = start;
        segments[at + SEISMIC_RESULT_0_STRIDE_2] = end;
    };
    const u32 total = keys.rows();
    u32 count = 0;
    int start = 0, end = 0;
    bool open = false;
    u32 last_key = 0;
    u64 span = 0;
    u32 span_first = 0;
    int span_lo = bound(0, 0);
    int span_hi = bound(0, 1);
    const u32 last_word = min(words, (total + 32u * SB - 1) / (32u * SB));
    for (u32 base = 0; base < last_word; base += 32) {
        const u32 word_index = base + lane;
        const u32 word = word_index < last_word ? bitmap[word_index] : 0u;
        for (u32 pending = __ballot_sync(0xffffffffu, word != 0u); pending != 0u; pending &= pending - 1u) {
            const int at = __ffs(pending) - 1;
            const u32 value = __shfl_sync(0xffffffffu, word, at);
            if (lane != 0) continue;
            for (u32 bit = 0; bit < 32; ++bit) {
                if ((value & (1u << bit)) == 0u) continue;
                const u32 block = (base + at) * 32 + bit;
                for (u32 key = block * SB; key < min((block + 1) * SB, total); ++key) {
                    while (key >= span_first + static_cast<u32>(max(span_hi - span_lo, 0))) {
                        span_first += static_cast<u32>(max(span_hi - span_lo, 0));
                        ++span;
                        span_lo = bound(span, 0);
                        span_hi = bound(span, 1);
                    }
                    const int stored = span_lo + static_cast<int>(key - span_first);
                    if (open && key == last_key + 1 && stored == end && stored % slab_rows != 0) {
                        ++end;
                    } else {
                        if (open && count < c) store(count++, start, end);
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
        if (open && count < c) store(count++, start, end);
        for (; count < c; ++count) store(count, 0, 0);
    }
}
