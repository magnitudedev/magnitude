// readout_top_rows: level one of a certified selection (`readout.seismic`):
// the projection library's head GEMV (or batched GEMV) over the 4-bit view
// of a progressive head (`lib/readout/progressive.h`) with the bound
// epilogue. Every threadgroup normalizes its rows in place and forms their
// lengths |x| over the rounded values the GEMV stages; threadgroup 0 also
// publishes the rows' features (A) and lengths. Each vocabulary row's
// lower-bound score raises its output row's threshold (a threadgroup atomic
// maximum, then the sync-scratch keys); the last threadgroup to arrive
// publishes the thresholds.
#include "lib/readout/progressive.h"

typedef ELEMENT_OF(SEISMIC_NORM) norm_element;

// The view logits to `logits`; each competing row's lower-bound score into
// `lowest`.
struct BoundOut {
    device float *logits;
    ulong row, col;
    device const float *radius;
    ulong radius_stride;
    threadgroup const float *lengths;
    threadgroup atomic_uint *lowest;
    progressive::Selection selection;
    void store(uint m, uint n, float value) const {
        logits[ulong(m) * row + ulong(n) * col] = value;
        if (!selection.competes(m, n))
            return;
        // A score is its noise-free score plus noise of at most
        // PROGRESSIVE_NOISE: a row that cannot reach the threshold so far
        // cannot raise it, so it skips the noise.
        const float lower = value - radius[ulong(n) * radius_stride] * lengths[m];
        const uint best = atomic_load_explicit(&lowest[m], memory_order_relaxed);
        if (best != 0u && lower / selection.divisor(m) + PROGRESSIVE_NOISE < progressive::value_of(best))
            return;
        atomic_fetch_max_explicit(&lowest[m], progressive::key(selection.score(m, n, lower)), memory_order_relaxed);
    }
    void store2(uint m, uint n, float first, float second) const {
        store(m, n, first);
        store(m, n + 1u, second);
    }
};

#define TOP_ROWS_ARGUMENTS                                                              \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                       \
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],                           \
    device const uint *top [[buffer(SEISMIC_BUFFER_TOP)]],                              \
    device const half *scales [[buffer(SEISMIC_BUFFER_SCALES)]],                        \
    device const float *radius [[buffer(SEISMIC_BUFFER_RADIUS)]],                       \
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],                     \
    device const uint *draws [[buffer(SEISMIC_BUFFER_DRAWS)]],                          \
    device const float *temperature [[buffer(SEISMIC_BUFFER_TEMPERATURE)]],             \
    device const uint *mask [[buffer(SEISMIC_BUFFER_MASK)]],                            \
    device const int *constrained [[buffer(SEISMIC_BUFFER_CONSTRAINED)]],               \
    device float *logits [[buffer(SEISMIC_RESULT_0_BUFFER)]],                           \
    device float *threshold [[buffer(SEISMIC_RESULT_1_BUFFER)]],                        \
    device uchar *features [[buffer(SEISMIC_RESULT_2_BUFFER)]],                         \
    device float *lengths_out [[buffer(SEISMIC_RESULT_3_BUFFER)]],                      \
    device atomic_uint *bounds [[buffer(SEISMIC_BUFFER_SCRATCH_BOUNDS)]],               \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define TOP_ROWS_THREADS                                                                \
    threadgroup uchar *shared [[threadgroup(0)]],                                       \
    uint tile [[threadgroup_position_in_grid]],                                         \
    uint tiles [[threadgroups_per_grid]],                                               \
    uint tid [[thread_index_in_threadgroup]],                                           \
    uint simdgroups [[simdgroups_per_threadgroup]],                                     \
    uint sg [[simdgroup_index_in_threadgroup]],                                         \
    uint lane [[thread_index_in_simdgroup]]

// The rows' normalized lengths into `lengths` (threadgroup 0 also publishes
// the features and lengths), the threshold keys cleared.
template <typename X>
inline void top_lengths(thread const X &x, uint rows, uint d, device uchar *features, device float *lengths_out,
    ulong length_stride, threadgroup float *lengths, threadgroup float *parts, threadgroup atomic_uint *lowest, uint tile, uint tid,
    uint simdgroups, uint sg, uint lane) {
    if (tid < PROGRESSIVE_ROWS)
        atomic_store_explicit(&lowest[tid], 0u, memory_order_relaxed);
    const uint threads = simdgroups * 32u;
    for (uint m = 0; m < rows; ++m) {
        float sum = 0.0f;
        for (uint i = 8u * tid; i < d; i += 8u * threads) {
            float4 even, odd;
            x.load8(m, i, 0.0f, even, odd);
            for (uint j = 0; j < 4u; ++j) {
                const float a = progressive::activation::round(even[j]);
                const float b = progressive::activation::round(odd[j]);
                sum = metal::fma(a, a, sum);
                sum = metal::fma(b, b, sum);
            }
            if (tile == 0)
                *reinterpret_cast<device uint4 *>(
                    reinterpret_cast<device typename progressive::activation::storage *>(features) + ulong(m) * d + i)
                    = progressive::activation::pack8(even, odd);
        }
        sum = simd_sum(sum);
        if (lane == 0)
            parts[sg] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = 0.0f;
            for (uint s = 0; s < simdgroups; ++s)
                total += parts[s];
            lengths[m] = metal::sqrt(total);
            if (tile == 0)
                lengths_out[m * length_stride] = lengths[m];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Raise the rows' thresholds; the last threadgroup publishes them (-inf where
// a row's length is not finite: no bound holds) and clears the keys.
inline void top_publish(threadgroup atomic_uint *lowest, threadgroup const float *lengths, device atomic_uint *bounds,
    threadgroup uint *last, device float *threshold, ulong threshold_stride, uint rows, uint tiles, uint tid) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < rows)
        atomic_fetch_max_explicit(&bounds[tid], atomic_load_explicit(&lowest[tid], memory_order_relaxed),
            memory_order_relaxed);
    if (arrive::last(&bounds[PROGRESSIVE_ROWS], tiles, last, tid) && tid < rows) {
        const uint k = atomic_load_explicit(&bounds[tid], memory_order_relaxed);
        const float raised = k == 0u ? -INFINITY : progressive::value_of(k);
        threshold[tid * threshold_stride] = metal::isfinite(lengths[tid]) ? raised : -INFINITY;
        atomic_store_explicit(&bounds[tid], 0u, memory_order_relaxed);
    }
}

#define TOP_ROWS_OPERANDS                                                               \
    const uint d = uint(SEISMIC_DIM_D);                                                 \
    const uint rows = uint(SEISMIC_DIM_O);                                              \
    projection::Rms<progressive::activation, norm_element, projection::SelectedRows> in{hidden, \
        SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1, norm, SEISMIC_NORM_STRIDE_0,  \
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), d, {out_rows}};                    \
    const projection::Weights<packets::ProgressiveTop> w{top, SEISMIC_TOP_STRIDE_0, scales, \
        SEISMIC_SCALES_STRIDE_0};                                                       \
    threadgroup float lengths[PROGRESSIVE_ROWS];                                        \
    threadgroup atomic_uint lowest[PROGRESSIVE_ROWS];                                   \
    threadgroup float parts[32];                                                        \
    threadgroup uint last;                                                              \
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);                            \
    projection::threadgroup_squares_runtime(in, rows, squares, simdgroups, sg, lane);   \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
    const auto x = projection::shared_norm(in, squares);                               \
    top_lengths(x, rows, d, features, lengths_out, SEISMIC_RESULT_3_STRIDE_0, lengths, parts, lowest, tile, tid, simdgroups, sg, lane); \
    /* Radius column 0: the 4-bit view's. */                                             \
    const BoundOut out{logits, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, radius, \
        SEISMIC_RADIUS_STRIDE_0, lengths, lowest,                                       \
        {draws, SEISMIC_DRAWS_STRIDE_0, SEISMIC_DRAWS_STRIDE_1, temperature, SEISMIC_TEMPERATURE_STRIDE_0, mask, \
            SEISMIC_MASK_STRIDE_0, SEISMIC_MASK_STRIDE_1, constrained, SEISMIC_CONSTRAINED_STRIDE_0}}

#ifdef SEISMIC_FORMING_READOUT_TOP_ROWS_GEMV
template <uint ROWS, uint LANES>
kernel void readout_top_rows_gemv(TOP_ROWS_ARGUMENTS, TOP_ROWS_THREADS) {
    TOP_ROWS_OPERANDS;
    PROJECTION_FOR_ROWS(rows,
        projection::gemv_runtime<packets::ProgressiveTop, ROWS, MAXM, LANES>(
            x, out, w, rows, uint(SEISMIC_DIM_V), d, tile, shared, simdgroups, sg, lane));
    top_publish(lowest, lengths, bounds, &last, threshold, SEISMIC_RESULT_1_STRIDE_0, rows, tiles, tid);
}
#endif

#ifdef SEISMIC_FORMING_READOUT_TOP_ROWS_BATCH
template <uint BATCH_ROWS>
kernel void readout_top_rows_batch(TOP_ROWS_ARGUMENTS, TOP_ROWS_THREADS) {
    TOP_ROWS_OPERANDS;
    projection::gemv_batch_runtime<packets::ProgressiveTop, BATCH_ROWS>(x, out, w, rows, uint(SEISMIC_DIM_V), d,
        tile, shared, simdgroups, sg, lane);
    top_publish(lowest, lengths, bounds, &last, threshold, SEISMIC_RESULT_1_STRIDE_0, rows, tiles, tid);
}
#endif
