// stream_mix_down: the per-stream RMS prologue over the S hyper-connection
// streams, then one segmented projection down | inject (an inject segment
// of zero rows, the exit site's, has no tiles), and the streams' inverses.
#define KERNEL_W0 SEISMIC_DOWN_WEIGHT
#define KERNEL_W1 SEISMIC_INJECT_WEIGHT
#include "lib/core/arrive.h"
#include "lib/projection/projection.h"
#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_GEMV
#include "lib/projection/gemv.h"
#endif

typedef element::Act activation;
typedef ELEMENT_OF(SEISMIC_NORM) norm_element;

#define STREAM_MIX_DOWN_ARGUMENTS                                                       \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                       \
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],                           \
    device const uchar *down_weight [[buffer(SEISMIC_BUFFER_DOWN_WEIGHT)]],             \
    device const uchar *inject_weight [[buffer(SEISMIC_BUFFER_INJECT_WEIGHT)]],         \
    device uchar *low [[buffer(SEISMIC_RESULT_0_BUFFER)]],                              \
    device float *inject [[buffer(SEISMIC_RESULT_1_BUFFER)]],                           \
    device float *inverse [[buffer(SEISMIC_RESULT_2_BUFFER)]],                          \
    device uchar *normalized [[buffer(SEISMIC_BUFFER_SCRATCH_NORMALIZED)]],             \
    arrive::coherent_device<float> partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]], \
    device atomic_uint *arrivals [[buffer(SEISMIC_BUFFER_SCRATCH_ARRIVALS)]],           \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define STREAM_MIX_DOWN_OPERANDS                                                        \
    const uint k = uint(SEISMIC_DIM_S * SEISMIC_DIM_H);                                 \
    const uint low_rows = uint(SEISMIC_DIM_LR);                                          \
    const uint inject_rows = uint(SEISMIC_DIM_C * SEISMIC_DIM_S);                       \
    projection::GroupRms<activation, norm_element, projection::AllRows> in{hidden,      \
        SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_2, norm, SEISMIC_NORM_STRIDE_1,  \
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), k, {}, uint(SEISMIC_DIM_H)};       \
    projection::Weights<packets::W0> down_w{down_weight, KERNEL_W0_LAYOUT(k), k};       \
    projection::Weights<packets::W1> inject_w{inject_weight, KERNEL_W1_LAYOUT(k), k};   \
    projection::MixLow<activation> low_out{low, SEISMIC_RESULT_0_STRIDE_0,              \
        SEISMIC_RESULT_0_STRIDE_1, float(SEISMIC_DIM_S)};                               \
    projection::MixInject inject_out{inject, SEISMIC_RESULT_1_STRIDE_0,                 \
        SEISMIC_RESULT_1_STRIDE_1, float(SEISMIC_DIM_S)}

// The streams' square sums in threadgroup memory; the first threadgroup
// publishes the inverses they give.
#define STREAM_MIX_DOWN_SQUARES(rows)                                                   \
    PROJECTION_SQUARES_SHARED(squares, SEISMIC_DIM_S * decltype(in)::parts);            \
    projection::threadgroup_squares_runtime(in, rows, squares, simdgroups, sg, lane);   \
    if (tile == 0) {                                                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                \
        projection::publish_inverses(in, rows, squares, inverse, SEISMIC_RESULT_2_STRIDE_0, \
            SEISMIC_RESULT_2_STRIDE_1, sg * 32 + lane, simdgroups * 32);                \
    }                                                                                   \
    const auto x = projection::shared_norm(in, squares)

// One stream's F32 partial sums of the down | inject outputs: output n of row
// m from stream s at partials[(m * S + s) * outputs + column + n].
struct StreamPartial {
    arrive::coherent_device<float> y;
    ulong row;
    ulong column;
    void store(uint m, uint n, float value) const { y[ulong(m) * row + column + n] = value; }
};

// Below BATCH_FROM rows the projection is split along its K = S * H inputs by
// stream: threadgroup `tile` serves stream tile / tiles and its row tile
// tile % tiles of down | inject (`tiles` a stream). It normalizes only its
// stream, from that stream's square sums (the first row tile of a stream
// publishes its inverses), so no threadgroup repeats another stream's norm,
// and its outputs are F32 partial sums over the stream's packets. The last
// threadgroup to arrive adds every output's S partials in stream order and
// applies the epilogues.
#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_GEMV
template <uint ROWS, uint LANES>
kernel void stream_mix_down_gemv(STREAM_MIX_DOWN_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    STREAM_MIX_DOWN_OPERANDS;
    const uint streams = uint(SEISMIC_DIM_S), width = uint(SEISMIC_DIM_H);
    const uint outputs = low_rows + inject_rows;
    const uint per = simdgroups * ROWS * (32u / LANES);
    const uint rows = uint(SEISMIC_DIM_M);
    const uint t0 = (low_rows + per - 1) / per;
    const uint tiles = t0 + (inject_rows + per - 1) / per;
    const uint stream = tile / tiles, local = tile % tiles;
    const ulong first = ulong(stream) * width;
    const projection::GroupRms<activation, norm_element, projection::AllRows> part{
        hidden + first * SEISMIC_HIDDEN_STRIDE_2, SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_2,
        norm + first * SEISMIC_NORM_STRIDE_1 * norm_element::bytes, SEISMIC_NORM_STRIDE_1,
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), width, {}, width};
    PROJECTION_SQUARES_SHARED(squares, decltype(part)::parts);
    projection::threadgroup_squares_runtime(part, rows, squares, simdgroups, sg, lane);
    if (local == 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        projection::publish_inverses(part, rows, squares, inverse + ulong(stream) * SEISMIC_RESULT_2_STRIDE_1,
            SEISMIC_RESULT_2_STRIDE_0, SEISMIC_RESULT_2_STRIDE_1, sg * 32 + lane, simdgroups * 32);
    }
    const auto x = projection::shared_norm(part, squares);
    const StreamPartial sums{partials, ulong(streams) * outputs, ulong(stream) * outputs};
    if (local < t0) {
        PROJECTION_FOR_ROWS(rows, projection::gemv_runtime<packets::W0, ROWS, MAXM, LANES>(
            x, sums, down_w, rows, low_rows, width, local, shared, simdgroups, sg, lane, stream * (width / 32u)));
    } else {
        const StreamPartial injected{partials, sums.row, sums.column + low_rows};
        PROJECTION_FOR_ROWS(rows, projection::gemv_runtime<packets::W1, ROWS, MAXM, LANES>(
            x, injected, inject_w, rows, inject_rows, width, local - t0, shared, simdgroups, sg, lane,
            stream * (width / 32u)));
    }
    threadgroup uint last;
    const uint thread_index = sg * 32 + lane, threads = simdgroups * 32;
    if (!arrive::last(arrivals, streams * tiles, &last, thread_index))
        return;
    for (uint item = thread_index; item < rows * outputs; item += threads) {
        const uint m = item / outputs, n = item % outputs;
        float sum = 0.0f;
        for (uint s = 0; s < streams; ++s)
            sum += partials[(ulong(m) * streams + s) * outputs + n];
        if (n < low_rows)
            low_out.store(m, n, sum);
        else
            inject_out.store(m, n - low_rows, sum);
    }
}
#endif

#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_BATCH
template <uint BATCH_ROWS>
kernel void stream_mix_down_batch(STREAM_MIX_DOWN_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    STREAM_MIX_DOWN_OPERANDS;
    const uint per = simdgroups * BATCH_ROWS * 8u;
    const uint rows = uint(SEISMIC_DIM_M);
    const uint t0 = (low_rows + per - 1) / per;
    STREAM_MIX_DOWN_SQUARES(rows);
    if (tile < t0)
        projection::gemv_batch_runtime<packets::W0, BATCH_ROWS>(x, low_out, down_w, rows, low_rows, k,
            tile, shared, simdgroups, sg, lane);
    else
        projection::gemv_batch_runtime<packets::W1, BATCH_ROWS>(x, inject_out, inject_w, rows, inject_rows, k,
            tile - t0, shared, simdgroups, sg, lane);
}
#endif

// Past 16 rows: one threadgroup per row and stream normalizes it into
// scratch and publishes its inverse.
#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_STAGE
kernel void stream_mix_down_stage(STREAM_MIX_DOWN_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    STREAM_MIX_DOWN_OPERANDS;
    const float value = projection::device_normalize<256>(in, item, normalized, k, norms, thread_index);
    if (thread_index == 0)
        inverse[ulong(item / uint(SEISMIC_DIM_S)) * SEISMIC_RESULT_2_STRIDE_0
            + ulong(item % uint(SEISMIC_DIM_S)) * SEISMIC_RESULT_2_STRIDE_1] = value;
}
#endif

#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_GEMM
template <uint TILE_M, uint TILE_N>
kernel void stream_mix_down_gemm(STREAM_MIX_DOWN_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_SHARED(shared, TILE_M, TILE_N);
    STREAM_MIX_DOWN_OPERANDS;
    projection::Plain<activation, projection::AllRows> x{normalized, k, 1, k, {}};
    const uint m = uint(SEISMIC_DIM_M);
    const uint t0 = (low_rows + TILE_N - 1) / TILE_N;
    if (tile.x < t0)
        projection::gemm<packets::W0, TILE_M, TILE_N>(x, low_out, down_w, m, low_rows, k, tile.y, tile.x,
            shared, sg, lane);
    else
        projection::gemm<packets::W1, TILE_M, TILE_N>(x, inject_out, inject_w, m, inject_rows, k, tile.y,
            tile.x - t0, shared, sg, lane);
}
#endif
