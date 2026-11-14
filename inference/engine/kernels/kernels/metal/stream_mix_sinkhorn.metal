// stream_mix_sinkhorn (contract and portable body in streams.seismic).
// Every row count takes one form: threadgroup (slice, tile) reduces the
// squares and every mix of a tile of a row's columns' contiguous slice
// [slice * width / SINKHORN_SPLIT rounded up, the next slice's first) into
// the scratch, in fixed order, and under P = 1 mixes its share of the
// tile's input columns by the previous site's pre (`streams::collapse`); a
// second launch sums each row's SINKHORN_SPLIT partials in slice order and
// forms pre, post and the Sinkhorn comb (`stream_mix_sinkhorn_finish`), and
// under P = 0 mixes the row's input by that pre, which nothing carries. The
// launch boundary orders every partial before its sum.
// Decode rows (M <= DECODE_ROWS, `stream_mix_sinkhorn_split`) take a tile of
// one row, all eight simdgroups over its columns, reading the function
// matrix straight from device memory: a row's function matrix streams
// through many cores. Prefill rows (`stream_mix_sinkhorn_tiles`) take tiles
// of SINKHORN_TILE rows, two a simdgroup: the slice's function columns are
// staged CHUNK at a time in threadgroup memory and read there by every row
// of the tile, so the function matrix is read once a tile instead of once a
// row.
#include "lib/core/functions.h"
#include "lib/core/streams.h"

constant constexpr uint STREAMS = uint(SEISMIC_DIM_S);
constant constexpr uint MIXES = (2u + STREAMS) * STREAMS;
// Whether the input mixes by the previous site's pre (P = 1) or this site's.
constant constexpr bool CARRIED = uint(SEISMIC_DIM_P) != 0u;
constant constexpr uint THREADS = 256u;
// A slice's reduced values: its squares, then every mix.
constant constexpr uint SUMS = 1u + MIXES;

// Pre (`own`), post and the Sinkhorn comb (`joined`) of a row from its
// reduced mixes (each already scaled by the row's inverse norm, held by
// every thread), formed by the whole threadgroup: thread e < S * S owns comb
// element (e / S, e % S) through `comb` (S * S floats of threadgroup memory),
// each of its sums over its row or column read there in index order, so
// every element takes the same operations in the same order as one thread
// forming them all would. Every thread of the threadgroup calls it.
// Independent of any entry ABI: the caller passes its views' strides, and
// `own` is device or threadgroup memory.
template <typename Own>
inline void publish_coefficients(thread const float (&mixes)[MIXES], threadgroup float *comb, uint thread_index,
    device const float *base, ulong base_stride, device const float *scale, ulong scale_stride, Own own,
    ulong own_stride, device float *joined, ulong joined_stride, float epsilon, uint iterations) {
    const uint source = thread_index / STREAMS, target = thread_index % STREAMS;
    const bool owns = thread_index < STREAMS * STREAMS;
    // The element's logit, then its row's softmax.
    float value = 0.0f;
    if (owns) {
        const uint at = 2u * STREAMS + thread_index;
        value = mixes[at] * scale[2u * scale_stride] + base[ulong(at) * base_stride];
        comb[thread_index] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (owns) {
        float maximum = -INFINITY;
        for (uint t = 0; t < STREAMS; ++t)
            maximum = metal::max(maximum, comb[source * STREAMS + t]);
        value = metal::exp(value - maximum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (owns)
        comb[thread_index] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (owns) {
        float total = 0.0f;
        for (uint t = 0; t < STREAMS; ++t)
            total += comb[source * STREAMS + t];
        value = value / total + epsilon;
    }
    // The Sinkhorn normalizations: rows (after the first sweep), then
    // columns.
    for (uint sweep = 0; sweep < iterations; ++sweep) {
        if (sweep > 0u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (owns)
                comb[thread_index] = value;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (owns) {
                float sum = 0.0f;
                for (uint t = 0; t < STREAMS; ++t)
                    sum += comb[source * STREAMS + t];
                value /= sum + epsilon;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (owns)
            comb[thread_index] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (owns) {
            float sum = 0.0f;
            for (uint s = 0; s < STREAMS; ++s)
                sum += comb[s * STREAMS + target];
            value /= sum + epsilon;
        }
    }
    if (owns)
        joined[ulong(STREAMS + thread_index) * joined_stride] = value;
    if (thread_index < STREAMS) {
        const uint stream = thread_index;
        own[ulong(stream) * own_stride] =
            functions::sigmoid(mixes[stream] * scale[0] + base[ulong(stream) * base_stride]) + epsilon;
        joined[ulong(stream) * joined_stride] = 2.0f
            * functions::sigmoid(mixes[STREAMS + stream] * scale[scale_stride]
                + base[ulong(STREAMS + stream) * base_stride]);
    }
}

// Rows a prefill tile holds, and the function columns staged at a time:
// whole 32-column passes within 24 KB (256 at S = 4, 64 at S = 8).
constant constexpr uint PREFILL_TILE = uint(SEISMIC_CONST_SINKHORN_TILE);
constant constexpr uint CHUNK = 6144u / MIXES / 32u * 32u;

// Threadgroup (slice, tile): the squares and every mix of the tile's rows
// tile * TILE .. below M over the slice's columns, each reduced in fixed
// order into its row's partials, and each row's input at its columns
// slice * 256 + t, every SINKHORN_SPLIT * 256. Simdgroup g owns rows
// g % GROUPS (and every GROUPS-th after it in the tile) and column stripe
// g / GROUPS: a lane takes, pass by pass, column pass + stripe * 32 + lane
// of each PASS = 32 * STRIPES columns, summing in pass order; a row's
// stripes then sum in stripe order. A tile of one row reads the function
// matrix from device memory (each of its columns once); a wider tile reads
// it from `staged`, CHUNK columns a round.
template <uint TILE>
inline void reduce_slice(device const float *hidden, device const float *previous, device const float *function,
    device float *partials, device float *input, constant ulong *seismic_words, uint2 group, uint thread_index,
    uint sg, uint lane, threadgroup float *staged, threadgroup float (*simdgroups)[THREADS / 32u]) {
    constexpr uint SIMDGROUPS = THREADS / 32u;
    constexpr uint GROUPS = TILE < SIMDGROUPS ? TILE : SIMDGROUPS;
    constexpr uint PER = TILE / GROUPS;
    constexpr uint STRIPES = SIMDGROUPS / GROUPS;
    constexpr uint PASS = 32u * STRIPES;
    static_assert(SIMDGROUPS % GROUPS == 0 && TILE % GROUPS == 0, "a tile's rows are whole simdgroups' rows");
    static_assert(TILE == 1u || (STRIPES == 1u && CHUNK % PASS == 0u),
        "a staged tile's simdgroups each take whole rows, a chunk whole passes");
    const uint h = uint(SEISMIC_DIM_H), width = STREAMS * h, split = uint(SEISMIC_CONST_SINKHORN_SPLIT);
    const uint rows = uint(SEISMIC_DIM_M), row0 = group.y * TILE, span = (width + split - 1u) / split;
    const uint first = group.x * span, last = metal::min(first + span, width);
    const uint stripe = sg / GROUPS;
    device const float *x[PER];
    bool live[PER];
    for (uint r = 0; r < PER; ++r) {
        const uint row = row0 + sg % GROUPS + r * GROUPS;
        live[r] = row < rows;
        x[r] = hidden + ulong(live[r] ? row : row0) * SEISMIC_HIDDEN_STRIDE_0;
    }
    float sums[PER][SUMS];
    for (uint r = 0; r < PER; ++r)
        for (uint j = 0; j < SUMS; ++j)
            sums[r][j] = 0.0f;
    // Column `at` of every row this lane holds.
    auto take = [&](uint at, thread const float (&f)[MIXES]) {
        const uint stream = at / h, column = at % h;
        for (uint r = 0; r < PER; ++r) {
            const float v = x[r][ulong(stream) * SEISMIC_HIDDEN_STRIDE_1 + ulong(column) * SEISMIC_HIDDEN_STRIDE_2];
            sums[r][0] = metal::fma(v, v, sums[r][0]);
            for (uint j = 0; j < MIXES; ++j)
                sums[r][1u + j] = metal::fma(f[j], v, sums[r][1u + j]);
        }
    };
    if constexpr (TILE == 1u) {
        for (uint at = first + stripe * 32u + lane; at < last; at += PASS) {
            float f[MIXES];
            for (uint j = 0; j < MIXES; ++j)
                f[j] = function[ulong(j) * SEISMIC_FUNCTION_STRIDE_0 + ulong(at) * SEISMIC_FUNCTION_STRIDE_1];
            take(at, f);
        }
    } else {
        for (uint chunk = first; chunk < last; chunk += CHUNK) {
            const uint count = metal::min(CHUNK, last - chunk);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = thread_index; e < MIXES * count; e += THREADS) {
                const uint j = e / count, c = e % count;
                staged[j * CHUNK + c] =
                    function[ulong(j) * SEISMIC_FUNCTION_STRIDE_0 + ulong(chunk + c) * SEISMIC_FUNCTION_STRIDE_1];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint c = lane; c < count; c += PASS) {
                float f[MIXES];
                for (uint j = 0; j < MIXES; ++j)
                    f[j] = staged[j * CHUNK + c];
                take(chunk + c, f);
            }
        }
    }
    // A row's sums: its lanes' (`simd_sum`), then its stripes' in stripe
    // order.
    for (uint r = 0; r < PER; ++r) {
        for (uint j = 0; j < SUMS; ++j) {
            const float total = simd_sum(sums[r][j]);
            if (STRIPES == 1u) {
                const uint row = row0 + sg % GROUPS + r * GROUPS;
                if (lane == 0u && live[r])
                    partials[(ulong(row) * split + group.x) * SUMS + j] = total;
            } else if (lane == 0u) {
                simdgroups[j][sg] = total;
            }
        }
    }
    if (STRIPES > 1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // GROUPS == TILE rows of one each: row g's stripes are simdgroups
        // g, g + GROUPS, ...
        for (uint e = thread_index; e < GROUPS * SUMS; e += THREADS) {
            const uint g = e / SUMS, j = e % SUMS;
            if (row0 + g >= rows)
                continue;
            float total = 0.0f;
            for (uint s = 0; s < STRIPES; ++s)
                total += simdgroups[j][s * GROUPS + g];
            partials[(ulong(row0 + g) * split + group.x) * SUMS + j] = total;
        }
    }
    // The sublayer's input: the streams mixed by the previous site's pre.
    if (CARRIED) {
        for (uint r = 0; r < TILE && row0 + r < rows; ++r) {
            const ulong row = row0 + r;
            streams::collapse<STREAMS>(hidden + row * SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1,
                SEISMIC_HIDDEN_STRIDE_2, previous + row * SEISMIC_PREVIOUS_STRIDE_0, SEISMIC_PREVIOUS_STRIDE_1,
                input + row * SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, h,
                group.x * THREADS + thread_index, split * THREADS);
        }
    }
}

#define STREAM_MIX_SLICE_KERNEL(NAME, TILE, STAGED)                                                            \
    kernel void NAME(device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                            \
        device const float *previous [[buffer(SEISMIC_BUFFER_PREVIOUS)]],                                     \
        device const float *function [[buffer(SEISMIC_BUFFER_FUNCTION)]],                                     \
        device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],                                   \
        device float *input [[buffer(SEISMIC_RESULT_0_BUFFER)]],                                              \
        constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],                                       \
        uint2 group [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]],      \
        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                \
        threadgroup float simdgroups[SUMS][THREADS / 32u];                                                     \
        threadgroup float staged[STAGED];                                                                      \
        reduce_slice<TILE>(hidden, previous, function, partials, input, seismic_words, group, thread_index, sg, \
            lane, staged, simdgroups);                                                                         \
    }

STREAM_MIX_SLICE_KERNEL(stream_mix_sinkhorn_split, 1u, 1u)
STREAM_MIX_SLICE_KERNEL(stream_mix_sinkhorn_tiles, PREFILL_TILE, MIXES * CHUNK)

// Threadgroup row: its SINKHORN_SPLIT partials summed in slice order, then its
// coefficients (`publish_coefficients`).
kernel void stream_mix_sinkhorn_finish(
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],
    device const float *base [[buffer(SEISMIC_BUFFER_BASE)]],
    device const float *scale [[buffer(SEISMIC_BUFFER_SCALE)]],
    device const float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
    device float *input [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device float *pre [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    device float *join [[buffer(SEISMIC_RESULT_2_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float reduced[SUMS];
    threadgroup float comb[STREAMS * STREAMS];
    const uint h = uint(SEISMIC_DIM_H), width = STREAMS * h, split = uint(SEISMIC_CONST_SINKHORN_SPLIT);
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    const float norm_epsilon = as_type<float>(uint(SEISMIC_PARAM_NORM_EPSILON));
    const uint iterations = uint(SEISMIC_PARAM_ITERATIONS);
    if (thread_index < SUMS) {
        float total = 0.0f;
        for (uint slice = 0; slice < split; ++slice)
            total += partials[(ulong(row) * split + slice) * SUMS + thread_index];
        reduced[thread_index] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inverse = metal::rsqrt(reduced[0] / float(width) + norm_epsilon);
    float mixes[MIXES];
    for (uint j = 0; j < MIXES; ++j)
        mixes[j] = reduced[1u + j] * inverse;
    if (CARRIED) {
        publish_coefficients(mixes, comb, thread_index, base, SEISMIC_BASE_STRIDE_0, scale, SEISMIC_SCALE_STRIDE_0,
            pre + ulong(row) * SEISMIC_RESULT_1_STRIDE_0, SEISMIC_RESULT_1_STRIDE_1,
            join + ulong(row) * SEISMIC_RESULT_2_STRIDE_0, SEISMIC_RESULT_2_STRIDE_1, epsilon, iterations);
    } else {
        // Every thread holds its mixes, so `reduced` takes the pre.
        publish_coefficients(mixes, comb, thread_index, base, SEISMIC_BASE_STRIDE_0, scale, SEISMIC_SCALE_STRIDE_0,
            reduced, 1ul, join + ulong(row) * SEISMIC_RESULT_2_STRIDE_0, SEISMIC_RESULT_2_STRIDE_1, epsilon,
            iterations);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // The sublayer's input: the streams mixed by this site's pre.
        float own[STREAMS];
        for (uint stream = 0; stream < STREAMS; ++stream)
            own[stream] = reduced[stream];
        streams::collapse<STREAMS>(hidden + ulong(row) * SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1,
            SEISMIC_HIDDEN_STRIDE_2, own, input + ulong(row) * SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1,
            h, thread_index, THREADS);
    }
}
