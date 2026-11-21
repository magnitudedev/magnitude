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
// Decode rows (M <= VERIFY_ROWS, `stream_mix_sinkhorn_split`) take a tile of
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

// Pre element `index` of a row: the result's, by its owner, or the
// threadgroup copy the finish launch mixes the input by.
inline void put_own(element::Out<element::F32> own, ulong index, float value) { own.put(index, value); }
inline void put_own(threadgroup float *own, ulong index, float value) { own[index] = value; }

// Pre (`own`), post and the Sinkhorn comb (`joined`) of a row from its
// reduced mixes (each already scaled by the row's inverse norm, held by
// threads < S * S), formed by simdgroup 0 in registers via `simd_shuffle`
// when S * S <= 32 (zero threadgroup barriers and zero threadgroup memory
// bounces), or by the whole threadgroup through `comb` (S * S floats of
// threadgroup memory) when S * S > 32. Each row or column sum runs in index
// order, so every element takes the same operations in the same order as one
// thread forming them all would. Every thread of the threadgroup calls it.
// Independent of any entry ABI: the caller passes its views' strides, and
// `own` is the pre result or threadgroup memory (`put_own`).
template <typename Own>
inline void publish_coefficients(thread const float (&mixes)[MIXES], threadgroup float *comb, uint thread_index,
    device const float *base, ulong base_stride, device const float *scale, ulong scale_stride, Own own,
    ulong own_stride, element::Out<element::F32> joined, ulong joined_stride, float epsilon, uint iterations) {
    const uint source = thread_index / STREAMS, target = thread_index % STREAMS;
    const bool owns = thread_index < STREAMS * STREAMS;
    if constexpr (STREAMS * STREAMS <= 32u) {
        if (thread_index >= 32u)
            return;
        const uint row_base = owns ? source * STREAMS : 0u;
        const uint col_base = owns ? target : 0u;
        // The element's logit, then its row's softmax.
        float value = 0.0f;
        if (owns) {
            const uint at = 2u * STREAMS + thread_index;
            value = mixes[at] * scale[2u * scale_stride] + base[ulong(at) * base_stride];
        }
        float maximum = -INFINITY;
        for (uint t = 0; t < STREAMS; ++t)
            maximum = metal::max(maximum, simd_shuffle(value, ushort(row_base + t)));
        value = metal::exp(value - maximum);
        float total = 0.0f;
        for (uint t = 0; t < STREAMS; ++t)
            total += simd_shuffle(value, ushort(row_base + t));
        value = value / total + epsilon;
        // The Sinkhorn normalizations: rows (after the first sweep), then
        // columns.
        for (uint sweep = 0; sweep < iterations; ++sweep) {
            if (sweep > 0u) {
                float sum = 0.0f;
                for (uint t = 0; t < STREAMS; ++t)
                    sum += simd_shuffle(value, ushort(row_base + t));
                value /= sum + epsilon;
            }
            float sum = 0.0f;
            for (uint s = 0; s < STREAMS; ++s)
                sum += simd_shuffle(value, ushort(s * STREAMS + col_base));
            value /= sum + epsilon;
        }
        // Order simdgroup 0's reads of `reduced` (which `own` may alias when
        // !CARRIED) before writing `own`.
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (owns)
            joined.put(ulong(STREAMS + thread_index) * joined_stride, value);
        if (thread_index < STREAMS) {
            const uint stream = thread_index;
            put_own(own, ulong(stream) * own_stride,
                functions::sigmoid(mixes[stream] * scale[0] + base[ulong(stream) * base_stride]) + epsilon);
            joined.put(ulong(stream) * joined_stride, 2.0f
                * functions::sigmoid(mixes[STREAMS + stream] * scale[scale_stride]
                    + base[ulong(STREAMS + stream) * base_stride]));
        }
        return;
    }
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
        joined.put(ulong(STREAMS + thread_index) * joined_stride, value);
    if (thread_index < STREAMS) {
        const uint stream = thread_index;
        put_own(own, ulong(stream) * own_stride,
            functions::sigmoid(mixes[stream] * scale[0] + base[ulong(stream) * base_stride]) + epsilon);
        joined.put(ulong(stream) * joined_stride, 2.0f
            * functions::sigmoid(mixes[STREAMS + stream] * scale[scale_stride]
                + base[ulong(STREAMS + stream) * base_stride]));
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
    device float *partials, element::Out<element::F32> input, constant ulong *seismic_words, uint2 group,
    uint thread_index,
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
    if constexpr (TILE == 1u) {
        const bool unit_strides = SEISMIC_HIDDEN_STRIDE_2 == 1ul && SEISMIC_FUNCTION_STRIDE_1 == 1ul
            && (SEISMIC_FUNCTION_STRIDE_0 & 3ul) == 0ul;
        const uint tid = stripe * 32u + lane;
        for (uint s = 0; s < STREAMS; ++s) {
            const uint s_begin = s * h;
            const uint s_first = metal::max(first, s_begin);
            const uint s_last = metal::min(last, s_begin + h);
            if (s_first >= s_last)
                continue;
            device const float *xs = x[0] + ulong(s) * SEISMIC_HIDDEN_STRIDE_1;
            const uint col0 = s_first - s_begin;
            uint tail_start = s_first;
            if (unit_strides && (ulong(xs + col0) & 15ul) == 0ul && (ulong(function + s_first) & 15ul) == 0ul) {
                const uint vec_count = (s_last - s_first) / 4u;
                for (uint k = tid; k < vec_count; k += PASS) {
                    const uint off = 4u * k;
                    const float4 v = *reinterpret_cast<device const float4 *>(xs + col0 + off);
                    sums[0][0] = metal::fma(v.x, v.x, sums[0][0]);
                    sums[0][0] = metal::fma(v.y, v.y, sums[0][0]);
                    sums[0][0] = metal::fma(v.z, v.z, sums[0][0]);
                    sums[0][0] = metal::fma(v.w, v.w, sums[0][0]);
                    const ulong func_off = ulong(s_first + off);
                    for (uint j = 0; j < MIXES; ++j) {
                        const float4 f = *reinterpret_cast<device const float4 *>(
                            function + ulong(j) * SEISMIC_FUNCTION_STRIDE_0 + func_off);
                        sums[0][1u + j] = metal::fma(f.x, v.x, sums[0][1u + j]);
                        sums[0][1u + j] = metal::fma(f.y, v.y, sums[0][1u + j]);
                        sums[0][1u + j] = metal::fma(f.z, v.z, sums[0][1u + j]);
                        sums[0][1u + j] = metal::fma(f.w, v.w, sums[0][1u + j]);
                    }
                }
                tail_start = s_first + vec_count * 4u;
            }
            for (uint at = tail_start + tid; at < s_last; at += PASS) {
                const float v = xs[ulong(at - s_begin) * SEISMIC_HIDDEN_STRIDE_2];
                sums[0][0] = metal::fma(v, v, sums[0][0]);
                for (uint j = 0; j < MIXES; ++j) {
                    const float f = function[ulong(j) * SEISMIC_FUNCTION_STRIDE_0 + ulong(at) * SEISMIC_FUNCTION_STRIDE_1];
                    sums[0][1u + j] = metal::fma(f, v, sums[0][1u + j]);
                }
            }
        }
    } else {
        const bool unit_function = SEISMIC_FUNCTION_STRIDE_1 == 1ul;
        const bool unit_hidden = SEISMIC_HIDDEN_STRIDE_2 == 1ul;
        for (uint chunk = first; chunk < last; chunk += CHUNK) {
            const uint count = metal::min(CHUNK, last - chunk);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (unit_function) {
                const uint vec_count = count >> 2;
                const uint vec_end = vec_count * 4u;
                for (uint j = sg; j < MIXES; j += SIMDGROUPS) {
                    device const float *src = function + ulong(j) * SEISMIC_FUNCTION_STRIDE_0 + ulong(chunk);
                    threadgroup float *dst = staged + j * CHUNK;
                    for (uint k = lane; k < vec_count; k += 32u) {
                        *reinterpret_cast<threadgroup float4 *>(dst + 4u * k) =
                            float4(*reinterpret_cast<device const packed_float4 *>(src + 4u * k));
                    }
                    for (uint c = vec_end + lane; c < count; c += 32u)
                        dst[c] = src[c];
                }
            } else {
                for (uint j = sg; j < MIXES; j += SIMDGROUPS) {
                    device const float *src = function + ulong(j) * SEISMIC_FUNCTION_STRIDE_0;
                    threadgroup float *dst = staged + j * CHUNK;
                    for (uint c = lane; c < count; c += 32u)
                        dst[c] = src[ulong(chunk + c) * SEISMIC_FUNCTION_STRIDE_1];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint chunk_end = chunk + count;
            for (uint s = 0; s < STREAMS; ++s) {
                const uint s_begin = s * h;
                const uint s_end = s_begin + h;
                const uint lo = metal::max(chunk, s_begin);
                const uint hi = metal::min(chunk_end, s_end);
                if (lo >= hi)
                    continue;
                const uint len = hi - lo;
                const uint c_lo = lo - chunk;
                const uint col_lo = lo - s_begin;
                device const float *xs[PER];
                for (uint r = 0; r < PER; ++r)
                    xs[r] = x[r] + ulong(s) * SEISMIC_HIDDEN_STRIDE_1;
                auto take_scalar = [&](uint d) {
                    const uint col = col_lo + d;
                    const uint c = c_lo + d;
                    float v[PER];
                    for (uint r = 0; r < PER; ++r) {
                        v[r] = xs[r][ulong(col) * SEISMIC_HIDDEN_STRIDE_2];
                        sums[r][0] = metal::fma(v[r], v[r], sums[r][0]);
                    }
                    for (uint j = 0; j < MIXES; ++j) {
                        const float fj = staged[j * CHUNK + c];
                        for (uint r = 0; r < PER; ++r)
                            sums[r][1u + j] = metal::fma(fj, v[r], sums[r][1u + j]);
                    }
                };
                if (unit_hidden) {
                    const uint head = (4u - (c_lo & 3u)) & 3u;
                    const uint vec_lo = metal::min(head, len);
                    const uint vec_hi = vec_lo + ((len - vec_lo) & ~3u);
                    for (uint d = lane; d < vec_lo; d += PASS)
                        take_scalar(d);
                    for (uint d = vec_lo + lane * 4u; d < vec_hi; d += PASS * 4u) {
                        float4 v[PER];
                        for (uint r = 0; r < PER; ++r) {
                            v[r] = float4(*reinterpret_cast<device const packed_float4 *>(xs[r] + col_lo + d));
                            sums[r][0] = metal::fma(v[r].x, v[r].x, sums[r][0]);
                            sums[r][0] = metal::fma(v[r].y, v[r].y, sums[r][0]);
                            sums[r][0] = metal::fma(v[r].z, v[r].z, sums[r][0]);
                            sums[r][0] = metal::fma(v[r].w, v[r].w, sums[r][0]);
                        }
                        for (uint j = 0; j < MIXES; ++j) {
                            const float4 fj =
                                *reinterpret_cast<threadgroup const float4 *>(staged + j * CHUNK + c_lo + d);
                            for (uint r = 0; r < PER; ++r) {
                                sums[r][1u + j] = metal::fma(fj.x, v[r].x, sums[r][1u + j]);
                                sums[r][1u + j] = metal::fma(fj.y, v[r].y, sums[r][1u + j]);
                                sums[r][1u + j] = metal::fma(fj.z, v[r].z, sums[r][1u + j]);
                                sums[r][1u + j] = metal::fma(fj.w, v[r].w, sums[r][1u + j]);
                            }
                        }
                    }
                    for (uint d = vec_hi + lane; d < len; d += PASS)
                        take_scalar(d);
                } else {
                    for (uint d = lane; d < len; d += PASS)
                        take_scalar(d);
                }
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
                input.offset(row * SEISMIC_RESULT_0_STRIDE_0), SEISMIC_RESULT_0_STRIDE_1, h,
                group.x * THREADS + thread_index, split * THREADS);
        }
    }
}

#define STREAM_MIX_SLICE_KERNEL(NAME, TILE, STAGED)                                                            \
    kernel void NAME(device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                            \
        device const float *previous [[buffer(SEISMIC_BUFFER_PREVIOUS)]],                                     \
        device const float *function [[buffer(SEISMIC_BUFFER_FUNCTION)]],                                     \
        device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],                                   \
        SEISMIC_RESULT_0_PARAM(input),                                                                         \
        constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],                                       \
        uint2 group [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]],      \
        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                \
        threadgroup float simdgroups[SUMS][THREADS / 32u];                                                     \
        threadgroup float4 staged_words[((STAGED) + 3u) / 4u];                                                 \
        reduce_slice<TILE>(hidden, previous, function, partials, element::Out<element::F32>(input), seismic_words, \
            group, thread_index, sg, lane, reinterpret_cast<threadgroup float *>(staged_words), simdgroups);  \
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
    SEISMIC_RESULT_0_PARAM(input),
    SEISMIC_RESULT_1_PARAM(pre),
    SEISMIC_RESULT_2_PARAM(join),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float reduced[SUMS];
    threadgroup float comb[STREAMS * STREAMS <= 32u ? 1u : STREAMS * STREAMS];
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
    float mixes[MIXES];
    if (thread_index < STREAMS * STREAMS) {
        const float inverse = metal::rsqrt(reduced[0] / float(width) + norm_epsilon);
        for (uint j = 0; j < MIXES; ++j)
            mixes[j] = reduced[1u + j] * inverse;
    }
    const element::Out<element::F32> joined
        = element::Out<element::F32>(join).offset(ulong(row) * SEISMIC_RESULT_2_STRIDE_0);
    if (CARRIED) {
        publish_coefficients(mixes, comb, thread_index, base, SEISMIC_BASE_STRIDE_0, scale, SEISMIC_SCALE_STRIDE_0,
            element::Out<element::F32>(pre).offset(ulong(row) * SEISMIC_RESULT_1_STRIDE_0), SEISMIC_RESULT_1_STRIDE_1,
            joined, SEISMIC_RESULT_2_STRIDE_1, epsilon, iterations);
    } else {
        // Only threads < S * S read `reduced` above, so `reduced` takes the pre
        // once their reads complete (ordered inside `publish_coefficients`).
        publish_coefficients(mixes, comb, thread_index, base, SEISMIC_BASE_STRIDE_0, scale, SEISMIC_SCALE_STRIDE_0,
            reduced, 1ul, joined, SEISMIC_RESULT_2_STRIDE_1, epsilon, iterations);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // The sublayer's input: the streams mixed by this site's pre.
        float own[STREAMS];
        for (uint stream = 0; stream < STREAMS; ++stream)
            own[stream] = reduced[stream];
        streams::collapse<STREAMS>(hidden + ulong(row) * SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1,
            SEISMIC_HIDDEN_STRIDE_2, own,
            element::Out<element::F32>(input).offset(ulong(row) * SEISMIC_RESULT_0_STRIDE_0),
            SEISMIC_RESULT_0_STRIDE_1, h, thread_index, THREADS);
    }
}
