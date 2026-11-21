// stream_mix_sinkhorn (contract and portable body in streams.seismic).
// Every row count takes one form: block (slice, tile) reduces the squares
// and every mix of a tile of rows over one contiguous slice of their columns
// [slice * width / SINKHORN_SPLIT rounded up, the next slice's first) into
// the scratch, in fixed order, and under P = 1 mixes its share of the tile's
// input columns by the previous site's pre (`streams::collapse`); a second
// launch sums each row's SINKHORN_SPLIT partials in slice order and forms
// pre, post and the Sinkhorn comb (`stream_mix_sinkhorn_finish`), and under
// P = 0 mixes the row's input by that pre, which nothing carries. The launch
// boundary orders every partial before its sum.
// Decode rows (M <= VERIFY_ROWS, `stream_mix_sinkhorn_split`) take a tile of
// one row, all eight warps over its columns, reading the function matrix
// straight from device memory: a row's function matrix streams through many
// SMs. Prefill rows (`stream_mix_sinkhorn_tiles`) take tiles of
// SINKHORN_TILE rows, two a warp: the slice's function columns are staged
// CHUNK at a time in shared memory and read there by every row of the tile,
// so the function matrix is read once a tile instead of once a row.
#include "lib/core/functions.cuh"
#include "lib/core/streams.cuh"

constexpr unsigned STREAMS = SEISMIC_DIM_S;
constexpr unsigned MIXES = (2u + STREAMS) * STREAMS;
// Whether the input mixes by the previous site's pre (P = 1) or this site's.
constexpr bool CARRIED = SEISMIC_DIM_P != 0;
constexpr unsigned THREADS = 256u;
// A slice's reduced values: its squares, then every mix.
constexpr unsigned SUMS = 1u + MIXES;

// Pre (`own`), post and the Sinkhorn comb (`joined`) of a row from its
// reduced sums (`reduced[1..1+MIXES]`, scaled by the row's inverse norm on
// participating threads): thread e < S * S owns comb element (e / S, e % S).
// When S * S <= 32, warp 0 runs the entire Sinkhorn in registers via
// `__shfl_sync` with zero `__syncthreads()`; wider S falls back to `comb`
// (S * S floats of shared memory). Both reduce every row and column in index
// order, bit-identical to one thread forming them all. Every thread of the
// block calls it. `own(stream, pre)` takes each stream's pre (a result row,
// or `own_pre` in shared memory). Independent of any entry ABI: the caller
// passes its views' strides.
template <class Own>
__device__ __forceinline__ void publish_coefficients(const float *reduced, float inverse, float *comb,
                                                     const float *base, unsigned long long base_stride,
                                                     const float *scale, unsigned long long scale_stride,
                                                     const Own &own, const element::Out<element::F32> &joined,
                                                     unsigned long long joined_stride, float epsilon,
                                                     unsigned iterations) {
    typedef unsigned long long u64;
    constexpr unsigned N2 = STREAMS * STREAMS;
    const unsigned source = threadIdx.x / STREAMS, target = threadIdx.x % STREAMS;
    const bool owns = threadIdx.x < N2;
    if constexpr (N2 <= 32u) {
        if (owns) {
            constexpr unsigned MASK = (N2 == 32u) ? 0xffffffffu : ((1u << N2) - 1u);
            const unsigned at = 2u * STREAMS + threadIdx.x;
            const float comb_mix = reduced[1u + at] * inverse;
            float value = comb_mix * scale[2u * scale_stride] + base[(u64)at * base_stride];
            float maximum = -__int_as_float(0x7f800000);
#pragma unroll
            for (unsigned t = 0; t < STREAMS; ++t)
                maximum = fmaxf(maximum, __shfl_sync(MASK, value, (int)(source * STREAMS + t)));
            value = expf(value - maximum);
            float total = 0.0f;
#pragma unroll
            for (unsigned t = 0; t < STREAMS; ++t)
                total += __shfl_sync(MASK, value, (int)(source * STREAMS + t));
            value = value / total + epsilon;
            // The Sinkhorn normalizations: rows (after the first sweep), then
            // columns.
            for (unsigned sweep = 0; sweep < iterations; ++sweep) {
                if (sweep > 0u) {
                    float sum = 0.0f;
#pragma unroll
                    for (unsigned t = 0; t < STREAMS; ++t)
                        sum += __shfl_sync(MASK, value, (int)(source * STREAMS + t));
                    value /= sum + epsilon;
                }
                float sum = 0.0f;
#pragma unroll
                for (unsigned s = 0; s < STREAMS; ++s)
                    sum += __shfl_sync(MASK, value, (int)(s * STREAMS + target));
                value /= sum + epsilon;
            }
            joined.put((u64)(STREAMS + threadIdx.x) * joined_stride, value);
            if (threadIdx.x < STREAMS) {
                const unsigned stream = threadIdx.x;
                const float pre_mix = reduced[1u + stream] * inverse;
                const float post_mix = reduced[1u + STREAMS + stream] * inverse;
                own(stream, functions::sigmoid(pre_mix * scale[0] + base[(u64)stream * base_stride]) + epsilon);
                joined.put((u64)stream * joined_stride,
                           2.0f * functions::sigmoid(post_mix * scale[scale_stride] +
                                                     base[(u64)(STREAMS + stream) * base_stride]));
            }
        }
        return;
    }
    // The element's logit, then its row's softmax.
    float value = 0.0f;
    if (owns) {
        const unsigned at = 2u * STREAMS + threadIdx.x;
        const float comb_mix = reduced[1u + at] * inverse;
        value = comb_mix * scale[2u * scale_stride] + base[(u64)at * base_stride];
        comb[threadIdx.x] = value;
    }
    __syncthreads();
    if (owns) {
        float maximum = -__int_as_float(0x7f800000);
        for (unsigned t = 0; t < STREAMS; ++t)
            maximum = fmaxf(maximum, comb[source * STREAMS + t]);
        value = expf(value - maximum);
    }
    __syncthreads();
    if (owns)
        comb[threadIdx.x] = value;
    __syncthreads();
    if (owns) {
        float total = 0.0f;
        for (unsigned t = 0; t < STREAMS; ++t)
            total += comb[source * STREAMS + t];
        value = value / total + epsilon;
    }
    // The Sinkhorn normalizations: rows (after the first sweep), then
    // columns.
    for (unsigned sweep = 0; sweep < iterations; ++sweep) {
        if (sweep > 0u) {
            __syncthreads();
            if (owns)
                comb[threadIdx.x] = value;
            __syncthreads();
            if (owns) {
                float sum = 0.0f;
                for (unsigned t = 0; t < STREAMS; ++t)
                    sum += comb[source * STREAMS + t];
                value /= sum + epsilon;
            }
        }
        __syncthreads();
        if (owns)
            comb[threadIdx.x] = value;
        __syncthreads();
        if (owns) {
            float sum = 0.0f;
            for (unsigned s = 0; s < STREAMS; ++s)
                sum += comb[s * STREAMS + target];
            value /= sum + epsilon;
        }
    }
    if (owns)
        joined.put((u64)(STREAMS + threadIdx.x) * joined_stride, value);
    if (threadIdx.x < STREAMS) {
        const unsigned stream = threadIdx.x;
        const float pre_mix = reduced[1u + stream] * inverse;
        const float post_mix = reduced[1u + STREAMS + stream] * inverse;
        own(stream, functions::sigmoid(pre_mix * scale[0] + base[(u64)stream * base_stride]) + epsilon);
        joined.put((u64)stream * joined_stride,
                   2.0f * functions::sigmoid(post_mix * scale[scale_stride] +
                                             base[(u64)(STREAMS + stream) * base_stride]));
    }
}

// Rows a prefill tile holds, and the function columns staged at a time:
// whole 32-column passes within 24 KB (256 at S = 4, 64 at S = 8).
constexpr unsigned PREFILL_TILE = SEISMIC_CONST_SINKHORN_TILE;
constexpr unsigned CHUNK = 6144u / MIXES / 32u * 32u;

// Block (slice, tile): the squares and every mix of the tile's rows
// tile * TILE .. below M over the slice's columns, each reduced in fixed
// order into its row's partials, and each row's input at its columns
// slice * 256 + t, every SINKHORN_SPLIT * 256. Warp w owns rows w % GROUPS
// (and every GROUPS-th after it in the tile) and column stripe w / GROUPS:
// a lane takes, pass by pass, column pass + stripe * 32 + lane of each
// PASS = 32 * STRIPES columns, summing in pass order; a row's stripes then
// sum in stripe order. A tile of one row reads the function matrix from
// device memory (each of its columns once); a wider tile reads it from
// `staged`, CHUNK columns a round.
template <unsigned TILE>
__device__ __forceinline__ void reduce_slice(const float *__restrict__ hidden, const float *__restrict__ previous,
                                             const float *__restrict__ function, float *__restrict__ partials,
                                             const element::Out<element::F32> &input,
                                             const seismic_words_t &seismic_words_value, float *staged,
                                             float (*warps)[THREADS / 32u]) {
    typedef unsigned long long u64;
    constexpr unsigned WARPS = THREADS / 32u;
    constexpr unsigned GROUPS = TILE < WARPS ? TILE : WARPS;
    constexpr unsigned PER = TILE / GROUPS;
    constexpr unsigned STRIPES = WARPS / GROUPS;
    constexpr unsigned PASS = 32u * STRIPES;
    static_assert(WARPS % GROUPS == 0 && TILE % GROUPS == 0, "a tile's rows are whole warps' rows");
    static_assert(TILE == 1u || (STRIPES == 1u && CHUNK % PASS == 0u),
                  "a staged tile's warps each take whole rows, a chunk whole passes");
    const unsigned h = SEISMIC_DIM_H, width = STREAMS * h, split = (unsigned)SEISMIC_CONST_SINKHORN_SPLIT;
    const unsigned rows = (unsigned)SEISMIC_DIM_M, row0 = blockIdx.y * TILE, span = (width + split - 1u) / split;
    const unsigned first = blockIdx.x * span, last = min(first + span, width);
    const unsigned warp = threadIdx.x / 32u, lane = threadIdx.x % 32u;
    const float *x[PER];
    bool live[PER];
#pragma unroll
    for (unsigned r = 0; r < PER; ++r) {
        const unsigned row = row0 + warp % GROUPS + r * GROUPS;
        live[r] = row < rows;
        x[r] = hidden + (u64)(live[r] ? row : row0) * SEISMIC_HIDDEN_STRIDE_0;
    }
    float sums[PER][SUMS];
#pragma unroll
    for (unsigned r = 0; r < PER; ++r)
#pragma unroll
        for (unsigned j = 0; j < SUMS; ++j)
            sums[r][j] = 0.0f;
    if constexpr (TILE == 1u) {
        const bool unit_strided = SEISMIC_HIDDEN_STRIDE_2 == 1 && SEISMIC_FUNCTION_STRIDE_1 == 1 &&
                                  (SEISMIC_FUNCTION_STRIDE_0 & 3u) == 0u;
        for (unsigned s = 0; s < STREAMS; ++s) {
            const unsigned s_begin = s * h;
            const unsigned s_end = s_begin + h;
            const unsigned lo = max(first, s_begin);
            const unsigned hi = min(last, s_end);
            if (lo >= hi)
                continue;
            const unsigned col_lo = lo - s_begin;
            const unsigned col_hi = hi - s_begin;
            const float *xs = x[0] + (u64)s * SEISMIC_HIDDEN_STRIDE_1;
            auto take_scalar = [&](unsigned col) {
                const unsigned at = s_begin + col;
                const float v = xs[(u64)col * SEISMIC_HIDDEN_STRIDE_2];
                sums[0][0] = __fmaf_rn(v, v, sums[0][0]);
#pragma unroll
                for (unsigned j = 0; j < MIXES; ++j) {
                    const float fj = function[(u64)j * SEISMIC_FUNCTION_STRIDE_0 + (u64)at * SEISMIC_FUNCTION_STRIDE_1];
                    sums[0][1u + j] = __fmaf_rn(fj, v, sums[0][1u + j]);
                }
            };
            const u64 x_addr = reinterpret_cast<u64>(xs + col_lo);
            const u64 f_addr = reinterpret_cast<u64>(function + lo);
            if (unit_strided && ((x_addr & 15u) == (f_addr & 15u))) {
                const unsigned mis = (unsigned)((x_addr >> 2) & 3u);
                const unsigned head = (mis == 0u) ? 0u : (4u - mis);
                const unsigned vec_lo = min(col_lo + head, col_hi);
                const unsigned vec_hi = vec_lo + ((col_hi - vec_lo) & ~3u);
                for (unsigned col = col_lo + threadIdx.x; col < vec_lo; col += PASS)
                    take_scalar(col);
                for (unsigned col = vec_lo + threadIdx.x * 4u; col < vec_hi; col += PASS * 4u) {
                    const float4 v = *reinterpret_cast<const float4 *>(xs + col);
                    sums[0][0] = __fmaf_rn(v.x, v.x, sums[0][0]);
                    sums[0][0] = __fmaf_rn(v.y, v.y, sums[0][0]);
                    sums[0][0] = __fmaf_rn(v.z, v.z, sums[0][0]);
                    sums[0][0] = __fmaf_rn(v.w, v.w, sums[0][0]);
                    const unsigned at = s_begin + col;
#pragma unroll
                    for (unsigned j = 0; j < MIXES; ++j) {
                        const float4 fj = *reinterpret_cast<const float4 *>(
                            function + (u64)j * SEISMIC_FUNCTION_STRIDE_0 + (u64)at);
                        sums[0][1u + j] = __fmaf_rn(fj.x, v.x, sums[0][1u + j]);
                        sums[0][1u + j] = __fmaf_rn(fj.y, v.y, sums[0][1u + j]);
                        sums[0][1u + j] = __fmaf_rn(fj.z, v.z, sums[0][1u + j]);
                        sums[0][1u + j] = __fmaf_rn(fj.w, v.w, sums[0][1u + j]);
                    }
                }
                for (unsigned col = vec_hi + threadIdx.x; col < col_hi; col += PASS)
                    take_scalar(col);
            } else {
                for (unsigned col = col_lo + threadIdx.x; col < col_hi; col += PASS)
                    take_scalar(col);
            }
        }
    } else {
        // Column `at` of every row this lane holds.
        auto take = [&](unsigned at, const float (&f)[MIXES]) {
            const unsigned stream = at / h, column = at % h;
#pragma unroll
            for (unsigned r = 0; r < PER; ++r) {
                const float v = x[r][(u64)stream * SEISMIC_HIDDEN_STRIDE_1 + (u64)column * SEISMIC_HIDDEN_STRIDE_2];
                sums[r][0] = __fmaf_rn(v, v, sums[r][0]);
#pragma unroll
                for (unsigned j = 0; j < MIXES; ++j)
                    sums[r][1u + j] = __fmaf_rn(f[j], v, sums[r][1u + j]);
            }
        };
        for (unsigned chunk = first; chunk < last; chunk += CHUNK) {
            const unsigned count = min(CHUNK, last - chunk);
            __syncthreads();
            for (unsigned e = threadIdx.x; e < MIXES * count; e += THREADS) {
                const unsigned j = e / count, c = e % count;
                staged[j * CHUNK + c] =
                    function[(u64)j * SEISMIC_FUNCTION_STRIDE_0 + (u64)(chunk + c) * SEISMIC_FUNCTION_STRIDE_1];
            }
            __syncthreads();
            for (unsigned c = lane; c < count; c += PASS) {
                float f[MIXES];
#pragma unroll
                for (unsigned j = 0; j < MIXES; ++j)
                    f[j] = staged[j * CHUNK + c];
                take(chunk + c, f);
            }
        }
    }
    // A row's sums: its lanes' (the warp sum), then its stripes' in stripe
    // order.
#pragma unroll
    for (unsigned r = 0; r < PER; ++r) {
#pragma unroll
        for (unsigned j = 0; j < SUMS; ++j) {
            const float total = seismic_warp_sum_f32(sums[r][j]);
            if (STRIPES == 1u) {
                const unsigned row = row0 + warp % GROUPS + r * GROUPS;
                if (lane == 0u && live[r])
                    partials[((u64)row * split + blockIdx.x) * SUMS + j] = total;
            } else if (lane == 0u) {
                warps[j][warp] = total;
            }
        }
    }
    if (STRIPES > 1u) {
        __syncthreads();
        // GROUPS == TILE rows of one each: row g's stripes are warps g,
        // g + GROUPS, ...
        for (unsigned e = threadIdx.x; e < GROUPS * SUMS; e += THREADS) {
            const unsigned g = e / SUMS, j = e % SUMS;
            if (row0 + g >= rows)
                continue;
            float total = 0.0f;
            for (unsigned w = 0; w < STRIPES; ++w)
                total += warps[j][w * GROUPS + g];
            partials[((u64)(row0 + g) * split + blockIdx.x) * SUMS + j] = total;
        }
    }
    // The sublayer's input: the streams mixed by the previous site's pre.
    if constexpr (CARRIED) {
        for (unsigned r = 0; r < TILE && row0 + r < rows; ++r) {
            const u64 row = row0 + r;
            streams::collapse<STREAMS>(hidden + row * SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1,
                                       SEISMIC_HIDDEN_STRIDE_2, previous + row * SEISMIC_PREVIOUS_STRIDE_0,
                                       SEISMIC_PREVIOUS_STRIDE_1, input.offset(row * SEISMIC_RESULT_0_STRIDE_0),
                                       SEISMIC_RESULT_0_STRIDE_1, h, blockIdx.x * THREADS + threadIdx.x,
                                       split * THREADS);
        }
    }
}

// The two forms' kernels: a tile of one decode row, and prefill tiles.
#define STREAM_MIX_SLICE_KERNEL(NAME, TILE, STAGED)                                                                 \
    extern "C" __global__ void NAME(SEISMIC_KERNEL_PARAMS) {                                                        \
        __shared__ float warps[SUMS][THREADS / 32u];                                                                \
        __shared__ float staged[STAGED];                                                                            \
        reduce_slice<TILE>(reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)),                    \
                           reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_PREVIOUS)),                  \
                           reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_FUNCTION)),                  \
                           reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS)),                \
                           SEISMIC_RESULT_0_OUT, seismic_words_value,                                              \
                           staged, warps);                                                                          \
    }

STREAM_MIX_SLICE_KERNEL(stream_mix_sinkhorn_split, 1u, 1u)
STREAM_MIX_SLICE_KERNEL(stream_mix_sinkhorn_tiles, PREFILL_TILE, MIXES * CHUNK)

// Block row: its SINKHORN_SPLIT partials summed in slice order, then its
// coefficients (`publish_coefficients`); under P = 0 its pre stays in shared
// memory and the block mixes the row's input by it.
extern "C" __global__ void stream_mix_sinkhorn_finish(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    __shared__ float reduced[SUMS];
    __shared__ float comb[STREAMS * STREAMS <= 32u ? 1u : STREAMS * STREAMS];
    __shared__ float own_pre[CARRIED ? 1u : STREAMS];
    const float *base = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BASE));
    const float *scale = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCALE));
    const float *partials = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS));
    const unsigned row = blockIdx.x;
    const element::Out<element::F32> join = SEISMIC_RESULT_2_OUT.offset((u64)row * SEISMIC_RESULT_2_STRIDE_0);
    const unsigned width = STREAMS * SEISMIC_DIM_H, split = (unsigned)SEISMIC_CONST_SINKHORN_SPLIT;
    const float epsilon = __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON);
    const float norm_epsilon = __uint_as_float((unsigned)SEISMIC_PARAM_NORM_EPSILON);
    const unsigned iterations = (unsigned)SEISMIC_PARAM_ITERATIONS;
    if (threadIdx.x < SUMS) {
        float total = 0.0f;
        for (unsigned slice = 0; slice < split; ++slice)
            total += partials[((u64)row * split + slice) * SUMS + threadIdx.x];
        reduced[threadIdx.x] = total;
    }
    __syncthreads();
    const float inverse =
        (threadIdx.x < STREAMS * STREAMS) ? rsqrtf(reduced[0] / (float)width + norm_epsilon) : 0.0f;
    if constexpr (CARRIED) {
        const element::Out<element::F32> pre = SEISMIC_RESULT_1_OUT.offset((u64)row * SEISMIC_RESULT_1_STRIDE_0);
        publish_coefficients(
            reduced, inverse, comb, base, SEISMIC_BASE_STRIDE_0, scale, SEISMIC_SCALE_STRIDE_0,
            [&](unsigned stream, float value) { pre.put((u64)stream * SEISMIC_RESULT_1_STRIDE_1, value); }, join,
            SEISMIC_RESULT_2_STRIDE_1, epsilon, iterations);
    } else {
        // `own_pre` is separate from `reduced`, so writing pre cannot race with
        // participating threads reading `reduced`.
        publish_coefficients(
            reduced, inverse, comb, base, SEISMIC_BASE_STRIDE_0, scale, SEISMIC_SCALE_STRIDE_0,
            [&](unsigned stream, float value) { own_pre[stream] = value; }, join, SEISMIC_RESULT_2_STRIDE_1, epsilon,
            iterations);
        __syncthreads();
        // The sublayer's input: the streams mixed by this site's pre.
        const float *hidden = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN));
        streams::collapse<STREAMS>(hidden + (u64)row * SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1,
                                   SEISMIC_HIDDEN_STRIDE_2, own_pre, 1u,
                                   SEISMIC_RESULT_0_OUT.offset((u64)row * SEISMIC_RESULT_0_STRIDE_0),
                                   SEISMIC_RESULT_0_STRIDE_1, SEISMIC_DIM_H, threadIdx.x, THREADS);
    }
}
