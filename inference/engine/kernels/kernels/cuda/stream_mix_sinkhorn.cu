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
// Decode rows (M <= DECODE_ROWS, `stream_mix_sinkhorn_split`) take a tile of
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
// reduced mixes (each already scaled by the row's inverse norm, held by
// every thread), formed by the whole block: thread e < S * S owns comb
// element (e / S, e % S) through `comb` (S * S floats of shared memory),
// each of its sums over its row or column read there in index order, so
// every element takes the same operations in the same order as one thread
// forming them all would. Every thread of the block calls it.
// Independent of any entry ABI: the caller passes its views' strides.
__device__ __forceinline__ void publish_coefficients(const float (&mixes)[MIXES], float *comb, const float *base,
                                                     unsigned long long base_stride, const float *scale,
                                                     unsigned long long scale_stride, float *own,
                                                     unsigned long long own_stride, float *joined,
                                                     unsigned long long joined_stride, float epsilon,
                                                     unsigned iterations) {
    typedef unsigned long long u64;
    const unsigned source = threadIdx.x / STREAMS, target = threadIdx.x % STREAMS;
    const bool owns = threadIdx.x < STREAMS * STREAMS;
    // The element's logit, then its row's softmax.
    float value = 0.0f;
    if (owns) {
        const unsigned at = 2u * STREAMS + threadIdx.x;
        value = mixes[at] * scale[2u * scale_stride] + base[(u64)at * base_stride];
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
        joined[(u64)(STREAMS + threadIdx.x) * joined_stride] = value;
    if (threadIdx.x < STREAMS) {
        const unsigned stream = threadIdx.x;
        own[(u64)stream * own_stride] =
            functions::sigmoid(mixes[stream] * scale[0] + base[(u64)stream * base_stride]) + epsilon;
        joined[(u64)stream * joined_stride] =
            2.0f * functions::sigmoid(mixes[STREAMS + stream] * scale[scale_stride] +
                                      base[(u64)(STREAMS + stream) * base_stride]);
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
                                             float *__restrict__ input,
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
    const unsigned warp = threadIdx.x / 32u, lane = threadIdx.x % 32u, stripe = warp / GROUPS;
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
    if constexpr (TILE == 1u) {
        for (unsigned at = first + stripe * 32u + lane; at < last; at += PASS) {
            float f[MIXES];
#pragma unroll
            for (unsigned j = 0; j < MIXES; ++j)
                f[j] = function[(u64)j * SEISMIC_FUNCTION_STRIDE_0 + (u64)at * SEISMIC_FUNCTION_STRIDE_1];
            take(at, f);
        }
    } else {
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
                                       SEISMIC_PREVIOUS_STRIDE_1, input + row * SEISMIC_RESULT_0_STRIDE_0,
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
                           reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)), seismic_words_value,   \
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
    __shared__ float comb[STREAMS * STREAMS];
    const float *base = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BASE));
    const float *scale = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCALE));
    const float *partials = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS));
    float *join = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_2_BUFFER));
    const unsigned row = blockIdx.x;
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
    const float inverse = rsqrtf(reduced[0] / (float)width + norm_epsilon);
    float mixes[MIXES];
#pragma unroll
    for (unsigned j = 0; j < MIXES; ++j)
        mixes[j] = reduced[1u + j] * inverse;
    if constexpr (CARRIED) {
        float *pre = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_1_BUFFER));
        publish_coefficients(mixes, comb, base, SEISMIC_BASE_STRIDE_0, scale, SEISMIC_SCALE_STRIDE_0,
                             pre + (u64)row * SEISMIC_RESULT_1_STRIDE_0, SEISMIC_RESULT_1_STRIDE_1,
                             join + (u64)row * SEISMIC_RESULT_2_STRIDE_0, SEISMIC_RESULT_2_STRIDE_1, epsilon,
                             iterations);
    } else {
        // Every thread holds its mixes, so `reduced` takes the pre.
        publish_coefficients(mixes, comb, base, SEISMIC_BASE_STRIDE_0, scale, SEISMIC_SCALE_STRIDE_0, reduced, 1u,
                             join + (u64)row * SEISMIC_RESULT_2_STRIDE_0, SEISMIC_RESULT_2_STRIDE_1, epsilon,
                             iterations);
        __syncthreads();
        // The sublayer's input: the streams mixed by this site's pre.
        const float *hidden = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN));
        float *input = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
        streams::collapse<STREAMS>(hidden + (u64)row * SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1,
                                   SEISMIC_HIDDEN_STRIDE_2, reduced, 1u, input + (u64)row * SEISMIC_RESULT_0_STRIDE_0,
                                   SEISMIC_RESULT_0_STRIDE_1, SEISMIC_DIM_H, threadIdx.x, THREADS);
    }
}
