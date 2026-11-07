// stream_mix_down: the per-stream RMS prologue (`GroupRms`) over the S
// hyper-connection streams, then one segmented projection down | inject (an
// inject segment of zero rows, the exit site's, has no tiles), and the
// streams' inverses. Rows are staged by the staging launch past one row. The
// GEMV classes split K by stream: a block serves one stream (a one-row block
// forms that stream's row itself, and the stream's first block publishes its
// inverse) over tiles of down | inject, into F32 partial sums that the last
// block to arrive adds in stream order through the epilogues.
#define KERNEL_W0 SEISMIC_DOWN_WEIGHT
#define KERNEL_W1 SEISMIC_INJECT_WEIGHT
#include "lib/core/arrive.cuh"
#include "lib/projection/projection.cuh"

SEISMIC_PROGRAMMATIC_DEPENDENCY

using Pro = projection::GroupRms<ELEMENT_OF(SEISMIC_NORM), projection::AllRows, SEISMIC_DIM_S>;
using Low = projection::MixLow<ELEMENT_OF(SEISMIC_ELEMENT_A)>;

#define WIDTH (SEISMIC_DIM_S * SEISMIC_DIM_H)
#define PROLOGUE                                                                                          \
    Pro {                                                                                                 \
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)), SEISMIC_HIDDEN_STRIDE_0,      \
            SEISMIC_PTR(SEISMIC_BUFFER_NORM), __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON),            \
            SEISMIC_DIM_H, projection::AllRows{}                                                          \
    }
#define STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define INVERSE reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_2_BUFFER))

using Norm = ELEMENT_OF(SEISMIC_NORM);
using Stream = projection::GroupRms<Norm, projection::AllRows, 1>;
using StreamSource = projection::GemvSource<Stream>::type;
#define PARTIALS reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS))
#define ARRIVALS reinterpret_cast<unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_ARRIVALS))
// Stream 0 of the streams' prologue; a GEMV block offsets it to its stream.
#define STREAM                                                                                            \
    Stream {                                                                                              \
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)), SEISMIC_HIDDEN_STRIDE_0,      \
            SEISMIC_PTR(SEISMIC_BUFFER_NORM), __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON),            \
            SEISMIC_DIM_H, projection::AllRows{}                                                          \
    }

// The two segments: rows, weights and results in segment order.
struct Segments {
    unsigned long long rows[2];
    packets::W0 down;
    packets::W1 inject;
    Low low;
    projection::MixInject injected;
};

#define SEGMENTS                                                                                          \
    Segments {                                                                                            \
        {SEISMIC_DIM_LR, SEISMIC_DIM_C * SEISMIC_DIM_S},                                                   \
            KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_DOWN_WEIGHT)),                                         \
            KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_INJECT_WEIGHT)),                                       \
            Low{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), SEISMIC_RESULT_0_STRIDE_0, (float)SEISMIC_DIM_S},    \
            projection::MixInject {                                                                       \
            reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_1_BUFFER)), SEISMIC_RESULT_1_STRIDE_0,    \
                (float)SEISMIC_DIM_S                                                                      \
        }                                                                                                 \
    }

// One stream's F32 partial sums of the down | inject outputs: output n of
// row m from stream s at partials[(m * S + s) * outputs + column + n].
struct StreamPartial {
    float *out;
    unsigned long long row;
    unsigned long long column;
    __device__ __forceinline__ void operator()(unsigned m, unsigned long long n, float projected, float) const {
        out[m * row + column + n] = projected;
    }
};

// The segmented GEMV over NB column blocks of 8 rows, K split by stream:
// the block's tile groups serve stream `tile_group / per` (`per` the tile groups
// of one stream's down | inject) over its window of H inputs.
template <int NB, int KSPLIT>
__device__ __forceinline__ void mix_gemv(const Stream &first_stream, projection::u8 *row,
                                         const projection::u8 *staged, float *inverse,
                                         unsigned long long inverse_stride, float *partials, unsigned *arrivals,
                                         unsigned M, const Segments &s) {
    using Shape = projection::GemvShape<8, 1, KSPLIT, NB>;
    __shared__ projection::GemvShared<Shape, StreamSource> shared;
    constexpr unsigned long long kblocks = SEISMIC_DIM_H / 64;
    const unsigned long long outputs = s.rows[0] + s.rows[1];
    const unsigned long long groups[2] = {projection::gemv_segment_groups<Shape>(s.rows[0]),
                                          projection::gemv_segment_groups<Shape>(s.rows[1])};
    const unsigned long long per = groups[0] + groups[1];
    const unsigned long long tile_group = Shape::tile_group();
    const unsigned stream = (unsigned)(tile_group / per);
    unsigned long long group = tile_group % per;
    const int segment = projection::locate_segment(group, groups);
    const unsigned long long first = (unsigned long long)stream * SEISMIC_DIM_H;
    Stream pro = first_stream;
    pro.x += first;
    pro.norm += first * Norm::bytes;
    // Block-collective at M = 1: every warp forms the stream's row; the
    // stream's first block publishes its inverse.
    const bool first_block = tile_group % per < Shape::GROUPS;
    const auto x = [=] {
        seismic_dependency_start();
        if (M > 1)
            return StreamSource{staged + first * element::Act::bytes, WIDTH, projection::AllRows{}};
        projection::form_row_publishing(pro, 0, SEISMIC_DIM_H, row,
                                        first_block ? inverse + stream * inverse_stride : nullptr, 1);
        __syncthreads();
        return StreamSource{row, SEISMIC_DIM_H, projection::AllRows{}};
    };
    const StreamPartial partial{partials, SEISMIC_DIM_S * outputs, stream * outputs};
    const unsigned long long window = stream * (kblocks / 4);
    if (segment == 0)
        projection::gemv_segment_ready<Shape>(shared, x, M, kblocks, group, s.rows[0],
                                              projection::Window<packets::W0>(s.down, window),
                                              projection::NoWeight{}, partial);
    else
        projection::gemv_segment_ready<Shape>(shared, x, M, kblocks, group, s.rows[1],
                                              projection::Window<packets::W1>(s.inject, window),
                                              projection::NoWeight{},
                                              StreamPartial{partial.out, partial.row, partial.column + s.rows[0]});
    __shared__ unsigned last;
    if (!arrive::last(arrivals, gridDim.x, &last))
        return;
    for (unsigned long long item = threadIdx.x; item < M * outputs; item += blockDim.x) {
        const unsigned m = (unsigned)(item / outputs);
        const unsigned long long n = item % outputs;
        float sum = 0.0f;
        for (unsigned stream_ = 0; stream_ < SEISMIC_DIM_S; ++stream_)
            sum += __ldcg(partials + (m * SEISMIC_DIM_S + stream_) * outputs + n);
        if (n < s.rows[0])
            s.low(m, n, sum, 0.0f);
        else
            s.injected(m, n - s.rows[0], sum, 0.0f);
    }
}

// The segmented GEMM of one row band over the staged rows: block column
// `gemm_column` of down | inject.
template <class Shape>
__device__ __forceinline__ void mix_gemm(const projection::u8 *staged, unsigned M, const Segments &s) {
    extern __shared__ uint4 dynamic_shared[];
    seismic_dependency_start();
    projection::u8 *shared = reinterpret_cast<projection::u8 *>(dynamic_shared);
    unsigned long long column = projection::gemm_column<Shape>();
    const unsigned long long columns[2] = {projection::gemm_columns(s.rows[0]), projection::gemm_columns(s.rows[1])};
    switch (projection::locate_segment(column, columns)) {
    case 0:
        projection::gemm_run<Shape, false>(shared, staged, WIDTH, nullptr, M, WIDTH, column, s.rows[0], s.down,
                                           projection::NoWeight{}, s.low);
        break;
    case 1:
        projection::gemm_run<Shape, false>(shared, staged, WIDTH, nullptr, M, WIDTH, column, s.rows[1], s.inject,
                                           projection::NoWeight{}, s.injected);
        break;
    }
}

// Past one row: one block per row stages its normalized streams and
// publishes their inverses.
#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_STAGE
extern "C" __global__ void stream_mix_down_stage(SEISMIC_KERNEL_PARAMS) {
    seismic_dependency_start();
    projection::form_row_publishing(PROLOGUE, blockIdx.x, WIDTH, STAGING,
                                    INVERSE + (unsigned long long)blockIdx.x * SEISMIC_RESULT_2_STRIDE_0, 1);
}
#endif

#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_GEMV
template <unsigned KSPLIT>
__global__ void stream_mix_down_gemv(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    mix_gemv<1, KSPLIT>(STREAM, reinterpret_cast<projection::u8 *>(dynamic_shared), STAGING, INVERSE,
                        SEISMIC_RESULT_2_STRIDE_1, PARTIALS, ARRIVALS, (unsigned)SEISMIC_DIM_M, SEGMENTS);
}
#endif

#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_GEMV16
template <unsigned KSPLIT>
__global__ void stream_mix_down_gemv16(SEISMIC_KERNEL_PARAMS) {
    mix_gemv<2, KSPLIT>(STREAM, nullptr, STAGING, INVERSE, SEISMIC_RESULT_2_STRIDE_1, PARTIALS, ARRIVALS,
                        (unsigned)SEISMIC_DIM_M, SEGMENTS);
}
#endif

#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_GEMM_SMALL
extern "C" __global__ void stream_mix_down_gemm_small(SEISMIC_KERNEL_PARAMS) {
    mix_gemm<projection::SmallGemm>(STAGING, (unsigned)SEISMIC_DIM_M, SEGMENTS);
}
#endif

#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_GEMM
template <unsigned ROTATE>
__global__ void stream_mix_down_gemm(SEISMIC_KERNEL_PARAMS) {
    mix_gemm<projection::LargeGemm<ROTATE>>(STAGING, (unsigned)SEISMIC_DIM_M, SEGMENTS);
}
#endif
