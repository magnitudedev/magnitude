// stream_mix_down: the per-stream RMS prologue (`GroupRms`) over the S
// hyper-connection streams, then one segmented projection down | inject (an
// inject segment of zero rows, the exit site's, has no tiles), and the
// streams' inverses. Rows are staged by the staging launch past one row; a
// one-row GEMV block forms its row itself and the first block publishes the
// inverses.
#define KERNEL_W0 SEISMIC_DOWN_WEIGHT
#define KERNEL_W1 SEISMIC_INJECT_WEIGHT
#include "lib/projection/projection.cuh"

SEISMIC_PROGRAMMATIC_DEPENDENCY

using Pro = projection::GroupRms<ELEMENT_OF(SEISMIC_NORM), projection::AllRows, SEISMIC_DIM_S>;
using Source = projection::GemvSource<Pro>::type;
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

// The segmented GEMV over NB column blocks of 8 rows: down | inject.
template <int NB, int KSPLIT>
__device__ __forceinline__ void mix_gemv(const Pro &pro, projection::u8 *row, const projection::u8 *staged,
                                         float *inverse, unsigned M, const Segments &s) {
    using Shape = projection::GemvShape<8, 1, KSPLIT, NB>;
    __shared__ projection::GemvShared<Shape, Source> shared;
    const unsigned long long kblocks = WIDTH / 64;
    unsigned long long group = Shape::tile_group();
    const unsigned long long groups[2] = {projection::gemv_groups<Shape>(s.rows[0]),
                                          projection::gemv_groups<Shape>(s.rows[1])};
    const int segment = projection::locate_segment(group, groups);
    // Block-collective at M = 1: every warp forms the row, including a group
    // past the last segment.
    const auto x = [=] {
        seismic_dependency_start();
        if (M > 1)
            return Source{staged, WIDTH, projection::AllRows{}};
        projection::form_row_publishing(pro, 0, WIDTH, row, blockIdx.x == 0 ? inverse : nullptr, 1);
        __syncthreads();
        return Source{row, WIDTH, projection::AllRows{}};
    };
    switch (segment) {
    case 0:
        projection::gemv_segment_ready<Shape>(shared, x, M, kblocks, group, s.rows[0], s.down,
                                              projection::NoWeight{}, s.low);
        break;
    case 1:
        projection::gemv_segment_ready<Shape>(shared, x, M, kblocks, group, s.rows[1], s.inject,
                                              projection::NoWeight{}, s.injected);
        break;
    default:
        projection::gemv_segment_ready<Shape>(shared, x, M, kblocks, 0, 0, s.inject, projection::NoWeight{},
                                              s.injected);
        break;
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
    mix_gemv<1, KSPLIT>(PROLOGUE, reinterpret_cast<projection::u8 *>(dynamic_shared), STAGING, INVERSE,
                        (unsigned)SEISMIC_DIM_M, SEGMENTS);
}
#endif

#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_GEMV16
template <unsigned KSPLIT>
__global__ void stream_mix_down_gemv16(SEISMIC_KERNEL_PARAMS) {
    mix_gemv<2, KSPLIT>(PROLOGUE, nullptr, STAGING, INVERSE, (unsigned)SEISMIC_DIM_M, SEGMENTS);
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
