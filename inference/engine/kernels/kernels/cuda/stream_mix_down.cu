// stream_mix_down: the per-stream RMS prologue (`GroupRms`) over the S
// hyper-connection streams, then the `dense` projection (segmented, K in S
// parts, the streams) down | inject (an inject segment of zero rows, the
// exit site's, has no tiles), and the streams' inverses. Rows are staged by
// the staging launch past one row. The GEMV classes are the projection's
// parts GEMV (a one-row block forms its stream's `shared_row`, and the
// stream's first block publishes its inverse), the GEMMs its segmented tiles
// (`lib/projection/operator.cuh`).
#define KERNEL_W0 SEISMIC_DOWN_WEIGHT
#define KERNEL_W1 SEISMIC_INJECT_WEIGHT
#define PROJECTION_PARTS SEISMIC_DIM_S
#include "lib/core/arrive.cuh"
#include "lib/projection/operator.cuh"

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
#define STAGED_ROWS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define INVERSE SEISMIC_RESULT_2_OUT

using Norm = ELEMENT_OF(SEISMIC_NORM);
using Stream = projection::GroupRms<Norm, projection::AllRows, 1>;
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
#define SEGMENT_ROWS SEISMIC_DIM_LR, SEISMIC_DIM_C * SEISMIC_DIM_S
using Segments = projection::op::SegmentSet<0, projection::op::Types<packets::W0, packets::W1>,
                                            projection::op::Types<Low, projection::MixInject>,
                                            projection::op::empty_mask(SEGMENT_ROWS)>;
#define SEGMENTS                                                                                          \
    Segments {                                                                                            \
        {SEGMENT_ROWS},                                                                                   \
            {KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_DOWN_WEIGHT)),                                        \
             KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_INJECT_WEIGHT))},                                     \
        {Low{SEISMIC_RESULT_0_OUT, SEISMIC_RESULT_0_STRIDE_0, (float)SEISMIC_DIM_S},       \
         projection::MixInject{SEISMIC_RESULT_1_OUT,            \
                               SEISMIC_RESULT_1_STRIDE_0, (float)SEISMIC_DIM_S}}                          \
    }

// The parts GEMV over NB column blocks of 8 rows: stream `part`'s rows, staged
// past one row, else the block's `shared_row` (block-collective), the
// stream's first block publishing its inverse.
template <int NB, int KSPLIT>
__device__ __forceinline__ void mix_gemv(const Stream streams, const projection::u8 *staged,
                                         element::Out<element::F32> inverse, unsigned long long inverse_stride,
                                         float *partials, unsigned *arrivals, unsigned M, const Segments &segments) {
    const auto row = [=](unsigned part, bool first) {
        const unsigned long long offset = (unsigned long long)part * SEISMIC_DIM_H;
        Stream pro = streams;
        pro.x += offset;
        pro.norm += offset * Norm::bytes;
        return [=] {
            seismic_dependency_start();
            const element::Out<element::F32> part_inverse = inverse.offset(part * inverse_stride);
            return projection::shared_row(pro, SEISMIC_DIM_H, first ? &part_inverse : nullptr);
        };
    };
    const auto rows = [=](unsigned part) {
        return [=] {
            seismic_dependency_start();
            return projection::Plain<element::Act, projection::AllRows>{
                staged + (unsigned long long)part * SEISMIC_DIM_H * element::Act::bytes, WIDTH, projection::AllRows{}};
        };
    };
    projection::op::gemv_parts<NB, KSPLIT, SEISMIC_DIM_S, SEISMIC_DIM_H, Stream>(row, rows, partials, arrivals, M,
                                                                                 segments);
}

// The projection's segmented GEMM of one row band over the staged rows.
template <class Shape>
__device__ __forceinline__ void mix_gemm(const projection::u8 *staged, unsigned M, const Segments &segments) {
    extern __shared__ uint4 dynamic_shared[];
    projection::op::gemm_segmented<Shape, false, false, projection::SemanticGemmOperands>(
        reinterpret_cast<projection::u8 *>(dynamic_shared), staged, nullptr, M, WIDTH, segments, 0u);
}

// Past one row: one block per row stages its normalized streams and
// publishes their inverses.
#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_STAGE
template <unsigned INT8, unsigned OPERAND>
__global__ void stream_mix_down_stage(SEISMIC_KERNEL_PARAMS) {
    seismic_dependency_start();
    const element::Out<element::F32> inverse = INVERSE.offset((unsigned long long)blockIdx.x * SEISMIC_RESULT_2_STRIDE_0);
    projection::form_row(PROLOGUE, blockIdx.x, WIDTH, STAGED_ROWS, &inverse, 1);
}
#endif

#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_GEMV
template <unsigned INT8, unsigned STAGING, unsigned KSPLIT>
__global__ void stream_mix_down_gemv(SEISMIC_KERNEL_PARAMS) {
    static_assert(INT8 == 0, "a parts projection's GEMV takes no int8_rule rows (`where`: PARTED == 0)");
    mix_gemv<1, KSPLIT>(STREAM, STAGED_ROWS, INVERSE, SEISMIC_RESULT_2_STRIDE_1, PARTIALS, ARRIVALS,
                        (unsigned)SEISMIC_DIM_M, SEGMENTS);
}
#endif

#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_GEMV16
template <unsigned INT8, unsigned STAGING, unsigned KSPLIT>
__global__ void stream_mix_down_gemv16(SEISMIC_KERNEL_PARAMS) {
    static_assert(INT8 == 0, "a parts projection's GEMV takes no int8_rule rows (`where`: PARTED == 0)");
    mix_gemv<2, KSPLIT>(STREAM, STAGED_ROWS, INVERSE, SEISMIC_RESULT_2_STRIDE_1, PARTIALS, ARRIVALS,
                        (unsigned)SEISMIC_DIM_M, SEGMENTS);
}
#endif

#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_GEMM_SMALL
template <unsigned INT8, unsigned OPERAND>
__global__ void stream_mix_down_gemm_small(SEISMIC_KERNEL_PARAMS) {
    mix_gemm<projection::SmallGemm>(STAGED_ROWS, (unsigned)SEISMIC_DIM_M, SEGMENTS);
}
#endif

#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_GEMM
template <unsigned INT8, unsigned OPERAND, unsigned ROTATE, unsigned BLOCKS>
__launch_bounds__(256, BLOCKS) __global__ void stream_mix_down_gemm(SEISMIC_KERNEL_PARAMS) {
    mix_gemm<projection::LargeGemm<ROTATE>>(STAGED_ROWS, (unsigned)SEISMIC_DIM_M, SEGMENTS);
}
#endif
