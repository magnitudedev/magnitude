// short_conv_project: RMS prologue over every residual row, then
// `dense_expand`'s bands over two segments along x: the u segment pairs the
// B and X streams into u = B * X (F32, columns 0..CH), the C segment stores C
// (F32, columns CH..2CH).
#define KERNEL_W0 SEISMIC_B_WEIGHT
#define KERNEL_W1 SEISMIC_X_WEIGHT
#define KERNEL_W2 SEISMIC_C_WEIGHT
#include "lib/projection/projection.cuh"

using Pro = projection::Rms<ELEMENT_OF(SEISMIC_NORM), projection::AllRows>;
using Source = projection::GemvSource<Pro>;
using Product = projection::Mul;
using Gate = projection::Store<element::F32>;

#define PROLOGUE                                                                                          \
    Pro {                                                                                                 \
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)), SEISMIC_RESIDUAL_STRIDE_0,  \
            SEISMIC_PTR(SEISMIC_BUFFER_NORM), __uint_as_float((unsigned)SEISMIC_PARAM_EPS), SEISMIC_DIM_H, \
            projection::AllRows{}                                                                         \
    }
#define STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define B_ROWS KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_B_WEIGHT))
#define X_ROWS KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_X_WEIGHT))
#define C_ROWS KERNEL_W2_AT(SEISMIC_PTR(SEISMIC_BUFFER_C_WEIGHT))
#define PRODUCT Product{reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)), SEISMIC_RESULT_0_STRIDE_0}
#define GATE Gate{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), SEISMIC_RESULT_0_STRIDE_0, SEISMIC_DIM_CH}

// The weights and epilogues of both segments (formed in the kernel body,
// where the ABI macros are defined).
struct Segments {
    packets::W0 b;
    packets::W1 x;
    packets::W2 c;
    Product product;
    Gate gate;
};
#define SEGMENTS Segments{B_ROWS, X_ROWS, C_ROWS, PRODUCT, GATE}

// The GEMV over NB column blocks of 8 rows: tile groups [0, S) are the u
// segment's, the rest the C segment's.
template <int NB, int KSPLIT>
__device__ __forceinline__ void project_gemv(const Pro &pro, projection::u8 *row, const projection::u8 *staged,
                                             unsigned M, unsigned long long H, unsigned long long CH,
                                             const Segments &segments) {
    using Shape = projection::GemvShape<8, 1, KSPLIT, NB>;
    __shared__ projection::GemvShared<Shape, Source::type> shared;
    const Source::type x = Source::make(pro, row, M, H, staged);
    const unsigned long long group = Shape::tile_group(), segment = projection::gemv_groups<Shape>(CH);
    if (group < segment)
        projection::gemv_segment<Shape>(shared, x, M, H / 64, group, CH, segments.b, segments.x, segments.product);
    else if (group < 2 * segment)
        projection::gemv_segment<Shape>(shared, x, M, H / 64, group - segment, CH, segments.c, projection::NoWeight{},
                                        segments.gate);
}

// The GEMM of one row band over the staged rows (A rows, or q8_1 rows with Q).
template <class Shape, bool Q>
__device__ __forceinline__ void project_gemm(const projection::u8 *staged, const void *groups, unsigned M,
                                             unsigned long long H, unsigned long long CH, const Segments &segments) {
    extern __shared__ uint4 dynamic_shared[];
    projection::u8 *shared = reinterpret_cast<projection::u8 *>(dynamic_shared);
    const unsigned long long segment = projection::gemm_columns(CH);
    const unsigned long long column = projection::gemm_column<Shape>();
    if (column < segment)
        projection::gemm_run<Shape, Q>(shared, staged, H, groups, M, H, column, CH, segments.b, segments.x,
                                       segments.product);
    else if (column < 2 * segment)
        projection::gemm_run<Shape, Q>(shared, staged, H, groups, M, H, column - segment, CH, segments.c,
                                       projection::NoWeight{}, segments.gate);
}

#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_STAGE
template <unsigned INT8>
__global__ void short_conv_project_stage(SEISMIC_KERNEL_PARAMS) {
    projection::stage_row<false>(PROLOGUE, blockIdx.x, SEISMIC_DIM_H, STAGING, GROUPS);
}
#endif

#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_STAGE_S8
template <unsigned INT8>
__global__ void short_conv_project_stage_s8(SEISMIC_KERNEL_PARAMS) {
    constexpr bool S8 = INT8 == 1 && projection::quantizable<packets::W0, packets::W1, packets::W2>;
    projection::stage_row<S8>(PROLOGUE, blockIdx.x, SEISMIC_DIM_H, STAGING, GROUPS);
}
#endif

#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_GEMV
template <unsigned KSPLIT>
__global__ void short_conv_project_gemv(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    project_gemv<1, KSPLIT>(PROLOGUE, reinterpret_cast<projection::u8 *>(dynamic_shared), STAGING,
                            (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_H, SEISMIC_DIM_CH, SEGMENTS);
}
#endif

#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_GEMV16
template <unsigned KSPLIT>
__global__ void short_conv_project_gemv16(SEISMIC_KERNEL_PARAMS) {
    project_gemv<2, KSPLIT>(PROLOGUE, nullptr, STAGING, (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_H, SEISMIC_DIM_CH,
                            SEGMENTS);
}
#endif

#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_GEMM_SMALL
template <unsigned INT8>
__global__ void short_conv_project_gemm_small(SEISMIC_KERNEL_PARAMS) {
    constexpr bool S8 = INT8 == 1 && projection::quantizable<packets::W0, packets::W1, packets::W2>;
    project_gemm<projection::SmallGemm, S8>(STAGING, GROUPS, (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_H, SEISMIC_DIM_CH,
                                            SEGMENTS);
}
#endif

#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_GEMM
template <unsigned ROTATE>
__global__ void short_conv_project_gemm(SEISMIC_KERNEL_PARAMS) {
    project_gemm<projection::LargeGemm<ROTATE>, false>(STAGING, GROUPS, (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_H,
                                                       SEISMIC_DIM_CH, SEGMENTS);
}
#endif
