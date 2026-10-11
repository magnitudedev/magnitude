// Prepared feature readout shares the packed projection primitives with
// raw-RMS readout. Full rows reuse weights across matrix tiles; indexed
// rows project one canonical packed row through the same exact-code GEMV.
// Indexed rows keep the physical packet view intact and select a canonical
// row from its GEMV tile. Packet strides and row-major element strides are
// distinct contracts; retargeting their base as a scalar row is invalid.
#define KERNEL_W0 SEISMIC_WEIGHT
#include "lib/projection/projection.cuh"

using Pro = projection::Strided<element::Act, projection::AllRows>;
using Source = projection::GemvSource<Pro>;
using Scaling = projection::scaling<(SEISMIC_DIM_WS != 0)>;
using Epi = Scaling::type<projection::Logits>;

#define PROLOGUE Pro{SEISMIC_PTR(SEISMIC_BUFFER_FEATURES), SEISMIC_FEATURES_STRIDE_0, \
                     SEISMIC_FEATURES_STRIDE_1, projection::AllRows{}}
#define STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define HEAD KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHT))
#define EPILOGUE                                                                                              \
    Scaling::wrap(projection::Logits{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), SEISMIC_RESULT_0_STRIDE_0,              \
                                     __uint_as_float((unsigned)SEISMIC_PARAM_SOFTCAP)},                           \
                  projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHT_SCALE), SEISMIC_DIM_WS, 0, 0), 1.0f)

// The GEMV over NB column blocks of 8 rows.
template <int NB, int KSPLIT>
__device__ __forceinline__ void head_gemv(const Pro &pro, projection::u8 *row, const projection::u8 *staged, unsigned O,
                                          unsigned long long D, unsigned long long V, const packets::W0 &head,
                                          const Epi &epi) {
    using Shape = projection::GemvShape<8, 1, KSPLIT, NB>;
    __shared__ projection::GemvShared<Shape, Source::type> shared;
    const Source::type x = Source::make(pro, row, O, D, staged);
    const unsigned long long group = Shape::tile_group();
    if (group < projection::gemv_groups<Shape>(V))
        projection::gemv_segment<Shape>(shared, x, O, D / 64, group, V, head, projection::NoWeight{}, epi);
}

// The GEMM of one row band over the staged A rows.
template <class Shape, class Operands>
__device__ __forceinline__ void head_gemm(const projection::u8 *staged, unsigned O, unsigned long long D,
                                          unsigned long long V, const packets::W0 &head, const Epi &epi) {
    extern __shared__ uint4 dynamic_shared[];
    const unsigned long long column = projection::gemm_column<Shape>();
    if (column < projection::gemm_columns(V))
        projection::gemm_run<Shape, false, Operands>(reinterpret_cast<projection::u8 *>(dynamic_shared), staged, D, nullptr, O,
                                           D, column, V, head, projection::NoWeight{}, epi);
}

#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_STAGE
template <unsigned OPERAND>
__global__ void readout_project_rows_stage(SEISMIC_KERNEL_PARAMS) {
    if constexpr (SEISMIC_DIM_I == 1)
        projection::form_row(PROLOGUE, blockIdx.x, SEISMIC_DIM_D, STAGING);
    else
        projection::stage_projection_row<false, typename projection::GemmOperands<OPERAND>::type>(PROLOGUE, blockIdx.x, SEISMIC_DIM_O, SEISMIC_DIM_D, STAGING, nullptr);
}
#endif

#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_GEMV
template <unsigned KSPLIT>
__global__ void readout_project_rows_gemv(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    head_gemv<1, KSPLIT>(PROLOGUE, reinterpret_cast<projection::u8 *>(dynamic_shared), STAGING, (unsigned)SEISMIC_DIM_O,
                 SEISMIC_DIM_D, SEISMIC_DIM_S, HEAD, EPILOGUE);
}
#endif

#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_GEMV16
template <unsigned KSPLIT>
__global__ void readout_project_rows_gemv16(SEISMIC_KERNEL_PARAMS) {
    head_gemv<2, KSPLIT>(PROLOGUE, nullptr, STAGING, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_D, SEISMIC_DIM_S, HEAD, EPILOGUE);
}
#endif

#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_GEMM_SMALL
template <unsigned OPERAND>
__global__ void readout_project_rows_gemm_small(SEISMIC_KERNEL_PARAMS) {
    head_gemm<projection::SmallGemm, typename projection::GemmOperands<OPERAND>::type>(STAGING, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_D, SEISMIC_DIM_S, HEAD, EPILOGUE);
}
#endif

#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_GEMM
template <unsigned OPERAND, unsigned ROTATE>
__global__ void readout_project_rows_gemm(SEISMIC_KERNEL_PARAMS) {
    head_gemm<projection::LargeGemm<ROTATE>, typename projection::GemmOperands<OPERAND>::type>(STAGING, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_D, SEISMIC_DIM_S, HEAD,
                                             EPILOGUE);
}
#endif

#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_INDEXED
struct IndexedLogits {
    projection::Logits output;
    projection::u64 canonical;
    __device__ __forceinline__ void operator()(projection::u32 m, projection::u64 n, float first, float second) const {
        if (n == canonical) output(0, 0, first, second);
    }
};
template <unsigned KSPLIT>
__global__ void readout_project_rows_indexed(SEISMIC_KERNEL_PARAMS) {
    using Shape = projection::GemvShape<8, 1, KSPLIT, 1>;
    using IndexedSource = projection::Plain<element::Act, projection::AllRows>;
    __shared__ projection::GemvShared<Shape, IndexedSource> shared;
    const projection::u64 column = blockIdx.x;
    const projection::u64 row = blockIdx.y;
    const int vocabulary = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKEN_ROWS))[
        row * SEISMIC_TOKEN_ROWS_STRIDE_0 + column * SEISMIC_TOKEN_ROWS_STRIDE_2];
    if (vocabulary < 0 || (projection::u64)vocabulary >= SEISMIC_DIM_V) {
        if (threadIdx.x == 0)
            reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER))[
                row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1] = __int_as_float(0xff800000);
        return;
    }
    const auto weight = HEAD;
    const IndexedSource source{STAGING + row * SEISMIC_DIM_D * 2, SEISMIC_DIM_D, projection::AllRows{}};
    const auto epilogue = Scaling::wrap(IndexedLogits{projection::Logits{
        SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER) + 4 * (row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1),
        0, __uint_as_float((unsigned)SEISMIC_PARAM_SOFTCAP)}, (projection::u64)vocabulary},
        projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHT_SCALE), SEISMIC_DIM_WS, 0, 0), 1.0f);
    // Only one weight-row group is live. Its named reduction barrier covers
    // precisely KSPLIT warps, as in the full canonical-row GEMV.
    if (Shape::group() == 0)
        projection::gemv_segment<Shape>(shared, source, 1, SEISMIC_DIM_D / 64,
                                       (projection::u64)vocabulary / 16, SEISMIC_DIM_V,
                                       weight, projection::NoWeight{}, epilogue);
}
#endif
