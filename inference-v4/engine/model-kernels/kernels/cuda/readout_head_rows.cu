// readout_head_rows: final RMS prologue over the `out_rows` rows, then the
// vocabulary projection into F32 logits (a K1 projection). GEMV for O <= 16
// (`gemv` to 8 rows, `gemv16` beyond; at O = 1 the block forms the A row in
// shared memory, else `stage` forms the A rows first), GEMM otherwise over
// the A rows (`gemm_small` to 64 rows, `gemm` beyond; the 16-bit path: head
// GEMMs are rare, so the INT8 candidate is not offered).
#define KERNEL_W0 SEISMIC_WEIGHT
#include "lib/projection/projection.cuh"

using Pro = projection::Rms<ELEMENT_OF(SEISMIC_NORM), projection::SelectedRows>;
using Source = projection::GemvSource<Pro>;
using Epi = projection::Store<element::F32>;

#define PROLOGUE                                                                                              \
    Pro {                                                                                                     \
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)), SEISMIC_HIDDEN_STRIDE_0,          \
            SEISMIC_PTR(SEISMIC_BUFFER_NORM), __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON), SEISMIC_DIM_D, \
            projection::SelectedRows{reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT_ROWS))}             \
    }
#define STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define HEAD KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHT))
#define EPILOGUE Epi{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), SEISMIC_RESULT_0_STRIDE_0, 0}

// The GEMV over NB column blocks of 8 rows.
template <int NB>
__device__ __forceinline__ void head_gemv(const Pro &pro, projection::u8 *row, const projection::u8 *staged, unsigned O,
                                          unsigned long long D, unsigned long long V, const packets::W0 &head,
                                          const Epi &epi) {
    using Shape = projection::GemvShape<8, 1, SEISMIC_TUNE_KSPLIT, NB>;
    __shared__ projection::GemvShared<Shape, Source::type> shared;
    const Source::type x = Source::make(pro, row, O, D, staged);
    const unsigned long long group = Shape::tile_group();
    if (group < projection::gemv_groups<Shape>(V))
        projection::gemv_segment<Shape>(shared, x, O, D / 64, group, V, head, projection::NoWeight{}, epi);
}

// The GEMM of one row band over the staged A rows.
template <class Shape>
__device__ __forceinline__ void head_gemm(const projection::u8 *staged, unsigned O, unsigned long long D,
                                          unsigned long long V, const packets::W0 &head, const Epi &epi) {
    extern __shared__ uint4 dynamic_shared[];
    if (blockIdx.x < projection::gemm_columns(V))
        projection::gemm_run<Shape, false>(reinterpret_cast<projection::u8 *>(dynamic_shared), staged, D, nullptr, O,
                                           D, blockIdx.x, V, head, projection::NoWeight{}, epi);
}

extern "C" __global__ void readout_head_rows_stage(SEISMIC_KERNEL_PARAMS) {
    projection::stage_row<false>(PROLOGUE, blockIdx.x, SEISMIC_DIM_D, STAGING, nullptr);
}

extern "C" __global__ void readout_head_rows_gemv(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    head_gemv<1>(PROLOGUE, reinterpret_cast<projection::u8 *>(dynamic_shared), STAGING, (unsigned)SEISMIC_DIM_O,
                 SEISMIC_DIM_D, SEISMIC_DIM_V, HEAD, EPILOGUE);
}

extern "C" __global__ void readout_head_rows_gemv16(SEISMIC_KERNEL_PARAMS) {
    head_gemv<2>(PROLOGUE, nullptr, STAGING, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_D, SEISMIC_DIM_V, HEAD, EPILOGUE);
}

extern "C" __global__ void readout_head_rows_gemm_small(SEISMIC_KERNEL_PARAMS) {
    head_gemm<projection::SmallGemm>(STAGING, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_D, SEISMIC_DIM_V, HEAD, EPILOGUE);
}

extern "C" __global__ void readout_head_rows_gemm(SEISMIC_KERNEL_PARAMS) {
    head_gemm<projection::LargeGemm>(STAGING, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_D, SEISMIC_DIM_V, HEAD, EPILOGUE);
}
