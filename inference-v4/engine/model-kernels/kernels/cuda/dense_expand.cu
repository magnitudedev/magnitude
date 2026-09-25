// dense_expand: RMS prologue over the `out_rows` rows of the F32
// residual, paired gate/up projection, SiLU(gate) * up epilogue. GEMV for
// O <= 16 (`gemv` to 8 rows, `gemv16` beyond; at O = 1 the block forms the A
// row in shared memory, else `stage` forms the A rows first), GEMM otherwise:
// `gemm_small` to 64 rows (A rows, or the INT8 candidate's q8_1 rows staged
// by `stage_s8`), `gemm` beyond (A rows). The declaration launches one of
// them.
#define KERNEL_W0 SEISMIC_GATE_WEIGHT
#define KERNEL_W1 SEISMIC_UP_WEIGHT
#include "lib/projection/projection.cuh"

constexpr bool S8 = SEISMIC_TUNE_INT8 == 1 && projection::quantizable<packets::W0, packets::W1>;
using Pro = projection::Rms<ELEMENT_OF(SEISMIC_NORM), projection::SelectedRows>;
using Source = projection::GemvSource<Pro>;
using Epi = projection::SiluMul<ELEMENT_OF(SEISMIC_ELEMENT_A)>;

#define PROLOGUE                                                                                          \
    Pro {                                                                                                 \
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)), SEISMIC_RESIDUAL_STRIDE_0,  \
            SEISMIC_PTR(SEISMIC_BUFFER_NORM), __uint_as_float((unsigned)SEISMIC_PARAM_EPS), SEISMIC_DIM_H, \
            projection::SelectedRows{reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT_ROWS))}  \
    }
#define STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define GATE KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_GATE_WEIGHT))
#define UP KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_UP_WEIGHT))
#define EPILOGUE Epi{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), SEISMIC_RESULT_0_STRIDE_0}

// The GEMV over NB column blocks of 8 rows.
template <int NB>
__device__ __forceinline__ void expand_gemv(const Pro &pro, projection::u8 *row, const projection::u8 *staged,
                                            unsigned M, unsigned long long H, unsigned long long F,
                                            const packets::W0 &gate, const packets::W1 &up, const Epi &epi) {
    using Shape = projection::GemvShape<8, 1, SEISMIC_TUNE_KSPLIT, NB>;
    __shared__ projection::GemvShared<Shape, Source::type> shared;
    const Source::type x = Source::make(pro, row, M, H, staged);
    const unsigned long long group = Shape::tile_group();
    if (group < projection::gemv_groups<Shape>(F))
        projection::gemv_segment<Shape>(shared, x, M, H / 64, group, F, gate, up, epi);
}

// The GEMM of one row band over the staged rows (A rows, or q8_1 rows with Q).
template <class Shape, bool Q>
__device__ __forceinline__ void expand_gemm(const projection::u8 *staged, const void *groups, unsigned M,
                                            unsigned long long H, unsigned long long F, const packets::W0 &gate,
                                            const packets::W1 &up, const Epi &epi) {
    extern __shared__ uint4 dynamic_shared[];
    if (blockIdx.x < projection::gemm_columns(F))
        projection::gemm_run<Shape, Q>(reinterpret_cast<projection::u8 *>(dynamic_shared), staged, H, groups, M, H,
                                       blockIdx.x, F, gate, up, epi);
}

extern "C" __global__ void dense_expand_stage(SEISMIC_KERNEL_PARAMS) {
    projection::stage_row<false>(PROLOGUE, blockIdx.x, SEISMIC_DIM_H, STAGING, GROUPS);
}

extern "C" __global__ void dense_expand_stage_s8(SEISMIC_KERNEL_PARAMS) {
    projection::stage_row<S8>(PROLOGUE, blockIdx.x, SEISMIC_DIM_H, STAGING, GROUPS);
}

extern "C" __global__ void dense_expand_gemv(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    expand_gemv<1>(PROLOGUE, reinterpret_cast<projection::u8 *>(dynamic_shared), STAGING, (unsigned)SEISMIC_DIM_O,
                   SEISMIC_DIM_H, SEISMIC_DIM_F, GATE, UP, EPILOGUE);
}

extern "C" __global__ void dense_expand_gemv16(SEISMIC_KERNEL_PARAMS) {
    expand_gemv<2>(PROLOGUE, nullptr, STAGING, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_H, SEISMIC_DIM_F, GATE, UP,
                   EPILOGUE);
}

extern "C" __global__ void dense_expand_gemm_small(SEISMIC_KERNEL_PARAMS) {
    expand_gemm<projection::SmallGemm, S8>(STAGING, GROUPS, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_H, SEISMIC_DIM_F, GATE,
                                           UP, EPILOGUE);
}

extern "C" __global__ void dense_expand_gemm(SEISMIC_KERNEL_PARAMS) {
    expand_gemm<projection::LargeGemm, false>(STAGING, GROUPS, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_H, SEISMIC_DIM_F,
                                              GATE, UP, EPILOGUE);
}
