// qwen_dense_expand: RMS prologue over the `out_rows` rows of the F32
// residual, paired gate/up projection, SiLU(gate) * up epilogue. GEMV for
// O <= 8 (each block forms the A rows in shared memory; q8_1 rows staged
// first for the INT8 candidate), GEMM otherwise (A rows or q8_1 rows staged
// first). The declaration launches one of the two.
#define SJ_W0 SEISMIC_GATE_WEIGHT
#define SJ_W1 SEISMIC_UP_WEIGHT
#include "common/projection.cuh"

constexpr bool S8 = SEISMIC_TUNE_INT8 == 1;
using Shape = sj::GemvShape<8, 1, SEISMIC_TUNE_KSPLIT>;
using GShape = sj::GemmShape<SEISMIC_TUNE_BM, 2, 4, 4 - SEISMIC_TUNE_BM / 64>;
using Pro = sj::Rms<SJ_DENSE_KIND(SEISMIC_NORM), sj::SelectedRows>;
using Source = sj::GemvSource<S8, Pro>;
using Epi = sj::SiluMul<SJ_DENSE_KIND(SEISMIC_ELEMENT_A)>;

#define PROLOGUE                                                                                          \
    Pro {                                                                                                 \
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)), SEISMIC_RESIDUAL_STRIDE_0,  \
            SEISMIC_PTR(SEISMIC_BUFFER_NORM), __uint_as_float((unsigned)SEISMIC_PARAM_EPS), SEISMIC_DIM_H, \
            sj::SelectedRows{reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT_ROWS))}         \
    }
#define STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define GATE SJ_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_GATE_WEIGHT))
#define UP SJ_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_UP_WEIGHT))
#define EPILOGUE Epi{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), SEISMIC_RESULT_0_STRIDE_0}

extern "C" __global__ void qwen_dense_expand_stage(SEISMIC_KERNEL_PARAMS) {
    sj::stage_row<S8>(PROLOGUE, blockIdx.x, SEISMIC_DIM_H, STAGING, GROUPS);
}

extern "C" __global__ void qwen_dense_expand_gemv(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    __shared__ sj::GemvShared<Shape, Source::type> shared;
    const Source::type x = Source::make(PROLOGUE, reinterpret_cast<sj::u8 *>(dynamic_shared), (unsigned)SEISMIC_DIM_O,
                                        SEISMIC_DIM_H, STAGING, GROUPS);
    const unsigned long long group = Shape::tile_group();
    if (group < sj::gemv_groups<Shape>(SEISMIC_DIM_F))
        sj::gemv_segment<Shape>(shared, x, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_H / 64, group, SEISMIC_DIM_F, GATE, UP,
                                EPILOGUE);
}

extern "C" __global__ void qwen_dense_expand_gemm(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    if (blockIdx.x < sj::gemm_columns(SEISMIC_DIM_F))
        sj::gemm_run<GShape, S8>(reinterpret_cast<sj::u8 *>(dynamic_shared), STAGING, SEISMIC_DIM_H, GROUPS,
                                 (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_H, blockIdx.x, SEISMIC_DIM_F, GATE, UP, EPILOGUE);
}
