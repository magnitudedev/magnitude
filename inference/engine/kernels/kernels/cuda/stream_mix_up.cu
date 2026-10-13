// stream_mix_up: the low-rank rows (read in place) through `up`, each output
// column the gate of its stream's column, publishing the per-stream
// partials (`Mix`).
#define KERNEL_W0 SEISMIC_UP_WEIGHT
#include "lib/projection/projection.cuh"

SEISMIC_PROGRAMMATIC_DEPENDENCY

using Pro = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Epi = projection::Mix<ELEMENT_OF(SEISMIC_NORM)>;

#define WIDTH (SEISMIC_DIM_S * SEISMIC_DIM_H)
#define LOW_ROWS Pro{SEISMIC_PTR(SEISMIC_BUFFER_LOW), SEISMIC_LOW_STRIDE_0, projection::AllRows{}}
#define UP KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_UP_WEIGHT))
#define EPILOGUE                                                                                         \
    Epi {                                                                                                \
        reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)), SEISMIC_RESULT_0_STRIDE_0,      \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)), SEISMIC_HIDDEN_STRIDE_0, \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_INVERSE)), SEISMIC_INVERSE_STRIDE_0, \
            SEISMIC_PTR(SEISMIC_BUFFER_NORM), SEISMIC_DIM_H, (float)SEISMIC_DIM_S                        \
    }

// The GEMV over NB column blocks of 8 rows.
template <int NB, int KSPLIT>
__device__ __forceinline__ void mix_gemv(const Pro &low, unsigned M, const packets::W0 &up, const Epi &epi) {
    using Shape = projection::GemvShape<8, 1, KSPLIT, NB>;
    __shared__ projection::GemvShared<Shape, Pro> shared;
    projection::gemv_segment_ready<Shape>(shared, projection::after_dependency(low), M, SEISMIC_DIM_LR / 64,
                                          Shape::tile_group(), WIDTH, up, projection::NoWeight{}, epi);
}

template <class Shape>
__device__ __forceinline__ void mix_gemm(const projection::u8 *low, projection::u64 low_stride, unsigned M,
                                         const packets::W0 &up, const Epi &epi) {
    extern __shared__ uint4 dynamic_shared[];
    seismic_dependency_start();
    const unsigned long long column = projection::gemm_column<Shape>();
    if (column < projection::gemm_columns(WIDTH))
        projection::gemm_run<Shape, false>(reinterpret_cast<projection::u8 *>(dynamic_shared),
                                           low, low_stride, nullptr, M,
                                           SEISMIC_DIM_LR, column, WIDTH, up, projection::NoWeight{}, epi);
}

#ifdef SEISMIC_FORMING_STREAM_MIX_UP_GEMV
template <unsigned KSPLIT>
__global__ void stream_mix_up_gemv(SEISMIC_KERNEL_PARAMS) {
    mix_gemv<1, KSPLIT>(LOW_ROWS, (unsigned)SEISMIC_DIM_M, UP, EPILOGUE);
}
#endif

#ifdef SEISMIC_FORMING_STREAM_MIX_UP_GEMV16
template <unsigned KSPLIT>
__global__ void stream_mix_up_gemv16(SEISMIC_KERNEL_PARAMS) {
    mix_gemv<2, KSPLIT>(LOW_ROWS, (unsigned)SEISMIC_DIM_M, UP, EPILOGUE);
}
#endif

#ifdef SEISMIC_FORMING_STREAM_MIX_UP_GEMM_SMALL
extern "C" __global__ void stream_mix_up_gemm_small(SEISMIC_KERNEL_PARAMS) {
    mix_gemm<projection::SmallGemm>(SEISMIC_PTR(SEISMIC_BUFFER_LOW), SEISMIC_LOW_STRIDE_0, (unsigned)SEISMIC_DIM_M, UP,
                                    EPILOGUE);
}
#endif

#ifdef SEISMIC_FORMING_STREAM_MIX_UP_GEMM
template <unsigned ROTATE>
__global__ void stream_mix_up_gemm(SEISMIC_KERNEL_PARAMS) {
    mix_gemm<projection::LargeGemm<ROTATE>>(SEISMIC_PTR(SEISMIC_BUFFER_LOW), SEISMIC_LOW_STRIDE_0,
                                            (unsigned)SEISMIC_DIM_M, UP, EPILOGUE);
}
#endif
