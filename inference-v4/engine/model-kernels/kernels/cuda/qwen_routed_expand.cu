// Decode expansion (M <= 8); the CUDA form of `metal/qwen_routed_expand.metal`.
// Block row y < M * K is choice (m, k): the K1 paired GEMV over expert
// routes[m, k]'s gate/up rows for one activation row. Row y = M * K is the
// shared expert over all M rows.
#define SJ_W0 SEISMIC_EXPERT_GATE
#define SJ_W1 SEISMIC_EXPERT_UP
#define SJ_W2 SEISMIC_SHARED_GATE
#define SJ_W3 SEISMIC_SHARED_UP
#include "common/projection.cuh"

using Shape = sj::GemvShape<4, SEISMIC_TUNE_TPW, SEISMIC_TUNE_KSPLIT>;
using Pro = sj::Plain<SJ_DENSE_KIND(SEISMIC_ELEMENT_A), sj::AllRows>;
using Epi = sj::SiluMul<SJ_DENSE_KIND(SEISMIC_ELEMENT_A)>;

// First stored row of expert `expert` in an [E, N, K] mma16 tensor: each
// expert's N rows are padded to whole 16-row tiles.
__device__ __forceinline__ sj::u64 routed_expert_row(sj::u64 expert, sj::u64 rows) {
    return expert * ((rows + 15) / 16 * 16);
}

extern "C" __global__ void qwen_routed_expand(SEISMIC_KERNEL_PARAMS) {
    __shared__ sj::GemvShared<Shape, Pro> shared;
    const sj::u8 *normalized = SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED);
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    const sj::u64 group = Shape::tile_group();
    const sj::u64 choices = SEISMIC_DIM_M * SEISMIC_DIM_K;

    if (blockIdx.y < choices) {
        if (group >= sj::gemv_groups<Shape>(SEISMIC_DIM_F))
            return;
        const sj::u64 m = blockIdx.y / SEISMIC_DIM_K, k = blockIdx.y % SEISMIC_DIM_K;
        const sj::u64 expert = (sj::u64)routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1];
        const sj::u64 first = routed_expert_row(expert, SEISMIC_DIM_F);
        const Pro pro{normalized + m * SEISMIC_NORMALIZED_STRIDE_0 * 2, SEISMIC_NORMALIZED_STRIDE_0, sj::AllRows{}};
        const Epi epi{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)
                          + (m * SEISMIC_RESULT_0_STRIDE_0 + k * SEISMIC_RESULT_0_STRIDE_1) * 2,
                      0};
        sj::gemv_segment<Shape>(shared, pro, 1u, SEISMIC_DIM_H / 64, group, SEISMIC_DIM_F,
                                SJ_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_GATE) + first * SJ_CAT(SJ_W0, _ROW_STRIDE_BYTES)),
                                SJ_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_UP) + first * SJ_CAT(SJ_W1, _ROW_STRIDE_BYTES)),
                                epi);
        return;
    }

    if (group >= sj::gemv_groups<Shape>(SEISMIC_DIM_S))
        return;
    const Pro pro{normalized, SEISMIC_NORMALIZED_STRIDE_0, sj::AllRows{}};
    const Epi epi{SEISMIC_PTR(SEISMIC_RESULT_1_BUFFER), SEISMIC_RESULT_1_STRIDE_0};
    sj::gemv_segment<Shape>(shared, pro, (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_H / 64, group, SEISMIC_DIM_S,
                            SJ_W2_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_GATE)),
                            SJ_W3_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_UP)), epi);
}
