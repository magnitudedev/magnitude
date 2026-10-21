// readout_project_rows: the vocabulary projection of already-normalized feature
// rows (the draft head's readout) into F32 logits.
#define KERNEL_W0 SEISMIC_WEIGHT
#include "lib/projection/projection.h"

typedef element::Act activation;
// The weight's operand row (the feature row, or NVFP4 operands), formed
// while the GEMV stages it.
constant constexpr bool NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4;
typedef projection::operands<NVFP4> Operands;

#define PROJECT_ROWS_ARGUMENTS \
    device const uchar *features [[buffer(SEISMIC_BUFFER_FEATURES)]], \
    device const uchar *weight [[buffer(SEISMIC_BUFFER_WEIGHT)]], \
    device const int *token_rows [[buffer(SEISMIC_BUFFER_TOKEN_ROWS)]], \
    device const float *weight_scale [[buffer(SEISMIC_BUFFER_WEIGHT_SCALE)]], \
    device float *logits [[buffer(SEISMIC_RESULT_0_BUFFER)]], \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_GEMV
template <uint ROWS, uint LANES>
kernel void readout_project_rows_gemv(PROJECT_ROWS_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint d = uint(SEISMIC_DIM_D);
    const uint row = tile.y;
    const projection::Plain<activation, projection::AllRows> in{
        features + ulong(row) * SEISMIC_FEATURES_STRIDE_0 * 2u,
        SEISMIC_FEATURES_STRIDE_0, SEISMIC_FEATURES_STRIDE_1, d, {}};
    device const int *selected = nullptr;
    ulong column = 0;
    if constexpr (SEISMIC_DIM_I == 1) {
        column = tile.x;
        selected = token_rows + ulong(row) * SEISMIC_TOKEN_ROWS_STRIDE_0 + column * SEISMIC_TOKEN_ROWS_STRIDE_2;
        if (*selected < 0 || ulong(*selected) >= SEISMIC_DIM_V) {
            if (sg == 0 && lane == 0) logits[ulong(row) * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1] = -INFINITY;
            return;
        }
    }
    projection::Weights<packets::W0> w{weight, KERNEL_W0_LAYOUT(d), d, selected, 0,
        SEISMIC_TOKEN_ROWS_STRIDE_2};
    const auto out = projection::scaling<(SEISMIC_DIM_WS != 0)>::wrap(
        projection::Logits{reinterpret_cast<device uchar *>(logits + ulong(row) * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1),
            SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1,
            as_type<float>(uint(SEISMIC_PARAM_SOFTCAP))},
        projection::scale_factor(weight_scale, SEISMIC_DIM_WS, 0, 0), 1.0f);
    PROJECTION_OPERANDS_SHARED(globals, NVFP4);
    const auto x = Operands::rows(in, 1u, d, globals, simdgroups, sg, lane);
    projection::gemv_runtime<packets::W0, ROWS, 1, LANES>(
        x, out, w, 1u, SEISMIC_DIM_I == 1 ? 1u : uint(SEISMIC_DIM_S), d, SEISMIC_DIM_I == 1 ? 0u : tile.x, shared, simdgroups, sg, lane);
}
#endif
