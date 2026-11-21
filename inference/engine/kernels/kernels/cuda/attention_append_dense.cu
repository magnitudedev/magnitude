#define KERNEL_A SEISMIC_ELEMENT_A
#define ATTENTION_QUERY_GROUP 1
#define ATTENTION_INTERLEAVED 0
#define ATTENTION_SEPARATE 0
// State-only geometry: common helpers have no query or gate rows.
// Ordered cache publication, one warp per fresh key/value head.
#define ATTENTION_APPEND_ONLY
#include "lib/attention/attention.cuh"
using attention::u32;
extern "C" __global__ void attention_append_dense(SEISMIC_KERNEL_PARAMS) {
    const attention::Inputs in = ATTENTION_APPEND_INPUTS();
    const attention::DenseHistory history{
        SEISMIC_OUT_HISTORY_KEY,
        SEISMIC_OUT_HISTORY_VALUE,
        SEISMIC_HISTORY_KEY_STRIDE_0, SEISMIC_HISTORY_KEY_STRIDE_1,
        SEISMIC_HISTORY_VALUE_STRIDE_0, SEISMIC_HISTORY_VALUE_STRIDE_1,
        SEISMIC_PARAM_SLAB_ROWS};
    const attention::u64 item = static_cast<attention::u64>(blockIdx.x) * 8 + threadIdx.x / 32;
    if (!attention::FRESH || item >= static_cast<attention::u64>(SEISMIC_DIM_M) * attention::KV) return;
    const u32 item32 = static_cast<u32>(item);
    constexpr u32 KV = static_cast<u32>(attention::KV);
    const u32 row = item32 / KV;
    const int kv = static_cast<int>(item32 % KV);
    const int lane = threadIdx.x % 32;
    if (ATTENTION_DESTINATION(in, row) < 0) return;
    extern __shared__ float shared[];
    float k[attention::DPL];
    // The stored key prepared and rounded as the attention entries do.
    attention::prepared_key(in, row, kv, k, shared + (threadIdx.x / 32) * attention::W, lane,
                            attention::key_rounding(SEISMIC_DIM_KR != 0, SEISMIC_PARAM_KEY_BLOCK,
                                                    SEISMIC_PARAM_KEY_ELEMENT, SEISMIC_PARAM_KEY_SCALE,
                                                    SEISMIC_PARAM_KEY_AMAX_FLOOR));
    attention::append(in, history, row, kv, k, lane);
}
