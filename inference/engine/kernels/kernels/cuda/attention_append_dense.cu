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
        reinterpret_cast<u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_KEY)),
        reinterpret_cast<u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_VALUE)),
        SEISMIC_HISTORY_KEY_STRIDE_0, SEISMIC_HISTORY_KEY_STRIDE_1,
        SEISMIC_HISTORY_VALUE_STRIDE_0, SEISMIC_HISTORY_VALUE_STRIDE_1,
        SEISMIC_PARAM_SLAB_ROWS};
    const attention::u64 item = static_cast<attention::u64>(blockIdx.x) * 8 + threadIdx.x / 32;
    const attention::u64 row = item / attention::KV;
    const int kv = item % attention::KV;
    const int lane = threadIdx.x % 32;
    if (!attention::FRESH || row >= SEISMIC_DIM_M || ATTENTION_DESTINATION(in, row) < 0) return;
    extern __shared__ float shared[];
    float k[attention::DPL];
    attention::prepared_key(in, row, kv, k, shared + (threadIdx.x / 32) * attention::W, lane);
    attention::append(in, history, row, kv, k, lane);
}
