// gated_attention_prefill (M >= 16): flash attention on tensor cores over
// dense history (bodies in lib/attention/prefill.cuh).

#include "lib/attention/prefill.cuh"

// Dense history planes [T, KV, W] (activation element).
#define HISTORY()                                                                            \
    attention::DenseHistory {                                                                \
        SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_KEY), SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_VALUE),  \
            SEISMIC_HISTORY_KEY_STRIDE_0, SEISMIC_HISTORY_KEY_STRIDE_1,                      \
            SEISMIC_HISTORY_VALUE_STRIDE_0, SEISMIC_HISTORY_VALUE_STRIDE_1                   \
    }

extern "C" __global__ void __launch_bounds__(256) gated_attention_prefill_prepare(SEISMIC_KERNEL_PARAMS) {
    attention::prefill::prepare(
        ATTENTION_INPUTS(), HISTORY(),
        reinterpret_cast<attention::u16 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_QUERIES)),
        reinterpret_cast<attention::u16 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_KEYS)),
        reinterpret_cast<attention::u16 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_VALUES)));
}

extern "C" __global__ void __launch_bounds__(attention::prefill::WARPS * 32, 1)
    gated_attention_prefill_attend(SEISMIC_KERNEL_PARAMS) {
    attention::prefill::attend(ATTENTION_INPUTS(), HISTORY(),
                               SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_QUERIES),
                               SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_KEYS),
                               SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_VALUES),
                               SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
}
